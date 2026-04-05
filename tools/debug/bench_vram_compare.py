"""
ConvNeXt-V2 + AttnRes (MLM) VRAM 与吞吐基准脚本。

功能:
1. 扫描 模型配置 x AMP 模式 x batch_size。
2. 统计参数量、显存峰值、吞吐与数值稳定性。
3. 自动给出在显存约束下的最优参数组合。

用法示例:
  python tools/debug/bench_vram_compare.py
  python tools/debug/bench_vram_compare.py --batch_sizes 8 12 16 20 --measure_steps 4
"""

import argparse
import gc
import os
import sys
import time
from typing import Any, Dict, List, Tuple

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
import torch.nn as nn

from src.models.latex_ocr_model import LatexOCRModel


# 模型扫描表（默认遵循你当前项目主干）
MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "small_3L": {
        "d_model": 256,
        "decoder_nhead": 8,
        "decoder_num_layers": 3,
        "decoder_dim_feedforward": 1024,
        "decoder_dropout": 0.1,
        "vision_model_name": "convnextv2_pico",
    },
    "base_3L": {
        "d_model": 512,
        "decoder_nhead": 8,
        "decoder_num_layers": 3,
        "decoder_dim_feedforward": 2048,
        "decoder_dropout": 0.1,
        "vision_model_name": "convnextv2_pico",
    },
    "base_4L": {
        "d_model": 512,
        "decoder_nhead": 8,
        "decoder_num_layers": 4,
        "decoder_dim_feedforward": 2048,
        "decoder_dropout": 0.1,
        "vision_model_name": "convnextv2_pico",
    },
}


def param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def make_synthetic_batch(
    batch_size: int,
    image_h: int,
    image_w: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device,
    mask_prob: float,
    pad_id: int = 0,
    bos_id: int = 1,
    eos_id: int = 2,
    mask_id: int = 3,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """构造与 MLM 训练形态一致的合成 batch。"""
    images = torch.randn(batch_size, 1, image_h, image_w, device=device)

    target_ids = torch.randint(4, vocab_size, (batch_size, seq_len), device=device)
    target_ids[:, 0] = bos_id

    min_len = max(8, int(seq_len * 0.5))
    lengths = torch.randint(low=min_len, high=seq_len + 1, size=(batch_size,), device=device)
    pos = torch.arange(seq_len, device=device).unsqueeze(0)

    pad_mask = pos >= lengths.unsqueeze(1)
    target_ids = target_ids.masked_fill(pad_mask, pad_id)

    eos_pos = torch.clamp(lengths - 1, min=1)
    target_ids.scatter_(1, eos_pos.unsqueeze(1), eos_id)

    input_ids = target_ids.clone()
    valid_mlm = (target_ids != pad_id) & (target_ids != bos_id) & (target_ids != eos_id)
    random_vals = torch.rand(batch_size, seq_len, device=device)
    mlm_mask = valid_mlm & (random_vals < mask_prob)
    input_ids = input_ids.masked_fill(mlm_mask, mask_id)

    return images, input_ids, target_ids


def bench_one(
    model_name: str,
    model_cfg: Dict[str, Any],
    amp_mode: str,
    batch_size: int,
    image_h: int,
    image_w: int,
    seq_len: int,
    vocab_size: int,
    mask_prob: float,
    label_smoothing: float,
    warmup_steps: int,
    measure_steps: int,
    encoder_pretrained: bool,
) -> Dict[str, Any]:
    assert amp_mode in {"fp32", "fp16", "bf16"}, f"未知 amp_mode: {amp_mode}"
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，无法运行 GPU 基准")

    device = torch.device("cuda")

    model = LatexOCRModel(
        vocab_size=vocab_size,
        d_model=int(model_cfg["d_model"]),
        pad_id=0,
        vision_model_name=str(model_cfg["vision_model_name"]),
        vision_pretrained=encoder_pretrained,
        vision_in_chans=1,
        decoder_nhead=int(model_cfg["decoder_nhead"]),
        decoder_num_layers=int(model_cfg["decoder_num_layers"]),
        decoder_dim_feedforward=int(model_cfg["decoder_dim_feedforward"]),
        decoder_dropout=float(model_cfg["decoder_dropout"]),
    ).to(device)
    model.train()

    params = param_count(model)
    criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)

    use_scaler = amp_mode == "fp16"
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    amp_enabled = amp_mode != "fp32"
    amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(amp_mode, torch.float32)

    def run_step(batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> float:
        images, input_ids, target_ids = batch
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
            logits = model(images=images, tgt_seq=input_ids, is_causal=False)
            loss = criterion(logits.reshape(-1, logits.shape[-1]), target_ids.reshape(-1))

        if not torch.isfinite(loss):
            return float("nan")

        if use_scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        return float(loss.detach().item())

    # 预热
    for _ in range(warmup_steps):
        run_step(
            make_synthetic_batch(
                batch_size=batch_size,
                image_h=image_h,
                image_w=image_w,
                seq_len=seq_len,
                vocab_size=vocab_size,
                device=device,
                mask_prob=mask_prob,
            )
        )
    torch.cuda.synchronize(device)

    # 正式测量
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    nan_count = 0
    total_time = 0.0
    total_samples = 0
    losses: List[float] = []

    for _ in range(measure_steps):
        batch = make_synthetic_batch(
            batch_size=batch_size,
            image_h=image_h,
            image_w=image_w,
            seq_len=seq_len,
            vocab_size=vocab_size,
            device=device,
            mask_prob=mask_prob,
        )
        start = time.perf_counter()
        loss_val = run_step(batch)
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start

        total_time += elapsed
        total_samples += batch_size
        losses.append(loss_val)

        if not (loss_val == loss_val) or loss_val == float("inf"):
            nan_count += 1

    peak_alloc = torch.cuda.max_memory_allocated(device) / 1024**3
    peak_reserved = torch.cuda.max_memory_reserved(device) / 1024**3
    throughput = total_samples / max(total_time, 1e-8)
    avg_loss = sum(losses) / max(len(losses), 1)

    del model, optimizer, scaler
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "model": model_name,
        "amp_mode": amp_mode,
        "batch_size": batch_size,
        "params_M": round(params / 1e6, 2),
        "peak_alloc_GB": round(peak_alloc, 3),
        "peak_reserved_GB": round(peak_reserved, 3),
        "throughput_sps": round(throughput, 1),
        "avg_loss": round(avg_loss, 4),
        "nan_steps": nan_count,
        "loss_nan": nan_count > 0,
        "oom": False,
    }


def print_results(results: List[Dict[str, Any]], total_vram_gb: float) -> None:
    print("\n" + "=" * 120)
    print(
        "  ConvNeXt-V2 + AttnRes 基准结果  "
        f"(设备: {torch.cuda.get_device_name(0)}, 总显存: {total_vram_gb:.1f} GB)"
    )
    print("=" * 120)
    print(
        f"{'model':<12} {'amp':<6} {'bs':<4} {'params(M)':<10} "
        f"{'alloc(GB)':<10} {'rsv(GB)':<9} {'throughput':<12} {'avg_loss':<10} {'status':<10} {'vram%':<8}"
    )
    print("-" * 120)

    for r in sorted(results, key=lambda x: (x["model"], x["amp_mode"], x["batch_size"])):
        vram_pct = (r["peak_alloc_GB"] / max(total_vram_gb, 1e-8)) * 100.0 if not r["oom"] else 0.0
        if r["oom"]:
            status = "OOM"
        elif r["loss_nan"]:
            status = "NaN"
        else:
            status = "OK"

        print(
            f"{r['model']:<12} {r['amp_mode']:<6} {r['batch_size']:<4d} {r.get('params_M', 0):<10} "
            f"{r.get('peak_alloc_GB', 0.0):<10.3f} {r.get('peak_reserved_GB', 0.0):<9.3f} "
            f"{r.get('throughput_sps', 0.0):<12.1f} {r.get('avg_loss', 0.0):<10.4f} {status:<10} {vram_pct:<8.1f}"
        )

    print("=" * 120)


def pick_best(
    results: List[Dict[str, Any]],
    total_vram_gb: float,
    max_vram_util: float,
) -> Tuple[Dict[str, Any], bool]:
    """返回最优配置，以及是否触发了显存约束放宽回退。"""
    stable = [r for r in results if (not r["oom"]) and (not r["loss_nan"])]
    if not stable:
        raise RuntimeError("所有组合都 OOM 或 NaN，无法给出推荐")

    strict = [
        r
        for r in stable
        if (r["peak_alloc_GB"] / max(total_vram_gb, 1e-8)) <= max_vram_util
    ]

    use_relaxed = False
    pool = strict
    if not pool:
        pool = stable
        use_relaxed = True

    best = max(pool, key=lambda x: (x["throughput_sps"], -x["peak_alloc_GB"]))
    return best, use_relaxed


def main() -> None:
    parser = argparse.ArgumentParser(description="ConvNeXt-V2 + AttnRes VRAM 与吞吐基准")
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[8, 12, 16, 20, 24], help="扫描的 batch size 列表")
    parser.add_argument("--image_h", type=int, default=192, help="合成图像高度")
    parser.add_argument("--image_w", type=int, default=768, help="合成图像宽度")
    parser.add_argument("--seq_len", type=int, default=160, help="目标序列长度")
    parser.add_argument("--vocab_size", type=int, default=8000, help="词表大小")
    parser.add_argument("--mask_prob", type=float, default=0.15, help="MLM 掩码比例")
    parser.add_argument("--label_smoothing", type=float, default=0.1, help="交叉熵 label smoothing")
    parser.add_argument("--warmup_steps", type=int, default=1, help="预热步数")
    parser.add_argument("--measure_steps", type=int, default=3, help="测量步数")
    parser.add_argument("--max_vram_util", type=float, default=0.92, help="推荐时显存利用率上限")
    parser.add_argument("--models", nargs="+", default=list(MODEL_CONFIGS.keys()), choices=list(MODEL_CONFIGS.keys()), help="参与扫描的模型配置")
    parser.add_argument("--amp_modes", nargs="+", default=["fp32", "fp16", "bf16"], choices=["fp32", "fp16", "bf16"], help="参与扫描的 AMP 模式")
    parser.add_argument("--encoder_pretrained", action="store_true", help="基准时使用预训练权重（更慢，通常建议关闭）")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，该脚本需要 GPU")

    total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU: {torch.cuda.get_device_name(0)} ({total_vram:.2f} GB)")
    print(
        f"输入: image={args.image_h}x{args.image_w}, seq_len={args.seq_len}, vocab={args.vocab_size}, "
        f"batch_sizes={args.batch_sizes}"
    )
    print(f"模型: {args.models}")
    print(f"精度: {args.amp_modes}\n")

    all_results: List[Dict[str, Any]] = []

    for model_name in args.models:
        cfg = MODEL_CONFIGS[model_name]
        for amp_mode in args.amp_modes:
            for batch_size in args.batch_sizes:
                trial_name = f"{model_name}_{amp_mode}_bs{batch_size}"
                print(f"运行 [{trial_name}] ...", end=" ", flush=True)
                try:
                    result = bench_one(
                        model_name=model_name,
                        model_cfg=cfg,
                        amp_mode=amp_mode,
                        batch_size=batch_size,
                        image_h=args.image_h,
                        image_w=args.image_w,
                        seq_len=args.seq_len,
                        vocab_size=args.vocab_size,
                        mask_prob=args.mask_prob,
                        label_smoothing=args.label_smoothing,
                        warmup_steps=args.warmup_steps,
                        measure_steps=args.measure_steps,
                        encoder_pretrained=args.encoder_pretrained,
                    )
                    all_results.append(result)
                    status = "NaN" if result["loss_nan"] else "OK"
                    print(
                        f"alloc={result['peak_alloc_GB']:.2f}GB "
                        f"sps={result['throughput_sps']:.1f} "
                        f"loss={result['avg_loss']:.4f} [{status}]"
                    )
                except torch.cuda.OutOfMemoryError:
                    print("OOM")
                    all_results.append(
                        {
                            "model": model_name,
                            "amp_mode": amp_mode,
                            "batch_size": batch_size,
                            "params_M": 0.0,
                            "peak_alloc_GB": 0.0,
                            "peak_reserved_GB": 0.0,
                            "throughput_sps": 0.0,
                            "avg_loss": 0.0,
                            "nan_steps": 0,
                            "loss_nan": False,
                            "oom": True,
                        }
                    )
                    gc.collect()
                    torch.cuda.empty_cache()

    print_results(all_results, total_vram)

    best, relaxed = pick_best(all_results, total_vram, args.max_vram_util)
    best_vram_pct = best["peak_alloc_GB"] / max(total_vram, 1e-8) * 100.0

    print("最优推荐:")
    if relaxed:
        print("- 注意: 未找到满足显存阈值的稳定组合，已放宽到稳定组合中吞吐最高项")
    print(f"- model      : {best['model']}")
    print(f"- amp_mode   : {best['amp_mode']}")
    print(f"- batch_size : {best['batch_size']}")
    print(f"- throughput : {best['throughput_sps']:.1f} samples/s")
    print(f"- peak_alloc : {best['peak_alloc_GB']:.3f} GB ({best_vram_pct:.1f}%)")


if __name__ == "__main__":
    main()
