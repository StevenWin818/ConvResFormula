"""基于训练集长度分布，自动搜索 Beam Search 动态 Alpha 最优超参数。"""
import argparse
import json
import os
import sys
import time
from datetime import datetime
from functools import partial
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
import yaml
from tokenizers import Tokenizer
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import FormulaDataset
from scripts.eval import (
    NodeKatexSvgRenderer,
    batched_infer_ar,
    build_model,
    check_visual_equivalence,
    collate_eval_batch,
    resolve_eval_runtime_args,
)


# ---------------------------------------------------------------------------
# 阶段 1：训练集长度分布分析
# ---------------------------------------------------------------------------

def analyze_length_distribution(
    h5_paths: List[str],
    max_samples_per_file: int = 0,
) -> Dict[str, object]:
    """扫描训练 H5 文件，统计 token 序列长度分布。"""
    all_lengths: List[int] = []

    for path in h5_paths:
        abs_path = str((PROJECT_ROOT / path).resolve()) if not os.path.isabs(path) else path
        if not os.path.exists(abs_path):
            print(f"⚠️  跳过不存在的文件: {abs_path}")
            continue

        with h5py.File(abs_path, "r") as f:
            labels = f["labels"]
            n = int(labels.shape[0])
            limit = n if max_samples_per_file <= 0 else min(max_samples_per_file, n)
            print(f"  扫描 {abs_path}  ({limit}/{n} 样本) ...", end=" ", flush=True)

            for i in range(limit):
                sample = labels[i]
                length = int(len(sample)) if hasattr(sample, "__len__") else 1
                all_lengths.append(length)
            print("完成")

    if not all_lengths:
        raise RuntimeError("未读取到任何样本长度数据，请检查训练集路径")

    arr = np.array(all_lengths, dtype=np.float64)
    percentiles = {f"P{p}": float(np.percentile(arr, p)) for p in [5, 10, 25, 50, 75, 90, 95]}

    stats = {
        "total_samples": len(arr),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        **percentiles,
    }
    return stats


def derive_search_ranges(stats: Dict[str, object]) -> Dict[str, List[float]]:
    """根据长度分布统计量，推导各超参数的搜索范围。"""
    p10, p25 = float(stats["P10"]), float(stats["P25"])
    p75, p90 = float(stats["P75"]), float(stats["P90"])

    # short_thresh 在 [P10, P25] 之间取 3 个点
    short_candidates = sorted(set([
        round(p10, 1),
        round((p10 + p25) / 2, 1),
        round(p25, 1),
    ]))
    # long_thresh 在 [P75, P90] 之间取 3 个点
    long_candidates = sorted(set([
        round(p75, 1),
        round((p75 + p90) / 2, 1),
        round(p90, 1),
    ]))

    return {
        "alpha_min": [0.3, 0.5, 0.6, 0.8],
        "alpha_max": [1.0, 1.4, 1.8, 2.0],
        "short_thresh": short_candidates,
        "long_thresh": long_candidates,
    }


# ---------------------------------------------------------------------------
# 阶段 2：网格搜索评估
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_single_config(
    model,
    eval_loader: DataLoader,
    tokenizer: Tokenizer,
    svg_renderer: NodeKatexSvgRenderer,
    pad_id: int,
    bos_id: int,
    eos_id: int,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    max_len: int,
    beam_size: int,
    alpha_cfg: Dict[str, float],
    max_samples: int,
) -> Dict[str, float]:
    """用指定的 alpha 配置跑一次评估，返回指标。"""
    processed = 0
    exact_match = 0
    total_ned = 0.0

    for batch in eval_loader:
        if processed >= max_samples:
            break

        images = batch["images"].to(device)
        target_ids = batch["target_ids"]

        pred_batch = batched_infer_ar(
            model=model,
            images=images,
            pad_id=pad_id,
            bos_id=bos_id,
            eos_id=eos_id,
            max_len=max_len,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            beam_size=beam_size,
            tokenizer=tokenizer,
            alpha_min=alpha_cfg["alpha_min"],
            alpha_max=alpha_cfg["alpha_max"],
            alpha_short_thresh=alpha_cfg["short_thresh"],
            alpha_long_thresh=alpha_cfg["long_thresh"],
        )

        for i in range(len(pred_batch)):
            if processed >= max_samples:
                break
            pred_text = tokenizer.decode(pred_batch[i], skip_special_tokens=True).strip()
            target_text = tokenizer.decode(target_ids[i].tolist(), skip_special_tokens=True).strip()

            is_correct = check_visual_equivalence(target_text, pred_text, svg_renderer)
            exact_match += int(is_correct)

            # 简化版 NED
            from scripts.eval import levenshtein_distance
            dist = levenshtein_distance(pred_text, target_text)
            total_ned += dist / max(1, len(target_text))
            processed += 1

    em_rate = exact_match / max(1, processed)
    avg_ned = total_ned / max(1, processed)
    return {"render_em": em_rate, "avg_ned": avg_ned, "processed": processed}


def run_grid_search(
    model,
    eval_loader: DataLoader,
    tokenizer: Tokenizer,
    svg_renderer: NodeKatexSvgRenderer,
    pad_id: int,
    bos_id: int,
    eos_id: int,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    max_len: int,
    beam_size: int,
    search_ranges: Dict[str, List[float]],
    max_samples: int,
) -> List[Dict]:
    """遍历参数组合并评估。"""
    keys = ["alpha_min", "alpha_max", "short_thresh", "long_thresh"]
    all_combos = list(product(*(search_ranges[k] for k in keys)))
    # 过滤无效组合：alpha_min < alpha_max, short_thresh < long_thresh
    valid_combos = [
        c for c in all_combos
        if c[0] < c[1] and c[2] < c[3]
    ]

    print(f"\n共 {len(valid_combos)} 个有效参数组合（总 {len(all_combos)} 个）")
    results: List[Dict] = []

    for idx, combo in enumerate(valid_combos):
        cfg = dict(zip(keys, combo))
        label = (
            f"[{idx+1}/{len(valid_combos)}] "
            f"α=[{cfg['alpha_min']:.1f},{cfg['alpha_max']:.1f}] "
            f"thresh=[{cfg['short_thresh']:.1f},{cfg['long_thresh']:.1f}]"
        )
        print(f"{label}", end=" ... ", flush=True)

        t0 = time.time()
        metrics = evaluate_single_config(
            model=model,
            eval_loader=eval_loader,
            tokenizer=tokenizer,
            svg_renderer=svg_renderer,
            pad_id=pad_id,
            bos_id=bos_id,
            eos_id=eos_id,
            device=device,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            max_len=max_len,
            beam_size=beam_size,
            alpha_cfg=cfg,
            max_samples=max_samples,
        )
        elapsed = time.time() - t0

        entry = {**cfg, **metrics, "elapsed_sec": round(elapsed, 1)}
        results.append(entry)
        print(f"RenderEM={metrics['render_em']*100:.2f}%  NED={metrics['avg_ned']:.4f}  ({elapsed:.1f}s)")

    # 按 RenderEM 降序、NED 升序排序
    results.sort(key=lambda r: (-r["render_em"], r["avg_ned"]))
    return results


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="自动搜索 Beam Search 动态 Alpha 最优参数")
    p.add_argument("--train_config", type=str, default=str(PROJECT_ROOT / "configs" / "train_ar.yaml"))
    p.add_argument("--model_config", type=str, default=str(PROJECT_ROOT / "configs" / "model_convnext_attnres.yaml"))
    p.add_argument("--eval_h5", type=str, default=str(PROJECT_ROOT / "datasets" / "val.h5"))
    p.add_argument("--tokenizer", type=str, default=str(PROJECT_ROOT / "tokenizer_bpe.json"))
    p.add_argument("--checkpoint", type=str, default=str(PROJECT_ROOT / "checkpoints" / "ar" / "epoch_65.pth"))
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--max_area", type=int, default=98304)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--beam_size", type=int, default=3, help="Beam Search 大小")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_len", type=int, default=160)
    p.add_argument("--eval_samples", type=int, default=2048, help="搜索评估使用的验证样本数")
    p.add_argument("--scan_samples", type=int, default=0, help="训练集扫描样本数上限 (<=0 全量)")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--amp_dtype", type=str, choices=["fp16", "bf16", "fp32"], default="bf16")
    p.add_argument("--use_gradient_checkpointing", action="store_true", default=True)
    p.add_argument("--checkpoint_decoder_layers", action="store_true", default=False)
    p.add_argument("--enable_compile", action="store_true", default=False)
    p.add_argument("--output", type=str, default=str(PROJECT_ROOT / "configs" / "alpha_config.json"),
                   help="最优参数输出路径")
    p.add_argument(
        "--svg_node_script", type=str,
        default=str(PROJECT_ROOT / "tools" / "debug" / "katex_svg_renderer.js"),
    )
    return p.parse_args()


def main():
    args = parse_args()
    resolve_eval_runtime_args(args)
    device = torch.device(args.device)
    tokenizer = Tokenizer.from_file(args.tokenizer)

    # ── 阶段 1：分析训练集长度分布 ──
    print("=" * 70)
    print("阶段 1：扫描训练集 token 长度分布")
    print("=" * 70)

    train_cfg_path = args.train_config
    train_cfg: Dict = {}
    if os.path.exists(train_cfg_path):
        with open(train_cfg_path, "r", encoding="utf-8") as f:
            train_cfg = yaml.safe_load(f) or {}

    train_h5_list = train_cfg.get("data", {}).get("train_h5", [])
    if isinstance(train_h5_list, str):
        train_h5_list = [train_h5_list]
    if not train_h5_list:
        print("⚠️  train_config 中未找到 train_h5，使用 eval_h5 替代分析")
        train_h5_list = [args.eval_h5]

    stats = analyze_length_distribution(train_h5_list, max_samples_per_file=args.scan_samples)

    print(f"\n📊 长度分布统计:")
    for k, v in stats.items():
        print(f"  {k:>15s}: {v}")

    search_ranges = derive_search_ranges(stats)
    print(f"\n🔍 推导的搜索范围:")
    for k, v in search_ranges.items():
        print(f"  {k:>15s}: {v}")

    # ── 阶段 2：网格搜索 ──
    print("\n" + "=" * 70)
    print("阶段 2：网格搜索最优 Alpha 配置")
    print("=" * 70)

    svg_renderer = NodeKatexSvgRenderer(node_script=args.svg_node_script)
    if not svg_renderer._start_process():
        raise RuntimeError("无法启动 SVG 渲染子进程")

    try:
        pad_id = tokenizer.token_to_id("[PAD]")
        bos_id = tokenizer.token_to_id("[BOS]")
        eos_id = tokenizer.token_to_id("[EOS]")
        if None in (pad_id, bos_id, eos_id):
            raise RuntimeError("Tokenizer 缺少 [PAD]/[BOS]/[EOS]")

        dataset = FormulaDataset(h5_path=args.eval_h5, tokenizer_path=args.tokenizer, max_area=args.max_area)
        n = min(args.eval_samples, len(dataset)) if args.eval_samples > 0 else len(dataset)
        subset = Subset(dataset, list(range(n)))

        eval_loader = DataLoader(
            subset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=partial(collate_eval_batch, pad_token_id=pad_id),
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )

        model = build_model(
            args.checkpoint, tokenizer, args.d_model, device,
            use_gradient_checkpointing=bool(args.use_gradient_checkpointing),
            checkpoint_decoder_layers=bool(args.checkpoint_decoder_layers),
            enable_compile=bool(args.enable_compile),
        )

        amp_dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
        eval_amp_dtype = amp_dtype_map[args.amp_dtype]
        amp_enabled = device.type == "cuda" and args.amp_dtype != "fp32"

        results = run_grid_search(
            model=model,
            eval_loader=eval_loader,
            tokenizer=tokenizer,
            svg_renderer=svg_renderer,
            pad_id=pad_id, bos_id=bos_id, eos_id=eos_id,
            device=device,
            amp_enabled=amp_enabled,
            amp_dtype=eval_amp_dtype,
            max_len=args.max_len,
            beam_size=args.beam_size,
            search_ranges=search_ranges,
            max_samples=n,
        )
    finally:
        svg_renderer.close()

    # ── 阶段 3：输出结果 ──
    print("\n" + "=" * 70)
    print("阶段 3：搜索结果排名 (按 RenderEM ↓, NED ↑)")
    print("=" * 70)

    print(f"\n{'Rank':>4s}  {'α_min':>5s}  {'α_max':>5s}  {'short':>6s}  {'long':>6s}  {'RenderEM':>9s}  {'NED':>8s}  {'Time':>5s}")
    print("-" * 65)
    for rank, r in enumerate(results[:20], 1):
        print(
            f"{rank:>4d}  {r['alpha_min']:>5.1f}  {r['alpha_max']:>5.1f}  "
            f"{r['short_thresh']:>6.1f}  {r['long_thresh']:>6.1f}  "
            f"{r['render_em']*100:>8.2f}%  {r['avg_ned']:>8.4f}  {r['elapsed_sec']:>5.1f}s"
        )

    # 保存最优配置
    best = results[0]
    best_config = {
        "alpha_min": best["alpha_min"],
        "alpha_max": best["alpha_max"],
        "short_thresh": best["short_thresh"],
        "long_thresh": best["long_thresh"],
        "search_metric": {
            "render_em": best["render_em"],
            "avg_ned": best["avg_ned"],
        },
        "search_meta": {
            "beam_size": args.beam_size,
            "eval_samples": n,
            "total_combos_tested": len(results),
            "timestamp": datetime.now().isoformat(),
            "length_distribution": stats,
        },
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(best_config, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 最优配置已保存到: {args.output}")
    print(f"   α_min={best['alpha_min']:.1f}  α_max={best['alpha_max']:.1f}  "
          f"short_thresh={best['short_thresh']:.1f}  long_thresh={best['long_thresh']:.1f}")
    print(f"   RenderEM={best['render_em']*100:.2f}%  NED={best['avg_ned']:.4f}")


if __name__ == "__main__":
    main()
