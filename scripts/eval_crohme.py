"""CROHME2019 PNG 测试集评估脚本（推理 + 可选官方 LgEval 打分）。"""

import argparse
import csv
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from tokenizers import Tokenizer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.latex_ocr_model import LatexOCRModel
from src.models.text_decoder import convert_legacy_attnres_state_dict

def robust_crop(img: np.ndarray, noise_threshold: int = 5, pad: int = 16) -> np.ndarray:
    """
    暴力抗噪裁剪：通过计算行列的像素强度总和来寻找真正的边界。
    忽略零星的噪点像素，彻底粉碎 1010x1010 的无效边框。
    """
    # 假设此时 img 已经是黑底白字 (0 是背景，255 是笔划)
    row_sums = np.sum(img, axis=1)
    col_sums = np.sum(img, axis=0)
    
    # 设定阈值：一行/一列中至少要有等效于 noise_threshold 个纯白像素，才认为是有效笔划
    # 彻底过滤掉图片边缘可能存在的 1px 细线边框或零星噪点
    valid_rows = np.where(row_sums > noise_threshold * 255)[0]
    valid_cols = np.where(col_sums > noise_threshold * 255)[0]
    
    if len(valid_rows) > 0 and len(valid_cols) > 0:
        y_min, y_max = valid_rows[0], valid_rows[-1]
        x_min, x_max = valid_cols[0], valid_cols[-1]
        
        # 裁剪并保留安全边距
        y_min = max(0, y_min - pad)
        y_max = min(img.shape[0], y_max + pad)
        x_min = max(0, x_min - pad)
        x_max = min(img.shape[1], x_max + pad)
        
        return img[y_min:y_max, x_min:x_max]
    return img

def calculate_dynamic_dims(
    h_orig: int,
    w_orig: int,
    max_area: int = 98304,
    min_size: int = 32,
    stride: int = 32,
) -> Tuple[int, int]:
    """按面积约束和步长对齐计算目标分辨率。"""
    aspect_ratio = float(w_orig) / max(float(h_orig), 1.0)
    target_h = (max_area / max(aspect_ratio, 1e-8)) ** 0.5
    target_w = target_h * aspect_ratio

    def align(v: float) -> int:
        return max(min_size, int(round(v / stride) * stride))

    return align(target_h), align(target_w)


@dataclass
class ImageSample:
    sample_id: str
    image: torch.Tensor


class CrohmePngDataset(Dataset):
    """仅用于离线 PNG 公式图像的推理数据集。"""

    def __init__(self, image_root: str, max_area: int = 98304):
        self.image_root = Path(image_root)
        if not self.image_root.exists():
            raise FileNotFoundError(f"找不到图像目录: {self.image_root}")

        self.paths: List[Path] = sorted(self.image_root.rglob("*.png"))
        if not self.paths:
            raise RuntimeError(f"目录中未找到 PNG 文件: {self.image_root}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> ImageSample:
        path = self.paths[idx]
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"读取图像失败: {path}")

        # 1. 自动反相：确保图像永远是“黑底白字”
        # 官方 CROHME PNG 通常是白底，均值极大。
        if np.mean(img) > 127:
            img = 255 - img

        img = robust_crop(img, noise_threshold=5, pad=16)

        coords = cv2.findNonZero(img)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            pad = 16  # 留一点安全边距
            y_min = max(0, y - pad)
            y_max = min(img.shape[0], y + h + pad)
            x_min = max(0, x - pad)
            x_max = min(img.shape[1], x + w + pad)
            img = img[y_min:y_max, x_min:x_max]

        img = cv2.GaussianBlur(img, (3, 3), 0)
        
        # 步骤B：对模糊后的图像进行距离变换平滑加粗 (比直接 dilate 柔和得多)
        img = cv2.distanceTransform(img, cv2.DIST_L2, 3)
        dst = np.zeros_like(img, dtype=np.uint8)
        img = cv2.normalize(img, dst, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        # 步骤C：适当的伽马校正（让主体变亮，边缘保持柔和）
        img = np.power(img / 255.0, 0.8) * 255.0
        img = img.astype(np.uint8)

        # 2. 严格对齐 32 步长：绝不 Resize！只在右侧和下方补纯黑像素 (0)
        h, w = img.shape
        pad_h = (32 - h % 32) % 32
        pad_w = (32 - w % 32) % 32
        
        if pad_h > 0 or pad_w > 0:
            img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)

        # 3. 转换为 Tensor。

        image_tensor = torch.from_numpy(img).float().unsqueeze(0) / 255.0
        
        return ImageSample(sample_id=path.stem, image=image_tensor)


def collate_images(batch: List[ImageSample]) -> Dict[str, torch.Tensor | List[str]]:
    sample_ids = [x.sample_id for x in batch]
    images = [x.image for x in batch]

    max_h = max(int(img.shape[1]) for img in images)
    max_w = max(int(img.shape[2]) for img in images)

    out = torch.zeros((len(images), 1, max_h, max_w), dtype=torch.float32)
    for i, img in enumerate(images):
        _, h, w = img.shape
        out[i, :, :h, :w] = img

    return {"sample_ids": sample_ids, "images": out}


@torch.no_grad()
def batched_infer_ar(
    model: LatexOCRModel,
    images: torch.Tensor,
    pad_id: int,
    bos_id: int,
    eos_id: int,
    max_len: int,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    beam_size: int = 1,
) -> List[List[int]]:
    batch_size = images.size(0)
    device = images.device

    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
        memory, memory_padding_mask = model.encode(images)

    if beam_size <= 1:
        decode_cache = model.init_decode_cache(memory)
        generated = torch.full((batch_size, max_len), pad_id, dtype=torch.long, device=device)
        generated[:, 0] = bos_id
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    
        for step in range(1, max_len):
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                current_token = generated[:, step - 1]
                logits, decode_cache, _attn_weights = model.decode_step_cached(
                    memory=memory,
                    token_id=current_token,
                    cache=decode_cache,
                    memory_padding_mask=memory_padding_mask,
                    return_attn_weights=True
                )
    
            next_tokens = logits.argmax(dim=-1)
            next_tokens = next_tokens.masked_fill(finished, pad_id)
            generated[:, step] = next_tokens
            finished |= next_tokens == eos_id
            if finished.all():
                break
    
        outputs: List[List[int]] = []
        for row in generated.cpu().tolist():
            if eos_id in row:
                row = row[: row.index(eos_id)]
            outputs.append(row)
        return outputs

    vocab_size = model.vocab_size
    memory = memory.repeat_interleave(beam_size, dim=0)
    if memory_padding_mask is not None:
        memory_padding_mask = memory_padding_mask.repeat_interleave(beam_size, dim=0)
    
    decode_cache = model.init_decode_cache(memory)

    generated = torch.full((batch_size * beam_size, max_len), pad_id, dtype=torch.long, device=device)
    generated[:, 0] = bos_id
    
    beam_scores = torch.full((batch_size, beam_size), -1e9, dtype=torch.float, device=device)
    beam_scores[:, 0] = 0.0
    beam_scores = beam_scores.view(-1)
    
    finished = torch.zeros(batch_size * beam_size, dtype=torch.bool, device=device)

    for step in range(1, max_len):
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            current_token = generated[:, step - 1]
            logits, decode_cache, _attn_weights = model.decode_step_cached(
                memory=memory,
                token_id=current_token,
                cache=decode_cache,
                memory_padding_mask=memory_padding_mask,
                return_attn_weights=True
            )

        log_probs = torch.log_softmax(logits, dim=-1)

        log_probs[finished, :] = -1e9
        log_probs[finished, pad_id] = 0.0

        next_scores = beam_scores.unsqueeze(1) + log_probs
        next_scores = next_scores.view(batch_size, beam_size * vocab_size)        

        topk_scores, topk_indices = torch.topk(next_scores, beam_size, dim=1)
        
        beam_indices = topk_indices // vocab_size
        next_tokens = topk_indices % vocab_size
        
        beam_scores = topk_scores.view(-1)
        
        batch_offset = torch.arange(batch_size, device=device).unsqueeze(1) * beam_size
        absolute_beam_indices = (beam_indices + batch_offset).view(-1)
        
        generated = generated[absolute_beam_indices]
        generated[:, step] = next_tokens.view(-1)
        
        finished = finished[absolute_beam_indices]
        finished |= (next_tokens.view(-1) == eos_id)
        
        if decode_cache.self_key_values is not None:
            new_self_kv = []
            for layer_kv in decode_cache.self_key_values:
                if layer_kv is None:
                    new_self_kv.append(None)
                else:
                    k, v = layer_kv
                    new_self_kv.append((k[absolute_beam_indices], v[absolute_beam_indices]))
            decode_cache.self_key_values = new_self_kv

        if finished.all():
            break

    generated = generated.view(batch_size, beam_size, max_len)
    beam_scores = beam_scores.view(batch_size, beam_size)

    lengths = (generated != pad_id).sum(dim=2).float()
    alpha = 1.4
    penalized_scores = beam_scores / (lengths ** alpha)

    best_beam_indices = penalized_scores.argmax(dim=1)
    best_sequences = generated[torch.arange(batch_size), best_beam_indices]

    outputs = []
    for row in best_sequences.cpu().tolist():
        if eos_id in row:
            row = row[: row.index(eos_id)]
        outputs.append(row)

    return outputs


def build_model(
    checkpoint_path: str,
    tokenizer: Tokenizer,
    d_model: int,
    device: torch.device,
) -> LatexOCRModel:
    vocab_size = tokenizer.get_vocab_size()
    pad_id = tokenizer.token_to_id("[PAD]")
    if pad_id is None:
        raise RuntimeError("Tokenizer 缺少 [PAD] token")

    model = LatexOCRModel(
        vocab_size=vocab_size,
        d_model=d_model,
        pad_id=pad_id,
        vision_pretrained=False, # 推理时不需要预先下载 ImageNet 权重
        use_gradient_checkpointing=False,
        checkpoint_decoder_layers=False,
    ).to(device)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"找不到 checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model", checkpoint)
    if isinstance(state_dict, dict):
        orig_prefix = "_orig_mod."
        if any(str(k).startswith(orig_prefix) for k in state_dict.keys()):
            state_dict = {
                (str(k)[len(orig_prefix):] if str(k).startswith(orig_prefix) else str(k)): v
                for k, v in state_dict.items()
            }

    state_dict = convert_legacy_attnres_state_dict(state_dict)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def run_inference(args: argparse.Namespace, out_dir: Path) -> Path:
    tokenizer = Tokenizer.from_file(args.tokenizer)
    device = torch.device(args.device)

    pad_id = tokenizer.token_to_id("[PAD]")
    bos_id = tokenizer.token_to_id("[BOS]")
    eos_id = tokenizer.token_to_id("[EOS]")
    if pad_id is None or bos_id is None or eos_id is None:
        raise RuntimeError("Tokenizer 缺少 [PAD]/[BOS]/[EOS]，无法进行 AR 推理")

    dataset = CrohmePngDataset(args.image_root, max_area=args.max_area)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_images,
    )

    model = build_model(
        checkpoint_path=args.checkpoint,
        tokenizer=tokenizer,
        d_model=args.d_model,
        device=device,
    )

    amp_dtype_map = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    amp_dtype = amp_dtype_map[args.amp_dtype]
    amp_enabled = device.type == "cuda" and args.amp_dtype != "fp32"

    pred_txt_dir = out_dir / "pred_tex"
    pred_txt_dir.mkdir(parents=True, exist_ok=True)
    pred_csv_path = out_dir / "predictions.csv"

    with open(pred_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "latex"])

        for batch in tqdm(loader, desc="CROHME 推理"):
            sample_ids = batch["sample_ids"]
            images = batch["images"].to(device)

            pred_ids = batched_infer_ar(
                model=model,
                images=images,
                pad_id=pad_id,
                bos_id=bos_id,
                eos_id=eos_id,
                max_len=args.max_len,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
                beam_size=args.beam_size,
            )

            for sid, ids in zip(sample_ids, pred_ids):
                latex = tokenizer.decode(ids, skip_special_tokens=True).strip()
                writer.writerow([sid, latex])
                wrapped_latex = f"$ {latex} $" if latex else "$ $"
                (pred_txt_dir / f"{sid}.txt").write_text(wrapped_latex + "\n", encoding="utf-8")

    return pred_txt_dir


def _run_subprocess(cmd: List[str], cwd: Optional[Path] = None, env: Optional[Dict[str, str]] = None) -> None:
    result = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "命令执行失败:\n"
            + " ".join(cmd)
            + "\nSTDOUT:\n"
            + (result.stdout or "")
            + "\nSTDERR:\n"
            + (result.stderr or "")
        )


def _first_existing_path(candidates: List[Path]) -> Optional[Path]:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _resolve_tool_paths() -> Tuple[Optional[Path], Optional[Path], Optional[Path]]:
    """查找 bash、pandoc、perl 的可执行文件路径。"""
    bash_path = _first_existing_path([
        Path(r"C:\Program Files\Git\bin\bash.exe"),
        Path(r"C:\Program Files\Git\usr\bin\bash.exe"),
        Path(r"C:\Program Files\Git\bin\sh.exe"),
    ])
    pandoc_path = _first_existing_path([
        Path(r"C:\ProgramData\chocolatey\bin\pandoc.exe"),
        Path(r"C:\Program Files\Pandoc\pandoc.exe"),
    ])
    perl_path = _first_existing_path([
        Path(r"C:\Strawberry\perl\bin\perl.exe"),
        Path(r"C:\Strawberry\perl\bin\wperl.exe"),
    ])
    return bash_path, pandoc_path, perl_path


def _prepare_python_shims(shim_dir: Path) -> Path:
    """创建 bash 可见的 python3/python 转发定义。"""
    shim_dir.mkdir(parents=True, exist_ok=True)
    python_exe = Path(sys.executable).resolve()
    shim_text = (
        "#!/bin/sh\n"
        f'PYTHON_EXE="{_to_bash_path(python_exe)}"\n'
        'python3() { "$PYTHON_EXE" "$@"; }\n'
        'python() { "$PYTHON_EXE" "$@"; }\n'
    )
    (shim_dir / "bash_env.sh").write_text(shim_text, encoding="utf-8")
    return shim_dir


def _to_bash_path(path: Path) -> str:
    """将 Windows 路径转换为 Git Bash 可识别的 POSIX 路径。"""
    resolved = path.resolve()
    text = str(resolved).replace("\\", "/")
    if len(text) >= 2 and text[1] == ":":
        drive = text[0].lower()
        remainder = text[2:].lstrip("/")
        return f"/{drive}/{remainder}"
    return text


def try_convert_tex_to_symlg(tex_dir: Path, symlg_dir: Path, lgeval_root: Path) -> bool:
    """用官方 CROHME 转换工具链把 tex/txt 转成 symLG。"""
    symlg_dir.mkdir(parents=True, exist_ok=True)

    convert_root = lgeval_root / "convert2symLG"
    process_mml_py = convert_root / "process_mml.py"
    update_node_tags_py = convert_root / "update_nodeTags.py"
    batch_mml2lg = convert_root / "batch_mml2lg"
    if not process_mml_py.exists() or not update_node_tags_py.exists() or not batch_mml2lg.exists():
        return False

    bash_path, pandoc_path, perl_path = _resolve_tool_paths()
    if bash_path is None or pandoc_path is None or perl_path is None:
        return False

    env = os.environ.copy()
    env["CROHMELibDir"] = _to_bash_path(lgeval_root.parent / "crohmelib")
    env["LgEvalDir"] = _to_bash_path(lgeval_root)

    path_parts = [
        str(bash_path.parent),
        str(pandoc_path.parent),
        str(perl_path.parent),
        str(perl_path.parents[1] / "c" / "bin"),  # Strawberry perl 的 c 库目录依赖
    ]
    if env.get("PATH"):
        path_parts.append(env["PATH"])
    env["PATH"] = os.pathsep.join(path_parts)

    work_dir = symlg_dir.parent / "tex2symlg_work"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    mml_dir = work_dir / "mml_temp"
    inkml_dir = work_dir / "inkml_temp"
    lg_dir = work_dir / "lg_temp"
    mml_dir.mkdir(parents=True, exist_ok=True)
    inkml_dir.mkdir(parents=True, exist_ok=True)
    lg_dir.mkdir(parents=True, exist_ok=True)

    tex_files = sorted(tex_dir.glob("*.txt"))
    if not tex_files:
        return False

    for tex_file in tex_files:
        mml_file = mml_dir / f"{tex_file.stem}.mml"
        result = subprocess.run(
            [
                str(pandoc_path),
                "-f",
                "latex",
                "-t",
                "html",
                "--mathml",
                str(tex_file),
                "-o",
                str(mml_file),
            ],
            cwd=str(convert_root),
            env=env,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
        )
        if result.returncode != 0:
            print(f"[WARN] 跳过无法转换的样本: {tex_file.name}")
            print(f"[WARN] Pandoc STDERR: {result.stderr[:300]}")
            continue
        if not mml_file.exists() or mml_file.stat().st_size == 0:
            print(f"[WARN] 跳过空输出样本: {tex_file.name}")
            continue

    try:
        print(f"[DEBUG] 步骤1: MML -> InkML 转换")
        print(f"[DEBUG] process_mml.py: {process_mml_py}")
        print(f"[DEBUG] MML 输入目录: {mml_dir}, 文件数: {len(list(mml_dir.glob('*.mml')))}")
        print(f"[DEBUG] InkML 输出目录: {inkml_dir}")
        try:
            result = subprocess.run(
                [sys.executable, str(process_mml_py), str(mml_dir), str(inkml_dir)],
                cwd=str(convert_root),
                env=env,
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
            )
            print(f"[DEBUG] process_mml.py 返回码: {result.returncode}")
            if result.stdout:
                print(f"[DEBUG] STDOUT:\n{result.stdout[:500]}")
            if result.stderr:
                print(f"[DEBUG] STDERR:\n{result.stderr[:500]}")
            if result.returncode != 0:
                print(f"[ERROR] process_mml.py 失败，完整 STDERR: {result.stderr}")
                return False
        except Exception as e:
            print(f"[ERROR] 执行 process_mml.py 异常: {e}")
            return False
        print(f"[DEBUG] 步骤1 完成，生成 {len(list(inkml_dir.glob('*.inkml')))} 个 InkML 文件")

        print(f"[DEBUG] 步骤2: InkML -> LG 转换（直接 perl 调用并重定向输出）")
        mml2lg_pl = convert_root / "mml2lg.pl"
        if not mml2lg_pl.exists():
            print(f"[ERROR] 找不到 mml2lg.pl: {mml2lg_pl}")
            return False

        for inkml_file in sorted(inkml_dir.glob("*.inkml")):
            lg_file = lg_dir / f"{inkml_file.stem}.lg"
            with open(lg_file, "w", encoding="utf-8") as f_out:
                subprocess.run(
                    [str(perl_path), str(mml2lg_pl), "-s", str(inkml_file)],
                    cwd=str(convert_root),
                    env=env,
                    stdout=f_out,
                    stderr=subprocess.DEVNULL,
                )
        print(f"[DEBUG] 步骤2 完成，生成 {len(list(lg_dir.glob('*.lg')))} 个 LG 文件")

        print(f"[DEBUG] 步骤3: 更新节点标签")
        _run_subprocess(
            [sys.executable, str(update_node_tags_py), str(lg_dir), str(symlg_dir)],
            cwd=convert_root,
            env=env,
        )
        print(f"[DEBUG] 步骤3 完成，生成 {len(list(symlg_dir.glob('*.lg')))} 个 SymLG 文件")

        return any(symlg_dir.glob("*.lg"))
    except Exception as e:
        print(f"[ERROR] 转换链失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_lgeval(pred_symlg_dir: Path, gt_symlg_dir: Path, lgeval_root: Path) -> Path:
    if not pred_symlg_dir.exists():
        raise FileNotFoundError(f"找不到预测 symLG 目录: {pred_symlg_dir}")
    if not gt_symlg_dir.exists():
        raise FileNotFoundError(f"找不到 GT symLG 目录: {gt_symlg_dir}")

    evallg_py = lgeval_root / "src" / "evallg.py"
    if not evallg_py.exists():
        raise FileNotFoundError(f"找不到 lgeval 入口: {evallg_py}")

    env = os.environ.copy()
    py_path_items = [str(lgeval_root.parent.resolve())]
    if env.get("PYTHONPATH"):
        py_path_items.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(py_path_items)

    _run_subprocess(
        [sys.executable, str(evallg_py), str(pred_symlg_dir), str(gt_symlg_dir)],
        cwd=lgeval_root,
        env=env,
    )

    result_dir = lgeval_root / f"Results_{pred_symlg_dir.name}"
    if not result_dir.exists():
        raise RuntimeError(f"lgeval 未生成结果目录: {result_dir}")
    return result_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CROHME2019 PNG 数据评估脚本")
    parser.add_argument("--image_root", type=str, default=r"C:\Projects\LatexProject\CROHME\IMG\test\CROHME2019_test")
    parser.add_argument("--tokenizer", type=str, default=r"C:\Projects\LatexProject\ConvResFormula\tokenizer_bpe.json")
    parser.add_argument("--checkpoint", type=str, default=r"C:\Projects\LatexProject\ConvResFormula\checkpoints\ar\epoch_30.pth")
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--max_area", type=int, default=98304)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_len", type=int, default=160)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp_dtype", type=str, choices=["fp16", "bf16", "fp32"], default="bf16")
    parser.add_argument("--beam_size", type=int, default=3, help="Beam Search size. 1 selects greedy decoding")

    parser.add_argument("--out_dir", type=str, default="")

    # 官方评测相关参数
    parser.add_argument("--lgeval_root", type=str, default=r"C:\Projects\LatexProject\CROHME_eval\lgeval")
    parser.add_argument("--gt_symlg_dir", type=str, default="", help="若提供，则直接进行 LgEval 打分")
    parser.add_argument("--gt_tex_dir", type=str, default="", help="可选：GT 的 tex/txt 目录，尝试转换为 symLG")
    parser.add_argument("--run_lgeval", action="store_true", help="启用官方 LgEval 打分")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = PROJECT_ROOT / "logs" / "crohme2019_eval" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_tex_dir = run_inference(args, out_dir)
    print(f"[OK] 推理完成，预测文本目录: {pred_tex_dir}")
    print(f"[OK] 预测汇总: {out_dir / 'predictions.csv'}")

    if not args.run_lgeval:
        print("[INFO] 未启用 --run_lgeval，仅完成预测导出。")
        return

    lgeval_root = Path(args.lgeval_root)
    pred_symlg_dir = out_dir / "pred_symlg"

    converted_pred = try_convert_tex_to_symlg(pred_tex_dir, pred_symlg_dir, lgeval_root)
    if not converted_pred:
        print(
            "[WARN] 预测 tex -> symLG 没有生成可用结果，已跳过官方 LgEval。"
        )
        return

    gt_symlg_dir: Optional[Path] = Path(args.gt_symlg_dir) if args.gt_symlg_dir else None
    if gt_symlg_dir is None and args.gt_tex_dir:
        gt_symlg_dir = out_dir / "gt_symlg"
        converted_gt = try_convert_tex_to_symlg(Path(args.gt_tex_dir), gt_symlg_dir, lgeval_root)
        if not converted_gt:
            print("[WARN] GT tex -> symLG 转换失败，已跳过官方 LgEval。")
            return

    if gt_symlg_dir is None:
        print("[WARN] 缺少 GT，已跳过官方 LgEval。")
        return

    result_dir = run_lgeval(pred_symlg_dir, gt_symlg_dir, lgeval_root)
    print(f"[OK] LgEval 完成，结果目录: {result_dir}")
    summary_txt = result_dir / "Summary.txt"
    if summary_txt.exists():
        print(f"[OK] 结果摘要: {summary_txt}")


if __name__ == "__main__":
    main()
