"""统计仅保留 model 权重后的 checkpoint 体积。"""

from __future__ import annotations

import argparse
import io
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import torch


# 常见的优化器状态字段名，兼容不同训练脚本。
OPTIMIZER_KEYS: Tuple[str, ...] = (
    "optimizer",
    "optimizer_state_dict",
    "optim",
    "opt",
    "adam",
    "adam_state_dict",
)

# 常见的 scheduler 状态字段名。
SCHEDULER_KEYS: Tuple[str, ...] = (
    "scheduler",
    "scheduler_state_dict",
    "lr_scheduler",
)

# 常见的 model 权重字段名。
MODEL_KEYS: Tuple[str, ...] = (
    "model",
    "model_state_dict",
    "state_dict",
    "net",
    "network",
)


def _format_size(num_bytes: int) -> str:
    mb = num_bytes / (1024 * 1024)
    return f"{mb:.2f} MiB ({num_bytes} bytes)"


def _serialized_size_bytes(obj: Any) -> int:
    buffer = io.BytesIO()
    torch.save(obj, buffer)
    return buffer.getbuffer().nbytes


def _build_model_only_checkpoint(ckpt: Dict[str, Any]) -> Tuple[Dict[str, Any], Iterable[str], str]:
    model_key = ""
    for key in MODEL_KEYS:
        if key in ckpt:
            model_key = key
            break

    if not model_key:
        raise KeyError(
            "未找到 model 权重字段，请确认 checkpoint 中包含 model/model_state_dict/state_dict 等键"
        )

    removed_keys = [k for k in (*OPTIMIZER_KEYS, *SCHEDULER_KEYS) if k in ckpt]
    model_only = {model_key: ckpt[model_key]}
    return model_only, removed_keys, model_key


def _to_fp16_if_float_tensor(obj: Any) -> Any:
    # 仅转换浮点 Tensor，避免误改整型索引/计数等字段。
    if isinstance(obj, torch.Tensor):
        return obj.half() if obj.is_floating_point() else obj
    if isinstance(obj, dict):
        return {k: _to_fp16_if_float_tensor(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_fp16_if_float_tensor(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_to_fp16_if_float_tensor(v) for v in obj)
    return obj


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="查看仅保留 model 权重后的 checkpoint 体积")
    parser.add_argument("checkpoint", type=Path, help="输入 checkpoint 路径，例如 checkpoints/ar/best.pth")
    parser.add_argument(
        "--save-stripped",
        type=Path,
        default=None,
        help="可选：保存仅保留 model 权重后的 checkpoint 路径",
    )
    parser.add_argument(
        "--save-fp16",
        type=Path,
        default=None,
        help="可选：保存仅保留 model 且转换为 FP16 的 checkpoint 路径",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"checkpoint 不存在: {args.checkpoint}")

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise TypeError("checkpoint 顶层不是 dict，无法按键移除 optimizer 状态")

    model_only_ckpt, removed_keys, model_key = _build_model_only_checkpoint(ckpt)
    model_only_fp16_ckpt = _to_fp16_if_float_tensor(model_only_ckpt)

    disk_size = args.checkpoint.stat().st_size
    serialized_raw = _serialized_size_bytes(ckpt)
    serialized_stripped = _serialized_size_bytes(model_only_ckpt)
    serialized_fp16 = _serialized_size_bytes(model_only_fp16_ckpt)
    reduced_bytes = serialized_raw - serialized_stripped
    fp16_reduced_bytes = serialized_stripped - serialized_fp16

    print(f"[Checkpoint] {args.checkpoint}")
    print(f"- 原始文件大小(磁盘): {_format_size(disk_size)}")
    print(f"- 原始序列化大小(内存): {_format_size(serialized_raw)}")
    print(f"- 仅保留 model 后大小(内存): {_format_size(serialized_stripped)}")
    print(f"- 体积减少: {_format_size(reduced_bytes)}")
    print(f"- FP16(仅 model)大小(内存): {_format_size(serialized_fp16)}")
    print(f"- FP16 相对仅 model 减少: {_format_size(fp16_reduced_bytes)}")
    print(f"- 保留字段: {model_key}")
    print(
        f"- 已移除优化状态字段: {', '.join(removed_keys) if removed_keys else '未找到 optimizer/scheduler 字段'}"
    )

    if args.save_stripped is not None:
        args.save_stripped.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model_only_ckpt, args.save_stripped)
        saved_size = args.save_stripped.stat().st_size
        print(f"- 已保存精简 checkpoint: {args.save_stripped}")
        print(f"- 精简文件大小(磁盘): {_format_size(saved_size)}")

    if args.save_fp16 is not None:
        args.save_fp16.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model_only_fp16_ckpt, args.save_fp16)
        fp16_saved_size = args.save_fp16.stat().st_size
        print(f"- 已保存 FP16 checkpoint: {args.save_fp16}")
        print(f"- FP16 文件大小(磁盘): {_format_size(fp16_saved_size)}")


if __name__ == "__main__":
    main()
