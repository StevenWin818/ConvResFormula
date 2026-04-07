"""审计 checkpoint 的参数量与体积构成。"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import torch


MODEL_KEYS: Tuple[str, ...] = ("model", "model_state_dict", "state_dict", "net", "network")


def _format_size(num_bytes: int) -> str:
    return f"{num_bytes / (1024 * 1024):.2f} MiB ({num_bytes} bytes)"


def _pick_model_key(ckpt: Dict[str, Any]) -> str:
    for key in MODEL_KEYS:
        if key in ckpt and isinstance(ckpt[key], dict):
            return key
    raise KeyError("未找到可用的模型权重字段（model/model_state_dict/state_dict/net/network）")


def _iter_tensors(state_dict: Dict[str, Any]) -> Iterable[Tuple[str, torch.Tensor]]:
    for name, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            yield name, value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="审计 checkpoint 参数量、体积构成与 backbone 深度")
    parser.add_argument("checkpoint", type=Path, help="checkpoint 路径")
    parser.add_argument("--topk", type=int, default=20, help="输出前缀统计 TopK")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"checkpoint 不存在: {args.checkpoint}")

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise TypeError("checkpoint 顶层不是 dict")

    model_key = _pick_model_key(ckpt)
    state_dict = ckpt[model_key]

    total_params = 0
    total_bytes = 0
    total_tensors = 0

    dtype_params: Dict[str, int] = defaultdict(int)
    dtype_bytes: Dict[str, int] = defaultdict(int)
    prefix_stats: Dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])  # params, bytes, tensors
    stage_blocks: Dict[int, set[int]] = defaultdict(set)
    stage_params: Dict[int, int] = defaultdict(int)

    for name, tensor in _iter_tensors(state_dict):
        numel = tensor.numel()
        nbytes = numel * tensor.element_size()

        total_params += numel
        total_bytes += nbytes
        total_tensors += 1

        dtype = str(tensor.dtype)
        dtype_params[dtype] += numel
        dtype_bytes[dtype] += nbytes

        prefix = name.split(".")[0]
        prefix_stats[prefix][0] += numel
        prefix_stats[prefix][1] += nbytes
        prefix_stats[prefix][2] += 1

        # 匹配 encoder.backbone.stages_2.blocks.7.xxx 这类键名
        match = re.match(r"encoder\.backbone\.stages_(\d+)\.blocks\.(\d+)\.", name)
        if match:
            stage_id = int(match.group(1))
            block_id = int(match.group(2))
            stage_blocks[stage_id].add(block_id)
            stage_params[stage_id] += numel

    print(f"[Checkpoint] {args.checkpoint}")
    print(f"- model 字段: {model_key}")
    print(f"- Tensor 数量: {total_tensors}")
    print(f"- 总参数量: {total_params:,}")
    print(f"- 参数体积(FP权重字节统计): {_format_size(total_bytes)}")

    print("- dtype 分布:")
    for dtype in sorted(dtype_params.keys()):
        print(f"  - {dtype}: params={dtype_params[dtype]:,}, size={_format_size(dtype_bytes[dtype])}")

    print(f"- 顶层前缀 Top{args.topk} (按参数量降序):")
    sorted_prefixes = sorted(prefix_stats.items(), key=lambda item: item[1][0], reverse=True)
    for prefix, (params, nbytes, tensors) in sorted_prefixes[: args.topk]:
        ratio = (params / total_params * 100.0) if total_params > 0 else 0.0
        print(
            f"  - {prefix}: params={params:,} ({ratio:.2f}%), size={_format_size(nbytes)}, tensors={tensors}"
        )

    if stage_blocks:
        print("- ConvNeXt stage 深度与参数:")
        for stage_id in sorted(stage_blocks.keys()):
            block_count = len(stage_blocks[stage_id])
            max_idx = max(stage_blocks[stage_id])
            print(
                f"  - stage_{stage_id}: blocks={block_count} (max_idx={max_idx}), "
                f"params={stage_params[stage_id]:,}, size={_format_size(stage_params[stage_id] * 4)}"
            )


if __name__ == "__main__":
    main()
