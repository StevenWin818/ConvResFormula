"""按比例缩放 AR checkpoint 中的学习率曲线。"""

from __future__ import annotations

import argparse
from copy import deepcopy
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch


def _format_lr_list(values: Iterable[Any]) -> str:
	items = []
	for value in values:
		try:
			items.append(f"{float(value):.8e}")
		except (TypeError, ValueError):
			items.append(str(value))
	return "[" + ", ".join(items) + "]"


def _scale_optimizer_state(optimizer_state: Dict[str, Any], factor: float) -> None:
	param_groups = optimizer_state.get("param_groups")
	if not isinstance(param_groups, list):
		return

	for group in param_groups:
		if not isinstance(group, dict):
			continue
		if "lr" in group:
			group["lr"] = float(group["lr"]) * factor
		if "initial_lr" in group:
			group["initial_lr"] = float(group["initial_lr"]) * factor


def _scale_scheduler_state(scheduler_state: Dict[str, Any], factor: float) -> None:
	base_lrs = scheduler_state.get("base_lrs")
	if isinstance(base_lrs, list):
		scheduler_state["base_lrs"] = [float(v) * factor for v in base_lrs]

	last_lrs = scheduler_state.get("_last_lr")
	if isinstance(last_lrs, list):
		scheduler_state["_last_lr"] = [float(v) * factor for v in last_lrs]

	if "_get_lr_called_within_step" in scheduler_state:
		# 这个状态只影响调度器内部告警，不随缩放变化。
		pass


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="按比例缩放 AR checkpoint 中的学习率曲线")
	parser.add_argument("checkpoint", type=Path, help="输入 checkpoint 路径，例如 checkpoints/ar/epoch_16.pth")
	parser.add_argument("--factor", type=float, default=1.5, help="学习率缩放倍数，默认 1.5")
	parser.add_argument("--output", type=Path, default=None, help="输出 checkpoint 路径；不指定时需配合 --in-place")
	parser.add_argument("--in-place", action="store_true", help="原地覆盖输入 checkpoint")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	if args.factor <= 0:
		raise ValueError(f"factor 必须大于 0，当前为 {args.factor}")
	if not args.checkpoint.exists():
		raise FileNotFoundError(f"checkpoint 不存在: {args.checkpoint}")
	if not args.in_place and args.output is None:
		raise ValueError("请指定 --output 或 --in-place")
	if args.output is not None and args.output == args.checkpoint and not args.in_place:
		raise ValueError("当输出路径与输入路径相同时，请使用 --in-place")

	ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
	if not isinstance(ckpt, dict):
		raise TypeError("checkpoint 顶层不是 dict，无法缩放学习率")

	original_optimizer_lrs: List[float] = []
	optimizer_state = ckpt.get("optimizer")
	if isinstance(optimizer_state, dict):
		param_groups = optimizer_state.get("param_groups")
		if isinstance(param_groups, list):
			for group in param_groups:
				if isinstance(group, dict) and "lr" in group:
					try:
						original_optimizer_lrs.append(float(group["lr"]))
					except (TypeError, ValueError):
						pass
		_scale_optimizer_state(optimizer_state, args.factor)

	original_scheduler_lrs: List[float] = []
	scheduler_state = ckpt.get("scheduler")
	if isinstance(scheduler_state, dict):
		base_lrs = scheduler_state.get("base_lrs")
		if isinstance(base_lrs, list):
			for value in base_lrs:
				try:
					original_scheduler_lrs.append(float(value))
				except (TypeError, ValueError):
					pass
		_scale_scheduler_state(scheduler_state, args.factor)

	if "ema_model" in ckpt and isinstance(ckpt["ema_model"], dict):
		# EMA 权重本身不需要缩放，但保留在 checkpoint 中，以便后续恢复训练。
		ckpt["ema_model"] = deepcopy(ckpt["ema_model"])

	output_path = args.checkpoint if args.in_place else args.output
	assert output_path is not None
	if args.in_place:
		backup_path = args.checkpoint.with_suffix(args.checkpoint.suffix + ".bak")
		shutil.copy2(args.checkpoint, backup_path)
		print(f"- 已创建备份: {backup_path}")
	output_path.parent.mkdir(parents=True, exist_ok=True)
	torch.save(ckpt, output_path)

	new_ckpt = torch.load(output_path, map_location="cpu", weights_only=False)
	new_optimizer_lrs: List[float] = []
	new_scheduler_lrs: List[float] = []
	if isinstance(new_ckpt, dict):
		optimizer_state = new_ckpt.get("optimizer")
		if isinstance(optimizer_state, dict):
			param_groups = optimizer_state.get("param_groups")
			if isinstance(param_groups, list):
				for group in param_groups:
					if isinstance(group, dict) and "lr" in group:
						new_optimizer_lrs.append(float(group["lr"]))
		scheduler_state = new_ckpt.get("scheduler")
		if isinstance(scheduler_state, dict):
			base_lrs = scheduler_state.get("base_lrs")
			if isinstance(base_lrs, list):
				new_scheduler_lrs = [float(value) for value in base_lrs]

	print(f"[Checkpoint] {args.checkpoint}")
	print(f"- 缩放倍数: {args.factor:.6f}")
	print(f"- 原始 optimizer lr: {_format_lr_list(original_optimizer_lrs)}")
	print(f"- 新的 optimizer lr: {_format_lr_list(new_optimizer_lrs)}")
	print(f"- 原始 scheduler base_lrs: {_format_lr_list(original_scheduler_lrs)}")
	print(f"- 新的 scheduler base_lrs: {_format_lr_list(new_scheduler_lrs)}")
	print(f"- 已保存到: {output_path}")


if __name__ == "__main__":
	main()