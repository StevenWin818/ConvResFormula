"""学习率调度策略集合。"""

import math
from typing import Callable, List

import torch


def infer_total_steps(epochs: int, steps_per_epoch: int) -> int:
	"""根据 epoch 与每轮 step 数估算总训练步数。"""
	if epochs <= 0:
		raise ValueError(f"epochs 必须 > 0，当前为 {epochs}")
	if steps_per_epoch <= 0:
		raise ValueError(f"steps_per_epoch 必须 > 0，当前为 {steps_per_epoch}")
	return epochs * steps_per_epoch


def build_linear_warmup_cosine_scheduler(
	optimizer: torch.optim.Optimizer,
	total_steps: int,
	warmup_steps: int = 0,
	warmup_start_lr: float = 1e-7,
	eta_min: float = 1e-6,
	last_epoch: int = -1,
) -> torch.optim.lr_scheduler.LambdaLR:
	"""
	线性预热 + 余弦退火调度器（按 step 更新）。

	- 先线性 warmup（从 warmup_start_lr 升至各 param_group 的基础 lr）
	- 再执行余弦退火（衰减到 eta_min）
	"""
	if total_steps <= 0:
		raise ValueError(f"total_steps 必须 > 0，当前为 {total_steps}")
	if warmup_steps < 0:
		raise ValueError(f"warmup_steps 不能为负数，当前为 {warmup_steps}")
	if warmup_start_lr < 0:
		raise ValueError(f"warmup_start_lr 不能为负数，当前为 {warmup_start_lr}")
	if eta_min < 0:
		raise ValueError(f"eta_min 不能为负数，当前为 {eta_min}")

	warmup_steps = min(warmup_steps, total_steps)
	base_lrs: List[float] = [float(group["lr"]) for group in optimizer.param_groups]

	def make_lambda(base_lr: float) -> Callable[[int], float]:
		if base_lr <= 0:
			return lambda _: 1.0

		min_factor = min(1.0, eta_min / base_lr)
		start_factor = min(1.0, warmup_start_lr / base_lr)
		cosine_steps = max(1, total_steps - warmup_steps)

		def lr_lambda(current_step: int) -> float:
			# LambdaLR 的 step 从 0 开始累加
			step = max(0, current_step)

			if warmup_steps > 0 and step < warmup_steps:
				progress = float(step + 1) / float(warmup_steps)
				return start_factor + progress * (1.0 - start_factor)

			if warmup_steps >= total_steps:
				return 1.0

			cosine_progress = float(step - warmup_steps) / float(cosine_steps)
			cosine_progress = max(0.0, min(1.0, cosine_progress))
			cosine = 0.5 * (1.0 + math.cos(math.pi * cosine_progress))
			return min_factor + (1.0 - min_factor) * cosine

		return lr_lambda

	lr_lambdas = [make_lambda(base_lr) for base_lr in base_lrs]
	return torch.optim.lr_scheduler.LambdaLR(
		optimizer,
		lr_lambda=lr_lambdas,
		last_epoch=last_epoch,
	)