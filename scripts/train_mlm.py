"""MLM 训练脚本入口。"""

import argparse
import os
import random
import sys
from collections import Counter
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.optim as optim
from tokenizers import Tokenizer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

try:
	import yaml
except ImportError as exc:
	raise RuntimeError("请先安装 PyYAML: pip install pyyaml") from exc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from src.data.collate import MLMCollate
from src.data.dataset import MLMFormulaDataset
from src.engine.lr_schedulers import build_linear_warmup_cosine_scheduler, infer_total_steps
from src.engine.trainer import MLMTrainer
from src.models.latex_ocr_model import LatexOCRModel


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="ConvNeXt-V2 + AttnRes 的 MLM 训练脚本")

	parser.add_argument("--train-config", type=str, default=str(PROJECT_ROOT / "configs/train_mlm.yaml"))
	parser.add_argument("--model-config", type=str, default=str(PROJECT_ROOT / "configs/model_convnext_attnres.yaml"))

	parser.add_argument("--train_h5", type=str, nargs="+", default=None, help="训练集 H5 列表，可传多个文件。")
	parser.add_argument("--tokenizer", type=str, default=None, help="BPE Tokenizer JSON 路径。")
	parser.add_argument("--eval_h5", type=str, default=None, help="评估集 H5 路径。")
	parser.add_argument("--max_area", type=int, default=None, help="图像动态缩放最大面积。")

	parser.add_argument("--d_model", type=int, default=None)
	parser.add_argument("--batch_size", type=int, default=None)
	parser.add_argument("--num_workers", type=int, default=None)
	parser.add_argument("--epochs", type=int, default=None)
	parser.add_argument("--seed", type=int, default=None)

	parser.add_argument("--encoder_lr", type=float, default=None)
	parser.add_argument("--decoder_lr", type=float, default=None)
	parser.add_argument("--weight_decay", type=float, default=None)

	parser.add_argument("--warmup_epochs", type=float, default=None)
	parser.add_argument("--warmup_start_lr", type=float, default=None)
	parser.add_argument("--eta_min", type=float, default=None)

	parser.add_argument("--mask_prob", type=float, default=None)
	parser.add_argument("--log_interval", type=int, default=None)
	parser.add_argument("--eval_interval", type=int, default=None, help="每隔多少个 epoch 评估一次。")
	parser.add_argument("--eval_batch_size", type=int, default=None)
	parser.add_argument("--eval_num_workers", type=int, default=None)
	parser.add_argument("--eval_samples", type=int, default=None, help="每轮评估抽样数量，<=0 表示全量")
	parser.add_argument("--eval_max_len", type=int, default=None)
	parser.add_argument("--eval_max_iter", type=int, default=None)

	eval_group = parser.add_mutually_exclusive_group()
	eval_group.add_argument("--skip_eval", dest="skip_eval", action="store_true", help="仅训练不做评估")
	eval_group.add_argument("--enable_eval", dest="skip_eval", action="store_false", help="启用训练期评估")
	parser.set_defaults(skip_eval=None)

	parser.add_argument("--ckpt_dir", type=str, default=None)
	parser.add_argument("--log_dir", type=str, default=None)
	parser.add_argument("--resume_ckpt", type=str, default=None)

	resume_group = parser.add_mutually_exclusive_group()
	resume_group.add_argument("--resume", dest="resume", action="store_true", help="从 resume_ckpt 恢复完整训练状态")
	resume_group.add_argument("--fresh_start", dest="resume", action="store_false", help="不恢复断点，冷启动训练")
	parser.set_defaults(resume=None)

	return parser.parse_args()


def _cfg_get(cfg: Dict[str, Any], path: Tuple[str, ...], default: Any = None) -> Any:
	cur: Any = cfg
	for key in path:
		if not isinstance(cur, dict) or key not in cur:
			return default
		cur = cur[key]
	return cur


def load_yaml_config(path: str) -> Dict[str, Any]:
	if not os.path.exists(path):
		raise FileNotFoundError(f"配置文件不存在: {path}")
	with open(path, "r", encoding="utf-8") as f:
		data = yaml.safe_load(f)
	return data if isinstance(data, dict) else {}


def apply_config_defaults(args: argparse.Namespace, train_cfg: Dict[str, Any], model_cfg: Dict[str, Any]) -> None:
	args.train_h5 = args.train_h5 or _cfg_get(train_cfg, ("data", "train_h5"), [])
	args.tokenizer = args.tokenizer or _cfg_get(train_cfg, ("data", "tokenizer"))
	args.eval_h5 = args.eval_h5 or _cfg_get(train_cfg, ("data", "eval_h5"))
	args.max_area = args.max_area if args.max_area is not None else int(_cfg_get(train_cfg, ("data", "max_area"), 98304))

	args.d_model = args.d_model if args.d_model is not None else int(_cfg_get(model_cfg, ("model", "d_model"), 512))
	args.batch_size = args.batch_size if args.batch_size is not None else int(_cfg_get(train_cfg, ("training", "batch_size"), 24))
	args.num_workers = args.num_workers if args.num_workers is not None else int(_cfg_get(train_cfg, ("training", "num_workers"), 4))
	args.epochs = args.epochs if args.epochs is not None else int(_cfg_get(train_cfg, ("training", "epochs"), 20))
	args.seed = args.seed if args.seed is not None else int(_cfg_get(train_cfg, ("training", "seed"), 42))

	args.encoder_lr = args.encoder_lr if args.encoder_lr is not None else float(_cfg_get(train_cfg, ("optimization", "encoder_lr"), 5e-5))
	args.decoder_lr = args.decoder_lr if args.decoder_lr is not None else float(_cfg_get(train_cfg, ("optimization", "decoder_lr"), 1e-4))
	args.weight_decay = args.weight_decay if args.weight_decay is not None else float(_cfg_get(train_cfg, ("optimization", "weight_decay"), 1e-2))

	args.warmup_epochs = args.warmup_epochs if args.warmup_epochs is not None else float(_cfg_get(train_cfg, ("scheduler", "warmup_epochs"), 1.0))
	args.warmup_start_lr = args.warmup_start_lr if args.warmup_start_lr is not None else float(_cfg_get(train_cfg, ("scheduler", "warmup_start_lr"), 1e-7))
	args.eta_min = args.eta_min if args.eta_min is not None else float(_cfg_get(train_cfg, ("scheduler", "eta_min"), 1e-6))

	args.mask_prob = args.mask_prob if args.mask_prob is not None else float(_cfg_get(train_cfg, ("mlm", "mask_prob"), 0.15))

	args.log_interval = args.log_interval if args.log_interval is not None else int(_cfg_get(train_cfg, ("logging", "log_interval"), 20))
	args.eval_interval = args.eval_interval if args.eval_interval is not None else int(_cfg_get(train_cfg, ("eval", "eval_interval"), 1))
	args.eval_batch_size = args.eval_batch_size if args.eval_batch_size is not None else int(_cfg_get(train_cfg, ("eval", "batch_size"), 32))
	args.eval_num_workers = args.eval_num_workers if args.eval_num_workers is not None else int(_cfg_get(train_cfg, ("eval", "num_workers"), 2))
	args.eval_samples = args.eval_samples if args.eval_samples is not None else int(_cfg_get(train_cfg, ("eval", "samples"), 1500))
	args.eval_max_len = args.eval_max_len if args.eval_max_len is not None else int(_cfg_get(train_cfg, ("eval", "max_len"), 160))
	args.eval_max_iter = args.eval_max_iter if args.eval_max_iter is not None else int(_cfg_get(train_cfg, ("eval", "max_iter"), 4))

	if args.skip_eval is None:
		args.skip_eval = bool(_cfg_get(train_cfg, ("eval", "skip_eval"), False))

	args.ckpt_dir = args.ckpt_dir or str(_cfg_get(train_cfg, ("runtime", "ckpt_dir"), r"checkpoints\mlm"))
	args.log_dir = args.log_dir or str(_cfg_get(train_cfg, ("runtime", "log_dir"), r"logs\mlm_logs"))
	args.resume_ckpt = args.resume_ckpt or str(_cfg_get(train_cfg, ("runtime", "resume_ckpt"), r"checkpoints\mlm\latest.pth"))

	if args.resume is None:
		args.resume = bool(_cfg_get(train_cfg, ("runtime", "resume"), False))

	if not args.tokenizer:
		raise RuntimeError("配置缺少 tokenizer 路径，请在 train_mlm.yaml:data.tokenizer 中设置")
	if not args.train_h5:
		raise RuntimeError("配置缺少 train_h5 列表，请在 train_mlm.yaml:data.train_h5 中设置")
	if (not args.skip_eval) and (not args.eval_h5):
		raise RuntimeError("启用了评估但缺少 eval_h5，请在 train_mlm.yaml:data.eval_h5 中设置")


def set_seed(seed: int) -> None:
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)


def build_optimizer(model: LatexOCRModel, encoder_lr: float, decoder_lr: float, weight_decay: float) -> optim.AdamW:
	encoder_params: List[torch.nn.Parameter] = []
	other_params: List[torch.nn.Parameter] = []

	for name, param in model.named_parameters():
		if not param.requires_grad:
			continue
		if name.startswith("encoder."):
			encoder_params.append(param)
		else:
			other_params.append(param)

	if not encoder_params:
		raise RuntimeError("未找到 encoder 可训练参数")
	if not other_params:
		raise RuntimeError("未找到 decoder/head 可训练参数")

	return optim.AdamW(
		[
			{"params": encoder_params, "lr": encoder_lr},
			{"params": other_params, "lr": decoder_lr},
		],
		weight_decay=weight_decay,
	)


def collate_eval_batch(batch, pad_token_id: int) -> Dict[str, torch.Tensor]:
	"""评估阶段的 batch 组装：保留 GT token，不执行 MLM 掩盖。"""
	images = []
	input_ids_list = []

	for item in batch:
		images.append(item["image"])
		seq = item["input_ids"]
		if not isinstance(seq, torch.Tensor):
			seq = torch.tensor(seq, dtype=torch.long)
		input_ids_list.append(seq)

	max_h = max(img.shape[1] for img in images)
	max_w = max(img.shape[2] for img in images)
	batch_size = len(images)
	batched_images = torch.zeros((batch_size, 1, max_h, max_w), dtype=torch.float32)

	for i, img in enumerate(images):
		_, h, w = img.shape
		batched_images[i, :, :h, :w] = img

	target_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=pad_token_id)
	return {"images": batched_images, "target_ids": target_ids}


def levenshtein_distance(s1: str, s2: str) -> int:
	if s1 == s2:
		return 0
	if len(s1) == 0:
		return len(s2)
	if len(s2) == 0:
		return len(s1)

	previous_row = list(range(len(s2) + 1))
	for i, c1 in enumerate(s1, start=1):
		current_row = [i]
		for j, c2 in enumerate(s2, start=1):
			insertions = previous_row[j] + 1
			deletions = current_row[j - 1] + 1
			substitutions = previous_row[j - 1] + (c1 != c2)
			current_row.append(min(insertions, deletions, substitutions))
		previous_row = current_row
	return previous_row[-1]


@torch.no_grad()
def batched_infer_mlm_iterative(
	model: LatexOCRModel,
	images: torch.Tensor,
	pad_id: int,
	mask_id: int,
	bos_id: int,
	eos_id: int,
	max_len: int,
	max_iter: int,
	amp_enabled: bool,
) -> List[List[int]]:
	"""批量版 Mask-Predict 解码，用于加速训练期评估。"""
	batch_size = images.size(0)
	device = images.device

	token_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long, device=device)
	token_ids[:, 0] = bos_id
	if max_len > 1:
		token_ids[:, 1:] = mask_id

	for step in range(max_iter):
		with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
			logits = model(images=images, tgt_seq=token_ids, is_causal=False)

		probs = torch.softmax(logits, dim=-1)
		max_probs, preds = torch.max(probs, dim=-1)

		is_mask = token_ids == mask_id
		token_ids[is_mask] = preds[is_mask]

		if step == max_iter - 1:
			break

		mask_ratio = 1.0 - ((step + 1) / max_iter)
		num_mask = int((max_len - 1) * mask_ratio)
		if num_mask <= 0:
			break

		valid_positions = (token_ids != bos_id) & (token_ids != pad_id)
		valid_probs = max_probs.masked_fill(~valid_positions, float("inf"))
		_, least_confident_indices = torch.topk(valid_probs, num_mask, dim=-1, largest=False)
		token_ids.scatter_(1, least_confident_indices, mask_id)

	output_batch: List[List[int]] = []
	for row in token_ids.cpu().tolist():
		if eos_id in row:
			row = row[: row.index(eos_id)]
		output_batch.append(row)

	return output_batch


@torch.no_grad()
def evaluate_epoch(
	model: LatexOCRModel,
	eval_loader: DataLoader,
	tokenizer: Tokenizer,
	device: torch.device,
	pad_id: int,
	mask_id: int,
	bos_id: int,
	eos_id: int,
	num_samples: int,
	max_len: int,
	max_iter: int,
) -> Tuple[float, Counter[str], Dict[str, float]]:
	"""参考 train_2d_phase2 的评估流程：按 epoch 抽样评估 + Top 错误统计。"""
	model.eval()
	amp_enabled = device.type == "cuda"

	processed = 0
	exact = 0
	total_ned = 0.0
	error_counter: Counter[str] = Counter()

	pbar = tqdm(eval_loader, desc="Eval", leave=False)
	for batch in pbar:
		if num_samples > 0 and processed >= num_samples:
			break

		images = batch["images"].to(device)
		target_ids = batch["target_ids"]

		pred_batch = batched_infer_mlm_iterative(
			model=model,
			images=images,
			pad_id=pad_id,
			mask_id=mask_id,
			bos_id=bos_id,
			eos_id=eos_id,
			max_len=max_len,
			max_iter=max_iter,
			amp_enabled=amp_enabled,
		)

		for i in range(len(pred_batch)):
			if num_samples > 0 and processed >= num_samples:
				break

			pred_text = tokenizer.decode(pred_batch[i], skip_special_tokens=True).strip()
			gt_text = tokenizer.decode(target_ids[i].tolist(), skip_special_tokens=True).strip()

			dist = levenshtein_distance(pred_text, gt_text)
			ned = dist / max(1, len(gt_text))

			exact += int(pred_text == gt_text)
			total_ned += ned
			processed += 1

			if pred_text != gt_text:
				error_counter[f"GT={gt_text} || PRED={pred_text}"] += 1

		if processed > 0:
			em = exact / processed
			avg_ned = total_ned / processed
			pbar.set_postfix({"EM": f"{em * 100:.2f}%", "NED": f"{avg_ned:.4f}"})

	em_rate = (exact / processed) if processed > 0 else 0.0
	avg_ned = (total_ned / processed) if processed > 0 else 1.0
	stats = {
		"processed": float(processed),
		"exact": float(exact),
		"em": em_rate,
		"avg_ned": avg_ned,
	}
	return em_rate, error_counter, stats


def main() -> None:
	args = parse_args()
	train_cfg = load_yaml_config(args.train_config)
	model_cfg = load_yaml_config(args.model_config)
	apply_config_defaults(args, train_cfg, model_cfg)
	set_seed(args.seed)

	os.makedirs(args.ckpt_dir, exist_ok=True)
	os.makedirs(args.log_dir, exist_ok=True)
	log_path = os.path.join(args.log_dir, "train_mlm.log")

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	amp_enabled = device.type == "cuda"

	tokenizer = Tokenizer.from_file(args.tokenizer)
	vocab_size = tokenizer.get_vocab_size()

	pad_id = tokenizer.token_to_id("[PAD]")
	mask_id = tokenizer.token_to_id("[MASK]")
	bos_id = tokenizer.token_to_id("[BOS]")
	eos_id = tokenizer.token_to_id("[EOS]")
	unk_id = tokenizer.token_to_id("[UNK]")

	if pad_id is None or mask_id is None:
		raise RuntimeError("Tokenizer 缺少 [PAD] 或 [MASK]，无法进行 MLM 训练")
	if bos_id is None or eos_id is None:
		raise RuntimeError("Tokenizer 缺少 [BOS] 或 [EOS]，无法进行迭代评估解码")

	special_token_ids = [idx for idx in [pad_id, mask_id, bos_id, eos_id, unk_id] if idx is not None]

	dataset_paths = [p for p in args.train_h5 if os.path.exists(p)]
	if not dataset_paths:
		raise FileNotFoundError(f"训练集不存在，请检查路径: {args.train_h5}")

	datasets = [
		MLMFormulaDataset(h5_path=path, tokenizer_path=args.tokenizer, max_area=args.max_area)
		for path in dataset_paths
	]
	train_dataset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)

	collate_fn = MLMCollate(
		pad_token_id=pad_id,
		mask_token_id=mask_id,
		vocab_size=vocab_size,
		special_token_ids=special_token_ids,
		mask_prob=args.mask_prob,
	)

	train_loader = DataLoader(
		train_dataset,
		batch_size=args.batch_size,
		shuffle=True,
		collate_fn=collate_fn,
		num_workers=args.num_workers,
		pin_memory=(device.type == "cuda"),
		persistent_workers=(args.num_workers > 0),
	)

	eval_loader = None
	if not args.skip_eval:
		if not os.path.exists(args.eval_h5):
			raise FileNotFoundError(f"评估集不存在: {args.eval_h5}")

		eval_dataset = MLMFormulaDataset(
			h5_path=args.eval_h5,
			tokenizer_path=args.tokenizer,
			max_area=args.max_area,
		)
		eval_loader = DataLoader(
			eval_dataset,
			batch_size=args.eval_batch_size,
			shuffle=False,
			collate_fn=partial(collate_eval_batch, pad_token_id=pad_id),
			num_workers=args.eval_num_workers,
			pin_memory=(device.type == "cuda"),
			persistent_workers=(args.eval_num_workers > 0),
		)

	model = LatexOCRModel(vocab_size=vocab_size, d_model=args.d_model, pad_id=pad_id).to(device)
	optimizer = build_optimizer(
		model=model,
		encoder_lr=args.encoder_lr,
		decoder_lr=args.decoder_lr,
		weight_decay=args.weight_decay,
	)

	total_steps = infer_total_steps(args.epochs, len(train_loader))
	warmup_steps = int(args.warmup_epochs * len(train_loader))
	scheduler = build_linear_warmup_cosine_scheduler(
		optimizer=optimizer,
		total_steps=total_steps,
		warmup_steps=warmup_steps,
		warmup_start_lr=args.warmup_start_lr,
		eta_min=args.eta_min,
	)

	scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
	trainer = MLMTrainer(
		model=model,
		optimizer=optimizer,
		scheduler=scheduler,
		device=device,
		scaler=scaler if amp_enabled else None,
	)

	start_epoch = 1
	best_loss = float("inf")
	best_em = 0.0

	if args.resume and os.path.exists(args.resume_ckpt):
		checkpoint = torch.load(args.resume_ckpt, map_location=device, weights_only=False)
		model.load_state_dict(checkpoint["model"], strict=True)
		optimizer.load_state_dict(checkpoint["optimizer"])
		scheduler.load_state_dict(checkpoint["scheduler"])
		if amp_enabled and checkpoint.get("scaler") is not None:
			scaler.load_state_dict(checkpoint["scaler"])

		start_epoch = int(checkpoint.get("epoch", 0)) + 1
		best_loss = float(checkpoint.get("best_loss", best_loss))
		best_em = float(checkpoint.get("best_em", best_em))
		print(f"恢复训练: epoch={start_epoch}, best_loss={best_loss:.4f}, best_em={best_em * 100:.2f}%")

	if not os.path.exists(log_path):
		with open(log_path, "w", encoding="utf-8") as f:
			f.write("Timestamp\tEpoch\tLoss\tMaskAcc(%)\tEvalProcessed\tEvalExact\tEvalEM(%)\tEvalNED\tLREnc\tLRDec\n")

	print("=" * 80)
	print(f"MLM 训练启动 | Device={device} | AMP={amp_enabled}")
	print(f"TrainConfig={args.train_config}")
	print(f"ModelConfig={args.model_config}")
	print(f"Datasets={dataset_paths}")
	if not args.skip_eval:
		print(f"EvalSet={args.eval_h5} | EvalBatch={args.eval_batch_size} | EvalSamples={args.eval_samples}")
	print(f"Total Steps={total_steps}, Warmup Steps={warmup_steps}")
	print("=" * 80)

	for epoch in range(start_epoch, args.epochs + 1):
		avg_loss, avg_acc = trainer.train_epoch(train_loader, epoch=epoch, log_interval=args.log_interval)

		eval_processed = 0.0
		eval_exact = 0.0
		eval_em = 0.0
		eval_ned = 1.0
		error_counter: Counter[str] = Counter()

		if (not args.skip_eval) and eval_loader is not None and (epoch % max(1, args.eval_interval) == 0):
			eval_em, error_counter, eval_stats = evaluate_epoch(
				model=model,
				eval_loader=eval_loader,
				tokenizer=tokenizer,
				device=device,
				pad_id=pad_id,
				mask_id=mask_id,
				bos_id=bos_id,
				eos_id=eos_id,
				num_samples=args.eval_samples,
				max_len=args.eval_max_len,
				max_iter=args.eval_max_iter,
			)
			eval_processed = eval_stats["processed"]
			eval_exact = eval_stats["exact"]
			eval_ned = eval_stats["avg_ned"]

		lr_enc = float(optimizer.param_groups[0]["lr"])
		lr_dec = float(optimizer.param_groups[1]["lr"])

		print(
			f"Epoch {epoch}/{args.epochs} | "
			f"Loss={avg_loss:.4f} | MaskAcc={avg_acc * 100:.2f}% | "
			f"EvalEM={eval_em * 100:.2f}% | EvalNED={eval_ned:.4f} | "
			f"LRE={lr_enc:.2e} | LRD={lr_dec:.2e}"
		)

		timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
		with open(log_path, "a", encoding="utf-8") as f:
			f.write(
				f"{timestamp}\t{epoch}\t{avg_loss:.6f}\t{avg_acc * 100:.4f}\t"
				f"{int(eval_processed)}\t{int(eval_exact)}\t{eval_em * 100:.4f}\t{eval_ned:.6f}\t"
				f"{lr_enc:.8e}\t{lr_dec:.8e}\n"
			)

		ckpt_payload: Dict[str, object] = {
			"epoch": epoch,
			"model": model.state_dict(),
			"optimizer": optimizer.state_dict(),
			"scheduler": scheduler.state_dict(),
			"best_loss": best_loss,
			"best_em": best_em,
			"latest_em": eval_em,
			"eval_processed": int(eval_processed),
			"eval_exact": int(eval_exact),
			"eval_ned": float(eval_ned),
			"config": vars(args),
		}
		if amp_enabled:
			ckpt_payload["scaler"] = scaler.state_dict()

		latest_ckpt = os.path.join(args.ckpt_dir, "latest.pth")
		epoch_ckpt = os.path.join(args.ckpt_dir, f"epoch_{epoch}.pth")
		best_ckpt = os.path.join(args.ckpt_dir, "best.pth")

		torch.save(ckpt_payload, latest_ckpt)
		torch.save(ckpt_payload, epoch_ckpt)

		if error_counter:
			err_path = os.path.join(args.log_dir, f"mlm_epoch_{epoch}_top_errors.txt")
			with open(err_path, "w", encoding="utf-8") as ef:
				for key, count in error_counter.most_common(20):
					ef.write(f"[{count:5d}] {key}\n")

		if eval_em >= best_em:
			best_em = eval_em
			ckpt_payload["best_em"] = best_em
			torch.save(ckpt_payload, best_ckpt)
			print(f"更新 best checkpoint: {best_ckpt} | best_em={best_em * 100:.2f}%")

		if avg_loss <= best_loss:
			best_loss = avg_loss
			ckpt_payload["best_loss"] = best_loss
			print(f"更新最小训练损失: {best_loss:.4f}")

	print("\nMLM 训练完成。")


if __name__ == "__main__":
	main()
