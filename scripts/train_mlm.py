"""MLM 训练脚本入口。"""

import argparse
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.optim as optim
from tokenizers import Tokenizer
from torch.utils.data import ConcatDataset, DataLoader

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

	parser.add_argument(
		"--train_h5",
		type=str,
		nargs="+",
		default=[
			r"C:\Projects\LatexProject\ConvResFormula\datasets\train.h5",
			r"C:\Projects\LatexProject\ConvResFormula\datasets\synthetic.h5",
			r"C:\Projects\LatexProject\ConvResFormula\datasets\symbols.h5",
		],
		help="训练集 H5 列表，可传多个文件。",
	)
	parser.add_argument(
		"--tokenizer",
		type=str,
		default=r"C:\Projects\LatexProject\ConvResFormula\tokenizer_bpe.json",
		help="BPE Tokenizer JSON 路径。",
	)
	parser.add_argument("--max_area", type=int, default=98304, help="图像动态缩放最大面积。")

	parser.add_argument("--d_model", type=int, default=512)
	parser.add_argument("--batch_size", type=int, default=24)
	parser.add_argument("--num_workers", type=int, default=4)
	parser.add_argument("--epochs", type=int, default=20)
	parser.add_argument("--seed", type=int, default=42)

	parser.add_argument("--encoder_lr", type=float, default=5e-5)
	parser.add_argument("--decoder_lr", type=float, default=1e-4)
	parser.add_argument("--weight_decay", type=float, default=1e-2)

	parser.add_argument("--warmup_epochs", type=float, default=1.0)
	parser.add_argument("--warmup_start_lr", type=float, default=1e-7)
	parser.add_argument("--eta_min", type=float, default=1e-6)

	parser.add_argument("--mask_prob", type=float, default=0.15)
	parser.add_argument("--log_interval", type=int, default=20)

	parser.add_argument("--ckpt_dir", type=str, default=r"checkpoints\mlm")
	parser.add_argument("--log_dir", type=str, default=r"logs\mlm_logs")
	parser.add_argument("--resume_ckpt", type=str, default=r"checkpoints\mlm\latest.pth")
	parser.add_argument("--resume", action="store_true", help="从 resume_ckpt 恢复完整训练状态")

	return parser.parse_args()


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


def main() -> None:
	args = parse_args()
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

	special_token_ids = [idx for idx in [pad_id, mask_id, bos_id, eos_id, unk_id] if idx is not None]

	dataset_paths = [p for p in args.train_h5 if os.path.exists(p)]
	if not dataset_paths:
		raise FileNotFoundError(f"训练集不存在，请检查路径: {args.train_h5}")

	datasets = [MLMFormulaDataset(h5_path=path, tokenizer_path=args.tokenizer, max_area=args.max_area) for path in dataset_paths]
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

	if args.resume and os.path.exists(args.resume_ckpt):
		checkpoint = torch.load(args.resume_ckpt, map_location=device, weights_only=False)
		model.load_state_dict(checkpoint["model"], strict=True)
		optimizer.load_state_dict(checkpoint["optimizer"])
		scheduler.load_state_dict(checkpoint["scheduler"])
		if amp_enabled and checkpoint.get("scaler") is not None:
			scaler.load_state_dict(checkpoint["scaler"])

		start_epoch = int(checkpoint.get("epoch", 0)) + 1
		best_loss = float(checkpoint.get("best_loss", best_loss))
		print(f"🔄 恢复训练: epoch={start_epoch}, best_loss={best_loss:.4f}")

	if not os.path.exists(log_path):
		with open(log_path, "w", encoding="utf-8") as f:
			f.write("Timestamp\tEpoch\tLoss\tMaskAcc(%)\tLREnc\tLRDec\n")

	print("=" * 80)
	print(f"MLM 训练启动 | Device={device} | AMP={amp_enabled}")
	print(f"Datasets={dataset_paths}")
	print(f"Total Steps={total_steps}, Warmup Steps={warmup_steps}")
	print("=" * 80)

	for epoch in range(start_epoch, args.epochs + 1):
		avg_loss, avg_acc = trainer.train_epoch(train_loader, epoch=epoch, log_interval=args.log_interval)

		lr_enc = float(optimizer.param_groups[0]["lr"])
		lr_dec = float(optimizer.param_groups[1]["lr"])

		print(
			f"📉 Epoch {epoch}/{args.epochs} | "
			f"Loss={avg_loss:.4f} | MaskAcc={avg_acc * 100:.2f}% | "
			f"LRE={lr_enc:.2e} | LRD={lr_dec:.2e}"
		)

		timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
		with open(log_path, "a", encoding="utf-8") as f:
			f.write(
				f"{timestamp}\t{epoch}\t{avg_loss:.6f}\t{avg_acc * 100:.4f}\t"
				f"{lr_enc:.8e}\t{lr_dec:.8e}\n"
			)

		ckpt_payload: Dict[str, object] = {
			"epoch": epoch,
			"model": model.state_dict(),
			"optimizer": optimizer.state_dict(),
			"scheduler": scheduler.state_dict(),
			"best_loss": best_loss,
			"config": vars(args),
		}
		if amp_enabled:
			ckpt_payload["scaler"] = scaler.state_dict()

		latest_ckpt = os.path.join(args.ckpt_dir, "latest.pth")
		epoch_ckpt = os.path.join(args.ckpt_dir, f"epoch_{epoch}.pth")
		best_ckpt = os.path.join(args.ckpt_dir, "best.pth")

		torch.save(ckpt_payload, latest_ckpt)
		torch.save(ckpt_payload, epoch_ckpt)

		if avg_loss <= best_loss:
			best_loss = avg_loss
			ckpt_payload["best_loss"] = best_loss
			torch.save(ckpt_payload, best_ckpt)
			print(f"✅ 更新 best checkpoint: {best_ckpt} | best_loss={best_loss:.4f}")

	print("\n🎉 MLM 训练完成。")


if __name__ == "__main__":
	main()
