"""AR 贪心解码评估脚本入口。"""

import argparse
import os
import sys
from collections import Counter
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Dict, List, Tuple
from typing import cast

import torch
from tokenizers import Tokenizer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import FormulaDataset
from src.models.latex_ocr_model import LatexOCRModel
from src.models.text_decoder import convert_legacy_attnres_state_dict
def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="AR 贪心解码评估脚本")
	parser.add_argument("--eval_h5", type=str, default=r"C:\Projects\LatexProject\ConvResFormula\datasets\val.h5")
	parser.add_argument("--tokenizer", type=str, default=r"C:\Projects\LatexProject\ConvResFormula\tokenizer_bpe.json")
	parser.add_argument("--checkpoint", type=str, default=r"checkpoints\ar\best.pth")
	parser.add_argument("--d_model", type=int, default=512)
	parser.add_argument("--max_area", type=int, default=98304)
	parser.add_argument("--batch_size", type=int, default=32)
	parser.add_argument("--num_workers", type=int, default=4)
	parser.add_argument("--max_len", type=int, default=160)
	parser.add_argument("--max_samples", type=int, default=0, help="评估样本数上限，<=0 表示全量")
	parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
	parser.add_argument("--report_dir", type=str, default=r"logs\ar_logs")
	parser.add_argument("--log_interval", type=int, default=100)
	return parser.parse_args()


def levenshtein_distance(s1: str, s2: str) -> int:
	"""计算两个字符串的编辑距离。"""
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


def collate_eval_batch(batch, pad_token_id: int) -> Dict[str, torch.Tensor]:
	"""评估阶段 batch 组装：只做图像与 target 拼接，不做掩码。"""
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


@torch.no_grad()
def batched_infer_ar(
	model: LatexOCRModel,
	images: torch.Tensor,
	pad_id: int,
	bos_id: int,
	eos_id: int,
	max_len: int,
	amp_enabled: bool,
) -> List[List[int]]:
	"""批量版 AR 贪心解码。"""
	batch_size = images.size(0)
	device = images.device

	with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
		memory, memory_padding_mask = model.encode(images)
		decode_cache = model.init_decode_cache(memory)

	generated = torch.full((batch_size, max_len), pad_id, dtype=torch.long, device=device)
	generated[:, 0] = bos_id
	finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
	# 循环从步数 1 开始，直到 max_len
	for step in range(1, max_len):
		with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
			current_token = generated[:, step - 1]
			logits, decode_cache = model.decode_step_cached(
				memory=memory,
				token_id=current_token,
				cache=decode_cache,
				memory_padding_mask=memory_padding_mask,
			)

		next_tokens = logits.argmax(dim=-1)
		next_tokens = next_tokens.masked_fill(finished, pad_id)

		# 直接原地切片赋值，避免逐步拼接带来的额外开销。
		generated[:, step] = next_tokens

		finished |= (next_tokens == eos_id)
		if finished.all():
			break

	output_batch: List[List[int]] = []
	for row in generated.cpu().tolist():
		if eos_id in row:
			row = row[: row.index(eos_id)]
		output_batch.append(row)

	return output_batch


def build_model(checkpoint_path: str, tokenizer: Tokenizer, d_model: int, device: torch.device) -> LatexOCRModel:
	vocab_size = tokenizer.get_vocab_size()
	pad_id = tokenizer.token_to_id("[PAD]")
	if pad_id is None:
		raise RuntimeError("Tokenizer 缺少 [PAD] token")

	model = LatexOCRModel(vocab_size=vocab_size, d_model=d_model, pad_id=pad_id).to(device)

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
	if hasattr(torch, "compile"):
		model = cast(LatexOCRModel, torch.compile(model))
	return model


def evaluate(args: argparse.Namespace) -> Tuple[float, float, int]:
	device = torch.device(args.device)
	tokenizer = Tokenizer.from_file(args.tokenizer)

	pad_id = tokenizer.token_to_id("[PAD]")
	bos_id = tokenizer.token_to_id("[BOS]")
	eos_id = tokenizer.token_to_id("[EOS]")
	if pad_id is None or bos_id is None or eos_id is None:
		raise RuntimeError("Tokenizer 缺少 [PAD]/[BOS]/[EOS]，无法进行 AR 评估")

	dataset = FormulaDataset(
		h5_path=args.eval_h5,
		tokenizer_path=args.tokenizer,
		max_area=args.max_area,
	)
	eval_loader = DataLoader(
		dataset,
		batch_size=args.batch_size,
		shuffle=False,
		collate_fn=partial(collate_eval_batch, pad_token_id=pad_id),
		num_workers=args.num_workers,
		pin_memory=(device.type == "cuda"),
		persistent_workers=(args.num_workers > 0),
	)
	model = build_model(args.checkpoint, tokenizer, args.d_model, device)
	amp_enabled = device.type == "cuda"

	max_samples = len(dataset) if args.max_samples <= 0 else min(args.max_samples, len(dataset))
	if max_samples <= 0:
		raise RuntimeError("评估集为空，无法评估")

	processed = 0
	exact_match = 0
	total_ed = 0
	total_ned = 0.0
	mismatch_counter: Counter[str] = Counter()
	report_rows: List[str] = []

	iterator = tqdm(eval_loader, desc="Eval")
	for batch in iterator:
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
			max_len=args.max_len,
			amp_enabled=amp_enabled,
		)

		for i in range(len(pred_batch)):
			if processed >= max_samples:
				break

			pred_text = tokenizer.decode(pred_batch[i], skip_special_tokens=True).strip()
			target_text = tokenizer.decode(target_ids[i].tolist(), skip_special_tokens=True).strip()

			dist = levenshtein_distance(pred_text, target_text)
			ned = dist / max(1, len(target_text))

			is_exact = pred_text == target_text
			exact_match += int(is_exact)
			total_ed += dist
			total_ned += ned

			if not is_exact:
				key = f"GT={target_text} || PRED={pred_text}"
				mismatch_counter[key] += 1
				report_rows.append(
					f"[idx={processed}] ED={dist}, NED={ned:.4f}\nGT  : {target_text}\nPRED: {pred_text}\n"
				)

			processed += 1

		if processed > 0 and (processed % max(1, args.log_interval) == 0):
			em = exact_match / processed
			avg_ned = total_ned / processed
			iterator.set_postfix({"EM": f"{em * 100:.2f}%", "NED": f"{avg_ned:.4f}"})

	em_rate = exact_match / processed
	avg_ed = total_ed / processed
	avg_ned = total_ned / processed

	os.makedirs(args.report_dir, exist_ok=True)
	ts = datetime.now().strftime("%Y%m%d_%H%M%S")
	report_path = os.path.join(args.report_dir, f"ar_eval_report_{ts}.txt")
	top_error_path = os.path.join(args.report_dir, f"ar_eval_top_errors_{ts}.txt")

	with open(report_path, "w", encoding="utf-8") as f:
		f.write("=== AR Eval Report ===\n")
		f.write(f"checkpoint: {args.checkpoint}\n")
		f.write(f"eval_h5: {args.eval_h5}\n")
		f.write(f"EvalProcessed: {processed}\n")
		f.write(f"EvalExact: {exact_match}\n")
		f.write(f"EvalEM(%): {em_rate * 100:.4f}\n")
		f.write(f"EvalNED: {avg_ned:.6f}\n")
		f.write(f"AvgED: {avg_ed:.4f}\n\n")

		f.write("=== Top 20 Errors (same format as train) ===\n")
		for err, cnt in mismatch_counter.most_common(20):
			f.write(f"[{cnt:5d}] {err}\n")

		f.write("\n=== Detailed Mismatches (Top 200) ===\n")
		for row in report_rows[:200]:
			f.write(row + "\n")

	with open(top_error_path, "w", encoding="utf-8") as ef:
		for err, cnt in mismatch_counter.most_common(20):
			ef.write(f"[{cnt:5d}] {err}\n")

	print("\n" + "=" * 80)
	print("评估完成")
	print(f"EvalProcessed : {processed}")
	print(f"EvalExact     : {exact_match}")
	print(f"EvalEM(%)     : {em_rate * 100:.4f}")
	print(f"EvalNED       : {avg_ned:.6f}")
	print(f"AvgED         : {avg_ed:.4f}")
	print(f"Report        : {report_path}")
	print(f"TopErrors     : {top_error_path}")
	print("=" * 80)

	return em_rate, avg_ned, processed


if __name__ == "__main__":
	evaluate(parse_args())
