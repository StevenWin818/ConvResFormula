"""AR 贪心解码评估脚本入口。"""

import argparse
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple, cast

import matplotlib
import numpy as np
import torch
from tokenizers import Tokenizer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.mathtext import MathTextParser

try:
	import latex2mathml.converter as latex2mathml_converter
except ImportError:
	latex2mathml_converter = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import FormulaDataset
from src.models.latex_ocr_model import LatexOCRModel
from src.models.text_decoder import convert_legacy_attnres_state_dict


# 当前 matplotlib 不支持 bitmap 输出，使用 agg 栅格输出执行像素比对。
mathtext_parser = MathTextParser("agg")


def robust_normalize_tex(tex: str) -> str:
	"""第一重：基础语义清理，抹除无意义的不可见字符。"""
	if tex is None:
		return ""

	tex = str(tex).strip()
	tex = re.sub(r"\\[,;!]", "", tex)
	tex = re.sub(r"\\quad", "", tex)
	tex = re.sub(r"\s+", "", tex)

	# 统一视觉等价宏，使用边界约束避免 \leq -> \leqq 这类误替换。
	tex = re.sub(r"\\le(?![A-Za-z])", r"\\leq", tex)
	tex = re.sub(r"\\ge(?![A-Za-z])", r"\\geq", tex)
	tex = re.sub(r"\\ne(?![A-Za-z])", r"\\neq", tex)
	tex = re.sub(r"\\rm(?![A-Za-z])", r"\\mathrm", tex)
	tex = re.sub(r"\\text(?![A-Za-z])", r"\\mathrm", tex)
	tex = re.sub(r"\\operatorname(?![A-Za-z])", r"\\mathrm", tex)

	tex = tex.replace(r"\left", "").replace(r"\right", "")
	return tex


def render_and_crop(tex: str) -> Optional[np.ndarray]:
	"""将 LaTeX 渲染为二值化图像并紧密裁剪，消除边缘空白。"""
	try:
		if not tex.startswith("$"):
			tex = f"${tex}$"

		raster = mathtext_parser.parse(tex, dpi=120)
		alpha = np.array(raster.image, dtype=np.uint8)

		binary_mask = (alpha > 50).astype(np.uint8)
		rows = np.any(binary_mask, axis=1)
		cols = np.any(binary_mask, axis=0)

		if not np.any(rows) or not np.any(cols):
			return None

		rmin, rmax = np.where(rows)[0][[0, -1]]
		cmin, cmax = np.where(cols)[0][[0, -1]]
		return binary_mask[rmin:rmax + 1, cmin:cmax + 1]
	except Exception:
		return None


def check_visual_equivalence(gt_tex: str, pred_tex: str) -> bool:
	"""终极视觉一致性检查核心逻辑。"""
	if not gt_tex or not pred_tex:
		return gt_tex == pred_tex

	norm_gt = robust_normalize_tex(gt_tex)
	norm_pred = robust_normalize_tex(pred_tex)
	if norm_gt == norm_pred:
		return True

	if latex2mathml_converter is not None:
		try:
			gt_mml = latex2mathml_converter.convert(norm_gt)
			pred_mml = latex2mathml_converter.convert(norm_pred)
			if gt_mml == pred_mml:
				return True
		except Exception:
			pass

	img_gt = render_and_crop(norm_gt)
	img_pred = render_and_crop(norm_pred)

	if img_gt is None or img_pred is None:
		return False

	if abs(img_gt.shape[0] - img_pred.shape[0]) > 2 or abs(img_gt.shape[1] - img_pred.shape[1]) > 2:
		return False

	min_h = min(img_gt.shape[0], img_pred.shape[0])
	min_w = min(img_gt.shape[1], img_pred.shape[1])
	img_gt_aligned = img_gt[:min_h, :min_w]
	img_pred_aligned = img_pred[:min_h, :min_w]

	diff_pixels = np.sum(img_gt_aligned != img_pred_aligned)
	error_rate = diff_pixels / (min_h * min_w)
	return error_rate < 0.02


SPACED_COMMAND_PATTERN = re.compile(r"\\[A-Za-z](?:\s+[A-Za-z]+)+")
MULTISPACE_PATTERN = re.compile(r"\s+")
LEFT_NON_ALPHA_PATTERN = re.compile(r"\\left(?![a-zA-Z])")
RIGHT_NON_ALPHA_PATTERN = re.compile(r"\\right(?![a-zA-Z])")
STYLE_COMMAND_PATTERN = re.compile(r"\\(?:textstyle|displaystyle|scriptstyle|scriptscriptstyle)\b")
SIMPLE_OVER_GROUP_PATTERN = re.compile(r"\{\s*([^{}]+?)\s*\\over\s*([^{}]+?)\s*\}")
TEXTLIKE_COMMAND_PATTERN = re.compile(r"\\(operatorname|mathrm|text|mbox|rm|textrm)\s*\{([^{}]*)\}")
REDUNDANT_COMMAND_GROUP_PATTERN = re.compile(
	r"\{\s*(\\(?:mathbf|mathit|mathrm|mathbb|mathcal|mathfrak|operatorname|text|mbox|rm|textrm)\s*\{[^{}]+\})\s*\}"
)

RENDER_ALIAS_MAP = {
	r"\neq": r"\ne",
	r"\leq": r"\le",
	r"\geq": r"\ge",
	r"\rightarrow": r"\to",
	r"\overrightarrow": r"\vec",
	r"\tfrac": r"\frac",
	r"\dfrac": r"\frac",
	r"\cfrac": r"\frac",
	r"\tbinom": r"\binom",
	r"\dbinom": r"\binom",
	r"\choose": r"\binom",
}

TEXTLIKE_CANONICAL_MAP = {
	"operatorname": "mathrm",
	"text": "mathrm",
	"mbox": "mathrm",
	"rm": "mathrm",
	"textrm": "mathrm",
	"mathrm": "mathrm",
}


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="AR 贪心解码评估脚本（渲染一致性）")
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
	parser.add_argument("--render_dpi", type=int, default=200, help="公式渲染 DPI")
	parser.add_argument("--render_iou_threshold", type=float, default=0.985, help="渲染 IoU 判定阈值")
	parser.add_argument("--render_overlap_threshold", type=float, default=0.985, help="渲染覆盖率判定阈值")
	parser.add_argument(
		"--enable_mathml_fallback",
		action="store_true",
		help="是否启用 MathML 语义兜底（默认关闭，仅按渲染一致判定）",
	)
	return parser.parse_args()


@dataclass
class RenderCompareResult:
	matched: bool
	sim: float
	mode: str


def _collapse_spaced_commands(latex: str) -> str:
	"""将被空格切开的命令重新拼接，例如 \\d b inom -> \\dbinom。"""
	def _repl(match: re.Match[str]) -> str:
		cmd = match.group(0)
		return "\\" + re.sub(r"\s+", "", cmd[1:])

	prev = None
	cur = latex
	while prev != cur:
		prev = cur
		cur = SPACED_COMMAND_PATTERN.sub(_repl, cur)
	return cur


def _normalize_simple_over_groups(latex: str) -> str:
	"""将简单的 {A \\over B} 形式规范化为 {\\frac{A}{B}}。"""
	prev = None
	cur = latex
	while prev != cur:
		prev = cur
		cur = SIMPLE_OVER_GROUP_PATTERN.sub(r"{ \\frac {\1}{\2} }", cur)
	return cur


def _strip_brace_wrapped_delims(latex: str) -> str:
	"""移除括号内包裹整体内容的冗余花括号，例如 ({ x }) -> (x)。"""
	delim_pairs = {"(": ")", "[": "]"}
	out: List[str] = []
	i = 0
	n = len(latex)

	while i < n:
		ch = latex[i]
		close_ch = delim_pairs.get(ch)
		if close_ch is None:
			out.append(ch)
			i += 1
			continue

		j = i + 1
		while j < n and latex[j].isspace():
			j += 1
		if j >= n or latex[j] != "{":
			out.append(ch)
			i += 1
			continue

		depth = 1
		k = j + 1
		while k < n and depth > 0:
			if latex[k] == "{":
				depth += 1
			elif latex[k] == "}":
				depth -= 1
			k += 1

		if depth != 0:
			out.append(ch)
			i += 1
			continue

		m = k
		while m < n and latex[m].isspace():
			m += 1
		if m >= n or latex[m] != close_ch:
			out.append(ch)
			i += 1
			continue

		inner = latex[j + 1:k - 1].strip()
		out.append(ch)
		out.append(inner)
		out.append(close_ch)
		i = m + 1

	return "".join(out)


def _compact_textlike_content(content: str) -> str:
	"""压缩文本样式命令中的冗余空白，减少字母被空格切开的差异。"""
	content = MULTISPACE_PATTERN.sub(" ", content).strip()
	if not content:
		return content

	# 仅在纯字母数字片段时拼接，避免误改复杂 LaTeX 结构。
	tokens = [tok for tok in content.split(" ") if tok]
	if tokens and all(re.fullmatch(r"[A-Za-z0-9]+", tok) for tok in tokens):
		return "".join(tokens)

	return content


def _normalize_textlike_commands(latex: str) -> str:
	"""统一 \\text/\\mbox/\\operatorname/\\rm 等文本样式命令到同一形式。"""
	def _repl(match: re.Match[str]) -> str:
		cmd = match.group(1)
		content = _compact_textlike_content(match.group(2))
		canonical_cmd = TEXTLIKE_CANONICAL_MAP.get(cmd, cmd)
		return f"\\{canonical_cmd} {{{content}}}"

	prev = None
	cur = latex
	while prev != cur:
		prev = cur
		cur = TEXTLIKE_COMMAND_PATTERN.sub(_repl, cur)
	return cur


def _strip_redundant_command_groups(latex: str) -> str:
	"""移除仅包裹单个样式命令的冗余花括号。"""
	prev = None
	cur = latex
	while prev != cur:
		prev = cur
		cur = REDUNDANT_COMMAND_GROUP_PATTERN.sub(r"\1", cur)
	return cur


def normalize_latex_for_visual_text(latex: str) -> str:
	"""用于视觉一致性预判的文本归一化。"""
	latex = normalize_latex_for_render(latex)
	latex = _normalize_textlike_commands(latex)
	latex = _strip_redundant_command_groups(latex)
	latex = MULTISPACE_PATTERN.sub(" ", latex)
	return latex.strip()


def normalize_latex_for_render(latex: str) -> str:
	"""渲染前仅做最小清洗，避免改变公式视觉语义。"""
	if not isinstance(latex, str):
		latex = "" if latex is None else str(latex)

	latex = latex.strip()
	if not latex:
		return ""

	if len(latex) >= 2 and latex[0] == "$" and latex[-1] == "$":
		latex = latex[1:-1].strip()

	# 仅折叠连续空白，避免文本比较因换行/制表符抖动。
	latex = MULTISPACE_PATTERN.sub(" ", latex)
	return latex


class FormulaRenderComparator:
	"""基于公式渲染结果的等价判定器。"""

	def __init__(
		self,
		dpi: int,
		iou_threshold: float,
		overlap_threshold: float,
		enable_mathml_fallback: bool = False,
	):
		self.dpi = int(dpi)
		self.iou_threshold = float(iou_threshold)
		self.overlap_threshold = float(overlap_threshold)
		self.enable_mathml_fallback = bool(enable_mathml_fallback)
		self.parser = MathTextParser("agg")
		self.render_cache: Dict[str, Optional[np.ndarray]] = {}
		self.mathml_cache: Dict[str, Optional[str]] = {}

	def _render_mask_from_key(self, key: str) -> Optional[np.ndarray]:
		if key in self.render_cache:
			return self.render_cache[key]

		try:
			raster = self.parser.parse(key, dpi=self.dpi)
			img = np.array(raster.image, dtype=np.uint8)
			mask = img > 0

			if mask.any():
				ys, xs = np.where(mask)
				mask = mask[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1]
			else:
				mask = np.zeros((1, 1), dtype=bool)

			self.render_cache[key] = mask
		except Exception:
			self.render_cache[key] = None

		return self.render_cache[key]

	def _to_mathml(self, latex: str) -> Optional[str]:
		if latex in self.mathml_cache:
			return self.mathml_cache[latex]

		if latex2mathml_converter is None:
			self.mathml_cache[latex] = None
			return None

		try:
			mathml = latex2mathml_converter.convert(latex)
			mathml = MULTISPACE_PATTERN.sub("", mathml)
			self.mathml_cache[latex] = mathml
		except Exception:
			self.mathml_cache[latex] = None

		return self.mathml_cache[latex]

	@staticmethod
	def _center_pad(mask: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
		canvas = np.zeros((target_h, target_w), dtype=bool)
		y = (target_h - mask.shape[0]) // 2
		x = (target_w - mask.shape[1]) // 2
		canvas[y: y + mask.shape[0], x: x + mask.shape[1]] = mask
		return canvas

	@staticmethod
	def _render_stats(mask_a: np.ndarray, mask_b: np.ndarray) -> Tuple[float, float, float]:
		inter = float(np.logical_and(mask_a, mask_b).sum())
		union = float(np.logical_or(mask_a, mask_b).sum())

		if union <= 0.0:
			return 1.0, 1.0, 1.0

		a_sum = float(mask_a.sum())
		b_sum = float(mask_b.sum())
		coverage_a = inter / max(1.0, a_sum)
		coverage_b = inter / max(1.0, b_sum)
		overlap = min(coverage_a, coverage_b)
		iou = inter / union
		sim = 0.5 * (iou + overlap)
		return iou, overlap, sim

	@staticmethod
	def _loose_key(latex: str) -> str:
		"""生成空白不敏感键，用于判定视觉上等价的纯排版差异。"""
		return MULTISPACE_PATTERN.sub("", latex)

	def compare(self, pred_latex: str, target_latex: str) -> RenderCompareResult:
		pred_key = normalize_latex_for_render(pred_latex)
		tgt_key = normalize_latex_for_render(target_latex)

		pred_mask = self._render_mask_from_key(pred_key)
		tgt_mask = self._render_mask_from_key(tgt_key)

		if pred_mask is not None and tgt_mask is not None:
			target_h = max(pred_mask.shape[0], tgt_mask.shape[0])
			target_w = max(pred_mask.shape[1], tgt_mask.shape[1])
			pred_canvas = self._center_pad(pred_mask, target_h, target_w)
			tgt_canvas = self._center_pad(tgt_mask, target_h, target_w)

			iou, overlap, sim = self._render_stats(pred_canvas, tgt_canvas)
			matched = (iou >= self.iou_threshold) and (overlap >= self.overlap_threshold)
			return RenderCompareResult(matched=matched, sim=sim, mode="render")

		# 严格模式下仅允许“完全同字符串”在渲染失败时通过，避免误判放宽。
		if pred_key == tgt_key:
			return RenderCompareResult(matched=True, sim=1.0, mode="fallback_text_exact")

		if self.enable_mathml_fallback:
			pred_mathml = self._to_mathml(pred_key)
			tgt_mathml = self._to_mathml(tgt_key)
			if pred_mathml is not None and tgt_mathml is not None:
				is_same = pred_mathml == tgt_mathml
				return RenderCompareResult(matched=is_same, sim=1.0 if is_same else 0.0, mode="mathml")

		return RenderCompareResult(matched=False, sim=0.0, mode="fallback_fail")


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
	text_exact_match = 0
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

			is_correct = check_visual_equivalence(target_text, pred_text)
			text_exact_match += int(pred_text == target_text)

			dist = levenshtein_distance(pred_text, target_text)
			ned = dist / max(1, len(target_text))

			exact_match += int(is_correct)
			total_ed += dist
			total_ned += ned

			if not is_correct:
				key = f"GT={target_text} || PRED={pred_text}"
				mismatch_counter[key] += 1
				report_rows.append(
					f"[idx={processed}] VisualMatch=False, "
					f"TextExact={pred_text == target_text}, ED={dist}, NED={ned:.4f}\n"
					f"GT  : {target_text}\nPRED: {pred_text}\n"
				)

			processed += 1

		if processed > 0 and (processed % max(1, args.log_interval) == 0):
			em = exact_match / processed
			avg_ned = total_ned / processed
			iterator.set_postfix({"RenderEM": f"{em * 100:.2f}%", "NED": f"{avg_ned:.4f}"})

	em_rate = exact_match / processed
	text_em_rate = text_exact_match / processed
	avg_ed = total_ed / processed
	avg_ned = total_ned / processed

	os.makedirs(args.report_dir, exist_ok=True)
	ts = datetime.now().strftime("%Y%m%d_%H%M%S")
	report_path = os.path.join(args.report_dir, f"ar_eval_report_{ts}.txt")
	top_error_path = os.path.join(args.report_dir, f"ar_eval_top_errors_{ts}.txt")

	with open(report_path, "w", encoding="utf-8") as f:
		top_errors = mismatch_counter.most_common(20)
		f.write("=== AR Eval Report (Visual Equivalence) ===\n")
		f.write(f"checkpoint: {args.checkpoint}\n")
		f.write(f"eval_h5: {args.eval_h5}\n")
		f.write("visual_rule: normalize -> mathml -> pixel\n")
		f.write(f"EvalProcessed: {processed}\n")
		f.write(f"EvalRenderExact: {exact_match}\n")
		f.write(f"EvalRenderEM(%): {em_rate * 100:.4f}\n")
		f.write(f"EvalTextExact: {text_exact_match}\n")
		f.write(f"EvalTextEM(%): {text_em_rate * 100:.4f}\n")
		f.write(f"EvalNED: {avg_ned:.6f}\n")
		f.write(f"AvgED: {avg_ed:.4f}\n\n")

		f.write("=== Top 20 Errors (same format as train) ===\n")
		for err, cnt in top_errors:
			f.write(f"[{cnt:5d}] {err}\n")

		f.write("\n=== Detailed Mismatches (Top 200) ===\n")
		for row in report_rows[:200]:
			f.write(row + "\n")

	with open(top_error_path, "w", encoding="utf-8") as ef:
		for err, cnt in top_errors:
			ef.write(f"[{cnt:5d}] {err}\n")

	print("\n" + "=" * 80)
	print("评估完成")
	print(f"EvalProcessed : {processed}")
	print(f"EvalRenderExact : {exact_match}")
	print(f"EvalRenderEM(%) : {em_rate * 100:.4f}")
	print(f"EvalTextExact   : {text_exact_match}")
	print(f"EvalTextEM(%)   : {text_em_rate * 100:.4f}")
	print("VisualRule      : normalize -> mathml -> pixel")
	print(f"EvalNED       : {avg_ned:.6f}")
	print(f"AvgED         : {avg_ed:.4f}")
	print(f"Report        : {report_path}")
	print(f"TopErrors     : {top_error_path}")
	print("=" * 80)

	return em_rate, avg_ned, processed


if __name__ == "__main__":
	evaluate(parse_args())
