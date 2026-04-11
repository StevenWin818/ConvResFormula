"""AR 贪心解码评估脚本入口。"""
import argparse
import json
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple, cast

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


SVG_ID_ATTRS = {"id", "aria-labelledby", "aria-describedby", "focusable"}
SVG_META_ATTR_PREFIXES = ("data-",)
SVG_REF_ATTRS = {
	"href",
	"xlink:href",
	"clip-path",
	"mask",
	"filter",
	"marker-start",
	"marker-mid",
	"marker-end",
}
SVG_URL_REF_PATTERN = re.compile(r"url\(\s*#([^\)]+)\s*\)")
SVG_ONLY_HASH_REF_PATTERN = re.compile(r"^#.+$")


def robust_normalize_tex(tex: str) -> str:
	"""增强版归一化：应对排版宏和冗余括号造成的视觉假阳性。"""
	if tex is None:
		return ""

	tex = str(tex).strip()
	
	# 1. 优先剔除所有空白字符，方便后续无缝模式匹配
	tex = re.sub(r"\s+", "", tex)

	# 2. 剔除纯风格与排版控制宏
	for macro in [r"\textstyle", r"\displaystyle", r"\scriptstyle", r"\scriptscriptstyle", r"\limits", r"\nolimits"]:
		tex = tex.replace(macro, "")

	# 3. 剔除 \left 和 \right
	tex = tex.replace(r"\left", "").replace(r"\right", "")

	# 4. 剥离无意义的安全括号包裹 (处理类似 \gcd({S_{i}}) -> \gcd(S_{i}) 的问题)
	# 循环执行 3 次以应对可能的嵌套包裹
	for _ in range(3):
		tex = tex.replace("({", "(").replace("})", ")")
		tex = tex.replace("[{", "[").replace("}]", "]")
		tex = tex.replace("\\{{", "\\{").replace("}\\}", "\\}")
		tex = tex.replace("|{", "|").replace("}|", "|")
		tex = tex.replace("_{}", "")  # 消除模型误生成的空下标
		tex = tex.replace("^{}", "")  # 消除模型误生成的空上标

	# 5. 剔除不可见的间距控制宏
	tex = re.sub(r"\\[,;!]", "", tex)
	tex = re.sub(r"\\[qQ]uad", "", tex)
	tex = re.sub(r"\\ ", "", tex)

	# 6. 统一常见视觉等价宏
	tex = re.sub(r"\\le(?![A-Za-z])", r"\\leq", tex)
	tex = re.sub(r"\\ge(?![A-Za-z])", r"\\geq", tex)
	tex = re.sub(r"\\ne(?![A-Za-z])", r"\\neq", tex)
	tex = re.sub(r"\\rm(?![A-Za-z])", r"\\mathrm", tex)
	tex = re.sub(r"\\text(?![A-Za-z])", r"\\mathrm", tex)
	tex = re.sub(r"\\operatorname(?![A-Za-z])", r"\\mathrm", tex)
	tex = re.sub(r"\{([^{}\\]+)\\over([^{}\\]+)\}", r"\\frac{\1}{\2}", tex)

	return tex

def _strip_xml_ns(tag: str) -> str:
	if "}" in tag:
		return tag.split("}", 1)[1]
	return tag


def _normalize_svg_attr_value(attr_name: str, value: str) -> str:
	if not isinstance(value, str):
		value = "" if value is None else str(value)

	value = SVG_URL_REF_PATTERN.sub("url(#REF)", value)
	if attr_name in SVG_REF_ATTRS and SVG_ONLY_HASH_REF_PATTERN.match(value):
		value = "#REF"

	value = re.sub(r"\s+", " ", value).strip()
	return value


def _serialize_svg_dom(elem: ET.Element) -> str:
	tag = _strip_xml_ns(elem.tag)

	attrs: List[Tuple[str, str]] = []
	for raw_key, raw_val in elem.attrib.items():
		key = _strip_xml_ns(raw_key)
		key_lower = key.lower()
		if key_lower in SVG_ID_ATTRS:
			continue
		if any(key_lower.startswith(prefix) for prefix in SVG_META_ATTR_PREFIXES):
			continue
		norm_val = _normalize_svg_attr_value(key_lower, raw_val)
		attrs.append((key, norm_val))

	attrs.sort(key=lambda x: x[0])
	attr_text = "".join(f' {k}="{v}"' for k, v in attrs)
	pieces = [f"<{tag}{attr_text}>"]

	if elem.text and elem.text.strip():
		pieces.append(re.sub(r"\s+", " ", elem.text).strip())

	for child in list(elem):
		pieces.append(_serialize_svg_dom(child))
		if child.tail and child.tail.strip():
			pieces.append(re.sub(r"\s+", " ", child.tail).strip())

	pieces.append(f"</{tag}>")
	return "".join(pieces)


def canonicalize_svg(svg_text: str) -> Optional[str]:
	"""
	移除随机属性、截断亚像素浮点数误差，
	并将字形 ID 进行确定性映射的 SVG DOM 规范化引擎。
	"""
	if not svg_text:
		return None

	try:
		root = ET.fromstring(svg_text)
	except ET.ParseError:
		return None

	# 安全的 ID 映射表，避免将 A 和 B 的引用全混成 #REF
	id_map = {}
	id_counter = [0]

	def _get_mapped_id(raw_id: str) -> str:
		base_id = raw_id.lstrip('#')
		if base_id not in id_map:
			id_counter[0] += 1
			id_map[base_id] = f"REF_{id_counter[0]}"
		return f"#{id_map[base_id]}"

	def _strip_xml_ns(tag: str) -> str:
		return tag.split("}", 1)[1] if "}" in tag else tag

	def _serialize_elem(elem: ET.Element) -> str:
		tag = _strip_xml_ns(elem.tag)
		attrs: List[Tuple[str, str]] = []
		
		for raw_key, raw_val in elem.attrib.items():
			key = _strip_xml_ns(raw_key)
			key_lower = key.lower()
			val = raw_val
			
			# 忽略与视觉渲染完全无关的语义类/无障碍类标签
			if key_lower in {"aria-labelledby", "aria-describedby", "focusable", "class"}:
				continue
			if any(key_lower.startswith(p) for p in ("data-",)):
				continue
				
			# 安全映射唯一的元素 ID
			if key_lower == "id":
				val = _get_mapped_id(val)[1:]
			
			# 映射相关的图形引用 href 和 url(#id)
			match = re.search(r"url\(\s*#([^\)]+)\s*\)", val)
			if match:
				val = re.sub(r"url\(\s*#([^\)]+)\s*\)", f"url({_get_mapped_id(match.group(1))})", val)
			if key_lower in {"href", "xlink:href", "clip-path", "mask"} and val.startswith('#'):
				val = _get_mapped_id(val)
			
			val = re.sub(r"\s+", " ", val).strip()
			
			# 【关键】截断浮点数坐标到3位小数，防止微弱排版偏移导致的误判
			val = re.sub(r"-?\d+\.\d{3,}", lambda m: f"{float(m.group(0)):.3f}", val)
			
			attrs.append((key, val))

		attrs.sort(key=lambda x: x[0])
		attr_text = "".join(f' {k}="{v}"' for k, v in attrs)
		pieces = [f"<{tag}{attr_text}>"]

		if elem.text and elem.text.strip():
			pieces.append(re.sub(r"\s+", " ", elem.text).strip())

		for child in list(elem):
			pieces.append(_serialize_elem(child))
			if child.tail and child.tail.strip():
				pieces.append(re.sub(r"\s+", " ", child.tail).strip())

		pieces.append(f"</{tag}>")
		return "".join(pieces)

	return _serialize_elem(root)


def svg_path_signature(svg_text: str) -> Optional[Tuple[Tuple[str, Tuple[Tuple[str, str], ...], str], ...]]:
	"""提取 SVG 可视元素签名，用于路径级比较。"""
	if not svg_text:
		return None

	try:
		root = ET.fromstring(svg_text)
	except ET.ParseError:
		return None

	tokens: List[Tuple[str, Tuple[Tuple[str, str], ...], str]] = []
	visual_tags = {
		"path",
		"use",
		"rect",
		"circle",
		"ellipse",
		"line",
		"polyline",
		"polygon",
		"text",
		"g",
	}

	for elem in root.iter():
		tag = _strip_xml_ns(elem.tag)
		if tag not in visual_tags:
			continue

		attrs: List[Tuple[str, str]] = []
		for raw_key, raw_val in elem.attrib.items():
			key = _strip_xml_ns(raw_key)
			key_lower = key.lower()
			if key_lower in SVG_ID_ATTRS:
				continue
			if any(key_lower.startswith(prefix) for prefix in SVG_META_ATTR_PREFIXES):
				continue
			attrs.append((key, _normalize_svg_attr_value(key_lower, raw_val)))

		attrs.sort(key=lambda x: x[0])
		text = re.sub(r"\s+", " ", elem.text or "").strip()
		tokens.append((tag, tuple(attrs), text))

	tokens.sort(key=lambda item: (item[0], item[1], item[2]))
	return tuple(tokens)


def compare_svg_dom_and_paths(svg_a: str, svg_b: str) -> bool:
	"""
	精简后的比对函数。
	由于 canonicalize_svg 已经非常健壮地处理了 ID 映射、浮点截断和无效标签剥离，
	直接做文本哈希比对就具备极高的准确性和安全性。
	"""
	canon_a = canonicalize_svg(svg_a)
	canon_b = canonicalize_svg(svg_b)
	
	if canon_a is None or canon_b is None:
		return False
		
	return canon_a == canon_b

class NodeKatexSvgRenderer:
	"""通过 Node.js 渲染 LaTeX 为 SVG，并复用长期子进程。"""

	def __init__(self, node_script: str):
		self.node_script = node_script
		self.process: Optional[subprocess.Popen[str]] = None
		self.cache: Dict[str, Optional[str]] = {}

	def _candidate_commands(self) -> List[List[str]]:
		return [["node", self.node_script]]

	def _start_process(self) -> bool:
		if self.process is not None and self.process.poll() is None:
			return True

		self.close()

		for cmd in self._candidate_commands():
			try:
				proc = subprocess.Popen(
					cmd,
					stdin=subprocess.PIPE,
					stdout=subprocess.PIPE,
					stderr=subprocess.DEVNULL,
					text=True,
					encoding="utf-8",
					bufsize=1,
				)
				if proc.stdin is None or proc.stdout is None:
					proc.kill()
					continue

				proc.stdin.write('{"ping": true}\n')
				proc.stdin.flush()
				line = proc.stdout.readline().strip()
				if not line:
					proc.kill()
					continue

				resp = json.loads(line)
				if resp.get("pong") is True:
					self.process = proc
					return True

				proc.kill()
			except Exception:
				continue

		return False

	def render(self, tex: str) -> Optional[str]:
		key = tex if isinstance(tex, str) else str(tex)
		if key in self.cache:
			return self.cache[key]

		if not self._start_process() or self.process is None:
			self.cache[key] = None
			return None

		assert self.process.stdin is not None
		assert self.process.stdout is not None
		try:
			payload = json.dumps({"tex": key}, ensure_ascii=False)
			self.process.stdin.write(payload + "\n")
			self.process.stdin.flush()
			line = self.process.stdout.readline().strip()
			if not line:
				self.close()
				self.cache[key] = None
				return None

			resp = json.loads(line)
			svg = resp.get("svg") if resp.get("ok") else None
			self.cache[key] = svg if isinstance(svg, str) and svg.strip() else None
			return self.cache[key]
		except Exception:
			self.close()
			self.cache[key] = None
			return None

	def close(self) -> None:
		if self.process is None:
			return

		try:
			if self.process.stdin is not None:
				self.process.stdin.close()
		except Exception:
			pass

		try:
			if self.process.poll() is None:
				self.process.terminate()
				self.process.wait(timeout=1.0)
		except Exception:
			try:
				self.process.kill()
			except Exception:
				pass

		self.process = None


def check_visual_equivalence(gt_tex: str, pred_tex: str, svg_renderer: NodeKatexSvgRenderer) -> bool:
	"""使用 KaTeX/Node 产出的 SVG 做 DOM 与路径双重比对。"""
	if not gt_tex or not pred_tex:
		return gt_tex == pred_tex

	norm_gt = robust_normalize_tex(gt_tex)
	norm_pred = robust_normalize_tex(pred_tex)
	if norm_gt == norm_pred:
		return True

	svg_gt = svg_renderer.render(norm_gt)
	svg_pred = svg_renderer.render(norm_pred)
	if svg_gt is None or svg_pred is None:
		return False

	return compare_svg_dom_and_paths(svg_gt, svg_pred)

def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="AR 贪心解码评估脚本（SVG DOM/Path 一致性）")
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
	parser.add_argument(
		"--svg_node_script",
		type=str,
		default=str(PROJECT_ROOT / "tools" / "debug" / "katex_svg_renderer.js"),
		help="Node SVG 渲染脚本路径",
	)
	parser.add_argument(
		"--svg_prefer_npx",
		action="store_true",
		help="兼容旧参数，当前版本已停用该开关（始终使用本地 Node 依赖）",
	)
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
	svg_node_script = args.svg_node_script
	if not os.path.isabs(svg_node_script):
		svg_node_script = str((PROJECT_ROOT / svg_node_script).resolve())
	if not os.path.exists(svg_node_script):
		raise FileNotFoundError(f"找不到 SVG 渲染脚本: {svg_node_script}")

	svg_renderer = NodeKatexSvgRenderer(
		node_script=svg_node_script,
	)
	if not svg_renderer._start_process():
		raise RuntimeError(
			"无法启动 SVG 渲染子进程。请先在项目根目录执行 npm install katex mathjax-full。"
		)
	try:
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
		tf_model = cast(LatexOCRModel, getattr(model, "_orig_mod", model))
		amp_enabled = device.type == "cuda"

		max_samples = len(dataset) if args.max_samples <= 0 else min(args.max_samples, len(dataset))
		if max_samples <= 0:
			raise RuntimeError("评估集为空，无法评估")

		processed = 0
		exact_match = 0
		text_exact_match = 0
		total_ed = 0
		total_ned = 0.0
		total_token_correct = 0
		total_token_count = 0
		tf_token_correct = 0
		tf_token_count = 0
		mismatch_counter: Counter[str] = Counter()
		report_rows: List[str] = []

		iterator = tqdm(eval_loader, desc="Eval")
		for batch in iterator:
			if processed >= max_samples:
				break

			images = batch["images"].to(device)
			target_ids = batch["target_ids"]

			# 训练口径 TokenAcc（Teacher-Forcing）：GT 前缀喂入，比较 next-token。
			tf_inputs = target_ids[:, :-1].to(device)
			tf_targets = target_ids[:, 1:].to(device)
			with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
				tf_logits = tf_model(images=images, tgt_seq=tf_inputs, is_causal=True)
			tf_preds = tf_logits.argmax(dim=-1)
			tf_valid_mask = tf_targets != pad_id
			if tf_valid_mask.any():
				tf_token_correct += int((tf_preds[tf_valid_mask] == tf_targets[tf_valid_mask]).sum().item())
				tf_token_count += int(tf_valid_mask.sum().item())

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

				# 评估期 TokenAcc：与训练口径对齐，比较 next-token 序列（去 BOS，保留 EOS）。
				target_row = target_ids[i].tolist()
				if pad_id in target_row:
					target_row = target_row[: target_row.index(pad_id)]
				target_eval_tokens = target_row[1:] if target_row and target_row[0] == bos_id else target_row

				pred_row = list(pred_batch[i])
				if pred_row and pred_row[0] == bos_id:
					pred_row = pred_row[1:]
				if eos_id in pred_row:
					pred_eval_tokens = pred_row[: pred_row.index(eos_id) + 1]
				else:
					pred_eval_tokens = pred_row + [eos_id]
				if target_eval_tokens:
					sample_total = len(target_eval_tokens)
					sample_correct = 0
					for pos, tgt_tok in enumerate(target_eval_tokens):
						if pos < len(pred_eval_tokens) and pred_eval_tokens[pos] == tgt_tok:
							sample_correct += 1
					total_token_correct += sample_correct
					total_token_count += sample_total

				is_correct = check_visual_equivalence(target_text, pred_text, svg_renderer)
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
		token_acc_rate = (total_token_correct / total_token_count) if total_token_count > 0 else 0.0
		tf_token_acc_rate = (tf_token_correct / tf_token_count) if tf_token_count > 0 else 0.0
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
			f.write("visual_rule: normalize -> svg_dom -> path_signature\n")
			f.write(f"svg_node_script: {svg_node_script}\n")
			f.write(f"EvalProcessed: {processed}\n")
			f.write(f"EvalRenderExact: {exact_match}\n")
			f.write(f"EvalRenderEM(%): {em_rate * 100:.4f}\n")
			f.write(f"EvalTextExact: {text_exact_match}\n")
			f.write(f"EvalTextEM(%): {text_em_rate * 100:.4f}\n")
			f.write(f"EvalTokenAcc(%): {token_acc_rate * 100:.4f}\n")
			f.write(f"EvalTFTokenAcc(%): {tf_token_acc_rate * 100:.4f}\n")
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
		print(f"EvalTokenAcc(%) : {token_acc_rate * 100:.4f}")
		print(f"EvalTFTokenAcc(%): {tf_token_acc_rate * 100:.4f}")
		print("VisualRule      : normalize -> svg_dom -> path_signature")
		print(f"SVGNodeScript   : {svg_node_script}")
		print(f"EvalNED       : {avg_ned:.6f}") 
		print(f"AvgED         : {avg_ed:.4f}")
		print(f"Report        : {report_path}")
		print(f"TopErrors     : {top_error_path}")
		print("=" * 80)

		return em_rate, avg_ned, processed
	finally:
		svg_renderer.close()


if __name__ == "__main__":
	evaluate(parse_args())
