"""ConvNeXt-V2 + AttnRes 模型 ONNX 导出脚本入口。"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict

import onnx
import torch
import yaml
from tokenizers import Tokenizer

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from src.models.latex_ocr_model import LatexOCRModel
from src.models.text_decoder import convert_legacy_attnres_state_dict


def _cfg_get(cfg: Dict[str, Any], path: tuple[str, ...], default: Any = None) -> Any:
	cur: Any = cfg
	for key in path:
		if not isinstance(cur, dict) or key not in cur:
			return default
		cur = cur[key]
	return cur


def load_yaml(path: str) -> Dict[str, Any]:
	with open(path, "r", encoding="utf-8") as f:
		data = yaml.safe_load(f)
	return data if isinstance(data, dict) else {}


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="导出 LatexOCRModel 为 ONNX（默认排除 CTC 辅助分支）")
	parser.add_argument(
		"--checkpoint",
		type=str,
		default=str(PROJECT_ROOT / "checkpoints" / "ar" / "best_loss.pth"),
		help="训练 checkpoint 路径 (.pth)",
	)
	parser.add_argument(
		"--tokenizer",
		type=str,
		default=str(PROJECT_ROOT / "tokenizer_bpe.json"),
		help="BPE tokenizer json 路径",
	)
	parser.add_argument(
		"--model-config",
		type=str,
		default=str(PROJECT_ROOT / "configs" / "model_convnext_attnres.yaml"),
		help="模型配置 yaml 路径",
	)
	parser.add_argument(
		"--output",
		type=str,
		default=str(PROJECT_ROOT / "checkpoints" / "ar" / "latex_ocr.onnx"),
		help="导出 onnx 文件路径",
	)
	parser.add_argument("--opset", type=int, default=17, help="ONNX opset 版本")
	parser.add_argument("--height", type=int, default=128, help="导出用 dummy 图像高度")
	parser.add_argument("--width", type=int, default=384, help="导出用 dummy 图像宽度")
	parser.add_argument("--seq-len", type=int, default=64, help="导出用 dummy token 长度")
	return parser.parse_args()


def build_model(args: argparse.Namespace) -> LatexOCRModel:
	if not os.path.exists(args.tokenizer):
		raise FileNotFoundError(f"Tokenizer 不存在: {args.tokenizer}")
	if not os.path.exists(args.model_config):
		raise FileNotFoundError(f"模型配置不存在: {args.model_config}")

	tokenizer = Tokenizer.from_file(args.tokenizer)
	vocab_size = tokenizer.get_vocab_size()
	pad_id = tokenizer.token_to_id("[PAD]")
	eos_id = tokenizer.token_to_id("[EOS]")
	if pad_id is None:
		raise RuntimeError("Tokenizer 缺少 [PAD]")

	cfg = load_yaml(args.model_config)
	d_model = int(_cfg_get(cfg, ("model", "d_model"), 256))
	use_pe = bool(_cfg_get(cfg, ("model", "use_learned_position_embeddings"), True))
	max_pe = int(_cfg_get(cfg, ("model", "max_position_embeddings"), 2048))
	use_ckpt = bool(_cfg_get(cfg, ("model", "use_gradient_checkpointing"), False))
	ckpt_segments = int(_cfg_get(cfg, ("model", "checkpoint_segments"), 1))
	ckpt_dec = bool(_cfg_get(cfg, ("model", "checkpoint_decoder_layers"), False))

	vision_model_name = str(_cfg_get(cfg, ("vision_encoder", "model_name"), "convnextv2_nano"))
	vision_pretrained = bool(_cfg_get(cfg, ("vision_encoder", "pretrained"), True))
	vision_in_chans = int(_cfg_get(cfg, ("vision_encoder", "in_chans"), 1))
	vision_drop_path_rate = float(_cfg_get(cfg, ("vision_encoder", "drop_path_rate"), 0.0))

	decoder_nhead = int(_cfg_get(cfg, ("text_decoder", "nhead"), 8))
	decoder_num_layers = int(_cfg_get(cfg, ("text_decoder", "num_layers"), 6))
	decoder_dim_feedforward = int(_cfg_get(cfg, ("text_decoder", "dim_feedforward"), 2048))
	decoder_dropout = float(_cfg_get(cfg, ("text_decoder", "dropout"), 0.1))

	model = LatexOCRModel(
		vocab_size=vocab_size,
		d_model=d_model,
		pad_id=int(pad_id),
		eos_id=int(eos_id) if eos_id is not None else None,
		vision_model_name=vision_model_name,
		vision_pretrained=vision_pretrained,
		vision_in_chans=vision_in_chans,
		vision_drop_path_rate=vision_drop_path_rate,
		decoder_nhead=decoder_nhead,
		decoder_num_layers=decoder_num_layers,
		decoder_dim_feedforward=decoder_dim_feedforward,
		decoder_dropout=decoder_dropout,
		use_learned_position_embeddings=use_pe,
		max_position_embeddings=max_pe,
		use_gradient_checkpointing=use_ckpt,
		checkpoint_segments=ckpt_segments,
		checkpoint_decoder_layers=ckpt_dec,
	)
	return model


def load_checkpoint(model: LatexOCRModel, checkpoint_path: str, device: torch.device) -> None:
	if not os.path.exists(checkpoint_path):
		raise FileNotFoundError(f"Checkpoint 不存在: {checkpoint_path}")

	state = torch.load(checkpoint_path, map_location=device, weights_only=False)
	state_dict = state.get("model_raw", state.get("model", state))
	model_state_dict = convert_legacy_attnres_state_dict(state_dict)
	incompatible = model.load_state_dict(model_state_dict, strict=False)
	if incompatible.missing_keys:
		print(f"警告: 缺少 {len(incompatible.missing_keys)} 个参数（可能是新增头或结构差异）。")
	if incompatible.unexpected_keys:
		print(f"警告: 多出 {len(incompatible.unexpected_keys)} 个参数（来自旧结构时常见）。")


class OnnxForwardWrapper(torch.nn.Module):
	"""固定默认推理路径，确保 ONNX 图不触发 return_aux 分支。"""

	def __init__(self, model: LatexOCRModel):
		super().__init__()
		self.model = model

	def forward(self, images: torch.Tensor, tgt_seq: torch.Tensor) -> torch.Tensor:
		return self.model(images=images, tgt_seq=tgt_seq, is_causal=True)


def inspect_no_ctc_head(onnx_path: str) -> None:
	graph = onnx.load(onnx_path).graph
	names = [init.name for init in graph.initializer]
	has_ctc = any("ctc_head" in n for n in names)
	if has_ctc:
		raise RuntimeError("检测到 ctc_head 参数被导入 ONNX，默认推理路径兼容性被破坏。")
	print("✅ ONNX 图检查通过：未包含 ctc_head 参数。")


def main() -> None:
	args = parse_args()
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	model = build_model(args).to(device)
	model.eval()
	load_checkpoint(model, args.checkpoint, device)

	wrapper = OnnxForwardWrapper(model).to(device)
	wrapper.eval()

	os.makedirs(os.path.dirname(args.output), exist_ok=True)

	with torch.no_grad():
		dummy_images = torch.randn(1, 1, int(args.height), int(args.width), device=device)
		dummy_tgt = torch.zeros(1, int(args.seq_len), dtype=torch.long, device=device)
		torch.onnx.export(
			wrapper,
			(dummy_images, dummy_tgt),
			args.output,
			input_names=["images", "tgt_seq"],
			output_names=["logits"],
			dynamic_axes={
				"images": {0: "batch", 2: "height", 3: "width"},
				"tgt_seq": {0: "batch", 1: "seq_len"},
				"logits": {0: "batch", 1: "seq_len"},
			},
			opset_version=int(args.opset),
			dynamo=False,
		)

	inspect_no_ctc_head(args.output)
	print(f"✅ 导出完成: {args.output}")


if __name__ == "__main__":
	main()
