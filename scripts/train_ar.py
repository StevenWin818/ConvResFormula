"""AR 训练脚本入口。"""

import argparse
import os
import random
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, cast

import numpy as np
import torch
import torch.optim as optim
from tokenizers import Tokenizer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import ConcatDataset, DataLoader, Sampler
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from tqdm import tqdm
import logging
import torch._inductor.config


# 强制开启硬件级 TF32 核心计算
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True

try:
	import yaml
except ImportError as exc:
	raise RuntimeError("请先安装 PyYAML: pip install pyyaml") from exc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from src.data.collate import ARCollate
from src.data.dataset import FormulaDataset
from src.engine.lr_schedulers import build_linear_warmup_cosine_scheduler, infer_total_steps
from src.engine.trainer import ARTrainer
from src.models.latex_ocr_model import LatexOCRModel
from src.models.text_decoder import convert_legacy_attnres_state_dict

torch._inductor.config.fallback_random = True
logging.getLogger("torch._inductor").setLevel(logging.ERROR)


def parse_args() -> argparse.Namespace:
	bootstrap = argparse.ArgumentParser(add_help=False)
	bootstrap.add_argument("--train-config", type=str, default=str(PROJECT_ROOT / "configs/train_ar.yaml"))
	bootstrap_args, _ = bootstrap.parse_known_args()

	help_cfg: Dict[str, str] = {}
	if os.path.exists(bootstrap_args.train_config):
		cfg = load_yaml_config(bootstrap_args.train_config)
		raw_help = _cfg_get(cfg, ("cli_help",), {})
		if isinstance(raw_help, dict):
			help_cfg = {str(k): str(v) for k, v in raw_help.items()}

	def h(key: str, fallback: str) -> str:
		return help_cfg.get(key, fallback)

	parser = argparse.ArgumentParser(description=h("description", "ConvNeXt-V2 + AttnRes 的 AR 训练脚本"))

	parser.add_argument("--train-config", type=str, default=str(PROJECT_ROOT / "configs/train_ar.yaml"), help=h("train_config", "训练配置文件路径"))
	parser.add_argument("--model-config", type=str, default=str(PROJECT_ROOT / "configs/model_convnext_attnres.yaml"), help=h("model_config", "模型配置文件路径"))

	parser.add_argument("--train_h5", type=str, nargs="+", default=None, help=h("train_h5", "训练集 H5 列表，可传多个文件"))
	parser.add_argument("--tokenizer", type=str, default=None, help=h("tokenizer", "BPE Tokenizer JSON 路径"))
	parser.add_argument("--eval_h5", type=str, default=None, help=h("eval_h5", "评估集 H5 路径"))
	parser.add_argument("--max_area", type=int, default=None, help=h("max_area", "图像动态缩放最大面积"))

	parser.add_argument("--d_model", type=int, default=None, help=h("d_model", "模型隐藏维度"))
	parser.add_argument("--batch_size", type=int, default=None, help=h("batch_size", "训练批大小"))
	parser.add_argument("--num_workers", type=int, default=None, help=h("num_workers", "训练 DataLoader 线程数"))
	parser.add_argument("--epochs", type=int, default=None, help=h("epochs", "训练轮数"))
	parser.add_argument("--seed", type=int, default=None, help=h("seed", "随机种子"))
	parser.add_argument("--mix_ratio", type=int, nargs="+", default=None, help=h("mix_ratio", "多数据源采样比例，例如 1 1 5"))
	parser.add_argument("--epoch_samples", type=int, default=None, help=h("epoch_samples", "每个 epoch 总采样数，<=0 使用全量"))
	parser.add_argument("--bucket_mega_factor", type=int, default=None, help=h("bucket_mega_factor", "分桶 mega-batch 系数"))
	parser.add_argument("--enable_extreme_augment", dest="enable_extreme_augment", action="store_true", help="启用极端数据增强（形态学/网格扭曲等）")
	parser.add_argument("--disable_extreme_augment", dest="enable_extreme_augment", action="store_false", help="禁用极端数据增强（独立于标准增强）")
	parser.set_defaults(enable_extreme_augment=None)

	parser.add_argument("--encoder_lr", type=float, default=None, help=h("encoder_lr", "视觉编码器骨干学习率"))
	parser.add_argument("--fpn_lr", type=float, default=None, help=h("fpn_lr", "视觉FPN及新增视觉自注意力层学习率"))
	parser.add_argument("--decoder_lr", type=float, default=None, help=h("decoder_lr", "解码器学习率"))
	parser.add_argument("--weight_decay", type=float, default=None, help=h("weight_decay", "权重衰减系数"))
	parser.add_argument("--bow_weight", type=float, default=None, help="BoW 辅助损失权重")
	parser.add_argument("--ctc_weight", type=float, default=None, help="[兼容旧参数] 等价映射到 bow_weight")
	parser.add_argument("--ohem_ratio", type=float, default=None, help="OHEM 困难样本保留比例")
	
	# SIGReg-like 参数
	parser.add_argument("--enable_sigreg", dest="enable_sigreg", action="store_true", help="启用 SIGReg-like 正则化")
	parser.add_argument("--disable_sigreg", dest="enable_sigreg", action="store_false", help="禁用 SIGReg-like 正则化")
	parser.set_defaults(enable_sigreg=None)
	parser.add_argument("--sigreg_weight", type=float, default=None, help="SIGReg-like 损失权重")
	parser.add_argument("--sigreg_num_projections", type=int, default=None, help="SIGReg 随机投影次数")
	parser.add_argument("--sigreg_collapse_threshold", type=float, default=None, help="SIGReg embedding 坍塌检测阈值")
	parser.add_argument("--decoder_input_noise_ratio", type=float, default=None, help="解码器输入随机 Token 替换概率（缓解曝光偏差）")

	parser.add_argument("--warmup_epochs", type=float, default=None, help=h("warmup_epochs", "学习率 warmup 轮数"))
	parser.add_argument("--warmup_start_lr", type=float, default=None, help=h("warmup_start_lr", "warmup 起始学习率"))
	parser.add_argument("--eta_min", type=float, default=None, help=h("eta_min", "余弦退火最小学习率"))

	parser.add_argument("--log_interval", type=int, default=None, help=h("log_interval", "训练日志打印间隔(step)"))
	parser.add_argument("--eval_interval", type=int, default=None, help=h("eval_interval", "每隔多少个 epoch 评估一次"))
	parser.add_argument("--eval_batch_size", type=int, default=None, help=h("eval_batch_size", "评估批大小"))
	parser.add_argument("--eval_num_workers", type=int, default=None, help=h("eval_num_workers", "评估 DataLoader 线程数"))
	parser.add_argument("--eval_samples", type=int, default=None, help=h("eval_samples", "每轮评估抽样数量，<=0 表示全量"))
	parser.add_argument("--eval_max_len", type=int, default=None, help=h("eval_max_len", "评估解码最大长度"))
	parser.add_argument("--eval_amp_dtype", type=str, choices=["fp16", "bf16", "fp32"], default=None, help="评估精度类型: fp16 / bf16 / fp32")
	parser.add_argument("--eval_use_gradient_checkpointing", dest="eval_use_gradient_checkpointing", action="store_true", help="评估阶段启用视觉梯度检查点")
	parser.add_argument("--eval_no_gradient_checkpointing", dest="eval_use_gradient_checkpointing", action="store_false", help="评估阶段禁用视觉梯度检查点")
	parser.set_defaults(eval_use_gradient_checkpointing=None)
	parser.add_argument("--amp_dtype", type=str, choices=["fp16", "bf16", "fp32"], default=None, help=h("amp_dtype", "训练精度类型: fp16 / bf16 / fp32"))

	parser.add_argument("--use_gradient_checkpointing", dest="use_gradient_checkpointing", action="store_true", 
		help="启用梯度检查点以降低显存占用（会增加计算时间）")
	parser.add_argument("--no_gradient_checkpointing", dest="use_gradient_checkpointing", action="store_false", 
		help="禁用梯度检查点")
	parser.set_defaults(use_gradient_checkpointing=None)

	parser.add_argument("--checkpoint_segments", type=int, default=None, 
		help="对视觉编码器分成多少段应用检查点 (1=整体, 2=前后层, 4=四分段)")
	parser.add_argument("--checkpoint_decoder_layers", dest="checkpoint_decoder_layers", action="store_true",
		help="对文本解码器层应用梯度检查点")
	parser.add_argument("--no_checkpoint_decoder", dest="checkpoint_decoder_layers", action="store_false",
		help="禁用解码器层检查点")
	parser.set_defaults(checkpoint_decoder_layers=None)

	eval_group = parser.add_mutually_exclusive_group()
	eval_group.add_argument("--skip_eval", dest="skip_eval", action="store_true", help=h("skip_eval", "仅训练不做评估"))
	eval_group.add_argument("--enable_eval", dest="skip_eval", action="store_false", help=h("enable_eval", "启用训练期评估"))
	parser.set_defaults(skip_eval=None)

	parser.add_argument("--ckpt_dir", type=str, default=None, help=h("ckpt_dir", "checkpoint 输出目录"))
	parser.add_argument("--log_dir", type=str, default=None, help=h("log_dir", "日志输出目录"))
	parser.add_argument("--resume_ckpt", type=str, default=None, help=h("resume_ckpt", "恢复训练的 checkpoint 路径"))
	parser.add_argument("--channels_last", dest="channels_last", action="store_true", help="启用 channels_last 内存格式")
	parser.add_argument("--contiguous_format", dest="channels_last", action="store_false", help="禁用 channels_last，使用 contiguous 格式")
	parser.set_defaults(channels_last=None)

	resume_group = parser.add_mutually_exclusive_group()
	resume_group.add_argument("--resume", dest="resume", action="store_true", help=h("resume", "从 resume_ckpt 恢复完整训练状态"))
	resume_group.add_argument("--fresh_start", dest="resume", action="store_false", help=h("fresh_start", "不恢复断点，冷启动训练"))
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


def get_best_loss_from_log(log_path: str, max_epoch: int | None = None) -> float | None:
	"""从训练日志回溯历史最小 loss，兼容旧 checkpoint 的 best_loss 滞后问题。"""
	if not os.path.exists(log_path):
		return None

	best_loss: float | None = None
	with open(log_path, "r", encoding="utf-8") as f:
		_ = f.readline()  # skip header
		for line in f:
			parts = line.strip().split("\t")
			if len(parts) < 3:
				continue
			try:
				epoch = int(parts[1])
				loss = float(parts[2])
			except ValueError:
				continue

			if max_epoch is not None and epoch > max_epoch:
				continue
			if not np.isfinite(loss):
				continue

			if best_loss is None or loss < best_loss:
				best_loss = loss

	return best_loss


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

	cfg_mix_ratio = _cfg_get(train_cfg, ("training", "mix_ratio"), None)
	if args.mix_ratio is None:
		if cfg_mix_ratio is None:
			args.mix_ratio = [1] * len(args.train_h5)
		else:
			args.mix_ratio = [int(v) for v in cfg_mix_ratio]
	else:
		args.mix_ratio = [int(v) for v in args.mix_ratio]

	args.epoch_samples = args.epoch_samples if args.epoch_samples is not None else int(_cfg_get(train_cfg, ("training", "epoch_samples"), 0))
	args.bucket_mega_factor = args.bucket_mega_factor if args.bucket_mega_factor is not None else int(_cfg_get(train_cfg, ("training", "bucket_mega_factor"), 30))

	args.encoder_lr = args.encoder_lr if args.encoder_lr is not None else float(_cfg_get(train_cfg, ("optimization", "encoder_lr"), 5e-5))
	args.decoder_lr = args.decoder_lr if args.decoder_lr is not None else float(_cfg_get(train_cfg, ("optimization", "decoder_lr"), 1e-4))
	args.weight_decay = args.weight_decay if args.weight_decay is not None else float(_cfg_get(train_cfg, ("optimization", "weight_decay"), 1e-2))
	if args.bow_weight is not None:
		args.bow_weight = float(args.bow_weight)
	elif args.ctc_weight is not None:
		args.bow_weight = float(args.ctc_weight)
	else:
		args.bow_weight = float(_cfg_get(train_cfg, ("optimization", "bow_weight"), _cfg_get(train_cfg, ("optimization", "ctc_weight"), 0.5)))
	args.ohem_ratio = args.ohem_ratio if args.ohem_ratio is not None else float(_cfg_get(train_cfg, ("optimization", "ohem_ratio"), 0.5))
	
	# SIGReg-like 参数
	if args.enable_sigreg is None:
		args.enable_sigreg = bool(_cfg_get(train_cfg, ("optimization", "enable_sigreg"), False))
	args.sigreg_weight = args.sigreg_weight if args.sigreg_weight is not None else float(_cfg_get(train_cfg, ("optimization", "sigreg_weight"), 0.1))
	args.sigreg_num_projections = args.sigreg_num_projections if args.sigreg_num_projections is not None else int(_cfg_get(train_cfg, ("optimization", "sigreg_num_projections"), 1024))
	args.sigreg_collapse_threshold = args.sigreg_collapse_threshold if args.sigreg_collapse_threshold is not None else float(_cfg_get(train_cfg, ("optimization", "sigreg_collapse_threshold"), 0.01))

	args.decoder_input_noise_ratio = args.decoder_input_noise_ratio if args.decoder_input_noise_ratio is not None else float(_cfg_get(train_cfg, ("training", "decoder_input_noise_ratio"), 0.08))

	args.reset_lr = getattr(args, "reset_lr", None)
	if args.reset_lr is None:
		args.reset_lr = bool(_cfg_get(train_cfg, ("runtime", "reset_lr"), False))

	args.warmup_epochs = args.warmup_epochs if args.warmup_epochs is not None else float(_cfg_get(train_cfg, ("scheduler", "warmup_epochs"), 1.0))
	args.warmup_start_lr = args.warmup_start_lr if args.warmup_start_lr is not None else float(_cfg_get(train_cfg, ("scheduler", "warmup_start_lr"), 1e-7))
	args.eta_min = args.eta_min if args.eta_min is not None else float(_cfg_get(train_cfg, ("scheduler", "eta_min"), 1e-6))

	args.label_smoothing = float(_cfg_get(train_cfg, ("optimization", "label_smoothing"), 0.1))

	augment_cfg = _cfg_get(train_cfg, ("augmentation",), {})
	if not isinstance(augment_cfg, dict):
		augment_cfg = {}
	args.enable_augment = bool(augment_cfg.get("enable", True))
	args.augmentation_cfg = augment_cfg

	extreme_augment_cfg = _cfg_get(train_cfg, ("augmentation_extreme",), {})
	if not isinstance(extreme_augment_cfg, dict):
		extreme_augment_cfg = {}
	if args.enable_extreme_augment is None:
		args.enable_extreme_augment = bool(extreme_augment_cfg.get("enable", False))
	args.extreme_augmentation_cfg = extreme_augment_cfg

	args.log_interval = args.log_interval if args.log_interval is not None else int(_cfg_get(train_cfg, ("logging", "log_interval"), 20))
	args.eval_interval = args.eval_interval if args.eval_interval is not None else int(_cfg_get(train_cfg, ("eval", "eval_interval"), 1))
	args.eval_batch_size = args.eval_batch_size if args.eval_batch_size is not None else int(_cfg_get(train_cfg, ("eval", "batch_size"), 32))
	args.eval_num_workers = args.eval_num_workers if args.eval_num_workers is not None else int(_cfg_get(train_cfg, ("eval", "num_workers"), 2))
	args.eval_samples = args.eval_samples if args.eval_samples is not None else int(_cfg_get(train_cfg, ("eval", "samples"), 1500))
	args.eval_max_len = args.eval_max_len if args.eval_max_len is not None else int(_cfg_get(train_cfg, ("eval", "max_len"), 160))
	args.eval_amp_dtype = str(args.eval_amp_dtype or _cfg_get(train_cfg, ("eval", "amp_dtype"), "bf16")).lower()
	if args.eval_amp_dtype not in {"fp16", "bf16", "fp32"}:
		raise ValueError(f"eval.amp_dtype 必须是 fp16/bf16/fp32，当前为: {args.eval_amp_dtype}")
	if args.eval_use_gradient_checkpointing is None:
		args.eval_use_gradient_checkpointing = bool(_cfg_get(train_cfg, ("eval", "use_gradient_checkpointing"), True))
	if args.skip_eval is None:
		args.skip_eval = bool(_cfg_get(train_cfg, ("eval", "skip_eval"), False))

	args.amp_dtype = str(args.amp_dtype or _cfg_get(train_cfg, ("runtime", "amp_dtype"), "fp16")).lower()
	if args.amp_dtype not in {"fp16", "bf16", "fp32"}:
		raise ValueError(f"amp_dtype 必须是 fp16/bf16/fp32，当前为: {args.amp_dtype}")

	args.ckpt_dir = args.ckpt_dir or str(_cfg_get(train_cfg, ("runtime", "ckpt_dir"), r"checkpoints\ar"))
	args.log_dir = args.log_dir or str(_cfg_get(train_cfg, ("runtime", "log_dir"), r"logs\ar_logs"))
	args.resume_ckpt = args.resume_ckpt or str(_cfg_get(train_cfg, ("runtime", "resume_ckpt"), r"checkpoints\ar\latest.pth"))
	args.cudnn_benchmark = bool(_cfg_get(train_cfg, ("runtime", "cudnn_benchmark"), False))
	if args.channels_last is None:
		args.channels_last = bool(_cfg_get(train_cfg, ("runtime", "channels_last"), False))
	if args.use_gradient_checkpointing is None:
		args.use_gradient_checkpointing = bool(_cfg_get(model_cfg, ("model", "use_gradient_checkpointing"), False))
	args.checkpoint_segments = args.checkpoint_segments if args.checkpoint_segments is not None else int(_cfg_get(train_cfg, ("model", "checkpoint_segments"), 1))
	if args.checkpoint_decoder_layers is None:
		args.checkpoint_decoder_layers = bool(_cfg_get(train_cfg, ("model", "checkpoint_decoder_layers"), False))
	args.kernel_warmup_batches = int(_cfg_get(train_cfg, ("runtime", "kernel_warmup_batches"), 8))
    
	if args.kernel_warmup_batches < 0:
		raise ValueError(f"runtime.kernel_warmup_batches 不能小于 0，当前为 {args.kernel_warmup_batches}")

	if args.resume is None:
		args.resume = bool(_cfg_get(train_cfg, ("runtime", "resume"), False))

	if not args.tokenizer:
		raise RuntimeError("配置缺少 tokenizer 路径，请在 train_ar.yaml:data.tokenizer 中设置")
	if not args.train_h5:
		raise RuntimeError("配置缺少 train_h5 列表，请在 train_ar.yaml:data.train_h5 中设置")
	if (not args.skip_eval) and (not args.eval_h5):
		raise RuntimeError("启用了评估但缺少 eval_h5，请在 train_ar.yaml:data.eval_h5 中设置")


def set_seed(seed: int) -> None:
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)


def ensure_msvc_cl_on_path() -> str:
	"""在 Windows 下自动发现 cl.exe 并注入 PATH，返回 cl 路径；失败时返回空串。"""
	if os.name != "nt":
		return ""

	cl_path = shutil.which("cl")
	if cl_path:
		return cl_path

	roots = [os.environ.get("ProgramFiles(x86)"), os.environ.get("ProgramFiles")]
	editions = ("BuildTools", "Community", "Professional", "Enterprise", "Preview")
	candidates: List[Path] = []

	for root in roots:
		if not root:
			continue
		vs_root = Path(root) / "Microsoft Visual Studio"
		if not vs_root.exists():
			continue
		for edition in editions:
			pattern = f"*/{edition}/VC/Tools/MSVC/*/bin/Hostx64/x64/cl.exe"
			candidates.extend(vs_root.glob(pattern))

	if not candidates:
		return ""

	def _ver_key(p: Path) -> Tuple[int, ...]:
		version_str = p.parents[3].name
		parts = []
		for token in version_str.split("."):
			try:
				parts.append(int(token))
			except ValueError:
				parts.append(0)
		return tuple(parts)

	best = max(candidates, key=_ver_key)
	bin_dir = str(best.parent)
	os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
	return shutil.which("cl") or str(best)


def bootstrap_vs_build_env() -> bool:
	"""通过 LaunchDevCmd 导入 VS 编译环境变量，补全 INCLUDE/LIB/LIBPATH。"""
	if os.name != "nt":
		return True

	# 若关键变量已存在，则不重复导入。
	if os.environ.get("INCLUDE") and os.environ.get("LIB"):
		return True

	vs_devcmd = Path(r"C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\Common7\Tools\LaunchDevCmd.bat")
	if not vs_devcmd.exists():
		return False

	cmd = f'"{vs_devcmd}" -arch=x64 -host_arch=x64 >nul && set'
	proc = subprocess.run(
		["cmd", "/d", "/s", "/c", cmd],
		capture_output=True,
		text=True,
		encoding="utf-8",
		errors="ignore",
	)
	if proc.returncode != 0:
		proc = None

	updated = 0
	if proc is not None:
		for line in proc.stdout.splitlines():
			if "=" not in line:
				continue
			key, value = line.split("=", 1)
			if key in {"PATH", "INCLUDE", "LIB", "LIBPATH", "UCRTVersion", "WindowsSdkDir", "WindowsSdkVerBinPath"}:
				os.environ[key] = value
				updated += 1

	if updated > 0 and os.environ.get("INCLUDE") and os.environ.get("LIB"):
		return True

	# 兜底路径：手动拼接 MSVC + Windows SDK 的头文件与库目录。
	cl_path = ensure_msvc_cl_on_path()
	if not cl_path:
		return False

	cl_file = Path(cl_path)
	msvc_root = cl_file.parents[3]  # .../VC/Tools/MSVC/<version>
	msvc_include = msvc_root / "include"
	msvc_lib = msvc_root / "lib" / "x64"

	kit_root = Path(r"C:\Program Files (x86)\Windows Kits\10")
	include_root = kit_root / "Include"
	lib_root = kit_root / "Lib"

	def _pick_latest(base: Path) -> Path | None:
		if not base.exists():
			return None
		versions = [p for p in base.iterdir() if p.is_dir()]
		if not versions:
			return None
		return sorted(versions, key=lambda p: p.name)[-1]

	kit_inc_ver = _pick_latest(include_root)
	kit_lib_ver = _pick_latest(lib_root)
	if kit_inc_ver is None or kit_lib_ver is None:
		return False

	include_parts = [
		str(msvc_include),
		str(kit_inc_ver / "ucrt"),
		str(kit_inc_ver / "shared"),
		str(kit_inc_ver / "um"),
		str(kit_inc_ver / "winrt"),
		str(kit_inc_ver / "cppwinrt"),
	]
	lib_parts = [
		str(msvc_lib),
		str(kit_lib_ver / "ucrt" / "x64"),
		str(kit_lib_ver / "um" / "x64"),
	]

	def _merge_env(name: str, new_parts: List[str]) -> None:
		existing = os.environ.get(name, "")
		existing_parts = [p for p in existing.split(os.pathsep) if p]
		merged: List[str] = []
		for p in new_parts + existing_parts:
			if p and p not in merged:
				merged.append(p)
		os.environ[name] = os.pathsep.join(merged)

	_merge_env("INCLUDE", include_parts)
	_merge_env("LIB", lib_parts)
	_merge_env("LIBPATH", [str(msvc_lib)])

	return bool(os.environ.get("INCLUDE") and os.environ.get("LIB"))


def build_optimizer(model: LatexOCRModel, encoder_lr: float, decoder_lr: float, weight_decay: float, fpn_lr: float = None) -> optim.AdamW:
	if fpn_lr is None:
		fpn_lr = encoder_lr

	encoder_backbone_params: List[torch.nn.Parameter] = []
	encoder_fpn_params: List[torch.nn.Parameter] = []
	other_params: List[torch.nn.Parameter] = []

	for name, param in model.named_parameters():
		if not param.requires_grad:
			continue
		
		# 识别是否是 Encoder 参数
		if name.startswith("encoder.") or ".encoder." in name:
			# 将预训练的 backbone 与 新增的（FPN, PE, Transformer Encoder）分离
			if "backbone" in name:
				encoder_backbone_params.append(param)
			else:
				encoder_fpn_params.append(param)
		else:
			other_params.append(param)

	if not encoder_backbone_params:
		raise RuntimeError("未找到 encoder_backbone 可训练参数")
	if not other_params:
		raise RuntimeError("未找到 decoder/head 可训练参数")

	param_groups = [
		{"params": encoder_backbone_params, "lr": encoder_lr},
		{"params": other_params, "lr": decoder_lr},
	]
	if encoder_fpn_params:
		param_groups.append({"params": encoder_fpn_params, "lr": fpn_lr})

	return optim.AdamW(
		param_groups,
		weight_decay=weight_decay,
	)


class RatioBucketBatchSampler(Sampler[List[int]]):
	"""分桶后随机采样器：按比例抽样后先做宽高比分箱，再在箱内按面积排序。"""

	def __init__(
		self,
		lengths: Sequence[int],
		aspect_ratios: Sequence[np.ndarray],
		area_values: Sequence[np.ndarray],
		height_values: Sequence[np.ndarray],
		width_values: Sequence[np.ndarray],
		ratios: Sequence[int],
		total_samples: int,
		batch_size: int,
		seed: int = 42,
		mega_factor: int = 30,
		aspect_bin_edges: Sequence[float] = (0.55, 0.75, 0.9, 1.1, 1.35, 1.7, 2.2, 3.0, 4.2, 6.0),
		drop_last: bool = False,
	):
		if (
			len(lengths) != len(ratios)
			or len(lengths) != len(aspect_ratios)
			or len(lengths) != len(area_values)
			or len(lengths) != len(height_values)
			or len(lengths) != len(width_values)
		):
			raise ValueError("lengths/ratios/aspect_ratios/area_values/height_values/width_values 长度不一致")
		if any(length <= 0 for length in lengths):
			raise ValueError(f"存在空数据集，无法混采: {lengths}")
		if any(ratio <= 0 for ratio in ratios):
			raise ValueError(f"ratio 必须全为正整数: {ratios}")
		if total_samples <= 0:
			raise ValueError(f"total_samples 必须 > 0，当前为 {total_samples}")
		if batch_size <= 0:
			raise ValueError(f"batch_size 必须 > 0，当前为 {batch_size}")

		self.lengths = [int(v) for v in lengths]
		self.aspect_ratios = [np.asarray(v, dtype=np.float32) for v in aspect_ratios]
		self.area_values = [np.asarray(v, dtype=np.float32) for v in area_values]
		self.height_values = [np.asarray(v, dtype=np.float32) for v in height_values]
		self.width_values = [np.asarray(v, dtype=np.float32) for v in width_values]
		self.ratios = [int(v) for v in ratios]
		self.total_samples = int(total_samples)
		self.batch_size = int(batch_size)
		self.seed = int(seed)
		self.mega_factor = max(1, int(mega_factor))
		self.aspect_bin_edges = tuple(float(v) for v in aspect_bin_edges)
		self.drop_last = bool(drop_last)
		self.epoch = 0

		self.offsets: List[int] = []
		running = 0
		for length in self.lengths:
			self.offsets.append(running)
			running += length

		self.target_counts = self._build_target_counts(self.total_samples)
		self.sample_count = int(sum(self.target_counts))
		if self.drop_last:
			self.batch_count = self.sample_count // self.batch_size
		else:
			self.batch_count = (self.sample_count + self.batch_size - 1) // self.batch_size
		self._epoch_batches: List[List[int]] = []
		self._built_epoch: int | None = None

	def _build_target_counts(self, total_samples: int) -> List[int]:
		ratio_sum = sum(self.ratios)
		raw_counts = [total_samples * ratio / ratio_sum for ratio in self.ratios]
		int_counts = [int(c) for c in raw_counts]
		remain = int(total_samples - sum(int_counts))

		if remain > 0:
			frac_with_idx = sorted(
				[(raw_counts[i] - int_counts[i], i) for i in range(len(int_counts))],
				key=lambda x: x[0],
				reverse=True,
			)
			for j in range(remain):
				int_counts[frac_with_idx[j % len(frac_with_idx)][1]] += 1

		return int_counts

	def set_epoch(self, epoch: int) -> None:
		new_epoch = int(epoch)
		if self.epoch == new_epoch and self._built_epoch == new_epoch:
			return
		self.epoch = new_epoch
		if self._built_epoch != self.epoch:
			self._epoch_batches = []
			self._built_epoch = None

	def _aspect_bin(self, aspect: float) -> int:
		for i, edge in enumerate(self.aspect_bin_edges):
			if aspect < edge:
				return i
		return len(self.aspect_bin_edges)

	def _rebuild_epoch_batches(self) -> None:
		rng = random.Random(self.seed + self.epoch)
		pool: List[Tuple[int, int, float, float, float]] = []

		for length, ars, areas, hs, ws, need, offset in zip(
			self.lengths,
			self.aspect_ratios,
			self.area_values,
			self.height_values,
			self.width_values,
			self.target_counts,
			self.offsets,
		):
			if need <= length:
				local_indices = rng.sample(range(length), k=need)
			else:
				local_indices = [rng.randrange(length) for _ in range(need)]

			for local_idx in local_indices:
				aspect = float(ars[local_idx]) if local_idx < len(ars) else 1.0
				area = float(areas[local_idx]) if local_idx < len(areas) else 1.0
				h = float(hs[local_idx]) if local_idx < len(hs) else 256.0
				w = float(ws[local_idx]) if local_idx < len(ws) else 384.0
				aspect_bin = self._aspect_bin(aspect)
				# 以 area 归一后叠加 h/w，兼顾规模与形状接近性。
				shape_score = area + (h * 1e-1) + (w * 1e-3)
				pool.append((offset + local_idx, aspect_bin, h, w, shape_score))

		rng.shuffle(pool)

		mega_size = self.batch_size * self.mega_factor
		mega_starts = list(range(0, len(pool), mega_size))
		rng.shuffle(mega_starts)
		epoch_batches: List[List[int]] = []
		spill_items: List[Tuple[int, int, float, float, float]] = []

		for start in mega_starts:
			mega = pool[start: start + mega_size]
			bin_groups: Dict[int, List[Tuple[int, int, float, float, float]]] = {}
			for item in mega:
				bin_groups.setdefault(item[1], []).append(item)

			local_batches: List[List[int]] = []
			local_spill: List[Tuple[int, int, float, float, float]] = []
			bin_ids = list(bin_groups.keys())
			rng.shuffle(bin_ids)
			for bin_id in bin_ids:
				group = bin_groups[bin_id]
				group.sort(key=lambda item: (item[2], item[3], item[4] + rng.uniform(-1e-6, 1e-6)))
				full_end = (len(group) // self.batch_size) * self.batch_size

				for i in range(0, full_end, self.batch_size):
					chunk = group[i: i + self.batch_size]
					batch_indices = [idx for idx, _, _, _, _ in chunk]
					rng.shuffle(batch_indices)
					local_batches.append(batch_indices)

				if full_end < len(group):
					local_spill.extend(group[full_end:])

			rng.shuffle(local_batches)
			epoch_batches.extend(local_batches)
			rng.shuffle(local_spill)
			spill_items.extend(local_spill)

		spill_groups: Dict[int, List[Tuple[int, int, float, float, float]]] = {}
		for item in spill_items:
			spill_groups.setdefault(item[1], []).append(item)

		spill_tail_indices: List[int] = []
		spill_bin_ids = list(spill_groups.keys())
		rng.shuffle(spill_bin_ids)
		for bin_id in spill_bin_ids:
			group = spill_groups[bin_id]
			group.sort(key=lambda item: (item[2], item[3], item[4] + rng.uniform(-1e-6, 1e-6)))
			full_end = (len(group) // self.batch_size) * self.batch_size

			for i in range(0, full_end, self.batch_size):
				chunk = group[i: i + self.batch_size]
				batch_indices = [idx for idx, _, _, _, _ in chunk]
				rng.shuffle(batch_indices)
				epoch_batches.append(batch_indices)

			if full_end < len(group):
				spill_tail_indices.extend(idx for idx, _, _, _, _ in group[full_end:])

		rng.shuffle(spill_tail_indices)
		if self.drop_last:
			usable = (len(spill_tail_indices) // self.batch_size) * self.batch_size
			spill_tail_indices = spill_tail_indices[:usable]

		for i in range(0, len(spill_tail_indices), self.batch_size):
			chunk = spill_tail_indices[i: i + self.batch_size]
			if len(chunk) < self.batch_size and self.drop_last:
				continue
			rng.shuffle(chunk)
			epoch_batches.append(chunk)

		rng.shuffle(epoch_batches)
		self._epoch_batches = epoch_batches
		self.batch_count = len(self._epoch_batches)
		self._built_epoch = self.epoch

	def __iter__(self):
		if self._built_epoch != self.epoch:
			self._rebuild_epoch_batches()
		for batch in self._epoch_batches:
			yield batch

	def __len__(self) -> int:
		if self._built_epoch == self.epoch:
			return len(self._epoch_batches)
		return self.batch_count


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
def batched_infer_ar(
	model: LatexOCRModel,
	images: torch.Tensor,
	pad_id: int,
	bos_id: int,
	eos_id: int,
	max_len: int,
	amp_enabled: bool,
	amp_dtype: torch.dtype,
) -> List[List[int]]:
	"""批量版 AR 贪心解码，用于加速训练期评估。"""
	batch_size = images.size(0)
	device = images.device

	# 视觉特征在一次迭代解码中不变，仅提取一次并复用
	with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
		memory, memory_padding_mask = model.encode(images, return_aux=False)
		decode_cache = model.init_decode_cache(memory)

	generated = torch.full((batch_size, max_len), pad_id, dtype=torch.long, device=device)
	generated[:, 0] = bos_id
	finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

	for step in range(1, max_len):
		with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
			current_token = generated[:, step - 1]
			logits, decode_cache = model.decode_step_cached(
				memory=memory,
				token_id=current_token,
				cache=decode_cache,
				memory_padding_mask=memory_padding_mask,
			)

		next_tokens = logits.argmax(dim=-1)
		next_tokens = next_tokens.masked_fill(finished, pad_id)
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


@torch.no_grad()
def evaluate_epoch(
	model: LatexOCRModel,
	eval_loader: DataLoader,
	tokenizer: Tokenizer,
	device: torch.device,
	amp_enabled: bool,
	amp_dtype: torch.dtype,
	channels_last: bool,
	pad_id: int,
	bos_id: int,
	eos_id: int,
	num_samples: int,
	max_len: int,
) -> Tuple[float, Counter[str], Dict[str, float]]:
	"""参考 train_2d_phase2 的评估流程：按 epoch 抽样评估 + Top 错误统计。"""
	model.eval()

	processed = 0
	exact = 0
	total_ned = 0.0
	error_counter: Counter[str] = Counter()

	pbar = tqdm(eval_loader, desc="Eval", leave=False)
	for batch in pbar:
		if num_samples > 0 and processed >= num_samples:
			break

		images = batch["images"].to(device)
		if channels_last and images.ndim == 4:
			images = images.contiguous(memory_format=torch.channels_last)
		target_ids = batch["target_ids"]

		pred_batch = batched_infer_ar(
			model=model,
			images=images,
			pad_id=pad_id,
			bos_id=bos_id,
			eos_id=eos_id,
			max_len=max_len,
			amp_enabled=amp_enabled,
			amp_dtype=amp_dtype,
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


def set_backbone_grad_checkpointing(model: LatexOCRModel, enabled: bool) -> None:
	"""在运行期切换视觉骨干的 timm 梯度检查点开关。"""
	raw_model = cast(LatexOCRModel, getattr(model, "_orig_mod", model))
	encoder = getattr(raw_model, "encoder", None)
	backbone = getattr(encoder, "backbone", None)
	setter = getattr(backbone, "set_grad_checkpointing", None)
	if callable(setter):
		setter(bool(enabled))


def warmup_kernel_cache(
	model: LatexOCRModel,
	train_loader: DataLoader,
	device: torch.device,
	amp_enabled: bool,
	amp_dtype: torch.dtype,
	channels_last: bool,
	warmup_batches: int,
) -> int:
	"""训练前执行少量前向预热，提前触发常见 shape 的内核/算法选择。"""
	if warmup_batches <= 0:
		return 0

	model.train()
	steps = 0
	loader_iter = iter(train_loader)

	with torch.no_grad():
		for _ in range(int(warmup_batches)):
			try:
				batch = next(loader_iter)
			except StopIteration:
				break

			images = batch["images"].to(device, non_blocking=True)
			if channels_last and images.ndim == 4:
				images = images.contiguous(memory_format=torch.channels_last)
			if "decoder_inputs" in batch:
				decoder_inputs = batch["decoder_inputs"].to(device, non_blocking=True)
			else:
				clean_token_ids = batch["clean_token_ids"].to(device, non_blocking=True)
				decoder_inputs = clean_token_ids[:, :-1]

			with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
				_ = model(images=images, tgt_seq=decoder_inputs, is_causal=True)

			steps += 1

	if device.type == "cuda":
		torch.cuda.synchronize()

	return steps


def main() -> None:
	args = parse_args()
	train_cfg = load_yaml_config(args.train_config)
	model_cfg = load_yaml_config(args.model_config)
	apply_config_defaults(args, train_cfg, model_cfg)
	set_seed(args.seed)

	# 强制关闭动态形状下的底层重编译/寻优抖动
	torch.backends.cudnn.benchmark = bool(args.cudnn_benchmark)
	# benchmark=True 时关闭 deterministic 以释放 cudnn 算法选择性能。
	torch.backends.cudnn.deterministic = not bool(args.cudnn_benchmark)

	os.makedirs(args.ckpt_dir, exist_ok=True)
	os.makedirs(args.log_dir, exist_ok=True)
	log_path = os.path.join(args.log_dir, "train_ar.log")

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	amp_dtype_map: Dict[str, torch.dtype] = {
		"fp16": torch.float16,
		"bf16": torch.bfloat16,
		"fp32": torch.float32,
	}
	amp_dtype = amp_dtype_map[args.amp_dtype]
	amp_enabled = (device.type == "cuda") and (args.amp_dtype != "fp32")
	eval_amp_dtype = amp_dtype_map[args.eval_amp_dtype]
	eval_amp_enabled = (device.type == "cuda") and (args.eval_amp_dtype != "fp32")
	if device.type != "cuda" and args.amp_dtype != "fp32":
		print("警告: 当前为 CPU 设备，已自动降级为 fp32 训练。")
		amp_dtype = torch.float32
		amp_enabled = False
	if device.type != "cuda" and args.eval_amp_dtype != "fp32":
		print("警告: 当前为 CPU 设备，评估精度已自动降级为 fp32。")
		eval_amp_dtype = torch.float32
		eval_amp_enabled = False

	tokenizer = Tokenizer.from_file(args.tokenizer)
	vocab_size = tokenizer.get_vocab_size()

	pad_id = tokenizer.token_to_id("[PAD]")
	bos_id = tokenizer.token_to_id("[BOS]")
	eos_id = tokenizer.token_to_id("[EOS]")

	if pad_id is None:
		raise RuntimeError("Tokenizer 缺少 [PAD]，无法进行 AR 训练")
	if bos_id is None or eos_id is None:
		raise RuntimeError("Tokenizer 缺少 [BOS] 或 [EOS]，无法进行 AR 评估解码")

	dataset_paths = [p for p in args.train_h5 if os.path.exists(p)]
	if not dataset_paths:
		raise FileNotFoundError(f"训练集不存在，请检查路径: {args.train_h5}")

	datasets = [
		FormulaDataset(
			h5_path=path,
			tokenizer_path=args.tokenizer,
			max_area=args.max_area,
			enable_augment=args.enable_augment,
			augment_config=args.augmentation_cfg,
			enable_extreme_augment=bool(args.enable_extreme_augment),
			extreme_augment_config=args.extreme_augmentation_cfg,
		)
		for path in dataset_paths
	]

	if len(args.mix_ratio) != len(datasets):
		raise ValueError(
			f"mix_ratio 长度({len(args.mix_ratio)})必须与训练数据集数量({len(datasets)})一致"
		)

	# 根据 mix_ratio 过滤掉比例为 0 的数据集
	enabled_indices = [i for i, ratio in enumerate(args.mix_ratio) if ratio > 0]
	if not enabled_indices:
		raise ValueError(f"mix_ratio 全为 0，没有有效的数据源。请检查配置: {args.mix_ratio}")

	datasets = [datasets[i] for i in enabled_indices]
	args.mix_ratio = [args.mix_ratio[i] for i in enabled_indices]
	dataset_paths = [dataset_paths[i] for i in enabled_indices]

	train_dataset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)

	dataset_lengths = [len(ds) for ds in datasets]
	aspect_ratios = [np.asarray(ds.aspect_ratios, dtype=np.float32) for ds in datasets]
	area_values = [
		np.asarray(getattr(ds, "resized_areas", np.ones((len(ds),), dtype=np.float32)), dtype=np.float32)
		for ds in datasets
	]
	height_values = [
		np.asarray(getattr(ds, "resized_heights", np.full((len(ds),), 256.0, dtype=np.float32)), dtype=np.float32)
		for ds in datasets
	]
	width_values = [
		np.asarray(getattr(ds, "resized_widths", np.full((len(ds),), 384.0, dtype=np.float32)), dtype=np.float32)
		for ds in datasets
	]
	epoch_samples = int(args.epoch_samples) if int(args.epoch_samples) > 0 else int(sum(dataset_lengths))

	train_batch_sampler = RatioBucketBatchSampler(
		lengths=dataset_lengths,
		aspect_ratios=aspect_ratios,
		area_values=area_values,
		height_values=height_values,
		width_values=width_values,
		ratios=[int(v) for v in args.mix_ratio],
		total_samples=epoch_samples,
		batch_size=args.batch_size,
		seed=args.seed,
		mega_factor=args.bucket_mega_factor,
		drop_last=False,
	)

	collate_fn = ARCollate(
		pad_token_id=pad_id,
		bos_token_id=bos_id,
		eos_token_id=eos_id,
	)

	train_loader = DataLoader(
		train_dataset,
		batch_sampler=train_batch_sampler,
		collate_fn=collate_fn,
		num_workers=args.num_workers,
		pin_memory=(device.type == "cuda"),
		persistent_workers=(args.num_workers > 0),
	)

	eval_loader = None
	if not args.skip_eval:
		if not os.path.exists(args.eval_h5):
			raise FileNotFoundError(f"评估集不存在: {args.eval_h5}")

		eval_dataset = FormulaDataset(
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

	model = LatexOCRModel(
		vocab_size=vocab_size, 
		d_model=args.d_model, 
		pad_id=pad_id,
		eos_id=eos_id,
		use_gradient_checkpointing=args.use_gradient_checkpointing,
		checkpoint_segments=args.checkpoint_segments if args.checkpoint_segments is not None else 1,
		checkpoint_decoder_layers=args.checkpoint_decoder_layers if args.checkpoint_decoder_layers is not None else False,
	).to(device)
	if args.channels_last:
		model = model.to(memory_format=torch.channels_last) # type: ignore
	if os.name == "nt":
		if bootstrap_vs_build_env():
			print("✅ 已导入 VS Build Tools 编译环境(INCLUDE/LIB/LIBPATH)。")
		else:
			print("⚠️ 未能导入 VS Build Tools 编译环境，torch.compile 可能回退。")
	compile_fn = getattr(torch, "compile", None)
	cl_path = ensure_msvc_cl_on_path() if os.name == "nt" else ""
	can_compile = (compile_fn is not None) and (os.name != "nt" or bool(cl_path))
	if compile_fn is not None and os.name == "nt" and not cl_path:
		print("⚠️ 检测到未安装 MSVC cl.exe，已禁用 torch.compile 以避免 Inductor 运行时崩溃。")
	if compile_fn is not None and os.name == "nt" and cl_path:
		print(f"✅ 已检测到 MSVC 编译器: {cl_path}")
	if can_compile:
		if compile_fn is None:
			raise RuntimeError("内部错误: can_compile=True 但 torch.compile 不可用")
		print("⚡ 开启 torch.compile(dynamic=True) 进行底层算子融合...")
		# 允许 Dynamo 在后续动态 shape 重编译失败时自动回退 eager，避免中途训练中断。
		torch._dynamo.config.suppress_errors = True
		compiled_model = cast(LatexOCRModel, compile_fn(model, dynamic=True))
		try:
			with torch.no_grad():
				probe_images = torch.zeros((1, 1, 64, 64), dtype=torch.float32, device=device)
				probe_tgt = torch.full((1, 2), pad_id, dtype=torch.long, device=device)
				_ = compiled_model(images=probe_images, tgt_seq=probe_tgt, is_causal=True)
			model = compiled_model
		except Exception as exc:
			print(f"⚠️ torch.compile 预热探测失败，已回退 eager 模式: {type(exc).__name__}: {exc}")
			if device.type == "cuda":
				torch.cuda.empty_cache()
	optimizer = build_optimizer(
		model=model,
		encoder_lr=args.encoder_lr,
		decoder_lr=args.decoder_lr,
		weight_decay=args.weight_decay,
		fpn_lr=getattr(args, "fpn_lr", args.encoder_lr),
	)

	total_steps = infer_total_steps(args.epochs, len(train_batch_sampler))
	warmup_steps = int(args.warmup_epochs * len(train_batch_sampler))
	scheduler = build_linear_warmup_cosine_scheduler(
		optimizer=optimizer,
		total_steps=total_steps,
		warmup_steps=warmup_steps,
		warmup_start_lr=args.warmup_start_lr,
		eta_min=args.eta_min,
	)

	scaler = None
	if device.type == "cuda" and args.amp_dtype == "fp16":
		new_grad_scaler = getattr(torch.amp, "GradScaler", None)
		if new_grad_scaler is not None:
			scaler = new_grad_scaler("cuda", enabled=True)
		else:
			scaler = torch.cuda.amp.GradScaler(enabled=True)
	ema_avg_fn = get_ema_multi_avg_fn(decay=0.999)
	ema_model: AveragedModel | None = None

	start_epoch = 1
	best_loss = float("inf")
	best_em = 0.0

	if args.resume and os.path.exists(args.resume_ckpt):
		checkpoint = torch.load(args.resume_ckpt, map_location=device, weights_only=False)

		# 使用 .get() 安全获取模型，兼容纯权重字典
		state_dict = checkpoint.get("model_raw", checkpoint.get("model", checkpoint))
		model_state_dict = convert_legacy_attnres_state_dict(state_dict)
		incompatible = model.load_state_dict(model_state_dict, strict=False)
		if incompatible.missing_keys:
			print(f"警告: checkpoint 缺少 {len(incompatible.missing_keys)} 个参数（含新增 BoW 头时属正常）。")
		if incompatible.unexpected_keys:
			print(f"警告: checkpoint 存在 {len(incompatible.unexpected_keys)} 个未使用参数。")

		ema_model = AveragedModel(model, multi_avg_fn=ema_avg_fn)
		ema_state_dict = checkpoint.get("ema_model")
		if isinstance(ema_state_dict, dict):
			try:
				ema_model.load_state_dict(ema_state_dict)
				print("成功恢复 EMA 模型状态。")
			except ValueError as exc:
				print(f"警告: EMA 状态与当前模型不兼容，已跳过恢复。原因: {exc}")
		elif checkpoint.get("ema_model_module") is not None:
			try:
				ema_model.module.load_state_dict(checkpoint["ema_model_module"], strict=False)
				print("成功恢复 EMA module 状态。")
			except ValueError as exc:
				print(f"警告: EMA module 状态与当前模型不兼容，已跳过恢复。原因: {exc}")

		# 安全检查：只有当 checkpoint 里确实包含 optimizer 时才去加载
		if getattr(args, "reset_lr", False):
			print("提示: 已启用 reset_lr，跳过加载 optimizer/scheduler 状态，触发冷启动预热阶段。")
		elif "optimizer" in checkpoint and "scheduler" in checkpoint:
			try:
				optimizer.load_state_dict(checkpoint["optimizer"])
				scheduler.load_state_dict(checkpoint["scheduler"])
				print("成功恢复 Optimizer 和 Scheduler 状态。")
			except ValueError as exc:
				print(f"警告: checkpoint 的 optimizer/scheduler 状态与当前模型不兼容，已跳过恢复。原因: {exc}")
		else:
			print("提示: 未在 checkpoint 中找到 optimizer，已作为全新微调任务启动（仅加载模型权重）。")

		if scaler is not None and checkpoint.get("scaler") is not None:
			scaler.load_state_dict(checkpoint["scaler"])

		start_epoch = int(checkpoint.get("epoch", 0)) + 1
		best_loss = float(checkpoint.get("best_loss", best_loss))
		best_em = float(checkpoint.get("best_em", best_em))

		best_ckpt_path = os.path.join(os.path.dirname(args.resume_ckpt), "best.pth")
		if os.path.exists(best_ckpt_path):
			best_ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
			best_loss = min(best_loss, float(best_ckpt.get("best_loss", best_loss)))
			best_em = max(best_em, float(best_ckpt.get("best_em", best_em)))

		best_loss_ckpt_path = os.path.join(os.path.dirname(args.resume_ckpt), "best_loss.pth")
		if os.path.exists(best_loss_ckpt_path):
			best_loss_ckpt = torch.load(best_loss_ckpt_path, map_location=device, weights_only=False)
			best_loss = min(best_loss, float(best_loss_ckpt.get("best_loss", best_loss)))

		log_best_loss = get_best_loss_from_log(log_path=log_path, max_epoch=start_epoch - 1)
		if log_best_loss is not None:
			best_loss = min(best_loss, float(log_best_loss))
		print(f"恢复训练: epoch={start_epoch}, best_loss={best_loss:.4f}, best_em={best_em * 100:.2f}%")

	if ema_model is None:
		ema_model = AveragedModel(model, multi_avg_fn=ema_avg_fn)

	eval_model_for_ckpt = cast(LatexOCRModel, ema_model.module if ema_model is not None else model)
	eval_model = eval_model_for_ckpt

	trainer = ARTrainer(
		model=model,
		optimizer=optimizer,
		scheduler=scheduler,
		device=device,
		scaler=scaler,
		amp_enabled=amp_enabled,
		amp_dtype=amp_dtype,
		channels_last=bool(args.channels_last),
		label_smoothing=args.label_smoothing,
		target_ignore_index=-100,
		bow_weight=args.bow_weight,
		ohem_ratio=args.ohem_ratio,
		ema_model=ema_model,
		enable_sigreg=bool(args.enable_sigreg),
		sigreg_weight=float(args.sigreg_weight),
		sigreg_num_projections=int(args.sigreg_num_projections),
		sigreg_collapse_threshold=float(args.sigreg_collapse_threshold),
		decoder_input_noise_ratio=float(args.decoder_input_noise_ratio),
	)

	if not os.path.exists(log_path):
		with open(log_path, "w", encoding="utf-8") as f:
			f.write("Timestamp\tEpoch\tLoss\tCE\tBoW\tSIGReg\tTokenAcc(%)\tEvalProcessed\tEvalExact\tEvalEM(%)\tEvalNED\tLREnc\tLRDec\n")

	print("=" * 80)
	print(
		f"AR 训练启动 | Device={device} | AMP={amp_enabled} | AMP_DTYPE={args.amp_dtype} | "
		f"LabelSmoothing={args.label_smoothing}"
	)
	print(
		f"Eval 默认策略 | EvalAMP={eval_amp_enabled} | EvalAMP_DTYPE={args.eval_amp_dtype} | "
		f"EvalGradCheckpoint={bool(args.eval_use_gradient_checkpointing)}"
	)
	print(f"TrainConfig={args.train_config}")
	print(f"ModelConfig={args.model_config}")
	print(f"Datasets={dataset_paths}")
	print(f"MixRatio={args.mix_ratio} | EpochSamples={train_batch_sampler.sample_count} | TrainBatches={len(train_batch_sampler)}")
	print(
		f"cudnn.benchmark={torch.backends.cudnn.benchmark} | "
		f"cudnn.deterministic={torch.backends.cudnn.deterministic} | "
		f"channels_last={bool(args.channels_last)} | "
		f"KernelWarmupBatches={args.kernel_warmup_batches} | "
		f"ExtremeAugment={bool(args.enable_extreme_augment)}"
	)
	if args.enable_sigreg:
		print(f"SIGReg: Enabled | Weight={args.sigreg_weight} | NumProjections={args.sigreg_num_projections} | CollapseThreshold={args.sigreg_collapse_threshold}")
	else:
		print(f"SIGReg: Disabled")
	if not args.skip_eval:
		print(f"EvalSet={args.eval_h5} | EvalBatch={args.eval_batch_size} | EvalSamples={args.eval_samples}")
	print(f"Total Steps={total_steps}, Warmup Steps={warmup_steps}")
	print("=" * 80)

	train_batch_sampler.set_epoch(start_epoch)
	warmed = warmup_kernel_cache(
		model=model,
		train_loader=train_loader,
		device=device,
		amp_enabled=amp_enabled,
		amp_dtype=amp_dtype,
		channels_last=bool(args.channels_last),
		warmup_batches=args.kernel_warmup_batches,
	)
	if warmed > 0:
		print(f"内核预热完成: {warmed} batches")

	for epoch in range(start_epoch, args.epochs + 1):
		train_batch_sampler.set_epoch(epoch)
		avg_loss, avg_ce, avg_bow, avg_sigreg, avg_acc = trainer.train_epoch(train_loader, epoch=epoch, log_interval=args.log_interval)

		eval_processed = 0.0
		eval_exact = 0.0
		eval_em = 0.0
		eval_ned = 1.0
		error_counter: Counter[str] = Counter()

		if (not args.skip_eval) and eval_loader is not None and (epoch % max(1, args.eval_interval) == 0):
			set_backbone_grad_checkpointing(eval_model, bool(args.eval_use_gradient_checkpointing))
			try:
				eval_em, error_counter, eval_stats = evaluate_epoch(
					model=eval_model,
					eval_loader=eval_loader,
					tokenizer=tokenizer,
					device=device,
					amp_enabled=eval_amp_enabled,
					amp_dtype=eval_amp_dtype,
					channels_last=bool(args.channels_last),
					pad_id=pad_id,
					bos_id=bos_id,
					eos_id=eos_id,
					num_samples=args.eval_samples,
					max_len=args.eval_max_len,
				)
			finally:
				set_backbone_grad_checkpointing(eval_model, bool(args.use_gradient_checkpointing))
			eval_processed = eval_stats["processed"]
			eval_exact = eval_stats["exact"]
			eval_ned = eval_stats["avg_ned"]

		lr_enc = float(optimizer.param_groups[0]["lr"])
		lr_dec = float(optimizer.param_groups[1]["lr"])

		print(
			f"Epoch {epoch}/{args.epochs} | "
			f"Loss={avg_loss:.4f} | TokenAcc={avg_acc * 100:.2f}% | "
			f"EvalEM={eval_em * 100:.2f}% | EvalNED={eval_ned:.4f} | "
			f"LRE={lr_enc:.2e} | LRD={lr_dec:.2e}"
		)

		timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
		with open(log_path, "a", encoding="utf-8") as f:
			f.write(
				f"{timestamp}\t{epoch}\t{avg_loss:.6f}\t{avg_ce:.6f}\t{avg_bow:.6f}\t{avg_sigreg:.6f}\t{avg_acc * 100:.4f}\t"
				f"{int(eval_processed)}\t{int(eval_exact)}\t{eval_em * 100:.4f}\t{eval_ned:.6f}\t"
				f"{lr_enc:.8e}\t{lr_dec:.8e}\n"
			)

		is_best_em = eval_em >= best_em
		is_best_loss = avg_loss <= best_loss
		if is_best_em:
			best_em = eval_em
		if is_best_loss:
			best_loss = avg_loss

		ckpt_payload: Dict[str, object] = {
			"epoch": epoch,
			"model": eval_model_for_ckpt.state_dict(),
			"model_raw": model.state_dict(),
			# "ema_model": ema_model.state_dict() if ema_model is not None else None,
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
		if scaler is not None:
			ckpt_payload["scaler"] = scaler.state_dict()

		latest_ckpt = os.path.join(args.ckpt_dir, "latest.pth")
		epoch_ckpt = os.path.join(args.ckpt_dir, f"epoch_{epoch}.pth")
		best_ckpt = os.path.join(args.ckpt_dir, "best.pth")
		best_loss_ckpt = os.path.join(args.ckpt_dir, "best_loss.pth")

		torch.save(ckpt_payload, latest_ckpt)
		torch.save(ckpt_payload, epoch_ckpt)

		if error_counter:
			err_path = os.path.join(args.log_dir, f"ar_epoch_{epoch}_top_errors.txt")
			with open(err_path, "w", encoding="utf-8") as ef:
				for key, count in error_counter.most_common(20):
					ef.write(f"[{count:5d}] {key}\n")

		if is_best_em:
			torch.save(ckpt_payload, best_ckpt)
			print(f"更新 best checkpoint: {best_ckpt} | best_em={best_em * 100:.2f}%")

		if is_best_loss:
			torch.save(ckpt_payload, best_loss_ckpt)
			print(f"更新最小训练损失: {best_loss:.4f}")

		# 强制清空显存碎片，为下一个 Epoch 准备更连续的可用显存
		if torch.cuda.is_available():
			torch.cuda.empty_cache()
		print(f"Epoch {epoch} 结束，准备进入下一轮...")

	print("\nAR 训练完成。")


if __name__ == "__main__":
	main()
