"""
动态分辨率转换脚本 (Dynamic Resolution Conversion) - 修复版
集成了 Tokenizer，完美对齐 2D Dataset 结构 (images, labels, widths, heights)
消除了 Pylance 的静态类型检查警告
"""

import argparse
import csv
import math
import os
import sys
import pickle
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import h5py
import numpy as np
from tqdm import tqdm

# 确保能导入项目内的 tokenizer
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))
try:
    from src.data.tokenizer import tokenize_latex
except ImportError:
    print("⚠️ 警告: 无法导入 tokenizer_latex，请确保您的项目结构正确。")


def calculate_dynamic_dims(
    h_orig: int,
    w_orig: int,
    max_area: int = 98304,
    min_size: int = 32,
    stride: int = 16,
) -> Tuple[int, int]:
    if h_orig <= 0 or w_orig <= 0:
        return min_size, min_size

    aspect_ratio = float(w_orig) / float(h_orig)
    target_h = math.sqrt(max_area / aspect_ratio)
    target_w = target_h * aspect_ratio

    def align_to_stride(val: float) -> int:
        aligned = int(round(val / stride) * stride)
        return max(min_size, aligned)

    aligned_h = align_to_stride(target_h)
    aligned_w = align_to_stride(target_w)

    if aligned_h * aligned_w > max_area + (stride * stride * 2):
        if aspect_ratio >= 1:
            aligned_w = max(min_size, aligned_w - stride)
        else:
            aligned_h = max(min_size, aligned_h - stride)

    return aligned_h, aligned_w


def bucket_by_aspect(aspect_ratio: float, buckets: Optional[List[Tuple[float, float, str]]] = None) -> str:
    if buckets is None:
        buckets = [
            (0.0, 0.5, "tall"),       
            (0.5, 1.0, "normal"),     
            (1.0, 1.5, "landscape"),  
            (1.5, 10.0, "ultra_wide"),
        ]
    for min_ar, max_ar, label in buckets:
        if min_ar <= aspect_ratio < max_ar:
            return label
    return "unknown"


def sanitize_for_filename(text: str, max_len: int = 48) -> str:
    safe = [ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in text]
    collapsed = "".join(safe).strip("_")
    return collapsed[:max_len] if collapsed else "nolabel"


def decode_label(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    return str(value)


def flat_to_strokes(points: np.ndarray, stroke_lengths: np.ndarray) -> List[np.ndarray]:
    strokes: List[np.ndarray] = []
    cursor = 0
    for sl in stroke_lengths:
        seg_len = int(sl)
        if seg_len > 0:
            strokes.append(points[cursor: cursor + seg_len].copy())
        cursor += seg_len
    return strokes


def render_trajectory_to_image(
    strokes: List[np.ndarray],
    image_height: int = 192,
    padding: int = 4,
    random_thickness: bool = False,
    blur_kernel: int = 1,
) -> Optional[np.ndarray]:
    valid_strokes = [s for s in strokes if s is not None and len(s) > 0]
    if not valid_strokes: 
        return None

    all_points = np.vstack(valid_strokes)
    min_x, min_y = np.min(all_points, axis=0)
    max_x, max_y = np.max(all_points, axis=0)

    width = float(max_x - min_x)
    height = float(max_y - min_y)
    if height <= 1e-8: 
        height = 1.0

    scale = max((image_height - 2 * padding) / height, 1e-8)
    target_width = max(int(round(width * scale)) + 2 * padding, 2 * padding + 1)
    canvas = np.zeros((image_height, target_width), dtype=np.uint8)

    thickness = int(np.random.choice([1, 2])) if random_thickness else 2

    for stroke in valid_strokes:
        scaled = np.zeros_like(stroke, dtype=np.int32)
        scaled[:, 0] = ((stroke[:, 0] - min_x) * scale).astype(np.int32) + padding
        scaled[:, 1] = ((stroke[:, 1] - min_y) * scale).astype(np.int32) + padding
        if len(scaled) == 1:
            cv2.circle(canvas, tuple(scaled[0].tolist()), radius=max(1, thickness - 1), color=255, thickness=-1)
        else:
            cv2.polylines(canvas, [scaled.reshape((-1, 1, 2))], isClosed=False, color=255, thickness=thickness, lineType=cv2.LINE_AA)

    if blur_kernel > 1 and blur_kernel % 2 == 1:
        canvas = cv2.GaussianBlur(canvas, (blur_kernel, blur_kernel), 0)

    return canvas


def choose_indices(total: int, sample_size: Optional[int], seed: int) -> List[int]:
    if total <= 0: return []
    if sample_size is None or sample_size >= total: return list(range(total))
    rng = np.random.RandomState(seed)
    return [int(i) for i in sorted(rng.choice(total, sample_size, replace=False))]


def load_label_map_from_metadata(input_dir: Path) -> Dict[str, str]:
    meta_path = input_dir / "metadata.csv"
    if not meta_path.exists(): return {}
    with open(meta_path, "r", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        if not reader.fieldnames or "file_name" not in reader.fieldnames or "label" not in reader.fieldnames:
            return {}
        return {str(row["file_name"]).strip(): str(row["label"]).strip() for row in reader if str(row["file_name"]).strip()}


class H5ImageWriter:
    """完美对齐 dataset_2d.py 的 HDF5 写入器"""
    def __init__(self, h5_path: str):
        self.h5_path = h5_path
        self.count = 0
        self._f = h5py.File(h5_path, "w")
        
        dt_bytes = h5py.vlen_dtype(np.uint8)
        dt_ints = h5py.vlen_dtype(np.int32)
        str_dtype = h5py.string_dtype(encoding="utf-8")

        self._d_images = self._f.create_dataset("images", shape=(0,), maxshape=(None,), dtype=dt_bytes, chunks=True)
        self._d_labels = self._f.create_dataset("labels", shape=(0,), maxshape=(None,), dtype=dt_ints, chunks=True)
        self._d_widths = self._f.create_dataset("widths", shape=(0,), maxshape=(None,), dtype=np.int32, chunks=True)
        self._d_heights = self._f.create_dataset("heights", shape=(0,), maxshape=(None,), dtype=np.int32, chunks=True)
        
        self._d_raw_labels = self._f.create_dataset("raw_labels", shape=(0,), maxshape=(None,), dtype=str_dtype, chunks=True)
        self._d_file_name = self._f.create_dataset("file_name", shape=(0,), maxshape=(None,), dtype=str_dtype, chunks=True)

    def append(self, img_bytes: np.ndarray, tokens_array: np.ndarray, width: int, height: int, raw_label: str, file_name: str) -> None:
        i = self.count
        next_size = i + 1

        self._d_images.resize((next_size,))
        self._d_labels.resize((next_size,))
        self._d_widths.resize((next_size,))
        self._d_heights.resize((next_size,))
        self._d_raw_labels.resize((next_size,))
        self._d_file_name.resize((next_size,))

        self._d_images[i] = img_bytes
        self._d_labels[i] = tokens_array
        self._d_widths[i] = width
        self._d_heights[i] = height
        self._d_raw_labels[i] = raw_label
        self._d_file_name[i] = file_name

        self.count = next_size

    def close(self) -> None:
        self._f.attrs["num_samples"] = self.count
        self._f.close()

    def __enter__(self): return self
    def __exit__(self, exc_type: Any, exc: Any, tb: Any): self.close()


def process_label(label_str: str, char2id: dict) -> Tuple[np.ndarray, int]:
    tokens_str = tokenize_latex(label_str)
    token_ids = []
    oov_count = 0
    for t in tokens_str:
        if t in char2id:
            token_ids.append(char2id[t])
        else:
            token_ids.append(0) 
            oov_count += 1
            
    final_tokens = [1] + token_ids + [2]
    return np.array(final_tokens, dtype=np.int32), oov_count


def convert_source_h5_to_resized_h5(
    source_h5: str, h5_output: str, output_dir: str, char2id: dict,
    max_area: int = 98304, stride: int = 16, min_size: int = 32,
    sample_size: Optional[int] = None, seed: int = 2026, print_limit: int = 50,
    image_height: int = 192, padding: int = 4, random_thickness: bool = False,
    blur_kernel: int = 1, save_images: bool = False, use_buckets: bool = False,
) -> None:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    bucket_dirs = {}
    if save_images and use_buckets:
        for bucket_name in ["tall", "normal", "landscape", "ultra_wide", "unknown"]:
            (out_path / bucket_name).mkdir(exist_ok=True)
            bucket_dirs[bucket_name] = out_path / bucket_name

    meta_csv = out_path / "metadata.csv"
    total_oovs = 0

    with h5py.File(source_h5, "r") as src:
        pt_off = np.asarray(src["sample_point_offsets"])
        st_off = np.asarray(src["sample_stroke_offsets"])
        
        # [修复] 通过添加 : Any 类型注解，消除 Pylance 找不到 __getitem__ 的报错
        d_points: Any = src["points"]
        d_stroke_lengths: Any = src["stroke_lengths"]
        d_labels: Any = src["labels"]

        indices = choose_indices(int(pt_off.shape[0]) - 1, sample_size, seed)

        with open(meta_csv, "w", newline="", encoding="utf-8") as fp, H5ImageWriter(h5_output) as h5_writer:
            writer = csv.writer(fp)
            writer.writerow(["file_name", "h5_idx", "label", "bucket", "target_h", "target_w"])

            converted, failed = 0, 0
            for rank, idx in enumerate(tqdm(indices, desc="Converting"), start=1):
                try:
                    p0, p1 = int(pt_off[idx]), int(pt_off[idx + 1])
                    s0, s1 = int(st_off[idx]), int(st_off[idx + 1])
                    if p1 <= p0:
                        failed += 1
                        continue

                    # [修复] np.asarray 调用处 Pylance 报错被上方 : Any 注解解决
                    strokes = flat_to_strokes(np.asarray(d_points[p0:p1], dtype=np.float32), np.asarray(d_stroke_lengths[s0:s1], dtype=np.int32))
                    image = render_trajectory_to_image(strokes, image_height, padding, random_thickness, blur_kernel)
                    if image is None:
                        failed += 1
                        continue

                    h_target, w_target = calculate_dynamic_dims(image.shape[0], image.shape[1], max_area, min_size, stride)
                    image_resized = cv2.resize(image, (w_target, h_target), interpolation=cv2.INTER_AREA)

                    success, encoded_img = cv2.imencode('.png', image_resized)
                    if not success:
                        failed += 1
                        continue
                    
                    label = decode_label(d_labels[idx])
                    tokens_array, oovs = process_label(label, char2id)
                    total_oovs += oovs
                    
                    if len(tokens_array) <= 2: 
                        failed += 1
                        continue

                    out_name = f"{rank:06d}_idx{idx:07d}_{sanitize_for_filename(label)}_{h_target}x{w_target}.png"
                    
                    h5_writer.append(
                        img_bytes=encoded_img.flatten(),
                        tokens_array=tokens_array,
                        width=w_target,
                        height=h_target,
                        raw_label=label,
                        file_name=out_name
                    )

                    converted += 1
                    writer.writerow([out_name, idx, label, bucket_by_aspect(w_target/h_target), h_target, w_target])

                except Exception as e:
                    _ = e  # 防止 exception 被 Pylance 报未使用
                    failed += 1

    print(f"\n✅ 转换完成: {converted} | ❌ 失败/跳过(含空标签): {failed}")
    print(f"⚠️ OOV(未登录词) 替换总数: {total_oovs}")


# [修复] 返回类型更改为 Any 以消除 Pylance 强制检查
def parse_args() -> Any:
    parser = argparse.ArgumentParser(description="动态分辨率转换脚本")
    parser.add_argument("--vocab_path", default=r"C:\Projects\LatexProject\datasets\vocab.pkl", help="词表路径")
    parser.add_argument("--source_h5", default=None, help="原始轨迹 H5")
    parser.add_argument("--input_dir", default=r"C:\Projects\LatexProject\2Ddatasets\train", help="输入图像目录")
    parser.add_argument("--output_dir", default=r"C:\Projects\LatexProject\2Ddatasets\train_resized", help="输出目录")
    parser.add_argument("--max_area", type=int, default=98304, help="最大面积约束")
    parser.add_argument("--stride", type=int, default=16, help="CNN 步长对齐")
    parser.add_argument("--min_size", type=int, default=32, help="最小允许边长")
    parser.add_argument("--use_buckets", action="store_true", help="按长宽比分桶存储")
    parser.add_argument("--sample_size", type=int, default=None, help="采样数量")
    parser.add_argument("--seed", type=int, default=2026, help="随机种子")
    parser.add_argument("--print_limit", type=int, default=50, help="打印日志限制")
    parser.add_argument("--h5_output", default=None, help="导出 H5 文件路径")
    parser.add_argument("--no_save_images", action="store_true", help="仅写 H5")
    parser.add_argument("--image_height", type=int, default=192)
    parser.add_argument("--padding", type=int, default=4)
    parser.add_argument("--blur_kernel", type=int, default=1)
    parser.add_argument("--fixed_thickness", action="store_true")
    
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    print(f"Loading vocabulary from {args.vocab_path}...")
    with open(args.vocab_path, "rb") as f:
        char2id = pickle.load(f)["char2id"]


    if not args.h5_output: 
        raise ValueError("source_h5 模式必须提供 --h5_output")
    convert_source_h5_to_resized_h5(
        source_h5=args.source_h5, h5_output=args.h5_output, output_dir=args.output_dir,
        char2id=char2id, max_area=args.max_area, stride=args.stride, min_size=args.min_size,
        sample_size=args.sample_size, seed=args.seed, print_limit=args.print_limit,
        image_height=args.image_height, padding=args.padding,
        random_thickness=not args.fixed_thickness, blur_kernel=args.blur_kernel,
        save_images=not args.no_save_images, use_buckets=args.use_buckets,
    )

if __name__ == "__main__":
    main()

# 使用： python tools/data_prep/convert_dynamic_resolution.py --source_h5 "C:\Projects\LatexProject\datasets\synthetic.h5"   --h5_output "C:\Projects\LatexProject\2Ddatasets\synthetic.h5"   --output_dir "C:\Projects\LatexProject\2Ddatasets\synthetic_meta"   --no_save_images