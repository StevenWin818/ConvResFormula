"""
全自动 2D HDF5 数据集构建管线 (适配 ConvNeXt-V2 + BPE Tokenizer)
包含：数据驱动的 Scale 分析 + BPE 切词 + 32 步长 Padding 对齐 + H5 高效存储
"""

import os
import csv
import math
from typing import Any

import numpy as np
import h5py
import cv2
from tqdm import tqdm
from tokenizers import Tokenizer
import re

# ==========================================
# 文本清洗逻辑
# ==========================================
def normalize_sub_sup(latex_str):
    if not latex_str: return latex_str
    pattern = r'([_^])([a-zA-Z0-9])'
    return re.sub(pattern, r'\1{\2}', latex_str)

def clean_latex_for_encoding(latex_str):
    if not isinstance(latex_str, str) or not latex_str.strip(): return ""
    latex_str = normalize_sub_sup(latex_str)
    
    latex_str = latex_str.replace(r"/dots", r"\dots ")
    latex_str = latex_str.replace(r"...", r"\dots ")
    latex_str = latex_str.replace(r"\not =", r"\ne ")
    
    latex_str = re.sub(r'\^\s*\{?\s*\\prime\s*\}?', "'", latex_str)
    latex_str = latex_str.replace(r"\prime", "'")
    latex_str = re.sub(r"\\left(?![a-zA-Z])", "", latex_str)
    latex_str = re.sub(r"\\right(?![a-zA-Z])", "", latex_str)
    
    spaced_funcs = ["sinh", "cosh", "tanh", "log", "sin", "cos", "tan", "lim", "exp", "max", "min"]
    for sf in spaced_funcs:
        spaced_pattern = r"(?:\s|\\[,;!]|\\quad|\\qquad)*".join(list(sf))
        pattern = rf"(?<![a-zA-Z\\]){spaced_pattern}(?![a-zA-Z])"
        latex_str = re.sub(pattern, r"\\" + sf, latex_str)
        
    # Alias Map 同义词映射
    alias_map = {
        r"\leq": r"\le", r"\geq": r"\ge", r"\neq": r"\ne",
        r"\widehat": r"\hat", r"\widetilde": r"\tilde"
    }
    for k, v in alias_map.items():
        latex_str = latex_str.replace(k, v)
        
    return latex_str

# ==========================================
# 核心视觉算子定义
# ==========================================

def get_optimal_scale(h5_path: str, target_pixel_height: int = 96, sample_limit: int = 50000) -> float:
    """提取最优全局渲染缩放比"""
    heights = []
    with h5py.File(h5_path, 'r') as f:
        pt_off = np.asarray(f["sample_point_offsets"])
        d_points: Any = f["points"]
        total = len(pt_off) - 1
        limit = min(total, sample_limit)
        
        np.random.seed(42)
        indices = np.random.choice(total, limit, replace=False)
        
        for idx in indices:
            p0, p1 = int(pt_off[idx]), int(pt_off[idx + 1])
            if p1 > p0:
                points = np.asarray(d_points[p0:p1], dtype=np.float32)
                if len(points) > 0:
                    h = float(np.max(points[:, 1]) - np.min(points[:, 1]))
                    if h > 1e-5: heights.append(h)
                    
    if not heights: return 1.0
    p95 = np.percentile(heights, 95)
    return float(target_pixel_height / p95)

def render_with_fixed_scale(strokes: list, fixed_scale: float, padding: int = 4) -> np.ndarray | None:
    valid_strokes = [s for s in strokes if s is not None and len(s) > 0]
    if not valid_strokes: return None

    all_points = np.vstack(valid_strokes)
    min_x, min_y = np.min(all_points, axis=0)
    max_x, max_y = np.max(all_points, axis=0)

    width = float(max_x - min_x)
    height = float(max_y - min_y)

    target_height = max(int(round(height * fixed_scale)) + 2 * padding, 2 * padding + 1)
    target_width = max(int(round(width * fixed_scale)) + 2 * padding, 2 * padding + 1)
    
    canvas = np.zeros((target_height, target_width), dtype=np.uint8)

    for stroke in valid_strokes:
        scaled = np.zeros_like(stroke, dtype=np.int32)
        scaled[:, 0] = ((stroke[:, 0] - min_x) * fixed_scale).astype(np.int32) + padding
        scaled[:, 1] = ((stroke[:, 1] - min_y) * fixed_scale).astype(np.int32) + padding
        if len(scaled) == 1:
            cv2.circle(canvas, tuple(scaled[0]), radius=1, color=255, thickness=-1)
        else:
            cv2.polylines(canvas, [scaled.reshape((-1, 1, 2))], isClosed=False, color=255, thickness=2, lineType=cv2.LINE_AA)
    return canvas

def pad_to_stride(image: np.ndarray, stride: int = 32, min_size: int = 32) -> np.ndarray:
    """纯 Padding 对齐 CNN 步长 (ConvNeXt-V2 要求 32 倍数)"""
    h, w = image.shape
    target_h = max(min_size, int(math.ceil(h / stride) * stride))
    target_w = max(min_size, int(math.ceil(w / stride) * stride))
    pad_h = target_h - h
    pad_w = target_w - w
    if pad_h > 0 or pad_w > 0:
        image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
    return image

# ==========================================
# 主流处理管道
# ==========================================

def build_dataset(name: str, source_h5: str, target_h5: str, tokenizer: Tokenizer, target_h: int = 128):
    print(f"\n" + "="*50)
    print(f"· 开始构建数据集: {name}")
    print(f"· 来源: {source_h5}")
    
    if not os.path.exists(source_h5):
        print(f"❌ 找不到源文件 {source_h5}，跳过。")
        return

    scale = get_optimal_scale(source_h5, target_pixel_height=target_h)
    print(f"✅ 自动采用全局缩放比: {scale:.4f} (基准高度: {target_h})")

    os.makedirs(os.path.dirname(target_h5), exist_ok=True)
    csv_path = target_h5.replace(".h5", "_metadata.csv")
    
    # 提取 BPE 的特殊 Token
    bos_id = tokenizer.token_to_id("[BOS]")
    eos_id = tokenizer.token_to_id("[EOS]")
    
    with h5py.File(source_h5, 'r') as src, \
         h5py.File(target_h5, 'w') as dst, \
         open(csv_path, 'w', newline='', encoding='utf-8') as f_csv:
         
        pt_off = np.asarray(src["sample_point_offsets"])
        st_off = np.asarray(src["sample_stroke_offsets"])
        d_points: Any = src["points"]
        d_stroke_lengths: Any = src["stroke_lengths"]
        d_labels: Any = src["labels"]
        
        total_samples = len(pt_off) - 1
        
        # [修复 H5 Resize 支持] 加入 maxshape=(None,) 和 chunks=True
        dt_bytes = h5py.vlen_dtype(np.dtype(np.uint8))
        dt_ints = h5py.vlen_dtype(np.dtype(np.int32))
        str_dtype = h5py.string_dtype(encoding="utf-8")
        
        dst.create_dataset("images", shape=(total_samples,), maxshape=(None,), dtype=dt_bytes, chunks=True)
        dst.create_dataset("labels", shape=(total_samples,), maxshape=(None,), dtype=dt_ints, chunks=True)
        dst.create_dataset("widths", shape=(total_samples,), maxshape=(None,), dtype=np.int32, chunks=True)
        dst.create_dataset("heights", shape=(total_samples,), maxshape=(None,), dtype=np.int32, chunks=True)
        dst.create_dataset("raw_labels", shape=(total_samples,), maxshape=(None,), dtype=str_dtype, chunks=True)

        images_ds: Any = dst["images"]
        labels_ds: Any = dst["labels"]
        widths_ds: Any = dst["widths"]
        heights_ds: Any = dst["heights"]
        raw_labels_ds: Any = dst["raw_labels"]
        
        writer = csv.writer(f_csv)
        writer.writerow(["file_name", "h5_idx", "label", "target_h", "target_w"])
        
        valid_count = 0
        
        for idx in tqdm(range(total_samples), desc=f"· 打包 {name}"):
            p0, p1 = int(pt_off[idx]), int(pt_off[idx + 1])
            s0, s1 = int(st_off[idx]), int(st_off[idx + 1])
            
            if p1 <= p0: continue
            points = np.asarray(d_points[p0:p1], dtype=np.float32)
            lengths = np.asarray(d_stroke_lengths[s0:s1], dtype=np.int32)
            
            strokes = []
            cursor = 0
            for l in lengths:
                if l > 0: strokes.append(points[cursor:cursor+l])
                cursor += l
                
            img = render_with_fixed_scale(strokes, fixed_scale=scale)
            if img is None: continue
            
            # 步长严格对齐 32
            img_aligned = pad_to_stride(img, stride=32)
            success, encoded = cv2.imencode('.png', img_aligned)
            if not success: continue
            
            # BPE 极速编码
            raw_label = d_labels[idx].decode("utf-8") if isinstance(d_labels[idx], bytes) else str(d_labels[idx])
            clean_label = clean_latex_for_encoding(raw_label)
            if not clean_label: continue
            
            bpe_encoded = tokenizer.encode(clean_label)
            token_ids = [bos_id] + bpe_encoded.ids + [eos_id]
            
            if len(token_ids) <= 2: continue # 只有首尾没有内容的抛弃
            
            # 写入
            h, w = img_aligned.shape
            images_ds[valid_count] = encoded.flatten()
            labels_ds[valid_count] = np.array(token_ids, dtype=np.int32)
            heights_ds[valid_count] = h
            widths_ds[valid_count] = w
            raw_labels_ds[valid_count] = raw_label
            
            file_name = f"idx_{valid_count:07d}_{h}x{w}.png"
            writer.writerow([file_name, valid_count, raw_label, h, w])
            
            valid_count += 1
            
        for dataset in [images_ds, labels_ds, widths_ds, heights_ds, raw_labels_ds]:
            dataset.resize((valid_count,))
            
        dst.attrs["num_samples"] = valid_count
        print(f"· {name} 构建成功！有效样本数: {valid_count} / {total_samples}")

if __name__ == "__main__":
    # === 1. 载入 BPE Tokenizer ===
    TOKENIZER_PATH = r"C:\Projects\LatexProject\ConvResFormula\tokenizer_bpe.json"
    print(f"· 正在加载 BPE Tokenizer: {TOKENIZER_PATH}")
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)

    DEFAULT_TARGET_H = 128

    # === 2. 配置数据集  ===
    DATASETS_TO_BUILD = [
        ("手写训练集 (Train)", 
         r"C:\Projects\LatexProject\datasets\train.h5", 
         r"C:\Projects\LatexProject\ConvResFormula\datasets\train.h5", DEFAULT_TARGET_H),
         
        ("手写验证集 (Val)", 
         r"C:\Projects\LatexProject\datasets\val.h5", 
         r"C:\Projects\LatexProject\ConvResFormula\datasets\val.h5", DEFAULT_TARGET_H),
         
        ("合成长公式 (Synthetic)", 
         r"C:\Projects\LatexProject\datasets\synthetic.h5", 
         r"C:\Projects\LatexProject\ConvResFormula\datasets\synthetic.h5", DEFAULT_TARGET_H),
         
        ("单字符集 (Symbols)", 
         r"C:\Projects\LatexProject\datasets\symbols.h5", 
         r"C:\Projects\LatexProject\ConvResFormula\datasets\symbols.h5", DEFAULT_TARGET_H),
    ]

    for name, src, dst, target_h in DATASETS_TO_BUILD:
        build_dataset(name, src, dst, tokenizer, target_h)
        
    print("\n✅ ConvNeXt-V2 适配版 H5 数据集全部构建完毕！")