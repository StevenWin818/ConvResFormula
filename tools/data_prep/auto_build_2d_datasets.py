"""
全自动 2D HDF5 数据集构建管线 (Auto 2D Pipeline)
包含：数据驱动的 Scale 分析 + 恒定物理尺度渲染 + 纯 Padding 步长对齐 + H5 高效存储
"""
import csv
import os
import math
import pickle
import numpy as np
import h5py
import cv2
from tqdm import tqdm
from typing import List, Tuple, Dict, Any

# 尝试导入 tokenizer
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))
try:
    from src.data.tokenizer import tokenize_latex
except ImportError:
    print("⚠️ 无法导入 tokenize_latex，请检查路径。")

# ==========================================
# 核心算子定义
# ==========================================

def get_optimal_scale(h5_path: str, target_pixel_height: int = 64, sample_limit: int = 50000) -> float:
    """第一步：数据驱动，提取最优全局渲染缩放比"""
    heights = []
    with h5py.File(h5_path, 'r') as f:
        pt_off = np.asarray(f["sample_point_offsets"])
        d_points = f["points"]
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
    scale = target_pixel_height / p95
    return float(scale)


def render_with_fixed_scale(strokes: List[np.ndarray], fixed_scale: float, padding: int = 4) -> np.ndarray:
    """第二步：使用全局固定缩放比进行物理真实性渲染"""
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
    thickness = 2 # 保持笔画粗细恒定

    for stroke in valid_strokes:
        scaled = np.zeros_like(stroke, dtype=np.int32)
        scaled[:, 0] = ((stroke[:, 0] - min_x) * fixed_scale).astype(np.int32) + padding
        scaled[:, 1] = ((stroke[:, 1] - min_y) * fixed_scale).astype(np.int32) + padding
        if len(scaled) == 1:
            cv2.circle(canvas, tuple(scaled[0]), radius=1, color=255, thickness=-1)
        else:
            cv2.polylines(canvas, [scaled.reshape((-1, 1, 2))], isClosed=False, color=255, thickness=thickness, lineType=cv2.LINE_AA)

    return canvas


def pad_to_stride(image: np.ndarray, stride: int = 16, min_size: int = 32) -> np.ndarray:
    """第三步：纯 Padding 对齐 CNN 步长，绝不拉伸图像"""
    h, w = image.shape
    target_h = max(min_size, int(math.ceil(h / stride) * stride))
    target_w = max(min_size, int(math.ceil(w / stride) * stride))
    pad_h = target_h - h
    pad_w = target_w - w
    if pad_h > 0 or pad_w > 0:
        image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
    return image


def process_label(label_str: str, char2id: dict) -> np.ndarray:
    tokens = tokenize_latex(label_str)
    ids = [1] + [char2id.get(t, 0) for t in tokens] + [2]
    return np.array(ids, dtype=np.int32)

# ==========================================
# 主流处理管道
# ==========================================

def build_dataset(name: str, source_h5: str, target_h5: str, char2id: dict, target_h: int = 64):
    print(f"\n" + "="*50)
    print(f"🚀 开始构建数据集: {name}")
    print(f"📁 来源: {source_h5}")
    print(f"💾 输出: {target_h5}")
    
    if not os.path.exists(source_h5):
        print(f"❌ 找不到源文件 {source_h5}，跳过。")
        return

    # 1. 自动测算最优缩放比
    print("🔍 正在提取全局最优物理缩放比 (Fixed Scale)...")
    scale = get_optimal_scale(source_h5, target_pixel_height=target_h)
    print(f"✅ 计算完成！采用全局缩放比: {scale:.4f}")

    # 2. 准备写入目标文件
    os.makedirs(os.path.dirname(target_h5), exist_ok=True)

    csv_path = target_h5.replace(".h5", "_metadata.csv")
    
    with h5py.File(source_h5, 'r') as src, \
         h5py.File(target_h5, 'w') as dst, \
         open(csv_path, 'w', newline='', encoding='utf-8') as f_csv:
        
        writer = csv.writer(f_csv)
        writer.writerow(["file_name", "h5_idx", "label", "target_h", "target_w"])
        pt_off = np.asarray(src["sample_point_offsets"])
        st_off = np.asarray(src["sample_stroke_offsets"])
        d_points = src["points"]
        d_stroke_lengths = src["stroke_lengths"]
        d_labels = src["labels"]
        
        total_samples = len(pt_off) - 1
        
        # 初始化目标 Dataset
        dt_bytes = h5py.vlen_dtype(np.uint8)
        dt_ints = h5py.vlen_dtype(np.int32)
        str_dtype = h5py.string_dtype(encoding="utf-8")
        
        dst.create_dataset("images", shape=(total_samples,), maxshape=(None,), dtype=dt_bytes, chunks=True)
        dst.create_dataset("labels", shape=(total_samples,), maxshape=(None,), dtype=dt_ints, chunks=True)
        dst.create_dataset("widths", shape=(total_samples,), maxshape=(None,), dtype=np.int32, chunks=True)
        dst.create_dataset("heights", shape=(total_samples,), maxshape=(None,), dtype=np.int32, chunks=True)
        dst.create_dataset("raw_labels", shape=(total_samples,), maxshape=(None,), dtype=str_dtype, chunks=True)
        
        valid_count = 0
        
        for idx in tqdm(range(total_samples), desc=f"📦 渲染与打包 {name}"):
            p0, p1 = int(pt_off[idx]), int(pt_off[idx + 1])
            s0, s1 = int(st_off[idx]), int(st_off[idx + 1])
            
            if p1 <= p0: continue
            
            points = np.asarray(d_points[p0:p1], dtype=np.float32)
            lengths = np.asarray(d_stroke_lengths[s0:s1], dtype=np.int32)
            
            # 分割笔画
            strokes = []
            cursor = 0
            for l in lengths:
                if l > 0: strokes.append(points[cursor:cursor+l])
                cursor += l
                
            # 渲染 -> Padding -> 编码
            img = render_with_fixed_scale(strokes, fixed_scale=scale)
            if img is None: continue
            
            img_aligned = pad_to_stride(img)
            success, encoded = cv2.imencode('.png', img_aligned)
            if not success: continue
            
            # 处理标签
            raw_label = d_labels[idx].decode("utf-8") if isinstance(d_labels[idx], bytes) else str(d_labels[idx])
            token_ids = process_label(raw_label, char2id)
            if len(token_ids) <= 2: continue
            
            # 写入 H5
            h, w = img_aligned.shape
            dst["images"][valid_count] = encoded.flatten()
            dst["labels"][valid_count] = token_ids
            dst["heights"][valid_count] = h
            dst["widths"][valid_count] = w
            dst["raw_labels"][valid_count] = raw_label

            file_name = f"idx_{valid_count:07d}_{h}x{w}.png"
            writer.writerow([file_name, valid_count, raw_label, h, w])
            
            valid_count += 1
            
        # 裁剪掉无效数据的空间
        for key in ["images", "labels", "widths", "heights", "raw_labels"]:
            dst[key].resize((valid_count,))
            
        dst.attrs["num_samples"] = valid_count
        print(f"🎉 {name} 构建成功！有效样本数: {valid_count} / {total_samples}")


if __name__ == "__main__":
    # === 1. 加载词表 ===
    VOCAB_PATH = r"C:\Projects\LatexProject\datasets\vocab.pkl"
    print(f"加载词表: {VOCAB_PATH}")
    with open(VOCAB_PATH, "rb") as f:
        char2id = pickle.load(f)["char2id"]

    # === 2. 配置你的 4 份数据集来源与目标位置 ===
    # 格式: (数据集代号, 原始1D文件路径, 目标2D文件路径, 期望的目标高度)
    # target_h = 64 是非常适合 CNN 提取特征的公式主体高度
    DATASETS_TO_BUILD = [
        ("手写训练集 (Train)", 
         r"C:\Projects\LatexProject\datasets\train.h5", 
         r"C:\Projects\LatexProject\2Ddatasets\train.h5", 96),
         
        ("手写验证集 (Val)", 
         r"C:\Projects\LatexProject\datasets\val.h5", 
         r"C:\Projects\LatexProject\2Ddatasets\val.h5", 96),
         
        ("合成长公式 (Synthetic)", 
         r"C:\Projects\LatexProject\datasets\synthetic.h5", 
         r"C:\Projects\LatexProject\2Ddatasets\synthetic.h5", 96),
         
        ("单字符集 (Symbols)", 
         r"C:\Projects\LatexProject\datasets\symbols.h5", 
         r"C:\Projects\LatexProject\2Ddatasets\symbols.h5", 96),
    ]

    # === 3. 一键执行全自动流程 ===
    for name, src, dst, target_h in DATASETS_TO_BUILD:
        build_dataset(name, src, dst, char2id, target_h)
        
    print("\n所有 2D 数据集已全部构建完毕，可以开始 Phase 2 训练了！")