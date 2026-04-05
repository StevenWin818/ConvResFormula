"""
统一 Dataset：支持 HDF5 懒加载与 BPE Tokenizer。
"""
import math
import h5py
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
from tokenizers import Tokenizer
from typing import Tuple, Dict, Any, Optional

def calculate_dynamic_dims(h_orig: int, w_orig: int, max_area: int = 98304, min_size: int = 32, stride: int = 32) -> Tuple[int, int]:
    """
    计算基于面积约束且符合 ConvNeXt-V2 步长要求的动态分辨率。
    注意：ConvNeXt 经过 5 次下采样，通常要求步长为 32。
    """
    aspect_ratio = w_orig / h_orig
    target_h = math.sqrt(max_area / aspect_ratio)
    target_w = target_h * aspect_ratio
    
    def align_to_stride(val):
        aligned = int(round(val / stride) * stride)
        return max(min_size, aligned)
    
    aligned_h = align_to_stride(target_h)
    aligned_w = align_to_stride(target_w)
            
    return aligned_h, aligned_w

class MLMFormulaDataset(Dataset):
    def __init__(self, h5_path: str, tokenizer_path: str, max_area: int = 98304, enable_augment: bool = False):
        """
        Args:
            h5_path: HDF5 数据集路径
            tokenizer_path: BPE 分词器 JSON 文件路径
            max_area: 最大图像面积限制
            enable_augment: 是否启用在线增强（建议仅训练集开启）
        """
        self.h5_path = h5_path
        self.max_area = max_area
        self.enable_augment = bool(enable_augment)
        
        # 加载 BPE 分词器
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        
        # 预先获取数据集长度，避免在主进程保持 h5 文件句柄
        with h5py.File(self.h5_path, 'r') as f:
            labels_ds: Any = f['labels']
            self.length = int(labels_ds.shape[0])
            if 'widths' in f and 'heights' in f:
                widths_ds: Any = f['widths']
                heights_ds: Any = f['heights']
                widths = np.asarray(widths_ds[:], dtype=np.float32)
                heights = np.asarray(heights_ds[:], dtype=np.float32)
                heights = np.maximum(heights, 1.0)
                self.aspect_ratios = widths / heights

                # 预估动态缩放后的面积，作为分桶的内存代理特征。
                target_h = np.sqrt(float(self.max_area) / self.aspect_ratios)
                target_w = target_h * self.aspect_ratios
                aligned_h = np.maximum(32.0, np.round(target_h / 32.0) * 32.0)
                aligned_w = np.maximum(32.0, np.round(target_w / 32.0) * 32.0)
                self.resized_heights = aligned_h.astype(np.float32)
                self.resized_widths = aligned_w.astype(np.float32)
                self.resized_areas = aligned_h * aligned_w
            else:
                # 回退：缺少宽高元数据时使用常数，避免分桶流程崩溃
                self.aspect_ratios = np.ones((self.length,), dtype=np.float32)
                self.resized_heights = np.full((self.length,), 256.0, dtype=np.float32)
                self.resized_widths = np.full((self.length,), 384.0, dtype=np.float32)
                self.resized_areas = np.ones((self.length,), dtype=np.float32)
            
        # 工作进程专用的 h5 句柄 (避免多进程 Dataloader 发生死锁)
        self.h5_file = None
        self.images_ds: Optional[Any] = None
        self.labels_ds: Optional[Any] = None
        self.raw_labels_ds: Optional[Any] = None

    def __len__(self) -> int:
        return self.length

    @staticmethod
    def _decode_raw_label(raw: Any) -> str:
        if isinstance(raw, bytes):
            return raw.decode('utf-8')
        if isinstance(raw, np.bytes_):
            return bytes(raw).decode('utf-8')
        return str(raw)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # 懒加载：在每个 worker 首次访问时打开文件
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r')
            self.images_ds = self.h5_file['images']
            self.labels_ds = self.h5_file['labels']
            self.raw_labels_ds = self.h5_file['raw_labels'] if 'raw_labels' in self.h5_file else None

        if self.images_ds is None or self.labels_ds is None:
            raise RuntimeError(f"HDF5 数据集未正确初始化: {self.h5_path}")
            
        # 1. 读取图像与标签
        img_data_any: Any = self.images_ds[idx]
        img_bytes = bytes(np.asarray(img_data_any, dtype=np.uint8))
        label_item: Any = self.labels_ds[idx]

        # 新版 H5: labels 为 token id 序列 (vlen int32)
        # 兼容旧版 H5: labels 为 utf-8 文本
        if isinstance(label_item, np.ndarray):
            input_ids = label_item.astype(np.int64).tolist()
        elif isinstance(label_item, (bytes, np.bytes_)):
            label_str = self._decode_raw_label(label_item)
            input_ids = self.tokenizer.encode(label_str).ids
        else:
            # 若 labels 是标量类型，优先尝试 raw_labels 作为文本来源
            if self.raw_labels_ds is not None:
                raw_label = self._decode_raw_label(self.raw_labels_ds[idx])
                input_ids = self.tokenizer.encode(raw_label).ids
            else:
                input_ids = [int(label_item)]
        
        # 2. 图像解码与动态分辨率缩放
        img_buffer = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_buffer, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"图像解码失败: idx={idx}, path={self.h5_path}")

        h_orig, w_orig = img.shape
        target_h, target_w = calculate_dynamic_dims(h_orig, w_orig, max_area=self.max_area, stride=32)
        
        if h_orig != target_h or w_orig != target_w:
            img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)

        # 训练专用在线增强：轻量噪声与对比度扰动，降低过拟合
        if self.enable_augment:
            img_float = img.astype(np.float32)

            if np.random.rand() < 0.2:
                noise = np.random.randn(*img_float.shape) * np.random.uniform(2.0, 8.0)
                img_float = img_float + noise

            if np.random.rand() < 0.3:
                alpha = np.random.uniform(0.7, 1.2)
                img_float = img_float * alpha

            img = np.clip(img_float, 0.0, 255.0).astype(np.uint8)
            
        # 转换为 Tensor，归一化到 [0, 1] 并增加通道维度 [1, H, W]
        img_tensor = torch.from_numpy(img).float().unsqueeze(0) / 255.0
        
        return {
            "image": img_tensor,
            "input_ids": input_ids
        }