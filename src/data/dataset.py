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

import os

# 禁用 albumentations 的网络版本检查，避免多进程 DataLoader 启动时严重超时与警告
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

try:
    import albumentations as A
except ImportError:
    A = None

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

class FormulaDataset(Dataset):
    def __init__(
        self,
        h5_path: str,
        tokenizer_path: str,
        max_area: int = 98304,
        enable_augment: bool = False,
        augment_config: Optional[Dict[str, Any]] = None,
        enable_extreme_augment: bool = False,
        extreme_augment_config: Optional[Dict[str, Any]] = None,
    ):
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
        self.augment_config = augment_config if isinstance(augment_config, dict) else {}
        self.enable_extreme_augment = bool(enable_extreme_augment)
        self.extreme_augment_config = extreme_augment_config if isinstance(extreme_augment_config, dict) else {}
        
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
        self.albu_transform = self._build_albu_transform()

    def __len__(self) -> int:
        return self.length

    @staticmethod
    def _decode_raw_label(raw: Any) -> str:
        if isinstance(raw, bytes):
            return raw.decode('utf-8')
        if isinstance(raw, np.bytes_):
            return bytes(raw).decode('utf-8')
        return str(raw)

    def _build_albu_transform(self):
        if (not self.enable_extreme_augment) or A is None:
            return None

        grid_cfg = self.extreme_augment_config.get("grid_distortion", {})
        elastic_cfg = self.extreme_augment_config.get("elastic_transform", {})

        transforms = [
            A.GridDistortion(
                num_steps=int(grid_cfg.get("num_steps", 5)),
                distort_limit=float(grid_cfg.get("distort_limit", 0.3)),
                p=float(grid_cfg.get("prob", 0.5)),
            ),
            A.ElasticTransform(
                alpha=float(elastic_cfg.get("alpha", 1.0)),
                sigma=float(elastic_cfg.get("sigma", 50.0)),
                p=float(elastic_cfg.get("prob", 0.5)),
            ),
        ]
        return A.Compose(transforms)

    def _apply_morphology(self, img: np.ndarray) -> np.ndarray:
        if not self.enable_extreme_augment:
            return img

        morph_cfg = self.extreme_augment_config.get("morphology", {})
        prob = float(morph_cfg.get("prob", 0.4))
        if np.random.rand() >= prob:
            return img

        kernel_size = max(1, int(morph_cfg.get("kernel_size", 3)))
        iterations = max(1, int(morph_cfg.get("iterations", 1)))
        ops = morph_cfg.get("ops", ["erode", "dilate"])
        if not isinstance(ops, list) or len(ops) == 0:
            ops = ["erode", "dilate"]

        op = str(np.random.choice(ops)).lower()
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        if op == "erode":
            return cv2.erode(img, kernel, iterations=iterations)
        return cv2.dilate(img, kernel, iterations=iterations)

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
        if len(img_buffer) == 0:
            print(f"⚠️ 警告: 发现空图像数据 idx={idx}, path={self.h5_path}，已自动跳过并替换为下一个样本。")
            return self.__getitem__((idx + 1) % self.length)

        img = cv2.imdecode(img_buffer, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"⚠️ 警告: 图像解码失败 idx={idx}, path={self.h5_path}，已自动跳过并替换为下一个样本。")
            return self.__getitem__((idx + 1) % self.length)

        h_orig, w_orig = img.shape
        target_h, target_w = calculate_dynamic_dims(h_orig, w_orig, max_area=self.max_area, stride=32)
        
        if h_orig != target_h or w_orig != target_w:
            img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)

        # 训练专用在线增强：轻量噪声与对比度扰动，降低过拟合
        if self.enable_augment:
            img_float = img.astype(np.float32)

            noise_cfg = self.augment_config.get("gaussian_noise", {})
            noise_prob = float(noise_cfg.get("prob", 0.2))
            sigma_min = float(noise_cfg.get("sigma_min", 2.0))
            sigma_max = float(noise_cfg.get("sigma_max", 8.0))
            if sigma_max < sigma_min:
                sigma_min, sigma_max = sigma_max, sigma_min

            if np.random.rand() < noise_prob:
                noise = np.random.randn(*img_float.shape) * np.random.uniform(sigma_min, sigma_max)
                img_float = img_float + noise

            contrast_cfg = self.augment_config.get("contrast", {})
            contrast_prob = float(contrast_cfg.get("prob", 0.3))
            alpha_min = float(contrast_cfg.get("alpha_min", 0.7))
            alpha_max = float(contrast_cfg.get("alpha_max", 1.2))
            if alpha_max < alpha_min:
                alpha_min, alpha_max = alpha_max, alpha_min

            if np.random.rand() < contrast_prob:
                alpha = np.random.uniform(alpha_min, alpha_max)
                img_float = img_float * alpha

            img = np.clip(img_float, 0.0, 255.0).astype(np.uint8)

        if self.enable_extreme_augment:
            if self.albu_transform is not None:
                img = self.albu_transform(image=img)["image"]
            img = self._apply_morphology(img)
            
        # 转换为 Tensor，归一化到 [0, 1] 并增加通道维度 [1, H, W]
        img_tensor = torch.from_numpy(img).float().unsqueeze(0) / 255.0
        
        return {
            "image": img_tensor,
            "input_ids": input_ids
        }