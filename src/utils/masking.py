"""掩码相关工具函数。"""

import torch
import torch.nn.functional as F


def build_memory_padding_mask(
    images: torch.Tensor,
    pool_kernel: int = 32,
    pool_stride: int = 32,
    empty_threshold: float = 1e-5,
) -> torch.Tensor:
    """
    基于输入图像构造视觉 memory 的 padding mask。

    参数:
        images: 单通道归一化图像张量，shape=[B, 1, H, W]。

    返回:
        Bool Tensor, shape=[B, T]
        True 表示该视觉 token 对应区域可视为无效填充。
    """
    if images.dim() != 4:
        raise ValueError(f"images 维度必须为 4 (B,C,H,W)，当前为 {tuple(images.shape)}")
    if images.size(1) != 1:
        raise ValueError(
            f"images 必须为单通道张量，期望 shape=[B,1,H,W]，当前为 {tuple(images.shape)}"
        )

    batch_size = images.size(0)
    downsampled_mask = F.max_pool2d(images, kernel_size=pool_kernel, stride=pool_stride)
    return downsampled_mask.view(batch_size, -1) <= empty_threshold
