"""
ConvNeXt-V2 视觉编码器接口。
依赖: pip install timm
"""

import math
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import timm

class PositionalEncoding2D(nn.Module):
    """
    2D 正余弦位置编码 (Sine-Cosine 2D PE)
    非常适合动态输入尺寸的图像，不需要像 learnable PE 那样做插值。
    """
    def __init__(self, d_model: int, max_h: int = 128, max_w: int = 1024):
        super().__init__()
        self.d_model = d_model
        assert d_model % 2 == 0, "d_model 必须是偶数"
        
        # d_model 会被平分为两半，一半给 Y 轴(高度)，一半给 X 轴(宽度)
        d_model_half = d_model // 2
        
        # 预计算最大尺寸的位置编码
        pe = torch.zeros(d_model, max_h, max_w)
        y_position = torch.arange(0, max_h).unsqueeze(1).float()
        x_position = torch.arange(0, max_w).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model_half, 2).float() * -(math.log(10000.0) / d_model_half))
        
        # Y轴 (前一半通道): [H, K] -> [K, H, W]
        y_embed = y_position * div_term
        pe[0:d_model_half:2, :, :] = torch.sin(y_embed).transpose(0, 1).unsqueeze(2).repeat(1, 1, max_w)
        pe[1:d_model_half:2, :, :] = torch.cos(y_embed).transpose(0, 1).unsqueeze(2).repeat(1, 1, max_w)

        # X轴 (后一半通道): [W, K] -> [K, H, W]
        x_embed = x_position * div_term
        pe[d_model_half::2, :, :] = torch.sin(x_embed).transpose(0, 1).unsqueeze(1).repeat(1, max_h, 1)
        pe[d_model_half+1::2, :, :] = torch.cos(x_embed).transpose(0, 1).unsqueeze(1).repeat(1, max_h, 1)
        
        # 注册为 buffer，这样它就不会成为模型可训练参数，但会随模型保存和移动(GPU/CPU)
        self.register_buffer('pe', pe)
        # 显式声明类型给 Pylance
        self.pe: torch.Tensor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [Batch, Channels, H, W]
        """
        _, _, h, w = x.size()
        # 截取当前实际特征图大小的 PE，并加上去
        return x + self.pe[:, :h, :w].unsqueeze(0)


class ConvNeXtV2Encoder(nn.Module):
    """
    基于 ConvNeXt-V2 的 2D 视觉主干网络。
    支持灰度图(in_chans=1)输入，输出展平后的序列 [Batch, Seq_Len, d_model]
    """
    def __init__(self, model_name: str = 'convnextv2_pico', pretrained: bool = True, d_model: int = 512, in_chans: int = 1):
        super().__init__()
        
        # 1. 使用 timm 拉取预训练骨干网络
        # features_only=True 表示我们不要它的全连接层和分类头，只要输出的特征图
        print(f"🚀 初始化视觉骨干: {model_name} (Pretrained={pretrained})")
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=in_chans,    # 直接支持单通道灰度图！timm会自动处理预训练权重的合并
            features_only=True,
            out_indices=(-1,)     # 只取最后一次下采样（32倍）的特征图
        )
        
        # 获取 ConvNeXt 最后一层的输出通道数 (pico 是 512, nano 是 640)
        # 强制转换为 int 来消除 Pylance 的类型推断噪音
        feature_info: Any = self.backbone.feature_info
        backbone_out_channels: int = int(feature_info[-1]['num_chs'])
        
        # 2. 维度投影层 (Projection)
        # 将 ConvNeXt 的输出通道数映射为 Decoder 需要的 d_model
        self.proj = nn.Conv2d(backbone_out_channels, d_model, kernel_size=1)
        
        # 3. 2D 位置编码
        # 注意：这里的 max_h 和 max_w 是指“特征图”的最大尺寸。
        self.pos_encoder = PositionalEncoding2D(d_model=d_model, max_h=128, max_w=1024)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [Batch, 1, H, W] 归一化后的图像张量
            mask: (可选) 目前不需要，保持接口兼容
        Returns:
            features: [Batch, Seq_Len, d_model] 给 Decoder 用的序列
            memory_padding_mask: [Batch, Seq_Len] 布尔掩码 (True 表示该位置是纯黑 Padding)
        """
        import torch.nn.functional as F

        # 1. 骨干网络提取二维特征图
        # 输出尺寸: [Batch, C, H_feat, W_feat]
        backbone_output: Any = self.backbone(x)
        features: torch.Tensor = backbone_output[0] 
        
        # 2. 投影到 d_model
        features = self.proj(features)
        
        # 3. 注入二维物理绝对位置信息
        features = self.pos_encoder(features)
        
        # 4. 构造视觉 memory 的 padding mask
        b, c, h_feat, w_feat = features.size()
        with torch.no_grad():
            downsampled_mask = F.max_pool2d(x, kernel_size=32, stride=32)
            memory_padding_mask = (downsampled_mask.view(b, -1) <= 1e-5)

        # 5. 展平为 1D 序列 (从 2D 图变为文字序列一样的排布)
        # [Batch, d_model, H_feat, W_feat] -> [Batch, d_model, H_feat * W_feat] -> [Batch, Seq_Len, d_model]
        features = features.flatten(2).permute(0, 2, 1)
        
        return features, memory_padding_mask