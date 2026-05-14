"""
ConvNeXt-V2 视觉编码器接口。
依赖: pip install timm
"""

import math
from typing import Any, Callable, Optional, Tuple, Union, cast

import torch
import torch.nn as nn
import timm
import re

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
        self.register_buffer('pe', pe, persistent=False)        # 显式声明类型给 Pylance
        self.pe: torch.Tensor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [Batch, Channels, H, W]
        """
        _, _, h, w = x.size()
        # 截取当前实际特征图大小的 PE，并加上去
        return x + self.pe[:, :h, :w].unsqueeze(0)


class RelativePositionBias2D(nn.Module):
    def __init__(self, num_heads, max_h=64, max_w=256):
        """
        二维相对位置偏置矩阵。
        max_h 和 max_w 需覆盖 Stride 16 下特征图的最大尺寸。
        """
        super().__init__()
        self.num_heads = num_heads
        self.max_h = max_h
        self.max_w = max_w
        self.bias_table = nn.Parameter(torch.zeros((2 * max_h - 1) * (2 * max_w - 1), num_heads))
        nn.init.trunc_normal_(self.bias_table, std=0.02)

    def forward(self, h, w, batch_size, device):
        # 计算相对坐标索引
        y = torch.arange(h, device=device)
        x = torch.arange(w, device=device)
        y_idx = (y.view(-1, 1) - y.view(1, -1)) + self.max_h - 1
        x_idx = (x.view(-1, 1) - x.view(1, -1)) + self.max_w - 1
        
        # 组合为一维索引: [H, H, W, W] -> [H, W, H, W] -> [H*W, H*W]
        idx = y_idx.view(h, h, 1, 1) * (2 * self.max_w - 1) + x_idx.view(1, 1, w, w)
        idx = idx.permute(0, 2, 1, 3).contiguous().view(h * w, h * w)

        max_valid_idx = (2 * self.max_h - 1) * (2 * self.max_w - 1) - 1
        idx = torch.clamp(idx, 0, max_valid_idx)

        # 提取偏置并调整维度: [H*W, H*W, num_heads] -> [num_heads, L, L]
        bias = self.bias_table[idx]
        bias = 8.0 * torch.tanh(bias / 8.0)
        bias = bias.permute(2, 0, 1).contiguous()

        # 适配 PyTorch MultiheadAttention 的 src_mask 形状要求: [B * num_heads, L, L]
        return bias.unsqueeze(0).repeat(batch_size, 1, 1, 1).view(batch_size * self.num_heads, h * w, h * w)


class ConvNeXtV2Encoder(nn.Module):
    """
    基于 ConvNeXt-V2 的 2D 视觉主干网络。
    支持灰度图(in_chans=1)输入，输出展平后的序列 [Batch, Seq_Len, d_model]
    """
    def __init__(
        self, 
        model_name: str = 'convnextv2_pico', 
        pretrained: bool = True, 
        d_model: int = 512, 
        ctc_vocab_size: int = 4001,
        in_chans: int = 1,
        drop_path_rate: float = 0.0, # 接收参数，默认为0保持向后兼容
        use_gradient_checkpointing: bool = True,
    ):
        super().__init__()
        self.downsample_stride = 16 # 改为输出 16 倍下采样特征以提高细节分辨率
        
        # 1. 使用 timm 拉取预训练骨干网络
        print(f"🚀 初始化视觉骨干: {model_name} (Pretrained={pretrained})")
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=in_chans,    # 恢复单通道灰度图，彻底废弃 CoordConv
            features_only=True,
            out_indices=(-2, -1),     # 提取 Stride 16 和 Stride 32 两层特征
            drop_path_rate=drop_path_rate,
        )

        set_grad_checkpointing = getattr(self.backbone, "set_grad_checkpointing", None)
        if callable(set_grad_checkpointing):
            set_grad_checkpointing_fn = cast(Callable[[bool], Any], set_grad_checkpointing)
            set_grad_checkpointing_fn(bool(use_gradient_checkpointing))
            if bool(use_gradient_checkpointing):
                print("✅ 视觉主干网络内部 Layer-level 梯度检查点已激活！")
            else:
                print("ℹ️ 视觉主干网络内部 Layer-level 梯度检查点已关闭。")
        
        feature_info: Any = self.backbone.feature_info
        ch_16 = int(feature_info[-2]['num_chs'])
        ch_32 = int(feature_info[-1]['num_chs'])
        
        # 2. 维度投影层与 FPN 融合
        self.proj_16 = nn.Conv2d(ch_16, d_model, kernel_size=1)
        self.proj_32 = nn.Conv2d(ch_32, d_model, kernel_size=1)
        self.fpn_fusion = nn.Conv2d(d_model, d_model, kernel_size=3, padding=1)
        
        # 安全初始化：确保在加载旧权重（没有 FPN 参数）时，FPN 模块等价于直通 (Identity)，防止输出随机乱码
        # 移除 proj_16 的全 0 初始化，让它使用默认的 Kaiming 初始化，配合高学习率快速收敛
        nn.init.dirac_(self.fpn_fusion.weight)
        nn.init.zeros_(self.fpn_fusion.bias)
        
        # 3. 2D 位置编码 (调整 max 尺寸以适应 stride 16)
        self.pos_encoder = PositionalEncoding2D(d_model=d_model, max_h=256, max_w=2048)

        # 4. 视觉自注意力层 (Transformer Encoder)
        # 在 Stride 16 扩展出 4 倍 Token 后，建立视觉 Token 间的全局上下文
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=2,
            enable_nested_tensor=False)
        
        # 5. 二维相对位置偏置 (Relative Position Bias)
        self.rel_pos_bias = RelativePositionBias2D(num_heads=8)

        # 6. BoW 辅助头
        self.bow_head = nn.Linear(d_model, int(ctc_vocab_size))

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_aux: bool = False,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        """
        Args:
            x: [Batch, 1, H, W] 归一化后的图像张量
            mask: (可选) 目前不需要，保持接口兼容
        Returns:
            features: [Batch, Seq_Len, d_model] 给 Decoder 用的序列
            memory_padding_mask: [Batch, Seq_Len] 布尔掩码 (True 表示该位置是纯黑 Padding)
        """
        import torch.nn.functional as F

        b, c, h, w = x.size()
        
        # 0. 移除破坏平移不变性的绝对 CoordConv，直接使用原图
        x_in = x

        # 1. 骨干提取 (多层特征)
        backbone_output: Any = self.backbone(x_in)
        feat_16: torch.Tensor = backbone_output[0] # Stride 16
        feat_32: torch.Tensor = backbone_output[1] # Stride 32
        
        # 2. FPN 融合到 Stride 16
        p_16 = self.proj_16(feat_16)
        p_32 = self.proj_32(feat_32)
        p_32_up = F.interpolate(p_32, size=p_16.shape[2:], mode='bilinear', align_corners=False)
        features = self.fpn_fusion(p_16 + p_32_up)
        
        # 3. 注入二维物理绝对位置信息
        features = self.pos_encoder(features)
        
        # 4. 构造视觉 memory 的 padding mask (按下采样步长 16)
        b_f, c_f, h_feat, w_feat = features.size()
        with torch.no_grad():
            downsampled_mask = F.max_pool2d(x, kernel_size=16, stride=16)
            memory_padding_mask = (downsampled_mask.view(b_f, -1) <= 1e-5)

        # 5. 展平为 1D 序列 (从 2D 图变为文字序列一样的排布)
        sigreg_embedding = features.flatten(2).permute(0, 2, 1)  # [B, Seq_Len, d_model]
        
        # 6. Visual DropToken (随机丢弃视觉序列，强制全局注意力)
        if self.training:
            # 以 12% 的概率丢弃 (keep_mask 为 True 表示保留)
            keep_mask = torch.rand(sigreg_embedding.shape[:2], device=sigreg_embedding.device) > 0.05
            # 缩放以保持期望值
            sigreg_embedding = (sigreg_embedding * keep_mask.unsqueeze(-1)) / 0.88
            
            # 【核心修复 1】将丢弃的 Token 同步到掩码中，防止“幽灵 Token”干扰自注意力
            # memory_padding_mask 中 True 表示无效位置
            memory_padding_mask = memory_padding_mask | (~keep_mask)

        # 7. 生成 2D 相对位置偏置
        attn_bias = self.rel_pos_bias(h_feat, w_feat, b_f, features.device)

        float_padding_mask = torch.zeros(
            memory_padding_mask.shape, 
            dtype=attn_bias.dtype, 
            device=memory_padding_mask.device
        )
        float_padding_mask.masked_fill_(memory_padding_mask, float('-inf'))

        # 8. 视觉自注意力层 (建立高分辨率视觉 Token 的全局上下文，同时注入相对位置先验)
        sigreg_embedding = self.transformer_encoder(
            sigreg_embedding, 
            mask=attn_bias,
            src_key_padding_mask=float_padding_mask
        )
        
        # 【核心修复 2】将 BoW 监督信号移动到 Transformer 之后，并修复平均池化
        bow_logits = None
        if return_aux:
            # 排除 padding 和被 Drop 的 Token (True 表示无效)
            valid_mask = (~memory_padding_mask).unsqueeze(-1).float() # [B, Seq_Len, 1]
            # 计算有效 Token 的平均池化 [B, d_model]
            global_feat = (sigreg_embedding * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1e-5)
            bow_logits = self.bow_head(global_feat)

        if return_aux and bow_logits is not None:
            return sigreg_embedding, memory_padding_mask, bow_logits, sigreg_embedding

        return sigreg_embedding, memory_padding_mask