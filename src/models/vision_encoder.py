"""
ConvNeXt-V2 视觉编码器接口。
依赖: pip install timm
"""

import math
from typing import Any, Callable, Optional, Tuple, Union, cast

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
        self.register_buffer('pe', pe, persistent=False)        # 显式声明类型给 Pylance
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
            in_chans=3,    # 强制 3 通道，容纳 CoordConv (Gray, X, Y)
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
        nn.init.zeros_(self.proj_16.weight)
        nn.init.zeros_(self.proj_16.bias)
        nn.init.dirac_(self.fpn_fusion.weight)
        nn.init.zeros_(self.fpn_fusion.bias)
        
        # 3. 2D 位置编码 (调整 max 尺寸以适应 stride 16)
        self.pos_encoder = PositionalEncoding2D(d_model=d_model, max_h=256, max_w=2048)

        # 预计算最大尺寸的坐标网格，用于 CoordConv (避免动态形状下 torch.linspace 触发编译回退)
        max_coord_h, max_coord_w = 2048, 2048
        y_pos = torch.arange(max_coord_h).view(1, 1, max_coord_h, 1)
        x_pos = torch.arange(max_coord_w).view(1, 1, 1, max_coord_w)
        self.register_buffer("y_pos", y_pos, persistent=False)
        self.register_buffer("x_pos", x_pos, persistent=False)
        self.y_pos: torch.Tensor
        self.x_pos: torch.Tensor

        # 4. BoW 辅助头
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
        
        # 0. 构造 CoordConv (兼容 torch.compile 的符号化形状推导)
        if c == 1:
            # 取出对应尺寸的预计算坐标，并做 -1 到 1 的归一化
            # (h - 1) 可能为 0，所以加 1e-5 防除零
            y_coords = (self.y_pos[:, :, :h, :w].expand(b, 1, h, w).float() / (h - 1 + 1e-5)) * 2.0 - 1.0
            x_coords = (self.x_pos[:, :, :h, :w].expand(b, 1, h, w).float() / (w - 1 + 1e-5)) * 2.0 - 1.0
            x_in = torch.cat([x, x_coords.to(x.dtype), y_coords.to(x.dtype)], dim=1)
        else:
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

        bow_logits = None
        if return_aux:
            # 全局平均池化得到 [B, d_model]，用于符号存在性预测
            global_feat = features.mean(dim=(2, 3))
            bow_logits = self.bow_head(global_feat)

        # 5. 展平为 1D 序列 (从 2D 图变为文字序列一样的排布)
        # [Batch, d_model, H_feat, W_feat] -> [Batch, d_model, H_feat * W_feat] -> [Batch, Seq_Len, d_model]
        sigreg_embedding = features.flatten(2).permute(0, 2, 1)  # [B, Seq_Len, d_model]
        
        if return_aux and bow_logits is not None:
            return sigreg_embedding, memory_padding_mask, bow_logits, sigreg_embedding

        return sigreg_embedding, memory_padding_mask