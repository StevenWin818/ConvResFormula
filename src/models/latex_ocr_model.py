"""
LaTeX OCR 主模型组装与 MLM 前向接口。
"""
import math
import torch
import torch.nn as nn
from typing import Optional

from .vision_encoder import ConvNeXtV2Encoder
from .text_decoder import AttnResTextDecoder

class PositionalEncoding1D(nn.Module):
    """标准的 1D 文本序列位置编码"""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        # 转为 [1, max_len, d_model] 适应 batch_first
        self.register_buffer('pe', pe.transpose(0, 1))
        self.pe: torch.Tensor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [Batch, SeqLen, d_model]"""
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class LatexOCRModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        pad_id: int = 0,
        vision_model_name: str = "convnextv2_pico",
        vision_pretrained: bool = True,
        vision_in_chans: int = 1,
        decoder_nhead: int = 8,
        decoder_num_layers: int = 4,
        decoder_dim_feedforward: int = 2048,
        decoder_dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_id = pad_id
        
        # 1. 2D 视觉主干
        self.encoder = ConvNeXtV2Encoder(
            model_name=vision_model_name,
            pretrained=vision_pretrained,
            d_model=d_model, 
            in_chans=vision_in_chans,
        )
        
        # 2. 文本词表 Embedding 与位置编码
        self.text_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.text_pos_enc = PositionalEncoding1D(d_model=d_model, dropout=0.1)
        
        # 3. 语言大脑：AttnRes 解码器
        self.decoder = AttnResTextDecoder(
            d_model=d_model, 
            nhead=decoder_nhead,
            num_layers=decoder_num_layers,
            dim_feedforward=decoder_dim_feedforward,
            dropout=decoder_dropout,
        )
        
        # 4. 预测分类头
        self.head = nn.Linear(d_model, vocab_size)

    def generate_causal_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        """生成用于自回归分支的下三角因果掩码"""
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """只运行一次视觉骨干网络"""
        return self.encoder(images)

    def decode(
        self,
        memory: torch.Tensor,
        tgt_seq: torch.Tensor,
        memory_padding_mask: Optional[torch.Tensor] = None,
        is_causal: bool = True,
    ) -> torch.Tensor:
        """自回归循环运行的大脑"""
        tgt_key_padding_mask = (tgt_seq == self.pad_id)
        tgt_emb = self.text_embedding(tgt_seq) * math.sqrt(self.d_model)
        tgt_emb = self.text_pos_enc(tgt_emb)

        seq_len = tgt_seq.size(1)
        tgt_mask = self.generate_causal_mask(seq_len, tgt_seq.device) if is_causal else None

        decoder_out = self.decoder(
            text_embeddings=tgt_emb,
            cross_features=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_padding_mask,
        )
        logits = self.head(decoder_out)
        return logits

    def forward(self, images: torch.Tensor, tgt_seq: torch.Tensor, is_causal: bool = True) -> torch.Tensor:
        """训练时使用的完整前向传播"""
        import torch.nn.functional as F

        memory = self.encode(images)
        batch_size = memory.size(0)

        # 通过池化近似估计视觉 token 的无效填充区域
        with torch.no_grad():
            downsampled_mask = F.max_pool2d(images, kernel_size=32, stride=32)
            memory_padding_mask = (downsampled_mask.view(batch_size, -1) <= 1e-5)

        return self.decode(memory, tgt_seq, memory_padding_mask, is_causal)