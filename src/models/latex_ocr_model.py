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
    def __init__(self, vocab_size: int, d_model: int = 512, pad_id: int = 0):
        super().__init__()
        self.d_model = d_model
        self.pad_id = pad_id
        
        # 1. 2D 视觉主干
        self.encoder = ConvNeXtV2Encoder(
            model_name='convnextv2_pico', 
            pretrained=True, 
            d_model=d_model, 
            in_chans=1
        )
        
        # 2. 文本词表 Embedding 与位置编码
        self.text_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.text_pos_enc = PositionalEncoding1D(d_model=d_model, dropout=0.1)
        
        # 3. 语言大脑：AttnRes 解码器
        self.decoder = AttnResTextDecoder(
            d_model=d_model, 
            nhead=8, 
            num_layers=4, 
            dim_feedforward=2048, 
            dropout=0.1
        )
        
        # 4. 预测分类头
        self.head = nn.Linear(d_model, vocab_size)

    def generate_causal_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        """生成用于自回归分支的下三角因果掩码"""
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, images: torch.Tensor, tgt_seq: torch.Tensor, is_causal: bool = True) -> torch.Tensor:
        """
        前向传播 (支持双分支模式)
        Args:
            images: [Batch, 1, H, W] 图像张量
            tgt_seq: [Batch, SeqLen] 输入给 Decoder 的文本 ID 
            is_causal: True 为标准 AR 模式(下三角掩码)，False 为 MLM 模式(无掩码，允许双向互看)
        Returns:
            logits: [Batch, SeqLen, vocab_size]
        """
        # --- 视觉编码阶段 ---
        # memory: [Batch, ImageSeqLen, d_model]
        memory = self.encoder(images)
        
        # --- 文本编码阶段 ---
        # 文本 padding 掩码 (告诉模型哪些是 [PAD]，不要将注意力放在上面)
        tgt_key_padding_mask = (tgt_seq == self.pad_id)
        
        # Token Embedding + Positional Encoding
        tgt_emb = self.text_embedding(tgt_seq) * math.sqrt(self.d_model)
        tgt_emb = self.text_pos_enc(tgt_emb)
        
        # 动态掩码生成
        seq_len = tgt_seq.size(1)
        tgt_mask = self.generate_causal_mask(seq_len, tgt_seq.device) if is_causal else None

        # --- 解码阶段 ---
        decoder_out = self.decoder(
            text_embeddings=tgt_emb,
            cross_features=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=None # 视觉侧通常不用 padding mask
        )
        
        # --- 分类预测 ---
        logits = self.head(decoder_out)
        return logits