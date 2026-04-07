"""
LaTeX OCR 主模型组装与 MLM 前向接口。
"""
import math
import torch
import torch.nn as nn
from typing import Optional, Tuple

from .vision_encoder import ConvNeXtV2Encoder
from .text_decoder import AttnResDecodeCache, AttnResTextDecoder

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

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        """x: [Batch, SeqLen, d_model]"""
        seq_len = x.size(1)
        x = x + self.pe[:, start_pos:start_pos + seq_len, :]
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

    def encode(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回特征序列和 Padding 掩码"""
        return self.encoder(images)

    def precompute_cross_kv(self, memory: torch.Tensor):
        """预计算解码器交叉注意力 K/V，用于自回归推理。"""
        return self.decoder.precompute_cross_kv(memory)

    def init_decode_cache(self, memory: torch.Tensor) -> AttnResDecodeCache:
        """初始化自回归解码缓存。"""
        return self.decoder.init_cache(memory)

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

    def decode_step(
        self,
        memory: torch.Tensor,
        token_id: torch.Tensor,
        past_key_values=None,
        memory_padding_mask: Optional[torch.Tensor] = None,
        precomputed_cross_kvs=None,
    ):
        """单步自回归解码，配合 KV Cache 使用。"""
        if token_id.dim() == 1:
            token_id = token_id.unsqueeze(1)
        if token_id.dim() != 2 or token_id.size(1) != 1:
            raise ValueError(f"decode_step 需要形状为 [B] 或 [B, 1] 的 token_id，当前为 {tuple(token_id.shape)}")

        if past_key_values is not None and len(past_key_values) > 0:
            start_pos = int(past_key_values[0][0].size(2))
        else:
            start_pos = 0

        tgt_emb = self.text_embedding(token_id) * math.sqrt(self.d_model)
        tgt_emb = self.text_pos_enc(tgt_emb, start_pos=start_pos)

        if precomputed_cross_kvs is None:
            precomputed_cross_kvs = self.precompute_cross_kv(memory)

        decoder_out, new_key_values = self.decoder.forward_step(
            text_embeddings=tgt_emb,
            cross_features=memory,
            past_key_values=past_key_values,
            memory_key_padding_mask=memory_padding_mask,
            precomputed_cross_kvs=precomputed_cross_kvs,
        )
        logits = self.head(decoder_out.squeeze(1))
        return logits, new_key_values

    def decode_step_cached(
        self,
        memory: torch.Tensor,
        token_id: torch.Tensor,
        cache: AttnResDecodeCache,
        memory_padding_mask: Optional[torch.Tensor] = None,
    ):
        """单步自回归解码，使用显式 KV Cache 容器。"""
        if token_id.dim() == 1:
            token_id = token_id.unsqueeze(1)
        if token_id.dim() != 2 or token_id.size(1) != 1:
            raise ValueError(f"decode_step_cached 需要形状为 [B] 或 [B, 1] 的 token_id，当前为 {tuple(token_id.shape)}")

        if cache.self_key_values and cache.self_key_values[0] is not None:
            start_pos = int(cache.self_key_values[0][0].size(2))
        else:
            start_pos = 0

        tgt_emb = self.text_embedding(token_id) * math.sqrt(self.d_model)
        tgt_emb = self.text_pos_enc(tgt_emb, start_pos=start_pos)

        decoder_out, new_cache = self.decoder.forward_step_cached(
            text_embeddings=tgt_emb,
            cross_features=memory,
            cache=cache,
            memory_key_padding_mask=memory_padding_mask,
        )
        logits = self.head(decoder_out.squeeze(1))
        return logits, new_cache

    def forward(self, images: torch.Tensor, tgt_seq: torch.Tensor, is_causal: bool = True) -> torch.Tensor:
        """训练时使用的完整前向传播"""
        memory, memory_padding_mask = self.encode(images)
        return self.decode(memory, tgt_seq, memory_padding_mask, is_causal)