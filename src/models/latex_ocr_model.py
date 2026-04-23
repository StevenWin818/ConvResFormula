"""
LaTeX OCR 主模型组装与 MLM 前向接口。
"""
import math
import torch
import torch.nn as nn
from typing import Optional, Tuple, Union, overload

from .vision_encoder import ConvNeXtV2Encoder
from .text_decoder import AttnResDecodeCache, AttnResTextDecoder

class LatexOCRModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        pad_id: int = 0,
        eos_id: Optional[int] = None,
        vision_model_name: str = "convnextv2_nano",
        vision_pretrained: bool = True,
        vision_in_chans: int = 1,
        vision_drop_path_rate: float = 0.0,
        decoder_nhead: int = 8,
        decoder_num_layers: int = 6,
        decoder_dim_feedforward: int = 2048,
        decoder_dropout: float = 0.1,
        use_learned_position_embeddings: bool = True,
        max_position_embeddings: int = 2048,
        use_gradient_checkpointing: bool = False,
        checkpoint_segments: int = 1,
        checkpoint_decoder_layers: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = int(vocab_size)
        self.pad_id = pad_id
        self.eos_id = eos_id
        self.use_learned_position_embeddings = use_learned_position_embeddings
        self.max_position_embeddings = max_position_embeddings
        self.use_gradient_checkpointing = bool(use_gradient_checkpointing)
        self.checkpoint_segments = max(1, int(checkpoint_segments))
        self.checkpoint_decoder_layers = bool(checkpoint_decoder_layers)
        
        # 1. 2D 视觉主干
        self.encoder = ConvNeXtV2Encoder(
            model_name=vision_model_name,
            pretrained=vision_pretrained,
            d_model=d_model, 
            ctc_vocab_size=int(vocab_size),
            in_chans=vision_in_chans,
            drop_path_rate=vision_drop_path_rate,
            use_gradient_checkpointing=self.use_gradient_checkpointing,
        )
        
        # 2. 文本词表 Embedding 与位置编码
        self.text_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.text_pos_embedding = nn.Embedding(max_position_embeddings, d_model)
        
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

    def _apply_position_embeddings(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        """为文本特征注入 learned position embeddings。"""
        if not self.use_learned_position_embeddings:
            return x

        seq_len = x.size(1)
        end_pos = start_pos + seq_len
        if end_pos > self.max_position_embeddings:
            raise ValueError(
                f"位置索引越界: end_pos={end_pos}, max_position_embeddings={self.max_position_embeddings}"
            )

        positions = torch.arange(start_pos, end_pos, device=x.device, dtype=torch.long)
        pos_emb = self.text_pos_embedding(positions).unsqueeze(0)
        return x + pos_emb

    def generate_causal_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        """生成用于自回归分支的下三角因果掩码"""
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    @overload
    def encode(
        self,
        images: torch.Tensor,
        return_aux: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]: ...

    @overload
    def encode(
        self,
        images: torch.Tensor,
        return_aux: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: ...

    def encode(
        self,
        images: torch.Tensor,
        return_aux: bool = False,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        """
        返回特征序列和 Padding 掩码
        
        Returns:
            当 return_aux=False: (memory, memory_mask)
            当 return_aux=True: (memory, memory_mask, bow_logits, sigreg_embedding)
        """
        return self.encoder(images, return_aux=return_aux)

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
        tgt_emb = self._apply_position_embeddings(tgt_emb, start_pos=0)

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
        tgt_emb = self._apply_position_embeddings(tgt_emb, start_pos=start_pos)

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
        tgt_emb = self._apply_position_embeddings(tgt_emb, start_pos=start_pos)

        decoder_out, new_cache = self.decoder.forward_step_cached(
            text_embeddings=tgt_emb,
            cross_features=memory,
            cache=cache,
            memory_key_padding_mask=memory_padding_mask,
        )
        logits = self.head(decoder_out.squeeze(1))
        return logits, new_cache

    def forward(
        self,
        images: torch.Tensor,
        tgt_seq: torch.Tensor,
        is_causal: bool = True,
        return_aux: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        前向传播
        
        Args:
            images: [B, 1, H, W] 图像张量
            tgt_seq: [B, T] 目标序列
            is_causal: 是否使用因果掩码
            return_aux: 是否返回辅助输出（BoW 和 SIGReg embedding）
        
        Returns:
            return_aux=False: logits [B, T, vocab_size]
            return_aux=True: (logits, bow_logits, sigreg_embedding)
        """
        # 1. 获取编码器输出
        encoder_output = self.encoder(images, return_aux=return_aux)
        
        # 2. 根据 return_aux 解包返回值
        bow_logits: Optional[torch.Tensor] = None
        sigreg_embedding: Optional[torch.Tensor] = None
        
        if return_aux:
            # return_aux=True 时返回 4 元组: (memory, memory_mask, bow_logits, sigreg_embedding)
            memory, memory_mask, bow_logits, sigreg_embedding = encoder_output
        else:
            # return_aux=False 时返回 2 元组: (memory, memory_mask)
            memory, memory_mask = encoder_output

        # 解码步骤
        tgt_key_padding_mask = (tgt_seq == self.pad_id)
        tgt_emb = self.text_embedding(tgt_seq) * math.sqrt(self.d_model)
        tgt_emb = self._apply_position_embeddings(tgt_emb, start_pos=0)

        seq_len = tgt_seq.size(1)
        tgt_mask = self.generate_causal_mask(seq_len, tgt_seq.device) if is_causal else None

        # 解码器占用显存较小，直接前向即可
        decoder_out = self.decoder(
            text_embeddings=tgt_emb,
            cross_features=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_mask,
        )

        logits = self.head(decoder_out)
        if return_aux and bow_logits is not None and sigreg_embedding is not None:
            return logits, bow_logits, sigreg_embedding
        return logits