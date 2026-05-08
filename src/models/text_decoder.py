"""AttnRes Transformer 文本解码器接口。"""

from __future__ import annotations

import math
from dataclasses import dataclass
import re
from typing import List, Optional, Tuple, cast

import torch
import torch.nn as nn
import torch.nn.functional as F


def _split_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    """[B, T, D] -> [B, H, T, Dh]"""
    batch_size, seq_len, hidden_dim = x.shape
    head_dim = hidden_dim // num_heads
    return x.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)


def _merge_heads(x: torch.Tensor) -> torch.Tensor:
    """[B, H, T, Dh] -> [B, T, D]"""
    batch_size, num_heads, seq_len, head_dim = x.shape
    return x.transpose(1, 2).contiguous().view(batch_size, seq_len, num_heads * head_dim)


def _additive_mask_from_padding(
    padding_mask: torch.Tensor,
    query_len: int,
    key_len: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """将 padding mask 转成 SDPA 可用的 additive mask。"""
    batch_size = padding_mask.size(0)
    expanded = padding_mask[:, None, None, :].expand(batch_size, 1, query_len, key_len)
    mask = torch.zeros((batch_size, 1, query_len, key_len), device=padding_mask.device, dtype=dtype)
    return mask.masked_fill(expanded, float("-inf"))


def convert_legacy_attnres_state_dict(state_dict: dict) -> dict:
    """将旧版 MultiheadAttention 参数转换为当前 KV-Cache 版本的参数名。"""
    converted = dict(state_dict)
    
    # 检测并处理 torch.compile 带来的 _orig_mod. 前缀
    prefix = ""
    if any(k.startswith("_orig_mod.") for k in converted.keys()):
        prefix = "_orig_mod."

    # 兼容 CoordConv 升级：如果 checkpoint 中的 stem_0 是单通道的，而我们现在需要 3 通道 (Gray, X, Y)
    # 我们用 0 填充新增的 X 和 Y 通道权重，这使得模型在加载权重的瞬间，计算结果与之前的单通道版本完全一致
    stem_key = f"{prefix}encoder.backbone.stem_0.weight"
    stem_w = converted.get(stem_key)
    if stem_w is not None and stem_w.dim() == 4 and stem_w.shape[1] == 1:
        padded = torch.zeros(stem_w.shape[0], 3, stem_w.shape[2], stem_w.shape[3], dtype=stem_w.dtype, device=stem_w.device)
        padded[:, 0:1, :, :] = stem_w
        converted[stem_key] = padded

    # 兼容 FPN 升级：将原本的 proj 权重映射给 proj_32
    proj_w_key = f"{prefix}encoder.proj.weight"
    proj_b_key = f"{prefix}encoder.proj.bias"
    if proj_w_key in converted:
        converted[f"{prefix}encoder.proj_32.weight"] = converted.pop(proj_w_key)
    if proj_b_key in converted:
        converted[f"{prefix}encoder.proj_32.bias"] = converted.pop(proj_b_key)

    legacy_pattern = re.compile(r"^decoder\.layers\.(\d+)\.(self_attn|cross_attn)\.in_proj_(weight|bias)$")
    legacy_layer_indices = sorted({int(match.group(1)) for key in converted.keys() if (match := legacy_pattern.match(key))})

    for idx in legacy_layer_indices:
        for attn_name, new_prefix in (("self_attn", "self"), ("cross_attn", "cross")):
            old_prefix = f"decoder.layers.{idx}.{attn_name}"
            new_prefix = f"decoder.layers.{idx}.{new_prefix}"

            in_proj_weight = converted.pop(f"{old_prefix}.in_proj_weight", None)
            in_proj_bias = converted.pop(f"{old_prefix}.in_proj_bias", None)
            out_proj_weight = converted.pop(f"{old_prefix}.out_proj.weight", None)
            out_proj_bias = converted.pop(f"{old_prefix}.out_proj.bias", None)

            if in_proj_weight is not None:
                q_w, k_w, v_w = in_proj_weight.chunk(3, dim=0)
                converted[f"{new_prefix}_q.weight"] = q_w
                converted[f"{new_prefix}_k.weight"] = k_w
                converted[f"{new_prefix}_v.weight"] = v_w

            if in_proj_bias is not None:
                q_b, k_b, v_b = in_proj_bias.chunk(3, dim=0)
                # 新版 q/k/v 线性层统一使用 bias=False，旧 bias 直接丢弃即可。
                del q_b, k_b, v_b

            if out_proj_weight is not None:
                converted[f"{new_prefix}_out.weight"] = out_proj_weight
            if out_proj_bias is not None:
                converted[f"{new_prefix}_out.bias"] = out_proj_bias

    return converted


@dataclass
class AttnResDecodeCache:
    """AttnRes 解码器的 KV Cache 容器。"""

    self_key_values: List[Optional[Tuple[torch.Tensor, torch.Tensor]]]
    cross_kvs: List[Tuple[torch.Tensor, torch.Tensor]]


class AttnResDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="gelu"):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model({d_model}) 必须能被 nhead({nhead}) 整除")

        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.dropout_p = float(dropout)

        # Self attention (支持 KV Cache)
        self.self_q = nn.Linear(d_model, d_model, bias=False)
        self.self_k = nn.Linear(d_model, d_model, bias=False)
        self.self_v = nn.Linear(d_model, d_model, bias=False)
        self.self_out = nn.Linear(d_model, d_model)

        # Cross attention (支持预计算 K/V)
        self.cross_q = nn.Linear(d_model, d_model, bias=False)
        self.cross_k = nn.Linear(d_model, d_model, bias=False)
        self.cross_v = nn.Linear(d_model, d_model, bias=False)
        self.cross_out = nn.Linear(d_model, d_model)

        # FFN
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Pre-Norm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.GELU() if activation == "gelu" else nn.ReLU()

    def _self_attention(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        past_k: Optional[torch.Tensor] = None,
        past_v: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, query_len, _ = x.shape
        q = _split_heads(self.self_q(x), self.nhead)
        k = _split_heads(self.self_k(x), self.nhead)
        v = _split_heads(self.self_v(x), self.nhead)

        if past_k is not None:
            if past_v is None:
                raise ValueError("past_k 不为空时 past_v 也必须不为空")
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        key_len = k.size(2)

        total_mask: Optional[torch.Tensor] = None
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            elif attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(1)
            total_mask = attn_mask.to(device=x.device, dtype=q.dtype)
        elif past_k is None and query_len > 1:
            causal = torch.triu(
                torch.ones((query_len, key_len), device=x.device, dtype=torch.bool),
                diagonal=1,
            )
            total_mask = torch.zeros((1, 1, query_len, key_len), device=x.device, dtype=q.dtype)
            total_mask = total_mask.masked_fill(causal.unsqueeze(0).unsqueeze(0), float("-inf"))

        if key_padding_mask is not None:
            padding_bias = _additive_mask_from_padding(key_padding_mask, query_len, key_len, q.dtype)
            total_mask = padding_bias if total_mask is None else (total_mask + padding_bias)

        dropout_p = self.dropout_p if self.training else 0.0
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=total_mask,
            dropout_p=dropout_p,
        )

        out = _merge_heads(out)
        out = self.self_out(out)
        return out, k, v

    def _cross_attention(
        self,
        x: torch.Tensor,
        cross_features: torch.Tensor,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        precomputed_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_attn_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, query_len, _ = x.shape
        q = _split_heads(self.cross_q(x), self.nhead)

        if precomputed_kv is not None:
            k, v = precomputed_kv
        else:
            k = _split_heads(self.cross_k(cross_features), self.nhead)
            v = _split_heads(self.cross_v(cross_features), self.nhead)

        key_len = k.size(2)
        total_mask: Optional[torch.Tensor] = None
        if memory_key_padding_mask is not None:
            total_mask = _additive_mask_from_padding(memory_key_padding_mask, query_len, key_len, q.dtype)

        dropout_p = self.dropout_p if self.training else 0.0
        
        if return_attn_weights:
            attn_weight = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
            if total_mask is not None:
                attn_weight += total_mask
            attn_weight = F.softmax(attn_weight, dim=-1)
            out = torch.matmul(attn_weight, v)
        else:
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=total_mask,
                dropout_p=dropout_p,
            )
            attn_weight = None

        out = _merge_heads(out)
        return self.cross_out(out), attn_weight

    def precompute_cross_kv(self, cross_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """预计算当前层的 cross-attention K/V。"""
        k = _split_heads(self.cross_k(cross_features), self.nhead)
        v = _split_heads(self.cross_v(cross_features), self.nhead)
        return k, v

    def forward(
        self,
        tgt,
        cross_features,
        tgt_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        precomputed_cross_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        past_self_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_attn_weights: bool = False,
    ):
        # 1. Pre-Norm Self Attention
        residual = tgt
        tgt2 = self.norm1(tgt)
        past_k = past_self_kv[0] if past_self_kv is not None else None
        past_v = past_self_kv[1] if past_self_kv is not None else None
        tgt2, new_k, new_v = self._self_attention(
            tgt2,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
            past_k=past_k,
            past_v=past_v,
        )
        tgt = residual + self.dropout1(tgt2)

        # 2. Pre-Norm Cross Attention
        attn_weights = None
        if cross_features is not None:
            residual = tgt
            tgt2 = self.norm2(tgt)
            tgt2, attn_weights = self._cross_attention(
                tgt2,
                cross_features,
                memory_key_padding_mask=memory_key_padding_mask,
                precomputed_kv=precomputed_cross_kv,
                return_attn_weights=return_attn_weights,
            )
            tgt = residual + self.dropout2(tgt2)

        # 3. Pre-Norm FFN
        residual = tgt
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = residual + self.dropout3(tgt2)

        return tgt, (new_k, new_v), attn_weights


class AttnResTextDecoder(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_layers=4, dim_feedforward=1024, dropout=0.1):
        """AttnRes (Attention Residual) 解码器。"""
        super().__init__()
        self.d_model = d_model

        self.layers: nn.ModuleList = nn.ModuleList(
            [AttnResDecoderLayer(d_model, nhead, dim_feedforward, dropout) for _ in range(num_layers)]
        )

        self.norm = nn.LayerNorm(d_model)

    def precompute_cross_kv(self, cross_features: torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """预计算每层交叉注意力的 K/V。"""
        return self._precompute_cross_kv(cross_features)

    def init_cache(self, cross_features: torch.Tensor) -> AttnResDecodeCache:
        """初始化一份可复用的解码缓存。"""
        num_layers = len(self.layers)
        return AttnResDecodeCache(
            self_key_values=[None] * num_layers,
            cross_kvs=self._precompute_cross_kv(cross_features),
        )

    def _precompute_cross_kv(self, cross_features: torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        cross_kvs: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for layer in self.layers:
            decoder_layer = cast(AttnResDecoderLayer, layer)
            cross_kvs.append(decoder_layer.precompute_cross_kv(cross_features))
        return cross_kvs

    def forward(
        self,
        text_embeddings,
        cross_features,
        tgt_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        precomputed_cross_kvs: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        return_attn_weights: bool = False,
    ):
        """训练时的全序列前向。"""
        x = text_embeddings

        if cross_features is not None and precomputed_cross_kvs is None:
            precomputed_cross_kvs = self._precompute_cross_kv(cross_features)

        all_attn_weights = []
        for idx, layer in enumerate(self.layers):
            cross_kv = None if precomputed_cross_kvs is None else precomputed_cross_kvs[idx]
            x, _, attn_w = layer(
                x,
                cross_features,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                precomputed_cross_kv=cross_kv,
                past_self_kv=None,
                return_attn_weights=return_attn_weights,
            )
            if attn_w is not None:
                all_attn_weights.append(attn_w)

        if return_attn_weights:
            return self.norm(x), all_attn_weights
        return self.norm(x)

    def forward_step(
        self,
        text_embeddings: torch.Tensor,
        cross_features: torch.Tensor,
        past_key_values: Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]],
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        precomputed_cross_kvs: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        return_attn_weights: bool = False,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]], Optional[List[torch.Tensor]]]:
        """单步自回归前向，配合 KV Cache 使用。"""
        x = text_embeddings

        if precomputed_cross_kvs is None:
            precomputed_cross_kvs = self._precompute_cross_kv(cross_features)

        new_key_values: List[Tuple[torch.Tensor, torch.Tensor]] = []
        all_attn_weights = []
        for idx, layer in enumerate(self.layers):
            past_self_kv = None if past_key_values is None else past_key_values[idx]
            x, new_self_kv, attn_w = layer(
                x,
                cross_features,
                tgt_mask=None,
                tgt_key_padding_mask=None,
                memory_key_padding_mask=memory_key_padding_mask,
                precomputed_cross_kv=precomputed_cross_kvs[idx],
                past_self_kv=past_self_kv,
                return_attn_weights=return_attn_weights,
            )
            new_key_values.append(new_self_kv)
            if attn_w is not None:
                all_attn_weights.append(attn_w)

        return self.norm(x), new_key_values, all_attn_weights if return_attn_weights else None

    def forward_step_cached(
        self,
        text_embeddings: torch.Tensor,
        cross_features: torch.Tensor,
        cache: AttnResDecodeCache,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        return_attn_weights: bool = False,
    ) -> Tuple[torch.Tensor, AttnResDecodeCache, Optional[List[torch.Tensor]]]:
        """单步自回归前向，显式使用 AttnResDecodeCache。"""
        decoder_out, new_self_key_values, attn_weights = self.forward_step(
            text_embeddings=text_embeddings,
            cross_features=cross_features,
            past_key_values=cache.self_key_values,
            memory_key_padding_mask=memory_key_padding_mask,
            precomputed_cross_kvs=cache.cross_kvs,
            return_attn_weights=return_attn_weights,
        )
        return decoder_out, AttnResDecodeCache(
            self_key_values=cast(List[Optional[Tuple[torch.Tensor, torch.Tensor]]], new_self_key_values),
            cross_kvs=cache.cross_kvs,
        ), attn_weights
