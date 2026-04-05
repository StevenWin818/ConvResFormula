"""AttnRes Transformer 文本解码器接口。"""
import torch
import torch.nn as nn

class AttnResDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="gelu"):
        super().__init__()
        # Self attention (用于文本特征内的注意力)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # Cross attention (用于文本特征对图像特征的交叉注意力)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # FFN (前馈神经网络)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Pre-Norm 架构的 LayerNorms (推荐用于深层大模型，梯度传播更稳定)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
        self.activation = nn.GELU() if activation == "gelu" else nn.ReLU()

    def forward(self, tgt, cross_features, tgt_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        # 1. Pre-Norm Self Attention
        tgt2 = self.norm1(tgt)
        # 通过 tgt_mask 支持 AR(因果) 与 MLM(双向) 两种模式动态切换
        tgt2, _ = self.self_attn(tgt2, tgt2, tgt2, 
                                 attn_mask=tgt_mask,
                                 key_padding_mask=tgt_key_padding_mask)
        # 残差连接
        tgt = tgt + self.dropout1(tgt2)
        
        # 2. Pre-Norm Cross Attention
        if cross_features is not None:
            tgt2 = self.norm2(tgt)
            tgt2, _ = self.cross_attn(tgt2, cross_features, cross_features, 
                                      attn_mask=None,
                                      key_padding_mask=memory_key_padding_mask)
            # 残差连接
            tgt = tgt + self.dropout2(tgt2)
            
        # 3. Pre-Norm FFN
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        # 残差连接
        tgt = tgt + self.dropout3(tgt2)
        
        return tgt

class AttnResTextDecoder(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_layers=4, dim_feedforward=1024, dropout=0.1):
        """
        AttnRes (Attention Residual) 解码器。
        使用无因果掩码的并行架构，支持 MLM 并行训练。
        """
        super().__init__()
        self.d_model = d_model
        
        self.layers = nn.ModuleList([
            AttnResDecoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        # 由于是 Pre-Norm，在最后输出前需要经过一次整体的 LayerNorm
        self.norm = nn.LayerNorm(d_model)

    def forward(self, text_embeddings, cross_features, tgt_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """
        Args:
            text_embeddings: (Batch, SeqLen, d_model) 融合了 Token 和 Positional Embedding 的文本特征
            cross_features: (Batch, MemoryLen, d_model) 视觉编码器输出的图像特征
            tgt_key_padding_mask: (Batch, SeqLen) 指示文本 padding 位置的布尔掩码 (True 为 padding)
            memory_key_padding_mask: (Batch, MemoryLen) 指示图像 padding 位置的布尔掩码 (若不需要可为 None)
            
        Returns:
            decoder_output: (Batch, SeqLen, d_model) 提取的解码器高级特征输出
        """
        x = text_embeddings
        
        for layer in self.layers:
            x = layer(x, cross_features, 
                      tgt_mask=tgt_mask,
                      tgt_key_padding_mask=tgt_key_padding_mask,
                      memory_key_padding_mask=memory_key_padding_mask)
                      
        x = self.norm(x)
        return x
