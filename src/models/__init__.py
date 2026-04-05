"""
模型架构核心模块

Modules:
  - encoder_1d.py       : 1D 在线笔迹编码器（1D-CNN/ResNet + Transformer Encoder）
  - encoder_2d.py       : 2D 离线图片编码器（ViT / ResNet） ✨ 规划中
  - decoder.py          : 跨模态共享解码器（Transformer Decoder + KV Cache）
  - unified_model.py    : 顶层组件（动态选择编码器 + 固定解码器）✨ 规划中
  - latex_lm.py         : LaTeX 语言模型（可选的符号预测约束）
"""
