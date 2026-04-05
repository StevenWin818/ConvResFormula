"""
部署与推理工具包

Scripts:
  - export_split_onnx.py      : 分离编码器、解码器并导出 ONNX（支持自回归推理）
  - onnx_infer.py             : 使用 ONNX Runtime 进行推理测试
  - export_vocab_header.py    : 生成词表 C/C++ 头文件（端侧部署）
"""
