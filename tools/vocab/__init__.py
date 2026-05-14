"""
词表管理工具包

Scripts:
  - build_unified_vocab.py    : 从多数据源构建统一词表（手写 + 合成 + Wiki 语料）
  - vocab.py                  : 词表类定义与序列化
  - latex_synonyms.txt        : LaTeX 符号别名映射（used for Many-to-One aliasing）
  - unicode-math-table.tex    : Unicode Math 符号表参考

Outputs:
  - vocab.h, vocab.pkl        : 最终词表文件（C++ / Python）
"""
