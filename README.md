# ConvResFormula

ConvResFormula 是一个高性能、端到端的 LaTeX 公式识别（OCR）框架。它结合了先进的卷积神经网络（ConvNeXt）作为视觉编码器与高效的残差注意力（Attention-Residual）机制作为文本解码器，旨在实现从公式图像到标准 LaTeX 序列的精准转换。

## 🌟 核心特性

- **混合架构**：编码器支持 ConvNeXt 系列（包括 V2-Tiny），解码器采用增强型残差注意力模型。
- **JEPA 感知增强**：引入 **SIGReg (Signal Regularization)** 机制，支持联合嵌入预测架构 (JEPA) 感知，强化编码器对公式深层语义特征的提取能力。
- **技术洞察与优化**：
  - **BoW 辅助训练**：经实验证实 1D-CTC 并不适用于此类高度复杂的视觉任务；本项目提出并采用 **词袋模型 (Bag-of-Words, BoW)** 解决该问题，显著提升了模型在复杂场景下的识别准确率。
  - **推理加速**：全面集成 **Kv Cache** 技术，大幅提高自回归推理速度，同时降低了推理所需的显存与算力消耗。
  - **精准解码**：引入 **动态 Alpha 值** 优化启发式搜索过程，进一步提高推理阶段的序列生成准确度。
- **多尺寸配置**：
  - **Mini (main 分支)**：约 **30M** 参数量，轻量化设计，优化推理速度，适合边缘设备部署。
  - **Large (ConvResFormula-large 分支)**：约 **75M** 参数量，采用更强大的视觉骨干网络与更宽的解码器。
- **全流程工具链**：支持数据准备、HDF5 导出、动态分辨率调整及公式渲染（KaTeX/SVG）。

## 📂 项目结构

```text
ConvResFormula/
├── configs/           # 模型与训练配置文件 (YAML)
├── scripts/           # 核心脚本：训练、推理、评估
├── src/
│   ├── data/          # 数据加载、BPE 分词器、数据增强
│   ├── engine/        # 训练引擎、学习率调度器
│   ├── models/        # 模型定义 (Encoder, Decoder, Loss, SIGReg)
│   └── utils/         # 评估指标、掩码工具、检查点管理
├── tools/             # 工具箱
│   ├── data_prep/     # 数据预处理与格式转换
│   ├── deploy/        # ONNX 导出与权重提取
│   └── vocab/         # 词表训练与同步处理
└── diagnostic/        # 诊断工具（ Attention 热力图）
```

## 🚀 快速上手

### 1. 环境准备
```bash
git clone https://github.com/StevenWin818/ConvResFormula.git
cd ConvResFormula
pip install -r requirements.txt
```

### 2. 模型版本选择
*   **Mini 版本** (~30M params)：`git checkout main`
*   **Large 版本** (~75M params)：`git checkout ConvResFormula-large`

### 3. 推理示例
```bash
python scripts/infer.py --img_path examples/formula.png --checkpoint checkpoints/latest.pth
```

## 📊 评估指标
项目除了支持常见的 CER、EM 外，还引入了：
- **首创视觉一致性评估 (Visual Consistency Evaluation)**：
  - **原理**：传统指标（如 BLEU/CER）难以衡量公式的实际数学等价性。本项目通过将预测的 LaTeX 实时渲染为图像，并与原始视觉特征进行结构一致性对比，从视觉感知维度判断识别的正确性。这对于手写公式等样式多变的场景具有极高的参考价值。

## 🛠 部署
支持导出为 ONNX 格式，支持 Kv Cache 优化加速。

## 📜 许可证
[Mozilla Public License 2.0](LICENSE)
