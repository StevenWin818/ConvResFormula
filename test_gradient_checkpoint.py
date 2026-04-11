#!/usr/bin/env python3
"""
快速验证梯度检查点功能的脚本
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from src.models.latex_ocr_model import LatexOCRModel


def test_gradient_checkpoint():
    """测试梯度检查点是否能正常工作"""
    print("=" * 60)
    print("梯度检查点功能测试")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}\n")
    
    # 创建模型（启用梯度检查点）
    model = LatexOCRModel(
        vocab_size=10000,
        d_model=256,
        use_gradient_checkpointing=True,
        checkpoint_decoder_layers=False,
    ).to(device)
    
    model.train()
    
    print("模型配置:")
    print(f"  - use_gradient_checkpointing: {model.use_gradient_checkpointing}")
    print(f"  - checkpoint_segments: {model.checkpoint_segments}")
    print(f"  - checkpoint_decoder_layers: {model.checkpoint_decoder_layers}\n")
    
    # 创建虚拟输入
    batch_size = 4
    img_h, img_w = 96, 384
    images = torch.randn(batch_size, 1, img_h, img_w).to(device)
    tgt_seq = torch.randint(1, 1000, (batch_size, 50)).to(device)
    
    print(f"输入形状:")
    print(f"  - images: {tuple(images.shape)}")
    print(f"  - tgt_seq: {tuple(tgt_seq.shape)}\n")
    
    # 前向传播
    print("执行前向传播...")
    try:
        logits = model(images, tgt_seq, is_causal=True)
        print(f"前向传播成功！输出形状: {tuple(logits.shape)}\n")
    except Exception as e:
        print(f"前向传播失败: {e}\n")
        return False
    
    # 反向传播
    print("执行反向传播...")
    try:
        loss = logits.mean()
        loss.backward()
        print(f"反向传播成功！\n")
    except Exception as e:
        print(f"反向传播失败: {e}\n")
        return False
    
    # 检查梯度
    print("检查梯度...")
    has_grad = False
    for name, param in model.named_parameters():
        if param.grad is not None:
            has_grad = True
            break
    
    if has_grad:
        print("✓ 梯度计算成功\n")
    else:
        print("✗ 未检测到梯度\n")
        return False
    
    # 测试推理模式（梯度检查点不应该在推理时激活）
    print("测试推理模式...")
    model.eval()
    with torch.no_grad():
        logits_eval = model(images, tgt_seq, is_causal=True)
        print(f"推理模式前向传播成功！输出形状: {tuple(logits_eval.shape)}\n")
    
    print("=" * 60)
    print("✓ 所有测试通过！梯度检查点功能正常运行")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = test_gradient_checkpoint()
    sys.exit(0 if success else 1)
