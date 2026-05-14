"""分析各数据集的特征序列长度和标签长度分布"""
import os
import sys
import numpy as np
from tqdm import tqdm

# 加入项目根目录
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data.dataset_1d import parse_inkml
from src.data.tokenizer import tokenize_latex, normalize_sub_sup


def count_points(strokes):
    """计算笔迹总点数"""
    return sum(len(s) for s in strokes if s.size > 0)


def analyze_dir(data_dir, name):
    if not os.path.exists(data_dir):
        print(f"⚠️ {name}: 目录不存在 {data_dir}")
        return

    feat_lens = []
    label_lens = []

    files = []
    for root, _, fnames in os.walk(data_dir):
        for f in fnames:
            if f.endswith(".inkml"):
                files.append(os.path.join(root, f))

    for path in tqdm(files, desc=name):
        try:
            strokes, label = parse_inkml(path)
            if strokes:
                feat_lens.append(count_points(strokes))
            if label and label.strip().lower() != "unknown":
                clean = normalize_sub_sup(label.replace("$", "").strip())
                tokens = tokenize_latex(clean)
                label_lens.append(len(tokens) + 2)  # +2 for SOS/EOS
        except Exception:
            continue

    if not feat_lens:
        print(f"⚠️ {name}: 无有效样本")
        return

    feat_arr = np.array(feat_lens)
    lab_arr = np.array(label_lens) if label_lens else np.array([0])

    print(f"\n{'='*50}")
    print(f"📊 {name} ({len(feat_lens)} 样本)")
    print(f"{'='*50}")

    print(f"\n  特征长度 (点数):")
    print(f"    最小:    {feat_arr.min()}")
    print(f"    25%:     {int(np.percentile(feat_arr, 25))}")
    print(f"    中位数:  {int(np.median(feat_arr))}")
    print(f"    75%:     {int(np.percentile(feat_arr, 75))}")
    print(f"    90%:     {int(np.percentile(feat_arr, 90))}")
    print(f"    95%:     {int(np.percentile(feat_arr, 95))}")
    print(f"    99%:     {int(np.percentile(feat_arr, 99))}")
    print(f"    最大:    {feat_arr.max()}")
    print(f"    均值:    {feat_arr.mean():.1f}")
    print(f"    >600:    {(feat_arr > 600).sum()} ({(feat_arr > 600).mean()*100:.2f}%)")
    print(f"    >1200:   {(feat_arr > 1200).sum()} ({(feat_arr > 1200).mean()*100:.2f}%)")
    print(f"    >1800:   {(feat_arr > 1800).sum()} ({(feat_arr > 1800).mean()*100:.2f}%)")
    print(f"    >2400:   {(feat_arr > 2400).sum()} ({(feat_arr > 2400).mean()*100:.2f}%)")

    if label_lens:
        print(f"\n  标签长度 (含SOS/EOS):")
        print(f"    最小:    {lab_arr.min()}")
        print(f"    中位数:  {int(np.median(lab_arr))}")
        print(f"    95%:     {int(np.percentile(lab_arr, 95))}")
        print(f"    99%:     {int(np.percentile(lab_arr, 99))}")
        print(f"    最大:    {lab_arr.max()}")
        print(f"    >200:    {(lab_arr > 200).sum()} ({(lab_arr > 200).mean()*100:.2f}%)")


if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))
    analyze_dir(os.path.join(base, "data", "raw", "train"), "手写训练集")
    analyze_dir(os.path.join(base, "data", "raw", "symbols"), "单字符号集")
    analyze_dir(os.path.join(base, "data", "raw", "synthetic"), "合成训练集")
    analyze_dir(os.path.join(base, "data", "raw", "valid"), "验证集")
