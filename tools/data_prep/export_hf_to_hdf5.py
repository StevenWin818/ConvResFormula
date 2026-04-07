"""
将 Hugging Face 公式数据集转换为 HDF5 懒加载格式
"""
import os
import re
import sys
from pathlib import Path

import h5py
import numpy as np
import cv2
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.tokenizer import normalize_sub_sup


def normalize_latex(formula: str) -> str:
    """执行与 BPE 训练一致的清洗，降低标签分布漂移。"""
    if not isinstance(formula, str):
        formula = "" if formula is None else str(formula)

    formula = formula.strip()
    if not formula:
        return ""

    formula = normalize_sub_sup(formula)
    formula = formula.replace(r"/dots", r"\dots ")
    formula = formula.replace(r"...", r"\dots ")
    formula = formula.replace(r"\not =", r"\ne ")
    formula = re.sub(r"\^\s*\{?\s*\\prime\s*\}?", "'", formula)
    formula = formula.replace(r"\prime", "'")
    formula = re.sub(r"\\left(?![a-zA-Z])", "", formula)
    formula = re.sub(r"\\right(?![a-zA-Z])", "", formula)

    spaced_funcs = ["sinh", "cosh", "tanh", "log", "sin", "cos", "tan", "lim", "exp", "max", "min"]
    for sf in spaced_funcs:
        spaced_pattern = r"(?:\s|\\[,;!]|\\quad|\\qquad)*".join(list(sf))
        pattern = rf"(?<![a-zA-Z\\]){spaced_pattern}(?![a-zA-Z])"
        formula = re.sub(pattern, r"\\" + sf, formula)

    alias_map = {
        r"\lnot": r"\neg",
        r"\land": r"\wedge",
        r"\lor": r"\vee",
        r"\neq": r"\ne",
        r"\leq": r"\le",
        r"\geq": r"\ge",
        r"\lt": "<",
        r"\gt": ">",
        r"\thickapprox": r"\approx",
        r"\varpropto": r"\propto",
        r"\implies": r"\Longrightarrow",
        r"\iff": r"\Longleftrightarrow",
        r"\gets": r"\leftarrow",
        r"\rightarrow": r"\to",
        r"\tbinom": r"\binom",
        r"\dbinom": r"\binom",
        r"\choose": r"\binom",
        r"\tfrac": r"\frac",
        r"\cfrac": r"\frac",
        r"\dfrac": r"\frac",
        r"\widehat": r"\hat",
        r"\widetilde": r"\tilde",
        r"\bigtriangleup": r"\triangle",
        r"\vartriangle": r"\triangle",
        r"\smallfrown": r"\frown",
        r"\smallsmile": r"\smile",
        r"\square": r"\Box",
        r"\ldots": r"\dots",
        r"\dotsc": r"\dots",
        r"\dotso": r"\dots",
        "…": r"\dots",
        r"\dotsb": r"\cdots",
        r"\dotsi": r"\cdots",
        r"\dotsm": r"\cdots",
        "⋯": r"\cdots",
        r"\lbrack": "[",
        r"\rbrack": "]",
        r"\lbrace": r"\{",
        r"\rbrace": r"\}",
        r"\Vert": r"\|",
        r"\lVert": r"\|",
        r"\rVert": r"\|",
        r"\vert": "|",
        r"\shortmid": r"\mid",
        r"\varnothing": r"\emptyset",
        r"\hslash": r"\hbar",
        r"\ast": "*",
        r"\colon": ":",
        "·": r"\cdot",
        "×": r"\times",
        "²": "^2",
    }
    for src, dst in alias_map.items():
        formula = formula.replace(src, dst)

    return formula

def process_and_encode_image(pil_img: Image.Image) -> bytes:
    """将带有透明通道的 PIL Image 转换为白底灰度 cv2 编码流"""
    # 1. 处理透明背景 (RGBA -> 白底 RGB)
    if pil_img.mode in ('RGBA', 'LA') or (pil_img.mode == 'P' and 'transparency' in pil_img.info):
        alpha = pil_img.convert('RGBA').split()[-1]
        bg = Image.new("RGBA", pil_img.size, (255, 255, 255, 255))
        bg.paste(pil_img, mask=alpha)
        pil_img = bg.convert('RGB')
    else:
        pil_img = pil_img.convert('RGB')
        
    # 2. 转为灰度 NumPy 数组
    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2GRAY)
    
    # 3. 编码为 PNG 字节流以节省空间
    success, encoded = cv2.imencode('.png', cv_img)
    if not success:
        raise RuntimeError("Image encoding failed")
        
    return encoded.tobytes()

def main():
    print("正在加载数据集...")
    ds = load_dataset(
        "parquet",
        data_files=r"C:\Projects\LatexProject\datasets\wikipedia-latex-formulas-319k.parquet",
        split="train",
    )
    
    # 拆分验证集 (例如保留最后 5000 张用于评估)
    eval_size = 5000
    train_ds = ds.select(range(len(ds) - eval_size))
    eval_ds = ds.select(range(len(ds) - eval_size, len(ds)))
    
    def export_h5(dataset, output_path):
        print(f"正在导出 {output_path} (共 {len(dataset)} 条)...")
        with h5py.File(output_path, 'w') as f:
            dt_images = h5py.vlen_dtype(np.dtype('uint8'))
            dt_labels = h5py.special_dtype(vlen=str)
            
            dset_images = f.create_dataset('images', (len(dataset),), dtype=dt_images, chunks=True)
            dset_labels = f.create_dataset('labels', (len(dataset),), dtype=dt_labels, chunks=True)
            
            for i, item in enumerate(tqdm(dataset)):
                img_bytes = process_and_encode_image(item['image'])
                dset_images[i] = np.frombuffer(img_bytes, dtype=np.uint8)
                raw_formula = item.get('formula', '')
                cleaned_formula = normalize_latex(raw_formula)
                if not cleaned_formula:
                    cleaned_formula = str(raw_formula).strip()
                dset_labels[i] = cleaned_formula
                
    export_h5(train_ds, "datasets/wiki_train_314k.h5")
    export_h5(eval_ds, "datasets/wiki_eval_5k.h5")
    print("全部 HDF5 导出完成！")

if __name__ == "__main__":
    main()