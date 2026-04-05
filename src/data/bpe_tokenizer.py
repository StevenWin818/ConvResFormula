"""BPE 分词器封装。"""
"""
全场景 LaTeX BPE Tokenizer 训练脚本
融合了维基百科 319k Parquet 数据与本地手写/合成 CSV 数据
"""

import os
import pandas as pd
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import WhitespaceSplit, Sequence, Split
from tokenizers import Regex
from tqdm import tqdm

# 借用你写好的优异清理函数
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))
from src.data.tokenizer import normalize_sub_sup

def clean_latex_for_training(latex_str):
    if not isinstance(latex_str, str) or not latex_str.strip(): return ""
    latex_str = normalize_sub_sup(latex_str)
    
    # 基础清洗
    latex_str = latex_str.replace(r"/dots", r"\dots ")
    latex_str = latex_str.replace(r"...", r"\dots ")
    latex_str = latex_str.replace(r"\not =", r"\ne ")
    
    import re
    latex_str = re.sub(r'\^\s*\{?\s*\\prime\s*\}?', "'", latex_str)
    latex_str = latex_str.replace(r"\prime", "'")
    latex_str = re.sub(r"\\left(?![a-zA-Z])", "", latex_str)
    latex_str = re.sub(r"\\right(?![a-zA-Z])", "", latex_str)
    
    spaced_funcs = ["sinh", "cosh", "tanh", "log", "sin", "cos", "tan", "lim", "exp", "max", "min"]
    for sf in spaced_funcs:
        spaced_pattern = r"(?:\s|\\[,;!]|\\quad|\\qquad)*".join(list(sf))
        pattern = rf"(?<![a-zA-Z\\]){spaced_pattern}(?![a-zA-Z])"
        latex_str = re.sub(pattern, r"\\" + sf, latex_str)

    # 与 tokenizer.py 保持一致的同义词清洗，降低标签异构度
    alias_map = {
        # 逻辑与关系
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

        # 箭头
        r"\gets": r"\leftarrow",
        r"\rightarrow": r"\to",

        # 分数与组合
        r"\tbinom": r"\binom",
        r"\dbinom": r"\binom",
        r"\choose": r"\binom",
        r"\tfrac": r"\frac",
        r"\cfrac": r"\frac",
        r"\dfrac": r"\frac",

        # 装饰符
        r"\widehat": r"\hat",
        r"\widetilde": r"\tilde",

        # 装饰符与几何
        r"\bigtriangleup": r"\triangle",
        r"\vartriangle": r"\triangle",
        r"\smallfrown": r"\frown",
        r"\smallsmile": r"\smile",
        r"\square": r"\Box",

        # 省略号
        r"\ldots": r"\dots",
        r"\dotsc": r"\dots",
        r"\dotso": r"\dots",
        "…": r"\dots",
        r"\dotsb": r"\cdots",
        r"\dotsi": r"\cdots",
        r"\dotsm": r"\cdots",
        "⋯": r"\cdots",

        # 括号与界定符
        r"\lbrack": "[",
        r"\rbrack": "]",
        r"\lbrace": r"\{",
        r"\rbrace": r"\}",
        r"\Vert": r"\|",
        r"\lVert": r"\|",
        r"\rVert": r"\|",
        r"\vert": "|",
        r"\shortmid": r"\mid",

        # 特殊符号与常量
        r"\varnothing": r"\emptyset",
        r"\hslash": r"\hbar",
        r"\ast": "*",
        r"\colon": ":",

        # Unicode 清洗
        "·": r"\cdot",
        "×": r"\times",
        "²": "^2",
    }
    for k, v in alias_map.items():
        latex_str = latex_str.replace(k, v)
        
    return latex_str

def train_universal_tokenizer(parquet_path, csv_paths, vocab_size=4000, output_path="tokenizer_bpe.json"):
    print("全场景 LaTeX BPE 词表生成...")
    corpus = []

    # ==========================================
    # 1. 挂载 Wikipedia 319k 
    # ==========================================
    if os.path.exists(parquet_path):
        print(f"正在加载维基百科全量数据: {parquet_path}")
        df = pd.read_parquet(parquet_path)
        text_column = 'formula' if 'formula' in df.columns else df.columns[0] 
        
        for text in tqdm(df[text_column].dropna(), desc="清洗 Wikipedia 数据"):
            corpus.append(clean_latex_for_training(text))
    else:
        print(f"⚠️ 未找到 Parquet 文件: {parquet_path}")

    # ==========================================
    # 2. 挂载本地训练集的 Metadata (CSV)
    # ==========================================
    for csv_path in csv_paths:
        if not os.path.exists(csv_path): continue
        print(f"正在加载本地数据集: {csv_path}")
        df_csv = pd.read_csv(csv_path)
        if 'label' in df_csv.columns:
            for text in tqdm(df_csv['label'].dropna(), desc=f"清洗 {os.path.basename(csv_path)}"):
                corpus.append(clean_latex_for_training(text))

    print(f"✅ 语料库构建完成！总计投入 {len(corpus)} 条公式。")

    # ==========================================
    # 3. 配置并训练 Rust 底层 BPE
    # ==========================================
    special_tokens = ["[PAD]", "[UNK]", "[BOS]", "[EOS]", "[MASK]"]
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

    # 保护 LaTeX 宏命令不被切碎
    pre_tokenizer = Sequence([
        WhitespaceSplit(),
        Split(Regex(r"\\[a-zA-Z]+"), behavior="isolated") 
    ])
    tokenizer.pre_tokenizer = pre_tokenizer

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        show_progress=True,
    )

    # 训练 (Rust 核心，极速计算)
    tokenizer.train_from_iterator(corpus, trainer)
    tokenizer.save(output_path)
    print(f"Tokenizer 生成完成！已存为 {output_path}")

if __name__ == "__main__":
    # 配置路径
    WIKI_PARQUET = r"C:\Projects\LatexProject\datasets\wikipedia-latex-formulas-319k.parquet"
    LOCAL_CSVS = [
        r"C:\Projects\LatexProject\2Ddatasets\train_metadata.csv",
        r"C:\Projects\LatexProject\2Ddatasets\synthetic_metadata.csv",
        r"C:\Projects\LatexProject\2Ddatasets\val_metadata.csv",
        r"C:\Projects\LatexProject\2Ddatasets\test_metadata.csv",
    ]
    
    train_universal_tokenizer(WIKI_PARQUET, LOCAL_CSVS, vocab_size=4000)