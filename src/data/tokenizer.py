import re

def normalize_sub_sup(latex_str):
    """
    上下标规范化：将 x_2, x^2 转换为 x_{2}, x^{2}
    避免 x_2 和 x_{2} 混用导致的模型困惑。
    """
    if not latex_str: return latex_str
    
    # 逻辑：匹配 _ 或 ^ 后面紧跟一个字母或数字的情况，将其包裹在 {} 中
    # 排除情况：如果已经是 _{...} 或 ^{...} 则不匹配
    # 示例: "x_2 + y^n" -> "x_{2} + y^{n}"
    # 示例: "x_{2}" -> 不变
    # 示例: "x_\alpha" -> 不变 (只处理单字符)
    pattern = r'([_^])([a-zA-Z0-9])'
    return re.sub(pattern, r'\1{\2}', latex_str)

def get_whitelist():
    """
    返回强制保留的 Token 白名单。
    即使频次很低，这些数学符号也必须保留，不能被当做噪声清洗。
    """
    whitelist = {
        # === 1. 基础符号与括号 ===
        "{", "}", "(", ")", "[", "]", "|", "+", "-", "=", "<", ">", 
        ".", ",", ";", ":", "!", "?", "/", "*", "'", "\"",
        
        # === 2. 转义字符 (必须保留) ===
        r"\%", r"\$", r"\#", r"\&", r"\_", r"\{", r"\}", r"\\",
        
        # === 3. 常用数学命令 ===
        r"\frac", r"\sqrt", r"\sum", r"\prod", r"\int", r"\oint", 
        r"\lim", r"\log", r"\ln", r"\sin", r"\cos", r"\tan", 
        r"\cot", r"\sec", r"\csc", r"\exp", r"\min", r"\max",
        r"\sup", r"\inf", r"\det", r"\dim", r"\ker", r"\deg",
        
        # === 4. 希腊字母 (小写 & 大写) ===
        r"\alpha", r"\beta", r"\gamma", r"\delta", r"\epsilon", r"\zeta",
        r"\eta", r"\theta", r"\iota", r"\kappa", r"\lambda", r"\mu",
        r"\nu", r"\xi", r"\pi", r"\rho", r"\sigma", r"\tau",
        r"\upsilon", r"\phi", r"\chi", r"\psi", r"\omega",
        r"\Gamma", r"\Delta", r"\Theta", r"\Lambda", r"\Xi",
        r"\Pi", r"\Sigma", r"\Upsilon", r"\Phi", r"\Psi", r"\Omega",
        
        # === 5. 逻辑、集合与关系 ===
        r"\in", r"\notin", r"\subset", r"\subseteq", r"\supset", r"\supseteq",
        r"\cap", r"\cup", r"\forall", r"\exists", r"\neg", r"\to", 
        r"\rightarrow", r"\leftarrow", r"\Rightarrow", r"\Leftrightarrow",
        r"\infty", r"\partial", r"\nabla", r"\neq", r"\approx", r"\sim",
        r"\propto", r"\le", r"\ge", r"\leq", r"\geq", r"\ll", r"\gg",
        r"\pm", r"\mp", r"\times", r"\div", r"\cdot", r"\bullet",
        
        # === 6. 用户指定的低频保留词 ===
        r"\sqsubseteq", r"\triangleleft", r"\bigcirc", r"\dotso", r"\And", 
        r"\dots", r"\cdots", # 补充类似的省略号
        
        # === 7. 布局与修饰 ===
        r"\left", r"\right", r"\hat", r"\bar", r"\dot", r"\ddot", 
        r"\tilde", r"\vec", r"\overline", r"\underline",
        r"\begin", r"\end", # 矩阵环境需要
        
        # === 8. 空格与特殊 ===
        r"\,", r"\;", r"\!", r"\quad", r"\qquad",

        # === 9. 旧词表兼容 (确保向后兼容) ===
        r"\Biggl", r"\Biggr",                          # 括号尺寸
        r"\Longrightarrow", r"\implies",                # 箭头与逻辑
        r"\coth",                                       # 双曲余切
        r"\displaystyle",                               # 排版
        r"\dotsm",                                      # 乘法省略号
        r"\gt", r"\lt",                                 # 比较符
        r"\it", r"\mbox",                               # 样式/文本框
        r"\text", r"\textbf", r"\textit",               # 文本格式
        r"\textrm", r"\textsf", r"\texttt",
    }
    return whitelist

import re

def tokenize_latex(latex_str):
    """
    增强版 Tokenizer (包含两阶段标签规范化)
    """
    if not latex_str: return []
    latex_str = latex_str.strip()
    
    # =======================================================
    # 阶段 1：字符串级别预处理
    # =======================================================
    # 1.1 替换错误的斜杠与连续标点
    latex_str = latex_str.replace(r"/dots", r"\dots ")
    latex_str = latex_str.replace(r"/ldots", r"\dots ")
    latex_str = latex_str.replace(r"...", r"\dots ")
    latex_str = latex_str.replace(r"\cdot\cdot\cdot", r"\cdots ")
    
    # 1.2 修复不等号与双竖线歧义
    latex_str = latex_str.replace(r"\not =", r"\ne ")
    latex_str = latex_str.replace(r"\not=", r"\ne ")
    latex_str = latex_str.replace(r"| |", r"\|")
    latex_str = latex_str.replace(r"||", r"\|")
    
    # 1.3 统一导数撇号 (直接打击报告中 230 次的 \prime 错判)
    # 强行将 ^\prime, ^{\prime}, ^{ \prime } 甚至单独的 \prime 全部统一为单撇号 '
    latex_str = re.sub(r'\^\s*\{?\s*\\prime\s*\}?', "'", latex_str)
    latex_str = latex_str.replace(r"\prime", "'")

    # 1.4 【强力优化】去除无视觉特征的排版命令 (大幅减少漏认和冗余)
    latex_str = re.sub(r"\\left(?![a-zA-Z])", "", latex_str)
    latex_str = re.sub(r"\\right(?![a-zA-Z])", "", latex_str)

    # 1.5 处理被空格隔开的数学函数 (直接打击报告中的最高频错误)
    spaced_funcs = [
        "sinh", "cosh", "tanh", "coth",
        "log", "sin", "cos", "tan", "cot", "sec", "csc", 
        "lim", "exp", "max", "min", "arg", "det", "sup", "inf"
    ]
    
    for sf in spaced_funcs:
        # 将单词拆成字母，并在每个字母中间插入允许任意数量的空格或 LaTeX 空格符的匹配模式
        spaced_pattern = r"(?:\s|\\[,;!]|\\quad|\\qquad)*".join(list(sf))
        
        # 【关键修复】前置否定环视增加 \\，防止把已经正确的 \sin 错误替换为 \\sin
        pattern = rf"(?<![a-zA-Z\\]){spaced_pattern}(?![a-zA-Z])"
        
        # 强行替换为标准的宏命令，如 \sin
        latex_str = re.sub(pattern, r"\\" + sf, latex_str)

    # =======================================================
    # 阶段 2：正则切词与 Token 级别映射
    # =======================================================
    pattern = r"(\\[a-zA-Z]+)|(\\[^\s])|([a-zA-Z0-9])|([^\s])"
    matches = re.findall(pattern, latex_str)
    
    # 单体同义词字典（含用户指定的非严格等价映射）
    alias_map = {
        # === 逻辑与关系 ===
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

        # === 箭头 ===
        r"\gets": r"\leftarrow",
        r"\rightarrow": r"\to",

        # === 分数与组合（图像难以区分） ===
        r"\tbinom": r"\binom",
        r"\dbinom": r"\binom",
        r"\choose": r"\binom",
        r"\tfrac": r"\frac",
        r"\cfrac": r"\frac",
        r"\dfrac": r"\frac",

        # === 装饰符（图像难以区分） ===
        r"\widehat": r"\hat",
        r"\widetilde": r"\tilde",

        # === 装饰符与几何（等价组） ===
        r"\bigtriangleup": r"\triangle",
        r"\vartriangle": r"\triangle",
        r"\smallfrown": r"\frown",
        r"\smallsmile": r"\smile",
        r"\square": r"\Box",

        # === 省略号 ===
        r"\ldots": r"\dots",
        r"\dotsc": r"\dots",
        r"\dotso": r"\dots",
        "…": r"\dots",
        r"\dotsb": r"\cdots",
        r"\dotsi": r"\cdots",
        r"\dotsm": r"\cdots",
        "⋯": r"\cdots",

        # === 括号与界定符 ===
        r"\lbrack": "[",
        r"\rbrack": "]",
        r"\lbrace": r"\{",
        r"\rbrace": r"\}",
        r"\Vert": r"\|",
        r"\lVert": r"\|",
        r"\rVert": r"\|",
        r"\vert": "|",
        r"\shortmid": r"\mid",

        # === 特殊符号与常量 ===
        r"\varnothing": r"\emptyset",
        r"\hslash": r"\hbar",
        r"\ast": "*",
        r"\colon": ":",

        # === Unicode 清洗 ===
        "·": r"\cdot",
        "×": r"\times",
        "²": "^2",
    }
    
    tokens = []
    for match in matches:
        token = next(filter(None, match), None)
        if token:
            standard_token = alias_map.get(token, token)
            tokens.append(standard_token)
            
    return tokens

if __name__ == "__main__":
    # 测试规范化
    s = "x_2 + y^n + z_{k}"
    print(f"原串: {s}")
    print(f"规范: {normalize_sub_sup(s)}") # 期望: x_{2} + y^{n} + z_{k}
    
    # 测试白名单
    wl = get_whitelist()
    print(f"\n白名单大小: {len(wl)}")
    assert r"\%" in wl
    assert r"\sqsubseteq" in wl