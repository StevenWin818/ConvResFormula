import pickle
import os

# 路径配置
CACHE_PATH = "cache/vocab.pkl"
OUTPUT_HEADER = "vocab.h"  # 确保路径对应你的 C++ include 目录

def main():
    if not os.path.exists(CACHE_PATH):
        print("找不到 cache/vocab.pkl")
        return

    with open(CACHE_PATH, "rb") as f:
        data = pickle.load(f)
        char2id = data["char2id"]
        id2char = data["id2char"]

    # 打印词表
    print(f"词表大小: {len(id2char)}")
    print("-" * 40)
    for i in range(len(id2char)):
        token = id2char.get(i, "???")
        print(f"  {i:>4d} | {token}")
    print("-" * 40)

    # 生成 C++ 头文件
    print(f"正在生成 {OUTPUT_HEADER} ...")
    with open(OUTPUT_HEADER, "w", encoding="utf-8") as f:
        f.write("#ifndef VOCAB_H\n")
        f.write("#define VOCAB_H\n\n")
        f.write("#include <vector>\n")
        f.write("#include <string>\n\n")
        f.write("static const std::vector<std::string> ID2CHAR = {\n")
        
        # 按 ID 顺序写入
        max_id = max(id2char.keys())
        for i in range(max_id + 1):
            token = id2char.get(i, "")
            # 转义特殊字符
            token = token.replace("\\", "\\\\").replace("\"", "\\\"")
            f.write(f'    "{token}", // {i}\n')
            
        f.write("};\n\n")
        f.write("#endif // VOCAB_H\n")
    
    print("C++ 词表头文件生成完毕！")

if __name__ == "__main__":
    main()