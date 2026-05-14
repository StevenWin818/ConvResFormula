"""删除 data/ 目录下所有与 inkml 混在一起的旧 .npz 缓存文件"""
import os
import glob

data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

count = 0
size = 0

for npz in glob.glob(os.path.join(data_dir, "**", "*.npz"), recursive=True):
    sz = os.path.getsize(npz)
    os.remove(npz)
    count += 1
    size += sz

# 同时清理残留的 .tmp 文件
for tmp in glob.glob(os.path.join(data_dir, "**", "*.tmp"), recursive=True):
    os.remove(tmp)

print(f"✅ 已删除 {count} 个 .npz 文件, 释放 {size / 1024 / 1024:.1f} MB")
