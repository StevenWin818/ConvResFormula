"""
将 Hugging Face 公式数据集转换为 HDF5 懒加载格式
"""
import h5py
import numpy as np
import cv2
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

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
                dset_labels[i] = item['formula']
                
    export_h5(train_ds, "datasets/wiki_train_314k.h5")
    export_h5(eval_ds, "datasets/wiki_eval_5k.h5")
    print("全部 HDF5 导出完成！")

if __name__ == "__main__":
    main()