import argparse
import csv
import random
from pathlib import Path
from typing import List, Optional

import cv2
import h5py
import numpy as np
from tqdm import tqdm


def render_trajectory_to_image(
    strokes: List[np.ndarray],
    image_height: int = 192,
    padding: int = 4,
    random_thickness: bool = False,
    blur_kernel: int = 0,            # <=1 表示不模糊
) -> Optional[np.ndarray]:
    """
    将 1D 轨迹坐标列表转换为 2D 图像。

    Args:
        strokes: List[np.ndarray]，每个数组 shape=[N, 2]，表示一个笔画的 (x, y)。
        image_height: 输出图像高度（固定）。
        padding: 四周留白像素。
        random_thickness: 是否在 {1, 2} 中随机线宽（用于轻度增强）。
        blur_kernel: 高斯模糊核大小，<=1 表示不模糊。
    """
    valid_strokes = [s for s in strokes if s is not None and len(s) > 0]
    if not valid_strokes:
        return None

    all_points = np.vstack(valid_strokes)
    min_x, min_y = np.min(all_points, axis=0)
    max_x, max_y = np.max(all_points, axis=0)

    width = float(max_x - min_x)
    height = float(max_y - min_y)
    if height <= 1e-8:
        height = 1.0

    scale = max((image_height - 2 * padding) / height, 1e-8)
    target_width = max(int(round(width * scale)) + 2 * padding, 2 * padding + 1)
    canvas = np.zeros((image_height, target_width), dtype=np.uint8)

    thickness = int(np.random.choice([1, 2])) if random_thickness else 2

    for stroke in valid_strokes:
        scaled_stroke = np.zeros_like(stroke, dtype=np.int32)
        scaled_stroke[:, 0] = ((stroke[:, 0] - min_x) * scale).astype(np.int32) + padding
        scaled_stroke[:, 1] = ((stroke[:, 1] - min_y) * scale).astype(np.int32) + padding

        if len(scaled_stroke) == 1:
            p = tuple(scaled_stroke[0].tolist())
            cv2.circle(canvas, p, radius=max(1, thickness - 1), color=255, thickness=-1)
        else:
            pts = scaled_stroke.reshape((-1, 1, 2))
            cv2.polylines(
                canvas,
                [pts],
                isClosed=False,
                color=255,
                thickness=thickness,
                lineType=cv2.LINE_AA,
            )

    if blur_kernel > 1 and blur_kernel % 2 == 1:
        canvas = cv2.GaussianBlur(canvas, (blur_kernel, blur_kernel), 0)

    return canvas


def flat_to_strokes(points: np.ndarray, stroke_lengths: np.ndarray) -> List[np.ndarray]:
    """将 H5 中平铺的 points + stroke_lengths 还原为 strokes 列表。"""
    strokes: List[np.ndarray] = []
    cursor = 0
    for sl in stroke_lengths:
        seg_len = int(sl)
        if seg_len > 0:
            strokes.append(points[cursor: cursor + seg_len].copy())
        cursor += seg_len
    return strokes


def decode_label(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    return str(value)


def sanitize_for_filename(text: str, max_len: int = 48) -> str:
    safe = []
    for ch in text:
        if ch.isalnum() or ch in {"_", "-"}:
            safe.append(ch)
        else:
            safe.append("_")
    collapsed = "".join(safe).strip("_")
    if not collapsed:
        collapsed = "nolabel"
    return collapsed[:max_len]


def choose_indices(total: int, sample_size: int, seed: int, full_export: bool) -> List[int]:
    if total <= 0:
        return []
    if full_export:
        return list(range(total))

    n = min(sample_size, total)
    rng = random.Random(seed)
    return rng.sample(range(total), n)


def export_h5_train_images(
    h5_path: str,
    output_dir: str,
    sample_size: int = 100,
    full_export: bool = False,      # 是否全量导出（忽略 sample_size）
    seed: int = 2026,
    image_height: int = 128,
    padding: int = 4,
    print_limit: int = 100,
    random_thickness: bool = True,  # 是否随机线宽
    blur_kernel: int = 1,           # <=1 表示不模糊
) -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if sample_size <= 0 and not full_export:
        raise ValueError("sample_size 必须 > 0，或使用 --full_export")

    with h5py.File(h5_path, "r") as f:
        required_keys = [
            "sample_point_offsets",
            "sample_stroke_offsets",
            "stroke_lengths",
            "points",
            "labels",
        ]
        for key in required_keys:
            if key not in f:
                raise KeyError(f"H5 缺少关键字段: {key}")

        pt_off = np.asarray(f["sample_point_offsets"])
        st_off = np.asarray(f["sample_stroke_offsets"])
        d_points = f["points"]
        d_stroke_lengths = f["stroke_lengths"]
        d_labels = f["labels"]

        total = int(pt_off.shape[0]) - 1
        indices = choose_indices(total, sample_size, seed, full_export)

        print(f"📂 输入 H5: {h5_path}")
        print(f"💾 输出目录: {out_dir}")
        print(f"📊 样本总数: {total}")
        print(f"🎯 本次导出: {len(indices)} 张")

        meta_csv = out_dir / "metadata.csv"
        with open(meta_csv, "w", newline="", encoding="utf-8") as fp:
            writer = csv.writer(fp)
            writer.writerow([
                "file_name",
                "h5_idx",
                "label",
                "num_strokes",
                "num_points",
                "height",
                "width",
            ])

            exported = 0
            skipped = 0
            for rank, idx in enumerate(tqdm(indices, desc="Rendering"), start=1):
                p0 = int(pt_off[idx])
                p1 = int(pt_off[idx + 1])
                s0 = int(st_off[idx])
                s1 = int(st_off[idx + 1])

                if p1 <= p0:
                    skipped += 1
                    continue

                points = np.asarray(d_points[p0:p1], dtype=np.float32)
                stroke_lengths = np.asarray(d_stroke_lengths[s0:s1], dtype=np.int32)
                strokes = flat_to_strokes(points, stroke_lengths)

                image = render_trajectory_to_image(
                    strokes,
                    image_height=image_height,
                    padding=padding,
                    random_thickness=random_thickness,
                    blur_kernel=blur_kernel,
                )
                if image is None:
                    skipped += 1
                    continue

                label = decode_label(d_labels[idx])
                safe_label = sanitize_for_filename(label)
                file_name = f"{rank:06d}_idx{idx:07d}_{safe_label}.png"
                save_path = out_dir / file_name
                ok = cv2.imwrite(str(save_path), image)
                if not ok:
                    skipped += 1
                    continue

                exported += 1
                writer.writerow([
                    file_name,
                    idx,
                    label,
                    len(strokes),
                    int(points.shape[0]),
                    int(image.shape[0]),
                    int(image.shape[1]),
                ])

                if rank <= print_limit:
                    print(
                        f"[{rank:03d}] idx={idx:<7} "
                        f"strokes={len(strokes):<3} points={points.shape[0]:<5} -> {save_path}"
                    )

    print(f"\n✅ 导出完成: {exported} 张，跳过: {skipped} 张")
    print(f"🧾 元数据: {meta_csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从 train.h5 渲染 2D 公式图片")
    parser.add_argument(
        "--h5",
        default=r"C:\Projects\LatexProject\datasets\synthetic.h5",
        help="训练集 H5 路径",
    )
    parser.add_argument(
        "--output_dir",
        default=r"C:\Projects\LatexProject\2Ddatasets\synthetic",
        help="输出图片目录",
    )
    parser.add_argument("--sample_size", type=int, default=1000, help="抽样导出数量")
    parser.add_argument("--seed", type=int, default=2026, help="抽样随机种子")
    parser.add_argument("--image_height", type=int, default=192, help="输出图像高度")
    parser.add_argument("--padding", type=int, default=4, help="图像边缘留白")
    parser.add_argument("--print_limit", type=int, default=100, help="最多打印前 N 条日志")
    parser.add_argument(
        "--blur_kernel",
        type=int,
        default=1,
        help="高斯模糊核(奇数, <=1 关闭)",
    )
    parser.add_argument(
        "--fixed_thickness",
        action="store_true",
        help="固定线宽为 2（关闭随机线宽增强）",
    )
    parser.add_argument(
        "--full_export",
        action="store_true",
        help="全量导出训练集（忽略 sample_size）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_h5_train_images(
        h5_path=args.h5,
        output_dir=args.output_dir,
        sample_size=args.sample_size,
        full_export=args.full_export,
        seed=args.seed,
        image_height=args.image_height,
        padding=args.padding,
        print_limit=args.print_limit,
        random_thickness=not args.fixed_thickness,
        blur_kernel=args.blur_kernel,
    )


if __name__ == "__main__":
    main()