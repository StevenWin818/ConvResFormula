"""
将 InkML 数据缓存导出为 HDF5（每个数据集一个文件，不分片）。

默认输出：
- train.h5
- val.h5
- synthetic.h5
- symbols.h5

输出目录默认：C:\\Projects\\LatexProject\\datasets
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple, Any

import numpy as np
import h5py
from tqdm import tqdm


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.dataset_1d import parse_inkml


@dataclass
class SampleMeta:
    path: str
    label: str
    n_strokes: int
    n_points: int


def _collect_inkml_paths(root_dir: str) -> List[str]:
    paths: List[str] = []
    for root, _, files in os.walk(root_dir):
        for name in files:
            if name.endswith(".inkml"):
                paths.append(os.path.join(root, name))
    paths.sort()
    return paths


def _pass1_collect_meta(file_paths: List[str], desc: str) -> Tuple[List[SampleMeta], Dict[str, int]]:
    records: List[SampleMeta] = []
    skip_no_label = 0
    skip_empty_strokes = 0
    parse_fail = 0

    for path in tqdm(file_paths, desc=f"{desc} | pass1"):
        try:
            strokes, label = parse_inkml(path)
        except Exception:
            parse_fail += 1
            continue

        if not label or str(label).strip().lower() == "unknown":
            skip_no_label += 1
            continue
        if not strokes:
            skip_empty_strokes += 1
            continue

        n_strokes = len(strokes)
        n_points = int(sum(int(s.shape[0]) for s in strokes if s is not None and s.size > 0))
        if n_strokes <= 0 or n_points <= 0:
            skip_empty_strokes += 1
            continue

        records.append(
            SampleMeta(
                path=path,
                label=str(label).strip(),
                n_strokes=n_strokes,
                n_points=n_points,
            )
        )

    stats = {
        "total_files": len(file_paths),
        "valid_samples": len(records),
        "skip_no_label": skip_no_label,
        "skip_empty_strokes": skip_empty_strokes,
        "parse_fail": parse_fail,
        "total_strokes": int(sum(r.n_strokes for r in records)),
        "total_points": int(sum(r.n_points for r in records)),
    }
    return records, stats


def _create_dataset(
    output_path: str,
    source_dir: str,
    records: List[SampleMeta],
    compression: str,
    desc: str,
) -> Dict[str, Any]:
    n_samples = len(records)
    total_strokes = int(sum(r.n_strokes for r in records))
    total_points = int(sum(r.n_points for r in records))

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    compression_arg = None if compression == "none" else compression
    str_dtype = h5py.string_dtype(encoding="utf-8")

    point_chunk = max(1, min(total_points if total_points > 0 else 1, 65536))
    stroke_chunk = max(1, min(total_strokes if total_strokes > 0 else 1, 65536))

    with h5py.File(output_path, "w") as f:
        f.attrs["format"] = "inkml_strokes_v1"
        f.attrs["created_at"] = datetime.now().isoformat(timespec="seconds")
        f.attrs["source_dir"] = os.path.abspath(source_dir)
        f.attrs["sample_count"] = n_samples
        f.attrs["total_strokes"] = total_strokes
        f.attrs["total_points"] = total_points
        f.attrs["compression"] = compression

        d_paths = f.create_dataset("paths", shape=(n_samples,), dtype=str_dtype)
        d_labels = f.create_dataset("labels", shape=(n_samples,), dtype=str_dtype)

        d_sample_point_offsets = f.create_dataset("sample_point_offsets", shape=(n_samples + 1,), dtype=np.int64)
        d_sample_stroke_offsets = f.create_dataset("sample_stroke_offsets", shape=(n_samples + 1,), dtype=np.int64)

        d_stroke_lengths = f.create_dataset(
            "stroke_lengths",
            shape=(total_strokes,),
            dtype=np.int32,
            compression=compression_arg,
            shuffle=(compression_arg is not None),
            chunks=(stroke_chunk,),
        )

        d_points = f.create_dataset(
            "points",
            shape=(total_points, 2),
            dtype=np.float32,
            compression=compression_arg,
            shuffle=(compression_arg is not None),
            chunks=(point_chunk, 2),
        )

        point_cursor = 0
        stroke_cursor = 0

        for i, rec in enumerate(tqdm(records, desc=f"{desc} | pass2")):
            strokes, label = parse_inkml(rec.path)
            if not strokes or not label:
                raise RuntimeError(f"二次解析失败: {rec.path}")

            n_strokes = len(strokes)
            n_points = int(sum(int(s.shape[0]) for s in strokes if s is not None and s.size > 0))
            if n_strokes != rec.n_strokes or n_points != rec.n_points:
                raise RuntimeError(
                    f"样本统计不一致: {rec.path} | pass1=({rec.n_strokes},{rec.n_points}) pass2=({n_strokes},{n_points})"
                )

            d_paths[i] = os.path.relpath(rec.path, source_dir).replace("\\", "/")
            d_labels[i] = rec.label

            d_sample_point_offsets[i] = point_cursor
            d_sample_stroke_offsets[i] = stroke_cursor

            for stroke in strokes:
                if stroke is None or stroke.size == 0:
                    continue
                stroke = np.asarray(stroke, dtype=np.float32)
                stroke_len = int(stroke.shape[0])
                d_stroke_lengths[stroke_cursor] = stroke_len
                d_points[point_cursor: point_cursor + stroke_len, :] = stroke[:, :2]
                stroke_cursor += 1
                point_cursor += stroke_len

        d_sample_point_offsets[n_samples] = point_cursor
        d_sample_stroke_offsets[n_samples] = stroke_cursor

        if point_cursor != total_points or stroke_cursor != total_strokes:
            raise RuntimeError(
                f"写入计数不一致: points {point_cursor}/{total_points}, strokes {stroke_cursor}/{total_strokes}"
            )

    return {
        "output_path": output_path,
        "sample_count": n_samples,
        "total_strokes": total_strokes,
        "total_points": total_points,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="导出 InkML 到单文件 HDF5 缓存（每数据集一个文件）")
    parser.add_argument("--project_root", type=str, default=PROJECT_ROOT)
    parser.add_argument("--train_dir", type=str, default=os.path.join(PROJECT_ROOT, "data", "raw", "train"))
    parser.add_argument("--valid_dir", type=str, default=os.path.join(PROJECT_ROOT, "data", "raw", "valid"))
    parser.add_argument("--synthetic_dir", type=str, default=os.path.join(PROJECT_ROOT, "data", "raw", "synthetic"))
    parser.add_argument("--symbols_dir", type=str, default=os.path.join(PROJECT_ROOT, "data", "raw", "symbols"))
    parser.add_argument("--output_dir", type=str, default=r"C:\Projects\LatexProject\datasets")
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="*",
        default=["train", "valid", "synthetic", "symbols"],
        help="要导出的数据集名称，可选: train valid synthetic symbols",
    )
    parser.add_argument(
        "--compression",
        type=str,
        default="gzip",
        choices=["none", "gzip"],
        help="HDF5 压缩方式",
    )
    args = parser.parse_args()

    name_to_source = {
        "train": args.train_dir,
        "valid": args.valid_dir,
        "synthetic": args.synthetic_dir,
        "symbols": args.symbols_dir,
    }
    name_to_output = {
        "train": "train.h5",
        "valid": "val.h5",
        "synthetic": "synthetic.h5",
        "symbols": "symbols.h5",
    }

    unknown = [name for name in args.datasets if name not in name_to_source]
    if unknown:
        raise ValueError(f"不支持的数据集: {unknown}")

    os.makedirs(args.output_dir, exist_ok=True)

    summary: Dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "output_dir": os.path.abspath(args.output_dir),
        "compression": args.compression,
        "datasets": {},
    }

    for name in args.datasets:
        source_dir = name_to_source[name]
        output_path = os.path.join(args.output_dir, name_to_output[name])

        if not os.path.exists(source_dir):
            print(f"⚠️ 跳过 {name}: 目录不存在 -> {source_dir}")
            summary["datasets"][name] = {
                "status": "skipped_missing_dir",
                "source_dir": source_dir,
                "output_path": output_path,
            }
            continue

        file_paths = _collect_inkml_paths(source_dir)
        print(f"\n{'=' * 72}")
        print(f"   处理数据集: {name}")
        print(f"   Source: {source_dir}")
        print(f"   InkML 文件数: {len(file_paths)}")

        records, pass1_stats = _pass1_collect_meta(file_paths, desc=name)
        print(
            f"   有效样本: {pass1_stats['valid_samples']} | "
            f"points={pass1_stats['total_points']} | strokes={pass1_stats['total_strokes']}"
        )

        export_stats = _create_dataset(
            output_path=output_path,
            source_dir=source_dir,
            records=records,
            compression=args.compression,
            desc=name,
        )

        summary["datasets"][name] = {
            "status": "ok",
            "source_dir": source_dir,
            "output_path": output_path,
            **pass1_stats,
            **export_stats,
        }
        print(f"✅ 已写入: {output_path}")

    stats_path = os.path.join(args.output_dir, "inkml_hdf5_export_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n  汇总统计: {stats_path}")
    print(" 导出完成")


if __name__ == "__main__":
    main()
