#!/usr/bin/env python
"""MVTec AD -> YOLOv8-seg 数据集转换。

用法示例
--------
# 全量转换（默认按类别分 15 类、硬链接图片、70/15/15 分层划分）
python tools/convert_mvtec_to_yolo.py --src data/mvtec_raw --out data/yolo_mvtec

# 只转 3 个类别，单一 defect 类，多复制一份数据
python tools/convert_mvtec_to_yolo.py --src data/mvtec_raw --out data/yolo_small \
    --categories bottle screw metal_nut --classes unified --link copy

# 只看统计不落盘
python tools/convert_mvtec_to_yolo.py --src data/mvtec_raw --dry-run

输出
----
``images/{train,val,test}/``  ``labels/{train,val,test}/``  ``data.yaml``
``dataset_stats.json``（类别/缺陷类型分布、缺陷面积分位数、多边形逼近保真度）
``manifest.csv``（每张输出图到原始文件的可追溯映射）

图片默认用**硬链接**放置：同盘情况下不额外占用磁盘（MVTec 全量约 4.9GB），
硬链接失败自动回退为复制。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from induvision.mvtec import MVTecLayoutError, convert_dataset  # noqa: E402


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="MVTec AD -> YOLOv8-seg",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--src", default=str(REPO_ROOT / "data" / "mvtec_raw"))
    ap.add_argument("--out", default=str(REPO_ROOT / "data" / "yolo_mvtec"))
    ap.add_argument("--categories", nargs="*", default=None)
    ap.add_argument("--classes", dest="classes_mode", choices=("per-category", "unified"),
                    default="per-category", help="per-category=按产品分15类, unified=单一 defect 类")
    ap.add_argument("--train-ratio", type=float, default=0.70)
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epsilon-ratio", type=float, default=0.002,
                    help="approxPolyDP 的 epsilon = 该值 × 轮廓周长")
    ap.add_argument("--min-area", type=float, default=4.0, help="小于该像素面积的连通域丢弃")
    ap.add_argument("--link", choices=("hardlink", "copy", "symlink"), default="hardlink")
    ap.add_argument("--no-good-train", action="store_true",
                    help="不把官方 train/good 作为背景负样本放进 train")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--lenient", action="store_true", help="缺掩码时只告警不报错")
    ap.add_argument("--keep-empty-defects", action="store_true",
                    help="保留掩码面积低于 --min-area 的缺陷图（默认丢弃，避免其退化为背景负样本）")
    ap.add_argument("--dry-run", action="store_true", help="只扫描+统计，不写图片与标签")
    args = ap.parse_args(argv)

    t0 = time.time()
    try:
        rep = convert_dataset(
            Path(args.src),
            Path(args.out),
            classes_mode=args.classes_mode,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
            epsilon_ratio=args.epsilon_ratio,
            min_area=args.min_area,
            link=args.link,
            include_good_train=not args.no_good_train,
            categories=args.categories,
            strict=not args.lenient,
            workers=args.workers,
            write_files=not args.dry_run,
            keep_empty_defects=args.keep_empty_defects,
        )
    except MVTecLayoutError as exc:
        print(f"[error] {exc}")
        print("提示：先用 tools/download_mvtec.py 获取数据，或用 --src 指定正确目录")
        return 1
    except ValueError as exc:
        print(f"[error] 参数不合法: {exc}")
        return 2

    st = rep.stats
    print(f"[scan ] 源目录 {st['source_root']}")
    print(f"[split] 协议 train={st['protocol']['train_ratio']} val={st['protocol']['val_ratio']} "
          f"test={st['protocol']['test_ratio']} seed={st['protocol']['seed']} "
          f"分层=category×defect_type")
    print(f"[class] mode={st['classes']['mode']} n={st['classes']['count']} "
          f"names={st['classes']['names'][:6]}{' …' if st['classes']['count'] > 6 else ''}")
    print(f"{'split':<7}{'images':>8}{'defect':>8}{'bg':>7}{'instances':>11}")
    for sp in ("train", "val", "test"):
        d = st["splits"][sp]
        print(f"{sp:<7}{d['images']:>8}{d['defect_images']:>8}{d['background_images']:>7}"
              f"{d['instances']:>11}")
    a = st["defect_area_ratio"]
    print(f"[area ] 缺陷面积占比 mean={a['mean']:.4%} p50={a['p50']:.4%} "
          f"p90={a['p90']:.4%} max={a['max']:.4%} (n={a['count']})")
    d = st["protocol"]["dropped_tiny_defect_total"]
    if d:
        print(f"[drop ] {d} 张缺陷图掩码面积 < min_area 被丢弃（避免退化为背景负样本）")
    p = st["polygon"]
    print(f"[poly ] eps={p['epsilon_ratio']} min_area={p['min_area']} "
          f"平均顶点={p['mean_vertices']} 面积保真度={p['area_fidelity_mean']:.4f}")
    if not args.dry_run:
        print(f"[write] {rep.n_written} 张 -> {rep.out_root}  (落盘方式: {rep.link_mode_used})")
        print(f"[write] data.yaml -> {rep.data_yaml}")
        print(f"[write] stats     -> {rep.stats_path}")
        print(f"[write] manifest  -> {rep.manifest_path}")
    else:
        print("[dry  ] 未写盘")
    if rep.warnings:
        print(f"[warn ] {len(rep.warnings)} 条，前 10 条：")
        for w in rep.warnings[:10]:
            print("        ", w)
    print(f"[done ] {time.time() - t0:.1f}s")

    if not args.dry_run:
        print("\n下一步训练（8GB 显存建议 imgsz=640 batch=8）：")
        print(f"  yolo segment train data={rep.data_yaml} model=yolov8s-seg.pt "
              f"imgsz=640 batch=8 epochs=100 patience=20 name=baseline")
        print("  # 不传 project，产物落在 runs/segment/baseline；")
        print("  # 传 project=runs 会被拼成 runs/segment/runs/baseline，多一层没用的嵌套。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())