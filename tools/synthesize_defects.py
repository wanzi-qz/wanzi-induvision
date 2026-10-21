#!/usr/bin/env python
"""Copy-Paste 缺陷合成 CLI：给稀有类补实例，产物是可直接训练的 YOLO-seg 数据集。

用法示例
--------
# 按「补齐到 train 最大类实例数 ×2」合成，输出增强版数据集
python tools/synthesize_defects.py --src out/yolo_bottle_tb --out out/yolo_bottle_tb_cp --cap-factor 2

# 指定每类至少 150 个实例，alpha 融合，出 8 张抽查图
python tools/synthesize_defects.py --src out/yolo_bottle_tb --out out/yolo_bottle_tb_cp \
    --target-per-class 150 --blend alpha --figures 8

# 只看配额不写盘
python tools/synthesize_defects.py --src out/yolo_bottle_tb --dry-run

产物
----
``images/{train,val,test}`` ``labels/{train,val,test}``：原图硬链接 + 合成图实文件
``data.yaml``              与源数据集同口径的类别表
``synth_manifest.csv``     每张合成图的完整审计（底板、贴了谁、变换、融合模式、可见比例）
``synth_report.json``      配额计划、前后实例分布、多边形保真度、拒绝采样计数、耗时
``figures/``               抽查对比图（--figures N）

防泄漏口径：缺陷贴片只从 train 划分抠（defect_bank 强制），val/test 原样硬链接，
合成图只进 train。因此 val/test 的指标不会因为增强而虚高。
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from induvision.balance import instance_histogram, plan_synthesis  # noqa: E402
from induvision.copy_paste import (  # noqa: E402
    PasteParams,
    masks_to_label_items,
    paste_instances,
    rasterize_instances,
    render_comparison,
)
from induvision.defect_bank import (  # noqa: E402
    build_bank,
    centroid_priors,
    instances_by_class,
    load_bank,
    save_bank,
)
from induvision.mvtec import _place_file  # noqa: E402
from induvision.yolo_dataset import YoloImage, read_dataset, write_label_file  # noqa: E402


# manifest 列顺序固定写死：合成结果为空时也能给出表头，不会 IndexError
MANIFEST_FIELDS = [
    "out_name", "split", "class_id", "base_image", "base_category", "n_pasted",
    "instance_ids", "blend_used", "visible_ratio_mean", "poly_area_fidelity",
    "polygons", "seed_child",
]


def _base_pool(ds, class_name: str) -> List[YoloImage]:
    """贴图底板：优先同类别的无标注图，退而求其次用全部无标注图。"""
    bgs = [im for im in ds.split("train") if im.is_background]
    same = [im for im in bgs if im.category == class_name]
    return same or bgs


def _synth_one(
    task: Tuple[int, int],
    ds,
    bank,
    pools: Dict[int, list],
    priors: Dict[int, np.ndarray],
    params: PasteParams,
    seed: int,
    remaining: Dict[int, int],
) -> Optional[dict]:
    """生成一张合成图，返回写盘所需的完整记录（不写盘，便于线程池并行）。"""
    class_id, seq = task
    pool = pools.get(class_id)
    if not pool or remaining.get(class_id, 0) <= 0:
        return None
    rng = np.random.default_rng(np.random.SeedSequence([seed, class_id, seq]))
    base = pool[int(rng.integers(len(pool)))]
    img = cv2.imread(str(base.path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    masks = rasterize_instances(base.instances, base.width, base.height)
    before = img.copy()
    n = min(params.max_per_image, remaining[class_id])
    res = paste_instances(img, masks, pool_instances(bank, class_id), rng, params,
                          priors=priors.get(class_id), n=n)
    if not res.records:
        return None
    items, poly_stats = masks_to_label_items(
        masks, epsilon_ratio=params.epsilon_ratio, min_area=params.min_area
    )
    return {
        "class_id": class_id,
        "seq": seq,
        "base_image": base.path.name,
        "base_category": base.category,
        "width": base.width,
        "height": base.height,
        "image": img,
        "before": before,
        "items": items,
        "poly_stats": poly_stats,
        "records": res.records,
        "rejected": res.rejected,
        "masks": masks,
    }


def pool_instances(bank, class_id: int):
    return instances_by_class(bank).get(class_id, [])


def _purge_stale_synth(out: Path) -> int:
    """清掉上一次运行留下的合成产物，返回删除个数。

    合成图的名字按 (类别, 序号) 生成，配额或参数一变，这一轮可能写得更少，
    上一轮多出来的文件就成了「不在 manifest 里、却还躺在 images/train 下」的孤儿。
    它的旧标签文件也还在，ultralytics 照样拿去训练 —— 数据集就悄悄不可复现了。
    """
    n = 0
    for root in (out / "images", out / "labels", out / "figures"):
        if not root.is_dir():
            continue
        for p in sorted(root.rglob("synth__*")):
            if p.is_file():
                p.unlink()
                n += 1
    return n


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Copy-Paste 缺陷合成 -> YOLO-seg 数据集",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--src", default=str(REPO_ROOT / "out" / "yolo_bottle_tb"))
    ap.add_argument("--out", default=str(REPO_ROOT / "out" / "yolo_bottle_tb_cp"))
    ap.add_argument("--target-per-class", type=int, default=None, help="每类实例数下限；不传则用 --strategy")
    ap.add_argument("--strategy", choices=("max", "none"), default="max")
    ap.add_argument("--cap-factor", type=float, default=2.0, help="strategy=max 时目标 = 最大类实例数 × 该值")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--blend", choices=("poisson", "alpha", "naive"), default="poisson")
    ap.add_argument("--scale", type=float, nargs=2, default=[0.6, 1.4], metavar=("LO", "HI"))
    ap.add_argument("--rotate", type=float, nargs=2, default=[-180.0, 180.0], metavar=("LO", "HI"))
    ap.add_argument("--flip-p", type=float, default=0.5)
    ap.add_argument("--max-per-image", type=int, default=3)
    ap.add_argument("--position", choices=("centroid_jitter", "uniform"), default="centroid_jitter")
    ap.add_argument("--jitter-sigma", type=float, default=0.08)
    ap.add_argument("--bg-std-min", type=float, default=0.03, help="落点邻域灰度标准差下限，挡住贴到无纹理背景上的情况")
    ap.add_argument("--min-visible", type=float, default=0.75)
    ap.add_argument("--max-overlap", type=float, default=0.0, help="缺陷间允许的重叠占比，默认 0 保证一个连通域=一个实例")
    ap.add_argument("--feather", type=float, default=1.5)
    ap.add_argument("--min-paste-area", type=float, default=64.0, help="warp 后硬掩码小于该像素面积则放弃这次贴图")
    ap.add_argument("--bank-cache", default=None, help="缺陷库 npz 缓存路径；不传则现建并存到 <out>/defect_bank.npz")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--figures", type=int, default=4, help="抽查对比图张数，0 表示不出")
    ap.add_argument("--link", choices=("hardlink", "copy", "symlink"), default="hardlink")
    ap.add_argument("--dry-run", action="store_true", help="只出配额计划与诊断，不写盘")
    args = ap.parse_args(argv)

    t0 = time.time()
    src = Path(args.src)
    out = Path(args.out)
    ds = read_dataset(src)
    if ds.warnings:
        print(f"[warn ] 读取源数据集 {len(ds.warnings)} 条告警，前 5 条：")
        for w in ds.warnings[:5]:
            print("        ", w)

    hist = instance_histogram(ds, "train")
    plan = plan_synthesis(
        hist,
        target=args.target_per_class,
        strategy=args.strategy,
        cap_factor=args.cap_factor,
        max_per_image=args.max_per_image,
    )
    print(f"[plan ] {plan.note or f'target={plan.target}'}")
    print(f"{'class':<14}{'have':>6}{'target':>8}{'add_inst':>10}{'add_img':>9}")
    for cid in sorted(plan.per_class):
        d = plan.per_class[cid]
        print(f"{d['name']:<14}{d['have_instances']:>6}{d['target_instances']:>8}"
              f"{d['add_instances']:>10}{d['add_images']:>9}")
    print(f"[plan ] 合计需补 {plan.total_instances} 个实例 / {plan.total_images} 张合成图")
    if args.dry_run or plan.total_images == 0:
        print("[dry  ] 未写盘" if args.dry_run else "[skip ] 无需合成")
        return 0

    params = PasteParams(
        scale=tuple(args.scale),
        rotate=tuple(args.rotate),
        flip_p=args.flip_p,
        blend=args.blend,
        max_per_image=args.max_per_image,
        position=args.position,
        jitter_sigma=args.jitter_sigma,
        bg_std_min=args.bg_std_min,
        min_visible_ratio=args.min_visible,
        max_overlap_frac=args.max_overlap,
        feather=args.feather,
        min_paste_area=args.min_paste_area,
    )

    cache = Path(args.bank_cache) if args.bank_cache else out / "defect_bank.npz"
    if cache.is_file() and not args.bank_cache:
        bank = load_bank(cache)
        print(f"[bank ] 复用缓存 {cache}（{len(bank)} 实例）")
    else:
        bank = build_bank(read_dataset(src, splits=("train",)))
        if not args.dry_run:
            save_bank(bank, cache)
        print(f"[bank ] 现建 {len(bank)} 个实例（仅 train 划分取材），缓存 -> {cache}")
    if not bank.instances:
        print("[error] 缺陷库为空，无法合成")
        return 1

    pools = {cid: pool_instances(bank, cid) for cid in plan.per_class if plan.per_class[cid]["add_images"]}
    priors = {cid: centroid_priors(bank, cid) for cid in pools}
    base_pools = {cid: _base_pool(ds, plan.per_class[cid]["name"]) for cid in pools}
    missing = [cid for cid in pools if not base_pools[cid]]
    if missing:
        print(f"[error] 类别 {[plan.per_class[c]['name'] for c in missing]} 没有可用的贴图底板（train 无 good 图）")
        return 1

    remaining = {cid: plan.per_class[cid]["add_instances"] for cid in pools}
    tasks: List[Tuple[int, int]] = []
    for cid in sorted(pools):
        for seq in range(plan.per_class[cid]["add_images"] * 2):  # 上限翻倍，实际按配额提前停
            tasks.append((cid, seq))

    results: List[dict] = []
    t1 = time.time()
    # 按波推进：每波 workers 个任务，波与波之间检查配额。
    # 每张合成图用 SeedSequence([seed, class_id, seq]) 派生独立 rng，
    # 所以「生成哪些图」只取决于任务顺序，与并发度无关，可复现。
    wave = max(1, args.workers)

    def _consume(r: Optional[dict]) -> None:
        if r is None:
            return
        remaining[r["class_id"]] -= len(r["records"])
        results.append(r)

    i = 0
    with ThreadPoolExecutor(max_workers=wave) as ex:
        while i < len(tasks):
            chunk = tasks[i : i + wave]
            for r in ex.map(lambda t: _synth_one(t, ds, bank, base_pools, priors, params, args.seed, remaining), chunk):
                _consume(r)
            i += len(chunk)
            if all(remaining.get(cid, 0) <= 0 for cid in pools):
                break
    print(f"[synth] {len(results)} 张合成图 / {time.time() - t1:.1f}s（workers={wave}）")
    if not results:
        print("[error] 一张合成图都没产出：所有落点都被拒绝了。")
        print("        可调 --bg-std-min（底板太平）/ --min-visible（画布太小贴不下）"
              "/ --min-paste-area（缺陷太小）后重试")
        return 1

    # ---- 落盘 ----
    out.mkdir(parents=True, exist_ok=True)
    stale = _purge_stale_synth(out)
    if stale:
        print(f"[clean] 清掉上一次运行遗留的 {stale} 个合成产物（孤儿图会混进训练集）")
    link_used: Dict[str, int] = {}
    n_linked = 0
    for im in ds.images:
        dst_img = out / "images" / im.split / im.path.name
        dst_lbl = out / "labels" / im.split / f"{im.path.stem}.txt"
        mode = _place_file(im.path, dst_img, args.link)
        link_used[mode] = link_used.get(mode, 0) + 1
        n_linked += 1
        if im.label_path.is_file():
            _place_file(im.label_path, dst_lbl, args.link)
    print(f"[link ] 原图 {n_linked} 张 -> {out}（{args.link}，{link_used}）")

    manifest_rows: List[dict] = []
    rejected_total: Dict[str, int] = {}
    fid: List[float] = []
    per_class_after: Dict[int, int] = {st.class_id: st.instances for st in hist}
    for i, r in enumerate(results):
        name = f"synth__{plan.per_class[r['class_id']]['name']}__{r['seq']:05d}.png"
        cv2.imwrite(str(out / "images" / "train" / name), r["image"])
        write_label_file(out / "labels" / "train" / f"{Path(name).stem}.txt", r["items"], r["width"], r["height"])
        per_class_after[r["class_id"]] += sum(1 for _ in r["items"])
        if r["poly_stats"]["area_fidelity"]:
            fid.append(r["poly_stats"]["area_fidelity"])
        for k, v in r["rejected"].items():
            rejected_total[k] = rejected_total.get(k, 0) + v
        manifest_rows.append(
            {
                "out_name": name,
                "split": "train",
                "class_id": r["class_id"],
                "base_image": r["base_image"],
                "base_category": r["base_category"],
                "n_pasted": len(r["records"]),
                "instance_ids": ";".join(rec.instance_id for rec in r["records"]),
                "blend_used": ";".join(rec.blend_used for rec in r["records"]),
                "visible_ratio_mean": round(float(np.mean([rec.visible_ratio for rec in r["records"]])), 4),
                "poly_area_fidelity": r["poly_stats"]["area_fidelity"],
                "polygons": r["poly_stats"]["polygons"],
                "seed_child": f"[{args.seed}, {r['class_id']}, {r['seq']}]",
            }
        )
        if i < args.figures:
            fig = render_comparison(r["before"], r["image"], r["masks"], r["records"])
            (out / "figures").mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out / "figures" / f"{Path(name).stem}.png"), fig)

    with (out / "synth_manifest.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        w.writerows(manifest_rows)

    names = {i: ds.class_name(i) for i in range(len(ds.names))}
    (out / "data.yaml").write_text(
        "path: %s\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n%s"
        % (out.resolve().as_posix(), "".join(f"  {k}: {v}\n" for k, v in sorted(names.items()))),
        encoding="utf-8",
    )

    report = {
        "source": str(src),
        "out": str(out),
        "seed": args.seed,
        "params": params.to_dict(),
        "plan": plan.to_dict(),
        "bank": {"instances": len(bank), "source_images": len({d.source_image for d in bank.instances}),
                 "built_from": list(bank.built_from)},
        "synth_images": len(results),
        "instances_added": {names[c]: v for c, v in per_class_after.items()},
        "instances_before": {st.name: st.instances for st in hist},
        "instances_after": {names[c]: v for c, v in per_class_after.items()},
        "remaining_quota": {names[c]: v for c, v in remaining.items()},
        "polygon_area_fidelity": {
            "mean": round(float(np.mean(fid)), 6) if fid else None,
            "p10": round(float(np.percentile(fid, 10)), 6) if fid else None,
        },
        "rejected_sampling": rejected_total,
        "link_mode": args.link,
        "seconds": round(time.time() - t0, 2),
    }
    (out / "synth_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"{'class':<14}{'before':>8}{'after':>8}")
    for st in hist:
        print(f"{st.name:<14}{st.instances:>8}{per_class_after[st.class_id]:>8}")
    if fid:
        print(f"[poly ] 合成标签面积保真度 mean={np.mean(fid):.4f} p10={np.percentile(fid, 10):.4f}")
    if rejected_total:
        print(f"[rej  ] 拒绝采样计数 {rejected_total}")
    print(f"[write] manifest -> {out / 'synth_manifest.csv'}")
    print(f"[write] report   -> {out / 'synth_report.json'}")
    if args.figures:
        print(f"[write] figures  -> {out / 'figures'}（前 {min(args.figures, len(results))} 张）")
    print(f"[done ] {time.time() - t0:.1f}s")
    print(f"\n下一步：yolo segment train data={out / 'data.yaml'} model=yolov8s-seg.pt imgsz=640 batch=8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())