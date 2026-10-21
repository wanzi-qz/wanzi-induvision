"""MVTec AD 数据集扫描与 YOLO-seg 转换核心逻辑。

只依赖 numpy / opencv / pyyaml，不引入 torch，保证转换阶段与 CI 轻量。

关于划分协议（重要）
--------------------
MVTec AD 官方 ``train/`` 目录只有 ``good`` 正常样本，**没有任何缺陷标注**，
它是为无监督异常检测设计的。要做有监督的 YOLO-seg 缺陷分割，标注只能来自
``test/`` 目录里的缺陷图。因此本模块默认在官方 test 集内部做分层划分：

    train 70% / val 15% / test 15%（按 category × defect_type 分层，固定随机种子）

官方 ``train/good`` 与 ``test/good`` 作为无标注背景负样本参与训练/评估，用于
抑制误检。任何对外报告的指标都必须写明"在 test 集内部划分"，避免与官方无监督
协议混淆。
"""
from __future__ import annotations

import csv
import json
import os
import random
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml

__all__ = [
    "CATEGORIES",
    "MVTecLayoutError",
    "Sample",
    "ScanResult",
    "ConversionReport",
    "find_dataset_root",
    "scan_dataset",
    "build_splits",
    "mask_to_polygons",
    "polygon_to_label_line",
    "convert_dataset",
]

#: MVTec AD 官方 15 个产品类别
CATEGORIES: Tuple[str, ...] = (
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
)

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

#: 常见的解压包装目录名
_WRAPPER_DIRS = ("mvtec_anomaly_detection", "MVTec_AD", "mvtec-ad", "mvtec_ad")


class MVTecLayoutError(RuntimeError):
    """目录结构不符合 MVTec AD 官方布局时抛出。"""


@dataclass(frozen=True)
class Sample:
    """一张 MVTec 原始图片。"""

    category: str
    defect_type: str  # "good" 表示无缺陷
    image: Path
    mask: Optional[Path]  # good 样本为 None
    official_split: str  # "train" | "test"

    @property
    def is_defect(self) -> bool:
        return self.defect_type != "good"

    @property
    def key(self) -> Tuple[str, str, str, str]:
        return (self.category, self.defect_type, self.official_split, self.image.name)


@dataclass
class ScanResult:
    samples: List[Sample]
    root: Path
    categories: List[str]
    warnings: List[str] = field(default_factory=list)

    def by_category(self) -> Dict[str, List[Sample]]:
        out: Dict[str, List[Sample]] = {}
        for s in self.samples:
            out.setdefault(s.category, []).append(s)
        return out


@dataclass
class ConversionReport:
    out_root: Path
    data_yaml: Optional[Path]
    stats_path: Optional[Path]
    manifest_path: Optional[Path]
    stats: dict
    warnings: List[str]
    n_written: int
    link_mode_used: str


# --------------------------------------------------------------------------- #
# 目录扫描
# --------------------------------------------------------------------------- #
def _sorted_images(d: Path) -> List[Path]:
    if not d.is_dir():
        return []
    return sorted(
        (p for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda p: p.name,
    )


def find_dataset_root(path: Path) -> Path:
    """定位真正包含 15 个类别目录的那一层（兼容 zip 解压出的包装目录）。"""
    path = Path(path)
    if not path.exists():
        raise MVTecLayoutError(f"路径不存在: {path}")

    def _looks_like_root(p: Path) -> bool:
        return p.is_dir() and any((p / c).is_dir() for c in CATEGORIES)

    if _looks_like_root(path):
        return path
    for name in _WRAPPER_DIRS:
        cand = path / name
        if _looks_like_root(cand):
            return cand
    for sub in sorted(p for p in path.iterdir() if p.is_dir()):
        if _looks_like_root(sub):
            return sub
    raise MVTecLayoutError(
        f"在 {path} 下找不到 MVTec AD 类别目录（应包含 {CATEGORIES[:3]} ... 共 15 个子目录）"
    )


def _resolve_mask(gt_dir: Path, defect_type: str, stem: str) -> Optional[Path]:
    dt_dir = gt_dir / defect_type
    if not dt_dir.is_dir():
        return None
    for suffix in (".png", ".bmp", ".jpg", ".jpeg", ".tif", ".tiff"):
        cand = dt_dir / f"{stem}_mask{suffix}"
        if cand.is_file():
            return cand
    hits = sorted(p for p in dt_dir.iterdir() if p.is_file() and p.stem.startswith(stem))
    return hits[0] if hits else None


def scan_category(root: Path, category: str, *, strict: bool = True) -> Tuple[List[Sample], List[str]]:
    cdir = root / category
    if not cdir.is_dir():
        raise MVTecLayoutError(f"缺少类别目录: {cdir}")

    samples: List[Sample] = []
    warnings: List[str] = []
    missing: List[str] = []

    for p in _sorted_images(cdir / "train" / "good"):
        samples.append(Sample(category, "good", p, None, "train"))

    test_dir = cdir / "test"
    gt_dir = cdir / "ground_truth"
    if not test_dir.is_dir():
        raise MVTecLayoutError(f"缺少 test 目录: {test_dir}")

    defect_types = sorted(d.name for d in test_dir.iterdir() if d.is_dir())
    if not defect_types:
        raise MVTecLayoutError(f"{test_dir} 下没有任何缺陷类型子目录")
    if "good" not in defect_types:
        warnings.append(f"{category}/test 下没有 good 子目录（实际: {defect_types}）")

    for dt in defect_types:
        for p in _sorted_images(test_dir / dt):
            if dt == "good":
                samples.append(Sample(category, "good", p, None, "test"))
                continue
            mask = _resolve_mask(gt_dir, dt, p.stem)
            if mask is None:
                missing.append(f"{category}/ground_truth/{dt}/{p.stem}_mask.png")
                continue
            samples.append(Sample(category, dt, p, mask, "test"))

    if missing:
        msg = f"{category}: {len(missing)} 张缺陷图缺少掩码，例如 {missing[:3]}"
        if strict:
            raise MVTecLayoutError(msg)
        warnings.append(msg)
    return samples, warnings


def scan_dataset(
    root: Path,
    categories: Optional[Sequence[str]] = None,
    *,
    strict: bool = True,
) -> ScanResult:
    """扫描 MVTec AD 目录树，缺陷类型由目录动态发现（不硬编码）。"""
    real_root = find_dataset_root(Path(root))
    present = sorted(d.name for d in real_root.iterdir() if d.is_dir())
    wanted = list(categories) if categories else [c for c in CATEGORIES if c in present]
    unknown = [c for c in wanted if c not in CATEGORIES]
    if unknown:
        raise MVTecLayoutError(f"未知类别: {unknown}，官方类别为 {list(CATEGORIES)}")
    absent = [c for c in wanted if not (real_root / c).is_dir()]
    if absent:
        raise MVTecLayoutError(f"请求的类别在 {real_root} 下不存在: {absent}")

    samples: List[Sample] = []
    warnings: List[str] = []
    for cat in wanted:
        s, w = scan_category(real_root, cat, strict=strict)
        samples.extend(s)
        warnings.extend(w)
    if not samples:
        raise MVTecLayoutError(f"{real_root} 下没有扫描到任何图片")
    samples.sort(key=lambda s: s.key)
    return ScanResult(samples=samples, root=real_root, categories=wanted, warnings=warnings)


# --------------------------------------------------------------------------- #
# 划分
# --------------------------------------------------------------------------- #
def build_splits(
    samples: Sequence[Sample],
    *,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 0,
    include_good_train: bool = True,
) -> Dict[str, List[Sample]]:
    """按 (category, defect_type) 分层做确定性划分。

    - 缺陷图：train_ratio / val_ratio / 余下为 test
    - 官方 train/good：全部进 train 作背景负样本（``include_good_train=False`` 则丢弃）
    - 官方 test/good：按 val:test 比例分给 val 与 test，作背景负样本
    """
    if not 0.0 < train_ratio < 1.0:
        raise ValueError(f"train_ratio 必须在 (0,1) 内，收到 {train_ratio}")
    if val_ratio < 0 or train_ratio + val_ratio > 1.0:
        raise ValueError(f"train_ratio+val_ratio 不能超过 1，收到 {train_ratio}+{val_ratio}")
    test_ratio = max(0.0, 1.0 - train_ratio - val_ratio)

    splits: Dict[str, List[Sample]] = {"train": [], "val": [], "test": []}
    rng = random.Random(seed)

    defect_groups: Dict[Tuple[str, str], List[Sample]] = {}
    good_train: Dict[str, List[Sample]] = {}
    good_test: Dict[str, List[Sample]] = {}
    for s in samples:
        if s.is_defect:
            defect_groups.setdefault((s.category, s.defect_type), []).append(s)
        elif s.official_split == "train":
            good_train.setdefault(s.category, []).append(s)
        else:
            good_test.setdefault(s.category, []).append(s)

    for key in sorted(defect_groups):
        group = sorted(defect_groups[key], key=lambda s: s.image.name)
        n = len(group)
        n_val = int(round(n * val_ratio))
        n_test = int(round(n * test_ratio))
        if n_val == 0 and val_ratio > 0 and n >= 2:
            n_val = 1  # 保证每个缺陷类型在验证集里至少出现一次
        n_val = min(n_val, max(0, n - 1))
        n_test = min(n_test, n - n_val)
        n_train = n - n_val - n_test
        order = list(range(n))
        rng.shuffle(order)
        for i, idx in enumerate(order):
            if i < n_train:
                splits["train"].append(group[idx])
            elif i < n_train + n_val:
                splits["val"].append(group[idx])
            else:
                splits["test"].append(group[idx])

    if include_good_train:
        for cat in sorted(good_train):
            splits["train"].extend(sorted(good_train[cat], key=lambda s: s.image.name))

    denom = val_ratio + test_ratio
    for cat in sorted(good_test):
        group = sorted(good_test[cat], key=lambda s: s.image.name)
        n_val = int(round(len(group) * (val_ratio / denom))) if denom > 0 else 0
        n_val = min(n_val, len(group))
        order = list(range(len(group)))
        rng.shuffle(order)
        for i, idx in enumerate(order):
            splits["val" if i < n_val else "test"].append(group[idx])

    for k in splits:
        splits[k].sort(key=lambda s: s.key)
    return splits


# --------------------------------------------------------------------------- #
# 掩码 -> 多边形
# --------------------------------------------------------------------------- #
@dataclass
class PolygonSet:
    polygons: List[np.ndarray]  # 每个元素为 (N,2) int32 像素坐标
    mask_pixels: int
    polygon_pixels: float
    width: int
    height: int


def mask_to_polygons(
    mask: np.ndarray,
    *,
    epsilon_ratio: float = 0.002,
    min_area: float = 4.0,
) -> PolygonSet:
    """二值掩码 -> YOLO-seg 多边形列表。

    用 ``RETR_EXTERNAL`` 只取外轮廓：YOLO-seg 不支持带孔多边形，donut 形缺陷的
    内孔会被填掉，面积膨胀量在 ``polygon_pixels / mask_pixels`` 中量化输出。
    每个连通域输出为一个独立实例（一行标签）。
    """
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    h, w = mask.shape[:2]
    binary = np.where(mask > 0, 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polys: List[np.ndarray] = []
    poly_px = 0.0
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon_ratio * peri, True).reshape(-1, 2)
        if len(approx) < 3:
            continue
        polys.append(np.ascontiguousarray(approx, dtype=np.int32))
        poly_px += float(cv2.contourArea(approx.reshape(-1, 1, 2)))
    return PolygonSet(
        polygons=polys,
        mask_pixels=int(np.count_nonzero(binary)),
        polygon_pixels=poly_px,
        width=int(w),
        height=int(h),
    )


def _polygon_to_yolo_line(cls_id: int, poly: np.ndarray, w: int, h: int) -> str:
    xs = np.clip(poly[:, 0].astype(np.float64) / max(w - 1, 1), 0.0, 1.0)
    ys = np.clip(poly[:, 1].astype(np.float64) / max(h - 1, 1), 0.0, 1.0)
    coords = " ".join(f"{v:.6f}" for xy in zip(xs, ys) for v in xy)
    return f"{cls_id} {coords}"


def polygon_to_label_line(cls_id: int, poly: np.ndarray, w: int, h: int) -> str:
    """公开版归一化：像素多边形 -> 一行 YOLO-seg 标签。

    单独暴露出来是为了让下游（缺陷合成、标签重写）复用**同一套**归一化口径，
    不要各自写一份 `/w` 或 `/(w-1)`，否则同一条边界会差出半个像素。
    """
    return _polygon_to_yolo_line(cls_id, poly, w, h)


# --------------------------------------------------------------------------- #
# 落盘
# --------------------------------------------------------------------------- #
def _place_file(src: Path, dst: Path, mode: str) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    if mode == "copy":
        shutil.copy2(src, dst)
        return "copy"
    if mode not in ("hardlink", "symlink"):
        raise ValueError(f"未知 link 模式: {mode}")
    try:
        if mode == "hardlink":
            os.link(str(src), str(dst))
        else:
            os.symlink(str(src), str(dst), target_is_directory=False)
        return mode
    except OSError:
        shutil.copy2(src, dst)
        return "copy(fallback)"


def _stem(s: Sample) -> str:
    return f"{s.category}__{s.official_split}__{s.defect_type}__{s.image.stem}"


def _process_one(
    s: Sample,
    split: str,
    cls_of: Dict[str, int],
    epsilon_ratio: float,
    min_area: float,
) -> dict:
    img = cv2.imread(str(s.image), cv2.IMREAD_COLOR)
    if img is None:
        return {"sample": s, "split": split, "error": f"图片解码失败: {s.image}"}
    ih, iw = img.shape[:2]
    rec: dict = {
        "sample": s,
        "split": split,
        "width": iw,
        "height": ih,
        "polygons": [],
        "mask_pixels": 0,
        "polygon_pixels": 0.0,
        "error": None,
        "warn": None,
    }
    if s.mask is None:
        return rec
    m = cv2.imread(str(s.mask), cv2.IMREAD_UNCHANGED)
    if m is None:
        rec["error"] = f"掩码解码失败: {s.mask}"
        return rec
    if m.ndim == 3:
        m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
    if m.shape[:2] != (ih, iw):
        rec["warn"] = f"掩码尺寸 {m.shape[1]}x{m.shape[0]} 与图片 {iw}x{ih} 不一致，已用最近邻缩放"
        m = cv2.resize(m, (iw, ih), interpolation=cv2.INTER_NEAREST)
    ps = mask_to_polygons(m, epsilon_ratio=epsilon_ratio, min_area=min_area)
    rec.update(
        polygons=ps.polygons,
        mask_pixels=ps.mask_pixels,
        polygon_pixels=ps.polygon_pixels,
    )
    if s.is_defect and not ps.polygons:
        rec["warn"] = f"缺陷掩码未产出多边形（min_area={min_area}）: {s.mask}"
    return rec


def _pct(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    return round(float(np.percentile(np.asarray(values, dtype=np.float64), q)), 6)


def convert_dataset(
    src_root: Path,
    out_root: Path,
    *,
    classes_mode: str = "per-category",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 0,
    epsilon_ratio: float = 0.002,
    min_area: float = 4.0,
    link: str = "hardlink",
    include_good_train: bool = True,
    categories: Optional[Sequence[str]] = None,
    strict: bool = True,
    workers: int = 1,
    write_files: bool = True,
    keep_empty_defects: bool = False,
) -> ConversionReport:
    """把 MVTec AD 转成 YOLOv8-seg 数据集，输出 data.yaml + 统计 + 清单。"""
    if classes_mode not in ("per-category", "unified"):
        raise ValueError(f"classes_mode 只能是 per-category / unified，收到 {classes_mode}")

    scan = scan_dataset(src_root, categories, strict=strict)
    splits = build_splits(
        scan.samples,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
        include_good_train=include_good_train,
    )

    if classes_mode == "per-category":
        names = list(scan.categories)
        cls_of = {c: i for i, c in enumerate(names)}
    else:
        names = ["defect"]
        cls_of = {c: 0 for c in scan.categories}

    warnings: List[str] = list(scan.warnings)
    tasks: List[Tuple[Sample, str]] = []
    for split in ("train", "val", "test"):
        for s in splits[split]:
            tasks.append((s, split))

    if workers and workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            records = list(
                ex.map(lambda t: _process_one(t[0], t[1], cls_of, epsilon_ratio, min_area), tasks)
            )
    else:
        records = [_process_one(s, sp, cls_of, epsilon_ratio, min_area) for s, sp in tasks]

    out_root = Path(out_root)
    manifest_rows: List[dict] = []
    link_used: Dict[str, int] = {}
    area_ratios: List[float] = []
    vertices: List[int] = []
    poly_fidelity: List[float] = []
    split_stats: Dict[str, dict] = {
        sp: {"images": 0, "background_images": 0, "defect_images": 0, "instances": 0}
        for sp in ("train", "val", "test")
    }
    dropped_tiny: Dict[str, int] = {"train": 0, "val": 0, "test": 0}
    cat_stats: Dict[str, dict] = {}
    n_written = 0

    for rec in records:
        s: Sample = rec["sample"]
        split: str = rec["split"]
        if rec["error"]:
            warnings.append(rec["error"])
            continue
        if rec["warn"]:
            warnings.append(rec["warn"])

        if s.is_defect and not rec["polygons"] and not keep_empty_defects:
            # 掩码面积低于 min_area：保留下来会变成"缺陷被当成背景"的标签噪声，默认丢弃
            dropped_tiny[split] += 1
            warnings.append(
                f"丢弃（缺陷面积 < min_area={min_area}）: {s.category}/{s.defect_type}/"
                f"{s.image.name} [{split}]"
            )
            continue

        st = split_stats[split]
        st["images"] += 1
        if rec["polygons"]:
            st["defect_images"] += 1
            st["instances"] += len(rec["polygons"])
        else:
            st["background_images"] += 1

        c = cat_stats.setdefault(
            s.category,
            {"images": 0, "defect_images": 0, "instances": 0, "defect_types": {}, "splits": {"train": 0, "val": 0, "test": 0}},
        )
        c["images"] += 1
        c["splits"][split] += 1
        dt = c["defect_types"].setdefault(
            s.defect_type, {"train": 0, "val": 0, "test": 0, "instances": 0, "mask_pixels": 0}
        )
        dt[split] += 1
        if rec["polygons"]:
            c["defect_images"] += 1
            c["instances"] += len(rec["polygons"])
            dt["instances"] += len(rec["polygons"])
            dt["mask_pixels"] += rec["mask_pixels"]
            px = rec["width"] * rec["height"]
            area_ratios.append(rec["mask_pixels"] / px if px else 0.0)
            if rec["mask_pixels"]:
                poly_fidelity.append(rec["polygon_pixels"] / rec["mask_pixels"])
            vertices.extend(len(p) for p in rec["polygons"])

        if not write_files:
            continue

        stem = _stem(s)
        img_dst = out_root / "images" / split / f"{stem}{s.image.suffix.lower()}"
        lbl_dst = out_root / "labels" / split / f"{stem}.txt"
        used = _place_file(s.image, img_dst, link)
        link_used[used] = link_used.get(used, 0) + 1

        lines = [
            _polygon_to_yolo_line(cls_of[s.category], p, rec["width"], rec["height"])
            for p in rec["polygons"]
        ]
        lbl_dst.parent.mkdir(parents=True, exist_ok=True)
        lbl_dst.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        n_written += 1
        manifest_rows.append(
            {
                "out_name": f"{stem}{s.image.suffix.lower()}",
                "split": split,
                "category": s.category,
                "defect_type": s.defect_type,
                "official_split": s.official_split,
                "class_id": cls_of[s.category],
                "instances": len(rec["polygons"]),
                "width": rec["width"],
                "height": rec["height"],
                "mask_pixels": rec["mask_pixels"],
                "src_image": str(s.image),
                "src_mask": str(s.mask) if s.mask else "",
            }
        )

    stats = {
        "source_root": str(scan.root),
        "protocol": {
            "note": "MVTec AD 官方 train 集无缺陷标注；有监督 YOLO-seg 在官方 test 集内部做分层划分，"
            "good 图作为无标注背景负样本。",
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": round(max(0.0, 1.0 - train_ratio - val_ratio), 6),
            "seed": seed,
            "stratify_by": ["category", "defect_type"],
            "include_good_train_as_background": include_good_train,
            "keep_empty_defects": keep_empty_defects,
            "dropped_tiny_defect_images": dropped_tiny,
            "dropped_tiny_defect_total": sum(dropped_tiny.values()),
        },
        "classes": {"mode": classes_mode, "count": len(names), "names": names},
        "polygon": {
            "epsilon_ratio": epsilon_ratio,
            "min_area": min_area,
            "contour_mode": "RETR_EXTERNAL",
            "mean_vertices": round(float(np.mean(vertices)), 3) if vertices else 0.0,
            "area_fidelity_mean": round(float(np.mean(poly_fidelity)), 4) if poly_fidelity else 0.0,
            "area_fidelity_p10": _pct(poly_fidelity, 10),
        },
        "splits": split_stats,
        "categories": {k: cat_stats[k] for k in sorted(cat_stats)},
        "defect_area_ratio": {
            "count": len(area_ratios),
            "mean": round(float(np.mean(area_ratios)), 6) if area_ratios else 0.0,
            "min": round(min(area_ratios), 6) if area_ratios else 0.0,
            "p50": _pct(area_ratios, 50),
            "p90": _pct(area_ratios, 90),
            "max": round(max(area_ratios), 6) if area_ratios else 0.0,
        },
        "totals": {
            "images_scanned": len(scan.samples),
            "images_written": n_written,
            "instances": sum(v["instances"] for v in split_stats.values()),
        },
    }

    data_yaml = stats_path = manifest_path = None
    if write_files:
        out_root.mkdir(parents=True, exist_ok=True)
        data_yaml = out_root / "data.yaml"
        data_yaml.write_text(
            yaml.safe_dump(
                {
                    "path": str(out_root.resolve()),
                    "train": "images/train",
                    "val": "images/val",
                    "test": "images/test",
                    "names": {i: n for i, n in enumerate(names)},
                },
                sort_keys=False,
                allow_unicode=True,
            ),
            encoding="utf-8",
        )
        stats_path = out_root / "dataset_stats.json"
        stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
        manifest_path = out_root / "manifest.csv"
        with manifest_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()) if manifest_rows else [])
            w.writeheader()
            w.writerows(manifest_rows)

    return ConversionReport(
        out_root=out_root,
        data_yaml=data_yaml,
        stats_path=stats_path,
        manifest_path=manifest_path,
        stats=stats,
        warnings=warnings,
        n_written=n_written,
        link_mode_used=",".join(sorted(k for k in link_used if k)) or link,
    )