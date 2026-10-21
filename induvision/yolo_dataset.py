"""读取 / 写回本仓库产出的 YOLO-seg 数据集。

``tools/convert_mvtec_to_yolo.py`` 的输出结构是::

    <root>/data.yaml                     names: {0: bottle, 1: toothbrush}
    <root>/images/{train,val,test}/*.png
    <root>/labels/{train,val,test}/*.txt 每行 ``cls x1 y1 x2 y2 ...``（归一化）
    <root>/dataset_stats.json            转换统计
    <root>/manifest.csv                  每张输出图到 MVTec 原始文件的映射

本模块只负责「读进来 / 写回去」，不含任何 MVTec 特有假设，所以缺陷合成、
类别均衡统计、训练脚本可以共用同一份解析与归一化口径。

归一化口径（重要）
------------------
写标签时用的是 ``x / (w - 1)``（见 ``mvtec.polygon_to_label_line``），读回来
就必须用 ``x * (w - 1)`` 反归一化，两边不一致会让贴边的多边形整体偏移半个像素，
在 900x900 的 MVTec 图上看不出，但会让「合成前后 IoU」这类指标凭空掉点。
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml

from .mvtec import polygon_to_label_line

__all__ = [
    "Instance",
    "YoloImage",
    "YoloDataset",
    "read_dataset",
    "read_manifest",
    "parse_label_file",
    "write_label_file",
    "SPLITS",
]

SPLITS: Tuple[str, ...] = ("train", "val", "test")


@dataclass(frozen=True)
class Instance:
    """一个缺陷实例：类别 id + 像素多边形。"""

    class_id: int
    polygon: np.ndarray  # (N,2) int32，像素坐标
    area_px: float  # cv2.contourArea，用于面积分布统计


@dataclass(frozen=True)
class YoloImage:
    """数据集里的一张图（含它的标签解析结果）。"""

    path: Path
    label_path: Path
    split: str
    width: int
    height: int
    instances: Tuple[Instance, ...]
    category: Optional[str] = None  # 来自 manifest / 文件名，非 MVTec 数据为 None
    defect_type: Optional[str] = None

    @property
    def stem(self) -> str:
        return self.path.stem

    @property
    def is_background(self) -> bool:
        """无标注图。MVTec 的 good 样本落在这里，是 copy-paste 的贴图底板。"""
        return not self.instances

    @property
    def class_ids(self) -> Tuple[int, ...]:
        return tuple(sorted({i.class_id for i in self.instances}))


@dataclass
class YoloDataset:
    root: Path
    names: List[str]
    images: List[YoloImage] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def split(self, name: str) -> List[YoloImage]:
        return [im for im in self.images if im.split == name]

    def by_class(self, class_id: int) -> List[YoloImage]:
        return [im for im in self.images if class_id in im.class_ids]

    def class_name(self, class_id: int) -> str:
        if 0 <= class_id < len(self.names):
            return self.names[class_id]
        return f"cls{class_id}"


def _load_names(root: Path) -> Tuple[List[str], List[str]]:
    """从 data.yaml 读类别名，返回 (names, warnings)。"""
    y = root / "data.yaml"
    if not y.is_file():
        return [], [f"缺少 {y}，类别名未知"]
    d = yaml.safe_load(y.read_text(encoding="utf-8")) or {}
    raw = d.get("names")
    warn: List[str] = []
    if isinstance(raw, dict):
        idx = sorted(int(k) for k in raw)
        if idx != list(range(len(idx))):
            warn.append(f"data.yaml names 的 id 不连续: {idx}")
        return [str(raw[k]) for k in idx], warn
    if isinstance(raw, (list, tuple)):
        return [str(x) for x in raw], warn
    return [], warn + ["data.yaml 里没有 names 字段"]


def read_manifest(root: Path) -> Dict[str, Dict[str, str]]:
    """读 manifest.csv，键是输出文件名。文件不存在时返回空 dict。"""
    p = root / "manifest.csv"
    if not p.is_file():
        return {}
    out: Dict[str, Dict[str, str]] = {}
    with p.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            name = row.get("out_name")
            if name:
                out[name] = row
    return out


def parse_label_file(path: Path) -> Tuple[List[Tuple[int, np.ndarray]], List[str]]:
    """解析一个 YOLO-seg 标签文件，返回 ([(cls_id, 归一化 (N,2) float64)], warnings)。"""
    items: List[Tuple[int, np.ndarray]] = []
    warns: List[str] = []
    if not path.is_file():
        return items, warns  # 背景图允许没有标签文件
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        try:
            cls_id = int(float(parts[0]))
            coords = [float(v) for v in parts[1:]]
        except ValueError:
            warns.append(f"{path.name}:{lineno} 无法解析，已跳过")
            continue
        if len(coords) < 6 or len(coords) % 2:
            warns.append(f"{path.name}:{lineno} 坐标数 {len(coords)} 不是 >=6 的偶数，已跳过")
            continue
        poly = np.asarray(coords, dtype=np.float64).reshape(-1, 2)
        if not np.isfinite(poly).all():
            warns.append(f"{path.name}:{lineno} 含非有限值，已跳过")
            continue
        items.append((cls_id, poly))
    return items, warns


def _denorm(poly_norm: np.ndarray, w: int, h: int) -> np.ndarray:
    out = poly_norm.copy()
    out[:, 0] *= max(w - 1, 1)
    out[:, 1] *= max(h - 1, 1)
    return np.ascontiguousarray(np.clip(np.rint(out), 0, [max(w - 1, 1), max(h - 1, 1)]).astype(np.int32))


def read_dataset(
    root: Path,
    splits: Optional[Sequence[str]] = None,
    *,
    require_labels: bool = False,
) -> YoloDataset:
    """扫描 ``root`` 下的 images/labels，把标签反归一化成像素多边形。

    ``splits`` 为 None 时读全部三个划分。``require_labels=True`` 时缺标签文件报错，
    默认宽松（背景图本来就可以没有标签）。
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"数据集目录不存在: {root}")
    names, warns = _load_names(root)
    manifest = read_manifest(root)
    want = tuple(splits) if splits else SPLITS
    for sp in want:
        if sp not in SPLITS:
            raise ValueError(f"未知划分 {sp}，只能是 {SPLITS}")

    images: List[YoloImage] = []
    for sp in want:
        img_dir = root / "images" / sp
        if not img_dir.is_dir():
            warns.append(f"缺少 {img_dir}")
            continue
        for p in sorted(img_dir.iterdir()):
            if not p.is_file() or p.suffix.lower() not in (".png", ".jpg", ".jpeg", ".bmp"):
                continue
            lp = root / "labels" / sp / f"{p.stem}.txt"
            if require_labels and not lp.is_file():
                raise FileNotFoundError(f"缺少标签文件: {lp}")
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is None:
                warns.append(f"图片解码失败，已跳过: {p}")
                continue
            h, w = img.shape[:2]
            parsed, pw = parse_label_file(lp)
            warns.extend(pw)
            insts: List[Instance] = []
            for cls_id, poly_norm in parsed:
                px = _denorm(poly_norm, w, h)
                if len(px) < 3:
                    continue
                insts.append(
                    Instance(
                        class_id=cls_id,
                        polygon=px,
                        area_px=float(cv2.contourArea(px.reshape(-1, 1, 2))),
                    )
                )
            row = manifest.get(p.name, {})
            images.append(
                YoloImage(
                    path=p,
                    label_path=lp,
                    split=sp,
                    width=int(w),
                    height=int(h),
                    instances=tuple(insts),
                    category=row.get("category") or None,
                    defect_type=row.get("defect_type") or None,
                )
            )
    images.sort(key=lambda im: (im.split, im.path.name))
    return YoloDataset(root=root, names=names, images=images, warnings=warns)


def write_label_file(path: Path, items: Iterable[Tuple[int, np.ndarray]], w: int, h: int) -> int:
    """把 [(cls_id, 像素多边形)] 写成 YOLO-seg 标签，返回写入行数。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [polygon_to_label_line(int(c), np.asarray(poly, dtype=np.int32), w, h) for c, poly in items]
    path.write_text("".join(l + "\n" for l in lines), encoding="utf-8")
    return len(lines)


def dump_stats_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path