"""缺陷实例库（defect bank）：从已标注图里把缺陷「抠」出来备用。

copy-paste 增强的原料。做法是把 YOLO-seg 标签的每个多边形光栅化成掩码，
按外接框从原图裁出 patch，连同掩码、类别、源文件、归一化质心一起入库。

两条硬约束
----------
1. **只能从 train 划分取材。** 从 val/test 抠缺陷再贴进 train，等于把评估集的
   真值喂给模型，指标会虚高且无法察觉。默认 ``splits=("train",)``，传入
   val/test 直接抛错，除非显式 ``allow_eval_leak=True``（只为做诊断图）。
2. **不存盘原图整张。** 库里只有 bbox 大小的 patch，几十 KB 级；缓存成 npz
   也只落在 ``out/``（已被 .gitignore 排除），不进仓库。
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .yolo_dataset import YoloDataset, YoloImage

__all__ = [
    "DefectInstance",
    "DefectBank",
    "build_bank",
    "instances_by_class",
    "centroid_priors",
    "bank_stats",
    "save_bank",
    "load_bank",
    "TRAIN_ONLY_SPLITS",
]

TRAIN_ONLY_SPLITS: Tuple[str, ...] = ("train",)


@dataclass(frozen=True)
class DefectInstance:
    """一个可复用的缺陷贴片。"""

    instance_id: str
    class_id: int
    class_name: str
    source_image: str  # 数据集里的文件名，可回溯到 manifest.csv 再回到 MVTec 原图
    source_split: str
    category: Optional[str]
    defect_type: Optional[str]
    bbox: Tuple[int, int, int, int]  # x, y, w, h（源图像素）
    patch: np.ndarray  # (bh, bw, 3) uint8 BGR
    mask: np.ndarray  # (bh, bw) uint8，0 / 255
    area_px: float  # 多边形面积（不是 patch 面积）
    centroid_norm: Tuple[float, float]  # 质心在源图里的归一化坐标，做位置先验
    src_size: Tuple[int, int]  # 源图 (w, h)

    @property
    def size(self) -> Tuple[int, int]:
        return int(self.mask.shape[1]), int(self.mask.shape[0])


@dataclass
class DefectBank:
    instances: List[DefectInstance] = field(default_factory=list)
    names: List[str] = field(default_factory=list)
    built_from: Tuple[str, ...] = TRAIN_ONLY_SPLITS
    source_root: Path = Path(".")
    params: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.instances)

    def class_name(self, class_id: int) -> str:
        if 0 <= class_id < len(self.names):
            return self.names[class_id]
        return f"cls{class_id}"


def _rasterize(poly: np.ndarray, w: int, h: int) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """多边形 -> (局部掩码, bbox)。bbox 已裁剪到图内，掩码尺寸与 bbox 一致。"""
    xs, ys = poly[:, 0], poly[:, 1]
    x0, y0 = int(max(xs.min(), 0)), int(max(ys.min(), 0))
    x1, y1 = int(min(xs.max(), w - 1)), int(min(ys.max(), h - 1))
    if x1 <= x0 or y1 <= y0:
        return np.zeros((0, 0), np.uint8), (0, 0, 0, 0)
    local = np.zeros((y1 - y0 + 1, x1 - x0 + 1), np.uint8)
    shifted = (poly - np.array([x0, y0], dtype=poly.dtype)).astype(np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(local, [shifted], 255)
    return local, (x0, y0, x1 - x0 + 1, y1 - y0 + 1)


def build_bank(
    ds: YoloDataset,
    *,
    splits: Sequence[str] = TRAIN_ONLY_SPLITS,
    min_area: float = 4.0,
    pad: int = 0,
    allow_eval_leak: bool = False,
) -> DefectBank:
    """从数据集里抽取全部缺陷实例。

    ``pad`` 表示 bbox 向外扩几个像素（连 patch 与掩码一起扩，扩出来的部分掩码为 0）。
    默认 0：只保留缺陷本体，不带源图背景，避免把 A 图的光照贴到 B 图上。
    """
    splits = tuple(splits)
    bad = [s for s in splits if s != "train"]
    if bad and not allow_eval_leak:
        raise ValueError(
            f"缺陷库只能从 train 划分取材，收到 {bad}。"
            "从 val/test 抠缺陷贴进 train 会造成真值泄漏、指标虚高；"
            "确实只是为了出诊断图时，显式传 allow_eval_leak=True。"
        )
    pad = int(max(pad, 0))
    bank = DefectBank(
        names=list(ds.names),
        built_from=splits,
        source_root=ds.root,
        params={"min_area": float(min_area), "pad": pad, "allow_eval_leak": bool(allow_eval_leak)},
    )

    picked: List[YoloImage] = [im for im in ds.images if im.split in splits and im.instances]
    picked.sort(key=lambda im: (im.split, im.path.name))
    for im in picked:
        img = cv2.imread(str(im.path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        for k, inst in enumerate(sorted(im.instances, key=lambda i: (i.class_id, -i.area_px))):
            if inst.area_px < min_area or len(inst.polygon) < 3:
                continue
            mask, bbox = _rasterize(inst.polygon, w, h)
            if mask.size == 0 or not np.count_nonzero(mask):
                continue
            x, y, bw, bh = bbox
            if pad:
                x0, y0 = max(x - pad, 0), max(y - pad, 0)
                x1, y1 = min(x + bw + pad, w), min(y + bh + pad, h)
                big_mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
                big_mask[y - y0 : y - y0 + bh, x - x0 : x - x0 + bw] = mask
                mask, bbox = big_mask, (x0, y0, x1 - x0, y1 - y0)
                x, y, bw, bh = bbox
            patch = img[y : y + bh, x : x + bw].copy()
            cx = float(np.average(inst.polygon[:, 0])) / max(w - 1, 1)
            cy = float(np.average(inst.polygon[:, 1])) / max(h - 1, 1)
            bank.instances.append(
                DefectInstance(
                    instance_id=f"{im.path.stem}#{k}",
                    class_id=int(inst.class_id),
                    class_name=ds.class_name(int(inst.class_id)),
                    source_image=im.path.name,
                    source_split=im.split,
                    category=im.category,
                    defect_type=im.defect_type,
                    bbox=(int(x), int(y), int(bw), int(bh)),
                    patch=patch,
                    mask=mask,
                    area_px=float(inst.area_px),
                    centroid_norm=(round(cx, 6), round(cy, 6)),
                    src_size=(int(w), int(h)),
                )
            )
    bank.instances.sort(key=lambda d: (d.source_split, d.source_image, d.class_id, d.instance_id))
    return bank


def instances_by_class(bank: DefectBank) -> Dict[int, List[DefectInstance]]:
    out: Dict[int, List[DefectInstance]] = defaultdict(list)
    for d in bank.instances:
        out[d.class_id].append(d)
    return dict(sorted(out.items()))


def centroid_priors(bank: DefectBank, class_id: int) -> np.ndarray:
    """该类缺陷在源图里的归一化质心集合，形状 (N,2)。

    贴哪儿不能瞎撒：MVTec 的缺陷都长在物体上，均匀撒一半会落到背景里。
    用真实质心分布当先验，再加抖动，是不引入额外标注成本的做法。
    N=0 时返回空数组，调用方需回退到均匀采样。
    """
    pts = [d.centroid_norm for d in bank.instances if d.class_id == class_id]
    return np.asarray(pts, dtype=np.float64).reshape(-1, 2)


def bank_stats(bank: DefectBank) -> dict:
    per_cls: Dict[str, dict] = {}
    for cid, items in instances_by_class(bank).items():
        areas = np.asarray([d.area_px for d in items], dtype=np.float64)
        per_cls[bank.class_name(cid)] = {
            "class_id": cid,
            "instances": len(items),
            "source_images": len({d.source_image for d in items}),
            "area_px_mean": round(float(areas.mean()), 2) if areas.size else 0.0,
            "area_px_p50": round(float(np.percentile(areas, 50)), 2) if areas.size else 0.0,
            "area_px_min": round(float(areas.min()), 2) if areas.size else 0.0,
            "area_px_max": round(float(areas.max()), 2) if areas.size else 0.0,
            "defect_types": sorted({d.defect_type for d in items if d.defect_type}),
        }
    return {
        "instances": len(bank.instances),
        "source_images": len({d.source_image for d in bank.instances}),
        "built_from": list(bank.built_from),
        "source_root": str(bank.source_root),
        "params": bank.params,
        "per_class": per_cls,
    }


def save_bank(bank: DefectBank, path: Path) -> Path:
    """缓存成单个 npz：patch/mask 逐个存，元信息存 JSON 字符串。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, np.ndarray] = {}
    meta = []
    for i, d in enumerate(bank.instances):
        payload[f"patch_{i}"] = d.patch
        payload[f"mask_{i}"] = d.mask
        meta.append(
            {
                "instance_id": d.instance_id,
                "class_id": d.class_id,
                "class_name": d.class_name,
                "source_image": d.source_image,
                "source_split": d.source_split,
                "category": d.category,
                "defect_type": d.defect_type,
                "bbox": list(d.bbox),
                "area_px": d.area_px,
                "centroid_norm": list(d.centroid_norm),
                "src_size": list(d.src_size),
            }
        )
    payload["meta_json"] = np.asarray(
        json.dumps(
            {"instances": meta, "names": bank.names, "built_from": list(bank.built_from),
             "source_root": str(bank.source_root), "params": bank.params},
            ensure_ascii=False,
        )
    )
    np.savez_compressed(str(path), **payload)
    return path


def load_bank(path: Path) -> DefectBank:
    path = Path(path)
    with np.load(str(path), allow_pickle=False) as z:
        meta = json.loads(str(z["meta_json"]))
        out = DefectBank(
            names=list(meta["names"]),
            built_from=tuple(meta["built_from"]),
            source_root=Path(meta["source_root"]),
            params=dict(meta["params"]),
        )
        for i, m in enumerate(meta["instances"]):
            out.instances.append(
                DefectInstance(
                    instance_id=m["instance_id"],
                    class_id=int(m["class_id"]),
                    class_name=m["class_name"],
                    source_image=m["source_image"],
                    source_split=m["source_split"],
                    category=m.get("category"),
                    defect_type=m.get("defect_type"),
                    bbox=tuple(int(v) for v in m["bbox"]),
                    patch=z[f"patch_{i}"],
                    mask=z[f"mask_{i}"],
                    area_px=float(m["area_px"]),
                    centroid_norm=(float(m["centroid_norm"][0]), float(m["centroid_norm"][1])),
                    src_size=(int(m["src_size"][0]), int(m["src_size"][1])),
                )
            )
    return out