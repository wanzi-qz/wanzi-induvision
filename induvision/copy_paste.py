"""Copy-Paste 缺陷合成：把 train 里抠出来的缺陷贴到别的图上，标签同步重建。

为什么必须自己写
----------------
MVTec AD 的缺陷实例极少（bottle 三类缺陷加起来 68 个实例，toothbrush 66 个），
YOLO-seg 这种有监督模型直接训会严重过拟合。albumentations 有 CopyPaste，但它
只吃 COCO/自带格式，且不会替你把「贴完之后」的分割标签重建出来——而分割标签
必须和像素**逐像素一致**，否则训练目标就是错的。

这里的做法是：贴图与贴掩码用**同一个仿射矩阵**，融合完成后从掩码重新跑
``mask_to_polygons`` 生成标签行。也就是「先有像素真值，再有多边形」，不存在
标签与像素对不上的可能，代价是多边形会带上融合羽化的轮廓（保真度在报告里量化）。

已知局限（不粉饰）
------------------
* 位置先验来自真实缺陷的质心分布 + 高斯抖动，能保证缺陷大概率落在物体上，
  但**不理解物体形状**：贴到瓶口边缘外、牙刷毛侧面这种物理上不合理的位置仍会发生，
  比例记录在报告的 ``rejected_*`` 计数里，靠拒绝采样压低而不是靠语义。
* Poisson 融合（seamlessClone）会让缺陷边缘的颜色被目标图「拉过去」，视觉上更自然，
  但也会削弱缺陷与背景的对比度——对 broken_large 这种靠形状识别的缺陷是好事，
  对 contamination 这种靠色差识别的缺陷可能反而变难。消融时要分开看。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .defect_bank import DefectBank, DefectInstance, centroid_priors, instances_by_class
from .mvtec import mask_to_polygons

__all__ = [
    "PasteParams",
    "PasteRecord",
    "SynthResult",
    "warp_instance",
    "photometric_jitter",
    "blend_patch",
    "choose_position",
    "paste_instances",
    "masks_to_label_items",
    "render_comparison",
    "rasterize_instances",
]


@dataclass
class PasteParams:
    """合成参数。全部随机量都从调用方传入的 rng 取，模块内不碰全局随机状态。"""

    scale: Tuple[float, float] = (0.6, 1.4)
    rotate: Tuple[float, float] = (-180.0, 180.0)
    flip_p: float = 0.5
    match_source_scale: bool = True  # 按 target_w / src_w 缩放，跨类别贴图时缺陷不会突然变大
    feather: float = 1.5  # 掩码羽化半径（像素），融合用软掩码，标签用 >=0.5 的硬掩码
    blend: str = "poisson"  # poisson | alpha | naive
    poisson_min_area: float = 64.0  # 小于该面积的掩码不做 Poisson（seamlessClone 会报错/失真）
    brightness: Tuple[float, float] = (-0.10, 0.10)
    contrast: Tuple[float, float] = (0.90, 1.10)
    gamma_p: float = 0.3  # 以该概率做一次 gamma 抖动
    gamma: Tuple[float, float] = (0.85, 1.18)
    max_per_image: int = 3
    min_visible_ratio: float = 0.75  # 缺陷被图边界裁掉超过 25% 就重采样位置
    max_overlap_frac: float = 0.0  # 与已有缺陷重叠面积占比上限；默认 0：缺陷互不重叠，
    # 否则两个缺陷的掩码会连成一个连通域，标签里就少一个实例，实例计数与像素对不上
    position: str = "centroid_jitter"  # centroid_jitter | uniform
    jitter_sigma: float = 0.08  # 质心先验的高斯抖动（归一化坐标）
    bg_std_min: float = 0.03  # 落点邻域灰度标准差下限（0~1）；MVTec 背景是近乎无纹理的
    # 纯白/纯黑，标准差接近 0，物体表面则有反光与纹理。低于该值的落点直接拒绝，
    # 否则模型会学到「白底上的一块亮斑 = 缺陷」这种错规律
    candidates: int = 24  # 每张图最多试多少个位置
    epsilon_ratio: float = 0.002  # 与转换管线保持一致
    min_area: float = 4.0
    min_paste_area: float = 64.0  #  warp 后硬掩码小于该面积就不贴：否则像素上有缺陷、标签里没多边形，等于自造标签噪声

    def __post_init__(self) -> None:
        if self.blend not in ("poisson", "alpha", "naive"):
            raise ValueError(f"blend 只能是 poisson/alpha/naive，收到 {self.blend}")
        if self.position not in ("centroid_jitter", "uniform"):
            raise ValueError(f"position 只能是 centroid_jitter/uniform，收到 {self.position}")
        for name in ("scale", "rotate", "brightness", "contrast", "gamma"):
            lo, hi = getattr(self, name)
            if lo > hi:
                raise ValueError(f"{name} 区间反了: {lo} > {hi}")
        if self.scale[0] <= 0:
            raise ValueError(f"scale 下界必须 > 0，收到 {self.scale[0]}")
        if self.max_per_image < 1:
            raise ValueError(f"max_per_image 必须 >= 1，收到 {self.max_per_image}")
        if not 0 < self.min_visible_ratio <= 1:
            raise ValueError(f"min_visible_ratio 应在 (0,1]，收到 {self.min_visible_ratio}")

    def to_dict(self) -> dict:
        d = asdict(self)
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in d.items()}


@dataclass
class PasteRecord:
    """一次成功贴图的完整审计信息，写进 synth_manifest.csv。"""

    instance_id: str
    class_id: int
    class_name: str
    source_image: str
    defect_type: Optional[str]
    dst_x: int
    dst_y: int
    scale: float
    angle: float
    flipped: bool
    blend_used: str
    visible_px: int
    total_px: int
    visible_ratio: float


@dataclass
class SynthResult:
    image: np.ndarray  # (H,W,3) uint8
    masks: Dict[int, np.ndarray]  # class_id -> (H,W) uint8 0/255
    records: List[PasteRecord] = field(default_factory=list)
    rejected: Dict[str, int] = field(default_factory=dict)


def rasterize_instances(instances: Sequence, width: int, height: int) -> Dict[int, np.ndarray]:
    """把已有标签光栅化成 {class_id: mask}，作为贴图前的底板真值。"""
    masks: Dict[int, np.ndarray] = {}
    for inst in instances:
        m = masks.get(int(inst.class_id))
        if m is None:
            m = np.zeros((height, width), np.uint8)
            masks[int(inst.class_id)] = m
        cv2.fillPoly(m, [np.asarray(inst.polygon, np.int32).reshape(-1, 1, 2)], 255)
    return masks


def warp_instance(
    inst: DefectInstance,
    *,
    scale: float,
    angle: float,
    flip: bool,
    feather: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """对 patch 与 mask 施加同一个仿射，返回 (patch, hard_mask, soft_alpha)。

    patch 用 BORDER_REPLICATE：黑色补边会在融合处留一圈暗环，复制边界更接近
    「缺陷本来就在这儿」的观感。掩码用线性插值再高斯羽化，硬掩码取 >=0.5。
    """
    patch = inst.patch
    mask = inst.mask
    if flip:
        patch = cv2.flip(patch, 1)
        mask = cv2.flip(mask, 1)
    h, w = mask.shape[:2]
    sw = max(1, int(round(w * scale)))
    sh = max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    patch = cv2.resize(patch, (sw, sh), interpolation=interp)
    mask = cv2.resize(mask, (sw, sh), interpolation=cv2.INTER_LINEAR)

    center = ((sw - 1) / 2.0, (sh - 1) / 2.0)
    m = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    ow = int(np.ceil(sh * sin + sw * cos))
    oh = int(np.ceil(sh * cos + sw * sin))
    m[0, 2] += (ow - sw) / 2.0
    m[1, 2] += (oh - sh) / 2.0
    patch = cv2.warpAffine(patch, m, (ow, oh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    soft = cv2.warpAffine(mask, m, (ow, oh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    soft = np.clip(soft.astype(np.float32) / 255.0, 0.0, 1.0)
    if feather > 0:
        k = int(max(1, round(feather * 2))) * 2 + 1
        soft = cv2.GaussianBlur(soft, (k, k), feather / 2.0)
        nz = soft > 0
        if nz.any():  # 羽化会把能量摊薄，重新拉满，避免缺陷整体变淡
            soft[nz] = np.clip(soft[nz] / float(soft[nz].max()), 0.0, 1.0)
    hard = np.where(soft >= 0.5, 255, 0).astype(np.uint8)
    return patch, hard, soft


def photometric_jitter(patch: np.ndarray, rng: np.random.Generator, p: PasteParams) -> np.ndarray:
    """亮度 / 对比度 / gamma 抖动，让同一个缺陷贴到不同图上不至于像素级雷同。"""
    out = patch.astype(np.float32)
    c = float(rng.uniform(*p.contrast))
    b = float(rng.uniform(*p.brightness))
    out = out * c + b * 255.0
    if rng.random() < p.gamma_p:
        g = float(rng.uniform(*p.gamma))
        out = 255.0 * np.power(np.clip(out, 0, 255) / 255.0, g)
    return np.clip(out, 0, 255).astype(np.uint8)


def blend_patch(
    img: np.ndarray,
    patch: np.ndarray,
    soft: np.ndarray,
    hard: np.ndarray,
    x: int,
    y: int,
    *,
    mode: str,
    poisson_min_area: float,
) -> Tuple[np.ndarray, str]:
    """把 patch 融进 img 的 (x,y) 处，返回 (融合后的 ROI, 实际用的模式)。"""
    h, w = img.shape[:2]
    ph, pw = patch.shape[:2]
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + pw, w), min(y + ph, h)
    if x1 <= x0 or y1 <= y0:
        return img[y0:y1, x0:x1], "none"
    roi = img[y0:y1, x0:x1]
    p_roi = patch[y0 - y : y1 - y, x0 - x : x1 - x]
    s_roi = soft[y0 - y : y1 - y, x0 - x : x1 - x][..., None]
    h_roi = hard[y0 - y : y1 - y, x0 - x : x1 - x]

    if mode == "poisson" and float(np.count_nonzero(h_roi)) >= poisson_min_area:
        # seamlessClone 要求掩码不贴 dst 边界、且完全落在 dst 内，否则抛异常
        interior = x0 > 0 and y0 > 0 and x1 < w and y1 < h
        if interior:
            try:
                m = np.where(h_roi > 0, 255, 0).astype(np.uint8)
                m = cv2.erode(m, np.ones((3, 3), np.uint8), iterations=1)
                if np.count_nonzero(m) >= max(16.0, poisson_min_area / 4):
                    full_src = np.zeros_like(img)
                    full_src[y0:y1, x0:x1] = p_roi
                    full_mask = np.zeros((h, w), np.uint8)
                    full_mask[y0:y1, x0:x1] = m
                    center = ((x0 + x1) // 2, (y0 + y1) // 2)
                    out = cv2.seamlessClone(full_src, img, full_mask, center, cv2.NORMAL_CLONE)
                    return out[y0:y1, x0:x1], "poisson"
            except cv2.error:
                pass  # 落到 alpha
    if mode == "naive":
        sel = h_roi > 0
        out = roi.copy()
        out[sel] = p_roi[sel]
        return out, "naive"
    return (roi.astype(np.float32) * (1.0 - s_roi) + p_roi.astype(np.float32) * s_roi).astype(np.uint8), "alpha"


def local_std_map(img: np.ndarray, k: int = 33) -> np.ndarray:
    """灰度局部标准差图（0~1），用来区分「物体表面」和「无纹理背景」。"""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    m1 = cv2.boxFilter(g, -1, (k, k))
    m2 = cv2.boxFilter(g * g, -1, (k, k))
    return np.sqrt(np.clip(m2 - m1 * m1, 0.0, None))


def choose_position(
    rng: np.random.Generator,
    priors: np.ndarray,
    patch_size: Tuple[int, int],
    target_size: Tuple[int, int],
    occupied: Optional[np.ndarray],
    p: PasteParams,
    texture: Optional[np.ndarray] = None,
) -> Tuple[Optional[Tuple[int, int]], Dict[str, int]]:
    """采样贴图左上角坐标，失败返回 (None, 拒绝原因计数)。"""
    pw, ph = patch_size
    tw, th = target_size
    rej: Dict[str, int] = {}
    for _ in range(max(1, p.candidates)):
        if p.position == "centroid_jitter" and priors.size:
            cx, cy = priors[rng.integers(len(priors))]
            cx = float(np.clip(rng.normal(cx, p.jitter_sigma), 0.0, 1.0))
            cy = float(np.clip(rng.normal(cy, p.jitter_sigma), 0.0, 1.0))
        else:
            cx, cy = float(rng.random()), float(rng.random())
        x = int(round(cx * (tw - 1) - pw / 2.0))
        y = int(round(cy * (th - 1) - ph / 2.0))

        x0, y0 = max(x, 0), max(y, 0)
        x1, y1 = min(x + pw, tw), min(y + ph, th)
        if x1 <= x0 or y1 <= y0:
            rej["out_of_canvas"] = rej.get("out_of_canvas", 0) + 1
            continue
        # 用一张临时掩码算「落在图内的比例」，避免自己写边界分支
        probe = np.zeros((ph, pw), np.uint8)
        probe[:] = 255
        inside = np.zeros((ph, pw), np.uint8)
        inside[y0 - y : y1 - y, x0 - x : x1 - x] = 255
        if float(np.count_nonzero(inside)) / float(probe.size) < p.min_visible_ratio:
            rej["clipped"] = rej.get("clipped", 0) + 1
            continue
        if texture is not None and p.bg_std_min > 0:
            if float(texture[y0:y1, x0:x1].mean()) < p.bg_std_min:
                rej["flat_background"] = rej.get("flat_background", 0) + 1
                continue
        if occupied is not None and np.count_nonzero(occupied):
            ov = occupied[y0:y1, x0:x1]
            sub = inside[y0 - y : y1 - y, x0 - x : x1 - x]
            frac = float(np.count_nonzero(ov & sub)) / max(float(np.count_nonzero(sub)), 1.0)
            if frac > p.max_overlap_frac:
                rej["overlap"] = rej.get("overlap", 0) + 1
                continue
        return (x, y), rej
    rej["exhausted"] = rej.get("exhausted", 0) + 1
    return None, rej


def paste_instances(
    img: np.ndarray,
    masks: Dict[int, np.ndarray],
    pool: Sequence[DefectInstance],
    rng: np.random.Generator,
    p: PasteParams,
    *,
    priors: Optional[np.ndarray] = None,
    n: Optional[int] = None,
) -> SynthResult:
    """往 img 上贴 n 个缺陷（就地修改 img 与 masks），返回结果与审计记录。"""
    h, w = img.shape[:2]
    res = SynthResult(image=img, masks=masks, rejected={})
    texture: Optional[np.ndarray] = None
    if not pool:
        res.rejected["empty_pool"] = 1
        return res
    n = int(n if n is not None else rng.integers(1, p.max_per_image + 1))
    n = max(1, min(n, p.max_per_image))
    priors = np.asarray(priors if priors is not None else np.zeros((0, 2)), dtype=np.float64).reshape(-1, 2)

    for _ in range(n):
        inst = pool[int(rng.integers(len(pool)))]
        scale = float(rng.uniform(*p.scale))
        if p.match_source_scale and inst.src_size[0] > 0:
            scale *= float(w) / float(inst.src_size[0])
        angle = float(rng.uniform(*p.rotate))
        flip = bool(rng.random() < p.flip_p)
        patch, hard, soft = warp_instance(inst, scale=scale, angle=angle, flip=flip, feather=p.feather)
        hard_px = int(np.count_nonzero(hard))
        if hard_px < max(p.min_paste_area, 1.0):
            res.rejected["too_small"] = res.rejected.get("too_small", 0) + 1
            continue
        patch = photometric_jitter(patch, rng, p)

        occupied = None
        if p.max_overlap_frac < 1.0 and masks:
            occupied = np.zeros((h, w), np.uint8)
            for m in masks.values():
                occupied |= m
        if texture is None and p.bg_std_min > 0:
            texture = local_std_map(res.image)
        (xy, rej) = choose_position(
            rng, priors, (patch.shape[1], patch.shape[0]), (w, h), occupied, p, texture=texture
        )
        for k, v in rej.items():
            res.rejected[k] = res.rejected.get(k, 0) + v
        if xy is None:
            continue
        x, y = xy

        out_roi, used = blend_patch(
            res.image, patch, soft, hard, x, y, mode=p.blend, poisson_min_area=p.poisson_min_area
        )
        # out_roi 是裁剪到画布内的那块（blend_patch 内部已处理越界），按同样的窗口写回
        x0, y0 = max(x, 0), max(y, 0)
        res.image[y0 : y0 + out_roi.shape[0], x0 : x0 + out_roi.shape[1]] = out_roi

        dst = masks.get(inst.class_id)
        if dst is None:
            dst = np.zeros((h, w), np.uint8)
            masks[inst.class_id] = dst
        ph, pw = hard.shape[:2]
        x0, y0 = max(x, 0), max(y, 0)
        x1, y1 = min(x + pw, w), min(y + ph, h)
        dst[y0:y1, x0:x1] |= hard[y0 - y : y1 - y, x0 - x : x1 - x]

        vis = int(np.count_nonzero(hard[y0 - y : y1 - y, x0 - x : x1 - x]))
        tot = int(np.count_nonzero(hard))
        res.records.append(
            PasteRecord(
                instance_id=inst.instance_id,
                class_id=int(inst.class_id),
                class_name=inst.class_name,
                source_image=inst.source_image,
                defect_type=inst.defect_type,
                dst_x=int(x),
                dst_y=int(y),
                scale=round(scale, 4),
                angle=round(angle, 2),
                flipped=flip,
                blend_used=used,
                visible_px=vis,
                total_px=tot,
                visible_ratio=round(vis / max(tot, 1), 4),
            )
        )
    return res


def masks_to_label_items(
    masks: Dict[int, np.ndarray],
    *,
    epsilon_ratio: float = 0.002,
    min_area: float = 4.0,
) -> Tuple[List[Tuple[int, np.ndarray]], dict]:
    """{class_id: mask} -> ([(cls_id, 像素多边形)], 保真度统计)。"""
    items: List[Tuple[int, np.ndarray]] = []
    fidelity: List[float] = []
    mask_px = 0
    poly_px = 0.0
    for cid in sorted(masks):
        ps = mask_to_polygons(masks[cid], epsilon_ratio=epsilon_ratio, min_area=min_area)
        mask_px += ps.mask_pixels
        poly_px += ps.polygon_pixels
        if ps.mask_pixels:
            fidelity.append(ps.polygon_pixels / ps.mask_pixels)
        for poly in ps.polygons:
            items.append((cid, poly))
    stats = {
        "mask_pixels": int(mask_px),
        "polygon_pixels": round(float(poly_px), 2),
        "area_fidelity": round(float(poly_px / mask_px), 6) if mask_px else None,
        "area_fidelity_p10": round(float(np.percentile(fidelity, 10)), 6) if fidelity else None,
        "polygons": len(items),
    }
    return items, stats


def render_comparison(
    before: np.ndarray,
    after: np.ndarray,
    masks: Dict[int, np.ndarray],
    records: Sequence[PasteRecord],
    *,
    palette: Sequence[Tuple[int, int, int]] = ((0, 0, 255), (0, 200, 0), (255, 140, 0), (200, 0, 200)),
) -> np.ndarray:
    """左原图 / 右合成图（掩码描边）+ 底部记录条，用于人工抽查合成质量。"""
    left = before.copy()
    right = after.copy()
    for cid, m in masks.items():
        color = palette[cid % len(palette)]
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(right, cnts, -1, color, 2)
    gap = np.full((left.shape[0], 6, 3), 255, np.uint8)
    bar_h = 22 + 14 * max(len(records), 1)
    bar = np.full((bar_h, left.shape[1] * 2 + 6, 3), 250, np.uint8)
    cv2.putText(bar, f"{len(records)} pasted", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    for i, r in enumerate(records):
        txt = f"{r.instance_id} s={r.scale:.2f} a={r.angle:.0f} {r.blend_used} vis={r.visible_ratio:.2f}"
        cv2.putText(bar, txt, (6, 30 + 14 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (30, 30, 30), 1, cv2.LINE_AA)
    return np.vstack([np.hstack([left, gap, right]), bar])