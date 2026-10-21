"""类别不均衡诊断与合成配额规划。

MVTec 的有监督划分里，缺陷实例数量差得很夸张：bottle 的 broken_small 只有 26 个
实例，而 toothbrush 的 defective 有 66 个；再叠上「官方 train 没有缺陷标注、
只能从 test 里抠」这条限制，稀有类可能只有个位数实例。直接训 YOLO-seg，
稀有类基本学不出来。

这个模块只回答一个问题：**每类还差多少实例、需要合成多少张图**。
它不碰像素，纯计数与规划，输出可以直接喂给 ``tools/synthesize_defects.py``。

配额语义
--------
* ``target=None`` + ``strategy="max"``：以 train 里实例最多的类为基准，把其他类
  补齐到它（再乘 ``cap_factor`` 上限，防止某一类实例极多时把数据量撑爆）。
* ``target=int``：每类至少补到 target 个实例。
* ``strategy="none"``：不合成，只出诊断表。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .yolo_dataset import YoloDataset

__all__ = [
    "ClassStat",
    "instance_histogram",
    "SynthesisPlan",
    "plan_synthesis",
    "plan_oversample",
]


@dataclass
class ClassStat:
    class_id: int
    name: str
    images: int  # train 里含该类实例的图数
    instances: int
    backgrounds: int  # train 里的无标注图数（贴图底板池）
    area_px_mean: float
    area_px_p50: float
    instances_per_image: float


def instance_histogram(ds: YoloDataset, split: str = "train") -> List[ClassStat]:
    """统计某个划分里每类的实例数 / 图数 / 面积，作为配额规划输入。"""
    ims = ds.split(split)
    per_cls: Dict[int, List[float]] = {}
    img_count: Dict[int, int] = {}
    for im in ims:
        seen = set()
        for inst in im.instances:
            per_cls.setdefault(inst.class_id, []).append(inst.area_px)
            seen.add(inst.class_id)
        for cid in seen:
            img_count[cid] = img_count.get(cid, 0) + 1
    backgrounds = sum(1 for im in ims if im.is_background)
    out: List[ClassStat] = []
    for cid in sorted(set(per_cls) | set(range(len(ds.names)))):
        areas = np.asarray(per_cls.get(cid, []), dtype=np.float64)
        n_img = img_count.get(cid, 0)
        n_inst = int(areas.size)
        out.append(
            ClassStat(
                class_id=cid,
                name=ds.class_name(cid),
                images=n_img,
                instances=n_inst,
                backgrounds=backgrounds,
                area_px_mean=round(float(areas.mean()), 2) if areas.size else 0.0,
                area_px_p50=round(float(np.percentile(areas, 50)), 2) if areas.size else 0.0,
                instances_per_image=round(n_inst / n_img, 4) if n_img else 0.0,
            )
        )
    return out


@dataclass
class SynthesisPlan:
    per_class: Dict[int, dict] = field(default_factory=dict)
    total_instances: int = 0
    total_images: int = 0
    strategy: str = "max"
    target: Optional[int] = None
    cap_factor: float = 4.0
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "target": self.target,
            "cap_factor": self.cap_factor,
            "total_instances": self.total_instances,
            "total_images": self.total_images,
            "per_class": {str(k): v for k, v in self.per_class.items()},
            "note": self.note,
        }


def plan_synthesis(
    hist: List[ClassStat],
    *,
    target: Optional[int] = None,
    strategy: str = "max",
    cap_factor: float = 4.0,
    max_per_image: int = 3,
) -> SynthesisPlan:
    """算出每类要补多少实例、折合成多少张合成图。"""
    if strategy not in ("max", "none"):
        raise ValueError(f"strategy 只能是 max/none，收到 {strategy}")
    plan = SynthesisPlan(strategy=strategy, target=target, cap_factor=cap_factor)
    if strategy == "none":
        plan.note = "strategy=none：只诊断不合成"
        for st in hist:
            plan.per_class[st.class_id] = {
                "name": st.name, "have_instances": st.instances, "target_instances": st.instances,
                "add_instances": 0, "add_images": 0,
            }
        return plan

    if target is not None:
        goal = int(target)
    else:
        have = [st.instances for st in hist]
        goal = int(max(have) * cap_factor) if have else 0
        plan.note = f"strategy=max：以 train 最大类实例数为基准 × cap_factor={cap_factor}"

    for st in hist:
        add_inst = max(0, goal - st.instances)
        per_img = min(max(st.instances_per_image, 1.0), float(max_per_image))
        add_img = int(np.ceil(add_inst / per_img)) if add_inst else 0
        plan.per_class[st.class_id] = {
            "name": st.name,
            "have_instances": st.instances,
            "target_instances": goal,
            "add_instances": int(add_inst),
            "add_images": add_img,
            "instances_per_image_assumed": round(per_img, 4),
        }
        plan.total_instances += int(add_inst)
        plan.total_images += add_img
    return plan


def plan_oversample(
    hist: List[ClassStat],
    *,
    target: Optional[int] = None,
    cap_factor: float = 2.0,
) -> Dict[int, int]:
    """真实图硬链接复制配额：把稀有类的**原图**整张复制，补足图级不均衡。

    与 copy-paste 的区别：这里不造新像素，只是让稀有类在 dataloader 里出现得更频繁，
    配合 copy-paste 用。复制倍数封顶 ``cap_factor``，避免同一张图被复制几十遍
    反而加剧过拟合。返回 {class_id: 额外复制的图数}。
    """
    have = [st.instances for st in hist if st.instances]
    if not have:
        return {}
    goal = int(target) if target is not None else int(max(have) * cap_factor)
    out: Dict[int, int] = {}
    for st in hist:
        if st.instances <= 0:
            continue
        need = max(0.0, goal - st.instances) / max(st.instances, 1)
        out[st.class_id] = int(min(np.floor(need), np.floor(cap_factor)) * st.images)
    return out