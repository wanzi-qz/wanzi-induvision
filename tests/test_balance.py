"""类别不均衡诊断与合成配额规划的测试。

配额语义要对得上 CLI 的行为：
* ``strategy="max"`` -> 目标 = train 最大类实例数 × cap_factor
* ``target=N``       -> 每类至少补到 N
* ``strategy="none"``-> 只诊断不合成
"""
from __future__ import annotations

import math

import pytest

from induvision.balance import (
    ClassStat,
    instance_histogram,
    plan_oversample,
    plan_synthesis,
)
from induvision.yolo_dataset import read_dataset


def _stat(cid: int, name: str, images: int, instances: int, area: float = 100.0) -> ClassStat:
    return ClassStat(
        class_id=cid, name=name, images=images, instances=instances, backgrounds=images,
        area_px_mean=area, area_px_p50=area,
        instances_per_image=round(instances / images, 4) if images else 0.0,
    )


def test_instance_histogram_counts(tiny_yolo):
    ds = read_dataset(tiny_yolo.root)
    hist = instance_histogram(ds, "train")
    assert [h.class_id for h in hist] == [0, 1]
    by_id = {h.class_id: h for h in hist}
    assert by_id[0].instances == 1 and by_id[0].images == 1 and by_id[0].name == "bottle"
    assert by_id[1].instances == 1 and by_id[1].images == 1
    assert by_id[0].backgrounds == 2  # train 里两张 good 图是贴图底板
    assert by_id[0].area_px_mean > 0


def test_histogram_of_empty_split(tiny_yolo):
    ds = read_dataset(tiny_yolo.root)
    hist = instance_histogram(ds, "test")
    assert all(h.instances == 0 for h in hist)
    assert all(h.instances_per_image == 0.0 for h in hist)


def test_plan_strategy_max_uses_cap_factor():
    hist = [_stat(0, "bottle", 40, 50), _stat(1, "toothbrush", 30, 20)]
    plan = plan_synthesis(hist, strategy="max", cap_factor=2.0)
    assert plan.per_class[0]["target_instances"] == 100
    assert plan.per_class[0]["add_instances"] == 50
    assert plan.per_class[1]["add_instances"] == 80
    assert plan.total_instances == 130
    assert "cap_factor" in plan.note


def test_plan_explicit_target_is_a_floor():
    hist = [_stat(0, "bottle", 40, 150), _stat(1, "toothbrush", 30, 20)]
    plan = plan_synthesis(hist, target=100)
    assert plan.per_class[0]["add_instances"] == 0  # 已超标的类不做减法
    assert plan.per_class[1]["add_instances"] == 80
    assert plan.target == 100


def test_plan_add_images_uses_per_image_rate():
    hist = [_stat(0, "bottle", 10, 30)]  # 每图 3 个实例
    plan = plan_synthesis(hist, target=60, max_per_image=3)
    assert plan.per_class[0]["add_instances"] == 30
    assert plan.per_class[0]["add_images"] == 10
    assert plan.total_images == 10

    capped = plan_synthesis(hist, target=60, max_per_image=2)
    assert capped.per_class[0]["add_images"] == 15  # 每图最多 2 个 -> 需要更多图

    sparse = [_stat(0, "bottle", 30, 30)]  # 每图 1 个
    p = plan_synthesis(sparse, target=60, max_per_image=3)
    assert p.per_class[0]["add_images"] == 30  # 但按实际密度 1/图 折算


def test_plan_none_only_diagnoses():
    hist = [_stat(0, "bottle", 40, 50), _stat(1, "toothbrush", 30, 20)]
    plan = plan_synthesis(hist, strategy="none")
    assert plan.total_instances == 0 and plan.total_images == 0
    assert all(v["add_instances"] == 0 and v["add_images"] == 0 for v in plan.per_class.values())
    assert all(v["have_instances"] == v["target_instances"] for v in plan.per_class.values())


def test_plan_rejects_unknown_strategy():
    with pytest.raises(ValueError, match="strategy"):
        plan_synthesis([_stat(0, "bottle", 1, 1)], strategy="smote")


def test_plan_to_dict_is_json_friendly():
    import json

    plan = plan_synthesis([_stat(0, "bottle", 4, 5)], target=10)
    d = plan.to_dict()
    assert json.loads(json.dumps(d))["per_class"]["0"]["add_instances"] == 5


def test_plan_oversample_caps_at_factor():
    hist = [_stat(0, "bottle", 40, 100), _stat(1, "toothbrush", 30, 10)]
    out = plan_oversample(hist, cap_factor=2.0)
    # 目标 = 100*2 = 200；toothbrush 需要 (200-10)/10 = 19 倍，被 cap 到 2 倍 -> 30*2
    assert out[1] == 60
    assert out[0] == 40  # bottle 需要 1 倍 -> 40 张


def test_plan_oversample_skips_empty_classes():
    hist = [_stat(0, "bottle", 40, 100), _stat(1, "ghost", 0, 0)]
    out = plan_oversample(hist)
    assert 1 not in out
    assert out[0] >= 0


def test_plan_oversample_empty_histogram():
    assert plan_oversample([]) == {}


def test_add_images_is_ceil():
    hist = [_stat(0, "bottle", 10, 7)]  # 每图 0.7 -> 取 1.0
    plan = plan_synthesis(hist, target=10)
    assert plan.per_class[0]["add_instances"] == 3
    assert plan.per_class[0]["add_images"] == math.ceil(3 / 1.0)