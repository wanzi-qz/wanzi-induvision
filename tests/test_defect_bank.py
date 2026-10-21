"""缺陷库（defect bank）测试：核心是**防真值泄漏**与 npz 缓存往返一致。

泄漏口径：缺陷贴片只能从 train 划分抠。从 val/test 抠缺陷贴进 train，
等于把评测集的答案喂给模型，指标会虚高——所以默认直接抛错，
只有显式 ``allow_eval_leak=True``（纯出诊断图）才放行。
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from induvision.defect_bank import (
    TRAIN_ONLY_SPLITS,
    bank_stats,
    build_bank,
    centroid_priors,
    instances_by_class,
    load_bank,
    save_bank,
)
from induvision.yolo_dataset import read_dataset


@pytest.fixture
def ds(tiny_yolo):
    return read_dataset(tiny_yolo.root)


@pytest.fixture
def bank(ds):
    return build_bank(ds)


def test_train_only_by_default(bank, ds):
    n_train_inst = sum(len(im.instances) for im in ds.split("train"))
    assert len(bank) == n_train_inst == 2
    assert {d.source_split for d in bank.instances} == {"train"}
    assert bank.built_from == TRAIN_ONLY_SPLITS == ("train",)
    assert bank.params["allow_eval_leak"] is False


def test_refuses_eval_splits(ds):
    with pytest.raises(ValueError, match="泄漏"):
        build_bank(ds, splits=("train", "val"))
    with pytest.raises(ValueError, match="泄漏"):
        build_bank(ds, splits=("test",))


def test_allow_eval_leak_is_explicit(ds):
    leaky = build_bank(ds, splits=("train", "val"), allow_eval_leak=True)
    assert len(leaky) == 3
    assert leaky.built_from == ("train", "val")
    assert leaky.params["allow_eval_leak"] is True
    assert {d.source_split for d in leaky.instances} == {"train", "val"}


def test_patch_mask_bbox_are_consistent(bank, tiny_yolo):
    for d in bank.instances:
        x, y, bw, bh = d.bbox
        assert d.mask.shape == (bh, bw), d.instance_id
        assert d.patch.shape == (bh, bw, 3)
        assert d.mask.dtype == np.uint8 and d.mask.max() == 255
        assert np.count_nonzero(d.mask) > 0
        assert d.src_size == tiny_yolo.size
        src = cv2.imread(str(tiny_yolo.image("train", d.source_image.split(".")[0])))
        np.testing.assert_array_equal(src[y : y + bh, x : x + bw], d.patch)
        # 掩码面积应与多边形面积同量级（矩形：contourArea 与 nonzero 相差在几个像素内）
        assert abs(np.count_nonzero(d.mask) - d.area_px) <= d.area_px * 0.15 + 8


def test_known_instance_geometry(bank):
    """def_a__000 的 class 0 矩形 (18,14)-(46,40)：bbox 必须正好框住它。"""
    d = next(x for x in bank.instances if x.class_id == 0)
    assert (d.source_image, d.class_name, d.defect_type) == ("def_a__000.png", "bottle", "broken_large")
    assert d.bbox == (18, 14, 29, 27)
    cx, cy = d.centroid_norm
    assert cx == pytest.approx(32 / 95, abs=1e-5)
    assert cy == pytest.approx(27 / 71, abs=1e-5)


def test_pad_expands_bbox_and_keeps_mask(bank, ds):
    padded = build_bank(ds, pad=2)
    d0 = next(x for x in bank.instances if x.class_id == 0)
    d1 = next(x for x in padded.instances if x.class_id == 0)
    assert d1.bbox == (16, 12, 33, 31)
    assert np.count_nonzero(d1.mask) == np.count_nonzero(d0.mask)
    # 扩出来的边框部分掩码为 0（只带缺陷本体，不带源图背景光照）
    assert d1.mask[0, 0] == 0 and d1.mask[-1, -1] == 0


def test_min_area_filters_tiny_instances(ds):
    assert len(build_bank(ds, min_area=10_000)) == 0
    assert len(build_bank(ds, min_area=4)) == 2


def test_instances_by_class_and_priors(bank):
    by_cls = instances_by_class(bank)
    assert sorted(by_cls) == [0, 1]
    assert [d.class_id for d in by_cls[0]] == [0]

    p0 = centroid_priors(bank, 0)
    assert p0.shape == (1, 2)
    assert ((p0 >= 0.0) & (p0 <= 1.0)).all()
    assert centroid_priors(bank, 99).shape == (0, 2)


def test_bank_stats(bank):
    st = bank_stats(bank)
    assert st["instances"] == 2
    assert st["source_images"] == 2
    assert st["built_from"] == ["train"]
    assert set(st["per_class"]) == {"bottle", "toothbrush"}
    b = st["per_class"]["bottle"]
    assert b["class_id"] == 0 and b["instances"] == 1
    assert b["defect_types"] == ["broken_large"]
    assert b["area_px_min"] <= b["area_px_p50"] <= b["area_px_max"]


def test_save_load_roundtrip_is_identical(bank, tmp_path):
    path = save_bank(bank, tmp_path / "cache" / "bank.npz")
    assert path.is_file()
    got = load_bank(path)

    assert got.names == bank.names
    assert got.built_from == bank.built_from
    assert got.params == bank.params
    assert str(got.source_root) == str(bank.source_root)
    assert len(got) == len(bank)
    for a, b in zip(bank.instances, got.instances):
        meta_a = (a.instance_id, a.class_id, a.class_name, a.source_image, a.source_split,
                  a.category, a.defect_type, a.bbox, a.area_px, a.centroid_norm, a.src_size)
        meta_b = (b.instance_id, b.class_id, b.class_name, b.source_image, b.source_split,
                  b.category, b.defect_type, b.bbox, b.area_px, b.centroid_norm, b.src_size)
        assert meta_a == meta_b
        np.testing.assert_array_equal(a.patch, b.patch)
        np.testing.assert_array_equal(a.mask, b.mask)
        assert a.size == b.size


def test_empty_bank_roundtrip(tmp_path):
    from induvision.defect_bank import DefectBank

    empty = DefectBank(names=["bottle"], source_root=tmp_path)
    got = load_bank(save_bank(empty, tmp_path / "empty.npz"))
    assert len(got) == 0 and got.names == ["bottle"]