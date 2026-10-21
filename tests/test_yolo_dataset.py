"""YOLO-seg 数据集读写与归一化口径的回归测试。

重点盯两件事：
1. **归一化口径**：写标签用 ``x/(w-1)``，读回来必须用 ``x*(w-1)``。两边不一致时
   贴边多边形会整体偏移半个像素，肉眼看不出来，但「合成前后 IoU」会凭空掉点。
2. **背景图是合法的**：MVTec 的 good 样本没有标签文件，不能当成错误。
"""
from __future__ import annotations

import numpy as np
import pytest

from conftest import build_tiny_yolo
from induvision.yolo_dataset import (
    SPLITS,
    _denorm,
    parse_label_file,
    read_dataset,
    read_manifest,
    write_label_file,
)


def test_roundtrip_preserves_pixel_polygons(tiny_yolo):
    """write_label_file -> read_dataset 必须逐像素还原多边形。"""
    ds = read_dataset(tiny_yolo.root)
    assert ds.names == ["bottle", "toothbrush"]
    assert not ds.warnings, ds.warnings
    for stem, want in tiny_yolo.expected.items():
        im = next(i for i in ds.images if i.stem == stem)
        assert len(im.instances) == len(want), stem
        for inst, (cls_id, poly) in zip(im.instances, want):
            assert inst.class_id == cls_id
            np.testing.assert_array_equal(inst.polygon, poly)
            assert inst.area_px > 0


def test_normalization_convention_is_w_minus_1(tmp_path):
    """贴边多边形归一化后必须正好是 1.0；用 /w 的话会是 0.975。"""
    w, h = 40, 30
    poly = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.int32)
    p = tmp_path / "edge.txt"
    assert write_label_file(p, [(0, poly)], w, h) == 1

    items, warns = parse_label_file(p)
    assert not warns
    norm = items[0][1]
    assert norm[:, 0].max() == pytest.approx(1.0)
    assert norm[:, 1].max() == pytest.approx(1.0)
    np.testing.assert_array_equal(_denorm(norm, w, h), poly)


def test_parse_label_file_skips_malformed_lines(tmp_path):
    p = tmp_path / "bad.txt"
    p.write_text(
        "\n".join(
            [
                "0 0.1 0.1 0.5 0.1 0.5 0.5",  # 合法
                "   ",  # 空白行
                "1 0.2 0.2 0.6 0.6",  # 只有 4 个坐标
                "x 0.1 0.1 0.5 0.1 0.5 0.5",  # 类别非数字
                "0 0.1 0.1 0.5 0.1 0.5",  # 5 个坐标（奇数）
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    items, warns = parse_label_file(p)
    assert len(items) == 1 and items[0][0] == 0
    assert len(warns) == 3, warns


def test_parse_label_file_missing_is_empty(tmp_path):
    items, warns = parse_label_file(tmp_path / "nope.txt")
    assert items == [] and warns == []


def test_background_images_are_allowed(tiny_yolo):
    ds = read_dataset(tiny_yolo.root)
    bgs = {im.stem for im in ds.split("train") if im.is_background}
    assert bgs == {"bg__000", "bg__001"}
    for im in ds.split("train"):
        if im.is_background:
            assert not im.label_path.is_file()
            assert im.class_ids == ()


def test_require_labels_raises(tiny_yolo):
    with pytest.raises(FileNotFoundError, match="标签"):
        read_dataset(tiny_yolo.root, require_labels=True)


def test_manifest_join_gives_category_and_defect_type(tiny_yolo):
    ds = read_dataset(tiny_yolo.root)
    im = next(i for i in ds.images if i.stem == "def_b__000")
    assert (im.category, im.defect_type) == ("toothbrush", "broken_large")
    good = next(i for i in ds.images if i.stem == "bg__000")
    assert good.defect_type == "good"
    man = read_manifest(tiny_yolo.root)
    assert man["def_a__000.png"]["category"] == "bottle"


def test_read_manifest_missing_returns_empty(tmp_path):
    assert read_manifest(tmp_path) == {}


def test_read_dataset_errors(tmp_path, tiny_yolo):
    with pytest.raises(FileNotFoundError):
        read_dataset(tmp_path / "nope")
    with pytest.raises(ValueError, match="未知划分"):
        read_dataset(tiny_yolo.root, splits=("train", "dev"))


def test_missing_data_yaml_degrades_to_warning(tmp_path):
    t = build_tiny_yolo(tmp_path / "y2")
    (t.root / "data.yaml").unlink()
    ds = read_dataset(t.root)
    assert ds.names == []
    assert any("data.yaml" in w for w in ds.warnings)
    assert ds.class_name(0) == "cls0"
    assert ds.class_name(1) == "cls1"


def test_split_filter_and_by_class(tiny_yolo):
    train_only = read_dataset(tiny_yolo.root, splits=("train",))
    assert {im.split for im in train_only.images} == {"train"}
    assert len(train_only.images) == 4

    full = read_dataset(tiny_yolo.root)
    assert {im.split for im in full.images} == set(SPLITS)
    assert [im.stem for im in full.by_class(1)] == ["def_b__000"]
    assert [im.stem for im in full.by_class(0)] == ["def_a__000", "val_a__000"]
    assert full.by_class(7) == []
    assert next(i for i in full.images if i.stem == "def_a__000").class_ids == (0,)


def test_labels_sorted_by_split_then_name(tiny_yolo):
    ds = read_dataset(tiny_yolo.root)
    keys = [(im.split, im.path.name) for im in ds.images]
    assert keys == sorted(keys)