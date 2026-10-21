"""induvision.mvtec 的单元测试。"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from conftest import build_mvtec_tree

from induvision.mvtec import (
    MVTecLayoutError,
    Sample,
    build_splits,
    convert_dataset,
    find_dataset_root,
    mask_to_polygons,
    scan_dataset,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


# ------------------------------- 扫描 ------------------------------- #
def test_scan_discovers_defect_types_dynamically(mvtec_tree):
    scan = scan_dataset(mvtec_tree)
    assert scan.categories == ["bottle", "screw"]
    assert len(scan.samples) == 33
    dts = {}
    for s in scan.samples:
        dts.setdefault(s.category, set()).add(s.defect_type)
    assert dts["bottle"] == {"good", "broken_large", "broken_small"}
    assert dts["screw"] == {"good", "thread", "speck"}
    defects = [s for s in scan.samples if s.is_defect]
    assert len(defects) == 13
    assert all(s.mask is not None and s.mask.is_file() for s in defects)
    assert all(s.mask is None for s in scan.samples if not s.is_defect)


def test_scan_subset_categories(mvtec_tree):
    scan = scan_dataset(mvtec_tree, ["bottle"])
    assert scan.categories == ["bottle"]
    assert {s.category for s in scan.samples} == {"bottle"}


def test_scan_unknown_category_raises(mvtec_tree):
    with pytest.raises(MVTecLayoutError):
        scan_dataset(mvtec_tree, ["not_a_category"])


def test_find_dataset_root_through_wrapper_dir(wrapped_tree):
    assert find_dataset_root(wrapped_tree).name == "mvtec_anomaly_detection"
    scan = scan_dataset(wrapped_tree)
    assert len(scan.samples) == 33


def test_missing_mask_strict_raises_lenient_warns(tmp_path):
    tree = build_mvtec_tree(tmp_path / "t", spec={"bottle": {"broken_large": 3}})
    victim = sorted((tree / "bottle" / "ground_truth" / "broken_large").glob("*_mask.png"))[0]
    victim.unlink()
    with pytest.raises(MVTecLayoutError, match="缺少掩码"):
        scan_dataset(tree)
    scan = scan_dataset(tree, strict=False)
    assert len([s for s in scan.samples if s.is_defect]) == 2
    assert any("缺少掩码" in w for w in scan.warnings)


def test_broken_root_raises(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(MVTecLayoutError):
        find_dataset_root(tmp_path / "empty")


# ----------------------------- 掩码 -> 多边形 ----------------------------- #
def test_donut_hole_uses_external_contour():
    mask = np.zeros((80, 80), np.uint8)
    cv2.circle(mask, (40, 40), 25, 255, -1)
    cv2.circle(mask, (40, 40), 10, 0, -1)  # 内孔
    ps = mask_to_polygons(mask)
    assert len(ps.polygons) == 1
    ring_px = np.count_nonzero(mask)
    # RETR_EXTERNAL 会把孔填掉，所以多边形面积必然大于掩码面积
    assert ps.polygon_pixels > ps.mask_pixels == ring_px
    # 内孔被填掉 + contourArea 对栅格圆的低估，实测约 1.15
    assert ps.polygon_pixels / ring_px > 1.05


def test_min_area_filters_tiny_specks():
    mask = np.zeros((64, 64), np.uint8)
    mask[20:22, 30:32] = 255  # 2x2 斑点，contourArea = 1
    assert mask_to_polygons(mask, min_area=4.0).polygons == []
    assert len(mask_to_polygons(mask, min_area=0.5).polygons) == 1


def test_multiple_components_become_multiple_instances():
    mask = np.zeros((64, 64), np.uint8)
    cv2.rectangle(mask, (5, 5), (20, 20), 255, -1)
    cv2.rectangle(mask, (35, 35), (55, 55), 255, -1)
    ps = mask_to_polygons(mask)
    assert len(ps.polygons) == 2
    assert all(len(p) >= 3 for p in ps.polygons)


def test_epsilon_ratio_controls_vertex_count():
    mask = np.zeros((120, 120), np.uint8)
    rng = np.random.default_rng(1)
    pts = np.array([[60 + 40 * np.cos(t), 60 + 40 * np.sin(t)] for t in np.linspace(0, 2 * np.pi, 60, endpoint=False)])
    pts += rng.normal(0, 4, pts.shape)
    cv2.fillPoly(mask, [pts.astype(np.int32)], 255)
    fine = mask_to_polygons(mask, epsilon_ratio=0.0005)
    coarse = mask_to_polygons(mask, epsilon_ratio=0.02)
    assert len(coarse.polygons[0]) < len(fine.polygons[0])


def test_bgr_mask_is_accepted():
    gray = np.zeros((40, 40), np.uint8)
    cv2.rectangle(gray, (8, 8), (20, 20), 255, -1)
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    assert len(mask_to_polygons(bgr).polygons) == 1


# -------------------------------- 划分 -------------------------------- #
def _fake_samples(cat: str, dt: str, n: int):
    return [Sample(cat, dt, Path(f"{cat}/test/{dt}/{i:03d}.png"), Path(f"m{i}.png"), "test")
            for i in range(n)]


def test_split_is_deterministic_for_same_seed():
    samples = _fake_samples("bottle", "broken_large", 40)
    a = build_splits(samples, seed=7)
    b = build_splits(samples, seed=7)
    key = lambda d: {k: [s.image.name for s in v] for k, v in d.items()}  # noqa: E731
    assert key(a) == key(b)


def test_split_changes_with_seed():
    samples = _fake_samples("bottle", "broken_large", 40)
    a = build_splits(samples, seed=1)
    b = build_splits(samples, seed=2)
    assert [s.image.name for s in a["train"]] != [s.image.name for s in b["train"]]


def test_split_ratios_and_no_leakage(mvtec_tree):
    scan = scan_dataset(mvtec_tree)
    splits = build_splits(scan.samples, train_ratio=0.7, val_ratio=0.15, seed=0)
    names = {k: {s.key for s in v} for k, v in splits.items()}
    # 三个子集互不重叠
    assert not (names["train"] & names["val"])
    assert not (names["train"] & names["test"])
    assert not (names["val"] & names["test"])
    total = sum(len(v) for v in splits.values())
    assert total == len(scan.samples)


def test_val_covers_every_defect_type_when_possible(mvtec_tree):
    scan = scan_dataset(mvtec_tree)
    splits = build_splits(scan.samples, seed=0)
    val_keys = {(s.category, s.defect_type) for s in splits["val"] if s.is_defect}
    sizes: dict = {}
    for s in scan.samples:
        if s.is_defect:
            sizes[(s.category, s.defect_type)] = sizes.get((s.category, s.defect_type), 0) + 1
    for k, n in sizes.items():
        if n >= 2:
            assert k in val_keys, f"{k}(n={n}) 未出现在验证集"


def test_good_train_goes_to_train_as_background(mvtec_tree):
    scan = scan_dataset(mvtec_tree)
    with_good = build_splits(scan.samples, seed=0, include_good_train=True)
    without = build_splits(scan.samples, seed=0, include_good_train=False)
    n_good_train = len([s for s in scan.samples if not s.is_defect and s.official_split == "train"])
    assert n_good_train == 12
    assert len(with_good["train"]) - len(without["train"]) == n_good_train
    assert not any(s.official_split == "train" and not s.is_defect for s in without["train"])


def test_invalid_ratios_raise():
    samples = _fake_samples("bottle", "x", 5)
    with pytest.raises(ValueError):
        build_splits(samples, train_ratio=0.0)
    with pytest.raises(ValueError):
        build_splits(samples, train_ratio=0.8, val_ratio=0.4)


# -------------------------------- 转换 -------------------------------- #
def _read_labels(labels_dir: Path):
    return {p: p.read_text(encoding="utf-8").splitlines() for p in sorted(labels_dir.glob("*.txt"))}


def test_convert_writes_yolo_dataset(mvtec_tree, tmp_path):
    out = tmp_path / "yolo"
    rep = convert_dataset(mvtec_tree, out, workers=1)
    # 33 张原图里有 1 张 screw/speck（2x2 掩码）被 min_area 过滤后丢弃
    assert rep.n_written == 32
    assert rep.stats["protocol"]["dropped_tiny_defect_total"] == 1
    cfg = yaml.safe_load(rep.data_yaml.read_text(encoding="utf-8"))
    assert cfg["train"] == "images/train"
    assert cfg["names"] == {0: "bottle", 1: "screw"}

    n_img = len(list((out / "images").rglob("*.png")))
    n_lbl = len(list((out / "labels").rglob("*.txt")))
    assert n_img == n_lbl == 32

    st = rep.stats
    assert st["totals"]["images_written"] == 32
    assert st["totals"]["images_scanned"] == 33
    assert sum(v["images"] for v in st["splits"].values()) == 32
    # speck 2x2 被 min_area 过滤 -> 实例数少于缺陷图数
    assert st["totals"]["instances"] == 12

    for sp in ("train", "val", "test"):
        for lbl in _read_labels(out / "labels" / sp).values():
            for line in lbl:
                toks = line.split()
                assert int(toks[0]) in (0, 1)
                coords = toks[1:]
                assert len(coords) >= 6 and len(coords) % 2 == 0
                assert all(0.0 <= float(c) <= 1.0 for c in coords)

    rows = list((out / "manifest.csv").read_text(encoding="utf-8").splitlines())
    assert len(rows) == 33  # 表头 + 32


def test_good_images_get_empty_labels(mvtec_tree, tmp_path):
    out = tmp_path / "yolo"
    convert_dataset(mvtec_tree, out, workers=1)
    labels = {}
    for sp in ("train", "val", "test"):
        labels.update(_read_labels(out / "labels" / sp))
    empties = [p for p, lines in labels.items() if not lines]
    # 20 张 good（12 train/good + 8 test/good）都是空标签背景图
    assert len(empties) == 20
    assert all("__good__" in p.name for p in empties)


def test_convert_unified_mode(mvtec_tree, tmp_path):
    out = tmp_path / "yolo_unified"
    rep = convert_dataset(mvtec_tree, out, classes_mode="unified", workers=1)
    cfg = yaml.safe_load(rep.data_yaml.read_text(encoding="utf-8"))
    assert cfg["names"] == {0: "defect"}
    for sp in ("train", "val", "test"):
        for lines in _read_labels(out / "labels" / sp).values():
            for line in lines:
                assert line.startswith("0 ")


def test_convert_dry_run_writes_nothing(mvtec_tree, tmp_path):
    out = tmp_path / "dry"
    rep = convert_dataset(mvtec_tree, out, write_files=False)
    assert not out.exists()
    assert rep.n_written == 0
    assert rep.data_yaml is None
    assert rep.stats["totals"]["images_scanned"] == 33


def test_convert_hardlink_by_default(mvtec_tree, tmp_path):
    out = tmp_path / "yolo_link"
    rep = convert_dataset(mvtec_tree, out, link="hardlink", workers=1)
    assert rep.link_mode_used == "hardlink"
    one = sorted((out / "images").rglob("*.png"))[0]
    assert one.stat().st_size > 0
    with (out / "manifest.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 32
    srcs = [Path(r["src_image"]) for r in rows]
    assert all(s.is_file() for s in srcs)
    assert {r["split"] for r in rows} == {"train", "val", "test"}


def test_convert_copy_mode(mvtec_tree, tmp_path):
    out = tmp_path / "yolo_copy"
    rep = convert_dataset(mvtec_tree, out, link="copy", workers=1)
    assert rep.link_mode_used == "copy"
    assert len(list((out / "images").rglob("*.png"))) == 32


def test_convert_reproducible_stats(mvtec_tree, tmp_path):
    a = convert_dataset(mvtec_tree, tmp_path / "a", seed=3, workers=1)
    b = convert_dataset(mvtec_tree, tmp_path / "b", seed=3, workers=1)
    drop = lambda d: {k: v for k, v in d.items() if k != "source_root"}  # noqa: E731
    assert json.dumps(drop(a.stats), sort_keys=True) == json.dumps(drop(b.stats), sort_keys=True)


def test_convert_mask_size_mismatch_is_repaired(tmp_path):
    tree = build_mvtec_tree(tmp_path / "t", spec={"bottle": {"broken_large": 3}}, size=(64, 48))
    bad = sorted((tree / "bottle" / "ground_truth" / "broken_large").glob("*_mask.png"))[0]
    m = cv2.imread(str(bad), cv2.IMREAD_UNCHANGED)
    cv2.imwrite(str(bad), cv2.resize(m, (32, 24), interpolation=cv2.INTER_NEAREST))
    rep = convert_dataset(tree, tmp_path / "y", workers=1)
    assert any("尺寸" in w and "不一致" in w for w in rep.warnings)
    assert rep.stats["totals"]["images_written"] == 13  # 6 train/good + 4 test/good + 3 缺陷
    assert rep.stats["totals"]["instances"] == 3


def test_convert_subset_categories(mvtec_tree, tmp_path):
    out = tmp_path / "yolo_sub"
    rep = convert_dataset(mvtec_tree, out, categories=["screw"], workers=1)
    assert rep.stats["classes"]["names"] == ["screw"]
    # screw: 6 train/good + 4 test/good + 3 thread（speck 被丢弃）
    assert rep.stats["totals"]["images_written"] == 13


def test_speck_warning_emitted(mvtec_tree, tmp_path):
    rep = convert_dataset(mvtec_tree, tmp_path / "y", workers=1)
    assert any("未产出多边形" in w for w in rep.warnings)
    assert any("丢弃" in w and "speck" in w for w in rep.warnings)


def test_keep_empty_defects_keeps_them_as_background(mvtec_tree, tmp_path):
    out = tmp_path / "yolo_keep"
    rep = convert_dataset(mvtec_tree, out, workers=1, keep_empty_defects=True)
    assert rep.n_written == 33
    empties = 0
    for sp in ("train", "val", "test"):
        empties += sum(1 for v in _read_labels(out / "labels" / sp).values() if not v)
    assert empties == 21  # 20 张 good + 1 张 speck


def test_module_does_not_import_torch():
    code = (
        "import sys, induvision.mvtec; "
        "assert 'torch' not in sys.modules, 'mvtec 不应引入 torch'; "
        "print('ok')"
    )
    r = subprocess.run([sys.executable, "-c", code], cwd=str(REPO_ROOT),
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "ok" in r.stdout


def test_env_check_tool_exists():
    assert (REPO_ROOT / "tools" / "env_check.py").is_file()
    assert (REPO_ROOT / "tools" / "download_mvtec.py").is_file()
    assert (REPO_ROOT / "tools" / "convert_mvtec_to_yolo.py").is_file()