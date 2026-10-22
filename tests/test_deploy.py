"""induvision/deploy.py 的单测：纯 numpy / opencv，CI 里不装 torch 也能全跑。

这里盯的是"两个后端吃到的输入是否逐字节相同"这件事本身：letterbox 的 ratio 取整、
pad 分配、BGR→RGB、/255、CHW 顺序，任何一项写歪，后面量出来的 .pt vs .onnx 差值
就全是前处理的差，W3 的结论会跟着一起歪。
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from induvision import deploy


def _img(h: int, w: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 255, (h, w, 3), dtype=np.uint8)


# ---------------------------------------------------------------- letterbox

def test_pad_color_is_ultralytics_default():
    # 改了这个常量，本仓库的对齐结果就和 ultralytics 推理口径不再可比
    assert deploy.PAD_COLOR == (114, 114, 114)


def test_letterbox_square_image_needs_no_pad():
    lb = deploy.letterbox(_img(900, 900), 640)
    assert lb.image.shape == (640, 640, 3)
    assert lb.new_unpad == (640, 640)
    assert lb.top == 0 and lb.left == 0
    assert lb.ratio == pytest.approx(640 / 900)


def test_letterbox_pads_to_square_and_splits_pad_asymmetrically():
    # 900x600（w x h）-> 缩放后 640x427，高方向余 213 像素：上 106 / 下 107。
    # 自己写 dh//2 会分成 106/106 或 107/107，和 ultralytics 差 1 像素。
    lb = deploy.letterbox(_img(600, 900), 640)
    assert lb.new_unpad == (640, 427)
    assert (lb.top, lb.image.shape[0] - lb.top - 427) == (106, 107)
    assert lb.top + (640 - lb.top - lb.new_unpad[1]) == 640 - lb.new_unpad[1]
    assert lb.left == 0


def test_letterbox_pad_rows_are_filled_with_pad_color():
    lb = deploy.letterbox(_img(600, 900), 640)
    assert tuple(lb.image[0, 0]) == deploy.PAD_COLOR
    assert tuple(lb.image[-1, -1]) == deploy.PAD_COLOR
    # 内容区正中间不该是填充色（随机图撞上 114,114,114 的概率忽略不计）
    inner = lb.image[lb.top:lb.top + lb.new_unpad[1], lb.left:lb.left + lb.new_unpad[0]]
    assert inner.shape == (427, 640, 3)
    assert not np.array_equal(inner[0, 0], np.array(deploy.PAD_COLOR, dtype=np.uint8))


def test_letterbox_content_matches_plain_resize():
    src = _img(600, 900, seed=3)
    lb = deploy.letterbox(src, 640)
    expect = cv2.resize(src, (640, 427), interpolation=cv2.INTER_LINEAR)
    inner = lb.image[lb.top:lb.top + 427, lb.left:lb.left + 640]
    assert np.array_equal(inner, expect)


def test_letterbox_scaleup_false_never_enlarges():
    src = _img(100, 80)
    lb = deploy.letterbox(src, 640, scaleup=False)
    assert lb.ratio == 1.0
    assert lb.new_unpad == (80, 100)
    assert lb.image.shape == (640, 640, 3)


def test_letterbox_scaleup_true_does_enlarge():
    lb = deploy.letterbox(_img(100, 80), 640)
    assert lb.ratio == pytest.approx(640 / 100)
    assert lb.new_unpad == (512, 640)


def test_letterbox_records_orig_shape_for_inverse_mapping():
    lb = deploy.letterbox(_img(600, 900), 640)
    assert lb.orig_shape == (600, 900)
    # 后处理映回原图用的就是这个公式
    x_orig = (320.0 - lb.dw) / lb.ratio
    assert 0 <= x_orig <= 900


# --------------------------------------------------------------- preprocess

def test_preprocess_shape_dtype_range_and_order():
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    img[..., 0] = 10  # B
    img[..., 1] = 20  # G
    img[..., 2] = 30  # R
    x, lb = deploy.preprocess(img, 640)
    assert x.shape == (3, 640, 640)
    assert x.dtype == np.float32
    assert x.flags["C_CONTIGUOUS"]
    assert float(x.min()) == pytest.approx(10 / 255.0)
    assert float(x.max()) == pytest.approx(30 / 255.0)
    # 通道序必须是 RGB：切错了模型照样出数，但精度会掉得莫名其妙
    assert np.allclose(x[0], 30 / 255.0)
    assert np.allclose(x[1], 20 / 255.0)
    assert np.allclose(x[2], 10 / 255.0)
    assert lb.orig_shape == (640, 640)


def test_preprocess_pad_area_is_pad_color_over_255():
    x, lb = deploy.preprocess(_img(600, 900), 640)
    assert np.allclose(x[:, 0, 0], 114 / 255.0)
    assert lb.top == 106


def test_preprocess_has_no_batch_dim():
    x, _ = deploy.preprocess(_img(480, 640), 640)
    # batch 维留给调用方加，这样单图对齐和多图压测共用同一份前处理
    assert x.ndim == 3


# ----------------------------------------------------------- compare_tensors

def test_compare_identical_tensors_is_zero_diff():
    a = np.arange(12, dtype=np.float32).reshape(3, 4)
    d = deploy.compare_tensors(a, a.copy())
    assert d["shape_match"] is True
    assert d["max_abs"] == 0.0 and d["mean_abs"] == 0.0 and d["rel_l2"] == 0.0
    assert d["cosine"] == pytest.approx(1.0)
    assert d["shape_a"] == [3, 4]


def test_compare_shape_mismatch_reports_shapes_and_no_metrics():
    d = deploy.compare_tensors(np.zeros((2, 3)), np.zeros((3, 2)))
    assert d["shape_match"] is False
    assert d["shape_a"] == [2, 3] and d["shape_b"] == [3, 2]
    assert d["max_abs"] is None and d["cosine"] is None


def test_compare_scaled_vector_keeps_cosine_but_not_rel_l2():
    a = np.array([1.0, 2.0])
    d = deploy.compare_tensors(a, a * 2)
    assert d["cosine"] == pytest.approx(1.0)
    # 方向一样但幅度翻倍：rel_l2 = 1.0。这就是"只看 cosine 会漏掉量化整体偏置"的例子
    assert d["rel_l2"] == pytest.approx(1.0)


def test_compare_orthogonal_vectors_cosine_zero():
    d = deploy.compare_tensors(np.array([1.0, 0.0]), np.array([0.0, 1.0]))
    assert d["cosine"] == pytest.approx(0.0)


def test_compare_all_zero_tensors_is_defined_not_nan():
    d = deploy.compare_tensors(np.zeros(4), np.zeros(4))
    assert d["cosine"] == 1.0 and d["rel_l2"] == 0.0


def test_compare_is_json_serializable():
    import json

    d = deploy.compare_tensors(np.zeros((1, 51, 8400), dtype=np.float32),
                               np.zeros((1, 51, 8400), dtype=np.float32))
    json.dumps(d)  # 不抛异常即通过：np.float64 会炸 json


# ------------------------------------------------------------- latency_stats

def test_latency_stats_basic_distribution():
    s = deploy.latency_stats([10, 20, 30])
    assert s["count"] == 3
    assert s["mean_ms"] == 20.0 and s["p50_ms"] == 20.0
    assert s["min_ms"] == 10.0 and s["max_ms"] == 30.0
    assert s["fps"] == 50.0


def test_latency_stats_empty_is_all_none():
    s = deploy.latency_stats([])
    assert s["count"] == 0 and s["mean_ms"] is None and s["fps"] is None


def test_latency_stats_drops_non_finite():
    s = deploy.latency_stats([10.0, float("nan"), float("inf"), 30.0])
    assert s["count"] == 2 and s["mean_ms"] == 20.0


def test_latency_stats_reports_tail_percentiles():
    # 100 帧里 5 帧毛刺：均值看不出来，P95 立刻暴露
    samples = [10.0] * 95 + [200.0] * 5
    s = deploy.latency_stats(samples)
    assert s["mean_ms"] == pytest.approx(19.5)
    assert s["p50_ms"] == 10.0
    assert s["p99_ms"] == 200.0  # 均值 19.5 ms 看着正常，P99 才暴露尾巴上的 20 倍毛刺


# -------------------------------------------------------------- sample_paths

def test_sample_paths_covers_head_and_tail_evenly():
    paths = [Path(f"f{i:03d}.png") for i in range(20)]
    got = deploy.sample_paths(paths, 5)
    assert len(got) == 5 == len(set(got))
    assert got[0] == paths[0] and got[-1] == paths[-1]


def test_sample_paths_is_deterministic():
    paths = [Path(f"f{i}.png") for i in range(37)]
    assert [str(p) for p in deploy.sample_paths(paths, 9)] == [str(p) for p in deploy.sample_paths(paths, 9)]


def test_sample_paths_edge_cases():
    paths = [Path(f"f{i}.png") for i in range(4)]
    assert deploy.sample_paths(paths, 0) == []
    assert deploy.sample_paths(paths, 1) == [paths[0]]
    assert deploy.sample_paths(paths, 10) == paths
    assert deploy.sample_paths([], 5) == []


# ---------------------------------------------------------- read_split_images

def _write_ds(root: Path, *, test_value="images/test", path_value=None) -> Path:
    (root / "images" / "test").mkdir(parents=True, exist_ok=True)
    for name in ("a.png", "b.PNG", "c.txt", "d.jpg"):
        p = root / "images" / "test" / name
        if name.endswith(".txt"):
            p.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        else:
            assert cv2.imwrite(str(p), _img(16, 16))
    d = {"path": str(root) if path_value is None else path_value,
         "train": "images/train", "val": "images/val", "test": test_value,
         "names": {0: "bottle"}}
    (root / "data.yaml").write_text(yaml.safe_dump(d, allow_unicode=True), encoding="utf-8")
    return root / "data.yaml"


def test_read_split_images_returns_only_images_sorted(tmp_path: Path):
    y = _write_ds(tmp_path / "ds")
    got = deploy.read_split_images(y, "test")
    assert [p.name for p in got] == ["a.png", "b.PNG", "d.jpg"]


def test_read_split_images_accepts_list_of_entries(tmp_path: Path):
    root = tmp_path / "ds"
    y = _write_ds(root, test_value=["images/test", "images/test"])
    got = deploy.read_split_images(y, "test")
    assert len(got) == 3  # 重复条目去重


def test_read_split_images_accepts_single_file(tmp_path: Path):
    root = tmp_path / "ds"
    y = _write_ds(root)
    one = root / "images" / "test" / "a.png"
    y.write_text(yaml.safe_dump({"path": str(root), "test": str(one)}), encoding="utf-8")
    assert [p.name for p in deploy.read_split_images(y, "test")] == ["a.png"]


def test_read_split_images_resolves_relative_path_against_yaml_dir(tmp_path: Path):
    root = tmp_path / "ds"
    y = _write_ds(root, path_value=".")
    assert len(deploy.read_split_images(y, "test")) == 3


def test_read_split_images_missing_key_raises(tmp_path: Path):
    y = _write_ds(tmp_path / "ds")
    with pytest.raises(KeyError):
        deploy.read_split_images(y, "nope")


def test_read_split_images_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        deploy.read_split_images(tmp_path / "nope.yaml", "test")


def test_read_split_images_empty_dir_returns_empty_list(tmp_path: Path):
    root = tmp_path / "ds"
    y = _write_ds(root)
    (root / "images" / "val").mkdir(parents=True)
    assert deploy.read_split_images(y, "val") == []


# ------------------------------------------------------------- read_image_bgr

def test_read_image_bgr_reads_real_png(tmp_path: Path):
    p = tmp_path / "x.png"
    src = _img(24, 32, seed=7)
    assert cv2.imwrite(str(p), src)
    got = deploy.read_image_bgr(p)
    assert got is not None and got.shape == (24, 32, 3) and got.dtype == np.uint8
    assert np.array_equal(got, src)


def test_read_image_bgr_missing_file_returns_none(tmp_path: Path):
    assert deploy.read_image_bgr(tmp_path / "nope.png") is None


def test_read_image_bgr_falls_back_to_imread(tmp_path: Path, monkeypatch):
    # 网络盘 / 超长路径这类 np.fromfile 啃不动的情况，得有人接住
    p = tmp_path / "y.png"
    assert cv2.imwrite(str(p), _img(8, 8))
    monkeypatch.setattr(deploy.cv2, "imdecode", lambda *a, **k: None)
    got = deploy.read_image_bgr(p)
    assert got is not None and got.shape == (8, 8, 3)


def test_read_image_bgr_survives_fromfile_oserror(tmp_path: Path, monkeypatch):
    p = tmp_path / "z.png"
    assert cv2.imwrite(str(p), _img(8, 8))

    def _boom(*a, **k):
        raise OSError("network path")

    monkeypatch.setattr(deploy.np, "fromfile", _boom)
    got = deploy.read_image_bgr(p)
    assert got is not None and got.shape == (8, 8, 3)