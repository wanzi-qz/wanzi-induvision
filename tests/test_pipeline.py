# -*- coding: utf-8 -*-
"""``induvision.pipeline`` 的单测：环形队列、NMS、掩码解码、后处理、流水线本体。

全部纯 CPU、不碰 torch/tensorrt/GPU —— CI 那台 ubuntu runner 没装 GPU 依赖，
这也是 pipeline.py 只依赖 numpy/opencv/标准库的原因（pinned 那部分在 test_pinned.py）。
"""
from __future__ import annotations

import threading
import time

import cv2
import numpy as np
import pytest

from induvision.deploy import LetterboxResult, letterbox, preprocess
from induvision.pipeline import (
    DEFAULT_CONF_THR,
    DEFAULT_IOU_THR,
    DEFAULT_MASK_THR,
    Detection,
    Pipeline,
    PipelineError,
    RingBuffer,
    _as_predictions,
    batched_nms,
    cxcywh_to_xyxy,
    detections_to_records,
    iou_one_vs_many,
    mask_iou,
    masks_from_proto,
    match_detections,
    nms,
    postprocess,
    run_serial,
    scale_boxes_to_orig,
    sigmoid,
    speedup_report,
)


# --------------------------------------------------------------------------
# 工具
# --------------------------------------------------------------------------

def _lb(orig_h: int = 480, orig_w: int = 640, imgsz: int = 640) -> LetterboxResult:
    return letterbox(np.zeros((orig_h, orig_w, 3), np.uint8), imgsz)


def _det(score=0.9, cls=0, box=(0.0, 0.0, 10.0, 10.0), mask=None) -> Detection:
    return Detection(score=score, cls=cls, box=np.asarray(box, dtype=np.float64),
                     box640=np.asarray(box, dtype=np.float64) * 0.5, mask=mask)


def _mk_out0(n_attrs: int, n_anchors: int, peaks=()) -> np.ndarray:
    """造一个 (1, n_attrs, n_anchors) 的 output0。peaks = [(anchor, cls, score_logit, cx,cy,w,h)]"""
    a = np.zeros((n_attrs, n_anchors), dtype=np.float32)
    for anchor, cls, logit, cx, cy, w, h in peaks:
        a[0:4, anchor] = (cx, cy, w, h)
        a[4 + cls, anchor] = logit
    return a[None]


# --------------------------------------------------------------------------
# RingBuffer
# --------------------------------------------------------------------------

class TestRingBuffer:
    def test_capacity_must_be_positive(self):
        with pytest.raises(ValueError, match="capacity"):
            RingBuffer(0)
        with pytest.raises(ValueError, match="capacity"):
            RingBuffer(-3)

    def test_fifo_order(self):
        rb = RingBuffer(4)
        for i in range(4):
            assert rb.put(i) is True
        assert [rb.get() for _ in range(4)] == [0, 1, 2, 3]

    def test_len_tracks_count(self):
        rb = RingBuffer(3)
        assert len(rb) == 0
        rb.put("a")
        assert len(rb) == 1
        rb.get()
        assert len(rb) == 0

    def test_wraparound_preserves_order(self):
        rb = RingBuffer(2)
        got = []
        for i in range(6):
            rb.put(i)
            if i % 2 == 1:
                got.append(rb.get())
                got.append(rb.get() if len(rb) else None)
        rb.put(100)
        rb.put(101)
        got = [rb.get(), rb.get()]
        assert got == [100, 101]

    def test_get_timeout_returns_none_when_empty(self):
        rb = RingBuffer(2)
        t0 = time.perf_counter()
        assert rb.get(timeout=0.05) is None
        assert time.perf_counter() - t0 >= 0.04

    def test_get_zero_timeout_returns_none_immediately(self):
        rb = RingBuffer(2)
        assert rb.get(timeout=0) is None

    def test_get_blocks_until_producer_puts(self):
        rb = RingBuffer(2)
        out = []

        def producer():
            time.sleep(0.05)
            rb.put("late")

        t = threading.Thread(target=producer, daemon=True)
        t.start()
        assert rb.get(timeout=2.0) == "late"
        t.join(2.0)
        assert out == []

    def test_put_timeout_returns_false_when_full(self):
        rb = RingBuffer(1)
        assert rb.put(1) is True
        t0 = time.perf_counter()
        assert rb.put(2, timeout=0.05) is False
        assert time.perf_counter() - t0 >= 0.04
        assert len(rb) == 1

    def test_put_succeeds_once_consumer_drains(self):
        rb = RingBuffer(1)
        rb.put(1)

        def consumer():
            time.sleep(0.05)
            rb.get()

        t = threading.Thread(target=consumer, daemon=True)
        t.start()
        assert rb.put(2, timeout=2.0) is True
        t.join(2.0)
        assert rb.get(timeout=0.5) == 2

    def test_put_after_close_raises(self):
        rb = RingBuffer(2, name="q")
        rb.close()
        with pytest.raises(RuntimeError, match="q"):
            rb.put(1)

    def test_close_is_idempotent(self):
        rb = RingBuffer(2)
        rb.close()
        rb.close()
        assert rb.closed is True

    def test_close_unblocks_waiting_get(self):
        rb = RingBuffer(2)
        got = []

        def consumer():
            got.append(rb.get(timeout=5.0))

        t = threading.Thread(target=consumer, daemon=True)
        t.start()
        time.sleep(0.05)
        rb.close()
        t.join(5.0)
        assert got == [None]

    def test_close_lets_pending_items_drain(self):
        rb = RingBuffer(4)
        rb.put("a")
        rb.put("b")
        rb.close()
        assert rb.get() == "a"
        assert rb.get() == "b"
        assert rb.get() is None  # 排空之后才是结束信号

    def test_stats_fields(self):
        rb = RingBuffer(3, name="qq")
        rb.put(1)
        rb.get()
        st = rb.stats()
        assert st["name"] == "qq" and st["capacity"] == 3
        assert st["puts"] == 1 and st["gets"] == 1
        assert st["depth"] == 0 and st["max_depth"] == 1
        assert st["closed"] is False

    def test_max_depth_records_peak(self):
        rb = RingBuffer(4)
        for i in range(3):
            rb.put(i)
        rb.get()
        assert rb.stats()["max_depth"] == 3

    def test_overflow_waits_counts_blocked_puts(self):
        rb = RingBuffer(1)
        rb.put(1)
        assert rb.put(2, timeout=0.02) is False
        assert rb.stats()["overflow_waits"] >= 1

    def test_underflow_waits_counts_blocked_gets(self):
        rb = RingBuffer(1)
        assert rb.get(timeout=0.02) is None
        assert rb.stats()["underflow_waits"] >= 1

    def test_slots_are_released_after_get(self):
        """取走之后槽位置 None，不然队列会一直持有大对象（掩码 0.8 MiB 一张）。"""
        rb = RingBuffer(2)
        rb.put(object())
        rb.get()
        assert all(s is None for s in rb._slots)

    def test_many_producers_one_consumer_no_loss(self):
        rb = RingBuffer(4)
        n_each, n_prod = 50, 4

        def producer(k):
            for i in range(n_each):
                rb.put(k * 1000 + i)

        ts = [threading.Thread(target=producer, args=(k,), daemon=True) for k in range(n_prod)]
        for t in ts:
            t.start()
        got = []
        while len(got) < n_each * n_prod:
            v = rb.get(timeout=5.0)
            assert v is not None
            got.append(v)
        for t in ts:
            t.join(5.0)
        assert sorted(got) == sorted(k * 1000 + i for k in range(n_prod) for i in range(n_each))


# --------------------------------------------------------------------------
# sigmoid / 坐标变换 / IoU
# --------------------------------------------------------------------------

class TestSigmoid:
    def test_zero_is_half(self):
        assert float(sigmoid(np.asarray([0.0], np.float32))[0]) == pytest.approx(0.5)

    def test_no_overflow_on_extremes(self):
        x = np.asarray([-1e9, -1000.0, 1000.0, 1e9], np.float32)
        with np.errstate(over="raise", invalid="raise"):
            y = sigmoid(x)
        assert np.all(np.isfinite(y))
        assert y[0] == pytest.approx(0.0, abs=1e-6)
        assert y[-1] == pytest.approx(1.0, abs=1e-6)

    def test_output_dtype_float32(self):
        assert sigmoid(np.asarray([1.0], np.float64)).dtype == np.float32

    def test_monotonic(self):
        x = np.linspace(-10, 10, 21).astype(np.float32)
        y = sigmoid(x)
        assert np.all(np.diff(y) >= 0)


class TestCxcywh:
    def test_known_conversion(self):
        out = cxcywh_to_xyxy(np.asarray([[10.0, 20.0, 4.0, 6.0]]))
        assert out.tolist() == [[8.0, 17.0, 12.0, 23.0]]

    def test_accepts_flat_input(self):
        assert cxcywh_to_xyxy(np.asarray([10.0, 20.0, 4.0, 6.0])).shape == (1, 4)

    def test_does_not_mutate_input(self):
        b = np.asarray([[10.0, 20.0, 4.0, 6.0]])
        cxcywh_to_xyxy(b)
        assert b.tolist() == [[10.0, 20.0, 4.0, 6.0]]

    def test_zero_size_box_degenerates_to_point(self):
        out = cxcywh_to_xyxy(np.asarray([[5.0, 5.0, 0.0, 0.0]]))[0]
        assert out[0] == out[2] and out[1] == out[3]

    def test_batch(self):
        out = cxcywh_to_xyxy(np.asarray([[0.0, 0.0, 2.0, 2.0], [10.0, 10.0, 2.0, 2.0]]))
        assert out.shape == (2, 4)
        assert out[1].tolist() == [9.0, 9.0, 11.0, 11.0]


class TestIou:
    def test_identical_is_one(self):
        assert float(iou_one_vs_many([0, 0, 10, 10], np.asarray([[0, 0, 10, 10]]))[0]) == 1.0

    def test_disjoint_is_zero(self):
        assert float(iou_one_vs_many([0, 0, 1, 1], np.asarray([[5, 5, 6, 6]]))[0]) == 0.0

    def test_half_overlap(self):
        # 交 5x10=50，并 100+100-50=150
        assert float(iou_one_vs_many([0, 0, 10, 10], np.asarray([[5, 0, 15, 10]]))[0]) \
            == pytest.approx(1 / 3)

    def test_containment(self):
        # 交 25，并 100
        assert float(iou_one_vs_many([0, 0, 10, 10], np.asarray([[0, 0, 5, 5]]))[0]) \
            == pytest.approx(0.25)

    def test_empty_boxes_gives_empty_result(self):
        out = iou_one_vs_many([0, 0, 1, 1], np.zeros((0, 4)))
        assert out.shape == (0,)

    def test_zero_area_box_is_zero_not_nan(self):
        out = iou_one_vs_many([0, 0, 0, 0], np.asarray([[0, 0, 0, 0]]))
        assert out[0] == 0.0 and np.isfinite(out[0])

    def test_touching_edges_are_zero(self):
        assert float(iou_one_vs_many([0, 0, 10, 10], np.asarray([[10, 0, 20, 10]]))[0]) == 0.0

    def test_vectorized_matches_loop(self):
        rng = np.random.default_rng(0)
        b = rng.uniform(0, 50, 4)
        bs = rng.uniform(0, 50, (7, 4))
        bs[:, 2:] += bs[:, :2]
        vec = iou_one_vs_many(b, bs)
        loop = [float(iou_one_vs_many(b, bs[i:i + 1])[0]) for i in range(7)]
        assert vec.tolist() == pytest.approx(loop)


# --------------------------------------------------------------------------
# NMS
# --------------------------------------------------------------------------

class TestNms:
    def test_bad_iou_thr(self):
        for thr in (0.0, -0.1, 1.5):
            with pytest.raises(ValueError, match="iou_thr"):
                nms(np.zeros((1, 4)), np.zeros(1), thr)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="数量不一致"):
            nms(np.zeros((2, 4)), np.zeros(3))

    def test_empty_input(self):
        assert nms(np.zeros((0, 4)), np.zeros(0)) == []

    def test_single_box_kept(self):
        assert nms(np.asarray([[0.0, 0, 10, 10]]), np.asarray([0.5])) == [0]

    def test_duplicate_keeps_higher_score(self):
        b = np.asarray([[0.0, 0, 10, 10], [0.4, 0, 10.4, 10]])
        assert nms(b, np.asarray([0.3, 0.9]), 0.5) == [1]

    def test_disjoint_boxes_both_kept(self):
        b = np.asarray([[0.0, 0, 10, 10], [50.0, 50, 60, 60]])
        assert sorted(nms(b, np.asarray([0.3, 0.9]), 0.5)) == [0, 1]

    def test_iou_equal_to_threshold_is_not_suppressed(self):
        """与 torchvision 同语义：IoU **大于** 阈值才抑制，等于阈值两个都留。"""
        b = np.asarray([[0.0, 0, 10, 10], [5.0, 0, 15, 10]])  # IoU = 1/3
        keep = nms(b, np.asarray([0.9, 0.8]), 1 / 3)
        assert sorted(keep) == [0, 1]

    def test_iou_just_above_threshold_suppresses(self):
        b = np.asarray([[0.0, 0, 10, 10], [5.0, 0, 15, 10]])
        assert nms(b, np.asarray([0.9, 0.8]), 0.33) == [0]

    def test_result_sorted_by_score_desc(self):
        b = np.asarray([[0.0, 0, 1, 1], [10.0, 10, 11, 11], [20.0, 20, 21, 21]])
        keep = nms(b, np.asarray([0.2, 0.9, 0.5]), 0.5)
        assert keep == [1, 2, 0]

    def test_top_k_limits_candidates(self):
        b = np.asarray([[0.0, 0, 1, 1], [10.0, 10, 11, 11], [20.0, 20, 21, 21]])
        assert nms(b, np.asarray([0.2, 0.9, 0.5]), 0.5, top_k=2) == [1, 2]

    def test_top_k_must_be_positive(self):
        with pytest.raises(ValueError, match="top_k"):
            nms(np.zeros((1, 4)), np.zeros(1), 0.5, top_k=0)

    def test_chain_suppression(self):
        # 一串逐步右移的重叠框，最高分那个把邻近的都吃掉
        b = np.asarray([[float(i), 0.0, float(i) + 10, 10] for i in range(4)])
        keep = nms(b, np.asarray([0.9, 0.8, 0.7, 0.6]), 0.3)
        assert keep[0] == 0 and len(keep) < 4

    def test_equal_scores_stable(self):
        b = np.asarray([[0.0, 0, 1, 1], [50.0, 50, 51, 51]])
        assert nms(b, np.asarray([0.5, 0.5]), 0.5) == [0, 1]


class TestBatchedNms:
    def test_different_classes_do_not_suppress(self):
        b = np.asarray([[0.0, 0, 10, 10], [0.2, 0, 10.2, 10]])
        keep = batched_nms(b, np.asarray([0.9, 0.8]), np.asarray([0, 1]), 0.5)
        assert sorted(keep) == [0, 1]

    def test_same_class_suppresses(self):
        b = np.asarray([[0.0, 0, 10, 10], [0.2, 0, 10.2, 10]])
        assert batched_nms(b, np.asarray([0.9, 0.8]), np.asarray([0, 0]), 0.5) == [0]

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="数量不一致"):
            batched_nms(np.zeros((2, 4)), np.zeros(2), np.zeros(3))

    def test_global_score_order(self):
        b = np.asarray([[0.0, 0, 1, 1], [10.0, 10, 11, 11], [20.0, 20, 21, 21]])
        keep = batched_nms(b, np.asarray([0.4, 0.95, 0.7]), np.asarray([0, 1, 2]), 0.5)
        assert keep == [1, 2, 0]

    def test_empty(self):
        assert batched_nms(np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)) == []

    def test_matches_per_class_nms_union(self):
        rng = np.random.default_rng(1)
        b = rng.uniform(0, 30, (12, 4))
        b[:, 2:] += b[:, :2] + 5
        s = rng.uniform(0.1, 0.9, 12)
        lab = rng.integers(0, 3, 12)
        expect = set()
        for c in np.unique(lab):
            sel = np.flatnonzero(lab == c)
            expect.update(int(sel[i]) for i in nms(b[sel], s[sel], 0.5))
        assert set(batched_nms(b, s, lab, 0.5)) == expect


# --------------------------------------------------------------------------
# 坐标映射：letterbox 640 -> 原图
# --------------------------------------------------------------------------

class TestScaleBoxes:
    def test_square_no_pad_identity(self):
        lb = _lb(640, 640)
        assert lb.ratio == 1.0 and lb.dw == 0.0 and lb.dh == 0.0
        b = np.asarray([[10.0, 20.0, 100.0, 200.0]])
        assert np.allclose(scale_boxes_to_orig(b, lb), b)

    def test_uses_ratio_and_pad(self):
        lb = _lb(480, 1280)                       # ratio 0.5, dh 200, dw 0
        assert lb.ratio == pytest.approx(0.5) and lb.dh == pytest.approx(200.0)
        b = np.asarray([[0.0, lb.dh, 640.0, lb.dh + 240.0]])
        assert np.allclose(scale_boxes_to_orig(b, lb), np.asarray([[0.0, 0.0, 1280.0, 480.0]]))

    def test_subtracts_float_pad_not_int_top(self):
        """余量为奇数时 lb.dh != lb.top；ultralytics 口径减的是前者，必须一致。"""
        lb = _lb(480, 641)                        # dh=80.5 而 top=80
        assert lb.dh != lb.top
        got = scale_boxes_to_orig(np.asarray([[100.0, lb.dh, 200.0, lb.dh + 10.0]]), lb)[0]
        assert got[1] == pytest.approx(0.0, abs=1e-9)
        assert abs(got[1] - (lb.dh - lb.top) / lb.ratio) > 1e-6

    def test_clips_to_orig(self):
        lb = _lb(480, 640)
        got = scale_boxes_to_orig(np.asarray([[-100.0, -100.0, 9999.0, 9999.0]]), lb)[0]
        assert list(got) == [0.0, 0.0, 640.0, 480.0]

    def test_empty(self):
        assert scale_boxes_to_orig(np.zeros((0, 4)), _lb(480, 640)).shape == (0, 4)

    def test_flat_input_reshaped(self):
        assert scale_boxes_to_orig(np.asarray([1.0, 2.0, 3.0, 4.0]), _lb(640, 640)).shape == (1, 4)

    def test_does_not_mutate_input(self):
        b = np.asarray([[10.0, 210.0, 100.0, 300.0]])
        snap = b.copy()
        scale_boxes_to_orig(b, _lb(480, 1280))
        assert np.array_equal(b, snap)


# --------------------------------------------------------------------------
# 掩码解码
# --------------------------------------------------------------------------

def _ref_masks(proto: np.ndarray, coeffs: np.ndarray, boxes640: np.ndarray,
               lb, mask_thr: float = 0.5) -> list:
    """参照实现（ultralytics 口径）：整张 proto upsample 到 letterbox 尺寸 -> 按框裁剪
    -> 仿射映射回原图 -> 阈值化。慢 16 倍，但坐标语义一目了然，专门用来校核快实现。

    注意两套坐标系：裁剪下标是**网格坐标 x scale**（letterbox 像素），仿射矩阵把
    letterbox 像素映射回原图像素。第一版把网格下标直接拿去切 640 画布，掩码整个跑出
    画面，IoU 全 0 —— 参照实现自己也要验，不然会拿它去"证明"对的实现是错的。
    """
    c, mh, mw = proto.shape
    lin = int(lb.image.shape[0])
    up = np.empty((c, lin, lin), np.float32)
    for i in range(c):
        up[i] = cv2.resize(proto[i].astype(np.float32), (lin, lin),
                           interpolation=cv2.INTER_LINEAR)
    oh, ow = lb.orig_shape
    inv = 1.0 / float(lb.ratio)
    M = np.asarray([[inv, 0.0, -float(lb.dw) * inv],
                    [0.0, inv, -float(lb.dh) * inv]], np.float32)
    scale = float(lin) / mw
    out = []
    for k in range(boxes640.shape[0]):
        m = sigmoid(np.tensordot(coeffs[k].astype(np.float32), up, axes=1))
        x1, y1, x2, y2 = boxes640[k] / scale                       # -> 网格坐标
        gx = [min(max(int(np.floor(v + 0.5)), 0), lim)
              for v, lim in ((x1, mw), (y1, mh), (x2, mw), (y2, mh))]
        px1, py1, px2, py2 = [int(round(v * scale)) for v in gx]   # -> letterbox 像素
        cx1, cx2 = max(0, min(px1, lin)), max(0, min(px2, lin))
        cy1, cy2 = max(0, min(py1, lin)), max(0, min(py2, lin))
        canvas = np.zeros((lin, lin), np.float32)
        if cx2 > cx1 and cy2 > cy1:
            canvas[cy1:cy2, cx1:cx2] = m[cy1:cy2, cx1:cx2]
        warped = cv2.warpAffine(canvas, M, (ow, oh), flags=cv2.INTER_LINEAR, borderValue=0.0)
        out.append(warped >= mask_thr)
    return out


def _smooth_proto(c: int = 32, mh: int = 160, mw: int = 160, seed: int = 7) -> np.ndarray:
    """低频随机场：白噪声会让掩码碎成一格一格的椒盐，比对时全是插值噪声，看不出坐标错。"""
    rng = np.random.default_rng(seed)
    small = rng.normal(0.0, 1.0, (c, 8, 8)).astype(np.float32)
    p = np.empty((c, mh, mw), np.float32)
    for i in range(c):
        p[i] = cv2.resize(small[i], (mw, mh), interpolation=cv2.INTER_LINEAR)
    return p


class TestMasksFromProto:
    def test_shape_dtype_and_length(self):
        lb = _lb(480, 640)
        boxes = np.asarray([[100.0, 120.0, 300.0, 340.0], [400.0, 200.0, 500.0, 300.0]])
        cf = np.zeros((2, 32), np.float32)
        cf[:, 0] = 1.0
        got = masks_from_proto(_smooth_proto(), cf, boxes, lb)
        assert len(got) == 2
        for g in got:
            assert g.dtype == bool and g.shape == (480, 640) and g.any()

    def test_accepts_4d_proto(self):
        lb = _lb(480, 640)
        p = np.full((1, 32, 160, 160), 6.0, np.float32)
        got = masks_from_proto(p, np.ones((1, 32), np.float32),
                               np.asarray([[100.0, 200.0, 200.0, 300.0]]), lb)
        assert got[0].any()

    def test_proto_batch_gt_one_raises(self):
        with pytest.raises(ValueError, match="batch"):
            masks_from_proto(np.zeros((2, 32, 160, 160), np.float32),
                             np.ones((1, 32), np.float32), np.zeros((1, 4)), _lb(480, 640))

    def test_proto_ndim_raises(self):
        with pytest.raises(ValueError, match="proto"):
            masks_from_proto(np.zeros((160, 160), np.float32), np.ones((1, 32), np.float32),
                             np.zeros((1, 4)), _lb(480, 640))

    def test_count_mismatch_raises(self):
        with pytest.raises(ValueError, match="数量不一致"):
            masks_from_proto(_smooth_proto(), np.ones((2, 32), np.float32),
                             np.zeros((1, 4)), _lb(480, 640))

    def test_box_outside_grid_gives_all_false(self):
        lb = _lb(480, 640)
        got = masks_from_proto(_smooth_proto(), np.ones((1, 32), np.float32),
                               np.asarray([[5000.0, 5000.0, 6000.0, 6000.0]]), lb)
        assert not got[0].any()

    def test_degenerate_box_gives_all_false(self):
        lb = _lb(480, 640)
        got = masks_from_proto(_smooth_proto(), np.ones((1, 32), np.float32),
                               np.asarray([[100.0, 100.0, 100.4, 100.4]]), lb)
        assert not got[0].any()

    def test_higher_threshold_shrinks_mask(self):
        lb = _lb(480, 640)
        proto = _smooth_proto()
        rng = np.random.default_rng(3)
        cf = rng.normal(0.0, 1.0, (1, 32)).astype(np.float32)
        boxes = np.asarray([[80.0, 120.0, 400.0, 420.0]])
        lo = masks_from_proto(proto, cf, boxes, lb, mask_thr=0.3)[0]
        hi = masks_from_proto(proto, cf, boxes, lb, mask_thr=0.7)[0]
        assert lo.sum() > hi.sum()

    def test_matches_full_upsample_reference(self):
        """回归：与"整图放大再裁剪"的参照实现比对（真 bug 就是这么抓出来的）。"""
        lb = _lb(480, 640)
        proto = _smooth_proto()
        rng = np.random.default_rng(11)
        boxes = np.asarray([[80.0, 120.0, 400.0, 420.0],
                            [200.0, 260.0, 560.0, 540.0],
                            [100.0, 90.0, 300.0, 300.0]])
        cf = rng.normal(0.0, 1.0, (3, 32)).astype(np.float32)
        ours = masks_from_proto(proto, cf, boxes, lb)
        ref = _ref_masks(proto, cf, boxes, lb)
        ious, ratios = [], []
        for o, r in zip(ours, ref):
            ious.append(mask_iou(o, r))
            ratios.append(o.sum() / max(int(r.sum()), 1))
        assert min(ious) >= 0.90, f"掩码 IoU 太低：{ious}"
        assert all(0.90 <= x <= 1.10 for x in ratios), f"像素比越界：{ratios}"

    def test_filled_region_area_follows_grid_geometry(self):
        """回归（面积口径）：贴回原图的宽高必须按"裁剪区的网格尺寸 x scale / ratio"算。

        第一版按**框的像素尺寸**拉伸，等于给掩码做了一次压缩：真实数据上表现为
        ours/theirs 像素比 0.98、掩码 IoU 0.847（卡门槛失败）。这里用整块填满的
        proto 造一个边界干净的掩码，直接比面积，偏差必须 < 2%。
        """
        lb = _lb(480, 640)
        proto = np.full((32, 160, 160), 6.0, np.float32)      # sigmoid(32*6) 约等于 1
        boxes = np.asarray([[101.5, 141.5, 140.5, 180.5]])    # 小框，1 格的误差就看得见
        cf = np.ones((1, 32), np.float32)
        o = masks_from_proto(proto, cf, boxes, lb)[0]
        r = _ref_masks(proto, cf, boxes, lb)[0]
        no, nr = int(o.sum()), int(r.sum())
        assert nr > 500, nr
        assert abs(no - nr) / nr < 0.02, f"ours={no} ref={nr} 比值 {no / nr:.4f}"
        assert mask_iou(o, r) >= 0.95

    def test_mask_stays_inside_image(self):
        lb = _lb(480, 640)
        proto = np.full((32, 160, 160), 6.0, np.float32)
        got = masks_from_proto(proto, np.ones((1, 32), np.float32),
                               np.asarray([[0.0, 0.0, 640.0, 640.0]]), lb)[0]
        assert got.shape == (480, 640)
        rows = np.flatnonzero(got.any(axis=1))
        assert rows.min() >= 0 and rows.max() < 480


class TestMaskIou:
    def _m(self, sl, shape=(20, 20)):
        a = np.zeros(shape, bool)
        a[sl] = True
        return a

    def test_both_empty_is_one(self):
        z = np.zeros((10, 10), bool)
        assert mask_iou(z, z) == 1.0

    def test_one_empty_is_zero(self):
        assert mask_iou(np.zeros((10, 10), bool), self._m(np.s_[:5, :5], (10, 10))) == 0.0

    def test_identical(self):
        m = self._m(np.s_[2:8, 2:8])
        assert mask_iou(m, m.copy()) == 1.0

    def test_disjoint(self):
        assert mask_iou(self._m(np.s_[:5, :5]), self._m(np.s_[10:15, 10:15])) == 0.0

    def test_known_value(self):
        a = self._m(np.s_[0:10, 0:10])
        b = self._m(np.s_[5:15, 0:10])          # inter 50, union 150
        assert mask_iou(a, b) == pytest.approx(50 / 150)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="形状不一致"):
            mask_iou(np.zeros((4, 4), bool), np.zeros((5, 5), bool))

    def test_negative_erode_raises(self):
        with pytest.raises(ValueError, match="erode_px"):
            mask_iou(np.zeros((4, 4), bool), np.zeros((4, 4), bool), erode_px=-1)

    def test_erode_removes_isolated_noise(self):
        a = self._m(np.s_[5:15, 5:15])
        b = a.copy()
        b[0, 0] = True
        assert mask_iou(a, b) < 1.0
        assert mask_iou(a, b, erode_px=1) == 1.0

    def test_erode_can_empty_both_masks(self):
        line = self._m(np.s_[5:6, 5:15])        # 1 px 粗，腐蚀 1 就没了
        assert mask_iou(line, line.copy(), erode_px=1) == 1.0
        assert mask_iou(line, np.zeros((20, 20), bool), erode_px=1) == 1.0

    def test_erode_keeps_solid_block_equal(self):
        a = self._m(np.s_[5:15, 5:15])
        assert mask_iou(a, a.copy(), erode_px=3) == 1.0


# --------------------------------------------------------------------------
# Detection / 记录序列化
# --------------------------------------------------------------------------

class TestDetection:
    def test_area(self):
        assert _det(box=(2.0, 3.0, 12.0, 8.0)).area() == 50.0

    def test_area_degenerate_is_zero(self):
        assert _det(box=(5.0, 5.0, 5.0, 5.0)).area() == 0.0
        assert _det(box=(10.0, 10.0, 4.0, 3.0)).area() == 0.0

    def test_to_dict_default_keys(self):
        r = _det(score=0.87654321, cls=2, box=(1.0, 2.0, 3.5, 4.5)).to_dict()
        assert set(r) == {"score", "cls", "box"}
        assert r["score"] == 0.876543 and r["cls"] == 2
        assert r["box"] == [1.0, 2.0, 3.5, 4.5]

    def test_to_dict_names(self):
        assert _det(cls=1).to_dict(names=["a", "b"])["name"] == "b"

    def test_to_dict_names_out_of_range(self):
        assert _det(cls=7).to_dict(names=["a"])["name"] == "7"

    def test_to_dict_mask_pixels(self):
        m = np.zeros((10, 10), bool)
        m[:4, :6] = True
        d = _det(mask=m)
        assert "mask_pixels" not in d.to_dict()
        assert d.to_dict(with_mask=True)["mask_pixels"] == 24

    def test_to_dict_mask_none(self):
        assert "mask_pixels" not in _det(mask=None).to_dict(with_mask=True)

    def test_records(self):
        recs = detections_to_records([_det(cls=0), _det(cls=1)], names=["x", "y"])
        assert [r["name"] for r in recs] == ["x", "y"]
        assert detections_to_records([]) == []

    def test_box640_kept_separately(self):
        d = _det(box=(0.0, 0.0, 10.0, 10.0))
        assert np.allclose(d.box640, np.asarray([0.0, 0.0, 5.0, 5.0]))
        assert "box640" not in d.to_dict()


# --------------------------------------------------------------------------
# output0 排布归一化
# --------------------------------------------------------------------------

class TestAsPredictions:
    def test_trt_layout_transposed(self):
        a = np.zeros((1, 38, 8400), np.float32)
        a[0, 5, 100] = 1.0
        p = _as_predictions(a, 38)
        assert p.shape == (8400, 38) and p[100, 5] == 1.0

    def test_ultralytics_layout_passthrough(self):
        a = np.zeros((8400, 38), np.float32)
        a[100, 5] = 1.0
        p = _as_predictions(a, 38)
        assert p.shape == (8400, 38) and p[100, 5] == 1.0

    def test_square_raises(self):
        with pytest.raises(ValueError, match="方阵"):
            _as_predictions(np.zeros((1, 38, 38), np.float32), 38)

    def test_batch_gt_one_raises(self):
        with pytest.raises(ValueError, match="batch"):
            _as_predictions(np.zeros((2, 38, 8400), np.float32), 38)

    def test_wrong_ndim_raises(self):
        with pytest.raises(ValueError, match="output0"):
            _as_predictions(np.zeros((38,), np.float32), 38)

    def test_neither_dim_matches(self):
        with pytest.raises(ValueError, match="都不是 n_attrs"):
            _as_predictions(np.zeros((1, 40, 8400), np.float32), 38)

    def test_result_is_contiguous(self):
        assert _as_predictions(np.zeros((1, 38, 100), np.float32), 38).flags["C_CONTIGUOUS"]


# --------------------------------------------------------------------------
# postprocess 端到端
# --------------------------------------------------------------------------

NC, C, N_ATTRS, N_ANCH = 2, 32, 38, 8400
# output0 的类别分数在导出图里已经 sigmoid 过，存的是概率不是 logit
_S09, _S08, _S02, _S04, _S07, _S095 = 0.9, 0.8, 0.2, 0.4, 0.7, 0.95


def _pp(peaks, lb=None, proto_val=6.0, coeff=1.0, out0=None, **kw):
    """按 peaks 造一对 (out0, out1) 跑完整后处理，返回 (dets, lb)。"""
    lb = _lb(480, 640) if lb is None else lb
    a = _mk_out0(N_ATTRS, N_ANCH, peaks) if out0 is None else out0
    if out0 is None:
        for anchor, *_ in peaks:
            a[0, 4 + NC:4 + NC + C, anchor] = coeff
    o1 = np.full((1, C, 160, 160), proto_val, np.float32)
    return postprocess(a, o1, lb, **kw), lb


class TestPostprocess:
    def test_single_peak(self):
        dets, lb = _pp([(500, 1, _S09, 320.0, 240.0, 100.0, 60.0)])
        assert len(dets) == 1
        d = dets[0]
        assert d.cls == 1 and d.score == pytest.approx(0.9, abs=1e-5)
        # cxcywh(320,240,100,60) -> xyxy(270,210,370,270)，减 dh=80 后映射回原图
        assert np.allclose(d.box, [270.0, 130.0, 370.0, 190.0], atol=1e-6)
        assert np.allclose(d.box640, [270.0, 210.0, 370.0, 270.0])
        assert d.mask is not None and d.mask.shape == lb.orig_shape and d.mask.any()

    def test_below_conf_dropped(self):
        dets, _ = _pp([(500, 0, _S02, 320.0, 240.0, 50.0, 50.0)])
        assert dets == []

    def test_conf_threshold_is_strictly_greater(self):
        dets, _ = _pp([(500, 0, _S09, 320.0, 240.0, 50.0, 50.0)], conf_thr=0.9)
        assert dets == []

    def test_sorted_by_score_desc(self):
        dets, _ = _pp([(100, 0, _S04, 100.0, 200.0, 40.0, 40.0),
                       (200, 0, _S095, 300.0, 300.0, 40.0, 40.0),
                       (300, 1, _S07, 500.0, 400.0, 40.0, 40.0)])
        assert [d.score for d in dets] == sorted((d.score for d in dets), reverse=True)
        assert [d.cls for d in dets] == [0, 1, 0]

    def test_nms_suppresses_duplicate(self):
        dets, _ = _pp([(100, 0, _S09, 320.0, 240.0, 100.0, 100.0),
                       (101, 0, _S08, 322.0, 242.0, 100.0, 100.0)],
                      iou_thr=0.5)
        assert len(dets) == 1 and dets[0].score == pytest.approx(0.9, abs=1e-5)

    def test_nms_keeps_different_classes(self):
        dets, _ = _pp([(100, 0, _S09, 320.0, 240.0, 100.0, 100.0),
                       (101, 1, _S08, 322.0, 242.0, 100.0, 100.0)],
                      iou_thr=0.5)
        assert len(dets) == 2

    def test_class_scores_are_probabilities_not_logits(self):
        """口径钉住：后处理**不能**再对类别分数套一次 sigmoid。

        YOLOv8-seg 导出的 output0 里类别分数已经是 [0,1] 概率，只有掩码系数那条
        链路要 sigmoid。喂一个 logit(0.9)=2.197 进去，出来就该还是 2.197。
        这条写错不会报错，只会让所有分数挤在 0.9 附近、conf 阈值形同虚设，很难查。
        """
        dets, _ = _pp([(500, 0, 2.1972246, 320.0, 240.0, 100.0, 100.0)])
        assert len(dets) == 1
        assert dets[0].score == pytest.approx(2.1972246, abs=1e-5)

    def test_need_masks_false(self):
        dets, _ = _pp([(500, 0, _S09, 320.0, 240.0, 100.0, 100.0)], need_masks=False)
        assert len(dets) == 1 and dets[0].mask is None

    def test_max_det_limits(self):
        peaks = [(10 * i, 0, _S09 - 0.01 * i, 60.0 + 40 * i, 240.0, 20.0, 20.0)
                 for i in range(5)]
        dets, _ = _pp(peaks, max_det=2)
        assert len(dets) == 2

    def test_max_det_non_positive_raises(self):
        with pytest.raises(ValueError, match="max_det"):
            _pp([(500, 0, _S09, 320.0, 240.0, 100.0, 100.0)], max_det=0)

    def test_nc_inferred_from_shape(self):
        dets, _ = _pp([(500, 1, _S09, 320.0, 240.0, 100.0, 60.0)])
        assert dets and dets[0].cls == 1     # 没传 nc，从 38-4-32=2 推出来

    def test_explicit_nc_must_match(self):
        with pytest.raises(ValueError):
            _pp([(500, 1, _S09, 320.0, 240.0, 100.0, 60.0)], nc=3)

    def test_square_out0_raises(self):
        with pytest.raises(ValueError):
            _pp([], out0=np.zeros((1, 38, 38), np.float32))

    def test_out1_batch_gt_one_raises(self):
        lb = _lb(480, 640)
        with pytest.raises(ValueError, match="batch"):
            postprocess(_mk_out0(N_ATTRS, N_ANCH, []), np.zeros((2, C, 160, 160), np.float32), lb)

    def test_2d_out0_accepted(self):
        a = _mk_out0(N_ATTRS, N_ANCH, [(500, 0, _S09, 320.0, 240.0, 100.0, 100.0)])[0]
        a[4 + NC:4 + NC + C, 500] = 1.0
        dets = postprocess(a, np.full((1, C, 160, 160), 6.0, np.float32), _lb(480, 640))
        assert len(dets) == 1

    def test_boxes_inside_orig_image(self):
        peaks = [(i, 0, _S09, 320.0, 320.0, 900.0, 900.0) for i in range(1)]
        dets, lb = _pp(peaks)
        x1, y1, x2, y2 = dets[0].box
        assert 0 <= x1 < x2 <= lb.orig_shape[1] and 0 <= y1 < y2 <= lb.orig_shape[0]

    def test_mask_threshold_effect(self):
        peaks = [(500, 0, _S09, 320.0, 240.0, 200.0, 200.0)]
        lo, _ = _pp(peaks, proto_val=0.0, coeff=0.0, mask_thr=0.5)
        hi, _ = _pp(peaks, proto_val=0.0, coeff=0.0, mask_thr=0.9)
        assert lo[0].mask.sum() > 0
        assert hi[0].mask.sum() == 0


# --------------------------------------------------------------------------
# 对拍匹配
# --------------------------------------------------------------------------

class TestMatchDetections:
    def _sq(self, w=10.0, h=10.0, x=0.0, y=0.0):
        return (x, y, x + w, y + h)

    def test_both_empty(self):
        r = match_detections([], [])
        assert r["match_rate"] == 1.0 and r["n_matched"] == 0
        assert r["mean_box_iou"] is None and r["mean_mask_iou"] is None
        assert r["mean_abs_score_diff"] is None and r["mask_erode_px"] == 0

    def test_identical(self):
        a = [_det(0.9, 0, self._sq()), _det(0.8, 1, self._sq(20, 20, 30, 30))]
        b = [_det(0.9, 0, self._sq()), _det(0.8, 1, self._sq(20, 20, 30, 30))]
        r = match_detections(a, b)
        assert r["n_matched"] == 2 and r["match_rate"] == 1.0
        assert r["mean_box_iou"] == 1.0 and r["min_box_iou"] == 1.0
        assert r["mean_abs_score_diff"] == 0.0
        assert r["unmatched_ours"] == [] and r["unmatched_theirs"] == []

    def test_class_mismatch_unmatched(self):
        r = match_detections([_det(0.9, 0, self._sq())], [_det(0.9, 1, self._sq())])
        assert r["n_matched"] == 0 and r["match_rate"] == 0.0
        assert r["unmatched_ours"] == [0] and r["unmatched_theirs"] == [0]

    def test_low_iou_unmatched(self):
        r = match_detections([_det(0.9, 0, self._sq())], [_det(0.9, 0, self._sq(10, 10, 50, 50))])
        assert r["n_matched"] == 0

    def test_iou_threshold_boundary_inclusive(self):
        a = _det(0.9, 0, self._sq(10, 10))
        b = _det(0.9, 0, self._sq(10, 20))          # inter 100 / union 200 = 0.5
        assert match_detections([a], [b], iou_thr=0.5)["n_matched"] == 1
        assert match_detections([a], [b], iou_thr=0.51)["n_matched"] == 0

    def test_denominator_is_max_side(self):
        ours = [_det(0.9, 0, self._sq())]
        theirs = [_det(0.9, 0, self._sq()), _det(0.7, 0, self._sq(10, 10, 100, 100))]
        r = match_detections(ours, theirs)
        assert r["match_rate"] == 0.5 and r["unmatched_theirs"] == [1]

    def test_greedy_takes_highest_score_first(self):
        ours = [_det(0.5, 0, self._sq()), _det(0.95, 0, self._sq())]
        r = match_detections(ours, [_det(0.9, 0, self._sq())])
        assert r["n_matched"] == 1 and r["unmatched_ours"] == [0]

    def test_one_theirs_cannot_be_reused(self):
        ours = [_det(0.95, 0, self._sq()), _det(0.9, 0, self._sq())]
        r = match_detections(ours, [_det(0.9, 0, self._sq())])
        assert r["n_matched"] == 1 and r["unmatched_ours"] == [1]

    def test_mask_iou_reported(self):
        m = np.zeros((20, 20), bool)
        m[2:12, 2:12] = True
        r = match_detections([_det(0.9, 0, self._sq(), mask=m)],
                             [_det(0.9, 0, self._sq(), mask=m.copy())])
        assert r["mean_mask_iou"] == 1.0

    def test_with_mask_off(self):
        m = np.zeros((20, 20), bool)
        m[:5, :5] = True
        r = match_detections([_det(0.9, 0, self._sq(), mask=m)],
                             [_det(0.9, 0, self._sq(), mask=m)], with_mask=False)
        assert r["mean_mask_iou"] is None

    def test_missing_mask_gives_none(self):
        r = match_detections([_det(0.9, 0, self._sq(), mask=None)],
                             [_det(0.9, 0, self._sq())])
        assert r["n_matched"] == 1 and r["mean_mask_iou"] is None

    def test_mask_shape_mismatch_skipped_not_raised(self):
        m1 = np.zeros((20, 20), bool)
        m1[:5, :5] = True
        m2 = np.zeros((30, 30), bool)
        m2[:5, :5] = True
        r = match_detections([_det(0.9, 0, self._sq(), mask=m1)],
                             [_det(0.9, 0, self._sq(), mask=m2)])
        assert r["n_matched"] == 1 and r["mean_mask_iou"] is None

    def test_erode_px_recorded_and_applied(self):
        m = np.zeros((20, 20), bool)
        m[2:12, 2:12] = True
        n = m.copy()
        n[0, 0] = True
        assert match_detections([_det(0.9, 0, self._sq(), mask=m)],
                                [_det(0.9, 0, self._sq(), mask=n)])["mean_mask_iou"] < 1.0
        r = match_detections([_det(0.9, 0, self._sq(), mask=m)],
                             [_det(0.9, 0, self._sq(), mask=n)], mask_erode_px=1)
        assert r["mask_erode_px"] == 1 and r["mean_mask_iou"] == 1.0


# --------------------------------------------------------------------------
# 串行对照组
# --------------------------------------------------------------------------

class TestRunSerial:
    def test_results_and_report(self):
        out, rep = run_serial([1, 2, 3], [("inc", lambda x: x + 1), ("dbl", lambda x: x * 2)])
        assert out == [4, 6, 8]
        assert rep["mode"] == "serial" and rep["n_items"] == 3 and rep["n_results"] == 3
        assert rep["errors"] == [] and rep["queues"] == []
        assert [s["name"] for s in rep["stages"]] == ["inc", "dbl"]
        assert [s["calls"] for s in rep["stages"]] == [3, 3]

    def test_zero_items(self):
        out, rep = run_serial([], [lambda x: x])
        assert out == [] and rep["n_items"] == 0
        assert rep["throughput_fps"] is None

    def test_bare_functions_get_default_names(self):
        _, rep = run_serial([1], [lambda x: x, lambda x: x])
        assert [s["name"] for s in rep["stages"]] == ["stage0", "stage1"]

    def test_util_bounded(self):
        _, rep = run_serial(list(range(30)), [("s", lambda x: x)])
        u = rep["stages"][0]["util"]
        assert u is not None and 0.0 <= u <= 1.0 + 1e-9
        assert rep["stages"][0]["calls"] == 30

    def test_stage_report_fields(self):
        _, rep = run_serial([1, 2], [("s", lambda x: x)])
        st = rep["stages"][0]
        for k in ("name", "calls", "busy_s", "util", "mean_ms", "p50_ms", "p95_ms", "max_ms"):
            assert k in st, k

    def test_exception_propagates(self):
        def boom(x):
            raise KeyError("k")
        with pytest.raises(KeyError):
            run_serial([1], [boom])


# --------------------------------------------------------------------------
# 阶段校验
# --------------------------------------------------------------------------

class TestStageValidation:
    def test_empty_stages_raise(self):
        with pytest.raises(ValueError, match="至少要有一个阶段"):
            Pipeline([])

    @pytest.mark.parametrize("cap", [0, -1])
    def test_bad_capacity_raises(self, cap):
        with pytest.raises(ValueError, match="capacity"):
            Pipeline([lambda x: x], capacity=cap)

    def test_duplicate_names_raise(self):
        with pytest.raises(ValueError, match="阶段名重复"):
            Pipeline([("a", lambda x: x), ("a", lambda x: x)])

    def test_not_callable_raises(self):
        with pytest.raises(TypeError, match="不是可调用对象"):
            Pipeline([("a", 42)])

    def test_bad_tuple_length_raises(self):
        with pytest.raises(ValueError, match="name, fn"):
            Pipeline([("a", lambda x: x, 1)])

    def test_names_are_stringified(self):
        pl = Pipeline([(7, lambda x: x)])
        assert pl.stages[0][0] == "7"

    def test_tuple_and_bare_forms_mix(self):
        pl = Pipeline([("a", lambda x: x), lambda x: x])
        assert [n for n, _ in pl.stages] == ["a", "stage1"]


# --------------------------------------------------------------------------
# 流水线本体
# --------------------------------------------------------------------------

class TestPipeline:
    def test_results_and_order(self):
        def jitter(x):
            time.sleep(0.001 * (5 - x % 5))
            return x * 2
        out, rep = Pipeline([("a", jitter), ("b", lambda x: x + 1)]).run(list(range(30)))
        assert out == [x * 2 + 1 for x in range(30)]
        assert rep["mode"] == "pipeline" and rep["n_items"] == 30 and rep["n_results"] == 30
        assert rep["errors"] == []
        assert rep["throughput_fps"] is not None and rep["throughput_fps"] > 0

    def test_zero_items_no_deadlock(self):
        out, rep = Pipeline([lambda x: x], capacity=1, join_timeout=5).run([])
        assert out == [] and rep["n_items"] == 0 and rep["n_results"] == 0

    def test_capacity_one_no_deadlock(self):
        out, _ = Pipeline([("a", lambda x: x + 1), ("b", lambda x: x * 3)],
                          capacity=1, join_timeout=5).run(list(range(20)))
        assert out == [(x + 1) * 3 for x in range(20)]

    def test_single_stage(self):
        out, rep = Pipeline([("only", lambda x: -x)], capacity=2, join_timeout=5).run([1, 2, 3])
        assert out == [-1, -2, -3] and len(rep["queues"]) == 2

    def test_stage_error_carries_context(self):
        def boom(x):
            if x == 7:
                raise ZeroDivisionError("炸了")
            return x
        with pytest.raises(PipelineError) as ei:
            Pipeline([("s1", lambda x: x), ("s2", boom)], join_timeout=5).run(list(range(20)))
        err = ei.value
        assert err.stage == "s2" and err.index == 7
        assert isinstance(err, RuntimeError)
        assert "ZeroDivisionError" in str(err.exc)
        assert "炸了" in str(err) and "s2" in str(err) and "#7" in str(err)

    def test_error_in_first_stage(self):
        def boom(x):
            raise ValueError("第一枪就炸")
        with pytest.raises(PipelineError) as ei:
            Pipeline([("d", boom), ("i", lambda x: x)], join_timeout=5).run([1, 2, 3])
        assert ei.value.stage == "d" and ei.value.index == 0   # index 是 item 位置，不是值

    def test_threads_all_joined(self):
        before = threading.active_count()
        Pipeline([lambda x: x, lambda x: x, lambda x: x], join_timeout=5).run(list(range(50)))
        deadline = time.time() + 5.0
        while threading.active_count() > before and time.time() < deadline:
            time.sleep(0.02)
        assert threading.active_count() == before, "流水线漏线程了"

    def test_report_queues_and_closed(self):
        _, rep = Pipeline([lambda x: x, lambda x: x, lambda x: x], capacity=2).run(list(range(10)))
        assert len(rep["queues"]) == 4                  # 3 条阶段间 + qout
        assert rep["capacity"] == 2
        assert all(q["closed"] for q in rep["queues"])
        assert sum(q["puts"] for q in rep["queues"][:3]) == 30

    def test_queue_names_point_at_consumer(self):
        _, rep = Pipeline([("dec", lambda x: x), ("inf", lambda x: x)]).run([1])
        names = [q["name"] for q in rep["queues"]]
        assert names == ["q0->dec", "q1->inf", "qout"]

    def test_slow_stage_causes_backpressure_upstream(self):
        """瓶颈阶段**前面**那条队列必然被塞满过，且 max_depth 顶到容量。"""
        def slow(x):
            time.sleep(0.004)
            return x
        _, rep = Pipeline([("fast", lambda x: x), ("slow", slow)],
                          capacity=1, join_timeout=10).run(list(range(40)))
        q_in_slow = [q for q in rep["queues"] if q["name"] == "q1->slow"][0]
        assert q_in_slow["overflow_waits"] > 0
        assert q_in_slow["max_depth"] == 1
        util = {s["name"]: s["util"] for s in rep["stages"]}
        assert util["slow"] > util["fast"]

    def test_slow_feeder_causes_underflow_downstream(self):
        def slow(x):
            time.sleep(0.004)
            return x
        _, rep = Pipeline([("slow", slow), ("fast", lambda x: x)],
                          capacity=1, join_timeout=10).run(list(range(40)))
        q_in_fast = [q for q in rep["queues"] if q["name"] == "q1->fast"][0]
        assert q_in_fast["underflow_waits"] > 0

    def test_pipeline_beats_serial_on_sleepy_stages(self):
        stages = [("a", lambda x: (time.sleep(0.01), x)[1]),
                  ("b", lambda x: (time.sleep(0.01), x + 1)[1])]
        items = list(range(30))
        _, srep = run_serial(items, stages)
        _, prep = Pipeline(stages, capacity=4, join_timeout=10).run(items)
        assert prep["wall_s"] < srep["wall_s"]
        cmp_ = speedup_report(srep, prep)
        assert cmp_["speedup"] > 1.2

    def test_results_survive_stage_exception_in_later_item(self):
        """出错时抛异常，不静默返回半截结果 —— 压测数字必须全有或全无。"""
        def boom(x):
            if x == 3:
                raise RuntimeError("nope")
            return x
        with pytest.raises(PipelineError):
            Pipeline([boom], join_timeout=5).run(list(range(10)))

    def test_items_are_not_copied(self):
        """阶段之间传的是对象引用，不是深拷贝 —— 大数组过流水线不该有额外内存开销。"""
        arr = np.zeros((640, 640, 3), np.uint8)
        out, _ = Pipeline([("a", lambda x: x), ("b", lambda x: x)], join_timeout=5).run([arr])
        assert out[0] is arr

    def test_reusable_instance(self):
        pl = Pipeline([("a", lambda x: x + 1)], capacity=2, join_timeout=5)
        assert pl.run([1, 2])[0] == [2, 3]
        assert pl.run([10])[0] == [11]


# --------------------------------------------------------------------------
# 加速比对照表
# --------------------------------------------------------------------------

class TestSpeedupReport:
    def _rep(self, mode, fps, wall, stages):
        return {"mode": mode, "throughput_fps": fps, "wall_s": wall,
                "n_items": 300, "stages": stages}

    def test_speedup_and_bottleneck(self):
        s = self._rep("serial", 39.5, 7.6, [{"name": "decode", "busy_s": 3.0, "util": 0.39},
                                            {"name": "infer", "busy_s": 2.4, "util": 0.32},
                                            {"name": "post", "busy_s": 0.6, "util": 0.08}])
        p = self._rep("pipeline", 56.5, 5.3, [{"name": "decode", "busy_s": 5.1, "util": 0.96},
                                              {"name": "infer", "busy_s": 5.2, "util": 0.98},
                                              {"name": "post", "busy_s": 1.4, "util": 0.26}])
        r = speedup_report(s, p)
        assert r["speedup"] == round(56.5 / 39.5, 3)
        assert r["bottleneck_stage"] == "infer"          # busy_s 最大的那个
        assert r["serial_fps"] == 39.5 and r["pipeline_fps"] == 56.5
        assert r["n_items"] == 300
        assert r["stage_util"] == {"decode": 0.96, "infer": 0.98, "post": 0.26}
        assert r["serial_stage_util"]["decode"] == 0.39

    def test_missing_fps_gives_none(self):
        r = speedup_report({"throughput_fps": None},
                           {"throughput_fps": 5.0, "stages": [{"name": "x", "busy_s": 1.0}]})
        assert r["speedup"] is None and r["bottleneck_stage"] == "x"

    def test_no_stages_gives_none_bottleneck(self):
        r = speedup_report({"throughput_fps": 3.0}, {"throughput_fps": 5.0})
        assert r["bottleneck_stage"] is None and r["stage_util"] == {}

    def test_rounding_three_decimals(self):
        r = speedup_report({"throughput_fps": 3.0}, {"throughput_fps": 5.0, "stages": []})
        assert r["speedup"] == 1.667

    def test_busy_s_none_treated_as_zero(self):
        r = speedup_report({"throughput_fps": 1.0},
                           {"throughput_fps": 2.0,
                            "stages": [{"name": "a", "busy_s": None},
                                       {"name": "b", "busy_s": 0.5}]})
        assert r["bottleneck_stage"] == "b"


# --------------------------------------------------------------------------
# 等待计数口径
# --------------------------------------------------------------------------

class TestRingBufferWaitCounting:
    """``overflow_waits`` / ``underflow_waits`` 按**被挡住的调用次数**记。

    早先是在 while 循环里记的：超时唤醒后发现队列还是空的会再记一次，一次 acquire
    记 2。压测报告里的 ``pool_waits`` 直接读这个数，虚高一倍就会把"池子够不够大"
    这个判断带偏，所以口径要钉住。
    """

    def test_timed_out_get_counts_once(self):
        rb = RingBuffer(1, name="q")
        t0 = time.perf_counter()
        assert rb.get(timeout=0.05) is None
        assert rb.stats()["underflow_waits"] == 1
        assert time.perf_counter() - t0 >= 0.04, "timeout 没真的等"

    def test_timed_out_put_counts_once(self):
        rb = RingBuffer(1, name="q")
        assert rb.put("x") is True
        assert rb.put("y", timeout=0.05) is False
        assert rb.stats()["overflow_waits"] == 1

    def test_repeated_calls_count_each(self):
        rb = RingBuffer(1, name="q")
        for _ in range(3):
            assert rb.get(timeout=0.01) is None
        assert rb.stats()["underflow_waits"] == 3

    def test_successful_get_does_not_count(self):
        rb = RingBuffer(2, name="q")
        rb.put("a")
        assert rb.get() == "a"
        assert rb.stats()["underflow_waits"] == 0 and rb.stats()["overflow_waits"] == 0

    def test_blocking_get_counts_once_when_producer_slow(self):
        rb = RingBuffer(2, name="q")
        got = []

        def consumer():
            got.append(rb.get(timeout=3.0))

        th = threading.Thread(target=consumer, daemon=True)
        th.start()
        time.sleep(0.1)
        rb.put("a")
        th.join(timeout=5)
        assert got == ["a"] and rb.stats()["underflow_waits"] == 1
