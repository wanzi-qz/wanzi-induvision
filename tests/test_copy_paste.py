"""Copy-Paste 缺陷合成引擎的测试。

盯四条不能破的性质：
1. **可复现**：同一个 SeedSequence 出来的像素与标签必须逐字节一致。
2. **像素与标签一致**：``max_overlap_frac=0`` 时，一个连通域 = 一个实例，
   否则实例计数对不上，训练目标就是错的。
3. **不贴到无纹理背景**：MVTec 背景近乎纯白/纯黑，贴上去模型会学到
   「白底上一块亮斑 = 缺陷」这种假规律，所以要有 local_std 门槛。
4. **改动范围可控**：naive 融合只允许改硬掩码内的像素。
"""
from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from induvision.copy_paste import (
    PasteParams,
    blend_patch,
    choose_position,
    local_std_map,
    masks_to_label_items,
    paste_instances,
    photometric_jitter,
    rasterize_instances,
    render_comparison,
    warp_instance,
)
from induvision.defect_bank import build_bank
from induvision.yolo_dataset import read_dataset

W, H = 96, 72


@pytest.fixture
def pool(tiny_yolo):
    return build_bank(read_dataset(tiny_yolo.root)).instances


def canvas(seed: int = 1, flat: bool = False, w: int = W, h: int = H) -> np.ndarray:
    """贴图底板：默认随机噪声（有纹理），flat=True 时是纯色（模拟 MVTec 白背景）。"""
    if flat:
        return np.full((h, w, 3), 200, np.uint8)
    return np.random.default_rng(seed).integers(0, 255, (h, w, 3), dtype=np.uint8)


def rng_for(seed: int = 0, class_id: int = 0, seq: int = 0) -> np.random.Generator:
    """与 CLI 完全一致的派生方式：SeedSequence([seed, class_id, seq])。"""
    return np.random.default_rng(np.random.SeedSequence([seed, class_id, seq]))


# ---------------------------------------------------------------- 可复现性


def test_same_seed_is_pixel_identical(pool):
    runs = []
    for _ in range(2):
        res = paste_instances(canvas(), {}, pool, rng_for(0), PasteParams(blend="naive"), n=2)
        runs.append((res.image.copy(),
                     {k: v.copy() for k, v in res.masks.items()},
                     [(r.instance_id, r.class_id, r.dst_x, r.dst_y, r.scale, r.angle,
                       r.flipped, r.blend_used, r.visible_px, r.total_px) for r in res.records]))
    assert runs[0][2], "一个都没贴上，测试无意义"
    np.testing.assert_array_equal(runs[0][0], runs[1][0])
    assert set(runs[0][1]) == set(runs[1][1])
    for k in runs[0][1]:
        np.testing.assert_array_equal(runs[0][1][k], runs[1][1][k])
    assert runs[0][2] == runs[1][2]


def test_different_seed_changes_placement(pool):
    a = paste_instances(canvas(), {}, pool, rng_for(0), PasteParams(blend="naive"), n=1)
    b = paste_instances(canvas(), {}, pool, rng_for(99), PasteParams(blend="naive"), n=1)
    assert a.records and b.records
    assert (a.records[0].dst_x, a.records[0].dst_y) != (b.records[0].dst_x, b.records[0].dst_y)


def test_empty_pool_is_reported_not_crash(pool):
    res = paste_instances(canvas(), {}, [], rng_for(0), PasteParams())
    assert res.records == [] and res.rejected == {"empty_pool": 1}


# ------------------------------------------------------- 像素 / 标签一致性


def test_zero_overlap_keeps_one_component_per_instance(pool):
    """max_overlap_frac=0 => 每个实例是独立连通域，标签行数才对得上实例数。"""
    priors = np.array([[0.25, 0.25], [0.75, 0.75]])
    p = PasteParams(blend="naive", jitter_sigma=0.01, scale=(0.5, 0.6), max_per_image=3)
    saw_two = False
    for seed in range(8):
        res = paste_instances(canvas(seed), {}, pool, rng_for(seed), p, priors=priors, n=2)
        for cid, m in res.masks.items():
            n_comp, _ = cv2.connectedComponents((m > 0).astype(np.uint8))
            n_rec = sum(1 for r in res.records if r.class_id == cid)
            assert n_comp - 1 == n_rec, (seed, cid, n_comp, n_rec)
        saw_two |= len(res.records) == 2
    assert saw_two, "8 个种子都没能贴上两个缺陷，参数或实现有问题"


def test_naive_blend_only_changes_pixels_inside_mask(pool):
    before = canvas(3)
    img = before.copy()
    res = paste_instances(img, {}, pool, rng_for(3), PasteParams(blend="naive"), n=2)
    assert res.records
    changed = (res.image != before).any(axis=2)
    union = np.zeros(changed.shape, bool)
    for m in res.masks.values():
        union |= m > 0
    assert changed.any()
    assert int((changed & ~union).sum()) == 0


def test_existing_masks_are_preserved_and_merged(pool, tiny_yolo):
    ds = read_dataset(tiny_yolo.root)
    base = next(i for i in ds.split("train") if i.stem == "def_a__000")
    masks = rasterize_instances(base.instances, base.width, base.height)
    assert set(masks) == {0}
    before_px = int(np.count_nonzero(masks[0]))

    res = paste_instances(canvas(4), masks, pool, rng_for(4), PasteParams(blend="naive"), n=2)
    # 传进去的 dict 就是返回的 dict（CLI 依赖这个：合并后的掩码直接用来重建标签）
    assert res.masks is masks
    assert int(np.count_nonzero(masks[0])) >= before_px
    assert sum(len(r.instance_id) > 0 for r in res.records) == len(res.records)


def test_rasterize_instances_matches_polygon(tiny_yolo):
    ds = read_dataset(tiny_yolo.root)
    im = next(i for i in ds.split("train") if i.stem == "def_a__000")
    masks = rasterize_instances(im.instances, im.width, im.height)
    manual = np.zeros((im.height, im.width), np.uint8)
    cv2.fillPoly(manual, [im.instances[0].polygon.reshape(-1, 1, 2)], 255)
    np.testing.assert_array_equal(masks[0], manual)


def test_masks_to_label_items_reports_fidelity():
    mask = np.zeros((H, W), np.uint8)
    cv2.rectangle(mask, (20, 15), (50, 45), 255, -1)
    items, stats = masks_to_label_items({0: mask})
    assert len(items) == 1 and items[0][0] == 0
    assert len(items[0][1]) >= 4
    assert 0.9 < stats["area_fidelity"] <= 1.05
    assert stats["polygons"] == 1
    assert stats["mask_pixels"] == int(np.count_nonzero(mask))


def test_masks_to_label_items_splits_connected_components():
    mask = np.zeros((H, W), np.uint8)
    cv2.rectangle(mask, (5, 5), (20, 20), 255, -1)
    cv2.rectangle(mask, (60, 40), (80, 60), 255, -1)
    items, stats = masks_to_label_items({1: mask})
    assert len(items) == 2 and stats["polygons"] == 2
    assert all(cid == 1 for cid, _ in items)


def test_masks_to_label_items_empty():
    items, stats = masks_to_label_items({})
    assert items == []
    assert stats["area_fidelity"] is None and stats["polygons"] == 0


# ------------------------------------------------------------ 落点与门槛


def test_flat_background_is_rejected(pool):
    img = canvas(flat=True)
    before = img.copy()
    res = paste_instances(img, {}, pool, rng_for(3), PasteParams(blend="naive"))
    assert res.records == []
    assert res.rejected.get("flat_background", 0) > 0
    np.testing.assert_array_equal(img, before)


def test_disabling_texture_gate_lets_flat_background_through(pool):
    """同一张平图、同一个种子：关掉 bg_std_min 就能贴上 -> 证明拒绝来自这道门槛。"""
    res = paste_instances(canvas(flat=True), {}, pool, rng_for(3),
                          PasteParams(blend="naive", bg_std_min=0.0))
    assert res.records
    assert "flat_background" not in res.rejected


def test_local_std_map_separates_texture_from_flat():
    assert local_std_map(canvas(flat=True)).max() == pytest.approx(0.0, abs=1e-6)
    assert local_std_map(canvas(1)).mean() > 0.03


def test_min_paste_area_drops_tiny_defects(pool):
    huge = PasteParams(blend="naive", scale=(0.05, 0.05), min_paste_area=64)
    res = paste_instances(canvas(6), {}, pool, rng_for(6), huge, n=3)
    assert res.records == []
    assert res.rejected.get("too_small", 0) > 0


def test_choose_position_respects_min_visible_ratio(pool):
    p = PasteParams(min_visible_ratio=1.0, candidates=50, bg_std_min=0.0)
    tex = local_std_map(canvas(7))
    xy, rej = choose_position(rng_for(7), np.array([[0.5, 0.5]]), (20, 20), (W, H), None, p, texture=tex)
    assert xy is not None
    x, y = xy
    assert 0 <= x <= W - 20 and 0 <= y <= H - 20  # 完全落在画布内


def test_choose_position_exhausts_and_reports():
    p = PasteParams(candidates=3, min_visible_ratio=1.0, bg_std_min=0.0)
    xy, rej = choose_position(rng_for(1), np.array([[0.0, 0.0]]), (200, 200), (W, H), None, p)
    assert xy is None
    assert rej.get("clipped", 0) + rej.get("exhausted", 0) > 0


def test_overlap_gate_blocks_same_spot(pool):
    """先验只给一个点、抖动为 0：第二个缺陷必然与第一个重叠 -> 被 overlap 挡掉。"""
    p = PasteParams(blend="naive", jitter_sigma=0.0, scale=(0.8, 0.8), max_overlap_frac=0.0)
    res = paste_instances(canvas(8), {}, pool, rng_for(8), p,
                          priors=np.array([[0.5, 0.5]]), n=3)
    assert len(res.records) == 1
    assert res.rejected.get("overlap", 0) > 0 or res.rejected.get("exhausted", 0) > 0


# -------------------------------------------------------------- 几何/光度


def test_warp_instance_identity(pool):
    inst = pool[0]
    patch, hard, soft = warp_instance(inst, scale=1.0, angle=0.0, flip=False, feather=0.0)
    assert patch.shape == inst.patch.shape
    np.testing.assert_array_equal(patch, inst.patch)
    np.testing.assert_array_equal(hard, inst.mask)
    assert set(np.unique(soft)) <= {0.0, 1.0}


def test_warp_instance_scale_and_flip(pool):
    inst = pool[0]
    _, hard1, _ = warp_instance(inst, scale=1.0, angle=0.0, flip=False, feather=0.0)
    _, hard2, _ = warp_instance(inst, scale=2.0, angle=0.0, flip=False, feather=0.0)
    assert hard2.shape[0] == pytest.approx(hard1.shape[0] * 2, abs=2)
    assert hard2.shape[1] == pytest.approx(hard1.shape[1] * 2, abs=2)

    _, hard_f, _ = warp_instance(inst, scale=1.0, angle=0.0, flip=True, feather=0.0)
    np.testing.assert_array_equal(hard_f, cv2.flip(hard1, 1))


def test_warp_instance_rotation_preserves_area(pool):
    inst = pool[0]
    base = np.count_nonzero(warp_instance(inst, scale=1.0, angle=0.0, flip=False, feather=0.0)[1])
    rot = np.count_nonzero(warp_instance(inst, scale=1.0, angle=37.0, flip=False, feather=0.0)[1])
    assert rot == pytest.approx(base, rel=0.15)


def test_soft_hard_mask_consistency(pool):
    _, hard, soft = warp_instance(pool[0], scale=1.0, angle=41.0, flip=True, feather=1.5)
    assert soft.min() >= 0.0 and soft.max() <= 1.0
    np.testing.assert_array_equal(hard, np.where(soft >= 0.5, 255, 0).astype(np.uint8))


def test_photometric_jitter_is_seeded_and_in_range(pool):
    p = PasteParams()
    a = photometric_jitter(pool[0].patch, np.random.default_rng(5), p)
    b = photometric_jitter(pool[0].patch, np.random.default_rng(5), p)
    np.testing.assert_array_equal(a, b)
    assert a.dtype == np.uint8 and a.shape == pool[0].patch.shape
    c = photometric_jitter(pool[0].patch, np.random.default_rng(6), p)
    assert not np.array_equal(a, c)


def test_blend_naive_copies_exact_pixels():
    img = canvas(11)
    patch = np.full((12, 12, 3), 7, np.uint8)
    soft = np.ones((12, 12), np.float32)
    hard = np.zeros((12, 12), np.uint8)
    hard[2:10, 2:10] = 255
    out, used = blend_patch(img, patch, soft, hard, 30, 20, mode="naive", poisson_min_area=64)
    assert used == "naive"
    sel = hard > 0
    np.testing.assert_array_equal(out[sel], patch[sel])
    np.testing.assert_array_equal(out[~sel], img[20:32, 30:42][~sel])


def test_blend_poisson_used_for_big_interior_mask():
    img = canvas(12)
    n = 24
    patch = np.full((n, n, 3), 30, np.uint8)
    soft = np.ones((n, n), np.float32)
    hard = np.full((n, n), 255, np.uint8)
    out, used = blend_patch(img, patch, soft, hard, 30, 20, mode="poisson", poisson_min_area=64)
    assert used == "poisson" and out.shape == (n, n, 3)


def test_blend_poisson_falls_back_when_mask_too_small():
    img = canvas(13)
    patch = np.full((8, 8, 3), 30, np.uint8)
    soft = np.ones((8, 8), np.float32)
    hard = np.zeros((8, 8), np.uint8)
    hard[2:6, 2:6] = 255  # 16 px < 64
    out, used = blend_patch(img, patch, soft, hard, 30, 20, mode="poisson", poisson_min_area=64)
    assert used == "alpha" and out.shape == (8, 8, 3)


def test_blend_out_of_canvas_returns_none_mode():
    img = canvas(14)
    patch = np.full((10, 10, 3), 30, np.uint8)
    soft = np.ones((10, 10), np.float32)
    hard = np.full((10, 10), 255, np.uint8)
    out, used = blend_patch(img, patch, soft, hard, 500, 500, mode="naive", poisson_min_area=64)
    assert used == "none" and out.size == 0


# ------------------------------------------------------------------ 产物


def test_render_comparison_layout(pool):
    before = canvas(15)
    res = paste_instances(before.copy(), {}, pool, rng_for(15), PasteParams(blend="naive"), n=2)
    fig = render_comparison(before, res.image, res.masks, res.records)
    bar_h = 22 + 14 * max(len(res.records), 1)
    assert fig.shape == (H + bar_h, W * 2 + 6, 3)
    assert fig.dtype == np.uint8


def test_render_comparison_handles_no_records():
    before = canvas(16)
    fig = render_comparison(before, before.copy(), {}, [])
    assert fig.shape[1] == W * 2 + 6


# --------------------------------------------------------------- 参数校验


@pytest.mark.parametrize(
    "kwargs",
    [
        {"blend": "magic"},
        {"position": "spiral"},
        {"scale": (1.4, 0.6)},
        {"rotate": (10.0, -10.0)},
        {"scale": (0.0, 1.0)},
        {"max_per_image": 0},
        {"min_visible_ratio": 0.0},
        {"min_visible_ratio": 1.5},
    ],
)
def test_paste_params_validation(kwargs):
    with pytest.raises(ValueError):
        PasteParams(**kwargs)


def test_paste_params_to_dict_is_json_serializable():
    d = PasteParams().to_dict()
    assert json.loads(json.dumps(d))["blend"] == "poisson"
    assert isinstance(d["scale"], list)


def test_defaults_are_the_safe_ones():
    """这几个默认值是有理由的，改动前先看 docs/augmentation.md。"""
    p = PasteParams()
    assert p.max_overlap_frac == 0.0  # 一个连通域 = 一个实例
    assert p.min_paste_area >= 4.0  # 不留「有像素没标签」的噪声
    assert p.bg_std_min > 0.0  # 不贴到无纹理背景
    assert p.blend == "poisson"