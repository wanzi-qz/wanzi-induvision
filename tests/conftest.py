"""构造一个微型但结构完整的合成 MVTec AD 目录树用于测试。"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import cv2
import numpy as np
import pytest

SPEC: Dict[str, Dict[str, int]] = {
    "bottle": {"broken_large": 5, "broken_small": 4},
    "screw": {"thread": 3, "speck": 1},
}


def _write_png(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), arr), f"写入失败 {path}"


def build_mvtec_tree(
    root: Path,
    *,
    spec: Optional[Dict[str, Dict[str, int]]] = None,
    n_train_good: int = 6,
    n_test_good: int = 4,
    size: Tuple[int, int] = (64, 48),
    seed: int = 0,
    extra_categories: Sequence[str] = (),
) -> Path:
    """生成 ``<root>/<category>/{train,test,ground_truth}/...`` 结构。

    掩码形态按缺陷类型区分，便于验证轮廓处理：
      * ``speck``  -> 2x2 像素小斑点（应被 min_area 过滤）
      * ``thread`` -> 细长斜条（多边形顶点较多）
      * 其它       -> 实心矩形
    """
    spec = dict(spec or SPEC)
    for cat in extra_categories:
        spec.setdefault(cat, {"scratch": 3})
    w, h = size
    rng = np.random.default_rng(seed)

    for cat, defects in spec.items():
        for i in range(n_train_good):
            _write_png(root / cat / "train" / "good" / f"{i:03d}.png",
                       rng.integers(0, 255, (h, w, 3), dtype=np.uint8))
        for i in range(n_test_good):
            _write_png(root / cat / "test" / "good" / f"{i:03d}.png",
                       rng.integers(0, 255, (h, w, 3), dtype=np.uint8))
        for dt, n in defects.items():
            for i in range(n):
                mask = np.zeros((h, w), dtype=np.uint8)
                if dt == "speck":
                    mask[20:22, 30:32] = 255
                elif dt == "thread":
                    cv2.line(mask, (5, 40), (w - 6, 10), 255, 2)
                else:
                    cv2.rectangle(mask, (10 + i, 12), (26 + i, 30), 255, -1)
                _write_png(root / cat / "test" / dt / f"{i:03d}.png",
                           rng.integers(0, 255, (h, w, 3), dtype=np.uint8))
                _write_png(root / cat / "ground_truth" / dt / f"{i:03d}_mask.png", mask)
    return root


@pytest.fixture
def mvtec_tree(tmp_path: Path) -> Path:
    return build_mvtec_tree(tmp_path / "mvtec_raw")


@pytest.fixture
def wrapped_tree(tmp_path: Path) -> Path:
    """解压后多套一层包装目录的情况。"""
    outer = tmp_path / "raw"
    build_mvtec_tree(outer / "mvtec_anomaly_detection")
    return outer