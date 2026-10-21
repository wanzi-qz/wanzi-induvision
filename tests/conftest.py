"""构造一个微型但结构完整的合成 MVTec AD 目录树用于测试。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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


# --------------------------------------------------------------------------
# 微型 YOLO-seg 数据集：给 W2（缺陷合成）的单测用。
# 不经过 convert_mvtec_to_yolo，直接按最终产物结构搭，测试更快也更可控。
# --------------------------------------------------------------------------

TINY_SIZE = (96, 72)  # (w, h)

# 图片名 -> [(class_id, 像素多边形)]；没有条目的图是背景（MVTec 的 good 样本）
TINY_LAYOUT = {
    "train": {
        "bg__000": [],
        "bg__001": [],
        "def_a__000": [(0, [(18, 14), (46, 14), (46, 40), (18, 40)])],
        "def_b__000": [(1, [(52, 30), (74, 30), (74, 52), (52, 52)])],
    },
    "val": {
        "val_a__000": [(0, [(10, 10), (30, 10), (30, 30), (10, 30)])],
    },
    "test": {
        "test_good__000": [],
    },
}

TINY_NAMES = {0: "bottle", 1: "toothbrush"}


@dataclass
class TinyYolo:
    """``build_tiny_yolo`` 的返回值：路径 + 写进去的真值，供断言逐一比对。"""

    root: Path
    size: Tuple[int, int]
    names: Dict[int, str]
    expected: Dict[str, List[Tuple[int, np.ndarray]]]  # 图片 stem -> [(cls, 像素多边形)]
    backgrounds: Tuple[str, ...]

    def image(self, split: str, stem: str) -> Path:
        return self.root / "images" / split / f"{stem}.png"

    def label(self, split: str, stem: str) -> Path:
        return self.root / "labels" / split / f"{stem}.txt"


def build_tiny_yolo(
    root: Path,
    *,
    size: Tuple[int, int] = TINY_SIZE,
    seed: int = 0,
    with_defects: bool = True,
) -> TinyYolo:
    """按 ``TINY_LAYOUT`` 搭一个最小但结构完整的 YOLO-seg 数据集。

    ``with_defects=False`` 时 train 里只留背景图，用来测「缺陷库为空」这类错误路径。
    """
    from induvision.yolo_dataset import write_label_file

    w, h = size
    rng = np.random.default_rng(seed)
    expected: Dict[str, List[Tuple[int, np.ndarray]]] = {}
    backgrounds: List[str] = []

    for split, images in TINY_LAYOUT.items():
        for stem, polys in images.items():
            img = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)  # 有纹理，能过 bg_std_min
            _write_png(root / "images" / split / f"{stem}.png", img)
            items = [] if (not with_defects and split == "train") else polys
            if items:
                write_label_file(
                    root / "labels" / split / f"{stem}.txt",
                    [(c, np.asarray(p, dtype=np.int32)) for c, p in items],
                    w, h,
                )
                expected[stem] = [(c, np.asarray(p, dtype=np.int32)) for c, p in items]
            else:
                backgrounds.append(stem)

    root.mkdir(parents=True, exist_ok=True)
    (root / "data.yaml").write_text(
        "path: %s\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n%s"
        % (root.resolve().as_posix(), "".join(f"  {k}: {v}\n" for k, v in sorted(TINY_NAMES.items()))),
        encoding="utf-8",
    )
    rows = ["out_name,split,category,defect_type"]
    for split, images in TINY_LAYOUT.items():
        for stem in images:
            cat = TINY_NAMES[1] if stem.startswith("def_b") else TINY_NAMES[0]
            dt = "broken_large" if expected.get(stem) else "good"
            rows.append(f"{stem}.png,{split},{cat},{dt}")
    (root / "manifest.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")

    return TinyYolo(
        root=root, size=size, names=dict(TINY_NAMES),
        expected=expected, backgrounds=tuple(sorted(backgrounds)),
    )


@pytest.fixture
def tiny_yolo(tmp_path: Path) -> TinyYolo:
    return build_tiny_yolo(tmp_path / "yolo")


def load_tool(name: str):
    """按文件路径加载 ``tools/`` 下的脚本模块（tools 不是包）。"""
    import importlib.util
    import sys

    path = Path(__file__).resolve().parents[1] / "tools" / f"{name}.py"
    assert path.is_file(), f"缺少工具脚本 {path}"
    spec = importlib.util.spec_from_file_location(f"tools_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod
