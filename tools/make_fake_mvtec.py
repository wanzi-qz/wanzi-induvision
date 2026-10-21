#!/usr/bin/env python
"""生成一棵结构完整的合成 MVTec AD 目录树（不含真实数据）。

用途
----
CI 与本地自测需要一个"长得像 MVTec"的目录来跑通
``download_mvtec.py --verify-only`` 和 ``convert_mvtec_to_yolo.py`` 的完整链路，
但不可能把 4.9GB 真实数据集放进流水线。这个脚本用随机噪声图 + 程序化掩码
造出微型同构目录，秒级生成，无任何许可问题。

掩码形态刻意做成三种，用来触发转换管线的不同分支：
  * ``speck``  2x2 像素小斑点   -> 应被 ``--min-area`` 过滤并告警
  * ``thread`` 细长斜条         -> approxPolyDP 顶点数偏多
  * 其它       实心矩形         -> 常规路径

与 ``tests/conftest.py::build_mvtec_tree`` 同源，但这里不依赖 pytest，
可以单独当命令行工具用。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

#: 默认两类，覆盖"多缺陷类型"和"单一缺陷类型"两种情况
DEFAULT_SPEC: Dict[str, Dict[str, int]] = {
    "bottle": {"broken_large": 5, "broken_small": 4},
    "screw": {"thread": 3, "speck": 2},
}


def _write_png(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), arr):
        raise RuntimeError(f"写入失败: {path}")


def build_fake_mvtec(
    root: Path,
    *,
    spec: Optional[Dict[str, Dict[str, int]]] = None,
    n_train_good: int = 6,
    n_test_good: int = 4,
    size: Sequence[int] = (96, 72),
    seed: int = 0,
) -> Path:
    """在 ``root`` 下生成 ``<category>/{train,test,ground_truth}/...``。"""
    spec = dict(spec or DEFAULT_SPEC)
    w, h = int(size[0]), int(size[1])
    rng = np.random.default_rng(seed)
    n_files = 0

    for cat, defects in spec.items():
        for i in range(n_train_good):
            _write_png(
                root / cat / "train" / "good" / f"{i:03d}.png",
                rng.integers(0, 255, (h, w, 3), dtype=np.uint8),
            )
            n_files += 1
        for i in range(n_test_good):
            _write_png(
                root / cat / "test" / "good" / f"{i:03d}.png",
                rng.integers(0, 255, (h, w, 3), dtype=np.uint8),
            )
            n_files += 1
        for dt, n in defects.items():
            for i in range(n):
                mask = np.zeros((h, w), dtype=np.uint8)
                if dt == "speck":
                    mask[20:22, 30:32] = 255
                elif dt == "thread":
                    cv2.line(mask, (6, h - 8), (w - 7, 10), 255, 2)
                else:
                    cv2.rectangle(mask, (12 + i, 14), (34 + i, 40), 255, -1)
                _write_png(
                    root / cat / "test" / dt / f"{i:03d}.png",
                    rng.integers(0, 255, (h, w, 3), dtype=np.uint8),
                )
                _write_png(root / cat / "ground_truth" / dt / f"{i:03d}_mask.png", mask)
                n_files += 2
    print(f"[fake] {root} -> {len(spec)} 类 / {n_files} 文件")
    return root


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="生成合成 MVTec AD 目录树")
    ap.add_argument("--out", default=str(REPO_ROOT / "ci-out" / "fake_mvtec"))
    ap.add_argument("--categories", nargs="*", default=None, help="只生成指定类别（默认 bottle screw）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--width", type=int, default=96)
    ap.add_argument("--height", type=int, default=72)
    args = ap.parse_args(argv)

    spec = DEFAULT_SPEC
    if args.categories:
        unknown = [c for c in args.categories if c not in DEFAULT_SPEC]
        if unknown:
            print(f"[fake] 未知类别 {unknown}，可用: {sorted(DEFAULT_SPEC)}")
            return 2
        spec = {c: DEFAULT_SPEC[c] for c in args.categories}

    out = Path(args.out).expanduser()
    build_fake_mvtec(out, spec=spec, seed=args.seed, size=(args.width, args.height))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())