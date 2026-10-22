#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成 TensorRT INT8 的固定校准清单（默认 ``configs/calib_sample.json``）。

为什么不直接给 ``verify_onnx.py`` 加个 ``--sample-n 500`` 就完事：

1. **校准集不能从 test split 出。** INT8 的动态范围是按校准集统计出来的，
   ``verify_onnx.py --split`` 只有 ``val/test``；拿 test 校准再拿 test 报 mAP，
   等于让评测集参与了一次调参，报出来的"INT8 掉几个点"必然偏乐观——而那正好是
   这一段唯一要量的东西。所以校准清单从 ``train`` split 抽（母体 4490），
   精度评测仍旧只在 test 的那份基准清单上做。
2. **口径必须和 §4.0 的基准清单完全一致**：等间距、只取决于排序后的列表、
   sha256 落进报告。所以这里复用 ``induvision.deploy`` 的 ``sample_paths`` /
   ``write_sample_list``，不另写一套抽样逻辑——两套实现迟早会漂移。

幂等：清单已存在且指纹相同就什么都不做（返回 0）；指纹不同则拒绝覆盖，
换口径必须显式 ``--force``。换清单 = 换校准口径，历史 INT8 数字就不能再直接相减了。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import yaml  # noqa: E402

from induvision.deploy import (  # noqa: E402 - 需要先插好 sys.path
    load_sample_list,
    read_split_images,
    sample_fingerprint,
    sample_paths,
    write_sample_list,
)

#: 校准集默认张数。deploy.md §8 定的是 500：够覆盖 15 类的动态范围，
#: 又不至于让一轮校准跑太久（500 张 / batch 8 = 63 个 batch）。
DEFAULT_CALIB_N = 500


def _git_head() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=15,
        )
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def dataset_root(data_yaml: Path) -> Path:
    """取 ``data.yaml`` 里的 ``path`` 当数据集根；相对串按 data.yaml 所在目录还原。

    和 ``read_split_images`` 保持同一套解析，否则清单里的相对路径和校验时还原出来的
    绝对路径对不上，指纹会莫名变化。
    """
    d = yaml.safe_load(Path(data_yaml).read_text(encoding="utf-8")) or {}
    root = Path(str(d.get("path") or Path(data_yaml).parent))
    if not root.is_absolute():
        root = (Path(data_yaml).parent / root).resolve()
    return root


def class_of(path) -> str:
    """从文件名取类别：``bottle__train__good__000.png`` -> ``bottle``。

    取不到就返回 ``"?"``。校准集最怕"看着 500 张、其实全落在两三个类上"，
    所以生成时必须把类别分布打出来看一眼，不能只看总数。
    """
    head = Path(path).name.split("__", 1)[0]
    return head or "?"


def coverage(paths: Sequence[Path]) -> Dict[str, int]:
    """按类别统计张数（键排序后返回，方便直接落 JSON / 抄进文档）。"""
    c = Counter(class_of(p) for p in paths)
    return {k: c[k] for k in sorted(c)}


def build_list(data_yaml: Path, split: str, n: int) -> Dict[str, object]:
    """列 split -> 等间距抽 n 张 -> 返回 ``{paths, all, cov}``。"""
    allp = read_split_images(data_yaml, split=split)
    if not allp:
        raise ValueError(f"{data_yaml} 的 {split} split 里没有图，无法生成校准清单")
    picked = sample_paths(allp, n)
    return {"paths": picked, "all": allp, "cov": coverage(picked)}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="生成 TensorRT INT8 固定校准清单（等间距、带指纹、可复现）"
    )
    ap.add_argument("--data", default="out/yolo_mvtec15/data.yaml", help="data.yaml")
    ap.add_argument("--split", default="train",
                    help="从哪个 split 抽校准图（默认 train；用 val/test 会污染评测，直接拒）")
    ap.add_argument("--n", type=int, default=DEFAULT_CALIB_N, help="抽多少张")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out", default="configs/calib_sample.json", help="清单落盘位置")
    ap.add_argument("--force", action="store_true",
                    help="已存在且指纹不同时也覆盖（等于宣布换校准口径）")
    ap.add_argument("--dry-run", action="store_true", help="只打印，不写盘")
    args = ap.parse_args(argv)

    def _p(s: str) -> Path:
        q = Path(s)
        return q if q.is_absolute() else (REPO_ROOT / q)

    data_yaml, out = _p(args.data), _p(args.out)
    if not data_yaml.is_file():
        print(f"[err  ] 缺 data.yaml：{data_yaml}")
        return 2
    if args.split in ("val", "test"):
        print(f"[err  ] --split {args.split} 会让校准集与评测集重叠，拒绝执行（校准请用 train）")
        return 2
    if args.n <= 0:
        print("[err  ] --n 要大于 0")
        return 2

    try:
        res = build_list(data_yaml, args.split, args.n)
    except (ValueError, KeyError, FileNotFoundError) as e:
        print(f"[err  ] {e}")
        return 2

    picked, allp, cov = res["paths"], res["all"], res["cov"]
    if len(picked) < args.n:
        print(f"[err  ] {args.split} 只有 {len(allp)} 张，抽不满 {args.n}；换 split 或调小 --n")
        return 2

    root = dataset_root(data_yaml)
    fp_new = sample_fingerprint(picked, root)

    # 已有清单：指纹一致就是同一份口径，不重写（重写只会给 git 造无意义 diff，
    # 因为 created_utc 每次都变）；指纹不同则要人显式确认，不能悄悄换掉。
    if out.is_file():
        try:
            _old, meta = load_sample_list(out, strict=False)
        except (ValueError, FileNotFoundError) as e:
            print(f"[err  ] 已有清单读不动：{e}（确认要重来加 --force）")
            return 3
        old = meta.get("sha256")
        if old == fp_new:
            print(f"[ok   ] 清单已存在且指纹一致，跳过：{out}")
            print(f"        split={meta.get('split')} count={len(picked)} sha256={fp_new}")
            return 0
        if not args.force:
            print(f"[err  ] 清单已存在且指纹不同（旧 {old} / 新 {fp_new}）。")
            print("        覆盖等于换校准口径，历史 INT8 数字不可再直接比较；确认要换加 --force")
            return 3
        print(f"[warn ] --force：覆盖旧清单（旧 sha256={old} -> 新 {fp_new}）")

    print(f"[info ] split={args.split} 母体={len(allp)} 抽取={len(picked)} imgsz={args.imgsz}")
    print(f"[info ] 类别覆盖 {len(cov)} 类：" + ", ".join(f"{k}={v}" for k, v in cov.items()))
    thin = [k for k, v in cov.items() if v < 3]
    if thin:
        print(f"[warn ] 这些类在校准集里不到 3 张，动态范围可能偏：{', '.join(thin)}")
    print(f"[info ] sha256={fp_new}")
    print(f"[info ] 落盘 {out}")
    if args.dry_run:
        print("[dry  ] --dry-run，未写盘")
        return 0

    payload = write_sample_list(
        picked, out, root=root, split=args.split, imgsz=args.imgsz,
        source_total=len(allp), sampling="equidistant", git_head=_git_head(),
    )
    print(f"[ok   ] 已写入 {out}（count={payload['count']} sha256={payload['sha256']}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())