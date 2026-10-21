#!/usr/bin/env python
"""在指定 split（默认 test）上评测一个训练好的分割权重，输出逐类 AP 表 + JSON。

为什么要单独一个脚本
--------------------
``train_seg.py`` 训练时只在 **val** 上算指标，而 val 参与了早停和选 best.pt，
拿它当最终成绩等于在选过的数据上报告结果。简历/报告里要写的那个数应当来自
**从没参与过任何决策的 test split**。这个脚本就是补这一步：

* 逐类 P / R / mAP50 / mAP50-95（box 与 mask 各一份）——15 类里哪几类拖后腿一眼可见；
* 落一份 ``eval_<split>.json``，含权重路径、数据集环境指纹、git HEAD、耗时，
  半年后能确认"这张表是哪个权重、哪份数据跑出来的"；
* ``--markdown`` 直接吐可贴进 ``docs/baseline.md`` 的表格。

用法
----
.. code-block:: powershell

    conda activate induvision
    python tools/eval_seg.py --weights runs/segment/baseline15/weights/best.pt `
        --data out/yolo_mvtec15/data.yaml --split test --name baseline15 --markdown

注意 ``split=test`` 需要 ``data.yaml`` 里有 ``test:`` 键（本仓库的转换器会写）。
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _git_head() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=15,
        )
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except Exception:  # noqa: BLE001 - git 不在 PATH 也不该让评测失败
        return "unknown"


def _env_fingerprint() -> Dict[str, str]:
    fp = {"python": platform.python_version(), "platform": platform.platform()}
    for mod in ("numpy", "cv2", "torch", "ultralytics"):
        try:
            m = __import__(mod)
            fp[mod] = getattr(m, "__version__", "?")
        except Exception:  # noqa: BLE001
            fp[mod] = "absent"
    try:
        import torch  # noqa: PLC0415

        fp["cuda"] = torch.version.cuda or "none"
        fp["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001
        pass
    return fp


def _dataset_env(data_yaml: Path) -> Optional[Dict[str, str]]:
    """读数据集目录下的 env 指纹，用来发现"评测集和训练集不是同一个环境转出来的"。"""
    for cand in (data_yaml.parent / "dataset_stats.json", data_yaml.parent / "synth_report.json"):
        if cand.exists():
            try:
                return json.loads(cand.read_text(encoding="utf-8")).get("env")
            except Exception:  # noqa: BLE001
                return None
    return None


def _per_class(metric, names: Sequence[str]) -> List[Dict[str, object]]:
    """把 ultralytics 的 Metric 拆成逐类字典；某类在 split 里没有实例时标 None。"""
    rows: List[Dict[str, object]] = []
    have = set(int(i) for i in getattr(metric, "ap_class_index", []))
    for ci, nm in enumerate(names):
        if ci in have:
            p, r, ap50, ap = metric.class_result(ci)
            rows.append({"class_id": ci, "class": nm, "P": float(p), "R": float(r),
                         "mAP50": float(ap50), "mAP50-95": float(ap)})
        else:
            rows.append({"class_id": ci, "class": nm, "P": None, "R": None,
                         "mAP50": None, "mAP50-95": None})
    return rows


def _md_table(rows: Sequence[Dict[str, object]], title: str) -> List[str]:
    out = [f"**{title}**", "", "| 类 | P | R | mAP50 | mAP50-95 |", "| --- | --- | --- | --- | --- |"]
    for r in rows:
        fmt = lambda v: "—" if v is None else f"{v:.4f}"  # noqa: E731
        out.append(f"| {r['class']} | {fmt(r['P'])} | {fmt(r['R'])} | {fmt(r['mAP50'])} | {fmt(r['mAP50-95'])} |")
    out.append("")
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="在 test/val split 上评测分割权重，输出逐类 AP")
    ap.add_argument("--weights", required=True, help="best.pt 路径")
    ap.add_argument("--data", required=True, help="data.yaml 路径")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--name", default=None, help="run 名，决定 JSON 落在哪个 runs/segment/<name>/ 下")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="0")
    ap.add_argument("--conf", type=float, default=None, help="默认用 ultralytics 评测口径，不手动指定")
    ap.add_argument("--markdown", action="store_true", help="打印可贴进文档的表格")
    ap.add_argument("--json-out", default=None, help="JSON 输出路径，默认 runs/segment/<name>/eval_<split>.json")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    w = Path(args.weights)
    d = Path(args.data)
    for p, what in ((w, "权重"), (d, "data.yaml")):
        if not p.exists():
            print(f"[err  ] {what}不存在：{p}")
            return 2
    if not w.is_absolute():
        w = (REPO_ROOT / w).resolve()
    if not d.is_absolute():
        d = (REPO_ROOT / d).resolve()

    # 没显式给 --name 时，从 runs/segment/<run>/weights/best.pt 反推 run 名
    if args.name:
        name = args.name
    elif w.parent.name == "weights":
        name = w.parents[1].name
    else:
        name = "eval"
    run_dir = REPO_ROOT / "runs" / "segment" / name
    out_json = Path(args.json_out) if args.json_out else run_dir / f"eval_{args.split}.json"
    if not out_json.is_absolute():
        out_json = (REPO_ROOT / out_json).resolve()

    print(f"[cfg  ] weights={w}")
    print(f"[cfg  ] data={d} split={args.split} imgsz={args.imgsz} batch={args.batch} device={args.device}")
    print(f"[cfg  ] json -> {out_json}")
    if args.dry_run:
        print("[dry  ] 未评测")
        return 0

    from ultralytics import YOLO  # 延迟导入：--dry-run 不需要 torch

    t0 = time.time()
    model = YOLO(str(w))
    kwargs = dict(data=str(d), split=args.split, imgsz=args.imgsz, batch=args.batch,
                  device=args.device, project=str(REPO_ROOT / "runs" / "segment"),
                  name=f"{name}_eval_{args.split}", exist_ok=True, plots=True, verbose=True)
    if args.conf is not None:
        kwargs["conf"] = args.conf
    metrics = model.val(**kwargs)
    dt = time.time() - t0

    names_map = metrics.names
    names = [names_map[i] for i in sorted(names_map)] if isinstance(names_map, dict) else list(names_map)
    box_rows = _per_class(metrics.box, names)
    mask_rows = _per_class(metrics.mask, names)

    def _g(obj, key):
        v = getattr(obj, key, None)
        return None if v is None else float(v)

    summary = {
        "box": {"P": _g(metrics.box, "mp"), "R": _g(metrics.box, "mr"),
                "mAP50": _g(metrics.box, "map50"), "mAP50-95": _g(metrics.box, "map")},
        "mask": {"P": _g(metrics.mask, "mp"), "R": _g(metrics.mask, "mr"),
                 "mAP50": _g(metrics.mask, "map50"), "mAP50-95": _g(metrics.mask, "map")},
    }

    payload = {
        "weights": str(w),
        "data": str(d),
        "split": args.split,
        "run": name,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "conf": args.conf,
        "seconds": round(dt, 1),
        "git_head": _git_head(),
        "env": _env_fingerprint(),
        "dataset_env": _dataset_env(d),
        "summary": summary,
        "per_class_box": box_rows,
        "per_class_mask": mask_rows,
    }
    try:
        rd = getattr(metrics, "results_dict", None)
        if rd:
            payload["results_dict"] = {str(k): (float(v) if isinstance(v, (int, float)) else str(v))
                                       for k, v in dict(rd).items()}
    except Exception:  # noqa: BLE001
        pass

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print()
    print(f"[done ] {dt:.1f} s -> {out_json}")
    for k in ("box", "mask"):
        s = summary[k]
        print(f"[{k:<5}] P={s['P']:.4f} R={s['R']:.4f} mAP50={s['mAP50']:.4f} mAP50-95={s['mAP50-95']:.4f}")

    if args.markdown:
        lines: List[str] = []
        lines += _md_table(mask_rows, f"mask 逐类（split={args.split}）")
        lines += _md_table(box_rows, f"box 逐类（split={args.split}）")
        print("\n".join(lines))

    # 单类 AP 全 None 说明这个 split 里一个缺陷实例都没有，八成是 data.yaml 的 test 键指错了
    if all(r["mAP50-95"] is None for r in mask_rows):
        print("[warn ] 逐类 AP 全空：该 split 里没有可评测实例，检查 data.yaml 的 test 键")
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
