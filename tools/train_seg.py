#!/usr/bin/env python
"""YOLO-seg 训练入口：统一 CLI + 可选 focal loss 注入 + 结果摘要。

用法示例
--------
# 基线（BCE）
python tools/train_seg.py --data out/yolo_bottle_tb_cp/data.yaml --name w1b_baseline --epochs 100

# focal loss 消融组
python tools/train_seg.py --data out/yolo_bottle_tb_cp/data.yaml --name w2_focal \
    --focal-gamma 2.0 --focal-alpha 0.25 --epochs 100

# 只看配置不训练（CI / 自查用）
python tools/train_seg.py --data out/yolo_bottle_tb_cp/data.yaml --dry-run

说明
----
* 不传 ``project``：ultralytics 默认落在 ``runs/segment/<name>``。传 ``project=runs``
  会被拼成 ``runs/segment/runs/<name>``，多一层没用的嵌套（踩过）。
* 8GB 显存建议 imgsz=640 batch=8 amp=True；显存被别的任务占着时先 ``nvidia-smi`` 看一眼。
* 训练前会把本次配置写进 ``<run_dir>/train_args.json``（ultralytics 自己也会存 args.yaml，
  这份多了 focal 参数与 git 信息，方便半年后复现）。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=REPO_ROOT, check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="YOLO-seg 训练（可选 focal loss）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="yolov8s-seg.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="0")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--name", default="induvision")
    ap.add_argument("--project", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--focal-gamma", type=float, default=0.0, help="0 表示关闭 focal，用默认 BCE")
    ap.add_argument("--focal-alpha", type=float, default=0.25)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    cfg = {
        "data": str(Path(args.data).resolve()),
        "model": args.model,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "patience": args.patience,
        "name": args.name,
        "project": args.project,
        "seed": args.seed,
        "amp": not args.no_amp,
        "focal_gamma": args.focal_gamma,
        "focal_alpha": args.focal_alpha if args.focal_gamma > 0 else None,
        "git_head": _git_head(),
    }
    print("[cfg  ] " + json.dumps(cfg, ensure_ascii=False))
    if args.dry_run:
        print("[dry  ] 未训练")
        return 0

    from ultralytics import YOLO  # 延迟导入：--dry-run 不需要 torch

    model = YOLO(args.model)
    if args.focal_gamma > 0:
        from induvision.focal import install_focal_loss

        install_focal_loss(model, gamma=args.focal_gamma, alpha=args.focal_alpha)
        print(f"[focal] gamma={args.focal_gamma} alpha={args.focal_alpha} 已注入分类分支")

    t0 = time.time()
    kwargs = dict(
        data=cfg["data"], epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        device=args.device, workers=args.workers, patience=args.patience,
        name=args.name, seed=args.seed, amp=cfg["amp"], exist_ok=True, verbose=True,
    )
    if args.project:
        kwargs["project"] = args.project
    if args.resume:
        results = model.train(resume=True)
    else:
        results = model.train(**kwargs)
    dt = time.time() - t0

    run_dir = Path(model.trainer.save_dir)
    extra = dict(cfg)
    extra["seconds"] = round(dt, 1)
    extra["run_dir"] = str(run_dir)
    (run_dir / "train_args.json").write_text(json.dumps(extra, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    names = list(results.names.values()) if isinstance(results.names, dict) else list(results.names)
    print(f"\n[done ] {dt / 60:.1f} min -> {run_dir}")
    print(f"{'metric':<22}{'value':>10}")
    for key in ("metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)",
                "metrics/precision(M)", "metrics/recall(M)", "metrics/mAP50(M)", "metrics/mAP50-95(M)"):
        if key in results.results_dict:
            print(f"{key:<22}{results.results_dict[key]:>10.4f}")
    print("[note ] 写简历前核对：mask mAP50-95 的 val 集只有十几张图时数字不可用，"
          "见 docs/baseline.md 的口径说明")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())