#!/usr/bin/env python
"""消融结果汇总：把 runs/segment/* 的 results.csv 拼成一张能直接贴进文档的表。

为什么需要这个脚本
------------------
手抄指标一定会抄错，而且消融的核心是「只动一个变量」——这件事得**能被验证**，
不能靠记忆。本脚本做两件事：

1. 从 ``results.csv`` 取每组的 best epoch（按指定指标）与末轮值，输出 Markdown 表；
2. 从 ``train_args.json``（本仓库训练入口写的）或 ultralytics 的 ``args.yaml`` 读回
   seed / imgsz / batch / epochs / patience / amp，做**协议一致性检查**：
   谁和多数组不一致就点名，避免"以为是单变量对比、其实 batch 都不一样"；
3. 回读各组数据集 ``dataset_stats.json`` / ``synth_report.json`` 里的转换环境指纹，查出"两份数据集是不同
   conda 环境转出来的"这种更隐蔽的污染（cv2 4.10 与 5.0 的轮廓结果差约 1 像素，
   会让本该完全一致的 val/test 悄悄不同）。

用法
----
    python tools/summarize_ablation.py                     # 扫 runs/segment 全部
    python tools/summarize_ablation.py --names b0_base b1_cp
    python tools/summarize_ablation.py --markdown          # 贴进 docs/baseline.md
    python tools/summarize_ablation.py --metric metrics/mAP50-95(B) --strict

``--strict`` 在协议不一致或某个 run 缺 results.csv 时返回非零退出码（CI 可用）。
只依赖标准库 + PyYAML，不需要 torch，不联网。
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]

PROTOCOL_KEYS = ("seed", "imgsz", "batch", "epochs", "patience", "amp")

METRIC_COLS = (
    "metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)",
    "metrics/precision(M)", "metrics/recall(M)", "metrics/mAP50(M)", "metrics/mAP50-95(M)",
)


def _to_float(v: object) -> Optional[float]:
    try:
        return float(str(v).strip())  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def read_results(csv_path: Path) -> List[Dict[str, float]]:
    """读 results.csv，返回按 epoch 升序的记录（键去掉首尾空格）。"""
    with csv_path.open(encoding="utf-8", newline="") as fh:
        rows = [{(k or "").strip(): v for k, v in row.items()} for row in csv.DictReader(fh)]
    out = []
    for r in rows:
        ep = _to_float(r.get("epoch"))
        if ep is None:
            continue
        rec: Dict[str, float] = {"epoch": ep}
        for c in METRIC_COLS:
            val = _to_float(r.get(c))
            if val is not None:
                rec[c] = val
        t = _to_float(r.get("time"))
        if t is not None:
            rec["_time"] = t
        out.append(rec)
    out.sort(key=lambda r: r["epoch"])
    return out


def read_config(run_dir: Path) -> Dict[str, object]:
    """优先读 train_args.json（含 focal 参数与 git HEAD），退回 ultralytics 的 args.yaml。"""
    tj = run_dir / "train_args.json"
    if tj.exists():
        try:
            cfg = json.loads(tj.read_text(encoding="utf-8"))
            cfg["_source"] = "train_args.json"
            return cfg
        except (OSError, json.JSONDecodeError):
            pass
    ay = run_dir / "args.yaml"
    if ay.exists():
        try:
            import yaml  # 延迟导入：只有落到 args.yaml 分支才需要

            cfg = yaml.safe_load(ay.read_text(encoding="utf-8")) or {}
            cfg["_source"] = "args.yaml"
            return cfg
        except Exception:
            pass
    return {"_source": "missing"}


def dataset_name(cfg: Dict[str, object]) -> str:
    d = str(cfg.get("data") or "")
    if not d:
        return "?"
    p = Path(d)
    return p.parent.name if p.name.endswith((".yaml", ".yml")) else p.name


def focal_str(cfg: Dict[str, object]) -> str:
    g = _to_float(cfg.get("focal_gamma"))
    if g is None:  # args.yaml 里没有这个键，说明没走 focal 注入
        return "关"
    if g <= 0:
        return "关"
    a = _to_float(cfg.get("focal_alpha"))
    return f"γ={g:g}" + (f" α={a:g}" if a is not None else "")


def summarize_run(run_dir: Path, metric: str) -> Optional[Dict[str, object]]:
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        return None
    rows = read_results(csv_path)
    if not rows:
        return None
    cfg = read_config(run_dir)

    scored = [r for r in rows if metric in r]
    best = max(scored, key=lambda r: r[metric]) if scored else rows[-1]
    last = rows[-1]
    rec: Dict[str, object] = {
        "name": run_dir.name,
        "dir": run_dir,
        "cfg": cfg,
        "dataset": dataset_name(cfg),
        "focal": focal_str(cfg),
        "epochs_done": int(last["epoch"]),
        "epochs_plan": _to_float(cfg.get("epochs")),
        "best_epoch": int(best["epoch"]),
        "last": last,
        "best": best,
        "seconds": _to_float(cfg.get("seconds")),
        "protocol": tuple(str(cfg.get(k)) for k in PROTOCOL_KEYS),
        "cfg_source": cfg.get("_source", "?"),
    }
    return rec


ENV_KEYS = ("python", "numpy", "cv2")

# 转换器写 dataset_stats.json；Copy-Paste 合成不写它，指纹在 synth_report.json 里
ENV_FILES = ("dataset_stats.json", "synth_report.json")


def dataset_env(cfg: Dict[str, object]) -> Optional[Dict[str, str]]:
    """从 run 的 ``data`` 路径回溯数据集目录，按 ``ENV_FILES`` 顺序读 env 指纹。

    读不到（文件不存在 / JSON 坏了 / 老数据集没这个字段）一律返回 ``None``，
    由调用方决定是提示还是告警。
    """
    d = str(cfg.get("data") or "")
    if not d:
        return None
    path = Path(d)
    ds_dir = path.parent if path.suffix.lower() in (".yaml", ".yml") else path
    for fname in ENV_FILES:
        f = ds_dir / fname
        if not f.exists():
            continue
        try:
            env = json.loads(f.read_text(encoding="utf-8")).get("env")
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(env, dict):
            return env
    return None


def check_env(recs: Sequence[Dict[str, object]]) -> tuple:
    """交叉核对各组数据集的转换环境。

    返回 ``(warnings, notes)``：
      * warnings —— 出现了**两个以上不同的已知指纹**，这会直接破坏单变量前提，
        ``--strict`` 下应当失败；
      * notes —— 只是有些 run 的数据集没有指纹（老数据集），不判失败。
    """
    per_run = {str(r["name"]): dataset_env(r.get("cfg") or {}) for r in recs}
    known = {k: v for k, v in per_run.items() if v}
    unknown = sorted(k for k, v in per_run.items() if not v)

    variants: Dict[tuple, List[str]] = {}
    for name, env in known.items():
        key = tuple((k, str(env.get(k, "?"))) for k in ENV_KEYS)
        variants.setdefault(key, []).append(name)

    warns: List[str] = []
    if len(variants) > 1:
        warns.append("数据集转换环境不一致：不同 cv2/numpy 下 findContours/approxPolyDP")
        warns.append("  会产出相差约 1 像素的标签，val/test 一旦不同，消融就不再是单变量对比")
        for key, names in sorted(variants.items(), key=lambda kv: sorted(kv[1])):
            warns.append(f"  {dict(key)} <- {', '.join(sorted(names))}")
        warns.append("  受影响 run：" + ", ".join(sorted(known)))

    notes: List[str] = []
    if unknown:
        notes.append("以下 run 的数据集没有 env 指纹（早于该字段引入，或 dataset_stats.json 缺失）："
                     + ", ".join(unknown))
        notes.append("  重跑 tools/convert_mvtec_to_yolo.py（合成集再加 tools/synthesize_defects.py）即可补上；"
                     "同环境同参数产物逐字节相同")
    return warns, notes


def check_protocol(recs: Sequence[Dict[str, object]]) -> List[str]:
    """返回协议不一致的告警行；全一致则返回空列表。"""
    if len(recs) < 2:
        return []
    votes = Counter(r["protocol"] for r in recs)
    majority, _ = votes.most_common(1)[0]
    warns = []
    if len(votes) > 1:
        keys = ", ".join(PROTOCOL_KEYS)
        warns.append(f"协议不一致（比对字段：{keys}）")
        warns.append(f"  多数组 = {dict(zip(PROTOCOL_KEYS, majority))}")
        for r in recs:
            if r["protocol"] != majority:
                warns.append(f"  例外：{r['name']} = {dict(zip(PROTOCOL_KEYS, r['protocol']))}")
    return warns


def _fmt(v: object, nd: int = 4) -> str:
    f = v if isinstance(v, float) else _to_float(v)
    return f"{f:.{nd}f}" if f is not None else "—"


def to_markdown(recs: Sequence[Dict[str, object]], metric: str) -> str:
    head = ("| 组 | 数据集 | focal | 实跑/计划 epoch | best ep | mask P | mask R | "
            "mask mAP50 | **mask mAP50-95** | box mAP50-95 | 时长 |")
    sep = "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    lines = [head, sep]
    for r in sorted(recs, key=lambda x: str(x["name"])):
        b = r["best"]
        plan = int(r["epochs_plan"]) if r["epochs_plan"] else "?"
        secs = r["seconds"]
        dur = f"{secs / 60:.1f} min" if secs else "—"
        star = " ⬅" if metric == "metrics/mAP50-95(M)" else ""
        lines.append(
            f"| {r['name']} | `{r['dataset']}` | {r['focal']} | {r['epochs_done']}/{plan} "
            f"| {r['best_epoch']} | {_fmt(b.get('metrics/precision(M)'))} "
            f"| {_fmt(b.get('metrics/recall(M)'))} | {_fmt(b.get('metrics/mAP50(M)'))} "
            f"| **{_fmt(b.get('metrics/mAP50-95(M)'))}**{star} "
            f"| {_fmt(b.get('metrics/mAP50-95(B)'))} | {dur} |"
        )
    return "\n".join(lines)


def to_text(recs: Sequence[Dict[str, object]], metric: str) -> str:
    lines = []
    for r in sorted(recs, key=lambda x: str(x["name"])):
        b, l = r["best"], r["last"]
        lines.append(f"[{r['name']}] data={r['dataset']} focal={r['focal']} cfg_from={r['cfg_source']}")
        lines.append(f"    epoch {r['epochs_done']}/{int(r['epochs_plan']) if r['epochs_plan'] else '?'}"
                     f"  best@{r['best_epoch']}  {metric}={_fmt(b.get(metric))}")
        lines.append(f"    best : mask P/R={_fmt(b.get('metrics/precision(M)'))}/{_fmt(b.get('metrics/recall(M)'))}"
                     f"  mAP50={_fmt(b.get('metrics/mAP50(M)'))}  mAP50-95={_fmt(b.get('metrics/mAP50-95(M)'))}")
        lines.append(f"    last : mask P/R={_fmt(l.get('metrics/precision(M)'))}/{_fmt(l.get('metrics/recall(M)'))}"
                     f"  mAP50={_fmt(l.get('metrics/mAP50(M)'))}  mAP50-95={_fmt(l.get('metrics/mAP50-95(M)'))}")
        lines.append(f"    box  : mAP50={_fmt(b.get('metrics/mAP50(B)'))}  mAP50-95={_fmt(b.get('metrics/mAP50-95(B)'))}")
    return "\n".join(lines)


def collect(runs_root: Path, names: Optional[Sequence[str]], metric: str,
            quiet: bool = False) -> List[Dict[str, object]]:
    if not runs_root.exists():
        raise SystemExit(f"[err ] 找不到 runs 目录：{runs_root}")
    recs, missing = [], []
    targets = [runs_root / n for n in names] if names else sorted(
        d for d in runs_root.iterdir() if d.is_dir())
    for d in targets:
        rec = summarize_run(d, metric) if d.exists() else None
        if rec is None:
            missing.append(d.name)
        else:
            recs.append(rec)
    if missing and not quiet:
        print(f"[warn] 以下 run 没有可读的 results.csv，已跳过：{', '.join(missing)}", file=sys.stderr)
    return recs


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="汇总消融训练结果并校验单变量协议",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--runs-root", default=str(REPO_ROOT / "runs" / "segment"))
    ap.add_argument("--names", nargs="*", default=None, help="只看这些 run 名；不传则扫全部")
    ap.add_argument("--metric", default="metrics/mAP50-95(M)", choices=list(METRIC_COLS))
    ap.add_argument("--markdown", action="store_true", help="输出可直接贴进 docs/baseline.md 的表")
    ap.add_argument("--strict", action="store_true", help="协议或转换环境不一致时返回非零")
    ap.add_argument("--no-env-check", action="store_true",
                    help="跳过数据集转换环境指纹核对")
    args = ap.parse_args(argv)

    recs = collect(Path(args.runs_root), args.names, args.metric)
    if not recs:
        print("[err ] 没有任何可汇总的 run", file=sys.stderr)
        return 2

    warns = check_protocol(recs)
    env_warns, env_notes = ([], []) if args.no_env_check else check_env(recs)
    warns = warns + env_warns

    if args.markdown:
        print(to_markdown(recs, args.metric))
    else:
        print(f"best-epoch 选取指标：{args.metric}")
        print(to_text(recs, args.metric))
    if warns:
        print()
        for w in warns:
            print("[warn] " + w if not w.startswith("  ") else w)
    else:
        print(f"\n[ok  ] {len(recs)} 组协议一致：{dict(zip(PROTOCOL_KEYS, recs[0]['protocol']))}")
    for n in env_notes:
        print("[note] " + n if not n.startswith("  ") else n)

    print("\n[note] 2 类子集的 val 只有 29 图 / 16 实例，这些绝对值只能读趋势，"
          "不能写简历（见 docs/baseline.md §3 读数警告）")
    return 1 if (args.strict and warns) else 0


if __name__ == "__main__":
    raise SystemExit(main())
