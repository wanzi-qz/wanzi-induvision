#!/usr/bin/env python
"""把 best.pt 导成 ONNX，并对产物做静态体检（opset / 动态轴 / 输出张量 / checker）。

为什么要单独一个脚本
--------------------
``model.export(format="onnx")`` 一行就能导出，但它把文件写在 best.pt 旁边、
用 ultralytics 当时的默认 opset，也不检查产物长什么样。W3 后面三档量化
（FP32/FP16/INT8）全都从这个 .onnx 出发：opset 太低 trtexec 直接不认，
动态轴没开则 batch 恒为 1，压测出来的吞吐没有意义，化简没做则图里一堆
冗余节点、量化时校准反而更容易偏。所以导出这一步把口径钉死：

* **opset 17**：TensorRT 10.x 的舒适区，也覆盖到 GridSample / Resize 这些分割模型要的算子；
* **dynamic batch**：输入第 0 维导成符号维，后面才能做 batch>1 的流水线与压测；
* **onnxslim 化简**：消掉常量折叠后剩下的冗余节点，图更小也更好读；
* 导完立刻用 ``onnx.checker`` + 图遍历做体检，把输入输出名字、形状、动态维、
  文件大小、opset、producer 全落进 JSON —— 半年后能确认这个 .onnx 是哪个权重、
  哪套环境、哪次提交导出来的。

出问题要在导出阶段就暴露，而不是等 trtexec 报一串算子不支持再回头查。

用法
----
.. code-block:: powershell

    conda activate induvision
    # 默认：FP32 + opset17 + dynamic + simplify，产物落 weights/<run>_fp32.onnx
    python tools/export_onnx.py --weights runs/segment/baseline15/weights/best.pt --markdown

    # FP16 权重版（注意：TensorRT 的 FP16 档不需要它，trtexec 自己会降精度；
    # 这份是给 ORT 侧对比 / 显存受限场景用的）
    python tools/export_onnx.py --weights ... --half --tag fp16

    # 只看配置不导出（CI / 自查）
    python tools/export_onnx.py --weights ... --dry-run

产物本身不入库（``weights/``、``*.onnx`` 都在 .gitignore 里：体积大 + ultralytics 是 AGPL-3.0），
入库的是这个脚本、它的单测，和 JSON 里那些数字被誊进 docs/deploy.md 的部分。
"""
from __future__ import annotations

import argparse
import json
import platform
import shutil
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
    except Exception:  # noqa: BLE001 - git 不在 PATH 也不该让导出失败
        return "unknown"


def _env_fingerprint() -> Dict[str, str]:
    fp = {"python": platform.python_version(), "platform": platform.platform()}
    for mod in ("numpy", "cv2", "torch", "ultralytics", "onnx", "onnxruntime", "onnxslim"):
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
    try:
        import onnxruntime as ort  # noqa: PLC0415

        fp["ort_providers"] = ",".join(ort.get_available_providers())
    except Exception:  # noqa: BLE001
        pass
    return fp


def _infer_run_name(weights: Path, explicit: Optional[str]) -> str:
    """没给 --name 时，从 ``runs/segment/<run>/weights/best.pt`` 反推 run 名。"""
    if explicit:
        return explicit
    if weights.parent.name == "weights":
        return weights.parents[1].name
    return weights.stem


def default_out_path(run: str, tag: str) -> Path:
    """导出产物的默认落点：``weights/<run>_<tag>.onnx``（gitignore 内，不入库）。"""
    return REPO_ROOT / "weights" / f"{run}_{tag}.onnx"


def _dim_to_str(dim) -> str:
    """把 onnx 的 dim 翻成可读字符串：动态维显示符号名，静态维显示数字。"""
    p = getattr(dim, "dim_param", "")
    if p:
        return str(p)
    v = getattr(dim, "dim_value", 0)
    return str(v) if v else "?"


def inspect_onnx(path: Path) -> Dict[str, object]:
    """读 ONNX 图，返回可入库的静态信息（不做推理）。

    ``load_external_data=False`` + ``check_model`` 分开调：外部数据文件缺失时
    也能先把图结构读出来，方便定位是"图有问题"还是"数据文件没跟上"。
    """
    import onnx  # 延迟导入：--dry-run 不需要

    model = onnx.load(str(path), load_external_data=False)
    g = model.graph

    def _io(vi) -> Dict[str, object]:
        t = vi.type.tensor_type
        shape = [_dim_to_str(d) for d in t.shape.dim]
        return {"name": vi.name, "shape": shape,
                "dynamic_axes": [i for i, s in enumerate(shape) if not s.isdigit()]}

    inputs = [_io(i) for i in g.input]
    outputs = [_io(o) for o in g.output]
    opsets = {op.domain or "ai.onnx": op.version for op in model.opset_import}
    batch_dynamic = bool(inputs and inputs[0]["dynamic_axes"] and 0 in inputs[0]["dynamic_axes"])
    return {
        "ir_version": int(model.ir_version),
        "producer": f"{model.producer_name} {model.producer_version}".strip(),
        "opsets": opsets,
        "inputs": inputs,
        "outputs": outputs,
        "n_nodes": len(g.node),
        "op_types": sorted({n.op_type for n in g.node}),
        "batch_dynamic": batch_dynamic,
        "size_bytes": path.stat().st_size,
        "size_mb": round(path.stat().st_size / 1024 / 1024, 2),
    }


def check_onnx(path: Path) -> Dict[str, object]:
    """跑 ``onnx.checker``，通过返回 ``{"ok": True}``，失败带回错误文本。"""
    import onnx  # 延迟导入

    try:
        model = onnx.load(str(path))
        onnx.checker.check_model(model, full_check=False)
        return {"ok": True, "error": None}
    except Exception as e:  # noqa: BLE001 - checker 的异常类型跨版本不稳，统一收口
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


def export(weights: Path, *, opset: int, imgsz: int, dynamic: bool, simplify: bool,
           half: bool, device: str) -> str:
    """调 ultralytics 导出，返回它实际写出的 .onnx 路径。"""
    from ultralytics import YOLO  # 延迟导入：--dry-run 不需要 torch

    model = YOLO(str(weights))
    out = model.export(format="onnx", opset=opset, imgsz=imgsz, dynamic=dynamic,
                       simplify=simplify, half=half, device=device)
    return str(out)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="导出 ONNX 并做静态体检（opset / 动态轴 / checker）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--weights", default="runs/segment/baseline15/weights/best.pt", help="best.pt 路径")
    ap.add_argument("--tag", default="fp32", help="产物文件名后缀，如 fp32 / fp16")
    ap.add_argument("--out", default=None, help="产物路径，默认 weights/<run>_<tag>.onnx")
    ap.add_argument("--name", default=None, help="run 名，决定 JSON 落点；默认从权重路径反推")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--no-dynamic", dest="dynamic", action="store_false", help="关掉动态 batch")
    ap.add_argument("--no-simplify", dest="simplify", action="store_false", help="跳过 onnxslim 化简")
    ap.add_argument("--half", action="store_true", help="导 FP16 权重版")
    ap.add_argument("--device", default="0", help="导出时用的设备；half 建议 GPU")
    ap.add_argument("--json-out", default=None, help="JSON 输出路径，默认 runs/segment/<name>/export_onnx_<tag>.json")
    ap.add_argument("--markdown", action="store_true", help="打印可贴进 docs/deploy.md 的表格")
    ap.add_argument("--dry-run", action="store_true")
    ap.set_defaults(dynamic=True, simplify=True)
    args = ap.parse_args(argv)

    w = Path(args.weights)
    if not w.exists():
        print(f"[err  ] 权重不存在：{w}")
        return 2
    if not w.is_absolute():
        w = (REPO_ROOT / w).resolve()

    run = _infer_run_name(w, args.name)
    out_path = Path(args.out) if args.out else default_out_path(run, args.tag)
    if not out_path.is_absolute():
        out_path = (REPO_ROOT / out_path).resolve()
    json_out = Path(args.json_out) if args.json_out else (
        REPO_ROOT / "runs" / "segment" / run / f"export_onnx_{args.tag}.json")
    if not json_out.is_absolute():
        json_out = (REPO_ROOT / json_out).resolve()

    cfg = {
        "weights": str(w), "run": run, "tag": args.tag, "onnx_out": str(out_path),
        "opset": args.opset, "imgsz": args.imgsz, "dynamic": args.dynamic,
        "simplify": args.simplify, "half": args.half, "device": args.device,
        "json_out": str(json_out),
    }
    print("[cfg  ] " + json.dumps(cfg, ensure_ascii=False))
    if args.dry_run:
        print("[dry  ] 未导出")
        return 0

    t0 = time.time()
    produced = export(w, opset=args.opset, imgsz=args.imgsz, dynamic=args.dynamic,
                      simplify=args.simplify, half=args.half, device=args.device)
    dt_export = time.time() - t0
    src = Path(produced)
    if not src.exists():
        print(f"[err  ] ultralytics 说导出成功但找不到文件：{src}")
        return 3
    if src.resolve() != out_path.resolve():
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(src), str(out_path))
    print(f"[ok   ] 导出 {dt_export:.1f} s -> {src}")
    if src.resolve() != out_path.resolve():
        print(f"[ok   ] 已复制到稳定路径 -> {out_path}")

    t1 = time.time()
    graph = inspect_onnx(out_path)
    chk = check_onnx(out_path)
    dt_check = time.time() - t1

    payload = {
        "run": run, "tag": args.tag, "weights": str(w), "onnx": str(out_path),
        "exported_from": str(src), "opset_requested": args.opset, "imgsz": args.imgsz,
        "dynamic": args.dynamic, "simplify": args.simplify, "half": args.half,
        "device": args.device, "seconds_export": round(dt_export, 1),
        "seconds_check": round(dt_check, 1), "git_head": _git_head(),
        "env": _env_fingerprint(), "graph": graph, "checker": chk,
        "verdict": "pass" if chk["ok"] else "fail",
    }
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"[ok   ] checker {'通过' if chk['ok'] else '失败'}，体检 {dt_check:.1f} s -> {json_out}")
    print(f"[graph] opset={graph['opsets']} ir={graph['ir_version']} nodes={graph['n_nodes']} "
          f"size={graph['size_mb']} MB producer={graph['producer']}")
    for i in graph["inputs"]:
        print(f"[in   ] {i['name']:<10} shape={i['shape']} dynamic_axes={i['dynamic_axes']}")
    for o in graph["outputs"]:
        print(f"[out  ] {o['name']:<10} shape={o['shape']} dynamic_axes={o['dynamic_axes']}")

    warn_rc = 0
    if not chk["ok"]:
        print(f"[err  ] onnx.checker 失败：{chk['error']}")
        warn_rc = 3
    if args.dynamic and not graph["batch_dynamic"]:
        print("[warn ] 要了 dynamic 但输入第 0 维不是符号维，batch 只能是 1，压测吞吐没意义")
        warn_rc = warn_rc or 3
    req_opset = int(graph["opsets"].get("ai.onnx", 0) or 0)
    if req_opset < args.opset:
        print(f"[warn ] 实际 opset {req_opset} 低于要求的 {args.opset}，TensorRT 可能不认")
        warn_rc = warn_rc or 3

    if args.markdown:
        lines: List[str] = [
            f"**ONNX 导出（tag=`{args.tag}`，run=`{run}`）**", "",
            "| 项 | 值 |", "| --- | --- |",
            f"| 源权重 | `{w.name}`（run `{run}`，git {payload['git_head']}） |",
            f"| 产物 | `{out_path.name}`，{graph['size_mb']} MB |",
            f"| opset / IR | {req_opset} / {graph['ir_version']}（producer {graph['producer']}） |",
            f"| 动态 batch | {'是' if graph['batch_dynamic'] else '否'} |",
            f"| 图节点数 | {graph['n_nodes']}（{len(graph['op_types'])} 种算子） |",
            f"| 输入 | " + "; ".join(f"`{i['name']}` {i['shape']}" for i in graph["inputs"]) + " |",
            f"| 输出 | " + "; ".join(f"`{o['name']}` {o['shape']}" for o in graph["outputs"]) + " |",
            f"| onnx.checker | {'通过' if chk['ok'] else '失败：' + str(chk['error'])} |",
            f"| 导出 / 体检耗时 | {dt_export:.1f} s / {dt_check:.1f} s |", "",
        ]
        print("\n".join(lines))

    print(f"[done ] verdict={payload['verdict']}")
    return warn_rc


if __name__ == "__main__":
    raise SystemExit(main())