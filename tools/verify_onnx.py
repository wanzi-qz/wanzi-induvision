#!/usr/bin/env python
"""校验 .pt 与 .onnx 是否等价：逐张量对齐 -> 端到端 mAP -> 单帧延迟基线。

为什么要三层，而不是一层
------------------------
换后端（PyTorch -> ONNX Runtime -> TensorRT）最容易被含糊过去的一句话是
"精度基本没掉"。这句话要能落到数字上，得回答三个不同的问题：

1. **算子层面等价吗？** 同一批输入、同一份前处理，两个后端的原始输出张量差多少。
   这层最灵敏，也是唯一能把差异归因到"导出/算子实现"的一层——因为它绕开了
   NMS、阈值、匹配这些后处理。判据用 cosine 与相对 L2，不用绝对差：分割输出里
   box 坐标是几百的量级、mask 系数是 0~1，绝对差跨张量没有可比性。
2. **业务指标层面等价吗？** 端到端在 test split 上重算 mAP，掉几个点。
   这层才是能写进报告的话："导出后 mask mAP50-95 从 X 到 Y"。
3. **换后端换来多少速度？** 单帧延迟的 mean / P50 / P95 / P99。报延迟只报均值
   等于没报，尾巴由队列抖动和后处理毛刺决定。

第 1 层默认把 PyTorch 也放到 CPU 上跑（``--align-device cpu``）：这样两边只差
"导出与算子实现"这一个变量；拿 GPU 上的 PyTorch 去比 CPU 上的 ORT，量到的差里
混着 cuDNN 卷积算法选择的不同，说不清是谁的锅。想量那个混合差就把它改成 ``0``。

前处理统一走 ``induvision/deploy.py``（复刻 ultralytics LetterBox(auto=False) +
predictor.pre_process），两个后端吃到的输入逐字节相同。

用法
----
.. code-block:: powershell

    conda activate induvision
    # 第 1 层：16 张 test 图逐张量对齐（默认，几十秒）
    python tools/verify_onnx.py --onnx weights/baseline15_fp32.onnx --markdown

    # 第 1 + 3 层：再加 30 帧延迟基线（ORT CPU 与 PyTorch GPU 各测一遍）
    python tools/verify_onnx.py --onnx ... --bench-n 30

    # 第 1 + 2 + 3 层：端到端 val（test 435 图，CPU 上要几分钟）
    python tools/verify_onnx.py --onnx ... --val --val-batch 8

    # 只看配置
    python tools/verify_onnx.py --onnx ... --dry-run
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from induvision.deploy import (  # noqa: E402 - 需要先插好 sys.path
    compare_tensors,
    latency_stats,
    preprocess,
    read_image_bgr,
    read_split_images,
    sample_paths,
)


def _git_head() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=15,
        )
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def _env_fingerprint() -> Dict[str, str]:
    fp = {"python": platform.python_version(), "platform": platform.platform()}
    for mod in ("numpy", "cv2", "torch", "ultralytics", "onnx", "onnxruntime"):
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

        # 只有 CPU EP 时延迟数字代表 CPU 推理，不能拿去和 TensorRT 的 GPU 数字比
        fp["ort_providers"] = ",".join(ort.get_available_providers())
    except Exception:  # noqa: BLE001
        pass
    return fp


def _torch_device(dev) -> str:
    """把 ultralytics 风格的设备号翻成 torch 认的写法：``"0"`` -> ``"cuda:0"``。

    ultralytics 的 ``device="0"`` 是它自己解析的，直接丢给 ``nn.Module.to()`` 会
    报 ``RuntimeError: Invalid device string: '0'``（踩过）。
    """
    s = str(dev).strip()
    return f"cuda:{s}" if s.isdigit() else s


def _to_numpy_list(out) -> List[np.ndarray]:
    """把 torch 前向的返回值拍平成 ndarray 列表（分割模型返回嵌套 tuple）。"""
    import torch  # 延迟导入

    if isinstance(out, torch.Tensor):
        return [out.detach().cpu().float().numpy()]
    if isinstance(out, (list, tuple)):
        res: List[np.ndarray] = []
        for o in out:
            res.extend(_to_numpy_list(o))
        return res
    return []


def _load_inputs(paths: Sequence[Path], imgsz: int) -> Tuple[List[np.ndarray], List[str]]:
    """读图 + 统一前处理，返回 (NCHW float32 列表, 图片名列表)。读不到的图直接跳过。"""
    xs, names, skipped = [], [], []
    for p in paths:
        img = read_image_bgr(p)
        if img is None:
            skipped.append(str(p))
            continue
        x, _lb = preprocess(img, imgsz=imgsz)
        xs.append(x)
        names.append(p.name)
    if skipped:
        print(f"[warn ] {len(skipped)} 张图读不出来，已跳过：{', '.join(skipped[:3])}")
    return xs, names


def _ort_session(onnx_path: Path, providers: Optional[Sequence[str]] = None):
    import onnxruntime as ort  # 延迟导入

    provs = list(providers) if providers else ort.get_available_providers()
    sess = ort.InferenceSession(str(onnx_path), providers=provs)
    return sess, [p for p in sess.get_providers()], sess.get_inputs()[0].name


def _net_forward(net, x_np: np.ndarray, device: str) -> List[np.ndarray]:
    """PyTorch 侧前向：eval + no_grad，返回拍平后的 ndarray 列表。"""
    import torch  # 延迟导入

    dev = _torch_device(device)
    t = torch.from_numpy(x_np).to(dev)
    with torch.no_grad():
        out = net(t)
    if dev.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()  # GPU 是异步提交，不同步量到的是提交耗时不是算完耗时
    return _to_numpy_list(out)


def align_tensors(xs: Sequence[np.ndarray], net, sess, input_name: str,
                  device: str) -> Dict[str, object]:
    """第 1 层：逐图逐输出张量比差，按输出位置聚合。

    聚合口径刻意取"最坏值"而不是平均值：max_abs 取最大、rel_l2 取最大、
    cosine 取最小。平均数会把一张图上的崩坏摊薄成看不见的量级。
    """
    per_out: List[Dict[str, object]] = []
    counts = {"pt": None, "ort": None}
    worst_cos = 1.0
    t0 = time.time()
    for x in xs:
        x4 = x[None, ...]  # (1,3,H,W)
        pt_out = _net_forward(net, x4, device)
        ort_out = sess.run(None, {input_name: x4.astype(np.float32)})
        counts["pt"], counts["ort"] = len(pt_out), len(ort_out)
        n = min(len(pt_out), len(ort_out))
        for i in range(n):
            d = compare_tensors(pt_out[i], ort_out[i])
            while len(per_out) <= i:
                per_out.append({
                    "index": len(per_out), "ort_name": None, "shape": None,
                    "max_abs": 0.0, "mean_abs": 0.0, "rel_l2": 0.0, "cosine": 1.0,
                    "shape_match": True, "n_images": 0,
                })
            slot = per_out[i]
            slot["ort_name"] = sess.get_outputs()[i].name if i < len(sess.get_outputs()) else None
            slot["shape"] = d["shape_a"]
            slot["max_abs"] = max(slot["max_abs"], d["max_abs"] or 0.0)
            slot["mean_abs"] = (slot["mean_abs"] * slot["n_images"] + (d["mean_abs"] or 0.0)) / (slot["n_images"] + 1)
            slot["rel_l2"] = max(slot["rel_l2"], d["rel_l2"] or 0.0)
            slot["cosine"] = min(slot["cosine"], d["cosine"] if d["cosine"] is not None else 1.0)
            slot["shape_match"] = slot["shape_match"] and bool(d["shape_match"])
            slot["n_images"] += 1
            if d["cosine"] is not None:
                worst_cos = min(worst_cos, float(d["cosine"]))
    for s in per_out:
        for k in ("max_abs", "mean_abs", "rel_l2", "cosine"):
            s[k] = round(float(s[k]), 8)
    return {
        "n_images": len(xs), "seconds": round(time.time() - t0, 1),
        "output_count_pt": counts["pt"], "output_count_ort": counts["ort"],
        "outputs": per_out, "worst_cosine": round(worst_cos, 8),
    }


def bench_latency(xs: Sequence[np.ndarray], sess, input_name: str, net=None,
                  device: Optional[str] = None, warmup: int = 3) -> Dict[str, object]:
    """第 3 层：单帧（batch=1）延迟分布。ORT 与 PyTorch 各测一遍，分开报。

    前 ``warmup`` 帧不计时：首帧要付算子初始化、显存分配、cudnn 选算法的钱，
    混进去会把 P99 拉成假的高。
    """
    out: Dict[str, object] = {"n_frames": len(xs), "warmup": warmup, "batch": 1}

    def _run_ort(x4):
        sess.run(None, {input_name: x4})

    ort_ms = []
    pool = list(xs)
    for i in range(warmup):
        _run_ort(pool[i % len(pool)][None, ...].astype(np.float32))
    for x in pool:
        t = time.perf_counter()
        _run_ort(x[None, ...].astype(np.float32))
        ort_ms.append((time.perf_counter() - t) * 1000)
    out["ort"] = latency_stats(ort_ms)
    out["ort"]["providers"] = [p for p in sess.get_providers()]

    if net is not None and device is not None:
        torch_ms = []
        for i in range(warmup):
            _net_forward(net, pool[i % len(pool)][None, ...], device)
        for x in pool:
            t = time.perf_counter()
            _net_forward(net, x[None, ...], device)
            torch_ms.append((time.perf_counter() - t) * 1000)
        out["torch"] = latency_stats(torch_ms)
        out["torch"]["device"] = _torch_device(device)
    return out


def val_onnx(onnx_path: Path, data_yaml: Path, split: str, imgsz: int, batch: int,
             device: str, run: str, tag: str) -> Dict[str, object]:
    """第 2 层：拿 ONNX 当权重在 test split 上重算 mAP（走 ultralytics 的 AutoBackend）。

    与 ``tools/eval_seg.py`` 的口径一致（同 data.yaml、同 split、同 imgsz），
    所以两边的数字可以直接相减。差值里除了后端实现，还包含设备差异
    （.pt 那次是 GPU，这次 ORT 只有 CPU EP），解读时要带上这一条。
    """
    from ultralytics import YOLO  # 延迟导入

    name = f"{run}_onnx{tag}_val_{split}"
    t0 = time.time()
    m = YOLO(str(onnx_path))
    metrics = m.val(data=str(data_yaml), split=split, imgsz=imgsz, batch=batch,
                    device=device, project=str(REPO_ROOT / "runs" / "segment"),
                    name=name, exist_ok=True, plots=False, verbose=False)
    dt = time.time() - t0

    seg = getattr(metrics, "seg", None) or getattr(metrics, "mask", None)

    def _g(obj, key):
        v = getattr(obj, key, None)
        return None if v is None else float(v)

    return {
        "seconds": round(dt, 1), "split": split, "batch": batch, "device": str(device),
        "run_dir": str(REPO_ROOT / "runs" / "segment" / name),
        "box": {"P": _g(metrics.box, "mp"), "R": _g(metrics.box, "mr"),
                "mAP50": _g(metrics.box, "map50"), "mAP50-95": _g(metrics.box, "map")},
        "mask": {"P": _g(seg, "mp"), "R": _g(seg, "mr"),
                 "mAP50": _g(seg, "map50"), "mAP50-95": _g(seg, "map")},
    }


def load_reference(run_dir: Path, split: str) -> Optional[Dict[str, object]]:
    """读 eval_seg.py 落的 ``eval_<split>.json``，作为 .pt 侧的参照成绩。"""
    p = run_dir / f"eval_{split}.json"
    if not p.is_file():
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    return {"path": str(p), "summary": d.get("summary"), "device": d.get("device"),
            "git_head": d.get("git_head"), "seconds": d.get("seconds")}


def _delta(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    return round(b - a, 6)


def check_align(a: Dict[str, object], min_cosine: float,
                max_rel_l2: float) -> Tuple[bool, List[str]]:
    """第 1 层判据，返回 ``(是否通过, 未通过原因)``。

    四条同时成立才算两个后端等价：

    1. 两端输出个数相同。个数不同说明图被动过（比如 NMS 被折进模型），
       这时候逐张量比差已经没有意义，必须先报出来；
    2. worst cosine >= ``min_cosine``；
    3. 每个输出形状逐个一致（``compare_tensors`` 在 ravel 前比形状，
       (1,51,8400) 与 (1,8400,51) 元素数相同，拉平后能算出一个"看着还行"的差）；
    4. 每个输出的相对 L2 <= ``max_rel_l2``。
    """
    why: List[str] = []
    npt, nort = a.get("output_count_pt"), a.get("output_count_ort")
    if npt != nort:
        why.append(f"输出数不一致 pt={npt} ort={nort}")
    cos = a.get("worst_cosine")
    if cos is None or float(cos) < min_cosine:
        why.append(f"worst cosine {cos} < {min_cosine}")
    for o in a.get("outputs") or []:
        i = o.get("index")
        if not o.get("shape_match", True):
            why.append(f"out{i} 形状不一致 {o.get('shape')}")
        rl = o.get("rel_l2")
        if rl is not None and float(rl) > max_rel_l2:
            why.append(f"out{i} rel_l2 {float(rl):.3e} > {max_rel_l2:.3e}")
    return (not why), why


def check_val(delta: Optional[float], max_map_delta: float) -> Tuple[bool, List[str]]:
    """第 2 层判据：mask mAP50-95 相对 .pt 参照的差不得超过 ``max_map_delta``。

    ``delta`` 为 None（找不到 ``eval_<split>.json`` 参照）时判通过：没有参照就不该
    报失败，这时报告里只写 ONNX 的绝对值并注明"无参照"。用绝对值判据而不是
    "掉了就失败"，是因为端到端 mAP 本身带设备/后处理抖动，W3 实测 .pt 与 .onnx
    差 +0.0001，卡在 0 上会让正常的浮点噪声变成红灯。
    """
    if delta is None:
        return True, []
    if abs(delta) > max_map_delta:
        return False, [f"mask mAP50-95 差 {delta:+.4f}，超过 ±{max_map_delta}"]
    return True, []


def _md(layers: Dict[str, object], cfg: Dict[str, object]) -> List[str]:
    lines: List[str] = [f"**ONNX 对齐校验（run=`{cfg['run']}`，tag=`{cfg['tag']}`）**", ""]
    al = layers.get("align")
    if al:
        lines += [
            f"第 1 层 逐张量：{al['n_images']} 张 {cfg['split']} 图，"
            f"PyTorch({cfg['align_device']}) vs ONNX Runtime({', '.join(al['providers'])})，{al['seconds']} s", "",
            "| 输出 | 形状 | max_abs | mean_abs | rel_l2 | cosine |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for o in al["outputs"]:
            lines.append(f"| `{o['ort_name']}` | {o['shape']} | {o['max_abs']:.3e} | "
                         f"{o['mean_abs']:.3e} | {o['rel_l2']:.3e} | {o['cosine']:.8f} |")
        lines.append("")
    v = layers.get("val")
    if v:
        ref = layers.get("reference") or {}
        rs = (ref.get("summary") or {})
        lines += [
            f"第 2 层 端到端：{cfg['split']} split 重算 mAP（ONNX 走 ultralytics AutoBackend，device={v['device']}）", "",
            "| 指标 | .pt 参照 | .onnx 实测 | 差值 |", "| --- | --- | --- | --- |",
        ]
        for kind in ("mask", "box"):
            for key in ("mAP50-95", "mAP50", "P", "R"):
                a = (rs.get(kind) or {}).get(key)
                b = (v.get(kind) or {}).get(key)
                fa = "—" if a is None else f"{a:.4f}"
                fb = "—" if b is None else f"{b:.4f}"
                fd = "—" if _delta(a, b) is None else f"{_delta(a, b):+.4f}"
                lines.append(f"| {kind} {key} | {fa} | {fb} | {fd} |")
        lines.append("")
    b = layers.get("bench")
    if b:
        lines += [f"第 3 层 单帧延迟（batch=1，imgsz={cfg['imgsz']}，{b['n_frames']} 帧，"
                  f"warmup {b['warmup']}）", "",
                  "| 后端 | 设备 | mean | p50 | p95 | p99 | max | FPS(mean) |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for key, label in (("ort", "ONNX Runtime"), ("torch", "PyTorch")):
            s = b.get(key)
            if not s or s.get("mean_ms") is None:
                continue
            dev = ", ".join(s["providers"]) if key == "ort" else str(s.get("device"))
            lines.append(f"| {label} | {dev} | {s['mean_ms']:.2f} ms | {s['p50_ms']:.2f} | "
                         f"{s['p95_ms']:.2f} | {s['p99_ms']:.2f} | {s['max_ms']:.2f} | {s['fps']} |")
        lines.append("")
    return lines


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="校验 .pt 与 .onnx 是否等价（逐张量 / 端到端 mAP / 延迟）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--weights", default="runs/segment/baseline15/weights/best.pt", help=".pt 权重")
    ap.add_argument("--onnx", default=None, help="ONNX 路径，默认 weights/<run>_fp32.onnx")
    ap.add_argument("--data", default="out/yolo_mvtec15/data.yaml", help="data.yaml")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--name", default=None, help="run 名，默认从权重路径反推")
    ap.add_argument("--tag", default="fp32", help="对应导出时的 tag，只影响默认 onnx 路径与 JSON 文件名")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--align-n", type=int, default=16, help="逐张量对齐用几张图")
    ap.add_argument("--align-device", default="cpu", help="对齐时 PyTorch 用的设备；cpu 才能把差异归因到导出本身")
    ap.add_argument("--bench-n", type=int, default=0, help="延迟测量帧数，0 = 跳过第 3 层")
    ap.add_argument("--bench-device", default="0", help="延迟测量时 PyTorch 用的设备")
    ap.add_argument("--bench-warmup", type=int, default=3)
    ap.add_argument("--val", action="store_true", help="跑第 2 层：用 ONNX 在 split 上重算 mAP")
    ap.add_argument("--val-batch", type=int, default=8)
    ap.add_argument("--val-device", default="cpu", help="ORT 只有 CPU EP，这里别填 GPU")
    ap.add_argument("--min-cosine", type=float, default=0.9999, help="对齐判据：cosine 下限")
    ap.add_argument("--max-rel-l2", type=float, default=1e-3, help="对齐判据：相对 L2 上限")
    ap.add_argument("--max-map-delta", type=float, default=0.01, help="端到端判据：mask mAP50-95 允许掉多少")
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--markdown", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    w = Path(args.weights)
    d = Path(args.data)
    if not w.exists():
        print(f"[err  ] 权重不存在：{w}")
        return 2
    if not d.exists():
        print(f"[err  ] data.yaml 不存在：{d}")
        return 2
    if not w.is_absolute():
        w = (REPO_ROOT / w).resolve()
    if not d.is_absolute():
        d = (REPO_ROOT / d).resolve()

    run = args.name or (w.parents[1].name if w.parent.name == "weights" else w.stem)
    onnx_path = Path(args.onnx) if args.onnx else (REPO_ROOT / "weights" / f"{run}_{args.tag}.onnx")
    if not onnx_path.is_absolute():
        onnx_path = (REPO_ROOT / onnx_path).resolve()
    if not onnx_path.exists():
        print(f"[err  ] ONNX 不存在：{onnx_path}（先跑 tools/export_onnx.py）")
        return 2
    run_dir = REPO_ROOT / "runs" / "segment" / run
    json_out = Path(args.json_out) if args.json_out else run_dir / f"verify_onnx_{args.tag}.json"
    if not json_out.is_absolute():
        json_out = (REPO_ROOT / json_out).resolve()

    cfg = {
        "weights": str(w), "onnx": str(onnx_path), "data": str(d), "split": args.split,
        "run": run, "tag": args.tag, "imgsz": args.imgsz, "align_n": args.align_n,
        "align_device": args.align_device, "bench_n": args.bench_n,
        "bench_device": args.bench_device, "val": args.val, "val_batch": args.val_batch,
        "val_device": args.val_device, "thresholds": {
            "min_cosine": args.min_cosine, "max_rel_l2": args.max_rel_l2,
            "max_map_delta": args.max_map_delta},
        "json_out": str(json_out),
    }
    print("[cfg  ] " + json.dumps(cfg, ensure_ascii=False))
    if args.dry_run:
        print("[dry  ] 未校验")
        return 0

    t_start = time.time()
    images = read_split_images(d, args.split)
    if not images:
        print(f"[err  ] {d} 的 {args.split} split 里没有图片")
        return 2
    picked = sample_paths(images, args.align_n if args.bench_n <= 0 else max(args.align_n, args.bench_n))
    xs, names = _load_inputs(picked, args.imgsz)
    if not xs:
        print("[err  ] 一张图都没读到，无法校验")
        return 2
    print(f"[data ] {args.split} 共 {len(images)} 图，取 {len(xs)} 张做对齐/压测")

    from ultralytics import YOLO  # 延迟导入：--dry-run 不需要 torch

    net = YOLO(str(w)).model
    net.eval()
    net.to(_torch_device(args.align_device))
    sess, providers, input_name = _ort_session(onnx_path)
    print(f"[ort  ] providers={providers} input={input_name}")

    layers: Dict[str, object] = {"providers": providers}
    align_x = xs[:args.align_n]
    layers["align"] = align_tensors(align_x, net, sess, input_name, args.align_device)
    layers["align"]["providers"] = providers
    a = layers["align"]
    print(f"[align] {a['n_images']} 图 / {a['seconds']} s，"
          f"输出数 pt={a['output_count_pt']} ort={a['output_count_ort']}，worst cosine={a['worst_cosine']:.8f}")
    for o in a["outputs"]:
        print(f"[align] out{o['index']} {o['ort_name']} shape={o['shape']} "
              f"max_abs={o['max_abs']:.3e} rel_l2={o['rel_l2']:.3e} cosine={o['cosine']:.8f}")

    ok_align, why_align = check_align(a, args.min_cosine, args.max_rel_l2)
    if not ok_align:
        print(f"[fail ] 第 1 层不达标（cosine>={args.min_cosine} / rel_l2<={args.max_rel_l2}）："
              + "；".join(why_align))
    else:
        print("[ok   ] 第 1 层通过：两个后端算子层面等价")

    if args.bench_n > 0:
        bench_x = xs[:args.bench_n]
        if len(bench_x) < args.bench_n:
            print(f"[warn ] 只有 {len(bench_x)} 张可用图，少于 --bench-n {args.bench_n}")
        net.to(_torch_device(args.bench_device)).eval()
        layers["bench"] = bench_latency(bench_x, sess, input_name, net=net,
                                        device=args.bench_device, warmup=args.bench_warmup)
        b = layers["bench"]
        for key in ("ort", "torch"):
            s = b.get(key)
            if s and s.get("mean_ms") is not None:
                print(f"[bench] {key:<6} mean={s['mean_ms']:.2f} ms p50={s['p50_ms']:.2f} "
                      f"p95={s['p95_ms']:.2f} p99={s['p99_ms']:.2f} fps={s['fps']}")
        net.to(_torch_device(args.align_device))

    ok_val = True
    if args.val:
        layers["reference"] = load_reference(run_dir, args.split)
        if not layers["reference"]:
            print(f"[warn ] 没找到 {run_dir / ('eval_' + args.split + '.json')}，"
                  f"第 2 层只报 ONNX 的绝对值，不给差值")
        layers["val"] = val_onnx(onnx_path, d, args.split, args.imgsz, args.val_batch,
                                 args.val_device, run, args.tag)
        v = layers["val"]
        print(f"[val  ] ONNX 端到端 {v['seconds']} s：mask mAP50-95={v['mask']['mAP50-95']:.4f} "
              f"mAP50={v['mask']['mAP50']:.4f} / box mAP50-95={v['box']['mAP50-95']:.4f}")
        ref_sum = (layers.get("reference") or {}).get("summary") or {}
        ref_mask = (ref_sum.get("mask") or {}).get("mAP50-95")
        if ref_mask is not None:
            dl = v["mask"]["mAP50-95"] - ref_mask
            print(f"[val  ] 对 .pt 参照 {ref_mask:.4f} -> {v['mask']['mAP50-95']:.4f}，差 {dl:+.4f}")
            ok_val, why_val = check_val(dl, args.max_map_delta)
            if not ok_val:
                print(f"[fail ] 第 2 层不达标（--max-map-delta {args.max_map_delta}）："
                      + "；".join(why_val))

    verdict = "pass" if (ok_align and ok_val) else "fail"
    payload = {
        "run": run, "tag": args.tag, "weights": str(w), "onnx": str(onnx_path),
        "data": str(d), "split": args.split, "imgsz": args.imgsz,
        "config": cfg, "images_used": names[:len(xs)],
        "align": a, "bench": layers.get("bench"), "val": layers.get("val"),
        "reference": layers.get("reference"), "verdict": verdict,
        "seconds": round(time.time() - t_start, 1), "git_head": _git_head(),
        "env": _env_fingerprint(),
    }
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[done ] verdict={verdict}，{payload['seconds']} s -> {json_out}")

    if args.markdown:
        md_cfg = dict(cfg)
        md_cfg.update({"providers": providers})
        print("\n".join(_md(layers, md_cfg)))
    return 0 if verdict == "pass" else 3


if __name__ == "__main__":
    raise SystemExit(main())