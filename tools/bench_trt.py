#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""TensorRT 三档 engine 的压测与验收（W3b）。

对 ``weights/<run>_trt_<prec>.engine`` 逐档做三层校验，口径全部沿用
``tools/verify_onnx.py``，所以 W3a（PyTorch / ONNX）和 W3b（TRT 三档）的数字能排进
同一张表：

1. **第 1 层｜逐张量对齐**：engine 输出 vs PyTorch ``best.pt`` 前向。同一份入库清单
   ``configs/bench_sample.json`` 的前 16 张、同一个 ``compare_tensors``。
   参照侧跑 CPU：engine 只能在 GPU 上跑，这一层因此是**跨设备**比较，fp32 档的
   cosine 会比 W3a 那次 ONNX↔PT（都在 CPU）略低，属于口径差异不是精度问题。
2. **第 3 层｜单帧延迟 + 显存**：清单前 30 张、batch=1、3 帧 warmup 不计时，
   报 mean/p50/p95/p99/max/FPS。延迟走 ``TrtRunner.forward_device``——输入常驻显存，
   量到的是纯推理，不含 H2D 拷贝（拷贝留给 W4 的 C++ 流水线单独算）。
   显存记两个数，别混：engine 自报的激活显存（``device_memory_size_v2``，可复现），
   和 ``mem_get_info`` 的进程差值（含 CUDA context、权重、torch 缓存分配器）。
3. **第 2 层｜端到端 mAP**：``YOLO(x.engine).val(split=test, batch=1)``。
   profile 是 min=opt=max=(1,3,640,640)，batch 只能填 1。

**为什么 INT8 不设第 1 层判据**：熵校准量化把激活动态范围压到 8 bit，逐张量差是
设计内的（cosine 掉到 0.99 量级很正常）。拿 FP16 的 0.9999 去卡它只会得到一盏恒红的灯，
红久了就没人看。所以 INT8 的第 1 层照报不照判，验收只留端到端 mAP 掉点这一条。

**掉点跟谁比**：fp32 档比 ``runs/segment/<run>/eval_test.json`` 里 .pt 的参照成绩；
fp16/int8 档比**本轮的 fp32 档**。理由是后者同为 TRT 后端，差值里只剩量化本身，
不掺后端实现与设备的系统性偏移。

返回码：0 = 全部通过；2 = 输入/环境错误；3 = 有判据未通过。
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT, REPO_ROOT / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from induvision.deploy import (  # noqa: E402 - 需要先插好 sys.path
    compare_tensors,
    latency_stats,
    read_split_images,
    relative_to_root,
)
from induvision.trt_infer import TrtRunner  # noqa: E402
import verify_onnx as vox  # noqa: E402 - 复用抽样口径与前处理，保证两次报告可比

#: 三档默认判据。``None`` = 该项只报不判，见模块文档里 INT8 那段。
#: 逐张量判据按"这一档实际用什么数值格式算"定，不能一套数套三档：
#: - fp32 档走 TRT 默认（BuilderFlag.TF32 开着），实际是 TF32，10-bit 尾数，
#:   rel_l2 停在 1e-3 量级是格式本身的机器精度，不是链路错了。要卡 1e-5，
#:   得用 build_trt.py --no-tf32 另建一档真 FP32 来对照（见 docs/deploy.md §8）。
#: - fp16 同理，尾数 10-bit，累积到 1e-2 以内属正常。
#: - int8 不做逐张量卡线（动态范围量化后逐元素差没有意义），只看端到端 mAP。
#: 三档最终都要过端到端 mAP 那关，那才是判据。
DEFAULT_THRESHOLDS: Dict[str, Dict[str, Optional[float]]] = {
    "fp32": {"min_cosine": 0.9999, "max_rel_l2": 1e-2, "max_map_delta": 0.01},
    "fp16": {"min_cosine": 0.9999, "max_rel_l2": 1e-2, "max_map_delta": 0.01},
    "int8": {"min_cosine": None, "max_rel_l2": None, "max_map_delta": 0.03},
}

#: 精度档 -> TRT inspector 报的 Format/Datatype 关键字。fp32 不在表里，
#: 见 low_prec_share。
LOW_PREC_KEYS = {"fp16": "Half", "int8": "Int8"}


# --------------------------------------------------------------------------
# 第 1 层：逐张量对齐
# --------------------------------------------------------------------------

def align_engine(xs: Sequence[np.ndarray], net, runner, device: str) -> Dict[str, object]:
    """engine vs PyTorch 逐图逐输出比差，按输出位置聚合，取最坏值。

    聚合口径与 ``vox.align_tensors`` 一致（max_abs/rel_l2 取最大、cosine 取最小）：
    平均数会把一张图上的崩坏摊薄到看不见的量级。
    """
    per_out: List[Dict[str, object]] = []
    counts = {"pt": None, "trt": None}
    worst_cos = 1.0
    t0 = time.time()
    for x in xs:
        x4 = x[None, ...]
        pt_out = vox._net_forward(net, x4, device)
        trt_out = runner(np.ascontiguousarray(x4, dtype=np.float32))
        counts["pt"], counts["trt"] = len(pt_out), len(trt_out)
        for i in range(min(len(pt_out), len(trt_out))):
            d = compare_tensors(pt_out[i], trt_out[i])
            while len(per_out) <= i:
                per_out.append({
                    "index": len(per_out), "trt_name": None, "shape": None,
                    "max_abs": 0.0, "mean_abs": 0.0, "rel_l2": 0.0, "cosine": 1.0,
                    "shape_match": True, "n_images": 0,
                })
            slot = per_out[i]
            names = runner.output_names
            slot["trt_name"] = names[i] if i < len(names) else None
            slot["shape"] = d["shape_a"]
            slot["max_abs"] = max(slot["max_abs"], d["max_abs"] or 0.0)
            slot["mean_abs"] = ((slot["mean_abs"] * slot["n_images"]
                                 + (d["mean_abs"] or 0.0)) / (slot["n_images"] + 1))
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
        "output_count_pt": counts["pt"], "output_count_trt": counts["trt"],
        "outputs": per_out, "worst_cosine": round(worst_cos, 8),
    }


def check_align(a: Dict[str, object], min_cosine: Optional[float],
                max_rel_l2: Optional[float]) -> Tuple[bool, List[str]]:
    """薄封装 ``vox.check_align``：判据只有一份实现，这里只做键名映射。

    ``vox.check_align`` 按 ``output_count_pt`` / ``output_count_ort`` 取键（它是为
    ONNX 写的）。为了不把那段逻辑抄第二遍——抄两份迟早会走偏——把 trt 侧的计数
    映射到它期望的键名上。
    """
    if min_cosine is None and max_rel_l2 is None:
        return True, []
    b = dict(a)
    b["output_count_ort"] = a.get("output_count_trt")
    ok, why = vox.check_align(b, -1.0 if min_cosine is None else min_cosine,
                              float("inf") if max_rel_l2 is None else max_rel_l2)
    return ok, why


# --------------------------------------------------------------------------
# 第 3 层：延迟与显存
# --------------------------------------------------------------------------

def bench_engine(xs: Sequence[np.ndarray], runner, device: str,
                 warmup: int = 3) -> Dict[str, object]:
    """单帧延迟：输入常驻 GPU，前 ``warmup`` 帧不计时。

    输出缓冲每帧新建。torch 的缓存分配器会复用显存，这一项的开销很小，但**不是零**；
    W4 的 C++ 服务是预分配 + 环形队列复用，那边的数会比这里略好，别直接互比。
    """
    import torch  # noqa: PLC0415 - 延迟导入

    dev = vox._torch_device(device)
    pool = [torch.from_numpy(np.ascontiguousarray(x[None, ...], dtype=np.float32)).to(dev)
            for x in xs]
    for i in range(warmup):
        runner.forward_device(pool[i % len(pool)])
    torch.cuda.synchronize()
    ms: List[float] = []
    for t in pool:
        t0 = time.perf_counter()
        runner.forward_device(t)
        ms.append((time.perf_counter() - t0) * 1000)
    stats = latency_stats(ms)
    stats.update({"warmup": warmup, "batch": 1, "n_frames": len(pool),
                  "device": dev, "input_resident": True})
    del pool
    return stats


def bench_torch_gpu(xs: Sequence[np.ndarray], net, device: str,
                    warmup: int = 3) -> Dict[str, object]:
    """同进程里再量一遍 PyTorch GPU 单帧延迟，给"加速比"一个同场对照。

    不抄 deploy.md 里那个 13.13 ms：那是另一次运行、另一个进程状态（当时机器上还
    有别的东西）。加速比只有在同一进程同一批图上量才有意义。
    """
    dev = vox._torch_device(device)
    net.to(dev).eval()
    pool = [x[None, ...].astype(np.float32) for x in xs]
    for i in range(warmup):
        vox._net_forward(net, pool[i % len(pool)], device)
    ms: List[float] = []
    for x4 in pool:
        t0 = time.perf_counter()
        vox._net_forward(net, x4, device)
        ms.append((time.perf_counter() - t0) * 1000)
    stats = latency_stats(ms)
    stats.update({"warmup": warmup, "batch": 1, "n_frames": len(pool), "device": dev})
    return stats


def _free_mib() -> Optional[float]:
    import torch  # noqa: PLC0415

    if not torch.cuda.is_available():
        return None
    free, _total = torch.cuda.mem_get_info()
    return free / 1024**2


def measure_tier(prec: str, engine_path: Path, *, net, xs_align: Sequence[np.ndarray],
                 xs_bench: Sequence[np.ndarray], device: str, align_device: str,
                 warmup: int) -> Dict[str, object]:
    """一档 engine 的全部实测：显存 -> 对齐 -> 延迟。顺序是刻意的。

    显存必须最先量：后面延迟压测会在 GPU 上放 30 个输入张量，那些是压测脚手架不是
    推理服务本身要的显存，混进 ``mem_get_info`` 的差值里就把数抬高了。
    """
    import torch  # noqa: PLC0415

    tier: Dict[str, object] = {"precision": prec, "engine": str(engine_path),
                               "file_mib": round(engine_path.stat().st_size / 1024**2, 2)}
    gc.collect()
    torch.cuda.empty_cache()
    dev = vox._torch_device(device)
    x0 = torch.from_numpy(np.ascontiguousarray(xs_bench[0][None, ...],
                                               dtype=np.float32)).to(dev)
    free_before = _free_mib()
    t0 = time.time()
    runner = TrtRunner(engine_path, device=dev)
    load_s = time.time() - t0
    runner.forward_device(x0)  # 首次执行要付 tactic 初始化的钱，先跑一次再读数
    free_after = _free_mib()

    tier["io"] = runner.io_info()
    tier["engine_bytes"] = runner.engine_bytes
    tier["engine_mib"] = round(runner.engine_bytes / 1024**2, 2)
    dm = runner.device_memory_bytes
    tier["activation_mib"] = round(dm / 1024**2, 2) if dm else None
    tier["process_vram_mib"] = (round(free_before - free_after, 2)
                                if free_before is not None and free_after is not None else None)
    tier["header_bytes"] = engine_path.stat().st_size - runner.engine_bytes
    tier["metadata_keys"] = sorted(runner.meta) if runner.meta else []
    tier["load_seconds"] = round(load_s, 2)
    del x0

    if xs_align:
        net.to(vox._torch_device(align_device)).eval()
        tier["align"] = align_engine(xs_align, net, runner, align_device)
        net.to(dev).eval()

    if xs_bench:
        tier["bench"] = bench_engine(xs_bench, runner, dev, warmup=warmup)

    runner.close()
    del runner
    gc.collect()
    torch.cuda.empty_cache()
    return tier


# --------------------------------------------------------------------------
# 第 2 层：端到端 mAP
# --------------------------------------------------------------------------

def val_engine(engine_path: Path, data_yaml: Path, split: str, imgsz: int,
               batch: int, device: str, run: str, prec: str) -> Dict[str, object]:
    """拿 engine 当权重在 test split 上重算 mAP。

    不复用 ``vox.val_onnx``：它把 run 名写死成 ``<run>_onnx<tag>_val_<split>``，
    三档 engine 会互相覆盖同一个目录。指标提取的口径完全照抄它，保证可比。
    """
    from ultralytics import YOLO  # noqa: PLC0415

    name = f"{run}_trt_{prec}_val_{split}"
    t0 = time.time()
    m = YOLO(str(engine_path))
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
        "task": getattr(m, "task", None), "names_count": len(getattr(m, "names", {}) or {}),
        "box": {"P": _g(metrics.box, "mp"), "R": _g(metrics.box, "mr"),
                "mAP50": _g(metrics.box, "map50"), "mAP50-95": _g(metrics.box, "map")},
        "mask": {"P": _g(seg, "mp"), "R": _g(seg, "mr"),
                 "mAP50": _g(seg, "map50"), "mAP50-95": _g(seg, "map")},
    }


# --------------------------------------------------------------------------
# 汇总、判据、报告
# --------------------------------------------------------------------------

def low_prec_share(fmts: Dict[str, int], prec: str) -> Optional[float]:
    """层输出里落在低精度上的占比（%）。量化到底生效没有的直接证据。

    fp32 档返回 None 而不是 0.0：这一档本来就不该有低精度层，写 0% 会和
    "inspector 没给格式信息"混成同一个数，两者含义完全不同。
    """
    key = LOW_PREC_KEYS.get(prec)
    if key is None or not fmts:
        return None
    tot = sum(fmts.values())
    return round(100.0 * fmts.get(key, 0) / tot, 1) if tot else None


def gate_tier(prec: str, tier: Dict[str, object], th: Dict[str, Optional[float]],
              base_mask: Optional[float], ref_mask: Optional[float]) -> Tuple[bool, List[str]]:
    """一档的验收。返回 ``(是否通过, 未通过原因)``。

    mAP 的对照基准分两种，见模块文档：fp32 比 .pt 参照，fp16/int8 比本轮 fp32 档。
    基准拿不到就不判这一项——没有参照就报失败是把环境问题算到模型头上。
    """
    why: List[str] = []
    a = tier.get("align")
    if a:
        ok, w = check_align(a, th.get("min_cosine"), th.get("max_rel_l2"))
        if not ok:
            why += [f"第1层：{x}" for x in w]
    base = ref_mask if prec == "fp32" else base_mask
    v = (tier.get("val") or {}).get("mask") or {}
    cur = v.get("mAP50-95")
    mmd = th.get("max_map_delta")
    if cur is not None and base is not None and mmd is not None:
        dl = cur - base
        tier["map_delta"] = round(dl, 6)
        tier["map_base"] = round(float(base), 6)
        if abs(dl) > mmd:
            why.append(f"第2层：mask mAP50-95 {base:.4f} -> {cur:.4f}，差 {dl:+.4f} 超过 ±{mmd}")
    elif cur is not None:
        tier["map_delta"] = None
        tier["map_base"] = None
    return (not why), why


def build_report_tiers(build_rep: Optional[Dict[str, object]]) -> Dict[str, Dict[str, object]]:
    """从 build_trt.py 的报告里取逐档结果，返回 ``{precision: entry}``。

    键名叫 ``"engines"``——不是 ``"builds"``。这个坑踩过一次：拿错名字不报错，
    只是表里的层数/格式占比悄悄全空，看着像"报告没生成"。
    """
    out: Dict[str, Dict[str, object]] = {}
    for b in ((build_rep or {}).get("engines") or []):
        if isinstance(b, dict) and b.get("precision"):
            out[str(b["precision"])] = b
    return out


def build_table(tiers: Dict[str, Dict[str, object]], build_rep: Optional[Dict[str, object]],
                torch_stats: Optional[Dict[str, object]]) -> List[Dict[str, object]]:
    """三档对照表。这一张表就是 W3b 要交付的东西，README 和简历都从它抄数。"""
    rows: List[Dict[str, object]] = []
    base = tiers.get("fp32") or {}
    base_ms = ((base.get("bench") or {}).get("mean_ms"))
    base_map = (((base.get("val") or {}).get("mask")) or {}).get("mAP50-95")
    built = build_report_tiers(build_rep)
    for prec in ("fp32", "fp16", "int8"):
        t = tiers.get(prec)
        if not t:
            continue
        b = built.get(prec) or {}
        bench = t.get("bench") or {}
        val = t.get("val") or {}
        mask = (val.get("mask") or {}).get("mAP50-95")
        mean = bench.get("mean_ms")
        fmts = b.get("output_formats") or {}
        rows.append({
            "precision": prec,
            "engine_mib": t.get("engine_mib"),
            "num_layers": b.get("num_layers"),
            "low_prec_pct": low_prec_share(fmts, prec),
            "activation_mib": t.get("activation_mib"),
            "process_vram_mib": t.get("process_vram_mib"),
            "build_seconds": b.get("build_seconds"),
            "mask_map": mask,
            "box_map": (val.get("box") or {}).get("mAP50-95"),
            "map_delta_vs_fp32": (round(mask - base_map, 6)
                                  if mask is not None and base_map is not None and prec != "fp32"
                                  else None),
            "mean_ms": mean,
            "p95_ms": bench.get("p95_ms"),
            "p99_ms": bench.get("p99_ms"),
            "fps": bench.get("fps"),
            "speedup_vs_fp32": (round(base_ms / mean, 2)
                                if mean and base_ms and prec != "fp32" else None),
            "speedup_vs_torch": (round((torch_stats or {}).get("mean_ms") / mean, 2)
                                 if mean and (torch_stats or {}).get("mean_ms") else None),
            "size_ratio_vs_fp32": (round((t.get("engine_mib") or 0) / base.get("engine_mib"), 3)
                                   if base.get("engine_mib") and prec != "fp32" else None),
        })
    if torch_stats:
        rows.append({
            "precision": "pytorch(.pt, GPU)", "engine_mib": None, "num_layers": None,
            "low_prec_pct": None, "activation_mib": None, "process_vram_mib": None,
            "build_seconds": None, "mask_map": None, "box_map": None,
            "map_delta_vs_fp32": None, "mean_ms": torch_stats.get("mean_ms"),
            "p95_ms": torch_stats.get("p95_ms"), "p99_ms": torch_stats.get("p99_ms"),
            "fps": torch_stats.get("fps"), "speedup_vs_fp32": None,
            "speedup_vs_torch": 1.0, "size_ratio_vs_fp32": None,
        })
    return rows


def _fmt(v, spec: str = ".4f", dash: str = "—") -> str:
    if v is None:
        return dash
    if isinstance(v, str):
        return v
    return format(v, spec)


def _md(report: Dict[str, object], cfg: Dict[str, object]) -> List[str]:
    L: List[str] = [f"**TensorRT 三档压测（run=`{cfg['run']}`，split=`{cfg['split']}`）**", ""]
    s = report.get("sample") or {}
    if s:
        src = "入库清单" if s.get("source") == "manifest" else "即时抽样"
        L += [f"抽样口径：{src} `{s.get('path')}`（{s.get('count')} 张 / 母体 "
              f"{s.get('source_total')} 张，sha256 `{s.get('sha256')}`）；"
              f"align 取前 {s.get('align_n')} 张、bench 取前 {s.get('bench_n')} 张", ""]
        for wn in s.get("warnings") or []:
            L += [f"> 注意：{wn}", ""]

    L += ["| 档位 | engine MiB | 层数 | 低精度层占比 | 激活显存 MiB | mask mAP50-95 | Δ vs fp32 "
          "| mean ms | p95 ms | FPS | vs fp32 | vs PyTorch |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in report.get("table") or []:
        L.append("| {p} | {sz} | {nl} | {lp} | {am} | {mm} | {dm} | {mean} | {p95} | {fps} "
                 "| {sf} | {st} |".format(
                     p=r["precision"], sz=_fmt(r["engine_mib"], ".2f"),
                     nl=_fmt(r["num_layers"], ".0f"),
                     lp=("—" if r["low_prec_pct"] is None else f"{r['low_prec_pct']}%"),
                     am=_fmt(r["activation_mib"], ".2f"), mm=_fmt(r["mask_map"]),
                     dm=_fmt(r["map_delta_vs_fp32"], "+.4f"), mean=_fmt(r["mean_ms"], ".2f"),
                     p95=_fmt(r["p95_ms"], ".2f"), fps=_fmt(r["fps"], ".2f"),
                     sf=_fmt(r["speedup_vs_fp32"], ".2f") + ("x" if r["speedup_vs_fp32"] else ""),
                     st=_fmt(r["speedup_vs_torch"], ".2f") + ("x" if r["speedup_vs_torch"] else "")))
    L.append("")

    for prec, t in (report.get("tiers") or {}).items():
        a = t.get("align")
        if not a:
            continue
        L += [f"- **{prec} 第 1 层**：{a['n_images']} 图，输出数 pt={a['output_count_pt']} / "
              f"trt={a['output_count_trt']}，worst cosine={a['worst_cosine']:.8f}"]
        for o in a["outputs"]:
            L.append(f"  - out{o['index']} `{o['trt_name']}` shape={o['shape']} "
                     f"max_abs={o['max_abs']:.3e} rel_l2={o['rel_l2']:.3e} "
                     f"cosine={o['cosine']:.8f}")
        g = t.get("gate") or {}
        L.append(f"  - 判据：{'通过' if g.get('ok') else '未通过'}"
                 + ("（INT8 第 1 层只报不判）" if prec == "int8" else "")
                 + ("；" + "；".join(g.get("why") or []) if g.get("why") else ""))
        L.append("")
    L += [f"verdict：**{report.get('verdict')}**，总耗时 {report.get('seconds')} s，"
          f"git `{report.get('git_head')}`", ""]
    return L


def load_build_report(path: Optional[Path]) -> Optional[Dict[str, object]]:
    """读 build_trt.py 落的 JSON，拿构建侧的层数/格式直方图（压测侧拿不到）。"""
    if not path or not Path(path).is_file():
        return None
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        print(f"[warn ] 构建报告读不了（{e}）：{path}")
        return None


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="TensorRT 三档 engine 压测（对齐 / 延迟显存 / 端到端 mAP）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--weights", default="runs/segment/baseline15/weights/best.pt",
                    help=".pt 权重，第 1 层的参照后端")
    ap.add_argument("--data", default="out/yolo_mvtec15/data.yaml")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--name", default=None, help="run 名，默认从权重路径反推")
    ap.add_argument("--engine-dir", default="weights", help="engine 所在目录")
    ap.add_argument("--precs", nargs="+", default=["fp32", "fp16", "int8"],
                    choices=["fp32", "fp16", "int8"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0", help="engine 与 PyTorch 延迟对照的 CUDA 设备")
    ap.add_argument("--align-device", default="cpu", help="第 1 层参照侧的设备")
    ap.add_argument("--align-n", type=int, default=16, help="第 1 层用清单前几张")
    ap.add_argument("--bench-n", type=int, default=30, help="延迟用清单前几张")
    ap.add_argument("--bench-warmup", type=int, default=3)
    ap.add_argument("--sample-list", default="configs/bench_sample.json")
    ap.add_argument("--sample-n", type=int, default=vox.DEFAULT_SAMPLE_N)
    ap.add_argument("--no-sample-list", action="store_true")
    ap.add_argument("--strict-sample", action="store_true")
    ap.add_argument("--val", action="store_true", help="跑第 2 层（每档一次全 test split）")
    ap.add_argument("--val-batch", type=int, default=1, help="profile 是 batch=1，别调大")
    ap.add_argument("--val-device", default="0")
    ap.add_argument("--build-report", default=None,
                    help="build_trt.py 的 JSON，默认 runs/logs/trt_build_<run>.json")
    ap.add_argument("--no-torch-bench", action="store_true", help="不同场量 PyTorch GPU 延迟")
    ap.add_argument("--min-cosine", type=float, default=None, help="覆盖 fp32/fp16 的 cosine 下限")
    ap.add_argument("--max-rel-l2", type=float, default=None, help="覆盖 fp32/fp16 的 rel_l2 上限")
    ap.add_argument("--max-map-delta", type=float, default=None, help="覆盖 fp32/fp16 的 mAP 判据")
    ap.add_argument("--int8-max-map-delta", type=float, default=None, help="覆盖 INT8 的 mAP 判据")
    ap.add_argument("--json-out", default=None, help="默认 runs/logs/trt_bench_<run>.json")
    ap.add_argument("--markdown", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    w = Path(args.weights)
    d = Path(args.data)
    for p, what in ((w, "权重"), (d, "data.yaml")):
        if not p.exists():
            print(f"[err  ] {what}不存在：{p}")
            return 2
    w = w if w.is_absolute() else (REPO_ROOT / w).resolve()
    d = d if d.is_absolute() else (REPO_ROOT / d).resolve()
    edir = Path(args.engine_dir)
    edir = edir if edir.is_absolute() else (REPO_ROOT / edir).resolve()
    run = args.name or (w.parents[1].name if w.parent.name == "weights" else w.stem)
    sample_list = None
    if args.sample_list and not args.no_sample_list:
        sample_list = Path(args.sample_list)
        if not sample_list.is_absolute():
            sample_list = (REPO_ROOT / sample_list).resolve()
    build_rep_path = Path(args.build_report) if args.build_report else \
        (REPO_ROOT / "runs" / "logs" / f"trt_build_{run}.json")
    if not build_rep_path.is_absolute():
        build_rep_path = (REPO_ROOT / build_rep_path).resolve()
    json_out = Path(args.json_out) if args.json_out else \
        (REPO_ROOT / "runs" / "logs" / f"trt_bench_{run}.json")
    if not json_out.is_absolute():
        json_out = (REPO_ROOT / json_out).resolve()

    th: Dict[str, Dict[str, Optional[float]]] = {
        k: dict(v) for k, v in DEFAULT_THRESHOLDS.items()}
    for prec in ("fp32", "fp16"):
        if args.min_cosine is not None:
            th[prec]["min_cosine"] = args.min_cosine
        if args.max_rel_l2 is not None:
            th[prec]["max_rel_l2"] = args.max_rel_l2
        if args.max_map_delta is not None:
            th[prec]["max_map_delta"] = args.max_map_delta
    if args.int8_max_map_delta is not None:
        th["int8"]["max_map_delta"] = args.int8_max_map_delta

    engines = {p: edir / f"{run}_trt_{p}.engine" for p in args.precs}
    missing = [str(v) for v in engines.values() if not v.is_file()]
    if missing:
        print("[err  ] 缺 engine（先跑 tools/build_trt.py）：")
        for m in missing:
            print(f"        {m}")
        return 2

    cfg = {
        "run": run, "weights": str(w), "data": str(d), "split": args.split,
        "imgsz": args.imgsz, "engine_dir": str(edir), "precs": list(args.precs),
        "engines": {k: str(v) for k, v in engines.items()},
        "align_n": args.align_n, "align_device": args.align_device,
        "bench_n": args.bench_n, "bench_warmup": args.bench_warmup,
        "device": args.device, "val": args.val, "val_batch": args.val_batch,
        "val_device": args.val_device, "sample_list": str(sample_list) if sample_list else None,
        "torch_bench": not args.no_torch_bench, "build_report": str(build_rep_path),
        "thresholds": th, "json_out": str(json_out),
    }
    print("[cfg  ] " + json.dumps(cfg, ensure_ascii=False))
    if args.dry_run:
        for prec, p in engines.items():
            print(f"[dry  ] {prec}: {p}（{round(p.stat().st_size / 1024**2, 2)} MiB）")
        return 0

    try:
        import torch  # noqa: PLC0415
    except ImportError:
        print("[err  ] 没有 torch，压测跑不了")
        return 2
    if not torch.cuda.is_available():
        print("[err  ] torch.cuda 不可用；TensorRT engine 只能在 GPU 上跑")
        return 2

    t_start = time.time()
    images = read_split_images(d, args.split)
    if not images:
        print(f"[err  ] {d} 的 {args.split} split 里没有图片")
        return 2
    need = max(args.align_n, args.bench_n)
    try:
        canonical, sample_meta = vox.resolve_sample(
            images, root=d.parent, sample_list=sample_list, sample_n=args.sample_n,
            use_list=not args.no_sample_list, split=args.split, strict=args.strict_sample)
    except (FileNotFoundError, ValueError) as e:
        print(f"[err  ] {e}")
        return 2
    for wn in sample_meta["warnings"]:
        print(f"[warn ] {wn}")
    if len(canonical) < need:
        print(f"[err  ] 只有 {len(canonical)} 张，撑不起 align {args.align_n} / "
              f"bench {args.bench_n}")
        return 2
    picked = canonical[:need]
    xs, names, idx = vox._load_inputs_indexed(picked, args.imgsz)
    if not xs:
        print("[err  ] 一张图都没读到")
        return 2
    sample_meta["names"] = [relative_to_root(q, d.parent) for q in picked]
    sample_meta.update({"picked": len(picked), "loaded": len(xs),
                        "skipped": len(picked) - len(xs),
                        "align_n": args.align_n, "bench_n": args.bench_n})
    align_x = vox._take_upto(xs, idx, args.align_n)
    bench_x = vox._take_upto(xs, idx, args.bench_n)
    print(f"[data ] {args.split} 共 {len(images)} 图；口径={sample_meta['source']} "
          f"{sample_meta['count']} 张（sha256:{sample_meta['sha256']}），"
          f"读入 {len(xs)}，align {len(align_x)} / bench {len(bench_x)}")

    from ultralytics import YOLO  # noqa: PLC0415

    net = YOLO(str(w)).model
    net.eval()

    build_rep = load_build_report(build_rep_path)
    if build_rep:
        tiers_in_rep = build_report_tiers(build_rep)
        print(f"[build] 读到构建报告 {build_rep_path.name}"
              f"（{len(tiers_in_rep)} 档：{', '.join(sorted(tiers_in_rep)) or '无'}）")
        absent = [p for p in args.precs if p not in tiers_in_rep]
        if absent:
            print(f"[warn ] 报告里没有 {', '.join(absent)} 档，表中对应的层数/格式占比会留空。"
                  f"构建报告要一次性把三档都建出来（--prec fp32 fp16 int8）")
    else:
        print(f"[warn ] 没有构建报告 {build_rep_path}，表里的层数/格式占比留空")

    torch_stats = None
    if not args.no_torch_bench:
        torch_stats = bench_torch_gpu(bench_x, net, args.device, warmup=args.bench_warmup)
        print(f"[torch] PyTorch GPU 同场对照 mean={torch_stats['mean_ms']:.2f} ms "
              f"p95={torch_stats['p95_ms']:.2f} fps={torch_stats['fps']}")

    tiers: Dict[str, Dict[str, object]] = {}
    for prec in args.precs:
        print(f"[tier ] {prec} 开始")
        tiers[prec] = measure_tier(
            prec, engines[prec], net=net, xs_align=align_x, xs_bench=bench_x,
            device=args.device, align_device=args.align_device, warmup=args.bench_warmup)
        t = tiers[prec]
        b = (t.get("bench") or {})
        print(f"[tier ] {prec} engine {t['engine_mib']} MiB / 激活 {t['activation_mib']} MiB "
              f"/ 进程显存增量 {t['process_vram_mib']} MiB")
        if t.get("align"):
            a = t["align"]
            print(f"[align] {prec} worst cosine={a['worst_cosine']:.8f}，"
                  f"输出数 pt={a['output_count_pt']} trt={a['output_count_trt']}")
            for o in a["outputs"]:
                print(f"[align] {prec} out{o['index']} shape={o['shape']} "
                      f"max_abs={o['max_abs']:.3e} rel_l2={o['rel_l2']:.3e} "
                      f"cosine={o['cosine']:.8f}")
        if b:
            print(f"[bench] {prec} mean={b['mean_ms']:.2f} p50={b['p50_ms']:.2f} "
                  f"p95={b['p95_ms']:.2f} p99={b['p99_ms']:.2f} max={b['max_ms']:.2f} "
                  f"fps={b['fps']}")
        if args.val:
            tiers[prec]["val"] = val_engine(engines[prec], d, args.split, args.imgsz,
                                           args.val_batch, args.val_device, run, prec)
            v = tiers[prec]["val"]
            print(f"[val  ] {prec} {v['seconds']} s：mask mAP50-95={v['mask']['mAP50-95']:.4f} "
                  f"mAP50={v['mask']['mAP50']:.4f} / box mAP50-95={v['box']['mAP50-95']:.4f}"
                  f"（task={v['task']}, names={v['names_count']}）")

    reference = vox.load_reference(REPO_ROOT / "runs" / "segment" / run, args.split)
    ref_mask = ((reference or {}).get("summary") or {}).get("mask", {}).get("mAP50-95")
    if reference:
        print(f"[ref  ] .pt 参照 mask mAP50-95={ref_mask}（{reference['path']}）")
    else:
        print(f"[warn ] 没有 .pt 参照 eval_{args.split}.json，fp32 档第 2 层只报绝对值")
    base_map = (((tiers.get("fp32") or {}).get("val") or {}).get("mask") or {}).get("mAP50-95")

    fails: List[str] = []
    for prec, t in tiers.items():
        ok, why = gate_tier(prec, t, th[prec], base_map, ref_mask)
        t["gate"] = {"ok": ok, "why": why, "thresholds": th[prec]}
        if ok:
            print(f"[ok   ] {prec} 判据通过")
        else:
            print(f"[fail ] {prec}：" + "；".join(why))
            fails += [f"{prec}: {x}" for x in why]

    table = build_table(tiers, build_rep, torch_stats)
    verdict = "pass" if not fails else "fail"
    payload = {
        "run": run, "split": args.split, "imgsz": args.imgsz, "config": cfg,
        "sample": sample_meta, "images_used": names, "reference": reference,
        "torch_gpu_bench": torch_stats, "tiers": tiers, "table": table,
        "build_report": str(build_rep_path) if build_rep else None,
        "verdict": verdict, "failures": fails,
        "seconds": round(time.time() - t_start, 1), "git_head": vox._git_head(),
        "env": vox._env_fingerprint(),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    print(f"[done ] verdict={verdict}，{payload['seconds']} s -> {json_out}")
    if args.markdown:
        print("\n".join(_md(payload, cfg)))
    return 0 if verdict == "pass" else 3


if __name__ == "__main__":
    raise SystemExit(main())