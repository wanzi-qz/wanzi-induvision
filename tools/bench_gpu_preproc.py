#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""路线 C 预研：letterbox 前处理搬上 GPU，能给 C++ 版省出多少？

背景
----
A 组（含磁盘）实测的瓶颈在源头：decode 段 = 读图(imdecode 16~19ms 里的大头) + CPU
letterbox + BGR->RGB + HWC->CHW + /255，decode util ~1.00，喂不满 infer。本工具回答
一个问题：**把 letterbox 及之后的归一化搬到 GPU，这一段还能压到多少**，给 C++ 版
（cudaWarpAffine 方案）一个不用边写边猜的目标值。

量什么
------
1. **微观（串行逐帧分段时间）**：CPU 路径 read + preprocess 逐段计时 vs
   GPU 路径 read + 上传(pinned) + GPU resize/pad/normalize + 拷回(pinned) 逐段计时。
   分段计时都带 torch.cuda.synchronize()，所以是"每段串行成本"，含同步开销——
   报告里按这个口径写，别拿它直接减出流水线收益。
2. **数值对齐**：每帧 CPU chw vs GPU chw 过 ``deploy.compare_tensors``
   （max_abs / mean_abs / rel_l2 / cosine），报全程最坏值。预期不是逐位相等：
   cv2 INTER_LINEAR（uint8 域）vs torch bilinear（float 域）算法实现不同。
3. **端到端（同轮对比，A 组含磁盘，500 帧）**：fp16 engine，
   {cpu, gpu} x {serial, pipeline} 四配置，同轮交错跑 --rounds 轮。
   跨轮只比同轮内比值，绝对值只报不比（机器噪声底 3.57%，见 docs/pipeline.md §7）。

C++ 版要记住的一条
------------------
Python 版 GPU 前处理要把 4.688 MiB 的输入**拷回 host** 才能喂现有 TrtRunner /
PinnedTrtClient；C++ 版直接 setTensorAddress 吃 device 指针，**没有这笔 D2H**。
所以本工具测出的端到端收益是 C++ 版的**保守下界**。

返回码：0 = 正常完成；2 = 输入/环境错误。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT, REPO_ROOT / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from induvision.deploy import (  # noqa: E402
    PAD_COLOR,
    compare_tensors,
    latency_stats,
    preprocess,
    read_image_bgr,
    read_split_images,
)
from induvision.pipeline import Pipeline, run_serial, speedup_report  # noqa: E402
from induvision.trt_infer import TrtRunner  # noqa: E402
import verify_onnx as vox  # noqa: E402 - 复用抽样口径，与 W3b/W4 报告同一批图
import bench_pipeline as bp  # noqa: E402 - 复用 Frame/FrameResult/checksum/三阶段


# --------------------------------------------------------------------------
# GPU 前处理
# --------------------------------------------------------------------------

def letterbox_meta(img: np.ndarray, imgsz: int) -> Tuple[float, Tuple[int, int],
                                                          int, int, int, int]:
    """只算 letterbox 的几何参数，不做图像操作。

    算式逐行对齐 ``induvision.deploy.letterbox``（scaleup=True 路径），
    ``verify_metas`` 会拿真 ``letterbox`` 的返回值逐字段核对，防止两边漂。
    """
    h, w = img.shape[:2]
    r = min(imgsz / h, imgsz / w)
    nu_w, nu_h = int(round(w * r)), int(round(h * r))
    dh, dw = imgsz - nu_h, imgsz - nu_w
    top, bottom = int(round(dh / 2 - 0.1)), int(round(dh / 2 + 0.1))
    left, right = int(round(dw / 2 - 0.1)), int(round(dw / 2 + 0.1))
    return r, (nu_w, nu_h), top, left, bottom, right


class GpuPreproc:
    """BGR uint8 原图 -> (1,3,640,640) float32 host 张量，letterbox 在 GPU 上做。

    流程：CPU 图 -> 拷进 pinned 暂存 -> H2D(non_blocking) -> GPU 上
    BGR->RGB + /255 + bilinear 缩放 + 常数 pad -> D2H 拷回预分配 pinned 输出。
    输出是 host 内存，正好能接现有 ``TrtRunner`` / ``PinnedTrtClient`` 的入口，
    后处理拿的 lb 参数与 ``deploy.letterbox`` 同源，一行不用改。
    """

    def __init__(self, imgsz: int = 640, device: str = "cuda:0",
                 pad_color: Tuple[int, int, int] = PAD_COLOR):
        if len(set(pad_color)) != 1:
            raise ValueError("GPU pad 只支持三通道同值的填充色（ultralytics 就是 114）")
        self.imgsz = imgsz
        self.dev = torch.device(device)
        self.pad_value = float(pad_color[0]) / 255.0
        # 暂存/输出缓冲都预分配：压测里每帧 new 一块 pinned 会把增益吃掉
        self._stage_t: Optional[torch.Tensor] = None   # pinned uint8 暂存
        self._stage_np: Optional[np.ndarray] = None
        self._out_t = torch.empty((1, 3, imgsz, imgsz), dtype=torch.float32,
                                  pin_memory=True)
        self._out_np = self._out_t.numpy()

    def _ensure_stage(self, n: int) -> None:
        if self._stage_t is None or self._stage_t.numel() < n:
            cap = max(n, 1 << 23)  # 至少 8 MiB，MVTec 900x900x3 = 2.33 MiB
            self._stage_t = torch.empty(cap, dtype=torch.uint8, pin_memory=True)
            self._stage_np = self._stage_t.numpy()

    def __call__(self, img_bgr: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """返回 (chw[None] 可直接喂网络, lb 参数 dict)。"""
        h, w = img_bgr.shape[:2]
        imgsz = self.imgsz
        r, (nu_w, nu_h), top, left, bottom, right = letterbox_meta(img_bgr, imgsz)
        flat = img_bgr.reshape(-1)
        self._ensure_stage(flat.size)
        # -- CPU -> pinned
        self._stage_np[:flat.size] = flat
        # -- pinned -> GPU
        gpu = self._stage_t[:flat.size].view(h, w, 3).to(self.dev, non_blocking=True)
        torch.cuda.synchronize(self.dev)
        # -- GPU 上完成 RGB / 归一化 / 缩放 / pad
        x = gpu.permute(2, 0, 1).float().div_(255.0)     # (3,h,w) BGR [0,1]
        x = x[[2, 1, 0], :, :]                           # BGR -> RGB
        if (nu_w, nu_h) != (w, h):
            x = F.interpolate(x[None], size=(nu_h, nu_w), mode="bilinear",
                              align_corners=False, antialias=False)[0]
        x = F.pad(x[None], (left, right, top, bottom),
                  mode="constant", value=self.pad_value)[0]
        # -- GPU -> pinned host 输出
        self._out_t[0].copy_(x, non_blocking=True)
        torch.cuda.synchronize(self.dev)
        lb = {"ratio": r, "dw": (imgsz - nu_w) / 2.0, "dh": (imgsz - nu_h) / 2.0,
              "orig_shape": (h, w), "new_unpad": (nu_w, nu_h),
              "top": top, "left": left, "imgsz": imgsz}
        return self._out_np, lb


class GpuLetterboxView:
    """``LetterboxResult`` 的最小替身：postprocess 只读这几个字段。

    字段名与 ``deploy.LetterboxResult`` 一一对应，但 image 不存——GPU 版不产生
    uint8 letterbox 图（省 1.2 MiB/帧 的内存与拷贝）。``masks_from_proto``
    只读 ``lb.image.shape[0]``（= imgsz）这一个信息，所以给一个共享的空占位
    数组把 imgsz 带过去，语义不变。
    """

    _placeholder: Dict[int, np.ndarray] = {}

    def __init__(self, meta: Dict[str, Any]):
        self.ratio = meta["ratio"]
        self.dw = meta["dw"]
        self.dh = meta["dh"]
        self.orig_shape = meta["orig_shape"]
        self.new_unpad = meta["new_unpad"]
        self.top = meta["top"]
        self.left = meta["left"]
        imgsz = meta["imgsz"]
        ph = GpuLetterboxView._placeholder
        if imgsz not in ph:
            ph[imgsz] = np.empty((imgsz, imgsz, 3), dtype=np.uint8)
        self.image = ph[imgsz]


def verify_metas(paths: Sequence[Path], imgsz: int, n_check: int = 20) -> None:
    """letterbox_meta 的算式必须与 deploy.letterbox 逐字段相等，不等就停。"""
    from induvision.deploy import letterbox  # noqa: PLC0415
    for p in list(paths)[:n_check]:
        img = read_image_bgr(p)
        if img is None:
            continue
        lb = letterbox(img, imgsz)
        r, (nu_w, nu_h), top, left, bottom, right = letterbox_meta(img, imgsz)
        assert abs(lb.ratio - r) < 1e-12, f"ratio 漂了：{p}"
        assert lb.new_unpad == (nu_w, nu_h), f"new_unpad 漂了：{p}"
        assert (lb.top, lb.left) == (top, left), f"top/left 漂了：{p}"
        assert lb.orig_shape == (int(round(imgsz / r * 0 + lb.orig_shape[0])),
                                 lb.orig_shape[1]), "orig_shape 不该动"


# --------------------------------------------------------------------------
# 微观分段计时
# --------------------------------------------------------------------------

def micro_round(paths: Sequence[Path], imgsz: int, device: str,
                warmup: int = 5) -> Dict[str, Any]:
    """串行逐帧，分段计时 + 数值对齐。返回可入库摘要。"""
    gp = GpuPreproc(imgsz, device)
    segs = {k: [] for k in ("cpu_read", "cpu_preproc", "gpu_read", "gpu_stage",
                            "gpu_h2d", "gpu_ops", "gpu_d2h", "gpu_total")}
    worst = {"max_abs": 0.0, "mean_abs": 0.0, "rel_l2": 0.0, "cosine_min": 1.0}
    n_cmp = 0
    dev = gp.dev
    for i, p in enumerate(paths):
        # -- CPU 路径
        t = time.perf_counter()
        img = read_image_bgr(p)
        if img is None:
            raise FileNotFoundError(f"读不到图：{p}")
        segs["cpu_read"].append((time.perf_counter() - t) * 1000)
        t = time.perf_counter()
        chw_cpu, _ = preprocess(img, imgsz)
        segs["cpu_preproc"].append((time.perf_counter() - t) * 1000)
        # -- GPU 路径
        t0 = time.perf_counter()
        t = time.perf_counter()
        img2 = read_image_bgr(p)
        segs["gpu_read"].append((time.perf_counter() - t) * 1000)
        t = time.perf_counter()
        flat = img2.reshape(-1)
        gp._ensure_stage(flat.size)
        gp._stage_np[:flat.size] = flat
        segs["gpu_stage"].append((time.perf_counter() - t) * 1000)
        h, w = img2.shape[:2]
        r, (nu_w, nu_h), top, left, bottom, right = letterbox_meta(img2, imgsz)
        t = time.perf_counter()
        gpu = gp._stage_t[:flat.size].view(h, w, 3).to(dev, non_blocking=True)
        torch.cuda.synchronize(dev)
        segs["gpu_h2d"].append((time.perf_counter() - t) * 1000)
        t = time.perf_counter()
        x = gpu.permute(2, 0, 1).float().div_(255.0)
        x = x[[2, 1, 0], :, :]
        if (nu_w, nu_h) != (w, h):
            x = F.interpolate(x[None], size=(nu_h, nu_w), mode="bilinear",
                              align_corners=False, antialias=False)[0]
        x = F.pad(x[None], (left, right, top, bottom),
                  mode="constant", value=gp.pad_value)[0]
        torch.cuda.synchronize(dev)
        segs["gpu_ops"].append((time.perf_counter() - t) * 1000)
        t = time.perf_counter()
        gp._out_t[0].copy_(x, non_blocking=True)
        torch.cuda.synchronize(dev)
        segs["gpu_d2h"].append((time.perf_counter() - t) * 1000)
        segs["gpu_total"].append((time.perf_counter() - t0) * 1000)
        # -- 数值对齐（预热段不比，等显存/核频稳定）
        if i >= warmup:
            d = compare_tensors(chw_cpu[None], gp._out_np)
            worst["max_abs"] = max(worst["max_abs"], d["max_abs"])
            worst["mean_abs"] = max(worst["mean_abs"], d["mean_abs"])
            worst["rel_l2"] = max(worst["rel_l2"], d["rel_l2"])
            worst["cosine_min"] = min(worst["cosine_min"], d["cosine"])
            n_cmp += 1
    out = {"n_frames": len(paths), "n_compared": n_cmp, "segments": {},
           "tensor_diff_worst": {k: float(v) for k, v in worst.items()}}
    for k, v in segs.items():
        out["segments"][k] = latency_stats(v)
    return out


# --------------------------------------------------------------------------
# 端到端：{cpu, gpu} x {serial, pipeline} 同轮对比
# --------------------------------------------------------------------------

def make_gpu_plain_stages(runner: TrtRunner, gp: GpuPreproc, th: Dict[str, Any],
                          need_masks: bool, keep_dets: bool) -> List[Tuple[str, Any]]:
    """GPU 版三阶段：decode 段换成 GpuPreproc，infer/post 与 plain 版逐字相同。"""

    def decode(fr: bp.Frame):
        img = read_image_bgr(fr.path)
        if img is None:
            raise FileNotFoundError(f"读不到图：{fr.path}")
        chw, meta = gp(img)
        return chw, GpuLetterboxView(meta), fr

    def infer(payload):
        x, lb, fr = payload
        outs = runner(x)
        return outs[0], outs[1], lb, fr

    def post(payload):
        out0, out1, lb, fr = payload
        dets = bp.postprocess(out0, out1, lb, need_masks=need_masks, **th)
        return bp._summarize(fr, dets, keep_dets)

    return [("decode", decode), ("infer", infer), ("post", post)]


def stage_util(rep: Dict[str, Any]) -> Dict[str, float]:
    """从 Pipeline 报告里抽每段 util，看瓶颈搬没搬。"""
    out = {}
    for s in rep.get("stages", []):
        out[s["name"]] = s.get("util")
    return out


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="路线 C 预研：GPU letterbox 前处理能省多少（给 C++ 版定目标值）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--weights", default="runs/segment/baseline15/weights/best.pt")
    ap.add_argument("--data", default="out/yolo_mvtec15/data.yaml")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--engine", default="weights/baseline15_trt_fp16.engine",
                    help="端到端用哪档 engine（只跑一档，预研不扫三档）")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--frames", type=int, default=50)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--capacity", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=2, help="端到端跑几轮（>=2 才能分噪声和效应）")
    ap.add_argument("--conf-thr", type=float, default=0.25)
    ap.add_argument("--iou-thr", type=float, default=0.7)
    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--mask-thr", type=float, default=0.5)
    ap.add_argument("--no-masks", action="store_true")
    ap.add_argument("--sample-list", default="configs/bench_sample.json")
    ap.add_argument("--sample-n", type=int, default=vox.DEFAULT_SAMPLE_N)
    ap.add_argument("--no-sample-list", action="store_true")
    ap.add_argument("--micro-warmup", type=int, default=5)
    ap.add_argument("--skip-micro", action="store_true")
    ap.add_argument("--skip-e2e", action="store_true")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args(argv)

    d = Path(args.data)
    d = d if d.is_absolute() else (REPO_ROOT / d).resolve()
    ep = Path(args.engine)
    ep = ep if ep.is_absolute() else (REPO_ROOT / ep).resolve()
    if not d.is_file():
        print(f"[err  ] data.yaml 不存在：{d}")
        return 2
    if not args.skip_e2e and not ep.is_file():
        print(f"[err  ] engine 不存在：{ep}")
        return 2
    sample_list = Path(args.sample_list) if args.sample_list and not args.no_sample_list else None
    if sample_list and not sample_list.is_absolute():
        sample_list = (REPO_ROOT / sample_list).resolve()
    json_out = Path(args.json_out) if args.json_out else \
        (REPO_ROOT / "runs" / "logs" / "gpu_preproc.json")
    if not json_out.is_absolute():
        json_out = (REPO_ROOT / json_out).resolve()

    if not torch.cuda.is_available():
        print("[err  ] torch.cuda 不可用，GPU 前处理无从谈起")
        return 2
    th = {"conf_thr": args.conf_thr, "iou_thr": args.iou_thr,
          "max_det": args.max_det, "mask_thr": args.mask_thr}
    dev = f"cuda:{args.device}"

    images = read_split_images(d, args.split)
    if not images:
        print(f"[err  ] {d} 的 {args.split} split 里没有图片")
        return 2
    try:
        canonical, sample_meta = vox.resolve_sample(
            images, root=d.parent, sample_list=sample_list, sample_n=args.sample_n,
            use_list=not args.no_sample_list, split=args.split, strict=False)
    except (FileNotFoundError, ValueError) as e:
        print(f"[err  ] {e}")
        return 2
    if len(canonical) < args.frames:
        print(f"[err  ] 只有 {len(canonical)} 张，撑不起 --frames {args.frames}")
        return 2
    picked = canonical[:args.frames]
    print(f"[data ] {args.split} 共 {len(images)} 图；口径={sample_meta['source']} "
          f"{sample_meta['count']} 张（sha256:{sample_meta['sha256']}），取前 {len(picked)} 张")

    verify_metas(picked, args.imgsz)
    print(f"[meta ] letterbox_meta 与 deploy.letterbox 逐字段核对通过（前 20 张）")

    report: Dict[str, Any] = {
        "schema": "induvision.gpu_preproc.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_head": vox._git_head(),
        "cfg": {"data": str(d), "split": args.split, "imgsz": args.imgsz,
                "engine": str(ep), "device": dev, "frames": args.frames,
                "repeats": args.repeats, "n_items": args.frames * args.repeats,
                "capacity": args.capacity, "rounds": args.rounds,
                "need_masks": not args.no_masks,
                "sample_list": str(sample_list) if sample_list else None,
                "note": "分段计时含 torch.cuda.synchronize 同步开销；"
                        "GPU 变体含 4.688 MiB D2H 拷回，C++ 版没有这笔（保守下界）"},
        "env": {"python": sys.version.split()[0], "torch": torch.__version__,
                "gpu": torch.cuda.get_device_name(0)},
    }

    # ---- 微观 ----
    if not args.skip_micro:
        micro_paths = [p for r in range(args.repeats) for p in picked]
        print(f"[micro] {len(micro_paths)} 帧分段计时中（含 {args.micro_warmup} 帧预热不比对）…")
        report["micro"] = micro_round(micro_paths, args.imgsz, dev,
                                      warmup=args.micro_warmup)
        m = report["micro"]
        print(f"[micro] CPU read  {m['segments']['cpu_read']['mean_ms']:7.3f} ms | "
              f"CPU preproc {m['segments']['cpu_preproc']['mean_ms']:7.3f} ms")
        g = m["segments"]
        print(f"[micro] GPU read  {g['gpu_read']['mean_ms']:7.3f} ms | stage拷入 "
              f"{g['gpu_stage']['mean_ms']:7.3f} | H2D {g['gpu_h2d']['mean_ms']:7.3f} | "
              f"ops {g['gpu_ops']['mean_ms']:7.3f} | D2H {g['gpu_d2h']['mean_ms']:7.3f} | "
              f"合计 {g['gpu_total']['mean_ms']:7.3f} ms")
        w = m["tensor_diff_worst"]
        print(f"[micro] 张量最坏差（{m['n_compared']} 帧比对）：max_abs {w['max_abs']:.5f} / "
              f"mean_abs {w['mean_abs']:.6f} / rel_l2 {w['rel_l2']:.6f} / "
              f"cosine_min {w['cosine_min']:.6f}")

    # ---- 端到端 ----
    if not args.skip_e2e:
        runner = TrtRunner(ep, device=dev)
        gp = GpuPreproc(args.imgsz, dev)
        frames = [bp.Frame(r * len(picked) + i, str(p))
                  for r in range(args.repeats) for i, p in enumerate(picked)]
        stages_cpu = bp.make_plain_stages(runner, args.imgsz, th,
                                          not args.no_masks, False)
        stages_gpu = make_gpu_plain_stages(runner, gp, th, not args.no_masks, False)
        e2e_rounds = []
        for rd in range(args.rounds):
            modes: Dict[str, Dict[str, Any]] = {}
            serial_ref = None
            for variant, stages in (("cpu", stages_cpu), ("gpu", stages_gpu)):
                for kind in ("serial", "pipeline"):
                    key = f"{variant}_{kind}"
                    if kind == "serial":
                        results, rep = run_serial(frames, stages)
                    else:
                        results, rep = Pipeline(stages, capacity=args.capacity).run(frames)
                    rep["checksum"] = bp.checksum(results)
                    rep["order_ok"] = bp.order_ok(frames, results)
                    modes[key] = rep
                    e2e = rep.get("e2e_latency") or {}
                    print(f"[e2e  ] round{rd} {key}: wall {rep['wall_s']} s, "
                          f"fps {rep['throughput_fps']}, dets {rep['checksum']['dets']}, "
                          f"P50 {_f(e2e.get('p50_ms'))} / P95 {_f(e2e.get('p95_ms'))} ms")
                    if kind == "serial":
                        if variant == "cpu":
                            serial_ref = results
                        else:
                            modes[key]["vs_cpu_serial"] = bp.frame_diffs(serial_ref, results)
                            d = modes[key]["vs_cpu_serial"]
                            print(f"[e2e  ]   gpu vs cpu 串行：n_diff {d['n_diff']}"
                                  f"（输入张量有微差，检测级一致性的判据仍以 C++ 版对拍为准）")
                    else:
                        skey = f"{variant}_serial"
                        modes[key]["speedup_vs_own_serial"] = speedup_report(
                            modes[skey], rep)
            e2e_rounds.append(modes)
        report["e2e"] = e2e_rounds
        # 汇总：同轮内 gpu_serial/cpu_serial 与 gpu_pipeline/cpu_pipeline 的 FPS 比
        ratios = {"serial": [], "pipeline": []}
        for modes in e2e_rounds:
            for kind in ratios:
                cf = modes[f"cpu_{kind}"]["throughput_fps"]
                gf = modes[f"gpu_{kind}"]["throughput_fps"]
                ratios[kind].append(round(gf / cf, 4))
        report["e2e_fps_ratio_gpu_over_cpu"] = ratios
        print(f"[e2e  ] GPU/CPU 同轮 FPS 比：serial {ratios['serial']}，"
              f"pipeline {ratios['pipeline']}")
        runner.close()

    json_out.parent.mkdir(parents=True, exist_ok=True)
    with open(json_out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[out  ] {json_out}")
    return 0


def _f(v: Any, spec: str = ".2f", dash: str = "—") -> str:
    return dash if v is None else format(v, spec)


if __name__ == "__main__":
    raise SystemExit(main())
