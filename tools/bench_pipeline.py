#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""三阶段流水线压测与验收（W4 第一步）。

量的东西
--------
把"解码+前处理 -> TRT 推理 -> 后处理"三段拆成三条线程（``induvision.pipeline.Pipeline``），
和同样三段的单线程串行版（``run_serial``）在**同一批图、同一台机器、同一轮**里对跑，
报加速比、每阶段利用率、队列背压次数。缓冲分两种：

* ``plain``：走 ``TrtRunner.__call__``，每帧新建可分页缓冲。这是改造前的基线。
* ``pinned``：走 ``induvision.pinned.PinnedTrtClient``，pinned host + 常驻 device +
  池化输出（``--pool-size``），H2D/D2H 全 non_blocking、每帧只 synchronize 一次。

判据（顺序就是排查顺序）
------------------------
1. **一致性与保序**：流水线结果条数 == 输入条数、idx 顺序不变、逐帧 (检测数, 掩码像素数,
   分数和) 与串行版完全相同。流水线要是把顺序或结果搞错了，快多少都没意义。
2. **加速比**：``pipeline_fps / serial_fps >= --min-speedup``（默认 1.0）。三段耗时不均时
   理论上限是"最慢段的串行占比"，所以这个数不是越大越好，而是**要和 util 表对着看**：
   瓶颈段 util 接近 1、其余段在它前面的队列上 ``overflow_waits`` 涨，就说明流水线打满了。
3. **后处理对拍**（``--check-ultralytics``）：本仓库自己实现的 NMS/掩码解码 vs
   ``ultralytics predict``，同一份 engine、同一批图。判据是 match_rate / 框 IoU / 掩码 IoU，
   掩码那条给得松（本实现只放大裁剪后的小块，边界有 1~2 px 的设计内差异，
   见 ``induvision.pipeline.masks_from_proto`` 的文档）。

**别拿 docs/deploy.md §8.2 的单帧延迟当分母**：那组数字输入常驻显存、不含 H2D/D2H、
不含前后处理。本工具的量的是端到端每帧墙钟，两个口径不可直接比。

返回码：0 = 全部通过；2 = 输入/环境错误；3 = 有判据未通过。
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT, REPO_ROOT / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from induvision.deploy import (  # noqa: E402 - 需要先插好 sys.path
    latency_stats,
    preprocess,
    read_image_bgr,
    read_split_images,
    relative_to_root,
)
from induvision.pipeline import (  # noqa: E402
    Detection,
    Pipeline,
    match_detections,
    postprocess,
    run_serial,
    speedup_report,
)
from induvision.prof import ResourceSampler  # noqa: E402
from induvision.trt_infer import TrtRunner  # noqa: E402
import verify_onnx as vox  # noqa: E402 - 复用抽样口径，保证与 W3b 报告同一批图

#: 默认判据。对拍那三条是按"两套后处理语义相同、只有掩码解码实现有别"定的：
#: 框走的是同一套 letterbox 逆变换，float32/float64 舍入差在 1e-2 像素量级，
#: 所以框 IoU 卡 0.99；掩码因为只放大裁剪小块，边界差 1~2 px，卡 0.85。
#: 实测数字见 runs/logs/pipeline_baseline15.json 与 docs/pipeline.md。
DEFAULT_MIN_SPEEDUP = 1.0
DEFAULT_MIN_MATCH_RATE = 0.98
DEFAULT_MIN_BOX_IOU = 0.99
DEFAULT_MIN_MASK_IOU = 0.85


# --------------------------------------------------------------------------
# 帧与结果
# --------------------------------------------------------------------------

@dataclass
class Frame:
    """流水线的一个输入项。``chw``/``lb`` 只在 --preload 时预先填好。"""

    idx: int
    path: str
    chw: Optional[np.ndarray] = None
    lb: Any = None


@dataclass
class FrameResult:
    """一个输出项。掩码只留统计量，不留数组 —— 900x900 的 bool 一张就 0.8 MiB，
    300 帧乘几个缺陷直接吃掉几 GB，压测跑完机器就卡了。对拍那一轮才带 dets。"""

    idx: int
    path: str
    n_det: int
    mask_pixels: int
    score_sum: float
    dets: Optional[List[Detection]] = None


def _summarize(fr: Frame, dets: Sequence[Detection], keep_dets: bool) -> FrameResult:
    return FrameResult(
        idx=fr.idx, path=fr.path, n_det=len(dets),
        mask_pixels=int(sum(int(np.count_nonzero(d.mask)) for d in dets if d.mask is not None)),
        score_sum=round(float(sum(d.score for d in dets)), 6),
        dets=list(dets) if keep_dets else None,
    )


def checksum(results: Sequence[FrameResult]) -> Dict[str, Any]:
    return {
        "frames": len(results),
        "dets": int(sum(r.n_det for r in results)),
        "mask_pixels": int(sum(r.mask_pixels for r in results)),
        "score_sum": round(float(sum(r.score_sum for r in results)), 4),
    }


def order_ok(frames: Sequence[Frame], results: Sequence[FrameResult]) -> bool:
    return [r.idx for r in results] == [f.idx for f in frames]


def frame_diffs(a: Sequence[FrameResult], b: Sequence[FrameResult],
                limit: int = 5) -> Dict[str, Any]:
    """逐帧比两份结果，返回不一致条数与前几个样本（排查用）。"""
    am = {r.idx: r for r in a}
    bm = {r.idx: r for r in b}
    bad: List[Dict[str, Any]] = []
    for i in sorted(set(am) & set(bm)):
        x, y = am[i], bm[i]
        if (x.n_det, x.mask_pixels, x.score_sum) != (y.n_det, y.mask_pixels, y.score_sum):
            if len(bad) < limit:
                bad.append({"idx": i, "serial": [x.n_det, x.mask_pixels, x.score_sum],
                            "pipeline": [y.n_det, y.mask_pixels, y.score_sum]})
    return {"n_diff": len(bad), "only_serial": sorted(set(am) - set(bm)),
            "only_pipeline": sorted(set(bm) - set(am)), "samples": bad}


# --------------------------------------------------------------------------
# 三阶段构造
# --------------------------------------------------------------------------

def make_plain_stages(runner: TrtRunner, imgsz: int, th: Dict[str, Any],
                      need_masks: bool, keep_dets: bool) -> List[Tuple[str, Any]]:
    """对照组：``TrtRunner.__call__``，每帧新建可分页缓冲。"""

    def decode(fr: Frame):
        if fr.chw is not None and fr.lb is not None:
            return fr.chw[None], fr.lb, fr
        img = read_image_bgr(fr.path)
        if img is None:
            raise FileNotFoundError(f"读不到图：{fr.path}")
        chw, lb = preprocess(img, imgsz)
        return chw[None], lb, fr

    def infer(payload):
        x, lb, fr = payload
        outs = runner(x)
        return outs[0], outs[1], lb, fr

    def post(payload):
        out0, out1, lb, fr = payload
        dets = postprocess(out0, out1, lb, need_masks=need_masks, **th)
        return _summarize(fr, dets, keep_dets)

    return [("decode", decode), ("infer", infer), ("post", post)]


def make_pinned_stages(client, imgsz: int, th: Dict[str, Any],
                       need_masks: bool, keep_dets: bool) -> List[Tuple[str, Any]]:
    """pinned 版：令牌贯穿三段，前处理直接写进该 slot 的 pinned 输入缓冲。"""

    def decode(fr: Frame):
        tok = client.acquire()
        try:
            if fr.chw is not None:
                chw, lb = fr.chw, fr.lb
            else:
                img = read_image_bgr(fr.path)
                if img is None:
                    raise FileNotFoundError(f"读不到图：{fr.path}")
                chw, lb = preprocess(img, imgsz)
            np.copyto(client.input_view(tok), chw)  # 零拷贝写入口
        except BaseException:
            client.release(tok)  # 出错也要还令牌，不然池子越用越小
            raise
        return tok, lb, fr

    def infer(payload):
        tok, lb, fr = payload
        return tok, client.forward(tok), lb, fr

    def post(payload):
        tok, out, lb, fr = payload
        try:
            dets = postprocess(out.arrays[0], out.arrays[1], lb,
                               need_masks=need_masks, **th)
            return _summarize(fr, dets, keep_dets)
        finally:
            client.release(tok)

    return [("decode", decode), ("infer", infer), ("post", post)]


def build_frames(paths: Sequence[Path], repeats: int, imgsz: int,
                 preload: bool) -> List[Frame]:
    frames: List[Frame] = []
    cache: Dict[int, Tuple[np.ndarray, Any]] = {}
    for r in range(repeats):
        for i, p in enumerate(paths):
            idx = r * len(paths) + i
            if not preload:
                frames.append(Frame(idx, str(p)))
                continue
            if i not in cache:
                img = read_image_bgr(p)
                if img is None:
                    raise FileNotFoundError(f"读不到图：{p}")
                cache[i] = preprocess(img, imgsz)
            chw, lb = cache[i]
            frames.append(Frame(idx, str(p), chw, lb))
    return frames


# --------------------------------------------------------------------------
# 对拍 ultralytics
# --------------------------------------------------------------------------

def ultra_detections(model, path: str, imgsz: int, th: Dict[str, Any],
                     device: str) -> List[Detection]:
    """``ultralytics predict`` 的结果转成本仓库的 Detection，好塞进 match_detections。"""
    import cv2  # noqa: PLC0415

    r = model.predict(source=str(path), imgsz=imgsz, conf=th["conf_thr"],
                      iou=th["iou_thr"], max_det=th["max_det"],
                      device=device, verbose=False)[0]
    oh, ow = (int(v) for v in r.orig_shape)
    if r.boxes is None or len(r.boxes) == 0:
        return []
    xyxy = r.boxes.xyxy.detach().cpu().numpy().astype(np.float64)
    conf = r.boxes.conf.detach().cpu().numpy().astype(np.float64)
    cls = r.boxes.cls.detach().cpu().numpy().astype(int)
    masks = r.masks.data.detach().cpu().numpy() if r.masks is not None else None
    out: List[Detection] = []
    for i in range(len(conf)):
        m = None
        if masks is not None and i < len(masks):
            m = masks[i] > 0.5
            if m.shape != (oh, ow):
                # ultralytics 有时给的是 640 尺度的掩码，放大到原图才能和我们的比
                m = cv2.resize(m.astype(np.uint8), (ow, oh),
                               interpolation=cv2.INTER_NEAREST).astype(bool)
        out.append(Detection(score=float(conf[i]), cls=int(cls[i]), box=xyxy[i],
                             box640=np.zeros(4, dtype=np.float64), mask=m))
    return out


def check_against_ultralytics(weights: Path, engine_path: Path, paths: Sequence[Path],
                              imgsz: int, th: Dict[str, Any], device: str,
                              our_dets: Sequence[Sequence[Detection]]) -> Dict[str, Any]:
    """逐图对拍。``our_dets`` 是本实现按同样顺序跑出来的结果（同一批 paths）。"""
    from ultralytics import YOLO  # noqa: PLC0415

    t0 = time.perf_counter()
    model = YOLO(str(engine_path))
    per_img: List[Dict[str, Any]] = []
    for k, p in enumerate(paths):
        theirs = ultra_detections(model, str(p), imgsz, th, device)
        ours = list(our_dets[k])
        m = match_detections(ours, theirs)
        # 腐蚀 1 px 再算一次：原始 IoU 分不清"解码错"和"边界亚像素差"，
        # 小掩码上 1 px 的边差就能把 IoU 拉下来一大截，两个口径要一起看。
        m1 = match_detections(ours, theirs, mask_erode_px=1)
        per_img.append({
            "path": str(p),
            **{kk: m[kk] for kk in ("n_ours", "n_theirs", "n_matched", "match_rate",
                                    "mean_box_iou", "mean_mask_iou", "mean_abs_score_diff")},
            "mean_mask_iou_erode1": m1["mean_mask_iou"],
            "px_ours": int(sum(int(np.count_nonzero(d.mask)) for d in ours
                               if d.mask is not None)),
            "px_theirs": int(sum(int(np.count_nonzero(d.mask)) for d in theirs
                                  if d.mask is not None)),
        })

    def _mean(key: str) -> Optional[float]:
        vals = [d[key] for d in per_img if d[key] is not None]
        return round(float(np.mean(vals)), 4) if vals else None

    n_ours = int(sum(d["n_ours"] for d in per_img))
    n_theirs = int(sum(d["n_theirs"] for d in per_img))
    n_matched = int(sum(d["n_matched"] for d in per_img))
    px_o = int(sum(d["px_ours"] for d in per_img))
    px_t = int(sum(d["px_theirs"] for d in per_img))
    return {
        "images": len(per_img), "n_ours": n_ours, "n_theirs": n_theirs,
        "n_matched": n_matched,
        "match_rate_mean": _mean("match_rate"),
        "match_rate_min": round(float(min(d["match_rate"] for d in per_img)), 4) if per_img else None,
        "mean_box_iou": _mean("mean_box_iou"),
        "min_box_iou": (round(float(min(d["mean_box_iou"] for d in per_img
                                        if d["mean_box_iou"] is not None)), 4)
                        if any(d["mean_box_iou"] is not None for d in per_img) else None),
        "min_mask_iou": (round(float(min(d["mean_mask_iou"] for d in per_img
                                         if d["mean_mask_iou"] is not None)), 4)
                         if any(d["mean_mask_iou"] is not None for d in per_img) else None),
        "mean_mask_iou": _mean("mean_mask_iou"),
        "mean_mask_iou_erode1": _mean("mean_mask_iou_erode1"),
        "mean_abs_score_diff": _mean("mean_abs_score_diff"),
        "imgs_not_full_match": int(sum(1 for d in per_img if d["match_rate"] < 1.0)),
        # 掩码像素总量之比：系统性偏大/偏小说明坐标映射或阈值口径错了，
        # 这个数就是靠它抓到 masks_from_proto 把裁剪区拉伸到框尺寸的 bug 的。
        "px_ours": px_o, "px_theirs": px_t,
        "px_ratio": round(px_o / px_t, 4) if px_t else None,
        "elapsed_s": round(time.perf_counter() - t0, 2),
        "per_image": per_img,
    }


# --------------------------------------------------------------------------
# 判据
# --------------------------------------------------------------------------

def gate_tier(tier: Dict[str, Any], th_gate: Dict[str, Optional[float]]) -> Dict[str, Any]:
    """逐条判，返回 {ok, passed[], failed[]}。failed 里的每条都带实测值与门槛。"""
    passed: List[str] = []
    failed: List[Dict[str, Any]] = []

    def want(cond: bool, label: str, **vals: Any) -> None:
        if cond:
            passed.append(label)
        else:
            failed.append({"gate": label, **vals})

    present = sorted({k.rsplit("_", 1)[0] for k in tier["modes"]})
    for mode in present:
        s = tier["modes"][f"{mode}_serial"]
        p = tier["modes"][f"{mode}_pipeline"]
        want(s["order_ok"] and p["order_ok"], f"{mode}:保序",
             serial=s["order_ok"], pipeline=p["order_ok"])
        want(p["n_results"] == p["n_items"], f"{mode}:结果条数",
             n_items=p["n_items"], n_results=p["n_results"])
        d = tier["consistency"][mode]
        want(d["same_checksum"] and d["n_diff"] == 0, f"{mode}:与串行逐帧一致", **d)

    prim = "pinned" if "pinned" in present else (present[0] if present else None)
    sp = tier["speedup"].get(prim, {}).get("speedup") if prim else None
    ms = th_gate.get("min_speedup")
    if ms is not None:
        want(sp is not None and sp >= ms, f"{prim}:加速比", speedup=sp, min=ms)

    chk = tier.get("check")
    if chk:
        for key, gk in (("match_rate_mean", "min_match_rate"),
                        ("mean_box_iou", "min_box_iou"),
                        ("mean_mask_iou", "min_mask_iou")):
            g = th_gate.get(gk)
            if g is None:
                continue
            v = chk.get(key)
            want(v is not None and v >= g, f"对拍:{key}", value=v, min=g)

    return {"ok": not failed, "passed": passed, "failed": failed}


# --------------------------------------------------------------------------
# Markdown
# --------------------------------------------------------------------------

def _f(v: Any, spec: str = ".2f", dash: str = "—") -> str:
    return dash if v is None else format(v, spec)


def _md(rep: Dict[str, Any], cfg: Dict[str, Any]) -> List[str]:
    L: List[str] = []
    L.append(f"# 流水线压测｜{rep['run']}")
    L.append("")
    L.append(f"- 生成时间：{rep['generated_at']}　git：`{rep['git_head']}`")
    sha = str((rep.get('sample') or {}).get('sha256') or '')
    L.append(f"- 帧数：{cfg['frames']} x {cfg['repeats']} = {cfg['n_items']}"
             f"（清单口径 {rep.get('sample', {}).get('source')}，sha256 {sha[:16]}，"
             f"preload={cfg['preload']}）")
    L.append(f"- 后处理阈值：conf {cfg['conf_thr']} / iou {cfg['iou_thr']} / "
             f"max_det {cfg['max_det']} / mask {cfg['mask_thr']}，掩码解码={cfg['need_masks']}")
    L.append(f"- 队列容量 {cfg['capacity']}，pinned 池 {cfg['pool_size']}")
    L.append("")
    for prec, t in rep["tiers"].items():
        L.append(f"## {prec}（{t['engine_mib']} MiB）")
        L.append("")
        L.append("| 缓冲 | 模式 | 墙钟 s | FPS | 加速比 | decode ms/util | infer ms/util | post ms/util |")
        L.append("|---|---|---|---|---|---|---|---|")
        present = sorted({k.rsplit("_", 1)[0] for k in t["modes"]})
        for mode in present:
            for kind in ("serial", "pipeline"):
                r = t["modes"][f"{mode}_{kind}"]
                st = {s["name"]: s for s in r["stages"]}
                cell = []
                for nm in ("decode", "infer", "post"):
                    s = st.get(nm) or {}
                    cell.append(f"{_f(s.get('mean_ms'))} / {_f(s.get('util'), '.2f')}")
                sp = _f(t["speedup"][mode].get("speedup"), ".3f") if kind == "pipeline" else "—"
                L.append(f"| {mode} | {kind} | {_f(r['wall_s'])} | {_f(r['throughput_fps'])} "
                         f"| {sp} | {cell[0]} | {cell[1]} | {cell[2]} |")
        L.append("")
        L.append("| 缓冲 | 模式 | 端到端 P50 ms | P95 ms | P99 ms | max ms |")
        L.append("|---|---|---|---|---|---|")
        for mode in present:
            for kind in ("serial", "pipeline"):
                e = (t["modes"][f"{mode}_{kind}"].get("e2e_latency") or {})
                L.append(f"| {mode} | {kind} | {_f(e.get('p50_ms'))} | {_f(e.get('p95_ms'))} "
                         f"| {_f(e.get('p99_ms'))} | {_f(e.get('max_ms'))} |")
        L.append("")
        L.append("> 端到端延迟含排队等待，所以流水线的 P50 比串行大是**正常的**："
                 "流水线拿单帧延迟换吞吐。这两个数必须一起报，只报 FPS 会掩盖延迟恶化。")
        L.append("")
        rz_rows = [(k, v.get("resources")) for k, v in t["modes"].items() if v.get("resources")]
        if rz_rows:
            L.append("| 组合 | CPU 核当量 均值/峰值 | GPU util 均值/峰值 % | 显存 MiB | 采样点 |")
            L.append("|---|---|---|---|---|")
            for k, rz in rz_rows:
                L.append(f"| {k} | {_f(rz.get('cpu_cores_mean'))} / {_f(rz.get('cpu_cores_max'))} "
                         f"| {_f(rz.get('gpu_util_mean_pct'), '.1f')} / "
                         f"{_f(rz.get('gpu_util_max_pct'), '.1f')} "
                         f"| {_f(rz.get('gpu_mem_used_max_mib'), '.0f')} "
                         f"| {rz.get('gpu_samples')} |")
            L.append("")
            L.append(f"> 资源采样间隔 {rz_rows[0][1].get('interval_s')} s。"
                     "**这一组不能和没开采样的那轮混着比 FPS** —— nvidia-smi 子进程自己也吃 CPU。")
            L.append("")
        pg = t.get("pinned_gain") or {}
        L.append(f"- pinned vs plain：串行 {_f(pg.get('serial'), '.3f')}x，"
                 f"流水线 {_f(pg.get('pipeline'), '.3f')}x（同口径 FPS 之比）")
        pool = t.get("pool")
        if pool:
            L.append(f"- pinned 池：pool_size {pool['pool_size']}，"
                     f"forwards {pool['forwards']}，acquire 等待 {pool['pool_waits']} 次，"
                     f"常驻显存 {pool['device_mib']} MiB / pinned 主机 {pool['pinned_host_mib']} MiB")
        q = t["modes"]["pinned_pipeline"].get("queues") or []
        if q:
            L.append("- 队列背压（pinned 流水线）：" + "；".join(
                f"{x['name']} max_depth {x['max_depth']}/overflow {x['overflow_waits']}"
                f"/underflow {x['underflow_waits']}" for x in q))
        _sp = t['speedup'].get('pinned') or t['speedup'].get(present[0]) or {}
        L.append(f"- 瓶颈阶段：{_sp.get('bottleneck_stage') or '—'}")
        chk = t.get("check")
        if chk:
            L.append(f"- 对拍 ultralytics（{chk['images']} 张）：检出 "
                     f"{chk['n_ours']} vs {chk['n_theirs']}，配上 {chk['n_matched']}；"
                     f"match_rate 均值 {_f(chk['match_rate_mean'], '.4f')}"
                     f"（最低 {_f(chk['match_rate_min'], '.4f')}），"
                     f"框 IoU {_f(chk['mean_box_iou'], '.4f')}，"
                     f"掩码 IoU {_f(chk['mean_mask_iou'], '.4f')}"
                     f"（腐蚀 1 px 后 {_f(chk['mean_mask_iou_erode1'], '.4f')}），"
                     f"|分数差| {_f(chk['mean_abs_score_diff'], '.5f')}，"
                     f"未全配的图 {chk['imgs_not_full_match']} 张")
            L.append(f"- 掩码像素总量 ours/theirs = {chk['px_ours']}/{chk['px_theirs']}"
                     f"（比值 {_f(chk['px_ratio'], '.4f')}）—— 偏离 1 说明是系统性口径差，"
                     f"不是边界噪声")
        g = t["gate"]
        L.append(f"- 判据：{'全部通过' if g['ok'] else '未通过'}（过 {len(g['passed'])} 条）")
        for f_ in g["failed"]:
            L.append(f"  - FAIL {f_['gate']}：{json.dumps(f_, ensure_ascii=False)}")
        L.append("")
    L.append(f"**总结论：{rep['verdict']}**")
    L.append("")
    return L


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="W4 三阶段流水线压测（解码/推理/后处理）与后处理对拍",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--weights", default="runs/segment/baseline15/weights/best.pt",
                    help=".pt 权重，只用来推导 run 名")
    ap.add_argument("--data", default="out/yolo_mvtec15/data.yaml")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--name", default=None, help="run 名，默认从权重路径反推")
    ap.add_argument("--engine-dir", default="weights")
    ap.add_argument("--tiers", nargs="+", default=["fp32", "fp16", "int8"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0", help="CUDA 设备号")
    ap.add_argument("--frames", type=int, default=30, help="清单里取前几张")
    ap.add_argument("--repeats", type=int, default=10, help="这批图重复几轮")
    ap.add_argument("--preload", action="store_true",
                    help="先把图解码进内存，隔离磁盘 I/O（量 CPU 前后处理与拷贝）")
    ap.add_argument("--capacity", type=int, default=4, help="阶段间环形队列容量")
    ap.add_argument("--pool-size", type=int, default=2, help="pinned 输出缓冲池大小")
    ap.add_argument("--join-timeout", type=float, default=60.0)
    ap.add_argument("--conf-thr", type=float, default=0.25)
    ap.add_argument("--iou-thr", type=float, default=0.7)
    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--mask-thr", type=float, default=0.5)
    ap.add_argument("--no-masks", action="store_true", help="跳过掩码解码（看 stage3 占比）")
    ap.add_argument("--modes", nargs="+", default=["plain", "pinned"],
                    choices=["plain", "pinned"])
    ap.add_argument("--check-ultralytics", action="store_true",
                    help="跑一遍后处理对拍（每档 +约 30 张图的 predict 时间）")
    ap.add_argument("--sample-list", default="configs/bench_sample.json")
    ap.add_argument("--sample-n", type=int, default=vox.DEFAULT_SAMPLE_N)
    ap.add_argument("--no-sample-list", action="store_true")
    ap.add_argument("--strict-sample", action="store_true")
    ap.add_argument("--min-speedup", type=float, default=DEFAULT_MIN_SPEEDUP)
    ap.add_argument("--min-match-rate", type=float, default=DEFAULT_MIN_MATCH_RATE)
    ap.add_argument("--min-box-iou", type=float, default=DEFAULT_MIN_BOX_IOU)
    ap.add_argument("--min-mask-iou", type=float, default=DEFAULT_MIN_MASK_IOU)
    ap.add_argument("--sample-resources", action="store_true",
                    help="采样 CPU 核数当量与 GPU 利用率/显存（默认关：nvidia-smi 子进程本身会扰动被测对象）")
    ap.add_argument("--sample-interval", type=float, default=0.2,
                    help="资源采样间隔秒，仅在 --sample-resources 时生效")
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--markdown", action="store_true", help="同时落一份 .md")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    w = Path(args.weights)
    d = Path(args.data)
    for p, what in ((d, "data.yaml"),):
        if not p.exists():
            print(f"[err  ] {what}不存在：{p}")
            return 2
    d = d if d.is_absolute() else (REPO_ROOT / d).resolve()
    edir = Path(args.engine_dir)
    edir = edir if edir.is_absolute() else (REPO_ROOT / edir).resolve()
    run = args.name or (w.parents[1].name if w.parent.name == "weights" else w.stem)
    engines = {t: edir / f"{run}_trt_{t}.engine" for t in args.tiers}
    missing = [str(v) for v in engines.values() if not v.is_file()]
    if missing:
        print("[err  ] 缺 engine（先跑 tools/build_trt.py）：")
        for m in missing:
            print(f"        {m}")
        return 2
    sample_list = None
    if args.sample_list and not args.no_sample_list:
        sample_list = Path(args.sample_list)
        if not sample_list.is_absolute():
            sample_list = (REPO_ROOT / sample_list).resolve()
    json_out = Path(args.json_out) if args.json_out else \
        (REPO_ROOT / "runs" / "logs" / f"pipeline_{run}.json")
    if not json_out.is_absolute():
        json_out = (REPO_ROOT / json_out).resolve()

    th = {"conf_thr": args.conf_thr, "iou_thr": args.iou_thr,
          "max_det": args.max_det, "mask_thr": args.mask_thr}
    th_gate = {"min_speedup": args.min_speedup, "min_match_rate": args.min_match_rate,
               "min_box_iou": args.min_box_iou, "min_mask_iou": args.min_mask_iou}
    n_items = args.frames * args.repeats
    cfg = {
        "run": run, "data": str(d), "split": args.split, "imgsz": args.imgsz,
        "engine_dir": str(edir), "tiers": list(args.tiers),
        "engines": {k: str(v) for k, v in engines.items()},
        "device": args.device, "frames": args.frames, "repeats": args.repeats,
        "n_items": n_items, "preload": bool(args.preload), "capacity": args.capacity,
        "pool_size": args.pool_size, "join_timeout": args.join_timeout,
        "need_masks": not args.no_masks, "modes": list(args.modes),
        "check_ultralytics": bool(args.check_ultralytics),
        "sample_resources": bool(args.sample_resources),
        "sample_interval_s": args.sample_interval,
        "sample_list": str(sample_list) if sample_list else None,
        "weights": str(w), "json_out": str(json_out), **th, **th_gate,
    }
    print("[cfg  ] " + json.dumps(cfg, ensure_ascii=False))
    if args.dry_run:
        for t, p in engines.items():
            print(f"[dry  ] {t}: {p}（{round(p.stat().st_size / 1024**2, 2)} MiB）")
        print(f"[dry  ] 每档 {len(args.modes)} 种缓冲 x 2 种模式 x {n_items} 帧")
        print(f"[dry  ] 资源采样={'开 @ ' + str(args.sample_interval) + ' s' if args.sample_resources else '关'}")
        return 0

    try:
        import torch  # noqa: PLC0415
    except ImportError as e:
        print(f"[err  ] {vox.__name__} 之外还缺 torch：{e}")
        return 2
    if not torch.cuda.is_available():
        print("[err  ] torch.cuda 不可用；TensorRT engine 只能在 GPU 上跑")
        return 2
    try:
        import tensorrt as trt  # noqa: PLC0415
    except ImportError as e:
        print(f"[err  ] 没有 tensorrt：{e}（见 docs/environment.md §6）")
        return 2

    images = read_split_images(d, args.split)
    if not images:
        print(f"[err  ] {d} 的 {args.split} split 里没有图片")
        return 2
    try:
        canonical, sample_meta = vox.resolve_sample(
            images, root=d.parent, sample_list=sample_list, sample_n=args.sample_n,
            use_list=not args.no_sample_list, split=args.split, strict=args.strict_sample)
    except (FileNotFoundError, ValueError) as e:
        print(f"[err  ] {e}")
        return 2
    for wn in sample_meta["warnings"]:
        print(f"[warn ] {wn}")
    if len(canonical) < args.frames:
        print(f"[err  ] 只有 {len(canonical)} 张，撑不起 --frames {args.frames}")
        return 2
    picked = canonical[:args.frames]
    print(f"[data ] {args.split} 共 {len(images)} 图；口径={sample_meta['source']} "
          f"{sample_meta['count']} 张（sha256:{sample_meta['sha256']}），取前 {len(picked)} 张")

    t_start = time.time()
    frames = build_frames(picked, args.repeats, args.imgsz, args.preload)
    print(f"[prep ] 构建 {len(frames)} 个输入项（preload={args.preload}）")

    env = {"python": sys.version.split()[0], "torch": torch.__version__,
           "tensorrt": trt.__version__,
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
           "cpus": None}
    try:
        import os
        env["cpus"] = os.cpu_count()
    except Exception:  # noqa: BLE001
        pass

    tiers: Dict[str, Dict[str, Any]] = {}
    for prec in args.tiers:
        ep = engines[prec]
        print(f"[tier ] {prec} 开始（{ep.name}，{round(ep.stat().st_size / 1024**2, 2)} MiB）")
        runner = TrtRunner(ep, device=f"cuda:{args.device.lstrip('cuda:')}")
        client = None
        if "pinned" in args.modes:
            from induvision.pinned import PinnedTrtClient  # noqa: PLC0415
            client = PinnedTrtClient(runner=runner, device=f"cuda:{args.device.lstrip('cuda:')}",
                                     pool_size=args.pool_size, imgsz=args.imgsz)
        modes: Dict[str, Dict[str, Any]] = {}
        consistency: Dict[str, Dict[str, Any]] = {}
        speedups: Dict[str, Dict[str, Any]] = {}
        for mode in args.modes:
            if mode == "plain":
                stages = make_plain_stages(runner, args.imgsz, th, not args.no_masks, False)
            else:
                stages = make_pinned_stages(client, args.imgsz, th, not args.no_masks, False)
            for kind in ("serial", "pipeline"):
                key = f"{mode}_{kind}"
                gc.collect()
                torch.cuda.empty_cache()
                t0 = time.perf_counter()
                sampler = ResourceSampler(interval_s=args.sample_interval) \
                    if args.sample_resources else None
                if sampler is not None:
                    sampler.start()
                try:
                    if kind == "serial":
                        results, rep = run_serial(frames, stages)
                    else:
                        results, rep = Pipeline(stages, capacity=args.capacity,
                                                join_timeout=args.join_timeout).run(frames)
                finally:
                    if sampler is not None:
                        rep_res = sampler.stop()
                rep["elapsed_incl_overhead_s"] = round(time.perf_counter() - t0, 4)
                if sampler is not None:
                    rep["resources"] = rep_res
                rep["checksum"] = checksum(results)
                rep["order_ok"] = order_ok(frames, results)
                modes[key] = rep
                e2e = rep.get("e2e_latency") or {}
                print(f"[run  ] {key}: wall {rep['wall_s']} s, fps {rep['throughput_fps']}, "
                      f"dets {rep['checksum']['dets']}, 保序 {rep['order_ok']}, "
                      f"e2e P50 {_f(e2e.get('p50_ms'))} / P95 {_f(e2e.get('p95_ms'))} ms")
                if rep.get("resources"):
                    rz = rep["resources"]
                    print(f"[res  ] {key}: CPU 均值 {_f(rz.get('cpu_cores_mean'))} 核 / "
                          f"峰值 {_f(rz.get('cpu_cores_max'))} 核，"
                          f"GPU util 均值 {_f(rz.get('gpu_util_mean_pct'), '.1f')}% / "
                          f"峰值 {_f(rz.get('gpu_util_max_pct'), '.1f')}%，"
                          f"显存 {_f(rz.get('gpu_mem_used_max_mib'), '.0f')} MiB"
                          f"（采样 {rz.get('gpu_samples')} 点 @ {rz.get('interval_s')} s）")
                if kind == "serial":
                    serial_ref = results
                    serial_rep = rep
                else:
                    consistency[mode] = {
                        "same_checksum": serial_rep["checksum"] == rep["checksum"],
                        **frame_diffs(serial_ref, results),
                    }
                    speedups[mode] = speedup_report(serial_rep, rep)
            if mode == "pinned" and client is not None:
                print(f"[pool ] {json.dumps(client.stats(), ensure_ascii=False)}")
        if "pinned" not in args.modes:
            speedups.setdefault("pinned", speedups.get(args.modes[-1], {}))

        check = None
        if args.check_ultralytics:
            stages_chk = (make_pinned_stages(client, args.imgsz, th, not args.no_masks, True)
                          if client is not None else
                          make_plain_stages(runner, args.imgsz, th, not args.no_masks, True))
            one = build_frames(picked, 1, args.imgsz, args.preload)
            res_chk, _ = run_serial(one, stages_chk)
            check = check_against_ultralytics(w, ep, picked, args.imgsz, th, args.device,
                                              [r.dets or [] for r in res_chk])
            print(f"[check] 对拍 {check['images']} 张：ours {check['n_ours']} / "
                  f"theirs {check['n_theirs']} / matched {check['n_matched']}，"
                  f"match_rate {check['match_rate_mean']}，框 IoU {check['mean_box_iou']}，"
                  f"掩码 IoU {check['mean_mask_iou']}")

        tier: Dict[str, Any] = {
            "engine": str(ep), "engine_mib": round(ep.stat().st_size / 1024**2, 2),
            "modes": modes, "consistency": consistency, "speedup": speedups,
            "pool": client.stats() if client is not None else None,
            "check": check,
        }
        tier["gate"] = gate_tier(tier, th_gate)
        tier["verdict"] = "pass" if tier["gate"]["ok"] else "fail"
        print(f"[gate ] {prec}: {tier['verdict']}"
              + ("" if tier["gate"]["ok"] else
                 " -> " + "; ".join(f["gate"] for f in tier["gate"]["failed"])))
        tiers[prec] = tier
        if client is not None:
            client.close()
        runner.close()
        del runner, client
        gc.collect()
        torch.cuda.empty_cache()

    report = {
        "schema": "induvision.pipeline_bench.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "run": run, "git_head": vox._git_head(), "env": env, "cfg": cfg,
        "sample": {k: sample_meta[k] for k in ("source", "count", "sha256", "warnings")},
        "tiers": tiers,
        "verdict": "pass" if all(t["gate"]["ok"] for t in tiers.values()) else "fail",
        "elapsed_s": round(time.time() - t_start, 2),
    }
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[out  ] JSON -> {json_out}")
    if args.markdown:
        md = json_out.with_suffix(".md")
        md.write_text("\n".join(_md(report, cfg)), encoding="utf-8")
        print(f"[out  ] Markdown -> {md}")
    print(f"[done ] 总结论 {report['verdict']}，用时 {report['elapsed_s']} s")
    return 0 if report["verdict"] == "pass" else 3


if __name__ == "__main__":
    sys.exit(main())
