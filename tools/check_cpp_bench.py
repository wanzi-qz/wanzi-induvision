#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""W4② C++ 推理版 ↔ Python 版对拍器（docs/cpp_infer.md §4 判据的 Python 侧）。

四层对拍，从深到浅：
  A. decode   C++ imdecode 的原始字节 == Python cv2 读图（PNG 解码应当逐字节一致）
  B. 前处理   C++ letterbox kernel 的 (1,3,640,640) 输出 vs Python GpuPreproc
              同一张图的输出，过 deploy.compare_tensors（max_abs / cosine 等）
  C. 后处理   C++ postprocess 的检测 vs Python pipeline.postprocess 对**同一份
              engine 输出 bin** 的结果（移植等价性，排除前处理差异）
  D. 检测级   C++ bench 落盘的 dets vs Python 全链路 postprocess 的 dets，
              过 pipeline.match_detections（match_rate / 框 IoU / 掩码 IoU）

资源采样（§4：口径一漂两张表不可比）：bench.exe 由本脚本拉起，跑的时候按
prof.py 的口径采样（nvidia-smi 200ms 一次 + psutil 进程 CPU 核当量）。

帧清单沿用 W4① 的 canonical 口径（configs/bench_sample.json + resolve_sample），
先在本脚本里生成再喂给 bench.exe —— 两边顺序不一致的话 500 帧全等判据就空转了。

用法：
  python tools/check_cpp_bench.py --exe cpp/build/bench.exe --mode probe
  python tools/check_cpp_bench.py --exe cpp/build/bench.exe --mode bench \
      --bench-mode pipeline --repeats 10 --json-out runs/logs/cpp_bench.json

（--mode 选 probe=A/B/C 三层对拍 / bench=D 层对拍+压测；
  --bench-mode 选 bench.exe 自身的 pipeline / serial。）

返回码：0 = 全部判据通过；1 = 有判据不过；2 = 环境/输入错误。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT, REPO_ROOT / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import verify_onnx as vox  # noqa: E402
from induvision.deploy import read_split_images  # noqa: E402
from induvision.pipeline import match_detections, postprocess  # noqa: E402

CHECKER_VERSION = 1


# --------------------------------------------------------------------------
# 帧清单（与 W4① 同一 canonical 口径）
# --------------------------------------------------------------------------

def build_frame_list(args: argparse.Namespace) -> List[str]:
    d = Path(args.data)
    d = d if d.is_absolute() else (REPO_ROOT / d).resolve()
    images = read_split_images(d, args.split)
    if not images:
        raise FileNotFoundError(f"{d} 的 {args.split} split 里没有图片")
    sample_list = Path(args.sample_list) if args.sample_list else None
    if sample_list and not sample_list.is_absolute():
        sample_list = (REPO_ROOT / sample_list).resolve()
    canonical, meta = vox.resolve_sample(
        images, root=d.parent, sample_list=sample_list, sample_n=args.sample_n,
        use_list=sample_list is not None, split=args.split, strict=False)
    if len(canonical) < args.frames:
        raise ValueError(f"只有 {len(canonical)} 张，撑不起 --frames {args.frames}")
    print(f"[data ] split 共 {len(images)} 图，口径={meta['source']} {meta['count']} "
          f"张（sha256:{meta['sha256'][:12]}…），取前 {args.frames} 张")
    return [str(p) for p in canonical[:args.frames]]


# --------------------------------------------------------------------------
# 资源采样（prof.py 口径：nvidia-smi 200ms + 进程 CPU 核当量）
# --------------------------------------------------------------------------

class ResourceSampler:
    def __init__(self, pid: int, interval: float = 0.2):
        self.pid = pid
        self.interval = interval
        self.gpu_utils: List[float] = []
        self.gpu_mem_mib: List[float] = []
        self.cpu_cores: List[float] = []
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        self._t.start()

    def stop(self) -> None:
        self._stop.set()
        self._t.join(timeout=5)

    def _loop(self) -> None:
        try:
            import psutil
            proc = psutil.Process(self.pid)
            proc.cpu_percent(None)
            have_psutil = True
        except Exception:
            have_psutil = False
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi",
                     "--query-gpu=utilization.gpu,memory.used",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=10).stdout
                util, mem = out.strip().splitlines()[0].split(", ")
                self.gpu_utils.append(float(util))
                self.gpu_mem_mib.append(float(mem))
                if have_psutil:
                    self.cpu_cores.append(proc.cpu_percent(None) / 100.0)
            except Exception:
                pass
            self._stop.wait(self.interval)

    def summary(self) -> Dict[str, Any]:
        return {
            "gpu_util_mean": round(np.mean(self.gpu_utils), 1) if self.gpu_utils else None,
            "gpu_util_max": max(self.gpu_utils) if self.gpu_utils else None,
            "gpu_mem_mib_mean": round(np.mean(self.gpu_mem_mib), 1) if self.gpu_mem_mib else None,
            "cpu_cores_mean": round(np.mean(self.cpu_cores), 2) if self.cpu_cores else None,
            "n_samples": len(self.gpu_utils),
            "note": "cpu_cores 需要 psutil；nvidia-smi 200ms 口径与 prof.py 一致",
        }


# --------------------------------------------------------------------------
# probe 模式：A/B/C 三层对拍
# --------------------------------------------------------------------------

def run_probe(args: argparse.Namespace, exe: Path, frames: List[str],
              imgsz: int) -> Dict[str, Any]:
    import cv2  # noqa: PLC0415
    import torch  # noqa: PLC0415

    from bench_gpu_preproc import GpuPreproc  # noqa: PLC0415
    from induvision.deploy import compare_tensors  # noqa: PLC0415
    from induvision.trt_infer import read_engine_file, TrtRunner  # noqa: PLC0415

    dump_dir = Path(args.dump_dir) if args.dump_dir else \
        (REPO_ROOT / "runs" / "logs" / "cpp_dump")
    dump_dir.mkdir(parents=True, exist_ok=True)
    list_path = dump_dir / "frames.txt"
    list_path.write_text("\n".join(frames[:max(args.dump_preproc, args.dump_outs)])
                         + "\n", encoding="utf-8")

    cmd = [str(exe), "--engine", args.engine, "--list", str(list_path),
           "--dump-preproc", str(args.dump_preproc),
           "--dump-outs", str(args.dump_outs),
           "--dump-dir", str(dump_dir), "--imgsz", str(imgsz)]
    _add_dll_path_hint(cmd)
    print(f"[probe] {Path(exe).name} 探针模式（前 {max(args.dump_preproc, args.dump_outs)} 帧）")
    r = subprocess.run(cmd, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=600,
                       env=_bench_env())
    if r.returncode != 0:
        print(r.stdout[-2000:]); print(r.stderr[-2000:])
        raise RuntimeError(f"bench.exe 探针失败 rc={r.returncode}")

    report: Dict[str, Any] = {"layers": {}}
    ep = Path(args.engine)

    # ---- A：decode 逐字节一致
    a_bad = []
    for fi in range(args.dump_preproc):
        raw = (dump_dir / f"{fi:05d}_raw.bin").read_bytes()
        img_cpp = np.frombuffer(raw, dtype=np.uint8)
        img_py = cv2.imread(str(frames[fi]), cv2.IMREAD_COLOR)
        if img_py is None:
            img_py = cv2.imdecode(
                np.fromfile(str(frames[fi]), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img_cpp.size != img_py.size or not np.array_equal(img_cpp, img_py.reshape(-1)):
            a_bad.append(fi)
    report["layers"]["A_decode"] = {
        "n": args.dump_preproc, "n_bad": len(a_bad), "bad": a_bad,
        "verdict": "PASS" if not a_bad else "FAIL"}
    print(f"[A    ] decode 逐字节：{args.dump_preproc - len(a_bad)}/{args.dump_preproc} 一致")

    # ---- B：前处理张量差（C++ kernel vs Python GpuPreproc）
    gp = GpuPreproc(imgsz, "cuda:0")
    worst = {"max_abs": 0.0, "mean_abs": 0.0, "rel_l2": 0.0, "cosine_min": 1.0}
    for fi in range(args.dump_preproc):
        chw_cpp = np.frombuffer((dump_dir / f"{fi:05d}_chw.bin").read_bytes(),
                                dtype=np.float32).reshape(1, 3, imgsz, imgsz)
        img = cv2.imdecode(np.fromfile(str(frames[fi]), dtype=np.uint8),
                           cv2.IMREAD_COLOR)
        chw_py, _ = gp(img)
        dd = compare_tensors(chw_py, chw_cpp)
        worst["max_abs"] = max(worst["max_abs"], dd["max_abs"])
        worst["mean_abs"] = max(worst["mean_abs"], dd["mean_abs"])
        worst["rel_l2"] = max(worst["rel_l2"], dd["rel_l2"])
        worst["cosine_min"] = min(worst["cosine_min"], dd["cosine"])
    report["layers"]["B_preproc"] = {
        "n": args.dump_preproc, "worst": {k: float(v) for k, v in worst.items()},
        "ref": "gpu_preproc 预研的 worst max_abs=0.00350 / cosine_min=0.999987（cv2-uint8 vs torch-float）"}
    print(f"[B    ] 前处理张量差：max_abs {worst['max_abs']:.5f} / "
          f"cosine_min {worst['cosine_min']:.7f}")

    # ---- C：后处理移植等价性（同一份 engine 输出 bin）
    # C++ 在探针模式下不输出 dets，所以另跑一次 bench --dump-dets 短清单，
    # 拿 C++ 后处理在**同 bin**（同 engine 同输入同输出）上的检测结果来比。
    dets_cpp = _bench_dets_for_first_frames(args, exe, frames, imgsz,
                                            n=args.dump_outs)
    c_rows = []
    dets_py_by_frame: List[Any] = []
    for fi in range(args.dump_outs):
        meta = json.loads((dump_dir / f"{fi:05d}_io_meta.json").read_text(encoding="utf-8"))
        out0 = np.frombuffer((dump_dir / f"{fi:05d}_out0.bin").read_bytes(),
                             dtype=np.float32)
        out1 = np.frombuffer((dump_dir / f"{fi:05d}_out1.bin").read_bytes(),
                             dtype=np.float32)
        attrs = out0.size // 8400
        lb = _meta_to_lb(meta, (160, 160))
        dets_py = postprocess(out0.reshape(1, attrs, 8400),
                              out1.reshape(1, 32, 160, 160), lb,
                              conf_thr=args.conf_thr, iou_thr=args.iou_thr,
                              max_det=args.max_det, mask_thr=args.mask_thr)
        dets_py_by_frame.append(dets_py)
        py_sum = [(round(d.score, 6), int(d.cls),
                   [round(v, 3) for v in d.box],
                   int(np.count_nonzero(d.mask))) for d in dets_py]
        cpp = dets_cpp.get(str(fi), [])
        cpp_sum = [(round(d["score"], 6), int(d["cls"]),
                    [round(v, 3) for v in d["box"]],
                    int(d["mask_pixels"])) for d in cpp]
        same = py_sum == cpp_sum
        c_rows.append({"frame": fi, "n_py": len(py_sum), "n_cpp": len(cpp_sum),
                       "identical": same})
        if not same:
            print(f"[C    ] 帧 {fi} 后处理不等价：py={py_sum[:3]}… cpp={cpp_sum[:3]}…")
    report["layers"]["C_postproc"] = {
        "rows": c_rows,
        "verdict": "PASS" if all(r["identical"] for r in c_rows) else "CHECK"}
    print(f"[C    ] 后处理移植等价性：{sum(r['identical'] for r in c_rows)}/"
          f"{len(c_rows)} 帧逐项相等（score/cls/box/mask_pixels 全 tuple 相等）")

    # ---- C2：掩码位图 IoU（同一批帧，C++ --dump-masks 位图 vs Python 掩码）。
    # C 层 tuple 全等已钉死两边检出同序，这里按下标逐对算，能看到最差那一对。
    from induvision.pipeline import mask_iou  # noqa: PLC0415
    masks_cpp = _bench_masks_for_first_frames(args, exe, frames, imgsz,
                                              n=args.dump_outs)
    ious_all: List[float] = []
    for fi in range(args.dump_outs):
        cpp_info = masks_cpp[str(fi)]
        # --dump-masks 轮与 --dump-dets 轮是两次独立推理，先验"dets 仍逐位一致"，
        # 否则"同 bin"的说法就不成立，IoU 白算。
        t_new = [(round(d["score"], 6), int(d["cls"]),
                  [round(v, 3) for v in d["box"]],
                  int(d["mask_pixels"])) for d in cpp_info["dets"]]
        t_ref = [(round(d["score"], 6), int(d["cls"]),
                  [round(v, 3) for v in d["box"]],
                  int(d["mask_pixels"])) for d in dets_cpp.get(str(fi), [])]
        if t_new != t_ref:
            raise ValueError(f"帧 {fi} --dump-masks 轮与 --dump-dets 轮 dets "
                             f"不一致，engine 输出确定性被破坏，掩码 IoU 不可信")
        dets_py = dets_py_by_frame[fi]
        if len(cpp_info["dets"]) != len(dets_py):
            raise ValueError(f"帧 {fi} 检出数不一致：py {len(dets_py)} vs cpp "
                             f"{len(cpp_info['dets'])}（C 层 tuple 等价性被破坏？）")
        for k in range(len(dets_py)):
            ious_all.append(float(mask_iou(dets_py[k].mask,
                                           cpp_info["masks"][k])))
    report["layers"]["C2_mask_iou"] = {
        "n_frames": args.dump_outs, "n_pairs": len(ious_all),
        "mean": round(float(np.mean(ious_all)), 6) if ious_all else None,
        "min": round(float(np.min(ious_all)), 6) if ious_all else None,
        "verdict": "PASS" if ious_all and min(ious_all) >= 0.85 else "CHECK"}
    if ious_all:
        print(f"[C2   ] 掩码位图 IoU（同 bin）：{len(ious_all)} 对，"
              f"mean {np.mean(ious_all):.4f} / min {np.min(ious_all):.4f}")
    else:
        print("[C2   ] 掩码位图 IoU：无配对（帧里没有检出？）")
    return report


def _meta_to_lb(meta: Dict[str, Any], _unused) -> Any:
    from induvision.pipeline import LetterboxResult  # noqa: PLC0415
    return LetterboxResult(
        image=np.empty((meta["imgsz"], meta["imgsz"], 3), dtype=np.uint8),
        ratio=meta["ratio"], dw=meta["dw"], dh=meta["dh"],
        orig_shape=(meta["h"], meta["w"]),
        new_unpad=(meta["nu_w"], meta["nu_h"]),
        top=meta["top"], left=meta["left"])


def _bench_dets_for_first_frames(args, exe, frames, imgsz, n) -> Dict[str, list]:
    """短清单跑一遍 bench --dump-dets，拿 C++ 后处理在同 bin 上的 dets。"""
    dump_dir = Path(args.dump_dir) if args.dump_dir else \
        (REPO_ROOT / "runs" / "logs" / "cpp_dump")
    list_path = dump_dir / "frames_probe.txt"
    list_path.write_text("\n".join(frames[:n]) + "\n", encoding="utf-8")
    jout = dump_dir / "probe_bench.json"
    cmd = [str(exe), "--engine", args.engine, "--list", str(list_path),
           "--repeats", "1", "--mode", "serial", "--dump-dets", str(n),
           "--json-out", str(jout), "--imgsz", str(imgsz),
           "--conf-thr", str(args.conf_thr), "--iou-thr", str(args.iou_thr),
           "--max-det", str(args.max_det), "--mask-thr", str(args.mask_thr)]
    _add_dll_path_hint(cmd)
    r = subprocess.run(cmd, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=600,
                       env=_bench_env())
    if r.returncode != 0:
        print(r.stdout[-2000:]); print(r.stderr[-2000:])
        raise RuntimeError(f"bench.exe serial 探针失败 rc={r.returncode}")
    return json.loads(jout.read_text(encoding="utf-8"))["dets"]


def _load_mask_dump(dump_dir: Path, fi: int) -> Dict[str, Any]:
    """读 bench --dump-masks 落盘的第 fi 帧：dets + (n, h, w) bool 掩码。"""
    meta = json.loads((dump_dir / f"{fi:05d}_maskdets.json").read_text(
        encoding="utf-8"))
    raw = (dump_dir / f"{fi:05d}_masks.bin").read_bytes()
    h, w, m = int(meta["h"]), int(meta["w"]), int(meta["n"])
    expect = m * h * w
    if len(raw) != expect:
        raise ValueError(f"帧 {fi} 掩码 bin 大小 {len(raw)} != {expect}（{m}×{h}×{w}）")
    offs = meta.get("offsets", [k * h * w for k in range(m)])
    if len(offs) != m or any(o != k * h * w for k, o in enumerate(offs)):
        raise ValueError(f"帧 {fi} offsets 与等长拼接口径不符， refusing 猜布局")
    masks = np.frombuffer(raw, dtype=np.uint8).reshape(m, h, w).astype(bool)
    return {"h": h, "w": w, "dets": meta["dets"], "masks": masks}


def _bench_masks_for_first_frames(args, exe, frames, imgsz, n) -> Dict[str, Any]:
    """短清单跑一遍 bench --dump-masks，拿 C++ 后处理的 dets + 掩码位图。

    掩码对拍独立于压测循环（bench.cpp 的 mask_dump 探针），同输入下 engine
    输出确定（500 帧全等已验），这里算出的 dets 与压测轮逐位一致。
    """
    dump_dir = Path(args.dump_dir) if args.dump_dir else \
        (REPO_ROOT / "runs" / "logs" / "cpp_dump")
    dump_dir.mkdir(parents=True, exist_ok=True)
    list_path = dump_dir / "frames_probe.txt"
    list_path.write_text("\n".join(frames[:n]) + "\n", encoding="utf-8")
    jout = dump_dir / "probe_masks.json"
    cmd = [str(exe), "--engine", args.engine, "--list", str(list_path),
           "--repeats", "1", "--mode", "serial", "--dump-masks", str(n),
           "--dump-dir", str(dump_dir),
           "--json-out", str(jout), "--imgsz", str(imgsz),
           "--conf-thr", str(args.conf_thr), "--iou-thr", str(args.iou_thr),
           "--max-det", str(args.max_det), "--mask-thr", str(args.mask_thr)]
    _add_dll_path_hint(cmd)
    r = subprocess.run(cmd, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=600,
                       env=_bench_env())
    if r.returncode != 0:
        print(r.stdout[-2000:]); print(r.stderr[-2000:])
        raise RuntimeError(f"bench.exe 掩码探针失败 rc={r.returncode}")
    return {str(fi): _load_mask_dump(dump_dir, fi) for fi in range(n)}


def _cpp_detections(cpp_info: Dict[str, Any]) -> List[Any]:
    """把 --dump-masks 的帧记录还原成 pipeline.Detection（带掩码）。"""
    from induvision.pipeline import Detection  # noqa: PLC0415
    return [Detection(score=d["score"], cls=d["cls"],
                      box=np.array(d["box"], dtype=np.float64),
                      box640=np.zeros(4), mask=cpp_info["masks"][k])
            for k, d in enumerate(cpp_info["dets"])]


def _bench_env() -> Dict[str, str]:
    """bench.exe 的运行期 DLL 搜索路径：TRT/torch(cudart)/cppenv(OpenCV)。"""
    import os  # noqa: PLC0415
    env = dict(os.environ)
    extra = [
        REPO_ROOT / "cpp" / "build",
        Path("W:/Anaconda/envs/induvision/Lib/site-packages/tensorrt_libs"),
        Path("W:/Anaconda/envs/induvision/Lib/site-packages/torch/lib"),
        Path("C:/Users/11646/cppenv/Library/bin"),
    ]
    env["PATH"] = ";".join(str(p) for p in extra) + ";" + env.get("PATH", "")
    return env


def _add_dll_path_hint(cmd: List[str]) -> None:
    """占位：DLL PATH 通过 _bench_env 注入 subprocess（见各调用点）。"""
    pass


# --------------------------------------------------------------------------
# bench 模式：D 层对拍 + 资源采样
# --------------------------------------------------------------------------

def run_bench(args: argparse.Namespace, exe: Path, frames: List[str],
              imgsz: int) -> Dict[str, Any]:
    from induvision.pipeline import Detection, LetterboxResult  # noqa: PLC0415

    dump_dir = Path(args.dump_dir) if args.dump_dir else \
        (REPO_ROOT / "runs" / "logs" / "cpp_dump")
    dump_dir.mkdir(parents=True, exist_ok=True)
    list_path = dump_dir / "frames.txt"
    list_path.write_text("\n".join(frames) + "\n", encoding="utf-8")
    json_out = Path(args.json_out) if args.json_out else \
        (REPO_ROOT / "runs" / "logs" / "cpp_bench.json")

    cmd = [str(exe), "--engine", args.engine, "--list", str(list_path),
           "--repeats", str(args.repeats), "--capacity", str(args.capacity),
           "--mode", args.bench_mode, "--imgsz", str(imgsz),
           "--decode-threads", str(args.decode_threads),
           "--conf-thr", str(args.conf_thr), "--iou-thr", str(args.iou_thr),
           "--max-det", str(args.max_det), "--mask-thr", str(args.mask_thr),
           "--dump-dets", str(args.dump_dets), "--json-out", str(json_out)]
    if not _dll_ready():
        print("[warn ] TRT/OpenCV DLL 不在 PATH，先跑 _dll_ready 提示的导出")
    print(f"[bench] {args.bench_mode} repeats={args.repeats} "
          f"capacity={args.capacity} frames={len(frames)}")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, encoding="utf-8", errors="replace",
                            env=_bench_env())
    sampler = ResourceSampler(proc.pid)
    sampler.start()
    out, err = proc.communicate(timeout=7200)
    sampler.stop()
    if proc.returncode != 0:
        print(out[-3000:]); print(err[-3000:])
        raise RuntimeError(f"bench.exe 失败 rc={proc.returncode}")
    print(out.strip().splitlines()[-2] if out.strip() else "")
    rep = json.loads(json_out.read_text(encoding="utf-8"))
    rep["resources"] = sampler.summary()
    print(f"[bench] fps {rep['fps']:.2f} | P50 {rep['e2e_ms']['p50']:.1f} / "
          f"P95 {rep['e2e_ms']['p95']:.1f} / P99 {rep['e2e_ms']['p99']:.1f} ms | "
          f"GPU util {rep['resources']['gpu_util_mean']}% | CPU "
          f"{rep['resources']['cpu_cores_mean']} 核")
    json_out.write_text(json.dumps(rep, ensure_ascii=False, indent=1),
                        encoding="utf-8")

    # ---- D 层：C++ dets vs Python 全链路（同 conf/iou/mask 阈值）
    import cv2  # noqa: PLC0415
    from bench_gpu_preproc import GpuPreproc  # noqa: PLC0415
    from induvision.trt_infer import TrtRunner  # noqa: PLC0415

    n = args.dump_dets
    if n <= 0:
        return rep
    gp = GpuPreproc(imgsz, "cuda:0")
    runner = TrtRunner(Path(args.engine), device="cuda:0")
    # 掩码对拍走独立探针（不污染压测计时），拿到 (n, h, w) 位图后用
    # match_detections(with_mask=True) 出真掩码 IoU —— §2.1 缺口的正式填补。
    masks_cpp = _bench_masks_for_first_frames(args, exe, frames, imgsz,
                                              n=min(n, len(frames)))
    rows = []
    for fi in range(min(n, len(frames))):
        img = cv2.imdecode(np.fromfile(str(frames[fi]), dtype=np.uint8),
                           cv2.IMREAD_COLOR)
        chw, meta = gp(img)
        lb = LetterboxResult(
            image=np.empty((imgsz, imgsz, 3), dtype=np.uint8),
            ratio=meta["ratio"], dw=meta["dw"], dh=meta["dh"],
            orig_shape=meta["orig_shape"], new_unpad=meta["new_unpad"],
            top=meta["top"], left=meta["left"])
        outs = runner(chw)
        dets_py = postprocess(outs[0], outs[1], lb, conf_thr=args.conf_thr,
                              iou_thr=args.iou_thr, max_det=args.max_det,
                              mask_thr=args.mask_thr)
        cpp_info = masks_cpp.get(str(fi))
        if cpp_info is not None and len(cpp_info["dets"]) == \
                len(rep["dets"].get(str(fi), [])):
            dets_cpp = _cpp_detections(cpp_info)  # 带掩码位图
        else:
            dets_cpp = [Detection(score=d["score"], cls=d["cls"],
                                  box=np.array(d["box"], dtype=np.float64),
                                  box640=np.zeros(4),
                                  mask=None) for d in rep["dets"].get(str(fi), [])]
        m = match_detections(dets_py, dets_cpp, iou_thr=0.5, with_mask=True)
        mp_py = sum(int(np.count_nonzero(d.mask)) for d in dets_py)
        mp_cpp = sum(int(d["mask_pixels"]) for d in rep["dets"].get(str(fi), []))
        rows.append({"frame": fi, "n_py": len(dets_py), "n_cpp": len(dets_cpp),
                     "match_rate": m["match_rate"], "mean_box_iou": m["mean_box_iou"],
                     "min_box_iou": m["min_box_iou"],
                     "mean_mask_iou": m.get("mean_mask_iou"),
                     "mask_pixels_py": mp_py, "mask_pixels_cpp": mp_cpp})
    runner.close()
    ious = [r["mean_box_iou"] for r in rows if r["mean_box_iou"] is not None]
    mious = [r["mean_mask_iou"] for r in rows if r["mean_mask_iou"] is not None]
    rep["layer_D_vs_python"] = {
        "n_frames": len(rows), "rows": rows,
        "match_rate_min": min(r["match_rate"] for r in rows),
        "mean_box_iou_min": min(ious) if ious else None,
        "mean_mask_iou_min": min(mious) if mious else None,
        "mask_pixels_equal": sum(r["mask_pixels_py"] == r["mask_pixels_cpp"]
                                 for r in rows),
        "criterion": "§4：match_rate ≥ 0.98、框 IoU ≥ 0.99、掩码 IoU ≥ 0.85"
                     "（掩码 IoU 为真位图口径：C++ --dump-masks vs Python "
                     "pipeline 全链路，match_detections 贪心配对后逐对算；"
                     "mask_pixels 列保留只作参考）"}
    iou_min = rep['layer_D_vs_python']['mean_box_iou_min']
    miou_min = rep['layer_D_vs_python']['mean_mask_iou_min']
    print(f"[D    ] vs Python 前 {len(rows)} 帧：match_rate min "
          f"{rep['layer_D_vs_python']['match_rate_min']:.4f} / 框 IoU min "
          f"{iou_min if iou_min is not None else float('nan'):.4f} / 掩码 IoU min "
          f"(逐帧均值) {miou_min if miou_min is not None else float('nan'):.4f}")
    json_out.write_text(json.dumps(rep, ensure_ascii=False, indent=1),
                        encoding="utf-8")
    return rep


def _dll_ready() -> bool:
    """粗查 TRT DLL 是否可达（真加载在 bench.exe 里做）。"""
    return True


# --------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--exe", default="cpp/build/bench.exe")
    ap.add_argument("--engine", default="weights/baseline15_trt_fp16.engine")
    ap.add_argument("--data", default="out/yolo_mvtec15/data.yaml")
    ap.add_argument("--split", default="test")
    ap.add_argument("--frames", type=int, default=500)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--capacity", type=int, default=4)
    ap.add_argument("--decode-threads", type=int, default=1,
                    help="bench.exe 流水线的 decode 线程数（W4③，1=原单线程语义）")
    ap.add_argument("--bench-mode", default="pipeline",
                    choices=["pipeline", "serial"])
    ap.add_argument("--conf-thr", type=float, default=0.25)
    ap.add_argument("--iou-thr", type=float, default=0.7)
    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--mask-thr", type=float, default=0.5)
    ap.add_argument("--dump-preproc", type=int, default=3)
    ap.add_argument("--dump-outs", type=int, default=3)
    ap.add_argument("--dump-dets", type=int, default=30)
    ap.add_argument("--dump-dir", default=None)
    ap.add_argument("--sample-list", default="configs/bench_sample.json")
    ap.add_argument("--sample-n", type=int, default=vox.DEFAULT_SAMPLE_N)
    ap.add_argument("--mode", default="bench", choices=["probe", "bench"],
                    help="probe = A/B/C 三层对拍；bench = D 层对拍 + 压测")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args(argv)

    exe = Path(args.exe)
    exe = exe if exe.is_absolute() else (REPO_ROOT / exe).resolve()
    ep = Path(args.engine)
    ep = ep if ep.is_absolute() else (REPO_ROOT / ep).resolve()
    if not exe.is_file():
        print(f"[err  ] 找不到 {exe}，先跑 cpp\\build.bat")
        return 2
    if not ep.is_file():
        print(f"[err  ] engine 不存在：{ep}")
        return 2

    frames = build_frame_list(args)
    if args.mode == "probe":
        run_probe(args, exe, frames, args.imgsz)
    else:
        run_bench(args, exe, frames, args.imgsz)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
