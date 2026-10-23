#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""W4② 里程碑 1 对拍：C++ infer_one 的输出 vs Python TrtRunner 的输出，逐元素比。

闭环设计（C++ 侧不依赖 OpenCV，Python 侧不出二进制细节）：

1. 本脚本取清单第一张图，走 ``deploy.preprocess`` 导出 ``(1,3,640,640)`` float32
   input.bin —— 这就是 C++ 侧的输入，两边吃同一份字节，**前处理差异不进入本轮**；
2. 调 C++ ``infer_one``（--out0/--out1 落盘），再用 ``TrtRunner`` 对同一输入跑一遍；
3. 两个输出各过一遍 ``deploy.compare_tensors``（max_abs / mean_abs / rel_l2 / cosine）。

判据（对齐 docs/cpp_infer.md §4 的精神，但注意口径不同）：
本轮比的是**同输入同 engine 的两套推理运行时**，预期接近逐位相等
（TRT 反序列化后是同一份 kernel 计划）。若 max_abs > 1e-3 或 cosine < 0.9999，
先怀疑 C++ 侧的 H2D/绑定/流同步，而不是精度。

用法（先 build，再跑）：
  python tools/check_cpp_infer.py --exe cpp/build/infer_one.exe \
      --engine weights/baseline15_trt_fp16.engine [--prec fp16]

返回码：0 = 对拍过；1 = 对拍不过；2 = 环境/输入错误。
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT, REPO_ROOT / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from induvision.deploy import compare_tensors, preprocess, read_image_bgr, read_split_images  # noqa: E402
from induvision.trt_infer import TrtRunner  # noqa: E402
import verify_onnx as vox  # noqa: E402

#: 里程碑 1 的判据。比的是"同一份字节进同一个 engine，两个运行时出的结果"，
#: 理论上限是逐位相等；给 1e-3 / 0.9999 的余量是留给出流/绑定的实现瑕疵，
#: 出现这个量级的差异就该查代码而不是解释精度。
MAX_ABS_LIMIT = 1e-3
COSINE_LIMIT = 0.9999


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="C++ infer_one vs Python TrtRunner 数值对拍")
    ap.add_argument("--exe", required=True, help="cpp/build/infer_one.exe")
    ap.add_argument("--engine", default="weights/baseline15_trt_fp16.engine")
    ap.add_argument("--data", default="out/yolo_mvtec15/data.yaml")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--sample-list", default="configs/bench_sample.json")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--tmpdir", default=None, help="中间 bin 的目录，默认临时目录")
    args = ap.parse_args(argv)

    exe = Path(args.exe)
    if not exe.is_file():
        print(f"[err  ] C++ 可执行文件不存在：{exe}（先 cmake --build）")
        return 2
    ep = Path(args.engine)
    ep = ep if ep.is_absolute() else (REPO_ROOT / ep).resolve()
    if not ep.is_file():
        print(f"[err  ] engine 不存在：{ep}")
        return 2
    d = Path(args.data)
    d = d if d.is_absolute() else (REPO_ROOT / d).resolve()

    images = read_split_images(d, args.split)
    if not images:
        print(f"[err  ] {d} 里没有图片")
        return 2
    sample_list = REPO_ROOT / args.sample_list
    canonical, _ = vox.resolve_sample(images, root=d.parent, sample_list=sample_list,
                                      sample_n=vox.DEFAULT_SAMPLE_N, use_list=True,
                                      split=args.split, strict=False)
    img = read_image_bgr(canonical[0])
    if img is None:
        print(f"[err  ] 读不到图：{canonical[0]}")
        return 2
    chw, lb = preprocess(img, 640)
    print(f"[data ] 第 1 张：{canonical[0].name}，chw {chw.shape} {chw.dtype}")

    tmp = Path(args.tmpdir) if args.tmpdir else Path(tempfile.mkdtemp(prefix="cpp_infer_"))
    tmp.mkdir(parents=True, exist_ok=True)
    in_bin = tmp / "input.bin"
    o0_bin = tmp / "out0.bin"
    o1_bin = tmp / "out1.bin"
    chw.astype(np.float32).tofile(in_bin)

    # ---- C++ 侧
    cmd = [str(exe), "--engine", str(ep), "--input", str(in_bin),
           "--out0", str(o0_bin), "--out1", str(o1_bin),
           "--iters", str(args.iters)]
    print("[cpp  ] " + " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    sys.stdout.write(r.stdout or "")
    sys.stderr.write(r.stderr or "")
    if r.returncode != 0:
        print(f"[err  ] infer_one 退出码 {r.returncode}")
        return 2

    # ---- Python 侧（同一份字节；TrtRunner 自己走 read_engine_file 剥头）
    runner = TrtRunner(ep)
    try:
        outs = runner(chw[None].copy())
        py0, py1 = np.asarray(outs[0]), np.asarray(outs[1])
    finally:
        runner.close()
    cpp0 = np.fromfile(o0_bin, dtype=np.float32).reshape(1, 51, 8400)
    cpp1 = np.fromfile(o1_bin, dtype=np.float32).reshape(1, 32, 160, 160)

    # ---- 比对
    ok = True
    for name, a, b in (("output0", py0, cpp0), ("output1", py1, cpp1)):
        t = compare_tensors(a, b)
        if not t.get("shape_match", False):
            print(f"[cmp  ] {name}: 形状不符 py{np.asarray(a).shape} vs cpp{np.asarray(b).shape}")
            ok = False
            continue
        bad = t["max_abs"] > MAX_ABS_LIMIT or t["cosine"] < COSINE_LIMIT
        ok = ok and not bad
        print(f"[cmp  ] {name}: max_abs {t['max_abs']:.6g} / mean_abs {t['mean_abs']:.6g} / "
              f"rel_l2 {t['rel_l2']:.6g} / cosine {t['cosine']:.8f} "
              f"-> {'FAIL' if bad else 'pass'}")
    print("[verdict]", "pass" if ok else "fail")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
