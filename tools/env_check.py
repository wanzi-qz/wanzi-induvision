"""GPU 环境自检脚本。

一条命令确认 torch / CUDA / ultralytics / onnx 是否都能真跑起来，
不只是"装上了"，而是"能算"。

用法：
    python tools/env_check.py                          # 默认：自检 + 单图 GPU 推理
    python tools/env_check.py --image path/to/a.jpg    # 指定测试图片
    python tools/env_check.py --export                 # 额外做一次 ONNX 导出验证
    python tools/env_check.py --matmul 4096            # 指定矩阵规模做算力粗测
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _ver(module: str) -> str:
    try:
        return str(getattr(__import__(module), "__version__"))
    except Exception as exc:
        return f"MISSING ({exc.__class__.__name__})"


def print_versions() -> dict:
    import torch

    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": str(torch.version.cuda),
        "cuda_available": torch.cuda.is_available(),
        "torchvision": _ver("torchvision"),
        "ultralytics": _ver("ultralytics"),
        "onnx": _ver("onnx"),
        "onnxruntime": _ver("onnxruntime"),
        "numpy": _ver("numpy"),
        "cv2": _ver("cv2"),
        "skimage": _ver("skimage"),
        "pandas": _ver("pandas"),
        "scipy": _ver("scipy"),
        "matplotlib": _ver("matplotlib"),
    }
    width = max(len(k) for k in info)
    print("=" * 66)
    print("[1/4] 版本清单")
    for key, val in info.items():
        print(f"  {key.ljust(width)} : {val}")
    return info


def check_gpu(matmul_n: int) -> dict:
    import torch

    print("-" * 66)
    print("[2/4] GPU 与算力粗测")
    if not torch.cuda.is_available():
        raise SystemExit("  CUDA 不可用，停止自检（先确认装的是 cu 版 torch）")
    prop = torch.cuda.get_device_properties(0)
    print(f"  device       : {torch.cuda.get_device_name(0)}")
    print(f"  capability   : {torch.cuda.get_device_capability(0)}")
    print(f"  memory       : {prop.total_memory / 1024 ** 3:.2f} GiB")
    print(f"  cudnn        : {torch.backends.cudnn.version()}")

    n = int(matmul_n)
    a = torch.randn(n, n, device="cuda")
    b = torch.randn(n, n, device="cuda")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    reps = 10
    for _ in range(reps):
        c = a @ b
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / reps
    tflops = 2 * n ** 3 / dt / 1e12
    print(f"  matmul {n}x{n}: {dt * 1000:.2f} ms/次  (~{tflops:.2f} TFLOPS FP32)")
    del a, b, c
    torch.cuda.empty_cache()
    return {"device": torch.cuda.get_device_name(0), "matmul_ms": round(dt * 1000, 2),
            "tflops_fp32": round(tflops, 2), "mem_gib": round(prop.total_memory / 1024 ** 3, 2)}


def make_sample(path: Path) -> Path:
    """没有指定图片时，用 scikit-image 内置样图（含人物，COCO 里有 person 类）。"""
    import cv2
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from skimage.data import astronaut

        img = astronaut()
        if img.max() <= 1.0:
            img = (img * 255).round()
        img = img.astype(np.uint8)
    except Exception as exc:
        print(f"  skimage 样图不可用（{exc}），改用合成图")
        img = np.full((640, 640, 3), 200, np.uint8)
        cv2.rectangle(img, (180, 160), (420, 520), (60, 60, 180), -1)
    cv2.imwrite(str(path), img)
    return path


def check_inference(model_path: str, image: str | None) -> dict:
    import torch
    from ultralytics import YOLO

    print("-" * 66)
    print("[3/4] 单图 GPU 推理")
    img = Path(image) if image else make_sample(ROOT / "data" / "samples" / "astronaut.png")
    if not img.exists():
        raise SystemExit(f"  测试图片不存在：{img}")
    weights = Path(model_path)
    if not weights.is_absolute():
        weights = ROOT / weights
    if not weights.exists():
        weights = Path(model_path)  # 交给 ultralytics 自动下载
    print(f"  image  : {img}")
    print(f"  model  : {weights}")

    model = YOLO(str(weights))
    for _ in range(3):  # 预热，排除首次 CUDA 上下文/权重加载开销
        model.predict(str(img), device=0, verbose=False)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    res = model.predict(str(img), device=0, verbose=False)[0]
    torch.cuda.synchronize()
    infer_ms = (time.perf_counter() - t0) * 1000
    peak = torch.cuda.max_memory_allocated() / 1024 ** 2

    names = [model.names[int(c)] for c in res.boxes.cls] if res.boxes is not None else []
    confs = [round(float(x), 3) for x in res.boxes.conf] if res.boxes is not None else []
    n_mask = 0 if res.masks is None else len(res.masks)
    print(f"  推理耗时   : {infer_ms:.2f} ms（预热后单张）")
    print(f"  峰值显存   : {peak:.1f} MiB")
    print(f"  检出目标   : {len(names)} 个 -> {list(zip(names, confs))}")
    print(f"  分割掩码   : {n_mask} 个")

    out = ROOT / "out"
    out.mkdir(parents=True, exist_ok=True)
    saved = out / f"env_check_{img.stem}.jpg"
    res.save(str(saved))
    print(f"  可视化结果 : {saved}")
    return {"image": str(img), "infer_ms": round(infer_ms, 2), "peak_mem_mib": round(peak, 1),
            "detections": names, "conf": confs, "masks": n_mask, "saved": str(saved)}


def check_onnx_export(model_path: str) -> dict:
    import onnx
    from ultralytics import YOLO

    print("-" * 66)
    print("[4/4] ONNX 导出验证")
    weights = Path(model_path)
    if not weights.is_absolute():
        weights = ROOT / weights
    model = YOLO(str(weights))
    produced = model.export(format="onnx", opset=17, dynamic=True, simplify=True, device=0)
    produced = Path(produced)
    m = onnx.load(str(produced))
    onnx.checker.check_model(m)
    size_mb = produced.stat().st_size / 1024 ** 2
    inputs = [(i.name, [d.dim_param or d.dim_value for d in i.type.tensor_type.shape.dim])
              for i in m.graph.input]
    outputs = [(o.name, [d.dim_param or d.dim_value for d in o.type.tensor_type.shape.dim])
               for o in m.graph.output]
    print(f"  导出文件 : {produced}")
    print(f"  体积     : {size_mb:.2f} MB")
    print(f"  opset    : {m.opset_import[0].version}")
    print(f"  输入     : {inputs}")
    print(f"  输出     : {outputs}")
    print("  onnx.checker 通过")
    return {"onnx": str(produced), "size_mb": round(size_mb, 2),
            "opset": int(m.opset_import[0].version),
            "inputs": str(inputs), "outputs": str(outputs)}


def main() -> int:
    ap = argparse.ArgumentParser(description="GPU 环境自检")
    ap.add_argument("--model", default="weights/yolov8s-seg.pt", help="权重路径或名称")
    ap.add_argument("--image", default=None, help="测试图片，缺省用内置样图")
    ap.add_argument("--matmul", default=4096, help="算力粗测的矩阵规模")
    ap.add_argument("--export", action="store_true", help="额外验证 ONNX 导出")
    args = ap.parse_args()

    result = {"versions": print_versions(), "gpu": check_gpu(args.matmul),
              "inference": check_inference(args.model, args.image)}
    if args.export:
        result["onnx_export"] = check_onnx_export(args.model)
    else:
        print("-" * 66)
        print("[4/4] ONNX 导出验证：跳过（加 --export 开启）")

    out = ROOT / "out"
    out.mkdir(parents=True, exist_ok=True)
    report = out / "env_check.json"
    report.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("=" * 66)
    print(f"自检通过，报告已写入 {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())