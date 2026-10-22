"""部署链路的纯 CPU 基础设施：前处理对齐、张量比差、延迟分位、样本抽样。

为什么要有这个模块
------------------
W3 起，同一个 ``best.pt`` 会以三种后端出现：PyTorch（.pt）、ONNX Runtime（.onnx）、
TensorRT（.engine）。换后端不该换语义，而"没换语义"这件事只能靠实测证明：拿同一批
输入分别喂两边，逐个输出张量比差。

比差的前提是两边吃到的输入**逐字节相同**。前处理只要差一点——ratio 的取整方式、
pad 是否居中、BGR/RGB 顺序、有没有 /255、CHW 还是 HWC——量出来的差就全是前处理
口径的差，后端真正的差反而被淹掉。所以这里把 ultralytics 的前处理复刻成
numpy/opencv 实现，两边共用同一份，差值才有归因意义。

复刻而不是 import 的原因：``import ultralytics`` 会连带拖进 torch，CI（不装 torch，
见 requirements-ci.txt）就跑不了这段逻辑的单测了。

口径来源（ultralytics 8.4.183）
------------------------------
* ``data/augment.py: LetterBox``，走 ``auto=False`` 分支：pad 成正方形，
  不向 stride 取整。ONNX/TensorRT 后端在 predictor 里就是这个分支
  （``auto=pt and self.pt``，非 .pt 权重恒为 False），所以复刻它。
* ``engine/predictor.py: pre_process``：BGR→RGB、HWC→CHW、float32、除 255、
  ``ascontiguousarray``。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml

__all__ = [
    "PAD_COLOR",
    "IMG_EXTS",
    "LetterboxResult",
    "letterbox",
    "preprocess",
    "read_image_bgr",
    "compare_tensors",
    "latency_stats",
    "sample_paths",
    "read_split_images",
]

#: ultralytics 默认填充色（灰）。改了它，两边的输入就不再逐字节相同了。
PAD_COLOR: Tuple[int, int, int] = (114, 114, 114)

IMG_EXTS: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


@dataclass(frozen=True)
class LetterboxResult:
    """``letterbox`` 的返回值：图 + 还原坐标需要的全部参数。

    这些字段留着不只是调试用：后处理要把检测框从 640x640 映回原图尺寸，
    用的就是 ``ratio`` / ``dw`` / ``dh``（``x_orig = (x_640 - dw) / ratio``）。
    """

    image: np.ndarray  # (H, W, 3) uint8，H == W == imgsz
    ratio: float  # 等比缩放系数，长宽共用一个（min 取，保证不裁内容）
    dw: float  # 宽度方向总 pad（像素，未取整前）
    dh: float  # 高度方向总 pad
    orig_shape: Tuple[int, int]  # 原图 (h, w)
    new_unpad: Tuple[int, int]  # 缩放后、加边前的 (w, h)
    top: int  # 实际上边 pad 行数
    left: int  # 实际左边 pad 列数


def letterbox(
    img: np.ndarray,
    new_shape: int = 640,
    color: Tuple[int, int, int] = PAD_COLOR,
    scaleup: bool = True,
) -> LetterboxResult:
    """等比缩放到正方形并居中 pad，逐行对齐 ultralytics ``LetterBox(auto=False)``。

    几个容易踩的点，全都按 ultralytics 的实现来：

    * ``ratio = min(new/h, new/w)``，长宽共用一个，图像不会变形；
    * ``scaleup=False`` 时 ``ratio`` 被夹到 ≤1（只缩不放）；
    * 缩放后的尺寸用 ``round`` 取整，pad 余量按 ``int(round(dh-0.1))`` /
      ``int(round(dh+0.1))`` 分给上下——**上下不对称时会差 1 像素**，
      自己写 ``dh//2`` 会和 ultralytics 对不上；
    * pad 用 ``BORDER_CONSTANT`` 填 ``color``，不是镜像也不是黑边。
    """
    h, w = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / h, new_shape[1] / w)
    if not scaleup:
        r = min(r, 1.0)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    dw_half = dw / 2
    dh_half = dh / 2
    out = img
    if (w, h) != new_unpad:
        out = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh_half - 0.1)), int(round(dh_half + 0.1))
    left, right = int(round(dw_half - 0.1)), int(round(dw_half + 0.1))
    out = cv2.copyMakeBorder(out, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return LetterboxResult(
        image=out, ratio=r, dw=dw_half, dh=dh_half, orig_shape=(h, w),
        new_unpad=new_unpad, top=top, left=left,
    )


def read_image_bgr(path) -> Optional[np.ndarray]:
    """读图成 BGR uint8；``cv2.imread`` 返回 None 时退回 ``np.fromfile`` + ``imdecode``。

    为什么要有这个退回：本仓库的数据集落在 ``out/yolo_mvtec15/``，绝对路径里带中文
    （``...\\ChatGPT\\简历\\induvision\\...``）。opencv-python 5.0.0 在 Windows 上
    ``imread`` 打不开这种路径，返回 None 并打一行
    ``cv::findDecoder imread_(...): can't open/read file: check file path/integrity``，
    看着像文件坏了，其实文件完好——同一张图用 ``np.fromfile`` + ``cv2.imdecode``
    能正常读出 900x900x3，PIL 也读得到。同一份数据集 ultralytics 训练时没事，
    是因为它内部有 PIL 兜底。

    顺序是"先 imdecode，兜底才 imread"：imdecode 那条路不挑路径编码，且不会往
    stderr 刷 WARN；imread 只留给 np.fromfile 啃不动的网络盘 / 超长路径。
    """
    pth = Path(path)
    try:
        buf = np.fromfile(str(pth), dtype=np.uint8)
    except OSError:
        buf = None
    if buf is not None and buf.size:
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is not None:
            return img
    # 兜底：网络盘 / 超长路径这类 fromfile 啃不动的情况，交回 OpenCV 自己处理。
    # 放在第二位是因为它失败时会往 stderr 打一行 cv::findDecoder 的 WARN，
    # 每张图刷一行，看着像数据坏了（就是上面那个坑一开始骗到我的地方）。
    return cv2.imread(str(pth), cv2.IMREAD_COLOR)


def preprocess(
    img_bgr: np.ndarray,
    imgsz: int = 640,
    color: Tuple[int, int, int] = PAD_COLOR,
) -> Tuple[np.ndarray, LetterboxResult]:
    """BGR uint8 原图 → ``(1) 可直接喂网络的 float32 CHW 张量, (2) letterbox 元信息``。

    输出形状 ``(3, imgsz, imgsz)``、dtype ``float32``、值域 ``[0, 1]``、通道序 RGB。
    batch 维留给调用方加（``np.stack``），这样单图对齐和多图压测能共用同一份代码。
    """
    lb = letterbox(img_bgr, imgsz, color=color)
    rgb = lb.image[..., ::-1]  # BGR -> RGB，ultralytics 就是这么切的
    chw = np.ascontiguousarray(rgb.transpose(2, 0, 1)).astype(np.float32) / 255.0
    return chw, lb


def compare_tensors(a: np.ndarray, b: np.ndarray) -> Dict[str, object]:
    """比两个后端在同一输入上的输出，返回可入库的差值摘要（全部 float，便于写 JSON）。

    四个量各管一件事，别只看其中一个：

    * ``max_abs`` 最坏情况。分割输出里有 sigmoid 前的 logits 和 proto 掩码，
      个别元素差 1e-2 属正常，所以它大不代表模型坏了；
    * ``mean_abs`` 平均水位，判断是"整体偏了"还是"个别点炸了"；
    * ``rel_l2`` 相对误差 ``||a-b||₂ / ||a||₂``，跨输出张量可比
      （box 坐标是几百的量级、mask 系数是 0~1，绝对差没有可比性）；
    * ``cosine`` 方向一致性，最贴近"检测结果会不会变"。**判 FP32 对齐主要看它
      和 ``rel_l2``**：cosine > 0.9999 且 rel_l2 < 1e-3 就可以认为两个后端等价。
    """
    a_raw, b_raw = np.asarray(a), np.asarray(b)
    if a_raw.shape != b_raw.shape:
        # 形状必须在 ravel 之前比：(1,51,8400) 和 (1,8400,51) 元素数相同，
        # 拉平后比差会得到一个"看着还行"的数，而实际上两个后端根本没对齐。
        return {
            "shape_a": list(a_raw.shape), "shape_b": list(b_raw.shape),
            "shape_match": False, "max_abs": None, "mean_abs": None,
            "rel_l2": None, "cosine": None,
        }
    a_arr = a_raw.astype(np.float64).ravel()
    b_arr = b_raw.astype(np.float64).ravel()
    diff = a_arr - b_arr
    denom = float(np.linalg.norm(a_arr))
    na, nb = float(np.linalg.norm(a_arr)), float(np.linalg.norm(b_arr))
    cos = float(np.dot(a_arr, b_arr) / (na * nb)) if na > 0 and nb > 0 else (1.0 if na == nb == 0 else 0.0)
    return {
        "shape_a": list(a_raw.shape), "shape_b": list(b_raw.shape),
        "shape_match": True,
        "max_abs": float(np.max(np.abs(diff))) if diff.size else 0.0,
        "mean_abs": float(np.mean(np.abs(diff))) if diff.size else 0.0,
        "rel_l2": float(np.linalg.norm(diff) / denom) if denom > 0 else 0.0,
        "cosine": cos,
    }


def latency_stats(samples_ms: Sequence[float]) -> Dict[str, Optional[float]]:
    """一堆单帧延迟（毫秒）→ 分位摘要。

    报延迟只报均值是自欺欺人：推理服务的体感由 P95/P99 决定（队列抖动、
    显存分配、后处理毛刺全在尾巴上）。这里一次把 mean / p50 / p90 / p95 / p99
    都算出来，后面写报告直接抄。``fps`` 按均值算，只作横向参考。
    """
    arr = np.asarray(list(samples_ms), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0, "mean_ms": None, "p50_ms": None, "p90_ms": None,
                "p95_ms": None, "p99_ms": None, "min_ms": None, "max_ms": None, "fps": None}
    mean = float(arr.mean())
    return {
        "count": int(arr.size),
        "mean_ms": round(mean, 3),
        "p50_ms": round(float(np.percentile(arr, 50)), 3),
        "p90_ms": round(float(np.percentile(arr, 90)), 3),
        "p95_ms": round(float(np.percentile(arr, 95)), 3),
        "p99_ms": round(float(np.percentile(arr, 99)), 3),
        "min_ms": round(float(arr.min()), 3),
        "max_ms": round(float(arr.max()), 3),
        "fps": round(1000.0 / mean, 2) if mean > 0 else None,
    }


def sample_paths(paths: Sequence[Path], n: int) -> List[Path]:
    """从有序列表里等间距抽 n 个，不足则全取。

    给两个场景用：精度对齐的输入子集，以及后面 TensorRT INT8 的校准集。
    刻意用等间距而不是随机：结果只取决于排序后的列表，换机器换时间跑都一样，
    而且能覆盖到数据分布的头尾（随机抽 32 张有可能全落在同一类上）。
    """
    items = list(paths)
    if n <= 0:
        return []
    if n >= len(items):
        return items
    idx = np.unique(np.linspace(0, len(items) - 1, n).round().astype(int))
    return [items[i] for i in idx]


def read_split_images(data_yaml: Path, split: str = "test") -> List[Path]:
    """按本仓库的 ``data.yaml`` 口径列出某个 split 的全部图片（排序后返回）。

    ``data.yaml`` 里的值是相对 ``path`` 的目录（如 ``images/test``）；也兼容
    直接给单个文件、或给一串目录/文件的写法。图片不存在时返回空列表而不是抛异常，
    让调用方决定是报错还是降级。
    """
    y = Path(data_yaml)
    if not y.is_file():
        raise FileNotFoundError(f"缺少 data.yaml：{y}")
    d = yaml.safe_load(y.read_text(encoding="utf-8")) or {}
    root = Path(d.get("path") or y.parent)
    if not root.is_absolute():
        root = (y.parent / root).resolve()
    raw = d.get(split)
    if raw is None:
        raise KeyError(f"{y} 里没有 '{split}:' 键，无法取图（转换器会写这个键）")
    entries = raw if isinstance(raw, (list, tuple)) else [raw]
    out: List[Path] = []
    for e in entries:
        p = Path(str(e))
        if not p.is_absolute():
            p = root / p
        if p.is_dir():
            out += [q for q in sorted(p.rglob("*")) if q.suffix.lower() in IMG_EXTS]
        elif p.is_file():
            out.append(p)
    return sorted(set(out))