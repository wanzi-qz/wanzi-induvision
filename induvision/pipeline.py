"""W4 第一步：解码 / 推理 / 后处理三线程流水线，加一份自己实现的 NMS 与掩码解码。

为什么先做 Python 版
--------------------
W4 的目标是 C++ 推理服务，但本机实测 ``nvcc`` / ``cmake`` / ``cl`` / ``ninja`` /
``msbuild`` 全缺（docs/deploy.md §8.10），装 VS Build Tools + CUDA Toolkit 是数 GB 的事。
而这一轮真正要验的工程要点 —— 环形队列解耦阶段、缓冲预分配不每帧 malloc、
pinned memory 零拷贝、异步 H2D/D2H —— 跟语言无关：C++ 版换的是 API 名字，不换结构。
所以先用 Python 把结构立起来并**量出数**（较单线程提升 N×），工具链到位后照搬。

后处理为什么要自己写一份
------------------------
``tools/bench_trt.py`` 的端到端 mAP 走 ultralytics 的 ``val``，后处理是它给的。
流水线要能脱离 ultralytics 独立跑，就必须自己实现 NMS 与掩码解码 —— C++ 版同样得自己写
这两个，先在 Python 里把语义定下来。自己写完必须对拍：同一份 engine、同一批图，
本实现 vs ``ultralytics predict``，逐图比检测数 / 框 IoU / 掩码 IoU，
见 ``tools/bench_pipeline.py --check-ultralytics``。

依赖边界：本模块只用 numpy / opencv / 标准库。torch 与 tensorrt 的 import 全在
``induvision/pinned.py`` 里、且延迟到函数内，CI 不装 GPU 依赖也能跑全部单测。
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .deploy import LetterboxResult, latency_stats

__all__ = [
    "DEFAULT_CONF_THR", "DEFAULT_IOU_THR", "DEFAULT_MAX_DET", "DEFAULT_MASK_THR",
    "RingBuffer", "PipelineError", "Detection",
    "cxcywh_to_xyxy", "iou_one_vs_many", "nms", "batched_nms", "sigmoid",
    "scale_boxes_to_orig", "masks_from_proto", "mask_iou", "postprocess",
    "detections_to_records", "match_detections",
    "run_serial", "Pipeline", "speedup_report",
]

#: 判据与默认值全部跟 ultralytics predict 的默认口径对齐，对拍时才没有额外变量。
DEFAULT_CONF_THR = 0.25
DEFAULT_IOU_THR = 0.7
DEFAULT_MAX_DET = 300
DEFAULT_MASK_THR = 0.5


# ==========================================================================
# 环形队列
# ==========================================================================

class RingBuffer:
    """定长环形队列：槽位一次性分配，``put`` / ``get`` 阻塞，可 ``close``。

    为什么不用 ``queue.Queue``：Queue 每次 put 都新建一个节点对象，长稳跑下来
    GC 压力随帧数涨；环形队列的槽位是预分配的定长 list，写满绕回，零新增分配。
    C++ 版对应的就是 ``std::array`` + 读写两个下标 + ``condition_variable``。

    ``close()`` 的语义是"不再收新的，但已有的能取完"：取空之后 ``get()`` 返回
    ``None`` 而不是永久阻塞。这一条是流水线不死锁的关键 —— 上游异常退出时，
    下游能自然收尾，不会留一条挂死的线程在那儿等。

    顺带记三个诊断量（压测报告里要用）：``max_depth``（队列最深到几，看瓶颈在谁后面）、
    ``overflow_waits``（put 被满队列挡了几次，说明下游慢）、
    ``underflow_waits``（get 被空队列挡了几次，说明上游慢）。

    这两个数按**被挡住的调用次数**记，不是循环轮数：超时唤醒后重新发现队列还是空的，
    不该再记一次（早先就是这么虚高的，``pool_waits`` 一次 acquire 记了 2）。
    """

    def __init__(self, capacity: int = 4, name: str = ""):
        if capacity <= 0:
            raise ValueError(f"capacity 必须 > 0，收到 {capacity}")
        self.capacity = int(capacity)
        self.name = name
        self._slots: List[Any] = [None] * self.capacity
        self._head = 0
        self._tail = 0
        self._count = 0
        self._closed = False
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._not_full = threading.Condition(self._lock)
        self.max_depth = 0
        self.overflow_waits = 0
        self.underflow_waits = 0
        self.puts = 0
        self.gets = 0

    def __len__(self) -> int:
        with self._lock:
            return self._count

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    def put(self, item: Any, timeout: Optional[float] = None) -> bool:
        """放进一个元素；队列满就阻塞等空位。返回 False 只在给了 timeout 且超时时发生。"""
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._not_full:
            waited = False
            while self._count >= self.capacity:
                if self._closed:
                    raise RuntimeError(f"RingBuffer({self.name}) 已关闭，不能再 put")
                if not waited:
                    self.overflow_waits += 1   # 一次阻塞调用只记一次，别按循环轮数记
                    waited = True
                if deadline is None:
                    self._not_full.wait()
                else:
                    left = deadline - time.monotonic()
                    if left <= 0:
                        return False
                    self._not_full.wait(left)
            if self._closed:
                raise RuntimeError(f"RingBuffer({self.name}) 已关闭，不能再 put")
            self._slots[self._tail] = item
            self._tail = (self._tail + 1) % self.capacity
            self._count += 1
            self.puts += 1
            self.max_depth = max(self.max_depth, self._count)
            self._not_empty.notify()
            return True

    def get(self, timeout: Optional[float] = None) -> Any:
        """取一个元素；队列空就阻塞。空且已关闭时返回 ``None``（收尾信号）。"""
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._not_empty:
            waited = False
            while self._count == 0:
                if self._closed:
                    return None
                if not waited:
                    self.underflow_waits += 1   # 同上：pool_waits 直接读这个数
                    waited = True
                if deadline is None:
                    self._not_empty.wait()
                else:
                    left = deadline - time.monotonic()
                    if left <= 0:
                        return None
                    self._not_empty.wait(left)
            item = self._slots[self._head]
            self._slots[self._head] = None
            self._head = (self._head + 1) % self.capacity
            self._count -= 1
            self.gets += 1
            self._not_full.notify()
            return item

    def close(self) -> None:
        """关闭：唤醒所有等待者，让它们把余量取完后收尾。幂等。"""
        with self._lock:
            self._closed = True
            self._not_empty.notify_all()
            self._not_full.notify_all()

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "name": self.name, "capacity": self.capacity, "depth": self._count,
                "max_depth": self.max_depth, "puts": self.puts, "gets": self.gets,
                "overflow_waits": self.overflow_waits,
                "underflow_waits": self.underflow_waits,
                "closed": self._closed,
            }


# ==========================================================================
# 后处理：NMS 与掩码解码
# ==========================================================================

def sigmoid(x: np.ndarray) -> np.ndarray:
    """先夹到 ±60 再算，避免 ``exp`` 溢出成 inf / 打出 RuntimeWarning。"""
    z = np.clip(np.asarray(x, dtype=np.float32), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-z, dtype=np.float32))


def cxcywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """YOLO 的 cxcywh（中心点 + 宽高）→ xyxy（左上 + 右下）。空输入返回空数组不炸。"""
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    out = np.empty_like(b)
    hw = b[:, 2] / 2.0
    hh = b[:, 3] / 2.0
    out[:, 0] = b[:, 0] - hw
    out[:, 1] = b[:, 1] - hh
    out[:, 2] = b[:, 0] + hw
    out[:, 3] = b[:, 1] + hh
    return out


def iou_one_vs_many(box: Sequence[float], boxes: np.ndarray) -> np.ndarray:
    """一个框对一批框的 IoU。并集为 0 时返回 0（两个空框不该算完全重合）。"""
    b = np.asarray(box, dtype=np.float64).reshape(4)
    bs = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    if bs.shape[0] == 0:
        return np.zeros(0, dtype=np.float64)
    x1 = np.maximum(b[0], bs[:, 0])
    y1 = np.maximum(b[1], bs[:, 1])
    x2 = np.minimum(b[2], bs[:, 2])
    y2 = np.minimum(b[3], bs[:, 3])
    inter = np.clip(x2 - x1, 0.0, None) * np.clip(y2 - y1, 0.0, None)
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    area_bs = np.clip(bs[:, 2] - bs[:, 0], 0.0, None) * np.clip(bs[:, 3] - bs[:, 1], 0.0, None)
    union = area_b + area_bs - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float = DEFAULT_IOU_THR,
        top_k: Optional[int] = None) -> List[int]:
    """贪心 NMS，返回保留下来的**下标**（按分数降序）。

    与 torchvision 的 ``ops.nms`` 同语义：IoU **大于** 阈值才抑制（等于阈值时两个都留）。
    ``top_k`` 只在候选里取分数最高的前 k 个参与抑制，跟 TF 版对齐；本项目用不到，
    留着是因为 C++ 版会需要（8400 个候选全排序在嵌入式上是白花的）。
    """
    if not 0.0 < iou_thr <= 1.0:
        raise ValueError(f"iou_thr 必须在 (0, 1]，收到 {iou_thr}")
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    if s.shape[0] != b.shape[0]:
        raise ValueError(f"boxes({b.shape[0]}) 与 scores({s.shape[0]}) 数量不一致")
    order = np.argsort(-s, kind="stable")
    if top_k is not None:
        if top_k <= 0:
            raise ValueError(f"top_k 必须 > 0，收到 {top_k}")
        order = order[:top_k]
    keep: List[int] = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        ious = iou_one_vs_many(b[i], b[rest])
        order = rest[ious <= iou_thr]
    return keep


def batched_nms(boxes: np.ndarray, scores: np.ndarray, labels: np.ndarray,
                iou_thr: float = DEFAULT_IOU_THR) -> List[int]:
    """逐类 NMS：不同类别之间互不抑制（等价于 torchvision ``nms(..., labels=cls)``）。

    为什么必须逐类：工业图上一个缺陷经常压着另一个缺陷（划伤上有个脏污），
    类无关 NMS 会把低分那个当重复框吃掉，直接反映成召回掉点。
    返回的下标按分数**全局**降序，方便后面直接 ``[:max_det]`` 截断。
    """
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    lb = np.asarray(labels).reshape(-1)
    if not (b.shape[0] == s.shape[0] == lb.shape[0]):
        raise ValueError(f"boxes/scores/labels 数量不一致：{b.shape[0]}/{s.shape[0]}/{lb.shape[0]}")
    keep: List[int] = []
    for c in np.unique(lb):
        sel = np.flatnonzero(lb == c)
        local = nms(b[sel], s[sel], iou_thr)
        keep.extend(int(sel[i]) for i in local)
    keep.sort(key=lambda i: -float(s[i]))
    return keep


def scale_boxes_to_orig(boxes: np.ndarray, lb: LetterboxResult) -> np.ndarray:
    """640 letterbox 坐标 → 原图像素坐标，逐行对齐 ultralytics ``scale_boxes``。

    减的是 ``lb.dw`` / ``lb.dh``（**浮点半 pad**），不是 ``lb.left`` / ``lb.top``
    （实际取整后的 pad）。两者在余量为奇数时差 1 像素，ultralytics 用的是前者，
    要跟它对拍就得一样。最后按原图尺寸 clip，框不会跑出画面。
    """
    bs = np.asarray(boxes, dtype=np.float64).reshape(-1, 4).copy()
    if bs.shape[0] == 0:
        return bs
    bs[:, [0, 2]] -= float(lb.dw)
    bs[:, [1, 3]] -= float(lb.dh)
    bs /= float(lb.ratio)
    oh, ow = lb.orig_shape
    bs[:, [0, 2]] = bs[:, [0, 2]].clip(0.0, float(ow))
    bs[:, [1, 3]] = bs[:, [1, 3]].clip(0.0, float(oh))
    return bs


def masks_from_proto(proto: np.ndarray, coeffs: np.ndarray, boxes640: np.ndarray,
                     lb: LetterboxResult, mask_thr: float = DEFAULT_MASK_THR) -> List[np.ndarray]:
    """prototype 掩码解码：``sigmoid(coeff @ proto)`` → 按框裁剪 → 放大到原图 → 贴回全图。

    与 ultralytics 的一处刻意差异：它先把整张 160×160 upsample 到原图再裁剪
    （``ops.process_mask``），这里**只放大裁剪后的小块**。省掉的是 (640/160)² = 16 倍的
    插值面积，矩阵乘也只在裁剪区内做（``proto[:, y1:y2, x1:x2]``）。

    但"放大到多大、贴到哪"必须按**网格区域自己的真实尺寸**算，不能按框的尺寸算。
    第一版就是按框尺寸拉伸的：裁剪区是框外扩到整格（``round``）后的范围，通常比框大
    半格到一格，把它硬拉回框的像素宽高 = 给掩码做了一次 ~5% 的压缩。对拍时表现为
    掩码 IoU 只有 0.85、且 ours 的像素数系统性比 ultralytics 少 2%（见 docs/pipeline.md）。
    现在改成：裁剪区 (cx2-cx1, cy2-cy1) 格 → 放大 (格数 × 每格 letterbox 像素 / ratio)
    → 贴到"网格左上角映射回原图"的位置，与 ultralytics 的整图放大在采样网格上等价，
    只剩 bilinear 的亚像素差。裁剪下标用 ``floor(v + 0.5)``，对齐 torch 的
    ``boxes.round().int()``（numpy 的 round 是四舍六入五成双，会差 1 格）。

    返回：与 ``boxes640`` 等长的 list，每项是 ``(orig_h, orig_w)`` 的 bool 数组。
    退化情况（框在 proto 网格外、放大后宽高为 0）返回全 False，不抛异常 ——
    流水线里一帧炸掉整条不值得。
    """
    p = np.asarray(proto, dtype=np.float32)
    if p.ndim == 4:
        if p.shape[0] != 1:
            raise ValueError(f"proto 的 batch 维只能是 1，收到 {p.shape}")
        p = p[0]
    if p.ndim != 3:
        raise ValueError(f"proto 需要 (C, H, W) 或 (1, C, H, W)，收到 {p.shape}")
    c, mh, mw = p.shape
    scale = int(lb.image.shape[0]) / float(mh)   # 一格 = 多少 letterbox 像素（640/160=4）
    inv = 1.0 / float(lb.ratio)                  # letterbox 像素 → 原图像素
    dw, dh = float(lb.dw), float(lb.dh)
    boxes = np.asarray(boxes640, dtype=np.float64).reshape(-1, 4)
    cf = np.asarray(coeffs, dtype=np.float32).reshape(-1, c)
    if cf.shape[0] != boxes.shape[0]:
        raise ValueError(f"coeffs({cf.shape[0]}) 与 boxes({boxes.shape[0]}) 数量不一致")
    oh, ow = lb.orig_shape
    out: List[np.ndarray] = []
    for k in range(boxes.shape[0]):
        full = np.zeros((oh, ow), dtype=bool)
        x1, y1, x2, y2 = boxes[k] / scale        # 框 → 网格坐标（浮点）
        # 对齐 torch 的 round().int()：四舍五入（远离 0），不是 numpy 的银行家舍入
        gx = [min(max(int(np.floor(v + 0.5)), 0), lim)
              for v, lim in ((x1, mw), (y1, mh), (x2, mw), (y2, mh))]
        cx1, cy1, cx2, cy2 = gx
        if cx2 <= cx1 or cy2 <= cy1:
            out.append(full)
            continue
        crop = p[:, cy1:cy2, cx1:cx2].reshape(c, -1)
        m = sigmoid(cf[k] @ crop).reshape(cy2 - cy1, cx2 - cx1)
        tw = max(1, int(round((cx2 - cx1) * scale * inv)))   # 这块网格在原图上的真实宽高
        th = max(1, int(round((cy2 - cy1) * scale * inv)))
        up = cv2.resize(m.astype(np.float32), (tw, th), interpolation=cv2.INTER_LINEAR)
        px1 = int(round((cx1 * scale - dw) * inv))           # 网格左上角 → 原图
        py1 = int(round((cy1 * scale - dh) * inv))
        dx1, dy1 = max(px1, 0), max(py1, 0)
        dx2, dy2 = min(px1 + tw, ow), min(py1 + th, oh)
        if dx2 <= dx1 or dy2 <= dy1:
            out.append(full)
            continue
        sx1, sy1 = dx1 - px1, dy1 - py1
        full[dy1:dy2, dx1:dx2] = up[sy1:sy1 + (dy2 - dy1), sx1:sx1 + (dx2 - dx1)] >= mask_thr
        out.append(full)
    return out


def mask_iou(a: np.ndarray, b: np.ndarray, erode_px: int = 0) -> float:
    """两个 bool 掩码的 IoU。两个都空 → 1.0（视为一致），只有一个空 → 0.0。

    ``erode_px > 0`` 时先把两个掩码各腐蚀掉这么多个像素再算 IoU。为什么要有这个参数：
    本实现的掩码解码是"裁剪小块再放大"（见 :func:`masks_from_proto`），与 ultralytics 的
    "整图放大再裁剪"在**边界 1~2 px** 上有设计内差异，而 MVTec 的缺陷掩码本身就小
    （几十个像素），1 px 的边差就能把 IoU 拉到 0.85 —— 这时候原始 IoU 分不清"解码错了"
    和"边界采样口径不同"。腐蚀 1 px 后仍高度重合，说明差异只在边上、对象是一致的。
    两个数都报，判据用哪个必须写在报告里，别只挑好看的那个。
    """
    x = np.asarray(a, dtype=bool)
    y = np.asarray(b, dtype=bool)
    if x.shape != y.shape:
        raise ValueError(f"掩码形状不一致：{x.shape} vs {y.shape}")
    if erode_px:
        if erode_px < 0:
            raise ValueError(f"erode_px 不能为负，收到 {erode_px}")
        k = np.ones((2 * int(erode_px) + 1, 2 * int(erode_px) + 1), np.uint8)
        x = cv2.erode(x.astype(np.uint8), k).astype(bool)
        y = cv2.erode(y.astype(np.uint8), k).astype(bool)
    inter = int(np.count_nonzero(x & y))
    union = int(np.count_nonzero(x | y))
    if union == 0:
        return 1.0
    return inter / float(union)


@dataclass
class Detection:
    """一个检测结果。坐标一律是**原图像素**，``box640`` 留着做裁剪与调试。"""

    score: float
    cls: int
    box: np.ndarray  # (4,) xyxy，原图坐标
    box640: np.ndarray  # (4,) xyxy，letterbox 640 坐标
    mask: Optional[np.ndarray] = None  # (orig_h, orig_w) bool

    def area(self) -> float:
        return float(max(0.0, self.box[2] - self.box[0]) * max(0.0, self.box[3] - self.box[1]))

    def to_dict(self, names: Optional[Sequence[str]] = None,
                with_mask: bool = False) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "score": round(float(self.score), 6),
            "cls": int(self.cls),
            "box": [round(float(v), 3) for v in self.box],
        }
        if names is not None:
            d["name"] = names[self.cls] if 0 <= self.cls < len(names) else str(self.cls)
        if with_mask and self.mask is not None:
            d["mask_pixels"] = int(np.count_nonzero(self.mask))
        return d


def detections_to_records(dets: Sequence[Detection], names: Optional[Sequence[str]] = None,
                          with_mask: bool = False) -> List[Dict[str, Any]]:
    return [d.to_dict(names=names, with_mask=with_mask) for d in dets]


def _as_predictions(out0: np.ndarray, n_attrs: int) -> np.ndarray:
    """把 engine 的 ``output0`` 统一成 ``(anchors, n_attrs)``。

    TRT/ONNX 给的是 ``(1, 51, 8400)``，ultralytics 内部用的是 ``(8400, 51)``。
    两种排布都收，靠"属性维等于 n_attrs"来认；两个维度都等于 n_attrs（方阵）时报错，
    让调用方显式给 ``nc``，不猜。
    """
    a = np.asarray(out0, dtype=np.float32)
    if a.ndim == 3:
        if a.shape[0] != 1:
            raise ValueError(f"batch 维只能是 1（profile 是 min=opt=max=1），收到 {a.shape}")
        a = a[0]
    if a.ndim != 2:
        raise ValueError(f"output0 需要 (1, attrs, anchors) 或 (attrs, anchors)，收到 {a.shape}")
    if a.shape[0] == n_attrs and a.shape[1] == n_attrs:
        raise ValueError(f"output0 是方阵 {a.shape}，认不出哪一维是属性维，请显式传 nc")
    if a.shape[0] == n_attrs:
        return np.ascontiguousarray(a.transpose())
    if a.shape[1] == n_attrs:
        return np.ascontiguousarray(a)
    raise ValueError(f"output0 的两个维度都不是 n_attrs={n_attrs}：{a.shape}")


def postprocess(out0: np.ndarray, out1: np.ndarray, lb: LetterboxResult,
                conf_thr: float = DEFAULT_CONF_THR, iou_thr: float = DEFAULT_IOU_THR,
                max_det: Optional[int] = DEFAULT_MAX_DET,
                mask_thr: float = DEFAULT_MASK_THR, nc: Optional[int] = None,
                need_masks: bool = True) -> List[Detection]:
    """YOLOv8-seg 的完整后处理：解框 → 过阈值 → 逐类 NMS → 还原坐标 → 掩码解码。

    ``out0`` = ``(1, 4+nc+32, 8400)``（框 cxcywh / 15 类分数 / 32 个掩码系数），
    ``out1`` = ``(1, 32, 160, 160)`` prototype。``nc`` 不给就从形状反推
    （属性维 = 4 + nc + proto 通道数），反推不出来就报错，不静默猜。

    与 ultralytics 的口径差异只有一处，且是有意的：``max_det`` 截断前先按分数全局降序
    （ultralytics 是逐类 NMS 的拼接顺序再截断）。同一批图上两者选出的集合可能差 1~2 个
    低分框，对拍时体现在 match_rate 上，不影响 mAP 的量级。
    """
    p = np.asarray(out1, dtype=np.float32)
    if p.ndim == 4:
        if p.shape[0] != 1:
            raise ValueError(f"output1 的 batch 维只能是 1，收到 {p.shape}")
        p = p[0]
    if p.ndim != 3:
        raise ValueError(f"output1 需要 (1, C, H, W) 或 (C, H, W)，收到 {p.shape}")
    c = int(p.shape[0])

    a = np.asarray(out0, dtype=np.float32)
    if a.ndim == 3:
        a = a[0]
    if a.ndim != 2:
        raise ValueError(f"output0 需要 (1, attrs, anchors) 或 (attrs, anchors)，收到 {a.shape}")
    if nc is None:
        attrs = min(a.shape)  # 属性维（51）永远远小于 anchor 维（8400）
        nc = attrs - 4 - c
        if nc <= 0:
            raise ValueError(f"从形状 {a.shape} 与 proto 通道 {c} 推不出 nc，请显式传 nc")
    n_attrs = 4 + nc + c
    pred = _as_predictions(out0, n_attrs)

    cls_scores = pred[:, 4:4 + nc]
    coeffs = pred[:, 4 + nc:4 + nc + c]
    score = cls_scores.max(axis=1)
    label = cls_scores.argmax(axis=1)
    keep = score > conf_thr
    if not bool(keep.any()):
        return []
    idx = np.flatnonzero(keep)
    boxes = cxcywh_to_xyxy(pred[idx, :4])
    sc = score[idx].astype(np.float64)
    lb_ids = label[idx]
    cf = coeffs[idx]

    sel = batched_nms(boxes, sc, lb_ids, iou_thr)
    if max_det is not None:
        if max_det <= 0:
            raise ValueError(f"max_det 必须 > 0，收到 {max_det}")
        sel = sel[:max_det]
    if not sel:
        return []
    boxes = boxes[sel]
    sc = sc[sel]
    lb_ids = lb_ids[sel]
    cf = cf[sel]
    boxes_o = scale_boxes_to_orig(boxes, lb)
    masks = masks_from_proto(p, cf, boxes, lb, mask_thr) if need_masks else [None] * len(sel)
    return [
        Detection(score=float(sc[i]), cls=int(lb_ids[i]), box=boxes_o[i],
                  box640=boxes[i], mask=masks[i])
        for i in range(len(sel))
    ]


def match_detections(ours: Sequence[Detection], theirs: Sequence[Detection],
                     iou_thr: float = 0.5, with_mask: bool = True,
                     mask_erode_px: int = 0) -> Dict[str, Any]:
    """两套后处理在同一张图上的对拍结果（贪心匹配，同类且框 IoU ≥ 阈值才算配上）。

    为什么用贪心不用匈牙利：对拍要的是"有没有系统性差异"，贪心按分数降序匹配已经够，
    而且 O(n²) 的匈牙利在 300 个框上纯属浪费。配不上的分别记进 unmatched，
    真出问题时先看 unmatched 的分数分布 —— 全是低分框就是阈值口径差，
    有高分框就是解码错。
    """
    used = set()
    pairs: List[Tuple[int, int, float, Optional[float]]] = []
    order = sorted(range(len(ours)), key=lambda i: -ours[i].score)
    for i in order:
        d = ours[i]
        best_j, best_iou = -1, 0.0
        for j, t in enumerate(theirs):
            if j in used or t.cls != d.cls:
                continue
            iou = float(iou_one_vs_many(d.box, t.box.reshape(1, 4))[0])
            if iou >= iou_thr and iou > best_iou:
                best_j, best_iou = j, iou
        if best_j >= 0:
            used.add(best_j)
            miou = None
            if with_mask and d.mask is not None and theirs[best_j].mask is not None:
                try:
                    miou = mask_iou(d.mask, theirs[best_j].mask,
                                    erode_px=mask_erode_px)
                except ValueError:
                    miou = None  # 两边原图尺寸口径不一致时不硬比
            pairs.append((i, best_j, best_iou, miou))
    box_ious = [p[2] for p in pairs]
    mask_ious = [p[3] for p in pairs if p[3] is not None]
    n_o, n_t = len(ours), len(theirs)
    denom = max(n_o, n_t)
    return {
        "n_ours": n_o, "n_theirs": n_t, "n_matched": len(pairs),
        "match_rate": round(len(pairs) / denom, 4) if denom else 1.0,
        "mean_box_iou": round(float(np.mean(box_ious)), 4) if box_ious else None,
        "min_box_iou": round(float(np.min(box_ious)), 4) if box_ious else None,
        "mean_mask_iou": round(float(np.mean(mask_ious)), 4) if mask_ious else None,
        "mask_erode_px": int(mask_erode_px),
        "mean_abs_score_diff": (round(float(np.mean([abs(ours[i].score - theirs[j].score)
                                                     for i, j, _, _ in pairs])), 5)
                                if pairs else None),
        "unmatched_ours": sorted(set(range(n_o)) - {p[0] for p in pairs}),
        "unmatched_theirs": sorted(set(range(n_t)) - used),
    }


# ==========================================================================
# 流水线
# ==========================================================================

class PipelineError(RuntimeError):
    """某个阶段抛异常时带上下文重抛：哪个阶段、第几个 item、原始异常。"""

    def __init__(self, stage: str, index: Optional[int], exc: BaseException):
        self.stage = stage
        self.index = index
        self.exc = exc
        super().__init__(f"阶段 {stage} 在 item #{index} 上失败：{type(exc).__name__}: {exc}")


StageFn = Callable[[Any], Any]


def _norm_stages(stages: Sequence[Any]) -> List[Tuple[str, StageFn]]:
    """接受 ``[fn, fn, fn]`` 或 ``[("decode", fn), ...]``，统一成 (name, fn) 列表。"""
    if not stages:
        raise ValueError("至少要有一个阶段")
    out: List[Tuple[str, StageFn]] = []
    for i, s in enumerate(stages):
        if isinstance(s, tuple):
            if len(s) != 2:
                raise ValueError(f"阶段 {i} 的元组必须是 (name, fn)，收到 {len(s)} 项")
            name, fn = s
        else:
            name, fn = f"stage{i}", s
        if not callable(fn):
            raise TypeError(f"阶段 {name} 不是可调用对象：{type(fn)}")
        out.append((str(name), fn))
    names = [n for n, _ in out]
    if len(set(names)) != len(names):
        raise ValueError(f"阶段名重复：{names}")
    return out


def _stage_report(name: str, durs: Sequence[float], wall: float) -> Dict[str, Any]:
    st = latency_stats(list(durs)) if durs else latency_stats([])
    busy = sum(durs) / 1000.0
    return {
        "name": name, "calls": len(durs), "busy_s": round(busy, 4),
        "util": round(busy / wall, 4) if wall > 0 else None,
        "mean_ms": st["mean_ms"], "p50_ms": st["p50_ms"],
        "p95_ms": st["p95_ms"], "max_ms": st["max_ms"],
    }


def _report(mode: str, n: int, names: Sequence[str], durs: Sequence[Sequence[float]],
            wall: float, queues: Sequence[RingBuffer], capacity: int,
            errors: Sequence[Dict[str, Any]], n_out: int,
            e2e: Optional[Sequence[float]] = None) -> Dict[str, Any]:
    """汇总一次运行。``e2e`` 是**逐帧端到端延迟**（毫秒），和各阶段耗时是两个口径：

    - 阶段耗时（``stages``）= 这一帧在某一段里被处理了多久，加起来是"干活时间"；
    - 端到端延迟（``e2e_latency``）= 这一帧从进队到结果出来经过了多久，**含排队等待**。

    所以流水线的 e2e 一定比串行的大（帧在队列里等），而吞吐反而更高 ——
    这不是矛盾，是"用单帧延迟换吞吐"这笔交易的账面。报数时两个都要给，
    只报 FPS 会掩盖延迟恶化，只报延迟会让人以为流水线没用。
    """
    return {
        "mode": mode, "n_items": n, "n_results": n_out, "wall_s": round(wall, 4),
        "throughput_fps": round(n_out / wall, 3) if wall > 0 and n_out else None,
        "capacity": capacity,
        "stages": [_stage_report(nm, d, wall) for nm, d in zip(names, durs)],
        "e2e_latency": latency_stats(list(e2e or [])),
        "queues": [q.stats() for q in queues],
        "errors": list(errors),
    }


def run_serial(items: Sequence[Any], stages: Sequence[Any]) -> Tuple[List[Any], Dict[str, Any]]:
    """单线程串行跑同样这几个阶段 —— 流水线的**对照组**，不是替代品。

    加速比必须是同一份 stage 函数、同一批 item、同一台机器同一轮里量出来的两个数之比。
    拿 §8.2 的单帧延迟当分母是错的：那组数字输入常驻显存、不含 H2D 拷贝、不含前后处理。
    """
    st = _norm_stages(stages)
    items = list(items)
    names = [n for n, _ in st]
    durs: List[List[float]] = [[] for _ in st]
    e2e: List[float] = []
    out: List[Any] = []
    t0 = time.perf_counter()
    for it in items:
        s_item = time.perf_counter()
        cur = it
        for k, (_, fn) in enumerate(st):
            s = time.perf_counter()
            cur = fn(cur)
            durs[k].append((time.perf_counter() - s) * 1000.0)
        out.append(cur)
        e2e.append((time.perf_counter() - s_item) * 1000.0)
    wall = time.perf_counter() - t0
    return out, _report("serial", len(items), names, durs, wall, [], 0, [], len(out), e2e)


class Pipeline:
    """N 阶段流水线：每阶段一条线程，阶段之间用 :class:`RingBuffer` 解耦。

    线程账：1 条 feeder（喂 item）+ N 条 stage worker + 主线程当 collector。
    feeder 必须是**独立线程**，不能由主线程兼任 —— 主线程要一边喂一边收，
    而队列是有界的：最后一个阶段的输出队列填满时主线程若还卡在 put 上，就是死锁。

    保序：每个 item 带 index 走完全程，collector 收完按 index 排序再返回。
    所以"流水线"不会打乱结果顺序 —— 压测数字可复现，出错也能定位到第几张图。

    背压：队列满 → 上游 put 阻塞（``overflow_waits`` +1）；队列空 → 下游 get 阻塞
    （``underflow_waits`` +1）。这两个数直接指出瓶颈在哪个阶段，比看 util 更直白。

    异常：任一阶段抛异常 → 记进 errors → 置 stop → 关掉所有队列 → 下游自然收尾 →
    collector 拿到的结果数不足时抛 :class:`PipelineError`，不会静默丢帧。
    """

    def __init__(self, stages: Sequence[Any], capacity: int = 4,
                 join_timeout: float = 30.0):
        self.stages = _norm_stages(stages)
        if capacity <= 0:
            raise ValueError(f"capacity 必须 > 0，收到 {capacity}")
        self.capacity = int(capacity)
        self.join_timeout = float(join_timeout)

    def run(self, items: Sequence[Any]) -> Tuple[List[Any], Dict[str, Any]]:
        items = list(items)
        n = len(items)
        stages = self.stages
        names = [nm for nm, _ in stages]
        qs = [RingBuffer(self.capacity, name=f"q{i}->{names[i]}") for i in range(len(stages))]
        qout = RingBuffer(self.capacity, name="qout")
        durs: List[List[float]] = [[] for _ in stages]
        errors: List[Dict[str, Any]] = []
        stop = threading.Event()
        locks = [threading.Lock() for _ in stages]
        # 逐帧端到端延迟的两个时间戳。enq_ts 只有 feeder 一条线程写，
        # 且写的是自己那格（按下标），collector 只在该格已写完之后才会读到它
        # （结果必然在 put 之后才可能出现在 qout），所以不需要加锁。
        enq_ts: List[Optional[float]] = [None] * n
        e2e: List[float] = []

        def feeder() -> None:
            try:
                for i, it in enumerate(items):
                    if stop.is_set():
                        return
                    # 时间戳打在 put **之前**：队列满时的阻塞等待算这一帧的延迟，
                    # 打在 put 之后就等于把背压藏起来，P95 会好看但没有意义。
                    enq_ts[i] = time.perf_counter()
                    qs[0].put((i, it))
            except RuntimeError:
                pass  # 队列已被关闭（下游出错收尾），feeder 直接退
            finally:
                qs[0].close()

        def worker(k: int) -> None:
            name, fn = stages[k]
            nxt = qs[k + 1] if k + 1 < len(qs) else qout
            try:
                while not stop.is_set():
                    got = qs[k].get()
                    if got is None:
                        break
                    i, payload = got
                    try:
                        s = time.perf_counter()
                        res = fn(payload)
                        dt = (time.perf_counter() - s) * 1000.0
                    except BaseException as e:  # noqa: BLE001 - 要带上下文重抛，不能吞
                        with locks[k]:
                            durs[k].append(0.0)
                        errors.append({"stage": name, "index": i,
                                       "error": f"{type(e).__name__}: {e}"})
                        stop.set()
                        break
                    with locks[k]:
                        durs[k].append(dt)
                    try:
                        nxt.put((i, res))
                    except RuntimeError:
                        break
            finally:
                nxt.close()

        threads = [threading.Thread(target=feeder, name="feeder", daemon=True)]
        threads += [threading.Thread(target=worker, args=(k,), name=f"stage-{names[k]}",
                                     daemon=True) for k in range(len(stages))]
        t0 = time.perf_counter()
        for t in threads:
            t.start()
        got: Dict[int, Any] = {}
        while len(got) < n:
            item = qout.get()
            if item is None:
                break
            t_out = time.perf_counter()
            idx = item[0]
            got[idx] = item[1]
            if enq_ts[idx] is not None:
                e2e.append((t_out - enq_ts[idx]) * 1000.0)
        wall = time.perf_counter() - t0
        stop.set()
        for q in qs:
            q.close()
        qout.close()
        for t in threads:
            t.join(timeout=self.join_timeout)
        results = [got[i] for i in sorted(got)]
        if errors:
            e0 = errors[0]
            raise PipelineError(e0["stage"], e0["index"],
                                RuntimeError(e0["error"]))
        return results, _report("pipeline", n, names, durs, wall, qs + [qout],
                                self.capacity, errors, len(results), e2e)


def speedup_report(serial_rep: Dict[str, Any], pipe_rep: Dict[str, Any]) -> Dict[str, Any]:
    """把串行与流水线两份报告并成一张对照表（加速比、每阶段 util、瓶颈阶段）。

    ``bottleneck`` 取流水线里 busy 时间最长的那个阶段：单线程时三个阶段互相等，
    流水线跑起来之后总吞吐由最慢的那一段决定，其它阶段会在那里排队
    （对应 ``overflow_waits``）。C++ 版要优化的就是这一段。
    """
    s_fps = serial_rep.get("throughput_fps")
    p_fps = pipe_rep.get("throughput_fps")
    stages = pipe_rep.get("stages") or []
    bottleneck = None
    if stages:
        b = max(stages, key=lambda s: (s.get("busy_s") or 0.0))
        bottleneck = b.get("name")
    return {
        "serial_wall_s": serial_rep.get("wall_s"),
        "pipeline_wall_s": pipe_rep.get("wall_s"),
        "serial_fps": s_fps,
        "pipeline_fps": p_fps,
        "speedup": round(p_fps / s_fps, 3) if (s_fps and p_fps) else None,
        "n_items": pipe_rep.get("n_items"),
        "bottleneck_stage": bottleneck,
        "stage_util": {s["name"]: s.get("util") for s in stages},
        "serial_stage_util": {s["name"]: s.get("util") for s in (serial_rep.get("stages") or [])},
    }
