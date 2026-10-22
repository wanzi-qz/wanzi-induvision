"""W4 第一步的下半场：pinned memory 零拷贝 + 输出缓冲池轮换的 TensorRT 客户端。

为什么不直接用 ``trt_infer.TrtRunner``
--------------------------------------
``TrtRunner.__call__`` 是"每帧新建缓冲"的写法：``torch.from_numpy(x).to(device)`` 走的是
**可分页**内存 H2D（CUDA 驱动内部还要先过一遍自己的 staging buffer），输出张量每帧
``torch.empty`` 一个新的，拷回 CPU 又是 ``.float().cpu().numpy()`` 连着两次落地。
单帧延迟压测（docs/deploy.md §8.2）这样写没毛病——那组数字量的是 kernel 本身，
输入常驻显存、不含拷贝。但流水线里 H2D/D2H 是每帧实打实要走的，缓冲怎么分配直接
决定这一段的耗时，于是有了这个类。

三件事把拷贝和分配的开销抠掉：

1. **pinned（page-locked）host 缓冲**。``pin_memory=True`` 的内存不会被操作系统换页，
   DMA 能直接读写，``non_blocking=True`` 才是真的异步（对可分页内存，non_blocking
   会退化成同步拷贝，这是最常踩的一个坑）。前处理结果直接 ``np.copyto`` 写进它的
   ``.numpy()`` 视图，中间不多一份 ndarray。
2. **常驻 device 缓冲**。输入输出张量整轮只分配一次，每帧只做 memcpy，不进 torch
   的缓存分配器；显存占用从"每帧抖动"变成一条平线，嵌入式上这点比速度更要紧。
3. **缓冲池轮换**（``pool_size``）。输出缓冲必须等后处理（stage3）读完才能复用。
   流水线里 stage2 可能已经在推第 n+1 帧，stage3 还在解第 n 帧的掩码——只有一份缓冲
   就是数据竞争，症状是"偶发的检测结果和串行不一致"，属于最难查的一类 bug。所以输入
   和输出都按 slot 隔离，令牌（token）贯穿 decode → infer → postprocess 三个阶段：
   stage1 ``acquire()`` 拿令牌并往该 slot 的 pinned 输入里写前处理结果，stage3 用完
   ``release()`` 还回去。空闲令牌用 :class:`RingBuffer` 排队，拿不到就阻塞，
   ``pool_waits`` 这个数直接告诉你 pool_size 是不是给小了。

依赖边界：torch / tensorrt 都在函数内延迟导入；``torch_mod`` / ``runner`` 可以注入，
CI（无 GPU、无 torch、无 tensorrt）用假对象就能覆盖全部分支，见 tests/test_pinned.py。
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .pipeline import RingBuffer

__all__ = [
    "PinnedOutputs", "PinnedTrtClient", "require_torch", "missing_gpu_dep_message",
]


def missing_gpu_dep_message(exc: BaseException) -> str:
    """缺 GPU 依赖时的统一报错口径。别写成 "No module named torch" 就完事。"""
    return (
        f"缺 GPU 依赖：{type(exc).__name__}: {exc}。TensorRT 这条路要按 docs/environment.md "
        f"§6 装 requirements-gpu.txt（CUDA 版 torch + tensorrt-cu13 三件套）；"
        f"CI 与本机 base 环境都不装，单测靠注入假的 torch_mod / runner 覆盖。"
    )


def require_torch():
    """拿 torch，拿不到就给出能照着做的报错（含装哪份 requirements）。"""
    try:
        import torch  # noqa: PLC0415 - 延迟导入，见模块文档
    except ImportError as e:  # pragma: no cover - 只在没装 torch 的环境走到
        raise RuntimeError(missing_gpu_dep_message(e)) from e
    return torch


@dataclass
class PinnedOutputs:
    """一次 ``forward`` 的产物：令牌 + 各输出张量的 numpy 视图（**零拷贝**）。

    注意 ``arrays`` 里的数组是池子里那块 pinned 内存的视图，不是副本：``release(token)``
    之后再读，读到的可能是下一帧的数据。要留到后面用就自己 ``np.array(x, copy=True)``。
    """

    token: int
    names: List[str]
    arrays: List[np.ndarray]
    owned: bool = False  # True = 这次 forward 自己 acquire 的令牌，调用方负责 release

    def get(self, name: str) -> np.ndarray:
        return self.arrays[self.names.index(name)]

    def shapes(self) -> Dict[str, Tuple[int, ...]]:
        return {n: tuple(a.shape) for n, a in zip(self.names, self.arrays)}


class PinnedTrtClient:
    """按 slot 隔离输入输出缓冲的 TRT 客户端，给流水线用。

    用法（三个阶段各持一个令牌）::

        tok = client.acquire()
        np.copyto(client.input_view(tok), chw)   # 前处理直接写进 pinned 内存
        out = client.forward(tok)                # H2D -> kernel -> D2H，全程 non_blocking
        dets = postprocess(out.arrays[0], out.arrays[1], lb)
        client.release(tok)

    线程安全：不同令牌之间互不干扰，可以多线程并行调 ``forward``——但 CUDA stream 是
    一条，真正的并行发生在"H2D/kernel/D2H 与 CPU 前后处理之间"，不是 kernel 之间。
    这正好是流水线要的形态。
    """

    def __init__(self, engine_path: Optional[Any] = None, *, device: str = "cuda:0",
                 pool_size: int = 2, imgsz: int = 640, channels: int = 3,
                 runner: Optional[Any] = None, torch_mod: Optional[Any] = None,
                 out_shapes: Optional[Dict[str, Sequence[int]]] = None,
                 out_dtypes: Optional[Dict[str, Any]] = None):
        if pool_size <= 0:
            raise ValueError(f"pool_size 必须 > 0，收到 {pool_size}")
        if imgsz <= 0:
            raise ValueError(f"imgsz 必须 > 0，收到 {imgsz}")
        self.torch = torch_mod if torch_mod is not None else require_torch()
        if runner is None:
            if engine_path is None:
                raise ValueError("没传 runner 就必须传 engine_path")
            from .trt_infer import TrtRunner  # noqa: PLC0415 - 内部才 import tensorrt
            runner = TrtRunner(engine_path, device=device)
            self._owns_runner = True
        else:
            self._owns_runner = False
        self.runner = runner
        self.pool_size = int(pool_size)
        self.imgsz = int(imgsz)
        self.channels = int(channels)
        self.in_shape: Tuple[int, ...] = (1, self.channels, self.imgsz, self.imgsz)
        self.device = device
        self.input_name = runner.input_name
        self.output_names = list(runner.output_names)
        if not self.output_names:
            raise ValueError("engine 没有输出张量，检查是不是拿错了文件")

        self._slots: List[Dict[str, Any]] = []
        self.out_shapes: Dict[str, Tuple[int, ...]] = {}
        self._build_pool(out_shapes, out_dtypes)
        self._free = RingBuffer(capacity=self.pool_size, name="pinned_pool")
        for i in range(self.pool_size):
            self._free.put(i)
        self.forwards = 0

    # -- 构建 ---------------------------------------------------------------
    def _probe_out_shapes(self) -> Dict[str, Tuple[int, ...]]:
        """问 context 要输出形状。**必须先 set_input_shape**，否则拿到的是 profile 的
        min/opt/max，不是本次输入对应的形状（trt_infer.TrtRunner 里记过这个坑）。"""
        ctx = self.runner.context
        ctx.set_input_shape(self.input_name, tuple(self.in_shape))
        return {n: tuple(int(v) for v in ctx.get_tensor_shape(n)) for n in self.output_names}

    def _resolve_out_dtype(self, name: str) -> Any:
        trt = getattr(self.runner, "trt", None)
        eng = getattr(self.runner, "engine", None)
        if trt is None or eng is None:
            return self.torch.float32
        try:
            npdt = trt.nptype(eng.get_tensor_dtype(name))
        except Exception:  # noqa: BLE001 - 假 runner / 老 API 都退回 float32
            return self.torch.float32
        return self.torch.from_numpy(np.empty((), dtype=npdt)).dtype

    def _build_pool(self, out_shapes: Optional[Dict[str, Sequence[int]]],
                    out_dtypes: Optional[Dict[str, Any]]) -> None:
        shapes = {n: tuple(int(v) for v in s) for n, s in (out_shapes or {}).items()}
        missing = [n for n in self.output_names if n not in shapes]
        if missing:
            shapes.update({n: s for n, s in self._probe_out_shapes().items() if n in missing})
        self.out_shapes = {n: shapes[n] for n in self.output_names}
        dt_default = self.torch.float32
        for _ in range(self.pool_size):
            in_host = self.torch.empty(self.in_shape, dtype=dt_default, pin_memory=True)
            slot: Dict[str, Any] = {
                "in_host": in_host,
                "in_np": in_host.numpy(),          # 零拷贝写入口，(1, C, H, W)
                "in_dev": self.torch.empty(self.in_shape, dtype=dt_default, device=self.device),
                "dev": {}, "host": {}, "np": {},
            }
            for n in self.output_names:
                dt = (out_dtypes or {}).get(n) or self._resolve_out_dtype(n)
                shp = self.out_shapes[n]
                slot["dev"][n] = self.torch.empty(shp, dtype=dt, device=self.device)
                slot["host"][n] = self.torch.empty(shp, dtype=dt, pin_memory=True)
                slot["np"][n] = slot["host"][n].numpy()
            self._slots.append(slot)

    # -- 令牌 ---------------------------------------------------------------
    def acquire(self, timeout: Optional[float] = None) -> int:
        """拿一个空闲 slot 的令牌；池子空了就阻塞（``pool_waits`` +1）。"""
        tok = self._free.get(timeout=timeout)
        if tok is None:
            raise RuntimeError(
                f"输出缓冲池（pool_size={self.pool_size}）已关闭或 acquire 超时"
                f"（timeout={timeout}）；如果是因为忘了 release，检查 stage3 的 finally")
        return int(tok)

    def release(self, token: int) -> None:
        self._free.put(int(token))

    @property
    def pool_waits(self) -> int:
        """acquire 时池子空、被迫等的次数。持续 > 0 说明 pool_size 给小了。"""
        return self._free.underflow_waits

    def input_view(self, token: int) -> np.ndarray:
        """该 slot 的 pinned 输入缓冲的 ``(C, H, W)`` numpy 视图，前处理直接往这里写。"""
        return self._slots[int(token)]["in_np"][0]

    # -- 执行 ---------------------------------------------------------------
    def _stream_ctx(self):
        """``with torch.cuda.stream(runner.stream)``；假 torch 没有 cuda 就退化成空上下文。"""
        cuda = getattr(self.torch, "cuda", None)
        fn = getattr(cuda, "stream", None) if cuda is not None else None
        if fn is None:
            return contextlib.nullcontext()
        return fn(self.runner.stream)

    def forward(self, token: int) -> PinnedOutputs:
        """跑该 slot 上已经写好的一帧：H2D -> execute_async_v3 -> D2H -> 一次 synchronize。

        整条链都在 ``runner.stream`` 上，中间不 sync；只在 D2H 全部提交之后同步一次。
        每帧 sync 两次（H2D 后一次、D2H 后一次）是常见写法，白白多两个 launch gap。
        """
        tok = int(token)
        slot = self._slots[tok]
        ctx = self.runner.context
        with self._stream_ctx():
            slot["in_dev"].copy_(slot["in_host"], non_blocking=True)
            ctx.set_input_shape(self.input_name, tuple(self.in_shape))
            ctx.set_tensor_address(self.input_name, int(slot["in_dev"].data_ptr()))
            for n in self.output_names:
                ctx.set_tensor_address(n, int(slot["dev"][n].data_ptr()))
            ok = ctx.execute_async_v3(self.runner.stream.cuda_stream)
            if not ok:
                raise RuntimeError("execute_async_v3 返回 False（engine 与当前 GPU/TRT 不匹配？）")
            for n in self.output_names:
                slot["host"][n].copy_(slot["dev"][n], non_blocking=True)
            self.runner.stream.synchronize()
        self.forwards += 1
        return PinnedOutputs(token=tok, names=list(self.output_names),
                             arrays=[slot["np"][n] for n in self.output_names])

    def forward_array(self, x: np.ndarray, token: Optional[int] = None) -> PinnedOutputs:
        """一次性接口：写 pinned -> forward。``token`` 为空时自己 acquire 并标记 owned。"""
        tok = self.acquire() if token is None else int(token)
        owned = token is None
        try:
            arr = np.asarray(x, dtype=np.float32)
            if arr.ndim == 3:
                arr = arr[None]
            if tuple(arr.shape) != tuple(self.in_shape):
                raise ValueError(f"输入形状 {arr.shape} 与 engine profile {self.in_shape} 不一致")
            np.copyto(self._slots[tok]["in_np"], arr)  # 写进 pinned，零拷贝
            out = self.forward(tok)
            out.owned = owned
            return out
        except BaseException:
            if owned:
                self.release(tok)  # 出错也要还令牌，否则池子越用越小
            raise

    # -- 统计与释放 ---------------------------------------------------------
    def _tensor_bytes(self, t: Any) -> int:
        try:
            return int(t.numel()) * int(t.element_size())
        except Exception:  # noqa: BLE001 - 假张量没有这俩方法，统计留空即可
            return 0

    def stats(self) -> Dict[str, Any]:
        pinned = dev = 0
        for s in self._slots:
            pinned += self._tensor_bytes(s["in_host"])
            dev += self._tensor_bytes(s["in_dev"])
            for n in self.output_names:
                pinned += self._tensor_bytes(s["host"][n])
                dev += self._tensor_bytes(s["dev"][n])
        return {
            "pool_size": self.pool_size, "forwards": self.forwards,
            "pool_waits": self.pool_waits, "free_now": len(self._free),
            "pinned_host_mib": round(pinned / 1024**2, 3),
            "device_mib": round(dev / 1024**2, 3),
            "in_shape": list(self.in_shape),
            "out_shapes": {n: list(s) for n, s in self.out_shapes.items()},
            "owns_runner": self._owns_runner,
        }

    def close(self) -> None:
        self._free.close()
        self._slots = []
        if self._owns_runner:
            self.runner.close()

    def __enter__(self) -> "PinnedTrtClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
