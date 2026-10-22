# -*- coding: utf-8 -*-
"""TensorRT engine 的读写与推理封装（W3b）。

放在 ``induvision`` 包里而不是 ``tools/`` 下，因为 ``build_trt.py`` 和 ``bench_trt.py``
两边都要用，而 ``tools`` 不是包、脚本之间互相 import 只能靠改 ``sys.path``。

依赖分层是刻意的：**header 编解码与 metadata 提取完全不依赖 ``tensorrt``**
（只用标准库 + ``onnx``），所以 base 环境的单测能覆盖；真正要 CUDA 的
``TrtRunner`` 把 ``import tensorrt`` 推迟到构造时，没装 TensorRT 的机器上
import 这个模块本身不会炸。
"""
from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

#: ultralytics 的 .engine header 里，JSON 长度占的字节数（小端、有符号）
HEADER_LEN_BYTES = 4

# --------------------------------------------------------------------------
# header 编解码：让自建的 engine 能被 ultralytics 直接加载
# --------------------------------------------------------------------------

def encode_header(meta: Optional[Dict[str, object]]) -> bytes:
    """把 metadata 编成 ultralytics 的 ``.engine`` 前置 header：4 字节小端长度 + JSON。

    为什么要迁就一个私有格式：engine 是我们自己用 TensorRT Python API 建的，文件里
    没有 ultralytics 期望的 metadata。缺了它，``AutoBackend`` 只能靠输出形状反推
    ``task`` / ``nc``，类别名退化成 ``0..14``，mAP 那一层就没法和 deploy.md §4.2 里
    PyTorch / ONNX 那两行放进同一张表比较。格式照抄
    ``ultralytics/utils/export/engine.py`` 的写盘段：

    .. code-block:: python

        t.write(len(meta).to_bytes(4, byteorder="little", signed=True))
        t.write(meta.encode())
        t.write(engine)
    """
    if not meta:
        return b""
    raw = json.dumps(meta)
    return len(raw).to_bytes(HEADER_LEN_BYTES, byteorder="little", signed=True) + raw.encode()

def decode_header(blob: Sequence[int]) -> Tuple[int, Dict[str, object]]:
    """读 header，返回 ``(engine 起始偏移, metadata)``；没有 header 就是 ``(0, {})``。

    判据与 ``ultralytics.nn.backends.base.AutoBackend.engine_header`` 一致：前 4 字节
    当小端长度，长度越界、或后面那段不是合法 JSON，就认定这个文件没有 header。
    依据是"engine 的原始字节几乎不可能正好构成一段合法 JSON"——这是 ultralytics
    自己的赌注，跟着它赌才能保证两边读同一个文件的结论一致。
    """
    b = bytes(blob)
    if len(b) < HEADER_LEN_BYTES:
        return 0, {}
    n = int.from_bytes(b[:HEADER_LEN_BYTES], byteorder="little")
    if not (0 < n <= len(b) - HEADER_LEN_BYTES):
        return 0, {}
    try:
        meta = json.loads(b[HEADER_LEN_BYTES:HEADER_LEN_BYTES + n])
    except ValueError:
        return 0, {}
    if not isinstance(meta, dict):
        return 0, {}
    return HEADER_LEN_BYTES + n, meta

def write_engine_file(path, engine_blob: bytes,
                      meta: Optional[Dict[str, object]] = None) -> int:
    """写 ``header + engine``，返回文件总字节数。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    head = encode_header(meta)
    p.write_bytes(head + engine_blob)
    return len(head) + len(engine_blob)

def read_engine_file(path) -> Tuple[bytes, Dict[str, object]]:
    """读回 ``(纯 engine 字节, metadata)``。给 ``TrtRunner`` 用：TRT 只吃纯 engine。"""
    b = Path(path).read_bytes()
    off, meta = decode_header(b)
    return b[off:], meta

# --------------------------------------------------------------------------
# 从源 ONNX 继承 metadata
# --------------------------------------------------------------------------

_INT_KEYS = ("stride", "batch", "channels")
_BOOL_KEYS = ("end2end",)
_LITERAL_KEYS = ("imgsz", "names", "args")

def _coerce(key: str, val: str):
    """把 ONNX metadata_props 里的字符串还原成 ultralytics 期望的类型。

    ONNX 的 metadata 只能存字符串，所以 ``imgsz`` 是 ``"[640, 640]"``、``names`` 是
    ``"{0: 'bottle', ...}"``、``end2end`` 是 ``"False"``。这里按 ultralytics 写进
    ONNX 之前的原始类型逐键还原：还原错了不会报错，只会在加载 engine 时让
    AutoBackend 拿到一个 ``imgsz="640"`` 之类的怪值，很难查。
    """
    if key in _INT_KEYS:
        try:
            return int(val)
        except (TypeError, ValueError):
            return val
    if key in _BOOL_KEYS:
        return str(val).strip().lower() in ("1", "true", "yes")
    if key in _LITERAL_KEYS:
        try:
            return ast.literal_eval(val)
        except (ValueError, SyntaxError):
            return val
    return val

def onnx_metadata(onnx_path) -> Dict[str, object]:
    """从源 ONNX 的 ``metadata_props`` 提取 ultralytics metadata。

    **继承而不是重新编造**：这份 metadata 是 W3a 导出时 ultralytics 从 ``best.pt``
    写进 ONNX 的，直接搬过来就保证 engine / ONNX / .pt 三者的 ``task``、``names``、
    ``imgsz``、``stride`` 完全一致。自己手写一份 ``names`` 迟早会和权重对不上，
    而那种错的表现是"mAP 突然变成 0"，不是报错。
    """
    import onnx  # 延迟导入：base 环境不一定装

    m = onnx.load(str(onnx_path), load_external_data=False)
    raw = {p.key: p.value for p in m.metadata_props}
    if not raw:
        return {}
    out: Dict[str, object] = {}
    for k, v in raw.items():
        out[k] = _coerce(k, v)
    return out

# --------------------------------------------------------------------------
# 推理封装
# --------------------------------------------------------------------------

def device_memory_bytes(engine) -> Optional[int]:
    """这份 engine 运行时需要的激活显存（字节），拿不到返回 None。

    用 TRT 自己报的数而不是 nvidia-smi 的读数：后者混着桌面、浏览器和 CUDA context
    的开销，两次跑对不上是常事，不能当可复现指标。``device_memory_size`` 在 TRT 10.1
    起废弃（直接访问会刷 DeprecationWarning），优先取 ``_v2``；v2 在新版里是方法，
    要带 profile 下标，所以两种形态都试一遍。
    """
    for attr in ("device_memory_size_v2", "device_memory_size"):
        v = getattr(engine, attr, None)
        if v is None:
            continue
        try:
            n = v(0) if callable(v) else v
        except Exception:  # noqa: BLE001
            continue
        if isinstance(n, int) and n > 0:
            return n
    return None


class TrtRunner:
    """把 ``.engine`` 跑起来的最小封装：反序列化 -> context -> 定形状 -> 异步执行 -> 拷回。

    用 torch 的 CUDA 张量当设备缓冲，省掉 pycuda（Windows 上要自己编，很容易卡住）。
    三处最容易错、且错了不报只出错数的地方：

    * **buffer 要一直持有到 synchronize 之后。** ``set_tensor_address`` 收的是裸
      ``data_ptr()``；张量一旦被 GC，TRT 就往已回收的显存里读写。所以 buffers 存在
      ``self._bufs`` 上，生命周期跟 runner 一样长。
    * **输出形状必须在 ``set_input_shape`` 之后再问 context。** 之前问拿到的是
      optimization profile 的 min/opt/max，不是本次输入对应的形状。
    * **``execute_async_v3`` 是异步的。** 不 ``synchronize`` 就拷回，读到的是上一帧
      的结果——数字看着还挺像，延迟也偏低，属于最难发现的一类错。
    """

    def __init__(self, engine_path, device: str = "cuda:0"):
        import tensorrt as trt  # noqa: PLC0415 - 延迟导入
        import torch  # noqa: PLC0415

        self.trt = trt
        self.torch = torch
        self.path = Path(engine_path)
        blob, meta = read_engine_file(self.path)
        self.meta = meta
        self.engine_bytes = len(blob)
        self.logger = trt.Logger(trt.Logger.ERROR)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(blob)
        if self.engine is None:
            raise RuntimeError(
                f"反序列化失败：{self.path}。engine 绑定构建时的 TRT 版本与 GPU 架构，"
                f"换任一个都得重建。")
        self.context = self.engine.create_execution_context()
        self.device = torch.device(device)
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.input_names = [n for n in names
                            if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        self.output_names = [n for n in names
                             if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
        if len(self.input_names) != 1:
            raise RuntimeError(f"期望 1 个输入张量，实际 {len(self.input_names)}：{self.input_names}")
        self.input_name = self.input_names[0]
        self.stream = torch.cuda.Stream(device=self.device)
        self._bufs: Dict[str, object] = {}

    # -- 只读信息 -----------------------------------------------------------
    @property
    def device_memory_bytes(self) -> Optional[int]:
        """这份 engine 运行时需要的激活显存，见模块级 ``device_memory_bytes``。"""
        return device_memory_bytes(self.engine)

    def io_info(self) -> Dict[str, List[Dict[str, object]]]:
        """各 I/O 张量的名字 / dtype / profile 形状，落进报告用来核对 engine 是不是想要的。"""
        out: Dict[str, List[Dict[str, object]]] = {"inputs": [], "outputs": []}
        for n in self.input_names + self.output_names:
            item = {
                "name": n,
                "dtype": str(self.trt.nptype(self.engine.get_tensor_dtype(n))),
                "shape": [int(x) for x in self.engine.get_tensor_shape(n)],
            }
            out["inputs" if n in self.input_names else "outputs"].append(item)
        return out

    # -- 执行 ---------------------------------------------------------------
    def _torch_dtype(self, name: str):
        npdt = self.trt.nptype(self.engine.get_tensor_dtype(name))
        return self.torch.from_numpy(np.empty((), dtype=npdt)).dtype

    def __call__(self, x):
        """``x``: NCHW float32 ndarray -> 输出张量列表（float32 ndarray，已拷回 CPU）。"""

        x = np.ascontiguousarray(x, dtype=np.float32)
        self.context.set_input_shape(self.input_name, tuple(x.shape))
        bufs: Dict[str, object] = {
            self.input_name: self.torch.from_numpy(x).to(self.device)
        }
        for n in self.output_names:  # 必须在 set_input_shape 之后问形状
            shape = tuple(int(v) for v in self.context.get_tensor_shape(n))
            bufs[n] = self.torch.empty(shape, dtype=self._torch_dtype(n), device=self.device)
        for n, t in bufs.items():
            self.context.set_tensor_address(n, int(t.data_ptr()))
        self._bufs = bufs  # 持有到同步之后，见类文档
        ok = self.context.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()
        if not ok:
            raise RuntimeError("execute_async_v3 返回 False")
        return [bufs[n].float().cpu().numpy() for n in self.output_names]

    def forward_device(self, t_in):
        """输入已在 GPU 上时直接跑，省一次 H2D。给延迟压测用，避免把拷贝时间算进推理。"""
        self.context.set_input_shape(self.input_name, tuple(t_in.shape))
        bufs: Dict[str, object] = {self.input_name: t_in}
        for n in self.output_names:
            shape = tuple(int(v) for v in self.context.get_tensor_shape(n))
            bufs[n] = self.torch.empty(shape, dtype=self._torch_dtype(n), device=self.device)
        for n, t in bufs.items():
            self.context.set_tensor_address(n, int(t.data_ptr()))
        self._bufs = bufs
        ok = self.context.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()
        if not ok:
            raise RuntimeError("execute_async_v3 返回 False")
        return bufs

    def close(self) -> None:
        self._bufs = {}
        self.context = None
        self.engine = None
        self.runtime = None

    def __enter__(self) -> "TrtRunner":
        return self

    def __exit__(self, *exc) -> None:
        self.close()