#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ONNX -> TensorRT engine，三档精度：FP32 / FP16 / INT8（熵校准）。

**为什么不用 trtexec**：pip 装的 ``tensorrt-cu13`` 只发 DLL——``nvinfer_10.dll``、
``nvinfer_plugin_10.dll``、``nvonnxparser_10.dll``，外加各 SM 的
``nvinfer_builder_resource_sm*_10.dll``（本机 10.16.1.11 合计约 2.3 GB），
**不带 ``trtexec.exe``**。那个只在 NVIDIA 官网的 GA zip 里，下载要登录勾 EULA，
没法脚本化、也没法在 CI 里复现。所以构建和压测都走 TensorRT Python API 自己实现。

顺带说一句 ``trtexec --calib``：它收的是**校准缓存文件**，不是图片目录。真要用真实
图片做 INT8 校准，本来就得自己实现 ``IInt8EntropyCalibrator2`` 把缓存生成出来
（deploy.md §8 已经预告了这一点）。下面的 ``EntropyCalibrator`` 就是这件事。

三个刻意的设计选择：

* **校准缓存的文件名里带校准清单的 sha256。** TRT 命中缓存就不再调 ``get_batch``，
  如果不把口径编进文件名，换一份校准集而缓存还在，会静默用旧动态范围建 engine——
  报出来的 INT8 掉点跟清单对不上，且没有任何地方会报警。
* **``profiling_verbosity=DETAILED``。** 多花一点构建时间，换来能统计"INT8 档到底
  有多少层真的落在 INT8 上"。只看端到端 mAP 判断量化生效没有，是间接证据；
  层精度直方图是直接证据，两者对不上时能立刻定位是量化没生效还是掉点另有原因。
* **device memory 用 TRT 自己报的数**，不用 nvidia-smi 的读数。``engine.device_memory_size``
  是这份 engine 运行时要的激活显存，可复现；nvidia-smi 混着桌面、浏览器、
  CUDA context 的开销，两次跑对不上是常事。压测那一侧另记 ``mem_get_info`` 的差值。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from induvision.deploy import (  # noqa: E402 - 需要先插好 sys.path
    load_sample_list,
    preprocess,
    read_image_bgr,
)
from induvision.trt_infer import (  # noqa: E402
    device_memory_bytes,
    onnx_metadata,
    write_engine_file,
)

#: 三档精度。fp32 是同机基线，fp16/int8 的掉点都相对它算。
PRECISIONS = ("fp32", "fp16", "int8")

#: 默认 workspace 上限（GB）。8 GB 显存的卡别贪：TRT 默认会把整卡报成可用
#: （本机实测 ``get_memory_pool_limit(WORKSPACE)`` = 8187.5 MB），真按这个建，
#: engine 的激活显存就没地方放了。
DEFAULT_WORKSPACE_GB = 4.0


def _git_head() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=15,
        )
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    """整文件 sha256。engine 是二进制产物，入库不了，只能靠指纹说明"是哪次构建的"。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def env_fingerprint() -> Dict[str, str]:
    """落进报告的环境指纹。换环境必须重建 engine——TRT engine 不跨版本、不跨 GPU 架构。"""
    fp = {"python": platform.python_version(), "platform": platform.platform()}
    for mod in ("numpy", "cv2", "torch", "ultralytics", "onnx", "tensorrt"):
        try:
            m = __import__(mod)
            fp[mod] = getattr(m, "__version__", "?")
        except Exception:  # noqa: BLE001
            fp[mod] = "absent"
    try:
        import torch  # noqa: PLC0415

        fp["cuda"] = torch.version.cuda or "none"
        if torch.cuda.is_available():
            fp["gpu"] = torch.cuda.get_device_name(0)
            cap = torch.cuda.get_device_capability(0)
            fp["gpu_sm"] = f"sm_{cap[0]}{cap[1]}"
            fp["gpu_mem_mb"] = str(round(torch.cuda.get_device_properties(0).total_memory / 1024**2))
        else:
            fp["gpu"] = "cpu"
    except Exception:  # noqa: BLE001
        pass
    try:
        import tensorrt as trt  # noqa: PLC0415

        fp["trt_builder_version"] = trt.__version__
    except Exception:  # noqa: BLE001
        pass
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=driver_version",
                              "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20)
        if out.returncode == 0:
            fp["driver"] = out.stdout.strip().splitlines()[0]
    except Exception:  # noqa: BLE001
        pass
    return fp


def load_calib_inputs(calib_list: Path, imgsz: int, limit: int,
                      batch_size: int) -> Tuple[List[np.ndarray], Dict[str, object]]:
    """按固定清单读图 + 前处理，切成 ``batch_size`` 一批的 NCHW float32 数组。

    前处理直接复用 ``induvision.deploy.preprocess``（W3a 逐行对齐过 ultralytics 的
    LetterBox），**不能在这里另写一份**：校准看到的输入分布必须和推理时一模一样，
    否则动态范围是对着另一个分布统计的。

    读不出来的图跳过并记进 ``meta["skipped"]``——跳过会改变实际喂进校准器的图，
    所以要留痕，不能装作 500 张全用上了。
    """
    paths, meta = load_sample_list(calib_list, strict=True)
    if limit and limit > 0:
        paths = paths[:limit]
    xs: List[np.ndarray] = []
    skipped: List[str] = []
    for p in paths:
        img = read_image_bgr(p)
        if img is None:
            skipped.append(str(p))
            continue
        x, _lb = preprocess(img, imgsz=imgsz)
        xs.append(x)
    if skipped:
        print(f"[warn ] {len(skipped)} 张校准图读不出来，已跳过：{', '.join(skipped[:3])}")
    if not xs:
        raise ValueError(f"{calib_list} 里没有一张图能读出来，无法校准")
    batches = [np.ascontiguousarray(np.stack(xs[i:i + batch_size]), dtype=np.float32)
               for i in range(0, len(xs), batch_size)]
    meta = dict(meta)
    meta["skipped"] = skipped
    meta["images_used"] = len(xs)
    meta["batches"] = len(batches)
    meta["batch_size"] = batch_size
    return batches, meta


#: 隐式量化的两种校准算法。键是我们在报告/缓存名里用的短名，值是 TRT
#: ``CalibrationAlgoType`` 里的枚举名——两种算法共用同一个基类 ``IInt8Calibrator``，
#: 靠 ``get_algorithm()`` 的返回值区分，不是靠继承不同的类。
#:
#: 为什么默认是 minmax 而不是通常推荐的 entropy——本机（RTX 4060 Laptop / SM89 +
#: TRT 10.16.1）实测：用 ``IInt8EntropyCalibrator2`` 构建这个 YOLOv8s-seg 一定失败，
#: 每次都停在同一个融合节点上，
#:   [E] Could not find any implementation for node
#:       /model.22/proto/cv3/conv/Conv + PWN(Sigmoid, Mul)
#: 试过三条路都不管用：workspace 4 GB -> 6 GB；打开 ``BuilderFlag.FP16`` 当退路；
#: 把 proto/cv3 那三层用 ``layer.precision=FP16`` + ``PREFER_PRECISION_CONSTRAINTS``
#: 钉死。换成 ``IInt8Calibrator``（MinMax）后同一个 ONNX 一次构建通过。
#: 差别在动态范围本身：熵校准会为最小化 KL 散度把范围收窄，收窄到那个"INT8 入、
#: FP32 出（Mul 就是网络输出 output1）"的融合节点上就一条可用 tactic 都不剩。
#: MinMax 取绝对最大值，范围宽，节点有解。代价是 MinMax 对离群值敏感，掉点通常
#: 比熵校准大——所以第 2 层的 mAP 判据必须照实报，不能因为构建过了就当没事。
#: 比上面那个更隐蔽、白白浪费三轮构建的坑：``IInt8Calibrator.get_algorithm`` 是
#: **纯虚函数**，Python 绑定不会给默认实现。只继承基类、只实现 get_batch /
#: read_calibration_cache / write_calibration_cache 那一套是不够的，一启动就
#:   [ERROR] Tried to call pure virtual function "IInt8Calibrator::get_algorithm"
#: 然后 builder 拿不到任何动态范围，报出来的却是上面那句"找不到 implementation"
#: ——错误信息完全指错方向。ultralytics 自己的 EngineCalibrator 就重写了这个方法。
CALIB_ALGOS = {"minmax": "MINMAX_CALIBRATION", "entropy": "ENTROPY_CALIBRATION_2"}
#: 两种算法共用的基类名（TRT 11 起隐式量化被移除，届时这个名字会不存在）。
CALIB_BASE = "IInt8Calibrator"


def make_calibrator(trt_mod, batches: Sequence[np.ndarray], batch_size: int,
                    cache_path: Path, device: str, algo: str = "minmax"):
    """造一个隐式量化校准器（继承 ``IInt8Calibrator``，算法由 ``get_algorithm()`` 决定）。

    用 torch 的 CUDA 张量当设备内存，省掉 pycuda 依赖（Windows 上 pycuda 要自己编，
    很容易卡住）。两个要点：

    * ``self._cur`` 必须一直持有当前批的张量。返回的是裸 ``data_ptr()``，
      Python 这边一旦把张量释放，TRT 读到的就是已回收的显存——不会报错，
      只会悄悄校准出一堆错的动态范围。
    * 每批只留一份、下一批来了才放掉上一批。500 张全驻留显存是 2.46 GB，
      8 GB 的卡还要同时装 builder 的 workspace，攒着必炸。
    """

    if algo not in CALIB_ALGOS:
        raise ValueError(f"未知校准算法：{algo}（可选 {sorted(CALIB_ALGOS)}）")
    enum_name = CALIB_ALGOS[algo]
    algo_type = getattr(getattr(trt_mod, "CalibrationAlgoType", None), enum_name, None)
    if algo_type is None:
        raise RuntimeError(f"这个 TensorRT 版本没有 CalibrationAlgoType.{enum_name}，"
                           f"隐式量化走不通；改用 fp16，或上显式量化（QDQ）")
    base = getattr(trt_mod, CALIB_BASE, None)
    if base is None:
        raise RuntimeError(f"这个 TensorRT 版本没有 {CALIB_BASE}（TRT 11 起隐式量化"
                           f"校准器被移除）；只能走显式量化 QDQ，或改用 fp16")

    class _Calib(base):
        def __init__(self) -> None:
            super().__init__()
            self._batches = list(batches)
            self._bs = batch_size
            self._cache = Path(cache_path)
            self._dev = device
            self._i = 0
            self._cur = None
            self.served = 0
            self.cache_hit = False
            self.cache_written = False
            self.algo = algo

        def get_algorithm(self):
            """纯虚函数，必须重写，否则 builder 根本跑不起来——见 CALIB_ALGOS 注释。"""
            return algo_type

        def get_batch_size(self) -> int:  # TRT 10 已废弃（恒为显式 batch），留着兼容
            return self._bs

        def get_batch(self, names):
            if self._i >= len(self._batches):
                return None
            import torch  # noqa: PLC0415

            arr = self._batches[self._i]
            self._i += 1
            self._cur = None  # 先放掉上一批，见类文档
            t = torch.from_numpy(arr).to(self._dev)
            torch.cuda.synchronize()
            self._cur = t
            self.served += 1
            if self.served % 10 == 0:
                print(f"[calib] {self.served}/{len(self._batches)} 批已喂")
            return [int(t.data_ptr())]

        def read_calibration_cache(self):
            if self._cache.is_file():
                self.cache_hit = True
                print(f"[calib] 命中缓存 {self._cache.name}，跳过 {len(self._batches)} 批的统计")
                return self._cache.read_bytes()
            return None

        def write_calibration_cache(self, cache) -> None:
            self._cache.parent.mkdir(parents=True, exist_ok=True)
            self._cache.write_bytes(bytes(cache))
            self.cache_written = True
            print(f"[calib] 缓存已写入 {self._cache}（{self._cache.stat().st_size} 字节）")

    return _Calib()


def pin_layers_fp16(network, trt_mod, substr: str) -> List[str]:
    """把名字含 ``substr`` 的层钉到 FP16，返回命中的层名。

    为什么要这个函数：本机（SM89 + TRT 10.16）隐式 INT8 构建一直失败在同一个
    融合节点上——

        [E] Could not find any implementation for node
            /model.22/proto/cv3/conv/Conv + PWN(Sigmoid, Mul)

    那是 mask proto 分支最后一层 1x1 卷积（256->32, 160x160）融了 SiLU，而
    SiLU 的输出 ``proto/cv3/act/Mul_output_0`` 就是网络输出 ``output1``。
    校准缓存里查过：``proto/cv3/conv`` 与 ``proto/cv3/act/Sigmoid`` 都有动态范围，
    唯独这个 Mul（网络输出）没有——网络输出默认要 FP32，于是这个融合节点被要求
    "INT8 入、FP32 出"，tactic 优化器在它的代价模型里一条实现都选不出来。

    单纯把 ``BuilderFlag.FP16`` 打开（"fallback"）不管用：那只是往候选集里加了
    Half，优化器仍然先去试 INT8 然后在同一个地方炸。钉 ``layer.precision`` 配
    ``PREFER_PRECISION_CONSTRAINTS`` 才是把 INT8 从这些层的候选里真正拿掉。
    用 PREFER 而不是 OBEY：OBEY 在钉不上时直接构建失败，等于把一条能走的路
    也堵死；PREFER 会在无法满足时退回自动选择并给性能告警。
    """
    hit: List[str] = []
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        name = layer.name or ""
        if substr and substr in name:
            layer.precision = trt_mod.float16
            for j in range(layer.num_outputs):
                layer.set_output_type(j, trt_mod.float16)
            hit.append(name)
    return hit


#: ultralytics 8.4.183 隐式 INT8 配方里的两个层名标记（utils/export/engine.py:433-465）。
HEAD_CONV_FP16_SUFFIX = ".2/Conv"
HEAD_DFL_MARK = "/dfl/"
#: 头部例外的两种模式：ultralytics = 按下面那三条规则钉精度；none = 不钉（对照用）。
INT8_HEAD_MODES = ("ultralytics", "none")


def constrain_head_precisions(network, trt_mod) -> List[Dict[str, str]]:
    """把检测/分割头里的关键层挡在 INT8 之外，返回命中明细。

    这是本机 INT8 构建从"必失败"到"能过"的那一步。配方不是拍脑袋想的，
    是照抄 ultralytics 8.4.183 ``utils/export/engine.py``——它在同一台机器、同一个
    YOLOv8s-seg 上构建成功过（产物 13,331,868 字节的 engine 就在本地，这是硬证据），
    而我们自己那三轮全炸在同一个节点上。三条规则：

    1. 头（层名里 ``/model.{最大索引}/``）中名字带 ``sigmoid`` 的 ACTIVATION 层 -> FP32。
       TRT 会把每个 SiLU 拆成 Sigmoid + Mul，所以**必须按头作用域筛**：全网筛会把
       backbone/neck 的 SiLU 一起挡掉，INT8 的加速就没了。留 FP32 是为了保住
       置信度的标定（对齐 OpenVINO 的 IgnoredScope 做法）。
    2. 头里的 CONVOLUTION、且层名以 ``.2/Conv`` 结尾（各分支最后一层输出卷积）
       或含 ``/dfl/`` -> FP16。
    3. 两类都设 ``layer.precision`` + 每个输出的 ``set_output_type``，然后打开
       ``OBEY_PRECISION_CONSTRAINTS``。

    为什么是 OBEY 而不是我们上一轮试的 PREFER：PREFER 在钉不上时会退回自动选择，
    约束等于没生效，那个 Conv+SiLU 融合节点又被塞回 INT8 候选，构建照样炸在
    "Could not find any implementation"。OBEY 钉不上就直接失败——对这一档来说，
    失败得明白比"看着成功实际没量化"好。

    ``pin_layers_fp16`` 是这条路线上失败的那次尝试，留着当对照，别删。
    """
    names = [network.get_layer(i).name or "" for i in range(network.num_layers)]
    idx = [int(m.group(1)) for n in names if (m := re.search(r"/model\.(\d+)/", n))]
    head = f"/model.{max(idx)}/" if idx else "/"
    hit: List[Dict[str, str]] = []
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        name = layer.name or ""
        if head not in name:
            continue
        if layer.type == trt_mod.LayerType.ACTIVATION and "sigmoid" in name.lower():
            dtype, kind, tag = trt_mod.float32, "activation-sigmoid", "float32"
        elif (layer.type == trt_mod.LayerType.CONVOLUTION
              and (name.endswith(HEAD_CONV_FP16_SUFFIX) or HEAD_DFL_MARK in name)):
            dtype, kind, tag = trt_mod.float16, "conv-output", "float16"
        else:
            continue
        layer.precision = dtype
        for j in range(layer.num_outputs):
            layer.set_output_type(j, dtype)
        hit.append({"name": name, "kind": kind, "precision": tag})
    shown = ", ".join(h["name"] for h in hit[:6])
    print(f"[build] int8 头部例外命中 {len(hit)} 层（head={head!r}）：{shown}"
          f"{' ...' if len(hit) > 6 else ''}")
    if not hit:
        print("[warn ] 一层都没命中：层名里没有 /model.N/ 形式的头前缀，"
              "接下去大概率还是会炸在 proto/cv3")
    return hit


def inspect_layers(serialized: bytes, trt_mod) -> Dict[str, object]:
    """统计 engine 各层**输出**的数据格式与层类型，返回 ``{output_formats, layer_types}``。

    这是"量化到底生效没有"的直接证据：只看端到端 mAP 是间接的，掉点了分不清是
    量化没生效还是别的原因；看格式分布能一眼确认。本机实测 FP16 档 183 层里
    151 层输出 ``Half``、32 层 TRT 判定仍走 ``Float``。

    两个坑，都实测过：

    * **inspector 的 JSON 里没有 ``Precision`` 这个键。** 一开始按 ``lay["Precision"]``
      取，两档都返回 265/183 个 ``unspecified``，看着像"profiling verbosity 没生效"，
      其实是键名猜错了。精度藏在每层 ``Outputs[*]["Format/Datatype"]`` 里
      （``Float`` / ``Half`` / ``Int8`` ...）。
    * **必须在构建时设 ``profiling_verbosity = DETAILED``**，否则 inspector 只给层名。

    拿不到就返回空 dict，不因此判定构建失败——这是诊断信息，不是判据。
    """
    try:
        runtime = trt_mod.Runtime(trt_mod.Logger(trt_mod.Logger.ERROR))
        engine = runtime.deserialize_cuda_engine(serialized)
        if engine is None:
            return {"output_formats": {}, "layer_types": {}}
        insp = engine.create_engine_inspector()
        raw = insp.get_engine_information(trt_mod.LayerInformationFormat.JSON)
        info = json.loads(raw) if isinstance(raw, str) else {}
        layers = [l for l in (info.get("Layers") or []) if isinstance(l, dict)]
        fmt: Counter = Counter()
        for lay in layers:
            for o in (lay.get("Outputs") or []):
                if isinstance(o, dict):
                    fmt[str(o.get("Format/Datatype") or "unspecified")] += 1
        types: Counter = Counter(str(lay.get("LayerType") or "?") for lay in layers)
        return {
            "num_layers": len(layers),
            "output_formats": dict(sorted(fmt.items())),
            "layer_types": dict(types.most_common()),
        }
    except Exception as e:  # noqa: BLE001
        print(f"[warn ] 取不到层信息：{e}")
        return {"output_formats": {}, "layer_types": {}}


def build_engine(onnx_path: Path, out_path: Path, prec: str, *, imgsz: int = 640,
                 batch: int = 1, workspace_gb: float = DEFAULT_WORKSPACE_GB,
                 calib_batches: Optional[Sequence[np.ndarray]] = None,
                 calib_cache: Optional[Path] = None, calib_batch_size: int = 8,
                 int8_fp16_fallback: bool = True,
                 int8_pin: Optional[str] = None, int8_calib: str = "minmax",
                 int8_head: str = "ultralytics", tf32: bool = True,
                 device: str = "cuda:0", verbose: bool = False) -> Dict[str, object]:
    """构建一档 engine 并落盘，返回可入库的报告 dict。构建失败抛 RuntimeError。"""

    if int8_pin and int8_head != "none":
        # 一个要 PREFER、一个要 OBEY，同时开 TRT 行为未定义；而且两者是同一个问题的
        # 两种解法（失败的那种和能过的那种），混用了掉点数就没法归因。
        raise ValueError("--int8-pin（PREFER_PRECISION_CONSTRAINTS）与 int8_head"
                         "（OBEY_PRECISION_CONSTRAINTS）不能同时用，二选一")
    if int8_head not in INT8_HEAD_MODES:
        raise ValueError(f"未知 int8_head：{int8_head}（可选 {list(INT8_HEAD_MODES)}）")

    import tensorrt as trt  # noqa: PLC0415

    level = trt.Logger.VERBOSE if verbose else trt.Logger.WARNING
    logger = trt.Logger(level)
    builder = trt.Builder(logger)
    network = builder.create_network(0)  # TRT 10 只有显式 batch，不再传 EXPLICIT_BATCH
    parser = trt.OnnxParser(network, logger)

    data = Path(onnx_path).read_bytes()
    if not parser.parse(data):
        errs = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise RuntimeError(f"ONNX 解析失败（{len(errs)} 条）：" + "; ".join(errs[:5]))

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1 << 30)))
    # 层精度直方图要 DETAILED 才拿得到，见 layer_precision_histogram
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED

    nin = network.num_inputs
    if nin != 1:
        raise RuntimeError(f"期望 ONNX 只有 1 个输入，实际 {nin} 个")
    in_name = network.get_input(0).name
    shape = (batch, 3, imgsz, imgsz)
    profile = builder.create_optimization_profile()
    # 端侧部署输入是定死的，min=opt=max 让 TRT 只为一组形状选 tactic，
    # 比给一个动态范围再让它折中要快，也更可复现
    profile.set_shape(in_name, shape, shape, shape)
    config.add_optimization_profile(profile)

    calib = None
    head_hits: List[Dict[str, str]] = []
    flags: List[str] = []
    if not tf32:
        # TRT 默认把 BuilderFlag.TF32 打开：所谓 FP32 档其实跑的是 TF32
        # （10-bit 尾数），逐张量 rel_l2 会停在 1e-3 量级，看着像对齐没做干净，
        # 实际是数值格式本身的机器精度。要做"真 FP32"对照必须显式清掉这个 flag。
        config.clear_flag(trt.BuilderFlag.TF32)
        flags.append("no-TF32")
    if prec == "fp16":
        if not builder.platform_has_fast_fp16:
            print("[warn ] 本机 platform_has_fast_fp16=False，FP16 档可能没有收益")
        config.set_flag(trt.BuilderFlag.FP16)
        flags.append("FP16")
    elif prec == "int8":
        if not builder.platform_has_fast_int8:
            print("[warn ] 本机 platform_has_fast_int8=False，INT8 档可能没有收益")
        if not calib_batches:
            raise RuntimeError("INT8 需要校准批次，calib_batches 为空")
        config.set_flag(trt.BuilderFlag.INT8)
        flags.append("INT8")
        if int8_fp16_fallback:
            # 不开这个开关，本机 INT8 档构建直接返回 None（实测），TRT 只留一行：
            #   [E] IBuilder::buildSerializedNetwork: Error Code 10: Internal Error
            #       (Could not find any implementation for node
            #        /model.22/proto/cv3/conv/Conv + PWN(Sigmoid, Mul))
            # 那是 mask proto 分支最后一层 32 通道 160x160 卷积融了 SiLU。
            # 隐式 INT8 的本意是"没有 INT8 tactic 的节点自动回退"，但这个融合节点
            # 在 SM89 上连一条可用实现都选不出来。同时打开 FP16 就是给它一条
            # Half 的退路——混合精度量化本来就是这么用的，代价是这一档不再是
            # 纯 INT8，所以层格式直方图必须照实报有多少层真落在 Int8 上。
            config.set_flag(trt.BuilderFlag.FP16)
            flags.append("FP16(fallback)")
        head_hits = constrain_head_precisions(network, trt) if int8_head == "ultralytics" else []
        if head_hits:
            config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
            flags.append("OBEY_PRECISION_CONSTRAINTS")
        pinned: List[str] = []
        if int8_pin:
            # 见 pin_layers_fp16 的文档：这条是绕过那个融合节点的关键
            config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
            flags.append("PREFER_PRECISION_CONSTRAINTS")
            pinned = pin_layers_fp16(network, trt, int8_pin)
            print(f"[build] int8 钉 FP16 命中 {len(pinned)} 层（substr={int8_pin!r}）："
                  f"{', '.join(pinned) or '无'}")
            if not pinned:
                print(f"[warn ] 没有任何层名含 {int8_pin!r}，钉精度等于没生效")
        calib = make_calibrator(trt, calib_batches, calib_batch_size,
                                calib_cache or out_path.with_suffix(".calib"), device,
                                algo=int8_calib)
        config.int8_calibrator = calib
        flags.append(f"calib={int8_calib}")
    elif prec != "fp32":
        raise ValueError(f"未知精度档：{prec}")

    print(f"[build] {prec} 开始构建（workspace={workspace_gb} GB, shape={shape}, "
          f"flags={','.join(flags) or 'FP32'}，TF32={tf32}）")
    t0 = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    dt = time.perf_counter() - t0
    if serialized is None:
        raise RuntimeError(f"{prec} 构建返回 None（TensorRT 日志见上方）")
    blob = bytes(serialized)

    # metadata 从源 ONNX 继承，按 ultralytics 的 header 格式前置写进文件：
    # 这样 YOLO(x.engine) 能直接拿去做 mAP，不用让 AutoBackend 靠输出形状猜 task/nc。
    meta = onnx_metadata(onnx_path)
    file_bytes = write_engine_file(out_path, blob, meta)

    runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
    engine = runtime.deserialize_cuda_engine(blob)
    dev_mem = n_layers = None
    io_names: Optional[List[str]] = None
    if engine is not None:
        n_layers = engine.num_layers
        io_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
        dev_mem = device_memory_bytes(engine)

    rep: Dict[str, object] = {
        "precision": prec,
        "engine": str(out_path),
        "engine_bytes": len(blob),
        "engine_mib": round(len(blob) / 1024**2, 2),
        "engine_sha256": hashlib.sha256(blob).hexdigest(),
        "file_bytes": file_bytes,
        "header_bytes": file_bytes - len(blob),
        "metadata_keys": sorted(meta) if meta else [],
        "build_seconds": round(dt, 1),
        "onnx": str(onnx_path),
        "onnx_bytes": Path(onnx_path).stat().st_size,
        "shape": list(shape),
        "builder_flags": flags,
        "tf32": bool(tf32),
        "pinned_layers": (pinned if prec == "int8" else []),
        "int8_pin": (int8_pin if prec == "int8" else None),
        "input_name": in_name,
        "workspace_gb": workspace_gb,
        "trt_version": trt.__version__,
        "num_layers": n_layers,
        "io_tensors": io_names,
        "device_memory_mib": (round(dev_mem / 1024**2, 2) if dev_mem else None),
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    if calib is not None:
        rep["calibration"] = {
            "algorithm": CALIB_ALGOS.get(str(int8_calib), str(int8_calib)),
            "calibrator_base": CALIB_BASE,
            "calib_algo": str(int8_calib),
            "cache": str(calib_cache) if calib_cache else None,
            "cache_hit": bool(calib.cache_hit),
            "cache_written": bool(calib.cache_written),
            "batches_served": int(calib.served),
            "batch_size": calib_batch_size,
        }
    if prec == "int8":
        rep["head_constraints"] = {"mode": int8_head, "count": len(head_hits),
                                   "layers": head_hits}
    insp = inspect_layers(blob, trt)
    rep["output_formats"] = insp.get("output_formats", {})
    rep["layer_types"] = insp.get("layer_types", {})
    print(f"[build] {prec} 完成：engine {rep['engine_mib']} MiB"
          f"（含 header 落盘 {round(file_bytes / 1024**2, 2)} MiB），"
          f"构建 {rep['build_seconds']} s，层数 {n_layers}，"
          f"激活显存 {rep['device_memory_mib']} MiB")
    if rep["output_formats"]:
        tot = sum(rep["output_formats"].values())
        print(f"[build] {prec} 层输出格式：" +
              ", ".join(f"{k}={v}({round(100.0 * v / tot, 1)}%)"
                        for k, v in rep["output_formats"].items()))
    if not meta:
        print(f"[warn ] {prec} 没从 ONNX 拿到 metadata，engine 将没有 ultralytics header，"
              f"YOLO(x.engine) 可能识别不出 task/names")
    return rep


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="ONNX -> TensorRT engine（FP32/FP16/INT8）")
    ap.add_argument("--onnx", default="weights/baseline15_fp32.onnx")
    ap.add_argument("--out-dir", default="weights", help="engine 落盘目录（gitignored）")
    ap.add_argument("--name", default=None, help="run 名，默认从 onnx 文件名反推")
    ap.add_argument("--prec", nargs="+", default=list(PRECISIONS), choices=list(PRECISIONS))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--workspace-gb", type=float, default=DEFAULT_WORKSPACE_GB)
    ap.add_argument("--calib-list", default="configs/calib_sample.json")
    ap.add_argument("--calib-n", type=int, default=500, help="0 = 用清单全部")
    ap.add_argument("--calib-batch", type=int, default=8)
    ap.add_argument("--calib-cache-dir", default="runs/logs", help="校准缓存目录（gitignored）")
    ap.add_argument("--int8-no-fp16-fallback", action="store_true",
                    help="INT8 档不同时打开 FP16；本机这样建会失败，见 build_engine 里的注释")
    ap.add_argument("--int8-calib", default="minmax", choices=sorted(CALIB_ALGOS),
                    help="隐式量化的校准算法；entropy 在本机构建必失败，见 CALIB_ALGOS")
    ap.add_argument("--int8-pin", default="",
                    help="把层名含该子串的层钉到 FP16（空串=不钉）。这是失败的那条路："
                         "PREFER 钉不住，本机实测照样炸在 proto/cv3，见 pin_layers_fp16")
    ap.add_argument("--int8-head", default="ultralytics", choices=list(INT8_HEAD_MODES),
                    help="ultralytics=按 ultralytics 配方把头部 sigmoid 钉 FP32、输出卷积钉 FP16"
                         "（本机唯一能构建成功的组合，见 constrain_head_precisions）；none=不钉，作对照")
    ap.add_argument("--no-calib-cache", action="store_true",
                    help="忽略已有缓存，强制重新统计动态范围")
    ap.add_argument("--device", default="0", help="校准用的 CUDA 设备")
    ap.add_argument("--json-out", default=None, help="默认 runs/logs/trt_build_<name>.json")
    ap.add_argument("--no-tf32", action="store_true",
                    help="清掉 BuilderFlag.TF32，让 FP32 档跑真 FP32（默认按 TRT 惯例开 TF32）")
    ap.add_argument("--verbose", action="store_true", help="打 TensorRT VERBOSE 日志")
    ap.add_argument("--dry-run", action="store_true", help="只打印将做什么，不构建")
    args = ap.parse_args(argv)

    def _p(s: str) -> Path:
        q = Path(s)
        return q if q.is_absolute() else (REPO_ROOT / q)

    onnx = _p(args.onnx)
    if not onnx.is_file():
        print(f"[err  ] 缺 ONNX：{onnx}（先跑 tools/export_onnx.py）")
        return 2
    bad = [p for p in args.prec if p not in PRECISIONS]
    if bad:
        print(f"[err  ] 未知精度档：{bad}")
        return 2

    name = args.name or onnx.stem.replace("_fp32", "").replace("_onnx", "")
    out_dir = _p(args.out_dir)
    dev = f"cuda:{args.device}" if str(args.device).isdigit() else str(args.device)

    calib_list = _p(args.calib_list)
    need_int8 = "int8" in args.prec
    if need_int8 and not calib_list.is_file():
        print(f"[err  ] INT8 需要校准清单：{calib_list} 不存在"
              f"（先跑 tools/make_calib_list.py）")
        return 2

    if args.dry_run:
        print(f"[dry  ] name={name} onnx={onnx} tf32={not args.no_tf32}")
        for p in args.prec:
            print(f"[dry  ]   -> {out_dir / f'{name}_trt_{p}.engine'}")
        if need_int8:
            print(f"[dry  ] calib_list={calib_list} n={args.calib_n} batch={args.calib_batch}")
            print(f"[dry  ] calib_algo={args.int8_calib} head={args.int8_head} "
                  f"pin={args.int8_pin or '无'}")
            print(f"[dry  ] calib_cache_dir={_p(args.calib_cache_dir)}")
        return 0

    try:
        import tensorrt as trt  # noqa: PLC0415
    except Exception as e:  # noqa: BLE001
        print(f"[err  ] import tensorrt 失败：{e}")
        return 3
    try:
        import torch  # noqa: PLC0415
        if not torch.cuda.is_available():
            print("[err  ] torch.cuda.is_available()=False，TensorRT 构建需要 CUDA 设备")
            return 3
    except Exception as e:  # noqa: BLE001
        print(f"[err  ] import torch 失败：{e}")
        return 3

    print(f"[info ] TensorRT {trt.__version__} | onnx={onnx} ({onnx.stat().st_size/1024**2:.1f} MiB)")

    calib_batches: Optional[List[np.ndarray]] = None
    calib_meta: Dict[str, object] = {}
    calib_cache: Optional[Path] = None
    if need_int8:
        try:
            calib_batches, calib_meta = load_calib_inputs(
                calib_list, args.imgsz, args.calib_n, args.calib_batch)
        except (ValueError, FileNotFoundError) as e:
            print(f"[err  ] 校准集加载失败：{e}")
            return 2
        # 缓存名里带清单指纹 + 校准算法：换清单不该误命中旧缓存，换算法更不该。
        # 两种算法统计的是同一个 ONNX 的同一批图，光看文件名分不出来；真复用了，
        # TRT 不会报错，只会拿另一套动态范围去量化，掉点数直接对不上账。
        fp = str(calib_meta.get("sha256") or "nofp")
        calib_cache = _p(args.calib_cache_dir) / \
            f"trt_calib_{name}_{args.int8_calib}_{fp}.cache"
        if args.no_calib_cache and calib_cache.is_file():
            calib_cache.unlink()
            print(f"[info ] --no-calib-cache：已删除旧缓存 {calib_cache.name}")
        print(f"[info ] 校准集：{calib_meta.get('images_used')} 张 / "
              f"{calib_meta['batches']} 批 x {args.calib_batch}，sha256={fp}")

    reports: List[Dict[str, object]] = []
    failed: List[str] = []
    for prec in args.prec:
        out_path = out_dir / f"{name}_trt_{prec}.engine"
        try:
            rep = build_engine(
                onnx, out_path, prec, imgsz=args.imgsz, batch=args.batch,
                workspace_gb=args.workspace_gb, calib_batches=calib_batches,
                calib_cache=calib_cache, calib_batch_size=args.calib_batch,
                int8_fp16_fallback=not args.int8_no_fp16_fallback,
                int8_pin=(args.int8_pin or None), int8_calib=args.int8_calib,
                int8_head=args.int8_head, tf32=not args.no_tf32,
                device=dev, verbose=args.verbose)
            reports.append(rep)
        except Exception as e:  # noqa: BLE001
            print(f"[err  ] {prec} 构建失败：{type(e).__name__}: {e}")
            failed.append(prec)

    payload = {
        "name": name,
        "git_head": _git_head(),
        "env": env_fingerprint(),
        "onnx_sha256": file_sha256(onnx),
        "calib_sample": {k: calib_meta.get(k) for k in
                         ("path", "split", "count", "source_total", "sha256",
                          "sampling", "images_used", "batches", "batch_size", "skipped")}
        if calib_meta else None,
        "engines": reports,
        "failed": failed,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    jout = _p(args.json_out) if args.json_out else \
        _p("runs/logs") / f"trt_build_{name}.json"
    jout.parent.mkdir(parents=True, exist_ok=True)
    jout.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8", newline="\n")
    print(f"[ok   ] 报告已写入 {jout}")
    if failed:
        print(f"[err  ] 失败档位：{', '.join(failed)}")
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())