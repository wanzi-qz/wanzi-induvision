"""induvision/trt_infer.py 的单测。

覆盖的是"不需要 GPU 也不需要 tensorrt"的那一半：``.engine`` 文件的 header 编解码、
ONNX metadata 的类型还原、engine 自报激活显存的取值优先级。``TrtRunner`` 真跑起来
要 CUDA，那部分归 ``tools/bench_trt.py`` 的实测报告管，不在 CI 里测。

盯得最紧的是 header：engine 是我们自己用 TensorRT Python API 建的，文件里本来没有
ultralytics 期望的 metadata。header 写歪一个字节，``YOLO(x.engine)`` 就认不出
task/nc/names，第 2 层的 mAP 直接变 0——而它不会报错。
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from induvision import trt_infer as ti


# --------------------------------------------------------------------------
# header 编解码
# --------------------------------------------------------------------------

def test_encode_header_uses_ultralytics_layout():
    """4 字节小端有符号长度 + JSON，和 ultralytics/utils/export/engine.py 一致。"""
    meta = {"task": "segment", "nc": 15}
    raw = json.dumps(meta)
    head = ti.encode_header(meta)
    assert head[: ti.HEADER_LEN_BYTES] == len(raw).to_bytes(4, byteorder="little", signed=True)
    assert head[ti.HEADER_LEN_BYTES:].decode() == raw


def test_encode_header_empty_without_meta():
    assert ti.encode_header(None) == b""
    assert ti.encode_header({}) == b""


def test_decode_header_returns_offset_and_meta():
    meta = {"task": "segment", "imgsz": [640, 640], "stride": 32}
    blob = b"\xde\xad\xbe\xefenginebytes"
    head = ti.encode_header(meta)
    off, got = ti.decode_header(head + blob)
    assert off == len(head)
    assert got == meta
    assert (head + blob)[off:] == blob


def test_decode_header_absent_on_raw_engine_bytes():
    """没有 header 的 engine（例如 trtexec 直接产出的）要判成 (0, {})，不能抛。"""
    blob = bytes(range(256)) * 8
    assert ti.decode_header(blob) == (0, {})


@pytest.mark.parametrize("n", [0, -1, 10**6, 2**31 - 1])
def test_decode_header_rejects_out_of_range_length(n):
    blob = n.to_bytes(4, byteorder="little", signed=True) + b'{"task": "segment"}'
    assert ti.decode_header(blob) == (0, {})


def test_decode_header_rejects_non_dict_json():
    raw = json.dumps([1, 2, 3])
    blob = len(raw).to_bytes(4, byteorder="little", signed=True) + raw.encode()
    assert ti.decode_header(blob) == (0, {})


def test_decode_header_rejects_truncated_json():
    raw = json.dumps({"task": "segment"})
    blob = len(raw).to_bytes(4, byteorder="little", signed=True) + raw[:5].encode()
    assert ti.decode_header(blob) == (0, {})


def test_decode_header_short_blob():
    assert ti.decode_header(b"") == (0, {})
    assert ti.decode_header(b"\x01\x02") == (0, {})


def test_write_read_engine_file_roundtrip(tmp_path: Path):
    p = tmp_path / "sub" / "x.engine"
    blob = b"\x00\x01\x02\x03" * 64
    meta = {"task": "segment", "names": {0: "bottle"}}
    n = ti.write_engine_file(p, blob, meta)
    assert p.is_file()
    assert n == p.stat().st_size
    got_blob, got_meta = ti.read_engine_file(p)
    assert got_blob == blob
    assert got_meta["task"] == "segment"


def test_write_engine_file_without_meta_is_pure_blob(tmp_path: Path):
    p = tmp_path / "y.engine"
    blob = b"ENGINE"
    assert ti.write_engine_file(p, blob, None) == len(blob)
    got_blob, got_meta = ti.read_engine_file(p)
    assert got_blob == blob
    assert got_meta == {}


def test_header_roundtrip_turns_int_keys_into_str():
    """已知且刻意保留的差异：JSON 没有 int 键。

    ``names`` 从 ONNX metadata 还原出来是 ``{0: 'bottle'}``，过一遍 header 就变成
    ``{'0': 'bottle'}``。ultralytics 自己写 ``.engine`` 也是这个结果，AutoBackend
    用 ``{int(k): v for ...}`` 兜住了，所以不影响加载。这条把行为钉住：谁哪天
    "顺手修好"了它，反而会和 ultralytics 的读法对不上。
    """
    meta = {"names": {0: "bottle", 1: "cable"}}
    _off, got = ti.decode_header(ti.encode_header(meta))
    assert got["names"] == {"0": "bottle", "1": "cable"}
    assert got != meta


# --------------------------------------------------------------------------
# metadata 类型还原
# --------------------------------------------------------------------------

@pytest.mark.parametrize("key,val,expected", [
    ("stride", "32", 32),
    ("batch", "1", 1),
    ("channels", "3", 3),
    ("end2end", "False", False),
    ("end2end", "True", True),
    ("end2end", "1", True),
    ("imgsz", "[640, 640]", [640, 640]),
    ("names", "{0: 'bottle', 1: 'cable'}", {0: "bottle", 1: "cable"}),
    ("description", "YOLOv8n-seg", "YOLOv8n-seg"),
    ("author", "Ultralytics", "Ultralytics"),
])
def test_coerce_restores_original_types(key, val, expected):
    assert ti._coerce(key, val) == expected


@pytest.mark.parametrize("key,val", [("stride", "abc"), ("batch", ""),
                                     ("imgsz", "[640"), ("names", "{0: 'a'")])
def test_coerce_falls_back_to_string_on_garbage(key, val):
    """还原失败不抛，退回原字符串：宁可让 AutoBackend 拿到个怪值也别在构建时炸。"""
    assert ti._coerce(key, val) == val


class _FakeProp:
    def __init__(self, key: str, value: str) -> None:
        self.key, self.value = key, value


def _fake_onnx(monkeypatch, props):
    model = types.SimpleNamespace(metadata_props=[_FakeProp(k, v) for k, v in props.items()])
    monkeypatch.setitem(sys.modules, "onnx",
                        types.SimpleNamespace(load=lambda *a, **k: model))


def test_onnx_metadata_coerces_every_value(monkeypatch):
    _fake_onnx(monkeypatch, {
        "description": "YOLOv8s-seg", "author": "Ultralytics", "task": "segment",
        "batch": "1", "imgsz": "[640, 640]", "names": "{0: 'bottle'}",
        "stride": "32", "end2end": "False",
    })
    got = ti.onnx_metadata("weights/whatever.onnx")
    assert got["task"] == "segment"
    assert got["batch"] == 1
    assert got["imgsz"] == [640, 640]
    assert got["names"] == {0: "bottle"}
    assert got["stride"] == 32
    assert got["end2end"] is False
    assert got["description"] == "YOLOv8s-seg"


def test_onnx_metadata_returns_empty_dict_without_props(monkeypatch):
    """没有 metadata 就返回 {}，调用方据此决定要不要写 header、并打一条 warn。"""
    _fake_onnx(monkeypatch, {})
    assert ti.onnx_metadata("weights/whatever.onnx") == {}


# --------------------------------------------------------------------------
# 激活显存取值
# --------------------------------------------------------------------------

def test_device_memory_prefers_v2_callable():
    """TRT 10.1 起 device_memory_size 废弃，v2 是带 profile 下标的方法。"""
    eng = types.SimpleNamespace(device_memory_size_v2=lambda i: 40265318,
                                device_memory_size=1)
    assert ti.device_memory_bytes(eng) == 40265318


def test_device_memory_falls_back_to_plain_int():
    eng = types.SimpleNamespace(device_memory_size=4096)
    assert ti.device_memory_bytes(eng) == 4096


def test_device_memory_none_when_absent():
    assert ti.device_memory_bytes(object()) is None


def test_device_memory_ignores_errors_and_zero():
    def boom(_i):
        raise RuntimeError("profile 下标越界")

    eng = types.SimpleNamespace(device_memory_size_v2=boom, device_memory_size=0)
    assert ti.device_memory_bytes(eng) is None


# --------------------------------------------------------------------------
# 依赖分层
# --------------------------------------------------------------------------

def test_module_does_not_import_tensorrt_at_top_level():
    """CI 与 base 环境都没有 tensorrt：顶层 import 会让整个测试文件收集失败。"""
    src = Path(ti.__file__).read_text(encoding="utf-8")
    top = [ln for ln in src.splitlines()
           if ln.startswith("import tensorrt") or ln.startswith("from tensorrt")]
    assert top == []


def test_trtrunner_import_raises_when_tensorrt_missing(tmp_path, monkeypatch):
    """sys.modules[name] = None 会让 import 直接抛 ImportError，等价于没装。"""
    monkeypatch.setitem(sys.modules, "tensorrt", None)
    with pytest.raises(ImportError):
        ti.TrtRunner(tmp_path / "nope.engine")