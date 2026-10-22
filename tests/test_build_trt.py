"""tools/build_trt.py 的单测（不需要 GPU，也不需要装 tensorrt）。

覆盖的是构建流程里"错了也不会立刻报错"的那几处：

* ``pin_layers_fp16``（失败的尝试）与 ``constrain_head_precisions``（能过的那个）。钉错层名等于没钉，构建照样
  在 proto/cv3 那个融合节点上返回 None；钉多了会把 INT8 的收益一起钉掉，而这两
  种错在报告里都只会表现为一个数字，看不出是代码写歪了。
* ``make_calibrator`` 的批生命周期——``get_batch`` 返回的是裸 ``data_ptr()``，
  Python 侧张量一释放，TRT 读的就是回收过的显存，不报错，只是悄悄校准出错范围。
* ``load_calib_inputs`` 的分批与跳过留痕——跳过读不出来的图会改变实际喂进校准
  器的样本，不留痕就等于换了口径还不说。
* ``main`` 的错误码约定（2=用法/输入问题，3=环境缺 tensorrt）。
"""
from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from conftest import load_tool

bt = load_tool("build_trt")


# --------------------------------------------------------------------------
# 测试替身
# --------------------------------------------------------------------------

def stub_trt():
    """只放 build_trt 真正用到的那几个名字。"""

    class IInt8Calibrator:
        """TRT 绑定里 get_algorithm 是纯虚函数：没重写就一定调不到，这里照实模拟。"""

        def __init__(self) -> None:
            pass

        def get_algorithm(self):
            raise NotImplementedError(
                'Tried to call pure virtual function "IInt8Calibrator::get_algorithm"')

    algo = types.SimpleNamespace(MINMAX_CALIBRATION="MINMAX_CALIBRATION",
                                 ENTROPY_CALIBRATION_2="ENTROPY_CALIBRATION_2")
    layer_type = types.SimpleNamespace(ACTIVATION="ACTIVATION",
                                       CONVOLUTION="CONVOLUTION",
                                       ELEMENTWISE="ELEMENTWISE")
    return types.SimpleNamespace(IInt8Calibrator=IInt8Calibrator,
                                 CalibrationAlgoType=algo, LayerType=layer_type,
                                 float16="trt.float16", float32="trt.float32")


class FakeLayer:
    def __init__(self, name: str, n_out: int = 2) -> None:
        self.name = name
        self._n_out = n_out
        self.precision = None
        self.out_types: dict[int, object] = {}

    @property
    def num_outputs(self) -> int:
        return self._n_out

    def set_output_type(self, j: int, t) -> None:
        self.out_types[j] = t


class FakeNetwork:
    def __init__(self, names) -> None:
        self._layers = [FakeLayer(n) for n in names]

    @property
    def num_layers(self) -> int:
        return len(self._layers)

    def get_layer(self, i: int) -> FakeLayer:
        return self._layers[i]


NAMES = [
    "/model.22/cv2.0/cv2.0.0/conv/Conv",
    "/model.22/proto/cv3/conv/Conv",
    "/model.22/proto/cv3/act/Sigmoid",
    "/model.22/proto/cv3/act/Mul",
    "/model.22/Reshape",
]


def write_manifest(path: Path, root: Path, files, *, split: str = "train",
                   imgsz: int = 640) -> dict:
    """写一份 load_sample_list 认的清单。sha256 它不校验，填个占位就行。"""
    from induvision import deploy

    payload = {
        "version": deploy.SAMPLE_LIST_VERSION, "split": split, "imgsz": imgsz,
        "root": str(root), "sampling": "equidistant", "count": len(files),
        "source_total": len(files), "sha256": "deadbeef", "git_head": None,
        "created_utc": "2026-10-23T00:00:00Z", "files": list(files),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8", newline="\n")
    return payload


@pytest.fixture
def png_root(tmp_path: Path):
    """4 张真图，够验证分批与 limit。"""
    import cv2

    root = tmp_path / "imgs"
    (root / "images" / "train").mkdir(parents=True)
    rng = np.random.default_rng(0)
    rel = []
    for i in range(4):
        rel.append(f"images/train/{i:03d}.png")
        assert cv2.imwrite(str(root / rel[-1]),
                           rng.integers(0, 255, (48, 64, 3), dtype=np.uint8))
    return root, rel


# --------------------------------------------------------------------------
# 常量与环境指纹
# --------------------------------------------------------------------------

def test_precisions_are_the_three_tiers():
    assert bt.PRECISIONS == ("fp32", "fp16", "int8")


def test_default_workspace_leaves_room_for_activations():
    """8 GB 的卡不能吃满：workspace 之外还要放激活显存和 CUDA context。"""
    assert 0 < bt.DEFAULT_WORKSPACE_GB <= 6.0


def test_git_head_returns_short_sha_or_unknown():
    head = bt._git_head()
    assert isinstance(head, str) and head
    assert head == "unknown" or len(head) >= 7


def test_file_sha256_matches_hashlib(tmp_path: Path):
    p = tmp_path / "blob.bin"
    p.write_bytes(b"\x00\x01\x02" * 5000)
    assert bt.file_sha256(p) == hashlib.sha256(p.read_bytes()).hexdigest()


def test_file_sha256_chunked_read_same_result(tmp_path: Path):
    """分块读和一次读完必须一致——chunk 边界写歪会得到一个"看着也对"的指纹。"""
    p = tmp_path / "blob.bin"
    p.write_bytes(bytes(range(256)) * 37)
    assert bt.file_sha256(p, chunk=7) == bt.file_sha256(p, chunk=1 << 20)


def test_env_fingerprint_covers_reported_modules(monkeypatch):
    monkeypatch.setitem(sys.modules, "tensorrt", None)  # import 直接 ImportError
    fp = bt.env_fingerprint()
    for k in ("python", "platform", "numpy", "cv2", "torch", "ultralytics",
              "onnx", "tensorrt"):
        assert k in fp, f"环境指纹缺 {k}"
    assert fp["tensorrt"] == "absent"
    assert all(isinstance(v, str) for v in fp.values())


# --------------------------------------------------------------------------
# pin_layers_fp16：绕开那个无 tactic 融合节点的**失败**尝试（留作对照）
# --------------------------------------------------------------------------

def test_pin_layers_fp16_only_hits_substring():
    net = FakeNetwork(NAMES)
    hit = bt.pin_layers_fp16(net, stub_trt(), "proto/cv3")
    assert hit == ["/model.22/proto/cv3/conv/Conv",
                   "/model.22/proto/cv3/act/Sigmoid",
                   "/model.22/proto/cv3/act/Mul"]
    for i, layer in enumerate(net._layers):
        want = NAMES[i] in hit
        assert (layer.precision == "trt.float16") is want
        assert (layer.out_types == {0: "trt.float16", 1: "trt.float16"}) is want


def test_pin_layers_fp16_sets_every_output():
    """两个输出都要设。漏一个，那一支还是会被优化器按 INT8 去选 tactic。"""
    net = FakeNetwork(["/model.22/proto/cv3/act/Mul"])
    bt.pin_layers_fp16(net, stub_trt(), "proto/cv3")
    assert net._layers[0].out_types == {0: "trt.float16", 1: "trt.float16"}


def test_pin_layers_fp16_no_match_returns_empty():
    net = FakeNetwork(NAMES)
    assert bt.pin_layers_fp16(net, stub_trt(), "no/such/layer") == []
    assert all(l.precision is None for l in net._layers)


def test_pin_layers_fp16_empty_substring_is_noop():
    """空串会 in 任何层名——这里必须当成"不钉"，否则整网都被钉成 FP16。"""
    net = FakeNetwork(NAMES)
    assert bt.pin_layers_fp16(net, stub_trt(), "") == []
    assert all(l.precision is None for l in net._layers)


def test_pin_layers_fp16_nameless_layer_does_not_crash():
    net = FakeNetwork(["x"])
    net._layers[0].name = None  # TRT 允许匿名层
    assert bt.pin_layers_fp16(net, stub_trt(), "x") == []


# --------------------------------------------------------------------------
# constrain_head_precisions：ultralytics 配方，本机 INT8 唯一能构建成功的组合
# --------------------------------------------------------------------------

class FakeTypedLayer(FakeLayer):
    def __init__(self, name: str, ltype: str, n_out: int = 2) -> None:
        super().__init__(name, n_out)
        self.type = ltype


def typed_net(spec):
    net = FakeNetwork([])
    net._layers = [FakeTypedLayer(n, ty) for n, ty in spec]
    return net


#: 仿真 YOLOv8s-seg 的层名：backbone/neck 里也到处是 SiLU（=Sigmoid+Mul），
#: 所以"按头作用域筛"这件事必须被测到。
HEAD_SPEC = [
    ("/model.3/cv2/conv/Conv", "CONVOLUTION"),
    ("/model.3/cv2/act/Sigmoid", "ACTIVATION"),          # neck 的 SiLU，不该被钉
    ("/model.22/cv2.0/cv2.0.2/Conv", "CONVOLUTION"),     # 分支输出卷积 -> FP16
    ("/model.22/cv3.1/cv3.1.2/Conv", "CONVOLUTION"),     # 同上
    ("/model.22/dfl/conv/Conv", "CONVOLUTION"),          # DFL -> FP16
    ("/model.22/proto/cv3/conv/Conv", "CONVOLUTION"),    # 不满足两个标记，不钉
    ("/model.22/proto/cv3/act/Sigmoid", "ACTIVATION"),   # 头部 sigmoid -> FP32
    ("/model.22/proto/cv3/act/Mul", "ELEMENTWISE"),      # Mul 不在配方里
    ("/model.22/Reshape", "ELEMENTWISE"),
]


def test_head_modes_are_ultralytics_or_none():
    assert bt.INT8_HEAD_MODES == ("ultralytics", "none")


def test_head_constraint_only_touches_head_layers():
    """全网筛 sigmoid 会把 backbone/neck 的 SiLU 一起钉掉，INT8 加速就没了。"""
    net = typed_net(HEAD_SPEC)
    hit = bt.constrain_head_precisions(net, stub_trt())
    names = [h["name"] for h in hit]
    assert "/model.3/cv2/act/Sigmoid" not in names
    assert "/model.22/proto/cv3/act/Sigmoid" in names


def test_head_constraint_head_is_largest_model_index():
    """nms=True 导出会多包一层，头的索引不一定是 22；取最大值才对。"""
    net = typed_net([("/model.7/cv2.0.2/Conv", "CONVOLUTION"),
                     ("/model.9/proto/cv3/act/Sigmoid", "ACTIVATION")])
    hit = bt.constrain_head_precisions(net, stub_trt())
    assert [h["name"] for h in hit] == ["/model.9/proto/cv3/act/Sigmoid"]


def test_head_constraint_sigmoid_goes_fp32_not_fp16():
    """置信度标定要 FP32。钉成 FP16 是上一轮失败的做法，也保不住 mAP。"""
    net = typed_net(HEAD_SPEC)
    hit = bt.constrain_head_precisions(net, stub_trt())
    sig = [h for h in hit if h["name"].endswith("Sigmoid")][0]
    assert sig["precision"] == "float32" and sig["kind"] == "activation-sigmoid"
    layer = net._layers[HEAD_SPEC.index(("/model.22/proto/cv3/act/Sigmoid", "ACTIVATION"))]
    assert layer.precision == "trt.float32"
    assert layer.out_types == {0: "trt.float32", 1: "trt.float32"}


def test_head_constraint_output_conv_and_dfl_go_fp16():
    net = typed_net(HEAD_SPEC)
    hit = {h["name"]: h for h in bt.constrain_head_precisions(net, stub_trt())}
    for n in ("/model.22/cv2.0/cv2.0.2/Conv", "/model.22/cv3.1/cv3.1.2/Conv",
              "/model.22/dfl/conv/Conv"):
        assert hit[n]["precision"] == "float16" and hit[n]["kind"] == "conv-output"


def test_head_constraint_skips_unmarked_conv_and_elementwise():
    """proto/cv3/conv 不以 .2/Conv 结尾也不含 /dfl/，公式里就不钉它；Mul 同理。"""
    net = typed_net(HEAD_SPEC)
    names = [h["name"] for h in bt.constrain_head_precisions(net, stub_trt())]
    assert "/model.22/proto/cv3/conv/Conv" not in names
    assert "/model.22/proto/cv3/act/Mul" not in names
    assert "/model.22/Reshape" not in names


def test_head_constraint_nameless_and_prefixless_layers_return_empty():
    """层名里没有 /model.N/ 就一层也不该钉——head 退化成 "/" 时不能误伤全网。"""
    net = typed_net([("conv1", "CONVOLUTION"), ("sigmoid_x", "ACTIVATION")])
    assert bt.constrain_head_precisions(net, stub_trt()) == []
    net2 = typed_net([("/model.22/proto/cv3/act/Sigmoid", "ACTIVATION")])
    net2._layers[0].name = None
    assert bt.constrain_head_precisions(net2, stub_trt()) == []


def test_head_constraint_marks_every_output():
    net = typed_net([("/model.22/cv2.0/cv2.0.2/Conv", "CONVOLUTION")])
    net._layers[0]._n_out = 3
    bt.constrain_head_precisions(net, stub_trt())
    assert net._layers[0].out_types == {0: "trt.float16", 1: "trt.float16", 2: "trt.float16"}


def test_build_engine_rejects_pin_together_with_head_mode(tmp_path: Path):
    """PREFER 与 OBEY 同时开行为未定义，而且混用了掉点数无法归因。"""
    with pytest.raises(ValueError, match="二选一"):
        bt.build_engine(tmp_path / "a.onnx", tmp_path / "a.engine", "int8",
                        int8_pin="proto/cv3")


def test_build_engine_rejects_unknown_head_mode(tmp_path: Path):
    with pytest.raises(ValueError, match="未知 int8_head"):
        bt.build_engine(tmp_path / "a.onnx", tmp_path / "a.engine", "int8",
                        int8_head="auto")


def test_build_engine_head_none_allows_pin(tmp_path: Path):
    """--int8-head none 是对照组，这时 pin 该能用（后面才会因为没装 TRT 而失败）。"""
    with pytest.raises(Exception) as e:
        bt.build_engine(tmp_path / "a.onnx", tmp_path / "a.engine", "int8",
                        int8_pin="proto/cv3", int8_head="none")
    assert "二选一" not in str(e.value)


# --------------------------------------------------------------------------
# make_calibrator
# --------------------------------------------------------------------------

def make_calib(tmp_path: Path, n_batches: int = 3, batch_size: int = 2, cache="c.cache"):
    batches = [np.zeros((batch_size, 3, 8, 8), dtype=np.float32) for _ in range(n_batches)]
    return bt.make_calibrator(stub_trt(), batches, batch_size, tmp_path / cache, "cpu")


def test_calib_algos_map_to_trt_enum_names():
    """报告里写的算法名必须是 TRT 的真枚举名，不能是我们自己编的短名。"""
    assert bt.CALIB_ALGOS == {"minmax": "MINMAX_CALIBRATION",
                              "entropy": "ENTROPY_CALIBRATION_2"}
    assert bt.CALIB_BASE == "IInt8Calibrator"


def test_calibrator_get_algorithm_is_overridden(tmp_path: Path):
    """get_algorithm 是纯虚函数，没重写就报错到校准器外面去（stub 里报 NotImplementedError）。"""
    trt_mod = stub_trt()
    c = bt.make_calibrator(trt_mod, _one_batch(), 2, tmp_path / "c.cache", "cpu")
    assert type(c).get_algorithm is not trt_mod.IInt8Calibrator.get_algorithm
    assert c.get_algorithm() == trt_mod.CalibrationAlgoType.MINMAX_CALIBRATION


def _one_batch():
    return [np.zeros((2, 3, 8, 8), dtype=np.float32)]


def test_calibrator_default_algo_is_minmax(tmp_path: Path):
    """默认必须是 minmax：本机熵校准构建一定失败在 proto/cv3，见 CALIB_ALGOS。"""
    trt_mod = stub_trt()
    c = bt.make_calibrator(trt_mod, _one_batch(), 2, tmp_path / "c.cache", "cpu")
    assert isinstance(c, trt_mod.IInt8Calibrator)
    assert c.algo == "minmax"
    assert c.get_algorithm() == trt_mod.CalibrationAlgoType.MINMAX_CALIBRATION
    assert c.get_batch_size() == 2


def test_calibrator_algo_entropy_shares_base_only_changes_enum(tmp_path: Path):
    """两种算法共用同一个基类，只靠 get_algorithm() 的返回值区分。"""
    trt_mod = stub_trt()
    c = bt.make_calibrator(trt_mod, _one_batch(), 2, tmp_path / "c.cache", "cpu",
                           algo="entropy")
    assert isinstance(c, trt_mod.IInt8Calibrator)
    assert c.algo == "entropy"
    assert c.get_algorithm() == trt_mod.CalibrationAlgoType.ENTROPY_CALIBRATION_2


def test_calibrator_unknown_algo_raises(tmp_path: Path):
    with pytest.raises(ValueError, match="未知校准算法"):
        bt.make_calibrator(stub_trt(), _one_batch(), 2, tmp_path / "c.cache", "cpu",
                           algo="kl")


def test_calibrator_missing_base_class_points_at_qdq_or_fp16(tmp_path: Path):
    """TRT 11 移除了 IInt8Calibrator；报错要直接说出下一步能干什么。"""
    trt_mod = stub_trt()
    trt_mod.IInt8Calibrator = None
    with pytest.raises(RuntimeError, match="fp16"):
        bt.make_calibrator(trt_mod, _one_batch(), 2, tmp_path / "c.cache", "cpu")


def test_calibrator_missing_algo_enum_raises(tmp_path: Path):
    trt_mod = stub_trt()
    trt_mod.CalibrationAlgoType = types.SimpleNamespace(ENTROPY_CALIBRATION_2="x")
    with pytest.raises(RuntimeError, match="MINMAX_CALIBRATION"):
        bt.make_calibrator(trt_mod, _one_batch(), 2, tmp_path / "c.cache", "cpu")



def test_calibrator_cache_miss_returns_none(tmp_path: Path):
    """没缓存时返回 None 让 TRT 去跑统计；返回空 bytes 会被当成"缓存是空的"。"""
    c = make_calib(tmp_path)
    assert c.read_calibration_cache() is None
    assert c.cache_hit is False


def test_calibrator_cache_hit_returns_bytes(tmp_path: Path):
    (tmp_path / "c.cache").write_bytes(b"\x01\x02\x03")
    c = make_calib(tmp_path)
    assert c.read_calibration_cache() == b"\x01\x02\x03"
    assert c.cache_hit is True


def test_calibrator_write_cache_creates_parent(tmp_path: Path):
    c = make_calib(tmp_path, cache="deep/nested/c.cache")
    c.write_calibration_cache(b"range-data")
    assert (tmp_path / "deep/nested/c.cache").read_bytes() == b"range-data"
    assert c.cache_written is True


def test_calibrator_get_batch_exhausts_then_none(tmp_path: Path, monkeypatch):
    # 校准器要拿真实张量的 data_ptr()，没 torch 就只能跳过（CI 刻意不装 torch）
    torch = pytest.importorskip("torch")

    calls: list[int] = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: calls.append(1))
    c = make_calib(tmp_path, n_batches=2)
    assert isinstance(c.get_batch(["images"])[0], int)
    assert isinstance(c.get_batch(["images"])[0], int)
    assert c.get_batch(["images"]) is None
    assert c.served == 2 and len(calls) == 2


def test_calibrator_holds_current_tensor_until_next_batch(tmp_path: Path, monkeypatch):
    """返回裸指针，张量必须一直挂在 self._cur 上；否则 TRT 读到的是回收的显存。"""
    torch = pytest.importorskip("torch")

    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    c = make_calib(tmp_path, n_batches=3)
    ptr1 = c.get_batch(["images"])[0]
    assert c._cur is not None and c._cur.data_ptr() == ptr1
    first = c._cur
    ptr2 = c.get_batch(["images"])[0]
    assert c._cur is not first  # 上一批已放掉
    assert c._cur.data_ptr() == ptr2


def test_calibrator_zero_batches_yields_none_immediately(tmp_path: Path):
    c = bt.make_calibrator(stub_trt(), [], 2, tmp_path / "c.cache", "cpu")
    assert c.get_batch(["images"]) is None
    assert c.served == 0


# --------------------------------------------------------------------------
# load_calib_inputs
# --------------------------------------------------------------------------

def test_load_calib_inputs_batches_and_meta(tmp_path: Path, png_root):
    root, rel = png_root
    man = tmp_path / "calib.json"
    write_manifest(man, root, rel)
    batches, meta = bt.load_calib_inputs(man, imgsz=320, limit=0, batch_size=3)
    assert [b.shape[0] for b in batches] == [3, 1]  # 4 张切 3+1
    assert all(b.dtype == np.float32 and b.shape[1:] == (3, 320, 320) for b in batches)
    assert all(b.flags["C_CONTIGUOUS"] for b in batches)
    assert meta["images_used"] == 4 and meta["batches"] == 2
    assert meta["batch_size"] == 3 and meta["skipped"] == []


def test_load_calib_inputs_respects_limit(tmp_path: Path, png_root):
    root, rel = png_root
    man = tmp_path / "calib.json"
    write_manifest(man, root, rel)
    batches, meta = bt.load_calib_inputs(man, imgsz=320, limit=2, batch_size=8)
    assert meta["images_used"] == 2 and len(batches) == 1
    # 只取清单前 N 张，不是随机挑——同一份清单两次跑必须喂同样的图
    assert batches[0].shape[0] == 2


def test_load_calib_inputs_records_unreadable(tmp_path: Path, png_root):
    """读不出来的图要跳过并留痕，不能装作全都用上了。"""
    root, rel = png_root
    bad = "images/train/bad.png"
    (root / bad).parent.mkdir(parents=True, exist_ok=True)
    (root / bad).write_bytes(b"not a png at all")
    man = tmp_path / "calib.json"
    write_manifest(man, root, rel + [bad])
    batches, meta = bt.load_calib_inputs(man, imgsz=320, limit=0, batch_size=8)
    assert meta["images_used"] == 4
    assert meta["skipped"] == [str(root / bad)]
    assert sum(b.shape[0] for b in batches) == 4


def test_load_calib_inputs_all_unreadable_raises(tmp_path: Path):
    root = tmp_path / "imgs"
    root.mkdir()
    bad = ["a.png", "b.png"]
    for n in bad:
        (root / n).write_bytes(b"junk")
    man = tmp_path / "calib.json"
    write_manifest(man, root, bad)
    with pytest.raises(ValueError, match="没有一张图能读出来"):
        bt.load_calib_inputs(man, imgsz=320, limit=0, batch_size=8)


def test_load_calib_inputs_missing_file_raises(tmp_path: Path):
    """strict 清单缺图必须炸，绝不静默换图（换图=换口径）。"""
    from induvision.deploy import load_sample_list  # noqa: F401 - 只为拿异常类型

    man = tmp_path / "calib.json"
    write_manifest(man, tmp_path / "nope", ["images/train/x.png"])
    with pytest.raises(FileNotFoundError):
        bt.load_calib_inputs(man, imgsz=320, limit=0, batch_size=8)


# --------------------------------------------------------------------------
# main() 错误码
# --------------------------------------------------------------------------

def dummy_onnx(tmp_path: Path, name="baseline15_fp32.onnx") -> Path:
    p = tmp_path / name
    p.write_bytes(b"\x08\x07\x12\x00fake-onnx")
    return p


def test_main_missing_onnx_returns_2(tmp_path: Path, capsys):
    rc = bt.main(["--onnx", str(tmp_path / "nope.onnx"), "--prec", "fp32"])
    assert rc == 2
    assert "缺 ONNX" in capsys.readouterr().out


def test_main_int8_without_calib_list_returns_2(tmp_path: Path, capsys):
    onnx = dummy_onnx(tmp_path)
    rc = bt.main(["--onnx", str(onnx), "--prec", "int8",
                  "--calib-list", str(tmp_path / "no_calib.json")])
    assert rc == 2
    assert "校准清单" in capsys.readouterr().out


def test_main_unknown_prec_is_rejected_by_argparse(tmp_path: Path):
    onnx = dummy_onnx(tmp_path)
    with pytest.raises(SystemExit):
        bt.main(["--onnx", str(onnx), "--prec", "bf16"])


def test_main_dry_run_lists_targets_without_building(tmp_path: Path, capsys):
    onnx = dummy_onnx(tmp_path)
    calib = tmp_path / "calib.json"
    write_manifest(calib, tmp_path, [])
    rc = bt.main(["--onnx", str(onnx), "--prec", "fp32", "int8",
                  "--calib-list", str(calib), "--out-dir", str(tmp_path / "w"),
                  "--dry-run"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "baseline15_trt_fp32.engine" in out
    assert "baseline15_trt_int8.engine" in out
    assert "calib_list=" in out
    assert not list((tmp_path / "w").glob("*.engine")) if (tmp_path / "w").exists() else True


def test_main_without_tensorrt_returns_3(tmp_path: Path, capsys, monkeypatch):
    """环境没装 tensorrt 报 3，和"用法写错"的 2 分开——排查方向不一样。"""
    onnx = dummy_onnx(tmp_path)
    monkeypatch.setitem(sys.modules, "tensorrt", None)
    rc = bt.main(["--onnx", str(onnx), "--prec", "fp32",
                  "--out-dir", str(tmp_path / "w")])
    assert rc == 3
    assert "import tensorrt 失败" in capsys.readouterr().out


def test_main_dry_run_name_defaults_from_onnx_stem(tmp_path: Path, capsys):
    """run 名从 onnx 文件名反推，去掉 _fp32 后缀——engine 命名全靠它。"""
    onnx = dummy_onnx(tmp_path, "baseline15_fp32.onnx")
    bt.main(["--onnx", str(onnx), "--prec", "fp16", "--dry-run"])
    out = capsys.readouterr().out
    assert "name=baseline15" in out


# --------------------------------------------------------------------------
# TF32：build_engine 全链路（fake builder），只为验证这个 flag 真被清掉了
# --------------------------------------------------------------------------

class FakeConfig:
    """记下落进来的每个 flag，build_engine 的精度行为全在这里留痕。"""

    def __init__(self) -> None:
        self.set_flags: list[object] = []
        self.cleared_flags: list[object] = []
        self.profiling_verbosity = None
        self.int8_calibrator = None
        self.profiles: list[object] = []
        self.pools: list[tuple[object, int]] = []

    def set_memory_pool_limit(self, pool, limit: int) -> None:
        self.pools.append((pool, limit))

    def add_optimization_profile(self, profile) -> None:
        self.profiles.append(profile)

    def set_flag(self, flag) -> None:
        self.set_flags.append(flag)

    def clear_flag(self, flag) -> None:
        self.cleared_flags.append(flag)


def fake_trt_module():
    """够 build_engine 跑完 FP32 全程的最小 TRT 假件。"""

    class FakeEngine:
        num_layers = 3
        num_io_tensors = 1

        def get_tensor_name(self, i: int) -> str:
            return "images"

    class FakeNetwork:
        num_inputs = 1
        num_layers = 0

        def get_input(self, i: int):
            return types.SimpleNamespace(name="images")

        def get_layer(self, i: int):
            raise IndexError

    class FakeBuilder:
        platform_has_fast_fp16 = True
        platform_has_fast_int8 = True
        last_config: FakeConfig | None = None

        def __init__(self, logger) -> None:
            self.logger = logger

        def create_network(self, flags):
            return FakeNetwork()

        def create_builder_config(self):
            c = FakeConfig()
            FakeBuilder.last_config = c
            return c

        def create_optimization_profile(self):
            return types.SimpleNamespace(set_shape=lambda *a, **k: None)

        def build_serialized_network(self, network, config):
            return bytearray(b"fake-engine-blob")

    class FakeParser:
        num_errors = 0

        def __init__(self, network, logger) -> None:
            pass

        def parse(self, data) -> bool:
            return True

    class FakeRuntime:
        def __init__(self, logger) -> None:
            pass

        def deserialize_cuda_engine(self, blob):
            return FakeEngine()

    flags = types.SimpleNamespace(TF32="TF32", FP16="FP16", INT8="INT8",
                                  OBEY_PRECISION_CONSTRAINTS="OBEY",
                                  PREFER_PRECISION_CONSTRAINTS="PREFER")
    class FakeLogger:
        """SimpleNamespace 塞 __call__ 是不生效的，Logger 得是真类。"""

        VERBOSE, WARNING, ERROR = "V", "W", "E"

        def __init__(self, level) -> None:
            self.level = level

    return types.SimpleNamespace(
        __version__="10.16.1.11-fake", Logger=FakeLogger,
        Builder=FakeBuilder, OnnxParser=FakeParser, Runtime=FakeRuntime,
        BuilderFlag=flags, MemoryPoolType=types.SimpleNamespace(WORKSPACE="WS"),
        ProfilingVerbosity=types.SimpleNamespace(DETAILED="D"),
        CalibrationAlgoType=types.SimpleNamespace(
            MINMAX_CALIBRATION="MINMAX_CALIBRATION",
            ENTROPY_CALIBRATION_2="ENTROPY_CALIBRATION_2"),
        LayerType=types.SimpleNamespace(ACTIVATION="A", CONVOLUTION="C",
                                        ELEMENTWISE="E"),
        IInt8Calibrator=type("IInt8Calibrator", (), {}),
        float16="trt.float16", float32="trt.float32")


@pytest.fixture()
def fake_build(tmp_path: Path, monkeypatch):
    """把 build_engine 的外部依赖全部换成假件，返回 (trt_mod, onnx, out)。"""
    trt_mod = fake_trt_module()
    monkeypatch.setitem(sys.modules, "tensorrt", trt_mod)
    onnx = tmp_path / "baseline15_fp32.onnx"
    onnx.write_bytes(b"\x08fake-onnx")
    out = tmp_path / "w" / "baseline15_trt_fp32.engine"
    monkeypatch.setattr(bt, "onnx_metadata", lambda p: {"task": "segment"})
    monkeypatch.setattr(bt, "write_engine_file",
                        lambda p, blob, meta: (Path(p).parent.mkdir(parents=True, exist_ok=True),
                                               Path(p).write_bytes(bytes(blob)),
                                               len(bytes(blob)))[2])
    monkeypatch.setattr(bt, "device_memory_bytes", lambda e: 2048)
    monkeypatch.setattr(bt, "inspect_layers",
                        lambda blob, trt: {"output_formats": {"Float": 3}, "layer_types": {}})
    return trt_mod, onnx, out


def test_build_engine_defaults_to_tf32_on(fake_build):
    """不传 tf32 就是 TRT 默认（开着）——报告里要照实记，别让人以为跑的是真 FP32。"""
    trt_mod, onnx, out = fake_build
    rep = bt.build_engine(onnx, out, "fp32")
    cfg = trt_mod.Builder.last_config
    assert rep["tf32"] is True
    assert cfg.cleared_flags == []
    assert "no-TF32" not in rep["builder_flags"]


def test_build_engine_no_tf32_clears_builder_flag(fake_build):
    """--no-tf32 必须真的调 clear_flag(TF32)：漏了这一步，报告写着真 FP32、
    实际还是 TF32，rel_l2 会停在 1e-3 量级，然后你就会去查前处理（本轮真查了）。"""
    trt_mod, onnx, out = fake_build
    rep = bt.build_engine(onnx, out, "fp32", tf32=False)
    cfg = trt_mod.Builder.last_config
    assert cfg.cleared_flags == [trt_mod.BuilderFlag.TF32]
    assert rep["tf32"] is False
    assert "no-TF32" in rep["builder_flags"]


def test_build_engine_no_tf32_keeps_fp16_flag(fake_build):
    """关 TF32 与开 FP16 是两件事，不能互相顶掉。"""
    trt_mod, onnx, out = fake_build
    out16 = out.with_name("baseline15_trt_fp16.engine")
    rep = bt.build_engine(onnx, out16, "fp16", tf32=False)
    cfg = trt_mod.Builder.last_config
    assert cfg.set_flags == [trt_mod.BuilderFlag.FP16]
    assert cfg.cleared_flags == [trt_mod.BuilderFlag.TF32]
    # 顺序不重要（TF32 的处理在精度分支之前），重要的是两个 flag 都在、都没被顶掉
    assert set(rep["builder_flags"]) == {"FP16", "no-TF32"}


def test_main_dry_run_reports_tf32_state(tmp_path: Path, capsys):
    """dry-run 要把 TF32 状态打出来：这是三档里唯一"看不见但影响数值"的开关。"""
    onnx = dummy_onnx(tmp_path)
    bt.main(["--onnx", str(onnx), "--prec", "fp32", "--dry-run"])
    assert "tf32=True" in capsys.readouterr().out
    bt.main(["--onnx", str(onnx), "--prec", "fp32", "--no-tf32", "--dry-run"])
    assert "tf32=False" in capsys.readouterr().out
