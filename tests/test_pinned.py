# -*- coding: utf-8 -*-
"""``induvision.pinned`` 的单测：全部用假 torch / 假 runner 跑，不碰 GPU。

为什么值得这么测：``PinnedTrtClient`` 里真正会出错的地方不是 CUDA 调用（那几行
交给 TensorRT 自己保证），而是**令牌与 slot 的配对**——哪个 slot 的输入被写脏了、
哪个输出缓冲还没被后处理读完就被下一帧覆盖。这类 bug 在有 GPU 的机器上表现为
"偶发的检测结果和串行不一致"，最难查；在没有 GPU 的 CI 上用假对象反而能把每一条
分支都钉死。所以这里的断言重点是：令牌不串槽、拷贝是 non_blocking、每帧只 sync 一次、
出错也要把令牌还回池子。
"""
from __future__ import annotations

import contextlib
import sys
import threading
import time
import types

import numpy as np
import pytest

from induvision.pinned import (
    PinnedOutputs,
    PinnedTrtClient,
    missing_gpu_dep_message,
    require_torch,
)

IN_SHAPE = (1, 3, 16, 16)
OUT_SHAPES = {"output0": (1, 5, 16), "output1": (1, 2, 4, 4)}
SMALL = dict(imgsz=16, channels=3, out_shapes=OUT_SHAPES)


# --------------------------------------------------------------------------
# 假对象
# --------------------------------------------------------------------------

class FakeStream:
    def __init__(self):
        self.cuda_stream = 0x1234
        self.syncs = 0

    def synchronize(self):
        self.syncs += 1


class FakeCtx:
    def __init__(self, out_shapes):
        self._out_shapes = dict(out_shapes)
        self.calls = []          # 按顺序记 ("set_input_shape"|"set_tensor_address"|"exec", ...)
        self.addrs = {}
        self.execs = 0
        self.exec_ok = True

    def set_input_shape(self, name, shape):
        self.calls.append(("set_input_shape", name, tuple(shape)))

    def get_tensor_shape(self, name):
        self.calls.append(("get_tensor_shape", name))
        return self._out_shapes[name]

    def set_tensor_address(self, name, ptr):
        self.calls.append(("set_tensor_address", name, ptr))
        self.addrs[name] = ptr

    def execute_async_v3(self, stream):
        self.calls.append(("exec", stream))
        self.execs += 1
        return self.exec_ok


class FakeTensor:
    """够用的假张量：shape / dtype / data_ptr / numpy() / copy_ / numel / element_size。"""

    _ptr = 0x10000

    def __init__(self, shape, dtype=np.float32, pin_memory=False, device=None):
        self.shape = tuple(int(v) for v in shape)
        self.dtype = dtype
        self.pin_memory = bool(pin_memory)
        self.device = device
        self._arr = np.zeros(self.shape, dtype=np.float32)
        FakeTensor._ptr += 0x100
        self._p = FakeTensor._ptr
        self.copies = []

    def numel(self):
        return int(np.prod(self.shape))

    def element_size(self):
        return 4

    def data_ptr(self):
        return self._p

    def numpy(self):
        return self._arr

    def copy_(self, src, non_blocking=False):
        self.copies.append((src, bool(non_blocking)))
        np.copyto(self._arr, src._arr)
        return self


class FakeCuda:
    def __init__(self):
        self.entered = []

    @contextlib.contextmanager
    def stream(self, s):
        self.entered.append(s)
        yield s


class FakeTorch:
    float32 = np.float32

    def __init__(self, with_cuda=True):
        self.made = []
        self.cuda = FakeCuda() if with_cuda else None

    def empty(self, shape, dtype=None, pin_memory=False, device=None):
        t = FakeTensor(shape, dtype=dtype if dtype is not None else np.float32,
                       pin_memory=pin_memory, device=device)
        self.made.append(t)
        return t

    def from_numpy(self, a):
        return FakeTensor(np.shape(a), dtype=np.asarray(a).dtype)


class FakeTrt:
    @staticmethod
    def nptype(dt):
        return {0: np.float32, 1: np.float16}[dt]


class FakeEngine:
    def __init__(self, mapping=None):
        self.mapping = mapping or {}

    def get_tensor_dtype(self, name):
        return self.mapping.get(name, 0)


class FakeRunner:
    def __init__(self, out_shapes=None, input_name="images",
                 output_names=("output0", "output1"), with_dtype=False,
                 dtype_map=None, dtype_raises=False):
        self.input_name = input_name
        self.output_names = list(output_names)
        self.stream = FakeStream()
        self.context = FakeCtx(out_shapes or OUT_SHAPES)
        self.closed = False
        if with_dtype:
            self.trt = FakeTrtRaises() if dtype_raises else FakeTrt()
            self.engine = FakeEngine(dtype_map or {})

    def close(self):
        self.closed = True


class FakeTrtRaises:
    @staticmethod
    def nptype(dt):
        raise NotImplementedError("老 API")


def _client(**kw) -> PinnedTrtClient:
    kw.setdefault("torch_mod", FakeTorch())
    kw.setdefault("runner", FakeRunner())
    for k, v in SMALL.items():
        kw.setdefault(k, v)
    return PinnedTrtClient(**kw)


# --------------------------------------------------------------------------
# 报错口径
# --------------------------------------------------------------------------

class TestMissingDep:
    def test_message_is_actionable(self):
        m = missing_gpu_dep_message(ImportError("No module named 'torch'"))
        for frag in ("缺 GPU 依赖", "ImportError", "requirements-gpu.txt",
                     "docs/environment.md", "No module named"):
            assert frag in m

    def test_require_torch_wraps_import_error(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch", None)   # import torch -> ImportError
        with pytest.raises(RuntimeError, match="缺 GPU 依赖"):
            require_torch()

    def test_ctor_without_torch_raises_actionable(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch", None)
        with pytest.raises(RuntimeError, match="缺 GPU 依赖"):
            PinnedTrtClient(runner=FakeRunner(), **SMALL)

    def test_require_torch_returns_real_module(self):
        torch = pytest.importorskip("torch")
        assert require_torch() is torch


# --------------------------------------------------------------------------
# 构造
# --------------------------------------------------------------------------

class TestCtor:
    @pytest.mark.parametrize("ps", [0, -1])
    def test_bad_pool_size(self, ps):
        with pytest.raises(ValueError, match="pool_size"):
            _client(pool_size=ps)

    @pytest.mark.parametrize("sz", [0, -640])
    def test_bad_imgsz(self, sz):
        with pytest.raises(ValueError, match="imgsz"):
            _client(imgsz=sz, out_shapes=OUT_SHAPES)

    def test_no_runner_no_engine_raises(self):
        with pytest.raises(ValueError, match="engine_path"):
            PinnedTrtClient(torch_mod=FakeTorch(), **SMALL)

    def test_no_outputs_raises(self):
        with pytest.raises(ValueError, match="没有输出张量"):
            _client(runner=FakeRunner(output_names=()))

    def test_in_shape_from_imgsz_and_channels(self):
        c = _client(imgsz=32, channels=1, out_shapes=OUT_SHAPES)
        assert c.in_shape == (1, 1, 32, 32)
        assert c.input_view(0).shape == (1, 32, 32)

    def test_pool_size_slots_built(self):
        c = _client(pool_size=3)
        assert len(c._slots) == 3
        assert c.stats()["free_now"] == 3

    def test_explicit_shapes_skip_probe(self):
        r = FakeRunner()
        _client(runner=r)
        assert ("set_input_shape", r.input_name, IN_SHAPE) not in r.context.calls

    def test_probe_asks_context_when_shapes_missing(self):
        r = FakeRunner(out_shapes={"output0": (1, 7, 16), "output1": (1, 2, 4, 4)})
        c = _client(runner=r, out_shapes=None)
        assert c.out_shapes == {"output0": (1, 7, 16), "output1": (1, 2, 4, 4)}
        names = [n for n, *_ in r.context.calls]
        assert names.index("set_input_shape") < names.index("get_tensor_shape"), \
            "必须先 set_input_shape 再问形状，否则拿到的是 profile 的 min/opt/max"

    def test_partial_shapes_probe_only_missing(self):
        r = FakeRunner(out_shapes={"output0": (1, 5, 16), "output1": (1, 9, 4, 4)})
        c = _client(runner=r, out_shapes={"output0": (1, 5, 16)})
        assert c.out_shapes["output1"] == (1, 9, 4, 4)

    def test_device_passed_through(self):
        c = _client(device="cuda:1")
        assert c.device == "cuda:1"
        assert all(t.device == "cuda:1" for t in c._slots[0]["dev"].values())

    def test_input_buffers_are_pinned(self):
        c = _client()
        for s in c._slots:
            assert s["in_host"].pin_memory is True
            assert all(h.pin_memory for h in s["host"].values())
            assert s["in_dev"].pin_memory is False


# --------------------------------------------------------------------------
# 输出 dtype 解析
# --------------------------------------------------------------------------

class TestDtypeResolve:
    def test_no_trt_attr_falls_back_to_float32(self):
        c = _client()
        assert c._resolve_out_dtype("output0") is np.float32

    def test_resolves_per_name(self):
        r = FakeRunner(with_dtype=True, dtype_map={"output0": 0, "output1": 1})
        c = _client(runner=r)
        assert c._resolve_out_dtype("output0") is np.dtype(np.float32) or \
            c._resolve_out_dtype("output0") == np.float32
        assert c._resolve_out_dtype("output1") == np.float16

    def test_nptype_failure_falls_back(self):
        r = FakeRunner(with_dtype=True, dtype_raises=True)
        assert _client(runner=r)._resolve_out_dtype("output0") == np.float32

    def test_explicit_out_dtypes_win(self):
        c = _client(out_dtypes={"output1": np.float16})
        assert c._slots[0]["dev"]["output1"].dtype == np.float16
        assert c._slots[0]["dev"]["output0"].dtype == np.float32


# --------------------------------------------------------------------------
# 令牌与池
# --------------------------------------------------------------------------

class TestTokens:
    def test_acquire_distinct_tokens(self):
        c = _client(pool_size=3)
        toks = [c.acquire() for _ in range(3)]
        assert sorted(toks) == [0, 1, 2]
        assert c.stats()["free_now"] == 0

    def test_acquire_blocks_then_times_out(self):
        c = _client(pool_size=1)
        c.acquire()
        with pytest.raises(RuntimeError, match="缓冲池"):
            c.acquire(timeout=0.02)
        assert c.pool_waits == 1

    def test_release_then_reacquire(self):
        c = _client(pool_size=1)
        t = c.acquire()
        c.release(t)
        assert c.acquire() == t
        assert c.pool_waits == 0

    def test_release_unblocks_waiter(self):
        import threading
        c = _client(pool_size=1)
        t0 = c.acquire()
        got = []

        def waiter():
            got.append(c.acquire(timeout=3.0))

        th = threading.Thread(target=waiter)
        th.start()
        time.sleep(0.05)
        c.release(t0)
        th.join(timeout=5)
        assert got == [t0]

    def test_input_views_do_not_alias(self):
        c = _client(pool_size=2)
        v0, v1 = c.input_view(0), c.input_view(1)
        assert v0.shape == v1.shape == (3, 16, 16)
        v0[...] = 7.0
        assert float(v1.max()) == 0.0, "不同令牌的输入缓冲共用内存了"

    def test_input_view_is_a_view_not_a_copy(self):
        c = _client()
        v = c.input_view(0)
        v[...] = 3.0
        assert float(c._slots[0]["in_np"].max()) == 3.0

    def test_close_drains_then_raises(self):
        """close() 的语义是"不再收新的，但已有的能取完"——排空之后 acquire 才抛。

        这条是流水线不死锁的关键：上游出错收尾时，下游不会永久挂在 get 上。
        """
        c = _client(pool_size=1)
        c.close()
        assert isinstance(c.acquire(), int)
        with pytest.raises(RuntimeError, match="已关闭或 acquire 超时"):
            c.acquire(timeout=0.01)


# --------------------------------------------------------------------------
# forward
# --------------------------------------------------------------------------

class TestForward:
    def test_single_sync_per_frame(self):
        c = _client()
        t = c.acquire()
        c.forward(t)
        assert c.runner.stream.syncs == 1, "每帧只该同步一次（D2H 全部提交之后）"
        c.forward(t)
        assert c.runner.stream.syncs == 2

    def test_all_copies_are_non_blocking(self):
        c = _client()
        t = c.acquire()
        c.forward(t)
        slot = c._slots[t]
        assert slot["in_dev"].copies and all(nb for _, nb in slot["in_dev"].copies)
        for n in c.output_names:
            assert slot["host"][n].copies
            assert all(nb for _, nb in slot["host"][n].copies), \
                "D2H 必须 non_blocking，否则每帧多一次同步"

    def test_h2d_source_is_the_pinned_host_buffer(self):
        c = _client()
        t = c.acquire()
        c.forward(t)
        src, _ = c._slots[t]["in_dev"].copies[-1]
        assert src is c._slots[t]["in_host"]

    def test_call_order_shape_then_addresses_then_exec(self):
        c = _client()
        t = c.acquire()
        c.forward(t)
        kinds = [k for k, *_ in c.runner.context.calls]
        assert kinds[0] == "set_input_shape"
        assert kinds[-1] == "exec"
        assert kinds[1:-1] == ["set_tensor_address"] * (1 + len(c.output_names))

    def test_addresses_point_at_this_slots_device_buffers(self):
        c = _client(pool_size=2)
        t = c.acquire()
        c.forward(t)
        a = c.runner.context.addrs
        assert a[c.runner.input_name] == c._slots[t]["in_dev"].data_ptr()
        for n in c.output_names:
            assert a[n] == c._slots[t]["dev"][n].data_ptr()
        other = 1 - t
        assert a[c.runner.input_name] != c._slots[other]["in_dev"].data_ptr()

    def test_executed_on_runner_stream(self):
        c = _client()
        t = c.acquire()
        c.forward(t)
        assert ("exec", c.runner.stream.cuda_stream) in c.runner.context.calls
        assert c.runner.stream in c.torch.cuda.entered

    def test_returns_views_of_pool_memory(self):
        c = _client()
        t = c.acquire()
        out = c.forward(t)
        assert isinstance(out, PinnedOutputs)
        assert out.token == t and out.names == list(c.output_names)
        for n, a in zip(out.names, out.arrays):
            assert a is c._slots[t]["np"][n], "输出必须是池内存的视图，不是副本"

    def test_output_helpers(self):
        c = _client()
        out = c.forward(c.acquire())
        assert out.get("output1").shape == OUT_SHAPES["output1"]
        assert out.shapes() == {"output0": OUT_SHAPES["output0"], "output1": OUT_SHAPES["output1"]}
        with pytest.raises(ValueError):
            out.get("nope")

    def test_execute_false_raises(self):
        r = FakeRunner()
        r.context.exec_ok = False
        c = _client(runner=r)
        with pytest.raises(RuntimeError, match="execute_async_v3"):
            c.forward(c.acquire())

    def test_forwards_counter(self):
        c = _client()
        t = c.acquire()
        for i in range(1, 4):
            c.forward(t)
            assert c.stats()["forwards"] == i

    def test_no_cuda_module_still_works(self):
        c = _client(torch_mod=FakeTorch(with_cuda=False))
        out = c.forward(c.acquire())
        assert c.runner.stream.syncs == 1 and out.arrays

    def test_written_input_reaches_device_buffer(self):
        c = _client()
        t = c.acquire()
        c.input_view(t)[...] = 1.5
        c.forward(t)
        assert float(c._slots[t]["in_dev"].numpy().max()) == 1.5


# --------------------------------------------------------------------------
# 池隔离：这才是 pool_size 存在的理由
# --------------------------------------------------------------------------

class TestPoolIsolation:
    def test_two_live_tokens_have_separate_outputs(self):
        c = _client(pool_size=2)
        t0, t1 = c.acquire(), c.acquire()
        o0 = c.forward(t0)
        snap0 = [a.copy() for a in o0.arrays]
        o0.arrays[0][...] = 42.0                      # 模拟 stage3 还在读第 n 帧
        o1 = c.forward(t1)                            # stage2 已经在推第 n+1 帧
        o1.arrays[0][...] = -7.0
        assert np.array_equal(o0.arrays[0], np.full(o0.arrays[0].shape, 42.0)), \
            "两个在飞的令牌共用了输出缓冲 —— 这就是流水线里偶发不一致的根因"
        assert all(a.ctypes.data != b.ctypes.data for a, b in zip(o0.arrays, o1.arrays))
        assert np.array_equal(snap0[1], o0.arrays[1])

    def test_release_then_reuse_overwrites(self):
        c = _client(pool_size=1)
        t = c.acquire()
        out = c.forward(t)
        out.arrays[0][...] = 5.0
        c.release(t)
        t2 = c.acquire()
        assert t2 == t
        c._slots[t2]["np"]["output0"][...] = 0.0
        assert float(out.arrays[0].max()) == 0.0, "release 之后旧视图就该被下一帧覆盖"

    def test_pipeline_style_rotation_no_race(self):
        """按流水线的真实节奏跑：acquire -> forward -> 攒着不 release -> 池满阻塞。"""
        c = _client(pool_size=2)
        held = []
        for _ in range(2):
            t = c.acquire()
            held.append((t, c.forward(t)))
        assert c.stats()["free_now"] == 0
        with pytest.raises(RuntimeError, match="缓冲池"):
            c.acquire(timeout=0.02)
        for t, _o in held:
            c.release(t)
        assert c.stats()["free_now"] == 2
        assert c.pool_waits == 1


# --------------------------------------------------------------------------
# forward_array
# --------------------------------------------------------------------------

class TestForwardArray:
    def test_3d_input_auto_expanded(self):
        c = _client()
        out = c.forward_array(np.zeros((3, 16, 16), np.float32))
        assert out.owned is True
        assert np.array_equal(c._slots[out.token]["in_np"], np.zeros(IN_SHAPE, np.float32))
        c.release(out.token)

    def test_4d_input_ok(self):
        c = _client()
        out = c.forward_array(np.zeros(IN_SHAPE, np.float32))
        assert out.arrays and out.owned
        c.release(out.token)

    def test_wrong_shape_raises(self):
        c = _client()
        with pytest.raises(ValueError, match="profile"):
            c.forward_array(np.zeros((3, 32, 32), np.float32))

    def test_dtype_coerced_to_float32(self):
        c = _client()
        out = c.forward_array(np.ones((3, 16, 16), np.uint8))
        assert c._slots[out.token]["in_np"].dtype == np.float32
        assert float(c._slots[out.token]["in_np"].max()) == 1.0
        c.release(out.token)

    def test_error_releases_owned_token(self):
        c = _client(pool_size=1)
        with pytest.raises(ValueError):
            c.forward_array(np.zeros((3, 99, 99), np.float32))
        assert c.stats()["free_now"] == 1, "出错没还令牌，池子会越用越小"

    def test_exec_failure_releases_owned_token(self):
        r = FakeRunner()
        r.context.exec_ok = False
        c = _client(runner=r, pool_size=1)
        with pytest.raises(RuntimeError, match="execute_async_v3"):
            c.forward_array(np.zeros((3, 16, 16), np.float32))
        assert c.stats()["free_now"] == 1

    def test_explicit_token_not_owned(self):
        c = _client(pool_size=2)
        t = c.acquire()
        out = c.forward_array(np.zeros((3, 16, 16), np.float32), token=t)
        assert out.token == t and out.owned is False
        assert c.stats()["free_now"] == 1

    def test_explicit_token_not_released_on_error(self):
        c = _client(pool_size=2)
        t = c.acquire()
        with pytest.raises(ValueError):
            c.forward_array(np.zeros((3, 99, 99), np.float32), token=t)
        assert c.stats()["free_now"] == 1, "调用方自己 acquire 的令牌，出错也得由调用方还"


# --------------------------------------------------------------------------
# 统计与生命周期
# --------------------------------------------------------------------------

class TestStatsAndClose:
    def test_stats_keys(self):
        c = _client(pool_size=2)
        st = c.stats()
        assert set(st) == {"pool_size", "forwards", "pool_waits", "free_now",
                           "pinned_host_mib", "device_mib", "in_shape", "out_shapes",
                           "owns_runner"}
        assert st["pool_size"] == 2 and st["forwards"] == 0 and st["pool_waits"] == 0
        assert st["in_shape"] == list(IN_SHAPE)
        assert st["out_shapes"] == {"output0": list(OUT_SHAPES["output0"]),
                                    "output1": list(OUT_SHAPES["output1"])}
        assert st["owns_runner"] is False

    def test_mib_matches_manual_accounting(self):
        c = _client(pool_size=2)
        per_slot = 3 * 16 * 16 + 5 * 16 + 2 * 4 * 4        # in + out0 + out1，float32
        expect = round(per_slot * 4 * 2 / 1024 ** 2, 3)
        st = c.stats()
        assert st["pinned_host_mib"] == expect
        assert st["device_mib"] == expect

    def test_real_scale_memory_footprint(self):
        """锁住压测报告里那个 18.893 MiB（pool_size=2 / imgsz=640 / 双输出）。"""
        c = PinnedTrtClient(torch_mod=FakeTorch(), runner=FakeRunner(out_shapes={
            "output0": (1, 51, 8400), "output1": (1, 32, 160, 160)}),
            imgsz=640, channels=3, pool_size=2,
            out_shapes={"output0": (1, 51, 8400), "output1": (1, 32, 160, 160)})
        st = c.stats()
        assert st["pinned_host_mib"] == 18.893 and st["device_mib"] == 18.893

    def test_stats_survives_fake_tensors_without_numel(self):
        c = _client()
        for s in c._slots:
            s["in_host"] = object()
        assert c.stats()["pinned_host_mib"] >= 0.0

    def test_context_manager_closes(self):
        with _client(pool_size=1) as c:
            out = c.forward_array(np.zeros((3, 16, 16), np.float32))
            assert out.owned is True          # 令牌还在外面，池子已空
        assert c._slots == [] and c._free.closed
        with pytest.raises(RuntimeError, match="已关闭或 acquire 超时"):
            c.acquire(timeout=0.01)

    def test_close_is_idempotent(self):
        c = _client()
        c.close()
        c.close()
        assert c._slots == []

    def test_injected_runner_not_closed(self):
        r = FakeRunner()
        c = _client(runner=r)
        c.close()
        assert r.closed is False, "不是自己建的 runner，就不该替别人关"

    def test_owned_runner_is_closed(self, monkeypatch):
        made = []

        class OwnedRunner(FakeRunner):
            def __init__(self, path, device=None):
                super().__init__()
                made.append((path, device))

        mod = types.ModuleType("induvision.trt_infer")
        mod.TrtRunner = OwnedRunner
        monkeypatch.setitem(sys.modules, "induvision.trt_infer", mod)
        c = PinnedTrtClient("weights/fake.engine", torch_mod=FakeTorch(),
                            device="cuda:0", **SMALL)
        assert c.stats()["owns_runner"] is True
        assert made == [("weights/fake.engine", "cuda:0")]
        c.close()
        assert c.runner.closed is True
