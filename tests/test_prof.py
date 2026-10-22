# -*- coding: utf-8 -*-
"""``induvision.prof`` 的单测：nvidia-smi 解析、CPU 核当量、采样线程。

全部纯 CPU、**不要求机器上有显卡**：

- ``smi_sample`` 的子进程调用通过 ``runner`` 注入假对象；
- ``ResourceSampler`` 的时钟通过 ``proc_clock`` / ``wall_clock`` / ``sleeper`` 注入假读数。

所以「nvidia-smi 不存在 / 超时 / 输出是垃圾 / 驱动返回 [N/A]」这些在生产上
偶发一次就毁掉一轮 20 分钟压测的分支，在 CI 上是每次必跑、且完全确定性的。
"""
from __future__ import annotations

import subprocess
import threading
import time

import pytest

from induvision.prof import (
    SMI_FIELDS,
    ResourceSampler,
    _max,
    _mean,
    _to_int,
    cpu_cores,
    parse_smi,
    smi_sample,
)


# --------------------------------------------------------------------------
# 工具
# --------------------------------------------------------------------------

class _CP:
    """假的 ``subprocess.CompletedProcess``：只用到 stdout 与 returncode 两个属性。"""

    def __init__(self, stdout="", returncode=0):
        self.stdout = stdout
        self.returncode = returncode


class _Runner:
    """可注入的假 runner：记下每次调用的 (cmd, timeout)，按脚本返回或抛异常。

    脚本项是异常实例就抛，是 ``_CP`` 就返回；脚本用完后重复最后一项。
    """

    def __init__(self, *script):
        self.script = list(script) or [_CP()]
        self.calls = []

    def __call__(self, cmd, timeout):
        self.calls.append((list(cmd), timeout))
        item = self.script[min(len(self.calls) - 1, len(self.script) - 1)]
        if isinstance(item, BaseException):
            raise item
        return item

    @property
    def last_cmd(self):
        return self.calls[-1][0]


class _Clocks:
    """假时钟：每次读数固定步进，让 CPU 核当量变成一个可以手算的确定值。"""

    def __init__(self, wall_step=1.0, proc_step=0.1, wall0=100.0, proc0=5.0):
        self.wall_step = float(wall_step)
        self.proc_step = float(proc_step)
        self._w = float(wall0)
        self._p = float(proc0)
        self.wall_calls = 0
        self.proc_calls = 0

    def wall(self):
        self.wall_calls += 1
        self._w += self.wall_step
        return self._w

    def proc(self):
        self.proc_calls += 1
        self._p += self.proc_step
        return self._p


def _sampler(**kw):
    """建一个「不真等、采满 max_samples 轮就收工」的采样器并等它跑完。

    ``max_samples`` 必须给：sleeper 是空操作，不封顶就是一个忙等死循环。
    返回时线程已经 join 完，之后的 stop() 只会补一行终值读数，行数是确定的。
    """
    assert kw.get("max_samples") is not None, "_sampler 必须指定 max_samples，否则死循环"
    kw.setdefault("interval_s", 0.2)
    kw.setdefault("sleeper", lambda s: None)
    rs = ResourceSampler(**kw)
    rs.start()
    rs._thread.join(timeout=10.0)
    return rs


_GOOD = "37, 2246, 8188"


# --------------------------------------------------------------------------
# _to_int：单格解析
# --------------------------------------------------------------------------

class TestToInt:
    def test_mib_suffix(self):
        assert _to_int("2246 MiB") == 2246

    def test_percent_suffix_and_padding(self):
        assert _to_int("  37 %  ") == 37

    def test_leading_digits_only(self):
        assert _to_int("12abc34") == 12, "遇到第一个非数字就停，不做全局抓数"

    def test_na_bracket(self):
        assert _to_int("[N/A]") is None

    def test_not_supported_bracket(self):
        assert _to_int("[Not Supported]") is None

    def test_empty(self):
        assert _to_int("") is None

    def test_whitespace_only(self):
        assert _to_int("   ") is None

    def test_no_digits(self):
        assert _to_int("MiB") is None

    def test_zero_is_zero_not_none(self):
        v = _to_int("0 %")
        assert v == 0 and v is not None, "0% 利用率是有效读数，不能和读不到混成 None"

    def test_negative_becomes_none(self):
        assert _to_int("-1") is None, "负号不是数字，'-' 处就停了 -> 无有效数字 -> None"


# --------------------------------------------------------------------------
# parse_smi：整行解析
# --------------------------------------------------------------------------

class TestParseSmi:
    def test_normal_row(self):
        assert parse_smi(_GOOD) == {
            "gpu_util_pct": 37, "mem_used_mib": 2246, "mem_total_mib": 8188,
        }

    def test_field_order_follows_smi_fields(self):
        assert SMI_FIELDS.split(",") == ["utilization.gpu", "memory.used", "memory.total"]

    def test_none(self):
        assert parse_smi(None) is None

    def test_empty_string(self):
        assert parse_smi("") is None

    def test_too_few_fields(self):
        assert parse_smi("37, 2246") is None

    def test_extra_fields_ignored(self):
        assert parse_smi("37, 2246, 8188, 999")["mem_total_mib"] == 8188

    def test_first_line_wins(self):
        got = parse_smi("1, 2, 3\n40, 50, 60")
        assert got == {"gpu_util_pct": 1, "mem_used_mib": 2, "mem_total_mib": 3}, \
            "多卡机器不带 --id 会返回多行，只取第一行（调用方用 gpu_index 指定哪块卡）"

    def test_blank_lines_skipped(self):
        assert parse_smi("\n\n" + _GOOD + "\n")["gpu_util_pct"] == 37

    def test_all_na_gives_dict_of_none(self):
        got = parse_smi("[N/A], [N/A], [N/A]")
        assert got == {"gpu_util_pct": None, "mem_used_mib": None, "mem_total_mib": None}
        assert got is not None, "命令通了但字段不支持：是「读到了、值为空」，不是「没读到」"

    def test_partial_na(self):
        got = parse_smi("37, [N/A], 8188")
        assert got == {"gpu_util_pct": 37, "mem_used_mib": None, "mem_total_mib": 8188}

    def test_garbage(self):
        assert parse_smi("hello world") is None

    def test_non_string(self):
        assert parse_smi(123) is None


# --------------------------------------------------------------------------
# smi_sample：子进程失败路径全覆盖
# --------------------------------------------------------------------------

class TestSmiSample:
    def test_success(self):
        r = _Runner(_CP(_GOOD))
        assert smi_sample(runner=r) == {
            "gpu_util_pct": 37, "mem_used_mib": 2246, "mem_total_mib": 8188,
        }

    def test_cmd_has_binary_and_id(self):
        r = _Runner(_CP(_GOOD))
        smi_sample(binary="my-smi", gpu_index=3, runner=r)
        assert r.last_cmd[0] == "my-smi"
        assert "--id=3" in r.last_cmd

    def test_cmd_has_query_and_format(self):
        r = _Runner(_CP(_GOOD))
        smi_sample(runner=r)
        assert f"--query-gpu={SMI_FIELDS}" in r.last_cmd
        assert "--format=csv,noheader,nounits" in r.last_cmd

    def test_timeout_forwarded(self):
        r = _Runner(_CP(_GOOD))
        smi_sample(runner=r, timeout=7.5)
        assert r.calls[-1][1] == 7.5

    def test_gpu_index_negative_still_formatted(self):
        r = _Runner(_CP(_GOOD))
        smi_sample(gpu_index=-1, runner=r)
        assert "--id=-1" in r.last_cmd

    def test_file_not_found(self):
        assert smi_sample(runner=_Runner(FileNotFoundError("no nvidia-smi"))) is None

    def test_os_error(self):
        assert smi_sample(runner=_Runner(OSError("perm"))) is None

    def test_timeout_expired(self):
        exc = subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=2.0)
        assert smi_sample(runner=_Runner(exc)) is None

    def test_generic_subprocess_error(self):
        assert smi_sample(runner=_Runner(subprocess.SubprocessError("x"))) is None

    def test_value_error(self):
        assert smi_sample(runner=_Runner(ValueError("bad args"))) is None

    def test_nonzero_returncode(self):
        assert smi_sample(runner=_Runner(_CP(_GOOD, returncode=1))) is None

    def test_runner_returns_none(self):
        assert smi_sample(runner=_Runner(None)) is None

    def test_stdout_garbage(self):
        assert smi_sample(runner=_Runner(_CP("!!! not csv !!!"))) is None

    def test_stdout_none(self):
        assert smi_sample(runner=_Runner(_CP(None))) is None

    def test_na_row_is_not_none(self):
        got = smi_sample(runner=_Runner(_CP("[N/A], [N/A], [N/A]")))
        assert got is not None and got["gpu_util_pct"] is None

    def test_default_runner_never_raises(self):
        """不注入 runner：这台机器有没有 nvidia-smi 都行，只能返回 dict 或 None。"""
        got = smi_sample(timeout=3.0)
        assert got is None or isinstance(got, dict)

    def test_repeated_calls_use_script_in_order(self):
        r = _Runner(_CP("1, 10, 100"), _CP("2, 20, 100"))
        assert smi_sample(runner=r)["gpu_util_pct"] == 1
        assert smi_sample(runner=r)["gpu_util_pct"] == 2


# --------------------------------------------------------------------------
# cpu_cores：核当量口径
# --------------------------------------------------------------------------

class TestCpuCores:
    def test_multi_core(self):
        assert cpu_cores(1.0, 3.5, 2.0) == 1.25

    def test_below_one_core(self):
        assert cpu_cores(0.0, 0.3, 1.0) == 0.3, "大部分时间在等 GPU / 等磁盘时这个数 < 1"

    def test_rounds_to_4(self):
        assert cpu_cores(0.0, 1.0, 3.0) == 0.3333

    def test_zero_delta_is_zero(self):
        v = cpu_cores(2.0, 2.0, 1.0)
        assert v == 0.0 and v is not None

    @pytest.mark.parametrize("wall", [0.0, -1.0, None])
    def test_bad_wall_gives_none(self, wall):
        assert cpu_cores(0.0, 1.0, wall) is None

    def test_regressing_process_time_gives_none(self):
        assert cpu_cores(5.0, 4.0, 1.0) is None, "读数倒退（时钟重置）不能报成负核数"


# --------------------------------------------------------------------------
# _mean / _max
# --------------------------------------------------------------------------

class TestMeanMax:
    def test_mean_empty(self):
        assert _mean([]) is None

    def test_mean_filters_none(self):
        assert _mean([1.0, None, 3.0]) == 2.0

    def test_mean_all_none(self):
        assert _mean([None, None]) is None

    def test_mean_rounds(self):
        assert _mean([1.0, 2.0, 2.0]) == 1.6667

    def test_max_empty(self):
        assert _max([]) is None

    def test_max_filters_none(self):
        assert _max([1.0, None, 5.0]) == 5.0

    def test_max_returns_float(self):
        v = _max([3, 1])
        assert v == 3.0 and isinstance(v, float)


# --------------------------------------------------------------------------
# ResourceSampler：参数校验
# --------------------------------------------------------------------------

class TestSamplerValidation:
    @pytest.mark.parametrize("iv", [0.0, -0.5])
    def test_bad_interval_raises(self, iv):
        with pytest.raises(ValueError, match="interval_s"):
            ResourceSampler(interval_s=iv)

    @pytest.mark.parametrize("ms", [0, -3])
    def test_bad_max_samples_raises(self, ms):
        with pytest.raises(ValueError, match="max_samples"):
            ResourceSampler(max_samples=ms)

    def test_nan_interval_raises(self):
        """NaN <= 0 是 False，老写法会让它溜进去，然后采样线程一轮就自杀。"""
        with pytest.raises(ValueError, match="interval_s"):
            ResourceSampler(interval_s=float("nan"))

    @pytest.mark.parametrize("kw", [{"interval_s": "abc"}, {"gpu_index": "x"},
                                    {"join_timeout": None}, {"max_samples": "many"}])
    def test_non_numeric_raises_value_error(self, kw):
        with pytest.raises(ValueError):
            ResourceSampler(**kw)

    def test_string_numbers_are_coerced(self):
        rs = ResourceSampler(interval_s="0.5", gpu_index="1", max_samples="3",
                             join_timeout="2")
        assert rs.interval_s == 0.5 and rs.gpu_index == 1
        assert rs.max_samples == 3 and rs._join_timeout == 2.0

    def test_max_samples_none_allowed(self):
        rs = ResourceSampler(max_samples=None)
        assert rs.max_samples is None

    def test_default_interval(self):
        assert ResourceSampler().interval_s == pytest.approx(0.2)


# --------------------------------------------------------------------------
# ResourceSampler：跑起来之后的读数
# --------------------------------------------------------------------------

class TestSamplerRun:
    def test_deterministic_cpu_cores(self):
        """wall 每步 +1.0、proc 每步 +0.1 -> 每个区间都是 0.1 核，均值=峰值=0.1。"""
        ck = _Clocks(wall_step=1.0, proc_step=0.1)
        rs = _sampler(smi=lambda gpu_index=0: None, proc_clock=ck.proc,
                      wall_clock=ck.wall, max_samples=3)
        s = rs.stop()
        assert s["cpu_cores_mean"] == 0.1
        assert s["cpu_cores_max"] == 0.1

    def test_row_count_is_max_samples_plus_two(self):
        ck = _Clocks()
        rs = _sampler(smi=lambda gpu_index=0: None, proc_clock=ck.proc,
                      wall_clock=ck.wall, max_samples=3)
        s = rs.stop()
        assert s["cpu_rows"] == 5, "start 1 行 + 每轮 1 行 x3 + stop 补 1 行终值"

    def test_wall_s_from_fake_clock(self):
        ck = _Clocks(wall_step=1.0)
        rs = _sampler(smi=lambda gpu_index=0: None, proc_clock=ck.proc,
                      wall_clock=ck.wall, max_samples=3)
        assert rs.stop()["wall_s"] == 4.0

    def test_multi_core_reads_above_one(self):
        ck = _Clocks(wall_step=1.0, proc_step=2.3)
        rs = _sampler(smi=lambda gpu_index=0: None, proc_clock=ck.proc,
                      wall_clock=ck.wall, max_samples=2)
        assert rs.stop()["cpu_cores_mean"] == 2.3, "三线程跑满时核当量 > 1，这正是想要的口径"

    def test_gpu_values_aggregated(self):
        rows = [{"gpu_util_pct": 10, "mem_used_mib": 100, "mem_total_mib": 8188},
                {"gpu_util_pct": 30, "mem_used_mib": 200, "mem_total_mib": 8188}]
        it = iter(rows + [rows[-1]] * 10)
        rs = _sampler(smi=lambda gpu_index=0: next(it), max_samples=2)
        s = rs.stop()
        assert s["gpu_samples"] == 2 and s["gpu_available"] is True
        assert s["gpu_util_mean_pct"] == 20.0 and s["gpu_util_max_pct"] == 30.0
        assert s["gpu_mem_used_mean_mib"] == 150.0 and s["gpu_mem_used_max_mib"] == 200.0
        assert s["gpu_mem_total_mib"] == 8188

    def test_gpu_index_forwarded_to_smi(self):
        seen = []

        def fake(gpu_index=0):
            seen.append(gpu_index)
            return None

        rs = _sampler(smi=fake, gpu_index=2, max_samples=2)
        rs.stop()
        assert seen and all(x == 2 for x in seen)

    def test_smi_none_means_unavailable(self):
        rs = _sampler(smi=lambda gpu_index=0: None, max_samples=3)
        s = rs.stop()
        assert s["gpu_available"] is False and s["gpu_samples"] == 0
        assert s["gpu_util_mean_pct"] is None and s["gpu_mem_total_mib"] is None
        assert s["cpu_rows"] == 5, "GPU 读不到时 CPU 那半边照常记"

    def test_smi_raising_does_not_kill_sampler(self):
        def boom(gpu_index=0):
            raise RuntimeError("nvidia-smi 炸了")

        rs = _sampler(smi=boom, max_samples=3)
        s = rs.stop()
        assert s["gpu_available"] is False and s["cpu_rows"] == 5

    def test_na_rows_count_as_samples_but_no_values(self):
        na = {"gpu_util_pct": None, "mem_used_mib": None, "mem_total_mib": None}
        rs = _sampler(smi=lambda gpu_index=0: dict(na), max_samples=2)
        s = rs.stop()
        assert s["gpu_available"] is True and s["gpu_samples"] == 2
        assert s["gpu_util_mean_pct"] is None, "「命令通了但字段是 [N/A]」：有采样点、无有效值"

    def test_none_fields_mixed(self):
        rows = [{"gpu_util_pct": None, "mem_used_mib": 100, "mem_total_mib": None},
                {"gpu_util_pct": 40, "mem_used_mib": None, "mem_total_mib": 8188}]
        it = iter(rows + [rows[-1]] * 10)
        rs = _sampler(smi=lambda gpu_index=0: next(it), max_samples=2)
        s = rs.stop()
        assert s["gpu_util_mean_pct"] == 40.0 and s["gpu_util_max_pct"] == 40.0
        assert s["gpu_mem_used_mean_mib"] == 100.0
        assert s["gpu_mem_total_mib"] == 8188

    def test_start_twice_is_noop(self):
        ck = _Clocks()
        rs = _sampler(smi=lambda gpu_index=0: None, proc_clock=ck.proc,
                      wall_clock=ck.wall, max_samples=1)
        before = len(rs._rows)
        assert rs.start() is rs
        assert len(rs._rows) == before, "重复 start 不能再补一行起点读数，否则均值分母被撑大"

    def test_stop_is_idempotent(self):
        rs = _sampler(smi=lambda gpu_index=0: None, max_samples=2)
        s1 = rs.stop()
        s2 = rs.stop()
        assert s1 == s2

    def test_stop_joins_thread(self):
        rs = _sampler(smi=lambda gpu_index=0: None, max_samples=2)
        rs.stop()
        assert rs._thread.is_alive() is False

    def test_summary_before_start_is_empty(self):
        rs = ResourceSampler(max_samples=2)
        s = rs.summary()
        assert s["cpu_rows"] == 0 and s["wall_s"] is None
        assert s["cpu_cores_mean"] is None and s["cpu_cores_max"] is None
        assert s["gpu_samples"] == 0 and s["gpu_available"] is False

    def test_summary_keys(self):
        rs = _sampler(smi=lambda gpu_index=0: None, max_samples=1)
        s = rs.stop()
        for k in ("interval_s", "gpu_index", "max_samples", "wall_s", "cpu_rows",
                  "cpu_cores_mean", "cpu_cores_max", "gpu_samples", "gpu_available",
                  "gpu_util_mean_pct", "gpu_util_max_pct", "gpu_mem_used_mean_mib",
                  "gpu_mem_used_max_mib", "gpu_mem_total_mib"):
            assert k in s, k

    def test_config_echoed_in_summary(self):
        rs = _sampler(smi=lambda gpu_index=0: None, interval_s=0.25, gpu_index=1,
                      max_samples=2)
        s = rs.stop()
        assert s["interval_s"] == 0.25 and s["gpu_index"] == 1 and s["max_samples"] == 2

    def test_context_manager_stops(self):
        gate = threading.Event()
        with ResourceSampler(interval_s=0.01, smi=lambda gpu_index=0: None,
                             sleeper=lambda s: gate.wait(0.05), max_samples=2) as rs:
            time.sleep(0.02)
            assert rs.summary()["cpu_rows"] >= 1
        gate.set()
        assert rs._thread.is_alive() is False
        assert rs.summary()["cpu_rows"] >= 2

    def test_summary_midway_keeps_thread_running(self):
        gate = threading.Event()
        rs = ResourceSampler(interval_s=0.2, smi=lambda gpu_index=0: None,
                             sleeper=lambda s: gate.wait(5.0), max_samples=5)
        rs.start()
        time.sleep(0.05)
        rs.summary()
        assert rs._thread.is_alive(), "summary() 只是读数，不能顺手把采样停了"
        gate.set()
        rs.stop()

    def test_thread_is_daemon_and_named(self):
        rs = _sampler(smi=lambda gpu_index=0: None, max_samples=1)
        assert rs._thread.daemon is True and rs._thread.name == "res-sampler"
        rs.stop()

    def test_sleep_receives_interval(self):
        got = []
        rs = ResourceSampler(interval_s=0.37, smi=lambda gpu_index=0: None,
                             sleeper=lambda s: got.append(s), max_samples=2)
        rs.start()
        rs._thread.join(timeout=10.0)
        rs.stop()
        assert got == [0.37], "采满 2 轮只 sleep 1 次：最后一轮直接 return，不多等一个间隔"

    def test_stop_before_max_samples_still_returns(self):
        """没采满就 stop：靠 _stop 事件退出，不能卡住。"""
        gate = threading.Event()
        rs = ResourceSampler(interval_s=0.2, smi=lambda gpu_index=0: None,
                             sleeper=lambda s: gate.wait(5.0), max_samples=100)
        rs.start()
        time.sleep(0.05)
        gate.set()
        t0 = time.perf_counter()
        s = rs.stop()
        assert time.perf_counter() - t0 < 3.0
        assert s["cpu_rows"] >= 2