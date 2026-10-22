"""W4① 补丁：压测期间的资源占用采样（CPU 核数当量 + GPU 利用率/显存）。

为什么自己写而不引 psutil
------------------------
本仓库的依赖边界是「核心库只用 numpy / opencv / pyyaml / 标准库」，CI 连 torch 都不装。
为一个采样线程引入 psutil 不值得，而要的两个数标准库都拿得到：

- **CPU**：``time.process_time()`` 是**进程级**（所有线程之和）的 CPU 时间，
  Δprocess_time / Δwall 就是「用了几个核」。三线程流水线跑满时这个数会 > 1，
  这正是想要的口径 —— 报「CPU 占用 230%」不如报「2.30 核当量」，后者不随机器核数失真。
- **GPU**：``nvidia-smi --query-gpu`` 走子进程，默认 200 ms 一次；拿不到就返回 None，
  绝不让采样把压测本身搞崩。

**采样器会扰动被测对象**：每 200 ms 起一个 nvidia-smi 子进程是要吃 CPU 的。
所以它默认关闭（``bench_pipeline.py --sample-resources`` 才开），开了的那一轮
单独报一组数，不和没开的那轮混在一张表里比。扰动实测多大见 docs/pipeline.md §6.5。
"""
from __future__ import annotations

import subprocess
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Sequence

__all__ = ["SMI_FIELDS", "parse_smi", "smi_sample", "cpu_cores", "ResourceSampler"]

#: nvidia-smi 的查询字段。只要三个，多要一个就多一份解析失败的概率。
SMI_FIELDS = "utilization.gpu,memory.used,memory.total"


def _to_int(tok: str) -> Optional[int]:
    """``" 2246 MiB"`` / ``"[N/A]"`` / ``"37 %"`` -> int / None。

    nvidia-smi 在部分驱动与虚拟化环境里会返回 ``[N/A]`` 或 ``[Not Supported]``，
    这种格子必须是 None 而不是 0 —— 0% 利用率和「读不到」是两件完全不同的事，
    混成 0 会让报告看起来像「GPU 全程空闲」。
    """
    s = tok.strip()
    if not s or s.startswith("["):
        return None
    num = ""
    for ch in s:
        if ch.isdigit():
            num += ch
        else:
            break
    return int(num) if num else None


def parse_smi(text: Optional[str]) -> Optional[Dict[str, Optional[int]]]:
    """nvidia-smi 的 CSV 输出 -> 字典。解析不出来一律返回 None，不抛。

    只取第一行有效数据：查询里没带 ``--id`` 时多卡机器会返回多行，
    压测只关心当前这块卡，调用方用 ``gpu_index`` 指定。
    """
    if not text:
        return None
    for line in str(text).splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p for p in line.split(",")]
        if len(parts) < 3:
            return None
        return {
            "gpu_util_pct": _to_int(parts[0]),
            "mem_used_mib": _to_int(parts[1]),
            "mem_total_mib": _to_int(parts[2]),
        }
    return None


def _default_runner(cmd: Sequence[str], timeout: float) -> Any:
    return subprocess.run(list(cmd), capture_output=True, text=True, timeout=timeout)


def smi_sample(binary: str = "nvidia-smi", gpu_index: int = 0, timeout: float = 2.0,
               runner: Optional[Callable[..., Any]] = None) -> Optional[Dict[str, Optional[int]]]:
    """采一次 GPU 利用率与显存。任何失败都返回 None。

    ``runner`` 可注入，单测里塞个假对象就能覆盖「nvidia-smi 不存在 / 超时 /
    返回非零 / 输出是垃圾」四条路径，不需要机器上真有 GPU。
    """
    run = runner or _default_runner
    cmd = [binary, f"--id={int(gpu_index)}", f"--query-gpu={SMI_FIELDS}",
           "--format=csv,noheader,nounits"]
    try:
        cp = run(cmd, timeout)
    except (FileNotFoundError, OSError, subprocess.SubprocessError, ValueError):
        return None
    if cp is None or getattr(cp, "returncode", 1) != 0:
        return None
    return parse_smi(getattr(cp, "stdout", None))


def cpu_cores(proc_a: float, proc_b: float, wall_s: float) -> Optional[float]:
    """两次 ``process_time()`` 读数 + 墙钟差 -> 这段时间平均用了几个核。

    返回值可以 > 1（多线程），也可以 < 1（大部分时间在等 GPU / 等磁盘）。
    ``wall_s <= 0`` 或读数倒退（跨进程时钟重置）返回 None，不返回 0。
    """
    if wall_s is None or wall_s <= 0:
        return None
    d = proc_b - proc_a
    if d < 0:
        return None
    return round(d / wall_s, 4)


def _mean(xs: Sequence[float]) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 4) if xs else None


def _max(xs: Sequence[float]) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    return round(float(max(xs)), 4) if xs else None


class ResourceSampler:
    """后台采样线程：每 ``interval_s`` 采一次 GPU，同时记 CPU 时间读数。

    用法（也是 ``bench_pipeline.py`` 里的用法）::

        with ResourceSampler(interval_s=0.2) as rs:
            results, rep = Pipeline(stages).run(frames)
        rep["resources"] = rs.summary()

    三条设计约束：

    1. **不抛异常**。采样是压测的附属品，nvidia-smi 挂了、超时了、输出格式变了，
       都只能让 ``gpu_available=False``，不能让一轮 20 分钟的压测白跑。
    2. **stop() 之后 summary() 才是终值**。中途调 summary() 拿到的是到当下为止的读数，
       线程还在跑；所以 ``__exit__`` 里先 stop。
    3. **CPU 报两个口径**：整轮均值（``cpu_cores_mean``，= 全程 Δprocess_time / Δwall）
       与逐区间峰值（``cpu_cores_max``）。只看均值会把「三线程同时忙」抹平成
       「一个核一直在忙」，只看峰值又会被一次 GC 毛刺带跑。
    """

    def __init__(self, interval_s: float = 0.2, gpu_index: int = 0,
                 smi: Callable[..., Optional[Dict[str, Optional[int]]]] = smi_sample,
                 proc_clock: Callable[[], float] = time.process_time,
                 wall_clock: Callable[[], float] = time.perf_counter,
                 sleeper: Optional[Callable[[float], None]] = None,
                 join_timeout: float = 5.0, max_samples: Optional[int] = None):
        # 先收敛类型再校验，两个坑：
        # 1) CLI/配置传进来的可能是字符串，"0.2" <= 0 直接 TypeError，报错信息还很难懂；
        # 2) float("nan") <= 0 是 False —— NaN 能溜过老写法，接着 Event.wait(nan) 抛
        #    ValueError 被 _loop 的兜底 except 吞掉，采样线程一轮就悄悄退出，
        #    报告里只剩 gpu_available=False，看起来像"这台机器没显卡"，其实是被自己噎死的。
        #    所以判据写成 not interval_s > 0，NaN 一起挡掉。
        try:
            interval_s = float(interval_s)
            gpu_index = int(gpu_index)
            join_timeout = float(join_timeout)
        except (TypeError, ValueError) as e:
            raise ValueError(f"interval_s / gpu_index / join_timeout 必须是数字：{e}") from e
        if not interval_s > 0:
            raise ValueError(f"interval_s 必须是 > 0 的实数，收到 {interval_s!r}")
        if max_samples is not None:
            try:
                max_samples = int(max_samples)
            except (TypeError, ValueError) as e:
                raise ValueError(f"max_samples 必须是整数或 None：{e}") from e
            if max_samples <= 0:
                raise ValueError(f"max_samples 必须 > 0 或 None，收到 {max_samples}")
        self.interval_s = interval_s
        self.gpu_index = gpu_index
        self._smi = smi
        self._proc = proc_clock
        self._wall = wall_clock
        self._join_timeout = join_timeout
        self._lock = threading.Lock()
        self._stop = threading.Event()
        # sleeper 可注入：单测里给一个"数到 N 次就 set stop"的假 sleep，
        # 不用真的等 200 ms，也能确定性地控制采到几个点。
        self._sleep = sleeper or (lambda s: self._stop.wait(s))
        self._rows: List[Any] = []      # [(wall, proc)]
        self._gpu: List[Dict[str, Any]] = []
        self._thread: Optional[threading.Thread] = None
        self._started = False
        #: 采满这么多轮就自己收工。生产上用来给采样封顶，单测里用来
        #: 不靠真等 200 ms 也能确定性地控制采到几个点（配 sleeper 注入）。
        self.max_samples = None if max_samples is None else int(max_samples)

    # -- 生命周期 ----------------------------------------------------------
    def start(self) -> "ResourceSampler":
        if self._started:
            return self
        self._started = True
        self._stop.clear()
        self._rows.append((self._wall(), self._proc()))
        self._thread = threading.Thread(target=self._loop, name="res-sampler", daemon=True)
        self._thread.start()
        return self

    def _loop(self) -> None:
        iters = 0
        while not self._stop.is_set():
            g = None
            try:
                g = self._smi(gpu_index=self.gpu_index)
            except BaseException:  # noqa: BLE001 - 采样绝不能掀翻压测
                g = None
            row = (self._wall(), self._proc())
            with self._lock:
                if g is not None:
                    self._gpu.append(g)
                self._rows.append(row)
            iters += 1
            # 计数按"循环轮数"记，不按"采到几个 GPU 点"记：nvidia-smi 全程失败时
            # _gpu 永远不涨，按它计数就死循环了。
            if self.max_samples is not None and iters >= self.max_samples:
                return
            try:
                self._sleep(self.interval_s)
            except BaseException:  # noqa: BLE001
                return

    def stop(self) -> Dict[str, Any]:
        """停线程并把终值读数补上，返回 summary()。可重复调用。"""
        self._stop.set()
        th = self._thread
        if th is not None and th.is_alive():
            th.join(timeout=self._join_timeout)
        if self._started:
            with self._lock:
                self._rows.append((self._wall(), self._proc()))
            self._started = False
        return self.summary()

    def __enter__(self) -> "ResourceSampler":
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.stop()

    # -- 读数 --------------------------------------------------------------
    def summary(self) -> Dict[str, Any]:
        with self._lock:
            rows = list(self._rows)
            gpu = list(self._gpu)
        wall = None
        cpu_mean = None
        cpu_max = None
        if len(rows) >= 2:
            wall = rows[-1][0] - rows[0][0]
            cpu_mean = cpu_cores(rows[0][1], rows[-1][1], wall)
            per = [cpu_cores(rows[i][1], rows[i + 1][1], rows[i + 1][0] - rows[i][0])
                   for i in range(len(rows) - 1)]
            cpu_max = _max([x for x in per if x is not None])
        util = [g.get("gpu_util_pct") for g in gpu]
        used = [g.get("mem_used_mib") for g in gpu]
        total = [g.get("mem_total_mib") for g in gpu if g.get("mem_total_mib") is not None]
        return {
            "interval_s": self.interval_s,
            "gpu_index": self.gpu_index,
            "max_samples": self.max_samples,
            "wall_s": round(wall, 4) if wall is not None else None,
            "cpu_rows": len(rows),
            "cpu_cores_mean": cpu_mean,
            "cpu_cores_max": cpu_max,
            "gpu_samples": len(gpu),
            "gpu_available": bool(gpu),
            "gpu_util_mean_pct": _mean(util),
            "gpu_util_max_pct": _max(util),
            "gpu_mem_used_mean_mib": _mean(used),
            "gpu_mem_used_max_mib": _max(used),
            "gpu_mem_total_mib": total[0] if total else None,
        }
