# W4② C++ 推理服务：施工单

一句话：**现在还写不了，本机缺三样**（MSVC 编译器 / CUDA 头与 import lib / TensorRT 的 import lib）。
原来缺五样，今晚已经补掉两样：**CMake + Ninja 装好了**（§2 S1），**TensorRT 头文件也拉下来了**（§2 S4）。
这份文档把"缺什么、怎么装、装完做什么、拿什么判据验收"钉死，工具链到位当天可以照着走，不用再调研一遍。

另外 §7 / §8 是**不需要编译器就能先做掉的部分**，已经做完了：三个 engine 的 I/O 契约（tensor 名 / shape /
dtype / 激活内存）逐个问出来了，计划要用的 TensorRT C++ 符号也在 v10.16 的头里逐个核对过。
顺手踩到一个会让下次开工白绕一圈的坑：**`.engine` 文件前面有 844 字节的 ultralytics 头，不是裸流**（§7.1）。

**实测与计划的边界**（本仓库的老规矩，不混着写）：

- 实测：§1 全部（每条给命令）、§2 里的 winget 包 ID 与版本号、TensorRT GitHub `v10.16` tag 与
  `include/` 的 14 个头文件（**已克隆到 `_vendor\`**）、pip `tensorrt_libs` 里 **11 个 DLL / 0 个 `.lib`**、
  torch 自带 `cudart64_13.dll`、**§7 的 engine I/O 契约与激活内存**（三个 engine 逐个 tensor 问出来的）、
  **§8 的 API 核对**（逐个符号 grep 头文件，带行号）。
- 计划 / 未验证：§2 S5(b) 从 DLL 合成 import lib 的做法、所有体积与耗时估算、conda 装 CUDA 头的备选路径。
  未验证的部分在正文里都单独标了。

---

## 1. 本机现状（2026-10-23 06:20 实测，11:35 复核）

| 组件 | 检查命令 | 结果 | 结论 |
| --- | --- | --- | --- |
| MSVC / VS | `Test-Path "C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"` | `False`；`C:\Program Files*\Microsoft Visual Studio` 目录不存在 | **缺**，没有 `cl.exe` 就没有 C++ |
| CMake / Ninja | `where.exe cmake ninja`；`Test-Path W:\Anaconda\Library\bin\cmake.exe` | 全部 `False` | 当时缺，**06:18 已装好**（§2 S1） |
| CUDA Toolkit | `Get-ChildItem "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"` | 目录不存在 | **缺**（没有 `nvcc`，也没有 `cuda_runtime.h`） |
| CUDA 头文件 | 11:35 复核：5 个候选路径递归搜 `cuda_runtime.h` —— `C:\Program Files\NVIDIA GPU Computing Toolkit`（目录不存在）、`C:\Program Files\NVIDIA Corporation`、`W:\Anaconda\Library\include`、`W:\Anaconda\envs\induvision\Library\include`、torch 的 `include\` | **5 处全 0 命中** | **缺**，只能按 S3 补 |
| CUDA 运行时 | `Get-ChildItem ...\torch\lib -Filter cu*.dll` | `cudart64_13.dll` 0.40 MB 在 | **不缺** —— torch 把 CUDA 13 运行时带进来了 |
| TensorRT python | `import tensorrt; tensorrt.__version__` | `10.16.1.11` | 推理能跑（W3b/W4① 就是它） |
| TensorRT 头文件 | 递归搜 `NvInfer.h` | 0 命中 | **缺** |
| TensorRT import lib | `tensorrt_libs\*.lib` 计数 | **0**（同目录 11 个 DLL，`nvinfer_10.dll` 358.40 MB） | **缺**，MSVC 链接必须要它 |
| winget | `winget --version` | `v1.29.380` | 可用，装东西走 winget |
| 磁盘 | `(Get-PSDrive C).Free/1GB` | **176.90 GB**（W 盘 284.90、G 盘 159.80；11:35 复核，06:20 那次是 177.7） | 装得下，不是瓶颈 |
| 内存 | `(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory` | **15.22 GB** | 装 VS / CUDA 期间别同时跑压测 |
| 编译器（11:35 复核） | `Get-Command nvcc,cl,link,dumpbin,gcc,g++` | **6 个全 NOT FOUND** | 见 S2。`conda` 在 `W:\Anaconda\Library\bin\conda.bat` |
| CUDA 头 · pip 路 | `pip download nvidia-cuda-runtime-cu13 --no-deps` | **0.0.1 的 sdist 空壳，落盘 0.00 MB**，里头没有任何头文件 | **死路**，见 S3 |
| CUDA 头 · conda 路 | `conda search --subdir win-64 -c conda-forge cuda-cudart-dev --info` | **13.4.92 命中，22 KB**（metapackage，实体在依赖 `cuda-cudart-dev_win-64`） | 可行，**实体包大小未验**，见 S3 |

关键的一条是 **cudart 运行时不缺，缺的只是头和 import lib**。它决定了下面 §3 的取舍：
本项目的 C++ 版**最小实现不需要 `nvcc`**，`cudaWarpAffine` / `cudaMemcpyAsync` / stream + event
全是 cudart 的 C API，编译期只要 `cuda_runtime.h` + `cudart.lib`。
只有要写自定义 kernel（把 letterbox + 归一化融成一个 kernel、掩码解码上 GPU）时才需要 nvcc。
装 CUDA Toolkit 是为了头和 lib，**不是为了编译器**。

**这条在 11:35 从推断升级成实测**：grep TRT v10.16 的 14 个头，对 CUDA 的 `#include` **只有 1 处**——
`NvInferRuntimeBase.h:24 #include <cuda_runtime_api.h>`；`cudaStream_t` / `cudaError_t` 在头里出现
**25 次**，全部由这一个 include 提供。结论：**S3 不需要 4 GB 的 Toolkit**，只要 cudart 的头和 lib。

---

## 2. 安装清单（按顺序，S1 最轻，S2/S3 是大头）

> **2026-10-23 15:00 实测更新：S2/S3/S5 全部完成，里程碑 1 已验收。**
> 过程里踩的坑比预想多，全部记在这里，复现时照抄：
>
> 1. **S2 的 winget 在线装必死锁（退出码 5008）**：channel 清单把安装器钉在
>    `3.14.2094`，而 feed 的 `latestinstaller.json` 要求 `4.11.24`，引导器自更新
>    比对失败（日志特征："Latest installer feed version ... greater than self update
>    min version ... Bootstrapper failed with known error"）。**解法**：先跑
>    `https://aka.ms/vs/install/latest/vs_Setup.exe --quiet` 把 VS Installer 本体装到
>    4.11.24（实测 2m39s），再跑 Build Tools 引导器就跳过自更新直接装
>    （19 分钟，MSVC 14.44.35207 + Win11 SDK 10.0.26100.0）。
> 2. **`--update --offline` 路不通**：手动下载的 OPC 在 Authenticode 校验挂
>    0x800b0003（"Unable to verify the certificate: NotSupported"，证书链验证需要
>    联网拉 CRL/AIA，本机网络环境下拿不到）。
> 3. **vcvars64.bat 不能用**：它内部要跑 `reg.exe`，被 WorkBuddy 的命令黑名单拦。
>    解法见 `cpp\build.bat`——不调 vcvars，直接显式设 `PATH/INCLUDE/LIB`
>    （MSVC bin + `Windows Kits\10\bin\<ver>\x64` 的 rc/mt + SDK 的 ucrt/um）。
> 4. **conda cudart 的头是拍平的**：`crt/` 子目录不存在，但所有头都引用
>    `#include "crt/..."`。**解法**：`conda install cuda-nvcc`（下载 91.1 MB，含
>    nvcc 本体 + 完整 `crt/` 头），顺带解锁了以后写自定义 kernel 的路。
>    `cudart.lib` 在 `Library\lib\x64\`（不是 `Library\lib\`）。
> 5. **S5(b) 合成 `.lib` 实测可行**（全链唯一不确定环节，已消除）：
>    `dumpbin /exports` + `lib /def:` 从 DLL 合出 `nvinfer.lib`（2184 符号，
>    2.2 MB）与 `nvinfer_plugin.lib`，工具是 `tools/make_trt_lib.py --vsdev`。
>    坑：subprocess 列表参数过 list2cmdline 会把整条命令再套引号，cmd 收到
>    转义引号后把 vcvars 当坏命令——所以脚本里写成临时 .bat 执行；vcvars
>    banner 是 GBK，`text=True` 默认 utf-8 会崩，必须 `encoding="gbk"`。
> 6. **.bat 文件里不能有 UTF-8 中文**：cmd 按 GBK 解析，`setlocal` 都会被拆碎。
>    `cpp\build.bat` 全 ASCII，中文路径用 `%~dp0..` 相对路径表达。
>
> **里程碑 1 验收**（`cpp\src\infer_one.cpp` + `engine_header.cpp`，对拍工具
> `tools/check_cpp_infer.py`）：844 字节头剥离偏移 848 与 Python 一致；三档 engine
> 的 I/O 契约自检全对；**fp16 上 C++ 与 Python 两个运行时同输入同 engine 输出
> 逐位相等（max_abs = 0，cosine = 1.0）**；单帧 H2D+enqueueV3+D2H+sync
> = 3.720 ms（不含前后处理，10 iter 均值）。

### S1 CMake + Ninja —— **已完成（2026-10-23 06:18 实测）**

走的是最轻的一条：pip 的 `cmake` / `ninja` wheel（零依赖，不动 conda 的包图，不需要管理员），
16 秒装完，落在 `W:\Anaconda\Scripts\`：

```powershell
W:\Anaconda\python.exe -m pip install cmake ninja
cmake --version   # -> cmake version 4.4.4
ninja --version   # -> 1.13.2.git.kitware.jobserver-pipe-1
```

版本与 winget 上的 `Kitware.CMake` 4.4.4 / `Ninja-build.Ninja` 1.13.2 一致（两个 ID 本机已验证能解析），
所以不必再走 winget 装一遍。注意 `W:\Anaconda\Scripts` 得在 PATH 里，或者构建时用全路径。

### S2 VS 2022 Build Tools + C++ 工作负载（管理员，约 3-7 GB，10-25 min，大头是下载）

```powershell
winget install --id Microsoft.VisualStudio.2022.BuildTools --exact --accept-source-agreements `
  --override "--quiet --wait --norestart --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
# 验证
& "C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe" -products * -property installationPath
```

winget 上当前版本 **17.14.42**。装完从"x64 Native Tools Command Prompt for VS 2022"进去，`cl` 才在 PATH 里。

**想少下一半**：把 `--includeRecommended` 去掉，只点编译器和 SDK 两个组件——

```powershell
winget install --id Microsoft.VisualStudio.2022.BuildTools --exact --accept-source-agreements `
  --override "--quiet --wait --norestart --add Microsoft.VisualStudio.Component.VC.Tools.x86.x64 --add Microsoft.VisualStudio.Component.Windows11SDK.26100"
```

体积从 3-7 GB 降到 **约 2-2.5 GB**（**这是估算，本机没实测过**：MSVC v143 工具集约 1.5 GB +
Win11 SDK 约 0.8 GB）。省掉的是 CMake/Ninja 的 VS 集成、调试器、ATL —— 本项目用不到，
CMake 和 Ninja 已经用 pip 装好了（S1）。
**但 `VC.Tools.x86.x64` 这个组件不能省**：`cl` / `link` / `lib` / **`dumpbin`** 都在里头，
S5(b) 从 DLL 合成 `.lib` 全靠 `dumpbin` 和 `lib`。

### S3 cudart 的头与 lib —— **原计划的 4 GB CUDA Toolkit 可以省掉**

原计划是 `winget install Nvidia.CUDA`（约 4 GB）。2026-10-23 11:35 实测之后改了主意，依据三条：

1. **TRT v10.16 的头对 CUDA 只有一处依赖**（实测 grep 14 个头，全仓仅一行）：
   `NvInferRuntimeBase.h:24 #include <cuda_runtime_api.h>`；`cudaStream_t` / `cudaError_t`
   出现 25 次，全部来自它。→ 编译期只要 `cuda_runtime_api.h` 那一族头 + `cudart.lib`，
   Toolkit 里其余上千个头文件、几百 MB 的 samples、nvcc、nsight 全用不上。
2. **运行期不缺**：torch 已经带了 `cudart64_13.dll`（§1 表）。
3. **不写自定义 kernel 就不需要 nvcc**（§1 结尾）。

所以 S3 的目标从「装 CUDA Toolkit」缩成「拿到 cudart 的头和 lib」。两条路都实测过了：

| 路 | 实测结果（2026-10-23 11:35） | 结论 |
| --- | --- | --- |
| pip `nvidia-cuda-runtime-cu13` | `pip download --no-deps` 拿到的是 **0.0.1 的 sdist 空壳，落盘 0.00 MB**，无任何头文件 | **死路**。NVIDIA 在 PyPI 上没给 Windows 发实体 wheel（Linux 侧才有） |
| conda `cuda-cudart-dev` | `conda search --subdir win-64 -c conda-forge --info` 命中 **13.4.92**，包体 **22 KB**，依赖 `cuda-cudart-dev_win-64 13.4.92` + `cuda-version >=13.4,<13.5.0a0` | **可行**。22 KB 是 metapackage，**实体包大小、是否含 `cudart.lib` 都还没验** |

conda 那条，装之前先 `--dry-run` 看实体包多大：

```powershell
conda install -n induvision -c conda-forge "cuda-cudart-dev=13.4.92" --dry-run
# 装完必须验这三样，缺一样就回退到 Toolkit
Test-Path "W:\Anaconda\envs\induvision\Library\include\cuda_runtime_api.h"
Get-ChildItem "W:\Anaconda\envs\induvision\Library\lib" -Filter "cudart*.lib"
Get-ChildItem "W:\Anaconda\envs\induvision\Library\bin" -Filter "cudart64*.dll"
```

**一个没验的顾虑**：`W:\Anaconda` 是全机安装，往 `envs\induvision` 写文件**可能要管理员权限**。
真弹 UAC，就建到用户目录去，只把路径传给 CMake，不动原环境：

```powershell
conda create -p "$env:USERPROFILE\envs\trtdev" -c conda-forge "cuda-cudart-dev=13.4.92" -y
# CMake 侧：-DCUDAToolkit_INCLUDE_DIRS=...\Library\include -DCUDAToolkit_LIBRARY=...\lib\cudart.lib
```

版本口径：conda 上是 **13.4.92**，torch 是 `cu130`，TensorRT 10.16 支持 CUDA 13.x，
**同大版本内 ABI 稳定**，13.4 的头配 13.0 的运行时正常能用。真在链接期报符号找不到，再退回重的：

```powershell
winget install --id Nvidia.CUDA --exact --accept-source-agreements   # 约 4 GB，需管理员
Test-Path "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\include\cuda_runtime_api.h"
```

winget 只给最新版 **13.4**，不能选小版本；要 13.0.x 得去 NVIDIA 归档站手动下 exe。

### S4 TensorRT 头文件（不用登录 NVIDIA 开发者站）

TensorRT 从 10.x 起在 GitHub 开源，头文件就在仓库里。本机已核实：tag 列表里有 `v10.16`，
`include/` 下 **14 个文件**（`NvInfer.h` / `NvInferRuntime.h` / `NvInferVersion.h` / `NvInferPlugin.h` 等）。

**这一步也已经做完了**（2026-10-23 06:20 实测，稀疏克隆只取 `include/`，13.5 s）：

```powershell
git clone --depth 1 --filter=blob:none --sparse --branch v10.16 `
    https://github.com/NVIDIA/TensorRT "C:\Users\11646\Documents\ChatGPT\简历\_vendor\TensorRT-v10.16"
cd "C:\Users\11646\Documents\ChatGPT\简历\_vendor\TensorRT-v10.16"; git sparse-checkout set include
```

拿到 `include\` 下 **14 个头 + `impl\`**（`NvInfer.h` 439 KB、`NvInferRuntime.h` 234 KB）。
放在仓库**外面**的 `_vendor\`，不入库（本仓库的入库边界由 CI 守着）。

**一个要记下的版本差**：头文件里 `NvInferVersion.h` 写的是 **10.16.0.72（GA）**，
pip 装的 DLL 是 **10.16.1.11** —— major.minor 一致，patch/build 差一档。
正常情况下 C++ 侧按 10.16 的 API 编译、运行期加载 10.16.1 的 DLL 没问题；
真撞上运行期版本断言（TRT 会拿 `getInferLibVersion()` 和头的宏比），
就去 NVIDIA 归档站下 10.16.1 的 Windows zip（要开发者账号登录，顺带解决 S5 的 `.lib`）。

### S5 TensorRT import lib（这一步是唯一有不确定性的）

pip 的 `tensorrt_libs` 里 **0 个 `.lib`**，MSVC 链接期过不去。两条路：

**(a) 官方 zip**（省事，但要 NVIDIA 开发者账号登录下载）：TensorRT 10.16 GA for Windows 的 zip 里
`lib\nvinfer.lib` 直接能用，`include\` 也能一起拿到（那样 S4 都可以省）。

**(b) 从 DLL 合成**（全离线，**未验证**，S2 装完第一件事就是试它）：

```bat
:: 先拿 41 MB 的 nvinfer_plugin_10.dll 试手，别拿 358 MB 的 nvinfer_10.dll 试
dumpbin /exports nvinfer_plugin_10.dll > nvinfer_plugin.def
lib /def:nvinfer_plugin.def /machine:x64 /out:nvinfer_plugin.lib
:: 通了再对 nvinfer_10.dll 做一遍
```

生成的 `.def` 头部是 `LIBRARY "nvinfer_10.dll"`，运行时加载器正好去找这个文件名，
和 pip 装法的 DLL 命名一致，不用改名。
`nvonnxparser_10.dll` **不需要合成** —— engine 在 W3b 已经用 Python 建好了，C++ 侧只反序列化。

---

## 3. 装完做什么（按性价比排序，对齐 [pipeline.md](pipeline.md) §9）

1. **前处理搬上 GPU**：`cudaWarpAffine` 一次做完 letterbox 的缩放 + pad。**但先看
   [gpu_preproc.md](gpu_preproc.md) 的实测分解**：那 16~19 ms 里 imdecode 占 ~15 ms、
   letterbox 链只有 ~4 ms——GPU 化 letterbox 净省 ~1.8 ms/帧，端到端流水线
   +17~28%（同轮两轮 45.4/45.7 → 53.3/58.4 FPS）。decode util 仍 1.00，瓶颈原位
   留在 imdecode 上。所以 C++ 版的真正台阶是 **nvJPEG/GPU 解码**；warpAffine 是
   顺手做掉的第一步，且 Python 版每帧多一笔 4.688 MiB D2H 拷回、C++ 版没有——
   实测收益是 C++ 版的保守下界。
2. **多 CUDA stream + event**：W4① 只有一条流，H2D / kernel / D2H 在流内串行，重叠只发生在
   "GPU 忙"与"CPU 忙"之间。多流才能让第 n 帧的 D2H 和第 n+1 帧的 H2D 真的并行。
3. **度量口径修正**：`acquire` 等待单独计时（`decode_wait_ms` vs `decode_busy_ms`），
   `bottleneck_stage` 用 busy 判；`pool_size` 默认提到 3~4（W4① 实测对吞吐无害，对报表诚实）。
4. **三线程流水线 + 环形队列 + pinned 零拷贝**：语义直接照搬 Python 版（pipeline.md §3 / §4），
   包括"每帧只 synchronize 一次""出错要把令牌还回池子""等待计数按调用记不按循环轮数记"这三条。
5. **CMake 工程**：C++17、Release、`/utf-8`（源码里有中文注释就会咬人）；`find_package(CUDAToolkit)`；
   TensorRT 的 include/lib 用 `-DTENSORRT_ROOT=...` 传进来，**不要写死路径**（CI 上路径不一样）。

**明确不要写的卖点：「绕开 GIL」。** pipeline.md §7 的 `pool_size` 消融已经证明 Python 版的瓶颈
不是 GIL（是等令牌与等磁盘），CPU 只吃到 1.65 核当量而不是 3 核。把 GIL 当 C++ 版的卖点，
等于用一个自己测过的数据打自己。卖点是第 1 条和第 2 条。

---

## 4. 验收判据（基线数字全部来自 pipeline.md §6.5，同清单同口径，一条都不松）

| 判据 | 阈值 | 基线出处 |
| --- | --- | --- |
| 与 **Python 版**对拍 | match_rate ≥ 0.98、框 IoU ≥ 0.99、掩码 IoU ≥ 0.85 | §6.3；Python 版实测掩码 IoU 0.910~0.916，C++ 版低于它就是实现错了 |
| 与 ultralytics 对拍 | 同上 | §6.3：match_rate 1.0000 / 框 IoU 1.0000 / 掩码 IoU 0.9198~0.9305 |
| 与串行逐帧一致 | `(n_det, mask_pixels, score_sum)` 500 帧全等，`n_diff = 0` | §6.4，钉的是跨帧数据竞争 |
| 吞吐 | 同轮内加速比 ≥ 1.44×，并与 Python 版**同轮并列**报绝对 FPS | §6.5.1：A 组 fp32 pinned 串行 40.66 / 流水线 63.94 FPS |
| 延迟 | P50 / P95 / **P99** | §6.5.2：Python 版只报到 max，P99 是 C++ 版要补的数 |
| 资源 | CPU 核当量 + GPU util + 整机显存，**与 Python 版同一套采样口径** | §6.5.3：串行 1.10 核 / 流水线 1.65 核；GPU util A 组 28~37% |
| 测量纪律 | 验精度与报吞吐**分两轮**；每个配置至少两轮 | §6.5.4：对拍污染吞吐 8.57%，机器噪声底 3.57% |

资源采样别在 C++ 里重写一份 —— 口径一漂，两张表就不可比了。要么照搬 `induvision/prof.py` 的算法
（`time.process_time()` 差 / 墙钟 = 核当量；`nvidia-smi --query-gpu` 200 ms 一次），
要么让 C++ 压测跑在 Python 采样器的监视下。整机显存那一列继续写死"整机"，
它和模型侧的 **18.893 MiB** 差 130 倍（§6.5.3）。

---

## 5. 回退方案

- **S2 或 S3 装不通 / 时间不够**：不写 C++ 版，也**绝不把没跑过的数写进报告和简历**。
  W4① 的 Python 版已经有完整可复现的 500 帧数据（§6.5），对外报数用它；
  报告里如实写"工具链缺口清单见本文档 §1"，这本身就是工程判断，不是减分项。
- **只有 S5 合成 `.lib` 失败**：先退回官方 zip 那条路（S5(a)）。别为了绕开链接去改
  `LoadLibrary` + `GetProcAddress` 手动取符号 —— TensorRT 的 C++ API 是类接口，函数指针那套走不通。
- **别做的事**：拿 CUDA 13.4 的头去配更老的 TRT，或者顺手把 engine 用 C++ 重建一遍。
  engine 是 W3b 的产物（fp32 65.25 / fp16 24.58 / int8 12.55 MiB），C++ 版只反序列化，
  重建一遍等于把三层精度对齐那套校验（deploy.md §5）再走一次，没有收益。

## 6. 时间估算（工具链到位之后）

| 阶段 | 内容 | 估时 |
| --- | --- | --- |
| 装机 | S1-S5，含下载与验证（**S1、S4 已完成**；剩 S2 VS Build Tools、S3 CUDA、S5 `.lib`） | 1-2 h（下载是大头） |
| 打通 | CMake 工程 + engine 反序列化 + 单帧推理，对拍 Python 版 1 张图 | 0.5 天 |
| 前处理上 GPU | `cudaWarpAffine` letterbox + 与 Python 前处理逐字节比对 | 1 天 |
| 流水线 | 三线程 + 多流 + pinned 零拷贝 + 500 帧全等校验 | 1 天 |
| 压测与报告 | 500 帧 P50/P95/P99 + 资源占用 + 与 §6.5 并列的对照表 | 0.5 天 |

合计约 **3 个工作日**。装机那 1-2 h 可以挂着不管，所以真正卡人的是 S5 的 `.lib`。

---

## 7. 接口契约（实测，2026-10-23 06:25）

这一节是把 W3b 建出来的三个 engine 用 Python 侧 TensorRT 打开、逐个 tensor 问出来的，
**C++ 版照这张表写，不用猜**。

### 7.1 第一个坑：`.engine` 文件不是裸的 TRT 序列化流

直接 `deserializeCudaEngine(整个文件)` 会报：

```
Serialization assertion header.magicTag == kEXPECTED_MAGIC_TAG failed.
Trying to load an engine created with incompatible serialization version (844 != 1953657958)
```

`844` 不是版本号，是**文件头的长度字段**。`tools/build_trt.py` 用 `write_engine_file()`
按 ultralytics 的约定把 metadata 前置写进了文件（这样 `YOLO("x.engine")` 才认得类别名）：

```
[4 字节小端长度 = 844] [844 字节 JSON metadata] [裸 TRT engine 流]
```

三档 engine 的 header 都是 **844 字节**，JSON 里有 15 个键（`names` / `imgsz` / `stride` / `task` /
`end2end` / `head` / `batch` / `channels` / `version`(8.4.183) / `date` / `description` / `author` /
`license` / `docs` / `args`）。剥掉头之后三档**全部反序列化成功**。

**Python 侧早就处理了这件事**：`induvision/trt_infer.py` 里有 `encode_header()` / `decode_header()` /
`read_engine_file()`，`read_engine_file()` 的 docstring 写得很直白 ——「给 `TrtRunner` 用：TRT 只吃纯 engine」。
W3b/W4① 的所有压测都是走它读的，所以一直没事。

**这次踩坑的是我自己**：查 I/O 契约时图省事直接 `rt.deserialize_cuda_engine(open(p,"rb").read())`，
没走 helper，于是撞上上面那个报错，第一反应是"engine 和 TRT 版本不兼容"，白绕一圈。
把报错里的 `844` 当成版本号去查是错的方向 —— **它是文件头的长度字段**，
`magicTag` 那句只是 TRT 在说"你喂我的前 4 个字节不是我的魔数"。

C++ 侧要把 `decode_header()` 的语义原样复刻（不是简化版）：

1. 前 4 字节按**小端**读成长度 `n`；
2. `n` 不满足 `0 < n <= 文件长度 - 4` → 认定**没有头**，偏移 0；
3. `[4, 4+n)` 那段 JSON 解析失败、或解析出来不是 object → 同样认定没有头，偏移 0；
4. 否则偏移 `4 + n`。

第 2、3 条那个"认不出来就当没有头"的回退必须留着：ultralytics 自己就是这么赌的
（赌 engine 的原始字节几乎不可能正好构成一段合法 JSON），两边判据不一致就会出现
"Python 能读、C++ 读不了"或者反过来的情况，那种 bug 最难查。
另一条路是让 `tools/build_trt.py` 另存一份**不带头的裸流**专供 C++（`encode_header(None)` 返回 `b""`，
现成的），代价是多一份 12~65 MB 的文件和"两份 engine 是否同一份"的疑问 —— 不推荐，复刻 20 行解析更干净。

### 7.2 三个 tensor，全静态 shape

| # | 名字 | 方向 | shape | dtype | 字节数 |
| --- | --- | --- | --- | --- | --- |
| 0 | `images` | INPUT | `(1, 3, 640, 640)` | FLOAT | 4,915,200（4.688 MiB） |
| 1 | `output0` | OUTPUT | `(1, 51, 8400)` | FLOAT | 1,713,600（1.634 MiB） |
| 2 | `output1` | OUTPUT | `(1, 32, 160, 160)` | FLOAT | 3,276,800（3.125 MiB） |

三档（fp32 / fp16 / int8）**完全一致** —— 量化只改内部层格式，不改 I/O 契约，
所以 C++ 版一份代码吃三个 engine，档位只是路径不同。

没有 `-1`，**shape 全静态**，于是不需要 `setInputShape()`，也不需要 profile 查询；
每帧只要 `setTensorAddress()` 三个地址 + `enqueueV3(stream)`。
输入 4.688 MiB H2D、输出 4.759 MiB D2H，和 pipeline.md §1 那笔"每帧 4.7 MiB"的账对得上。

### 7.3 激活内存：多流要按 context 数乘

`getDeviceMemorySizeV2()` 实测（Python 侧属性 `device_memory_size`，同一份数据）：

| 档 | engine 文件 | 激活内存（每 context） |
| --- | --- | --- |
| fp32 | 65.25 MB | **39.844 MiB** |
| fp16 | 24.58 MB | **18.286 MiB** |
| int8 | 12.55 MB | **18.409 MiB** |

**并发执行必须一个 stream 一个 `IExecutionContext`，而每个 context 要一份自己的激活内存**
（`createExecutionContextWithoutDeviceMemory()` + 手动共享那块内存的做法只在**串行复用**时安全，
两个 context 同时跑就会互相踩）。所以 §3 第 2 条"多 CUDA stream"的真实代价是
2 流 fp32 = 79.7 MiB、3 流 = 119.5 MiB 常驻显存。8 GB 的卡上无所谓，但这笔账要写进报告，
别只报吞吐不报显存 —— 这和 pipeline.md §6.5.3 的 pinned 缓冲 18.893 MiB 是**两笔不同的钱**，
不要相加也不要混着写。

`device_memory_size` 在 TRT 10.1 起已 deprecated（Python 侧会打 DeprecationWarning），
C++ 侧直接用 `getDeviceMemorySizeV2()`。

---

## 8. API 核对（实测 grep v10.16 头文件）

计划里要用的符号，逐个在 `_vendor\TensorRT-v10.16\include\` 里搜过，全部存在：

| 符号 | 所在头 | 用途 |
| --- | --- | --- |
| `IRuntime::deserializeCudaEngine(void const*, std::size_t)` | `NvInferRuntime.h:1967` | 反序列化（喂剥过头的裸流） |
| `ICudaEngine::getNbIOTensors()` / `getIOTensorName(int32_t)` | `NvInferRuntime.h:3818 / 3830` | 遍历 I/O，别写死 tensor 名 |
| `ICudaEngine::getTensorShape / getTensorIOMode / getTensorDataType` | `NvInferImpl.h` | 校验 §7.2 那张表，开机自检 |
| `ICudaEngine::getDeviceMemorySizeV2()` | `NvInferRuntime.h:3419` | §7.3 的激活内存 |
| `ICudaEngine::createExecutionContext()` | `NvInfer.h` | 每条流一个 |
| `IExecutionContext::setTensorAddress(char const*, void*)` | `NvInferImpl.h` | 绑输入/输出设备指针 |
| `IExecutionContext::enqueueV3(cudaStream_t)` | `NvInfer.h` | TRT 10 的推理入口（V2 那套已废） |
| `IExecutionContext::setInputShape()` | `NvInfer.h` | **本项目不需要**（shape 全静态） |
| `IExecutionContext::setAuxStreams()` | `NvInfer.h` | 有多流 DLA/aux 需求才用，当前用不上 |

另外两条实测：`getTensorProfileShape` 在 v10.16 的头里**搜不到**（静态 shape 用不着，
真需要动态 batch 时再查当时的 API 名）；`cudaStreamCreateWithFlags` 属于 CUDA 运行时，
不在 TRT 头里，等 S3 装完 CUDA Toolkit 才有。
