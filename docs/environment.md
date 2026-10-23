# 环境说明（conda env: induvision + W4② 的 C++ 工具链）

本文记录项目实际使用的 Python/GPU 环境与 C++ 编译工具链，所有数字均为本机实测，不是官方标称值。
C++ 那部分集中在 §7。

## 1. 环境怎么来的

没有重装 torch，而是**克隆**了本机已有的可用 CUDA 环境，省掉约 5GB 下载：

```powershell
conda create -n induvision --clone yoloworld
```

克隆后做了一处关键改动：`yoloworld` 里的 ultralytics 和 clip 是**开发版（editable）**，
源码指向 `W:\Yolo World\ultralytics-8.4.14`。两个项目共用一份源码会互相影响，
所以在 induvision 里卸载后换成 PyPI 正式版：

```powershell
W:\Anaconda\envs\induvision\python.exe -m pip uninstall -y ultralytics clip
W:\Anaconda\envs\induvision\python.exe -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple `
    ultralytics onnx onnxruntime onnxslim pytest scikit-image pandas pyyaml
```

原 `yoloworld` 环境（硕士课题 YOLO-World 用）未被改动。
`base` 环境（myimg 图像算子对拍台用，796 条测试）也未被改动，仍是 CPU 版 torch。

## 2. 使用方式

```powershell
conda activate induvision
# 或不激活，直接用绝对路径：
W:\Anaconda\envs\induvision\python.exe -X utf8 tools\env_check.py
```

环境路径：`W:\Anaconda\envs\induvision`，占用 3.83 GB。

## 3. 版本清单（实测）

| 组件 | 版本 |
| --- | --- |
| Python | 3.12.5 |
| OS | Windows 11 (10.0.26300) |
| torch | 2.13.0+cu130 |
| torch.version.cuda | 13.0 |
| torch.cuda.is_available() | True |
| torchvision | 0.28.0+cu130 |
| ultralytics | 8.4.183 |
| onnx | 1.23.2 |
| onnxruntime | 1.31.0（CPU 版，见 §6） |
| onnxslim | 0.1.98 |
| tensorrt-cu13 | 10.16.1.11（含 `_bindings` / `_libs` 同版，W3b 装，见 §6） |
| numpy | 2.5.2 |
| opencv-python | 5.0.0.93 |
| scikit-image | 0.26.0 |
| pandas | 3.0.6 |
| scipy | 1.18.1 |
| matplotlib | 3.11.2 |
| pytest | 9.1.1 |

依赖已钉在 `requirements-gpu.txt`。

## 4. 自检结果（实测，2026-10-21）

命令：`python tools/env_check.py --export`

硬件：NVIDIA GeForce RTX 4060 Laptop GPU，compute capability 8.9，显存 8.00 GiB，cuDNN 92000，驱动 616.92。

| 项目 | 结果 |
| --- | --- |
| FP32 matmul 4096x4096 | 37.85 ms/次，约 3.63 TFLOPS |
| YOLOv8s-seg 单图推理（预热后，640，device=0） | 22.76 ms |
| 推理峰值显存 | 123.0 MiB |
| 检出结果（样图 astronaut.png） | person 0.861、tie 0.598、donut 0.503，3 个分割掩码 |
| ONNX 导出 | 成功，45.40 MB，opset 17，dynamic batch |
| ONNX 输入 | `images: [batch, 3, height, width]` |
| ONNX 输出 | `output0: [batch, 116, anchors]`、`output1: [batch, 32, mask_height, mask_width]` |
| onnx.checker | 通过 |

> 这里的 **116 = 4（box）+ 80（COCO 类）+ 32（mask 系数）**，因为自检用的是 COCO 预训练权重。
> 本项目 15 类的 baseline15 导出后同一位置是 **51 = 4 + 15 + 32**，两个数不要互相印证，
> 见 [deploy.md](deploy.md) §3。

模型规模：YOLOv8s-seg 85 层、11,810,560 参数、40.1 GFLOPs。

tie / donut 是 COCO 预训练权重对这张图的误检，属正常现象，只用于验证推理链路能跑通。
完整机器可读报告见 `out/env_check.json`（不入库）。

## 5. 复现自检

```powershell
cd <项目根目录>
W:\Anaconda\envs\induvision\python.exe -X utf8 tools\env_check.py            # 版本 + GPU + 单图推理
W:\Anaconda\envs\induvision\python.exe -X utf8 tools\env_check.py --export   # 再加 ONNX 导出验证
W:\Anaconda\envs\induvision\python.exe -X utf8 tools\env_check.py --image 自己的图.jpg
```

## 6. W3b 装了什么，以及还没装的东西（W4 才需要）

TensorRT 走的是 **pip wheel**，不是官网的 Windows zip 包：

```powershell
W:\Anaconda\envs\induvision\python.exe -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple tensorrt-cu13
```

装完是三个包：`tensorrt_cu13` / `tensorrt_cu13_bindings` / `tensorrt_cu13_libs`，均 10.16.1.11。
选 cu13 而不是 cu12 是因为本环境 torch 是 `2.13.0+cu130`，CUDA 大版本要对上。

这个装法有一个直接后果，写在前面免得后面反复解释：

> **wheel 里没有 `trtexec.exe`，只有 DLL。** 所以本项目的 engine 构建与压测
> 全部走 TensorRT **Python API**（`tools/build_trt.py` 与 `tools/bench_trt.py`）：
> 自己建 Builder / Network / OnnxParser / OptimizationProfile，自己写
> `IInt8Calibrator` 子类喂真实图，自己从 engine 的 inspector 里报层格式直方图。
> 比调 trtexec 麻烦，但每一步都能拿到中间量，而且 W4 的 C++ 推理服务本来就要自己写这些。

| 工具 | 状态 | 用途 / 阶段 |
| --- | --- | --- |
| TensorRT 10.16.1.11 | **已装**（pip wheel） | FP32/FP16/INT8 三档构建与压测，W3b；实测见 [deploy.md](deploy.md) §8 |
| trtexec.exe | 未装（wheel 不带） | 不需要；对照数据用 ultralytics 自带导出交叉验证过，见 deploy.md §8.8 |
| CUDA Toolkit 12.x/13.x（nvcc） | **未装整套**；只用 conda 的 `cuda-cudart-dev` + `cuda-nvcc` 13.4.92 | 编 `preproc.cu`，W4②；为什么不装 4 GB 整套见 §7.1 |
| cuDNN | 已随 torch 自带（92000） | C++ 工程不直接链它（TRT 自带 kernel），不需要单独装 |
| Visual Studio Build Tools（C++ 工作负载） | **已装**（MSVC 14.44.35207 + Win11 SDK 26100） | `cl.exe` / `link.exe` / `dumpbin.exe`，W4② |
| CMake + Ninja | **已装**（pip，cmake 4.4.4 / ninja 1.13.2） | 组织 C++ 工程，W4② |
| OpenCV（C++） | **已装**（conda-forge 5.0.0，独立前缀 env `C:\Users\11646\cppenv`） | C++ 侧 `imdecode` / `resize`，W4②；为什么不装进 induvision 见 §7.4 |
| TensorRT import lib（`nvinfer.lib`） | **已合成**（`dumpbin /exports` + `lib /def:`，2184 符号） | MSVC 链接 TRT，W4②；见 §7.3 |

ONNX 三件套（onnx 1.23.2 / onnxruntime 1.31.0 / onnxslim 0.1.98）**已装好并在 W3a 用上**：
导出与三层精度对齐的实测见 [deploy.md](deploy.md)。

onnxruntime 装的是 **CPU 版**。刻意不装 onnxruntime-gpu：GPU 推理这条路最终由
TensorRT 承担，再装一份 onnxruntime-gpu 会带来 CUDA/cuDNN DLL 版本冲突，收益为零。
ONNX 在本项目里的角色是"导出中间格式 + 数值对齐校验"，CPU 足够。
副作用要记着：`verify_onnx.py` 第 3 层量到的 ORT 延迟是 **CPU 延迟**（providers 里没有
CUDAExecutionProvider），只能当同机基线，不能拿去和 GPU 上的 PyTorch 比后端优劣。

## 7. C++ 工具链（W4② 补装，2026-10-23）

W4② 之前本机 `cl` / `link` / `dumpbin` / `nvcc` / `gcc` / `g++` 六个命令全缺。这一节记的是
**实际装成了什么、每样为什么这么装**，逐条踩坑的完整过程在 [cpp_infer.md](cpp_infer.md) §2。

### 7.1 MSVC + Windows SDK（唯一需要管理员的一步）

winget 装 VS Build Tools 时先撞到一个死锁：`winget` 返回 **5008**，原因是它自带的 channel
钉在 `3.14.2094`、而 feed 已经到 `4.11.24`，两边互不认。绕法是先跑官方最新 Installer
（`aka.ms/vs/install/latest/vs_Setup.exe`）把 Installer 组件升到 4.11.24，再用它装
「使用 C++ 的桌面开发」工作负载。`--offline` 那条路不通：离线包要走 OPC 签名校验，
本机报 `0x800b0003`。

装完实测：**MSVC 14.44.35207** + **Windows SDK 10.0.26100.0**，`cl` / `link` / `dumpbin` 均在
`C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64`。

### 7.2 CUDA：只装头和 nvcc，不装 4 GB Toolkit

依据是实测出来的：TensorRT v10.16 的 14 个头文件里对 CUDA **只有 1 处 `#include`**
（`NvInferRuntimeBase.h:24` 的 `cuda_runtime_api.h`）。所以不需要整套 Toolkit：

```powershell
conda install -n induvision -c conda-forge cuda-cudart-dev cuda-nvcc   # 12.6 MB + 91 MB
```

`cudart.lib` 落在 `W:\Anaconda\envs\induvision\Library\lib\x64`，头在 `Library\include`。
单装 `cuda-cudart-dev` 时 `nvcc` 编不过：conda 把 `crt/` 目录拍平了，`#include <crt/host_defines.h>`
变成递归包含（**C1014**）——补 `cuda-nvcc`（带真正的 `crt/` 头）才解决。

反向的死路也记一笔：pip 的 `nvidia-cuda-runtime-cu13` 是 **0.0.1 空壳 sdist（0.00 MB）**，
里面没有任何头文件，别浪费时间去试。

### 7.3 `nvinfer.lib` 是合成的（pip wheel 里 0 个 `.lib`）

TensorRT 的 pip wheel 只有 11 个 DLL（`nvinfer_10.dll` 358 MB 等），**一个 import lib 都没有**，
而 MSVC 链接必须要 `.lib`。做法是 `tools/make_trt_lib.py`：对每个 DLL 跑 `dumpbin /exports`
取导出表，生成 `.def`，再 `lib /def:... /machine:x64 /out:nvinfer.lib`。
合成结果 **2184 个符号**，`cpp/trt_lib/nvinfer.lib`（不入库）。运行期仍由 PATH 里的
`tensorrt_libs` 目录提供 DLL。

### 7.4 OpenCV 装在独立前缀 env 里

```powershell
conda create -p C:\Users\11646\cppenv -c conda-forge libopencv     # 5.0.0
```

刻意**不装进 induvision**：那个环境的 python 是 anaconda build，从 conda-forge 装 libopencv
会把它换成 conda-forge build，等于动了正在跑训练和压测的解释器本体。独立前缀只贡献
`Library\include` 与 `Library\lib`（core500 / imgproc500 / imgcodecs500）给 C++ 链接用。

### 7.5 构建：`cpp\build.bat` 为什么自己设环境

`vcvars64.bat` 内部要调 `reg.exe`，而本机的命令白名单会把它拦掉，所以脚本不调 vcvars，
直接把 MSVC / SDK 的 `PATH` / `INCLUDE` / `LIB` 写死（路径固定在本机，换机器要改脚本头部的
几个 `set`）。两条编译期的坑：

- **`nvcc` 默认静态 CRT（/MT）**，与 CMake 的 `/MD` 混链会报一堆 `LNK4217` / `LNK2019`
  （`__imp_puts`）——加 `-Xcompiler /MD` 对齐；
- 源码里有中文注释，加 `-Xcompiler /utf-8`；`-arch=native` 只对当前卡有效，换机器要显式给
  `sm_86` 之类。

```powershell
cpp\build.bat          # nvcc 编 preproc.obj -> CMake/Ninja 出 infer_one.exe 与 bench.exe
cpp\build.bat clean    # 清掉 cpp\build
```

### 7.6 运行期 DLL

`bench.exe` 起来要能找到 `nvinfer_10.dll` / `cudart64_13.dll` / OpenCV 的 DLL。
`tools/check_cpp_bench.py::_bench_env` 会拼好 PATH（`tensorrt_libs` + `torch/lib` + `cppenv/bin`），
所以**对拍与压测一律走这个脚本**，别手工双击 exe。

## 8. 已知注意事项

1. **github.com:443 时通时不通**。下载 release 资产走 `https://gh-proxy.com/<原始 github 链接>`，
   本次 22.8 MB 权重 3.6 秒下完。`git push` 也会中招，2026-10-23 00:14 复现过一次硬失败：
   DNS 把 `github.com` 解析到 `20.205.243.166`，ping 通但 TCP 443 不通，`git push` 连试 6 次全是
   `Failed to connect to github.com:443`。排查与绕法（照抄可用）：

   ```powershell
   # 1) 先确认不是本机断网：这条通，说明只有上面那个 IP 被卡
   Test-NetConnection www.baidu.com -Port 443 -InformationLevel Quiet
   # 2) 兄弟域名解析出的 IP 是通的 -> 问题在 DNS 给的这一个地址，不在链路
   Test-NetConnection codeload.github.com -Port 443 -InformationLevel Quiet
   # 3) 用 curloptResolve 把域名钉到一个能连的 GitHub IP
   #    2026-10-23 06:31 复测：140.82.112.3 **当天连不上了**（21051 ms 连接超时），
   #    113.3 / 112.4 / 114.4 / 121.3 的 TCP 443 全 True。**IP 会失效，别把某一个当长期答案。**
   #    坑：这 4 个 IP 的 **ICMP 全部 False**（GitHub 屏蔽 ping）。判活只能看 Test-NetConnection 的
   #    TcpTestSucceeded，拿 PingSucceeded 判会误判成"整个 GitHub 挂了"。
   git -c http.https://github.com/.curloptResolve="github.com:443:140.82.113.3" push origin master
   # 4) 想长期生效（写进 .git/config，只影响本仓库，不进版本库）
   git config --local http.https://github.com/.curloptResolve "github.com:443:140.82.113.3"
   # 5) 这个 IP 哪天失效了：换上面另一个，或删掉配置恢复默认 DNS
   git config --local --unset http.https://github.com/.curloptResolve
   ```

   本仓库已按第 4 步设了 local 配置（**当前钉在 `140.82.113.3`**，2026-10-23 06:31 换的——
   原先钉的 `140.82.112.3` 那天连不上了，换完 **3 秒推完**），所以直接 `git push` 就能过。
   顺带记一条：`git push` 失败时先跑 `git status -sb`，`ahead N` 说明 commit 根本没上去，
   别把"没推成"误读成"CI 没跑"（本轮就踩过：CI 查不到记录，原因是 commit 还在本地）。
   注：本机系统代理（`127.0.0.1:7897`）当时是关着的，git 也没配 `http.proxy`，
   所以别急着往代理方向查，先量端口。
2. pip 一律加 `-i https://pypi.tuna.tsinghua.edu.cn/simple`，实测延迟 209 ms。
3. **不要用 `pip install torch` 升级本环境的 torch**：PyPI 默认是 CPU 版，会把 cu130 覆盖掉，
   `cuda.is_available()` 立刻变 False。
4. 显存只有 8 GB：训练用 YOLOv8s-seg、imgsz=640、batch 8~16 起步，不要上 l/x 型号。
5. 磁盘余量：C 盘 186.7 GB，W 盘 287.3 GB（本环境装在 W 盘）。
   **2026-10-23 11:35 复测：C 176.90 GB / W 284.90 GB / G 159.80 GB，物理内存 15.22 GB。**
   C 盘比上次记的少了 9.8 GB，**没逐项归因**；能确定的只有仓库里的 `weights\`（三档 engine
   合计 102.38 MB）和 `runs\`（压测 JSON 产物），两者都不入库，量级也不足以解释差额。
   VS Build Tools 已于 2026-10-23 装好（约 2 GB，占用见 §7.1），装之前量过 C 盘 176.90 GB。

6. **Codex 桌面 App 重启会杀掉它 shell 派生的整棵子进程树**，WMI
   `Win32_Process.Create` 脱离也保不住（2026-10-22 04:12 实测，baseline15 练到
   epoch 32 被端掉）。超过 10 分钟的活一律用 `tools/launch_task.ps1` 挂到
   Task Scheduler 上：任务由服务启动，不属于任何 App 的 job object。
7. **本机是 S0 新型待机，AC 睡眠超时出厂就是 600 秒（10 分钟）**，03:24 真进过
   一次待机。长训练前先 `powercfg /change standby-timeout-ac 0`
   （`launch_task.ps1 -HardenPower` 会做），否则随时可能被待机打断。
8. **8GB 显存要先腾干净再上卡**。腾讯视频 / 弹幕 / 迅雷缩略图这类常驻 GPU 的
   娱乐进程能占掉 2GB 以上；04:28 关掉后显存 3421→1576 MiB，baseline15 单 epoch
   从 105 s 降到 78 s（-26%），而且余量更大、TDR 风险更低。

## 9. 许可与数据合规

- 权重 `yolov8s-seg.pt` 与 Ultralytics 代码为 **AGPL-3.0**；MVTec AD 数据集为
  **CC BY-NC-SA 4.0（非商用）**。
- 因此 `.gitignore` 已排除 `weights/`、`*.pt`、`*.onnx`、`*.engine` 和 `data/`（保留 `data/README.md`）。
  仓库里只放脚本、配置、文档和自己产出的报告，不放第三方权重与数据集图片。