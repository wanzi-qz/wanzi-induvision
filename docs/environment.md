# GPU 环境说明（conda env: induvision）

本文记录项目实际使用的 Python/GPU 环境，所有数字均为本机实测，不是官方标称值。

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

## 6. 还没装的东西（W3b/W4 才需要）

| 工具 | 用途 | 阶段 |
| --- | --- | --- |
| CUDA Toolkit 12.x/13.x（nvcc） | 编译 C++/CUDA 扩展 | W4 |
| cuDNN | 已随 torch 自带（92000），单独装仅为 C++ 工程链接用 | W4 |
| TensorRT 10.x（Windows zip 版） | FP16/INT8 量化、trtexec | W3b |
| Visual Studio Build Tools（C++ 工作负载） | cl.exe 编译器 | W4 |
| CMake | 组织 C++ 工程 | W4 |

ONNX 三件套（onnx 1.23.2 / onnxruntime 1.31.0 / onnxslim 0.1.98）**已装好并在 W3a 用上**：
导出与三层精度对齐的实测见 [deploy.md](deploy.md)。

onnxruntime 装的是 **CPU 版**。刻意不装 onnxruntime-gpu：GPU 推理这条路最终由
TensorRT 承担，再装一份 onnxruntime-gpu 会带来 CUDA/cuDNN DLL 版本冲突，收益为零。
ONNX 在本项目里的角色是"导出中间格式 + 数值对齐校验"，CPU 足够。
副作用要记着：`verify_onnx.py` 第 3 层量到的 ORT 延迟是 **CPU 延迟**（providers 里没有
CUDAExecutionProvider），只能当同机基线，不能拿去和 GPU 上的 PyTorch 比后端优劣。

## 7. 已知注意事项

1. **github.com:443 间歇性不通**（本次直连失败 2 次）。下载 release 资产走
   `https://gh-proxy.com/<原始 github 链接>`，本次 22.8 MB 权重 3.6 秒下完。
2. pip 一律加 `-i https://pypi.tuna.tsinghua.edu.cn/simple`，实测延迟 209 ms。
3. **不要用 `pip install torch` 升级本环境的 torch**：PyPI 默认是 CPU 版，会把 cu130 覆盖掉，
   `cuda.is_available()` 立刻变 False。
4. 显存只有 8 GB：训练用 YOLOv8s-seg、imgsz=640、batch 8~16 起步，不要上 l/x 型号。
5. 磁盘余量：C 盘 186.7 GB，W 盘 287.3 GB（本环境装在 W 盘）。

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

## 8. 许可与数据合规

- 权重 `yolov8s-seg.pt` 与 Ultralytics 代码为 **AGPL-3.0**；MVTec AD 数据集为
  **CC BY-NC-SA 4.0（非商用）**。
- 因此 `.gitignore` 已排除 `weights/`、`*.pt`、`*.onnx`、`*.engine` 和 `data/`（保留 `data/README.md`）。
  仓库里只放脚本、配置、文档和自己产出的报告，不放第三方权重与数据集图片。