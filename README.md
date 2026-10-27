# wanzi-induvision

## 介绍

工业质检的漏检，多数不是"看不见"，而是**目标太小、样本太少**：本项目使用的 MVTec AD 里，缺陷
面积中位数只占整图 **1.54%**，两类的冒烟子集里缺陷图也仅占 **23.6%**——小目标 + 正样本稀缺，
直接决定了后面的方法选择（缺陷合成增强、按 test split 重算成绩）。落地还有第二个约束：模型要
跑进产线节拍，而 PyTorch 的推理链在算力有限的设备上留不出余量，于是有了量化与 C++ 推理服务。

本项目以 **MVTec AD 15 类工业缺陷（5354 张图）**为对象，跑通一条"训练 → 量化 → 端侧推理服务"的
完整闭环：把官方掩码转成 YOLO-seg 多边形标签并做分层划分；用 Copy-Paste 把 train 划分里的真实缺陷
实例贴到正常图上扩样，缓解小样本与类别不均衡；训练 YOLOv8-seg 并做消融（合成增强 / Focal Loss /
去负样本）；随后导出 ONNX、构建 TensorRT **FP32 / FP16 / INT8** 三档 engine；最后分别用 TensorRT
**Python API 与 C++ API** 写两套推理服务（三线程流水线 + pinned memory 零拷贝、GPU 融合 letterbox
kernel、decode 线程池），并在同一台机器上做精度-延迟-显存对照与**逐帧对拍**。

结果：Copy-Paste 把 mask mAP50-95 从 **0.245 抬到 0.407（+66%）**（2 类消融子集，看相对提升）；
INT8 engine **12.55 MiB / 173 FPS**（较同场 PyTorch **2.41×**）；C++ 版单帧 **P50 4.1 ms**、
吞吐 **230.89 FPS**，与 Python 版逐帧对拍结果完全一致（match_rate **1.0000**、掩码 IoU **0.997**）。

## 项目贡献

- **一条可复现的完整链路**：数据下载校验 → 标签转换 → 合成增强 → 训练/消融 → ONNX/TRT 量化 →
  Python 与 C++ 两套推理服务 → 压测报告。每步都有命令行、单测（**892 条**）和 CI。
- **合成增强的收益与边界都量化**：小样本场景 mask mAP50-95 +66%；同时如实记录 focal 在这个规模
  是**负收益**、去负样本组因训练步数不可比而**结论挂起**——不把无效方法写成有效。
- **跨实现一致性有证据**：四层对拍（解码逐字节 / 前处理逐位 / 后处理逐项 / 检测级 IoU）+ 500 帧
  checksum 逐位全等，量化与用 C++ 重写都不"变味"；哪条判据没验也明写（不宣称）。
- **端侧性能靠工程而非调参**：GPU 融合前处理 kernel、pinned 零拷贝、decode 线程池，吞吐
  **62 → 230.89 FPS**、单帧 P50 **21.8 → 4.1 ms**，一致性一分不让。

## 数据集

[MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad)（Bergmann et al., CVPR 2019）：
15 类工业物体/纹理，5354 张图 + 像素级掩码标注；官方 `train/good` 3629 张整体作为**背景负样本**
进 train。许可 **CC BY-NC-SA 4.0（非商用）**，数据不入库——脚本从 `hf-mirror` 逐文件下载并用
sha256 对比 HF LFS 的 `lfs.oid` 校验。

## 实验环境

RTX 4060 Laptop **8GB** / Windows 11 / conda env（构建过程见 [docs/environment.md](docs/environment.md)）。
所有工程决策围绕"8GB 显存怎么把链路跑完整"来做。文档里的数字**全部本机实测**，没跑到的标
**待测**，不预填、不估；同一台机器不同轮次的延迟不可直接相减（时钟与温度会漂），跨组比较只在同轮内成立。

[![CI](https://github.com/wanzi-qz/wanzi-induvision/actions/workflows/ci.yml/badge.svg)](https://github.com/wanzi-qz/wanzi-induvision/actions/workflows/ci.yml)

想按顺序看论证过程，读 [docs/overview.md](docs/overview.md)：按 **背景 → 方法 → 结果** 一次讲完。
下面这张进度表是它的索引版，每行链到对应的细节文档。

## 进度

| 阶段 | 内容 | 状态 |
| --- | --- | --- |
| W0 | GPU 环境搭建 + 自检脚本（版本 / GPU / 推理 / ONNX 导出） | 完成 → [environment.md](docs/environment.md) |
| W1a | MVTec 下载校验 + YOLO-seg 转换管线 + 单测 + CI | 完成 → [dataset.md](docs/dataset.md) |
| W1b | 15 类全量数据 + 基线训练 | 完成 → [baseline.md](docs/baseline.md) §2（test split mask mAP50-95 **0.3740**） |
| W2 | Copy-Paste 合成增强 + Focal Loss + 小样本/不均衡消融 | 完成（5 组全出数）→ [baseline.md](docs/baseline.md) §3（mask mAP50-95 0.2453 → **0.4071**，+66%；focal 负收益） |
| W3a | ONNX 导出 + 三层精度对齐（逐张量 / mAP / 延迟） | 完成 verdict **pass** → [deploy.md](docs/deploy.md)（cosine 1.00000000，mAP 0.3740 → 0.3741） |
| W3b | TensorRT 三档量化 FP32/FP16/INT8 + 三层压测 | 完成 verdict **pass** → [deploy.md](docs/deploy.md) §8（INT8 **12.55 MiB / 173 FPS**，mAP 掉 0.0058） |
| W4 | 推理服务：零拷贝 + 三线程流水线 + C++/CUDA 版 + 压测 | 完成 → [pipeline.md](docs/pipeline.md)（INT8 含磁盘 51.77 / preload **330 FPS**）、[cpp_bench.md](docs/cpp_bench.md)（C++ 版 **230.89 FPS**、P50 **4.1 ms**、四层对拍全过） |

每行的判据、口径、踩坑与"哪些数不能跟谁比"都在链接的文档里，表里只留一个能一眼看懂的结果。
W4 的三部分分别是：Python 流水线 → [pipeline.md](docs/pipeline.md)；C++ 版对拍与压测 →
[cpp_bench.md](docs/cpp_bench.md) §1~§5；decode 线程池 → 同文 §6。

## 仓库结构

```
induvision/          核心库（只依赖 numpy / opencv / pyyaml，CI 能全量跑）
  mvtec.py               MVTec 目录扫描 + 分层划分 + 掩码→多边形
  yolo_dataset.py        读写本仓库产出的 YOLO-seg 数据集
  defect_bank.py         缺陷实例库：只从 train 划分抠缺陷，从源头杜绝 val/test 泄漏
  copy_paste.py          Copy-Paste 贴图与融合（poisson / alpha），标签同步重建
  balance.py             类别不均衡诊断 + 合成配额规划
  focal.py               Focal Loss：给 ultralytics 分类分支换损失，可装可卸（LIFO）
  deploy.py              部署链路的纯 CPU 基础设施：复刻 ultralytics 前处理口径、张量比差、延迟分位、抽样清单读写与指纹
  trt_infer.py           TensorRT 运行期：engine 读写（带 ultralytics 风格 metadata header）、device_memory、TrtRunner
  pipeline.py            三阶段流水线：RingBuffer 环形队列 + 解码/推理/后处理三线程 + 自研 NMS 与掩码解码
  pinned.py              pinned memory 令牌池：零拷贝 H2D/D2H，每帧只 stream.synchronize() 一次
cpp/                  W4② C++ 推理服务（CMake + nvcc；build/ 与 trt_lib/ 不入库）
  src/engine_header.*   剥掉 .engine 前面 844 字节的 ultralytics 自描述头（JSON + 长度前缀）
  src/infer_one.cpp     单帧最小闭环：反序列化 → enqueueV3 → 取输出，与 Python 侧逐位相等
  src/preproc.cu        融合 kernel：BGR→RGB + /255 + bilinear + pad 114，直接写 engine 的 device 输入
  src/postproc.*        pipeline.py 后处理族的逐条移植（sigmoid / batched NMS / scale_boxes / 掩码解码）
  src/bench.cpp         serial + 三线程流水线（环形队列 + pinned slot 池 + event），带三个对拍探针
  build.bat             nvcc 单 TU + 显式 MSVC/SDK 环境（不调 vcvars，原因见 docs/cpp_infer.md §2）
  CMakeLists.txt        Release / C++17 / /utf-8；TensorRT 路径走 -DTENSORRT_ROOT= 传入，不写死
tools/
  env_check.py           环境与推理链路自检（可选 --export 验 ONNX 导出）
  download_mvtec.py      取数据：官方 zip 手动模式 / HF 镜像自动模式，逐文件 sha256
  convert_mvtec_to_yolo.py   MVTec AD → YOLOv8-seg 数据集
  synthesize_defects.py  Copy-Paste 缺陷合成，给已有数据集补齐少数类实例配额
  train_seg.py           训练入口：统一 CLI + 可选 focal 注入 + 配置留档 + 指标摘要
  eval_seg.py            在 test split 上重算成绩 + 逐类 AP 表（best.pt 是 val 选的，不能拿 val 报数）
  export_onnx.py         .pt -> .onnx，导完立刻静态体检（opset / 动态轴 / checker / 算子清单）
  verify_onnx.py         .pt 与 .onnx 是否等价：固定抽样清单 -> 逐张量对齐 -> 端到端 mAP -> 单帧延迟，各层带判据
  make_calib_list.py     INT8 校准清单：只从 train split 等间距抽，带 sha256 指纹与逐类覆盖统计（拒绝 val/test）
  build_trt.py           .onnx -> TensorRT engine：三档精度、minmax 校准器、头精度配方、--no-tf32 对照档
  bench_trt.py           三档压测：逐张量对齐 / 端到端 mAP / 单帧延迟 + PyTorch 同场对照，各层带判据与 verdict
  bench_pipeline.py      流水线压测：三档 × {plain,pinned} × {串行,流水线} 全组合，顺带和 ultralytics 参照实现对拍
  bench_gpu_preproc.py   GPU letterbox 预研：CPU/GPU 前处理交错对比，拆出 imdecode 与 letterbox 各占多少
  check_cpp_infer.py     C++ 单帧版对拍器：decode 逐字节 / 前处理张量 / 后处理 tuple 三层
  check_cpp_bench.py     C++ 压测对拍器 + 资源采样（nvidia-smi 200 ms + psutil，照搬 prof.py 口径）
  make_trt_lib.py        从 tensorrt_libs 的 DLL 导出表合成 nvinfer.lib（wheel 里 0 个 .lib）
  summarize_ablation.py  汇总各组 results.csv 出 Markdown 表，校验「单变量协议」与数据集转换环境指纹
  run_ablation_queue.ps1 消融队列（幂等 + 崩溃自动重试）：跑完的组自动跳过，中断的组搬残档后原地重跑
  run_train_guarded.ps1  单组训练守护：等显存 -> 跑 -> 崩了搬残档重试，最多 -MaxAttempts 次
  run_overnight.ps1      串联：补跑消融 -> baseline15 -> test 评测 -> 汇总表，一次挂上不用守
  run_download_mvtec.ps1 后台拉全量数据（脱离宿主 shell，日志落 runs/logs/）
  make_fake_mvtec.py     生成合成 MVTec 目录树，供 CI 端到端冒烟
configs/             入库的配置：bench_sample.json 固定抽样清单（部署校验的口径基准）、
                     calib_sample.json INT8 校准清单（train split 500 张），都带 sha256 指纹
tests/               892 条 pytest，纯 CPU、不联网、不需要真实数据集
docs/                overview.md 项目说明（背景 / 方法 / 结果，先看这个）
                     environment.md 环境 / dataset.md 数据与协议
                     baseline.md 训练记录 / augmentation.md 合成增强与 focal 消融
                     deploy.md 部署链路（ONNX 导出、三层精度对齐与固定抽样口径）
                     pipeline.md 推理流水线（pinned 令牌池 / 三线程 / 与串行结果对拍 / 500 帧延迟与资源占用）
                     gpu_preproc.md GPU letterbox 预研（16~19 ms 的 decode 段怎么拆、搬上 GPU 净省多少）
                     cpp_infer.md W4② C++ 版施工单（工具链缺口实测 / 安装清单 / 验收判据 / engine I/O 契约）
                     cpp_bench.md W4② C++ 版实测报告（四层对拍 / 500 帧延迟与资源 / 1.23× 的口径分析）
requirements-gpu.txt 本地 GPU 环境（钉版本）
requirements-ci.txt  CI 最小依赖（刻意不含 torch）
data/ weights/ out/ runs/   全部不入库，见下方"许可与入库边界"
```

## 快速开始

```powershell
# 0) 环境（conda env: induvision，构建过程见 docs/environment.md）
conda activate induvision

# 1) 取数据：走 hf-mirror、逐文件 sha256 校验（口径见 docs/dataset.md）
python tools/download_mvtec.py --source hf --categories bottle toothbrush

# 2) 转成 YOLO-seg（硬链接落盘，同盘不额外占空间）
python tools/convert_mvtec_to_yolo.py --src data/mvtec_raw --out out/yolo_mvtec

# 3) 训练（8GB 显存默认 imgsz 640 / batch 8 / AMP on）
python tools/train_seg.py --data out/yolo_mvtec/data.yaml `
     --model weights/yolov8s-seg.pt --name baseline --epochs 100

# 3b) 小样本 + 类别不均衡：先合成缺陷再训练；合成只加 train，val/test 逐字节不动
python tools/synthesize_defects.py --src out/yolo_mvtec --out out/yolo_mvtec_cp `
     --cap-factor 2.0 --seed 0 --workers 4
python tools/train_seg.py --data out/yolo_mvtec_cp/data.yaml `
     --model weights/yolov8s-seg.pt --name cp_focal `
     --focal-gamma 2.0 --focal-alpha 0.25 --epochs 100

# 4) 消融：四组各 100 epoch，幂等队列后台跑（跑完的组自动跳过）
powershell -NoProfile -ExecutionPolicy Bypass -File tools/run_ablation_queue.ps1

# 5) 汇总成表，并校验各组超参与环境指纹一致
python tools/summarize_ablation.py --markdown
python tools/summarize_ablation.py --strict            # 不一致就返回非零

# 6) 成绩必须在 test split 上重算：best.pt 是按 val 早停选出来的
python tools/eval_seg.py --weights runs/segment/baseline/weights/best.pt `
     --data out/yolo_mvtec/data.yaml --split test --name baseline --markdown

# 7) 导出 ONNX，再验一遍「换后端没换语义」（逐张量 / 端到端 mAP / 延迟）
python tools/export_onnx.py --weights runs/segment/baseline15/weights/best.pt --markdown
python tools/verify_onnx.py --sample-n 64 --sample-list-out configs/bench_sample.json --dry-run
python tools/verify_onnx.py --weights runs/segment/baseline15/weights/best.pt `
     --align-n 16 --bench-n 30 --val --val-batch 8 --markdown
# 抽样清单入库后别重新生成，否则和历史报告不可比；判据与踩坑见 docs/deploy.md

# 7b) TensorRT 三档量化与压测（要 GPU + tensorrt）
python tools/make_calib_list.py --data out/yolo_mvtec15/data.yaml --split train --n 500
python tools/build_trt.py --prec fp32 fp16 int8
python tools/bench_trt.py --val --markdown
# 本机没有 trtexec，三档全走 Python API；TF32 与 INT8 连挂的根因见 docs/deploy.md §8

# 7c) 推理流水线：三档 × {plain,pinned} × {串行,流水线}
python tools/bench_pipeline.py --check-ultralytics --markdown              # A 组含磁盘
python tools/bench_pipeline.py --preload --check-ultralytics --markdown    # B 组 preload
# 对外报数用 A 组；A/B 两组别混读，为什么见 docs/pipeline.md §5

# 端到端延迟 + CPU/GPU 占用（500 帧 = 50 张 × 10 轮）
python tools/bench_pipeline.py --frames 50 --repeats 10 --sample-resources --markdown

# 7d) C++ 版：C++17 + nvcc + TensorRT 10.16 + OpenCV，产物落 cpp/build/，不入库
cpp\build.bat
python tools\check_cpp_infer.py --exe cpp\build\infer_one.exe --engine weights\baseline15_trt_fp16.engine
python tools\check_cpp_bench.py --mode bench --bench-mode serial   --frames 500 --repeats 10
python tools\check_cpp_bench.py --mode bench --bench-mode pipeline --frames 500 --repeats 10
# 构建脚本自己设 MSVC/SDK 环境（不调 vcvars64），原因见 docs/cpp_infer.md §2；
# 对拍判据与 1.23× 为什么没到 1.44×，见 docs/cpp_bench.md

# 8) 单测（892 条，约 20 秒；base 环境缺 onnx/ultralytics 会自动跳 6 条）
python -m pytest -q
```

长训练一律走守护脚本，别裸跑：这台机器驱动会 TDR（一晚两次，见 [docs/baseline.md](docs/baseline.md) §3）。
脚本按 `train_args.json` 判断"这组到底跑完没有"，崩了搬残档、探一次 `nvidia-smi`、原地重跑：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools\run_train_guarded.ps1 `
    -Name baseline15 -Data out/yolo_mvtec15/data.yaml      # 单组，最多重试 3 次
powershell -NoProfile -ExecutionPolicy Bypass -File tools\run_overnight.ps1   # 整晚串联
```

不装 GPU 环境也能跑通 2、3b、5：转换、合成、汇总只依赖 numpy / opencv / pyyaml。CI 就是这么验的
（ubuntu + windows × py3.10/3.12），且在 CI 里端到端跑了一遍缺陷合成、逐字节断言 val/test 未改动。

## 三个关键设计决策

**1. 有监督划分只能落在官方 test 集内部。**
MVTec AD 官方 `train/` 只有 `good` 正常样本，一张缺陷标注都没有——它是为无监督
异常检测设计的。要做 YOLO-seg 就得在 `test/` 的缺陷图里按 `category × defect_type`
分层划 70/15/15，固定种子。所有对外指标必须注明这一点，本项目的 mAP 不能和论文里的
AUROC-PRO 直接比。

**2. 面积过小的缺陷图默认丢弃，不留成空标签。**
留着等于告诉模型"这块缺陷是背景"，是标签噪声。丢弃数写进
`dataset_stats.json`，需要保留时加 `--keep-empty-defects`。

**3. 用 RETR_EXTERNAL 取外轮廓，代价可量化。**
`contamination` 这类缺陷带内部孔洞，取外轮廓会把孔填掉，多边形面积因此偏大。
偏大多少用 `area_fidelity = 多边形面积 / 掩码面积` 记录：bottle+toothbrush
实测均值 0.9765、p10 0.9518。

细节和取舍理由都在 [docs/dataset.md](docs/dataset.md)。

## 实测记录

只列每个阶段的关键结果。判据、口径、踩坑、以及"哪些数不能跟谁比"，都在每节末尾链接的文档里。

### 数据与训练

| 项 | 值 |
| --- | --- |
| 数据 | 15 类 / 5354 图，逐文件 sha256 对 HF LFS `lfs.oid` 全过；转换 27.7 s |
| 划分（图 / 实例） | train 4490 / 1261，val 429 / 299，test 435 / 328 |
| 缺陷特性 | 面积占比 mean 4.39% / p50 **1.54%**（15 类全量；冒烟集 2 类 p50 3.50%）；冒烟集缺陷图仅 23.6% —— 小目标 + 样本稀缺，W2 就冲这两点 |
| 基线 baseline15 | test mask mAP50-95 **0.3740** / mAP50 0.6730，box 0.4229 / 0.6652，best @ ep75 |
| + Copy-Paste（B1） | mask mAP50-95 0.2453 → **0.4071（+66%）**，mask P 0.3885 → 0.9025；标签面积保真 mean 0.9675 |
| 消融结论 | focal 负收益（C1 **0.1953**）；去负样本 B4 训练步数不可比，**结论挂起** |
| 数据安全 | 合成只加 train（335 → 375），val / test 逐字节不动（CI 断言） |

参数表与两个 focal 踩坑见 [docs/augmentation.md](docs/augmentation.md)，训练与消融全量结果见
[docs/baseline.md](docs/baseline.md) §2 / §3。

### W3 部署（ONNX + TensorRT）

ONNX 三层对齐：**45.53 MB** / opset 17 / 逐张量 cosine **1.00000000**（rel_l2 2.4e-7）/
端到端 mask mAP50-95 0.3740 → **0.3741**，verdict **pass**。

| 档 | engine | 构建 | mask mAP50-95 | mean / p95 ms | FPS | 较 `.pt` |
| --- | --- | --- | --- | --- | --- | --- |
| FP32（含 TF32） | 65.25 MiB | 94.0 s | 0.3707 | 9.51 / 11.35 | 105 | 1.46× |
| FP16 | 24.58 MiB | 210.9 s | 0.3702 | 6.73 / 10.66 | 149 | 2.07× |
| INT8（minmax） | **12.55 MiB** | 152.0 s | 0.3649 | **5.77** / 9.73 | **173** | **2.41×** |
| PyTorch `.pt` 同场 | — | — | 0.3740（基准） | 13.90 / 15.89 | 72 | 1.00× |

三档 verdict 全 **pass**（INT8 掉点 -0.0058，判据 ≤0.03 只用掉 19%）。三条读数注意：
**TF32 默认开着**（"FP32 档" rel_l2 3.3e-3 过不了，`--no-tf32` 才降到 2.5e-7）；**延迟只能同轮比**
（同档两轮差过 40%）；**构建层数本身不确定**（166 vs 168），比对要看 `engine_sha256`。
见 [docs/deploy.md](docs/deploy.md) §4.0 / §8。

### W4① Python 推理流水线

| 组 | 口径 | FP32 | FP16 | INT8 |
| --- | --- | --- | --- | --- |
| A 含磁盘（对外报数） | 读盘 + 解码 + 推理 + 后处理 | 36.86 → **57.53**（1.561×） | 34.55 → **55.82**（1.616×） | 33.12 → **51.77**（1.563×） |
| B preload（链路上限） | 图常驻内存 | 126.36 → **158.95**（1.258×） | 215.46 → **305.52**（1.418×） | 233.61 → **330.05**（1.413×） |

箭头左是串行单帧、右是 pinned + 三线程流水线；加速比只在同档同组内成立。A 组瓶颈是解码
（util 1.00），**pinned 的账要到 B 组才算得出**（INT8 330.05 vs plain 263.20，+25%）。
吞吐是拿延迟换的：串行 P50 24~27 ms → 流水线 98~107 ms（放大 4.0~4.2×），换来 ~40 → ~65 FPS
（+60%），两组的数都得报。
口径与这笔账见 [docs/pipeline.md](docs/pipeline.md) §5 / §6.5。

与 ultralytics 参照实现对拍（先修了掩码拉伸的错法）：match_rate **1.0000** / 框 IoU **1.0000** /
掩码 IoU **0.9100~0.9158**（判据 0.85），三档各 10/10 条判据全过。见同文 §4.2。

### W4② C++/CUDA 版

| 层 | 对拍内容 | 判据 | 实测 |
| --- | --- | --- | --- |
| A | decode：C++ `imdecode` vs `cv2` | 逐字节 | **5/5** |
| B | 前处理：融合 kernel vs `GpuPreproc` | max_abs / cosine | **0 / 1.0（逐位相等）** |
| C | 后处理移植：同一份 engine 输出喂两边 | 全 tuple 相等 | **20/20 帧** |
| D | 检测级：C++ vs Python 全链路 | match_rate ≥0.98 / 框 IoU ≥0.99 | **1.0000 / min 0.9967** |
| D2 | 掩码位图 IoU（`--dump-masks`） | ≥0.85 | **min 0.9972 / mean 0.9995** |

500 帧：serial **48.42 / 48.23 FPS**（P50 21.8，**P99 32.2**）→ 流水线 **59.47 FPS**
（P50 **4.1**，P95 6.5，**P99 10.8**）；`(n_det, mask_pixels, score_sum)` 两轮 **500/500 全等，
n_diff = 0**，dets 270 与 Python 完全一致。CPU 1.00 → 1.31 核。

**加速比 1.23×（另一轮 1.28×）没到 1.44× 判据，照实写**：C++ serial 本身已比 Python 同清单快
11~21%，上限又被 decode 段压住（单飞 ~15 ms → ~66 FPS）。收益落在 P50 **21.8 → 4.1 ms（5.3×）**、
吞吐比 C++ serial +23%。归因见 [docs/cpp_bench.md](docs/cpp_bench.md) §3.1。

W4③ decode 线程池：`bench --decode-threads N`，PNG 一字节不动、结果逐位不变。500 帧
**62 → 219 FPS（dt=4）→ 290（dt=8，CPU 打满）**，六组 checksum 逐位一致；dt=4 验收
**230.89 FPS / GPU util 25.1% → 60.6% / CPU 4.26 核**。为什么不是 nvJPEG、e2e 分位数为什么随 N
升高，见同文 §6。移植时对拍抓到三个真 bug（掩码系数步长、掩码解码喂错框域、中文路径），见 §4。

测试规模 563 → **892** 条 pytest（W4① 新增 329 条）。这 329 条一条都不碰 GPU —— 环形队列、
令牌池、CUDA 流、`nvidia-smi` 子进程全用假对象测，"每帧只 synchronize 一次""出错要还令牌"
都是断言出来的；延迟那条还钉了一个口径：**时间戳必须打在 `put` 之前**，否则背压被藏起来。

## 许可与入库边界

| 内容 | 许可 | 是否入库 |
| --- | --- | --- |
| 本仓库脚本、文档、配置 | MIT | 是 |
| MVTec AD 数据集 | CC BY-NC-SA 4.0（非商用） | 否（`.gitignore` 排除 `data/*`，只留 `data/README.md`） |
| YOLOv8 权重与 ultralytics 代码 | AGPL-3.0 | 否（排除 `weights/`、`*.pt`、`*.onnx`、`*.engine`） |
| 训练产物、量化引擎、压测报告原始文件 | — | 否（排除 `out/`、`runs/`）；报告以 Markdown 形式入库 |

CI 里有一条断言专门守这条边界：`git ls-files` 出现 `data/`、`weights/`、
`out/`、`runs/` 或任何 `.pt/.onnx/.engine/.trt` 就直接失败。

## 网络注意事项（国内环境）

- `github.com:443` 间歇性不通。release 资产走 `https://gh-proxy.com/<原始链接>`。
- HuggingFace 走 `https://hf-mirror.com`（脚本默认 endpoint）。
- pip 加 `-i https://pypi.tuna.tsinghua.edu.cn/simple`。
- ultralytics 首次训练会联网下 `yolo26n.pt` 做 AMP 自检，网络不通会卡住重试三次。
  提前把它放到运行命令的工作目录（仓库根）即可跳过；急着跑也可以 `amp=False`。