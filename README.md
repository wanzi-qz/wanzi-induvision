# wanzi-induvision

工业缺陷检测与端侧部署闭环。在 MVTec AD 上训练 YOLOv8-seg 做缺陷分割，
再走 ONNX → TensorRT 三档量化（FP32/FP16/INT8），最后用 TensorRT C++ API
写推理服务、做端到端压测，产出一份精度-速度-显存对照报告。

个人求职项目。硬件是 RTX 4060 Laptop 8GB / Windows 11，所以所有工程决策都围绕
"8GB 显存怎么把链路跑完整"来做。文档里的数字全部是本机实测，
还没跑到的阶段一律标 **待测**，不预填、不估。

[![CI](https://github.com/wanzi-qz/wanzi-induvision/actions/workflows/ci.yml/badge.svg)](https://github.com/wanzi-qz/wanzi-induvision/actions/workflows/ci.yml)

第一次看这个仓库，先读 [docs/overview.md](docs/overview.md)：按 **背景 → 方法 → 结果**
把整条链路一次讲完。下面这张进度表是它的索引版，每行链到对应的细节文档。

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

# 1) 取数据。官方站要勾 EULA 不能直连，HF 镜像可脚本化，逐文件校验 sha256
python tools/download_mvtec.py --source hf --categories bottle toothbrush

# 2) 转成 YOLO-seg（默认硬链接落盘，同盘不额外占空间）
python tools/convert_mvtec_to_yolo.py --src data/mvtec_raw --out out/yolo_mvtec

# 3) 训练。统一走 train_seg.py：配置留档、可选 focal 注入、跑完打指标摘要
python tools/train_seg.py --data out/yolo_mvtec/data.yaml `
     --model weights/yolov8s-seg.pt --name baseline --epochs 100
# 8GB 显存起步参数就是默认的 imgsz 640 / batch 8 / AMP on（实测 2.3~3.0 GiB）
# 产物落在 runs/segment/baseline。别传 --project runs，会拼成 runs/segment/runs/baseline

# 3b) 小样本 + 类别不均衡：先合成缺陷补齐少数类，再训练
python tools/synthesize_defects.py --src out/yolo_mvtec --out out/yolo_mvtec_cp `
     --cap-factor 2.0 --seed 0 --workers 4
python tools/train_seg.py --data out/yolo_mvtec_cp/data.yaml `
     --model weights/yolov8s-seg.pt --name cp_focal `
     --focal-gamma 2.0 --focal-alpha 0.25 --epochs 100
# 合成只加 train，val/test 逐字节不动；细节和防泄漏口径见 docs/augmentation.md

# 4) 消融：四组各 100 epoch 要一小时以上，用幂等队列放后台跑
#    跑完的组自动跳过，被系统重启打断后重跑同一条命令就从断点续
powershell -NoProfile -ExecutionPolicy Bypass -File tools/run_ablation_queue.ps1

# 5) 汇总成表，顺带校验各组超参与数据集转换环境真的一致
python tools/summarize_ablation.py --markdown          # 可直接贴进 docs/baseline.md
python tools/summarize_ablation.py --strict            # 协议或环境指纹不一致就返回非零

# 6) 报成绩要在 test split 上重算：best.pt 是靠 val 早停选出来的，
#    拿 val 的数当最终成绩等于在选过的数据上报告结果
python tools/eval_seg.py --weights runs/segment/baseline/weights/best.pt `
     --data out/yolo_mvtec/data.yaml --split test --name baseline --markdown

# 7) 部署链路：导出 ONNX，再验一遍「换后端没换语义」（逐张量 / 端到端 mAP / 延迟）
python tools/export_onnx.py --weights runs/segment/baseline15/weights/best.pt --markdown
# 7a) 抽样口径基准只做一次，产物入库（仓库里已有一份，别重复生成，否则和历史报告不可比）
python tools/verify_onnx.py --sample-n 64 --sample-list-out configs/bench_sample.json --dry-run
python tools/verify_onnx.py --weights runs/segment/baseline15/weights/best.pt `
     --align-n 16 --bench-n 30 --val --val-batch 8 --markdown
# 三层全跑 81.5 s（端到端那层在 CPU 上最慢）。只看配置不写盘加 --dry-run
# 清单在就用清单，对不上就降级并在 stdout / JSON / Markdown 三处留痕；不接受降级加 --strict-sample
# 前处理复刻口径、三条判据、固定抽样清单与踩坑记录见 docs/deploy.md

# 7b) TensorRT 三档量化与压测（要 GPU + tensorrt，本机在 conda env: induvision 里）
python tools/make_calib_list.py --data out/yolo_mvtec15/data.yaml --split train --n 500
python tools/build_trt.py --prec fp32 fp16 int8      # 三档连建 456.9 s
python tools/bench_trt.py --val --markdown           # 三层压测 78.9 s，出 verdict
# 本机没有 trtexec.exe（pip wheel 只带运行期 .dll），三档全走 TensorRT Python API
# 想看关掉 TF32 的真 FP32 对照档：build_trt.py --name x --prec fp32 --no-tf32
# 三档实测数字、TF32 排查、INT8 连挂四次的根因见 docs/deploy.md §8

# 7c) 推理流水线压测（同样要 GPU + tensorrt）：三档 × {plain,pinned} × {串行,流水线} 全组合
python tools/bench_pipeline.py --check-ultralytics --markdown            # A 组含磁盘，92.2 s
python tools/bench_pipeline.py --preload --check-ultralytics --markdown  # B 组 preload，27.0 s
# 对外报数用 A 组（含读盘的真实口径）：INT8 51.77 FPS，较同档串行 1.563x
# B 组是把图先读进内存、只量推理链路的上限：INT8 330.05 FPS（1.413x）
# 两组数字别混着读，为什么见 docs/pipeline.md §5

# 端到端延迟 + CPU/GPU 占用（500 帧 = 50 张 × 10 轮；采样默认关，扰动实测小于噪声底）
python tools/bench_pipeline.py --frames 50 --repeats 10 --sample-resources --markdown

# 7d) C++ 版（W4②）：C++17 + nvcc + TensorRT 10.16 + OpenCV，产物落在 cpp/build/，不入库
cpp\build.bat                                  # nvcc 编 preproc.obj -> CMake/Ninja 出 bench.exe
# 构建脚本不调 vcvars64.bat：它内部要跑 reg.exe，而本机的命令白名单会拦掉，
# 所以脚本自己把 MSVC/SDK 的 PATH/INCLUDE/LIB 写死（路径见 docs/cpp_infer.md §2）
python tools\check_cpp_infer.py --exe cpp\build\infer_one.exe --engine weights\baseline15_trt_fp16.engine  # A/B/C 三层对拍
python tools\check_cpp_bench.py --mode bench --bench-mode serial   --frames 500 --repeats 10
python tools\check_cpp_bench.py --mode bench --bench-mode pipeline --frames 500 --repeats 10
# 两条命令各跑一遍、各跑两轮；500 帧的 (n_det, mask_pixels, score_sum) 必须两两全等（n_diff=0）
# 四层对拍判据、1.23× 加速比为什么没到 1.44×，见 docs/cpp_bench.md

# 8) 单测（892 条，约 20 秒；base 环境缺 onnx/ultralytics 会自动跳 6 条）
python -m pytest -q
```

长训练一律走守护脚本，别裸跑 —— 这台机器驱动会 TDR（见 docs/baseline.md §3
「队列与中断恢复」：一晚两次，一次蓝屏 `0x00000116`，一次 `CUDA error: unknown error`）。
守护脚本认 `train_args.json` 判断"这组到底跑完没有"，崩了就搬残档、探一次 `nvidia-smi`、原地重跑：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools\run_train_guarded.ps1 `
    -Name baseline15 -Data out/yolo_mvtec15/data.yaml      # 单组，最多重试 3 次
powershell -NoProfile -ExecutionPolicy Bypass -File tools\run_overnight.ps1   # 整晚串联
```

不装 GPU 环境也能跑通 2、3b 和 5：转换、合成与汇总只依赖 numpy / opencv / pyyaml（第 4 步的训练队列要 GPU）。
CI 就是这么验的（ubuntu + windows × py3.10/3.12），并且在 CI 里端到端跑了一遍
缺陷合成，逐字节断言 val/test 没被改动。

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