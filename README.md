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
| W0 | GPU 环境搭建 + 自检脚本（版本/GPU/推理/ONNX 导出） | 完成 → [docs/environment.md](docs/environment.md) |
| W1a | MVTec 下载校验 + YOLO-seg 转换管线 + 单测 + CI | 完成 → [docs/dataset.md](docs/dataset.md) |
| W1b | 15 类全量数据 + 基线训练（mAP50-95、单类 AP、混淆矩阵） | 完成 → [docs/baseline.md](docs/baseline.md) §2：15 类 / 5354 图（sha256 全过，转换 27.7 s），baseline15 在 test split 上 mask mAP50-95 **0.3740** / mAP50 0.6730，box 0.4229 / 0.6652，best @ ep75、ep95 早停 |
| W2 | 缺陷合成 Copy-Paste 增强 + Focal Loss + 小样本/不均衡消融 | 完成（5 组全出数）→ [docs/baseline.md](docs/baseline.md) §3：Copy-Paste 把 mask mAP50-95 从 0.2453 抬到 **0.4071**（+66%），整张复制的过采样 **0.4003** 与它基本打平（差 0.0068），focal 在这个规模上是**负收益**（C1 0.1953，B2 0.0039 直接学崩）。绝对值不可外引（val 仅 29 图 / 16 实例） |
| W3a | ONNX 导出 + 三层精度对齐（逐张量 / 端到端 mAP / 单帧延迟）+ 固定抽样清单 | 完成 → [docs/deploy.md](docs/deploy.md)：45.53 MB / opset 17 / 动态 batch / checker 通过；逐张量 cosine **1.00000000**、rel_l2 2.4e-7；端到端 mask mAP50-95 0.3740 → **0.3741**；verdict **pass**。口径基准 `configs/bench_sample.json`（64/435，sha256 `eccfd964426a2637`），见 §4.0 |
| W3b | TensorRT 三档量化 FP32/FP16/INT8（minmax 校准，500 张校准集）+ 三层压测 | 完成 → [docs/deploy.md](docs/deploy.md) §8：INT8 **12.55 MiB**（较 FP32 档 5.2× 压缩）/ 5.77 ms / **173 FPS**（较同场 PyTorch `.pt` **2.41×**），mask mAP50-95 0.3707 → 0.3649（-0.0058，判据 ≤0.03 只用掉 19%），三档 verdict **pass**。踩坑：TensorRT 默认开 TF32，"FP32 档"逐张量 rel_l2 3.3e-3 卡不过判据，`--no-tf32` 才降到 2.5e-7（§8.4） |
| W4 | TensorRT 推理服务：零拷贝 + 三线程流水线 + 压测报告 | ①完成 → [docs/pipeline.md](docs/pipeline.md)：三档 × {plain,pinned} × {串行,流水线} 全组合实测，A 组含磁盘 **1.44~1.71×**（INT8 51.77 FPS）、B 组 preload **1.19~1.42×**（INT8 **330 FPS**）；与 ultralytics 对拍 match_rate **1.0000** / 框 IoU **1.0000** / 掩码 IoU 0.910~0.916，三档 verdict **pass**。② 完成 → [docs/cpp_bench.md](docs/cpp_bench.md)：TensorRT C++ API（`enqueueV3` + GPU 融合 letterbox kernel + 三线程流水线 + 自研后处理移植）；四层对拍 A decode 逐字节 5/5、B 前处理 **max_abs 0 / cosine 1.0（逐位相等）**、C 后处理 20/20 帧全 tuple 相等、D 检测级 match_rate **1.0000** / 框 IoU min **0.9967** / **掩码位图 IoU 逐帧均值 min 0.9972（判据 0.85，真位图口径）**；500 帧 serial vs pipeline **n_diff=0** 两轮全等（dets 270 与 Python 一致）；流水线 **59.47 FPS**（另一轮 61.59，serial 48.42）、**P50 4.1 ms / P95 6.5 / P99 10.8**，比 Python 同口径 fp16 快 **31~36%**。**加速比 1.23× 未达 1.44× 判据**，原因与口径分析见 §3.1（P50 5.3×、CPU 1.65→1.31 核，下一个台阶是 nvJPEG 解码） |

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

bottle + toothbrush 冒烟集（487 文件 / 254 MB，2026-10-21）：

| 项 | 值 |
| --- | --- |
| 原始数据核对 | bottle 209/20/63，toothbrush 60/12/30，与官方论文一致 |
| sha256 校验 | 487/487 通过（校验值取自 HF LFS `lfs.oid`） |
| 转换产物 | 394 张图 / 134 个实例，硬链接 394/394 成功，耗时 1.9 s |
| 缺陷面积占比 | mean 5.85%，p50 3.50%，min 0.24%，max 27.42% |
| 缺陷图占比 | 93/394 = 23.6%，背景负样本是其 3.2 倍 |
| GPU 自检 | YOLOv8s-seg 单图推理 22.76 ms（640，预热后），峰值显存 123 MiB |
| 训练冒烟 | 13 epoch 跑通 训练→验证→保存，显存 4.34 GiB @ batch8/640/AMP off，3.4 it/s |
| CI 首轮 | ubuntu + windows × py3.10/3.12 共 4 job 全绿 |
| ONNX 导出 | 成功，45.40 MB，opset 17，dynamic batch，onnx.checker 通过 |

这两行数字直接指出了 W2 要解决什么：**缺陷样本稀缺（23.6%）+ 目标极小（p50 3.5%）**，
对应 Copy-Paste 缺陷合成和 Focal Loss / 过采样。

W2 合成管线实测（bottle + toothbrush，seed=0 / cap-factor=2 / workers=4）：

| 项 | 值 |
| --- | --- |
| 缺陷库 | 97 实例 / 66 图，`built_from=["train"]` |
| 合成产出 | 40 张新图 / 109 个实例，配额打满，耗时 13.8 s |
| 实例均衡 | bottle 50 → 101，toothbrush 47 → 105 |
| 标签面积保真度 | mean 0.9675，p10 0.9215，min 0.8154 |
| 划分变化 | train 335 → 375，val 29 / test 30 **零改动**（CI 逐字节断言） |
| 测试规模 | 36 → **228** 条 pytest（W2 新增 179 条，后续补环境指纹与评测入口 13 条） |
| B0 基线（2 类子集） | mask mAP50-95 **0.2453** @ ep75，95/100 早停，15.7 min，峰值显存 2.3~3.0 GiB |
| 环境漂移实测 | cv2 4.10 与 5.0 转出的 788 个标签里 **12 个差约 1 px**，含 val/test 各 1 个 |
| 可复现性 | 同环境同参数重跑：转换 789 文件、合成 870 文件**逐字节全等** |

15 类全量数据与消融结果（2026-10-22）：

| 项 | 值 |
| --- | --- |
| 全量下载 | 15 类 / 5354 图，逐文件 sha256 对 HF LFS `lfs.oid`，约 12 min（5~7 MB/s） |
| 全量转换 | 27.7 s，硬链接落盘；train 4490 图 / 1261 实例，val 429 / 299，test 435 / 328 |
| 消融 B1（+Copy-Paste） | mask mAP50-95 0.2453 → **0.4071**（相对 +66%），mask P 0.3885 → 0.9025 |
| 消融 C1（+Copy-Paste+Focal） | **0.1953**，比 B1 低 0.2118 —— focal 在这个规模上是负收益 |
| 消融 B4（去背景负样本） | 0.0109，但 train 只剩 66 图 / 23 ep ≈ 207 iter（B0 是 3990 iter），**步数不可比，结论挂起** |
| GPU 稳定性 | 一晚两次 TDR：02:12 蓝屏 `0x00000116`（nvlddmkm.sys），02:56 `CUDA error: unknown error`。已给队列加自动重试 + 残档留档 |

完整参数表、拒绝采样统计与两个 focal 踩坑记录见
[docs/augmentation.md](docs/augmentation.md)。
训练结果（含 15 类全量基线与消融）见 [docs/baseline.md](docs/baseline.md)。

W3 部署链路实测（2026-10-22，RTX 4060 Laptop 8GB；抽样口径 = 入库清单，sha256 `eccfd964426a2637`）：

| 项 | 值 |
| --- | --- |
| ONNX 导出 | `baseline15_fp32.onnx` **45.53 MB**，opset 17 / IR 8，动态 batch，357 节点 22 算子，checker 通过，8.2 s |
| 抽样口径 | `configs/bench_sample.json`，64 张 / 母体 435，覆盖全部 15 类；align 取前 16、bench 取前 30，共用同一个头 |
| 第 1 层 逐张量对齐 | 16 张 test 图，PyTorch(cpu) vs ONNX Runtime：`output0` rel_l2 **2.4e-7** / cosine **1.00000000**，`output1` rel_l2 1.3e-6 / cosine 1.00000000，8.6 s |
| 第 2 层 端到端 mAP | test 435 图重算（整份，与清单无关）：mask mAP50-95 0.3740 → **0.3741**（+0.0001），box 0.4229 → 0.4228，63.9 s |
| 第 3 层 单帧延迟 | batch=1 / 640 / 30 帧：ORT(CPU) mean 94.58 ms · p95 99.59 · 10.57 FPS；PyTorch(cuda:0) mean 13.13 ms · p95 15.91 · 76.17 FPS |
| verdict | **pass**，三层全跑 81.5 s，落 `runs/segment/baseline15/verify_onnx_fp32.json`（不入库） |
| 测试规模 | 270 → **391** 条 pytest（W3 新增 121 条：deploy 51 / export 11+4skip / verify 59） |

ORT 那行慢是**设备差不是后端差**：本机装的是 onnxruntime CPU 版，providers 里没有
CUDAExecutionProvider（只有 `AzureExecutionProvider, CPUExecutionProvider`），
这组数字是留给 TensorRT 三档量化当同机基线的。

表里是**固定清单口径**这一轮。生成清单之前那轮（第 3 层 ORT mean 98.14 ms / PyTorch 16.08 ms）
用的不是同一批图，两轮数字不可直接相减 —— 为什么要固定、固定成什么样，见
[docs/deploy.md](docs/deploy.md) §4.0；完整口径、读数提醒与踩坑记录同见该文。

W3b TensorRT 三档量化实测（2026-10-23，同一台机器同一轮压测；校准集 `configs/calib_sample.json`，
train split 500 / 4490 张，sha256 `20b7ab0c9ce76175`）：

| 档 | engine | 层数 | 构建 | mask mAP50-95 | Δ | mean / p95 ms | FPS | 较 `.pt` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FP32（TRT 默认含 TF32） | 65.25 MiB | 253 | 94.0 s | 0.3707 | -0.0033 vs `.pt` | 9.51 / 11.35 | 105 | 1.46× |
| FP16 | 24.58 MiB | 190 | 210.9 s | 0.3702 | -0.0005 vs FP32 档 | 6.73 / 10.66 | 149 | 2.07× |
| INT8（minmax + 头配方） | **12.55 MiB** | 168 | 152.0 s | 0.3649 | -0.0058 vs FP32 档 | **5.77** / 9.73 | **173** | **2.41×** |
| 同场对照 PyTorch `.pt`(cuda:0) | — | — | — | 0.3740（基准） | — | 13.90 / 15.89 | 72 | 1.00× |

verdict **pass**，三层判据全过；INT8 的掉点判据是 ≤0.03，实测 -0.0058 只用掉 19%。
体积压缩 FP16 2.7×、INT8 **5.2×**；进程显存增量 FP32 150 MiB、FP16 / INT8 各 52 MiB。

三条读数提醒（都在 [docs/deploy.md](docs/deploy.md) §8 里写透了）：

- **TF32**：TensorRT 默认打开 `BuilderFlag.TF32`，"FP32 档"跑的是 10-bit 尾数，逐张量 rel_l2 3.3e-3
  过不了 1e-3；`--no-tf32` 另建真 FP32 档后掉到 **2.5e-7**（cosine 1.0），代价 12% 延迟（9.51 → 10.71 ms）。
- **延迟只能同一轮内比**：笔记本 GPU 的时钟与温度，第一轮 FP16 mean 4.81 ms、第二轮 6.73 ms（+40%），
  同轮内三档排序两轮一致。
- **构建本身不确定**：单跑 INT8 得 166 层 / 12.53 MiB，三档连跑得 168 层 / 12.55 MiB，比对要看 `engine_sha256`。

测试规模 391 → **563** 条 pytest（W3b 新增 172 条：trt_infer 36 / make_calib_list 24 / build_trt 50 / bench_trt 62）。
这 172 条一条都不需要装 tensorrt —— `import tensorrt` 全延迟到函数内，CI 只跑逻辑不跑 GPU。

W4① 推理流水线实测（2026-10-23，同一台机器同一轮；30 帧 = 入库清单头部，`--repeats 10` → 每组 300 帧）：

| 组 | 口径 | FP32 | FP16 | INT8 | 瓶颈阶段 |
| --- | --- | --- | --- | --- | --- |
| A 含磁盘 | 读盘+解码+推理+后处理全链路，**对外报数用这组** | 36.86 → **57.53** FPS（1.561×） | 34.55 → **55.82**（1.616×） | 33.12 → **51.77**（1.563×） | decode（util 1.00） |
| B preload | 图先常驻内存，只量推理链路上限 | 126.36 → **158.95**（1.258×） | 215.46 → **305.52**（1.418×） | 233.61 → **330.05**（1.413×） | decode / infer 打平 |

箭头左边是同档**串行单帧**、右边是 **pinned + 三线程流水线**，加速比只在同档同组内成立。
A 组里 pinned 与 plain 基本打平（fp32 两者流水线都是 57.53 FPS；plain 的加速比 1.441× / 1.642× /
1.705× 忽高忽低，是因为分母——它自己的串行基线——变了，不是因为流水线更快）。
瓶颈是 16~19 ms 的读图 + letterbox，拷贝那点开销藏在它后面，**pinned 的账要到 B 组才算得出来**：
INT8 流水线 330.05 vs plain 263.20（**+25%**），省下的正是每帧新建可分页缓冲的那次同步拷贝。
两组口径与这笔账见 [docs/pipeline.md](docs/pipeline.md) §5 / §6.1 / §6.2。

吞吐是拿单帧延迟换的，这笔账也量了（500 帧 = 50 张 × 10 轮，A 组含磁盘，同一台机器）：
串行端到端 **P50 24~27 ms / P95 29~33 ms**，流水线 **P50 98~107 ms / P95 118~132 ms**，
放大 **4.0~4.2×**，换来 ~40 → ~65 FPS（**+60%**）。串行的在飞帧数按 Little's law（L = λ·W）反推是
**1.00**（三位有效数字内），等于自证延迟的时间戳没打错地方；流水线约 **6.4 帧在飞**，
缓冲上限 15 只用了 43%（瓶颈在源头，加缓冲只加等待）。CPU 从 **1.10 核当量**涨到 **1.65 核**
（峰值 2.81，16 逻辑核）；GPU util A 组只有 **28~37%**，和 TensorRT 侧的阶段 util 0.33~0.46 互证，
把磁盘拿掉（B 组）才冲到 **92.9% 均值 / 95% 峰值**，INT8 pinned 流水线 **316.66 FPS**。
离线批量质检要吞吐、在线联动产线要延迟（那时该把 capacity 调小甚至退回串行），**两个数都得报**。
详见 [docs/pipeline.md](docs/pipeline.md) §6.5。

与 ultralytics 参照实现对拍（同图同阈值，FP32/FP16 13/13、INT8 17/17 检出全部配上）：
match_rate **1.0000** / 框 IoU **1.0000** / 掩码 IoU 0.9100~0.9158（判据 0.85），三档各 10/10 条判据
全过、verdict **pass**。掩码那 0.09 的差不是随机噪声：第一版把 proto mask 按框尺寸拉伸是错的
（9×9 小框 IoU 只有 0.5625，框越大越看不出来），修法与面积扫描验证见
[docs/pipeline.md](docs/pipeline.md) §4.2。常驻显存与 pinned 主机内存各 **18.893 MiB**
（pool_size 2；调到 3/4/6 吞吐几乎不动，见 §7）。

W4② C++ 版实测（2026-10-23，`cpp/build/bench.exe`，fp16 engine，capacity 4，500 帧 = 50 张 × 10 轮；
工具链是这轮现装的：MSVC 14.44 + Win11 SDK 26100 + conda 的 cudart 头/nvcc + OpenCV 5.0 独立前缀）：

| 层 | 对拍内容 | 判据 | 实测 |
| --- | --- | --- | --- |
| A | decode：C++ `imdecode` vs Python `cv2` | 逐字节一致 | **5/5** |
| B | 前处理：融合 kernel vs Python `GpuPreproc` | max_abs / cosine | **max_abs 0 / cosine 1.0（逐位相等）** |
| C | 后处理移植：同一份 engine 输出喂两边 | 全 tuple 相等 | **20/20 帧** |
| D | 检测级：C++ vs Python 全链路 | match_rate ≥0.98 / 框 IoU ≥0.99 | **1.0000 / min 0.9967**（13 检出全配上） |

500 帧：serial **48.42 / 48.23 FPS**（P50 21.8 / 22.1，P95 28.6，**P99 32.2**），
流水线 **59.47 FPS**（另一轮 61.59；P50 **4.1**，P95 6.5，**P99 10.8**）；
`(n_det, mask_pixels, score_sum)` 两轮 **500/500 全等，n_diff = 0**，checksum dets **270**
与 Python 版 500 帧完全一致。CPU 核当量 1.00 → **1.31**，GPU util 21.9% → 25.1%（激活显存 18.29 MiB）。

**加速比 1.23×（另一轮 1.28×），没到判据要求 1.44×** —— 这一条照实写：C++ serial 已经比
Python 同清单 serial 快（48.4 FPS vs 40.2~43.6 FPS，+11~21%），流水线的上限又被 decode 段压住，
decode 单飞约 15 ms/帧、理论上限 ~66 FPS，实测 59.5（Amdahl，瓶颈仍在 `imdecode`）。收益落在别处：P50 **21.8 → 4.1 ms（5.3×）**、
吞吐比 C++ serial +23%（比 Python 同口径 fp16 流水线 +31~36%）、CPU 更省。下一个台阶是 nvJPEG/GPU 解码，不是更激进的三线程。
掩码那条判据已补齐为**真位图 IoU**（`--dump-masks` 探针，2026-10-23 晚）：30 帧 / 13 个检出，
逐帧均值 IoU **min 0.9972 / mean 0.9995**，判据 0.85 过线；同 bin 的 C2 层 IoU = 1.0000。
此前用像素数代理（30 帧 24 帧计数相等，其余差 ≤14 px / 0.04%）的历史记录与补法见
[docs/cpp_bench.md](docs/cpp_bench.md) §2.1 / §2.2。

移植时被对拍抓到的三个真 bug（都写在 cpp_bench.md §4）：掩码系数沿属性维的步长写成 1
（帧 0 掩码 102559 → 4682）、掩码解码喂了原图坐标而不是 640 域框（→ 72140）、
以及 Windows 中文路径三连（MSVC `main` 的 argv 是 GBK，`filesystem::u8path` 一路要用对）。

测试规模 563 → **892** 条 pytest（W4① 新增 329 条：pipeline 183 / pinned 59 / prof 87，其中 500 帧端到端延迟与资源采样这轮又补了 102 条）。这 329 条一条都不碰 GPU —— 环形队列、令牌池、CUDA 流、`nvidia-smi` 子进程全用假对象测，
连"每帧只 synchronize 一次""出错要把令牌还回池子""采样线程遇 `[N/A]` 不许抛"都是断言出来的。
延迟那条还钉了一个反直觉的口径：**时间戳必须打在 `put` 之前**，否则背压被藏起来、P95 好看但没意义。

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