# GPU letterbox 前处理预研（路线 C，2026-10-23）

**问题**：把 W4① 流水线 decode 段里的 letterbox + 归一化搬上 GPU，能省多少？
给 W4②（C++ 版）一个不用边写边猜的目标值。

**工具**：`tools/bench_gpu_preproc.py`（新增）
**原始数据**：`runs/logs/gpu_preproc_micro.json`（微观 500 帧分段）、
`runs/logs/gpu_preproc_e2e.json`（端到端 2 轮 × 4 配置 × 500 帧）、
`gpu_preproc_micro_smoke.json`（冷启动证据）
**环境**：`W:\Anaconda\envs\induvision\python.exe`，torch 2.13.0+cu130，
RTX 4060 Laptop 8 GB，fp16 engine，清单 `configs/bench_sample.json` 前 50 张 ×10 轮

---

## 1. 结论（先看这个）

1. **decode 段的 16~19 ms 要拆开看：imdecode ~15 ms，letterbox 链只有 ~4 ms。**
   能搬上 GPU 的只有后者。搬完（含 pinned 上传 0.26 + H2D 0.56 + GPU ops 1.08 +
   D2H 拷回 0.68 = 2.58 ms），每帧净省 **约 1.8 ms**。
2. **端到端仍然值得**：A 组（含磁盘）500 帧、fp16、同轮对比两轮，
   流水线 FPS **45.4 / 45.7 → 53.3 / 58.4（+17.4% / +27.6%）**，
   e2e P50 **140 → 109~120 ms**。串行比两轮分得开（1.04 / 1.25），
   单轮读数照旧不可信，同轮内比值才是口径。
3. **下一个台阶不是 letterbox，是 imdecode**。GPU 化之后 decode util 仍是 1.00，
   段均值从 ~19 ms 降到 ~17.3 ms——瓶颈原封不动地待在图像解码上。
   C++ 版想在 1.44× 之上再上台阶，方向是解码段，而不是更激进的
   warpAffine 融合。
   > **2026-10-23 晚补记（接手 Agent）**：这条当时写的是"nvJPEG / GPU 解码"，
   > 开工预研发现前提不成立——MVTec 是 PNG，nvJPEG 只解 JPEG，能解 PNG 的
   > nvImageCodec 没有 Windows 分发。真正落地的是 **decode 线程池**
   > （`bench --decode-threads N`，PNG 不动、checksum 逐位一致），
   > 62 → 219 FPS（dt=4），见 cpp_bench.md §6。
4. **数值对齐没问题**：torch bilinear（float 域，antialias=False）vs
   cv2 INTER_LINEAR（uint8 域），495 帧最坏差
   max_abs **0.00298** / rel_l2 **0.0053** / cosine_min **0.999987**。
   检测级：500 帧检出总数完全一致（270），仅 5 帧掩码像素数差 ~0.1%、
   分数和差 ~0.4%（阈值边缘效应，方向不一）。C++ 版对拍仍按 cpp_infer.md §4 判据走。

## 2. 微观分段（500 帧，串行逐帧，分段含 `torch.cuda.synchronize` 同步开销）

| 路径 | 分段 | mean (ms) |
| --- | --- | ---: |
| CPU | read_image_bgr（imdecode） | 15.019 |
| CPU | preprocess（letterbox+BGR→RGB+CHW+/255） | 4.038 |
| GPU | read_image_bgr（同上，不动） | 14.675 |
| GPU | 图拷入 pinned 暂存 | 0.260 |
| GPU | H2D（non_blocking + sync） | 0.557 |
| GPU | GPU ops（flip/归一化/bilinear 缩放/pad） | 1.081 |
| GPU | D2H 拷回 pinned 输出 + sync | 0.680 |
| GPU | **合计（不含读图）** | **2.578** |

对照冷启动：首帧 GPU ops 12.2 ms（kernel 编译/分配器预热），5 帧小样本会把这个
数字当真——**所有 GPU 计时必须过预热**，smoke 那份 JSON 就是证据。

## 3. 端到端（A 组含磁盘，500 帧，fp16，同轮交错，两轮）

| 轮 | cpu_serial | cpu_pipeline | gpu_serial | gpu_pipeline |
| --- | --- | --- | --- | --- |
| 0 | 38.77 FPS | 45.44 FPS | 40.18 FPS | **53.33 FPS** |
| 1 | 34.92 FPS | 45.74 FPS | 43.60 FPS | **58.37 FPS** |

- 同轮 GPU/CPU FPS 比：serial **1.036 / 1.249**，pipeline **1.174 / 1.276**。
- e2e P50：cpu_pipeline 140 ms 级 → gpu_pipeline **109~120 ms**。
- 流水线 util（两轮同型）：decode 仍 ~1.00，infer ~0.28，post ~0.09——
  decode 依旧是瓶颈，只是瓶颈内容变了（见 §1 第 3 条）。
- 保序与检出总数：全部一致。

## 4. 口径与局限（引用前必读）

- 分段计时含同步开销，**别拿 §2 的 2.578 ms 直接减出流水线收益**，端到端看 §3。
- GPU 变体每帧多一笔 **4.688 MiB D2H**（现有 `TrtRunner`/`PinnedTrtClient` 吃 host 缓冲）。
  **C++ 版 `setTensorAddress` 直接吃 device 指针，没有这笔** → 本工具测出的收益是
  C++ 版的**保守下界**。
- torch CUDA 上下文与 TRT 共卡，压测期间显存多占一份上下文（几百 MiB 量级），
  8 GB 卡上做正式压测时要记进环境说明。
- 预热帧（前 5 帧）不参与张量比对；分段计时未剔除预热（均值含首帧污染，
  P50/P95 不受影响，见 JSON）。
- 本轮没测 GPU 解码——那是下一个实验，不是本文件的结论。（后续实测修正：见 cpp_bench.md §6.1，nvJPEG 此路不通，decode 线程池落地。）

## 5. 实现（`tools/bench_gpu_preproc.py`）

- `GpuPreproc`：BGR uint8 → pinned 暂存 → H2D → GPU 上
  flip/`/255`/`F.interpolate(bilinear, antialias=False)`/`F.pad(114/255)` → D2H
  拷回预分配 pinned `(1,3,640,640)` float32。暂存/输出缓冲全预分配。
- `letterbox_meta` 只算几何参数，`verify_metas()` 拿真 `deploy.letterbox`
  逐字段核对（ratio/new_unpad/top/left），防两边漂移。
- `GpuLetterboxView`：`LetterboxResult` 最小替身。`postprocess` 全链只读
  ratio/dw/dh/top/left/orig_shape 和 `image.shape[0]`（=imgsz），后者用共享
  空占位数组带过去，语义不变——**不需要为 GPU 版改 pipeline 一行**。
- 全量单测跑过：892 passed / 6 skipped（22.99 s，2026-10-23，与 W4① 基线一致）。
