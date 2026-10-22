# 项目说明：工业缺陷检测与端侧部署闭环

一句话讲完这个项目做了什么：在 MVTec AD 上训练一个能分割 15 类工业缺陷的 YOLOv8-seg 模型，
再把它从 PyTorch 搬到 ONNX（下一步是 TensorRT），每搬一步都用数字证明「搬完没变味」。

按 **背景 → 方法 → 结果** 三段读。每段末尾链到细节文档，本文只讲清「做了什么、为什么这么做」。

---

## 1. 背景：要解决什么问题

### 1.1 为什么选这个题目

求职方向是图像算法。手上原有的项目是硕士课题的显微图像分析，领域太窄，
面试官容易读成「把论文换个壳」。所以另起一个业内一听就懂的题目：工业缺陷检测。
数据集选 MVTec AD —— 公开、15 类、业内标准，不需要花时间解释数据是什么。

### 1.2 这个题目难在哪

两个数字先摆出来（本机实测，口径见 [dataset.md](dataset.md)）：

- **缺陷样本稀缺**：缺陷图只占 23.6%（93/394，冒烟集口径），背景负样本是它的 3.2 倍；
- **目标极小**：缺陷面积占比 p50 只有 3.50%，最小 0.24%。

「样本少 + 目标小 + 类别不均衡」三件事叠在一起。照官方划分直接训，
很容易得到一个 loss 看着在降、实际什么都检不出来的模型。W2 整段就是在处理这件事。

### 1.3 一个绕不开的口径问题

MVTec AD 是为**无监督异常检测**设计的：官方 `train/` 里只有 `good` 正常样本，
一张缺陷标注都没有。要做 YOLO-seg 这种有监督分割，标注只能从官方 `test/` 的缺陷图里来。

本项目的做法：在官方 `test/` 内部按 `category × defect_type` 分层划 70/15/15，固定种子。

**这条必须写在每个对外指标旁边**：本项目的 mAP 和论文里的 AUROC-PRO 不是一回事，不能直接比。

### 1.4 为什么不止于「训个模型」

只报一个 mAP 的项目太多了。工业场景真正卡人的是另一件事：模型能不能在产线上的盒子里跑到实时。
所以后半段是导出、量化、C++ 推理服务、压测。

硬件是 RTX 4060 Laptop 8GB / Windows 11，所有工程决策都围绕「8GB 显存怎么把链路跑完整」来做。
文档里的数字全部本机实测，没跑到的一律标 **待测**，不预填、不估。

---

## 2. 方法：具体怎么做的

四个阶段，每阶段都有单测和文档兜着。

### W1 数据管线

`tools/download_mvtec.py` → `tools/convert_mvtec_to_yolo.py`

1. 取数据。官方站要勾 EULA 不能直连，走 HF 镜像；**逐文件对 sha256**（校验值取自 HF LFS 的 `lfs.oid`）。
2. 掩码 PNG → YOLO-seg 多边形 txt。三个决策：
   - 划分只落在官方 `test/` 内部（§1.3）；
   - 面积过小的缺陷图默认**丢弃**，不留成空标签 —— 留着等于告诉模型「这块缺陷是背景」，是标签噪声；
   - 用 `RETR_EXTERNAL` 取外轮廓，代价可量化：`contamination` 这类带内部孔洞的缺陷会被填掉，
     偏大多少用 `area_fidelity = 多边形面积 / 掩码面积` 记录在案。
3. 转换默认硬链接落盘，同盘不额外占空间。

细节见 [dataset.md](dataset.md)。

### W2 增强与消融

`induvision/defect_bank.py` + `copy_paste.py` + `balance.py` + `focal.py`，
入口 `tools/synthesize_defects.py` / `tools/train_seg.py`

1. **缺陷实例库只从 train 划分抠**，从源头杜绝 val/test 泄漏。CI 里逐字节断言 val/test 没被改动。
2. Copy-Paste 贴图（poisson / alpha 两种融合），标签同步重建，按少数类配额补齐。
3. Focal Loss 做成可装可卸的补丁（LIFO），给 ultralytics 的分类分支换损失。
4. 消融用幂等队列跑：跑完的组自动跳过，被系统重启或驱动 TDR 打断后，重跑同一条命令就从断点续。
   （这台机器一晚 TDR 两次 —— 一次蓝屏 `0x00000116`，一次 `CUDA error: unknown error`，
   所以长训练一律走守护脚本，不裸跑。）
5. **报成绩一律在 test split 上重算**：`best.pt` 是靠 val 早停选出来的，
   拿 val 的数当最终成绩，等于在选过的数据上报告结果。

细节见 [augmentation.md](augmentation.md) 与 [baseline.md](baseline.md)。

### W3 部署与对齐

`induvision/deploy.py` + `tools/export_onnx.py` + `tools/verify_onnx.py`

1. **导出**：ONNX opset 17 / 动态 batch。导完立刻静态体检 —— opset 有没有真的落到 17、
   输入第 0 维是不是符号维、`onnx.checker` 过不过、图里有哪些算子（TensorRT 认不认某个算子，
   看算子表比翻日志快）。导出不等于导对。
2. **三层校验**。换后端最容易被含糊过去的一句话是「精度基本没掉」，这句话拆成三层来量：

   | 层 | 问什么 | 怎么量 | 判据 |
   | --- | --- | --- | --- |
   | 1 | 算子层面等价吗 | 同一批输入、同一份前处理，两个后端的原始输出张量逐个比差 | cosine ≥ 0.9999 且 rel_l2 ≤ 1e-3 |
   | 2 | 业务指标等价吗 | 拿 `.onnx` 当权重在 test split 上重算 mAP，与 `.pt` 参照相减 | \|Δ mask mAP50-95\| ≤ 0.01 |
   | 3 | 换来多少速度 | batch=1 单帧延迟的 mean / p50 / p95 / p99 / max | 不设判据，只记录 |

   第 1 层最灵敏，也是唯一能把差异归因到「导出与算子实现」的一层 —— 它绕开了 NMS、阈值、匹配这些后处理。
   第 2 层才是能写进报告和简历的话。第 3 层只报均值等于没报，尾巴由队列抖动和后处理毛刺决定，所以分位一起给。
3. **前处理自己复刻一份**。比差的前提是两边吃到的输入逐字节相同 —— ratio 取整方式、pad 是否居中、
   BGR/RGB 顺序、有没有除 255、CHW 还是 HWC，任一处不同，量到的差就全是前处理口径的差。
   所以 `deploy.py` 把 ultralytics 的 `LetterBox(auto=False)` + `predictor.pre_process` 复刻成
   numpy/opencv 实现，两个后端共用。复刻而不是 `import ultralytics`，是为了 CI 不装 torch 也能测这段逻辑。
4. **抽样口径固定成入库清单** `configs/bench_sample.json`：一条顺序确定的相对路径列表 + sha256 指纹，
   align 取前 16、bench 取前 30，共用同一个头。在这之前抽样长度跟着 `--align-n`/`--bench-n` 走，
   加不加一个参数就会换掉整批图，两轮的延迟没法直接相减。清单对不上当前数据集时降级成即时抽样，
   并在 stdout / JSON / Markdown 三处同时留痕，不会悄悄发生。

细节见 [deploy.md](deploy.md)。

### W3b 三档量化：已完成

TensorRT 10.16.1.11 装上了（pip wheel，见 [environment.md](environment.md) §6），三档量化实测通关，
verdict **pass**。原计划用 `trtexec`，但本机根本没有 `trtexec.exe` —— wheel 只带运行期 `.dll`，
不带头文件、`lib` 和命令行工具，所以构建全部改走 TensorRT Python API（`tools/build_trt.py`）。
口径、判据与踩坑见 [deploy.md](deploy.md) §8。

这一轮最值钱的不是数字，是排查：**TensorRT 默认打开 `BuilderFlag.TF32`**，所谓"FP32 档"实际跑的是
10-bit 尾数的 TF32，逐张量 rel_l2 3.3e-3 卡不过 1e-3 的判据 —— 链路没错，是数值格式的机器精度不够。
加 `--no-tf32` 另建真 FP32 档，rel_l2 掉到 **2.5e-7**（cosine 1.0），与 W3a 的 ORT 口径一致，
代价 12% 延迟（9.51 → 10.71 ms）与 21 层额外开销。判据也据此改口径：rel_l2 按数值格式的机器精度定
（FP32/FP16 收到 1e-2），最终判据是端到端 mAP。

### W4 拆两步

1. **Python 侧先把工程要点做出来并可测**（不依赖 C++ 工具链）：pinned memory 零拷贝 + 异步 H2D/D2H、
   解码 / 推理 / 后处理三线程流水线 + 环形队列，NMS 与掩码解码落到 numpy/torch，
   用同一份 30 张清单量吞吐，和 §3.3b 的单帧延迟对照，直接产出"较单线程提升 N×"的数；
2. **C++ 版**：TensorRT C++ API（engine 反序列化 → CUDA 流 → `cudaWarpAffine` 前处理 → 推理 → 后处理）
   + CMake 工程 + 500 张端到端 P50/P95 压测。本机实测 `nvcc` / `cmake` / `cl` / `ninja` / `msbuild`
   全部 not found，也没有 CUDA Toolkit 与 VS 安装目录，得先装 VS Build Tools + CUDA Toolkit（数 GB）。

---

## 3. 结果：实测指标

### 3.1 数据

| 项 | 值 |
| --- | --- |
| 全量下载 | 15 类 / 5354 图，逐文件 sha256 全过，约 12 min |
| 全量转换 | 27.7 s，硬链接落盘 |
| 划分结果 | train 4490 图 / 1261 实例，val 429 / 299，test 435 / 328（15 类全量） |
| 缺陷图占比 | 93/394 = 23.6%，背景负样本是其 3.2 倍（冒烟集口径） |
| 缺陷面积占比 | mean 5.85%，p50 3.50%，min 0.24%，max 27.42%（冒烟集 bottle+toothbrush 口径） |

### 3.2 训练与消融

baseline15（15 类全量，**test split**）：

| 指标 | 值 |
| --- | --- |
| mask mAP50-95 | **0.3740** |
| mask mAP50 | 0.6730 |
| box mAP50-95 / mAP50 | 0.4229 / 0.6652 |
| 收敛 | best @ ep75，ep95 早停 |

W2 消融（2 类子集口径，**绝对值不可外引** —— val 仅 29 图 / 16 实例，只看组间相对关系）：

| 组 | mask mAP50-95 | 读数 |
| --- | --- | --- |
| B0 基线 | 0.2453 | — |
| B1 +Copy-Paste | **0.4071** | 相对 +66%，mask P 0.3885 → 0.9025 |
| B3 整张复制过采样 | 0.4003 | 与 Copy-Paste 基本打平（差 0.0068） |
| C1 +Copy-Paste+Focal | 0.1953 | 比 B1 低 0.2118，**focal 在这个规模上是负收益** |
| B2 只有 Focal | 0.0039 | 直接学崩 |

### 3.3 部署对齐（固定清单口径，sha256 `eccfd964426a2637`）

| 项 | 值 |
| --- | --- |
| ONNX 产物 | `baseline15_fp32.onnx` **45.53 MB**，opset 17 / IR 8，动态 batch，357 节点 22 算子，checker 通过 |
| 第 1 层 逐张量 | 16 图：`output0` rel_l2 **2.4e-7** / cosine **1.00000000**；`output1` rel_l2 1.3e-6 / cosine 1.00000000 |
| 第 2 层 端到端 | test 435 图重算：mask mAP50-95 0.3740 → **0.3741**（+0.0001，判据 ±0.01） |
| 第 3 层 延迟 | ORT(CPU) mean 94.58 ms · p95 99.59 · 10.57 FPS；PyTorch(cuda:0) mean 13.13 ms · p95 15.91 · 76.17 FPS |
| verdict | **pass**，三层全跑 81.5 s |

第 3 层那两行是「**CPU 比 GPU 慢**」，不是「ORT 比 PyTorch 慢」—— 本机装的是 onnxruntime CPU 版，
providers 里没有 CUDAExecutionProvider。这组数字是留给 W3b 三档量化当同机基线、
给 W4 的 C++ 服务当「Python 单线程能跑多快」的对照组。

### 3.3b W3b 三档量化（TensorRT 10.16.1.11，同机同轮）

| 档 | engine | 层数 | 激活显存 | mask mAP50-95 | Δ | mean / p95 ms | FPS | 较 `.pt` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FP32（TRT 默认含 TF32） | 65.25 MiB | 253 | 39.84 MiB | 0.3707 | -0.0033 vs `.pt` | 9.51 / 11.35 | 105 | 1.46× |
| FP16 | 24.58 MiB | 190 | 18.29 MiB | 0.3702 | -0.0005 vs FP32 档 | 6.73 / 10.66 | 149 | 2.07× |
| INT8（minmax + 头配方） | **12.55 MiB** | 168 | 18.41 MiB | 0.3649 | -0.0058 vs FP32 档 | **5.77** / 9.73 | **173** | **2.41×** |
| 同场对照 PyTorch `.pt`(cuda:0) | — | — | — | 0.3740（基准） | — | 13.90 / 15.89 | 72 | 1.00× |

verdict **pass**；三档构建 456.9 s，压测 78.9 s；校准集 train split 500 / 4490 张
（`configs/calib_sample.json`，sha256 `20b7ab0c9ce76175`）；进程显存增量 FP32 150 MiB、FP16 / INT8 各 52 MiB。
体积压缩 FP16 2.7×、INT8 **5.2×**。gate 基准口径：FP32 档对 `.pt`（0.3740），FP16 / INT8 对本轮 FP32 档（0.3707），
两列 Δ 都给了，别混着减。

读数提醒三条：**延迟只能同一轮内比**（笔记本 GPU 时钟与温度，第一轮 FP16 mean 4.81 ms、第二轮 6.73 ms，+40%，
同轮内排序两轮一致）；**构建本身不确定**（单跑 INT8 166 层 / 12.53 MiB，三档连跑 168 层 / 12.55 MiB，
比对看 `engine_sha256`）；**TF32 不是链路错**（`--no-tf32` 对照档 rel_l2 2.5e-7，见 [deploy.md](deploy.md) §8.4）。

### 3.4 工程

| 项 | 值 |
| --- | --- |
| 单测 | **563 条 pytest**（W3b 新增 172 条：trt_infer 36 / make_calib_list 24 / build_trt 50 / bench_trt 62），纯 CPU、不联网、不需要真实数据集、也不需要装 tensorrt（`import tensorrt` 全延迟到函数内），base 环境 19.7 s |
| CI | ubuntu + windows × py3.10/3.12 共 4 job，全绿（`21ce9bb` 起）。CI 比本地少收 `tests/test_focal.py` 一个文件（模块级 `importorskip("torch")`，CI 刻意不装 torch）；W3b 新增的 4 个测试文件在 CI 里全量跑 |
| 可复现性 | 同环境同参数重跑：转换 789 文件、合成 870 文件**逐字节全等** |
| 入库边界 | CI 断言守着：`git ls-files` 出现 `data/`、`weights/`、`out/`、`runs/` 或任何 `.pt/.onnx/.engine/.trt` 就失败 |

顺带记一条跨平台坑，因为它把 ubuntu job 连着挂了 7 个提交才被发现：
`tools/summarize_ablation.py` 的 `dataset_name()` 原先用 `Path()` 直读 `args.yaml` 里的 `data`，
而那条路径是 Windows 训练写下的反斜杠串；POSIX 上 `Path` 不把 `\` 当分隔符，
整条路径被当成一个文件名，`parent.name` 于是返回空串。修法是先把 `\` 换成 `/` 再解析，
函数返回值从此不随运行平台变化，四个平台组合同一套期望值。

同期给 CI 的 `Unit tests` 步骤加了一件事：pytest 挂掉时把短摘要、失败块名、`E` 断言行
写成 workflow annotation。原因是 actions 的日志下载要鉴权（本地没有 token 直接 403），
而 annotation 走公开 API 就能读 —— 上面那条反斜杠的 bug 就是这么定位的，不用去点网页。

---

## 4. 文档索引

| 想看什么 | 去哪 |
| --- | --- |
| 环境怎么搭的、版本钉在哪 | [environment.md](environment.md) |
| 数据集下载 / 转换 / 划分口径 | [dataset.md](dataset.md) |
| 训练记录、15 类基线、逐类 AP | [baseline.md](baseline.md) |
| Copy-Paste 合成与 focal 消融全过程 | [augmentation.md](augmentation.md) |
| ONNX 导出、三层校验、固定抽样清单 | [deploy.md](deploy.md) |
| 怎么把整条链路跑起来 | [../README.md](../README.md) 的「快速开始」 |