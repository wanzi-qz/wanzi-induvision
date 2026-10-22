# W4① 推理流水线：令牌缓冲池 + 三阶段线程流水线 + 自研后处理

一句话结论：把「读图 / 前处理 → TensorRT 推理 → NMS + 掩码解码」拆成三条线程用环形队列串起来，
配合 pinned memory 零拷贝缓冲池，**同一批图、同一台机器、同一轮**量下来吞吐提升
**1.44~1.71×**（含磁盘读图的真实口径）；把图预先常驻内存、隔离掉磁盘之后是 **1.19~1.42×**，
INT8 档流水线 **330 FPS**。三档全部通过 10 条判据，与 ultralytics 对拍 match_rate **1.0000**、
框 IoU **1.0000**、掩码 IoU **0.910~0.916**。
吞吐是拿单帧延迟换的：500 帧实测流水线端到端 **P50 98~107 ms / P95 118~132 ms**，
是串行（P50 24~27 ms）的 **4.0~4.2 倍**，代价换来 +60% 吞吐（§6.5）。
CPU 吃 **1.65 核当量**（串行 1.10），含磁盘那组 GPU util 只有 **28~37%**，
把磁盘拿掉才冲到 **93%**（§6.5.3）。

按 **背景 → 方法 → 结果 → 踩坑** 读。本文所有数字都来自本机实测，
原始报告在 `runs/logs/pipeline_baseline15*.json`（不入库），表格由脚本从 JSON 生成，不手抄。

---

## 1. 这一段要回答什么

W3b 已经把三档 engine 建出来并压过一轮（[deploy.md](deploy.md) §8），但那一轮量的是
**单帧延迟**：输入常驻显存、不含 H2D/D2H 拷贝、不含前后处理，一次只喂一帧。
它回答的是「kernel 有多快」，回答不了「产线上一秒能过几张图」。

差在三个地方：

1. **拷贝**：每帧 1×3×640×640 float32 = 4.7 MiB 要 H2D，输出 `output0` 1×51×8400 +
   `output1` 1×32×160×160 要 D2H。单帧延迟那轮把这些全省了。
2. **前后处理**：读图 + letterbox 在本机实测 **16~19 ms/帧**（含磁盘），比 INT8 的 kernel（2.9 ms）
   慢一个数量级。串行跑的时候 GPU 大部分时间在等 CPU。
3. **重叠**：串行是「CPU 干活时 GPU 闲着，GPU 干活时 CPU 闲着」。流水线就是把这两段叠起来。

所以这一轮要产出的是一个能写进报告的数：**端到端吞吐提升 N×**，而且 N 的分母必须是
同机同轮的串行实现，不能拿 §8.2 的单帧延迟去除（那条在 §5 单独讲）。

---

## 2. 为什么先做 Python 版

W4 原计划直接上 C++。本机实测 `nvcc` / `cmake` / `cl`(MSVC) / `ninja` / `msbuild` **全部 not found**，
也没有 CUDA Toolkit 与 VS 安装目录 —— TensorRT 是 pip wheel 装的，只带运行期 `.dll`，
不带头文件和 `lib`（详见 [deploy.md](deploy.md) §8.10）。装 VS Build Tools + CUDA Toolkit 是数 GB 的事。

但这一轮真正要验的工程要点跟语言无关：

| 要点 | 语言相关吗 |
| --- | --- |
| 环形队列解耦阶段、背压可观测 | 否 |
| 缓冲预分配、不每帧 malloc | 否 |
| pinned（page-locked）host 内存 + `non_blocking` 异步拷贝 | 否（CUDA 语义） |
| 输出缓冲按 slot 隔离，防跨帧数据竞争 | 否 |
| NMS / 掩码解码自己实现（脱离 ultralytics） | 否（C++ 版同样得自己写） |
| 保序、错误传播、超时收尾 | 否 |

C++ 版换的是 API 名字，不换结构。所以先用 Python 把结构立起来、**量出数**、用单测钉住语义，
工具链到位后照搬。这不是偷懒的替代品 —— 这一轮的产物里有 227 条单测和一份对拍报告，
C++ 版落地时直接拿它当参照实现。

---

## 3. 结构：三阶段 + 环形队列 + 令牌池

```
feeder ──q0──> [stage1 decode] ──q1──> [stage2 infer] ──q2──> [stage3 post] ──qout──> 主线程 collector
                读图 + letterbox        H2D → kernel → D2H       NMS + 掩码解码 + 坐标还原
                写进 pinned slot        只 sync 一次             读完 release 令牌
```

代码在 `induvision/pipeline.py`（769 行，只依赖 numpy / opencv / 标准库）、
`induvision/pinned.py`（291 行，torch/tensorrt 延迟导入）、
压测入口 `tools/bench_pipeline.py`（695 行）。

### 3.1 线程账：为什么是 5 条而不是 3 条

3 个 stage worker + 1 条 feeder + 主线程当 collector。

**feeder 必须独立成一条线程**，这是踩出来的：最初让主线程一边 `put(items)` 一边收结果，
`q0` 容量 4，主线程 put 到第 5 个就阻塞等空位 —— 而唯一能腾出空位的是 stage1，
stage1 的产物要一路走到 `qout` 才有人取，取的人是主线程，主线程正卡在 put 上。**死锁**。
把喂数据和收结果分到两条线程就没了：feeder 只管塞，主线程只管收。

主线程当 collector 而不是再开一条，是因为结果最终要回到调用方手里，
多一条线程只是多一次交接。收的时候按 `index` 塞进 dict，最后 `sorted` 输出 —— 见 §3.3。

### 3.2 环形队列：`RingBuffer`

自己写的定长环形缓冲，不是 `queue.Queue`。差别在于**它把背压变成可读的数**：

| 字段 | 含义 | 怎么读 |
| --- | --- | --- |
| `max_depth` | 历史最深水位 | 等于 capacity → 下游吃不消；一直是 1 → 队列没起缓冲作用 |
| `overflow_waits` | put 被「队列满」挡住的**调用次数** | 高 → 这个阶段比下游快，在等 |
| `underflow_waits` | get 被「队列空」挡住的调用次数 | 高 → 这个阶段比上游快，在饿 |
| `puts` / `gets` | 吞吐计数 | 收尾时两者应相等，不等就是漏了 |

`overflow_waits` / `underflow_waits` 的计数口径中途改过一次：原先写在 `while` 循环体里，
一次阻塞的 `put` 如果内部转了 3 圈就记 3 次；带 timeout 的路径更糟，一次 acquire 能记 2 次，
`pool_waits` 直接虚高。现在用 `waited` 标志位，**一次被挡住的调用只记一次**，
这个数才等于「有多少帧在这里排队过」，才能拿去和帧数比。有 5 条单测专门钉这个口径。

`close()` 是幂等的，且唤醒所有等待者：worker 收到 `None` 就退出循环，
`finally` 里关掉自己下游的队列，收尾信号顺着链一路传下去。任何一条线程崩了都不会留下悬挂的等待者。

### 3.3 保序

流水线天然乱序（stage2 可能已经在推第 n+1 帧，stage3 还在解第 n 帧），但**输出必须按输入顺序**，
否则「第 i 张图的检测结果」就不知道是谁的。做法是每个 item 带一个 `index` 走完全程，
collector 收进 `dict[index]`，最后 `sorted` 输出。

这条有硬判据：`order_ok()` 逐条比 `[r.idx for r in results] == [f.idx for f in frames]`，
压测报告里 `order_ok: true` 才算过。三档 × 两种缓冲 × 流水线，全部 true。

### 3.4 错误传播

stage 函数抛异常时：记下 `{stage, index, error}` → `stop.set()` → 跳出循环 → `finally` 关下游队列。
主线程收不满 n 个结果就退出，`run()` 最后把第一条错误包成
`PipelineError(stage, index, exc)` 重抛。**不吞异常**，而且报错里带着是哪个阶段、第几个 item ——
300 帧里某一帧读图失败，你要的是「第 137 帧、decode 阶段、FileNotFoundError」，
不是一个光秃秃的 traceback。

`join(timeout=60)`：收尾时 join 所有线程，超时就放着不管（线程都是 daemon），不让一条卡死的线程把进程挂住。

### 3.5 令牌池：`PinnedTrtClient`

这是这一轮真正抠性能的地方。`trt_infer.TrtRunner.__call__` 是「每帧新建缓冲」的写法：
`torch.from_numpy(x).to(device)` 走**可分页**内存 H2D（驱动内部还要先过一遍自己的 staging buffer），
输出张量每帧 `torch.empty` 一个新的，拷回 CPU 又是 `.float().cpu().numpy()` 连着两次落地。

`PinnedTrtClient` 改三件事：

1. **pinned host 缓冲**。`pin_memory=True` 的内存不被操作系统换页，DMA 能直接读写，
   `non_blocking=True` 才是**真的**异步 —— 对可分页内存，`non_blocking` 会静默退化成同步拷贝。
   这是最常踩的一个坑：参数写了，行为没变，而且不报错。
   前处理结果直接 `np.copyto(client.input_view(tok), chw)` 写进它的 `.numpy()` 视图，中间不多一份 ndarray。
2. **常驻 device 缓冲**。输入输出张量整轮只分配一次，每帧只做 memcpy，不进 torch 的缓存分配器。
   实测常驻显存 **18.893 MiB**、pinned 主机内存 **18.893 MiB**，三档一模一样（因为是按 shape 分配的，
   和精度无关），整轮 630 次 forward 不抖动。8GB 显存的机器上，「占用是一条平线」比「快 5%」更要紧。
3. **每帧只 `stream.synchronize()` 一次**。H2D、kernel、D2H 全部 `non_blocking` 排进同一条流，
   最后同步一次拿结果。同步三次和同步一次的差别不是三倍开销，是三次「等 GPU 排空」。

**为什么需要令牌（token）**：输出缓冲必须等 stage3 读完才能复用。只有一份缓冲的话，
stage2 推第 n+1 帧时会把 stage3 正在解的第 n 帧掩码覆盖掉 —— 症状是「偶发的检测结果和串行不一致」，
复现率跟机器负载有关，属于最难查的一类 bug。所以输入输出都按 slot 隔离，
令牌贯穿三个阶段：stage1 `acquire()` 拿令牌并往该 slot 写前处理结果，stage3 用完 `release()` 还回去。

两个实现细节值得记：

- **出错也要还令牌**。stage1 里 `acquire()` 之后到 `return` 之前的任何异常（读图失败、形状不对）
  都在 `except BaseException` 里 `release(tok)` 再 `raise`。漏了这一句，池子就永久少一个 slot，
  跑久了直接饿死 —— 而且是「越跑越慢」，不是「跑挂了」，更难定位。
- **`forward_array()` 自己 acquire 的令牌，`owned=True`**，调用方负责 release；
  流水线里传进来的令牌 `owned=False`，client 不动它。所有权写在返回值里，不靠约定。

`pool_waits` 这个数直接告诉你 `pool_size` 是不是给小了 —— §7 整节都在讲这个数怎么骗人。

---

## 4. 后处理自己写一份，并且对拍

`tools/bench_trt.py` 的端到端 mAP 走 ultralytics 的 `val`，后处理是它给的。
流水线要能脱离 ultralytics 独立跑（C++ 版更是必须自己写），就得自己实现 NMS 与掩码解码。
自己写完必须对拍，否则「跑通了」和「跑对了」是两件事。

### 4.1 口径

- `postprocess(out0, out1, lb, ...)`：`out0` 是 `(1, 4+nc+32, 8400)`，本项目 nc=15 → 51 个属性
  （框 cxcywh 4 + 15 类分数 + 32 个掩码系数），`out1` 是 `(1, 32, 160, 160)` prototype。
  `_as_predictions` 把 `(1, 51, 8400)` 转置成 ultralytics 内部用的 `(8400, 51)`，两种排布都收，
  靠「属性维等于 4+nc+32」来认；**两个维度都等于属性维（方阵）时直接报错**，让调用方显式给 `nc`，不猜。
  `nc` 不给就从形状反推（属性维 − 4 − proto 通道数），反推不出来也报错。
- **类别分数不再套一次 sigmoid。** 导出图里 `4:4+nc` 已经是 [0,1] 的概率，
  再套一次 sigmoid 会把分数全压到 0.5 附近，conf 0.25 的阈值形同虚设、检出数暴涨。
  只有掩码那条链路（`sigmoid(coeff @ proto)`）需要 sigmoid，因为 proto 是 logit。
  判断依据不是「看起来对不对」，是和 ultralytics 对拍的 `mean_abs_score_diff`：三档全是 **0.00000**。
- `batched_nms`：按类别分别做 NMS（`iou_thr=0.7`），不是全局做一次 —— 不同缺陷类别互相压制是错的。
- 与 ultralytics 还有一处**有意的**口径差：`max_det` 截断前先按分数全局降序
  （ultralytics 是逐类 NMS 的拼接顺序再截断）。同一批图上两者选出的集合可能差 1~2 个低分框，
  体现在 match_rate 上，不影响 mAP 量级。本轮 30 张实测没有触发这个差异（match_rate 1.0000）。
- `scale_boxes_to_orig`：先减 letterbox 的 pad，再除 ratio，再裁到原图边界。
  **pad 用的是 `lb.dh`（浮点半 pad）而不是 `lb.top`（取整后的实际边）**，
  余量是奇数时两者差 1 px，而 ultralytics 的 `scale_boxes` 减的是前者。这条在 W3 就踩过（[deploy.md](deploy.md) §6）。
- 阈值全部对齐 `model.predict(imgsz=640, conf=0.25, iou=0.7, max_det=300, verbose=False)`。
  对拍时两边不能有额外变量，否则量到的差不知道是谁的。

### 4.2 掩码解码那个 bug（这一轮最值钱的一段）

**做法上的刻意差异**：ultralytics 的 `ops.process_mask` 是「先把整张 160×160 proto upsample 到原图，再按框裁剪」；
本实现是「先按框在 proto 网格上裁剪，只放大小那一小块」。省掉的是 (640/160)² = **16 倍**的插值面积，
矩阵乘也只在裁剪区内做（`proto[:, y1:y2, x1:x2]`）。在 8GB 显存 + 笔记本 CPU 上这个差别是实打实的。

**代价是「放大到多大、贴到哪」必须自己算对**，而第一版算错了。

第一版：裁剪区是框外扩到整格（`round`）后的范围，通常比框大半格到一格；
却把它**硬拉伸回框的像素宽高**，贴到框的左上角。等于给掩码做了一次系统性压缩。

用「整块填满的 proto」（`sigmoid(32×6) ≈ 1`，掩码就该正好等于裁剪区）做面积口径的对拍，
第一版 vs 参照实现的差距随框变小而急剧放大 —— 而 MVTec 的缺陷本来就小（面积占比 p50 只有 3.5%）：

| 框尺寸(letterbox px，起点带半格偏移) | 参照实现 px | 第一版 px | 第一版 IoU | 现版 px | 现版 IoU |
|---|---|---|---|---|---|
| 9x9 | 144 | 81 | 0.5625 | 144 | **1.0000** |
| 19x15 | 320 | 285 | 0.7139 | 320 | **1.0000** |
| 39x31 | 1280 | 1209 | 0.8451 | 1280 | **1.0000** |
| 79x63 | 5120 | 4977 | 0.9192 | 5120 | **1.0000** |
| 159x127 | 20480 | 20193 | 0.9587 | 20480 | **1.0000** |
| 319x301 | 97280 | 96019 | 0.9809 | 97280 | **1.0000** |

修法两条：

1. **贴回尺寸按裁剪区自己的网格尺寸算**：`tw = round((cx2-cx1) × scale / ratio)`，
   `scale` = 一格等于多少 letterbox 像素（640/160 = 4），`ratio` 是 letterbox 的缩放比。
   贴的位置是「网格左上角 `(cx1, cy1)` 映射回原图」，不是框的左上角。
   这样和 ultralytics 的「整图放大再裁剪」在采样网格上等价，只剩 bilinear 的亚像素差。
2. **裁剪下标用 `floor(v + 0.5)`**，对齐 torch 的 `boxes.round().int()`（四舍五入、远离 0）。
   numpy 的 `np.round` 是**四舍六入五成双**（银行家舍入），`round(0.5) = 0`、`round(1.5) = 2`，
   和 torch 差半格。用 numpy 的 round 会有约一半的框偏一格，小框上就是 20%+ 的面积差。

修完之后，与参照实现在随机 proto + 随机框上的对比（`tests/test_pipeline.py::test_matches_full_upsample_reference`）：

| 框 | 参照 px | 第一版 IoU / 像素比 | 现版 IoU / 像素比 |
|---|---|---|---|
| 0 | 35398 | 0.9993 / 0.9997 | **0.9993** / 0.9997 |
| 1 | 51124 | 0.9997 / 1.0002 | **0.9997** / 1.0002 |
| 2 | 6248 | 0.9334 / 1.0080 | **0.9989** / 0.9989 |

**这个 bug 是怎么被抓住的**：不是靠肉眼看可视化，是靠对拍报告里两个数一起看 ——
掩码 IoU **0.847**（卡在 0.85 门槛下，判据 fail），以及 `px_ours / px_theirs = 0.98`。
第二个数才是关键：如果差异是边界噪声，像素总量应该基本相等（一边多几个像素一边少几个）；
**系统性偏小 2% 只可能是口径错了**，不是精度问题。报告里因此固定加一行像素比，
并注明「偏离 1 说明是系统性口径差，不是边界噪声」。

另外补一个 `mask_iou(..., erode_px=1)`：把两个掩码各腐蚀 1 px 再算 IoU。
本实现和 ultralytics 在边界 1~2 px 上有**设计内**差异（裁剪后放大 vs 整图放大后裁剪），
而 MVTec 的缺陷掩码本身就几十个像素，1 px 的边差就能把 IoU 拉到 0.85 ——
这时候原始 IoU 分不清「解码错了」和「边界采样口径不同」。两个数都报，
判据用哪个写在报告里，不挑好看的那个。

### 4.3 剩下的 0.09 是什么

真实数据上掩码 IoU 是 **0.910~0.916**，不是合成数据上的 0.999。差在哪：
合成数据比的是本实现 vs **自己的参照实现**（同一套采样网格，只有取整差），
真实数据比的是本实现 vs **ultralytics 的 `process_mask`**（整图 bilinear upsample 后再裁剪，
边界像素的插值权重和「先裁剪后放大」不一样）。腐蚀 1 px 后 IoU 从 0.910 掉到 0.899（掉得很少），
说明差异集中在边界；像素比 1.0002~1.0043（≈1），说明没有系统性口径差了。
门槛定在 **0.85**，判据是 `mean_mask_iou`，写死在报告里。

---

## 5. 压测口径：两组，别混着读

同一份 30 张清单（`configs/bench_sample.json`，母体 64/435，sha256 `eccfd964426a2637`）重复 10 轮 = **300 帧**，
三档 × {plain, pinned} × {serial, pipeline} = 12 组，每组都逐帧校验 + 与串行对拍。

| 组 | `--preload` | 量的是什么 | 用途 |
| --- | --- | --- | --- |
| A 含磁盘 | 关 | 读图 + letterbox + 拷贝 + kernel + 后处理，全流程 | **对外报数用这个**：产线上图就是从磁盘/相机来的 |
| B preload | 开 | 图预先读好、letterbox 预先算好，只量拷贝 + kernel + 后处理 | 隔离磁盘，看流水线本身的上限在哪 |

**加速比的分母必须是同一轮同机同批的 `run_serial`**，不能拿 [deploy.md](deploy.md) §8.2 的单帧延迟去除。
理由：§8.2 那组数字输入常驻显存、不含 H2D/D2H、不含前后处理，量的是 kernel；
流水线的分子里含着 16~19 ms 的前处理。拿两个不同口径的数相除，得到的倍数没有任何物理意义 ——
而且会**偏大**，看着很好看，一问就露。

笔记本 GPU 还有时钟与温度的问题（W3b 已经记过：FP16 mean 第一轮 4.81 ms、第二轮 6.73 ms，+40%）。
所以 12 组全在一个进程里连着跑完，A 组 92.2 s、B 组 27.0 s。跨轮比数字一律不认。

---

## 6. 结果

环境：Python 3.12.5 / torch 2.13.0+cu130 / tensorrt 10.16.1.11 / RTX 4060 Laptop 8GB / 16 逻辑核，
git `625e279` + 本轮改动。队列容量 4，`pool_size` 2，imgsz 640，掩码解码开。

### 6.1 A 组：含磁盘（真实口径，对外报数用这组）

| 档 | pinned 串行 FPS | pinned 流水线 FPS | 加速比 | 瓶颈阶段 | decode util | infer util |
|---|---|---|---|---|---|---|
| fp32 | 36.86 | **57.53** | **1.561x** | decode | 1.00 | 0.43 |
| fp16 | 34.55 | **55.82** | **1.616x** | decode | 1.00 | 0.35 |
| int8 | 33.12 | **51.77** | **1.563x** | decode | 1.00 | 0.46 |

四种组合的完整明细：

| 档 | 缓冲 | 模式 | 墙钟 s | FPS | 加速比 | decode ms/util | infer ms/util | post ms/util |
|---|---|---|---|---|---|---|---|---|
| fp32 | pinned | serial | 8.14 | 36.86 | — | 16.20 / 0.60 | 9.64 / 0.36 | 1.28 / 0.05 |
| fp32 | pinned | pipeline | 5.21 | 57.53 | 1.561x | 17.32 / 1.00 | 7.52 / 0.43 | 1.41 / 0.08 |
| fp32 | plain | serial | 7.51 | 39.93 | — | 15.77 / 0.63 | 7.84 / 0.31 | 1.42 / 0.06 |
| fp32 | plain | pipeline | 5.21 | 57.53 | 1.441x | 17.30 / 1.00 | 7.98 / 0.46 | 1.51 / 0.09 |
| fp16 | pinned | serial | 8.68 | 34.55 | — | 16.52 / 0.57 | 11.21 / 0.39 | 1.21 / 0.04 |
| fp16 | pinned | pipeline | 5.37 | 55.82 | 1.616x | 17.85 / 1.00 | 6.34 / 0.35 | 1.39 / 0.08 |
| fp16 | plain | serial | 8.67 | 34.59 | — | 16.18 / 0.56 | 11.47 / 0.40 | 1.25 / 0.04 |
| fp16 | plain | pipeline | 5.28 | 56.80 | 1.642x | 17.53 / 1.00 | 6.89 / 0.39 | 1.43 / 0.08 |
| int8 | pinned | serial | 9.06 | 33.12 | — | 17.77 / 0.59 | 10.92 / 0.36 | 1.49 / 0.05 |
| int8 | pinned | pipeline | 5.79 | 51.77 | 1.563x | 19.25 / 1.00 | 8.96 / 0.46 | 1.63 / 0.08 |
| int8 | plain | serial | 9.31 | 32.21 | — | 17.58 / 0.57 | 11.97 / 0.39 | 1.48 / 0.05 |
| int8 | plain | pipeline | 5.46 | 54.93 | 1.705x | 18.12 / 1.00 | 5.99 / 0.33 | 1.57 / 0.09 |

读数三条：

- **加速比 1.44~1.71×**，pinned 与 plain 在这组里基本打平（fp32 流水线两者都是 57.53 FPS）。
  原因很直接：这一组的瓶颈是 decode（读图 + letterbox 16~19 ms），
  拷贝那点开销（4.7 MiB H2D）藏在它后面，pinned 省下的时间被磁盘吃掉了。
  **pinned 的价值要在 B 组才看得见**，这不是「pinned 没用」，是「这一组里它不是瓶颈」。
- 瓶颈阶段三档都是 **decode**，util 1.00；infer 只有 0.33~0.46 —— GPU 三分之二时间在等 CPU 喂图。
  这正是流水线要解决的问题，也是它只能解决一部分的问题：把 GPU 那段完全藏进 decode 之后，
  吞吐上限就是 decode 本身，1.7× 已经接近这个上限（16 ms decode vs 串行 27 ms 全链路）。
- 想再往上只有一条路：**把 decode 挪到 GPU**（`cudaWarpAffine` 一次做完缩放 + pad），
  因为 A 组的吞吐上限就是这 16~19 ms 的 CPU 前处理，GPU 已经能完全藏进去了。
  这就是 W4② 的立项理由 —— 注意理由**不是**「Python 有 GIL 所以慢」，§7 的实测不支持这个说法。

### 6.2 B 组：preload（隔离磁盘，看流水线上限）

| 档 | pinned 串行 FPS | pinned 流水线 FPS | 加速比 | 瓶颈阶段 | decode util | infer util |
|---|---|---|---|---|---|---|
| fp32 | 126.36 | **158.95** | **1.258x** | infer | 0.99 | 1.00 |
| fp16 | 215.46 | **305.52** | **1.418x** | decode | 0.99 | 0.98 |
| int8 | 233.61 | **330.05** | **1.413x** | decode | 0.99 | 0.98 |

| 档 | 缓冲 | 模式 | 墙钟 s | FPS | 加速比 | decode ms/util | infer ms/util | post ms/util |
|---|---|---|---|---|---|---|---|---|
| fp32 | pinned | serial | 2.37 | 126.36 | — | 0.22 / 0.03 | 6.29 / 0.79 | 1.40 / 0.18 |
| fp32 | pinned | pipeline | 1.89 | 158.95 | 1.258x | 6.23 / 0.99 | 6.27 / 1.00 | 1.63 / 0.26 |
| fp32 | plain | serial | 2.46 | 121.81 | — | 0.00 / 0.00 | 6.94 / 0.85 | 1.26 / 0.15 |
| fp32 | plain | pipeline | 2.08 | 144.46 | 1.186x | 0.01 / 0.00 | 6.89 / 1.00 | 1.84 / 0.27 |
| fp16 | pinned | serial | 1.39 | 215.46 | — | 0.20 / 0.04 | 3.23 / 0.70 | 1.20 / 0.26 |
| fp16 | pinned | pipeline | 0.98 | 305.52 | 1.418x | 3.23 / 0.99 | 3.22 / 0.98 | 1.53 / 0.47 |
| fp16 | plain | serial | 1.62 | 185.34 | — | 0.00 / 0.00 | 4.13 / 0.77 | 1.26 / 0.23 |
| fp16 | plain | pipeline | 1.16 | 257.85 | 1.391x | 0.00 / 0.00 | 3.85 / 0.99 | 1.74 / 0.45 |
| int8 | pinned | serial | 1.28 | 233.61 | — | 0.23 / 0.05 | 2.90 / 0.68 | 1.15 / 0.27 |
| int8 | pinned | pipeline | 0.91 | 330.05 | 1.413x | 2.99 / 0.99 | 2.97 / 0.98 | 1.42 / 0.47 |
| int8 | plain | serial | 1.34 | 223.36 | — | 0.00 / 0.00 | 3.30 / 0.74 | 1.17 / 0.26 |
| int8 | plain | pipeline | 1.14 | 263.20 | 1.178x | 0.01 / 0.00 | 3.77 / 0.99 | 1.84 / 0.48 |

- **INT8 流水线 330.05 FPS**（串行 233.61），FP16 305.52（串行 215.46），FP32 158.95（串行 126.36）。
  这是本仓库目前最高的端到端吞吐数。
- 加速比降到 **1.26~1.42×**，比 A 组低。合理：磁盘那 16 ms 一旦被拿掉，
  能被重叠掉的比例就小了，剩下的主要是 H2D/D2H 与 kernel 的重叠。
- 表里「瓶颈阶段」那一列先别当真：fp16/int8 报的是 decode，§7 有一整节用 `pool_size` 消融
  证明那是**度量假象**（等令牌的时间被算进了 decode），真瓶颈是 GPU infer。
- **pinned 在这组里明显赢 plain**：int8 流水线 330.05 vs 263.20（+25%），fp32 158.95 vs 144.46（+10%）。
  plain+preload 的 decode 阶段几乎不做事（直接返回缓存的数组，0.00 ms），
  但 infer 阶段每帧新建可分页缓冲、同步拷贝，4.13 ms vs pinned 的 3.23 ms。
  **省下来的就是拷贝**，这才是 pinned memory 的账。

### 6.3 对拍 ultralytics

每档跑完流水线，另起一轮 `model.predict` 对同样 30 张图，逐图比：

| 档 | 检出 ours/theirs | 配上 | match_rate | 框 IoU | 掩码 IoU | 腐蚀1px | 像素比 | verdict |
|---|---|---|---|---|---|---|---|---|
| fp32 | 13 / 13 | 13 | 1.0000 | 1.0000 | 0.9100 | 0.8991 | 1.0043 | pass |
| fp16 | 13 / 13 | 13 | 1.0000 | 1.0000 | 0.9104 | 0.8995 | 1.0038 | pass |
| int8 | 17 / 17 | 17 | 1.0000 | 1.0000 | 0.9158 | 0.9086 | 1.0002 | pass |

`mean_abs_score_diff` 三档全是 **0.00000**（§4.1 那条 sigmoid 的决定就是这么验的），
未全配的图 **0 张**，`match_rate_min` 也是 1.0000 —— 不是「平均下来还行」，是**每一张都全配上了**。

### 6.4 判据

每档 10 条，全过：

| 档 | 通过判据数 | 失败 |
|---|---|---|
| fp32 | 10 / 10 | 无 |
| fp16 | 10 / 10 | 无 |
| int8 | 10 / 10 | 无 |

10 条 = 保序 ×2（plain/pinned）+ 结果条数 ×2 + 与串行逐帧一致 ×2 + 加速比 ≥1.0 +
对拍 match_rate ≥0.98 + 框 IoU ≥0.99 + 掩码 IoU ≥0.85。

「与串行逐帧一致」比的是 `(n_det, mask_pixels, score_sum)` 三元组，300 帧全等（`n_diff: 0`）。
这条钉的就是 §3.5 说的跨帧数据竞争：**如果令牌池串了槽，这里会偶发不一致**，
而且是负载相关的偶发 —— 跑一次过了不代表对。

### 6.5 500 帧端到端延迟与 CPU/GPU 占用

W4 计划里还差两个数：**500 帧的逐帧端到端 P50/P95** 和 **压测期间的 CPU/GPU 占用**。
补上这两个，"流水线值不值"才算说完整 —— 只报 FPS 会掩盖延迟恶化，只报延迟又让人以为流水线没用。

环境同 §6（Python 3.12.5 / torch 2.13.0+cu130 / tensorrt 10.16.1.11 / RTX 4060 Laptop 8GB / 16 逻辑核），
git `c645e19` + 本轮改动，原始报告 `runs/logs/pipeline500_*.json`（不入库）。

#### 6.5.1 口径先说清楚：这一节是 500 帧，和 §6.1 的 300 帧不能横向比绝对值

同一份清单（`configs/bench_sample.json`，母体 64/435，sha256 `eccfd964426a2637`）取
**前 50 张 × 10 轮 = 500 帧**，其余参数与 §6.1 完全一致（capacity 4、pool_size 2、imgsz 640、掩码解码开）。

图换了（30 张 → 50 张），decode 耗时跟着变，所以绝对 FPS 比 §6.1 高一截
（fp32 pinned 串行 40.70 vs §6.1 的 36.86）。**加速比反而更稳**：本轮四跑 12 格落在
**1.44~1.67×**，§6.1 是 1.44~1.71×。老规矩：跨轮跨清单只比同轮内部的串行/流水线之比，绝对值不外推。

#### 6.5.2 端到端延迟：流水线的 P50 是串行的 4 倍，这是正常的

两个口径别混：**阶段耗时**（§6.1 表里那三列）是"这一帧在某段里被处理了多久"，加起来是干活时间；
**端到端延迟**是"这一帧从进队到结果出来经过了多久"，**含在队列里等的时间**。

A 组（含磁盘，500 帧，采样关，`runs/logs/pipeline500_nosample2.json`）：

| 档 | 缓冲 | 串行 P50 | 串行 P95 | 串行 max | 流水线 P50 | 流水线 P95 | 流水线 max | P50 放大 |
|---|---|---|---|---|---|---|---|---|
| fp32 | plain | 24.513 | 28.919 | 78.694 | 102.675 | 126.342 | 135.140 | 4.19x |
| fp32 | pinned | 24.980 | 29.848 | 34.388 | 100.121 | 120.379 | 131.493 | 4.01x |
| fp16 | plain | 23.540 | 29.754 | 33.325 | 99.820 | 119.248 | 125.177 | 4.24x |
| fp16 | pinned | 24.329 | 31.201 | 34.947 | 98.287 | 117.942 | 126.634 | 4.04x |
| int8 | plain | 25.292 | 30.506 | 37.151 | 105.452 | 132.479 | 146.781 | 4.17x |
| int8 | pinned | 26.850 | 33.322 | 38.653 | 107.404 | 131.391 | 141.955 | 4.00x |

（单位 ms。同轮 FPS：串行 36.47~42.67，流水线 59.56~65.91。）

**读法**：串行 P50 24~27 ms，流水线 P50 98~107 ms，放大 **4.00~4.24×**；
同一台机器同一批图，吞吐从 ~40 FPS 涨到 ~65 FPS（**+60%**）。
这就是"用单帧延迟换吞吐"的账面 —— 帧在队列里排着，所以每帧的**体感**变慢，
但单位时间**处理完的帧数**变多。要不要这笔交易看场景：离线批量质检要吞吐，
在线联动产线要延迟（那时候该把 `capacity` 调小，甚至退回串行）。**报数时两个都得给。**

**Little's law 交叉验证**（L = λ·W，在飞帧数 = 吞吐 × 端到端延迟）：

| 组合 | 串行 L | 流水线 L | 缓冲上限 |
|---|---|---|---|
| fp32 pinned | **1.00** | 6.49 | 15 |
| fp16 pinned | **1.00** | 6.42 | 15 |
| int8 pinned | **1.00** | 6.41 | 15 |

串行那一列全是 **1.00** —— 不是约等于，是三位有效数字内的 1.00。串行本来就只有一帧在飞，
这个数等于 1 说明 **e2e 的时间戳口径是自洽的**：时间戳要是打错了地方（打在 `put` 之后、
或者漏掉了排队那一段），L 就会偏掉。这条比任何单元测试都硬，因为它是拿两组独立测出来的数
（FPS 和 e2e 均值）互推的。

流水线 L ≈ 6.4 帧在飞，缓冲上限是 `capacity(4) × 3 段 + 3 个正在处理的 = 15`，只用了 **43%**。
队列没顶满，因为 decode 是瓶颈（util 1.00），后两段一直在等它，q1/q2 大部分时间是空的。
**这顺手解释了为什么把 capacity 从 4 调到 8 收益不大**：瓶颈在源头，加缓冲只是让帧多等一会儿
（e2e 变大，FPS 不变）。

B 组（preload，500 帧，`runs/logs/pipeline500_preload_sampled.json`）：

| 档 | 缓冲 | 串行 FPS | 串行 P50 | 串行 P95 | 流水线 FPS | 流水线 P50 | 流水线 P95 | P50 放大 |
|---|---|---|---|---|---|---|---|---|
| fp32 | plain | 123.33 | 7.866 | 9.030 | 140.60 | 79.698 | 84.857 | 10.13x |
| fp32 | pinned | 127.31 | 7.696 | 8.826 | 157.31 | 50.488 | 52.114 | 6.56x |
| fp16 | plain | 200.07 | 4.795 | 6.497 | 249.16 | 45.402 | 50.822 | 9.47x |
| fp16 | pinned | 215.19 | 4.462 | 5.726 | 299.19 | 26.035 | 29.732 | 5.83x |
| int8 | plain | 207.87 | 4.605 | 6.217 | 275.93 | 41.031 | 46.518 | 8.91x |
| int8 | pinned | 227.52 | 4.252 | 5.344 | **316.66** | 24.610 | 27.594 | 5.79x |

放大倍数跳到 **5.8~10.1×**，比 A 组大。原因同上但更极端：单帧只要 4~8 ms，
而 λ 涨到 250~317 FPS，L = λ·W 里的 W 自然被推高。
注意 **pinned 的放大倍数（5.8~6.6×）明显小于 plain（8.9~10.1×）** ——
pinned 流水线吞吐更高、单帧更快穿过队列，排队更短。这是 §6.2 那笔拷贝账在**延迟维度**上的另一面：
pinned 不只是快，它还把延迟的尾巴压得更平（int8 P95 27.6 ms vs plain 46.5 ms）。

#### 6.5.3 CPU/GPU 占用

采样口径写在 `induvision/prof.py`：CPU 用 `time.process_time()`（**进程级**，所有线程之和）
两次读数之差除以墙钟 = **核当量**，可以 > 1；GPU 走 `nvidia-smi --query-gpu` 子进程，200 ms 一次。
不引 psutil —— 核心库的依赖边界是 numpy/opencv/pyyaml/标准库，为一个采样线程破例不值。

跨 12 格聚合（A 组取 `pipeline500_sampled2.json`，B 组取 `pipeline500_preload_sampled.json`）：

| 组 | 模式 | CPU 核当量 均值 | CPU 峰值(最大) | GPU util 均值 | GPU util 峰值(最大) | 整机显存 MiB |
|---|---|---|---|---|---|---|
| A 含磁盘 | 串行 | 1.105（1.064~1.133） | 1.93 | 31.5%（29.4~33.6） | 46% | 2439 / 2523 |
| A 含磁盘 | 流水线 | **1.654**（1.601~1.724） | 2.81 | 33.6%（28.3~37.0） | 60% | 2439 / 2523 |
| B preload | 串行 | 1.079（1.009~1.212） | 1.81 | 60.1%（42.7~75.2） | 94% | 2439 / 2523 |
| B preload | 流水线 | **1.600**（1.366~1.887） | 2.46 | 70.8%（42.0~92.9） | 95% | 2439 / 2523 |

逐格明细（24 行）：

| 组 | 档 | 缓冲 | 模式 | FPS | CPU 均值 | CPU 峰值 | GPU util 均值 | GPU util 峰值 | 整机显存 MiB | 采样点 |
|---|---|---|---|---|---|---|---|---|---|---|
| A | fp32 | plain | serial | 41.22 | 1.128 | 1.730 | 29.4% | 46% | 2523 | 48 |
| A | fp32 | plain | pipeline | 59.26 | 1.724 | 2.162 | 36.1% | 51% | 2523 | 32 |
| A | fp32 | pinned | serial | 40.66 | 1.082 | 1.931 | 32.5% | 45% | 2523 | 49 |
| A | fp32 | pinned | pipeline | 63.94 | 1.679 | 2.771 | 37.0% | 48% | 2523 | 31 |
| A | fp16 | plain | serial | 40.24 | 1.117 | 1.754 | 29.8% | 44% | 2439 | 49 |
| A | fp16 | plain | pipeline | 63.88 | 1.655 | 2.205 | 32.5% | 44% | 2439 | 30 |
| A | fp16 | pinned | serial | 38.71 | 1.064 | 1.434 | 33.6% | 46% | 2439 | 51 |
| A | fp16 | pinned | pipeline | 64.68 | 1.643 | 2.412 | 35.4% | 54% | 2439 | 30 |
| A | int8 | plain | serial | 38.55 | 1.104 | 1.640 | 32.6% | 44% | 2439 | 51 |
| A | int8 | plain | pipeline | 64.70 | 1.601 | 2.165 | 32.5% | 60% | 2439 | 30 |
| A | int8 | pinned | serial | 39.89 | 1.133 | 1.737 | 31.1% | 42% | 2439 | 50 |
| A | int8 | pinned | pipeline | 65.09 | 1.621 | 2.808 | 28.3% | 53% | 2439 | 30 |
| B | fp32 | plain | serial | 123.33 | 1.029 | 1.303 | 64.8% | 76% | 2523 | 16 |
| B | fp32 | plain | pipeline | 140.60 | 1.484 | 2.094 | 81.7% | 85% | 2523 | 14 |
| B | fp32 | pinned | serial | 127.31 | 1.018 | 1.085 | 75.2% | 84% | 2523 | 16 |
| B | fp32 | pinned | pipeline | 157.31 | 1.366 | 1.919 | 92.9% | 95% | 2523 | 13 |
| B | fp16 | plain | serial | 200.07 | 1.212 | 1.812 | 63.5% | 94% | 2439 | 10 |
| B | fp16 | plain | pipeline | 249.16 | 1.439 | 2.070 | 68.0% | 73% | 2439 | 8 |
| B | fp16 | pinned | serial | 215.19 | 1.069 | 1.333 | 62.1% | 73% | 2439 | 9 |
| B | fp16 | pinned | pipeline | 299.19 | 1.887 | 2.353 | 76.7% | 87% | 2439 | 7 |
| B | int8 | plain | serial | 207.87 | 1.140 | 1.805 | 42.7% | 87% | 2439 | 10 |
| B | int8 | plain | pipeline | 275.93 | 1.671 | 2.461 | 42.0% | 65% | 2439 | 7 |
| B | int8 | pinned | serial | 227.52 | 1.009 | 1.306 | 52.0% | 62% | 2439 | 9 |
| B | int8 | pinned | pipeline | 316.66 | 1.750 | 2.395 | 63.3% | 72% | 2439 | 6 |

三条读数：

- **CPU：串行 ~1.10 核，流水线 ~1.65 核**，峰值 2.81 核。三条工作线程只吃到 1.65 核而不是 3 核 ——
  GIL 是一部分原因，更主要的是**大部分时间在等**（等磁盘、等 GPU、等令牌），§7 已经用
  `pool_size` 消融证过这件事。报"1.65 核当量"比报"CPU 占用 165%"有用：
  后者随机器核数失真（16 核机器上的 165% 和 4 核机器上的 165% 完全是两件事），前者不会。
- **GPU util：A 组只有 28~37%，B 组 42~93%**。A 组低是因为 CPU 前处理喂不满 GPU，
  这和 §6.1 的 infer util 0.33~0.46 完全对得上 —— **两个独立口径（TensorRT 侧的阶段 util
  和 nvidia-smi 侧的整卡 util）互相印证**，不是巧合。B 组把磁盘拿掉后 fp32 pinned 流水线冲到
  **92.9% 均值 / 95% 峰值**，这才是这块卡在这套模型上的真实上限。
- **显存那两列（2439 / 2523 MiB）是 nvidia-smi 的整机读数，含桌面、浏览器和其他进程**，
  不是模型占用。模型侧的数是 §3.5 的 **18.893 MiB**（pinned host + device 各一份）。
  两个数差 130 倍，混着报会闹笑话，所以表头写死"整机显存"。
  fp32 档 2523、fp16/int8 档 2439，差 84 MiB ≈ engine 体积差（65.25 vs 24.58/12.55 MiB）
  加上 TensorRT 工作区，量级对得上。

#### 6.5.4 采样器扰动实测：测不出来（这一段是本轮最值钱的）

`--sample-resources` 每 200 ms 起一个 `nvidia-smi` 子进程，这玩意儿自己要吃 CPU。
按直觉它会把被测对象拖慢，所以**默认关**。但"直觉上会拖慢"不能写进文档，
于是做了一组 A/B：**同一份 500 帧配置跑四轮**，只有采样开关和是否带对拍在变
（`pipeline500_nosample.json` / `nosample2` / `sampled` / `sampled2`）。

| 对比 | FPS 相对差 均值 | 最小 | 最大 | 绝对值均值 |
|---|---|---|---|---|
| 无采样(RUN4) → 开采样(RUN2) | −1.38% | −12.29% | +9.20% | 4.92% |
| 无采样(RUN4) → 开采样(RUN5) | −0.31% | −7.00% | +9.38% | 4.28% |
| 开采样(RUN2) → 开采样(RUN5)　**← 噪声底** | +1.26% | −7.02% | +11.84% | **3.57%** |
| 无采样(RUN1) → 无采样(RUN4)　**← 噪声底（含对拍差）** | +4.77% | −9.92% | +22.94% | **8.57%** |

每格 12 个（3 档 × 2 缓冲 × 2 模式），相对差 = (后一轮 FPS − 前一轮 FPS) / 前一轮 FPS。

**结论：采样器的扰动被机器噪声完全淹没了。**
同配置两轮之间的 |差| 均值本来就有 **3.57%**，开了采样只多 0.7~1.3 个百分点，
12 格里有 5 格开了采样反而更快 —— 这在统计上就是"测不出来"。

中途我一度以为测出来了：RUN2 的 fp32 pinned_pipeline 只有 **57.17 FPS**，
比 RUN4 的 65.18 低 **12.3%**，看着就是采样器干的。但同一格 RUN5 又跑回 **63.94 FPS**。
单格 −12% 是噪声，不是效应 —— **这正是为什么必须跑两轮以上才敢下结论**：
一轮 A/B 的差值里，噪声和效应是分不开的，而笔记本 GPU 的噪声比台式机大得多
（W3b 就记过 FP16 单帧延迟两轮差 +40%，时钟和温度都在飘）。

所以"采样默认关"的真正理由**不是**"它扰动大"，而是这两条：

1. **换不来更准的数**：既然扰动 < 噪声，开着它也提不了精度；
2. **多一个失败面**：nvidia-smi 不在 PATH、驱动不认这几个字段、子进程超时、返回 `[N/A]`，
   每一条都得兜（`tests/test_prof.py` 里 17 条 smi 失败路径的单测就是为此写的）。
   为一个测不出差别的数字增加失败面，不值。

要报资源占用就**单独跑一轮开采样的**。本节的 FPS/延迟表全部取自 RUN4（无采样），
资源表全部取自 RUN5（开采样），**两组数字不同源，这是刻意的**，别拿它们互相除。

顺带量到一个更狠的噪声源：**带不带 `--check-ultralytics` 差 8.57%**（RUN1 vs RUN4）。
对拍那一步会另起一轮 `model.predict`，torch 的显存分配和 GPU 时钟状态都变了，
后面几档的 FPS 跟着飘（fp16 串行 34.70 → 42.67，**+22.9%**）。
**要报吞吐就关掉对拍单独跑，要验精度就单独跑对拍**，别指望一轮里两个数都干净。

---

## 7. 读数陷阱：util ≈ 0.99 不等于算力打满

B 组有个数看着很不对劲：流水线三个阶段 util 全是 **0.99**，decode 的 mean 从串行的
0.22 ms 变成 6.23 ms，虚高 **28 倍**。第一反应是「GIL 卡住了 / 内存带宽打满了」。

| 档 | 缓冲 | decode 串行 ms | decode 流水线 ms | 虚高倍数 | infer 串行 ms | infer 流水线 ms |
|---|---|---|---|---|---|---|
| fp32 | pinned | 0.22 | 6.23 | 28.3x | 6.29 | 6.27 |
| fp32 | plain | 0.00 | 0.01 | 1.7x | 6.94 | 6.89 |
| fp16 | pinned | 0.20 | 3.23 | 15.8x | 3.23 | 3.22 |
| fp16 | plain | 0.00 | 0.00 | 2.0x | 4.13 | 3.85 |
| int8 | pinned | 0.23 | 2.99 | 13.1x | 2.90 | 2.97 |
| int8 | plain | 0.00 | 0.01 | 2.5x | 3.30 | 3.77 |

（A 组没有这个现象，虚高只有 1.1×，因为 decode 本来就 16 ms，多出来的等待占比很小。）

**真因不是 GIL，是度量口径**：stage 的计时包住了整个 stage 函数，
而 pinned 版的 decode 函数**第一件事就是 `client.acquire()`** —— 拿不到令牌就阻塞。
`pool_size=2` 时最多 2 帧在飞，decode 比 infer 快得多，于是它每帧都在等令牌，
等待时间被算进了「decode 忙」。同一轮的 `pool_waits = 298`（300 帧里 298 帧等过）就是这个意思。

验证方法很便宜：只改 `--pool-size`，别的都不动，重跑 B 组。

| pool_size | 档 | 流水线 FPS | 加速比 | decode ms | infer ms | post ms | acquire 等待次数 | 瓶颈(按 busy) | q1 underflow |
|---|---|---|---|---|---|---|---|---|---|
| 2 | fp32 | 158.95 | 1.258x | 6.23 | 6.27 | 1.63 | 298 | infer | 0 |
| 2 | fp16 | 305.52 | 1.418x | 3.23 | 3.22 | 1.53 | 298 | decode | 13 |
| 2 | int8 | 330.05 | 1.413x | 2.99 | 2.97 | 1.42 | 298 | decode | 19 |
| 3 | fp32 | 159.36 | 1.268x | 6.19 | 6.25 | 1.50 | 297 | infer | 0 |
| 3 | fp16 | 302.48 | 1.382x | 3.25 | 3.28 | 1.44 | 297 | infer | 0 |
| 3 | int8 | 330.17 | 1.507x | 2.98 | 3.01 | 1.46 | 297 | infer | 1 |
| 4 | fp32 | 159.48 | 1.265x | 6.17 | 6.25 | 1.50 | 296 | infer | 0 |
| 4 | fp16 | 305.79 | 1.407x | 3.21 | 3.25 | 1.39 | 295 | infer | 0 |
| 4 | int8 | 324.01 | 1.444x | 3.03 | 3.06 | 1.67 | 296 | infer | 3 |
| 6 | fp32 | 159.28 | 1.259x | 1.63 | 6.25 | 1.43 | 294 | infer | 0 |
| 6 | fp16 | 303.91 | 1.402x | 1.47 | 3.26 | 1.29 | 293 | infer | 0 |
| 6 | int8 | 331.14 | 1.481x | 1.55 | 2.99 | 1.36 | 294 | infer | 0 |

（`pool_size=2` 那三行就是 §6.2 的 B 组，同一份报告；3/4/6 是本轮补跑的三组，各 10.0 s，verdict 全 pass。）

结论很干净：

1. **`pool_size` 从 2 加到 6，decode 的表观耗时从 6.23/3.23/2.99 ms 掉到 1.63/1.47/1.55 ms**，
   而 infer 纹丝不动（6.25/3.26/2.99）。掉下去的那部分就是等令牌的时间，不是真在干活。
2. **FPS 基本没变**（fp32 158.95→159.28，fp16 305.52→303.91，int8 330.05→331.14，都在 ±1% 内）。
   所以吞吐上限在 **GPU infer**，不在前处理、也不在令牌池。加 pool_size 只是让报表好看，不提速。
3. **瓶颈判定跟着翻了**：`pool_size=2` 时 fp16/int8 报的是 `bottleneck=decode`，
   3/4/6 全部报 `infer`。也就是说 §6.2 表里那个「瓶颈阶段：decode」是**度量假象**。
   `speedup_report` 取的是 busy_s 最大的阶段，而 busy 里含阻塞等待 —— 这个口径要改（§9）。

**能带走的通用教训**：阻塞式队列 + 阻塞式令牌池下，「阶段 util」量的是**线程被占用的时间**，
不是「真在算的时间」。util 高只说明这段线程没闲着，可能全在等。
要区分真忙和等待，必须同时看 `pool_waits`、`overflow_waits`、`underflow_waits`。
单看 util 会把「等令牌」误读成「解码慢」，然后去优化一个根本不是瓶颈的阶段。

顺带说队列那边（B 组 pinned 流水线）：

| 场景 | 档 | 队列 | max_depth | overflow_waits | underflow_waits |
|---|---|---|---|---|---|
| 含磁盘 | fp32 | q0->decode | 4 | 296 | 0 |
| 含磁盘 | fp32 | q1->infer | 1 | 0 | 300 |
| 含磁盘 | fp32 | q2->post | 1 | 0 | 300 |
| 含磁盘 | fp32 | qout | 1 | 0 | 300 |
| 含磁盘 | fp16 | q0->decode | 4 | 296 | 0 |
| 含磁盘 | fp16 | q1->infer | 1 | 0 | 300 |
| 含磁盘 | fp16 | q2->post | 1 | 0 | 300 |
| 含磁盘 | fp16 | qout | 1 | 0 | 300 |
| 含磁盘 | int8 | q0->decode | 4 | 296 | 0 |
| 含磁盘 | int8 | q1->infer | 1 | 0 | 300 |
| 含磁盘 | int8 | q2->post | 1 | 0 | 300 |
| 含磁盘 | int8 | qout | 1 | 0 | 300 |
| 场景 | 档 | 队列 | max_depth | overflow_waits | underflow_waits |
|---|---|---|---|---|---|
| preload | fp32 | q0->decode | 4 | 295 | 0 |
| preload | fp32 | q1->infer | 2 | 0 | 0 |
| preload | fp32 | q2->post | 1 | 0 | 300 |
| preload | fp32 | qout | 1 | 0 | 300 |
| preload | fp16 | q0->decode | 4 | 296 | 0 |
| preload | fp16 | q1->infer | 1 | 0 | 13 |
| preload | fp16 | q2->post | 1 | 0 | 290 |
| preload | fp16 | qout | 1 | 0 | 300 |
| preload | int8 | q0->decode | 4 | 296 | 0 |
| preload | int8 | q1->infer | 1 | 0 | 19 |
| preload | int8 | q2->post | 1 | 0 | 285 |
| preload | int8 | qout | 1 | 0 | 300 |

`q0->decode` 的 `overflow_waits ≈ 296`（feeder 塞得比 decode 取得快，队列常年满，水位 4），
`q1->infer` 在 `pool_size=2` 时有 13/19 次 underflow（infer 偶尔饿着等 decode），
`q2->post` 与 `qout` 各 300 次 underflow（下游一直比上游快，队列常年空）。
A 组（含磁盘）那边 `q1` 的 underflow 是 300/300 —— **每一帧 infer 都在饿**，
这就是「GPU 三分之二时间在等 CPU」的另一种说法，和 util 0.33~0.46 对得上。

---

## 8. 踩坑清单

1. **掩码贴回尺寸按框算 → 系统性压缩 2%**（§4.2）。抓它的是像素比不是 IoU。
   修法：按裁剪区的网格尺寸 × scale / ratio，贴到网格左上角映射回原图的位置。
2. **`np.round` 不是四舍五入**，是银行家舍入，和 torch 的 `round().int()` 差半格。
   一律用 `floor(v + 0.5)`。
3. **类别分数不能重复 sigmoid**（§4.1）。导出图里已经是概率；只有 proto 那条链是 logit。
   验法是 `mean_abs_score_diff` 必须为 0，不是「看着差不多」。
4. **feeder 必须独立线程**，否则主线程 put 阻塞 + collector 停摆 = 死锁（§3.1）。
5. **`non_blocking=True` 对可分页内存静默退化成同步**，不报错、不变慢得离谱，
   只有对比 pinned 才看得出差别（B 组 int8：330 vs 263 FPS）。
6. **acquire 之后出错必须 release**，否则池子越用越小，症状是「越跑越慢」而不是「跑挂」（§3.5）。
7. **`RingBuffer` 的等待计数要按调用记，不按循环轮数记**，否则 `pool_waits` 虚高，
   §7 那套诊断就全废了。
8. **util 不是算力占用率**（§7）。这是本轮最贵的一课，因为它差点把 W4② 的方向带偏
   （「Python 有 GIL，所以前处理慢」→ 实际是「前处理根本不慢，是等令牌的时间被算进了前处理」）。
9. **加速比的分母必须同轮同机同批**（§5）。跨口径相除得到的倍数是编的。
10. **掩码数组不能留在结果里**。900×900 的 bool 一张 0.8 MiB，300 帧 × 每帧几个缺陷 = 几 GB，
    压测跑完机器就卡了。`FrameResult` 只留 `(n_det, mask_pixels, score_sum)` 统计量，
    只有对拍那一轮才带 `dets`。
11. **端到端时间戳必须打在 `put` 之前**（§6.5.2）。打在之后就等于把背压藏起来，
    P95 会好看但没有意义。守它的是 `test_pipeline_e2e_exceeds_serial_because_of_queueing`
    和 Little's law 那条 L=1.00。
12. **一轮 A/B 分不开噪声和效应**（§6.5.4）。单格 −12.3% 看着像采样器拖的，
    再跑一轮同一格回到 −1.9%。笔记本 GPU 的噪声底就有 3.6%，结论至少要两轮才敢下。
13. **`--check-ultralytics` 会污染后面的吞吐数**（§6.5.4，差 8.57%）。验精度和报吞吐分两轮跑。
14. **`float("nan") <= 0` 是 `False`**。采样间隔的校验写成 `if interval_s <= 0` 会放 NaN 过去，
    接着 `Event.wait(nan)` 抛 ValueError 被兜底 `except` 吞掉，采样线程一轮就悄悄退出，
    报告里只剩 `gpu_available=False`，看着像"这台机器没显卡"。判据要写 `if not x > 0`。

---

## 9. W4② C++ 版还剩什么

先说本机缺什么（实测，不是猜）：`nvcc` / `cmake` / `cl` / `ninja` / `msbuild` 全部 not found，
无 CUDA Toolkit、无 VS 安装目录。要装 VS Build Tools（含 C++ 工作负载）+ CUDA Toolkit，数 GB。
**逐样东西怎么装、TensorRT 的头与 `.lib` 从哪来（pip wheel 里 0 个 `.lib`）、装完按什么顺序做、
拿什么判据验收，全部写在 [cpp_infer.md](cpp_infer.md)** —— 本节只留方向。

工具链到位后要做的：

1. **前处理搬上 GPU**：`cudaWarpAffine` 做 letterbox（缩放 + pad 一次搞定），
   替掉 A 组那 16~19 ms 的 CPU 读图 + resize。这是 A 组 1.7× 之上的下一个台阶，
   因为 A 组的瓶颈就是它。
2. **多 CUDA stream**：本轮只有一条流，H2D/kernel/D2H 在流内是串行的，
   重叠只发生在「GPU 忙」与「CPU 忙」之间。多流 + event 才能让第 n 帧的 D2H 和第 n+1 帧的 H2D 真的并行。
3. **绕开 GIL**：C++ 天然没有。但注意本轮的证据（§7）**并不支持**「Python 版是被 GIL 卡住的」这个说法 ——
   别把这条当成 C++ 版的卖点写进报告，卖点是 1 和 2。
4. **修度量口径**：把 `acquire` 等待单独计时（`decode_wait_ms` vs `decode_busy_ms`），
   `bottleneck_stage` 用后者判；`pool_size` 默认提到 3~4（本轮实测对吞吐无害，对报表诚实）。
5. **把 §6.5 那套 500 帧口径搬到 C++ 上重跑**：Python 侧的基线已经有了
   （500 帧 = 50 张 × 10 轮，报 e2e P50/P95/max + CPU 核当量 + GPU util/整机显存），
   C++ 版用**同一份清单、同一套分位口径**再跑一遍，直接和那张表对读。
   还差两个数：**P99**（Python 版只报到 max）和 `decode_wait_ms` / `decode_busy_ms` 的拆分（见第 4 条）。
   注意别拿 C++ 的 500 帧数去比 Python 的 300 帧数 —— 图不一样，decode 不一样（§6.5.1）。
6. **CMake 工程**：engine 反序列化 → CUDA 流 → 前处理 → 推理 → 后处理（NMS + 掩码解码），
   后处理直接照搬本轮的 Python 语义（§4），**并且拿本轮的对拍结果当验收标准**：
   match_rate ≥0.98、框 IoU ≥0.99、掩码 IoU ≥0.85，一条都不能松。

---

## 10. 测试覆盖

| 文件 | 条数 | 覆盖 |
| --- | --- | --- |
| `tests/test_pipeline.py` | **183** | `RingBuffer`：容量边界、put/get 往返、满/空阻塞与超时返回、`close` 幂等与唤醒所有等待者、关闭后 put 抛错而 get 返回 `None`、`max_depth`/`puts`/`gets` 计数、**等待计数按调用记不按循环轮数记（5 条专门钉这个口径）**；后处理：`_as_predictions` 的形状与属性数、`cxcywh_to_xyxy`、`iou_one_vs_many` 的退化情形（零面积、完全重合、不相交）、`nms` 的分数排序与 iou 阈值边界、`batched_nms` 按类隔离（不同类不互相压制）、`scale_boxes_to_orig` 的 pad/ratio 口径与越界裁剪；掩码：`masks_from_proto` 的 3D/4D proto、batch>1 抛错、coeffs 与 boxes 数量不一致抛错、框在网格外返回全 False、退化框返回全 False、阈值越高掩码越小、**与「整图放大再裁剪」参照实现比对（§4.2 的真 bug 就是这么抓出来的，已固化成回归测试）**、**贴回面积按网格几何算（偏差 <2%）**、掩码不越出图像、`mask_iou` 的两个都空=1.0 / 只有一个空=0.0 / 形状不一致抛错 / `erode_px` 负数抛错 / 腐蚀能把两个都清空；`postprocess`：conf/iou/max_det/mask 四个阈值各自的作用、`need_masks=False` 时 mask 为 None、分数不再套 sigmoid（分数值域与排序）、无检出时返回空；对拍：`match_detections` 的全配 / 漏配 / 多配 / 空对空、`Detection.to_dict` 的掩码像素数与 None 分支；流水线：`run_serial` 与 `Pipeline` 结果逐条相等、保序（乱序 stage 也保序）、多阶段与单阶段、空输入、stage 抛错包成 `PipelineError(stage, index, exc)`、错误后所有线程收尾不悬挂、`speedup_report` 的加速比与瓶颈判定、`_stage_report` 的 util 与分位、加速比对照的分母口径；
**端到端延迟 15 条**（`e2e_latency`）：串行 count 等于帧数、八个分位键都在、
**e2e ≥ 各阶段耗时之和**（串行没有队列）、**1000/e2e 均值 ≈ FPS（两套数互为交叉验证，差 <20%）**、
空输入、分位单调；`_report` 不传 e2e / 传 None / 传列表三种情形；
流水线 count 等于帧数、空输入、值为正且有限、
**排队效应（三段均衡 8 ms + capacity 3 → 流水线吞吐 >1.3× 且 e2e P50 >1.5× 串行，
守的是「时间戳打在 put 之前」）**、**capacity=1 单阶段时在飞帧数 ≈2（比值落在 1.2~3.5）**、
**capacity 变大 e2e 随之变大（延迟换吞吐的旋钮）** |
| `tests/test_prof.py` | **87** | `induvision.prof`（W4① 补丁）：`_to_int` 的 `"2246 MiB"`/`"37 %"`/`"[N/A]"`/`"[Not Supported]"`/空/无数字/`"0"` 是 0 不是 None/负号不解析；`parse_smi` 的正常行、字段顺序与 `SMI_FIELDS` 一致、None/空、字段不足、多余字段忽略、**多行只取首行**、跳空行、**全 `[N/A]` 返回「值为空的字典」而不是 None**（命令通了但字段不支持 ≠ 没读到）、部分 `[N/A]`、垃圾、非字符串；`smi_sample` 用**注入的假 runner** 覆盖 17 条路径：成功、cmd 里有 `--id=` 与 `--query-gpu=`、timeout 传到位、`FileNotFoundError`/`OSError`/`TimeoutExpired`/`SubprocessError`/`ValueError`/returncode≠0/返回 None/stdout 是垃圾或 None/`[N/A]` 行、**不注入 runner 时也绝不抛**（CI 上没显卡）、多次调用按脚本顺序返回；`cpu_cores` 的多核、亚单核、四位小数舍入、Δ=0 是 0.0 不是 None、wall ≤0 或 None → None、**读数倒退 → None（不能报负核数）**；`_mean`/`_max` 的空表、过滤 None、舍入、返回 float；`ResourceSampler` 参数校验（interval ≤0、max_samples ≤0、**NaN**、非数字类型、字符串数字自动收敛）+ 用**注入假时钟**做确定性断言（wall 步进 1.0 / proc 步进 0.1 → 每区间 0.1 核，均值=峰值=0.1）、**行数 = max_samples+2**、核当量 >1、GPU 读数聚合、`gpu_index` 透传、smi 返回 None → `gpu_available=False` 而 CPU 那半边照记、**smi 抛异常不掀翻采样**、`[N/A]` 行算采样点但无有效值、start 两次是 no-op、stop 幂等且 join 线程、未 start 就 summary 全空、summary 的 14 个键、配置回显、上下文管理器、**summary() 不顺手停线程**、线程是 daemon 且叫 `res-sampler`、sleeper 收到的间隔数（采满 N 轮只 sleep N−1 次）、没采满就 stop 也不卡住 |
| `tests/test_pinned.py` | **59** | 全部用**假 torch / 假 runner**，不碰 GPU：`require_torch` 与 `missing_gpu_dep_message` 的报错口径（要指出装哪份 requirements）、`PinnedOutputs.get/shapes`、`owned` 语义（自己 acquire 的要调用方 release，传进来的不动）；池：`_build_pool` 的 slot 数与 shape/dtype 解析、`_resolve_out_dtype`、`acquire`/`release` 往返、**令牌不串槽（并发拿两个令牌，写各自 slot 互不覆盖）**、池耗尽时 `acquire` 阻塞与 timeout、**出错路径也还令牌（异常后池子大小不变）**、`pool_waits` 计数、`forward` 每帧**只 `stream.synchronize()` 一次**（假流上数调用次数）、H2D/D2H 全部带 `non_blocking=True`（假张量上断言关键字参数）、`input_view` 返回的是 pinned 缓冲的视图不是副本（`np.copyto` 写进去能被 runner 读到）、`forward_array` 的自建令牌路径、`stats` 的字段、`close` 幂等、上下文管理器 `__enter__/__exit__`、没装 tensorrt 时的报错指向 `docs/environment.md §6` |

W4① 三个文件 **329 条**（pipeline 183 / pinned 59 / prof 87），加上原有的 563 条，
全量 **892 passed / 6 skipped**（898 collected，21.80 s），
base 环境（`W:\Anaconda\python.exe`，纯 CPU、不联网、无 torch/tensorrt/真实数据集）。

CI 怎么验：`test_pipeline.py` 与 `test_prof.py` 只 import numpy/opencv/标准库，全量跑；
`test_pinned.py` 靠注入假对象覆盖全部分支，只有 1 条 `importorskip("torch")` 会跳。
`test_prof.py` **不要求机器上有显卡**：`nvidia-smi` 通过 `runner` 注入假对象，
时钟通过 `proc_clock`/`wall_clock`/`sleeper` 注入假读数，
所以「驱动不在 / 超时 / 输出是垃圾 / 字段返回 `[N/A]`」这些生产上偶发一次就毁掉一轮压测的分支，
在 CI 上是每次必跑且完全确定性的。无 torch 环境实测（临时目录里放一个 `torch/__init__.py` 直接 `raise ImportError`，
再用 `PYTHONPATH` 顶掉真 torch）：pipeline + pinned + prof **328 passed / 1 skipped**。

各文件条数：balance 12 / bench_trt 62 / build_trt 50 / copy_paste 38 / defect_bank 11 / deploy 51 /
eval_seg 16 / export_onnx 15 / focal 25 / make_calib_list 24 / mvtec 32 / oversample_cli 20 /
**pinned 59** / **pipeline 183** / **prof 87** / summarize_ablation 64 / synth_cli 12 / tools_cli 9 /
trt_infer 36 / verify_onnx 59 / verify_oversample 21 / yolo_dataset 12。

---

## 11. 复现命令

```powershell
conda activate induvision        # 要 GPU + tensorrt

# A 组：含磁盘读图（真实口径，92.2 s，对外报数用这组）
python tools/bench_pipeline.py --check-ultralytics --markdown

# B 组：图常驻内存，隔离磁盘（27.0 s）
python tools/bench_pipeline.py --preload --check-ultralytics --markdown `
    --json-out runs/logs/pipeline_baseline15_preload.json

# §7 的 pool_size 消融（各 10.0 s，只跑 pinned）
python tools/bench_pipeline.py --preload --modes pinned --pool-size 6

# 只看 stage3 占比（关掉掩码解码）
python tools/bench_pipeline.py --preload --no-masks

# §6.5.2 端到端延迟：500 帧，不开采样也不开对拍（FPS/延迟表用这一轮）
python tools/bench_pipeline.py --frames 50 --repeats 10 --markdown `
    --json-out runs/logs/pipeline500_nosample.json

# §6.5.3 资源占用：500 帧 + CPU 核当量 / GPU util / 显存
python tools/bench_pipeline.py --frames 50 --repeats 10 --sample-resources `
    --sample-interval 0.2 --markdown --json-out runs/logs/pipeline500_sampled.json

# §6.5.4 采样器扰动 A/B：把上面两条各跑两轮，比同格 FPS 相对差，
# 再拿「两轮都开采样」的差值当噪声底。一轮分不开噪声和效应。

# 单测（892 条，纯 CPU，base 环境即可）
python -m pytest -q
```

默认值：`--frames 30 --repeats 10 --capacity 4 --pool-size 2 --imgsz 640`，
阈值 conf 0.25 / iou 0.7 / max_det 300 / mask 0.5，
判据 `--min-speedup 1.0 --min-match-rate 0.98 --min-box-iou 0.99 --min-mask-iou 0.85`。
`--dry-run` 只看配置不跑。产物落 `runs/logs/`（不入库，数字以本文为准）。
