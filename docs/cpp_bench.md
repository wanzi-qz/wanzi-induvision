# W4② C++ 推理版：实测报告（里程碑 2+3）

> 2026-10-23 实测。基线：[cpp_infer.md](cpp_infer.md) §4 的判据表（数字出处
> pipeline.md §6.5，同清单同口径）；预研：[gpu_preproc.md](gpu_preproc.md)。
> 实测与估算分开标注，与 W4① 的老规矩一致。

一句话：**C++ 版全链路（imdecode → GPU letterbox → TRT → CPU 后处理）打通，
四层对拍全过、500 帧全等 n_diff=0；流水线 59.5/61.6 FPS，比 Python 同口径
快 31~35%，P50 21.8 → 4.1 ms（5.3×）。加速比 1.23× 未达 §4 的 1.44× 判据，
原因与口径分析见 §3——这条如实报告，不改判据。**

---

## 1. 交付物

| 文件 | 内容 |
| --- | --- |
| `cpp/src/preproc.cu` | 融合 kernel：BGR→RGB + /255 + bilinear(align_corners=False) + pad 114，直接写 engine 输入 device 缓冲（**没有 Python 版那笔 4.688 MiB D2H**，预研文档说的"保守下界"差距就在这） |
| `cpp/src/postproc.cpp/.h` | `pipeline.py::postprocess` 一族的逐条移植（sigmoid / batched_nms / scale_boxes / masks_from_proto），掩码 upsample 用与 Python 同一个 `cv2.resize` 算法 |
| `cpp/src/bench.cpp` | serial + 三线程流水线（环形队列 + slot 池 pinned + event），`--dump-preproc/--dump-outs/--dump-dets` 三个对拍探针 |
| `tools/check_cpp_bench.py` | Python 侧对拍器 + 资源采样（nvidia-smi 200ms + psutil，prof.py 口径） |
| `cpp/build.bat` / `CMakeLists.txt` | nvcc 单 TU 编译 + 显式 MSVC/SDK 环境（不调 vcvars，见 cpp_infer.md §2） |

## 2. 对拍结果（§4 判据逐条）

探针口径：`--dump-preproc 5 --dump-outs 20`，fp16 engine。

| 层 | 内容 | 判据 | 实测 | 结论 |
| --- | --- | --- | --- | --- |
| A | decode：C++ `imdecode` vs Python `cv2` | 逐字节一致 | **5/5** | PASS |
| B | 前处理：C++ kernel vs Python `GpuPreproc` | max_abs / cosine | **max_abs = 0，cosine = 1.0（逐位相等）** | PASS |
| C | 后处理移植：同一份 engine 输出 bin 喂两边 | score/cls/box/mask_pixels 全 tuple 相等 | **20/20** | PASS |
| D | 检测级：C++ dets vs Python 全链路 | match_rate ≥ 0.98、框 IoU ≥ 0.99、掩码 IoU ≥ 0.85 | match_rate **1.0000**，框 IoU min **0.9967**（30 帧） | PASS |

B 层的逐位相等值得多说一句：kernel 按 torch `upsample_bilinear2d` 的实现逐项
复刻（源坐标 `(dst+0.5)*in/out-0.5` 负值夹 0、四邻线性组合的浮点顺序、先 /255
再插值），比路线 C 预研里 cv2-vs-torch 的 max_abs 0.00350 还干净——那 0.0035
本来就是 uint8 域插值与 float 域插值的口径差，不是不可消除的。

D 层掩码像素数：30 帧里 24 帧逐像素相等，6 帧差 **±1~14 px（±0.04%）**——
跨运行时（numpy SIMD vs C++ 标量）1ulp 级别落在 0.5 阈值边界的翻转，量级在
"边界 1~2 px"口径内，掩码 IoU 远高于 0.85 判据。

## 3. 压测结果（500 items = 50 帧 × 10 repeats，fp16，capacity 4）

| 指标 | C++ serial | C++ pipeline | Python serial¹ | Python pipeline¹ |
| --- | --- | --- | --- | --- |
| FPS | 48.42 / 48.23 | **59.47 / 61.59** | 40.66 | 63.94（fp32）/ 45.4、45.7（fp16） |
| P50 ms | 21.8 / 22.1 | **4.1 / 3.8** | — | 140 → 109~120（GPU 前处理后） |
| P95 ms | 28.6 | 6.5 | — | — |
| P99 ms | 32.2 | 10.8 | —（Python 只报到 max） | — |
| CPU 核当量 | 0.98~1.00 | 1.31 | 1.10 | 1.65 |
| GPU util | 20.9~21.9% | 25.1% | — | 28~37%（A 组） |

¹ 出处 pipeline.md §6.5（fp32 那行）与 gpu_preproc.md（fp16 那行）；不同轮次，
绝对值只并列报、不跨轮比（机器噪声底 3.57%）。

**500 帧全等（§6.4 口径，钉跨帧数据竞争）**：`(n_det, mask_pixels, score_sum)`
逐帧比较 serial vs pipeline，两轮全部 **500/500，n_diff = 0**；两次 serial 之间
也 500/500（确定性）。checksum：dets 270 —— 与 Python 版 500 帧的 270 完全一致。

### 3.1 加速比 1.23× 未达 1.44× 判据：原因与口径（如实报告）

§4 的"同轮内加速比 ≥ 1.44×"基线是 **Python fp32：serial 40.66 / pipeline 63.94 =
1.57×**（pipeline.md §6.5.1）。C++ 实测 serial→pipeline 只有 **1.228× / 1.233×**。
拆开看：

- C++ serial（48.4 FPS）已经比 Python serial（40.66）**快 19%**——serial 路径
  本来就吃到了 GPU 前处理（预研实测省 ~1.8 ms/帧）和 C++ 的低开销；
- 流水线收益上限被 decode 段压住：Amdahl。decode 线程单飞约 15 ms/帧
  （900×900 PNG），理论上限 ~66 FPS，实测 59.5（decode util ≈ 0.9）——
  **瓶颈原位在 imdecode**，与 gpu_preproc.md 的结论一致，C++ 版没有消除它。

收益去了哪：**P50 21.8 → 4.1 ms（5.3×）**、吞吐 +23%、CPU 1.65 → 1.31 核。
"1.44×"这个数是对着 Python 的 CPU-preproc serial 基线定的；serial 基线被 C++
强化之后，同一个流水线架构能给出的比值必然缩水——**判据该按新基线重校，
而不是把流水线改差去凑比值**。要再上台阶，下一刀是 nvJPEG/GPU 解码
（cpp_infer.md §3 第 1 条的真正主角），不是更激进的三线程。

### 3.2 资源口径

资源采样在 Python 侧做（§4：口径一漂两张表不可比）：`check_cpp_bench.py` 拉起
bench.exe，按 prof.py 的口径 200ms 采样 nvidia-smi + psutil。激活显存
`getDeviceMemorySizeV2()` = 18.286 MiB（fp16，单 context）；pinned slot 池
4 × ~7.4 MiB ≈ 29.7 MiB 是 host 内存，与 §7.3 那笔显存账分开记。

## 4. 移植时的三个真 bug（都在对拍里抓到的，写下来防复发）

1. **掩码系数步长**：`out0` 是 (attrs, anchors) 行主序，系数沿属性维的步长是
   `anchors` 不是 1——第一版按连续读，帧 0 掩码 102559 → 4682。
2. **框域搞反**：掩码解码吃的是 **640 域**框（scale_boxes 之前的），第一版把
   原图坐标喂了进去（102559 → 72140）。两处都是 C 层对拍抓到的，靠
   "同一份 bin 两边跑"这个探针设计才可能定位——没有 C 层，这两个 bug 会被
   B 层的逐位相等掩盖到检测级 IoU 里，变成"查不出来的 0.9"。
3. **Windows 中文路径三连**：MSVC `main` 的 argv 是 ANSI(GBK) 编码，`--dump-dir`
   带中文经 `filesystem::u8path` 直接 terminate（0xC0000409）——统一从
   `GetCommandLineW` 重取转 UTF-8；`ifstream`/`create_directories` 吃窄字符
   路径同样按 GBK 误读，全部走 `u8path`。这与 Python 侧 `cv2.imread` 的坑
   （deploy.read_image_bgr 的 docstring）同源。

工程侧还有一条：**nvcc 默认静态 CRT（/MT）**，与 CMake 的 /MD 混链会报一堆
LNK4217/LNK2019（`__imp_puts`）——`-Xcompiler /MD` 对齐。

## 5. 工具链增量（相对 cpp_infer.md §2）

- **OpenCV（C++ 侧 decode/resize）**：`conda create -p C:\Users\11646\cppenv
  -c conda-forge libopencv`（OpenCV 5.0.0，模块版 .lib：core500/imgproc500/
  imgcodecs500）。**特意开独立前缀 env**：往 induvision 里装会动 python 本体
  （anaconda build → conda-forge build），这个环境跑着训练和压测，不值得冒。
- nvcc 13.4.92 编 `preproc.cu`：`-arch=native -Xcompiler /utf-8 /MD`。
- 复现：`cpp\build.bat`（先 nvcc 后 cmake），运行期 DLL PATH 见
  `check_cpp_bench.py::_bench_env`（tensorrt_libs + torch/lib + cppenv/bin）。
