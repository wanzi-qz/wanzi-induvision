# 部署链路（W3）：ONNX 导出与精度对齐

口径同 [baseline.md](baseline.md)：本文所有数字都是本机实测。没跑到的一律写 **待测**，不预填、不估。

---

## 1. 这一段要回答什么

W2 结束时手上有一份能用的权重：`runs/segment/baseline15/weights/best.pt`，test split 上
mask mAP50-95 **0.3740**（15 类全量，435 图 / 328 实例，见 [baseline.md](baseline.md) §2）。
W3 要把它搬出 PyTorch：先导出 ONNX，之后才是 TensorRT 三档量化与 C++ 推理服务。

换后端最容易被含糊过去的一句话是"精度基本没掉"。这句话要落到数字上，得同时回答三个问题：

| 层 | 问题 | 手段 | 判据 |
| --- | --- | --- | --- |
| 1 | 算子层面等价吗 | 同一批输入、同一份前处理，两个后端的原始输出张量逐个比差 | cosine ≥ 0.9999 且 rel_l2 ≤ 1e-3，形状与输出数一致 |
| 2 | 业务指标等价吗 | 拿 .onnx 当权重在 test split 上重算 mAP，与 .pt 的参照成绩相减 | \|Δ mask mAP50-95\| ≤ 0.01 |
| 3 | 换后端换来多少速度 | batch=1 单帧延迟的 mean / p50 / p95 / p99 / max | 不设判据，只记录 |

第 1 层最灵敏，也是唯一能把差异归因到"导出与算子实现"的一层——它绕开了 NMS、阈值、
匹配这些后处理。第 2 层才是能写进报告和简历的话。第 3 层报延迟只报均值等于没报，
尾巴由队列抖动和后处理毛刺决定，所以分位一起给。

三层都在 `tools/verify_onnx.py` 里，结果落 `runs/segment/baseline15/verify_onnx_fp32.json`
（不入库），加 `--markdown` 会把表格直接打出来，本文的三张表就是这么生成的。

这三层之前还有一个**第 0 层**：把"到底用哪几张图"固定成一份入库清单。口径不固定，
上面三层的数字在两轮之间就没法比 —— §4.0 讲这件事，本文所有数字都出自同一份清单。

## 2. 前处理为什么要自己复刻一份

比差的前提是两边吃到的输入**逐字节相同**。前处理只要差一点——ratio 的取整方式、
pad 是否居中、BGR/RGB 顺序、有没有除 255、CHW 还是 HWC——量出来的差就全是前处理口径
的差，后端真正的差反而被淹掉。

所以有了 `induvision/deploy.py`：把 ultralytics 8.4.183 的前处理复刻成 numpy/opencv 实现，
PyTorch 与 ONNX Runtime 共用同一份。

| 复刻对象 | 口径 |
| --- | --- |
| `data/augment.py: LetterBox` 的 `auto=False` 分支 | 等比缩放到正方形后居中 pad，**不**向 stride 取整。ONNX/TensorRT 后端在 predictor 里走的就是这个分支（`auto=pt and self.pt`，非 .pt 权重恒为 False） |
| pad 的上下分配 | `int(round(dh/2 - 0.1))` / `int(round(dh/2 + 0.1))`。余量为奇数时上下差 1 像素，自己写 `dh//2` 会和 ultralytics 对不上 |
| pad 颜色 | `(114, 114, 114)` + `BORDER_CONSTANT`，不是镜像也不是黑边 |
| `engine/predictor.py: pre_process` | BGR→RGB、HWC→CHW、float32、除 255、`ascontiguousarray`；batch 维留给调用方加，单图对齐与多图压测就能共用一份代码 |

**复刻而不是 `import ultralytics`**：那会连带拖进 torch，CI（刻意不装 torch，见
`requirements-ci.txt`）就跑不了这段逻辑的单测。deploy.py 只依赖 numpy / opencv / pyyaml，
35 条单测在 CI 里全绿。

这个模块同时是 W3 后半段的地基：INT8 的校准集抽样（`sample_paths`）、TensorRT 的
延迟分位（`latency_stats`）、C++ 侧要对齐的前处理口径，都从这里出。

## 3. 导出：tools/export_onnx.py

```powershell
conda activate induvision
python tools/export_onnx.py --weights runs/segment/baseline15/weights/best.pt --markdown
```

导出不等于导对。所以脚本在 ultralytics 导完之后立刻做一遍**静态体检**，把结果一起落档：
opset 有没有真的落到 17、输入第 0 维是不是符号维（动态 batch）、`onnx.checker` 过不过、
图里有哪些算子（TensorRT 认不认某个算子，看这张算子表比翻日志快）。三条任一不达标
返回码 3：checker 失败、要了 dynamic 但没生效、实际 opset 低于要求。

产物会复制一份到稳定路径 `weights/<run>_<tag>.onnx`：ultralytics 自己落在 run 目录里，
换 run 名就找不到，后面的压测脚本不该依赖那个位置。

### 实测（FP32，2026-10-22，git de6060b）

| 项 | 值 |
| --- | --- |
| 源权重 | `best.pt`（run `baseline15`） |
| 产物 | `weights/baseline15_fp32.onnx`，**45.53 MB**（47,737,525 B） |
| opset / IR | **17** / 8，producer `pytorch 2.13.0` |
| 动态 batch | 是（第 0 维是符号维 `batch`） |
| 图节点数 | 357（22 种算子） |
| 输入 | `images` `[batch, 3, height, width]`，动态轴 0/2/3 |
| 输出 | `output0` `[batch, 51, anchors]`；`output1` `[batch, 32, mask_height, mask_width]` |
| onnx.checker | 通过 |
| 导出 / 体检耗时 | 8.2 s / 0.2 s |
| verdict | **pass** |

`output0` 的第二维 **51 = 4（box）+ 15（类别）+ 32（mask 系数）**。
[environment.md](environment.md) §4 那次自检用的是 COCO 预训练权重，同一位置是
**116 = 4 + 80 + 32**，两个数别混。`output1` 是 32 通道原型掩码，160×160（imgsz 640 的 1/4）。

算子清单：Add、Cast、Concat、ConstantOfShape、Conv、ConvTranspose、Div、Expand、Gather、
MaxPool、Mul、Range、Reshape、Resize、Shape、Sigmoid、Slice、Softmax、Split、Sub、
Transpose、Unsqueeze。全是 TensorRT 常规支持的算子，没有 GridSample / NonMaxSuppression
这类容易卡住的——NMS 在后处理里，没被折进图。

机器可读版落 `runs/segment/baseline15/export_onnx_fp32.json`（不入库）。

## 4. 三层校验实测

```powershell
python tools/verify_onnx.py --weights runs/segment/baseline15/weights/best.pt `
    --align-n 16 --bench-n 30 --val --val-batch 8 --markdown
```

一次跑完 **81.5 s**：逐张量对齐 8.6 s + 端到端 val 63.9 s + 延迟压测与读图约 9.0 s。
verdict **pass**。本节数字全部出自这一轮，抽样口径是入库清单 `configs/bench_sample.json`
（sha256 `eccfd964426a2637`）—— 它为什么必须固定，见 §4.0。

### 4.0 第 0 层：抽样口径要落成清单

三层里有两层要吃图：第 1 层逐张量对齐要 16 张，第 3 层延迟压测要 30 帧。
原来的口径是 `sample_paths(images, max(align_n, bench_n))` 等间距抽样，再取前 `align_n` 张。
听起来是确定的（等间距只取决于排序后的列表），实际上**抽样长度跟着参数变**：
加不加 `--bench-n`，母体的切分点就变了，第 1 层用的那 16 张跟着换一批。

结果是两轮报告的数字没法直接比。同一份权重、同一台机器实测过：

| 口径 | `output0` max_abs | `output0` rel_l2 | `output1` max_abs |
| --- | --- | --- | --- |
| `--bench-n 30`（抽 30 张取前 16） | 1.617e-03 | 2.400e-07 | 2.241e-05 |
| 只跑第 1 层（`sample_paths(435, 16)`） | 2.441e-03 | 3.000e-07 | 2.193e-05 |

两个数都离判据（rel_l2 ≤ 1e-3）差三个数量级，谁也没"错"；但 `output1` 的最坏值还**反向变小**了 ——
差的是图，不是后端。延迟那层更麻烦：换 30 张图，mean 能差好几毫秒，
而这几毫秒正好是 FP16/INT8 要量的量级。口径不固定，量化掉点根本没法归因。

所以把清单固化下来，落成入库的 `configs/bench_sample.json`：

| 字段 | 值 | 为什么要有 |
| --- | --- | --- |
| `files` | 64 条相对 `root` 的 POSIX 路径，**顺序即口径** | align 取前 16、bench 取前 30，共用同一个头。顺序变了口径就变了，所以指纹计算刻意**不排序** |
| `root` | `out/yolo_mvtec15`（相对仓库根） | 清单里绝不出现绝对路径：盘符、用户名、仓库位置任何一个变了都全废 |
| `sha256` | `eccfd964426a2637` | 相对路径列表的 sha256 前 16 位，与机器、时间无关。两轮报告指纹相同 = 数字可以直接相减 |
| `count` / `source_total` | 64 / 435 | 母体记下来，日后数据集扩容能一眼看出这份清单是旧的 |
| `split` / `imgsz` | `test` / 640 | 也是口径的一部分，读清单时和当前运行参数比对 |
| `version` | 1 | 清单格式版本。不符直接抛，不按旧格式静默读 |
| `git_head` / `created_utc` | `591cd7c` / `2026-10-22T15:41:56Z` | 追溯这份清单出自哪个提交、什么时候生成 |

64 这个长度是折中：喂得起第 1 层（每张图两个后端各前向一遍）和第 3 层（30 帧压测），
又不至于让一轮校验跑几分钟。这 64 张覆盖全部 15 类。

生成清单只做一次，产物入库：

```powershell
python tools/verify_onnx.py --sample-n 64 `
    --sample-list-out configs/bench_sample.json --dry-run
```

`--dry-run` 时唯一允许落盘的就是清单本身（生成基准不该先把两个后端跑一遍），
而且刻意 `use_list=False` 从数据集重抽 —— 拿旧清单当母体等于没换口径。
注意 dry-run 仍然要求权重、`data.yaml` 与 ONNX 都在，只是不真的跑校验。

读清单是**清单优先**：文件存在且能对上当前数据集就用它。对不上时（文件不存在 / 有图读不到 /
`version` 不符 / 记的 `split` 不符 / 清单里的图不在本次 split 中）降级成即时等间距抽样，
并且在三处同时留痕，不会悄悄发生：

- stdout 的 `[warn ]`；
- JSON 的 `sample.source`（`manifest` = 用了清单，`derived` = 现抽的）与 `sample.warnings`；
- Markdown 报告里的 `> 注意：` 引文。

不接受降级就加 `--strict-sample`，直接返回码 2。非 dry-run 还有一条硬检查：
清单张数 < `max(align_n, bench_n)` 就报错退出并提示重新生成 —— 切不出片比降级更糟。

### 4.1 第 1 层：逐张量对齐

16 张 test 图（清单 `configs/bench_sample.json` 的前 16 条，具体哪 16 张见 JSON 的
`images_used`），PyTorch 与 ONNX Runtime 各前向一遍。
**PyTorch 侧钉在 CPU**（`--align-device cpu`，默认值）：这样两边只差"导出与算子实现"
这一个变量。拿 GPU 上的 PyTorch 去比 CPU 上的 ORT，量到的差里混着 cuDNN 卷积算法选择
的不同，说不清是谁的锅。想量那个混合差就把它改成 `0`。

| 输出 | 形状 | max_abs | mean_abs | rel_l2 | cosine |
| --- | --- | --- | --- | --- | --- |
| `output0` | [1, 51, 8400] | 1.617e-03 | 1.880e-06 | 2.400e-07 | 1.00000000 |
| `output1` | [1, 32, 160, 160] | 1.478e-05 | 3.800e-07 | 1.300e-06 | 1.00000000 |

怎么读这三列：

- **判等价看 rel_l2 与 cosine，不看 max_abs。** `output0` 里 box 坐标是几百的量级、
  mask 系数是 0~1，绝对差跨张量没有可比性；1.6e-3 的最大绝对差摊在 43 万个元素上，
  相对 L2 只有 2.4e-7。
- **聚合取最坏值，不取平均**：max_abs / rel_l2 取最大、cosine 取最小。平均数会把
  一张图上的崩坏摊薄成看不见的量级。
- 8400 = 80² + 40² + 20²，三个检测头的 anchor 总数；160×160 是原型掩码分辨率。
  两个数都与 imgsz 640 对得上，说明导出没有偷偷改输入尺寸。

判据实现在 `check_align()`：四条同时成立才 pass（输出数一致、worst cosine 达标、
每个输出形状一致、每个输出 rel_l2 达标），不达标时把**每一条**原因都打出来，
不是只丢一个 "fail"。
### 4.2 第 2 层：端到端 mAP

拿 `.onnx` 当权重走 ultralytics 的 AutoBackend，在 test split（435 图，**整份，与抽样清单无关**）上重算，
口径与 `tools/eval_seg.py` 完全一致（同 data.yaml、同 split、同 imgsz 640），
所以两边的数字可以直接相减。参照取自 `runs/segment/baseline15/eval_test.json`。

| 指标 | .pt 参照 | .onnx 实测 | 差值 |
| --- | --- | --- | --- |
| mask mAP50-95 | 0.3740 | 0.3741 | +0.0001 |
| mask mAP50 | 0.6730 | 0.6733 | +0.0004 |
| mask P | 0.7246 | 0.7262 | +0.0017 |
| mask R | 0.6346 | 0.6332 | -0.0014 |
| box mAP50-95 | 0.4229 | 0.4228 | -0.0001 |
| box mAP50 | 0.6652 | 0.6653 | +0.0000 |
| box P | 0.7123 | 0.7123 | -0.0000 |
| box R | 0.6192 | 0.6192 | +0.0001 |

**导出没有掉点**：主指标 mask mAP50-95 实测差 +0.000059，比判据 ±0.01 小两个数量级。

两个读数提醒：

1. 判据用绝对值 ±0.01 而不是"掉了就失败"。端到端 mAP 本身带设备与后处理抖动，
   卡在 0 上会让正常的浮点噪声变成红灯。这次四个指标微涨、四个微跌，最大摆幅 0.0017，
   就是这种抖动的量级。
2. **差值里混着设备差异**：参照那次 `device=0`（GPU，26.9 s），这次 ORT 跑在 CPU 上
   （63.9 s）。第 1 层已经把"导出本身"单独量干净了，所以这组数字读作
   "端到端换到 ONNX 后端成绩不变"，不要读成"CPU 与 GPU 等价"。

### 4.3 第 3 层：单帧延迟

batch=1、imgsz 640、30 帧（清单前 30 条），前 3 帧 warmup 不计时（首帧要付算子初始化、显存分配、
cudnn 选算法的钱，混进去会把 P99 拉成假的高）。

| 后端 | 设备 | mean | p50 | p95 | p99 | max | FPS(mean) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ONNX Runtime | AzureExecutionProvider, CPUExecutionProvider | 94.58 ms | 94.19 | 99.59 | 100.35 | 100.50 | 10.57 |
| PyTorch | cuda:0 | 13.13 ms | 13.36 | 15.91 | 16.41 | 16.59 | 76.17 |

**这两行不是"ORT 比 PyTorch 慢 7 倍"，是"CPU 比 GPU 慢 7 倍"。** 本机装的是
onnxruntime CPU 版（刻意不装 `-gpu`，理由见 [environment.md](environment.md) §6），
providers 里没有 CUDAExecutionProvider。表里那个 `AzureExecutionProvider` 是
onnxruntime 1.31 自带注册的 EP，模型仍旧跑在 CPU 上（94.58 ms 就是证据）；上一轮
providers 里只有 `CPUExecutionProvider`，两轮的 EP 列表对不上是环境变了，不是代码变了。
ONNX 在本项目里的角色是导出中间格式与数值对齐的载体，速度这一段要等 TensorRT。

所以这组数字的正确用法有两个：给 W3 后半段的 FP32/FP16/INT8 三档当**同机基线**，
以及给 W4 的 C++ 推理服务当"Python 单线程能跑多快"的对照组。

顺带一个可读的信号，也是把口径固定下来最直接的收益：这轮两行尾巴都很平
（ORT p99/p50 = 1.07，max 100.50 几乎贴着 p99 100.35；PyTorch p99/p50 = 1.23）。
生成清单之前那轮 ORT 是 p99 118.62 / max 124.84，明显翘尾 —— 图不一样、机器负载也不一样，
连"尾巴平不平"这种定性结论都会跟着口径变。

## 5. 返回码与落档

| 返回码 | 含义 |
| --- | --- |
| 0 | 判据全过，verdict=pass |
| 2 | 输入缺失：权重 / data.yaml / onnx 不存在，该 split 没有图，或抽样清单用不了（`--strict-sample` 下缺图 / split 不符 / `version` 不符）、清单张数撑不起 `max(align_n, bench_n)` |
| 3 | 判据不达标（verdict=fail），或导出体检告警 |

JSON 里除三层结果外还落 `config`（含三条阈值与五个清单参数）、`sample`（口径来源、指纹、告警、跳过数）、`images_used`、`git_head`、
`env`（python / numpy / cv2 / torch / ultralytics / onnx / onnxruntime 版本 + CUDA +
GPU 型号 + ort providers）。

**`sample` 与 `images_used` 不是装饰，两次跑对不上数时先看它们。** `sample.sha256`
是这一轮口径的指纹，`sample.source` 说明这批图是入库清单（`manifest`）还是现抽的
（`derived`），`sample.skipped` 是有几张读不出来被跳过了，`images_used` 是实际喂进去的那几张。
**指纹不同就别把两轮的延迟直接相减。**

这个坑已经用清单填掉了，两次实测对照见 §4.0。修完之后 `--align-n` / `--bench-n`
只决定"从同一个头上切多少张"，不再决定"切的是哪批图"。本文 §4 的全部数字都出自
sha256 `eccfd964426a2637` 这一份清单（`--align-n 16 --bench-n 30 --val --val-batch 8`，三层全跑）。

环境指纹不是装饰。W2 实测过 cv2 4.10 与 5.0 转出的 788 个标签里有 12 个差约 1 像素
（[baseline.md](baseline.md) §4 第 7 条）。数值对齐对环境更敏感，**换环境必须重跑校验**，
不能引用旧 JSON 的结论。

## 6. 踩过的坑

1. **`cv2.imread` 在中文路径上返回 None。** 数据集落在 `...\ChatGPT\简历\induvision\out\`，
   opencv-python 5.0.0 + Windows 下 `imread` 读不出来，看着像文件坏了，其实文件完好
   （同一张图 `np.fromfile` + `imdecode` 能正常读出 900×900×3，PIL 也读得到）。
   `read_image_bgr()` 的顺序是"先 imdecode，兜底才 imread"：imdecode 不挑路径编码，
   也不会往 stderr 每张图刷一行 `cv::findDecoder` 的 WARN——那行 WARN 一开始把我骗到了，
   以为是数据集损坏。
2. **`net.to("0")` 报 `RuntimeError: Invalid device string: '0'`。** ultralytics 的
   `device="0"` 是它自己解析的，直接丢给 torch 不行。`_torch_device()` 把纯数字翻成 `cuda:0`。
3. **比形状必须在 ravel 之前。** `(1,51,8400)` 与 `(1,8400,51)` 元素数相同，拉平后能算出
   一个"看着还行"的差，而两个后端根本没对齐。`compare_tensors()` 先比 shape，
   不一致直接返回 `shape_match=False`、其余指标为 `None`。
4. **配置是 `json.dumps` 打出来的，Windows 路径的反斜杠被转义成 `\\`。** 单测里拿裸字符串
   做子串匹配必然失败，要解析回 JSON 再比 `Path`。
5. **切片要按清单下标切，不能按读进来的顺序切。** `_load_inputs_indexed()` 会跳过读不出来的
   坏图，跳一张，后面所有图就往前挪一位；这时再 `xs[:align_n]` 拿到的就不是清单的前 16 条了，
   口径悄悄平移且没人报警。`_take_upto(xs, idx, limit)` 按每张图在清单里的**原始下标**筛，
   坏图只会让这一轮少一张（记在 `sample.skipped`），不会换掉别的图。
6. **providers 里多出 `AzureExecutionProvider` 不是代码改的。** onnxruntime 装了什么 EP
   就注册什么，模型跑在 CPU 上的事实没变。它会落进 JSON 的 `env.ort_providers`，
   两轮对不上先看这里，别去翻 diff。

## 7. 测试覆盖

| 文件 | 条数 | 覆盖 |
| --- | --- | --- |
| `tests/test_deploy.py` | 51 | letterbox 逐条对齐 ultralytics（含奇数余量时上下差 1 px、pad 颜色、`scaleup` 两种口径、还原坐标用的元信息）、preprocess 的形状/dtype/值域/通道序/无 batch 维、compare_tensors 的形状前置与退化输入（全零、正交、等比缩放）、latency_stats 的分位与空输入、sample_paths 的等间距与确定性、read_split_images 的六种 data.yaml 写法、read_image_bgr 的两条兜底路径；**新增 16 条**盖清单读写：relative_to_root 的相对/绝对两条分支、sample_fingerprint 的顺序敏感与机器无关、write/load_sample_list 的往返一致、version 不符抛 ValueError、JSON 损坏抛 ValueError、strict 缺图抛 FileNotFoundError、非 strict 把缺图列进 `meta["missing"]`、root 的三级优先级 |
| `tests/test_export_onnx.py` | 11 + 4 skip | 路径与命名推导、CLI 口径是否正确落到配置、ONNX 图的静态体检解析（动态/静态 batch 两种）、checker 失败时的收口；skip 的 4 条要 `onnx`（CI 不装，`importorskip`） |
| `tests/test_verify_onnx.py` | 59 | `_torch_device`、`_delta`、`load_reference`（缺文件 / JSON 损坏）、`check_align` 与 `check_val` 的每条判据、`_md` 三层渲染（含无参照时的 `—`、只有一个后端时跳过另一行）、CLI 的路径推导与四种错误收口、`_load_inputs` 的前处理与坏图跳过；**新增 22 条**盖固定口径：`_load_inputs_indexed` 的下标与坏图跳过、`_take_upto` 按清单下标切（坏图不平移口径）、`resolve_sample` 的全部分支（清单命中 / `--no-sample-list` / 清单缺失 / version 不符 / split 不符 / 图不在本次 split / `strict=True` 抛）、CLI 的五个清单参数与相对路径按 `REPO_ROOT` 还原、`--dry-run --sample-list-out` 真的落清单、`--strict-sample` 返回 2、清单张数不够返回 2、`_md` 的抽样口径行与降级 `> 注意：` 引文 |
| `tests/test_trt_infer.py` | 36 | engine 文件的 ultralytics 风格 metadata header：`encode_header`/`decode_header` 往返、`_coerce` 把 `names`/`stride`/`imgsz` 还原成 dict/int、无 metadata 时退化成纯 blob、`write_engine_file`/`read_engine_file` 往返一致、`onnx_metadata` 缺 props 返回空、`device_memory_bytes`、`TrtRunner` 在没装 tensorrt 时的报错口径 |
| `tests/test_make_calib_list.py` | 24 | 校准清单：`class_of` 从路径反推类别、`dataset_root` 的三级回退、`build_list` 的等间距抽样与覆盖统计、拒绝 val/test split（校准集只能用 train，防止拿评测集校准）、拒绝 n<=0 与母体不足、幂等（同参数重跑同指纹）、指纹不同拒绝覆盖、瘦类别覆盖告警、dry-run 不落盘、写出的清单能被 strict 模式读回 |
| `tests/test_build_trt.py` | 50 | `pin_layers_fp16`（失败对照）与 `constrain_head_precisions`（能过的配方）：头是最大 model 索引、sigmoid→FP32、`.2/Conv` 与 `/dfl/`→FP16、非头层不动、无名层不崩、逐输出都 set；`make_calibrator` 的批生命周期（`get_batch` 返回裸 `data_ptr()`，张量必须挂在 `self._cur`）、`get_algorithm` 已重写、`CALIB_ALGOS` 映射、缓存命中/未命中/写缓存建父目录、缺基类与缺枚举时报错指向下一步；`load_calib_inputs` 的分批与跳过留痕；`--int8-pin` 与 `--int8-head` 互斥；`main` 的返回码（2 用法/输入、3 缺 tensorrt）与 dry-run 命名推导；**新增 4 条**盖 TF32：fake builder 全链路 harness（`FakeConfig` 记 `set_flag`/`clear_flag`，`FakeLogger` 必须是真类 —— `SimpleNamespace` 塞 `__call__` 不生效）、默认开 TF32、`--no-tf32` 真的 `clear_flag`、与 FP16 flag 共存（断言用 `set()`，顺序不重要）、dry-run 报 `tf32=` 状态 |
| `tests/test_bench_trt.py` | 62 | 三层判据逐条：`align_engine` 的形状前置与逐输出指标、`check_align` 的 cosine/rel_l2 卡线、`gate_tier` 三档各自的判据（含 INT8 只看 mAP）、`low_prec_share` 的层格式占比、`build_report_tiers` 读构建报告（键是 `engines` 不是 `builds`）、`build_table` 引报告里的体积/层数、`latency_stats` 分位、`val_engine` 的 mAP 抽取、`_md` 渲染（无参照时的 `—`）、`main` 的缺 engine/缺 data.yaml 返回 2、run 名从权重路径反推、阈值覆盖只作用在 fp32/fp16；**阈值那条改写成** `test_thresholds_fp32_fp16_match_their_numeric_format`（卡 1e-2，理由写在注释里：按数值格式的机器精度定判据，最终判据是端到端 mAP），另加 `test_rel_l2_threshold_still_admits_tightening_via_cli`（`--max-rel-l2 1e-5 --dry-run` 仍可收紧，判据不是写死的） |
| `tests/test_pipeline.py` | 183 | 环形队列、后处理、掩码解码、流水线保序/错误传播、对拍匹配、加速比口径；**端到端延迟 15 条**（`e2e_latency`）：e2e ≥ 各阶段耗时之和、`1000/e2e 均值 ≈ FPS` 两套数互验、**排队效应（流水线 e2e P50 > 1.5× 串行，守的是「时间戳打在 `put` 之前」，打在之后就等于把背压藏起来）**、capacity 变大 e2e 随之变大 |
| `tests/test_prof.py` | 87 | 资源采样器：`nvidia-smi` 输出解析（`[N/A]` / `[Not Supported]` / 垃圾 / 多行只取首行）、`smi_sample` 的 17 条失败路径全回 None 且**绝不抛**（子进程缺失/超时/非零退出）、`cpu_cores` 的核当量与「读数倒退不许报负数」、`ResourceSampler` 的参数校验（≤0 / **NaN** / 非数字 / 字符串数字收敛）+ 注入假时钟做确定性断言、采样点计数、start 幂等、stop 幂等且 join、daemon 线程名 |
| `tests/test_pinned.py` | 59 | 令牌池、non_blocking、每帧单 sync、出错还令牌（全用假对象，不碰 GPU） |

全量 **892 passed / 6 skipped**（base 环境 `W:\Anaconda\python.exe`，纯 CPU、不联网、
不需要真实数据集，19.88 s；跳的 6 条是 onnx 4 + ultralytics 2）。W3b 新增 172 条：trt_infer 36 /
make_calib_list 24 / build_trt 50 / bench_trt 62；W4① 再新增 329 条：pipeline 183 / pinned 59 / prof 87。
**这些测试既不需要装 tensorrt，也不需要 GPU** —— `build_trt` / `bench_trt` 里的
`import tensorrt` 全部延迟到函数内，流水线的环形队列、令牌池与 CUDA 流交互全用假对象断言
（连"每帧只 synchronize 一次""异常路径要把令牌还回池子"都是测出来的），CI 只跑逻辑不跑 GPU。
CI 里的 skip = 要 onnx 的 4 条 + 要 torch 的（校准器那 2 条 + 原有 2 条）。

记一条 CI 抓出来的测试写法问题，因为它在本机**永远不会红**：新增那条
`--max-rel-l2` 收紧判据的测试忘了传 `base_args`，`main()` 就落到仓库默认路径
`weights/baseline15_trt_fp32.engine`。本机那份真 engine 在，dry-run 返回 0，测试通过；
CI 上没有 engine，返回 2，四个 job 一起红。**本机有真产物 = 忘传参数的测试会假装通过**，
所以 CLI 类测试一律显式传 `--weights/--data/--engine-dir` 指向 tmp_path 里的假文件。
复现 CI 的无 torch 环境也有个便宜办法：建一个 `_cishim/torch/__init__.py`
里面 `raise ImportError`，`PYTHONPATH=_cishim pytest` 就能本地验 skip 分支。

## 8. W3b 三档量化实测（TensorRT 10.16.1.11）

实测时间 2026-10-23，git `17ded7d` + 本轮工具改动。

### 8.0 这轮的口径先说清楚

`pip install tensorrt-cu13` 装的是 wheel，**机器上没有 trtexec.exe**（只有 Python 模块）。
所以本节所有 engine 都是 `tools/build_trt.py` 用 Python API（`trt.Builder` +
`build_serialized_network`）建的，压测走 `tools/bench_trt.py`，engine 的反序列化 / metadata
header / 前处理上显存封在 `induvision/trt_infer.py::TrtRunner`。前面 §8 老版本里写的
「trtexec 直接建 engine」那条计划作废，命令见 §8.9。

三档共用同一份 ONNX（W3a 导出的 `weights/baseline15_fp32.onnx`，sha256 `261623c3913a0340`）。
抽样口径是入库清单 `configs/bench_sample.json`（64/435，sha256 `eccfd964426a2637`，前 16 张做逐张量对齐、
前 30 张做延迟）；校准集另有一份 500 张的 `configs/calib_sample.json`（sha256 `20b7ab0c9ce76175`，从 train split 4490 张等间距抽，
由 `tools/make_calib_list.py` 生成并入库）。
环境：NVIDIA GeForce RTX 4060 Laptop GPU / TRT 10.16.1.11 / CUDA 13.0 / Python 3.12.5 / torch 2.13.0+cu130 / ultralytics 8.4.183。

### 8.1 构建结果

| 档 | engine | 落盘（含 header） | 构建 | 层数 | 激活显存 | builder flags | 层输出格式 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FP32 | 65.25 MiB | 65.25 MiB (header 848 B) | 94.0 s | 253 | 39.84 MiB | `无` | Float 253 (100.0%) |
| FP16 | 24.58 MiB | 24.58 MiB (header 848 B) | 210.9 s | 190 | 18.29 MiB | `FP16` | Half 163 (85.8%) / Float 27 (14.2%) |
| INT8 | 12.55 MiB | 12.55 MiB (header 848 B) | 152.0 s | 168 | 18.41 MiB | `INT8, FP16(fallback), OBEY_PRECISION_CONSTRAINTS, calib=minmax` | Int8 107 (63.7%) / Half 44 (26.2%) / Float 17 (10.1%) |

三点值得说：

- **INT8 档不是纯 INT8**：168 层里只有 107 层（63.7%）真落在 Int8，44 层 Half、17 层 Float。这是 §8.3 那个构建失败的直接代价——
  不给它 FP16 退路，SM89 上连一条可用实现都选不出来。报压缩比的时候必须连着这个直方图一起报，
  否则会高估量化程度。
- **压缩比**：FP32 65.25 → FP16 24.58（2.65×）→ INT8 12.55 MiB（5.2×）。
- **激活显存**（`engine.device_memory_size`，不含权重）：39.84 → 18.29 → 18.41 MiB。
  INT8 比 FP16 略高，混合精度要在两种格式之间留 reformat 缓冲。

### 8.2 压测三层结果

| 档 | mask mAP50-95 | Δ vs 判据基准 | Δ vs `.pt` | mask mAP50 | box mAP50-95 | worst cosine | out0 / out1 rel_l2 | mean / p50 / p95 ms | FPS | 进程显存增量 | gate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FP32 | 0.3707 | -0.0033 | -0.0033 | 0.6694 | 0.4152 | 0.99999465 | 8.342e-04 / 3.275e-03 | 9.51 / 9.39 / 11.35 | 105.2 | 150.0 MiB | pass |
| FP16 | 0.3702 | -0.0005 | -0.0038 | 0.6690 | 0.4146 | 0.99997915 | 1.110e-03 / 6.515e-03 | 6.73 / 6.69 / 10.66 | 148.6 | 52.0 MiB | pass |
| INT8 | 0.3649 | -0.0058 | -0.0091 | 0.6676 | 0.4002 | 0.93541（不卡） | 6.261e-02 / 4.541e-01 | 5.76 / 5.47 / 9.73 | 173.5 | 52.0 MiB | pass |

参照：`.pt` 在 test split（435 张）上 mask mAP50-95 = **0.3740**（`runs/segment/baseline15/eval_test.json`）。同场量的 PyTorch GPU 单线程基线 mean 13.90 ms / p95 15.89 / 72.0 FPS。

**两列 Δ 的基准不一样，别混着看**：FP32 档的判据基准是 `.pt`（0.3740），FP16 / INT8 的判据基准是本轮 FP32 档（0.3707）——端侧真正在乎的是「相对已经上了车的部署基线掉了多少」，而 FP32 engine 就是 TRT 侧的基线；
`.pt` 那一列是给你看整条链路（导出 + 量化）加起来一共掉了多少。

**verdict = pass**，三档 gate 全过。按判据基准算的掉点：FP32 -0.0033（对 `.pt`）、FP16 -0.0005、INT8 -0.0058（都对 FP32 档）；对 `.pt` 算则分别是 -0.0033 / -0.0038 / -0.0091。INT8 的判据是 ≤0.03，实测只用掉约 19%。

- **INT8 用 12.55 MiB（5.2× 压缩）换 0.0058 mAP（对 FP32 档；对 `.pt` 是 0.0091）**，端侧这一档划算。
- FP16 几乎不掉点（-0.0005，和 FP32 的 -0.0033 在同一噪声量级），显存够又要稳就选 FP16。
- 相对 PyTorch GPU 单线程：FP16 2.07×、INT8 2.41×（同轮内比较，见 §8.5）。

### 8.3 INT8 构建挂了四次，根因是两层

报错信息是误导性的，所以写详细一点。

**第一层：`IInt8Calibrator.get_algorithm()` 是纯虚函数，Python 绑定不给默认实现。**
不重写它，TRT 每取一次校准算法就抛

```
[ERROR] Exception caught in get_algorithm(): Tried to call pure virtual function "IInt8Calibrator::get_algorithm"
```

builder 拿不到算法，最后报出来的却是另一句：

```
[TRT] [E] IBuilder::buildSerializedNetwork: Error Code 10: Internal Error
          (Could not find any implementation for node /model.22/proto/cv3/Conv + PWN(Sigmoid, Mul))
```

**这行指向 mask proto 分支，跟校准器一点关系都没有。** 照着它调了三轮精度都是白费。
修法：`_Calib.get_algorithm()` 返回 `trt.CalibrationAlgoType.<枚举>`，枚举名走
`CALIB_ALGOS = {'minmax': 'MINMAX_CALIBRATION', 'entropy': 'ENTROPY_CALIBRATION_2'}`，
基类 `IInt8Calibrator`。本机 `entropy` 仍然建不出来（同一条 Error Code 10），
`minmax` 能过，所以默认 minmax（本轮 INT8 用的就是它，校准 0 批 × 8 张，
缓存命中 True）。

**第二层：光有校准器不够，检测头必须显式留精度例外。** 照抄 ultralytics 8.4.183
`utils/export/engine.py:433-465` 的配方，实现为 `constrain_head_precisions()`：

- 头 = 层名以 `/model.{最大索引}/` 开头的那些（本模型是 `/model.22/`，命中 **32 层**）；
- 头里名字含 `sigmoid` 的 ACTIVATION → 钉 **FP32**（sigmoid 在 INT8 下动态范围被压平，置信度整体偏）；
- 头里以 `.2/Conv` 结尾、或名字含 `/dfl/` 的 CONVOLUTION → 钉 **FP16**（DFL 的 softmax 回归和
  输出卷积对量化最敏感）；
- 逐输出 `set_output_type()` + 打开 **`OBEY_PRECISION_CONSTRAINTS`**。

失败路线留档（都真跑过）：

| 尝试 | 结果 | 结论 |
| --- | --- | --- |
| workspace 提到 6 GB | 同一条 Error Code 10 | 不是 workspace 不够，别往这个方向调 |
| 只开 `INT8 + FP16(fallback)`、不钉头 | 同上失败 | fallback 的意思是「没有 INT8 tactic 时可以退」，但那个融合节点连退路都选不出来，必须显式指定精度 |
| `PREFER_PRECISION_CONSTRAINTS` + 钉 `proto/cv3` 三层 FP16 | 命中 3 层，仍失败 | PREFER 是「尽量」，TRT 可以不采纳；要 OBEY。而且只钉 3 层不够，得整个头 |
| `--int8-calib entropy` | 构建失败 | 本机 SM89 + TRT 10.16 上熵校准建不出 engine，minmax 可以 |

`--int8-pin`（PREFER 那条）留在代码里当失败对照，与 `--int8-head` 互斥：同时开 TRT 行为未定义，
而且两者是同一个问题的两种解法，混用了掉点数没法归因。

### 8.4 FP32 档 rel_l2 3.3e-3 不是链路错，是 TF32

第一轮压测 verdict=fail，挂在 FP32 档 out1 rel_l2 3.275e-03 > 1e-3。
W3a 里 ONNX Runtime 对 PyTorch 是 2.4e-7，差四个数量级，看着像前处理或输出解码写歪了。
实际原因：**TensorRT 默认打开 `BuilderFlag.TF32`**，所谓 FP32 档跑的是 TF32（10-bit 尾数），
rel_l2 停在 1e-3 量级就是这个格式的机器精度，跟链路无关。

加了 `build_trt.py --no-tf32`（`config.clear_flag(trt.BuilderFlag.TF32)`）另建一档真 FP32 验证：

| | TF32 开（默认） | TF32 关（`--no-tf32`） |
| --- | --- | --- |
| 层数 | 253 | 274 |
| engine | 65.25 MiB | 65.33 MiB |
| 激活显存 | 39.84 MiB | 41.73 MiB |
| 构建 | 94.0 s | 96.7 s |
| out0 rel_l2 | 8.342e-04 | 2.500e-07 |
| out1 rel_l2 | 3.275e-03 | 1.290e-06 |
| worst cosine | 0.99999465 | 1.00000000 |
| mean 延迟 | 9.51 ms | 10.71 ms |
| mask mAP50-95 | 0.3707 | 0.3700 |

关掉 TF32 后 rel_l2 回到 2.5e-07 / 1.3e-06，和 W3a 的 ORT 口径对齐，**证明前处理与输出解码没问题**；代价是 13% 延迟和多 21 层（少了 TF32 可用的 tactic，融合方式变了）。端到端 mAP 两边差 0.0006，在噪声里。这一档 verdict 也是 pass（判据 rel_l2 ≤ 1e-5）。

所以逐张量判据按数值格式重定了（`DEFAULT_THRESHOLDS`）：FP32/FP16 的 `max_rel_l2` 从 1e-3 放宽到
**1e-2**，cosine 仍卡 0.9999；INT8 不做逐张量卡线（动态范围量化后逐元素差没有物理意义，
本轮 out1 rel_l2 就到 4.54e-01、cosine 0.935），只看端到端 mAP ≤ 0.03。
**最终判据始终是端到端 mAP**；逐张量那层是用来定位「错在链路哪一段」的，不是用来卡发布的。
要卡 1e-5，就用 `--no-tf32` 那一档。

### 8.5 延迟只能同一轮内比

同一批 engine 连跑两轮，数字差得不小：

| 档 | 第一轮 mean / p95 / FPS | 第二轮 mean / p95 / FPS | 差 |
| --- | --- | --- | --- |
| FP32 | 9.00 / 11.40 / 111.1 | 9.51 / 11.35 / 105.2 | mean +6% |
| FP16 | 4.81 / 7.59 / 207.7 | 6.73 / 10.66 / 148.6 | mean +40% |
| INT8 | 4.74 / 8.02 / 211.1 | 5.76 / 9.73 / 173.5 | mean +22% |

FP16 两轮差 40%。笔记本 GPU（RTX 4060 Laptop 8 GB）的时钟与温度状态主导了这个差异，
30 张图 + 3 轮 warmup 的样本量压不住它。

能用的结论只有**同轮内的相对关系**，两轮一致：延迟 FP32 > FP16 ≳ INT8。
INT8 与 FP16 的延迟差在本机基本被前/后处理吃掉了（mean 5.76 vs 6.73 ms），
**INT8 在本机的收益是体积和显存，不是速度**。跨轮引用延迟数字等于编数：报告里落的是第二轮
（`runs/logs/trt_bench_baseline15.json`），第一轮日志留档在 `runs/logs/_bench15.log`。

### 8.6 构建本身不确定

同一个 ONNX、同一份校准缓存、同一条命令，INT8 档两次构建结果不同：单档跑 166 层 / 12.53 MiB / 159.3 s，三档连着跑 168 层 / 12.55 MiB / 152.0 s。
TRT 选 tactic 靠现场计时，受当时 GPU 状态影响，层融合结果跟着变。
**所以层数和体积只能当区间看，要比对就比 `engine_sha256`**（本轮三档的 sha 都落在构建报告里）。

### 8.7 三类告警是良性的

- `buildCaskGraphShaderImpl: CASK failed to generate a shader. This may be a bug.` —— 几十条，
  构建照样成功。CASK 是 TRT 的 kernel 生成后端之一，某个 tactic 生成失败就换一个。
- `Missing scale and zero-point for tensor /model.22/Range_*_output_0, expect fall back to non-int8` ——
  这些是检测头里 `Range` / `ShapeShuffle` 产生的索引张量（整数坐标，本来就不该量化），
  TRT 自己说明会回退到非 INT8 实现，正是想要的行为。
- `DeprecationWarning: ... Superseded by explicit quantization.` —— 隐式量化（calibrator）在 TRT 10
  已标记废弃，TRT 11 会移除 `IInt8Calibrator`；`make_calibrator` 已经为这种情况准备了直指
  QDQ / FP16 的报错，不至于到那时只拿到一句 AttributeError。

### 8.8 与 ultralytics 原生导出交叉对照

`yolo export format=engine int8=True` 自己导出的 `runs/segment/baseline15/weights/best.engine` 是
13,331,868 B（12.71 MiB）。本轮自建的 INT8 是 12.55 MiB（13156580 B 落盘），同一量级、略小
（头部例外的钉法与它不完全一样）。这个对照的意义是**证明自建链路没跑偏**：差出一个数量级
就说明头配方或校准集口径有问题。

### 8.9 复现命令

```powershell
conda activate induvision
# 1) 校准清单：500 张，从 train split 等间距抽，入库（configs/calib_sample.json）
python tools/make_calib_list.py --data out/yolo_mvtec15/data.yaml --split train --n 500
# 2) 三档 engine：INT8 = minmax 校准 + ultralytics 头配方 + OBEY_PRECISION_CONSTRAINTS
python tools/build_trt.py --prec fp32 fp16 int8
# 3) 三层压测：逐张量对齐 16 张 / 端到端 mAP 全 test 435 张 / 延迟 30 张 + PyTorch 同场对照
python tools/bench_trt.py --val --markdown
# 4) 真 FP32 对照档（验 §8.4 的 TF32 结论）
python tools/build_trt.py --name baseline15_notf32 --prec fp32 --no-tf32
python tools/bench_trt.py --name baseline15_notf32 --precs fp32 --val --max-rel-l2 1e-5
```

落档：`runs/logs/trt_build_baseline15.json`（三档构建）、`trt_bench_baseline15.json`（三层压测）、
`trt_build_baseline15_notf32.json` / `trt_bench_baseline15_notf32.json`（TF32 对照）。都不入库。

### 8.10 W4 到哪一步了，以及本机还缺什么

W4 拆两步，**① Python 侧流水线已完成**：pinned memory 令牌池零拷贝 + 异步 H2D/D2H、
解码 / 推理 / 后处理三线程流水线 + 环形队列、NMS 与掩码解码自己写一份并和 ultralytics 对拍。
用的是 §4.0 那份 30 张固定清单，所以和 §8.2 的单帧延迟可直接对照。实测：

- A 组（含读盘，真实口径）：FP32 1.561× / FP16 1.616× / INT8 1.563×，INT8 **51.77 FPS**；
- B 组（preload，只量推理链路上限）：FP32 1.258× / FP16 1.418× / INT8 1.413×，INT8 **330.05 FPS**；
- 对拍：match_rate **1.0000** / 框 IoU **1.0000** / 掩码 IoU 0.9100~0.9158，三档各 10/10 条判据 pass。

数字、口径、掩码解码那个 bug 的全过程和读数陷阱都在 [pipeline.md](pipeline.md)，这里不重复。

**② C++ 版还没做**，卡在本机工具链（写这段时，2026-10-22）：`nvcc` / `cmake` / `cl`(MSVC) / `ninja` / `msbuild`
全部 not found，也没有 CUDA Toolkit 与 Visual Studio 安装目录（TRT 是 pip wheel 装的，
只带运行期 `.dll`，不带头文件与 `lib`）。要做 C++ 版得先装 VS Build Tools + CUDA Toolkit（数 GB）。

> **2026-10-23 11:35 复核**：`cmake` 4.4.4 / `ninja` 1.13.2 已装好，TRT v10.16 的 14 个头已拿到
> （稀疏克隆到仓库外的 `_vendor\`，不入库），`cl` / `link` / `dumpbin` / `nvcc` 仍缺。
> 并且实测 TRT 头对 CUDA **只有 1 处 `#include`**（`NvInferRuntimeBase.h:24`），
> **CUDA Toolkit 那 4 GB 可以省**，换成 conda `cuda-cudart-dev`（win-64 13.4.92，22 KB metapackage，
> 实体包大小未验）；pip 的 `nvidia-cuda-runtime-cu13` 实测是 **0.0.1 空壳 sdist（0.00 MB）**，此路不通。
> 缺口清单、安装顺序与验收判据见 [cpp_infer.md](cpp_infer.md) §1 / §2。

② 的范围：`TensorRT C++ API`（engine 反序列化 → CUDA 流 → `cudaWarpAffine` 前处理 → 推理 →
后处理）+ CMake 工程 + 500 张端到端 P50/P95 压测。等工具链装好再做，不预写命令 ——
① 已经把语义和判据钉死了（同一份清单、同一套阈值、同一批对拍指标），C++ 版只要能复现
同一组数就算对齐，不需要重新设计口径。
