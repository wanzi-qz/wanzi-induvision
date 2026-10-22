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
    --val --val-batch 8 --bench-n 30 --markdown
```

一次跑完 **84.0 s**：逐张量对齐 8.6 s + 端到端 val 68.2 s + 延迟压测与读图约 7.2 s。
verdict **pass**。

### 4.1 第 1 层：逐张量对齐

16 张 test 图（`sample_paths` 等间距抽样，具体哪 16 张见 JSON 的 `images_used`），
PyTorch 与 ONNX Runtime 各前向一遍。
**PyTorch 侧钉在 CPU**（`--align-device cpu`，默认值）：这样两边只差"导出与算子实现"
这一个变量。拿 GPU 上的 PyTorch 去比 CPU 上的 ORT，量到的差里混着 cuDNN 卷积算法选择
的不同，说不清是谁的锅。想量那个混合差就把它改成 `0`。

| 输出 | 形状 | max_abs | mean_abs | rel_l2 | cosine |
| --- | --- | --- | --- | --- | --- |
| `output0` | [1, 51, 8400] | 1.617e-03 | 1.940e-06 | 2.400e-07 | 1.00000000 |
| `output1` | [1, 32, 160, 160] | 2.241e-05 | 5.100e-07 | 1.610e-06 | 1.00000000 |

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

拿 `.onnx` 当权重走 ultralytics 的 AutoBackend，在 test split（435 图）上重算，
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
2. **差值里混着设备差异**：参照那次 `device=0`（GPU，26.9 s），这次 ORT 只有 CPU EP
   （68.2 s）。第 1 层已经把"导出本身"单独量干净了，所以这组数字读作
   "端到端换到 ONNX 后端成绩不变"，不要读成"CPU 与 GPU 等价"。

### 4.3 第 3 层：单帧延迟

batch=1、imgsz 640、30 帧，前 3 帧 warmup 不计时（首帧要付算子初始化、显存分配、
cudnn 选算法的钱，混进去会把 P99 拉成假的高）。

| 后端 | 设备 | mean | p50 | p95 | p99 | max | FPS(mean) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ONNX Runtime | AzureExecutionProvider, CPUExecutionProvider | 98.14 ms | 97.08 | 102.25 | 118.62 | 124.84 | 10.19 |
| PyTorch | cuda:0 | 16.08 ms | 16.30 | 18.37 | 19.02 | 19.17 | 62.20 |

**这两行不是"ORT 比 PyTorch 慢 6 倍"，是"CPU 比 GPU 慢 6 倍"。** 本机装的是
onnxruntime CPU 版（刻意不装 `-gpu`，理由见 [environment.md](environment.md) §6），
providers 里没有 CUDAExecutionProvider。ONNX 在本项目里的角色是导出中间格式与
数值对齐的载体，速度这一段要等 TensorRT。

所以这组数字的正确用法有两个：给 W3 后半段的 FP32/FP16/INT8 三档当**同机基线**，
以及给 W4 的 C++ 推理服务当"Python 单线程能跑多快"的对照组。

顺带一个可读的信号：PyTorch GPU 那行 p99/p50 = 1.17，尾巴很平；ORT CPU 那行
p99 118.62 对 max 124.84，抖动主要在 CPU 侧的调度上，不在模型里。

## 5. 返回码与落档

| 返回码 | 含义 |
| --- | --- |
| 0 | 判据全过，verdict=pass |
| 2 | 输入缺失：权重 / data.yaml / onnx 不存在，或该 split 没有图 |
| 3 | 判据不达标（verdict=fail），或导出体检告警 |

JSON 里除三层结果外还落 `config`（含三条阈值）、`images_used`、`git_head`、
`env`（python / numpy / cv2 / torch / ultralytics / onnx / onnxruntime 版本 + CUDA +
GPU 型号 + ort providers）。

**`images_used` 不是装饰，两次跑对不上数时先看它。** 抽样口径是
`sample_paths(images, max(align_n, bench_n))` 再取前 `align_n` 张，所以加不加
`--bench-n` 会改变第 1 层用的是哪 16 张图。实测同一份权重、同一台机器：
带 `--bench-n 30` 那次 `output0` 的 max_abs 1.617e-03 / rel_l2 2.400e-07，
只跑第 1 层（`sample_paths(435, 16)`）那次是 2.441e-03 / 3.000e-07，
`output1` 反向从 2.241e-05 变到 2.193e-05 —— 图不一样，最坏值自然不一样，
两个数都离判据（rel_l2 ≤ 1e-3）差三个数量级。本文表格里的是
`--val --bench-n 30` 那次三层全跑的结果。

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

## 7. 测试覆盖

| 文件 | 条数 | 覆盖 |
| --- | --- | --- |
| `tests/test_deploy.py` | 35 | letterbox 逐条对齐 ultralytics（含奇数余量时上下差 1 px、pad 颜色、`scaleup` 两种口径、还原坐标用的元信息）、preprocess 的形状/dtype/值域/通道序/无 batch 维、compare_tensors 的形状前置与退化输入（全零、正交、等比缩放）、latency_stats 的分位与空输入、sample_paths 的等间距与确定性、read_split_images 的六种 data.yaml 写法、read_image_bgr 的两条兜底路径 |
| `tests/test_export_onnx.py` | 11 + 4 skip | 路径与命名推导、CLI 口径是否正确落到配置、ONNX 图的静态体检解析（动态/静态 batch 两种）、checker 失败时的收口；skip 的 4 条要 `onnx`（CI 不装，`importorskip`） |
| `tests/test_verify_onnx.py` | 37 | `_torch_device`、`_delta`、`load_reference`（缺文件 / JSON 损坏）、`check_align` 与 `check_val` 的每条判据、`_md` 三层渲染（含无参照时的 `—`、只有一个后端时跳过另一行）、CLI 的路径推导与四种错误收口、`_load_inputs` 的前处理与坏图跳过 |

全量 **353 passed / 6 skipped**（base 环境，纯 CPU、不联网、不需要真实数据集，17 s）。
CI 里那 6 条 skip = 4 条要 onnx + 原有 2 条。

## 8. 下一步（W3 后半段，待测）

TensorRT 还没装（见 [environment.md](environment.md) §6）。装完先跑
`tools/env_check.py --export` 自检，再动量化。要记的数：

| 档 | 要点 | 要记的数 |
| --- | --- | --- |
| FP32 | trtexec 直接建 engine | 体积、mAP、mean/p95 延迟、显存 |
| FP16 | 加 `--fp16` | 同上 + 相对 FP32 的掉点 |
| INT8 | 熵校准，500 张校准图 | 同上 + 校准集口径 + 判据放宽到多少 |

两点先说清楚，免得后面走弯路：

- **校准集用 `deploy.sample_paths(test 图, 500)`**：等间距抽样只取决于排序后的列表，
  换机器换时间跑都一样，而且能覆盖到分布的头尾（随机抽 500 张有可能全落在少数几类上，
  校准出来的动态范围会偏）。
- **trtexec 的 `--calib` 是校准缓存文件，不是图片目录。** 要用真实图做 INT8 校准，
  得自己实现 `IInt8EntropyCalibrator2`（C++ 或 Python）先把缓存生成出来。这一步的口径
  等 TensorRT 装完实测确认，不在这里预写命令。

每档跑完都要重走一遍第 1、2 层校验。FP16 沿用 cosine ≥ 0.9999 / rel_l2 ≤ 1e-3；
INT8 的 rel_l2 判据必然要放宽，放宽到多少、为什么放宽，实测出来再写进这一节。
（三档的精度-速度-显存对照表等实测出来再填，这里不预填数字。）