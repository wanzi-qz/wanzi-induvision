# 数据集与转换管线（MVTec AD → YOLOv8-seg）

本文写清三件事：数据从哪来、划分协议为什么这么做、掩码怎么变成 YOLO 多边形标签。
所有统计数字都是本机在真实下载数据上跑出来的，脚本可复现。

---

## 1. MVTec AD 是什么

MVTec Anomaly Detection，工业异常检测领域最常用的公开基准，15 个产品类别
（bottle / cable / capsule / carpet / grid / hazelnut / leather / metal_nut / pill /
screw / tile / toothbrush / transistor / wood / zipper），纹理类 5 个 + 物体类 10 个，
共 5354 张高分辨率图，缺陷以像素级掩码标注。许可 **CC BY-NC-SA 4.0（非商用）**。

选它的理由很直接：公开可复现、不需要额外解释数据来源，且缺陷尺度跨度大
（本项目实测单缺陷面积占整图 0.24% ~ 27.4%），适合做小目标与类别不均衡的实验。

下载与校验见 `../data/README.md`。

## 2. 划分协议：为什么在官方 test 集内部划分

这是整个管线里最容易被质疑的一点，先说清楚。

**MVTec AD 官方 `train/` 目录只有 `good` 正常样本，一张缺陷标注都没有。**
它是为无监督异常检测设计的协议：只用正常图训练，在 test 上找异常。

但本项目做的是**有监督实例分割**（YOLOv8-seg），必须有缺陷标注才能训练，
而缺陷标注只存在于 `test/` + `ground_truth/`。所以：

```
官方 train/good  ─┐
                  ├─→ 无标注背景负样本（抑制误检）
官方 test/good   ─┘

官方 test/<defect_type> + ground_truth  ─→ 分层划分 70% / 15% / 15%
                                            train / val / test
```

- **分层键**：`category × defect_type`，保证每种缺陷在三个 split 里都有样本，
  不会出现"某类缺陷全在 val、train 一张没见过"的泄漏式偏斜；
- **随机种子固定** `seed=0`，同一份数据反复转换结果完全一致（有单测覆盖）；
- **对外报告指标时必须写明"在官方 test 集内部划分"**，否则会被误认为跑了官方
  无监督协议。本项目的 mAP 与论文里的 AUROC-PRO 不可直接比较。

`--no-good-train` 可以关掉官方 train/good 作为背景样本（默认开启）。
背景负样本的标签文件是空的 `.txt`，YOLO 会当作纯背景图参与训练。

## 3. 掩码 → 多边形

MVTec 的 ground_truth 是与原图同尺寸的二值 PNG（缺陷区 255，其余 0）。
YOLO-seg 需要的是归一化多边形顶点序列，转换步骤：

1. `cv2.threshold` 二值化 → `findContours(RETR_EXTERNAL)` 取**外轮廓**；
2. `approxPolyDP(epsilon = 0.002 × 轮廓周长)` 折线近似，压缩顶点数；
3. 面积 < `--min-area`（默认 4 px）的轮廓直接丢掉；
4. 顶点归一化到 `[0,1]`，写 `class_id x1 y1 x2 y2 ...` 一行一个实例。

两个刻意的取舍：

**用 RETR_EXTERNAL 而不是 RETR_TREE**：`contamination` 这类缺陷常带内部孔洞
（donut 形），取外轮廓会把孔洞一起填掉。代价是多边形面积比真实掩码面积大，
这个膨胀用 `area_fidelity = 多边形填充面积 / 掩码面积` 量化，
bottle + toothbrush 冒烟集实测均值 **0.9765**、p10 **0.9518**，
即最坏的 10% 也只膨胀不到 5%，对分割训练可接受。

**面积过小的缺陷图默认丢弃，而不是留成空标签**：如果一张图有缺陷但掩码面积低于
阈值，留着它就等于告诉模型"这里是背景"，属于标签噪声。默认丢弃并打印告警，
需要保留时加 `--keep-empty-defects`。丢弃数记在 `dataset_stats.json` 的
`protocol.dropped_tiny_defect_images` 里，冒烟集实测为 **0**。

## 4. 输出产物

```
out/yolo_<name>/
├── images/{train,val,test}/     # 原图，默认硬链接（同盘零占盘），失败回退复制
├── labels/{train,val,test}/     # YOLO-seg 多边形标签，背景图为空文件
├── data.yaml                    # 训练入口：path/train/val/test/names
├── dataset_stats.json           # 全量统计：划分协议、逐类逐缺陷类型计数、面积分位数、多边形保真度
└── manifest.csv                 # 每张图一行：输出名、split、类别、缺陷类型、实例数、掩码像素、源路径
```

文件名编码了来源信息，便于回溯：`bottle__test__broken_large__000.png`
= 类别 `bottle` / 官方 split `test` / 缺陷类型 `broken_large` / 原序号。

类别体系两种模式：

| 模式 | 说明 | 适用 |
| --- | --- | --- |
| `--classes per-category`（默认） | 15 个产品类别各占一类，缺陷类型不区分 | 多类别联合训练、类别间泛化实验 |
| `--classes unified` | 所有缺陷合成一个 `defect` 类 | 只关心"有无缺陷"的产线二分类落地 |

## 5. 实测统计（2026-10-21，bottle + toothbrush 冒烟集）

命令：

```powershell
python tools/convert_mvtec_to_yolo.py --src data/mvtec_raw --out out/yolo_bottle_tb
```

原始数据核对（与 MVTec 官方论文数字一致）：

| 类别 | train/good | test/good | 缺陷图 | 合计 |
| --- | --- | --- | --- | --- |
| bottle | 209 | 20 | 63（broken_large 20 / broken_small 22 / contamination 21） | 292 |
| toothbrush | 60 | 12 | 30（defective 30） | 102 |

转换结果：

| split | 图片数 | 含缺陷图 | 背景图 | 实例数 |
| --- | --- | --- | --- | --- |
| train | 335 | 66 | 269 | 97 |
| val | 29 | 13 | 16 | 16 |
| test | 30 | 14 | 16 | 21 |
| **合计** | **394** | **93** | **301** | **134** |

其他实测值：

| 指标 | 值 |
| --- | --- |
| 缺陷面积占比 mean / p50 / p90 / max | 5.85% / 3.50% / 14.87% / 27.42% |
| 缺陷面积占比 min | 0.24% |
| 多边形平均顶点数 | 46.5 |
| 多边形面积保真度 mean / p10 | 0.9765 / 0.9518 |
| 因面积过小被丢弃的缺陷图 | 0 |
| 落盘方式 | hardlink（394/394 成功，未回退复制） |
| 耗时 | 1.9 s（含扫描、划分、轮廓提取、写盘） |

从这组数字能直接读出 W2 要解决的两个问题：**缺陷图只占 23.6%（93/394）**，
背景负样本是它的 3.2 倍；**面积中位数仅 3.5%**，属于典型小目标。
这两点分别对应 Copy-Paste 缺陷合成增强和 Focal Loss / 过采样的必要性。

`toothbrush` 尤其极端：30 张缺陷图贡献 66 个实例（一图多缺陷），
但 `defective` 只有一种类型，分层划分后 val/test 各只有 4~5 张，
指标方差会很大——所以基线训练至少要在 15 类全量数据上跑，
两类的数字只能证明管线通，不能当模型性能结论。

## 6. 复现命令

```powershell
conda activate induvision

# 只扫描统计，不写盘
python tools/convert_mvtec_to_yolo.py --src data/mvtec_raw --dry-run

# 全量 15 类，统一 defect 类
python tools/convert_mvtec_to_yolo.py --src data/mvtec_raw --out out/yolo_mvtec --classes unified

# 调整划分与多边形精度
python tools/convert_mvtec_to_yolo.py --src data/mvtec_raw --out out/yolo_mvtec `
    --train-ratio 0.7 --val-ratio 0.15 --seed 0 --epsilon-ratio 0.002 --min-area 4
```

单元测试（30 条，4.5 s，不依赖 torch 也不依赖真实数据集，用合成小图跑）：

```powershell
python -m pytest -q
```

覆盖：目录扫描、包装目录识别、掩码缺失报错、分层划分的确定性与比例、
多边形归一化边界、硬链接回退、`--dry-run` 不写盘、模块不引入 torch 依赖。

## 7. 已知坑

1. **不要把官方 train/good 当训练正样本**：它没有标注，混进去只会稀释缺陷比例。
2. **Windows 硬链接要求源和目标同盘**：数据在 `data/`、输出在 `out/`，同盘才生效；
   跨盘脚本会自动回退复制并打印实际落盘方式。
3. **github.com:443 间歇不通**：下载权重/资产走 `https://gh-proxy.com/<原链接>`，
   数据集走 `hf-mirror.com`，不要直连 huggingface.co。
4. **ultralytics 首次训练会联网下 `yolo26n.pt` 做 AMP 自检**，网络不通会卡住重试。
   提前把它放进项目根目录或 `weights/` 即可跳过。
5. **MVTec 图像尺寸不统一**（700×700 到 1024×1024 不等），
   标签用归一化坐标所以不受影响，但训练时 `imgsz` 会引入不同的缩放比例，
   跨类别比较指标时要留意。