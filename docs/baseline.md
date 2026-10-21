# 训练记录与基线

口径：本文所有数字都是本机实测。没跑完的写"中断"，没跑的写"待测"，一律不填估计值。

---

## 1. 冒烟训练：验证管线端到端（2026-10-21）

目的**不是拿指标**，是验证"转换出来的 YOLO-seg 标签能被 ultralytics 正确解析，
并跑完 训练 → 验证 → 保存 全链路"。数据管线写得再漂亮，标签格式错一个数字就是废的，
所以这一步必须先过。

### 配置

| 项 | 值 |
| --- | --- |
| 数据 | `out/yolo_bottle_tb`：bottle + toothbrush，394 图 / 134 实例 / 2 类 |
| 模型 | `yolov8s-seg.pt`（COCO 预训练） |
| 迁移 | 重建成 151 层 / 11,790,870 参数 / 40.2 GFLOPs，`Remapped 2/2 cls head rows by class name`，`Transferred 417/417` |
| imgsz / batch | 640 / 8 |
| AMP | **关闭**（AMP 自检要联网下 `yolo26n.pt`，当时没下完，直接跳过） |
| seed | 0 |
| GPU | RTX 4060 Laptop 8GB |
| 计划 / 实际 epoch | 30 / **13（中断）** |

**中断原因**：跑到 epoch 13 时，同一块 GPU 上有另一个任务在训练
（毕业论文的 LowLightVideo U-Net，4000 iters），两者显存合计 7533 / 8188 MiB，
本项目迭代速度从 3.5 it/s 掉到 1.4 it/s。为避免两边一起 OOM，
主动停掉本项目的冒烟训练，把卡让给课题任务。冒烟目的在 epoch 1 就已达成，损失为零。

### 资源占用实测（这部分有参考价值，与指标无关）

| 项 | 值 |
| --- | --- |
| 训练显存 | 4.34 GiB（batch 8 / imgsz 640 / AMP off） |
| 迭代速度 | 3.4~3.5 it/s（GPU 独占），42 iter/epoch → 约 12 s/epoch |
| 单轮验证 | 6.5 s（29 图 / 2 batch） |
| 权重体积 | `best.pt` 118.39 MB、`last.pt` 118.39 MB（含优化器状态） |
| 产物 | `results.csv`、`labels.jpg`、`train_batch0~2.jpg`、`args.yaml` 正常落盘 |

由此可推 8GB 显存的余量：AMP 关闭时 batch 8 只吃 4.34 GiB，
开 AMP 后 batch 16 应该也能放下——**这是推测，W1b 实跑时验证**。

### val 曲线（11 个记录点）

| epoch | box P | box R | box mAP50 | box mAP50-95 | mask P | mask R | mask mAP50 | mask mAP50-95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.206 | 0.381 | 0.195 | 0.0874 | 0.182 | 0.325 | 0.148 | 0.0743 |
| 2 | 0.858 | 0.071 | 0.074 | 0.0320 | 0.858 | 0.071 | 0.074 | 0.0368 |
| 3 | 0.001 | 0.381 | 0.001 | 0.0003 | 0.001 | 0.143 | 0.000 | 0.0002 |
| 4 | 0.731 | 0.071 | 0.046 | 0.0124 | 0.731 | 0.071 | 0.047 | 0.0123 |
| 5 | 0.830 | 0.071 | 0.090 | 0.0487 | 0.830 | 0.071 | 0.089 | 0.0249 |
| 6 | 0.163 | 0.071 | 0.074 | 0.0332 | 0.163 | 0.071 | 0.074 | 0.0260 |
| 7 | 0.248 | 0.071 | 0.075 | 0.0516 | 0.249 | 0.127 | 0.075 | 0.0514 |
| 8 | 0.964 | 0.071 | 0.103 | 0.0296 | 0.357 | 0.127 | 0.103 | 0.0376 |
| 9 | 0.373 | 0.294 | 0.129 | 0.0533 | 0.463 | 0.183 | 0.145 | **0.0824** |
| 10 | 0.302 | 0.238 | 0.144 | 0.0731 | 0.224 | 0.183 | 0.132 | 0.0596 |
| 11 | 0.188 | 0.183 | 0.107 | 0.0534 | 0.188 | 0.183 | 0.107 | 0.0577 |

### 结论

**通了**：标签解析、类别头迁移、训练、验证、保存全链路无报错，
说明 `convert_mvtec_to_yolo.py` 产出的多边形标签是合法的 YOLO-seg 格式。

**指标不能当性能看**，四个原因：

1. **样本量太小**：全库只有 93 张缺陷图，val 只有 29 图 / 16 实例。
   漏检 1 个实例 recall 直接掉 6.25 个点，所以曲线在 0.03~0.08 之间乱跳是数学必然，
   不是模型问题。
2. **只有 2 类**，且分布极不均衡：toothbrush 的 `defective` 是 30 图 66 实例（一图多缺陷），
   bottle 三种缺陷各 20 张左右。
3. AMP 关闭 + 中途与别的任务抢 GPU，速度不稳。
4. 13 epoch 远未收敛（原计划 30，正式基线 100）。

**唯一能读出的信号**：mask mAP50-95 最高 0.0824 出现在 epoch 9，随后回落。
这个数据量下就算跑满 30 epoch 也出不来可用曲线——所以基线必须上 15 类全量。

---

## 2. W1b 正式基线（待跑）

协议先固定下来，跑完把数字填进表格，**不预先填任何值**。

```powershell
conda activate induvision

# 15 类全量，约 4.9GB，逐文件 sha256 校验，支持断点续传
python tools/download_mvtec.py --source hf

python tools/convert_mvtec_to_yolo.py --src data/mvtec_raw --out out/yolo_mvtec15

python tools/train_seg.py --data out/yolo_mvtec15/data.yaml `
     --model weights/yolov8s-seg.pt --name baseline15 --epochs 100
# 默认就是 imgsz=640 / batch=8 / patience=20 / seed=0 / amp=True，不用逐个敲
```

要记录的指标：

| 指标 | 值 |
| --- | --- |
| 数据集规模（图 / 实例 / 类） | 待测 |
| box mAP50-95 / mAP50 | 待测 |
| mask mAP50-95 / mAP50 | 待测 |
| 单类 AP（15 类逐个） | 待测 |
| 混淆矩阵 | 待测（`runs/segment/baseline15/confusion_matrix.png`） |
| 收敛 epoch / 总时长 | 待测 |
| 峰值显存 / 迭代速度 | 待测 |

## 3. W2 消融

协议：每个实验只动一个变量，其余全部固定
（seed=0 / imgsz=640 / batch=8 / epochs=100 / patience=20 / AMP on / yolov8s-seg）。

统一走 `tools/train_seg.py`，不用裸 `yolo segment train`，两个原因：
focal 注入只能从这个入口开；它会把本次配置连同 focal 参数和 git HEAD 写进
`<run_dir>/train_args.json`，半年后还能对着复现。

**编号口径以本表为准**，`docs/augmentation.md` 引用同一套（早前那篇把"两者叠加"
写成 B3，已改为 C1，避免和这里的过采样组撞号）。

| 编号 | 数据集 | focal | 想验证什么 | 结果 |
| --- | --- | --- | --- | --- |
| B0 | `out/yolo_bottle_tb` | 关 | 参照点 | 待测 |
| B1 | `out/yolo_bottle_tb_cp` | 关 | Copy-Paste 合成对 recall 与少数类 AP 的提升 | 待测 |
| B2 | `out/yolo_bottle_tb` | γ=2.0 α=0.25 | 难例加权的独立贡献 | 待测 |
| B3 | `out/yolo_bottle_tb_os` | 关 | 过采样（稀有类原图整张复制） | 待测 |
| B4 | `out/yolo_bottle_tb_nogood` | 关 | 关掉 good 背景负样本，量化它对误检率的抑制 | 待测 |
| C1 | `out/yolo_bottle_tb_cp` | γ=2.0 α=0.25 | B1+B2 叠加：两个手段是否互补 | 待测 |

B1 vs B3 是这组里最有信息量的一对：如果"把旧图复制几遍"就能拿到和"合成新像素"
差不多的收益，那 Copy-Paste 那套工程复杂度就不值得；反过来才说明合成管线有必要。

### 命令

```powershell
conda activate induvision
nvidia-smi --query-gpu=memory.used,memory.total --format=csv   # 上卡前必看，8GB 一次只跑一个

# 单变量组 + 复合组
python tools/train_seg.py --data out/yolo_bottle_tb/data.yaml    --model weights/yolov8s-seg.pt --name b0_base     --epochs 100
python tools/train_seg.py --data out/yolo_bottle_tb_cp/data.yaml --model weights/yolov8s-seg.pt --name b1_cp       --epochs 100
python tools/train_seg.py --data out/yolo_bottle_tb/data.yaml    --model weights/yolov8s-seg.pt --name b2_focal    --focal-gamma 2.0 --focal-alpha 0.25 --epochs 100
python tools/train_seg.py --data out/yolo_bottle_tb_cp/data.yaml --model weights/yolov8s-seg.pt --name c1_cp_focal --focal-gamma 2.0 --focal-alpha 0.25 --epochs 100

# B4：重跑一遍转换，关掉 good 背景图
python tools/convert_mvtec_to_yolo.py --src data/mvtec_raw --out out/yolo_bottle_tb_nogood --no-good-train
python tools/train_seg.py --data out/yolo_bottle_tb_nogood/data.yaml --model weights/yolov8s-seg.pt --name b4_nogood --epochs 100
```

B3 还差一个落盘 CLI：配额函数 `balance.plan_oversample()` 已经写好也有单测，
但没有对应的 `tools/` 入口把它变成数据集目录，所以 B3 目前是**待实现**状态，不排进本轮。

B4 要留意一点：`--no-good-train` 只动 train，val / test 里的 good 背景图照旧保留，
所以它和 B0 的评测集是同一批图 —— 这才是干净的单变量对比，误检率的差异才归因得到这一个开关上。

### 汇总与协议自检

跑完别手抄数字，用 `tools/summarize_ablation.py`：

```powershell
python tools/summarize_ablation.py --markdown    # 直接贴进上面那张表的"结果"列
python tools/summarize_ablation.py --strict      # 各组 seed/imgsz/batch/epochs/patience/amp 不一致就非零退出
```

它从 `results.csv` 取 best epoch（默认按 mask mAP50-95），并回读每组的训练配置做一致性检查。
加 `--strict` 的意义在于：**消融最怕的不是指标难看，是"以为是单变量对比、其实 batch 都不一样"**，
这种事靠记忆一定出错，交给脚本查。

### 每组要记的数

box / mask 各自的 P、R、mAP50、mAP50-95，两个类分别的 AP，收敛 epoch，
总时长，峰值显存。来源：`runs/segment/<name>/results.csv` + 训练末尾的指标摘要。

### 读数警告（重要）

这一轮用的是 bottle + toothbrush 两类子集，**val 只有 29 图 / 16 实例**：
漏检 1 个实例 recall 直接掉 6.25 个点。所以这组消融**只能读趋势和相对差**，
绝对值不能写简历；能写进简历的数必须来自 §2 的 15 类全量基线。

## 4. 复现注意事项

1. **AMP 自检资产**：ultralytics 首次训练会联网下 `yolo26n.pt` 做 AMP 检查，
   它只在**运行命令的工作目录**（仓库根）找这个文件。仓库根已经放了一份（5.5 MB，
   不入库，`*.pt` 被 `.gitignore` 排除），克隆下来要自己补，或临时 `--no-amp`。
   AMP 开着时 yolov8s-seg @ batch 8 / 640 实测只吃 2.3~3.0 GiB，比 §1 关 AMP 的
   4.34 GiB 省一半，8GB 卡上余量很足。
2. **不要用 `pip install torch` 升级 induvision 环境**：PyPI 默认 CPU 版，
   会把 cu130 覆盖掉，`cuda.is_available()` 立刻变 False。
3. **`runs/` 不入库**。要留档的曲线图、混淆矩阵截图复制到 `docs/figures/` 再提交。
4. **别和别的 GPU 任务同时跑**：8GB 显存下两个训练任务会互相拖慢并有 OOM 风险，
   §1 的冒烟就是这么中断的。上卡前先
   `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv`
   确认没有别人的 python 进程。
5. **`--model` 要写全路径** `weights/yolov8s-seg.pt`。默认值是裸文件名 `yolov8s-seg.pt`，
   仓库根没有这个文件，ultralytics 会转头去联网下载，国内网络下会卡很久。
6. **`runs/segment/<name>/results.csv` 是唯一可信的指标来源**。控制台进度条会被
   `\r` 刷屏，日志里想按 epoch 取数得先过滤；直接读 csv 省事。