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

## 2. W1b 正式基线（15 类全量）

协议先固定下来，跑完把数字填进表格，**不预先填任何值**。
§3 的两类子集只用来做趋势消融，**能写进简历的数只能来自这一节**。

### 数据已就位（2026-10-22 03:05）

`download_mvtec.py --source hf` 从 hf-mirror 拉完 15 类全量，逐文件 sha256 对过 HF LFS 的
`lfs.oid`，校验通过：**5354 张**（官方 train/good 3629 + test 467 good + 1258 缺陷，
ground_truth 掩码 1258 份）。约 12 分钟，均速 5~7 MB/s，支持断点续传，`data/` 不入库。

`convert_mvtec_to_yolo.py` 27.7 s 转完，硬链接落盘（不额外占 4.9GB），
环境指纹与 §3 三个子集一致（Python 3.12.3 / numpy 1.26.4 / cv2 4.10.0），
`--strict` 汇总时不会被判成跨环境漂移。

**这里有个必须说清楚的口径决定**：MVTec 官方 `train/` 全是正常样本，那是给无监督异常检测用的；
做**有监督**缺陷分割必须拿缺陷图训练，而缺陷图全在官方 `test/` 里。所以转换器把
官方 test 的 1258 张缺陷图 + 467 张 good 重新按 `category×defect_type` 分层切成
train/val/test = 0.70/0.15/0.15（seed=0），官方 train/good 3629 张整体作为背景负样本进 train：

| split | 图 | 其中缺陷图 | 其中背景图 | 实例 |
| --- | --- | --- | --- | --- |
| train | 4490 | 861 | 3629 | 1261 |
| val | 429 | 198 | 231 | 299 |
| test | 435 | 199 | 236 | 328 |

缺陷面积占比 mean 4.39% / p50 1.54% / p90 9.51% / max 49.32%；
`approxPolyDP` eps=0.002×周长、min_area=4 px，平均 41.9 顶点/多边形，面积保真度 0.9764。
**val 有 299 个实例，比 §3 那 16 个高一个数量级**，这一节的数字才有资格往外写。

### 命令

```powershell
conda activate induvision

# 15 类全量，约 4.9GB，逐文件 sha256 校验，支持断点续传
python tools/download_mvtec.py --source hf

python tools/convert_mvtec_to_yolo.py --src data/mvtec_raw --out out/yolo_mvtec15

python tools/train_seg.py --data out/yolo_mvtec15/data.yaml `
     --model weights/yolov8s-seg.pt --name baseline15 --epochs 100
# 默认就是 imgsz=640 / batch=8 / patience=20 / seed=0 / amp=True，不用逐个敲

# best.pt 是靠 val 早停选出来的，最终成绩要在没参与过任何决策的 test split 上重算
python tools/eval_seg.py --weights runs/segment/baseline15/weights/best.pt `
     --data out/yolo_mvtec15/data.yaml --split test --name baseline15 --markdown
```

4490 张 train 图 = 562 iter/epoch，按 §3 实测的 0.19 s/iter 推算约 112 s/epoch，
**100 epoch 约 3 小时**。这台机器驱动不稳（见 §3「队列与中断恢复」），别裸跑，用守护脚本：

```powershell
# 单跑 baseline15，崩了自动搬残档 + 重试，最多 3 次
powershell -NoProfile -ExecutionPolicy Bypass -File tools\run_train_guarded.ps1 `
    -Name baseline15 -Data out/yolo_mvtec15/data.yaml

# 或者一串挂上：补跑消融 -> baseline15 -> test 评测 -> 汇总表
powershell -NoProfile -ExecutionPolicy Bypass -File tools\run_overnight.ps1
```

要记录的指标：

| 指标 | 值 |
| --- | --- |
| 数据集规模（图 / 实例 / 类） | train 4490 图 / 1261 实例，val 429 / 299，test 435 / 328，15 类（见上） |
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
| B0 | `out/yolo_bottle_tb` | 关 | 参照点 | mask mAP50-95 **0.2453** @ ep75（95/100 早停，15.7 min） |
| B1 | `out/yolo_bottle_tb_cp` | 关 | Copy-Paste 合成对 recall 与少数类 AP 的提升 | mask mAP50-95 **0.4071** @ ep81（跑满 100，16.8 min）；对 B0 **+0.1618**（+66%）。mask mAP50 0.4736→0.6824，box mAP50-95 0.2517→0.4554，mask P/R 0.9025/0.6032 |
| B2 | `out/yolo_bottle_tb` | γ=2.0 α=0.25 | 难例加权的独立贡献 | mask mAP50-95 **0.0039** @ ep27（31/100 早停，4.6 min）；mask P/R 0.0003/0.2778，box mAP50-95 0.0021，val 的 box/cls/dfl loss 全是 nan，彻底学崩。第一轮 02:56 被 nvlddmkm TDR 打死在 ep29（`_queue.log` 记为 exit=1），03:14 补跑时同名 run 目录被 ultralytics **原地覆盖**，第一轮残档已丢 —— 现在 `runs/logs/b2_focal.log` 和 `runs/segment/b2_focal/` 里全是补跑那轮的东西。补跑这轮完整跑完，所以“学崩”与驱动无关；覆盖这个坑见「队列与中断恢复」 |
| B3 | `out/yolo_bottle_tb_os` | 关 | 过采样（稀有类原图整张复制） | 数据集已落盘并验收（2026-10-22 03:52，train **401 图 / 194 实例**，副本 66 图 + 66 标签，val/test 与源逐字节一致）；训练已排队（`tools/run_b3_after.ps1` 等 overnight 退出后自动上卡），指标待测 |
| B4 | `out/yolo_bottle_tb_nogood` | 关 | 关掉 good 背景负样本，量化它对误检率的抑制 | mask mAP50-95 **0.0109** @ ep3（23/100 早停，1.5 min）；mask P/R 0.0833/0.0714，基本没学起来。**但这组不能直接归因给背景负样本，见下面 B4 的口径问题** |
| C1 | `out/yolo_bottle_tb_cp` | γ=2.0 α=0.25 | B1+B2 叠加：两个手段是否互补 | mask mAP50-95 **0.1953** @ ep91（跑满 100，14.6 min）。比 B1 的 0.4071 **低 0.2118**，甚至低于 B0 基线 —— 在这个规模上 focal 是**负收益**，叠加也救不回来 |

B1 vs B3 是这组里最有信息量的一对：如果"把旧图复制几遍"就能拿到和"合成新像素"
差不多的收益，那 Copy-Paste 那套工程复杂度就不值得；反过来才说明合成管线有必要。

### 本轮读出来的三件事

**1. Copy-Paste 合成是真有效，而且幅度不小。** B1 对 B0：mask mAP50-95
0.2453 → 0.4071（+0.1618，相对 +66%），mask mAP50 0.4736 → 0.6824，
box mAP50-95 0.2517 → 0.4554。mask P 从 0.3885 跳到 0.9025 说明合成样本
把"缺陷长什么样"的边界撑开了，误检明显下来；R 只到 0.6032，漏检仍是短板 ——
和"缺陷样本本来就少"的直觉一致。

**2. focal loss 在这个数据规模上是负收益，别默认它一定有用。**
C1（Copy-Paste + focal）0.1953 比 B1（只有 Copy-Paste）0.4071 低了 0.2118，
比不加任何增强的 B0 基线还低。γ=2 α=0.25 会把大量 easy negative 的梯度压掉，
而这个子集 train 只有 335 图 / 97 实例，本来就没多少梯度可压，压完直接学不动。
**教训：focal 是给"目标极多、正负样本比悬殊"的检测头设计的，
小样本分割上先别开，要用也得从 γ=0.5~1.0 起扫。**

**3. B4 这组本身口径有问题，结论先挂起。** 关掉 good 背景图之后
train 只剩 **66 图 / 9 batch-iter**，23 epoch 早停 = 总共约 **207 次迭代**；
而 B0 是 335 图 / 42 iter × 95 epoch ≈ **3990 次迭代**，差了 19 倍。
所以 B4 的 0.0109 里混着"迭代步数根本不够"这个混杂因素，
**不能直接说成"背景负样本贡献了 0.23 个 mAP"**。要重做就得把步数对齐：
要么 `--epochs` 提到 ~600 让它跑够迭代，要么固定 `nbs` / 关掉 warmup 占比过高的前 3 轮。
这一条留着，等 15 类全量基线出来再决定要不要补。

### 数字怎么用

上表的绝对值**一律不能写进简历**（原因见下面"读数警告"：val 只有 29 图 / 16 实例）。
能写的是**相对关系和结论**：Copy-Paste 在这个 setup 下带来约 +66% 的相对提升、
focal 在小样本上是负收益、以及"消融必须核对迭代步数是否可比"这个方法论教训。
简历上要引绝对值，只能用 §2 的 15 类全量基线。

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

B3 的落盘 CLI 已经补上了：`tools/oversample_dataset.py`（配额仍走 `balance.plan_oversample()`）。

```powershell
# 只看配额不写盘
python tools/oversample_dataset.py --src out/yolo_bottle_tb --out out/yolo_bottle_tb_os --target-per-class 100 --dry-run
# 落盘：必须用 base 环境（W:\Anaconda\python.exe），它会拿当前环境指纹和源数据集的 env 对一遍，不一致直接报错
python tools/oversample_dataset.py --src out/yolo_bottle_tb --out out/yolo_bottle_tb_os --target-per-class 100
python tools/train_seg.py --data out/yolo_bottle_tb_os/data.yaml --model weights/yolov8s-seg.pt --name b3_oversample --epochs 100
```

`--target-per-class 100` 是对齐 B1 的计划（B1 也把每类推到 ~100 实例）：
bottle 50→100、toothbrush 47→94，各 ×1，复制 66 张缺陷原图 →
**train 335→401 图、97→194 实例**。B1 是 375 图 / 206 实例，两组并不严格等量：
**B3 的迭代步数比 B1 少约 7%、实例少约 6%**，读结果时必须把这点差异一起说出来，
不能当成等量对比（B4 就是栽在步数不可比上）。

三条铁律，由 `tests/test_oversample_cli.py` 的 20 个用例守着：

* `val` / `test` 逐文件原样硬链接，一个字节都不动 —— 评测集必须和 B0/B1/B2 完全一致；
* 副本的图片和标签**都硬链接自源文件**，标签逐字节相同，
  不存在“反归一化再归一化”的多边形往返误差；
* 背景图（good）不复制 —— 过采样针对的是稀有缺陷类，不是负样本。

落盘自查全过：132 个文件 sha256 全部匹配、val/test 与源一致、train 规模符合预期、
环境指纹与源相同（python 3.12.3 / numpy 1.26.4 / cv2 4.10.0）。除数据集目录外还落
`manifest.csv`（继承源 schema，副本行沿用源图的 category / defect_type）、
`dataset_stats.json`（`splits` 按含副本重算；`categories` / `polygon` 这类唯一图维度的统计不放大）
和 `oversample_report.json`（配额、每类倍数、校验结果、指纹、耗时）。重跑幂等：
上一轮遗留的孤儿副本（`os__` 前缀）会被清掉，源文件名一个都不碰。

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

除了超参，它还会交叉核对**数据集是哪个环境转出来的**：读各组数据集目录下的
`dataset_stats.json`（转换产物）或 `synth_report.json`（合成产物）里的 `env` 指纹，
出现两个以上不同的已知指纹就告警，`--strict` 下直接失败；只是缺指纹（老数据集）则记为
note，不判失败。这条是 W2 真踩出来的坑，见 §4 第 7 条。

### 队列与中断恢复

四组各 100 epoch，8GB 卡上一组约 16~20 分钟，连着跑要一小时以上，只能放后台。
用 `tools/run_ablation_queue.ps1`，它是**幂等**的：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools\run_ablation_queue.ps1
```

每组正常跑完时 `train_seg.py` 会落一份 `runs/segment/<name>/train_args.json`，
脚本认这个文件跳过已完成的组；上卡前还会等显存降到 3GB 以下，避免和别的任务撞车。
被中断的组不会留下 `train_args.json`，于是下次重跑自动重来 —— 这正是想要的：
**消融宁可重跑，也不要拿一个截断的 run 去和跑满的 run 比**。

这条不是纸上谈兵。2026-10-22 一晚连着中了两次，**根因不是 Windows Update，是显卡驱动 TDR**：

| 时间 | 现象 | 证据 |
| --- | --- | --- |
| 02:12:34 | 整机蓝屏重启，B1 停在 86/100（best@83，mask mAP50-95 0.3856） | bugcheck `0x00000116` = VIDEO_TDR_FAILURE，可疑驱动 `nvlddmkm.sys`，转储 `C:\WINDOWS\Minidump\102226-14937-01.dmp`；事件日志 `Kernel-Power 41` + `WER 1001/1019` |
| 02:55:40~02:56:11 | 系统没重启，B2 在 epoch 29 抛 `torch.AcceleratorError: CUDA error: unknown error` 退出 | `nvlddmkm` 事件 153/14 连报；崩前那一个 iter 卡了 **109.2 s**（正常 0.19 s），是驱动超时复位的典型形状 |

第一次的残档已挪到 `runs/_killed/b1_cp_ep86/`（weights 226MB 丢弃，可复现）。
第二次这种"进程死了、系统还活着"的情况更阴，日志里只有 CUDA 报错，
得去翻事件查看器才知道是驱动 —— 所以排查训练崩溃时
`Get-WinEvent -FilterHashtable @{LogName='System'; ProviderName='nvlddmkm'}` 要一起看。

于是队列脚本加了**自动重试**（`-MaxAttempts`，默认 3）：单组非零退出且没落
`train_args.json` 时，先把残档搬进 `runs/_killed/<name>_a<k>_ep<n>/`，
探一次 `nvidia-smi` 确认驱动恢复了，再原地重跑；每次失败的日志单独存
`<name>.attempt<k>.log`，不互相覆盖。单独跑一组用
`tools/run_train_guarded.ps1 -Name <name> -Data <data.yaml>`，同一套逻辑。

**这版重试还有个洞，10-22 当场踩到了。** `Quarantine` 原来只在“同一次调用内第 k 次
重试之前”搬残档，`attempt=1` 之前不清场。于是上一次调用（甚至是加 Quarantine 之前的
旧版脚本）留下的残档，会被下一次调用的 attempt=1 原地覆盖 —— b2_focal 第一轮 ep29
的残档就是这么没的，事后只能靠 `_queue.log` 里那行 `exit=1` 证明它存在过。
修法：每组在 `attempt=1` 开跑前先 `Quarantine <name> 0`，把遗留目录标成 `_a0_ep<n>`
收进 `runs/_killed/`；同名残档已存在时叶节点再加 `HHmmss`，避免 `Move-Item -Force`
把新残档塞进旧目录。`tools/run_ablation_queue.ps1` 已改。
`tools/run_train_guarded.ps1` 当时正在跑 baseline15，没去动一个正在执行的文件，
**等这轮结束再同步同一处修复**；在那之前如果要重挂 guarded，先手动把
`runs/segment/<name>/` 挪进 `runs/_killed/`。

想让它不随宿主 shell 一起被回收，用 WMI 建进程 —— 新进程的父级是 `WmiPrvSE.exe`，
脱离了调用方的作业对象：

```powershell
Invoke-CimMethod -ClassName Win32_Process -Method Create -Arguments @{
    CommandLine      = 'powershell.exe -NoProfile -ExecutionPolicy Bypass -File "<repo>\tools\run_ablation_queue.ps1"'
    CurrentDirectory = '<repo>'
}
```

**同一张卡上排多个长任务时，别指望 `Wait-Gpu`，要按 PID 等。**
队列里的 `Wait-Gpu` 只轮询 10 分钟（60 次 × 10 秒），等不到就往下走 ——
前面那个任务要是还得跑两小时，这层保护等于没有。B3 就是这么排的：
`tools/run_b3_after.ps1 -WaitPid <overnight 的 PID>` 盯着 `run_overnight.ps1` 主进程，
退出后再睡 90 秒让显存和文件句柄释放，然后 `run_ablation_queue.ps1 -Only b3_oversample`
上卡，跑完顺手刷新 `runs/logs/ablation_summary.md`。它自己也是 WMI 脱离宿主挂的，
日志在 `runs/logs/_b3_queue.log`，上卡前还会读一遍 `oversample_report.json` 体检
（val/test 不一致或指纹不匹配就直接 abort，宁可不出数也不出脏数）。
走队列而不是直接调 `run_train_guarded.ps1`，还有一层原因：`attempt=1` 前清场那个修复
目前只在队列里，guarded 要等 baseline15 跑完才同步。

这能挡住 shell 回收，**挡不住蓝屏**。这台机器上长任务的现实预期是
"一定会被打断，靠守护脚本自己爬起来"，而不是"想办法不被打断"。
要降低概率可以从两头下手：驱动侧（616.92 这个版本一晚 TDR 两次，可以试回滚或换 Studio 驱动）、
负载侧（关 Wallpaper Engine 之类常驻吃 GPU 的进程，别把显存逼到上限）。

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
7. **数据转换与合成必须固定同一个 Python 环境**。W2 实测：base 环境
   （cv2 4.10.0 / numpy 1.26.4）与 GPU 环境（cv2 5.0.0 / numpy 2.5.2）转出的
   788 个标签里有 **12 个差约 1 像素**，其中 val、test 各 1 个。评测集一旦不同，
   "单变量对比"当场失效，而且指标看起来完全正常、不报任何错 —— 因为
   `findContours` / `approxPolyDP` 的浮点行为随版本变。
   现在两份产物都会写 `env` 指纹（转换器写 `dataset_stats.json`，合成写
   `synth_report.json`），`summarize_ablation.py --strict` 查出两个指纹就失败。
   补救办法是**用同一个环境重跑一遍转换/合成**：同环境同参数产物逐字节相同
   （实测 789 与 870 个文件全等），可以安全覆盖。
8. **长训练放后台要用幂等队列**，别手写 `Start-Process` 一串命令，见 §3「队列与中断恢复」。