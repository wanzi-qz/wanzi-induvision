# wanzi-induvision

工业缺陷检测与端侧部署闭环。在 MVTec AD 上训练 YOLOv8-seg 做缺陷分割，
再走 ONNX → TensorRT 三档量化（FP32/FP16/INT8），最后用 TensorRT C++ API
写推理服务、做端到端压测，产出一份精度-速度-显存对照报告。

个人求职项目。硬件是 RTX 4060 Laptop 8GB / Windows 11，所以所有工程决策都围绕
"8GB 显存怎么把链路跑完整"来做。文档里的数字全部是本机实测，
还没跑到的阶段一律标 **待测**，不预填、不估。

[![CI](https://github.com/wanzi-qz/wanzi-induvision/actions/workflows/ci.yml/badge.svg)](https://github.com/wanzi-qz/wanzi-induvision/actions/workflows/ci.yml)

## 进度

| 阶段 | 内容 | 状态 |
| --- | --- | --- |
| W0 | GPU 环境搭建 + 自检脚本（版本/GPU/推理/ONNX 导出） | 完成 → [docs/environment.md](docs/environment.md) |
| W1a | MVTec 下载校验 + YOLO-seg 转换管线 + 单测 + CI | 完成 → [docs/dataset.md](docs/dataset.md) |
| W1b | 15 类全量数据 + 基线训练（mAP50-95、单类 AP、混淆矩阵） | 数据已就位（15 类 / 5354 图，sha256 全过，转换 27.7 s）；baseline15 训练跑中 → [docs/baseline.md](docs/baseline.md) §2 |
| W2 | 缺陷合成 Copy-Paste 增强 + Focal Loss + 小样本/不均衡消融 | 4/5 组已出数 → [docs/baseline.md](docs/baseline.md) §3：Copy-Paste 把 mask mAP50-95 从 0.2453 抬到 **0.4071**（+66%），focal 在这个规模上是**负收益**（C1 只有 0.1953）；B2 补跑中。绝对值不可外引（val 仅 29 图） |
| W3 | ONNX 导出 + trtexec FP32/FP16/INT8（熵校准，500 张校准集） | 未开始（工具链待装） |
| W4 | TensorRT C++ 推理服务：零拷贝 + 三线程流水线 + CMake + 压测报告 | 未开始（工具链待装） |

## 仓库结构

```
induvision/          核心库（只依赖 numpy / opencv / pyyaml，CI 能全量跑）
  mvtec.py               MVTec 目录扫描 + 分层划分 + 掩码→多边形
  yolo_dataset.py        读写本仓库产出的 YOLO-seg 数据集
  defect_bank.py         缺陷实例库：只从 train 划分抠缺陷，从源头杜绝 val/test 泄漏
  copy_paste.py          Copy-Paste 贴图与融合（poisson / alpha），标签同步重建
  balance.py             类别不均衡诊断 + 合成配额规划
  focal.py               Focal Loss：给 ultralytics 分类分支换损失，可装可卸（LIFO）
tools/
  env_check.py           环境与推理链路自检（可选 --export 验 ONNX 导出）
  download_mvtec.py      取数据：官方 zip 手动模式 / HF 镜像自动模式，逐文件 sha256
  convert_mvtec_to_yolo.py   MVTec AD → YOLOv8-seg 数据集
  synthesize_defects.py  Copy-Paste 缺陷合成，给已有数据集补齐少数类实例配额
  train_seg.py           训练入口：统一 CLI + 可选 focal 注入 + 配置留档 + 指标摘要
  eval_seg.py            在 test split 上重算成绩 + 逐类 AP 表（best.pt 是 val 选的，不能拿 val 报数）
  summarize_ablation.py  汇总各组 results.csv 出 Markdown 表，校验「单变量协议」与数据集转换环境指纹
  run_ablation_queue.ps1 消融队列（幂等 + 崩溃自动重试）：跑完的组自动跳过，中断的组搬残档后原地重跑
  run_train_guarded.ps1  单组训练守护：等显存 -> 跑 -> 崩了搬残档重试，最多 -MaxAttempts 次
  run_overnight.ps1      串联：补跑消融 -> baseline15 -> test 评测 -> 汇总表，一次挂上不用守
  run_download_mvtec.ps1 后台拉全量数据（脱离宿主 shell，日志落 runs/logs/）
  make_fake_mvtec.py     生成合成 MVTec 目录树，供 CI 端到端冒烟
tests/               228 条 pytest，纯 CPU、不联网、不需要真实数据集
docs/                environment.md 环境 / dataset.md 数据与协议
                     baseline.md 训练记录 / augmentation.md 合成增强与 focal 消融
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

# 7) 单测（228 条，装 torch 约 17 秒；没装 torch 会自动跳 2 条真模型用例）
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