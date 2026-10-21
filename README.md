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
| W1b | 15 类全量数据 + 基线训练（mAP50-95、单类 AP、混淆矩阵） | 进行中 → [docs/baseline.md](docs/baseline.md) |
| W2 | 缺陷合成 Copy-Paste 增强 + Focal Loss + 小样本/不均衡消融 | 未开始 |
| W3 | ONNX 导出 + trtexec FP32/FP16/INT8（熵校准，500 张校准集） | 未开始（工具链待装） |
| W4 | TensorRT C++ 推理服务：零拷贝 + 三线程流水线 + CMake + 压测报告 | 未开始（工具链待装） |

## 仓库结构

```
induvision/          核心库：目录扫描、分层划分、掩码→多边形、落盘与统计
tools/
  env_check.py           环境与推理链路自检（可选 --export 验 ONNX 导出）
  download_mvtec.py      取数据：官方 zip 手动模式 / HF 镜像自动模式，逐文件 sha256
  convert_mvtec_to_yolo.py   MVTec AD → YOLOv8-seg 数据集
  make_fake_mvtec.py     生成合成 MVTec 目录树，供 CI 端到端冒烟
tests/               36 条 pytest，纯 CPU、不联网、不需要真实数据集
docs/                environment.md 环境 / dataset.md 数据与协议 / baseline.md 训练结果
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

# 3) 训练。8GB 显存：s 型号 + imgsz 640 + batch 8 起步
yolo segment train data=out/yolo_mvtec/data.yaml model=weights/yolov8s-seg.pt `
     imgsz=640 batch=8 epochs=100 patience=20 project=runs name=baseline

# 4) 单测（36 条 / 约 4 秒）
python -m pytest -q
```

不装 GPU 环境也能跑通 2 和 4：转换管线只依赖 numpy / opencv / pyyaml。
CI 就是这么验的（ubuntu + windows × py3.10/3.12）。

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
| ONNX 导出 | 成功，45.40 MB，opset 17，dynamic batch，onnx.checker 通过 |

这两行数字直接指出了 W2 要解决什么：**缺陷样本稀缺（23.6%）+ 目标极小（p50 3.5%）**，
对应 Copy-Paste 缺陷合成和 Focal Loss / 过采样。

训练结果（含 15 类全量基线）见 [docs/baseline.md](docs/baseline.md)。

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