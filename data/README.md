# data/ 目录说明

**这个目录整体被 .gitignore 排除**（只保留本文件）。原因有两条：

1. MVTec AD 数据集许可为 **CC BY-NC-SA 4.0（仅限非商业研究用途）**，不允许再分发；
2. 全量 15 类约 4.9 GB，塞进 git 仓库毫无意义。

仓库里只有获取数据的脚本，数据本身请在本机执行下载。

## 期望布局

```
data/
└── mvtec_raw/                 # 官方原始数据（下载后校验 sha256）
    ├── bottle/
    │   ├── train/good/        # 正常样本，无缺陷标注
    │   ├── test/
    │   │   ├── good/          # 正常样本
    │   │   ├── broken_large/  # 缺陷样本，一类一个目录
    │   │   ├── broken_small/
    │   │   └── contamination/
    │   └── ground_truth/
    │       └── broken_large/xxx_mask.png   # 与 test 下缺陷图同名 + _mask
    └── ...（其余 14 个类别同构）
```

缺陷类型不硬编码，由 `test/` 下的子目录动态发现（toothbrush 只有 `defective` 一种，
screw 有 4 种，脚本都能处理）。

## 两种下载方式

```powershell
# A. 官方 zip（需要在网页勾选 EULA，脚本不能直连：downloads.mvtec.com 返回 403）
python tools/download_mvtec.py --source manual
#    按提示把 zip 放进 data/mvtec_raw/，再跑一次同样的命令，脚本负责
#    校验 sha256 -> 安全解压（拒绝越界条目）-> 验证目录结构

# B. HuggingFace 镜像仓库 foersben/mvtec-ad（国内可达，默认走 hf-mirror.com）
python tools/download_mvtec.py --source hf --categories bottle toothbrush
#    每个文件的 sha256 取自 HF LFS 的 lfs.oid 字段，下载后逐个校验；
#    支持断点续传与 --jobs 并发，失败重跑同一条命令即可
```

只校验不下载：

```powershell
python tools/download_mvtec.py --verify-only --categories bottle
```

校验会打印每个类别的 `train/good`、`test/good`、缺陷图数量和总量，可与官方论文
Table 1 逐类核对（bottle 实测 209 / 20 / 63，与官方一致）。

## 转换产物放哪

转换后的 YOLO-seg 数据集写到 `out/`（同样不入库）：

```powershell
python tools/convert_mvtec_to_yolo.py --src data/mvtec_raw --out out/yolo_mvtec
```

产物结构、划分协议和实测统计见 `../docs/dataset.md`。

## 磁盘占用参考

| 内容 | 大小 |
| --- | --- |
| bottle + toothbrush 两类原始数据（487 文件） | 254 MB |
| 同两类的 YOLO-seg 转换产物（394 张，硬链接落盘） | 约 0 MB（同盘硬链接不额外占空间） |
| 全量 15 类原始数据 | 约 4.9 GB |

`--link hardlink` 是默认落盘方式，只在同盘有效；跨盘会自动回退成复制。