# 缺陷合成与类别均衡（W2）

口径同 [baseline.md](baseline.md)：本文所有数字都是本机实测。没跑的写"待 GPU"，不预填、不估。

---

## 1. 要解决的两重不均衡

W1a 转换出来的 bottle + toothbrush 数据集（394 图 / 134 实例）实测分布：

| 现象 | 实测值 | 后果 |
| --- | --- | --- |
| 缺陷图占比低 | 93 / 394 = 23.6% | 背景负样本是缺陷图的 3.2 倍，误检压不住 |
| 目标极小 | 缺陷面积占比 p50 = 3.50%，min 0.24% | 正样本 anchor 稀疏，正/负样本比严重失衡 |
| 类间实例悬殊 | 有监督划分后 train 里 bottle 50 个实例、toothbrush 47 个 | 稀有类基本学不出来 |

对应两重不均衡、两个手段：

- **类间不均衡**（稀有类实例太少）→ Copy-Paste 缺陷合成，造新像素
- **前景/背景不均衡**（一张 640×640 的图上 TAL 给出上万个候选，正样本只有几个）→ Focal Loss 给 cls 分支加权

过采样（重复旧图）作为对照组留在消融计划里，用来回答"合成到底比重复采样强多少"。

## 2. 管线

四个模块 + 两个 CLI，核心库全部 CPU、不依赖 torch：

| 文件 | 职责 |
| --- | --- |
| `induvision/yolo_dataset.py` | 读写本仓库产出的 YOLO-seg 数据集（含归一化口径） |
| `induvision/balance.py` | 实例直方图 + 合成配额规划 + 过采样配额 |
| `induvision/defect_bank.py` | 从 train 划分抠缺陷贴片库（patch / 硬掩码 / bbox / 元信息），可缓存成 npz |
| `induvision/copy_paste.py` | 几何 warp、落点选择、拒绝采样、三种融合、光度抖动、掩码→标签多边形 |
| `induvision/focal.py` | Focal Loss 与注入 / 回滚 |
| `tools/synthesize_defects.py` | 串起来：诊断 → 建库 → 合成 → 落盘 → 审计产物 |
| `tools/train_seg.py` | 训练入口，`--focal-gamma` 开关，配置落盘到 `train_args.json` |

一条命令跑完合成：

```powershell
python tools/synthesize_defects.py --src out/yolo_bottle_tb --out out/yolo_bottle_tb_cp --cap-factor 2 --figures 6
```

产物：

| 文件 | 内容 |
| --- | --- |
| `images/{train,val,test}`、`labels/{train,val,test}` | 原图硬链接（同盘不占空间）+ 合成图实文件 |
| `data.yaml` | 与源数据集同口径的类别表 |
| `synth_manifest.csv` | 每张合成图的完整审计：底板、贴了谁、融合模式、可见比例、标签保真度、子种子 |
| `synth_report.json` | 配额计划、前后实例分布、保真度分位数、拒绝采样计数、耗时、参数快照 |
| `defect_bank.npz` | 缺陷贴片库缓存，下次跑直接复用 |
| `figures/synth__*.png` | 抽查对比图（原图 / 合成图 / 掩码 / 落点框） |

## 3. 防泄漏口径（最重要的一条）

MVTec 的有监督划分本来就是从官方 `test/` 里切出来的（见 [dataset.md](dataset.md)），
所以 val/test 的缺陷实例一旦流进训练用的贴片库，指标就是自欺欺人。三条硬规矩：

1. **缺陷库只从 train 抠。** `build_bank()` 默认只接受 train 划分，传进 val/test 直接抛
   `ValueError`；要故意做泄漏实验必须显式 `allow_eval_leak=True`。
   `synth_report.json` 里的 `bank.built_from` 永远是 `["train"]`，可事后核对。
2. **合成图只写进 train。** manifest 每行的 `split` 都是 `train`。
3. **val/test 原样搬运。** 硬链接或复制，不改一个字节。

守这条线的测试：`test_defect_bank.py`（泄漏报错 + `allow_eval_leak`）、
`test_synth_cli.py::test_cli_only_touches_train_split`（val/test 与源逐字节比对）、
CI 的 `Assert synthesis artifacts`（同样逐字节比对，ubuntu + windows 双平台）。

## 4. 合成细节

**底板**：优先同类别的 train 背景图（MVTec 的 `good` 样本），没有同类就退化为全部背景图。
实测 bottle / toothbrush 各 20 张合成图的底板全部来自本类别。

**贴片**：从缺陷库取 `patch + 硬掩码 + bbox`，做相似变换（scale 0.6–1.4、rotate ±180°、flip p=0.5），
掩码用 `INTER_NEAREST` 重采样，保证出来还是二值。

**落点**：默认 `centroid_jitter` —— 用 train 里同类缺陷的质心先验加 σ=0.08 的抖动，
让合成图的空间分布贴近真实分布；`uniform` 是全图均匀撒。
落点邻域的灰度标准差低于 `--bg-std-min` 就换点：把缺陷贴到一块纯色上，
等于白送一个"背景里有东西"的假样本。

**拒绝采样**（每次拒绝都计数进 `synth_report.json`）：

| 原因 | 含义 |
| --- | --- |
| `flat_background` | 落点邻域太平坦 |
| `overlap` | 与已贴实例的重叠超过 `--max-overlap`（默认 0） |
| `clipped` | 贴完可见比例低于 `--min-visible` |
| `exhausted` | 候选点试满 `candidates` 次仍没落下 |
| （面积门限） | warp 后硬掩码面积 < `--min-paste-area` |

`--max-overlap 0` 是刻意的：**保证一个连通域就是一个实例**，
于是"贴了几个"和"标签里有几个多边形"必须相等 —— 这条等式被单测和 CI 当断言用，
一旦哪天轮廓处理写坏了，测试立刻红。

**融合**：`poisson`（`cv2.seamlessClone`，默认）→ 贴片面积小于 `poisson_min_area`
或克隆失败时自动退回 `alpha`（羽化 1.5 px）→ `naive`（只改掩码内像素，做对照）。
再叠一层光度抖动（亮度 ±0.1、对比度 0.9–1.1、γ 变换 p=0.3），让贴片光照跟底板一致。

**标签**：贴完的掩码走与转换管线完全相同的口径 —— `RETR_EXTERNAL` 外轮廓 →
`approxPolyDP(epsilon_ratio=0.002)` → `x/(w-1)` 归一化写回。
两边口径一致很关键：`yolo_dataset._denorm` 读回来用的是 `x*(w-1)`，
差半个像素在 900×900 的图上看不出来，但会让"合成前后 IoU"这类指标凭空掉点。

## 5. 可复现性

- 每张合成图的 rng 由 `SeedSequence([seed, class_id, seq])` 派生，
  **只取决于 (类别, 序号)，与并发度无关**；`synth_manifest.csv` 的 `seed_child` 列直接写了出来。
- 波次推进：每波 `--workers` 个任务，波与波之间检查配额，打满就 break。
- 已知行为：同一波的任务共享一份 `remaining` 快照，所以 **workers 越大越可能超额合成**。
  微型数据集上实测 workers=1 → 2 张、workers=4 → 4 张（配额都是每类 2 个实例）。
  超额无害（多几个实例），欠额是 bug，`test_synth_cli.py` 两边都断言了。
- 跨进程可复现由 CI 守：同参数跑两次，图像与 manifest 必须逐字节一致；
  换并发度跑，共有的合成图也必须逐像素一致。
- 重跑会先清掉上一轮的 `synth__*` 产物。**不清会出孤儿图**：
  旧合成图不在新 manifest 里、却还在 `images/train` 下，旧标签也还在，
  ultralytics 照样拿去训练，增强集就悄悄不可复现了。这个坑实跑时踩到过
  （`out/yolo_bottle_tb_cp` 里留下 44 张孤儿图 + 44 个孤儿标签），已修并加了回归测试。

## 6. 配额语义

`plan_synthesis()` 三种模式：

| 参数 | 语义 |
| --- | --- |
| `--strategy max --cap-factor 2`（默认） | 目标 = train 里实例最多的类 × cap_factor，其余类补齐到它 |
| `--target-per-class N` | 每类至少补到 N 个实例，优先于 `--strategy` |
| `--strategy none` | 只出诊断表，不合成 |

需要多少张合成图 = `ceil(待补实例数 / 每图实例数)`，
每图实例数取 `min(该类现有的 instances_per_image, --max-per-image)`。
实际生成时按 `--max-per-image`（默认 3）贴，所以**真实图数往往比计划少**：
bottle + toothbrush 计划 70 张，实际 40 张就把配额打满了。
任务列表按计划的 2 倍铺开，配额一满就提前停，不会白跑。

## 7. Focal Loss

**只替换分类分支的 BCE**（`v8DetectionLoss.bce`，seg 任务的 cls 分支继承同一个）。
掩码分支（`BCEDiceLoss`）与 box / DFL 分支不动 —— 那些分支的目标分布不是
"海量易分负样本"，换 focal 没有理论依据。`test_focal.py` 里有一条断言专门盯这个。

数学上 `gamma=0 且 alpha=None` 时必须严格退化成标准 BCE，这是等价性单测的锚点。

实现上有两个坑，都是踩过之后写进代码的（ultralytics 8.4.183 实测）：

1. **criterion 有两种形态**：
   - v8 系（`yolov8s-seg`）：`init_criterion()` 返回 `v8SegmentationLoss`，`.bce` 就在它自己身上；
   - E2E 系（`yolo26n` 这类端到端免 NMS 模型）：返回 `E2ELoss`，**自己没有 `.bce`**，
     真正的 `.bce` 藏在 `one2many` / `one2one` 两个 `v8DetectionLoss` 里。

   所以替换必须递归找，规则是"自己有 bce 就替换、并停止继续下钻"，深度上限 2。
2. **补丁必须挂在 `on_train_start` 回调上**：`Model.train()` 内部会执行
   `trainer.get_model(cfg=self.model.yaml)` **重建**一个 nn.Module 再赋回 `self.model`。
   训练前在旧对象上打的实例补丁会被整个丢掉。
   `install_focal_loss()` 因此做两件事：注册回调 + 立刻给当前对象也打上
   （不经过 `train()` 的 val / 单测场景同样要生效）。

顺带一个易错点：`resolve_base_model()` 不能用 `hasattr(model, "init_criterion")` 判断包装器 ——
`Model.__getattr__` 会把未知属性转发给内层 nn.Module，包装器上"看起来"什么都有。
判据是 `isinstance(cand, nn.Module) and hasattr(type(cand), "init_criterion")`。

`install_focal_loss()` 返回 `FocalHandle`，`undo()` 按 LIFO 复原、顺带丢掉懒建的 `criterion` 缓存，
做 A/B 消融时不用重建模型。递归找不到 `bce` 时**主动抛 RuntimeError**：
宁可训练起不来，也不要静默训了个 BCE 还以为 focal 生效了。

训练入口已经把开关接好：

```powershell
# 基线（BCE）
python tools/train_seg.py --data out/yolo_bottle_tb_cp/data.yaml --name b0_baseline

# focal 组
python tools/train_seg.py --data out/yolo_bottle_tb_cp/data.yaml --name b2_focal `
     --focal-gamma 2.0 --focal-alpha 0.25
```

`--focal-gamma 0` 就是关闭。每次运行会把完整配置（含 focal 参数与 `git_head`）写进
`<run_dir>/train_args.json`，半年后还能对上号。别传 `--project runs`，
会拼成 `runs/segment/runs/<name>`，多一层没用的嵌套。

## 8. 参数表（synthesize_defects.py）

| 参数 | 默认 | 什么时候调 |
| --- | --- | --- |
| `--src` / `--out` | `out/yolo_bottle_tb` / `..._cp` | — |
| `--strategy` | `max` | 只想看诊断用 `none` |
| `--cap-factor` | `2.0` | 数据量撑不住降到 1.5；类间差距大升到 3 |
| `--target-per-class` | 无 | 给稀有类定死下限时用 |
| `--seed` | `0` | 消融必须固定 |
| `--blend` | `poisson` | 出接缝伪影换 `alpha`；做对照用 `naive` |
| `--scale` | `0.6 1.4` | 缺陷尺寸分布窄就收窄，别造出不合物理的样本 |
| `--rotate` | `-180 180` | 有明确方向性的缺陷（如划痕）应限制角度 |
| `--flip-p` | `0.5` | 左右不对称的缺陷设 0 |
| `--max-per-image` | `3` | 一图多缺陷的场景调高；小画布调低 |
| `--position` | `centroid_jitter` | 缺陷位置与工艺强相关时保持默认 |
| `--jitter-sigma` | `0.08` | 位置先验太紧就放大 |
| `--bg-std-min` | `0.03` | `flat_background` 报太多就调低（但别调到 0） |
| `--min-visible` | `0.75` | 画布小、缺陷大时 `clipped` 会多，可降到 0.6 |
| `--max-overlap` | `0.0` | 调高会让"连通域 = 实例"失效、标签互相污染，非必要不动 |
| `--feather` | `1.5` | `alpha` 融合的羽化半径 |
| `--min-paste-area` | `64.0` | 微小缺陷（< 8×8）会被挡掉，按需降低 |
| `--bank-cache` | `<out>/defect_bank.npz` | 多次跑同一份数据时指定固定路径复用 |
| `--workers` | `4` | 要精确控制合成张数就用 1 |
| `--figures` | `4` | 抽查图张数，0 表示不出 |
| `--link` | `hardlink` | 跨盘或权限受限时用 `copy` |
| `--dry-run` | 关 | 只看配额计划，不写盘 |

## 9. 实测

### 真实数据：bottle + toothbrush（900×900，seed=0，cap-factor=2，workers=4）

| 项 | 值 |
| --- | --- |
| 源数据集 | `out/yolo_bottle_tb`：394 图 / 134 实例 / 2 类 |
| 缺陷库 | 97 个实例，来自 66 张 train 图，`built_from=["train"]` |
| 配额计划 | bottle 50→100（补 50 实例 / 46 图）、toothbrush 47→100（补 53 / 24），合计 70 图 |
| 实际产出 | **40 张合成图 / 109 个实例**，配额打满（remaining −1 / −5） |
| 实例分布 | bottle 50 → **101**，toothbrush 47 → **105** |
| 每图贴几个 | 3 个 × 30 张、2 个 × 9 张、1 个 × 1 张 |
| 融合模式 | poisson 93 次 / alpha 16 次（小面积自动退回） |
| 可见比例 | mean 0.9986，p10 1.0000，min 0.9528 |
| 标签面积保真度 | mean **0.9675**，p10 0.9215，min 0.8154 |
| 拒绝采样 | overlap 258、flat_background 66、clipped 37、exhausted 11 |
| 划分变化 | train 335 → **375**，val 29 → 29，test 30 → 30（零改动） |
| 耗时 | 13.8 s（复用 bank 缓存，含硬链接 394 张原图） |

计划 70 张、实际 40 张就打满配额，原因见 §6：计划按现有"每图实例密度"折算，
实际生成允许一图贴 3 个。

### CI 合成树（64×48 玩具数据，seed=0，cap-factor=2，workers=1）

| 项 | 值 |
| --- | --- |
| 缺陷库 | 7 实例 / 7 图，`built_from=["train"]` |
| 产出 | 12 张合成图，bottle 5 → 10、screw 2 → 10 |
| 保真度 | mean 0.7443，p10 0.6252 |
| 可见比例 | mean 0.972，min 0.8819 |
| 拒绝采样 | clipped 242、overlap 189、exhausted 17 |
| 耗时 | 0.4 s |

保真度比真实数据低一大截是预期的：64×48 的画布把 `approxPolyDP` 的相对误差放大了，
玩具树的掩码本身就是硬画的矩形和斜线。这条数字只用来守回归（`0.5 < fid <= 1.3`），
**不要写进简历**。

### 训练指标：待 GPU

消融组是 B0 基线 / B1 +Copy-Paste / B2 +Focal / C1 两者叠加（**编号以
[baseline.md §3](baseline.md) 为准**，本篇早先把复合组写成 B3，和那边的过采样组撞号了，已改）。
协议与命令见 baseline.md §3，跑完把 mask mAP50-95、单类 AP、recall 填进那张表。
在那之前，"合成增强带来多少提升"这句话一个数字都不能写。

另外提醒一遍：这一轮的 val 只有 29 图 / 16 实例，读数只能看趋势和相对差，
绝对值不进简历。

## 10. 测试覆盖

W2 新增 112 条用例（仓库总计 148 条，`pytest -q` 约 12 s）：

| 文件 | 条数 | 守什么 |
| --- | --- | --- |
| `test_yolo_dataset.py` | 12 | 归一化往返、坏行跳过、背景图合法、manifest 关联、错误路径 |
| `test_defect_bank.py` | 11 | train-only 守卫、泄漏报错、patch/mask/bbox 一致、npz 往返、统计 |
| `test_balance.py` | 12 | 直方图、max/target/none 策略、图数折算、过采样封顶 |
| `test_copy_paste.py` | 38 | 同 seed 像素级复现、overlap=0 ⇒ 连通域 = 实例数、三种融合、参数校验 |
| `test_synth_cli.py` | 11 | 端到端产物、配额达成、防泄漏、跨并发度可复现、孤儿清理、错误路径 |
| `test_focal.py` | 25 | focal 数学、递归找 bce、工厂补丁、install/undo、真模型注入 |
| `test_tools_cli.py`（+3） | 9 | 训练入口 `--dry-run` 的配置字典 |

torch / ultralytics 是可选依赖：`test_focal.py` 模块级 `importorskip("torch")`；
两个真模型用例还要本地有 `weights/yolov8s-seg.pt` 与 `yolo26n.pt`（按许可都不入库），
CI 里自动跳过。核心库不 import torch，CI 里有一条断言专门守这条边界。