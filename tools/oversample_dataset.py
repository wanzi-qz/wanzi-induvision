#!/usr/bin/env python
"""B3：过采样数据集落盘 —— 把稀有类的**原图整张复制**，不造一个新像素。

为什么要有这个工具
------------------
``docs/baseline.md`` §3 里 B1（Copy-Paste 合成）和 B3（过采样）是最有信息量的一对：
如果"把旧图复制几遍"就能拿到和"合成新像素"差不多的收益，那 Copy-Paste 那套
工程复杂度就不值得。配额逻辑早就在 ``induvision.balance.plan_oversample`` 里，
缺的只是把它落成一个能直接喂给 ultralytics 的数据集目录，这就是本脚本干的活。

与 B1 的对照关系（必须严格单变量，否则又是一次 B4 式的无效对比）
--------------------------------------------------------------
* ``val`` / ``test`` 逐文件原样链接，一个像素都不动 —— 验证集必须和 B0/B1 完全一致；
* ``train`` 只增加"整图复制"，副本的图片和标签**都硬链接自源文件**，
  所以标签是逐字节相同的，不存在"反归一化再归一化"的多边形往返误差；
* 背景图（MVTec 的 good）不参与复制 —— 过采样针对的是稀有缺陷类，不是负样本；
* 复制倍数由 ``plan_oversample`` 的 ``cap_factor`` 封顶，避免同一张图被复制几十遍
  反而加剧过拟合。

产物
----
``images/{train,val,test}/``、``labels/{train,val,test}/``  默认硬链接
``data.yaml``               指向本目录，类别表与源一致
``manifest.csv``            继承源 manifest 的 schema，追加副本行（category/defect_type 跟随源图）
``dataset_stats.json``      继承源统计，``splits`` 按含副本重算，``env`` 指纹保留
``oversample_report.json``  配额、每类倍数、校验结果、env 指纹、耗时

用法
----
    python tools/oversample_dataset.py --src out/yolo_bottle_tb \
        --out out/yolo_bottle_tb_os --target-per-class 100

    # 只出计划不写盘
    python tools/oversample_dataset.py --dry-run

注意：数据侧的一切操作都要用**同一个** Python 环境跑（见 docs/baseline.md §4 第 7 条）。
本脚本会拿当前环境指纹和源数据集 ``dataset_stats.json`` 里的 ``env`` 对一遍，
不一致就报错，除非显式 ``--allow-env-mismatch``。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from induvision.balance import ClassStat, instance_histogram, plan_oversample  # noqa: E402
from induvision.mvtec import _place_file, environment_fingerprint  # noqa: E402
from induvision.yolo_dataset import (  # noqa: E402
    SPLITS,
    YoloDataset,
    YoloImage,
    dump_stats_json,
    read_dataset,
    read_manifest,
)

__all__ = [
    "COPY_PREFIX",
    "multiplicities",
    "copy_plan",
    "copy_name",
    "purge_stale",
    "check_env",
    "build_dataset",
    "main",
]

#: 副本文件名前缀。清理孤儿时只认这个前缀，绝不动源文件链接过来的原名文件。
COPY_PREFIX = "os__"

ENV_KEYS = ("python", "numpy", "cv2")
IMG_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp")


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for blk in iter(lambda: fh.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def multiplicities(hist: Sequence[ClassStat], quota: Dict[int, int]) -> Dict[int, int]:
    """把 ``plan_oversample`` 给的"额外复制图数"换算回每类的复制倍数。

    ``plan_oversample`` 返回的是 ``倍数 × 该类图数``，这里除回去，
    除不尽就说明配额和直方图对不上，直接报错而不是悄悄取整。
    """
    out: Dict[int, int] = {}
    for st in hist:
        n = int(quota.get(st.class_id, 0))
        if n <= 0:
            out[st.class_id] = 0
            continue
        if st.images <= 0 or n % st.images:
            raise ValueError(
                f"类 {st.name} 的配额 {n} 不是其图数 {st.images} 的整数倍，无法换算复制倍数"
            )
        out[st.class_id] = n // st.images
    return out


def copy_name(stem: str, suffix: str, r: int) -> str:
    """第 r 份副本的文件名：``os__<r>__<源 stem><后缀>``，前缀便于识别与清理。"""
    return f"{COPY_PREFIX}{r}__{stem}{suffix}"


def copy_plan(ds: YoloDataset, mult: Dict[int, int], split: str = "train") -> List[Tuple[YoloImage, int]]:
    """每张图要复制几份。背景图恒为 0；一图含多类时取各类倍数的最大值。"""
    plan: List[Tuple[YoloImage, int]] = []
    for im in ds.split(split):
        if im.is_background:
            continue
        m = max((int(mult.get(c, 0)) for c in im.class_ids), default=0)
        if m > 0:
            plan.append((im, m))
    return plan


def purge_stale(out: Path, keep: Sequence[str]) -> int:
    """删掉上一轮遗留、这一轮计划里没有的副本（孤儿图会混进训练集）。

    只碰 ``out/{images,labels}/<split>/`` 下、文件名以 ``COPY_PREFIX`` 开头的文件；
    解析后的路径必须仍在 ``out`` 里面，否则拒绝删除。
    """
    keep_set = set(keep)
    base = out.resolve()
    n = 0
    for sp in SPLITS:
        for sub in ("images", "labels"):
            d = out / sub / sp
            if not d.is_dir():
                continue
            for p in sorted(d.iterdir()):
                if not p.is_file() or not p.name.startswith(COPY_PREFIX) or p.name in keep_set:
                    continue
                if not p.resolve().is_relative_to(base):
                    raise RuntimeError(f"拒绝删除，路径不在输出目录内：{p}")
                p.unlink()
                n += 1
    return n


def check_env(src_env: Dict[str, object], cur_env: Dict[str, str]) -> List[str]:
    """比对源数据集与当前环境的指纹，返回不一致的项（空列表 = 一致）。"""
    diff: List[str] = []
    for k in ENV_KEYS:
        a, b = str(src_env.get(k, "?")), str(cur_env.get(k, "?"))
        if a != b:
            diff.append(f"{k}: 源={a} 当前={b}")
    return diff


def _split_stats(ds: YoloDataset) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {}
    for sp in SPLITS:
        ims = ds.split(sp)
        out[sp] = {
            "images": len(ims),
            "background_images": sum(1 for im in ims if im.is_background),
            "defect_images": sum(1 for im in ims if not im.is_background),
            "instances": sum(len(im.instances) for im in ims),
        }
    return out


def _write_manifest(src: Path, out: Path, rows: List[dict]) -> bool:
    """继承源 manifest 的表头写一份新的，副本行沿用源图的 category / defect_type。"""
    sm = src / "manifest.csv"
    if not sm.is_file():
        return False
    with sm.open("r", encoding="utf-8", newline="") as fh:
        rd = csv.DictReader(fh)
        fields = list(rd.fieldnames or [])
        old = list(rd)
    if not fields:
        return False
    with (out / "manifest.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(old)
        w.writerows(rows)
    return True

def _read_src_env(src: Path) -> Dict[str, object]:
    for fname in ("dataset_stats.json", "synth_report.json"):
        p = src / fname
        if p.is_file():
            try:
                env = json.loads(p.read_text(encoding="utf-8")).get("env")
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(env, dict):
                return env
    return {}


def build_dataset(
    src: Path,
    out: Path,
    *,
    target_per_class: Optional[int] = None,
    cap_factor: float = 2.0,
    link: str = "hardlink",
    dry_run: bool = False,
    allow_env_mismatch: bool = False,
    verify: bool = True,
) -> dict:
    """按计划把 ``src`` 落成一个过采样数据集 ``out``，返回可序列化的报告。"""
    t0 = time.time()
    src = Path(src)
    out = Path(out)
    if not (src / "data.yaml").is_file():
        raise FileNotFoundError(f"源数据集缺 data.yaml：{src}")
    if src.resolve() == out.resolve():
        raise ValueError("--out 不能和 --src 是同一个目录，那会就地污染源数据集")

    ds = read_dataset(src)
    if ds.warnings:
        for w in ds.warnings:
            print(f"[warn ] 源数据集：{w}")
    hist = instance_histogram(ds, "train")
    quota = plan_oversample(hist, target=target_per_class, cap_factor=cap_factor)
    mult = multiplicities(hist, quota)
    plan = copy_plan(ds, mult)
    before = _split_stats(ds)

    inst_before = {st.name: st.instances for st in hist}
    inst_after = dict(inst_before)
    for im, m in plan:
        per_cls: Dict[int, int] = {}
        for inst in im.instances:
            per_cls[inst.class_id] = per_cls.get(inst.class_id, 0) + 1
        for cid, c in per_cls.items():
            inst_after[ds.class_name(cid)] += m * c

    src_env = _read_src_env(src)
    cur_env = environment_fingerprint()
    env_diff = check_env(src_env, cur_env) if src_env else []
    if env_diff and not allow_env_mismatch:
        raise RuntimeError(
            "环境指纹与源数据集不一致，转换/合成必须用同一个 Python 环境"
            "（docs/baseline.md §4 第 7 条）：" + "；".join(env_diff)
            + "。确认无害再传 --allow-env-mismatch"
        )

    n_copies = sum(m for _, m in plan)
    print(f"[src  ] {src}  train {before['train']['images']} 图 / {before['train']['instances']} 实例")
    print(f"{'class':<14}{'images':>8}{'inst':>7}{'x':>4}{'inst_after':>12}")
    for st in hist:
        print(f"{st.name:<14}{st.images:>8}{st.instances:>7}{mult.get(st.class_id, 0):>4}"
              f"{inst_after.get(st.name, st.instances):>12}")
    print(f"[plan ] 复制 {len(plan)} 张源图，共 {n_copies} 份副本 -> "
          f"train {before['train']['images'] + n_copies} 图 / "
          f"{sum(inst_after.values())} 实例（val/test 不动）")
    if env_diff:
        print(f"[warn ] 环境指纹不一致但已放行：{'；'.join(env_diff)}")

    report: dict = {
        "source": str(src),
        "out": str(out),
        "dry_run": bool(dry_run),
        "params": {"target_per_class": target_per_class, "cap_factor": cap_factor},
        "link_mode": link,
        "quota_extra_images": {st.name: int(quota.get(st.class_id, 0)) for st in hist},
        "multiplicity": {st.name: int(mult.get(st.class_id, 0)) for st in hist},
        "instances_before": inst_before,
        "instances_after": inst_after,
        "splits_before": before,
        "copies": {"images": n_copies, "source_images": len(plan)},
        "env": cur_env,
        "env_source": src_env,
        "env_mismatch": env_diff,
    }
    if dry_run:
        report["seconds"] = round(time.time() - t0, 2)
        print("[dry  ] 只出计划，没有写盘")
        return report

    # ---- 落盘 ----
    for sp in SPLITS:
        (out / "images" / sp).mkdir(parents=True, exist_ok=True)
        (out / "labels" / sp).mkdir(parents=True, exist_ok=True)

    link_used: Dict[str, int] = {}
    n_linked = 0
    for im in ds.images:
        mode = _place_file(im.path, out / "images" / im.split / im.path.name, link)
        link_used[mode] = link_used.get(mode, 0) + 1
        n_linked += 1
        if im.label_path.is_file():
            _place_file(im.label_path, out / "labels" / im.split / f"{im.path.stem}.txt", link)
    print(f"[link ] 源文件 {n_linked} 张 -> {out}（{link}，{link_used}）")

    src_manifest = read_manifest(src)
    pairs: List[Tuple[Path, Path]] = []
    man_rows: List[dict] = []
    keep: List[str] = []
    for im, m in plan:
        for r in range(1, m + 1):
            name = copy_name(im.path.stem, im.path.suffix, r)
            dst_img = out / "images" / im.split / name
            _place_file(im.path, dst_img, link)
            pairs.append((dst_img, im.path))
            keep.append(name)
            if im.label_path.is_file():
                lname = f"{Path(name).stem}.txt"
                dst_lbl = out / "labels" / im.split / lname
                _place_file(im.label_path, dst_lbl, link)
                pairs.append((dst_lbl, im.label_path))
                keep.append(lname)
            row = dict(src_manifest.get(im.path.name, {}))
            row["out_name"] = name
            row["split"] = im.split
            row.setdefault("category", im.category or "")
            row.setdefault("defect_type", im.defect_type or "")
            man_rows.append(row)

    purged = purge_stale(out, keep)
    if purged:
        print(f"[clean] 清掉上一轮遗留的 {purged} 个孤儿副本")

    names = {i: ds.class_name(i) for i in range(len(ds.names))}
    (out / "data.yaml").write_text(
        "path: %s\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n%s"
        % (out.resolve().as_posix(), "".join(f"  {k}: {v}\n" for k, v in sorted(names.items()))),
        encoding="utf-8",
    )
    if _write_manifest(src, out, man_rows):
        print(f"[write] manifest -> {out / 'manifest.csv'}（副本 {len(man_rows)} 行）")

    if verify:
        report["verify"] = _verify(src, out, ds, before, plan, pairs)
        print(f"[check] 副本 sha256 全部匹配 {report['verify']['sha256_checked']} 个；"
              f"val/test 与源一致 = {report['verify']['val_test_identical']}")

    sp_after = report.get("verify", {}).get("splits_after") or _split_stats(read_dataset(out))
    stats_path = src / "dataset_stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.is_file() else {}
    stats["splits"] = sp_after
    stats["splits_counting"] = (
        "含过采样副本；categories / defect_area_ratio / polygon 等仍按源数据集的唯一图统计，"
        "没有随复制放大"
    )
    stats["oversample"] = {
        "source": str(src),
        "target_per_class": target_per_class,
        "cap_factor": cap_factor,
        "multiplicity": report["multiplicity"],
        "copies": report["copies"],
        "instances_before": inst_before,
        "instances_after": inst_after,
    }
    stats["env"] = cur_env
    dump_stats_json(out / "dataset_stats.json", stats)

    report["splits_after"] = sp_after
    report["purged_stale"] = purged
    report["link_used"] = link_used
    report["seconds"] = round(time.time() - t0, 2)
    (out / "oversample_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[write] report   -> {out / 'oversample_report.json'}")
    print(f"[done ] {report['seconds']}s")
    print(f"\n下一步：python tools/train_seg.py --data {out.as_posix()}/data.yaml "
          f"--model weights/yolov8s-seg.pt --name b3_oversample --epochs 100")
    return report


def _verify(
    src: Path,
    out: Path,
    ds_src: YoloDataset,
    before: Dict[str, Dict[str, int]],
    plan: Sequence[Tuple[YoloImage, int]],
    pairs: Sequence[Tuple[Path, Path]],
) -> dict:
    """落盘后自查：val/test 必须没被动过，副本必须与源逐字节相同。"""
    bad = [str(a) for a, b in pairs if _sha256(a) != _sha256(b)]
    ds_out = read_dataset(out)
    after = _split_stats(ds_out)

    identical = True
    for sp in ("val", "test"):
        if after[sp] != before[sp]:
            identical = False
    src_names = {sp: sorted(im.path.name for im in ds_src.split(sp)) for sp in ("val", "test")}
    out_names = {
        sp: sorted(im.path.name for im in ds_out.split(sp) if not im.path.name.startswith(COPY_PREFIX))
        for sp in ("val", "test")
    }
    if src_names != out_names:
        identical = False

    expect_imgs = before["train"]["images"] + sum(m for _, m in plan)
    expect_inst = before["train"]["instances"] + sum(m * len(im.instances) for im, m in plan)
    train_ok = after["train"]["images"] == expect_imgs and after["train"]["instances"] == expect_inst
    n_copy_in_train = sum(1 for im in ds_out.split("train") if im.path.name.startswith(COPY_PREFIX))
    bg_copied = [im.path.name for im in ds_out.split("train")
                 if im.path.name.startswith(COPY_PREFIX) and im.is_background]

    problems: List[str] = []
    if bad:
        problems.append(f"{len(bad)} 个副本与源文件 sha256 不一致，例如 {bad[0]}")
    if not identical:
        problems.append(f"val/test 与源数据集不一致：源={ {k: before[k] for k in ('val','test')} } 现={ {k: after[k] for k in ('val','test')} }")
    if not train_ok:
        problems.append(f"train 规模不符预期：期望 {expect_imgs} 图 / {expect_inst} 实例，"
                        f"实际 {after['train']['images']} 图 / {after['train']['instances']} 实例")
    if n_copy_in_train != sum(m for _, m in plan):
        problems.append(f"副本数不符：期望 {sum(m for _, m in plan)}，实际 {n_copy_in_train}")
    if bg_copied:
        problems.append(f"背景图被复制了（不该发生）：{bg_copied[:3]}")
    if problems:
        raise RuntimeError("过采样数据集自查未通过：\n  - " + "\n  - ".join(problems))

    return {
        "sha256_checked": len(pairs),
        "sha256_mismatch": bad,
        "val_test_identical": identical,
        "train_expected": {"images": expect_imgs, "instances": expect_inst},
        "splits_after": after,
        "copies_in_train": n_copy_in_train,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="B3 过采样数据集落盘：整张复制稀有类原图，val/test 原样不动",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--src", default=str(REPO_ROOT / "out" / "yolo_bottle_tb"))
    ap.add_argument("--out", default=str(REPO_ROOT / "out" / "yolo_bottle_tb_os"))
    ap.add_argument("--target-per-class", type=int, default=None,
                    help="每类实例数目标；不传则用最大类实例数 × --cap-factor")
    ap.add_argument("--cap-factor", type=float, default=2.0, help="复制倍数上限")
    ap.add_argument("--link", choices=("hardlink", "copy", "symlink"), default="hardlink")
    ap.add_argument("--dry-run", action="store_true", help="只出配额计划，不写盘")
    ap.add_argument("--no-verify", action="store_true", help="跳过落盘后的 sha256/规模自查（不建议）")
    ap.add_argument("--allow-env-mismatch", action="store_true",
                    help="当前 Python 环境与源数据集指纹不一致时仍然继续")
    args = ap.parse_args(argv)

    try:
        build_dataset(
            Path(args.src), Path(args.out),
            target_per_class=args.target_per_class,
            cap_factor=args.cap_factor,
            link=args.link,
            dry_run=args.dry_run,
            allow_env_mismatch=args.allow_env_mismatch,
            verify=not args.no_verify,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"[abort] {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())