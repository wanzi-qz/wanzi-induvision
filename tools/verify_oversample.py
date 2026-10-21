#!/usr/bin/env python
"""过采样数据集落盘验收：不信任 oversample_report.json 的自述，从磁盘重算一遍再对账。

为什么需要这个
--------------
B3 这组消融的唯一变量是"train split 里稀有类被复制了几份"。只要副本落盘时手滑
——比如把背景图也复制了、或者 val/test 被顺手改了一个字节——两组就不是单变量对照，
跑出来的 mAP 差异毫无意义，而且**事后看不出来**。所以过采样必须有独立验收：
重算磁盘上的真实状态，再和 oversample_report.json 声称的对账，两边不一致就非零退出。

三条铁律（对应下面的检查项）
---------------------------
1. **副本逐字节等于源**：``os__<src>`` 的图片和标签 sha256 必须和源文件完全一致。
   过采样是"同一张图多算几次"，不是"生成新图"，任何像素级改动都改变了实验含义。
2. **val/test 一个字节都不许动**：文件名列表和内容 sha256 都要和源一致，且不含副本。
   验证/测试集一旦被污染，早停选出来的 best.pt 就不可信了。
3. **只复制带缺陷的图**：背景图（标签为空）复制多少份都不会增加正样本，
   只会稀释 defect 比例、拖慢每个 epoch，属于纯负收益。

用法
----
    python tools/verify_oversample.py --src out/yolo_bottle_tb --dst out/yolo_bottle_tb_os
    python tools/verify_oversample.py ... --json-out runs/logs/verify_b3.json --quiet
    python tools/verify_oversample.py ... --limit-sha 20   # 抽样 20 个，快速自查

退出码：0 = 全部通过；1 = 有 FAIL（CI 可直接用）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "val", "test")
IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp")
PREFIX_DEFAULT = "os__"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def list_images(root: Path, split: str) -> List[str]:
    d = root / "images" / split
    if not d.is_dir():
        return []
    return sorted(f for f in os.listdir(d) if f.lower().endswith(IMG_EXTS))


def list_labels(root: Path, split: str) -> List[str]:
    d = root / "labels" / split
    if not d.is_dir():
        return []
    return sorted(f for f in os.listdir(d) if f.endswith(".txt"))


def label_lines(root: Path, split: str, fname: str) -> List[str]:
    p = root / "labels" / split / fname
    if not p.is_file():
        return []
    return [ln for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]


def stem_of_copy(fname: str, prefix: str) -> Optional[str]:
    """副本文件名 -> 源文件名；不是副本则返回 None。

    oversample_dataset.py 的副本命名是 ``os__<序号>__<原名>``：序号用来区分
    "同一张源图被复制了第几份"，所以剥掉前缀后还要再剥掉一段纯数字的序号。
    少剥这层就会拿 ``1__bottle__...png`` 去找源文件，必然 FileNotFound。
    """
    if not fname.startswith(prefix):
        return None
    rest = fname[len(prefix):]
    head, sep, tail = rest.partition("__")
    if sep and head.isdigit():
        return tail
    return rest


class Checker:
    def __init__(self, quiet: bool = False) -> None:
        self.quiet = quiet
        self.rows: List[Dict[str, object]] = []
        self.ok = True

    def check(self, label: str, cond: bool, extra: str = "") -> bool:
        cond = bool(cond)
        self.ok = self.ok and cond
        self.rows.append({"label": label, "pass": cond, "extra": extra})
        if not self.quiet:
            print(f"{'PASS' if cond else 'FAIL'}  {label}" + (f"  {extra}" if extra else ""))
        return cond


def _sha_sample(items: Sequence[str], limit: int) -> Sequence[str]:
    if limit <= 0 or limit >= len(items):
        return items
    step = max(1, len(items) // limit)
    picked = list(items[::step])[:limit]
    return picked


def verify(src: Path, dst: Path, prefix: str = PREFIX_DEFAULT, limit_sha: int = 0,
           quiet: bool = False) -> Tuple[bool, List[Dict[str, object]]]:
    ck = Checker(quiet)

    if not src.is_dir():
        ck.check(f"源数据集存在：{src}", False)
        return ck.ok, ck.rows
    if not dst.is_dir():
        ck.check(f"过采样输出存在：{dst}", False)
        return ck.ok, ck.rows

    rep_path = dst / "oversample_report.json"
    rep: Dict[str, object] = {}
    if rep_path.is_file():
        rep = json.loads(rep_path.read_text(encoding="utf-8"))
    ck.check("oversample_report.json 存在", bool(rep), str(rep_path))

    # ---------- 1/2/3 split 逐个查 ----------
    copies_by_split: Dict[str, List[str]] = {}
    for sp in SPLITS:
        si, di = list_images(src, sp), list_images(dst, sp)
        sl, dl = list_labels(src, sp), list_labels(dst, sp)
        copies = [f for f in di if f.startswith(prefix)]
        copies_by_split[sp] = copies

        if not si and not di:
            ck.check(f"{sp}: 源与输出都没有图（跳过）", True)
            continue

        if sp == "train":
            ck.check("train 图数 = 源图数 + 副本数",
                     len(di) == len(si) + len(copies),
                     f"{len(si)}+{len(copies)}={len(di)}")
            # 注意：不能要求"标签数 == 图数"。YOLO-seg 允许背景图没有 .txt
            # （tiny fixture 就是这样，convert_mvtec_to_yolo 则写空文件），两种都合法。
            # 真正必须成立的是：副本一定带标签（副本只从缺陷图里挑），且没有孤儿标签。
            no_lbl = [f for f in copies if not (dst / "labels" / sp / (f.rsplit(".", 1)[0] + ".txt")).is_file()]
            ck.check("每张副本图都有标签文件", not no_lbl, f"{no_lbl[:3]}")
            img_stems = {f.rsplit(".", 1)[0] for f in di}
            orphan = [f for f in dl if f[:-4] not in img_stems]
            ck.check("没有孤儿标签（有 .txt 却没有对应图）", not orphan, f"{orphan[:3]}")
            ck.check("副本标签数 = 副本图数",
                     len([f for f in dl if f.startswith(prefix)]) == len(copies),
                     f"{len(copies)}")
            ck.check("源 train 文件名全部保留", set(si) <= set(di),
                     f"missing={len(set(si) - set(di))}")
            ck.check("train 至少有副本（否则等于没过采样）", len(copies) > 0, f"copies={len(copies)}")

            # 铁律 1：副本逐字节等于源
            sample = _sha_sample(copies, limit_sha)
            bad_img: List[str] = []
            bad_lbl: List[str] = []
            for f in sample:
                origin = stem_of_copy(f, prefix)
                if origin is None:
                    bad_img.append(f"{f}(命名不合前缀)")
                    continue
                p_dst_img, p_src_img = dst / "images" / sp / f, src / "images" / sp / origin
                if not p_dst_img.is_file():
                    bad_img.append(f"{f}(输出缺文件)")
                    continue
                if not p_src_img.is_file():
                    bad_img.append(f"{f}(源缺 {origin})")
                    continue
                if sha256(p_dst_img) != sha256(p_src_img):
                    bad_img.append(f)
                lf, lo = f.rsplit(".", 1)[0] + ".txt", origin.rsplit(".", 1)[0] + ".txt"
                pf, po = dst / "labels" / sp / lf, src / "labels" / sp / lo
                if not pf.is_file():
                    bad_lbl.append(f"{lf}(输出缺标签)")
                elif not po.is_file():
                    bad_lbl.append(f"{lf}(源缺标签 {lo})")
                elif sha256(pf) != sha256(po):
                    bad_lbl.append(lf)
            ck.check("副本图片 sha256 == 源", not bad_img,
                     f"checked={len(sample)} mismatch={bad_img[:3]}")
            ck.check("副本标签 sha256 == 源", not bad_lbl,
                     f"checked={len(sample)} mismatch={bad_lbl[:3]}")

            # 铁律 3：不许复制背景图
            bg = [f for f in copies if not label_lines(dst, sp, f.rsplit(".", 1)[0] + ".txt")]
            ck.check("没有背景图被复制（复制背景只稀释正样本比例）", not bg, f"{bg[:3]}")

            # 副本标签的 class id 必须合法
            bad_cls: List[str] = []
            for f in _sha_sample(copies, limit_sha):
                for ln in label_lines(dst, sp, f.rsplit(".", 1)[0] + ".txt"):
                    parts = ln.split()
                    if len(parts) < 5 or not parts[0].isdigit():
                        bad_cls.append(f"{f}:{ln[:24]}")
            ck.check("副本标签格式合法（class id + >=4 坐标）", not bad_cls, f"{bad_cls[:3]}")
        else:
            # 铁律 2：val/test 一个字节都不许动
            ck.check(f"{sp} 文件名与源完全一致", si == di and sl == dl,
                     f"img {len(si)}/{len(di)}  lbl {len(sl)}/{len(dl)}")
            ck.check(f"{sp} 不含任何副本", not copies, f"copies={len(copies)}")
            sample_i = _sha_sample(si, limit_sha)
            sample_l = _sha_sample(sl, limit_sha)
            same = all(sha256(src / "images" / sp / f) == sha256(dst / "images" / sp / f) for f in sample_i)
            same = same and all(sha256(src / "labels" / sp / f) == sha256(dst / "labels" / sp / f) for f in sample_l)
            ck.check(f"{sp} 内容逐字节一致", same,
                     f"checked img={len(sample_i)} lbl={len(sample_l)}")

    # ---------- 统计与报告对账 ----------
    ds_path, ss_path = dst / "dataset_stats.json", src / "dataset_stats.json"
    ds = json.loads(ds_path.read_text(encoding="utf-8")) if ds_path.is_file() else {}
    ss = json.loads(ss_path.read_text(encoding="utf-8")) if ss_path.is_file() else {}
    ck.check("dataset_stats.json 两侧都存在（源缺 = 不是转换器产出，env 指纹无从核对）",
             bool(ds) and bool(ss), f"src={bool(ss)} dst={bool(ds)}")

    if ds and ss:
        ck.check("val/test 的 splits 统计未变",
                 ds.get("splits", {}).get("val") == ss.get("splits", {}).get("val")
                 and ds.get("splits", {}).get("test") == ss.get("splits", {}).get("test"))
        ck.check("categories 未被复制放大", ds.get("categories") == ss.get("categories"))
        ck.check("env 指纹与源一致（否则两组不是同一套转换器产出）",
                 ds.get("env") == ss.get("env"),
                 json.dumps(ds.get("env"), ensure_ascii=False))
        ck.check("dataset_stats 里有 oversample 块", "oversample" in ds)

        # train 实例数：重算磁盘上的真实实例数，和 stats 声称的对账
        real_inst = 0
        for f in list_labels(dst, "train"):
            real_inst += len(label_lines(dst, "train", f))
        claimed = (ds.get("splits", {}).get("train", {}) or {}).get("instances")
        ck.check("train 实例数：磁盘重算 == dataset_stats 声称",
                 real_inst == claimed, f"recount={real_inst} claimed={claimed}")
        src_inst = 0
        for f in list_labels(src, "train"):
            src_inst += len(label_lines(src, "train", f))
        ck.check("train 实例数确实增加了（过采样生效）", real_inst > src_inst,
                 f"{src_inst} -> {real_inst}")

        # 重算 train 图数与 stats 对账
        real_img = len(list_images(dst, "train"))
        ck.check("train 图数：磁盘重算 == dataset_stats 声称",
                 real_img == (ds.get("splits", {}).get("train", {}) or {}).get("images"),
                 f"recount={real_img}")

    if rep:
        v = rep.get("verify") or {}
        ck.check("report.verify.val_test_identical 为真", bool(v.get("val_test_identical")))
        ck.check("report.verify.sha256_mismatch 为空", not v.get("sha256_mismatch"),
                 f"sha_checked={v.get('sha256_checked')}")
        ck.check("report.env_mismatch 为空", rep.get("env_mismatch") == [],
                 json.dumps(rep.get("env_mismatch"), ensure_ascii=False))
        ck.check("report.link_used 全是 hardlink（软链/复制会让 CI 与本地行为分叉）",
                 set(rep.get("link_used") or []) in ({"hardlink"}, set()),
                 json.dumps(rep.get("link_used"), ensure_ascii=False))
        # 报告声称的副本数 == 磁盘上真实的副本数
        ck.check("report.copies.images == 磁盘 train 副本数",
                 (rep.get("copies") or {}).get("images") == len(copies_by_split.get("train", [])),
                 f"report={(rep.get('copies') or {}).get('images')} disk={len(copies_by_split.get('train', []))}")
        ck.check("report.splits_after == dataset_stats.splits",
                 rep.get("splits_after") == ds.get("splits") if ds else True)

    # ---------- data.yaml / manifest ----------
    y = dst / "data.yaml"
    ck.check("data.yaml 存在", y.is_file())
    if y.is_file():
        txt = y.read_text(encoding="utf-8")
        ck.check("data.yaml 指向输出目录自身（而不是源目录）",
                 dst.resolve().as_posix() in txt.replace("\\", "/")
                 or dst.name in txt, dst.name)
        names = (ds.get("names") if isinstance(ds.get("names"), list) else None)
        if names:
            missing = [n for n in names if str(n) not in txt]
            ck.check("data.yaml 列全了所有类名", not missing, f"missing={missing}")

    man, sman = dst / "manifest.csv", src / "manifest.csv"
    if man.is_file() and sman.is_file():
        ml = man.read_text(encoding="utf-8").strip().splitlines()
        sl_ = sman.read_text(encoding="utf-8").strip().splitlines()
        ncopy = len(copies_by_split.get("train", []))
        ck.check("manifest 行数 = 源行数 + 副本数", len(ml) == len(sl_) + ncopy,
                 f"{len(sl_)}+{ncopy}={len(ml)}")
        ck.check("manifest 表头继承自源", bool(ml) and bool(sl_) and ml[0] == sl_[0],
                 ml[0] if ml else "")

    if not quiet:
        npass = sum(1 for r in ck.rows if r["pass"])
        print(f"\n总体：{npass}/{len(ck.rows)} 通过 —— " + ("全部通过" if ck.ok else "有 FAIL，必须查清再训练"))
    return ck.ok, ck.rows


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="过采样数据集落盘验收（独立重算 + 与 report 对账）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--src", required=True, help="源数据集目录")
    ap.add_argument("--dst", required=True, help="过采样输出目录")
    ap.add_argument("--prefix", default=PREFIX_DEFAULT, help="副本文件名前缀")
    ap.add_argument("--limit-sha", type=int, default=0,
                    help="sha256 抽样上限，0=全查（默认全查，慢但彻底）")
    ap.add_argument("--json-out", default=None, help="把逐项结果写成 JSON")
    ap.add_argument("--quiet", action="store_true", help="不打印逐项，只看退出码")
    args = ap.parse_args(argv)

    def resolve(p: str) -> Path:
        q = Path(p)
        return q if q.is_absolute() else (REPO_ROOT / q)

    ok, rows = verify(resolve(args.src), resolve(args.dst),
                      prefix=args.prefix, limit_sha=args.limit_sha, quiet=args.quiet)
    if args.json_out:
        out = resolve(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"ok": ok, "src": str(resolve(args.src)),
                                   "dst": str(resolve(args.dst)), "checks": rows},
                                  ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if not args.quiet:
            print(f"[out  ] {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())