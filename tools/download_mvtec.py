#!/usr/bin/env python
"""MVTec AD 获取脚本：官方 zip 手动模式 / HuggingFace 镜像自动模式。

两种来源
--------
``--source manual``（默认）
    MVTec 官网下载需要勾选 EULA，无法脚本直连（实测
    ``downloads.mvtec.com`` 返回 403）。脚本会打印官方页面链接，你把 zip 放进
    ``data/mvtec_raw/`` 后由脚本校验 sha256、解压并验证目录结构。

``--source hf``
    从 HuggingFace 镜像仓库 ``foersben/mvtec-ad`` 拉取，目录结构与官方完全一致。
    默认走 ``https://hf-mirror.com``（国内可达，实测 526KB/1.9s）。
    每个文件的 sha256 来自 HF LFS ``lfs.oid``，下载后逐个校验，不是本地伪造值。

许可：MVTec AD = CC BY-NC-SA 4.0（仅限非商业研究用途）。数据不入库，
``.gitignore`` 已排除 ``data/*``。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from induvision.mvtec import (  # noqa: E402
    CATEGORIES,
    MVTecLayoutError,
    find_dataset_root,
    scan_dataset,
)

OFFICIAL_PAGE = "https://www.mvtec.com/company/research/datasets/mvtec-ad"
DEFAULT_ENDPOINT = "https://hf-mirror.com"
DEFAULT_REPO = "foersben/mvtec-ad"
CHUNK = 1 << 20


def _sha256(path: Path, bufsize: int = CHUNK) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for blk in iter(lambda: f.read(bufsize), b""):
            h.update(blk)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# manual
# --------------------------------------------------------------------------- #
def _safe_extract(zf: zipfile.ZipFile, dest: Path) -> int:
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    for info in zf.infolist():
        target = (dest / info.filename).resolve()
        if not str(target).startswith(str(dest.resolve())):
            raise RuntimeError(f"zip 条目越界，拒绝解压: {info.filename}")
        if info.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(info) as src, target.open("wb") as out:
            while True:
                blk = src.read(CHUNK)
                if not blk:
                    break
                out.write(blk)
        n += 1
    return n


def run_manual(args: argparse.Namespace, dest: Path) -> int:
    dest.mkdir(parents=True, exist_ok=True)
    zips = sorted(dest.glob("*.zip"))
    if not zips:
        print("[manual] 没有找到 zip。请按下面步骤操作：")
        print(f"  1) 打开官方页面并勾选 EULA 下载: {OFFICIAL_PAGE}")
        print("     （可直接下载整包 mvtec_anomaly_detection.zip，约 4.9GB；")
        print("       也可只下需要的单类 zip，如 bottle.zip）")
        print(f"  2) 把 zip 放到: {dest}")
        print("  3) 重新运行本脚本")
        return 2

    ck_path = dest / "mvtec_checksums.local.json"
    recorded: Dict[str, str] = {}
    if args.checksums and Path(args.checksums).is_file():
        recorded = json.loads(Path(args.checksums).read_text(encoding="utf-8"))
        print(f"[manual] 载入校验文件: {args.checksums}")
    elif ck_path.is_file():
        recorded = json.loads(ck_path.read_text(encoding="utf-8"))
        print(f"[manual] 载入本地已记录校验值: {ck_path.name}")

    bad = 0
    computed: Dict[str, str] = {}
    for z in zips:
        t0 = time.time()
        digest = _sha256(z)
        computed[z.name] = digest
        expect = recorded.get(z.name)
        mb = z.stat().st_size / 1048576
        if expect is None:
            print(f"[sha256] {z.name:42s} {mb:8.1f}MB  {digest[:16]}…  (首次记录, {time.time()-t0:.1f}s)")
        elif expect.lower() == digest:
            print(f"[sha256] {z.name:42s} {mb:8.1f}MB  OK")
        else:
            bad += 1
            print(f"[sha256] {z.name:42s} {mb:8.1f}MB  不匹配! 期望 {expect[:16]}… 实得 {digest[:16]}…")

    if args.record_checksums:
        merged = dict(recorded)
        merged.update(computed)
        ck_path.write_text(json.dumps(merged, indent=2, sort_keys=True), encoding="utf-8")
        print(f"[manual] 已写入本地校验值 {ck_path}（注意：这是本机计算值，"
              f"官方校验值请以 {OFFICIAL_PAGE} 公布为准）")
    if bad:
        print(f"[manual] {bad} 个 zip 校验失败，终止解压")
        return 1

    if args.verify_only:
        return verify(dest, args)

    total = 0
    for z in zips:
        t0 = time.time()
        with zipfile.ZipFile(z) as zf:
            n = _safe_extract(zf, dest)
        total += n
        print(f"[unzip ] {z.name}: {n} 个文件, {time.time()-t0:.1f}s")
    print(f"[manual] 解压完成，共 {total} 个文件")
    return verify(dest, args)


# --------------------------------------------------------------------------- #
# hf
# --------------------------------------------------------------------------- #
def _hf_list(endpoint: str, repo: str, sub: str, session) -> List[dict]:
    """列出仓库子树下的所有文件（自动跟随 Link 分页）。"""
    url = f"{endpoint}/api/datasets/{repo}/tree/main/{sub}".rstrip("/")
    params = {"recursive": "true"}
    out: List[dict] = []
    while url:
        r = session.get(url, params=params, timeout=60)
        r.raise_for_status()
        batch = r.json()
        out.extend(x for x in batch if x.get("type") == "file")
        url, params = None, None
        link = r.headers.get("Link", "")
        for part in link.split(","):
            if 'rel="next"' in part:
                url = part.split("<", 1)[1].split(">", 1)[0].strip()
    return out


def _hf_download_one(session, url: str, dst: Path, size: int, expect_sha: Optional[str],
                     retries: int = 4) -> Tuple[str, str]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    for attempt in range(1, retries + 1):
        have = tmp.stat().st_size if tmp.exists() else 0
        if dst.exists() and dst.stat().st_size == size:
            if expect_sha and _sha256(dst) != expect_sha:
                dst.unlink()
                continue
            return dst.name, "skip"
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with session.get(url, headers=headers, stream=True, timeout=120) as r:
                if r.status_code == 416:  # 已经下完
                    tmp.rename(dst)
                else:
                    r.raise_for_status()
                    mode = "ab" if have and r.status_code == 206 else "wb"
                    with tmp.open(mode) as f:
                        for blk in r.iter_content(CHUNK):
                            if blk:
                                f.write(blk)
                    if tmp.stat().st_size != size:
                        raise IOError(f"字节数不符 {tmp.stat().st_size}/{size}")
                    tmp.replace(dst)
            if expect_sha:
                got = _sha256(dst)
                if got != expect_sha:
                    dst.unlink()
                    raise IOError(f"sha256 不符 {got[:12]}… != {expect_sha[:12]}…")
            return dst.name, "new"
        except Exception as exc:  # noqa: BLE001
            if attempt == retries:
                return dst.name, f"FAIL: {type(exc).__name__}: {exc}"
            time.sleep(min(2 ** attempt, 15))
    return dst.name, "FAIL"


def run_hf(args: argparse.Namespace, dest: Path) -> int:
    try:
        import requests
    except ImportError:
        print("[hf] 需要 requests：pip install requests")
        return 2

    endpoint = os.environ.get("HF_ENDPOINT", args.endpoint).rstrip("/")
    dest.mkdir(parents=True, exist_ok=True)
    cats: Sequence[str] = args.categories or list(CATEGORIES)
    unknown = [c for c in cats if c not in CATEGORIES]
    if unknown:
        print(f"[hf] 未知类别: {unknown}")
        return 2

    sess = requests.Session()
    sess.headers.update({"User-Agent": "induvision-dataset-fetcher/0.1"})
    print(f"[hf] endpoint={endpoint} repo={args.repo}")

    files: List[dict] = []
    for cat in cats:
        t0 = time.time()
        try:
            entries = _hf_list(endpoint, args.repo, cat, sess)
        except Exception as exc:  # noqa: BLE001
            print(f"[hf] 列目录失败 {cat}: {type(exc).__name__}: {exc}")
            return 1
        keep = []
        for e in entries:
            parts = e["path"].split("/")
            if len(parts) < 3:
                continue
            if args.parts and parts[1] not in args.parts:
                continue
            keep.append(e)
        files.extend(keep)
        print(f"[hf] {cat}: {len(keep)} 个文件 "
              f"({sum(e.get('size',0) for e in keep)/1048576:.1f}MB), 列目录 {time.time()-t0:.1f}s")

    if args.limit:
        files = files[: args.limit]
        print(f"[hf] --limit 生效，仅处理前 {len(files)} 个文件")
    if not files:
        print("[hf] 没有匹配到任何文件")
        return 2

    total_mb = sum(f.get("size", 0) for f in files) / 1048576
    print(f"[hf] 待下载 {len(files)} 个文件 / {total_mb:.1f}MB, 并发 {args.jobs}")

    report = dest / "download_report.json"
    if args.verify_only:
        return verify(dest, args)

    t0 = time.time()
    counts: Dict[str, int] = {}
    failures: List[str] = []
    done_bytes = 0
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = {}
        for e in files:
            url = f"{endpoint}/datasets/{args.repo}/resolve/main/{e['path']}"
            dst = dest / Path(*e["path"].split("/"))
            sha = (e.get("lfs") or {}).get("oid")
            futs[ex.submit(_hf_download_one, sess, url, dst, e.get("size", 0), sha)] = e
        for i, fut in enumerate(as_completed(futs), 1):
            e = futs[fut]
            name, status = fut.result()
            key = "skip" if status == "skip" else ("new" if status == "new" else "fail")
            counts[key] = counts.get(key, 0) + 1
            if key == "fail":
                failures.append(f"{e['path']}: {status}")
            done_bytes += e.get("size", 0)
            if i % 50 == 0 or i == len(files):
                el = time.time() - t0
                print(f"[hf] {i}/{len(files)}  new={counts.get('new',0)} "
                      f"skip={counts.get('skip',0)} fail={counts.get('fail',0)}  "
                      f"{done_bytes/1048576:.0f}MB  {el:.0f}s  "
                      f"{done_bytes/1048576/max(el,1e-6):.1f}MB/s")

    (dest / "download_report.json").write_text(
        json.dumps(
            {
                "endpoint": endpoint,
                "repo": args.repo,
                "categories": list(cats),
                "parts": args.parts,
                "requested": len(files),
                "counts": counts,
                "failures": failures[:50],
                "elapsed_sec": round(time.time() - t0, 1),
                "integrity": "sha256 verified against HF LFS lfs.oid",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[hf] 完成 {counts} 用时 {time.time()-t0:.1f}s，报告 -> {report}")
    if failures:
        print(f"[hf] {len(failures)} 个文件失败，重跑本脚本可断点续传")
        for f in failures[:5]:
            print("       ", f)
    return verify(dest, args)


# --------------------------------------------------------------------------- #
def verify(dest: Path, args: argparse.Namespace) -> int:
    print("[verify] 扫描目录结构 …")
    try:
        root = find_dataset_root(dest)
        scan = scan_dataset(dest, args.categories, strict=True)
    except MVTecLayoutError as exc:
        print(f"[verify] 结构校验失败: {exc}")
        print("[verify] 若只下载了部分类别，请加 --categories 指定后重试")
        return 1
    print(f"[verify] 数据集根目录: {root}")
    by_cat: Dict[str, Dict[str, int]] = {}
    for s in scan.samples:
        d = by_cat.setdefault(s.category, {"good_train": 0, "good_test": 0, "defect": 0})
        if s.is_defect:
            d["defect"] += 1
        elif s.official_split == "train":
            d["good_train"] += 1
        else:
            d["good_test"] += 1
    print(f"{'category':<12}{'train/good':>12}{'test/good':>11}{'defect':>9}{'total':>8}")
    tot = {"good_train": 0, "good_test": 0, "defect": 0}
    for cat in sorted(by_cat):
        d = by_cat[cat]
        for k in tot:
            tot[k] += d[k]
        print(f"{cat:<12}{d['good_train']:>12}{d['good_test']:>11}{d['defect']:>9}"
              f"{sum(d.values()):>8}")
    print(f"{'TOTAL':<12}{tot['good_train']:>12}{tot['good_test']:>11}{tot['defect']:>9}"
          f"{sum(tot.values()):>8}")
    for w in scan.warnings[:10]:
        print("[warn]", w)
    print(f"[verify] 通过：{len(scan.categories)} 个类别 / {len(scan.samples)} 张图")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="获取并校验 MVTec AD 数据集", formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=("manual", "hf"), default="manual")
    ap.add_argument("--dest", default=str(REPO_ROOT / "data" / "mvtec_raw"))
    ap.add_argument("--categories", nargs="*", default=None, help="只处理指定类别（默认 15 类全量）")
    ap.add_argument("--verify-only", action="store_true", help="只扫描校验，不下载/不解压")
    # manual
    ap.add_argument("--checksums", default=None, help="外部 sha256 校验文件(json: {文件名: sha256})")
    ap.add_argument("--record-checksums", action="store_true", help="首次下载后把本机算出的 sha256 记录下来")
    # hf
    ap.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--parts", nargs="*", default=None, choices=("train", "test", "ground_truth"),
                    help="只下载指定子目录")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="只下载前 N 个文件（冒烟测试）")
    args = ap.parse_args(argv)

    dest = Path(args.dest).expanduser().resolve()
    print(f"[dest] {dest}")
    if args.verify_only:
        # 纯本地操作：不需要 zip、不联网。必须排在 source 分发之前，
        # 否则 run_manual 会先要求目录里有 zip，run_hf 会先去远端列目录。
        return verify(dest, args)
    if args.source == "manual":
        return run_manual(args, dest)
    return run_hf(args, dest)


if __name__ == "__main__":
    raise SystemExit(main())