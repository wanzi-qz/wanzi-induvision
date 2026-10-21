"""tools/verify_oversample.py 的测试。

验收脚本本身是"防手滑"的最后一道闸，所以它的测试重点不是"能不能跑通"，而是
**该拦的到底拦不拦得住**。B3 这组消融只有一个变量（稀有类被复制几份），闸门要是漏了，
跑出来的 mAP 差异就是假的，而且从数字上完全看不出来 —— 这类 bug 只能用测试钉死。

覆盖的失效模式（每条都对应一次真实踩过的坑或一个显而易见的误用）：
* val/test 被改一个字节 -> 必须 FAIL（消融要求评测集完全一致）；
* 副本混进 val -> 必须 FAIL；
* 背景图被复制 -> 必须 FAIL（只稀释正样本比例，纯负收益）；
* 副本内容与源不一致 -> 必须 FAIL（过采样不是数据增强）；
* 源文件丢了 -> 判 FAIL 而不是抛 FileNotFoundError（曾真实崩溃过一次）；
* report 自述与磁盘不符 -> 必须 FAIL（不能只信报告）；
* ``--limit-sha`` 抽样时仍要能抓到第一个坏副本。

不依赖 torch / ultralytics / 真实 MVTec，CI 里直接跑。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import pytest

from conftest import build_tiny_yolo, load_tool

PREFIX = "os__"


@pytest.fixture(scope="module")
def os_tool():
    return load_tool("oversample_dataset")


@pytest.fixture(scope="module")
def vf():
    return load_tool("verify_oversample")


@pytest.fixture
def pair(tmp_path: Path, os_tool) -> Tuple[Path, Path]:
    """造一份 tiny 源数据集，再真实跑一遍过采样，返回 (src_root, dst_root)。"""
    from induvision.mvtec import environment_fingerprint

    src = build_tiny_yolo(tmp_path / "yolo")
    # 真实链路里 dataset_stats.json 由 convert_mvtec_to_yolo.py 落盘，tiny fixture 没有。
    # 补一份最小的，否则 env 指纹 / categories / 实例数对账那几项验收根本没有输入。
    (src.root / "dataset_stats.json").write_text(json.dumps({
        "source_root": str(src.root),
        "protocol": "tiny-fixture",
        "classes": src.names,
        "env": environment_fingerprint(),
        # splits 必须是真值：oversample 会按含副本重算 dst 的 splits，验收项
        # "val/test 的 splits 统计未变" 就是拿两侧对比，写空字典会让这项假失败。
        "splits": os_tool._split_stats(os_tool.read_dataset(src.root)),
        "categories": {},
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    dst = tmp_path / "yolo_os"
    rc = os_tool.main(["--src", str(src.root), "--out", str(dst)])
    assert rc == 0, "过采样自身就该跑通，否则后面的验收测试没有意义"
    return src.root, dst


def _verify(vf, src: Path, dst: Path, **kw) -> Tuple[bool, List[Dict[str, object]]]:
    return vf.verify(src, dst, prefix=kw.pop("prefix", PREFIX), quiet=True, **kw)


def _failed(rows: List[Dict[str, object]]) -> List[str]:
    return [str(r["label"]) for r in rows if not r["pass"]]


def _copies(dst: Path, split: str = "train") -> List[str]:
    d = dst / "images" / split
    return sorted(p.name for p in d.iterdir() if p.name.startswith(PREFIX))


def _flip_one_byte(p: Path) -> None:
    """改最后一个字节。

    **必须先 unlink 再写**：oversample_dataset.py 是硬链接落盘，副本/源文件在
    磁盘上是同一个 inode。直接 write_bytes 会把源文件一起改掉，两边 sha256 依旧
    相等，验收照样 PASS —— 测试就成了摆设。断链才是真实的"副本被写坏"场景。
    """
    b = bytearray(p.read_bytes())
    b[-1] ^= 0xFF
    p.unlink()
    p.write_bytes(bytes(b))


def _rewrite(p: Path, content: str) -> None:
    """同上，改文本也要先断链。"""
    p.unlink()
    p.write_text(content, encoding="utf-8")


def _read_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- 正常路径


def test_happy_path_all_pass(vf, pair):
    src, dst = pair
    ok, rows = _verify(vf, src, dst)
    assert ok, f"干净落盘必须全过，实际 FAIL：{_failed(rows)}"
    assert len(rows) >= 25, f"检查项被意外削减了，只剩 {len(rows)} 条"
    assert not _failed(rows)


def test_copies_exist_and_match_source(vf, pair):
    """副本确实是 os__<序号>__<原名>，且逐字节等于源。"""
    src, dst = pair
    copies = _copies(dst)
    assert copies, "tiny 数据集默认配额下应该产生副本"
    for c in copies:
        origin = vf.stem_of_copy(c, PREFIX)
        assert origin is not None and not origin.startswith(PREFIX)
        assert (dst / "images" / "train" / c).read_bytes() == \
               (src / "images" / "train" / origin).read_bytes()


def test_val_test_untouched(vf, pair):
    src, dst = pair
    for sp in ("val", "test"):
        assert sorted(p.name for p in (src / "images" / sp).iterdir()) == \
               sorted(p.name for p in (dst / "images" / sp).iterdir())
        assert not _copies(dst, sp)


def test_instance_count_grew(vf, pair):
    src, dst = pair
    si = sum(1 for sp in ("train",) for f in (src / "labels" / sp).glob("*.txt")
             for ln in f.read_text(encoding="utf-8").splitlines() if ln.strip())
    di = sum(1 for f in (dst / "labels" / "train").glob("*.txt")
             for ln in f.read_text(encoding="utf-8").splitlines() if ln.strip())
    assert di > si, f"{si} -> {di}"


# ---------------------------------------------------------------- 该拦的


def test_missing_dirs_fail(vf, tmp_path):
    ok, rows = _verify(vf, tmp_path / "nope", tmp_path / "nope2")
    assert not ok
    assert _failed(rows)


def test_tampered_val_image_fails(vf, pair):
    src, dst = pair
    victim = next((dst / "images" / "val").glob("*.png"))
    _flip_one_byte(victim)
    ok, rows = _verify(vf, src, dst)
    assert not ok
    assert any("val" in f and "逐字节" in f for f in _failed(rows)), _failed(rows)


def test_copy_leaked_into_val_fails(vf, pair):
    src, dst = pair
    c = _copies(dst)[0]
    stem = c.rsplit(".", 1)[0]
    (dst / "images" / "val" / c).write_bytes((dst / "images" / "train" / c).read_bytes())
    (dst / "labels" / "val" / f"{stem}.txt").write_text(
        (dst / "labels" / "train" / f"{stem}.txt").read_text(encoding="utf-8"), encoding="utf-8")
    ok, rows = _verify(vf, src, dst)
    assert not ok
    assert any("val" in f and "副本" in f for f in _failed(rows)), _failed(rows)


def test_background_copy_fails(vf, pair):
    """复制背景图不会增加任何正样本，只会拖慢 epoch，必须拦住。"""
    src, dst = pair
    def _is_bg(q: Path) -> bool:
        if q.name.startswith(PREFIX):
            return False
        lp = dst / "labels" / "train" / (q.stem + ".txt")
        # 背景图两种落盘方式都合法：没有 .txt（tiny fixture），或有空 .txt（转换器）
        return not lp.is_file() or not lp.read_text(encoding="utf-8").strip()

    bg = next(p for p in (dst / "images" / "train").glob("*.png") if _is_bg(p))
    (dst / "images" / "train" / f"{PREFIX}9__{bg.name}").write_bytes(bg.read_bytes())
    (dst / "labels" / "train" / f"{PREFIX}9__{bg.stem}.txt").write_text("", encoding="utf-8")
    ok, rows = _verify(vf, src, dst)
    assert not ok
    assert any("背景" in f for f in _failed(rows)), _failed(rows)


def test_tampered_copy_image_fails(vf, pair):
    src, dst = pair
    _flip_one_byte(dst / "images" / "train" / _copies(dst)[0])
    ok, rows = _verify(vf, src, dst)
    assert not ok
    assert any("副本图片 sha256" in f for f in _failed(rows)), _failed(rows)


def test_tampered_copy_label_fails(vf, pair):
    src, dst = pair
    c = _copies(dst)[0]
    lp = dst / "labels" / "train" / (c.rsplit(".", 1)[0] + ".txt")
    _rewrite(lp, lp.read_text(encoding="utf-8") + "\n0 0.5 0.5 0.1 0.1 0.2 0.2 0.3 0.3\n")
    ok, rows = _verify(vf, src, dst)
    assert not ok
    assert any("副本标签 sha256" in f for f in _failed(rows)), _failed(rows)


def test_missing_source_image_fails_without_crash(vf, pair):
    """源文件丢了要判 FAIL，不能抛 FileNotFoundError（真实崩溃过一次）。"""
    src, dst = pair
    c = _copies(dst)[0]
    origin = vf.stem_of_copy(c, PREFIX)
    (src / "images" / "train" / origin).unlink()
    ok, rows = _verify(vf, src, dst)
    assert not ok
    assert any("副本图片 sha256" in f for f in _failed(rows)), _failed(rows)


def test_stray_label_line_breaks_instance_recount(vf, pair):
    """往源图对应的标签里塞一行，磁盘重算就和 dataset_stats 声称的对不上。"""
    src, dst = pair
    victim = next(p for p in (dst / "labels" / "train").glob("*.txt")
                  if not p.name.startswith(PREFIX) and p.read_text(encoding="utf-8").strip())
    _rewrite(victim, victim.read_text(encoding="utf-8") + "\n1 0.5 0.5 0.1 0.1 0.2 0.2 0.3 0.3\n")
    ok, rows = _verify(vf, src, dst)
    assert not ok
    assert any("磁盘重算" in f for f in _failed(rows)), _failed(rows)


def test_report_self_claim_mismatch_fails(vf, pair):
    """report 里的副本数被改错，要和磁盘对不上账 -> FAIL（不能只信报告自述）。"""
    src, dst = pair
    rp = dst / "oversample_report.json"
    rep = _read_json(rp)
    rep["copies"]["images"] = rep["copies"]["images"] + 7
    rp.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    ok, rows = _verify(vf, src, dst)
    assert not ok
    assert any("report.copies.images" in f for f in _failed(rows)), _failed(rows)


def test_env_fingerprint_mismatch_fails(vf, pair):
    """B4 那次对比失效就是因为两组数据集出自不同环境，这条必须守住。"""
    src, dst = pair
    sp = dst / "dataset_stats.json"
    st = _read_json(sp)
    st["env"] = dict(st.get("env") or {}, cv2="9.9.9")
    sp.write_text(json.dumps(st, ensure_ascii=False, indent=2), encoding="utf-8")
    ok, rows = _verify(vf, src, dst)
    assert not ok
    assert any("env" in f for f in _failed(rows)), _failed(rows)


# ---------------------------------------------------------------- 抽样与 CLI


def test_limit_sha_sampling_still_catches_first_bad(vf, pair):
    src, dst = pair
    _flip_one_byte(dst / "images" / "train" / _copies(dst)[0])
    ok, rows = _verify(vf, src, dst, limit_sha=1)
    assert not ok, "抽样也必须抓到第一个坏副本"
    assert any("副本图片 sha256" in f for f in _failed(rows)), _failed(rows)


def test_prefix_override(vf, pair):
    src, dst = pair
    ok, rows = _verify(vf, src, dst, prefix="zz__")
    assert not ok, "前缀对不上就找不到副本，应该判 FAIL 而不是静默通过"
    assert any("至少有副本" in f for f in _failed(rows)), _failed(rows)


def test_cli_exit_code_zero_on_clean(vf, pair, tmp_path):
    src, dst = pair
    out = tmp_path / "v.json"
    rc = vf.main(["--src", str(src), "--dst", str(dst), "--json-out", str(out), "--quiet"])
    assert rc == 0
    data = _read_json(out)
    assert data["ok"] is True and len(data["checks"]) >= 25


def test_cli_exit_code_one_on_tamper(vf, pair):
    src, dst = pair
    _flip_one_byte(next((dst / "images" / "test").glob("*.png")))
    rc = vf.main(["--src", str(src), "--dst", str(dst), "--quiet"])
    assert rc == 1


def test_quiet_prints_nothing(vf, pair, capsys):
    src, dst = pair
    vf.main(["--src", str(src), "--dst", str(dst), "--quiet"])
    assert capsys.readouterr().out == ""


def test_verbose_prints_summary(vf, pair, capsys):
    src, dst = pair
    vf.main(["--src", str(src), "--dst", str(dst)])
    out = capsys.readouterr().out
    assert "PASS" in out and "总体" in out


def test_stem_of_copy_parses_index_layer():
    """os__1__a.png -> a.png；os__a.png -> a.png；非副本 -> None。"""
    from conftest import load_tool as _lt
    m = _lt("verify_oversample")
    assert m.stem_of_copy("os__1__a.png", PREFIX) == "a.png"
    assert m.stem_of_copy("os__12__b__c.png", PREFIX) == "b__c.png"
    assert m.stem_of_copy("os__a.png", PREFIX) == "a.png"
    assert m.stem_of_copy("a.png", PREFIX) is None