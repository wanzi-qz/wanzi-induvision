"""tools/oversample_dataset.py（B3 过采样落盘）的端到端测试。

B3 存在的意义是给 B1（Copy-Paste 合成）当对照组：如果"把原图复制几遍"就能拿到
差不多的收益，那合成那套工程复杂度就不值得。所以这里盯的不是"能不能跑通"，
而是**这个对照组干不干净**：

* 副本必须是源图的逐字节复制（图片和标签都是），不能引入任何像素或多边形误差；
* ``val`` / ``test`` 一个文件都不能动 —— 消融要求各组评测集完全一致；
* 背景图（good）不参与复制，配额只作用在稀有缺陷类上；
* 环境指纹不一致要拦住 —— 这正是 B4 那次对比失效的根因；
* 重跑幂等，且要清掉上一轮遗留的孤儿副本。

不依赖 torch / ultralytics / 真实 MVTec，CI 里直接跑。
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List

import pytest
import yaml

from conftest import load_tool
from induvision.balance import ClassStat
from induvision.mvtec import environment_fingerprint

PREFIX = "os__"

# tiny_yolo 的 train：2 张背景 + def_a__000(cls0, 1 实例) + def_b__000(cls1, 1 实例)
# 默认配额 goal = max(实例)×cap_factor = 1×2 = 2 -> 每类 need=1 -> 各 ×1，共 2 份副本
DEFAULT_COPIES = {"os__1__def_a__000.png", "os__1__def_b__000.png"}
SOURCE_OF_COPY = {"os__1__def_a__000.png": "def_a__000.png",
                  "os__1__def_b__000.png": "def_b__000.png"}


@pytest.fixture(scope="module")
def tool():
    return load_tool("oversample_dataset")


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _images(root: Path, split: str) -> Dict[str, bytes]:
    d = root / "images" / split
    if not d.is_dir():
        return {}
    return {p.name: p.read_bytes() for p in sorted(d.iterdir()) if p.is_file()}


def _labels(root: Path, split: str) -> Dict[str, str]:
    d = root / "labels" / split
    if not d.is_dir():
        return {}
    return {p.name: p.read_text(encoding="utf-8") for p in sorted(d.glob("*.txt"))}


def _copies(root: Path, split: str = "train") -> List[str]:
    return sorted(n for n in _images(root, split) if n.startswith(PREFIX))


def _report(out: Path) -> dict:
    return json.loads((out / "oversample_report.json").read_text(encoding="utf-8"))


def _stats(out: Path) -> dict:
    return json.loads((out / "dataset_stats.json").read_text(encoding="utf-8"))


def _names(root: Path) -> dict:
    return yaml.safe_load((root / "data.yaml").read_text(encoding="utf-8"))["names"]


def _manifest(root: Path) -> List[dict]:
    with (root / "manifest.csv").open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _run(tool, tiny_yolo, out: Path, *extra: str) -> int:
    return tool.main(["--src", str(tiny_yolo.root), "--out", str(out), *extra])


def _stat(cid: int, name: str, images: int, instances: int) -> ClassStat:
    return ClassStat(
        class_id=cid, name=name, images=images, instances=instances, backgrounds=images,
        area_px_mean=100.0, area_px_p50=100.0,
        instances_per_image=round(instances / images, 4) if images else 0.0,
    )


# --------------------------------------------------------------------------
# 计划与落盘
# --------------------------------------------------------------------------


def test_dry_run_prints_plan_without_writing(tool, tiny_yolo, tmp_path: Path, capsys):
    out = tmp_path / "os"
    assert _run(tool, tiny_yolo, out, "--dry-run") == 0
    txt = capsys.readouterr().out
    assert "[plan ]" in txt and "[dry  ]" in txt
    assert not out.exists(), "dry-run 不该创建输出目录"


def test_cli_lands_complete_dataset(tool, tiny_yolo, tmp_path: Path, capsys):
    out = tmp_path / "os"
    assert _run(tool, tiny_yolo, out) == 0
    txt = capsys.readouterr().out
    for tag in ("[src  ]", "[plan ]", "[link ]", "[write]", "[check]", "[done ]"):
        assert tag in txt, f"输出里少了 {tag}"

    for name in ("data.yaml", "manifest.csv", "dataset_stats.json", "oversample_report.json"):
        assert (out / name).is_file(), f"缺少产物 {name}"
    assert _names(out) == _names(tiny_yolo.root)

    rep = _report(out)
    assert rep["dry_run"] is False
    assert rep["multiplicity"] == {"bottle": 1, "toothbrush": 1}
    assert rep["copies"] == {"images": 2, "source_images": 2}
    assert rep["instances_before"] == {"bottle": 1, "toothbrush": 1}
    assert rep["instances_after"] == {"bottle": 2, "toothbrush": 2}
    assert rep["splits_after"]["train"] == {
        "images": 6, "background_images": 2, "defect_images": 4, "instances": 4,
    }
    assert rep["splits_after"]["val"] == rep["splits_before"]["val"]
    assert rep["splits_after"]["test"] == rep["splits_before"]["test"]
    assert rep["verify"]["val_test_identical"] is True
    assert rep["verify"]["sha256_mismatch"] == []
    assert rep["verify"]["sha256_checked"] == 4  # 2 图 + 2 标签
    assert rep["env_mismatch"] == []
    assert rep["seconds"] >= 0


def test_default_hardlink_shares_the_same_file(tool, tiny_yolo, tmp_path: Path):
    out = tmp_path / "os"
    assert _run(tool, tiny_yolo, out) == 0
    assert set(_copies(out)) == DEFAULT_COPIES
    for cname, sname in SOURCE_OF_COPY.items():
        copy_img = out / "images" / "train" / cname
        src_img = tiny_yolo.image("train", Path(sname).stem)
        assert os.path.samefile(copy_img, src_img), "默认应是硬链接，不该真的复制一份"
        assert os.path.samefile(
            out / "labels" / "train" / (Path(cname).stem + ".txt"),
            tiny_yolo.label("train", Path(sname).stem),
        )


def test_link_copy_makes_independent_files(tool, tiny_yolo, tmp_path: Path):
    out = tmp_path / "os"
    assert _run(tool, tiny_yolo, out, "--link", "copy") == 0
    assert set(_report(out)["link_used"]) == {"copy"}
    cname, sname = "os__1__def_a__000.png", "def_a__000.png"
    copy_img, src_img = out / "images" / "train" / cname, tiny_yolo.image("train", "def_a__000")
    assert copy_img.read_bytes() == src_img.read_bytes()
    assert not os.path.samefile(copy_img, src_img)
    # 改副本不能污染源数据集（硬链接模式下这一步会毁掉源文件）
    copy_img.write_bytes(b"not-an-image-anymore")
    assert src_img.read_bytes() != b"not-an-image-anymore"
    assert sname in _images(tiny_yolo.root, "train")


# --------------------------------------------------------------------------
# 对照组的干净程度：副本内容 / val,test / 背景图
# --------------------------------------------------------------------------


def test_copies_are_byte_identical_to_source(tool, tiny_yolo, tmp_path: Path):
    out = tmp_path / "os"
    assert _run(tool, tiny_yolo, out, "--link", "copy") == 0
    for cname, sname in SOURCE_OF_COPY.items():
        stem = Path(sname).stem
        assert _sha(out / "images" / "train" / cname) == _sha(tiny_yolo.image("train", stem))
        cl = out / "labels" / "train" / (Path(cname).stem + ".txt")
        assert _sha(cl) == _sha(tiny_yolo.label("train", stem))
        # 标签内容不是空文件，且与源标签逐字符相同（没有归一化往返误差）
        assert cl.read_text(encoding="utf-8") == tiny_yolo.label("train", stem).read_text(encoding="utf-8")
        assert cl.read_text(encoding="utf-8").strip()


def test_val_and_test_are_untouched(tool, tiny_yolo, tmp_path: Path):
    out = tmp_path / "os"
    assert _run(tool, tiny_yolo, out) == 0
    for sp in ("val", "test"):
        assert _images(out, sp) == _images(tiny_yolo.root, sp), f"{sp} 图片被改动"
        assert _labels(out, sp) == _labels(tiny_yolo.root, sp), f"{sp} 标签被改动"
        assert not _copies(out, sp), f"{sp} 里不该出现副本"
    # 源数据集自身也没被写坏
    assert len(_images(tiny_yolo.root, "train")) == 4


def test_backgrounds_are_never_copied(tool, tiny_yolo, tmp_path: Path):
    out = tmp_path / "os"
    assert _run(tool, tiny_yolo, out) == 0
    copies = _copies(out)
    assert set(copies) == DEFAULT_COPIES
    assert not [c for c in copies if "bg__" in c]
    for c in copies:
        lbl = out / "labels" / "train" / (Path(c).stem + ".txt")
        assert lbl.is_file() and lbl.read_text(encoding="utf-8").strip(), "副本必须带非空标签"
    # train 图数 = 源 4 张 + 2 份副本，背景图数量不变
    assert len(_images(out, "train")) == 6
    assert len([n for n in _labels(out, "train") if not n.startswith(PREFIX)]) == 2


# --------------------------------------------------------------------------
# 配额与上限
# --------------------------------------------------------------------------


def test_target_per_class_raises_multiplicity(tool, tiny_yolo, tmp_path: Path):
    out = tmp_path / "os"
    assert _run(tool, tiny_yolo, out, "--target-per-class", "3") == 0
    rep = _report(out)
    assert rep["multiplicity"] == {"bottle": 2, "toothbrush": 2}
    assert rep["copies"]["images"] == 4
    assert rep["splits_after"]["train"]["images"] == 8
    assert rep["instances_after"] == {"bottle": 3, "toothbrush": 3}


def test_cap_factor_clamps_multiplicity(tool, tiny_yolo, tmp_path: Path):
    out = tmp_path / "os"
    assert _run(tool, tiny_yolo, out, "--target-per-class", "10", "--cap-factor", "1") == 0
    rep = _report(out)
    assert rep["multiplicity"] == {"bottle": 1, "toothbrush": 1}, "cap_factor 应该把倍数压回 1"
    assert rep["copies"]["images"] == 2


def test_target_below_current_is_a_noop_but_still_lands(tool, tiny_yolo, tmp_path: Path):
    out = tmp_path / "os"
    assert _run(tool, tiny_yolo, out, "--target-per-class", "1") == 0
    rep = _report(out)
    assert rep["copies"] == {"images": 0, "source_images": 0}
    assert rep["splits_after"]["train"] == rep["splits_before"]["train"]
    assert (out / "data.yaml").is_file() and _names(out) == _names(tiny_yolo.root)
    assert _images(out, "train") == _images(tiny_yolo.root, "train")


def test_multiplicities_rejects_quota_that_is_not_a_whole_multiple(tool):
    mod = load_tool("oversample_dataset")
    hist = [_stat(0, "bottle", 3, 5)]
    assert mod.multiplicities(hist, {0: 6}) == {0: 2}
    assert mod.multiplicities(hist, {0: 0}) == {0: 0}
    with pytest.raises(ValueError, match="整数倍"):
        mod.multiplicities(hist, {0: 7})


def test_copy_name_uses_prefix_and_round_index(tool):
    mod = load_tool("oversample_dataset")
    assert mod.copy_name("def_a__000", ".png", 1) == "os__1__def_a__000.png"
    assert mod.copy_name("def_a__000", ".png", 12) == "os__12__def_a__000.png"
    assert mod.COPY_PREFIX == PREFIX


# --------------------------------------------------------------------------
# 幂等与环境护栏
# --------------------------------------------------------------------------


def test_rerun_is_idempotent_and_purges_orphans(tool, tiny_yolo, tmp_path: Path, capsys):
    out = tmp_path / "os"
    assert _run(tool, tiny_yolo, out) == 0
    first = _images(out, "train")
    capsys.readouterr()

    # 上一轮遗留的孤儿副本：既不在这一轮计划里，也必须被清掉
    (out / "images" / "train" / "os__9__ghost.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 32)
    (out / "labels" / "train" / "os__9__ghost.txt").write_text("0 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2\n",
                                                               encoding="utf-8")

    assert _run(tool, tiny_yolo, out) == 0
    txt = capsys.readouterr().out
    assert "[clean]" in txt
    assert _report(out)["purged_stale"] == 2
    assert _images(out, "train") == first, "重跑结果应与第一次逐字节一致"
    assert not (out / "images" / "train" / "os__9__ghost.png").exists()
    assert not (out / "labels" / "train" / "os__9__ghost.txt").exists()


def test_env_mismatch_blocks_until_explicitly_allowed(tool, tiny_yolo, tmp_path: Path, capsys):
    out = tmp_path / "os"
    bogus = {"python": "2.7.0", "numpy": "0.0.0", "cv2": "0.0.0", "platform": "Somewhere-Else"}
    (tiny_yolo.root / "dataset_stats.json").write_text(
        json.dumps({"env": bogus, "splits": {"train": {"images": 4}}}, ensure_ascii=False),
        encoding="utf-8",
    )
    assert _run(tool, tiny_yolo, out) == 2
    txt = capsys.readouterr().out
    assert "[abort]" in txt and "环境指纹" in txt
    assert not out.exists(), "指纹不一致时不该写盘"

    assert _run(tool, tiny_yolo, out, "--allow-env-mismatch") == 0
    txt = capsys.readouterr().out
    assert "[warn ]" in txt
    rep = _report(out)
    assert rep["env_mismatch"] and "python" in rep["env_mismatch"][0]
    assert rep["env_source"] == bogus
    assert rep["env"] == environment_fingerprint()
    # 落盘的 stats 用当前环境的指纹，避免下游再被误导
    assert _stats(out)["env"] == environment_fingerprint()


def test_stats_inherit_source_and_recompute_splits(tool, tiny_yolo, tmp_path: Path):
    out = tmp_path / "os"
    src_stats = {
        "source_root": "data/mvtec_raw",
        "protocol": {"seed": 0, "train_ratio": 0.7},
        "categories": {"bottle": {"instances": 1}},
        "splits": {"train": {"images": 4, "instances": 2}},
        "env": environment_fingerprint(),
    }
    (tiny_yolo.root / "dataset_stats.json").write_text(
        json.dumps(src_stats, ensure_ascii=False), encoding="utf-8")

    assert _run(tool, tiny_yolo, out) == 0
    st = _stats(out)
    assert st["source_root"] == src_stats["source_root"]
    assert st["protocol"] == src_stats["protocol"]
    assert st["categories"] == src_stats["categories"], "唯一图维度的统计不该被复制放大"
    assert st["splits"]["train"] == {"images": 6, "background_images": 2,
                                     "defect_images": 4, "instances": 4}
    assert "splits_counting" in st
    os_blk = st["oversample"]
    assert os_blk["target_per_class"] is None and os_blk["cap_factor"] == 2.0
    assert os_blk["multiplicity"] == {"bottle": 1, "toothbrush": 1}
    assert os_blk["instances_before"] == {"bottle": 1, "toothbrush": 1}
    assert os_blk["instances_after"] == {"bottle": 2, "toothbrush": 2}
    assert Path(os_blk["source"]).name == "yolo"
    assert st["env"] == environment_fingerprint()


def test_manifest_appends_copy_rows_with_source_metadata(tool, tiny_yolo, tmp_path: Path):
    out = tmp_path / "os"
    assert _run(tool, tiny_yolo, out) == 0
    src_rows, dst_rows = _manifest(tiny_yolo.root), _manifest(out)
    assert len(dst_rows) == len(src_rows) + 2
    with (out / "manifest.csv").open("r", encoding="utf-8", newline="") as fh:
        assert list(csv.DictReader(fh).fieldnames) == list(src_rows[0].keys()), "表头必须继承源 schema"
    assert dst_rows[:len(src_rows)] == src_rows, "源行应原样保留在前面"

    by_name = {r["out_name"]: r for r in src_rows}
    for r in dst_rows[len(src_rows):]:
        assert r["out_name"].startswith(PREFIX)
        assert r["split"] == "train"
        src_row = by_name[SOURCE_OF_COPY[r["out_name"]]]
        assert r["category"] == src_row["category"]
        assert r["defect_type"] == src_row["defect_type"]


def test_no_verify_skips_selfcheck_but_still_reports_splits(tool, tiny_yolo, tmp_path: Path):
    out = tmp_path / "os"
    assert _run(tool, tiny_yolo, out, "--no-verify") == 0
    rep = _report(out)
    assert "verify" not in rep
    assert rep["splits_after"]["train"]["images"] == 6


# --------------------------------------------------------------------------
# 错误路径
# --------------------------------------------------------------------------


def test_out_equals_src_is_refused(tool, tiny_yolo, capsys):
    assert tool.main(["--src", str(tiny_yolo.root), "--out", str(tiny_yolo.root)]) == 2
    txt = capsys.readouterr().out
    assert "[abort]" in txt and "--out" in txt
    assert len(_images(tiny_yolo.root, "train")) == 4, "源数据集不能被就地污染"
    assert not _copies(tiny_yolo.root)


def test_missing_source_data_yaml(tool, tmp_path: Path, capsys):
    src = tmp_path / "not_a_dataset"
    (src / "images" / "train").mkdir(parents=True)
    assert tool.main(["--src", str(src), "--out", str(tmp_path / "os")]) == 2
    txt = capsys.readouterr().out
    assert "[abort]" in txt and "data.yaml" in txt


def test_build_dataset_raises_for_unknown_link_mode(tool, tiny_yolo, tmp_path: Path):
    from induvision.mvtec import _place_file

    with pytest.raises(ValueError, match="link"):
        _place_file(tiny_yolo.image("train", "def_a__000"), tmp_path / "x.png", "teleport")