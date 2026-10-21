"""tools/synthesize_defects.py 的端到端测试。

链路：微型 YOLO-seg 数据集 -> copy-paste 合成 -> 增强版数据集。
不依赖 torch / ultralytics / 真实 MVTec，CI 里直接跑。

盯三件事：产物齐不齐、val/test 有没有被污染、结果可不可复现。
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict

import pytest
import yaml

from conftest import build_tiny_yolo, load_tool

SPLITS = ("train", "val", "test")


@pytest.fixture(scope="module")
def tool():
    return load_tool("synthesize_defects")


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


def _rows(out: Path):
    with (out / "synth_manifest.csv").open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _report(out: Path) -> dict:
    return json.loads((out / "synth_report.json").read_text(encoding="utf-8"))


def _names(root: Path) -> dict:
    return yaml.safe_load((root / "data.yaml").read_text(encoding="utf-8"))["names"]


# --------------------------------------------------------------------------
# 正常路径
# --------------------------------------------------------------------------


def test_cli_writes_complete_dataset(tool, tiny_yolo, tmp_path: Path, capsys):
    out = tmp_path / "cp"
    rc = tool.main(["--src", str(tiny_yolo.root), "--out", str(out), "--figures", "2", "--workers", "1"])
    assert rc == 0
    txt = capsys.readouterr().out
    for tag in ("[plan ]", "[bank ]", "[synth]", "[write]", "[done ]"):
        assert tag in txt, f"输出里少了 {tag}"

    for name in ("data.yaml", "synth_manifest.csv", "synth_report.json", "defect_bank.npz"):
        assert (out / name).is_file(), f"缺少产物 {name}"
    assert len(list((out / "figures").glob("*.png"))) == 2
    assert _names(out) == _names(tiny_yolo.root)

    rows = _rows(out)
    rep = _report(out)
    assert rep["synth_images"] == len(rows) == 2
    for r in rows:
        assert r["split"] == "train"
        # 底板是背景图，贴了几个实例标签里就该有几个多边形
        assert int(r["n_pasted"]) == int(r["polygons"]) >= 1
        assert r["base_image"] in {"bg__000.png", "bg__001.png"}
        assert float(r["visible_ratio_mean"]) >= 0.75
        assert r["seed_child"].startswith("[0,")

    # 配额：strategy=max + cap-factor=2 -> 每类 1 个实例补到 2 个
    assert rep["instances_before"] == {"bottle": 1, "toothbrush": 1}
    assert rep["plan"]["per_class"]["0"]["target_instances"] == 2
    assert all(v >= 2 for v in rep["instances_after"].values())
    assert all(v <= 0 for v in rep["remaining_quota"].values())
    assert rep["bank"]["built_from"] == ["train"]
    assert rep["bank"]["instances"] == 2
    assert rep["polygon_area_fidelity"]["mean"] is not None


def test_cli_only_touches_train_split(tool, tiny_yolo, tmp_path: Path):
    """防泄漏：合成图只进 train，val/test 的图与标签一个字节都不能变。"""
    out = tmp_path / "cp"
    # workers=1 让「补到配额就停」变成确定性的，图数才能精确断言
    assert tool.main(["--src", str(tiny_yolo.root), "--out", str(out), "--figures", "0",
                      "--workers", "1", "--link", "copy"]) == 0

    src_img = {s: _images(tiny_yolo.root, s) for s in SPLITS}
    dst_img = {s: _images(out, s) for s in SPLITS}
    assert len(dst_img["train"]) == len(src_img["train"]) + 2
    for s in ("val", "test"):
        assert dst_img[s] == src_img[s], f"{s} 划分被改动了"
        assert _labels(out, s) == _labels(tiny_yolo.root, s)
        assert not [n for n in dst_img[s] if n.startswith("synth__")]

    src_lbl, dst_lbl = _labels(tiny_yolo.root, "train"), _labels(out, "train")
    for name, text in src_lbl.items():
        assert dst_lbl[name] == text, f"原图标签 {name} 被改写"

    synth = sorted(n for n in dst_img["train"] if n.startswith("synth__"))
    assert set(synth) == {r["out_name"] for r in _rows(out)}
    # 合成图必须跟底板不一样（真的贴上去了）
    bg = {src_img["train"]["bg__000.png"], src_img["train"]["bg__001.png"]}
    assert all(dst_img["train"][n] not in bg for n in synth)
    for n in synth:
        assert (out / "labels" / "train" / f"{Path(n).stem}.txt").is_file()


def test_cli_target_per_class_reaches_quota(tool, tiny_yolo, tmp_path: Path):
    out = tmp_path / "cp"
    rc = tool.main(
        ["--src", str(tiny_yolo.root), "--out", str(out), "--target-per-class", "4",
         "--max-per-image", "1", "--workers", "1", "--link", "copy", "--figures", "0"]
    )
    assert rc == 0
    rep = _report(out)
    assert rep["plan"]["target"] == 4
    assert rep["plan"]["total_instances"] == 6  # 两类各差 3 个
    assert rep["synth_images"] == 6
    assert all(v >= 4 for v in rep["instances_after"].values()), rep["instances_after"]


def test_cli_reuses_cached_defect_bank(tool, tiny_yolo, tmp_path: Path, capsys):
    out = tmp_path / "cp"
    args = ["--src", str(tiny_yolo.root), "--out", str(out), "--workers", "1",
            "--link", "copy", "--figures", "0"]
    assert tool.main(args) == 0
    first, first_train = _report(out), _images(out, "train")
    capsys.readouterr()

    # 第二次跑：复用缺陷库缓存，并且能覆盖已有产物（_place_file 先 unlink 再写）
    assert tool.main(args) == 0
    assert "复用缓存" in capsys.readouterr().out
    second = _report(out)
    assert second["synth_images"] == first["synth_images"] == 2
    assert _images(out, "train") == first_train, "重跑之后产物变了，缓存复用没走到同一条路径"


# --------------------------------------------------------------------------
# 可复现性
# --------------------------------------------------------------------------


def test_cli_is_bit_reproducible(tool, tiny_yolo, tmp_path: Path):
    a, b = tmp_path / "a", tmp_path / "b"
    for out in (a, b):
        assert tool.main(["--src", str(tiny_yolo.root), "--out", str(out), "--workers", "1",
                          "--link", "copy", "--figures", "0"]) == 0
    for s in SPLITS:
        assert _images(a, s) == _images(b, s), f"{s} 图像不可复现"
        assert _labels(a, s) == _labels(b, s), f"{s} 标签不可复现"
    assert _rows(a) == _rows(b)


def test_cli_reproducible_across_worker_counts(tool, tiny_yolo, tmp_path: Path):
    """并发度只影响「合成到第几张停」，不影响每张图的内容。

    每张图用 SeedSequence([seed, class_id, seq]) 派生独立 rng，
    所以只要 (class_id, seq) 相同，产物就必须逐像素相同。
    """
    w1, w4 = tmp_path / "w1", tmp_path / "w4"
    common = ["--src", str(tiny_yolo.root), "--link", "copy", "--figures", "0", "--seed", "7"]
    assert tool.main(common + ["--out", str(w1), "--workers", "1"]) == 0
    assert tool.main(common + ["--out", str(w4), "--workers", "4"]) == 0

    a, b = _images(w1, "train"), _images(w4, "train")
    shared = sorted(n for n in set(a) & set(b) if n.startswith("synth__"))
    assert shared, "两次运行没有共同的合成图，无法比对"
    for n in shared:
        assert a[n] == b[n], f"{n} 在不同 workers 下像素不一致"
        lbl = Path(n).stem + ".txt"
        assert _labels(w1, "train")[lbl] == _labels(w4, "train")[lbl]

    # 一波 workers 个任务共享同一份 remaining 快照，所以并发度越高越可能「超额合成」。
    # 允许超额（多补几个实例无害），不允许欠额（配额没打满就是 bug）。
    r1, r4 = _report(w1), _report(w4)
    assert r4["synth_images"] >= r1["synth_images"]
    for rep in (r1, r4):
        assert all(v <= 0 for v in rep["remaining_quota"].values()), rep["remaining_quota"]
        assert all(v >= 2 for v in rep["instances_after"].values()), rep["instances_after"]


def test_cli_purges_orphans_from_previous_run(tool, tiny_yolo, tmp_path: Path, capsys):
    """上一轮合成得多、这一轮合成得少时，多出来的旧文件必须被清掉。

    孤儿合成图不在 manifest 里，却还躺在 images/train 下、旧标签也还在，
    ultralytics 照样会拿去训练 —— 增强集就悄悄不可复现了。
    """
    out = tmp_path / "cp"
    common = ["--src", str(tiny_yolo.root), "--out", str(out), "--workers", "1", "--link", "copy"]

    # 第一轮：每类补到 4 个实例 -> 6 张合成图 + 2 张抽查图
    assert tool.main(common + ["--target-per-class", "4", "--max-per-image", "1", "--figures", "2"]) == 0
    first = {p.name for p in (out / "images" / "train").glob("synth__*.png")}
    assert len(first) == 6
    assert len(list((out / "figures").glob("*.png"))) == 2

    # 第二轮：默认配额（cap-factor=2）只要 2 张
    capsys.readouterr()
    assert tool.main(common + ["--figures", "0"]) == 0
    assert "[clean]" in capsys.readouterr().out

    rows = _rows(out)
    on_disk = {p.name for p in (out / "images" / "train").glob("synth__*.png")}
    assert on_disk == {r["out_name"] for r in rows}, "有孤儿合成图没被清掉"
    assert len(on_disk) == 2 < len(first)
    assert {p.name for p in (out / "labels" / "train").glob("synth__*.txt")} == {
        Path(n).stem + ".txt" for n in on_disk
    }
    assert not list((out / "figures").glob("*.png")), "上一轮的抽查图也该清掉"
    # 原图一张都不能少
    assert len(_images(out, "train")) == len(_images(tiny_yolo.root, "train")) + 2


# --------------------------------------------------------------------------
# 错误路径
# --------------------------------------------------------------------------


def test_cli_dry_run_writes_nothing(tool, tiny_yolo, tmp_path: Path, capsys):
    out = tmp_path / "dry"
    assert tool.main(["--src", str(tiny_yolo.root), "--out", str(out), "--dry-run"]) == 0
    txt = capsys.readouterr().out
    assert "[dry  ]" in txt and "未写盘" in txt
    assert not out.exists(), "dry-run 不该创建输出目录"


def test_cli_strategy_none_only_diagnoses(tool, tiny_yolo, tmp_path: Path, capsys):
    out = tmp_path / "none"
    assert tool.main(["--src", str(tiny_yolo.root), "--out", str(out), "--strategy", "none"]) == 0
    assert "[skip ] 无需合成" in capsys.readouterr().out
    assert not out.exists()


def test_cli_empty_defect_bank_fails_fast(tool, tmp_path: Path, capsys):
    """train 里一个缺陷实例都没有 -> 缺陷库为空，必须报错退出而不是静默产出空数据集。"""
    src = build_tiny_yolo(tmp_path / "yolo_nodefect", with_defects=False)
    out = tmp_path / "cp"
    rc = tool.main(["--src", str(src.root), "--out", str(out), "--target-per-class", "5"])
    assert rc == 1
    assert "缺陷库为空" in capsys.readouterr().out
    assert not (out / "data.yaml").exists()
    assert not (out / "synth_manifest.csv").exists()


def test_cli_all_pastes_rejected_fails_with_hint(tool, tiny_yolo, tmp_path: Path, capsys):
    """bg-std-min 拉到不可能满足 -> 所有落点被拒，要给出可调参数的提示。"""
    out = tmp_path / "cp"
    rc = tool.main(["--src", str(tiny_yolo.root), "--out", str(out), "--bg-std-min", "999",
                    "--link", "copy", "--figures", "0"])
    assert rc == 1
    txt = capsys.readouterr().out
    assert "一张合成图都没产出" in txt
    assert "--bg-std-min" in txt and "--min-visible" in txt
    assert not (out / "data.yaml").exists()
    assert not (out / "synth_manifest.csv").exists()