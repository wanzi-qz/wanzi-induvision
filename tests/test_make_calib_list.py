"""tools/make_calib_list.py 的单测：INT8 校准清单的口径与幂等。

这里守两条线：

* **校准集不许从 val/test 出。** 拿评测集统计动态范围，再用同一个评测集报 mAP，
  等于让考卷参与了划重点——报出来的"INT8 掉几个点"必然偏乐观，而那正好是 W3b
  唯一要量的东西。所以 ``--split val/test`` 是直接拒（rc 2），不是警告。
* **换清单必须显式。** 清单指纹变了还悄悄覆盖，历史 INT8 数字就不能再和新的相减，
  而报告上看不出来。所以指纹不同要 ``--force``（rc 3）。
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from conftest import load_tool
from induvision.deploy import load_sample_list


@pytest.fixture
def tool():
    return load_tool("make_calib_list")


def _args(tiny_yolo, out: Path, *extra: str):
    return ["--data", str(tiny_yolo.root / "data.yaml"), "--split", "train",
            "--n", "3", "--imgsz", "64", "--out", str(out), *extra]


def _add_train_image(tiny, stem: str, seed: int = 7) -> Path:
    """往 train split 里塞一张图，让等间距抽样的落点变化（指纹随之变化）。"""
    p = tiny.root / "images" / "train" / f"{stem}.png"
    p.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    assert cv2.imwrite(str(p), rng.integers(0, 255, (72, 96, 3), dtype=np.uint8))
    return p


# --------------------------------------------------------------------------
# 纯函数
# --------------------------------------------------------------------------

def test_class_of_takes_category_before_double_underscore(tool):
    assert tool.class_of("images/train/bottle__train__good__000.png") == "bottle"
    assert tool.class_of(Path("a/b/cable__test__cut__006.png")) == "cable"


def test_class_of_without_delimiter_returns_whole_stem(tool):
    assert tool.class_of("images/train/plain.png") == "plain.png"


def test_class_of_empty_name_is_placeholder(tool):
    assert tool.class_of("") == "?"


def test_coverage_counts_and_sorts(tool):
    cov = tool.coverage([Path("b__1.png"), Path("a__1.png"), Path("a__2.png")])
    assert cov == {"a": 2, "b": 1}
    assert list(cov) == sorted(cov)


def test_coverage_empty(tool):
    assert tool.coverage([]) == {}


def test_dataset_root_reads_path_key(tool, tiny_yolo):
    got = tool.dataset_root(tiny_yolo.root / "data.yaml")
    assert Path(got) == Path(tiny_yolo.root)


def test_dataset_root_resolves_relative_against_yaml_dir(tool, tmp_path: Path):
    ds = tmp_path / "ds"
    (ds / "images" / "train").mkdir(parents=True)
    y = ds / "data.yaml"
    y.write_text("path: .\ntrain: images/train\n", encoding="utf-8")
    assert tool.dataset_root(y) == ds.resolve()


def test_dataset_root_falls_back_to_yaml_dir(tool, tmp_path: Path):
    y = tmp_path / "data.yaml"
    y.write_text("train: images/train\n", encoding="utf-8")
    assert tool.dataset_root(y) == tmp_path.resolve()


def test_build_list_returns_paths_pool_and_coverage(tool, tiny_yolo):
    res = tool.build_list(tiny_yolo.root / "data.yaml", "train", 3)
    assert len(res["paths"]) == 3
    assert len(res["all"]) == 4
    assert sum(res["cov"].values()) == 3


def test_build_list_raises_when_split_is_empty(tool, tmp_path: Path):
    y = tmp_path / "data.yaml"
    y.write_text(f"path: {tmp_path.as_posix()}\ntrain: images/train\n", encoding="utf-8")
    with pytest.raises(ValueError, match="没有图"):
        tool.build_list(y, "train", 4)


# --------------------------------------------------------------------------
# CLI：错误路径
# --------------------------------------------------------------------------

def test_main_rejects_missing_data_yaml(tool, tmp_path: Path, capsys):
    rc = tool.main(["--data", str(tmp_path / "nope.yaml"), "--out", str(tmp_path / "c.json")])
    assert rc == 2
    assert "缺 data.yaml" in capsys.readouterr().out


@pytest.mark.parametrize("split", ["val", "test"])
def test_main_refuses_evaluation_splits(tool, tiny_yolo, tmp_path: Path, capsys, split):
    """这条是整个脚本存在的理由：校准集碰了评测集，INT8 的掉点数就不可信了。"""
    rc = tool.main(["--data", str(tiny_yolo.root / "data.yaml"), "--split", split,
                    "--out", str(tmp_path / "c.json")])
    out = capsys.readouterr().out
    assert rc == 2
    assert "拒绝执行" in out and "校准请用 train" in out
    assert not (tmp_path / "c.json").exists()


def test_main_rejects_non_positive_n(tool, tiny_yolo, tmp_path: Path, capsys):
    rc = tool.main(["--data", str(tiny_yolo.root / "data.yaml"), "--split", "train",
                    "--n", "0", "--out", str(tmp_path / "c.json")])
    assert rc == 2
    assert "--n 要大于 0" in capsys.readouterr().out


def test_main_rejects_when_pool_is_smaller_than_n(tool, tiny_yolo, tmp_path: Path, capsys):
    rc = tool.main(["--data", str(tiny_yolo.root / "data.yaml"), "--split", "train",
                    "--n", "999", "--out", str(tmp_path / "c.json")])
    out = capsys.readouterr().out
    assert rc == 2
    assert "抽不满" in out
    assert not (tmp_path / "c.json").exists()


def test_main_rejects_split_without_images(tool, tmp_path: Path, capsys):
    y = tmp_path / "data.yaml"
    y.write_text(f"path: {tmp_path.as_posix()}\ntrain: images/train\n", encoding="utf-8")
    rc = tool.main(["--data", str(y), "--split", "train", "--n", "2",
                    "--out", str(tmp_path / "c.json")])
    assert rc == 2


# --------------------------------------------------------------------------
# CLI：正常路径
# --------------------------------------------------------------------------

def test_main_writes_reproducible_list(tool, tiny_yolo, tmp_path: Path):
    out = tmp_path / "calib.json"
    assert tool.main(_args(tiny_yolo, out)) == 0
    d = json.loads(out.read_text(encoding="utf-8"))
    assert d["version"] == 1
    assert d["split"] == "train"
    assert d["count"] == 3
    assert d["source_total"] == 4
    assert d["sampling"] == "equidistant"
    assert d["imgsz"] == 64
    assert len(d["sha256"]) == 16
    # 清单要入库、要在别的机器上还原，里面不许出现绝对路径
    assert all(not Path(f).is_absolute() for f in d["files"])
    assert all(f.startswith("images/train/") for f in d["files"])


def test_written_list_is_loadable_in_strict_mode(tool, tiny_yolo, tmp_path: Path):
    """清单写出来就得能被 strict 模式读回：strict 缺图直接抛，不静默换图。"""
    out = tmp_path / "calib.json"
    assert tool.main(_args(tiny_yolo, out)) == 0
    paths, meta = load_sample_list(out, root=tiny_yolo.root, strict=True)
    assert len(paths) == 3
    assert all(p.is_file() for p in paths)
    assert meta["split"] == "train"


def test_main_is_idempotent(tool, tiny_yolo, tmp_path: Path, capsys):
    """指纹相同就跳过：重写只会让 created_utc 变，给 git 造无意义 diff。"""
    out = tmp_path / "calib.json"
    assert tool.main(_args(tiny_yolo, out)) == 0
    first = out.read_text(encoding="utf-8")
    capsys.readouterr()
    assert tool.main(_args(tiny_yolo, out)) == 0
    out_text = capsys.readouterr().out
    assert "指纹一致" in out_text
    assert out.read_text(encoding="utf-8") == first


def test_main_warns_about_thin_class_coverage(tool, tiny_yolo, tmp_path: Path, capsys):
    """每类不到 3 张就打 warn：校准集"看着够多、其实全落在两三个类上"是常见坑。"""
    out = tmp_path / "calib.json"
    assert tool.main(_args(tiny_yolo, out)) == 0
    assert "不到 3 张" in capsys.readouterr().out


def test_main_dry_run_writes_nothing(tool, tiny_yolo, tmp_path: Path, capsys):
    out = tmp_path / "calib.json"
    assert tool.main(_args(tiny_yolo, out, "--dry-run")) == 0
    text = capsys.readouterr().out
    assert "--dry-run" in text
    assert not out.exists()


def test_main_refuses_to_overwrite_different_fingerprint(tool, tiny_yolo, tmp_path: Path, capsys):
    out = tmp_path / "calib.json"
    assert tool.main(_args(tiny_yolo, out)) == 0
    old_sha = json.loads(out.read_text(encoding="utf-8"))["sha256"]
    _add_train_image(tiny_yolo, "zz__000")
    capsys.readouterr()

    assert tool.main(_args(tiny_yolo, out)) == 3
    text = capsys.readouterr().out
    assert "指纹不同" in text and "--force" in text
    assert json.loads(out.read_text(encoding="utf-8"))["sha256"] == old_sha

    assert tool.main(_args(tiny_yolo, out, "--force")) == 0
    text = capsys.readouterr().out
    assert "--force" in text
    new_sha = json.loads(out.read_text(encoding="utf-8"))["sha256"]
    assert new_sha != old_sha


def test_main_reports_unreadable_existing_list(tool, tiny_yolo, tmp_path: Path, capsys):
    out = tmp_path / "calib.json"
    out.write_text("{ not json", encoding="utf-8")
    rc = tool.main(_args(tiny_yolo, out))
    assert rc == 3
    text = capsys.readouterr().out
    assert "读不动" in text and "--force" in text


def test_fingerprint_only_depends_on_paths_and_order(tool, tiny_yolo):
    """指纹是"口径身份证"：同一批图同一顺序，换机器换时间都得一样。"""
    from induvision.deploy import sample_fingerprint, sample_paths, read_split_images

    allp = read_split_images(tiny_yolo.root / "data.yaml", "train")
    picked = sample_paths(allp, 3)
    a = sample_fingerprint(picked, tiny_yolo.root)
    b = sample_fingerprint(picked, tiny_yolo.root)
    assert a == b and len(a) == 16
    # 换顺序 = 换口径（align/bench 取前几张会跟着变），指纹必须能反映
    assert sample_fingerprint(list(reversed(picked)), tiny_yolo.root) != a