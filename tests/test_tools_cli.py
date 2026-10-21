"""命令行工具的端到端测试：不联网、不依赖真实数据集、不依赖 torch。

CI 里跑的正是这条链路：合成 MVTec 树 -> verify -> convert -> 检查产物。
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS = REPO_ROOT / "tools"


def _load_tool(name: str):
    """按文件路径加载 tools/ 下的脚本模块（tools 不是包）。"""
    path = TOOLS / f"{name}.py"
    assert path.is_file(), f"缺少工具脚本 {path}"
    spec = importlib.util.spec_from_file_location(f"tools_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def fake_root(tmp_path: Path) -> Path:
    from conftest import build_mvtec_tree

    return build_mvtec_tree(tmp_path / "mvtec_raw")


def test_make_fake_mvtec_builds_official_layout(tmp_path: Path):
    tool = _load_tool("make_fake_mvtec")
    out = tmp_path / "fake"
    rc = tool.main(["--out", str(out), "--categories", "bottle"])
    assert rc == 0
    assert (out / "bottle" / "train" / "good" / "000.png").is_file()
    assert (out / "bottle" / "test" / "good" / "000.png").is_file()
    gt = sorted(p.name for p in (out / "bottle" / "ground_truth" / "broken_large").iterdir())
    assert gt and all(n.endswith("_mask.png") for n in gt)


def test_make_fake_mvtec_rejects_unknown_category(tmp_path: Path):
    tool = _load_tool("make_fake_mvtec")
    assert tool.main(["--out", str(tmp_path / "x"), "--categories", "not_a_category"]) == 2


def test_verify_only_is_local_and_offline(fake_root: Path, capsys):
    """--verify-only 必须在没有 zip、没有网络的情况下也能成功。

    回归点：早期实现把 verify_only 判断放在 run_manual 的"找 zip"之后，
    导致纯本地校验反而要求先下载 zip。
    """
    tool = _load_tool("download_mvtec")
    assert not list(fake_root.glob("*.zip"))
    rc = tool.main(
        ["--verify-only", "--dest", str(fake_root), "--categories", "bottle", "screw"]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "TOTAL" in out and "通过" in out


def test_convert_cli_writes_complete_dataset(fake_root: Path, tmp_path: Path):
    tool = _load_tool("convert_mvtec_to_yolo")
    out = tmp_path / "yolo"
    rc = tool.main(
        ["--src", str(fake_root), "--out", str(out), "--categories", "bottle", "screw", "--lenient"]
    )
    assert rc == 0
    for name in ("data.yaml", "dataset_stats.json", "manifest.csv"):
        assert (out / name).is_file(), f"缺少产物 {name}"

    stats = json.loads((out / "dataset_stats.json").read_text(encoding="utf-8"))
    n_img = sum(v["images"] for v in stats["splits"].values())
    n_lbl = sum(1 for p in (out / "labels").rglob("*.txt"))
    assert n_img == n_lbl == stats["totals"]["images_written"] > 0
    # speck 掩码只有 2x2 像素，应被 min_area 过滤 -> 触发丢弃路径
    assert stats["protocol"]["dropped_tiny_defect_total"] >= 1


def test_convert_cli_dry_run_writes_nothing(fake_root: Path, tmp_path: Path):
    tool = _load_tool("convert_mvtec_to_yolo")
    out = tmp_path / "dry"
    assert tool.main(["--src", str(fake_root), "--out", str(out), "--dry-run", "--lenient"]) == 0
    assert not (out / "data.yaml").exists()


def test_convert_cli_reports_missing_source(tmp_path: Path, capsys):
    tool = _load_tool("convert_mvtec_to_yolo")
    rc = tool.main(["--src", str(tmp_path / "nope"), "--out", str(tmp_path / "y")])
    assert rc != 0
    assert "路径不存在" in capsys.readouterr().out

# --------------------------------------------------------------------------
# 训练入口：--dry-run 不导入 torch / ultralytics，CI 里也能守配置字典
# --------------------------------------------------------------------------


def _cfg_of(tool, data: Path, capsys, *extra) -> dict:
    capsys.readouterr()
    assert tool.main(["--data", str(data), "--dry-run", *extra]) == 0
    out = capsys.readouterr().out
    assert "[dry  ] 未训练" in out
    return json.loads(out.split("[cfg  ]", 1)[1].splitlines()[0])


def test_train_seg_dry_run_records_full_config(tmp_path: Path, capsys):
    tool = _load_tool("train_seg")
    data = tmp_path / "data.yaml"
    data.write_text("names:\n  0: bottle\n", encoding="utf-8")
    cfg = _cfg_of(tool, data, capsys, "--name", "w2_focal", "--focal-gamma", "2.0", "--epochs", "3")
    assert cfg["data"] == str(data.resolve())
    assert cfg["name"] == "w2_focal" and cfg["epochs"] == 3
    assert cfg["focal_gamma"] == 2.0 and cfg["focal_alpha"] == 0.25
    assert cfg["seed"] == 0 and cfg["amp"] is True and cfg["project"] is None
    assert len(cfg["git_head"]) >= 7, "复现要靠 git_head，取不到就是环境问题"


def test_train_seg_dry_run_flag_combinations(tmp_path: Path, capsys):
    tool = _load_tool("train_seg")
    data = tmp_path / "data.yaml"
    data.write_text("names:\n  0: bottle\n", encoding="utf-8")

    assert _cfg_of(tool, data, capsys, "--no-amp")["amp"] is False
    off = _cfg_of(tool, data, capsys)  # focal 关闭时 alpha 不该混进配置
    assert off["focal_gamma"] == 0.0 and off["focal_alpha"] is None
    assert _cfg_of(tool, data, capsys, "--focal-gamma", "1.5", "--focal-alpha", "0.5")["focal_alpha"] == 0.5
    assert _cfg_of(tool, data, capsys, "--project", "runs")["project"] == "runs"


def test_train_seg_requires_data(tmp_path: Path):
    tool = _load_tool("train_seg")
    with pytest.raises(SystemExit):
        tool.main(["--dry-run"])
