"""tools/eval_seg.py 的单测：纯 CPU、不装 torch、不碰真实 runs/。

评测脚本本身要跑 val 必须有 torch+GPU，CI 里没有；所以这里只测**不需要模型的那半**：
逐类指标拆解、Markdown 渲染、run 名推断、路径校验、数据集环境指纹读取。
真正跑模型的那步靠 `--dry-run` 断言"参数解析与路径推导都对了、但一个字节都没写"。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import load_tool


@pytest.fixture
def tool():
    return load_tool("eval_seg")


class FakeMetric:
    """顶着 ultralytics Metric 的接口，只实现 _per_class 用到的那几个成员。"""

    def __init__(self, values):
        # values: {class_id: (P, R, mAP50, mAP50-95)}
        self._values = values
        self.ap_class_index = sorted(values)

    def class_result(self, i):
        return self._values[i]


def test_per_class_fills_missing_classes_with_none(tool):
    metric = FakeMetric({0: (0.9, 0.6, 0.68, 0.41), 2: (0.5, 0.4, 0.45, 0.2)})
    rows = tool._per_class(metric, ["bottle", "cable", "tile"])
    assert [r["class"] for r in rows] == ["bottle", "cable", "tile"]
    assert rows[0]["mAP50-95"] == pytest.approx(0.41)
    # 中间那类在这个 split 里没有实例：必须是 None，不能被填成 0
    # 填 0 会把"没样本"和"全错"混为一谈，逐类表直接失去诊断价值
    assert rows[1] == {"class_id": 1, "class": "cable", "P": None, "R": None,
                       "mAP50": None, "mAP50-95": None}
    assert rows[2]["class_id"] == 2


def test_per_class_handles_empty_index(tool):
    assert tool._per_class(FakeMetric({}), ["a"]) == [
        {"class_id": 0, "class": "a", "P": None, "R": None, "mAP50": None, "mAP50-95": None}
    ]


def test_md_table_shape_and_dash_for_none(tool):
    rows = tool._per_class(FakeMetric({0: (0.5, 0.4, 0.3, 0.2)}), ["bottle", "cable"])
    lines = tool._md_table(rows, "mask 逐类（split=test）")
    assert lines[0] == "**mask 逐类（split=test）**"
    assert lines[2].startswith("| 类 | P | R | mAP50 | mAP50-95 |")
    body = [l for l in lines if l.startswith("| bottle") or l.startswith("| cable")]
    assert len(body) == 2
    assert "| 0.2000 |" in body[0]
    assert "—" in body[1]


def test_env_fingerprint_is_self_describing(tool):
    fp = tool._env_fingerprint()
    assert fp["python"] and fp["platform"]
    # CI 里没装 torch/cv2 时也要有键，值写成 absent 而不是 KeyError
    for k in ("numpy", "cv2", "torch", "ultralytics"):
        assert k in fp and isinstance(fp[k], str) and fp[k]


def test_dataset_env_reads_fingerprint_next_to_data_yaml(tool, tmp_path: Path):
    d = tmp_path / "ds"
    d.mkdir()
    (d / "data.yaml").write_text("path: x\n", encoding="utf-8")
    env = {"python": "3.12.3", "cv2": "4.10.0"}
    (d / "dataset_stats.json").write_text(json.dumps({"env": env}), encoding="utf-8")
    assert tool._dataset_env(d / "data.yaml") == env


def test_dataset_env_falls_back_to_synth_report(tool, tmp_path: Path):
    d = tmp_path / "ds"
    d.mkdir()
    env = {"python": "3.12.3", "cv2": "5.0.0"}
    (d / "synth_report.json").write_text(json.dumps({"env": env}), encoding="utf-8")
    assert tool._dataset_env(d / "data.yaml") == env


def test_dataset_env_returns_none_when_absent(tool, tmp_path: Path):
    assert tool._dataset_env(tmp_path / "nope.yaml") is None


def _pair(tmp_path: Path, run: str = "baseline15"):
    """搭一个假的 <run>/weights/best.pt + data.yaml，只测路径逻辑。"""
    w = tmp_path / "runs" / "segment" / run / "weights" / "best.pt"
    w.parent.mkdir(parents=True)
    w.write_bytes(b"fake")
    d = tmp_path / "out" / "ds"
    d.mkdir(parents=True)
    (d / "data.yaml").write_text("path: x\nnames:\n  0: bottle\n", encoding="utf-8")
    return w, d


def test_dry_run_infers_run_name_from_weights_path(tool, tmp_path: Path, capsys):
    w, d = _pair(tmp_path, run="baseline15")
    rc = tool.main(["--weights", str(w), "--data", str(d), "--split", "test", "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "json -> " in out
    # 从 .../segment/baseline15/weights/best.pt 反推出 run 名
    assert Path(out.split("json -> ")[1].splitlines()[0].strip()).name == "eval_test.json"
    assert "baseline15" in out
    assert "未评测" in out


def test_dry_run_writes_nothing(tool, tmp_path: Path):
    w, d = _pair(tmp_path)
    before = sorted(p.name for p in w.parent.parent.iterdir())
    assert tool.main(["--weights", str(w), "--data", str(d), "--dry-run"]) == 0
    assert sorted(p.name for p in w.parent.parent.iterdir()) == before


def test_explicit_name_wins_over_inference(tool, tmp_path: Path, capsys):
    w, d = _pair(tmp_path, run="baseline15")
    out_json = tmp_path / "custom" / "e.json"
    rc = tool.main(["--weights", str(w), "--data", str(d), "--name", "myrun",
                    "--json-out", str(out_json), "--dry-run"])
    assert rc == 0
    assert str(out_json) in capsys.readouterr().out


def test_missing_weights_returns_2(tool, tmp_path: Path, capsys):
    _, d = _pair(tmp_path)
    rc = tool.main(["--weights", str(tmp_path / "nope.pt"), "--data", str(d), "--dry-run"])
    assert rc == 2
    assert "不存在" in capsys.readouterr().out


def test_missing_data_yaml_returns_2(tool, tmp_path: Path, capsys):
    w, _ = _pair(tmp_path)
    rc = tool.main(["--weights", str(w), "--data", str(tmp_path / "nope.yaml"), "--dry-run"])
    assert rc == 2
    assert "不存在" in capsys.readouterr().out


def test_split_is_restricted_to_val_and_test(tool, tmp_path: Path):
    w, d = _pair(tmp_path)
    # train split 参与过训练，拿它报成绩是自欺欺人，CLI 层面就不给这个选项
    with pytest.raises(SystemExit):
        tool.main(["--weights", str(w), "--data", str(d), "--split", "train", "--dry-run"])
