"""tools/export_onnx.py 的单测：只测不需要 torch 的那半。

真导出要 torch + ultralytics（CI 刻意不装），所以这里覆盖：路径与命名推导、
CLI 口径（opset / dynamic / simplify / half 是否正确落到配置里）、ONNX 图的静态
体检解析、checker 失败时的收口。真导出那步靠 ``--dry-run`` 断言"配置对、但一个字节没写"。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import load_tool


@pytest.fixture
def tool():
    return load_tool("export_onnx")


class _Dim:
    """仿 onnx 的 TensorShapeProto.Dimension：要么符号维，要么静态维。"""

    def __init__(self, param="", value=0):
        self.dim_param = param
        self.dim_value = value


def _fake_pt(tmp_path: Path, run: str = "baseline15") -> Path:
    w = tmp_path / "runs" / "segment" / run / "weights" / "best.pt"
    w.parent.mkdir(parents=True)
    w.write_bytes(b"fake-weights")
    return w


def _tiny_onnx(path: Path, *, batch_dynamic: bool = True, opset: int = 17) -> Path:
    """造一个最小但合法的 ONNX（Identity 图），给静态体检的解析逻辑当真输入。"""
    import onnx
    from onnx import TensorProto, helper

    b = "batch" if batch_dynamic else 1
    x = helper.make_tensor_value_info("images", TensorProto.FLOAT, [b, 3, 8, 8])
    y = helper.make_tensor_value_info("output0", TensorProto.FLOAT, [b, 3, 8, 8])
    node = helper.make_node("Identity", ["images"], ["output0"])
    model = helper.make_model(helper.make_graph([node], "g", [x], [y]),
                              opset_imports=[helper.make_opsetid("", opset)])
    model.ir_version = 8
    model.producer_name = "pytest"
    model.producer_version = "0"
    onnx.checker.check_model(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path))
    return path


# ------------------------------------------------------------------ 纯函数

def test_dim_to_str_shows_symbol_for_dynamic_and_number_for_static(tool):
    assert tool._dim_to_str(_Dim(param="batch")) == "batch"
    assert tool._dim_to_str(_Dim(value=640)) == "640"
    assert tool._dim_to_str(_Dim()) == "?"


def test_infer_run_name_from_weights_path(tool, tmp_path: Path):
    w = _fake_pt(tmp_path, run="baseline15")
    assert tool._infer_run_name(w, None) == "baseline15"


def test_infer_run_name_explicit_wins(tool, tmp_path: Path):
    w = _fake_pt(tmp_path)
    assert tool._infer_run_name(w, "myrun") == "myrun"


def test_infer_run_name_falls_back_to_stem(tool, tmp_path: Path):
    w = tmp_path / "somewhere" / "last.pt"
    w.parent.mkdir(parents=True)
    assert tool._infer_run_name(w, None) == "last"


def test_default_out_path_is_weights_dir_with_run_and_tag(tool):
    p = tool.default_out_path("baseline15", "fp16")
    assert p.name == "baseline15_fp16.onnx"
    assert p.parent.name == "weights"
    assert p.parent.parent == tool.REPO_ROOT


# ------------------------------------------------------------------ CLI

def test_dry_run_prints_config_and_writes_nothing(tool, tmp_path: Path, capsys):
    w = _fake_pt(tmp_path)
    before = sorted(p.name for p in w.parent.iterdir())
    rc = tool.main(["--weights", str(w), "--dry-run"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "未导出" in out
    cfg = json.loads(out.split("[cfg  ] ")[1].splitlines()[0])
    assert cfg["opset"] == 17 and cfg["dynamic"] is True and cfg["simplify"] is True
    assert cfg["half"] is False and cfg["imgsz"] == 640
    assert cfg["run"] == "baseline15" and cfg["tag"] == "fp32"
    assert cfg["onnx_out"].endswith("baseline15_fp32.onnx")
    assert cfg["json_out"].endswith(str(Path("baseline15") / "export_onnx_fp32.json"))
    assert sorted(p.name for p in w.parent.iterdir()) == before


def test_dry_run_no_dynamic_turns_dynamic_off(tool, tmp_path: Path, capsys):
    w = _fake_pt(tmp_path)
    tool.main(["--weights", str(w), "--no-dynamic", "--dry-run"])
    cfg = json.loads(capsys.readouterr().out.split("[cfg  ] ")[1].splitlines()[0])
    assert cfg["dynamic"] is False


def test_dry_run_no_simplify_and_half_and_tag(tool, tmp_path: Path, capsys):
    w = _fake_pt(tmp_path)
    tool.main(["--weights", str(w), "--no-simplify", "--half", "--tag", "fp16", "--dry-run"])
    cfg = json.loads(capsys.readouterr().out.split("[cfg  ] ")[1].splitlines()[0])
    assert cfg["simplify"] is False and cfg["half"] is True
    assert cfg["onnx_out"].endswith("baseline15_fp16.onnx")
    assert cfg["json_out"].endswith("export_onnx_fp16.json")


def test_dry_run_opset_and_imgsz_are_configurable(tool, tmp_path: Path, capsys):
    w = _fake_pt(tmp_path)
    tool.main(["--weights", str(w), "--opset", "13", "--imgsz", "512", "--dry-run"])
    cfg = json.loads(capsys.readouterr().out.split("[cfg  ] ")[1].splitlines()[0])
    assert cfg["opset"] == 13 and cfg["imgsz"] == 512


def test_explicit_out_and_json_paths_win(tool, tmp_path: Path, capsys):
    w = _fake_pt(tmp_path)
    out = tmp_path / "custom" / "model.onnx"
    js = tmp_path / "custom" / "meta.json"
    rc = tool.main(["--weights", str(w), "--out", str(out), "--json-out", str(js), "--dry-run"])
    assert rc == 0
    # 配置是 json.dumps 打出来的，Windows 路径里的反斜杠被转义成 \\，
    # 拿裸字符串做子串匹配必然失败。解析回来比 Path 才是比"路径有没有被尊重"。
    cfg = json.loads(capsys.readouterr().out.split("[cfg  ] ")[1].splitlines()[0])
    assert Path(cfg["onnx_out"]) == out
    assert Path(cfg["json_out"]) == js


def test_missing_weights_returns_2(tool, tmp_path: Path, capsys):
    assert tool.main(["--weights", str(tmp_path / "nope.pt"), "--dry-run"]) == 2
    assert "不存在" in capsys.readouterr().out


# ------------------------------------------------------- 静态体检（需要 onnx）

def test_inspect_onnx_reads_graph_and_flags_dynamic_batch(tool, tmp_path: Path):
    onnx = pytest.importorskip("onnx")  # noqa: F841 - CI 不装 onnx 时整条跳过
    p = _tiny_onnx(tmp_path / "m.onnx", batch_dynamic=True, opset=17)
    g = tool.inspect_onnx(p)
    assert g["opsets"] == {"ai.onnx": 17}
    assert g["batch_dynamic"] is True
    assert g["inputs"][0]["name"] == "images"
    assert g["inputs"][0]["shape"] == ["batch", "3", "8", "8"]
    assert g["inputs"][0]["dynamic_axes"] == [0]
    assert g["outputs"][0]["name"] == "output0"
    assert g["n_nodes"] == 1 and g["op_types"] == ["Identity"]
    assert g["size_bytes"] == p.stat().st_size > 0
    json.dumps(g)  # 必须能直接入库


def test_inspect_onnx_detects_static_batch(tool, tmp_path: Path):
    pytest.importorskip("onnx")
    p = _tiny_onnx(tmp_path / "m_static.onnx", batch_dynamic=False)
    g = tool.inspect_onnx(p)
    assert g["batch_dynamic"] is False
    assert g["inputs"][0]["shape"][0] == "1"
    assert g["inputs"][0]["dynamic_axes"] == []


def test_check_onnx_passes_on_valid_model(tool, tmp_path: Path):
    pytest.importorskip("onnx")
    p = _tiny_onnx(tmp_path / "ok.onnx")
    assert tool.check_onnx(p) == {"ok": True, "error": None}


def test_check_onnx_reports_error_instead_of_raising(tool, tmp_path: Path):
    pytest.importorskip("onnx")
    p = tmp_path / "broken.onnx"
    p.write_bytes(b"not-an-onnx-model")
    res = tool.check_onnx(p)
    assert res["ok"] is False
    assert isinstance(res["error"], str) and res["error"]