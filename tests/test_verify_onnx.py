"""tools/verify_onnx.py 的单测：只覆盖不需要 torch / onnxruntime 的那半。

三层校验里，第 1、3 层要真把两个后端跑起来（torch + ONNX Runtime），CI 刻意
不装这两个（见 requirements-ci.txt），所以这里盯住的是**判定与报告**这半：
设备号翻译、差值口径、参照读取、两条判据、Markdown 渲染、CLI 的路径推导与
错误收口、以及喂给两个后端的那份前处理输入。真跑一遍的实测结果
（verdict=pass：16 图逐张量对齐 / test 435 图端到端 / 30 帧延迟）记在
docs/deploy.md，不在这里重复断言。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from conftest import load_tool


@pytest.fixture
def tool():
    return load_tool("verify_onnx")


# W3 实跑出来的数字（runs/segment/baseline15/verify_onnx_fp32.json），
# 直接拿来当测试夹具：断言的就是"报告能正确渲染真实结果"。
ALIGN = {
    "n_images": 16, "seconds": 8.6, "output_count_pt": 2, "output_count_ort": 2,
    "worst_cosine": 1.0, "providers": ["CPUExecutionProvider"],
    "outputs": [
        {"index": 0, "ort_name": "output0", "shape": [1, 51, 8400],
         "max_abs": 0.00161743, "mean_abs": 1.94e-06, "rel_l2": 2.4e-07,
         "cosine": 1.0, "shape_match": True, "n_images": 16},
        {"index": 1, "ort_name": "output1", "shape": [1, 32, 160, 160],
         "max_abs": 2.241e-05, "mean_abs": 5.1e-07, "rel_l2": 1.61e-06,
         "cosine": 1.0, "shape_match": True, "n_images": 16},
    ],
}
VAL = {
    "seconds": 68.2, "split": "test", "batch": 8, "device": "cpu", "run_dir": "x",
    "box": {"P": 0.7123146, "R": 0.6192202, "mAP50": 0.6652539, "mAP50-95": 0.4228309},
    "mask": {"P": 0.7262395, "R": 0.6332178, "mAP50": 0.6733477, "mAP50-95": 0.3740504},
}
REFERENCE = {
    "path": "runs/segment/baseline15/eval_test.json", "device": "0",
    "git_head": "5dc2d20", "seconds": 26.9,
    "summary": {
        "box": {"P": 0.7123327, "R": 0.6191668, "mAP50": 0.6652458, "mAP50-95": 0.4229209},
        "mask": {"P": 0.7245576, "R": 0.6346151, "mAP50": 0.6729925, "mAP50-95": 0.3739910},
    },
}
BENCH = {
    "n_frames": 30, "warmup": 3, "batch": 1,
    "ort": {"count": 30, "mean_ms": 98.138, "p50_ms": 97.079, "p90_ms": 100.664,
            "p95_ms": 102.25, "p99_ms": 118.62, "min_ms": 94.051, "max_ms": 124.839,
            "fps": 10.19, "providers": ["CPUExecutionProvider"]},
    "torch": {"count": 30, "mean_ms": 16.078, "p50_ms": 16.299, "p90_ms": 17.505,
              "p95_ms": 18.367, "p99_ms": 19.018, "min_ms": 13.084, "max_ms": 19.174,
              "fps": 62.2, "device": "cuda:0"},
}
MD_CFG = {"run": "baseline15", "tag": "fp32", "split": "test",
          "align_device": "cpu", "imgsz": 640}


def _fake_run(tmp_path: Path, run: str = "fakerun", *, tag: str = "fp32",
              n_test_images: int = 3, with_onnx: bool = True) -> dict:
    """搭一套 verify_onnx 认得的最小输入：权重 + data.yaml + onnx + test 图。

    刻意不用 ``baseline15`` 当 run 名：那会让"默认 onnx 路径"指到仓库里真实存在的
    ``weights/baseline15_fp32.onnx``，测试在有没有这个文件的机器上行为就不一样了。
    """
    import cv2

    w = tmp_path / "runs" / "segment" / run / "weights" / "best.pt"
    w.parent.mkdir(parents=True, exist_ok=True)
    w.write_bytes(b"fake-weights")

    ds = tmp_path / "out" / "yolo_fake"
    img_dir = ds / "images" / "test"
    img_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for i in range(n_test_images):
        assert cv2.imwrite(str(img_dir / f"{i:03d}.png"),
                           rng.integers(0, 255, (48, 64, 3), dtype=np.uint8))
    (ds / "data.yaml").write_text(
        "path: %s\ntrain: images/train\nval: images/val\ntest: images/test\n"
        % ds.resolve().as_posix(), encoding="utf-8")

    onnx = tmp_path / "weights" / f"{run}_{tag}.onnx"
    if with_onnx:
        onnx.parent.mkdir(parents=True, exist_ok=True)
        onnx.write_bytes(b"fake-onnx")
    return {"weights": w, "data": ds / "data.yaml", "onnx": onnx, "run": run,
            "run_dir": tmp_path / "runs" / "segment" / run, "images": img_dir}


def _stdout(capsys):
    return capsys.readouterr().out


def _cfg(capsys) -> dict:
    """把 ``[cfg  ] `` 那行 JSON 解析回来（同时清空 capsys 缓冲）。"""
    return json.loads(_stdout(capsys).split("[cfg  ] ")[1].splitlines()[0])


# ------------------------------------------------------------- 设备号翻译

def test_torch_device_translates_bare_index(tool):
    """ultralytics 写 ``device="0"``，torch 只认 ``cuda:0``（踩过 Invalid device string）。"""
    assert tool._torch_device("0") == "cuda:0"
    assert tool._torch_device(0) == "cuda:0"
    assert tool._torch_device("1") == "cuda:1"


def test_torch_device_passes_named_devices_through(tool):
    assert tool._torch_device("cpu") == "cpu"
    assert tool._torch_device("cuda:1") == "cuda:1"
    assert tool._torch_device("mps") == "mps"


def test_torch_device_strips_whitespace(tool):
    assert tool._torch_device("  0  ") == "cuda:0"


# ------------------------------------------------------------------- 差值

def test_delta_rounds_to_six(tool):
    assert tool._delta(0.3739910, 0.3740504) == pytest.approx(5.9e-05)
    assert tool._delta(0.4, 0.35) == pytest.approx(-0.05)


def test_delta_returns_none_when_either_side_missing(tool):
    assert tool._delta(None, 0.37) is None
    assert tool._delta(0.37, None) is None
    assert tool._delta(None, None) is None


# --------------------------------------------------------------- 参照读取

def _write_ref(run_dir: Path, split: str, payload) -> Path:
    p = run_dir / f"eval_{split}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(payload if isinstance(payload, str)
                 else json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return p


def test_load_reference_reads_summary(tool, tmp_path: Path):
    _write_ref(tmp_path, "test", {"summary": REFERENCE["summary"], "device": "0",
                                  "git_head": "5dc2d20", "seconds": 26.9})
    ref = tool.load_reference(tmp_path, "test")
    assert ref is not None
    assert ref["device"] == "0" and ref["git_head"] == "5dc2d20"
    assert ref["summary"]["mask"]["mAP50-95"] == pytest.approx(0.3739910)
    assert Path(ref["path"]).name == "eval_test.json"


def test_load_reference_missing_file_returns_none(tool, tmp_path: Path):
    assert tool.load_reference(tmp_path, "test") is None


def test_load_reference_broken_json_returns_none(tool, tmp_path: Path):
    _write_ref(tmp_path, "test", "{ not json")
    assert tool.load_reference(tmp_path, "test") is None


# ------------------------------------------------------- 第 1 层判据

def test_check_align_passes_on_measured_fp32_numbers(tool):
    ok, why = tool.check_align(ALIGN, min_cosine=0.9999, max_rel_l2=1e-3)
    assert ok is True and why == []


def test_check_align_flags_low_cosine(tool):
    a = json.loads(json.dumps(ALIGN))
    a["worst_cosine"] = 0.91
    ok, why = tool.check_align(a, 0.9999, 1e-3)
    assert ok is False and any("cosine" in w for w in why)


def test_check_align_flags_rel_l2_over_budget(tool):
    a = json.loads(json.dumps(ALIGN))
    a["outputs"][1]["rel_l2"] = 0.02
    ok, why = tool.check_align(a, 0.9999, 1e-3)
    assert ok is False and any("out1" in w and "rel_l2" in w for w in why)


def test_check_align_flags_shape_mismatch(tool):
    a = json.loads(json.dumps(ALIGN))
    a["outputs"][0]["shape_match"] = False
    ok, why = tool.check_align(a, 0.9999, 1e-3)
    assert ok is False and any("形状" in w for w in why)


def test_check_align_flags_output_count_mismatch(tool):
    """输出个数不同说明图被改过（例如 NMS 折进模型），逐张量比差已经没意义。"""
    a = json.loads(json.dumps(ALIGN))
    a["output_count_ort"] = 1
    ok, why = tool.check_align(a, 0.9999, 1e-3)
    assert ok is False and any("输出数不一致" in w for w in why)


def test_check_align_collects_every_reason(tool):
    a = json.loads(json.dumps(ALIGN))
    a["worst_cosine"] = 0.5
    a["output_count_pt"] = 3
    a["outputs"][0]["rel_l2"] = 9.0
    ok, why = tool.check_align(a, 0.9999, 1e-3)
    assert ok is False and len(why) == 3


def test_check_align_without_outputs_still_needs_matching_counts(tool):
    ok, why = tool.check_align({"output_count_pt": 0, "output_count_ort": 0,
                                "worst_cosine": 1.0, "outputs": []}, 0.9999, 1e-3)
    assert ok is True and why == []


# ------------------------------------------------------- 第 2 层判据

def test_check_val_tolerates_measured_drift(tool):
    """W3 实测 .pt 与 .onnx 的 mask mAP50-95 差 +0.0001，必须判过。"""
    ok, why = tool.check_val(0.3740504 - 0.3739910, 0.01)
    assert ok is True and why == []


def test_check_val_flags_drop_beyond_budget(tool):
    ok, why = tool.check_val(-0.05, 0.01)
    assert ok is False and any("±" in w for w in why)


def test_check_val_flags_gain_beyond_budget(tool):
    """涨太多同样要报：多半是评测口径变了（split / imgsz / 后处理），不是模型变好。"""
    ok, _ = tool.check_val(0.09, 0.01)
    assert ok is False


def test_check_val_without_reference_passes(tool):
    ok, why = tool.check_val(None, 0.01)
    assert ok is True and why == []


# ------------------------------------------------------- 环境指纹与 git

def test_env_fingerprint_lists_absent_backends_without_raising(tool):
    fp = tool._env_fingerprint()
    assert fp["python"] and fp["platform"]
    for mod in ("numpy", "cv2", "torch", "ultralytics", "onnx", "onnxruntime"):
        assert mod in fp and isinstance(fp[mod], str) and fp[mod]
    json.dumps(fp)  # 必须能直接入库


def test_git_head_is_short_sha_or_unknown(tool):
    head = tool._git_head()
    assert isinstance(head, str) and head
    assert head == "unknown" or all(c in "0123456789abcdef" for c in head)


# ----------------------------------------------------------- Markdown 渲染

def test_md_renders_all_three_layers(tool):
    text = "\n".join(tool._md({"align": ALIGN, "val": VAL, "reference": REFERENCE,
                               "bench": BENCH}, MD_CFG))
    assert "baseline15" in text and "fp32" in text
    assert "第 1 层" in text and "第 2 层" in text and "第 3 层" in text
    assert "| `output0` | [1, 51, 8400] | 1.617e-03 |" in text
    assert "| `output1` | [1, 32, 160, 160] | 2.241e-05 |" in text
    assert "| mask mAP50-95 | 0.3740 | 0.3741 | +0.0001 |" in text
    assert "| box mAP50-95 | 0.4229 | 0.4228 | -0.0001 |" in text
    assert "ONNX Runtime" in text and "PyTorch" in text
    assert "98.14 ms" in text and "16.08 ms" in text


def test_md_val_table_uses_dash_without_reference(tool):
    text = "\n".join(tool._md({"align": ALIGN, "val": VAL, "reference": None}, MD_CFG))
    assert "| mask mAP50-95 | — | 0.3741 | — |" in text


def test_md_skips_backend_without_latency(tool):
    b = {"n_frames": 30, "warmup": 3, "batch": 1, "ort": BENCH["ort"]}
    text = "\n".join(tool._md({"bench": b}, MD_CFG))
    assert "ONNX Runtime" in text and "PyTorch" not in text


def test_md_header_only_when_no_layers(tool):
    lines = tool._md({}, MD_CFG)
    assert len(lines) == 2 and lines[0].startswith("**ONNX 对齐校验")


# ------------------------------------------------------------------- CLI

def test_dry_run_prints_config_and_writes_nothing(tool, tmp_path: Path, capsys):
    f = _fake_run(tmp_path)
    rc = tool.main(["--weights", str(f["weights"]), "--data", str(f["data"]),
                    "--onnx", str(f["onnx"]), "--dry-run"])
    text = _stdout(capsys)          # capsys 读一次就清空，先存下来
    cfg = json.loads(text.split("[cfg  ] ")[1].splitlines()[0])
    assert rc == 0
    assert "未校验" in text
    assert Path(cfg["weights"]) == f["weights"]
    assert Path(cfg["onnx"]) == f["onnx"]
    # JSON 默认落在仓库的 runs/segment/<run>/ 而不是 tmp_path：训练产物本来就在那里，
    # 校验结果要挨着 best.pt 放才对得上号（runs/ 不入库）。
    default_json = tool.REPO_ROOT / "runs" / "segment" / f["run"] / "verify_onnx_fp32.json"
    assert Path(cfg["json_out"]) == default_json
    assert not default_json.exists()


def test_dry_run_defaults_keep_align_on_cpu_and_skip_layers_2_3(tool, tmp_path: Path, capsys):
    """默认只跑第 1 层，且 PyTorch 侧钉在 CPU：这样差值只反映导出本身。"""
    f = _fake_run(tmp_path)
    tool.main(["--weights", str(f["weights"]), "--data", str(f["data"]),
               "--onnx", str(f["onnx"]), "--dry-run"])
    cfg = _cfg(capsys)
    assert cfg["align_device"] == "cpu" and cfg["align_n"] == 16
    assert cfg["bench_n"] == 0 and cfg["val"] is False
    assert cfg["split"] == "test" and cfg["imgsz"] == 640
    assert cfg["thresholds"] == {"min_cosine": 0.9999, "max_rel_l2": 1e-3,
                                 "max_map_delta": 0.01}


def test_dry_run_name_and_tag_change_paths(tool, tmp_path: Path, capsys):
    f = _fake_run(tmp_path, run="myrun", tag="fp16")
    tool.main(["--weights", str(f["weights"]), "--data", str(f["data"]),
               "--onnx", str(f["onnx"]), "--name", "myrun", "--tag", "fp16", "--dry-run"])
    cfg = _cfg(capsys)
    assert cfg["run"] == "myrun" and cfg["tag"] == "fp16"
    p = Path(cfg["json_out"])
    assert p.name == "verify_onnx_fp16.json" and p.parent.name == "myrun"


def test_dry_run_infers_run_from_weights_path(tool, tmp_path: Path, capsys):
    f = _fake_run(tmp_path, run="baseline99")
    tool.main(["--weights", str(f["weights"]), "--data", str(f["data"]),
               "--onnx", str(f["onnx"]), "--dry-run"])
    assert _cfg(capsys)["run"] == "baseline99"


def test_dry_run_thresholds_and_layers_are_configurable(tool, tmp_path: Path, capsys):
    f = _fake_run(tmp_path)
    tool.main(["--weights", str(f["weights"]), "--data", str(f["data"]),
               "--onnx", str(f["onnx"]), "--align-n", "4", "--align-device", "0",
               "--bench-n", "30", "--bench-device", "cpu", "--val", "--val-batch", "4",
               "--min-cosine", "0.99", "--max-rel-l2", "0.01", "--max-map-delta", "0.02",
               "--imgsz", "512", "--split", "val", "--dry-run"])
    cfg = _cfg(capsys)
    assert cfg["align_n"] == 4 and cfg["align_device"] == "0" and cfg["imgsz"] == 512
    assert cfg["bench_n"] == 30 and cfg["bench_device"] == "cpu"
    assert cfg["val"] is True and cfg["val_batch"] == 4 and cfg["split"] == "val"
    assert cfg["thresholds"] == {"min_cosine": 0.99, "max_rel_l2": 0.01,
                                 "max_map_delta": 0.02}


def test_explicit_json_out_wins(tool, tmp_path: Path, capsys):
    f = _fake_run(tmp_path)
    js = tmp_path / "custom" / "v.json"
    tool.main(["--weights", str(f["weights"]), "--data", str(f["data"]),
               "--onnx", str(f["onnx"]), "--json-out", str(js), "--dry-run"])
    # 路径走 json.dumps，反斜杠被转义，所以解析回来比 Path
    assert Path(_cfg(capsys)["json_out"]) == js


def test_missing_weights_returns_2(tool, tmp_path: Path, capsys):
    f = _fake_run(tmp_path)
    assert tool.main(["--weights", str(tmp_path / "nope.pt"), "--data", str(f["data"]),
                      "--onnx", str(f["onnx"]), "--dry-run"]) == 2
    assert "权重不存在" in _stdout(capsys)


def test_missing_data_returns_2(tool, tmp_path: Path, capsys):
    f = _fake_run(tmp_path)
    assert tool.main(["--weights", str(f["weights"]), "--data", str(tmp_path / "no.yaml"),
                      "--onnx", str(f["onnx"]), "--dry-run"]) == 2
    assert "data.yaml 不存在" in _stdout(capsys)


def test_missing_onnx_returns_2_and_points_to_export_tool(tool, tmp_path: Path, capsys):
    """不给 --onnx 时按 weights/<run>_<tag>.onnx 推导；推不出来就得指路去导出。"""
    f = _fake_run(tmp_path, run="fakerun", with_onnx=False)
    rc = tool.main(["--weights", str(f["weights"]), "--data", str(f["data"]), "--dry-run"])
    out = _stdout(capsys)
    assert rc == 2
    assert "ONNX 不存在" in out and "fakerun_fp32.onnx" in out
    assert "tools/export_onnx.py" in out


def test_empty_split_returns_2(tool, tmp_path: Path, capsys):
    f = _fake_run(tmp_path, n_test_images=0)
    rc = tool.main(["--weights", str(f["weights"]), "--data", str(f["data"]),
                    "--onnx", str(f["onnx"])])
    assert rc == 2
    assert "split 里没有图片" in _stdout(capsys)


# ------------------------------------------------------- 前处理输入装载

def test_load_inputs_preprocesses_and_skips_unreadable(tool, tmp_path: Path, capsys):
    """读不出来的图要跳过并告警，不能让整轮校验挂掉；读出来的必须是 CHW/float32/[0,1]。"""
    f = _fake_run(tmp_path, n_test_images=2)
    (f["images"] / "broken.png").write_bytes(b"not-an-image")
    paths = tool.read_split_images(f["data"], "test")
    assert len(paths) == 3
    xs, names = tool._load_inputs(paths, imgsz=64)
    out = _stdout(capsys)
    assert len(xs) == 2 and len(names) == 2
    assert "读不出来" in out and "broken.png" in out
    for x in xs:
        assert x.shape == (3, 64, 64) and x.dtype == np.float32
        assert 0.0 <= float(x.min()) and float(x.max()) <= 1.0


def test_load_inputs_returns_empty_when_nothing_readable(tool, tmp_path: Path, capsys):
    f = _fake_run(tmp_path, n_test_images=1)
    for p in f["images"].iterdir():
        p.write_bytes(b"junk")
    xs, names = tool._load_inputs(tool.read_split_images(f["data"], "test"), imgsz=64)
    assert xs == [] and names == []