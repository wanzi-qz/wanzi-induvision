# -*- coding: utf-8 -*-
"""tools/summarize_ablation.py 的单测：纯 CPU、不联网、不需要真跑过训练。"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

sa = None


def setup_module(_module):
    global sa
    from conftest import load_tool

    sa = load_tool("summarize_ablation")


CSV_HEADER = (
    "epoch,time,train/box_loss,metrics/precision(B),metrics/recall(B),"
    "metrics/mAP50(B),metrics/mAP50-95(B),metrics/precision(M),metrics/recall(M),"
    "metrics/mAP50(M),metrics/mAP50-95(M)\n"
)


def _row(epoch, mb, mm, pm=0.3, rm=0.4, m50=0.5, pb=0.3, rb=0.4, b50=0.5):
    return (f"{epoch},1.0,2.0,{pb},{rb},{b50},{mb},{pm},{rm},{m50},{mm}\n")


def write_run(root: Path, name: str, rows, *, cfg=None, args_yaml=None, header=CSV_HEADER):
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "results.csv").write_text(header + "".join(rows), encoding="utf-8")
    if cfg is not None:
        (d / "train_args.json").write_text(json.dumps(cfg), encoding="utf-8")
    if args_yaml is not None:
        (d / "args.yaml").write_text(args_yaml, encoding="utf-8")
    return d


BASE_CFG = {"seed": 0, "imgsz": 640, "batch": 8, "epochs": 100, "patience": 20,
            "amp": True, "focal_gamma": 0.0, "data": "/x/out/yolo_bottle_tb/data.yaml"}


@pytest.fixture
def two_runs(tmp_path: Path) -> Path:
    root = tmp_path / "segment"
    write_run(root, "b0_base", [_row(1, 0.10, 0.08), _row(2, 0.12, 0.20), _row(3, 0.11, 0.15)],
              cfg=dict(BASE_CFG))
    c = dict(BASE_CFG, focal_gamma=2.0, focal_alpha=0.25,
             data="/x/out/yolo_bottle_tb_cp/data.yaml")
    write_run(root, "b1_cp", [_row(1, 0.10, 0.09), _row(2, 0.13, 0.25)], cfg=c)
    return root


# ---------------------------------------------------------------- read_results

def test_read_results_sorts_and_typed(two_runs):
    rows = sa.read_results(two_runs / "b0_base" / "results.csv")
    assert [r["epoch"] for r in rows] == [1.0, 2.0, 3.0]
    assert rows[1]["metrics/mAP50-95(M)"] == pytest.approx(0.20)
    assert rows[0]["_time"] == pytest.approx(1.0)


def test_read_results_unsorted_input_gets_sorted(tmp_path):
    d = write_run(tmp_path, "r", [_row(3, 0.1, 0.3), _row(1, 0.1, 0.1)])
    rows = sa.read_results(d / "results.csv")
    assert [r["epoch"] for r in rows] == [1.0, 3.0]


def test_read_results_skips_rows_without_epoch(tmp_path):
    d = write_run(tmp_path, "r", [_row(1, 0.1, 0.1)])
    p = d / "results.csv"
    p.write_text(p.read_text(encoding="utf-8") + ",9,,,,,,,,,,\n", encoding="utf-8")
    rows = sa.read_results(p)
    assert [r["epoch"] for r in rows] == [1.0]


def test_read_results_tolerates_blank_metric_cells(tmp_path):
    d = write_run(tmp_path, "r", ["1,1.0,2.0,0.3,0.4,0.5,0.10,,,,\n"])
    rows = sa.read_results(d / "results.csv")
    assert rows[0]["metrics/mAP50-95(B)"] == pytest.approx(0.10)
    assert "metrics/mAP50-95(M)" not in rows[0]


# ------------------------------------------------------------------ read_config

def test_read_config_prefers_train_args_json(two_runs):
    cfg = sa.read_config(two_runs / "b0_base")
    assert cfg["_source"] == "train_args.json"
    assert cfg["focal_gamma"] == 0.0


def test_read_config_falls_back_to_args_yaml(tmp_path):
    d = write_run(tmp_path, "r", [_row(1, 0.1, 0.1)],
                  args_yaml="seed: 0\nimgsz: 640\nbatch: 8\nepochs: 100\npatience: 20\namp: true\n")
    cfg = sa.read_config(d)
    assert cfg["_source"] == "args.yaml"
    assert cfg["imgsz"] == 640
    assert "focal_gamma" not in cfg


def test_read_config_broken_json_falls_back(tmp_path):
    d = write_run(tmp_path, "r", [_row(1, 0.1, 0.1)], args_yaml="seed: 0\n")
    (d / "train_args.json").write_text("{not json", encoding="utf-8")
    assert sa.read_config(d)["_source"] == "args.yaml"


def test_read_config_missing(tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    assert sa.read_config(d) == {"_source": "missing"}


# ------------------------------------------------- dataset_name / focal_str

@pytest.mark.parametrize("data,expected", [
    ("/x/out/yolo_bottle_tb/data.yaml", "yolo_bottle_tb"),
    (r"C:\a\out\ds\data.yml", "ds"),
    ("/x/out/plain_dir", "plain_dir"),
    ("", "?"),
])
def test_dataset_name(data, expected):
    assert sa.dataset_name({"data": data}) == expected


def test_dataset_name_missing_key():
    assert sa.dataset_name({}) == "?"


@pytest.mark.parametrize("cfg,expected", [
    ({}, "关"),
    ({"focal_gamma": 0.0, "focal_alpha": 0.25}, "关"),
    ({"focal_gamma": 2.0, "focal_alpha": 0.25}, "γ=2 α=0.25"),
    ({"focal_gamma": 1.5}, "γ=1.5"),
    ({"focal_gamma": "nope"}, "关"),
])
def test_focal_str(cfg, expected):
    assert sa.focal_str(cfg) == expected


# -------------------------------------------------------------- summarize_run

def test_summarize_run_picks_best_by_metric(two_runs):
    rec = sa.summarize_run(two_runs / "b0_base", "metrics/mAP50-95(M)")
    assert rec["best_epoch"] == 2
    assert rec["best"]["metrics/mAP50-95(M)"] == pytest.approx(0.20)
    assert rec["last"]["epoch"] == 3.0
    assert rec["epochs_done"] == 3
    assert rec["dataset"] == "yolo_bottle_tb"
    assert rec["focal"] == "关"


def test_summarize_run_best_follows_metric_choice(two_runs):
    rec = sa.summarize_run(two_runs / "b0_base", "metrics/mAP50-95(B)")
    assert rec["best_epoch"] == 2


def test_summarize_run_none_without_csv(tmp_path):
    d = tmp_path / "nores"
    d.mkdir()
    assert sa.summarize_run(d, "metrics/mAP50-95(M)") is None


def test_summarize_run_none_for_header_only_csv(tmp_path):
    d = write_run(tmp_path, "r", [])
    assert sa.summarize_run(d, "metrics/mAP50-95(M)") is None


def test_summarize_run_metric_absent_falls_back_to_last(tmp_path):
    d = write_run(tmp_path, "r", ["1,1.0,2.0,0.3,0.4,0.5,0.10,,,,\n"])
    rec = sa.summarize_run(d, "metrics/mAP50-95(M)")
    assert rec["best_epoch"] == 1
    assert rec["best"]["metrics/mAP50-95(B)"] == pytest.approx(0.10)


def test_summarize_run_records_focal_and_protocol(two_runs):
    rec = sa.summarize_run(two_runs / "b1_cp", "metrics/mAP50-95(M)")
    assert rec["focal"] == "γ=2 α=0.25"
    assert rec["protocol"] == ("0", "640", "8", "100", "20", "True")


# ------------------------------------------------------------- check_protocol

def test_check_protocol_single_run_no_warning(two_runs):
    recs = [sa.summarize_run(two_runs / "b0_base", "metrics/mAP50-95(M)")]
    assert sa.check_protocol(recs) == []


def test_check_protocol_consistent(two_runs):
    recs = [sa.summarize_run(two_runs / n, "metrics/mAP50-95(M)") for n in ("b0_base", "b1_cp")]
    assert sa.check_protocol(recs) == []


def test_check_protocol_flags_outlier(tmp_path):
    root = tmp_path / "segment"
    for n in ("a", "b"):
        write_run(root, n, [_row(1, 0.1, 0.1)], cfg=dict(BASE_CFG))
    write_run(root, "odd", [_row(1, 0.1, 0.1)], cfg=dict(BASE_CFG, batch=16))
    recs = [sa.summarize_run(root / n, "metrics/mAP50-95(M)") for n in ("a", "b", "odd")]
    warns = sa.check_protocol(recs)
    assert warns
    assert any("odd" in w for w in warns)
    assert any("协议不一致" in w for w in warns)


def test_check_protocol_majority_is_the_reference(tmp_path):
    root = tmp_path / "segment"
    for n in ("a", "b", "c"):
        write_run(root, n, [_row(1, 0.1, 0.1)], cfg=dict(BASE_CFG))
    write_run(root, "odd", [_row(1, 0.1, 0.1)], cfg=dict(BASE_CFG, seed=7))
    recs = [sa.summarize_run(root / n, "metrics/mAP50-95(M)") for n in ("a", "b", "c", "odd")]
    warns = sa.check_protocol(recs)
    assert sum("例外" in w for w in warns) == 1


# ------------------------------------------------------------------ renderers

def test_to_markdown_shape(two_runs):
    recs = [sa.summarize_run(two_runs / n, "metrics/mAP50-95(M)") for n in ("b1_cp", "b0_base")]
    md = sa.to_markdown(recs, "metrics/mAP50-95(M)")
    lines = md.split("\n")
    assert lines[0].startswith("| 组 |")
    assert set(lines[1]) <= set("|- ")
    assert len(lines) == 4
    assert "b0_base" in lines[2] and "b1_cp" in lines[3]  # 按名字排序
    assert "**0.2000**" in lines[2]
    assert "γ=2 α=0.25" in lines[3]


def test_to_markdown_marks_non_default_metric_without_arrow(two_runs):
    recs = [sa.summarize_run(two_runs / "b0_base", "metrics/mAP50-95(B)")]
    assert "⬅" not in sa.to_markdown(recs, "metrics/mAP50-95(B)")


def test_to_markdown_duration_and_plan(two_runs):
    d = write_run(two_runs, "withtime", [_row(1, 0.1, 0.1)],
                  cfg=dict(BASE_CFG, seconds=90.0))
    md = sa.to_markdown([sa.summarize_run(d, "metrics/mAP50-95(M)")], "metrics/mAP50-95(M)")
    assert "1.5 min" in md
    assert "1/100" in md


def test_to_markdown_dash_for_missing_values(tmp_path):
    d = write_run(tmp_path, "r", [_row(1, 0.1, 0.1)], cfg={"epochs": 5})
    md = sa.to_markdown([sa.summarize_run(d, "metrics/mAP50-95(M)")], "metrics/mAP50-95(M)")
    assert "—" in md
    assert "?" in md


def test_to_text_contains_key_fields(two_runs):
    recs = [sa.summarize_run(two_runs / "b0_base", "metrics/mAP50-95(M)")]
    txt = sa.to_text(recs, "metrics/mAP50-95(M)")
    assert "[b0_base]" in txt
    assert "best@2" in txt
    assert "cfg_from=train_args.json" in txt


# --------------------------------------------------------------------- collect

def test_collect_skips_dirs_without_csv(two_runs, capsys):
    (two_runs / "junk").mkdir()
    recs = sa.collect(two_runs, None, "metrics/mAP50-95(M)")
    assert {r["name"] for r in recs} == {"b0_base", "b1_cp"}
    assert "junk" in capsys.readouterr().err


def test_collect_quiet_suppresses_warning(two_runs, capsys):
    (two_runs / "junk").mkdir()
    sa.collect(two_runs, None, "metrics/mAP50-95(M)", quiet=True)
    assert capsys.readouterr().err == ""


def test_collect_names_filter(two_runs):
    recs = sa.collect(two_runs, ["b1_cp"], "metrics/mAP50-95(M)")
    assert [r["name"] for r in recs] == ["b1_cp"]


def test_collect_names_filter_ignores_unknown(two_runs):
    assert sa.collect(two_runs, ["nope"], "metrics/mAP50-95(M)") == []


def test_collect_missing_root_exits(tmp_path):
    with pytest.raises(SystemExit):
        sa.collect(tmp_path / "nope", None, "metrics/mAP50-95(M)")


# ------------------------------------------------------------------------ main

def test_main_text_mode_exit_zero(two_runs, capsys):
    assert sa.main(["--runs-root", str(two_runs)]) == 0
    out = capsys.readouterr().out
    assert "b0_base" in out and "协议一致" in out


def test_main_markdown_mode(two_runs, capsys):
    assert sa.main(["--runs-root", str(two_runs), "--markdown"]) == 0
    assert capsys.readouterr().out.startswith("| 组 |")


def test_main_no_runs_returns_2(tmp_path, capsys):
    empty = tmp_path / "segment"
    empty.mkdir()
    assert sa.main(["--runs-root", str(empty)]) == 2
    assert "没有任何可汇总" in capsys.readouterr().err


def test_main_strict_flags_protocol_drift(tmp_path, capsys):
    root = tmp_path / "segment"
    for n in ("a", "b"):
        write_run(root, n, [_row(1, 0.1, 0.1)], cfg=dict(BASE_CFG))
    write_run(root, "odd", [_row(1, 0.1, 0.1)], cfg=dict(BASE_CFG, batch=4))
    assert sa.main(["--runs-root", str(root), "--strict", "--markdown"]) == 1
    assert "协议不一致" in capsys.readouterr().out


def test_main_strict_passes_when_consistent(two_runs):
    assert sa.main(["--runs-root", str(two_runs), "--strict"]) == 0


def test_main_names_filter(two_runs, capsys):
    assert sa.main(["--runs-root", str(two_runs), "--names", "b0_base"]) == 0
    out = capsys.readouterr().out
    assert "b0_base" in out and "b1_cp" not in out


def test_main_rejects_unknown_metric(two_runs):
    with pytest.raises(SystemExit):
        sa.main(["--runs-root", str(two_runs), "--metric", "metrics/nope"])
# ------------------------------------------------- 转换环境指纹交叉核对（W2 踩坑）

FP_A = {"python": "3.12.3", "numpy": "1.26.4", "cv2": "4.10.0", "platform": "Windows-11"}
FP_B = {"python": "3.12.5", "numpy": "2.5.2", "cv2": "5.0.0", "platform": "Windows-11"}


def make_dataset(tmp_path: Path, name: str, env=None, *, broken=False, no_file=False) -> Path:
    """造一个只有 data.yaml + dataset_stats.json 的假数据集目录。"""
    d = tmp_path / "out" / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "data.yaml").write_text("path: .\ntrain: images/train\n", encoding="utf-8")
    if not no_file:
        body = "{not json" if broken else json.dumps({"env": env} if env is not None else {})
        (d / "dataset_stats.json").write_text(body, encoding="utf-8")
    return d


def make_synth_dataset(tmp_path: Path, name: str, env=None, *, broken=False) -> Path:
    """Copy-Paste 合成出来的数据集：没有 dataset_stats.json，指纹在 synth_report.json。"""
    d = tmp_path / "out" / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "data.yaml").write_text("path: .\ntrain: images/train\n", encoding="utf-8")
    body = "{not json" if broken else json.dumps({"env": env} if env is not None else {})
    (d / "synth_report.json").write_text(body, encoding="utf-8")
    return d


def make_run_for(tmp_path: Path, runs_root: Path, run_name: str, ds_dir: Path, **cfg_over):
    cfg = dict(BASE_CFG, data=str(ds_dir / "data.yaml"), **cfg_over)
    return write_run(runs_root, run_name, [_row(1, 0.10, 0.10), _row(2, 0.12, 0.18)], cfg=cfg)


def test_dataset_env_reads_fingerprint(tmp_path):
    d = make_dataset(tmp_path, "ds", FP_A)
    assert sa.dataset_env({"data": str(d / "data.yaml")}) == FP_A


def test_dataset_env_accepts_dir_path(tmp_path):
    d = make_dataset(tmp_path, "ds", FP_A)
    assert sa.dataset_env({"data": str(d)}) == FP_A


def test_dataset_env_none_when_no_data_key(tmp_path):
    assert sa.dataset_env({}) is None
    assert sa.dataset_env({"data": ""}) is None


def test_dataset_env_none_when_stats_missing(tmp_path):
    d = make_dataset(tmp_path, "ds", FP_A, no_file=True)
    assert sa.dataset_env({"data": str(d / "data.yaml")}) is None


def test_dataset_env_none_when_json_broken(tmp_path):
    d = make_dataset(tmp_path, "ds", FP_A, broken=True)
    assert sa.dataset_env({"data": str(d / "data.yaml")}) is None


def test_dataset_env_none_when_field_absent_or_wrong_type(tmp_path):
    d1 = make_dataset(tmp_path, "ds1", None)
    assert sa.dataset_env({"data": str(d1 / "data.yaml")}) is None
    d2 = make_dataset(tmp_path, "ds2")
    (d2 / "dataset_stats.json").write_text(json.dumps({"env": "4.10"}), encoding="utf-8")
    assert sa.dataset_env({"data": str(d2 / "data.yaml")}) is None


def test_check_env_all_same_no_warning(tmp_path):
    runs = tmp_path / "segment"
    d = make_dataset(tmp_path, "ds", FP_A)
    for n in ("a", "b"):
        make_run_for(tmp_path, runs, n, d)
    recs = sa.collect(runs, None, "metrics/mAP50-95(M)", quiet=True)
    warns, notes = sa.check_env(recs)
    assert warns == [] and notes == []


def test_check_env_flags_cross_env_drift(tmp_path):
    """这就是 W2 真踩到的坑：base 环境 cv2 4.10 与 GPU 环境 cv2 5.0 转出的标签差约 1px。"""
    runs = tmp_path / "segment"
    da = make_dataset(tmp_path, "ds_base", FP_A)
    db = make_dataset(tmp_path, "ds_gpu", FP_B)
    make_run_for(tmp_path, runs, "b0_base", da)
    make_run_for(tmp_path, runs, "b4_nogood", db)
    recs = sa.collect(runs, None, "metrics/mAP50-95(M)", quiet=True)
    warns, notes = sa.check_env(recs)
    assert any("转换环境不一致" in w for w in warns)
    assert any("4.10.0" in w for w in warns) and any("5.0.0" in w for w in warns)
    assert any("b0_base" in w and "b4_nogood" in w for w in warns)
    assert notes == []


def test_check_env_unknown_is_note_not_warning(tmp_path):
    runs = tmp_path / "segment"
    d = make_dataset(tmp_path, "ds_old", None)
    make_run_for(tmp_path, runs, "legacy", d)
    recs = sa.collect(runs, None, "metrics/mAP50-95(M)", quiet=True)
    warns, notes = sa.check_env(recs)
    assert warns == []
    assert any("legacy" in n for n in notes)


def test_check_env_mixed_known_and_unknown(tmp_path):
    runs = tmp_path / "segment"
    d1 = make_dataset(tmp_path, "ds_new", FP_A)
    d2 = make_dataset(tmp_path, "ds_old", None)
    make_run_for(tmp_path, runs, "new_run", d1)
    make_run_for(tmp_path, runs, "old_run", d2)
    recs = sa.collect(runs, None, "metrics/mAP50-95(M)", quiet=True)
    warns, notes = sa.check_env(recs)
    assert warns == []                      # 只有一个已知指纹，谈不上"不一致"
    assert any("old_run" in n for n in notes)


def test_main_strict_fails_on_env_drift(tmp_path, capsys):
    runs = tmp_path / "segment"
    make_run_for(tmp_path, runs, "a", make_dataset(tmp_path, "ds_a", FP_A))
    make_run_for(tmp_path, runs, "b", make_dataset(tmp_path, "ds_b", FP_B))
    assert sa.main(["--runs-root", str(runs), "--strict"]) == 1
    assert "转换环境不一致" in capsys.readouterr().out


def test_main_no_env_check_suppresses(tmp_path, capsys):
    runs = tmp_path / "segment"
    make_run_for(tmp_path, runs, "a", make_dataset(tmp_path, "ds_a", FP_A))
    make_run_for(tmp_path, runs, "b", make_dataset(tmp_path, "ds_b", FP_B))
    assert sa.main(["--runs-root", str(runs), "--strict", "--no-env-check"]) == 0
    assert "转换环境不一致" not in capsys.readouterr().out


def test_main_prints_note_for_legacy_dataset(tmp_path, capsys):
    runs = tmp_path / "segment"
    make_run_for(tmp_path, runs, "legacy", make_dataset(tmp_path, "ds_old", None))
    assert sa.main(["--runs-root", str(runs)]) == 0
    assert "没有 env 指纹" in capsys.readouterr().out


# ------------------------------------------- 合成数据集的指纹回退（synth_report.json）

def test_dataset_env_falls_back_to_synth_report(tmp_path):
    d = make_synth_dataset(tmp_path, "cp", FP_A)
    assert sa.dataset_env({"data": str(d / "data.yaml")}) == FP_A


def test_dataset_env_prefers_stats_over_synth(tmp_path):
    """两个文件都在时以 dataset_stats.json 为准，顺序固定才好推理。"""
    d = make_synth_dataset(tmp_path, "cp", FP_B)
    (d / "dataset_stats.json").write_text(json.dumps({"env": FP_A}), encoding="utf-8")
    assert sa.dataset_env({"data": str(d)}) == FP_A


def test_dataset_env_synth_broken_json(tmp_path):
    d = make_synth_dataset(tmp_path, "cp", FP_A, broken=True)
    assert sa.dataset_env({"data": str(d / "data.yaml")}) is None


def test_dataset_env_synth_without_env_field(tmp_path):
    d = make_synth_dataset(tmp_path, "cp", None)
    assert sa.dataset_env({"data": str(d / "data.yaml")}) is None


def test_check_env_mixes_convert_and_synth_datasets(tmp_path):
    """真实场景：B0 用转换集、B1 用合成集，两边指纹都读得到才比得起来。"""
    runs = tmp_path / "segment"
    make_run_for(tmp_path, runs, "b0_base", make_dataset(tmp_path, "ds_conv", FP_A))
    make_run_for(tmp_path, runs, "b1_cp", make_synth_dataset(tmp_path, "ds_synth", FP_A))
    recs = sa.collect(runs, None, "metrics/mAP50-95(M)", quiet=True)
    warns, notes = sa.check_env(recs)
    assert warns == [] and notes == []


def test_check_env_flags_synth_vs_convert_drift(tmp_path):
    """转换集与合成集来自不同环境，同样要报出来——B0 与 B1 就没法比了。"""
    runs = tmp_path / "segment"
    make_run_for(tmp_path, runs, "b0_base", make_dataset(tmp_path, "ds_conv", FP_A))
    make_run_for(tmp_path, runs, "b1_cp", make_synth_dataset(tmp_path, "ds_synth", FP_B))
    recs = sa.collect(runs, None, "metrics/mAP50-95(M)", quiet=True)
    warns, notes = sa.check_env(recs)
    assert any("转换环境不一致" in w for w in warns)
    assert any("b0_base" in w and "b1_cp" in w for w in warns)
    assert notes == []
