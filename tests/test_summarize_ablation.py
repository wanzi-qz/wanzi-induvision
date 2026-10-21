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
