"""tools/bench_trt.py 的单测（不需要 GPU，也不需要装 tensorrt）。

真跑 engine 的那部分归实测报告管。这里盯的是"报告里的数是怎么算出来的"：

* 判据的对照基准——fp32 比 .pt 参照，fp16/int8 比本轮 fp32 档。搞反了会把
  "量化掉点"记到"导出掉点"头上，两个问题的修法完全不同。
* ``low_prec_share``——量化到底生效没有的唯一直接证据。fp32 档必须返回 None
  而不是 0.0，否则"这一档不该有低精度层"和"inspector 没给格式"会混成一个数。
* ``build_table`` 读构建报告的键名。键名写错不会报错，只会让整列变成"—"，
  而表格照样出得来。
* ``align_engine`` 的聚合口径：max/rel_l2 取最大、cosine 取最小。取平均会把
  一张图上的崩坏摊薄到看不见。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from conftest import load_tool

bn = load_tool("bench_trt")


# --------------------------------------------------------------------------
# 判据默认值
# --------------------------------------------------------------------------

def test_thresholds_int8_first_layer_is_report_only():
    """INT8 逐张量比差本来就对不上，判了只会得到一条永远红的记录。"""
    th = bn.DEFAULT_THRESHOLDS["int8"]
    assert th["min_cosine"] is None and th["max_rel_l2"] is None
    assert th["max_map_delta"] == 0.03


def test_thresholds_fp32_fp16_match_their_numeric_format():
    """rel_l2 的卡线按这一档实际用的数值格式定，不能拿真 FP32 后端的数去卡 TF32/FP16。

    TRT 默认打开 ``BuilderFlag.TF32``，所以"FP32 档"跑的是 TF32（10-bit 尾数），
    FP16 也是 10-bit：两者的逐张量 rel_l2 天然停在 1e-3 量级，那是格式的机器精度，
    不是链路错。把线画在 1e-3 只会得到一条永远红的记录，然后你就会去查前处理
    （本轮真查了半天，最后发现是 TF32，见 docs/deploy.md §8.4）。
    要卡 1e-5，就用 ``build_trt.py --no-tf32`` 另建一档真 FP32，
    再用 ``--max-rel-l2 1e-5`` 收紧判据；最终判据始终是端到端 mAP。
    """
    for prec in ("fp32", "fp16"):
        th = bn.DEFAULT_THRESHOLDS[prec]
        assert th["min_cosine"] == 0.9999
        assert th["max_rel_l2"] == 1e-2
        assert th["max_map_delta"] == 0.01


def test_rel_l2_threshold_still_admits_tightening_via_cli(bench_env, capsys):
    """放宽默认值不等于把这条判据废掉：--max-rel-l2 仍能收紧到真 FP32 那一档。"""
    rc = bn.main(["--precs", "fp32", "--max-rel-l2", "1e-5", "--dry-run"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "1e-05" in out or "1e-5" in out


def test_thresholds_dicts_are_independent():
    """main 里逐档覆盖阈值时改的是副本；共用一个 dict 会串档。"""
    a = bn.DEFAULT_THRESHOLDS["fp32"]
    b = bn.DEFAULT_THRESHOLDS["fp16"]
    assert a is not b


# --------------------------------------------------------------------------
# low_prec_share
# --------------------------------------------------------------------------

def test_low_prec_share_fp32_is_none_not_zero():
    assert bn.low_prec_share({"Float": 252}, "fp32") is None


def test_low_prec_share_fp16_counts_half():
    assert bn.low_prec_share({"Float": 24, "Half": 161}, "fp16") == 87.0


def test_low_prec_share_int8_counts_int8():
    assert bn.low_prec_share({"Int8": 90, "Half": 8, "Float": 2}, "int8") == 90.0


def test_low_prec_share_missing_key_is_zero():
    """一层都没落到低精度 = 0.0%，这是"量化没生效"的证据，不能报成 None。"""
    assert bn.low_prec_share({"Float": 10}, "fp16") == 0.0


def test_low_prec_share_empty_formats_is_none():
    assert bn.low_prec_share({}, "fp16") is None


def test_low_prec_share_unknown_prec_is_none():
    assert bn.low_prec_share({"Half": 1}, "bf16") is None


# --------------------------------------------------------------------------
# check_align：只做键名映射，判据本体在 vox.check_align
# --------------------------------------------------------------------------

def _out(i, rel_l2, cosine=0.99999, shape_match=True):
    """_md 会逐字段读这些键，缺一个就 KeyError——报告生成不该比压测本身脆。"""
    return {"index": i, "trt_name": f"output{i}", "shape": (1, 4),
            "max_abs": 1e-4, "mean_abs": 1e-5, "rel_l2": rel_l2,
            "cosine": cosine, "shape_match": shape_match, "n_images": 4}


ALIGN_OK = {
    "n_images": 4, "seconds": 3.2, "output_count_pt": 2, "output_count_trt": 2,
    "worst_cosine": 0.99999,
    "outputs": [_out(0, 1e-5), _out(1, 2e-5)],
}


def test_check_align_pass_maps_trt_count_to_ort_key():
    ok, why = bn.check_align(dict(ALIGN_OK), 0.9999, 1e-3)
    assert ok and why == []


def test_check_align_both_thresholds_none_is_report_only():
    """INT8 第 1 层只报不判：阈值全 None 时必须直接放行，不去比数。"""
    bad = dict(ALIGN_OK, worst_cosine=0.5, output_count_trt=9)
    ok, why = bn.check_align(bad, None, None)
    assert ok and why == []


def test_check_align_output_count_mismatch_fails():
    ok, why = bn.check_align(dict(ALIGN_OK, output_count_trt=1), 0.9999, 1e-3)
    assert not ok
    assert any("输出数不一致" in w for w in why)


def test_check_align_low_cosine_fails():
    ok, why = bn.check_align(dict(ALIGN_OK, worst_cosine=0.98), 0.9999, 1e-3)
    assert not ok and any("cosine" in w for w in why)


def test_check_align_shape_mismatch_fails_even_with_same_element_count():
    """(1,51,8400) 与 (1,8400,51) 拉平后差值看着还行，必须靠形状这一条拦住。"""
    a = dict(ALIGN_OK)
    a["outputs"] = [{"index": 0, "shape_match": False, "rel_l2": 1e-9}]
    ok, why = bn.check_align(a, 0.9999, 1e-3)
    assert not ok and any("形状不一致" in w for w in why)


def test_check_align_rel_l2_too_big_fails():
    a = dict(ALIGN_OK)
    a["outputs"] = [{"index": 0, "shape_match": True, "rel_l2": 0.5}]
    ok, why = bn.check_align(a, 0.9999, 1e-3)
    assert not ok and any("rel_l2" in w for w in why)


def test_check_align_none_side_is_skipped_other_still_judged():
    """min_cosine=None 映射成 -1.0：cosine 再烂也不判，但 rel_l2 仍然要判。

    这正是 INT8 第 1 层"只报不判"想要的效果——只关掉一项，不是整个关掉。
    """
    a = dict(ALIGN_OK, worst_cosine=0.1)
    a["outputs"] = [_out(0, 1e-5)]
    assert bn.check_align(a, None, 1e-3) == (True, [])
    a["outputs"] = [_out(0, 0.5)]
    ok, why = bn.check_align(a, None, 1e-3)
    assert not ok and any("rel_l2" in w for w in why)
    assert not any("cosine" in w for w in why)


# --------------------------------------------------------------------------
# gate_tier
# --------------------------------------------------------------------------

def tier_with_map(m):
    return {"align": dict(ALIGN_OK), "val": {"mask": {"mAP50-95": m},
                                             "box": {"mAP50-95": m}}}


def test_gate_fp32_uses_pt_reference_not_fp32_tier():
    """fp32 档的对照基准是 .pt；拿它自己跟自己比永远 0 差，等于没判。"""
    t = tier_with_map(0.40)
    ok, why = bn.gate_tier("fp32", t, bn.DEFAULT_THRESHOLDS["fp32"],
                           base_mask=0.99, ref_mask=0.405)
    assert ok, why
    assert t["map_base"] == 0.405 and abs(t["map_delta"] - (-0.005)) < 1e-9


def test_gate_fp16_uses_this_round_fp32_tier():
    t = tier_with_map(0.395)
    ok, why = bn.gate_tier("fp16", t, bn.DEFAULT_THRESHOLDS["fp16"],
                           base_mask=0.40, ref_mask=0.20)
    assert ok, why
    assert t["map_base"] == 0.40 and t["map_delta"] == round(0.395 - 0.40, 6)


def test_gate_map_delta_beyond_threshold_fails():
    t = tier_with_map(0.30)
    ok, why = bn.gate_tier("fp16", t, bn.DEFAULT_THRESHOLDS["fp16"],
                           base_mask=0.40, ref_mask=0.40)
    assert not ok
    assert any("mAP50-95" in w and w.startswith("第2层") for w in why)


def test_gate_delta_on_the_threshold_line_is_float_sensitive():
    """0.39 vs 0.40 的差在 IEEE754 下是 0.010000000000000009，压线即判失败。

    记下来是提醒：报告里出现"差 0.0100 超过 ±0.01"不是判据写错，是浮点。
    要么把阈值留余量，要么别把数字正好卡在线上。
    """
    assert 0.39 - 0.40 != -0.01
    t = tier_with_map(0.39)
    ok, why = bn.gate_tier("fp16", t, bn.DEFAULT_THRESHOLDS["fp16"],
                           base_mask=0.40, ref_mask=0.40)
    assert not ok and any("超过" in w for w in why)


def test_gate_int8_tolerates_bigger_delta():
    """INT8 的容忍度是 0.03，0.02 的掉点该判过。"""
    t = tier_with_map(0.38)
    ok, why = bn.gate_tier("int8", t, bn.DEFAULT_THRESHOLDS["int8"],
                           base_mask=0.40, ref_mask=0.40)
    assert ok, why


def test_gate_no_reference_is_not_a_failure():
    """基准拿不到就不判这一项：把环境问题算到模型头上是误报。"""
    t = tier_with_map(0.10)
    ok, why = bn.gate_tier("fp16", t, bn.DEFAULT_THRESHOLDS["fp16"],
                           base_mask=None, ref_mask=None)
    assert ok, why == []
    assert t["map_delta"] is None and t["map_base"] is None


def test_gate_no_val_section_skips_map_check():
    t = {"align": dict(ALIGN_OK)}
    ok, why = bn.gate_tier("fp32", t, bn.DEFAULT_THRESHOLDS["fp32"],
                           base_mask=None, ref_mask=0.40)
    assert ok, why


def test_gate_first_layer_failure_is_prefixed():
    t = {"align": dict(ALIGN_OK, worst_cosine=0.5), "val": {}}
    ok, why = bn.gate_tier("fp32", t, bn.DEFAULT_THRESHOLDS["fp32"],
                           base_mask=None, ref_mask=None)
    assert not ok and why[0].startswith("第1层")


# --------------------------------------------------------------------------
# build_table
# --------------------------------------------------------------------------

BUILD_REP = {"engines": [
    {"precision": "fp32", "num_layers": 252, "build_seconds": 89.0,
     "output_formats": {"Float": 252}},
    {"precision": "fp16", "num_layers": 185, "build_seconds": 195.4,
     "output_formats": {"Float": 24, "Half": 161}},
]}


def make_tiers():
    return {
        "fp32": {"engine_mib": 65.31, "activation_mib": 38.28, "process_vram_mib": 400.0,
                 "bench": {"mean_ms": 8.0, "p95_ms": 9.0, "p99_ms": 9.5, "fps": 125.0},
                 "val": {"mask": {"mAP50-95": 0.400}, "box": {"mAP50-95": 0.500}}},
        "fp16": {"engine_mib": 24.53, "activation_mib": 17.83, "process_vram_mib": 300.0,
                 "bench": {"mean_ms": 4.0, "p95_ms": 5.0, "p99_ms": 5.5, "fps": 250.0},
                 "val": {"mask": {"mAP50-95": 0.398}, "box": {"mAP50-95": 0.499}}},
    }


def test_build_table_reads_engines_key_not_builds():
    rows = bn.build_table(make_tiers(), BUILD_REP, None)
    by = {r["precision"]: r for r in rows}
    # 键名写错的话这两列会全变成 None，表格照样出得来
    assert by["fp32"]["num_layers"] == 252
    assert by["fp16"]["build_seconds"] == 195.4


def test_build_table_computes_speedup_and_size_ratio():
    rows = bn.build_table(make_tiers(), BUILD_REP, None)
    by = {r["precision"]: r for r in rows}
    assert by["fp16"]["speedup_vs_fp32"] == 2.0
    assert by["fp16"]["size_ratio_vs_fp32"] == round(24.53 / 65.31, 3)
    assert by["fp16"]["low_prec_pct"] == 87.0
    assert by["fp16"]["map_delta_vs_fp32"] == round(0.398 - 0.400, 6)


def test_build_table_fp32_row_has_no_self_comparison():
    rows = bn.build_table(make_tiers(), BUILD_REP, None)
    fp32 = [r for r in rows if r["precision"] == "fp32"][0]
    assert fp32["speedup_vs_fp32"] is None
    assert fp32["map_delta_vs_fp32"] is None
    assert fp32["low_prec_pct"] is None


def test_build_table_appends_torch_row_with_unit_speedup():
    torch_stats = {"mean_ms": 13.13, "p95_ms": 15.91, "p99_ms": 16.0, "fps": 76.17}
    rows = bn.build_table(make_tiers(), BUILD_REP, torch_stats)
    assert rows[-1]["precision"].startswith("pytorch")
    assert rows[-1]["speedup_vs_torch"] == 1.0
    assert rows[-1]["mean_ms"] == 13.13
    fp16 = [r for r in rows if r["precision"] == "fp16"][0]
    assert fp16["speedup_vs_torch"] == round(13.13 / 4.0, 2)


def test_build_table_skips_absent_tiers_and_keeps_order():
    rows = bn.build_table({"fp16": make_tiers()["fp16"]}, BUILD_REP, None)
    assert [r["precision"] for r in rows] == ["fp16"]
    # 没有 fp32 基准，speedup 只能是 None，不能除零
    assert rows[0]["speedup_vs_fp32"] is None


def test_build_table_without_build_report_still_yields_rows():
    rows = bn.build_table(make_tiers(), None, None)
    assert len(rows) == 2
    assert all(r["num_layers"] is None for r in rows)


# --------------------------------------------------------------------------
# _fmt / _md
# --------------------------------------------------------------------------

def test_fmt_none_is_dash():
    assert bn._fmt(None) == "—"


def test_fmt_string_passthrough():
    assert bn._fmt("fp16") == "fp16"


def test_fmt_applies_spec():
    assert bn._fmt(1.23456, ".2f") == "1.23"
    assert bn._fmt(-0.002, "+.4f") == "-0.0020"


def md_report():
    return {
        "sample": {"source": "manifest", "path": "configs/bench_sample.json",
                   "count": 64, "source_total": 435, "sha256": "eccfd964426a2637",
                   "align_n": 16, "bench_n": 30, "warnings": ["清单比母体旧"]},
        "table": bn.build_table(make_tiers(), BUILD_REP,
                                {"mean_ms": 13.13, "p95_ms": 15.91, "p99_ms": 16.0,
                                 "fps": 76.17}),
        "tiers": {"fp32": {"align": dict(ALIGN_OK, seconds=3.2),
                           "gate": {"ok": True, "why": []}},
                  "int8": {"align": dict(ALIGN_OK, worst_cosine=0.91),
                           "gate": {"ok": True, "why": ["第2层：差 -0.02"]}}},
        "verdict": "PASS", "seconds": 900, "git_head": "abc1234",
    }


def test_md_has_table_header_and_verdict():
    L = bn._md(md_report(), {"run": "baseline15", "split": "test"})
    txt = "\n".join(L)
    assert "TensorRT 三档压测" in txt and "run=`baseline15`" in txt
    assert "| 档位 | engine MiB |" in txt
    assert "verdict：**PASS**" in txt
    assert "git `abc1234`" in txt
    assert all(x.startswith("|") or x == "" or not x.startswith("\t") for x in L)


def test_md_renders_sample_warnings():
    L = bn._md(md_report(), {"run": "r", "split": "test"})
    assert any("清单比母体旧" in x and x.startswith("> 注意") for x in L)


def test_md_marks_int8_first_layer_as_report_only():
    L = bn._md(md_report(), {"run": "r", "split": "test"})
    assert any("INT8 第 1 层只报不判" in x for x in L)


def test_md_dashes_missing_low_prec_share():
    rep = md_report()
    rep["table"] = bn.build_table(make_tiers(), None, None)
    L = bn._md(rep, {"run": "r", "split": "test"})
    row = [x for x in L if x.startswith("| fp32 ")][0]
    assert "— |" in row


def test_md_handles_empty_sample_section():
    rep = md_report()
    rep["sample"] = {}
    L = bn._md(rep, {"run": "r", "split": "test"})
    assert any(x.startswith("| 档位 |") for x in L)


# --------------------------------------------------------------------------
# load_build_report
# --------------------------------------------------------------------------

def test_load_build_report_none_path():
    assert bn.load_build_report(None) is None


def test_load_build_report_missing_file(tmp_path: Path):
    assert bn.load_build_report(tmp_path / "nope.json") is None


def test_load_build_report_reads_json(tmp_path: Path):
    p = tmp_path / "rep.json"
    p.write_text(json.dumps({"engines": [{"precision": "fp32"}]}), encoding="utf-8")
    got = bn.load_build_report(p)
    assert got["engines"][0]["precision"] == "fp32"


def test_load_build_report_bad_json_returns_none(tmp_path: Path, capsys):
    p = tmp_path / "rep.json"
    p.write_text("{not json", encoding="utf-8")
    assert bn.load_build_report(p) is None
    assert "构建报告读不了" in capsys.readouterr().out


# --------------------------------------------------------------------------
# align_engine：聚合口径
# --------------------------------------------------------------------------

class FakeRunner:
    """按脚本回放每图的输出，好断言聚合方式。"""

    def __init__(self, outs, names=("output0", "output1")):
        self._outs = list(outs)
        self.output_names = list(names)
        self.calls = 0

    def __call__(self, x):
        assert x.shape[0] == 1 and x.dtype == np.float32
        o = self._outs[self.calls]
        self.calls += 1
        return o


def test_align_engine_aggregates_worst_not_mean(monkeypatch):
    """图 1 干净、图 2 崩坏：取平均会看不出来，必须取最坏。"""
    z = np.zeros((1, 4), dtype=np.float32)
    bad = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
    # 每张图两个输出：out0 在第二张图上崩，out1 始终干净
    ref = iter([[z, z], [z, z]])
    trt = [[z, z], [bad, z]]
    monkeypatch.setattr(bn.vox, "_net_forward", lambda net, x4, device: next(ref))
    xs = [np.zeros((3, 8, 8), dtype=np.float32) for _ in range(2)]
    a = bn.align_engine(xs, net=None, runner=FakeRunner(trt), device="cpu")
    assert a["n_images"] == 2
    assert a["output_count_pt"] == 2 and a["output_count_trt"] == 2
    o0, o1 = a["outputs"]
    assert o0["n_images"] == 2 and o0["trt_name"] == "output0"
    assert o0["max_abs"] == 1.0                       # 取最大，不是平均 0.5
    assert o0["mean_abs"] == pytest.approx(0.125)     # 0 和 0.25 的均值
    assert o0["cosine"] < 1.0
    assert a["worst_cosine"] == o0["cosine"]          # 全局最坏取自 out0
    assert o1["max_abs"] == 0.0 and o1["shape_match"] is True


def test_align_engine_flags_shape_mismatch(monkeypatch):
    """元素数一样、形状不同：拉平后能算出差值，必须靠 shape_match 报出来。"""
    monkeypatch.setattr(bn.vox, "_net_forward",
                        lambda net, x4, device: [np.zeros((1, 2, 3), dtype=np.float32)])
    runner = FakeRunner([[np.zeros((1, 3, 2), dtype=np.float32)]], names=("out",))
    a = bn.align_engine([np.zeros((3, 8, 8), dtype=np.float32)], None, runner, "cpu")
    assert a["outputs"][0]["shape_match"] is False
    ok, why = bn.check_align(a, 0.9999, 1e-3)
    assert not ok and any("形状不一致" in w for w in why)


def test_align_engine_records_output_count_mismatch(monkeypatch):
    monkeypatch.setattr(bn.vox, "_net_forward",
                        lambda net, x4, device: [np.zeros((1, 2), dtype=np.float32),
                                                 np.zeros((1, 2), dtype=np.float32)])
    runner = FakeRunner([[np.zeros((1, 2), dtype=np.float32)]], names=("out0",))
    a = bn.align_engine([np.zeros((3, 8, 8), dtype=np.float32)], None, runner, "cpu")
    assert a["output_count_pt"] == 2 and a["output_count_trt"] == 1
    assert not bn.check_align(a, 0.9999, 1e-3)[0]


def test_align_engine_empty_inputs(monkeypatch):
    monkeypatch.setattr(bn.vox, "_net_forward", lambda net, x4, device: [])
    a = bn.align_engine([], None, FakeRunner([]), "cpu")
    assert a["n_images"] == 0 and a["outputs"] == []
    assert a["output_count_pt"] is None


# --------------------------------------------------------------------------
# build_report_tiers：从构建报告里取逐档结果
# --------------------------------------------------------------------------

def test_build_report_tiers_reads_engines_key():
    rep = {"engines": [{"precision": "fp32", "num_layers": 252},
                       {"precision": "int8", "num_layers": 166}]}
    got = bn.build_report_tiers(rep)
    assert sorted(got) == ["fp32", "int8"] and got["int8"]["num_layers"] == 166


def test_build_report_tiers_legacy_builds_key_is_not_read():
    """错键名不报错、只是表里全空——所以取数只能走这一个函数，不能各处自己拼。"""
    assert bn.build_report_tiers({"builds": [{"precision": "fp32"}]}) == {}


def test_build_report_tiers_skips_garbage_entries():
    rep = {"engines": [None, {"num_layers": 1}, {"precision": ""}, {"precision": "fp16"}]}
    assert sorted(bn.build_report_tiers(rep)) == ["fp16"]


def test_build_report_tiers_none_and_missing_key_are_safe():
    assert bn.build_report_tiers(None) == {}
    assert bn.build_report_tiers({}) == {}


def test_build_table_uses_report_layers_and_formats():
    """表里的层数/低精度占比只能来自构建报告，压测侧拿不到。"""
    tiers = {"fp32": {"engine_mib": 65.3, "bench": {"mean_ms": 8.0},
                      "val": {"mask": {"mAP50-95": 0.37}}},
             "int8": {"engine_mib": 12.5, "bench": {"mean_ms": 5.0},
                      "val": {"mask": {"mAP50-95": 0.35}}}}
    rep = {"engines": [{"precision": "int8", "num_layers": 166,
                        "output_formats": {"Int8": 106, "Half": 42, "Float": 18}}]}
    rows = {r["precision"]: r for r in bn.build_table(tiers, rep, None)}
    assert rows["int8"]["num_layers"] == 166
    assert rows["fp32"]["num_layers"] is None


# --------------------------------------------------------------------------
# main() 错误码与 dry-run
# --------------------------------------------------------------------------

@pytest.fixture
def bench_env(tmp_path: Path):
    """搭一个能让 main 走到 dry-run 的最小环境：权重 + data.yaml + 三档 engine。"""
    wdir = tmp_path / "runs" / "seg" / "weights"
    wdir.mkdir(parents=True)
    w = wdir / "best.pt"
    w.write_bytes(b"fake-pt")
    data = tmp_path / "data.yaml"
    data.write_text("path: .\ntrain: images/train\nval: images/val\ntest: images/test\n",
                    encoding="utf-8")
    edir = tmp_path / "w"
    edir.mkdir()
    for p in ("fp32", "fp16", "int8"):
        (edir / f"seg_trt_{p}.engine").write_bytes(b"fake-engine" * 100)
    return w, data, edir


def base_args(w, data, edir, *extra):
    return ["--weights", str(w), "--data", str(data), "--engine-dir", str(edir),
            "--name", "seg", "--sample-list", str(data.parent / "no_sample.json"),
            *extra]


def test_main_missing_weights_returns_2(tmp_path: Path, capsys):
    rc = bn.main(["--weights", str(tmp_path / "no.pt"), "--data", str(tmp_path / "d.yaml"),
                  "--dry-run"])
    assert rc == 2
    assert "权重不存在" in capsys.readouterr().out


def test_main_missing_data_yaml_returns_2(tmp_path: Path, capsys):
    w = tmp_path / "best.pt"
    w.write_bytes(b"x")
    rc = bn.main(["--weights", str(w), "--data", str(tmp_path / "no.yaml"), "--dry-run"])
    assert rc == 2
    assert "data.yaml不存在" in capsys.readouterr().out


def test_main_missing_engine_returns_2(bench_env, capsys):
    w, data, edir = bench_env
    (edir / "seg_trt_int8.engine").unlink()
    rc = bn.main(base_args(w, data, edir, "--dry-run"))
    out = capsys.readouterr().out
    assert rc == 2 and "缺 engine" in out and "seg_trt_int8.engine" in out


def test_main_dry_run_lists_engines_and_sizes(bench_env, capsys):
    w, data, edir = bench_env
    rc = bn.main(base_args(w, data, edir, "--dry-run"))
    out = capsys.readouterr().out
    assert rc == 0
    for p in ("fp32", "fp16", "int8"):
        assert f"[dry  ] {p}:" in out
    assert "MiB" in out


def test_main_dry_run_respects_precs_subset(bench_env, capsys):
    w, data, edir = bench_env
    rc = bn.main(base_args(w, data, edir, "--precs", "fp16", "--dry-run"))
    out = capsys.readouterr().out
    assert rc == 0 and "[dry  ] fp16:" in out and "[dry  ] int8:" not in out


def test_main_run_name_derived_from_weights_parent(bench_env, capsys):
    """权重在 .../<run>/weights/best.pt 时，run 名取 <run>；engine 命名全靠它。"""
    w, data, edir = bench_env
    bn.main(base_args(w, data, edir, "--dry-run"))
    out = capsys.readouterr().out
    assert '"run": "seg"' in out


def test_main_threshold_overrides_apply_to_fp32_fp16_only(bench_env, capsys):
    w, data, edir = bench_env
    bn.main(base_args(w, data, edir, "--min-cosine", "0.9", "--dry-run"))
    cfg = json.loads([x for x in capsys.readouterr().out.splitlines()
                      if x.startswith("[cfg  ]")][0][len("[cfg  ] "):])
    assert cfg["thresholds"]["fp32"]["min_cosine"] == 0.9
    assert cfg["thresholds"]["fp16"]["min_cosine"] == 0.9
    assert cfg["thresholds"]["int8"]["min_cosine"] is None


def test_main_int8_map_delta_override(bench_env, capsys):
    w, data, edir = bench_env
    bn.main(base_args(w, data, edir, "--int8-max-map-delta", "0.05", "--dry-run"))
    cfg = json.loads([x for x in capsys.readouterr().out.splitlines()
                      if x.startswith("[cfg  ]")][0][len("[cfg  ] "):])
    assert cfg["thresholds"]["int8"]["max_map_delta"] == 0.05
    assert cfg["thresholds"]["fp16"]["max_map_delta"] == 0.01


def test_main_dry_run_does_not_touch_torch(bench_env, monkeypatch):
    """dry-run 要能在没装 torch/tensorrt 的机器上跑通，否则 CI 里没法验路径。"""
    w, data, edir = bench_env
    import sys as _sys

    monkeypatch.setitem(_sys.modules, "torch", None)
    monkeypatch.setitem(_sys.modules, "tensorrt", None)
    assert bn.main(base_args(w, data, edir, "--dry-run")) == 0


def test_main_unknown_prec_is_rejected_by_argparse(bench_env):
    w, data, edir = bench_env
    with pytest.raises(SystemExit):
        bn.main(base_args(w, data, edir, "--precs", "bf16"))