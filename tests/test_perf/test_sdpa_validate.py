# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""SDPA/TopK validation harness (T4.4): ops_perf_results CSV parser, row to config mapping,
statistics and the gap_analysis 4.3 criterion."""
from __future__ import annotations

import copy
import math
import statistics
from pathlib import Path

import numpy as np
import pytest

from ttsim.perf import sdpa_validate as sv

DATA = Path(__file__).parent / "data"
FIXTURE = DATA / "ops_perf_results_sdpa_fixture.csv"
POSITIONS = DATA / "ops_perf_results_sdpa_fixture_positions.csv"


def _rows():
    _, raw = sv.read_ops_csv(FIXTURE)
    return sv.aggregate_replays(sv.select_op_rows(raw))


def _row(kind):
    return [r for r in _rows() if r.kind == kind][0]


# ---------------------------------------------------------------- parsing

@pytest.mark.unit
def test_parse_attr_value_forms():
    assert sv.parse_attr_value("std::nullopt") is None
    assert sv.parse_attr_value("true") is True and sv.parse_attr_value("false") is False
    assert sv.parse_attr_value("MathFidelity::HiFi2") == "HiFi2"
    assert sv.parse_attr_value("{1023, 7}") == [1023, 7]
    assert sv.parse_attr_value("{}") == []
    assert sv.parse_attr_value("(x=8,y=8)") == (8, 8)
    assert sv.parse_attr_value("CoreCoord(x=7,y=10)") == (7, 10)
    assert sv.parse_attr_value("0.0883883") == pytest.approx(0.0883883)
    assert sv.parse_attr_value("-1") == -1
    ck = sv.parse_attr_value("ComputeKernelConfig(math_fidelity=HiFi4,math_approx_mode=0,fp32_dest_acc_en=1,"
                             "packer_l1_acc=1,dst_full_sync_en=0,throttle_level=NO_THROTTLE)")
    assert ck == {"math_fidelity": "HiFi4", "math_approx_mode": 0, "fp32_dest_acc_en": 1, "packer_l1_acc": 1,
                  "dst_full_sync_en": 0, "throttle_level": "NO_THROTTLE"}


@pytest.mark.unit
def test_parse_attributes_undoes_csv_semicolons():
    # The tracy writer replaces every ',' with ';' inside the cell (process_ops_logs.py csv_row loop).
    cell = ("{'scale': '0.088'; 'program_config': 'SDPAProgramConfig(compute_with_storage_grid_size=(x=8;y=8);"
            "sub_core_grids=std::nullopt;q_chunk_size=256;k_chunk_size=256;exp_approx_mode=0;"
            "max_cores_per_head_batch=16)'; 'is_causal': 'true'; 'cur_pos': '{5; 9}'}")
    a = sv.parse_attributes(cell)
    assert a["program_config"]["compute_with_storage_grid_size"] == (8, 8)
    assert a["program_config"]["q_chunk_size"] == 256 and a["program_config"]["sub_core_grids"] is None
    assert a["program_config"]["exp_approx_mode"] == 0
    assert a["is_causal"] is True and a["cur_pos"] == [5, 9] and a["scale"] == pytest.approx(0.088)
    assert sv.parse_attributes("") == {}


@pytest.mark.unit
def test_parse_dim_and_dtype():
    assert sv.parse_dim("4096[4096]") == (4096, 4096)
    assert sv.parse_dim("1056[1024]") == (1056, 1024)
    assert sv.parse_dim("32") == (32, 32)
    assert sv.normalize_dtype("BFLOAT8_B") == "bfp8_b"
    assert sv.normalize_dtype("DataType::BFLOAT16") == "bfloat16"
    assert sv.normalize_dtype("INT32") == "int32"
    assert sv.normalize_fidelity("MathFidelity.HiFi4") == "HiFi4"


@pytest.mark.unit
def test_classify_op_code():
    assert sv.classify_op_code("ScaledDotProductAttention") == "sdpa_prefill"
    assert sv.classify_op_code("ScaledDotProductAttentionDecode") == "sdpa_decode"
    assert sv.classify_op_code("SdpaDecodeDeviceOperation") == "sdpa_decode"
    assert sv.classify_op_code("TopK") == "topk"
    assert sv.classify_op_code("TopkLargeIndices") == "topk_large_indices"
    assert sv.classify_op_code("topk_route_prep") == "topk_route"
    assert sv.classify_op_code("MatmulDeviceOperation") is None


@pytest.mark.unit
def test_fixture_selects_sdpa_and_topk_rows_and_medians_replays():
    comments, raw = sv.read_ops_csv(FIXTURE)
    assert comments and comments[0].startswith("# SYNTHETIC")
    assert len(raw) == 8                      # 3 synthetic op rows + a replay + a matmul + 2 real T2.5 rows + a signpost
    selected = sv.select_op_rows(raw)
    assert [r.kind for r in selected] == ["sdpa_prefill", "sdpa_decode", "sdpa_decode", "topk_large_indices",
                                          "sdpa_prefill", "sdpa_decode"]
    rows = sv.aggregate_replays(selected)
    assert [r.kind for r in rows] == ["sdpa_prefill", "sdpa_decode", "topk_large_indices", "sdpa_prefill", "sdpa_decode"]
    dec = rows[1]
    assert dec.replays == 2 and dec.meas_ns == statistics.median([254400, 255600])
    assert [t.shape for t in dec.inputs] == [[1, 32, 32, 128], [1024, 8, 32, 128], [1024, 8, 32, 128],
                                             [1, 1, 1, 32], [1, 1, 32, 1024]]
    assert dec.inputs[3].is_int and dec.inputs[1].dtype == "bfp8_b" and dec.inputs[0].dtype == "bfloat16"


@pytest.mark.unit
def test_aggregate_replays_takes_median_of_traced_rows():
    base = _row("sdpa_prefill")
    rows = []
    for meas in (100.0, 300.0, 200.0):
        r = copy.deepcopy(base)
        r.trace_id, r.meas_ns = "3", meas
        rows.append(r)
    out = sv.aggregate_replays(rows)
    assert len(out) == 1 and out[0].meas_ns == 200.0 and out[0].replays == 3


@pytest.mark.unit
def test_select_op_rows_keeps_only_device_rows_with_a_duration():
    _, raw = sv.read_ops_csv(FIXTURE)
    base = dict(raw[0])
    host = dict(base, **{"OP TYPE": "tt_dnn_host"})
    zero = dict(base, **{"DEVICE KERNEL DURATION [ns]": "0"})
    blank = dict(base, **{"DEVICE KERNEL DURATION [ns]": ""})
    assert len(sv.select_op_rows([base, host, zero, blank])) == 1


# ---------------------------------------------------------------- row to config

@pytest.mark.unit
def test_prefill_row_builds_production_config():
    cfg = sv.prefill_config(_row("sdpa_prefill"))
    assert (cfg.S, cfg.num_heads, cfg.num_kv_heads, cfg.head_dim) == (1024, 32, 8, 128)
    assert (cfg.q_chunk, cfg.k_chunk, cfg.num_cores) == (64, 64, 64)
    assert (cfg.fidelity, cfg.exp_approx_mode, cfg.fp32_dest_acc) == ("HiFi4", False, True)
    assert cfg.accum_dtype == "float32" and cfg.input_dtype == "bfp8_b" and cfg.kv_input_dtype == ""
    assert cfg.is_causal and not cfg.has_attn_mask and not cfg.paged
    assert not any(x.endswith("_absent") for x in cfg.fallback_reasons)


@pytest.mark.unit
def test_joint_row_is_priced_over_the_concatenated_streams():
    # JointSDPADeviceOperation rows carry q, k, v and the three joint tensors; the config spans both streams.
    r = copy.deepcopy(_row("sdpa_prefill"))
    r.op_code = "JointSDPADeviceOperation"
    q, k, v = r.inputs[:3]
    joint = []
    for t in (q, k, v):
        j = copy.deepcopy(t)
        j.shape = list(t.shape[:-2]) + [333, t.shape[-1]]
        j.padded = list(t.padded[:-2]) + [352, t.padded[-1]]
        joint.append(j)
    r.inputs = [q, k, v] + joint
    cfg = sv.prefill_config(r)
    assert cfg.is_joint and cfg.joint_seq == 333 and not cfg.is_causal and cfg.S == q.shape[-2] + 333
    assert sv.predict(cfg).regime == "joint"


@pytest.mark.unit
def test_sub_core_grids_set_the_grid_core_count():
    assert sv._core_range_set_size("{[(x=0;y=0) - (x=7;y=3)]}") == 32
    assert sv._core_range_set_size([[(0, 0), (7, 7)], [(0, 8), (3, 9)]]) == 72
    assert sv._core_range_set_size("nonsense") == 0
    r = copy.deepcopy(_row("sdpa_prefill"))
    r.attrs["program_config"]["sub_core_grids"] = "{[(x=0;y=0) - (x=7;y=3)]}"
    assert sv._common_attrs(r)["grid_cores"] == 32
    r.core_count = 0                                   # no CORE COUNT column: the config grid stands in
    assert sv._common_attrs(r)["num_cores"] == 32
    r.attrs["program_config"]["sub_core_grids"] = "{}"
    a = sv._common_attrs(r)
    assert "grid_cores" not in a and a["sub_core_grids_unparsed"] is True


@pytest.mark.unit
def test_decode_row_accepts_a_scalar_cur_pos():
    r = copy.deepcopy(_row("sdpa_decode"))
    r.attrs["cur_pos"] = 1023
    kw = sv.decode_config(r)
    assert kw["cur_pos"] == 1023 and not kw["cur_pos_unknown"]


@pytest.mark.unit
def test_decode_row_builds_config_with_sidecar_position():
    kw = sv.decode_config(_row("sdpa_decode"), sidecar={"cur_pos": 1023, "page_block_size": 32})
    assert (kw["batch"], kw["num_q_heads"], kw["num_kv_heads"], kw["head_dim"]) == (32, 32, 8, 128)
    assert kw["q_in_dram"] is True          # the synthetic row keeps its query in DRAM
    # 1024 blocks of 32 tokens over 32 users: the cache holds 1024 tokens per user, not the table width
    assert kw["cache_len"] == 1024 and kw["cur_pos"] == 1023 and not kw["cur_pos_unknown"]
    assert kw["k_chunk"] == 512 and kw["num_cores"] == 64 and kw["max_cores_per_head_batch"] == 16
    assert (kw["fidelity"], kw["accum_dtype"], kw["input_dtype"], kw["kv_input_dtype"]) == \
        ("HiFi2", "float32", "bfloat16", "bfp8_b")
    assert kw["paged"] and kw["page_block_size"] == 32 and kw["is_causal"]


@pytest.mark.unit
def test_decode_row_without_position_is_flagged_and_excluded():
    run = sv.validate(FIXTURE)
    dec = [x for x in run.results if x.regime == "decode"][0]
    assert dec.exclude_reason == "cur_pos_unknown" and "cur_pos_unknown" in dec.reasons
    run = sv.validate(FIXTURE, positions_csv=POSITIONS)
    dec = [x for x in run.results if x.regime == "decode"][0]
    assert not dec.excluded and abs(dec.err) <= 0.03      # the 254.4 us anchor at position 1023 (1024 rows)
    assert dec.config["attended"] == 1024 and dec.config["cur_pos"] == 1023


@pytest.mark.unit
def test_optional_inputs_identified_by_attributes_not_position():
    # List-form cur_pos: the only optional input is the page table, which must not be read as cur_pos.
    r = copy.deepcopy(_row("sdpa_decode"))
    r.attrs["cur_pos"] = [7, 1023, 500]
    r.inputs = r.inputs[:3] + [r.inputs[4]]
    kw = sv.decode_config(r)
    assert kw["paged"] and kw["cache_len"] == 1024 and kw["cur_pos"] == 1023 and not kw["cur_pos_unknown"]
    # Non-paged decode with one int input: that input is the position tensor.
    r2 = copy.deepcopy(_row("sdpa_decode"))
    r2.attrs["paged_attention"] = False
    r2.inputs = r2.inputs[:3] + [r2.inputs[3]]
    r2.inputs[1].shape = r2.inputs[2].shape = [32, 8, 4096, 128]
    kw2 = sv.decode_config(r2)
    assert not kw2["paged"] and kw2["cache_len"] == 4096 and kw2["cur_pos_unknown"]
    # Prefill with a dense mask: a float input whose last dim is the KV length.
    p = copy.deepcopy(_row("sdpa_prefill"))
    p.attrs["is_causal"] = False
    p.inputs.append(sv.TensorInfo(shape=[1, 1, 1024, 1024], padded=[1, 1, 1024, 1024], dtype="bfloat16"))
    cfg = sv.prefill_config(p)
    assert cfg.has_attn_mask and not cfg.is_causal and not cfg.paged


@pytest.mark.unit
def test_fidelity_column_mismatch_excludes_row():
    r = copy.deepcopy(_row("sdpa_prefill"))
    r.fidelity_col = "HiFi2"
    res = sv.predict_row(r)
    assert res.exclude_reason.startswith("fidelity_column_mismatch") and math.isnan(res.err)
    r = copy.deepcopy(_row("sdpa_prefill"))
    r.arch = "wormhole_b0"
    assert sv.predict_row(r).exclude_reason == "arch_not_blackhole:wormhole_b0"


@pytest.mark.unit
def test_topk_row_config_and_pricing():
    r = _row("topk_large_indices")
    assert sv.topk_config(r) == dict(N=8192, K=2048, rows=2048, num_cores=110)
    res = sv.predict_row(r)
    # The TopK roofline lives on its own branch; without it the row is listed and excluded, never priced.
    if res.exclude_reason:
        assert res.exclude_reason == "topk_model_unavailable" and math.isnan(res.err)
    else:
        assert math.isfinite(res.err) and res.regime == "topk_large_indices"


@pytest.mark.unit
def test_prefill_row_error_and_reasons_reach_the_table():
    res = sv.predict_row(_row("sdpa_prefill"))
    assert res.regime == "prefill_causal" and not res.excluded
    assert res.err == pytest.approx((res.pred_kernel_ns - 355000.0) / 355000.0)
    assert "legacy_path_production_family" in res.reasons and res.config_mismatch == ""
    rec = sv.row_record(res)
    assert rec["kernel_path"] == "fp32_legacy" and rec["fidelity"] == "HiFi4" and rec["excluded"] == 0


# ---------------------------------------------------------------- statistics and criterion

@pytest.mark.unit
def test_regime_statistics_known_answers():
    errs = [0.05, -0.05, 0.20, 0.0]
    meas = [100.0, 100.0, 100.0, 700.0]
    s = sv.summarize_regime("synthetic", errs, meas, resamples=2000, seed=1)
    assert s.n == 4 and s.mean_err == pytest.approx(0.05)
    assert s.tw_mean_err == pytest.approx((5 - 5 + 20 + 0) / 1000.0)
    assert s.std_err == pytest.approx(statistics.stdev(errs))
    assert s.frac_ops_within == pytest.approx(0.75) and s.frac_time_within == pytest.approx(0.9)
    assert s.passed                                        # |2 percent| <= 3 and 90 percent of time within
    assert s.tw_mean_ci[0] <= s.tw_mean_err <= s.tw_mean_ci[1]
    assert s.mean_ci[0] <= s.mean_err <= s.mean_ci[1]


@pytest.mark.unit
def test_bootstrap_is_deterministic_for_a_fixed_seed():
    errs = np.array([0.01, -0.04, 0.12, 0.03, -0.09, 0.0])
    w = np.array([1.0, 2.0, 1.0, 5.0, 1.0, 3.0])
    a = sv.bootstrap_mean_ci(errs, w, resamples=3000, seed=7)
    b = sv.bootstrap_mean_ci(errs, w, resamples=3000, seed=7)
    c = sv.bootstrap_mean_ci(errs, w, resamples=3000, seed=8)
    assert a == b and a != c and a[0] < a[1]
    assert sv.bootstrap_mean_ci([0.2], [1.0]) == (0.2, 0.2)
    assert all(math.isnan(v) for v in sv.bootstrap_mean_ci([], []))


@pytest.mark.unit
def test_pass_criterion_logic():
    assert sv.passes_criterion(0.03, 0.90)
    assert sv.passes_criterion(-0.03, 1.0)
    assert not sv.passes_criterion(0.031, 0.95)
    assert not sv.passes_criterion(0.0, 0.89)
    assert not sv.passes_criterion(float("nan"), 1.0)


@pytest.mark.unit
def test_pass_criterion_excludes_flagged_rows():
    base = _row("sdpa_prefill")

    def res(err, exclude="", regime="prefill_causal"):
        r = copy.deepcopy(base)
        r.meas_ns = 1000.0
        return sv.RowResult(row=r, regime=regime, err=err, pred_kernel_ns=1000.0 * (1 + err), exclude_reason=exclude)

    results = [res(0.01), res(-0.02), res(0.50, "cur_pos_unknown"), res(0.40, "config_mismatch"),
               res(0.0, regime="decode"), res(0.3, "topk_model_unavailable", regime="topk")]
    summaries = sv.summarize(results, resamples=500, seed=3)
    by = {s.regime: s for s in summaries}
    assert by["prefill_causal"].n == 2 and by["prefill_causal"].n_excluded == 2 and by["prefill_causal"].passed
    assert by["prefill_causal"].mean_err == pytest.approx(-0.005)
    assert by["decode"].n == 1 and by["decode"].passed
    assert by["topk"].n == 0 and not by["topk"].passed and sv._verdict(by["topk"]) == "NO DATA"
    run = sv.RunResult(results=results, summaries=summaries, comments=[])
    assert run.passed                                      # unscored regimes do not veto the verdict


# ---------------------------------------------------------------- outputs

@pytest.mark.unit
def test_write_outputs_tables_and_figure(tmp_path):
    run = sv.validate(FIXTURE, model="fixture", positions_csv=POSITIONS)
    paths = sv.write_outputs(run, tmp_path, model="fixture", provenance=["synthetic fixture"])
    assert paths["figure"].exists() and paths["figure"].stat().st_size > 1000
    comments, rows = sv.read_ops_csv(paths["rows"])
    assert any("PROVENANCE: synthetic fixture" in c for c in comments)
    assert [r["regime"] for r in rows] == ["prefill_causal", "decode", "topk_large_indices", "prefill_causal", "decode"]
    assert list(rows[0].keys()) == sv.ROW_COLUMNS
    assert rows[1]["cur_pos"] == "1023" and rows[1]["replays"] == "2" and rows[1]["kv_dtype"] == "bfp8_b"
    _, summary = sv.read_ops_csv(paths["summary"])
    assert list(summary[0].keys()) == sv.SUMMARY_COLUMNS
    verdict = {s["regime"]: s["pass"] for s in summary}
    # both decode rows sit within 1 percent on the T2.8 law (the synthetic 254.4 us at position 1023,
    # the real 281.9 us at 1024); prefill passes on both rows
    assert verdict["decode"] == "PASS" and verdict["prefill_causal"] == "PASS"


@pytest.mark.unit
def test_device_projection_uses_the_real_p100a_device():
    dev = sv.make_device()
    assert (dev.devname, dev.name) == ("Blackhole", "p100a")
    res = sv.predict_row(_row("sdpa_prefill"), device=dev)
    # fused cycles are booked as-is on the calibrated SKU; the projection adds only the ramp penalty
    assert res.pred_device_ns >= res.pred_kernel_ns and res.pred_device_ns < 1.01 * res.pred_kernel_ns


@pytest.mark.unit
def test_device_projection_off_the_calibrated_sku_books_the_generic_estimate():
    # p150a is outside the calibration gate, so execute_op prices the generic instruction path; the seeded
    # per element count keeps that from being zero compute (sdpa_sinf seeds the same count).
    dev = sv.make_device(device_name="p150a")
    row = _row("sdpa_prefill")
    ps = sv.predict(sv.prefill_config(row)).to_polaris_op_perf_stats()
    n = int(np.prod(row.inputs[0].shape))
    assert sv.projected_op(dev, ps).compute_cycles == 0
    assert sv.projected_op(dev, ps, fallback_elems=n).compute_cycles > 0
    assert sv.predict_row(row, device=dev).pred_device_ns == pytest.approx(sv.device_projection_ns(dev, ps, n))


@pytest.mark.unit
def test_cli_main_writes_outputs(tmp_path):
    rc = sv.main([str(FIXTURE), "-o", str(tmp_path), "--model", "cli", "--positions", str(POSITIONS),
                  "--no-device", "--no-figure", "--resamples", "200", "--provenance", "cli test"])
    assert rc in (0, 1)
    assert (tmp_path / "cli_summary.csv").exists() and (tmp_path / "cli_signed_errors.csv").exists()
    assert not (tmp_path / "cli_signed_errors.png").exists()


@pytest.mark.unit
def test_plot_wall_components_draws_model_and_measured_bars(tmp_path):
    from ttsim.perf.roofline_sdpa import ARCH_BH, SdpaConfig, predict
    r = predict(SdpaConfig(S=4096, num_heads=32, num_kv_heads=8, num_cores=110, arch=ARCH_BH))
    measured = {"compute_floor": 915586.0, "reader_wait": 979638.0, "control": 81963.0, "mask_bracket": 9398.0,
                "init": 900.0, "fe_issue": r.wall_clock_cycles - 915586.0 - 979638.0 - 81963.0 - 9398.0 - 900.0}
    out = sv.plot_wall_components([("causal q128 k128", r.components, measured), ("model only", r.components, None)],
                                  tmp_path / "components.png", title="grid 1")
    assert out.exists() and out.stat().st_size > 1000
    assert {n for n, _, _ in sv.WALL_TERM_STYLE} >= set(r.components)


# ---------------------------------------------------------------- real T2.5 rows (validation set, not fit)

def _real_rows():
    _, raw = sv.read_ops_csv(FIXTURE)
    rows = sv.aggregate_replays(sv.select_op_rows(raw))
    return rows[3], rows[4]          # SDPAOperation 28672, SdpaDecodeDeviceOperation 103424


@pytest.mark.unit
def test_real_prefill_row_parses_the_t25_attribute_strings():
    pre, _ = _real_rows()
    assert pre.op_code == "SDPAOperation" and pre.meas_ns == 347015.0
    pc = pre.attrs["program_config"]
    assert pc["compute_with_storage_grid_size"] == (8, 8) and pc["sub_core_grids"] is None
    assert (pc["q_chunk_size"], pc["k_chunk_size"], pc["exp_approx_mode"], pc["max_cores_per_head_batch"]) == (64, 64, 0, 16)
    ck = pre.attrs["compute_kernel_config"]
    assert (ck["math_fidelity"], ck["math_approx_mode"], ck["fp32_dest_acc_en"], ck["packer_l1_acc"]) == ("HiFi4", 0, 1, 1)
    assert ck["throttle_level"] == "NO_THROTTLE" and pre.attrs["chunk_start_idx"] is None
    assert pre.attrs["is_causal"] is True and pre.attrs["is_windowed"] is False and pre.attrs["use_mla"] is False
    assert sv._common_attrs(pre)["grid_cores"] == 64
    cfg = sv.prefill_config(pre)
    assert (cfg.S, cfg.q_chunk, cfg.k_chunk, cfg.num_cores, cfg.fidelity, cfg.exp_approx_mode, cfg.fp32_dest_acc) == \
        (1024, 64, 64, 64, "HiFi4", False, True)
    res = sv.predict_row(pre)
    assert not res.excluded and abs(res.err) <= 0.01            # 347026 ns predicted against 347015 measured
    assert "legacy_path_production_family" in res.reasons


@pytest.mark.unit
def test_real_decode_row_parses_and_scores_with_the_sidecar():
    _, dec = _real_rows()
    assert dec.op_code == "SdpaDecodeDeviceOperation" and dec.meas_ns == 281932.0
    assert dec.attrs["cur_pos"] == [] and dec.attrs["paged_attention"] is True and dec.attrs["k_chunk_size"] == 0
    assert dec.attrs["program_config"]["compute_with_storage_grid_size"] == (8, 8)
    assert [t.shape for t in dec.inputs] == [[1, 32, 32, 128], [8192, 8, 32, 128], [8192, 8, 32, 128],
                                             [1, 1, 1, 32], [1, 1, 32, 256]]
    kw = sv.decode_config(dec, sidecar={"cur_pos": 1024, "page_block_size": 32})
    # 8192 blocks of 32 tokens over 32 users and a 256-wide table: 8192 tokens per user, position from the sidecar
    assert (kw["cache_len"], kw["cur_pos"], kw["batch"], kw["num_cores"], kw["k_chunk"]) == (8192, 1024, 32, 64, 0)
    assert dec.inputs[0].memory == "DEV_0_L1_HEIGHT_SHARDED" and kw["q_in_dram"] is False
    assert (kw["fidelity"], kw["accum_dtype"], kw["input_dtype"], kw["kv_input_dtype"]) == ("HiFi2", "float32", "bfloat16", "bfp8_b")
    run = sv.validate(FIXTURE, positions_csv=POSITIONS)
    res = [x for x in run.results if x.regime == "decode"][1]
    assert not res.excluded and res.config["attended"] == 1025 and res.config["k_chunk"] == 128
    assert res.config["q_in_dram"] is False and res.config["kv_bytes"] == 32 * 8 * 36 * 8 * 1088
    # validation-set row, not a fit target: the T2.8 law (fit on the direct-op sweep) puts it within 1 percent
    assert abs(res.err) <= 0.03
