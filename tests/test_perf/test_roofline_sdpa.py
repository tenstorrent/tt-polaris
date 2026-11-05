# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""SDPA roofline (TEN-4716) driven through the ttsim device cost path, pinned on the walls and perf counters
of one campaign (p100a, firmware 19.9.0, tt-metal 72620d5 kernels): T2.1 to T2.4 and T2.8 zone, counter,
production and decode blocks, the R1a to R1g re-measurement blocks. The R1c and R1g hold-outs and the T2.5
model-level rows are validation only and pin nothing."""
from __future__ import annotations

import math
from dataclasses import replace
from types import SimpleNamespace

import pytest

from ttsim.back.device import Device
from ttsim.perf.roofline_sdpa import (SdpaConfig, ArchConfig, predict, predict_decode, TAIL_RATE_MIN_CORES,
                                       sdpa_perf_stats, sdpa_config_from_shapes, ARCH_BH,
                                       WALL_TERMS_BH, WallTerms, WALL_ONLY_FIT, STREAM_RESIDUAL_FIT,
                                       MASK_BRACKET_PER_STEP, Section, StepGraph, KernelDescriptor,
                                       default_sdpa_step, evaluate_step, THREAD_SPLIT_FIELDS,
                                       THREAD_SPLIT_TERMS, _kv_stream_fixed, _pack_lane)

CLK = 1350.0


def _mla_cfg(S):
    return SdpaConfig(S=S, head_dim=576, v_head_dim=512, q_chunk=32, k_chunk=128,
                      num_heads=16, num_kv_heads=1, fidelity="HiFi4", input_dtype="bfloat16",
                      accum_dtype="bfloat16", is_causal=True, exp_approx_mode=False,
                      num_cores=110, arch=ARCH_BH)


def _baseline_cfg(S):
    return SdpaConfig(S=S, num_cores=110, arch=ARCH_BH)


class _MockIPGroup:
    def __init__(self, iptype):
        self.iptype = iptype


class _MockSimConfig:
    def __init__(self, freq_mhz=1350, devname="Blackhole", name=None):
        self._freq_mhz = freq_mhz
        self.devname = devname
        self.name = name or ("p100a" if devname == "Blackhole" else "test_device")
        self.ipgroups = [_MockIPGroup("compute"), _MockIPGroup("memory")]

    def frequency(self, pipe, units="MHz"):
        return self._freq_mhz

    def mem_frequency(self, units="MHz"):
        return self._freq_mhz

    def mem_size(self, units="GB"):
        return 32.0

    def peak_bandwidth(self, freq_units="GHz"):
        return 1000.0

    def peak_bandwidth_per_cycle(self):
        return 10.0

    def peak_flops(self, pipe, instr, precision, mul_factor=1):
        return 100.0

    def peak_ipc(self, pipe, instr, precision):
        return 128.0


class _RealBHSimConfig(_MockSimConfig):
    # BH p100a device params for the whole-pipeline correlation (real DRAM BW + mem clock).
    worker_core_count = 110     # as config/tt_bh.yaml declares for p100a

    def mem_frequency(self, units="MHz"):
        return 1000.0

    def peak_bandwidth_per_cycle(self):
        return 448.0        # 448 GB/s at 1 GHz mem clock -> bytes/cycle

    def ramp_penalty(self):
        return 100.0


def _sdpa_op(name, cfg):
    return SimpleNamespace(
        name=name, optype="SDPA", uses_compute_pipe="matrix", precision="bfp8",
        repeat_count=1, removed_in_optimization=False, fused_in_optimization=False,
        fused_with_op=None, fused_op_cycles=None, exec_stats={},
        compute_cycles=0, mem_rd_cycles=0, mem_wr_cycles=0,
        mem_rd_cycles_fractional=0.0, mem_wr_cycles_fractional=0.0,
        perf_stats=sdpa_perf_stats(cfg),
    )


@pytest.mark.unit
@pytest.mark.parametrize("S", [1024, 4096, 8192, 32768])
def test_fused_compute_cycles_override_is_honored(S):
    device = Device(_MockSimConfig())
    cfg = _baseline_cfg(S)
    op = _sdpa_op(f"sdpa_S{S}", cfg)
    device.execute_op(op)
    assert op.compute_cycles == int(math.ceil(predict(cfg).wall_clock_cycles))
    ipc_path = sum(math.ceil(c / (128.0 * device.DG_COMPUTE_UTIL_CONSTANT))
                   for c in op.perf_stats["instrs"].values())
    assert op.compute_cycles != ipc_path


@pytest.mark.unit
def test_arch_gate_falls_back_on_non_bh_device():
    # The roofline is BH-calibrated; off-BH execute_op drops it for the generic estimate, not crash.
    device = Device(_MockSimConfig(devname="Wormhole"))
    op = _sdpa_op("sdpa_wh", _baseline_cfg(4096))
    op.perf_stats["instrs"] = {"mov": 4096}   # sinf leaves a device-agnostic fallback count
    device.execute_op(op)
    assert op.compute_cycles != int(op.perf_stats["fused_compute_cycles"])


@pytest.mark.unit
def test_arch_gate_falls_back_on_other_bh_sku():
    # Constants are p100a-calibrated; a p150a instance of the same package must fall back too.
    device = Device(_MockSimConfig(devname="Blackhole", name="p150a"))
    op = _sdpa_op("sdpa_p150", _baseline_cfg(4096))
    op.perf_stats["instrs"] = {"mov": 4096}
    device.execute_op(op)
    assert op.compute_cycles != int(op.perf_stats["fused_compute_cycles"])
    # and the calibrated SKU keeps the roofline
    device = Device(_MockSimConfig(devname="Blackhole", name="p100a"))
    op = _sdpa_op("sdpa_p100", _baseline_cfg(4096))
    device.execute_op(op)
    assert op.compute_cycles == int(op.perf_stats["fused_compute_cycles"])


@pytest.mark.unit
def test_arch_gate_falls_back_off_bh():
    # Decode is BH-calibrated, so off-BH the gate must drop the roofline cost and use the generic
    # instr estimate (not crash, and not book the BH cost).
    from ttsim.perf.roofline_sdpa import decode_perf_stats
    device = Device(_MockSimConfig(devname="Wormhole"))
    ps = decode_perf_stats([1, 32, 1, 128], [1, 8, 4096, 128], attrs={"element_size": 2})
    ps["instrs"] = {"mov": 128}   # sinf leaves a device-agnostic fallback count
    op = SimpleNamespace(
        name="dec_wh", optype="SDPA", uses_compute_pipe="matrix", precision="bfp8",
        repeat_count=1, removed_in_optimization=False, fused_in_optimization=False,
        fused_with_op=None, fused_op_cycles=None, exec_stats={}, compute_cycles=0,
        compute_is_lower_bound=False,
        mem_rd_cycles=0, mem_wr_cycles=0, mem_rd_cycles_fractional=0.0,
        mem_wr_cycles_fractional=0.0, perf_stats=ps)
    device.execute_op(op)
    assert op.compute_cycles != int(ps["fused_compute_cycles"])   # generic path, not the BH cost


@pytest.mark.unit
def test_decode_v_head_dim_from_v_cache_shape():
    # FlashMLA decode: v_head_dim comes from the v_cache tensor, with attrs as the only fallback.
    from ttsim.perf.roofline_sdpa import decode_config_from_shapes
    cfg = decode_config_from_shapes([1, 32, 1, 576], [1, 1, 4096, 576],
                                    v_shape=[1, 1, 4096, 512], attrs={})
    assert cfg["v_head_dim"] == 512
    fallback = decode_config_from_shapes([1, 32, 1, 576], [1, 1, 4096, 576],
                                         attrs={"head_dim_v": 512})
    assert fallback["v_head_dim"] == 512


@pytest.mark.unit
def test_decode_layout_is_told_apart_by_the_cache_batch():
    # An MQA query [1, 32, 1, 128] is 32 users of one head when the cache (or the page table) carries
    # batch 32; over a batch 1 cache the same shape stays [1, heads, 1, d]. [batch, heads, 1, d] is untouched.
    from ttsim.perf.roofline_sdpa import decode_config_from_shapes
    mqa = decode_config_from_shapes([1, 32, 1, 128], [32, 1, 4096, 128], attrs={})
    assert (mqa["batch"], mqa["num_q_heads"], mqa["num_kv_heads"]) == (32, 1, 1)
    paged = decode_config_from_shapes([1, 32, 1, 128], [1024, 1, 32, 128], attrs={"paged": True},
                                      page_table_shape=[32, 64])
    assert (paged["batch"], paged["num_q_heads"], paged["cache_len"]) == (32, 1, 1024)
    one = decode_config_from_shapes([1, 32, 1, 128], [1, 8, 4096, 128], attrs={})
    assert (one["batch"], one["num_q_heads"]) == (1, 32)
    alt = decode_config_from_shapes([32, 8, 1, 128], [32, 8, 4096, 128], attrs={})
    assert (alt["batch"], alt["num_q_heads"]) == (32, 8)


@pytest.mark.unit
def test_decode_kv_heads_zero_is_the_mha_shorthand():
    mha = predict_decode(cache_len=4096, num_q_heads=8, num_kv_heads=0, head_dim=128, batch=4)
    same = predict_decode(cache_len=4096, num_q_heads=8, num_kv_heads=8, head_dim=128, batch=4)
    assert mha.wall_clock_cycles == same.wall_clock_cycles and mha.dram_in_bytes == same.dram_in_bytes


@pytest.mark.unit
def test_output_bytes_follow_the_query_dtype_not_the_accumulator():
    # sdpa_device_operation.cpp compute_output_specs writes the output in q's dtype, fp32 DEST or not.
    from ttsim.perf.roofline_sdpa import BYTES_PER_TILE
    bf16 = predict(SdpaConfig(S=4096, num_cores=110, arch=ARCH_BH))
    fp32 = predict(SdpaConfig(S=4096, num_cores=110, arch=ARCH_BH, accum_dtype="float32"))
    assert fp32.dram_out_bytes == bf16.dram_out_bytes == 32 * 128 * 4 * BYTES_PER_TILE["bfp8_b"]
    sparse = [predict(SdpaConfig(S=2048, is_sparse=True, kv_seq=2048, num_cores=110, arch=ARCH_BH,
                                 accum_dtype=d)).dram_out_bytes for d in ("bfloat16", "float32")]
    assert sparse[0] == sparse[1] == 32 * 64 * 4 * BYTES_PER_TILE["bfp8_b"]


@pytest.mark.unit
def test_wall_regime_cross_only_when_kv_seq_differs():
    # kv_seq == S is plain self-attention, not cross, so it keeps the prefill wall terms.
    from ttsim.perf.roofline_sdpa import _wall_regime
    r = SimpleNamespace(regime="prefill", is_mla=False)
    same = SdpaConfig(S=4096, kv_seq=4096, is_causal=True, is_sparse=False)
    cross = SdpaConfig(S=4096, kv_seq=2048, is_causal=True, is_sparse=False)
    assert _wall_regime(same, r) == "prefill_causal"
    assert _wall_regime(cross, r) == "cross"
    assert predict(SdpaConfig(S=4096, kv_seq=2048, num_cores=110, arch=ARCH_BH)).wall_regime == "cross"


@pytest.mark.unit
def test_op_cost_is_wall_clock_with_floor_in_breakdown():
    # The op cost is now the full wall-clock estimate (not a floor); the compute floor stays in the
    # breakdown for reference.
    cfg = _baseline_cfg(4096)
    ps = sdpa_perf_stats(cfg)
    assert ps["sdpa_compute_is_floor"] is False and ps["sdpa_calibrated_arch"] == "Blackhole"
    assert ps["fused_compute_cycles"] == predict(cfg).wall_clock_cycles
    assert ps["sdpa_cycle_breakdown"]["compute_floor"] == predict(cfg).compute_latency_cycles
    assert ps["fused_compute_cycles"] > ps["sdpa_cycle_breakdown"]["compute_floor"]


@pytest.mark.unit
def test_floor_flag_is_accepted_by_opstats_schema():
    # Regression: execute_op writes compute_is_lower_bound into op.exec_stats, which hlmstats
    # merges into rows validated by TTSimHLWlDevRunOperatorPerfStats (extra='forbid'). The field
    # must be declared there or dump_stats raises ValidationError for every op.
    from ttsim.config.validators import TTSimHLWlDevRunOperatorPerfStats as Row
    assert "compute_is_lower_bound" in Row.model_fields
    row = dict(pipe="matrix", precision="bfp8", opnum=0, opname="sdpa", is_input_node=False,
               is_output_node=False, optype="ScaledDotProductAttention", op_rpt_count=1, attrs={},
               inList=[], outList=[], input_tensors="[]", output_tensors="[]", weight_tensors="[]",
               domain="ttnn", opclass="COMMON", removed=False, fused=False, fused_with_op="NA",
               inElems=0, outElems=0, inBytes=1, outBytes=1, instrs={}, inParamCount=0, inActCount=0,
               outActCount=0, instr_count=0, compute_cycles=1.0, mem_rd_cycles=0.0, mem_wr_cycles=0.0,
               ramp_penalty=100.0, rsrc_bnck="COMP", ideal_cycles=1.0, ideal_msecs=1.0, cycles=1.0,
               matrix_cycles=1.0, vector_cycles=0.0, msecs=1.0, matrix_pipe_util=0.5,
               vector_pipe_util=0.0, mem_rd_util=0.0, mem_wr_util=0.0, memory_traffic=0.0,
               mem_util=0.0, uses_perf_lookup=False)
    assert Row(**row, compute_is_lower_bound=True).compute_is_lower_bound is True


@pytest.mark.unit
def test_mla_config_is_flagged_low_confidence():
    # Asymmetric v_head_dim (FlashMLA) is emitted low-confidence: it is calibrated on a single
    # config family, so off-family MLA shapes should be treated as lower confidence.
    assert sdpa_perf_stats(_mla_cfg(4096))["sdpa_mla_low_confidence"] is True
    # Symmetric SDPA stays high-confidence.
    assert sdpa_perf_stats(_baseline_cfg(4096))["sdpa_mla_low_confidence"] is False


@pytest.mark.unit
def test_generic_instrs_path_unchanged_without_override():
    device = Device(_MockSimConfig())
    op = SimpleNamespace(
        name="plain_mm", optype="MatMul", uses_compute_pipe="matrix", precision="fp32",
        repeat_count=1, removed_in_optimization=False, fused_in_optimization=False,
        fused_with_op=None, fused_op_cycles=None, exec_stats={},
        compute_cycles=0, mem_rd_cycles=0, mem_wr_cycles=0,
        mem_rd_cycles_fractional=0.0, mem_wr_cycles_fractional=0.0,
        perf_stats={"inBytes": 4000, "outBytes": 4000, "instrs": {"mac": 4096}},
    )
    device.execute_op(op)
    assert op.compute_cycles == math.ceil(4096 / (128.0 * device.DG_COMPUTE_UTIL_CONSTANT))


@pytest.mark.unit
def test_sdpa_config_from_shapes_gqa():
    from ttsim.perf.roofline_sdpa import sdpa_config_from_shapes
    # Llama GQA: q[1,32,4096,128], k/v[1,8,4096,128], causal
    cfg = sdpa_config_from_shapes([1, 32, 4096, 128], [1, 8, 4096, 128], [1, 8, 4096, 128],
                                  {"is_causal": True, "element_size": 1})
    assert (cfg.num_heads, cfg.num_kv_heads, cfg.head_dim, cfg.S) == (32, 8, 128, 4096)
    assert cfg.input_dtype == "bfp8_b" and cfg.is_causal
    ps = sdpa_perf_stats(cfg)
    assert ps["fused_compute_cycles"] > 0 and ps["inBytes"] > 0


@pytest.mark.unit
def test_sdpa_sinf_routes_prefill_and_decode_variants():
    from ttsim.ops.desc.ttsim_layout import sdpa_sinf
    def T(shape):
        return SimpleNamespace(shape=list(shape), dtype="bfloat16")
    # prefill: 3 inputs -> compute roofline (wall-clock cost)
    q, k, v = T([1, 32, 4096, 128]), T([1, 8, 4096, 128]), T([1, 8, 4096, 128])
    out = SimpleNamespace(shape=None, dtype=None)
    op = SimpleNamespace(attrs={"is_causal": True, "element_size": 1}, perf_stats=None)
    sdpa_sinf([q, k, v], [out], op)
    assert op.perf_stats.get("fused_compute_cycles", 0) > 0 and op.perf_stats["inBytes"] > 0
    assert op.perf_stats["sdpa_regime"] == "prefill"
    # decode: 4 inputs (q, k_cache, v_cache, cur_pos) -> MEMORY-bound KV-stream roofline
    op2 = SimpleNamespace(attrs={"element_size": 2}, perf_stats=None)
    sdpa_sinf([T([1, 32, 1, 128]), k, v, T([1])], [SimpleNamespace(shape=None, dtype=None)], op2)
    assert op2.perf_stats["sdpa_regime"] == "decode"
    assert op2.perf_stats["sdpa_cycle_breakdown"]["is_memory_bound"] is True
    assert op2.perf_stats["sdpa_compute_is_floor"] is False  # memory bound, not a compute floor
    # KV cache (8 heads x 4096 x 128 x2) dominates the tiny single-token compute:
    assert op2.perf_stats["inBytes"] > 10_000_000


@pytest.mark.unit
def test_sdpa_sinf_routes_mla_sparse_joint_mladecode_variants():
    # The distinct MLA / sparse / joint / MLA-decode ops route via the sdpa_variant tag and produce
    # a real roofline cost (not the passthrough mov-estimate).
    from ttsim.ops.desc.ttsim_layout import sdpa_sinf
    def T(shape):
        return SimpleNamespace(shape=list(shape), dtype="bfloat16")
    def run(iTList, attrs):
        op = SimpleNamespace(attrs=attrs, perf_stats=None)
        sdpa_sinf(iTList, [SimpleNamespace(shape=None, dtype=None)], op)
        return op.perf_stats

    # MLA prefill: V = K (latent), head_dim_v=512 asymmetric -> flagged low-confidence.
    mla = run([T([1, 16, 4096, 576]), T([1, 1, 4096, 576]), T([1, 1, 4096, 576])],
              {"sdpa_variant": "mla", "head_dim_v": 512, "is_causal": True, "fidelity": "HiFi4",
               "input_dtype": "bfloat16", "exp_approx_mode": False})
    assert mla["fused_compute_cycles"] > 0 and mla["sdpa_mla_low_confidence"] is True
    assert mla["sdpa_regime"] and mla["outBytes"] > 0   # real roofline cost, not passthrough

    # Sparse: kv_seq = TOPK, is_sparse -> the per-token gather law.
    sp = run([T([1, 32, 2048, 576]), T([1, 1, 8192, 576]), T([1, 1, 8192, 576]), T([1, 1, 2048, 1024])],
             {"sdpa_variant": "sparse", "is_sparse": True, "head_dim_v": 512, "kv_seq": 1024,
              "is_causal": False, "fidelity": "HiFi4", "input_dtype": "bfloat16", "q_chunk_size": 128, "k_chunk_size": 128})
    assert sp["fused_compute_cycles"] > 0 and sp["sdpa_wall_regime"] == "sparse"

    # Joint (SD3/Flux): cost over S_eff = main + joint, non-causal.
    jt = run([T([1, 24, 4096, 64]), T([1, 24, 4096, 64]), T([1, 24, 4096, 64])],
             {"sdpa_variant": "joint", "joint_seq": 512, "is_causal": False})
    # no program config on the call: the kernel runs 32x32 chunks, so the expected config says so
    jt_expected = predict(SdpaConfig(S=4096 + 512, joint_seq=512, num_heads=24, head_dim=64, is_causal=False,
                                     is_joint=True, input_dtype="bfloat16", q_chunk=32, k_chunk=32,
                                     num_cores=110, arch=ARCH_BH))
    assert jt["fused_compute_cycles"] == jt_expected.wall_clock_cycles
    assert jt["sdpa_regime"] == "joint"

    # MLA decode: memory-bound; the latent V head dim comes from the attr, and without mla_v_read the
    # K tensor is streamed alone (V is its latent part).
    dec = run([T([1, 128, 1, 576]), T([1, 1, 8192, 576]), T([1, 1, 8192, 576]), T([1])],
              {"sdpa_variant": "mla_decode", "head_dim_v": 512})
    assert dec["sdpa_cycle_breakdown"]["is_memory_bound"] is True and dec["inBytes"] > 0
    assert dec["sdpa_wall_regime"] == "mla_decode" and dec["sdpa_config"]["v_head_dim"] == 512
    assert dec["sdpa_config"]["mla_v_read"] is False and dec["sdpa_config"]["kv_bytes"] == 1 * 1 * 256 * 18 * 2048


@pytest.mark.unit
def test_decode_is_memory_bound_and_scales_with_kv_cache():
    from ttsim.perf.roofline_sdpa import predict_decode
    # KV-stream bytes must dominate the single-token compute, and grow linearly with cache_len.
    r1 = predict_decode(cache_len=4096, num_q_heads=32, num_kv_heads=8, head_dim=128)
    r2 = predict_decode(cache_len=8192, num_q_heads=32, num_kv_heads=8, head_dim=128)
    assert r1.is_memory_bound and r1.low_confidence and r1.regime == "decode"
    # 2x cache -> 2x KV bytes; the query read (one tile block per active core) does not grow with it
    assert r2.config_echo["kv_bytes"] == 2 * r1.config_echo["kv_bytes"]
    assert r2.config_echo["q_bytes"] == r1.config_echo["q_bytes"] > 0
    assert r2.dram_in_bytes == r2.config_echo["kv_bytes"] + r2.config_echo["q_bytes"]
    # memory volume vastly exceeds the single-token compute (the whole point of P1)
    assert r1.dram_in_bytes > 100 * r1.math_active_cycles


@pytest.mark.unit
def test_decode_sliding_window_caps_kv_stream():
    from ttsim.perf.roofline_sdpa import predict_decode
    # Windowed decode (Gemma/Mistral SWA) attends only the last `window` keys -> KV stream (and
    # the memory bound) caps at the window, not the full cache.
    full = predict_decode(cache_len=32768, num_q_heads=32, num_kv_heads=8, head_dim=128)
    win = predict_decode(cache_len=32768, num_q_heads=32, num_kv_heads=8, head_dim=128,
                         sliding_window=4096)
    assert win.dram_in_bytes < full.dram_in_bytes
    # capped at the window: same bytes as a full cache of the window length
    capped = predict_decode(cache_len=4096, num_q_heads=32, num_kv_heads=8, head_dim=128)
    assert win.dram_in_bytes == capped.dram_in_bytes


@pytest.mark.unit
def test_decode_rate_by_grid_and_paging_and_its_flags():
    from ttsim.perf.roofline_sdpa import _decode_kv_stream_gbps, DECODE_CALIBRATED_GRIDS
    a = ARCH_BH
    assert DECODE_CALIBRATED_GRIDS == (64, 110)
    assert _decode_kv_stream_gbps(a, 64, True) == 298.9 and _decode_kv_stream_gbps(a, 110, True) == 330.5
    assert 298.9 < _decode_kv_stream_gbps(a, 80, True) < 330.5
    assert _decode_kv_stream_gbps(a, 32, True) == 298.9 and _decode_kv_stream_gbps(a, 130, True) == 330.5
    # non-paged: the R1b 110-grid rate, scaled by the paged grid ratio elsewhere
    assert _decode_kv_stream_gbps(a, 110, False) == a.decode_kv_stream_gbps_nonpaged == 342.3
    assert _decode_kv_stream_gbps(a, 64, False) == pytest.approx(342.3 * 298.9 / 330.5)
    for grid in (32, 80, 130):
        assert "decode_grid_uncalibrated" in _t28(grid, 32, "bfp8_b", 1024).low_confidence_reasons
    np_ = _t28(64, 32, "bfp8_b", 1024, paged=False)
    assert "decode_nonpaged_grid_inferred" in np_.low_confidence_reasons and np_.config_echo["kv_stream_gbps"] == 309.6
    assert "decode_nonpaged_grid_inferred" not in _t28(110, 32, "bfp8_b", 1024, paged=False).low_confidence_reasons
    assert "decode_page_block_uncalibrated" in _t28(64, 32, "bfp8_b", 1024, page_block_size=128).low_confidence_reasons
    assert "decode_page_block_uncalibrated" not in _t28(64, 32, "bfp8_b", 1024).low_confidence_reasons
    assert "paged_geometry_unmodelled" not in _t28(64, 32, "bfp8_b", 1024).low_confidence_reasons


@pytest.mark.unit
def test_decode_core_split_follows_the_factory():
    from ttsim.perf.roofline_sdpa import _decode_core_split
    # sdpa_decode_program_factory.cpp:196-209 with max_cores_per_head_batch 16 (the production config)
    assert _decode_core_split(64, 32, 8, 16) == (1, 4, 64)
    assert _decode_core_split(110, 32, 8, 16) == (1, 4, 64)     # the 11x10 grid spreads the same 64 cores
    assert _decode_core_split(64, 8, 8, 16) == (1, 1, 64)
    assert _decode_core_split(64, 16, 8, 16) == (1, 2, 64)
    assert _decode_core_split(110, 8, 8, 16) == (1, 1, 64)      # R1b non-paged b8: 64 cores on the 110 grid
    assert _decode_core_split(110, 8, 1, 16) == (13, 1, 104)    # MLA b8 nh16: 13 cores per user, tree reduction
    assert _decode_core_split(110, 16, 1, 16) == (6, 1, 96)     # MLA b4 nh128, 4 head slices: 16 virtual users
    assert _decode_core_split(110, 32, 1, 16) == (3, 1, 96)     # MLA b8 nh128
    assert _decode_core_split(64, 1, 8, 0) == (8, 1, 64)        # no program config: the whole grid


@pytest.mark.unit
def test_decode_chunk_rules_follow_the_kernel():
    from ttsim.perf.roofline_sdpa import decode_dynamic_chunk_tiles
    # paged k_chunk 0: tiles of cur_pos + 1 rounded up to a power of two, capped at the DEST size
    assert [decode_dynamic_chunk_tiles(p, 4) for p in (0, 31, 32, 64, 96, 128, 1024, 4096)] == [1, 1, 2, 4, 4, 4, 4, 4]
    assert decode_dynamic_chunk_tiles(1024, 8) == 8 and decode_dynamic_chunk_tiles(64, 8) == 4
    # whole chunks are streamed: 8 / 36 / 132 tiles per head and user at positions 128 / 1024 / 4096
    for pos, tiles in ((128, 8), (1024, 36), (4096, 132)):
        r = _t28(64, 32, "bfp8_b", pos)
        assert r.config_echo["k_chunk"] == 128 and r.config_echo["kv_bytes"] == 32 * 8 * tiles * 8 * 1088
    # bf16 DEST doubles the cap; an explicit chunk is used as given
    assert _t28(64, 32, "bfp8_b", 1024, accum_dtype="bfloat16").config_echo["k_chunk"] == 256
    assert _t28(64, 32, "bfp8_b", 1024, k_chunk=512).config_echo["k_chunk"] == 512
    # non-paged k_chunk 0: the op picks it from the cache length (sdpa_decode.cpp get_chunk_size)
    r = predict_decode(cache_len=4096, cur_pos=1024, num_q_heads=32, num_kv_heads=8, head_dim=128, k_chunk=0)
    assert r.config_echo["k_chunk"] == 512 and r.config_echo["kv_bytes"] == 8 * 48 * 8 * 2048


@pytest.mark.unit
def test_decode_q_and_output_bytes_follow_the_kernel_tiles():
    # reader_decode_all.cpp reads one padded head-row tile block (PNHt x DHt tiles) per active core;
    # the output core writes PNHt x vDHt tiles per user in the query dtype.
    r = _t28(64, 32, "bfp8_b", 1024)
    assert r.config_echo["q_bytes"] == 64 * 1 * 4 * 2048 and r.dram_out_bytes == 32 * 1 * 4 * 2048
    assert _t28(64, 32, "bfp8_b", 1024, q_in_dram=False).config_echo["q_bytes"] == 0
    mla = predict_decode(cache_len=4096, num_q_heads=16, num_kv_heads=1, head_dim=576, v_head_dim=512, batch=8, num_cores=110)
    assert mla.config_echo["q_bytes"] == 104 * 1 * 18 * 2048


@pytest.mark.unit
def test_decode_components_sum_to_wall_when_memory_or_compute_bound():
    cases = [_t28(64, 32, "bfp8_b", 4096), _t28(110, 8, "bfloat16", 128),
             predict_decode(cache_len=4096, num_q_heads=16, num_kv_heads=1, head_dim=576, v_head_dim=512, batch=8),
             predict_decode(cache_len=128, cur_pos=127, num_q_heads=32, num_kv_heads=32, head_dim=128, num_cores=1,
                            fidelity="HiFi4")]
    for r in cases:
        assert list(r.components) == ["init", "compute_floor", "kv_stream_wait"]
        assert round(sum(r.components.values())) == r.wall_clock_cycles
        assert all(v >= 0.0 for v in r.components.values()) and r.components["init"] > 0.0
    # 32 MHA heads on one core at HiFi4 are compute bound: the stream hides under the math, the wait is zero
    assert cases[-1].components["kv_stream_wait"] == 0.0 and cases[0].components["kv_stream_wait"] > 0.0
    assert cases[-1].wall_clock_cycles == round(cases[-1].components["init"] + cases[-1].math_active_cycles)


@pytest.mark.unit
def test_decode_kv_bytes_scale_with_kv_heads_not_query_heads():
    from ttsim.perf.roofline_sdpa import predict_decode
    # GQA: KV traffic depends on num_kv_heads, not num_q_heads (MQA reads the least).
    mha = predict_decode(cache_len=4096, num_q_heads=32, num_kv_heads=32, head_dim=128)
    gqa = predict_decode(cache_len=4096, num_q_heads=32, num_kv_heads=8, head_dim=128)
    mqa = predict_decode(cache_len=4096, num_q_heads=32, num_kv_heads=1, head_dim=128)
    assert mha.dram_in_bytes > gqa.dram_in_bytes > mqa.dram_in_bytes


@pytest.mark.unit
def test_mla_decode_reuses_k_as_v_cutting_dram():
    from ttsim.perf.roofline_sdpa import predict_decode
    # Without a V tensor MLA decode reuses K's latent part as V (an L1 transpose), so its KV DRAM read
    # excludes V; with a V tensor the V tiles are streamed too (the R1b non-paged form).
    mla = predict_decode(cache_len=4096, num_q_heads=16, num_kv_heads=1, head_dim=576,
                         v_head_dim=512, fidelity="HiFi4")
    assert mla.is_mla and mla.is_memory_bound and mla.config_echo["mla_v_read"] is False
    sym = predict_decode(cache_len=4096, num_q_heads=16, num_kv_heads=1, head_dim=576)
    assert mla.dram_in_bytes < sym.dram_in_bytes
    withv = predict_decode(cache_len=4096, num_q_heads=16, num_kv_heads=1, head_dim=576, v_head_dim=512, fidelity="HiFi4",
                           mla_v_read=True)
    assert withv.config_echo["kv_bytes"] == mla.config_echo["kv_bytes"] * 34 / 18


@pytest.mark.unit
def test_sliding_window_saturates_not_unbounded():
    # A sliding window must SATURATE at W (not grow like the (K+1)/2 causal diagonal). This is the
    # Gemma/Mistral SWA overcount fix: at S=32k the window K_eff must be far below full-causal.
    full = predict(SdpaConfig(S=32768, num_cores=110, arch=ARCH_BH))
    win = predict(SdpaConfig(S=32768, sliding_window=4096, num_cores=110, arch=ARCH_BH))
    assert win.k_eff < 0.3 * full.k_eff
    assert win.fpu_matmul_cycles < full.fpu_matmul_cycles
    assert win.regime == "windowed" and win.low_confidence
    assert win.sfpu_exp_cycles < full.sfpu_exp_cycles       # the band's k tiles, not the causal triangle
    # Doubling S at fixed window leaves per-q-chunk K_eff ~unchanged (bounded by the window).
    win2 = predict(SdpaConfig(S=65536, sliding_window=4096, num_cores=110, arch=ARCH_BH))
    assert win2.k_eff == pytest.approx(win.k_eff, rel=0.05)


@pytest.mark.unit
def test_cross_attention_kv_seq_differs_from_q_seq():
    # Cross-attention (kv_seq != q_seq): K count and K/V DRAM follow kv_seq, Q/output follow S.
    cross = predict(SdpaConfig(S=1024, kv_seq=4096, num_heads=16, is_causal=False,
                               num_cores=110, arch=ARCH_BH))
    assert cross.k_chunks_per_q == 4096 // 128       # K from kv_seq, not S
    # same shapes but kv_seq==S would read less K/V DRAM
    square = predict(SdpaConfig(S=1024, num_heads=16, is_causal=False, num_cores=110, arch=ARCH_BH))
    assert cross.dram_in_bytes > square.dram_in_bytes


@pytest.mark.unit
def test_chunked_prefill_adds_dense_prefix_and_scatter_derate():
    # Chunked/paged prefill: this Q chunk attends the full DENSE prefix (rectangular, no causal
    # halving) plus the causal ramp within the chunk. K_eff = prefix/k_chunk + (q_nc+1)/2.
    ck = predict(SdpaConfig(S=2048, kv_seq=6144, chunk_start_idx=4096, q_chunk=128, k_chunk=128,
                            num_cores=110, arch=ARCH_BH))
    assert ck.k_eff == pytest.approx(4096 / 128 + (2048 // 128 + 1) / 2)  # 40.5
    assert ck.regime == "chunked" and ck.low_confidence
    # a plain first chunk (no prefix) visits far fewer K-chunks
    plain = predict(SdpaConfig(S=2048, q_chunk=128, k_chunk=128, num_cores=110, arch=ARCH_BH))
    assert ck.k_eff > plain.k_eff
    # paged scatter derate inflates effective K/V DRAM bytes vs a non-scattered read
    no_derate = predict(SdpaConfig(S=2048, kv_seq=6144, chunk_start_idx=4096, q_chunk=128,
                                   k_chunk=128, dram_scatter_derate=1.0, num_cores=110, arch=ARCH_BH))
    derated = predict(SdpaConfig(S=2048, kv_seq=6144, chunk_start_idx=4096, q_chunk=128,
                                 k_chunk=128, dram_scatter_derate=1.15, num_cores=110, arch=ARCH_BH))
    assert derated.dram_in_bytes > no_derate.dram_in_bytes


@pytest.mark.unit
def test_router_dispatches_chunked_prefill():
    from ttsim.ops.desc.ttsim_layout import _infer_sdpa_variant
    assert _infer_sdpa_variant([0, 1, 2], {}) == "prefill"
    assert _infer_sdpa_variant([0, 1, 2, 3], {}) == "decode"
    assert _infer_sdpa_variant([0, 1, 2, 3], {"chunk_start_idx": 4096}) == "chunked"
    assert _infer_sdpa_variant([0, 1, 2, 3], {"chunk_start_idx": 0}) == "decode"  # first chunk == decode-arity


@pytest.mark.unit
def test_batch_scales_prefill_work_and_dram():
    # Batch>1 prefill: per-core work and whole-op DRAM scale linearly with batch (card-validated
    # -3.6/-3.7/-4.5% at b=1/2/4).
    b1 = predict(SdpaConfig(S=2048, num_heads=16, batch=1, num_cores=110, arch=ARCH_BH))
    b4 = predict(SdpaConfig(S=2048, num_heads=16, batch=4, num_cores=110, arch=ARCH_BH))
    assert b4.fpu_matmul_cycles == pytest.approx(4 * b1.fpu_matmul_cycles, rel=1e-3)  # ~ exact (rounding)
    assert b4.dram_in_bytes == 4 * b1.dram_in_bytes


def _polaris_device_us(perf_stats):
    # Projected device latency through the real Device.execute_op + get_exec_stats projection
    # (ideal = max(compute, mem) + ramp), with BH device params.
    device = Device(_RealBHSimConfig())
    op = _sdpa_op("corr", _baseline_cfg(4096))
    op.perf_stats = perf_stats
    device.execute_op(op)
    ideal = math.ceil(max(op.compute_cycles, op.mem_rd_cycles + op.mem_wr_cycles) + 100.0)
    return ideal / 1350.0


@pytest.mark.unit
def test_worker_core_count_comes_from_the_device_instance():
    from types import SimpleNamespace as NS
    from ttsim.back.device import resolve_worker_core_count
    ipg = NS(num_units=120)
    cfg = NS(worker_core_count=110, compute_grid_size=[12, 10], get_ipgroup=lambda iptype: ipg)
    assert resolve_worker_core_count(cfg) == 110                     # the explicit field wins
    cfg.worker_core_count = None
    assert resolve_worker_core_count(cfg) == 120                     # else the grid product
    cfg.compute_grid_size = None
    assert resolve_worker_core_count(cfg) == 120                     # else compute num_units
    assert resolve_worker_core_count(NS()) == 0                      # nothing to go on


@pytest.mark.unit
def test_device_reprices_sdpa_with_its_own_grid_and_clock():
    # Shape inference has no device, so it prices with the calibration's grid and clock; the backend
    # re-prices with the device's. p100a declares 110 worker cores at 1350 MHz, which is what the
    # constants were measured at, so booking is unchanged there.
    import ttsim.front.ttnn as ttnn
    dev = _shim_dev()
    ttnn.transformer.scaled_dot_product_attention(*_qkv(dev), is_causal=True)
    op = _last_sdpa(dev)
    assert op.perf_stats["sdpa_num_cores_defaulted"] is True
    assert op.perf_stats["sdpa_clock_ghz"] == pytest.approx(1.35)
    sinf_cycles = op.perf_stats["fused_compute_cycles"]

    device = Device(_RealBHSimConfig())
    assert device.worker_core_count == 110
    device.execute_op(op)
    assert op.compute_cycles == math.ceil(sinf_cycles)               # same grid, same clock: no change

    # Prefill cycles do not move with the clock: the K/V stream rate is fitted in bytes per cycle per
    # core, so a frequency sweep rescales the wall in ns, not in cycles.
    slow = Device(_RealBHSimConfig())
    slow.freq_MHz = 1000.0
    ttnn.transformer.scaled_dot_product_attention(*_qkv(dev), is_causal=True)
    op2 = _last_sdpa(dev)
    slow.execute_op(op2)
    assert op2.perf_stats["sdpa_clock_ghz"] == pytest.approx(1.0)
    assert op2.compute_cycles == math.ceil(sinf_cycles)

    # A narrower device grid: the op carried no program config, so the device's count is the one used.
    narrow = Device(_RealBHSimConfig())
    narrow.worker_core_count = 64
    ttnn.transformer.scaled_dot_product_attention(*_qkv(dev), is_causal=True)
    op3 = _last_sdpa(dev)
    narrow.execute_op(op3)
    assert op3.perf_stats["sdpa_config"]["num_cores"] == 64
    assert op3.compute_cycles > math.ceil(sinf_cycles)               # fewer cores, longer wall


@pytest.mark.unit
def test_decode_wall_follows_the_device_clock():
    # Decode is a DRAM stream priced in GB/s, so its cycle count does scale with the device clock: at
    # 1000 MHz the same bytes take 1.35x fewer cycles than at 1350.
    import ttsim.front.ttnn as ttnn
    dev = _shim_dev()
    q, kc, vc, pt, cp = _decode_tensors(dev)
    ttnn.transformer.paged_scaled_dot_product_attention_decode(q, kc, vc, page_table_tensor=pt,
                                                               cur_pos=[1023] * 32)
    op = _last_sdpa(dev)
    at_1350 = op.perf_stats["fused_compute_cycles"]
    slow = Device(_RealBHSimConfig())
    slow.freq_MHz = 1000.0
    slow.execute_op(op)
    assert op.perf_stats["sdpa_clock_ghz"] == pytest.approx(1.0)
    assert op.compute_cycles < math.ceil(at_1350)
    # same wall in ns either way: the byte stream is what it is
    assert op.compute_cycles / 1.0 == pytest.approx(at_1350 / 1.35, rel=0.02)


@pytest.mark.unit
def test_device_does_not_override_a_program_config_grid():
    # tt_transformers pins an (8, 8) grid. That is the op's own configuration, not a device default,
    # so the backend leaves it alone even though the device offers 110 cores.
    import ttsim.front.ttnn as ttnn
    dev = _shim_dev()
    pc, ck = _llama_prefill_cfgs()
    ttnn.transformer.scaled_dot_product_attention(*_qkv(dev), is_causal=True, program_config=pc,
                                                  compute_kernel_config=ck)
    op = _last_sdpa(dev)
    assert op.perf_stats["sdpa_num_cores_defaulted"] is False
    booked = op.perf_stats["fused_compute_cycles"]
    Device(_RealBHSimConfig()).execute_op(op)
    assert op.perf_stats["sdpa_config"]["num_cores"] == 64 and op.compute_cycles == math.ceil(booked)


@pytest.mark.unit
def test_reprice_keeps_the_regime_and_the_decode_bytes():
    # Re-pricing runs the same law, not a different one: the regime and the variant flags survive it,
    # and a decode's KV bytes follow the re-priced core split.
    from ttsim.perf.roofline_sdpa import decode_perf_stats, reprice
    ps = decode_perf_stats([1, 32, 1, 128], [1, 8, 4096, 128], attrs={"element_size": 2})
    again = reprice(ps["sdpa_reprice"], num_cores=64, clock_ghz=1.35, arch_name="Blackhole")
    assert again["sdpa_regime"] == ps["sdpa_regime"] == "decode"
    assert again["sdpa_config"]["num_cores"] == 64
    assert again["fused_compute_cycles"] != ps["fused_compute_cycles"]
    same = reprice(ps["sdpa_reprice"], num_cores=110, clock_ghz=1.35, arch_name="Blackhole")
    assert same["fused_compute_cycles"] == ps["fused_compute_cycles"]


@pytest.mark.unit
def test_multichip_ring_is_out_of_scope_not_handled():
    # This roofline is single-chip only. A ring/multichip SDPA variant must NOT be given a
    # single-chip cost -- it has no front-end handler and falls through to passthrough.
    from ttsim.ops.desc.ttsim_layout import _SDPA_FRONTENDS
    assert "ring_distributed" not in _SDPA_FRONTENDS
    assert "ring_joint" not in _SDPA_FRONTENDS
    # Single-chip variants only (joint = single-chip SD3/Flux, not the multichip ring_joint).
    assert set(_SDPA_FRONTENDS) == {"prefill", "decode", "chunked", "mla", "sparse", "mla_decode", "joint"}


@pytest.mark.unit
def test_perf_stats_has_all_keys_downstream_reads():
    # hlmstats / device stats read these unconditionally; missing any is a KeyError at runtime.
    ps = sdpa_perf_stats(_baseline_cfg(4096))
    required = {"inBytes", "outBytes", "inElems", "outElems", "inParamCount",
                "inActCount", "outActCount", "instrs", "fused_compute_cycles"}
    assert required.issubset(ps), f"missing: {required - set(ps)}"
    assert ps["fused_compute_cycles"] > 0 and ps["inBytes"] > 0
    assert ps["inElems"] == 0  # sentinel so the bytes-per-element precision check is skipped


@pytest.mark.unit
def test_predict_rejects_bad_dtype_and_fidelity_but_pads_shapes():
    # dtype/fidelity must be known. Non-chunk-divisible S is PADDED up (matching the kernel), not
    # rejected -- so a 32-aligned-but-not-128-divisible prefill stays on the roofline instead of
    # silently falling to the passthrough mov-cost. head_dim need not be a multiple of 32 either.
    with pytest.raises(ValueError):
        predict(SdpaConfig(S=4096, input_dtype="fp8_e4m3", num_cores=110, arch=ARCH_BH))
    with pytest.raises(ValueError):
        predict(SdpaConfig(S=4096, fidelity="HiFi9", num_cores=110, arch=ARCH_BH))
    with pytest.raises(ValueError):
        predict(SdpaConfig(S=4096, kv_input_dtype="int8", num_cores=110, arch=ARCH_BH))
    # S=5000 (not a multiple of q_chunk=128) is padded, not rejected:
    import math
    padded = predict(SdpaConfig(S=5000, num_cores=110, arch=ARCH_BH))
    assert padded.compute_latency_cycles > 0
    assert padded.q_chunks_per_core == (math.ceil(5000 / 128) * 32) / 110


@pytest.mark.unit
def test_offcalibration_chunking_is_low_confidence():
    # The lanes were measured on the S 4096 grid q {64, 128, 256, 512} x k {128, 256, 512}; other
    # chunkings extrapolate the laws and are flagged.
    for q, k in ((128, 128), (128, 256), (128, 512), (64, 128), (256, 128), (512, 512)):
        assert not predict(SdpaConfig(S=4096, q_chunk=q, k_chunk=k, num_cores=110, arch=ARCH_BH)).low_confidence
    assert "off_calibration_chunk" in predict(SdpaConfig(S=4096, k_chunk=384, num_cores=110, arch=ARCH_BH)).low_confidence_reasons
    assert "off_calibration_chunk" in predict(SdpaConfig(S=4096, q_chunk=96, num_cores=110, arch=ARCH_BH)).low_confidence_reasons
    assert "off_calibration_chunk" in predict(SdpaConfig(S=4096, k_chunk=64, num_cores=110, arch=ARCH_BH)).low_confidence_reasons


@pytest.mark.unit
def test_predict_rejects_nonpositive_dims():
    # Negative dims pass the modulo checks (-4096 % 128 == 0) and would hide inside a plausible
    # positive total; predict() is public, so it fails fast on them.
    with pytest.raises(ValueError):
        predict(SdpaConfig(S=-4096, num_cores=110, arch=ARCH_BH))
    with pytest.raises(ValueError):
        predict(SdpaConfig(S=4096, head_dim=-128, num_cores=110, arch=ARCH_BH))
    with pytest.raises(ValueError):
        predict(SdpaConfig(S=4096, num_heads=0, num_cores=110, arch=ARCH_BH))
    with pytest.raises(ValueError):
        predict_decode(cache_len=-1024, num_q_heads=8, num_kv_heads=1, head_dim=128)


@pytest.mark.unit
def test_config_arch_is_per_call_not_singleton():
    # sdpa_config_from_shapes must not hand every config the module-level ARCH_BH singleton:
    # a caller mutating what looks like a local arch would rewrite the global.
    q = [1, 32, 4096, 128]
    c1 = sdpa_config_from_shapes(q, q, q, {})
    c2 = sdpa_config_from_shapes(q, q, q, {})
    assert c1.arch is not c2.arch
    assert c1.arch is not ARCH_BH
    custom = ArchConfig(name="custom")
    c3 = sdpa_config_from_shapes(q, q, q, {}, arch=custom)
    assert c3.arch is custom


@pytest.mark.unit
def test_decode_without_cur_pos_routes_as_decode():
    # cur_pos_tensor=None filters to a 3-input op; the explicit entry-point tag must keep it on
    # the decode path instead of the arity heuristic reading it as prefill.
    from ttsim.ops.desc.ttsim_layout import _infer_sdpa_variant
    assert _infer_sdpa_variant([None, None, None], {'sdpa_variant': 'decode'}) == 'decode'
    # and the arity fallback still reads an untagged 3-input op as prefill
    assert _infer_sdpa_variant([None, None, None], {}) == 'prefill'


@pytest.mark.unit
@pytest.mark.parametrize("hd", [18, 72, 96, 256])
def test_non_tile_aligned_head_dim_is_padded_not_rejected(hd):
    # Vision/VAE head_dims that are not multiples of 32 are rounded up to whole tiles (matching
    # the kernel's tile padding) instead of being rejected. Cost must scale with ceil(hd/32).
    import math
    r = predict(SdpaConfig(S=4096, head_dim=hd, num_heads=16, num_cores=110, arch=ARCH_BH))
    assert r.fpu_matmul_cycles > 0
    ref = predict(SdpaConfig(S=4096, head_dim=math.ceil(hd / 32) * 32, num_heads=16,
                             num_cores=110, arch=ARCH_BH))
    assert r.fpu_matmul_cycles == ref.fpu_matmul_cycles  # padded to the same tile count


@pytest.mark.unit
def test_v_head_dim_default_is_symmetric():
    # v_head_dim=0 must reproduce the symmetric head_dim result exactly.
    sym = predict(SdpaConfig(S=4096, head_dim=128, num_cores=110, arch=ARCH_BH))
    explicit = predict(SdpaConfig(S=4096, head_dim=128, v_head_dim=128, num_cores=110, arch=ARCH_BH))
    assert sym.fpu_matmul_cycles == explicit.fpu_matmul_cycles
    assert sym.compute_latency_cycles == explicit.compute_latency_cycles


@pytest.mark.unit
def test_asymmetric_v_head_dim_scales_pv_matmul():
    # MLA-style: v_head_dim > head_dim raises the P*V matmul, so total matmul cycles grow.
    base = predict(SdpaConfig(S=4096, head_dim=192, num_cores=110, arch=ARCH_BH))
    mla = predict(SdpaConfig(S=4096, head_dim=192, v_head_dim=512, num_cores=110, arch=ARCH_BH))
    assert mla.fpu_matmul_cycles > base.fpu_matmul_cycles
    # matmul cycles scale as (dct_qk + dct_v): (6+16) vs (6+6)
    assert mla.fpu_matmul_cycles == round(base.fpu_matmul_cycles * (6 + 16) / (6 + 6))


# ---- compute floor against this campaign's perf counters -----------------------------------------

def _bh(**kw):
    return SdpaConfig(num_cores=110, arch=ARCH_BH, **kw)


_GQA = dict(num_heads=32, num_kv_heads=8)
_MLA = dict(head_dim=576, v_head_dim=512, num_kv_heads=1, fidelity="HiFi4", input_dtype="bfloat16",
            exp_approx_mode=False)
_PROD = dict(num_heads=32, num_kv_heads=8, fidelity="HiFi4", exp_approx_mode=False, fp32_dest_acc=True,
             accum_dtype="float32")
_SPARSE = dict(S=2048, head_dim=576, v_head_dim=512, q_chunk=128, k_chunk=128, num_heads=32, num_kv_heads=1,
               is_sparse=True, is_causal=False, input_dtype="bfloat16", fidelity="HiFi4", exp_approx_mode=False)
_JOINT = dict(S=4096 + 333, joint_seq=333, num_heads=24, is_causal=False, is_joint=True, q_chunk=128, k_chunk=512,
              input_dtype="bfloat16", fidelity="HiFi2", exp_approx_mode=True)

# R1f perf-counter captures on the unmodified kernel (S4096 nh32 nkv8 bfp8 HiFi2, 110 cores unless stated)
# plus the T2.3, T2.4 and R1c zoned captures. Columns: kwargs, FPU, SFPU, MATH, union tolerance or None.
COUNTERS_R1F = [
    (dict(**_GQA), 724508, 304463, 876921, 0.02),
    (dict(k_chunk=256, **_GQA), 693527, 236842, 784245, 0.02),
    (dict(k_chunk=512, **_GQA), 678931, 203031, 734024, 0.02),
    (dict(q_chunk=64, **_GQA), 714529, 308615, 1022567, 0.02),
    (dict(q_chunk=256, **_GQA), 747036, 310421, 828566, 0.02),
    (dict(q_chunk=512, **_GQA), 792092, 327252, 857383, 0.03),
    (dict(q_chunk=512, k_chunk=512, **_GQA), 738807, 217288, 767349, 0.04),
    (dict(is_causal=False, **_GQA), 1403662, 597085, 1608905, 0.02),
    (dict(is_causal=False, k_chunk=256, **_GQA), 1340509, 461843, 1458989, 0.02),
    (dict(is_causal=False, k_chunk=512, **_GQA), 1308933, 394221, 1381694, 0.02),
    (dict(is_causal=False, q_chunk=64, **_GQA), 1403662, 610192, 2012167, 0.02),
    (dict(is_causal=False, q_chunk=256, **_GQA), 1403662, 590531, 1540889, 0.02),
    (dict(is_causal=False, q_chunk=512, **_GQA), 1403662, 587255, 1505555, 0.02),
    (dict(is_causal=False, q_chunk=512, k_chunk=512, **_GQA), 1308933, 391764, 1355958, 0.04),
    # the overlap law is fit at HiFi2 with the approx exp, so these three rows carry the union error it leaves
    # at other fidelities and exp modes; every wall at these settings is stream bound and ignores the union.
    (dict(fidelity="LoFi", **_GQA), 385061, 304463, 553490, 0.09),                            # R1f LoFi anchor
    (dict(fidelity="HiFi3", **_GQA), 1063923, 304463, 1182059, 0.06),                         # R1f HiFi3 anchor
    (dict(exp_approx_mode=False, **_GQA), 724508, 271839, 859336, 0.04),                      # R1f accurate exp anchor
    (dict(q_chunk=256, k_chunk=256, num_cores=110, **_PROD), 1407164, 257743, 1664907, 0.02),  # R1f production g110
    (dict(q_chunk=256, k_chunk=256, num_cores=64, **{**_PROD, "fp32_dest_acc": False, "accum_dtype": "bfloat16"}),
     2384910, 393132, 2449748, 0.03),                                                          # R1f A10
    (dict(S=2048, kv_seq=8192, num_heads=16, is_causal=False), 701831, 301205, None, None),    # T2.3 cross (zoned)
    (dict(S=2048, q_chunk=32, num_heads=16, **_MLA), 707407, 35919, 743326, 0.05),             # T2.3 MLA
    (dict(q_chunk=256, k_chunk=256, num_cores=64, **_PROD), 2418688, 442996, 2861684, 0.02),   # T2.4 production g64
    (dict(head_dim=64, **_GQA), 390274, 304463, 516926, None),                                 # R1c head_dim 64 (union reported)
]


@pytest.mark.unit
@pytest.mark.parametrize("kw,fpu,sfpu,math_ctr,union_tol", COUNTERS_R1F)
def test_floor_tracks_the_campaign_counters(kw, fpu, sfpu, math_ctr, union_tol):
    # FPU within 0.3 percent (1.5 at HiFi4 and head_dim 64); SFPU within 3 on approx exp, 1 on the accurate
    # exp anchor, 4 on fp32 DEST and MLA, 13 on A10; union within the row tolerance (head_dim 64 not pinned).
    r = predict(SdpaConfig(**{**dict(S=4096, num_cores=110, arch=ARCH_BH), **kw}))
    fpu_tol = 0.015 if (kw.get("fidelity") == "HiFi4" or kw.get("head_dim") == 64) else 0.003
    assert abs(r.fpu_cycles / fpu - 1.0) <= fpu_tol
    if kw.get("exp_approx_mode", True):
        assert abs(r.sfpu_cycles / sfpu - 1.0) <= 0.03
    elif kw.get("fp32_dest_acc") or "v_head_dim" in kw:
        assert abs(r.sfpu_cycles / sfpu - 1.0) <= 0.04
    elif kw.get("fidelity") == "HiFi4":
        assert abs(r.sfpu_cycles / sfpu - 1.0) <= 0.13
    else:
        assert abs(r.sfpu_cycles / sfpu - 1.0) <= 0.01
    if union_tol is not None:
        assert abs(r.math_active_cycles / math_ctr - 1.0) <= union_tol


@pytest.mark.unit
def test_causal_tile_counts_use_the_truncated_diagonal_chunk():
    # The kernel narrows the diagonal k chunk to the tiles at or below it, so a causal head's exp tiles and
    # tile MACs are St (St + qct) / 2 whatever the k chunk (T2.1: FPU per tile MAC equal across k chunks).
    k128 = predict(_bh(S=4096, **_GQA))
    k512 = predict(_bh(S=4096, k_chunk=512, **_GQA))
    assert k128.fpu_matmul_cycles == k512.fpu_matmul_cycles == round(128 * 132 / 2 * 32 / 110 * 8 * 32)
    assert k128.sfpu_exp_cycles == k512.sfpu_exp_cycles == round(128 * 132 / 2 * 32 / 110 * 70.0)
    assert (k128.k_eff, k512.k_eff) == (16.5, 4.5)
    nc = predict(_bh(S=4096, is_causal=False, **_GQA))
    assert nc.fpu_matmul_cycles == round(128 * 128 * 32 / 110 * 8 * 32)
    q64 = predict(_bh(S=4096, q_chunk=64, **_GQA))
    assert q64.fpu_matmul_cycles == round(128 * 130 / 2 * 32 / 110 * 8 * 32)


@pytest.mark.unit
def test_fpu_law_has_no_q_chunk_term_and_a_per_qtile_step_term():
    # T2.1: at fixed k chunk the FPU counter is identical across q chunks (the per-q-tile-per-step overhead
    # times qct x steps is constant), and it falls with the k chunk as the term is amortised over more tiles.
    a = ArchConfig()
    fpu = {q: predict(_bh(S=4096, is_causal=False, q_chunk=q, **_GQA)).fpu_cycles for q in (64, 128, 256, 512)}
    assert max(fpu.values()) - min(fpu.values()) <= 2
    k128, k512 = predict(_bh(S=4096, is_causal=False, **_GQA)), predict(_bh(S=4096, is_causal=False, k_chunk=512, **_GQA))
    assert k512.fpu_cycles < k128.fpu_cycles and k512.fpu_matmul_cycles == k128.fpu_matmul_cycles
    per_step = (k128.fpu_overhead_cycles - k128.fpu_matmul_cycles * a.fpu_overhead_frac) / k128.inner_iters
    assert per_step == pytest.approx((a.fpu_overhead_tile_macs_per_qtile_step * 32 + a.fpu_overhead_cycles_per_qtile_step) * 4, rel=1e-3)
    # the fidelity ladder at the anchor: 16 cycles per tile MAC per level plus the fidelity independent part
    lofi, hifi2, hifi3 = (predict(_bh(S=4096, fidelity=f, **_GQA)).fpu_cycles for f in ("LoFi", "HiFi2", "HiFi3"))
    assert hifi2 - lofi == pytest.approx(hifi3 - hifi2, rel=1e-3) and lofi > 0.5 * hifi2
    # the effective overhead fraction at the anchor is the measured 0.151
    anchor = predict(_bh(S=4096, **_GQA))
    assert anchor.fpu_overhead_cycles / anchor.fpu_matmul_cycles == pytest.approx(0.151, abs=0.002)


@pytest.mark.unit
def test_sfpu_law_terms_and_the_exp_cost_from_the_a6_ablation():
    a = ArchConfig()
    assert a.exp_tile_cycles == 70.0 and a.sfpu_overhead_per_step == 47.8 and a.sfpu_overhead_per_qtile_step == 204.2
    r = predict(_bh(S=4096, **_GQA))
    steps, qct = r.inner_iters, 4
    assert r.sfpu_overhead_cycles == round(204.2 * steps * qct + 47.8 * steps)
    assert r.sfpu_recip_cycles == round(17.0 * r.q_chunks_per_core * qct) and r.sfpu_reduce_cycles == 0
    assert r.sfpu_cycles == r.sfpu_exp_cycles + r.sfpu_recip_cycles + r.sfpu_overhead_cycles + r.sfpu_extra_cycles
    acc = predict(_bh(S=4096, exp_approx_mode=False, **_GQA))
    assert acc.sfpu_exp_cycles == round(r.sfpu_exp_cycles / 70.0 * a.exp_tile_cycles_accurate) and a.exp_tile_cycles_accurate == 56.7
    # the fp32 DEST kernel issues the accurate exp from MATH at its own cost per tile
    lg = predict(SdpaConfig(S=4096, q_chunk=256, k_chunk=256, num_cores=64, arch=ARCH_BH, **_PROD))
    assert lg.sfpu_exp_cycles == round(lg.q_chunks_per_core * 8 * 68 * a.exp_tile_cycles_accurate_legacy) and a.exp_tile_cycles_accurate_legacy == 75.3
    assert a.cycles_per_tile_mac == {"LoFi": 16.0, "HiFi2": 32.0, "HiFi3": 48.0, "HiFi4": 64.0}


@pytest.mark.unit
def test_overlap_hides_a_fixed_share_of_the_fpu_time_growing_with_the_q_chunk():
    # R1f MATH counters (unmodified kernel): hidden = r (1 - 4 / qct^2) FPU capped at the SFPU, r 0.310 causal
    # / 0.347 non-causal; nothing overlaps at q64 (one q sub-block), MLA (qct 1) and the fp32 DEST kernel.
    from ttsim.perf.roofline_sdpa import _overlap_hidden, OVERLAP_R_CAUSAL, OVERLAP_R_NONCAUSAL
    assert (OVERLAP_R_CAUSAL, OVERLAP_R_NONCAUSAL) == (0.310, 0.347)
    assert _overlap_hidden(2, 1000.0, 500.0, True, False) == 0.0
    assert _overlap_hidden(4, 1000.0, 500.0, True, False) == pytest.approx(OVERLAP_R_CAUSAL * 0.75 * 1000.0)
    assert _overlap_hidden(4, 1000.0, 500.0, False, False) == pytest.approx(OVERLAP_R_NONCAUSAL * 0.75 * 1000.0)
    assert _overlap_hidden(16, 1000.0, 100.0, True, False) == 100.0                 # capped at the SFPU
    assert _overlap_hidden(4, 1000.0, 500.0, True, True) == 0.0                     # MLA
    anchor = predict(_bh(S=4096, **_GQA))
    assert anchor.overlap_frac == pytest.approx(0.551, abs=0.005)                  # measured 0.499 (union -1.8 percent)
    assert predict(_bh(S=4096, q_chunk=64, **_GQA)).overlap_frac == 0.0
    assert predict(_bh(S=4096, q_chunk=512, is_causal=False, **_GQA)).overlap_frac > anchor.overlap_frac
    mla = predict(_mla_cfg(2048))
    assert mla.overlap_frac == 0.0 and mla.math_active_cycles == mla.fpu_cycles + mla.sfpu_cycles


@pytest.mark.unit
def test_anchor_floor_pinned():
    r = predict(_bh(S=4096, **_GQA))
    assert (r.math_active_cycles, r.compute_latency_cycles) == (861427, 864327)
    assert r.init_overhead_cycles == ARCH_BH.wall_fixed_cycles == 2900


# ---- every wall of the campaign ----------------------------------------------------------------

# (label, block, config, measured cycles). Zones-off device walls, mean of invocations 1 and 2 (T2.3 / T2.4 /
# R1a / R1d), of iterations 1 and 2 (T2.1); the wall core is the most loaded core. 5 percent unless marked.
CAMPAIGN_WALLS = [
    ("causal bf16 K/V", "T2.3", _bh(S=4096, kv_input_dtype="bfloat16", **_GQA), 4_419_128),
    ("causal all bf16", "T2.3", _bh(S=4096, input_dtype="bfloat16", kv_input_dtype="bfloat16", **_GQA), 4_456_116),
    ("causal nkv32", "T2.3", _bh(S=4096, num_heads=32, num_kv_heads=32), 2_563_786),
    ("causal nkv1", "T2.3", _bh(S=4096, num_heads=32, num_kv_heads=1), 2_565_033),
    ("causal 64 cores", "T2.3", SdpaConfig(S=4096, num_cores=64, arch=ARCH_BH, **_GQA), 2_962_746),
    ("causal S1024", "T2.3", _bh(S=1024, **_GQA), 258_568),
    ("causal S16384", "T2.3", _bh(S=16384, **_GQA), 38_629_694),
    ("windowed S8192 W1024", "T2.3", _bh(S=8192, num_heads=16, sliding_window=1024), 1_411_340),
    ("chunked start4096 Sq2048", "T2.3", _bh(S=2048, kv_seq=6144, chunk_start_idx=4096, num_heads=16, paged=True, page_block_size=128), 2_026_025),
    ("MLA nh16 S2048", "T2.3", _bh(S=2048, q_chunk=32, num_heads=16, **_MLA), 9_574_958),
    ("cross 2048/8192", "T2.3", _bh(S=2048, kv_seq=8192, num_heads=16, is_causal=False), 1_649_702),
    ("masked S8192 d0.25", "T2.3", _bh(S=8192, num_heads=16, is_causal=False, has_attn_mask=True), 7_717_703),
    ("causal S2048", "R1a", _bh(S=2048, **_GQA), 754_364),
    ("causal S8192", "R1a", _bh(S=8192, **_GQA), 10_094_706),
    ("non-causal S2048", "R1a", _bh(S=2048, is_causal=False, **_GQA), 733_819),
    ("non-causal S8192", "R1a", _bh(S=8192, is_causal=False, **_GQA), 10_875_202),
    ("cross 1024/8192", "R1a", _bh(S=1024, kv_seq=8192, num_heads=16, is_causal=False), 1_107_792),
    ("cross 4096/16384", "R1a", _bh(S=4096, kv_seq=16384, num_heads=16, is_causal=False), 5_462_958),
    ("windowed S4096 W1024", "R1a", _bh(S=4096, num_heads=16, sliding_window=1024), 759_909),
    ("windowed S8192 W2048", "R1a", _bh(S=8192, num_heads=16, sliding_window=2048), 2_486_860),
    ("windowed S8192 W4096", "R1a", _bh(S=8192, num_heads=16, sliding_window=4096), 4_345_654),
    ("masked S4096 d0.25", "R1a", _bh(S=4096, num_heads=16, is_causal=False, has_attn_mask=True), 1_945_668),
    ("masked S8192 d0.5", "R1a", _bh(S=8192, num_heads=16, is_causal=False, has_attn_mask=True), 7_725_170),
    ("chunked start2048 Sq2048", "R1a", _bh(S=2048, kv_seq=4096, chunk_start_idx=2048, num_heads=16, paged=True, page_block_size=128), 1_241_686),
    ("chunked start4096 Sq1024", "R1a", _bh(S=1024, kv_seq=5120, chunk_start_idx=4096, num_heads=16, paged=True, page_block_size=128), 686_071),
    ("chunked start8192 Sq2048", "R1a", _bh(S=2048, kv_seq=10240, chunk_start_idx=8192, num_heads=16, paged=True, page_block_size=128), 3_593_322),
    ("MLA nh16 S1024", "R1a", _bh(S=1024, q_chunk=32, num_heads=16, **_MLA), 2_640_221),
    ("MLA nh16 S4096", "R1a", _bh(S=4096, q_chunk=32, num_heads=16, **_MLA), 37_137_796),
    ("MLA nh32 S2048", "R1a", _bh(S=2048, q_chunk=32, num_heads=32, **_MLA), 19_315_356),
    # R1J2 (2026-09-16): the head axis with the q chunk pinned at 32, so only the head count moves.
    # Predictions, never in any fit; the axis is linear on the device (1.00 / 2.02 / 2.99 / 3.98x).
    ("MLA nh48 S2048", "R1J2", _bh(S=2048, q_chunk=32, num_heads=48, **_MLA), 28_633_519),
    ("MLA nh64 S2048", "R1J2", _bh(S=2048, q_chunk=32, num_heads=64, **_MLA), 38_149_720),
    ("sparse T8192 TOPK1024", "R1a", _bh(kv_seq=1024, **_SPARSE), 8_603_386),
    ("sparse T8192 TOPK2048", "R1a", _bh(kv_seq=2048, **_SPARSE), 16_081_570),
    ("sparse T16384 TOPK2048", "R1a", _bh(kv_seq=2048, **_SPARSE), 16_172_522),
    ("joint N4096 L333 d128", "R1a", _bh(head_dim=128, **_JOINT), 8_578_988),
    ("joint N4096 L333 d64", "R1a", _bh(head_dim=64, **_JOINT), 4_532_419),
]
# Fit points of the regime constants (stream factors, mask, sparse and joint terms) on the R1a and T2.3 walls;
# the cross, non-causal and causal S rows and the masked S8192 d0.25 wall are predictions.
FIT_POINTS = {"causal bf16 K/V", "causal 64 cores", "windowed S8192 W1024", "chunked start4096 Sq2048", "MLA nh16 S2048",
              "windowed S4096 W1024", "windowed S8192 W2048", "windowed S8192 W4096", "masked S4096 d0.25",
              "masked S8192 d0.5", "chunked start2048 Sq2048", "chunked start4096 Sq1024", "chunked start8192 Sq2048",
              "MLA nh16 S1024", "MLA nh16 S4096", "MLA nh32 S2048", "sparse T8192 TOPK1024", "sparse T8192 TOPK2048",
              "sparse T16384 TOPK2048", "joint N4096 L333 d128", "joint N4096 L333 d64"}
_TAIL = ("the stream lane charges the 110 core rate on every step of the wall core; its steps beyond the light "
         "cores' count run on fewer readers at a faster rate the single-rate lane does not carry")
@pytest.mark.unit
def test_kernel_rev_main_prices_the_fp32_path_on_main_tip():
    """FP32M: the fp32_dest_acc_en prefill kernel is the one path that moved between the calibration tree
    and main (PR 49948 made the QK intermediate and the row sums fp32). Four main-tip walls, ns."""
    measured = {(1024, 64): 341_022, (2048, 128): 723_412, (4096, 128): 2_251_090, (8192, 128): 8_765_334}
    prod = dict(num_heads=32, num_kv_heads=8, head_dim=128, num_cores=110, is_causal=True,
                fidelity="HiFi4", fp32_dest_acc=True, accum_dtype="float32", exp_approx_mode=False,
                input_dtype="bfp8_b")
    for (S, q), ns in measured.items():
        main = predict(SdpaConfig(S=S, q_chunk=q, k_chunk=q, kernel_rev="main", **prod))
        assert main.wall_clock_cycles / 1.35 == pytest.approx(ns, rel=0.05), (S, q)
        # the calibration-tree default is the slower-kernel fit and reads low against main
        cal = predict(SdpaConfig(S=S, q_chunk=q, k_chunk=q, **prod))
        assert cal.wall_clock_cycles < main.wall_clock_cycles
    # the knob touches only the legacy path: the streaming anchor is bit-identical
    stream = dict(S=4096, q_chunk=128, k_chunk=128, num_heads=32, num_kv_heads=8, num_cores=110,
                  is_causal=True)
    assert predict(SdpaConfig(**stream)).wall_clock_cycles == \
        predict(SdpaConfig(kernel_rev="main", **stream)).wall_clock_cycles


@pytest.mark.unit
def test_small_grid_stream_rate_saturates_below_the_contended_grids():
    """GRIDR: the anchor config on smaller grids. The per-core K/V rate saturates once DRAM stops being
    contended, so the old table's clamp at the 64-core rate was pricing a 16-core grid 39 percent slow."""
    measured = {16: 9_321_734, 32: 4_665_358, 48: 3_216_878, 64: 2_959_780}
    for cores, cyc in measured.items():
        r = predict(SdpaConfig(S=4096, q_chunk=128, k_chunk=128, num_heads=32, num_kv_heads=8,
                               head_dim=128, num_cores=cores, is_causal=True))
        assert r.wall_clock_cycles == pytest.approx(cyc, rel=0.05), cores
    # the rate is flat from 16 to 48 cores and then falls away as the grid contends
    a = ArchConfig()
    flat = [a.kv_stream_rate_bpc[c] for c in (16, 32, 48)]
    assert max(flat) - min(flat) < 0.05
    assert a.kv_stream_rate_bpc[64] < min(flat) and a.kv_stream_rate_bpc[110] < a.kv_stream_rate_bpc[64]


_BEYOND_5: dict[str, str] = {
    # Empty since 2026-09-16: the MLA tail now recovers to the uncontended rate (per-regime
    # tail_rate_floor_cores) and the stream factor was refit on six MLA walls instead of four.
    # Every campaign wall is within 5 percent, worst 4.66.
}


def _wall_rows():
    rows = []
    for label, block, c, meas in CAMPAIGN_WALLS:
        if label in _BEYOND_5:
            rows.append(pytest.param(label, c, meas, marks=pytest.mark.xfail(strict=True, reason=_BEYOND_5[label])))
        else:
            rows.append((label, c, meas))
    return rows


@pytest.mark.unit
@pytest.mark.parametrize("label,cfg,meas", _wall_rows())
def test_campaign_walls_within_5_percent(label, cfg, meas):
    r = predict(cfg)
    assert abs(r.wall_clock_cycles - meas) / meas <= 0.05, f"{label}: {r.wall_clock_cycles} vs {meas}"


@pytest.mark.unit
def test_campaign_wall_table_is_complete():
    labels = [w[0] for w in CAMPAIGN_WALLS]
    assert len(labels) == len(set(labels)) == 36
    assert FIT_POINTS <= set(labels) and set(_BEYOND_5) <= set(labels)
    assert sum(1 for w in CAMPAIGN_WALLS if w[1] == "R1a") == 22 and sum(1 for w in CAMPAIGN_WALLS if w[1] == "T2.3") == 12


# Grid 1 (T2.1 zones-off device walls, cycles, S 4096 nh32 nkv8 bfp8 HiFi2, 110 cores).
GRID1_WALLS = [
    (dict(), 2_562_642), (dict(k_chunk=256), 2_596_634.5), (dict(k_chunk=512), 2_729_800.5),
    (dict(q_chunk=256), 1_826_179.5), (dict(q_chunk=512), 2_445_391), (dict(q_chunk=512, k_chunk=512), 1_834_463.5),
    (dict(q_chunk=64), 4_932_323),
    (dict(is_causal=False), 2_884_626), (dict(is_causal=False, k_chunk=256), 3_009_305),
    (dict(is_causal=False, k_chunk=512), 3_206_662.5), (dict(is_causal=False, q_chunk=256), 2_644_797.5),
    (dict(is_causal=False, q_chunk=512), 3_098_890), (dict(is_causal=False, q_chunk=512, k_chunk=512), 2_248_111),
    (dict(is_causal=False, q_chunk=64), 5_609_143.5),
]


@pytest.mark.unit
@pytest.mark.parametrize("kw,wall", GRID1_WALLS)
def test_grid1_walls_within_5_percent(kw, wall):
    # 11 of the 14 are fit points of the stream, PACK and control lanes; causal q128 k256, q128 k512 and q64
    # k128 are predictions (+1.7 / +2.4 / +3.9).
    r = predict(_bh(S=4096, **kw, **_GQA))
    assert abs(r.wall_clock_cycles - wall) / wall <= 0.05, f"{kw}: {r.wall_clock_cycles} vs {wall}"
    assert not r.low_confidence, r.low_confidence_reasons
    assert r.components["sfpu_issue"] == 0.0          # A6: the exp issue is hidden on every grid 1 point


@pytest.mark.unit
@pytest.mark.parametrize("wall", [2_562_642, 2_559_756, 2_561_014, 2_563_848])
def test_anchor_within_1_percent_of_the_t21_wall_and_the_r1d_repeats(wall):
    # The causal S4096 q128 k128 anchor was measured four times over the campaign (T2.1, then R1d at the
    # start, middle and end of the R1 block, 0.16 percent apart); the model sits within 0.12 percent of each.
    # The stream rate is pinned on this wall, so the T2.1 reading is reproduced to 9 cycles.
    r = predict(_bh(S=4096, **_GQA))
    assert abs(r.wall_clock_cycles - wall) / wall <= 0.01
    assert r.wall_clock_cycles == 2_562_633


@pytest.mark.unit
def test_wall_clock_is_wall_core_chunks_times_the_step_lanes():
    # Device wall = fixed cost + the most loaded core's steps, each the longer of the K/V stream and compute
    # lanes. At the anchor the stream binds the contended phase: kct * 691 + 34816 B / 2.6767 = 15,772 per
    # step. The wall core's steps beyond the light cores' count run in the tail phase, where only the 72
    # remainder cores still read, so they fall back on the compute lane.
    cfg = _baseline_cfg(4096)
    r = predict(cfg)
    assert r.q_chunks_wall_core == math.ceil(r.q_chunks_per_core) == 10 and r.k_eff == 16.5
    a = cfg.arch
    dram_step = 4 * (a.kv_stream_fixed_per_ktile + 8 * a.kv_stream_fixed_per_kv_tile) + 4 * 8 * 1088 / a.kv_stream_rate_bpc[110]
    assert dram_step == pytest.approx(15772, abs=5.0)
    light_steps = r.q_chunks_per_core * r.k_eff
    tail_steps = r.steps_wall_core - light_steps
    assert tail_steps > 0                                   # 512 pairs over 110 cores: 72 cores take a 5th
    tail_step = r.components["straggler"] / tail_steps
    assert tail_step < dram_step                            # fewer readers, so the tail is the cheaper step
    assert r.wall_clock_cycles == pytest.approx(
        a.wall_fixed_cycles + light_steps * dram_step + tail_steps * tail_step, rel=1e-6)
    assert r.components["reader_wait"] > 0 and r.wall_clock_cycles > r.compute_latency_cycles
    # 64 cores stream faster per core (4.121 B per cycle measured) and run 16 chunks each
    r64 = predict(SdpaConfig(S=4096, num_cores=64, arch=ARCH_BH, **_GQA))
    step64 = 4 * (a.kv_stream_fixed_per_ktile + 8 * a.kv_stream_fixed_per_kv_tile) + 34816 / a.kv_stream_rate_bpc[64]
    assert r64.wall_clock_cycles == pytest.approx(a.wall_fixed_cycles + 16 * 16.5 * step64, rel=1e-6)


@pytest.mark.unit
def test_wall_steps_when_chunks_spill_past_core_count():
    # Straggler quantization: one q chunk past a multiple of num_cores adds a whole extra chunk to the wall
    # core, stepping the wall by about a full per-chunk cost (nh=13: 104 chunks, 1 per core; nh=14: 112, 2).
    def wall(nh):
        return predict(SdpaConfig(S=1024, num_heads=nh, is_causal=False, num_cores=110, arch=ARCH_BH))
    r13, r14 = wall(13), wall(14)
    assert (r13.q_chunks_wall_core, r14.q_chunks_wall_core) == (1, 2)
    per_chunk_13 = r13.wall_clock_cycles - r13.components["init"]
    per_chunk_14 = (r14.wall_clock_cycles - r14.components["init"]) / 2
    assert per_chunk_14 == pytest.approx(per_chunk_13, rel=0.01)   # wall ~doubles minus init


WALL_TERM_NAMES = ["init", "compute_floor", "fe_issue", "reader_wait", "mask_bracket", "sfpu_issue",
                   "dest_roundtrip", "control", "straggler"]


# ---- production (legacy fp32 DEST path, T2.4 walls) ---------------------------------------------

# Llama 3.1 8B production SDPA: 64 cores, HiFi4, accurate exp, fp32 DEST, bfp8 Q/K/V, causal; q = k = 64 at
# S 1024, 256 above (bh/production_config.md). Cycles, mean of 2 invocations; the R1d repeat 11 hours later.
PRODUCTION_WALLS = [
    ("S1024 q64 g64", SdpaConfig(S=1024, q_chunk=64, k_chunk=64, num_cores=64, arch=ARCH_BH, **_PROD), 468_469, "fit"),
    ("S2048 q256 g64", SdpaConfig(S=2048, q_chunk=256, k_chunk=256, num_cores=64, arch=ARCH_BH, **_PROD), 1_072_058, "pred"),
    ("S4096 q256 g64", SdpaConfig(S=4096, q_chunk=256, k_chunk=256, num_cores=64, arch=ARCH_BH, **_PROD), 3_983_028, "fit"),
    ("S8192 q256 g64", SdpaConfig(S=8192, q_chunk=256, k_chunk=256, num_cores=64, arch=ARCH_BH, **_PROD), 15_392_503, "pred"),
    ("S4096 q256 g110", SdpaConfig(S=4096, q_chunk=256, k_chunk=256, num_cores=110, arch=ARCH_BH, **_PROD), 3_019_747, "pred"),
    ("S4096 q256 g64 repeat", SdpaConfig(S=4096, q_chunk=256, k_chunk=256, num_cores=64, arch=ARCH_BH, **_PROD), 3_982_135, "repeat"),
]


@pytest.mark.unit
@pytest.mark.parametrize("label,cfg,wall,kind", PRODUCTION_WALLS)
def test_production_walls_within_3_percent(label, cfg, wall, kind):
    # The two fit points (S1024 q64, S4096 q256) pin legacy_step_cycles and dest_roundtrip_cycles on the
    # counter-fit union; S2048, S8192 and the 110-core point are predictions (-1.4 / +0.4 / -1.0).
    r = predict(cfg)
    assert r.kernel_path == "legacy" and r.overlap_frac == 0.0
    assert abs(r.wall_clock_cycles - wall) / wall <= 0.03, f"{label}: {r.wall_clock_cycles} vs {wall}"
    assert r.components["dest_roundtrip"] > 0.0 and r.components["reader_wait"] == 0.0
    assert "legacy_path_production_family" in r.low_confidence_reasons


@pytest.mark.unit
def test_legacy_path_serialises_exp_with_the_matmuls():
    # fp32 DEST selects compute_common.hpp, where exp runs on MATH: MATH = FPU + SFPU to the cycle
    # (T2.4 counters 2861684 = 2418688 + 442996), so no overlap and no PACK issue lane.
    st = predict(SdpaConfig(S=4096, q_chunk=256, k_chunk=256, num_cores=64, arch=ARCH_BH,
                            **{**_PROD, "fp32_dest_acc": False, "accum_dtype": "bfloat16"}))
    lg = predict(SdpaConfig(S=4096, q_chunk=256, k_chunk=256, num_cores=64, arch=ARCH_BH, **_PROD))
    assert (st.kernel_path, lg.kernel_path) == ("streaming", "legacy")
    assert lg.math_active_cycles == lg.fpu_cycles + lg.sfpu_cycles and st.overlap_frac > 0.5
    assert lg.fpu_cycles == st.fpu_cycles and lg.sfpu_cycles > st.sfpu_cycles       # the MATH-issued exp costs 75.3 per tile
    assert abs(lg.math_active_cycles / 2_861_684 - 1.0) <= 0.01 and abs(lg.fpu_cycles / 2_418_688 - 1.0) <= 0.005
    assert abs(lg.sfpu_cycles / 442_996 - 1.0) <= 0.01
    assert st.components["dest_roundtrip"] == 0.0 and lg.components["sfpu_issue"] == 0.0
    # the DEST round trip term counts windows of 4 tiles per step plus a per-step part
    steps = lg.q_chunks_per_core * lg.k_eff
    a = ARCH_BH
    assert lg.components["dest_roundtrip"] / steps == pytest.approx(a.legacy_step_cycles + a.dest_roundtrip_cycles * 64 / 4)
    # MLA keeps its own path whatever the accumulator
    mla = predict(_bh(S=2048, q_chunk=32, num_heads=16, fp32_dest_acc=True, accum_dtype="float32", **_MLA))
    assert mla.kernel_path == "streaming" and "fp32_path_uncalibrated" in mla.low_confidence_reasons


# ---- head_dim term: the R1c head_dim 64 pair ---------------------------------------------------

# The two R1c S4096 q128 k128 head_dim 64 walls fit the head_dim parts of the lanes: the causal one (stream
# bound) splits the fixed stream cost per k tile, the non-causal one (PACK bound) the per-q-k-tile PACK cost.
HEAD_DIM_FIT_WALLS = [
    ("causal head_dim 64", _bh(S=4096, head_dim=64, **_GQA), 1_416_986),
    ("non-causal head_dim 64", _bh(S=4096, head_dim=64, is_causal=False, **_GQA), 2_309_290),
]


@pytest.mark.unit
@pytest.mark.parametrize("label,cfg,meas", HEAD_DIM_FIT_WALLS)
def test_head_dim_fit_pair_within_3_percent(label, cfg, meas):
    r = predict(cfg)
    assert abs(r.wall_clock_cycles - meas) / meas <= 0.03, f"{label}: {r.wall_clock_cycles} vs {meas}"
    # causal: the stream lane binds at 8,570 per step (the PACK lane is 6,875); non-causal: PACK bound at 7,207
    if cfg.is_causal:
        assert r.components["reader_wait"] > 0.15 * r.wall_clock_cycles
    else:
        assert r.components["fe_issue"] > 0 and r.components["reader_wait"] == 0


@pytest.mark.unit
def test_head_dim_term_leaves_every_head_dim_128_lane_unchanged():
    # Both head_dim parts are pinned so that dct_sum 8 gives back the head_dim 128 constants; the head_dim 64
    # step is shorter on both lanes (stream 8,570 against 15,517; causal PACK 6,875 against 8,568).
    a = ARCH_BH
    assert _kv_stream_fixed(a, 4, 8) == pytest.approx(4 * 691.0) and _kv_stream_fixed(a, 4, 4) == pytest.approx(4 * 548.4)
    t = WALL_TERMS_BH["prefill_causal"]
    assert _pack_lane(t, 4, 4, 8) == pytest.approx(4 * (750 + 4 * 348)) and _pack_lane(t, 4, 4, 4) == pytest.approx(6875.0, abs=1)
    n = WALL_TERMS_BH["prefill_noncausal"]
    assert _pack_lane(n, 4, 4, 8) == pytest.approx(8900.0) and _pack_lane(n, 4, 4, 4) == pytest.approx(7207.0, abs=1)
    assert _pack_lane(n, 4, 8, 4) == pytest.approx(10626.0, abs=1)          # k256 at head_dim 64: the R1g hold-out's PACK lane


# ---- hold-outs: validation only, never used for any fit -------------------------------------------

# Hold-outs, zones-off device walls in cycles: the R1c anchor with one axis changed each (its head_dim 64 pair
# sits in the fit set above), the sparse T 32768 point and the four R1g head_dim 64 walls.
HOLDOUT_WALLS = [
    ("causal S2048 q64 k64", _bh(S=2048, q_chunk=64, k_chunk=64, **_GQA), 1_286_729),
    ("causal all bf16", _bh(S=4096, input_dtype="bfloat16", kv_input_dtype="bfloat16", **_GQA), 4_458_372),
    ("causal batch 2 nh16", _bh(S=4096, batch=2, num_heads=16, num_kv_heads=8), 2_562_633),
    ("causal HiFi4 fp32 off", _bh(S=4096, fidelity="HiFi4", **_GQA), 2_842_621),
    ("causal LoFi", _bh(S=4096, fidelity="LoFi", **_GQA), 2_529_248),
    ("causal MHA nh16 nkv16", _bh(S=4096, num_heads=16, num_kv_heads=16), 1_401_709),
    ("causal S512 q64 k64", _bh(S=512, q_chunk=64, k_chunk=64, **_GQA), 132_298),
    ("sparse T32768 TOPK2048", _bh(kv_seq=2048, **_SPARSE), 16_155_094),
    # R1g head_dim 64 hold-outs of the head_dim term (S2048 pair, causal q256 k128, non-causal q128 k256)
    ("R1g causal S2048 head_dim 64", _bh(S=2048, head_dim=64, **_GQA), 462_832),
    ("R1g non-causal S2048 head_dim 64", _bh(S=2048, head_dim=64, is_causal=False, **_GQA), 585_244),
    ("R1g causal q256 k128 head_dim 64", _bh(S=4096, q_chunk=256, head_dim=64, **_GQA), 1_499_473),
    ("R1g non-causal q128 k256 head_dim 64", _bh(S=4096, k_chunk=256, head_dim=64, is_causal=False, **_GQA), 1_751_182),
]


@pytest.mark.unit
@pytest.mark.parametrize("label,cfg,meas", HOLDOUT_WALLS)
def test_holdout_walls_within_10_percent(label, cfg, meas):
    """Validation only: these walls were never used to fit any constant (R1c hold-out block, the sparse T 32768
    row, the R1g head_dim 64 block); they are scored at 10 percent and report the model's reach, not a pin."""
    r = predict(cfg)
    assert abs(r.wall_clock_cycles - meas) / meas <= 0.10, f"{label}: {r.wall_clock_cycles} vs {meas}"


# ---- decode: T2.8 paged sweep, R1b non-paged and MLA rows -------------------------------------

# T2.8 paged decode sweep (bh/decode_sweep.md), the op as tt_transformers issues it, DEVICE KERNEL DURATION
# medians in us, Q bf16 in DRAM, KV bfp8 unless stated; plus the R1b paged 110-core position 8192 prediction.
_T28 = dict(num_q_heads=32, num_kv_heads=8, head_dim=128, k_chunk=0, fidelity="HiFi2", input_dtype="bfloat16",
            kv_input_dtype="bfp8_b", accum_dtype="float32", paged=True, page_block_size=32, max_cores_per_head_batch=16)
DECODE_SWEEP_T28 = [
    # grid cores, batch, kv dtype, position (cur_pos), measured us
    (64, 32, "bfp8_b", 128, 73.8), (64, 32, "bfp8_b", 512, 162.8), (64, 32, "bfp8_b", 1024, 284.0),
    (64, 32, "bfp8_b", 2048, 523.1), (64, 32, "bfp8_b", 4096, 1005.5), (64, 32, "bfp8_b", 8192, 1964.8),
    (64, 8, "bfp8_b", 1024, 78.8), (64, 16, "bfp8_b", 1024, 148.8),
    (64, 32, "bfloat16", 1024, 505.1),
    (110, 32, "bfp8_b", 1024, 257.2), (110, 32, "bfp8_b", 4096, 901.9),
    (110, 32, "bfp8_b", 8192, 1759.3),
]


def _t28(grid, batch, kv_dtype, pos, **kw):
    return predict_decode(cache_len=16384, cur_pos=pos, batch=batch, num_cores=grid,
                          **{**_T28, "kv_input_dtype": kv_dtype, **kw})


@pytest.mark.unit
@pytest.mark.parametrize("grid,batch,kv_dtype,pos,meas_us", DECODE_SWEEP_T28)
def test_decode_sweep_points_within_3_percent(grid, batch, kv_dtype, pos, meas_us):
    # Fit points of the paged decode law (fixed 13000 + 930 per sequential head group, 298.9 / 330.5 GB/s on
    # the 64 / 110 grids); the R1b 110-core position 8192 point is a prediction (+0.4), bf16 K/V sits at +2.8.
    r = _t28(grid, batch, kv_dtype, pos)
    us = r.wall_clock_cycles / CLK
    assert abs(us - meas_us) / meas_us <= 0.03, f"grid {grid} b{batch} {kv_dtype} pos {pos}: {us:.1f} vs {meas_us}"
    assert r.config_echo["k_chunk"] == 128 and r.config_echo["active_cores"] == 64
    assert not {"decode_grid_uncalibrated", "decode_nonpaged_grid_inferred"} & set(r.low_confidence_reasons)


# R1b non-paged decode on the 110 grid (decode_latency_sweep form: Q bf16 in DRAM, k_chunk 128, HiFi2,
# cur_pos = cache - 1), DEVICE KERNEL DURATION medians of invocations 1 and 2 (us).
DECODE_NONPAGED_R1B = [(32, "bfp8_b", 1024, 229.1), (32, "bfp8_b", 4096, 877.2), (8, "bfloat16", 1024, 105.6), (8, "bfloat16", 4096, 397.3)]


@pytest.mark.unit
@pytest.mark.parametrize("batch,kv_dtype,cache,meas_us", DECODE_NONPAGED_R1B)
def test_decode_nonpaged_r1b_points_within_5_percent(batch, kv_dtype, cache, meas_us):
    # Fit points of the non-paged rate (342.3 GB/s) under the paged fixed law; the bfp8 rows sit low and the
    # bf16 rows high by 3 to 4 percent (bf16 tiles stream faster per byte, as on the paged sweep).
    r = predict_decode(cache_len=cache, cur_pos=cache - 1, num_q_heads=32, num_kv_heads=8, head_dim=128, batch=batch,
                       k_chunk=128, fidelity="HiFi2", input_dtype="bfloat16", kv_input_dtype=kv_dtype, num_cores=110)
    us = r.wall_clock_cycles / CLK
    assert abs(us - meas_us) / meas_us <= 0.05, f"b{batch} {kv_dtype} cache {cache}: {us:.1f} vs {meas_us}"
    assert r.config_echo["kv_stream_gbps"] == 342.3 and r.config_echo["active_cores"] == 64
    assert r.config_echo["heads_per_core"] == (4 if batch == 32 else 1)
    assert not {"decode_grid_uncalibrated", "decode_nonpaged_grid_inferred"} & set(r.low_confidence_reasons)


# R1b MLA decode, DEVICE KERNEL DURATION medians in us: the non-paged model form (b8 nh16 nkv1 kv_lora 512,
# bf16 K and V, HiFi4) and the paged test geometry (nh128, Q height sharded on 64 cores, bfp8 K reused as V).
_MLAD_NP = dict(num_q_heads=16, num_kv_heads=1, head_dim=576, v_head_dim=512, batch=8, k_chunk=128, fidelity="HiFi4",
                input_dtype="bfloat16", kv_input_dtype="bfloat16", num_cores=110, mla_v_read=True)
_MLAD_P = dict(num_q_heads=128, num_kv_heads=1, head_dim=576, v_head_dim=512, k_chunk=128, fidelity="HiFi4",
               input_dtype="bfloat16", kv_input_dtype="bfp8_b", num_cores=110, paged=True, page_block_size=64,
               max_cores_per_head_batch=16, q_in_dram=False, q_shard_cores=64, cache_len=16384)
MLA_DECODE_R1B = [
    ("non-paged b8 nh16 cache1024", dict(_MLAD_NP, cache_len=1024, cur_pos=1023), 82.1),
    ("non-paged b8 nh16 cache4096", dict(_MLAD_NP, cache_len=4096, cur_pos=4095), 243.5),
    ("non-paged b8 nh16 cache8192", dict(_MLAD_NP, cache_len=8192, cur_pos=8191), 447.2),
    ("paged b4 nh128 pos1024", dict(_MLAD_P, batch=4, cur_pos=1024), 91.8),
    ("paged b4 nh128 pos4096", dict(_MLAD_P, batch=4, cur_pos=4096), 181.4),
    ("paged b4 nh128 pos8192", dict(_MLAD_P, batch=4, cur_pos=8192), 295.3),
    ("paged b8 nh128 pos1024", dict(_MLAD_P, batch=8, cur_pos=1024), 106.8),
    ("paged b8 nh128 pos4096", dict(_MLAD_P, batch=8, cur_pos=4096), 287.9),
    ("paged b8 nh128 pos8192", dict(_MLAD_P, batch=8, cur_pos=8192), 531.3),
]

# R1h (2026-09-16, p100a): the query-slice sweep. Query heads 32 / 64 / 128 on the same 64-core Q shard give
# 1 / 2 / 4 slices at a fixed cache and position, at batch 4 and 8, so the slice count moves while everything
# else holds. The walls step by 15 to 18 us per doubling of slices and per doubling of batch alike, i.e. with
# the KV bytes (which carry the per-slice re-read) and not with a per-slice fixed cost: subtracting the bytes
# at 342 GB/s leaves 41 to 59 us of fixed cost, four of the six rows inside 58 to 59.
MLA_DECODE_R1H = [
    ("R1h paged b4 nh32 slices1", dict(_MLAD_P, batch=4, num_q_heads=32, cache_len=4096, cur_pos=1024), 59.5),
    ("R1h paged b4 nh64 slices2", dict(_MLAD_P, batch=4, num_q_heads=64, cache_len=4096, cur_pos=1024), 74.4),
    ("R1h paged b4 nh128 slices4", dict(_MLAD_P, batch=4, num_q_heads=128, cache_len=4096, cur_pos=1024), 92.4),
    ("R1h paged b8 nh32 slices1", dict(_MLAD_P, batch=8, num_q_heads=32, cache_len=4096, cur_pos=1024), 75.2),
    ("R1h paged b8 nh64 slices2", dict(_MLAD_P, batch=8, num_q_heads=64, cache_len=4096, cur_pos=1024), 92.0),
    ("R1h paged b8 nh128 slices4", dict(_MLAD_P, batch=8, num_q_heads=128, cache_len=4096, cur_pos=1024), 107.2),
]
_MLAD_R1H_BEYOND_5: dict[str, str] = {}  # the cores-per-group fixed cost (R1i) took all six


# R1i (2026-09-16, p100a): the cores-per-group axis. Same bytes, same slices, only max_cores_per_head_batch
# moves, so the only thing that changes is how many cores the factory leaves on one head group. The wall is
# not monotonic in that count: at batch 4 nh128 it reads 112.1 / 79.6 / 71.2 / 80.4 / 91.8 us at 1 / 2 / 3 /
# 4 / 6 cores per group, so the factory's default (6 here) costs 29 percent over the best split.
MLA_DECODE_R1I = [
    ("R1i b4 nh128 cpg1", dict(_MLAD_P, batch=4, cache_len=4096, cur_pos=1024, max_cores_per_head_batch=1), 112.1),
    ("R1i b4 nh128 cpg2", dict(_MLAD_P, batch=4, cache_len=4096, cur_pos=1024, max_cores_per_head_batch=2), 79.6),
    ("R1i b4 nh128 cpg3", dict(_MLAD_P, batch=4, cache_len=4096, cur_pos=1024, max_cores_per_head_batch=3), 71.2),
    ("R1i b4 nh128 cpg4", dict(_MLAD_P, batch=4, cache_len=4096, cur_pos=1024, max_cores_per_head_batch=4), 80.4),
    ("R1i b4 nh32 cpg4", dict(_MLAD_P, batch=4, num_q_heads=32, cache_len=4096, cur_pos=1024,
                             max_cores_per_head_batch=4), 58.5),
    ("R1i b4 nh32 cpg8", dict(_MLAD_P, batch=4, num_q_heads=32, cache_len=4096, cur_pos=1024,
                             max_cores_per_head_batch=8), 61.9),
]


@pytest.mark.unit
@pytest.mark.parametrize("label,kw,meas_us", MLA_DECODE_R1I)
def test_mla_decode_cores_per_group_sweep_within_5_percent(label, kw, meas_us):
    r = predict_decode(**kw)
    got = r.wall_clock_cycles / 1350.0
    assert abs(got - meas_us) / meas_us <= 0.05, f"{label}: {got:.1f} us vs {meas_us} us"


@pytest.mark.unit
def test_mla_decode_fixed_cost_is_not_monotonic_in_the_cores_per_group():
    # The measured shape: one core per group pays the most, three the least, and six costs more than three
    # again. A model that only knew the bytes would price all five the same (R1i).
    walls = {}
    for cap in (1, 2, 3, 4, 6):
        r = predict_decode(**dict(_MLAD_P, batch=4, cache_len=4096, cur_pos=1024, max_cores_per_head_batch=cap))
        assert r.config_echo["cores_per_head"] == cap and r.config_echo["kv_bytes"] == 16 * 9 * 4 * 18 * 1088
        walls[cap] = r.wall_clock_cycles
    assert walls[1] > walls[2] > walls[3] < walls[4] < walls[6]
    assert walls[6] / walls[3] > 1.2
    # a split the sweep did not measure is interpolated and says so
    r5 = predict_decode(**dict(_MLAD_P, batch=4, cache_len=4096, cur_pos=1024, max_cores_per_head_batch=5))
    assert "mla_decode_cores_per_group_interpolated" in r5.low_confidence_reasons
    assert walls[4] < r5.wall_clock_cycles < walls[6]


@pytest.mark.unit
@pytest.mark.parametrize("label,kw,meas_us", [
    pytest.param(l, k, m, marks=pytest.mark.xfail(strict=True, reason=_MLAD_R1H_BEYOND_5[l]))
    if l in _MLAD_R1H_BEYOND_5 else (l, k, m) for l, k, m in MLA_DECODE_R1H])
def test_mla_decode_slice_sweep_within_5_percent(label, kw, meas_us):
    r = predict_decode(**kw)
    got = r.wall_clock_cycles / 1350.0
    assert abs(got - meas_us) / meas_us <= 0.05, f"{label}: {got:.1f} us vs {meas_us} us"


@pytest.mark.unit
def test_mla_decode_bytes_scale_with_slices_and_the_fixed_cost_does_not():
    # The reader re-reads the latent cache once per query-head slice, so the KV bytes scale with the slice
    # count while the fixed cost is the paged path's and is charged once (R1h).
    rows = [predict_decode(**dict(_MLAD_P, batch=4, num_q_heads=nh, cache_len=4096, cur_pos=1024))
            for nh in (32, 64, 128)]
    slices = [r.config_echo["q_head_slices"] for r in rows]
    assert slices == [1, 2, 4]
    kv = [r.config_echo["kv_bytes"] for r in rows]
    assert kv[1] == 2 * kv[0] and kv[2] == 4 * kv[0]
    # one fixed cost, not one per slice; nh32 leaves 16 cores on a group, nh64 13 and nh128 6, and the
    # paged fixed cost follows that split alone (R1i)
    table = ARCH_BH.decode_fixed_cycles_mla_paged_by_cores_per_group
    assert [r.components["init"] for r in rows] == [table[r.config_echo["cores_per_head"]] for r in rows]


_MLAD_BEYOND_5: dict[str, str] = {}      # the cores-per-group fixed cost (R1i) took the last two


@pytest.mark.unit
@pytest.mark.parametrize("label,kw,meas_us", [
    pytest.param(l, k, m, marks=pytest.mark.xfail(strict=True, reason=_MLAD_BEYOND_5[l])) if l in _MLAD_BEYOND_5 else (l, k, m)
    for l, k, m in MLA_DECODE_R1B])
def test_mla_decode_r1b_points_within_5_percent(label, kw, meas_us):
    # Fit points of the MLA decode law: 25800 cycles non-paged, the paged table by cores per group, 342.1 GB/s.
    r = predict_decode(**kw)
    us = r.wall_clock_cycles / CLK
    assert r.is_mla and r.is_memory_bound and r.config_echo["kv_stream_gbps"] == 342.1
    assert "mla_decode_fit_forms" not in r.low_confidence_reasons
    assert abs(us - meas_us) / meas_us <= 0.05, f"{label}: {us:.1f} vs {meas_us}"


@pytest.mark.unit
def test_mla_decode_geometry_follows_the_factory():
    from ttsim.perf.roofline_sdpa import _mla_q_head_slices
    # A height sharded query of 4 x 128 heads on 64 cores has 32-row shards: 4 slices of 32 heads, each a
    # virtual user with its own K stream; batch 8 on the same shards gives the same 4 slices.
    assert _mla_q_head_slices(4, 128, 64) == 4 and _mla_q_head_slices(8, 128, 64) == 4
    assert _mla_q_head_slices(8, 16, 0) == 1 and _mla_q_head_slices(1, 128, 64) == 4 and _mla_q_head_slices(32, 128, 64) == 2
    b4 = predict_decode(**dict(_MLAD_P, batch=4, cur_pos=8192))
    b8 = predict_decode(**dict(_MLAD_P, batch=8, cur_pos=8192))
    assert (b4.config_echo["q_head_slices"], b4.config_echo["cores_per_head"], b4.active_cores) == (4, 6, 96)
    assert (b8.config_echo["q_head_slices"], b8.config_echo["cores_per_head"], b8.active_cores) == (4, 3, 96)
    # K only (reused as V), 65 whole chunks of 128 rows per slice, no Q traffic from the L1 shards
    assert b4.config_echo["kv_bytes"] == 16 * 65 * 4 * 18 * 1088 and b4.config_echo["q_bytes"] == 0
    # The fixed cost belongs to the paged path and to the cores the factory leaves on one head group, not
    # to the slicing: b4 keeps 6 cores per group and b8 3, and each carries its own value once (R1h, R1i).
    table = ARCH_BH.decode_fixed_cycles_mla_paged_by_cores_per_group
    assert (b4.components["init"], b8.components["init"]) == (table[6], table[3])
    # the non-paged form streams K and V (34 tiles per row) once per user with the query from DRAM
    np_ = predict_decode(**dict(_MLAD_NP, cache_len=8192, cur_pos=8191))
    assert np_.config_echo["kv_bytes"] == 8 * 64 * 4 * 34 * 2048 and np_.config_echo["q_bytes"] == 104 * 18 * 2048
    assert (np_.config_echo["q_head_slices"], np_.config_echo["cores_per_head"], np_.active_cores) == (1, 13, 104)
    # forms outside the two measured ones are flagged
    assert "mla_decode_fit_forms" in predict_decode(**dict(_MLAD_NP, cache_len=8192, cur_pos=8191, mla_v_read=False)).low_confidence_reasons
    assert "mla_decode_fit_forms" in predict_decode(**dict(_MLAD_P, batch=4, cur_pos=8192, num_cores=64)).low_confidence_reasons


@pytest.mark.unit
def test_decode_fixed_term_follows_the_sequential_head_groups():
    # 64 cores run batch x 8 KV heads as 1 / 2 / 4 groups per core at b8 / b16 / b32; the fixed part of
    # the wall (init) grows by one per-group cost each time, the stream part by the bytes.
    a = ARCH_BH
    for batch, hpc in ((8, 1), (16, 2), (32, 4)):
        r = _t28(64, batch, "bfp8_b", 1024)
        assert r.config_echo["heads_per_core"] == hpc
        assert r.components["init"] == a.decode_fixed_overhead_cycles + a.decode_fixed_per_head_group_cycles * hpc
    b8, b32 = _t28(64, 8, "bfp8_b", 1024), _t28(64, 32, "bfp8_b", 1024)
    assert b32.config_echo["kv_bytes"] == 4 * b8.config_echo["kv_bytes"]
    # the T2.8 batch axis: 9.9 / 9.3 / 8.9 us per user measured, near linear with a shared fixed part
    per_user = [_t28(64, b, "bfp8_b", 1024).wall_clock_cycles / 1350.0 / b for b in (8, 16, 32)]
    assert per_user[0] > per_user[1] > per_user[2] > 8.5


@pytest.mark.unit
def test_whole_pipeline_correlation_vs_measured():
    # End-to-end: the projected device latency through the real Polaris pipeline tracks the measured
    # device wall (R1d anchor 1897.0 us, R1b decode 877.2 us) within 5 percent.
    us = _polaris_device_us(sdpa_perf_stats(_bh(S=4096, **_GQA)))
    assert abs(us - 1897.0) / 1897.0 <= 0.05, f"projected {us:.0f}us vs measured 1897.0us"
    dec = predict_decode(cache_len=4096, cur_pos=4095, num_q_heads=32, num_kv_heads=8, head_dim=128, batch=32, k_chunk=128,
                         fidelity="HiFi2", input_dtype="bfloat16", kv_input_dtype="bfp8_b", num_cores=110)
    assert abs(_polaris_device_us(dec.to_polaris_op_perf_stats()) - 877.2) / 877.2 <= 0.05


# ---- regime mechanics -------------------------------------------------------------------------

@pytest.mark.unit
def test_masked_prefill_visits_every_k_chunk_and_streams_the_mask_tiles():
    # R1a and T2.3: the provided-mask path visits every k chunk (640 steps for 10 chunks at S8192) and pays a
    # density independent cost per mask tile; S4096 d0.25 and S8192 d0.5 fit it, S8192 d0.25 is a prediction.
    mask = predict(_bh(S=8192, num_heads=16, is_causal=False, has_attn_mask=True))
    nc = predict(_bh(S=8192, num_heads=16, is_causal=False))
    assert mask.k_eff == nc.k_eff == 64.0 and mask.steps_wall_core == 640.0
    assert mask.fpu_matmul_cycles == nc.fpu_matmul_cycles and mask.regime == "masked" and mask.wall_regime == "masked"
    assert mask.components["mask_bracket"] / (mask.q_chunks_per_core * mask.k_eff) == pytest.approx(16 * WALL_TERMS_BH["masked"].mask_per_tile)
    assert mask.wall_clock_cycles > nc.wall_clock_cycles and WALL_ONLY_FIT in mask.low_confidence_reasons
    # a causal op with a mask keeps the causal visit count and the lightweight bracket
    cm = predict(_bh(S=4096, is_causal=True, has_attn_mask=True, **_GQA))
    assert cm.k_eff == 16.5 and cm.regime == "masked"
    sink = predict(SdpaConfig(S=4096, is_causal=False, attention_sink=True, num_cores=110, arch=ARCH_BH))
    assert sink.sfpu_extra_cycles > 0 and sink.low_confidence


@pytest.mark.unit
def test_sparse_wall_is_per_token_gather_independent_of_t():
    # R1a: the sparse kernel splits the query tokens over the cores; per token the wall pays a serial part
    # plus the TOPK latent key row gather at 2.979 B per cycle per core, so T does not enter (within 0.6).
    t = WALL_TERMS_BH["sparse"]
    r = predict(_bh(kv_seq=2048, **_SPARSE))
    assert r.regime == r.wall_regime == "sparse" and WALL_ONLY_FIT in r.low_confidence_reasons
    assert r.q_chunks_wall_core == 19 and r.active_cores == 110 and r.k_eff == 16.0
    assert r.components["fe_issue"] == pytest.approx(2048 / 110 * t.fe_per_token)
    gather_token = 2048 * 18 * 2048 / 32 / t.gather_rate_bpc
    floor_token = r.components["compute_floor"] / (2048 / 110)
    assert r.components["reader_wait"] == pytest.approx(2048 / 110 * (gather_token - floor_token))
    assert r.fpu_matmul_cycles == round(2048 / 110 * 1 * 64 * 34 * 64)         # one q tile row of 32 heads
    assert abs(r.fpu_cycles * 19 / (2048 / 110) / 3_034_224 - 1.0) < 0.08     # R1a FPU counter, 19-token core
    half = predict(_bh(kv_seq=1024, **_SPARSE))
    assert 0.5 < half.wall_clock_cycles / r.wall_clock_cycles < 0.55
    assert r.dram_in_bytes > 2048 * 2048 * 18 * 64 and r.components["sfpu_issue"] == 0.0
    # off the fit family: heads, dtype or chunk
    assert "sparse_fit_family" in predict(_bh(kv_seq=2048, **{**_SPARSE, "num_heads": 16})).low_confidence_reasons
    assert "sparse_fit_family" in predict(_bh(kv_seq=2048, **{**_SPARSE, "k_chunk": 256})).low_confidence_reasons
    assert "off_calibration_chunk" not in r.low_confidence_reasons


@pytest.mark.unit
def test_joint_wall_follows_its_factory_split_and_a_per_tile_mac_law():
    # R1a: the joint kernel pads the main and joint streams to whole chunks separately, runs 24 heads on 4
    # cores each (96 active, 81 steps), exp on MATH; the d128 and d64 walls fit a serial cost per tile MAC.
    from ttsim.perf.roofline_sdpa import _joint_split
    assert _joint_split(110, 1, 24, 35) == (9, 8.75, 96) and _joint_split(110, 1, 128, 35) is None
    r = predict(_bh(head_dim=128, **_JOINT))
    assert (r.regime, r.wall_regime, r.active_cores, r.q_chunks_wall_core, r.k_eff, r.steps_wall_core) == ("joint", "joint", 96, 9, 9.0, 81.0)
    assert r.overlap_frac == 0.0 and r.math_active_cycles == r.fpu_cycles + r.sfpu_cycles
    assert abs(r.fpu_cycles / 1_428_280 - 1.0) < 0.06        # R1a FPU counter, mean core (8.75 chunks)
    t = WALL_TERMS_BH["joint"]
    assert t.fe_flat_per_kchunk == 0.0 and t.pack_per_qtile == 0.0 and t.dram_law == "none"
    per_step_fe = r.components["fe_issue"] / (r.q_chunks_per_core * r.k_eff)
    assert per_step_fe == pytest.approx(t.fe_per_tile_mac * 4 * 16 * 8)
    d64 = predict(_bh(head_dim=64, **_JOINT))
    assert 0.5 < d64.wall_clock_cycles / r.wall_clock_cycles < 0.55 and WALL_ONLY_FIT in d64.low_confidence_reasons
    # more heads than cores fall back to the flat split and say so
    big = predict(_bh(head_dim=128, **{**_JOINT, "num_heads": 128}))
    assert "joint_split_off_factory" in big.low_confidence_reasons


@pytest.mark.unit
def test_wall_only_regimes_are_flagged():
    # Dense mask, sparse and joint have wall-only fits (no zone decomposition of their own terms).
    for kw in (dict(S=8192, num_heads=16, is_causal=False, has_attn_mask=True),
               dict(kv_seq=1024, **_SPARSE),
               dict(head_dim=64, **_JOINT)):
        r = predict(_bh(**kw))
        assert WALL_ONLY_FIT in r.low_confidence_reasons and r.components["fe_issue"] > 0.0
    for name in ("prefill_causal", "prefill_noncausal", "cross"):
        assert WALL_TERMS_BH[name].low_confidence == "" and WALL_TERMS_BH[name].fe_flat_per_kchunk == 0.0
        assert WALL_TERMS_BH[name].stream_scale == 1.0 and WALL_TERMS_BH[name].mask_per_tile == 0.0


@pytest.mark.unit
def test_stream_bound_regimes_ride_the_kv_stream_lane_with_a_fitted_factor():
    # T2.3 zones: MLA, windowed and chunked wait on the K/V stream, so they ride the causal stream lane at
    # their own bytes per step and a fitted factor (chunked 0.80, windowed 0.98, MLA 1.02 on R1a and T2.3).
    for kw in (dict(S=8192, num_heads=16, sliding_window=1024), dict(S=2048, kv_seq=6144, chunk_start_idx=4096, num_heads=16),
               dict(S=2048, q_chunk=32, num_heads=16, **_MLA)):
        r = predict(_bh(**kw))
        t = WALL_TERMS_BH[r.wall_regime]
        assert t.dram_law == "all_cores" and t.fe_flat_per_kchunk == 0.0
        assert STREAM_RESIDUAL_FIT in r.low_confidence_reasons
        assert r.components["reader_wait"] > 0.25 * r.wall_clock_cycles   # chunked 30, windowed 53, MLA 85 percent
    a = ARCH_BH
    lane = 4 * (a.kv_stream_fixed_per_ktile + (18 + 16) * a.kv_stream_fixed_per_kv_tile) + 4 * (18 + 16) * 2048 / a.kv_stream_rate_bpc[110]
    assert lane == pytest.approx(110_533, abs=5)
    # mla refit to 1.0400 on six walls (nh 16/32/48/64) once its tail was allowed to recover; chunked
    # unchanged. mla is the one regime whose tail reaches the uncontended rate.
    assert WALL_TERMS_BH["mla"].stream_scale == 1.0400 and WALL_TERMS_BH["chunked"].stream_scale == 0.8028
    assert WALL_TERMS_BH["mla"].tail_rate_floor_cores == 0
    assert WALL_TERMS_BH["prefill_causal"].tail_rate_floor_cores == TAIL_RATE_MIN_CORES == 64
    assert WALL_TERMS_BH["windowed"].stream_scale == 0.9753
    # Chunked keeps one rate over both phases: its factor absorbs the tail, the others price it explicitly.
    assert WALL_TERMS_BH["chunked"].tail_phase is False
    assert all(WALL_TERMS_BH[k].tail_phase for k in ("prefill_causal", "windowed", "mla"))
    # the 64-core chunked point keeps the factor on the 64-core lane
    r64 = predict(_bh(S=1024, kv_seq=5120, chunk_start_idx=4096, num_heads=16))
    assert r64.active_cores == 64 and "off_calibration_cores" not in r64.low_confidence_reasons


@pytest.mark.unit
def test_cross_and_noncausal_walls_sit_on_the_pack_lane_without_refit():
    # R1a cross 1024/8192 and 4096/16384 and non-causal S2048 / S8192 land within 5 percent of the T2.1 PACK
    # and injector lanes (+3.1 / +4.3 / +0.6 / +2.8), so no constant was refit on them.
    for kw, meas in ((dict(S=1024, kv_seq=8192, num_heads=16, is_causal=False), 1_107_792),
                     (dict(S=4096, kv_seq=16384, num_heads=16, is_causal=False), 5_462_958),
                     (dict(S=2048, is_causal=False, **_GQA), 733_819), (dict(S=8192, is_causal=False, **_GQA), 10_875_202)):
        r = predict(_bh(**kw))
        assert 0.0 <= (r.wall_clock_cycles - meas) / meas <= 0.05
    cross = predict(_bh(S=1024, kv_seq=8192, num_heads=16, is_causal=False))
    assert cross.wall_regime == "cross" and cross.steps_wall_core == 128.0 and cross.components["reader_wait"] == 0.0


@pytest.mark.unit
def test_windowed_wall_core_runs_the_longest_band():
    # Band pairs differ in length, so the wall core is the range with the most visits: 90 steps at S8192 W1024
    # (T2.3; the mean visit count gives 84), 315 / 170 / 54 on the R1a walls, all equal to the zoned STEP_N.
    for kw, steps in ((dict(S=8192, num_heads=16, sliding_window=1024), 90.0), (dict(S=8192, num_heads=16, sliding_window=4096), 315.0),
                      (dict(S=8192, num_heads=16, sliding_window=2048), 170.0), (dict(S=4096, num_heads=16, sliding_window=1024), 54.0)):
        r = predict(_bh(**kw))
        assert r.steps_wall_core == steps and r.components["straggler"] > 0.0
    r = predict(_bh(S=8192, num_heads=16, sliding_window=1024))
    assert r.q_chunks_wall_core == 10 and r.k_eff == pytest.approx(8.4375)
    # every other regime keeps chunks x visits; the zoned wall cores of the R1a regimes agree
    for kw, steps in ((dict(S=4096, **_GQA), 165.0), (dict(S=8192, **_GQA), 650.0), (dict(S=2048, **_GQA), 51.0),
                      (dict(S=2048, kv_seq=4096, chunk_start_idx=2048, num_heads=16), 98.0),
                      (dict(S=1024, kv_seq=5120, chunk_start_idx=4096, num_heads=16), 73.0),
                      (dict(S=2048, kv_seq=10240, chunk_start_idx=8192, num_heads=16), 290.0),
                      (dict(S=1024, q_chunk=32, num_heads=16, **_MLA), 27.0), (dict(S=4096, q_chunk=32, num_heads=16, **_MLA), 330.0),
                      (dict(S=2048, q_chunk=32, num_heads=32, **_MLA), 170.0),
                      (dict(S=1024, kv_seq=8192, num_heads=16, is_causal=False), 128.0),
                      (dict(S=4096, kv_seq=16384, num_heads=16, is_causal=False), 640.0),
                      (dict(S=8192, num_heads=16, is_causal=False, has_attn_mask=True), 640.0),
                      (dict(S=4096, num_heads=16, is_causal=False, has_attn_mask=True), 160.0)):
        r = predict(_bh(**kw))
        assert r.steps_wall_core == r.q_chunks_wall_core * r.k_eff == steps, kw


@pytest.mark.unit
def test_causal_visits_follow_the_kernel_diagonal():
    # q-chunk i runs ceil((i + 1) q_chunk / k_chunk) k-chunks: (K + 1) / 2 at q <= k, more above it
    # (the wall core at S 4096 q256 k128 does 6 x 17 = 102 steps, T2.1 STEP_N, not 6 x 16.5).
    from ttsim.perf.roofline_sdpa import _causal_k_eff
    assert _causal_k_eff(4096, 128, 128) == 16.5 and _causal_k_eff(4096, 64, 128) == 16.5
    assert _causal_k_eff(4096, 128, 512) == 4.5
    assert _causal_k_eff(4096, 256, 128) == 17.0 and _causal_k_eff(4096, 512, 128) == 18.0
    assert predict(_bh(S=4096, q_chunk=256, **_GQA)).k_eff == 17.0
    assert predict(_bh(S=4096, q_chunk=256, is_causal=False, **_GQA)).k_eff == 32.0


@pytest.mark.unit
def test_reader_wait_is_the_exposed_stream_beyond_the_compute_lane():
    # At the anchor the DRAM stream binds: a heavier compute lane (HiFi4) eats into reader_wait and
    # leaves the wall untouched until the compute lane passes the stream.
    base = predict(_bh(S=4096, **_GQA))
    hifi4 = predict(_bh(S=4096, fidelity="HiFi4", **_GQA))
    assert hifi4.components["compute_floor"] > base.components["compute_floor"]
    assert hifi4.components["reader_wait"] < base.components["reader_wait"]
    assert hifi4.wall_clock_cycles == base.wall_clock_cycles
    # Doubling k_chunk doubles the stream bytes per step and halves the steps: the wall is nearly flat.
    k256 = predict(_bh(S=4096, k_chunk=256, **_GQA))
    assert abs(k256.wall_clock_cycles / base.wall_clock_cycles - 1.0) < 0.05
    per_step = lambda r, n: r.components[n] / (r.q_chunks_per_core * r.k_eff)
    assert per_step(k256, "reader_wait") + per_step(k256, "fe_issue") > per_step(base, "reader_wait") + per_step(base, "fe_issue")
    # q256 at k128 is PACK-thread bound: no stream wait, the front end carries the PACK lane excess.
    q256 = predict(_bh(S=4096, q_chunk=256, **_GQA))
    assert q256.components["reader_wait"] == 0.0 and q256.components["fe_issue"] > 0.0


@pytest.mark.unit
def test_kv_stream_rate_interpolates_between_measured_grids_and_flags_others():
    from ttsim.perf.roofline_sdpa import _kv_stream_rate
    a = ArchConfig()
    assert _kv_stream_rate(a, 110) == a.kv_stream_rate_bpc[110] and _kv_stream_rate(a, 64) == a.kv_stream_rate_bpc[64]
    assert a.kv_stream_rate_bpc[64] > _kv_stream_rate(a, 80) > a.kv_stream_rate_bpc[110]
    # GRIDR measured 48, 32 and 16, so a standalone small grid reads its own rate; only the tail keeps the
    # 64-core floor, and above the widest grid the table still clamps.
    assert _kv_stream_rate(a, 32) == a.kv_stream_rate_bpc[32] and _kv_stream_rate(a, 140) == a.kv_stream_rate_bpc[110]
    assert _kv_stream_rate(a, 32, tail=True) == a.kv_stream_rate_bpc[64]
    for cores in (80, 32, 140):
        r = predict(SdpaConfig(S=4096, num_cores=cores, arch=ARCH_BH, **_GQA))
        assert "off_calibration_cores" in r.low_confidence_reasons
    # the injector law (non-causal chains) has one measured grid
    assert "off_calibration_cores" in predict(SdpaConfig(S=4096, num_cores=64, is_causal=False, arch=ARCH_BH, **_GQA)).low_confidence_reasons
    assert "off_calibration_cores" not in predict(SdpaConfig(S=4096, num_cores=64, arch=ARCH_BH, **_GQA)).low_confidence_reasons


@pytest.mark.unit
def test_chainless_noncausal_heads_stream_from_every_core():
    from ttsim.perf.roofline_sdpa import _noncausal_chains, CALIBRATED_INJECTORS
    # T2.1 non-causal S4096 q128 nh32: 32 chunks per head over 110 cores, one chain per head (32 injectors)
    assert _noncausal_chains(32, 1024, 110) == 32 == CALIBRATED_INJECTORS
    assert _noncausal_chains(8, 128, 110) == 16          # cross S1024 q128 nh16: 16 chains, the measured points
    assert _noncausal_chains(1, 32, 110) == 0            # one chunk per head: no head spans two cores
    assert _noncausal_chains(4, 512, 128) == 0           # heads aligned to whole cores: no chain either
    assert _noncausal_chains(4, 512, 110) > 0
    calibrated = predict(_bh(S=4096, is_causal=False, **_GQA))
    assert "chain_geometry_off_calibration" not in calibrated.low_confidence_reasons and calibrated.config_echo["kv_chains"] == 32
    cross16 = predict(_bh(S=1024, kv_seq=4096, num_heads=16, is_causal=False))
    assert "chain_geometry_off_calibration" not in cross16.low_confidence_reasons     # measured within 5 percent
    # cross S128 over an 8192 KV: 32 chunks on 32 cores, every core reads the whole K/V alone
    alone = predict(_bh(S=128, kv_seq=8192, is_causal=False, **_GQA))
    assert alone.config_echo["kv_chains"] == 0 and "chain_geometry_off_calibration" in alone.low_confidence_reasons
    # With only 32 of the 110 cores reading, DRAM is not contended: at the measured 32-core rate (GRIDR)
    # the stream fits under the compute lane, so this shape is compute bound and pays no K/V wait. It was
    # stream bound while the table clamped a 32-core grid to the 64-core rate. No measured wall moves.
    assert alone.components["reader_wait"] == 0.0
    assert alone.wall_clock_cycles == 572_500        # the PACK-only price
    assert alone.wall_clock_cycles > 1.2 * predict(_bh(S=128, kv_seq=1024, is_causal=False, **_GQA)).wall_clock_cycles
    # fewer chains than the calibration count keep the injector lane but say so
    nc16 = predict(_bh(S=4096, num_heads=16, is_causal=False))
    assert nc16.config_echo["kv_chains"] == 16 and "chain_geometry_off_calibration" in nc16.low_confidence_reasons
    assert "kv_chains" not in predict(_bh(S=4096, **_GQA)).config_echo


@pytest.mark.unit
def test_noncausal_and_cross_share_the_pack_lane():
    # Cross attention runs the non-causal kernel; its R1a and T2.3 walls sit on the same PACK lane.
    nc, cx = WALL_TERMS_BH["prefill_noncausal"], WALL_TERMS_BH["cross"]
    assert (cx.pack_per_step, cx.pack_per_qtile, cx.pack_per_qktile, cx.pack_per_qk_dtile) == \
        (nc.pack_per_step, nc.pack_per_qtile, nc.pack_per_qktile, nc.pack_per_qk_dtile)
    assert nc.dram_law == "injector" and cx.dram_law == "none"
    assert predict(_bh(S=1024, kv_seq=4096, num_heads=16, is_causal=False)).components["reader_wait"] == 0.0
    # the chain head at q64 k128 is stream bound (T2.1: 37 percent K/V wait), at q128 k128 barely (2 percent)
    q64 = predict(_bh(S=4096, is_causal=False, q_chunk=64, **_GQA))
    q128 = predict(_bh(S=4096, is_causal=False, **_GQA))
    assert q64.components["reader_wait"] / q64.wall_clock_cycles > 0.3
    assert q128.components["reader_wait"] / q128.wall_clock_cycles < 0.05

@pytest.mark.unit
def test_components_sum_to_wall_and_are_nonnegative():
    cfgs = [c for _, _, c, _ in CAMPAIGN_WALLS] + [_bh(S=4096, **kw, **_GQA) for kw, _ in GRID1_WALLS]
    cfgs.append(SdpaConfig(S=4096, q_chunk=256, k_chunk=256, num_cores=64, arch=ARCH_BH, **_PROD))
    for cfg in cfgs:
        r = predict(cfg)
        assert list(r.components) == WALL_TERM_NAMES
        assert round(sum(r.components.values())) == r.wall_clock_cycles
        assert all(v >= 0.0 for v in r.components.values())
        assert r.components["init"] == cfg.arch.wall_fixed_cycles
        assert r.components["compute_floor"] == max(r.math_active_cycles, max(r.unpack_min_cycles, r.pack_min_cycles))


@pytest.mark.unit
def test_sfpu_issue_cannot_double_count_the_pack_lane():
    # The exp issue is PACK-thread work inside the fitted PACK lane: a row with a small FPU step and no
    # stream law (cross, LoFi, head_dim 32) must not add it on top of the lane (review R6).
    r = predict(_bh(S=4096, kv_seq=8192, num_heads=16, head_dim=32, fidelity="LoFi", is_causal=False))
    assert r.components["sfpu_issue"] == 0.0 and "sfpu_issue_exposed" not in r.low_confidence_reasons
    # it fires only on a wall table whose PACK lane sits under the exp issue with nothing else to hide it
    arch = ArchConfig(wall_terms={k: WallTerms(pack_per_qtile=10.0, pack_per_qktile=10.0) for k in WALL_TERMS_BH})
    thin = predict(SdpaConfig(S=4096, num_cores=110, num_heads=55, head_dim=32, fidelity="LoFi", arch=arch, is_causal=False))
    assert thin.components["sfpu_issue"] > 0.0 and "sfpu_issue_exposed" in thin.low_confidence_reasons
    # the wall-only rows (masked, sparse, joint) keep the exp under their lanes
    for cfg in (_bh(S=8192, num_heads=16, is_causal=False, has_attn_mask=True),
                _bh(S=2048, kv_seq=1024, num_heads=32, is_sparse=True, is_causal=False, **_MLA)):
        assert predict(cfg).components["sfpu_issue"] == 0.0


def _synthetic_terms(**kw):
    # One synthetic row for every regime so any config lands on the same constants.
    base = dict(pack_per_step=1000.0, pack_per_qtile=2000.0, pack_per_qktile=100.0, mask_branch_per_kchunk=8.0,
                control_per_kchunk=40.0, control_per_qtile=5.0)
    base.update(kw)
    return {k: WallTerms(**base) for k in WALL_TERMS_BH}


def _synth_cfg(**kw):
    arch = ArchConfig(wall_terms=_synthetic_terms(**kw.pop("terms", {})))
    # Two cores per head: the non-causal rows form chains and keep their own (synthetic) lanes.
    return SdpaConfig(num_cores=110, num_heads=55, arch=arch, **kw)


@pytest.mark.unit
def test_each_term_scales_with_its_declared_variable():
    # No DRAM law on the synthetic rows: the step is the PACK lane (it exceeds the floor), so the
    # front end per step is pack - floor - mask - control and control is per step.
    k128 = predict(_synth_cfg(S=4096, is_causal=False, k_chunk=128))
    k256 = predict(_synth_cfg(S=4096, is_causal=False, k_chunk=256))
    steps = lambda r: r.q_chunks_per_core * r.k_eff
    per_step = lambda r, n: r.components[n] / steps(r)
    assert steps(k256) == steps(k128) / 2                              # per k-chunk terms halve their count
    assert per_step(k128, "control") == per_step(k256, "control") == 40.0 + 5.0 * 4
    assert k128.components["mask_bracket"] == 0.0 and k256.components["mask_bracket"] == 0.0   # causal only
    pack = lambda qct, kct: 1000.0 + qct * (2000.0 + 100.0 * kct)
    for r, kct in ((k128, 4), (k256, 8)):
        assert per_step(r, "fe_issue") + per_step(r, "compute_floor") + per_step(r, "control") == pytest.approx(pack(4, kct))
    assert k128.components["reader_wait"] == 0.0 and k128.components["sfpu_issue"] == 0.0
    # Doubling S doubles the k-chunks each q-chunk visits and the q-chunks per core: per chunk the
    # step terms double.
    r8 = predict(_synth_cfg(S=8192, is_causal=False, k_chunk=128))
    assert r8.q_chunks_per_core == 2 * k128.q_chunks_per_core
    per_chunk = lambda r, n: r.components[n] / r.q_chunks_per_core
    for name in ("fe_issue", "control"):
        assert per_chunk(r8, name) == pytest.approx(2 * per_chunk(k128, name), rel=1e-3)   # the recip is per q chunk
    # Causal gets the mask bracket per visited k-chunk.
    cz = predict(_synth_cfg(S=4096, is_causal=True))
    assert cz.components["mask_bracket"] == pytest.approx(steps(cz) * 8.0)
    # A flat per-step term and a per-tile-MAC term add on top of the lanes and never go negative.
    flat = predict(_synth_cfg(S=4096, is_causal=False, terms=dict(fe_flat_per_kchunk=300.0)))
    assert per_step(flat, "fe_issue") == pytest.approx(per_step(k128, "fe_issue") + 300.0)
    tm = predict(_synth_cfg(S=4096, is_causal=False, terms=dict(fe_per_tile_mac=2.0)))
    assert per_step(tm, "fe_issue") == pytest.approx(per_step(k128, "fe_issue") + 2.0 * 4 * 4 * 8)
    # A provided-mask tile cost is serial with the lanes and lands in the mask term next to the bracket
    # (a masked op is causal-like); the non-causal mask path visits every k chunk.
    mk = predict(_synth_cfg(S=4096, is_causal=False, has_attn_mask=True, terms=dict(mask_per_tile=10.0)))
    assert per_step(mk, "mask_bracket") == pytest.approx(10.0 * 16 + 8.0) and mk.k_eff == 32.0
    # The DRAM lane: when it exceeds the compute lane the difference is reader_wait, per step; the
    # regime's stream factor scales the whole lane.
    for scale in (1.0, 1.3):
        dr = predict(_synth_cfg(S=4096, is_causal=False, terms=dict(dram_law="all_cores", stream_scale=scale, pack_per_step=0.0,
                                                                    pack_per_qtile=0.0, pack_per_qktile=0.0)))
        a = ArchConfig()
        dram_step = scale * (4 * (a.kv_stream_fixed_per_ktile + 8 * a.kv_stream_fixed_per_kv_tile) + 4 * 8 * 1088 / a.kv_stream_rate_bpc[110])
        compute_step = per_step(dr, "compute_floor") + per_step(dr, "fe_issue") + per_step(dr, "control")
        assert per_step(dr, "reader_wait") == pytest.approx(dram_step - compute_step)


@pytest.mark.unit
def test_pack_lane_matches_the_free_reader_ablation_points():
    # A4 (reader stub) walls per step at q128: 8872 / 13683 / 23041 cycles at k128 / k256 / k512 after the
    # 2900 fixed cycles; the causal PACK lane qct * (750 + 348 kct) meets these fit points within 3.5 / 3.5 / 10.
    t = WALL_TERMS_BH["prefill_causal"]
    pack = lambda qct, kct: t.pack_per_step + qct * (t.pack_per_qtile + (t.pack_per_qktile + 8 * t.pack_per_qk_dtile) * kct)
    for kct, meas, tol in ((4, 8872, 0.035), (8, 13683, 0.035), (16, 23041, 0.10)):
        assert abs(pack(4, kct) / meas - 1.0) <= tol
    # and the compute-bound T2.1 walls per step within 4.2 percent
    for qct, kct, meas in ((8, 4, 17875), (16, 4, 33923), (16, 16, 101754)):
        assert abs(pack(qct, kct) / meas - 1.0) <= 0.042
    assert t.mask_branch_per_kchunk == MASK_BRACKET_PER_STEP == 57.0


@pytest.mark.unit
def test_straggler_term_is_zero_when_chunks_divide_cores():
    # 110 heads x 8 q-chunks = 880 chunks = exactly 8 per core: no straggler. 111 heads spill one
    # chunk onto some cores, which then set the wall.
    even = predict(_bh(S=1024, num_heads=110, is_causal=False))
    odd = predict(_bh(S=1024, num_heads=111, is_causal=False))
    assert even.straggler_cycles == 0 and even.q_chunks_wall_core == 8
    assert odd.q_chunks_wall_core == 9 and odd.straggler_cycles > 0
    per_chunk = (odd.wall_clock_cycles - odd.components["init"]) / odd.q_chunks_wall_core
    assert odd.straggler_cycles == pytest.approx(per_chunk * (9 - odd.q_chunks_per_core), abs=1.0)


@pytest.mark.unit
def test_unknown_wall_regime_fails_fast():
    with pytest.raises(ValueError):
        ArchConfig().wall_terms_for("ring")


@pytest.mark.unit
def test_perf_stats_carries_wall_components():
    # The validation harness and the charts read the named terms from perf_stats; they must sum
    # to the booked cost and name the wall-terms row that produced them.
    ps = sdpa_perf_stats(_bh(S=4096, **_GQA))
    comp = ps["sdpa_wall_components"]
    assert list(comp) == WALL_TERM_NAMES
    assert round(sum(comp.values())) == ps["fused_compute_cycles"]
    assert ps["sdpa_wall_regime"] == "prefill_causal"
    assert ps["sdpa_cycle_breakdown"]["straggler"] >= 0
    assert 0.0 <= ps["sdpa_cycle_breakdown"]["overlap_frac"] < 1.0
    # decode reports its own three terms (fixed, math, DRAM wait beyond the math), same contract
    dec = predict_decode(cache_len=8192, num_q_heads=32, num_kv_heads=8, head_dim=128, batch=8)
    dps = dec.to_polaris_op_perf_stats()
    assert list(dps["sdpa_wall_components"]) == ["init", "compute_floor", "kv_stream_wait"]
    assert round(sum(dps["sdpa_wall_components"].values())) == dps["fused_compute_cycles"]
    assert dps["sdpa_wall_regime"] == "decode"
    # and the booked op cost through the device path is the same wall
    device = Device(_MockSimConfig())
    op = _sdpa_op("sdpa_comp", _bh(S=4096, **_GQA))
    device.execute_op(op)
    assert op.compute_cycles == round(sum(op.perf_stats["sdpa_wall_components"].values()))


# ---- dedicated SFPU issue thread (thread_split) ------------------------------------------------

# Synthetic constants only (round numbers, not measurements): the section-graph mechanism is exercised with BH
# geometry and whatever lane costs a test wants to see. No architecture has these values.
_SPLIT = dict(name="split", thread_split=True, fpu_cycles_per_pass={"LoFi": 10.0, "HiFi2": 20.0, "HiFi4": 40.0},
              t_unpack_pass=20.0, t_pack_pass=10.0,
              sfpu_slots_per_tile={"exp": 100.0, "passthrough": 10.0, "max_combine": 20.0, "recip": 100.0,
                                   "row_max": 50.0, "sub": 10.0, "row_sum": 50.0},
              f_riscv=1.0, t_handoff=5.0, t_l1_turn=10.0, c_arb=0.0,
              startup_a=1000.0, startup_b=100.0, startup_g0=4.0, compute_units_per_core=1, dest_tiles_half=4)


def _split_arch(**kw):
    return ArchConfig(**{**_SPLIT, **kw})


def _graph(f, s, p, dependent):
    # Sections whose lanes cost exactly f / s / p cycles under _SPLIT (HiFi2 pass 20, exp slot 100, pack 10).
    secs = tuple(Section(f"s{i}", tile_passes=fi / 20.0, sfpu={"exp": si / 100.0}, pack_tiles=pi / 10.0)
                 for i, (fi, si, pi) in enumerate(zip(f, s, p)))
    return StepGraph(secs, tuple(dependent))


def _step(f, s, p, dependent, **kernel):
    return evaluate_step(_graph(f, s, p, dependent), KernelDescriptor(**kernel), _split_arch(), "HiFi2")


@pytest.mark.unit
def test_thread_split_components_sum_to_wall_and_are_flagged():
    for cfg in (SdpaConfig(S=4096, num_cores=110, arch=_split_arch(), **_GQA),
                SdpaConfig(S=1024, kv_seq=4096, is_causal=False, q_chunk=256, num_cores=64, arch=_split_arch()),
                SdpaConfig(S=2048, has_attn_mask=True, fidelity="LoFi", accum_dtype="float32", num_cores=110,
                           arch=_split_arch(compute_units_per_core=4, f_riscv=1.5, c_arb=0.2))):
        r = predict(cfg)
        assert set(r.components) == set(THREAD_SPLIT_TERMS)
        assert all(v >= 0.0 for v in r.components.values())
        assert round(sum(r.components.values())) == r.wall_clock_cycles > 0
        assert r.low_confidence and "thread_split_uncalibrated_in_repo" in r.low_confidence_reasons
        assert r.kernel_path == "thread_split" and r.overlap_frac == 0.0
        assert not any(reason.startswith("off_calibration") for reason in r.low_confidence_reasons)
        stats = r.to_polaris_op_perf_stats()
        assert stats["fused_compute_cycles"] == r.wall_clock_cycles
        assert round(sum(stats["sdpa_wall_components"].values())) == r.wall_clock_cycles
    # the issue floor and the arbitration share appear only when their constants say so
    assert r.components["issue_share"] > 0.0 and r.components["dest_arb"] > 0.0


@pytest.mark.unit
def test_independent_boundaries_overlap_to_the_longest_lane_plus_handoffs():
    n = 4
    for f, s, p in (([100] * n, [20] * n, [10] * n), ([10] * n, [100] * n, [10] * n), ([10] * n, [20] * n, [100] * n)):
        c = _step(f, s, p, [False] * n)
        assert c.period == pytest.approx(max(sum(f), sum(s), sum(p)) + n * 5.0)
        assert c.dep_serial == n * 5.0 and c.n_dependent == 0
        assert sum(c.terms().values()) == pytest.approx(c.period)
        assert c.fpu_stream == sum(f)                      # the FPU lane is always fully on the period
        assert c.sfpu_stream == c.pack_stream == 0.0       # nothing drains through an L1 edge
        assert c.sync_pingpong == pytest.approx(c.period - c.dep_serial - sum(f))   # bank waits carry the rest
    # no lane dominates the other two: the two-bank ring binds at (f + s + p) / 2 per section
    ring = _step([10] * n, [10] * n, [10] * n, [False] * n)
    assert ring.period == pytest.approx(n * 15.0 + n * 5.0) and ring.sync_pingpong == pytest.approx(n * 5.0)


@pytest.mark.unit
def test_dependent_boundaries_serialise_to_the_lane_sum():
    n = 4
    f, s, p = [100, 30, 10, 60], [20, 80, 10, 40], [10, 10, 90, 20]
    c = _step(f, s, p, [True] * n)
    assert c.period == pytest.approx(sum(f) + sum(s) + sum(p) + n * (5.0 + 10.0))
    assert (c.fpu_stream, c.sfpu_stream, c.pack_stream) == (sum(f), sum(s), sum(p))
    assert c.sync_pingpong == 0.0 and c.n_dependent == n
    # full sync serialises the bank even without data edges, but pays no L1 turnaround
    full = _step(f, s, p, [False] * n, dst_sync="full")
    assert full.period == pytest.approx(sum(f) + sum(s) + sum(p) + n * 5.0)
    assert full.sync_pingpong == pytest.approx(sum(s) + sum(p))
    # the issue floor over the engine time is booked separately and only where the SFPU is exposed
    arch = _split_arch(f_riscv=1.5)
    x = evaluate_step(_graph(f, s, p, [True] * n), KernelDescriptor(), arch, "HiFi2")
    assert x.sfpu_stream == pytest.approx(sum(s)) and x.issue_share == pytest.approx(0.5 * sum(s))
    y = evaluate_step(_graph([100] * n, [20] * n, [10] * n, [False] * n), KernelDescriptor(), arch, "HiFi2")
    assert y.issue_share == 0.0


@pytest.mark.unit
def test_data_edges_never_reduce_the_step():
    import itertools
    n = 4
    for f, s, p in (([100, 30, 10, 60], [20, 80, 10, 40], [10, 10, 90, 20]), ([50] * n, [50] * n, [50] * n)):
        period = {pat: _step(f, s, p, pat).period for pat in itertools.product([False, True], repeat=n)}
        for pat, value in period.items():
            for i in range(n):
                if not pat[i]:
                    flipped = tuple(True if j == i else d for j, d in enumerate(pat))
                    assert period[flipped] >= value - 1e-9
            assert _step(f, s, p, pat, dst_sync="full").period >= value - 1e-9
        assert period[(False,) * n] < period[(True,) * n]
    # the l1 chain (SFPU reads the packed tiles) is a valid chain with no arbitration term
    c = _step([100] * n, [20] * n, [10] * n, [True] * n, handoff_chain="l1")
    assert c.period == pytest.approx(n * (100 + 20 + 10) + n * 15.0) and c.dest_arb == 0.0
    d = _step([100] * n, [20] * n, [10] * n, [True] * n, handoff_chain="dest")
    arb = evaluate_step(_graph([100] * n, [20] * n, [10] * n, [True] * n), KernelDescriptor(),
                        _split_arch(c_arb=0.5), "HiFi2")
    assert arb.dest_arb == pytest.approx(0.5 * min(n * 20, n * 10)) and arb.period == pytest.approx(d.period + arb.dest_arb)


@pytest.mark.unit
def test_default_sdpa_step_follows_the_geometry():
    qct, kct, dct_qk, dct_v, tiles = 2, 8, 4, 2, 4
    first = default_sdpa_step(qct, kct, dct_qk, dct_v, tiles, rescale=False, normalize=False)
    last = default_sdpa_step(qct, kct, dct_qk, dct_v, tiles, rescale=True, normalize=True)
    mid = default_sdpa_step(qct, kct, dct_qk, dct_v, tiles, rescale=True, normalize=False)
    names = [s.name for s in first.sections][:len(first.sections) // qct]
    assert names == ["qk0", "qk1", "row_max", "max_update", "exp0", "exp1", "pv0", "park_max", "park_sum", "park_out"]
    assert [s.name for s in last.sections][:12] == ["qk0", "qk1", "row_max", "max_update", "exp0", "exp1", "pv0",
                                                    "alpha", "sum_rescale", "row_sum_final", "out_rescale", "normalize"]
    assert len(first.sections) == 10 * qct and len(last.sections) == 12 * qct and len(mid.sections) == 13 * qct
    # sub pass boundaries and the wrap to the next q tile are independent, every other one is an L1 edge
    assert first.dependent[:10] == (False, True, True, True, False, True, True, True, True, False)
    assert first.n_dependent == 7 * qct and last.n_dependent == 9 * qct
    by_name = {s.name: s for s in first.sections[:10]}
    assert by_name["qk0"].tile_passes == tiles * dct_qk and by_name["qk1"].pack_tiles == tiles
    assert sum(s.tile_passes for s in first.sections) == qct * kct * (dct_qk + dct_v)
    assert by_name["exp0"].sfpu == {"exp": float(tiles)} and by_name["exp0"].pack_tiles == 2 * tiles   # sum collapse
    assert by_name["row_max"].fpu_tiles == kct and by_name["pv0"].pack_tiles == dct_v
    assert sum(s.sfpu.get("exp", 0.0) for s in last.sections) == qct * (kct + 1)    # plus the alpha exp
    # a non dividing k chunk leaves a short last sub pass; a dense mask adds one section per q tile
    odd = default_sdpa_step(1, 6, dct_qk, dct_v, tiles, rescale=False, normalize=False, mask=True)
    assert [s.name for s in odd.sections][:4] == ["qk0", "qk1", "mask", "row_max"]
    assert odd.sections[1].pack_tiles == 2 and odd.sections[2].fpu_tiles == 6
    # row statistics on the SFPU: the max rides in the QK^T sections, sub and sum in the exp sections
    sf = default_sdpa_step(1, kct, dct_qk, dct_v, tiles, rescale=False, normalize=False, stats_placement="sfpu")
    assert "row_max" not in [s.name for s in sf.sections] and sf.sections[0].sfpu == {"row_max": float(tiles)}
    e = next(s for s in sf.sections if s.name == "exp0")
    assert e.sfpu == {"sub": float(tiles), "exp": float(tiles), "row_sum": float(tiles)} and e.pack_tiles == tiles
    # another exp path only changes which slot table entry the exp sections read
    alt = default_sdpa_step(1, kct, dct_qk, dct_v, tiles, rescale=False, normalize=False, exp_path="exp_fast")
    assert next(s for s in alt.sections if s.name == "exp0").sfpu == {"exp_fast": float(tiles)}
    # the l1 chain has no passthroughs to keep a DEST ring moving
    l1 = default_sdpa_step(1, kct, dct_qk, dct_v, tiles, rescale=False, normalize=False, handoff_chain="l1")
    assert all("passthrough" not in s.sfpu for s in l1.sections)
    with pytest.raises(ValueError):
        default_sdpa_step(0, kct, dct_qk, dct_v, tiles, rescale=False, normalize=False)
    with pytest.raises(ValueError):
        KernelDescriptor(dst_sync="quarter")
    with pytest.raises(ValueError):
        StepGraph(first.sections, first.dependent[:-1])


@pytest.mark.unit
def test_thread_split_wall_follows_the_descriptor_and_the_geometry():
    a = _split_arch()
    cfg = SdpaConfig(S=1024, kv_seq=2048, is_causal=False, q_chunk=128, k_chunk=128, num_heads=1, num_cores=1, arch=a)
    r = predict(cfg)
    qct, kct, dct = 4, 4, 4
    tiles = 4
    kd = KernelDescriptor()
    first = evaluate_step(default_sdpa_step(qct, kct, dct, dct, tiles, rescale=False, normalize=False), kd, a, "HiFi2")
    mid = evaluate_step(default_sdpa_step(qct, kct, dct, dct, tiles, rescale=True, normalize=False), kd, a, "HiFi2")
    last = evaluate_step(default_sdpa_step(qct, kct, dct, dct, tiles, rescale=True, normalize=True), kd, a, "HiFi2")
    # 8 q chunks on one core and unit, each visiting 16 k chunks: first, 14 middle, last, plus one startup per chunk
    per_chunk = first.period + 14 * mid.period + last.period + (1000.0 + 100.0 * (qct - 4))
    assert r.wall_clock_cycles == round(8 * per_chunk)
    assert r.components["startup_transient"] == 8 * 1000.0 and r.q_chunks_wall_unit == 8
    # k chunk 128 fits one DEST section of 4 tiles: 8 / 11 / 10 sections per q tile, 4 q tiles per chunk
    assert r.config_echo["sections"] == {"first": 32, "step": 44, "last": 40}
    # a caller supplied graph is used for every step; the startup term scales with the sub block count
    one = StepGraph((Section("only", tile_passes=1.0, sfpu={"exp": 1.0}, pack_tiles=1.0),), (True,))
    own = predict(cfg, kernel=KernelDescriptor(step=one))
    assert own.components["fpu_stream"] == pytest.approx(8 * 16 * 20.0)
    assert own.components["dep_serial"] == pytest.approx(8 * 16 * (5.0 + 10.0))
    wide = predict(replace(cfg, q_chunk=256))
    assert wide.components["startup_transient"] == 4 * (1000.0 + 100.0 * 4)
    # compute units share the core's q chunks, ceil quantised
    two = predict(replace(cfg, arch=_split_arch(compute_units_per_core=2)))
    three = predict(replace(cfg, arch=_split_arch(compute_units_per_core=3)))
    assert two.wall_clock_cycles == round(4 * per_chunk) and three.wall_clock_cycles == round(3 * per_chunk)
    assert three.q_chunks_wall_unit == 3 and three.straggler_cycles > 0
    # full sync and a slower unpack never make the step cheaper
    assert predict(cfg, kernel=KernelDescriptor(dst_sync="full")).wall_clock_cycles >= r.wall_clock_cycles
    assert predict(replace(cfg, arch=_split_arch(t_unpack_pass=40.0))).wall_clock_cycles > r.wall_clock_cycles
    # fp32 accumulation halves the tiles per DEST section: QK^T, exp and P.V each need a second sub pass
    assert predict(replace(cfg, accum_dtype="float32")).config_echo["sections"]["first"] == 32 + 3 * 4


@pytest.mark.unit
def test_thread_split_refuses_missing_or_out_of_range_constants():
    with pytest.raises(ValueError, match="thread_split needs") as e:
        predict(SdpaConfig(S=4096, num_cores=110, arch=ArchConfig(thread_split=True)))
    for name in THREAD_SPLIT_FIELDS:
        assert name in str(e.value)
        with pytest.raises(ValueError, match=name):
            predict(SdpaConfig(S=4096, num_cores=110, arch=_split_arch(**{name: None})))
    assert ArchConfig().thread_split is False and all(getattr(ArchConfig(), n) is None for n in THREAD_SPLIT_FIELDS)
    for bad in (dict(c_arb=1.5), dict(f_riscv=0.5), dict(compute_units_per_core=0), dict(dest_tiles_half=0),
                dict(t_l1_turn=-1.0), dict(fpu_cycles_per_pass={"LoFi": 10.0})):
        with pytest.raises(ValueError):
            predict(SdpaConfig(S=4096, num_cores=110, arch=_split_arch(**bad)))
    with pytest.raises(ValueError, match="row_max"):
        predict(SdpaConfig(S=4096, num_cores=110, arch=_split_arch(sfpu_slots_per_tile={"exp": 100.0})),
                kernel=KernelDescriptor(stats_placement="sfpu"))
    # decode has no section graph yet and must not fall back to the BH union law
    with pytest.raises(ValueError, match="decode"):
        predict_decode(cache_len=4096, num_q_heads=32, num_kv_heads=8, head_dim=128, arch=_split_arch())


@pytest.mark.unit
def test_arch_overrides_loader_rejects_unknown_keys_and_accepts_a_complete_file(tmp_path):
    import json
    complete = dict(_SPLIT)
    yaml_path = tmp_path / "arch.yaml"
    lines = ["# synthetic test values"]
    for k, v in complete.items():
        lines.append(f"{k}: {json.dumps(v)}")
    yaml_path.write_text("\n".join(lines) + "\n")
    a = ArchConfig.from_overrides(yaml_path)
    assert a.thread_split is True and a.name == "split" and a.t_unpack_pass == 20.0
    assert a.sfpu_slots_per_tile["exp"] == 100.0 and a.fpu_cycles_per_pass == complete["fpu_cycles_per_pass"]
    assert a.wall_fixed_cycles == ArchConfig().wall_fixed_cycles      # untouched fields keep the BH defaults
    r = predict(SdpaConfig(S=4096, num_cores=110, arch=a))
    assert r.wall_clock_cycles == predict(SdpaConfig(S=4096, num_cores=110, arch=_split_arch())).wall_clock_cycles
    # json works too; per grid tables get integer keys back
    json_path = tmp_path / "arch.json"
    json_path.write_text(json.dumps({**complete, "kv_stream_rate_bpc": {"110": 2.0, "64": 4.0},
                                     "wall_terms": {"prefill_causal": {"pack_per_qtile": 1.0}}}))
    b = ArchConfig.from_overrides(json_path)
    assert b.kv_stream_rate_bpc == {110: 2.0, 64: 4.0} and b.wall_terms["prefill_causal"].pack_per_qtile == 1.0
    # unknown keys are refused by name, a non mapping file too
    (tmp_path / "bad.yaml").write_text("thread_split: true\nunknown_field: 3\n")
    with pytest.raises(ValueError, match="unknown_field"):
        ArchConfig.from_overrides(tmp_path / "bad.yaml")
    (tmp_path / "list.json").write_text("[1, 2]")
    with pytest.raises(ValueError):
        ArchConfig.from_overrides(tmp_path / "list.json")
    # a partial file over a base keeps the base's values
    (tmp_path / "part.yaml").write_text("t_handoff: 7.0\n")
    c = ArchConfig.from_overrides(tmp_path / "part.yaml", base=a)
    assert c.t_handoff == 7.0 and c.thread_split is True and c.t_pack_pass == 10.0


# ---- per-config overrides (production program configs) -----------------------------------------

@pytest.mark.unit
def test_predict_overrides_match_explicit_config_fields():
    from dataclasses import replace
    cfg = _bh(S=4096, **_GQA)
    # no overrides: bit-identical to the plain call, and inside the calibration envelope
    plain = predict(cfg)
    assert predict(cfg, num_cores=None, fidelity=None).wall_clock_cycles == plain.wall_clock_cycles
    assert not plain.low_confidence
    # each override equals setting the field on the config; all of them leave the envelope
    prod = predict(cfg, num_cores=64, fidelity="HiFi4", exp_approx_mode=False, fp32_dest_acc=True)
    explicit = predict(replace(cfg, num_cores=64, fidelity="HiFi4", exp_approx_mode=False, fp32_dest_acc=True))
    assert prod.wall_clock_cycles == explicit.wall_clock_cycles
    assert prod.q_chunks_wall_core == math.ceil(32 * 32 / 64) and prod.low_confidence
    assert prod.fpu_matmul_cycles == pytest.approx(plain.fpu_matmul_cycles * 2 * 110 / 64, rel=1e-6)  # HiFi4 = 2x cpt
    # the caller's config object is not mutated
    assert cfg.num_cores == 110 and cfg.fidelity == "HiFi2" and not cfg.fp32_dest_acc


@pytest.mark.unit
def test_unknown_override_falls_back_to_defaults_and_flags():
    cfg = _bh(S=4096, **_GQA)
    plain = predict(cfg)
    for bad in (dict(fidelity="HiFi9"), dict(num_cores=0), dict(num_cores=-4), dict(num_cores="64")):
        r = predict(cfg, **bad)
        assert r.wall_clock_cycles == plain.wall_clock_cycles, bad
        assert r.low_confidence, bad
    # a bad value on the config itself still fails fast (predict is public)
    with pytest.raises(ValueError):
        predict(_bh(S=4096, fidelity="HiFi9"))


@pytest.mark.unit
def test_calibration_envelope_flags_production_axes():
    # Each production axis outside the calibration family flags the prediction on its own.
    assert not predict(_bh(S=4096, **_GQA)).low_confidence
    assert predict(SdpaConfig(S=4096, num_cores=80, arch=ARCH_BH, **_GQA)).low_confidence
    assert predict(_bh(S=4096, fidelity="HiFi4", **_GQA)).low_confidence
    assert predict(_bh(S=4096, exp_approx_mode=False, **_GQA)).low_confidence
    assert predict(_bh(S=4096, fp32_dest_acc=True, **_GQA)).low_confidence
    # the flag reaches perf_stats and the attr spelling used by the program config is read
    ps = sdpa_perf_stats(_bh(S=4096, fp32_dest_acc=True, **_GQA))
    assert ps["sdpa_mla_low_confidence"] is True
    cfg = sdpa_config_from_shapes([1, 32, 4096, 128], [1, 8, 4096, 128], [1, 8, 4096, 128],
                                  {"is_causal": True, "element_size": 1, "fp32_dest_acc_en": True})
    assert cfg.fp32_dest_acc is True


# ---- factory work split: causal pairs ----------------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize("cfg,nq,ceil_nq", [
    (_bh(S=1024, **_GQA), 4, 3),                       # 128 pairs over 110 cores: 18 cores take 2 pairs
    (_bh(S=4096, q_chunk=256, **_GQA), 6, 5),
    (_bh(S=4096, q_chunk=512, **_GQA), 4, 3),
    (_bh(S=4096, q_chunk=64, **_GQA), 20, 19),
    (_bh(S=16384, q_chunk=256, **_GQA), 20, 19),
    (_bh(S=2048, kv_seq=6144, chunk_start_idx=4096, num_heads=16), 4, 3),
    (_bh(S=1024, q_chunk=32, num_heads=16, **_MLA), 6, 5),
    (_bh(S=4096, **_GQA), 10, 10),                     # anchor: both rules agree
])
def test_causal_pair_split_sets_wall_core_chunks(cfg, nq, ceil_nq):
    # The factory hands out light/heavy chunk pairs when causal and the chunk count per head is
    # even, so the most loaded core owns (pairs // cores) * 2 + 2 chunks, not ceil(total / cores).
    r = predict(cfg)
    assert r.q_chunks_wall_core == nq
    assert math.ceil(r.q_chunks_per_core) == ceil_nq
    # The extra chunks are the tail phase, priced per step on their own (never above a contended step).
    light = sum(v for k, v in r.components.items() if k not in ("init", "straggler"))
    tail_ratio = r.components["straggler"] / light * r.q_chunks_per_core / max(1e-9, nq - r.q_chunks_per_core)
    assert 0.0 < tail_ratio <= 1.0 + 1e-9


@pytest.mark.unit
def test_tail_cores_are_the_remainder_of_the_split():
    from ttsim.perf.roofline_sdpa import _tail_cores
    # Causal hands out pairs, so the remainder is over pairs: S1024 is 128 pairs over 110 cores, 18 of which
    # take a second one; S2048 36 of 256; S8192 34 of 1024. An exact split leaves no tail.
    assert _tail_cores(8 * 32, 8, 110, True) == 18
    assert _tail_cores(16 * 32, 16, 110, True) == 36
    assert _tail_cores(64 * 32, 64, 110, True) == 34
    assert _tail_cores(32 * 32, 32, 64, True) == 64            # 512 pairs over 64 cores divides exactly
    assert _tail_cores(112, 8, 110, False) == 2                # flat split: the remainder is over chunks
    assert _tail_cores(40, 2, 110, True) == 20                 # fewer units than cores: all of them


@pytest.mark.unit
def test_tail_phase_prices_the_extra_steps_with_fewer_readers():
    # The wall core's steps beyond the light cores run once the light cores are done, so only the remainder
    # cores are on the DRAM and the step falls back on the compute lane. Measured on the card: the extra-pair
    # steps run at 9 to 12 k cycles against the 15.5 k contended lane (bh/zone_decomposition.md).
    from ttsim.perf.roofline_sdpa import _kv_stream_rate
    r = predict(_bh(S=1024, **_GQA))
    light = r.q_chunks_per_core * r.k_eff
    tail_steps = r.steps_wall_core - light
    contended = sum(v for k, v in r.components.items() if k not in ("init", "straggler")) / light
    tail = r.components["straggler"] / tail_steps
    assert tail_steps == pytest.approx((r.q_chunks_wall_core - r.q_chunks_per_core) * r.k_eff, rel=1e-9)
    assert tail < contended
    assert 8_000 < tail < 13_000 and 14_000 < contended < 17_000
    # 18 readers instead of 110 is a higher per-core rate. The rate table clamps at its measured grids, so
    # the tail is charged the measured 64-core rate rather than an extrapolation below it: the speed-up the
    # model takes is never larger than one it was shown on the card.
    assert _kv_stream_rate(ARCH_BH, 18, tail=True) == _kv_stream_rate(ARCH_BH, 64) == ARCH_BH.kv_stream_rate_bpc[64]
    assert _kv_stream_rate(ARCH_BH, 18) > _kv_stream_rate(ARCH_BH, 110)
    # An exactly divisible split has no tail at all: every core runs the same number of pairs
    exact = predict(SdpaConfig(S=4096, num_cores=64, arch=ARCH_BH, **_GQA))
    assert exact.components["straggler"] == 0.0


@pytest.mark.unit
@pytest.mark.parametrize("S,active", [(1024, 16), (2048, 32), (4096, 64)])
def test_pair_split_active_cores_match_reduced_core_runs(S, active):
    # q512 k512 nh16 at these lengths puts one pair per core and leaves the rest idle.
    r = predict(_bh(S=S, q_chunk=512, k_chunk=512, num_heads=16))
    assert (r.active_cores, r.q_chunks_wall_core) == (active, 2)


@pytest.mark.unit
def test_non_causal_and_odd_chunk_counts_use_the_flat_split():
    from ttsim.perf.roofline_sdpa import _wall_core_chunks
    nc = predict(_bh(S=1024, is_causal=False, **_GQA))
    assert nc.q_chunks_wall_core == math.ceil(nc.q_chunks_per_core) == 3
    assert nc.active_cores == 110
    # causal with an odd chunk count per head (S=640 / q128 = 5) is not paired
    assert _wall_core_chunks(5 * 32, 5, 110, True) == (2, 110)
    assert _wall_core_chunks(8 * 32, 8, 110, True) == (4, 110)
    assert _wall_core_chunks(8 * 32, 8, 110, False) == (3, 110)
    # fewer chunks than cores: every chunk (or pair) lands on its own core
    assert _wall_core_chunks(40, 2, 110, True) == (2, 20)
    assert _wall_core_chunks(40, 2, 110, False) == (1, 40)


# ---- attribute plumbing: program / compute configs reach the model ------------------------------

def _shim_dev():
    from ttsim.front.ttnn.device import ARCH, Device
    dev = Device(device_id=0)
    dev.architecture = ARCH.BLACKHOLE
    return dev


def _tt(dev, shape, name, dtype=None, layout=None):
    from ttsim.front.ttnn.tensor import DataType, Layout, Tensor
    return Tensor(name=name, shape=shape, dtype=dtype or DataType.BFLOAT16,
                  layout=layout or Layout.TILE_LAYOUT, device=dev)


def _last_sdpa(dev):
    return [o for o in dev.ops.values() if o.optype == "ScaledDotProductAttention"][-1]


def _qkv(dev, S=4096, q_dtype=None, kv_dtype=None):
    from ttsim.front.ttnn.tensor import DataType
    q_dtype = q_dtype or DataType.BFLOAT8_B
    kv_dtype = kv_dtype or q_dtype
    return (_tt(dev, [1, 32, S, 128], "q", q_dtype), _tt(dev, [1, 8, S, 128], "k", kv_dtype),
            _tt(dev, [1, 8, S, 128], "v", kv_dtype))


def _llama_prefill_cfgs(q=256, k=256):
    import ttsim.front.ttnn as ttnn
    pc = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=ttnn.CoreCoord(8, 8), q_chunk_size=q,
                                k_chunk_size=k, exp_approx_mode=False)
    ck = ttnn.BlackholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                           fp32_dest_acc_en=True, packer_l1_acc=True)
    return pc, ck


def _decode_tensors(dev, batch=32, blocks=1024, block=32, kv_dtype=None):
    from ttsim.front.ttnn.tensor import DataType, Layout
    kv_dtype = kv_dtype or DataType.BFLOAT8_B
    q = _tt(dev, [1, batch, 32, 128], "dq")
    kc = _tt(dev, [blocks, 8, block, 128], "dk", kv_dtype)
    vc = _tt(dev, [blocks, 8, block, 128], "dv", kv_dtype)
    pt = _tt(dev, [batch, blocks], "dpt", DataType.INT32, Layout.ROW_MAJOR_LAYOUT)
    cp = _tt(dev, [batch], "dcp", DataType.INT32, Layout.ROW_MAJOR_LAYOUT)
    return q, kc, vc, pt, cp


@pytest.mark.unit
def test_shim_forwards_program_config_chunks_and_grid():
    import ttsim.front.ttnn as ttnn
    dev = _shim_dev()
    pc, ck = _llama_prefill_cfgs()
    ttnn.transformer.scaled_dot_product_attention(*_qkv(dev), is_causal=True, program_config=pc,
                                                  compute_kernel_config=ck)
    a = _last_sdpa(dev).attrs
    assert (a["q_chunk_size"], a["k_chunk_size"], a["num_cores"], a["exp_approx_mode"]) == (256, 256, 64, False)
    assert a["sdpa_defaulted"] == ""
    echo = _last_sdpa(dev).perf_stats["sdpa_config"]
    assert (echo["q_chunk"], echo["k_chunk"], echo["num_cores"], echo["exp_approx_mode"]) == (256, 256, 64, False)


@pytest.mark.unit
def test_shim_sub_core_grids_core_count_overrides_grid():
    import ttsim.front.ttnn as ttnn
    dev = _shim_dev()
    scg = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))])
    pc = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=(11, 10), sub_core_grids=scg,
                                q_chunk_size=128, k_chunk_size=128)
    ttnn.transformer.scaled_dot_product_attention(*_qkv(dev), program_config=pc)
    assert _last_sdpa(dev).attrs["num_cores"] == 64


@pytest.mark.unit
def test_shim_forwards_compute_kernel_config_fields():
    import ttsim.front.ttnn as ttnn
    dev = _shim_dev()
    ck = ttnn.BlackholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                           fp32_dest_acc_en=True, packer_l1_acc=False)
    ttnn.transformer.scaled_dot_product_attention(*_qkv(dev), compute_kernel_config=ck)
    a = _last_sdpa(dev).attrs
    assert (a["fidelity"], a["fp32_dest_acc_en"], a["packer_l1_acc"], a["math_approx_mode"]) == ("HiFi4", True, False, False)
    assert "fidelity" not in a["sdpa_defaulted"] and "q_chunk_size" in a["sdpa_defaulted"]


@pytest.mark.unit
def test_shim_bare_math_fidelity_enum_accepted():
    import ttsim.front.ttnn as ttnn
    dev = _shim_dev()
    ttnn.transformer.scaled_dot_product_attention(*_qkv(dev), compute_kernel_config=ttnn.MathFidelity.HiFi4)
    a = _last_sdpa(dev).attrs
    assert a["fidelity"] == "HiFi4"
    for f in ("fp32_dest_acc_en", "packer_l1_acc", "math_approx_mode"):
        assert f in a["sdpa_defaulted"].split(",")


@pytest.mark.unit
def test_shim_exp_mode_comes_from_program_config_not_math_approx():
    import ttsim.front.ttnn as ttnn
    dev = _shim_dev()
    pc = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=(8, 8), q_chunk_size=128, k_chunk_size=128)
    ck = ttnn.BlackholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=True)
    ttnn.transformer.scaled_dot_product_attention(*_qkv(dev), program_config=pc, compute_kernel_config=ck)
    op = _last_sdpa(dev)
    assert "exp_approx_mode" not in op.attrs and "exp_approx_mode" in op.attrs["sdpa_defaulted"].split(",")
    assert op.perf_stats["sdpa_config"]["exp_approx_mode"] is True
    assert "exp_mode_absent" in op.perf_stats["sdpa_low_confidence_reasons"].split(",")


@pytest.mark.unit
def test_shim_none_configs_fall_to_kernel_defaults_and_flag():
    import ttsim.front.ttnn as ttnn
    dev = _shim_dev()
    ttnn.transformer.scaled_dot_product_attention(*_qkv(dev), is_causal=True)
    ps = _last_sdpa(dev).perf_stats
    echo = ps["sdpa_config"]
    assert (echo["q_chunk"], echo["k_chunk"], echo["num_cores"], echo["fidelity"]) == (32, 32, 110, "HiFi2")
    assert echo["exp_approx_mode"] is True and echo["fp32_dest_acc"] is False
    reasons = set(ps["sdpa_low_confidence_reasons"].split(","))
    assert {"program_config_absent", "grid_absent", "exp_mode_absent", "fidelity_absent",
            "fp32_acc_absent", "off_calibration_chunk"} <= reasons
    assert ps["sdpa_mla_low_confidence"] is True
    assert set(ps["sdpa_defaulted_attrs"].split(",")) == {"q_chunk_size", "k_chunk_size", "num_cores",
                                                        "exp_approx_mode", "fidelity", "fp32_dest_acc_en",
                                                        "packer_l1_acc", "math_approx_mode"}


@pytest.mark.unit
def test_shim_prefill_records_attention_sink_and_kv_dtype():
    import ttsim.front.ttnn as ttnn
    from ttsim.front.ttnn.tensor import DataType
    dev = _shim_dev()
    q, k, v = _qkv(dev, q_dtype=DataType.BFLOAT16, kv_dtype=DataType.BFLOAT8_B)
    sink = _tt(dev, [1, 32, 1, 1], "sink")
    ttnn.transformer.scaled_dot_product_attention(q, k, v, is_causal=False, attention_sink=sink)
    op = _last_sdpa(dev)
    assert op.attrs["attention_sink"] is True
    assert (op.attrs["element_size"], op.attrs["kv_element_size"]) == (2, 1)
    echo = op.perf_stats["sdpa_config"]
    assert (echo["input_dtype"], echo["kv_input_dtype"], echo["attention_sink"]) == ("bfloat16", "bfp8_b", True)
    # a bf8 cache under a bf16 query reads half the K/V bytes of an all-bf16 op
    dev2 = _shim_dev()
    ttnn.transformer.scaled_dot_product_attention(*_qkv(dev2, q_dtype=DataType.BFLOAT16), is_causal=False)
    assert op.perf_stats["inBytes"] < _last_sdpa(dev2).perf_stats["inBytes"]


@pytest.mark.unit
def test_shim_chunked_entry_point_routes_and_records_start():
    import ttsim.front.ttnn as ttnn
    from ttsim.front.ttnn.tensor import DataType, Layout
    dev = _shim_dev()
    q = _tt(dev, [1, 32, 2048, 128], "cq", DataType.BFLOAT8_B)
    k = _tt(dev, [192, 8, 32, 128], "ck", DataType.BFLOAT8_B)
    v = _tt(dev, [192, 8, 32, 128], "cv", DataType.BFLOAT8_B)
    pt = _tt(dev, [1, 192], "cpt", DataType.INT32, Layout.ROW_MAJOR_LAYOUT)
    pc, ck = _llama_prefill_cfgs()
    out = ttnn.transformer.chunked_scaled_dot_product_attention(
        q, k, v, pt, chunk_start_idx=4096, program_config=pc, compute_kernel_config=ck,
        paged_cache_geometry=ttnn.PagedCacheGeometryOverride(block_size=32, num_kv_heads=8))
    assert out.logical_shape().as_list() == [1, 32, 2048, 128]
    op = _last_sdpa(dev)
    assert (op.attrs["sdpa_variant"], op.attrs["chunk_start_idx"], op.attrs["page_block_size"]) == ("chunked", 4096, 32)
    assert op.perf_stats["sdpa_regime"] == "chunked"
    assert op.perf_stats["sdpa_config"]["chunk_start_idx"] == 4096 and op.perf_stats["sdpa_config"]["paged"]
    assert len(op.inList) == 4


@pytest.mark.unit
def test_shim_chunked_prefill_kv_length_is_prefix_plus_chunk():
    # Llama 8B chunked prefill over a paged prefix: K/V are [blocks, nkv, 32, 128], so the KV length is
    # chunk_start_idx + S, not the block size; start 0 must price like the plain causal op, not like cross.
    import ttsim.front.ttnn as ttnn
    from ttsim.front.ttnn.tensor import DataType, Layout
    dev = _shim_dev()
    pc, ck = _llama_prefill_cfgs()

    def chunked(start):
        q = _tt(dev, [1, 32, 2048, 128], "cq", DataType.BFLOAT8_B)
        k = _tt(dev, [192, 8, 32, 128], "ck", DataType.BFLOAT8_B)
        v = _tt(dev, [192, 8, 32, 128], "cv", DataType.BFLOAT8_B)
        pt = _tt(dev, [1, 192], "cpt", DataType.INT32, Layout.ROW_MAJOR_LAYOUT)
        ttnn.transformer.chunked_scaled_dot_product_attention(
            q, k, v, pt, chunk_start_idx=start, program_config=pc, compute_kernel_config=ck,
            paged_cache_geometry=ttnn.PagedCacheGeometryOverride(block_size=32, num_kv_heads=8))
        return _last_sdpa(dev).perf_stats

    ttnn.transformer.scaled_dot_product_attention(*_qkv(dev, S=2048), is_causal=True, program_config=pc,
                                                  compute_kernel_config=ck)
    plain = _last_sdpa(dev).perf_stats
    c0, c4k = chunked(0), chunked(4096)
    for ps in (c0, c4k):
        assert ps["sdpa_regime"] == ps["sdpa_wall_regime"] == "chunked"
    assert (c0["sdpa_config"]["kv_seq"], c4k["sdpa_config"]["kv_seq"]) == (2048, 6144)
    # start 4096: same bytes and wall as the direct config (22.3 MB before the 1.15 paged gather
    # derate on K/V, 24.3 MB with it; the shim used to read 9.0 MB from the block size).
    direct = predict(SdpaConfig(S=2048, kv_seq=6144, chunk_start_idx=4096, q_chunk=256, k_chunk=256,
                                num_heads=32, num_kv_heads=8, num_cores=64, fidelity="HiFi4",
                                exp_approx_mode=False, fp32_dest_acc=True, accum_dtype="float32",
                                dram_scatter_derate=1.15, arch=ARCH_BH))
    assert c4k["inBytes"] == direct.dram_in_bytes > 22e6
    assert c4k["fused_compute_cycles"] == direct.wall_clock_cycles
    # start 0: the plain causal op plus the chunked front-end constant and the K/V gather derate.
    assert 1.0 <= c0["fused_compute_cycles"] / plain["fused_compute_cycles"] <= 1.15
    assert plain["inBytes"] < c0["inBytes"] <= 1.15 * plain["inBytes"]


@pytest.mark.unit
def test_chunked_config_routes_on_the_variant_and_bounds_an_unknown_start():
    q, kv = [1, 32, 2048, 128], [192, 8, 32, 128]
    attrs = {"sdpa_variant": "chunked", "paged": True, "element_size": 1, "q_chunk_size": 128,
             "k_chunk_size": 128}
    first = sdpa_config_from_shapes(q, kv, kv, {**attrs, "chunk_start_idx": 0})
    assert first.is_chunked and first.chunk_start_idx == 0 and first.kv_seq == 0     # attends S tokens
    r = predict(first)
    assert (r.regime, r.wall_regime) == ("chunked", "chunked") and "regime_chunked" in r.low_confidence_reasons
    assert r.k_eff == predict(_bh(S=2048, **_GQA)).k_eff
    # A tensor-form start has no host value: the page table capacity bounds the prefix, flagged.
    unknown = sdpa_config_from_shapes(q, kv, kv, {**attrs, "chunk_start_unknown": True},
                                      page_table_shape=[1, 192])
    assert (unknown.chunk_start_idx, unknown.kv_seq) == (4096, 6144)
    assert "chunk_start_unknown" in predict(unknown).low_confidence_reasons


@pytest.mark.unit
def test_shim_chunk_start_idx_tensor_flags_unknown():
    import ttsim.front.ttnn as ttnn
    from ttsim.front.ttnn.tensor import DataType, Layout
    dev = _shim_dev()
    q, k, v = _qkv(dev, S=2048)
    pt = _tt(dev, [1, 64], "pt", DataType.INT32, Layout.ROW_MAJOR_LAYOUT)
    cs = _tt(dev, [1], "cs", DataType.INT32, Layout.ROW_MAJOR_LAYOUT)
    ttnn.transformer.chunked_scaled_dot_product_attention(q, k, v, pt, chunk_start_idx_tensor=cs)
    op = _last_sdpa(dev)
    assert op.attrs["chunk_start_unknown"] is True and "chunk_start_idx" not in op.attrs
    assert "chunk_start_unknown" in op.perf_stats["sdpa_low_confidence_reasons"].split(",")


@pytest.mark.unit
def test_shim_sparse_defaults_to_the_kernel_chunking_and_keeps_the_anchor():
    # sparse_sdpa takes k_chunk_size (default 128) and has no q chunk parameter, so the 32x32 prefill
    # fallback must not reprice it: the calibrated shape stays on its R1a wall.
    import ttsim.front.ttnn as ttnn
    from ttsim.front.ttnn.tensor import DataType, Layout
    dev = _shim_dev()
    q = _tt(dev, [1, 32, 2048, 576], "sq")
    kv = _tt(dev, [1, 1, 8192, 576], "skv")
    idx = _tt(dev, [1, 1, 2048, 1024], "sidx", DataType.UINT32, Layout.ROW_MAJOR_LAYOUT)
    ttnn.transformer.sparse_sdpa(q, kv, idx, 512)
    op = _last_sdpa(dev)
    assert (op.attrs["q_chunk_size"], op.attrs["k_chunk_size"]) == (128, 128)
    assert {"q_chunk_size", "k_chunk_size"} <= set(op.attrs["sdpa_defaulted"].split(","))
    sparse_wall = predict(_bh(S=2048, kv_seq=1024, num_heads=32, is_sparse=True, is_causal=False, **_MLA)).wall_clock_cycles
    assert op.perf_stats["fused_compute_cycles"] == sparse_wall      # the R1a TOPK 1024 wall
    assert abs(sparse_wall - 8_603_386) / 8_603_386 <= 0.01
    assert "program_config_absent" not in op.perf_stats["sdpa_low_confidence_reasons"].split(",")
    ttnn.transformer.sparse_sdpa(q, kv, idx, 512, k_chunk_size=256)
    op = _last_sdpa(dev)
    assert op.attrs["k_chunk_size"] == 256 and "k_chunk_size" not in op.attrs["sdpa_defaulted"].split(",")
    # a gather-bound token wall does not move with the k chunk; the fit family says so
    assert "sparse_fit_family" in op.perf_stats["sdpa_low_confidence_reasons"].split(",")
    assert op.perf_stats["sdpa_cycle_breakdown"]["compute_floor"] != predict(_bh(S=2048, kv_seq=1024, num_heads=32, is_sparse=True, is_causal=False, **_MLA)).compute_latency_cycles


@pytest.mark.unit
def test_shim_decode_records_is_causal_mask_and_sink():
    import ttsim.front.ttnn as ttnn
    dev = _shim_dev()
    q, kc, vc, pt, cp = _decode_tensors(dev)
    kc2 = _tt(dev, [32, 8, 4096, 128], "k2"); vc2 = _tt(dev, [32, 8, 4096, 128], "v2")
    mask = _tt(dev, [32, 1, 32, 4096], "mask")
    sink = _tt(dev, [1, 32, 1, 1], "sink")
    ttnn.transformer.scaled_dot_product_attention_decode(q, kc2, vc2, is_causal=False, attn_mask=mask,
                                                         cur_pos_tensor=cp, attention_sink=sink)
    op = _last_sdpa(dev)
    assert (op.attrs["is_causal"], op.attrs["has_attn_mask"], op.attrs["attention_sink"]) == (False, True, True)
    assert op.perf_stats["sdpa_config"]["is_causal"] is False
    assert "decode_mask_unmodelled" in op.perf_stats["sdpa_low_confidence_reasons"].split(",")


@pytest.mark.unit
def test_shim_decode_cur_pos_list_sets_attended_length():
    import ttsim.front.ttnn as ttnn
    dev = _shim_dev()
    q, kc, vc, pt, cp = _decode_tensors(dev)
    ttnn.transformer.paged_scaled_dot_product_attention_decode(q, kc, vc, page_table_tensor=pt,
                                                               cur_pos=[1023] * 32)
    ps = _last_sdpa(dev).perf_stats
    assert ps["sdpa_config"]["attended"] == 1024 and ps["sdpa_config"]["cur_pos"] == 1023
    assert "cur_pos_unknown" not in ps["sdpa_low_confidence_reasons"]
    ref = predict_decode(cache_len=1024, num_q_heads=32, num_kv_heads=8, head_dim=128, batch=32,
                         k_chunk=0, input_dtype="bfloat16", kv_input_dtype="bfp8_b")
    assert ps["inBytes"] == ref.dram_in_bytes


@pytest.mark.unit
def test_shim_decode_cur_pos_tensor_flags_unknown_position():
    import ttsim.front.ttnn as ttnn
    dev = _shim_dev()
    q, kc, vc, pt, cp = _decode_tensors(dev, blocks=128)     # 128 blocks shared by 32 users
    ttnn.transformer.paged_scaled_dot_product_attention_decode(q, kc, vc, page_table_tensor=pt, cur_pos_tensor=cp)
    op = _last_sdpa(dev)
    assert op.attrs["cur_pos_unknown"] is True
    echo = op.perf_stats["sdpa_config"]
    assert echo["attended"] == echo["cache_len"] == 4 * 32     # physical blocks per user x block size
    assert "cur_pos_unknown" in op.perf_stats["sdpa_low_confidence_reasons"].split(",")


@pytest.mark.unit
def test_shim_prefill_chunk_start_zero_is_the_first_chunk():
    import ttsim.front.ttnn as ttnn
    dev = _shim_dev()
    ttnn.transformer.scaled_dot_product_attention(*_qkv(dev), chunk_start_idx=0)
    op = _last_sdpa(dev)
    assert (op.attrs["sdpa_variant"], op.attrs["chunk_start_idx"]) == ("chunked", 0)
    assert op.perf_stats["sdpa_regime"] == "chunked" and op.perf_stats["sdpa_config"]["chunk_start_idx"] == 0


@pytest.mark.unit
def test_shim_optional_tensors_stay_inputs_of_the_op():
    # mask, sink and window tensors are recorded as inputs, not only as flags, so their producers keep an
    # edge to the SDPA op; the paged decode still finds its page table as the last input.
    import ttsim.front.ttnn as ttnn
    dev = _shim_dev()
    q, k, v = _qkv(dev)
    mask = _tt(dev, [1, 1, 4096, 4096], "pmask")
    sink = _tt(dev, [1, 32, 1, 1], "psink")
    ttnn.transformer.scaled_dot_product_attention(q, k, v, attn_mask=mask, attention_sink=sink)
    op = _last_sdpa(dev)
    assert op.inList[-2:] == [mask.name, sink.name] and op.attrs["has_attn_mask"] and op.attrs["attention_sink"]
    dq, kc, vc, pt, cp = _decode_tensors(dev)
    dmask = _tt(dev, [32, 1, 32, 1024], "dmask")
    ttnn.transformer.paged_scaled_dot_product_attention_decode(dq, kc, vc, page_table_tensor=pt,
                                                               cur_pos=[1023] * 32, is_causal=False,
                                                               attn_mask=dmask)
    op = _last_sdpa(dev)
    assert op.inList == [dq.name, kc.name, vc.name, dmask.name, pt.name]
    assert op.perf_stats["sdpa_config"]["paged"] and op.perf_stats["sdpa_config"]["cache_len"] == 1024


@pytest.mark.unit
def test_shim_paged_decode_unknown_position_streams_at_most_the_physical_cache():
    # Shipped dual-mode decode shape: 32 users, cache [1024, 8, 32, 128], page table [32, 1024]. The table
    # width is a bound, not the 32 blocks the cache holds per user; pricing it streamed 2290 MB from 71 MB.
    import ttsim.front.ttnn as ttnn
    dev = _shim_dev()
    q, kc, vc, pt, cp = _decode_tensors(dev)
    ttnn.transformer.paged_scaled_dot_product_attention_decode(q, kc, vc, page_table_tensor=pt, cur_pos_tensor=cp)
    ps = _last_sdpa(dev).perf_stats
    assert ps["sdpa_config"]["cache_len"] == ps["sdpa_config"]["attended"] == 1024
    ref = predict_decode(cache_len=1024, num_q_heads=32, num_kv_heads=8, head_dim=128, batch=32,
                         k_chunk=0, input_dtype="bfloat16", kv_input_dtype="bfp8_b")
    assert ps["inBytes"] == ref.dram_in_bytes < 100e6            # K/V cache plus the 32 query tokens
    assert "cur_pos_unknown" in ps["sdpa_low_confidence_reasons"].split(",")


@pytest.mark.unit
def test_shim_l1_sharded_query_is_not_dram_traffic():
    # tt_transformers hands decode a height sharded query in L1 (T2.5 ops CSV INPUT_0_MEMORY); only a
    # DRAM query is counted in the streamed bytes.
    import ttsim.front.ttnn as ttnn
    from ttsim.front.ttnn.buffer import BufferType, TensorMemoryLayout
    from ttsim.front.ttnn.memory import MemoryConfig
    dev = _shim_dev()
    q, kc, vc, pt, cp = _decode_tensors(dev)
    ttnn.transformer.paged_scaled_dot_product_attention_decode(q, kc, vc, page_table_tensor=pt, cur_pos=[1023] * 32)
    dram = _last_sdpa(dev)
    q._memory_config = MemoryConfig(TensorMemoryLayout.HEIGHT_SHARDED, BufferType.L1)
    ttnn.transformer.paged_scaled_dot_product_attention_decode(q, kc, vc, page_table_tensor=pt, cur_pos=[1023] * 32)
    l1 = _last_sdpa(dev)
    assert "q_in_l1" not in dram.attrs and l1.attrs["q_in_l1"] is True
    assert dram.perf_stats["sdpa_config"]["q_in_dram"] is True and l1.perf_stats["sdpa_config"]["q_in_dram"] is False
    assert 0 < dram.perf_stats["inBytes"] - l1.perf_stats["inBytes"] < 0.15 * dram.perf_stats["inBytes"]


@pytest.mark.unit
def test_shim_paged_decode_records_geometry():
    import ttsim.front.ttnn as ttnn
    dev = _shim_dev()
    q, kc, vc, pt, cp = _decode_tensors(dev, blocks=64, block=128)   # 2 blocks per user
    ttnn.transformer.paged_scaled_dot_product_attention_decode(
        q, kc, vc, page_table_tensor=pt, cur_pos_tensor=cp,
        paged_cache_geometry=ttnn.PagedCacheGeometryOverride(block_size=128, num_kv_heads=8))
    op = _last_sdpa(dev)
    assert op.attrs["paged"] is True and op.attrs["page_block_size"] == 128
    echo = op.perf_stats["sdpa_config"]
    assert (echo["paged"], echo["page_block_size"], echo["cache_len"]) == (True, 128, 2 * 128)
    assert "decode_page_block_uncalibrated" in op.perf_stats["sdpa_low_confidence_reasons"].split(",")


@pytest.mark.unit
def test_shim_decode_max_cores_per_head_batch_forwarded():
    import ttsim.front.ttnn as ttnn
    dev = _shim_dev()
    q, kc, vc, pt, cp = _decode_tensors(dev, batch=1)
    for mcphb, expect in ((16, 8), (2, 2)):    # 64 cores / 8 kv heads = 8, capped by the config
        pc = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=(8, 8), q_chunk_size=0, k_chunk_size=0,
                                    max_cores_per_head_batch=mcphb)
        ttnn.transformer.paged_scaled_dot_product_attention_decode(q, kc, vc, page_table_tensor=pt,
                                                                   cur_pos_tensor=cp, program_config=pc)
        op = _last_sdpa(dev)
        assert op.attrs["max_cores_per_head_batch"] == mcphb
        assert op.perf_stats["sdpa_config"]["cores_per_head"] == expect


@pytest.mark.unit
def test_sdpa_config_from_shapes_honors_num_cores_attr():
    q = [1, 32, 4096, 128]
    cfg = sdpa_config_from_shapes(q, [1, 8, 4096, 128], [1, 8, 4096, 128],
                                  {"is_causal": False, "element_size": 1, "num_cores": 64,
                                   "q_chunk_size": 128, "k_chunk_size": 128})
    assert cfg.num_cores == 64 and "grid_absent" not in cfg.fallback_reasons
    r = predict(cfg)
    assert r.q_chunks_wall_core == math.ceil(32 * 32 / 64) == 16
    assert "off_calibration_cores" in r.low_confidence_reasons


@pytest.mark.unit
def test_fidelity_hifi4_doubles_fpu_matmul_cycles():
    lo = predict(_bh(S=4096, fidelity="HiFi2", **_GQA))
    hi = predict(_bh(S=4096, fidelity="HiFi4", **_GQA))
    assert hi.fpu_matmul_cycles == pytest.approx(2 * lo.fpu_matmul_cycles, abs=1)
    assert "off_calibration_fidelity" in hi.low_confidence_reasons


@pytest.mark.unit
def test_fp32_dest_acc_switches_accum_bytes_and_flags_path():
    q = [1, 32, 4096, 128]
    base = {"is_causal": True, "element_size": 1, "q_chunk_size": 128, "k_chunk_size": 128, "num_cores": 110,
            "exp_approx_mode": True, "fidelity": "HiFi2", "kv_element_size": 1}
    bf = predict(sdpa_config_from_shapes(q, q, q, {**base, "fp32_dest_acc_en": False}))
    fp = predict(sdpa_config_from_shapes(q, q, q, {**base, "fp32_dest_acc_en": True}))
    assert fp.pack_bytes_total == pytest.approx(2 * bf.pack_bytes_total, abs=1)
    assert "legacy_path_production_family" in fp.low_confidence_reasons
    assert not any(x.startswith("legacy") or x.startswith("fp32") for x in bf.low_confidence_reasons)
    # fp32 DEST selects the legacy kernel: exp on MATH, so the union is the plain FPU + SFPU sum
    assert fp.math_active_cycles == fp.fpu_cycles + fp.sfpu_cycles > bf.math_active_cycles


@pytest.mark.unit
def test_kernel_default_chunks_are_32_not_128_when_program_config_absent():
    q = [1, 32, 4096, 128]
    cfg = sdpa_config_from_shapes(q, q, q, {"element_size": 1})
    assert (cfg.q_chunk, cfg.k_chunk) == (32, 32)
    assert "program_config_absent" in cfg.fallback_reasons
    given = sdpa_config_from_shapes(q, q, q, {"element_size": 1, "q_chunk_size": 256, "k_chunk_size": 512})
    assert (given.q_chunk, given.k_chunk) == (256, 512) and "program_config_absent" not in given.fallback_reasons


@pytest.mark.unit
def test_decode_k_chunk_auto_matches_get_chunk_size():
    from ttsim.perf.roofline_sdpa import decode_chunk_size
    # largest power of two dividing the length, capped at 512
    assert decode_chunk_size(4096) == 512 and decode_chunk_size(4608) == 512
    assert decode_chunk_size(1056) == 32 and decode_chunk_size(1088) == 64 and decode_chunk_size(96) == 32
    r = predict_decode(cache_len=4608, num_q_heads=32, num_kv_heads=8, head_dim=128, k_chunk=0)
    assert r.config_echo["k_chunk"] == 512 and "k_chunk_auto" in r.low_confidence_reasons
    given = predict_decode(cache_len=4608, num_q_heads=32, num_kv_heads=8, head_dim=128, k_chunk=128)
    assert given.config_echo["k_chunk"] == 128 and "k_chunk_auto" not in given.low_confidence_reasons


@pytest.mark.unit
def test_low_confidence_reasons_are_listed_and_bool_derived():
    r = predict(_bh(S=4096, **_GQA))
    assert r.low_confidence_reasons == [] and r.low_confidence is False
    q = [1, 32, 4096, 128]
    full = {"is_causal": True, "element_size": 1, "kv_element_size": 1, "q_chunk_size": 128, "k_chunk_size": 128,
            "num_cores": 110, "exp_approx_mode": True, "fidelity": "HiFi2", "fp32_dest_acc_en": False}
    r2 = predict(sdpa_config_from_shapes(q, [1, 8, 4096, 128], [1, 8, 4096, 128], full))
    assert r2.low_confidence_reasons == [] and r2.low_confidence is False
    r3 = predict(sdpa_config_from_shapes(q, [1, 8, 4096, 128], [1, 8, 4096, 128], {**full, "num_cores": 80}))
    assert r3.low_confidence_reasons == ["off_calibration_cores"] and r3.low_confidence is True
    # 64 cores is a measured grid of the stream law (T2.3), so the causal wall there is not flagged
    r4 = predict(sdpa_config_from_shapes(q, [1, 8, 4096, 128], [1, 8, 4096, 128], {**full, "num_cores": 64}))
    assert r4.low_confidence_reasons == []


@pytest.mark.unit
def test_perf_stats_echoes_config_and_defaulted_attrs():
    q = [1, 32, 4096, 128]
    cfg = sdpa_config_from_shapes(q, [1, 8, 4096, 128], [1, 8, 4096, 128],
                                  {"element_size": 1, "sdpa_defaulted": "fidelity,fp32_dest_acc_en"})
    ps = sdpa_perf_stats(cfg)
    assert ps["sdpa_defaulted_attrs"] == "fidelity,fp32_dest_acc_en"
    assert ps["sdpa_config"]["q_chunk"] == 32 and ps["sdpa_config"]["num_kv_heads"] == 8
    assert "program_config_absent" in ps["sdpa_low_confidence_reasons"].split(",")
    # the extra keys ride through the device cost path untouched
    device = Device(_MockSimConfig())
    op = _sdpa_op("echo", cfg)
    device.execute_op(op)
    assert op.compute_cycles == ps["fused_compute_cycles"]


@pytest.mark.unit
def test_production_llama8b_prefill_config_reaches_predict():
    import ttsim.front.ttnn as ttnn
    dev = _shim_dev()
    pc, ck = _llama_prefill_cfgs()
    ttnn.transformer.scaled_dot_product_attention(*_qkv(dev), is_causal=True, program_config=pc,
                                                  compute_kernel_config=ck)
    ps = _last_sdpa(dev).perf_stats
    echo = ps["sdpa_config"]
    assert echo == dict(S=4096, kv_seq=4096, batch=1, num_heads=32, num_kv_heads=8, head_dim=128, v_head_dim=128,
                        q_chunk=256, k_chunk=256, num_cores=64, fidelity="HiFi4", exp_approx_mode=False,
                        fp32_dest_acc=True, input_dtype="bfp8_b", kv_input_dtype="bfp8_b", is_causal=True,
                        has_attn_mask=False, sliding_window=0, attention_sink=False, chunk_start_idx=0,
                        is_chunked=False, paged=False, page_block_size=0, is_sparse=False, is_joint=False)
    reasons = set(ps["sdpa_low_confidence_reasons"].split(","))
    assert "legacy_path_production_family" in reasons and not any(x.endswith("_absent") for x in reasons)
    assert ps["sdpa_defaulted_attrs"] == ""


@pytest.mark.unit
def test_production_anchor_llama8b_prefill_s1024_355us():
    # Published 355.0 us (tracy ops CSV, Llama 3.1 8B report); T2.4 reproduces the op at 347.0 us.
    import ttsim.front.ttnn as ttnn
    dev = _shim_dev()
    pc, ck = _llama_prefill_cfgs(q=64, k=64)
    ttnn.transformer.scaled_dot_product_attention(*_qkv(dev, S=1024), is_causal=True, program_config=pc,
                                                  compute_kernel_config=ck)
    us = _last_sdpa(dev).perf_stats["fused_compute_cycles"] / 1350.0
    assert abs(us - 355.0) / 355.0 <= 0.10, f"{us:.0f} us"
    assert abs(us - 347.0) / 347.0 <= 0.03, f"{us:.0f} us"


@pytest.mark.unit
@pytest.mark.parametrize("pos,meas_us", [(1024, 284.0), (1023, 254.4)])
def test_production_anchor_llama8b_decode(pos, meas_us):
    # 32 users, 64 cores, HiFi2, fp32 acc, bf8 KV under a bf16 query through the shim: T2.8 measured 284.0 us
    # at position 1024 (1152 rows in whole 128-row chunks), the Confluence 254.4 us at 1023 (exactly 1024 rows).
    import ttsim.front.ttnn as ttnn
    dev = _shim_dev()
    q, kc, vc, pt, cp = _decode_tensors(dev, blocks=16384)
    pc = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=(8, 8), q_chunk_size=0, k_chunk_size=0,
                                exp_approx_mode=False, max_cores_per_head_batch=16)
    ck = ttnn.BlackholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=True,
                                           fp32_dest_acc_en=True, packer_l1_acc=True)
    ttnn.transformer.paged_scaled_dot_product_attention_decode(
        q, kc, vc, page_table_tensor=pt, cur_pos=[pos] * 32, program_config=pc, compute_kernel_config=ck,
        paged_cache_geometry=ttnn.PagedCacheGeometryOverride(block_size=32, num_kv_heads=8))
    us = _last_sdpa(dev).perf_stats["fused_compute_cycles"] / 1350.0
    assert abs(us - meas_us) / meas_us <= 0.03, f"{us:.0f} us"


@pytest.mark.unit
def test_production_decode_config_reaches_predict_decode():
    import ttsim.front.ttnn as ttnn
    dev = _shim_dev()
    q, kc, vc, pt, cp = _decode_tensors(dev)
    pc = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=(8, 8), q_chunk_size=0, k_chunk_size=0,
                                exp_approx_mode=False, max_cores_per_head_batch=16)
    ck = ttnn.BlackholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=True,
                                           fp32_dest_acc_en=True, packer_l1_acc=True)
    ttnn.transformer.paged_scaled_dot_product_attention_decode(
        q, kc, vc, page_table_tensor=pt, cur_pos=[1023] * 32, program_config=pc, compute_kernel_config=ck)
    ps = _last_sdpa(dev).perf_stats
    echo = ps["sdpa_config"]
    assert (echo["batch"], echo["num_heads"], echo["num_kv_heads"], echo["attended"]) == (32, 32, 8, 1024)
    assert (echo["num_cores"], echo["fidelity"], echo["kv_input_dtype"], echo["input_dtype"]) == (64, "HiFi2", "bfp8_b", "bfloat16")
    assert (echo["paged"], echo["page_block_size"], echo["k_chunk"]) == (True, 32, 128)   # dynamic rule, fp32 DEST
    assert ps["sdpa_wall_regime"] == "decode" and ps["sdpa_defaulted_attrs"] == ""


@pytest.mark.unit
def test_arch_constants_are_the_campaign_fits():
    # The floor constants are the T2.1 counter fits; the regime, legacy and decode constants the R1a, T2.4,
    # T2.8 and R1b wall fits. Nothing else may be read in silently.
    a = ArchConfig()
    assert (a.cycles_per_tile_mac["HiFi2"], a.cycles_per_tile_mac["HiFi4"]) == (32.0, 64.0)
    assert (a.fpu_overhead_frac, a.fpu_overhead_tile_macs_per_qtile_step, a.fpu_overhead_cycles_per_qtile_step) == (0.0489, 0.957, 74.3)
    assert (a.legacy_step_cycles, a.dest_roundtrip_cycles) == (2406.3, 835.6)
    # head_dim term (R1c head_dim 64 pair): the two parts of the stream fixed cost and of the PACK per-q-k-tile
    # cost sum back to 691 and 348 / 319.5 at head_dim 128 (8 K and V tiles per k tile).
    assert (a.kv_stream_fixed_per_ktile, a.kv_stream_fixed_per_kv_tile) == (405.8, 35.65)
    assert a.kv_stream_fixed_per_ktile + 8 * a.kv_stream_fixed_per_kv_tile == pytest.approx(691.0)
    for reg, pk in (("prefill_causal", 348.0), ("prefill_noncausal", 319.5), ("cross", 319.5), ("masked", 319.5)):
        t = WALL_TERMS_BH[reg]
        assert t.pack_per_qk_dtile == 26.45 and t.pack_per_qktile + 8 * t.pack_per_qk_dtile == pytest.approx(pk)
    assert (a.decode_kv_stream_gbps_nonpaged, a.decode_kv_stream_gbps_mla) == (342.3, 342.1)
    assert a.decode_fixed_overhead_cycles_mla == 25800.0
    assert a.decode_fixed_cycles_mla_paged_by_cores_per_group == {
        1: 63600.0, 2: 62900.0, 3: 55700.0, 4: 60500.0, 6: 79700.0, 8: 72400.0, 13: 78700.0, 16: 69200.0}
    assert WALL_TERMS_BH["masked"].mask_per_tile == 200.2 and WALL_TERMS_BH["joint"].fe_per_tile_mac == 161.5
    assert (WALL_TERMS_BH["sparse"].fe_per_token, WALL_TERMS_BH["sparse"].gather_rate_bpc) == (56689.0, 2.979)
