# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Device.get_exec_stats operator perf LUT path (stubbed lookup)."""

from __future__ import annotations

import math
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest

from ttsim.back.device import Device
from tools.perf_lookup.lookup_operator_perf import (
    MasterPerfStats,
    OperatorPerfLUTValidationError,
)


class MockIPGroup:
    def __init__(self, iptype):
        self.iptype = iptype


class MockSimConfigWithPeakIpc:
    """Minimal sim config for Device.get_exec_stats (matches test_device_disable_fusion)."""

    def __init__(self, freq_mhz=1000, ramp_penalty=10):
        self._freq_mhz = freq_mhz
        self._ramp_penalty = ramp_penalty
        self._mem_size_gb = 32.0
        self.devname = "TestArch"
        self.name = "test_device"
        self.ipgroups = [MockIPGroup("compute"), MockIPGroup("memory")]

    def frequency(self, pipe, units="MHz"):
        return self._freq_mhz

    def mem_frequency(self, units="MHz"):
        return self._freq_mhz

    def ramp_penalty(self):
        return self._ramp_penalty

    def mem_size(self, units="GB"):
        return self._mem_size_gb

    def peak_bandwidth(self, freq_units="GHz"):
        return 1000.0

    def peak_bandwidth_per_cycle(self):
        return 10.0

    def peak_flops(self, pipe, instr, precision, mul_factor=1):
        return 100.0

    def peak_ipc(self, pipe, instr, precision):
        return 128.0


def _create_stub_op(name, optype, compute_pipe="matrix"):
    # Plain namespace: unittest.Mock would make hasattr(op, "ideal_cycles") true (child Mock)
    # and break op_stat_iter in get_exec_stats.
    return SimpleNamespace(
        name=name,
        optype=optype,
        removed_in_optimization=False,
        fused_in_optimization=False,
        fused_with_op=None,
        uses_compute_pipe=compute_pipe,
        compute_cycles=1000,
        mem_rd_cycles=500,
        mem_wr_cycles=500,
        mem_rd_cycles_fractional=500.0,
        mem_wr_cycles_fractional=500.0,
        repeat_count=1,
        precision="fp32",
        fused_op_cycles=None,
        exec_stats={},
        perf_stats={
            "inElems": 1000,
            "outElems": 1000,
            "inBytes": 4000,
            "outBytes": 4000,
            "instrs": {"mac": 1000},
            "inParamCount": 500,
            "inActCount": 500,
            "outActCount": 1000,
        },
    )


def _make_wlgraph(op):
    g = MagicMock()
    g.get_ordered_nodes.return_value = ["op1"]

    def get_op(name):
        assert name == "op1"
        return op

    g.get_op = get_op
    g.is_input_node = MagicMock(return_value=False)
    g.is_output_node = MagicMock(return_value=False)
    return g


def _lut_recomputed_cycles(msecs_lut: float, dev_freq_mhz: float, guardband: float) -> tuple[int, int, float]:
    """Mirror device.py LUT branch (uses_perf_lookup)."""
    cycles = int(math.ceil(msecs_lut * dev_freq_mhz * 1e3))
    ideal_cycles = int(math.ceil(cycles / (1 + guardband)))
    ideal_msecs = ideal_cycles / dev_freq_mhz / 1e3
    return cycles, ideal_cycles, ideal_msecs


@pytest.mark.unit
def test_get_exec_stats_lut_msecs_cycles_mem_util_and_flag():
    device = Device(MockSimConfigWithPeakIpc(freq_mhz=1000, ramp_penalty=10))
    op = _create_stub_op("op1", "MatMul", compute_pipe="matrix")
    graph = _make_wlgraph(op)

    msecs_lut = 2.0
    master = MasterPerfStats(
        msecs=msecs_lut,
        matrix_pipe_util=50.0,
        vector_pipe_util=40.0,
        memory_traffic=12_345.0,
        mem_util=30.0,
    )
    perf_map = Mock()
    perf_map.lookup = Mock(return_value=master)
    device.operator_perf_map = perf_map

    device.get_exec_stats(graph, bs=1)

    perf_map.lookup.assert_called_once()
    call_args = perf_map.lookup.call_args[0]
    assert call_args[0] is op
    assert call_args[1] is graph
    assert isinstance(call_args[2], int)

    st = op.exec_stats
    assert st["uses_perf_lookup"] is True
    assert st["msecs"] == pytest.approx(msecs_lut)
    assert st["mem_rd_util"] == 0.0
    assert st["mem_wr_util"] == 0.0

    exp_cycles, exp_ideal_cycles, exp_ideal_msecs = _lut_recomputed_cycles(
        msecs_lut, 1000.0, Device.G_GUARDBAND
    )
    assert st["cycles"] == float(exp_cycles)
    assert st["ideal_cycles"] == float(exp_ideal_cycles)
    assert st["ideal_msecs"] == pytest.approx(exp_ideal_msecs)

    assert st["matrix_pipe_util"] == pytest.approx(0.5)
    assert st["vector_pipe_util"] == pytest.approx(0.4)
    assert st["memory_traffic"] == pytest.approx(12_345.0)
    assert st["mem_util"] == pytest.approx(0.3)


@pytest.mark.unit
def test_get_exec_stats_lut_miss_keeps_analytical_mem_util():
    device = Device(MockSimConfigWithPeakIpc(freq_mhz=1000, ramp_penalty=10))
    op = _create_stub_op("op1", "MatMul", compute_pipe="matrix")
    graph = _make_wlgraph(op)

    perf_map = Mock()
    perf_map.lookup = Mock(return_value=None)
    device.operator_perf_map = perf_map

    device.get_exec_stats(graph, bs=1)

    st = op.exec_stats
    assert st["uses_perf_lookup"] is False
    # Analytical path: mem-bound tie at 1000 cycles, ideal_cycles = ceil(1000 + 10) = 1010
    ideal_cycles = 1010
    assert st["ideal_cycles"] == float(ideal_cycles)
    assert st["msecs"] != 2.0  # not the LUT value from the other test
    exp_mem_rd = 500 / ideal_cycles * Device.DG_MEMORY_UTIL_CONSTANT
    exp_mem_wr = 500 / ideal_cycles * Device.DG_MEMORY_UTIL_CONSTANT
    assert st["mem_rd_util"] == pytest.approx(exp_mem_rd)
    assert st["mem_wr_util"] == pytest.approx(exp_mem_wr)


@pytest.mark.unit
def test_opstats_row_merge_includes_uses_perf_lookup():
    """Same pattern as ttsim.stats.hlmstats.HLMStats.dump_stats: val.update(op.exec_stats)."""
    device = Device(MockSimConfigWithPeakIpc())
    op = _create_stub_op("op1", "MatMul")
    graph = _make_wlgraph(op)

    master = MasterPerfStats(
        msecs=1.5,
        matrix_pipe_util=10.0,
        vector_pipe_util=20.0,
        memory_traffic=None,
        mem_util=None,
    )
    device.operator_perf_map = Mock(lookup=Mock(return_value=master))

    device.get_exec_stats(graph, bs=1)

    val = {"opname": "op1", "archname": "x"}
    val.update(op.exec_stats)
    assert val["uses_perf_lookup"] is True
    assert val["msecs"] == pytest.approx(1.5)


@pytest.mark.unit
def test_get_exec_stats_lut_validation_error_propagates():
    device = Device(MockSimConfigWithPeakIpc())
    op = _create_stub_op("op1", "MatMul")
    graph = _make_wlgraph(op)

    device.operator_perf_map = Mock(
        lookup=Mock(side_effect=OperatorPerfLUTValidationError("bad row"))
    )

    with pytest.raises(OperatorPerfLUTValidationError, match="bad row"):
        device.get_exec_stats(graph, bs=1)


# --- ignore_lut gate in Device.__init__ -------------------------------------------------
# A run with ignore_lut set must never load the LUT: if the gate is reordered, measured times
# would silently be reported as analytical estimates, with no error.

_IGNORE_LUT_UNSET = object()


def _lookup_cfg(lookup_file, ignore_lut=_IGNORE_LUT_UNSET):
    cfg = MockSimConfigWithPeakIpc()
    cfg.operator_lookup_file = lookup_file
    if ignore_lut is not _IGNORE_LUT_UNSET:
        cfg.ignore_lut = ignore_lut
    return cfg


@pytest.mark.unit
def test_ignore_lut_true_does_not_resolve_lfc_path(monkeypatch):
    resolve = Mock(return_value=None)
    monkeypatch.setattr("ttsim.back.device.resolve_lfc_path", resolve)
    device = Device(_lookup_cfg("lfc://hlm-lut/some_lut.yaml", ignore_lut=True))
    assert resolve.call_count == 0
    assert device.operator_perf_map is None


@pytest.mark.unit
@pytest.mark.parametrize("ignore_lut", [False, _IGNORE_LUT_UNSET], ids=["false", "attribute_missing"])
def test_ignore_lut_false_or_missing_still_resolves_lfc_path(monkeypatch, ignore_lut):
    resolve = Mock(return_value=None)
    monkeypatch.setattr("ttsim.back.device.resolve_lfc_path", resolve)
    Device(_lookup_cfg("lfc://hlm-lut/some_lut.yaml", ignore_lut=ignore_lut))
    assert resolve.call_count == 1
    resolve.assert_called_once_with("lfc://hlm-lut/some_lut.yaml")


@pytest.mark.unit
@pytest.mark.parametrize(
    "ignore_lut, expect_loaded",
    [(True, False), (False, True), (_IGNORE_LUT_UNSET, True)],
    ids=["true", "false", "attribute_missing"],
)
def test_ignore_lut_gates_local_lookup_file(monkeypatch, tmp_path, ignore_lut, expect_loaded):
    """Same gate for a plain (non-lfc) path: the LUT file is opened only when ignore_lut is not set."""
    (tmp_path / "lut.yaml").write_text("{}\n")
    monkeypatch.chdir(tmp_path)
    sentinel = object()
    perf_map_cls = Mock(return_value=sentinel)
    monkeypatch.setattr("tools.perf_lookup.lookup_operator_perf.OperatorPerfMap", perf_map_cls)

    device = Device(_lookup_cfg("lut.yaml", ignore_lut=ignore_lut))

    if expect_loaded:
        assert perf_map_cls.call_count == 1
        assert device.operator_perf_map is sentinel
    else:
        assert perf_map_cls.call_count == 0
        assert device.operator_perf_map is None


@pytest.mark.unit
def test_ignore_lut_true_logs_lookup_disabled(monkeypatch):
    # Patch the module's logger rather than adding a loguru sink: other tests call
    # logger.disable("ttsim") (e.g. test_sigmoid.py), which silences ttsim.* sinks process-wide.
    fake_logger = Mock()
    monkeypatch.setattr("ttsim.back.device.logger", fake_logger)
    Device(_lookup_cfg(None, ignore_lut=True))
    logged = [str(c.args[0]) for c in fake_logger.info.call_args_list if c.args]
    assert any("lookup disabled" in m for m in logged), logged


@pytest.mark.unit
@pytest.mark.parametrize(
    "coverage, memory_traffic, mem_util, exp_traffic, exp_mem_util",
    [
        ("time_only", None, None, None, None),
        ("full", 2048.0, 40.0, 2048.0, 0.4),
    ],
)
def test_get_exec_stats_lut_stats_coverage(coverage, memory_traffic, mem_util, exp_traffic, exp_mem_util):
    """A time-only hit reports memory_traffic / mem_util as None (not measured), never 0."""
    device = Device(MockSimConfigWithPeakIpc())
    op = _create_stub_op("op1", "MatMul")
    graph = _make_wlgraph(op)
    master = MasterPerfStats(
        msecs=1.5,
        matrix_pipe_util=10.0,
        vector_pipe_util=20.0,
        memory_traffic=memory_traffic,
        mem_util=mem_util,
        stats_coverage=coverage,
    )
    device.operator_perf_map = Mock(lookup=Mock(return_value=master))

    device.get_exec_stats(graph, bs=1)

    st = op.exec_stats
    assert st["lut_stats_coverage"] == coverage
    assert st["memory_traffic"] == (None if exp_traffic is None else pytest.approx(exp_traffic))
    assert st["mem_util"] == (None if exp_mem_util is None else pytest.approx(exp_mem_util))


@pytest.mark.unit
@pytest.mark.parametrize("with_lut", [True, False], ids=["lut_miss", "no_lut"])
def test_get_exec_stats_analytical_op_reports_memory_fields_not_measured(with_lut):
    """An analytical op (LUT miss, or no LUT) measures no memory traffic: None, never 0."""
    device = Device(MockSimConfigWithPeakIpc())
    op = _create_stub_op("op1", "MatMul")
    graph = _make_wlgraph(op)
    if with_lut:
        device.operator_perf_map = Mock(lookup=Mock(return_value=None))

    device.get_exec_stats(graph, bs=1)

    assert op.exec_stats["uses_perf_lookup"] is False
    assert op.exec_stats["lut_stats_coverage"] is None
    assert op.exec_stats["memory_traffic"] is None
    assert op.exec_stats["mem_util"] is None


@pytest.mark.unit
def test_opstats_models_accept_not_measured_memory_fields():
    from ttsim.config.validators import TTSimHLWlDevRunOperatorPerfStats, TTSimHLWlDevRunOpCSVPerfStats

    for model in (TTSimHLWlDevRunOperatorPerfStats, TTSimHLWlDevRunOpCSVPerfStats):
        for field in ("memory_traffic", "mem_util", "lut_stats_coverage"):
            assert type(None) in getattr(model.model_fields[field].annotation, "__args__", ()), (model, field)
