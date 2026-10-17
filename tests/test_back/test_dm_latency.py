#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""--enable_dm_latency: read cycles become max(bandwidth-limited, read latency).

Small transactions are latency-bound and cost more than bytes/bandwidth predicts;
large transactions stay bandwidth-bound and are unaffected. Uses the real Blackhole
p100a package (config/tt_bh.yaml), which carries the calibrated ``read_latency``
block, and drives single ops through ``Device.execute_op``.
"""

import pytest

from ttsim.back.device import Device
from ttsim.config import get_arspec_from_yaml
from ttsim.front.ttnn.buffer import BufferType, TensorMemoryLayout
from ttsim.front.ttnn.memory import MemoryConfig
from ttsim.graph.wl_graph import WorkloadGraph
from ttsim.ops.op import SimOp
from ttsim.ops.tensor import SimTensor

ARCH_YAML = 'config/tt_bh.yaml'
PACKAGE = 'p100a'

# 512x512 fp16 reads (512 KB) stream enough bytes to stay bandwidth-bound;
# 32x32 (2 KB) is a single tile per core and is dominated by fixed latency.
LARGE_DIM = 512
SMALL_DIM = 32


def _make_device(**kwargs) -> Device:
    _, packages = get_arspec_from_yaml(ARCH_YAML)
    return Device(packages[PACKAGE], **kwargs)


def _build_graph(dim: int) -> tuple[WorkloadGraph, SimOp]:
    graph = WorkloadGraph('g')
    in_t = SimTensor({'name': 'in1', 'shape': [dim, dim], 'dtype': 'float32'})
    out_t = SimTensor({'name': 'out1', 'shape': [dim, dim], 'dtype': 'float32'})
    in_t.op_in = ['op']
    out_t.op_out = ['op']
    graph.add_tensor(in_t)
    graph.add_tensor(out_t)
    op = SimOp({'name': 'op', 'optype': 'Add', 'inList': ['in1'], 'outList': ['out1']})
    op.uses_compute_pipe = 'matrix'
    op.precision = 'fp16'
    op.repeat_count = 1
    op.perf_stats = {
        'inBytes': dim * dim * 2, 'outBytes': dim * dim * 2, 'instrs': {'add': 1000},
        'inParamCount': 0, 'inActCount': 100, 'outActCount': 100,
    }
    graph.add_op(op)
    graph.construct_graph()
    return graph, op


def _run_single_op(device: Device, dim: int = LARGE_DIM) -> SimOp:
    _, op = _build_graph(dim)
    device.execute_op(op)
    return op


@pytest.mark.unit
def test_disabled_by_default():
    device = _make_device()
    assert not device.enable_dm_latency
    assert device.dm_read_cfg is None


@pytest.mark.unit
def test_enabled_device_loads_calibration():
    device = _make_device(enable_dm_latency=True)
    assert device.dm_read_cfg is not None
    assert device.dm_read_channels == 7
    # fclk defaults to the matrix compute clock when the YAML omits fclk_mhz.
    assert device.dm_fclk_MHz == device.freq_MHz


@pytest.mark.unit
def test_small_transaction_becomes_latency_bound():
    baseline = _run_single_op(_make_device(), dim=SMALL_DIM)
    latency = _run_single_op(_make_device(enable_dm_latency=True), dim=SMALL_DIM)
    # A single 2 KB read is nowhere near bandwidth-limited: the fixed DRAM + NoC
    # head dominates, so enabling the model must raise the read cost.
    assert latency.dm_read_latency_cycles > 0.0
    assert latency.mem_rd_cycles_fractional > baseline.mem_rd_cycles_fractional
    assert latency.mem_rd_cycles > baseline.mem_rd_cycles


@pytest.mark.unit
def test_large_transaction_stays_bandwidth_bound():
    baseline = _run_single_op(_make_device(), dim=LARGE_DIM)
    latency = _run_single_op(_make_device(enable_dm_latency=True), dim=LARGE_DIM)
    # The model still runs and produces a real prediction here - the point is that
    # max() discards it, not that it silently returned zero.
    assert latency.dm_read_latency_cycles > 0.0
    # Bandwidth already dominates, so max() picks the bandwidth term unchanged.
    assert latency.mem_rd_cycles_fractional == pytest.approx(baseline.mem_rd_cycles_fractional)


@pytest.mark.unit
def test_fclk_to_devclk_conversion_scales_the_prediction():
    """A prediction in fclk cycles must be converted into the op's device clock.

    Every shipped config has fclk == the matrix clock, so the conversion is a no-op
    in practice and an inverted ratio would pass every other test here. Force a
    decoupled NoC clock to pin the direction: a *slower* fclk means each fclk cycle
    is longer, so the same latency spans *more* device-clock cycles.
    """
    device = _make_device(enable_dm_latency=True)
    devfreq = device.freq_MHz
    assert device.dm_fclk_MHz == devfreq, "fixture assumption: shipped fclk == devclk"

    op = _run_single_op(device, dim=SMALL_DIM)
    tlat_fclk = op.dm_read_latency_cycles
    at_same_clock = device._dm_read_latency_devclk(op, devfreq)
    assert at_same_clock == pytest.approx(tlat_fclk)

    device.dm_fclk_MHz = devfreq / 2.0
    at_half_clock = device._dm_read_latency_devclk(op, devfreq)
    assert at_half_clock == pytest.approx(2.0 * tlat_fclk)

    device.dm_fclk_MHz = devfreq * 2.0
    at_double_clock = device._dm_read_latency_devclk(op, devfreq)
    assert at_double_clock == pytest.approx(tlat_fclk / 2.0)


@pytest.mark.unit
def test_partial_page_rounds_up():
    """Page count and queue depth both round up, so a partial tile still costs one.

    ``inBytes`` one byte past a whole number of pages must land on the next Q, not
    round back down to it.
    """
    device = _make_device(enable_dm_latency=True)
    cfg = device.dm_read_cfg
    assert cfg is not None
    num_cores = device.compute_ip.num_units
    page_bytes = 2048  # 32x32 tile at fp16

    op = _run_single_op(device, dim=SMALL_DIM)
    # Exactly num_cores pages -> Q=1; one byte more -> some core issues a second read.
    op.perf_stats['inBytes'] = num_cores * page_bytes
    device._dm_read_latency_devclk(op, device.freq_MHz)
    at_q1 = op.dm_read_latency_cycles
    op.perf_stats['inBytes'] = num_cores * page_bytes + 1
    device._dm_read_latency_devclk(op, device.freq_MHz)
    at_q2 = op.dm_read_latency_cycles

    channels = device.dm_read_channels
    assert at_q1 == pytest.approx(
        cfg.predict_read_latency(page_bytes, 1, num_channels=channels)
    )
    assert at_q2 == pytest.approx(
        cfg.predict_read_latency(page_bytes, 2, num_channels=channels)
    )
    assert at_q2 > at_q1


@pytest.mark.unit
def test_page_width_follows_tensor_bytes_not_compute_precision():
    """The page is sized by the data's storage width, not by ``op.precision``.

    ``op.precision`` comes from the arch mapping keyed by optype and describes the
    precision the math runs at; on TTNN workloads it disagrees with the tensors (an
    INT8 compute precision over bfloat16 data). Sizing the page from it would both
    halve N and double the page count, since ``inBytes`` counts storage bytes.
    """
    device = _make_device(enable_dm_latency=True)
    cfg = device.dm_read_cfg
    assert cfg is not None
    op = _run_single_op(device, dim=SMALL_DIM)

    # Two 32x32 bfloat16 tiles under an INT8 compute precision, as llama3_decode emits.
    op.precision = 'int8'
    op.perf_stats['inBytes'] = 4096
    op.perf_stats['inElems'] = 2048
    device._dm_read_latency_devclk(op, device.freq_MHz)

    assert device._dm_read_bytes_per_elem(op, 4096) == 2
    # 2048 B pages (2 of them, Q=1) -- not 1024 B pages with 4 of them.
    assert op.dm_read_latency_cycles == pytest.approx(
        cfg.predict_read_latency(2048, 1, num_channels=device.dm_read_channels)
    )

    # With no element count reported, op.precision remains the fallback.
    del op.perf_stats['inElems']
    assert device._dm_read_bytes_per_elem(op, 4096) == 1


@pytest.mark.unit
@pytest.mark.parametrize(
    'memory_layout, buffer_type, applies',
    [
        (TensorMemoryLayout.INTERLEAVED, BufferType.DRAM, True),
        (TensorMemoryLayout.HEIGHT_SHARDED, BufferType.L1, False),
        (TensorMemoryLayout.BLOCK_SHARDED, BufferType.L1, False),
        (TensorMemoryLayout.INTERLEAVED, BufferType.L1, False),
        # DRAM, but sharded: the calibration is for interleaved banks specifically.
        (TensorMemoryLayout.WIDTH_SHARDED, BufferType.DRAM, False),
    ],
)
def test_model_prices_dram_interleaved_reads_only(memory_layout, buffer_type, applies):
    """Only DRAM-interleaved operands are priced; the rest keep the bandwidth estimate.

    The constants describe a NoC walk to an interleaved DRAM bank, so an L1-resident or
    sharded read is a transaction the model was never fit against. TTNN decode workloads
    are almost entirely L1-sharded, and pricing those reads with these constants
    over-charges them.
    """
    device = _make_device(enable_dm_latency=True)
    graph, op = _build_graph(SMALL_DIM)
    graph._tensors['in1']._memory_config = MemoryConfig(memory_layout, buffer_type)
    device.execute_op(op, graph)

    baseline = _run_single_op(_make_device(), dim=SMALL_DIM)
    if applies:
        assert op.mem_rd_cycles_fractional > baseline.mem_rd_cycles_fractional
    else:
        assert op.mem_rd_cycles_fractional == pytest.approx(baseline.mem_rd_cycles_fractional)


@pytest.mark.unit
def test_one_sharded_operand_disqualifies_the_op():
    """Every input is gated, not just the first.

    A read is one transaction shape; if any operand sits somewhere the model was not
    calibrated for, the op as a whole is not something it can price.
    """
    device = _make_device(enable_dm_latency=True)
    graph, op = _build_graph(SMALL_DIM)
    extra = SimTensor({'name': 'in2', 'shape': [SMALL_DIM, SMALL_DIM], 'dtype': 'float32'})
    extra.op_in = ['op']
    graph.add_tensor(extra)
    op.inList = ['in1', 'in2']
    graph._tensors['in1']._memory_config = MemoryConfig(TensorMemoryLayout.INTERLEAVED, BufferType.DRAM)
    assert device._dm_read_model_applies(op, graph)

    extra._memory_config = MemoryConfig(TensorMemoryLayout.HEIGHT_SHARDED, BufferType.L1)
    assert not device._dm_read_model_applies(op, graph)


@pytest.mark.unit
def test_operands_without_a_memory_config_stay_eligible():
    """Unplaced operands are eligible: absence of a config is not evidence of sharding.

    The ONNX front end never attaches one, and Polaris's flat memory model already
    assumes that traffic is DRAM, so refusing them would disable the model on every
    non-TTNN workload.
    """
    device = _make_device(enable_dm_latency=True)
    graph, op = _build_graph(SMALL_DIM)
    assert graph._tensors['in1'].memory_config() is None
    assert device._dm_read_model_applies(op, graph)
    # And with no graph to consult at all (callers that cost a bare op).
    assert device._dm_read_model_applies(op, None)


@pytest.mark.unit
@pytest.mark.parametrize('enabled', [False, True])
def test_read_side_over_count_check_is_relaxed_only_when_enabled(enabled):
    """The read-side over-count error is suppressed by the flag, and only by the flag.

    Charging more read cycles than tot_inBytes justifies is a hard error in the
    default path; with the model on it is legitimate, because latency-bound reads
    exceed bytes/bandwidth by design. Provoked by shrinking inBytes after the op has
    been costed, which lowers the expected cycles without touching utilization.
    """
    device = _make_device(enable_dm_latency=enabled)
    graph, op = _build_graph(LARGE_DIM)
    device.execute_op(op)
    op.perf_stats['inBytes'] //= 4

    if enabled:
        device.get_exec_stats(graph, bs=1)
    else:
        with pytest.raises(ValueError, match='Memory bandwidth validation failed'):
            device.get_exec_stats(graph, bs=1)


@pytest.mark.unit
def test_read_cycles_never_decrease():
    # max() semantics: enabling the model is monotonic at every size.
    for dim in (32, 64, 128, 256, 512, 1024):
        baseline = _run_single_op(_make_device(), dim=dim)
        latency = _run_single_op(_make_device(enable_dm_latency=True), dim=dim)
        assert latency.mem_rd_cycles_fractional >= baseline.mem_rd_cycles_fractional - 1e-9
        # Writes are untouched by the read model.
        assert latency.mem_wr_cycles_fractional == pytest.approx(baseline.mem_wr_cycles_fractional)


@pytest.mark.unit
def test_aggregate_bandwidth_check_tolerates_latency_bound_reads():
    # get_exec_stats runs the memory-bandwidth self-check. Latency-bound reads
    # legitimately exceed bytes/BW, so the check must not raise on the over-count
    # side when the model is enabled.
    device = _make_device(enable_dm_latency=True)
    graph, op = _build_graph(SMALL_DIM)
    device.execute_op(op)
    summary = device.get_exec_stats(graph, bs=1)
    assert summary['tot_mem_rd_cycles'] == op.mem_rd_cycles
