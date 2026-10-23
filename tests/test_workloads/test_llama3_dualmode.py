#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the dual-mode llama3 workloads (workloads/ttnn/llama3_dualmode, built on
workloads/ttnn/tt_transformers_dualmode).

Prefill is pinned to the tt-metal f8c634a silicon captures (bs=1, seq_len 128, max_seq_len 1024, one
generated token; each merged to the first replay session plus its eager tail):

  BH p100a  llama3-prefill-b1-l1-t1-f8c634a-blackhole-p100a-261018      1 layer   44 ops   4.3474 ms
  WH n150   llama3-prefill-b1-f8c634a-wormhole_b0-n150_L-261019         1 layer   34 ops   6.9425 ms
  WH n150   llama3-prefill-b1-f8c634a-wormhole_b0-n150_L-261013 (2L)    2 layers  54 ops   9.7727 ms
  WH n150   llama3-prefill-b1-f8c634a-wormhole_b0-n150_L-261013 (6L)    6 layers 134 ops  21.4748 ms

Each capture has its own instance (llama3_8b_dualmode_prefill_bh, _wh, _wh_2L, _wh_6L), all with the same
f8c634a settings: the 21-op traced forward (rope tables padded to 2048 rows), then the eager tail -- runtime-bound slice, norm, lm_head split by the backend (BH 8 x 16032, WH 3 x 42752), Copy
and the greedy force-argmax sampler (Untilize -> ArgMax).

data/llama3_prefill_b1_bh_p100a_lut.yaml is the tt_perf_mapper output for the BH capture (448 GB/s).
data/llama3_prefill_b1_wh_n150_lut.yaml is the output for the WH 1-layer capture 261019 (288 GB/s),
captured with tracy --no-op-info-cache so every op records its own tensor shapes. The 2- and 6-layer
captures (261013, made without that flag) only check layer scaling against their network totals. A full LUT hit rate proves every emitted op's key (shapes, dtypes, memory,
fidelity) matches silicon.

Decode is pinned to its current front-end op counts, so a change in the shared
tt_transformers_dualmode components that alters the decode graph fails here instead of silently.
"""
import csv
import re
from collections import Counter
from pathlib import Path

import pytest

import polaris
import ttsim.front.ttnn as ttnn
from tests.common import reset_typespec  # noqa: F401  (pytest fixture)
from ttsim.front.ttnn.tensor import DataType
from workloads.ttnn.llama3_dualmode.test_llama3_decode import run_llama3_dualmode_decode
from workloads.ttnn.llama3_dualmode.test_llama3_prefill import run_llama3_dualmode

REPO = Path(polaris.__file__).parent
PREFILL_LUT = Path(__file__).parent / 'data' / 'llama3_prefill_b1_bh_p100a_lut.yaml'
PREFILL_LUT_WH = Path(__file__).parent / 'data' / 'llama3_prefill_b1_wh_n150_lut.yaml'

# Instances llama3_8b_dualmode_prefill_bh / _wh: Llama-3.1-8B's traced prefill lengths on N150 and P100.
F8C634A = {'rope_trace_prefill_seq_lens': [128, 1024]}
# The traced forward alone (lm_head: false): the runner config for the 21-op forward graph.
PREFILL_CFG = {'bs': 1, 'model_name': 'llama3-8B', 'n_layers': 1, 'max_generated_tokens': 1,
               'max_seq_len': 1024, 'lm_head': False, **F8C634A}
STEP_CFG = dict(PREFILL_CFG, lm_head=True, prompt_len=80)

# Capture op order, as Polaris optypes.
PREFILL_OPS = [
    'Embedding', 'LayerNormalization', 'MatMul', 'CreateQKVHeads',
    'RotaryEmbeddingLlama', 'RotaryEmbeddingLlama', 'Typecast', 'Typecast',
    'PagedFillCache', 'PagedFillCache', 'Typecast', 'ScaledDotProductAttention',
    'ConcatHeads', 'MatMul', 'Add', 'LayerNormalization',
    'MatMul', 'MatMul', 'Mul', 'MatMul', 'Add',
]
LAYER_OPS = PREFILL_OPS[1:]  # one decoder layer (20 ops)
TAIL_OPS = [  # front-end graph, before the backend splits the lm_head MatMul per arch
    'Slice', 'LayerNormalization', 'InterleavedToSharded', 'MatMul', 'Copy', 'Untilize', 'ArgMax',
]


def _capture_op_counts(n_layers, lm_head_chunks):
    """Opstats optype counts for an n-layer step after the backend's lm_head split."""
    counts = Counter(PREFILL_OPS[:1] + LAYER_OPS * n_layers + TAIL_OPS)
    counts['MatMul'] += lm_head_chunks - 1           # the one lm_head MatMul becomes N chunks
    counts['ShardedToInterleaved'] += lm_head_chunks  # + one STS per chunk
    counts['Concat'] += 1                             # + the concat back to full vocab
    return dict(counts)


# (instance, arch file, SKU, fixture LUT, layers, lm_head chunks, capture ms, tolerance)
BH_LUT = Path(__file__).parent / 'data' / 'llama3_prefill_b1_bh_p100a_lut.yaml'
WH_LUT = Path(__file__).parent / 'data' / 'llama3_prefill_b1_wh_n150_lut.yaml'
CAPTURES = [
    ('llama3_8b_dualmode_prefill_bh', 'tt_bh_llama3.yaml', 'p100a', BH_LUT, 1, 8, 4.3474, 0.01),
    # WH: the LUT is built from the 1-layer capture 261019 alone. The 2- and 6-layer captures are separate
    # runs whose per-layer matmul times vary run to run (FF2 646-761 us), so they get the skills' 2 % gate.
    ('llama3_8b_dualmode_prefill_wh', 'tt_wh_llama3.yaml', 'n150', WH_LUT, 1, 3, 6.9425, 0.01),
    ('llama3_8b_dualmode_prefill_wh_2L', 'tt_wh_llama3.yaml', 'n150', WH_LUT, 2, 3, 9.7727, 0.02),
    ('llama3_8b_dualmode_prefill_wh_6L', 'tt_wh_llama3.yaml', 'n150', WH_LUT, 6, 3, 21.4748, 0.02),
]

def _build(runner, cfg):
    """Run a dual-mode runner on the Polaris shim; return (ops in emission order, tensors)."""
    dev = ttnn.open_device(device_id=0)
    try:
        runner('llama3_dualmode', dev, cfg)
        return list(dev.ops.values()), dict(dev.tensors)
    finally:
        ttnn.close_device(dev)


def _ttnn_dtype(t):
    return getattr(t, '_ttnn_dtype', None)


@pytest.mark.unit
def test_prefill_graph_matches_capture():
    ops, tensors = _build(run_llama3_dualmode, PREFILL_CFG)
    assert [o.optype for o in ops] == PREFILL_OPS

    def inp(op, i):
        return tensors[op.inList[i]]

    def out(op):
        return tensors[op.outList[0]]

    # Matmul fidelity: QKV / WO / FF2 HiFi2, FF1 / FF3 LoFi (capture MATH FIDELITY column).
    matmuls = [o for o in ops if o.optype == 'MatMul']
    assert [m.attrs.get('math_fidelity') for m in matmuls] == ['HiFi2', 'HiFi2', 'LoFi', 'LoFi', 'HiFi2']

    # Rope reads the whole padded cos/sin tables, not seq_len-sized slices: f8c634a pads them for the
    # traced prefill lengths, [128, 1024] -> 2048 rows (both captures: [1,1,2048,128]).
    for rope in (o for o in ops if o.optype == 'RotaryEmbeddingLlama'):
        assert list(inp(rope, 1).shape) == [1, 1, 2048, 128]
        assert list(inp(rope, 2).shape) == [1, 1, 2048, 128]

    # Paged KV fill takes the page table as a third, INT32 operand.
    for fill in (o for o in ops if o.optype == 'PagedFillCache'):
        assert len(fill.inList) == 3
        assert _ttnn_dtype(inp(fill, 2)) == DataType.INT32

    # Q is cast to bf8 before SDPA (activation_dtype or bfloat8_b), and all SDPA inputs are bf8.
    q_cast = ops[PREFILL_OPS.index('ScaledDotProductAttention') - 1]
    assert q_cast.optype == 'Typecast' and _ttnn_dtype(out(q_cast)) == DataType.BFLOAT8_B
    sdpa = ops[PREFILL_OPS.index('ScaledDotProductAttention')]
    assert all(_ttnn_dtype(inp(sdpa, i)) == DataType.BFLOAT8_B for i in range(3))

    # WO output is bf8; both residual adds stay bf16 (a bf8 add output leaks into the ff_norm/MLP keys).
    assert _ttnn_dtype(out(matmuls[1])) == DataType.BFLOAT8_B
    for add in (o for o in ops if o.optype == 'Add'):
        assert _ttnn_dtype(out(add)) == DataType.BFLOAT16


def _run_polaris_with_fixture_lut(tmp_path, instance, archfile, sku, lut):
    """Run Polaris on one llama3 prefill instance against a fixture LUT; return its opstats rows."""
    arch = (REPO / 'config' / archfile).read_text()
    arch = re.sub(r'operator_lookup_file:.*', f'operator_lookup_file: {lut}', arch)
    arch_path = tmp_path / f'{sku}_llama3_prefill_test.yaml'
    arch_path.write_text(arch)

    rc = polaris.main([
        '--odir', str(tmp_path), '--study', 'llama3_prefill',
        '--wlspec', str(REPO / 'config' / 'all_workloads.yaml'),
        '--archspec', str(arch_path),
        '--wlmapspec', str(REPO / 'config' / 'wl2archmapping.yaml'),
        '--filterarch', sku, '--filterwli', instance,
        '--dump_stats_csv',
    ])
    assert rc == 0
    stats = list((tmp_path / 'llama3_prefill' / 'STATS').glob(f'*-{instance}-*opstats.csv'))
    assert len(stats) == 1, stats
    return list(csv.DictReader(stats[0].open()))


@pytest.mark.unit
def test_prefill_step_graph():
    ops, tensors = _build(run_llama3_dualmode, STEP_CFG)
    assert [o.optype for o in ops] == PREFILL_OPS + TAIL_OPS

    tail = ops[len(PREFILL_OPS):]
    # Runtime-bound slice of the last token's 32-row tile: the start/end int32 tensors are inputs 1-2.
    tail_slice = tail[0]
    assert len(tail_slice.inList) == 3
    assert list(tensors[tail_slice.outList[0]].shape) == [1, 1, 32, 4096]
    assert all(_ttnn_dtype(tensors[t]) == DataType.INT32 for t in tail_slice.inList[1:])
    # lm_head: width-sharded L1 bf16 input (the DRAM-sharded matmul), bf8 vocab-wide output.
    lm_head = tail[3]
    lm_in = tensors[lm_head.inList[0]]
    assert lm_in._memory_config.is_sharded() and _ttnn_dtype(lm_in) == DataType.BFLOAT16
    assert list(tensors[lm_head.outList[0]].shape) == [1, 1, 32, 128256]
    assert _ttnn_dtype(tensors[lm_head.outList[0]]) == DataType.BFLOAT8_B
    # Copy moves the logits to DRAM (capture: Untilize in0 DRAM bf8).
    copy_out = tensors[tail[4].outList[0]]
    assert copy_out._memory_config.buffer_type.name == 'DRAM' and _ttnn_dtype(copy_out) == DataType.BFLOAT8_B
    # Untilize turns the bf8 logits into bf16 (row-major cannot hold bf8), which ArgMax reads.
    untilize, argmax = tail[5], tail[6]
    assert _ttnn_dtype(tensors[untilize.outList[0]]) == DataType.BFLOAT16
    assert list(tensors[argmax.outList[0]].shape) == [1, 1, 32]


@pytest.mark.unit
@pytest.mark.parametrize('instance,archfile,sku,lut,n_layers,chunks,capture_ms,rel', CAPTURES,
                         ids=['bh_p100a_1L', 'wh_n150_1L', 'wh_n150_2L', 'wh_n150_6L'])
def test_prefill_lut_hits_and_projection(reset_typespec, tmp_path, instance, archfile, sku, lut,  # noqa: F811
                                         n_layers, chunks, capture_ms, rel):
    rows = _run_polaris_with_fixture_lut(tmp_path, instance, archfile, sku, lut)
    assert dict(Counter(r['optype'] for r in rows)) == _capture_op_counts(n_layers, chunks)
    misses = [r['optype'] for r in rows if r['uses_perf_lookup'] != 'True']
    assert not misses, f'LUT misses: {misses}'
    total_ms = sum(float(r['msecs']) for r in rows)
    assert total_ms == pytest.approx(capture_ms, rel=rel)


@pytest.mark.unit
@pytest.mark.parametrize('n_layers,expected_ops', [(1, 51), (None, 733)], ids=['1layer', 'all_layers'])
def test_decode_graph_op_count(n_layers, expected_ops):
    # Front-end graph counts. The opstats show 16 more (67 / 749): the backend splits lm_head into
    # 8 column-tile matmuls + 8 ShardedToInterleaved + Concat (device._split_column_ops, #477).
    cfg = {'bs': 1, 'model_name': 'llama3-8B', 'max_seq_len': 1024}
    if n_layers is not None:
        cfg['n_layers'] = n_layers
    ops, _ = _build(run_llama3_dualmode_decode, cfg)
    assert len(ops) == expected_ops
