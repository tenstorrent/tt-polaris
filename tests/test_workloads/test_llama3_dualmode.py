#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the dual-mode llama3 workloads (workloads/ttnn/llama3_dualmode, built on
workloads/ttnn/tt_transformers_dualmode).

Prefill is pinned to the BH p100a silicon capture llama3-prefill-b1-l1-t1-02b76a9-blackhole-p100a-
260924 (bs=1, 1 layer, seq_len 128, max_seq_len 1024), merged with the #539 trace+replay merge so one
step includes the eager tail: 56 device ops, 26.0174 ms. Of those, the traced forward is 21 ops
(3.24 ms) and the tail is last-token norm + 8-way-split lm_head + on-device sampling (35 ops).
data/llama3_prefill_b1_bh_p100a_lut.yaml is the tt_perf_mapper output for that merge (448 GB/s), so a
full LUT hit rate proves every emitted op's key (shapes, dtypes, memory, fidelity) matches silicon.

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

# The traced forward alone (lm_head: false): the runner config for the 21-op forward graph.
PREFILL_CFG = {'bs': 1, 'model_name': 'llama3-8B', 'n_layers': 1, 'max_generated_tokens': 1,
               'max_seq_len': 1024, 'lm_head': False}

# Capture op order, as Polaris optypes.
PREFILL_OPS = [
    'Embedding', 'LayerNormalization', 'MatMul', 'CreateQKVHeads',
    'RotaryEmbeddingLlama', 'RotaryEmbeddingLlama', 'Typecast', 'Typecast',
    'PagedFillCache', 'PagedFillCache', 'Typecast', 'ScaledDotProductAttention',
    'ConcatHeads', 'MatMul', 'Add', 'LayerNormalization',
    'MatMul', 'MatMul', 'Mul', 'MatMul', 'Add',
]
# Full step (instance llama3_8b_dualmode_prefill; its defaults are lm_head/sampling on): the forward above plus the eager tail, in the
# order Polaris emits it. The index-chunk Slices and ManualSeed have no data dependence on the logits,
# so the opstats list them earlier than the capture dispatches them; the multiset is what must match.
TTFT_CFG = dict(PREFILL_CFG, lm_head=True, sampling=True, prompt_len=80)
TTFT_TAIL_OPS = [  # front-end graph, before the backend splits the lm_head MatMul 8 ways
    'Slice', 'LayerNormalization', 'InterleavedToSharded', 'MatMul', 'Copy',
    'Typecast', 'Slice', 'Slice', 'Slice', 'Slice', 'TopK', 'TopK', 'Concat', 'Concat',
    'Typecast', 'Add', 'Untilize', 'ManualSeed', 'Sampling',
]
CAPTURE_TTFT_MS = 26.0174
CAPTURE_TTFT_OP_COUNTS = {
    'MatMul': 13, 'ShardedToInterleaved': 8, 'Slice': 5, 'Typecast': 5, 'LayerNormalization': 3,
    'Add': 3, 'Concat': 3, 'RotaryEmbeddingLlama': 2, 'PagedFillCache': 2, 'TopK': 2, 'Embedding': 1,
    'CreateQKVHeads': 1, 'ScaledDotProductAttention': 1, 'ConcatHeads': 1, 'Mul': 1,
    'InterleavedToSharded': 1, 'Copy': 1, 'Untilize': 1, 'ManualSeed': 1, 'Sampling': 1,
}


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

    # Rope reads the full max_seq_len cos/sin tables, not seq_len-sized slices.
    for rope in (o for o in ops if o.optype == 'RotaryEmbeddingLlama'):
        assert list(inp(rope, 1).shape) == [1, 1, 1024, 128]
        assert list(inp(rope, 2).shape) == [1, 1, 1024, 128]

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


def _run_polaris_with_fixture_lut(tmp_path, instance):
    """Run Polaris on one llama3 prefill instance against the fixture LUT; return its opstats rows."""
    arch = (REPO / 'config' / 'tt_bh_llama3.yaml').read_text()
    arch = re.sub(r'operator_lookup_file:.*', f'operator_lookup_file: {PREFILL_LUT}', arch)
    arch_path = tmp_path / 'tt_bh_llama3_prefill_test.yaml'
    arch_path.write_text(arch)

    rc = polaris.main([
        '--odir', str(tmp_path), '--study', 'llama3_prefill',
        '--wlspec', str(REPO / 'config' / 'all_workloads.yaml'),
        '--archspec', str(arch_path),
        '--wlmapspec', str(REPO / 'config' / 'wl2archmapping.yaml'),
        '--filterarch', 'p100a', '--filterwli', instance,
        '--dump_stats_csv',
    ])
    assert rc == 0
    stats = list((tmp_path / 'llama3_prefill' / 'STATS').glob(f'*-{instance}-*opstats.csv'))
    assert len(stats) == 1, stats
    return list(csv.DictReader(stats[0].open()))


@pytest.mark.unit
def test_prefill_ttft_tail_graph():
    ops, tensors = _build(run_llama3_dualmode, TTFT_CFG)
    assert [o.optype for o in ops] == PREFILL_OPS + TTFT_TAIL_OPS

    tail = ops[len(PREFILL_OPS):]
    # The tail runs on the last token's 32-row tile only (tt-metal slices before the final norm).
    assert list(tensors[tail[0].outList[0]].shape) == [1, 1, 32, 4096]
    # lm_head: width-sharded L1 bf16 input (the DRAM-sharded matmul), bf8 vocab-wide output.
    lm_head = tail[3]
    lm_in = tensors[lm_head.inList[0]]
    assert lm_in._memory_config.is_sharded() and _ttnn_dtype(lm_in) == DataType.BFLOAT16
    assert list(tensors[lm_head.outList[0]].shape) == [1, 1, 32, 128256]
    assert _ttnn_dtype(tensors[lm_head.outList[0]]) == DataType.BFLOAT8_B
    # Copy moves the logits to DRAM, so the sampler reads DRAM (capture: Typecast in0 DRAM bf8).
    copy_out = tensors[tail[4].outList[0]]
    assert copy_out._memory_config.buffer_type.name == 'DRAM' and _ttnn_dtype(copy_out) == DataType.BFLOAT8_B


@pytest.mark.unit
def test_prefill_ttft_lut_hits_and_projection(reset_typespec, tmp_path):  # noqa: F811
    rows = _run_polaris_with_fixture_lut(tmp_path, 'llama3_8b_dualmode_prefill')
    # After the backend splits the lm_head into 8 column chunks (+ 8 ShardedToInterleaved + Concat).
    assert dict(Counter(r['optype'] for r in rows)) == CAPTURE_TTFT_OP_COUNTS
    misses = [r['optype'] for r in rows if r['uses_perf_lookup'] != 'True']
    assert not misses, f'LUT misses: {misses}'
    total_ms = sum(float(r['msecs']) for r in rows)
    assert total_ms == pytest.approx(CAPTURE_TTFT_MS, rel=0.01)


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
