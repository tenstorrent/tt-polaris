#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for tools.profiling.compare_layers."""

import pytest
from tools.profiling.compare_layers import (
    compare_layers,
    find_next_match,
    normalize_optype,
)


# ── find_next_match tests ──────────────────────────────────────────────


@pytest.mark.unit
def test_find_next_match_basic():
    """Test basic matching of operation types."""
    layers = [
        {'optype': 'MatMul'},
        {'optype': 'Add'},
        {'optype': 'Softmax'},
        {'optype': 'MatMul'},
    ]

    # Find first MatMul
    result = find_next_match(layers, 0, normalize_optype('MatMul'))
    assert result == 0

    # Find second MatMul
    result = find_next_match(layers, 1, normalize_optype('MatMul'))
    assert result == 3

    # Find Add
    result = find_next_match(layers, 0, normalize_optype('Add'))
    assert result == 1


@pytest.mark.unit
def test_find_next_match_not_found():
    """Test when operation type is not found."""
    layers = [
        {'optype': 'MatMul'},
        {'optype': 'Add'},
    ]

    result = find_next_match(layers, 0, normalize_optype('Softmax'))
    assert result is None


@pytest.mark.unit
def test_find_next_match_with_max_distance():
    """Test max_distance parameter limits search range."""
    layers = [
        {'optype': 'MatMul'},
        {'optype': 'Add'},
        {'optype': 'Softmax'},
        {'optype': 'MatMul'},
    ]

    # Should find it within distance
    result = find_next_match(layers, 0, normalize_optype('Softmax'), max_distance=5)
    assert result == 2

    # Should not find it (too far)
    result = find_next_match(layers, 0, normalize_optype('Softmax'), max_distance=2)
    assert result is None


@pytest.mark.unit
def test_find_next_match_normalization():
    """Test that operation type normalization works in matching."""
    layers = [
        {'optype': 'MatMul'},
        {'optype': 'Add'},
    ]

    # 'MatMul' normalizes to 'matmul'
    result = find_next_match(layers, 0, normalize_optype('MatMul'))
    assert result == 0

    # Test with different case
    result = find_next_match(layers, 0, normalize_optype('matmul'))
    assert result == 0


# ── compare_layers tests ──────────────────────────────────────────────


@pytest.mark.unit
def test_compare_layers_exact_match():
    """Test comparison with exact matching layers."""
    polaris_layers = [
        {
            'seqno': 0,
            'optype': 'MatMul',
            'input_tensors': ['1x64x128'],
            'output_tensors': ['1x64x256'],
        }
    ]
    profiler_layers = [
        {
            'seqno': 0,
            'optype': 'MatMul',
            'input_tensors': ['1x64x128'],
            'output_tensors': ['1x64x256'],
        }
    ]

    stats = compare_layers(polaris_layers, profiler_layers)
    assert stats.total_matches == 1
    assert stats.unmatched_polaris == 0
    assert stats.unmatched_profiler == 0


@pytest.mark.unit
def test_compare_layers_type_mismatch():
    """Test comparison with mismatched operation types."""
    polaris_layers = [
        {
            'seqno': 0,
            'optype': 'MatMul',
            'input_tensors': ['1x64x128'],
            'output_tensors': ['1x64x256'],
        }
    ]
    profiler_layers = [
        {
            'seqno': 0,
            'optype': 'Add',
            'input_tensors': ['1x64x128'],
            'output_tensors': ['1x64x128'],
        }
    ]

    stats = compare_layers(polaris_layers, profiler_layers)
    assert stats.total_matches == 0
    # When types don't match and polaris op not found ahead, marked as name_mismatch
    assert stats.name_mismatches == 1
    # Remaining profiler entry becomes unmatched
    assert stats.unmatched_profiler == 1


@pytest.mark.unit
def test_compare_layers_out_of_order():
    """Test comparison with out-of-order operations - forward search only."""
    polaris_layers = [
        {
            'seqno': 0,
            'optype': 'MatMul',
            'input_tensors': ['1x64x128'],
            'output_tensors': ['1x64x256'],
        },
        {
            'seqno': 1,
            'optype': 'Add',
            'input_tensors': ['1x64x256'],
            'output_tensors': ['1x64x256'],
        },
    ]
    profiler_layers = [
        {
            'seqno': 0,
            'optype': 'Add',
            'input_tensors': ['1x64x256'],
            'output_tensors': ['1x64x256'],
        },
        {
            'seqno': 1,
            'optype': 'MatMul',
            'input_tensors': ['1x64x128'],
            'output_tensors': ['1x64x256'],
        },
    ]

    # Algorithm searches forward in profiler for polaris ops
    # Polaris MatMul finds profiler MatMul at index 1, skipping profiler Add
    # Polaris Add is then not found in remaining profiler entries (exhausted)
    stats = compare_layers(polaris_layers, profiler_layers)
    assert stats.total_matches == 1  # Only MatMul matches
    assert stats.unmatched_polaris == 1  # Polaris Add not matched
    assert stats.unmatched_profiler == 1  # Profiler Add skipped


@pytest.mark.unit
def test_compare_layers_unmatched_polaris():
    """Test when Polaris has extra operations."""
    polaris_layers = [
        {
            'seqno': 0,
            'optype': 'MatMul',
            'input_tensors': ['1x64x128'],
            'output_tensors': ['1x64x256'],
        },
        {
            'seqno': 1,
            'optype': 'Softmax',
            'input_tensors': ['1x64x256'],
            'output_tensors': ['1x64x256'],
        },
    ]
    profiler_layers = [
        {
            'seqno': 0,
            'optype': 'MatMul',
            'input_tensors': ['1x64x128'],
            'output_tensors': ['1x64x256'],
        }
    ]

    stats = compare_layers(polaris_layers, profiler_layers)
    assert stats.total_matches == 1
    assert stats.unmatched_polaris == 1
    assert stats.unmatched_profiler == 0


@pytest.mark.unit
def test_compare_layers_unmatched_profiler():
    """Test when profiler has extra operations."""
    polaris_layers = [
        {
            'seqno': 0,
            'optype': 'MatMul',
            'input_tensors': ['1x64x128'],
            'output_tensors': ['1x64x256'],
        }
    ]
    profiler_layers = [
        {
            'seqno': 0,
            'optype': 'MatMul',
            'input_tensors': ['1x64x128'],
            'output_tensors': ['1x64x256'],
        },
        {
            'seqno': 1,
            'optype': 'Add',
            'input_tensors': ['1x64x256'],
            'output_tensors': ['1x64x256'],
        },
    ]

    stats = compare_layers(polaris_layers, profiler_layers)
    assert stats.total_matches == 1
    assert stats.unmatched_polaris == 0
    assert stats.unmatched_profiler == 1


@pytest.mark.unit
def test_compare_layers_binary_special_case():
    """Test binary operation special handling (scalar operands)."""
    # Binary ops may have scalar operands that don't appear in profiler
    polaris_layers = [
        {
            'seqno': 0,
            'optype': 'Add',
            'input_tensors': ['1x64x256', '1'],
            'output_tensors': ['1x64x256'],
        }
    ]
    profiler_layers = [
        {
            'seqno': 0,
            'optype': 'Add',
            'input_tensors': ['1x64x256'],  # Scalar operand not tracked
            'output_tensors': ['1x64x256'],
        }
    ]

    stats = compare_layers(polaris_layers, profiler_layers)
    # Should still match due to binary op special handling
    assert stats.total_matches == 1


@pytest.mark.unit
def test_compare_layers_reshape_special_case():
    """Test reshape operation with strip_leading_ones."""
    polaris_layers = [
        {
            'seqno': 0,
            'optype': 'Reshape',
            'input_tensors': ['1x64x256'],
            'output_tensors': ['16384'],
        }
    ]
    profiler_layers = [
        {
            'seqno': 0,
            'optype': 'Reshape',
            'input_tensors': ['64x256'],
            'output_tensors': ['16384'],
        }
    ]

    # With strip_leading_ones, the input shapes become equivalent
    stats = compare_layers(polaris_layers, profiler_layers, strip_leading_ones=True)
    assert stats.total_matches == 1


@pytest.mark.unit
def test_compare_layers_strip_leading_ones():
    """Test strip_leading_ones option."""
    polaris_layers = [
        {
            'seqno': 0,
            'optype': 'MatMul',
            'input_tensors': ['1x1x64x128'],
            'output_tensors': ['1x1x64x256'],
        }
    ]
    profiler_layers = [
        {
            'seqno': 0,
            'optype': 'MatMul',
            'input_tensors': ['64x128'],
            'output_tensors': ['64x256'],
        }
    ]

    # Without strip_leading_ones, shapes don't match
    stats = compare_layers(polaris_layers, profiler_layers, strip_leading_ones=False)
    assert stats.total_matches == 0

    # With strip_leading_ones, shapes should match
    stats = compare_layers(polaris_layers, profiler_layers, strip_leading_ones=True)
    assert stats.total_matches == 1


@pytest.mark.unit
def test_compare_layers_empty_lists():
    """Test comparison with empty lists."""
    stats = compare_layers([], [])
    assert stats.total_matches == 0
    assert stats.unmatched_polaris == 0
    assert stats.unmatched_profiler == 0


# ── _aggregate_by_lut_key tests ───────────────────────────────────────────

@pytest.mark.unit
def test_aggregate_by_lut_key_prefers_resolved_over_literal():
    """Polaris-side rows present both ``lut_key`` (literal) and ``lut_key_resolved``;
    aggregation must group by the resolved tuple so polaris/profiler rows that
    semantically share a LUT entry land in the same bucket.
    """
    from tools.profiling.compare_layers import _aggregate_by_lut_key

    literal_a = ('halo', 1, 1, 1024, 512, 'ROW_MAJOR', 'BFLOAT16', 'DEV_1_L1_HEIGHT_SHARDED', 'N/A')
    resolved_a = ('halo', 1, 1, 1024, 512, 'ROW_MAJOR', 'BFLOAT16', 'DEV_1_L1_BLOCK_SHARDED', 'N/A')
    literal_b = ('move', 1, 1, 1024, 512, 'ROW_MAJOR', 'BFLOAT16', 'DEV_1_L1_HEIGHT_SHARDED', 'N/A')
    resolved_b = ('move', 1, 1, 1024, 512, 'ROW_MAJOR', 'BFLOAT16', 'DEV_1_L1_HEIGHT_SHARDED', 'N/A',
                   1, 1, 1024, 512, 'ROW_MAJOR', 'BFLOAT16', 'DEV_1_L1_HEIGHT_SHARDED')
    layers = [
        # Row 1: HEIGHT→BLOCK shard fallback resolved the lookup.
        {'optype': 'halo', 'lut_key': literal_a, 'lut_key_resolved': resolved_a,
         'duration_ms': 0.05, 'uses_perf_lookup': True},
        # Row 2: arity-1→arity-2 dup fallback resolved the lookup.
        {'optype': 'move', 'lut_key': literal_b, 'lut_key_resolved': resolved_b,
         'duration_ms': 0.01, 'uses_perf_lookup': True},
    ]
    by_key = _aggregate_by_lut_key(layers)
    keys = list(by_key.keys())
    # Two distinct resolved keys → two buckets, each with count 1.
    assert len(keys) == 2
    resolved_a_str = tuple(str(f) for f in resolved_a)
    resolved_b_str = tuple(str(f) for f in resolved_b)
    assert resolved_a_str in by_key
    assert resolved_b_str in by_key
    # Literal-only keys must NOT appear when a resolved key is present.
    literal_a_str = tuple(str(f) for f in literal_a)
    literal_b_str = tuple(str(f) for f in literal_b)
    assert literal_a_str not in by_key
    assert literal_b_str not in by_key


@pytest.mark.unit
def test_aggregate_by_lut_key_falls_back_to_literal_when_no_resolved():
    """Profiler-side rows only carry ``lut_key`` (their literal key IS a LUT entry).
    Aggregation must fall back to it when ``lut_key_resolved`` is absent or None.
    """
    from tools.profiling.compare_layers import _aggregate_by_lut_key

    key = ('halo', 1, 1, 1024, 512, 'ROW_MAJOR', 'BFLOAT16', 'DEV_1_L1_BLOCK_SHARDED', 'N/A')
    layers_no_resolved = [
        {'optype': 'halo', 'lut_key': key, 'duration_ms': 0.05, 'uses_perf_lookup': False},
    ]
    layers_none_resolved = [
        {'optype': 'halo', 'lut_key': key, 'lut_key_resolved': None,
         'duration_ms': 0.05, 'uses_perf_lookup': False},
    ]
    key_str = tuple(str(f) for f in key)
    for layers in (layers_no_resolved, layers_none_resolved):
        by_key = _aggregate_by_lut_key(layers)
        assert key_str in by_key
        cnt, ms, lut = by_key[key_str]
        assert cnt == 1
        assert ms == pytest.approx(0.05)


@pytest.mark.unit
def test_aggregate_by_lut_key_skips_rows_without_any_key():
    """Rows lacking both ``lut_key`` and ``lut_key_resolved`` are skipped (no key to group by)."""
    from tools.profiling.compare_layers import _aggregate_by_lut_key

    layers = [
        {'optype': 'halo', 'duration_ms': 0.05},
        {'optype': 'move', 'lut_key': None, 'lut_key_resolved': None, 'duration_ms': 0.01},
    ]
    by_key = _aggregate_by_lut_key(layers)
    assert by_key == {}


# ── _lut_key_n_slots / _lut_key_slot_field tests ──────────────────────────

@pytest.mark.unit
def test_lut_key_n_slots_halo_v4_is_one_slot():
    """Halo v4 16-tuple must be treated as 1 input slot, not 2 (binary)."""
    from tools.profiling.compare_layers import _lut_key_n_slots

    halo_v4 = ('halo', 1, 1, 16384, 128, 'ROW_MAJOR', 'BFLOAT16', 'DEV_1_L1_HEIGHT_SHARDED',
                'N/A', 2, 2, 0, 0, 0, 0, False)
    assert len(halo_v4) == 16
    assert _lut_key_n_slots(halo_v4) == 1


@pytest.mark.unit
def test_lut_key_n_slots_binary_is_two_slots():
    """Standard binary 16-tuple (two operands) must report 2 slots."""
    from tools.profiling.compare_layers import _lut_key_n_slots

    binary = ('add', 1, 1, 1024, 512, 'TILE', 'BFLOAT16', 'DEV_1_L1_HEIGHT_SHARDED',
              'N/A', 1, 1, 1024, 512, 'TILE', 'BFLOAT16', 'DEV_1_L1_HEIGHT_SHARDED')
    assert len(binary) == 16
    assert _lut_key_n_slots(binary) == 2


@pytest.mark.unit
def test_lut_key_slot_field_binary_slot1_memory():
    """Slot-1 field-6 (memory) of a binary 16-tuple must return index 15, not ''."""
    from tools.profiling.compare_layers import _lut_key_slot_field

    binary = ('add', 1, 1, 1024, 512, 'TILE', 'BFLOAT16', 'MEM_A',
              'HiFi4', 1, 1, 1024, 512, 'TILE', 'BFLOAT16', 'MEM_B')
    assert _lut_key_slot_field(binary, 1, 6) == 'MEM_B'


@pytest.mark.unit
def test_lut_key_slot_field_binary_mf_not_in_slot():
    """math_fidelity at index 8 must not appear as a slot-1 field."""
    from tools.profiling.compare_layers import _lut_key_slot_field

    binary = ('add', 1, 1, 1024, 512, 'TILE', 'BFLOAT16', 'MEM_A',
              'HiFi4', 11, 22, 33, 44, 'TILE', 'BFLOAT16', 'MEM_B')
    # slot-1 field-0 must be w1=11, not MF='HiFi4' (index 8)
    assert _lut_key_slot_field(binary, 1, 0) == '11'


# ── _lut_key_variant_tail / _lut_key_variant_col_names tests ─────────────

@pytest.mark.unit
def test_lut_key_variant_tail_halo_v3():
    from tools.profiling.compare_layers import _lut_key_variant_tail

    key = ('halo', 1, 1, 512, 128, 'ROW_MAJOR', 'BFLOAT16', 'L1', 'N/A',
           3, 3, 1, 1, 1, 1)
    assert len(key) == 15
    tail = _lut_key_variant_tail(key)
    assert tail == [
        ('kernel_h', '3'), ('kernel_w', '3'),
        ('stride_h', '1'), ('stride_w', '1'),
        ('padding_h', '1'), ('padding_w', '1'),
    ]


@pytest.mark.unit
def test_lut_key_variant_tail_halo_v4():
    from tools.profiling.compare_layers import _lut_key_variant_tail

    key = ('halo', 1, 1, 512, 128, 'ROW_MAJOR', 'BFLOAT16', 'L1', 'N/A',
           2, 2, 2, 2, 0, 0, True)
    assert len(key) == 16
    tail = _lut_key_variant_tail(key)
    assert tail == [
        ('kernel_h', '2'), ('kernel_w', '2'),
        ('stride_h', '2'), ('stride_w', '2'),
        ('padding_h', '0'), ('padding_w', '0'),
        ('is_transpose', 'True'),
    ]


@pytest.mark.unit
def test_lut_key_variant_tail_its():
    from tools.profiling.compare_layers import _lut_key_variant_tail

    key = ('interleavedtosharded', 1, 1, 512, 128, 'TILE', 'BFLOAT16', 'DRAM', 'N/A',
           'L1_BLOCK_SHARDED')
    assert len(key) == 10
    assert _lut_key_variant_tail(key) == [('output_0_memory', 'L1_BLOCK_SHARDED')]


@pytest.mark.unit
def test_lut_key_variant_tail_standard_empty():
    from tools.profiling.compare_layers import _lut_key_variant_tail

    key = ('conv2d', 1, 1, 512, 128, 'TILE', 'BFLOAT16', 'L1', 'N/A')
    assert _lut_key_variant_tail(key) == []


@pytest.mark.unit
def test_lut_key_variant_col_names_ordering():
    """Union of variant fields across mixed keys preserves insertion order."""
    from tools.profiling.compare_layers import _lut_key_variant_col_names

    halo_v3 = ('halo', 1, 1, 512, 128, 'ROW_MAJOR', 'BFLOAT16', 'L1', 'N/A',
                3, 3, 1, 1, 1, 1)
    its = ('interleavedtosharded', 1, 1, 512, 128, 'TILE', 'BFLOAT16', 'DRAM', 'N/A',
           'L1_BLOCK_SHARDED')
    standard = ('conv2d', 1, 1, 512, 128, 'TILE', 'BFLOAT16', 'L1', 'N/A')

    names = _lut_key_variant_col_names([halo_v3, its, standard])
    assert names == ['kernel_h', 'kernel_w', 'stride_h', 'stride_w', 'padding_h', 'padding_w', 'output_0_memory']


# ── _print_perf_comparison LUT column split tests ─────────────────────────


def _make_layer(optype: str, duration_ms: float, uses_perf_lookup: bool = False) -> dict:
    return {'optype': optype, 'duration_ms': duration_ms, 'uses_perf_lookup': uses_perf_lookup}


@pytest.mark.unit
def test_perf_comparison_mixed_mode_shows_only_lut1(capsys):
    """In mixed mode (polaris-vs-profiler) only side-1 has LUT hits.
    Output must show LUT1 column but NOT LUT2, so the hit rate is not
    diluted by the profiler-side zero count."""
    from tools.profiling.compare_layers import _print_perf_comparison

    layers1 = [_make_layer('conv2d', 1.0, uses_perf_lookup=True)]
    layers2 = [_make_layer('conv2d', 1.1, uses_perf_lookup=False)]

    _print_perf_comparison(layers1, layers2, label1='Polaris', label2='Profiler')
    out = capsys.readouterr().out

    assert 'LUT1' in out
    assert 'LUT2' not in out
    # Hit ratio must be 1/1, not 1/2
    assert '1/1' in out
    assert '1/2' not in out


@pytest.mark.unit
def test_perf_comparison_same_type_shows_both_lut_columns(capsys):
    """In same-type mode (e.g. polaris-vs-polaris) both sides carry LUT hits.
    Output must show both LUT1 and LUT2 columns."""
    from tools.profiling.compare_layers import _print_perf_comparison

    layers1 = [_make_layer('conv2d', 1.0, uses_perf_lookup=True)]
    layers2 = [_make_layer('conv2d', 1.1, uses_perf_lookup=True)]

    _print_perf_comparison(layers1, layers2, label1='RunA', label2='RunB')
    out = capsys.readouterr().out

    assert 'LUT1' in out
    assert 'LUT2' in out


@pytest.mark.unit
def test_perf_comparison_no_lut_hides_all_lut_columns(capsys):
    """When neither side has LUT hits, no LUT columns appear."""
    from tools.profiling.compare_layers import _print_perf_comparison

    layers1 = [_make_layer('conv2d', 1.0, uses_perf_lookup=False)]
    layers2 = [_make_layer('conv2d', 1.1, uses_perf_lookup=False)]

    _print_perf_comparison(layers1, layers2)
    out = capsys.readouterr().out

    assert 'LUT1' not in out
    assert 'LUT2' not in out


def _write_trace_csv(path, rows):
    """rows: (op_code, trace_id, session_id, duration_ns or '') -> minimal profiler CSV."""
    import csv
    with open(path, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['OP CODE', 'METAL TRACE ID', 'METAL TRACE REPLAY SESSION ID', 'DEVICE KERNEL DURATION [ns]'])
        w.writerows(rows)


@pytest.mark.unit
def test_trace_replay_step_keeps_eager_tail(tmp_path):
    """A step is a replay session of the first trace plus the eager rows after it (#538/#539).

    llama3 prefill traces only the forward; the lm_head and sampling run eagerly after each replay
    with a blank session id. Selecting by session id alone dropped them.
    """
    from tools.profiling.compare_layers import _detect_trace_replay
    fwd = [('Matmul', '0', None, 100), ('Add', '0', None, 10)]
    tail = [('Matmul', '', '', 50), ('TopK', '', '', 70)]
    rows = [('Tilize', '', '', 5),                         # setup
            ('Matmul', '', '', 100), ('Add', '', '', 10),  # compile run
            ('Matmul', '', '', ''), ('Add', '', '', '')]   # trace capture (no device time)
    for sid in ('1', '2'):
        rows += [(op, tid, sid, d) for op, tid, _, d in fwd] + tail
    path = tmp_path / 'trace.csv'
    _write_trace_csv(path, rows)

    info, steps = _detect_trace_replay(str(path))
    assert info.has_trace_replay and info.n_sessions == 2 and info.ops_per_session == 4
    assert info.setup_op_count == 5                    # setup + compile + capture rows
    assert [len(v) for v in steps.values()] == [4, 4]
    assert info.selected_duration_ns == 230.0          # forward 110 + eager tail 120


@pytest.mark.unit
def test_trace_replay_step_on_merged_csv(tmp_path):
    """A #539-merged CSV is one step: one session's rows followed by its eager tail."""
    from tools.profiling.compare_layers import _detect_trace_replay
    path = tmp_path / 'merged.csv'
    _write_trace_csv(path, [('Matmul', '0', '2', 100), ('Add', '0', '2', 10), ('TopK', '', '', 70)])
    info, steps = _detect_trace_replay(str(path))
    assert info.n_sessions == 1 and info.ops_per_session == 3 and info.setup_op_count == 0
