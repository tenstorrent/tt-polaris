# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import csv
from pathlib import Path

import pytest

from tools.si_profiling_helpers.ops_perf_three_csv_merge import (
    COL_FPU_UTIL_RAW,
    COL_MEM_UTIL_RAW,
    NOCTRACE_ONLY_OUTPUT_COLUMNS,
    MergeError,
    classify_file,
    detect_by_op_code_period,
    detect_iteration_boundaries,
    drop_signpost_rows,
    iteration_summary,
    pick_median_iteration,
    run_merge,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rows(op_codes, kdurs=None, drams=None, fpus=None):
    """Build minimal row dicts for testing."""
    n = len(op_codes)
    kdurs = kdurs or ['100'] * n
    drams = drams or [''] * n
    fpus = fpus or [''] * n
    return [
        {
            'OP CODE': op_codes[i],
            'OP TYPE': 'op',
            'GLOBAL CALL COUNT': str(i),
            'DEVICE KERNEL DURATION [ns]': kdurs[i],
            'DRAM BW UTIL (%)': drams[i],
            'FPU Util Median (%)': fpus[i],
            'SFPU Util Median (%)': '',
            'NOC UTIL (%)': '',
            'MULTICAST NOC UTIL (%)': '',
            'ETH BW UTIL (%)': '',
            'NPE CONG IMPACT (%)': '',
        }
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# detect_iteration_boundaries
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_detect_iteration_boundaries_marker_basic():
    rows = _rows(['conv', 'relu', 'pool', 'conv', 'relu', 'pool'])
    iters = detect_iteration_boundaries(rows)
    assert iters == [(0, 3), (3, 6)]


@pytest.mark.unit
def test_detect_iteration_boundaries_single_iteration():
    rows = _rows(['conv', 'relu', 'pool'])
    iters = detect_iteration_boundaries(rows)
    assert iters == [(0, 3)]


@pytest.mark.unit
def test_detect_iteration_boundaries_fixed_size():
    rows = _rows(['a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c'])
    iters = detect_iteration_boundaries(rows, ops_per_iteration=3)
    assert iters == [(0, 3), (3, 6), (6, 9)]


@pytest.mark.unit
def test_detect_iteration_boundaries_fixed_size_not_divisible():
    rows = _rows(['a', 'b', 'c', 'a', 'b'])
    with pytest.raises(MergeError, match='does not evenly divide'):
        detect_iteration_boundaries(rows, ops_per_iteration=3)


@pytest.mark.unit
def test_detect_iteration_boundaries_unequal_sizes_falls_back_to_single():
    # 'a' recurs intra-pass at irregular positions -> unequal marker chunks.
    # This is a single-pass capture (marker recurs within one pass), so detection
    # falls back to ONE iteration spanning all rows (rather than raising).
    rows = _rows(['a', 'b', 'a', 'b', 'a', 'b', 'c'])
    iters = detect_iteration_boundaries(rows)
    assert iters == [(0, 7)]


def _replay_rows(op_codes_one_iter, n_iters, n_replay):
    """Rows for a trace-replay capture: the op sequence repeats n_iters times.

    The last ``n_replay`` iterations reuse a fixed GCC sequence (as a real trace
    replay does), so their (gcc, op, type) join keys duplicate across iterations.
    The earlier iterations get unique, monotonic GCCs (eager capture/warmup).
    """
    rows = []
    g = 1000
    for it in range(n_iters):
        for j, op in enumerate(op_codes_one_iter):
            if it < n_iters - n_replay:
                gcc = g
                g += 1
            else:
                gcc = 900000 + j  # replay: identical GCCs each replayed iteration
            rows.append({
                'OP CODE': op,
                'OP TYPE': 'op',
                'GLOBAL CALL COUNT': str(gcc),
                'DEVICE KERNEL DURATION [ns]': '100',
                'DRAM BW UTIL (%)': '',
                'FPU Util Median (%)': '',
                'SFPU Util Median (%)': '',
                'NOC UTIL (%)': '',
                'MULTICAST NOC UTIL (%)': '',
                'ETH BW UTIL (%)': '',
                'NPE CONG IMPACT (%)': '',
            })
    return rows


@pytest.mark.unit
def test_op_code_period_bh_like_marker_would_fail():
    # BH VGG pathology: the first op recurs 3x WITHIN each iteration (unequal marker
    # chunks would make the marker heuristic bail to a single iteration), but the op
    # sequence is periodic and trace-replay iterations duplicate join keys. The
    # op-code-period detector must recover the true 4 iterations of 6 ops.
    one_iter = ['Reshard', 'A', 'Reshard', 'B', 'Reshard', 'C']
    rows = _replay_rows(one_iter, n_iters=4, n_replay=2)
    assert detect_iteration_boundaries(rows) == [(0, 6), (6, 12), (12, 18), (18, 24)]


@pytest.mark.unit
def test_op_code_period_no_duplicate_keys_defers_to_marker():
    # Monotonic GCCs (no trace replay) => no duplicate join keys => the period
    # detector declines (returns None) and the marker heuristic handles it.
    rows = _rows(['conv', 'relu', 'pool', 'conv', 'relu', 'pool'])
    assert detect_by_op_code_period(rows) is None
    assert detect_iteration_boundaries(rows) == [(0, 3), (3, 6)]


@pytest.mark.unit
def test_op_code_period_single_iteration_returns_none():
    # One iteration, all unique keys -> nothing to reduce -> None (marker -> single).
    rows = _rows(['conv', 'relu', 'pool'])
    assert detect_by_op_code_period(rows) is None


# ---------------------------------------------------------------------------
# iteration_summary
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_iteration_summary_complete():
    rows = _rows(['a'] * 4, kdurs=['100', '200', '150', '50'])
    iters = [(0, 4)]
    summary = iteration_summary(rows, iters)
    assert len(summary) == 1
    start, end, is_complete, total = summary[0]
    assert (start, end) == (0, 4)
    assert is_complete is True
    assert total == pytest.approx(500.0)


@pytest.mark.unit
def test_iteration_summary_float_duration_counted():
    # '123.0' must be counted — the fix ensures parse_finite_float is used, not int()
    rows = _rows(['a'] * 3, kdurs=['100', '123.0', '50'])
    iters = [(0, 3)]
    summary = iteration_summary(rows, iters)
    _, _, is_complete, total = summary[0]
    assert is_complete is True
    assert total == pytest.approx(273.0)


@pytest.mark.unit
def test_iteration_summary_sparse_incomplete():
    # Only 1/4 rows have a duration — below the 95% threshold
    rows = _rows(['a'] * 4, kdurs=['100', '', '', ''])
    iters = [(0, 4)]
    summary = iteration_summary(rows, iters)
    _, _, is_complete, _ = summary[0]
    assert is_complete is False


@pytest.mark.unit
def test_iteration_summary_dash_treated_as_empty():
    # '-' is non-parseable and must not count toward completeness
    rows = _rows(['a'] * 4, kdurs=['100', '-', '-', '-'])
    iters = [(0, 4)]
    summary = iteration_summary(rows, iters)
    _, _, is_complete, _ = summary[0]
    assert is_complete is False


@pytest.mark.unit
def test_iteration_summary_multiple_iters():
    rows = _rows(['a'] * 6, kdurs=['10', '20', '', '30', '40', '50'])
    iters = [(0, 3), (3, 6)]
    summary = iteration_summary(rows, iters)
    # First iter: 2/3 rows — below 95%
    assert summary[0][2] is False
    # Second iter: 3/3 rows — complete
    assert summary[1][2] is True
    assert summary[1][3] == pytest.approx(120.0)


# ---------------------------------------------------------------------------
# pick_median_iteration
# ---------------------------------------------------------------------------

def _make_summary(totals, complete=None):
    if complete is None:
        complete = [True] * len(totals)
    return [(i * 10, (i + 1) * 10, complete[i], float(totals[i])) for i in range(len(totals))]


@pytest.mark.unit
def test_pick_median_odd():
    # 3 complete iters with totals 30, 10, 20 → sorted [10, 20, 30] → median = 20
    summary = _make_summary([30, 10, 20])
    start, end, idx = pick_median_iteration(summary)
    assert summary[idx][3] == pytest.approx(20.0)


@pytest.mark.unit
def test_pick_median_even_lower_middle():
    # 4 complete iters, totals 10, 40, 20, 30 → sorted [10, 20, 30, 40]
    # lower-middle index = (4-1)//2 = 1 → value 20
    summary = _make_summary([10, 40, 20, 30])
    start, end, idx = pick_median_iteration(summary)
    assert summary[idx][3] == pytest.approx(20.0)


@pytest.mark.unit
def test_pick_min():
    summary = _make_summary([30, 10, 20])
    _, _, idx = pick_median_iteration(summary, select='min')
    assert summary[idx][3] == pytest.approx(10.0)


@pytest.mark.unit
def test_pick_max():
    summary = _make_summary([30, 10, 20])
    _, _, idx = pick_median_iteration(summary, select='max')
    assert summary[idx][3] == pytest.approx(30.0)


@pytest.mark.unit
def test_pick_first():
    summary = _make_summary([30, 10, 20])
    _, _, idx = pick_median_iteration(summary, select='first')
    assert idx == 0


@pytest.mark.unit
def test_pick_last():
    summary = _make_summary([30, 10, 20])
    _, _, idx = pick_median_iteration(summary, select='last')
    assert idx == 2


@pytest.mark.unit
def test_pick_skips_incomplete():
    # Only iter 1 is complete
    summary = _make_summary([30, 10, 20], complete=[False, True, False])
    _, _, idx = pick_median_iteration(summary)
    assert idx == 1


@pytest.mark.unit
def test_pick_no_complete_raises():
    summary = _make_summary([30, 10, 20], complete=[False, False, False])
    with pytest.raises(MergeError, match='No iterations available'):
        pick_median_iteration(summary)


@pytest.mark.unit
def test_pick_measured_indices_override():
    # measured_indices=[0, 2] → candidates are iters 0 and 2 regardless of completeness
    summary = _make_summary([30, 10, 20], complete=[False, True, False])
    _, _, idx = pick_median_iteration(summary, measured_indices=[0, 2], select='min')
    assert summary[idx][3] == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# drop_signpost_rows
# ---------------------------------------------------------------------------

def _signpost_row(op_code):
    # TT-Metal emits signpost markers with OP TYPE 'signpost' and no GLOBAL CALL COUNT.
    return {'OP CODE': op_code, 'OP TYPE': 'signpost', 'GLOBAL CALL COUNT': ''}


@pytest.mark.unit
def test_drop_signpost_rows_removes_markers():
    rows = [_signpost_row('start'), *_rows(['conv', 'relu']), _signpost_row('stop')]
    kept = drop_signpost_rows(rows, Path('x.csv'))
    assert [r['OP CODE'] for r in kept] == ['conv', 'relu']
    assert all(r['OP TYPE'] != 'signpost' for r in kept)


@pytest.mark.unit
def test_drop_signpost_rows_case_insensitive():
    rows = [{'OP CODE': 'start', 'OP TYPE': 'SignPost', 'GLOBAL CALL COUNT': ''}, *_rows(['conv'])]
    assert len(drop_signpost_rows(rows, Path('x.csv'))) == 1


@pytest.mark.unit
def test_drop_signpost_rows_noop_when_absent():
    rows = _rows(['conv', 'relu'])
    kept = drop_signpost_rows(rows, Path('x.csv'))
    assert kept == rows


# ---------------------------------------------------------------------------
# classify_file
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_classify_noctrace():
    rows = _rows(['conv'], drams=['45.2'])
    assert classify_file(rows) == 'noctrace'


@pytest.mark.unit
def test_classify_fpu():
    rows = _rows(['conv'], drams=[''], fpus=['62.1'])
    assert classify_file(rows) == 'fpu'


@pytest.mark.unit
def test_classify_pm_fpu_alone_is_vanilla():
    # PM FPU UTIL (%) is present in all three CSV types and must not be used as
    # a classification signal. A file with only PM FPU UTIL (non-zero) and no
    # FPU Util Median / SFPU Util Median columns must classify as vanilla.
    rows = [
        {
            'OP CODE': 'conv', 'OP TYPE': 'op', 'GLOBAL CALL COUNT': '0',
            'DEVICE KERNEL DURATION [ns]': '100',
            'DRAM BW UTIL (%)': '', 'FPU Util Median (%)': '', 'SFPU Util Median (%)': '',
            'PM FPU UTIL (%)': '55.0',
            'NOC UTIL (%)': '', 'MULTICAST NOC UTIL (%)': '',
            'ETH BW UTIL (%)': '', 'NPE CONG IMPACT (%)': '',
        }
    ]
    assert classify_file(rows) == 'vanilla'


@pytest.mark.unit
def test_classify_vanilla():
    rows = _rows(['conv'], drams=[''], fpus=[''])
    assert classify_file(rows) == 'vanilla'


@pytest.mark.unit
def test_classify_empty_rows():
    assert classify_file([]) == 'unassigned'


# ---------------------------------------------------------------------------
# End-to-end merge: derived overhead-ratio columns
# ---------------------------------------------------------------------------

_E2E_COLS = [
    'GLOBAL CALL COUNT', 'OP CODE', 'OP TYPE', 'DEVICE KERNEL DURATION [ns]',
    'DRAM BW UTIL (%)', 'FPU Util Median (%)', 'SFPU Util Median (%)',
    'NOC UTIL (%)', 'MULTICAST NOC UTIL (%)', 'ETH BW UTIL (%)', 'NPE CONG IMPACT (%)',
]


def _write_csv(path, rows, cols=_E2E_COLS):
    with path.open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, '') for c in cols})


def _e2e_row(gcc, op, ns, dram='', fpu=''):
    return {
        'GLOBAL CALL COUNT': str(gcc), 'OP CODE': op, 'OP TYPE': 'op',
        'DEVICE KERNEL DURATION [ns]': str(ns), 'DRAM BW UTIL (%)': dram,
        'FPU Util Median (%)': fpu, 'SFPU Util Median (%)': '',
        'NOC UTIL (%)': '', 'MULTICAST NOC UTIL (%)': '', 'ETH BW UTIL (%)': '',
        'NPE CONG IMPACT (%)': '',
    }


@pytest.mark.unit
def test_merge_appends_overhead_ratio_columns(tmp_path):
    # Two ops with matching join keys across the three passes. Row A has a normal raw
    # duration; row B has raw=0 to exercise the blank-on-zero guard.
    keys = [(1024, 'Matmul'), (2048, 'Add')]
    # vanilla (raw) ns, fpu ns, noc ns per op
    van = {'Matmul': 1000, 'Add': 0}
    fpu = {'Matmul': 1100, 'Add': 50}
    noc = {'Matmul': 1200, 'Add': 60}

    _write_csv(tmp_path / 'noctrace.csv',
               [_e2e_row(g, op, noc[op], dram='45.2') for g, op in keys])
    _write_csv(tmp_path / 'fpuutil.csv',
               [_e2e_row(g, op, fpu[op], fpu='62.1') for g, op in keys])
    _write_csv(tmp_path / 'vanilla.csv',
               [_e2e_row(g, op, van[op]) for g, op in keys])

    out = tmp_path / 'merged_ops.csv'
    run_merge(tmp_path, out, dram_peak_bw_gbps=288.0, duration_rel_tol=1e9, encoding='utf-8')

    with out.open(newline='') as fh:
        rows = {r['OP CODE']: r for r in csv.DictReader(fh)}

    # Columns appended.
    assert COL_FPU_UTIL_RAW in rows['Matmul']
    assert COL_MEM_UTIL_RAW in rows['Matmul']
    # Row A: ratios = fpu/van and noc/van.
    assert float(rows['Matmul'][COL_FPU_UTIL_RAW]) == pytest.approx(1.1)
    assert float(rows['Matmul'][COL_MEM_UTIL_RAW]) == pytest.approx(1.2)
    # Row B: raw=0 -> blank (no division), not '0' or 'inf'.
    assert rows['Add'][COL_FPU_UTIL_RAW] == ''
    assert rows['Add'][COL_MEM_UTIL_RAW] == ''


def _e2e_signpost(op_code):
    # Signpost marker row as tt-metal writes it: OP TYPE 'signpost', empty GLOBAL CALL COUNT.
    r = {c: '' for c in _E2E_COLS}
    r['OP CODE'] = op_code
    r['OP TYPE'] = 'signpost'
    return r


@pytest.mark.unit
def test_merge_skips_signpost_rows(tmp_path):
    # ResNet50/WH capture pathology: each pass wraps its ops in start/stop Tracy
    # signposts, emitted as rows with an empty GLOBAL CALL COUNT. Pre-fix these
    # crashed join-key parsing ("Empty 'GLOBAL CALL COUNT'"); they must now be
    # dropped so the merge succeeds and they never reach the output.
    keys = [(1024, 'Matmul'), (2048, 'Add')]

    def with_signposts(op_rows):
        return [_e2e_signpost('start'), *op_rows, _e2e_signpost('stop')]

    _write_csv(tmp_path / 'noctrace.csv',
               with_signposts([_e2e_row(g, op, 1200, dram='45.2') for g, op in keys]))
    _write_csv(tmp_path / 'fpuutil.csv',
               with_signposts([_e2e_row(g, op, 1100, fpu='62.1') for g, op in keys]))
    _write_csv(tmp_path / 'vanilla.csv',
               with_signposts([_e2e_row(g, op, 1000) for g, op in keys]))

    out = tmp_path / 'merged_ops.csv'
    run_merge(tmp_path, out, dram_peak_bw_gbps=288.0, duration_rel_tol=1e9, encoding='utf-8')

    with out.open(newline='') as fh:
        rows = list(csv.DictReader(fh))
    assert {r['OP CODE'] for r in rows} == {'Matmul', 'Add'}
    assert all(r['OP TYPE'] != 'signpost' for r in rows)


# fpu-only analysis columns (extras the fpu pass carries beyond the noctrace schema)
_FPU_EXTRAS = ['Packer Efficiency Avg (%)', 'Math Util Median (%)']
_OP_HOST_FUNC = 'TT_DNN_DEVICE_OP_TT_HOST_FUNC [ns]'


def _three_passes(tmp_path, noc_cols, van_cols, fpu_cols):
    """Write the standard 3 passes (one Matmul row) with the given per-file columns."""
    def row(ns, dram='', fpu='', extra=''):
        r = _e2e_row(1024, 'Matmul', ns, dram=dram, fpu=fpu)
        r[_OP_HOST_FUNC] = '5'
        for c in _FPU_EXTRAS:
            r[c] = extra
        return r
    _write_csv(tmp_path / 'noctrace.csv', [row(1200, dram='45.2')], cols=noc_cols)
    _write_csv(tmp_path / 'vanilla.csv', [row(1000)], cols=van_cols)
    _write_csv(tmp_path / 'fpuutil.csv', [row(1100, fpu='62.1', extra='7.5')], cols=fpu_cols)
    out = tmp_path / 'merged_ops.csv'
    run_merge(tmp_path, out, dram_peak_bw_gbps=288.0, duration_rel_tol=1e9, encoding='utf-8')
    with out.open(newline='') as fh:
        r = list(csv.reader(fh))
    return r[0], r[1:]


@pytest.mark.unit
def test_merge_fpu_extras_trailing_old_layout(tmp_path):
    # Backward compat: older tt-metal appends the fpu analysis columns at the very end
    # (fpu header is a strict prefix-extension of noctrace).
    base = _E2E_COLS
    hdr, rows = _three_passes(
        tmp_path, noc_cols=base, van_cols=base, fpu_cols=[*base, *_FPU_EXTRAS]
    )
    assert len(rows) == 1
    for c in _FPU_EXTRAS:
        assert c in hdr  # fpu extras carried through to the merged output
    assert hdr.count(COL_FPU_UTIL_RAW) == 1


@pytest.mark.unit
def test_merge_fpu_extras_spliced_before_shared_trailing(tmp_path):
    # The vit-WH-260623 failure: fpu splices its analysis block BEFORE a trailing
    # op-type column (TT_DNN_DEVICE_OP_TT_HOST_FUNC) that noctrace/vanilla also carry,
    # so noctrace is an ordered subsequence of fpu but not a prefix.
    base_plus = [*_E2E_COLS, _OP_HOST_FUNC]
    fpu_cols = [*_E2E_COLS, *_FPU_EXTRAS, _OP_HOST_FUNC]
    hdr, rows = _three_passes(
        tmp_path, noc_cols=base_plus, van_cols=base_plus, fpu_cols=fpu_cols
    )
    assert len(rows) == 1
    for c in _FPU_EXTRAS:
        assert c in hdr
    # The shared trailing op-type column is carried once (from vanilla), not duplicated.
    assert hdr.count(_OP_HOST_FUNC) == 1
    assert hdr.count(COL_FPU_UTIL_RAW) == 1


# ---------------------------------------------------------------------------
# --allow-missing-noctrace: merge a vanilla CSV and an fpu CSV
# ---------------------------------------------------------------------------

_FPU_UTIL_COLS = ['FPU Util Median (%)', 'SFPU Util Median (%)']
# tt-metal layout: noctrace and vanilla carry no FPU/SFPU util columns; only fpu does.
_BASE_NO_FPU_UTIL = [c for c in _E2E_COLS if c not in _FPU_UTIL_COLS]


def _write_passes(d, base, with_noctrace=True):
    """Write vanilla + fpu (+ noctrace) passes of two ops into ``d`` with ``base`` columns."""
    shared = [*base, _OP_HOST_FUNC]
    fpu_cols = [*base, *(c for c in _FPU_UTIL_COLS if c not in base), *_FPU_EXTRAS, _OP_HOST_FUNC]

    def rows(ns, **kw):
        out = []
        for gcc, op in [(1024, 'Matmul'), (2048, 'Add')]:
            r = _e2e_row(gcc, op, ns, **kw)
            r[_OP_HOST_FUNC] = '5'
            for c in _FPU_EXTRAS:
                r[c] = '7.5'
            out.append(r)
        return out

    d.mkdir()
    if with_noctrace:
        _write_csv(d / 'noctrace.csv', rows(1200, dram='45.2'), cols=shared)
    _write_csv(d / 'fpuutil.csv', rows(1100, fpu='62.1'), cols=fpu_cols)
    _write_csv(d / 'vanilla.csv', rows(1000), cols=shared)


def _merge(d, out, **kw):
    run_merge(d, out, dram_peak_bw_gbps=288.0, duration_rel_tol=1e9, encoding='utf-8', **kw)
    with out.open(newline='') as fh:
        return list(csv.reader(fh))


@pytest.mark.unit
@pytest.mark.parametrize('base', [_BASE_NO_FPU_UTIL, _E2E_COLS], ids=['tt-metal-layout', 'fpu-util-everywhere'])
def test_merge_without_noctrace_is_three_csv_merge_minus_noctrace_columns(tmp_path, base):
    _write_passes(tmp_path / 'three', base)
    _write_passes(tmp_path / 'two', base, with_noctrace=False)
    full = _merge(tmp_path / 'three', tmp_path / 'three.csv')
    two = _merge(tmp_path / 'two', tmp_path / 'two.csv', allow_missing_noctrace=True)

    keep = [i for i, c in enumerate(full[0]) if c not in NOCTRACE_ONLY_OUTPUT_COLUMNS]
    assert len(full[0]) - len(keep) == len(NOCTRACE_ONLY_OUTPUT_COLUMNS)
    assert two == [[r[i] for i in keep] for r in full]


@pytest.mark.unit
def test_allow_missing_noctrace_with_three_csvs_is_byte_identical(tmp_path):
    _write_passes(tmp_path / 'in', _BASE_NO_FPU_UTIL)
    _merge(tmp_path / 'in', tmp_path / 'plain.csv')
    _merge(tmp_path / 'in', tmp_path / 'flag.csv', allow_missing_noctrace=True)
    assert (tmp_path / 'plain.csv').read_bytes() == (tmp_path / 'flag.csv').read_bytes()


@pytest.mark.unit
def test_missing_noctrace_without_flag_names_the_flag(tmp_path):
    _write_passes(tmp_path / 'in', _BASE_NO_FPU_UTIL, with_noctrace=False)
    with pytest.raises(MergeError, match='--allow-missing-noctrace'):
        _merge(tmp_path / 'in', tmp_path / 'out.csv')


@pytest.mark.unit
def test_allow_missing_noctrace_still_requires_fpu_and_vanilla(tmp_path):
    _write_passes(tmp_path / 'in', _BASE_NO_FPU_UTIL)
    (tmp_path / 'in' / 'fpuutil.csv').unlink()
    with pytest.raises(MergeError, match='Expected exactly one fpu CSV, found 0'):
        _merge(tmp_path / 'in', tmp_path / 'out.csv', allow_missing_noctrace=True)
