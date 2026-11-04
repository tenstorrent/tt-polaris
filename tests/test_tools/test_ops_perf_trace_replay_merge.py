# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Tests for the trace+replay merge sibling (ops_perf_trace_replay_merge.py).

Focus is the reduction step that differs from the iterative tool: dropping the
prologue and capture rows and selecting one step so join keys are unique.
The shared classify/join/output machinery is covered by
test_ops_perf_three_csv_merge.py and is exercised here only end-to-end.
"""

import csv

import pytest

from tools.si_profiling_helpers.ops_perf_three_csv_merge import NOCTRACE_ONLY_OUTPUT_COLUMNS
from tools.si_profiling_helpers.ops_perf_trace_replay_merge import (
    COL_REPLAY_SESSION_ID,
    COL_TRACE_ID,
    MergeError,
    main,
    reduce_trace_replay,
    select_step,
    split_steps,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TR_COLS = [
    'GLOBAL CALL COUNT', 'OP CODE', 'OP TYPE', 'DEVICE KERNEL DURATION [ns]',
    'DRAM BW UTIL (%)', 'FPU Util Median (%)', 'SFPU Util Median (%)',
    'NOC UTIL (%)', 'MULTICAST NOC UTIL (%)', 'ETH BW UTIL (%)', 'NPE CONG IMPACT (%)',
    COL_REPLAY_SESSION_ID, COL_TRACE_ID,
]


def _tr_row(gcc, op, ns, rsid, dram='', fpu='', tid=''):
    return {
        'GLOBAL CALL COUNT': str(gcc), 'OP CODE': op, 'OP TYPE': 'op',
        'DEVICE KERNEL DURATION [ns]': str(ns), 'DRAM BW UTIL (%)': dram,
        'FPU Util Median (%)': fpu, 'SFPU Util Median (%)': '',
        'NOC UTIL (%)': '', 'MULTICAST NOC UTIL (%)': '', 'ETH BW UTIL (%)': '',
        'NPE CONG IMPACT (%)': '', COL_REPLAY_SESSION_ID: rsid,
        COL_TRACE_ID: tid if rsid else '',
    }


def _keys(rows):
    return [(r['GLOBAL CALL COUNT'], r['OP CODE'], r['OP TYPE']) for r in rows]


def _is_capture(r):
    return not r[COL_REPLAY_SESSION_ID] and not r['DEVICE KERNEL DURATION [ns]']


class _Gcc:
    """Fresh, monotonically increasing GLOBAL CALL COUNTs for eager ops."""

    def __init__(self, start=5000):
        self.n = start

    def __call__(self):
        self.n += 1
        return self.n


def _prefill_like(tails, replay=(('1024', 'Emb'), ('2048', 'Layer')), tail_ns=50):
    """A prefill-shaped capture: a prologue (setup + eager compile + capture rows),
    then one step per entry of ``tails``: a replay session of trace 0 followed by
    that tail's eager ops (fresh GCCs).  Replay sessions reuse the replay GCCs.
    """
    gcc = _Gcc()
    rows = [_tr_row(gcc(), 'Tilize', 1, '')]  # one-time setup
    rows += [_tr_row(gcc(), op, 9, '') for _, op in replay]  # eager compile run
    rows += [_tr_row(g, op, '', '') for g, op in replay]  # capture rows: no duration
    for i, tail in enumerate(tails, start=1):
        rows += [_tr_row(g, op, 10, str(i), tid='0') for g, op in replay]
        rows += [_tr_row(gcc(), op, tail_ns, '') for op in tail]
    return rows


def _capture_rows(ns_by_session, ops=(('1024', 'Matmul'), ('2048', 'Add')), dram='', fpu=''):
    """Build a trace+replay capture: a compile pass (rsid='') plus one block per
    replay session, every block carrying the same ops/join keys.

    ``ns_by_session`` maps session id ('' for compile) -> per-op kernel ns, so a
    session's total (and thus median/min/max selection) is controllable.
    """
    rows = []
    for sid, ns in ns_by_session.items():
        for gcc, op in ops:
            rows.append(_tr_row(gcc, op, ns, sid, dram=dram, fpu=fpu))
    return rows


# ---------------------------------------------------------------------------
# split_steps / select_step
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_split_steps_one_step_per_session_prologue_dropped():
    rows = _capture_rows({'': 999, '1': 100, '2': 200})
    steps = split_steps(rows)
    assert [{r[COL_REPLAY_SESSION_ID] for r in s} for s in steps] == [{'1'}, {'2'}]


@pytest.mark.unit
def test_select_step_median_lower_middle():
    # step 0 total < step 1 total -> median (lower-middle of 2) picks step 0.
    steps = split_steps(_capture_rows({'': 999, '1': 100, '2': 200}))
    assert select_step(steps, 'median') == 0


@pytest.mark.unit
def test_select_step_min_max():
    steps = split_steps(_capture_rows({'': 999, '1': 100, '2': 200}))
    assert select_step(steps, 'min') == 0
    assert select_step(steps, 'max') == 1


@pytest.mark.unit
def test_select_step_first_last_capture_order():
    steps = split_steps(_capture_rows({'': 999, '2': 200, '1': 100}))  # session 2 appears first
    assert steps[select_step(steps, 'first')][0][COL_REPLAY_SESSION_ID] == '2'
    assert steps[select_step(steps, 'last')][0][COL_REPLAY_SESSION_ID] == '1'


@pytest.mark.unit
def test_split_steps_no_sessions_raises():
    rows = _capture_rows({'': 999})  # compile pass only, no replay session
    with pytest.raises(MergeError, match='trace\\+replay'):
        split_steps(rows)


@pytest.mark.unit
@pytest.mark.parametrize('select', ['last', 'first', 'median', 'min', 'max'])
def test_select_step_never_picks_a_non_modal_step(select):
    # Warmup-like first steps carry a longer tail; with any selection the chosen
    # step has the most frequent signature.
    rows = _prefill_like([['Norm', 'Pen', 'TopK'], ['Norm', 'TopK'], ['Norm', 'TopK'], ['Norm', 'TopK']])
    steps = split_steps(rows)
    sigs = [tuple(r['OP CODE'] for r in s) for s in steps]
    modal = max(set(sigs), key=sigs.count)
    assert sigs[select_step(steps, select)] == modal


@pytest.mark.unit
def test_select_step_last_not_modal_raises():
    rows = _prefill_like([['Norm', 'TopK'], ['Norm', 'TopK'], ['Norm', 'TopK', 'Extra']])
    with pytest.raises(MergeError, match='last step'):
        select_step(split_steps(rows), 'last')


@pytest.mark.unit
def test_select_step_tie_raises():
    rows = _prefill_like([['Norm', 'Pen', 'TopK'], ['Norm', 'TopK']])
    with pytest.raises(MergeError, match='most frequent'):
        select_step(split_steps(rows), 'median')


# ---------------------------------------------------------------------------
# reduce_trace_replay
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_reduce_trace_replay_drops_compile_and_selects_one_session():
    rows = _capture_rows({'': 999, '1': 100, '2': 200})
    noc, fpu, van = reduce_trace_replay(list(rows), list(rows), list(rows))
    # 2 ops kept (one step), compile pass and the other session dropped.
    assert len(van) == 2
    # the default selects the first candidate step -> session '1', ns=100.
    assert {r[COL_REPLAY_SESSION_ID] for r in van} == {'1'}
    assert len(set(_keys(van))) == len(van)
    assert noc is not None
    assert len(noc) == len(fpu) == len(van)


@pytest.mark.unit
def test_reduce_trace_replay_step_includes_eager_tail():
    rows = _prefill_like([['Norm', 'TopK'], ['Norm', 'TopK']])
    _, _, van = reduce_trace_replay(list(rows), list(rows), list(rows), select_session='last')
    last_replay_row = max(i for i, r in enumerate(rows) if r[COL_REPLAY_SESSION_ID] == '2')
    # The kept rows are the last replay session followed by the eager rows after it.
    assert van == [r for r in rows if r[COL_REPLAY_SESSION_ID] == '2'] + rows[last_replay_row + 1:]
    assert len(set(_keys(van))) == len(van)


@pytest.mark.unit
def test_reduce_trace_replay_drops_prologue_and_capture_rows():
    # Every step also carries a capture row (e.g. another trace being captured);
    # neither those nor any prologue row may reach the output.
    rows = _prefill_like([['Norm', 'TopK'], ['Norm', 'TopK'], ['Norm', 'TopK']])
    first_replay = next(i for i, r in enumerate(rows) if r[COL_REPLAY_SESSION_ID])
    prologue = rows[:first_replay]
    for i in reversed([i for i, r in enumerate(rows) if r['OP CODE'] == 'Norm']):
        rows.insert(i, _tr_row(90000 + i, 'Captured', '', ''))
    for select in ('last', 'first', 'median'):
        _, _, van = reduce_trace_replay(list(rows), list(rows), list(rows), select_session=select)
        assert not any(_is_capture(r) for r in van)
        assert not any(r in van for r in prologue)


@pytest.mark.unit
@pytest.mark.parametrize('bad_ns', ['-', 'nan', 'inf', 'n/a'])
def test_eager_row_with_malformed_duration_is_kept(bad_ns):
    # Only a blank duration cell marks a capture row. An eager tail row with a
    # malformed duration must stay in its step, so the merge's duration check
    # reports it instead of the row vanishing from the output.
    rows = _prefill_like([['Norm', 'TopK'], ['Norm', 'TopK']])
    last_topk = max(i for i, r in enumerate(rows) if r['OP CODE'] == 'TopK')
    rows[last_topk]['DEVICE KERNEL DURATION [ns]'] = bad_ns
    _, _, van = reduce_trace_replay(list(rows), list(rows), list(rows), select_session='last')
    assert rows[last_topk] in van


@pytest.mark.unit
def test_main_reports_malformed_eager_duration(tmp_path, capsys):
    # End to end: an op whose duration is malformed in every step keeps all steps'
    # signatures identical, so dropping it as a capture row would merge silently
    # without that op. Instead it must reach the duration validation and fail loudly.
    def capture(**kw):
        rows = _prefill_like([['Norm', 'TopK'], ['Norm', 'TopK']])
        for r in rows:
            r.update(kw)
            if r['OP CODE'] == 'TopK':
                r['DEVICE KERNEL DURATION [ns]'] = '-'
        return rows

    _write_csv(tmp_path / 'trace.csv', capture(**{'DRAM BW UTIL (%)': '45.2'}))
    _write_csv(tmp_path / 'perf.csv', capture(**{'FPU Util Median (%)': '62.1'}))
    _write_csv(tmp_path / 'raw.csv', capture())
    rc = main(['--input-dir', str(tmp_path), '--dram-peak-bw-gbps', '288.0',
               '--output', str(tmp_path / 'merged_ops.csv'), '--duration-rel-tol', '1e9'])
    assert rc == 1
    assert 'DEVICE KERNEL DURATION [ns]' in capsys.readouterr().err
    assert not (tmp_path / 'merged_ops.csv').exists()


@pytest.mark.unit
def test_reduce_trace_replay_no_eager_rows_matches_session_filter():
    # Decode-shaped: each step is a replay of trace 0 then a replay of trace 1
    # (traced sampling); no eager rows between sessions.  The step reducer must
    # reproduce a plain filter on the selected session id.
    gcc = _Gcc()
    fwd, samp = [('1024', 'Emb'), ('2048', 'PlusOne')], [('3072', 'Sampling')]
    rows = [_tr_row(gcc(), op, 9, '') for _, op in fwd]  # eager compile
    rows += [_tr_row(g, op, '', '') for g, op in fwd]  # capture trace 0
    rows += [_tr_row(gcc(), op, 9, '') for _, op in samp]  # eager sampling
    rows += [_tr_row(g, op, '', '') for g, op in samp]  # capture trace 1
    for sid in ('1', '2'):
        rows += [_tr_row(g, op, 10, sid, tid='0') for g, op in fwd]
        rows += [_tr_row(g, op, 20, sid, tid='1') for g, op in samp]
    for select in ('last', 'first'):
        _, _, van = reduce_trace_replay(list(rows), list(rows), list(rows), select_session=select)
        sid = '2' if select == 'last' else '1'
        assert van == [r for r in rows if r[COL_REPLAY_SESSION_ID] == sid]


@pytest.mark.unit
def test_reduce_trace_replay_step_count_mismatch_raises():
    rows = _prefill_like([['Norm'], ['Norm']])
    short = _prefill_like([['Norm']])
    with pytest.raises(MergeError, match='Step count differs'):
        reduce_trace_replay(list(rows), list(rows), short)


@pytest.mark.unit
def test_reduce_trace_replay_requires_rsid_column():
    row = {'GLOBAL CALL COUNT': '1', 'OP CODE': 'Matmul', 'OP TYPE': 'op',
           'DEVICE KERNEL DURATION [ns]': '10'}  # no replay-session column
    with pytest.raises(MergeError, match='trace\\+replay'):
        reduce_trace_replay([row], [row], [row])


@pytest.mark.unit
def test_reduce_trace_replay_compose_iteration_within_session():
    # A single replay session that itself holds two iterations (op sequence AB
    # repeated with per-position join keys) -> keys not unique -> the iteration
    # reducer composes to collapse it to one iteration.
    seq = [('1024', 'A'), ('2048', 'B')]
    rows = []
    for _ in range(2):  # two iterations within session '1'
        for gcc, op in seq:
            rows.append(_tr_row(gcc, op, 100, '1'))
    noc, fpu, van = reduce_trace_replay(list(rows), list(rows), list(rows))
    assert len(van) == 2  # collapsed to a single iteration
    keys = [(r['GLOBAL CALL COUNT'], r['OP CODE'], r['OP TYPE']) for r in van]
    assert len(keys) == len(set(keys))


@pytest.mark.unit
def test_reduce_trace_replay_no_compose_raises_on_duplicate_keys():
    seq = [('1024', 'A'), ('2048', 'B')]
    rows = []
    for _ in range(2):
        for gcc, op in seq:
            rows.append(_tr_row(gcc, op, 100, '1'))
    with pytest.raises(MergeError, match='no-compose-iteration'):
        reduce_trace_replay(list(rows), list(rows), list(rows), compose_iteration=False)


# ---------------------------------------------------------------------------
# End-to-end via main(): reproduces the duplicate-key failure and its fix
# ---------------------------------------------------------------------------

def _write_csv(path, rows):
    with path.open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=_TR_COLS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, '') for c in _TR_COLS})


@pytest.mark.unit
def test_main_merges_trace_replay_capture(tmp_path):
    # Compile pass + two replay sessions, all sharing join keys -> the iterative
    # tool would abort on a duplicate key; the trace+replay tool keeps one session.
    _write_csv(tmp_path / 'trace.csv',
               _capture_rows({'': 999, '1': 100, '2': 200}, dram='45.2'))
    _write_csv(tmp_path / 'perf.csv',
               _capture_rows({'': 999, '1': 100, '2': 200}, fpu='62.1'))
    _write_csv(tmp_path / 'raw.csv',
               _capture_rows({'': 999, '1': 100, '2': 200}))

    out = tmp_path / 'merged_ops.csv'
    rc = main([
        '--input-dir', str(tmp_path), '--dram-peak-bw-gbps', '288.0',
        '--output', str(out), '--duration-rel-tol', '1e9',
    ])
    assert rc == 0

    with out.open(newline='') as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 2  # one step, compile pass dropped
    keys = [(r['GLOBAL CALL COUNT'], r['OP CODE'], r['OP TYPE']) for r in rows]
    assert len(keys) == len(set(keys))  # unique -> join did not abort
    # the default selects the first candidate step (session '1'); its vanilla ns is canonical.
    assert {r['DEVICE KERNEL DURATION [ns]'] for r in rows} == {'100'}


def _tr_signpost(name):
    # Tracy signpost marker as TT-Metal emits it: OP TYPE 'signpost', empty GLOBAL CALL
    # COUNT and no replay session.
    r = {c: '' for c in _TR_COLS}
    r['OP CODE'] = name
    r['OP TYPE'] = 'signpost'
    return r


@pytest.mark.unit
def test_main_drops_signpost_rows_in_trace_replay(tmp_path):
    # A trace+replay capture whose passes are wrapped in start/stop Tracy signposts
    # (empty GLOBAL CALL COUNT). The signpost drop lives in the shared run_merge, so it
    # must cover the trace+replay variant too: the merge must succeed and the marker
    # rows must never reach the output. (Same root cause as the ResNet50 3-CSV failure.)
    def wrapped(**kw):
        capture = _capture_rows({'': 999, '1': 100, '2': 200}, **kw)
        return [_tr_signpost('start'), *capture, _tr_signpost('stop')]

    _write_csv(tmp_path / 'trace.csv', wrapped(dram='45.2'))
    _write_csv(tmp_path / 'perf.csv', wrapped(fpu='62.1'))
    _write_csv(tmp_path / 'raw.csv', wrapped())

    out = tmp_path / 'merged_ops.csv'
    rc = main([
        '--input-dir', str(tmp_path), '--dram-peak-bw-gbps', '288.0',
        '--output', str(out), '--duration-rel-tol', '1e9',
    ])
    assert rc == 0

    with out.open(newline='') as fh:
        rows = list(csv.DictReader(fh))
    assert {r['OP CODE'] for r in rows} == {'Matmul', 'Add'}
    assert all(r['OP TYPE'] != 'signpost' for r in rows)


@pytest.mark.unit
def test_main_merges_without_noctrace_with_flag(tmp_path):
    # A prefill-shaped capture with no NOC-trace pass: the merge keeps the same step and
    # rows as the three-CSV merge and omits exactly the noctrace-only columns.
    capture = _prefill_like([['Norm', 'LMHead'], ['Norm', 'LMHead']])
    for d, files in (('three', ('trace', 'perf', 'raw')), ('two', ('perf', 'raw'))):
        (tmp_path / d).mkdir()
        kw = {'trace': {'DRAM BW UTIL (%)': '45.2'}, 'perf': {'FPU Util Median (%)': '62.1'}, 'raw': {}}
        for name in files:
            _write_csv(tmp_path / d / f'{name}.csv', [{**r, **kw[name]} for r in capture])

    def merge(d, *extra):
        out = tmp_path / f'{d}.csv'
        rc = main(['--input-dir', str(tmp_path / d), '--dram-peak-bw-gbps', '288.0',
                   '--output', str(out), '--duration-rel-tol', '1e9', *extra])
        assert rc == 0
        with out.open(newline='') as fh:
            return list(csv.reader(fh))

    full = merge('three')
    two = merge('two', '--allow-missing-noctrace')
    keep = [i for i, c in enumerate(full[0]) if c not in NOCTRACE_ONLY_OUTPUT_COLUMNS]
    assert len(full[0]) - len(keep) == len(NOCTRACE_ONLY_OUTPUT_COLUMNS)
    assert two == [[r[i] for i in keep] for r in full]


@pytest.mark.unit
def test_reduce_trace_replay_without_noctrace_rows():
    rows = _prefill_like([['Norm'], ['Norm']])
    noc, fpu, van = reduce_trace_replay(None, list(rows), list(rows))
    assert noc is None
    assert _keys(fpu) == _keys(van) == _keys(reduce_trace_replay(list(rows), list(rows), list(rows))[2])


@pytest.mark.unit
def test_main_errors_on_non_trace_replay_capture(tmp_path):
    # No populated replay-session id -> not a trace+replay capture -> exit 1.
    _write_csv(tmp_path / 'trace.csv', _capture_rows({'': 100}, dram='45.2'))
    _write_csv(tmp_path / 'perf.csv', _capture_rows({'': 100}, fpu='62.1'))
    _write_csv(tmp_path / 'raw.csv', _capture_rows({'': 100}))
    rc = main(['--input-dir', str(tmp_path), '--dram-peak-bw-gbps', '288.0'])
    assert rc == 1
