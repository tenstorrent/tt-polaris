#!/usr/bin/env python3
# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Merge three TT-Metal ops perf CSVs from a **non-iterative trace+replay** capture.

Specification: doc/SPEC_ops_perf_trace_replay_merge.md

This is the trace+replay sibling of ``ops_perf_three_csv_merge.py`` (the iterative,
VGG-style merge).  It reuses that module's classify / header-check / join / output
machinery verbatim and only swaps the *row-reduction* step: instead of detecting
and selecting one iteration, it selects one **step** (defined below), so the join
keys are unique.

Why a separate reducer
----------------------
A trace+replay capture (ViT, llama3 decode/prefill) is *not* the concatenation of
equal-size iterations that the iterative reducer expects.  It is a prologue (setup
ops, an eager compile run of the traced region, and the trace-capture rows)
followed by N replays of the captured trace.  TT-Metal tags each replayed op with
``METAL TRACE ID`` and ``METAL TRACE REPLAY SESSION ID``; replays of one trace reuse
identical ``(GLOBAL CALL COUNT, OP CODE, OP TYPE)`` join keys, while eager ops always
get fresh ones.

Not every op of a model step is traced.  In llama3 prefill only the embeddings and
decoder layers are traced; the final norm, LM head and sampling run eagerly after
each replay and carry a blank session id.  In llama3 decode, sampling is a second
trace replayed right after the first.  So one step is not one replay session:

* **capture row** — blank session id and blank device kernel duration (recorded
  while a trace is captured; nothing ran on device).  Always dropped.
* **step** — one replay session of the *first replayed trace*, plus every following
  row up to that trace's next replay session (eager rows and other traces' replays
  included, capture rows excluded).  Rows before the first step are the prologue
  and are dropped.
* **step signature** — the step's ordered ``OP CODE`` sequence.

Only steps with the most frequent signature are candidates, which excludes warmup
variants (e.g. first steps with sampling penalties) and steps that contain another
trace's compile run.  The default selection is the **first** candidate step.  In
llama3 decode NOC traces, the last replay session under-reports DRAM traffic while
its kernel time is unchanged; the first candidate carries plausible DRAM values.
``last`` selects the demo's measured call (the final one) when that is wanted.

Composition (trace+replay+iterative)
------------------------------------
If a future capture is trace+replay **and** iterative (a step that itself contains
several concatenated iterations), the selected step's join keys will not be unique.
In that case (unless ``--no-compose-iteration``) the iterative reducer is applied
*within* the selected step, composing the two reductions.  For the common
one-inference-per-step case the keys are already unique and no iteration reduction
runs (so the iterative marker heuristic — and its warning — is never triggered).

Uses stdlib **csv**; **loguru** for warnings/info.  Exit 0 on success, 1 on error.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from loguru import logger

# Reuse the shared machinery from the iterative merge tool verbatim.  Only the
# row-reduction strategy differs (replay-session selection vs iteration selection).
# Dual import so this works both as a package module under pytest / `mypy ./`
# (imported as tools.si_profiling_helpers.*) and as an on-device script, where the
# rsynced bundle is a flat directory of scripts with no package path (the bare name
# resolves because the script's own directory is sys.path[0]).  The package import
# is listed first so `mypy ./` (run from the repo root) resolves the module.
try:
    from tools.si_profiling_helpers import ops_perf_three_csv_merge as _merge
except ModuleNotFoundError:  # on-device flat script bundle: no package path
    import ops_perf_three_csv_merge as _merge  # type: ignore[import-not-found, no-redef]

COL_DEVICE_KERNEL_NS = _merge.COL_DEVICE_KERNEL_NS
COL_GLOBAL_CALL_COUNT = _merge.COL_GLOBAL_CALL_COUNT
COL_OP_CODE = _merge.COL_OP_CODE
COL_OP_TYPE = _merge.COL_OP_TYPE
MergeError = _merge.MergeError
parse_finite_float = _merge.parse_finite_float
reduce_iterative = _merge.reduce_iterative
run_merge = _merge.run_merge
_strip_cell = _merge._strip_cell

# TT-Metal tags each replayed op with its trace and replay session; eager ops and
# capture rows carry a blank session id.
COL_REPLAY_SESSION_ID = "METAL TRACE REPLAY SESSION ID"
COL_TRACE_ID = "METAL TRACE ID"

SESSION_SELECT_CHOICES: Tuple[str, ...] = ("first", "last", "median", "min", "max")


def _session_id(row: Dict[str, str]) -> str:
    """Stripped replay-session id for a row ('' = eager op or capture row)."""
    return _strip_cell(row.get(COL_REPLAY_SESSION_ID))


def _trace_id(row: Dict[str, str]) -> str:
    """Stripped trace id for a row ('' when the column is absent or blank)."""
    return _strip_cell(row.get(COL_TRACE_ID))


def _is_capture_row(row: Dict[str, str]) -> bool:
    """True for a trace-capture row: blank session id and a blank device kernel duration.

    Only a blank duration cell marks a capture row. An eager row with a malformed duration
    ('-', 'nan', non-numeric) is kept, so the merge's duration validation reports it.
    """
    return not _session_id(row) and not _strip_cell(row.get(COL_DEVICE_KERNEL_NS))


def split_steps(rows: Sequence[Dict[str, str]]) -> List[List[Dict[str, str]]]:
    """Split a capture into steps; the prologue and all capture rows are dropped.

    A step starts at each new replay session of the first replayed trace and runs
    up to that trace's next replay session, so it carries the eager rows and other
    traces' replays that follow its session (see the module docstring).
    """
    first_trace = next((_trace_id(r) for r in rows if _session_id(r)), None)
    if first_trace is None:
        raise MergeError(
            f"No populated {COL_REPLAY_SESSION_ID!r} values found — this does not look like a "
            "trace+replay capture. Use ops_perf_three_csv_merge.py for iterative (VGG-style) runs."
        )
    steps: List[List[Dict[str, str]]] = []
    current_sid: Optional[str] = None
    for r in rows:
        sid = _session_id(r)
        if sid and _trace_id(r) == first_trace and sid != current_sid:
            current_sid = sid
            steps.append([])
        if steps and not _is_capture_row(r):
            steps[-1].append(r)
    return steps


def _step_signature(step: Sequence[Dict[str, str]]) -> Tuple[str, ...]:
    """Ordered OP CODE sequence of a step."""
    return tuple(_strip_cell(r.get(COL_OP_CODE)) for r in step)


def _step_total_kdur_ns(step: Sequence[Dict[str, str]]) -> float:
    """Sum of DEVICE KERNEL DURATION [ns] over the rows of one step."""
    total = 0.0
    for r in step:
        v = parse_finite_float(r.get(COL_DEVICE_KERNEL_NS))
        if v is not None:
            total += v
    return total


def select_step(steps: Sequence[Sequence[Dict[str, str]]], select: str) -> int:
    """Return the index of the step to keep, chosen from the vanilla CSV's steps.

    Candidates are the steps with the most frequent step signature.
      * ``first`` — the first candidate (the default).
      * ``last`` — the last step; it must have the most frequent signature.
      * ``median`` — lower-middle candidate by total kernel ns.
      * ``min`` / ``max`` — least / greatest candidate total kernel ns.
    """
    sigs = [_step_signature(s) for s in steps]
    ranked = Counter(sigs).most_common(2)
    if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
        raise MergeError(
            f"No single most frequent step signature among {len(steps)} step(s) "
            f"(top two occur {ranked[0][1]} time(s) each); cannot choose a representative step."
        )
    modal = ranked[0][0]
    candidates = [i for i, s in enumerate(sigs) if s == modal]
    if select == "last":
        if sigs[-1] != modal:
            raise MergeError(
                f"The last step ({len(sigs[-1])} ops) does not have the most frequent step signature "
                f"({len(modal)} ops, {len(candidates)} of {len(steps)} steps); refusing to select it."
            )
        return len(steps) - 1
    if select == "first":
        return candidates[0]
    totals = [(i, _step_total_kdur_ns(steps[i])) for i in candidates]
    if select == "min":
        return min(totals, key=lambda x: x[1])[0]
    if select == "max":
        return max(totals, key=lambda x: x[1])[0]
    if select == "median":
        ordered = sorted(totals, key=lambda x: x[1])
        return ordered[(len(ordered) - 1) // 2][0]  # lower-middle for even counts
    raise MergeError(f"Unknown --select-session value: {select!r}")


def _keys_unique(rows: Sequence[Dict[str, str]]) -> bool:
    """True if every (GLOBAL CALL COUNT, OP CODE, OP TYPE) join key in rows is distinct."""
    seen: set = set()
    for r in rows:
        k = (
            _strip_cell(r.get(COL_GLOBAL_CALL_COUNT)),
            _strip_cell(r.get(COL_OP_CODE)),
            _strip_cell(r.get(COL_OP_TYPE)),
        )
        if k in seen:
            return False
        seen.add(k)
    return True


def reduce_trace_replay(
    rows_noc: List[Dict[str, str]],
    rows_fpu: List[Dict[str, str]],
    rows_van: List[Dict[str, str]],
    *,
    select_session: str = "first",
    compose_iteration: bool = True,
    cli_ops_per_iteration: Optional[int] = None,
    cli_measured_indices: Optional[List[int]] = None,
    cli_select_iteration: str = "median",
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]:
    """Reduce a trace+replay capture to a single step.

    Split each variant into steps (dropping the prologue and capture rows), choose
    one step from the vanilla CSV, and keep the same step in all three variants
    (they share step layout).  If the kept step still has duplicate join keys (a
    trace+replay *and* iterative capture) and ``compose_iteration`` is set, apply
    the iterative reducer within the step to collapse it to one iteration.
    """
    if COL_REPLAY_SESSION_ID not in (rows_van[0] if rows_van else {}):
        raise MergeError(
            f"Vanilla CSV has no {COL_REPLAY_SESSION_ID!r} column — not a trace+replay capture. "
            "Use ops_perf_three_csv_merge.py for iterative (VGG-style) runs."
        )

    steps_noc, steps_fpu, steps_van = split_steps(rows_noc), split_steps(rows_fpu), split_steps(rows_van)
    if not len(steps_noc) == len(steps_fpu) == len(steps_van):
        raise MergeError(
            f"Step count differs across variants (noctrace={len(steps_noc)} fpu={len(steps_fpu)} "
            f"vanilla={len(steps_van)}); step layouts must be synchronized across the three variants."
        )
    chosen = select_step(steps_van, select_session)
    chosen_sid = _session_id(steps_van[chosen][0])
    n_prologue = next(i for i, r in enumerate(rows_van) if _session_id(r))
    rows_noc, rows_fpu, rows_van = steps_noc[chosen], steps_fpu[chosen], steps_van[chosen]
    logger.info(
        "Step filter: {} step(s); dropped {} prologue row(s); selected step {} (replay session {!r}, "
        "select={!r}) -> {} op(s) per variant.",
        len(steps_van),
        n_prologue,
        chosen,
        chosen_sid,
        select_session,
        len(rows_van),
    )

    if not _keys_unique(rows_van):
        if compose_iteration:
            logger.info(
                "Selected step {} has duplicate join keys "
                "(trace+replay+iterative); composing the iteration reducer within the step.",
                chosen,
            )
            rows_noc, rows_fpu, rows_van = reduce_iterative(
                rows_noc,
                rows_fpu,
                rows_van,
                cli_ops_per_iteration=cli_ops_per_iteration,
                cli_measured_indices=cli_measured_indices,
                cli_select_iteration=cli_select_iteration,
            )
        else:
            raise MergeError(
                f"Selected step {chosen} has duplicate join keys and "
                "--no-compose-iteration was given; the join requires unique keys. "
                "Re-run without --no-compose-iteration to reduce iterations within the step."
            )
    return rows_noc, rows_fpu, rows_van


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Merge three TT-Metal ops perf CSVs from a non-iterative trace+replay capture "
            "(ViT, llama3) under --input-dir into one CSV, keeping a single step."
        ),
        epilog=(
            "Default --output is <input-dir>/merged_ops.csv. For iterative (VGG-style) runs use "
            "ops_perf_three_csv_merge.py instead. See doc/SPEC_ops_perf_trace_replay_merge.md."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help=(
            "Root directory; recursively discovers exactly three ops_perf_results_*.csv files "
            "(falls back to *.csv when none match), classified by content."
        ),
    )
    p.add_argument(
        "--dram-peak-bw-gbps",
        type=float,
        required=True,
        help="Peak DRAM bandwidth in GB/s (used with DRAM BW UTIL (%%) for MemTraffic).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: <input-dir>/merged_ops.csv).",
    )
    p.add_argument(
        "--duration-rel-tol",
        type=float,
        default=0.05,
        help="Relative tolerance comparing DEVICE KERNEL DURATION [ns] between fpu and vanilla rows.",
    )
    p.add_argument(
        "--encoding",
        default="utf-8",
        help="Text encoding for input and output CSV files.",
    )
    p.add_argument(
        "--select-session",
        choices=SESSION_SELECT_CHOICES,
        default="first",
        help=(
            "How to choose the step to keep. 'first', 'median' (lower-middle by total kernel ns), "
            "'min' and 'max' select among the steps with the most frequent step signature. "
            "'last' selects the last step (the demo's measured call), which must have the most "
            "frequent step signature. The prologue and capture rows are always dropped."
        ),
    )
    p.add_argument(
        "--no-compose-iteration",
        action="store_true",
        help=(
            "Disable the iteration reducer that otherwise runs when the selected step still has "
            "duplicate join keys (a trace+replay+iterative capture). With this flag such a capture "
            "is a hard error instead."
        ),
    )
    # Forwarded to the composed iteration reducer; only take effect for a
    # trace+replay+iterative capture (a step with duplicate keys).
    p.add_argument(
        "--ops-per-iteration",
        type=int,
        default=None,
        help="Ops per iteration for the composed iteration reducer (trace+replay+iterative only).",
    )
    p.add_argument(
        "--measured-iteration-indices",
        type=str,
        default=None,
        help="Comma-separated 0-based iteration indices for the composed iteration reducer.",
    )
    p.add_argument(
        "--select-iteration",
        choices=("median", "min", "max", "first", "last"),
        default="median",
        help="Iteration selection for the composed iteration reducer (trace+replay+iterative only).",
    )
    return p


def _parse_measured_indices(s: Optional[str]) -> Optional[List[int]]:
    if s is None:
        return None
    out: List[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(part))
        except ValueError as e:
            raise MergeError(f"--measured-iteration-indices: cannot parse {part!r} as int") from e
    return out if out else None


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(list(argv) if argv is not None else None)
    input_dir = args.input_dir.resolve()
    out = args.output
    out = input_dir / "merged_ops.csv" if out is None else Path(out).resolve()
    try:
        measured = _parse_measured_indices(args.measured_iteration_indices)
        reducer = partial(
            reduce_trace_replay,
            select_session=args.select_session,
            compose_iteration=not args.no_compose_iteration,
            cli_ops_per_iteration=args.ops_per_iteration,
            cli_measured_indices=measured,
            cli_select_iteration=args.select_iteration,
        )
        run_merge(
            input_dir=input_dir,
            output_path=out,
            dram_peak_bw_gbps=float(args.dram_peak_bw_gbps),
            duration_rel_tol=float(args.duration_rel_tol),
            encoding=str(args.encoding),
            reducer=reducer,
        )
    except MergeError as e:
        print(str(e), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
