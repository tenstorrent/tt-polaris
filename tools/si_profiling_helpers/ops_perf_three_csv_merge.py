#!/usr/bin/env python3
# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Merge exactly three TT-Metal-style ops perf CSVs (noctrace, fpuutil, vanilla) into one CSV.

Specification: doc/SPEC_ops_perf_three_csv_merge.md
Uses stdlib **csv** for I/O (no pandas); **loguru** for warnings. Exit 0 on success, 1 on error.
Warnings (e.g. fpuutil vs vanilla duration beyond tolerance) do not stop the merge.
With ``--allow-missing-noctrace``, a vanilla CSV and an fpu CSV merge without the noctrace
CSV; the noctrace-only output columns are then omitted.

Two derived overhead-ratio columns are appended last (blank when raw duration is 0):
``FPUUtil/Raw`` = ``[ms]_fpuutil / [ms]`` and ``MemUtil/Raw`` = ``[ms]_noctrace / [ms]``.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from functools import partial
from pathlib import Path
from typing import Callable, Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple

from loguru import logger

# --- Column names (exact TT-Metal ops perf strings) ---
COL_GLOBAL_CALL_COUNT = "GLOBAL CALL COUNT"
COL_OP_CODE = "OP CODE"
COL_OP_TYPE = "OP TYPE"
COL_DEVICE_KERNEL_NS = "DEVICE KERNEL DURATION [ns]"
COL_DEVICE_KERNEL_MS = "DEVICE KERNEL DURATION [ms]"
COL_DRAM_BW_UTIL = "DRAM BW UTIL (%)"
COL_PM_FPU_UTIL = "PM FPU UTIL (%)"
COL_FPU_UTIL_MED = "FPU Util Median (%)"
COL_SFPU_UTIL_MED = "SFPU Util Median (%)"
COL_NOC_UTIL = "NOC UTIL (%)"
COL_MC_NOC_UTIL = "MULTICAST NOC UTIL (%)"
COL_ETH_BW_UTIL = "ETH BW UTIL (%)"
COL_NPE_CONG = "NPE CONG IMPACT (%)"

# Tracy signpost markers (tracy.signpost("...")) are emitted as CSV rows by TT-Metal's
# ops-log post-processor with OP TYPE == "signpost" and no GLOBAL CALL COUNT. It is the
# only non-op OP TYPE the processor ever writes (the sole literal OP TYPE assignment in
# tt-metal tools/tracy/process_ops_logs.py). These rows are host-side navigation markers,
# not device ops, so they are dropped before validation / iteration detection / join-key
# parsing. Matched case-insensitively for robustness.
OP_TYPE_SIGNPOST = "signpost"

JOIN_KEYS: Tuple[str, ...] = (COL_GLOBAL_CALL_COUNT, COL_OP_CODE, COL_OP_TYPE)

UTILIZATION_COLUMNS: Tuple[str, ...] = (
    COL_DRAM_BW_UTIL,
    COL_FPU_UTIL_MED,
    COL_SFPU_UTIL_MED,
    COL_NOC_UTIL,
    COL_MC_NOC_UTIL,
    COL_ETH_BW_UTIL,
    COL_NPE_CONG,
)

# Vanilla may omit these; noctrace and fpu must still include them.
VANILLA_OMITTABLE_UTIL_COLUMNS: FrozenSet[str] = frozenset({COL_FPU_UTIL_MED, COL_SFPU_UTIL_MED})

# Required in every input before classification (FPU/SFPU optional for vanilla only).
BASE_INPUT_COLUMNS: FrozenSet[str] = frozenset(
    set(JOIN_KEYS)
    | {COL_DEVICE_KERNEL_NS, COL_DRAM_BW_UTIL}
    | {COL_NOC_UTIL, COL_MC_NOC_UTIL, COL_ETH_BW_UTIL, COL_NPE_CONG}
)

# Steps 3–4 source names, excluding MemTraffic (derived only).
OVERLAY_SOURCE_NAMES: FrozenSet[str] = frozenset(
    {
        COL_DEVICE_KERNEL_NS,
        COL_DEVICE_KERNEL_MS,
        COL_NOC_UTIL,
        COL_MC_NOC_UTIL,
        COL_ETH_BW_UTIL,
        COL_NPE_CONG,
        COL_DRAM_BW_UTIL,
        COL_FPU_UTIL_MED,
        COL_SFPU_UTIL_MED,
    }
)

NOCTRACE_OUTPUT_SUFFIX = "_noctrace"
FPU_OUTPUT_SUFFIX = "_fpuutil"

# Derived overhead-ratio columns (appended last): how much the fpu-util and
# noc-trace profiling passes inflate the raw (vanilla) device-kernel duration.
# Blank when the raw duration is 0 (no division).
COL_FPU_UTIL_RAW = "FPUUtil/Raw"  # = [ms]_fpuutil / [ms]
COL_MEM_UTIL_RAW = "MemUtil/Raw"  # = [ms]_noctrace / [ms]

# Order of suffixed noctrace output columns (logical names before suffix).
NOCTRACE_OUTPUT_BODY: Tuple[str, ...] = (
    "MemTraffic",  # derived; output MemTraffic_noctrace
    COL_NOC_UTIL,
    COL_MC_NOC_UTIL,
    COL_ETH_BW_UTIL,
    COL_NPE_CONG,
    COL_DRAM_BW_UTIL,
)

# Output columns only the noctrace CSV supplies. With --allow-missing-noctrace and no
# noctrace CSV they are omitted, never written blank: consumers read a blank cell as 0.
NOCTRACE_ONLY_OUTPUT_COLUMNS: Tuple[str, ...] = (
    f"{COL_DEVICE_KERNEL_NS}{NOCTRACE_OUTPUT_SUFFIX}",
    f"{COL_DEVICE_KERNEL_MS}{NOCTRACE_OUTPUT_SUFFIX}",
    *(f"{c}{NOCTRACE_OUTPUT_SUFFIX}" for c in NOCTRACE_OUTPUT_BODY),
    COL_MEM_UTIL_RAW,
)


class MergeError(Exception):
    """User-facing merge validation error."""


def _strip_cell(val: Optional[str]) -> str:
    if val is None:
        return ""
    return val.strip()


def parse_finite_float(cell: Optional[str]) -> Optional[float]:
    """Return finite float if cell parses; empty/whitespace -> None; '-' or bad -> None."""
    s = _strip_cell(cell)
    if not s or s == "-":
        return None
    try:
        x = float(s)
    except ValueError:
        return None
    if not math.isfinite(x):
        return None
    return x


def dram_value_for_classification(cell: Optional[str]) -> float:
    """Missing/empty DRAM BW UTIL -> 0 for noctrace/vanilla classification."""
    v = parse_finite_float(cell)
    return 0.0 if v is None else v


def is_blank_or_zero_dram(cell: Optional[str]) -> bool:
    """DRAM blank/whitespace or parses to 0.0 (fpu post-check)."""
    s = _strip_cell(cell)
    if not s:
        return True
    v = parse_finite_float(cell)
    return v is None or v == 0.0


def is_blank_or_zero_fpu_sfpu(cell: Optional[str]) -> bool:
    """Vanilla: FPU/SFPU blank or parses to 0.0."""
    s = _strip_cell(cell)
    if not s:
        return True
    v = parse_finite_float(cell)
    return v is None or v == 0.0


def row_has_parseable_fpu_or_sfpu(row: Dict[str, str]) -> bool:
    fpu = parse_finite_float(row.get(COL_FPU_UTIL_MED))
    sfpu = parse_finite_float(row.get(COL_SFPU_UTIL_MED))
    return fpu is not None or sfpu is not None


def classify_file(rows: Sequence[Dict[str, str]]) -> str:
    """Return 'noctrace', 'fpu', 'vanilla', or 'unassigned' (assign-once order)."""
    if not rows:
        return "unassigned"

    # (1) noctrace
    for r in rows:
        if dram_value_for_classification(r.get(COL_DRAM_BW_UTIL)) != 0.0:
            return "noctrace"

    # (2) fpu
    any_fpu_sfpu = any(row_has_parseable_fpu_or_sfpu(r) for r in rows)
    if any_fpu_sfpu:
        for r in rows:
            if not is_blank_or_zero_dram(r.get(COL_DRAM_BW_UTIL)):
                return "unassigned"
        return "fpu"

    # (3) vanilla
    for r in rows:
        if not is_blank_or_zero_dram(r.get(COL_DRAM_BW_UTIL)):
            return "unassigned"
        if not is_blank_or_zero_fpu_sfpu(r.get(COL_FPU_UTIL_MED)):
            return "unassigned"
        if not is_blank_or_zero_fpu_sfpu(r.get(COL_SFPU_UTIL_MED)):
            return "unassigned"
    return "vanilla"


def read_csv(path: Path, encoding: str) -> Tuple[Tuple[str, ...], List[Dict[str, str]]]:
    with path.open("r", newline="", encoding=encoding) as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise MergeError(f"CSV has no header row: {path}")
        header = tuple(reader.fieldnames)
        rows: List[Dict[str, str]] = []
        for row in reader:
            if None in row:
                raise MergeError(f"CSV row has more fields than header: {path}")
            rows.append(row)
        return header, rows


def drop_signpost_rows(rows: List[Dict[str, str]], path: Path) -> List[Dict[str, str]]:
    """Drop Tracy signpost marker rows (OP TYPE == "signpost").

    Signposts are host-side timeline markers (e.g. the ``start``/``stop`` pair a perf test
    wraps around the profiled region); TT-Metal writes them as rows with an empty
    GLOBAL CALL COUNT, which would otherwise break join-key parsing. They carry no device-op
    data, so removing them here keeps every downstream step (validation, iteration detection,
    join) operating on real ops only. See ``OP_TYPE_SIGNPOST``.
    """
    kept = [r for r in rows if _strip_cell(r.get(COL_OP_TYPE)).lower() != OP_TYPE_SIGNPOST]
    dropped = len(rows) - len(kept)
    if dropped:
        logger.info("Dropped {} signpost marker row(s) from {}", dropped, path)
    return kept


def require_columns(header: Sequence[str], required: Iterable[str], path: Path) -> None:
    s = set(header)
    missing = [c for c in required if c not in s]
    if missing:
        raise MergeError(f"Missing columns in {path}: {', '.join(missing)}")


def require_fpu_signal_columns(header: Sequence[str], path: Path) -> None:
    """Fpu file must contain at least one of the two FPU/SFPU median columns.

    PM FPU UTIL (%) is intentionally excluded: it is present in all three CSV types so
    cannot serve as a classification signal, and any file reaching this check will already
    have FPU Util Median or SFPU Util Median (that is what classified it as fpu).
    """
    if COL_FPU_UTIL_MED not in header and COL_SFPU_UTIL_MED not in header:
        raise MergeError(
            f"Fpu CSV must include {COL_FPU_UTIL_MED!r} or {COL_SFPU_UTIL_MED!r}: {path}"
        )


def validate_utilization_cells(
    rows: Sequence[Dict[str, str]], path: Path, header: Sequence[str]
) -> None:
    """Check that utilization columns parse to finite floats; warn on out-of-[0,100].

    TT-Metal sometimes reports values >100 for NoC/DRAM util (e.g. multicast
    fan-out double-counts) — the spec used to error on this but real data
    routinely violates the bound, so we now warn and continue.  Negative values
    still warn (likely measurement error or sign mishap).

    TODO(util-over-100): the merge operation should not produce util >100; the
    merge script propagates raw HW values verbatim and the >100 cases originate
    in tt-metal's NoC counters (multicast overcount).  Resolve by either
    (a) clipping at 100 in the merge output with a one-time loguru info, or
    (b) coordinating with tt-metal to fix the source counters.  Until then the
    warning-only behavior is a placeholder so the pipeline can run on real data.
    """
    present = set(header)
    n_over = 0
    n_under = 0
    for ri, r in enumerate(rows):
        for col in UTILIZATION_COLUMNS:
            if col not in present:
                continue
            v = parse_finite_float(r.get(col))
            if v is None:
                continue
            if v > 100:
                n_over += 1
                if n_over <= 3:
                    logger.warning(
                        "Utilization >100 in {} row {} column {!r}: {} (TT-Metal "
                        "multicast/overcount; warning only)",
                        path, ri + 1, col, v,
                    )
            elif v < 0:
                n_under += 1
                if n_under <= 3:
                    logger.warning(
                        "Utilization <0 in {} row {} column {!r}: {}",
                        path, ri + 1, col, v,
                    )
    if n_over > 3:
        logger.warning("...and {} more rows with util>100 in {}", n_over - 3, path)
    if n_under > 3:
        logger.warning("...and {} more rows with util<0 in {}", n_under - 3, path)


def assert_vanilla_header_matches_noctrace(
    h_noctrace: Sequence[str], h_vanilla: Sequence[str], path: Path
) -> None:
    """Vanilla header = noctrace order, optionally dropping FPU/SFPU util columns only."""
    canon_set = set(h_noctrace)
    for c in h_vanilla:
        if c not in canon_set:
            raise MergeError(
                f"Vanilla CSV {path} has column {c!r} not present in the noctrace CSV header."
            )
    subseq = tuple(c for c in h_noctrace if c in set(h_vanilla))
    if subseq != tuple(h_vanilla):
        raise MergeError(
            f"Vanilla CSV {path} header must list the same columns as the noctrace CSV in the same "
            f"order, optionally omitting {COL_FPU_UTIL_MED!r} and/or {COL_SFPU_UTIL_MED!r}."
        )
    dropped = set(h_noctrace) - set(h_vanilla)
    if not dropped.issubset(VANILLA_OMITTABLE_UTIL_COLUMNS):
        raise MergeError(
            f"Vanilla CSV {path} omits columns that are not allowed to be omitted: "
            f"{', '.join(sorted(dropped - VANILLA_OMITTABLE_UTIL_COLUMNS))}"
        )


def assert_fpu_header_extends_noctrace(
    h_noctrace: Sequence[str], h_fpu: Sequence[str], path: Path
) -> None:
    """Every noctrace column must appear in the fpu header in the same order.

    The fpu file carries extra columns (the perf-counter analysis block), which
    tt-metal interleaves *before* trailing op-type columns (e.g.
    ``TT_DNN_DEVICE_OP_TT_HOST_FUNC [ns]``) rather than strictly appending. So the
    relationship is an ordered **subsequence**, not a prefix: noctrace's columns
    occur in fpu in order, with the fpu-only columns spliced in anywhere.
    """
    it = iter(h_fpu)
    missing = [c for c in h_noctrace if c not in it]  # consumes it in order
    if missing:
        raise MergeError(
            f"Fpu CSV {path} header must contain every noctrace column in the same order "
            f"(noctrace must be an ordered subsequence of the fpu header); "
            f"missing or out-of-order: {missing}"
        )


def discover_csv_paths(root: Path) -> List[Path]:
    """Find ops_perf_results_*.csv files under root (recursive).

    Filters to the TT-Metal ops_perf CSV naming convention so that ancillary
    files (e.g. profile_log_device.csv from the same report directory) are
    excluded automatically.  Falls back to *.csv only if the filtered set is
    empty, preserving backward compatibility for callers that stage 3 CSVs in
    a directory with arbitrary names.
    """
    if not root.is_dir():
        raise MergeError(f"Input directory not found or not a directory: {root}")
    paths = sorted({p.resolve() for p in root.rglob("ops_perf_results_*.csv") if p.is_file()})
    if not paths:
        paths = sorted({p.resolve() for p in root.rglob("*.csv") if p.is_file()})
    return paths


def parse_join_key(row: Dict[str, str], path: Path, row_index: int) -> Tuple[int, str, str]:
    gcc_s = _strip_cell(row.get(COL_GLOBAL_CALL_COUNT))
    if not gcc_s:
        raise MergeError(f"Empty {COL_GLOBAL_CALL_COUNT!r} in {path} row {row_index + 1}")
    try:
        gcc = int(gcc_s, 10)
    except ValueError as e:
        raise MergeError(
            f"{COL_GLOBAL_CALL_COUNT!r} not an integer in {path} row {row_index + 1}: {gcc_s!r}"
        ) from e
    op_c = _strip_cell(row.get(COL_OP_CODE))
    op_t = _strip_cell(row.get(COL_OP_TYPE))
    if not op_c or not op_t:
        raise MergeError(f"Empty OP CODE or OP TYPE in {path} row {row_index + 1}")
    return (gcc, op_c, op_t)


def sort_rows_by_key(
    rows: List[Dict[str, str]], path: Path
) -> Tuple[List[Dict[str, str]], List[Tuple[int, str, str]]]:
    keyed: List[Tuple[Tuple[int, str, str], Dict[str, str]]] = []
    for i, r in enumerate(rows):
        k = parse_join_key(r, path, i)
        keyed.append((k, r))
    keyed.sort(key=lambda x: (x[0][0], x[0][1], x[0][2]))
    seen: set = set()
    out_rows: List[Dict[str, str]] = []
    out_keys: List[Tuple[int, str, str]] = []
    for k, r in keyed:
        if k in seen:
            raise MergeError(f"Duplicate join key {k!r} in {path}")
        seen.add(k)
        out_rows.append(r)
        out_keys.append(k)
    return out_rows, out_keys


def keys_equal(a: Tuple[int, str, str], b: Tuple[int, str, str]) -> bool:
    return a == b


# ---------------------------------------------------------------------------
# Iteration detection and median-iteration selection
# ---------------------------------------------------------------------------
#
# TT-Metal profiler harnesses (e.g. run_device_perf with num_iterations=N) often
# produce CSVs containing multiple iterations of the same model concatenated as
# successive blocks of rows.  This script reduces those to a single iteration
# (used downstream by tt_perf_mapper) by detecting iteration boundaries from
# the row sequence and selecting one representative iteration by total kernel
# duration.
#
# Detection (in priority order):
#   0. --ops-per-iteration: authoritative fixed-size chunking when supplied.
#   1. OP CODE period (primary): a trace-replay capture is N copies of one
#      inference's op sequence, so the OP CODE column is periodic. The iteration
#      size is the minimum duplicate join-key stride, corroborated by OP CODE
#      periodicity at that stride and unique join keys within one period. This is
#      robust when the first op recurs *within* an iteration (e.g. BH VGG emits
#      ReshardDeviceOperation 3x per pass), which defeats the marker heuristic.
#   2. First-op marker (fallback): iteration starts are positions where row 0's
#      OP CODE recurs. Equal-sized chunks => a genuine multi-iteration run (used
#      for non-trace captures with monotonic GCCs that have no duplicate keys).
#      Unequal sizes (marker recurs intra-pass) => one iteration (logged);
#      pass --ops-per-iteration to override.
#
# Completeness: an iteration is "complete" if at least 95% of its rows have a
# non-empty DEVICE KERNEL DURATION [ns].  TT-Metal sometimes emits sparse
# sync/teardown iterations between launches; those are excluded.
#
# Selection: among complete iterations, pick the one with median total kernel
# duration.  Returns its (start_idx, end_idx) row range.
#
ITER_COMPLETENESS_THRESHOLD = 0.95


def _min_duplicate_key_stride(rows: List[Dict[str, str]]) -> Optional[int]:
    """Smallest row-distance between two rows sharing a (gcc, op_code, op_type) key.

    In a trace-replay capture the same join key recurs once per iteration (keys are
    unique *within* an iteration but repeat *across* iterations), so the minimum gap
    between two occurrences of any key equals the ops-per-iteration. Returns None if
    no key repeats (nothing to reduce) or any key field is blank (cannot trust it).
    """
    last_pos: Dict[Tuple[str, str, str], int] = {}
    best: Optional[int] = None
    for i, r in enumerate(rows):
        gcc = _strip_cell(r.get(COL_GLOBAL_CALL_COUNT))
        op_c = _strip_cell(r.get(COL_OP_CODE))
        op_t = _strip_cell(r.get(COL_OP_TYPE))
        if not gcc or not op_c or not op_t:
            return None
        k = (gcc, op_c, op_t)
        if k in last_pos:
            d = i - last_pos[k]
            if best is None or d < best:
                best = d
        last_pos[k] = i
    return best


def _op_code_periodic(rows: List[Dict[str, str]], period: int) -> bool:
    """True if the OP CODE column repeats with the given period across all rows."""
    ops = [_strip_cell(r.get(COL_OP_CODE)) for r in rows]
    if any(not o for o in ops):
        return False
    return all(ops[i] == ops[i - period] for i in range(period, len(ops)))


def _period_keys_unique(rows: List[Dict[str, str]], start: int, end: int) -> bool:
    """True if the (gcc, op_code, op_type) join keys in rows[start:end] are all distinct."""
    seen: set = set()
    for i in range(start, end):
        r = rows[i]
        k = (
            _strip_cell(r.get(COL_GLOBAL_CALL_COUNT)),
            _strip_cell(r.get(COL_OP_CODE)),
            _strip_cell(r.get(COL_OP_TYPE)),
        )
        if "" in k or k in seen:
            return False
        seen.add(k)
    return True


def detect_by_op_code_period(rows: List[Dict[str, str]]) -> Optional[List[Tuple[int, int]]]:
    """Detect iteration boundaries from the OP CODE sequence period.

    A trace-replay capture is N copies of one inference's op sequence concatenated,
    so the OP CODE column is periodic with period = ops-per-iteration. Unlike the
    first-op marker heuristic, this does not care which op is first or how many times
    it recurs within an iteration -- e.g. BH VGG emits ReshardDeviceOperation 3x per
    pass, giving unequal marker chunks that defeat marker-based detection while the op
    sequence stays perfectly periodic.

    The iteration size is taken from the minimum duplicate-key stride (the granularity
    at which join keys actually repeat) and corroborated two ways before use:
      1. the OP CODE sequence is periodic at that stride, and
      2. one period's join keys are unique (so the downstream join is collision-free).
    Using the duplicate-key stride rather than the smallest tiling OP CODE period
    avoids over-splitting a model whose op sequence has an internal sub-period (the
    stride reflects where keys genuinely repeat, i.e. true iteration boundaries).
    Returns one (start, end) range per iteration, or None when no clean period is
    found -- the caller then falls back to the first-op marker heuristic.
    """
    n = len(rows)
    if n < 2:
        return None
    stride = _min_duplicate_key_stride(rows)
    # No repeating join key => nothing to reduce here; let the marker heuristic decide
    # (it handles single-iteration and equal-chunk non-trace multi-iteration captures).
    if stride is None or stride <= 0 or stride >= n or n % stride != 0:
        return None
    if not _op_code_periodic(rows, stride):
        return None
    if not _period_keys_unique(rows, 0, stride):
        return None
    return [(s, s + stride) for s in range(0, n, stride)]


def detect_iteration_boundaries(
    rows: List[Dict[str, str]],
    ops_per_iteration: Optional[int] = None,
) -> List[Tuple[int, int]]:
    """Return list of (start_idx, end_idx_exclusive) tuples — one per iteration.

    When ``ops_per_iteration`` is given, rows are sliced into fixed-size chunks of
    that length (authoritative override — useful when row 0's OP CODE recurs
    intra-iteration, which defeats the default marker-based detection).
    Otherwise iteration boundaries are inferred from positions where the first
    row's OP CODE recurs. Equal-sized chunks are treated as multiple iterations;
    unequal sizes (marker recurs intra-pass, e.g. a single-pass device-perf
    capture) fall back to a single iteration spanning all rows.
    """
    if not rows:
        return []
    if ops_per_iteration is not None:
        if ops_per_iteration <= 0:
            raise MergeError(f"--ops-per-iteration must be positive, got {ops_per_iteration}")
        if len(rows) % ops_per_iteration != 0:
            raise MergeError(
                f"--ops-per-iteration={ops_per_iteration} does not evenly divide row count "
                f"{len(rows)}; expected an integer multiple"
            )
        return [
            (s, s + ops_per_iteration) for s in range(0, len(rows), ops_per_iteration)
        ]
    # Primary: OP CODE sequence period (robust when the first op recurs intra-iteration,
    # which defeats the marker heuristic below -- e.g. BH VGG's ReshardDeviceOperation).
    period_iters = detect_by_op_code_period(rows)
    if period_iters is not None:
        logger.info(
            "Iteration auto-detect via OP CODE period: {} iteration(s) of {} ops "
            "(duplicate-key stride corroborated).",
            len(period_iters),
            period_iters[0][1] - period_iters[0][0],
        )
        return period_iters
    # Fallback: first-op marker heuristic.
    first_op = _strip_cell(rows[0].get(COL_OP_CODE))
    if not first_op:
        raise MergeError("First row has empty OP CODE; cannot detect iteration boundaries")
    starts = [i for i, r in enumerate(rows) if _strip_cell(r.get(COL_OP_CODE)) == first_op]
    if not starts or starts[0] != 0:
        raise MergeError(
            f"First-op marker {first_op!r} does not occur at row 0; cannot detect iterations"
        )
    ends = starts[1:] + [len(rows)]
    iters = list(zip(starts, ends))
    sizes = [e - s for s, e in iters]
    if len(set(sizes)) > 1:
        # Unequal marker-based chunks mean row 0's OP CODE recurs *within* a single
        # pass rather than at true iteration boundaries — typical of single-pass
        # device-perf captures (is_device_perf_test drops warmup -> one clean pass).
        # A genuine multi-iteration averaged run yields equal chunks, so here we
        # fall back to treating the whole input as ONE iteration rather than failing.
        # Pass --ops-per-iteration to force fixed-size splitting if this really is a
        # multi-iteration file whose marker happens to recur intra-iteration.
        logger.warning(
            "Iteration auto-detect: marker {!r} gives unequal chunk sizes {}; "
            "treating input as a single iteration ({} ops). "
            "Pass --ops-per-iteration if this is a multi-iteration run.",
            first_op, sizes, len(rows),
        )
        return [(0, len(rows))]
    return iters


def iteration_summary(
    rows: List[Dict[str, str]],
    iters: List[Tuple[int, int]],
) -> List[Tuple[int, int, bool, float]]:
    """Return (start, end, is_complete, total_kdur_ns) for each iteration."""
    out: List[Tuple[int, int, bool, float]] = []
    for s, e in iters:
        chunk = rows[s:e]
        nonempty = 0
        total = 0.0
        for r in chunk:
            v = parse_finite_float(r.get(COL_DEVICE_KERNEL_NS))
            if v is not None:
                total += v
                nonempty += 1
        is_complete = (nonempty / len(chunk)) >= ITER_COMPLETENESS_THRESHOLD
        out.append((s, e, is_complete, total))
    return out


def pick_median_iteration(
    summary: List[Tuple[int, int, bool, float]],
    measured_indices: Optional[List[int]] = None,
    select: str = "median",
) -> Tuple[int, int, int]:
    """Return (start, end, iter_idx) of the chosen iteration.

    Candidate set:
      - If ``measured_indices`` is given: those iteration indices (0-based).
      - Else: all complete iterations.

    Selection by ``select`` ∈ {median, min, max, first, last}.  Median uses the
    lower middle when the candidate count is even (stable choice).
    """
    if measured_indices is not None:
        cand = [(i, *summary[i]) for i in measured_indices if 0 <= i < len(summary)]
        # All candidates kept regardless of completeness; user knows what they're picking.
    else:
        cand = [(i, *s) for i, s in enumerate(summary) if s[2]]  # is_complete

    if not cand:
        raise MergeError(
            "No iterations available for selection "
            "(no complete iterations found and no --measured-iteration-indices)"
        )

    if select == "first":
        chosen = cand[0]
    elif select == "last":
        chosen = cand[-1]
    elif select == "min":
        chosen = min(cand, key=lambda x: x[4])
    elif select == "max":
        chosen = max(cand, key=lambda x: x[4])
    elif select == "median":
        cand_sorted = sorted(cand, key=lambda x: x[4])
        # Lower-middle for even counts (stable, deterministic).
        chosen = cand_sorted[(len(cand_sorted) - 1) // 2]
    else:
        raise MergeError(f"Unknown --select-iteration value: {select!r}")

    idx, start, end, _is_complete, _total = chosen
    return start, end, idx


def filter_to_iteration(
    rows: List[Dict[str, str]],
    start: int,
    end: int,
) -> List[Dict[str, str]]:
    """Return rows[start:end] (no copying of inner dicts; existing rows reused)."""
    return rows[start:end]


def validate_iteration_args(
    summary: List[Tuple[int, int, bool, float]],
    *,
    cli_num_iterations: Optional[int],
    cli_measured_indices: Optional[List[int]],
) -> None:
    """Validate CLI iteration args against auto-detected summary; mismatches warn (loguru)."""
    n_complete = sum(1 for s in summary if s[2])
    n_total = len(summary)

    if cli_num_iterations is not None and cli_num_iterations != n_complete:
        logger.warning(
            "--num-iterations={} but detected {} complete iterations (of {} total); "
            "completeness threshold = {:.0%}.",
            cli_num_iterations,
            n_complete,
            n_total,
            ITER_COMPLETENESS_THRESHOLD,
        )

    if cli_measured_indices is not None:
        for i in cli_measured_indices:
            if not (0 <= i < n_total):
                logger.warning(
                    "--measured-iteration-indices contains out-of-range index {} (have {} iterations)",
                    i,
                    n_total,
                )
            elif not summary[i][2]:
                logger.warning(
                    "--measured-iteration-indices includes iteration {} which is incomplete "
                    "(<{:.0%} of rows have kernel duration); using anyway.",
                    i,
                    ITER_COMPLETENESS_THRESHOLD,
                )


# A row reducer takes the three variant row lists (noctrace, fpu, vanilla) and
# returns reduced lists of equal length — one row per op for a single
# representative pass.  It is injected into run_merge so alternate reduction
# strategies (e.g. the trace+replay replay-session reducer in
# ops_perf_trace_replay_merge.py) can replace the default iteration reducer
# without duplicating the shared classify / header-check / join / output
# machinery.  When run_merge is called without a reducer it builds the default
# iteration reducer below from its CLI iteration args.  The noctrace rows are None
# when the merge runs without a noctrace CSV (--allow-missing-noctrace).
Reducer = Callable[
    [Optional[List[Dict[str, str]]], List[Dict[str, str]], List[Dict[str, str]]],
    Tuple[Optional[List[Dict[str, str]]], List[Dict[str, str]], List[Dict[str, str]]],
]


def reduce_iterative(
    rows_noc: Optional[List[Dict[str, str]]],
    rows_fpu: List[Dict[str, str]],
    rows_van: List[Dict[str, str]],
    *,
    cli_num_iterations: Optional[int] = None,
    cli_ops_per_iteration: Optional[int] = None,
    cli_measured_indices: Optional[List[int]] = None,
    cli_select_iteration: str = "median",
) -> Tuple[Optional[List[Dict[str, str]]], List[Dict[str, str]], List[Dict[str, str]]]:
    """Reduce a multi-iteration capture to a single representative iteration.

    Detect iterations on the vanilla CSV (canonical timing), pick one iteration
    by ``cli_select_iteration`` (default: median total kernel duration), then
    apply the same row range to all three CSVs (the three variants share
    iteration layout — same model invocation pattern, same GCC numbering).
    A single detected iteration passes all rows through unchanged.
    """
    iters = detect_iteration_boundaries(rows_van, ops_per_iteration=cli_ops_per_iteration)
    summary = iteration_summary(rows_van, iters)
    iter_size = iters[0][1] - iters[0][0] if iters else 0
    validate_iteration_args(
        summary,
        cli_num_iterations=cli_num_iterations,
        cli_measured_indices=cli_measured_indices,
    )
    if len(iters) > 1:
        start, end, chosen_idx = pick_median_iteration(
            summary,
            measured_indices=cli_measured_indices,
            select=cli_select_iteration,
        )
        # Sanity-check that all 3 CSVs have the same row count (synchronized iteration layout).
        for label, r in (("noctrace", rows_noc), ("fpu", rows_fpu), ("vanilla", rows_van)):
            if r is not None and len(r) != len(rows_van):
                raise MergeError(
                    f"Pre-filter row count differs ({label}={len(r)} vs vanilla={len(rows_van)}); "
                    "iteration layouts must be synchronized across the three variants."
                )
        if rows_noc is not None:
            rows_noc = filter_to_iteration(rows_noc, start, end)
        rows_fpu = filter_to_iteration(rows_fpu, start, end)
        rows_van = filter_to_iteration(rows_van, start, end)
        logger.info(
            "Iteration filter: detected {} iteration(s) of {} ops each ({} complete); "
            "selected iter {} (rows {}..{}, total kdur={} ns, select={!r}).",
            len(iters),
            iter_size,
            sum(1 for s in summary if s[2]),
            chosen_idx,
            start,
            end - 1,
            summary[chosen_idx][3],
            cli_select_iteration,
        )
    return rows_noc, rows_fpu, rows_van


def run_merge(
    input_dir: Path,
    output_path: Path,
    dram_peak_bw_gbps: float,
    duration_rel_tol: float,
    encoding: str,
    cli_num_iterations: Optional[int] = None,
    cli_ops_per_iteration: Optional[int] = None,
    cli_measured_indices: Optional[List[int]] = None,
    cli_select_iteration: str = "median",
    reducer: Optional[Reducer] = None,
    allow_missing_noctrace: bool = False,
) -> None:
    paths = discover_csv_paths(input_dir)
    if len(paths) != 3 and not (allow_missing_noctrace and len(paths) == 2):
        expected = "2 or 3" if allow_missing_noctrace else "exactly 3"
        hint = (
            " Pass --allow-missing-noctrace to merge a vanilla CSV and an fpu CSV without the noctrace CSV."
            if len(paths) == 2 and not allow_missing_noctrace
            else ""
        )
        raise MergeError(
            f"Expected {expected} CSV files under {input_dir}, found {len(paths)}: "
            f"{', '.join(str(p) for p in paths)}.{hint}"
        )
    missing_noctrace = len(paths) == 2

    loaded: List[Tuple[Path, Tuple[str, ...], List[Dict[str, str]]]] = []
    path_kind: Dict[Path, str] = {}
    for p in paths:
        header, rows = read_csv(p, encoding)
        rows = drop_signpost_rows(rows, p)
        loaded.append((p, header, rows))

    # Classify
    by_kind: Dict[str, List[Path]] = {"noctrace": [], "fpu": [], "vanilla": [], "unassigned": []}
    for p, header, rows in loaded:
        # Inputs need not include DEVICE KERNEL DURATION [ms] (derived from ns when absent).
        require_columns(header, BASE_INPUT_COLUMNS, p)
        validate_utilization_cells(rows, p, header)
        kind = classify_file(rows)
        path_kind[p] = kind
        by_kind[kind].append(p)

    for p, header, rows in loaded:
        if path_kind[p] == "fpu":
            require_fpu_signal_columns(header, p)

    for k in ("unassigned",):
        if by_kind[k]:
            raise MergeError(
                f"Unclassified CSV(s): {', '.join(str(x) for x in by_kind[k])}"
            )

    if (
        len(by_kind["vanilla"]) == 3
        and not by_kind["noctrace"]
        and not by_kind["fpu"]
    ):
        ps = sorted(by_kind["vanilla"], key=lambda x: str(x))
        by_kind["vanilla"] = []
        by_kind["noctrace"] = [ps[0]]
        by_kind["fpu"] = [ps[1]]
        by_kind["vanilla"] = [ps[2]]
        path_kind[ps[0]] = "noctrace"
        path_kind[ps[1]] = "fpu"
        path_kind[ps[2]] = "vanilla"
        logger.warning(
            "All three CSVs classify as vanilla (no non-zero DRAM BW UTIL in any row and no "
            "FPU/SFPU Util Median columns). Assigning roles by sorted file path — noctrace={!s}, "
            "fpu={!s}, vanilla={!s}.",
            ps[0],
            ps[1],
            ps[2],
        )

    # With two files, exactly one fpu and one vanilla CSV leaves no noctrace CSV.
    for kind in ("fpu", "vanilla") if missing_noctrace else ("noctrace", "fpu", "vanilla"):
        ps = by_kind[kind]
        if len(ps) != 1:
            raise MergeError(
                f"Expected exactly one {kind} CSV, found {len(ps)}: {', '.join(str(x) for x in ps)}"
            )

    path_noc = None if missing_noctrace else by_kind["noctrace"][0]
    path_fpu = by_kind["fpu"][0]
    path_van = by_kind["vanilla"][0]

    h_fpu = next(h for p, h, _ in loaded if p == path_fpu)
    h_van = next(h for p, h, _ in loaded if p == path_van)

    rows_noc: Optional[List[Dict[str, str]]] = None
    if path_noc is not None:
        h_noc = next(h for p, h, _ in loaded if p == path_noc)
        assert_fpu_header_extends_noctrace(h_noc, h_fpu, path_fpu)
        assert_vanilla_header_matches_noctrace(h_noc, h_van, path_van)
        header = h_noc
        known_cols = set(h_noc)
        rows_noc = next(r for p, _, r in loaded if p == path_noc)
    else:
        # The vanilla header stands in for the noctrace header. In tt-metal captures the two
        # are identical (neither carries the FPU/SFPU util columns), so the base and
        # fpu-only output columns match a three-CSV merge.
        assert_fpu_header_extends_noctrace(h_van, h_fpu, path_fpu)
        header = h_van
        known_cols = set(h_van)
        logger.warning(
            "No noctrace CSV under {!s} (--allow-missing-noctrace): omitting {}. The output has no "
            "NOC-trace data and cannot build a LUT.",
            input_dir,
            ", ".join(NOCTRACE_ONLY_OUTPUT_COLUMNS),
        )
    # Fpu-only columns identified by name (not position): tt-metal may splice the
    # perf-counter analysis block before trailing op-type columns that noctrace also
    # carries, so the extras are not a positional suffix. Preserve fpu column order.
    fpu_trailing_cols: Tuple[str, ...] = tuple(c for c in h_fpu if c not in known_cols)
    rows_fpu = next(r for p, _, r in loaded if p == path_fpu)
    rows_van = next(r for p, _, r in loaded if p == path_van)

    # Row reduction: collapse the capture to a single representative pass (one
    # row per op) before the join.  The default reducer detects and selects one
    # iteration on the vanilla CSV, then applies the same range to all three
    # (the three variants share iteration layout).  A caller-supplied reducer
    # (e.g. the trace+replay replay-session reducer) replaces this strategy.
    if reducer is None:
        reducer = partial(
            reduce_iterative,
            cli_num_iterations=cli_num_iterations,
            cli_ops_per_iteration=cli_ops_per_iteration,
            cli_measured_indices=cli_measured_indices,
            cli_select_iteration=cli_select_iteration,
        )
    rows_noc, rows_fpu, rows_van = reducer(rows_noc, rows_fpu, rows_van)

    if (rows_noc is not None and len(rows_noc) != len(rows_van)) or len(rows_fpu) != len(rows_van):
        n_noc = "-" if rows_noc is None else str(len(rows_noc))
        raise MergeError(
            f"Row count mismatch: noctrace={n_noc} fpu={len(rows_fpu)} vanilla={len(rows_van)}"
        )

    rows_fpu_s, keys_fpu = sort_rows_by_key(list(rows_fpu), path_fpu)
    rows_van_s, keys_van = sort_rows_by_key(list(rows_van), path_van)
    # Without a noctrace CSV the vanilla keys stand in, so the join checks reduce to fpu vs vanilla.
    rows_noc_s: Optional[List[Dict[str, str]]] = None
    keys_noc = keys_van
    if rows_noc is not None and path_noc is not None:
        rows_noc_s, keys_noc = sort_rows_by_key(list(rows_noc), path_noc)

    set_noc = set(keys_noc)
    set_fpu = set(keys_fpu)
    set_van = set(keys_van)
    if set_noc != set_fpu or set_noc != set_van:
        raise MergeError("Join key sets differ across the three files.")

    dram_bps = dram_peak_bw_gbps * 1e9

    out_rows: List[Dict[str, str]] = []
    total_fpu_kernel_ns = 0.0
    total_vanilla_kernel_ns = 0.0
    warned_missing_noctrace_dram = False

    for i in range(len(rows_van_s)):
        kn, kf, kv = keys_noc[i], keys_fpu[i], keys_van[i]
        if not (keys_equal(kn, kf) and keys_equal(kn, kv)):
            raise MergeError(
                f"Join key mismatch at sorted row index {i}: noctrace={kn!r} fpu={kf!r} vanilla={kv!r}"
            )

        rn = None if rows_noc_s is None else rows_noc_s[i]
        rf = rows_fpu_s[i]
        rv = rows_van_s[i]

        fpu_ns_v = parse_finite_float(rf.get(COL_DEVICE_KERNEL_NS))
        van_ns_v = parse_finite_float(rv.get(COL_DEVICE_KERNEL_NS))
        if fpu_ns_v is None or van_ns_v is None:
            raise MergeError(
                f"Non-finite or missing {COL_DEVICE_KERNEL_NS!r} at row index {i} "
                f"(fpu={path_fpu}, vanilla={path_van})"
            )
        fpu_ns = float(fpu_ns_v)
        van_ns = float(van_ns_v)
        total_fpu_kernel_ns += fpu_ns
        total_vanilla_kernel_ns += van_ns
        if fpu_ns == 0.0 and van_ns == 0.0:
            pass
        else:
            denom = max(abs(fpu_ns), abs(van_ns), 1e-12)
            rel_diff = abs(fpu_ns - van_ns) / denom
            if rel_diff > duration_rel_tol:
                logger.warning(
                    "Duration mismatch fpuutil vs vanilla at row index {} (keys gcc={!r} op_code={!r} "
                    "op_type={!r}): fpu_ns={} vanilla_ns={} rel_diff={:.6g} (rel_tol={}); "
                    "continuing merge (output uses vanilla ns/ms as canonical; fpu block from fpu file).",
                    i,
                    kn[0],
                    kn[1],
                    kn[2],
                    fpu_ns,
                    van_ns,
                    rel_diff,
                    duration_rel_tol,
                )

        if rn is not None:
            u_dram = parse_finite_float(rn.get(COL_DRAM_BW_UTIL))
            if u_dram is None:
                if not warned_missing_noctrace_dram:
                    logger.warning(
                        "Noctrace CSV {!s} has blank or unparsable {!r} on at least one row; using 0 "
                        "for MemTraffic for those rows.",
                        path_noc,
                        COL_DRAM_BW_UTIL,
                    )
                    warned_missing_noctrace_dram = True
                u_dram_f = 0.0
            else:
                u_dram_f = float(u_dram)
            ns_noc = parse_finite_float(rn.get(COL_DEVICE_KERNEL_NS))
            if ns_noc is None:
                raise MergeError(
                    f"{COL_DEVICE_KERNEL_NS!r} must be finite on noctrace row index {i}: {path_noc}"
                )
            mem_traffic = round((u_dram_f / 100.0) * dram_bps * (float(ns_noc) / 1e9))
            ms_noc = float(ns_noc) / 1e6

        ms_fpu = float(parse_finite_float(rf.get(COL_DEVICE_KERNEL_NS)) or 0.0) / 1e6
        # vanilla canonical ns/ms from vanilla row
        van_ns_num = float(van_ns_v)
        ms_van = van_ns_num / 1e6

        out: Dict[str, str] = {}

        for k in JOIN_KEYS:
            out[k] = _strip_cell(rv.get(k))

        # Base columns: vanilla header order, exclude join keys and overlay source names
        for col in header:
            if col in JOIN_KEYS:
                continue
            if col in OVERLAY_SOURCE_NAMES:
                continue
            out[col] = rv.get(col, "")

        out[COL_DEVICE_KERNEL_NS] = _strip_cell(rv.get(COL_DEVICE_KERNEL_NS, ""))
        out[COL_DEVICE_KERNEL_MS] = str(ms_van)

        if rn is not None:
            out[f"{COL_DEVICE_KERNEL_NS}{NOCTRACE_OUTPUT_SUFFIX}"] = _strip_cell(
                rn.get(COL_DEVICE_KERNEL_NS, "")
            )
            out[f"{COL_DEVICE_KERNEL_MS}{NOCTRACE_OUTPUT_SUFFIX}"] = str(ms_noc)
            out[f"MemTraffic{NOCTRACE_OUTPUT_SUFFIX}"] = str(mem_traffic)
            for c in NOCTRACE_OUTPUT_BODY[1:]:
                out[f"{c}{NOCTRACE_OUTPUT_SUFFIX}"] = _strip_cell(rn.get(c, ""))

        out[f"{COL_DEVICE_KERNEL_NS}{FPU_OUTPUT_SUFFIX}"] = _strip_cell(
            rf.get(COL_DEVICE_KERNEL_NS, "")
        )
        out[f"{COL_DEVICE_KERNEL_MS}{FPU_OUTPUT_SUFFIX}"] = str(ms_fpu)
        fpu_med_cell = _strip_cell(rf.get(COL_FPU_UTIL_MED, "")) or _strip_cell(
            rf.get(COL_PM_FPU_UTIL, "")
        )
        out[f"{COL_FPU_UTIL_MED}{FPU_OUTPUT_SUFFIX}"] = fpu_med_cell
        out[f"{COL_SFPU_UTIL_MED}{FPU_OUTPUT_SUFFIX}"] = _strip_cell(rf.get(COL_SFPU_UTIL_MED, ""))

        for col in fpu_trailing_cols:
            out[col] = _strip_cell(rf.get(col))

        # Derived overhead ratios (blank when raw duration is 0 -> no division).
        out[COL_FPU_UTIL_RAW] = "" if ms_van == 0.0 else str(ms_fpu / ms_van)
        if rn is not None:
            out[COL_MEM_UTIL_RAW] = "" if ms_van == 0.0 else str(ms_noc / ms_van)

        out_rows.append(out)

    # Build output fieldnames in spec order
    base_cols = [c for c in header if c not in JOIN_KEYS and c not in OVERLAY_SOURCE_NAMES]
    fieldnames: List[str] = list(JOIN_KEYS)
    fieldnames.extend(base_cols)
    fieldnames.extend([COL_DEVICE_KERNEL_NS, COL_DEVICE_KERNEL_MS])
    # noctrace block
    fieldnames.append(f"{COL_DEVICE_KERNEL_NS}{NOCTRACE_OUTPUT_SUFFIX}")
    fieldnames.append(f"{COL_DEVICE_KERNEL_MS}{NOCTRACE_OUTPUT_SUFFIX}")
    fieldnames.append(f"MemTraffic{NOCTRACE_OUTPUT_SUFFIX}")
    for c in NOCTRACE_OUTPUT_BODY[1:]:
        fieldnames.append(f"{c}{NOCTRACE_OUTPUT_SUFFIX}")
    # fpu block
    fieldnames.append(f"{COL_DEVICE_KERNEL_NS}{FPU_OUTPUT_SUFFIX}")
    fieldnames.append(f"{COL_DEVICE_KERNEL_MS}{FPU_OUTPUT_SUFFIX}")
    fieldnames.append(f"{COL_FPU_UTIL_MED}{FPU_OUTPUT_SUFFIX}")
    fieldnames.append(f"{COL_SFPU_UTIL_MED}{FPU_OUTPUT_SUFFIX}")
    fieldnames.extend(fpu_trailing_cols)
    # Derived overhead ratios, appended last.
    fieldnames.append(COL_FPU_UTIL_RAW)
    fieldnames.append(COL_MEM_UTIL_RAW)
    if rows_noc_s is None:
        fieldnames = [c for c in fieldnames if c not in NOCTRACE_ONLY_OUTPUT_COLUMNS]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding=encoding) as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="raise")
        w.writeheader()
        for out_row in out_rows:
            w.writerow(out_row)

    n = len(out_rows)
    if total_fpu_kernel_ns == 0.0 and total_vanilla_kernel_ns == 0.0:
        rel_sum = 0.0
    else:
        denom_sum = max(abs(total_fpu_kernel_ns), abs(total_vanilla_kernel_ns), 1e-12)
        rel_sum = abs(total_fpu_kernel_ns - total_vanilla_kernel_ns) / denom_sum
    logger.info(
        "Sum of {} over {} merged rows — fpuutil total={} ns, vanilla total={} ns, "
        "relative difference={:.6g} (|fpu_total - vanilla_total| / max(|fpu_total|,|vanilla_total|,1e-12)).",
        COL_DEVICE_KERNEL_NS,
        n,
        total_fpu_kernel_ns,
        total_vanilla_kernel_ns,
        rel_sum,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Merge exactly three TT-Metal-style ops perf CSVs under --input-dir "
            "(noctrace, fpuutil, vanilla) into one CSV."
        ),
        epilog=(
            "Default --output is <input-dir>/merged_ops.csv (same directory as --input-dir). "
            "See doc/SPEC_ops_perf_three_csv_merge.md for validation rules and column contracts."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help=(
            "Root directory; recursively discovers exactly three ops_perf_results_*.csv files "
            "(two with --allow-missing-noctrace) "
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
        "--num-iterations",
        type=int,
        default=None,
        help=(
            "Expected number of complete iterations in the input CSVs (e.g. "
            "run_device_perf's num_iterations).  Auto-detected; if specified, "
            "a mismatch is reported as a warning but does not stop the merge."
        ),
    )
    p.add_argument(
        "--ops-per-iteration",
        type=int,
        default=None,
        help=(
            "Number of ops per iteration (model size).  When specified, rows are "
            "sliced into fixed-size chunks of this length, bypassing marker-based "
            "auto-detection.  Use this when row 0's OP CODE recurs intra-iteration "
            "(e.g. BH VGG UNet has ReshardDeviceOperation at multiple rows per iter). "
            "When omitted, iteration size is auto-detected from row 0 OP CODE recurrence."
        ),
    )
    p.add_argument(
        "--measured-iteration-indices",
        type=str,
        default=None,
        help=(
            "Comma-separated 0-based iteration indices to consider as 'measured' "
            "(e.g. '3,4,5' to exclude warmup/sync iterations).  If omitted, all "
            "complete iterations are candidates."
        ),
    )
    p.add_argument(
        "--select-iteration",
        choices=("median", "min", "max", "first", "last"),
        default="median",
        help=(
            "How to choose among the candidate iterations.  'median' uses the "
            "iteration with median total DEVICE KERNEL DURATION [ns]."
        ),
    )
    p.add_argument(
        "--allow-missing-noctrace",
        action="store_true",
        help=(
            "Accept a vanilla CSV and an fpu CSV without the noctrace CSV. The output omits the "
            "noctrace-only columns (the _noctrace block and MemUtil/Raw) and cannot build a LUT."
        ),
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
            raise MergeError(
                f"--measured-iteration-indices: cannot parse {part!r} as int"
            ) from e
    return out if out else None


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(list(argv) if argv is not None else None)
    input_dir = args.input_dir.resolve()
    out = args.output
    if out is None:
        out = input_dir / "merged_ops.csv"
    else:
        out = Path(out).resolve()
    try:
        measured = _parse_measured_indices(args.measured_iteration_indices)
        run_merge(
            input_dir=input_dir,
            output_path=out,
            dram_peak_bw_gbps=float(args.dram_peak_bw_gbps),
            duration_rel_tol=float(args.duration_rel_tol),
            encoding=str(args.encoding),
            cli_num_iterations=args.num_iterations,
            cli_ops_per_iteration=args.ops_per_iteration,
            cli_measured_indices=measured,
            cli_select_iteration=args.select_iteration,
            allow_missing_noctrace=args.allow_missing_noctrace,
        )
    except MergeError as e:
        print(str(e), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
