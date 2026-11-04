#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Drive TT-Polaris projections + compare_layers XLSX comparisons over a fixed (workload x device) matrix.

v1 hardcodes the combo list.  Future iterations: auto-discover from
``config/all_workloads.yaml`` + ``config/tt_{wh,bh}.yaml``.

HW-reference CSVs are not kept in the tree.  Each combo names its reference by the path it occupies
under ``hlm-refrun/`` on LFC, and ``ttsim.utils.lfc.resolve_lfc_path`` fetches it on first use into
``__ext/hlm-refrun/`` -- the same mechanism the arch configs use for ``operator_lookup_file``.  The
cache is re-checked weekly and falls back to the cached copy when LFC is unreachable, except under
``GITHUB_ACTIONS=true``, where ``ttsim.utils.lfc.download_lfc_file`` skips the stale fallback.

Usage::

    python tools/profiling/run_projections_and_compare.py --run-tag my_run

    # Evaluate the analytical model: skip the operator perf LUT (TT-Polaris --ignore-lut)
    python tools/profiling/run_projections_and_compare.py --run-tag my_run_nolut --ignore-lut

Outputs (under ``__output/<run-tag>/``):
  - ``<combo>/STATS/...-opstats.csv``       (TT-Polaris)
  - ``<combo>/STATS/...-opstats.complete``  (written by this tool once TT-Polaris has finished)
  - ``<combo>/<combo>_comparison.xlsx``     (compare_layers)
  - ``<combo>/polaris.log``                 (TT-Polaris stdout/stderr)
  - ``<combo>/compare.log``                 (compare_layers stdout/stderr)
  - ``summary.md``                          (the aggregated gap / LUT-hit table, also printed)
  - plus whatever else TT-Polaris writes under the study directory -- ``clear_stale_outputs`` lists
    what a run leaves today, and empties the directory rather than tracking that list
"""

from __future__ import annotations

import argparse
import contextlib
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ttsim.utils.lfc import resolve_lfc_path  # noqa: E402

# LFC subtree holding the HW-reference merged-ops CSVs; see its README for the layout rules.
REFRUN_LFC_ROOT = 'hlm-refrun'

# Per-subprocess ceiling. A healthy cell finishes in two to three minutes, so this leaves room
# for a loaded machine or a cold LUT fetch inside TT-Polaris while still ending a wedged process
# in minutes rather than never. The code treats a hung matrix as its worst outcome -- nothing
# fails and nothing finishes -- and a wedged subprocess is the likeliest route to one.
_DEFAULT_TIMEOUT_S = 900


@dataclass(frozen=True)
class ParsedSummary:
    """The five fields ``extract_summary`` reads out of a compare_layers log.

    Declared rather than a bare dict: mypy does not cover ``tools/profiling/``, so a mistyped key
    would surface as a ``KeyError`` inside ``emit_report`` -- which runs after the table has gone
    to stdout, so it would take the report with it.
    """
    polaris_ms: float
    profiler_ms: float
    gap_pct: float | None      # None when compare_layers prints "Gap: N/A" (reference total 0)
    lut_hits: int | None       # both None when neither side has a hit, so the line is absent
    lut_total: int | None


@dataclass(frozen=True)
class SummaryRow:
    """One reported row: a parsed summary plus the combo identity it belongs to."""
    label: str
    bs: int
    parsed: ParsedSummary


@dataclass
class Combo:
    """One (workload x device) verification cell.

    ``refrun_rel`` locates the HW reference CSV as ``<workload>/<arch_family>/<sku>/<file>``
    under ``hlm-refrun/`` on LFC.
    """
    label: str               # short tag for output dir + summary row
    archfile: str            # path to arch YAML (relative to REPO_ROOT)
    filterarch: str          # --filterarch value (e.g. 'n150', 'p100a')
    filterwli: str           # --filterwli value (workload instance name)
    workload_yaml_name: str  # the wlname inside config/all_workloads.yaml's group
    instance_name: str       # the wli inside the group (usually == filterwli)
    bs_runtime: int          # batch size to use (--batchsize override; matches HW refrun)
    refrun_rel: str          # HW refrun CSV path on LFC -- see the class docstring


# The ViT cells project the ``*_device_perf`` workloads (entry point ``run_vit_device_ops``), NOT
# the ``*_perf_report`` ones that config/runcfg_vitoptim_{wh,bh}.yaml select for the older
# tools/profiling/correlate_perf*.sh flow. Both correlate against the same 260415 capture, so the
# distinction is easy to lose: ``*_device_perf`` emits the per-op graph this tool compares op by op
# against a merged-ops CSV, while ``*_perf_report`` produces the report-shaped run. Each row of
# summary.md names the workload and instance it came from, so a report cannot be mistaken for the
# other flow's. Whether the two drivers should be reconciled is tracked separately.
COMBOS: list[Combo] = [
    Combo(
        label='vit_wh_n150',
        archfile='config/tt_wh.yaml',
        filterarch='n150',
        filterwli='vitoptim_wh_b16_device_perf',
        workload_yaml_name='vitoptim_wh_device_perf',
        instance_name='vitoptim_wh_b16_device_perf',
        bs_runtime=8,
        refrun_rel='vit/wh/n150/merged_ops_opt_sharded_260415.csv',
    ),
    Combo(
        label='vit_bh_p100a',
        archfile='config/tt_bh.yaml',
        filterarch='p100a',
        filterwli='vitoptim_bh_b16_device_perf',
        workload_yaml_name='vitoptim_bh_device_perf',
        instance_name='vitoptim_bh_b16_device_perf',
        bs_runtime=10,
        refrun_rel='vit/bh/p100a/merged_ops_opt_sharded_260426.csv',
    ),
    Combo(
        label='vgg_wh_n150',
        archfile='config/tt_wh.yaml',
        filterarch='n150',
        filterwli='vgg_unet_wh_b1_device_perf',
        workload_yaml_name='vgg_unet_wh_device_perf',
        instance_name='vgg_unet_wh_b1_device_perf',
        bs_runtime=1,
        refrun_rel='vgg_unet/wh/n150/merged_ops_dualref_260515.csv',
    ),
    Combo(
        label='vgg_bh_p100a',
        archfile='config/tt_bh.yaml',
        filterarch='p100a',
        filterwli='vgg_unet_bh_b1_device_perf',
        workload_yaml_name='vgg_unet_bh_device_perf',
        instance_name='vgg_unet_bh_b1_device_perf',
        bs_runtime=1,
        refrun_rel='vgg_unet/bh/p100a/merged_ops_dualref_260519.csv',
    ),
    # llama3 dual-mode decode (BH p100a). Its arch config + workload landed on main 2026-08-07, so
    # this combo is live here. bs=1 / seq_len=1024 (set on the all_workloads instance) match the
    # batch-1 reference. Verified 100% (749/749) at +0.53% gap 2026-08-10. Combos whose arch config
    # or refrun CSV is absent on a given branch are auto-skipped by validate_combos().
    Combo(
        label='llama3_decode_bh_p100a',
        archfile='config/tt_bh_llama3.yaml',
        filterarch='p100a',
        filterwli='llama3_8b_dualmode_decode_b1',
        workload_yaml_name='llama3_dualmode_decode',
        instance_name='llama3_8b_dualmode_decode_b1',
        bs_runtime=1,
        refrun_rel='llama3/bh/p100a/merged_ops_decode_260703.csv',
    ),
]


def opstats_path_for(run_tag: str, combo: Combo) -> Path:
    """Where TT-Polaris will leave this combo's opstats CSV.

    This reconstructs a filename TT-Polaris owns: the other half of the contract is
    ``ttsim.stats.hlmstats.HLMStats.dump_stats``, which assembles the same parts and appends
    ``b{batchsize}``. A change to that ordering breaks every combo here at once, so it is named
    rather than left to be rediscovered.

    ``bs_runtime`` must therefore be a batch size TT-Polaris actually simulates. The ViT cells force
    that with a ``--batchsize`` override; the ``bs_runtime == 1`` cells rely on the instance
    carrying ``bs: 1`` in config/all_workloads.yaml, which a test pins.
    """
    return (
        REPO_ROOT / '__output' / run_tag / combo.label / 'STATS' /
        f'{combo.filterarch}-TTNN-{combo.workload_yaml_name}-{combo.instance_name}'
        f'-b{combo.bs_runtime}-opstats.csv'
    )


def xlsx_path_for(run_tag: str, combo: Combo) -> Path:
    return REPO_ROOT / '__output' / run_tag / combo.label / f'{combo.label}_comparison.xlsx'


def log_path_for(run_tag: str, combo: Combo, name: str) -> Path:
    """One construction site for the per-combo log paths, so clearing cannot drift from writing."""
    return REPO_ROOT / '__output' / run_tag / combo.label / name


def summary_path_for(run_tag: str) -> Path:
    """The run tag's top-level report. One construction site, for the same reason as the logs."""
    return REPO_ROOT / '__output' / run_tag / 'summary.md'


def completion_marker_for(run_tag: str, combo: Combo) -> Path:
    """Written only once TT-Polaris has exited 0 *and* left its opstats CSV in place.

    The CSV alone cannot answer "did TT-Polaris finish?". ``ttsim.utils.common.print_csv`` opens the
    file and writes rows as it goes, so an interrupted run leaves a truncated file that still
    parses -- and ``--skip-polaris`` would then reuse it and report a network total computed from
    part of the network. Existence of the CSV proves a run started; this marker proves one
    finished, so it is what ``--skip-polaris`` tests.
    """
    return opstats_path_for(run_tag, combo).with_suffix('.complete')


def opstats_artifacts(run_tag: str, combo: Combo) -> tuple[bool, bool]:
    """``(csv present, marker present)`` for this combo, each as a regular file.

    ``is_file()`` rather than ``exists()``, so a directory carrying either name cannot answer for
    it. Returned as a pair because the caller needs the two answers separately: which of them is
    missing is the difference between "an earlier run did not finish" and "a marker outlived its
    CSV", and both readings are gone once clearing has run.
    """
    return (opstats_path_for(run_tag, combo).is_file(),
            completion_marker_for(run_tag, combo).is_file())


def reusable_opstats(artifacts: tuple[bool, bool]) -> bool:
    """True only when a completed run left a CSV that is actually there to reuse.

    Both artifacts are required. Either one alone is a state no run produces: a CSV without its
    marker is what an interrupted run leaves behind, and a marker without its CSV -- if the CSV
    were removed independently -- would skip simulation and hand compare_layers a path that does
    not exist.
    """
    csv_there, marker_there = artifacts
    return csv_there and marker_there


def run_polaris(combo: Combo, run_tag: str, timeout_s: int = _DEFAULT_TIMEOUT_S,
                *, ignore_lut: bool = False) -> Path:
    """Run polaris.py for this combo. Returns the opstats CSV path.

    ``ignore_lut`` forwards ``--ignore-lut``: the arch config's operator perf LUT is not loaded and
    every op is timed by the analytical model.
    """
    study = f'{run_tag}/{combo.label}'
    cmd = [
        sys.executable, 'polaris.py',
        '-w', 'config/all_workloads.yaml',
        '-a', combo.archfile,
        '-m', 'config/wl2archmapping.yaml',
        '--filterarch', combo.filterarch,
        '--filterwli', combo.filterwli,
        '--study', study,
        '-o', '__output/',
        '--dump_stats_csv',
    ]
    if ignore_lut:
        cmd.append('--ignore-lut')
    # No --datatype on purpose. It sets op_data_type_spec.global_type, which set_precision writes
    # onto every op as op.precision -- but the LUT keys these comparisons depend on are read from
    # tensor state: shape_canonical.tensor_datatype() checks _hw_dtype, then _ttnn_dtype, then
    # t.dtype, and uses op.precision only as a last-resort fallback. Shim tensors carry their own
    # dtype, so passing it would not change a single key. Where op.precision does still reach the
    # numbers is the analytical fallback and memory-traffic accounting (see issue #398), neither of
    # which applies while every op hits the LUT.
    if combo.bs_runtime != 1:
        # Reads like it should be [bs, bs, bs], and cannot be: polaris.RangeArgument asserts
        # start != end, so [8, 8, 8] is rejected outright and [1, 8, 8] yields [1, 8]. The price of
        # the forced leading 1 is that these cells also simulate a bs=1 network nothing reads --
        # roughly double the wall clock for the cells that need an override. A single-value
        # batch-size form in TT-Polaris would remove both the idiom and the waste.
        cmd.extend(['--batchsize', '1', str(combo.bs_runtime), str(combo.bs_runtime)])
    log_path = log_path_for(run_tag, combo, 'polaris.log')
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # The caller has already cleared any CSV left by an earlier run under this tag (see
    # clear_stale_outputs), so the check below tests what *this* run produced -- otherwise a
    # surviving stale file would be compared as though it described the current run.
    # This used to carry a second job: TT-Polaris exited 0 with "completed with 0 experiments" when
    # --filterarch/--filterwli selected nothing, so check=True could not catch a bad filter. It
    # now rejects an unmatched filter entry and a zero-experiment selection itself (issue #517),
    # so check=True is sufficient for that case and only the staleness concern remains here.
    opstats = opstats_path_for(run_tag, combo)
    with log_path.open('w') as f:
        subprocess.run(cmd, cwd=str(REPO_ROOT), stdout=f, stderr=subprocess.STDOUT, check=True,
                       timeout=timeout_s)
    # is_file(), not exists(): a directory wearing this name would pass an existence test and then
    # be refused by compare_layers, which requires a regular file. Same reasoning as
    # opstats_artifacts, so the same test.
    if not opstats.is_file():
        raise FileNotFoundError(
            f'TT-Polaris ran but opstats CSV not found: {opstats}\n'
            f'see log: {log_path}'
        )
    # Last, so the marker can only exist for a CSV that a completed run produced. UTC, because the
    # marker's whole job is to be reconciled later -- possibly on another machine.
    completion_marker_for(run_tag, combo).write_text(
        f'{datetime.now(timezone.utc).isoformat(timespec="seconds")}\n', encoding='utf-8'
    )
    return opstats


def clear_stale_outputs(run_tag: str, combo: Combo, *, keep_opstats: bool) -> None:
    """Delete this combo's outputs from any earlier run under the same tag, before anything runs.

    Called for every *selected* combo before validation, not per producing step and not per runnable
    combo. A combo can drop out at validation, at TT-Polaris, at compare, or at parsing, and in every
    one of those cases it is correctly left out of the summary while a file from a previous run stays
    in the output directory looking current. Clearing next to a producing step only covers failures
    after that step; clearing inside the run loop misses combos that never reach it.

    The logs are cleared here too. ``open('w')`` truncates them, but only for a combo that actually
    reaches a subprocess -- a combo dropped at validation opens neither file, so an earlier run's
    logs would sit beside the new report. Every output named in the module docstring is covered here,
    which is the point: no producing step is trusted to clean up after itself.

    It clears the whole combo directory rather than a list of known outputs, because enumerating
    them kept falling behind what the subprocesses actually write. Beyond the four files named in the
    module docstring, one run also leaves ``CONFIG/<dev>.json``, a ``-opstats.json`` per simulated
    batch size, ``SUMMARY/study-summary.{json,csv}``, and -- for the ViT cells, which sweep
    ``--batchsize 1 <bs> <bs>`` -- a ``-b1-`` CSV that is never consumed. The directory is wholly
    owned by this tool for a given run tag, so emptying it is both simpler and complete, and it stays
    correct if TT-Polaris grows another output.

    ``keep_opstats`` is set when ``--skip-polaris`` is reusing the CSV of an earlier completed run.
    That CSV is this run's input rather than a leftover, so it is exempt -- and so is its
    completion marker, since clearing one without the other either strands a marker over a deleted
    CSV or drops the proof that the kept CSV is complete.

    The run tag's top-level ``summary.md`` is deliberately not this function's business: it is one
    file for the whole matrix, so the caller clears it once before the per-combo loop.
    """
    combo_dir = REPO_ROOT / '__output' / run_tag / combo.label
    # The chain test main() applies to the run directory, for the same reasons, plus one specific
    # to here: this path is rglob's root below, so a link at it puts the whole clearing loop inside
    # the target. The caller records a raise as this combo's problem and carries on.
    if combo_dir.resolve() != REPO_ROOT.resolve() / '__output' / run_tag / combo.label:
        raise OSError(f'{combo_dir} does not resolve to its own place under __output; '
                      f'refusing to clear through it')
    if not combo_dir.is_dir():
        return
    # Both kept together -- see ``keep_opstats`` in the docstring for why.
    kept = ([opstats_path_for(run_tag, combo), completion_marker_for(run_tag, combo)]
            if keep_opstats else [])
    for keep in kept:
        if not keep.resolve().is_relative_to(combo_dir.resolve()):
            # Kept paths and the directories holding them are protected from clearing below, so a
            # link anywhere on that chain would survive and then be read through -- comparing a
            # file from outside this combo's directory as though this run had produced it.
            raise OSError(f'{keep} resolves outside {combo_dir}; refusing to reuse an opstats CSV '
                          f'reached through a link')
        # Containment is not enough on its own: a link that stays *inside* the combo directory
        # passes it, and clearing then removes the link's target as a separate tree entry while
        # preserving the lexical kept path -- leaving --skip-polaris a dangling CSV to reuse. Every
        # component from combo_dir down to the kept artifact therefore has to be a real directory.
        probe = keep
        while probe != combo_dir:
            if probe.is_symlink():
                raise OSError(f'{probe} is a symlink on the path to a reused artifact; refusing '
                              f'to reuse through it')
            probe = probe.parent
    protected = {p for keep in kept for p in (keep, *keep.parents)}
    # Deepest first, so a directory is only removed once its contents are gone.
    for path in sorted(combo_dir.rglob('*'), key=lambda p: len(p.parts), reverse=True):
        if path in protected:
            continue
        # Stated as "a directory, or else unlink it", not as a list of the types worth removing.
        # Enumerating them (link, regular file, ...) silently spares everything not on the list:
        # a FIFO or a socket is none of those, so it survived an "empty the directory" operation,
        # and a stale FIFO at polaris.log or the opstats path then makes the next open('w') block
        # forever waiting for a reader -- a hung matrix rather than a failed combo.
        #
        # is_symlink() comes first because every other test here follows a link: for a link to a
        # directory is_file() is false, is_dir() is true, and rmdir() raises into the suppression,
        # so the link would survive and this run's writes would land in its target. rglob does not
        # descend into it, so the target's own contents are never touched.
        if path.is_symlink() or not path.is_dir():
            path.unlink(missing_ok=True)
        else:
            # No suppression: `protected` skips every directory holding a reused artifact, and
            # deepest-first ordering has already emptied the others, so an OSError here means the
            # combo directory was not cleared. The caller records that as this combo's failure
            # rather than running on with stale outputs in place.
            path.rmdir()


def resolve_refrun(combo: Combo) -> Path:
    """Return the local path to this combo's HW-reference CSV, fetching it from LFC on first use.

    ``resolve_lfc_path`` returns a path relative to the working directory and validates the cache
    location against a relative ``__ext``, so pin the CWD to the repo root -- this script is
    runnable from anywhere.
    """
    with contextlib.chdir(REPO_ROOT):
        return REPO_ROOT / resolve_lfc_path(f'lfc://{REFRUN_LFC_ROOT}/{combo.refrun_rel}')


def compare_env() -> dict[str, str]:
    """Environment for the compare_layers subprocess.

    compare_layers validates every input path against ``POLARIS_BASE_DIR``, falling back to
    ``HOME``, and refuses anything outside it. Both of our inputs live under the repo (``__output/``
    for opstats, ``__ext/`` for the LFC-cached reference), so point the base at the repo: a checkout
    outside the user's home directory would otherwise be rejected for every combo.
    """
    return {**os.environ, 'POLARIS_BASE_DIR': str(REPO_ROOT)}


def run_compare(combo: Combo, run_tag: str, opstats: Path, refrun: Path,
                timeout_s: int = _DEFAULT_TIMEOUT_S) -> tuple[Path, Path]:
    """Run compare_layers.py. Returns (xlsx_path, compare_log_path).

    The caller clears any stale workbook before the combo starts -- see ``clear_stale_outputs``.
    """
    xlsx_path = xlsx_path_for(run_tag, combo)
    log_path = log_path_for(run_tag, combo, 'compare.log')
    cmd = [
        sys.executable, 'tools/profiling/compare_layers.py',
        '--perf', '--by-lut-key',
        '--xlsx', str(xlsx_path),
        str(opstats),
        str(refrun),
    ]
    with log_path.open('w') as f:
        try:
            subprocess.run(cmd, cwd=str(REPO_ROOT), stdout=f, stderr=subprocess.STDOUT, check=True,
                           env=compare_env(), timeout=timeout_s)
        except (subprocess.SubprocessError, OSError):
            # compare_layers writes the workbook last (openpyxl saves at the end), so a failure
            # or a timeout can leave a truncated XLSX. Clearing runs before launch only, so
            # nothing else would remove it, and it would sit beside a report that marks this
            # combo not reported. The original exception carries the reason, so re-raise it.
            with contextlib.suppress(OSError):
                xlsx_path.unlink(missing_ok=True)
            raise
    return xlsx_path, log_path


# The LUT-hits line and a numeric gap are both optional: compare_layers.py only prints
# "<label> LUT hits" when at least one side has a hit (so a 0-hit baseline run omits it), and prints
# "Gap:  N/A" when the reference total is 0. Neither should cost us the row.
_SUMMARY_RE = re.compile(
    r'Network total:\s*\n'
    r'\s*(?:TT-)?Polaris:\s*([\d.]+)\s*ms\s*\n'
    r'\s*Profiler:\s*([\d.]+)\s*ms\s*\n'
    r'\s*Gap:\s*(N/A|[+-]?[\d.]+)%?[^\n]*\n'
    r'(?:\s*(?:TT-)?Polaris LUT hits:\s*(\d+)/(\d+))?'
)


def extract_summary(compare_log: Path) -> ParsedSummary | None:
    """Pull the network totals out of a compare_layers log.

    ``--perf --by-lut-key`` prints *two* ``Network total`` blocks: the LUT-key rollup first, then the
    full performance summary. Take the **last** one. The rollup aggregates only layers that have a
    LUT key -- ``_aggregate_by_lut_key`` skips a layer whose ``lut_key``/``lut_key_resolved`` is
    None -- so on a workload with any keyless op its total and hit denominator are short of the real
    network. The performance summary covers every layer.
    """
    text = compare_log.read_text(encoding='utf-8', errors='replace')
    matches = list(_SUMMARY_RE.finditer(text))
    if not matches:
        return None
    m = matches[-1]
    gap, hits, total = m.group(3), m.group(4), m.group(5)
    return ParsedSummary(
        polaris_ms=float(m.group(1)),
        profiler_ms=float(m.group(2)),
        gap_pct=None if gap == 'N/A' else float(gap),
        lut_hits=None if hits is None else int(hits),
        lut_total=None if total is None else int(total),
    )


def validate_combos(combos: list[Combo]) -> tuple[list[tuple[Combo, Path]], list[tuple[str, str]]]:
    """Pair each combo with its resolved refrun CSV, splitting off the ones that can't run.

    A combo is dropped when its arch config is absent from this branch (letting combos for
    workloads that live on other branches ship here dormant and auto-activate once those land on
    main) or when its refrun CSV is neither cached nor fetchable from LFC. Resolving up front means
    a missing reference is reported before a multi-minute simulation rather than after it. Drops are
    reported and make the run exit nonzero, so a partial matrix is never mistaken for a clean one.
    """
    ok: list[tuple[Combo, Path]] = []
    skipped: list[tuple[str, str]] = []
    for c in combos:
        if not (REPO_ROOT / c.archfile).exists():
            reason = f'arch config not on this branch: {c.archfile}'
        else:
            try:
                refrun = resolve_refrun(c)
                # Resolving it is not the same as it being usable. resolve_lfc_path validates the
                # cache location against a *relative* ``__ext``, so a redirected ``__ext`` passes
                # here while compare_layers -- which resolves its inputs and requires them under
                # POLARIS_BASE_DIR, pinned by compare_env to REPO_ROOT -- refuses the reference
                # with "Access denied". Left to that point, the combo simulates for minutes and
                # then fails at comparison; drops belong here, where they are reported up front.
                # Containment is not the only way a resolved reference can be unusable, and both
                # ways have to be caught here rather than downstream: resolve_lfc_path treats any
                # fresh *existing* path as a cache hit, a directory included, while compare_layers
                # requires a regular file. Either mismatch would pass validation, spend the
                # simulation, and fail at comparison.
                resolved = refrun.resolve()
                if not resolved.is_relative_to(REPO_ROOT.resolve()):
                    reason = (f'refrun {c.refrun_rel} resolves outside the repo ({resolved}); '
                              f'compare_layers would refuse it')
                elif not resolved.is_file():
                    reason = (f'refrun {c.refrun_rel} is not a regular file ({resolved}); '
                              f'compare_layers would refuse it')
                else:
                    ok.append((c, refrun))
                    continue
            except (RuntimeError, ValueError, OSError) as e:
                reason = f'refrun {c.refrun_rel} unavailable (local cache and LFC): {e}'
        print(f'WARN: skipping {c.label!r} - {reason}', file=sys.stderr)
        skipped.append((c.label, reason))
    return ok, skipped


def emit_report(run_tag: str, summaries: list[SummaryRow], problems: list[tuple[str, str]],
                n_requested: int) -> bool:
    """Print the aggregated table and write the same content to ``summary.md``.

    Returns whether the file was written. Writing can fail for reasons the table cannot --
    an unusable run tag, a full or read-only filesystem -- and by then the table and the
    reasons have already gone to stdout, so raising would lose them as well as the file.
    The caller turns a failed write into a nonzero exit rather than a traceback.

    Called on every path, including the ones where nothing ran: when the whole matrix fails, the
    problem list *is* the report, and a caller looking for ``summary.md`` afterwards should find the
    reasons rather than an absent file.
    """
    labels = [f'{s.label} (bs={s.bs})' for s in summaries]
    col_label = max([len('Combo')] + [len(x) for x in labels])
    header = (f'| {"Combo":<{col_label}} | {"TT-Polaris (ms)":>16} | {"Profiler (ms)":>14} '
              f'| {"Gap":>8} | {"LUT hits":>11} |')
    sep = f'|{"-"*(col_label+2)}|{"-"*18}|{"-"*16}|{"-"*10}|{"-"*13}|'
    rows = []
    for s, label in zip(summaries, labels):
        gap = 'N/A' if s.parsed.gap_pct is None else f'{s.parsed.gap_pct:+.2f}%'
        hits = ('n/a' if s.parsed.lut_hits is None
                else f'{s.parsed.lut_hits}/{s.parsed.lut_total}')
        rows.append(f'| {label:<{col_label}} | {s.parsed.polaris_ms:>16.4f} '
                    f'| {s.parsed.profiler_ms:>14.4f} | {gap:>8} | {hits:>11} |')
    problem_lines = [f'- `{label}`: {reason}' for label, reason in problems]
    no_rows_note = 'No combo produced a summary row.'

    print()
    print('=== Summary ===')
    if rows:
        print(header)
        print(sep)
        for r in rows:
            print(r)
    else:
        print(no_rows_note)
    if problem_lines:
        print()
        print(f'=== Not reported ({len(problems)} of {n_requested} requested) ===')
        for line in problem_lines:
            print(line)

    # Name the workload and instance behind every reported row. A label alone ("vit_wh_n150") does
    # not say which of several ViT workloads was projected, and more than one driver in this repo
    # correlates ViT against the same capture through a different instance -- so a report that
    # omits this cannot be told apart from the other one's.
    by_label = {c.label: c for c in COMBOS}
    provenance = [
        f'- `{s.label}`: arch `{by_label[s.label].filterarch}`, '
        f'workload `{by_label[s.label].workload_yaml_name}`, '
        f'instance `{by_label[s.label].instance_name}`, bs {s.bs}, '
        f'reference `{by_label[s.label].refrun_rel}`'
        for s in summaries if s.label in by_label
    ]

    md_lines = [
        f'# Projection + compare summary - run_tag `{run_tag}`',
        '',
        f'Generated: {datetime.now(timezone.utc).isoformat(timespec="seconds")}',
        f'Combos reported: {len(summaries)} of {n_requested} requested ({len(COMBOS)} known)',
        '',
    ]
    md_lines += [header, sep, *rows, ''] if rows else [f'_{no_rows_note}_', '']
    if provenance:
        md_lines += ['## What each row projected', '', *provenance, '']
    if problem_lines:
        md_lines += ['## Not reported', '', *problem_lines, '']
    summary_path = summary_path_for(run_tag)
    # Written to a sibling temporary file and moved into place. Path.write_text truncates first,
    # so a failure part-way through replaced a complete earlier report with a partial one -- and
    # the handler below then said nothing was saved, which was untrue in exactly that case.
    # os.replace is atomic within a filesystem, so the reader sees the old report or the new one.
    tmp_path = summary_path.with_name(summary_path.name + '.partial')
    try:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        # The temp path is a fixed name, so treat whatever is already there the way the rest of
        # this tool treats stale output: remove the entry itself rather than writing through it. A
        # stale symlink would otherwise be followed -- overwriting its target, possibly outside the
        # repository -- and a FIFO would block the report forever. 'x' then refuses to follow
        # anything raced in between the unlink and the open, and raises OSError if it happens.
        tmp_path.unlink(missing_ok=True)
        with open(tmp_path, 'x', encoding='utf-8') as fh:
            fh.write('\n'.join(md_lines))
        os.replace(tmp_path, summary_path)
    except OSError as e:
        with contextlib.suppress(OSError):
            tmp_path.unlink(missing_ok=True)
        print()
        print(f'WARNING: could not write {summary_path}: {e}', file=sys.stderr)
        print('The table above is the whole report; any earlier report is unchanged.',
              file=sys.stderr)
        return False
    print()
    print(f'Summary saved: {summary_path.relative_to(REPO_ROOT)}')
    return True


# fullmatch, not match: with `$` a trailing newline is still a match, so `match()` would accept
# "run1\n" despite it being outside the documented character set.
#
# The first character is restricted twice over. Not `-`: the tag is interpolated into TT-Polaris's
# `--study <value>`, and argparse in the child reads a leading-hyphen value as an option, so every
# subprocess would exit before running. Not `.` either: `--run-tag .hidden` would put a whole run
# somewhere a directory listing does not show, and `...` is worse -- it is legal, distinct from
# `..`, and nearly invisible. Excluding `.` here also makes this regex the single gate, replacing
# a separate {'.', '..'} test that had to be kept in step with it.
_SAFE_RUN_TAG = re.compile(r'[A-Za-z0-9_][A-Za-z0-9._-]*')

# A directory component is capped at 255 bytes on every filesystem this runs on, and the tag is
# only the first of several components in the paths built from it. 128 leaves room for the rest
# while being longer than any tag a person types.
_MAX_RUN_TAG = 128


def timeout_arg(value: str) -> int:
    """Accept only a positive number of seconds.

    ``subprocess.run(timeout=0)`` and ``timeout=-5`` both raise ``TimeoutExpired`` immediately --
    neither means "no timeout" -- so a non-positive ceiling reports every combo as having exceeded
    it, and the report is indistinguishable from a real matrix-wide failure. Refused here, where
    the problem can be named, for the same reason ``run_tag_arg`` refuses an over-long tag.
    """
    try:
        seconds = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f'must be an integer number of seconds, got {value!r}')
    if seconds < 1:
        raise argparse.ArgumentTypeError(f'must be at least 1 second, got {seconds}')
    return seconds


def run_tag_arg(value: str) -> str:
    """Accept only a single relative directory name, of usable length.

    The tag is both a path component under ``__output/`` and TT-Polaris's ``--study`` path, so a value
    containing a separator or ``..`` would write outside the output tree, and an absolute one would
    make ``Path.__truediv__`` discard the ``__output`` prefix entirely and then trip the
    ``relative_to(REPO_ROOT)`` calls used for display.

    Length is checked here because the character set says nothing about it: a regex-valid 300-byte
    tag reaches the filesystem and raises ``ENAMETOOLONG`` out of the first probe that touches it
    (``Path.is_file()`` propagates that rather than answering ``False``). Rejecting it at the CLI
    names the actual problem, where a caught ``OSError`` further in can only say the path is
    unusable.
    """
    if not _SAFE_RUN_TAG.fullmatch(value):
        raise argparse.ArgumentTypeError(
            f'must be a single relative directory name matching [A-Za-z0-9_][A-Za-z0-9._-]*, '
            f'got {value!r}')
    if len(value.encode()) > _MAX_RUN_TAG:
        raise argparse.ArgumentTypeError(
            f'must be at most {_MAX_RUN_TAG} bytes, got {len(value.encode())}')
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run-tag', type=run_tag_arg,
                        help='Output subdirectory tag, e.g. "session_2026-05-25". A single relative '
                             'directory name. Required unless --list is used.')
    parser.add_argument('--skip-polaris', action='store_true',
                        help='Reuse the opstats CSV of an earlier *completed* run instead of '
                             'simulating. A CSV left behind by a run that did not finish is not '
                             'reused, since it parses but describes only part of the network. '
                             'Combos with nothing reusable are still simulated, with a note on '
                             'stderr, so an interrupted matrix can be topped up in one re-run.')
    parser.add_argument('--timeout-per-combo', type=timeout_arg, default=_DEFAULT_TIMEOUT_S,
                        metavar='SECONDS',
                        help=f'Ceiling for each TT-Polaris and compare_layers subprocess '
                             f'(default {_DEFAULT_TIMEOUT_S}). A cell that exceeds it is '
                             f'recorded like any other failed combo and the matrix continues.')
    parser.add_argument('--ignore-lut', action='store_true',
                        help='Pass --ignore-lut to TT-Polaris: do not load the operator perf LUT, so '
                             'every op uses the analytical model. Only affects combos that are '
                             'simulated; an opstats CSV reused via --skip-polaris is left as it was '
                             'produced.')
    parser.add_argument('--only', metavar='COMBO_LABEL', action='append',
                        help='Run only the named combo(s) (default: all of them). Repeatable.')
    parser.add_argument('--list', action='store_true', help='List available combo labels and exit')
    args = parser.parse_args()

    if args.list:
        width = max(len(c.label) for c in COMBOS)
        for c in COMBOS:
            print(f'{c.label:<{width}}  arch={c.filterarch:<5}  bs={c.bs_runtime}  '
                  f'wli={c.filterwli}')
        return 0

    if not args.run_tag:
        parser.error('--run-tag is required (unless using --list)')

    if args.ignore_lut and args.skip_polaris:
        # A reused CSV is not re-simulated, so it keeps whatever LUT setting its own run used.
        print('NOTE: --ignore-lut only applies to combos that are simulated; opstats CSVs reused '
              'via --skip-polaris keep the LUT setting of the run that produced them.',
              file=sys.stderr)

    # Reject unknown labels rather than filtering them away: a silently-dropped typo would not be
    # counted in n_requested, so it could not turn the exit status nonzero and would leave a
    # partial run looking clean.
    known = {c.label for c in COMBOS}
    if args.only:
        unknown = sorted(set(args.only) - known)
        if unknown:
            parser.error(f'unknown --only label(s): {", ".join(unknown)}. '
                         f'Known labels: {", ".join(sorted(known))}')

    selected = COMBOS if not args.only else [c for c in COMBOS if c.label in args.only]
    n_requested = len(selected)

    # Clear stale outputs for every *selected* combo, before validation rather than inside the run
    # loop. A combo dropped by validate_combos (arch config absent, reference unfetchable) never
    # enters that loop, so clearing there left its previous workbook and CSV sitting next to a report
    # saying the combo was not reported. Deciding reuse per combo here keeps --skip-polaris's input.
    # Read both artifacts once, here, and keep the answer. Reuse needs the CSV *and* its marker
    # (see reusable_opstats), and the run loop needs to know *which* was missing to explain itself
    # -- but by the time it runs, clearing has deleted both for every combo not being reused, so
    # asking the filesystem again there can only ever answer "neither".
    # This tool deletes inside the run tag's directory, so nothing on the path to it may redirect
    # where those deletions land. Test the resolved chain rather than one component: a link at any
    # level redirects both the deletions below and every output written afterwards, and testing
    # the run tag alone would miss one planted on __output. Only REPO_ROOT is resolved on the
    # right-hand side, so a link at __output is rejected too.
    #
    # Redirecting __output to another volume was considered as a disk-quota measure and
    # deliberately rejected: compare_layers' sanitize_file_path() resolves its inputs and requires
    # them to stay under POLARIS_BASE_DIR, which compare_env() pins to REPO_ROOT, so an opstats CSV
    # on another volume is refused with "Access denied" and every combo fails. Accepting the link
    # here would start a clean-looking multi-minute run that then loses every comparison. Serving
    # that case needs compare_layers to allow a second base directory, which is its own change.
    # Both properties are checked at both levels, because each test says nothing about the other
    # and a failure at either level ends the same way -- a traceback out of the first write instead
    # of a controlled refusal. Resolving to the expected place does not make a path a directory: a
    # regular file at __output leaves run_dir.resolve() lexically correct and run_dir.exists()
    # false, so the type test has to be applied to the root as well, not only to the run tag.
    output_root = REPO_ROOT / '__output'
    run_dir = output_root / args.run_tag
    # The whole pre-flight sits inside one OSError boundary. The checks below answer questions
    # about the path; they do not stop the filesystem from refusing to answer. A symlink *loop*
    # is the case that proves it: non-strict resolve() returns the lexical path for a loop and
    # exists() is false, so both checks pass, and the unlink then fails with ELOOP -- which, left
    # uncaught, is a traceback where every other malformed output path gets a refusal.
    try:
        # Inside the boundary, not above it: these are filesystem probes like the ones below, and
        # a path can be regex-valid yet unusable -- a 300-character run tag makes is_file() raise
        # ENAMETOOLONG, which pathlib propagates rather than answering False.
        artifacts = {c.label: opstats_artifacts(args.run_tag, c) for c in selected}
        reuse = {c.label: args.skip_polaris and reusable_opstats(artifacts[c.label])
                 for c in selected}
        for path in (output_root, run_dir):
            if path.resolve() != REPO_ROOT.resolve() / path.relative_to(REPO_ROOT):
                print(f'refusing to run: {path.relative_to(REPO_ROOT)} does not resolve to its '
                      f'own place in the repo, so this tool would delete in a directory it does '
                      f'not own', file=sys.stderr)
                return 1
            if path.exists() and not path.is_dir():
                print(f'refusing to run: {path.relative_to(REPO_ROOT)} exists and is not a '
                      f'directory', file=sys.stderr)
                return 1
        # The top-level report is cleared here too, not only the per-combo directories. It is
        # written last, so an interrupted or aborted run would otherwise leave the previous run's
        # summary.md at the top of this run tag's output, read as the current result.
        summary_path_for(args.run_tag).unlink(missing_ok=True)
    except OSError as e:
        print(f'refusing to run: cannot use {run_dir.relative_to(REPO_ROOT)}: {e}',
              file=sys.stderr)
        return 1
    problems: list[tuple[str, str]] = []
    cleared: list[Combo] = []
    for combo in selected:
        # A combo whose directory cannot be cleared is this combo's failure, not the matrix's: an
        # unwritable directory used to abort every remaining combo before emit_report() ran, which
        # both lost the aggregate and (with the unlink above) left no report at all.
        try:
            clear_stale_outputs(args.run_tag, combo, keep_opstats=reuse[combo.label])
        except OSError as e:
            reason = f'could not clear stale outputs: {e}'
            print(f'  FAILED: {combo.label}: {reason}', file=sys.stderr)
            problems.append((combo.label, reason))
            continue
        cleared.append(combo)

    runnable, validation_problems = validate_combos(cleared)
    problems += validation_problems
    if not runnable:
        # Not only "arch config or reference missing" any more: a combo can also drop out here
        # because its output directory could not be cleared, so point at the reasons themselves.
        print('No runnable combos - see "Not reported" below for the reason on each.',
              file=sys.stderr)
        emit_report(args.run_tag, [], problems, n_requested)
        return 1

    summaries: list[SummaryRow] = []
    for combo, refrun in runnable:
        print(f'=== {combo.label} ===', flush=True)
        # A failure here must not abort the rest of the matrix: this tool exists to aggregate all
        # combos, so a broken one is recorded and the remaining combos still run. The refrun line
        # is inside the boundary for that reason: relative_to() raises ValueError for a path
        # outside REPO_ROOT, and validate_combos checks the *resolved* path, not the lexical one.
        try:
            print(f'  refrun:  {refrun.relative_to(REPO_ROOT)}')
            opstats = opstats_path_for(args.run_tag, combo)
            reusing = reuse[combo.label]
            if reusing:
                print(f'  reusing opstats: {opstats.relative_to(REPO_ROOT)}')
            else:
                if args.skip_polaris:
                    # Falling through deliberately, so an interrupted matrix can be topped up with
                    # one --skip-polaris re-run. Say so: the flag implies no simulation, and a
                    # silent multi-minute TT-Polaris run here would be a surprise. Name which case
                    # it is -- "no CSV" and "a CSV an interrupted run left behind" are
                    # indistinguishable in the output directory, and only one is safe to reuse.
                    csv_there, marker_there = artifacts[combo.label]
                    if csv_there and not marker_there:
                        reason = 'an earlier run left one but did not finish'
                    elif marker_there and not csv_there:
                        reason = 'a completion marker outlived its CSV'
                    else:
                        reason = 'none present'
                    print(f'  NOTE: --skip-polaris given but no completed opstats to reuse '
                          f'({reason}); simulating anyway', file=sys.stderr)
                print(f'  running TT-Polaris (study={args.run_tag}/{combo.label})...', flush=True)
                opstats = run_polaris(combo, args.run_tag, args.timeout_per_combo,
                                      ignore_lut=args.ignore_lut)
                print(f'  opstats: {opstats.relative_to(REPO_ROOT)}')
            print('  running compare_layers...', flush=True)
            xlsx, compare_log = run_compare(combo, args.run_tag, opstats, refrun,
                                            args.timeout_per_combo)
            print(f'  xlsx:    {xlsx.relative_to(REPO_ROOT)}')
            # Reading the summary belongs inside this boundary: it opens, decodes and converts, so
            # a bad log has to cost this combo its row and nothing more.
            s = extract_summary(compare_log)
        except subprocess.TimeoutExpired as e:
            reason = (f'{Path(e.cmd[1]).name} exceeded {args.timeout_per_combo}s - see '
                      f'__output/{args.run_tag}/{combo.label}/')
            print(f'  FAILED: {reason}', file=sys.stderr)
            problems.append((combo.label, reason))
            continue
        except subprocess.CalledProcessError as e:
            reason = f'{Path(e.cmd[1]).name} exited {e.returncode} - see __output/{args.run_tag}/{combo.label}/'
            print(f'  FAILED: {reason}', file=sys.stderr)
            problems.append((combo.label, reason))
            continue
        except OSError as e:
            print(f'  FAILED: {e}', file=sys.stderr)
            problems.append((combo.label, str(e)))
            continue
        except ValueError as e:
            reason = f'unreadable numbers in the compare summary: {e}'
            print(f'  FAILED: {reason}', file=sys.stderr)
            problems.append((combo.label, reason))
            continue
        if s is None:
            reason = f'could not parse summary from {compare_log.relative_to(REPO_ROOT)}'
            print(f'  WARN: {reason}', file=sys.stderr)
            problems.append((combo.label, reason))
            continue
        summaries.append(SummaryRow(label=combo.label, bs=combo.bs_runtime, parsed=s))

    written = emit_report(args.run_tag, summaries, problems, n_requested)
    if not written:
        return 1
    if not summaries:
        print('No combo produced a parseable summary.', file=sys.stderr)
        return 1
    # Exit nonzero if any requested combo is missing from the table, so a partial matrix can't be
    # mistaken for a clean one by a caller that only checks the status.
    if problems:
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
