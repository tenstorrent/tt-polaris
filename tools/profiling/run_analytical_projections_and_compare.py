#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Run TT-Polaris in analytical mode, compare against the local profiler/reference CSVs,
and emit both terminal + file summaries.

This is a focused variant of run_projections_and_compare.py for the cached local tree:

- LUT files are expected under __ext/hlm-lut/
- merged-ops CSV reference files are expected under __ext/hlm-refrun/
- TT-Polaris runs use the analytical path by default via --ignore-lut
- results are summarized to stdout and __output/<run-tag>/summary.md

The combo matrix intentionally matches the workloads already defined in config/all_workloads.yaml
and the known ResNet50 entries used by this repo.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import re
import subprocess
import sys
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

LUT_CACHE_ROOT = REPO_ROOT / '__ext' / 'hlm-lut'
REFRUN_CACHE_ROOT = REPO_ROOT / '__ext' / 'hlm-refrun'

_DEFAULT_TIMEOUT_S = 900


@dataclass(frozen=True)
class ParsedSummary:
    polaris_ms: float
    profiler_ms: float
    gap_pct: float | None
    lut_hits: int | None
    lut_total: int | None


@dataclass(frozen=True)
class SummaryRow:
    label: str
    bs: int
    parsed: ParsedSummary


@dataclass(frozen=True)
class Combo:
    label: str
    archfile: str
    filterarch: str
    filterwli: str
    workload_yaml_name: str
    instance_name: str
    bs_runtime: int
    refrun_rel: str


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
    Combo(
        label='resnet50_wh_n150',
        archfile='config/tt_wh.yaml',
        filterarch='n150',
        filterwli='resnet50_wh_b16_device_perf',
        workload_yaml_name='resnet50_wh_device_perf',
        instance_name='resnet50_wh_b16_device_perf',
        bs_runtime=16,
        refrun_rel='resnet50/wh/n150/merged_ops_resnet50-02b76a9-wormhole_b0-n150_L-260916.csv',
    ),
    Combo(
        label='resnet50_bh_p100a',
        archfile='config/tt_bh.yaml',
        filterarch='p100a',
        filterwli='resnet50_bh_b16_device_perf',
        workload_yaml_name='resnet50_bh_device_perf',
        instance_name='resnet50_bh_b16_device_perf',
        bs_runtime=16,
        refrun_rel='resnet50/bh/p100a/merged_ops_resnet50-02b76a9-blackhole-p100a-260816.csv',
    ),
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
    # Prefill: one instance per silicon capture (tt-metal f8c634a, bs=1, seq_len=128, 1 layer);
    # see the comments above llama3_dualmode_prefill in config/all_workloads.yaml.
    Combo(
        label='llama3_prefill_bh_p100a',
        archfile='config/tt_bh_llama3.yaml',
        filterarch='p100a',
        filterwli='llama3_8b_dualmode_prefill_bh',
        workload_yaml_name='llama3_dualmode_prefill',
        instance_name='llama3_8b_dualmode_prefill_bh',
        bs_runtime=1,
        refrun_rel='llama3_prefill/bh/p100a/merged_ops_llama3-prefill-b1-l1-t1-f8c634a-blackhole-p100a-261018_new.csv',
    ),
    Combo(
        label='llama3_prefill_wh_n150',
        archfile='config/tt_wh_llama3.yaml',
        filterarch='n150',
        filterwli='llama3_8b_dualmode_prefill_wh',
        workload_yaml_name='llama3_dualmode_prefill',
        instance_name='llama3_8b_dualmode_prefill_wh',
        bs_runtime=1,
        refrun_rel='llama3_prefill/wh/n150/merged_ops_llama3-prefill-b1-f8c634a-wormhole_b0-n150_L-261019.csv',
    ),
]


def opstats_path_for(run_tag: str, combo: Combo) -> Path:
    return (
        REPO_ROOT / '__output' / run_tag / combo.label / 'STATS' /
        f'{combo.filterarch}-TTNN-{combo.workload_yaml_name}-{combo.instance_name}'
        f'-b{combo.bs_runtime}-opstats.csv'
    )


def xlsx_path_for(run_tag: str, combo: Combo) -> Path:
    return REPO_ROOT / '__output' / run_tag / combo.label / f'{combo.label}_comparison.xlsx'


def log_path_for(run_tag: str, combo: Combo, name: str) -> Path:
    return REPO_ROOT / '__output' / run_tag / combo.label / name


def summary_path_for(run_tag: str) -> Path:
    return REPO_ROOT / '__output' / run_tag / 'summary.md'


def completion_marker_for(run_tag: str, combo: Combo) -> Path:
    return opstats_path_for(run_tag, combo).with_suffix('.complete')


def opstats_artifacts(run_tag: str, combo: Combo) -> tuple[bool, bool]:
    return (opstats_path_for(run_tag, combo).is_file(), completion_marker_for(run_tag, combo).is_file())


def reusable_opstats(artifacts: tuple[bool, bool]) -> bool:
    csv_there, marker_there = artifacts
    return csv_there and marker_there


def local_lut_for_combo(combo: Combo) -> Path:
    if combo.label.startswith('llama3'):
        # llama3 (decode and prefill) has its own LUTs: bh_p100a_llama3_lut_v9-1.yaml (preferred),
        # then any other bh_p100a_llama3_lut*.yaml; whb0_n150_llama3_*.yaml on Wormhole.
        llama3 = {
            'p100a': [LUT_CACHE_ROOT / 'bh_p100a_llama3_lut_v9-1.yaml',
                      *sorted(LUT_CACHE_ROOT.glob('bh_p100a_llama3_lut*.yaml'), reverse=True)],
            'n150': [*sorted(LUT_CACHE_ROOT.glob('whb0_n150_llama3_*.yaml'), reverse=True)],
        }
        for candidate in llama3.get(combo.filterarch, []):
            if candidate.is_file():
                return candidate
    candidates = {
        'n150': [
            LUT_CACHE_ROOT / 'whb0_n150_lut_v6.yaml',
            LUT_CACHE_ROOT / 'resnet50' / 'whb0_n150_lut_v6.yaml',
            *sorted(LUT_CACHE_ROOT.glob('whb0_n150_lut*.yaml')),
            *sorted((LUT_CACHE_ROOT / 'resnet50').glob('whb0_n150_lut*.yaml')),
        ],
        'p100a': [
            LUT_CACHE_ROOT / 'bh_p100a_lut_v6.yaml',
            LUT_CACHE_ROOT / 'resnet50' / 'bh_p100a_lut_v6.yaml',
            *sorted(LUT_CACHE_ROOT.glob('bh_p100a_lut*.yaml')),
            *sorted((LUT_CACHE_ROOT / 'resnet50').glob('bh_p100a_lut*.yaml')),
        ],
    }
    for candidate in candidates.get(combo.filterarch, []):
        if candidate.is_file():
            return candidate
    if combo.filterarch in candidates:
        return candidates[combo.filterarch][0]
    raise ValueError(f'No local LUT mapping for archfile={combo.archfile!r}')


def local_refrun_for_combo(combo: Combo) -> Path:
    candidate = REFRUN_CACHE_ROOT / combo.refrun_rel
    if candidate.is_file():
        return candidate
    if 'resnet50/' in combo.refrun_rel:
        alt = REFRUN_CACHE_ROOT / 'resnet50' / combo.refrun_rel.split('resnet50/', 1)[1]
        if alt.is_file():
            return alt
    return candidate


def run_polaris(combo: Combo, run_tag: str, timeout_s: int = _DEFAULT_TIMEOUT_S,
                *, ignore_lut: bool = True) -> Path:
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
    # ResNet50's 224x224 device-perf runner rejects batch sizes 1 and 2 with sharded data, and
    # the instance config already carries the correct bs=16. Forcing a [1, bs, bs] sweep therefore
    # creates an invalid run for this workload even though the combo is otherwise valid.
    # Keep the forced range only for workloads that truly need it; otherwise let the workload's
    # configured batch size drive the run.
    if combo.bs_runtime != 1 and not combo.label.startswith('resnet50_'):
        cmd.extend(['--batchsize', '1', str(combo.bs_runtime), str(combo.bs_runtime)])

    log_path = log_path_for(run_tag, combo, 'polaris.log')
    log_path.parent.mkdir(parents=True, exist_ok=True)
    opstats = opstats_path_for(run_tag, combo)
    with log_path.open('w') as f:
        subprocess.run(cmd, cwd=str(REPO_ROOT), stdout=f, stderr=subprocess.STDOUT, check=True,
                       timeout=timeout_s)
    if not opstats.is_file():
        raise FileNotFoundError(f'TT-Polaris ran but opstats CSV not found: {opstats}\nsee log: {log_path}')
    completion_marker_for(run_tag, combo).write_text(
        f'{datetime.now(timezone.utc).isoformat(timespec="seconds")}\n', encoding='utf-8'
    )
    return opstats


def clear_stale_outputs(run_tag: str, combo: Combo, *, keep_opstats: bool) -> None:
    combo_dir = REPO_ROOT / '__output' / run_tag / combo.label
    if not combo_dir.is_dir():
        return
    kept = ([opstats_path_for(run_tag, combo), completion_marker_for(run_tag, combo)]
            if keep_opstats else [])
    protected = {p for keep in kept for p in (keep, *keep.parents)}
    for path in sorted(combo_dir.rglob('*'), key=lambda p: len(p.parts), reverse=True):
        if path in protected:
            continue
        if path.is_symlink() or not path.is_dir():
            path.unlink(missing_ok=True)
        else:
            path.rmdir()


def compare_env() -> dict[str, str]:
    return {**os.environ, 'POLARIS_BASE_DIR': str(REPO_ROOT)}


def run_compare(combo: Combo, run_tag: str, opstats: Path, refrun: Path,
                timeout_s: int = _DEFAULT_TIMEOUT_S) -> tuple[Path, Path]:
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
            with contextlib.suppress(OSError):
                xlsx_path.unlink(missing_ok=True)
            raise
    return xlsx_path, log_path


_SUMMARY_RE = re.compile(
    r'Network total:\s*\n'
    r'\s*(?:TT-)?Polaris:\s*([\d.]+)\s*ms\s*\n'
    r'\s*Profiler:\s*([\d.]+)\s*ms\s*\n'
    r'\s*Gap:\s*(N/A|[+-]?[\d.]+)%?[^\n]*\n'
    r'(?:\s*(?:TT-)?Polaris LUT hits:\s*(\d+)/(\d+))?'
)


def extract_summary(compare_log: Path) -> ParsedSummary | None:
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
    ok: list[tuple[Combo, Path]] = []
    skipped: list[tuple[str, str]] = []
    for c in combos:
        if not (REPO_ROOT / c.archfile).exists():
            reason = f'arch config not on this branch: {c.archfile}'
        else:
            local_lut = local_lut_for_combo(c)
            refrun = local_refrun_for_combo(c)
            try:
                if not local_lut.is_file():
                    reason = f'local LUT missing: {local_lut}'
                elif not refrun.is_file():
                    reason = f'local refrun missing: {refrun}'
                else:
                    ok.append((c, refrun))
                    continue
            except OSError as e:
                reason = f'local cache unusable for {c.label}: {e}'
        print(f'WARN: skipping {c.label!r} - {reason}', file=sys.stderr)
        skipped.append((c.label, reason))
    return ok, skipped


_WORKLOAD_NAMES = {
    'vit': 'ViT',
    'vgg': 'VGG-UNet',
    'resnet50': 'ResNet50',
    'llama3_decode': 'Llama3-8B decode',
    'llama3_prefill': 'Llama3-8B prefill',
}
_ARCH_NAMES = {'p100a': 'Blackhole p100a', 'n150': 'Wormhole n150'}
_MODES = ('lut', 'analytical')
_GREEN_PCT = 10.0  # abs(ERR %) <= 10 is green, > 10 red, N/A grey (same legend as the correlation PDF)


def split_mode(label: str) -> tuple[str, str] | None:
    for mode in _MODES:
        if label.endswith(f'_{mode}'):
            return label[:-len(mode) - 1], mode
    return None


def workload_display(base_label: str) -> str:
    for prefix, name in _WORKLOAD_NAMES.items():
        if base_label.startswith(prefix):
            return name
    return base_label


def _pct(num: float, den: float) -> float | None:
    return None if not den else num / den * 100.0


def read_layer_types(xlsx: Path) -> dict[str, tuple[int, float, int, float]]:
    """'By Layer Type' sheet -> {type: (n_polaris, polaris_ms, n_profiler, profiler_ms)}; TOTAL excluded."""
    import openpyxl  # local import: only needed for the workbook output

    wb = openpyxl.load_workbook(xlsx, read_only=True, data_only=True)
    try:
        rows = list(wb['By Layer Type'].iter_rows(values_only=True))
    finally:
        wb.close()
    col = {name: i for i, name in enumerate(rows[0])}
    out: dict[str, tuple[int, float, int, float]] = {}
    for r in rows[1:]:
        name = r[col['Layer Type']]
        if name is None or str(name).upper() == 'TOTAL':
            continue
        out[str(name)] = (int(r[col['# TT-Polaris']] or 0), float(r[col['TT-Polaris ms']] or 0.0),
                          int(r[col['# Profiler']] or 0), float(r[col['Profiler ms']] or 0.0))
    return out


@dataclass(frozen=True)
class LayerRow:
    layer_type: str
    n_profiler: int
    n_polaris: int
    profiler_ms: float
    lut_ms: float | None
    analytical_ms: float | None


def merge_layer_types(lut: dict | None, ana: dict | None) -> list[LayerRow]:
    names = set(lut or {}) | set(ana or {})

    def prof(n: str) -> float:
        return max((d[n][3] for d in (lut, ana) if d and n in d), default=0.0)

    rows = []
    for n in sorted(names, key=lambda n: (-prof(n), n)):
        src = ana if ana and n in ana else lut
        assert src is not None
        rows.append(LayerRow(
            layer_type=n,
            n_profiler=max((d[n][2] for d in (lut, ana) if d and n in d), default=0),
            n_polaris=src[n][0],
            profiler_ms=prof(n),
            lut_ms=lut[n][1] if lut and n in lut else None,
            analytical_ms=ana[n][1] if ana and n in ana else None,
        ))
    return rows


def _err_gap(prof_ms: float, pol_ms: float | None) -> tuple[float | None, float | None, float | None]:
    """(ERR %, Abs Gap, Gap %) with the correlation-dashboard conventions.

    ERR % = (Polaris - Profiler) / Profiler; Abs Gap = Profiler - Polaris; Gap % = Abs Gap / Polaris.
    """
    if pol_ms is None:
        return None, None, None
    return _pct(pol_ms - prof_ms, prof_ms), prof_ms - pol_ms, _pct(prof_ms - pol_ms, pol_ms)


def write_summary_workbook(run_tag: str, summaries: list[SummaryRow], dest: Path) -> None:
    """Write the correlation-dashboard numbers to an .xlsx: a network 'Summary' sheet (one row
    per workload, LUT and analytical side by side, Mismatch Abs Gap included) and one sheet per
    combo with the per-layer-type table. Plain numeric cells so the values can be copied straight
    into the dashboard."""
    import openpyxl
    from openpyxl.styles import Alignment, Font, PatternFill

    by_label = {c.label: c for c in COMBOS}
    green = PatternFill('solid', fgColor='C6EFCE')
    red = PatternFill('solid', fgColor='FFC7CE')
    grey = PatternFill('solid', fgColor='D9D9D9')
    bold = Font(bold=True)

    def put(ws, row: int, col: int, value, fmt: str | None = None, *, err: bool = False, is_bold=False):
        cell = ws.cell(row=row, column=col, value=value)
        if fmt and isinstance(value, (int, float)):
            cell.number_format = fmt
        if err:
            if value is None:
                cell.value = 'N/A'
                cell.fill = grey
            else:
                cell.fill = green if abs(value) <= _GREEN_PCT else red
        if is_bold:
            cell.font = bold
        return cell

    def header(ws, row: int, names: list[str]) -> None:
        for i, n in enumerate(names, 1):
            c = ws.cell(row=row, column=i, value=n)
            c.font = bold
            c.alignment = Alignment(wrap_text=True, horizontal='center')

    # group the summaries by base combo
    modes: dict[str, dict[str, SummaryRow]] = {}
    for s in summaries:
        sm = split_mode(s.label)
        if sm:
            modes.setdefault(sm[0], {})[sm[1]] = s

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = 'Summary'
    ws.cell(row=1, column=1, value=f'TT-Polaris correlation summary - run_tag {run_tag}').font = bold
    ws.cell(row=2, column=1, value='ERR % = (Polaris - Profiler) / Profiler.  Abs Gap = Profiler - Polaris.  '
                                   'Mismatch Abs Gap = sum over layer types of |Abs Gap|; Mismatch ERR % = that / Profiler total.')
    hdr = ['Workload', 'Profiler ms', 'LUT-correlated ms', 'LUT ERR %', 'Analytical ms',
           'Analytical ERR %', 'Analytical mismatch Abs Gap (ms)', 'Analytical mismatch ERR %']
    header(ws, 4, hdr)

    layer_sheets: list[tuple[str, list[LayerRow], str]] = []
    r = 5
    order = [c.label for c in COMBOS]
    for base in sorted(modes, key=lambda b: (b not in order, order.index(b) if b in order else 0)):
        combo = by_label.get(base)
        files: dict[str, dict | None] = {}
        for mode in _MODES:
            if mode in modes[base]:
                try:
                    files[mode] = read_layer_types(
                        REPO_ROOT / '__output' / run_tag / f'{base}_{mode}' / f'{base}_{mode}_comparison.xlsx')
                except (OSError, KeyError, ValueError) as e:
                    print(f'WARN: no layer-type data for {base}_{mode}: {e}', file=sys.stderr)
                    files[mode] = None
            else:
                files[mode] = None
        layers = merge_layer_types(files['lut'], files['analytical'])
        layer_sheets.append((base, layers, combo.filterarch if combo else ''))

        any_s = next(iter(modes[base].values()))
        prof_ms = any_s.parsed.profiler_ms
        lut_ms = modes[base]['lut'].parsed.polaris_ms if 'lut' in modes[base] else None
        ana_ms = modes[base]['analytical'].parsed.polaris_ms if 'analytical' in modes[base] else None
        mismatch = (sum(abs(row.profiler_ms - row.analytical_ms) for row in layers
                        if row.analytical_ms is not None) if files['analytical'] else None)
        arch = _ARCH_NAMES.get(combo.filterarch, '') if combo else ''
        put(ws, r, 1, f'{workload_display(base)} ({arch}, bs={any_s.bs})' if arch else workload_display(base))
        put(ws, r, 2, prof_ms, '0.000')
        put(ws, r, 3, lut_ms, '0.000')
        put(ws, r, 4, _pct(lut_ms - prof_ms, prof_ms) if lut_ms is not None else None, '+0.0;-0.0;0.0', err=True)
        put(ws, r, 5, ana_ms, '0.000')
        put(ws, r, 6, _pct(ana_ms - prof_ms, prof_ms) if ana_ms is not None else None, '+0.0;-0.0;0.0', err=True)
        put(ws, r, 7, mismatch, '0.00')
        put(ws, r, 8, _pct(mismatch, prof_ms) if mismatch is not None else None, '+0.0;-0.0;0.0', err=True)
        r += 1
    for col, w in zip('ABCDEFGH', (38, 13, 18, 12, 14, 17, 22, 22)):
        ws.column_dimensions[col].width = w
    ws.freeze_panes = 'A5'

    lhdr = ['Layer type', '# Profiler ops', '# Polaris ops', 'Profiler ms',
            'LUT ms', 'LUT ERR %', 'LUT Abs Gap (ms)', 'LUT Gap %',
            'Analytical ms', 'Analytical ERR %', 'Analytical Abs Gap (ms)', 'Analytical Gap %',
            'Mismatch Abs Gap (ms)', 'Mismatch %']
    for base, layers, _arch in layer_sheets:
        sh = wb.create_sheet(base[:31])
        sh.cell(row=1, column=1, value=f'{workload_display(base)} - {base}').font = bold
        header(sh, 3, lhdr)
        total_mismatch = sum(abs(x.profiler_ms - x.analytical_ms) for x in layers if x.analytical_ms is not None)
        rr = 4
        for x in layers:
            l_err, l_gap, l_gp = _err_gap(x.profiler_ms, x.lut_ms)
            a_err, a_gap, a_gp = _err_gap(x.profiler_ms, x.analytical_ms)
            mism = abs(a_gap) if a_gap is not None else None
            put(sh, rr, 1, x.layer_type)
            put(sh, rr, 2, x.n_profiler)
            put(sh, rr, 3, x.n_polaris)
            put(sh, rr, 4, x.profiler_ms, '0.0000')
            put(sh, rr, 5, x.lut_ms, '0.0000')
            put(sh, rr, 6, l_err, '+0.0;-0.0;0.0', err=l_err is not None or x.lut_ms is None)
            put(sh, rr, 7, l_gap, '+0.0000;-0.0000;0.0000')
            put(sh, rr, 8, l_gp, '+0.0;-0.0;0.0')
            put(sh, rr, 9, x.analytical_ms, '0.0000')
            put(sh, rr, 10, a_err, '+0.0;-0.0;0.0', err=True)
            put(sh, rr, 11, a_gap, '+0.0000;-0.0000;0.0000')
            put(sh, rr, 12, a_gp, '+0.0;-0.0;0.0')
            put(sh, rr, 13, mism, '0.00')
            put(sh, rr, 14, _pct(mism, total_mismatch) if mism is not None else None, '0')
            rr += 1
        # total row
        t_prof = sum(x.profiler_ms for x in layers)
        t_lut = sum(x.lut_ms for x in layers if x.lut_ms is not None) if any(x.lut_ms is not None for x in layers) else None
        t_ana = sum(x.analytical_ms for x in layers if x.analytical_ms is not None) if any(x.analytical_ms is not None for x in layers) else None
        l_err, l_gap, l_gp = _err_gap(t_prof, t_lut)
        a_err, a_gap, a_gp = _err_gap(t_prof, t_ana)
        put(sh, rr, 1, f'{workload_display(base)} total', is_bold=True)
        put(sh, rr, 2, sum(x.n_profiler for x in layers), is_bold=True)
        put(sh, rr, 3, sum(x.n_polaris for x in layers), is_bold=True)
        put(sh, rr, 4, t_prof, '0.0000', is_bold=True)
        put(sh, rr, 5, t_lut, '0.0000', is_bold=True)
        put(sh, rr, 6, l_err, '+0.0;-0.0;0.0', err=True)
        put(sh, rr, 7, l_gap, '+0.0000;-0.0000;0.0000', is_bold=True)
        put(sh, rr, 8, l_gp, '+0.0;-0.0;0.0', is_bold=True)
        put(sh, rr, 9, t_ana, '0.0000', is_bold=True)
        put(sh, rr, 10, a_err, '+0.0;-0.0;0.0', err=True)
        put(sh, rr, 11, a_gap, '+0.0000;-0.0000;0.0000', is_bold=True)
        put(sh, rr, 12, a_gp, '+0.0;-0.0;0.0', is_bold=True)
        put(sh, rr, 13, total_mismatch if t_ana is not None else None, '0.00', is_bold=True)
        put(sh, rr, 14, 100 if t_ana is not None else None, '0', is_bold=True)
        sh.column_dimensions['A'].width = 30
        for ci in range(2, 15):
            sh.column_dimensions[openpyxl.utils.get_column_letter(ci)].width = 14
        sh.freeze_panes = 'B4'

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + '.partial')
    wb.save(tmp)
    os.replace(tmp, dest)


def emit_workbook(run_tag: str, summaries: list[SummaryRow]) -> None:
    if not summaries:
        return
    for dest in (REPO_ROOT / '__output' / run_tag / 'summary.xlsx', REPO_ROOT / '__output' / 'summary.xlsx'):
        try:
            write_summary_workbook(run_tag, summaries, dest)
            print(f'Workbook saved: {dest.relative_to(REPO_ROOT)}')
        except (OSError, ImportError, KeyError, ValueError) as e:
            print(f'WARNING: could not write {dest}: {e}', file=sys.stderr)


def emit_report(run_tag: str, summaries: list[SummaryRow], problems: list[tuple[str, str]],
                n_requested: int) -> bool:
    # One row per workload (LUT-on and analytical runs merged), same columns as summary.xlsx.
    modes: dict[str, dict[str, SummaryRow]] = {}
    for s in summaries:
        sm = split_mode(s.label)
        if sm:
            modes.setdefault(sm[0], {})[sm[1]] = s
    known = {c.label: c for c in COMBOS}
    order = [c.label for c in COMBOS]

    def fmt_ms(v: float | None) -> str:
        return 'N/A' if v is None else f'{v:.4f}'

    def fmt_err(v: float | None) -> str:
        return 'N/A' if v is None else f'{v:+.1f}%'

    table: list[list[str]] = []
    for base in sorted(modes, key=lambda b: (b not in order, order.index(b) if b in order else 0)):
        combo = known.get(base)
        any_s = next(iter(modes[base].values()))
        prof = any_s.parsed.profiler_ms
        lut = modes[base]['lut'].parsed.polaris_ms if 'lut' in modes[base] else None
        ana = modes[base]['analytical'].parsed.polaris_ms if 'analytical' in modes[base] else None
        mism = None
        if 'analytical' in modes[base]:
            try:
                layers = read_layer_types(REPO_ROOT / '__output' / run_tag / f'{base}_analytical'
                                          / f'{base}_analytical_comparison.xlsx')
                mism = sum(abs(v[3] - v[1]) for v in layers.values())
            except (OSError, KeyError, ValueError, ImportError):
                mism = None
        arch = _ARCH_NAMES.get(combo.filterarch, '') if combo else ''
        name = f'{workload_display(base)} ({arch}, bs={any_s.bs})' if arch else workload_display(base)
        table.append([
            name, fmt_ms(prof), fmt_ms(lut),
            fmt_err(_pct(lut - prof, prof) if lut is not None else None),
            fmt_ms(ana),
            fmt_err(_pct(ana - prof, prof) if ana is not None else None),
            fmt_ms(mism),
            fmt_err(_pct(mism, prof) if mism is not None else None),
        ])
    cols = ['Workload', 'Profiler ms', 'LUT-correlated ms', 'LUT ERR %', 'Analytical ms',
            'Analytical ERR %', 'Mismatch Abs Gap (ms)', 'Mismatch ERR %']
    widths = [max(len(cols[i]), *(len(r[i]) for r in table)) if table else len(cols[i])
              for i in range(len(cols))]
    header = '| ' + ' | '.join(f'{c:<{widths[0]}}' if i == 0 else f'{c:>{widths[i]}}'
                               for i, c in enumerate(cols)) + ' |'
    sep = '|' + '|'.join('-' * (w + 2) for w in widths) + '|'
    rows = ['| ' + ' | '.join(f'{v:<{widths[0]}}' if i == 0 else f'{v:>{widths[i]}}'
                              for i, v in enumerate(r)) + ' |' for r in table]
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

    provenance = [
        f'- `{b}`: arch `{known[b].filterarch}`, workload `{known[b].workload_yaml_name}`, '
        f'instance `{known[b].instance_name}`, bs {known[b].bs_runtime}, '
        f'reference `{known[b].refrun_rel}`'
        for b in modes if b in known
    ]
    md_lines = [
        f'# Analytical projection + compare summary - run_tag `{run_tag}`',
        '',
        f'Generated: {datetime.now(timezone.utc).isoformat(timespec="seconds")}',
        f'Runs reported: {len(summaries)} of {n_requested} requested ({len(COMBOS)} combos known)',
        '',
        'ERR % = (Polaris - Profiler) / Profiler. Mismatch Abs Gap = sum over layer types of '
        '|Profiler - Analytical|; Mismatch ERR % = that / Profiler.',
        '',
    ]
    md_lines += [header, sep, *rows, ''] if rows else [f'_{no_rows_note}_', '']
    if provenance:
        md_lines += ['## What each row projected', '', *provenance, '']
    if problem_lines:
        md_lines += ['## Not reported', '', *problem_lines, '']

    summary_paths = [summary_path_for(run_tag), REPO_ROOT / '__output' / 'summary.md']
    wrote_any = False
    for summary_path in summary_paths:
        tmp_path = summary_path.with_name(summary_path.name + '.partial')
        try:
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path.unlink(missing_ok=True)
            with open(tmp_path, 'x', encoding='utf-8') as fh:
                fh.write('\n'.join(md_lines))
            os.replace(tmp_path, summary_path)
            wrote_any = True
        except OSError as e:
            with contextlib.suppress(OSError):
                tmp_path.unlink(missing_ok=True)
            print()
            print(f'WARNING: could not write {summary_path}: {e}', file=sys.stderr)
            print('The table above is the whole report; any earlier report is unchanged.',
                  file=sys.stderr)
    if not wrote_any:
        return False
    print()
    for summary_path in summary_paths:
        try:
            print(f'Summary saved: {summary_path.relative_to(REPO_ROOT)}')
        except ValueError:
            print(f'Summary saved: {summary_path}')
    return True


_SAFE_RUN_TAG = re.compile(r'[A-Za-z0-9_][A-Za-z0-9._-]*')
_MAX_RUN_TAG = 128


def timeout_arg(value: str) -> int:
    try:
        seconds = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f'must be an integer number of seconds, got {value!r}')
    if seconds < 1:
        raise argparse.ArgumentTypeError(f'must be at least 1 second, got {seconds}')
    return seconds


def run_tag_arg(value: str) -> str:
    if not _SAFE_RUN_TAG.fullmatch(value):
        raise argparse.ArgumentTypeError(
            f'must be a single relative directory name matching [A-Za-z0-9_][A-Za-z0-9._-]*, '
            f'got {value!r}')
    if len(value.encode()) > _MAX_RUN_TAG:
        raise argparse.ArgumentTypeError(
            f'must be at most {_MAX_RUN_TAG} bytes, got {len(value.encode())}')
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run-tag', type=run_tag_arg,
                        help='Output subdirectory tag, e.g. "session_2026-05-25". Required unless --list is used.')
    parser.add_argument('--timeout-per-combo', type=timeout_arg, default=_DEFAULT_TIMEOUT_S,
                        metavar='SECONDS',
                        help=f'Ceiling for each TT-Polaris and compare_layers subprocess (default {_DEFAULT_TIMEOUT_S}).')
    parser.add_argument('--ignore-lut', action='store_true',
                        help='Run the analytical model by forcing TT-Polaris --ignore-lut. This is the default behavior for this analytical script.')
    parser.add_argument('--with-lut', action='store_true',
                        help='Run the LUT-backed path instead of the analytical path for comparison. Not the default.')
    parser.add_argument('--only', metavar='COMBO_LABEL', action='append',
                        help='Run only the named combo(s). Repeatable.')
    parser.add_argument('--list', action='store_true', help='List available combo labels and exit')
    args = parser.parse_args()

    if args.list:
        width = max(len(c.label) for c in COMBOS)
        for c in COMBOS:
            print(f'{c.label:<{width}}  arch={c.filterarch:<5}  bs={c.bs_runtime}  wli={c.filterwli}')
        return 0

    if not args.run_tag:
        parser.error('--run-tag is required (unless using --list)')

    if args.with_lut and args.ignore_lut:
        parser.error('choose either --ignore-lut (analytical) or --with-lut (LUT-backed), not both')

    known = {c.label for c in COMBOS}
    if args.only:
        unknown = sorted(set(args.only) - known)
        if unknown:
            parser.error(f'unknown --only label(s): {", ".join(unknown)}. Known labels: {", ".join(sorted(known))}')

    if args.ignore_lut and args.with_lut:
        parser.error('choose either --ignore-lut (analytical) or --with-lut (LUT-backed), not both')

    selected = COMBOS if not args.only else [c for c in COMBOS if c.label in args.only]
    modes = ['analytical', 'lut'] if not args.ignore_lut and not args.with_lut else (
        ['analytical'] if args.ignore_lut else ['lut']
    )
    n_requested = len(selected) * len(modes)

    output_root = REPO_ROOT / '__output'
    run_dir = output_root / args.run_tag
    try:
        for path in (output_root, run_dir):
            if path.resolve() != REPO_ROOT.resolve() / path.relative_to(REPO_ROOT):
                print(f'refusing to run: {path.relative_to(REPO_ROOT)} does not resolve to its own place in the repo, so this tool would delete in a directory it does not own', file=sys.stderr)
                return 1
            if path.exists() and not path.is_dir():
                print(f'refusing to run: {path.relative_to(REPO_ROOT)} exists and is not a directory', file=sys.stderr)
                return 1
        summary_path_for(args.run_tag).unlink(missing_ok=True)
    except OSError as e:
        print(f'refusing to run: cannot use {run_dir.relative_to(REPO_ROOT)}: {e}', file=sys.stderr)
        return 1

    problems: list[tuple[str, str]] = []
    cleared: list[Combo] = []
    for combo in selected:
        for mode in modes:
            mode_combo = replace(combo, label=f'{combo.label}_{mode}')
            try:
                clear_stale_outputs(args.run_tag, mode_combo, keep_opstats=False)
            except OSError as e:
                reason = f'could not clear stale outputs: {e}'
                print(f'  FAILED: {mode_combo.label}: {reason}', file=sys.stderr)
                problems.append((mode_combo.label, reason))
                continue
            cleared.append(mode_combo)

    runnable, validation_problems = validate_combos(cleared)
    problems += validation_problems
    if not runnable:
        print('No runnable combos - see "Not reported" below for the reason on each.', file=sys.stderr)
        emit_report(args.run_tag, [], problems, n_requested)
        return 1

    summaries: list[SummaryRow] = []
    for combo, refrun in runnable:
        print(f'=== {combo.label} ===', flush=True)
        mode = 'analytical' if combo.label.endswith('_analytical') else 'lut' if combo.label.endswith('_lut') else 'unknown'
        try:
            print(f'  refrun:  {refrun.relative_to(REPO_ROOT)}')
            opstats = opstats_path_for(args.run_tag, combo)
            if mode == 'analytical':
                print(f'  running TT-Polaris in analytical mode (study={args.run_tag}/{combo.label})...', flush=True)
                opstats = run_polaris(combo, args.run_tag, args.timeout_per_combo, ignore_lut=True)
            else:
                print(f'  running TT-Polaris with LUT path (study={args.run_tag}/{combo.label})...', flush=True)
                opstats = run_polaris(combo, args.run_tag, args.timeout_per_combo, ignore_lut=False)
            print(f'  opstats: {opstats.relative_to(REPO_ROOT)}')
            print('  running compare_layers...', flush=True)
            xlsx, compare_log = run_compare(combo, args.run_tag, opstats, refrun, args.timeout_per_combo)
            print(f'  xlsx:    {xlsx.relative_to(REPO_ROOT)}')
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
    emit_workbook(args.run_tag, summaries)
    if not written:
        return 1
    if not summaries:
        print('No combo produced a parseable summary.', file=sys.stderr)
        return 1
    if problems:
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
