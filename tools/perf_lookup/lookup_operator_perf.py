#!/usr/bin/env python3
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Operator performance lookup: tt-perf **master** YAML only (``correqn.tt-perf-master``).

Loads via ``tools.perf_lookup.tt_perf_master_loader.load_existing_yaml``. Maps workload ops + tensors to a
logical **9-tuple** (one input), **16-tuple** (two inputs), or **23-tuple** (three inputs) key, or to a per-op
variant: **16-tuple** for ``halo`` (standard 9 + kernel_h/w, stride_h/w, padding_h/w, is_transpose from op.attrs) and
**10-tuple** for ``interleavedtosharded`` (standard 9 + ``output_0_memory`` from the output tensor). Resolves
``single`` (flat ``num_cores`` + stat scalars), ``curve``, and ``hybrid``. For hybrid rows, whether ``curve`` is used
is controlled by :class:`OperatorPerfMap` ``use_hybrid_curve`` (default ``False``: ``single`` only).
See ``doc/tools/perf_lookup/LOOKUP_TABLE_MASTER.md``.

On a **hit** (``msecs`` resolves), **``matrix_pipe_util``** and **``vector_pipe_util``** must resolve
(finite percentages in **[0, 100]** inclusive; **0** allowed). Optional util keys
``mem_util``, ``noc_util``, ``noc_multicast_util``, ``npe_cong_impact_pct`` are validated the same way
when present. Failures raise :class:`OperatorPerfLUTValidationError` (``Device`` re-raises and terminates).

TTNN stores logical BF16 as ``numpy.float16`` (see ``ttsim/front/ttnn/tensor.py``); for LUT keys we
treat that storage as ``BFLOAT16`` unless ``op_precision`` is IEEE FP16 (``fp16`` / ``FLOAT16``).

Master lookup keys always use each tensor's **logical** ``shape`` (rank-4 WZYX after promotion), not
tile-padded extents; ``padded_shape`` is ignored for key construction.

For ``reshape`` master keys, ``input_0`` logical WZYX follows tt-perf convention ``(1, 1, w*z*y, x)``
from the tensor's rank-4 logical ``(w, z, y, x)`` (see :func:`tools.profiling.shape_canonical.reshape_input0_wzyx`).

For ``add``, after a 15-tuple miss, lookup may retry with a key that duplicates the full operand's
WZYX on the broadcast operand when one side is exactly ``(1, 1, 1, X)`` and the other is a
non-trivial rank-4 shape with the same ``X`` (see :func:`build_master_key_tuple_15_add_broadcast_duplicate_full`).
"""

from __future__ import annotations

import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

# Repo root so ``from tools.perf_lookup…`` works when this file is run as a script
# (``python tools/perf_lookup/lookup_operator_perf.py``); pytest uses ``pythonpath = .``.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.perf_lookup.tt_perf_master_loader import load_existing_yaml
from tools.profiling.shape_canonical import (
    coerce_shape_to_list,
    createqkvheads_input0_wzyx,
    promote_to_rank4,
    reshape_input0_wzyx,
    tensor_layout_str,
    tensor_datatype,
    tensor_memory_str,
    precision_to_master_datatype,
)
from tools.perf_lookup.tt_perf_master_schema import (
    STANDARD_KEY_TUPLE_LENGTHS,
    MASTER_CURVE_FAMILY_KEY,
    MASTER_CURVE_FAMILY_LINEAR,
    MASTER_CURVE_FAMILY_POWER,
    MASTER_DURATION_MS_KEY,
    MASTER_ENTRY_TYPE_CURVE,
    MASTER_ENTRY_TYPE_HYBRID,
    MASTER_ENTRY_TYPE_KEY,
    MASTER_ENTRY_TYPE_SINGLE,
    MASTER_HYBRID_CURVE_KEY,
    MASTER_HYBRID_SINGLE_KEY,
    MASTER_SINGLE_NUM_CORES_KEY,
    MASTER_STATS_COVERAGE_FULL,
    MASTER_STATS_COVERAGE_TIME_ONLY,
    master_coverage_counts,
    payload_stats_coverage,
    tuple_to_labeled_key_map,
    MATH_FIDELITY_NA,
)

from loguru import logger

_DEFAULT_CORE_COUNT_FALLBACK = 64

# Profiler rows may use an 8-tuple (input_0 only) while the workload graph lists two operands
# (15-tuple). After a full binary key miss, try the first-input 8-tuple for these opcodes.
# ``reshape``: second input is often a small shape/constant tensor, not profiled as input_1.
_BINARY_LUT_FALLBACK_TO_INPUT0_KEY_OPCODES = frozenset({"mul", "reshape"})

# Hardware ops that are arity-1 in Polaris but arity-2 in profiler output (src + dst with
# same shape/layout/memory).  After an arity-1 (9-tuple) miss, try a 16-tuple with t0 used
# for both input_0 and input_1 positions.
_UNARY_POLARIS_BINARY_HW_OPCODES = frozenset({"move"})

# InterleavedToSharded: hardware records ITS from DRAM in most VGG UNet decoder stages
# (ShardedToInterleaved output lands in DRAM on device), but Polaris models the STI output
# as L1_INTERLEAVED (per model code using ttnn.L1_MEMORY_CONFIG).  After an L1_INTERLEAVED
# arity-1 miss, try DRAM_INTERLEAVED — same shape/layout, just different input staging.
_ITS_MEM_FALLBACK_OPCODES = frozenset({"interleavedtosharded"})
_L1_INTERLEAVED_STR = "DEV_1_L1_INTERLEAVED"
_DRAM_INTERLEAVED_STR = "DEV_1_DRAM_INTERLEAVED"

# Halo and element-wise ops (Sigmoid): the LUT was built with TILE layout while Polaris models
# VGG UNet tensors as ROW_MAJOR throughout.  After a ROW_MAJOR arity-1 miss, retry with TILE
# — position 5 of the 9-tuple.  Timing should be layout-insensitive for these data-movement /
# element-wise ops.
_LAYOUT_ROWMAJOR_TO_TILE_FALLBACK_OPCODES = frozenset({"halo", "sigmoid"})
_ROW_MAJOR_STR = "ROW_MAJOR"
_TILE_STR = "TILE"

# Device-index reconciliation between sim and LUT.
# The sim's key builder (shape_canonical.tensor_memory_str) hardcodes a "DEV_1_"
# prefix on every memory tag (a fixed single-chip placeholder — Polaris is a
# single-chip simulator), whereas a LUT built from a profiler capture carries the
# real silicon DEVICE ID (e.g. "DEV_0_" for device 0). Keys are compared verbatim,
# so a capture-built LUT would miss on every memory field purely because of the
# device prefix. The device-normalized fallback (see lookup) reconciles this WITHOUT
# mutating stored LUT data: it strips the DEV_<n>_ prefix on both sides and matches
# only when exactly one LUT entry agrees on every non-device field. Genuine
# multi-device captures (>1 entry differing only in device) stay distinct and are
# left to exact matching — the fallback declines rather than guess.
_DEVICE_PREFIX_RE = re.compile(r"^DEV_\d+_", re.IGNORECASE)


def _device_normalized_key(key: tuple) -> tuple:
    """Return *key* with the ``DEV_<n>_`` prefix stripped from every memory-tag
    field, so keys that differ only in device index compare equal."""
    return tuple(
        _DEVICE_PREFIX_RE.sub("", x) if isinstance(x, str) else x for x in key
    )

# VGG UNet decoder: Polaris propagates HEIGHT_SHARDED through the post-concat ITS path, but
# the LUT records those ops as BLOCK_SHARDED (hardware auto-selects BLOCK for smaller tensors).
# Applies to arity-1 (halo, move) and arity-3 (conv2d, convtranspose) ops — position 7 of the
# lookup key is input_0_memory in both 9-tuple and 23-tuple formats.
_HALO_HEIGHT_TO_BLOCK_FALLBACK_OPCODES = frozenset({"halo", "move", "conv2d", "convtranspose"})
_HEIGHT_SHARDED_STR = "DEV_1_L1_HEIGHT_SHARDED"
_BLOCK_SHARDED_STR = "DEV_1_L1_BLOCK_SHARDED"

# Percentages 0–100 inclusive in master YAML / curve-evaluated stats (not 0–1 fractions).
LUT_OPTIONAL_UTIL_PERCENT_KEYS = frozenset(
    {"mem_util", "noc_util", "noc_multicast_util", "npe_cong_impact_pct"}
)


class OperatorPerfLUTValidationError(ValueError):
    """Raised when a matched LUT row is missing required stats or has invalid utilization percentages."""


def _raise_lut_validation(lut_path: Path, key_t: tuple, detail: str) -> None:
    raise OperatorPerfLUTValidationError(
        f"Operator perf LUT validation failed (file={lut_path}, key={key_t}): {detail}"
    )


def _validate_required_util_percent(
    name: str, raw: Any, lut_path: Path, key_t: tuple
) -> float:
    """Require a finite, non-negative percentage. ``raw`` is the resolved scalar from the LUT row.

    Values above 100 are accepted with a single one-time warning per field name —
    TT-Metal records multicast/overcount cases where DRAM BW UTIL etc. can exceed
    100% legitimately.

    TODO(util-over-100): mirrors the placeholder in
    ``tools/profiling/ops_perf_three_csv_merge.py::validate_utilization_cells``.
    When the merge step is updated to clip / fix-at-source, this loader-side
    relaxation should be tightened back to a hard error.
    """
    if raw is None:
        _raise_lut_validation(
            lut_path,
            key_t,
            f"required field {name!r} is missing or null; when a LUT row matches, "
            "matrix_pipe_util and vector_pipe_util must both be provided "
            "(percentages >= 0; 0 is allowed).",
        )
    if isinstance(raw, bool):
        _raise_lut_validation(
            lut_path, key_t, f"field {name!r} has invalid type bool, value={raw!r}"
        )
    if not isinstance(raw, (int, float)):
        _raise_lut_validation(
            lut_path,
            key_t,
            f"field {name!r} must be a number, got {type(raw).__name__}, value={raw!r}",
        )
    v = float(raw)
    if not math.isfinite(v):
        _raise_lut_validation(lut_path, key_t, f"field {name!r} must be finite, got {raw!r}")
    if v < 0.0:
        _raise_lut_validation(
            lut_path,
            key_t,
            f"field {name!r} must be non-negative, got {v!r}",
        )
    if v > 100.0:
        _warn_util_over_100_once(name, v, lut_path)
    return v


def _validate_optional_util_percent(
    name: str, raw: Any, lut_path: Path, key_t: tuple
) -> Optional[float]:
    """If ``raw`` is None, return None; else same rules as required util percent."""
    if raw is None:
        return None
    if isinstance(raw, bool):
        _raise_lut_validation(
            lut_path, key_t, f"field {name!r} has invalid type bool, value={raw!r}"
        )
    if not isinstance(raw, (int, float)):
        _raise_lut_validation(
            lut_path,
            key_t,
            f"field {name!r} must be a number or null, got {type(raw).__name__}, value={raw!r}",
        )
    v = float(raw)
    if not math.isfinite(v):
        _raise_lut_validation(
            lut_path, key_t, f"field {name!r} must be finite when present, got {raw!r}"
        )
    if v < 0.0:
        _raise_lut_validation(
            lut_path,
            key_t,
            f"field {name!r} must be non-negative when present, got {v!r}",
        )
    if v > 100.0:
        _warn_util_over_100_once(name, v, lut_path)
    return v


# One-shot warning bookkeeping: keep the log quiet when many rows of the same
# field exceed 100 (common for DRAM BW UTIL on multicast-heavy ops).
_OVER_100_WARNED: set[str] = set()


def _warn_util_over_100_once(name: str, v: float, lut_path: Path) -> None:
    if name in _OVER_100_WARNED:
        return
    _OVER_100_WARNED.add(name)
    logger.warning(
        "LUT field {!r} value {} >100 in {} (TT-Metal multicast/overcount; "
        "warning shown once per field — subsequent occurrences silent)",
        name, v, lut_path,
    )


@dataclass(frozen=True)
class MasterPerfStats:
    """Resolved profiler row for one op (after single/curve/hybrid evaluation).

    ``matrix_pipe_util`` and ``vector_pipe_util`` are percentages, typically in
    **[0, 100]** (not fractions). Values >100 are accepted with a one-time warning
    per field — TT-Metal multicast/overcount can legitimately exceed 100% (see
    ``_warn_util_over_100_once`` above). Downstream consumers must not assume a
    hard 100% cap. ``Device.get_exec_stats`` divides by 100 for exec_stats / CSV.
    Optional ``mem_util`` is the same.

    ``key_literal`` is the tuple originally built from the workload op + its tensor state
    (the *would-have-been* key before any fallback substitution).  ``key_resolved`` is the
    tuple the lookup actually matched in the LUT — i.e. the literal key after any
    HEIGHT→BLOCK, L1→DRAM, ROW_MAJOR→TILE, or arity-1→arity-2 fallback chain in
    ``OperatorPerfMap.lookup``.  Downstream tooling (compare_layers ``--by-lut-key``)
    uses ``key_resolved`` so polaris and profiler ops that semantically share a LUT
    entry group together regardless of the literal-shape divergence the runtime
    fallback chain papers over.
    """

    msecs: float
    matrix_pipe_util: float
    vector_pipe_util: float
    memory_traffic: Optional[float] = None
    mem_util: Optional[float] = None
    key_literal: Optional[tuple] = None
    key_resolved: Optional[tuple] = None
    # Diagnostic label for which lookup path produced the hit. One of:
    #   "direct"                   — literal key matched on the first lookup
    #   "add_broadcast_dup"        — add op: broadcast operand duplicated to full shape
    #   "move_arity_dup"           — Move/STS/ITS/Reshard: arity-1 → arity-2 duplicate
    #   "move_arity_dup_tile"      — above + ROW_MAJOR→TILE substitution
    #   "move_arity_dup_block"     — above + HEIGHT→BLOCK substitution
    #   "its_l1_to_dram"           — ITS: input_0_memory L1_INTERLEAVED → DRAM_INTERLEAVED
    #   "halo_rowmajor_to_tile"    — Halo/Sigmoid: input_0_layout ROW_MAJOR → TILE
    #   "halo_height_to_block"     — Halo/Conv/ConvTranspose: input_0_memory HEIGHT → BLOCK
    #   "halo_tile_block_combined" — above + ROW_MAJOR→TILE simultaneously
    #   "binary_to_unary"          — mul/reshape: drop input_1 from 15-tuple to 9-tuple
    # When the literal key matched directly the value is "direct"; when ``lookup``
    # returns None entirely (analytical fallback in the caller), no MasterPerfStats
    # is produced, so device.py emits "analytical" in the CSV based on the absence.
    hit_source: Optional[str] = None
    # Coverage of the payload that produced this hit: "full", or "time_only" when the LUT entry
    # has no NOC-trace stats. For "time_only", memory_traffic and mem_util are None: not measured.
    stats_coverage: str = MASTER_STATS_COVERAGE_FULL


def _eval_curve_value(family: str, a: float, b: float, core_count: int) -> float:
    """Evaluate regression at ``core_count`` (Core_Count in master spec)."""
    n = float(core_count)
    if family == MASTER_CURVE_FAMILY_LINEAR:
        return a * n + b
    if family == MASTER_CURVE_FAMILY_POWER:
        if n <= 0:
            return b
        return a * (n**b)
    raise ValueError(f"Unknown curve family: {family!r}")


def _resolve_scalar_from_flat(flat: dict, stat_name: str) -> Optional[float]:
    v = flat.get(stat_name)
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    return None


def _resolve_stat_from_flat_single(flat: dict, stat_name: str) -> Optional[float]:
    """Read a stat scalar from a flat single payload (``num_cores`` + ``MASTER_SINGLE_STAT_KEYS``)."""
    return _resolve_scalar_from_flat(flat, stat_name)


def _resolve_stat_from_curve(
    curve_payload: dict, stat_name: str, core_count: int
) -> Optional[float]:
    """Evaluate one stat from a curve mapping (top-level ``curve`` or ``hybrid.curve``)."""
    family = curve_payload.get(MASTER_CURVE_FAMILY_KEY)
    if family not in (MASTER_CURVE_FAMILY_LINEAR, MASTER_CURVE_FAMILY_POWER):
        return None
    sub = curve_payload.get(stat_name)
    if not isinstance(sub, dict):
        return None
    a, b = sub.get("a"), sub.get("b")
    if a is None or b is None:
        return None
    return _eval_curve_value(family, float(a), float(b), core_count)


def _resolved_payload_coverage(entry_val: dict, entry_type: str, use_hybrid_curve: bool) -> str:
    """Stats coverage of the payload ``_build_stat_resolver`` reads for this entry."""
    if entry_type == MASTER_ENTRY_TYPE_HYBRID:
        curve_sub = entry_val.get(MASTER_HYBRID_CURVE_KEY)
        if use_hybrid_curve and isinstance(curve_sub, dict):
            return payload_stats_coverage(curve_sub)
        single_sub = entry_val.get(MASTER_HYBRID_SINGLE_KEY)
        return payload_stats_coverage(single_sub) if isinstance(single_sub, dict) else MASTER_STATS_COVERAGE_FULL
    return payload_stats_coverage(entry_val)


def _maybe_warn_single_core_mismatch(
    flat: dict, core_count: int, key_t: tuple, lut_path: Path
) -> None:
    nc = flat.get(MASTER_SINGLE_NUM_CORES_KEY)
    if nc is None:
        return
    try:
        n_table = int(nc)
    except (TypeError, ValueError):
        return
    if int(core_count) != n_table:
        logger.debug(
            "Perf lookup single-entry: runtime core_count={} != table num_cores={} "
            "(using table scalars as-is) key={} lut={}",
            core_count,
            n_table,
            key_t,
            lut_path,
        )


def _build_stat_resolver(
    entry_val: dict,
    entry_type: str,
    core_count: int,
    key_t: tuple,
    lut_path: Path,
    *,
    use_hybrid_curve: bool = False,
) -> Optional[Callable[[str], Optional[float]]]:
    """Return ``resolve(stat_name)`` for the loaded entry type, or ``None`` if unsupported."""
    if entry_type == MASTER_ENTRY_TYPE_SINGLE:
        flat = {k: v for k, v in entry_val.items() if k != MASTER_ENTRY_TYPE_KEY}
        _maybe_warn_single_core_mismatch(flat, core_count, key_t, lut_path)

        def resolve(stat: str) -> Optional[float]:
            return _resolve_stat_from_flat_single(flat, stat)

        return resolve

    if entry_type == MASTER_ENTRY_TYPE_CURVE:

        def resolve(stat: str) -> Optional[float]:
            return _resolve_stat_from_curve(entry_val, stat, core_count)

        return resolve

    if entry_type == MASTER_ENTRY_TYPE_HYBRID:
        curve_sub = entry_val.get(MASTER_HYBRID_CURVE_KEY)
        if use_hybrid_curve and isinstance(curve_sub, dict):

            def resolve(stat: str) -> Optional[float]:
                return _resolve_stat_from_curve(curve_sub, stat, core_count)

            return resolve

        single_sub = entry_val.get(MASTER_HYBRID_SINGLE_KEY)
        if isinstance(single_sub, dict):
            _maybe_warn_single_core_mismatch(single_sub, core_count, key_t, lut_path)

            def resolve(stat: str) -> Optional[float]:
                return _resolve_stat_from_flat_single(single_sub, stat)

            return resolve

    return None


def _op_code(op: Any) -> str:
    from tools.profiling.op_canonical import normalize_polaris_optype

    return normalize_polaris_optype(str(getattr(op, "optype", "")).strip())


def _shape_wzyx(tensor: Any) -> Tuple[int, int, int, int]:
    """Rank-4 WZYX shape for LUT key construction.

    Prefers tensor.hw_shape (NHWC-flattened [1, 1, N*H*W, C] set by conv/pool
    shape-inference) over tensor.shape (logical NCHW [N, C, H, W]).  hw_shape
    matches the representation hardware profiler output uses so LUT keys agree.
    Falls back to promoting the logical shape when hw_shape is absent.

    When ``tensor.x_pad_logical`` or ``tensor.y_pad_logical`` is set (tagged by the
    arch-aware annotation pass in ``ttsim/back/device.py``), the returned X / Y
    dimension is replaced with the HW-matching value.  See doc/TTNN_SHIM_ARCHITECTURE.md §17.
    """
    hw = getattr(tensor, 'hw_shape', None)
    if hw is not None:
        w, z, y, x = promote_to_rank4(hw)
    else:
        raw = getattr(tensor, 'shape', None)
        if raw is None:
            raise ValueError('tensor has no shape')
        raw_list = coerce_shape_to_list(raw)
        w, z, y, x = promote_to_rank4(raw_list)
    y_pad = getattr(tensor, 'y_pad_logical', None)
    if y_pad is not None:
        y = int(y_pad)
    x_pad = getattr(tensor, 'x_pad_logical', None)
    if x_pad is not None:
        x = int(x_pad)
    return (w, z, y, x)


def _input0_wzyx_for_master_key(op: Any, tensor_0: Any) -> Tuple[int, int, int, int]:
    w, z, y, x = _shape_wzyx(tensor_0)
    if _op_code(op) == "reshape":
        return reshape_input0_wzyx(w, z, y, x)
    if _op_code(op) == "createqkvheads":
        return createqkvheads_input0_wzyx(w, z, y, x)
    return (w, z, y, x)


# layernorm: caller sets mf (capture HiFi2). The llama3-only ops below are added here too —
# they are absent from every existing (VGG/ViT) LUT, so this cannot break any current hit
# (capture shows them at HiFi4, which is exactly this set's default). matmul/conv2d/pool are
# deliberately NOT added: existing LUTs key them under N/A, so adding them would regress those
# hits until a coordinated LUT repopulation (see project_math_fidelity_set_too_narrow).
_MATH_FIDELITY_CALLER_CONTROLLED_OPS = frozenset({
    "layernorm",
    "rotaryembeddingllamafusedqk", "pagedfusedupdatecache", "topk", "sampling",
})


def _op_math_fidelity(op: Any) -> str:
    """Extract math fidelity from SimOp attrs.

    For ops in ``_MATH_FIDELITY_CALLER_CONTROLLED_OPS`` (e.g. layernorm),
    returns the explicit value from attrs when present, or ``HiFi4`` (the
    hardware default) when omitted. For all other ops, returns ``N/A``
    matching the mapper's normalization.
    """
    attrs = getattr(op, "attrs", None)
    if isinstance(attrs, dict):
        mf = attrs.get("math_fidelity")
        if mf is not None:
            return str(mf)
    if _op_code(op) in _MATH_FIDELITY_CALLER_CONTROLLED_OPS:
        return "HiFi4"
    return MATH_FIDELITY_NA


def build_master_key_tuple_8(op: Any, tensor_0: Any) -> Tuple[Any, ...]:
    """Logical 9-tuple (first input + math fidelity); order matches ``KEY_TUPLE_YAML_KEYS[:9]``."""
    w0, z0, y0, x0 = _input0_wzyx_for_master_key(op, tensor_0)
    return (
        _op_code(op),
        w0,
        z0,
        y0,
        x0,
        tensor_layout_str(tensor_0),
        tensor_datatype(tensor_0, getattr(op, "precision", None)),
        tensor_memory_str(tensor_0),
        _op_math_fidelity(op),
    )


def build_master_key_tuple_15(
    op: Any,
    tensor_0: Any,
    tensor_1: Any,
) -> Tuple[Any, ...]:
    """Logical 16-tuple matching ``tools.perf_lookup.tt_perf_master_schema.KEY_TUPLE_YAML_KEYS`` order."""
    w0, z0, y0, x0 = _input0_wzyx_for_master_key(op, tensor_0)
    w1, z1, y1, x1 = _shape_wzyx(tensor_1)
    return (
        _op_code(op),
        w0,
        z0,
        y0,
        x0,
        tensor_layout_str(tensor_0),
        tensor_datatype(tensor_0, getattr(op, "precision", None)),
        tensor_memory_str(tensor_0),
        _op_math_fidelity(op),
        w1,
        z1,
        y1,
        x1,
        tensor_layout_str(tensor_1),
        tensor_datatype(tensor_1, getattr(op, "precision", None)),
        tensor_memory_str(tensor_1),
    )


def build_master_key_tuple_22(
    op: Any,
    tensor_0: Any,
    tensor_1: Any,
    tensor_2: Any,
) -> Tuple[Any, ...]:
    """Logical 23-tuple matching ``tools.perf_lookup.tt_perf_master_schema.KEY_TUPLE_YAML_KEYS`` order."""
    w0, z0, y0, x0 = _input0_wzyx_for_master_key(op, tensor_0)
    w1, z1, y1, x1 = _shape_wzyx(tensor_1)
    w2, z2, y2, x2 = _shape_wzyx(tensor_2)
    prec = getattr(op, "precision", None)
    return (
        _op_code(op),
        w0,
        z0,
        y0,
        x0,
        tensor_layout_str(tensor_0),
        tensor_datatype(tensor_0, prec),
        tensor_memory_str(tensor_0),
        _op_math_fidelity(op),
        w1,
        z1,
        y1,
        x1,
        tensor_layout_str(tensor_1),
        tensor_datatype(tensor_1, prec),
        tensor_memory_str(tensor_1),
        w2,
        z2,
        y2,
        x2,
        tensor_layout_str(tensor_2),
        tensor_datatype(tensor_2, prec),
        tensor_memory_str(tensor_2),
    )


def build_master_key_tuple_n(op: Any, tensors: Sequence[Any]) -> Tuple[Any, ...]:
    """Full-arity logical key: op_code + input_0 (7) + math_fidelity + input_i (7) for each i>=1.

    Generalizes build_master_key_tuple_8/15/22 to any arity (matches them exactly for 1/2/3
    inputs). Used for arity>=4 ops (e.g. PagedFusedUpdateCache, ScaledDotProductAttention) so the
    lookup keys on every operand — mirroring the mapper (tt_perf_mapper.build_key_tuple), which now
    keys all populated INPUT slots. Order matches KEY_TUPLE_YAML_KEYS. Requires >=1 tensor.
    """
    prec = getattr(op, "precision", None)
    t0 = tensors[0]
    w0, z0, y0, x0 = _input0_wzyx_for_master_key(op, t0)
    key: list = [
        _op_code(op), w0, z0, y0, x0,
        tensor_layout_str(t0), tensor_datatype(t0, prec), tensor_memory_str(t0),
        _op_math_fidelity(op),
    ]
    for t in tensors[1:]:
        w, z, y, x = _shape_wzyx(t)
        key += [w, z, y, x, tensor_layout_str(t), tensor_datatype(t, prec), tensor_memory_str(t)]
    return tuple(key)


def build_master_key_tuple_halo(op: Any, tensor_0: Any) -> Optional[Tuple[Any, ...]]:
    """Logical 16-tuple for halo: standard 9 + kernel_h/w, stride_h/w, padding_h/w, is_transpose.

    Reads ``op.attrs['kernel_size']``, ``op.attrs['stride']``, ``op.attrs['padding']``,
    ``op.attrs['is_transpose']`` — set by the ``_with_halo`` shim wrapper (conv_transpose2d
    sets is_transpose=True; conv2d/pool2d set False). Returns ``None`` when geometry attrs
    are missing; callers treat that as a lookup miss.
    """
    base = build_master_key_tuple_8(op, tensor_0)
    attrs = getattr(op, "attrs", None)
    if not isinstance(attrs, dict):
        return None
    ks = attrs.get("kernel_size")
    st = attrs.get("stride")
    pd = attrs.get("padding")
    if ks is None or st is None or pd is None:
        return None
    try:
        kH, kW = int(ks[0]), int(ks[1])
        sH, sW = int(st[0]), int(st[1])
        pH, pW = int(pd[0]), int(pd[1])
    except (TypeError, IndexError, ValueError):
        return None
    is_transpose = bool(attrs.get("is_transpose", False))
    return base + (kH, kW, sH, sW, pH, pW, is_transpose)


def build_master_key_tuple_its(
    op: Any,
    tensor_0: Any,
    output_tensor: Any,
) -> Tuple[Any, ...]:
    """Logical 10-tuple for InterleavedToSharded: standard 9 + output_0_memory."""
    base = build_master_key_tuple_8(op, tensor_0)
    return base + (tensor_memory_str(output_tensor),)


def build_master_key_tuple_15_add_broadcast_duplicate_full(
    op: Any,
    tensor_0: Any,
    tensor_1: Any,
) -> Optional[Tuple[Any, ...]]:
    """
    16-tuple key with the broadcast operand's WZYX replaced by the full operand's WZYX.

    Used when the master row stored both inputs with the full logical shape while the graph has
    ``(1, 1, 1, X)`` on one operand (either ``tensor_0`` or ``tensor_1``). Layout, datatype, and
    memory for each input stay tied to that input's tensor.

    Returns ``None`` unless ``op`` is ``add``, one operand is exactly ``(1,1,1,X)``, the other
    has the same ``X`` and at least one of W, Z, Y greater than 1, and the pattern is unambiguous.
    """
    if _op_code(op) != "add":
        return None
    w0, z0, y0, x0 = _input0_wzyx_for_master_key(op, tensor_0)
    w1, z1, y1, x1 = _shape_wzyx(tensor_1)
    b0 = w0 == 1 and z0 == 1 and y0 == 1
    b1 = w1 == 1 and z1 == 1 and y1 == 1
    full0 = not b0
    full1 = not b1
    prec = getattr(op, "precision", None)
    mf = _op_math_fidelity(op)
    lay0, dt0, mem0 = (
        tensor_layout_str(tensor_0),
        tensor_datatype(tensor_0, prec),
        tensor_memory_str(tensor_0),
    )
    lay1, dt1, mem1 = (
        tensor_layout_str(tensor_1),
        tensor_datatype(tensor_1, prec),
        tensor_memory_str(tensor_1),
    )
    if b1 and full0 and x1 == x0:
        wf, zf, yf, xf = w0, z0, y0, x0
        return (
            _op_code(op),
            wf,
            zf,
            yf,
            xf,
            lay0,
            dt0,
            mem0,
            mf,
            wf,
            zf,
            yf,
            xf,
            lay1,
            dt1,
            mem1,
        )
    if b0 and full1 and x0 == x1:
        wf, zf, yf, xf = w1, z1, y1, x1
        return (
            _op_code(op),
            wf,
            zf,
            yf,
            xf,
            lay0,
            dt0,
            mem0,
            mf,
            wf,
            zf,
            yf,
            xf,
            lay1,
            dt1,
            mem1,
        )
    return None


def _wzyx_int_tuple(t4: tuple) -> tuple:
    """Four logical dims as ints (YAML / numpy-safe)."""
    out: list[int] = []
    for x in t4:
        if hasattr(x, "item") and callable(getattr(x, "item", None)):
            out.append(int(x))
        else:
            out.append(int(x))
    return tuple(out)


def _lut_keys_matching_op_and_wzyx(entries: Dict[tuple, dict], key_t: tuple) -> Tuple[tuple, ...]:
    """
    LUT keys with same ``op_code`` and WZYX as ``key_t``
    (input 0 for 9/10/15-tuple; inputs 0 and 1 for 16-tuple; inputs 0, 1, and 2 for 23-tuple).

    Layout, datatype, and memory may differ — useful diagnostics when the full key misses.
    """
    n = len(key_t)
    if n not in (STANDARD_KEY_TUPLE_LENGTHS | {10, 15}):
        return ()
    oc = key_t[0]
    matched: list[tuple] = []
    if n in (9, 10, 15) or (n == 16 and oc == "halo"):
        # Unary ops (9/10-tuple), Halo v3 (15-tuple), and Halo v4 (16-tuple).
        # Halo v4's positions 9:13 carry kernel/stride geometry fields (not a
        # second input's WZYX), so we match only on input_0 WZYX here.
        w0 = _wzyx_int_tuple(key_t[1:5])
        for k in entries:
            if len(k) != n or k[0] != oc:
                continue
            if _wzyx_int_tuple(k[1:5]) == w0:
                matched.append(k)
    elif n == 16:
        # Non-halo 16-tuple key (binary op) — positions 9:13 are input_1 WZYX.
        w0 = _wzyx_int_tuple(key_t[1:5])
        w1 = _wzyx_int_tuple(key_t[9:13])
        for k in entries:
            if len(k) != 16 or k[0] != oc:
                continue
            if _wzyx_int_tuple(k[1:5]) == w0 and _wzyx_int_tuple(k[9:13]) == w1:
                matched.append(k)
    else:
        # 23-tuple and longer full-arity keys (30/37/44/51/58): match op + first-3 inputs' WZYX
        # and same arity (tuple length). input_0/1/2 WZYX live at 1:5 / 9:13 / 16:20 for all.
        w0 = _wzyx_int_tuple(key_t[1:5])
        w1 = _wzyx_int_tuple(key_t[9:13])
        w2 = _wzyx_int_tuple(key_t[16:20])
        for k in entries:
            if len(k) != n or k[0] != oc:
                continue
            if (
                _wzyx_int_tuple(k[1:5]) == w0
                and _wzyx_int_tuple(k[9:13]) == w1
                and _wzyx_int_tuple(k[16:20]) == w2
            ):
                matched.append(k)
    return tuple(sorted(matched))


def _lut_keys_matching_op_code_only(entries: Dict[tuple, dict], key_t: tuple) -> Tuple[tuple, ...]:
    """
    All LUT keys with the same ``op_code`` and tuple length as ``key_t`` (unary vs binary vs halo/ITS).

    Listed on full-key miss when no row matches operator type and all key attributes; contrasts
    with :func:`_lut_keys_matching_op_and_wzyx` which also requires WZYX match.
    """
    n = len(key_t)
    if n not in (STANDARD_KEY_TUPLE_LENGTHS | {10, 15}):
        return ()
    oc = key_t[0]
    matched = [k for k in entries if len(k) == n and k[0] == oc]
    return tuple(sorted(matched))


def _warn_labeled_lut_key_candidates(
    group_label: str,
    keys: Tuple[tuple, ...],
    *,
    reference_key: Optional[tuple] = None,
) -> None:
    """Log each LUT key tuple for miss diagnostics.

    When ``reference_key`` is set (the constructed lookup key), each candidate is compared
    field-by-field and **only differing** attributes are logged as ``lookup=`` vs ``lut_row=``.
    Otherwise the full labeled mapping is logged for each candidate.
    """
    n = len(keys)
    if n == 0:
        return
    ref_labeled: Optional[Dict[str, Any]] = None
    if reference_key is not None:
        try:
            ref_labeled = tuple_to_labeled_key_map(reference_key)
        except ValueError:
            ref_labeled = None

    for i, k in enumerate(keys):
        logger.warning("LUT {} candidate {}/{}:", group_label, i + 1, n)
        try:
            labeled = tuple_to_labeled_key_map(k)
        except ValueError:
            logger.warning("  raw={!r}", k)
            continue
        if ref_labeled is None:
            for name, val in labeled.items():
                logger.warning("  {}={}", name, val)
            continue
        diffs: list[tuple[str, Any, Any]] = []
        for name, lut_val in labeled.items():
            wanted = ref_labeled.get(name)
            if wanted != lut_val:
                diffs.append((name, wanted, lut_val))
        if diffs:
            logger.warning(
                "  vs lookup: {} attribute(s) differ — {}",
                len(diffs),
                ", ".join(nm for nm, _, _ in diffs),
            )
            for name, wanted, lut_val in diffs:
                logger.warning("    {}: lookup={!r} lut_row={!r}", name, wanted, lut_val)
        else:
            logger.warning(
                "  vs lookup: no differing labeled fields (tuple mismatch may be subtle); "
                "full row: {}",
                labeled,
            )


class OperatorPerfMap:
    """
    Master-format operator performance table: lookup by constructed 9-, 16-, or 23-tuple key
    and core count.
    """

    def __init__(
        self,
        yaml_file: Union[str, Path],
        *,
        use_hybrid_curve: bool = False,
    ):
        yaml_path = Path(yaml_file)
        if not yaml_path.exists():
            raise FileNotFoundError(f"YAML file not found: {yaml_file}")

        self._entries: Dict[tuple, dict] = load_existing_yaml(yaml_path)
        self._source_path = yaml_path
        self._use_hybrid_curve = bool(use_hybrid_curve)
        self._device_norm_index: Optional[Dict[tuple, list]] = None
        self.coverage_counts: Dict[str, int] = master_coverage_counts(self._entries)
        n_time_only = self.coverage_counts[MASTER_STATS_COVERAGE_TIME_ONLY]
        if n_time_only:
            logger.warning(
                "LUT {} has {} time-only entr{} of {}: hits on them report memory traffic and "
                "DRAM utilization as not measured (empty), not 0.",
                yaml_path,
                n_time_only,
                "y" if n_time_only == 1 else "ies",
                len(self._entries),
            )

    def __len__(self) -> int:
        return len(self._entries)

    def _device_normalized_index(self) -> Dict[tuple, list]:
        """Lazily build (and cache) an index of entries keyed by their
        device-normalized key, mapping to the list of ``(original_key, entry)``
        that collapse to it. A list longer than one means genuine multi-device
        ambiguity, which the device-normalized fallback declines to resolve."""
        if self._device_norm_index is None:
            idx: Dict[tuple, list] = {}
            for k, v in self._entries.items():
                idx.setdefault(_device_normalized_key(k), []).append((k, v))
            self._device_norm_index = idx
        return self._device_norm_index

    def _stats_from_entry(
        self,
        key_t: tuple,
        entry_val: dict,
        core_count: int,
        key_literal: Optional[tuple] = None,
        hit_source: Optional[str] = None,
    ) -> Optional[MasterPerfStats]:
        et = entry_val.get(MASTER_ENTRY_TYPE_KEY)
        if not isinstance(et, str):
            return None
        resolve = _build_stat_resolver(
            entry_val,
            et,
            core_count,
            key_t,
            self._source_path,
            use_hybrid_curve=self._use_hybrid_curve,
        )
        if resolve is None:
            return None

        msecs = resolve(MASTER_DURATION_MS_KEY)
        if msecs is None:
            logger.warning(
                "Perf lookup hit but no msecs for key={} core_count={} (path={})",
                key_t,
                core_count,
                self._source_path,
            )
            return None

        lut_path = self._source_path
        matrix_pipe_util = _validate_required_util_percent(
            "matrix_pipe_util", resolve("matrix_pipe_util"), lut_path, key_t
        )
        vector_pipe_util = _validate_required_util_percent(
            "vector_pipe_util", resolve("vector_pipe_util"), lut_path, key_t
        )

        mem_util = _validate_optional_util_percent(
            "mem_util", resolve("mem_util"), lut_path, key_t
        )
        for opt in LUT_OPTIONAL_UTIL_PERCENT_KEYS:
            if opt == "mem_util":
                continue
            v = resolve(opt)
            if v is not None:
                _validate_optional_util_percent(opt, v, lut_path, key_t)

        return MasterPerfStats(
            msecs=float(msecs),
            matrix_pipe_util=matrix_pipe_util,
            vector_pipe_util=vector_pipe_util,
            memory_traffic=resolve("memory_traffic"),
            mem_util=mem_util,
            key_literal=key_literal if key_literal is not None else key_t,
            key_resolved=key_t,
            hit_source=hit_source,
            stats_coverage=_resolved_payload_coverage(entry_val, et, self._use_hybrid_curve),
        )

    def _build_literal_key_with_tensors(
        self,
        op: Any,
        wlgraph: Any,
    ) -> Optional[Tuple[tuple, Tuple[Any, ...]]]:
        """Build the literal LUT key tuple and capture the input tensors used.

        Returns ``(key_t, (t0,) | (t0, t1) | (t0, t1, t2))`` on success, ``None`` on
        any of: zero arity, missing tensor, missing shape, key-build failure.
        Internal helper; ``lookup`` uses this then proceeds with entry matching, while
        ``build_literal_key`` exposes just the key half publicly.

        Ops with more than 3 inputs (e.g. PagedFusedUpdateCache, ScaledDotProductAttention)
        are keyed on ALL their inputs (full-arity key, up to MAX_KEY_INPUTS) — mirroring the
        mapper (tt_perf_mapper.build_key_tuple), which keys every populated INPUT slot. The sim's
        inList order matches the profiler input order, so the operand slots correspond.
        """
        in_list = getattr(op, "inList", [])
        n_in = len(in_list)
        tensors = getattr(wlgraph, "_tensors", {})

        if n_in == 0:
            logger.debug(
                "Perf lookup skipped (arity 0 has no master key): op={} optype={}",
                getattr(op, "name", "?"),
                getattr(op, "optype", "?"),
            )
            return None

        if n_in == 1:
            t0_name = in_list[0]
            if t0_name not in tensors:
                return None
            t0 = tensors[t0_name]
            if t0.shape is None:
                return None
            try:
                op_code_norm = _op_code(op)
                if op_code_norm == "halo":
                    halo_key = build_master_key_tuple_halo(op, t0)
                    if halo_key is None:
                        logger.debug(
                            "Perf lookup: halo op {} missing kernel/stride/padding "
                            "attrs; cannot form 16-tuple key (treating as miss)",
                            getattr(op, "name", "?"),
                        )
                        return None
                    return halo_key, (t0,)
                if op_code_norm == "interleavedtosharded":
                    out_list = getattr(op, "outList", [])
                    t_out = tensors.get(out_list[0]) if out_list else None
                    if t_out is None or getattr(t_out, "shape", None) is None:
                        logger.debug(
                            "Perf lookup: ITS op {} has no resolvable output tensor; "
                            "cannot form 10-tuple key (treating as miss)",
                            getattr(op, "name", "?"),
                        )
                        return None
                    return build_master_key_tuple_its(op, t0, t_out), (t0,)
                return build_master_key_tuple_8(op, t0), (t0,)
            except Exception as e:
                logger.debug("Perf lookup key build failed for op {}: {}", getattr(op, "name", "?"), e)
                return None
        if n_in == 2:
            t0_name, t1_name = in_list[0], in_list[1]
            if t0_name not in tensors or t1_name not in tensors:
                return None
            t0, t1 = tensors[t0_name], tensors[t1_name]
            if t0.shape is None or t1.shape is None:
                return None
            try:
                return build_master_key_tuple_15(op, t0, t1), (t0, t1)
            except Exception as e:
                logger.debug("Perf lookup key build failed for op {}: {}", getattr(op, "name", "?"), e)
                return None
        # n_in >= 3: key on ALL inputs (full arity). Arity 3 uses the dedicated 23-tuple builder
        # (byte-identical to the historical key); arity>=4 uses the generic full-arity builder,
        # mirroring the mapper which keys every populated INPUT slot.
        ten: list = []
        for nm in in_list:
            if nm not in tensors:
                return None
            t = tensors[nm]
            if getattr(t, "shape", None) is None:
                return None
            ten.append(t)
        try:
            if n_in == 3:
                return build_master_key_tuple_22(op, ten[0], ten[1], ten[2]), (ten[0], ten[1], ten[2])
            return build_master_key_tuple_n(op, ten), tuple(ten)
        except Exception as e:
            logger.debug("Perf lookup key build failed for op {}: {}", getattr(op, "name", "?"), e)
            return None

    def build_literal_key(self, op: Any, wlgraph: Any) -> Optional[tuple]:
        """Public: return the literal LUT key tuple for ``op`` (pre-fallback), or ``None``.

        Same as the key ``lookup`` builds internally — exposed so callers (e.g. device.py's
        per-op stats writer) can emit ``lut_key`` even when the entry lookup misses.
        """
        built = self._build_literal_key_with_tensors(op, wlgraph)
        return built[0] if built is not None else None

    def lookup(
        self,
        op: Any,
        wlgraph: Any,
        core_count: int,
    ) -> Optional[MasterPerfStats]:
        """
        Return resolved stats when the table has a matching 9-, 16-, or 23-tuple key, else ``None``.

        ``core_count`` should match profiler Core_Count bucketing (see package config).
        """
        built = self._build_literal_key_with_tensors(op, wlgraph)
        if built is None:
            return None
        key_t, ts = built
        n_in = len(ts)
        t0 = ts[0]
        t1 = ts[1] if n_in >= 2 else None
        t2 = ts[2] if n_in >= 3 else None

        entry_val = self._entries.get(key_t)
        lookup_key = key_t
        hit_source: Optional[str] = "direct" if entry_val is not None else None

        if entry_val is None and n_in == 2 and _op_code(op) == "add":
            try:
                key_add_bc = build_master_key_tuple_15_add_broadcast_duplicate_full(op, t0, t1)
            except Exception as e:
                logger.debug(
                    "Perf lookup add broadcast key build failed for op {}: {}",
                    getattr(op, "name", "?"),
                    e,
                )
                key_add_bc = None
            if key_add_bc is not None:
                ev_add = self._entries.get(key_add_bc)
                if ev_add is not None:
                    entry_val = ev_add
                    lookup_key = key_add_bc
                    hit_source = "add_broadcast_dup"
                    logger.debug(
                        "Perf lookup: 15-tuple miss for add op={!r}; using broadcast-duplicated "
                        "LUT key {} (lut={})",
                        getattr(op, "name", None),
                        key_add_bc,
                        self._source_path,
                    )

        # Move: arity-1 in Polaris, arity-2 in hardware profiler (src + dst same tensor).
        # After 9-tuple miss, try 16-tuple with t0 duplicated for both positions.
        if entry_val is None and n_in == 1 and _op_code(op) in _UNARY_POLARIS_BINARY_HW_OPCODES:
            try:
                key16 = build_master_key_tuple_15(op, t0, t0)
            except Exception as e:
                logger.debug('Perf lookup move dup-key build failed for op {}: {}', getattr(op, 'name', '?'), e)
                key16 = None
            if key16 is not None:
                ev16 = self._entries.get(key16)
                if ev16 is not None:
                    entry_val = ev16
                    lookup_key = key16
                    hit_source = "move_arity_dup"
                elif len(key16) >= 16 and key16[5] == _ROW_MAJOR_STR:
                    # Also try TILE layout for both input_0 (pos 5) and input_1 (pos 13).
                    key16_tile = key16[:5] + (_TILE_STR,) + key16[6:13] + (_TILE_STR,) + key16[14:]
                    ev16_tile = self._entries.get(key16_tile)
                    if ev16_tile is not None:
                        entry_val = ev16_tile
                        lookup_key = key16_tile
                        hit_source = "move_arity_dup_tile"
                # HEIGHT→BLOCK fallback for 16-tuple dup keys: substitute BLOCK_SHARDED at
                # both input_0_memory (pos 7) and input_1_memory (pos 15) simultaneously.
                if entry_val is None and len(key16) >= 16 and key16[7] == _HEIGHT_SHARDED_STR:
                    key16_block = key16[:7] + (_BLOCK_SHARDED_STR,) + key16[8:15] + (_BLOCK_SHARDED_STR,)
                    ev16_block = self._entries.get(key16_block)
                    if ev16_block is not None:
                        entry_val = ev16_block
                        lookup_key = key16_block
                        hit_source = "move_arity_dup_block"

        # InterleavedToSharded: hardware stages through DRAM in most VGG UNet decoder paths,
        # but Polaris models the predecessor STI output as L1_INTERLEAVED.  After L1 miss,
        # substitute DRAM_INTERLEAVED in position 7 of the 9-tuple and retry.
        if (
            entry_val is None
            and n_in == 1
            and _op_code(op) in _ITS_MEM_FALLBACK_OPCODES
            and len(lookup_key) >= 9
            and lookup_key[7] == _L1_INTERLEAVED_STR
        ):
            key_dram = lookup_key[:7] + (_DRAM_INTERLEAVED_STR,) + lookup_key[8:]
            ev_dram = self._entries.get(key_dram)
            if ev_dram is not None:
                entry_val = ev_dram
                lookup_key = key_dram
                hit_source = "its_l1_to_dram"

        # Halo / Sigmoid: LUT was built with TILE layout; Polaris models VGG UNet as ROW_MAJOR.
        # After ROW_MAJOR arity-1 miss, substitute TILE in position 5 of the 9-tuple.
        if (
            entry_val is None
            and n_in == 1
            and _op_code(op) in _LAYOUT_ROWMAJOR_TO_TILE_FALLBACK_OPCODES
            and len(lookup_key) >= 9
            and lookup_key[5] == _ROW_MAJOR_STR
        ):
            key_tile = lookup_key[:5] + (_TILE_STR,) + lookup_key[6:]
            ev_tile = self._entries.get(key_tile)
            if ev_tile is not None:
                entry_val = ev_tile
                lookup_key = key_tile
                hit_source = "halo_rowmajor_to_tile"

        # VGG UNet decoder HEIGHT→BLOCK fallback: after HEIGHT_SHARDED miss, substitute
        # DEV_1_L1_BLOCK_SHARDED at position 7 (input_0_memory) of the lookup key.
        # Works for arity-1 (9-tuple: halo, move) and arity-3 (23-tuple: conv2d, convtranspose)
        # since position 7 is input_0_memory in both formats.
        if (
            entry_val is None
            and n_in in (1, 3)
            and _op_code(op) in _HALO_HEIGHT_TO_BLOCK_FALLBACK_OPCODES
            and len(lookup_key) >= 9
            and lookup_key[7] == _HEIGHT_SHARDED_STR
        ):
            key_block = lookup_key[:7] + (_BLOCK_SHARDED_STR,) + lookup_key[8:]
            ev_block = self._entries.get(key_block)
            if ev_block is not None:
                entry_val = ev_block
                lookup_key = key_block
                hit_source = "halo_height_to_block"
            elif lookup_key[5] == _ROW_MAJOR_STR:
                # Also try TILE layout + BLOCK memory combined
                key_tile_block = lookup_key[:5] + (_TILE_STR,) + lookup_key[6:7] + (_BLOCK_SHARDED_STR,) + lookup_key[8:]
                ev_tb = self._entries.get(key_tile_block)
                if ev_tb is not None:
                    entry_val = ev_tb
                    lookup_key = key_tile_block
                    hit_source = "halo_tile_block_combined"

        if (
            entry_val is None
            and n_in == 2
            and _op_code(op) in _BINARY_LUT_FALLBACK_TO_INPUT0_KEY_OPCODES
        ):
            try:
                key8 = build_master_key_tuple_8(op, t0)
            except Exception as e:
                logger.debug(
                    "Perf lookup unary fallback key build failed for op {}: {}",
                    getattr(op, "name", "?"),
                    e,
                )
                key8 = None
            if key8 is not None:
                ev8 = self._entries.get(key8)
                if ev8 is not None:
                    entry_val = ev8
                    lookup_key = key8
                    hit_source = "binary_to_unary"
                    logger.debug(
                        "Perf lookup: 15-tuple miss for {} op={!r}; using unary LUT key {} (lut={})",
                        _op_code(op),
                        getattr(op, "name", None),
                        key8,
                        self._source_path,
                    )

        # Device-index reconciliation (last fallback): the sim hardcodes a "DEV_1_"
        # memory prefix while a capture-built LUT carries the real DEVICE ID (e.g.
        # "DEV_0_"). After all exact/substitution attempts miss, retry ignoring the
        # device prefix — but only when exactly one LUT entry agrees on every
        # non-device field. >1 candidate = genuine multi-device ambiguity → decline.
        if entry_val is None:
            cands = self._device_normalized_index().get(_device_normalized_key(key_t))
            if cands is not None and len(cands) == 1:
                orig_key, ev = cands[0]
                entry_val = ev
                lookup_key = orig_key
                hit_source = "device_normalized"
            elif cands is not None and len(cands) > 1:
                logger.debug(
                    "Perf lookup: device-normalized fallback declined for op={!r} — "
                    "{} multi-device candidates differ only in device index (lut={})",
                    getattr(op, "name", None),
                    len(cands),
                    self._source_path,
                )

        if entry_val is None:
            same_op_shape_keys = _lut_keys_matching_op_and_wzyx(self._entries, key_t)
            same_op_only_keys = _lut_keys_matching_op_code_only(self._entries, key_t)
            logger.warning(
                "Perf lookup miss (no matching LUT row) opname={!r} optype={!r} arity={} "
                "key={} core_count={} lut={} n_lut_keys_same_op_and_shapes={} "
                "n_lut_keys_same_op_code_only={}",
                getattr(op, "name", None),
                getattr(op, "optype", None),
                n_in,
                key_t,
                core_count,
                self._source_path,
                len(same_op_shape_keys),
                len(same_op_only_keys),
            )
            _warn_labeled_lut_key_candidates(
                "same-op+shape", same_op_shape_keys, reference_key=key_t
            )
            _warn_labeled_lut_key_candidates(
                "same-op-code-only", same_op_only_keys, reference_key=key_t
            )
            logger.warning("------------------------------------------")
            return None

        logger.debug(
            "Perf lookup hit opname={!r} optype={!r} key={} core_count={} lut={} entry={}",
            getattr(op, "name", None),
            getattr(op, "optype", None),
            lookup_key,
            core_count,
            self._source_path,
            entry_val,
        )
        return self._stats_from_entry(lookup_key, entry_val, core_count, key_literal=key_t, hit_source=hit_source)


def resolve_operator_lookup_core_count(simcfg_obj: Any, package_model: Any) -> int:
    """
    Core count for master single/curve evaluation.

    Order: ``operator_lookup_core_count`` on package if set; else compute IPGroup
    ``num_units``; else ``_DEFAULT_CORE_COUNT_FALLBACK``.
    """
    raw = getattr(package_model, "operator_lookup_core_count", None)
    if raw is not None:
        try:
            n = int(raw)
            if n > 0:
                return n
        except (TypeError, ValueError):
            pass

    try:
        compute_group = package_model.get_ipgroup(iptype="compute")
        nu = int(compute_group.num_units)
        if nu > 0:
            return nu
    except Exception:
        pass

    logger.info(
        "operator_lookup_core_count unset and compute num_units missing; using fallback {}",
        _DEFAULT_CORE_COUNT_FALLBACK,
    )
    return _DEFAULT_CORE_COUNT_FALLBACK
