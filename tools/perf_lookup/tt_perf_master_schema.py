# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Wire-format constants and logical key ↔ labeled YAML mapping for tt-perf master files.

Normative field names: ``doc/YAML_MASTER_FORMAT.md``. Excel pipeline and CLI: ``tools/perf_lookup/tt_perf_mapper.py``.
On-disk ``schema_version`` stays **1** until the first release; bump only then for incompatible layout changes.
``entry_type`` may be ``single``, ``curve``, or ``hybrid`` (matmul). Used by ``tt_perf_master_loader`` and
``tt_perf_mapper`` (serialize, merge, Excel→stats keys).

Shared normalization and scalar checks live here (stdlib only) so loader and mapper stay aligned.
"""

from __future__ import annotations

import math
import numbers

# Labeled record key under ``entries[i]['key']``. Order aligns with Excel ``KEY_TUPLE_COLUMN_NAMES``.
#
# Layout: op_code, input_0 (7 fields), math_fidelity, then input_1..input_{MAX-1} (7 fields each).
# The key covers ALL populated inputs (full-arity fidelity) — profiler rows expose INPUT_0..INPUT_7,
# so ops with >3 operands (e.g. PagedFusedUpdateCache, ScaledDotProductAttention) are keyed on every
# input rather than a first-3 truncation. Arity 1/2/3 keys are byte-identical to the historical
# 9/16/23-tuples (input_0/1/2 order unchanged), so <=3-input ops (VGG/ViT/most llama3) are unaffected.
# Per-arity tuple lengths: 9,16,23,30,37,44,51,58 for 1..8 inputs.
MAX_KEY_INPUTS = 8  # profiler emits INPUT_0..INPUT_7
_INPUT_SLOT_SUFFIXES = ("w_pad_logical", "z_pad_logical", "y_pad_logical", "x_pad_logical",
                        "layout", "datatype", "memory")


def _input_slot_keys(i: int) -> list:
    return [f"input_{i}_{s}" for s in _INPUT_SLOT_SUFFIXES]


KEY_TUPLE_YAML_KEYS = (
    ["op_code"] + _input_slot_keys(0) + ["math_fidelity"]
    + [k for i in range(1, MAX_KEY_INPUTS) for k in _input_slot_keys(i)]
)

_KEY_TUPLE_YAML_KEYS_SET = frozenset(KEY_TUPLE_YAML_KEYS)

# Valid standard-op key tuple lengths: 9 (1 input) + 7 per extra input -> {9,16,23,30,37,44,51,58}
# for arity 1..MAX_KEY_INPUTS. (Per-op variants halo/ITS use their own lengths, handled separately.)
STANDARD_KEY_TUPLE_LENGTHS = frozenset(9 + 7 * k for k in range(MAX_KEY_INPUTS))

# Per-op variant key tuples.
# Halo v3: standard 9 fields + 6 sliding-window geometry fields = 15 fields.
# Halo v4: adds is_transpose to disambiguate ConvTranspose halos = 16 fields.
_HALO_V3_EXTRA_YAML_KEYS = ["kernel_h", "kernel_w", "stride_h", "stride_w", "padding_h", "padding_w"]
HALO_V3_KEY_TUPLE_YAML_KEYS = KEY_TUPLE_YAML_KEYS[:9] + _HALO_V3_EXTRA_YAML_KEYS
HALO_KEY_TUPLE_YAML_KEYS = KEY_TUPLE_YAML_KEYS[:9] + _HALO_V3_EXTRA_YAML_KEYS + ["is_transpose"]

# InterleavedToSharded: standard 9 fields + destination memory = 10 fields.
ITS_KEY_TUPLE_YAML_KEYS = KEY_TUPLE_YAML_KEYS[:9] + [
    "output_0_memory",
]

# Field names that uniquely identify each per-op variant (used for detection on load).
# Any halo key (v3 or v4) has kernel_h; is_transpose distinguishes v4 from v3.
_HALO_EXTRA_KEY_NAMES: frozenset[str] = frozenset(HALO_KEY_TUPLE_YAML_KEYS[9:])
_ITS_EXTRA_KEY_NAMES: frozenset[str] = frozenset(ITS_KEY_TUPLE_YAML_KEYS[9:])

# Device kernel duration in master YAML (milliseconds). Excel uses nanoseconds column title.
MASTER_DURATION_MS_KEY = "msecs"

MASTER_ENTRY_TYPE_KEY = "entry_type"
MASTER_ENTRY_TYPE_SINGLE = "single"
MASTER_ENTRY_TYPE_CURVE = "curve"
MASTER_ENTRY_TYPE_HYBRID = "hybrid"

# Flat ``entry_type: single`` payload: ``num_cores`` plus the stat keys below (same under ``hybrid.single``).
MASTER_SINGLE_NUM_CORES_KEY = "num_cores"
MASTER_SINGLE_STAT_KEYS = (
    MASTER_DURATION_MS_KEY,
    "memory_traffic",
    "mem_util",
    "noc_util",
    "noc_multicast_util",
    "npe_cong_impact_pct",
    "vector_pipe_util",
    "matrix_pipe_util",
)

MASTER_SINGLE_STAT_KEYS_SET = frozenset(MASTER_SINGLE_STAT_KEYS)

# Stats only the NOC-trace profiling pass measures. A ``time_only`` payload omits them: absent
# means "not measured", never 0.
MASTER_NOCTRACE_STAT_KEYS = (
    "memory_traffic",
    "mem_util",
    "noc_util",
    "noc_multicast_util",
    "npe_cong_impact_pct",
)
MASTER_TIME_ONLY_STAT_KEYS = tuple(k for k in MASTER_SINGLE_STAT_KEYS if k not in MASTER_NOCTRACE_STAT_KEYS)

# Per-payload stats coverage (flat single, ``hybrid.single``, curve, ``hybrid.curve``). A payload
# without the key is ``full``, so pre-v6 files stay valid.
MASTER_STATS_COVERAGE_KEY = "stats_coverage"
MASTER_STATS_COVERAGE_FULL = "full"
MASTER_STATS_COVERAGE_TIME_ONLY = "time_only"
MASTER_STATS_COVERAGE_VALUES = (MASTER_STATS_COVERAGE_FULL, MASTER_STATS_COVERAGE_TIME_ONLY)

# Flat single payload keys that are utilization-like percentages / scores (not duration or traffic).
MASTER_SINGLE_UTIL_STAT_KEYS = frozenset(
    (
        "mem_util",
        "noc_util",
        "noc_multicast_util",
        "npe_cong_impact_pct",
        "vector_pipe_util",
        "matrix_pipe_util",
    )
)

# Each per-stat curve fit object (under ``curve`` / ``hybrid.curve``) uses these keys.
MASTER_CURVE_STAT_ENTRY_KEYS = frozenset(("a", "b", "r2", "equation"))

# Matmul hybrid: ``single`` = flat ``num_cores`` + stats; ``curve`` = ``curve_family`` + per-stat fits.
MASTER_HYBRID_SINGLE_KEY = "single"
MASTER_HYBRID_CURVE_KEY = "curve"

MASTER_CURVE_FAMILY_KEY = "curve_family"
MASTER_CURVE_FAMILY_LINEAR = "linear"
MASTER_CURVE_FAMILY_POWER = "power"

MATH_FIDELITY_NA = "N/A"

MASTER_YAML_SCHEMA_NAME = "correqn.tt-perf-master"
MASTER_YAML_SCHEMA_NAME_KEY = "schema_name"
# v2: math_fidelity added to key tuple (v1 files still load via default-fill in labeled_key_map_to_tuple).
# v3: per-op variant tuples — halo (15-field) and interleavedtosharded (10-field).
# v4: halo gains is_transpose (16-field) so ConvTranspose halos don't collide with Conv halos.
# v5: full-arity keys — standard ops keyed on ALL populated inputs (input_3..input_7), so >3-input
#     ops (PagedFusedUpdateCache, ScaledDotProductAttention) key on every operand instead of a
#     first-3 truncation. Adds standard-op tuple lengths 30/37/44/51/58; arity<=3 keys unchanged.
# v6: time-only entries — a payload may carry ``stats_coverage: time_only`` and omit the NOC-trace
#     stats; the root carries a ``coverage`` block (entry count per coverage).
MASTER_YAML_SCHEMA_VERSION = 6
MASTER_YAML_SCHEMA_VERSION_KEY = "schema_version"
MASTER_YAML_ENTRIES_KEY = "entries"
MASTER_YAML_COVERAGE_KEY = "coverage"
MASTER_YAML_RECORD_KEY_FIELD = "key"
MASTER_YAML_ENTRY_VALUE_FIELD = "value"


def tuple_to_labeled_key_map(key_t: tuple) -> dict:
    """Serialize logical key tuple to YAML mapping (under ``entries[i]['key']``)."""
    n = len(key_t)
    # Per-op variants are disambiguated by op_code (key_t[0]) — schema v4's halo
    # is 16-field which collides on length with the standard binary-op tuple.
    if n >= 1 and key_t[0] == "halo":
        if n == len(HALO_KEY_TUPLE_YAML_KEYS):
            return dict(zip(HALO_KEY_TUPLE_YAML_KEYS, key_t))
        if n == len(HALO_V3_KEY_TUPLE_YAML_KEYS):
            return dict(zip(HALO_V3_KEY_TUPLE_YAML_KEYS, key_t))
        raise ValueError(
            f"Halo key tuple length must be {len(HALO_V3_KEY_TUPLE_YAML_KEYS)} (v3) "
            f"or {len(HALO_KEY_TUPLE_YAML_KEYS)} (v4), got {n}"
        )
    if n >= 1 and key_t[0] == "interleavedtosharded":
        if n != len(ITS_KEY_TUPLE_YAML_KEYS):
            raise ValueError(
                f"ITS key tuple length must be {len(ITS_KEY_TUPLE_YAML_KEYS)}, got {n}"
            )
        return dict(zip(ITS_KEY_TUPLE_YAML_KEYS, key_t))
    if n not in STANDARD_KEY_TUPLE_LENGTHS:
        raise ValueError(
            f"Key tuple length must be one of {sorted(STANDARD_KEY_TUPLE_LENGTHS)} for standard ops "
            f"(or use halo/its variant), got {n}")
    return dict(zip(KEY_TUPLE_YAML_KEYS[:n], key_t))


def labeled_key_map_to_tuple(d: dict) -> tuple:
    """Parse YAML labeled mapping back to logical key tuple."""
    if not isinstance(d, dict):
        raise TypeError(f"Labeled key must be a dict, got {type(d)}")
    str_keys = {k if isinstance(k, str) else str(k) for k in d}

    # Per-op variant detection: unique extra fields identify halo and ITS.
    if str_keys & _HALO_EXTRA_KEY_NAMES:
        # Detect schema version by presence of is_transpose (v4) or absence (v3 legacy).
        if "is_transpose" in str_keys:
            halo_keys = HALO_KEY_TUPLE_YAML_KEYS  # 16-field v4
        else:
            halo_keys = HALO_V3_KEY_TUPLE_YAML_KEYS  # 15-field v3 legacy
        expected = frozenset(halo_keys)
        missing = [nm for nm in halo_keys if nm not in d]
        if missing:
            raise ValueError(f"Halo labeled key missing fields: {missing}")
        unknown = sorted(str_keys - expected)
        if unknown:
            raise ValueError(f"Unknown halo labeled-key field(s): {unknown}")
        return tuple(d[nm] for nm in halo_keys)

    if str_keys & _ITS_EXTRA_KEY_NAMES:
        expected = frozenset(ITS_KEY_TUPLE_YAML_KEYS)
        missing = [nm for nm in ITS_KEY_TUPLE_YAML_KEYS if nm not in d]
        if missing:
            raise ValueError(f"ITS labeled key missing fields: {missing}")
        unknown = sorted(str_keys - expected)
        if unknown:
            raise ValueError(f"Unknown ITS labeled-key field(s): {unknown}")
        return tuple(d[nm] for nm in ITS_KEY_TUPLE_YAML_KEYS)

    # Standard 9/16/23 path.
    filtered: dict = {}
    unknown_list: list = []
    for k, v in d.items():
        ks = k if isinstance(k, str) else str(k)
        if ks in _KEY_TUPLE_YAML_KEYS_SET:
            filtered[ks] = v
        else:
            unknown_list.append(k)
    if unknown_list:
        raise ValueError(f"Unknown labeled-key field(s): {unknown_list}")
    if "math_fidelity" not in filtered:
        filtered["math_fidelity"] = MATH_FIDELITY_NA
    # Full-arity standard path: the key covers input_0..input_k for the highest populated slot k
    # (input slots must be contiguous — no gaps). Tuple length = 9 + 7*k (9/16/23/30/37/44/51/58).
    present_slots = sorted(
        i for i in range(MAX_KEY_INPUTS)
        if any(str(k).startswith(f"input_{i}_") for k in filtered)
    )
    if not present_slots or present_slots[0] != 0:
        raise ValueError("Labeled key must define input_0 fields")
    max_slot = present_slots[-1]
    if present_slots != list(range(max_slot + 1)):
        missing_slots = [i for i in range(max_slot + 1) if i not in present_slots]
        raise ValueError(
            f"Labeled-key input slots must be contiguous; input_{max_slot} present but "
            f"input slot(s) {missing_slots} missing"
        )
    n = 9 + 7 * max_slot
    names = KEY_TUPLE_YAML_KEYS[:n]
    missing = [nm for nm in names if nm not in filtered]
    if missing:
        raise ValueError(
            f"Labeled key (arity {max_slot + 1}) must define all {n} fields; missing: {missing}"
        )
    return tuple(filtered[nm] for nm in names)


def yaml_labeled_key_to_tuple(k: dict) -> tuple:
    """Parse ``entries[i]['key']`` labeled mapping to logical key tuple."""
    if not isinstance(k, dict):
        raise TypeError(f"Entry key must be a mapping, got {type(k)!r}")
    key_t = labeled_key_map_to_tuple(k)
    n = len(key_t)
    # standard-op lengths (9/16/23/30/37/44/51/58) plus per-op variants: ITS=10, halo v3=15.
    # halo v4 is 16 (v3 + is_transpose) and is accepted because 16 is ALSO the 2-input standard
    # length, not because of an explicit entry here — revisit if the standard lengths ever change.
    # test_halo_v4_key_length_accepted pins that coincidence.
    if n not in (STANDARD_KEY_TUPLE_LENGTHS | {10, 15}):
        raise ValueError(
            f"Key tuple length must be one of {sorted(STANDARD_KEY_TUPLE_LENGTHS | {10, 15})}, got {n}")
    return key_t


def normalize_num_cores_scalar(nc):
    """Whole-number ``float`` (e.g. ``64.0``) → ``int``; pass through ``int`` and other types."""
    if type(nc) is bool:
        return nc
    if isinstance(nc, float) and nc.is_integer():
        return int(nc)
    return nc


def normalize_flat_single_payload(d: dict) -> dict:
    """Copy flat single / ``hybrid.single`` payload and normalize ``num_cores`` after YAML load."""
    if not isinstance(d, dict):
        raise TypeError(f"flat single payload must be a dict, got {type(d)!r}")
    out = dict(d)
    if MASTER_SINGLE_NUM_CORES_KEY in out:
        out[MASTER_SINGLE_NUM_CORES_KEY] = normalize_num_cores_scalar(
            out[MASTER_SINGLE_NUM_CORES_KEY]
        )
    for uk in MASTER_SINGLE_UTIL_STAT_KEYS:
        if uk not in out:
            continue
        v = out[uk]
        if type(v) is bool:
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(fv):
            out[uk] = 0.0
    return out


def payload_stats_coverage(payload: dict) -> str:
    """Coverage of one flat-single or curve payload; a payload without the key is ``full``."""
    return payload.get(MASTER_STATS_COVERAGE_KEY, MASTER_STATS_COVERAGE_FULL)


def entry_stats_coverage(val: dict) -> str:
    """Coverage of one entry ``value``: ``time_only`` if any of its payloads is time-only."""
    if val.get(MASTER_ENTRY_TYPE_KEY) == MASTER_ENTRY_TYPE_HYBRID:
        payloads = [p for p in (val.get(MASTER_HYBRID_SINGLE_KEY), val.get(MASTER_HYBRID_CURVE_KEY))
                    if isinstance(p, dict)]
    else:
        payloads = [val]
    if any(payload_stats_coverage(p) == MASTER_STATS_COVERAGE_TIME_ONLY for p in payloads):
        return MASTER_STATS_COVERAGE_TIME_ONLY
    return MASTER_STATS_COVERAGE_FULL


def master_coverage_counts(master: dict) -> dict:
    """Root ``coverage`` block: entry count per coverage value, in a fixed order."""
    counts = {c: 0 for c in MASTER_STATS_COVERAGE_VALUES}
    for val in master.values():
        counts[entry_stats_coverage(val)] += 1
    return counts


def is_real_stat_scalar(x) -> bool:
    """True for finite numeric stats: ``int``/``float``/``Integral``/``Real``, excluding ``bool``.

    Accepts NumPy scalar types (``numpy.floating``, etc.) when present without importing NumPy.
    """
    if type(x) is bool:
        return False
    if isinstance(x, numbers.Integral):
        return True
    if isinstance(x, numbers.Real):
        try:
            fx = float(x)
        except (TypeError, ValueError, OverflowError):
            return False
        return math.isfinite(fx)
    if hasattr(x, "dtype"):
        raw_kind = getattr(x.dtype, "kind", None)
        if isinstance(raw_kind, str) and raw_kind in ("i", "u", "f"):
            try:
                fx = float(x)
            except (TypeError, ValueError, OverflowError):
                return False
            return math.isfinite(fx)
    return False
