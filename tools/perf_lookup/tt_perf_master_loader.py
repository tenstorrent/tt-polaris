# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Load tt-perf master YAML into ``dict[tuple, dict]`` (logical key → entry payload).

Contract: ``doc/YAML_MASTER_FORMAT.md``. CLI and Excel pipeline: ``tools/perf_lookup/tt_perf_mapper.py``. Accepts
``schema_version`` 1 up to ``MASTER_YAML_SCHEMA_VERSION`` (older versions emit a deprecation warning); matmul entries
must use ``entry_type: hybrid``. A payload may be ``stats_coverage: time_only``; see ``tt_perf_master_schema``.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import yaml

try:
    from .tt_perf_master_schema import (
        MASTER_CURVE_FAMILY_KEY,
        MASTER_CURVE_FAMILY_LINEAR,
        MASTER_CURVE_FAMILY_POWER,
        MASTER_CURVE_STAT_ENTRY_KEYS,
        MASTER_DURATION_MS_KEY,
        MASTER_ENTRY_TYPE_CURVE,
        MASTER_ENTRY_TYPE_HYBRID,
        MASTER_ENTRY_TYPE_KEY,
        MASTER_ENTRY_TYPE_SINGLE,
        MASTER_HYBRID_CURVE_KEY,
        MASTER_HYBRID_SINGLE_KEY,
        MASTER_SINGLE_NUM_CORES_KEY,
        MASTER_SINGLE_STAT_KEYS,
        MASTER_NOCTRACE_STAT_KEYS,
        MASTER_STATS_COVERAGE_KEY,
        MASTER_STATS_COVERAGE_TIME_ONLY,
        MASTER_STATS_COVERAGE_VALUES,
        MASTER_TIME_ONLY_STAT_KEYS,
        MASTER_YAML_COVERAGE_KEY,
        master_coverage_counts,
        payload_stats_coverage,
        is_real_stat_scalar,
        normalize_flat_single_payload,
        MASTER_YAML_ENTRIES_KEY,
        MASTER_YAML_ENTRY_VALUE_FIELD,
        MASTER_YAML_RECORD_KEY_FIELD,
        MASTER_YAML_SCHEMA_NAME,
        MASTER_YAML_SCHEMA_NAME_KEY,
        MASTER_YAML_SCHEMA_VERSION,
        MASTER_YAML_SCHEMA_VERSION_KEY,
        yaml_labeled_key_to_tuple,
    )
except ImportError:
    from tt_perf_master_schema import (  # type: ignore[import-not-found,no-redef]
    MASTER_CURVE_FAMILY_KEY,
    MASTER_CURVE_FAMILY_LINEAR,
    MASTER_CURVE_FAMILY_POWER,
    MASTER_CURVE_STAT_ENTRY_KEYS,
    MASTER_DURATION_MS_KEY,
    MASTER_ENTRY_TYPE_CURVE,
    MASTER_ENTRY_TYPE_HYBRID,
    MASTER_ENTRY_TYPE_KEY,
    MASTER_ENTRY_TYPE_SINGLE,
    MASTER_HYBRID_CURVE_KEY,
    MASTER_HYBRID_SINGLE_KEY,
    MASTER_SINGLE_NUM_CORES_KEY,
    MASTER_SINGLE_STAT_KEYS,
    MASTER_NOCTRACE_STAT_KEYS,
    MASTER_STATS_COVERAGE_KEY,
    MASTER_STATS_COVERAGE_TIME_ONLY,
    MASTER_STATS_COVERAGE_VALUES,
    MASTER_TIME_ONLY_STAT_KEYS,
    MASTER_YAML_COVERAGE_KEY,
    master_coverage_counts,
    payload_stats_coverage,
    is_real_stat_scalar,
    normalize_flat_single_payload,
    MASTER_YAML_ENTRIES_KEY,
    MASTER_YAML_ENTRY_VALUE_FIELD,
    MASTER_YAML_RECORD_KEY_FIELD,
    MASTER_YAML_SCHEMA_NAME,
    MASTER_YAML_SCHEMA_NAME_KEY,
    MASTER_YAML_SCHEMA_VERSION,
    MASTER_YAML_SCHEMA_VERSION_KEY,
        yaml_labeled_key_to_tuple,
    )

# TT-Polaris matmul layer type (first key field); must match tt_perf_mapper.POLARIS_LAYER_MATMUL.
_POLARIS_LAYER_MATMUL = "matmul"


def _normalize_loaded_master_value(val: dict) -> dict:
    """After YAML load: normalize ``num_cores`` whole-number floats to ``int``."""
    t = val.get(MASTER_ENTRY_TYPE_KEY)
    if t == MASTER_ENTRY_TYPE_CURVE:
        return dict(val)
    if t == MASTER_ENTRY_TYPE_HYBRID:
        out = dict(val)
        sk = out.get(MASTER_HYBRID_SINGLE_KEY)
        if isinstance(sk, dict):
            out[MASTER_HYBRID_SINGLE_KEY] = normalize_flat_single_payload(sk)
        return out
    if t == MASTER_ENTRY_TYPE_SINGLE:
        return normalize_flat_single_payload(dict(val))
    return dict(val)


def _validate_stats_dict(d: dict, pair_index: int, ctx: str) -> None:
    if MASTER_DURATION_MS_KEY not in d:
        raise ValueError(
            f"entries[{pair_index}]['value']{ctx} stats missing {MASTER_DURATION_MS_KEY!r}"
        )


def _validate_num_cores(nc, pair_index: int, ctx: str) -> None:
    if type(nc) is bool:
        raise ValueError(
            f"entries[{pair_index}]['value']{ctx} {MASTER_SINGLE_NUM_CORES_KEY!r} invalid (bool)"
        )
    if isinstance(nc, int):  # type: ignore[unreachable]
        return
    if isinstance(nc, float):
        if not nc.is_integer():
            raise ValueError(
                f"entries[{pair_index}]['value']{ctx} {MASTER_SINGLE_NUM_CORES_KEY!r} "
                f"must be a whole number, got {nc!r}"
            )
        return
    raise ValueError(
        f"entries[{pair_index}]['value']{ctx} {MASTER_SINGLE_NUM_CORES_KEY!r} "
        f"must be numeric, got {type(nc)!r}"
    )


def _validate_stats_coverage(d: dict, pair_index: int, ctx: str) -> str:
    """Return the payload's stats coverage; reject values outside MASTER_STATS_COVERAGE_VALUES."""
    cov = payload_stats_coverage(d)
    if cov not in MASTER_STATS_COVERAGE_VALUES:
        raise ValueError(
            f"entries[{pair_index}]['value']{ctx} {MASTER_STATS_COVERAGE_KEY!r} must be one of "
            f"{list(MASTER_STATS_COVERAGE_VALUES)!r}, got {cov!r}"
        )
    return cov


def _validate_flat_single_payload(
    d: dict, pair_index: int, ctx: str
) -> None:
    """``num_cores`` plus the stat keys of the payload's coverage; extra keys → warning only.

    A ``full`` payload carries all ``MASTER_SINGLE_STAT_KEYS``. A ``time_only`` payload carries
    ``MASTER_TIME_ONLY_STAT_KEYS`` and none of ``MASTER_NOCTRACE_STAT_KEYS``.
    """
    if MASTER_SINGLE_NUM_CORES_KEY not in d:
        raise ValueError(
            f"entries[{pair_index}]['value']{ctx} missing {MASTER_SINGLE_NUM_CORES_KEY!r}"
        )
    _validate_num_cores(d[MASTER_SINGLE_NUM_CORES_KEY], pair_index, ctx)
    cov = _validate_stats_coverage(d, pair_index, ctx)
    time_only = cov == MASTER_STATS_COVERAGE_TIME_ONLY
    stat_keys = MASTER_TIME_ONLY_STAT_KEYS if time_only else MASTER_SINGLE_STAT_KEYS
    required = frozenset((MASTER_SINGLE_NUM_CORES_KEY, *stat_keys))
    got = frozenset(d)
    missing = sorted(required - got)
    if missing:
        raise ValueError(
            f"entries[{pair_index}]['value']{ctx} {cov} single payload missing keys: {missing!r}"
        )
    if time_only:
        present = sorted(got & frozenset(MASTER_NOCTRACE_STAT_KEYS))
        if present:
            raise ValueError(
                f"entries[{pair_index}]['value']{ctx} time_only single payload must not carry "
                f"NOC-trace stats: {present!r}"
            )
    extra = sorted(got - required - {MASTER_STATS_COVERAGE_KEY})
    for ek in extra:
        warnings.warn(
            f"entries[{pair_index}]['value']{ctx} has unknown key {ek!r} (ignored by tools; "
            f"prefer only {sorted(required)!r})",
            stacklevel=2,
        )
    for sk in stat_keys:
        v = d[sk]
        if not is_real_stat_scalar(v):
            raise ValueError(
                f"entries[{pair_index}]['value']{ctx}[{sk!r}] must be a finite real number"
            )
    _validate_stats_dict(d, pair_index, ctx)


def _validate_curve_stat_entry(
    stat_key: str,
    sub: object,
    pair_index: int,
    ctx: str,
) -> None:
    if not isinstance(sub, dict):
        raise ValueError(
            f"entries[{pair_index}]['value']{ctx}[{stat_key!r}] must be a mapping, "
            f"got {type(sub)!r}"
        )
    got_keys = frozenset(sub)
    missing = sorted(MASTER_CURVE_STAT_ENTRY_KEYS - got_keys)
    if missing:
        raise ValueError(
            f"entries[{pair_index}]['value']{ctx}[{stat_key!r}] missing keys {missing!r}"
        )
    extra = sorted(got_keys - MASTER_CURVE_STAT_ENTRY_KEYS)
    for ek in extra:
        warnings.warn(
            f"entries[{pair_index}]['value']{ctx}[{stat_key!r}] has unknown key {ek!r}",
            stacklevel=2,
        )
    for nk in ("a", "b", "r2"):
        if not is_real_stat_scalar(sub[nk]):
            raise ValueError(
                f"entries[{pair_index}]['value']{ctx}[{stat_key!r}][{nk!r}] "
                f"must be a finite real number"
            )
    eq = sub["equation"]
    if not isinstance(eq, str):
        raise ValueError(
            f"entries[{pair_index}]['value']{ctx}[{stat_key!r}]['equation'] must be str"
        )


def _validate_curve_payload(curve: dict, pair_index: int, ctx: str) -> None:
    fam = curve.get(MASTER_CURVE_FAMILY_KEY)
    if fam not in (MASTER_CURVE_FAMILY_LINEAR, MASTER_CURVE_FAMILY_POWER):
        raise ValueError(
            f"entries[{pair_index}]['value']{ctx} requires {MASTER_CURVE_FAMILY_KEY!r} "
            f"{MASTER_CURVE_FAMILY_LINEAR!r} or {MASTER_CURVE_FAMILY_POWER!r}, got {fam!r}"
        )
    if _validate_stats_coverage(curve, pair_index, ctx) == MASTER_STATS_COVERAGE_TIME_ONLY:
        present = sorted(set(curve) & set(MASTER_NOCTRACE_STAT_KEYS))
        if present:
            raise ValueError(
                f"entries[{pair_index}]['value']{ctx} time_only curve payload must not carry "
                f"NOC-trace stat fits: {present!r}"
            )
    for k, v in curve.items():
        if k in (MASTER_ENTRY_TYPE_KEY, MASTER_CURVE_FAMILY_KEY, MASTER_STATS_COVERAGE_KEY):
            continue
        _validate_curve_stat_entry(str(k), v, pair_index, f"{ctx} / stat {k!r}")


def _validate_loaded_entry_value(val: dict, pair_index: int) -> None:
    """Ensure ``value`` has ``entry_type`` and curve/hybrid entries are well-formed."""
    t = val.get(MASTER_ENTRY_TYPE_KEY)
    if t not in (
        MASTER_ENTRY_TYPE_SINGLE,
        MASTER_ENTRY_TYPE_CURVE,
        MASTER_ENTRY_TYPE_HYBRID,
    ):
        raise ValueError(
            f"entries[{pair_index}]['value'] requires {MASTER_ENTRY_TYPE_KEY!r} "
            f"{MASTER_ENTRY_TYPE_SINGLE!r}, {MASTER_ENTRY_TYPE_CURVE!r}, or "
            f"{MASTER_ENTRY_TYPE_HYBRID!r}, got {t!r}"
        )
    if t == MASTER_ENTRY_TYPE_CURVE:
        _validate_curve_payload(val, pair_index, " (curve)")
        return
    if t == MASTER_ENTRY_TYPE_HYBRID:
        sk = val.get(MASTER_HYBRID_SINGLE_KEY)
        ck = val.get(MASTER_HYBRID_CURVE_KEY)
        if sk is None and ck is None:
            raise ValueError(
                f"entries[{pair_index}]['value'] (hybrid) must include "
                f"{MASTER_HYBRID_SINGLE_KEY!r} and/or {MASTER_HYBRID_CURVE_KEY!r}"
            )
        if sk is not None:
            if not isinstance(sk, dict):
                raise ValueError(
                    f"entries[{pair_index}]['value'].single must be a mapping, got {type(sk)!r}"
                )
            _validate_flat_single_payload(sk, pair_index, " (hybrid.single)")
        if ck is not None:
            if not isinstance(ck, dict):
                raise ValueError(
                    f"entries[{pair_index}]['value'].curve must be a mapping, got {type(ck)!r}"
                )
            _validate_curve_payload(ck, pair_index, " (hybrid.curve)")
        return
    # single: flat num_cores + stat keys
    inner = {k: v for k, v in val.items() if k != MASTER_ENTRY_TYPE_KEY}
    _validate_flat_single_payload(inner, pair_index, " (single)")


def _validate_matmul_requires_hybrid(
    val: dict, key_t: tuple, pair_index: int
) -> None:
    if not key_t or str(key_t[0]) != _POLARIS_LAYER_MATMUL:
        return
    t = val.get(MASTER_ENTRY_TYPE_KEY)
    if t != MASTER_ENTRY_TYPE_HYBRID:
        raise ValueError(
            f"entries[{pair_index}]: op_code matmul requires "
            f"{MASTER_ENTRY_TYPE_KEY!r} {MASTER_ENTRY_TYPE_HYBRID!r}, got {t!r}"
        )


def _entries_list_to_master(entries: list) -> dict[tuple, dict]:
    """Parse ``entries`` list of ``{key: labeled_map, value: entry_dict}`` into dict[tuple, dict]."""
    out: dict[tuple, dict] = {}
    for i, item in enumerate(entries):
        if not isinstance(item, dict):
            raise ValueError(
                f"entries[{i}] must be a mapping with {MASTER_YAML_RECORD_KEY_FIELD!r} and "
                f"{MASTER_YAML_ENTRY_VALUE_FIELD!r}, got {type(item)!r}"
            )
        extra = set(item.keys()) - {
            MASTER_YAML_RECORD_KEY_FIELD,
            MASTER_YAML_ENTRY_VALUE_FIELD,
        }
        if extra:
            raise ValueError(
                f"entries[{i}] has unknown keys {sorted(extra)!r}; "
                f"only {MASTER_YAML_RECORD_KEY_FIELD!r} and {MASTER_YAML_ENTRY_VALUE_FIELD!r} allowed"
            )
        if MASTER_YAML_RECORD_KEY_FIELD not in item or MASTER_YAML_ENTRY_VALUE_FIELD not in item:
            raise ValueError(
                f"entries[{i}] must contain {MASTER_YAML_RECORD_KEY_FIELD!r} and "
                f"{MASTER_YAML_ENTRY_VALUE_FIELD!r}"
            )
        k = item[MASTER_YAML_RECORD_KEY_FIELD]
        v = item[MASTER_YAML_ENTRY_VALUE_FIELD]
        key_t = yaml_labeled_key_to_tuple(k)
        if not isinstance(v, dict):
            raise ValueError(
                f"entries[{i}]['value'] must be a mapping, got {type(v)!r}"
            )
        norm = _normalize_loaded_master_value(v)
        _validate_loaded_entry_value(norm, i)
        _validate_matmul_requires_hybrid(norm, key_t, i)
        out[key_t] = norm
    return out


def load_existing_yaml(path: Path) -> dict[tuple, dict]:
    """Load YAML into dict[tuple, dict].

    Expects the versioned document: ``schema_name``, ``schema_version``, ``entries`` as a list of
    mappings ``{key: labeled_key_map, value: entry_dict}``.
    """
    with open(path) as f:
        raw = yaml.safe_load(f)
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"YAML root must be a mapping, got {type(raw)}")
    if MASTER_YAML_ENTRIES_KEY not in raw:
        raise ValueError(
            f"YAML root must contain {MASTER_YAML_ENTRIES_KEY!r} "
            f"(versioned master document: schema_name, schema_version, entries)."
        )
    if MASTER_YAML_SCHEMA_VERSION_KEY not in raw:
        raise ValueError(
            f"YAML has {MASTER_YAML_ENTRIES_KEY!r} but missing "
            f"{MASTER_YAML_SCHEMA_VERSION_KEY!r}"
        )
    ver = raw[MASTER_YAML_SCHEMA_VERSION_KEY]
    try:
        ver_int = int(ver)
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"{MASTER_YAML_SCHEMA_VERSION_KEY!r} must be an integer, got {ver!r}"
        ) from e
    _MASTER_YAML_SCHEMA_MIN_VERSION = 1
    if ver_int < _MASTER_YAML_SCHEMA_MIN_VERSION or ver_int > MASTER_YAML_SCHEMA_VERSION:
        raise ValueError(
            f"{MASTER_YAML_SCHEMA_VERSION_KEY!r} is {ver_int}, expected "
            f"{_MASTER_YAML_SCHEMA_MIN_VERSION}–{MASTER_YAML_SCHEMA_VERSION}"
        )
    if ver_int < MASTER_YAML_SCHEMA_VERSION:
        warnings.warn(
            f"Loading legacy schema_version={ver_int} LUT file; "
            f"re-export to version {MASTER_YAML_SCHEMA_VERSION} to suppress this warning.",
            DeprecationWarning,
            stacklevel=2,
        )
    sid = raw.get(MASTER_YAML_SCHEMA_NAME_KEY)
    if sid != MASTER_YAML_SCHEMA_NAME:
        raise ValueError(
            f"{MASTER_YAML_SCHEMA_NAME_KEY!r} must be {MASTER_YAML_SCHEMA_NAME!r}, "
            f"got {sid!r}"
        )
    ent = raw[MASTER_YAML_ENTRIES_KEY]
    if not isinstance(ent, list):
        raise ValueError(
            f"{MASTER_YAML_ENTRIES_KEY!r} must be a list, got {type(ent)}"
        )
    master = _entries_list_to_master(ent)
    # The root ``coverage`` block is derived from the entries. When present it must agree with them,
    # so a hand-edited or partially rewritten file cannot misreport how many entries are time-only.
    if MASTER_YAML_COVERAGE_KEY in raw:
        declared = raw[MASTER_YAML_COVERAGE_KEY]
        actual = master_coverage_counts(master)
        if declared != actual:
            raise ValueError(
                f"{MASTER_YAML_COVERAGE_KEY!r} is {declared!r}, but the entries give {actual!r}"
            )
    return master
