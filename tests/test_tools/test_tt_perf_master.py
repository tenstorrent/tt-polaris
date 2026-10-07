# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Contract tests: schema helpers, loader, merge, and Excel→single stats alignment."""

from __future__ import annotations

import math
import warnings
from pathlib import Path

import numpy as np
import pytest
import yaml

from tools.perf_lookup.tt_perf_master_loader import load_existing_yaml
from tools.perf_lookup.tt_perf_master_schema import (
    KEY_TUPLE_YAML_KEYS,
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
    MASTER_SINGLE_STAT_KEYS,
    MASTER_NOCTRACE_STAT_KEYS,
    MASTER_STATS_COVERAGE_FULL,
    MASTER_STATS_COVERAGE_KEY,
    MASTER_STATS_COVERAGE_TIME_ONLY,
    MASTER_TIME_ONLY_STAT_KEYS,
    MASTER_YAML_COVERAGE_KEY,
    MASTER_YAML_ENTRIES_KEY,
    MASTER_YAML_ENTRY_VALUE_FIELD,
    MASTER_YAML_RECORD_KEY_FIELD,
    MASTER_YAML_SCHEMA_NAME,
    MASTER_YAML_SCHEMA_NAME_KEY,
    MASTER_YAML_SCHEMA_VERSION,
    MASTER_YAML_SCHEMA_VERSION_KEY,
    is_real_stat_scalar,
    labeled_key_map_to_tuple,
    normalize_flat_single_payload,
    tuple_to_labeled_key_map,
    entry_stats_coverage,
    master_coverage_counts,
    payload_stats_coverage,
    yaml_labeled_key_to_tuple,
)
from tools.profiling.op_canonical import normalize_profiler_opcode
from tools.perf_lookup.tt_perf_mapper import (
    EXCEL_COL_DRAM_BW_UTIL,
    EXCEL_COL_FPU,
    EXCEL_COL_MULTICAST_NOC,
    EXCEL_COL_NOC_UTIL,
    EXCEL_COL_NPE_CONG,
    EXCEL_COL_SFPU,
    EXCEL_DURATION_COLUMN,
    KEY_TUPLE_COLUMN_NAMES,
    InputTable,
    POLARIS_LAYER_MATMUL,
    STATS_COLUMN_OPTIONAL,
    STATS_COLUMNS_REQUIRED,
    build_key_tuple,
    build_master_dict,
    build_stats_row,
    canonicalize_master_for_write,
    format_dry_run_report,
    is_curve_entry,
    is_hybrid_entry,
    is_single_entry,
    merge_entry_for_key,
    merge_master,
    serialize_master_for_yaml,
    validate_columns,
)

_STAT_COLS_WITH_OPTIONAL = [*STATS_COLUMNS_REQUIRED, STATS_COLUMN_OPTIONAL]


def _single_stats_payload():
    return {k: float(i) for i, k in enumerate(MASTER_SINGLE_STAT_KEYS)}


def _versioned_doc(entries: list) -> dict:
    return {
        MASTER_YAML_SCHEMA_NAME_KEY: MASTER_YAML_SCHEMA_NAME,
        MASTER_YAML_SCHEMA_VERSION_KEY: MASTER_YAML_SCHEMA_VERSION,
        MASTER_YAML_ENTRIES_KEY: entries,
    }


_PAD_SUFFIXES = (
    "_W_PAD[LOGICAL]",
    "_Z_PAD[LOGICAL]",
    "_Y_PAD[LOGICAL]",
    "_X_PAD[LOGICAL]",
)


def _series_for_key_slots(*, input1: str, input2: str) -> dict:
    """``input1`` / ``input2``: ``'fill'`` (valid pads+metadata), ``'blank'``, or ``'mixed'`` (invalid)."""
    d: dict = {"OP CODE": "layernormalization"}
    for col in KEY_TUPLE_COLUMN_NAMES[1:]:
        if col == "MATH FIDELITY":
            d[col] = "HiFi4"
            continue
        if col.startswith("INPUT_0_"):
            slot = "0"
        elif col.startswith("INPUT_1_"):
            slot = "1"
        elif col.startswith("INPUT_2_"):
            slot = "2"
        else:
            # INPUT_3..7 (full-arity schema v5): this 3-slot helper leaves them blank
            d[col] = ""
            continue
        mode = input1 if slot != "2" else input2
        if slot == "0":
            mode = "fill"
        if mode == "fill":
            if any(col.endswith(s) for s in _PAD_SUFFIXES):
                d[col] = "8[8]"
            elif "LAYOUT" in col:
                d[col] = "TILE"
            elif "DATATYPE" in col:
                d[col] = "BFLOAT16"
            else:
                d[col] = "DEV_1_DRAM_INTERLEAVED"
        elif mode == "blank":
            d[col] = ""
        elif mode == "mixed":
            if col.endswith("_W_PAD[LOGICAL]"):
                d[col] = "1[1]"
            else:
                d[col] = ""
        else:
            raise ValueError(mode)
    return d


def _curve_stat(a=1.0, b=0.0, r2=0.9, equation="y = x"):
    return {"a": a, "b": b, "r2": r2, "equation": equation}


def _full_curve_value():
    return {
        MASTER_ENTRY_TYPE_KEY: MASTER_ENTRY_TYPE_CURVE,
        MASTER_CURVE_FAMILY_KEY: MASTER_CURVE_FAMILY_LINEAR,
        **{sk: _curve_stat(equation=f"{sk}=c") for sk in MASTER_SINGLE_STAT_KEYS},
    }


def test_normalize_flat_single_coerces_whole_float_num_cores():
    d = {MASTER_SINGLE_NUM_CORES_KEY: 64.0, **_single_stats_payload()}
    n = normalize_flat_single_payload(d)
    assert n[MASTER_SINGLE_NUM_CORES_KEY] == 64
    assert type(n[MASTER_SINGLE_NUM_CORES_KEY]) is int


def test_normalize_flat_single_coerces_nan_util_to_zero():
    base = _single_stats_payload()
    base["vector_pipe_util"] = float("nan")
    base["matrix_pipe_util"] = float("inf")
    n = normalize_flat_single_payload({MASTER_SINGLE_NUM_CORES_KEY: 1, **base})
    assert n["vector_pipe_util"] == 0.0
    assert n["matrix_pipe_util"] == 0.0
    assert math.isfinite(n["mem_util"])


def test_build_stats_row_coerces_nan_util_to_zero():
    row = {
        EXCEL_DURATION_COLUMN: 1_000_000.0,
        EXCEL_COL_DRAM_BW_UTIL: float("nan"),
        EXCEL_COL_NOC_UTIL: float("nan"),
        EXCEL_COL_MULTICAST_NOC: float("inf"),
        EXCEL_COL_NPE_CONG: 3.0,
        EXCEL_COL_SFPU: float("nan"),
        EXCEL_COL_FPU: 4.0,
    }
    out = build_stats_row(
        row, 100.0, _STAT_COLS_WITH_OPTIONAL, EXCEL_DURATION_COLUMN, "CORE COUNT"
    )
    assert out is not None
    assert out["mem_util"] == 0.0
    assert out["noc_util"] == 0.0
    assert out["noc_multicast_util"] == 0.0
    assert out["vector_pipe_util"] == 0.0
    assert out["matrix_pipe_util"] == 4.0


def test_is_real_stat_scalar_numpy_and_nan():
    assert is_real_stat_scalar(np.float64(1.0))
    assert is_real_stat_scalar(np.int32(3))
    assert not is_real_stat_scalar(True)
    assert not is_real_stat_scalar(float("nan"))
    assert not is_real_stat_scalar("1")


def test_build_stats_row_keys_match_schema_order():
    row = {
        EXCEL_DURATION_COLUMN: 1_000_000.0,
        EXCEL_COL_DRAM_BW_UTIL: 10.0,
        EXCEL_COL_NOC_UTIL: 1.0,
        EXCEL_COL_MULTICAST_NOC: 2.0,
        EXCEL_COL_NPE_CONG: 3.0,
        EXCEL_COL_SFPU: 4.0,
        EXCEL_COL_FPU: 5.0,
    }
    out = build_stats_row(
        row, 100.0, _STAT_COLS_WITH_OPTIONAL, EXCEL_DURATION_COLUMN, "CORE COUNT"
    )
    assert out is not None
    assert list(out.keys()) == list(MASTER_SINGLE_STAT_KEYS)
    assert out["noc_multicast_util"] == 2.0


def test_load_minimal_master_yaml(tmp_path: Path):
    key_map = {k: f"v_{k}" for k in KEY_TUPLE_YAML_KEYS[:9]}
    val = {
        MASTER_ENTRY_TYPE_KEY: MASTER_ENTRY_TYPE_SINGLE,
        MASTER_SINGLE_NUM_CORES_KEY: 1,
        **_single_stats_payload(),
    }
    doc = {
        MASTER_YAML_SCHEMA_NAME_KEY: "correqn.tt-perf-master",
        MASTER_YAML_SCHEMA_VERSION_KEY: 1,
        MASTER_YAML_ENTRIES_KEY: [
            {
                MASTER_YAML_RECORD_KEY_FIELD: key_map,
                MASTER_YAML_ENTRY_VALUE_FIELD: val,
            }
        ],
    }
    p = tmp_path / "m.yaml"
    p.write_text(yaml.safe_dump(doc, sort_keys=False))
    m = load_existing_yaml(p)
    kt = tuple(key_map[k] for k in KEY_TUPLE_YAML_KEYS[:9])
    assert kt in m
    assert m[kt][MASTER_SINGLE_NUM_CORES_KEY] == 1


def test_loader_warns_on_extra_single_key(tmp_path: Path):
    key_map = {k: f"v_{k}" for k in KEY_TUPLE_YAML_KEYS[:9]}
    val = {
        MASTER_ENTRY_TYPE_KEY: MASTER_ENTRY_TYPE_SINGLE,
        MASTER_SINGLE_NUM_CORES_KEY: 1,
        **_single_stats_payload(),
        "extra_key": 0,
    }
    doc = {
        MASTER_YAML_SCHEMA_NAME_KEY: "correqn.tt-perf-master",
        MASTER_YAML_SCHEMA_VERSION_KEY: 1,
        MASTER_YAML_ENTRIES_KEY: [
            {
                MASTER_YAML_RECORD_KEY_FIELD: key_map,
                MASTER_YAML_ENTRY_VALUE_FIELD: val,
            }
        ],
    }
    p = tmp_path / "m.yaml"
    p.write_text(yaml.safe_dump(doc, sort_keys=False))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        load_existing_yaml(p)
    assert any("extra_key" in str(x.message) for x in w)


def test_loader_rejects_curve_stat_missing_equation(tmp_path: Path):
    key_map = {k: f"v_{k}" for k in KEY_TUPLE_YAML_KEYS[:9]}
    bad_curve = {
        MASTER_ENTRY_TYPE_KEY: MASTER_ENTRY_TYPE_CURVE,
        MASTER_CURVE_FAMILY_KEY: MASTER_CURVE_FAMILY_LINEAR,
        MASTER_DURATION_MS_KEY: {"a": 1.0, "b": 0.0, "r2": 0.5},
    }
    doc = {
        MASTER_YAML_SCHEMA_NAME_KEY: "correqn.tt-perf-master",
        MASTER_YAML_SCHEMA_VERSION_KEY: 1,
        MASTER_YAML_ENTRIES_KEY: [
            {
                MASTER_YAML_RECORD_KEY_FIELD: key_map,
                MASTER_YAML_ENTRY_VALUE_FIELD: bad_curve,
            }
        ],
    }
    p = tmp_path / "m.yaml"
    p.write_text(yaml.safe_dump(doc, sort_keys=False))
    with pytest.raises(ValueError, match="missing keys"):
        load_existing_yaml(p)


def test_merge_matmul_single_and_curve_to_hybrid():
    kt = (POLARIS_LAYER_MATMUL,) + tuple(f"x{i}" for i in range(8))
    single = {
        MASTER_ENTRY_TYPE_KEY: MASTER_ENTRY_TYPE_SINGLE,
        MASTER_SINGLE_NUM_CORES_KEY: 2,
        **_single_stats_payload(),
    }
    curve = {
        MASTER_ENTRY_TYPE_KEY: MASTER_ENTRY_TYPE_CURVE,
        MASTER_CURVE_FAMILY_KEY: MASTER_CURVE_FAMILY_LINEAR,
        **{
            sk: {
                "a": 1.0,
                "b": 0.0,
                "r2": 0.99,
                "equation": f"{sk} = 1*c + 0",
            }
            for sk in MASTER_SINGLE_STAT_KEYS
        },
    }
    h = merge_entry_for_key(kt, single, curve)
    assert h[MASTER_ENTRY_TYPE_KEY] == MASTER_ENTRY_TYPE_HYBRID
    assert MASTER_HYBRID_SINGLE_KEY in h
    assert MASTER_HYBRID_CURVE_KEY in h
    assert h[MASTER_HYBRID_SINGLE_KEY][MASTER_SINGLE_NUM_CORES_KEY] == 2


def test_normalize_flat_single_payload_requires_dict():
    with pytest.raises(TypeError, match="dict"):
        normalize_flat_single_payload("not-a-dict")  # type: ignore[arg-type]


def test_tuple_to_labeled_key_map_round_trip_9_16_and_23():
    t9 = tuple(f"a{i}" for i in range(9))
    assert labeled_key_map_to_tuple(tuple_to_labeled_key_map(t9)) == t9
    t16 = tuple(f"b{i}" for i in range(16))
    assert labeled_key_map_to_tuple(tuple_to_labeled_key_map(t16)) == t16
    t23 = tuple(f"c{i}" for i in range(23))
    assert labeled_key_map_to_tuple(tuple_to_labeled_key_map(t23)) == t23


def test_tuple_to_labeled_key_map_rejects_bad_length():
    with pytest.raises(ValueError, match="length"):
        tuple_to_labeled_key_map(tuple("x" * 3))


def test_labeled_key_map_rejects_unknown_field():
    km = {k: f"v_{k}" for k in KEY_TUPLE_YAML_KEYS[:9]}
    km["not_a_valid_key"] = 1
    with pytest.raises(ValueError, match="Unknown"):
        labeled_key_map_to_tuple(km)


def test_labeled_key_map_input2_partial_requires_all_23():
    km = {k: f"v_{k}" for k in KEY_TUPLE_YAML_KEYS[:17]}
    km["input_2_w_pad_logical"] = 1
    with pytest.raises(ValueError, match="23"):
        labeled_key_map_to_tuple(km)


def test_build_key_tuple_9_16_and_23():
    row9 = _series_for_key_slots(input1="blank", input2="blank")
    kt9, err9 = build_key_tuple(row9, KEY_TUPLE_COLUMN_NAMES, _PAD_SUFFIXES)
    assert err9 is None and len(kt9) == 9

    row16 = _series_for_key_slots(input1="fill", input2="blank")
    kt16, err16 = build_key_tuple(row16, KEY_TUPLE_COLUMN_NAMES, _PAD_SUFFIXES)
    assert err16 is None and len(kt16) == 16

    row23 = _series_for_key_slots(input1="fill", input2="fill")
    kt23, err23 = build_key_tuple(row23, KEY_TUPLE_COLUMN_NAMES, _PAD_SUFFIXES)
    assert err23 is None and len(kt23) == 23


def test_build_key_tuple_rejects_input2_without_input1():
    row = _series_for_key_slots(input1="blank", input2="fill")
    kt, err = build_key_tuple(row, KEY_TUPLE_COLUMN_NAMES, _PAD_SUFFIXES)
    # INPUT_2 set with INPUT_1 blank = a gap; full-arity build rejects non-contiguous slots.
    assert kt is None and err is not None and "INPUT_2" in err and "contiguous" in err


def test_build_key_tuple_rejects_mixed_input2():
    row = _series_for_key_slots(input1="fill", input2="mixed")
    kt, err = build_key_tuple(row, KEY_TUPLE_COLUMN_NAMES, _PAD_SUFFIXES)
    assert kt is None and err is not None and "INPUT_2" in err


def test_load_minimal_master_yaml_22_field_key(tmp_path: Path):
    key_map = {k: f"v_{k}" for k in KEY_TUPLE_YAML_KEYS}
    val = {
        MASTER_ENTRY_TYPE_KEY: MASTER_ENTRY_TYPE_SINGLE,
        MASTER_SINGLE_NUM_CORES_KEY: 1,
        **_single_stats_payload(),
    }
    doc = {
        MASTER_YAML_SCHEMA_NAME_KEY: "correqn.tt-perf-master",
        MASTER_YAML_SCHEMA_VERSION_KEY: 1,
        MASTER_YAML_ENTRIES_KEY: [
            {
                MASTER_YAML_RECORD_KEY_FIELD: key_map,
                MASTER_YAML_ENTRY_VALUE_FIELD: val,
            }
        ],
    }
    p = tmp_path / "m22.yaml"
    p.write_text(yaml.safe_dump(doc, sort_keys=False))
    m = load_existing_yaml(p)
    kt = tuple(key_map[k] for k in KEY_TUPLE_YAML_KEYS)
    assert kt in m
    assert m[kt][MASTER_SINGLE_NUM_CORES_KEY] == 1


def test_yaml_labeled_key_to_tuple_requires_mapping():
    with pytest.raises(TypeError, match="mapping"):
        yaml_labeled_key_to_tuple([])  # type: ignore[arg-type]


def test_load_existing_yaml_null_document(tmp_path: Path):
    p = tmp_path / "empty.yaml"
    p.write_text("null\n")
    assert load_existing_yaml(p) == {}


def test_load_rejects_missing_entries(tmp_path: Path):
    p = tmp_path / "m.yaml"
    p.write_text(
        yaml.safe_dump(
            {
                MASTER_YAML_SCHEMA_NAME_KEY: MASTER_YAML_SCHEMA_NAME,
                MASTER_YAML_SCHEMA_VERSION_KEY: MASTER_YAML_SCHEMA_VERSION,
            },
            sort_keys=False,
        )
    )
    with pytest.raises(ValueError, match="entries"):
        load_existing_yaml(p)


def test_load_rejects_wrong_schema_version(tmp_path: Path):
    key_map = {k: f"v_{k}" for k in KEY_TUPLE_YAML_KEYS[:9]}
    doc = _versioned_doc(
        [
            {
                MASTER_YAML_RECORD_KEY_FIELD: key_map,
                MASTER_YAML_ENTRY_VALUE_FIELD: {
                    MASTER_ENTRY_TYPE_KEY: MASTER_ENTRY_TYPE_SINGLE,
                    MASTER_SINGLE_NUM_CORES_KEY: 1,
                    **_single_stats_payload(),
                },
            }
        ]
    )
    doc[MASTER_YAML_SCHEMA_VERSION_KEY] = 999
    p = tmp_path / "m.yaml"
    p.write_text(yaml.safe_dump(doc, sort_keys=False))
    with pytest.raises(ValueError, match="expected"):
        load_existing_yaml(p)


def test_load_rejects_bad_schema_name(tmp_path: Path):
    key_map = {k: f"v_{k}" for k in KEY_TUPLE_YAML_KEYS[:9]}
    doc = _versioned_doc(
        [
            {
                MASTER_YAML_RECORD_KEY_FIELD: key_map,
                MASTER_YAML_ENTRY_VALUE_FIELD: {
                    MASTER_ENTRY_TYPE_KEY: MASTER_ENTRY_TYPE_SINGLE,
                    MASTER_SINGLE_NUM_CORES_KEY: 1,
                    **_single_stats_payload(),
                },
            }
        ]
    )
    doc[MASTER_YAML_SCHEMA_NAME_KEY] = "wrong.name"
    p = tmp_path / "m.yaml"
    p.write_text(yaml.safe_dump(doc, sort_keys=False))
    with pytest.raises(ValueError, match="schema_name"):
        load_existing_yaml(p)


def test_load_rejects_matmul_non_hybrid(tmp_path: Path):
    key_map = {k: f"v_{k}" for k in KEY_TUPLE_YAML_KEYS[:9]}
    key_map["op_code"] = POLARIS_LAYER_MATMUL
    p = tmp_path / "m.yaml"
    p.write_text(
        yaml.safe_dump(
            _versioned_doc(
                [
                    {
                        MASTER_YAML_RECORD_KEY_FIELD: key_map,
                        MASTER_YAML_ENTRY_VALUE_FIELD: {
                            MASTER_ENTRY_TYPE_KEY: MASTER_ENTRY_TYPE_SINGLE,
                            MASTER_SINGLE_NUM_CORES_KEY: 1,
                            **_single_stats_payload(),
                        },
                    }
                ]
            ),
            sort_keys=False,
        )
    )
    with pytest.raises(ValueError, match="matmul"):
        load_existing_yaml(p)


def test_load_hybrid_matmul_single_and_curve(tmp_path: Path):
    key_map = {k: f"v_{k}" for k in KEY_TUPLE_YAML_KEYS[:9]}
    key_map["op_code"] = POLARIS_LAYER_MATMUL
    curve_inner = {
        MASTER_CURVE_FAMILY_KEY: MASTER_CURVE_FAMILY_POWER,
        **{sk: _curve_stat(equation=f"{sk}=c") for sk in MASTER_SINGLE_STAT_KEYS},
    }
    val = {
        MASTER_ENTRY_TYPE_KEY: MASTER_ENTRY_TYPE_HYBRID,
        MASTER_HYBRID_SINGLE_KEY: {
            MASTER_SINGLE_NUM_CORES_KEY: 4,
            **_single_stats_payload(),
        },
        MASTER_HYBRID_CURVE_KEY: curve_inner,
    }
    p = tmp_path / "m.yaml"
    p.write_text(
        yaml.safe_dump(
            _versioned_doc(
                [{MASTER_YAML_RECORD_KEY_FIELD: key_map, MASTER_YAML_ENTRY_VALUE_FIELD: val}]
            ),
            sort_keys=False,
        )
    )
    m = load_existing_yaml(p)
    kt = yaml_labeled_key_to_tuple(key_map)
    assert m[kt][MASTER_ENTRY_TYPE_KEY] == MASTER_ENTRY_TYPE_HYBRID
    assert m[kt][MASTER_HYBRID_SINGLE_KEY][MASTER_SINGLE_NUM_CORES_KEY] == 4
    assert m[kt][MASTER_HYBRID_CURVE_KEY][MASTER_CURVE_FAMILY_KEY] == MASTER_CURVE_FAMILY_POWER


def test_load_rejects_entry_item_with_extra_top_level_keys(tmp_path: Path):
    key_map = {k: f"v_{k}" for k in KEY_TUPLE_YAML_KEYS[:9]}
    p = tmp_path / "m.yaml"
    p.write_text(
        yaml.safe_dump(
            _versioned_doc(
                [
                    {
                        MASTER_YAML_RECORD_KEY_FIELD: key_map,
                        MASTER_YAML_ENTRY_VALUE_FIELD: {
                            MASTER_ENTRY_TYPE_KEY: MASTER_ENTRY_TYPE_SINGLE,
                            MASTER_SINGLE_NUM_CORES_KEY: 1,
                            **_single_stats_payload(),
                        },
                        "oops": 1,
                    }
                ]
            ),
            sort_keys=False,
        )
    )
    with pytest.raises(ValueError, match="unknown keys"):
        load_existing_yaml(p)


def test_load_rejects_non_mapping_value(tmp_path: Path):
    key_map = {k: f"v_{k}" for k in KEY_TUPLE_YAML_KEYS[:9]}
    p = tmp_path / "m.yaml"
    p.write_text(
        yaml.safe_dump(
            _versioned_doc(
                [{MASTER_YAML_RECORD_KEY_FIELD: key_map, MASTER_YAML_ENTRY_VALUE_FIELD: "bad"}]
            ),
            sort_keys=False,
        )
    )
    with pytest.raises(ValueError, match="mapping"):
        load_existing_yaml(p)


def test_load_rejects_entries_not_a_list(tmp_path: Path):
    doc = {
        MASTER_YAML_SCHEMA_NAME_KEY: MASTER_YAML_SCHEMA_NAME,
        MASTER_YAML_SCHEMA_VERSION_KEY: MASTER_YAML_SCHEMA_VERSION,
        MASTER_YAML_ENTRIES_KEY: {},
    }
    p = tmp_path / "m.yaml"
    p.write_text(yaml.safe_dump(doc, sort_keys=False))
    with pytest.raises(ValueError, match="list"):
        load_existing_yaml(p)


def test_entry_type_predicates():
    s = {
        MASTER_ENTRY_TYPE_KEY: MASTER_ENTRY_TYPE_SINGLE,
        MASTER_SINGLE_NUM_CORES_KEY: 1,
        **_single_stats_payload(),
    }
    c = _full_curve_value()
    h = {
        MASTER_ENTRY_TYPE_KEY: MASTER_ENTRY_TYPE_HYBRID,
        MASTER_HYBRID_CURVE_KEY: {
            MASTER_CURVE_FAMILY_KEY: MASTER_CURVE_FAMILY_LINEAR,
            **{sk: _curve_stat() for sk in MASTER_SINGLE_STAT_KEYS},
        },
    }
    assert is_single_entry(s)
    assert not is_curve_entry(s)
    assert is_curve_entry(c)
    assert not is_single_entry(c)
    assert is_hybrid_entry(h)


def test_merge_non_matmul_overwrites_same_kind():
    kt = ("add",) + tuple(f"k{i}" for i in range(8))
    first = {
        MASTER_ENTRY_TYPE_KEY: MASTER_ENTRY_TYPE_SINGLE,
        MASTER_SINGLE_NUM_CORES_KEY: 1,
        **_single_stats_payload(),
    }
    second = {
        MASTER_ENTRY_TYPE_KEY: MASTER_ENTRY_TYPE_SINGLE,
        MASTER_SINGLE_NUM_CORES_KEY: 8,
        **{k: 99.0 for k in MASTER_SINGLE_STAT_KEYS},
    }
    assert merge_entry_for_key(kt, first, second) == second


def test_merge_non_matmul_rejects_entry_type_change():
    kt = ("mul",) + tuple(f"k{i}" for i in range(8))
    single = {
        MASTER_ENTRY_TYPE_KEY: MASTER_ENTRY_TYPE_SINGLE,
        MASTER_SINGLE_NUM_CORES_KEY: 1,
        **_single_stats_payload(),
    }
    curve = _full_curve_value()
    with pytest.raises(ValueError, match="entry type"):
        merge_entry_for_key(kt, single, curve)


def test_merge_matmul_rejects_empty_hybrid():
    kt = (POLARIS_LAYER_MATMUL,) + tuple(f"k{i}" for i in range(8))
    empty_hybrid = {MASTER_ENTRY_TYPE_KEY: MASTER_ENTRY_TYPE_HYBRID}
    with pytest.raises(ValueError, match="neither single nor curve"):
        merge_entry_for_key(kt, None, empty_hybrid)


def test_merge_matmul_rejects_unknown_entry_type():
    kt = (POLARIS_LAYER_MATMUL,) + tuple(f"k{i}" for i in range(8))
    with pytest.raises(ValueError, match="Matmul merge expected"):
        merge_entry_for_key(kt, None, {MASTER_ENTRY_TYPE_KEY: "other"})


def test_canonicalize_master_promotes_matmul_single_to_hybrid():
    kt = (POLARIS_LAYER_MATMUL,) + tuple(f"k{i}" for i in range(8))
    single = {
        MASTER_ENTRY_TYPE_KEY: MASTER_ENTRY_TYPE_SINGLE,
        MASTER_SINGLE_NUM_CORES_KEY: 3,
        **_single_stats_payload(),
    }
    out = canonicalize_master_for_write({kt: single})
    assert is_hybrid_entry(out[kt])
    assert MASTER_HYBRID_SINGLE_KEY in out[kt]


def test_merge_master_merges_disjoint_keys():
    ka = ("add",) + tuple(f"a{i}" for i in range(8))
    kb = ("mul",) + tuple(f"b{i}" for i in range(8))
    a_val = {
        MASTER_ENTRY_TYPE_KEY: MASTER_ENTRY_TYPE_SINGLE,
        MASTER_SINGLE_NUM_CORES_KEY: 1,
        **_single_stats_payload(),
    }
    b_val = dict(a_val)
    merged = merge_master({ka: a_val}, {kb: b_val}, emit_overwrite_warning=False)
    assert set(merged.keys()) == {ka, kb}


def test_serialize_master_for_yaml_round_trip_document_shape():
    kt = ("add",) + tuple(f"s{i}" for i in range(8))
    val = {
        MASTER_ENTRY_TYPE_KEY: MASTER_ENTRY_TYPE_SINGLE,
        MASTER_SINGLE_NUM_CORES_KEY: 2,
        **_single_stats_payload(),
    }
    doc = serialize_master_for_yaml({kt: val})
    assert doc[MASTER_YAML_SCHEMA_NAME_KEY] == MASTER_YAML_SCHEMA_NAME
    assert doc[MASTER_YAML_SCHEMA_VERSION_KEY] == MASTER_YAML_SCHEMA_VERSION
    assert len(doc[MASTER_YAML_ENTRIES_KEY]) == 1
    item0 = doc[MASTER_YAML_ENTRIES_KEY][0]
    assert set(item0.keys()) == {MASTER_YAML_RECORD_KEY_FIELD, MASTER_YAML_ENTRY_VALUE_FIELD}


def test_format_dry_run_report_includes_curve_summary():
    kt = (POLARIS_LAYER_MATMUL,) + tuple(f"d{i}" for i in range(8))
    curve_inner = {
        MASTER_CURVE_FAMILY_KEY: MASTER_CURVE_FAMILY_LINEAR,
        MASTER_DURATION_MS_KEY: _curve_stat(equation="msecs = 1*c+0"),
        **{sk: _curve_stat(equation=f"{sk}=c") for sk in MASTER_SINGLE_STAT_KEYS if sk != MASTER_DURATION_MS_KEY},
    }
    hybrid = {
        MASTER_ENTRY_TYPE_KEY: MASTER_ENTRY_TYPE_HYBRID,
        MASTER_HYBRID_CURVE_KEY: curve_inner,
    }
    meta = {
        kt: {
            "family": "linear",
            "r2_linear_duration": 0.91,
            "r2_power_duration": 0.82,
            "core_counts": [1.0, 2.0, 4.0],
        }
    }
    text = format_dry_run_report(
        Path("/tmp/out_master.yaml"),
        keys_added=[kt],
        keys_updated=[],
        merged={kt: hybrid},
        curve_meta=meta,
    )
    assert "Would add 1 entry" in text
    assert "curve:" in text
    assert "msecs" in text


def test_build_stats_row_returns_none_when_required_stat_non_numeric():
    row = {
        EXCEL_DURATION_COLUMN: 1_000_000.0,
        EXCEL_COL_DRAM_BW_UTIL: 10.0,
        EXCEL_COL_NOC_UTIL: "not-a-number",
        EXCEL_COL_MULTICAST_NOC: 0.0,
        EXCEL_COL_NPE_CONG: 0.0,
        EXCEL_COL_SFPU: 0.0,
        EXCEL_COL_FPU: 0.0,
    }
    assert (
        build_stats_row(row, 100.0, [], EXCEL_DURATION_COLUMN, "CORE COUNT") is None
    )


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("TilizeWithValPaddingDeviceOperation", "tilizewithvalpadding"),
        ("TilizeDeviceOperation", "tilize"),
        ("UntilizeWithUnpaddingDeviceOperation", "untilizewithunpadding"),
        ("UntilizeDeviceOperation", "untilize"),
    ],
)
def test_excel_op_code_tilize_untilize_variants(raw, expected):
    assert normalize_profiler_opcode(raw) == expected


def test_unary_device_operation_gelu_from_op_chain_attributes():
    attr = (
        "{'op_chain': '{BasicUnaryWithParam<float; int; unsigned int>("
        "base=BasicUnaryWithParam<float>(op_type=UnaryOpType::GELU;param={0}))}'; "
        "'output_dtype': 'DataType::BFLOAT16'}"
    )
    assert normalize_profiler_opcode("UnaryDeviceOperation", attr) == "gelu"


# ---------------------------------------------------------------------------
# Time-only entries (stats_coverage: time_only): no NOC-trace stats
# ---------------------------------------------------------------------------


def _time_only_single(num_cores=1, msecs=1.0):
    return {
        MASTER_ENTRY_TYPE_KEY: MASTER_ENTRY_TYPE_SINGLE,
        MASTER_SINGLE_NUM_CORES_KEY: num_cores,
        **{k: msecs if k == MASTER_DURATION_MS_KEY else 5.0 for k in MASTER_TIME_ONLY_STAT_KEYS},
        MASTER_STATS_COVERAGE_KEY: MASTER_STATS_COVERAGE_TIME_ONLY,
    }


def _full_single(num_cores=1, msecs=2.0):
    return {
        MASTER_ENTRY_TYPE_KEY: MASTER_ENTRY_TYPE_SINGLE,
        MASTER_SINGLE_NUM_CORES_KEY: num_cores,
        **_single_stats_payload(),
        MASTER_DURATION_MS_KEY: msecs,
    }


def _key9(op="add"):
    return (op,) + tuple(f"t{i}" for i in range(8))


def _write_master(tmp_path: Path, master: dict) -> Path:
    p = tmp_path / "m.yaml"
    p.write_text(yaml.safe_dump(serialize_master_for_yaml(master), sort_keys=False))
    return p


def _time_only_table(extra_cols=()):
    cols = [*KEY_TUPLE_COLUMN_NAMES, "CORE COUNT", EXCEL_DURATION_COLUMN, EXCEL_COL_FPU, EXCEL_COL_SFPU, *extra_cols]
    return InputTable(columns=cols, rows=[])


def test_noctrace_and_time_only_stat_keys_partition_the_stat_keys():
    assert set(MASTER_NOCTRACE_STAT_KEYS) | set(MASTER_TIME_ONLY_STAT_KEYS) == set(MASTER_SINGLE_STAT_KEYS)
    assert not set(MASTER_NOCTRACE_STAT_KEYS) & set(MASTER_TIME_ONLY_STAT_KEYS)


def test_validate_columns_without_noctrace_columns_is_time_only():
    _, _, _, required, stat_cols = validate_columns(_time_only_table())
    assert required == [EXCEL_DURATION_COLUMN]
    assert stat_cols == [EXCEL_DURATION_COLUMN]


def test_validate_columns_with_partial_noctrace_columns_still_raises():
    with pytest.raises(KeyError, match="Stats columns missing"):
        validate_columns(_time_only_table(extra_cols=(EXCEL_COL_DRAM_BW_UTIL,)))


def test_build_stats_row_time_only_omits_noctrace_stats():
    row = {EXCEL_DURATION_COLUMN: 2_000_000.0, EXCEL_COL_SFPU: 1.0, EXCEL_COL_FPU: 7.0}
    out = build_stats_row(row, 100.0, [EXCEL_DURATION_COLUMN], EXCEL_DURATION_COLUMN, "CORE COUNT", time_only=True)
    assert out is not None
    assert list(out) == list(MASTER_TIME_ONLY_STAT_KEYS)
    assert out[MASTER_DURATION_MS_KEY] == 2.0


def test_build_master_dict_marks_time_only_single_and_curve():
    stats = {MASTER_DURATION_MS_KEY: 1.0, "vector_pipe_util": 2.0, "matrix_pipe_util": 3.0}
    single, _ = build_master_dict([(_key9(), 4.0, stats, "add", [stats])], "single")
    assert single[_key9()][MASTER_STATS_COVERAGE_KEY] == MASTER_STATS_COVERAGE_TIME_ONLY
    mm = _key9(POLARIS_LAYER_MATMUL)
    accepted = [(mm, float(c), {**stats, MASTER_DURATION_MS_KEY: float(c)}, POLARIS_LAYER_MATMUL, []) for c in (4, 8)]
    curve, _ = build_master_dict(accepted, "curve")
    assert curve[mm][MASTER_STATS_COVERAGE_KEY] == MASTER_STATS_COVERAGE_TIME_ONLY
    assert not set(curve[mm]) & set(MASTER_NOCTRACE_STAT_KEYS)


def test_build_master_dict_full_entries_carry_no_marker():
    stats = dict(_single_stats_payload())
    single, _ = build_master_dict([(_key9(), 4.0, stats, "add", [stats])], "single")
    assert MASTER_STATS_COVERAGE_KEY not in single[_key9()]


def test_load_time_only_single_round_trip(tmp_path: Path):
    m = load_existing_yaml(_write_master(tmp_path, {_key9(): _time_only_single()}))
    assert payload_stats_coverage(m[_key9()]) == MASTER_STATS_COVERAGE_TIME_ONLY
    assert not set(m[_key9()]) & set(MASTER_NOCTRACE_STAT_KEYS)


def test_loader_does_not_warn_on_stats_coverage_key(tmp_path: Path):
    p = _write_master(tmp_path, {_key9(): _time_only_single()})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        load_existing_yaml(p)


@pytest.mark.parametrize("noc_key", MASTER_NOCTRACE_STAT_KEYS)
def test_loader_rejects_time_only_single_with_noctrace_stat(tmp_path: Path, noc_key):
    val = {**_time_only_single(), noc_key: 0.0}
    with pytest.raises(ValueError, match="must not carry NOC-trace stats"):
        load_existing_yaml(_write_master(tmp_path, {_key9(): val}))


def test_loader_rejects_full_single_missing_noctrace_stat(tmp_path: Path):
    val = _full_single()
    del val["mem_util"]
    with pytest.raises(ValueError, match="missing keys"):
        load_existing_yaml(_write_master(tmp_path, {_key9(): val}))


def test_loader_rejects_unknown_stats_coverage(tmp_path: Path):
    val = {**_time_only_single(), MASTER_STATS_COVERAGE_KEY: "partial"}
    with pytest.raises(ValueError, match="must be one of"):
        load_existing_yaml(_write_master(tmp_path, {_key9(): val}))


def test_loader_rejects_time_only_curve_with_noctrace_fit(tmp_path: Path):
    curve = {
        MASTER_ENTRY_TYPE_KEY: MASTER_ENTRY_TYPE_CURVE,
        MASTER_CURVE_FAMILY_KEY: MASTER_CURVE_FAMILY_LINEAR,
        MASTER_DURATION_MS_KEY: _curve_stat(),
        "memory_traffic": _curve_stat(),
        MASTER_STATS_COVERAGE_KEY: MASTER_STATS_COVERAGE_TIME_ONLY,
    }
    with pytest.raises(ValueError, match="must not carry NOC-trace stat fits"):
        load_existing_yaml(_write_master(tmp_path, {_key9("mul"): curve}))


def test_serialize_writes_coverage_block_matching_entries():
    master = {_key9("add"): _full_single(), _key9("mul"): _time_only_single()}
    doc = serialize_master_for_yaml(master)
    assert doc[MASTER_YAML_COVERAGE_KEY] == {MASTER_STATS_COVERAGE_FULL: 1, MASTER_STATS_COVERAGE_TIME_ONLY: 1}
    assert doc[MASTER_YAML_COVERAGE_KEY] == master_coverage_counts(master)


def test_loader_rejects_coverage_block_that_disagrees_with_entries(tmp_path: Path):
    doc = serialize_master_for_yaml({_key9(): _time_only_single()})
    doc[MASTER_YAML_COVERAGE_KEY] = {MASTER_STATS_COVERAGE_FULL: 1, MASTER_STATS_COVERAGE_TIME_ONLY: 0}
    p = tmp_path / "m.yaml"
    p.write_text(yaml.safe_dump(doc, sort_keys=False))
    with pytest.raises(ValueError, match="coverage"):
        load_existing_yaml(p)


def test_entry_stats_coverage_hybrid_is_time_only_if_any_payload_is():
    hybrid = {
        MASTER_ENTRY_TYPE_KEY: MASTER_ENTRY_TYPE_HYBRID,
        MASTER_HYBRID_SINGLE_KEY: {k: v for k, v in _time_only_single().items() if k != MASTER_ENTRY_TYPE_KEY},
    }
    assert entry_stats_coverage(hybrid) == MASTER_STATS_COVERAGE_TIME_ONLY
    assert entry_stats_coverage(_full_single()) == MASTER_STATS_COVERAGE_FULL


def test_merge_time_only_does_not_replace_full():
    full, time_only = _full_single(), _time_only_single()
    assert merge_entry_for_key(_key9(), full, time_only) == full


def test_merge_full_replaces_time_only():
    full, time_only = _full_single(), _time_only_single()
    assert merge_entry_for_key(_key9(), time_only, full) == full


def test_merge_hybrid_time_only_single_does_not_replace_full_single():
    mm = _key9(POLARIS_LAYER_MATMUL)
    merged = merge_entry_for_key(mm, _full_single(num_cores=2), _time_only_single(num_cores=2))
    assert payload_stats_coverage(merged[MASTER_HYBRID_SINGLE_KEY]) == MASTER_STATS_COVERAGE_FULL
    assert set(MASTER_NOCTRACE_STAT_KEYS) <= set(merged[MASTER_HYBRID_SINGLE_KEY])


def test_merge_hybrid_single_is_replaced_whole_not_field_merged():
    # A full single replacing a time-only single must not inherit the old payload's marker.
    mm = _key9(POLARIS_LAYER_MATMUL)
    merged = merge_entry_for_key(mm, _time_only_single(num_cores=2), _full_single(num_cores=2))
    assert MASTER_STATS_COVERAGE_KEY not in merged[MASTER_HYBRID_SINGLE_KEY]
