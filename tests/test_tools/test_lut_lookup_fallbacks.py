# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The two LUT lookup fallbacks added by the ResNet-50 dual-mode port.

Both live in ``tools/perf_lookup/lookup_operator_perf.py`` and both fire only
after the literal key misses, substituting a different key and recording which
substitution succeeded in ``hit_source``:

``add_output_as_input2``
    For the opcodes in ``_BINARY_OUTPUT_AS_INPUT2_OPCODES`` (currently just
    ``add``), the profiler records an in-place residual add with its OUTPUT
    tensor in the input_2 position.  Polaris builds the key from input_1, so the
    literal key misses; the fallback rebuilds it with the op's own output.

``reshard_y_tile_pad``
    For the opcodes in ``_RESHARD_Y_TILE_PAD_FALLBACK_OPCODES``, retry with Y
    rounded up to a tile row.  The LUT is internally mixed on this point -- some
    rows were recorded with the logical Y and some with the tile-padded Y.

These are the two fallbacks most able to move OTHER workloads' hit rates: they
key on the opcode alone, so any ``add`` or ``reshard`` in any model reaches
them.  Each test therefore pins the negative case too -- that the fallback does
NOT fire for an opcode outside its set, and does not fire when the literal key
already hit.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from tools.perf_lookup.tt_perf_master_schema import MASTER_SINGLE_STAT_KEYS
from ttsim.ops.tensor import SimTensor


def _value(num_cores: int, **overrides) -> dict:
    """Valid ``entry_type: single`` payload."""
    out = {
        "entry_type": "single",
        "num_cores": num_cores,
        "msecs": 0.0,
        "memory_traffic": 0.0,
        "mem_util": 0.0,
        "noc_util": 0.0,
        "noc_multicast_util": 0.0,
        "npe_cong_impact_pct": 0.0,
        "vector_pipe_util": 0.0,
        "matrix_pipe_util": 0.0,
    }
    for k in MASTER_SINGLE_STAT_KEYS:
        assert k in out
    out.update(overrides)
    return out


#: A bare ``SimTensor`` carries no layout or memory config, and the key builder
#: renders that as TILE / DRAM_INTERLEAVED.  The LUT rows below must use the
#: same spelling or every lookup misses on a field these tests are not about --
#: which is how the first draft of this file "passed" its negative cases.
_DEFAULT_LAYOUT = "TILE"
_DEFAULT_MEMORY = "DEV_1_DRAM_INTERLEAVED"


def _in0(y: int, x: int, layout: str = _DEFAULT_LAYOUT, mem: str = _DEFAULT_MEMORY) -> dict:
    return {
        "input_0_w_pad_logical": 1,
        "input_0_z_pad_logical": 1,
        "input_0_y_pad_logical": y,
        "input_0_x_pad_logical": x,
        "input_0_layout": layout,
        "input_0_datatype": "BFLOAT16",
        "input_0_memory": mem,
    }


def _inN(n: int, y: int, x: int, layout: str = _DEFAULT_LAYOUT,
         mem: str = _DEFAULT_MEMORY) -> dict:
    return {
        f"input_{n}_w_pad_logical": 1,
        f"input_{n}_z_pad_logical": 1,
        f"input_{n}_y_pad_logical": y,
        f"input_{n}_x_pad_logical": x,
        f"input_{n}_layout": layout,
        f"input_{n}_datatype": "BFLOAT16",
        f"input_{n}_memory": mem,
    }


def _write_lut(tmp_path: Path, entries: list[dict], name: str = "lut.yaml") -> Path:
    doc = {"schema_name": "correqn.tt-perf-master", "schema_version": 2, "entries": entries}
    p = tmp_path / name
    p.write_text(yaml.dump(doc, sort_keys=False), encoding="utf-8")
    return p


def _t(name: str, shape: list[int]) -> SimTensor:
    return SimTensor({"name": name, "shape": list(shape), "op_in": [], "op_out": []})


# ---------------------------------------------------------------------------
# add_output_as_input2
# ---------------------------------------------------------------------------

def _binary_op_and_graph(optype: str, out_shape: list[int]):
    """An arity-2 op whose OUTPUT differs in shape from input_1.

    The two differ deliberately: if they were identical the fallback key would
    equal the literal key and the test could not tell which one hit.
    """
    t0 = _t("a", [1, 1, 64, 32])
    t1 = _t("b", [1, 1, 64, 32])
    tout = _t("c", out_shape)
    op = SimpleNamespace(
        optype=optype, precision="BF16", name="op0",
        inList=["a", "b"], outList=["c"],
    )
    g = SimpleNamespace(_tensors={"a": t0, "b": t1, "c": tout})
    return op, g


@pytest.mark.unit
def test_add_output_as_input2_fires_when_the_literal_key_misses(tmp_path: Path):
    """The LUT holds the OUTPUT-as-input_2 spelling; the literal key is absent."""
    from tools.perf_lookup.lookup_operator_perf import OperatorPerfMap

    key = {"op_code": "add", **_in0(64, 32), **_inN(1, 64, 32), **_inN(2, 96, 32),
           "math_fidelity": "N/A"}
    m = OperatorPerfMap(_write_lut(tmp_path, [{"key": key, "value": _value(8, msecs=0.05)}]))

    op, g = _binary_op_and_graph("Add", [1, 1, 96, 32])
    st = m.lookup(op, g, core_count=8)

    assert st is not None, "add_output_as_input2 did not fire"
    assert st.hit_source == "add_output_as_input2"
    assert st.msecs == pytest.approx(0.05)
    # The substitution really happened: resolved != literal.
    assert st.key_literal != st.key_resolved
    assert st.key_resolved in m._entries


@pytest.mark.unit
def test_add_output_as_input2_does_not_fire_for_a_non_add_opcode(tmp_path: Path):
    """Keyed on the opcode set, so a Mul with the same shapes must NOT match.

    This is the half that protects other workloads: the fallback reaches every
    binary in every model, and only ``add`` is evidenced.
    """
    from tools.perf_lookup.lookup_operator_perf import OperatorPerfMap

    key = {"op_code": "mul", **_in0(64, 32), **_inN(1, 64, 32), **_inN(2, 96, 32),
           "math_fidelity": "N/A"}
    m = OperatorPerfMap(_write_lut(tmp_path, [{"key": key, "value": _value(8, msecs=0.05)}]))

    op, g = _binary_op_and_graph("Mul", [1, 1, 96, 32])
    st = m.lookup(op, g, core_count=8)

    assert st is None or st.hit_source != "add_output_as_input2", (
        "the output-as-input_2 substitution leaked to a non-add opcode"
    )


@pytest.mark.unit
def test_add_literal_hit_is_not_relabelled_as_a_fallback(tmp_path: Path):
    """Positive control: when the literal key hits, hit_source must stay direct.

    Without this, a fallback that fired unconditionally would still pass the
    first test.
    """
    from tools.perf_lookup.lookup_operator_perf import OperatorPerfMap

    # NO input_2: for an arity-2 op the literal key is the 16-tuple
    # (input_0 + input_1 + math_fidelity).  A row carrying input_2 is a 23-tuple
    # and can only ever be reached through the fallback -- writing one here is
    # how the first draft of this test accidentally asserted the fallback path
    # while claiming to test the direct one.
    key = {"op_code": "add", **_in0(64, 32), **_inN(1, 64, 32), "math_fidelity": "N/A"}
    m = OperatorPerfMap(_write_lut(tmp_path, [{"key": key, "value": _value(8, msecs=0.07)}]))

    op, g = _binary_op_and_graph("Add", [1, 1, 96, 32])
    st = m.lookup(op, g, core_count=8)

    assert st is not None
    assert st.hit_source == "direct", f"literal hit was relabelled {st.hit_source!r}"
    assert st.key_literal == st.key_resolved


# ---------------------------------------------------------------------------
# reshard_y_tile_pad
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_reshard_y_tile_pad_fires_on_a_non_tile_multiple_y(tmp_path: Path):
    """Y=100 misses; the LUT holds Y rounded up to the next tile row (128)."""
    from tools.perf_lookup.lookup_operator_perf import OperatorPerfMap

    key = {"op_code": "reshard", **_in0(128, 64), "math_fidelity": "N/A"}
    m = OperatorPerfMap(_write_lut(tmp_path, [{"key": key, "value": _value(8, msecs=0.02)}]))

    t0 = _t("a", [1, 1, 100, 64])
    op = SimpleNamespace(optype="Reshard", precision="BF16", name="op0",
                         inList=["a"], outList=["b"])
    g = SimpleNamespace(_tensors={"a": t0, "b": _t("b", [1, 1, 100, 64])})

    st = m.lookup(op, g, core_count=8)
    assert st is not None, "reshard_y_tile_pad did not fire"
    assert st.hit_source == "reshard_y_tile_pad"
    assert st.msecs == pytest.approx(0.02)
    assert st.key_literal != st.key_resolved


@pytest.mark.unit
def test_reshard_y_tile_pad_does_not_fire_for_an_unlisted_opcode(tmp_path: Path):
    """``move`` is not in _RESHARD_Y_TILE_PAD_FALLBACK_OPCODES, so no padding retry."""
    from tools.perf_lookup.lookup_operator_perf import OperatorPerfMap

    key = {"op_code": "move", **_in0(128, 64), "math_fidelity": "N/A"}
    m = OperatorPerfMap(_write_lut(tmp_path, [{"key": key, "value": _value(8, msecs=0.02)}]))

    t0 = _t("a", [1, 1, 100, 64])
    op = SimpleNamespace(optype="Move", precision="BF16", name="op0",
                         inList=["a"], outList=["b"])
    g = SimpleNamespace(_tensors={"a": t0, "b": _t("b", [1, 1, 100, 64])})

    st = m.lookup(op, g, core_count=8)
    assert st is None or st.hit_source != "reshard_y_tile_pad", (
        "the tile-padding retry leaked to an opcode outside its set"
    )


@pytest.mark.unit
def test_reshard_y_already_tile_aligned_takes_the_literal_key(tmp_path: Path):
    """Y=128 is already a tile multiple, so the guard skips the retry entirely."""
    from tools.perf_lookup.lookup_operator_perf import OperatorPerfMap

    key = {"op_code": "reshard", **_in0(128, 64), "math_fidelity": "N/A"}
    m = OperatorPerfMap(_write_lut(tmp_path, [{"key": key, "value": _value(8, msecs=0.02)}]))

    t0 = _t("a", [1, 1, 128, 64])
    op = SimpleNamespace(optype="Reshard", precision="BF16", name="op0",
                         inList=["a"], outList=["b"])
    g = SimpleNamespace(_tensors={"a": t0, "b": _t("b", [1, 1, 128, 64])})

    st = m.lookup(op, g, core_count=8)
    assert st is not None
    assert st.hit_source == "direct"
    assert st.key_literal == st.key_resolved


# ---------------------------------------------------------------------------
# the opcode sets themselves
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_fallback_opcode_sets_stay_narrow():
    """Both sets key on the opcode alone, so widening one widens its blast radius.

    Pinned because the cost of getting this wrong is measured rather than
    hypothetical: the ``reshard_y_tile_pad`` comment records that the
    fallback-vs-substitution ordering costs ViT 13 reshard hits when wrong.
    """
    from tools.perf_lookup.lookup_operator_perf import (
        _BINARY_OUTPUT_AS_INPUT2_OPCODES,
        _RESHARD_Y_TILE_PAD_FALLBACK_OPCODES,
    )

    assert _BINARY_OUTPUT_AS_INPUT2_OPCODES == frozenset({"add"})
    assert _RESHARD_Y_TILE_PAD_FALLBACK_OPCODES == frozenset(
        {"nlpcreateqkvheadsdecode", "reshard"}
    )
