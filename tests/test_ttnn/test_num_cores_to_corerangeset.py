# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""``device.num_cores_to_corerangeset`` — the degenerate-grid compatibility shim.

This function is the one place in the port with a *demonstrated* silent-regression
history, which is why it gets its own file.

It began as a stub returning the tuple ``(1, 1)``. Replacing it with a faithful
port of tt-metal's ``num_cores_to_corerangeset``
(``tt_metal/common/work_split.cpp``) broke llama3 at graph-build time: five call
sites in ``workloads/ttnn/tt_transformers*`` reach it with a device whose
``grid_x``/``grid_y`` are still 0, and the faithful implementation asserts on such
a grid. llama3 was not in the 3-workload verification gate at the time, so nothing
caught it. A straight revert then broke ResNet-50, which is a sixth caller and does
need a real CoreRangeSet (it calls ``.num_cores()`` on the result).

The resolution is a split: ``conv_sharding._num_cores_to_corerangeset`` is the
faithful port and asserts on a bad grid; the public ``device`` entry point keeps
the legacy ``(1, 1)`` for a zero or missing grid and delegates otherwise. These
tests pin BOTH halves, because either one alone regresses a real workload.
"""

import pytest

from ttsim.front.ttnn.conv_sharding import _num_cores_to_corerangeset
from ttsim.front.ttnn.device import num_cores_to_corerangeset

# ---------------------------------------------------------------------------
# the public entry point: legacy (1, 1) for a grid it cannot use
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize(
    'grid_size, target, why',
    [
        ((0, 0), 8, 'tt_transformers devices whose grid was never set'),
        ((0, 8), 8, 'partially-zero grid'),
        ((8, 0), 8, 'partially-zero grid, other axis'),
        (None, 8, 'grid omitted entirely'),
        ((8, 8), 0, 'no cores requested'),
        ((8, 8), None, 'target omitted'),
    ],
)
def test_degenerate_grid_returns_legacy_tuple(grid_size, target, why) -> None:
    """A grid this cannot model must NOT assert — llama3 et al. depend on that.

    The return value is deliberately the opaque legacy tuple rather than a
    CoreRangeSet: those callers treat it as an opaque value and never inspect it.
    """
    assert num_cores_to_corerangeset(target, grid_size) == (1, 1), why


@pytest.mark.unit
def test_real_grid_delegates_and_returns_a_corerangeset() -> None:
    """ResNet-50's path: a real grid must produce something with .num_cores()."""
    result = num_cores_to_corerangeset(98, (12, 10))
    assert not isinstance(result, tuple), f'expected a CoreRangeSet, got {result!r}'
    assert result.num_cores() == 98


@pytest.mark.unit
@pytest.mark.parametrize('target', [1, 8, 64, 98, 105, 120])
def test_core_count_is_exactly_what_was_asked_for(target) -> None:
    """The split must conserve the requested core count on a 12x10 (p100a) grid."""
    assert num_cores_to_corerangeset(target, (12, 10)).num_cores() == target


@pytest.mark.unit
def test_kwargs_and_positional_agree() -> None:
    """Callers pass these both ways; the shim normalises kwargs over positionals."""
    a = num_cores_to_corerangeset(64, (8, 8))
    b = num_cores_to_corerangeset(target_num_cores=64, grid_size=(8, 8))
    assert a.num_cores() == b.num_cores() == 64


@pytest.mark.unit
def test_grid_object_accepted_as_well_as_tuple() -> None:
    """``compute_with_storage_grid_size`` returns a tuple in the shim but an
    object with .x/.y in real ttnn; both spellings must work."""
    class _Grid:
        x, y = 8, 8

    assert num_cores_to_corerangeset(64, _Grid()).num_cores() == 64


# ---------------------------------------------------------------------------
# the private port: asserts rather than silently mis-modelling
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_private_port_asserts_on_degenerate_grid() -> None:
    """This is the assertion that took llama3 down, and it must stay.

    conv_sharding always has a real grid from ``compute_grid_size``, so reaching
    here with (0, 0) means the caller lost the grid — worth failing loudly rather
    than modelling a zero-core device.
    """
    with pytest.raises(ValueError, match='needs a real compute grid'):
        _num_cores_to_corerangeset(8, (0, 0))


@pytest.mark.unit
def test_private_port_asserts_on_oversized_request() -> None:
    """Asking for more cores than the grid holds is a modelling error."""
    with pytest.raises(ValueError, match='greater than total available'):
        _num_cores_to_corerangeset(121, (12, 10))


@pytest.mark.unit
def test_at_most_three_ranges() -> None:
    """cpp: a partial row/col, then whole rows/cols, then a partial — max three."""
    for target in (1, 7, 13, 64, 98, 119):
        crs = _num_cores_to_corerangeset(target, (12, 10))
        assert len(crs.ranges) <= 3, (target, crs.ranges)


@pytest.mark.unit
@pytest.mark.parametrize('row_wise', [False, True])
def test_row_wise_conserves_core_count(row_wise) -> None:
    """Both traversal orders must split the same total, however they arrange it."""
    for target in (1, 13, 64, 98):
        crs = _num_cores_to_corerangeset(target, (12, 10), row_wise)
        assert crs.num_cores() == target, (target, row_wise)


# ---------------------------------------------------------------------------
# CoreRangeSet equality: decomposition, not covered-coordinate set
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_row_wise_and_col_wise_same_cores_compare_unequal() -> None:
    """Same covered cores, different shard placement — must NOT compare equal.

    Both calls cover {(0, 0), (1, 0), (0, 1)}, but row-wise puts shard 1 on
    (1, 0) and column-wise puts it on (0, 1). tt-metal's
    ``operator==(CoreRangeSet, CoreRangeSet)`` walks the range vectors pairwise
    (``tt_metal/common/core_coord.cpp:484``) and separates them, so silicon
    reshards between these two configs. An equality that compared covered
    coordinate sets instead would call them identical and suppress that Reshard.
    """
    row = _num_cores_to_corerangeset(3, (2, 2), True)
    col = _num_cores_to_corerangeset(3, (2, 2), False)
    assert {(c.x, c.y) for c in row} == {(c.x, c.y) for c in col}
    assert [(c.x, c.y) for c in row] != [(c.x, c.y) for c in col]
    assert row != col
    assert hash(row) != hash(col)


@pytest.mark.unit
def test_equal_grids_compare_equal_and_hash_equal() -> None:
    """The bug this equality exists for: two independently built identical grids.

    Before ``CoreRangeSet.__eq__`` existed these compared by identity, so
    ResNet-50's ``if ds_out.memory_config() != out.memory_config()`` fired
    unconditionally and emitted three Reshards the p100a capture does not have.
    """
    for target, grid, row_wise in [(112, (13, 10), True), (72, (12, 10), True),
                                   (98, (12, 10), False), (1, (8, 8), True)]:
        a = _num_cores_to_corerangeset(target, grid, row_wise)
        b = _num_cores_to_corerangeset(target, grid, row_wise)
        assert a is not b
        assert a == b, (target, grid, row_wise)
        assert hash(a) == hash(b), (target, grid, row_wise)


@pytest.mark.unit
def test_no_reachable_grid_has_two_decompositions_of_one_rectangle() -> None:
    """A full rectangle is only ever emitted as a single range.

    The failure mode a covered-set comparison would guard against — one
    rectangular range versus several adjacent row ranges over the same grid — is
    not reachable: ``_num_cores_to_corerangeset`` collapses whole rows (or
    columns) into one range, so ``n == gx * gy`` always yields exactly the same
    single range the block-sharded path builds by hand.
    """
    for gx, gy in [(2, 2), (8, 8), (12, 10), (13, 10), (1, 7), (7, 1)]:
        full_row = _num_cores_to_corerangeset(gx * gy, (gx, gy), True)
        full_col = _num_cores_to_corerangeset(gx * gy, (gx, gy), False)
        assert len(full_row.ranges) == 1, (gx, gy, full_row.ranges)
        assert len(full_col.ranges) == 1, (gx, gy, full_col.ranges)
        assert full_row == full_col
