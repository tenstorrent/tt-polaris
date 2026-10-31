# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The two shard-spec constructors in ``ttsim/front/ttnn/memory.py``.

``P:<line>`` cites the ported model in this repository,
``workloads/ttnn/resnet50/common/ttnn_functional_resnet50.py``.  Its own
header cites the tt-metal canonical source as ``C:<line>``; the two files
differ in length, so the prefixes are not interchangeable.  ``P:`` is used
here because that file is in the repo, while the canonical checkout is not.

Both used to discard the shard spec -- ``create_sharded_memory_config`` returned
``shard_spec=None`` and ``create_sharded_memory_config_`` stored the placeholder
``(1, 1)``.  Three consumers read that spec: ``needs_shard_or_reshard`` (every
clause in ``tt-metal/ttnn/cpp/ttnn/operations/conv/conv2d/conv2d_utils.cpp:686-745`` indexes
``input_shard_spec.shape``),
``get_input_channels_alignment``, and ``MemoryConfig.__eq__``.  With the stub,
``shard_w % alignment`` was ``1 % 32``, so the reshard predicate answered True for
every conv that reached it, and two configs that hardware treats as different
compared equal.

Neither constructor changes a tensor shape, so a regression here does not crash
anything -- it moves a LUT key.  That is what these tests are for.

The two are NOT interchangeable and the distinction is the easiest thing to get
wrong when editing either:

* ``create_sharded_memory_config``  takes the FULL tensor shape plus a
  ``ShardStrategy``, and DERIVES the per-core extent from the grid.  32 callers.
* ``create_sharded_memory_config_`` takes the per-core extent ALREADY COMPUTED by
  the caller plus a ``TensorMemoryLayout``, and stores it verbatim.  2 callers,
  both in ResNet-50.
"""

from typing import Any

import pytest

from ttsim.front.ttnn.buffer import BufferType, ShardOrientation, TensorMemoryLayout
from ttsim.front.ttnn.core import CoreCoord, CoreRange, CoreRangeSet
from ttsim.front.ttnn.memory import (
    MemoryConfig,
    create_sharded_memory_config,
    create_sharded_memory_config_,
)
from ttsim.front.ttnn.tensor import ShardStrategy

ROW = ShardOrientation.ROW_MAJOR
COL = ShardOrientation.COL_MAJOR


def _core_grid(x: int, y: int):
    """``CoreGrid(x=, y=)`` -- the form that exposes ``.x`` / ``.y``."""
    from ttsim.front.ttnn.core import CoreGrid
    return CoreGrid(x=x, y=y)


def _bare_range_set(x: int, y: int) -> CoreRangeSet:
    """A plain CoreRangeSet: same cores, but no ``.x`` / ``.y`` attributes."""
    return CoreRangeSet([CoreRange(CoreCoord(0, 0), CoreCoord(x - 1, y - 1))])


# ===========================================================================
# create_sharded_memory_config -- derives the extent from the grid
# ===========================================================================

@pytest.mark.unit
@pytest.mark.parametrize(
    'strategy, expected_layout, expected_shard',
    [
        # grid 8x4 -> gx=8, gy=4, total=32; tensor (1, 1, 1024, 256) -> sh=1024, sw=256
        (ShardStrategy.BLOCK, TensorMemoryLayout.BLOCK_SHARDED, (256, 32)),
        (ShardStrategy.HEIGHT, TensorMemoryLayout.HEIGHT_SHARDED, (32, 256)),
        (ShardStrategy.WIDTH, TensorMemoryLayout.WIDTH_SHARDED, (1024, 8)),
    ],
)
def test_strategy_selects_both_the_layout_and_the_split_axis(
        strategy, expected_layout, expected_shard) -> None:
    """Ported from ``ttnn/ttnn/core.py::create_sharded_memory_config``::

        BLOCK  -> (sh / grid.y, sw / grid.x)
        HEIGHT -> (sh / (grid.x * grid.y), sw)
        WIDTH  -> (sh, sw / (grid.x * grid.y))

    The three shapes are mutually distinct on a non-square grid, which is why the
    grid here is 8x4 and not 8x8: on a square grid BLOCK and a transposed BLOCK
    produce the same pair and the test would not notice an axis swap.
    """
    mc = create_sharded_memory_config(
        (1, 1, 1024, 256), _core_grid(8, 4), strategy, ROW)
    assert mc.memory_layout == expected_layout
    assert mc.buffer_type == BufferType.L1
    assert mc.shard_spec is not None
    assert mc.shard_spec.shape == expected_shard


@pytest.mark.unit
def test_leading_dims_multiply_into_the_shard_height() -> None:
    """``shard_height = prod(batch dims) * height``.  A 2x3 batch makes the
    height six times larger; ignoring the leading dims would leave it unchanged,
    which the paired assertion below is what detects."""
    grid = _core_grid(8, 4)
    batched = create_sharded_memory_config(
        (2, 3, 1024, 256), grid, ShardStrategy.HEIGHT, ROW)
    plain = create_sharded_memory_config(
        (1, 1, 1024, 256), grid, ShardStrategy.HEIGHT, ROW)
    assert batched.shard_spec.shape == (192, 256)
    assert plain.shard_spec.shape == (32, 256)


@pytest.mark.unit
def test_indivisible_extents_round_up_instead_of_raising() -> None:
    """A deliberate divergence from real ttnn, which raises when a dimension does
    not divide the grid.  Polaris builds graphs for shapes silicon may never run,
    so a hard failure at graph-build time is worse than an approximate extent.

    1000 over 32 cores is 31.25 -> 32, and 100 over 8 is 12.5 -> 13.  Truncating
    division would give 31 and 12, so this also pins the direction of rounding.
    """
    grid = _core_grid(8, 4)
    height = create_sharded_memory_config(
        (1, 1, 1000, 100), grid, ShardStrategy.HEIGHT, ROW)
    block = create_sharded_memory_config(
        (1, 1, 1000, 100), grid, ShardStrategy.BLOCK, ROW)
    assert height.shard_spec.shape == (32, 100)
    assert block.shard_spec.shape == (250, 13)


@pytest.mark.unit
def test_use_height_and_width_flag_takes_the_last_two_dims_verbatim() -> None:
    """With the flag set, the caller has already computed the per-core extent, so
    the last two dims are stored as-is -- the batch dims AND the grid are both
    ignored.  ResNet-50 sets it at four sites (P:1146, P:1485, P:1521, P:1575)
    and VGG UNet at four more.

    The control alongside it uses the same shape and grid with the flag off, so a
    change that made the flag a no-op would fail rather than coincide.
    """
    grid = _core_grid(8, 4)
    flagged = create_sharded_memory_config(
        (2, 3, 1024, 256), grid, ShardStrategy.BLOCK, ROW,
        use_height_and_width_as_shard_shape=True)
    derived = create_sharded_memory_config(
        (2, 3, 1024, 256), grid, ShardStrategy.BLOCK, ROW,
        use_height_and_width_as_shard_shape=False)
    assert flagged.shard_spec.shape == (1024, 256)
    assert derived.shard_spec.shape != (1024, 256)


@pytest.mark.unit
@pytest.mark.parametrize(
    'orientation, expected',
    [(ROW, ROW), (COL, COL), (None, ROW), ('not-an-orientation', ROW)],
)
def test_orientation_is_preserved_or_defaulted(orientation, expected) -> None:
    """``orientation`` defaults to None (many callers omit it) and real ttnn
    enums arrive as foreign objects, so anything that is not a shim
    ShardOrientation falls back to ROW_MAJOR.  COL_MAJOR must survive, or every
    block-sharded config silently becomes row-major."""
    mc = create_sharded_memory_config(
        (1, 1, 1024, 256), _core_grid(8, 4), ShardStrategy.BLOCK, orientation)
    assert mc.shard_spec.orientation == expected


@pytest.mark.unit
def test_a_grid_without_x_and_y_yields_no_shard_spec() -> None:
    """KNOWN LIMITATION, pinned so it is a decision rather than a surprise.

    The extent is derived from ``core_grid.x`` / ``core_grid.y``.  ``CoreGrid``
    always exposes both (core.py:198 and :204, even for the positional form), but
    a bare ``CoreRangeSet`` does not -- and three call sites in
    ``workloads/ttnn/tt_transformers_dualmode/attention.py`` (:268, :307, :330)
    pass exactly that.  For those the spec stays None and this constructor is
    inert, as it was before the port.

    Asserted against a CoreGrid over the identical cores, so the test fails if
    either half of that split changes.
    """
    same_cores = create_sharded_memory_config(
        (1, 1, 1024, 256), _core_grid(8, 4), ShardStrategy.BLOCK, ROW)
    bare = create_sharded_memory_config(
        (1, 1, 1024, 256), _bare_range_set(8, 4), ShardStrategy.BLOCK, ROW)
    assert same_cores.shard_spec is not None
    assert bare.shard_spec is None
    assert bare.memory_layout == TensorMemoryLayout.BLOCK_SHARDED


@pytest.mark.unit
@pytest.mark.parametrize(
    'shape, strategy, expected_layout',
    [
        ((1, 1, 1024, 256), 'nonsense', TensorMemoryLayout.INTERLEAVED),
        (None, ShardStrategy.BLOCK, TensorMemoryLayout.BLOCK_SHARDED),
        ((1024,), ShardStrategy.BLOCK, TensorMemoryLayout.BLOCK_SHARDED),
        (('a', 'b'), ShardStrategy.BLOCK, TensorMemoryLayout.BLOCK_SHARDED),
    ],
)
def test_unusable_input_degrades_quietly_rather_than_failing_the_build(
        shape, strategy, expected_layout) -> None:
    """An unmapped strategy, a missing shape, a 1-D shape and a non-numeric shape
    must all return the pre-port result -- the right layout with no spec -- rather
    than raise.  This runs during graph construction for every workload, so an
    exception here takes down the build of a model that would otherwise simulate.
    """
    mc = create_sharded_memory_config(shape, _core_grid(8, 4), strategy, ROW)
    assert mc.memory_layout == expected_layout
    assert mc.shard_spec is None
    assert mc.buffer_type == BufferType.L1


# ===========================================================================
# create_sharded_memory_config_ -- stores the caller's extent verbatim
# ===========================================================================

@pytest.mark.unit
def test_underscore_variant_stores_a_two_element_extent_verbatim() -> None:
    """ResNet-50's pre-layer4 block config passes
    ``[nearest_32(h // grid_y), w // grid_x]`` (P:1479-1480) -- already
    per-core.  It must be stored, not recomputed: the grid here is 8x4, so a
    constructor that divided again would return something other than (320, 128).
    """
    mc = create_sharded_memory_config_(
        [320, 128], _core_grid(8, 4), TensorMemoryLayout.BLOCK_SHARDED, ROW, True)
    assert mc.shard_spec.shape == (320, 128)
    assert mc.memory_layout == TensorMemoryLayout.BLOCK_SHARDED
    assert mc.buffer_type == BufferType.L1


@pytest.mark.unit
@pytest.mark.parametrize(
    'shape, why',
    [
        ([1, 1, 320, 128], 'a full 4-D tensor shape, as P:1349 passes'),
        ([8, 4, 320, 128], 'a 4-D shape whose leading dims are NOT 1'),
        ([0, 128], 'a zero extent'),
        ([-4, 128], 'a negative extent'),
        (5, 'not a sequence at all'),
        (None, 'nothing'),
    ],
)
def test_underscore_variant_falls_back_to_the_placeholder(shape, why) -> None:
    """Anything that is not a usable 2-tuple of positive ints keeps the old
    ``(1, 1)`` placeholder rather than raising.

    The 4-D rows are not hypothetical: ``ttnn_functional_resnet50.py:1285`` passes
    ``x.shape`` to this constructor, so that call site still gets the placeholder
    and behaves exactly as it did before the port.

    Both 4-D rows are needed.  Relaxing the guard to ``len(shape) >= 2`` would
    read the first two entries instead of rejecting the shape -- and for
    ``[1, 1, 320, 128]`` those are (1, 1), the very value the fallback produces,
    so that row alone cannot see the difference.  ``[8, 4, 320, 128]`` can.
    Paired with the test above, which proves the stored-verbatim path is
    reachable: without it, a constructor that ALWAYS returned (1, 1) would
    satisfy every row here.
    """
    mc = create_sharded_memory_config_(
        shape, _core_grid(8, 4), TensorMemoryLayout.BLOCK_SHARDED, ROW, True)
    assert mc.shard_spec.shape == (1, 1), why


@pytest.mark.unit
def test_underscore_variant_normalises_a_list_grid_to_a_tuple() -> None:
    """A list grid is unhashable and compares unequal to the tuple form, which
    matters because ``ShardSpec.__eq__`` compares grids and ``MemoryConfig.__eq__``
    compares specs."""
    mc = create_sharded_memory_config_(
        [320, 128], [8, 4], TensorMemoryLayout.BLOCK_SHARDED, ROW, True)
    assert mc.shard_spec.grid == (8, 4)
    assert isinstance(mc.shard_spec.grid, tuple)


@pytest.mark.unit
def test_underscore_variant_maps_a_foreign_orientation_by_name() -> None:
    """On the hardware path the orientation is a real ttnn enum, not the shim's.
    It is converted by ``.name`` so COL_MAJOR survives the crossing; anything with
    no usable name defaults to ROW_MAJOR."""
    class _ForeignColMajor:
        name = 'COL_MAJOR'

    grid = _core_grid(8, 4)
    foreign = create_sharded_memory_config_(
        [320, 128], grid, TensorMemoryLayout.BLOCK_SHARDED, _ForeignColMajor(), True)
    native = create_sharded_memory_config_(
        [320, 128], grid, TensorMemoryLayout.BLOCK_SHARDED, COL, True)
    unusable = create_sharded_memory_config_(
        [320, 128], grid, TensorMemoryLayout.BLOCK_SHARDED, 'zzz', True)
    assert foreign.shard_spec.orientation == ShardOrientation.COL_MAJOR
    assert native.shard_spec.orientation == ShardOrientation.COL_MAJOR
    assert unusable.shard_spec.orientation == ShardOrientation.ROW_MAJOR


# ===========================================================================
# What the spec is actually FOR -- the consumers that read it back
# ===========================================================================

@pytest.mark.unit
def test_two_configs_differing_only_in_shard_shape_compare_unequal() -> None:
    """The reason the spec exists.  ResNet-50's bottleneck does::

        if ds_out.memory_config() != out.memory_config():   # canonical C:325-326
            ds_out = ttnn.to_memory_config(ds_out, out.memory_config())

    With ``shard_spec=None`` on both sides these compared equal and the Reshard
    was never emitted -- polaris showed 3 against the capture's 9.  Same layout,
    same buffer type, different per-core extent must be unequal.
    """
    grid = _core_grid(8, 4)
    a = create_sharded_memory_config((1, 1, 1024, 256), grid, ShardStrategy.BLOCK, ROW)
    b = create_sharded_memory_config((1, 1, 2048, 256), grid, ShardStrategy.BLOCK, ROW)
    same = create_sharded_memory_config((1, 1, 1024, 256), grid, ShardStrategy.BLOCK, ROW)
    assert a.memory_layout == b.memory_layout and a.buffer_type == b.buffer_type
    assert a != b
    assert a == same


@pytest.mark.unit
def test_the_shard_width_a_conv_would_read_is_tile_aligned_not_one() -> None:
    """``needs_shard_or_reshard`` computes ``shard_w % alignment``.  Under the old
    ``(1, 1)`` placeholder that was ``1 % 32`` -- non-zero for every conv, so the
    predicate returned True unconditionally and the conv's output sharding was
    decided by a constant.  Checked end-to-end through the real predicate rather
    than by asserting on the number, so the test tracks the consumer.
    """
    from ttsim.front.ttnn.conv_sharding import needs_shard_or_reshard
    from ttsim.front.ttnn.tensor import Layout

    good = create_sharded_memory_config_(
        [320, 128], _core_grid(8, 4), TensorMemoryLayout.BLOCK_SHARDED, ROW, True)
    placeholder = create_sharded_memory_config_(
        [1, 1, 320, 128], _core_grid(8, 4), TensorMemoryLayout.BLOCK_SHARDED, ROW, True)
    kw: dict[str, Any] = dict(in_channels=1024, is_mm_conv=False, input_on_device=True,
              input_tensor_layout=Layout.TILE_LAYOUT)
    assert good.shard_spec.shape[1] % 32 == 0
    assert needs_shard_or_reshard(good, **kw) is False
    assert placeholder.shard_spec.shape[1] % 32 != 0
    assert needs_shard_or_reshard(placeholder, **kw) is True


@pytest.mark.unit
def test_an_unsharded_config_is_still_reported_as_unsharded() -> None:
    """``is_sharded()`` keys off the layout, not the spec, so the degraded paths
    above must not start claiming to be sharded just because a spec is absent --
    nor the reverse."""
    interleaved = MemoryConfig(TensorMemoryLayout.INTERLEAVED, BufferType.DRAM)
    degraded = create_sharded_memory_config(
        None, _core_grid(8, 4), ShardStrategy.BLOCK, ROW)
    assert interleaved.is_sharded() is False
    assert degraded.is_sharded() is True
    assert degraded.shard_spec is None


# ===========================================================================
# BLOCK_SHARDED is the only orientation-dependent strategy
# ===========================================================================
#
# BLOCK splits across BOTH grid axes, so which dim takes which axis follows the
# traversal order: real ttnn uses (height // grid.y, width // grid.x) for
# ROW_MAJOR and swaps to (height // grid.x, width // grid.y) for COL_MAJOR.
# The builder ignored orientation entirely and always used the ROW_MAJOR form,
# then stored the caller's orientation verbatim -- producing a spec whose shape
# and orientation contradicted each other.  HEIGHT and WIDTH divide by the total
# core count, which is orientation-invariant, so they must NOT move.
#
# This went unnoticed because no test in this file parametrised the orientation.

@pytest.mark.unit
@pytest.mark.parametrize(
    'strategy, orientation, expected_shard',
    [
        # grid 8x4 -> gx=8, gy=4, total=32; tensor (1, 1, 1024, 256) -> sh=1024, sw=256
        (ShardStrategy.BLOCK,  ROW, (256, 32)),   # 1024//4, 256//8
        (ShardStrategy.BLOCK,  COL, (128, 64)),   # 1024//8, 256//4   <- was (256, 32)
        (ShardStrategy.HEIGHT, ROW, (32, 256)),   # 1024//32, 256
        (ShardStrategy.HEIGHT, COL, (32, 256)),
        (ShardStrategy.WIDTH,  ROW, (1024, 8)),   # 1024, 256//32
        (ShardStrategy.WIDTH,  COL, (1024, 8)),
    ],
    ids=['block-row', 'block-col', 'height-row', 'height-col', 'width-row', 'width-col'],
)
def test_orientation_swaps_the_grid_axes_for_block_only(
        strategy, orientation, expected_shard) -> None:
    mc = create_sharded_memory_config(
        (1, 1, 1024, 256), _core_grid(8, 4), strategy, orientation)
    assert mc.shard_spec is not None
    assert mc.shard_spec.shape == expected_shard
    # The stored orientation must agree with the shape it was used to compute.
    assert mc.shard_spec.orientation == orientation


@pytest.mark.unit
def test_block_orientations_produce_configs_that_compare_unequal() -> None:
    """The consequence, not just the arithmetic.

    Two BLOCK configs differing only in orientation describe genuinely
    different core->shard mappings, so ``MemoryConfig.__eq__`` must separate
    them.  While the shape ignored orientation they compared EQUAL on shape and
    differed only on the orientation field -- and any consumer reading the
    shape alone (``needs_shard_or_reshard``, ``get_input_channels_alignment``)
    saw two identical shards.
    """
    grid = _core_grid(8, 4)
    row = create_sharded_memory_config((1, 1, 1024, 256), grid, ShardStrategy.BLOCK, ROW)
    col = create_sharded_memory_config((1, 1, 1024, 256), grid, ShardStrategy.BLOCK, COL)
    assert row.shard_spec.shape != col.shard_spec.shape
    assert row != col


@pytest.mark.unit
@pytest.mark.parametrize('orientation', [ROW, COL], ids=['row', 'col'])
def test_a_square_grid_hides_the_orientation_bug(orientation) -> None:
    """Positive control for the tests above.

    On a SQUARE grid the two formulas coincide, so a square fixture cannot tell
    a correct implementation from the broken one.  Pinned so nobody 'simplifies'
    the 8x4 grid above to 8x8 and quietly makes those tests vacuous.
    """
    mc = create_sharded_memory_config(
        (1, 1, 1024, 256), _core_grid(8, 8), ShardStrategy.BLOCK, orientation)
    assert mc.shard_spec.shape == (128, 32)


# ===========================================================================
# ShardSpec.shape is normalised, so spelling does not affect equality
# ===========================================================================

@pytest.mark.unit
def test_a_list_built_spec_equals_a_tuple_built_one() -> None:
    """``ShardSpec.__eq__`` compares ``shape`` with ``==``, and
    ``(448, 64) != [448, 64]``.  Both spellings are live in-tree: the
    conv_sharding / memory builders pass tuples, ViT passes
    ``[N * H * W // n_cores, C]`` at six call sites.  Without normalisation
    that is a second route to the false inequality ``CoreRangeSet.__eq__`` was
    added to close, since ``MemoryConfig.__eq__`` compares ``shard_spec``.
    """
    from ttsim.front.ttnn.buffer import ShardSpec

    grid = _core_grid(8, 4)
    as_tuple = ShardSpec(grid, (448, 64))
    as_list = ShardSpec(grid, [448, 64])
    assert as_tuple.shape == as_list.shape == (448, 64)
    assert as_tuple == as_list
    assert MemoryConfig(TensorMemoryLayout.BLOCK_SHARDED, BufferType.L1, as_tuple) == \
        MemoryConfig(TensorMemoryLayout.BLOCK_SHARDED, BufferType.L1, as_list)
    # Indexing still works, and a None shape is still tolerated.
    assert as_list.shape[1] == 64
    assert ShardSpec(grid, None).shape is None


# ===========================================================================
# an OUTPUT config pads channels by TILE_WIDTH, whichever path builds it
# ===========================================================================
#
# ``_apply_conv_output_memory_config`` (ttsim/front/ttnn/op.py) has two ways to
# give a conv output its config:
#
#   inherit  -- _rebuild_shard_shape_for_output: keep the input's grid/scheme/
#               orientation, recompute only the shard shape
#   reshard  -- derive a fresh parallel config for the requested shard_layout
#
# Both end at create_sharded_memory_config_from_parallel_config, where the
# ``input_channels_alignment`` argument is the channel padding GRANULARITY.  cpp
# uses two different ones and so must this port:
#
#   input  config  round_up(in_channels,  num_cores_c * input_channels_alignment)
#                  (tt-metal/ttnn/cpp/ttnn/operations/conv/conv2d/
#                   conv2d_utils.cpp:641-642)
#   output config  round_up(out_channels, num_cores_channels * TILE_WIDTH)
#                  (same file, :1286-1287)
#
# So BOTH paths above build an output and both pass TILE_WIDTH; only the input
# reshard target passes the value ``get_input_channels_alignment`` computes.  An
# earlier revision had the output paths pass that value too, which is correct
# only while it happens to equal 32.  Since MemoryConfig.__eq__ compares the
# shard shape, a wrong granularity is exactly the condition
# ``ds_out.memory_config() != out.memory_config()`` tests.

@pytest.mark.unit
@pytest.mark.parametrize(
    'channels, num_cores_channels, alignment_a, alignment_b, expect_differ',
    [
        # The reviewer's worked case: out_c=512 over 7 channel cores.
        #   align 16 -> ceil((512/16)/7)*16 = ceil(32/7)*16 = 5*16 = 80
        #   align 32 -> ceil((512/32)/7)*32 = ceil(16/7)*32 = 3*32 = 96
        (512, 7, 16, 32, True),
        # A case where the two alignments coincide, so the test above is not
        # merely asserting "any two alignments differ".
        (512, 8, 16, 32, False),
    ],
    ids=['512ch-7cores-differs', '512ch-8cores-coincides'],
)
def test_channel_alignment_changes_the_shard_width(
        channels, num_cores_channels, alignment_a, alignment_b, expect_differ) -> None:
    """The arithmetic the two paths can disagree on, in isolation."""
    import math

    def _channel_shard(ch, cores, align):
        return math.ceil((ch / align) / cores) * align

    a = _channel_shard(channels, num_cores_channels, alignment_a)
    b = _channel_shard(channels, num_cores_channels, alignment_b)
    assert (a != b) is expect_differ, (
        f'{channels}ch over {num_cores_channels} channel cores: '
        f'align {alignment_a} -> {a}, align {alignment_b} -> {b}'
    )


@pytest.mark.unit
def test_inherit_path_pads_output_channels_by_tile_width() -> None:
    """End to end: the inherit path pads output channels by TILE_WIDTH.

    ``_rebuild_shard_shape_for_output`` builds an OUTPUT config, so per
    conv2d_utils.cpp:1286-1287 its granularity is TILE_WIDTH -- not the
    input-channel alignment, and not the constructor's default.

    This fixture cannot tell those apart on its own: the input config is BLOCK
    over 8 channel cores with 512 channels, so its shard width is 64, and
    ``get_input_channels_alignment`` returns TILE_WIDTH for it anyway.  That is
    deliberate -- it keeps this test about the end-to-end plumbing (does the
    inherit path reach the constructor at all, with the grid and scheme it
    inherited) and leaves the granularity arithmetic to
    test_channel_pad_granularity_is_the_callers_choice in
    test_conv_sharding_units.py, which parametrizes the cases where 32, 16 and 8
    give different answers.
    """
    from ttsim.front.ttnn.conv_sharding import (
        create_sharded_memory_config_from_parallel_config,
        get_input_channels_alignment,
    )
    from ttsim.front.ttnn.op import _rebuild_shard_shape_for_output
    from ttsim.front.ttnn.device import ARCH, Device
    from ttsim.front.ttnn.tensor import DataType, Layout, Tensor

    # Device grid left at its 0x0 default ON PURPOSE.  With a real grid,
    # _rebuild_shard_shape_for_output also runs determine_output_parallel_config
    # and rewrites pcfg.grid, so a mismatch below could come from the grid
    # rather than the alignment.  Skipping that branch isolates what this test
    # is about.
    device = Device(device_id=0)
    device.architecture = ARCH.WORMHOLE_B0

    grid = _core_grid(8, 7)
    in_mc = create_sharded_memory_config(
        (1, 1, 3136, 512), grid, ShardStrategy.BLOCK, ROW)

    x = Tensor(name='x', shape=[1, 512, 56, 56], dtype=DataType.BFLOAT16,
               layout=Layout.ROW_MAJOR_LAYOUT, device=device)
    x._memory_config = in_mc
    result = Tensor(name='y', shape=[1, 512, 56, 56], dtype=DataType.BFLOAT16,
                    layout=Layout.ROW_MAJOR_LAYOUT, device=device)

    kwargs: dict[str, Any] = {
        'input_tensor': x, 'batch_size': 1, 'out_channels': 512,
        'input_height': 56, 'input_width': 56,
        'kernel_size': (3, 3), 'stride': (1, 1), 'padding': (1, 1), 'dilation': (1, 1),
    }
    inherited = _rebuild_shard_shape_for_output(in_mc, kwargs, result)
    assert inherited is not None, 'inherit path bailed out; the test proves nothing'

    # What cpp's output formula gives for the same conv.
    from ttsim.front.ttnn.conv_sharding import TILE_HEIGHT, TILE_WIDTH, ParallelConfig
    pcfg = ParallelConfig(in_mc.shard_spec.grid, in_mc.memory_layout,
                          in_mc.shard_spec.orientation)
    expected = create_sharded_memory_config_from_parallel_config(
        1 * 56 * 56, 512, pcfg, tile_size=TILE_HEIGHT,
        input_channels_alignment=TILE_WIDTH)

    assert inherited.shard_spec is not None
    assert inherited.shard_spec.shape == expected.shard_spec.shape, (
        f'inherit path built {inherited.shard_spec.shape}, '
        f'cpp output padding gives {expected.shard_spec.shape}'
    )

    # Pin the fixture's own premise, so a future change to
    # get_input_channels_alignment that made these differ shows up here rather
    # than silently turning the assertion above into a tautology.
    assert get_input_channels_alignment(
        in_mc.memory_layout, x.get_layout(), False, False, in_mc) == TILE_WIDTH
