# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Unit coverage for ``ttsim/front/ttnn/conv_sharding.py``.

conv_sharding is 407 new lines and 12 functions ported from
``ttnn/cpp/ttnn/operations/conv/conv2d/conv2d_utils.cpp``. Before this file only
``determine_parallel_config`` was referenced by any test, and only indirectly
through ``test_conv_input_reshard_gate``. That made it the largest untested
surface in the ResNet-50 port.

These functions do not change tensor shapes, which is what makes a regression
here quiet: a wrong answer changes whether ``needs_shard_or_reshard`` predicts a
reshard, which decides the conv's output sharding, which shifts the LUT key. The
failure mode is a silent drop in hit rate, not a crash.

``op_canonical``'s averagepool rename used to be covered here; it moved to
``tests/test_tools/test_op_canonical.py`` to mirror the module under test.
"""

from typing import Any

import pytest

from ttsim.front.ttnn.buffer import BufferType, ShardSpec, TensorMemoryLayout
from ttsim.front.ttnn.conv_sharding import (
    TILE_HEIGHT,
    TILE_WIDTH,
    ParallelConfig,
    ShardOrientation,
    _grid_size,
    create_sharded_memory_config_from_parallel_config,
    determine_output_parallel_config,
    determine_parallel_config,
    find_closest_largest_divisor,
    get_input_channels_alignment,
    get_num_cores_channels,
    get_num_cores_nhw,
    needs_shard_or_reshard,
)
from ttsim.front.ttnn.core import CoreCoord, CoreRange, CoreRangeSet
from ttsim.front.ttnn.memory import MemoryConfig
from ttsim.front.ttnn.tensor import Layout

#: Real logical tensix grids, from ``config/tt_bh.yaml:95`` and
#: ``config/tt_wh.yaml:88``.  Used instead of round numbers so the divisor walks
#: below exercise the same values ResNet-50 hits on silicon.
BH_P100A_GRID = (12, 10)
WH_N150_GRID = (8, 9)


def _grid(x: int, y: int) -> CoreRangeSet:
    return CoreRangeSet([CoreRange(CoreCoord(0, 0), CoreCoord(x - 1, y - 1))])


def _sharded(layout, shard_shape, orientation=ShardOrientation.ROW_MAJOR, grid=None):
    """An L1 MemoryConfig carrying a real shard spec."""
    spec = ShardSpec(grid if grid is not None else _grid(8, 8), shard_shape, orientation)
    return MemoryConfig(layout, BufferType.L1, spec)


# ---------------------------------------------------------------------------
# find_closest_largest_divisor — tt::find_closest_largest_divisor
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize(
    'num, start, expected',
    [
        (98, 98, 98),    # already a divisor
        (100, 98, 50),   # walks down to 50
        (98, 12, 7),     # 12..8 do not divide 98; 7 does
        (64, 8, 8),
        (97, 12, 1),     # prime: falls all the way to 1
    ],
)
def test_find_closest_largest_divisor(num, start, expected) -> None:
    """Largest d <= start dividing num. Never returns 0 — the loop stops at 1."""
    assert find_closest_largest_divisor(num, start) == expected


@pytest.mark.unit
def test_divisor_never_returns_zero() -> None:
    """A zero would be a division-by-zero further down determine_parallel_config."""
    for num in range(1, 200):
        assert find_closest_largest_divisor(num, 12) >= 1


@pytest.mark.unit
@pytest.mark.parametrize(
    'num, start, num2, expected',
    [
        # (out_channels_ntiles, start_divisor_c, input_channels_blocks)
        (8, 12, 2, 2),    # ResNet-50 64->256 expansion: one-count gives 8
        (16, 12, 4, 4),   # 128->512:                    one-count gives 8
        (32, 12, 8, 8),   # 256->1024: both forms agree
        (9, 12, 32, 1),   # no common divisor above 1
        (2, 12, 8, 2),    # num already divides both
    ],
)
def test_two_count_form_requires_a_common_divisor(num, start, num2, expected) -> None:
    """The second numerator is load-bearing, not decoration.

    ``tt-metal/ttnn/cpp/ttnn/operations/conv/conv2d/conv2d_utils.cpp:62`` is a
    separate overload whose loop is ``num1 % d != 0 or num2 % d != 0``, and
    cpp:214 calls THAT one on the BLOCK/padding-disabled path.  A common divisor
    is <= either individual divisor, so omitting num2 can only over-estimate
    num_cores_c -- which lands in the shard shape and therefore the LUT key.

    The rows above are ResNet-50's own (out_channels_ntiles,
    input_channels_blocks) pairs at 32-channel alignment: 7 of its 32
    layout/grid combinations diverge between the two forms, so this is not a
    latent difference that only exotic channel counts reach.
    """
    assert find_closest_largest_divisor(num, start, num2=num2) == expected
    # And the one-count form is unchanged by the new parameter's existence.
    assert find_closest_largest_divisor(num, start) >= expected


# ---------------------------------------------------------------------------
# get_num_cores_nhw / get_num_cores_channels — cpp:302 and cpp:331
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_num_cores_nhw_by_layout() -> None:
    """HEIGHT spreads NHW over every core; WIDTH puts it on one; BLOCK on a row."""
    grid = _grid(8, 8)
    o = ShardOrientation.ROW_MAJOR
    assert get_num_cores_nhw(grid, TensorMemoryLayout.HEIGHT_SHARDED, o) == 64
    assert get_num_cores_nhw(grid, TensorMemoryLayout.WIDTH_SHARDED, o) == 1
    assert get_num_cores_nhw(grid, TensorMemoryLayout.BLOCK_SHARDED, o) == 8


@pytest.mark.unit
def test_num_cores_channels_by_layout() -> None:
    """The mirror image of the above — channels is the other axis."""
    grid = _grid(8, 8)
    o = ShardOrientation.ROW_MAJOR
    assert get_num_cores_channels(grid, TensorMemoryLayout.HEIGHT_SHARDED, o) == 1
    assert get_num_cores_channels(grid, TensorMemoryLayout.WIDTH_SHARDED, o) == 64
    assert get_num_cores_channels(grid, TensorMemoryLayout.BLOCK_SHARDED, o) == 8


@pytest.mark.unit
def test_nhw_times_channels_covers_block_grid() -> None:
    """For BLOCK the two factors must multiply back to the grid's core count."""
    for gx, gy in ((8, 8), (12, 10), (8, 9)):
        grid = _grid(gx, gy)
        o = ShardOrientation.ROW_MAJOR
        nhw = get_num_cores_nhw(grid, TensorMemoryLayout.BLOCK_SHARDED, o)
        ch = get_num_cores_channels(grid, TensorMemoryLayout.BLOCK_SHARDED, o)
        assert nhw * ch == gx * gy, (gx, gy, nhw, ch)


# ---------------------------------------------------------------------------
# get_input_channels_alignment — cpp:92
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_alignment_mm_conv_is_tile_width() -> None:
    """A matmul-lowered conv always aligns to TILE_WIDTH, whatever the layout."""
    assert get_input_channels_alignment(
        TensorMemoryLayout.HEIGHT_SHARDED, Layout.ROW_MAJOR_LAYOUT,
        False, True, None) == TILE_WIDTH


@pytest.mark.unit
def test_alignment_width_sharded_is_tile_width() -> None:
    assert get_input_channels_alignment(
        TensorMemoryLayout.WIDTH_SHARDED, Layout.ROW_MAJOR_LAYOUT,
        False, False, None) == TILE_WIDTH


@pytest.mark.unit
def test_alignment_tile_input_falls_through_to_tile_width() -> None:
    """The regression this port fixed: an earlier version returned a constant 16
    for the non-width, non-mm case, missing that the rule applies only to a
    ROW_MAJOR input. A TILE input must fall through to TILE_WIDTH."""
    assert get_input_channels_alignment(
        TensorMemoryLayout.HEIGHT_SHARDED, Layout.TILE_LAYOUT,
        False, False, None) == TILE_WIDTH


@pytest.mark.unit
def test_alignment_row_major_unsharded_is_not_tile_width() -> None:
    """A ROW_MAJOR interleaved input is the case the cpp rule actually targets."""
    got = get_input_channels_alignment(
        TensorMemoryLayout.INTERLEAVED, Layout.ROW_MAJOR_LAYOUT,
        False, False, None)
    assert got != TILE_WIDTH and got >= 1, got


@pytest.mark.unit
def test_alignment_accepts_none_layout() -> None:
    """Callers pass ``getattr(t, 'layout', None)``; None must not raise."""
    assert get_input_channels_alignment(None, None, False, False, None) >= 1


@pytest.mark.unit
@pytest.mark.parametrize(
    'shard_width, expected',
    [
        (64, TILE_WIDTH),   # 64 % 32 == 0
        (32, TILE_WIDTH),   # exactly one tile
        (48, 16),           # 48 % 32 == 16, but 48 % 16 == 0
        (24, 8),            # 24 % 16 == 8, but 24 % 8 == 0
        (20, TILE_WIDTH),   # divides by none of 32/16/8 -> the cpp fallthrough
    ],
)
def test_alignment_derived_from_shard_width(shard_width, expected) -> None:
    """cpp:92 — when the input is already sharded the alignment comes from the
    shard width itself (32 / 16 / 8), not from a constant.

    The 20 case is the one worth keeping: a width divisible by none of the three
    falls through to TILE_WIDTH rather than to the L1 element alignment, and an
    implementation that ended the chain with ``return 8`` would pass every other
    row here.
    """
    mc = _sharded(TensorMemoryLayout.HEIGHT_SHARDED, (32, shard_width))
    got = get_input_channels_alignment(
        TensorMemoryLayout.HEIGHT_SHARDED, Layout.ROW_MAJOR_LAYOUT, False, False, mc)
    assert got == expected, (shard_width, got, expected)


@pytest.mark.unit
def test_alignment_sharded_without_spec_uses_l1_element_alignment() -> None:
    """Sharded but spec-less: there is no shard width to read, so the rule falls
    back to the 8-element L1 stick.  Paired with the test above so that a change
    collapsing the two paths into one constant fails on one side or the other."""
    mc = MemoryConfig(TensorMemoryLayout.HEIGHT_SHARDED, BufferType.L1, None)
    spec_less = get_input_channels_alignment(
        TensorMemoryLayout.HEIGHT_SHARDED, Layout.ROW_MAJOR_LAYOUT, False, False, mc)
    with_spec = get_input_channels_alignment(
        TensorMemoryLayout.HEIGHT_SHARDED, Layout.ROW_MAJOR_LAYOUT, False, False,
        _sharded(TensorMemoryLayout.HEIGHT_SHARDED, (32, 64)))
    assert spec_less == 8, spec_less
    assert with_spec == TILE_WIDTH, with_spec


@pytest.mark.unit
def test_alignment_sliced_op_enters_the_rule_without_row_major() -> None:
    """``is_row_major or sliced_op`` — a TILE input normally falls straight through
    to TILE_WIDTH (pinned above); ``sliced_op=True`` must let it in instead."""
    tile_plain = get_input_channels_alignment(
        TensorMemoryLayout.HEIGHT_SHARDED, Layout.TILE_LAYOUT, False, False, None)
    tile_sliced = get_input_channels_alignment(
        TensorMemoryLayout.HEIGHT_SHARDED, Layout.TILE_LAYOUT, True, False, None)
    assert tile_plain == TILE_WIDTH
    assert tile_sliced == 8, tile_sliced


# ---------------------------------------------------------------------------
# _grid_size — the empty-CoreRangeSet guard
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_grid_size_of_empty_corerangeset_is_zero() -> None:
    """``CoreRangeSet.bounding_box()`` returns None for an empty set (core.py:151).

    Without the guard this raises AttributeError on ``bb.end_coord`` deep inside
    get_num_cores_nhw, where the traceback says nothing useful.
    """
    assert _grid_size(CoreRangeSet([])) == (0, 0)
    assert _grid_size(_grid(12, 10)) == (12, 10)


# ---------------------------------------------------------------------------
# determine_parallel_config — the BLOCK and WIDTH branches (cpp:169)
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_block_sharded_picks_resnet50_layer3_channel_cores() -> None:
    """The one exactly-known value in this file.

    ResNet-50 layer 3 on BH p100a: 1024 input channels, 32-byte alignment, a
    12x10 grid.  tt-metal's divisor walk lands on 11 channel cores and pads 32
    blocks to 33 -- 1056 channels.  1056 is the number the silicon capture
    records for those keys, which is why the padding pass exists at all
    (ttsim/back/device.py:621, and this module's own docstring).
    """
    pcfg = determine_parallel_config(
        TensorMemoryLayout.BLOCK_SHARDED,
        batch_size=16, input_channels=1024, output_height=14, output_width=14,
        output_channels=256, input_channels_alignment=TILE_WIDTH,
        compute_grid_size=BH_P100A_GRID, block_shard_orientation=ShardOrientation.ROW_MAJOR,
        enable_channels_padding=True)
    assert get_num_cores_channels(
        pcfg.grid, pcfg.shard_scheme, pcfg.shard_orientation) == 11
    assert get_num_cores_nhw(
        pcfg.grid, pcfg.shard_scheme, pcfg.shard_orientation) == 10
    assert pcfg.grid.num_cores() == 110


@pytest.mark.unit
@pytest.mark.parametrize(
    'grid_size, orientation, expected_grid',
    [
        # Traced by hand through cpp:188-204 with in_channels=1024 (=32 blocks at
        # 32-byte alignment) and out_nhw_ntiles = ceil(16*14*14 / 32) = 98.
        #   BH ROW_MAJOR: nhw walks from gy=10 -> 10 ; chan walks from gx=12 -> 11
        #   BH COL_MAJOR: nhw walks from gx=12 -> 11 ; chan walks from gy=10 ->  8
        #   WH ROW_MAJOR: nhw walks from gy= 9 ->  9 ; chan walks from gx= 8 ->  8
        #   WH COL_MAJOR: nhw walks from gx= 8 ->  8 ; chan walks from gy= 9 ->  8
        # and cpp:198-199 then puts nhw on x for COL_MAJOR, on y for ROW_MAJOR.
        (BH_P100A_GRID, ShardOrientation.ROW_MAJOR, (11, 10)),
        (BH_P100A_GRID, ShardOrientation.COL_MAJOR, (11, 8)),
        (WH_N150_GRID, ShardOrientation.ROW_MAJOR, (8, 9)),
        (WH_N150_GRID, ShardOrientation.COL_MAJOR, (8, 8)),
    ],
)
def test_block_sharded_places_cores_on_the_orientation_axis(
        grid_size, orientation, expected_grid) -> None:
    """cpp:196-200 assigns the nhw and channel counts to x or y according to the
    orientation.  The grids are pinned as (x, y) extents rather than as core
    totals because BH COL_MAJOR (11x8 = 88) and a transposed 8x11 would tie on
    any count-based assertion while meaning opposite things to the shard spec.

    The BH ROW_MAJOR row is also the layer-3 config the capture confirms.
    """
    pcfg = determine_parallel_config(
        TensorMemoryLayout.BLOCK_SHARDED,
        batch_size=16, input_channels=1024, output_height=14, output_width=14,
        output_channels=256, input_channels_alignment=TILE_WIDTH,
        compute_grid_size=grid_size, block_shard_orientation=orientation,
        enable_channels_padding=True)
    assert pcfg.shard_scheme == TensorMemoryLayout.BLOCK_SHARDED
    assert pcfg.shard_orientation == orientation
    assert _grid_size(pcfg.grid) == expected_grid
    assert pcfg.grid.num_cores() <= grid_size[0] * grid_size[1]


@pytest.mark.unit
def test_block_sharded_padding_flag_switches_which_channel_count_is_read() -> None:
    """``enable_channels_padding`` is not a rounding toggle, and the two branches
    do not even read the same NUMBER of inputs.

    Padded takes the padding-tolerant walk over *input* channel blocks alone
    (cpp:213).  Unpadded takes the TWO-count overload (cpp:214 -> cpp:62) and
    needs a divisor common to *output* channel tiles AND input channel blocks.
    ttsim/front/ttnn/op.py passes ``enable_channels_padding=not is_mm_conv``,
    so both are live.

    480 in-channels and 384 out-channels at 32-alignment give 15 blocks and 12
    tiles, chosen so that FOUR quantities are mutually distinct on a 12-wide
    grid -- which is what makes this test able to fail:

        padded walk over 15 blocks          -> 8   (the padded branch)
        common divisor of 12 and 15         -> 3   (the unpadded branch)
        one-count divisor of 12 tiles       -> 12  (the superseded port)
        one-count divisor of 15 blocks      -> 5   (reading input only)

    Every earlier revision of this test collapsed two of those together.  1024/
    1536 made the unpadded answer 8, equal to the input-only reading; 1024/2048
    made it 8 as well, equal to the padded answer.  Both passed against the
    one-count implementation this file now pins against.
    """
    common: dict[str, Any] = dict(
        batch_size=16, input_channels=480, output_height=14, output_width=14,
        output_channels=384, input_channels_alignment=TILE_WIDTH,
        compute_grid_size=BH_P100A_GRID, block_shard_orientation=ShardOrientation.ROW_MAJOR)
    padded = determine_parallel_config(
        TensorMemoryLayout.BLOCK_SHARDED, enable_channels_padding=True, **common)
    unpadded = determine_parallel_config(
        TensorMemoryLayout.BLOCK_SHARDED, enable_channels_padding=False, **common)
    ch_padded = get_num_cores_channels(
        padded.grid, padded.shard_scheme, padded.shard_orientation)
    ch_unpadded = get_num_cores_channels(
        unpadded.grid, unpadded.shard_scheme, unpadded.shard_orientation)
    assert (ch_padded, ch_unpadded) == (8, 3), (ch_padded, ch_unpadded)


@pytest.mark.unit
@pytest.mark.parametrize(
    'grid_size, override, expected_nhw',
    [
        # ResNet-50 conv1: folded 115x115, k=4 s=1 -> 112x112, batch 16, so
        # out_nhw_ntiles = 16*112*112/32 = 6272 on both arches.
        #
        # The override values are the ones ResNet-50 actually sets.  conv1's are
        # declared in tt-metal's models/demos/vision/classification/resnet50/
        # ttnn_resnet/tt/ttnn_functional_resnet50.py and mirrored in this repo's
        # workloads/ttnn/resnet50/common/ttnn_functional_resnet50.py, which is
        # also where the layer1/layer2 override lives.
        (WH_N150_GRID,  1568, 64),    # WH conv1.  mult = 49
        (WH_N150_GRID,     0, 72),    # the whole grid, if the override is dropped
        (BH_P100A_GRID, 1024, 98),    # BH p100a conv1 at batch 32.  mult = 32
        (BH_P100A_GRID,    0, 119),   # 21 cores more than the real answer
        (BH_P100A_GRID,  512, 98),    # the layer1/layer2 override (16 * 32)
    ],
)
def test_act_block_h_override_changes_the_nhw_core_count(
        grid_size, override, expected_nhw) -> None:
    """``act_block_h_override`` is a multiplier on the nhw divisor search, not a
    cosmetic hint.

    determine_parallel_config turns it into ``act_block_h_ntiles`` and passes
    that as the ``mult`` of find_closest_largest_divisor_with_num_padding_and_mult
    (cpp:200-201), so dropping it searches with mult=1 and returns a different
    core count.  All three cpp call sites pass it -- cpp:616, cpp:752 and
    cpp:1057 -- and the count propagates into the OUTPUT config too, because
    determine_output_parallel_config reads num_cores_nhw off the input config
    (cpp:258), which is what lands in the output shard shape.

    The zero rows are the regression this pins: an implementation that forgets
    to forward the override returns 72 instead of 64 on WH and 119 instead of 98
    on BH.  ResNet-50's end-to-end LUT hit counts do not move either way, so this
    is the only place the difference is observable.
    """
    pcfg = determine_parallel_config(
        TensorMemoryLayout.HEIGHT_SHARDED,
        batch_size=16, input_channels=16, output_height=112, output_width=112,
        output_channels=64, input_channels_alignment=TILE_WIDTH,
        compute_grid_size=grid_size,
        block_shard_orientation=ShardOrientation.ROW_MAJOR,
        enable_channels_padding=True,
        act_block_h_override=override)
    assert get_num_cores_nhw(
        pcfg.grid, pcfg.shard_scheme, pcfg.shard_orientation) == expected_nhw


@pytest.mark.unit
def test_width_sharded_spreads_channels_over_cores() -> None:
    """WIDTH puts every core on the channel axis, so num_cores_channels is the
    whole grid slice and num_cores_nhw is 1 (cpp:302/331)."""
    pcfg = determine_parallel_config(
        TensorMemoryLayout.WIDTH_SHARDED,
        batch_size=16, input_channels=1024, output_height=14, output_width=14,
        output_channels=1024, input_channels_alignment=TILE_WIDTH,
        compute_grid_size=BH_P100A_GRID, block_shard_orientation=ShardOrientation.ROW_MAJOR,
        enable_channels_padding=True)
    assert pcfg.shard_scheme == TensorMemoryLayout.WIDTH_SHARDED
    assert pcfg.shard_orientation == ShardOrientation.ROW_MAJOR
    assert get_num_cores_nhw(pcfg.grid, pcfg.shard_scheme, pcfg.shard_orientation) == 1
    assert get_num_cores_channels(
        pcfg.grid, pcfg.shard_scheme, pcfg.shard_orientation) == 32
    assert pcfg.grid.num_cores() == 32


@pytest.mark.unit
def test_non_conv_layout_raises_rather_than_guessing() -> None:
    """cpp supports Height, Block and Width only.  INTERLEAVED must raise, not
    fall through to an arbitrary grid -- a silent default here would produce a
    shard spec for an unsharded tensor and shift every downstream LUT key."""
    with pytest.raises(ValueError, match='Height, Block or Width'):
        determine_parallel_config(
            TensorMemoryLayout.INTERLEAVED,
            batch_size=16, input_channels=1024, output_height=14, output_width=14,
            output_channels=256, input_channels_alignment=TILE_WIDTH,
            compute_grid_size=BH_P100A_GRID, block_shard_orientation=ShardOrientation.ROW_MAJOR,
            enable_channels_padding=True)


# ---------------------------------------------------------------------------
# create_sharded_memory_config_from_parallel_config — cpp:357
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_channel_shard_pads_to_the_captured_1056() -> None:
    """The multi-channel-core branch, pinned against the capture.

    1024 channels / 16-byte alignment = 64 blocks; over 11 channel cores that is
    ceil(64/11) = 6 blocks each = 96 channels per core, so the tensor is padded
    to 11 * 96 = 1056.  Asserting 96 alone would not catch an off-by-one in the
    core count, so both are pinned.
    """
    pcfg = ParallelConfig(
        grid=_grid(11, 10),
        shard_scheme=TensorMemoryLayout.BLOCK_SHARDED,
        shard_orientation=ShardOrientation.ROW_MAJOR)
    mc = create_sharded_memory_config_from_parallel_config(
        nhw=16 * 14 * 14, channels=1024, pconfig=pcfg, tile_size=TILE_HEIGHT,
        input_channels_alignment=16)
    num_cores_channels = get_num_cores_channels(
        pcfg.grid, pcfg.shard_scheme, pcfg.shard_orientation)
    assert num_cores_channels == 11
    assert mc.shard_spec.shape[1] == 96, mc.shard_spec.shape
    assert num_cores_channels * mc.shard_spec.shape[1] == 1056


@pytest.mark.unit
@pytest.mark.parametrize(
    'channels, cores_c, granularity, expected_shard',
    [
        # cpp's OUTPUT formula is round_up(out_c, cores_c * TILE_WIDTH) / cores_c
        # (cpp:1286-1287).  These are the cases where the input-channel alignment
        # would give a different answer, so passing 16 or 8 here is observable.
        (512, 7, TILE_WIDTH, 96),   # round_up(512, 224)/7  = 96
        (512, 7, 16, 80),           # the superseded value at alignment 16
        (512, 7, 8, 80),
        (64, 7, TILE_WIDTH, 32),    # round_up(64, 224)/7   = 32
        (64, 7, 16, 16),
        # ...and a case where every granularity agrees, as a control: a test
        # using only these would pass against the wrong implementation.
        (1024, 11, TILE_WIDTH, 96),
        (1024, 11, 16, 96),
    ],
)
def test_channel_pad_granularity_is_the_callers_choice(
        channels, cores_c, granularity, expected_shard) -> None:
    """``input_channels_alignment`` is a padding GRANULARITY, and cpp uses two.

    An input config pads by ``num_cores_c * input_channels_alignment``
    (cpp:641-642); an output config pads by ``num_cores_channels * TILE_WIDTH``
    (cpp:1286-1287).  cpp pre-pads and then divides; this port folds the padding
    into create_sharded_memory_config_from_parallel_config, so the granularity
    arrives as this parameter and the caller must pick by direction.

    The first five rows are where it matters.  ResNet-50 cannot reach them: its
    conv inputs are TILE layout and its 1x1s are mm convs, both of which return
    TILE_WIDTH from get_input_channels_alignment, so alignment == 32 there and
    every granularity agrees.  A ROW_MAJOR non-mm input can yield 16 or 8.
    """
    pcfg = ParallelConfig(
        grid=_grid(cores_c, 10),
        shard_scheme=TensorMemoryLayout.BLOCK_SHARDED,
        shard_orientation=ShardOrientation.ROW_MAJOR)
    mc = create_sharded_memory_config_from_parallel_config(
        nhw=16 * 14 * 14, channels=channels, pconfig=pcfg, tile_size=TILE_HEIGHT,
        input_channels_alignment=granularity)
    assert mc.shard_spec.shape[1] == expected_shard, mc.shard_spec.shape


@pytest.mark.unit
def test_single_channel_core_takes_the_channels_verbatim() -> None:
    """The ``num_cores_channels <= 1`` branch must NOT pad.  Paired with the test
    above: an implementation that always padded would return 1024 -> 1024 here
    only by accident of 1024 already being aligned, so a non-aligned channel
    count is used instead."""
    pcfg = ParallelConfig(
        grid=_grid(8, 8),
        shard_scheme=TensorMemoryLayout.HEIGHT_SHARDED,
        shard_orientation=ShardOrientation.ROW_MAJOR)
    mc = create_sharded_memory_config_from_parallel_config(
        nhw=16 * 14 * 14, channels=1000, pconfig=pcfg, tile_size=TILE_HEIGHT,
        input_channels_alignment=16)
    assert get_num_cores_channels(
        pcfg.grid, pcfg.shard_scheme, pcfg.shard_orientation) == 1
    assert mc.shard_spec.shape[1] == 1000, mc.shard_spec.shape
    assert mc.memory_layout == TensorMemoryLayout.HEIGHT_SHARDED
    assert mc.buffer_type == BufferType.L1


@pytest.mark.unit
def test_width_sharded_does_not_pad_the_nhw_extent() -> None:
    """A WIDTH-sharded tensor keeps its whole NHW extent on one core, so there is
    no core count to round up against and cpp leaves the value alone.

    3000 is deliberately not a tile multiple: WIDTH must return it verbatim while
    HEIGHT, on the same grid, rounds 3000 up to 64 cores x 32 = 4096 and shards it
    to 64 per core.  Rounding WIDTH too would silently give 3008 -- a plausible
    number, a wrong shard spec, and a different LUT key.
    """
    def _shape(scheme):
        pcfg = ParallelConfig(
            grid=_grid(8, 8), shard_scheme=scheme,
            shard_orientation=ShardOrientation.ROW_MAJOR)
        return create_sharded_memory_config_from_parallel_config(
            nhw=3000, channels=1024, pconfig=pcfg, tile_size=TILE_HEIGHT,
            input_channels_alignment=16).shard_spec.shape

    assert _shape(TensorMemoryLayout.WIDTH_SHARDED) == (3000, 16)
    assert _shape(TensorMemoryLayout.HEIGHT_SHARDED) == (64, 1024)


# ---------------------------------------------------------------------------
# determine_output_parallel_config — cpp:239
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize(
    'layout',
    [TensorMemoryLayout.HEIGHT_SHARDED, TensorMemoryLayout.BLOCK_SHARDED,
     TensorMemoryLayout.WIDTH_SHARDED],
)
@pytest.mark.parametrize(
    'orientation', [ShardOrientation.ROW_MAJOR, ShardOrientation.COL_MAJOR])
@pytest.mark.parametrize('is_mm_conv', [False, True])
def test_output_config_inherits_scheme_and_orientation(
        layout, orientation, is_mm_conv) -> None:
    """The module docstring's central claim: only ``.grid`` is ever recomputed.

    If a future edit made the output adopt a requested layout instead of the
    input's, every conv output memory config would change and the LUT keys with
    them -- with no crash anywhere.  This is the assertion that would fail.
    """
    grid_before = _grid(11, 10)
    inp = ParallelConfig(
        grid=grid_before, shard_scheme=layout, shard_orientation=orientation)
    out = determine_output_parallel_config(
        inp, BH_P100A_GRID, 1024, inp.shard_orientation, is_mm_conv=is_mm_conv)
    assert out.shard_scheme == layout
    assert out.shard_orientation == orientation
    # cpp builds a fresh ParallelConfig; the caller's own config must survive the
    # call untouched, since ttsim/front/ttnn/op.py:2392-2393 reuses ``pcfg``
    # after asking for ``ocfg``.
    assert inp.grid is grid_before
    assert inp.shard_scheme == layout


@pytest.mark.unit
def test_output_config_matmul_conv_keeps_the_input_grid() -> None:
    """cpp:243 returns early for a matmul-lowered conv -- the grid is inherited
    verbatim, not recomputed.  Paired with the WIDTH test below, which shows the
    same input DOES get a new grid when is_mm_conv is False; without that pair
    this would also pass against a function that never recomputed anything."""
    inp = ParallelConfig(
        grid=_grid(11, 10), shard_scheme=TensorMemoryLayout.WIDTH_SHARDED,
        shard_orientation=ShardOrientation.ROW_MAJOR)
    as_mm = determine_output_parallel_config(
        inp, BH_P100A_GRID, 1024, inp.shard_orientation, is_mm_conv=True)
    as_conv = determine_output_parallel_config(
        inp, BH_P100A_GRID, 1024, inp.shard_orientation, is_mm_conv=False)
    assert as_mm.grid is inp.grid
    assert as_conv.grid.num_cores() == 32
    assert as_conv.grid.num_cores() != inp.grid.num_cores()


@pytest.mark.unit
def test_output_config_block_sharded_rewrites_only_the_channel_axis() -> None:
    """BLOCK output: the channel cores come from the OUT channel count, the nhw
    cores are carried over from the input grid (cpp:248-254).  1024 out-channels
    on a 12-wide grid give 11 channel cores; the input's 10 nhw cores survive."""
    inp = ParallelConfig(
        grid=_grid(11, 10), shard_scheme=TensorMemoryLayout.BLOCK_SHARDED,
        shard_orientation=ShardOrientation.ROW_MAJOR)
    out = determine_output_parallel_config(
        inp, BH_P100A_GRID, 1024, inp.shard_orientation, is_mm_conv=False)
    assert get_num_cores_channels(
        out.grid, out.shard_scheme, out.shard_orientation) == 11
    assert get_num_cores_nhw(out.grid, out.shard_scheme, out.shard_orientation) == 10


@pytest.mark.unit
@pytest.mark.parametrize(
    'orientation, expected_grid, expected_nhw, expected_chan',
    [
        # Input grid 11x10 BLOCK, out_channels=1024 (32 tiles), compute grid 12x10.
        # ROW_MAJOR: channel divisor starts at gx=12 -> 11 cores; nhw read off the
        #            input grid's y = 10.  Laid out x=chan, y=nhw -> (11, 10).
        # COL_MAJOR: channel divisor starts at gy=10 -> 8 cores; nhw read off the
        #            input grid's x = 11.  Laid out x=nhw, y=chan -> (11, 8).
        (ShardOrientation.ROW_MAJOR, (11, 10), 10, 11),
        (ShardOrientation.COL_MAJOR, (11, 8), 11, 8),
    ],
)
def test_output_config_lays_the_grid_out_along_the_given_orientation(
        orientation, expected_grid, expected_nhw, expected_chan) -> None:
    """cpp:256-262 puts the channel and nhw counts on different axes per
    orientation, and picks a different starting divisor for the channel walk.

    THE REGRESSION.  ``block_shard_orientation`` used to default to ROW_MAJOR and
    both ttsim/front/ttnn/op.py call sites omitted it, so a COL_MAJOR input
    produced a config
    labelled COL_MAJOR whose grid had been laid out ROW_MAJOR -- (11, 11) with
    chan=11, where COL_MAJOR should give (11, 8) with chan=8.  Nothing crashed;
    get_num_cores_channels simply read the wrong extent and the shard width, and
    with it the LUT key, was wrong.  cpp:239-244 makes the argument mandatory,
    which is why the bug cannot be written there; the default is now gone here
    too, and cpp:833 / cpp:1077 show both callers passing the input's own
    orientation.

    Asserted as an (x, y) extent rather than a core count: (11, 8) and a
    transposed (8, 11) both total 88 and mean opposite things to a shard spec.
    """
    inp = ParallelConfig(
        grid=_grid(11, 10), shard_scheme=TensorMemoryLayout.BLOCK_SHARDED,
        shard_orientation=orientation)
    out = determine_output_parallel_config(
        inp, BH_P100A_GRID, 1024, inp.shard_orientation, is_mm_conv=False)
    assert _grid_size(out.grid) == expected_grid
    assert get_num_cores_nhw(
        out.grid, out.shard_scheme, out.shard_orientation) == expected_nhw
    assert get_num_cores_channels(
        out.grid, out.shard_scheme, out.shard_orientation) == expected_chan


@pytest.mark.unit
def test_output_config_requires_an_explicit_orientation() -> None:
    """Both trailing arguments are mandatory, mirroring the declaration at
    ``tt-metal/ttnn/cpp/ttnn/operations/conv/conv2d/conv2d_utils.hpp:98-102``, which gives neither a default.

    ``block_shard_orientation`` is the one with teeth: with a default, a caller
    that forgets it gets a silently ROW_MAJOR grid on a COL_MAJOR config, which
    picks the wrong grid axis and shifts every downstream LUT key.  ``is_mm_conv``
    reaches the same divisor walk one level up --
    ``tt-metal/ttnn/cpp/ttnn/operations/conv/conv2d/conv2d_utils.cpp:626`` passes
    ``enable_channels_padding && !is_mm_conv`` into determine_parallel_config,
    where ``cpp:212`` and ``cpp:222`` select between the padded and unpadded
    divisor searches -- so a default there silently picks one for a caller that
    did not choose.

    A TypeError is the correct outcome for either, and this pins it so nobody
    reinstates the convenience.  The ``match`` admits either name because Python
    reports whichever required argument it notices first, and that is not a fact
    this test should depend on -- the point is that the call is refused, not
    which parameter the message happens to blame.
    """
    inp = ParallelConfig(
        grid=_grid(11, 10), shard_scheme=TensorMemoryLayout.BLOCK_SHARDED,
        shard_orientation=ShardOrientation.COL_MAJOR)
    with pytest.raises(TypeError, match='block_shard_orientation|is_mm_conv'):
        determine_output_parallel_config(inp, BH_P100A_GRID, 1024)  # type: ignore[call-arg]


@pytest.mark.unit
def test_output_config_height_sharded_grid_is_untouched() -> None:
    """HEIGHT matches neither elif, so the grid falls through unchanged.  Stated
    as an identity check rather than a core count, because a recomputed grid of
    the same size would satisfy the weaker assertion."""
    inp = ParallelConfig(
        grid=_grid(11, 10), shard_scheme=TensorMemoryLayout.HEIGHT_SHARDED,
        shard_orientation=ShardOrientation.ROW_MAJOR)
    out = determine_output_parallel_config(
        inp, BH_P100A_GRID, 1024, inp.shard_orientation, is_mm_conv=False)
    assert out.grid is inp.grid


# ---------------------------------------------------------------------------
# needs_shard_or_reshard — cpp:686-745
# ---------------------------------------------------------------------------
#
# Every clause below returns True.  A function body of ``return True`` would
# therefore satisfy all of them, which is why the first test here establishes the
# False case: it is the positive control for the whole section, and each clause
# test is a single-field mutation away from it.

_RESHARD_BASE: dict[str, Any] = dict(
    in_channels=256, is_mm_conv=False,
    input_on_device=True, input_tensor_layout=Layout.TILE_LAYOUT)


@pytest.mark.unit
def test_no_reshard_when_the_input_is_already_well_formed() -> None:
    """THE POSITIVE CONTROL.  A tile-aligned, ROW_MAJOR, height-sharded input on
    device needs nothing done to it.  If this ever starts returning True, every
    other test in this section has stopped proving anything."""
    mc = _sharded(TensorMemoryLayout.HEIGHT_SHARDED, (32, 32))
    assert needs_shard_or_reshard(mc, **_RESHARD_BASE) is False


@pytest.mark.unit
def test_reshard_when_input_is_not_on_device() -> None:
    mc = _sharded(TensorMemoryLayout.HEIGHT_SHARDED, (32, 32))
    assert needs_shard_or_reshard(
        mc, **{**_RESHARD_BASE, 'input_on_device': False}) is True


@pytest.mark.unit
def test_reshard_when_memory_config_is_absent() -> None:
    assert needs_shard_or_reshard(None, **_RESHARD_BASE) is True


@pytest.mark.unit
def test_reshard_when_input_is_interleaved() -> None:
    """An interleaved input is by definition not sharded, so the conv must shard it."""
    assert needs_shard_or_reshard(
        MemoryConfig(TensorMemoryLayout.INTERLEAVED, BufferType.DRAM),
        **_RESHARD_BASE) is True


@pytest.mark.unit
@pytest.mark.parametrize('spec', [None, ShardSpec(_grid(8, 8), None)])
def test_reshard_when_the_shard_shape_is_unknown(spec) -> None:
    """The regression this guard exists for: sharded-but-shapeless configs cannot
    be evaluated against cpp:686-745, and answering "no reshard" suppressed six
    of ResNet-50's nine Reshards.  Unknown must mean yes."""
    mc = MemoryConfig(TensorMemoryLayout.HEIGHT_SHARDED, BufferType.L1, spec)
    assert needs_shard_or_reshard(mc, **_RESHARD_BASE) is True


@pytest.mark.unit
def test_reshard_when_shard_width_violates_the_alignment() -> None:
    """A TILE input aligns to 32, so a 24-wide shard must reshard.  The same
    24-wide shard with a ROW_MAJOR input derives an alignment of 8 from the width
    itself and does NOT -- which is the pairing that proves the clause is reading
    the alignment rather than hardcoding a tile."""
    mc = _sharded(TensorMemoryLayout.HEIGHT_SHARDED, (32, 24))
    assert needs_shard_or_reshard(mc, **_RESHARD_BASE) is True
    assert needs_shard_or_reshard(
        mc, **{**_RESHARD_BASE, 'input_tensor_layout': Layout.ROW_MAJOR_LAYOUT}) is False


@pytest.mark.unit
def test_reshard_when_a_non_block_input_is_col_major() -> None:
    """Only BLOCK may be COL_MAJOR; anything else must be reshuffled to ROW_MAJOR."""
    col = _sharded(TensorMemoryLayout.HEIGHT_SHARDED, (32, 32), ShardOrientation.COL_MAJOR)
    row = _sharded(TensorMemoryLayout.HEIGHT_SHARDED, (32, 32), ShardOrientation.ROW_MAJOR)
    assert needs_shard_or_reshard(col, **_RESHARD_BASE) is True
    assert needs_shard_or_reshard(row, **_RESHARD_BASE) is False


@pytest.mark.unit
def test_block_sharded_matmul_checks_channels_against_the_core_split() -> None:
    """cpp:735 — for a matmul-lowered BLOCK conv the channels must divide exactly
    across the channel cores.  An 8x8 grid gives 8 channel cores at 32 wide, so
    256 fits and 512 does not."""
    mc = _sharded(TensorMemoryLayout.BLOCK_SHARDED, (32, 32))
    mm = {**_RESHARD_BASE, 'is_mm_conv': True}
    assert needs_shard_or_reshard(mc, **{**mm, 'in_channels': 256}) is False
    assert needs_shard_or_reshard(mc, **{**mm, 'in_channels': 512}) is True


@pytest.mark.unit
def test_matmul_conv_requires_a_tile_multiple_shard_height() -> None:
    """cpp:741 applies to matmul-lowered convs only, so the non-mm control must
    tolerate the same ragged height."""
    ragged = _sharded(TensorMemoryLayout.HEIGHT_SHARDED, (48, 32))
    assert needs_shard_or_reshard(ragged, **{**_RESHARD_BASE, 'is_mm_conv': True}) is True
    assert needs_shard_or_reshard(ragged, **_RESHARD_BASE) is False
