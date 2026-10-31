#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Shard-spec modelling for conv2d, ported from tt-metal.

Why this exists
---------------
The shim used to give a conv output ``MemoryConfig(shard_layout, L1)`` — a
*layout* and nothing else.  That is too coarse for two things:

1. **Which layout the output actually has.**  tt-metal does NOT apply
   ``Conv2dConfig.shard_layout`` unconditionally.  It inherits the input
   tensor's parallel config and only consults the request when a reshard fires
   (``tt-metal/ttnn/cpp/ttnn/operations/conv/conv2d/conv2d_utils.cpp:744-750``)::

       ParallelConfig parallel_config = input_tensor_parallel_config;
       if (conv_config.reshard_if_not_optimal || needs_shard_or_reshard) { ... }

2. **Whether two memory configs differ at all.**  ResNet-50's bottleneck does
   ``if ds_out.memory_config() != out.memory_config(): to_memory_config(...)``.
   With only a layout to compare, configs that hardware treats as different
   compare equal and the Reshard is never emitted — polaris showed 3 against
   the capture's 9.  The difference lives in the per-core shard *shape*.

Every ``needs_shard_or_reshard`` clause in ``tt-metal/ttnn/cpp/ttnn/operations/conv/conv2d/conv2d_utils.cpp:686-745``
reads
``input_shard_spec.shape``, so neither can be modelled without shard shapes.
This module supplies them.

Ported functions, with their C++ origins in
``tt-metal/ttnn/cpp/ttnn/operations/conv/conv2d/conv2d_utils.cpp``:

===========================================  ======
``determine_parallel_config``                169
``get_num_cores_nhw``                        302
``get_num_cores_channels``                   331
``create_sharded_memory_config_from_parallel_config``  357
``determine_output_parallel_config``         239
===========================================  ======

Note ``determine_parallel_config`` TAKES the shard layout and returns it as
``shard_scheme`` — it computes the core grid, it does not choose the layout.
And ``determine_output_parallel_config`` only ever rewrites ``.grid``; the
output's ``shard_scheme`` always equals the input's.  Both facts matter and
neither is obvious from the names.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Sequence

if TYPE_CHECKING:
    from .tensor import Layout

from .buffer import BufferType, ShardOrientation, ShardSpec, TensorMemoryLayout
from .core import CoreCoord, CoreRange, CoreRangeSet
from .memory import MemoryConfig

TILE_HEIGHT = 32
TILE_WIDTH = 32


def _num_cores_to_corerangeset(target_num_cores: int, grid_size: Sequence[int],
                               row_wise: bool = False) -> CoreRangeSet:
    """Split ``target_num_cores`` out of ``grid_size`` into a CoreRangeSet.

    Mirror of tt-metal's ``num_cores_to_corerangeset``
    (``tt_metal/common/work_split.cpp``).  The cpp's general form is a partial
    row/col at the start, whole rows/cols, then a partial at the end — three
    ranges — but that leading partial only arises from a non-origin
    ``start_core``.  This port pins ``start_core`` to (0, 0), so it emits at
    most TWO: the whole-row/col block, then the remainder.
    ``test_at_most_three_ranges`` bounds it at three, which is the cpp
    guarantee and therefore still correct, just not tight.

    PRIVATE TO THIS MODULE on purpose.  Five live callers in
    ``workloads/ttnn/tt_transformers*`` reach the public
    ``device.num_cores_to_corerangeset`` with a device whose ``grid_x``/``grid_y``
    are still 0, and the validation below takes down llama3 / mistral / mixtral / qwen /
    gemma3 at graph-build time on such a grid.  The public entry point therefore
    keeps returning the legacy ``(1, 1)`` for a zero or missing grid and delegates
    here only when the grid is real; see its docstring.  Here the grid always comes
    from ``compute_grid_size``, so reaching the raise means a caller lost it.

    Both halves are pinned by tests/test_ttnn/test_num_cores_to_corerangeset.py —
    either one alone regresses a real workload, which is how this shape was found.
    """
    start_core = CoreCoord(0, 0)
    # ``grid_size`` is always a tuple/list: SimConfig declares
    # ``compute_grid_size: Optional[List[int]]`` (ttsim/config/simconfig.py:498) and every
    # caller forwards a 2-tuple — ttsim/front/ttnn/device.py:278-280, the three
    # sites below, and ttsim/back/device.py:698.  A CoreCoord-shaped branch here was unreachable.
    num_cores_x, num_cores_y = int(grid_size[0]), int(grid_size[1])
    target_num_cores = int(target_num_cores)
    # ValueError, not assert: ``python -O`` strips asserts, and these two are
    # the validation that stops a bad grid from yielding a plausible-looking but
    # wrong CoreRangeSet -- which lands in a shard spec and then in a LUT key,
    # so the failure would be a silent hit-rate drop rather than a crash.
    # Matches the same-class check at :256 (unsupported shard layout), which
    # this module already raises rather than asserts.
    if not (0 < num_cores_x and 0 < num_cores_y):
        raise ValueError(
            f"conv_sharding needs a real compute grid, got ({num_cores_x}, {num_cores_y})")
    total_available = num_cores_x * num_cores_y   # row_wise only changes the SPLIT, not the total
    if not 0 < target_num_cores <= total_available:
        raise ValueError(
            f"Target number of cores {target_num_cores} is greater than total available {total_available}")

    all_cores = []
    leftover = target_num_cores
    sx, sy = start_core.x, start_core.y
    if row_wise:
        if leftover > num_cores_x:
            num_full_rows = leftover // num_cores_x
            rng = CoreRange(CoreCoord(sx, sy), CoreCoord(num_cores_x - 1, sy + num_full_rows - 1))
            all_cores.append(rng); leftover -= rng.num_cores(); sx, sy = 0, sy + num_full_rows
        if leftover > 0:
            all_cores.append(CoreRange(CoreCoord(sx, sy), CoreCoord(sx + leftover - 1, sy)))
    else:
        if leftover > num_cores_y:
            num_full_cols = leftover // num_cores_y
            rng = CoreRange(CoreCoord(sx, sy), CoreCoord(sx + num_full_cols - 1, num_cores_y - 1))
            all_cores.append(rng); leftover -= rng.num_cores(); sx, sy = sx + num_full_cols, 0
        if leftover > 0:
            all_cores.append(CoreRange(CoreCoord(sx, sy), CoreCoord(sx, sy + leftover - 1)))
    return CoreRangeSet(all_cores)


def _div_up(a: int, b: int) -> int:
    return (a + b - 1) // b


def _round_up(a: int, b: int) -> int:
    return b * _div_up(a, b)


def find_closest_largest_divisor(num: int, start_divisor: int, num2: int | None = None) -> int:
    """``tt::find_closest_largest_divisor`` — largest d <= start_divisor dividing num.

    cpp declares TWO overloads,
    ``tt-metal/ttnn/cpp/ttnn/operations/conv/conv2d/conv2d_utils.cpp:54`` (one
    numerator) and ``:62`` (two).  The two-count form requires ``d`` to divide
    BOTH numerators -- a common divisor::

        while (num1 % divisor != 0 or num2 % divisor != 0) divisor--;

    Passing ``num2`` selects it.  The distinction is not cosmetic: a common
    divisor is <= either individual divisor, so the one-count form can only
    over-estimate.  For ResNet-50's 64->256 expansion conv
    (out_channels_ntiles=8, input_channels_blocks=2) on a 12-wide grid the
    one-count form gives 8 and the two-count form gives 2.

    cpp's loops can walk below 1; the ``divisor > 1`` guard here is a
    deliberate difference, shared with the single-count form since inception.
    """
    divisor = start_divisor
    while divisor > 1 and (num % divisor != 0 or (num2 is not None and num2 % divisor != 0)):
        divisor -= 1
    return divisor


# Walk divisors down until the padding overhead is under one shard's worth.
#
# Re-exported from the existing port in tools/perf_lookup/conv_parallel_config.py
# rather than reimplemented here.  A first attempt used a minimise-total-work
# heuristic instead of the real loop condition
# ``(padded_num - num) >= (padded_num // divisor)`` and returned 1 core for
# every BLOCK-sharded case -- plausible-looking output, completely wrong.  One
# definition, already exercised by _annotate_conv_x_pad_logical.
from tools.perf_lookup.conv_parallel_config import (  # noqa: E402
    find_closest_largest_divisor_with_num_padding,
    find_closest_largest_divisor_with_num_padding_and_mult,
)


@dataclass
class ParallelConfig:
    """cpp ``ParallelConfig`` — grid, scheme and orientation."""
    grid: CoreRangeSet
    shard_scheme: TensorMemoryLayout
    shard_orientation: ShardOrientation


def _grid_size(grid: CoreRangeSet) -> tuple[int, int]:
    """Bounding-box extent as (x, y).

    cpp calls ``cores.bounding_box().grid_size()``.  The shim's ``CoreRange``
    has no ``grid_size()``; it exposes ``start_coord`` / ``end_coord``, so the
    extent is computed here (inclusive, hence the +1).

    An earlier version probed for a ``grid_size`` attribute first and fell back
    to the arithmetic.  ``CoreRange`` (core.py:80-115) defines no such attribute,
    so ``getattr`` always returned None and the probe could never fire — a branch
    that cannot be reached is a branch that cannot be tested.
    """
    bb = grid.bounding_box()
    if bb is None:  # core.py:151 — an empty CoreRangeSet has no bounding box
        return (0, 0)
    return (int(bb.end_coord.x - bb.start_coord.x + 1),
            int(bb.end_coord.y - bb.start_coord.y + 1))


def get_num_cores_nhw(grid: CoreRangeSet, shard_layout: TensorMemoryLayout,
                      shard_orientation: ShardOrientation) -> int:
    """cpp:302."""
    if shard_layout == TensorMemoryLayout.WIDTH_SHARDED:
        return 1
    if shard_layout == TensorMemoryLayout.HEIGHT_SHARDED:
        return int(grid.num_cores())
    gx, gy = _grid_size(grid)
    return gx if shard_orientation == ShardOrientation.COL_MAJOR else gy


def get_num_cores_channels(grid: CoreRangeSet, shard_layout: TensorMemoryLayout,
                           shard_orientation: ShardOrientation) -> int:
    """cpp:331."""
    if shard_layout == TensorMemoryLayout.HEIGHT_SHARDED:
        return 1
    if shard_layout == TensorMemoryLayout.WIDTH_SHARDED:
        return int(grid.num_cores())
    gx, gy = _grid_size(grid)
    return gy if shard_orientation == ShardOrientation.COL_MAJOR else gx


def determine_parallel_config(
    shard_layout: TensorMemoryLayout,
    batch_size: int,
    input_channels: int,
    output_height: int,
    output_width: int,
    output_channels: int,
    input_channels_alignment: int,
    compute_grid_size: tuple[int, int],
    # No default, matching cpp: tt-metal/ttnn/cpp/ttnn/operations/conv/conv2d/conv2d_utils.hpp:82-92 declares
    # ``block_shard_orientation`` as a plain parameter and starts its defaults at
    # ``is_shard_height_tile_multiple``.  The same finding was accepted for
    # determine_output_parallel_config -- the default was the divergence, not the
    # parameter -- and it applies verbatim here: a caller that omits the
    # orientation silently gets ROW_MAJOR, which for BLOCK_SHARDED picks the
    # wrong grid axis and shifts every downstream LUT key.  Both production call
    # sites (ttsim/front/ttnn/op.py:2108 and :2386) already pass it.
    block_shard_orientation: ShardOrientation,
    # Same reasoning, same hpp line range: ``enable_channels_padding`` is also a
    # plain parameter in cpp, and a default ``True`` here would silently hand a
    # caller the PADDED divisor search at :272 instead of the common-divisor one
    # at :278.  Those return different core counts on ResNet-50's own channel
    # pairs, so the default decided a shard shape for anyone who omitted it.
    # Both production call sites pass ``not is_mm_conv``, mirroring cpp:626.
    enable_channels_padding: bool,
    is_shard_height_tile_multiple: bool = True,
    is_shard_width_tile_multiple: bool = True,
    act_block_h_override: int = 0,
) -> ParallelConfig:
    """cpp:169.  Computes the core grid for a GIVEN layout; does not pick one."""

    eff_tile_h = TILE_HEIGHT if is_shard_height_tile_multiple else 1
    eff_tile_w = TILE_WIDTH if is_shard_width_tile_multiple else TILE_WIDTH // 2
    out_nhw_ntiles = _div_up(batch_size * output_height * output_width, eff_tile_h)
    out_channels_ntiles = _div_up(output_channels, eff_tile_w)
    act_block_h_ntiles = 1 if act_block_h_override == 0 else act_block_h_override // TILE_HEIGHT

    gx, gy = int(compute_grid_size[0]), int(compute_grid_size[1])
    max_num_cores = gx * gy

    if shard_layout == TensorMemoryLayout.HEIGHT_SHARDED:
        num_cores_nhw = find_closest_largest_divisor_with_num_padding_and_mult(
            out_nhw_ntiles, max_num_cores, act_block_h_ntiles)
        grid = _num_cores_to_corerangeset(num_cores_nhw, (gx, gy), True)
    elif shard_layout == TensorMemoryLayout.BLOCK_SHARDED:
        input_channels_blocks = _div_up(input_channels, input_channels_alignment)
        start_divisor = gx if block_shard_orientation == ShardOrientation.COL_MAJOR else gy
        num_cores_nhw = find_closest_largest_divisor_with_num_padding_and_mult(
            out_nhw_ntiles, start_divisor, act_block_h_ntiles)
        start_divisor_c = gy if block_shard_orientation == ShardOrientation.COL_MAJOR else gx
        num_cores_c = (
            find_closest_largest_divisor_with_num_padding(input_channels_blocks, start_divisor_c)
            if enable_channels_padding
            # cpp:214 uses the TWO-count overload here -- the divisor must divide
            # input_channels_blocks as well.  The WIDTH branch below is the
            # one-count form in cpp too (cpp:224), so the two are not
            # interchangeable by accident.
            else find_closest_largest_divisor(
                out_channels_ntiles, start_divisor_c, num2=input_channels_blocks)
        )
        cores_x = num_cores_nhw if block_shard_orientation == ShardOrientation.COL_MAJOR else num_cores_c
        cores_y = num_cores_c if block_shard_orientation == ShardOrientation.COL_MAJOR else num_cores_nhw
        grid = CoreRangeSet([CoreRange(CoreCoord(0, 0), CoreCoord(cores_x - 1, cores_y - 1))])
    elif shard_layout == TensorMemoryLayout.WIDTH_SHARDED:
        input_channels_ntiles = _div_up(input_channels, eff_tile_w)
        num_cores_c = (
            find_closest_largest_divisor_with_num_padding(input_channels_ntiles, max_num_cores)
            if enable_channels_padding
            else find_closest_largest_divisor(input_channels_ntiles, max_num_cores)
        )
        grid = _num_cores_to_corerangeset(num_cores_c, (gx, gy), True)
    else:
        raise ValueError(f'conv2d supports Height, Block or Width sharding; got {shard_layout}')

    orientation = (
        block_shard_orientation
        if shard_layout == TensorMemoryLayout.BLOCK_SHARDED
        else ShardOrientation.ROW_MAJOR
    )
    return ParallelConfig(grid=grid, shard_scheme=shard_layout, shard_orientation=orientation)


def create_sharded_memory_config_from_parallel_config(
    nhw: int,
    channels: int,
    pconfig: ParallelConfig,
    tile_size: int,
    input_channels_alignment: int,
) -> MemoryConfig:
    """cpp:357.  ``nhw`` is N*H*W of the NHWC-flat tensor.

    cpp is handed an ALREADY-PADDED shape, not the logical one, which is why its
    ``channels % num_cores_channels`` assertion holds and its channel split is a
    plain division.  This port folds that padding in here, so
    ``input_channels_alignment`` is the padding GRANULARITY and the caller picks
    it by direction -- cpp uses two different ones:

        input  config (cpp:641-642)
            round_up(in_channels,  num_cores_c        * input_channels_alignment)
        output config (cpp:1286-1287)
            round_up(out_channels, num_cores_channels * TILE_WIDTH)

    So an input config passes the value from ``get_input_channels_alignment``
    and an output config passes ``TILE_WIDTH``.  The parameter keeps its cpp
    name for greppability even though the output callers do not pass an
    "input channels" alignment.

    Folding is equivalent, not approximate: grouping into blocks and rounding
    each core's share up gives ceil(ceil(c/a)/n)*a, and cpp's pre-pad then
    divide gives ceil(c/(a*n))*a -- the same value for positive integers.

    For ResNet-50 layer 3 that is 1024 -> 64 blocks -> 11 cores -> 6 blocks/core
    -> 96 per core -> 1056 padded, and 1056 is exactly the value the capture
    records for those keys.
    """
    num_cores_nhw = get_num_cores_nhw(pconfig.grid, pconfig.shard_scheme, pconfig.shard_orientation)
    num_cores_channels = get_num_cores_channels(
        pconfig.grid, pconfig.shard_scheme, pconfig.shard_orientation)

    nhw_padded = nhw
    if pconfig.shard_scheme != TensorMemoryLayout.WIDTH_SHARDED:
        nhw_padded = _round_up(nhw, num_cores_nhw * tile_size)
    nhw_shard = nhw_padded // max(num_cores_nhw, 1)

    if num_cores_channels <= 1:
        channel_shard = channels
    else:
        blocks = _div_up(channels, input_channels_alignment)
        channel_shard = _div_up(blocks, num_cores_channels) * input_channels_alignment

    spec = ShardSpec(pconfig.grid, (nhw_shard, channel_shard), pconfig.shard_orientation)
    return MemoryConfig(pconfig.shard_scheme, BufferType.L1, spec)


def determine_output_parallel_config(
    input_parallel_config: ParallelConfig,
    compute_grid_size: tuple[int, int],
    out_channels: int,
    block_shard_orientation: ShardOrientation,
    is_mm_conv: bool,
) -> ParallelConfig:
    """cpp:239.  Only ``.grid`` is recomputed — ``shard_scheme`` is inherited.

    ``block_shard_orientation`` is REQUIRED, matching cpp:239-244 which gives it
    no default.  Both cpp call sites hand it the input config's own orientation
    (cpp:833 passes ``parallel_config.shard_orientation``; cpp:1077 passes the
    same variable used at cpp:1066 to build ``input_parallel_config``), so it is
    always consistent with the orientation this function inherits at the top.

    An earlier version defaulted it to ROW_MAJOR.  Every caller then omitted it,
    so a COL_MAJOR input produced a config labelled COL_MAJOR whose grid axes had
    been laid out ROW_MAJOR, and get_num_cores_channels read the wrong extent
    back.  cpp cannot express that mistake because the argument is mandatory;
    dropping the default restores the same protection.
    """
    out = ParallelConfig(
        grid=input_parallel_config.grid,
        shard_scheme=input_parallel_config.shard_scheme,
        shard_orientation=input_parallel_config.shard_orientation,
    )
    if is_mm_conv:
        return out

    gx, gy = int(compute_grid_size[0]), int(compute_grid_size[1])
    out_channels_ntiles = _div_up(out_channels, TILE_WIDTH)

    if input_parallel_config.shard_scheme == TensorMemoryLayout.WIDTH_SHARDED:
        out.grid = _num_cores_to_corerangeset(
            find_closest_largest_divisor_with_num_padding(out_channels_ntiles, gx * gy), (gx, gy), True)
    elif input_parallel_config.shard_scheme == TensorMemoryLayout.BLOCK_SHARDED:
        start_divisor_c = gy if block_shard_orientation == ShardOrientation.COL_MAJOR else gx
        num_cores_c = find_closest_largest_divisor_with_num_padding(out_channels_ntiles, start_divisor_c)
        num_cores_nhw = get_num_cores_nhw(
            input_parallel_config.grid, input_parallel_config.shard_scheme,
            input_parallel_config.shard_orientation)
        cores_x = num_cores_nhw if block_shard_orientation == ShardOrientation.COL_MAJOR else num_cores_c
        cores_y = num_cores_c if block_shard_orientation == ShardOrientation.COL_MAJOR else num_cores_nhw
        out.grid = CoreRangeSet([CoreRange(CoreCoord(0, 0), CoreCoord(cores_x - 1, cores_y - 1))])
    return out


#: ``tt::tt_metal::hal::get_l1_alignment()`` is 16 bytes; the cpp comment
#: explains the /2 — Halo emits ROW_MAJOR and the smallest dtype is bfloat16, so
#: 8 elements is the smallest NoC-aligned stick.
_L1_ALIGNMENT_ELEMS = 8


def get_input_channels_alignment(
    input_tensor_memory_layout: Optional[TensorMemoryLayout],
    input_tensor_layout: Optional['Layout'] = None,
    sliced_op: bool = False,
    is_mm_conv: bool = False,
    input_memory_config: Optional[MemoryConfig] = None,
) -> int:
    """cpp:92, ported faithfully.

    An earlier version here returned a constant 16 for the non-width, non-mm
    case.  Two things it missed, both load-bearing:

    * the rule only applies when the input tensor is **ROW_MAJOR** (or the op is
      sliced) — a TILE input falls straight through to ``TILE_WIDTH``;
    * when the input is sharded, the alignment is derived from the shard width
      itself (32 / 16 / 8), not fixed.

    A wrong answer here does not change any tensor shape; it changes whether
    ``needs_shard_or_reshard`` predicts a reshard, which is how the conv's
    output sharding gets decided.
    """
    from .tensor import Layout

    is_row_major = input_tensor_layout == Layout.ROW_MAJOR_LAYOUT

    if (not is_mm_conv
            and input_tensor_memory_layout != TensorMemoryLayout.WIDTH_SHARDED
            and (is_row_major or sliced_op)):
        if input_memory_config is not None and input_memory_config.is_sharded():
            spec = getattr(input_memory_config, 'shard_spec', None)
            if spec is not None and getattr(spec, 'shape', None) is not None:
                shard_width = int(spec.shape[1])
                if shard_width % TILE_WIDTH == 0:
                    return TILE_WIDTH
                if shard_width % 16 == 0:
                    return 16
                if shard_width % 8 == 0:
                    return 8
                return TILE_WIDTH
        return _L1_ALIGNMENT_ELEMS
    return TILE_WIDTH


def needs_shard_or_reshard(
    input_memory_config: Optional[MemoryConfig],
    in_channels: int,
    is_mm_conv: bool,
    input_on_device: bool = True,
    input_tensor_layout: Optional['Layout'] = None,
) -> bool:
    """cpp:686-745.  True when the conv must (re)shard rather than inherit.

    Only the clauses whose inputs polaris models are evaluated; the
    ``override_sharding_config`` branch needs an explicit core grid the shim
    does not carry, and is treated as not firing.

    ``input_on_device`` is never passed False by any *production* caller, and is
    kept anyway: it ports cpp:687 (``if (!input_tensor_on_device) {
    needs_shard_or_reshard = true; }``) faithfully.  Every shim conv today
    builds from a device tensor, so the clause cannot fire in a workload --
    dropping the parameter would silently delete a cpp branch rather than
    simplify a polaris-only one.  It is not untested: the False arm is pinned by
    ``tests/test_ttnn/test_conv_sharding_units.py`` (the ``input_on_device:
    False`` case), so the ported branch stays covered even while no workload
    reaches it.
    """
    if not input_on_device or input_memory_config is None:
        return True
    if not input_memory_config.is_sharded():
        return True
    spec = getattr(input_memory_config, 'shard_spec', None)
    if spec is None or getattr(spec, 'shape', None) is None:
        # Sharded but shape unknown: cannot evaluate the clauses below, and
        # guessing "no reshard" is what suppressed six of ResNet-50's nine.
        return True

    shard_h, shard_w = int(spec.shape[0]), int(spec.shape[1])
    layout = input_memory_config.memory_layout
    alignment = get_input_channels_alignment(
        layout, input_tensor_layout, False, is_mm_conv, input_memory_config)

    if shard_w % alignment != 0:
        return True
    if layout != TensorMemoryLayout.BLOCK_SHARDED and \
            spec.orientation != ShardOrientation.ROW_MAJOR:
        return True
    if is_mm_conv and layout == TensorMemoryLayout.BLOCK_SHARDED:
        num_cores_c = get_num_cores_channels(spec.grid, layout, spec.orientation)
        if in_channels != num_cores_c * shard_w:
            return True
    if is_mm_conv and shard_h % TILE_HEIGHT != 0:
        return True
    return False
