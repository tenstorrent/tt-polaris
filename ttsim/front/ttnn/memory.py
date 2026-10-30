#!/usr/bin/env python
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import ClassVar

from .buffer import BufferType, TensorMemoryLayout, ShardSpec, ShardOrientation

# Matches profiler / tools.profiling.shape_canonical canonical memory tags.
_CANONICAL_LAYOUT_SUFFIX: dict[str, str] = {
    "INTERLEAVED": "INTERLEAVED",
    "HEIGHT_SHARDED": "HEIGHT_SHARDED",
    "BLOCK_SHARDED": "BLOCK_SHARDED",
    "WIDTH_SHARDED": "WIDTH_SHARDED",
}


class MemoryConfig:
    """Mirrors tt-metal's ``ttnn.MemoryConfig``.

    Constructor signature matches real TTNN::

        MemoryConfig(memory_layout, buffer_type, shard_spec=None)

    Positional order: memory_layout, buffer_type, shard_spec (matching
    ``ttnn.MemoryConfig(TensorMemoryLayout.BLOCK_SHARDED, BufferType.L1, spec)``).
    """

    DRAM: ClassVar[MemoryConfig]
    L1: ClassVar[MemoryConfig]

    def __init__(self, memory_layout=TensorMemoryLayout.INTERLEAVED,
                 buffer_type=BufferType.DRAM, shard_spec=None):
        self.memory_layout = memory_layout
        self.buffer_type = buffer_type
        self.shard_spec = shard_spec

    def is_sharded(self):
        return self.memory_layout != TensorMemoryLayout.INTERLEAVED

    def __eq__(self, other):
        if not isinstance(other, MemoryConfig):
            return False
        return (self.memory_layout == other.memory_layout
                and self.buffer_type == other.buffer_type
                and self.shard_spec == other.shard_spec)

    def __repr__(self):
        return (f"MemoryConfig(memory_layout={self.memory_layout!r}, "
                f"buffer_type={self.buffer_type!r})")

    def to_canonical_memory_tag(self) -> str:
        """Short form matching HW profiler memory strings (e.g. ``L1_BLOCK_SHARDED``)."""
        buf = self.buffer_type.name.upper()
        lay = self.memory_layout.name.upper()
        suffix = _CANONICAL_LAYOUT_SUFFIX.get(lay, "INTERLEAVED")
        return f"{buf}_{suffix}"

    def __str__(self) -> str:
        return self.to_canonical_memory_tag()


# Singleton defaults matching real TTNN's ttnn.DRAM_MEMORY_CONFIG / L1_MEMORY_CONFIG.
MemoryConfig.DRAM = MemoryConfig(TensorMemoryLayout.INTERLEAVED, BufferType.DRAM)
MemoryConfig.L1 = MemoryConfig(TensorMemoryLayout.INTERLEAVED, BufferType.L1)

_SHARD_STRATEGY_TO_LAYOUT = {
    "HEIGHT": TensorMemoryLayout.HEIGHT_SHARDED,
    "WIDTH": TensorMemoryLayout.WIDTH_SHARDED,
    "BLOCK": TensorMemoryLayout.BLOCK_SHARDED,
}


def _div_up(a: int, b: int) -> int:
    return (a + b - 1) // b


def create_sharded_memory_config(shape, core_grid, strategy, orientation=None,
                                 use_height_and_width_as_shard_shape=False):
    """Build an L1-sharded ``MemoryConfig`` from a ``ShardStrategy``.

    Maps ``ShardStrategy.{HEIGHT,BLOCK,WIDTH}`` to the corresponding
    ``TensorMemoryLayout`` -- the same mapping real tt-metal performs.
    """
    strategy_name = getattr(strategy, "name", str(strategy)).upper()
    mem_layout = _SHARD_STRATEGY_TO_LAYOUT.get(
        strategy_name, TensorMemoryLayout.INTERLEAVED,
    )

    # Shard spec, ported from ttnn/ttnn/core.py::create_sharded_memory_config.
    # This used to return a bare MemoryConfig with shard_spec=None, which is
    # WORSE than the (1, 1) stub the trailing-underscore variant had: every
    # consumer that reads a shard spec then takes its "unknown" branch.
    # `needs_shard_or_reshard` returns True unconditionally on a missing spec,
    # `MemoryConfig.__eq__` cannot tell two configs apart, and
    # `_needs_reshard_workaround` declines.  Callers in-tree today are ViT
    # (bh/wh), VGG UNet and tt_transformers; grepping this branch for ResNet-50
    # callers finds none because its model tree is stage 2 and not committed
    # here.  Once it lands, the WORMHOLE path uses this constructor at both of
    # its arch-specific transitions -- pre-layer3 and pre-layer4, both under
    # ``is_wormhole_b0()`` -- so on WH the shard modelling was blind exactly
    # where it matters.  Canonical:
    # tt-metal models/demos/vision/classification/resnet50/ttnn_resnet/tt/
    # ttnn_functional_resnet50.py:763 and :833.
    #
    #     shard_height = prod(batch dims) * height ; shard_width = width
    #     BLOCK  -> (shard_height // grid.y, shard_width // grid.x)   ROW_MAJOR
    #            -> (shard_height // grid.x, shard_width // grid.y)   COL_MAJOR
    #     HEIGHT -> (shard_height // (grid.x*grid.y), shard_width)
    #     WIDTH  -> (shard_height, shard_width // (grid.x*grid.y))
    #
    #     Only BLOCK is orientation-dependent: it splits across BOTH grid axes,
    #     so which axis takes which dim follows the traversal order.  HEIGHT and
    #     WIDTH divide by the total core count, which is orientation-invariant.
    #
    # Real ttnn raises when a dim does not divide the grid; here it is rounded
    # up instead.  Polaris builds graphs for shapes silicon may never run, and a
    # hard failure at graph-build time is worse than an approximate shard shape.
    shard_spec = None
    try:
        dims = [int(d) for d in shape]
        gx = int(getattr(core_grid, 'x', 0) or 0)
        gy = int(getattr(core_grid, 'y', 0) or 0)
        if len(dims) >= 2 and gx > 0 and gy > 0:
            height, width = dims[-2], dims[-1]
            batch = 1
            for d in dims[:-2]:
                batch *= d
            if use_height_and_width_as_shard_shape:
                shard_shape = (height, width)
            else:
                sh, sw = batch * height, width
                total = gx * gy
                if mem_layout == TensorMemoryLayout.BLOCK_SHARDED:
                    # Orientation decides which grid axis each dim is split
                    # across.  Real ttnn uses (height // grid.x, width //
                    # grid.y) for COL_MAJOR -- the axes swap.  Ignoring it
                    # produced a spec whose shape and orientation contradicted
                    # each other, which then fed get_num_cores_channels,
                    # needs_shard_or_reshard and -- now that CoreRangeSet.__eq__
                    # exists -- MemoryConfig.__eq__.  For (1, 1, 1024, 256) on
                    # CoreGrid(x=8, y=4) both arms used to yield (256, 32);
                    # COL_MAJOR is (128, 64).
                    _is_col_major = (
                        isinstance(orientation, ShardOrientation)
                        and orientation == ShardOrientation.COL_MAJOR
                    )
                    if _is_col_major:
                        shard_shape = (_div_up(sh, gx), _div_up(sw, gy))
                    else:
                        shard_shape = (_div_up(sh, gy), _div_up(sw, gx))
                elif mem_layout == TensorMemoryLayout.HEIGHT_SHARDED:
                    shard_shape = (_div_up(sh, total), sw)
                elif mem_layout == TensorMemoryLayout.WIDTH_SHARDED:
                    shard_shape = (sh, _div_up(sw, total))
                else:
                    shard_shape = None
            if shard_shape is not None:
                _orient = orientation if isinstance(orientation, ShardOrientation) \
                    else ShardOrientation.ROW_MAJOR
                shard_spec = ShardSpec(core_grid, shard_shape, _orient)
    except (TypeError, ValueError, AttributeError):
        shard_spec = None  # keep the previous behaviour rather than fail a build

    return MemoryConfig(mem_layout, BufferType.L1, shard_spec=shard_spec)


def create_sharded_memory_config_(shape, grid, mem_layout, orientation, tile_layout):
    """Create a sharded MemoryConfig with ShardSpec metadata.

    This is a minimal implementation for TT-Polaris simulation that preserves
    sharding metadata (mem_layout, grid, orientation) without performing
    actual shard shape calculations.

    Args:
        shape: Per-core shard extent ``[shard_h, shard_w]`` when given as a
            2-element sequence; anything else falls back to the (1, 1) placeholder.
        grid: CoreGrid, CoreRangeSet, or grid-like object specifying core layout
        mem_layout: TensorMemoryLayout (HEIGHT_SHARDED, BLOCK_SHARDED, etc.)
        orientation: ShardOrientation (ROW_MAJOR or COL_MAJOR)
        tile_layout: Whether to use tiled layout (unused in this stub)

    Returns:
        MemoryConfig with the specified memory_layout and a minimal ShardSpec.
    """
    # Normalize grid - convert list to tuple for consistency
    # Otherwise store as-is (could be tuple, CoreRangeSet, or any grid-like object)
    core_grid = tuple(grid) if isinstance(grid, list) else grid

    # Normalize orientation
    if isinstance(orientation, ShardOrientation):
        shard_orientation = orientation
    elif hasattr(orientation, 'name'):
        # Enum-like object, convert by name
        shard_orientation = ShardOrientation[orientation.name]
    else:
        # Default fallback
        shard_orientation = ShardOrientation.ROW_MAJOR

    # Every ResNet-50 / VGG UNet call site passes ``shape`` as the per-core shard
    # extent already (``use_height_and_width_as_shard_shape=True`` in real ttnn),
    # e.g. the pre-layer4 block config passes
    # ``[nearest_32(x.shape[2] // grid_y), x.shape[3] // grid_x]`` = [320, 128].
    # Storing (1, 1) instead threw that away, and three consumers need it:
    # ``needs_shard_or_reshard`` (every clause in
    # tt-metal/ttnn/cpp/ttnn/operations/conv/conv2d/conv2d_utils.cpp:686-745 reads
    # ``input_shard_spec.shape``), ``get_input_channels_alignment``, and
    # ``MemoryConfig.__eq__``.  With the stub, ``shard_w % alignment`` was
    # ``1 % 32`` and the predicate returned True for every such conv.
    #
    # Kept permissive: a shape that is not a usable 2-tuple of positive ints
    # falls back to the old placeholder rather than raising, so call sites that
    # pass a full tensor shape (or nothing meaningful) behave as before.
    _shard_shape = (1, 1)
    try:
        if shape is not None and len(shape) == 2:
            _h, _w = int(shape[0]), int(shape[1])
            if _h > 0 and _w > 0:
                _shard_shape = (_h, _w)
    except (TypeError, ValueError):
        pass
    shard_spec = ShardSpec(
        grid=core_grid,
        shape=_shard_shape,
        orientation=shard_orientation
    )

    # Return MemoryConfig with the specified memory_layout and shard_spec
    return MemoryConfig(mem_layout, BufferType.L1, shard_spec=shard_spec)


def get_memory_config(x):
    mc = getattr(x, '_memory_config', None)
    if mc is not None:
        return mc
    return MemoryConfig.L1
