#!/usr/bin/env python
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
from enum import Enum, auto
from typing import Optional, Sequence

class TensorMemoryLayout(Enum):
    INTERLEAVED    = auto()
    HEIGHT_SHARDED = auto()
    WIDTH_SHARDED  = auto()
    BLOCK_SHARDED  = auto()

    @classmethod
    def enumvalue(cls, s:str):
        return TensorMemoryLayout[s.upper()]

    @property
    def cname(self)->str:
        return self.name.lower()

class ShardOrientation(Enum):
    ROW_MAJOR = auto()
    COL_MAJOR = auto()

class ShardDistributionStrategy(Enum):
    ROUND_ROBIN_1D = auto() # Distribute each shard to each of the cores in a linearized list in a round-robin manner.
    GRID_2D        = auto() # Distribute a 2D grid of shards to a 2D grid of cores with one to one mapping.

class ShardMode(Enum):
    PHYSICAL = auto() #TODO: Deprecate this option to treat shard shape as physical
    LOGICAL  = auto()

class BufferType(Enum):
    DRAM          = auto()
    L1            = auto()
    SYSTEM_MEMORY = auto()
    L1_SMALL      = auto()
    TRACE         = auto()


class ShardSpec:
    """Minimal ShardSpec for simulation - preserves sharding metadata.

    Mirrors tt-metal's ShardSpec structure with grid, shape, and orientation.
    For TT-Polaris simulation, this is primarily metadata tracking - actual shard
    calculations are not performed.

    The grid can be any grid-like object (CoreGrid, CoreRangeSet, tuple, etc.)
    and is stored as-is for later extraction by call sites.
    """
    def __init__(self, grid, shape: Optional[Sequence[int]],
                 orientation: ShardOrientation = ShardOrientation.ROW_MAJOR):
        """Initialize ShardSpec.

        Args:
            grid: Core grid specification (CoreGrid, CoreRangeSet, or grid-like object)
            shape: (height, width) shard dimensions, as a tuple or list, or
                None when the shard shape is unknown.  Both spellings and the
                None case are live -- see the normalisation note below.
            orientation: ROW_MAJOR or COL_MAJOR core traversal order
        """
        self.grid = grid
        # Normalised to a tuple so equality does not depend on how the caller
        # spelled it.  ``__eq__`` below compares ``self.shape == other.shape``,
        # and ``(448, 64) != [448, 64]`` in Python -- so a tuple-built spec
        # could never equal a list-built one describing the same shard.  Both
        # spellings are live: the conv_sharding / memory builders use tuples,
        # while ViT constructs ``ShardSpec(grid, [N * H * W // n_cores, C], ...)``
        # at six call sites.  That is a second route to the false inequality
        # ``CoreRangeSet.__eq__`` was added to close -- MemoryConfig.__eq__
        # compares shard_spec, so two byte-identical sharded configs would
        # compare unequal and fire a spurious Reshard.
        self.shape = tuple(shape) if shape is not None else None
        self.orientation = orientation

    def __repr__(self):
        return f"ShardSpec(grid={self.grid}, shape={self.shape}, orientation={self.orientation})"

    def __eq__(self, other):
        if not isinstance(other, ShardSpec):
            return False
        return (self.grid == other.grid and
                self.shape == other.shape and
                self.orientation == other.orientation)

