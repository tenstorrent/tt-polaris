#!/usr/bin/env python
#SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#SPDX-License-Identifier: Apache-2.0

from typing import Iterator, Optional, Iterable
from enum import Enum, auto


#part of memory allocation system, linked to soc_descriptor YAML
#used by "program" module for kernel placement validation
class AllocCoreType(Enum):
    DISPATCH           = auto()
    STORAGE_ONLY       = auto() #Allocates multiple banks per storage cores
    COMPUTE_ONLY       = auto() #runs dispatch kernels
    COMPUTE_AND_STORE  = auto() #Regular Tensix Cores
    INVALID            = auto()

#from umd/device/tt_core_coordinates
class CoreType(Enum):
    ARC         = auto()
    DRAM        = auto()
    ACTIVE_ETH  = auto()
    IDLE_ETH    = auto()
    PCIE        = auto()
    ROUTER_ONLY = auto()
    SECURITY    = auto()

    #part of coordinate system framework used to manage PEs on BH devices
    # specifically at following physical coordinates: (8-10, 8-4, 8-8, 8-6)
    # takes care of:
    #    physical mapping, virtual coordinates, core-management, NOC1 mapping
    # not there for WH devices
    L2CPU       = auto()

    #TENSIX      = auto() -- incompatibility btw ttnn.CoreType & UMD??
    #UNKNOWN     = auto() -- incompatibility btw ttnn.CoreType & UMD??

    #kept for compatibility with soc_descriptor but not used later on
    HARVESTED   = auto()
    ETH         = auto()
    WORKER      = auto()


    @classmethod
    def enumvalue(cls, s:str):
        return CoreType[s.upper()]

    @property
    def cname(self)->str:
        return self.name.lower()

class CoreCoord:
    def __init__(self, *args):
        if isinstance(args[0], (tuple, list)):
            args = args[0] #type: ignore

        assert isinstance(args, (list, tuple)) and len(args) == 2 and \
                all([isinstance(x, int) and x >= 0 for x in args]), \
                f"CoreCoord should a non-negative xy-pair; Illegal inputs ={args} to constructor"

        self.x: int = args[0]
        self.y: int = args[1]

        return

    def __eq__(self, other): return self.x == other.x and self.y == other.y
    def __lt__(self, other): return self.x < other.x or (self.x == other.x and self.y < other.y)
    def __le__(self, other): return self < other or self == other
    def __str__(self)      : return f"({self.x:2d}, {self.y:2d})"

    def __getitem__(self, index):
        assert (0 <= index <= 1), f"Illegal index= {index}"
        val = self.x if index == 0 else self.y
        return val

    def __setitem__(self, index): raise AssertionError("CoreCoord is immutable!!")
    def __delitem__(self, index): raise AssertionError("CoreCoord is immutable!!")

class CoreRange:
    def __init__(self, start: CoreCoord, end: CoreCoord):
        self.start_coord: CoreCoord = start
        self.end_coord  : CoreCoord = end
        assert self.num_cores() > 0, f"CoreRange: {self} not valid!! num_cores={self.num_cores()}"
        return

    def contains(self, coord: CoreCoord) -> bool:
        return (self.start_coord.x <= coord.x <= self.end_coord.x and
                self.start_coord.y <= coord.y <= self.end_coord.y)

    def num_cores(self) -> int:
        return ((self.end_coord.x - self.start_coord.x + 1) *
                (self.end_coord.y - self.start_coord.y + 1))

    def check_overlap(self, other) -> bool:
        is_R = self.start_coord.x > other.end_coord.x #self completely on right
        is_L = self.end_coord.x < other.start_coord.x #self completely on left
        is_B = self.start_coord.y < other.end_coord.y #self completely below
        is_A = self.end_coord.y > other.start_coord.y #self completely above

        if any([is_R, is_L, is_B, is_A]):
               res = False
        else:
            res = True

        return res

    def __iter__(self) -> Iterator[CoreCoord]:
        for yy in range(self.start_coord.y, self.end_coord.y + 1):
            for xx in range(self.start_coord.x, self.end_coord.x + 1):
                yield (CoreCoord(xx, yy))

    def __str__(self):
        return f"CoreRange({self.start_coord} -> {self.end_coord})"

    # Value equality, like tt-metal's CoreRange.  Without it two structurally
    # identical ranges compared by identity, which propagated up through
    # ShardSpec.__eq__ and MemoryConfig.__eq__ and made every
    # ``a.memory_config() != b.memory_config()`` test true — see CoreRangeSet
    # below for what that cost.
    def __eq__(self, other):
        if not isinstance(other, CoreRange):
            return NotImplemented
        return (self.start_coord == other.start_coord
                and self.end_coord == other.end_coord)

    def __hash__(self):
        return hash((self.start_coord.x, self.start_coord.y,
                     self.end_coord.x, self.end_coord.y))

class CoreRangeSet:
    """A collection of non-overlapping core ranges."""
    def __init__(self, ranges: Iterable[CoreRange]):
        self.ranges = list(ranges)
        #check for overlap between all upper triangular pair matrix
        #because overlap(x,y) = overlap(y,x)
        for i, range1 in enumerate(self.ranges):
            for j, range2 in enumerate(self.ranges[i+1:], i+1):
                if range1.check_overlap(range2):
                    raise ValueError(f"Overlapping ranges: {range1} and {range2}")
        return

    def contains(self, coord: CoreCoord) -> bool:
        return any(r.contains(coord) for r in self.ranges)

    def num_cores(self) -> int:
        return sum(r.num_cores() for r in self.ranges)

    def bounding_box(self) -> Optional[CoreRange]:
        """Return smallest rectangle containing all ranges"""
        if not self.ranges:
            return None

        min_x = min(r.start_coord.x  for r in self.ranges)
        min_y = min(r.start_coord.y  for r in self.ranges)
        max_x = max(r.end_coord.x    for r in self.ranges)
        max_y = max(r.end_coord.y    for r in self.ranges)

        return CoreRange(CoreCoord(min_x, min_y), CoreCoord(max_x, max_y))

    def __iter__(self) -> Iterator[CoreCoord]:
        for range_obj in self.ranges:
            yield from range_obj

    # Value equality, like tt-metal's CoreRangeSet.  ShardSpec.__eq__ compares
    # ``self.grid == other.grid``, and MemoryConfig.__eq__ compares shard_spec,
    # so without this two byte-identical sharded configs compared UNEQUAL.
    # ResNet-50's bottleneck does
    # ``if ds_out.memory_config() != out.memory_config(): to_memory_config(...)``
    # (tt-metal/models/demos/vision/classification/resnet50/ttnn_resnet/tt/ttnn_functional_resnet50.py:325-326), which
    # therefore fired unconditionally: polaris emitted a
    # Reshard at three positions where both configs were identical
    # (HEIGHT_SHARDED/L1 (448, 256) on 112 cores, and BLOCK_SHARDED/L1 (96, 256)
    # on 72 cores twice) and the p100a capture has no Reshard at all.
    #
    # Compare the range *decomposition*, not the covered coordinate set, because
    # that is what tt-metal does: ``operator==(const CoreRangeSet&, const
    # CoreRangeSet&)`` (tt_metal/common/core_coord.cpp:484-496) checks
    # ``ranges().size()`` and then walks both vectors pairwise, and
    # ``std::hash<CoreRangeSet>`` (ibid.:699-706) combines the range count and
    # each range in turn. A coordinate-set comparison would be *looser* than
    # hardware: ``_num_cores_to_corerangeset(3, (2, 2), row_wise=True)`` and the
    # column-wise call cover the same three cores but place shard 1 on (1, 0)
    # versus (0, 1), so silicon reshards between them and the model must too.
    # Only the vector order is relaxed here (sorted, not pairwise) — every
    # construction site emits full block first, remainder second, so the order is
    # already canonical and the sort is defensive rather than a semantic change.
    def __eq__(self, other):
        if not isinstance(other, CoreRangeSet):
            return NotImplemented
        if len(self.ranges) != len(other.ranges):
            return False
        key = lambda r: (r.start_coord.x, r.start_coord.y, r.end_coord.x, r.end_coord.y)
        return sorted(self.ranges, key=key) == sorted(other.ranges, key=key)

    def __hash__(self) -> int:
        key = lambda r: (r.start_coord.x, r.start_coord.y, r.end_coord.x, r.end_coord.y)
        return hash(tuple(sorted((key(r) for r in self.ranges))))

class CoreGrid(CoreRangeSet):
    """Mimic real ttnn's ``CoreGrid(x=.., y=..)`` grid descriptor while remaining a
    ``CoreRangeSet`` (TT-Polaris represents grids as range sets). Dual-mode workloads must use the
    real-ttnn ``CoreGrid(x=X, y=Y)`` form so they also run on hardware; legacy TT-Polaris-only callers
    pass a positional iterable of ``CoreRange``. Exposes ``.x`` / ``.y`` like real ttnn.
    """
    def __init__(self, ranges=None, *, x=None, y=None):
        if x is not None or y is not None:
            # ValueError, not assert: this validates a CALLER's arguments, and
            # ``python -O`` strips asserts.  Stripped, ``CoreGrid(x=8)`` would
            # reach ``int(None)`` and die with a TypeError three lines down
            # instead of saying what is wrong.  Same reasoning as
            # conv_sharding._num_cores_to_corerangeset.
            if x is None or y is None:
                raise ValueError("CoreGrid(x=, y=) requires both x and y")
            self.x = int(x)
            self.y = int(y)
            super().__init__([CoreRange(CoreCoord(0, 0), CoreCoord(self.x - 1, self.y - 1))])
        else:
            super().__init__(ranges if ranges is not None else [])
            bb = self.bounding_box()
            self.x = (bb.end_coord.x - bb.start_coord.x + 1) if bb is not None else 0
            self.y = (bb.end_coord.y - bb.start_coord.y + 1) if bb is not None else 0

if __name__ == '__main__':
    testcases = [
        ( (0, 0), (9,  9), ( 5,  5), (14, 14), True ),
        ( (0, 10), (10,  0), (15,  5), (25, 15), False),
        ( (0, 10), (10,  0), ( 0, 20), (10, 15), False),
        ( (0, 10), (10,  0), (10, 10), (20,  0), False),
        ( (0, 10), (10,  0), ( 0,  0), (10, 10), False),
        ( (2,  8), ( 8,  2), ( 3,  7), ( 7,  3), True ),
        ( (0, 10), (10,  0), ( 0, 10), (10,  0), True ),
        ( (0, 10), (10,  0), (10,  0), (20, 10), False),
        ( (5, 10), ( 5, 15), ( 4,  5), ( 6,  5), False),
    ]

    for test_no,(x0,y0,x1,y1,expected) in enumerate(testcases):
        r1 = CoreRange(CoreCoord(x0), CoreCoord(y0))
        r2 = CoreRange(CoreCoord(x1), CoreCoord(y1))
        print(r1, r2, r1.num_cores(), r2.num_cores())
        exit(0)
        #overlap = r1.check_overlap(r2)
        #print(test_no, r1, r1.num_cores(), r2, r2.num_cores(), overlap, expected)
        #assert overlap == expected
