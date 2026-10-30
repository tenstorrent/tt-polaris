#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from ttsim.graph import WorkloadGraph

from enum import Enum, auto
from dataclasses import dataclass

from typing import Any, Optional, Union

################################x################################x################################x################################
# utility types ---  mostly from tt-umd
ChipId       = int
EthChannelID = int
CoreCoord    = tuple[int, int] #xy-pair

@dataclass
class EthCoord:
    cluster_id : int
    x          : int
    y          : int
    rack       : int
    shelf      : int

class ARCH(Enum):
    GRAYSKULL   = auto()
    WORMHOLE_B0 = auto()
    BLACKHOLE   = auto()
    QUASAR      = auto()
    UNKNOWN     = auto()

    @classmethod
    def enumvalue(cls, s:str):
        return ARCH[s.upper()]

    @property
    def cname(self)->str:
        return self.name.lower()

Arch = ARCH

class BoardType(Enum):
    N150    = auto()
    N300    = auto()
    P100    = auto()
    P150    = auto()
    P300    = auto()
    GALAXY  = auto()
    UNKNOWN = auto()

    @classmethod
    def enumvalue(cls, s:str):
        return BoardType[s.upper()]

    @property
    def cname(self)->str:
        return self.name.lower()
################################x################################x################################x################################

def coerce_arch(arch: Union[ARCH, str, None]) -> ARCH:
    """Normalise an ARCH, an arch name, or None into an ARCH.

    ``None`` maps to ``WORMHOLE_B0`` so that callers which never specify an
    architecture keep the behaviour they had before arch became settable.
    """
    if arch is None:
        return ARCH.WORMHOLE_B0
    if isinstance(arch, ARCH):
        return arch
    if isinstance(arch, str):
        return ARCH.enumvalue(arch)
    raise TypeError(f"arch must be an ARCH, an arch name, or None; got {type(arch).__name__}")


class Device:
    def __init__(self, **kwargs):
        self.device_id            : (Any|int) = kwargs.get('device_id')
        self.l1_small_size        : int = kwargs.get('l1_small_size',        0)
        self.trace_region_size    : int = kwargs.get('trace_region_size',    0)
        self.worker_l1_size       : int = kwargs.get('worker_l1_size',       0)
        self.dispatch_core_config : int = kwargs.get('dispatch_core_config', 0) #DispatchCoreConfig

        #Placeholders
        self.core_grid            : int = 0
        self.grid_x               : int = 0
        self.grid_y               : int = 0

        # Target architecture.  Defaults to WORMHOLE_B0 when unspecified, which
        # is what every caller effectively got before this was settable.  A
        # per-arch workload entry (the way vgg_unet and resnet50 split wh/ and
        # bh/) sets this so that arch-conditional model code -- ttnn.is_blackhole()
        # and friends -- takes the branch hardware would take.  Note the polaris
        # front-end graph is cached per (group, name, instance, batch) and NOT per
        # device (polaris.py get_wlgraph), so arch-dependent graphs must come from
        # separate workload entries, not from re-running one entry on two arches.
        self.architecture         : ARCH = coerce_arch(kwargs.get('arch'))

        # The arch-spec device instance this graph is being built for -- the
        # ``name:`` under ``devices:`` in config/tt_{bh,wh}.yaml, e.g. 'p100a' or
        # 'n150'.  ``None`` means "SKU not declared", which is the safe default:
        # capture-derived tables consulted by the front-end shim must MISS on an
        # undeclared SKU rather than fall back to some other card's measurements.
        #
        # This is identity, not a gate -- there is no process-wide layer behind
        # it -- so unlike the opt-in gates below it IS initialised here, the same way
        # ``architecture`` is.  The backend reads the equivalent from its own
        # ``self.name`` (ttsim/back/device.py:656); the front-end has no path to that,
        # so a workload that needs SKU-keyed behaviour declares it with
        # ``device.set_capture_sku(...)``.  A workload is already SKU-specific by
        # construction -- both ResNet-50 infras hardcode their compute grid
        # (bh:36, wh:35) -- and the front-end graph is cached per workload entry,
        # not per device, so the declaration belongs with the workload.
        self.capture_sku          : Optional[str] = kwargs.get('capture_sku')

        # The opt-in shim gates are deliberately NOT set here.  They are defined
        # in ttsim/utils/shim_gates.py -- that module is the list, so this
        # comment does not carry a count or a copy of the names that would go
        # stale the next time one is added.
        #
        # Each gate resolves as ``getattr(device, <name>, <module default>)``, where
        # the module default lives in ttsim/utils/shim_gates.py and is itself
        # seeded from the matching ``TTSIM_*`` env var at import.  Two layers,
        # in order:
        #
        #   absent attribute -> process-wide default (env var, or ``set_<gate>(x)``
        #                       called with no device)
        #   present attribute -> that device's explicit override, set only by
        #                       ``set_<gate>(x, device)``
        #
        # Materialising the env value onto every Device would collapse that: the
        # attribute would always exist, so the fallback arm of every ``getattr``
        # would be dead code and the process-wide layer unreachable.  Concretely,
        # ``set_tilize_before_matmul(True)`` followed by ``Device()`` would hand
        # back a device pinned to the env's False, silently ignoring the call.
        #
        # Leaving the attributes unset costs nothing in device scoping, which is
        # what they exist for: polaris builds every workload graph in one process,
        # and a gate turned on with ``set_<gate>(True, device)`` still lives on that
        # Device alone and cannot leak into the next workload.  Both ResNet-50 test
        # infras use exactly that form (see their set_<gate>(True, device) block).

        self.args    = kwargs
        self.tensors = {}
        self.ops     = {}

        return

    def set_arch(self, arch: Union[ARCH, str, None]) -> None:
        """Set the target architecture (ARCH or arch name)."""
        self.architecture = coerce_arch(arch)

    def set_capture_sku(self, sku: Optional[str]) -> None:
        """Declare the arch-spec device instance this graph targets ('p100a', 'n150').

        Only needed by a workload whose shim behaviour is keyed to a specific
        capture.  Leaving it unset means SKU-keyed tables miss, which is the
        conservative outcome for a card nobody has measured.
        """
        self.capture_sku = sku

    def get_arch_name(self):
        # `.cname` (e.g. 'blackhole'), matching the module-level ttnn.get_arch_name()
        # and tt-metal's own arch strings.  Previously str(Enum), i.e.
        # 'ARCH.WORMHOLE_B0', which matched neither.
        return self.architecture.cname

    def arch(self):
        return self.architecture

    def get_num_devices(self):
        #TODO: Check this logic
        return 1

    def compute_with_storage_grid_size(self, grid_x: (Any | int) = None, grid_y: (Any | int) = None):

        if grid_x is not None: self.grid_x = grid_x
        if grid_y is not None: self.grid_y = grid_y

        return (self.grid_x, self.grid_y)

    def add_tensor(self, t):
        if t.name not in self.tensors:
            self.tensors[t.name] = t

    def add_op(self, o):
        if o.name not in self.ops:
            self.ops[o.name] = o

    def get_graph(self):
        gg = WorkloadGraph('xxx')
        for _,t in self.tensors.items():
            gg.add_tensor(t)
        for _,o in self.ops.items():
            gg.add_op(o)
        gg.construct_graph()
        return gg

    def __str__(self):
        return f"(Device: {self.args})"


def open_device(**kwargs):
    return Device(**kwargs) # Normally returns ttnn.multi_device.MeshDevice

def close_device(device: Device):
    # Clear the default-device pointer when the device being closed IS the
    # default.  Now that ttnn.get_arch_name() reads the default device's arch, a
    # lingering pointer would leak one workload's architecture into the next
    # workload built in the same process — polaris.py builds every workload
    # graph in one process, so that leak would be silent and cross-workload.
    global _default_device
    if _default_device is device:
        _default_device = None
    return

_default_device = None

def set_default_device(device: Optional[Device]) -> None:
    global _default_device
    _default_device = device

def get_default_device()->Device:
    assert _default_device is not None
    return _default_device

def try_get_default_device() -> Optional[Device]:
    """The default device, or None when none has been set.

    Non-asserting counterpart to ``get_default_device`` for callers that must
    work before a device exists — notably the module-level
    ``ttnn.get_arch_name()``, which model code may call at import time.
    """
    return _default_device


# Placeholder for ``Tensor(..., device=...)`` / factories: use ``set_default_device`` result.
USE_DEFAULT_DEVICE = object()


def resolve_device(device):
    """Map ``USE_DEFAULT_DEVICE`` → ``get_default_device()``; pass ``None`` and ``Device`` through."""
    if device is USE_DEFAULT_DEVICE:
        return get_default_device()
    if device is None:
        return None
    if isinstance(device, Device):
        return device
    raise TypeError(
        f"device must be None, USE_DEFAULT_DEVICE, or Device, got {type(device).__name__}"
    )


def num_cores_to_corerangeset(target_num_cores=None, grid_size=None, row_wise=False, **kwargs):
    """Split ``target_num_cores`` out of ``grid_size`` into a CoreRangeSet.

    Mirror of tt-metal's ``num_cores_to_corerangeset``
    (``tt_metal/common/work_split.cpp``); delegates to the port in
    ``conv_sharding``, which holds the splitting rule.

    **Degenerate-grid compatibility.**  This used to be a stub returning the
    tuple ``(1, 1)``.  Five call sites in ``workloads/ttnn/tt_transformers*``
    reach it with a device whose ``grid_x`` / ``grid_y`` are still 0 and treat
    the result as an opaque value; a faithful implementation rejects that
    grid and takes down llama3 / mistral / mixtral / qwen / gemma3 at
    graph-build time.  A zero or missing grid therefore still returns
    ``(1, 1)``.  That is a compatibility shim, not a model — those workloads get
    a meaningless grid either way, and the real fix is to give their device a
    ``compute_with_storage_grid_size`` first.  ResNet-50 (which calls
    ``.num_cores()`` on the result) passes a real grid and gets a real
    CoreRangeSet.
    """
    # No ``kwargs.get`` re-reads for the three named parameters: they are in the
    # signature, so a caller passing them by keyword binds them to the parameters
    # and they never reach ``**kwargs``. ``**kwargs`` stays to absorb the extra
    # arguments real ttnn's signature accepts and this port ignores.
    try:
        gx, gy = ((int(grid_size[0]), int(grid_size[1]))
                  if isinstance(grid_size, (tuple, list))
                  else (int(grid_size.x), int(grid_size.y)))
    except Exception:
        gx = gy = 0
    if gx <= 0 or gy <= 0 or not target_num_cores:
        return (1, 1)  # legacy behaviour; see docstring

    from .conv_sharding import _num_cores_to_corerangeset
    return _num_cores_to_corerangeset(target_num_cores, (gx, gy), bool(row_wise))


def create_sharded_memory_config(*args, **kwargs):
    from .memory import create_sharded_memory_config as _create
    return _create(*args, **kwargs)

def ReadDeviceProfiler(device: Device):
    return
