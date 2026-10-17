#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from .device import open_device, close_device, ARCH, num_cores_to_corerangeset, create_sharded_memory_config, ReadDeviceProfiler
from .device import USE_DEFAULT_DEVICE, resolve_device, set_default_device, get_default_device
from .ttnn_shim import interleaved_to_sharded, sharded_to_interleaved, reshard, to_memory_config
from .tensor import (
    Tensor,
    _rand,
    full,
    zeros,
    ones,
    from_torch,
    to_torch,
    to_device,
    DataType,
    ShardTensor2dMesh,
    typecast,
    pad,
    require_ttnn_tensor,
)
from .tensor import Layout, as_tensor, arange, stack, ShardStrategy, unsqueeze_to_4D, ReplicateTensorToMesh
from .config import Conv2dConfig, WormholeComputeKernelConfig, init_device_compute_kernel_config
from .config import MatmulMultiCoreReuseMultiCast1DProgramConfig
from .config import BlackholeComputeKernelConfig, SDPAProgramConfig, PagedCacheGeometryOverride
from .buffer import TensorMemoryLayout, ShardOrientation, BufferType, ShardSpec
from .memory import MemoryConfig, create_sharded_memory_config_, get_memory_config
from .types import TILE_HEIGHT, TILE_WIDTH
from .core   import CoreCoord, CoreRange, CoreRangeSet, CoreGrid
from .op     import *
from .ttnn_shim import to_layout, permute, ttnn_reshape as reshape
from .ttnn_shim import untilize_with_unpadding, tilize_with_val_padding, untilize
from ttsim.ops.tensor import Shape


def _tensor_permute(self, *args, memory_config=None, **kwargs):
    """Variadic ``tensor.permute(0,2,1)`` or ``tensor.permute([0,2,1])``; records **Permute**."""
    mc = kwargs.pop('memory_config', memory_config)
    if kwargs:
        raise TypeError(f"permute got unexpected keyword arguments: {sorted(kwargs)}")
    if not args:
        raise TypeError('permute expected at least one dimension index or a sequence')
    if len(args) == 1 and isinstance(args[0], (list, tuple)):
        order = [int(x) for x in args[0]]
    else:
        order = [int(x) for x in args]
    return permute(self, order, memory_config=mc)


setattr(Tensor, 'permute', _tensor_permute)

float32  = DataType.FLOAT32
bfloat16 = DataType.BFLOAT16
int64    = DataType.INT64
uint32   = DataType.UINT32
uint16   = DataType.UINT16
bfloat8_b = DataType.BFLOAT8_B
bfloat4_b = DataType.BFLOAT4_B
bool      = DataType.BOOL
int32     = DataType.INT32

ROW_MAJOR_LAYOUT = Layout.ROW_MAJOR_LAYOUT
TILE_LAYOUT      = Layout.TILE_LAYOUT
TILE_SIZE        = 32

DRAM_MEMORY_CONFIG = MemoryConfig.DRAM  # MemoryConfig(INTERLEAVED, BufferType.DRAM)
L1_MEMORY_CONFIG   = MemoryConfig.L1   # MemoryConfig(INTERLEAVED, BufferType.L1)

L1_WIDTH_SHARDED_MEMORY_CONFIG = 0

#placeholders

def name_to_datatype(dtype_name: str) -> DataType:
    try:
        return DataType[dtype_name.upper()]
    except KeyError:
        valid_dtypes = [dt.name.lower() for dt in DataType]
        raise ValueError(
            f"Invalid dtype name '{dtype_name}'. Valid options are: {', '.join(valid_dtypes)}"
        )

def get_arch_name():
    return ARCH.WORMHOLE_B0.cname

def is_tensor_storage_on_device(ttnn_tensor_like):
    return True

def prepare_conv_weights(weight_tensor, weights_format, input_memory_config, input_layout,
                         has_bias, input_dtype, **kwargs):
    return weight_tensor

def prepare_conv_bias(bias_tensor, input_memory_config, input_layout, input_dtype, **kwargs,):
    return bias_tensor


def deallocate(x): pass
def reallocate(x): return x

def copy_host_to_device_tensor(src, dst): return dst
def synchronize_device(device): pass

