#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from .op import MathFidelity
from .tensor import DataType, Layout
from .buffer import TensorMemoryLayout
from .device import ARCH

from dataclasses import dataclass

@dataclass
class Conv2dConfig:
    weights_dtype                   : DataType
    shard_layout                    : TensorMemoryLayout
    activation                      : str  = 'relu'
    deallocate_activation           : bool = False
    reallocate_halo_output          : bool = True
    reshard_if_not_optimal          : bool = False
    override_sharding_config        : bool = False
    transpose_shards                : bool = False
    enable_act_double_buffer        : bool = False
    enable_weights_double_buffer    : bool = False
    enable_split_reader             : bool = False
    enable_subblock_padding         : bool = False
    in_place                        : bool = False
    full_inner_dim                  : bool = False
    enable_kernel_stride_folding    : bool = False
    act_block_h_override            : int  = 0
    act_block_w_div                 : int  = 1
    output_layout                   : Layout = Layout.TILE_LAYOUT
    #core_grid                       : CoreRangeSet = null

@dataclass
class GrayskullComputeKernelConfig:
    math_fidelity    : MathFidelity = MathFidelity.LoFi
    math_approx_mode : bool = True
    dst_full_sync_en : bool = False

@dataclass
class WormholeComputeKernelConfig:
    math_fidelity    : MathFidelity = MathFidelity.LoFi
    math_approx_mode : bool         = True
    dst_full_sync_en : bool         = False
    fp32_dest_acc_en : bool         = False
    packer_l1_acc    : bool         = False
    #throttle_level   : ThrottleLevel = NO_THROTTLE

BlackholeComputeKernelConfig = WormholeComputeKernelConfig


@dataclass
class SDPAProgramConfig:
    """Mirror of ttnn.SDPAProgramConfig. Field order follows the ttnn struct; a k_chunk_size of 0 on
    the decode path means the kernel picks the chunk."""
    compute_with_storage_grid_size : object            # CoreCoord or (x, y)
    sub_core_grids                 : object = None     # CoreRangeSet, overrides the grid core count
    q_chunk_size                   : int  = 32
    k_chunk_size                   : int  = 32
    exp_approx_mode                : bool | None = None
    max_cores_per_head_batch       : int  = 16


@dataclass
class PagedCacheGeometryOverride:
    block_size   : int = 0
    num_kv_heads : int = 0

#Compute Configs
def init_device_compute_kernel_config(
        dev_arch,
        math_fidelity,
        math_approx_mode = True,
        fp32_dest_acc_en = False,
        packer_l1_acc    = False,
        ):
    return {
            ARCH.GRAYSKULL  : GrayskullComputeKernelConfig(math_fidelity, math_approx_mode),
            ARCH.WORMHOLE_B0: WormholeComputeKernelConfig(math_fidelity,
                                                          math_approx_mode,
                                                          fp32_dest_acc_en,
                                                          packer_l1_acc),
            ARCH.BLACKHOLE  : BlackholeComputeKernelConfig(math_fidelity,
                                                           math_approx_mode,
                                                           fp32_dest_acc_en,
                                                           packer_l1_acc),
            }[dev_arch]

class MatmulMultiCoreReuseMultiCast1DProgramConfig:

    def __init__(self, **kwargs):
        self.compute_with_storage_grid_size = kwargs.get('compute_with_storage_grid_size', (8, 8)),
        self.in0_block_w                    = kwargs.get('in0_block_w',                    1),
        self.out_subblock_h                 = kwargs.get('out_subblock_h',                 1),
        self.out_subblock_w                 = kwargs.get('out_subblock_w',                 1),
        self.per_core_M                     = kwargs.get('per_core_M',                     1),
        self.per_core_N                     = kwargs.get('per_core_N',                     1),
        self.fuse_batch                     = kwargs.get('fuse_batch',                     True),
        self.fused_activation               = kwargs.get('fused_activation',               None),
        self.mcast_in0                      = kwargs.get('mcast_in0',                      True),
        return

