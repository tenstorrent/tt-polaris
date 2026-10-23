#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Dual-mode RMSNorm for the llama3 tt_transformers port.

Basis: shim-only workloads/ttnn/tt_transformers/rmsnorm.py, made dual-mode.
The gamma weight shape (1, 1, dim//TILE, TILE) matches the HW LayerNormDeviceOperation
in1 (e.g. dim=4096 -> 1x1x128x32, ROW_MAJOR/bf16). Audit vs the shim-only base: the
per-dim if/elif weight shapes are replaced by the generic (1,1,dim//TILE,TILE) (same
values, no hardcoded dim list). Dummy weights (config-only); shim path uses ttnn._rand,
HW path a torch dummy.
"""
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

IS_POLARIS = os.getenv('IRD_ARCH_NAME', '') == ''
if IS_POLARIS:
    import ttsim.front.ttnn as ttnn
else:
    import ttnn  # type: ignore[import-not-found, no-redef]

TILE = 32
SHARD_HEIGHT = TILE  # ttnn.rms_norm requires shard height = a single tile


class RMSNorm:
    def __init__(self, device=None, dim=None, eps=1e-5, state_dict=None, weight_cache_path=None,
                 state_dict_prefix='', weight_dtype=ttnn.bfloat16, weight_key='ffn_norm',
                 is_distributed=False, add_unit_offset=False, sharded_program_config=None,
                 sharded_output_config=None, ccl_topology=None):
        self.device = device
        self.dim = dim
        self.eps = eps
        self.state_dict = state_dict
        self.weight_cache_path = weight_cache_path
        self.state_dict_prefix = state_dict_prefix
        self.weight_dtype = weight_dtype
        self.weight_key = weight_key
        self.is_distributed = is_distributed
        self.add_unit_offset = add_unit_offset
        self.sharded_program_config = sharded_program_config
        self.sharded_output_config = sharded_output_config
        self.ccl_topology = ccl_topology
        self.compute_kernel_config_hifi2 = ttnn.MathFidelity.HiFi2
        if not IS_POLARIS:
            # tt-metal models/common/rmsnorm.py (f8c634a) compute_kernel_config_hifi2: HiFi2,
            # math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True. BH p100a 261018 and WH n150
            # 261015 captures: every prefill norm runs with exactly this; without it ttnn defaults to HiFi4
            # and the LUT key misses.
            self._hw_compute_kernel_config = ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False,
                fp32_dest_acc_en=True, packer_l1_acc=True,
            )

        # gamma weight: (1, 1, dim//TILE, TILE), ROW_MAJOR — matches HW LayerNorm in1
        wshape = (1, 1, dim // TILE, TILE)
        if IS_POLARIS:
            self.weight = ttnn._rand(shape=wshape, device=device, dtype=weight_dtype)
        else:
            import torch  # type: ignore[import-not-found]
            self.weight = ttnn.from_torch(
                torch.randn(*wshape), dtype=weight_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device,
            )

    def __call__(self, x, mode='decode'):
        if IS_POLARIS:
            # Shim signature: weight_tensor / dim / compute_kernel_config carry the LUT-key info
            # (the LayerNorm emission + math_fidelity).
            return ttnn.rms_norm(
                x,
                epsilon=self.eps,
                weight_tensor=self.weight,
                memory_config=None,
                compute_kernel_config=self.compute_kernel_config_hifi2,
                dim=self.dim,
            )
        # Real ttnn.rms_norm: weight= (not weight_tensor), no dim; compute_kernel_config must be a
        # ComputeKernelConfig, not a bare MathFidelity — omit it (defaults) for the HW run.
        # This branch only executes on real ttnn (IRD_ARCH_NAME set), where ttnn.rms_norm's
        # parameter is `weight=`. But mypy always resolves the dual-mode `ttnn` symbol to the shim
        # stubs (ttsim.front.ttnn), whose rms_norm uses `weight_tensor=`, so it flags `weight=` here
        # even though the shim never runs this line. The type: ignore is deliberate — the alternative
        # (adding a `weight=` alias to the shim's rms_norm just to satisfy the type checker) would
        # diverge the shim API from a real tt-metal op for no runtime benefit. Keep the pragma
        # localized to this real-ttnn-only call.
        # Prefill only: tt-metal passes the same config in decode too, but the decode HW path is pinned
        # to its own capture and left unchanged here.
        hw_kw = {'compute_kernel_config': self._hw_compute_kernel_config} if mode == 'prefill' else {}
        return ttnn.rms_norm(x, epsilon=self.eps, weight=self.weight, memory_config=None, **hw_kw)  # type: ignore[call-arg]
