#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Dual-mode MLP for the llama3 tt_transformers port.

Basis: shim-only workloads/ttnn/tt_transformers/mlp.py, made dual-mode and audited
against tt-metal models/tt_transformers/tt/mlp.py.

Audit fixes vs the shim-only base:
  - Intermediate size uses args.hidden_dim (14336 for llama3-8B) instead of the
    int(3.5 * dim) heuristic. 3.5*4096 == 14336 by luck for 8B, but is wrong for
    3B/70B. Matches tt-metal mlp.py:51-52 (create_dram_sharded_mem_config(dim, hidden_dim)).
  - The gate*up multiply applies the FUSED silu via
    input_tensor_a_activations=[ttnn.UnaryOpType.SILU], mirroring tt-metal mlp.py:239
    (activation_type=ttnn.UnaryOpType.SILU). The shim-only base passed None, dropping
    the silu entirely. This emits ONE Mul op (matching the HW BinaryNg) — no separate
    silu op — so the shim does not over-emit vs HW.
  - Weight dtype = args.WEIGHTS_DTYPE (bfloat8_b), matching the HW matmul in1 dtype in
    the capture; the shim-only base used bfloat16.
  - tt_all_reduce is an identity passthrough (workloads...tt_transformers_dualmode.ccl):
    correct for BOTH modes at num_devices=1 (nothing to reduce on a single chip). The
    multi-chip kwargs (cluster_axis / ccl_dtype / topology / *_links) are omitted because
    they are dead at num_devices=1, keeping the lean ModelArgs free of multi-chip CCL
    config. The galaxy (TG) / reduce_scatter branches in the shim-only base were dead
    (is_galaxy=False) and are dropped here. See design doc §8c (single-chip).
"""
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

IS_POLARIS = os.getenv('IRD_ARCH_NAME', '') == ''
if IS_POLARIS:
    import ttsim.front.ttnn as ttnn
else:
    import ttnn  # type: ignore[import-not-found, no-redef]

from workloads.ttnn.tt_transformers_dualmode.ccl import tt_all_reduce
from workloads.ttnn.tt_transformers_dualmode.model_config import dram_sharded_weight_memcfg

# DRAM-sharded decode matmuls require a sharded output on HW (tt-metal get_mlp_ff1_3_mem_config /
# get_mlp_ff2_mem_config decode = L1_WIDTH_SHARDED, no-prefetcher). None on TT-Polaris (shim's
# constant is a 0-stub; the None path already gives 713/713). See attention._DECODE_MM_MEMCFG.
_DECODE_MM_MEMCFG = None if IS_POLARIS else ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG


def _dummy_weight(shape, device, dtype):
    """Config-only dummy weight (no HF checkpoint), fabricated directly on-device in both
    modes (no host torch materialization). Values are irrelevant; accuracy not validated.

    Weights are DRAM width-sharded (see dram_sharded_weight_memcfg) so the matmul's
    input_1 memory tag matches the silicon capture (DRAM_WIDTH_SHARDED, not INTERLEAVED)."""
    w = ttnn.zeros(list(shape), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    cfg = dram_sharded_weight_memcfg(device, int(shape[-2]), int(shape[-1]))
    if IS_POLARIS:
        w._memory_config = cfg
    else:
        w = ttnn.to_memory_config(w, cfg)
    return w


class MLP:
    def __init__(
        self, mesh_device, args, state_dict, weight_cache_path, layer_num, dtype, model_config, state_dict_prefix=None
    ):
        self.state_dict = state_dict
        self.mesh_device = mesh_device
        self.args = args
        self.dim = args.dim
        self.hidden_dim = args.hidden_dim
        self.model_config = model_config
        self.layer_num = layer_num
        wdtype = args.WEIGHTS_DTYPE
        ff13_dtype = getattr(args, 'FF1_FF3_WEIGHTS_DTYPE', wdtype)
        # w1 -> gate_proj, w2 -> down_proj, w3 -> up_proj. w1/w3 (FF1/FF3) are BFP4 on
        # silicon, w2 (down_proj) stays BFP8 — see ModelArgs.FF1_FF3_WEIGHTS_DTYPE.
        self.w1 = _dummy_weight((self.dim, self.hidden_dim), mesh_device, ff13_dtype)
        self.w2 = _dummy_weight((self.hidden_dim, self.dim), mesh_device, wdtype)
        self.w3 = _dummy_weight((self.dim, self.hidden_dim), mesh_device, ff13_dtype)

    def forward(self, x: 'ttnn.Tensor', mode) -> 'ttnn.Tensor':
        """HF reference: self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))."""
        # DRAM-sharded weights need explicit program configs on HW (None on TT-Polaris): DRAM-sharded 1D in
        # decode, 2D multicast in prefill. See model_config.dram_matmul_config / *_prefill_program_config.
        if mode == 'decode':
            ff13_pc = self.args.mlp_ff13_decode_program_config()
            ff2_pc = self.args.mlp_ff2_decode_program_config()
        else:
            assert x.shape is not None, 'MLP prefill input must have a shape'
            seq_len = x.shape[-2]
            ff13_pc = self.args.mlp_ff13_prefill_program_config(seq_len)
            ff2_pc = self.args.mlp_ff2_prefill_program_config(seq_len)
        # Decode DRAM-sharded matmuls need a sharded output (see _DECODE_MM_MEMCFG); prefill outputs
        # go to DRAM (tt-metal get_mlp_ff1_3/ff2_mem_config(PREFILL); BH p100a prefill capture: every
        # MLP matmul/mul output DRAM_INTERLEAVED). None let the shim shard them.
        mm_memcfg = _DECODE_MM_MEMCFG if mode == 'decode' else ttnn.DRAM_MEMORY_CONFIG
        # FF1/FF3 (gate/up) matmuls run at LoFi under the performance preset
        # (tt-metal DecodersPrecision.performance: OpGroup.LI_FF1_FF3 = LOFI). Prefill uses tt-metal's
        # lofi config (no fp32 acc / approx; see model_config); decode keeps its pinned config.
        ff13_ckc = self.args.compute_kernel_config_lofi if mode == 'decode' else self.args.compute_kernel_config_lofi_prefill
        # FF2 prefill: hifi2_fp16 (BH p100a and WH n150 prefill captures: HiFi2/approx 0/fp32 0).
        ff2_ckc = self.args.compute_kernel_config_hifi2 if mode == 'decode' else self.args.compute_kernel_config_hifi2_fp16
        w1_out = ttnn.linear(
            x, self.w1, core_grid=None, compute_kernel_config=ff13_ckc,
            program_config=ff13_pc, memory_config=mm_memcfg,
        )
        w3_out = ttnn.linear(
            x, self.w3, core_grid=None, compute_kernel_config=ff13_ckc,
            program_config=ff13_pc, memory_config=mm_memcfg,
        )
        ttnn.deallocate(x)

        # Fused silu on the gate (w1_out) — one Mul op, mirrors tt-metal mlp.py:236.
        # The mul downcasts its output to bfp8 (tt-metal: dtype=activation_dtype or
        # bfloat8_b), so the down_proj (w2) sees a bf8 activation in the capture.
        w2_in = ttnn.multiply(
            w1_out, w3_out, input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
            memory_config=None if mode == 'decode' else w1_out.memory_config(),  # tt-metal: w1_out's
            dtype=ttnn.bfloat8_b,
        )
        ttnn.deallocate(w3_out)
        ttnn.deallocate(w1_out)

        # Reshard w2_in onto the FF2 (down-proj) input grid before the matmul (tt-metal mlp.py:244-246,
        # commit d9d52dfe7b6: `w2_in = ttnn.to_memory_config(w2_in, get_mlp_binary_mult_mem_config(mode))`
        # "w2 may use a different core grid"). This is the capture's 32x14336 L1_WIDTH_SHARDED reshard
        # (one per decode block). NOTE: we call ttnn.reshard, not to_memory_config — on real ttnn a
        # sharded->sharded to_memory_config lowers to a single ReshardDeviceOperation (matching the
        # capture), but the TT-Polaris shim's to_memory_config emits STI+ITS (two ops) for that transition,
        # whereas ttnn.reshard emits the single Reshard on both paths. Decode-only (prefill mem cfg is None).
        if mode == 'decode':
            w2_in = ttnn.reshard(w2_in, self.args.mlp_binary_mult_mem_config())

        # FF2 (down) matmul runs at HiFi2 (default LI_FF2 fidelity in the preset).
        w2_out = ttnn.linear(
            w2_in, self.w2, compute_kernel_config=ff2_ckc,
            dtype=ttnn.bfloat16, program_config=ff2_pc, memory_config=mm_memcfg, core_grid=None,
        )
        ttnn.deallocate(w2_in)

        return tt_all_reduce(w2_out)  # identity at num_devices=1

    def __call__(self, x: 'ttnn.Tensor', mode) -> 'ttnn.Tensor':
        return self.forward(x, mode)
