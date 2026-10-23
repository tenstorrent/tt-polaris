#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Dual-mode LMHead for the llama3 tt_transformers port.

Basis: shim-only workloads/ttnn/tt_transformers/lm_head.py (simulation path).

Audit vs the shim-only base:
  - Simulation path only (num_splits=1, num_experts=1): the galaxy / column-split / dram-
    sharded-matmul machinery is dropped (dead for our single-chip, lean ModelArgs). The
    output weight is one dummy (dim, vocab_size).
  - The capture splits the lm_head matmul into column chunks wrapped by InterleavedToSharded
    / ShardedToInterleaved (ops 48-59) for L1 sizing. This port emits the single-matmul sim
    path + one sharded_to_interleaved; the chunk-split is a Phase-5 correlation parity item
    (see project_llama3_prefill_op_sequence).
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


def _dummy_weight(shape, device, dtype):
    """Config-only dummy weight (no HF checkpoint), fabricated directly on-device in both
    modes. For llama3-8B the output weight (dim x vocab) is large, so the HW path avoids
    materializing it as a host torch tensor; values are irrelevant (accuracy not validated).

    Weights are DRAM width-sharded (see dram_sharded_weight_memcfg) so the matmul's
    input_1 memory tag matches the silicon capture (DRAM_WIDTH_SHARDED, not INTERLEAVED)."""
    w = ttnn.zeros(list(shape), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    cfg = dram_sharded_weight_memcfg(device, int(shape[-2]), int(shape[-1]))
    if IS_POLARIS:
        w._memory_config = cfg
    else:
        w = ttnn.to_memory_config(w, cfg)
    return w


class LMHead:
    def __init__(self, args, mesh_device, dtype, state_dict, state_dict_prefix, weight_cache_path):
        self.args = args
        self.mesh_device = mesh_device
        self.dtype = dtype
        self.vocab_size = args.vocab_size
        # Column-split the vocab into chunks (the max_columns_per_device split).
        # The HW column-tiles the vocab (L1 sizing): num_splits and chunk width are
        # arch-dependent (tt-metal get_lm_head_max_columns_per_device: BH single-chip
        # 128256//8=16032 -> 8 chunks; WH 668*num_cores=42752 -> 3 chunks), each a
        # DRAM-sharded matmul + ShardedToInterleaved, then Concat back to full vocab.
        #
        # DUAL-MODE split of responsibility:
        #  - TT-Polaris path: emit ONE canonical GEMM (dim x vocab). The arch-specific
        #    column split is realized in the backend (wl2archmap op_split_spec, issue
        #    #477) where the arch is known, keeping this front-end graph device-independent.
        #  - HW path: must do the real column split here — a single dim x vocab GEMM would
        #    L1-OOM on silicon. Split sizes are formula-based per arch (model_config
        #    .lm_head_split_sizes: BH single-chip 8x16032, WH 668*cores) — a hardcoded WH-style
        #    3-way 42752 split overflowed BH L1 in the matmul circular buffers.
        if IS_POLARIS:
            self.output_weight = _dummy_weight((args.dim, args.vocab_size), mesh_device, args.WEIGHTS_DTYPE)
        else:
            # Arch-correct column split (tt-metal get_lm_head_max_columns_per_device): BH single-chip
            # -> 8 x 16032, WH -> 668*cores. A too-wide chunk OOMs BH L1 in the matmul CBs, so the
            # split is formula-based per arch, not hardcoded. Sizes may be uneven (last = remainder).
            self.split_sizes = args.lm_head_split_sizes()
            self.output_weights = [
                _dummy_weight((args.dim, sz), mesh_device, args.WEIGHTS_DTYPE)
                for sz in self.split_sizes
            ]

    def forward(self, x, mode='decode'):
        # lm_head matmul explicitly downcasts its output to bfp8 (tt-metal lm_head.py:
        # dtype=lm_head_dtype, default bfloat8_b) so the downstream ShardedToInterleaved
        # sees a bf8 activation in the capture — pass it rather than inheriting bf16.
        lm_head_dtype = getattr(self.args, 'lm_head_dtype', ttnn.bfloat8_b)
        if IS_POLARIS:
            # Single logical GEMM (dim x vocab); the backend column-split transform expands it to
            # N matmuls + N ShardedToInterleaved + Concat per arch. Output memory = L1 interleaved:
            # the logical lm_head result is the post-concat logits, which tt-metal places in L1
            # interleaved (ttnn.concat memory_config=L1_MEMORY_CONFIG). This is what the downstream
            # sampling typecast reads, so the TopK sees an L1-interleaved input (matches the capture).
            output = ttnn.linear(x, self.output_weight,
                                 compute_kernel_config=self.args.compute_kernel_config_hifi2,
                                 program_config=None, memory_config=ttnn.L1_MEMORY_CONFIG,
                                 dtype=lm_head_dtype)
            return tt_all_reduce(output)  # identity (single-chip)
        # Each column chunk is a DRAM-sharded matmul (chunk-N-wide weight) -> DRAM-sharded program
        # config on HW (see model_config.dram_matmul_config). Program config is per-chunk since split
        # sizes may be uneven (last chunk = remainder).
        # Prefill stages each chunk to L1 interleaved and concats in L1 (tt-metal f8c634a lm_head.py:
        # to_memory_config(output, get_lm_head_sharded_output_mem_config()) = L1_MEMORY_CONFIG without a
        # prefetcher; captures: ShardedToInterleaved -> L1_INTERLEAVED, Concat in L1). The default (None)
        # is DRAM on HW, where a last-dim concat of DRAM chunks lowers to Transposes + a height Concat.
        # Decode keeps None: its HW path is pinned to its own capture and left unchanged here.
        chunk_memcfg = ttnn.L1_MEMORY_CONFIG if mode == 'prefill' else None
        # Prefill chunks run tt-metal's hifi2_fp16 (BH p100a 261018 and WH n150 261015 captures: HiFi2/
        # approx 0/fp32 0/packer 1).
        chunk_ckc = (self.args.compute_kernel_config_hifi2_fp16 if mode == 'prefill'
                     else self.args.compute_kernel_config_hifi2)
        outputs = []
        for w, sz in zip(self.output_weights, self.split_sizes):
            lm_head_pc = self.args.lm_head_decode_program_config(sz)
            # DRAM-sharded matmul needs a sharded output (tt-metal get_lm_head_output_mem_config
            # decode = L1_WIDTH_SHARDED, no-prefetcher); the following sharded_to_interleaved undoes it.
            o = ttnn.linear(x, w, compute_kernel_config=chunk_ckc,
                            program_config=lm_head_pc, memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                            dtype=lm_head_dtype)
            o = ttnn.sharded_to_interleaved(o, memory_config=chunk_memcfg)  # capture STS rows 54/56/58
            outputs.append(o)
        # combine the column chunks back to full vocab (capture Concat row 59)
        if len(outputs) == 1:
            output = outputs[0]
        elif mode == 'prefill':
            output = ttnn.concat(outputs, dim=-1, memory_config=ttnn.L1_MEMORY_CONFIG)
        else:
            output = ttnn.concat(outputs, dim=-1)
        return tt_all_reduce(output)  # identity (single-chip)

    def __call__(self, x, mode='decode'):
        return self.forward(x, mode=mode)
