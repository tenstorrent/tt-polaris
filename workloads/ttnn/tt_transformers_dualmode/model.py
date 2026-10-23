#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Dual-mode Transformer (top-level model) for the llama3 tt_transformers port.

Basis: shim-only workloads/ttnn/tt_transformers/model.py.

Structure: embedding -> N x TransformerBlock -> final RMSNorm -> LMHead. Maps to the capture
(project_llama3_prefill_op_sequence): EmbeddingsDeviceOperation (op 21), the per-block
sequence (24-45) x n_layers, final LayerNorm (op 47), lm_head tail (48-59).

Audit vs the shim-only base:
  - llama3-only: the moe / Mixtral norm branch is dropped (args.moe is False).
  - Prefill: get_last_token == -1 stops after the last block (the traced forward, 21 ops at 1
    layer). Otherwise it is the last token's index and the prefill tail runs (_prefill_logits:
    slice -> norm -> lm_head -> DRAM logits), which tt-metal runs eagerly after each trace replay.
"""
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

IS_POLARIS = os.getenv('IRD_ARCH_NAME', '') == ''
if IS_POLARIS:
    import ttsim.front.ttnn as ttnn
else:
    import ttnn  # type: ignore[import-not-found, no-redef]

from workloads.ttnn.tt_transformers_dualmode.decoder import TransformerBlock
from workloads.ttnn.tt_transformers_dualmode.embedding import Embedding
from workloads.ttnn.tt_transformers_dualmode.lm_head import LMHead
from workloads.ttnn.tt_transformers_dualmode.model_config import dram_shard_core_grid_for_k
from workloads.ttnn.tt_transformers_dualmode.rmsnorm import RMSNorm
from workloads.ttnn.tt_transformers_dualmode.rope import RotarySetup


class Transformer:
    def __init__(
        self,
        args,
        dtype,
        mesh_device,
        state_dict,
        weight_cache_path,
        paged_attention_config=None,
        use_paged_kv_cache=False,
    ):
        self.args = args
        self.vocab_size = args.vocab_size
        assert self.vocab_size > 0
        self.n_layers = args.n_layers
        self.mesh_device = mesh_device
        self.dtype = dtype

        self.embd = Embedding(
            mesh_device=mesh_device, args=args, weight_cache_path=weight_cache_path,
            state_dict=state_dict, dtype=ttnn.bfloat16, dim=args.dim,
        )
        self.rope_setup = RotarySetup(
            mesh_device, args.max_batch_size, args.head_dim, args.max_seq_len,
            args.rope_theta, args.rope_scaling_factor, args.orig_context_len,
        )
        self.trans_mats_dict = self.rope_setup.get_both_trans_mats()

        self.layers = [
            TransformerBlock(
                args=args, mesh_device=mesh_device, dtype=dtype, state_dict=state_dict,
                weight_cache_path=weight_cache_path, layer_num=i,
                transformation_mats=self.trans_mats_dict,
                paged_attention_config=paged_attention_config,
                use_paged_kv_cache=use_paged_kv_cache,
            )
            for i in range(self.n_layers)
        ]
        self.norm = RMSNorm(
            device=mesh_device, dim=args.dim, eps=args.norm_eps, state_dict=state_dict,
            state_dict_prefix='', weight_cache_path=None if args.dummy_weights else weight_cache_path,
            weight_dtype=ttnn.bfloat16, weight_key='norm', add_unit_offset=args.rms_norm_add_unit_offset,
            is_distributed=args.is_distributed_norm,
        )
        self.lm_head = LMHead(
            args=args, mesh_device=mesh_device, dtype=dtype, state_dict=state_dict,
            state_dict_prefix='', weight_cache_path=weight_cache_path,
        )

    def forward(self, x, current_pos, rot_mats=None, user_id=0, mode='decode', page_table=None,
                chunk_page_table=None, chunk_start_idx=None, get_last_token=-1, kv_cache=None):
        if mode == 'decode':
            # Width-shard the decode activation into the transformer block (capture ITS, row 30:
            # token-embedding output DRAM-interleaved -> L1 width-sharded for the first rms_norm).
            x = ttnn.interleaved_to_sharded(x, ttnn.create_sharded_memory_config(
                shape=(list(x.shape)[-2], self.args.dim),
                core_grid=dram_shard_core_grid_for_k(self.args.dim),
                strategy=ttnn.ShardStrategy.WIDTH,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=False,
            ))
        for i, layer in enumerate(self.layers):
            x = layer(
                x, current_pos, rot_mats, user_id, mode, page_table,
                chunk_page_table=chunk_page_table, chunk_start_idx=chunk_start_idx,
                kv_cache=kv_cache[i] if kv_cache is not None else None,
            )

        if mode == 'prefill' and get_last_token == -1:
            return x

        if mode == 'decode':
            # Final-norm input reshard: the capture has an extra 32x4096 WIDTH_SHARDED reshard right
            # before the final LayerNorm (tt-metal's final DistributedNorm reshards its input). The
            # per-block out-reshard alone left us 1 short at this boundary — matches the capture's
            # last-segment 2 reshards. Same width-shard spec as the decode residual/norm boundaries.
            x = ttnn.reshard(x, ttnn.create_sharded_memory_config(
                shape=(list(x.shape)[-2], self.args.dim),
                core_grid=dram_shard_core_grid_for_k(self.args.dim),
                strategy=ttnn.ShardStrategy.WIDTH,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=False,
            ))
        if mode == 'prefill':
            return self._prefill_logits(x, get_last_token)
        x = self.norm(x, mode=mode)        # final norm (capture op 47)
        x = self.lm_head(x)                # lm_head (capture ops 48-59)
        return x

    def _prefill_logits(self, x, last_token_idx):
        """Prefill tail: last-token tile -> final norm -> lm_head -> DRAM logits.

        tt-metal model.process_logits_after_prefill_trace + _apply_norm_and_lm_head. The profiler
        records these eagerly after each trace replay (f8c634a captures: Slice, LayerNorm,
        InterleavedToSharded, N x (Matmul + ShardedToInterleaved), Concat, Copy; N = 8 on BH, 3 on WH):
          - slice the 32-row tile holding the last token ((idx // 32) * 32);
          - final RMSNorm in prefill mode on [1,1,32,dim];
          - width-shard it for the DRAM-sharded lm_head (get_lm_head_input_mem_config(PREFILL):
            [32, dim // 64] shards on the 8x8 lm_head grid);
          - lm_head (Polaris: one GEMM that the backend splits 8 ways; HW: the 8 chunks directly);
          - to_memory_config(DRAM): an interleaved L1 -> DRAM move, a CopyDeviceOperation on HW.
        """
        dim = self.args.dim
        x = self._slice_last_token_tile(x, last_token_idx)
        x = self.norm(x, mode='prefill')
        x = ttnn.interleaved_to_sharded(x, ttnn.create_sharded_memory_config(
            (32, dim // 64), ttnn.CoreGrid(x=8, y=8), ttnn.ShardStrategy.WIDTH,
            ttnn.ShardOrientation.ROW_MAJOR, use_height_and_width_as_shard_shape=True,
        ))
        x = self.lm_head(x, mode='prefill')
        return ttnn.to_memory_config(x, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def _slice_last_token_tile(self, x, last_token_idx):
        """Rows [(idx // 32) * 32, +32) of the prefill output: the tile holding the last token.

        The bounds are int32 device tensors (tt-metal f8c634a process_logits_after_prefill_trace:
        ttnn.slice(starts=, ends=, slice_dim=2, num_devices=seq_len // 32), so one program serves every
        prompt offset); the HW op then takes them as inputs 1-2 (BH p100a 261018 and WH n150 261015
        captures: Slice arity 3, inputs 1-2 [1,1,1,4] INT32 ROW_MAJOR DRAM).
        """
        start = (last_token_idx // 32) * 32
        seq_len, dim = int(x.shape[-2]), int(x.shape[-1])
        if IS_POLARIS:
            # The shim has no tensor values; the output shape comes from slice_dim / num_devices, as on HW.
            starts = ttnn.zeros([4], dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.mesh_device)
            ends = ttnn.zeros([4], dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.mesh_device)
        else:
            import torch  # type: ignore[import-not-found]
            starts = ttnn.from_torch(torch.tensor([0, 0, start, 0], dtype=torch.int32), device=self.mesh_device)
            ends = ttnn.from_torch(torch.tensor([1, 1, start + 32, dim], dtype=torch.int32), device=self.mesh_device)
        return ttnn.slice(input_tensor=x, starts=starts, ends=ends, slice_dim=2, num_devices=seq_len // 32)

    def ttnn_prefill_forward(self, x, rot_mats, user_id, page_table=None, chunk_page_table=None,
                             chunk_start_idx=None, get_last_token=-1, kv_cache=None):
        return self.forward(
            x, current_pos=None, rot_mats=rot_mats, user_id=user_id, mode='prefill',
            page_table=page_table, chunk_page_table=chunk_page_table, chunk_start_idx=chunk_start_idx,
            get_last_token=get_last_token, kv_cache=kv_cache,
        )

    def __call__(self, x, current_pos, rot_mats=None, user_id=0, mode='decode', page_table=None,
                 chunk_page_table=None, chunk_start_idx=None, get_last_token=-1, kv_cache=None):
        return self.forward(
            x, current_pos, rot_mats=rot_mats, user_id=user_id, mode=mode, page_table=page_table,
            chunk_page_table=chunk_page_table, chunk_start_idx=chunk_start_idx,
            get_last_token=get_last_token, kv_cache=kv_cache,
        )
