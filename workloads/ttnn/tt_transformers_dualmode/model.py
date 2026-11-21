#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Dual-mode Transformer (top-level model) for the llama3 tt_transformers port.

Basis: shim-only workloads/ttnn/tt_transformers/model.py.

Structure: embedding -> N x TransformerBlock -> final RMSNorm -> LMHead. Maps to the capture:
EmbeddingsDeviceOperation (op 21), the per-block sequence (24-45) x n_layers, final LayerNorm
(op 47), lm_head tail (48-59).

Audit vs the shim-only base:
  - llama3-only: the moe / Mixtral norm branch is dropped (args.moe is False).
  - Final norm + lm_head run when get_last_token != -1 (last-token logits) — the capture
    includes them, so the prefill graph is built with get_last_token set.
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
        x = self.norm(x, mode=mode)        # final norm (capture op 47)
        x = self.lm_head(x)                # lm_head (capture ops 48-59)
        if mode == 'prefill':
            x = ttnn.to_layout(x, layout=ttnn.ROW_MAJOR_LAYOUT)
            x = ttnn.to_memory_config(x, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return x

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
