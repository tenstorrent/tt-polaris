#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Dual-mode TransformerBlock (decoder layer) for the llama3 tt_transformers port.

Basis: shim-only workloads/ttnn/tt_transformers/decoder.py.

Block: attention_norm -> attention -> residual add -> ff_norm -> mlp -> residual add.
Maps to capture ops 24 (LayerNorm/attn-norm), 25-38 (attention), 39 (BinaryNg residual),
40 (LayerNorm/ff-norm), 41-44 (mlp), 45 (BinaryNg residual).

Audit vs the shim-only base:
  - llama3-only: the moe / Mixtral (TtMoeLayer / MixtralRMSNorm / TtMixtralMLP) branches are
    dropped (args.moe is False for llama3; mixtral has its own untouched workload dir).
"""
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

IS_POLARIS = os.getenv('IRD_ARCH_NAME', '') == ''
if IS_POLARIS:
    import ttsim.front.ttnn as ttnn
else:
    import ttnn  # type: ignore[import-not-found, no-redef]

from workloads.ttnn.tt_transformers_dualmode.attention import Attention
from workloads.ttnn.tt_transformers_dualmode.mlp import MLP
from workloads.ttnn.tt_transformers_dualmode.model_config import dram_shard_core_grid_for_k
from workloads.ttnn.tt_transformers_dualmode.rmsnorm import RMSNorm


class TransformerBlock:
    def __init__(
        self,
        args,
        mesh_device,
        dtype,
        state_dict,
        layer_num,
        weight_cache_path,
        transformation_mats,
        paged_attention_config=None,
        use_paged_kv_cache=False,
    ):
        self.state_dict = state_dict
        self.mesh_device = mesh_device
        self.args = args
        self.hidden_size = args.dim
        self.dim = args.dim
        self.layer_num = layer_num

        self.attention = Attention(
            mesh_device=mesh_device,
            state_dict=state_dict,
            weight_cache_path=weight_cache_path,
            layer_num=layer_num,
            dtype=dtype,
            transformation_mats=transformation_mats,
            configuration=args,
            paged_attention_config=paged_attention_config,
            use_paged_kv_cache=use_paged_kv_cache,
        )
        self.feed_forward = MLP(
            mesh_device=mesh_device,
            args=args,
            state_dict=state_dict,
            weight_cache_path=weight_cache_path,
            layer_num=layer_num,
            dtype=dtype,
            model_config=None,
        )
        self.attention_norm = RMSNorm(
            device=mesh_device,
            dim=args.dim,
            eps=args.norm_eps,
            state_dict=state_dict,
            state_dict_prefix='',
            weight_cache_path=None if args.dummy_weights else weight_cache_path,
            weight_dtype=ttnn.bfloat16,
            weight_key='attention_norm',
            is_distributed=args.is_distributed_norm,
            add_unit_offset=args.rms_norm_add_unit_offset,
        )
        self.ff_norm = RMSNorm(
            device=mesh_device,
            dim=args.dim,
            eps=args.norm_eps,
            state_dict=state_dict,
            state_dict_prefix='',
            weight_cache_path=None if args.dummy_weights else weight_cache_path,
            weight_dtype=ttnn.bfloat16,
            weight_key='ffn_norm',
            is_distributed=args.is_distributed_norm,
            add_unit_offset=args.rms_norm_add_unit_offset,
        )

    def __call__(
        self,
        x,
        current_pos,
        rot_mats=None,
        user_id=0,
        mode='decode',
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
        kv_cache=None,
    ):
        decode = (mode == 'decode')
        if decode:
            # width-shard spec for the residual/norm boundary reshards (capture rows 43/49/51)
            wcfg = ttnn.create_sharded_memory_config(
                shape=(list(x.shape)[-2], self.dim),
                core_grid=dram_shard_core_grid_for_k(self.dim),
                strategy=ttnn.ShardStrategy.WIDTH,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=False,
            )
        attn_in = self.attention_norm(x, mode)
        attn_out = self.attention.forward(
            attn_in, current_pos, rot_mats, user_id, mode,
            page_table=page_table, chunk_page_table=chunk_page_table,
            chunk_start_idx=chunk_start_idx, kv_cache=kv_cache,
        )
        # tt-metal reshards the ATTENTION output onto the residual spec BEFORE residual #1
        # (decoder.py:274 `attn_out = ttnn.to_memory_config(attn_out, skip_mem_cfg)`, commit
        # d9d52dfe7b6), and adds the MLP output directly (line 299, no pre-reshard). We previously
        # did the mirror (resharded ff_out, not attn_out) — same 3 reshards/block but phase-shifted
        # vs the capture, netting 1 fewer 32x4096 reshard at the final-norm boundary. Match tt-metal.
        if decode:
            attn_out = ttnn.reshard(attn_out, wcfg)     # attn out -> residual spec (capture Reshard row 21)
        h = ttnn.add(x, attn_out, memory_config=None)   # residual #1
        ttnn.deallocate(attn_out)
        if decode:
            h = ttnn.reshard(h, wcfg)                    # residual -> ff_norm spec (capture Reshard row 43)

        ff_in = self.ff_norm(h, mode)
        ff_out = self.feed_forward.forward(ff_in, mode)
        out = ttnn.add(h, ff_out, memory_config=None)   # residual #2 (MLP out added directly, per tt-metal)
        if decode:
            out = ttnn.reshard(out, wcfg)                # block out -> next-norm spec (capture Reshard row 51)
        return out
