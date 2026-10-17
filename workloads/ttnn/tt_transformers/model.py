#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
 
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
import ttsim.front.ttnn as ttnn
from loguru import logger
from workloads.ttnn.tt_transformers.rmsnorm import RMSNorm
from workloads.ttnn.mixtral.mixtral_rmsnorm import RMSNorm as MixtralRMSNorm
from workloads.ttnn.tt_transformers.decoder import TransformerBlock
from workloads.ttnn.tt_transformers.embedding import Embedding
from workloads.ttnn.tt_transformers.lm_head import LMHead
from workloads.ttnn.tt_transformers.rope import RotarySetup

def copy_host_to_device(tensor, device):
    if tensor.device != device:
        tensor = tensor.to(device)
    return tensor

class Transformer():
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
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        assert self.vocab_size > 0
        self.n_layers = args.n_layers
        self.mesh_device = mesh_device
        self.dtype = dtype
        self.model_config = None #args.get_model_config()
        self.grid_size = self.args.max_grid_size
        state_dict_prefix = "" #args.get_state_dict_prefix("", None)

        self.embd = Embedding(
            mesh_device=mesh_device,
            args=args,
            weight_cache_path=args.weight_cache_path(dtype),
            state_dict=state_dict,
            dtype=ttnn.bfloat16,  # Row major layout requires bfloat16
            dim=self.args.dim,
        )

        self.rope_setup = RotarySetup(
            mesh_device,
            args.max_batch_size,
            args.head_dim,
            args.max_seq_len,
            args.rope_theta,
            args.rope_scaling_factor,
            args.orig_context_len,
        )
        self.trans_mats_dict = self.rope_setup.get_both_trans_mats()

        self.layers = [
            TransformerBlock(
                args=args,
                mesh_device=mesh_device,
                dtype=dtype,
                state_dict=state_dict,
                weight_cache_path=weight_cache_path,
                layer_num=i,
                transformation_mats=self.trans_mats_dict,
                paged_attention_config=paged_attention_config,
                use_paged_kv_cache=use_paged_kv_cache,
            )
            for i in range(self.n_layers)
        ]
        if self.args.moe:
            self.norm = MixtralRMSNorm(
                device=mesh_device,
                dim=args.dim,
                args=self.args,
                eps=args.norm_eps,
                state_dict=state_dict,
                state_dict_prefix="", #args.get_state_dict_prefix("", None),
                weight_cache_path=None if args.dummy_weights else weight_cache_path,
                weight_dtype=ttnn.bfloat16,
                weight_key="norm",
                add_unit_offset=self.args.rms_norm_add_unit_offset,
                is_distributed=self.args.is_distributed_norm,
                sharded_program_config=None, #self.model_config["SHARDED_NORM_LM_HEAD_PRGM_CFG"],
                sharded_output_config=None, #self.model_config["LM_HEAD_INPUT_MEMCFG"],
                ccl_topology=self.args.ccl_topology(),
            )
        else:
            self.norm = RMSNorm( # type: ignore[assignment]
                device=mesh_device,
                dim=args.dim,
                eps=args.norm_eps,
                state_dict=state_dict,
                state_dict_prefix="", #args.get_state_dict_prefix("", None),
                weight_cache_path=None if args.dummy_weights else weight_cache_path,
                weight_dtype=ttnn.bfloat16,
                weight_key="norm",
                add_unit_offset=self.args.rms_norm_add_unit_offset,
                is_distributed=self.args.is_distributed_norm,
                sharded_program_config=None, #self.model_config["SHARDED_NORM_LM_HEAD_PRGM_CFG"],
                sharded_output_config=None, #self.model_config["LM_HEAD_INPUT_MEMCFG"],
                ccl_topology=self.args.ccl_topology(),
            )

        self.lm_head = LMHead(
            args=args,
            mesh_device=mesh_device,
            dtype=dtype,
            state_dict=state_dict,
            state_dict_prefix=state_dict_prefix,
            weight_cache_path=weight_cache_path,
            #max_columns_per_device=self.args.max_columns_per_device_lm_head,
        )
    
    def forward(
        self,
        x: ttnn.Tensor,
        current_pos,
        rot_mats=None,
        user_id=0,
        mode="decode",
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
        get_last_token=-1,
        kv_cache=None,
        cur_pos=None,
        ):
        for i, layer in enumerate(self.layers):
            x = layer(
                x,
                current_pos,
                rot_mats,
                user_id,
                mode,
                page_table,
                chunk_page_table=chunk_page_table,
                chunk_start_idx=chunk_start_idx,
                kv_cache=kv_cache[i] if kv_cache is not None else None,
                cur_pos=cur_pos,
            )

        if mode == "prefill" and get_last_token == -1:
            return x

        # Slicing the tensor to the nearest ceiling/floor multiples of 32 for the prefill_len, to get the last token
        if get_last_token != -1:
            pass #x = ttnn.slice(x, (0, 0, get_last_token, 0), (1, 1, get_last_token + 32, x.shape[-1]))

        # Output norm
        x = self.norm(x, mode=mode)

        # if mode == "prefill" and self.model_config["LM_HEAD_INPUT_MEMCFG"].is_sharded():
        #     x = ttnn.interleaved_to_sharded(x, self.model_config["LM_HEAD_INPUT_MEMCFG"])

        x = self.lm_head(x)

        if mode == "prefill":
            x = ttnn.to_layout(x, layout=ttnn.ROW_MAJOR_LAYOUT)
            x = ttnn.to_memory_config(x, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return x

    def __call__(
        self,
        x: ttnn.Tensor,
        current_pos,
        rot_mats=None,
        user_id=0,
        mode="decode",
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
        get_last_token=-1,
        kv_cache=None,
        cur_pos=None,
        ):
        return self.forward(
            x,
            current_pos,
            rot_mats,
            user_id,
            mode="decode",
            page_table=page_table,
            chunk_page_table=chunk_page_table,
            chunk_start_idx=chunk_start_idx,
            get_last_token=get_last_token,
            kv_cache=kv_cache,
            cur_pos=cur_pos,
        )

    def ttnn_prefill_forward(
        self,
        x,
        rot_mats,
        user_id,
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
        get_last_token=-1,
        kv_cache=None,
    ):
        """
        This method will take device tensors and any other args to run forward.
        It returns ttnn device tensors.
        """
        return self.forward(
            x,
            current_pos=None,
            rot_mats=rot_mats,
            user_id=user_id,
            mode="prefill",
            page_table=page_table,
            chunk_page_table=chunk_page_table,
            chunk_start_idx=chunk_start_idx,
            get_last_token=get_last_token,
            kv_cache=kv_cache,
        )
