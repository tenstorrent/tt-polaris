#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import math
from enum import Enum
import os, sys
from ttsim.front.ttnn.tensor import DataType
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
import ttsim.front.ttnn as ttnn
import workloads.ttnn.tt_transformers.utils as utils

def tt_all_reduce(tensor, *args, **kwargs):
    return tensor

def tt_all_gather(input_tensor, *args, **kwargs):
    return input_tensor

class OpGroup(Enum):
    """
    LI_* are linear operator groups
    SDPA_* are scaled_dot_product_attention operator groups
    """
    LI_FF1_FF3 = "li_ff1_3"
    LI_FF2 = "li_ff2"
    LI_QKV_DECODE = "li_qkv_decode"
    LI_O_DECODE = "li_o_decode"
    SDPA_DECODE = "sdpa_decode"
    LI_QKV_PREFILL = "li_qkv_prefill"
    LI_O_PREFILL = "li_o_prefill"
    SDPA_PREFILL = "sdpa_prefill"

class TensorGroup(Enum):
    FF1_FF3 = "ff1_3"
    FF2 = "ff2"
    WQKV = "wqkv"
    WO = "wo"
    KV_CACHE = "kv_cache"
    ACTIVATION = "activation"

class Attention():
    def __init__(
        self,
        mesh_device,
        state_dict,
        weight_cache_path,
        layer_num,
        dtype,
        transformation_mats,
        configuration,
        paged_attention_config=None,
        use_paged_kv_cache=False,
    ):
        super().__init__()

        self.state_dict = state_dict
        self.mesh_device = mesh_device
        self.num_devices = configuration.num_devices
        self.TG = self.num_devices == 32
        self.moe = configuration.moe
        self.num_experts = configuration.num_experts
        self.hidden_size = configuration.dim
        self.n_heads = configuration.n_heads
        self.head_dim = configuration.head_dim
        self.max_seq_len = configuration.max_seq_len
        self.max_batch_size = configuration.max_batch_size
        self.n_kv_heads = configuration.n_kv_heads
        self.paged_attention_config = paged_attention_config
        self.min_kv_prefill_shard_seqlen = configuration.min_kv_prefill_shard_seqlen
        self.ccl_dtype = configuration.ccl_dtype
        self.num_reduce_scatter_links = configuration.num_reduce_scatter_links
        self.num_all_gather_links = configuration.num_all_gather_links
        self.MAX_QKV_MM_SEQ_LEN = configuration.MAX_QKV_MM_SEQ_LEN
        self.tile_size = configuration.tile_size
        self.rms_norm_add_unit_offset = configuration.rms_norm_add_unit_offset
        self.num_device_groups = self.num_devices // self.n_kv_heads
        self.num_devices_per_group = self.n_kv_heads if self.TG else self.num_devices
        self.batch_size_per_device_group = (
            max(self.max_batch_size // self.num_device_groups, 1) if self.TG else self.max_batch_size
        )

        self.n_local_heads = self.n_heads // self.num_devices_per_group
        self.n_local_kv_heads = self.n_kv_heads // self.num_devices_per_group

        self.arch_name = configuration.arch_name
        self.dtype = dtype

        self.max_seq_len = configuration.max_seq_len
        self.grid_size = configuration.max_grid_size

        self.compute_kernel_config_hifi2 = configuration.compute_kernel_config_hifi2
        self.compute_kernel_config_hifi2_fp16 = configuration.compute_kernel_config_hifi2_fp16

        self.compute_kernel_config_hifi4 = configuration.compute_kernel_config_hifi4

        self.transformation_mats = transformation_mats
        self.use_fused_qkv_op = getattr(configuration, 'use_fused_qkv_op', True)

        self.model_config = configuration.get_model_config()
        self.ccl_topology = configuration.ccl_topology()
        self.is_multichip = configuration.is_multichip
        self.activation_dtype = ttnn.bfloat16
        self.wqkv_dtype = ttnn.bfloat16
        self.wo_dtype = ttnn.bfloat16
        self.kv_cache_dtype = ttnn.bfloat16
        self.li_qkv_decode_compute_kernel_cfg = ttnn.bfloat16
        self.li_o_decode_compute_kernel_cfg = ttnn.bfloat16
        # Real SDPA compute configs (tt-metal compute_kernel_config_hifi2 / _hifi4); the roofline reads
        # fidelity, fp32 DEST accumulation and the packer flag from them.
        self.sdpa_decode_compute_kernel_cfg = ttnn.BlackholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=True,
            fp32_dest_acc_en=True, packer_l1_acc=True)
        self.sdpa_prefill_compute_kernel_cfg = ttnn.BlackholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
            fp32_dest_acc_en=True, packer_l1_acc=True)
        self.li_qkv_prefill_compute_kernel_cfg = ttnn.bfloat16
        self.li_o_prefill_compute_kernel_cfg = ttnn.bfloat16

        layer_name = f"{__name__}_{layer_num}"

        if configuration.dummy_weights or (weight_cache_path is None):
            cache_name = lambda _: None
        else:
            cache_name = lambda name: weight_cache_path / (f"{layer_name}.{name}")

        wq_str = f"{layer_name}.wq"
        wk_str = f"{layer_name}.wk"
        wv_str = f"{layer_name}.wv"
        wo_str = f"{layer_name}.wo"
        q_norm_str = f"{layer_name}.q_norm"
        k_norm_str = f"{layer_name}.k_norm"

        # Initialize bias tensors as None
        self.wqkv_bias_decode = None
        self.wqkv_bias_prefill = None

        # when splitting the devices, we need to make sure that the number of heads is divisible by the number of devices
        assert self.n_heads % self.num_devices_per_group == 0
        assert self.n_kv_heads % self.num_devices_per_group == 0
        assert configuration.qkv_size % self.num_devices_per_group == 0
        assert configuration.dim % self.num_devices_per_group == 0

        wqkv_mem_config = None #dummy

        qkv_list = []
        for i in range(self.num_devices_per_group):
            wq = ttnn._rand((self.hidden_size, self.head_dim * self.n_heads // self.num_experts), device=self.mesh_device, dtype=ttnn.bfloat16)
            wk = ttnn._rand((self.hidden_size, self.head_dim * self.n_kv_heads // self.num_experts), device=self.mesh_device, dtype=ttnn.bfloat16)
            wv = ttnn._rand((self.hidden_size, self.head_dim * self.n_kv_heads // self.num_experts), device=self.mesh_device, dtype=ttnn.bfloat16)

            qkv = ttnn.cat([wq, wk, wv], dim=-1)
            qkv_list.append(qkv)

        if self.moe:
            qkv_cat = qkv_list[0].unsqueeze(0).unsqueeze(0) # For MOE, do not concat weights across experts, as each expert will select its own weights
        else:
            qkv_cat = ttnn.cat(qkv_list, dim=-1).unsqueeze(0).unsqueeze(0)

        self.wqkv = ttnn.as_tensor(
            qkv_cat,
            dtype=self.wqkv_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG if self.TG else wqkv_mem_config,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                self.mesh_device, dims=(3, 2) if self.TG else (2, 3), mesh_shape=configuration.cluster_shape
            ),
            cache_file_name=cache_name("wqkv_sharded_2d"),
        )

        self.q_norm = lambda x, mode: x
        self.k_norm = lambda x, mode: x

        self.use_fused_all_gather_matmul = False #self.model_config["USE_FUSED_ALL_GATHER_MATMUL"]
        wo_str_weight = ttnn._rand((self.hidden_size, self.hidden_size//self.num_experts), device=self.mesh_device, dtype=ttnn.bfloat16)
        pt_wo = wo_str_weight.unsqueeze(0).unsqueeze(0)
        wo_mem_config = None

        self.wo = ttnn.as_tensor(
            pt_wo,
            dtype=self.wo_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG if (self.use_fused_all_gather_matmul or self.TG) else wo_mem_config,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                self.mesh_device,
                dims=(2, 3) if (self.use_fused_all_gather_matmul or self.TG) else (3, 2),
                mesh_shape=configuration.cluster_shape,
            ),
            cache_file_name=(
                cache_name("wo_width_sharded_2d") if (self.use_fused_all_gather_matmul or self.TG) else cache_name("wo")
            ),
        )
        if not use_paged_kv_cache:
            # vLLM provides its own kv cache
            self.init_kv_cache(configuration, weight_cache_path, device=mesh_device)

        if configuration.query_pre_attn_scalar is not None:
            self.scale = configuration.query_pre_attn_scalar**-0.5
        else:
            self.scale = self.head_dim**-0.5

    def init_kv_cache(self, configuration, weight_cache_path, device=None):
        """
        Generates empty KV cache and pushed to device memory
        """
        if self.paged_attention_config:
            ## False, not taken path
            cache_k = ttnn.zeros(
                    [self.paged_attention_config.max_num_blocks,
                    self.n_local_kv_heads,
                    self.paged_attention_config.block_size,
                    self.head_dim],
                    dtype=ttnn.bfloat16,
                    device=device, layout=ttnn.TILE_LAYOUT
            )
            cache_v = ttnn.zeros(
                    [self.paged_attention_config.max_num_blocks,
                    self.n_local_kv_heads,
                    self.paged_attention_config.block_size,
                    self.head_dim],
                    dtype=ttnn.bfloat16,
                    device=device, layout=ttnn.TILE_LAYOUT
            )
        else:
            cache_k = ttnn.zeros(
                    [self.batch_size_per_device_group,
                    self.n_local_kv_heads,
                    self.max_seq_len,
                    self.head_dim,], device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
            cache_v = ttnn.zeros(
                [self.batch_size_per_device_group,
                    self.n_local_kv_heads,
                    self.max_seq_len,
                    self.head_dim,], device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

        self.layer_past = [
            ttnn.as_tensor(
                k_or_v,
                dtype=self.kv_cache_dtype,
                layout=ttnn.TILE_LAYOUT, #self.model_config["ATTN_W_LAYOUT_TILE"],
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                cache_file_name=(
                    f"{weight_cache_path}/kvcache_{k_or_v.shape}"
                    if weight_cache_path and not configuration.dummy_weights
                    else None
                ),
            )
            for k_or_v in [cache_k, cache_v]
        ]

    def forward_decode(
        self,
        x: ttnn.Tensor,
        current_pos,
        rot_mats=None,
        page_table=None,
        kv_cache=None,
        cur_pos=None,
    ) -> ttnn.Tensor:
        """
        x: (seq_len, 1, batch, dim)
        current_pos: (batch_size), current token position in the sequence for each user
        cur_pos: the same positions as a host list, when the caller knows them
        """
        ###
        # QKV matmuls
        # Use HiFi2 for DRAM-sharded matmuls as they are otherwise flop-bound. Loses 1 bit of activation precision.
        ###
        xqkv_fused_sharded = ttnn.linear(
            x,
            self.wqkv,
        )
        # FIXME: File bug against dram-sharded matmuls with bias
        if self.wqkv_bias_decode:
            # select the bias tensor based on the number of tiles in the rows
            # WARNING: must not change the batch size between compiling and executing a trace
            num_tiles = int(math.ceil(xqkv_fused_sharded.shape[-2] / self.tile_size))
            xqkv_fused_sharded = xqkv_fused_sharded + self.wqkv_bias_decode[num_tiles - 1]

        ttnn.deallocate(x)
        xqkv_fused = tt_all_reduce(
            xqkv_fused_sharded,
            self.mesh_device,
            cluster_axis=1,
            num_reduce_scatter_links=self.num_reduce_scatter_links,
            num_all_gather_links=self.num_all_gather_links,
            memory_config=None, #self.model_config["QKV_OUT_GATHERED_MEMCFG"](list(self.mesh_device.shape)[1]),
            sharded=True,
            dtype=self.ccl_dtype,
            topology=self.ccl_topology,
        )
        ttnn.deallocate(xqkv_fused_sharded)
        # Reshape such that true unpadded batch is tracked in shape
        fqkv_shape = xqkv_fused.shape
        xqkv_fused = ttnn.reshape(
            xqkv_fused, (1, 1, self.batch_size_per_device_group, fqkv_shape[3]), (1, 1, 32, fqkv_shape[3])
        )

        (
            q_heads_pre_rot_1BQD,
            k_heads_pre_rot_1BKD,
            v_heads_1BKD,
        ) = utils.nlp_create_qkv_heads_decode(
            xqkv_fused,
            num_heads=self.n_local_heads,
            num_kv_heads=self.n_local_kv_heads,
            memory_config=None, #self.model_config["CREATE_QKV_DECODE_SHARD"],
        )

        q_heads_pre_rot_1BQD = self.q_norm(q_heads_pre_rot_1BQD, mode="decode")
        k_heads_pre_rot_1BKD = self.k_norm(k_heads_pre_rot_1BKD, mode="decode")
        ttnn.deallocate(xqkv_fused)

        # Q Rotary Embeddings
        q_heads_1BQD = utils.rotary_embedding_llama(
            q_heads_pre_rot_1BQD, rot_mats[0], rot_mats[1], self.transformation_mats["decode"], is_decode_mode=True
        )

        # K Rotary Embeddings
        k_heads_1BKD = utils.rotary_embedding_llama(
            k_heads_pre_rot_1BKD, rot_mats[0], rot_mats[1], self.transformation_mats["decode"], is_decode_mode=True
        )

        ttnn.deallocate(q_heads_pre_rot_1BQD)
        ttnn.deallocate(k_heads_pre_rot_1BKD)

        ###
        # KV update
        ###
        if kv_cache:
            keys = kv_cache[0]
            values = kv_cache[1]
        else:
            keys = self.layer_past[0]
            values = self.layer_past[1]

        utils.paged_update_cache(keys, k_heads_1BKD, update_idxs_tensor=current_pos, page_table=page_table)
        utils.paged_update_cache(values, v_heads_1BKD, update_idxs_tensor=current_pos, page_table=page_table)

        ttnn.deallocate(k_heads_1BKD)
        ttnn.deallocate(v_heads_1BKD)

        # NOTE: Varying the batch size will result in slightly different outputs.
        # For example, a prompt w/ 1 user vs, the same prompt repeated N times for N users, will produce different outputs
        # This is because the SDPA op in decode mode has different number of reductions depending on batch size
        # Which leads to slightly different outputs from attention (due to accumulated errors)

        # cur_pos (host list) rides along with the position tensor so the attended KV length is known.
        sdpa_decode_kwargs = dict(
            cur_pos_tensor=current_pos,
            cur_pos=cur_pos,
            scale=self.scale,
            program_config=self.model_config["SDPA_DECODE_PROGCFG"],
            compute_kernel_config=self.sdpa_decode_compute_kernel_cfg,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        if page_table:
            attn_output_1G4D = ttnn.transformer.paged_scaled_dot_product_attention_decode(
                q_heads_1BQD,
                keys,
                values,
                page_table_tensor=page_table,
                **sdpa_decode_kwargs,
            )
        else:
            attn_output_1G4D = ttnn.transformer.scaled_dot_product_attention_decode(
                q_heads_1BQD,
                keys,
                values,
                **sdpa_decode_kwargs,
            )
        ttnn.deallocate(q_heads_1BQD)

        attn_output_11BH = ttnn.to_memory_config(
            attn_output_1G4D,
            memory_config=None #self.model_config["SCORES_BATCHED_MM_OUTPUT_MEMCFG"](self.batch_size_per_device_group),
        )

        attn_output_cat = utils.nlp_concat_heads_decode(
            attn_output_11BH,
            num_heads=self.n_local_heads,
        )
        ttnn.deallocate(attn_output_11BH)
        ttnn.deallocate(attn_output_1G4D)

        if self.use_fused_all_gather_matmul:
            attn_output_cat = ttnn.to_memory_config(
                attn_output_cat, None, #self.model_config["ATTN_ALL_GATHER_MATMUL_OUTPUT_MEMCFG"]
            )
            _, dense_out_sharded, _ = ttnn.experimental.all_gather_matmul(
                attn_output_cat,
                self.wo,
                dim=3,
                all_gather_core_grid_offset=(0, 4),
                num_links=1,
                program_config=None, #self.model_config["ATTN_ALL_GATHER_MATMUL_PROGCFG"],
                compute_kernel_config=self.li_o_decode_compute_kernel_cfg,
                memory_config_ag=None, #self.model_config["ATTN_ALL_GATHER_MATMUL_OUTPUT_MEMCFG"],
                memory_config_mm=None, #self.model_config["DECODE_RESIDUAL_MEMCFG"],
            )
            ttnn.deallocate(attn_output_cat)
            dense_out_sharded = ttnn.to_memory_config(dense_out_sharded) #, self.model_config["DECODE_RESIDUAL_MEMCFG"])
            return dense_out_sharded

        else:
            attn_output = tt_all_gather(
                attn_output_cat,
                self.mesh_device,
                dim=2,
                cluster_axis=1,
                num_links=2,
                memory_config=None, #self.model_config["GATHER_USERS_MEMCFG"](list(self.mesh_device.shape)[1]),
                sharded=True,
                # dtype=self.ccl_dtype,  # Running bf16 until we have SDPA output bfp8 df; otherwise we have two sharded to interleaved/interleaved to sharded conversions
            )

            attn_output = ttnn.cat([attn_output] * self.num_experts, dim=-1)
            gather_wo = ttnn.repeat(self.wo, [1, 1, 1, self.num_experts])
            dense_out_sharded = ttnn.matmul(
                attn_output,
                gather_wo
            )

            ttnn.deallocate(attn_output_cat)

            # All reduce
            dense_out_reduced = tt_all_reduce(
                dense_out_sharded,
                self.mesh_device,
                cluster_axis=0,
                num_reduce_scatter_links=self.num_reduce_scatter_links,
                num_all_gather_links=self.num_all_gather_links,
                dim=0 if (self.TG and self.hidden_size < 8192) else 3,
                topology=self.ccl_topology,
                sharded=True,
                dtype=self.ccl_dtype,
                use_composite=True if self.hidden_size == 8192 else False,
            )
            if self.num_experts > 1:
                o_shape = dense_out_reduced.shape
                dense_out_reduced = ttnn._rand(shape=[o_shape[0], o_shape[2], o_shape[1], o_shape[3]//self.num_experts],
                                            device=self.mesh_device, dtype=ttnn.bfloat16) # mimics reduction of experts' outputs

            if not self.TG:
                dense_out_reduced = ttnn.to_memory_config(
                    dense_out_reduced, None #self.model_config["DECODE_RESIDUAL_MEMCFG"]
                )

            return dense_out_reduced

    def forward_prefill(
        self,
        x_11SH,
        rot_mats,
        user_id: int = 0,
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
        kv_cache=None,
    ):
        seq_len = x_11SH.shape[-2]
        assert seq_len % 128 == 0 and seq_len > 0, "Seqlen must be divisible by 128"
        ###
        # QKV matmuls
        ###

        # reshaping long sequence to matmul fit on device
        if seq_len > self.MAX_QKV_MM_SEQ_LEN:
            if seq_len % self.MAX_QKV_MM_SEQ_LEN != 0:
                raise ValueError(f"seq_len {seq_len} must be divisible by {self.MAX_QKV_MM_SEQ_LEN}")
            x_11SH = ttnn.reshape(x_11SH, [1, seq_len // self.MAX_QKV_MM_SEQ_LEN, self.MAX_QKV_MM_SEQ_LEN, -1])

        xqkv_fused = ttnn.linear(
            x_11SH,
            self.wqkv,
            # dtype=self.ccl_dtype if self.TG else self.activation_dtype or ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.li_qkv_prefill_compute_kernel_cfg,
            program_config=None, #self.model_config["XQKV_PREFILL_PROGCFG"](seq_len),
        )
        if self.wqkv_bias_prefill is not None:
            xqkv_fused = xqkv_fused + self.wqkv_bias_prefill

        xqkv_fused = tt_all_reduce(
            xqkv_fused,
            self.mesh_device,
            cluster_axis=1,
            num_reduce_scatter_links=self.num_reduce_scatter_links,
            num_all_gather_links=self.num_all_gather_links,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=self.ccl_dtype,
        )

        if seq_len > self.MAX_QKV_MM_SEQ_LEN:
            xqkv_fused = ttnn.reshape(xqkv_fused, [1, 1, seq_len, -1])

        ttnn.deallocate(x_11SH)

        # split qkv into heads
        (
            q_heads_1QSD_pre_rot,
            k_heads_1KSD_pre_rot,
            v_heads_1VSD,
        ) = utils.nlp_create_qkv_heads(
            xqkv_fused,
            num_heads=self.n_local_heads,
            num_kv_heads=self.n_local_kv_heads,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            fused=self.use_fused_qkv_op,
        )

        q_heads_1QSD_pre_rot = self.q_norm(q_heads_1QSD_pre_rot, mode="prefill")
        k_heads_1KSD_pre_rot = self.k_norm(k_heads_1KSD_pre_rot, mode="prefill")

        ttnn.deallocate(xqkv_fused)

        ###
        # Rotary embeddings
        ###
        if DataType.from_numpy(q_heads_1QSD_pre_rot.dtype) != ttnn.bfloat16:  # Rotary embeddings require bfloat16 inputs
            q_heads_1QSD_pre_rot = ttnn.typecast(q_heads_1QSD_pre_rot, dtype=ttnn.bfloat16)

        q_heads_1QSD = utils.rotary_embedding_llama(
            q_heads_1QSD_pre_rot,
            rot_mats[0],
            rot_mats[1],
            self.transformation_mats["prefill"],
            is_decode_mode=False,
        )
        ttnn.deallocate(q_heads_1QSD_pre_rot)

        if DataType.from_numpy(k_heads_1KSD_pre_rot.dtype) != ttnn.bfloat16:  # Rotary embeddings require bfloat16 inputs
            k_heads_1KSD_pre_rot = ttnn.typecast(k_heads_1KSD_pre_rot, dtype=ttnn.bfloat16)

        k_heads_1KSD = utils.rotary_embedding_llama(
            k_heads_1KSD_pre_rot,
            rot_mats[0],
            rot_mats[1],
            self.transformation_mats["prefill"],
            is_decode_mode=False,
        )
        ttnn.deallocate(k_heads_1KSD_pre_rot)

        k_heads_1KSD_8b = ttnn.typecast(k_heads_1KSD, dtype=ttnn.bfloat8_b)#keys_BKSD.dtype)
        v_heads_1VSD_8b = ttnn.typecast(v_heads_1VSD, dtype=ttnn.bfloat8_b)#values_BKSD.dtype)

        # SDPA
        q_heads_1QSD_8b = ttnn.typecast(q_heads_1QSD, dtype=self.activation_dtype) # or ttnn.bfloat8_b)
        ttnn.deallocate(q_heads_1QSD)

        attn_output_84SD = ttnn.transformer.scaled_dot_product_attention(
            q_heads_1QSD_8b,
            k_heads_1KSD_8b,
            v_heads_1VSD_8b,
            is_causal=True,
            scale=self.scale,
            compute_kernel_config=self.sdpa_prefill_compute_kernel_cfg,
            program_config=self.model_config["SDPA_PROGCFG"](seq_len),
        )

        ttnn.deallocate(q_heads_1QSD_8b)
        ttnn.deallocate(k_heads_1KSD_8b)
        ttnn.deallocate(v_heads_1VSD_8b)

        attn_output_1QSD = ttnn.reshape(attn_output_84SD, [1, self.n_local_heads, -1, self.head_dim])

        ###
        # Output matmul
        ###
        attn_output_11SH = utils.nlp_concat_heads(
            attn_output_1QSD,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            fused=self.use_fused_qkv_op,
        )
        ttnn.deallocate(attn_output_1QSD)

        # Non fused All Gather Matmul
        if self.use_fused_all_gather_matmul:  # is true for Ring topology
            attn_output_11SH = ttnn.all_gather(
                attn_output_11SH,
                dim=3,
                num_links=1,
                topology=self.ccl_topology,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        attn_output_11SH = ttnn.cat([attn_output_11SH] * self.num_experts, dim=-1) # gather attn_output for all experts

        output_11SH = ttnn.linear(
            attn_output_11SH,
            self.wo,
            compute_kernel_config=self.li_o_prefill_compute_kernel_cfg,
            dtype=self.activation_dtype, # or ttnn.bfloat8_b,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=None, #self.model_config["WO_PREFILL_PROGCFG"](seq_len),
        )

        if seq_len > 1024:
            output_11SH = ttnn.reshape(output_11SH, [1, 1, seq_len, -1])
        ttnn.deallocate(attn_output_11SH)
        # Reduce-scatter
        if not self.use_fused_all_gather_matmul:
            output_11SH = tt_all_reduce(
                output_11SH,
                self.mesh_device,
                cluster_axis=0,
                dim=0 if self.TG else 3,
                num_reduce_scatter_links=self.num_reduce_scatter_links,
                num_all_gather_links=self.num_all_gather_links,
                topology=self.ccl_topology,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                dtype=self.ccl_dtype,
            )

        return output_11SH

    def __call__(self,
        attention_input,
        current_pos=None,
        rot_mats=None,
        user_id=0,
        mode="prefill",
        page_table=None,
        cur_pos=None,
    ):
        return self.forward(
            attention_input,
            current_pos=current_pos,
            rot_mats=rot_mats,
            user_id=user_id,
            mode=mode,
            page_table=page_table,
            cur_pos=cur_pos,
        )

    def forward(
        self,
        x,
        current_pos,
        rot_mats=None,
        user_id=0,
        mode="decode",
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
        kv_cache=None,
        cur_pos=None,
    ):
        if mode == "prefill":
            return self.forward_prefill(
                x,
                rot_mats,
                user_id,
                page_table=page_table,
                chunk_page_table=chunk_page_table,
                chunk_start_idx=chunk_start_idx,
                kv_cache=kv_cache,
            )
        else:
            return self.forward_decode(x, current_pos, rot_mats, page_table=page_table, kv_cache=kv_cache,
                                       cur_pos=cur_pos)
