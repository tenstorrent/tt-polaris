#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
 
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
import ttsim.front.ttnn as ttnn

class ModelArgs:
    def __init__(self, mesh_device, model_name="llama3-8B", max_batch_size=1, max_seq_len=256, instruct=False, dummy_weights=False):
        self.mesh_device = mesh_device

        # Make the check completely case-insensitive
        name_lower = model_name.lower()

        # 1. Standard defaults that apply to ALMOST all models
        self.head_dim = 128
        self.n_kv_heads = 8
        self.norm_eps = 1e-5
        self.vocab_size = 128256
        self.n_layers = 1
        self.num_devices = 1
        self.num_experts = 1
        self.moe = False
        self.hidden_size = None # Prevent missing attribute errors
        self.dim = None         # Prevent missing attribute errors

        # 2. Model-Specific Overrides
        if "llama3-8b" in name_lower:
            self.model_name = "llama3-8B"
            self.dim = 4096
            self.n_heads = 32
        elif "llama3-3b" in name_lower:
            self.model_name = "llama3-3B"
            self.dim = 3072
            self.n_heads = 24
        elif "llama3-1b" in name_lower:
            self.model_name = "llama3-1B"
            self.dim = 2048
            self.n_heads = 32
            self.head_dim = 64
        elif "llama3-70b" in name_lower:
            self.model_name = "llama3-70B"
            self.dim = 8192
            self.n_heads = 64
        elif "mistral" in name_lower:
            self.model_name = "mistral-7B"
            self.dim = 4096
            self.n_heads = 32
            self.n_kv_heads = 8
            self.vocab_size = 32000
            self.norm_eps = 1e-5
            self.hidden_size = 4096
        elif "qwen" in name_lower:
            self.model_name = "qwen2.5-7B"
            self.dim = 3584
            self.n_heads = 28
            self.n_kv_heads = 4
            self.vocab_size = 152064
            self.norm_eps = 1e-6
            self.hidden_size = 3584
        elif "mixtral" in name_lower:
            self.model_name = "mixtral-8x7B"
            self.dim = 4096
            self.n_heads = 32
            self.n_kv_heads = 8
            self.vocab_size = 32000
            self.norm_eps = 1e-5
            self.hidden_size = 4096
            self.moe = True
            self.num_experts = 8
            self.num_devices = 8

        # 3. Simulator/General Settings
        self.rms_norm_add_unit_offset = False
        self.max_batch_size = max_batch_size
        self.num_reduce_scatter_links = 1
        self.arch_name = ttnn.get_arch_name()
        self.compute_kernel_config_hifi2 = ttnn.MathFidelity.HiFi2
        self.compute_kernel_config_hifi2_fp16 = ttnn.MathFidelity.HiFi2
        self.compute_kernel_config_hifi4 = ttnn.MathFidelity.HiFi4
        self.max_grid_size = ttnn.CoreGrid([ttnn.CoreRange(ttnn.CoreCoord(0,0), ttnn.CoreCoord(7,7))])
        self.MAX_QKV_MM_SEQ_LEN = 2048
        self.num_all_gather_links = 1
        self.instruct = instruct
        self.qkv_size = 5120
        self.ccl_dtype = ttnn.bfloat8_b
        self.tile_size = 32
        self.min_kv_prefill_shard_seqlen = 256.0
        self.max_seq_len = max_seq_len
        self.rope_scaling_factor = None
        self.orig_context_len = None
        self.rope_theta = 1000000.0 if "qwen" in name_lower else 500000.0
        self.rope_scaling = None
        self.model_config = None
        self.is_multichip = False
        self.dummy_weights = True
        self.cluster_shape = [1,1]
        self.use_qk_fused = False
        self.use_fused_qkv_op = True
        self.query_pre_attn_scalar = None
        self.is_galaxy = False
        self.is_distributed_norm = False
        self.padded_vocab_size = None
        self.checkpoint_type = "simulation"
        self.WEIGHTS_DTYPE = ttnn.bfloat8_b

    def sdpa_prefill_program_config(self, seq_len):
        # tt-metal model_config SDPA_PROGCFG: 8x8 grid, accurate exp, 256-token chunks from 2k tokens up
        chunk = 256 if seq_len >= 2048 else 64
        return ttnn.SDPAProgramConfig(compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
                                      q_chunk_size=chunk, k_chunk_size=chunk, exp_approx_mode=False)

    def sdpa_decode_program_config(self):
        # SDPA_DECODE_PROGCFG: chunk 0 lets the kernel size the chunk at run time, cur_pos + 1 in tiles
        # rounded up to a power of two and capped at the DEST size (rt_args_common.hpp get_dynamic_Sk_chunk_t)
        return ttnn.SDPAProgramConfig(compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
                                      q_chunk_size=0, k_chunk_size=0, exp_approx_mode=False)

    def get_model_config(self):
        return {
            "SDPA_PROGCFG": self.sdpa_prefill_program_config,
            "SDPA_DECODE_PROGCFG": self.sdpa_decode_program_config(),
        }

    def weight_cache_path(self, dtype):
        return None

    def ccl_topology(self):
        return None
    
    def prepare_residual_tensor_decode(self, x, input_mem_cfg, args=None, force_replicated=False, on_host=False):
        batch = x.shape[0]
        seq_len = x.shape[1]
        assert x.shape[2] == self.dim

        x = ttnn.transpose(x, 0, 1).unsqueeze(0)
        x_shape = x.shape

        if args is not None:
            if args.moe:
                x = ttnn._rand(shape=(x_shape[0], seq_len, batch, x_shape[3]//args.num_experts),
                                device=x.device, dtype=ttnn.bfloat16) # mimics splitting the residual into num_experts parts

        return x

    def load_state_dict(self):
        return None
    
    def is_vision(self):
        return False
    
    def is_simulation(self):
        return True
