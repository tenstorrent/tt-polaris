# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import sys
import os
from loguru import logger

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
import ttsim.front.ttnn as ttnn
from workloads.ttnn.tt_transformers.model import Transformer
from workloads.ttnn.tt_transformers.model_config import ModelArgs
from workloads.ttnn.tt_transformers.rope import RotarySetup
from ttsim.front.ttnn.device import Device as TTNNDevice

class PagedAttentionConfig:
    def __init__(self, block_size, max_num_blocks):
        self.block_size = block_size
        self.max_num_blocks = max_num_blocks

def test_model_inference(wlname: str, mesh_device: TTNNDevice, cfg: dict):
    layers = 1
    max_seq_len = 128
    batch_size = cfg.get('bs', 1)
    model_name = cfg.get('model_name', 'Mistral-7B')

    # --- PAGED ATTENTION SETTINGS ---
    paged_attention = False
    page_block_size = 32
    page_max_num_blocks = 1024

    logger.info(f"Running inference test for {wlname} --> {model_name} (Paged Attention: {paged_attention})...")

    # --- SETUP MODEL ARGS (CLEANED) ---
    # Passing model_name here triggers the automatic config in model_config.py
    model_args = ModelArgs(mesh_device, model_name=model_name)

    # Only set non-standard/runtime overrides
    model_args.instruct = False
    model_args.max_batch_size = batch_size
    model_args.max_seq_len = max_seq_len
    model_args.n_layers = layers

    rope_setup = RotarySetup(
        mesh_device,
        model_args.max_batch_size,
        model_args.head_dim,
        model_args.max_seq_len,
        model_args.rope_theta,
        model_args.rope_scaling_factor,
        model_args.orig_context_len,
    )
    logger.info("RotarySetup Initialized.")

    pa_config = None
    page_table_tt = None

    if paged_attention:
        pa_config = PagedAttentionConfig(block_size=page_block_size, max_num_blocks=page_max_num_blocks)
        table_shape = [batch_size, page_max_num_blocks // batch_size]
        page_table_tt = ttnn.Tensor(shape=table_shape, device=mesh_device, dtype=ttnn.int32)
        logger.info("Paged Attention Configured.")

    logger.info(f"Initializing Transformer (Dim: {model_args.dim})...")
    tt_model = Transformer(
        args=model_args,
        mesh_device=mesh_device,
        dtype=ttnn.bfloat16,
        state_dict={},
        weight_cache_path=None,
        paged_attention_config=pa_config
    )
    logger.info("Model loaded.")

    seqlen = 1
    batch = batch_size
    generation_length = 2

    tt_decode_input = ttnn.Tensor(shape=[batch, seqlen, model_args.dim], device=mesh_device, dtype=ttnn.bfloat16)

    def create_pos_tensor(pos_idx):
        return ttnn.Tensor(shape=[batch], device=mesh_device, dtype=ttnn.int32)

    current_pos_val = 0
    current_pos_tensor = create_pos_tensor(current_pos_val)

    for i in range(generation_length):
        logger.info(f"[Model] Generating token {i}")

        current_pos_for_rope = ttnn.Tensor(shape=(1, batch), device=mesh_device, dtype=ttnn.int32)
        rot_mats = rope_setup.get_rot_mats(current_pos_for_rope)

        tt_out = tt_model(
            tt_decode_input,
            current_pos_tensor,
            rot_mats=rot_mats,
            mode="decode",
            page_table=page_table_tt,
            cur_pos=[current_pos_val] * batch,
        )

        expected_shape = [1, batch, seqlen, model_args.vocab_size]
        actual_shape = list(tt_out.shape)

        if actual_shape == expected_shape:
            logger.success(f"[Model] Token {i} Passed! Output shape matches logits: {actual_shape}")
        else:
            logger.error(f"[Model] Token {i} Failed! Expected shape: {expected_shape}, got: {actual_shape}")

        tt_decode_input = ttnn.Tensor(shape=[batch, seqlen, model_args.dim], device=mesh_device, dtype=ttnn.bfloat16)

        current_pos_val += 1
        current_pos_tensor = create_pos_tensor(current_pos_val)

    logger.info("Test Completed!")

if __name__ == "__main__":
    ttnn_device = ttnn.open_device(device_id=0)
    test_model_inference(wlname="mistral_decode", mesh_device=ttnn_device, cfg={'bs': 1, 'model_name': 'Mistral-7B'})
    ttnn.close_device(ttnn_device)
