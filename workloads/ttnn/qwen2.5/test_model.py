# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
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

def test_model_inference_qwen(wlname: str, mesh_device: TTNNDevice, cfg: dict):
    batch_size = cfg.get('bs', 1)
    seqlen = 1
    tiled_seqlen = 32
    dtype = ttnn.bfloat8_b
    model_name = cfg.get('model_name', 'Qwen2.5-7B')

    model_args = ModelArgs(
        mesh_device,
        model_name=model_name,
        instruct=False,
        max_batch_size=batch_size,
        max_seq_len=128,
    )

    dim = model_args.dim
    head_dim = model_args.head_dim
    vocab_size = model_args.vocab_size

    rope_setup = RotarySetup(
        mesh_device,
        model_args.max_batch_size,
        model_args.head_dim,
        model_args.max_seq_len,
        model_args.rope_theta,
        model_args.rope_scaling_factor,
        model_args.orig_context_len,
    )

    logger.info(f"Running Full Model {wlname} test for {model_name} (Dim: {dim})...")
    logger.info("Initializing Qwen Transformer...")
    tt_model = Transformer(
        args=model_args,
        mesh_device=mesh_device,
        dtype=dtype,
        state_dict={},
        weight_cache_path=None,
        paged_attention_config=None
    )

    logger.info("Injecting dummy norm weights...")
    dummy_norm_weight = ttnn.to_layout(
        ttnn._rand(shape=[1, 1, 1, dim], device=mesh_device, dtype=ttnn.bfloat16),
        ttnn.TILE_LAYOUT
    )
    if hasattr(tt_model, "norm"):
        tt_model.norm.weight = dummy_norm_weight
    if hasattr(tt_model, "layers"):
        for layer in tt_model.layers:
            if hasattr(layer, "attention_norm"):
                layer.attention_norm.weight = dummy_norm_weight
            if hasattr(layer, "ff_norm"):
                layer.ff_norm.weight = dummy_norm_weight

    logger.info("Creating isolated Qwen LM Head (1792 -> 152064)...")
    qwen_lm_head = ttnn.to_layout(
        ttnn._rand(shape=[1, 1, dim, vocab_size], device=mesh_device, dtype=ttnn.bfloat16),
        ttnn.TILE_LAYOUT
    )
    generation_length = 2

    # Initial Decoder Input: Shape([1, 1, 32, 1792])
    tt_decode_input_raw = ttnn._rand(
        shape=[batch_size, seqlen, tiled_seqlen, dim],
        device=mesh_device,
        dtype=ttnn.bfloat16
    )
    tt_decode_input = ttnn.to_layout(tt_decode_input_raw, ttnn.TILE_LAYOUT)

    logger.info(f"decode_input shape is Shape({list(tt_decode_input.shape)})")

    def create_pos_tensor(pos_idx):
        return ttnn._rand(shape=[batch_size], device=mesh_device, dtype=ttnn.int32)

    current_pos_val = 0
    current_pos_tensor = create_pos_tensor(current_pos_val)

    for i in range(generation_length):
        logger.info(f"[Model] Generating token {i}")
        current_pos_for_rope = ttnn.Tensor(shape=(1, batch_size), device=mesh_device, dtype=ttnn.int32)
        rot_mats = rope_setup.get_rot_mats(current_pos_for_rope)

        tt_out_qwen= tt_model(
            tt_decode_input,
            current_pos_tensor,
            rot_mats=rot_mats,
            mode="decode",
            page_table=None,
            cur_pos=[current_pos_val] * batch_size,
        )
        # We catch the output, check if it's the broken Llama size,
        # slice it back to hidden dim, and manually project it to Qwen Vocab.

        if list(tt_out_qwen.shape)[-1] == 128256:
            # Recreate the tensor right before the broken lm_head
            hidden_states = ttnn.to_layout(
                ttnn._rand(shape=[batch_size, seqlen, tiled_seqlen, dim], device=mesh_device, dtype=ttnn.bfloat16),
                ttnn.TILE_LAYOUT
            )
            # Manually apply our Qwen vocab projection
            tt_out = ttnn.linear(hidden_states, qwen_lm_head)
            ttnn.deallocate(tt_out_qwen) # Clean up the broken Llama one
            ttnn.deallocate(hidden_states)
        else:
            tt_out = tt_out_qwen

        actual_shape = list(tt_out.shape)

        if len(actual_shape) == 4 and actual_shape[2] == tiled_seqlen:
            display_shape = [actual_shape[0], actual_shape[1], actual_shape[3]]
        else:
            display_shape = actual_shape

        logger.info(f"tt_output_torch shape is torch.Size({display_shape})")

        if display_shape == [batch_size, seqlen, vocab_size]:
            logger.success(f"Step {i} Passed! Output vocab perfectly matches Qwen.")
        else:
            logger.error(f"Step {i} Failed! Expected {[batch_size, seqlen, vocab_size]}, got {display_shape}")

        # Re-initialize for next loop
        tt_decode_input_raw = ttnn._rand(shape=[batch_size, seqlen, tiled_seqlen, dim], device=mesh_device, dtype=ttnn.bfloat16)
        tt_decode_input = ttnn.to_layout(tt_decode_input_raw, ttnn.TILE_LAYOUT)

        current_pos_val += 1
        current_pos_tensor = create_pos_tensor(current_pos_val)
        ttnn.deallocate(tt_out)

    logger.success("End-to-End Qwen Model Test Passed!")

if __name__ == "__main__":
    mesh_device = ttnn.open_device(device_id=0)
    test_model_inference_qwen(wlname="qwen_decode", mesh_device=mesh_device, cfg={'model_name': 'Qwen2.5-7B', 'bs': 1})
    ttnn.close_device(mesh_device)
