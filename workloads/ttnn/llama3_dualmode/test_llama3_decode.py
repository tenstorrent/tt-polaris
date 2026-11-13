#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Dual-mode llama3 DECODE runner.

Companion to test_llama3_prefill.py, using the same workloads.ttnn.tt_transformers_dualmode
components in decode mode (seqlen=1 per user, batch users, paged KV cache). The decode path
in attention.forward_decode emits NLPCreateQKVHeadsDecode -> RotaryEmbeddingLlamaFusedQK ->
PagedFusedUpdateCache -> SdpaDecode -> NLPConcatHeadsDecode, and rope.get_rot_mats supplies the
decode cos/sin via embedding + transpose + interleaved_to_sharded — matching the BH decode
capture (project_llama3_prefill_op_sequence). Builds the decode graph on-device; TT-Polaris
extracts the WorkloadGraph after this returns.
"""
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from loguru import logger

IS_POLARIS = os.getenv('IRD_ARCH_NAME', '') == ''
if IS_POLARIS:
    import ttsim.front.ttnn as ttnn
else:
    import ttnn  # type: ignore[import-not-found, no-redef]

from workloads.ttnn.tt_transformers_dualmode.model import Transformer
from workloads.ttnn.tt_transformers_dualmode.model_config import ModelArgs

_NUM_LAYERS = {'llama3-8B': 32, 'llama3-3B': 28, 'llama3-1B': 16, 'llama3-70B': 80}


class PagedAttentionConfig:
    def __init__(self, block_size=32, max_num_blocks=1024):
        self.block_size = block_size
        self.max_num_blocks = max_num_blocks


def run_llama3_dualmode_decode(wlname: str, ttnn_device, cfg: dict):
    assert isinstance(cfg, dict), 'cfg must be a dictionary'
    assert isinstance(wlname, str), 'wlname must be a string'

    model_name = cfg.get('model_name', 'llama3-8B')
    batch = cfg.get('bs', 32)            # decode generates one token per user across `batch` users
    num_layers = cfg.get('n_layers', _NUM_LAYERS.get(model_name, 32))
    dtype = ttnn.bfloat8_b
    seqlen = 1

    paged_attention_config = PagedAttentionConfig(
        block_size=cfg.get('page_block_size', 32),
        max_num_blocks=cfg.get('page_max_num_blocks', 1024),
    )

    logger.info(f'Loading dual-mode TT model {model_name} ({num_layers} layers, decode batch={batch})...')
    args = ModelArgs(
        ttnn_device, model_name=model_name, instruct=False,
        max_batch_size=batch, max_seq_len=cfg.get('max_seq_len', 4096), dummy_weights=True,
    )
    args.n_layers = num_layers
    if IS_POLARIS:
        # Decode reads the paged KV cache for positions 0..max_seq_len-1; the SDPA/KV-cache perf model
        # sizes its traffic from this position (the last slot, the worst case for max_seq_len).
        ttnn_device.kv_cache_pos = args.max_seq_len - 1

    state_dict = None
    model = Transformer(
        args=args, mesh_device=ttnn_device, dtype=dtype, state_dict=state_dict,
        weight_cache_path=None, paged_attention_config=paged_attention_config,
        use_paged_kv_cache=False,
    )

    # HW decode runs batch-1 (the reference: simple_text_demo -k "performance and batch-1",
    # seq_len=1024) and tile-pads to 32 user-slots BY DESIGN — the activation and the batch-carrying
    # index tensors (tokens, rope rot_idxs) are 32-wide even for a single user (capture activation
    # logical-y=32), while the paged index tensors (current_pos/page_table) address the ACTIVE batch.
    # So act_users=32 is the correct PERMANENT model of HW tile-padding, NOT a temporary bs=1
    # workaround: decode is demoed at batch-1, there is no batch-32 decode capture. Pad the
    # batch-carrying activation/index tensors to act_users; leave paged index tensors at `batch`.
    act_users = ((batch + 31) // 32) * 32
    # One token per user (activation tile-padded to act_users) -> embedding -> [1,1,act_users,dim].
    # Tokens are uint32 to match the HW EmbeddingsDeviceOperation index dtype (capture input_0 UINT32).
    if IS_POLARIS:
        tokens = ttnn._rand(shape=[1, act_users], device=ttnn_device, dtype=ttnn.uint32)
    else:
        import torch  # type: ignore[import-not-found]
        tokens = ttnn.from_torch(
            torch.randint(0, args.vocab_size, (1, act_users)), dtype=ttnn.uint32, device=ttnn_device,
        )
    x = model.embd(tokens)
    x = ttnn.unsqueeze_to_4D(x)

    # Per-user current positions ([1, batch]) — active-batch index into the paged KV cache
    # (feeds fused-update-cache / paged-SDPA), so it stays at the active `batch`, not act_users.
    if IS_POLARIS:
        current_pos = ttnn._rand(shape=[1, batch], device=ttnn_device, dtype=ttnn.int32)
    else:
        import torch  # type: ignore[import-not-found]
        current_pos = ttnn.from_torch(torch.zeros(1, batch, dtype=torch.int32), dtype=ttnn.int32, device=ttnn_device)

    # Separate rope-index counter — real llama3 decode keeps rot_mat_idxs distinct from the
    # KV-cache current_pos and advances both per step (HW emits PlusOne x2). Tile-padded to
    # act_users (32) and uint32 to match the HW rope EmbeddingsDeviceOperation index (capture
    # input_0 = [1,32] UINT32), same treatment as `tokens`; get_rot_idxs pads/casts identically.
    if IS_POLARIS:
        rot_idxs = ttnn._rand(shape=[1, act_users], device=ttnn_device, dtype=ttnn.uint32)
    else:
        import torch  # type: ignore[import-not-found]
        rot_idxs = ttnn.from_torch(torch.zeros(1, act_users, dtype=torch.int32), dtype=ttnn.uint32, device=ttnn_device)
    rot_mats = model.rope_setup.get_rot_mats(rot_idxs)

    # Paged KV-cache page table [batch, max_num_blocks] (int32). Real paged decode addresses the
    # cache through it, and the fused-update-cache + paged-SDPA ops take it as an operand (capture:
    # page_table (1,1,1,max_num_blocks) INT32). Passing None dropped it, making those ops emit one
    # fewer operand than hardware (arity mismatch under full-arity LUT keys).
    if IS_POLARIS:
        page_table = ttnn._rand(shape=[batch, paged_attention_config.max_num_blocks],
                                device=ttnn_device, dtype=ttnn.int32)
    else:
        import torch  # type: ignore[import-not-found]
        page_table = ttnn.from_torch(
            torch.zeros(batch, paged_attention_config.max_num_blocks, dtype=torch.int32),
            dtype=ttnn.int32, device=ttnn_device,
        )

    tt_out = model(x, current_pos, rot_mats=rot_mats, mode='decode', page_table=page_table)
    logger.info(f'Decode output shape {list(tt_out.shape)}')

    # Advance position counters for the next decode step (matches HW PlusOne x2).
    ttnn.plus_one(current_pos, skip_negative_entries=True)
    ttnn.plus_one(rot_idxs)

    # Sampling head: extend the graph through the decode sampling tail so it spans
    # the full HW capture (logits -> sampled token), not just lm_head.
    from workloads.ttnn.tt_transformers_dualmode.sampling import sample_decode
    tt_tok = sample_decode(tt_out, ttnn_device)
    logger.info(f'Decode sampled token shape {list(tt_tok.shape)}')
    logger.info(f'Finished dual-mode decode for {model_name}.')


if __name__ == '__main__':
    name = sys.argv[1] if len(sys.argv) > 1 else 'llama3-8B'
    dev = ttnn.open_device(device_id=0)
    # Standalone HW dual-mode run: match the batch-1 / seq_len=1024 reference (the run-llama3dual
    # preset is "b1"). Without bs, batch defaulted to 32 -> fused-QK doubled_batch=64 overran the
    # 32-wide rope cos/sin. n_layers=1 keeps the failcheck/quick pass fast.
    run_llama3_dualmode_decode(wlname='llama3_dualmode', ttnn_device=dev,
                               cfg={'model_name': name, 'n_layers': 1, 'bs': 1, 'max_seq_len': 1024})
