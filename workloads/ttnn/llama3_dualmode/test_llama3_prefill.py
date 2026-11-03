#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Dual-mode llama3 PREFILL runner.

Uses the dual-mode workloads.ttnn.tt_transformers_dualmode components (which run on real HW
via real ttnn when IRD_ARCH_NAME is set, and analytically via the TT-Polaris shim otherwise).
The existing shim-only workloads/ttnn/llama3 + workloads/ttnn/tt_transformers are left
untouched. Builds the prefill graph on-device; TT-Polaris extracts the WorkloadGraph from the
device after this returns.

Config (paged_attention=1, num_devices=1, dummy weights) is pinned per the migration plan;
graph is grounded in the BH prefill capture (project_llama3_prefill_op_sequence).
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
from workloads.ttnn.tt_transformers_dualmode.rope import get_prefill_rot_mat, prefill_rope_table_len

_NUM_LAYERS = {'llama3-8B': 32, 'llama3-3B': 28, 'llama3-1B': 16, 'llama3-70B': 80}


class PagedAttentionConfig:
    """Minimal paged-attention config (block_size + max_num_blocks) for the KV cache shape."""
    def __init__(self, block_size=32, max_num_blocks=1024):
        self.block_size = block_size
        self.max_num_blocks = max_num_blocks


def run_llama3_dualmode(wlname: str, ttnn_device, cfg: dict):
    assert isinstance(cfg, dict), 'cfg must be a dictionary'
    assert isinstance(wlname, str), 'wlname must be a string'

    model_name = cfg.get('model_name', 'llama3-8B')
    seq_len = cfg.get('seq_len', 128)
    assert seq_len % 128 == 0, 'prefill seq_len must be divisible by 128'
    batch_size = 1  # prefill supports batch_size = 1 only
    num_layers = cfg.get('n_layers', _NUM_LAYERS.get(model_name, 32))
    max_seq_len = cfg.get('max_seq_len', 128 * 1024)
    # Prefill tail. tt-metal traces only the embedding + decoder blocks; after each replay it runs
    # the last-token norm + lm_head and on-device sampling eagerly. `lm_head: false` stops after the
    # blocks (the traced forward alone); `sampling` (default: follows lm_head) adds the greedy
    # force-argmax sampler (Untilize -> ArgMax), which the demo's greedy settings select.
    with_lm_head = cfg.get('lm_head', True)
    with_sampling = cfg.get('sampling', with_lm_head)
    assert with_lm_head or not with_sampling, 'sampling needs the lm_head logits (lm_head: true)'
    # Traced prefill lengths that size the cos/sin tables (see rope.prefill_rope_table_len); model- and
    # device-specific in tt-metal. Llama-3.1-8B on N150 and P100: [128, 1024]. Unset = max_seq_len rows.
    rope_trace_prefill_seq_lens = cfg.get('rope_trace_prefill_seq_lens')
    # Real prompt length (<= seq_len, which is the 128-padded length). Only picks which 32-row tile
    # the tail slices; the reference capture's prompt is 80 tokens (last token in tile 64..96).
    prompt_len = cfg.get('prompt_len', seq_len)
    assert 0 < prompt_len <= seq_len, f'prompt_len ({prompt_len}) must be in 1..seq_len ({seq_len})'
    dtype = ttnn.bfloat8_b

    paged_attention_config = PagedAttentionConfig(
        block_size=cfg.get('page_block_size', 32),
        max_num_blocks=cfg.get('page_max_num_blocks', 1024),
    )

    logger.info(f'Loading dual-mode TT model {model_name} ({num_layers} layers, seq_len={seq_len})...')
    args = ModelArgs(
        ttnn_device, model_name=model_name, instruct=True,
        max_batch_size=batch_size, max_seq_len=max_seq_len,
    )
    args.n_layers = num_layers

    state_dict: dict = {}
    model = Transformer(
        args=args, mesh_device=ttnn_device, dtype=dtype, state_dict=state_dict,
        weight_cache_path=None, paged_attention_config=paged_attention_config,
        use_paged_kv_cache=False,
    )

    # Encoded prompt -> embedding -> 4D hidden state (the runner unsqueezes; model.forward does not).
    # Tokens are uint32 to match the HW EmbeddingsDeviceOperation index dtype (and the HW branch
    # below), so the embedding op's input dtype / LUT key is identical across both modes.
    if IS_POLARIS:
        tokens = ttnn._rand(shape=[batch_size, seq_len], device=ttnn_device, dtype=ttnn.uint32)
    else:
        import torch  # type: ignore[import-not-found]
        tokens = ttnn.from_torch(
            torch.randint(0, args.vocab_size, (batch_size, seq_len)), dtype=ttnn.uint32, device=ttnn_device,
        )
    x = model.embd(tokens)
    x = ttnn.unsqueeze_to_4D(x)

    # tt-metal prepare_inputs_prefill slices the whole cos/sin table for traced prefill ("we don't create a
    # new tensor"), and pads it for the traced prefill lengths -> RotaryEmbeddingLlama sees
    # [1,1,table_len,head_dim] (both captures: [1,1,2048,128] at max_seq_len=1024), not [1,1,seq_len,head_dim].
    rope_table_len = prefill_rope_table_len(max_seq_len, rope_trace_prefill_seq_lens)
    rot_mats = get_prefill_rot_mat(
        args.head_dim, ttnn_device, rope_table_len, theta=args.rope_theta,
        scale_factor=args.rope_scaling_factor, orig_context_len=args.orig_context_len,
    )

    # Paged KV cache: PagedFillCache takes the page table as its third operand (capture
    # (1,1,1,max_num_blocks) ROW_MAJOR INT32). Passing None dropped that operand.
    if IS_POLARIS:
        page_table = ttnn._rand(shape=[batch_size, paged_attention_config.max_num_blocks],
                                device=ttnn_device, dtype=ttnn.int32)
    else:
        import torch  # type: ignore[import-not-found]
        page_table = ttnn.from_torch(
            torch.arange(paged_attention_config.max_num_blocks, dtype=torch.int32).reshape(batch_size, -1),
            dtype=ttnn.int32, device=ttnn_device,
        )

    tt_out = model.ttnn_prefill_forward(x, rot_mats=rot_mats, user_id=0, page_table=page_table,
                                        get_last_token=prompt_len - 1 if with_lm_head else -1)
    # Logits cover the last token's 32-row tile only (tt-metal slices it before the norm).
    expected = [1, 1, 32, args.vocab_size] if with_lm_head else [1, 1, seq_len, args.dim]
    if with_sampling:
        # Greedy force-argmax tail: Untilize -> ArgMax over the vocab (one token id per row).
        from workloads.ttnn.tt_transformers_dualmode.sampling import sample_argmax
        tt_out = sample_argmax(tt_out)
        expected = [1, 1, 32]
    if list(tt_out.shape) == expected:
        logger.info(f'Output shape is correct {expected}')
    else:
        logger.warning(f'Output shape {list(tt_out.shape)} != expected {expected}')
    logger.info(f'Finished dual-mode prefill for {model_name}.')


if __name__ == '__main__':
    name = sys.argv[1] if len(sys.argv) > 1 else 'llama3-8B'
    dev = ttnn.open_device(device_id=0)
    # Full prefill step, as instances llama3_8b_dualmode_prefill_bh / _wh (the tt-metal f8c634a captures on
    # BH p100a and WH n150 use the same settings). prompt_len 80 = the demo prompt.
    cfg = {'model_name': name, 'n_layers': 1, 'max_seq_len': 1024, 'prompt_len': 80,
           'rope_trace_prefill_seq_lens': [128, 1024]}
    run_llama3_dualmode(wlname='llama3_dualmode', ttnn_device=dev, cfg=cfg)
