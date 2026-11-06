#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Standalone smoke test for the Qwen3.5-35B-A3B TTSIM workload.

Builds the forward graph for a tiny config and the full 35B-A3B decode
step, validates output shapes, checks the analytical parameter count
against the published ~35B figure, and dumps the workload graph.
"""

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from workloads.qwen3_5.qwen35_model import Qwen35Moe

# Qwen3.5-35B-A3B total parameters, as published by Qwen (~35B; exact count
# from the HF checkpoint README is 34.7B). See test below.
EXPECTED_35B_PARAMS = 34_700_000_000


def test_tiny_forward():
    tiny_cfg = {
        'vocab_size': 512, 'hidden_size': 64, 'num_hidden_layers': 8,
        'full_attention_interval': 4,
        'num_attention_heads': 4, 'num_key_value_heads': 2, 'head_dim': 32,
        'linear_num_value_heads': 4, 'linear_num_key_heads': 2,
        'linear_key_head_dim': 16, 'linear_value_head_dim': 16,
        'linear_conv_kernel_dim': 4,
        'num_experts': 8, 'num_experts_per_tok': 2,
        'moe_intermediate_size': 32, 'shared_expert_intermediate_size': 32,
        'bs': 1, 'seq_len': 4, 'rms_norm_eps': 1e-6,
    }
    model = Qwen35Moe('qwen35_tiny', tiny_cfg)
    model.create_input_tensors()
    out = model()
    assert list(out.shape) == [tiny_cfg['bs'], tiny_cfg['seq_len'], tiny_cfg['vocab_size']], \
        f'tiny output shape mismatch: {out.shape}'
    gg = model.get_forward_graph()
    assert gg is not None
    print(f'tiny forward OK: output={out.shape}, params={model.analytical_param_count():,}')


def test_35b_decode():
    model = Qwen35Moe('qwen35_35b_a3b_decode', {'bs': 1, 'seq_len': 1})
    model.create_input_tensors()
    out = model()
    assert list(out.shape) == [1, 1, 248320], f'35B decode output shape mismatch: {out.shape}'

    n_params = model.analytical_param_count()
    # sanity band around the published figure (avoids brittleness while catching
    # gross regressions in the param accounting)
    assert 0.95 * EXPECTED_35B_PARAMS < n_params < 1.05 * EXPECTED_35B_PARAMS, \
        f'35B param count off: got {n_params:,}, expected ~{EXPECTED_35B_PARAMS:,}'

    gg = model.get_forward_graph()
    assert gg is not None
    print(f'35B-A3B decode OK: output={out.shape}, params={n_params:,}')


def test_35b_prefill():
    model = Qwen35Moe('qwen35_35b_a3b_prefill', {'bs': 1, 'seq_len': 128})
    model.create_input_tensors()
    out = model()
    assert list(out.shape) == [1, 128, 248320], f'35B prefill output shape mismatch: {out.shape}'
    print(f'35B-A3B prefill(128) OK: output={out.shape}, params={model.analytical_param_count():,}')


if __name__ == '__main__':
    test_tiny_forward()
    test_35b_decode()
    test_35b_prefill()
    print('All Qwen3.5 workload tests passed!')