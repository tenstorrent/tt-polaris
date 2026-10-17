#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Tests for the Qwen3.5-35B-A3B TTSIM workload (workloads/qwen3_5/)."""
from typing import Any, Dict

import pytest

from workloads.qwen3_5.qwen35_model import Qwen35Moe

# Published total parameter count for Qwen/Qwen3.5-35B-A3B is ~35B (34.7B exact);
# the analytic model lands within 1% of it.
EXPECTED_35B_PARAMS = 34_700_000_000
PARAM_COUNT_TOLERANCE = 0.05

TINY_CFG: Dict[str, Any] = {
    'vocab_size': 512,
    'hidden_size': 64,
    'num_hidden_layers': 8,
    'full_attention_interval': 4,
    'num_attention_heads': 4,
    'num_key_value_heads': 2,
    'head_dim': 32,
    'linear_num_value_heads': 4,
    'linear_num_key_heads': 2,
    'linear_key_head_dim': 16,
    'linear_value_head_dim': 16,
    'linear_conv_kernel_dim': 4,
    'num_experts': 8,
    'num_experts_per_tok': 2,
    'moe_intermediate_size': 32,
    'shared_expert_intermediate_size': 32,
    'bs': 1,
    'seq_len': 4,
    'rms_norm_eps': 1e-6,
}


@pytest.mark.unit
def test_qwen35_tiny_forward():
    model = Qwen35Moe('qwen35_tiny', TINY_CFG)
    model.create_input_tensors()
    out = model()
    assert list(out.shape) == [TINY_CFG['bs'], TINY_CFG['seq_len'], TINY_CFG['vocab_size']]
    gg = model.get_forward_graph()
    assert gg is not None


@pytest.mark.unit
def test_qwen35_35b_param_count():
    model = Qwen35Moe('qwen35_35b_a3b', {'bs': 1, 'seq_len': 1})
    n_params = model.analytical_param_count()
    assert EXPECTED_35B_PARAMS * (1 - PARAM_COUNT_TOLERANCE) < n_params \
        < EXPECTED_35B_PARAMS * (1 + PARAM_COUNT_TOLERANCE), \
        f'param count {n_params:,} outside tolerance of {EXPECTED_35B_PARAMS:,}'


@pytest.mark.unit
def test_qwen35_35b_decode():
    model = Qwen35Moe('qwen35_35b_a3b_decode', {'bs': 1, 'seq_len': 1})
    model.create_input_tensors()
    out = model()
    assert list(out.shape) == [1, 1, 248320]
    gg = model.get_forward_graph()
    assert gg is not None


@pytest.mark.unit
def test_qwen35_35b_prefill():
    model = Qwen35Moe('qwen35_35b_a3b_prefill', {'bs': 1, 'seq_len': 32})
    model.create_input_tensors()
    out = model()
    assert list(out.shape) == [1, 32, 248320]
    gg = model.get_forward_graph()
    assert gg is not None


@pytest.mark.unit
def test_qwen35_layer_kind_counts():
    model = Qwen35Moe('qwen35_counts', {'bs': 1, 'seq_len': 1})
    assert model.n_linear == 30
    assert model.n_full == 10