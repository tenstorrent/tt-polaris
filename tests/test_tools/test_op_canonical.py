#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for tools.profiling.op_canonical — the single source of truth
for operation-type normalization."""

import pytest
from tools.profiling.op_canonical import (
    BINARY_OP_ENUM_TO_CANONICAL,
    COMPARISON_GROUPS,
    POLARIS_SYNONYMS,
    UNARY_OP_ENUM_TO_CANONICAL,
    canonical_to_stats_display,
    normalize_polaris_optype,
    normalize_profiler_opcode,
    to_comparison_group,
)


# ── normalize_profiler_opcode: DeviceOperation suffix stripping ──────────


@pytest.mark.unit
@pytest.mark.parametrize(
    "raw,expected",
    [
        ("MatmulDeviceOperation", "matmul"),
        ("SoftmaxDeviceOperation", "softmax"),
        ("ReshapeViewDeviceOperation", "reshape"),
        ("ReshapeDeviceOperation", "reshape"),
        ("TransposeDeviceOperation", "transpose"),
        ("PermuteDeviceOperation", "permute"),
        ("TilizeDeviceOperation", "tilize"),
        ("TilizeWithValPaddingDeviceOperation", "tilizewithvalpadding"),
        ("UntilizeDeviceOperation", "untilize"),
        ("UntilizeWithUnpaddingDeviceOperation", "untilizewithunpadding"),
        ("FoldDeviceOperation", "fold"),
        ("ConcatDeviceOperation", "concat"),
        ("EmbeddingDeviceOperation", "embedding"),
    ],
)
def test_profiler_device_operation_suffix_stripping(raw, expected):
    assert normalize_profiler_opcode(raw) == expected


# ── normalize_profiler_opcode: BinaryNg / attribute resolution ───────────


@pytest.mark.unit
@pytest.mark.parametrize(
    "attrs,expected",
    [
        ({"binary_op_type": "BinaryOpType::ADD"}, "add"),
        ({"binary_op_type": "BinaryOpType::MUL"}, "mul"),
        ({"binary_op_type": "BinaryOpType::SUB"}, "subtract"),
        ({"binary_op_type": "BinaryOpType::DIV"}, "divide"),
    ],
)
def test_binary_ng_attribute_resolution(attrs, expected):
    assert normalize_profiler_opcode("BinaryNgDeviceOperation", attrs) == expected


@pytest.mark.unit
def test_binary_ng_falls_back_to_eltwise_without_attrs():
    assert normalize_profiler_opcode("BinaryNgDeviceOperation") == "eltwise"


# ── normalize_profiler_opcode: Unary / attribute resolution ──────────────


@pytest.mark.unit
@pytest.mark.parametrize(
    "attrs,expected",
    [
        ({"unary_op_type": "UnaryOpType::GELU"}, "gelu"),
        ({"unary_op_type": "UnaryOpType::RELU"}, "relu"),
        ({"unary_op_type": "UnaryOpType::SIGMOID"}, "sigmoid"),
        ({"unary_op_type": "UnaryOpType::TANH"}, "tanh"),
        ({"unary_op_type": "UnaryOpType::SILU"}, "silu"),
    ],
)
def test_unary_attribute_resolution(attrs, expected):
    assert normalize_profiler_opcode("UnaryDeviceOperation", attrs) == expected


@pytest.mark.unit
def test_unary_op_chain_resolution():
    attrs = {
        "op_chain": (
            "{BasicUnaryWithParam<float; int; unsigned int>("
            "base=BasicUnaryWithParam<float>(op_type=UnaryOpType::GELU;param={0}))}"
        ),
        "output_dtype": "DataType::BFLOAT16",
    }
    assert normalize_profiler_opcode("UnaryDeviceOperation", attrs) == "gelu"


# ── normalize_profiler_opcode: bare opcode strings ───────────────────────


@pytest.mark.unit
@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Matmul", "matmul"),
        ("ADD", "add"),
        ("MUL", "mul"),
        ("Softmax", "softmax"),
        ("LayerNorm", "layernorm"),
        ("RMSNorm", "rms_norm"),
        ("Gelu", "gelu"),
        ("Relu", "relu"),
        # llama3/transformer profiler op codes (real DeviceOperation forms).
        ("RotaryEmbeddingLlamaFusedQKDeviceOperation", "rotaryembeddingllamafusedqk"),
        ("RotaryEmbeddingLlamaDeviceOperation", "rotaryembeddingllama"),
        ("PagedFusedUpdateCacheDeviceOperation", "pagedfusedupdatecache"),
        ("PagedFillCacheDeviceOperation", "pagedfillcache"),
        # Prefill head split: the profiler spells it NlpCreateHeads (no "QKV"), BH p100a 260923/260924.
        ("NlpCreateHeadsDeviceOperation", "createqkvheads"),
        ("CopyDeviceOperation", "copy"),
        ("SDPAOperation", "scaleddotproductattention"),
        ("SdpaDecodeDeviceOperation", "scaleddotproductattention"),
    ],
)
def test_bare_opcode_prefix_matching(raw, expected):
    assert normalize_profiler_opcode(raw) == expected


# ── normalize_profiler_opcode: edge cases ────────────────────────────────


@pytest.mark.unit
def test_numeric_opcode():
    assert normalize_profiler_opcode(42) == "numeric"
    assert normalize_profiler_opcode(3.0) == "numeric"


@pytest.mark.unit
def test_empty_and_none():
    assert normalize_profiler_opcode(None) == ""
    assert normalize_profiler_opcode("") == ""
    assert normalize_profiler_opcode(float("nan")) == ""


# ── normalize_polaris_optype ─────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.parametrize(
    "raw,expected",
    [
        ("LayerNormalization", "layernorm"),
        ("ReshapeView", "reshape"),
        ("CreateQKVHeads", "createqkvheads"),
        ("MatMul", "matmul"),
        ("Add", "add"),
        ("Softmax", "softmax"),
    ],
)
def test_normalize_polaris_optype(raw, expected):
    assert normalize_polaris_optype(raw) == expected


# ── to_comparison_group ──────────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.parametrize(
    "canonical,expected",
    [
        ("add", "binary"),
        ("mul", "binary"),
        ("subtract", "binary"),
        ("eltwise", "binary"),
        ("tilizewithvalpadding", "tilize"),
        ("matmul", "matmul"),
        ("softmax", "softmax"),
    ],
)
def test_to_comparison_group(canonical, expected):
    assert to_comparison_group(canonical) == expected


# ── canonical_to_stats_display ───────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.parametrize(
    "canonical,expected",
    [
        ("matmul", "MatMul"),
        ("add", "Add"),
        ("mul", "Mul"),
        ("reshape", "Reshape"),
        ("createqkvheads", "CreateQKVHeads"),
    ],
)
def test_canonical_to_stats_display(canonical, expected):
    assert canonical_to_stats_display(canonical) == expected


@pytest.mark.unit
def test_unknown_canonical_capitalised():
    assert canonical_to_stats_display("unknown_op") == "Unknown_op"


# ── consistency checks ───────────────────────────────────────────────────


@pytest.mark.unit
def test_all_binary_enums_have_canonical():
    for enum_key, canonical in BINARY_OP_ENUM_TO_CANONICAL.items():
        assert isinstance(canonical, str) and len(canonical) > 0, (
            f"BINARY_OP_ENUM_TO_CANONICAL[{enum_key!r}] is invalid"
        )


@pytest.mark.unit
def test_all_unary_enums_have_canonical():
    for enum_key, canonical in UNARY_OP_ENUM_TO_CANONICAL.items():
        assert isinstance(canonical, str) and len(canonical) > 0, (
            f"UNARY_OP_ENUM_TO_CANONICAL[{enum_key!r}] is invalid"
        )


@pytest.mark.unit
def test_comparison_groups_are_lowercase():
    for key, group in COMPARISON_GROUPS.items():
        assert key == key.lower(), f"COMPARISON_GROUPS key {key!r} not lowercase"
        assert group == group.lower(), f"COMPARISON_GROUPS value {group!r} not lowercase"


@pytest.mark.unit
def test_polaris_synonyms_are_lowercase():
    for key, canonical in POLARIS_SYNONYMS.items():
        assert key == key.lower(), f"POLARIS_SYNONYMS key {key!r} not lowercase"
        assert canonical == canonical.lower(), f"POLARIS_SYNONYMS value {canonical!r} not lowercase"


@pytest.mark.unit
def test_resolve_conv_subtype_dict_attrs_true():
    """Dict attrs with is_transpose=True must return 'convtranspose' without relying on str()."""
    from tools.profiling.op_canonical import _resolve_conv_subtype

    assert _resolve_conv_subtype({'is_transpose': True}) == 'convtranspose'


@pytest.mark.unit
def test_resolve_conv_subtype_dict_attrs_false():
    from tools.profiling.op_canonical import _resolve_conv_subtype

    assert _resolve_conv_subtype({'is_transpose': False}) == 'conv2d'
    assert _resolve_conv_subtype({}) == 'conv2d'


@pytest.mark.unit
def test_resolve_conv_subtype_string_attrs():
    """String ATTRIBUTES from profiler CSV still resolve correctly via regex."""
    from tools.profiling.op_canonical import _resolve_conv_subtype

    sliding_window_false = 'SlidingWindowConfig { kernel_size: [3, 3], is_transpose = false }'
    sliding_window_true = 'SlidingWindowConfig { kernel_size: [2, 2], is_transpose = true }'
    assert _resolve_conv_subtype(sliding_window_false) == 'conv2d'
    assert _resolve_conv_subtype(sliding_window_true) == 'convtranspose'
