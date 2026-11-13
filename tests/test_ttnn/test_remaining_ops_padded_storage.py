# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Padded-storage counting for the remaining ops: Tilize, TilizeWithValPadding, ConcatHeads,
CreateQKVHeads, Typecast, RotaryEmbeddingLlama(+FusedQK), NLP{Create,Concat}HeadsDecode, SDPA,
PagedFusedUpdateCache (covered in test_kv_cache_poc_perf). PlusOne has no front-end builder; ManualSeed and
Sampling act on row-major index tensors, so they count their logical shape.

TILE tensors are counted on whole 32x32 tiles (last two dims rounded up) on both the input and the
output side; ROW_MAJOR tensors are counted on their logical shape. Expected values are derived by hand.
"""

import pytest

from ttsim.front.ttnn.device import ARCH, Device
from ttsim.front.ttnn.tensor import DataType, Layout, Tensor, typecast
from ttsim.front.ttnn.ttnn_shim import (
    manual_seed_op,
    nlp_concat_heads_decode_op,
    nlp_concat_heads_op,
    nlp_create_qkv_heads_decode_op,
    nlp_create_qkv_heads_op,
    paged_fused_update_cache_op,
    rotary_embedding_llama_fused_qk_op,
    rotary_embedding_llama_op,
    sampling_op,
    scaled_dot_product_attention_op,
    tilize_op,
    tilize_with_val_padding_op,
)

BFP8_TILE = 1088
TILE = 1024


def _dev():
    d = Device(device_id=0)
    d.architecture = ARCH.BLACKHOLE
    return d


def _t(d, shape, name, dtype=DataType.BFLOAT16, layout=Layout.TILE_LAYOUT):
    return Tensor(name=name, shape=shape, dtype=dtype, layout=layout, device=d)


def _perf(d, optype):
    return [o for o in d.ops.values() if o.optype == optype][0].perf_stats


# ---- ConcatHeads: vit [8, 12, 197, 64] -> [8, 197, 768] ----------------------------------------
@pytest.mark.unit
def test_concat_heads_pads_seq_both_sides(monkeypatch):
    d = _dev()
    nlp_concat_heads_op(_t(d, [8, 12, 197, 64], "x"))
    p = _perf(d, "ConcatHeads")
    assert p["inElems"] == 8 * 12 * 224 * 64          # 197 -> 224 rows
    assert p["outElems"] == 8 * 224 * 768             # same tiles, regrouped
    assert p["inBytes"] == p["inElems"] * 2 and p["outBytes"] == p["outElems"] * 2


@pytest.mark.unit
def test_concat_heads_bfp8_tile_bytes(monkeypatch):
    d = _dev()
    nlp_concat_heads_op(_t(d, [8, 12, 197, 64], "x", DataType.BFLOAT8_B))
    p = _perf(d, "ConcatHeads")
    assert p["inBytes"] == 8 * 12 * 7 * 2 * BFP8_TILE == 1344 * BFP8_TILE
    assert p["outBytes"] == 8 * 7 * 24 * BFP8_TILE == 1344 * BFP8_TILE


# ---- CreateQKVHeads: [8, 197, 2304] -> Q,K,V [8,12,197,64] (K transposed [8,12,64,197]) --------
@pytest.mark.unit
@pytest.mark.parametrize("transpose_k", [False, True])
def test_create_qkv_heads_pads_each_output(monkeypatch, transpose_k):
    d = _dev()
    nlp_create_qkv_heads_op(_t(d, [8, 197, 2304], "x"), num_heads=12, transpose_k_heads=transpose_k)
    p = _perf(d, "CreateQKVHeads")
    assert p["inElems"] == 8 * 224 * 2304
    q = v = 8 * 12 * 224 * 64
    k = 8 * 12 * 64 * 224 if transpose_k else q      # K^T [.., 64, 197] pads the 197 columns to 224
    assert p["outElems"] == q + k + v
    assert p["outBytes"] == p["outElems"] * 2


# ---- Tilize / TilizeWithValPadding -------------------------------------------------------------
@pytest.mark.unit
def test_tilize_with_val_padding_bytes(monkeypatch):
    """vit patch embedding: row-major bf16 [8, 197, 768] -> TILE [8, 224, 768]."""
    d = _dev()
    tilize_with_val_padding_op(_t(d, [1, 8, 197, 768], "x", layout=Layout.ROW_MAJOR_LAYOUT),
                               [1, 8, 224, 768], 0.0, output_logical_shape=[1, 8, 197, 768])
    p = _perf(d, "TilizeWithValPadding")
    assert p["inElems"] == 8 * 197 * 768 and p["outElems"] == 8 * 224 * 768
    assert p["inBytes"] == p["inElems"] * 2 and p["outBytes"] == p["outElems"] * 2


@pytest.mark.unit
def test_tilize_output_bfp8_uses_tile_bytes(monkeypatch):
    d = _dev()
    x = _t(d, [1, 1, 1, 33], "x", DataType.BFLOAT8_B, layout=Layout.ROW_MAJOR_LAYOUT)
    tilize_op(x)
    p = _perf(d, "Tilize")
    assert p["outBytes"] == 2 * BFP8_TILE             # [1,1,32,64] = 2 tiles
    assert p["inBytes"] == int(33 * BFP8_TILE / TILE)  # row-major block-float: exponent overhead only


# ---- Typecast ----------------------------------------------------------------------------------
@pytest.mark.unit
def test_typecast_counts_each_side_with_its_own_dtype(monkeypatch):
    d = _dev()
    typecast(_t(d, [1, 1, 1, 33], "x", DataType.BFLOAT16), DataType.FLOAT32)
    p = _perf(d, "Typecast")
    assert p["inElems"] == p["outElems"] == 32 * 64
    assert p["inBytes"] == 2 * TILE * 2 == 4096        # 2 bf16 tiles
    assert p["outBytes"] == 2 * TILE * 4 == 8192       # 2 fp32 tiles


# ---- Rotary / decode head ops (decode batch 32, 8 kv heads, 32 q heads, head_dim 128) ----------
@pytest.mark.unit
def test_rotary_fused_qk_pads_heads_to_tile(monkeypatch):
    d = _dev()
    q = _t(d, [1, 32, 8, 128], "q")                    # 8 head rows pad to 32
    k = _t(d, [1, 32, 8, 128], "k")
    cos = _t(d, [1, 32, 1, 128], "cos")
    sin = _t(d, [1, 32, 1, 128], "sin")
    tm = _t(d, [1, 1, 32, 32], "tm")
    rotary_embedding_llama_fused_qk_op(q, k, cos, sin, tm)
    p = _perf(d, "RotaryEmbeddingLlamaFusedQK")
    qk = 32 * 32 * 128
    assert p["outElems"] == 2 * qk
    assert p["inElems"] == 2 * qk + 2 * (32 * 32 * 128) + 32 * 32


@pytest.mark.unit
def test_nlp_create_qkv_heads_decode_pads_outputs(monkeypatch):
    d = _dev()
    nlp_create_qkv_heads_decode_op(_t(d, [1, 1, 32, 6144], "x"), num_heads=32, num_kv_heads=8, head_dim=128)
    p = _perf(d, "NLPCreateQKVHeadsDecode")
    assert p["inElems"] == 32 * 6144
    assert p["outElems"] == 32 * 128 + 2 * (32 * 128)  # Q [..,32,128]; K,V [..,8,128] pad 8 -> 32 rows


@pytest.mark.unit
def test_nlp_concat_heads_decode_pads(monkeypatch):
    d = _dev()
    nlp_concat_heads_decode_op(_t(d, [1, 32, 8, 128], "x"), num_heads=8)
    p = _perf(d, "NLPConcatHeadsDecode")
    assert p["inElems"] == 32 * 32 * 128               # 8 rows -> 32
    assert p["outElems"] == 32 * 32 * 1024


@pytest.mark.unit
def test_sdpa_prefill_counts_padded(monkeypatch):
    d = _dev()
    q = _t(d, [1, 12, 197, 64], "q")
    scaled_dot_product_attention_op(q, _t(d, [1, 12, 197, 64], "k"), _t(d, [1, 12, 197, 64], "v"))
    p = _perf(d, "ScaledDotProductAttention")
    assert p["inElems"] == 3 * 12 * 224 * 64 and p["outElems"] == 12 * 224 * 64


# ---- Concat: inputs keep their TT dtype and TILE layout through concat_sinf's normalising copy ----
# concat_sinf counts inBytes on a detached copy of each input. The copy used to drop _ttnn_dtype and
# the layout, so TILE inputs were counted on their logical shape at the op precision.
@pytest.mark.unit
def test_concat_inputs_count_padded_tiles():
    # two bf16 TILE [1,1,1,33] -> each padded to [32,64] = 2 tiles = 4096 B (logical was 66 B)
    from ttsim.front.ttnn.op import concat

    d = _dev()
    concat(_t(d, [1, 1, 1, 33], "ca"), _t(d, [1, 1, 1, 33], "cb"), axis=2)
    p = _perf(d, "Concat")
    assert p["inBytes"] == 2 * 2 * 2048                # 8192 B (was 132 B)
    assert p["outBytes"] == 2 * 2048                   # [1,1,2,33] -> one tile row x 2 tiles


@pytest.mark.unit
@pytest.mark.parametrize("dtype, tile_bytes", [
    (DataType.BFLOAT16, 2048),
    (DataType.BFLOAT8_B, BFP8_TILE),
    (DataType.BFLOAT4_B, 576),
])
def test_concat_inputs_use_their_tt_dtype(dtype, tile_bytes):
    # two TILE [1,1,32,64] (2 tiles each) concatenated on dim 2 -> [1,1,64,64] (4 tiles)
    from ttsim.front.ttnn.op import concat

    d = _dev()
    concat(_t(d, [1, 1, 32, 64], "ca", dtype), _t(d, [1, 1, 32, 64], "cb", dtype), axis=2)
    p = _perf(d, "Concat")
    assert p["inBytes"] == 2 * 2 * tile_bytes          # bfp8 4352 B (was 16384 B at 4 B / element)
    assert p["outBytes"] == 4 * tile_bytes
