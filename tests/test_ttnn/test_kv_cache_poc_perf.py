# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Tests for the KV-cache perf model (always on; decode SDPA needs the device's ``kv_cache_pos``).

Shapes and dtypes mirror the llama3-8B dual-mode decode instance
``llama3_8b_dualmode_decode_b1`` (bs=1, max_seq_len=1024), as recorded from the real workload:

    k/v caches      [1024, 8, 32, 128]  BFLOAT8_B   (num_blocks, kv_heads, block_size, head_dim)
    new K / V       [1, 1, 8, 128]      BFLOAT16
    Q               [1, 1, 32, 128]     BFLOAT16
    update_idxs     [1, 1]              INT32        (cur_pos)
    page_table      [1, 1024]           INT32

Every expected value is derived by hand from the formulas below, NOT by calling the helpers under
test. Bytes per element are exact fractions: bfp8 = 1088/1024 = 1.0625, bf16 = 2, int32 = 4.

    tile = 32 x 32 = 1024 elems
    update : tiles = users * kv_heads * ceil(head_dim/32)
    fill   : tiles = prod(lead dims) * ceil(rows/32) * ceil(cols/32)
    sdpa   : positions = 32 * ceil((POS + 1) / 32)
"""

import math

import pytest

from ttsim.front.ttnn.device import ARCH, Device
from ttsim.front.ttnn.tensor import DataType, Layout, Tensor
from ttsim.front.ttnn.ttnn_shim import (
    paged_fill_cache_op,
    paged_fused_update_cache_op,
    scaled_dot_product_attention_op,
)

# ---- workload constants (llama3-8B, bs=1, max_seq_len=1024) -----------------------------------
NUM_BLOCKS, KV_HEADS, BLOCK_SIZE, HEAD_DIM = 1024, 8, 32, 128
Q_HEADS = 32
USERS = 1
TILE_ELEMS = 32 * 32

BFP8 = 1088 / 1024   # bytes / element, BFLOAT8_B (1024 mantissa + 64 shared exponents per tile)
BF16 = 2
INT32 = 4


# ---- helpers ----------------------------------------------------------------------------------
def _device(pos=1023):
    """Device as the llama3 decode workload leaves it: kv_cache_pos = max_seq_len - 1 = 1023
    (pass ``pos=None`` for a workload that declares no decode position)."""
    device = Device(device_id=0)
    device.architecture = ARCH.BLACKHOLE
    if pos is not None:
        device.kv_cache_pos = pos
    return device


def _t(device, shape, name, dtype=DataType.BFLOAT16, layout=Layout.TILE_LAYOUT):
    return Tensor(name=name, shape=shape, dtype=dtype, layout=layout, device=device)


def _idx(device, shape, name):
    return _t(device, shape, name, dtype=DataType.INT32, layout=Layout.ROW_MAJOR_LAYOUT)


def _kv_caches(device, tag):
    shape = [NUM_BLOCKS, KV_HEADS, BLOCK_SIZE, HEAD_DIM]
    return (_t(device, shape, f"{tag}_kc", dtype=DataType.BFLOAT8_B),
            _t(device, shape, f"{tag}_vc", dtype=DataType.BFLOAT8_B))


def _update_op(device, tag="u"):
    """Decode PagedFusedUpdateCache exactly as the workload emits it; returns the SimOp."""
    kc, vc = _kv_caches(device, tag)
    k = _t(device, [1, USERS, KV_HEADS, HEAD_DIM], f"{tag}_k")
    v = _t(device, [1, USERS, KV_HEADS, HEAD_DIM], f"{tag}_v")
    upd = _idx(device, [1, USERS], f"{tag}_upd")
    pt = _idx(device, [USERS, NUM_BLOCKS], f"{tag}_pt")
    paged_fused_update_cache_op(kc, k, vc, v, update_idxs_tensor=upd, page_table=pt)
    return [o for o in device.ops.values() if o.optype == "PagedFusedUpdateCache"][0]


def _sdpa_decode_op(device, tag="s"):
    """Decode SDPA (5 inputs: q, k_cache, v_cache, cur_pos, page_table); returns the SimOp."""
    kc, vc = _kv_caches(device, tag)
    q = _t(device, [1, USERS, Q_HEADS, HEAD_DIM], f"{tag}_q")
    cur_pos = _idx(device, [1, USERS], f"{tag}_cp")
    pt = _idx(device, [USERS, NUM_BLOCKS], f"{tag}_pt")
    scaled_dot_product_attention_op(q, kc, vc, cur_pos, pt)
    return [o for o in device.ops.values() if o.optype == "ScaledDotProductAttention"][0]


def _fill_op(device, prompt_len, tag="f", inp_dtype=DataType.BFLOAT8_B):
    """Prefill PagedFillCache for one cache; returns the SimOp."""
    cache = _t(device, [NUM_BLOCKS, KV_HEADS, BLOCK_SIZE, HEAD_DIM], f"{tag}_c", dtype=DataType.BFLOAT8_B)
    inp = _t(device, [1, KV_HEADS, prompt_len, HEAD_DIM], f"{tag}_in", dtype=inp_dtype)
    pt = _idx(device, [1, 1, 1, prompt_len // BLOCK_SIZE], f"{tag}_pt")
    paged_fill_cache_op(cache, inp, page_table=pt, batch_idx=0)
    return [o for o in device.ops.values() if o.optype == "PagedFillCache"][0]


def _positions(pos):
    return 32 * math.ceil((pos + 1) / 32)


# =========================================================================
# PagedFusedUpdateCache (decode)
# =========================================================================

@pytest.mark.unit
def test_update_cache_perf_exact():
    p = _update_op(_device()).perf_stats

    tile_elems = USERS * KV_HEADS * math.ceil(HEAD_DIM / 32) * TILE_ELEMS      # 1*8*4*1024 = 32768 per cache
    assert tile_elems == 32768
    cache_bytes = 2 * tile_elems * BFP8                                         # K + V, bfp8 -> 69632
    side_elems = (USERS * KV_HEADS * HEAD_DIM) * 2 + 1 * USERS + USERS * NUM_BLOCKS   # k_in, v_in, upd, page_table
    side_bytes = (USERS * KV_HEADS * HEAD_DIM) * 2 * BF16 + 1 * USERS * INT32 + USERS * NUM_BLOCKS * INT32

    assert p["inElems"] == 2 * tile_elems + side_elems == 68609
    assert p["outElems"] == 2 * tile_elems == 65536
    assert p["inBytes"] == int(cache_bytes + side_bytes) == 77828
    assert p["outBytes"] == int(cache_bytes) == 69632
    assert p["instrs"] == {"mov": 2 * tile_elems}


@pytest.mark.unit
def test_update_cache_does_not_scale_with_cache_size():
    """The whole point of the fix: cost is per touched tile, independent of num_blocks."""
    p = _update_op(_device()).perf_stats
    whole_cache_elems = 2 * NUM_BLOCKS * KV_HEADS * BLOCK_SIZE * HEAD_DIM       # K+V, 67,108,864
    # one 32-row tile per head touched out of NUM_BLOCKS*BLOCK_SIZE rows -> NUM_BLOCKS*BLOCK_SIZE/32 = 1024x fewer
    assert p["outElems"] * (NUM_BLOCKS * BLOCK_SIZE // 32) == whole_cache_elems


# =========================================================================
# ScaledDotProductAttention (decode)
# =========================================================================

@pytest.mark.unit
def test_sdpa_decode_perf_exact_last_slot():
    """max_seq_len=1024, 1 generated token -> prompt fills 0..1022, decode token at POS=1023."""
    p = _sdpa_decode_op(_device()).perf_stats

    positions = _positions(1023)
    assert positions == 1024
    kv_elems = USERS * KV_HEADS * positions * HEAD_DIM                          # per cache, 1,048,576
    kv_bytes = 2 * kv_elems * BFP8                                              # 2,228,224 (bfp8)
    q_elems = USERS * Q_HEADS * HEAD_DIM                                        # 4096
    side_elems = q_elems + 1 * USERS + USERS * NUM_BLOCKS                       # q, cur_pos, page_table
    side_bytes = q_elems * BF16 + 1 * USERS * INT32 + USERS * NUM_BLOCKS * INT32

    assert p["inElems"] == 2 * kv_elems + side_elems == 2102273
    assert p["inBytes"] == int(kv_bytes + side_bytes) == 2240516
    assert p["outElems"] == USERS * Q_HEADS * HEAD_DIM == 4096
    assert p["outBytes"] == 4096 * BF16 == 8192                                 # bf16 output
    assert p["instrs"]["mac"] == 2 * USERS * Q_HEADS * positions * HEAD_DIM == 8388608
    assert p["instrs"]["exp"] == USERS * Q_HEADS * positions == 32768


@pytest.mark.unit
@pytest.mark.parametrize("pos, expected_positions", [
    (0, 32),        # first token: 1 valid position, one whole tile read
    (31, 32),       # last position still inside tile 0
    (32, 64),       # crosses a tile boundary
    (100, 128),
    (1023, 1024),   # last slot of max_seq_len=1024
    (1024, 1056),   # one past 1024 -> extra tile
])
def test_sdpa_decode_positions_tile_rounded(pos, expected_positions):
    p = _sdpa_decode_op(_device(pos), tag=f"s{pos}").perf_stats

    assert _positions(pos) == expected_positions
    kv_elems = KV_HEADS * expected_positions * HEAD_DIM
    side_bytes = Q_HEADS * HEAD_DIM * BF16 + 1 * INT32 + NUM_BLOCKS * INT32
    assert p["inElems"] == 2 * kv_elems + Q_HEADS * HEAD_DIM + 1 + NUM_BLOCKS
    assert p["inBytes"] == int(2 * kv_elems * BFP8 + side_bytes)
    assert p["instrs"]["mac"] == 2 * Q_HEADS * expected_positions * HEAD_DIM
    assert p["instrs"]["exp"] == Q_HEADS * expected_positions


@pytest.mark.unit
def test_sdpa_decode_without_declared_pos_keeps_whole_cache():
    """No kv_cache_pos on the device (a workload that does not declare it): whole-cache accounting."""
    p = _sdpa_decode_op(_device(pos=None)).perf_stats
    cache_elems = NUM_BLOCKS * KV_HEADS * BLOCK_SIZE * HEAD_DIM
    assert p["inElems"] == 2 * cache_elems + Q_HEADS * HEAD_DIM + 1 + NUM_BLOCKS == 67113985
    assert p["outElems"] == 4096
    assert "exp" not in p["instrs"]


@pytest.mark.unit
def test_sdpa_prefill_with_three_inputs_not_treated_as_decode():
    """len(iTList) < 4 (no cur_pos) is prefill SDPA: unchanged even when a decode position is declared."""
    device = _device()
    kc, vc = _kv_caches(device, "pf")
    q = _t(device, [1, 32, 128, 128], "pf_q")
    scaled_dot_product_attention_op(q, kc, vc)
    p = [o for o in device.ops.values() if o.optype == "ScaledDotProductAttention"][0].perf_stats
    assert "exp" not in p["instrs"]          # decode model emits exp; passthrough does not


# =========================================================================
# PagedFillCache (prefill)
# =========================================================================

@pytest.mark.unit
def test_fill_cache_perf_exact():
    prompt_len = 1024  # whole tiles; the 1023-token (non-multiple) case is covered by the padding test
    p = _fill_op(_device(), prompt_len).perf_stats

    tile_elems = KV_HEADS * math.ceil(prompt_len / 32) * math.ceil(HEAD_DIM / 32) * TILE_ELEMS   # 8*32*4*1024
    in_elems = KV_HEADS * prompt_len * HEAD_DIM + (prompt_len // BLOCK_SIZE)    # input + page_table
    in_bytes = KV_HEADS * prompt_len * HEAD_DIM * BFP8 + (prompt_len // BLOCK_SIZE) * INT32

    assert tile_elems == 1048576
    assert p["outElems"] == tile_elems
    assert p["outBytes"] == int(tile_elems * BFP8)                              # written as bfp8
    assert p["inElems"] == in_elems
    assert p["inBytes"] == int(in_bytes)
    assert p["instrs"] == {"mov": tile_elems}


@pytest.mark.unit
def test_fill_cache_pads_prompt_to_whole_tiles():
    """A 1023-token prompt writes ceil(1023/32)=32 tile rows, same as 1024."""
    p1023 = _fill_op(_device(), 1023, tag="f1023").perf_stats
    p1024 = _fill_op(_device(), 1024, tag="f1024").perf_stats
    assert p1023["outElems"] == p1024["outElems"] == KV_HEADS * 32 * 4 * TILE_ELEMS
    assert p1023["outBytes"] == p1024["outBytes"]


@pytest.mark.unit
def test_fill_cache_does_not_read_cache():
    """Fill only writes the cache; inElems must exclude the (huge) cache tensor."""
    prompt_len = 128
    p = _fill_op(_device(), prompt_len).perf_stats
    # input [1, 8, 128, 128] + page_table [1,1,1,4] only; the 33.5M-element cache is not an input.
    assert p["inElems"] == KV_HEADS * prompt_len * HEAD_DIM + prompt_len // BLOCK_SIZE == 131076
    assert p["inBytes"] == int(KV_HEADS * prompt_len * HEAD_DIM * BFP8 + (prompt_len // BLOCK_SIZE) * INT32)


# =========================================================================
# dtype handling: bytes use the cache tensors' own TT dtype
# =========================================================================

@pytest.mark.unit
def test_bytes_use_tt_dtype_not_fp32():
    """bfp8 caches must be 1.0625 B/elem, not numpy float32's 4 B/elem nor a flat 2 B."""
    p = _update_op(_device()).perf_stats
    tile_elems = 2 * KV_HEADS * 4 * TILE_ELEMS
    side = USERS * KV_HEADS * HEAD_DIM * 2 * BF16 + INT32 + NUM_BLOCKS * INT32
    assert p["outBytes"] == int(tile_elems * BFP8)
    assert p["outBytes"] != tile_elems * 4
    assert p["outBytes"] != tile_elems * 2
    assert p["inBytes"] == int(tile_elems * BFP8 + side)


@pytest.mark.unit
def test_bfp4_bytes():
    """BFLOAT4_B cache: 576 B / tile -> 0.5625 B/elem."""
    device = _device()
    shape = [NUM_BLOCKS, KV_HEADS, BLOCK_SIZE, HEAD_DIM]
    kc = _t(device, shape, "b4_kc", dtype=DataType.BFLOAT4_B)
    vc = _t(device, shape, "b4_vc", dtype=DataType.BFLOAT4_B)
    k = _t(device, [1, USERS, KV_HEADS, HEAD_DIM], "b4_k")
    v = _t(device, [1, USERS, KV_HEADS, HEAD_DIM], "b4_v")
    paged_fused_update_cache_op(kc, k, vc, v, update_idxs_tensor=_idx(device, [1, USERS], "b4_u"),
                                page_table=_idx(device, [USERS, NUM_BLOCKS], "b4_pt"))
    p = [o for o in device.ops.values() if o.optype == "PagedFusedUpdateCache"][0].perf_stats
    assert p["outBytes"] == int(2 * KV_HEADS * 4 * TILE_ELEMS * 576 / 1024)


