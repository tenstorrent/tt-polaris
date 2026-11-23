# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Layout ops (Tilize, Untilize, TilizeWithValPadding, UntilizeWithUnpadding) count bytes by the
tensor's own TT dtype.

Before, Untilize and UntilizeWithUnpadding computed ``elements * element_size`` regardless of dtype, so a
bfp8 tile (1088 B) and a bf16 tile (2048 B) cost the same per element, and the output dtype was resolved
after the byte count. Every expected value is derived by hand.

    one 32x32 tile:  BF16 = 32 * 32 * 2 = 2048 B     BFP8 = 1088 B     BFP4 = 576 B
    row-major block-float counts only the exponent overhead: elements * tile_bytes / 1024
"""

import pytest

from ttsim.front.ttnn.device import ARCH, Device
from ttsim.front.ttnn.tensor import DataType, Layout, Tensor
from ttsim.front.ttnn.ttnn_shim import (
    tilize_op,
    tilize_with_val_padding_op,
    untilize_op,
    untilize_with_unpadding_op,
)

BF16_TILE = 2048
BFP8_TILE = 1088
BFP4_TILE = 576


def _device():
    device = Device(device_id=0)
    device.architecture = ARCH.BLACKHOLE
    return device


def _tensor(logical, layout, dtype, padded=None, name="x"):
    return Tensor(name=name, shape=logical, dtype=dtype, layout=layout, padded_shape=padded, device=_device())


def _perf(out, optype):
    return [o for o in out.device.ops.values() if o.optype == optype][0].perf_stats


# =========================================================================
# Untilize: TILE in (block-float tiles), ROW_MAJOR out
# =========================================================================

@pytest.mark.unit
@pytest.mark.parametrize("dtype, tile_bytes", [
    (DataType.BFLOAT16, BF16_TILE),
    (DataType.BFLOAT8_B, BFP8_TILE),
    (DataType.BFLOAT4_B, BFP4_TILE),
])
def test_untilize_bytes_follow_the_tensor_dtype(dtype, tile_bytes):
    # [1,1,32,64] = 2 tiles in, 2048 row-major elements out
    x = _tensor([1, 1, 32, 64], Layout.TILE_LAYOUT, dtype)
    out = untilize_op(x)
    ps = _perf(out, "Untilize")
    assert ps["inBytes"] == 2 * tile_bytes
    # row-major: block-float elements carry only the exponent overhead; bf16 is 2 B per element
    expected_out = 2048 * 2 if dtype == DataType.BFLOAT16 else 2048 * tile_bytes // 1024
    assert ps["outBytes"] == expected_out
    assert ps["inElems"] == 2048 and ps["outElems"] == 2048


@pytest.mark.unit
def test_untilize_bfp8_is_cheaper_than_bf16():
    bf16 = _perf(untilize_op(_tensor([1, 1, 32, 64], Layout.TILE_LAYOUT, DataType.BFLOAT16)), "Untilize")
    bfp8 = _perf(untilize_op(_tensor([1, 1, 32, 64], Layout.TILE_LAYOUT, DataType.BFLOAT8_B)), "Untilize")
    assert bfp8["inBytes"] < bf16["inBytes"]


# =========================================================================
# UntilizeWithUnpadding: the ResNet-50 classifier head, logical [1,1,16,1000], padded [1,1,32,1024]
# =========================================================================

@pytest.mark.unit
def test_untilize_with_unpadding_bfp8_reads_padded_block_float_tiles():
    # padded [32, 1024] = 1 * 32 tiles; logical output [16, 1000] = 16000 elements
    x = _tensor([1, 1, 16, 1000], Layout.TILE_LAYOUT, DataType.BFLOAT8_B, padded=[1, 1, 32, 1024])
    out = untilize_with_unpadding_op(x, [1, 1, 16, 1000])
    ps = _perf(out, "UntilizeWithUnpadding")
    assert ps["inElems"] == 32 * 1024
    assert ps["inBytes"] == 32 * BFP8_TILE              # 34,816 B (was 32768 at element_size 1/2)
    assert ps["outElems"] == 16000
    assert ps["outBytes"] == 16000 * BFP8_TILE // 1024  # 17,000 B row-major block-float


@pytest.mark.unit
def test_untilize_with_unpadding_bf16_is_unchanged():
    x = _tensor([1, 1, 16, 1000], Layout.TILE_LAYOUT, DataType.BFLOAT16, padded=[1, 1, 32, 1024])
    ps = _perf(untilize_with_unpadding_op(x, [1, 1, 16, 1000]), "UntilizeWithUnpadding")
    assert ps["inBytes"] == 32 * BF16_TILE              # 65,536 B
    assert ps["outBytes"] == 16000 * 2                  # 32,000 B


# =========================================================================
# Tilize / TilizeWithValPadding: ROW_MAJOR in, TILE out
# =========================================================================

@pytest.mark.unit
@pytest.mark.parametrize("dtype, tile_bytes", [
    (DataType.BFLOAT16, BF16_TILE),
    (DataType.BFLOAT8_B, BFP8_TILE),
    (DataType.BFLOAT4_B, BFP4_TILE),
])
def test_tilize_output_is_whole_tiles_of_its_dtype(dtype, tile_bytes):
    x = _tensor([1, 1, 32, 64], Layout.ROW_MAJOR_LAYOUT, dtype)
    ps = _perf(tilize_op(x), "Tilize")
    assert ps["outBytes"] == 2 * tile_bytes
    assert ps["outElems"] == 2048


@pytest.mark.unit
@pytest.mark.parametrize("dtype, tile_bytes", [
    (DataType.BFLOAT16, BF16_TILE),
    (DataType.BFLOAT8_B, BFP8_TILE),
])
def test_tilize_with_val_padding_counts_padded_tiles(dtype, tile_bytes):
    # logical [1,1,20,40] padded up to [1,1,32,64] = 2 tiles
    x = _tensor([1, 1, 20, 40], Layout.ROW_MAJOR_LAYOUT, dtype)
    out = tilize_with_val_padding_op(x, [1, 1, 32, 64], 0.0)
    ps = _perf(out, "TilizeWithValPadding")
    assert ps["outElems"] == 32 * 64
    assert ps["outBytes"] == 2 * tile_bytes
