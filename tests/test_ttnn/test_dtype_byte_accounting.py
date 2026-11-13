# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Tests for dtype-correct byte accounting (always on) in SimTensor.nbytes.

Covers the conv paths (ResNet-50 shapes) and the precedence rules:

    * the tensor's own TT dtype wins over the caller's itemprec (descriptors pass op.precision,
      which the backend always sets), so the fix is not bypassed;
    * _hw_dtype / _hw_layout / hw_shape (what the tensor is on HW) win over _ttnn_dtype / layout;
    * row-major block-float tensors are not tile padded; tiled ones pad by hw_shape when set.

Every expected value is derived by hand, not by calling the helpers under test.

    bfp8 tile = 1088 B, bfp4 tile = 576 B, tile = 32 x 32 = 1024 elems
    tiled bytes = lead * ceil(rows/32) * ceil(cols/32) * tile_bytes   (rows, cols of the stored shape)
    row-major bytes = nelems * tile_bytes / 1024
"""

import numpy as np
import pytest

from ttsim.front.ttnn.device import ARCH, Device
from ttsim.front.ttnn.tensor import DataType, Layout, Tensor
from ttsim.ops.tensor import SimTensor

BFP8_TILE = 1088
BFP4_TILE = 576
FP16 = np.dtype(np.float16)   # stands in for the op-level precision descriptors pass in
FP32 = np.dtype(np.float32)


def _device():
    device = Device(device_id=0)
    device.architecture = ARCH.BLACKHOLE
    return device


def _t(shape, name, dtype=DataType.BFLOAT8_B, layout=Layout.TILE_LAYOUT):
    return Tensor(name=name, shape=shape, dtype=dtype, layout=layout, device=_device())


# =========================================================================
# ResNet-50 conv shapes
# =========================================================================

@pytest.mark.unit
def test_rowmajor_conv_weight_not_tile_padded():
    """1x1 conv weight OIHW [256, 64, 1, 1], row-major bfp8: exponent overhead only.

    Padding the logical last two dims (1x1 -> 32x32) would count 256*64 tiles = 17.8 MB (1024x).
    """
    w = _t([256, 64, 1, 1], "w1x1", layout=Layout.ROW_MAJOR_LAYOUT)
    assert w.nbytes() == 256 * 64 * BFP8_TILE // 1024 == 17408


@pytest.mark.unit
def test_rowmajor_bfp4_conv_weight():
    w = _t([256, 64, 1, 1], "w1x1_bfp4", dtype=DataType.BFLOAT4_B, layout=Layout.ROW_MAJOR_LAYOUT)
    assert w.nbytes() == 256 * 64 * BFP4_TILE // 1024 == 9216


@pytest.mark.unit
def test_hw_weight_uses_hw_dtype_layout_and_shape():
    """conv1 weight: logical OIHW [64, 3, 7, 7] fp32/row-major, on HW [1, 1, 147, 64] bfp8 TILE.

    tiles = ceil(147/32) * ceil(64/32) = 5 * 2 = 10 -> 10880 B. The logical-shape answers would be
    9408 elems * 1.0625 = 9996 B (row-major) or 64*3*2*2 = 768 tiles (padding 7x7).
    """
    w = _t([64, 3, 7, 7], "conv1_w", dtype=DataType.FLOAT32, layout=Layout.ROW_MAJOR_LAYOUT)
    w.hw_shape = [1, 1, 3 * 7 * 7, 64]
    w._hw_dtype = DataType.BFLOAT8_B
    w._hw_layout = Layout.TILE_LAYOUT
    assert w.nbytes() == 5 * 2 * BFP8_TILE == 10880


@pytest.mark.unit
def test_hw_bias_uses_hw_dtype():
    """bias [64] -> hw [1, 1, 1, 64] bfp8 TILE: 1 * ceil(1/32) * ceil(64/32) = 2 tiles."""
    b = _t([64], "conv1_b", dtype=DataType.FLOAT32, layout=Layout.ROW_MAJOR_LAYOUT)
    b.hw_shape = [1, 1, 1, 64]
    b._hw_dtype = DataType.BFLOAT8_B
    b._hw_layout = Layout.TILE_LAYOUT
    assert b.nbytes() == 2 * BFP8_TILE == 2176


@pytest.mark.unit
def test_activation_padded_by_hw_shape_not_logical():
    """Activation logical [1, 64, 56, 56], stored [1, 1, 3136, 64] bfp8 TILE.

    stored tiles = 98 * 2 = 196 -> 213248 B. Padding the logical H x W (56x56 -> 64x64) would give
    1 * 64 * 2 * 2 = 256 tiles = 278528 B.
    """
    x = _t([1, 64, 56, 56], "act")
    x.hw_shape = [1, 1, 1 * 56 * 56, 64]
    assert x.nbytes() == 98 * 2 * BFP8_TILE == 213248
    assert x.nbytes() != 256 * BFP8_TILE


@pytest.mark.unit
def test_tiled_without_hw_shape_pads_logical_last_two_dims():
    """Plain tiled tensor [1, 1, 50, 70] bfp8: ceil(50/32) * ceil(70/32) = 2 * 3 = 6 tiles."""
    x = _t([1, 1, 50, 70], "plain")
    assert x.nbytes() == 6 * BFP8_TILE == 6528


# =========================================================================
# precedence: tensor's own dtype wins over itemprec (op.precision)
# =========================================================================

@pytest.mark.unit
def test_itemprec_does_not_bypass_block_float():
    """Descriptors call nbytes(op.precision); the TT dtype must still win."""
    x = _t([1, 1, 64, 64], "x")                       # 4 tiles bfp8
    assert x.nbytes() == x.nbytes(FP16) == x.nbytes(FP32) == 4 * BFP8_TILE


@pytest.mark.unit
def test_non_block_float_uses_tensor_tt_dtype():
    """bf16 tensor sized by its own 2 B even if op.precision says fp32; int32 stays 4 B."""
    a = _t([1, 1, 32, 32], "a", dtype=DataType.BFLOAT16)
    b = _t([1, 1, 32, 32], "b", dtype=DataType.INT32, layout=Layout.ROW_MAJOR_LAYOUT)
    assert a.nbytes(FP32) == 1024 * 2
    assert b.nbytes(FP16) == 1024 * 4


# =========================================================================
# tensors without a TT dtype keep the numpy accounting
# =========================================================================

@pytest.mark.unit
def test_tensor_without_tt_dtype_keeps_numpy_accounting():
    t = SimTensor({"name": "plain", "shape": [4, 8], "dtype": np.dtype(np.float32)})
    assert t.nbytes() == 4 * 8 * 4
    assert t.nbytes(FP16) == 4 * 8 * 2


# =========================================================================
# op outputs: dtype resolved before perf stats (outBytes sized by bfp8)
# =========================================================================

@pytest.mark.unit
def test_matmul_output_sized_as_bfp8():
    """matmul [1,1,64,64] x [1,1,64,128], bfp8 in/out: outBytes = 2*4 tiles * 1088 = 8704.

    Before the fix, outBytes was computed before the output's TT dtype was propagated, so the
    output fell back to its numpy dtype (fp32 -> 64*128*4 = 32768).
    """
    from ttsim.front.ttnn.op import matmul

    a = _t([1, 1, 64, 64], "a")
    b = _t([1, 1, 64, 128], "b")
    c = matmul(a, b)
    op = [o for o in c.device.ops.values() if o.optype == "MatMul"][0]
    assert c._ttnn_dtype == DataType.BFLOAT8_B
    assert op.perf_stats["outBytes"] == 2 * 4 * BFP8_TILE == 8704
    assert op.perf_stats["inBytes"] == (2 * 2 + 2 * 4) * BFP8_TILE      # A: 2x2 tiles, B: 2x4 tiles


@pytest.mark.unit
def test_linear_explicit_output_dtype_sizes_output():
    """linear(..., dtype=BFLOAT16): output [1,1,64,128] = 64*128*2 B, not the bfp8 activation size."""
    from ttsim.front.ttnn.op import linear

    a = _t([1, 1, 64, 64], "la")
    w = _t([1, 1, 64, 128], "lw")
    c = linear(a, w, dtype=DataType.BFLOAT16)
    op = [o for o in c.device.ops.values() if o.optype == "MatMul"][0]
    assert op.perf_stats["outBytes"] == 64 * 128 * 2


# =========================================================================
# Permute, Fold, Pad: block-float bytes follow the tensor's own TT dtype
# =========================================================================
#
# Before: permute_op and fold counted bytes before resolving the output TT dtype, so a bfp8 / bfp4
# output fell back to 4 B per element; pad_channels_nchw computed elements * element_size(), which
# drops the block-float exponent overhead (bfp8 counted at 1 B and bfp4 at 0.5 B per element).

DT_TILE = [
    (DataType.BFLOAT16, 2048),
    (DataType.BFLOAT8_B, BFP8_TILE),
    (DataType.BFLOAT4_B, BFP4_TILE),
]


def _row_major_bytes(nelems, dtype, tile_bytes):
    """bf16: 2 B per element; block-float row-major: nelems * tile_bytes / 1024."""
    return nelems * 2 if dtype == DataType.BFLOAT16 else nelems * tile_bytes // 1024


def _perf(out, optype):
    return [o for o in out.device.ops.values() if o.optype == optype][-1].perf_stats


@pytest.mark.unit
@pytest.mark.parametrize("dtype, tile_bytes", DT_TILE)
def test_permute_output_bytes_follow_dtype(dtype, tile_bytes):
    """[1,1,32,64] TILE permuted to [1,1,64,32]: 2 tiles in, 2 tiles out."""
    from ttsim.front.ttnn.ttnn_shim import permute_op

    out = permute_op(_t([1, 1, 32, 64], "px", dtype=dtype), [0, 1, 3, 2])
    ps = _perf(out, "Permute")
    assert ps["inBytes"] == 2 * tile_bytes
    assert ps["outBytes"] == 2 * tile_bytes          # was 8192 (2048 elems * 4 B) for bfp8 / bfp4


@pytest.mark.unit
@pytest.mark.parametrize("dtype, tile_bytes", DT_TILE)
def test_fold_output_bytes_follow_dtype(dtype, tile_bytes):
    """ROW_MAJOR [1,32,32,32] folded by 2x2 to [1,16,16,128]: 32768 elements in and out."""
    from ttsim.front.ttnn.op import fold

    out = fold(_t([1, 32, 32, 32], "fx", dtype=dtype, layout=Layout.ROW_MAJOR_LAYOUT), 2, 2)
    ps = _perf(out, "Fold")
    expected = _row_major_bytes(32768, dtype, tile_bytes)  # bf16 65536, bfp8 34816, bfp4 18432
    assert ps["inBytes"] == expected
    assert ps["outBytes"] == expected                # was 131072 (32768 * 4 B) for bfp8 / bfp4


@pytest.mark.unit
@pytest.mark.parametrize("dtype, tile_bytes", DT_TILE)
def test_pad_channels_nchw_bytes_follow_dtype(dtype, tile_bytes):
    """ROW_MAJOR NCHW [1,3,32,32] padded to [1,16,32,32]: 3072 elements in, 16384 out."""
    from ttsim.front.ttnn.op import pad_channels_nchw

    out = pad_channels_nchw(_t([1, 3, 32, 32], "padx", dtype=dtype, layout=Layout.ROW_MAJOR_LAYOUT), 16)
    ps = _perf(out, "Pad")
    # bf16 6144 / 32768 (unchanged); bfp8 3264 / 17408 (was 3072 / 16384); bfp4 1728 / 9216 (was 1536 / 8192)
    assert ps["inBytes"] == _row_major_bytes(3072, dtype, tile_bytes)
    assert ps["outBytes"] == _row_major_bytes(16384, dtype, tile_bytes)
    assert isinstance(ps["inBytes"], int) and isinstance(ps["outBytes"], int)
