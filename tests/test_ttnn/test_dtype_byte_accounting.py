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


# =========================================================================
# Concat: inputs sized from the original tensors (TT dtype, layout, padded shape)
# =========================================================================

@pytest.mark.unit
@pytest.mark.parametrize("dtype, tile_bytes", [
    (DataType.BFLOAT16, 2048),
    (DataType.BFLOAT8_B, BFP8_TILE),
    (DataType.BFLOAT4_B, BFP4_TILE),
])
def test_concat_inputs_and_output_agree(dtype, tile_bytes):
    """Two TILE [1,1,64,64] (4 tiles each) concatenated on the last dim -> [1,1,64,128] (8 tiles).

    Before, inBytes was summed over normalised copies that drop the TT dtype, so bfp8 inputs fell back
    to 4 B per element: inBytes = 32768 while outBytes = 8704.
    """
    from ttsim.front.ttnn.op import concat

    out = concat(_t([1, 1, 64, 64], "ca", dtype=dtype), _t([1, 1, 64, 64], "cb", dtype=dtype), axis=3)
    ps = [o for o in out.device.ops.values() if o.optype == "Concat"][0].perf_stats
    assert ps["inBytes"] == 2 * 4 * tile_bytes
    assert ps["outBytes"] == 8 * tile_bytes
    assert ps["inBytes"] == ps["outBytes"]


@pytest.mark.unit
def test_concat_input_keeps_explicit_padded_shape():
    """bf16 TILE logical [1,1,1,33] stored padded [1,1,64,64] (4 tiles, beyond tile rounding's 2)."""
    from ttsim.front.ttnn.op import concat

    def padded(name):
        return Tensor(name=name, shape=[1, 1, 1, 33], dtype=DataType.BFLOAT16, layout=Layout.TILE_LAYOUT,
                      padded_shape=[1, 1, 64, 64], device=_device())

    out = concat(padded("pa"), padded("pb"), axis=2)
    ps = [o for o in out.device.ops.values() if o.optype == "Concat"][0].perf_stats
    assert ps["inBytes"] == 2 * 4 * 2048


# =========================================================================
# padded_shape(): explicit padding is used; a stale one is not
# =========================================================================

def _padded_tensor(logical, padded, dtype=DataType.BFLOAT16, name="p"):
    return Tensor(name=name, shape=logical, dtype=dtype, layout=Layout.TILE_LAYOUT,
                  padded_shape=padded, device=_device())


@pytest.mark.unit
def test_explicit_padded_shape_beyond_tile_rounding():
    """logical [1,1,1,33] padded [1,1,64,64]: 2 x 2 = 4 tiles, not the 1 x 2 tile rounding gives."""
    assert _padded_tensor([1, 1, 1, 33], [1, 1, 64, 64]).nbytes() == 4 * 2048


@pytest.mark.unit
def test_stale_padded_shape_is_ignored():
    """A shape assigned directly (bypassing set_shape) leaves the padded shape stale.

    [1,1,1,33] padded [1,1,64,64], then shape -> [1,1,1,10]: the stale 4-tile padded shape is ignored and
    the logical shape is tile-rounded to 1 tile. (The old p >= l heuristic accepted the stale shape.)
    """
    from ttsim.ops.tensor import Shape

    t = _padded_tensor([1, 1, 1, 33], [1, 1, 64, 64])
    t.shape = Shape([1, 1, 1, 10])
    assert t.nbytes() == 1 * 2048


@pytest.mark.unit
def test_set_shape_padded_shape_is_trusted():
    """set_shape recomputes the padded shape for the new logical shape, which stays trusted."""
    t = _padded_tensor([1, 1, 1, 33], [1, 1, 64, 64])
    t.set_shape([1, 1, 40, 40])                         # padded -> [1,1,64,64]
    assert t.nbytes() == 4 * 2048


# =========================================================================
# hw_shape: used when consistent; a smaller (stale) one falls back to the logical shape
# =========================================================================

@pytest.mark.unit
def test_larger_hw_shape_is_used():
    """A halo-extended hw_shape [1,1,96,64] (more elements than logical) is the stored shape: 3x2 tiles."""
    x = _t([1, 1, 64, 64], "halo")
    x.hw_shape = [1, 1, 96, 64]
    assert x.nbytes() == 3 * 2 * BFP8_TILE


@pytest.mark.unit
def test_smaller_hw_shape_falls_back_and_warns_once(monkeypatch):
    """hw_shape [1,1,32,32] holds fewer elements than logical [1,1,64,64]: bytes use the logical 4 tiles.

    The module logger is replaced with a recorder, so the check does not depend on loguru's global sink
    state, which other tests change.
    """
    import ttsim.ops.tensor as tensor_mod

    warnings_seen = []

    class _Recorder:
        def warning(self, msg, *args):
            warnings_seen.append(msg.format(*args))

        def debug(self, *args, **kwargs):
            pass

    monkeypatch.setattr(tensor_mod, "logger", _Recorder())
    x = _t([1, 1, 64, 64], "stale_hw")
    x.hw_shape = [1, 1, 32, 32]
    assert x.nbytes() == 4 * BFP8_TILE
    assert x.nbytes() == 4 * BFP8_TILE                  # second call: no second warning
    assert len(warnings_seen) == 1
    assert "stale_hw" in warnings_seen[0] and "hw_shape" in warnings_seen[0]


# =========================================================================
# dtype / layout corner cases
# =========================================================================

@pytest.mark.unit
def test_hw_dtype_without_hw_layout_uses_tensor_layout():
    """_hw_dtype bfp8 with no _hw_layout: the tensor's own ROW_MAJOR layout applies.

    [1,1,33,64] row-major bfp8 = 2112 elems * 1088 / 1024 = 2244 B (tiled would be 2 x 2 tiles = 4352 B).
    """
    w = _t([1, 1, 33, 64], "hwd", dtype=DataType.FLOAT32, layout=Layout.ROW_MAJOR_LAYOUT)
    w._hw_dtype = DataType.BFLOAT8_B
    assert w.nbytes() == 33 * 64 * BFP8_TILE // 1024 == 2244


@pytest.mark.unit
def test_rank1_tile_tensor_is_one_row_of_tiles():
    """Rank-1 TILE [64] is stored as [1, 64]: ceil(1/32) * ceil(64/32) = 2 tiles."""
    assert _t([64], "r1").nbytes() == 2 * BFP8_TILE


@pytest.mark.unit
def test_rank0_tile_tensor_is_one_tile():
    """Rank-0 TILE tensor is stored as one [1, 1] -> one 32x32 tile."""
    assert _t([], "r0").nbytes() == 1 * BFP8_TILE


@pytest.mark.unit
def test_pad_falls_back_to_element_size_without_tt_dtype():
    """pad_channels_nchw on a tensor with no TT dtype (numpy float32): elements * element_size() = 4 B."""
    from ttsim.front.ttnn.op import pad_channels_nchw

    x = Tensor(name="np_pad", shape=[1, 3, 32, 32], dtype=np.dtype(np.float32),
               layout=Layout.ROW_MAJOR_LAYOUT, device=_device())
    assert x._ttnn_dtype is None
    out = pad_channels_nchw(x, 16)
    assert out.device is not None
    ps = [o for o in out.device.ops.values() if o.optype == "Pad"][-1].perf_stats
    assert ps["inBytes"] == 3 * 32 * 32 * 4
    assert ps["outBytes"] == 16 * 32 * 32 * 4


@pytest.mark.unit
def test_unaligned_matmul_is_tile_exact():
    """[1,1,1,4096] x [1,1,4096,128] bfp8: A = 1 x 128 tiles, B = 128 x 4 tiles, out = 1 x 4 tiles.

    Per-element sizing would give A = 4096 * 1.0625 = 4352 B; tile-exact gives 128 * 1088 = 139264 B.
    """
    from ttsim.front.ttnn.op import matmul

    c = matmul(_t([1, 1, 1, 4096], "ua"), _t([1, 1, 4096, 128], "ub"))
    ps = [o for o in c.device.ops.values() if o.optype == "MatMul"][0].perf_stats
    assert ps["inBytes"] == (1 * 128 + 128 * 4) * BFP8_TILE
    assert ps["outBytes"] == 1 * 4 * BFP8_TILE


# =========================================================================
# device memory-bandwidth validation slack
# =========================================================================

@pytest.mark.unit
@pytest.mark.parametrize("actual, expected, ok", [
    (1886186, 1886185.0, True),          # one cycle above an integer expected: the final ceil
    (1886185, 1886185.0, True),
    (1886184, 1886185.0, True),
    (1886186.0000005, 1886185.0, True),  # within the 1e-6 float slack
    (1886186.00001, 1886185.0, False),   # beyond the slack
    (1886187, 1886185.0, False),
    (1886183, 1886185.0, False),
])
def test_mem_cycles_validation_slack(actual, expected, ok):
    from ttsim.back.device import _mem_cycles_within_slack

    assert _mem_cycles_within_slack(actual, expected) is ok


@pytest.mark.unit
def test_smaller_halo_hw_shape_is_kept():
    """A strided halo output is legitimately smaller than its logical shape and must be used as stored.

    WH resnet50 layer2.0 downsample: logical [16,256,56,56] (50176 x 256), halo hw_shape [1,1,46984,256]
    as the capture records. bfp8 TILE: ceil(46984/32) * ceil(256/32) = 1469 * 8 tiles.
    """
    x = _t([16, 256, 56, 56], "halo_out")
    x.hw_shape = [1, 1, 46984, 256]
    x._hw_shape_halo = True
    assert x.nbytes() == 1469 * 8 * BFP8_TILE


@pytest.mark.unit
def test_halo_marker_propagates_through_passthrough_ops():
    """Move / ITS / STI / Reshard / Copy pass the halo marker on with hw_shape."""
    from ttsim.ops.desc.ttsim_layout import _propagate_hw_shape

    src = _t([16, 256, 56, 56], "src")
    src.hw_shape, src._hw_shape_halo = [1, 1, 46984, 256], True
    dst = _t([16, 256, 56, 56], "dst")
    _propagate_hw_shape(src, dst)
    assert dst.hw_shape == [1, 1, 46984, 256] and dst._hw_shape_halo is True


# =========================================================================
# output dtype: a data dtype wins over an index dtype (Embedding)
# =========================================================================

@pytest.mark.unit
def test_embedding_output_takes_table_dtype():
    """Embedding(indices uint32 [1,32], table bf16 [1024,128]) -> bf16 [1,32,128] = 4096 * 2 B.

    The output dtype used to be the first input's (the uint32 indices), so with the dtype resolved before
    the byte count the output was sized at 4 B per element (16384 B).
    """
    from ttsim.front.ttnn.op import embedding

    idx = _t([1, 32], "idx", dtype=DataType.UINT32, layout=Layout.ROW_MAJOR_LAYOUT)
    table = _t([1024, 128], "table", dtype=DataType.BFLOAT16, layout=Layout.ROW_MAJOR_LAYOUT)
    out = embedding(idx, table)
    ps = next(o for o in out.device.ops.values() if o.optype == "Embedding").perf_stats
    assert out._ttnn_dtype == DataType.BFLOAT16
    assert ps["outBytes"] == 1 * 32 * 128 * 2


@pytest.mark.unit
@pytest.mark.parametrize("inputs, expected", [
    ([DataType.UINT32, DataType.BFLOAT16], DataType.BFLOAT16),    # index + data: data wins
    ([DataType.BOOL, DataType.FLOAT32], DataType.FLOAT32),        # mask + data
    ([DataType.UINT32, DataType.INT32], DataType.UINT32),         # all integer: first input, as before
    ([DataType.BFLOAT16, DataType.BFLOAT8_B], DataType.BFLOAT8_B),  # compact dtype still wins
])
def test_propagated_output_dtype(inputs, expected):
    from ttsim.front.ttnn.op import _propagate_ttnn_dtype

    ins = [_t([1, 1, 32, 32], f"i{k}", dtype=d) for k, d in enumerate(inputs)]
    out = _t([1, 1, 32, 32], "o")
    out._ttnn_dtype = None
    _propagate_ttnn_dtype(ins, [out])
    assert out._ttnn_dtype == expected
