# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Data-movement ops keep the input's padded shape and count traffic on padded storage.

Covers InterleavedToSharded, ShardedToInterleaved, Reshard, Move and Halo with a TILE-layout input
that is not tile-aligned (and, for the shim builders, whose padded shape exceeds tile rounding).

    BF16 TILE, logical [1, 1, 1, 33], padded [1, 1, 32, 64]:
        logical bytes = 33 * 2 = 66        (what was recorded before)
        padded bytes  = 32 * 64 * 2 = 4096 (what the hardware moves)

Copy (added by #551) is covered through ``copy_op``.
Every expected value is derived by hand.
"""

import pytest

from ttsim.front.ttnn.device import ARCH, Device
from ttsim.front.ttnn.op import _move, halo
from ttsim.front.ttnn.tensor import DataType, Layout, Tensor
from ttsim.front.ttnn.ttnn_shim import copy_op, interleaved_to_sharded_op, reshard_op, sharded_to_interleaved_op

BFP8_TILE = 1088
class _MC:
    memory_layout = None
    buffer_type = None


SHIM_OPS = [
    ("Copy", lambda x: copy_op(x, _MC())),
    ("InterleavedToSharded", interleaved_to_sharded_op),
    ("ShardedToInterleaved", sharded_to_interleaved_op),
    ("Reshard", reshard_op),
]
ALL_OPS = SHIM_OPS + [("Move", _move), ("Halo", halo)]


def _device():
    device = Device(device_id=0)
    device.architecture = ARCH.BLACKHOLE
    return device


def _tile_tensor(logical, padded=None, dtype=DataType.BFLOAT16, name="x"):
    return Tensor(name=name, shape=logical, dtype=dtype, layout=Layout.TILE_LAYOUT,
                  padded_shape=padded, device=_device())


def _perf(out, optype):
    return [o for o in out.device.ops.values() if o.optype == optype][0].perf_stats


# =========================================================================
# defect 1: the shim builders restore the input's padded shape on the output
# =========================================================================

@pytest.mark.unit
@pytest.mark.parametrize("optype, builder", SHIM_OPS)
def test_builder_keeps_explicit_padded_shape(optype, builder):
    """logical [1,1,32,32], padded [1,1,64,32] (beyond tile rounding): the output keeps both."""
    x = _tile_tensor([1, 1, 32, 32], padded=[1, 1, 64, 32])
    out = builder(x)
    assert list(out.padded_shape()) == [1, 1, 64, 32]
    assert list(out.logical_shape()) == [1, 1, 32, 32]


@pytest.mark.unit
@pytest.mark.parametrize("optype, builder", SHIM_OPS)
def test_builder_default_padding_unchanged(optype, builder):
    """No explicit padding: the padded shape is plain tile rounding, as before."""
    x = _tile_tensor([1, 1, 1, 33])
    out = builder(x)
    assert list(out.padded_shape()) == [1, 1, 32, 64]


# =========================================================================
# defect 2: traffic is counted on the padded storage, flag off (elements x element_size)
# =========================================================================

@pytest.mark.unit
@pytest.mark.parametrize("optype, op_fn", ALL_OPS)
def test_traffic_counts_padded_storage(optype, op_fn, monkeypatch):
    x = _tile_tensor([1, 1, 1, 33])           # padded [1, 1, 32, 64]
    out = op_fn(x)
    p = _perf(out, optype)
    assert p["inElems"] == p["outElems"] == 32 * 64
    assert p["inBytes"] == p["outBytes"] == 32 * 64 * 2 == 4096     # not 33 * 2 = 66
    assert p["instrs"] == {"mov": 32 * 64}
    assert list(out.logical_shape()) == [1, 1, 1, 33]               # output logical shape unchanged


@pytest.mark.unit
@pytest.mark.parametrize("optype, op_fn", ALL_OPS)
def test_traffic_counts_explicit_padding(optype, op_fn, monkeypatch):
    """Explicit padded shape [1,1,64,32] beyond tile rounding of [1,1,32,32]."""
    x = _tile_tensor([1, 1, 32, 32], padded=[1, 1, 64, 32])
    p = _perf(op_fn(x), optype)
    assert p["inBytes"] == p["outBytes"] == 64 * 32 * 2 == 4096


@pytest.mark.unit
@pytest.mark.parametrize("optype, op_fn", ALL_OPS)
def test_aligned_tile_input_unchanged(optype, op_fn, monkeypatch):
    x = _tile_tensor([1, 1, 64, 64])
    p = _perf(op_fn(x), optype)
    assert p["inElems"] == p["outElems"] == 64 * 64
    assert p["inBytes"] == p["outBytes"] == 64 * 64 * 2


# =========================================================================
# defect 2: dtype- and tile-aware bytes
# =========================================================================

@pytest.mark.unit
@pytest.mark.parametrize("optype, op_fn", ALL_OPS)
def test_poc_bytes_bf16_padded_tiles(optype, op_fn, monkeypatch):
    p = _perf(op_fn(_tile_tensor([1, 1, 1, 33])), optype)
    assert p["inBytes"] == p["outBytes"] == 1 * 2 * 1024 * 2 == 4096   # 2 bf16 tiles


@pytest.mark.unit
@pytest.mark.parametrize("optype, op_fn", ALL_OPS)
def test_poc_bytes_bfp8_padded_tiles(optype, op_fn, monkeypatch):
    """[1,1,1,33] bfp8 pads to 2 tiles = 2176 B (element_size 2 would give 4096)."""
    p = _perf(op_fn(_tile_tensor([1, 1, 1, 33], dtype=DataType.BFLOAT8_B)), optype)
    assert p["inBytes"] == p["outBytes"] == 2 * BFP8_TILE == 2176


@pytest.mark.unit
@pytest.mark.parametrize("optype, op_fn", ALL_OPS)
def test_conv_path_tensor_uses_hw_shape_not_logical(optype, op_fn, monkeypatch):
    """Logical NCHW [1,64,56,56], stored [1,1,3136,64] bfp8 TILE: 98 x 2 tiles = 213248 B.

    Padding the logical H x W (56x56 -> 64x64) would count 1 * 64 * 2 * 2 = 256 tiles.
    """
    x = _tile_tensor([1, 64, 56, 56], dtype=DataType.BFLOAT8_B)
    x.hw_shape = [1, 1, 56 * 56, 64]
    p = _perf(op_fn(x), optype)
    assert p["inElems"] == p["outElems"] == 3136 * 64
    assert p["inBytes"] == p["outBytes"] == 98 * 2 * BFP8_TILE == 213248
    assert p["inBytes"] != 256 * BFP8_TILE


@pytest.mark.unit
def test_halo_output_sized_by_its_own_dtype(monkeypatch):
    """Halo output is bf16 row-major: bytes follow the OUTPUT dtype and layout, not the bfp8 input's."""
    x = _tile_tensor([1, 64, 56, 56], dtype=DataType.BFLOAT8_B)
    x.hw_shape = [1, 1, 56 * 56, 64]
    out = halo(x, dtype=DataType.BFLOAT16, layout=Layout.ROW_MAJOR_LAYOUT)
    p = _perf(out, "Halo")
    assert p["inBytes"] == 98 * 2 * BFP8_TILE
    assert p["outBytes"] == 3136 * 64 * 2          # row-major bf16, no padding, no tile overhead
