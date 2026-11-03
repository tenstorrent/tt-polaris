#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Tests that Halo SimOps are auto-emitted before conv2d / max_pool2d / conv_transpose2d.

On Tenstorrent hardware, halo extraction is dispatched implicitly by the conv and
pool kernels.  The ttnn shim mirrors this by auto-emitting a Halo SimOp before each
Conv / MaxPool / ConvTranspose SimOp so that profiler-vs-TT-Polaris sequence matching
finds a corresponding entry for every hardware halo row.
"""

import pytest
import numpy as np

import ttsim.front.ttnn as ttnn
from ttsim.front.ttnn.device import ARCH, Device
from ttsim.front.ttnn.tensor import DataType, Layout, Tensor
from ttsim.front.ttnn.memory import MemoryConfig
from ttsim.front.ttnn.buffer import TensorMemoryLayout, BufferType


def _make_device():
    device = Device(device_id=0)
    device.architecture = ARCH.WORMHOLE_B0
    return device


def _make_tensor(name, shape, device):
    return Tensor(
        name=name,
        shape=shape,
        dtype=DataType.BFLOAT16,
        layout=Layout.ROW_MAJOR_LAYOUT,
        device=device,
    )


def _op_sequence(device):
    """Return list of (optype, op) in insertion order."""
    return [(op.optype, op) for op in device.ops.values()]


# ---------------------------------------------------------------------------
# conv2d: should emit Halo → Conv
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_conv2d_emits_halo_then_conv():
    device = _make_device()
    x = _make_tensor("x", [1, 3, 8, 8], device)
    w = _make_tensor("w", [4, 3, 3, 3], device)
    b = _make_tensor("b", [4], device)

    out = ttnn.conv2d(
        input_tensor=x,
        weight_tensor=w,
        bias_tensor=b,
        in_channels=3,
        out_channels=4,
        batch_size=1,
        input_height=8,
        input_width=8,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=(1, 1),
        dilation=(1, 1),
        groups=1,
        device=device,
    )

    seq = _op_sequence(device)
    assert len(seq) == 2, f"Expected 2 ops (Halo+Conv), got {[s[0] for s in seq]}"
    assert seq[0][0] == "Halo"
    assert seq[1][0] == "Conv"

    # Halo output shape == input shape (passthrough)
    halo_op = seq[0][1]
    assert halo_op.inList == [x.name]
    halo_out_name = halo_op.outList[0]

    # Conv input is the halo output
    conv_op = seq[1][1]
    assert halo_out_name in conv_op.inList

    # Final output shape correct
    assert out.shape == [1, 4, 8, 8]


@pytest.mark.unit
def test_conv2d_halo_shape_passthrough():
    device = _make_device()
    shape = [1, 16, 32, 32]
    x = _make_tensor("x", shape, device)
    w = _make_tensor("w", [32, 16, 3, 3], device)
    b = _make_tensor("b", [32], device)

    ttnn.conv2d(
        input_tensor=x, weight_tensor=w, bias_tensor=b,
        in_channels=16, out_channels=32, batch_size=1,
        input_height=32, input_width=32,
        kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
        dilation=(1, 1), groups=1, device=device,
    )

    halo_op = _op_sequence(device)[0][1]
    # Halo perf_stats reflects input shape
    assert halo_op.perf_stats["inElems"] == 1 * 16 * 32 * 32
    assert halo_op.perf_stats["outElems"] == 1 * 16 * 32 * 32


# ---------------------------------------------------------------------------
# max_pool2d: should emit Halo → MaxPool
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_max_pool2d_emits_halo_then_pool():
    device = _make_device()
    x = _make_tensor("x", [1, 64, 16, 16], device)

    ttnn.max_pool2d(
        input_tensor=x,
        batch_size=1,
        input_h=16,
        input_w=16,
        channels=64,
        kernel_size=[2, 2],
        stride=[2, 2],
        padding=[0, 0],
        dilation=[1, 1],
    )

    seq = _op_sequence(device)
    assert len(seq) == 2, f"Expected 2 ops (Halo+MaxPool), got {[s[0] for s in seq]}"
    assert seq[0][0] == "Halo"
    assert seq[1][0] == "MaxPool"

    halo_out_name = seq[0][1].outList[0]
    assert halo_out_name in seq[1][1].inList


# ---------------------------------------------------------------------------
# conv_transpose2d: should emit Halo → ConvTranspose
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_conv_transpose2d_emits_halo_then_convtranspose():
    device = _make_device()
    x = _make_tensor("x", [1, 16, 8, 8], device)
    w = _make_tensor("w", [16, 8, 2, 2], device)
    b = _make_tensor("b", [8], device)

    ttnn.conv_transpose2d(
        input_tensor=x,
        weight_tensor=w,
        bias_tensor=b,
        in_channels=16,
        out_channels=8,
        batch_size=1,
        input_height=8,
        input_width=8,
        kernel_size=(2, 2),
        stride=(2, 2),
        padding=(0, 0),
        dilation=(1, 1),
        groups=1,
        device=device,
        output_padding=(0, 0),
    )

    seq = _op_sequence(device)
    assert len(seq) == 2, f"Expected 2 ops (Halo+ConvTranspose), got {[s[0] for s in seq]}"
    assert seq[0][0] == "Halo"
    assert seq[1][0] == "ConvTranspose"

    halo_out_name = seq[0][1].outList[0]
    assert halo_out_name in seq[1][1].inList


# ---------------------------------------------------------------------------
# Multiple ops on same device: each call adds its own Halo
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_multiple_conv2d_each_get_own_halo():
    device = _make_device()

    for i in range(3):
        x = _make_tensor(f"x{i}", [1, 8, 4, 4], device)
        w = _make_tensor(f"w{i}", [8, 8, 3, 3], device)
        b = _make_tensor(f"b{i}", [8], device)
        ttnn.conv2d(
            input_tensor=x, weight_tensor=w, bias_tensor=b,
            in_channels=8, out_channels=8, batch_size=1,
            input_height=4, input_width=4,
            kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
            dilation=(1, 1), groups=1, device=device,
        )

    seq = _op_sequence(device)
    halo_count = sum(1 for optype, _ in seq if optype == "Halo")
    conv_count = sum(1 for optype, _ in seq if optype == "Conv")
    assert halo_count == 3
    assert conv_count == 3
    # Order: Halo, Conv, Halo, Conv, Halo, Conv
    for i in range(3):
        assert seq[i * 2][0] == "Halo"
        assert seq[i * 2 + 1][0] == "Conv"


# ---------------------------------------------------------------------------
# 1×1 conv: hardware uses matmul, so no Halo should be emitted
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_conv2d_1x1_no_halo():
    device = _make_device()
    x = _make_tensor("x", [1, 64, 256, 256], device)
    w = _make_tensor("w", [1, 64, 1, 1], device)
    b = _make_tensor("b", [1], device)

    ttnn.conv2d(
        input_tensor=x, weight_tensor=w, bias_tensor=b,
        in_channels=64, out_channels=1, batch_size=1,
        input_height=256, input_width=256,
        kernel_size=(1, 1), stride=(1, 1), padding=(0, 0),
        dilation=(1, 1), groups=1, device=device,
    )

    seq = _op_sequence(device)
    assert len(seq) == 1, f"Expected 1 op (MatMul only, no Halo for 1×1), got {[s[0] for s in seq]}"
    assert seq[0][0] == "MatMul"


@pytest.mark.unit
def test_conv2d_1x1_matmul_shape_passthrough():
    """1×1 conv emits MatMul with correct NCHW output shape [N, C_out, H, W]."""
    device = _make_device()
    x = _make_tensor('x', [1, 64, 32, 32], device)
    w = _make_tensor('w', [128, 64, 1, 1], device)
    b = _make_tensor('b', [128], device)

    out = ttnn.conv2d(
        input_tensor=x, weight_tensor=w, bias_tensor=b,
        in_channels=64, out_channels=128, batch_size=1,
        input_height=32, input_width=32,
        kernel_size=(1, 1), stride=(1, 1), padding=(0, 0),
        dilation=(1, 1), groups=1, device=device,
    )

    seq = _op_sequence(device)
    assert seq[0][0] == 'MatMul'
    assert list(out.shape) == [1, 128, 32, 32]


@pytest.mark.unit
def test_conv2d_1x1_stride2_matmul_shape():
    """1×1 conv with stride=2 emits MatMul with halved spatial dims."""
    device = _make_device()
    x = _make_tensor('x', [1, 64, 32, 32], device)
    w = _make_tensor('w', [128, 64, 1, 1], device)
    b = _make_tensor('b', [128], device)

    out = ttnn.conv2d(
        input_tensor=x, weight_tensor=w, bias_tensor=b,
        in_channels=64, out_channels=128, batch_size=1,
        input_height=32, input_width=32,
        kernel_size=(1, 1), stride=(2, 2), padding=(0, 0),
        dilation=(1, 1), groups=1, device=device,
    )

    assert list(out.shape) == [1, 128, 16, 16]


# ---------------------------------------------------------------------------
# InterleavedToSharded auto-emission: interleaved input → ITS → Halo → Conv
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_conv2d_interleaved_input_emits_its_then_halo():
    device = _make_device()
    x = _make_tensor("x", [1, 64, 16, 16], device)
    x._memory_config = MemoryConfig(TensorMemoryLayout.INTERLEAVED, BufferType.L1)
    w = _make_tensor("w", [128, 64, 3, 3], device)
    b = _make_tensor("b", [128], device)

    ttnn.conv2d(
        input_tensor=x, weight_tensor=w, bias_tensor=b,
        in_channels=64, out_channels=128, batch_size=1,
        input_height=16, input_width=16,
        kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
        dilation=(1, 1), groups=1, device=device,
    )

    seq = _op_sequence(device)
    assert len(seq) == 3, f"Expected ITS+Halo+Conv, got {[s[0] for s in seq]}"
    assert seq[0][0] == "InterleavedToSharded"
    assert seq[1][0] == "Halo"
    assert seq[2][0] == "Conv"

    its_out_name = seq[0][1].outList[0]
    assert its_out_name in seq[1][1].inList


@pytest.mark.unit
def test_conv2d_sharded_input_no_its():
    """Sharded input should only emit Halo → Conv, no ITS."""
    device = _make_device()
    x = _make_tensor("x", [1, 64, 16, 16], device)
    x._memory_config = MemoryConfig(TensorMemoryLayout.HEIGHT_SHARDED, BufferType.L1)
    w = _make_tensor("w", [128, 64, 3, 3], device)
    b = _make_tensor("b", [128], device)

    ttnn.conv2d(
        input_tensor=x, weight_tensor=w, bias_tensor=b,
        in_channels=64, out_channels=128, batch_size=1,
        input_height=16, input_width=16,
        kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
        dilation=(1, 1), groups=1, device=device,
    )

    seq = _op_sequence(device)
    assert len(seq) == 2, f"Expected Halo+Conv only (sharded input), got {[s[0] for s in seq]}"
    assert seq[0][0] == "Halo"
    assert seq[1][0] == "Conv"


@pytest.mark.unit
def test_conv2d_no_memory_config_no_its():
    """Tensor with no _memory_config should only emit Halo → Conv."""
    device = _make_device()
    x = _make_tensor("x", [1, 64, 16, 16], device)
    # no _memory_config set
    w = _make_tensor("w", [128, 64, 3, 3], device)
    b = _make_tensor("b", [128], device)

    ttnn.conv2d(
        input_tensor=x, weight_tensor=w, bias_tensor=b,
        in_channels=64, out_channels=128, batch_size=1,
        input_height=16, input_width=16,
        kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
        dilation=(1, 1), groups=1, device=device,
    )

    seq = _op_sequence(device)
    assert len(seq) == 2, f"Expected Halo+Conv only (no _mc), got {[s[0] for s in seq]}"
    assert seq[0][0] == "Halo"
    assert seq[1][0] == "Conv"


# ---------------------------------------------------------------------------
# Move auto-emission: requires deallocate_activation=True AND L1-sharded input
# ---------------------------------------------------------------------------

def _make_l1_sharded_tensor(name, shape, device, layout=TensorMemoryLayout.HEIGHT_SHARDED):
    """Helper: tensor with L1 sharded _memory_config so _with_move fires."""
    t = _make_tensor(name, shape, device)
    t._memory_config = MemoryConfig(layout, BufferType.L1)
    return t


@pytest.mark.unit
def test_conv2d_deallocate_l1sharded_emits_move():
    """deallocate_activation=True + L1-sharded input → Halo+Move+Conv (hardware order)."""
    device = _make_device()
    x = _make_l1_sharded_tensor('x', [1, 8, 4, 4], device)
    w = _make_tensor('w', [8, 8, 3, 3], device)
    b = _make_tensor('b', [8], device)

    out = ttnn.conv2d(
        input_tensor=x, weight_tensor=w, bias_tensor=b,
        in_channels=8, out_channels=8, batch_size=1,
        input_height=4, input_width=4,
        kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
        dilation=(1, 1), groups=1, device=device,
        deallocate_activation=True,
    )

    seq = _op_sequence(device)
    assert len(seq) == 3, f'Expected Halo+Move+Conv, got {[s[0] for s in seq]}'
    assert seq[0][0] == 'Halo'
    assert seq[1][0] == 'Move'
    assert seq[2][0] == 'Conv'
    halo_out_name = seq[0][1].outList[0]
    assert halo_out_name in seq[1][1].inList
    move_out_name = seq[1][1].outList[0]
    assert move_out_name in seq[2][1].inList
    assert out.name == seq[2][1].outList[0]


@pytest.mark.unit
def test_conv2d_deallocate_no_memory_config_no_move():
    """deallocate_activation=True but no _memory_config → no Move (DRAM default)."""
    device = _make_device()
    x = _make_tensor('x', [1, 8, 4, 4], device)   # no _memory_config set
    w = _make_tensor('w', [8, 8, 3, 3], device)
    b = _make_tensor('b', [8], device)

    ttnn.conv2d(
        input_tensor=x, weight_tensor=w, bias_tensor=b,
        in_channels=8, out_channels=8, batch_size=1,
        input_height=4, input_width=4,
        kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
        dilation=(1, 1), groups=1, device=device,
        deallocate_activation=True,
    )

    seq = _op_sequence(device)
    assert len(seq) == 2, f'Expected Halo+Conv only (no mc), got {[s[0] for s in seq]}'
    assert seq[0][0] == 'Halo'
    assert seq[1][0] == 'Conv'


@pytest.mark.unit
def test_conv2d_deallocate_interleaved_l1_emits_move_after_its():
    """deallocate_activation=True with interleaved L1 input → auto-ITS converts
    to L1-sharded, then Move fires (mirroring tt-metal where decoder convs with
    do_sharded_to_interleaved=True still emit Move post-halo)."""
    device = _make_device()
    x = _make_tensor('x', [1, 8, 4, 4], device)
    x._memory_config = MemoryConfig(TensorMemoryLayout.INTERLEAVED, BufferType.L1)
    w = _make_tensor('w', [8, 8, 3, 3], device)
    b = _make_tensor('b', [8], device)

    ttnn.conv2d(
        input_tensor=x, weight_tensor=w, bias_tensor=b,
        in_channels=8, out_channels=8, batch_size=1,
        input_height=4, input_width=4,
        kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
        dilation=(1, 1), groups=1, device=device,
        deallocate_activation=True,
    )

    seq = _op_sequence(device)
    optypes = [s[0] for s in seq]
    # Sequence should be ITS → Halo → Move → Conv
    assert optypes == ['InterleavedToSharded', 'Halo', 'Move', 'Conv'], optypes


@pytest.mark.unit
def test_conv2d_no_deallocate_no_move():
    """deallocate_activation=False → no Move regardless of memory config."""
    device = _make_device()
    x = _make_l1_sharded_tensor('x', [1, 8, 4, 4], device)
    w = _make_tensor('w', [8, 8, 3, 3], device)
    b = _make_tensor('b', [8], device)

    ttnn.conv2d(
        input_tensor=x, weight_tensor=w, bias_tensor=b,
        in_channels=8, out_channels=8, batch_size=1,
        input_height=4, input_width=4,
        kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
        dilation=(1, 1), groups=1, device=device,
        deallocate_activation=False,
    )

    seq = _op_sequence(device)
    assert all(s[0] != 'Move' for s in seq), f'Unexpected Move with deallocate=False: {[s[0] for s in seq]}'


@pytest.mark.unit
def test_conv2d_move_shape_passthrough():
    """Move output shape must equal Conv output shape."""
    device = _make_device()
    x = _make_l1_sharded_tensor('x', [1, 8, 8, 8], device)
    w = _make_tensor('w', [16, 8, 3, 3], device)
    b = _make_tensor('b', [16], device)

    out = ttnn.conv2d(
        input_tensor=x, weight_tensor=w, bias_tensor=b,
        in_channels=8, out_channels=16, batch_size=1,
        input_height=8, input_width=8,
        kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
        dilation=(1, 1), groups=1, device=device,
        deallocate_activation=True,
    )

    assert out.shape == [1, 16, 8, 8]


@pytest.mark.unit
def test_conv2d_1x1_deallocate_l1sharded_no_halo_no_move():
    """1×1 conv → MatMul only (no Halo, no trailing Move).

    HW's 1×1-conv → MatMul path does not emit MoveDeviceOperation regardless of
    ``deallocate_activation`` (the HW profiler shows no Move row at the 1×1 conv
    output shape), so the shim drops the trailing Move emission.
    """
    device = _make_device()
    x = _make_l1_sharded_tensor('x', [1, 64, 16, 16], device)
    w = _make_tensor('w', [32, 64, 1, 1], device)
    b = _make_tensor('b', [32], device)

    ttnn.conv2d(
        input_tensor=x, weight_tensor=w, bias_tensor=b,
        in_channels=64, out_channels=32, batch_size=1,
        input_height=16, input_width=16,
        kernel_size=(1, 1), stride=(1, 1), padding=(0, 0),
        dilation=(1, 1), groups=1, device=device,
        deallocate_activation=True,
    )

    seq = _op_sequence(device)
    optypes = [s[0] for s in seq]
    assert optypes == ['MatMul'], f'Expected MatMul only (no Halo, no Move for 1×1), got {optypes}'


@pytest.mark.unit
def test_conv2d_block_sharded_deallocate_emits_move():
    """BLOCK_SHARDED L1 input with deallocate=True also triggers Move."""
    device = _make_device()
    x = _make_l1_sharded_tensor('x', [1, 16, 8, 8], device, TensorMemoryLayout.BLOCK_SHARDED)
    w = _make_tensor('w', [16, 16, 3, 3], device)
    b = _make_tensor('b', [16], device)

    ttnn.conv2d(
        input_tensor=x, weight_tensor=w, bias_tensor=b,
        in_channels=16, out_channels=16, batch_size=1,
        input_height=8, input_width=8,
        kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
        dilation=(1, 1), groups=1, device=device,
        deallocate_activation=True,
    )

    seq = _op_sequence(device)
    optypes = [s[0] for s in seq]
    assert 'Move' in optypes, f'Expected Move in sequence, got {optypes}'
    move_idx = optypes.index('Move')
    assert move_idx < optypes.index('Conv'), f'Expected Move before Conv, got {optypes}'


# ---------------------------------------------------------------------------
# to_memory_config: sharded→different-sharded emits STI+ITS
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_to_memory_config_sharded_to_different_sharded_emits_sti_its():
    device = _make_device()
    x = _make_tensor("x", [1, 64, 32, 32], device)
    x._memory_config = MemoryConfig(TensorMemoryLayout.BLOCK_SHARDED, BufferType.L1)

    target_mc = MemoryConfig(TensorMemoryLayout.HEIGHT_SHARDED, BufferType.L1)
    ttnn.to_memory_config(x, target_mc)

    seq = _op_sequence(device)
    assert len(seq) == 2, f"Expected STI+ITS, got {[s[0] for s in seq]}"
    assert seq[0][0] == "ShardedToInterleaved"
    assert seq[1][0] == "InterleavedToSharded"


@pytest.mark.unit
def test_to_memory_config_same_sharded_no_ops():
    """Same sharded config → no STI+ITS emitted."""
    device = _make_device()
    x = _make_tensor("x", [1, 64, 32, 32], device)
    mc = MemoryConfig(TensorMemoryLayout.HEIGHT_SHARDED, BufferType.L1)
    x._memory_config = mc

    ttnn.to_memory_config(x, mc)

    seq = _op_sequence(device)
    assert len(seq) == 0, f"Expected no ops for same config, got {[s[0] for s in seq]}"


@pytest.mark.unit
def test_to_memory_config_interleaved_to_sharded_no_sti():
    """Interleaved→sharded via to_memory_config: no STI (only ITS would come later via _with_halo)."""
    device = _make_device()
    x = _make_tensor("x", [1, 64, 32, 32], device)
    x._memory_config = MemoryConfig(TensorMemoryLayout.INTERLEAVED, BufferType.L1)

    target_mc = MemoryConfig(TensorMemoryLayout.HEIGHT_SHARDED, BufferType.L1)
    ttnn.to_memory_config(x, target_mc)

    seq = _op_sequence(device)
    assert len(seq) == 0, f"Expected no ops (interleaved→sharded not a reshard), got {[s[0] for s in seq]}"


# ---------------------------------------------------------------------------
# to_memory_config: interleaved→different-interleaved emits Copy (tt-metal prim::copy fallback)
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_to_memory_config_interleaved_l1_to_dram_emits_one_copy():
    device = _make_device()
    x = _make_tensor("x", [1, 1, 32, 128256], device)
    l1 = MemoryConfig(TensorMemoryLayout.INTERLEAVED, BufferType.L1)
    dram = MemoryConfig(TensorMemoryLayout.INTERLEAVED, BufferType.DRAM)
    x._memory_config = l1

    out = ttnn.to_memory_config(x, dram)

    seq = _op_sequence(device)
    assert [s[0] for s in seq] == ["Copy"]
    assert out is not x
    assert out._memory_config == dram
    assert x._memory_config == l1  # the input tensor is not relabelled
    assert out.logical_shape().as_list() == x.logical_shape().as_list()


@pytest.mark.unit
def test_to_memory_config_same_interleaved_no_ops():
    device = _make_device()
    x = _make_tensor("x", [1, 1, 32, 64], device)
    dram = MemoryConfig(TensorMemoryLayout.INTERLEAVED, BufferType.DRAM)
    x._memory_config = dram

    out = ttnn.to_memory_config(x, MemoryConfig(TensorMemoryLayout.INTERLEAVED, BufferType.DRAM))

    assert _op_sequence(device) == []
    assert out is x


@pytest.mark.unit
def test_copy_optype_matches_profiler_canonical():
    # The LUT key must match between the shim op and the profiler's CopyDeviceOperation.
    from tools.profiling.op_canonical import normalize_polaris_optype, normalize_profiler_opcode

    assert normalize_polaris_optype("Copy") == normalize_profiler_opcode("CopyDeviceOperation")
