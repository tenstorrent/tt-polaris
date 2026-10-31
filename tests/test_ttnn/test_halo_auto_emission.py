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
    from ttsim.utils.shim_gates import set_move_on_reallocate_halo_output
    device = _make_device()
    # The Move below exists only under set_move_on_reallocate_halo_output; OFF
    # keys on deallocate_activation, which this call leaves at its False default.
    set_move_on_reallocate_halo_output(True, device)
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

    # ITS -> Halo -> Move -> Conv.  This asserted ITS+Halo+Conv (no Move) until the
    # Move was re-keyed from `deallocate_activation` to `reallocate_halo_output`
    # (tt-metal/ttnn/cpp/ttnn/operations/conv/conv2d/conv2d.cpp:297-299 gates ttnn::move on the latter; the former
    # drives
    # only the deallocate at :291-293 and emits no op).  This call passes no
    # conv_config, and `reallocate_halo_output` defaults to True
    # (tt-metal/ttnn/cpp/ttnn/operations/conv/conv2d/conv2d_nanobind.cpp:267), so hardware does dispatch a Move here.
    seq = _op_sequence(device)
    assert len(seq) == 4, f"Expected ITS+Halo+Move+Conv, got {[s[0] for s in seq]}"
    assert seq[0][0] == "InterleavedToSharded"
    assert seq[1][0] == "Halo"
    assert seq[2][0] == "Move"
    assert seq[3][0] == "Conv"

    its_out_name = seq[0][1].outList[0]
    assert its_out_name in seq[1][1].inList


@pytest.mark.unit
def test_conv2d_sharded_input_no_its():
    """Sharded input emits no ITS.  The point of this test is the ABSENCE of ITS.

    Left on shim defaults, so the sequence is Halo -> Conv: the pre-conv Move is
    gated behind ``set_move_on_reallocate_halo_output`` and this call sets
    neither that gate nor ``deallocate_activation``.  The Move itself is covered
    by ``test_move_gate_off_reads_deallocate_activation``; the no-ITS assertion,
    which is what this test exists for, holds either way.
    """
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
    assert "InterleavedToSharded" not in [s[0] for s in seq], \
        f"sharded input must not get an ITS, got {[s[0] for s in seq]}"
    assert len(seq) == 2, f"Expected Halo+Conv (sharded input), got {[s[0] for s in seq]}"
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
def test_move_is_gated_on_reallocate_halo_output_not_deallocate_activation():
    """Which flag gates the pre-conv Move.

    This replaces ``test_conv2d_no_deallocate_no_move``, which asserted
    ``deallocate_activation=False -> no Move``. That was the shim's old rule and it
    is not hardware's. tt-metal gates ``ttnn::move`` on ``reallocate_halo_output``::

        ttnn/cpp/ttnn/operations/conv/conv2d/conv2d.cpp:297-299
            input_tensor_post_tm = std::move(halo_output);
            if (conv_config.reallocate_halo_output) {
                input_tensor_post_tm = ttnn::move(input_tensor_post_tm);
            }

    ``deallocate_activation`` drives the deallocate at :291-293, which is a
    different thing and emits no op. Keying the Move off it made polaris emit one
    at every conv with ``deallocate_activation=True`` — 17 for ResNet-50 where the
    capture shows 2.
    """
    from ttsim.utils.shim_gates import set_move_on_reallocate_halo_output

    def _seq(**extra):
        device = _make_device()
        # This rule IS the gate; OFF restores the deallocate_activation keying.
        set_move_on_reallocate_halo_output(True, device)
        x = _make_l1_sharded_tensor('x', [1, 8, 4, 4], device)
        w = _make_tensor('w', [8, 8, 3, 3], device)
        b = _make_tensor('b', [8], device)
        ttnn.conv2d(
            input_tensor=x, weight_tensor=w, bias_tensor=b,
            in_channels=8, out_channels=8, batch_size=1,
            input_height=4, input_width=4,
            kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
            dilation=(1, 1), groups=1, device=device, **extra,
        )
        return [s[0] for s in _op_sequence(device)]

    # deallocate_activation does NOT gate the Move: reallocate_halo_output defaults
    # True (tt-metal/ttnn/cpp/ttnn/operations/conv/conv2d/conv2d_nanobind.cpp:267), so hardware moves either way.
    assert 'Move' in _seq(deallocate_activation=False)
    assert 'Move' in _seq(deallocate_activation=True)

    # reallocate_halo_output DOES gate it. ResNet-50 sets this False on its
    # downsample and conv2, which is how the capture ends at 2 Moves.
    assert 'Move' not in _seq(reallocate_halo_output=False)
    assert 'Move' not in _seq(deallocate_activation=True, reallocate_halo_output=False)


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
# to_memory_config: sharded→different-sharded reshards
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_to_memory_config_sharded_to_different_sharded_emits_reshard():
    """A sharded -> different-sharded change is ONE Reshard, not STI+ITS.

    This asserted STI+ITS until ``to_memory_config`` was made reshard-primary.
    tt-metal reshards directly and treats the round-trip through interleaved as a
    fallback for the cases reshard cannot express::

        ttnn/cpp/ttnn/operations/data_movement/to_memory_config/device/
            tt-metal/ttnn/cpp/ttnn/operations/core/to_memory_config/to_memory_config_op.cpp:279-320

    The capture agrees: ResNet-50 p100a logs 9 ReshardDeviceOperation rows and no
    STI/ITS pair between two sharded configs. ``_needs_reshard_workaround`` still
    routes to STI+ITS where the C++ does.
    """
    from ttsim.front.ttnn.buffer import ShardSpec
    from ttsim.front.ttnn.core import CoreGrid
    from ttsim.utils.shim_gates import set_to_memory_config_reshard
    device = _make_device()
    # Reshard-primary is gated; OFF keeps the historical STI+ITS pair, which is
    # what test_to_memory_config_gate_off_emits_sti_its_pair pins.
    set_to_memory_config_reshard(True, device)

    # BOTH configs carry a real shard spec, and the shard WIDTHS match.
    # _needs_reshard_workaround needs both specs to evaluate the cpp predicate
    # at all -- a spec-less MemoryConfig is unevaluable and now takes the
    # conservative STI+ITS pair -- and with equal widths the predicate is
    # False, which is the Reshard case.  The original version of this test used
    # spec-less configs and so never reached the decision it claimed to pin.
    _grid = CoreGrid(x=8, y=8)
    x = _make_tensor("x", [1, 64, 32, 32], device)
    x._memory_config = MemoryConfig(
        TensorMemoryLayout.BLOCK_SHARDED, BufferType.L1, ShardSpec(_grid, (128, 64)))

    target_mc = MemoryConfig(
        TensorMemoryLayout.HEIGHT_SHARDED, BufferType.L1, ShardSpec(_grid, (256, 64)))
    ttnn.to_memory_config(x, target_mc)

    seq = _op_sequence(device)
    assert [s[0] for s in seq] == ["Reshard"], f"Expected one Reshard, got {[s[0] for s in seq]}"


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
# Halo-Move suppression is keyed by declared SKU, never by architecture
# ---------------------------------------------------------------------------
#
# The suppression table encodes an L1 allocator outcome measured on one card.
# It used to be keyed on ``device.architecture``, which handed p100a's result to
# every Blackhole device — including p150a and p150b, which config/tt_bh.yaml
# declares alongside p100a and which nobody has captured.  These pin the new
# contract: the key is ``Device.capture_sku``, and an undeclared SKU misses.

def _conv_emitting_halo(device, deallocate=True):
    """Halo(+Move)+Conv on an L1-sharded input; returns the op-type sequence.

    The input carries an explicit ``hw_shape``.  The guard keys on the halo
    OUTPUT's NHWC-flat view, and ``halo_sinf`` only synthesises one when the
    geometry is in ``_HALO_EXT_Y`` or in this device's block of
    ``_HALO_EXT_Y_OVERRIDES_BY_DEVICE`` — (16, 8, 3, 3, 1, 1, False) is in
    neither, so it
    propagates the input's instead (ttsim/ops/desc/ttsim_layout.py:696) and an input without one
    leaves the guard unreachable.  Real conv inputs always have it: the workload
    hands ttnn an NHWC-flat activation, and ``conv_sinf`` / ``maxpool`` set it on
    every downstream tensor (nn.py:276, 587).
    """
    x = _make_l1_sharded_tensor('x', [1, 8, 4, 4], device)
    x.hw_shape = [1, 1, 1 * 4 * 4, 8]   # NCHW [1,8,4,4] -> NHWC-flat [1,1,N*H*W,C]
    w = _make_tensor('w', [8, 8, 3, 3], device)
    b = _make_tensor('b', [8], device)
    ttnn.conv2d(
        input_tensor=x, weight_tensor=w, bias_tensor=b,
        in_channels=8, out_channels=8, batch_size=1,
        input_height=4, input_width=4,
        kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
        dilation=(1, 1), groups=1, device=device,
        deallocate_activation=deallocate,
    )
    return [ot for ot, _ in _op_sequence(device)]


def _halo_output_hw_key(device):
    """The (nhw, channels) the guard in ``_with_halo`` sees, for this geometry."""
    halo = next(op for ot, op in _op_sequence(device) if ot == 'Halo')
    hw = getattr(device.tensors[halo.outList[0]], 'hw_shape', None)
    assert hw is not None and len(hw) >= 4, (
        'test calibration failed: the halo output carries no hw_shape, so the '
        f'suppression guard could never fire for this geometry (hw_shape={hw!r})'
    )
    return (int(hw[2]), int(hw[3]))


@pytest.mark.unit
def test_capture_sku_defaults_to_none():
    """An undeclared SKU is the safe default — capture tables must miss on it."""
    assert Device(device_id=0).capture_sku is None
    d = Device(device_id=0)
    d.set_capture_sku('p100a')
    assert d.capture_sku == 'p100a'
    d.set_capture_sku(None)
    assert d.capture_sku is None


@pytest.mark.unit
def test_suppression_table_is_keyed_by_sku_not_arch():
    """Keys must be arch-spec device instances, not architecture names.

    A key like 'blackhole' would cover p150a and p150b too, which is exactly the
    over-application this table must not make.
    """
    from ttsim.front.ttnn.device import ARCH
    from ttsim.ops.desc.ttsim_layout import (
        _HALO_EXT_Y_OVERRIDES_BY_DEVICE,
        _HALO_MOVE_SUPPRESSED_BY_SKU,
    )
    arch_names = {a.cname for a in ARCH}
    assert not (set(_HALO_MOVE_SUPPRESSED_BY_SKU) & arch_names), (
        'suppression table is keyed by architecture; it must be keyed by SKU'
    )
    # Same key space as the sibling capture table, so the two stay comparable.
    assert set(_HALO_MOVE_SUPPRESSED_BY_SKU) <= set(_HALO_EXT_Y_OVERRIDES_BY_DEVICE), (
        'suppression table names a SKU the extended-Y table does not know'
    )
    # Only cards with a capture may appear.  p150a/p150b/n300 are declared in the
    # arch YAMLs but have never been profiled for this workload.
    assert not (set(_HALO_MOVE_SUPPRESSED_BY_SKU) & {'p150a', 'p150b', 'n300'})


@pytest.mark.unit
def test_move_suppressed_only_for_the_declared_sku(monkeypatch):
    """The whole point: same arch, same geometry, different SKU -> different outcome."""
    import ttsim.ops.desc.ttsim_layout as layout

    # Calibrate against this geometry with the table empty, so the key is the one
    # the guard actually computes rather than one hardcoded here.
    monkeypatch.setattr(layout, '_HALO_MOVE_SUPPRESSED_BY_SKU', {}, raising=True)
    probe = _make_device()
    assert _conv_emitting_halo(probe) == ['Halo', 'Move', 'Conv']
    key = _halo_output_hw_key(probe)
    assert key == (16, 8), f'geometry drifted; the rest of this test keys on {key}'

    monkeypatch.setattr(
        layout, '_HALO_MOVE_SUPPRESSED_BY_SKU', {'captured_sku': {key}}, raising=True)

    declared = _make_device()
    declared.set_capture_sku('captured_sku')
    assert _conv_emitting_halo(declared) == ['Halo', 'Conv'], 'declared SKU did not suppress'

    for sku, why in [(None, 'undeclared SKU inherited another card\'s measurement'),
                     ('uncaptured_sku', 'a different SKU inherited the measurement')]:
        d = _make_device()
        d.set_capture_sku(sku)
        assert _conv_emitting_halo(d) == ['Halo', 'Move', 'Conv'], why


# ---------------------------------------------------------------------------
# The two gates that keep non-ResNet workloads on main's op sequence
# ---------------------------------------------------------------------------
#
# Both were unconditional before, and both cost vgg_unet p100a a LUT hit:
# 102/102 -> 100/102.  OFF must reproduce main exactly; ON is what ResNet-50
# opts into.

@pytest.mark.unit
def test_move_gate_off_reads_deallocate_activation():
    """OFF: the Move follows deallocate_activation, which defaults False.

    A conv that sets neither flag therefore emits no Move — main's behaviour,
    and what the vgg_unet capture shows at its d4 ConvTranspose (Halo 256->1320
    straight into Conv2d).
    """
    from ttsim.utils.shim_gates import set_move_on_reallocate_halo_output
    dev = _make_device()
    assert not hasattr(dev, 'move_on_reallocate_halo_output'), 'gate must default OFF'
    # Neither flag set -> no Move.
    assert _conv_emitting_halo(dev, deallocate=False) == ['Halo', 'Conv']
    # deallocate_activation=True is what OFF listens to.
    assert _conv_emitting_halo(_make_device(), deallocate=True) == ['Halo', 'Move', 'Conv']

    # ON flips which flag is read: reallocate_halo_output defaults True, so a
    # conv that sets neither now DOES emit the Move.
    on = _make_device()
    set_move_on_reallocate_halo_output(True, on)
    try:
        assert _conv_emitting_halo(on, deallocate=False) == ['Halo', 'Move', 'Conv']
    finally:
        set_move_on_reallocate_halo_output(False, on)


@pytest.mark.unit
def test_to_memory_config_gate_off_emits_sti_its_pair():
    """OFF: sharded -> different-sharded goes through STI+ITS, never Reshard.

    tt-metal calls ttnn::reshard first, but deciding that needs both shard specs
    to evaluate use_reshard_workaround; a workload that does not propagate
    memory configs has none, and guessing Reshard cost vgg_unet the STI+ITS pair
    its p100a capture records at rows 20-21.
    """
    import ttsim.front.ttnn as ttnn
    from ttsim.front.ttnn.buffer import BufferType, TensorMemoryLayout
    from ttsim.front.ttnn.memory import MemoryConfig
    from ttsim.utils.shim_gates import set_to_memory_config_reshard

    from ttsim.front.ttnn.buffer import ShardSpec
    from ttsim.front.ttnn.core import CoreGrid

    def _transition(device):
        # Real specs on both sides with equal shard widths, so the cpp
        # predicate is evaluable and False -- the Reshard case when the gate
        # allows it.  Spec-less configs are unevaluable and always take
        # STI+ITS, which would make the ON and OFF arms indistinguishable.
        _grid = CoreGrid(x=8, y=8)
        before = len(device.ops)
        t = _make_l1_sharded_tensor('x', [1, 8, 4, 4], device,
                                    layout=TensorMemoryLayout.HEIGHT_SHARDED)
        t._memory_config = MemoryConfig(
            TensorMemoryLayout.HEIGHT_SHARDED, BufferType.L1, ShardSpec(_grid, (16, 8)))
        ttnn.to_memory_config(t, MemoryConfig(
            TensorMemoryLayout.BLOCK_SHARDED, BufferType.L1, ShardSpec(_grid, (8, 8))))
        return [o.optype for o in list(device.ops.values())[before:]]

    off = _make_device()
    assert not hasattr(off, 'to_memory_config_reshard'), 'gate must default OFF'
    assert _transition(off) == ['ShardedToInterleaved', 'InterleavedToSharded']

    on = _make_device()
    set_to_memory_config_reshard(True, on)
    try:
        assert 'Reshard' in _transition(on)
    finally:
        set_to_memory_config_reshard(False, on)


# ---------------------------------------------------------------------------
# The Move-suppression guard keys off the halo INPUT, not the halo output
# ---------------------------------------------------------------------------
#
# _HALO_MOVE_SUPPRESSED_BY_SKU stores the halo INPUT's (nhw, channels) -- see
# its "Key: ... AS SEEN AT EMISSION TIME" block.  The guard used to read
# kwargs['input_tensor'] AFTER halo() had rebound it to the halo output, which
# gave the right answer only because the base _HALO_EXT_Y table happens to carry
# no ResNet-50 geometry, so halo_sinf took its passthrough branch and output ==
# input.  These tests use a geometry that IS in _HALO_EXT_Y, where the two
# differ, so the distinction is pinned rather than incidental.

#: _HALO_EXT_Y[(256, 512, 3, 3, 1, 1, False)] == 576.  NCHW [1, 512, 16, 16]
#: gives nhw = 1*16*16 = 256, so the halo INPUT keys (256, 512) and its OUTPUT
#: keys (576, 512).
_EXT_Y_NCHW = [1, 512, 16, 16]
_EXT_Y_IN_KEY = (256, 512)
_EXT_Y_OUT_KEY = (576, 512)


def _conv_on_ext_y_geometry(device):
    """Halo(+Move)+Conv on a geometry present in the base _HALO_EXT_Y table."""
    x = _make_l1_sharded_tensor('x', _EXT_Y_NCHW, device)
    x.hw_shape = [1, 1, 256, 512]
    w = _make_tensor('w', [512, 512, 3, 3], device)
    b = _make_tensor('b', [512], device)
    ttnn.conv2d(
        input_tensor=x, weight_tensor=w, bias_tensor=b,
        in_channels=512, out_channels=512, batch_size=1,
        input_height=16, input_width=16,
        kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
        dilation=(1, 1), groups=1, device=device,
        deallocate_activation=True,
    )
    return [ot for ot, _ in _op_sequence(device)]


@pytest.mark.unit
def test_the_chosen_geometry_really_does_extend_the_halo_output():
    """Positive control: without it the two tests below prove nothing.

    If _HALO_EXT_Y ever loses this entry, the halo output would fall back to the
    input's hw_shape, input and output keys would coincide, and the direction
    tests would pass no matter which side the guard read.
    """
    from ttsim.ops.desc.ttsim_layout import _HALO_EXT_Y

    assert _HALO_EXT_Y.get((256, 512, 3, 3, 1, 1, False)) == 576, (
        'the geometry these tests rely on changed; pick another entry whose '
        'ext_y differs from its nhw'
    )
    device = _make_device()
    _conv_on_ext_y_geometry(device)
    halo = next(op for ot, op in _op_sequence(device) if ot == 'Halo')
    out_hw = device.tensors[halo.outList[0]].hw_shape
    assert out_hw is not None and (int(out_hw[2]), int(out_hw[3])) == _EXT_Y_OUT_KEY
    assert _EXT_Y_IN_KEY != _EXT_Y_OUT_KEY


@pytest.mark.unit
def test_suppression_fires_on_the_halo_input_key(monkeypatch):
    """The input key suppresses.  Before the fix this did NOT suppress."""
    import ttsim.ops.desc.ttsim_layout as layout

    monkeypatch.setattr(
        layout, '_HALO_MOVE_SUPPRESSED_BY_SKU', {'captured_sku': {_EXT_Y_IN_KEY}}, raising=True)
    dev = _make_device()
    dev.set_capture_sku('captured_sku')
    assert _conv_on_ext_y_geometry(dev) == ['Halo', 'Conv'], (
        'the guard did not suppress on the halo INPUT key -- it is reading the '
        'post-halo tensor again'
    )


@pytest.mark.unit
def test_suppression_does_not_fire_on_the_halo_output_key(monkeypatch):
    """The output key must NOT suppress.  Before the fix this DID suppress.

    This is the half that catches the regression: once a ResNet-50 geometry
    lands in the base _HALO_EXT_Y, a guard reading the output would match the
    extended value and keep working locally while silently dropping the real
    stem suppression.
    """
    import ttsim.ops.desc.ttsim_layout as layout

    monkeypatch.setattr(
        layout, '_HALO_MOVE_SUPPRESSED_BY_SKU', {'captured_sku': {_EXT_Y_OUT_KEY}}, raising=True)
    dev = _make_device()
    dev.set_capture_sku('captured_sku')
    assert _conv_on_ext_y_geometry(dev) == ['Halo', 'Move', 'Conv'], (
        'the guard suppressed on the halo OUTPUT key -- it is keyed off the '
        'wrong side of the Halo'
    )
