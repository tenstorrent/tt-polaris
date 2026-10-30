# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""1x1 conv -> MatMul dispatch, as ResNet-50's downsamples exercise it.

Hardware lowers a 1x1 conv to a matmul only under the full condition in
``tt-metal/ttnn/cpp/ttnn/operations/conv/conv2d/conv2d_utils.cpp:519-531``::

    kernel_size == (1,1) && stride[0] == stride[1] && stride[0] == 1 &&
    padding == (0,0,0,0) && dilation == (1,1) && !is_width_sharded

A 1x1 kernel on its own is NOT sufficient: a strided 1x1 conv subsamples its
input, so hardware dispatches the real Conv2d kernel and the profiler records a
Conv2dDeviceOperation row.  ResNet-50 hits this at three of its four
downsamples (layer2/3/4 are 1x1 stride 2; only layer1 is stride 1).

That condition lives in the shim as ``_lowers_to_matmul``, shared by
``_with_halo`` and ``_conv2d_dispatch`` so the two cannot disagree; the
predicate's own truth table is pinned by test_lowers_to_matmul.py.  What these
tests pin is the resulting EMISSION: that ``ttnn.conv2d`` — and the workload's
``_emit_conv2d``, now a direct passthrough to it — produce Halo + Conv2d for a
strided 1x1 and a bare MatMul for an unstrided one.

This file previously also tested a workload-local copy of the predicate, which
existed while the shim's dispatch still tested the kernel size alone.  Both the
copy and the Conv-only routing around it are gone.
"""

import pytest

import ttsim.front.ttnn as ttnn
from ttsim.front.ttnn.device import Device
from ttsim.front.ttnn.tensor import DataType, Tensor
from workloads.ttnn.resnet50.common.ttnn_functional_resnet50 import _emit_conv2d


def _conv_kwargs(device, *, stride, padding=(0, 0), dilation=(1, 1), shard_layout=None):
    if shard_layout is None:
        shard_layout = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    return {
        'input_tensor': Tensor(shape=[1, 64, 16, 16], dtype=DataType.BFLOAT8_B, device=device),
        'weight_tensor': Tensor(shape=[128, 64, 1, 1], dtype=DataType.BFLOAT8_B, device=device),
        'bias_tensor': Tensor(shape=[128], dtype=DataType.BFLOAT8_B, device=device),
        'in_channels': 64,
        'out_channels': 128,
        'batch_size': 1,
        'input_height': 16,
        'input_width': 16,
        'kernel_size': (1, 1),
        'stride': stride,
        'padding': padding,
        'dilation': dilation,
        'groups': 1,
        'device': device,
        'conv_config': ttnn.Conv2dConfig(
            weights_dtype=DataType.BFLOAT8_B,
            shard_layout=shard_layout,
        ),
    }


def _emit(**kw):
    """Route one 1x1 conv through the workload helper; return emitted op types."""
    device = Device(device_id=0)
    _emit_conv2d(**_conv_kwargs(device, **kw))
    return [op.optype for op in device.ops.values()]


# --------------------------------------------------------------------------
# what the workload helper emits
# --------------------------------------------------------------------------

@pytest.mark.unit
def test_1x1_stride1_lowers_to_matmul() -> None:
    optypes = _emit(stride=(1, 1))
    assert 'MatMul' in optypes
    assert 'Conv' not in optypes


@pytest.mark.unit
def test_1x1_stride2_stays_conv() -> None:
    """ResNet-50 layer2/3/4 downsample."""
    optypes = _emit(stride=(2, 2))
    assert 'Conv' in optypes
    assert 'MatMul' not in optypes


@pytest.mark.unit
def test_forcing_conv_also_emits_a_halo() -> None:
    """A 1x1 stride-2 conv DOES get a halo — this assertion used to say the opposite.

    The original version asserted ``'Halo' not in optypes``, on the reasoning
    that emitting one "would diverge from HW and leave three polaris-only ops".
    The p100a capture says the reverse: it records 22 Halo
    against 20 Conv + 2 Pool, i.e. one per conv including these three, and
    bh_p100a_lut_v6 carries halo entries keyed ``kernel=1x1 stride=2x2
    padding=0x0``. Polaris was emitting 19 and the missing three were exactly
    these downsamples.

    The old expectation came from ``_with_halo`` gating on ``kernel != (1, 1)``
    alone, which is narrower than tt-metal's ``use_matmul_for_1x1_conv``
    (``tt-metal/ttnn/cpp/ttnn/operations/conv/conv2d/conv2d_utils.cpp:519-531``):
    that also requires stride 1, padding 0, dilation 1 and not width-sharded.
    See tests/test_ttnn/test_halo_1x1_stride_gate.py.
    """
    optypes = _emit(stride=(2, 2))
    assert 'Halo' in optypes


@pytest.mark.unit
def test_resnet50_downsample_split_is_one_matmul_three_conv() -> None:
    """layer1 downsample is stride 1 (matmul); layers 2-4 are stride 2 (conv)."""
    kinds = []
    for stride in [(1, 1), (2, 2), (2, 2), (2, 2)]:
        optypes = _emit(stride=stride)
        kinds.append('MatMul' if 'MatMul' in optypes else 'Conv')
    assert kinds == ['MatMul', 'Conv', 'Conv', 'Conv']


# --------------------------------------------------------------------------
# the same, entered through the shim directly rather than the workload helper
# --------------------------------------------------------------------------

@pytest.mark.unit
def test_strided_1x1_is_halo_plus_conv_not_matmul() -> None:
    """A 1x1 STRIDE-2 conv must be Halo + Conv2d, never Halo + MatMul.

    `use_matmul_for_1x1_conv`
    (tt-metal/ttnn/cpp/ttnn/operations/conv/conv2d/conv2d_utils.cpp:519-531) requires stride 1, so a
    strided 1x1 stays a real convolution and silicon dispatches Halo + Conv2d.

    This replaces a test that pinned the opposite as deliberate. Its reasoning
    was wrong: `_with_halo` had already been corrected to the full predicate
    while `_conv2d_dispatch` still tested the kernel alone, so the shim was not
    "lowering every 1x1 to MatMul" — it was emitting Halo + MatMul, a pair that
    cannot occur on hardware. Both sites now share `_lowers_to_matmul`.
    """
    device = Device(device_id=0)
    ttnn.conv2d(**_conv_kwargs(device, stride=(2, 2)))
    optypes = [op.optype for op in device.ops.values()]
    assert 'Halo' in optypes, optypes
    assert 'Conv' in optypes, optypes
    assert 'MatMul' not in optypes, f'impossible Halo+MatMul pair: {optypes}'


def test_unstrided_1x1_still_lowers_to_matmul() -> None:
    """The other side of the predicate: a plain 1x1 has no Halo and is a MatMul."""
    device = Device(device_id=0)
    ttnn.conv2d(**_conv_kwargs(device, stride=(1, 1)))
    optypes = [op.optype for op in device.ops.values()]
    assert 'MatMul' in optypes, optypes
    assert 'Halo' not in optypes, optypes
