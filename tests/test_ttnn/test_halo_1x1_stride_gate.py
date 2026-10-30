# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Halo is skipped for matmul-lowered convs, not for every 1x1 kernel.

``_with_halo`` used to gate on ``kernel_size != (1, 1)``, reasoning that
hardware turns 1x1 convs into matmuls and never dispatches a halo. That is only
half the rule. tt-metal's ``use_matmul_for_1x1_conv``
(``tt-metal/ttnn/cpp/ttnn/operations/conv/conv2d/conv2d_utils.cpp:519-531``) requires::

    kernel 1x1  AND  stride 1  AND  padding 0  AND  dilation 1  AND  not width-sharded

so a 1x1 conv at stride 2 stays a real convolution and does get a halo. ResNet-50
has three such downsample convs, which is why it emitted Halo 19 against the
p100a capture's 22 — and ``bh_p100a_lut_v6`` carries halo entries keyed exactly
``kernel=1x1 stride=2x2 padding=0x0``, i.e. silicon really does dispatch them.
"""

import pytest

import ttsim.front.ttnn as ttnn
from ttsim.front.ttnn.buffer import TensorMemoryLayout
from ttsim.front.ttnn.device import close_device, open_device, set_default_device
from ttsim.front.ttnn.tensor import DataType, Tensor


@pytest.fixture
def device():
    dev = open_device()
    dev.set_arch('blackhole')
    set_default_device(dev)
    dev.compute_with_storage_grid_size(12, 10)
    yield dev
    close_device(dev)


def _conv(device, *, kernel, stride, padding=(0, 0), shard_layout=None):
    x = Tensor(shape=[1, 64, 32, 32], dtype=DataType.BFLOAT8_B, device=device)
    w = Tensor(shape=[128, 64, kernel[0], kernel[1]], dtype=DataType.BFLOAT8_B, device=device)
    b = Tensor(shape=[128], dtype=DataType.BFLOAT8_B, device=device)
    cfg = ttnn.Conv2dConfig(weights_dtype=DataType.BFLOAT8_B, shard_layout=shard_layout)
    before = len(device.ops)
    ttnn.conv2d(
        input_tensor=x, weight_tensor=w, bias_tensor=b,
        in_channels=64, out_channels=128, batch_size=1,
        input_height=32, input_width=32,
        kernel_size=kernel, stride=stride, padding=padding,
        dilation=(1, 1), groups=1, device=device, conv_config=cfg,
    )
    return [o.optype for o in list(device.ops.values())[before:]]


@pytest.mark.unit
def test_1x1_stride_2_gets_a_halo(device) -> None:
    """The ResNet-50 downsample case, and the regression this file exists for."""
    assert 'Halo' in _conv(device, kernel=(1, 1), stride=(2, 2))


@pytest.mark.unit
def test_1x1_stride_1_does_not(device) -> None:
    """Unchanged: this one really is lowered to a matmul.

    vgg_unet and vit depend on this staying true — both were verified to emit
    identical graphs (102/Halo 32 and 192/Halo 0) under the old and new gates.
    """
    assert 'Halo' not in _conv(device, kernel=(1, 1), stride=(1, 1))


@pytest.mark.unit
def test_3x3_always_does(device) -> None:
    assert 'Halo' in _conv(device, kernel=(3, 3), stride=(1, 1), padding=(1, 1))


@pytest.mark.unit
def test_1x1_with_padding_gets_a_halo(device) -> None:
    """padding != 0 also disqualifies matmul lowering in tt-metal's predicate."""
    assert 'Halo' in _conv(device, kernel=(1, 1), stride=(1, 1), padding=(1, 1))


@pytest.mark.unit
def test_1x1_width_sharded_gets_a_halo(device) -> None:
    """The last clause of use_matmul_for_1x1_conv: `not is_width_sharded`."""
    assert 'Halo' in _conv(
        device, kernel=(1, 1), stride=(1, 1),
        shard_layout=TensorMemoryLayout.WIDTH_SHARDED,
    )


@pytest.mark.unit
@pytest.mark.parametrize('stride', [1, (1, 1)])
def test_scalar_and_tuple_stride_agree(device, stride) -> None:
    """Call sites pass either spelling; the gate must not depend on which."""
    assert 'Halo' not in _conv(device, kernel=(1, 1), stride=stride)
