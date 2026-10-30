# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""A lowered 1x1 matmul tilizes a ROW_MAJOR activation first — when opted in.

Hardware matmuls consume TILE. When a 1x1 conv is lowered to a matmul and its
activation is still ROW_MAJOR, silicon dispatches a Tilize: the ResNet-50 p100a
capture records exactly that at rows 12 and 19, converting the maxpool's
ROW_MAJOR output ahead of the first two matmuls. ``_matmul_1x1_with_hw_fields``
had documented the behaviour ("the activation is tilized before the matmul")
without modelling it, so polaris emitted 0 Tilize against hardware's 2.

OFF by default and device-scoped, like ``set_memory_config_propagation``: it adds
ops, and polaris builds every workload graph in one process, so a process-wide
flag would silently change whichever workload was built next. vgg_unet and vit
were both verified to leave the gate off and keep 102 / 192 ops unchanged.
"""

import pytest

import ttsim.front.ttnn as ttnn
from ttsim.front.ttnn.buffer import TensorMemoryLayout
from ttsim.front.ttnn.device import Device, close_device, open_device, set_default_device
from ttsim.utils.shim_gates import set_tilize_before_matmul, tilize_before_matmul_enabled
from ttsim.front.ttnn.tensor import DataType, Layout, Tensor


@pytest.fixture(autouse=True)
def _restore_process_default():
    previous = tilize_before_matmul_enabled()
    yield
    set_tilize_before_matmul(previous)


def _emit_1x1(device, layout):
    x = Tensor(shape=[1, 64, 32, 32], dtype=DataType.BFLOAT8_B, device=device, layout=layout)
    w = Tensor(shape=[128, 64, 1, 1], dtype=DataType.BFLOAT8_B, device=device)
    b = Tensor(shape=[128], dtype=DataType.BFLOAT8_B, device=device)
    before = len(device.ops)
    ttnn.conv2d(
        input_tensor=x, weight_tensor=w, bias_tensor=b,
        in_channels=64, out_channels=128, batch_size=1,
        input_height=32, input_width=32,
        kernel_size=(1, 1), stride=(1, 1), padding=(0, 0),
        dilation=(1, 1), groups=1, device=device,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=DataType.BFLOAT8_B,
            shard_layout=TensorMemoryLayout.HEIGHT_SHARDED,
        ),
    )
    return [o.optype for o in list(device.ops.values())[before:]]


@pytest.mark.unit
def test_off_by_default() -> None:
    assert tilize_before_matmul_enabled() is False
    # No override on a fresh Device: the gate must fall through to the
    # process-wide default, not a snapshot taken at construction.
    assert not hasattr(Device(device_id=0), 'tilize_before_matmul')
    assert tilize_before_matmul_enabled(Device(device_id=0)) is False


@pytest.mark.unit
def test_row_major_activation_is_not_tilized_when_off() -> None:
    """The pre-existing behaviour every other workload was verified against."""
    dev = open_device(); set_default_device(dev)
    assert 'Tilize' not in _emit_1x1(dev, Layout.ROW_MAJOR_LAYOUT)
    close_device(dev)


@pytest.mark.unit
def test_row_major_activation_is_tilized_when_on() -> None:
    dev = open_device(); set_default_device(dev)
    set_tilize_before_matmul(True, dev)
    optypes = _emit_1x1(dev, Layout.ROW_MAJOR_LAYOUT)
    assert optypes.count('Tilize') == 1
    assert optypes.index('Tilize') < optypes.index('MatMul'), 'Tilize must precede the matmul'
    close_device(dev)


@pytest.mark.unit
def test_tile_activation_is_left_alone() -> None:
    """Already TILE: hardware dispatches nothing, so neither do we."""
    dev = open_device(); set_default_device(dev)
    set_tilize_before_matmul(True, dev)
    assert 'Tilize' not in _emit_1x1(dev, Layout.TILE_LAYOUT)
    close_device(dev)


@pytest.mark.unit
def test_enabling_on_one_device_does_not_affect_another() -> None:
    """polaris builds every workload graph in one process."""
    opted_in = open_device(); set_default_device(opted_in)
    set_tilize_before_matmul(True, opted_in)
    other = open_device(); set_default_device(other)
    assert tilize_before_matmul_enabled(other) is False
    assert 'Tilize' not in _emit_1x1(other, Layout.ROW_MAJOR_LAYOUT)
    assert tilize_before_matmul_enabled(opted_in) is True
    close_device(opted_in); close_device(other)


@pytest.mark.unit
def test_set_returns_previous_value() -> None:
    dev = open_device()
    assert set_tilize_before_matmul(True, dev) is False
    assert set_tilize_before_matmul(False, dev) is True
    close_device(dev)
