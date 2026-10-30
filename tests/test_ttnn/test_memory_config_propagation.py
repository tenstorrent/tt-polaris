# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Conv-output memory-config propagation — opt-in, device-scoped.

On hardware a conv output carries the memory config implied by its
``Conv2dConfig.shard_layout``, so model code can meaningfully ask
``out.memory_config()``.  The shim left conv/matmul outputs with
``_memory_config = None``, which makes ResNet-50's bottleneck reshard
unevaluable::

    if ds_out.memory_config() != out.memory_config():
        ds_out = ttnn.to_memory_config(ds_out, out.memory_config())

(tt-metal ``models/demos/vision/classification/resnet50/ttnn_resnet/tt/
ttnn_functional_resnet50.py:325-326``.)

With both sides ``None`` the condition is always False, so the Reshard that the
p100a refrun shows at every downsampling bottleneck was never emitted (polaris 3,
hardware 9).  Enabling propagation closes that to 9/9.

It is **off by default** and **scoped to a Device**, because enabling it can add
data-movement ops and polaris builds every workload graph in one process — a
process-wide flag would silently change whichever workload happened to be built
next.
"""

import pytest

import ttsim.front.ttnn as ttnn
from ttsim.front.ttnn.buffer import BufferType, TensorMemoryLayout
from ttsim.front.ttnn.device import Device, close_device, open_device
from ttsim.utils.shim_gates import (
    memory_config_propagation_enabled,
    set_memory_config_propagation,
)
from ttsim.front.ttnn.tensor import DataType, Tensor


@pytest.fixture(autouse=True)
def _restore_process_default():
    previous = memory_config_propagation_enabled()
    yield
    set_memory_config_propagation(previous)


def _conv(device, shard_layout=None):
    if shard_layout is None:
        shard_layout = TensorMemoryLayout.HEIGHT_SHARDED
    x = Tensor(shape=[1, 64, 16, 16], dtype=DataType.BFLOAT8_B, device=device)
    w = Tensor(shape=[128, 64, 3, 3], dtype=DataType.BFLOAT8_B, device=device)
    b = Tensor(shape=[128], dtype=DataType.BFLOAT8_B, device=device)
    return ttnn.conv2d(
        input_tensor=x, weight_tensor=w, bias_tensor=b,
        in_channels=64, out_channels=128, batch_size=1,
        input_height=16, input_width=16,
        kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
        dilation=(1, 1), groups=1, device=device,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=DataType.BFLOAT8_B, shard_layout=shard_layout,
        ),
    )


# --------------------------------------------------------------------------
# default: off, and inert
# --------------------------------------------------------------------------

@pytest.mark.unit
def test_off_by_default() -> None:
    assert memory_config_propagation_enabled() is False
    # A fresh Device carries no override, so it resolves to the process-wide
    # default rather than pinning a snapshot of it — see the two-layer note in
    # ttsim/front/ttnn/device.py::Device.__init__.
    assert not hasattr(Device(device_id=0), 'propagate_memory_config')
    assert memory_config_propagation_enabled(Device(device_id=0)) is False


@pytest.mark.unit
def test_conv_output_has_no_memory_config_when_off() -> None:
    """The pre-existing behaviour every other workload was verified against."""
    device = open_device()
    out = _conv(device)
    assert out.memory_config() is None
    close_device(device)


# --------------------------------------------------------------------------
# enabled: the conv output carries its shard layout
# --------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize('shard_layout', [
    TensorMemoryLayout.HEIGHT_SHARDED,
    TensorMemoryLayout.BLOCK_SHARDED,
])
def test_conv_output_carries_shard_layout_when_enabled(shard_layout) -> None:
    device = open_device()
    set_memory_config_propagation(True, device)
    out = _conv(device, shard_layout)
    mc = out.memory_config()
    assert mc is not None
    assert mc.memory_layout == shard_layout
    assert mc.buffer_type == BufferType.L1
    close_device(device)


@pytest.mark.unit
def test_differing_shard_layouts_compare_unequal() -> None:
    """The comparison canonical's bottleneck reshard depends on.

    HEIGHT vs BLOCK must compare unequal, or `to_memory_config` is never called
    and the Reshard never emitted.
    """
    device = open_device()
    set_memory_config_propagation(True, device)
    height = _conv(device, TensorMemoryLayout.HEIGHT_SHARDED)
    block = _conv(device, TensorMemoryLayout.BLOCK_SHARDED)
    assert height.memory_config() != block.memory_config()
    close_device(device)


@pytest.mark.unit
def test_no_shard_layout_is_not_invented() -> None:
    """An unset shard_layout leaves the output alone rather than guessing one."""
    device = open_device()
    set_memory_config_propagation(True, device)
    x = Tensor(shape=[1, 64, 16, 16], dtype=DataType.BFLOAT8_B, device=device)
    w = Tensor(shape=[128, 64, 3, 3], dtype=DataType.BFLOAT8_B, device=device)
    b = Tensor(shape=[128], dtype=DataType.BFLOAT8_B, device=device)
    out = ttnn.conv2d(
        input_tensor=x, weight_tensor=w, bias_tensor=b,
        in_channels=64, out_channels=128, batch_size=1,
        input_height=16, input_width=16,
        kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
        dilation=(1, 1), groups=1, device=device,
    )
    assert out.memory_config() is None
    close_device(device)


# --------------------------------------------------------------------------
# scoping: one workload's opt-in must not reach the next
# --------------------------------------------------------------------------

@pytest.mark.unit
def test_enabling_on_one_device_does_not_affect_another() -> None:
    """polaris builds every workload graph in one process.

    A process-wide flag would silently enable propagation for whichever workload
    was built after the one that opted in — the same leak class as the
    default-device arch pointer.
    """
    opted_in = open_device()
    set_memory_config_propagation(True, opted_in)

    other = open_device()
    assert memory_config_propagation_enabled(other) is False
    assert _conv(other).memory_config() is None

    assert memory_config_propagation_enabled(opted_in) is True
    close_device(opted_in)
    close_device(other)


@pytest.mark.unit
def test_device_opt_in_leaves_the_process_default_alone() -> None:
    device = open_device()
    set_memory_config_propagation(True, device)
    assert memory_config_propagation_enabled() is False
    close_device(device)
    assert memory_config_propagation_enabled() is False


@pytest.mark.unit
def test_set_returns_previous_value() -> None:
    device = open_device()
    assert set_memory_config_propagation(True, device) is False
    assert set_memory_config_propagation(False, device) is True
    close_device(device)
