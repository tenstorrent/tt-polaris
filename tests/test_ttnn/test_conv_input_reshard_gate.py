# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Contract tests for the conv input Reshard (+Move) gate.

tt-metal reference: ``tt-metal/ttnn/cpp/ttnn/operations/conv/conv2d/conv2d_utils.cpp:750-751`` (reshard_if_not_optimal
forces
the recompute) and ``:883-887`` (the to_memory_config, and the ttnn::move that
rides along when the activation is deallocated out of L1).
"""
import pytest

import ttsim.front.ttnn as ttnn
from ttsim.front.ttnn.buffer import BufferType, TensorMemoryLayout
from ttsim.front.ttnn.device import set_default_device
from ttsim.front.ttnn.memory import MemoryConfig
from ttsim.utils.shim_gates import conv_input_reshard_enabled, set_conv_input_reshard
from ttsim.front.ttnn.tensor import ttnn_random


def _dev():
    d = ttnn.open_device(l1_small_size=24576, device_id=0)
    d.set_arch('blackhole')
    # The emitter needs a compute grid to run determine_parallel_config;
    # open_device leaves grid_x/grid_y at 0 until a caller declares one.
    d.compute_with_storage_grid_size(12, 10)
    set_default_device(d)
    return d


def _seq(dev):
    return [(o.optype, o) for o in dev.ops.values()]


def _conv(dev, *, reshard, dealloc=True, in_c=64, out_c=256, h=56, w=56):
    x = ttnn_random((1, in_c, h, w), -1, 1, dtype=ttnn.bfloat16)
    x._memory_config = MemoryConfig(TensorMemoryLayout.HEIGHT_SHARDED, BufferType.L1)
    wt = ttnn_random((out_c, in_c, 1, 1), -1, 1, dtype=ttnn.bfloat16)
    bs = ttnn_random((1, 1, 1, out_c), -1, 1, dtype=ttnn.bfloat16)
    return ttnn.conv2d(
        input_tensor=x, weight_tensor=wt, bias_tensor=bs,
        in_channels=in_c, out_channels=out_c,
        batch_size=1, input_height=h, input_width=w, kernel_size=(1, 1),
        stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, device=dev,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.bfloat8_b,
            shard_layout=TensorMemoryLayout.HEIGHT_SHARDED,
            reshard_if_not_optimal=reshard,
            deallocate_activation=dealloc,
        ),
    )


@pytest.mark.unit
def test_gate_is_off_by_default():
    dev = _dev()
    assert conv_input_reshard_enabled(dev) is False


@pytest.mark.unit
def test_no_reshard_emitted_when_gate_off():
    dev = _dev()
    set_conv_input_reshard(False, dev)
    _conv(dev, reshard=True)
    assert not any(t == 'Reshard' for t, _ in _seq(dev))


@pytest.mark.unit
def test_no_reshard_when_flag_unset_even_with_gate_on():
    """reshard_if_not_optimal=False must stay inherit-only (cpp:750)."""
    dev = _dev()
    set_conv_input_reshard(True, dev)
    _conv(dev, reshard=False)
    assert not any(t == 'Reshard' for t, _ in _seq(dev))


@pytest.mark.unit
def test_reshard_precedes_the_conv_when_flag_set():
    dev = _dev()
    set_conv_input_reshard(True, dev)
    _conv(dev, reshard=True)
    types = [t for t, _ in _seq(dev)]
    assert 'Reshard' in types, types
    # Reshard must come before whatever the conv lowered to.
    assert types.index('Reshard') == 0, types


@pytest.mark.unit
def test_move_rides_along_when_activation_deallocated():
    """cpp:884-887 — Move only when deallocate_activation and not DRAM."""
    dev = _dev()
    set_conv_input_reshard(True, dev)
    _conv(dev, reshard=True, dealloc=True)
    types = [t for t, _ in _seq(dev)]
    assert 'Reshard' in types and 'Move' in types, types
    assert types.index('Move') == types.index('Reshard') + 1, types


@pytest.mark.unit
def test_no_move_when_activation_not_deallocated():
    dev = _dev()
    set_conv_input_reshard(True, dev)
    _conv(dev, reshard=True, dealloc=False)
    types = [t for t, _ in _seq(dev)]
    assert 'Reshard' in types, types
    assert types.index('Reshard') == 0
    # Nothing between the Reshard and the conv.
    assert types[1] != 'Move', types


@pytest.mark.unit
def test_gate_is_device_scoped():
    a = _dev()
    b = _dev()
    set_conv_input_reshard(True, a)
    assert conv_input_reshard_enabled(a) is True
    assert conv_input_reshard_enabled(b) is False
