# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""``ttnn.get_arch_name()`` must report the device's architecture.

It previously returned the ``WORMHOLE_B0`` constant unconditionally, which made
``is_blackhole()`` permanently False and left every Blackhole branch in every
workload unreachable — including the whole BH half of ResNet-50's four-combo
verification target.

The polaris front-end graph is cached per (group, name, instance, batch) and NOT
per device (``polaris.py`` ``get_wlgraph``), so an arch-dependent graph has to
come from a separate workload entry — the wh/ + bh/ split that vgg_unet and
resnet50 both use. These tests pin the device-level contract that makes that
split mean something.
"""

import pytest

import ttsim.front.ttnn as ttnn
from ttsim.front.ttnn.device import (
    ARCH,
    Device,
    close_device,
    coerce_arch,
    num_cores_to_corerangeset,
    open_device,
    set_default_device,
)


@pytest.fixture(autouse=True)
def _clear_default_device():
    """Keep the module-level default device from leaking between tests."""
    set_default_device(None)
    yield
    set_default_device(None)


# --------------------------------------------------------------------------
# default behaviour is preserved
# --------------------------------------------------------------------------

@pytest.mark.unit
def test_defaults_to_wormhole_with_no_device() -> None:
    """Model code may call this at import time, before any device exists."""
    assert ttnn.get_arch_name() == 'wormhole_b0'


@pytest.mark.unit
def test_device_defaults_to_wormhole() -> None:
    """An unspecified arch keeps what every caller got before arch was settable."""
    device = Device(device_id=0)
    assert device.arch() is ARCH.WORMHOLE_B0
    assert device.get_arch_name() == 'wormhole_b0'
    set_default_device(device)
    assert ttnn.get_arch_name() == 'wormhole_b0'


# --------------------------------------------------------------------------
# arch is now settable, and reported
# --------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize('arch_in, expected', [
    ('blackhole', 'blackhole'),
    ('BLACKHOLE', 'blackhole'),
    (ARCH.BLACKHOLE, 'blackhole'),
    ('wormhole_b0', 'wormhole_b0'),
    (ARCH.GRAYSKULL, 'grayskull'),
])
def test_arch_is_reported_from_the_device(arch_in, expected) -> None:
    device = open_device(arch=arch_in)
    set_default_device(device)
    assert ttnn.get_arch_name() == expected
    assert device.get_arch_name() == expected


@pytest.mark.unit
def test_set_arch_after_construction() -> None:
    device = open_device()
    set_default_device(device)
    assert ttnn.get_arch_name() == 'wormhole_b0'
    device.set_arch('blackhole')
    assert ttnn.get_arch_name() == 'blackhole'


@pytest.mark.unit
def test_coerce_arch_rejects_nonsense() -> None:
    assert coerce_arch(None) is ARCH.WORMHOLE_B0
    with pytest.raises(TypeError):
        coerce_arch(3)
    with pytest.raises(KeyError):
        coerce_arch('not_an_arch')


@pytest.mark.unit
def test_close_device_clears_the_default() -> None:
    """Otherwise one workload's arch leaks into the next built in the same process.

    polaris.py builds every workload graph in a single process, so a lingering
    default device would silently give the next workload the wrong arch.
    """
    device = open_device(arch='blackhole')
    set_default_device(device)
    assert ttnn.get_arch_name() == 'blackhole'
    close_device(device)
    assert ttnn.get_arch_name() == 'wormhole_b0'


@pytest.mark.unit
def test_close_device_leaves_a_different_default_alone() -> None:
    keeper = open_device(arch='blackhole')
    other = open_device(arch='wormhole_b0')
    set_default_device(keeper)
    close_device(other)
    assert ttnn.get_arch_name() == 'blackhole'


# --------------------------------------------------------------------------
# num_cores_to_corerangeset — was a stub returning (1, 1)
# --------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize('target, grid, row_wise', [
    (120, (12, 10), True),    # BH p100a, full grid
    (72, (8, 9), True),       # WH n150, full grid
    (100, (12, 10), True),    # partial: full rows + remainder
    (37, (12, 10), True),
    (1, (12, 10), True),
    (60, (12, 10), False),    # column-wise
    (37, (12, 10), False),
])
def test_corerangeset_covers_exactly_the_requested_cores(target, grid, row_wise) -> None:
    crs = num_cores_to_corerangeset(target, grid, row_wise)
    assert crs.num_cores() == target
    # tt-metal/tt_metal/common/work_split.cpp generates at most 3 ranges
    assert len(crs.ranges) <= 3


@pytest.mark.unit
def test_corerangeset_accepts_a_coordinate_like_grid() -> None:
    """compute_with_storage_grid_size() returns a tuple in the shim, an object in real ttnn."""
    class _Grid:
        x, y = 12, 10
    assert num_cores_to_corerangeset(120, _Grid(), True).num_cores() == 120


@pytest.mark.unit
def test_corerangeset_rejects_an_oversized_request() -> None:
    with pytest.raises(ValueError, match='greater than total available'):
        num_cores_to_corerangeset(121, (12, 10), True)
