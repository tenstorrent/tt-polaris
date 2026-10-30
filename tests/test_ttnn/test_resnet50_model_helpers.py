# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The device-free helpers in ``ttnn_functional_resnet50.py``.

Each one exists because the shim and real ttnn disagree somewhere, and each is
numbered as a "divergence note" in the model file.  They are small, they are
called from deep inside the bottleneck, and every one of them is silent when
wrong: none changes a tensor's logical shape, they change the *shard extents and
grids* that LUT keys are built from.  A mistake here shows up as a hit-rate drop
in a projection run, not as a failure.

``resnet50.run()`` itself is exercised by ``test_resnet50_graph_build.py``;
this file is the part that needs no model.
"""

import math
from typing import Any

import pytest

import ttsim.front.ttnn as ttnn
from ttsim.front.ttnn.device import Device, close_device, set_default_device
from workloads.ttnn.resnet50.common.ttnn_functional_resnet50 import (
    IS_POLARIS,
    ResnetLinear,
    _conv2d_config,
    _grid_xy,
    _nearest_y,
    _relu_activation,
    _shard_dims,
    is_blackhole,
    is_blackhole_p100,
    is_wormhole_b0,
    nearest_32,
)

pytestmark = pytest.mark.skipif(
    not IS_POLARIS, reason='these pin the Polaris half of the dual-mode helpers')

#: From ``config/tt_bh.yaml:95`` and ``config/tt_wh.yaml:88``.
BH_P100A_GRID = (12, 10)
WH_N150_GRID = (8, 9)


@pytest.fixture
def device_factory():
    """Devices with an explicit arch and grid, torn down so no default leaks.

    ``ttnn.get_arch_name()`` reads the *default* device, and polaris builds every
    workload graph in one process — ``close_device`` clears the pointer for
    exactly that reason (``device.py:194-201``).
    """
    made: list[Device] = []

    def make(arch, grid):
        d = Device(device_id=len(made))
        d.set_arch(arch)
        d.compute_with_storage_grid_size(*grid)
        set_default_device(d)
        made.append(d)
        return d

    yield make
    for d in made:
        close_device(d)
    set_default_device(None)


class _FakeCoreCoord:
    """Real ttnn returns a CoreCoord-like object from
    ``compute_with_storage_grid_size``; the shim returns a plain tuple
    (``device.py:163-168``).  Divergence note 8."""

    def __init__(self, x, y):
        self.x, self.y = x, y


class _FakeDevice:
    def __init__(self, grid):
        self._grid = grid

    def compute_with_storage_grid_size(self):
        return self._grid


class _Tensor:
    """The two attributes ``_shard_dims`` reads."""

    def __init__(self, shape, hw_shape=None):
        self.shape = list(shape)
        if hw_shape is not None:
            self.hw_shape = list(hw_shape)


# ---------------------------------------------------------------------------
# rounding
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize(
    'x, expected',
    [(1, 32), (31, 32), (32, 32), (33, 64), (3136, 3136), (14, 32), (313, 320)],
)
def test_nearest_32_rounds_up_and_is_idempotent(x, expected) -> None:
    """``ceil(x/32)*32``.  The 32 and 3136 rows are the ones that matter: an
    already-aligned value must not be bumped to the next tile, which is what a
    ``(x // 32 + 1) * 32`` spelling would do."""
    assert nearest_32(x) == expected
    assert nearest_32(nearest_32(x)) == expected


@pytest.mark.unit
@pytest.mark.parametrize('y', [8, 16, 32])
def test_nearest_y_generalises_nearest_32(y) -> None:
    """``nearest_32`` is ``_nearest_y(x, 32)``; if they ever disagree one of them
    has been edited alone."""
    for x in (1, 7, 8, 31, 32, 33, 64, 1000):
        assert _nearest_y(x, y) == math.ceil(x / y) * y
        if y == 32:
            assert _nearest_y(x, y) == nearest_32(x)


# ---------------------------------------------------------------------------
# _grid_xy — divergence note 8
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_grid_xy_accepts_both_the_shim_tuple_and_a_ttnn_core_coord() -> None:
    """Both spellings must yield the same pair.  Only the tuple branch runs under
    Polaris, so the CoreCoord branch would rot unnoticed — and it is the branch
    the hardware path depends on."""
    assert _grid_xy(_FakeDevice((12, 10))) == (12, 10)
    assert _grid_xy(_FakeDevice([12, 10])) == (12, 10)
    assert _grid_xy(_FakeDevice(_FakeCoreCoord(12, 10))) == (12, 10)


@pytest.mark.unit
def test_grid_xy_reads_a_real_device(device_factory) -> None:
    dev = device_factory('blackhole', BH_P100A_GRID)
    assert _grid_xy(dev) == BH_P100A_GRID


# ---------------------------------------------------------------------------
# arch predicates
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize(
    'arch, grid, wh, bh',
    [('wormhole_b0', WH_N150_GRID, True, False),
     ('blackhole', BH_P100A_GRID, False, True)],
)
def test_arch_predicates_follow_the_default_device(
        device_factory, arch, grid, wh, bh) -> None:
    """``is_wormhole_b0`` / ``is_blackhole`` read ``ttnn.get_arch_name()``, which
    reads the default device.  Both are asserted on every row: a predicate that
    returned True for everything would pass a one-sided check."""
    device_factory(arch, grid)
    assert ttnn.get_arch_name() == arch
    assert is_wormhole_b0() is wh
    assert is_blackhole() is bh


@pytest.mark.unit
@pytest.mark.parametrize(
    'arch, grid, expected',
    [
        ('blackhole', (12, 10), True),    # p100a: 120 cores
        ('blackhole', (13, 10), False),   # the 130-core part this test distinguishes
        ('wormhole_b0', (12, 10), False),  # right core count, wrong arch
    ],
)
def test_blackhole_p100_needs_both_the_arch_and_the_core_count(
        device_factory, arch, grid, expected) -> None:
    """``is_blackhole() and gx*gy != 130``.

    Mirrors ``tt-metal/models/demos/vision/classification/resnet50/
    ttnn_resnet/tt/ttnn_functional_resnet50_model_utils.py:8-12``.

    Three rows because the predicate is a conjunction: 120-core blackhole is
    True, 130-core blackhole is False, and a 120-core *wormhole* must also be
    False.  Dropping either conjunct still passes two of the three.
    """
    dev = device_factory(arch, grid)
    assert is_blackhole_p100(dev) is expected


# ---------------------------------------------------------------------------
# _shard_dims — the [32, 1] vs [320, 128] bug
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_shard_dims_prefers_the_flattened_view_over_the_logical_shape() -> None:
    """Canonical computes shard extents off ``x.shape[2]`` / ``x.shape[3]``
    because on hardware ``x`` is already NHWC-flat.  Polaris carries logical
    NCHW, so the same expression read H and W: the pre-layer4 block config came
    out ``[nearest_32(14 // 10), 14 // 8]`` = ``[32, 1]`` where it should be
    ``[nearest_32(3136 // 10), 1024 // 8]`` = ``[320, 128]``.

    Both the reading AND the downstream extents are asserted, because (14, 14)
    and (3136, 1024) are both plausible-looking pairs -- the bug was invisible
    precisely because nothing crashed on the wrong one.
    """
    logical = [16, 1024, 14, 14]
    flat = [1, 1, 3136, 1024]
    with_view = _Tensor(logical, flat)
    without = _Tensor(logical)

    assert _shard_dims(with_view) == (3136, 1024)
    assert _shard_dims(without) == (14, 14)

    h, w = _shard_dims(with_view)
    assert [nearest_32(h // 10), w // 8] == [320, 128]
    h, w = _shard_dims(without)
    assert [nearest_32(h // 10), w // 8] == [32, 1]


@pytest.mark.unit
@pytest.mark.parametrize('hw_shape', [None, [1, 1], [1, 1, 3136]])
def test_shard_dims_falls_back_when_the_view_is_absent_or_too_short(hw_shape) -> None:
    """``len(hw) >= 4`` — a shorter view cannot be indexed at [2] and [3], and a
    real ttnn tensor has no ``hw_shape`` at all."""
    t = _Tensor([16, 1024, 14, 14], hw_shape)
    assert _shard_dims(t) == (14, 14)


# ---------------------------------------------------------------------------
# _conv2d_config — divergence note 2
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_conv2d_config_drops_the_field_the_shim_cannot_model() -> None:
    """``Conv2dConfig`` (``config.py:13-31``) is a dataclass with no
    ``enable_activation_reuse`` field, so forwarding it raises TypeError.  The
    wrapper pops it on the Polaris path and forwards it verbatim on HW.

    The paired raise is the point: without it this test would pass against a
    wrapper that did nothing, on a day when the shim happened to accept the
    field.
    """
    kwargs: dict[str, Any] = dict(
        weights_dtype=ttnn.bfloat8_b,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        enable_activation_reuse=True,
    )
    cfg = _conv2d_config(**kwargs)
    assert cfg.shard_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    assert not hasattr(cfg, 'enable_activation_reuse')
    with pytest.raises(TypeError):
        ttnn.Conv2dConfig(**kwargs)


@pytest.mark.unit
def test_conv2d_config_passes_everything_else_through() -> None:
    """Only that one field is dropped; a wrapper that filtered more would quietly
    change the conv's behaviour."""
    cfg = _conv2d_config(
        weights_dtype=ttnn.bfloat8_b,
        shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        activation='relu',
        deallocate_activation=True,
        reallocate_halo_output=False,
        act_block_h_override=64,
    )
    assert cfg.activation == 'relu'
    assert cfg.deallocate_activation is True
    assert cfg.reallocate_halo_output is False
    assert cfg.act_block_h_override == 64


# ---------------------------------------------------------------------------
# _relu_activation — divergence note 1
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_relu_activation_is_the_string_the_shim_understands() -> None:
    """Conv2dConfig.activation is a plain str in the shim (``config.py:16``);
    real ttnn wants a ``UnaryWithParam``."""
    act = _relu_activation()
    assert act == 'relu'
    assert _conv2d_config(
        weights_dtype=ttnn.bfloat8_b,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        activation=act,
    ).activation == 'relu'


# ---------------------------------------------------------------------------
# _reinterpret_shape — relabel without emitting an op
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_reinterpret_shape_relabels_in_place_and_emits_nothing() -> None:
    """Hardware regroups these for free — the p100a capture has no profiler row
    for any of them — so emitting a Reshape would create a polaris-only op at a
    position where nothing runs."""
    from workloads.ttnn.resnet50.common.ttnn_functional_resnet50 import _reinterpret_shape

    t = _Tensor([16, 224, 224, 3])
    got = _reinterpret_shape(t, [16, 3, 224, 224])
    assert got is t
    assert t.shape == [16, 3, 224, 224]


@pytest.mark.unit
def test_reinterpret_shape_refuses_to_change_the_element_count() -> None:
    """The guard that keeps this from being a silent data-losing relabel.  Only
    safe because it is element-count preserving and each tensor is consumed
    once."""
    from workloads.ttnn.resnet50.common.ttnn_functional_resnet50 import _reinterpret_shape

    t = _Tensor([16, 3, 224, 224])
    with pytest.raises(AssertionError, match='element-count preserving'):
        _reinterpret_shape(t, [16, 3, 224, 223])
    assert t.shape == [16, 3, 224, 224], 'must not have been mutated before the check'


# ---------------------------------------------------------------------------
# ResnetLinear — the classifier's HW-view overlay
# ---------------------------------------------------------------------------

class _Param:
    """A model parameter as ResnetLinear sees it: a shape, plus the two HW-view
    overlay attributes it sets.  Those are declared, not assigned, so ``hasattr``
    stays False until ResnetLinear actually sets them — which is what the tests
    below distinguish."""

    _hw_layout: Any
    _hw_dtype: Any

    def __init__(self, shape):
        self.shape = list(shape)


@pytest.mark.unit
def test_resnet_linear_tags_the_classifier_weights_as_tile_bfloat16() -> None:
    """model_preprocessing builds every parameter at the model's WEIGHTS_DTYPE
    (bfloat8_b), but the classifier is not requantised on silicon: the capture's
    fc matmul (row 113) records input_1 and input_2 as TILE / BFLOAT16.  These
    are key-only overlays — they must not touch ``dtype`` itself, or activation
    dtype propagation changes with them.
    """
    w, b = _Param([1, 1, 1000, 2048]), _Param([1, 1, 1, 1000])
    fn = ResnetLinear(
        weight=w, bias=b,
        output_mem_config=ttnn.L1_MEMORY_CONFIG,
        model_config={'ACTIVATIONS_DTYPE': ttnn.bfloat8_b},
        compute_kernel_config=None,
    )
    assert callable(fn)
    for p in (w, b):
        assert p._hw_layout == ttnn.TILE_LAYOUT
        assert p._hw_dtype == ttnn.bfloat16
        assert not hasattr(p, 'dtype'), 'the overlay must not set a real dtype'


@pytest.mark.unit
def test_resnet_linear_tolerates_a_missing_bias() -> None:
    """``for _p in (weight, bias): if _p is not None`` — the None guard."""
    w = _Param([1, 1, 1000, 2048])
    fn = ResnetLinear(
        weight=w, bias=None,
        output_mem_config=ttnn.L1_MEMORY_CONFIG,
        model_config={'ACTIVATIONS_DTYPE': ttnn.bfloat8_b},
        compute_kernel_config=None,
    )
    assert callable(fn)
    assert w._hw_dtype == ttnn.bfloat16
