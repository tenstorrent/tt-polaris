#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""``hw_shape`` propagation across an elementwise binary — opt-in, device-scoped.

``hw_shape`` is the NHWC-flattened ``[1, 1, N*H*W, C]`` view that LUT keys and
profiler comparison are built from.  Every data-movement sinf carries it across
— Move, Reshard, InterleavedToSharded, ShardedToInterleaved (``ttsim/ops/desc/ttsim_layout.py``
:434, :458, :482, :673).  ``bidir_bcast`` did not, so ResNet-50's residual
``Add`` dropped the flattened form and every op downstream keyed on logical NCHW
``(16, 1024, 14, 14)`` where the capture records ``(1, 1, 3136, 1024)``: one
missing line, roughly 27 mis-keyed rows.

Nothing here changes a tensor's ``.shape``, so a regression is silent — it moves
LUT keys and shows up as a hit-rate drop, not a failure.  Hence these tests.

The gate is OFF by default and scoped to a ``Device``
(``TTSIM_PROPAGATE_BCAST_HW_SHAPE``, ``ttsim/utils/shim_gates.py:169``) because it changes the
key of every elementwise binary in every workload, and polaris builds all
workload graphs in one process — a process-wide flag would silently alter
whichever workload happened to be built next.
"""

import numpy as np
import pytest

from ttsim.ops.desc.helpers import _propagate_bcast_hw_shape, bidir_bcast
from ttsim.ops.tensor import SimTensor

#: ResNet-50 layer 3 at batch 16: logical NCHW and the flattened view the
#: p100a capture actually records for the same tensor.
NCHW = [16, 1024, 14, 14]
NHWC_FLAT = [1, 1, 3136, 1024]


class _Device:
    """Stands in for ``ttsim.front.ttnn.device.Device``, which is what carries
    this flag in production (``ttsim/utils/shim_gates.py:187-188``).  A bare object is used rather
    than the real Device so these stay unit tests of the descriptor."""

    def __init__(self, enabled: bool):
        self.propagate_bcast_hw_shape = enabled


class _Op:
    """The slice of ``SimOp`` that ``bidir_bcast`` touches: ``optype`` to pick a
    compute function and an instruction name, ``precision`` for the byte counts
    (None is SimOp's own default, ``ttsim/ops/op.py:45``), and ``perf_stats`` assigned on
    the way out."""

    def __init__(self, optype='Add'):
        self.optype = optype
        self.precision = None
        self.perf_stats = None


def _t(name, shape, hw_shape=None, device=None):
    t = SimTensor({'name': name, 'shape': list(shape), 'dtype': np.dtype(np.float32)})
    t.hw_shape = list(hw_shape) if hw_shape is not None else None
    if device is not None:
        # SimTensor has no ``device`` field; the shim's gate accessor reads it off
        # whatever object it is handed (``getattr(device, ..., module_default)``),
        # so the attribute is set dynamically here exactly as the shim expects it.
        t.device = device  # type: ignore[attr-defined]
    return t


def _out(name='Y', shape=None, hw_shape=None):
    t = SimTensor({'name': name, 'shape': list(shape or []), 'dtype': np.dtype(np.float32)})
    t.hw_shape = list(hw_shape) if hw_shape is not None else None
    return t


# ---------------------------------------------------------------------------
# the gate
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.opunit
def test_off_by_default_and_on_when_enabled() -> None:
    """THE CONTROL FOR THIS FILE.  Identical inputs, only the flag differs.

    Every other test here asserts that ``Y.hw_shape`` stays None for some
    reason; without this pair they would all pass against a function whose body
    was ``return``.
    """
    def run(enabled):
        dev = _Device(enabled)
        X0 = _t('X0', NCHW, NHWC_FLAT, dev)
        X1 = _t('X1', NCHW, NHWC_FLAT, dev)
        Y = _out(shape=NCHW)
        _propagate_bcast_hw_shape(X0, X1, Y)
        return Y.hw_shape

    assert run(False) is None
    assert run(True) == NHWC_FLAT


@pytest.mark.unit
@pytest.mark.opunit
def test_no_device_on_either_operand_is_not_an_error() -> None:
    """``SimTensor`` has no ``device`` attribute at all (ttsim/ops/tensor.py:136-175); only
    the ttnn shim's Tensor sets one.  Graphs built through the functional front
    end must not raise here, and must not propagate either."""
    X0 = _t('X0', NCHW, NHWC_FLAT)
    X1 = _t('X1', NCHW, NHWC_FLAT)
    Y = _out(shape=NCHW)
    _propagate_bcast_hw_shape(X0, X1, Y)
    assert Y.hw_shape is None


@pytest.mark.unit
@pytest.mark.opunit
def test_device_is_taken_from_the_second_operand_when_the_first_has_none() -> None:
    """``dev = X0.device or X1.device``.  A residual add often has one operand
    straight off a device op and the other a host-side constant, so reading only
    X0 would leave the gate off for half of them."""
    dev = _Device(True)
    X0 = _t('X0', NCHW, NHWC_FLAT)            # no device
    X1 = _t('X1', NCHW, NHWC_FLAT, dev)
    Y = _out(shape=NCHW)
    _propagate_bcast_hw_shape(X0, X1, Y)
    assert Y.hw_shape == NHWC_FLAT


# ---------------------------------------------------------------------------
# the four guards past the gate
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.opunit
def test_an_existing_view_on_the_output_is_not_overwritten() -> None:
    """Idempotency.  If ``Y`` already carries a view — a sinf re-run, or an
    output tensor a producer already annotated — the operand's view must not
    clobber it."""
    dev = _Device(True)
    already = [1, 1, 3136, 256]
    X0 = _t('X0', NCHW, NHWC_FLAT, dev)
    X1 = _t('X1', NCHW, NHWC_FLAT, dev)
    Y = _out(shape=NCHW, hw_shape=already)
    _propagate_bcast_hw_shape(X0, X1, Y)
    assert Y.hw_shape == already


@pytest.mark.unit
@pytest.mark.opunit
def test_nothing_to_propagate_when_the_first_operand_has_no_view() -> None:
    """Most binaries in most workloads have no flattened view on either side —
    attention projections, embeddings, anything non-NCHW.  Those must come out
    as None, not as an empty list or a copy of the logical shape."""
    dev = _Device(True)
    X0 = _t('X0', NCHW, None, dev)
    X1 = _t('X1', NCHW, NHWC_FLAT, dev)
    Y = _out(shape=NCHW)
    _propagate_bcast_hw_shape(X0, X1, Y)
    assert Y.hw_shape is None


@pytest.mark.unit
@pytest.mark.opunit
@pytest.mark.parametrize(
    'x0_shape, y_shape, propagates',
    [
        (NCHW, NCHW, True),                          # shape-preserving: the residual add
        ([1, 1024, 1, 1], NCHW, False),              # X0 broadcast up: a squeeze-excite scale
        ([16, 1024, 14, 14], [16, 1024, 28, 28], False),   # genuine reshape
    ],
)
def test_only_a_shape_preserving_binary_inherits_the_view(
        x0_shape, y_shape, propagates) -> None:
    """The guard that does the real work.  ``hw_shape`` encodes N*H*W folded into
    rows; if the op changed the logical shape then that fold no longer describes
    the output, and inheriting it would key the op against a tensor size the
    hardware never saw.

    The middle row is the case that makes this matter — a 1x1-spatial operand
    broadcasting up to NCHW is common, and it carries a perfectly valid
    ``hw_shape`` of its own that simply does not describe ``Y``.
    """
    dev = _Device(True)
    X0 = _t('X0', x0_shape, NHWC_FLAT, dev)
    X1 = _t('X1', NCHW, NHWC_FLAT, dev)
    Y = _out(shape=y_shape)
    _propagate_bcast_hw_shape(X0, X1, Y)
    assert (Y.hw_shape == NHWC_FLAT) is propagates


# ---------------------------------------------------------------------------
# DAG safety
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.opunit
def test_the_output_gets_a_copy_not_the_operand_s_list() -> None:
    """A SimTensor is a DAG edge that can fan out to several consumers, so the
    propagated view must be a copy.  Aliasing would make a later edit to one
    tensor's ``hw_shape`` silently rewrite the other's -- the same contract
    ``test_concat_graph_safety.py`` pins for ``.shape`` and ``.data``."""
    dev = _Device(True)
    X0 = _t('X0', NCHW, NHWC_FLAT, dev)
    X1 = _t('X1', NCHW, NHWC_FLAT, dev)
    Y = _out(shape=NCHW)
    _propagate_bcast_hw_shape(X0, X1, Y)
    assert Y.hw_shape == X0.hw_shape
    assert Y.hw_shape is not X0.hw_shape
    Y.hw_shape[2] = 9999
    assert X0.hw_shape == NHWC_FLAT


@pytest.mark.unit
@pytest.mark.opunit
def test_the_operands_are_left_untouched() -> None:
    """Shape inference must not rewrite its inputs."""
    dev = _Device(True)
    X0 = _t('X0', NCHW, NHWC_FLAT, dev)
    X1 = _t('X1', NCHW, None, dev)
    Y = _out(shape=NCHW)
    _propagate_bcast_hw_shape(X0, X1, Y)
    assert X0.hw_shape == NHWC_FLAT
    assert X0.shape == NCHW
    assert X1.hw_shape is None


# ---------------------------------------------------------------------------
# the call site
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.opunit
def test_bidir_bcast_actually_calls_the_propagation() -> None:
    """Everything above exercises the helper directly, which would keep passing
    if the call at ``ttsim/ops/desc/helpers.py:482`` were deleted.  This drives the real
    descriptor instead, and also pins that ``Y.shape`` is still inferred."""
    dev = _Device(True)
    X0 = _t('X0', NCHW, NHWC_FLAT, dev)
    X1 = _t('X1', NCHW, NHWC_FLAT, dev)
    Y = _out()
    bidir_bcast([X0, X1], [Y], _Op('Add'))
    assert Y.shape == NCHW
    assert Y.hw_shape == NHWC_FLAT


@pytest.mark.unit
@pytest.mark.opunit
def test_bidir_bcast_still_broadcasts_with_the_gate_off() -> None:
    """The default path: shape inference unchanged, no view carried.  This is
    what every workload other than an opted-in ResNet-50 run sees."""
    dev = _Device(False)
    X0 = _t('X0', [16, 1024, 14, 14], NHWC_FLAT, dev)
    X1 = _t('X1', [1024, 1, 1], NHWC_FLAT, dev)
    Y = _out()
    bidir_bcast([X0, X1], [Y], _Op('Add'))
    assert Y.shape == [16, 1024, 14, 14]
    assert Y.hw_shape is None
