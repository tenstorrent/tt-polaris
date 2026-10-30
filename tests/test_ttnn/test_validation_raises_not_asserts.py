# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Caller-input validation must raise, not assert.

``python -O`` (and its quieter twin ``PYTHONOPTIMIZE=1``, which a Dockerfile or
CI job can set without anything on the command line hinting at it) removes
``assert`` statements at COMPILE time -- the check is not evaluated and skipped,
it is absent from the bytecode.  So a bare ``assert`` guarding a caller's
arguments provides nothing under ``-O``.

The distinction that matters is not "asserts bad" but who the check is for:

    assert   -- an internal invariant.  Cannot fail unless this module has a
                bug, so compiling it away is harmless.
    raise    -- validation of something crossing a boundary.  Fires in a
                CORRECT program given bad input, so it must survive.

All five converted here were the second category spelled as the first.  Each guards a value
that, unchecked, produces a plausible-looking wrong result rather than a crash:
a ``CoreGrid`` with a missing axis, or a non-rank-4 shape reaching an
``N, C, H, W`` unpack.  Those feed shard specs, which feed LUT keys -- so the
symptom would be a silent hit-rate drop in a run that looked like it worked.

The two matching checks in ``conv_sharding._num_cores_to_corerangeset`` are
pinned in ``test_num_cores_to_corerangeset.py`` rather than duplicated here.

Deliberately NOT converted: the ~410 other asserts in ``ttsim/``, including
``ttsim/ops/tensor.py::require_shape_list``.  Most are genuine internal
invariants, and converting them wholesale would be cargo-culting the rule
rather than applying it.
"""

from __future__ import annotations

from typing import Any

import pytest

from ttsim.front.ttnn.core import CoreGrid
from ttsim.front.ttnn.op import (
    fold_transpose_sharded_nchw,
    pad_channels_nchw,
    permute_reshape_to_nhwc_flat,
)
from ttsim.ops.tensor import SimTensor


def _t(shape: list[int]) -> Any:
    """A bare shape carrier for the validation paths under test.

    Returns ``Any`` rather than ``SimTensor`` on purpose.  The functions below
    are annotated ``Tensor`` (the ttnn front-end one), but every check here
    fires on ``len(shape)`` BEFORE anything Tensor-specific is touched, so a
    SimTensor is sufficient and avoids dragging a device into a pure-validation
    test.  Widening the production annotations to accept both would weaken them
    for the callers that genuinely need a Tensor; typing the stand-in as ``Any``
    keeps the looseness where it belongs -- in the fixture.
    """
    return SimTensor({'name': 't', 'shape': list(shape), 'op_in': [], 'op_out': []})


# ---------------------------------------------------------------------------
# core.py — CoreGrid(x=, y=)
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize('kw', [{'x': 8}, {'y': 8}])
def test_core_grid_requires_both_axes(kw) -> None:
    """One axis without the other is a caller error, not an internal invariant.

    Stripped by ``-O``, the old assert let ``CoreGrid(x=8)`` fall through to
    ``int(None)`` and die with a bare ``TypeError`` from a line that does not
    mention the real problem.
    """
    with pytest.raises(ValueError, match='requires both x and y'):
        CoreGrid(**kw)


@pytest.mark.unit
def test_core_grid_both_axes_still_builds() -> None:
    """Positive control: the validation did not become over-eager."""
    g = CoreGrid(x=8, y=4)
    assert (g.x, g.y) == (8, 4)
    assert g.num_cores() == 32


@pytest.mark.unit
def test_core_grid_positional_ranges_form_is_unaffected() -> None:
    """The legacy positional spelling takes the other branch and must not raise."""
    assert CoreGrid([]).x == 0


# ---------------------------------------------------------------------------
# op.py — the fold path's rank-4 check
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize('shape', [[1, 2, 3], [1, 2, 3, 4, 5]])
def test_fold_transpose_sharded_nchw_rejects_non_rank4(shape) -> None:
    """Guards the ``N, C, H, W = in_shape`` unpack immediately below it.

    Without the check a rank-3 shape raises ``ValueError: not enough values to
    unpack`` -- the same exception class, but naming neither the function nor
    the shape, from a line that looks like a bug in op.py rather than bad input.
    """
    bad = _t(shape)
    with pytest.raises(ValueError, match='fold expects rank-4 NCHW input'):
        fold_transpose_sharded_nchw(bad, 2, 2, pad_h=3, pad_w=3, pad_c=1)


# ---------------------------------------------------------------------------
# op.py — the vgg_unet pair
# ---------------------------------------------------------------------------
#
# ``pad_channels_nchw`` and ``permute_reshape_to_nhwc_flat`` are consecutive
# calls in ``workloads/ttnn/vgg_unet/common/ttnn_vgg_unet.py:294-295`` and both
# public through the ``ttnn`` namespace.  They carry the same rank-4 check as the
# one above and were converted with it so the pattern is actually gone from
# op.py rather than reduced -- ``grep 'assert len(in_shape) == 4'`` now returns
# nothing.

@pytest.mark.unit
@pytest.mark.parametrize('shape', [[1, 2, 3], [1, 2, 3, 4, 5], [16, 3]])
def test_pad_channels_nchw_rejects_non_rank4(shape) -> None:
    bad = _t(shape)
    with pytest.raises(ValueError, match='NCHW input with C<target required'):
        pad_channels_nchw(bad, target_channels=16)


@pytest.mark.unit
@pytest.mark.parametrize('c', [16, 32])
def test_pad_channels_nchw_rejects_channels_not_below_target(c) -> None:
    """The second conjunct, and the one with teeth under ``-O``.

    A rank error falls through to an unpack that at least raises.  ``C >=
    target`` does not: ``pad_amount = target_channels - in_shape[1]`` goes zero
    or NEGATIVE, and a Pad op with a negative pad is a silently wrong shape, not
    an error.  That is the asymmetry that makes this the more important of the
    two halves to keep at runtime.
    """
    bad = _t([1, c, 224, 224])
    with pytest.raises(ValueError, match='NCHW input with C<target required'):
        pad_channels_nchw(bad, target_channels=16)


#: No positive control for ``pad_channels_nchw`` / ``permute_reshape_to_nhwc_flat``:
#: past the validation they emit SimOps and build ``ttnn.Tensor``s from
#: ``input_tensor.get_layout()`` and ``.device``, which a bare ``SimTensor``
#: does not carry.  A "positive" test here would need the full device fixture and
#: would then be testing emission, not this guard -- the happy path for both is
#: already covered end-to-end by the vgg_unet entry-path tests.  The negatives
#: above are sound precisely because they raise BEFORE any of that.


@pytest.mark.unit
@pytest.mark.parametrize('shape', [[1, 2, 3], [1, 2, 3, 4, 5]])
def test_permute_reshape_to_nhwc_flat_rejects_non_rank4(shape) -> None:
    bad = _t(shape)
    with pytest.raises(ValueError, match='expects rank-4 NCHW input'):
        permute_reshape_to_nhwc_flat(bad)
