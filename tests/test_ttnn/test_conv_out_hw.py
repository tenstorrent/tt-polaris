# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""``_padding_n4`` and ``_conv_out_hw`` — the conv output-geometry helper.

This geometry used to be computed inline in three places in
``ttsim/front/ttnn/op.py``, each with its own copy of a six-line dilation
comment, and each spelling the padding term ``2*pd[0]`` / ``2*pd[-1]``.  That
form agrees with tt-metal for the two-element padding spelling ``(pad_h,
pad_w)`` and is WRONG for an asymmetric four-element one, because tt-metal's
``padding_n4`` is two-sided per axis::

    input_padded_height = input_height + padding_n4[0] + padding_n4[1]
    input_padded_width  = input_width  + padding_n4[2] + padding_n4[3]
    # tt-metal/ttnn/cpp/ttnn/operations/conv/conv2d/conv2d_utils.cpp:1405-1406, :1514-1515

The callers derive a grid and a shard spec from these dimensions, and the shard
spec lands in the LUT key — so a wrong output size here is a silent drop in hit
rate, not a crash.  Hence tests rather than trust.

Every expectation below is computed from the cpp formula
(``calculate_output_image_size``, ``tt-metal/ttnn/cpp/ttnn/operations/conv/conv2d/conv2d_utils.cpp:277-284``) by hand,
not
from the implementation.
"""

from __future__ import annotations

import pytest

from ttsim.front.ttnn.op import _conv_out_hw, _padding_n4


def _kw(**over) -> dict:
    """Conv kwargs for a 3x3 stride-1 pad-1 conv on a 56x56 input."""
    kw = dict(input_height=56, input_width=56, kernel_size=(3, 3),
              stride=(1, 1), padding=(1, 1), dilation=(1, 1))
    kw.update(over)
    return kw


# ---------------------------------------------------------------------------
# _padding_n4 — the (top, bottom, left, right) layout
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize('given, expected', [
    (None,          (0, 0, 0, 0)),
    (0,             (0, 0, 0, 0)),
    (2,             (2, 2, 2, 2)),
    ((1,),          (1, 1, 1, 1)),
    ((1, 2),        (1, 1, 2, 2)),      # (pad_h, pad_w) -> two-sided per axis
    ([1, 2],        (1, 1, 2, 2)),      # list spelling: resnet50 uses padding=[1, 1]
    ((1, 2, 3, 4),  (1, 2, 3, 4)),      # already n4
    ((0, 0, 0, 0),  (0, 0, 0, 0)),
])
def test_padding_n4_normalisation(given, expected) -> None:
    assert _padding_n4(given) == expected


@pytest.mark.unit
def test_padding_n4_two_element_is_not_h_w_h_w() -> None:
    """The trap this helper exists to prevent.

    A plausible-but-wrong reading of the four-element form is (h, w, h, w).
    Under that reading ``(1, 2)`` would expand to ``(1, 2, 1, 2)`` and the width
    pad would come out as ``1 + 2``.  tt-metal's layout is two-sided per axis,
    so height gets ``1 + 1`` and width ``2 + 2``.
    """
    n4 = _padding_n4((1, 2))
    assert (n4[0] + n4[1], n4[2] + n4[3]) == (2, 4)
    assert n4 != (1, 2, 1, 2)


@pytest.mark.unit
@pytest.mark.parametrize('bad', [(1, 2, 3), (1, 2, 3, 4, 5), (1, 1, 1, 1, 0, 32)])
def test_padding_n4_rejects_other_lengths(bad) -> None:
    """A 3-, 5- or 6-element padding is not a conv padding.

    The 6-element case is real: ResNet-50's fold passes
    ``padding=[pad_h, pad_h, pad_w, pad_w, 0, pad_c]`` (tt-metal
    ``tt-metal/models/demos/vision/classification/resnet50/ttnn_resnet/tt/ttnn_functional_resnet50.py:614``).  That
    goes to ``fold``, never to a conv, and guessing at it here would be worse
    than declining.
    """
    with pytest.raises(ValueError, match='1, 2 or 4 elements'):
        _padding_n4(bad)


# ---------------------------------------------------------------------------
# _conv_out_hw — cpp:277-284
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize('over, expected', [
    # 3x3 s1 p1 on 56x56 -> same size.  ResNet-50 layer1 conv2.
    ({}, (56, 56)),
    # 1x1 s1 p0 -> same size.  The bottleneck's conv1/conv3.
    ({'kernel_size': (1, 1), 'padding': (0, 0)}, (56, 56)),
    # 1x1 s2 p0 -> halves.  The three downsample convs.
    ({'kernel_size': (1, 1), 'stride': (2, 2), 'padding': (0, 0)}, (28, 28)),
    # 3x3 s2 p1 -> halves.
    ({'stride': (2, 2)}, (28, 28)),
    # 7x7 s2 p3 on 224x224 -> 112x112.  The stem.
    ({'input_height': 224, 'input_width': 224, 'kernel_size': (7, 7),
      'stride': (2, 2), 'padding': (3, 3)}, (112, 112)),
    # Non-square input and kernel: in 56x28, k 3x1, p (1, 0)
    #   h = (56 + 1+1 - 1*(3-1) - 1)//1 + 1 = 56
    #   w = (28 + 0+0 - 1*(1-1) - 1)//1 + 1 = 28
    ({'input_width': 28, 'kernel_size': (3, 1), 'padding': (1, 0)}, (56, 28)),
    # Dilation 2, k3, p2: effective extent 2*(3-1)+1 = 5
    #   h = (56 + 2+2 - 2*2 - 1)//1 + 1 = 56
    ({'padding': (2, 2), 'dilation': (2, 2)}, (56, 56)),
    # Scalar spellings must behave like the tuple ones.
    ({'kernel_size': 3, 'stride': 1, 'padding': 1}, (56, 56)),
])
def test_conv_out_hw_matches_cpp(over, expected) -> None:
    assert _conv_out_hw(_kw(**over)) == expected


@pytest.mark.unit
def test_symmetric_four_element_agrees_with_two_element() -> None:
    """(1, 1, 1, 1) and (1, 1) describe the same padding, so same answer.

    This is the case the old ``2*pd[0]`` form got right by accident, and it must
    keep working -- it is what every in-tree conv actually passes.
    """
    assert _conv_out_hw(_kw(padding=(1, 1, 1, 1))) == _conv_out_hw(_kw(padding=(1, 1)))


@pytest.mark.unit
def test_asymmetric_four_element_padding_is_the_regression() -> None:
    """The bug the helper fixes, pinned with numbers.

    ``padding=(2, 0, 2, 0)`` on a 56x56 3x3 stride-1 conv:

        cpp     h = (56 + 2+0 - 1*(3-1) - 1)//1 + 1 = 56
                w = (56 + 2+0 - 1*(3-1) - 1)//1 + 1 = 56

    The old inline form used ``2*pd[0]`` for height (2*2 = 4, not 2+0 = 2) and
    ``2*pd[-1]`` for width (2*0 = 0, not 2+0 = 2), giving (58, 54) -- wrong on
    BOTH axes, and in opposite directions, which is why neither a spot-check nor
    a symmetric test would have caught it.
    """
    assert _conv_out_hw(_kw(padding=(2, 0, 2, 0))) == (56, 56)

    in_h = in_w = 56
    pd = (2, 0, 2, 0)
    old_h = (in_h + 2 * pd[0] - 1 * (3 - 1) - 1) // 1 + 1
    old_w = (in_w + 2 * pd[-1] - 1 * (3 - 1) - 1) // 1 + 1
    assert (old_h, old_w) == (58, 54), 'the old form is mis-stated in this test'
    assert _conv_out_hw(_kw(padding=pd)) != (old_h, old_w)


@pytest.mark.unit
def test_transpose_declines_rather_than_returning_the_forward_value() -> None:
    """``is_transpose=True`` must return None, not a forward-conv answer.

    tt-metal's ``calculate_ct2d_output_image_size`` (cpp:288-299) MULTIPLIES by
    stride and SUBTRACTS padding where the forward formula divides and adds::

        out_h = ((in_h - 1)*stride) - (pad[0]+pad[1]) + dil*(k-1) + out_pad + 1

    For 28x28, k3, s2, p1 that is ``27*2 - 2 + 2 + 0 + 1 = 55``, against the
    forward formula's 14 -- a factor of ~4.  Returning the forward value would
    hand the caller a shard spec for the wrong tensor, so the helper declines
    and the caller early-returns instead.

    Both call sites reach here: ``conv_transpose2d`` is wrapped by ``_with_halo``
    just as ``conv2d`` is, so a transposed conv does traverse
    ``_maybe_emit_conv_input_reshard``.
    """
    kw = _kw(input_height=28, input_width=28, stride=(2, 2))
    assert _conv_out_hw(kw, is_transpose=False) == (14, 14)
    assert _conv_out_hw(kw, is_transpose=True) is None


@pytest.mark.unit
@pytest.mark.parametrize('over', [
    {'input_height': 0},                      # geometry not yet known
    {'input_width': 0},
    {'input_height': 1, 'kernel_size': (9, 9), 'padding': (0, 0)},  # out_h <= 0
    {'stride': (0, 0)},                       # would divide by zero
    {'padding': (1, 2, 3)},                   # unusable padding length
])
def test_conv_out_hw_declines_instead_of_raising(over) -> None:
    """Every unusable input returns None.

    The callers are on the graph-build path and early-return on None; a raise
    here would take down a whole workload over a conv whose shard config is
    optional.  ``stride=(0, 0)`` is the one that would otherwise be a
    ZeroDivisionError rather than a wrong number.
    """
    assert _conv_out_hw(_kw(**over)) is None


@pytest.mark.unit
def test_missing_keys_entirely_declines() -> None:
    """Called with kwargs carrying no geometry at all (the early-build case)."""
    assert _conv_out_hw({}) is None
