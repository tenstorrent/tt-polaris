# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""_lowers_to_matmul / _norm_pair — use_matmul_for_1x1_conv.

Canonical: ``tt-metal/ttnn/cpp/ttnn/operations/conv/conv2d/conv2d_utils.cpp:519``.
"""
import pytest

from ttsim.front.ttnn.buffer import TensorMemoryLayout
from ttsim.front.ttnn.op import _lowers_to_matmul, _norm_pair


class _Cfg:
    def __init__(self, shard_layout=None):
        self.shard_layout = shard_layout


@pytest.mark.unit
def test_norm_pair_scalar_and_none_and_sequence():
    assert _norm_pair(1, (3, 3)) == (1, 1)
    assert _norm_pair(None, (3, 3)) == (3, 3)
    assert _norm_pair((2, 2), (1, 1)) == (2, 2)
    # 4-element padding (avg_pool2d) keeps its length
    assert _norm_pair([0, 0, 0, 0], (0, 0)) == (0, 0, 0, 0)


@pytest.mark.unit
def test_plain_1x1_lowers():
    assert _lowers_to_matmul((1, 1), (1, 1), (0, 0), (1, 1)) is True


@pytest.mark.unit
def test_scalar_kwargs_accepted():
    assert _lowers_to_matmul(1, 1, 0, 1) is True


@pytest.mark.parametrize("ks,st,pd,dl", [
    ((3, 3), (1, 1), (1, 1), (1, 1)),   # not 1x1
    ((1, 1), (2, 2), (0, 0), (1, 1)),   # stride 2 — resnet50's downsamples
    ((1, 1), (1, 2), (0, 0), (1, 1)),   # asymmetric stride
    ((1, 1), (1, 1), (1, 1), (1, 1)),   # padded
    ((1, 1), (1, 1), (0, 0), (2, 2)),   # dilated
])
@pytest.mark.unit
def test_each_term_blocks_lowering(ks, st, pd, dl):
    assert _lowers_to_matmul(ks, st, pd, dl) is False


@pytest.mark.unit
def test_width_sharded_blocks_lowering():
    cfg = _Cfg(TensorMemoryLayout.WIDTH_SHARDED)
    assert _lowers_to_matmul((1, 1), (1, 1), (0, 0), (1, 1), cfg) is False
    assert _lowers_to_matmul((1, 1), (1, 1), (0, 0), (1, 1),
                             _Cfg(TensorMemoryLayout.HEIGHT_SHARDED)) is True


@pytest.mark.unit
def test_four_element_zero_padding_still_lowers():
    assert _lowers_to_matmul((1, 1), (1, 1), (0, 0, 0, 0), (1, 1)) is True


@pytest.mark.unit
def test_resnet50_downsamples_are_not_matmuls():
    """The three 1x1 stride-2 projections; Halo + Conv2d on silicon."""
    for _in, _out in ((256, 512), (512, 1024), (1024, 2048)):
        assert _lowers_to_matmul((1, 1), (2, 2), (0, 0), (1, 1)) is False
