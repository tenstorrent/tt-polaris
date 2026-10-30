# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""conv1's fold is emitted as the seven ops silicon runs, not one Fold SimOp.

Exercises ``ttsim.front.ttnn.op.fold_transpose_sharded_nchw``, a port of
tt-metal's ``fold_with_transpose_sharded_``
(``ttnn/cpp/ttnn/operations/data_movement/fold/fold.cpp:149``), which dispatches
Pad -> Transpose -> Pad -> Transpose -> view -> Transpose -> view -> Transpose
-> Slice.  The two ``ttnn::experimental::view`` calls are free, hence seven
profiler rows.

``ttnn.fold(use_transpose_as_fold=True)`` is a composite on hardware: the p100a
capture records Pad, Transpose, Pad, Transpose, Transpose, Transpose, Slice.
The shim's first-class Fold SimOp collapsed that to one row, so the graph
carried Fold + 2 Reshape against hardware's 7 — a 9-op gap, and the entire
op-sequence divergence from the capture at this position.

Batch
-----
The shapes are a function of ``(C, H, W, pad_*, stride_*)`` and the batch
dimension ONLY — ``ttsim/front/ttnn/op.py:1405-1485`` never lets N into a trailing dim.  So the
tests run at both batches the BH p100a refruns cover, b16 and b32
(``ttsim/front/ttnn/op.py:1375-1403``), with N templated into position 0 and every other extent
read off the capture unchanged.

This file asserts structure and shape only.

The trailing extents asserted here are READ OFF the capture
(``__refrun_cache/resnet50/bh/p100a/merged_ops_…-260816.csv`` rows 0-6).  They
are measurements, not derivations: if they ever need to change, the new numbers
come from a new capture, not from recomputing them.
"""

import pytest

import ttsim.front.ttnn as ttnn
from ttsim.front.ttnn.buffer import BufferType, TensorMemoryLayout
from ttsim.front.ttnn.device import close_device, open_device, set_default_device
from ttsim.front.ttnn.memory import MemoryConfig
from ttsim.front.ttnn.op import fold_transpose_sharded_nchw
from ttsim.front.ttnn.tensor import ttnn_random

#: Batches the BH p100a refrun covers.  Both open with the same seven ops and
#: the same trailing extents.
BATCHES = (16, 32)

#: conv1's fold output channel count: (3 + pad_c=1) * stride_h=2 * stride_w=2.
#: Spelled out because it is numerically 16 and so is the default batch — the
#: two are unrelated and an assertion that conflates them proves nothing.
FOLD_OUT_C = 16
#: conv1's input height/width after the fold.  113 is what the old shim's
#: one-sided padding produced; 115 is what silicon does.
FOLD_OUT_HW = 115


def expected_rows(batch: int):
    """(optype, input NCHW, output NCHW) for each of the seven profiler rows.

    Only position 0 depends on ``batch``.  The gaps between one row's output and
    the next row's input — [b,256,4,256] to [b,128,8,256], and [b,128,256,8] to
    [b,128,128,16] — are the two free ``experimental::view`` relabels, which
    hardware does not profile.
    """
    b = batch
    return [
        ('Pad',       [b, 3, 224, 224],   [b, 4, 256, 224]),
        ('Transpose', [b, 4, 256, 224],   [b, 4, 224, 256]),
        ('Pad',       [b, 4, 224, 256],   [b, 4, 256, 256]),
        ('Transpose', [b, 4, 256, 256],   [b, 256, 4, 256]),
        ('Transpose', [b, 128, 8, 256],   [b, 128, 256, 8]),
        ('Transpose', [b, 128, 128, 16],  [b, 128, 128, 16]),
        # DEVIATION, deliberate: the capture records this Slice's OUTPUT as NHWC
        # [b, 115, 115, 16].  fold_transpose_sharded_nchw folds the NHWC->NCHW
        # convention handoff into the Slice and returns [b, 16, 115, 115], because
        # polaris's conv_sinf reads C_in from dim 1.  The LUT key is unaffected — it
        # is built from an op's INPUT, and this Slice's input matches the capture
        # exactly — but the op's recorded OUTPUT shape does not match the profiler's
        # for this one row, which compare-layers will show as an output-shape
        # mismatch.  The downstream Halo reads hw_shape [1, 1, b*115*115, 16], which
        # is correct.
        ('Slice',     [b, 128, 128, 16],  [b, FOLD_OUT_C, FOLD_OUT_HW, FOLD_OUT_HW]),
    ]


@pytest.fixture(params=BATCHES, ids=[f'b{b}' for b in BATCHES])
def folded(request):
    """Emit the fold on a device configured the way the BH infra does."""
    batch = request.param
    device = open_device()
    device.set_arch('blackhole')
    set_default_device(device)
    device.compute_with_storage_grid_size(12, 10)

    x = ttnn_random((batch, 3, 224, 224), -1, 1, dtype=ttnn.bfloat16)
    # The infra reaches the fold via to_memory_config(HEIGHT_SHARDED, L1); the
    # LUT keys record that memory, so a bare tensor here would miss on
    # input_0_memory alone.
    x = ttnn.to_memory_config(
        x, MemoryConfig(TensorMemoryLayout.HEIGHT_SHARDED, BufferType.L1)
    )
    before = len(device.ops)
    out = fold_transpose_sharded_nchw(x, 2, 2, pad_h=3, pad_w=3, pad_c=1)
    ops = list(device.ops.values())[before:]
    yield batch, device, ops, out
    close_device(device)
    set_default_device(None)


@pytest.mark.unit
def test_emits_exactly_the_seven_hardware_ops(folded) -> None:
    batch, _device, ops, _out = folded
    assert [o.optype for o in ops] == [e[0] for e in expected_rows(batch)], (
        'op sequence diverged from the capture; the layer-by-layer correlation '
        'depends on this being exact'
    )


@pytest.mark.unit
def test_no_fold_or_reshape_survives(folded) -> None:
    """The two op types this change exists to remove.

    A Reshape here would mean the free, unprofiled reshapes between HW ops #3/#4
    and #4/#5 got emitted as real ops instead of in-place shape rewrites.
    """
    _batch, _device, ops, _out = folded
    assert not [o for o in ops if o.optype in ('Fold', 'Reshape')]


@pytest.mark.unit
@pytest.mark.parametrize('idx', range(7))
def test_shapes_match_the_capture(folded, idx) -> None:
    batch, _device, ops, _out = folded
    optype, in_shape, out_shape = expected_rows(batch)[idx]
    op = ops[idx]
    assert op.optype == optype
    assert op._frozen_input_shapes[0] == in_shape
    assert op._frozen_output_shapes[0] == out_shape


@pytest.mark.unit
def test_only_the_batch_dimension_tracks_the_batch(folded) -> None:
    """The property that makes this file batch-generic at all.

    Every extent except position 0 is fixed by the 224x224 input and the fold
    parameters, so a change that let N leak into a trailing dim — a shard-derived
    extent, say — would break the port's shapes at every batch but 16 and would
    otherwise be invisible here.
    """
    batch, _device, ops, _out = folded
    for op, (_optype, in_shape, out_shape) in zip(ops, expected_rows(batch)):
        assert op._frozen_input_shapes[0][0] == batch
        assert op._frozen_output_shapes[0][0] == batch
        assert op._frozen_input_shapes[0][1:] == in_shape[1:]
        assert op._frozen_output_shapes[0][1:] == out_shape[1:]


@pytest.mark.unit
def test_output_feeds_conv1(folded) -> None:
    """NCHW (N, 16, 115, 115): 115 is conv1_input_height, 16 its in-channels.

    115 is the number the old shim fold got wrong — its one-sided padding gave
    (224 + 3) // 2 = 113 and silently shrank conv1's input by two rows and
    columns.  Neither 115 nor the channel count moves with batch; only the
    hw_shape row count does.

    The order is NCHW rather than the capture's NHWC because
    fold_transpose_sharded_nchw performs the convention handoff itself; see the
    note on the Slice row in expected_rows().  hw_shape stays the NHWC-flat view
    that the Halo and every downstream LUT key read.
    """
    batch, _device, _ops, out = folded
    assert list(out.shape) == [batch, FOLD_OUT_C, FOLD_OUT_HW, FOLD_OUT_HW]
    assert list(out.hw_shape) == [
        1, 1, batch * FOLD_OUT_HW * FOLD_OUT_HW, FOLD_OUT_C,
    ]


@pytest.mark.unit
def test_rejects_non_nchw_input(folded) -> None:
    """Hardware pads the channel dim first, so the input must be NCHW."""
    batch, _device, _ops, _out = folded
    bad_rank = ttnn_random((batch, 224, 224), -1, 1, dtype=ttnn.bfloat16)
    with pytest.raises(ValueError, match='rank-4 NCHW'):
        fold_transpose_sharded_nchw(bad_rank, 2, 2, pad_h=3, pad_w=3, pad_c=1)
