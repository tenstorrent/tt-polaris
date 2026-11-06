#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Layout-propagation contract for the ttnn shim, anchored to tt-metal op definitions.

Each assertion below encodes what the *real* tt-metal op does with the ``.layout``
of its output tensor, cited to the authoritative C++ op definition in the sibling
``../tt-metal`` checkout (path/line references are to the revision current when
these tests were written; grep the cited function if line numbers drift).  The
goal is to keep the shim a sound mimic of HW — not merely self-consistent — so a
future refactor of the op constructors cannot silently reintroduce the
ROW_MAJOR-default bug (which made the whole llama3 decode block emit ROW_MAJOR LUT
keys while silicon is TILE) or break the VGG halo→conv ROW_MAJOR boundary.

Ground-truth sources used here:
  * tt-metal op ``compute_output_specs`` (the ``PageConfig(Layout::…)`` it stamps
    on the output TensorSpec) — the definitive per-op output layout, and
  * silicon captures (the profiler ``OUTPUT_0_LAYOUT`` / ``INPUT_0_LAYOUT``
    columns) — empirical confirmation on real hardware.
"""

import pytest

import ttsim.front.ttnn as ttnn
from ttsim.front.ttnn.device import ARCH, Device
from ttsim.front.ttnn.tensor import DataType, Layout, Tensor


def _make_device():
    device = Device(device_id=0)
    device.architecture = ARCH.WORMHOLE_B0
    return device


def _t(name, shape, device, layout=Layout.ROW_MAJOR_LAYOUT, dtype=DataType.BFLOAT16):
    return Tensor(name=name, shape=shape, dtype=dtype, layout=layout, device=device)


# ---------------------------------------------------------------------------
# single_output_immediate_op — layout-preserving eltwise (Add / Mul / …)
#
# tt-metal anchor: binary_ng_device_operation.cpp — BinaryNgDeviceOperation::
#   compute_output_specs stamps PageConfig(attributes.output_layout) (~L467/473),
#   and output_layout is resolved (~L578-581) as:
#       output_layout = Layout::TILE;
#       if both inputs ROW_MAJOR -> output_layout = Layout::ROW_MAJOR;
#   i.e. matched-layout inputs are preserved on the output (TILE→TILE, RM→RM).
# Capture confir: BinaryNgDeviceOperation INPUT_0_LAYOUT == its producer's layout.
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_add_preserves_tile_layout():
    device = _make_device()
    a = _t("a", [1, 1, 32, 128], device, layout=Layout.TILE_LAYOUT)
    b = _t("b", [1, 1, 32, 128], device, layout=Layout.TILE_LAYOUT)
    out = ttnn.add(a, b)
    assert out.get_layout() == Layout.TILE_LAYOUT


@pytest.mark.unit
def test_add_preserves_row_major_layout():
    device = _make_device()
    a = _t("a", [1, 1, 32, 128], device, layout=Layout.ROW_MAJOR_LAYOUT)
    b = _t("b", [1, 1, 32, 128], device, layout=Layout.ROW_MAJOR_LAYOUT)
    out = ttnn.add(a, b)
    assert out.get_layout() == Layout.ROW_MAJOR_LAYOUT


# ---------------------------------------------------------------------------
# Explicit layout= kwarg wins — Embedding (RM indices → TILE activation)
#
# tt-metal anchor: embedding.cpp (~L46-53) sets fused_tilized=True when the
#   caller passes layout == ttnn::TILE_LAYOUT; embedding_device_operation.cpp
#   (~L119-120) then computes:
#       output_layout = (tilized && input.layout() != TILE) ? TILE : ROW_MAJOR;
#   So ttnn.embedding(indices_RM, weight_RM, layout=TILE_LAYOUT) yields a TILE
#   output even though both inputs are ROW_MAJOR. This is the exact call the
#   llama3 token-embedding makes, and the source of the block's TILE activation.
# Capture confirm: EmbeddingsDeviceOperation INPUT_0_LAYOUT=ROW_MAJOR (token ids),
#   OUTPUT_0_LAYOUT=TILE.
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_embedding_layout_kwarg_forces_tile_from_rm_inputs():
    device = _make_device()
    indices = _t("ids", [1, 32], device, layout=Layout.ROW_MAJOR_LAYOUT, dtype=DataType.INT32)
    weight = _t("w", [128256, 4096], device, layout=Layout.ROW_MAJOR_LAYOUT)
    out = ttnn.embedding(indices, weight, layout=ttnn.TILE_LAYOUT)
    assert out.get_layout() == Layout.TILE_LAYOUT


@pytest.mark.unit
def test_embedding_without_layout_kwarg_stays_row_major():
    # No layout= → not tilized → ROW_MAJOR output (embedding_device_operation.cpp
    # L119-120 with tilized=False). Confirms the kwarg — not a blanket default —
    # is what drives TILE.
    device = _make_device()
    indices = _t("ids", [1, 32], device, layout=Layout.ROW_MAJOR_LAYOUT, dtype=DataType.INT32)
    weight = _t("w", [128256, 4096], device, layout=Layout.ROW_MAJOR_LAYOUT)
    out = ttnn.embedding(indices, weight)
    assert out.get_layout() == Layout.ROW_MAJOR_LAYOUT


# ---------------------------------------------------------------------------
# Layout-changing data-movement ops — Tilize / Untilize
#
# tt-metal anchors:
#   to_layout (ttnn public entry) dispatches a ROW_MAJOR→TILE conversion to
#     TilizeDeviceOperation — tilize_device_operation.cpp: input asserted
#     ROW_MAJOR (~L98); output spec PageConfig(Layout::TILE) (~L161/168/172).
#   untilize_device_operation.cpp — input asserted TILE (~L101); output spec
#     PageConfig(Layout::ROW_MAJOR) (~L268).
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_to_layout_tile_outputs_tile():
    device = _make_device()
    x = _t("x", [1, 1, 32, 128], device, layout=Layout.ROW_MAJOR_LAYOUT)
    out = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
    assert out.get_layout() == Layout.TILE_LAYOUT


@pytest.mark.unit
def test_untilize_outputs_row_major():
    device = _make_device()
    x = _t("x", [1, 1, 32, 128], device, layout=Layout.TILE_LAYOUT)
    out = ttnn.untilize(x)
    assert out.get_layout() == Layout.ROW_MAJOR_LAYOUT


@pytest.mark.unit
@pytest.mark.parametrize('in_dt,out_dt', [
    (ttnn.bfloat8_b, ttnn.bfloat16),   # row-major cannot hold a block float
    (ttnn.bfloat16, ttnn.bfloat16),
    (ttnn.uint32, ttnn.uint32),
])
def test_untilize_output_dtype(in_dt, out_dt):
    """tt-metal untilize_output_dtype (data_movement/common/common.hpp): BFLOAT8_B -> BFLOAT16, else
    unchanged (WH n150 llama3 prefill capture: Untilize [1,1,32,128256] BFLOAT8_B -> BFLOAT16)."""
    device = _make_device()
    x = ttnn.zeros([1, 1, 32, 128], dtype=in_dt, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.untilize(x)
    assert out._ttnn_dtype == out_dt


# ---------------------------------------------------------------------------
# matmul / linear — output follows the (TILE) activation
#
# tt-metal anchor: matmul_device_operation.cpp — both inputs must be TILE
#   (TT_FATAL ~L750) and the output layout is
#       output_layout = attributes.untilize_out ? ROW_MAJOR : TILE;   (~L1834)
#   stamped into the output spec (~L1912-1954). So a (non-untilize_out) matmul of
#   TILE activations produces a TILE result. ttnn.linear is a matmul with a fused
#   bias, same output-layout rule.
# Capture confirm: MatmulDeviceOperation INPUT_0_LAYOUT=TILE (both VGG & llama3).
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_matmul_tile_activation_outputs_tile():
    device = _make_device()
    a = _t("a", [1, 1, 32, 64], device, layout=Layout.TILE_LAYOUT)
    b = _t("b", [1, 1, 64, 128], device, layout=Layout.TILE_LAYOUT)
    out = ttnn.matmul(a, b)
    assert out.get_layout() == Layout.TILE_LAYOUT


@pytest.mark.unit
def test_linear_tile_activation_outputs_tile():
    # linear was a separate constructor that also dropped layout — the qkv / mlp /
    # o projections in the llama3 block all go through it, so this is the exact
    # path that poisoned the block with ROW_MAJOR before the fix.
    device = _make_device()
    a = _t("a", [1, 1, 32, 4096], device, layout=Layout.TILE_LAYOUT)
    w = _t("w", [1, 1, 4096, 6144], device, layout=Layout.TILE_LAYOUT)
    out = ttnn.linear(a, w)
    assert out.get_layout() == Layout.TILE_LAYOUT


# ---------------------------------------------------------------------------
# unsqueeze_to_4D — metadata reshape, layout-preserving
#
# tt-metal anchor: core.cpp unsqueeze_to_4D (~L19) is implemented as
#   ttnn::reshape(...) (a view/metadata op); reshape does not change layout, so a
#   TILE input stays TILE. In the llama3 decode path this sits between the TILE
#   embedding and model.forward, and previously reset the activation to RM.
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize("layout", [Layout.TILE_LAYOUT, Layout.ROW_MAJOR_LAYOUT])
def test_unsqueeze_to_4d_preserves_layout(layout):
    device = _make_device()
    x = _t("x", [32, 4096], device, layout=layout)
    out = ttnn.unsqueeze_to_4D(x)
    assert out.get_layout() == layout


# ---------------------------------------------------------------------------
# multiple_output_immediate_op — every output inherits the input layout (Split)
#
# tt-metal anchor: split is a data-movement (slice) op — SliceDeviceOperation
#   preserves the input layout on each output shard (capture: SliceDeviceOperation
#   INPUT_0_LAYOUT==OUTPUT_0_LAYOUT). Guards the multi-output constructor.
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_split_outputs_inherit_tile_layout():
    device = _make_device()
    x = _t("x", [1, 1, 32, 64], device, layout=Layout.TILE_LAYOUT)
    outs = ttnn.split(x, 32, dim=3)
    assert len(outs) == 2
    for o in outs:
        assert o.get_layout() == Layout.TILE_LAYOUT


# ---------------------------------------------------------------------------
# Halo → ROW_MAJOR (unconditional) — the VGG conv-boundary invariant
#
# tt-metal anchor: halo_device_operation.cpp — compute_output_specs builds the
#   output TensorSpec with PageConfig(Layout::ROW_MAJOR) (~L87), UNCONDITIONALLY
#   (it untilizes a TILE input on the way through — see the "no need to untilize"
#   fast-path ~L26 for RM input). So HaloDeviceOperation always emits ROW_MAJOR,
#   which is what feeds the ROW_MAJOR-input conv.
# Capture confirm: HaloDeviceOperation OUTPUT_0_LAYOUT=ROW_MAJOR 32/32, and
#   Conv2dDeviceOperation INPUT_0_LAYOUT=ROW_MAJOR 28/28.
# Without this, global input-layout inheritance would carry a tile-layout conv
# output through halo→move→conv and break VGG's conv LUT keys.
# ---------------------------------------------------------------------------

def _emitted_halo_output_layout(device):
    halo_ops = [op for op in device.ops.values() if op.optype == "Halo"]
    assert halo_ops, "expected a Halo SimOp to be auto-emitted"
    return device.tensors[halo_ops[0].outList[0]].get_layout()


@pytest.mark.unit
@pytest.mark.parametrize("in_layout", [Layout.ROW_MAJOR_LAYOUT, Layout.TILE_LAYOUT])
def test_halo_output_is_row_major_regardless_of_input(in_layout):
    device = _make_device()
    x = _t("x", [1, 3, 8, 8], device, layout=in_layout)
    w = _t("w", [4, 3, 3, 3], device)
    b = _t("b", [4], device)
    ttnn.conv2d(
        input_tensor=x, weight_tensor=w, bias_tensor=b,
        in_channels=3, out_channels=4, batch_size=1,
        input_height=8, input_width=8, kernel_size=(3, 3),
        stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1, device=device,
    )
    assert _emitted_halo_output_layout(device) == Layout.ROW_MAJOR_LAYOUT


# ---------------------------------------------------------------------------
# End-to-end propagation chain — embedding(TILE) → unsqueeze → add stays TILE
#
# This is the failure mode the fix targets: TILE must survive across a chain of
# layout-preserving ops so the transformer block's LUT keys are TILE (matching
# silicon), not ROW_MAJOR. See the per-op anchors above for each hop.
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_tile_propagates_through_chain():
    device = _make_device()
    indices = _t("ids", [1, 32], device, layout=Layout.ROW_MAJOR_LAYOUT, dtype=DataType.INT32)
    weight = _t("w", [128256, 4096], device, layout=Layout.ROW_MAJOR_LAYOUT)
    x = ttnn.embedding(indices, weight, layout=ttnn.TILE_LAYOUT)   # RM inputs → TILE
    x = ttnn.unsqueeze_to_4D(x)                                     # preserves TILE
    x = ttnn.add(x, x)                                              # preserves TILE
    assert x.get_layout() == Layout.TILE_LAYOUT


# ---------------------------------------------------------------------------
# Non-interference: setting output layout must not disturb the existing dtype /
# memory_config propagation (_propagate_ttnn_dtype / _propagate_memory_config).
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_layout_fix_does_not_clobber_dtype_and_memory_propagation():
    device = _make_device()
    a = _t("a", [1, 1, 32, 128], device, layout=Layout.TILE_LAYOUT)
    b = _t("b", [1, 1, 32, 128], device, layout=Layout.TILE_LAYOUT)
    a._ttnn_dtype = DataType.BFLOAT8_B
    out = ttnn.add(a, b)
    assert out.get_layout() == Layout.TILE_LAYOUT
    # dtype still propagates from the input overlay
    assert getattr(out, "_ttnn_dtype", None) == DataType.BFLOAT8_B
