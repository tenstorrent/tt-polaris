#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Layout, shard, and transformer head op descriptors:
  Tilize, Untilize, TilizeWithValPadding, UntilizeWithUnpadding,
  InterleavedToSharded, ShardedToInterleaved, Reshard, Halo, Move,
  ConcatHeads, CreateQKVHeads.
Used by the TTNN front-end tracking-only operator APIs (*_op helpers) in ttnn_shim.
"""

from ttsim.ops.desc.registry import register_ops
from ttsim.ops.tensor import _coerce_shape_to_list, require_shape_list

TILE_HEIGHT = 32
TILE_WIDTH = 32

# SimOp domain for layout ops (TTNN / device kernel names; not standard ONNX opset).
_TTNN_OP_DOMAIN = "com.tenstorrent.ttnn"

# ---------------------------------------------------------------------------
# Halo geometry: physical halo-extended Y dimension lookup table.
#
# Key: (NHW, C, kH, kW, pH, pW, is_transpose)
#   NHW         = N * H * W  (logical spatial volume from NCHW input shape)
#   C           = channel count
#   kH, kW      = kernel height/width
#   pH, pW      = padding height/width
#   is_transpose= True for conv_transpose2d (SlidingWindowConfig is_transpose flag)
#
# Value: halo_ext_y = max_out_nsticks_per_core × num_cores_nhw
#
# Source: VGG UNet WH profiler trace (2026-04-23); shape verified against
# SlidingWindowConfig ATTRIBUTES in HaloDeviceOperation profiler rows.
# MaxPool 2×2 (is_transpose=False, kH=kW=2, pH=pW=0) is intentionally absent:
# no border-pixel exchange is needed for non-overlapping strides, so the halo
# buffer equals the input (pass-through).
# ---------------------------------------------------------------------------
_HALO_EXT_Y: dict[tuple[int, int, int, int, int, int, bool], int] = {
    # --- Regular 3×3 conv, pad=1, stride=1 ---
    # Stage 1 (256×256, NHW=65536)
    (65536, 16,  3, 3, 1, 1, False): 99072,
    (65536, 64,  3, 3, 1, 1, False): 99072,
    (65536, 128, 3, 3, 1, 1, False): 99072,
    # Stage 2 (128×128, NHW=16384)
    (16384, 64,  3, 3, 1, 1, False): 33280,
    (16384, 128, 3, 3, 1, 1, False): 33280,
    (16384, 256, 3, 3, 1, 1, False): 33280,
    # Stage 3 (64×64, NHW=4096)
    (4096,  128, 3, 3, 1, 1, False): 12672,
    (4096,  256, 3, 3, 1, 1, False): 5280,
    (4096,  512, 3, 3, 1, 1, False): 5280,
    # Stage 4 (32×32, NHW=1024)
    (1024,  256,  3, 3, 1, 1, False): 1632,
    (1024,  512,  3, 3, 1, 1, False): 1632,
    (1024,  1024, 3, 3, 1, 1, False): 1632,
    # Bottleneck (16×16, NHW=256)
    (256,   512,  3, 3, 1, 1, False): 576,
    # --- ConvTranspose 2×2, pad=0, stride=2 (decoder upsampling) ---
    (256,   512, 2, 2, 0, 0, True): 1320,
    (1024,  512, 2, 2, 0, 0, True): 4680,
    (4096,  256, 2, 2, 0, 0, True): 24768,
    (16384, 128, 2, 2, 0, 0, True): 82240,
}


# Per-arch overrides to ``_HALO_EXT_Y``. The base table is empirical (WH n150 trace);
# some halo positions emit a different extended-y on other arches because
# ``determine_parallel_config`` picks a workload-specific ``num_cores_nhw``.
# Applied by the annotation pass in ttsim/back/device.py at execute_graph time, when
# the back Device's instance name is known. See doc/TTNN_SHIM_ARCHITECTURE.md §17.
_HALO_EXT_Y_OVERRIDES_BY_DEVICE: dict[
    str, dict[tuple[int, int, int, int, int, int, bool], int]
] = {
    # BH p100a (grid 12x10): d2.conv1 (4096 nhw, 512 channels, k=3 p=1 s=1) lands on
    # num_cores_nhw=10 (cores=80, 10x8 grid) per tt-metal's parallel-config picker
    # — different from the base table's WH assumption of num_cores_nhw=8 (cores=64).
    # HW-captured extended y = 5620 (verified against
    # __refrun_cache/vgg_unet/bh/p100a/merged_ops_dualref_260519.csv).
    "p100a": {
        (4096, 512, 3, 3, 1, 1, False): 5620,
        # d4 up convtranspose: WH n150 emits y=82240 (the base table value); BH p100a's
        # parallel-config for the underlying conv2d (post-zero-insert upsample + halo) picks
        # different num_cores → HW captures y=88408 (verified against
        # __refrun_cache/vgg_unet/bh/p100a/merged_ops_dualref_260519.csv convtranspose entry).
        (16384, 128, 2, 2, 0, 0, True): 88408,
    },
}


def _round_up(value, multiple):
    return ((value + multiple - 1) // multiple) * multiple


def _pad_to_tile_shape(shape):
    """Return list with last two dims rounded up to tile boundaries."""
    if len(shape) < 2:
        return list(shape)
    s = list(shape)
    s[-2] = _round_up(s[-2], TILE_HEIGHT)
    s[-1] = _round_up(s[-1], TILE_WIDTH)
    return s


def _nelems(shape):
    if not shape:
        return 1
    n = 1
    for d in shape:
        n *= int(d)
    return n


def _input_storage_shape_for_perf(X, op, logical_shape_list):
    """Dims for input element/byte perf (padded TILE storage when known); output logical shape stays separate."""
    attr_ps = op.attrs.get('input_padded_shape')
    if attr_ps is not None:
        try:
            ps_list = _coerce_shape_to_list(attr_ps)
        except TypeError:
            ps_list = []
        if ps_list:
            return ps_list
    ps_fn = getattr(X, 'padded_shape', None)
    if callable(ps_fn):
        try:
            ps_obj = ps_fn()
        except Exception:
            ps_obj = None
        if ps_obj is not None:
            try:
                return _coerce_shape_to_list(ps_obj)
            except TypeError:
                pass
    return list(logical_shape_list)


def tilize_sinf(iTList, oTList, op, **kwargs):
    """Shape inference for Tilize: 1 input (ROW_MAJOR), output same logical shape, padded to tile."""
    assert len(iTList) == 1 and len(oTList) == 1
    X = iTList[0]
    in_shape = require_shape_list(
        X.shape,
        "Tilize shape inference: input tensor shape must be known for element/byte accounting",
    )

    # Output logical shape is unchanged: tilize only reorders data within tile boundaries.
    out_logical = in_shape
    # Padded shape rounds last two dims up to TILE_HEIGHT/TILE_WIDTH so data fits in whole tiles.
    out_padded = _pad_to_tile_shape(in_shape) if len(in_shape) >= 2 else in_shape

    oTList[0].shape = out_logical
    oTList[0].dtype = X.dtype

    # perf_stats: element counts and bytes for input (row-major) and output (tiled); instrs model data movement.
    elem_size = op.attrs.get('element_size', 2)
    in_elems = _nelems(in_shape)
    out_elems = _nelems(out_padded)
    in_bytes = in_elems * elem_size
    out_bytes = out_elems * elem_size

    op.perf_stats = {
        'inElems': in_elems,
        'outElems': out_elems,
        'inBytes': in_bytes,
        'outBytes': out_bytes,
        'instrs': {'mov': out_elems},
    }
    return


def untilize_sinf(iTList, oTList, op, **kwargs):
    """Shape inference for Untilize: 1 input (TILE), output same logical shape (ROW_MAJOR)."""
    assert len(iTList) == 1 and len(oTList) == 1
    X = iTList[0]
    in_shape = require_shape_list(
        X.shape,
        "Untilize shape inference: input tensor shape must be known for element/byte accounting",
    )

    # Untilize preserves logical shape; output is row-major with same dimensions (no padding change in logical view).
    out_shape = in_shape

    oTList[0].shape = out_shape
    oTList[0].dtype = X.dtype

    # perf_stats: input from TILE storage (padded) when available; output logical row-major; mov = elements written out.
    elem_size = op.attrs.get('element_size', 2)
    in_storage = _input_storage_shape_for_perf(X, op, in_shape)
    in_elems = _nelems(in_storage)
    out_elems = _nelems(in_shape)

    op.perf_stats = {
        'inElems': in_elems,
        'outElems': out_elems,
        'inBytes': in_elems * elem_size,
        'outBytes': out_elems * elem_size,
        'instrs': {'mov': out_elems},
    }
    return


def typecast_sinf(iTList, oTList, op, **kwargs):
    """Shape inference for Typecast: elementwise dtype cast — output logical shape = input,
    layout preserved (unlike Untilize). Output dtype is set on the tensor at emit time; don't
    overwrite it here. Perf: one elementwise pass (mov = elements)."""
    assert len(iTList) == 1 and len(oTList) == 1
    X = iTList[0]
    in_shape = require_shape_list(
        X.shape,
        "Typecast shape inference: input tensor shape must be known for element/byte accounting",
    )
    oTList[0].shape = in_shape
    elem_size = op.attrs.get('element_size', 2)
    n = _nelems(in_shape)
    op.perf_stats = {
        'inElems': n,
        'outElems': n,
        'inBytes': n * elem_size,
        'outBytes': n * elem_size,
        'instrs': {'mov': n},
    }
    return


def tilize_with_val_padding_sinf(iTList, oTList, op, **kwargs):
    """Shape inference for TilizeWithValPadding: 1 input, output padded from attrs."""
    assert len(iTList) == 1 and len(oTList) == 1
    X = iTList[0]
    in_shape = require_shape_list(
        X.shape,
        "TilizeWithValPadding shape inference: input tensor shape must be known for element/byte accounting",
    )

    # Output padded shape comes from op attrs (caller may request larger tile-aligned shape); fallback = tile-round input.
    output_padded_shape = op.attrs.get('output_padded_shape', _pad_to_tile_shape(in_shape))
    out_padded = list(output_padded_shape)
    # Logical shape defaults to input; tile reshape may set output_logical_shape (row-major view before tilize).
    out_logical = list(op.attrs.get('output_logical_shape', in_shape))

    oTList[0].shape = out_logical
    oTList[0].dtype = X.dtype

    # perf_stats: in_elems from input shape, out_elems from padded shape (includes padding); mov = elements written out.
    elem_size = op.attrs.get('element_size', 2)
    in_elems = _nelems(in_shape)
    out_elems = _nelems(out_padded)
    op.perf_stats = {
        'inElems': in_elems,
        'outElems': out_elems,
        'inBytes': in_elems * elem_size,
        'outBytes': out_elems * elem_size,
        'instrs': {'mov': out_elems},
    }
    return


def untilize_with_unpadding_sinf(iTList, oTList, op, **kwargs):
    """Shape inference for UntilizeWithUnpadding: 1 input, output shape from attrs."""
    assert len(iTList) == 1 and len(oTList) == 1
    X = iTList[0]
    in_shape = require_shape_list(
        X.shape,
        "UntilizeWithUnpadding shape inference: input tensor shape must be known for element/byte accounting",
    )

    # Output logical shape is given in attrs (unpadded/cropped size); input may be tile-padded, output is smaller.
    output_shape = op.attrs.get('output_shape', in_shape)
    out_shape = list(output_shape)

    oTList[0].shape = out_shape
    oTList[0].dtype = X.dtype

    # perf_stats: input from TILE storage (padded) when available; out_elems from requested output shape.
    elem_size = op.attrs.get('element_size', 2)
    in_storage = _input_storage_shape_for_perf(X, op, in_shape)
    in_elems = _nelems(in_storage)
    out_elems = _nelems(out_shape)
    op.perf_stats = {
        'inElems': in_elems,
        'outElems': out_elems,
        'inBytes': in_elems * elem_size,
        'outBytes': out_elems * elem_size,
        'instrs': {'mov': out_elems},
    }
    return


def interleaved_to_sharded_sinf(iTList, oTList, op, **kwargs):
    """Shape inference for InterleavedToSharded: 1->1, same logical shape (memory layout change only)."""
    assert len(iTList) == 1 and len(oTList) == 1
    X = iTList[0]
    in_shape = require_shape_list(
        X.shape,
        "InterleavedToSharded shape inference: input tensor shape must be known",
    )
    oTList[0].shape = list(in_shape)
    oTList[0].dtype = X.dtype
    oTList[0].hw_shape = getattr(X, 'hw_shape', None)  # propagate NHWC-flattened hw shape

    elem_size = op.attrs.get('element_size', 2)
    elems = _nelems(in_shape)
    op.perf_stats = {
        'inElems': elems,
        'outElems': elems,
        'inBytes': elems * elem_size,
        'outBytes': elems * elem_size,
        'instrs': {'mov': elems},
    }
    return


def sharded_to_interleaved_sinf(iTList, oTList, op, **kwargs):
    """Shape inference for ShardedToInterleaved: 1->1, same logical shape (memory layout change only)."""
    assert len(iTList) == 1 and len(oTList) == 1
    X = iTList[0]
    in_shape = require_shape_list(
        X.shape,
        "ShardedToInterleaved shape inference: input tensor shape must be known",
    )
    oTList[0].shape = list(in_shape)
    oTList[0].dtype = X.dtype
    oTList[0].hw_shape = getattr(X, 'hw_shape', None)  # propagate NHWC-flattened hw shape

    elem_size = op.attrs.get('element_size', 2)
    elems = _nelems(in_shape)
    op.perf_stats = {
        'inElems': elems,
        'outElems': elems,
        'inBytes': elems * elem_size,
        'outBytes': elems * elem_size,
        'instrs': {'mov': elems},
    }
    return


def reshard_sinf(iTList, oTList, op, **kwargs):
    """Shape inference for Reshard: 1->1, same logical shape (shard-to-shard re-layout)."""
    assert len(iTList) == 1 and len(oTList) == 1
    X = iTList[0]
    in_shape = require_shape_list(
        X.shape,
        "Reshard shape inference: input tensor shape must be known",
    )
    oTList[0].shape = list(in_shape)
    oTList[0].dtype = X.dtype
    oTList[0].hw_shape = getattr(X, 'hw_shape', None)  # propagate NHWC-flattened hw shape

    elem_size = op.attrs.get('element_size', 2)
    elems = _nelems(in_shape)
    op.perf_stats = {
        'inElems': elems,
        'outElems': elems,
        'inBytes': elems * elem_size,
        'outBytes': elems * elem_size,
        'instrs': {'mov': elems},
    }
    return


def nlp_concat_heads_sinf(iTList, oTList, op, **kwargs):
    """Shape inference for ConcatHeads: [B, num_heads, S, head_dim] -> [B, S, num_heads*head_dim]."""
    assert len(iTList) == 1 and len(oTList) == 1
    X = iTList[0]
    in_shape = require_shape_list(
        X.shape,
        "ConcatHeads shape inference: input tensor shape must be known",
    )
    assert len(in_shape) == 4, f"ConcatHeads expects rank-4 input, got rank {len(in_shape)}"
    B, num_heads, S, head_dim = in_shape
    out_shape = [B, S, num_heads * head_dim]

    oTList[0].shape = out_shape
    oTList[0].dtype = X.dtype

    elem_size = op.attrs.get('element_size', 2)
    elems = _nelems(in_shape)
    op.perf_stats = {
        'inElems': elems,
        'outElems': elems,
        'inBytes': elems * elem_size,
        'outBytes': elems * elem_size,
        'instrs': {'mov': elems},
    }
    return


def nlp_create_qkv_heads_sinf(iTList, oTList, op, **kwargs):
    """Shape inference for CreateQKVHeads: 1-2 inputs -> 3 outputs (Q, K, V).

    Input: [B, S, (num_heads + 2*num_kv_heads) * head_dim] (fused QKV)
    Optional second input: [B, S, 2*num_kv_heads * head_dim] (separate KV)
    Outputs: Q=[B, num_heads, S, head_dim],
             K=[B, num_kv_heads, head_dim, S] if transpose_k_heads else [B, num_kv_heads, S, head_dim],
             V=[B, num_kv_heads, S, head_dim]
    """
    assert 1 <= len(iTList) <= 2 and len(oTList) == 3
    X = iTList[0]
    in_shape = require_shape_list(
        X.shape,
        "CreateQKVHeads shape inference: input tensor shape must be known",
    )

    num_heads = op.attrs.get('num_heads', 1)
    num_kv_heads = op.attrs.get('num_kv_heads', num_heads)
    head_dim = op.attrs.get('head_dim', None)
    # HW returns K pre-transposed when this attr is set; propagate to shape.
    transpose_k = op.attrs.get('transpose_k_heads', False)

    if len(iTList) == 2:
        B = in_shape[0] if len(in_shape) >= 3 else 1
        S = in_shape[-2] if len(in_shape) >= 2 else in_shape[0]
        if head_dim is None:
            head_dim = in_shape[-1] // num_heads
    else:
        B = in_shape[0] if len(in_shape) >= 3 else 1
        S = in_shape[-2] if len(in_shape) >= 2 else in_shape[0]
        if head_dim is None:
            head_dim = in_shape[-1] // (num_heads + 2 * num_kv_heads)

    q_shape = [B, num_heads, S, head_dim]
    k_shape = [B, num_kv_heads, head_dim, S] if transpose_k else [B, num_kv_heads, S, head_dim]
    v_shape = [B, num_kv_heads, S, head_dim]

    oTList[0].shape = q_shape
    oTList[0].dtype = X.dtype
    oTList[1].shape = k_shape
    oTList[1].dtype = X.dtype
    oTList[2].shape = v_shape
    oTList[2].dtype = X.dtype

    elem_size = op.attrs.get('element_size', 2)
    in_elems = sum(_nelems(t.shape) for t in iTList if t.shape)
    out_elems = _nelems(q_shape) + _nelems(k_shape) + _nelems(v_shape)
    op.perf_stats = {
        'inElems': in_elems,
        'outElems': out_elems,
        'inBytes': in_elems * elem_size,
        'outBytes': out_elems * elem_size,
        'instrs': {'mov': out_elems},
    }
    return


def _halo_ext_y(in_shape: list[int], attrs: dict) -> 'int | None':
    """Look up the halo-extended Y dimension from _HALO_EXT_Y.

    ``in_shape`` is the NCHW logical shape [N, C, H, W] of the Halo input.
    ``attrs`` contains 'kernel_size', 'padding', and 'is_transpose' set by
    ``_with_halo`` in ``ttsim/front/ttnn/op.py``.

    Returns the extended Y (= max_out_nsticks_per_core × num_cores_nhw) or
    None when the combination is not in the table (e.g. MaxPool 2×2).
    """
    if len(in_shape) < 4:
        return None
    N, C, H, W = in_shape[0], in_shape[1], in_shape[2], in_shape[3]
    nhw = N * H * W

    ks = attrs.get('kernel_size')
    pad = attrs.get('padding')
    is_tp = bool(attrs.get('is_transpose', False))
    if ks is None or pad is None:
        return None

    kH = int(ks[0]) if hasattr(ks, '__getitem__') else int(ks)
    kW = int(ks[1]) if hasattr(ks, '__getitem__') else int(ks)
    pH = int(pad[0]) if hasattr(pad, '__getitem__') else int(pad)
    pW = int(pad[1]) if hasattr(pad, '__getitem__') else int(pad)

    return _HALO_EXT_Y.get((nhw, C, kH, kW, pH, pW, is_tp))


def halo_sinf(iTList, oTList, op, **kwargs):
    """Shape inference for Halo: logical shape passthrough with halo-extended hw_shape.

    The logical tensor shape is unchanged (halo extraction is transparent to
    the compute graph).  The hardware physical buffer grows by shard-border
    overlap rows; ``hw_shape`` is set to [1, 1, halo_ext_y, C] so that
    downstream Conv/Move LUT keys match the hardware profiler's recorded shapes.

    When ``op.attrs`` carries 'kernel_size', 'padding', and 'is_transpose'
    (set by ``_with_halo`` in ``op.py``), the extended Y is looked up in
    ``_HALO_EXT_Y``.  If the combination is absent (e.g. MaxPool 2×2 pass-through),
    the pre-halo hw_shape is propagated unchanged.
    """
    assert len(iTList) == 1 and len(oTList) == 1
    X = iTList[0]
    in_shape = require_shape_list(
        X.shape,
        "Halo shape inference: input tensor shape must be known",
    )
    oTList[0].shape = list(in_shape)
    oTList[0].dtype = X.dtype

    attrs = getattr(op, 'attrs', None) or {}
    elem_size = op.attrs.get('element_size', 2)
    elems = _nelems(in_shape)  # logical NCHW element count (inElems)

    ext_y = _halo_ext_y(in_shape, attrs)
    if ext_y is not None:
        # Halo extracts beyond the logical NCHW input — the output buffer
        # carries the halo-extended physical layout [1, 1, ext_y, C].
        # outElems must reflect this physical buffer so memory-traffic / util
        # estimates from analytical fallback are consistent with hw_shape.
        C = in_shape[1]  # NCHW channel dim
        oTList[0].hw_shape = [1, 1, ext_y, C]
        out_elems = ext_y * C
    else:
        oTList[0].hw_shape = getattr(X, 'hw_shape', None)
        out_elems = elems  # logical passthrough

    op.perf_stats = {
        'inElems': elems,
        'outElems': out_elems,
        'inBytes': elems * elem_size,
        'outBytes': out_elems * elem_size,
        'instrs': {'mov': out_elems},
    }
    return


def move_sinf(iTList, oTList, op, **kwargs):
    """Shape inference for Move: logical shape passthrough.

    MoveDeviceOperation copies the activation buffer to a new memory region
    (triggered by deallocate_activation=True in conv args).  The logical
    tensor shape and dtype are unchanged; cost is two memory transfers.
    """
    assert len(iTList) == 1 and len(oTList) == 1
    X = iTList[0]
    in_shape = require_shape_list(
        X.shape,
        'Move shape inference: input tensor shape must be known',
    )
    oTList[0].shape = list(in_shape)
    oTList[0].dtype = X.dtype
    oTList[0].hw_shape = getattr(X, 'hw_shape', None)  # propagate NHWC-flattened hw shape

    elem_size = op.attrs.get('element_size', 2)
    elems = _nelems(in_shape)
    op.perf_stats = {
        'inElems': elems,
        'outElems': elems,
        'inBytes': elems * elem_size,
        'outBytes': elems * elem_size,
        'instrs': {'mov': elems},
    }
    return


def _passthrough_perf(in_shapes, out_shape, elem_size):
    """Build a passthrough perf_stats dict: 1 mov per output element."""
    in_elems = sum(_nelems(s) for s in in_shapes if s)
    out_elems = _nelems(out_shape)
    return {
        'inElems': in_elems,
        'outElems': out_elems,
        'inBytes': in_elems * elem_size,
        'outBytes': out_elems * elem_size,
        'instrs': {'mov': out_elems},
    }


def rotary_embedding_llama_sinf(iTList, oTList, op, **kwargs):
    """Shape inference for RotaryEmbeddingLlama (prefill): output = in0 (x) shape.

    Inputs: x, cos, sin, trans_mat (rotary applied in-place on x's layout).
    """
    assert 1 <= len(iTList) <= 4 and len(oTList) == 1
    X = iTList[0]
    in_shape = require_shape_list(
        X.shape,
        "RotaryEmbeddingLlama shape inference: input tensor shape must be known",
    )
    oTList[0].shape = list(in_shape)
    oTList[0].dtype = X.dtype

    elem_size = op.attrs.get('element_size', 2)
    op.perf_stats = _passthrough_perf([t.shape for t in iTList], in_shape, elem_size)
    return


def plus_one_sinf(iTList, oTList, op, **kwargs):
    """Shape inference for PlusOne (in-place position-counter increment): output = input shape."""
    assert len(iTList) == 1 and len(oTList) == 1
    X = iTList[0]
    in_shape = require_shape_list(
        X.shape, "PlusOne shape inference: input tensor shape must be known",
    )
    oTList[0].shape = list(in_shape)
    oTList[0].dtype = X.dtype
    elem_size = op.attrs.get('element_size', 2)
    op.perf_stats = _passthrough_perf([t.shape for t in iTList], in_shape, elem_size)
    return


def manual_seed_sinf(iTList, oTList, op, **kwargs):
    """Shape inference for ManualSeed: output = seeds (in0) shape."""
    assert 1 <= len(iTList) <= 2 and len(oTList) == 1
    X = iTList[0]
    in_shape = require_shape_list(X.shape, "ManualSeed shape inference: seeds shape must be known")
    oTList[0].shape = list(in_shape)
    oTList[0].dtype = X.dtype
    elem_size = op.attrs.get('element_size', 2)
    op.perf_stats = _passthrough_perf([t.shape for t in iTList], in_shape, elem_size)
    return


def sampling_sinf(iTList, oTList, op, **kwargs):
    """Shape inference for Sampling: one sampled index per user (topk_values last dim -> 1).

    Inputs: topk_values, topk_indices, [k, p, temp, output_tensor].
    output_tensor (input_5) is HW's preallocated output operand (arity-6 on the profiler).
    """
    assert 2 <= len(iTList) <= 6 and len(oTList) == 1
    V = iTList[0]
    in_shape = require_shape_list(V.shape, "Sampling shape inference: topk_values shape must be known")
    out_shape = list(in_shape[:-1]) + [1]
    oTList[0].shape = out_shape
    oTList[0].dtype = iTList[1].dtype
    elem_size = op.attrs.get('element_size', 2)
    op.perf_stats = _passthrough_perf([t.shape for t in iTList], out_shape, elem_size)
    return


def rotary_embedding_llama_fused_qk_sinf(iTList, oTList, op, **kwargs):
    """Shape inference for RotaryEmbeddingLlamaFusedQK (decode): 2 outputs (q, k).

    Inputs: q, k, cos, sin, trans_mat. out0 = q shape, out1 = k shape.
    """
    assert 2 <= len(iTList) <= 5 and len(oTList) == 2
    Q = iTList[0]
    K = iTList[1]
    q_shape = require_shape_list(
        Q.shape, "RotaryEmbeddingLlamaFusedQK shape inference: q shape must be known",
    )
    k_shape = require_shape_list(
        K.shape, "RotaryEmbeddingLlamaFusedQK shape inference: k shape must be known",
    )
    oTList[0].shape = list(q_shape)
    oTList[0].dtype = Q.dtype
    oTList[1].shape = list(k_shape)
    oTList[1].dtype = K.dtype

    elem_size = op.attrs.get('element_size', 2)
    in_elems = sum(_nelems(t.shape) for t in iTList if t.shape)
    out_elems = _nelems(q_shape) + _nelems(k_shape)
    op.perf_stats = {
        'inElems': in_elems,
        'outElems': out_elems,
        'inBytes': in_elems * elem_size,
        'outBytes': out_elems * elem_size,
        'instrs': {'mov': out_elems},
    }
    return


def paged_fill_cache_sinf(iTList, oTList, op, **kwargs):
    """Shape inference for PagedFillCache (prefill): output = in0 (cache) shape.

    Inputs: cache, input (k or v), page_table. The cache is filled in place;
    the op returns the cache tensor unchanged in shape.
    """
    # PagedFillCache requires (cache, input) at minimum; page_table is optional.
    assert 2 <= len(iTList) <= 3 and len(oTList) == 1
    cache = iTList[0]
    cache_shape = require_shape_list(
        cache.shape, "PagedFillCache shape inference: cache shape must be known",
    )
    oTList[0].shape = list(cache_shape)
    oTList[0].dtype = cache.dtype

    elem_size = op.attrs.get('element_size', 2)
    op.perf_stats = _passthrough_perf([t.shape for t in iTList], cache_shape, elem_size)
    return


def paged_fused_update_cache_sinf(iTList, oTList, op, **kwargs):
    """Shape inference for PagedFusedUpdateCache (decode): 2 outputs (k_cache, v_cache).

    Inputs: k_cache, k, v_cache, v, update_idxs_tensor, page_table.
    out0 = k_cache (in0) shape, out1 = v_cache (in2) shape; updated in place.
    """
    # Op semantics require (k_cache, k, v_cache, v) at minimum; update_idxs/page_table optional.
    assert 4 <= len(iTList) <= 6 and len(oTList) == 2
    k_cache = iTList[0]
    v_cache = iTList[2]
    k_cache_shape = require_shape_list(
        k_cache.shape, "PagedFusedUpdateCache shape inference: k_cache shape must be known",
    )
    v_cache_shape = require_shape_list(
        v_cache.shape, "PagedFusedUpdateCache shape inference: v_cache shape must be known",
    )
    oTList[0].shape = list(k_cache_shape)
    oTList[0].dtype = k_cache.dtype
    oTList[1].shape = list(v_cache_shape)
    oTList[1].dtype = v_cache.dtype

    elem_size = op.attrs.get('element_size', 2)
    in_elems = sum(_nelems(t.shape) for t in iTList if t.shape)
    out_elems = _nelems(k_cache_shape) + _nelems(v_cache_shape)
    op.perf_stats = {
        'inElems': in_elems,
        'outElems': out_elems,
        'inBytes': in_elems * elem_size,
        'outBytes': out_elems * elem_size,
        'instrs': {'mov': out_elems},
    }
    return


def nlp_create_qkv_heads_decode_sinf(iTList, oTList, op, **kwargs):
    """Shape inference for NLPCreateQKVHeadsDecode: 1 input -> 3 outputs (Q, K, V).

    Decode WZYX convention: input [1, 1, B, (num_q + 2*num_kv) * head_dim],
    outputs put heads on the Y axis and head_dim on X:
      Q = [1, 1, num_q_heads, head_dim]
      K = [1, 1, num_kv_heads, head_dim]
      V = [1, 1, num_kv_heads, head_dim]
    """
    assert len(iTList) == 1 and len(oTList) == 3
    X = iTList[0]
    in_shape = require_shape_list(
        X.shape, "NLPCreateQKVHeadsDecode shape inference: input shape must be known",
    )

    num_q_heads = op.attrs.get('num_heads', 1)
    num_kv_heads = op.attrs.get('num_kv_heads', num_q_heads)
    head_dim = op.attrs.get('head_dim', None)
    hidden = in_shape[-1]
    if head_dim is None:
        total_heads = num_q_heads + 2 * num_kv_heads
        if hidden % total_heads != 0:
            # Mirror nlp_create_qkv_heads_decode_op, which raises for this case.
            raise ValueError(
                f"NLPCreateQKVHeadsDecode shape inference: hidden {hidden} not divisible by "
                f"num_heads + 2*num_kv_heads ({total_heads})"
            )
        head_dim = hidden // total_heads

    lead = list(in_shape[:-2]) if len(in_shape) >= 2 else [1, 1]
    while len(lead) < 2:
        lead = [1] + lead
    q_shape = lead + [num_q_heads, head_dim]
    k_shape = lead + [num_kv_heads, head_dim]
    v_shape = lead + [num_kv_heads, head_dim]

    oTList[0].shape = q_shape
    oTList[0].dtype = X.dtype
    oTList[1].shape = k_shape
    oTList[1].dtype = X.dtype
    oTList[2].shape = v_shape
    oTList[2].dtype = X.dtype

    elem_size = op.attrs.get('element_size', 2)
    in_elems = _nelems(in_shape)
    out_elems = _nelems(q_shape) + _nelems(k_shape) + _nelems(v_shape)
    op.perf_stats = {
        'inElems': in_elems,
        'outElems': out_elems,
        'inBytes': in_elems * elem_size,
        'outBytes': out_elems * elem_size,
        'instrs': {'mov': out_elems},
    }
    return


def nlp_concat_heads_decode_sinf(iTList, oTList, op, **kwargs):
    """Shape inference for NLPConcatHeadsDecode: concatenate heads on the X axis.

    Decode WZYX convention: input [1, 1, num_heads, head_dim] (heads on Y),
    output [1, 1, num_heads, num_heads * head_dim] -- the input Y dim (in_shape[-2])
    is kept and head_dim is folded into X. HW keeps the Y dim (batch placeholder) and
    grows X = num_heads * head_dim.
    """
    assert len(iTList) == 1 and len(oTList) == 1
    X = iTList[0]
    in_shape = require_shape_list(
        X.shape, "NLPConcatHeadsDecode shape inference: input shape must be known",
    )
    assert len(in_shape) >= 2, f"NLPConcatHeadsDecode expects rank>=2 input, got {len(in_shape)}"
    num_heads = op.attrs.get('num_heads', in_shape[-2])
    head_dim = in_shape[-1]
    out_shape = list(in_shape[:-1]) + [num_heads * head_dim]

    oTList[0].shape = out_shape
    oTList[0].dtype = X.dtype

    elem_size = op.attrs.get('element_size', 2)
    op.perf_stats = _passthrough_perf([in_shape], out_shape, elem_size)
    return


def _sdpa_known_shape(t, what):
    """Roofline handlers fail fast with ValueError (the type sdpa_sinf catches) on unknown shapes."""
    if t.shape is None:
        raise ValueError(f"SDPA roofline: {what} shape must be known")
    return require_shape_list(t.shape)


def _infer_sdpa_variant(iTList, attrs):
    """Classify an SDPA op into a variant: explicit sdpa_variant tag (every ttnn entry point sets one),
    else by arity/attrs for untagged ops (3 inputs = prefill; chunk_start_idx>0 = chunked; else decode)."""
    v = attrs.get('sdpa_variant')
    if v:
        return v
    if len(iTList) == 3:
        return 'prefill'
    if int(attrs.get('chunk_start_idx') or 0) > 0 or attrs.get('is_chunked_prefill'):
        return 'chunked'
    return 'decode'


def _sdpa_prefill_perf(iTList, q_shape, op):
    """Prefill / MLA / cross / windowed / masked SDPA -> compute roofline."""
    from ttsim.perf.roofline_sdpa import sdpa_config_from_shapes, sdpa_perf_stats
    k_shape = _sdpa_known_shape(iTList[1], "k")
    v_shape = _sdpa_known_shape(iTList[2], "v")
    # Chunked prefill carries the page table as its fourth input; its width bounds the KV length.
    pt_shape = None
    if op.attrs.get('paged') and len(iTList) >= 4 and iTList[3].shape is not None:
        pt_shape = require_shape_list(iTList[3].shape)
    cfg = sdpa_config_from_shapes(q_shape, k_shape, v_shape, op.attrs, page_table_shape=pt_shape)
    op.perf_stats = sdpa_perf_stats(cfg)


def _sdpa_decode_perf(iTList, q_shape, op):
    """Single-token / paged decode SDPA (incl. MLA decode) -> memory-bound KV-stream roofline
    (paged only scatters the same bytes; MLA reuses K as V so no separate V read)."""
    from ttsim.perf.roofline_sdpa import decode_perf_stats
    k_shape = _sdpa_known_shape(iTList[1], "k_cache")
    v_shape = _sdpa_known_shape(iTList[2], "v_cache")
    # Paged: the page table is the last input and its width is the per-user capacity in blocks.
    pt_shape = None
    if op.attrs.get('paged') and len(iTList) >= 4 and iTList[-1].shape is not None:
        pt_shape = require_shape_list(iTList[-1].shape)
    op.perf_stats = decode_perf_stats(q_shape, k_shape, v_shape=v_shape, attrs=op.attrs,
                                      page_table_shape=pt_shape)


def _sdpa_joint_perf(iTList, q_shape, op):
    """Joint SDPA (SD3/Flux): main + joint token streams attend over their concatenation, non-causal.
    Model as one non-causal prefill over S_eff = main_seq + joint_seq."""
    from ttsim.perf.roofline_sdpa import sdpa_config_from_shapes, sdpa_perf_stats
    k_shape = _sdpa_known_shape(iTList[1], "joint k")
    v_shape = _sdpa_known_shape(iTList[2], "joint v")
    joint_seq = int(op.attrs.get("joint_seq") or 0)
    q, k, v = (list(map(int, s)) for s in (q_shape, k_shape, v_shape))
    seff = q[-2] + joint_seq
    qj, kj, vj = q[:-2] + [seff, q[-1]], k[:-2] + [seff, k[-1]], v[:-2] + [seff, v[-1]]
    attrs = dict(op.attrs); attrs["is_causal"] = False; attrs["is_joint"] = True
    op.perf_stats = sdpa_perf_stats(sdpa_config_from_shapes(qj, kj, vj, attrs))


# Single-chip only. Chunked prefill reuses the compute path; multi-chip ring is out of scope
# (falls to passthrough). Unhandled variants / errors also fall to passthrough.
_SDPA_FRONTENDS = {
    'prefill': _sdpa_prefill_perf,
    'decode': _sdpa_decode_perf,
    'chunked': _sdpa_prefill_perf,
    'mla': _sdpa_prefill_perf,           # FlashMLA prefill (asymmetric v_head_dim via head_dim_v attr)
    'sparse': _sdpa_prefill_perf,        # sparse-MLA (is_sparse + kv_seq=TOPK attrs)
    'mla_decode': _sdpa_decode_perf,     # FlashMLA decode (memory-bound, reuses K as V)
    'joint': _sdpa_joint_perf,           # SD3/Flux joint attention (concatenated streams)
}


def sdpa_sinf(iTList, oTList, op, **kwargs):
    """Shape inference + per-variant cost routing for SDPA. Output shape = q; cost via _SDPA_FRONTENDS,
    unhandled variants use the passthrough estimate."""
    # Op-table arity ARITY_VARIADIC[3-7]: q, k, v plus the optional mask, sink, window, position and
    # page table tensors the entry points pass through so their producers stay connected in the graph.
    assert 3 <= len(iTList) <= 7 and len(oTList) == 1
    Q = iTList[0]
    q_shape = require_shape_list(
        Q.shape, "ScaledDotProductAttention shape inference: q shape must be known",
    )
    oTList[0].shape = list(q_shape)
    oTList[0].dtype = Q.dtype

    elem_size = op.attrs.get('element_size', 2)
    variant = _infer_sdpa_variant(iTList, op.attrs)
    handler = _SDPA_FRONTENDS.get(variant)
    if handler is not None:
        try:
            handler(iTList, q_shape, op)
            # Keep a device-agnostic instr count so non-calibrated devices still get an estimate.
            if not op.perf_stats.get('instrs'):
                op.perf_stats['instrs'] = {'mov': _nelems(q_shape)}
            return
        except ValueError as e:
            # The roofline fails fast with ValueError on unsupported shapes/attrs (a thread_split arch
            # without constants too) -> passthrough; anything else propagates. Warn once.
            from loguru import logger
            logger.warning(f"SDPA roofline ({variant}) unavailable for {getattr(op, 'name', '?')}, "
                           f"using passthrough estimate: {e}", once=True)
    op.perf_stats = _passthrough_perf([t.shape for t in iTList], q_shape, elem_size)
    return


def register_layout_ops():
    d = _TTNN_OP_DOMAIN
    _optbl = [
        ['Tilize', 'ARITY_1->1', d, 'COMMON', 24, 21, 1, 1, 1, 1, tilize_sinf, True, True, True, True, True],
        ['Untilize', 'ARITY_1->1', d, 'COMMON', 24, 21, 1, 1, 1, 1, untilize_sinf, True, True, True, True, True],
        ['Typecast', 'ARITY_1->1', d, 'COMMON', 24, 21, 1, 1, 1, 1, typecast_sinf, True, True, True, True, True],
        ['TilizeWithValPadding', 'ARITY_1->1', d, 'COMMON', 24, 21, 1, 1, 1, 1, tilize_with_val_padding_sinf, True, True, True, True, True],
        ['UntilizeWithUnpadding', 'ARITY_1->1', d, 'COMMON', 24, 21, 1, 1, 1, 1, untilize_with_unpadding_sinf, True, True, True, True, True],
        ['InterleavedToSharded', 'ARITY_1->1', d, 'COMMON', 24, 21, 1, 1, 1, 1, interleaved_to_sharded_sinf, True, True, True, True, True],
        ['ShardedToInterleaved', 'ARITY_1->1', d, 'COMMON', 24, 21, 1, 1, 1, 1, sharded_to_interleaved_sinf, True, True, True, True, True],
        ['Reshard', 'ARITY_1->1', d, 'COMMON', 24, 21, 1, 1, 1, 1, reshard_sinf, True, True, True, True, True],
        ['Halo', 'ARITY_1->1', d, 'COMMON', 24, 21, 1, 1, 1, 1, halo_sinf, True, True, True, True, True],
        ['Move', 'ARITY_1->1', d, 'COMMON', 24, 21, 1, 1, 1, 1, move_sinf, True, True, True, True, True],
        ['ConcatHeads', 'ARITY_1->1', d, 'COMMON', 24, 21, 1, 1, 1, 1, nlp_concat_heads_sinf, True, True, True, True, True],
        ['CreateQKVHeads', 'ARITY_VARIADIC[1-2]->3', d, 'COMMON', 24, 21, 2, 1, 3, 3, nlp_create_qkv_heads_sinf, True, True, True, True, True],
        ['RotaryEmbeddingLlama', 'ARITY_VARIADIC[1-4]->1', d, 'COMMON', 24, 21, 4, 1, 1, 1, rotary_embedding_llama_sinf, True, True, True, True, True],
        ['RotaryEmbeddingLlamaFusedQK', 'ARITY_VARIADIC[2-5]->2', d, 'COMMON', 24, 21, 5, 2, 2, 2, rotary_embedding_llama_fused_qk_sinf, True, True, True, True, True],
        ['PagedFillCache', 'ARITY_VARIADIC[2-3]->1', d, 'COMMON', 24, 21, 3, 2, 1, 1, paged_fill_cache_sinf, True, True, True, True, True],
        ['PagedFusedUpdateCache', 'ARITY_VARIADIC[4-6]->2', d, 'COMMON', 24, 21, 6, 4, 2, 2, paged_fused_update_cache_sinf, True, True, True, True, True],
        ['NLPCreateQKVHeadsDecode', 'ARITY_1->3', d, 'COMMON', 24, 21, 1, 1, 3, 3, nlp_create_qkv_heads_decode_sinf, True, True, True, True, True],
        ['NLPConcatHeadsDecode', 'ARITY_1->1', d, 'COMMON', 24, 21, 1, 1, 1, 1, nlp_concat_heads_decode_sinf, True, True, True, True, True],
        ['ScaledDotProductAttention', 'ARITY_VARIADIC[3-7]->1', d, 'COMMON', 24, 21, 7, 3, 1, 1, sdpa_sinf, True, True, True, True, True],
        ['PlusOne', 'ARITY_1->1', d, 'COMMON', 24, 21, 1, 1, 1, 1, plus_one_sinf, True, True, True, True, True],
        ['ManualSeed', 'ARITY_VARIADIC[1-2]->1', d, 'COMMON', 24, 21, 2, 1, 1, 1, manual_seed_sinf, True, True, True, True, True],
        # arity up to 6: HW SamplingDeviceOperation takes a preallocated output_tensor as input_5.
        ['Sampling', 'ARITY_VARIADIC[2-6]->1', d, 'COMMON', 24, 21, 6, 2, 1, 1, sampling_sinf, True, True, True, True, True],
    ]
    register_ops('layout', _optbl)
    return
