#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from loguru import logger

from ttsim.ops.desc.helpers import build_tmp_data_tensor, unary_fwd, update_output_tensor, multidirectional_broadcast_shape_inference, gridsample_sinf
from ttsim.ops.desc.registry import register_ops
from ttsim.ops.tensor import SimTensor
from ttsim.utils.common import prod_ints


def trilu_sinf(iTList, oTList, op, **kwargs):
    """
    Shape/perf inference for Trilu with dual behavior:

    - len(iTList) == 1: DLRM legacy path that flattens the strict upper
      triangle (i < j) into a feature vector (output shape != input shape).
    - len(iTList) == 2: generic ONNX Trilu(A, k) where the output tensor
      has the same shape and dtype as the input tensor A; k is ignored
      for shape/perf purposes.
    """

    if len(iTList) == 1:
        X = iTList[0].clone_by_shape()

        row_indices, col_indices = [], []
        batch_size, num_features1, num_features2 = X.shape
        assert (
            num_features1 == num_features2
        ), f"Input should be an batch of square matrices: {X.shape}"
        num_features = num_features1
        for i in range(num_features):
            for j in range(i + 1, num_features):
                row_indices.append(i)
                col_indices.append(j)
        tmp_data = X.data[:, row_indices, col_indices]
        tmp_outT = build_tmp_data_tensor(tmp_data, op.name + "__tmp_out__")
        update_output_tensor(op, tmp_outT, oTList[0])

        op.perf_stats = {
            "inElems": X.nelems(),
            "inBytes": X.nbytes(op.precision),
            "outElems": oTList[0].nelems(),
            "outBytes": oTList[0].nbytes(op.precision),
            "instrs": {"mov": int(oTList[0].nelems())},
        }
        return op.perf_stats

    X = iTList[0]
    Y = oTList[0]

    if not Y.check_shape():
        assert X.check_shape(), f"Input tensor shape not defined: {X}"
        Y.shape = list(X.shape)
    else:
        assert list(Y.shape) == list(X.shape), (
            f"Preallocated output shape mismatch in Trilu: "
            f"expected {X.shape}, got {Y.shape}"
        )

    Y.dtype = X.dtype

    inBytes = sum(t.nbytes(op.precision) for t in iTList)
    inElems = sum(t.nelems() for t in iTList)
    outBytes = Y.nbytes(op.precision)
    outElems = Y.nelems()

    op.perf_stats = {
        "inElems": int(inElems),
        "inBytes": int(inBytes),
        "outElems": int(outElems),
        "outBytes": int(outBytes),
        "instrs": {"mov": int(outElems)},
    }
    return op.perf_stats

def where_sinf(iTList, oTList, op, **kwargs):
    condT = iTList[0]
    XT    = iTList[1]
    YT    = iTList[2]
    outT  = oTList[0]

    assert condT.check_shape(), f"Illegal shape for cond in Where: {condT}"
    assert XT.check_shape(),    f"Illegal shape for X in Where: {XT}"
    assert YT.check_shape(),    f"Illegal shape for Y in Where: {YT}"

    if not (np.issubdtype(condT.dtype, np.bool_) or np.issubdtype(condT.dtype, np.number)):
        raise AssertionError(f"Where cond must be bool or numeric, got {condT.dtype}")

    assert XT.dtype == YT.dtype, f"Where X/Y dtypes must match: {XT.dtype} vs {YT.dtype}"
    out_shape = multidirectional_broadcast_shape_inference(
        [condT.shape, XT.shape, YT.shape]
    )

    outT.shape = out_shape
    outT.dtype = XT.dtype

    # Compute data if inputs have data
    from ttsim.ops.desc.data_compute import try_compute_data, compute_where
    outT.data = try_compute_data(compute_where, iTList, op)

    op.perf_stats = {
        'inElems' : condT.nelems() + XT.nelems() + YT.nelems(),
        'outElems': outT.nelems(),
        'inBytes' : condT.nbytes(op.precision)
                    + XT.nbytes(op.precision)
                    + YT.nbytes(op.precision),
        'outBytes': outT.nbytes(op.precision),
        'instrs'  : {'mov': outT.nelems(), 'cmp': outT.nelems()},
    }
    return

def cast_sinf(iTList, oTList, op, **kwargs):
    '''Cast operation: converts tensor to specified dtype'''
    assert iTList[0].check_shape()
    # Propagate shape from input to output (Cast always preserves shape).
    # Always trust the input's shape here rather than only filling in an
    # unset output shape: a previously-declared output shape may be a stale
    # placeholder (e.g. an unresolved ONNX dim coerced to [1] at load time),
    # which would otherwise silently disagree with the real computed data.
    if oTList[0].check_shape() and list(oTList[0].shape) != list(iTList[0].shape):
        logger.warning(
            "Overriding stale/placeholder SHAPE for SimTensor({}): declared={} -> computed={}",
            oTList[0].name, oTList[0].shape, iTList[0].shape,
        )
    oTList[0].shape = list(iTList[0].shape)

    # ONNX dtype code mapping to numpy dtypes
    ONNX_DTYPE_MAP = {
        1: np.dtype(np.float32),
        2: np.dtype(np.uint8),
        3: np.dtype(np.int8),
        4: np.dtype(np.uint16),
        5: np.dtype(np.int16),
        6: np.dtype(np.int32),
        7: np.dtype(np.int64),
        9: np.dtype(np.bool_),
        10: np.dtype(np.float16),
        11: np.dtype(np.float64),
        12: np.dtype(np.uint32),
        13: np.dtype(np.uint64),
        16: np.dtype(np.float16),  # Placeholder for bfloat16; adjust as needed
    }
    # String dtype name mapping (case-insensitive)
    STRING_DTYPE_MAP = {
        'float': np.dtype(np.float32),
        'float32': np.dtype(np.float32),
        'float16': np.dtype(np.float16),
        'float64': np.dtype(np.float64),
        'int': np.dtype(np.int32),
        'int32': np.dtype(np.int32),
        'int64': np.dtype(np.int64),
        'int16': np.dtype(np.int16),
        'int8': np.dtype(np.int8),
        'uint8': np.dtype(np.uint8),
        'uint16': np.dtype(np.uint16),
        'uint32': np.dtype(np.uint32),
        'uint64': np.dtype(np.uint64),
        'bool': np.dtype(np.bool_),
    }

    # Set output dtype based on 'to' attribute
    to_dtype_code = op.attrs.get('to')
    if to_dtype_code is not None:
        if isinstance(to_dtype_code, str):
            key = to_dtype_code.lower()
            if key not in STRING_DTYPE_MAP:
                raise ValueError(f"Unsupported 'to' dtype string: {to_dtype_code}")
            oTList[0].dtype = STRING_DTYPE_MAP[key]
        else:
            try:
                oTList[0].dtype = ONNX_DTYPE_MAP[to_dtype_code]
            except KeyError:
                raise ValueError(f"Unsupported 'to' dtype code: {to_dtype_code}")
    else:
        # Fallback: preserve input dtype if 'to' not specified
        oTList[0].dtype = iTList[0].dtype

    # Compute actual data if input has data -- needed by downstream ops
    # (Concat/Reshape/Slice etc.) that require concrete values, not just shape.
    if iTList[0].data is not None:
        oTList[0].data = np.asarray(iTList[0].data).astype(oTList[0].dtype)

    op.perf_stats = {
            'inBytes' : iTList[0].nbytes(op.precision),
            'outBytes': oTList[0].nbytes(op.precision),
            'inElems' : iTList[0].nelems(),
            'outElems': oTList[0].nelems(),
            'instrs'  : {'mov': oTList[0].nelems()}
            }
    return

def isnan_sinf(iTList, oTList, op, **kwargs):
    X = iTList[0]
    Y = oTList[0]
    if Y.check_shape():
        pass
    else:
        assert X.check_shape(), f"Input tensor shape not defined: {X}"
        Y.shape = list(X.shape)
    Y.dtype = X.dtype

    n = Y.nelems()
    op.perf_stats = {
        "inElems": X.nelems(),
        "outElems": n,
        "inBytes": X.nbytes(op.precision),
        "outBytes": Y.nbytes(op.precision),
        "instrs": {"cmp": n},
    }
    return

def pad_sinf(iTList, oTList, op, **kwargs):
    X = iTList[0]
    pad_tensor = iTList[1]
    if pad_tensor.data is not None:
        pads = [int(x) for x in pad_tensor.data.flatten().tolist()]
    else:
        # Fallback: pad amount tensor has no concrete data (e.g. depends on an
        # unresolved dynamic dim upstream) -- default to no padding rather
        # than hard-failing.
        rank_guess = X.rank()
        logger.warning(
            "Pad amount tensor has no concrete data for SimTensor({}) -- defaulting to zero padding (rank={})",
            pad_tensor.name, rank_guess,
        )
        pads = [0] * (2 * rank_guess)

    rank = X.rank()
    assert rank >= 2, "Pad expects at least 2D input"
    if len(pads) == 4:
        h_beg, w_beg, h_end, w_end = pads
        pad_before = [0] * (rank - 2) + [h_beg, w_beg]
        pad_after  = [0] * (rank - 2) + [h_end, w_end]
    elif len(pads) == 2 * rank:
        pad_before = list(pads[:rank])
        pad_after  = list(pads[rank:])
    else:
        raise AssertionError(
            f"pads length {len(pads)} != 4 or 2*rank (2*{rank})"
        )

    output_shape = [
        int(X.shape[i]) + pad_before[i] + pad_after[i]
        for i in range(rank)
    ]

    oTList[0].shape = output_shape
    oTList[0].dtype = X.dtype

    # Compute data if available
    from ttsim.ops.desc.data_compute import try_compute_data, compute_pad
    oTList[0].data = try_compute_data(compute_pad, iTList, op)

    nElem_in = X.nelems()
    nElem_out = int(np.prod(output_shape))

    op.perf_stats = {
        'inElems': nElem_in + pad_tensor.nelems(),
        'inBytes': X.nbytes(op.precision) + pad_tensor.nbytes(op.precision),
        'outElems': nElem_out,
        'outBytes': oTList[0].nbytes(op.precision),
        'instrs': {'mov': nElem_out},
    }
    return

def tile_sinf(iTList, oTList, op, **kwargs):
    assert iTList[0].check_shape(), f"Illegal Shape for {iTList[0]}"
    dataT    = iTList[0]
    repeatsT = iTList[1].clone_by_shape(data_maybe_missing=False)
    # repeatsT can pick up a spurious extra leading/trailing size-1 dim from the
    # same scalar-as-[1]/double-wrapping artifact handled elsewhere (see
    # _squeeze_to_rank1) -- e.g. a correctly-computed Gather now returning
    # [128] wrapped as [[128]]. Flatten to rank-1 so repeats.data[i] is a plain
    # scalar rather than a length-1 array.
    repeats = np.asarray(repeatsT.data).reshape(-1)
    assert len(repeats) == dataT.rank(), \
            f"repeats={repeats} should have same length as input shape={dataT.shape}"

    checkshape = [d > 0 for d in repeats]
    assert all(checkshape), f"repeats={repeats} should be > 0"

    outshape   = [dim * repeats[i] for i,dim in enumerate(dataT.shape)]

    oTList[0].shape = outshape
    oTList[0].dtype = dataT.dtype

    # Compute data if available
    from ttsim.ops.desc.data_compute import try_compute_data, compute_tile
    oTList[0].data = try_compute_data(compute_tile, iTList, op)

    op.perf_stats = {
            'inBytes' : int(iTList[0].nbytes(op.precision)) + int(iTList[1].nbytes(op.precision)),
            'inElems' : int(iTList[0].nelems()) + int(iTList[1].nelems()),
            'outBytes': int(oTList[0].nbytes(op.precision)),
            'outElems': int(oTList[0].nelems()),
            'instrs'  : {'mov': int(oTList[0].nelems())}
            }
    return

def squeeze_sinf(iTList, oTList, op, **kwargs):
    assert iTList[0].check_shape(), f"Illegal Shape for {iTList[0]}"
    dataT = iTList[0]
    if len(iTList) == 1:
        # Squeeze with no axes: remove all dims of size 1
        outshape = [dim for dim in dataT.shape if dim != 1]
        oTList[0].shape = outshape
        oTList[0].dtype = dataT.dtype

        # Compute data if inputs have data
        from ttsim.ops.desc.data_compute import try_compute_data, compute_squeeze
        oTList[0].data = try_compute_data(compute_squeeze, iTList, op)

        op.perf_stats = {
                'inBytes' : int(iTList[0].nbytes(op.precision)),
                'outBytes': int(oTList[0].nbytes(op.precision)),
                'inElems' : int(iTList[0].nelems()),
                'outElems': int(oTList[0].nelems()),
                'instrs'  : {'mov': int(oTList[0].nelems())}
                }
        return

    axesT = iTList[1].clone_by_shape(data_maybe_missing=False) #Y.data must be present

    data_rank  = dataT.rank()
    data_idx   = [ d + data_rank if d < 0 else d for d in axesT.data]
    checkshape = [d >= 0 and d < data_rank for d in data_idx]
    assert all(checkshape), f"axes={axesT.data} out of bounds: [-{data_rank}, {data_rank-1}]"
    # Squeeze only removes dimensions of size 1. This validates that.
    for idx in data_idx:
        if dataT.shape[idx] != 1:
            raise ValueError(f"Cannot squeeze dimension {idx} with size {dataT.shape[idx]} (must be 1)")
    outshape   = [dim for i,dim in enumerate(dataT.shape) if i not in data_idx]

    oTList[0].shape = outshape
    oTList[0].dtype = dataT.dtype

    # Compute data if inputs have data
    from ttsim.ops.desc.data_compute import try_compute_data, compute_squeeze
    oTList[0].data = try_compute_data(compute_squeeze, iTList, op)

    op.perf_stats = {
            'inBytes' : int(iTList[0].nbytes(op.precision) + iTList[1].nbytes(op.precision)),
            'outBytes': int(oTList[0].nbytes(op.precision)),
            'inElems' : int(iTList[0].nelems() + iTList[1].nelems()),
            'outElems': int(oTList[0].nelems()),
            'instrs'  : {'mov': int(oTList[0].nelems())}
            }
    return

def unsqueeze_sinf(iTList, oTList, op, **kwargs):
    assert iTList[0].check_shape(), f"Illegal Shape for {iTList[0]}"
    Y = iTList[1].clone_by_shape(data_maybe_missing=False) #Y.data must be present
    newshape = list(iTList[0].shape)
    for d in Y.data:
        newrank = len(newshape)
        if d < 0:
            d = newrank + d + 1
        if d < 0 or d > newrank:
            raise ValueError(f"Axis {d} out of bounds: [-{newrank+1}, {newrank}]")
        newshape.insert(d, 1)

    #tmp_outT = build_tmp_data_tensor(np_out, op.name + '__tmp_out__')
    #update_output_tensor(op, tmp_outT, oTList[0])
    oTList[0].shape = [int(i) for i in newshape]
    oTList[0].dtype = iTList[0].dtype

    # Compute actual data if inputs have data
    from ttsim.ops.desc.data_compute import try_compute_data, compute_unsqueeze
    oTList[0].data = try_compute_data(compute_unsqueeze, iTList, op)

    op.perf_stats = {
            'inBytes' : iTList[0].nbytes(op.precision) + iTList[1].nbytes(op.precision),
            'outBytes': oTList[0].nbytes(op.precision),
            'inElems' : iTList[0].nelems() + iTList[1].nelems(),
            'outElems': oTList[0].nelems(),
            'instrs'  : {'mov': oTList[0].nelems()}
            }
    return

def _detached_copy(t):
    """Independent copy of `t` for local shape normalization.

    Unlike SimTensor.clone_by_shape(), this ALWAYS copies (that method returns
    `self` whenever .data is present), and never fabricates data -- `data`
    stays None when the source has none, so try_compute_data's "all inputs
    have data" guard keeps working. hw_shape/x_pad_logical/y_pad_logical are
    instance attrs that the SimTensor constructor does NOT accept via cfg, so
    they must be assigned explicitly; they participate in LUT key construction.
    """
    c = SimTensor({
        'name'   : t.name + '.norm',
        'shape'  : list(t.shape),
        'dtype'  : t.dtype,
        'data'   : None if t.data is None else np.array(t.data, copy=True),
        'resolve': t.resolve,
        'op_in'  : t.op_in,
        'op_out' : t.op_out,
    })
    c.hw_shape      = getattr(t, 'hw_shape', None)
    c.x_pad_logical = getattr(t, 'x_pad_logical', None)
    c.y_pad_logical = getattr(t, 'y_pad_logical', None)
    return c

def _squeeze_to_rank1(t):
    """Squeeze away spurious leading size-1 dims (e.g. from a scalar Gather
    index that TT-Polaris represents as [1] instead of a true rank-0 scalar,
    then gets double-wrapped by a later Unsqueeze into [1,1]) so Slice's
    starts/ends/axes/steps tensors end up as genuine rank-1 arrays.

    Operates on a detached copy: `t` may be a shared graph node with other
    consumers, so the squeeze must not be written back into it."""
    if len(t.shape) <= 1:
        return t
    out = _detached_copy(t)
    while len(out.shape) > 1 and out.shape[0] == 1:
        out.shape = list(out.shape[1:])
        if out.data is not None:
            out.data = np.asarray(out.data).reshape(out.shape)
    return out

def slice_sinf(iTList, oTList, op, **kwargs):
    dataT = iTList[0]
    # ttnn-style slice: arity-1, output bounds carried as an 'output_shape' attr (not starts/ends
    # tensor inputs like the ONNX form). The HW SliceDeviceOperation is arity-1; keep the key arity-1.
    # With runtime bounds (use_tensor_args) inputs 1-2 are the start/end tensors, not ONNX data.
    if "output_shape" in op.attrs and (len(iTList) == 1 or op.attrs.get("use_tensor_args")):
        out_shape = [int(i) for i in op.attrs["output_shape"]]
        Y = oTList[0]
        Y.shape = out_shape
        Y.dtype = dataT.dtype
        assert Y.check_shape(), "SHAPE INFERENCE ERROR!!"
        op.perf_stats = {
            "inBytes": int(dataT.nbytes(op.precision)),
            "inElems": int(dataT.nelems()),
            "outBytes": int(Y.nbytes(op.precision)),
            "outElems": int(Y.nelems()),
            "instrs": {"mov": int(Y.nelems())},
        }
        return
    startsT = _squeeze_to_rank1(iTList[1].clone_by_shape(data_maybe_missing=False))
    endsT = _squeeze_to_rank1(iTList[2].clone_by_shape(data_maybe_missing=False))

    if len(iTList) >= 4:
        axesT = _squeeze_to_rank1(iTList[3].clone_by_shape(data_maybe_missing=False))
    else:
        axesT = build_tmp_data_tensor(
            np.array([i for i in range(dataT.rank())]),
            op.name + "__tmp_axesT__",
        )

    if len(iTList) == 5:
        stepsT = _squeeze_to_rank1(iTList[4].clone_by_shape(data_maybe_missing=False))
    else:
        stepsT = build_tmp_data_tensor(
            np.array([1 for _ in range(len(axesT.data))]),
            op.name + "__tmp_stepsT__",
        )

    assert startsT.rank() == 1, f"Slice Error 0, {startsT.shape}, rank != 1"
    assert startsT.shape == endsT.shape, f"Slice Error 1, {startsT.shape} != {endsT.shape}"
    assert startsT.shape == axesT.shape, f"Slice Error 2, {startsT.shape} != {axesT.shape}"
    assert startsT.shape == stepsT.shape, f"Slice Error 3, {startsT.shape} != {stepsT.shape}"

    Y = oTList[0]
    api = getattr(op, "api", None)

    if api == "ONNX" and Y.check_shape():
        out_shape = Y.shape
    elif "out_shape" in op.attrs and op.attrs["out_shape"] is not None:
        out_shape = [int(i) for i in op.attrs["out_shape"]]
    else:
        out_shape = list(dataT.shape)

    rank = dataT.rank()
    for s in range(startsT.rank()):
        axis = int(axesT.data[s])
        if axis < 0:
            axis += rank
        assert 0 <= axis < rank, f"Slice axis {axis} out of bounds for rank {rank}"

        dim = int(dataT.shape[axis])
        start = int(startsT.data[s])
        end = int(endsT.data[s])
        step = int(stepsT.data[s])

        if step == 0:
            raise ValueError(f"Slice step == 0 not supported (got {step})")

        if start < 0:
            start += dim
        if end < 0:
            end += dim

        if step > 0:
            start = max(0, min(dim, start))
            end = max(0, min(dim, end))
            length = 0 if end <= start else (end - start + step - 1) // step
        else:
            start = max(0, min(dim - 1, start))
            end = max(-1, min(dim - 1, end))
            length = 0 if start <= end else (start - end + (-step) - 1) // (-step)

        out_shape[axis] = length

    Y.shape = out_shape
    Y.dtype = dataT.dtype

    # Compute data if inputs have data
    from ttsim.ops.desc.data_compute import try_compute_data, compute_slice
    oTList[0].data = try_compute_data(compute_slice, iTList, op)

    inBytes = sum(x.nbytes(op.precision) for x in iTList)
    inElems = sum(x.nelems() for x in iTList)

    assert Y.check_shape(), "SHAPE INFERENCE ERROR!!"
    op.perf_stats = {
        "inBytes": int(inBytes),
        "inElems": int(inElems),
        "outBytes": int(Y.nbytes(op.precision)),
        "outElems": int(Y.nelems()),
        "instrs": {"mov": int(Y.nelems())},
    }
    return

def resize_sinf(iTList, oTList, op, **kwargs):
    if oTList[0].check_shape():
        pass
    else:
        assert len(iTList) >= 3, f"RESIZE #inputs ({len(iTList)}) should be >= 3"
        assert iTList[0].check_shape(), "Illegal Resize Input  Tensor Shape"
        assert iTList[1].check_shape(), "Illegal Resize ROI    Tensor Shape"
        assert iTList[2].check_shape(), "Illegal Resize SCALES Tensor Shape"
        assert iTList[2].data is not None, "SCALES data missing"
        XRank = iTList[0].rank()
        scales = [1.0] * XRank
        scales[-1] = iTList[2].data[-1]
        scales[-2] = iTList[2].data[-2]
        oTList[0].shape = [int(scales[i] * x) for i,x in enumerate(iTList[0].shape)]
        oTList[0].dtype = iTList[0].dtype

    # Compute data if input has data
    from ttsim.ops.desc.data_compute import try_compute_data, compute_resize
    if iTList[0].data is not None:
        oTList[0].data = try_compute_data(compute_resize, iTList, op)

    nElem = iTList[0].nelems()
    op.perf_stats = {
            'inElems' : iTList[0].nelems(),
            'inBytes' : iTList[0].nbytes(op.precision),
            'outElems': oTList[0].nelems(),
            'outBytes': oTList[0].nbytes(op.precision),
            'instrs'  : {'cmp': nElem, 'mov': nElem}
            }
    return

def upsample_sinf(iTList, oTList, op, **kwargs):
    X = iTList[0]
    input_shape = X.shape
    mode = op.attrs.get('mode', 'nearest')
    #scales = op.attrs.get('scales', None)
    scale_factor = op.attrs.get('scale_factor', 1)
    size = op.attrs.get('size', None)

    # Determine output shape
    if size is not None:
        # size is a tuple/list of output spatial dims
        # torch.nn.Upsample expects size to specify the output spatial dims
        # For 2D: input (N, C, H, W), size = (out_H, out_W)
        output_shape = list(input_shape)
        spatial_dims = len(size)
        output_shape[-spatial_dims:] = list(size)
    # elif scales is not None:
    elif scale_factor is not None:
        # scales is a list/tuple of scaling factors for each dim
        output_shape = list(input_shape)
        if len(output_shape) >= 2:
            output_shape[-2] = int(round(output_shape[-2] * scale_factor))
            output_shape[-1] = int(round(output_shape[-1] * scale_factor))
    else:
        raise ValueError("UpsampleOp requires either 'size' or 'scale_factor' attribute")

    oTList[0].shape = output_shape
    oTList[0].dtype = X.dtype

    # Compute data if inputs have data
    from ttsim.ops.desc.data_compute import try_compute_data, compute_upsample
    oTList[0].data = try_compute_data(compute_upsample, iTList, op)

    # Estimate instruction count: each output element is a copy or interpolation
    nElem_in = X.nelems()
    nElem_out = np.prod(output_shape)
    instr_count = {}
    if mode == 'nearest':
        instr_count['mov'] = nElem_out
    elif mode in ['linear', 'bilinear', 'trilinear', 'bicubic']:
        # Each output element: interpolation (mul/add per neighbor)
        # For bilinear: 4 neighbors, trilinear: 8, linear: 2, bicubic: 16
        neighbors = {'linear': 2, 'bilinear': 4, 'trilinear': 8, 'bicubic': 16}
        n = neighbors.get(mode, 4)
        instr_count['mul'] = nElem_out * n
        instr_count['add'] = nElem_out * (n - 1)
    else:
        raise ValueError(f"Unsupported Upsample mode: {mode}")

    op.perf_stats = {
        'inElems': nElem_in,
        'inBytes': X.nbytes(op.precision),
        'outElems': nElem_out,
        'outBytes': oTList[0].nbytes(op.precision),
        'instrs': instr_count
    }
    return

def concat_sinf(iTList, oTList, op, **kwargs):
    axis = op.attrs['axis']
    assert len(iTList) > 0, "empty input list in Concat!!"

    # Some upstream tensors (e.g. a scalar Gather-index result that TT-Polaris's
    # frontend represents as shape [1] instead of a true rank-0 scalar) can
    # pick up spurious extra leading size-1 dims through ops like Unsqueeze
    # (e.g. [1] -> [1,1] instead of the ONNX-true [] -> [1]). Squeeze away
    # such benign leading size-1 dims so ranks line up, rather than hard
    # failing on what is really just a rank-tracking artifact.
    # NOTE: clone into `norm` before squeezing -- iTList elements are shared
    # SimTensor graph nodes that can fan out to other consumers, so mutating
    # x.shape/x.data on iTList directly would corrupt those other consumers.
    norm = [_detached_copy(x) for x in iTList]
    min_rank = min(x.rank() for x in norm)
    for x in norm:
        while x.rank() > min_rank and len(x.shape) > 0 and x.shape[0] == 1:
            logger.warning(
                "Squeezing spurious leading size-1 dim for SimTensor({}) before Concat: {} -> {}",
                x.name, list(x.shape), list(x.shape[1:]),
            )
            x.shape = list(x.shape[1:])
            if x.data is not None:
                x.data = np.asarray(x.data).reshape(x.shape)
        # Same artifact can also show up as a spurious *trailing* size-1 dim
        # (e.g. a Cast/Reshape upstream that kept an extra [..., 1] dim instead
        # of collapsing to the ONNX-true rank), so squeeze from the tail too.
        while x.rank() > min_rank and len(x.shape) > 0 and x.shape[-1] == 1:
            logger.warning(
                "Squeezing spurious trailing size-1 dim for SimTensor({}) before Concat: {} -> {}",
                x.name, list(x.shape), list(x.shape[:-1]),
            )
            x.shape = list(x.shape[:-1])
            if x.data is not None:
                x.data = np.asarray(x.data).reshape(x.shape)

    base_rank = norm[0].rank()
    assert all(x.rank() == base_rank for x in norm), (
        f"input tensors rank mismatch: shapes={[list(x.shape) for x in norm]}, "
        f"names={[x.name for x in norm]}"
    )
    if axis < 0:
        axis = base_rank + axis
    if axis < 0 or axis >= base_rank:
        raise ValueError(f"Axis {axis} is out of bounds for tensors with rank {base_rank}. "
                    f"Valid range is [-{base_rank}, {base_rank-1}].")

    for i, x in enumerate(norm[1:], 1):
        for dim in range(x.rank()):
            if dim != axis and x.shape[dim] != norm[0].shape[dim]:
                    raise ValueError(f"Incompatible shapes at dim {i}: {x.shape} vs {norm[0].shape}. "
                                     f"All dimensions except the concat axis ({axis}) must match.")

    oshape        = list(norm[0].shape)
    oshape[axis]  = sum(x.shape[axis] for x in norm)
    oTList[0].shape = oshape
    oTList[0].dtype = norm[0].dtype

    # Propagate hw_shape when all inputs carry one and we're concatenating along
    # the NCHW channel axis (axis=1).  For channel-concat: hw_shape is [1, 1, N*H*W, C]
    # so we sum the last dim (C) across inputs.  Spatial-axis concat is not representable
    # in the flattened hw_shape format, so hw_shape is left None for those cases.
    # TODO: if spatial-axis concat becomes LUT-relevant, rethink the hw_shape convention.
    if axis == 1 and base_rank == 4:
        hw_shapes = [getattr(x, 'hw_shape', None) for x in norm]
        if all(hw is not None for hw in hw_shapes):
            total_c = sum(hw[3] for hw in hw_shapes)  # type: ignore[index]
            oTList[0].hw_shape = [hw_shapes[0][0], hw_shapes[0][1], hw_shapes[0][2], total_c]  # type: ignore[index]

    # Compute data if inputs have data
    from ttsim.ops.desc.data_compute import try_compute_data, compute_concat
    oTList[0].data = try_compute_data(compute_concat, norm, op)

    # Placeholder: For Training, it may be required to output per-input tensor shape
    # Assumption: per-input tensor shape is a 1D-Tensor where each element
    # represents the length of the corresponding input along the axis
    #
    #out2_shape = [len(iTList)]
    #out2_data  = [x.shape for x in Xs]

    inBytes = sum((x.nbytes(op.precision) for x in norm))
    inElems = sum((x.nelems() for x in norm))
    op.perf_stats = {
            'inBytes' : inBytes,
            'inElems' : inElems,
            'outBytes': oTList[0].nbytes(op.precision),
            'outElems': oTList[0].nelems(),
            'instrs'  : {'mov': oTList[0].nelems()}
            }
    return

def reshape_sinf(iTList, oTList, op, **kwargs):
    allowzero = op.attrs.get('allowzero', 0)

    #A = iTList[0].clone_by_shape()
    B = iTList[1].clone_by_shape(data_maybe_missing=False) #B.data should exist
    assert iTList[0].check_shape(), f"Illegal Input Shape: {iTList[0].shape}"
    assert B.dtype == np.int64, f"Input data type should be np.int64, was {B.dtype} for reshape's shape tensor {B}"
    assert B.data is not None, f"Input Data should exist for reshape's shape tensor {B}"
    input_shape  = iTList[0].shape
    input_size   = iTList[0].nelems()
    target_shape = [x.item() for x in B.data]

    minus_one_count = 0
    minus_one_index = None
    zeros_count     = 0
    zeros_index     = []
    for i,x in enumerate(target_shape):
        if x == -1:
            minus_one_count += 1
            minus_one_index = i
        elif x == 0:
            zeros_count += 1
            zeros_index.append(i)
        else:
            pass
    assert minus_one_count <= 1, f"Only one -1 is allowed in target shape {target_shape}"

    if allowzero == 1 and minus_one_count == 1 and zeros_count > 0:
        assert False, f"Cannot have -1 and zeros simultaneously with allowzero in target_shape({target_shape})"

    #copy dims from input_shape, if required
    output_shape = [x for x in target_shape]
    if allowzero == 0:
        for idx in zeros_index:
            assert idx < len(input_shape), f"Illegal index({idx}) for input_shape({input_shape}) with allowzero=0"
            output_shape[idx] = input_shape[idx]

    # Handle -1 inference
    if minus_one_count == 1:
        output_size = prod_ints([x for x in output_shape if x != -1])
        assert input_size >= output_size and input_size % output_size == 0, \
                f"Cannot infer -1: input size {input_size}/{output_size}"
        inferred_dim = input_size // output_size
        output_shape[minus_one_index] = inferred_dim #type: ignore

    # Final validation
    final_output_size = prod_ints(output_shape)
    assert input_size  == final_output_size, \
            f"in({input_size}) & out({final_output_size}) sizes are not equal!!"

    #np_out = reshape_reference_implementation(A.data, B.data, allowzero=allowzero)
    #tmp_outT = build_tmp_data_tensor(np_out, op.name + '__tmp_C_out__')
    #update_output_tensor(op, tmp_outT, oTList[0])
    oTList[0].shape = output_shape
    oTList[0].dtype = iTList[0].dtype

    # Compute data if inputs have data
    from ttsim.ops.desc.data_compute import try_compute_data, compute_reshape
    oTList[0].data = try_compute_data(compute_reshape, iTList, op)

    op.perf_stats = {
            'inElems' : int(iTList[0].nelems() + B.nelems()),
            'outElems': int(oTList[0].nelems()),
            'inBytes' : int(iTList[0].nbytes(op.precision) + B.nbytes(op.precision)),
            'outBytes': int(oTList[0].nbytes(op.precision)),
            'instrs'  : {'mov': int(oTList[0].nelems())}
            }
    return

def expand_sinf(iTList, oTList, op, **kwargs):
    A = iTList[0]
    shapeT = iTList[1].clone_by_shape(data_maybe_missing=False)
    assert shapeT.data is not None, "ExpandOp requires shape tensor with data"

    target_shape = [int(x) for x in shapeT.data]
    input_shape = list(A.shape)
    # Align shapes by prepending 1s to input_shape if needed
    if len(target_shape) > len(input_shape):
        input_shape = [1] * (len(target_shape) - len(input_shape)) + list(input_shape)

    out_shape = multidirectional_broadcast_shape_inference(
        [input_shape, target_shape]
    )

    oTList[0].shape = out_shape
    oTList[0].dtype = A.dtype

    # Compute data if input has data
    if A.data is not None:
        oTList[0].data = np.broadcast_to(np.asarray(A.data), out_shape).copy()

    inElems  = A.nelems() + shapeT.nelems()
    outElems = oTList[0].nelems()
    op.perf_stats = {
        'inElems':  inElems,
        'outElems': outElems,
        'inBytes':  A.nbytes(op.precision) + shapeT.nbytes(op.precision),
        'outBytes': oTList[0].nbytes(op.precision),
        'instrs':   {'mov': outElems},
    }
    return

def gatherelements_sinf(iTList, oTList, op, **kwargs):
    axis = op.attrs.get('axis', 0)
    assert isinstance(axis, int), f"attribute axis ({axis}) is not an int!!"

    dataT    = iTList[0]
    indicesT = iTList[1]
    assert dataT.check_shape(), f"Illegal input dataT shape: {dataT}!!"
    assert indicesT.check_shape(), f"Illegal input indicesT shape: {indicesT}!!"

    data_rank = dataT.rank()
    assert indicesT.rank() == data_rank, (
        f"GatherElements: indices rank ({indicesT.rank()}) must equal data rank ({data_rank})"
    )
    axis = axis if axis >= 0 else data_rank + axis
    assert 0 <= axis < data_rank, f"Axis {axis} out of bounds for dataT.shape {dataT.shape}"

    # GatherElements output shape == indices shape (indices has same rank as
    # data, but can differ in size along `axis`).
    oTList[0].shape = list(indicesT.shape)
    oTList[0].dtype = dataT.dtype

    if dataT.data is not None and indicesT.data is not None:
        idx = indicesT.data.astype(np.int64)
        idx = np.where(idx < 0, idx + dataT.shape[axis], idx)
        oTList[0].data = np.take_along_axis(dataT.data, idx, axis=axis)

    op.perf_stats = {
            'inElems' : int(oTList[0].nelems()),
            'outElems': int(oTList[0].nelems()),
            'inBytes' : int(oTList[0].nbytes(op.precision)),
            'outBytes': int(oTList[0].nbytes(op.precision)),
            'instrs'  : {'mov': int(oTList[0].nelems())}
            }
    return

def split_sinf(iTList, oTList, op, **kwargs):
    num_outputs = op.attrs.get('num_outputs', len(oTList))
    axis        = op.attrs.get('axis',0)

    A      = iTList[0]
    splitT = iTList[1] if len(iTList) == 2 else None
    assert A.check_shape(), "Illegal shape!!"
    if splitT is None or splitT.data is None:
        split_dim = A.shape[axis] // num_outputs
        split = [split_dim for i in range(num_outputs)]
    else:
        split = [x.item() for x in splitT.data]
    assert len(split) == num_outputs, f"split mismatch len( {split} ) != {num_outputs}"

    outShapes = []
    for tout_idx in range(num_outputs):
        tout_shape = A.shape.copy()
        tout_shape[axis] = split[tout_idx]
        outShapes.append(tout_shape)

    outBytes = 0
    outElems = 0
    for tidx, tout in enumerate(oTList):
        tshape0 = outShapes[tidx]
        tout.shape = tshape0
        tout.dtype = A.dtype
        outBytes += tout.nbytes(op.precision)
        outElems += tout.nelems()

    # Compute data if input has data
    if A.data is not None:
        from ttsim.ops.desc.data_compute import compute_split

        split_results = compute_split(iTList, op)
        for tidx, tout in enumerate(oTList):
            tout.data = split_results[tidx]

    splitT_nelems = 0 if splitT is None else splitT.nelems()
    splitT_nbytes = 0 if splitT is None else splitT.nbytes(op.precision)
    op.perf_stats = {
            'inElems' : A.nelems() + splitT_nelems,
            'outElems': outElems,
            'inBytes' : A.nbytes(op.precision) + splitT_nbytes,
            'outBytes': outBytes,
            'instrs'  : {'mov': outElems}
            }
    return

def gather_sinf(iTList, oTList, op, **kwargs):
    axis = op.attrs.get('axis', 0)
    assert isinstance(axis, int), f"attribute axis ({axis}) is not an int!!"

    dataT    = iTList[0]
    indicesT = iTList[1]
    assert dataT.check_shape(), f"Illegal input dataT shape: {dataT}!!"
    assert indicesT.check_shape(), f"Illegal input indicesT shape: {indicesT}!!"

    data_rank  = dataT.rank()
    data_shape = dataT.shape
    # Normalize negative axis
    axis = axis if axis >= 0 else data_rank + axis
    assert axis >= 0 and axis < data_rank, f"Axis {axis} is out of bounds for dataT.shape {dataT.shape}"
    oTList[0].shape = data_shape[:axis] + list(indicesT.shape) + data_shape[axis + 1:]
    oTList[0].dtype = dataT.dtype

    # Compute data if inputs have data
    from ttsim.ops.desc.data_compute import try_compute_data, compute_gather
    oTList[0].data = try_compute_data(compute_gather, iTList, op)

    if oTList[0].data is not None and list(oTList[0].data.shape) != list(oTList[0].shape):
        # dataT/indicesT's declared .shape metadata was stale (e.g. an
        # unresolved ONNX dim), so the shape computed above from .shape
        # disagrees with the real computed .data. Trust the real data.
        logger.warning(
            "Overriding stale/placeholder SHAPE for SimTensor({}): declared={} -> computed={}",
            oTList[0].name, oTList[0].shape, list(oTList[0].data.shape),
        )
        oTList[0].shape = list(oTList[0].data.shape)

    op.perf_stats = {
            'inElems' : int(oTList[0].nelems()), #read just what we need, not the whole embed. tbl
            'outElems': int(oTList[0].nelems()),
            'inBytes' : int(oTList[0].nbytes(op.precision)),
            'outBytes': int(oTList[0].nbytes(op.precision)),
            'instrs'  : {'mov': int(oTList[0].nelems())}
            }
    return

def embedding_sinf(iTList, oTList, op, **kwargs):
    """ttnn.embedding row lookup (HW EmbeddingsDeviceOperation).

    Operand order matches the hardware op: iTList[0] = tokens (indices), iTList[1] = weight
    [vocab, embed_dim] — tokens-first, the OPPOSITE of ONNX Gather (data-first, see
    gather_sinf). Output = tokens.shape + [embed_dim]. Kept a distinct op-code from Gather so
    the LUT key canonicalizes to 'embedding' (matching the capture) rather than 'gather'."""
    tokensT = iTList[0]
    weightT = iTList[1]
    assert tokensT.check_shape(), f"Illegal embedding tokens shape: {tokensT}!!"
    assert weightT.check_shape(), f"Illegal embedding weight shape: {weightT}!!"
    embed_dim = list(weightT.shape)[1:]  # weight [vocab, embed_dim] -> [embed_dim]
    oTList[0].shape = list(tokensT.shape) + embed_dim
    oTList[0].dtype = weightT.dtype
    n = int(oTList[0].nelems())
    op.perf_stats = {
            'inElems' : n,   # read just the rows we look up, not the whole embed table
            'outElems': n,
            'inBytes' : int(oTList[0].nbytes(op.precision)),
            'outBytes': int(oTList[0].nbytes(op.precision)),
            'instrs'  : {'mov': n}
            }
    return


def gathernd_sinf(iTList, oTList, op, **kwargs):
    data = iTList[0]
    indices = iTList[1]

    data_shape = list(data.shape)
    idx_shape = list(indices.shape)
    batch_dims = int(op.attrs.get("batch_dims", 0))
    assert isinstance(batch_dims, int) and batch_dims >= 0, (
        f"GatherND: batch_dims ({batch_dims}) must be a non-negative integer"
    )
    assert batch_dims == 0, "GatherND: batch_dims != 0 not yet supported"
    assert len(idx_shape) >= 1, "GatherND: indices must have rank >= 1"
    q = idx_shape[-1]
    assert q <= len(data_shape), (
        f"GatherND: last dim of indices ({q}) > rank of data ({len(data_shape)})"
    )
    out_shape = idx_shape[:-1] + data_shape[q:]

    oTList[0].shape = out_shape
    oTList[0].dtype = data.dtype

    in_elems = data.nelems() + indices.nelems()
    out_elems = oTList[0].nelems()

    op.perf_stats = {
        "inElems": in_elems,
        "inBytes": data.nbytes(op.precision) + indices.nbytes(op.precision),
        "outElems": out_elems,
        "outBytes": oTList[0].nbytes(op.precision),
        "instrs": {"mov": out_elems},
    }
    return

def shape_op_inf_func(iTList, oTList, op, **kwargs):
    A = iTList[0].clone_by_shape()

    start =  op.attrs.get('start', 0)
    end   =  op.attrs.get('end')
    start = 0 if start < 0 else start
    end   = A.rank() if end is None or end > A.rank() else end
    end   = A.rank() + end if end < 0 else end
    tdata = np.array(A.shape[start:end], dtype=np.int64)
    oTList[0].shape = list(tdata.shape)
    # Shape's output is always int64 per the ONNX spec. The previous code
    # (`oTList[0].dtype = np.int64`) assigned the bare Python/numpy *class*
    # rather than a numpy dtype *instance*, which downstream code (e.g.
    # nbytes()) couldn't handle: "TypeError: Unsupported dtype type:
    # <class 'type'>". np.dtype(np.int64) is the fix -- same int64 type,
    # just wrapped as a proper dtype instance.
    oTList[0].dtype = np.dtype(np.int64)
    oTList[0].data = tdata
    op.perf_stats = {
            'inElems' : A.rank(),
            'outElems': A.rank(),
            'inBytes' : A.rank() * 4,
            'outBytes': A.rank() * 4,
            'instrs'  : {'mov': A.rank()} # 4Bytes per Index
            }
    return

def transpose_op_inf_func(iTList, oTList, op, **kwargs):
    perms  = op.attrs['perm']
    assert len(perms) == iTList[0].rank(), f"perms({perms}) must be equal to input rank ({iTList[0].rank()})!!"
    oTList[0].shape = [iTList[0].shape[i] for i in perms]
    oTList[0].dtype = iTList[0].dtype

    # Compute data if available
    from ttsim.ops.desc.data_compute import try_compute_data, compute_transpose
    oTList[0].data = try_compute_data(compute_transpose, iTList, op)

    op.perf_stats = {
            'inElems' : iTList[0].nelems(),
            'outElems': oTList[0].nelems(),
            'inBytes' : iTList[0].nbytes(op.precision),
            'outBytes': oTList[0].nbytes(op.precision),
            'instrs'  : {'mov': oTList[0].nelems()}
            }
    return

def scatter_nd_sinf(iTList, oTList, op, **kwargs):
    """Shape inference for ScatterND (ONNX).

    Input:
        iTList[0] = data    — the base tensor
        iTList[1] = indices — int64 tensor, last dim indexes into data
        iTList[2] = updates — values to scatter
    Output shape = data.shape  (always same as input data).
    """
    dataT    = iTList[0]
    indicesT = iTList[1]
    updatesT = iTList[2]
    assert dataT.check_shape(),    f"ScatterND: bad data shape: {dataT}!!"
    assert indicesT.check_shape(), f"ScatterND: bad indices shape: {indicesT}!!"
    assert updatesT.check_shape(), f"ScatterND: bad updates shape: {updatesT}!!"

    oTList[0].shape = list(dataT.shape)
    oTList[0].dtype = dataT.dtype

    from ttsim.ops.desc.data_compute import try_compute_data, compute_scatter_nd
    oTList[0].data = try_compute_data(compute_scatter_nd, iTList, op)

    outElems = int(oTList[0].nelems())
    op.perf_stats = {
        'inElems' : outElems,
        'outElems': outElems,
        'inBytes' : int(oTList[0].nbytes(op.precision)),
        'outBytes': int(oTList[0].nbytes(op.precision)),
        'instrs'  : {'mov': outElems}
    }
    return


def permute_op_inf_func(iTList, oTList, op, **kwargs):
    """Same inference as Transpose; distinct optype for TTNN permute_op / profiler naming."""
    transpose_op_inf_func(iTList, oTList, op, **kwargs)


def fold_sinf(iTList, oTList, op, **kwargs):
    """Shape inference for Fold: 4D [N,H,W,C] -> [N,H//sh,W//sw,C*sh*sw]; attrs stride_h, stride_w; optional pad_*."""
    assert len(iTList) == 1 and len(oTList) == 1
    X = iTList[0]
    assert X.check_shape(), f"Fold: input shape not defined: {X}"
    assert X.rank() == 4, f"Fold expects rank-4 [N,H,W,C], got rank {X.rank()}"
    N, H, W, C = (int(X.shape[0]), int(X.shape[1]), int(X.shape[2]), int(X.shape[3]))
    stride_h = int(op.attrs['stride_h'])
    stride_w = int(op.attrs['stride_w'])
    pad_h = int(op.attrs.get('pad_h', 0))
    pad_w = int(op.attrs.get('pad_w', 0))
    pad_c = int(op.attrs.get('pad_c', 0))
    if pad_h > 0:
        H += pad_h
    if pad_w > 0:
        W += pad_w
    if pad_c > 0:
        C += pad_c
    assert H % stride_h == 0 and W % stride_w == 0, (
        f"Fold: H,W must divide by strides; H={H}, W={W}, stride_h={stride_h}, stride_w={stride_w}"
    )
    Hs = H // stride_h
    Ws = W // stride_w
    Cs = C * stride_h * stride_w
    # Design decision: the default (flatten_nd absent or False) produces the
    # standard mathematical 4D shape [N, Hs, Ws, Cs].  This keeps fold_sinf
    # frontend-agnostic -- any non-TTNN caller (e.g. ONNX, future frontends)
    # gets correct behaviour without needing to know about Tenstorrent HW
    # layout semantics.  Only the TTNN frontend explicitly opts into the
    # flattened [1,1,N*Hs*Ws,Cs] form by setting flatten_nd=True, which
    # mirrors the conditional logic in tt-metal's fold_device_op.cpp
    # (prim::fold flattens for L1-sharded ROW_MAJOR; compute_output_specs
    # preserves 4D for DRAM-interleaved or tiled inputs).
    flatten_nd = bool(op.attrs.get('flatten_nd', False))
    if flatten_nd:
        out_shape = [1, 1, N * Hs * Ws, Cs]
    else:
        out_shape = [N, Hs, Ws, Cs]
    oTList[0].shape = out_shape
    oTList[0].dtype = X.dtype
    in_elems = X.nelems()
    out_elems = prod_ints(out_shape)
    op.perf_stats = {
        'inElems': in_elems,
        'outElems': out_elems,
        'inBytes': X.nbytes(op.precision),
        'outBytes': oTList[0].nbytes(op.precision),
        'instrs': {'mov': out_elems},
    }
    return

def register_tensor_ops():
    _optbl = [
            ##['Size',           'ARITY_1->1', 'ai.onnx',  'COMMON',  24,  21,  1,  1,  1,  1,  'inline_lambda',  True,  True,  True,  True,  True],
            ##['SpaceToDepth',   'ARITY_1->1', 'ai.onnx',  'COMMON',  13,  13,  1,  1,  1,  1,  'inline_lambda',  True,  True,  True,  True,  True],
            ##['DepthToSpace',   'ARITY_1->1', 'ai.onnx',  'COMMON',  13,  13,  1,  1,  1,  1,  'inline_lambda',  True,  True,  True,  True,  True],
            ##['IsNaN',          'ARITY_1->1', 'ai.onnx',  'COMMON',  20,  20,  1,  1,  1,  1,  'inline_lambda',  True,  True,  True,  True,  True],
            ##['IsInf',          'ARITY_1->1', 'ai.onnx',  'COMMON',  20,  20,  1,  1,  1,  1,  'inline_lambda',  True,  True,  True,  True,  True],
            ##['NonZero',        'ARITY_1->1', 'ai.onnx',  'COMMON',  13,  13,  1,  1,  1,  1,  'inline_lambda',  True,  True,  True,  True,  True],
            ##['CastLike',       'ARITY_2->1', 'ai.onnx',  'COMMON',  24,  21,  2,  2,  1,  1,  'inline_lambda',  True,  True,  True,  True,  True],
            ##['GatherElements', 'ARITY_2->1', 'ai.onnx',  'COMMON',  13,  13,  2,  2,  1,  1,  'inline_lambda',  True,  True,  True,  True,  True],
            ['GridSample',     'ARITY_2->1', 'ai.onnx',  'COMMON',  22,  22,  2,  2,  1,  1,  gridsample_sinf,  True,  True,  True,  True,  True],
            ##['AffineGrid',     'ARITY_2->1', 'ai.onnx',  'COMMON',  20,  20,  2,  2,  1,  1,  'inline_lambda',  True,  True,  True,  True,  True],
            ##['Compress',       'ARITY_2->1', 'ai.onnx',  'COMMON',  11,  11,  2,  2,  1,  1,  'inline_lambda',  True,  True,  True,  True,  True],
            ##['ReverseSequence','ARITY_2->1', 'ai.onnx',  'COMMON',  10,  10,  2,  2,  1,  1,  'inline_lambda',  True,  True,  True,  True,  True],
            ##['GatherND',       'ARITY_2->1', 'ai.onnx',  'COMMON',  13,  13,  2,  2,  1,  1,  'inline_lambda',  True,  True,  True,  True,  True],
            ##['CenterCropPad',  'ARITY_2->1', 'ai.onnx',  'COMMON',  18,  18,  2,  2,  1,  1,  'inline_lambda',  True,  True,  True,  True,  True],
            ##['Scatter',        'ARITY_3->1', 'ai.onnx',  'COMMON',  11,  11,  3,  3,  1,  1,  'inline_lambda',  True,  True,  True,  True,  True],
            ['ScatterND',      'ARITY_3->1', 'ai.onnx',  'COMMON',  18,  18,  3,  3,  1,  1,  scatter_nd_sinf,  True,  True,  True,  True,  True],
            ##['ScatterElements','ARITY_3->1', 'ai.onnx',  'COMMON',  18,  18,  3,  3,  1,  1,  'inline_lambda',  True,  True,  True,  True,  True],
            ##['OneHot',         'ARITY_3->1', 'ai.onnx',  'COMMON',  11,  11,  3,  3,  1,  1,  'inline_lambda',  True,  True,  True,  True,  True],
            ##['Unique',  'ARITY_1->VARIADIC[1-4]', 'ai.onnx',  'COMMON',  11,  11,  1,  1,  4,  1,  'inline_lmbda', True, True, True, True, True],

            ['Shape',     'ARITY_1->1', 'ai.onnx',  'COMMON',  24,  21,  1,  1,  1,  1, shape_op_inf_func,  True,  True,  True,  True,  True],
            ['Transpose', 'ARITY_1->1', 'ai.onnx',  'COMMON',  24,  21,  1,  1,  1,  1, transpose_op_inf_func,  True,  True,  True,  True,  True],
            ['Permute',   'ARITY_1->1', 'com.tenstorrent.ttnn',  'COMMON',  24,  21,  1,  1,  1,  1, permute_op_inf_func,    True,  True,  True,  True,  True],
            ['Identity',  'ARITY_1->1', 'ai.onnx',  'COMMON',  24,  21,  1,  1,  1,  1, unary_fwd,  True,  True,  True,  True,  True],

            ['Gather',    'ARITY_2->1', 'ai.onnx',  'COMMON',  13,  13,  2,  2,  1,  1, gather_sinf,  True,  True,  True,  True,  True],
            ['GatherElements', 'ARITY_2->1', 'ai.onnx',  'COMMON',  13,  13,  2,  2,  1,  1, gatherelements_sinf, True, True, True, True, True],
            ['Embedding', 'ARITY_2->1', 'custom',   'NA',      -1,  -1,  2,  2,  1,  1, embedding_sinf, True, True, True, True, True],
            ['GatherND',  'ARITY_2->1', 'ai.onnx',  'COMMON',  13,  13,  2,  2,  1,  1, gathernd_sinf, True, True, True, True, True],
            ['Cast',      'ARITY_1->1', 'ai.onnx',  'COMMON',  24,  21,  1,  1,  1,  1, cast_sinf,  True,  True,  True,  True,  True],
            ['IsNaN',     'ARITY_1->1', 'ai.onnx',   'COMMON',  20,  20,  1,  1,  1,  1, isnan_sinf,    True, True, True, True, True],

            ['Reshape',   'ARITY_2->1', 'ai.onnx',  'COMMON',  24,  21,  2,  2,  1,  1, reshape_sinf,  True,  True,  True,  True,  True],
            ['Fold',      'ARITY_1->1', 'com.tenstorrent.ttnn',  'COMMON',  24,  21,  1,  1,  1,  1, fold_sinf,  True,  True,  True,  True,  True],
            ['Expand',    'ARITY_2->1', 'ai.onnx',  'COMMON',  13,  13,  2,  2,  1,  1, expand_sinf,  True,  True,  True,  True,  True],

            ['Unsqueeze', 'ARITY_2->1', 'ai.onnx',  'COMMON',  24,  21,  2,  2,  1,  1,
             unsqueeze_sinf,  True,  True,  True,  True,  True],
            ['Tile',      'ARITY_2->1', 'ai.onnx',  'COMMON',  13,  13,  2,  2,  1,  1, tile_sinf,  True,  True,  True,  True,  True],
            ['Upsample',  'ARITY_2->1', 'ai.onnx',  'COMMON',  10,  10,  2,  2,  1,  1,
             upsample_sinf,  True,  True,  True,  True,  True],
            ['Where',     'ARITY_3->1', 'ai.onnx',  'COMMON',  16,  16,  3,  3,  1,  1, where_sinf,  True,  True,  True,  True,  True],

            ['Squeeze',   'ARITY_VARIADIC[1-2]->1', 'ai.onnx',  'COMMON',  24,  21,  2,  1,  1,  1,
             squeeze_sinf, True, True, True, True, True],
            ['Trilu',     'ARITY_VARIADIC[1-2]->1', 'ai.onnx',  'COMMON',  14,  14,  2,  1,  1,  1,
             trilu_sinf, True, True, True, True, True],
            ['Resize',    'ARITY_VARIADIC[1-4]->1', 'ai.onnx',  'COMMON',  19,  19,  4,  1,  1,  1,
             resize_sinf, True, True, True, True, True],
            ['Slice',     'ARITY_VARIADIC[1-5]->1', 'ai.onnx',  'COMMON',  13,  13,  5,  1,  1,  1,
             slice_sinf, True, True, True, True, True],
            ['Pad',       'ARITY_VARIADIC[2-4]->1', 'ai.onnx',  'COMMON',  24,  21,  4,  2,  1,  1,
             pad_sinf, True, True, True, True, True],

            ['Concat', 'ARITY_VARIADIC[1-*]->1',             'ai.onnx', 'COMMON', 13, 13,
             2147483647, 1, 1, 1, concat_sinf, True, True, True, True, True],
            ['Split',  'ARITY_VARIADIC[1-2]->VARIADIC[1-*]', 'ai.onnx', 'COMMON', 18, 18, 2, 1,
             2147483647, 1, split_sinf, True, True, True, True, True],
            ]

    register_ops('tensor', _optbl)
    return
