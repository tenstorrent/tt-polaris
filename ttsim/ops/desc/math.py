#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from ttsim.ops.desc.registry import register_ops
from ttsim.ops.desc.helpers import (
    unary_fwd,
    bidir_bcast,
    bidirectional_broadcast_shape_inference,
)

import numpy as np
import copy


def binary_cross_entropy_with_logits_sinf(iTList, oTList, op, **kwargs):
    """
    Binary Cross Entropy with Logits shape inference.
    Computes: -[target * log(sigmoid(input)) + (1 - target) * log(1 - sigmoid(input))]

    Args:
        iTList: [input, target]
            input: [..., any_shape] - logits (unnormalized)
            target: [..., any_shape] (must match input)
        oTList: [output_tensor]
        op: SimOp with optional attr 'reduction' ('none', 'mean', 'sum')
    """
    from .data_compute import compute_binary_cross_entropy_with_logits

    input_t = iTList[0]
    target_t = iTList[1]
    output_t = oTList[0]

    # Get reduction mode
    reduction = op.attrs.get("reduction", "mean")

    # Shape inference
    assert input_t.check_shape(), f"Input shape not defined: {input_t}"
    assert target_t.check_shape(), f"Target shape not defined: {target_t}"
    assert (
        input_t.shape == target_t.shape
    ), f"Shape mismatch: {input_t.shape} vs {target_t.shape}"

    if reduction == "none":
        output_t.shape = list(input_t.shape)
    else:
        output_t.shape = []  # Scalar output for 'mean' or 'sum'

    output_t.dtype = input_t.dtype

    # Performance stats
    n_elems = input_t.nelems()
    op.perf_stats = {
        "inElems": n_elems * 2,
        "outElems": n_elems if reduction == "none" else 1,
        "inBytes": input_t.nbytes(op.precision) + target_t.nbytes(op.precision),
        "outBytes": output_t.nbytes(op.precision),
        "instrs": {
            "sigmoid": n_elems,
            "log": n_elems * 2,
            "mul": n_elems * 2,
            "add": n_elems * 2 if reduction != "none" else n_elems,
        },
    }

    # Numerical computation (only if data available)
    if input_t.data is not None and target_t.data is not None:
        output_t.data = compute_binary_cross_entropy_with_logits(iTList, op)
    else:
        output_t.data = None

    return output_t

def softmax_cross_entropy_loss_sinf(iTList, oTList, op, **kwargs):
    """
    SoftmaxCrossEntropyLoss shape/perf inference.
    """
    scores = iTList[0]
    labels = iTList[1]

    loss_out = oTList[0]
    log_prob_out = oTList[1] if len(oTList) > 1 else None

    reduction = op.attrs.get("reduction", "mean")

    assert scores.check_shape(), f"Scores shape not defined: {scores}"
    assert labels.check_shape(), f"Labels shape not defined: {labels}"

    axis = op.attrs.get("axis", 1)
    scores_shape = list(scores.shape)
    labels_shape = list(labels.shape)

    if axis < 0:
        axis += len(scores_shape)
    assert 0 <= axis < len(scores_shape), (
        f"Invalid axis={axis} for scores shape {scores_shape}"
    )

    index_label_shape = scores_shape[:axis] + scores_shape[axis + 1 :]

    assert (
        labels_shape == scores_shape
        or labels_shape == index_label_shape
    ), (
        f"Incompatible scores/labels shapes for SoftmaxCrossEntropyLoss: "
        f"scores={scores_shape}, labels={labels_shape}, axis={axis}"
    )

    if reduction == "none":
        loss_out.shape = list(labels.shape)
    else:
        loss_out.shape = []
    loss_out.dtype = scores.dtype

    if log_prob_out is not None:
        log_prob_out.shape = list(scores.shape)
        log_prob_out.dtype = scores.dtype

    inBytes = sum(t.nbytes(op.precision) for t in iTList)
    inElems = sum(t.nelems() for t in iTList)
    outBytes = sum(
        t.nbytes(op.precision) for t in oTList if t.check_shape()
    )
    outElems = sum(t.nelems() for t in oTList if t.check_shape())

    op.perf_stats = {
        "inElems": int(inElems),
        "inBytes": int(inBytes),
        "outElems": int(outElems),
        "outBytes": int(outBytes),
        "instrs": {"mov": int(outElems)},
    }

    return loss_out

def einsum_sinf(iTList, oTList, op, **kwargs):
    """
    Einstein summation shape inference.

    Implements torch.einsum / numpy.einsum functionality.

    Args:
        iTList: List of input tensors (variable length)
        oTList: [output_tensor]
        op: SimOp with required attr 'subscripts' (einsum notation string)
            Example: "bqnc,bnchw->bqnhw"
    """
    from .data_compute import compute_einsum

    output_t = oTList[0]
    # ONNX's documented attribute name is "equation" (e.g. "bhwc,hkc->bhwk").
    # Some frontends/older exports may still populate "subscripts" instead,
    # so fall back to that rather than assuming every model uses "equation" --
    # this keeps any other model already relying on "subscripts" working too.
    subscripts = op.attrs.get("equation") or op.attrs.get("subscripts", "")

    assert subscripts, "Einsum requires an 'equation' (or legacy 'subscripts') attribute"
    assert len(iTList) >= 1, "Einsum requires at least one input"

    # Parse subscripts to determine output shape
    # Format: "abc,bcd->ad" or "abc,bcd" (implicit output)
    if "->" in subscripts:
        input_subs, output_sub = subscripts.split("->")
    else:
        # Implicit mode: output contains labels that appear exactly once
        input_subs = subscripts
        output_sub = None

    input_sub_list = input_subs.split(",")
    assert len(input_sub_list) == len(
        iTList
    ), f"Number of subscripts ({len(input_sub_list)}) must match inputs ({len(iTList)})"

    # Build dimension mapping: label -> size
    dim_map: dict[str, int] = {}
    for tensor, sub in zip(iTList, input_sub_list):
        assert tensor.check_shape(), f"Input shape not defined: {tensor}"
        assert len(sub) == len(
            tensor.shape
        ), (f"Subscript length ({len(sub)}) must match tensor rank ({len(tensor.shape)}) "
            f"for tensor {tensor.name} shape={list(tensor.shape)} equation={subscripts!r}")

        for label, size in zip(sub, tensor.shape):
            if label != " ":  # Skip spaces
                if label in dim_map:
                    assert (
                        dim_map[label] == size
                    ), f"Dimension mismatch for '{label}': {dim_map[label]} vs {size}"
                else:
                    dim_map[label] = size

    # Determine output shape
    if output_sub is None:
        # Implicit mode: include labels that appear exactly once
        label_counts: dict[str, int] = {}
        for sub in input_sub_list:
            for label in sub:
                if label != " ":
                    label_counts[label] = label_counts.get(label, 0) + 1
        output_sub = "".join(sorted([l for l, c in label_counts.items() if c == 1]))

    output_t.shape = [dim_map[label] for label in output_sub if label != " "]
    output_t.dtype = iTList[0].dtype

    # Performance stats
    total_in_elems = sum(t.nelems() for t in iTList)
    output_t_elems = output_t.nelems()

    op.perf_stats = {
        "inElems": total_in_elems,
        "outElems": output_t_elems,
        "inBytes": sum(t.nbytes(op.precision) for t in iTList),
        "outBytes": output_t.nbytes(op.precision),
        "instrs": {
            "mul": output_t_elems * max(len(t.shape) for t in iTList),
            "add": output_t_elems,
        },
    }

    # Numerical computation
    if all(t.data is not None for t in iTList):
        output_t.data = compute_einsum(iTList, op)
    else:
        output_t.data = None

    return output_t


def cdist_sinf(iTList, oTList, op, **kwargs):
    """
    Pairwise distance computation shape inference.

    Computes pairwise distance between rows of two tensors using p-norm.

    Args:
        iTList: [x1, x2]
            x1: [..., P, M] - First collection of row vectors
            x2: [..., R, M] - Second collection of row vectors (must have same batch dims and M)
        oTList: [output_tensor]
        op: SimOp with optional attr 'p' (norm order, default 2.0)
            p=1: Manhattan distance (L1)
            p=2: Euclidean distance (L2)

    Returns:
        output: [..., P, R] - Pairwise distances
    """
    from .data_compute import compute_cdist

    x1 = iTList[0]
    x2 = iTList[1]
    output_t = oTList[0]

    p = op.attrs.get("p", 2.0)

    # Shape validation
    assert x1.check_shape(), f"x1 shape not defined: {x1}"
    assert x2.check_shape(), f"x2 shape not defined: {x2}"
    assert len(x1.shape) >= 2, f"x1 must be at least 2D, got {len(x1.shape)}D"
    assert len(x2.shape) >= 2, f"x2 must be at least 2D, got {len(x2.shape)}D"
    assert len(x1.shape) == len(
        x2.shape
    ), f"x1 and x2 must have same rank: {len(x1.shape)} vs {len(x2.shape)}"

    # Check batch dimensions match
    batch_dims = x1.shape[:-2]
    assert list(x1.shape[:-2]) == list(
        x2.shape[:-2]
    ), f"Batch dimensions must match: {x1.shape[:-2]} vs {x2.shape[:-2]}"

    # Check feature dimension matches
    M1 = x1.shape[-1]
    M2 = x2.shape[-1]
    assert M1 == M2, f"Feature dimension must match: {M1} vs {M2}"

    # Output shape: [...batch..., P, R]
    P = x1.shape[-2]
    R = x2.shape[-2]
    output_t.shape = list(batch_dims) + [P, R]
    output_t.dtype = x1.dtype

    # Performance stats
    x1_elems = x1.nelems()
    x2_elems = x2.nelems()
    output_elems = output_t.nelems()

    op.perf_stats = {
        "inElems": x1_elems + x2_elems,
        "outElems": output_elems,
        "inBytes": x1.nbytes(op.precision) + x2.nbytes(op.precision),
        "outBytes": output_t.nbytes(op.precision),
        "instrs": {
            "sub": output_elems * M1,
            "abs": output_elems * M1 if p == 1 else 0,
            "mul": output_elems * M1 if p == 2 else 0,
            "add": output_elems * (M1 - 1),
            "sqrt": output_elems if p == 2 else 0,
        },
    }

    # Numerical computation
    if x1.data is not None and x2.data is not None:
        output_t.data = compute_cdist(iTList, op)
    else:
        output_t.data = None

    return output_t


def diag_sinf(iTList, oTList, op, **kwargs):
    """
    Diagonal extraction/construction shape inference.
    - If input is 1D: creates 2D matrix with input on diagonal
    - If input is 2D: extracts diagonal as 1D vector

    Args:
        iTList: [input_tensor]
            input_tensor: [N] or [N, M]
        oTList: [output_tensor]
        op: SimOp with optional attr 'diagonal' (offset, default 0)
    """
    from .data_compute import compute_diag

    input_t = iTList[0]
    output_t = oTList[0]

    diagonal = op.attrs.get("diagonal", 0)

    # Shape inference
    assert input_t.check_shape(), f"Input shape not defined: {input_t}"

    if len(input_t.shape) == 1:
        # 1D -> 2D diagonal matrix
        n = input_t.shape[0]
        output_t.shape = [n, n]
    elif len(input_t.shape) == 2:
        # 2D -> 1D diagonal extraction
        rows, cols = input_t.shape
        if diagonal >= 0:
            diag_len = min(rows, cols - diagonal)
        else:
            diag_len = min(rows + diagonal, cols)
        output_t.shape = [max(0, diag_len)]
    else:
        raise ValueError(f"diag expects 1D or 2D input, got shape {input_t.shape}")

    output_t.dtype = input_t.dtype

    # Performance stats
    out_elems = int(np.prod(output_t.shape)) if output_t.shape else 0
    op.perf_stats = {
        "inElems": input_t.nelems(),
        "outElems": out_elems,
        "inBytes": input_t.nbytes(op.precision),
        "outBytes": output_t.nbytes(op.precision),
        "instrs": {"copy": out_elems},
    }

    # Numerical computation (only if data available)
    if input_t.data is not None:
        output_t.data = compute_diag(iTList, op)
    else:
        output_t.data = None

    return output_t


def inverse_sigmoid_sinf(iTList, oTList, op, **kwargs):
    """
    Inverse sigmoid shape inference.
    Computes: log(x / (1 - x))

    Args:
        iTList: [input_tensor]
            input_tensor: [..., any_shape]
        oTList: [output_tensor]
        op: SimOp with optional attr 'eps' (default 1e-5) for numerical stability
    """
    from .data_compute import compute_inverse_sigmoid

    input_t = iTList[0]
    output_t = oTList[0]

    # Get epsilon from attributes (with default)
    eps = op.attrs.get("eps", 1e-5)

    # Shape inference - output has same shape as input
    assert input_t.check_shape(), f"Input shape not defined: {input_t}"
    output_t.shape = list(input_t.shape)
    output_t.dtype = input_t.dtype

    # Performance stats
    n_elems = input_t.nelems()
    op.perf_stats = {
        "inElems": n_elems,
        "outElems": n_elems,
        "inBytes": input_t.nbytes(op.precision),
        "outBytes": output_t.nbytes(op.precision),
        "instrs": {
            "log": n_elems,
            "div": n_elems * 2,
            "sub": n_elems,
        },  # log(x / (1-x))
    }

    # Numerical computation (only if data available)
    if input_t.data is not None:
        output_t.data = compute_inverse_sigmoid(iTList, op)
    else:
        output_t.data = None

    return output_t


def glu_sinf(iTList, oTList, op, **kwargs):
    """
    Gated Linear Unit (GLU) shape inference.
    Computes: GLU(x) = x[:, :n] * sigmoid(x[:, n:]) where input is split in half.

    Args:
        iTList: [input_tensor]
            input_tensor: [..., 2*dim, ...] where split dimension must be even
        oTList: [output_tensor]
        op: SimOp with optional attr 'dim' (default -1)
    """
    from .data_compute import compute_glu

    input_t = iTList[0]
    output_t = oTList[0]

    # Get split dimension
    dim = op.attrs.get("dim", -1)

    # Shape inference - output has half size along split dimension
    assert input_t.check_shape(), f"Input shape not defined: {input_t}"

    input_shape = list(input_t.shape)
    rank = len(input_shape)

    # Normalize dimension
    if dim < 0:
        dim += rank

    assert 0 <= dim < rank, f"Invalid dimension {dim} for shape {input_shape}"
    assert (
        input_shape[dim] % 2 == 0
    ), f"GLU requires even dimension at axis {dim}, got {input_shape[dim]}"

    # Output shape: halve the specified dimension
    output_shape = input_shape.copy()
    output_shape[dim] = input_shape[dim] // 2

    output_t.shape = output_shape
    output_t.dtype = input_t.dtype

    # Performance stats
    out_elems = int(np.prod(output_shape))
    in_elems = input_t.nelems()

    op.perf_stats = {
        "inElems": in_elems,
        "outElems": out_elems,
        "inBytes": input_t.nbytes(op.precision),
        "outBytes": output_t.nbytes(op.precision),
        "instrs": {
            "sigmoid": out_elems,  # sigmoid(x[:, n:])
            "mul": out_elems,  # x[:, :n] * sigmoid(x[:, n:])
        },
    }

    # Numerical computation (only if data available)
    if input_t.data is not None:
        output_t.data = compute_glu(iTList, op)
    else:
        output_t.data = None

    return output_t


def variadic_sinf(iTList, oTList, op, **kwargs):
    """
    Shape inference for variadic input operations (Sum, Mean, Max, Min).

    For reduction operations (with 'dim' or 'axis' attribute):
        - Reduces along specified dimension
        - Output has reduced dimensions

    For element-wise operations (no 'dim'/'axis' attribute):
        - For Max/Min with multiple inputs: element-wise maximum/minimum
        - Broadcasts inputs and produces element-wise result
        - Output shape is broadcast result of all inputs
    """
    dim = op.attrs.get('dim', None)
    axis = op.attrs.get('axis', None)  # Some ops use 'axis' instead of 'dim'

    if dim is None and axis is not None:
        dim = axis

    input_tensor = iTList[0]
    assert input_tensor.check_shape(), f"Illegal Shape for {input_tensor}"

    optype_lower = op.optype.lower()

    # Element-wise operation: multiple inputs, no reduction dimension
    if dim is None and len(iTList) > 1:
        output_shape = iTList[0].shape
        for i in range(1, len(iTList)):
            assert iTList[i].check_shape(), f"Illegal Shape for input {i}"
            output_shape = bidirectional_broadcast_shape_inference(output_shape, iTList[i].shape)

        oTList[0].shape = output_shape
        oTList[0].dtype = input_tensor.dtype

        total_elems = sum(t.nelems() for t in iTList)
        out_elems = oTList[0].nelems()
        n_inputs = len(iTList)

        if optype_lower in ['max', 'min']:
            instrs = {'cmp': out_elems * (n_inputs - 1)}
        elif optype_lower == 'sum':
            instrs = {'add': out_elems * (n_inputs - 1)}
        elif optype_lower == 'mean':
            instrs = {'add': out_elems * (n_inputs - 1), 'div': out_elems}
        elif optype_lower == 'sum':
            instrs = {'add': out_elems * (n_inputs - 1)}
        else:
            raise ValueError(f"Unsupported variadic operation type: {op.optype}")

        op.perf_stats = {
            'inElems': total_elems,
            'inBytes': sum(t.nbytes(op.precision) for t in iTList),
            'outElems': out_elems,
            'outBytes': oTList[0].nbytes(op.precision),
            'instrs': instrs,
        }

        from ttsim.ops.desc.data_compute import (
            try_compute_data, compute_elementwise_max, compute_elementwise_min,
            compute_elementwise_sum, compute_elementwise_mean,
        )
        compute_map = {
            'max': compute_elementwise_max,
            'min': compute_elementwise_min,
            'sum': compute_elementwise_sum,
            'mean': compute_elementwise_mean,
        }
        oTList[0].data = try_compute_data(compute_map[optype_lower], iTList, op)

        return

    rank = input_tensor.rank()
    if dim is not None and dim < 0:
        dim += rank

    if dim is None:
        # operate on all elements, output is scalar
        output_shape = []
    else:
        assert 0 <= dim < rank, f"dim {dim} out of bounds for rank {rank}"
        # Output shape is input shape with dim removed
        output_shape = [s for i, s in enumerate(input_tensor.shape) if i != dim]

    oTList[0].shape = output_shape
    oTList[0].dtype = input_tensor.dtype

    i_elems = input_tensor.nelems()
    o_elems = oTList[0].nelems()
    optype2instr = {
            'mean': {'add': i_elems, 'div': o_elems},
            'sum' : {'add': i_elems},
            'max' : {'cmp': i_elems},  # Element-wise comparison for max
            'min' : {'cmp': i_elems},  # Element-wise comparison for min
            }
    op.perf_stats = {
        'inElems': i_elems,
        'inBytes': input_tensor.nbytes(op.precision),
        'outElems': o_elems,
        'outBytes': oTList[0].nbytes(op.precision),
        'instrs': optype2instr[op.optype.lower()],
    }
    return

def matmul_shape_inf(iTList, oTList, op, **kwargs):
    from ttsim.ops.desc.helpers import bidirectional_broadcast_shape_inference
    from .data_compute import compute_matmul

    # 1×1 conv lowered to MatMul: conv2d_pp injects kernel_shape=[1,1] into attrs.
    # Apply NCHW conv shape inference so downstream ops see [N, C_out, H_out, W_out].
    if op.attrs.get('kernel_shape') == [1, 1]:
        assert iTList[0].check_shape(), 'MatMul(1×1-conv): input shape undefined'
        assert iTList[1].check_shape(), 'MatMul(1×1-conv): weight shape undefined'
        N, C_in, H_in, W_in = iTList[0].shape
        C_out = iTList[1].shape[0]
        strides = op.attrs.get('strides', [1, 1])
        H_out = (H_in - 1) // strides[0] + 1
        W_out = (W_in - 1) // strides[1] + 1
        out_shape = [N, C_out, H_out, W_out]
        oTList[0].shape = out_shape
        oTList[0].dtype = iTList[0].dtype
        oTList[0].data = None
        from ttsim.ops.tensor import nchw_to_nhwc_flat
        oTList[0].hw_shape = nchw_to_nhwc_flat(out_shape)
        in_elems = sum(t.nelems() for t in iTList)
        in_bytes = sum(t.nbytes(op.precision) for t in iTList)
        out_elems = oTList[0].nelems()
        # Local name (not `instrs`) to avoid colliding with the matmul-path
        # ``instrs: dict = {...}`` annotation later in this function.
        instrs_1x1 = {'mac': out_elems * C_in}
        # conv2d_pp always passes a bias as 3rd input (None-bias is materialised
        # upstream), so the 1×1-conv MatMul always performs an output-shaped
        # element-wise add. Mirror the normal matmul path's `if len(iTList) > 2`
        # accounting so analytical fallback / instr-count consumers see the
        # full work.
        if len(iTList) > 2:
            instrs_1x1['add'] = out_elems
        op.perf_stats = {
            'inElems': in_elems, 'outElems': out_elems,
            'inBytes': in_bytes, 'outBytes': oTList[0].nbytes(op.precision),
            'instrs': instrs_1x1,
        }
        return

    A, B = iTList[0], iTList[1]
    assert A.check_shape(), f"Input tensor-A shape not defined: {A}"
    assert B.check_shape(), f"Input tensor-B shape not defined: {B}"

    AShape, BShape, CShape = A.shape, B.shape, None
    if len(AShape) < 1 or len(BShape) < 1:
        raise ValueError("Shapes must have at least 1 dimension")

    # Handle 1D cases
    # Note: 1D cases must be handled separately and return early to avoid executing
    # the 2D+ code path, which would try to access mat2[-2] on 1-element lists,
    # causing IndexError. Additionally, we must set reduced_dim here since mat1/mat2
    # are not defined for 1D cases.
    if len(AShape) == 1 and len(BShape) == 1:
        # Vector-Vector: [n] × [n] -> [] (scalar result)
        if AShape[0] != BShape[0]:
            raise ValueError(f"Matmul incompatible: {AShape[0]} != {BShape[0]}")
        CShape = []  # Scalar result
        reduced_dim = AShape[0]  # The dimension being reduced
    elif len(AShape) == 1:
        # Vector-Matrix: [n] × [..., n, m] -> [..., m]
        if AShape[0] != BShape[-2]:
            raise ValueError(f"Matmul incompatible: {AShape[0]} != {BShape[-2]}")
        CShape = BShape[:-2] + [BShape[-1]]
        reduced_dim = AShape[0]  # The vector length (dimension being reduced)
    elif len(BShape) == 1:
        # Matrix-Vector: [..., m, n] × [n] -> [..., m]
        if AShape[-1] != BShape[0]:
            raise ValueError(f"Matmul incompatible: {AShape[-1]} != {BShape[0]}")
        CShape = AShape[:-1]
        reduced_dim = AShape[
            -1
        ]  # The last dimension of the matrix (dimension being reduced)
    else:
        # Handle 2D+ cases
        # For 2D and higher-dimensional tensors, split into batch and matrix dimensions
        batch1, mat1 = AShape[:-2], AShape[-2:]
        batch2, mat2 = BShape[:-2], BShape[-2:]

        # Check matrix multiplication compatibility: inner dimensions must match
        if mat1[-1] != mat2[-2]:
            raise ValueError(f"Matmul incompatible: {mat1[-1]} != {mat2[-2]}")
        # Broadcast batch dimensions and compute output shape
        broadcast_batch = bidirectional_broadcast_shape_inference(batch1, batch2)
        CShape = broadcast_batch + [mat1[0], mat2[-1]]
        reduced_dim = mat1[-1]  # The inner dimension being reduced
    oTList[0].shape = CShape
    oTList[0].dtype = A.dtype

    # NUMERICAL COMPUTATION (if data available)
    if A.data is not None and B.data is not None:
        import numpy as np
        result = compute_matmul(iTList, op)
        # When ttnn.linear fuses bias as a 3rd input, add it after matmul.
        # Bias shapes are always [1, out_features] (see make_info() in the
        # workload runners), so numpy broadcasting against the matmul result
        # [B, S, out_features] is well-defined and matches linear-layer
        # semantics.  Incompatible shapes would raise a numpy ValueError,
        # never silently produce wrong results.
        if len(iTList) > 2 and iTList[2].data is not None:
            result = result + iTList[2].data

        # Apply fused activation if present (ttnn.linear activation parameter)
        fused_activation = op.attrs.get("fused_activation")
        if fused_activation is not None:
            if fused_activation == "relu":
                result = np.maximum(0, result)
            elif fused_activation == "gelu":
                try:
                    from scipy.special import erf  # type: ignore[import-untyped]
                except ImportError:
                    from math import erf as _erf
                    erf = np.vectorize(_erf)  # type: ignore[assignment]
                result = (0.5 * result * (1.0 + erf(result / np.sqrt(2.0)))).astype(result.dtype)
            elif fused_activation == "silu":
                # SiLU/Swish: x * sigmoid(x) = x / (1 + exp(-x))
                result = result / (1.0 + np.exp(-np.clip(result, -20, 20)))
            else:
                raise ValueError(f"Unsupported fused activation: {fused_activation}")

        oTList[0].data = result
    else:
        oTList[0].data = None

    # Accumulate input stats across all inputs.  Plain matmul has 2 inputs;
    # ttnn.linear fuses bias as a 3rd input on the same MatMul SimOp (see
    # op.py linear()).  Summing over iTList ensures the bias tensor's memory
    # footprint is reflected in inElems/inBytes.
    import ttsim.front.ttnn.memory as ttnn_memory
    iTList_residency = [t.memory_config().buffer_type if isinstance(t.memory_config(), ttnn_memory.MemoryConfig) else None for t in iTList]
    in_elems = sum(t.nelems() if ir != ttnn_memory.BufferType.L1 else 0 for t, ir in zip(iTList, iTList_residency))
    in_bytes = sum(t.nbytes(op.precision) if ir != ttnn_memory.BufferType.L1 else 0 for t, ir in zip(iTList, iTList_residency))
    instrs: dict = {"mac": oTList[0].nelems() * reduced_dim}
    if len(iTList) > 2:
        instrs["add"] = oTList[0].nelems()

    # Add instruction counts for fused activation (ttnn.linear activation parameter)
    fused_activation = op.attrs.get("fused_activation")
    if fused_activation is not None:
        from ttsim.ops.desc.helpers import gelu_instr_counts
        nelem = oTList[0].nelems()
        if fused_activation == "relu":
            instrs["cmp"] = instrs.get("cmp", 0) + nelem
            instrs["mov"] = instrs.get("mov", 0) + nelem
        elif fused_activation == "gelu":
            gelu_instrs = gelu_instr_counts(nelem)
            for key, val in gelu_instrs.items():
                instrs[key] = instrs.get(key, 0) + val
        elif fused_activation == "silu":
            # SiLU = x * sigmoid(x): sigmoid (exp + add + div) + multiply
            instrs["exp"] = instrs.get("exp", 0) + nelem
            instrs["add"] = instrs.get("add", 0) + nelem
            instrs["div"] = instrs.get("div", 0) + nelem
            instrs["mul"] = instrs.get("mul", 0) + nelem

    oTList_residency = oTList[0].memory_config().buffer_type if isinstance(oTList[0].memory_config(), ttnn_memory.MemoryConfig) else None
    op.perf_stats = {
        "inElems": in_elems,
        "outElems": oTList[0].nelems() if oTList_residency != ttnn_memory.BufferType.L1 else 0,
        "inBytes": in_bytes,
        "outBytes": oTList[0].nbytes(op.precision) if oTList_residency != ttnn_memory.BufferType.L1 else 0,
        "instrs": instrs,
    }
    return


def grid_sample_inf(iTList, oTList, op, **kwargs):
    A = iTList[0]
    B = iTList[1]
    assert A.check_shape(), f"Input tensor-A shape not defined: {A}"
    assert B.check_shape(), f"Input tensor-B shape not defined: {B}"
    AShape = iTList[0].shape
    BShape = iTList[1].shape
    if len(AShape) != 4:
        raise ValueError(f"GridSample expects 4D input tensor-A shape; got {AShape}")
    if len(BShape) != 4:
        raise ValueError(f"GridSample expects 4D input tensor-B shape; got {BShape}")
    N, C, H, W = AShape
    N_grid, H_out, W_out, _ = BShape
    if N != N_grid:
        raise ValueError(f"GridSample batch-size mismatch: {N} != {N_grid}")
    C_out = C
    oTList[0].shape = [N, C_out, H_out, W_out]
    oTList[0].dtype = A.dtype

    mode = op.attrs.get("mode", "bilinear")
    nElem_out = N * C_out * H_out * W_out
    instr_count = {}
    if mode == "nearest":
        instr_count["mov"] = nElem_out
    elif mode == "bilinear":
        instr_count["mul"] = nElem_out * 4
        instr_count["add"] = nElem_out * 3
    elif mode == "bicubic":
        instr_count["mul"] = nElem_out * 16
        instr_count["add"] = nElem_out * 15
    else:
        raise ValueError(f"Unsupported GridSample mode: {mode}")

    op.perf_stats = {
        "inElems": iTList[0].nelems() + iTList[1].nelems(),
        "outElems": oTList[0].nelems(),
        "inBytes": iTList[0].nbytes(op.precision) + iTList[1].nbytes(op.precision),
        "outBytes": oTList[0].nbytes(op.precision),
        "instrs": {"mov": oTList[0].nelems()},  # TODO: refine this
    }
    return


def expand_sinf(iTList, oTList, op, **kwargs):
    A = iTList[0]
    AShape = iTList[0].shape

    if iTList[1].data is None:
        target_shape = AShape
    else:
        B = iTList[1].clone_by_shape(data_maybe_missing=False) #B.data should exist
        assert A.check_shape(), f"Input tensor-A shape not defined: {A}"
        assert B.check_shape(), f"Input tensor-B shape not defined: {B}"
        assert B.dtype == np.int64, f"Input Data-Type should be np.int64 {B}"
        target_shape = [x.item() for x in B.data]

    CShape       = bidirectional_broadcast_shape_inference(AShape, target_shape)
    oTList[0].shape = CShape
    oTList[0].dtype = A.dtype
    op.perf_stats = {
            'inElems' : iTList[0].nelems() + iTList[1].nelems(),
            'outElems': oTList[0].nelems(),
            'inBytes' : iTList[0].nbytes(op.precision) + iTList[1].nbytes(op.precision),
            'outBytes': oTList[0].nbytes(op.precision),
            'instrs'  : {'mov': oTList[0].nelems()}
            }
    return


def det_sinf(iTList, oTList, op, **kwargs):
    A = iTList[0]
    assert A.check_shape(), f"Input tensor-A shape not defined: {A}"
    AShape = iTList[0].shape
    ARank  = iTList[0].rank()
    if ARank < 2:
        raise ValueError("Det expects at least 2D input of shape [*, M, M]")

    if AShape[-2] != AShape[-1]:
        raise ValueError(f"Det expects [*,M,M] tensor shapes; got {AShape}")

    M = AShape[-1]
    oTList[0].shape = AShape[:-2]
    oTList[0].dtype = A.dtype
    op.perf_stats = {
            'inElems' : iTList[0].nelems(),
            'outElems': oTList[0].nelems(),
            'inBytes' : iTList[0].nbytes(op.precision),
            'outBytes': oTList[0].nbytes(op.precision),
            'instrs'  : {'mac': M**3/3, 'div': M**2/2}, #assume LU based algo for determinant
            }
    return


def data_window_sinf(iTList, oTList, op, **kwargs):
    A = iTList[0].clone_by_shape(data_maybe_missing=False)
    if iTList[0].rank() != 0:
        raise ValueError("data windows expect a scalar as input: {A}")
    dtype = op.attrs['output_datatype']
    oTList[0].shape = [x.item() for x in A.data]
    oTList[0].dtype = op.precision
    op.perf_stats = {
            'inElems' : iTList[0].nelems(),
            'outElems': oTList[0].nelems(),
            'inBytes' : iTList[0].nbytes(op.precision),
            'outBytes': oTList[0].nbytes(op.precision),
            'instrs'  : {'mov': 0} #TODO: fix this for HammingWindow, HannWindow, BlackmanWindow costs
            }
    return


# ---------------------------------------------------------------------------
# TopK compute model for the tt-metal TopKDeviceOperation (tiled input).
#
# Structure comes from the compute kernel (ttnn/.../reduction/topk/device/kernels/compute/topk.cpp):
#   - input is [rows, W] in 32x32 tiles: Ht = ceil(rows/32) tile rows, Wt = ceil(W/32) tiles per row
#   - per tile row it is an insertion sort: the first iteration takes 2 tiles, every later one takes 1,
#     so Wt-1 merges; each merge runs `output_tiles = ceil(k/32)` topk_local_sort calls (steady state)
#   - topk_local_sort(end_phase=5) is a full 64-element bitonic sort per column: phases 0..5 =
#     1+2+...+6 = 21 compare-exchange layers; 64 elems/column x 32 columns = 1024 pairs per layer
#   - tile rows are split over cores, so a single tile row (Ht=1, i.e. <=32 users) runs on ONE core
#     (capture: llama3-8B p100a decode, CORE COUNT = 1)
# The vector-pipe instructions per compare-exchange are calibrated, not derived: _TOPK_VEC_OPS_PER_CE = 6
# (swap of value + index = 2 `cmp`, load/store of value + index to DST = 4 `mov`) reproduces the capture
# (13.50M cycles measured for 2003 merges) within ~0.5% at the default DG_COMPUTE_UTIL_CONSTANT = 0.6.
# Not modeled: the multi-core split-width variant, and the first-iterations ramp for k > 32.
# ---------------------------------------------------------------------------
_TOPK_TILE = 32
_TOPK_CE_LAYERS_PER_MERGE = 21                       # 1+2+3+4+5+6 bitonic layers for 64 elements
_TOPK_CE_PER_LAYER = _TOPK_TILE * _TOPK_TILE         # 32 columns x 32 pairs
_TOPK_CMP_PER_CE = 2                                 # value swap + index swap
_TOPK_MOV_PER_CE = 4                                 # DST load/store of value and of index


def _is_tt_tiled_topk(op, X) -> bool:
    """True for the real-ttnn TopK (k recorded by topk_pp) on a TILE_LAYOUT input; other callers
    (ONNX-style TopK) keep the legacy memory-only model."""
    layout = getattr(getattr(X, 'layout', None), 'name', '')
    return layout == 'TILE_LAYOUT' and op.attrs.get('k') is not None


def _tt_topk_compute(in_shape: list, axis: int, k: int):
    """Returns (instrs, active_cores) for TopKDeviceOperation over `axis` == last dim."""
    width = int(in_shape[axis])
    rows = 1
    for i, d in enumerate(in_shape):
        if i != axis:
            rows *= int(d)
    ht = -(-rows // _TOPK_TILE)
    wt = -(-width // _TOPK_TILE)
    output_tiles = -(-int(k) // _TOPK_TILE)
    merges = ht * max(wt - 1, 0) * output_tiles
    ce = merges * _TOPK_CE_LAYERS_PER_MERGE * _TOPK_CE_PER_LAYER
    return {'cmp': ce * _TOPK_CMP_PER_CE, 'mov': ce * _TOPK_MOV_PER_CE}, ht


def topk_sinf(iTList, oTList, op, **kwargs):
    X = iTList[0]
    assert X.check_shape(), f"Input tensor-X shape not defined: {X}"
    XShape = list(X.shape)  # plain list copy — avoids mutating X.shape via shared Shape._shape
    XRank  = X.rank()
    _axis   : int = op.attrs.get('axis',   -1)
    _largest: int = op.attrs.get('largest', 1)
    _sorted : int = op.attrs.get('sorted',  1)

    if XRank < 1:
        raise ValueError("TopK expects input of rank >= 1: {X}")

    if _axis < 0: #type: ignore
        _axis = XRank + _axis #type: ignore

    if _axis < 0 or _axis >= XRank:
        raise ValueError(f"Axis {_axis} is out of bounds for X {X}")

    XShape_in = list(XShape)  # input shape, kept before XShape is reused as the output shape
    outshape = XShape
    d_axis   = XShape[_axis]
    # K may come from a second input K-tensor (ONNX-style, a scalar int64) or from the 'k' attr
    # (real ttnn: ttnn.topk(x, k=32, ...) — topk_pp records k in attrs). Only treat input_1 as the
    # K-tensor when it is a scalar (nelems==1); the real-ttnn second input is a large uint16
    # indices_tensor operand, which must NOT be read as K.
    if len(iTList) >= 2 and iTList[1].data is not None and iTList[1].nelems() == 1:
        K = iTList[1].clone_by_shape(data_maybe_missing=False)
        assert K.dtype == np.int64, f"Input tensor-K Data-Type should be np.int64 {K}"
        k_value = [x.item() for x in K.data]
        assert len(k_value) == 1, "TopK requires K-Tensor should be 1D with a single scalar value"
        k_scalar_value = k_value[0]
    else:
        k_scalar_value = int(op.attrs.get('k', 1))
    if k_scalar_value < 0:
        raise ValueError(f"TopK requires K value > 0: {k_scalar_value}")
    if k_scalar_value > d_axis:
        raise ValueError(f"TopK requires K value({k_scalar_value}) < dim(axis) = {d_axis}")
    outshape[_axis] = k_scalar_value

    oTList[0].shape = outshape
    oTList[1].shape = outshape
    oTList[0].dtype = X.dtype
    oTList[1].dtype = np.dtype(np.int64)
    in_elems = sum(t.nelems() for t in iTList)
    instrs: dict = {'mov': 0}
    active_cores = None
    if _is_tt_tiled_topk(op, X):
        instrs, active_cores = _tt_topk_compute(XShape_in, _axis, k_scalar_value)
    op.perf_stats = {
            'inElems' : in_elems,
            'outElems': oTList[0].nelems() + oTList[1].nelems(),
            'inBytes' : sum(t.nbytes(op.precision) for t in iTList),
            'outBytes': oTList[0].nbytes(op.precision) + oTList[1].nbytes(op.precision),
            'instrs'  : instrs,
            }
    if active_cores is not None:
        op.perf_stats['activeCores'] = active_cores
    return


def nonzero_sinf(iTList, oTList, op, **kwargs):
    A = iTList[0]
    assert A.check_shape(), f"Input tensor-A shape not defined: {A}"
    AElems = A.nelems()
    nonzero_count = (
        AElems // 2
    )  # Assume half the elements are non-zero for perf estimation

    rank = A.rank()
    out_shape = [rank, nonzero_count]
    oTList[0].shape = out_shape
    oTList[0].dtype = np.dtype(np.int64)

    op.perf_stats = {
        "inElems": iTList[0].nelems(),
        "outElems": oTList[0].nelems(),
        "inBytes": iTList[0].nbytes(op.precision),
        "outBytes": oTList[0].nbytes(op.precision),
        "instrs": {"cmp": AElems, "mov": nonzero_count * rank},
    }
    return


def gemm_sinf(iTList, oTList, op, **kwargs):
    A = iTList[0]
    B = iTList[1]
    assert A.check_shape(), f"Input tensor-A shape not defined: {A}"
    assert B.check_shape(), f"Input tensor-B shape not defined: {B}"
    transA = op.attrs.get("transA", 0)
    transB = op.attrs.get("transB", 0)
    AShape, BShape, CShape = A.shape, B.shape, None
    if transA:
        AShape = AShape[:-2] + [AShape[-1], AShape[-2]]
    if transB:
        BShape = BShape[:-2] + [BShape[-1], BShape[-2]]

    if len(AShape) < 2 or len(BShape) < 2:
        raise ValueError("Shapes must have at least 2 dimensions")

    batch1, mat1 = AShape[:-2], AShape[-2:]
    batch2, mat2 = BShape[:-2], BShape[-2:]

    # Check matrix multiplication compatibility
    if mat1[-1] != mat2[-2]:
        raise ValueError(f"Gemm incompatible: {mat1[-1]} != {mat2[-2]}")
    broadcast_batch = bidirectional_broadcast_shape_inference(batch1, batch2)
    CShape = broadcast_batch + [mat1[-2], mat2[-1]]

    reduced_dim = mat1[-1]
    oTList[0].shape = CShape
    oTList[0].dtype = A.dtype
    op.perf_stats = {
        "inElems": iTList[0].nelems() + iTList[1].nelems(),
        "outElems": oTList[0].nelems(),
        "inBytes": iTList[0].nbytes(op.precision) + iTList[1].nbytes(op.precision),
        "outBytes": oTList[0].nbytes(op.precision),
        "instrs": {"mac": oTList[0].nelems() * reduced_dim},
    }
    return


def register_math_ops():
    _xoptbl = [
        [
            "Clip",
            "ARITY_VARIADIC[1-3]->1",
            "ai.onnx",
            "COMMON",
            13,
            13,
            3,
            1,
            1,
            1,
            unary_fwd,
            True,
            True,
            True,
            True,
            True,
        ],
        [
            "TopK",
            "ARITY_VARIADIC[1-2]->2",
            "ai.onnx",
            "COMMON",
            24,
            11,
            2,
            1,
            2,
            2,
            topk_sinf,
            True,
            True,
            True,
            True,
            True,
        ],
        [
            "BinaryCrossEntropyWithLogits",
            "ARITY_2->1",
            "ai.onnx",
            "COMMON",
            22,
            22,
            2,
            2,
            1,
            1,
            binary_cross_entropy_with_logits_sinf,
            True,
            True,
            True,
            True,
            True,
        ],
        [
            "SoftmaxCrossEntropyLoss",
            "ARITY_VARIADIC[2-3]->VARIADIC[1-2]",
            "ai.onnx",
            "COMMON",
            13,
            13,
            3,
            2,
            2,
            1,
            softmax_cross_entropy_loss_sinf,
            True,
            True,
            True,
            True,
            True,
        ],
        [
            "Einsum",
            "ARITY_VARIADIC[1-*]->1",
            "ai.onnx",
            "COMMON",
            12,
            12,
            2147483647,
            1,
            1,
            1,
            einsum_sinf,
            True,
            True,
            True,
            True,
            True,
        ],
        [
            "Cdist",
            "ARITY_2->1",
            "ai.onnx",
            "COMMON",
            22,
            22,
            2,
            2,
            1,
            1,
            cdist_sinf,
            True,
            True,
            True,
            True,
            True,
        ],
        # complex shape inference
        # ['NegativeLogLikelihoodLoss', 'ARITY_VARIADIC[2-3]->1',              'ai.onnx', 'COMMON',  22,  22,  3,  2,  1,  1,  'inline_lambda',  True,  True,  True,  True,  True],
        # ['SoftmaxCrossEntropyLoss',   'ARITY_VARIADIC[2-3]->VARIADIC[1-2]',  'ai.onnx', 'COMMON',  13,  13,  3,  2,  2,  1,  'inline_lambda',  True,  True,  True,  True,  True],
        # ['Einsum',           'ARITY_VARIADIC[1-*]->1', 'ai.onnx', 'COMMON',  12,  12,  2147483647,  1,  1,  1,  'inline_lambda',  True,  True,  True,  True,  True],
        # ['DFT',              'ARITY_VARIADIC[1-3]->1', 'ai.onnx', 'COMMON',  20,  20,  3,  1,  1,  1,  'inline_lambda',  True,  True,  True,  True,  True],
        [
            "Gemm",
            "ARITY_VARIADIC[2-3]->1",
            "ai.onnx",
            "COMMON",
            13,
            13,
            3,
            2,
            1,
            1,
            gemm_sinf,
            True,
            True,
            True,
            True,
            True,
        ],
        # ['MatMulInteger',    'ARITY_VARIADIC[2-4]->1', 'ai.onnx', 'COMMON',  10,  10,  4,  2,  1,  1,  'inline_lambda',  True,  True,  True,  True,  True],
        # ['STFT',             'ARITY_VARIADIC[2-4]->1', 'ai.onnx', 'COMMON',  17,  17,  4,  2,  1,  1,  'inline_lambda',  True,  True,  True,  True,  True],
        # ['QLinearMatMul',    'ARITY_8->1',             'ai.onnx', 'COMMON',  21,  21,  8,  8,  1,  1,  'QLinearMatMulShapeInference',  True,  True,  True,  True,  True],
        # ['MelWeightMatrix',  'ARITY_5->1',             'ai.onnx', 'COMMON',  17,  17,  5,  5,  1,  1,  'inline_lambda', True,  True,  True,  True,  True],
    ]

    _binary_optbl = [
        [
            "Add",
            "ARITY_2->1",
            "ai.onnx",
            "COMMON",
            14,
            14,
            2,
            2,
            1,
            1,
            bidir_bcast,
            True,
            True,
            True,
            True,
            True,
        ],
        [
            "Sub",
            "ARITY_2->1",
            "ai.onnx",
            "COMMON",
            14,
            14,
            2,
            2,
            1,
            1,
            bidir_bcast,
            True,
            True,
            True,
            True,
            True,
        ],
        [
            "Mul",
            "ARITY_2->1",
            "ai.onnx",
            "COMMON",
            14,
            14,
            2,
            2,
            1,
            1,
            bidir_bcast,
            True,
            True,
            True,
            True,
            True,
        ],
        [
            "Div",
            "ARITY_2->1",
            "ai.onnx",
            "COMMON",
            14,
            14,
            2,
            2,
            1,
            1,
            bidir_bcast,
            True,
            True,
            True,
            True,
            True,
        ],
        [
            "Pow",
            "ARITY_2->1",
            "ai.onnx",
            "COMMON",
            15,
            15,
            2,
            2,
            1,
            1,
            bidir_bcast,
            True,
            True,
            True,
            True,
            True,
        ],
        [
            "Mod",
            "ARITY_2->1",
            "ai.onnx",
            "COMMON",
            13,
            13,
            2,
            2,
            1,
            1,
            bidir_bcast,
            True,
            True,
            True,
            True,
            True,
        ],
        [
            "Expand",
            "ARITY_2->1",
            "ai.onnx",
            "COMMON",
            13,
            13,
            2,
            2,
            1,
            1,
            expand_sinf,
            True,
            True,
            True,
            True,
            True,
        ],
        [
            "CumSum",
            "ARITY_2->1",
            "ai.onnx",
            "COMMON",
            14,
            14,
            2,
            2,
            1,
            1,
            unary_fwd,
            True,
            True,
            True,
            True,
            True,
        ],
        [
            "PRelu",
            "ARITY_2->1",
            "ai.onnx",
            "COMMON",
            16,
            16,
            2,
            2,
            1,
            1,
            unary_fwd,
            True,
            True,
            True,
            True,
            True,
        ],
        # max_input=3 (was 2): ttnn.linear fuses bias as a 3rd input on the
        # same MatMul SimOp, so the op must accept 2 or 3 inputs.
        [
            "MatMul",
            "ARITY_VARIADIC[2-3]->1",
            "ai.onnx",
            "COMMON",
            13,
            13,
            3,
            2,
            1,
            1,
            matmul_shape_inf,
            True,
            True,
            True,
            True,
            True,
        ],
        [
            "GridSample",
            "ARITY_2->1",
            "ai.onnx",
            "COMMON",
            22,
            22,
            2,
            2,
            1,
            1,
            grid_sample_inf,
            True,
            True,
            True,
            True,
            True,
        ],
        [
            "Atan2",
            "ARITY_2->1",
            "ai.onnx",
            "COMMON",
            22,
            22,
            2,
            2,
            1,
            1,
            bidir_bcast,
            True,
            True,
            True,
            True,
            True,
        ],
    ]

    _variadic_input_unary_output_optbl = [
        [
            "Max",
            "ARITY_VARIADIC[1-*]->1",
            "ai.onnx",
            "COMMON",
            13,
            13,
            2147483647,
            1,
            1,
            1,
            variadic_sinf,
            True,
            True,
            True,
            True,
            True,
        ],
        [
            "Min",
            "ARITY_VARIADIC[1-*]->1",
            "ai.onnx",
            "COMMON",
            13,
            13,
            2147483647,
            1,
            1,
            1,
            variadic_sinf,
            True,
            True,
            True,
            True,
            True,
        ],
        [
            "Sum",
            "ARITY_VARIADIC[1-*]->1",
            "ai.onnx",
            "COMMON",
            13,
            13,
            2147483647,
            1,
            1,
            1,
            variadic_sinf,
            True,
            True,
            True,
            True,
            True,
        ],
        [
            "Mean",
            "ARITY_VARIADIC[1-*]->1",
            "ai.onnx",
            "COMMON",
            13,
            13,
            2147483647,
            1,
            1,
            1,
            variadic_sinf,
            True,
            True,
            True,
            True,
            True,
        ],
    ]

    _unary_optbl = [
        [
            "Softmax",
            "ARITY_1->1",
            "ai.onnx",
            "COMMON",
            13,
            13,
            1,
            1,
            1,
            1,
            unary_fwd,
            True,
            True,
            True,
            True,
            True,
        ],
        [
            "LogSoftmax",
            "ARITY_1->1",
            "ai.onnx",
            "COMMON",
            13,
            13,
            1,
            1,
            1,
            1,
            unary_fwd,
            True,
            True,
            True,
            True,
            True,
        ],
        [
            "Hardmax",
            "ARITY_1->1",
            "ai.onnx",
            "COMMON",
            13,
            13,
            1,
            1,
            1,
            1,
            unary_fwd,
            True,
            True,
            True,
            True,
            True,
        ],
        [
            "Neg",
            "ARITY_1->1",
            "ai.onnx",
            "COMMON",
            13,
            13,
            1,
            1,
            1,
            1,
            unary_fwd,
            True,
            True,
            True,
            True,
            True,
        ],
        [
            "Abs",
            "ARITY_1->1",
            "ai.onnx",
            "COMMON",
            13,
            13,
            1,
            1,
            1,
            1,
            unary_fwd,
            True,
            True,
            True,
            True,
            True,
        ],
        [
            "Reciprocal",
            "ARITY_1->1",
            "ai.onnx",
            "COMMON",
            13,
            13,
            1,
            1,
            1,
            1,
            unary_fwd,
            True,
            True,
            True,
            True,
            True,
        ],
        [
            "Floor",
            "ARITY_1->1",
            "ai.onnx",
            "COMMON",
            13,
            13,
            1,
            1,
            1,
            1,
            unary_fwd,
            True,
            True,
            True,
            True,
            True,
        ],
        [
            "Ceil",
            "ARITY_1->1",
            "ai.onnx",
            "COMMON",
            13,
            13,
            1,
            1,
            1,
            1,
            unary_fwd,
            True,
            True,
            True,
            True,
            True,
        ],
        [
            "Sqrt",
            "ARITY_1->1",
            "ai.onnx",
            "COMMON",
            13,
            13,
            1,
            1,
            1,
            1,
            unary_fwd,
            True,
            True,
            True,
            True,
            True,
        ],
        [
            "Relu",
            "ARITY_1->1",
            "ai.onnx",
            "COMMON",
            14,
            14,
            1,
            1,
            1,
            1,
            unary_fwd,
            True,
            True,
            True,
            True,
            True,
        ],
        [
            "LeakyRelu",
            "ARITY_1->1",
            "ai.onnx",
            "COMMON",
            16,
            16,
            1,
            1,
            1,
            1,
            unary_fwd,
            True,
            True,
            True,
            True,
            True,
        ],
        [
            "ThresholdedRelu",
            "ARITY_1->1",
            "ai.onnx",
            "COMMON",
            22,
            22,
            1,
            1,
            1,
            1,
            unary_fwd,
            True,
            True,
            True,
            True,
            True,
        ],
        [
            "Selu",
            "ARITY_1->1",
            "ai.onnx",
            "COMMON",
            22,
            22,
            1,
            1,
            1,
            1,
            unary_fwd,
            True,
            True,
            True,
            True,
            True,
        ],
        [
            "Elu",
            "ARITY_1->1",
            "ai.onnx",
            "COMMON",
            22,
            22,
            1,
            1,
            1,
            1,
            unary_fwd,
            True,
            True,
            True,
            True,
            True,
        ],
        [
            "Mish",
            "ARITY_1->1",
            "ai.onnx",
            "COMMON",
            22,
            22,
            1,
            1,
            1,
            1,
            unary_fwd,
            True,
            True,
            True,
            True,
            True,
        ],
        [
            "Celu",
            "ARITY_1->1",
            "ai.onnx",
            "COMMON",
            12,
            12,
            1,
            1,
            1,
            1,
            unary_fwd,
            True,
            True,
            True,
            True,
            True,
        ],
        [
            "Gelu",
            "ARITY_1->1",
            "ai.onnx",
            "COMMON",
            20,
            20,
            1,
            1,
            1,
            1,
            unary_fwd,
            True,
            True,
            True,
            True,
            True,
        ],
        [
            "Glu",
            "ARITY_1->1",
            "ai.onnx",
            "COMMON",
            22,
            22,
            1,
            1,
            1,
            1,
            glu_sinf,
            True,
            True,
            True,
            True,
            True,
        ],
        [
            "Exp",
            "ARITY_1->1",
            "ai.onnx",
            "COMMON",
            13,
            13,
            1,
            1,
            1,
            1,
            unary_fwd,
            True,
            True,
            True,
            True,
            True,
        ],
        [
            "Log",
            "ARITY_1->1",
            "ai.onnx",
            "COMMON",
            13,
            13,
            1,
            1,
            1,
            1,
            unary_fwd,
            True,
            True,
            True,
            True,
            True,
        ],
        [
            "Tanh",
            "ARITY_1->1",
            "ai.onnx",
            "COMMON",
            13,
            13,
            1,
            1,
            1,
            1,
            unary_fwd,
            True,
            True,
            True,
            True,
            True,
        ],
        [
            "Sigmoid",
            "ARITY_1->1",
            "ai.onnx",
            "COMMON",
            13,
            13,
            1,
            1,
            1,
            1,
            unary_fwd,
            True,
            True,
            True,
            True,
            True,
        ],
        [
            "HardSigmoid",
            "ARITY_1->1",
            "ai.onnx",
            "COMMON",
            22,
            22,
            1,
            1,
            1,
            1,
            unary_fwd,
            True,
            True,
            True,
            True,
            True,
        ],
        [
            "HardSwish",
            "ARITY_1->1",
            "ai.onnx",
            "COMMON",
            22,
            22,
            1,
            1,
            1,
            1,
            unary_fwd,
            True,
            True,
            True,
            True,
            True,
        ],
        [
            "Softsign",
            "ARITY_1->1",
            "ai.onnx",
            "COMMON",
            22,
            22,
            1,
            1,
            1,
            1,
            unary_fwd,
            True,
            True,
            True,
            True,
            True,
        ],
        [
            "Softplus",
            "ARITY_1->1",
            "ai.onnx",
            "COMMON",
            22,
            22,
            1,
            1,
            1,
            1,
            unary_fwd,
            True,
            True,
            True,
            True,
            True,
        ],
        [
            "Sin",
            "ARITY_1->1",
            "ai.onnx",
            "COMMON",
            22,
            22,
            1,
            1,
            1,
            1,
            unary_fwd,
            True,
            True,
            True,
            True,
            True,
        ],
        [
            "Cos",
            "ARITY_1->1",
            "ai.onnx",
            "COMMON",
            22,
            22,
            1,
            1,
            1,
            1,
            unary_fwd,
            True,
            True,
            True,
            True,
            True,
        ],
        [
            "Tan",
            "ARITY_1->1",
            "ai.onnx",
            "COMMON",
            22,
            22,
            1,
            1,
            1,
            1,
            unary_fwd,
            True,
            True,
            True,
            True,
            True,
        ],
        [
            "Asin",
            "ARITY_1->1",
            "ai.onnx",
            "COMMON",
            22,
            22,
            1,
            1,
            1,
            1,
            unary_fwd,
            True,
            True,
            True,
            True,
            True,
        ],
        [
            "Acos",
            "ARITY_1->1",
            "ai.onnx",
            "COMMON",
            22,
            22,
            1,
            1,
            1,
            1,
            unary_fwd,
            True,
            True,
            True,
            True,
            True,
        ],
        [
            "Atan",
            "ARITY_1->1",
            "ai.onnx",
            "COMMON",
            22,
            22,
            1,
            1,
            1,
            1,
            unary_fwd,
            True,
            True,
            True,
            True,
            True,
        ],
        [
            "Sinh",
            "ARITY_1->1",
            "ai.onnx",
            "COMMON",
            22,
            22,
            1,
            1,
            1,
            1,
            unary_fwd,
            True,
            True,
            True,
            True,
            True,
        ],
        [
            "Cosh",
            "ARITY_1->1",
            "ai.onnx",
            "COMMON",
            22,
            22,
            1,
            1,
            1,
            1,
            unary_fwd,
            True,
            True,
            True,
            True,
            True,
        ],
        [
            "Asinh",
            "ARITY_1->1",
            "ai.onnx",
            "COMMON",
            22,
            22,
            1,
            1,
            1,
            1,
            unary_fwd,
            True,
            True,
            True,
            True,
            True,
        ],
        [
            "Acosh",
            "ARITY_1->1",
            "ai.onnx",
            "COMMON",
            22,
            22,
            1,
            1,
            1,
            1,
            unary_fwd,
            True,
            True,
            True,
            True,
            True,
        ],
        [
            "Atanh",
            "ARITY_1->1",
            "ai.onnx",
            "COMMON",
            22,
            22,
            1,
            1,
            1,
            1,
            unary_fwd,
            True,
            True,
            True,
            True,
            True,
        ],
        [
            "Sign",
            "ARITY_1->1",
            "ai.onnx",
            "COMMON",
            13,
            13,
            1,
            1,
            1,
            1,
            unary_fwd,
            True,
            True,
            True,
            True,
            True,
        ],
        [
            "Erf",
            "ARITY_1->1",
            "ai.onnx",
            "COMMON",
            13,
            13,
            1,
            1,
            1,
            1,
            unary_fwd,
            True,
            True,
            True,
            True,
            True,
        ],
        [
            "Round",
            "ARITY_1->1",
            "ai.onnx",
            "COMMON",
            22,
            22,
            1,
            1,
            1,
            1,
            unary_fwd,
            True,
            True,
            True,
            True,
            True,
        ],
        [
            "Det",
            "ARITY_1->1",
            "ai.onnx",
            "COMMON",
            22,
            22,
            1,
            1,
            1,
            1,
            det_sinf,
            True,
            True,
            True,
            True,
            True,
        ],
        [
            "NonZero",
            "ARITY_1->1",
            "ai.onnx",
            "COMMON",
            13,
            13,
            1,
            1,
            1,
            1,
            nonzero_sinf,
            True,
            True,
            True,
            True,
            True,
        ],
        [
            "Diag",
            "ARITY_1->1",
            "ai.onnx",
            "COMMON",
            22,
            22,
            1,
            1,
            1,
            1,
            diag_sinf,
            True,
            True,
            True,
            True,
            True,
        ],
        [
            "HannWindow",
            "ARITY_1->1",
            "ai.onnx",
            "COMMON",
            17,
            17,
            1,
            1,
            1,
            1,
            data_window_sinf,
            True,
            True,
            True,
            True,
            True,
        ],
        [
            "HammingWindow",
            "ARITY_1->1",
            "ai.onnx",
            "COMMON",
            17,
            17,
            1,
            1,
            1,
            1,
            data_window_sinf,
            True,
            True,
            True,
            True,
            True,
        ],
        [
            "BlackmanWindow",
            "ARITY_1->1",
            "ai.onnx",
            "COMMON",
            17,
            17,
            1,
            1,
            1,
            1,
            data_window_sinf,
            True,
            True,
            True,
            True,
            True,
        ],
        [
            "InverseSigmoid",
            "ARITY_1->1",
            "ai.onnx",
            "COMMON",
            22,
            22,
            1,
            1,
            1,
            1,
            inverse_sigmoid_sinf,
            True,
            True,
            True,
            True,
            True,
        ],
    ]

    _optbl = _xoptbl + _binary_optbl + _variadic_input_unary_output_optbl + _unary_optbl
    register_ops("math", _optbl)
    return
