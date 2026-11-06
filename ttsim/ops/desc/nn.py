#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from ttsim.ops.desc.registry import register_ops
from ttsim.ops.desc.helpers import unary_fwd, pooling_shape_inference
from ttsim.utils.common import prod_ints
import numpy as np
from .data_compute import compute_grid_sample, compute_dropout


def grid_sample_fwd(iTList, oTList, op, **kwargs):
    """
    Forward pass for grid_sample operation.
    Handles both shape inference and numerical computation in one function.
    Args:
        iTList: [input_tensor, grid_tensor]
            input_tensor: [N, C, H_in, W_in]
            grid_tensor: [N, H_out, W_out, 2]
        oTList: [output_tensor]
        op: SimOp with attrs:
            - mode: 'bilinear' or 'nearest'
            - padding_mode: 'zeros', 'border', 'reflection'
            - align_corners: bool
    Returns:
        output_tensor with shape [N, C, H_out, W_out]
    """
    input_t, grid_t = iTList[0], iTList[1]
    output_t = oTList[0]
    # SHAPE INFERENCE (always)
    N_in, C, H_in, W_in = input_t.shape
    N_grid, H_out, W_out, _ = grid_t.shape
    # Validate batch sizes match
    if N_in != N_grid:
        raise ValueError(f"Batch size mismatch: input N={N_in}, grid N={N_grid}")
    # Output shape: same N and C as input, spatial dims from grid
    output_t.shape = [N_in, C, H_out, W_out]
    output_t.dtype = input_t.dtype

    # NUMERICAL COMPUTATION (only if data available)
    if input_t.data is not None and grid_t.data is not None:
        # Call numerical implementation
        output_t.data = compute_grid_sample(iTList, op)
    else:
        # Shape inference mode
        output_t.data = None

    return output_t


"""
Linear layer operation for neural networks.
"""


def linear_fwd(iTList, oTList, op, **kwargs):
    """
    Forward pass for Linear (fully connected) layer.
    Computes: y = xW^T + b

    Args:
        iTList: [input, weight] or [input, weight, bias]
            input: [..., in_features]
            weight: [out_features, in_features]
            bias: [out_features] (optional)
        oTList: [output]
        op: SimOp
    """
    from .data_compute import compute_linear

    input_t = iTList[0]
    weight_t = iTList[1]
    output_t = oTList[0]

    # Shape inference
    input_shape = list(input_t.shape)
    out_features = weight_t.shape[0]
    output_t.shape = input_shape[:-1] + [out_features]
    output_t.dtype = input_t.dtype

    # Numerical computation
    if input_t.data is not None and weight_t.data is not None:
        output_t.data = compute_linear(iTList, op)
    else:
        output_t.data = None

    return output_t


def dropout_sinf(iTList, oTList, op, **kwargs):
    # Spec: https://github.com/onnx/onnx/blob/main/docs/Operators.md#Dropout
    # with train_modeB as True, oTList is a random dropout
    # ratio is same as drop_probability
    # oTList = scale * dataT * maskT, where scale = 1./(1-ratio).
    seed = op.attrs.get("seed", 1.0)
    X = iTList[0]

    inBytes = X.nbytes(op.precision)
    inElems = X.nelems()

    # Priority 1: Check op.attrs for 'prob' and 'train_mode' (set by F.Dropout)
    # Priority 2: Fall back to iTList for ONNX-style inputs
    # Priority 3: Use safe defaults (no dropout)
    ratio = op.attrs.get("prob", None)
    training_mode = op.attrs.get("train_mode", None)

    # If not in attrs, try to get from iTList (ONNX-style)
    if ratio is None:
        if len(iTList) >= 2 and iTList[1].data is not None:
            ratio = float(iTList[1].data)
            inBytes += iTList[1].dtype.itemsize
            inElems += 1
        else:
            ratio = 0.0  # Default: no dropout (safer default)

    if training_mode is None:
        if len(iTList) >= 3 and iTList[2].data is not None:
            training_mode = bool(iTList[2].data)
            inBytes += iTList[2].dtype.itemsize
            inElems += 1
        else:
            training_mode = False  # Default: inference mode

    # Compute numerical output if input data is available
    if X.data is not None:
        np_out, np_mask_out = compute_dropout(iTList, op)
        oTList[0].data = np_out.astype(X.dtype)

        return_mask = True if len(oTList) == 2 else False
        if return_mask:
            oTList[1].data = np_mask_out

    # Instruction count for hardware cost estimation
    # No dropout if not training or ratio <= 0
    if not training_mode or ratio <= 0:
        instr_count = {"mov": 0}
    else:
        instr_count = {
            "mov": X.nelems(),  # mask
            "mul": X.nelems(),  # mask * x * scale
        }

    oTList[0].shape = X.shape
    oTList[0].dtype = X.dtype

    return_mask = True if len(oTList) == 2 else False

    if return_mask:
        oTList[1].shape = X.shape
        oTList[1].dtype = np.dtype(np.bool_)
        oTList[1].has_grad = False

    # Use default precision if not set
    precision = getattr(op, "precision", "fp32")
    if precision is None or not isinstance(precision, str):
        precision = "fp32"

    outBytes = oTList[0].nbytes(precision)
    outBytes += oTList[1].nbytes(precision) if return_mask else 0
    outElems = oTList[0].nelems()
    outElems += oTList[1].nelems() if return_mask else 0

    op.perf_stats = {
        "inElems": inElems,
        "inBytes": inBytes,
        "outElems": outElems,
        "outBytes": outBytes,
        "instrs": instr_count,
    }
    return


def avgpool_sinf(iTList, oTList, op, **kwargs):
    assert iTList[0].check_shape(), f"Illegal Shape for {iTList[0]}"
    input_shape = iTList[0].shape

    # Handle adaptive pooling if specified
    is_adaptive = op.attrs.get("adaptive", False)

    if is_adaptive:
        # For adaptive pooling, calculate kernel_shape dynamically based on input and desired output size
        output_size = op.attrs.get("output_size", 1)

        # Ensure output_size is a tuple of appropriate length
        if isinstance(output_size, int):
            output_size = (output_size,)

        # Determine if we're doing 1D or 2D pooling based on input shape
        num_spatial_dims = len(input_shape) - 2  # Subtract batch and channel dims

        if num_spatial_dims == 1:
            # 1D pooling
            input_length = input_shape[-1]
            output_length = output_size[0]

            # Calculate kernel size and stride to achieve desired output size
            kernel_length = input_length // output_length if output_length > 0 else 1
            kernel_shape = (kernel_length,)
            strides = (kernel_length,)
            pads = [0, 0]  # No padding for adaptive pooling

        elif num_spatial_dims == 2:
            # 2D pooling
            input_height, input_width = input_shape[-2], input_shape[-1]
            output_height, output_width = (
                output_size
                if len(output_size) == 2
                else (output_size[0], output_size[0])
            )

            # Calculate kernel sizes and strides
            kernel_height = input_height // output_height if output_height > 0 else 1
            kernel_width = input_width // output_width if output_width > 0 else 1
            kernel_shape = (kernel_height, kernel_width)  # type: ignore[assignment]
            strides = (kernel_height, kernel_width)  # type: ignore[assignment]
            pads = [0, 0, 0, 0]  # No padding for adaptive pooling

        else:
            raise ValueError(
                f"Unsupported number of spatial dimensions: {num_spatial_dims}"
            )

        # Set attributes for the pooling_shape_inference function
        op.attrs["kernel_shape"] = kernel_shape
        op.attrs["strides"] = strides
        op.attrs["pads"] = pads

        # Use pooling_shape_inference instead of direct shape setting
        output_shape = pooling_shape_inference(input_shape, kernel_shape, op.attrs)
        oTList[0].shape = output_shape
        oTList[0].dtype = iTList[0].dtype
    else:
        # Traditional AveragePool with explicit kernel_shape
        kernel_shape = op.attrs.get("kernel_shape")  # Required attribute
        output_shape = pooling_shape_inference(input_shape, kernel_shape, op.attrs)
        oTList[0].shape = output_shape
        oTList[0].dtype = iTList[0].dtype

    # Compute actual data if inputs have data
    from ttsim.ops.desc.data_compute import try_compute_data, compute_avgpool2d
    oTList[0].data = try_compute_data(compute_avgpool2d, iTList, op)

    instr_count = {
        "add": iTList[0].nelems(),
        "div": oTList[0].nelems(),
        "mov": oTList[0].nelems(),
    }

    op.perf_stats = {
        "inElems": iTList[0].nelems(),
        "outElems": oTList[0].nelems(),
        "inBytes": iTList[0].nbytes(op.precision),
        "outBytes": oTList[0].nbytes(op.precision),
        "instrs": instr_count,
    }
    return


def maxpool_sinf(iTList, oTList, op, **kwargs):

    assert iTList[0].check_shape(), f"Illegal Shape for {iTList[0]}"
    input_shape = iTList[0].shape
    kernel_shape = op.attrs.get("kernel_shape")  # required attribute
    output_shape = pooling_shape_inference(input_shape, kernel_shape, op.attrs)
    oTList[0].shape = output_shape
    oTList[0].dtype = iTList[0].dtype

    if len(oTList) == 2:
        oTList[1].shape = output_shape
        oTList[1].dtype = np.dtype(np.int64)

    # Propagate NHWC-flattened hw_shape through MaxPool so downstream conv/halo LUT keys
    # use the correct [1, 1, N*H_out*W_out, C] format rather than the NCHW fallback.
    if len(output_shape) == 4:
        from ttsim.ops.tensor import nchw_to_nhwc_flat
        oTList[0].hw_shape = nchw_to_nhwc_flat(output_shape)

    # Compute actual data if inputs have data
    from ttsim.ops.desc.data_compute import try_compute_data, compute_maxpool2d
    oTList[0].data = try_compute_data(compute_maxpool2d, iTList, op)

    instr_count = {"cmp": iTList[0].nelems(), "mov": oTList[0].nelems()}
    op.perf_stats = {
        "inElems": iTList[0].nelems(),
        "outElems": oTList[0].nelems(),
        "inBytes": iTList[0].nbytes(op.precision),
        "outBytes": oTList[0].nbytes(op.precision),
        "instrs": instr_count,
    }
    return


def globalavgpool_sinf(iTList, oTList, op, **kwargs):
    assert iTList[0].check_shape(), f"Illegal Shape for {iTList[0]}"
    input_shape = iTList[0].shape
    N, C = input_shape[0], input_shape[1]
    spatial_dims = input_shape[2:]
    output_shape = [N, C] + [1] * len(spatial_dims)
    oTList[0].shape = output_shape
    oTList[0].dtype = iTList[0].dtype
    instr_count = {
        "add": iTList[0].nelems(),
        "div": oTList[0].nelems(),
        "mov": oTList[0].nelems(),
    }
    op.perf_stats = {
        "inElems": iTList[0].nelems(),
        "outElems": oTList[0].nelems(),
        "inBytes": iTList[0].nbytes(op.precision),
        "outBytes": oTList[0].nbytes(op.precision),
        "instrs": instr_count,
    }
    return


def flatten_sinf(iTList, oTList, op, **kwargs):
    assert iTList[0].check_shape(), f"Illegal Shape for {iTList[0]}"
    input_shape = iTList[0].shape
    axis = op.attrs.get("axis", 1)
    if axis < 0:
        axis += len(input_shape)
    assert (
        0 <= axis <= len(input_shape)
    ), f"Axis {axis} out of bounds for input shape {input_shape}"
    dim0 = prod_ints(input_shape[:axis])
    dim1 = prod_ints(input_shape[axis:])
    output_shape = [dim0, dim1]
    oTList[0].shape = output_shape
    oTList[0].dtype = iTList[0].dtype
    instr_count = {"mov": iTList[0].nelems()}
    op.perf_stats = {
        "inElems": iTList[0].nelems(),
        "outElems": oTList[0].nelems(),
        "inBytes": iTList[0].nbytes(op.precision),
        "outBytes": oTList[0].nbytes(op.precision),
        "instrs": instr_count,
    }
    return


def bn_sinf(iTList, oTList, op, **kwargs):
    assert all(
        [itensor.check_shape() for itensor in iTList]
    ), f"input tensor shapes not well formed!!"
    assert len(oTList) in [1, 3], f"output can either be 1 or 3"
    x = iTList[0]
    scale = iTList[1]
    bias = iTList[2]
    input_mean = iTList[3]
    input_var = iTList[4]

    oTList[0].shape = x.shape
    oTList[0].dtype = x.dtype
    if len(oTList) == 3:
        oTList[1].shape = scale.shape
        oTList[1].dtype = scale.dtype
        oTList[2].shape = scale.shape
        oTList[2].dtype = scale.dtype

    # Compute data if inputs have data
    from ttsim.ops.desc.data_compute import try_compute_data, compute_batchnorm
    oTList[0].data = try_compute_data(compute_batchnorm, iTList, op)

    instr_count = {
        "add": x.nelems() + 1,
        "mac": x.nelems(),
        "rsqrt": 1,
        "sub": 1,
        "mul": 1,
    }
    op.perf_stats = {
        "inElems": sum([i.nelems() for i in iTList]),
        "outElems": sum([o.nelems() for o in oTList]),
        "inBytes": sum([i.nbytes(op.precision) for i in iTList]),
        "outBytes": sum([o.nbytes(op.precision) for o in oTList]),
        "instrs": instr_count,
    }
    return


def conv_transpose_sinf(iTList, oTList, op, **kwargs):
    X = iTList[0]  # Input: (N, C_in, H_in, W_in)
    W = iTList[1]  # Weights: (C_in, C_out/groups, kH, kW)
    B = iTList[2] if len(iTList) == 3 else None

    assert X.rank() == 4, f"Input must be 4D (N, C_in, H_in, W_in), got {X.shape}"
    assert (
        W.rank() == 4
    ), f"Weight must be 4D (C_in, C_out/groups, kH, kW), got {W.shape}"

    N, C_in, H_in, W_in = X.shape
    C_in_w, C_out_per_group, kH, kW = W.shape

    stride = tuple(op.attrs.get("strides", (1, 1)))
    padding = tuple(op.attrs.get("padding", (0, 0)))
    output_padding = tuple(op.attrs.get("output_padding", (0, 0)))
    dilation = tuple(op.attrs.get("dilation", (1, 1)))
    kernel_size = tuple(op.attrs.get("kernel_size", (kH, kW)))
    groups = op.attrs.get("groups", 1)

    # Validate shapes
    assert C_in == C_in_w * groups, f"C_in mismatch: {C_in} != {C_in_w} * {groups}"
    C_out = C_out_per_group * groups

    # Output shape calculation (NCHW)
    H_out = (
        (H_in - 1) * stride[0]
        - 2 * padding[0]
        + dilation[0] * (kernel_size[0] - 1)
        + output_padding[0]
        + 1
    )
    W_out = (
        (W_in - 1) * stride[1]
        - 2 * padding[1]
        + dilation[1] * (kernel_size[1] - 1)
        + output_padding[1]
        + 1
    )

    output_shape = [N, C_out, H_out, W_out]
    oTList[0].shape = output_shape
    oTList[0].dtype = X.dtype
    # Set hw_shape for 2-D spatial conv_transpose (same NHWC-flattened convention as conv_sinf).
    from ttsim.ops.tensor import nchw_to_nhwc_flat
    from ttsim.front.ttnn.tensor import DataType, Layout
    oTList[0].hw_shape = nchw_to_nhwc_flat(output_shape)
    # Hardware pre-processes weights to [1, 1, C_in*kH*kW, C_out] BFLOAT8_B TILE DRAM.
    # Use _hw_dtype/_hw_layout (not _ttnn_dtype/layout) to avoid propagation into activations.
    W.hw_shape = [1, 1, C_in_w * kH * kW, C_out]
    W._hw_dtype = DataType.BFLOAT8_B
    W._hw_layout = Layout.TILE_LAYOUT
    if B is not None:
        B.hw_shape = [1, 1, 1, C_out]
        B._hw_dtype = DataType.BFLOAT8_B
        B._hw_layout = Layout.TILE_LAYOUT

    # Compute actual data if inputs have data
    from ttsim.ops.desc.data_compute import try_compute_data, compute_conv_transpose2d
    oTList[0].data = try_compute_data(compute_conv_transpose2d, iTList, op)

    # MACs: each output element is a sum over (C_in/groups * kH * kW)
    macs_per_output = (C_in // groups) * kH * kW
    output_elements = N * C_out * H_out * W_out
    total_macs = output_elements * macs_per_output
    instr_count = {"mac": int(total_macs)}
    if B is not None:
        instr_count["add"] = output_elements

    bias_elems = B.nelems() if B is not None else 0
    bias_bytes = B.nbytes(op.precision) if B is not None else 0
    inElems = X.nelems() + W.nelems() + bias_elems
    inBytes = X.nbytes(op.precision) + W.nbytes(op.precision) + bias_bytes

    op.perf_stats = {
        "inElems": inElems,
        "outElems": oTList[0].nelems(),
        "inBytes": inBytes,
        "outBytes": oTList[0].nbytes(op.precision),
        "instrs": instr_count,
    }
    return


def conv_sinf(iTList, oTList, op, **kwargs):
    assert iTList[0].check_shape(), f"Illegal Shape for {iTList[0]}"
    assert iTList[1].check_shape(), f"Illegal Shape for {iTList[1]}"
    if len(iTList) == 3:
        assert iTList[2].check_shape(), f"Illegal Shape for {iTList[2]}"

    X = iTList[0]
    W = iTList[1]
    if len(iTList) == 3:
        B = iTList[2]

    num_spatial_dims = X.rank() - 2
    if num_spatial_dims < 1:
        raise ValueError(
            "X must have at least 1 spatial dimension (N, C, spatial...): {X}"
        )

    group = op.attrs.get("group", 1)
    dilations = op.attrs.get("dilations", [1] * num_spatial_dims)
    strides = op.attrs.get("strides", [1] * num_spatial_dims)
    pads = op.attrs.get("pads", [0] * (2 * num_spatial_dims))
    auto_pad = op.attrs.get("auto_pad", "NOTSET")
    kernel_shape = op.attrs.get("kernel_shape", None)

    # Validate inputs
    if W.rank() != num_spatial_dims + 2:
        raise ValueError(
            f"Weight shape must have {num_spatial_dims + 2} dims (C_out, C_in/group, kernel_dims): {W}"
        )
    if (
        len(dilations) != num_spatial_dims
        or len(strides) != num_spatial_dims
        or len(pads) != 2 * num_spatial_dims
    ):
        raise ValueError("Dilations, strides, and pads must match spatial dimensions")
    if group <= 0 or X.shape[1] % group != 0:
        raise ValueError(f"C_in {X.shape[1]} must be divisible by group {group}")
    if W.shape[1] > X.shape[1] // group:
        raise ValueError(
            f"Weight C_in/group {W.shape[1]} exceeds input C_in/group {X.shape[1] // group}"
        )
    # Allow X.shape[1] > W.shape[1] * group: hardware pads input C for memory
    # alignment (e.g. VGG UNet C=3 → C=16); the kernel slices only weight C_in
    # channels, and the LUT records weight at its original (unpadded) C_in.
    if len(iTList) == 3:
        if B.rank() != 1 or B.shape[0] != W.shape[0]:
            raise ValueError(
                f"Bias shape {B.shape} must be (C_out,) matching weight C_out {W.shape[0]}"
            )

    N, C_in = X.shape[0], X.shape[1]
    C_out = W.shape[0]
    spatial_dims = X.shape[2:]
    kernel_dims = W.shape[2:]

    if len(kernel_dims) != num_spatial_dims:
        raise ValueError("Kernel spatial dims must match input spatial dims")

    if kernel_shape is not None:
        if kernel_shape != kernel_dims:
            raise ValueError(
                "Kernel Shape does not match Kernel-dims calculated from input spatial dims"
            )

    # Compute effective kernel size with dilation
    effective_kernel = [
        (kernel_dims[i] - 1) * dilations[i] + 1 for i in range(num_spatial_dims)
    ]

    # Compute output spatial dimensions
    output_spatial = []
    for i in range(num_spatial_dims):
        Di = spatial_dims[i]
        ki = effective_kernel[i]
        stride_i = strides[i]
        if auto_pad == "NOTSET":
            pad_begin_i = pads[i]
            pad_end_i = pads[i + num_spatial_dims]
            total_padding = pad_begin_i + pad_end_i
            Oi = (Di + total_padding - ki) // stride_i + 1
        elif auto_pad == "VALID":
            Oi = (Di - ki) // stride_i + 1
        elif auto_pad == "SAME_UPPER" or auto_pad == "SAME_LOWER":
            Oi = int(np.ceil(Di / stride_i))
        else:
            raise ValueError(f"Unsupported auto_pad value: {auto_pad}")

        if Oi <= 0:
            raise ValueError(f"Output dimension {i} would be <= 0: {Oi}")
        output_spatial.append(Oi)

    output_shape = [N, C_out] + output_spatial
    # print(">> X.shape         :", X.shape)
    # print(">> W.shape         :", W.shape)
    # if len(iTList) == 3: print(">> B.shape         :", B.shape)
    # print(">> group           :", group)
    # print(">> dilations       :", dilations)
    # print(">> strides         :", strides)
    # print(">> pads            :", pads)
    # print(">> auto_pad        :", auto_pad)
    # print(">> N               :", N)
    # print(">> C_in            :", C_in)
    # print(">> C_out           :", C_out)
    # print(">> spatial_dims    :", spatial_dims)
    # print(">> kernel_shape    :", kernel_shape)
    # print(">> kernel_dims     :", kernel_dims)
    # print(">> num_spatial_dims:", num_spatial_dims)
    # print(">> output_spatial  :", output_spatial)
    # print(">> output_shape    :", output_shape)
    # if len(iTList) == 3: print(">> B.shape         :", B.shape)

    if X.shape[0] != output_shape[0] or W.shape[0] != output_shape[1]:
        raise ValueError("Batch size (N) and C_out must match across shapes")

    oTList[0].shape = output_shape
    oTList[0].dtype = X.dtype

    # Set hw_shape for 2-D spatial conv (NCHW → [1, 1, N*H*W, C] on hardware).
    # 1-D / 3-D spatial convolutions do not use this layout so hw_shape is left None.
    if num_spatial_dims == 2:
        from ttsim.ops.tensor import nchw_to_nhwc_flat
        from ttsim.front.ttnn.tensor import DataType, Layout
        oTList[0].hw_shape = nchw_to_nhwc_flat(output_shape)
        # Hardware pre-processes weights to [1, 1, kernel_C_in*kH*kW, C_out] BFLOAT8_B TILE DRAM.
        # Use W.shape[1] (weight's own C_in/group) rather than X.shape[1]//group so the LUT
        # key matches when input C is padded above weight C (VGG UNet entry: input C=16,
        # weight C_in=3 → weight hw_shape = 3*3*3 = 27 in the LUT).
        kH_int, kW_int = int(kernel_dims[0]), int(kernel_dims[1])
        W.hw_shape = [1, 1, int(W.shape[1]) * kH_int * kW_int, C_out]
        W._hw_dtype = DataType.BFLOAT8_B
        W._hw_layout = Layout.TILE_LAYOUT
        if len(iTList) == 3:
            B.hw_shape = [1, 1, 1, C_out]
            B._hw_dtype = DataType.BFLOAT8_B
            B._hw_layout = Layout.TILE_LAYOUT

    macs_per_output = int(W.shape[1]) * np.prod(kernel_dims)
    output_elements = N * C_out * np.prod(output_spatial)
    total_macs = output_elements * macs_per_output
    instr_count = {"mac": int(total_macs)}
    if len(iTList) == 3:
        instr_count["add"] = int(output_elements)

    bias_elems = B.nelems() if len(iTList) == 3 else 0
    bias_bytes = B.nbytes(op.precision) if len(iTList) == 3 else 0
    inElems = X.nelems() + W.nelems() + bias_elems
    inBytes = X.nbytes(op.precision) + W.nbytes(op.precision) + bias_bytes

    op.perf_stats = {
        "inElems": inElems,
        "outElems": oTList[0].nelems(),
        "inBytes": inBytes,
        "outBytes": oTList[0].nbytes(op.precision),
        "instrs": instr_count,
    }

    # NUMERICAL COMPUTATION (if data available)
    from .data_compute import compute_conv2d

    if X.data is not None and W.data is not None:
        oTList[0].data = compute_conv2d(iTList, op)
    else:
        oTList[0].data = None

    return


def ln_sinf(iTList, oTList, op, **kwargs):
    axis = op.attrs.get("axis", -1)
    epsilon = op.attrs.get("epsilon", 1e-5)
    stash_type = op.attrs.get("stash_type", 1)

    X = iTList[0]
    scaleT = iTList[1]
    biasT = iTList[2] if len(iTList) == 3 else None
    assert X.check_shape(), f"Illegal Shape for {X}"
    XShape = X.shape
    XRank = X.rank()

    # LayerNormalization implementation with Numpy....
    # From Spec: https://github.com/onnx/onnx/blob/main/docs/Operators.md#LayerNormalization
    # Equations:
    #
    # [Stage-1]
    #   normalized_axes = [axis, ..., rank of X - 1]
    #     Mean       = ReduceMean<axes=normalized_axes>(X)    := X_mu
    #     D          = Sub(X, Mean)                           := X - X_mu
    #     DD         = Mul(D, D)                              := (X - X_mu)^2
    #     Var        = ReduceMean<axes=normalized_axes>(DD)   := X_sigma^2
    #     VarEps     = Add(Var, epsilon)                      := X_sigma^2 + eps
    #     StdDev     = Sqrt(VarEps)                           := sqrt(X_sigma^2 + eps)
    #     InvStdDev  = Reciprocal(StdDev)                     := 1/sqrt(X_sigma^2 + eps)
    #     Normalized = Mul(D, InvStdDev)                      := (X - X_mu)/(1/sqrt(X_sigma^2 + eps))
    # Stage-2
    #     NormalizedScaled = Mul(Normalized, Scale)
    #     if (Bias): Y = Add(NormalizedScaled, Bias)

    if axis < 0:
        axis += XRank
    normalized_axes = XShape[axis:]
    unsqueezed_rank = XRank - axis
    reduction_shape = XShape[0:axis] + [1] * unsqueezed_rank

    instr_count = {
        "add": 0,
        "sub": 0,
        "mul": 0,
        "div": 0,
        "mac": 0,
        "rsqrt": 0,
    }  # dict to hold instr counts
    input_count = X.nelems()
    reduction_count = prod_ints(reduction_shape)

    # -------x------- Stage-1 Implementation -------x-------
    # Parameter used to convert N-D tensor layer norm
    # to equivalent 2-D matirx operations.
    row, col = 1, 1
    for i in range(XRank):
        if i < axis:
            row *= XShape[i]
        else:
            col *= XShape[i]
    # After reshaping input tensor X into a matrix, layer norm
    # is equivalent to conducting standardization on each column
    # (s.t. each col has zero mean and unit variance).
    # x_mat = np.reshape(X.data, (row, col))

    # compute mean for every x_mat's col
    # x_mean = np.sum(x_mat, axis=1, keepdims=True)/col
    instr_count["add"] += input_count
    instr_count["div"] += reduction_count
    # x_diff = x_mat - x_mean
    instr_count["sub"] += input_count
    # x_squared_diff = x_diff * x_diff
    instr_count["mul"] += input_count
    # compute variance for every x_mat's col
    # variance = np.sum(x_squared_diff, axis=1, keepdims=True)/col
    instr_count["add"] += input_count
    instr_count["div"] += reduction_count
    # variance_eps = variance + epsilon
    instr_count["add"] += reduction_count
    # std_dev = np.sqrt(variance_eps)
    # inv_std_dev = np.reciprocal(std_dev)
    instr_count["rsqrt"] += reduction_count

    # Standardization step. y_mat is zero-mean and unit-variance.
    # y_mat = x_diff * inv_std_dev
    instr_count["mul"] += input_count

    # -------x------- Stage-2 Implementation -------x-------
    # Apply affine transform on normalization outcome.
    # assert scaleT.data is not None, f"Illegal DATA in Tensor {scaleT}"
    # y_mat = np.reshape(y_mat, XShape) * scaleT.data
    instr_count["mac"] += input_count
    if biasT is not None:
        # Check: this add is already counted in the 'mac' above?
        # y_mat = y_mat + biasT.data
        pass

    oTList[0].shape = X.shape
    oTList[0].dtype = X.dtype

    if len(oTList) >= 2:
        # reshape needed because of initial tensor-to-matrix reshape in Step-1.
        # X_mean = np.reshape(x_mean, reduction_shape)
        oTList[1].shape = reduction_shape
        oTList[1].dtype = X.dtype

    if len(oTList) == 3:
        # reshape needed because of initial tensor-to-matrix reshape in Step-1.
        # X_invSDT = np.reshape(inv_std_dev, reduction_shape)
        oTList[2].shape = reduction_shape
        oTList[2].dtype = X.dtype

    biasElems = 0 if biasT is None else biasT.nelems()
    meanElems = 0 if len(oTList) < 2 else oTList[1].nelems()
    invSDTElems = 0 if len(oTList) < 3 else oTList[2].nelems()
    biasBytes = 0 if biasT is None else biasT.nbytes(op.precision)
    meanBytes = 0 if len(oTList) < 2 else oTList[1].nbytes(op.precision)
    invSDTBytes = 0 if len(oTList) < 3 else oTList[2].nbytes(op.precision)
    op.perf_stats = {
        "inElems": iTList[0].nelems() + iTList[1].nelems() + biasElems,
        "outElems": oTList[0].nelems() + meanElems + invSDTElems,
        "inBytes": iTList[0].nbytes(op.precision)
        + iTList[1].nbytes(op.precision)
        + biasBytes,
        "outBytes": oTList[0].nbytes(op.precision) + meanBytes + invSDTBytes,
        "instrs": instr_count,
    }

    # NUMERICAL COMPUTATION (if data available)
    from .data_compute import compute_layernorm

    if X.data is not None and scaleT.data is not None:
        oTList[0].data = compute_layernorm(iTList, op)
    else:
        oTList[0].data = None

    return


def skip_ln_sinf(iTList, oTList, op, **kwargs):
    X = iTList[0]
    skip = iTList[1]
    gamma = iTList[2]
    beta = iTList[3] if len(iTList) > 3 else None
    bias = iTList[4] if len(iTList) > 4 else None
    assert X.check_shape(), f"Illegal Shape for {X}"

    input_count = X.nelems()
    reduction_count = prod_ints(X.shape[:-1]) if X.rank() > 0 else 1

    instr_count = {
        "add": 0,
        "sub": 0,
        "mul": 0,
        "div": 0,
        "mac": 0,
        "rsqrt": 0,
    }
    # value = input + skip (+ bias)
    instr_count["add"] += input_count           # + skip
    if bias is not None:
        instr_count["add"] += input_count       # + bias
    # LayerNorm over last dim
    instr_count["add"] += input_count           # sum for mean
    instr_count["div"] += reduction_count       # mean
    instr_count["sub"] += input_count           # x - mean
    instr_count["mul"] += input_count           # squared diff
    instr_count["add"] += input_count           # sum for variance
    instr_count["div"] += reduction_count       # variance
    instr_count["add"] += reduction_count       # variance + epsilon
    instr_count["rsqrt"] += reduction_count     # 1/sqrt(var + eps)
    instr_count["mul"] += input_count           # normalize
    instr_count["mac"] += input_count           # scale (*gamma) (+beta)

    oTList[0].shape = X.shape
    oTList[0].dtype = X.dtype

    reduction_shape = X.shape[:-1] + [1]
    if len(oTList) >= 2:
        oTList[1].shape = reduction_shape
        oTList[1].dtype = X.dtype
    if len(oTList) >= 3:
        oTList[2].shape = reduction_shape
        oTList[2].dtype = X.dtype
    if len(oTList) >= 4:
        oTList[3].shape = X.shape
        oTList[3].dtype = X.dtype

    biasElems = 0 if bias is None else bias.nelems()
    betaElems = 0 if beta is None else beta.nelems()
    biasBytes = 0 if bias is None else bias.nbytes(op.precision)
    betaBytes = 0 if beta is None else beta.nbytes(op.precision)
    outElems = sum(t.nelems() for t in oTList)
    outBytes = sum(t.nbytes(op.precision) for t in oTList)
    op.perf_stats = {
        "inElems": X.nelems() + skip.nelems() + gamma.nelems() + betaElems + biasElems,
        "outElems": outElems,
        "inBytes": X.nbytes(op.precision)
        + skip.nbytes(op.precision)
        + gamma.nbytes(op.precision)
        + betaBytes
        + biasBytes,
        "outBytes": outBytes,
        "instrs": instr_count,
    }

    oTList[0].data = None
    return


def groupnorm_sinf(iTList, oTList, op, **kwargs):
    # For GroupNormalization, the shape inference is similar to LayerNormalization
    # The output shape is the same as input shape
    X = iTList[0]
    scaleT = iTList[1]
    biasT = iTList[2] if len(iTList) == 3 else None
    assert X.check_shape(), f"Illegal Shape for {X}"

    oTList[0].shape = X.shape
    oTList[0].dtype = X.dtype

    instr_count = {
        "add": X.nelems(),
        "sub": X.nelems(),
        "mul": X.nelems(),
        "div": X.nelems(),
        "mac": X.nelems() * 2,
        "rsqrt": X.nelems() // 1000,  # Rough estimate
    }

    # Use default precision if not set
    precision = getattr(op, "precision", "fp32")
    if precision is None or not isinstance(precision, str):
        precision = "fp32"

    biasElems = 0 if biasT is None else biasT.nelems()
    biasBytes = 0 if biasT is None else biasT.nbytes(precision)

    op.perf_stats = {
        "inElems": iTList[0].nelems() + iTList[1].nelems() + biasElems,
        "outElems": oTList[0].nelems(),
        "inBytes": iTList[0].nbytes(precision)
        + iTList[1].nbytes(precision)
        + biasBytes,
        "outBytes": oTList[0].nbytes(precision),
        "instrs": instr_count,
    }

    # NUMERICAL COMPUTATION (if data available)
    from .data_compute import compute_groupnorm

    if X.data is not None and scaleT.data is not None:
        oTList[0].data = compute_groupnorm(iTList, op)
    else:
        oTList[0].data = None

    return


def multihead_attention_sinf(iTList, oTList, op, **kwargs):
    """
    Multi-head attention shape inference.

    Args:
        iTList: [query, key, value] or [query, key, value, key_padding_mask, attn_mask]
            query: [*, tgt_len, embed_dim]
            key: [*, src_len, embed_dim]
            value: [*, src_len, embed_dim]
            key_padding_mask: [batch, src_len] (optional)
            attn_mask: [tgt_len, src_len] (optional)
        oTList: [output, attn_weights]
        op: SimOp with attrs:
            - embed_dim: int
            - num_heads: int
            - dropout: float
    """
    from .data_compute import compute_multihead_attention

    query, key, value = iTList[0], iTList[1], iTList[2]
    output_t = oTList[0]

    # Shape inference
    embed_dim = op.attrs.get("embed_dim")
    num_heads = op.attrs.get("num_heads")

    assert query.check_shape(), f"Query shape not defined: {query}"
    assert key.check_shape(), f"Key shape not defined: {key}"
    assert value.check_shape(), f"Value shape not defined: {value}"

    # Output has same shape as query
    output_t.shape = list(query.shape)
    output_t.dtype = query.dtype

    # Attention weights shape: [*batch, num_heads, tgt_len, src_len]
    if len(oTList) > 1:
        attn_weights_t = oTList[1]
        batch_shape = query.shape[:-2]
        tgt_len = query.shape[-2]
        src_len = key.shape[-2]
        attn_weights_t.shape = list(batch_shape) + [num_heads, tgt_len, src_len]
        attn_weights_t.dtype = query.dtype

    # Performance stats
    tgt_len = query.shape[-2]
    src_len = key.shape[-2]
    head_dim = embed_dim // num_heads

    # MACs for Q*K^T, attention*V, and projections
    qk_macs = np.prod(query.shape[:-1]) * src_len * head_dim
    attn_v_macs = np.prod(query.shape[:-1]) * src_len * head_dim
    proj_macs = np.prod(query.shape) * embed_dim * 4  # in_proj + out_proj

    op.perf_stats = {
        "inElems": query.nelems() + key.nelems() + value.nelems(),
        "outElems": output_t.nelems(),
        "inBytes": query.nbytes(op.precision)
        + key.nbytes(op.precision)
        + value.nbytes(op.precision),
        "outBytes": output_t.nbytes(op.precision),
        "instrs": {"mac": int(qk_macs + attn_v_macs + proj_macs)},
    }

    # Numerical computation (only if data available)
    if query.data is not None and key.data is not None and value.data is not None:
        output_data, attn_weights_data = compute_multihead_attention(iTList, op)
        output_t.data = output_data
        if len(oTList) > 1:
            oTList[1].data = attn_weights_data
    else:
        output_t.data = None
        if len(oTList) > 1:
            oTList[1].data = None

    return output_t


def register_nn_ops():
    _optbl = [
        # ['Shrink',                    'ARITY_1->1', 'ai.onnx', 'COMMON',   9,   9,  1,  1,  1,  1, unary_fwd,     True,  True,  True,  True,  True],
        # ['LRN',                       'ARITY_1->1', 'ai.onnx', 'COMMON',  13,  13,  1,  1,  1,  1, unary_fwd,     True,  True,  True,  True,  True],
        # ['LpPool',                    'ARITY_1->1', 'ai.onnx', 'COMMON',  22,  22,  1,  1,  1,  1, 'inline_lamda',  True,  True,  True,  True,  True],
        # ['GlobalMaxPool',             'ARITY_1->1', 'ai.onnx', 'COMMON',  22,  22,  1,  1,  1,  1, 'inline_lamda',  True,  True,  True,  True,  True],
        # ['LpNormalization',           'ARITY_1->1', 'ai.onnx', 'COMMON',  22,  22,  1,  1,  1,  1, 'inline_lamda',  True,  True,  True,  True,  True],
        # ['TfIdfVectorizer',           'ARITY_1->1', 'ai.onnx', 'COMMON',   9,   9,  1,  1,  1,  1, 'inline_lamda',  True,  True,  True,  True,  True],
        # ['GlobalLpPool',              'ARITY_1->1', 'ai.onnx', 'COMMON',  22,  22,  1,  1,  1,  1, 'no_inference',  True,  True,  True,  True,  True],
        # ['MeanVarianceNormalization', 'ARITY_1->1', 'ai.onnx', 'COMMON',  13,  13,  1,  1,  1,  1, 'no_inference',  True,  True,  True,  True,  True],
        # ['MaxRoiPool',            'ARITY_2->1', 'ai.onnx', 'COMMON',  22,  22,  2,  2,  1,  1, 'inline_lamda',  True,  True,  True,  True,  True],
        # ['InstanceNormalization', 'ARITY_3->1', 'ai.onnx', 'COMMON',  22,  22,  3,  3,  1,  1, 'inline_lamda',  True,  True,  True,  True,  True],
        # ['Col2Im',                'ARITY_3->1', 'ai.onnx', 'COMMON',  18,  18,  3,  3,  1,  1, 'inline_lamda',  True,  True,  True,  True,  True],
        [
            "GroupNormalization",
            "ARITY_3->1",
            "ai.onnx",
            "COMMON",
            21,
            21,
            3,
            3,
            1,
            1,
            groupnorm_sinf,
            True,
            True,
            True,
            True,
            True,
        ],
        [
            "MultiheadAttention",
            "ARITY_VARIADIC[3-5]->VARIADIC[1-2]",
            "com.tenstorrent",
            "COMMON",
            22,
            22,
            5,
            3,
            2,
            1,
            multihead_attention_sinf,
            True,
            True,
            True,
            True,
            True,
        ],
        # ['MaxUnpool',             'ARITY_VARIADIC[2-3]->1',             'ai.onnx', 'COMMON',  22,  22,  3,  2,  1,  1, 'inline_lamda',  True,  True,  True,  True,  True],
        # ['ConvInteger',           'ARITY_VARIADIC[2-4]->1',             'ai.onnx', 'COMMON',  10,  10,  4,  2,  1,  1, 'inline_lamda',  True,  True,  True,  True,  True],
        # ['DeformConv',            'ARITY_VARIADIC[3-5]->1',             'ai.onnx', 'COMMON',  22,  22,  5,  3,  1,  1, 'inline_lamda',  True,  True,  True,  True,  True],
        # ['QLinearConv',           'ARITY_VARIADIC[8-9]->1',             'ai.onnx', 'COMMON',  10,  10,  9,  8,  1,  1, 'inline_lamda',  True,  True,  True,  True,  True],
        [
            "Flatten",
            "ARITY_1->1",
            "ai.onnx",
            "COMMON",
            24,
            21,
            1,
            1,
            1,
            1,
            flatten_sinf,
            True,
            True,
            True,
            True,
            True,
        ],
        [
            "GlobalAveragePool",
            "ARITY_1->1",
            "ai.onnx",
            "COMMON",
            22,
            22,
            1,
            1,
            1,
            1,
            globalavgpool_sinf,
            True,
            True,
            True,
            True,
            True,
        ],
        [
            "AveragePool",
            "ARITY_1->1",
            "ai.onnx",
            "COMMON",
            22,
            22,
            1,
            1,
            1,
            1,
            avgpool_sinf,
            True,
            True,
            True,
            True,
            True,
        ],
        [
            "MaxPool",
            "ARITY_1->VARIADIC[1-2]",
            "ai.onnx",
            "COMMON",
            22,
            22,
            1,
            1,
            2,
            1,
            maxpool_sinf,
            True,
            True,
            True,
            True,
            True,
        ],
        [
            "Conv",
            "ARITY_VARIADIC[2-3]->1",
            "ai.onnx",
            "COMMON",
            22,
            22,
            3,
            2,
            1,
            1,
            conv_sinf,
            True,
            True,
            True,
            True,
            True,
        ],
        [
            "ConvTranspose",
            "ARITY_VARIADIC[2-3]->1",
            "ai.onnx",
            "COMMON",
            22,
            22,
            3,
            2,
            1,
            1,
            conv_transpose_sinf,
            True,
            True,
            True,
            True,
            True,
        ],
        [
            "BatchNormalization",
            "ARITY_5->VARIADIC[1-3]",
            "ai.onnx",
            "COMMON",
            15,
            15,
            5,
            5,
            3,
            1,
            bn_sinf,
            True,
            True,
            True,
            True,
            True,
        ],
        [
            "Dropout",
            "ARITY_VARIADIC[1-3]->VARIADIC[1-2]",
            "ai.onnx",
            "COMMON",
            22,
            22,
            3,
            1,
            2,
            1,
            dropout_sinf,
            True,
            True,
            True,
            True,
            True,
        ],
        [
            "LayerNormalization",
            "ARITY_VARIADIC[2-3]->VARIADIC[1-3]",
            "ai.onnx",
            "COMMON",
            17,
            17,
            3,
            2,
            3,
            1,
            ln_sinf,
            True,
            True,
            True,
            True,
            True,
        ],
        [
            "SkipLayerNormalization",
            "ARITY_VARIADIC[3-5]->VARIADIC[1-4]",
            "com.microsoft",
            "COMMON",
            1,
            1,
            5,
            3,
            4,
            1,
            skip_ln_sinf,
            True,
            True,
            True,
            True,
            True,
        ],
    ]

    register_ops("nn", _optbl)
    return
