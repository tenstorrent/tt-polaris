#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from enum import Enum, auto

# NOTE: this module rebinds several builtin names at module scope as ttnn ops — `min`, `max`,
# `pow`, `slice`, `sum`. Python resolves globals at call time, so those builtins are shadowed
# inside *every* function in this file, including ones defined above the rebinding. Reach them
# via `builtins.<name>` — never the bare name.
import builtins
import numbers
import numpy as np
from loguru import logger

from ttsim.ops.op import SimOp
from ttsim.ops.tensor import Shape, require_shape_list

from .buffer import BufferType, TensorMemoryLayout
from .memory import MemoryConfig
from .tensor import DataType, Layout, Tensor, generate_new_op_name, require_ttnn_tensor, zeros


class MathFidelity(Enum):
    LoFi = auto()
    HiFi2 = auto()
    HiFi3 = auto()
    HiFi4 = auto()

    @classmethod
    def enumvalue(cls, s: str):
        return MathFidelity[s]

    @property
    def cname(self) -> str:
        return self.name.lower()


_COMPACT_DTYPES: frozenset[DataType] = frozenset()  # populated after DataType import below
# Precedence order for compact dtypes (most compact first)
_COMPACT_DTYPE_PRECEDENCE: list[DataType] = []  # populated after DataType import


def _propagate_ttnn_dtype(inputs: list[Tensor], outputs: list[Tensor]) -> None:
    """Propagate _ttnn_dtype from inputs to unannotated outputs.

    When inputs carry different DataTypes (e.g. BFLOAT16 + BFLOAT8_B), the more
    compact type wins.  This mirrors HW behaviour where ``activations_dtype``
    (typically BFLOAT8_B) dominates after the first conversion point.

    Compact dtypes are prioritized by explicit precedence: BFLOAT4_B (most compact)
    is preferred over BFLOAT8_B, ensuring deterministic propagation regardless of
    input order.
    """
    candidates = [
        getattr(t, "_ttnn_dtype", None)
        for t in inputs
        if getattr(t, "_ttnn_dtype", None) is not None
    ]
    if not candidates:
        return

    # Find the most compact dtype according to precedence order
    compact = [d for d in candidates if d in _COMPACT_DTYPES]
    if compact:
        # Pick the first dtype in precedence order that appears in candidates
        src = next((d for d in _COMPACT_DTYPE_PRECEDENCE if d in compact), compact[0])
    else:
        src = candidates[0]

    for o in outputs:
        if getattr(o, "_ttnn_dtype", None) is None:
            o._ttnn_dtype = src


_COMPACT_DTYPES = frozenset({DataType.BFLOAT8_B, DataType.BFLOAT4_B})
# BFLOAT4_B (4 bits) is more compact than BFLOAT8_B (8 bits)
_COMPACT_DTYPE_PRECEDENCE = [DataType.BFLOAT4_B, DataType.BFLOAT8_B]


def _propagate_memory_config(inputs: list[Tensor], outputs: list[Tensor]) -> None:
    """Propagate _memory_config from the first input that has one set.

    On real tt-metal, op outputs inherit the memory placement of their
    inputs unless the op explicitly changes it (e.g. to_memory_config).
    """
    src = None
    for t in inputs:
        mc = getattr(t, "_memory_config", None)
        if mc is not None:
            src = mc
            break
    if src is None:
        return
    for o in outputs:
        if getattr(o, "_memory_config", None) is None:
            o._memory_config = src


def _resolve_output_layout(kwargs: dict, tensor_args: list[Tensor]) -> "Layout | None":
    """Determine an op output's ``layout`` the way tt-metal does.

    An explicit ``layout=`` kwarg names the output layout of a layout-changing
    op (e.g. ``ttnn.embedding(..., layout=TILE_LAYOUT)`` emits a tilized result
    from ROW_MAJOR indices), so it wins.  Otherwise the output inherits the
    primary tensor input's layout — layout-preserving ops (layer_norm, add,
    mul, softmax, reshape, …) keep the activation's layout, matching HW.

    Companion to ``_propagate_ttnn_dtype`` / ``_propagate_memory_config`` for
    the ``.layout`` attribute, which SimTensor tracks directly (not as an
    overlay), so it must be set at construction time — the padded shape is
    derived from the layout during shape inference.  Returns ``None`` when no
    layout can be determined (Tensor coerces that to ``Layout.DEFAULT``).
    """
    lay = kwargs.get("layout", None)
    if lay is not None:
        return lay
    for t in tensor_args:
        if isinstance(t, Tensor):
            return t.get_layout()
    return None


def single_output_immediate_op(optype, /, preprocess=None):

    def _impl(*args, **kwargs):

        if preprocess:
            args, kwargs = preprocess(args, kwargs)

        tensor_args: list[Tensor] = []
        for i, x in enumerate(args):
            if isinstance(x, (int, float)):
                continue
            tensor_args.append(
                require_ttnn_tensor(x, f"ttnn.op({optype}) argument {i}")
            )
        if not tensor_args:
            raise TypeError(
                f"ttnn.op({optype}) requires at least one ttsim.front.ttnn.tensor.Tensor argument"
            )
        devchk_list = [x.device for x in tensor_args]
        device = devchk_list[0]
        # assert device and all(x == device for x in devchk_list), f"device check: {devchk_list}"

        op_name = generate_new_op_name()
        opinfo  = {'name': op_name, 'optype': optype, 'inList': [], 'attrs': kwargs}
        C = Tensor(name=op_name + ".out", op_out=[op_name], device=device,
                   layout=_resolve_output_layout(kwargs, tensor_args))

        new_args = []
        for i, x in enumerate(args):
            if isinstance(x, Tensor):
                x.op_in.append(op_name)
                opinfo["inList"].append(x.name)
                new_args.append(x)
            elif isinstance(x, (int, float)):
                if optype in ["Add", "Sub", "Mul"]:
                    assert (
                        len(tensor_args) == 1
                    ), f"Only one tensor input supported for {optype} with scalar input"
                    opinfo["attrs"]["scalar"] = x
                    # On tt-metal, the scalar is broadcast-filled into a
                    # full-shape tensor buffer.  The profiler records both
                    # inputs, so include this broadcast tensor in inList to
                    # produce an arity-2 LUT key matching the profiler.
                    tensor_shape = list(tensor_args[0].shape) if tensor_args[0].shape is not None else [1]
                    tensor_dtype = DataType.from_numpy(tensor_args[0].dtype)

                    # Populate data for EXECUTE/EXECUTE_AND_TRACK modes so that
                    # compute helpers (compute_add/sub/mul) can access iTList[1].data
                    scalar_data = None
                    if hasattr(tensor_args[0], 'has_data') and tensor_args[0].has_data():
                        # Create broadcast-filled array matching tensor shape
                        np_dtype = tensor_args[0].dtype
                        scalar_data = np.full(tensor_shape, x, dtype=np_dtype)

                    tmp = Tensor(
                        name=f"{op_name}.scalar",
                        shape=tensor_shape,
                        dtype=tensor_dtype,
                        layout=tensor_args[0].get_layout(),
                        device=device,
                        data=scalar_data,
                    )
                    src_mc = getattr(tensor_args[0], "_memory_config", None)
                    if src_mc is not None:
                        tmp._memory_config = src_mc
                    src_dt = getattr(tensor_args[0], "_ttnn_dtype", None)
                    if src_dt is not None:
                        tmp._ttnn_dtype = src_dt
                    tmp.op_in.append(op_name)
                    opinfo["inList"].append(tmp.name)
                    new_args.append(tmp)
                else:
                    logger.warning(
                        "Scalar operand not supported for {} (only Add, Sub, Mul accept scalars); "
                        "got type {}",
                        optype,
                        type(x).__name__,
                    )
                    raise TypeError(
                        f"Scalar input is not supported for ttnn.op({optype}); "
                        f"only Add, Sub, and Mul accept scalar operands"
                    )
            else:
                assert False, f"Unknown input type in ttnn.op({optype}) : {type(x)}"
        opinfo["outList"] = [C.name]

        opobj = SimOp(opinfo)
        perf_stats = opobj.get_perf_counts(new_args, [C])
        opobj.update_tensor_counts(new_args, [C])

        _propagate_ttnn_dtype(tensor_args, [C])
        # An explicit output dtype (e.g. ttnn.mul(..., dtype=bfloat8_b) — the MLP
        # gate*up downcast) wins over the propagated dtype, matching real ttnn where
        # the dtype= kwarg names the output type. Mirrors linear()'s handling.
        out_dt = kwargs.get("output_dtype", kwargs.get("dtype", None))
        if isinstance(out_dt, DataType):
            C._ttnn_dtype = out_dt
        _propagate_memory_config(tensor_args, [C])

        mem_cfg = kwargs.get("memory_config", None)
        if mem_cfg is not None:
            C._memory_config = mem_cfg

        device.add_op(opobj)  # type: ignore[union-attr]

        return C

    return _impl


def multiple_output_immediate_op(optype, /, preprocess=None):
    def _impl(*args, **kwargs):

        if preprocess:
            args, kwargs = preprocess(args, kwargs)

        for i, x in enumerate(args):
            require_ttnn_tensor(x, f"ttnn.op({optype}) argument {i}")
        tensor_args: list[Tensor] = list(args)
        devchk_list = [x.device for x in tensor_args]
        device = devchk_list[0]
        # assert device and all(x == device for x in devchk_list), f"device check: {devchk_list}"

        op_name = generate_new_op_name()
        opinfo = {"name": op_name, "optype": optype, "inList": [], "attrs": kwargs}
        out_tensors = []
        num_outputs = kwargs.get("num_outputs", 2)
        out_layout = _resolve_output_layout(kwargs, tensor_args)
        for out_idx in range(num_outputs):
            C = Tensor(name=f"{op_name}.out.{out_idx}", op_out=[op_name], device=device,
                       layout=out_layout)
            out_tensors.append(C)

        new_args = []
        for i, x in enumerate(args):
            if isinstance(x, Tensor):
                x.op_in.append(op_name)
                opinfo["inList"].append(x.name)
                new_args.append(x)
            else:
                assert False, f"Unknown input type in ttnn.op({optype}) : {type(x)}"
        opinfo["outList"] = [C.name for C in out_tensors]

        opobj = SimOp(opinfo)
        perf_stats = opobj.get_perf_counts(new_args, out_tensors)
        # print(f"{optype}:: {perf_stats}")
        opobj.update_tensor_counts(new_args, out_tensors)

        _propagate_ttnn_dtype(tensor_args, out_tensors)
        _propagate_memory_config(tensor_args, out_tensors)
        device.add_op(opobj)  # type: ignore[union-attr]

        return tuple(out_tensors)

    return _impl


def argmax_pp(args_list, kwargs_dict):
    # translate attribs
    kwargs_dict["axis"] = kwargs_dict.get("dim", 0)
    if "keepdim" in kwargs_dict:
        kwargs_dict["keepdims"] = 1 if kwargs_dict["keepdim"] else 0
    else:
        kwargs_dict["keepdims"] = 0
    return args_list, kwargs_dict


def reshape_pp(args_list, kwargs_dict):
    assert len(args_list) <= 3, "ttnn.reshape has 3 inputs (special case for TT h/w)"
    inT = require_ttnn_tensor(args_list[0], "ttnn.reshape input")
    outShape = args_list[1]
    if isinstance(outShape, Shape):
        outShape = outShape.as_list()
    elif isinstance(outShape, (list, tuple)):
        outShape = list(outShape)
    else:
        assert False, (
            "ttnn.reshape 2nd input should be a list, tuple, or ttsim.ops.tensor.Shape"
        )
    assert all(
        isinstance(x, (int, np.integer)) for x in outShape
    ), "ttnn.reshape 2nd input should be a sequence of integer sizes"

    in_dtype = DataType.from_numpy(inT.dtype)
    if len(args_list) == 3:
        # write code to get slice (batch 0) of the input tensor and return it
        inT = Tensor(shape=outShape, device=inT.device, dtype=in_dtype)

    # NOTE: dimensions in the shape should be integer type
    shapeData = np.array(outShape, dtype=np.int64)
    # NOTE: shapeData is not the reshape output, but it holds the shape tensor data,
    # so it should be of type INT64
    shapeT = Tensor(
        shape=shapeData.shape, dtype=DataType.INT64, device=inT.device, data=shapeData
    )
    return (inT, shapeT), kwargs_dict


def expand_pp(args_list, kwargs_dict):
    inT = require_ttnn_tensor(args_list[0], "ttnn.expand input")
    outShape = args_list[1]
    if isinstance(outShape, Shape):
        outShape = outShape.as_list()
    elif isinstance(outShape, (list, tuple)):
        outShape = list(outShape)
    else:
        assert False, (
            "ttnn.expand 2nd input should be a list, tuple, or ttsim.ops.tensor.Shape"
        )
    outData = np.array(outShape, dtype=np.int64)
    outT = Tensor(
        shape=outData.shape, dtype=DataType.INT64, device=inT.device, data=outData
    )
    return (inT, outT), kwargs_dict


def split_pp(args_list, kwargs_dict):
    inT = require_ttnn_tensor(args_list[0], "ttnn.split input")
    axis = kwargs_dict.get("dim", 0)
    second = args_list[1] if len(args_list) > 1 else None
    if isinstance(second, Tensor):  # legacy ONNX output-template form
        kwargs_dict["split_sizes"] = kwargs_dict.get("split_sizes", None)
        kwargs_dict["num_splits"] = kwargs_dict.get("num_splits", None)
        kwargs_dict["axis"] = axis
        return (inT, second), kwargs_dict
    # Real ttnn: ttnn.split(x, split_size:int, dim=...) — chunks of split_size along dim.
    split_size = second if isinstance(second, int) else kwargs_dict.get("split_size")
    assert split_size is not None, "ttnn.split requires split_size (int) or an output-template tensor"
    dim_size = require_shape_list(inT.shape, "ttnn.split input shape must be set")[axis]
    assert dim_size % split_size == 0, f"ttnn.split: dim size {dim_size} not divisible by split_size {split_size}"
    kwargs_dict["num_outputs"] = dim_size // split_size
    kwargs_dict["axis"] = axis
    return (inT,), kwargs_dict


def permute_pp(args_list, kwargs_dict):
    inT = require_ttnn_tensor(args_list[0], "ttnn.permute input")
    assert isinstance(
        args_list[1], (list, tuple)
    ), "ttnn.permute 2nd input should be a list|tuple of ints"
    kwargs_dict["perm"] = list(args_list[1])
    return (inT,), kwargs_dict


def embedding_pp(args_list, kwargs_dict):
    # TTNN passes (indices, weights); the Embedding op records them in the SAME order as the
    # hardware EmbeddingsDeviceOperation: input_0 = tokens (indices), input_1 = weight. This is
    # tokens-first (opposite of ONNX Gather's data-first order) — see embedding_sinf.
    assert len(args_list) == 2, "ttnn.embedding has 2 inputs"
    input_tensor = require_ttnn_tensor(args_list[0], "ttnn.embedding indices")
    weight_tensor = require_ttnn_tensor(args_list[1], "ttnn.embedding weight")
    return (input_tensor, weight_tensor), kwargs_dict


def layer_norm_pp(args_list, kwargs_dict):
    input_tensor = args_list[0]
    weight_tensor = kwargs_dict["weight"] if "weight" in kwargs_dict else None
    bias_tensor = kwargs_dict["bias"] if "bias" in kwargs_dict else None
    axis = kwargs_dict["axis"] if "axis" in kwargs_dict else None
    epsilon = kwargs_dict["epsilon"] if "epsilon" in kwargs_dict else None
    memory_config = (
        kwargs_dict["memory_config"] if "memory_config" in kwargs_dict else None
    )
    compute_kernel_config = (
        kwargs_dict["compute_kernel_config"]
        if "compute_kernel_config" in kwargs_dict
        else None
    )

    input_tensor = require_ttnn_tensor(input_tensor, "ttnn.layer_norm input")
    if weight_tensor is not None:
        weight_tensor = require_ttnn_tensor(weight_tensor, "ttnn.layer_norm weight")
    if bias_tensor is not None:
        bias_tensor = require_ttnn_tensor(bias_tensor, "ttnn.layer_norm bias")

    kwargs_dict = {}
    if axis is not None:
        kwargs_dict['axis'] = axis
    if epsilon is not None:
        kwargs_dict['epsilon'] = epsilon
    if memory_config is not None:
        kwargs_dict['memory_config'] = memory_config
    if compute_kernel_config is not None:
        # compute_kernel_config is normally a ComputeKernelConfig (has .math_fidelity), but
        # callers may pass a bare MathFidelity enum — honor both.
        mf = getattr(compute_kernel_config, 'math_fidelity', None)
        if mf is None and isinstance(compute_kernel_config, MathFidelity):
            mf = compute_kernel_config
        if mf is not None:
            kwargs_dict['math_fidelity'] = mf.name

    if bias_tensor is not None:
        return (input_tensor, weight_tensor, bias_tensor), kwargs_dict
    else:
        return (input_tensor, weight_tensor), kwargs_dict


def conv2d_pp(args_list, kwargs_dict):
    input_tensor = require_ttnn_tensor(
        kwargs_dict["input_tensor"], "ttnn.conv2d input_tensor"
    )
    weight_tensor = require_ttnn_tensor(
        kwargs_dict["weight_tensor"], "ttnn.conv2d weight_tensor"
    )
    bias_tensor = require_ttnn_tensor(
        kwargs_dict["bias_tensor"], "ttnn.conv2d bias_tensor"
    )
    strides = kwargs_dict.get("stride", (1, 1))
    padding_size = kwargs_dict["padding"][0]
    pads = [padding_size for i in range(4)]
    kwargs_dict = {
        "pads": pads,
        "kernel_shape": list(kwargs_dict["kernel_size"]),
        "strides": list(strides),
    }
    return (input_tensor, weight_tensor, bias_tensor), kwargs_dict


def outer_pp(args_list, kwargs_dict):
    """Preprocessor for outer product operation."""
    assert len(args_list) == 2, "ttnn.outer has 2 inputs"
    tensor_a = require_ttnn_tensor(args_list[0], "ttnn.outer input a")
    tensor_b = require_ttnn_tensor(args_list[1], "ttnn.outer input b")
    assert tensor_a.shape is not None and tensor_b.shape is not None

    # Validate that inputs are 1D tensors
    if len(tensor_a.shape) != 1:
        raise ValueError(
            f"ttnn.outer expects 1D tensors, got tensor_a with shape {tensor_a.shape}"
        )
    if len(tensor_b.shape) != 1:
        raise ValueError(
            f"ttnn.outer expects 1D tensors, got tensor_b with shape {tensor_b.shape}"
        )

    # Output shape will be (len(tensor_a), len(tensor_b))
    output_shape = [tensor_a.shape[0], tensor_b.shape[0]]
    kwargs_dict["output_shape"] = output_shape

    return (tensor_a.unsqueeze(1), tensor_b.unsqueeze(0)), kwargs_dict


def torchgather_pp(args_list, kwargs_dict):
    """Preprocessor for torch gather operation. Torch Gather differs vs. ONNX Gather."""
    assert len(args_list) == 3, "ttnn.gather has 3 inputs"
    input_tensor = require_ttnn_tensor(args_list[0], "ttnn.gather input")
    dim = args_list[1]
    index_tensor = require_ttnn_tensor(args_list[2], "ttnn.gather index")

    # Ensure the index tensor is of integer type
    if index_tensor.dtype != np.int64:
        raise ValueError(
            f"ttnn.gather expects index tensor to be of type int64, got {index_tensor.dtype}"
        )

    kwargs_dict["axis"] = dim
    return (input_tensor, index_tensor), kwargs_dict


def transpose_pp(args_list, kwargs_dict):
    inT = require_ttnn_tensor(args_list[0], "ttnn.transpose input")
    dim1 = args_list[1]
    dim2 = args_list[2]
    in_rank = len(
        require_shape_list(inT.shape, "ttnn.transpose input shape must be set")
    )
    out_dims = [i for i in range(in_rank)]
    out_dims[dim2] = dim1
    out_dims[dim1] = dim2
    kwargs_dict = {"perm": out_dims}
    return ([inT], kwargs_dict)


def cat(tensors, dim=0):
    """Concatenate a list of tensors along a specified dimension.

    Delegates to ``concat`` (which emits a ``Concat`` SimOp) so that the
    operation is visible in the device graph regardless of whether the
    caller spells it ``ttnn.cat`` or ``ttnn.concat``.
    """
    return concat(*tensors, axis=dim)


def where_pp(args, kwargs_dict):
    mask_tensor = require_ttnn_tensor(args[0], "ttnn.where condition")
    input_tensor = require_ttnn_tensor(args[1], "ttnn.where x")
    value_tensor = require_ttnn_tensor(args[2], "ttnn.where y")
    return (mask_tensor, input_tensor, value_tensor), kwargs_dict


def rms_norm(
    input_tensor,
    weight_tensor=None,
    bias_tensor=None,
    epsilon=1e-6,
    memory_config=None,
    compute_kernel_config=None,
    dim=3072,
):
    input_tensor = require_ttnn_tensor(input_tensor, "ttnn.rms_norm input")
    if weight_tensor is not None:
        weight_tensor = require_ttnn_tensor(weight_tensor, "ttnn.rms_norm weight")
    if bias_tensor is not None:
        bias_tensor = require_ttnn_tensor(bias_tensor, "ttnn.rms_norm bias")
    # HW computes RMS-normalization (gamma scale + optional bias) in a SINGLE
    # LayerNormDeviceOperation, so emit one op to match the silicon capture. The previous
    # layer_norm + div + multiply decomposition over-emitted (3 ops vs 1 on HW) and was not a
    # sound mimic. `dim` is retained for call-site compatibility but is no longer needed.
    return layer_norm(
        input_tensor,
        weight=weight_tensor,
        bias=bias_tensor,
        epsilon=epsilon,
        axis=-1,
        memory_config=memory_config,
        compute_kernel_config=compute_kernel_config,
    )


def max_pool2d_pp(args_list, kwargs_dict):
    input_tensor = require_ttnn_tensor(
        kwargs_dict["input_tensor"], "ttnn.max_pool2d input_tensor"
    )
    kernel_size = kwargs_dict["kernel_size"]
    stride = kwargs_dict.get("stride", kernel_size)
    padding = kwargs_dict.get("padding", (0, 0))
    dilation = kwargs_dict.get("dilation", (1, 1))
    ceil_mode = kwargs_dict.get("ceil_mode", False)

    kwargs_dict = {
        "kernel_shape": list(kernel_size),
        "strides": list(stride),
        "pads": [
            padding[0],
            padding[0],
            padding[1],
            padding[1],
        ],  # [pad_top, pad_left, pad_bottom, pad_right]
        "dilations": list(dilation),
        "ceil_mode": ceil_mode,
    }
    return (input_tensor,), kwargs_dict


def conv_transpose2d_pp(args_list, kwargs_dict):
    input_tensor = require_ttnn_tensor(
        kwargs_dict["input_tensor"], "ttnn.conv_transpose2d input_tensor"
    )
    weight_tensor = require_ttnn_tensor(
        kwargs_dict["weight_tensor"], "ttnn.conv_transpose2d weight_tensor"
    )
    bias_tensor = require_ttnn_tensor(
        kwargs_dict["bias_tensor"], "ttnn.conv_transpose2d bias_tensor"
    )
    padding_size = kwargs_dict["padding"][0]
    pads = [padding_size for i in range(4)]
    output_padding = kwargs_dict.get("output_padding", (0, 0))
    strides = kwargs_dict.get("stride", (1, 1))
    kwargs_dict = {
        "padding": pads,
        "kernel_size": list(kwargs_dict["kernel_size"]),
        "output_padding": list(output_padding),
        "strides": list(strides),
    }
    return (input_tensor, weight_tensor, bias_tensor), kwargs_dict


def slice_spec_out_shape(in_shape, slice_spec):
    """Output shape of numpy basic indexing `slice_spec` applied to shape `in_shape`.

    Computed analytically rather than by materialising a dummy `np.empty(in_shape)` and
    reading `dummy[slice_spec].shape`: the decode sampling path slices logits of shape
    [1, 1, 32, vocab], so the dummy would be tens of MB of address space per call.

    Only basic indexing (ints, slices, Ellipsis) is handled — that is the whole of what the
    hardware SliceDeviceOperation expresses, and advanced indexing would not be a view.
    """
    spec = slice_spec if isinstance(slice_spec, tuple) else (slice_spec,)

    n_ellipsis = builtins.sum(1 for s in spec if s is Ellipsis)
    assert n_ellipsis <= 1, f"ttnn.slice: at most one Ellipsis allowed, got {slice_spec}"
    if n_ellipsis:
        pos = spec.index(Ellipsis)
        fill = len(in_shape) - (len(spec) - 1)
        assert fill >= 0, f"ttnn.slice: spec {slice_spec} over-indexes shape {in_shape}"
        spec = spec[:pos] + (builtins.slice(None),) * fill + spec[pos + 1:]

    assert len(spec) <= len(in_shape), f"ttnn.slice: spec {slice_spec} over-indexes shape {in_shape}"
    # dims the spec does not mention are kept whole
    spec = spec + (builtins.slice(None),) * (len(in_shape) - len(spec))

    out_shape = []
    for dim, s in zip(in_shape, spec):
        if isinstance(s, builtins.slice):
            out_shape.append(len(range(*s.indices(dim))))
        else:
            # an integer index drops the dimension
            idx = int(s)
            assert -dim <= idx < dim, f"ttnn.slice: index {idx} out of range for dim of size {dim}"
    return out_shape


def as_pp(args_list, kwargs_dict):
    input_tensor = require_ttnn_tensor(args_list[0], "ttnn.slice input")
    slice_spec = kwargs_dict.get("slice", None)
    assert (
        slice_spec is not None
    ), "ttnn.slice requires 'slice' attribute specifying indices"

    in_shape = require_shape_list(
        input_tensor.shape, "ttnn.slice input shape must be set"
    )
    kwargs_dict["output_shape"] = slice_spec_out_shape(in_shape, slice_spec)
    return (input_tensor,), kwargs_dict


def topk_pp(args_list, kwargs_dict):
    input_tensor = require_ttnn_tensor(args_list[0], "ttnn.topk input")
    axis = kwargs_dict.get("dim", -1)
    largest = kwargs_dict.get("largest", True)
    sorted = kwargs_dict.get("sorted", True)

    # k may arrive as: a positional K-tensor (ONNX-style ttnn.topk(x, k_tensor)),
    # a positional int (ttnn.topk(x, 32)), or a k= int kwarg (real ttnn
    # ttnn.topk(x, k=32, ...)).
    inputs: tuple[Tensor, ...] = (input_tensor,)
    k_val = None
    if len(args_list) > 1:
        second = args_list[1]
        if isinstance(second, Tensor):
            k_val = require_shape_list(second.shape, "ttnn.topk k shape must be set")[0]
            inputs = (input_tensor, second)
        elif isinstance(second, int):
            k_val = int(second)
    if k_val is None:
        k_kw = kwargs_dict.get("k")
        assert k_kw is not None, "ttnn.topk requires k (positional tensor/int or k= int)"
        k_val = int(k_kw)

    # Real ttnn passes a pre-allocated index operand via indices_tensor= (uint16), which the HW
    # TopKDeviceOperation takes as its SECOND input (used to track original positions across a
    # multi-step reduction). Record it so the sim op is arity-2 like the capture; topk_sinf keys
    # k off the attr, and skips the ONNX K-tensor path for a non-scalar index operand. (sub_core_grids
    # remains a HW-only hint and is dropped.)
    indices_tensor = kwargs_dict.get("indices_tensor")
    if indices_tensor is not None and isinstance(indices_tensor, Tensor):
        inputs = inputs + (indices_tensor,)

    new_kwargs = {
        "k": k_val,
        "axis": axis,
        "largest": 1 if largest else 0,
        "sorted": 1 if sorted else 0,
    }
    return inputs, new_kwargs


def zeros_like(input_tensor, memory_config=None):
    require_ttnn_tensor(input_tensor, "ttnn.zeros_like input")
    return zeros(
        shape=input_tensor.shape,
        dtype=DataType.from_numpy(input_tensor.dtype.name),
        layout=(
            Layout.from_numpy(input_tensor.layout.name)
            if input_tensor.layout
            else Layout.TILE_LAYOUT
        ),
        device=input_tensor.device,
    )


def compare(input_tensor_a, input_tensor_b, op_type):
    require_ttnn_tensor(input_tensor_a, "ttnn.compare input a")
    require_ttnn_tensor(input_tensor_b, "ttnn.compare input b")
    assert op_type in [
        "equal",
        "not_equal",
        "greater",
        "less",
        "greater_equal",
        "less_equal",
    ], f"Unsupported compare op_type: {op_type}"
    # For simplicity, we return a tensor of the same shape with boolean dtype
    return Tensor(
        shape=input_tensor_a.shape, dtype=DataType.BOOL, device=input_tensor_a.device
    )


def maximum(input_tensor_a, input_tensor_b):
    require_ttnn_tensor(input_tensor_a, "ttnn.maximum input a")
    require_ttnn_tensor(input_tensor_b, "ttnn.maximum input b")
    return input_tensor_a  # Placeholder implementation


def unsqueeze(input_tensor, dim):
    require_ttnn_tensor(input_tensor, "ttnn.unsqueeze input")
    return input_tensor.unsqueeze(dim)


def divide(input_tensor, divisor, use_legacy=False):
    require_ttnn_tensor(input_tensor, "ttnn.divide input")
    if isinstance(divisor, (int, float)):
        divisor = Tensor(
            shape=input_tensor.shape,
            dtype=DataType.from_numpy(input_tensor.dtype.name),
            device=input_tensor.device,
            data=np.full(input_tensor.shape, divisor, dtype=input_tensor.dtype),
        )
    require_ttnn_tensor(divisor, "ttnn.divide divisor")
    return div(input_tensor, divisor)


def clone(input_tensor, memory_config=None):
    require_ttnn_tensor(input_tensor, "ttnn.clone input")
    return input_tensor


def squeeze(input_tensor, dim):
    require_ttnn_tensor(input_tensor, "ttnn.squeeze input")
    return input_tensor.squeeze(dim)


def repeat(input_tensor, repeats):
    require_ttnn_tensor(input_tensor, "ttnn.repeat input")
    output_shape = [i * j for i, j in zip(list(input_tensor.shape), repeats)]
    return Tensor(
        shape=output_shape,
        dtype=DataType.from_numpy(input_tensor.dtype.name),
        device=input_tensor.device,
    )


def _sdpa_config_attrs(program_config, compute_kernel_config):
    """Scalar attrs from an SDPA program config and compute kernel config, duck-typed so real ttnn
    objects and the shim dataclasses both work. Fields that are missing go into sdpa_defaulted."""
    a: dict = {}
    defaulted: list = []
    if program_config is not None:
        a['q_chunk_size'] = int(program_config.q_chunk_size)
        a['k_chunk_size'] = int(program_config.k_chunk_size)
        scg = getattr(program_config, 'sub_core_grids', None)
        if scg is not None and hasattr(scg, 'num_cores'):
            a['num_cores'] = int(scg.num_cores())
        else:
            g = program_config.compute_with_storage_grid_size
            gx, gy = (g.x, g.y) if hasattr(g, 'x') else (g[0], g[1])
            a['num_cores'] = int(gx) * int(gy)
        exp = getattr(program_config, 'exp_approx_mode', None)
        if exp is None:
            defaulted.append('exp_approx_mode')
        else:
            a['exp_approx_mode'] = bool(exp)
        a['max_cores_per_head_batch'] = int(getattr(program_config, 'max_cores_per_head_batch', 16) or 16)
    else:
        defaulted += ['q_chunk_size', 'k_chunk_size', 'num_cores', 'exp_approx_mode']
    mf = getattr(compute_kernel_config, 'math_fidelity', None)
    if mf is None and isinstance(compute_kernel_config, MathFidelity):
        mf = compute_kernel_config
    if mf is not None:
        a['fidelity'] = mf.name if hasattr(mf, 'name') else str(mf)
        for f in ('fp32_dest_acc_en', 'packer_l1_acc', 'math_approx_mode'):
            v = getattr(compute_kernel_config, f, None)
            if v is None:
                defaulted.append(f)
            else:
                a[f] = bool(v)
    else:
        defaulted += ['fidelity', 'fp32_dest_acc_en', 'packer_l1_acc', 'math_approx_mode']
    a['sdpa_defaulted'] = ','.join(defaulted)
    return a


def _shard_cores(t):
    """Core count of a sharded tensor's grid when the memory config carries one, else 0."""
    mc = t.memory_config() if hasattr(t, 'memory_config') else None
    spec = getattr(mc, 'shard_spec', None)
    grid = getattr(spec, 'grid', None) if spec is not None else None
    if grid is None:
        return 0
    if hasattr(grid, 'num_cores'):
        try:
            return int(grid.num_cores())
        except TypeError:
            return int(grid.num_cores)
    try:
        return len(grid)
    except TypeError:
        return 0


def _tensor_in_l1(t):
    """True when the tensor's memory config places it in L1; an unset config reads as DRAM."""
    mc = t.memory_config() if hasattr(t, 'memory_config') else None
    return getattr(mc, 'buffer_type', None) == BufferType.L1


def _sdpa_decode_attrs(program_config, compute_kernel_config, k, *, q=None, is_causal=True, attn_mask=None,
                       cur_pos=None, cur_pos_tensor=None, attention_sink=None, sliding_window_size=None,
                       page_table_tensor=None, paged_cache_geometry=None, cache_position_modulo=None):
    """Decode-side attrs: configs plus the position, mask, sink and paging fields the roofline prices from."""
    a = _sdpa_config_attrs(program_config, compute_kernel_config)
    a['is_causal'] = bool(is_causal)
    # An L1 sharded query (tt_transformers decode) is not DRAM traffic for the roofline.
    if q is not None and _tensor_in_l1(q):
        a['q_in_l1'] = True
        cores = _shard_cores(q)
        if cores:
            a['q_shard_cores'] = cores
    if attn_mask is not None:
        a['has_attn_mask'] = True
    if attention_sink is not None:
        a['attention_sink'] = True
    if sliding_window_size:
        a['sliding_window_size'] = int(sliding_window_size)
    # List form: the slowest user bounds the wall. Tensor form carries no host-side value.
    if cur_pos is not None and len(cur_pos) > 0:
        a['cur_pos'] = int(builtins.max(cur_pos))
    else:
        a['cur_pos_unknown'] = True
    if page_table_tensor is not None:
        a['paged'] = True
        bs = getattr(paged_cache_geometry, 'block_size', 0) or 0
        if bs:
            a['page_block_size'] = int(bs)
    if cache_position_modulo:
        a['cache_position_modulo'] = int(cache_position_modulo)
    a['kv_element_size'] = k.element_size()
    return a


class transformer:
    def __init__(self, config):
        pass

    @staticmethod
    def scaled_dot_product_attention(q, k, v, attn_mask=None, *, is_causal=True, scale=None,
                                     sliding_window_size=None, memory_config=None, program_config=None,
                                     compute_kernel_config=None, attention_sink=None,
                                     cu_window_seqlens=None, **kwargs):
        """Prefill SDPA; output = q shape. Program and compute configs land in attrs as scalars."""
        from .ttnn_shim import scaled_dot_product_attention_op as _sdpa
        # Pass scale through unchanged: when the caller omits it (None), the shim op
        # drops it from the recorded attrs rather than fabricating a 0.0.
        # chunk_start_idx 0 is a valid first chunk, so test for presence, not truth.
        chunk_start = kwargs.get('chunk_start_idx')
        variant = 'chunked' if chunk_start is not None else 'prefill'
        extra = _sdpa_config_attrs(program_config, compute_kernel_config)
        if chunk_start is not None:
            extra['chunk_start_idx'] = int(chunk_start)
        if sliding_window_size:
            extra['sliding_window_size'] = int(sliding_window_size)
        if attn_mask is not None:
            extra['has_attn_mask'] = True
        if attention_sink is not None:
            extra['attention_sink'] = True
        if cu_window_seqlens is not None:
            extra['is_windowed'] = True
        extra['kv_element_size'] = k.element_size()
        # The optional tensors are inputs of the op, not only flags, so their producers stay connected.
        return _sdpa(q, k, v, attn_mask, attention_sink, cu_window_seqlens, memory_config=memory_config,
                     is_causal=bool(is_causal), scale=scale, sdpa_variant=variant, **extra)

    @staticmethod
    def chunked_scaled_dot_product_attention(q, k, v, page_table_tensor, *, chunk_start_idx=None,
                                             chunk_start_idx_tensor=None, scale=None, memory_config=None,
                                             program_config=None, compute_kernel_config=None,
                                             paged_cache_geometry=None, **kwargs):
        """Chunked prefill over a paged prefix; output = q shape. The page table rides along as an
        input; a tensor-only chunk start has no host value and is flagged."""
        from .ttnn_shim import scaled_dot_product_attention_op as _sdpa
        extra = _sdpa_config_attrs(program_config, compute_kernel_config)
        if chunk_start_idx is not None:
            extra['chunk_start_idx'] = int(chunk_start_idx)
        elif chunk_start_idx_tensor is not None:
            extra['chunk_start_unknown'] = True
        extra['paged'] = True
        bs = getattr(paged_cache_geometry, 'block_size', 0) or 0
        if bs:
            extra['page_block_size'] = int(bs)
        extra['kv_element_size'] = k.element_size()
        return _sdpa(q, k, v, page_table_tensor, chunk_start_idx_tensor, memory_config=memory_config,
                     is_causal=True, scale=scale, sdpa_variant='chunked', **extra)

    @staticmethod
    def scaled_dot_product_attention_decode(q, k, v, *, is_causal=True, attn_mask=None, cur_pos=None,
                                            cur_pos_tensor=None, attention_sink=None, scale=None,
                                            sliding_window_size=None, memory_config=None,
                                            program_config=None, compute_kernel_config=None,
                                            share_cache=None, **kwargs):
        """Decode SDPA (non-paged); output = q shape."""
        from .ttnn_shim import scaled_dot_product_attention_op as _sdpa
        # Explicit tag: with cur_pos_tensor=None the shim drops the input and the op would
        # otherwise be classified by arity as a 3-input prefill.
        extra = _sdpa_decode_attrs(program_config, compute_kernel_config, k, q=q, is_causal=is_causal,
                                   attn_mask=attn_mask, cur_pos=cur_pos, cur_pos_tensor=cur_pos_tensor,
                                   attention_sink=attention_sink, sliding_window_size=sliding_window_size)
        # The mask and sink ride along as inputs so their producers stay connected in the graph.
        return _sdpa(q, k, v, cur_pos_tensor, attn_mask, attention_sink, memory_config=memory_config,
                     scale=scale, sdpa_variant='decode', **extra)

    @staticmethod
    def paged_scaled_dot_product_attention_decode(q, k, v, page_table_tensor=None, *, is_causal=True,
                                                  attn_mask=None, cur_pos=None, cur_pos_tensor=None,
                                                  attention_sink=None, scale=None, sliding_window_size=None,
                                                  memory_config=None, program_config=None,
                                                  compute_kernel_config=None, paged_cache_geometry=None,
                                                  cache_position_modulo=None, **kwargs):
        """Paged decode SDPA; output = q shape."""
        from .ttnn_shim import scaled_dot_product_attention_op as _sdpa
        extra = _sdpa_decode_attrs(program_config, compute_kernel_config, k, q=q, is_causal=is_causal,
                                   attn_mask=attn_mask, cur_pos=cur_pos, cur_pos_tensor=cur_pos_tensor,
                                   attention_sink=attention_sink, sliding_window_size=sliding_window_size,
                                   page_table_tensor=page_table_tensor,
                                   paged_cache_geometry=paged_cache_geometry,
                                   cache_position_modulo=cache_position_modulo)
        # Mask and sink go before the page table: the decode roofline reads the table as the last input.
        return _sdpa(q, k, v, cur_pos_tensor, attn_mask, attention_sink, page_table_tensor,
                     memory_config=memory_config,
                     scale=scale, sdpa_variant='decode', **extra)

    @staticmethod
    def flash_mla_prefill(q, k, *, head_dim_v, scale=None, is_causal=True,
                          program_config=None, compute_kernel_config=None,
                          memory_config=None, **kwargs):
        """FlashMLA prefill (latent form): V = K[..., :head_dim_v], so K serves as V. Routes to the
        MLA roofline via the head_dim_v attr (asymmetric d_qk/d_v)."""
        from .ttnn_shim import scaled_dot_product_attention_op as _sdpa
        # Without configs fall back to the calibration family (HiFi4, accurate exp), listed as defaulted.
        extra = _sdpa_config_attrs(program_config, compute_kernel_config)
        extra.setdefault('fidelity', 'HiFi4')
        extra.setdefault('exp_approx_mode', False)
        extra['kv_element_size'] = k.element_size()
        return _sdpa(q, k, k, memory_config=memory_config, sdpa_variant='mla',
                     head_dim_v=int(head_dim_v), is_causal=bool(is_causal), scale=scale, **extra)

    @staticmethod
    def flash_mla_decode(q, k, v=None, *, head_dim_v, cur_pos_tensor=None, page_table_tensor=None,
                         cur_pos=None, scale=None, program_config=None, compute_kernel_config=None,
                         paged_cache_geometry=None, memory_config=None, **kwargs):
        """FlashMLA decode (memory-bound). Without a V tensor the kernel reuses K's latent part as V and
        streams K alone; a V tensor is streamed too. Routes to the decode roofline."""
        from .ttnn_shim import scaled_dot_product_attention_op as _sdpa
        extra = _sdpa_decode_attrs(program_config, compute_kernel_config, k, q=q, cur_pos=cur_pos,
                                   cur_pos_tensor=cur_pos_tensor, page_table_tensor=page_table_tensor,
                                   paged_cache_geometry=paged_cache_geometry)
        if v is not None:
            extra['mla_v_read'] = True
        return _sdpa(q, k, v if v is not None else k, cur_pos_tensor, page_table_tensor, memory_config=memory_config,
                     sdpa_variant='mla_decode', head_dim_v=int(head_dim_v), scale=scale, **extra)

    @staticmethod
    def sparse_sdpa(q, kv, sparse_idx, head_dim_v, *, scale=None, is_causal=False, k_chunk_size=None,
                    program_config=None, compute_kernel_config=None, memory_config=None, **kwargs):
        """Sparse-MLA: each query attends its TOPK selected latent KV (indices [..., TOPK]). Routes to
        the roofline as MLA cross-attention with kv_seq=TOPK + the scattered-gather uplift."""
        from .ttnn_shim import scaled_dot_product_attention_op as _sdpa
        topk = int(sparse_idx.logical_shape()._shape[-1])
        extra = _sdpa_config_attrs(program_config, compute_kernel_config)
        # The sparse kernel takes k_chunk_size (default 128) and has no q chunk parameter, so the
        # 32x32 prefill fallback does not apply; price at the calibration chunking and say so.
        defaulted = [d for d in extra['sdpa_defaulted'].split(',') if d]
        if k_chunk_size is not None:
            extra['k_chunk_size'] = int(k_chunk_size)
            defaulted = [d for d in defaulted if d != 'k_chunk_size']
        for key in ('q_chunk_size', 'k_chunk_size'):
            if key not in extra:
                extra[key] = 128
                if key not in defaulted:
                    defaulted.append(key)
        extra['sdpa_defaulted'] = ','.join(defaulted)
        extra.setdefault('fidelity', 'HiFi4')
        extra.setdefault('exp_approx_mode', False)
        extra['kv_element_size'] = kv.element_size()
        return _sdpa(q, kv, kv, sparse_idx, memory_config=memory_config, sdpa_variant='sparse',
                     is_sparse=True, head_dim_v=int(head_dim_v), kv_seq=topk, is_causal=bool(is_causal),
                     scale=scale, **extra)

    @staticmethod
    def joint_scaled_dot_product_attention(q, k, v, joint_tensor_q, joint_tensor_k, joint_tensor_v,
                                           *, joint_strategy=None, scale=None, program_config=None,
                                           compute_kernel_config=None, memory_config=None, **kwargs):
        """Joint SDPA (SD3/Flux): main + joint streams attend over their concatenation, non-causal.
        Routes to the joint roofline (cost over S_eff = main + joint). The shim op is single-output,
        so the second element is joint_tensor_q passed through as a placeholder, not a computed joint_out."""
        from .ttnn_shim import scaled_dot_product_attention_op as _sdpa
        joint_seq = int(joint_tensor_q.logical_shape()._shape[-2])
        extra = _sdpa_config_attrs(program_config, compute_kernel_config)
        extra['kv_element_size'] = k.element_size()
        out = _sdpa(q, k, v, memory_config=memory_config, sdpa_variant='joint',
                    joint_seq=joint_seq, is_causal=False, scale=scale, **extra)
        return out, joint_tensor_q


class experimental:
    def __init__(self):
        pass

    def all_gather_matmul(self, *args, **kwargs):
        pass

    @staticmethod
    def nlp_create_qkv_heads(input_tensor, kv_input_tensor=None, *,
                              num_heads, num_kv_heads=None,
                              transpose_k_heads=False, memory_config=None):
        """Delegate to ttnn_shim; mirrors HW's single-op QKV head split."""
        from .ttnn_shim import nlp_create_qkv_heads as _nlp_create_qkv_heads
        return _nlp_create_qkv_heads(
            input_tensor, kv_input_tensor,
            num_heads=num_heads, num_kv_heads=num_kv_heads,
            transpose_k_heads=transpose_k_heads, memory_config=memory_config,
        )

    @staticmethod
    def nlp_concat_heads(input_tensor, memory_config=None):
        """Delegate to ttnn_shim; mirrors HW's single-op head concatenation."""
        from .ttnn_shim import nlp_concat_heads as _nlp_concat_heads
        return _nlp_concat_heads(input_tensor, memory_config=memory_config)

    @staticmethod
    def rotary_embedding_llama(x, cos, sin, trans_mat, is_decode_mode=False,
                               memory_config=None):
        """Prefill rotary embedding (called separately for q and k); output = x shape."""
        from .ttnn_shim import rotary_embedding_llama_op as _rope
        return _rope(x, cos, sin, trans_mat, is_decode_mode=is_decode_mode,
                     memory_config=memory_config)

    @staticmethod
    def rotary_embedding_llama_fused_qk(q, k, cos, sin, trans_mat, memory_config=None):
        """Decode fused rotary embedding; returns (q, k), each = its input shape."""
        from .ttnn_shim import rotary_embedding_llama_fused_qk_op as _rope_qk
        return _rope_qk(q, k, cos, sin, trans_mat, memory_config=memory_config)

    @staticmethod
    def paged_fill_cache(cache, input_tensor, page_table=None, *, batch_idx=0,
                         memory_config=None):
        """Prefill KV-cache fill (in place); output = cache shape."""
        from .ttnn_shim import paged_fill_cache_op as _fill
        return _fill(cache, input_tensor, page_table=page_table, batch_idx=batch_idx,
                     memory_config=memory_config)

    @staticmethod
    def paged_fused_update_cache(k_cache, k, v_cache, v, *, update_idxs_tensor=None,
                                 page_table=None, memory_config=None):
        """Decode fused KV-cache update (in place); returns (k_cache, v_cache)."""
        from .ttnn_shim import paged_fused_update_cache_op as _upd
        return _upd(k_cache, k, v_cache, v, update_idxs_tensor=update_idxs_tensor,
                    page_table=page_table, memory_config=memory_config)

    @staticmethod
    def nlp_create_qkv_heads_decode(input_tensor, *, num_heads, num_kv_heads=None,
                                    head_dim=None, memory_config=None, **kwargs):
        """Decode QKV head split; returns (q, k, v) with heads on the Y axis."""
        from .ttnn_shim import nlp_create_qkv_heads_decode_op as _create
        return _create(input_tensor, num_heads=num_heads, num_kv_heads=num_kv_heads,
                       head_dim=head_dim, memory_config=memory_config)

    @staticmethod
    def nlp_concat_heads_decode(input_tensor, *, num_heads=None, memory_config=None,
                                **kwargs):
        """Decode head concatenation; output folds head_dim into the X axis."""
        from .ttnn_shim import nlp_concat_heads_decode_op as _concat
        return _concat(input_tensor, num_heads=num_heads, memory_config=memory_config)


def all_gather(*args, **kwargs):
    raise NotImplementedError("all_gather is not implemented yet!!")


def eqz(input_tensor):
    return compare(input_tensor, zeros_like(input_tensor), op_type='equal')


def moe(gate_logits, expert_mask, topE_mask, k, k_tensor):
    gate_logits = require_ttnn_tensor(gate_logits, "ttnn.moe gate_logits")
    expert_mask = require_ttnn_tensor(expert_mask, "ttnn.moe expert_mask")
    topE_mask = require_ttnn_tensor(topE_mask, "ttnn.moe topE_mask")
    k_tensor = require_ttnn_tensor(k_tensor, "ttnn.moe k_tensor")
    N, C, H, W = gate_logits.shape
    assert expert_mask.shape == [N, C, 1, W], "expert_mask must be [N, C, 1, W]"
    assert topE_mask.shape[-1] == k, "topE_mask last dim must be k"

    # 1) Apply expert_mask to zero out padded experts (set to -inf)
    #    Broadcast over H: [N,C,H,W] + [N,C,1,W] -> [N,C,H,W]
    masked_logits = sum(gate_logits, expert_mask)

    # 2) Top-k over experts (last dimension)
    topk_values, topk_indices = topk(masked_logits, k_tensor, dim=-1)  # [N,C,H,k]

    # 3) Apply topE_mask (implements "e" effective experts within top-k)
    #    topE_mask: [N,C,1,k] -> broadcast to [N,C,H,k]
    topk_scores = softmax(topk_values + topE_mask, dim=-1)      # [N,C,H,k]

    # 4) Select only entries that correspond to `target_expert`
    expert_selector = eqz(topk_indices)       # [N,C,H,k]

    # 5) Aggregate weight for that expert over its (up to e) positions in top-k
    #    Result is shape [N,C,H,1]
    weights = sum(topk_scores * expert_selector, dim=-1)
    weights = unsqueeze(weights, -1)

    return weights


# Pointwise Unary
cos         = single_output_immediate_op('Cos')
gelu        = single_output_immediate_op('Gelu')
identity    = single_output_immediate_op('Identity')
leaky_relu  = single_output_immediate_op('LeakyRelu')
neg         = single_output_immediate_op('Neg')
relu        = single_output_immediate_op('Relu')
sigmoid     = single_output_immediate_op('Sigmoid')
sin         = single_output_immediate_op('Sin')
softmax     = single_output_immediate_op('Softmax')
tanh        = single_output_immediate_op('Tanh')
clamp       = single_output_immediate_op('Clip')
log         = single_output_immediate_op('Log')
min         = single_output_immediate_op('Min')
max         = single_output_immediate_op('Max')
sqrt        = single_output_immediate_op('Sqrt')

class UnaryOpType:
    """Fused unary activations for binary ops — mirrors ttnn.UnaryOpType.

    Pass e.g. ``input_tensor_a_activations=[UnaryOpType.SILU]`` to a binary op
    (``multiply``/``add``) so the fused activation is modeled as a SINGLE op — the HW
    BinaryNg carries the activation (cf. tt-metal mlp.py:239 SILU-fused gate*up). The
    kwarg is forwarded into the SimOp attrs by single_output_immediate_op, so NO
    separate activation op is emitted (which would over-emit vs HW). Values mirror the
    ttnn.UnaryOpType enum names.
    """
    SILU = 'SILU'
    GELU = 'GELU'
    RELU = 'RELU'
    SIGMOID = 'SIGMOID'
    TANH = 'TANH'


# Pointwise Binary
add = single_output_immediate_op("Add")
sub = single_output_immediate_op("Sub")
multiply = single_output_immediate_op("Mul")
subtract = single_output_immediate_op("Sub")
div = single_output_immediate_op("Div")
pow = single_output_immediate_op("Pow")
mean = single_output_immediate_op("Mean")
sum = single_output_immediate_op("Sum")
atan = single_output_immediate_op("Atan")
exp = single_output_immediate_op("Exp")

# Pointwise Ternary
where = single_output_immediate_op("Where", preprocess=where_pp)
nonzero = single_output_immediate_op("NonZero")

# Reduction
argmax = single_output_immediate_op("ArgMax", preprocess=argmax_pp)

# Data Movement
_concat_impl = single_output_immediate_op("Concat")


def concat(first, *rest, **kwargs):
    """Concatenate tensors along a dimension.

    Accepts both calling conventions so that TT-Polaris matches the tt-metal canonical form:

        ttnn.concat([t1, t2], dim, memory_config=cfg)   # tt-metal list-first form
        ttnn.concat(t1, t2, axis=dim)                    # existing positional form

    ``dim`` keyword is normalised to ``axis`` for the underlying implementation.
    """
    if 'dim' in kwargs and 'axis' not in kwargs:
        kwargs['axis'] = kwargs.pop('dim')
    if isinstance(first, (list, tuple)):
        tensors = list(first)
        # bool-before-Integral order is intentional: bool is a subclass of int (and therefore
        # numbers.Integral), so isinstance(True, numbers.Integral) is True.  We want to treat
        # a bare True/False as an unknown positional arg, not as a dimension.  Checking the bool
        # identity first also avoids a mypy false-positive ("unreachable" branch) that triggers
        # when the narrowing order is reversed.
        if rest and type(rest[0]) is not bool and isinstance(rest[0], numbers.Integral):
            if 'axis' in kwargs:
                raise TypeError(f'concat() got conflicting values for axis: positional {rest[0]} and keyword {kwargs["axis"]}')
            kwargs['axis'] = rest[0]
            rest = rest[1:]
        return _concat_impl(*tensors, *rest, **kwargs)
    return _concat_impl(first, *rest, **kwargs)


reshape = single_output_immediate_op("Reshape", preprocess=reshape_pp)
expand = single_output_immediate_op("Expand", preprocess=expand_pp)
embedding = single_output_immediate_op("Embedding", preprocess=embedding_pp)
permute = single_output_immediate_op("Transpose", preprocess=permute_pp)
gather = single_output_immediate_op("TorchGather", preprocess=torchgather_pp)
transpose = single_output_immediate_op("Transpose", preprocess=transpose_pp)
split = multiple_output_immediate_op("Split", preprocess=split_pp)
# ttnn.slice(x, slice=(...)) — arity-1 Slice op (bounds via the 'slice' spec -> output_shape attr,
# consumed by slice_sinf's arity-1 branch). Matches HW SliceDeviceOperation (arity-1).
slice = single_output_immediate_op("Slice", preprocess=as_pp)

# Normalization
layer_norm = single_output_immediate_op("LayerNormalization", preprocess=layer_norm_pp)
batch_norm = single_output_immediate_op("BatchNormalization")

# Halo: auto-emitted by the shim before every conv2d / pool2d / conv_transpose2d,
# mirroring the hardware dispatch where halo extraction is implicit inside those ops.
halo = single_output_immediate_op("Halo")

# InterleavedToSharded: auto-emitted when conv/pool receives an interleaved tensor,
# mirroring the hardware dispatch where the kernel internally converts to sharded
# before halo extraction.
_interleaved_to_sharded = single_output_immediate_op("InterleavedToSharded")

# Move: auto-emitted after conv2d / conv_transpose2d when deallocate_activation=True.
# On hardware, MoveDeviceOperation copies the output buffer to a new memory region
# to free the old one.  The shim mirrors this by emitting a Move SimOp after the
# conv output, matching the hardware profiler op sequence.
_move = single_output_immediate_op("Move")


# Producers whose output is already a freshly allocated buffer — tt-metal's conv
# kernel skips its internal Move when the input came from one of these, because
# no reallocation is needed.  Used by ``_with_halo`` to suppress the Move that
# would otherwise be emitted before the conv.  (See HW Move LUT entries: the
# first conv of every stage following a MaxPool has no matching Move row.)
_HW_FRESH_BUFFER_PRODUCERS: frozenset[str] = frozenset({"MaxPool", "ConvTranspose"})


def _producer_optype(tensor) -> str:
    """Return the optype of the op that produced ``tensor``, or '' if unknown."""
    if tensor is None or getattr(tensor, "device", None) is None:
        return ""
    op_out = getattr(tensor, "op_out", None)
    if not op_out:
        return ""
    producer = tensor.device.ops.get(op_out[0])
    if producer is None:
        return ""
    return str(getattr(producer, "optype", ""))

# Pad: used by VGG UNet entry path to model the C=3 → C=16 channel pad that the
# hardware applies before the first conv (the LUT records this Pad).
_pad = single_output_immediate_op("Pad")


def pad_channels_nchw(input_tensor: 'Tensor', target_channels: int) -> 'Tensor':
    """Emit a Pad SimOp that grows the NCHW channel dim from in_shape[1] to target_channels.

    VGG UNet enters with C=3 but the canonical hardware form pads to C=16 before
    the first conv (see HW ttnn_vgg_unet.py).  The profiler records this Pad with
    the pre-pad input shape; emitting it here makes the LUT entry hit and shifts
    downstream Halo/Move/Conv onto the C=16 LUT entries that the hardware uses.

    The op is emitted as arity-1 (no pads_tensor in inList) so the LUT key matches
    the profiler's 9-tuple recording, with pad values carried in attrs.  The output
    is given an NCHW ``hw_shape`` ``[N, target_channels, H, W]`` so the *next*
    op's LUT key sees the post-Pad NCHW shape — the HW emits two Transposes
    after the Pad (NCHW → NHWC), and those Transposes' LUT entries record the
    NCHW input shape.  The Halo / Conv path then receives an NHWC-flat shape
    via the second Transpose's ``hw_shape``.  See ``permute_reshape_to_nhwc_flat``.
    """
    in_shape = list(input_tensor.shape)  # type: ignore[arg-type]
    assert len(in_shape) == 4 and in_shape[1] < target_channels, (
        f"pad_channels_nchw: NCHW input with C<target required; "
        f"got shape={in_shape}, target={target_channels}"
    )
    N, _, H, W = in_shape
    pad_amount = target_channels - in_shape[1]
    out_shape = [N, target_channels, H, W]

    op_name = generate_new_op_name()
    out_tensor = Tensor(
        name=op_name + '.out',
        shape=out_shape,
        dtype=input_tensor.dtype,
        layout=input_tensor.get_layout(),
        op_out=[op_name],
        device=input_tensor.device,
    )
    # hw_shape mirrors the logical NCHW shape so the first entry-path Transpose
    # presents a (1, C, H, W) input to its LUT lookup.
    #
    # TODO(batch>1): the entry-path Transpose LUT keys are documented with W=1
    # (batch folded into later NHWC-flat N*H*W). Multi-batch use needs either
    # (a) extending the LUT entries to cover N>1 keys, or (b) folding N into
    # H/W here so the existing keys still match. Today only batch_size=1 is
    # exercised; assert that explicitly so multi-batch use fails fast rather
    # than producing silent LUT-key mismatches downstream.
    assert int(out_shape[0]) == 1, (
        f"pad_channels_nchw currently supports batch_size=1 only "
        f"(got N={out_shape[0]}). See TODO above."
    )
    out_tensor.hw_shape = list(out_shape)
    input_tensor.op_in.append(op_name)

    opinfo = {
        'name': op_name,
        'optype': 'Pad',
        'inList': [input_tensor.name],
        'outList': [out_tensor.name],
        'attrs': {'pads': [0, 0, 0, 0, 0, pad_amount, 0, 0], 'mode': 'constant', 'value': 0},
    }
    opobj = SimOp(opinfo)

    # Manual perf stats (bypass pad_sinf which requires a pad_tensor input).
    # Pad is dtype-preserving; query the input tensor for the correct byte
    # width (handles BFLOAT8_B, BFLOAT4_B etc.) instead of hardcoding bfloat16.
    elem_size = input_tensor.element_size()
    nelems_in = 1
    for d in in_shape:
        nelems_in *= int(d)
    nelems_out = 1
    for d in out_shape:
        nelems_out *= int(d)
    opobj.perf_stats = {
        'inElems': nelems_in,
        'outElems': nelems_out,
        'inBytes': nelems_in * elem_size,
        'outBytes': nelems_out * elem_size,
        'instrs': {'mov': nelems_out},
    }
    # Pre-populating ``perf_stats`` short-circuits ``SimOp.get_perf_counts()``
    # and skips its post-shape-inference snapshot block. Mirror that snapshot
    # here so downstream stats serialization reads the frozen (post-pad) shapes
    # rather than live tensor state that later ops may mutate.
    opobj._frozen_input_shapes = [list(in_shape)]
    opobj._frozen_output_shapes = [list(out_shape)]
    opobj.update_tensor_counts([input_tensor], [out_tensor])

    _propagate_ttnn_dtype([input_tensor], [out_tensor])
    _propagate_memory_config([input_tensor], [out_tensor])

    input_tensor.device.add_op(opobj)  # type: ignore[union-attr]
    return out_tensor


def _emit_entry_transpose(input_tensor: 'Tensor', output_hw_shape: list) -> 'Tensor':
    """Emit a tracking-only Transpose SimOp with an explicit output ``hw_shape``.

    Used by ``permute_reshape_to_nhwc_flat`` to model the two HW transposes that
    convert NCHW (post-Pad) to NHWC-flat (pre-Halo).  Logical shape is a
    passthrough (TT-Polaris models conv math in NCHW); only ``hw_shape`` is
    advanced so the downstream LUT keys land on the right entries.
    """
    in_shape = list(input_tensor.shape)  # type: ignore[arg-type]
    op_name = generate_new_op_name()
    out_tensor = Tensor(
        name=op_name + '.out',
        shape=in_shape,
        dtype=input_tensor.dtype,
        layout=input_tensor.get_layout(),
        op_out=[op_name],
        device=input_tensor.device,
    )
    out_tensor.hw_shape = list(output_hw_shape)
    input_tensor.op_in.append(op_name)

    opinfo = {
        'name': op_name,
        'optype': 'Transpose',
        'inList': [input_tensor.name],
        'outList': [out_tensor.name],
        'attrs': {},
    }
    opobj = SimOp(opinfo)

    # Transpose is dtype-preserving; query the input tensor for the correct
    # byte width (handles BFLOAT8_B, BFLOAT4_B etc.) instead of hardcoding bfloat16.
    elem_size = input_tensor.element_size()
    nelems = 1
    for d in in_shape:
        nelems *= int(d)
    opobj.perf_stats = {
        'inElems': nelems,
        'outElems': nelems,
        'inBytes': nelems * elem_size,
        'outBytes': nelems * elem_size,
        'instrs': {'mov': nelems},
    }
    # Pre-populating ``perf_stats`` short-circuits ``SimOp.get_perf_counts()``
    # and skips its post-shape-inference snapshot block. Mirror that snapshot
    # here. Transpose's logical shape is a passthrough (only ``hw_shape``
    # differs), so input and output frozen shapes are both ``in_shape``.
    opobj._frozen_input_shapes = [list(in_shape)]
    opobj._frozen_output_shapes = [list(in_shape)]
    opobj.update_tensor_counts([input_tensor], [out_tensor])

    _propagate_ttnn_dtype([input_tensor], [out_tensor])
    _propagate_memory_config([input_tensor], [out_tensor])

    input_tensor.device.add_op(opobj)  # type: ignore[union-attr]
    return out_tensor


def permute_reshape_to_nhwc_flat(input_tensor: 'Tensor') -> 'Tensor':
    """Emit the two Transpose SimOps that follow Pad in the VGG UNet entry path.

    The HW profiler shows two TransposeDeviceOperation rows between Pad and the
    first Halo, converting NCHW [N, C, H, W] to NHWC-flat [1, 1, N*H*W, C].
    LUT entries record:
      T1 input: (1, C, H, W)         — post-Pad NCHW
      T2 input: (1, H, C, W)         — intermediate
    Output of T2 has hw_shape ``[1, 1, N*H*W, C]`` (NHWC-flat) for downstream
    Halo to consume.  Logical (NCHW) shape is preserved through both transposes
    so TT-Polaris's conv shape inference continues to operate in NCHW.
    """
    in_shape = list(input_tensor.shape)  # type: ignore[arg-type]
    assert len(in_shape) == 4, (
        f"permute_reshape_to_nhwc_flat expects rank-4 NCHW input; got shape={in_shape}"
    )
    N, C, H, W = in_shape
    t1 = _emit_entry_transpose(input_tensor, [1, H, C, W])
    t2 = _emit_entry_transpose(t1, [1, 1, N * H * W, C])
    return t2


def _with_halo(op_fn, is_transpose: bool = False, move_before_conv: bool = False):
    """Return a wrapper that auto-emits a Halo SimOp before the main op.

    Halo is skipped for 1×1 kernels: hardware implements those as matmul
    and never dispatches a halo extraction step.

    When the input has an interleaved memory config, an InterleavedToSharded
    SimOp is emitted first, matching the hardware dispatch where conv/pool
    kernels internally convert interleaved activations to sharded layout
    before halo extraction.

    Args:
        is_transpose: True when wrapping conv_transpose2d (sets is_transpose attr on Halo).
        move_before_conv: When True, emits Move after Halo but before the main op,
            matching the hardware op sequence ITS → Halo → Move → Conv.
            Only emits Move when deallocate_activation=True and the original
            input tensor is L1-sharded (same guard as _with_move).
    """
    def _impl(*args, **kwargs):
        ks = kwargs.get('kernel_size', (3, 3))
        # Normalise scalar int to 2-tuple — several upstream call sites pass
        # `kernel_size=3` rather than `kernel_size=(3, 3)`. Without this, the
        # downstream `tuple(ks)` would raise TypeError at graph-build time.
        # Write the normalised form back into kwargs so downstream
        # preprocessors (conv2d_pp / max_pool2d_pp) also see a 2-tuple — they
        # call `list(kernel_size)` and would otherwise still trip on a scalar.
        if isinstance(ks, int):
            ks = (ks, ks)
        ks = tuple(ks)
        kwargs['kernel_size'] = ks
        if ks != (1, 1):
            # --- 3×3+ kernel: ITS → Halo → [Move] → Conv ---
            # Capture original input BEFORE ITS/Halo so Move guard uses the right tensor.
            original_input = kwargs.get('input_tensor') or (args[0] if args else None)

            if original_input is not None:
                mc = getattr(original_input, '_memory_config', None)
                if mc is not None and not mc.is_sharded():
                    its_out = _interleaved_to_sharded(
                        original_input,
                        element_size=original_input.element_size(),
                    )
                    # Mirror tt-metal: the conv's Conv2dConfig.shard_layout drives the
                    # auto-emitted ITS's output memory layout. Falls back to
                    # HEIGHT_SHARDED when shard_layout is unset (None) — matches the
                    # default the TT-Polaris shim has historically used.
                    conv_cfg_for_its = kwargs.get('conv_config')
                    its_shard = getattr(conv_cfg_for_its, 'shard_layout', None)
                    if not isinstance(its_shard, TensorMemoryLayout):
                        its_shard = TensorMemoryLayout.HEIGHT_SHARDED
                    its_out._memory_config = MemoryConfig(its_shard, BufferType.L1)
                    if 'input_tensor' in kwargs:
                        kwargs['input_tensor'] = its_out
                    else:
                        args = (its_out,) + args[1:]

            # Normalise padding/stride scalars to 2-tuples — same shape robustness
            # as the kernel_size normalisation above. The Halo LUT key builder
            # treats these as indexable 2-D geometry, and ``halo_sinf`` reads them
            # from ``op.attrs``; either would break on a scalar int. Write the
            # normalised values back into kwargs so downstream preprocessors
            # (conv2d_pp / conv_transpose2d_pp / max_pool2d_pp) — which do
            # ``padding[0]`` / ``list(stride)`` — also see 2-tuples.
            padding = kwargs.get('padding', (0, 0))
            stride = kwargs.get('stride', (1, 1))
            if isinstance(padding, int):
                padding = (padding, padding)
            if isinstance(stride, int):
                stride = (stride, stride)
            padding = tuple(padding)
            stride = tuple(stride)
            kwargs['padding'] = padding
            kwargs['stride'] = stride

            # The tensor fed into Halo may be the post-ITS tensor (if input was
            # interleaved); capture it here so ``element_size`` in halo_attrs is
            # dtype-accurate for halo_sinf's analytical perf_stats fallback.
            halo_input = kwargs.get('input_tensor') or args[0]

            # Pass sliding-window config attrs so halo_sinf can compute the
            # halo-extended hw_shape (matching hardware's physical buffer size).
            halo_attrs = {
                'kernel_size': ks,
                'padding': padding,
                'stride': stride,
                'is_transpose': is_transpose,
                'element_size': halo_input.element_size(),
                # HW HaloDeviceOperation always emits a ROW_MAJOR output (the
                # layout-normalization point feeding the RM-input conv), regardless
                # of whether its input arrived TILE (post tile-layout conv) or RM.
                # Force it here so the auto-emitted Move (which inherits) and the
                # downstream conv both see ROW_MAJOR, matching the silicon capture.
                'layout': Layout.ROW_MAJOR_LAYOUT,
            }
            if 'input_tensor' in kwargs:
                kwargs['input_tensor'] = halo(kwargs['input_tensor'], **halo_attrs)
            elif args:
                args = (halo(args[0], **halo_attrs),) + args[1:]

            # Emit Move AFTER Halo but BEFORE the main op, mirroring hardware's
            # ITS → Halo → Move → Conv sequence (not Conv → Move as _with_move did).
            # Three guards in order of precedence:
            #   1. Workload opts out per-position via ``emit_move_before_conv=False``
            #      kwarg.  Used for stages where the HW profiler shows no Move row
            #      despite deallocate_activation being True (constant-shape buffer
            #      reuse in bottleneck, fresh-from-Concat inputs in decoder d3).
            #   2. Skip when the conv's input came from a fresh-buffer producer
            #      (MaxPool / ConvTranspose) — the kernel doesn't need to
            #      reallocate.
            #   3. The original deallocate_activation + L1-sharded guard below.
            emit_move = kwargs.get('emit_move_before_conv', True)
            if (
                move_before_conv
                and emit_move
                and _producer_optype(original_input) not in _HW_FRESH_BUFFER_PRODUCERS
            ):
                conv_cfg = kwargs.get('conv_config')
                deallocate = kwargs.get(
                    'deallocate_activation',
                    getattr(conv_cfg, 'deallocate_activation', False),
                )
                if deallocate:
                    # Check the POST-ITS+Halo input tensor's memory config (not
                    # original_input). When the original input was L1_INTERLEAVED or
                    # DRAM_INTERLEAVED, auto-ITS converts it to L1-sharded, and
                    # tt-metal's conv kernel then fires the Move kernel to reallocate
                    # the halo output. Using original_input would skip Move at
                    # decoder convs whose input comes from an explicit STS (e.g.
                    # d1.conv1, d2.conv1 in VGG UNet).
                    current_input = kwargs.get('input_tensor') or (args[0] if args else None)
                    current_mc = getattr(current_input, '_memory_config', None)
                    is_l1_sharded = (
                        current_mc is not None
                        and current_mc.is_sharded()
                        and getattr(current_mc, 'buffer_type', None) == BufferType.L1
                    )
                    if is_l1_sharded:
                        if 'input_tensor' in kwargs:
                            kwargs['input_tensor'] = _move(
                                kwargs['input_tensor'],
                                element_size=kwargs['input_tensor'].element_size(),
                            )
                        elif args:
                            args = (
                                _move(args[0], element_size=args[0].element_size()),
                            ) + args[1:]

            return op_fn(*args, **kwargs)

        else:
            # --- 1×1 kernel (lowered to MatMul): no Move ---
            # tt-metal's 1×1-conv→MatMul path does NOT emit MoveDeviceOperation
            # after the matmul (the HW profiler shows no Move row at the final
            # 1×1 conv output shape).  Drop the trailing Move that the previous
            # implementation emitted; emit only the MatMul.
            return op_fn(*args, **kwargs)
    return _impl


def _with_move(op_fn):
    """Return a wrapper that auto-emits a Move SimOp after the main op.

    Move is emitted only when two conditions both hold:
      1. deallocate_activation=True (direct kwarg or via Conv2dConfig.conv_config)
      2. The input tensor's _memory_config is L1-sharded (HEIGHT or BLOCK sharded
         on L1 buffer).

    Condition 2 mirrors hardware: MoveDeviceOperation is only dispatched when the
    activation lives in L1 sharded memory.  Tensors in DRAM or interleaved L1 are
    left in place and no Move is emitted.
    """
    def _impl(*args, **kwargs):
        # Capture input tensor BEFORE op_fn runs (conv_config/halo may transform it).
        input_tensor = kwargs.get('input_tensor') or (args[0] if args else None)

        conv_cfg = kwargs.get('conv_config')
        deallocate = kwargs.get(
            'deallocate_activation',
            getattr(conv_cfg, 'deallocate_activation', False),
        )
        result = op_fn(*args, **kwargs)
        if deallocate:
            mc = getattr(input_tensor, '_memory_config', None)
            is_l1_sharded = (
                mc is not None
                and mc.is_sharded()
                and getattr(mc, 'buffer_type', None) == BufferType.L1
            )
            if is_l1_sharded:
                result = _move(result)
        return result
    return _impl


# Convolution (ITS→Halo auto-emitted before, Move auto-emitted after when deallocate_activation=True)
_conv2d_raw = single_output_immediate_op("Conv", preprocess=conv2d_pp)
# 1×1 conv: hardware lowers to MatMul; use same conv2d_pp attrs so matmul_shape_inf
# detects kernel_shape=[1,1] and applies NCHW conv output-shape logic.
_matmul_1x1_raw = single_output_immediate_op("MatMul", preprocess=conv2d_pp)


def _matmul_1x1_with_hw_fields(*args, **kwargs):
    """1×1 conv lowered to MatMul; apply HW weight/bias preprocessing.

    On hardware, the 1×1 conv kernel pre-processes the weight to NHWC-flat
    ``[1, 1, C_in*kH*kW, C_out]`` in TILE / BFLOAT8_B / DRAM and the bias to
    ``[1, 1, 1, C_out]`` in the same layout; the activation is tilized before
    the matmul.  Setting these ``hw_shape`` / ``_hw_layout`` / ``_hw_dtype``
    fields on the input/weight/bias tensors makes the LUT key match the
    profiler's recorded matmul entry for the 1×1 conv (Conv path already does
    this via ``conv_sinf``; matmul-routed path didn't).
    """
    input_t = kwargs.get('input_tensor')
    weight_t = kwargs.get('weight_tensor')
    bias_t = kwargs.get('bias_tensor')

    result = _matmul_1x1_raw(*args, **kwargs)

    if weight_t is not None and weight_t.shape is not None and len(weight_t.shape) == 4:
        C_out = int(weight_t.shape[0])
        C_in = int(weight_t.shape[1])
        kH = int(weight_t.shape[2])
        kW = int(weight_t.shape[3])
        weight_t.hw_shape = [1, 1, C_in * kH * kW, C_out]
        weight_t._hw_dtype = DataType.BFLOAT8_B
        weight_t._hw_layout = Layout.TILE_LAYOUT
        if bias_t is not None:
            bias_t.hw_shape = [1, 1, 1, C_out]
            bias_t._hw_dtype = DataType.BFLOAT8_B
            bias_t._hw_layout = Layout.TILE_LAYOUT
    if input_t is not None:
        input_t._hw_layout = Layout.TILE_LAYOUT

    return result


def _apply_conv_output_layout(result, kwargs):
    """Mirror tt-metal: ``Conv2dConfig.output_layout`` controls the conv output tensor layout.

    Without this, TT-Polaris's Conv2d / Conv_transpose2d output defaults to ``Layout.DEFAULT``
    (= ROW_MAJOR_LAYOUT, see tensor.py), so downstream ops (next halo, etc.) always see
    ROW_MAJOR input — even when the model code requested ``output_layout=TILE_LAYOUT`` via
    ``Conv2dConfig``. Mirrors tt-metal: the kernel emits in the layout requested by the
    config (relevant for VGG UNet's tile_layout convs).
    """
    conv_cfg = kwargs.get('conv_config')
    if conv_cfg is not None:
        ol = getattr(conv_cfg, 'output_layout', None)
        if isinstance(ol, Layout):
            result.layout = ol
    return result


def _conv2d_dispatch(*args, **kwargs):
    ks = kwargs.get('kernel_size', (3, 3))
    if tuple(ks) == (1, 1):
        result = _matmul_1x1_with_hw_fields(*args, **kwargs)
    else:
        result = _conv2d_raw(*args, **kwargs)
    return _apply_conv_output_layout(result, kwargs)


conv2d = _with_halo(_conv2d_dispatch, is_transpose=False, move_before_conv=True)


_conv_transpose2d_raw_inner = single_output_immediate_op("ConvTranspose", preprocess=conv_transpose2d_pp)


def _conv_transpose2d_raw(*args, **kwargs):
    result = _conv_transpose2d_raw_inner(*args, **kwargs)
    return _apply_conv_output_layout(result, kwargs)


conv_transpose2d = _with_halo(_conv_transpose2d_raw, is_transpose=True, move_before_conv=True)

# Pooling (Halo auto-emitted before max_pool2d, matching hardware sub-op sequence)
global_avg_pool2d = single_output_immediate_op("GlobalAveragePool")
_max_pool2d_raw = single_output_immediate_op("MaxPool", preprocess=max_pool2d_pp)
max_pool2d = _with_halo(_max_pool2d_raw)

# ttnn.avg_pool2d: windowed 2D average pooling. Reuses max_pool2d_pp for kwarg normalization
# (kernel_size/stride/padding/dilation/ceil_mode -> kernel_shape/strides/pads/dilations/ceil_mode
# is identical for both pooling flavors) but targets the "AveragePool" op descriptor
# (ttsim/ops/desc/nn.py:avgpool_sinf), which has its own shape inference and perf-stats formula
# (add/div/mov instruction counts) distinct from MaxPool's (cmp/mov) -- so TT-Polaris's cost model
# actually distinguishes avg-pool from max-pool rather than the two being silently interchangeable.
_avg_pool2d_raw = single_output_immediate_op("AveragePool", preprocess=max_pool2d_pp)
avg_pool2d = _with_halo(_avg_pool2d_raw)

# Matrix Multiplication
matmul = single_output_immediate_op("MatMul")
outer = single_output_immediate_op("MatMul", preprocess=outer_pp)

# Funky Ops
grid_sample = single_output_immediate_op("GridSample")
assign = single_output_immediate_op("Assign", preprocess=as_pp)
topk = multiple_output_immediate_op("TopK", preprocess=topk_pp)
plus_one = single_output_immediate_op("PlusOne")  # in-place position-counter increment (decode bookkeeping)


def manual_seed(seeds=None, user_ids=None, sub_core_grids=None):
    """Seed the on-device RNG for decode sampling (emits a ManualSeed op)."""
    from .ttnn_shim import manual_seed_op as _ms
    return _ms(seeds, user_ids, sub_core_grids=sub_core_grids)


def sampling(topk_values, topk_indices, *, k=None, p=None, temp=None,
             output_tensor=None, sub_core_grids=None, memory_config=None):
    """Sample one token per user from gathered top-k values/indices (emits a Sampling op)."""
    from .ttnn_shim import sampling_op as _samp
    return _samp(topk_values, topk_indices, k=k, p=p, temp=temp,
                 output_tensor=output_tensor, sub_core_grids=sub_core_grids,
                 memory_config=memory_config)


Tensor.__add__ = add  # type: ignore
Tensor.__sub__ = subtract  # type: ignore
Tensor.__mul__ = multiply  # type: ignore
Tensor.__div__ = div  # type: ignore
Tensor.__pow__ = pow  # type: ignore
Tensor.__matmul__ = matmul  # type: ignore
Tensor.reshape = reshape  # type: ignore


def silu(x):
    return x * sigmoid(x)


# Multi-operator functions
def linear(*args, **kwargs):
    """Fused linear: emits a single MatMul SimOp with optional bias (3rd input)
    and optional fused activation, matching HW's MatmulDeviceOperation.

    Previous implementation decomposed linear into separate matmul → add → activation
    SimOps.  HW's MatmulDeviceOperation fuses all three into one kernel, so the
    decomposed graph produced extra ops that did not appear in profiler traces.
    Emitting a single MatMul SimOp (with bias as an optional 3rd input and activation
    as an attribute) keeps the TT-Polaris op graph 1-to-1 with HW profiler output.
    """
    assert len(args) == 2, f"linear args #-inputs({len(args)}) != 2"
    A = require_ttnn_tensor(args[0], "ttnn.linear input")
    B = require_ttnn_tensor(args[1], "ttnn.linear weight")
    bias_tensor = kwargs.get("bias", None)
    act = kwargs.get("activation", None)

    device = A.device if hasattr(A, 'device') and A.device else (
        B.device if hasattr(B, 'device') else None)

    op_name = generate_new_op_name()
    attrs = {}
    if act is not None:
        attrs["fused_activation"] = act
    # Record the matmul math_fidelity from compute_kernel_config (a ComputeKernelConfig
    # with .math_fidelity, or a bare MathFidelity enum) — mirrors layer_norm_pp. This is
    # currently key-neutral (matmul is not in _MATH_FIDELITY_CALLER_CONTROLLED_OPS, so the
    # LUT key stores 'N/A'); it makes the source-faithful per-matmul fidelity available in
    # attrs for when matmul mf is keyed in a coordinated LUT rebuild.
    ckc = kwargs.get("compute_kernel_config", None)
    if ckc is not None:
        _mf = getattr(ckc, "math_fidelity", None)
        if _mf is None and isinstance(ckc, MathFidelity):
            _mf = ckc
        if _mf is not None:
            attrs["math_fidelity"] = _mf.name
    opinfo = {'name': op_name, 'optype': 'MatMul', 'inList': [], 'attrs': attrs}
    # Output layout follows the activation input A (matmul preserves layout on HW),
    # unless an explicit layout= kwarg overrides it — mirrors single_output_immediate_op.
    C = Tensor(name=op_name + ".out", op_out=[op_name], device=device,
               layout=_resolve_output_layout(kwargs, [A, B]))

    input_tensors = []
    for x in [A, B]:
        x.op_in.append(op_name)
        opinfo["inList"].append(x.name)
        input_tensors.append(x)

    if bias_tensor is not None:
        bias_tensor = require_ttnn_tensor(bias_tensor, "ttnn.linear bias")
        bias_tensor.op_in.append(op_name)
        opinfo["inList"].append(bias_tensor.name)
        input_tensors.append(bias_tensor)

    opinfo["outList"] = [C.name]
    opobj = SimOp(opinfo)
    opobj.get_perf_counts(input_tensors, [C])
    opobj.update_tensor_counts(input_tensors, [C])

    # Handle both dtype and output_dtype parameters (dtype is an alias for output_dtype)
    output_dtype = kwargs.get("output_dtype", None)
    if output_dtype is None:
        output_dtype = kwargs.get("dtype", None)
    if output_dtype is not None and isinstance(output_dtype, DataType):
        C._ttnn_dtype = output_dtype
    else:
        # Matmul output dtype follows the ACTIVATION (input A) only — not the weight
        # (B) or bias. A compact weight dtype (e.g. the bf4 FF1/FF3 weights) does NOT
        # downcast the output on HW (tt-metal: output = activation dtype unless an
        # explicit dtype= is given; the capture's w1/w3 outputs feeding Mul are bf16
        # despite bf4 weights). Propagating from all inputs would leak the weight's
        # compact dtype into every downstream activation.
        _propagate_ttnn_dtype([A], [C])

    _propagate_memory_config(input_tensors, [C])

    mem_cfg = kwargs.get("memory_config", None)
    if mem_cfg is not None:
        C._memory_config = mem_cfg

    if device is not None:
        device.add_op(opobj)
    return C


# fold:
# takes an input tensor with shape (N, H, W, C) and transforms it to shape
# (N, H//stride_h, W//stride_w, C*stride_h*stride_w) by reshaping and permuting
# the spatial dimensions. This operation is commonly used as a preprocessing step
# for convolution operations, similar to the im2col operation in other deep learning
# frameworks, to reorganize input data in a format suitable for efficient matrix
# multiplication on Tenstorrent hardware.
def fold(
    ttnn_tensor_like,
    stride_h: int,
    stride_w: int,
    *,
    use_transpose_as_fold=False,
    output_shape=None,  # ttnn.Shape  -- accepted for ttnn API compat, unused
    pad_c: int = 0,
    pad_h: int = 0,
    pad_w: int = 0,
    grid_size=None,  # ttnn.CoreRangeSet  -- accepted for ttnn API compat, unused
    override_memory_config: MemoryConfig | None = None,  # accepted for ttnn API compat, unused
):
    """Fold: (N,H,W,C) → (N, H//stride_h, W//stride_w, C*stride_h*stride_w).

    Reorganises spatial dimensions similarly to im2col, commonly used as a
    preprocessing step for convolution on Tenstorrent hardware.

    Two execution paths:
      - ``use_transpose_as_fold=True``: decomposed into reshape/transpose ops,
        always produces logical 4D output.
      - ``use_transpose_as_fold=False`` (default): creates a first-class Fold
        SimOp whose output shape depends on the input tensor's memory
        configuration (see ``flatten_nd`` design comment below).

    Note: The parameters ``output_shape``, ``grid_size``, and
    ``override_memory_config`` are accepted for API compatibility but are
    currently ignored by the simulator.
    """
    ttnn_tensor_like = require_ttnn_tensor(ttnn_tensor_like, "ttnn.fold input")
    assert (
        ttnn_tensor_like.rank() == 4
    ), f"fold input should be a rank-4 [N, H, W, C] tensor, got rank {ttnn_tensor_like.rank()} with shape {ttnn_tensor_like.shape}"
    N, H, W, C = ttnn_tensor_like.shape

    assert (
        isinstance(stride_h, int) and stride_h > 0 and stride_h <= H
    ), f"stride_h({stride_h}) should be in (0, {H}]"
    assert (
        isinstance(stride_w, int) and stride_w > 0 and stride_w <= W
    ), f"stride_w({stride_w}) should be in (0, {W}]"

    if pad_h > 0:
        H += pad_h
    if pad_w > 0:
        W += pad_w
    if pad_c > 0:
        C += pad_c

    Hs = H // stride_h
    Ws = W // stride_w

    if use_transpose_as_fold:
        # fold implemented as a series of reshape/transpose
        reshaped1 = ttnn_tensor_like.reshape(N, Hs, stride_h, Ws, stride_w, C)
        transposed = reshaped1.permute(0, 1, 3, 2, 4, 5)
        reshaped2 = transposed.reshape(N, Hs, Ws, C * stride_h * stride_w)
    else:
        # Fold as first-class SimOp (matches hardware Fold kernel naming vs reshape shortcut).
        assert ttnn_tensor_like.device is not None, "fold requires input tensor on device"
        op_name = generate_new_op_name()
        # Design decision: flatten_nd is *computed* from the input tensor's
        # layout and memory configuration rather than being a caller-supplied
        # parameter.  This mirrors the tt-metal C++ implementation where
        # prim::fold always produces [1,1,N*Hs*Ws,Cs] but the higher-level
        # ttnn::fold conditionally reshapes back to [N,Hs,Ws,Cs] for tiled
        # or DRAM-interleaved inputs (see fold.cpp / fold_device_op.cpp).
        #
        # The choice of `(device is not None) and not (is_tiled or is_dram)` means:
        #   - Device ROW_MAJOR (any memcfg, including None) → flatten_nd=True  (ViT, typical models)
        #   - DRAM-interleaved     → flatten_nd=False  (preserve 4D)
        #   - TILE_LAYOUT          → flatten_nd=False  (preserve 4D)
        # (Host tensors are excluded by the assert above; device is always non-None here.)
        #
        # This attr is forwarded to fold_sinf in tensor.py, which is
        # frontend-agnostic; see the comment there for the default rationale.
        is_tiled = getattr(ttnn_tensor_like, 'layout', None) == Layout.TILE_LAYOUT
        mc = ttnn_tensor_like.memory_config() if hasattr(ttnn_tensor_like, 'memory_config') else None
        is_dram = (getattr(mc, 'buffer_type', None) is BufferType.DRAM) if mc is not None else False
        fold_attrs = {
            'stride_h': stride_h,
            'stride_w': stride_w,
            'pad_h': pad_h,
            'pad_w': pad_w,
            'pad_c': pad_c,
            'flatten_nd': (ttnn_tensor_like.device is not None) and not (is_tiled or is_dram),
        }
        out_tensor = Tensor(
            name=op_name + '.out',
            op_out=[op_name],
            device=ttnn_tensor_like.device,
            dtype=ttnn_tensor_like.dtype,
            layout=ttnn_tensor_like.layout,
        )
        ttnn_tensor_like.op_in.append(op_name)
        opinfo = {
            'name': op_name,
            'optype': 'Fold',
            'inList': [ttnn_tensor_like.name],
            'outList': [out_tensor.name],
            'attrs': fold_attrs,
        }
        opobj = SimOp(opinfo)
        opobj.get_perf_counts([ttnn_tensor_like], [out_tensor])
        opobj.update_tensor_counts([ttnn_tensor_like], [out_tensor])
        _propagate_ttnn_dtype([ttnn_tensor_like], [out_tensor])
        out_tensor._memory_config = MemoryConfig(
            TensorMemoryLayout.HEIGHT_SHARDED, BufferType.L1,
        )
        ttnn_tensor_like.device.add_op(opobj)
        reshaped2 = out_tensor

    return reshaped2
