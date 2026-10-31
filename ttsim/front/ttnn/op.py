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

# Gate STATE lives in ttsim/utils/shim_gates.py, not here, so that bidir_bcast
# in ttsim/ops/desc/helpers.py can read one without ttsim.ops importing
# ttsim.front -- see that module's docstring for the cycle that back-edge
# closed.  Only the accessors THIS module calls are imported; setters are
# imported from shim_gates directly by whoever sets them (the workload infras
# and the gate tests).
from ttsim.utils.shim_gates import (
    conv_input_reshard_enabled,
    conv_math_fidelity_in_key_enabled,
    halo_output_dtype_promotion_enabled,
    memory_config_propagation_enabled,
    move_on_reallocate_halo_output_enabled,
    tilize_before_matmul_enabled,
)

from typing import Any, Optional, Sequence, Union

from .buffer import BufferType, ShardOrientation, TensorMemoryLayout
from .memory import MemoryConfig
from .tensor import DataType, Layout, Tensor, generate_new_op_name, require_ttnn_tensor, zeros
from .types import TILE_HEIGHT, TILE_WIDTH


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
        opobj.get_perf_counts(new_args, [C])
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
        opobj.get_perf_counts(new_args, out_tensors)
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


# Single-slot stash for the math fidelity conv2d_pp saw, read by
# _matmul_1x1_with_hw_fields immediately afterwards.  A module-level slot rather
# than a return value because conv2d_pp's signature is fixed by
# single_output_immediate_op's preprocess contract.  Safe because graph building
# is single-threaded and the read happens in the same call.
_CONV_PP_MATH_FIDELITY: dict = {'last': None}


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

    # Rebuilding kwargs from scratch (below) is what discarded `compute_config`,
    # and with it the math fidelity.  The consequence: every 1x1 conv lowered to
    # a MatMul keyed math_fidelity='N/A' while the p100a capture recorded 'LoFi'
    # — 33 of ResNet-50's 34 matmuls, six of them differing in that field alone.
    # (`ttnn.linear` already carries it, which is why the fc matmul is the one
    # that does not have this problem.)
    #
    # Gated per-device: `matmul` is the only op_code any LUT keys under a real
    # fidelity, and the LUTs disagree with each other — bh_p100a_lut_v6 carries
    # both 'LoFi' x14 (resnet50) and 'N/A' x9 (ViT/VGG).  Emitting the true value
    # unconditionally would fix the first group and break the second.  The
    # `math_fidelity -> N/A` lookup fallback added alongside this covers the
    # second group; the gate keeps workloads that have not been re-verified on
    # their current behaviour regardless.
    # Stash the fidelity rather than writing it into the key here.  The capture's
    # convention is per-op-code, not per-call: it records `matmul` under the real
    # fidelity ('LoFi') and `conv2d` under 'N/A'.  conv2d_pp serves BOTH the Conv
    # path and the 1x1-lowered MatMul path, so attaching it unconditionally put
    # 'LoFi' on conv2d keys the LUT records as 'N/A' — adding a differing field
    # to all 20 conv misses.  `_matmul_1x1_with_hw_fields` applies it to the
    # matmul only; see _CONV_PP_MATH_FIDELITY.
    _mf_name = None
    _cfg = kwargs_dict.get("compute_config") or kwargs_dict.get("compute_kernel_config")
    if _cfg is not None:
        _mf = getattr(_cfg, "math_fidelity", None)
        if _mf is None and isinstance(_cfg, MathFidelity):
            _mf = _cfg
        if _mf is not None:
            _mf_name = _mf.name

    kwargs_dict = {
        "pads": pads,
        "kernel_shape": list(kwargs_dict["kernel_size"]),
        "strides": list(strides),
    }
    _CONV_PP_MATH_FIDELITY['last'] = _mf_name
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
    kwargs_dict_in = kwargs_dict
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
    # This preprocessor rebuilds kwargs from scratch, which silently dropped
    # ``dtype``.  In real ttnn it names the pool's OUTPUT dtype, and
    # single_output_immediate_op already applies it — ResNet-50's avg_pool2d
    # passes dtype=bfloat8_b (tt-metal
    # tt-metal/models/demos/vision/classification/resnet50/ttnn_resnet/tt/ttnn_functional_resnet50.py:904) and the
    # capture's
    # Pool2D row 111 duly
    # records out=BFLOAT8_B against a BFLOAT16 input, so dropping it left the
    # whole fc tail keyed on BFLOAT16.  Same failure mode as conv2d_pp dropping
    # compute_config.  Forwarded only when present, so pools that do not pass it
    # are unaffected.
    for _passthrough in ("dtype", "output_dtype"):
        if _passthrough in kwargs_dict_in:
            kwargs_dict[_passthrough] = kwargs_dict_in[_passthrough]
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


class transformer:
    def __init__(self, config):
        pass

    @staticmethod
    def scaled_dot_product_attention(q, k, v, *, is_causal=True, scale=None,
                                     sliding_window_size=None, compute_kernel_config=None,
                                     program_config=None, memory_config=None, **kwargs):
        """Prefill SDPA (non-paged, non-chunked); output = q shape."""
        from .ttnn_shim import scaled_dot_product_attention_op as _sdpa
        # Pass scale through unchanged: when the caller omits it (None), the shim op
        # drops it from the recorded attrs rather than fabricating a 0.0 (which is not
        # a valid SDPA scale and would pollute attrs / LUT keys vs an omitted attribute).
        return _sdpa(q, k, v, memory_config=memory_config, is_causal=bool(is_causal),
                     scale=scale)

    @staticmethod
    def scaled_dot_product_attention_decode(q, k, v, *, cur_pos_tensor=None, scale=None,
                                            sliding_window_size=None, program_config=None,
                                            compute_kernel_config=None, memory_config=None,
                                            **kwargs):
        """Decode SDPA (non-paged); output = q shape."""
        from .ttnn_shim import scaled_dot_product_attention_op as _sdpa
        return _sdpa(q, k, v, cur_pos_tensor, memory_config=memory_config,
                     scale=scale)

    @staticmethod
    def paged_scaled_dot_product_attention_decode(q, k, v, *, page_table_tensor=None,
                                                  cur_pos_tensor=None, scale=None,
                                                  sliding_window_size=None, program_config=None,
                                                  compute_kernel_config=None, memory_config=None,
                                                  **kwargs):
        """Paged decode SDPA; output = q shape."""
        from .ttnn_shim import scaled_dot_product_attention_op as _sdpa
        return _sdpa(q, k, v, cur_pos_tensor, page_table_tensor,
                     memory_config=memory_config,
                     scale=scale)


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
    # ValueError, not assert: ``python -O`` strips asserts at COMPILE time, so a
    # stripped build has no check at all.  Public through the ttnn namespace
    # (workloads/ttnn/vgg_unet/common/ttnn_vgg_unet.py:294), so this validates a
    # CALLER's input.  Under ``-O`` the rank half would fall through to the unpack
    # below, and the C<target half would silently produce a NEGATIVE pad_amount.
    if not (len(in_shape) == 4 and in_shape[1] < target_channels):
        raise ValueError(
            f"pad_channels_nchw: NCHW input with C<target required; "
            f"got shape={in_shape}, target={target_channels}")
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
    # ValueError, not assert: ``python -O`` strips asserts at COMPILE time.  Public
    # through the ttnn namespace
    # (workloads/ttnn/vgg_unet/common/ttnn_vgg_unet.py:295, the line after
    # pad_channels_nchw), so this validates a CALLER's input.
    if len(in_shape) != 4:
        raise ValueError(
            f"permute_reshape_to_nhwc_flat expects rank-4 NCHW input; got shape={in_shape}")
    N, C, H, W = in_shape
    t1 = _emit_entry_transpose(input_tensor, [1, H, C, W])
    t2 = _emit_entry_transpose(t1, [1, 1, N * H * W, C])
    return t2


def _emit_dm_op_4d(input_tensor: 'Tensor', optype: str, out_shape: list,
                   attrs: dict | None = None,
                   hw_shape: list | None = None) -> 'Tensor':
    """Emit an arity-1 data-movement SimOp with an explicit rank-4 output shape.

    Used to build op sequences whose logical shapes must track a tt-metal C++
    decomposition exactly (see ``fold_transpose_sharded_nchw``).  Arity-1 keeps
    the LUT key a 9-tuple, matching how the profiler records Pad / Transpose /
    Slice; ``perf_stats`` is populated here because these ops' ONNX-style shape
    inference expects extra tensor inputs (a pads/starts tensor) that hardware
    does not dispatch.

    ``hw_shape`` is left ``None`` by default on purpose: ``_shape_wzyx`` in
    ``tools/perf_lookup/lookup_operator_perf.py`` prefers ``hw_shape`` over the
    logical shape when building a key, and the profiler records these ops under
    their logical rank-4 shape.
    """
    in_shape = [int(d) for d in require_shape_list(
        input_tensor.shape, 'data-movement op input shape must be set')]
    out_shape = [int(d) for d in out_shape]
    op_name = generate_new_op_name()
    out_tensor = Tensor(
        name=op_name + '.out',
        shape=out_shape,
        dtype=input_tensor.dtype,
        layout=input_tensor.get_layout(),
        op_out=[op_name],
        device=input_tensor.device,
    )
    if hw_shape is not None:
        out_tensor.hw_shape = list(hw_shape)
    input_tensor.op_in.append(op_name)

    opinfo = {
        'name': op_name,
        'optype': optype,
        'inList': [input_tensor.name],
        'outList': [out_tensor.name],
        'attrs': dict(attrs or {}),
    }
    opobj = SimOp(opinfo)

    elem_size = input_tensor.element_size()
    n_in = 1
    for d in in_shape:
        n_in *= d
    n_out = 1
    for d in out_shape:
        n_out *= d
    opobj.perf_stats = {
        'inElems': n_in,
        'outElems': n_out,
        'inBytes': n_in * elem_size,
        'outBytes': n_out * elem_size,
        'instrs': {'mov': n_out},
    }
    # Pre-populating perf_stats short-circuits SimOp.get_perf_counts(), which
    # would otherwise snapshot shapes after inference; mirror that snapshot.
    opobj._frozen_input_shapes = [list(in_shape)]
    opobj._frozen_output_shapes = [list(out_shape)]
    opobj.update_tensor_counts([input_tensor], [out_tensor])

    _propagate_ttnn_dtype([input_tensor], [out_tensor])
    _propagate_memory_config([input_tensor], [out_tensor])

    input_tensor.device.add_op(opobj)  # type: ignore[union-attr]
    return out_tensor

def relabel_shape(tensor: 'Tensor', shape: list) -> 'Tensor':
    """Rewrite a tensor's *logical* shape in place, emitting no SimOp.

    Models ``ttnn::experimental::view`` — a free regrouping of elements that
    hardware performs with no profiler row. Safe because the producing op's
    ``perf_stats`` and frozen shapes are snapshotted at emission and its LUT key
    is built from its *input*, so neither moves; only valid for a tensor
    consumed exactly once, which is the case at every call site here.

    Returns the same tensor object, for call-site readability.
    """
    shape = [int(d) for d in shape]
    tensor.shape = Shape(shape)
    return tensor

def fold_transpose_sharded_nchw(input_tensor: 'Tensor', stride_h: int, stride_w: int,
                                *, pad_h: int, pad_w: int, pad_c: int) -> 'Tensor':
    """Height-sharded transpose-fold, mirroring tt-metal row-for-row.

    Reproduces ``fold_with_transpose_sharded_`` in
    ``ttnn/cpp/ttnn/operations/data_movement/fold/fold.cpp``, which for a
    height-sharded input dispatches::

        Pad -> Transpose(2,3) -> Pad -> Transpose(1,2) -> view
             -> Transpose(2,3) -> view -> Transpose(1,2) -> Slice

    The two ``ttnn::experimental::view`` calls are free, so hardware records
    **seven** rows.  Verified against the resnet50 refruns: WH n150 b16 and
    BH p100a b16/b32 all open with
    Pad, Transpose, Pad, Transpose, Transpose, Transpose, Slice and the exact
    intermediate shapes this function produces.

    Distinct from ``fold`` above, which assumes an NHWC input with end-only
    padding and cannot express this model's NCHW + symmetric-padding fold.
    ``fold`` is left untouched so ViT's ``Fold`` SimOp is unaffected.

    The returned tensor deliberately carries a **logical NCHW** shape
    ``[N, C*sh*sw, H_out, W_out]`` with an NHWC-flat ``hw_shape``
    ``[1, 1, N*H_out*W_out, C*sh*sw]``.  Hardware's fold output is NHWC, but
    polaris's ``conv_sinf`` requires NCHW; the trailing Slice is the
    convention-handoff point.  Both LUT keys stay correct because a key is
    built from an op's *input* (the Slice's input is NHWC, matching hardware)
    and the downstream Halo reads ``hw_shape``.
    """
    in_shape = [int(d) for d in require_shape_list(
        input_tensor.shape, 'fold input shape must be set')]
    # ValueError, not assert: ``python -O`` strips asserts at COMPILE time, and a
    # stripped build would reach the rank-4 unpack below with no diagnostic.
    if len(in_shape) != 4:
        raise ValueError(f'fold expects rank-4 NCHW input, got {in_shape}')
    N, C, H, W = in_shape

    padded_c = C + pad_c
    padded_h = H + 2 * pad_h
    padded_w = W + 2 * pad_w
    padded_h32 = ((padded_h + TILE_HEIGHT - 1) // TILE_HEIGHT) * TILE_HEIGHT
    padded_w32 = ((padded_w + TILE_HEIGHT - 1) // TILE_HEIGHT) * TILE_HEIGHT
    out_h = padded_h // stride_h
    out_w = padded_w // stride_w
    out_c = padded_c * stride_h * stride_w

    # 1. pad input tensor: {n, padded_c, padded_h32, w}
    x = _emit_dm_op_4d(input_tensor, 'Pad', [N, padded_c, padded_h32, W],
                       attrs={'pads': [0, 0, pad_h, 0, 0, pad_c, 0, 0],
                              'mode': 'constant', 'value': 0})
    # 2. transpose(2, 3)
    x = _emit_dm_op_4d(x, 'Transpose', [N, padded_c, W, padded_h32],
                       attrs={'perm': [0, 1, 3, 2]})
    # 3. pad tensor W dim: {n, padded_c, padded_h32, padded_w32}
    x = _emit_dm_op_4d(x, 'Pad', [N, padded_c, padded_w32, padded_h32],
                       attrs={'pads': [0, 0, pad_w, 0, 0, 0, 0, 0],
                              'mode': 'constant', 'value': 0})
    # 4. transpose(1, 2)
    x = _emit_dm_op_4d(x, 'Transpose', [N, padded_w32, padded_c, padded_h32],
                       attrs={'perm': [0, 2, 1, 3]})
    # 5. view -> {n, w/stride_w, c*stride_w, h}  (ttnn::experimental::view; free,
    #    no profiler row, but it IS what the next Transpose sees — the capture's
    #    row-4 input is [n, padded_w32/sw, padded_c*sw, padded_h32], so relabel
    #    rather than feeding the next op the pre-view shape).
    x = relabel_shape(x, [N, padded_w32 // stride_w, padded_c * stride_w, padded_h32])
    # 6. transpose(2, 3)
    x = _emit_dm_op_4d(x, 'Transpose',
                       [N, padded_w32 // stride_w, padded_h32, padded_c * stride_w],
                       attrs={'perm': [0, 1, 3, 2]})
    # 7. view -> {n, w, h/stride_h, c*stride_h}  (free, as above)
    x = relabel_shape(x, [N, padded_w32 // stride_w, padded_h32 // stride_h, out_c])
    # 8. transpose(1, 2)
    x = _emit_dm_op_4d(x, 'Transpose',
                       [N, padded_h32 // stride_h, padded_w32 // stride_w, out_c],
                       attrs={'perm': [0, 2, 1, 3]})
    # 9. slice to the logical fold output {n, target_h, target_w, target_c},
    #    handing back an NCHW logical shape (see docstring).
    x = _emit_dm_op_4d(x, 'Slice', [N, out_c, out_h, out_w],
                       attrs={'output_shape': [N, out_h, out_w, out_c]},
                       hw_shape=[1, 1, N * out_h * out_w, out_c])
    return x


def _with_halo(op_fn, is_transpose: bool = False, move_before_conv: bool = False):
    """Return a wrapper that auto-emits a Halo SimOp before the main op.

    Halo is skipped only when the conv is lowered to a matmul — kernel 1×1 AND
    stride 1 AND padding 0 AND dilation 1 AND not width-sharded, mirroring
    tt-metal's use_matmul_for_1x1_conv. A 1×1 conv at stride 2 stays a real
    convolution on hardware and does get a halo.

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

        # Halo is skipped only when the conv is actually LOWERED TO A MATMUL,
        # which is a narrower condition than "the kernel is 1x1".  tt-metal's
        # use_matmul_for_1x1_conv (tt-metal/ttnn/cpp/ttnn/operations/conv/conv2d/conv2d_utils.cpp:519-531) requires
        #   kernel 1x1 AND stride 1 AND padding 0 AND dilation 1 AND not width-sharded
        # A 1x1 conv at stride 2 therefore stays a real convolution on hardware
        # and DOES get a halo.  This gate previously tested the kernel alone, so
        # ResNet-50's three 1x1 stride-2 downsample convs were emitted with no
        # Halo — Halo 19 against the capture's 22 — while bh_p100a_lut_v6 carries
        # halo entries keyed exactly kernel=1x1 stride=2x2 padding=0x0, i.e.
        # silicon does dispatch them.
        # Normalise scalars into kwargs BEFORE the branch, not inside it.  Call
        # sites pass `stride=1` as readily as `stride=(1, 1)`, and downstream
        # preprocessors do `list(stride)` / `padding[0]`.  While this lived
        # inside the non-1x1 branch, a 1x1 conv with a scalar stride reached
        # conv2d_pp un-normalised and died with "'int' object is not iterable".
        _st = _norm_pair(kwargs.get('stride'), (1, 1))
        _pd = _norm_pair(kwargs.get('padding'), (0, 0))
        _dl = _norm_pair(kwargs.get('dilation'), (1, 1))
        kwargs['stride'], kwargs['padding'], kwargs['dilation'] = _st, _pd, _dl
        _cfg = kwargs.get('conv_config')
        _lowered_to_matmul = _lowers_to_matmul(ks, _st, _pd, _dl, _cfg)

        # Input-side Reshard (+Move), ahead of BOTH branches — hardware runs it
        # before the halo for a real conv (capture 59->60->61->62) and before the
        # tilize for a matmul-lowered one (11->12->13, 17->18->19->20).  Gated;
        # see set_conv_input_reshard.
        args, kwargs = _maybe_emit_conv_input_reshard(args, kwargs, is_transpose)

        if not _lowered_to_matmul:
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

            # ``stride`` / ``padding`` are already 2-tuples in kwargs: ``_norm_pair``
            # normalises them ahead of the matmul-lowering branch (see the block
            # above ``_lowered_to_matmul``), precisely so both branches and the
            # downstream preprocessors see tuples. Re-normalising here was dead —
            # ``isinstance(padding, int)`` cannot hold at this point and the
            # write-back re-assigned an identical value. Read them instead; the
            # locals feed ``halo_attrs`` below.
            padding = kwargs['padding']
            stride = kwargs['stride']

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
            # HW HaloDeviceOperation also picks its OUTPUT dtype from a switch
            # on the input dtype
            # (tt-metal/ttnn/cpp/ttnn/operations/sliding_window/halo/device/halo_device_operation.cpp:66-70): FLOAT32
            # and
            # UINT16 pass through, everything else — BFLOAT8_B included — comes
            # out BFLOAT16.  Without this the shim inherits BFLOAT8_B and every
            # downstream conv2d keys on the wrong dtype (capture: all 20 conv2d
            # rows report INPUT_0_DATATYPE=BFLOAT16).  Gated; see
            # set_halo_output_dtype_promotion.
            if halo_output_dtype_promotion_enabled(getattr(halo_input, 'device', None)):
                halo_attrs['dtype'] = _halo_output_dtype(
                    getattr(halo_input, '_ttnn_dtype', None)
                )
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
            #   3. The reallocate_halo_output + L1-sharded guard below.
            #
            # On (3): tt-metal gates this Move on ``reallocate_halo_output``, NOT on
            # ``deallocate_activation``
            # (``ttnn/cpp/ttnn/operations/conv/conv2d/conv2d.cpp:297-299``)::
            #
            #     input_tensor_post_tm = std::move(halo_output);
            #     if (conv_config.reallocate_halo_output) {
            #         input_tensor_post_tm = ttnn::move(input_tensor_post_tm);
            #     }
            #
            # `deallocate_activation` drives the deallocate two lines above
            # (tt-metal/ttnn/cpp/ttnn/operations/conv/conv2d/conv2d.cpp:291-293), which is a different thing and emits
            # no op.
            # Keying the Move off it made polaris emit a Move at every conv with
            # deallocate_activation=True — 17 of them for ResNet-50 where the
            # refrun shows 2 — an over-emission that is a shim artefact rather
            # than a workload divergence.
            emit_move = kwargs.get('emit_move_before_conv', True)

            # ttnn::move is a runtime allocator decision: tt-metal calls it, but
            # tt-metal/ttnn/cpp/ttnn/operations/data_movement/move/move.cpp:69-76 dispatches nothing when the realloc
            # lands at the
            # same address.  Polaris has no allocator and would emit a Move at
            # every call site, so the positions the capture shows no Move at are
            # named in _HALO_MOVE_SUPPRESSED_BY_SKU.  See that table for why a
            # flag rule cannot replace it and why every entry is provisional.
            if emit_move:
                # Key off ``halo_input`` -- the PRE-halo tensor captured above,
                # before ``halo(...)`` rebound kwargs['input_tensor'] to the halo
                # OUTPUT.  The table stores the halo INPUT's (nhw, channels); see
                # its "Key: ... AS SEEN AT EMISSION TIME" block.
                _hw = getattr(halo_input, 'hw_shape', None)
                if _hw is not None and len(_hw) >= 4:
                    from ttsim.ops.desc.ttsim_layout import _HALO_MOVE_SUPPRESSED_BY_SKU
                    # Keyed by the declared SKU, never by arch.  These are
                    # allocator outcomes measured on one card; an undeclared SKU
                    # (``capture_sku`` None) must miss rather than inherit
                    # another card's result -- config/tt_bh.yaml carries p150a
                    # and p150b beside p100a, and nobody has captured those.
                    _sku = getattr(getattr(halo_input, 'device', None), 'capture_sku', None)
                    _empty: frozenset[tuple[int, int]] = frozenset()
                    _suppressed = _HALO_MOVE_SUPPRESSED_BY_SKU.get(_sku or '', _empty)
                    if (int(_hw[2]), int(_hw[3])) in _suppressed:
                        emit_move = False
            if (
                move_before_conv
                and emit_move
                and _producer_optype(original_input) not in _HW_FRESH_BUFFER_PRODUCERS
            ):
                conv_cfg = kwargs.get('conv_config')
                # Default True when unspecified, matching BOTH tt-metal
                # (`tt-metal/ttnn/cpp/ttnn/operations/conv/conv2d/conv2d_nanobind.cpp:267`,
                # `nb::arg("reallocate_halo_output") = true`)
                # and the shim's own Conv2dConfig
                # (`ttsim/front/ttnn/config.py:18`).  A conv that
                # passes no conv_config therefore still emits the Move, as hardware
                # would; only an explicit False suppresses it — which is what
                # ResNet-50 sets on its downsample and conv2 (tt-metal
                # tt-metal/models/demos/vision/classification/resnet50/ttnn_resnet/tt/ttnn_functional_resnet50.py:126
                # and :232).
                # Which flag drives the Move is itself gated, because the two
                # have OPPOSITE defaults -- reallocate_halo_output True,
                # deallocate_activation False -- so the choice also decides what
                # an unset conv_config means.  See
                # set_move_on_reallocate_halo_output.
                _mv_dev = getattr(original_input, 'device', None)
                if move_on_reallocate_halo_output_enabled(_mv_dev):
                    should_move = kwargs.get(
                        'reallocate_halo_output',
                        getattr(conv_cfg, 'reallocate_halo_output', True),
                    )
                else:
                    should_move = kwargs.get(
                        'deallocate_activation',
                        getattr(conv_cfg, 'deallocate_activation', False),
                    )
                if should_move:
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

    # Hardware matmuls consume TILE.  When the activation arrives ROW_MAJOR,
    # silicon dispatches a Tilize first — the ResNet-50 p100a capture records
    # exactly that at rows 12 and 19, converting the maxpool's ROW_MAJOR output
    # ahead of the first two matmuls.  Emit it so the graph carries the op
    # instead of only the docstring above claiming it happens.
    #
    # Opt-in and device-scoped (see set_tilize_before_matmul): it ADDS ops, and
    # every workload graph is built in one process.
    _act = kwargs.get('input_tensor') or (args[0] if args else None)
    if _act is not None and tilize_before_matmul_enabled(getattr(_act, 'device', None)):
        if _act.get_layout() == Layout.ROW_MAJOR_LAYOUT and getattr(_act, 'device', None) is not None:
            from .ttnn_shim import tilize_op as _tilize_op
            _src_mc = getattr(_act, '_memory_config', None)
            _act = _tilize_op(_act, element_size=_act.element_size())
            # Tilize is a layout change, not a re-shard: the tensor keeps its
            # sharding.  tilize_op does not carry `_memory_config` across, which
            # left the matmul downstream with None and broke the chain that
            # decides whether a Reshard is needed.
            if _src_mc is not None and getattr(_act, '_memory_config', None) is None:
                _act._memory_config = _src_mc
            if 'input_tensor' in kwargs:
                kwargs['input_tensor'] = _act
            else:
                args = (_act,) + tuple(args[1:])

    result = _matmul_1x1_raw(*args, **kwargs)

    # The capture keys `matmul` under the real fidelity and `conv2d` under
    # 'N/A', so the value conv2d_pp stashed is applied HERE — on the
    # matmul-lowered path only.  Gated per-device (set_conv_math_fidelity_in_key).
    # Read and consume: conv2d_pp stashes the fidelity, this is the only
    # reader, and leaving it set would leak into the next conv.
    _mf_name, _CONV_PP_MATH_FIDELITY['last'] = _CONV_PP_MATH_FIDELITY.get('last'), None
    # Two conditions, each doing separate work:
    #
    #   _dev is not None  -- Tensor.device is genuinely optional here
    #       (``_emit_dm_op_4d`` carries a ``# type: ignore[union-attr]`` on
    #       ``input_tensor.device.add_op(...)`` for the same reason), and there
    #       is no op registry to annotate without one.  Load-bearing, not
    #       defensive: the accessor resolves ON for a device-less result
    #       whenever TTSIM_CONV_MATH_FIDELITY_IN_KEY=1, so omitting it makes
    #       ``_dev.ops`` below an AttributeError at graph build.
    #
    #   the accessor, not a raw getattr -- the two still disagree for a device
    #       carrying no per-device override, where the accessor falls through to
    #       the module-level flag the env var controls and a raw getattr would
    #       yield the literal False default.
    _dev = getattr(result, 'device', None)
    if _mf_name is not None and _dev is not None and conv_math_fidelity_in_key_enabled(_dev):
        for _op_name in (getattr(result, 'op_out', None) or []):
            _op = _dev.ops.get(_op_name)
            if _op is not None and isinstance(getattr(_op, 'attrs', None), dict):
                _op.attrs['math_fidelity'] = _mf_name

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
    # The activation the MATMUL consumes — not the one this function was called
    # with.  When set_tilize_before_matmul is on, a Tilize was inserted above and
    # kwargs['input_tensor'] was rebound to its output; tagging the ORIGINAL
    # tensor here would relabel the Tilize's own input as TILE, and a tilize
    # whose input_0 is already tilized is not a row the capture has (its two
    # Tilize rows, 12 and 19, both record INPUT_0_LAYOUT=ROW_MAJOR).  Re-read it.
    input_t = kwargs.get('input_tensor', input_t)
    if input_t is not None and input_t.get_layout() != Layout.TILE_LAYOUT:
        input_t._hw_layout = Layout.TILE_LAYOUT

    return result






def _halo_output_dtype(input_dtype: Optional[DataType]) -> DataType:
    """FLOAT32/UINT16 pass through, else BFLOAT16.

    ``None`` is a legitimate input, not a guard: the sole caller passes
    ``getattr(halo_input, '_ttnn_dtype', None)``, and a tensor carrying no
    ttnn dtype falls through to BFLOAT16 like any other non-FLOAT32/UINT16
    value.  Annotating it ``DataType`` would reject that call.

    Canonical: ``tt-metal/ttnn/cpp/ttnn/operations/sliding_window/halo/device/halo_device_operation.cpp:66-70``.
    """
    if input_dtype in (DataType.FLOAT32, DataType.UINT16):
        return input_dtype
    return DataType.BFLOAT16




def _norm_pair(v: Union[int, Sequence[int], None],
               default: Sequence[int]) -> tuple[int, ...]:
    """Normalise a scalar / sequence geometry kwarg to a tuple, leaving length alone.

    Several upstream call sites pass ``stride=1`` or ``padding=1`` rather than a
    2-tuple, and ``avg_pool2d`` passes a 4-element padding. Downstream consumers
    index these (``conv2d_pp`` does ``padding[0]``, the Halo key builder treats
    them as 2-D geometry), so they must be normalised *before* any branch reads
    them — see ``_with_halo``.
    """
    if v is None:
        v = default
    if isinstance(v, int):
        return (v, v)
    return tuple(v)


def _padding_n4(padding: Union[int, Sequence[int], None]) -> tuple[int, int, int, int]:
    """Normalise a conv padding to tt-metal's ``padding_n4`` = (top, bottom, left, right).

    tt-metal spells the padded extents as::

        input_padded_height = input_height + padding_n4[0] + padding_n4[1]
        input_padded_width  = input_width  + padding_n4[2] + padding_n4[3]

    (``tt-metal/ttnn/cpp/ttnn/operations/conv/conv2d/conv2d_utils.cpp:1405-1406`` and ``:1514-1515``), so the
    four-element
    form is TWO-SIDED PER AXIS -- (top, bottom, left, right) -- and emphatically
    not (h, w, h, w).  Getting that backwards is the whole reason this helper
    exists rather than the padding being indexed inline.

    Accepted spellings, all of which reach conv kwargs in-tree:

        1              -> (1, 1, 1, 1)
        (ph, pw)       -> (ph, ph, pw, pw)
        (t, b, l, r)   -> unchanged

    Raises ``ValueError`` for any other length; :func:`_conv_out_hw` catches it
    and declines rather than letting a graph build die here.
    """
    if padding is None:
        padding = (0, 0)
    if isinstance(padding, int):
        return (padding, padding, padding, padding)
    p = tuple(int(v) for v in padding)
    if len(p) == 1:
        return (p[0], p[0], p[0], p[0])
    if len(p) == 2:
        return (p[0], p[0], p[1], p[1])
    if len(p) == 4:
        return (p[0], p[1], p[2], p[3])
    raise ValueError(
        f"conv padding must have 1, 2 or 4 elements, got {len(p)}: {p}")


def _conv_out_hw(kwargs: dict, is_transpose: bool = False) -> tuple[int, int] | None:
    """Conv output (height, width) from *kwargs*, or ``None`` when not derivable.

    Mirror of tt-metal's ``calculate_output_image_size``
    (``tt-metal/ttnn/cpp/ttnn/operations/conv/conv2d/conv2d_utils.cpp:277-284``)::

        out_h = ((in_h - k_h - (k_h-1)*(dil_h-1) + (pad[0]+pad[1])) / st_h) + 1
        out_w = ((in_w - k_w - (k_w-1)*(dil_w-1) + (pad[2]+pad[3])) / st_w) + 1

    ``- k - (k-1)*(dil-1)`` collapses to ``- dil*(k-1) - 1``, which is the form
    written below; the two are algebraically identical, and the second is what
    the two former inline copies used.  Those copies spelled the padding term
    ``2*pd[0]`` / ``2*pd[-1]``, which agrees with cpp for the two-element
    spelling ``(ph, pw)`` and is WRONG for an asymmetric four-element one:
    ``2*top`` instead of ``top+bottom``, and ``2*right`` instead of
    ``left+right``.  Extracting the helper is what makes that fixable in one
    place -- see ``_padding_n4`` for the layout.

    **Effective kernel extent, not the bare kernel**: a dilated conv covers
    ``dilation * (k - 1) + 1`` input positions per output position.  Identical
    to the undilated form at dilation 1, which is all ResNet-50 uses, but the
    grid and shard spec the callers derive from this are wrong for anything
    dilated.

    **``is_transpose`` declines rather than computing.**  A transposed conv has
    a different formula entirely: ``calculate_ct2d_output_image_size``
    (``cpp:288-299``) MULTIPLIES by stride and SUBTRACTS padding where this one
    divides and adds, so handing back the forward result would be silently
    wrong by roughly a factor of stride squared.  Callers early-return on
    ``None``, which is the conservative outcome -- no shard config rather than a
    wrong one -- and that is strictly better than what the inline copies did,
    which was to compute the forward value for a transpose without noticing.

    Nothing reaches here with ``is_transpose=True`` today: the only caller that
    can see a transposed conv is ``_maybe_emit_conv_input_reshard``, which sits
    behind ``conv_input_reshard_enabled``; only the ResNet-50 infras enable that
    gate and ResNet-50 has no transposed convs, while vgg_unet (the one
    ``conv_transpose2d`` user in-tree) runs with the gate off.  The guard is
    therefore for the future caller, not a live path -- but it is the path that
    opens the moment anyone enables the gate for vgg_unet.
    """
    if is_transpose:
        return None
    in_h = int(kwargs.get('input_height', 0) or 0)
    in_w = int(kwargs.get('input_width', 0) or 0)
    if not (in_h and in_w):
        return None
    ks = _norm_pair(kwargs.get('kernel_size'), (1, 1))
    st = _norm_pair(kwargs.get('stride'), (1, 1))
    dil = _norm_pair(kwargs.get('dilation'), (1, 1))
    try:
        pad = _padding_n4(kwargs.get('padding'))
    except (ValueError, TypeError):
        return None
    if int(st[0]) <= 0 or int(st[-1]) <= 0:
        return None
    out_h = (in_h + pad[0] + pad[1] - int(dil[0]) * (int(ks[0]) - 1) - 1) // int(st[0]) + 1
    out_w = (in_w + pad[2] + pad[3] - int(dil[-1]) * (int(ks[-1]) - 1) - 1) // int(st[-1]) + 1
    if out_h <= 0 or out_w <= 0:
        return None
    return out_h, out_w


def _lowers_to_matmul(
    kernel_size: Union[int, Sequence[int], None],
    stride: Union[int, Sequence[int], None],
    padding: Union[int, Sequence[int], None],
    dilation: Union[int, Sequence[int], None],
    conv_config: Optional[Any] = None,
) -> bool:
    """Mirror of tt-metal's ``use_matmul_for_1x1_conv``.

    Source: ``ttnn/cpp/ttnn/operations/conv/conv2d/conv2d_utils.cpp:519``::

        kernel_size == {1,1} && stride[0] == stride[1] && stride[0] == 1
          && padding[0..3] == 0 && dilation == {1,1} && !is_width_sharded

    When this holds tt-metal runs the conv as a plain matmul and dispatches no
    HaloDeviceOperation; otherwise it dispatches Halo + Conv2d. Every term
    matters:

    * **stride** — resnet50's three downsample convs are 1x1 with stride 2 and
      are Halo + Conv2d on silicon (WH n150 rows 33/34, 56/57, 89/90).
    * **padding / dilation / width-sharding** — no current workload exercises a
      padded, dilated or width-sharded 1x1 conv, but testing only the kernel
      made the predicate broader than tt-metal's and would silently drop the
      Halo for any of those.
    """
    ks = _norm_pair(kernel_size, (3, 3))
    st = _norm_pair(stride, (1, 1))
    pad = _norm_pair(padding, (0, 0))
    dil = _norm_pair(dilation, (1, 1))
    is_width_sharded = (
        getattr(conv_config, 'shard_layout', None) == TensorMemoryLayout.WIDTH_SHARDED
    )
    return (
        tuple(ks) == (1, 1)
        and len(st) >= 2 and st[0] == st[1] and st[0] == 1
        and all(int(p) == 0 for p in pad)
        and all(int(d) == 1 for d in dil)
        and not is_width_sharded
    )


def _maybe_emit_conv_input_reshard(
    args: tuple, kwargs: dict, is_transpose: bool = False,
) -> tuple[tuple, dict]:
    """Emit Reshard (+Move) ahead of a conv whose config sets ``reshard_if_not_optimal``.

    Returns possibly-rebound ``(args, kwargs)``.  A no-op unless the gate is on,
    the flag is set, and the target input memory config differs from the current
    one.  See ``set_conv_input_reshard`` for the tt-metal references.

    ``is_transpose`` is threaded in from ``_with_halo`` because this is the one
    conv path a transposed conv can reach (``conv_transpose2d`` is also wrapped
    by ``_with_halo``), and the output-geometry formula differs for it -- see
    ``_conv_out_hw``, which declines rather than returning the forward value.
    """
    conv_cfg = kwargs.get('conv_config')
    if conv_cfg is None or not bool(getattr(conv_cfg, 'reshard_if_not_optimal', False)):
        return args, kwargs
    inp = kwargs.get('input_tensor') or (args[0] if args else None)
    if inp is None:
        # Guard stated locally.  Without it, `inp` reaching inp.get_layout() /
        # inp.element_size() below is prevented only indirectly — None input means
        # None device, which makes the grid tuple (0, 0) and trips `not all(grid)`.
        # That held, but it made two attribute accesses depend on a two-hop
        # inference three guards away.
        return args, kwargs
    device = getattr(inp, 'device', None)
    if device is None or not conv_input_reshard_enabled(device):
        return args, kwargs
    shard_layout = getattr(conv_cfg, 'shard_layout', None)
    if not isinstance(shard_layout, TensorMemoryLayout):
        return args, kwargs
    grid = (getattr(device, 'grid_x', 0), getattr(device, 'grid_y', 0))
    if not all(grid):
        return args, kwargs

    from .conv_sharding import (
        create_sharded_memory_config_from_parallel_config,
        determine_parallel_config,
        get_input_channels_alignment,
    )

    is_mm_conv = _lowers_to_matmul(
        kwargs.get('kernel_size'), kwargs.get('stride'),
        kwargs.get('padding'), kwargs.get('dilation'), conv_cfg)
    batch = int(kwargs.get('batch_size', 1) or 1)
    in_c = int(kwargs.get('in_channels', 0) or 0)
    out_c = int(kwargs.get('out_channels', 0) or 0)
    in_h = int(kwargs.get('input_height', 0) or 0)
    in_w = int(kwargs.get('input_width', 0) or 0)
    if not (in_c and out_c and in_h and in_w):
        return args, kwargs
    # cpp:277-284, via the shared helper.  Declines (None) for a transposed
    # conv, whose formula is different -- see _conv_out_hw.
    _hw = _conv_out_hw(kwargs, is_transpose=is_transpose)
    if _hw is None:
        return args, kwargs
    out_h, out_w = _hw

    in_mc = getattr(inp, '_memory_config', None)
    alignment = get_input_channels_alignment(
        shard_layout, inp.get_layout(), False, is_mm_conv, in_mc)
    try:
        # cpp:749 / cpp:1033 derive this from the conv config rather than
        # assuming ROW_MAJOR:
        #     conv_config.transpose_shards ? COL_MAJOR : ROW_MAJOR
        _block_orient = (ShardOrientation.COL_MAJOR
                         if getattr(conv_cfg, 'transpose_shards', False)
                         else ShardOrientation.ROW_MAJOR)
        pcfg = determine_parallel_config(
            shard_layout, batch, in_c, out_h, out_w, out_c, alignment, grid,
            _block_orient,
            enable_channels_padding=not is_mm_conv,
            act_block_h_override=int(getattr(conv_cfg, 'act_block_h_override', 0) or 0),
        )
        target = create_sharded_memory_config_from_parallel_config(
            batch * in_h * in_w, in_c, pcfg, tile_size=TILE_HEIGHT,
            input_channels_alignment=alignment)
    except Exception as e:  # pragma: no cover - never fail the graph build over this
        logger.debug("conv input reshard: parallel-config build failed ({}); skipping", e)
        return args, kwargs

    if in_mc is not None and in_mc == target:
        return args, kwargs  # tt-metal's to_memory_config no-ops here

    from .ttnn_shim import reshard_op as _reshard_op
    new_t = _reshard_op(inp, memory_config=target, element_size=inp.element_size())

    # cpp:884-887 — the Move rides along only when the activation is being
    # deallocated and did not live in DRAM.
    if (bool(getattr(conv_cfg, 'deallocate_activation', False))
            and in_mc is not None
            and getattr(in_mc, 'buffer_type', None) == BufferType.L1):
        new_t = _move(new_t)
        new_t._memory_config = target

    if 'input_tensor' in kwargs:
        kwargs['input_tensor'] = new_t
    else:
        args = (new_t,) + tuple(args[1:])
    return args, kwargs




def _rebuild_shard_shape_for_output(in_mc: Any, kwargs: dict,
                                    result: Tensor) -> Optional[Any]:
    """Carry ``in_mc``'s grid/scheme/orientation but recompute the shard shape.

    Returns None when the output dims are not derivable, so the caller can fall
    back to the config as-is.  See the call site for why verbatim inheritance is
    wrong.
    """
    spec = getattr(in_mc, 'shard_spec', None)
    if spec is None or getattr(spec, 'grid', None) is None:
        return None
    from .conv_sharding import (
        create_sharded_memory_config_from_parallel_config,
        ParallelConfig,
    )
    batch = int(kwargs.get('batch_size', 1) or 1)
    out_c = int(kwargs.get('out_channels', 0) or 0)
    in_h = int(kwargs.get('input_height', 0) or 0)
    in_w = int(kwargs.get('input_width', 0) or 0)
    if not (out_c and in_h and in_w):
        return None
    # cpp:277-284, via the shared helper.  ``is_transpose`` is left at its
    # default False on purpose: this function is reached only from
    # ``_apply_conv_output_memory_config``, which only the conv2d dispatch
    # calls -- ``_conv_transpose2d_raw`` applies the output LAYOUT but not the
    # output memory config, so a transposed conv never arrives here.
    _hw = _conv_out_hw(kwargs)
    if _hw is None:
        return None
    out_h, out_w = _hw
    pcfg = ParallelConfig(spec.grid, in_mc.memory_layout, spec.orientation)
    try:
        # cpp:846-848 runs determine_output_parallel_config on the (inherited)
        # input config before building the output's memory config; it rewrites
        # only .grid, to suit the output channel count.
        from .conv_sharding import determine_output_parallel_config
        ks_mm = _lowers_to_matmul(
            kwargs.get('kernel_size'), kwargs.get('stride'),
            kwargs.get('padding'), kwargs.get('dilation'),
            kwargs.get('conv_config'))
        _grid = (int(getattr(result.device, 'grid_x', 0) or 0),
                 int(getattr(result.device, 'grid_y', 0) or 0))
        if all(_grid):
            # cpp:833 passes parallel_config.shard_orientation — the input
            # config's own orientation — so the grid axes are laid out the
            # way the output's inherited orientation says they are.
            pcfg = determine_output_parallel_config(
                pcfg, _grid, out_c, pcfg.shard_orientation, is_mm_conv=ks_mm)
        # An OUTPUT config pads channels by TILE_WIDTH, not by the input-channel
        # alignment.  cpp keeps the two granularities separate:
        #
        #   input  (conv2d_utils.cpp:641-642)
        #       round_up(in_channels,  num_cores_c        * input_channels_alignment)
        #   output (conv2d_utils.cpp:1286-1287)
        #       round_up(out_channels, num_cores_channels * TILE_WIDTH)
        #
        # Both then divide plainly inside
        # create_sharded_memory_config_from_parallel_config (cpp:357), whose
        # ``channels % num_cores_channels`` assertion holds precisely because the
        # caller pre-padded.  This port folds the padding into that function, so
        # the granularity arrives as a parameter and each caller must supply the
        # one its direction uses.
        #
        # Passing TILE_WIDTH is exactly equivalent to cpp's formula, not merely
        # close: ceil(ceil(c/32)/n)*32 == ceil(c/(32n))*32 for positive integers.
        #
        # ResNet-50 could not see this.  Its conv inputs are TILE layout and its
        # 1x1s are mm convs, and both routes return TILE_WIDTH from
        # get_input_channels_alignment, so alignment == 32 == TILE_WIDTH there.
        # A ROW_MAJOR, non-mm, non-width-sharded input can yield 16 or 8, and
        # then the two disagree: out_c=512 on 7 channel cores is 96 per cpp but
        # ceil(32/7)*16 = 80 at alignment 16.  MemoryConfig.__eq__ compares the
        # shard shape, so that gap is the exact condition
        # ``ds_out.memory_config() != out.memory_config()`` tests.
        return create_sharded_memory_config_from_parallel_config(
            batch * out_h * out_w, out_c, pcfg, tile_size=TILE_HEIGHT,
            input_channels_alignment=TILE_WIDTH)
    except Exception:  # pragma: no cover - fall back to verbatim inheritance
        return None


def _apply_conv_output_memory_config(result: Tensor, kwargs: dict) -> Tensor:
    """Give a conv/matmul output the memory config its Conv2dConfig implies.

    No-op unless propagation is enabled.  Only sets a config when the conv
    declares a ``shard_layout``; an unset shard_layout leaves the output as it
    was, rather than inventing one.
    """
    # Resolve the device the way every other gate in this file does -- off the
    # tensor, with the kwarg only as a fallback.  ``device=`` is optional on
    # ttnn.conv2d: ResNet-50 supplies it (inside ``conv_kwargs``, forwarded as
    # ``**conv_kwargs`` -- ttnn_functional_resnet50.py:586), but vgg_unet omits
    # it at both of its call sites (workloads/ttnn/vgg_unet/ttnn_vgg_unet.py:74 and
    # workloads/ttnn/vgg_unet/common/ttnn_vgg_unet.py:136).  Keying on the kwarg alone therefore
    # resolved the gate per call site rather than per graph: on for the convs
    # that pass it, off for the ones that do not, within one workload.
    #
    # ``result`` is in the chain because this helper receives only kwargs, not
    # args: a positional ``ttnn.conv2d(x, w, b, ...)`` puts no input_tensor in
    # kwargs, and the conv output carries the same device either way.
    #
    # Verified neutral for ResNet-50 (which already resolved a real device):
    # 115/115 p100a and 108/108 n150, zero op records differing.  Note the gate
    # itself is NOT redundant there -- dropping the opt-in costs p100a three ops
    # and a hit (118 ops, 114/118).
    device = (
        getattr(kwargs.get('input_tensor'), 'device', None)
        or getattr(result, 'device', None)
        or kwargs.get('device')
    )
    enabled = memory_config_propagation_enabled(device)
    if not enabled:
        return result
    if not hasattr(result, '_memory_config'):
        return result

    conv_cfg = kwargs.get('conv_config')
    shard_layout = getattr(conv_cfg, 'shard_layout', None) if conv_cfg is not None else None

    # Precedence follows tt-metal (tt-metal/ttnn/cpp/ttnn/operations/conv/conv2d/conv2d_utils.cpp:676-750), which is the
    # OPPOSITE of what this function used to do.  There:
    #
    #     ParallelConfig parallel_config = input_tensor_parallel_config;   // inherit
    #     if (conv_config.reshard_if_not_optimal || needs_shard_or_reshard) {
    #         ... = determine_parallel_config(shard_layout, ...);          // request
    #     }
    #
    # i.e. the conv INHERITS the input tensor's existing sharding, and
    # `Conv2dConfig.shard_layout` is only consulted when a reshard actually
    # fires.  Note determine_parallel_config does not *choose* a layout — it
    # takes shard_layout as an argument and returns it as `.shard_scheme`,
    # computing only the core grid.
    #
    # Applying the request unconditionally, as this did, is what made ResNet-50
    # report BLOCK_SHARDED on ~40 ops the capture records as HEIGHT_SHARDED:
    # canonical passes `height_sharding` only to each layer's *_module1
    # (tt-metal/models/demos/vision/classification/resnet50/ttnn_resnet/tt/ttnn_functional_resnet50.py:691, 726,
    # 775, 859),
    # so modules 2 and 3 fall to the `height_sharding=None`
    # default and request BLOCK — but they also leave `reshard_if_not_optimal`
    # False, so hardware never looks at that request and keeps the incoming
    # HEIGHT sharding.
    #
    # `needs_shard_or_reshard` (cpp:686-745) is evaluated by conv_sharding,
    # which models the per-core shard spec the clauses there read.
    from .conv_sharding import (
        create_sharded_memory_config_from_parallel_config,
        determine_output_parallel_config,
        determine_parallel_config,
        get_input_channels_alignment,
        needs_shard_or_reshard,
    )

    inp = kwargs.get('input_tensor')
    in_mc = getattr(inp, '_memory_config', None) if inp is not None else None
    reshard_if_not_optimal = bool(getattr(conv_cfg, 'reshard_if_not_optimal', False))
    is_mm_conv = _lowers_to_matmul(
        kwargs.get('kernel_size'), kwargs.get('stride'),
        kwargs.get('padding'), kwargs.get('dilation'), conv_cfg)

    _in_layout = inp.get_layout() if inp is not None else None
    must_reshard = reshard_if_not_optimal or needs_shard_or_reshard(
        in_mc, int(kwargs.get('in_channels', 0) or 0), is_mm_conv,
        input_tensor_layout=_in_layout)

    if not must_reshard and in_mc is not None:
        # Inherit the incoming split, exactly as cpp:744 does — but that line
        # inherits the *ParallelConfig* (grid, scheme, orientation), not the
        # shard SHAPE.  The output tensor's config is still built by
        # create_sharded_memory_config_from_parallel_config from the OUTPUT
        # dims, so the per-core shape tracks the output while the grid is
        # carried over.
        #
        # Copying in_mc verbatim propagated a stale shape down whole stages:
        # every conv in layer 2 inherits, so layer3_module1's conv1 saw
        # HEIGHT_SHARDED (448, 256) — layer 1's 56x56x256 shard — against a real
        # input of 12544x512, and MaxPool reported (1696, 64) where the capture
        # has (448, 64).  Anything comparing configs (the bottleneck's
        # ``ds_out.memory_config() != out.memory_config()``, needs_shard_or_reshard,
        # the conv input reshard) was then reading fiction.
        _inherited = _rebuild_shard_shape_for_output(in_mc, kwargs, result)
        result._memory_config = _inherited if _inherited is not None else in_mc
        return result

    if not isinstance(shard_layout, TensorMemoryLayout):
        return result

    # Reshard: derive the grid for the requested layout, adjust it for the
    # output channel count, and build a config that carries the per-core shard
    # shape — without which two genuinely different configs compare equal and
    # the bottleneck's `ds_out.memory_config() != out.memory_config()` never
    # fires.
    grid = getattr(device, 'grid_x', 0), getattr(device, 'grid_y', 0)
    if not all(grid):
        result._memory_config = MemoryConfig(shard_layout, BufferType.L1)
        return result

    batch = int(kwargs.get('batch_size', 1) or 1)
    in_c = int(kwargs.get('in_channels', 0) or 0)
    out_c = int(kwargs.get('out_channels', 0) or 0)
    # Output spatial extent, per cpp:277-284.  It sizes the output shard spec
    # built below -- create_sharded_memory_config_from_parallel_config is handed
    # ``batch * out_h * out_w`` as the NHWC-flat row count -- so getting it wrong
    # moves the shard shape and therefore the LUT key.
    #
    # ``_conv_out_hw`` returns None for every case where that extent is not
    # derivable: unknown geometry (zero input_height/width), a non-positive
    # result, a stride of zero, a padding that is not 1, 2 or 4 elements, or a
    # transposed conv, whose formula differs (cpp:288-299).  ``is_transpose`` is
    # left at its default here because this function is reached only from the
    # conv2d dispatch; conv_transpose2d applies the output LAYOUT but not the
    # output memory config, so it never arrives.
    #
    # On None -- or with no out_channels to shard across -- fall back to a
    # layout-only MemoryConfig carrying no shard spec.  That is the honest
    # answer: consumers that read a spec (needs_shard_or_reshard,
    # get_num_cores_channels, MemoryConfig.__eq__) then take their "unknown"
    # branch rather than trusting a shape derived from geometry we do not have.
    _hw = _conv_out_hw(kwargs)
    if _hw is None or not out_c:
        result._memory_config = MemoryConfig(shard_layout, BufferType.L1)
        return result
    out_h, out_w = _hw

    alignment = get_input_channels_alignment(
        shard_layout, _in_layout, False, is_mm_conv, in_mc)
    # cpp:749 / cpp:1033 derive this from the conv config rather than
    # assuming ROW_MAJOR:
    #     conv_config.transpose_shards ? COL_MAJOR : ROW_MAJOR
    _block_orient = (ShardOrientation.COL_MAJOR
                     if getattr(conv_cfg, 'transpose_shards', False)
                     else ShardOrientation.ROW_MAJOR)
    # cpp:1057 is the call this mirrors, and it passes BOTH ``!is_mm_conv`` and
    # ``conv_config.act_block_h_override`` -- as do the other two call sites
    # (cpp:616, cpp:752).  The override is not cosmetic: determine_parallel_config
    # turns it into ``act_block_h_ntiles`` and feeds that as the ``mult`` of
    # find_closest_largest_divisor_with_num_padding_and_mult, so leaving it at 0
    # searches with mult=1 and returns a different num_cores_nhw.  That count
    # reaches ocfg below (cpp:258 reads it off the input config) and therefore
    # the output shard shape.  ResNet-50 sets the override on conv1 (1568 on WH,
    # 1024 or 1568 on BH batch 32) and on the layer1/layer2 convs (512, 32, 448),
    # so this is a live path, not a defensive argument.
    pcfg = determine_parallel_config(
        shard_layout, batch, in_c, out_h, out_w, out_c, alignment, grid, _block_orient,
        enable_channels_padding=not is_mm_conv,
        act_block_h_override=int(getattr(conv_cfg, 'act_block_h_override', 0) or 0),
    )
    # cpp:1077 passes the same orientation used to build pcfg (cpp:1066).
    ocfg = determine_output_parallel_config(
        pcfg, grid, out_c, pcfg.shard_orientation, is_mm_conv=is_mm_conv)
    # TILE_WIDTH, not ``alignment``: this builds the OUTPUT config, and cpp pads
    # output channels by ``num_cores_channels * TILE_WIDTH`` (cpp:1286-1287).
    # ``alignment`` above is still correct where it is used -- determine_parallel_config
    # takes input_channels_alignment and divides the INPUT channel count with it
    # (cpp:641-642) -- so the two are deliberately different here.  See
    # _rebuild_shard_shape_for_output for the full note.
    result._memory_config = create_sharded_memory_config_from_parallel_config(
        batch * out_h * out_w, out_c, ocfg, tile_size=TILE_HEIGHT,
        input_channels_alignment=TILE_WIDTH)
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
    # MUST use the same predicate as _with_halo, which wraps this function and
    # has already decided Halo-vs-no-Halo from it.  Testing the kernel alone
    # here made the two disagree for a 1x1 STRIDE-2 conv: _with_halo emitted a
    # Halo (correct — tt-metal dispatches Halo + Conv2d) while this lowered to a
    # matmul, producing a Halo + MatMul pair that cannot occur on hardware.
    # ResNet-50's three downsample convs are exactly that shape.
    if _lowers_to_matmul(kwargs.get('kernel_size'), kwargs.get('stride'),
                         kwargs.get('padding'), kwargs.get('dilation'),
                         kwargs.get('conv_config')):
        result = _matmul_1x1_with_hw_fields(*args, **kwargs)
    else:
        result = _conv2d_raw(*args, **kwargs)
    result = _apply_conv_output_layout(result, kwargs)
    return _apply_conv_output_memory_config(result, kwargs)


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
def _tensor_reshape(self, *shape: Any) -> Tensor:
    """``Tensor.reshape`` accepting a sequence OR varargs dims, as real ttnn does.

    Bound directly to the ``reshape`` op, this only accepted
    ``t.reshape([a, b, c])``; ``t.reshape(a, b, c)`` arrived as N positional args
    and tripped ``reshape_pp``'s ``len(args_list) <= 3`` guard with the
    misleading "ttnn.reshape has 3 inputs".  That is what broke ``ttnn.fold``'s
    ``use_transpose_as_fold=True`` path, which does
    ``t.reshape(N, Hs, stride_h, Ws, stride_w, C)`` — six dims.
    """
    if len(shape) == 1 and isinstance(shape[0], (list, tuple, Shape)):
        new_shape = shape[0]
    else:
        new_shape = list(shape)
    return reshape(self, new_shape)


def _tensor_permute(self, *perm: Any) -> Tensor:
    """``Tensor.permute`` on the ttnn path, sequence or varargs.

    Without this, ``Tensor`` inherited ``SimTensor.permute`` from the functional
    front-end (``ttsim/front/functional/tensor_op.py:212``), which needs a
    ``link_module`` the ttnn path never sets — so a ttnn tensor's ``.permute``
    asserted instead of emitting a Transpose SimOp.
    """
    if len(perm) == 1 and isinstance(perm[0], (list, tuple)):
        order = list(perm[0])
    else:
        order = list(perm)
    return permute(self, order)


Tensor.reshape = _tensor_reshape  # type: ignore[attr-defined]
Tensor.permute = _tensor_permute  # type: ignore[attr-defined]


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
        # or DRAM-interleaved inputs (see tt-metal/ttnn/cpp/ttnn/operations/data_movement/fold/fold.cpp /
        # tt-metal/ttnn/cpp/ttnn/operations/data_movement/fold/device/fold_device_op.cpp).
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
