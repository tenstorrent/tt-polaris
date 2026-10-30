#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from ttsim.ops.tensor import SimTensor
from ttsim.utils.shim_gates import bcast_hw_shape_propagation_enabled

import math
from loguru import logger

LOG = logger
INFO = LOG.info
DEBUG = LOG.debug

# Import compute functions for data propagation
from ttsim.ops.desc.data_compute import (
    compute_mish,
    compute_sigmoid,
    compute_gelu,
    compute_relu,
    compute_relu6,
    compute_identity,
    compute_tanh,
    compute_exp,
    compute_log,
    compute_sqrt,
    compute_softmax,
    compute_clip,
    compute_add,
    compute_mul,
    compute_sub,
    compute_div,
    compute_pow,
    compute_sin,
    compute_cos,
    compute_atan,
    compute_abs,
    compute_neg,
    compute_less,
    compute_argmax,
    compute_reducemax,
    compute_sign,
    compute_atan2,
    compute_mod,
    compute_equal,
    compute_not,
    try_compute_data,
)

def update_output_tensor(op, in_tensor, out_tensor):
    assert in_tensor.check_shape(), f"ERROR: {op} Invalid Input SHAPE in {in_tensor}"

    # Allow updating if out_tensor has default empty shape (uninitialized) or invalid shape
    if out_tensor.check_shape() and out_tensor.shape != [] and out_tensor.shape == in_tensor.shape:
        # Output tensor has a valid, non-default shape and it matches
        DEBUG("Validated SimTensor({}) SHAPE: {}", out_tensor.name, out_tensor.shape)
    elif out_tensor.check_shape() and out_tensor.shape != []:
        # out_tensor already has a "declared" shape, but it disagrees with the
        # freshly (re-)computed in_tensor shape. This happens when the source
        # ONNX graph had an unresolved/unknown dim (e.g. symbolic dim_param, or
        # a fully unknown-rank tensor) that TT-Polaris's frontend coerced to a
        # placeholder shape (commonly [1]) at load time. That placeholder is
        # not a real constraint -- it's a stand-in for "unknown". Once
        # TT-Polaris's own data/shape propagation computes the real shape here,
        # it should take precedence (override the placeholder) instead of
        # hard-failing.
        LOG.warning(
            "Overriding stale/placeholder SHAPE for SimTensor({}): declared={} -> computed={}",
            out_tensor.name, out_tensor.shape, in_tensor.shape,
        )
        out_tensor.shape = in_tensor.shape
    else:
        # Update the shape (either invalid or default empty)
        DEBUG("Updating SimTensor({}) SHAPE: {} <- {}", out_tensor.name, out_tensor.shape, in_tensor.shape)
        out_tensor.shape = in_tensor.shape

    if in_tensor.data is not None:
        if out_tensor.data is None:
            out_tensor.data = in_tensor.data
            out_tensor.dtype = in_tensor.dtype
            DEBUG("Updating DATA SimTensor({})", out_tensor)


def build_tmp_data_tensor(data, name):
    return SimTensor(
        {
            "name": name,
            "shape": list(data.shape),
            "dtype": data.dtype,
            "data": data,
            "resolve": "_",
            "op_in": [],
            "op_out": [],
        }
    )


def bidirectional_broadcast_shape_inference(shape1, shape2):
    """
    more robust than multidirectional_broadcast_shape_inference
    can handle numpy like broadcasting with [1,0] and [1]
    """
    max_len = max(len(shape1), len(shape2))
    padded1 = list(shape1[::-1]) + [1] * (max_len - len(shape1))
    padded2 = list(shape2[::-1]) + [1] * (max_len - len(shape2))
    result = []
    for d1, d2 in zip(padded1, padded2):
        if d1 == d2:
            result.append(d1)
        elif d1 == 1:
            result.append(d2)
        elif d2 == 1:
            result.append(d1)
        else:
            raise ValueError(f"Shapes {shape1} and {shape2} not broadcast-compatible")
    return result[::-1]


def multidirectional_broadcast_shape_inference(shapes):
    """
    cannot handle numpy like broadcasting with [1,0] and [1]
    """
    max_len = max(len(shape) for shape in shapes)
    num_shapes = len(shapes)

    padded_shapes = []
    for shape in shapes:
        shape_rev = list(shape[::-1])
        shape_rev.extend([1] * (max_len - len(shape)))
        padded_shapes.append(shape_rev)

    result = []
    for i in range(max_len):
        dim_list = [padded_shapes[j][i] for j in range(num_shapes)]
        max_dim = max(dim_list)
        check = all(d == 1 or d == max_dim for d in dim_list)
        assert (
            check
        ), f"Incompatible shapes at dim: {max_len-1-i}: expect {max_dim} or 1, got {dim_list}"
        result.append(max_dim)

    return result[::-1]


def propagate_shape_and_type(inT, outT, i_idx, o_idx):
    assert len(inT) > i_idx
    assert len(outT) > o_idx
    assert inT[i_idx].check_shape(), f"Illegal Shape for input tensor @ index {i_idx}"
    outT[o_idx].shape = inT[i_idx].shape
    outT[o_idx].dtype = inT[i_idx].dtype
    return


def gelu_instr_counts(nElem):
    """
    ONNX opset-20 defines GELU w/ 2 variants controlled by the attribute 'approximate'
      if approximate = 'tanh', we use GELU (Gaussian Error Linear Unit) approximation:
         Y = 0.5 * X * (1 + tanh(math.sqrt(2 / math.pi) * (X + 0.044715 * pow(X, 3))))
      else (default)
         Y = 0.5 * X * (1 + erf(X/sqrt(2)))
    we assume approximate to be 'tanh' always for now....
    TODO: add default option as well...

    instr count calc.
     Y= <const> * X * ( <const> + tanh( <const> * ( X + <const> * X^3 ) ) )
    """
    mul_count, add_count, tanh_count = 0, 0, 0
    mul_count += 2 * nElem  # X^3
    mul_count += nElem  # <const> * X^3
    add_count += nElem  # X + <const> * X^3
    mul_count += nElem  # <const> * ( X + ...)
    tanh_count += nElem  # tanh (...)
    add_count += nElem  # <const + tanh(...)
    mul_count += 2 * nElem  # <const> * X * (...)
    return {"mul": mul_count, "add": add_count, "tanh": tanh_count}


def pooling_shape_inference(input_shape, kernel_shape, attrs):
    """Shape inference for pooling operators"""

    # Validate inputs
    if len(input_shape) < 2:
        raise ValueError(f"Expected at least 2D input tensor, got shape {input_shape}")

    num_spatial_dims = len(kernel_shape)
    if num_spatial_dims > len(input_shape) - 2:
        raise ValueError(
            f"Too many spatial dimensions ({num_spatial_dims}) for input shape {input_shape}"
        )

    auto_pad = attrs.get("auto_pad", "NOTSET")
    ceil_mode = attrs.get("ceil_mode", 0)
    dilations = attrs.get("dilations", [1] * num_spatial_dims)
    pads = attrs.get("pads", [0] * (2 * num_spatial_dims))
    storage_order = attrs.get("storage_order", 0)
    strides = attrs.get("strides", [1] * num_spatial_dims)

    # Extract spatial dimensions (assume last num_spatial_dims are spatial)
    non_spatial_dims = input_shape[:-num_spatial_dims]
    spatial_dims = input_shape[-num_spatial_dims:]
    if len(spatial_dims) != num_spatial_dims:
        raise ValueError(
            f"Expected {num_spatial_dims} spatial dimensions, got {spatial_dims}"
        )

    # Handle padding
    if pads is not None:
        if len(pads) != 2 * num_spatial_dims:
            raise ValueError(f"Expected pads length 2 * {num_spatial_dims}, got {pads}")
        pad_before = pads[:num_spatial_dims]
        pad_after = pads[num_spatial_dims:]
    else:
        if auto_pad == "VALID":
            pad_before = [0] * num_spatial_dims
            pad_after = [0] * num_spatial_dims
        elif auto_pad in ["SAME_UPPER", "SAME_LOWER"]:
            pad_before = []
            pad_after = []
            for i in range(num_spatial_dims):
                # Effective kernel size with dilation
                effective_kernel_size = (kernel_shape[i] - 1) * dilations[i] + 1
                # For SAME padding, output size is ceil(input_size / stride)
                out_size = math.ceil(spatial_dims[i] / strides[i])
                # Compute total padding needed
                pad_total = max(
                    (out_size - 1) * strides[i]
                    + effective_kernel_size
                    - spatial_dims[i],
                    0,
                )
                # Distribute padding
                if auto_pad == "SAME_UPPER":
                    pad_b = pad_total // 2
                    pad_a = pad_total - pad_b
                else:  # SAME_LOWER
                    pad_a = pad_total // 2
                    pad_b = pad_total - pad_a
                pad_before.append(pad_b)
                pad_after.append(pad_a)
        else:  # NOTSET with pads=None
            pad_before = [0] * num_spatial_dims
            pad_after = [0] * num_spatial_dims

    # Compute output spatial dimensions
    output_spatial_dims = []
    for i in range(num_spatial_dims):
        # Compute effective kernel size with dilation
        effective_kernel_size = (kernel_shape[i] - 1) * dilations[i] + 1
        # Compute output size
        padded_size = spatial_dims[i] + pad_before[i] + pad_after[i]
        if ceil_mode == 0:
            out_size = (
                math.floor((padded_size - effective_kernel_size) / strides[i]) + 1
            )
        else:  # ceil_mode == 1
            out_size = math.ceil((padded_size - effective_kernel_size) / strides[i]) + 1
        if out_size <= 0:
            raise ValueError(
                f"Invalid output dimension {i}: size={out_size}. "
                f"Check input shape {input_shape}, kernel {kernel_shape}, "
                f"strides {strides}, pads {pads}, auto_pad {auto_pad}, "
                f"ceil_mode {ceil_mode}, dilations {dilations}."
            )
        output_spatial_dims.append(out_size)

    # Construct output shape
    output_shape = non_spatial_dims + output_spatial_dims
    return output_shape


# shape inference functions
def unary_fwd(iTList, oTList, op, **kwargs):
    X, Y = iTList[0], oTList[0]
    assert X.check_shape(), f"Input tensor shape not defined: {X}"
    Y.shape = X.shape
    Y.dtype = X.dtype

    # Compute actual data if inputs have data
    _unary_compute_funcs = {
        'Mish': compute_mish,
        'Sigmoid': compute_sigmoid,
        'Gelu': compute_gelu,
        'Relu': compute_relu,
        'Relu6': compute_relu6,
        'Identity': compute_identity,
        'Tanh': compute_tanh,
        'Exp': compute_exp,
        'Log': compute_log,
        'Sqrt': compute_sqrt,
        'Softmax': compute_softmax,
        'Clip': compute_clip,
        'Sin': compute_sin,
        'Cos': compute_cos,
        'Atan': compute_atan,
        'Abs': compute_abs,
        'Neg': compute_neg,
        'ArgMax': compute_argmax,
        'Sign': compute_sign,
        'Not': compute_not,
    }
    if op.optype in _unary_compute_funcs:
        Y.data = try_compute_data(_unary_compute_funcs[op.optype], iTList, op)

    optype2instr = {
        "Identity": {"mov": 0},
        "Clip": {"cmp": 2 * Y.nelems(), "mov": Y.nelems()},
        "CumSum": {
            "add": Y.nelems()
        },  # TODO: handle axis: we need to divide by X.shape[axis]
        "PRelu": {"cmp": Y.nelems(), "mul": Y.nelems()},  # slope * x if x < 0 else = x
        "BitwiseNot": {"not": Y.nelems()},
        "Not": {"not": Y.nelems()},
        # everything below here is math unary operators
        "Softmax": {
            # TODO: take care of the axis
            "cmp": X.nelems(),  # max_x = max(x)
            "sub": X.nelems(),  # y = x - max_x
            "exp": X.nelems(),  # exp(y)
            "add": X.nelems(),  # z = sum(exp(y))
            "div": X.nelems(),  # o = exp(y) / z
        },
        "LogSoftmax": {
            # TODO: take care of the axis
            # LogSoftmax(input, axis) = Log(Softmax(input, axis=axis))
            # Softmax...
            "cmp": X.nelems(),  # max_x = max(x)
            "sub": X.nelems(),  # y = x - max_x
            "exp": X.nelems(),  # exp(y)
            "add": X.nelems(),  # z = sum(exp(y))
            "div": X.nelems(),  # o = exp(y) / z
            "log": X.nelems(),  # log(o)
        },
        "Gelu": gelu_instr_counts(X.nelems()),
        "Softplus": {
            # Softplus(x) = log(1 + exp(x))
            "exp": X.nelems(),  # exp(x)
            "add": X.nelems(),  # 1 + exp(x)
            "log": X.nelems(),  # log(1 + exp(x))
            "mov": X.nelems(),  # for output
        },
        "Sigmoid": {
            # 1 / (1 + exp(-x))
            "exp": X.nelems(),
            "add": X.nelems(),
            "div": X.nelems(),
        },
        "Celu": {
            # max(0,x) + min(0,alpha*(exp(x/alpha)-1))
            #      y = max(0,x)
            #      z = x * 1/alpha
            #      z = exp(z)
            #      z = alpha * z
            #      z = z - 1
            #      z = min(0,z)
            #      w = y + z
            "cmp": 2 * X.nelems(),
            "mul": 2 * X.nelems(),
            "exp": X.nelems(),
            "sub": X.nelems(),
            "add": X.nelems(),
        },
        "Selu": {
            # y = gamma * (alpha * e^x - alpha) if x <= 0 else gamma * x
            # assume worst case: x <= 0
            "exp": X.nelems(),
            "mul": 2 * X.nelems(),
            "sub": X.nelems(),
        },
        "Elu": {
            # f(x) = alpha * (exp(x) - 1.) if x < 0 else  x
            # we assume the worst case cost (i.e. x < 0)
            "cmp": X.nelems(),  # x < 0
            "exp": X.nelems(),  # y = exp(x)
            "sub": X.nelems(),  # y -= 1
            "mul": X.nelems(),  # y *= alpha
        },
        "Mish": {
            # x * tanh(softplus(x)) = x * tanh(ln(1 + e^{x}))
            "exp": X.nelems(),  # y = exp(x)
            "add": X.nelems(),  # y = y + 1
            "log": X.nelems(),  # y = ln(y)
            "tanh": X.nelems(),  # y = tanh(y)
            "mul": X.nelems(),  # y = x * y
        },
        "HardSigmoid": {
            # max(0, min(1, alpha * x + beta))
            "mul": X.nelems(),  # x = alpha * x
            "add": X.nelems(),  # x = x + beta
            "cmp": 2 * X.nelems(),  # x = min(1, x); x = max(0, x)
        },
        "HardSwish": {
            # x * HardSigmoid<alpha, beta>(x)
            "mul": 2 * X.nelems(),  # x = alpha * x; x * HardSigmoid
            "add": X.nelems(),  # x = x + beta
            "cmp": 2 * X.nelems(),  # x = min(1, x); x = max(0, x)
        },
        "Softsign": {
            # x/(1+|x|)
            "abs": X.nelems(),
            "add": X.nelems(),
            "div": X.nelems(),
        },
    }

    optype2instr['Relu6'] = {'cmp': 2 * X.nelems(), 'mov': X.nelems()}

    for xopname in ['Sign', 'Relu', 'LeakyRelu', 'ThresholdedRelu', 'Hardmax']:
        #Sign            : x <=> 0
        #Relu            : max(0,x)
        #LeakyRelu       : alpha * x if x < 0 else x; additional 'add' accounted outside this loop
        #ThresholdedRelu : x for x > alpha else 0
        #Hardmax         : Hardmax(x, axis) = 1 if x == first_max_val_along_xis else 0; TODO: account for axis
        optype2instr[xopname] = {'cmp': X.nelems(), 'mov': X.nelems()}
    optype2instr['LeakyRelu']['add'] = X.nelems()

    # Relu6: min(max(0, x), 6) — two comparisons and a move
    optype2instr["Relu6"] = {"cmp": 2 * X.nelems(), "mov": X.nelems()}

    # Neg uses subtraction: -x = 0 - x
    optype2instr['Neg'] = {'sub': X.nelems()}

    for xopname in ['Reciprocal', 'Floor', 'Ceil', 'Sqrt', 'Exp', 'Log',
                    'Abs', 'Sin', 'Cos', 'Tan', 'Asin', 'Acos', 'Atan',
                    'Sinh', 'Cosh', 'Tanh', 'Asinh', 'Acosh', 'Atanh', 'Erf',
                    'Round', ]:
        optype2instr[xopname] = {xopname.lower(): X.nelems()}

    op.perf_stats = {
        "inElems": X.nelems(),
        "outElems": Y.nelems(),
        "inBytes": X.nbytes(op.precision),
        "outBytes": Y.nbytes(op.precision),
        "instrs": optype2instr[op.optype],
    }
    return


def _propagate_bcast_hw_shape(X0: Any, X1: Any, Y: Any) -> None:
    """Carry the NHWC-flattened ``hw_shape`` across an elementwise binary.

    Every other data-movement sinf propagates ``hw_shape`` — Move, Reshard,
    InterleavedToSharded, ShardedToInterleaved all do (``ttsim/ops/desc/ttsim_layout.py``).
    ``bidir_bcast`` did not, so ResNet-50's residual ``Add`` dropped the
    flattened form and every op downstream of it keyed on logical NCHW
    ``(16, 1024, 14, 14)`` where the LUT records ``(1, 1, 3136, 1024)`` — one
    missing line cascading into roughly 27 mis-keyed rows.

    Takes ``X0``'s view unconditionally, guarded only on the logical shape being
    unchanged, which is the same rule the parallel WH implementation uses for
    every shape-preserving op.  An earlier version also handled two further
    cases — the operands carrying *different* views (decline), and one carrying
    it while the other is a true broadcast (take the one that has it).  Neither
    has ever fired: instrumenting every workload with the gate on shows the only
    case that occurs is both operands carrying the SAME view, 16 times, from
    ResNet-50's residual adds.  Speculative branches that no capture exercises
    are worse than no branches, so they are gone; the shape guard below is what
    actually keeps a reshaping broadcast from inheriting a stale view.

    Gated per-device (``Device.propagate_bcast_hw_shape``) rather than applied
    unconditionally: it changes the LUT key of every elementwise binary in every
    workload, and polaris builds all workload graphs in one process.  See
    ``ttsim/utils/shim_gates.py::set_bcast_hw_shape_propagation``.

    NOTE on where this lives.  ``bidir_bcast`` is registered under ``ai.onnx``
    and is reached by all THREE front ends -- onnx, functional and ttnn -- so a
    ttnn-only behaviour sits here only because ttnn's ``add`` reuses the shared
    ``Add`` optype (``ttsim/front/ttnn/op.py:1012``).  The device check below is what
    keeps the other two out; see FUTURE-RISK(onnx-device) there for when that
    stops being true.

    The structurally clean fix is a ttnn-domain binary optype in
    ``ttsim/ops/desc/ttsim_layout.py`` under ``com.tenstorrent.ttnn``, which the onnx and
    functional loaders have no registration for and so cannot reach.  It was
    prototyped and backed out as too large for this change: the optype is
    load-bearing in four places at once.  ``tools/profiling/op_canonical.py`` normalises it in
    three tables and the new name has to normalise IDENTICALLY (``Sub`` maps to
    ``sub``, not ``subtract`` -- easy to get wrong); the LUT keys ResNet-50's 16
    residual adds under ``add``, including the ``add_output_as_input2``
    fallback; ``ttsim/front/ttnn/op.py:176`` branches on ``optype in ["Add", "Sub", "Mul"]`` for
    scalar operands; and ``_binary_compute_funcs`` below keys constant folding
    off it.  The registry is also keyed by opname ALONE (``ttsim/ops/desc/registry.py:13``)
    with layout ops registering LAST (``ttsim/ops/desc/__init__.py:32``), so the ttnn
    binary cannot reuse the name ``Add`` -- it would overwrite the ONNX entry
    rather than sit beside it.
    """
    dev = getattr(X0, 'device', None) or getattr(X1, 'device', None)
    if dev is None:
        # ONNX/functional exclusion.  ``bidir_bcast`` is registered under
        # ``ai.onnx`` and serves all THREE front ends -- onnx, functional
        # (ttsim/front/functional/op.py:981) and ttnn (ttsim/front/ttnn/op.py:1012 binds
        # ``add`` to the shared ``Add`` optype) -- so a ttnn-only behaviour has
        # to exclude the other two itself.
        #
        # "Has a device" is the discriminator because today only the ttnn shim
        # attaches one: ``SimTensor`` has no ``device`` attribute, so an ONNX or
        # functional graph resolves ``dev`` to None and returns here.  Returning
        # BEFORE the gate is deliberate -- the gate alone is not an exclusion.
        # ONNX conv/maxpool/matmul sinfs do set ``hw_shape``
        # (ttsim/ops/desc/nn.py:276, 427, 587; ttsim/ops/desc/math.py:601), so
        # there is something real to propagate, and
        # ``TTSIM_PROPAGATE_BCAST_HW_SHAPE=1`` or a device-less
        # ``set_bcast_hw_shape_propagation(True)`` would otherwise turn it on
        # for every pure-ONNX model.
        #
        # The env var still reaches ttnn graphs: it seeds the module default,
        # which a device-carrying tensor reads through the getattr fallback.
        #
        # FUTURE-RISK(onnx-device): this guard is INERT the moment ``SimTensor``
        # gains a ``device`` attribute, or the ONNX loader starts attaching one.
        # ``dev`` then becomes non-None for ONNX tensors, the gate resolves
        # through the process-wide default again, and the exclusion is gone.
        #
        # The failure is SILENT -- no exception.  The symptom is that ONNX
        # graphs re-key every elementwise binary onto the NHWC-flat view, so
        # ops downstream of an Add look up a LUT key hardware never recorded:
        # LUT misses that fall back to analytical estimates, and a
        # compare-layers report whose shapes no longer match the profiler.
        #
        # Nothing currently FAILS when that happens -- the existing
        # ``test_no_device_on_either_operand_is_not_an_error`` runs with the
        # process-wide gate at its default False, so it passes with or without
        # this guard.  If you are here because ONNX gained devices, the two
        # durable fixes are: require an explicit per-device override here (drop
        # the process-wide layer for this one gate), or give the ttnn shim its
        # own binary optype under ``com.tenstorrent.ttnn`` -- see the NOTE in
        # the docstring above for the four sites that second one touches.
        return
    if not bcast_hw_shape_propagation_enabled(dev):
        return
    if getattr(Y, 'hw_shape', None) is not None:
        return
    src = getattr(X0, 'hw_shape', None)
    if src is None:
        return
    # Only when the op preserves the logical shape; a genuinely reshaping
    # broadcast must not inherit X0's view.
    if list(getattr(X0, 'shape', [])) != list(getattr(Y, 'shape', [])):
        return
    Y.hw_shape = list(src)


def bidir_bcast(iTList, oTList, op, **kwargs):
    X0,X1, Y = iTList[0], iTList[1], oTList[0]
    assert X0.check_shape(), f"Input tensor-0 shape not defined: {X0}"
    assert X1.check_shape(), f"Input tensor-1 shape not defined: {X1}"
    Y.shape = bidirectional_broadcast_shape_inference(X0.shape, X1.shape)
    _propagate_bcast_hw_shape(X0, X1, Y)

    # Compute actual data if inputs have data
    _binary_compute_funcs = {
        'Add': compute_add,
        'Mul': compute_mul,
        'Sub': compute_sub,
        'Div': compute_div,
        'Pow': compute_pow,
        'Less': compute_less,
        'Atan2': compute_atan2,
        'Mod': compute_mod,
        'Equal': compute_equal,
    }
    if op.optype in _binary_compute_funcs:
        Y.data = try_compute_data(_binary_compute_funcs[op.optype], iTList, op)

    # Map operation to hardware instruction
    # Comparison operations (Less, Greater, etc.) use 'cmp' instruction
    optype_to_instr = {
        'Less': 'cmp',
        'And': 'cmp',
        'Or': 'cmp',
        'Xor': 'cmp',
        'Greater': 'cmp',
        'Equal': 'cmp',
        'LessOrEqual': 'cmp',
        'GreaterOrEqual': 'cmp',
        'BitwiseAnd': 'and',
        'BitwiseOr': 'or',
        'BitwiseXor': 'xor',
    }
    instr_name = optype_to_instr.get(op.optype, op.optype.lower())

    if instr_name == 'cmp':
        Y.dtype = 'bool'  # output dtype is bool for comparison ops
    else:
        Y.dtype = X0.dtype  # output dtype same as input for these ops

    op.perf_stats =  {
            'inElems' : X0.nelems() + X1.nelems(),
            'outElems': Y.nelems(),
            'inBytes' : X0.nbytes(op.precision) + X1.nbytes(op.precision),
            'outBytes': Y.nbytes(op.precision),
            'instrs'  : {instr_name: Y.nelems()}
            }
    return

def gridsample_sinf(iTList, oTList, op, **kwargs):
    """
    GridSample shape inference.
    Input shapes:
        iTList[0] (input): [N, C, H_in, W_in]
        iTList[1] (grid):  [N, H_out, W_out, 2]
    Output shape:
        oTList[0]: [N, C, H_out, W_out]
    """
    input_tensor = iTList[0]
    grid_tensor = iTList[1]

    assert input_tensor.check_shape(), f"Input tensor shape not defined: {input_tensor}"
    assert grid_tensor.check_shape(), f"Grid tensor shape not defined: {grid_tensor}"
    assert input_tensor.rank() == 4, f"Input must be 4D [N,C,H,W], got shape {input_tensor.shape}"
    assert grid_tensor.rank() == 4, f"Grid must be 4D [N,H_out,W_out,2], got shape {grid_tensor.shape}"
    assert grid_tensor.shape[3] == 2, f"Grid last dim must be 2 (x,y), got {grid_tensor.shape[3]}"

    N, C, H_in, W_in = input_tensor.shape
    N_grid, H_out, W_out, _ = grid_tensor.shape

    assert N == N_grid, f"Batch size mismatch: input={N}, grid={N_grid}"

    # Output shape: [N, C, H_out, W_out]
    oTList[0].shape = [N, C, H_out, W_out]
    oTList[0].dtype = input_tensor.dtype

    # Compute data if inputs have data
    from ttsim.ops.desc.data_compute import try_compute_data, compute_grid_sample
    oTList[0].data = try_compute_data(compute_grid_sample, iTList, op)

    # Performance stats: differentiate between nearest and bilinear modes
    mode = op.attrs.get('mode', 'bilinear')
    instr_count = {}

    if mode == 'nearest':
        instr_count['mov'] = oTList[0].nelems()
    elif mode == 'bilinear':
        instr_count['mul'] = oTList[0].nelems() * 4
        instr_count['add'] = oTList[0].nelems() * 3
    else:
        instr_count['mul'] = oTList[0].nelems() * 4
        instr_count['add'] = oTList[0].nelems() * 3

    op.perf_stats = {
        'inElems': input_tensor.nelems() + grid_tensor.nelems(),
        'outElems': oTList[0].nelems(),
        'inBytes': input_tensor.nbytes(op.precision) + grid_tensor.nbytes(op.precision),
        'outBytes': oTList[0].nbytes(op.precision),
        'instrs': instr_count
    }
    return
