#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Optional, Tuple, Union

from ttsim.ops.op import SimOp
from ttsim.ops.tensor import SimTensor, Shape, require_shape_list
from .device import Device, USE_DEFAULT_DEVICE, resolve_device
from .types import TILE_HEIGHT, TILE_WIDTH

from enum import Enum, auto
from itertools import count
from numbers import Integral

from loguru import logger

import numpy as np


# Global op-name counter, shared by every shim op via generate_new_op_name(). Defined here in
# the base tensor module (not op.py) so op.py can import it at top level without a tensor<->op
# circular import (op.py already imports from tensor).
op_counter = count(start=1, step=1)


def generate_new_op_name():
    return f"ttsim.ttnn.Op_{next(op_counter)}"


########################################## DataType ##########################################
class DataType(Enum):
    UINT8 = auto()
    UINT16 = auto()
    UINT32 = auto()
    INT32 = auto()
    INT64 = auto()
    FLOAT32 = auto()
    BFLOAT16 = auto()
    BFLOAT8_B = auto()
    BFLOAT4_B = auto()
    BOOL = auto()

    @classmethod
    def enumvalue(cls, s: str):
        return DataType[s.upper()]

    @property
    def itemsize(self) -> float:
        return {
            "UINT8": 1,
            "UINT16": 2,
            "UINT32": 4,
            "INT32": 4,
            "INT64": 8,
            "FLOAT32": 4,
            "BFLOAT16": 2,
            "BFLOAT8_B": 1,  # 8-bit compact format
            "BFLOAT4_B": 0.5,  # 4-bit format, 2 elements per byte
            "BOOL": 1,
        }[self.name]

    @property
    def to_numpy(self):
        return {
            "UINT8": np.dtype(np.uint8),
            "UINT16": np.dtype(np.uint16),
            "UINT32": np.dtype(np.uint32),
            "INT32": np.dtype(np.int32),
            "INT64": np.dtype(np.int64),
            "FLOAT32": np.dtype(np.float32),
            "BFLOAT16": np.dtype(np.float16),  # float16 not supported in onnx dump!!
            "BFLOAT8_B": np.dtype(np.float32),
            "BFLOAT4_B": np.dtype(np.float32),
            "BOOL": np.dtype(np.uint8),
        }[self.name]

    @classmethod
    def from_numpy(cls, numpy_dtype):
        if hasattr(numpy_dtype, "dtype"):
            numpy_dtype = numpy_dtype.dtype

        if hasattr(numpy_dtype, "name"):
            dtype_str = numpy_dtype.name
        elif isinstance(numpy_dtype, str):
            dtype_str = numpy_dtype
        else:
            dtype_str = str(numpy_dtype)

        # Mapping from numpy dtype names to DataType enums
        dtype_mapping = {
            "uint8": cls.UINT8,
            "uint16": cls.UINT16,
            "int32": cls.INT32,
            "uint32": cls.UINT32,
            "int64": cls.INT64,
            "float32": cls.FLOAT32,
            "float16": cls.BFLOAT16,
            "bool": cls.UINT8,  # Map bool to uint8
        }

        # Try exact match first
        if dtype_str in dtype_mapping:
            return dtype_mapping[dtype_str]

        # Try with numpy dtype object mapping
        numpy_dtype_mapping = {
            np.dtype(np.uint8): cls.UINT8,
            np.dtype(np.uint16): cls.UINT16,
            np.dtype(np.int32): cls.INT32,
            np.dtype(np.uint32): cls.UINT32,
            np.dtype(np.int64): cls.INT64,
            np.dtype(np.float32): cls.FLOAT32,
            np.dtype(np.float16): cls.BFLOAT16,
        }

        if numpy_dtype in numpy_dtype_mapping:
            return numpy_dtype_mapping[numpy_dtype]

        # Default fallback
        return cls.FLOAT32

    @property
    def cname(self) -> str:
        return self.name.lower()


class Layout(Enum):
    ROW_MAJOR_LAYOUT = auto()
    ROW_MAJOR = auto()
    TILE_LAYOUT = auto()
    DEFAULT  = ROW_MAJOR_LAYOUT

    @classmethod
    def enumvalue(cls, s: str):
        return Layout[s.upper()]

    @property
    def cname(self) -> str:
        return self.name.lower()

    @classmethod
    def from_numpy(cls, numpy_layout_str):
        layout_str = numpy_layout_str.lower()
        if layout_str in ["c", "row_major", "row_major_layout"]:
            return cls.ROW_MAJOR_LAYOUT
        elif layout_str in ["f", "column_major", "column_major_layout"]:
            raise NotImplementedError("Column major layout not supported yet")
        else:
            raise ValueError(f"Unknown numpy layout string: {numpy_layout_str}")

class Tensor(SimTensor):
    tensor_counter = count(start=1, step=1)

    def __init__(self, *args, **kwargs):
        if args:
            assert (
                len(args) == 1
            ), f"More than 1 positional argument in Tensor constructor!!: {args}"
            tensor_like = args[0]
            assert isinstance(tensor_like, np.ndarray), (
                f"Tensor single positional argument must be a numpy.ndarray, "
                f"got {type(tensor_like).__name__}"
            )
            dtype, shape = tensor_like.dtype, tensor_like.shape
            # ignoring dtype for now -- eventually will need to reconcile these with kwargs!!
            kwargs["shape"] = tensor_like.shape

        kwargs['layout'] = kwargs.get('layout', Layout.DEFAULT)
        if kwargs["layout"] is None:
            kwargs["layout"] = Layout.DEFAULT
        # Omitted device → USE_DEFAULT_DEVICE → get_default_device(); explicit device=None → host.
        kwargs['device'] = resolve_device(kwargs.get('device', USE_DEFAULT_DEVICE))
        if 'dtype' in kwargs and not isinstance(kwargs['dtype'], (DataType, np.dtype)):
            raise TypeError(f"Error: Tensor Creation -- attribute dtype={kwargs['dtype']} should be of type DataType or numpy.dtype")
        if 'layout' in kwargs and not isinstance(kwargs['layout'], Layout):
            raise TypeError(f"Error: Tensor Creation -- attribute layout={kwargs['layout']} should be of type Layout")
        if kwargs['device'] is not None and not isinstance(kwargs['device'], Device):
            raise TypeError(f"Error: Tensor Creation -- attribute device={kwargs['device']} should be of type Device or None")

        # NumPy has no native bfloat8/bfloat4 types, so DataType.BFLOAT8_B and
        # BFLOAT4_B both map to np.float32 via to_numpy -- making them
        # indistinguishable from FLOAT32.  We preserve the original DataType
        # enum so that stats/CSV output can report the true ttnn dtype.
        self._ttnn_dtype: DataType | None = None
        if 'dtype' in kwargs and isinstance(kwargs['dtype'], DataType):
            self._ttnn_dtype = kwargs['dtype']
            kwargs['dtype'] = kwargs['dtype'].to_numpy

        if "name" not in kwargs:
            kwargs["name"] = f"ttsim.ttnn.Tensor_{next(self.tensor_counter)}"

        if 'shape' in kwargs:
            shape = kwargs['shape']
            if isinstance(shape, (list, tuple)):
                kwargs['shape'] = list(shape)
                self._logical_shape = Shape(shape)
            elif isinstance(shape, Shape):
                self._logical_shape = Shape(shape)
            else:
                raise TypeError(f"Invalid shape type: {type(shape)}")
        else:
            kwargs['shape'] = []
            self._logical_shape = Shape([])
        super().__init__(kwargs)

        self.device     = kwargs['device']
        self.layout     = kwargs['layout']

        # Calculate padded shape if not provided
        padded_shape = kwargs.get('padded_shape', None)
        if padded_shape is None:
            self._padded_shape = self._calculate_padded_shape(self._logical_shape, self.layout)
        else:
            if isinstance(padded_shape, (list, tuple)):
                self._padded_shape = Shape(padded_shape)
            elif isinstance(padded_shape, Shape):
                self._padded_shape = Shape(padded_shape)
            else:
                raise TypeError(f"Invalid padded_shape type: {type(padded_shape)}")

        logger.debug("Tensor constructor: {} logical_shape {} and padded_shape {}", self.name, self._logical_shape, self._padded_shape)

        self.fill_value = kwargs.get('fill_value', None)
        self._memory_config = kwargs.get('memory_config', None)
        self._storage_type = "DEVICE" if self.device is not None else "HOST"
        # Essential for to_layout: device vs host is decided by buffer() is not None; without this,
        # ttsim Tensors would always be treated as host and layout _op (tilize_op, etc.) would never be used.
        self._buffer = None if self.device is None else "buffer_placeholder"
        if self.device:
            self.device.add_tensor(self)
        return

    def buffer(self):
        """Return the buffer placeholder when tensor is on device; None when on host. Used by to_layout to distinguish device vs host path."""
        return getattr(self, '_buffer', None)

    def __str__(self):
        return f"{super().__str__()} ==> ttnn: {self.device}, {self.layout}"

    @property
    def T(self):
        opname = self.name + ".transpose_op"
        optype = "Transpose"
        perm = [i for i in range(self.rank())]
        perm[-2], perm[-1] = perm[-1], perm[-2]  # swap last 2 dims
        opinfo = {
            "name": opname,
            "optype": optype,
            "inList": [self.name],
            "attrs": {"perm": perm},
        }
        outT = Tensor(name=opname + ".out", op_out=[opname], device=self.device)
        opinfo["outList"] = [outT.name]

        opobj = SimOp(opinfo)
        pstats = opobj.get_perf_counts([self], [outT])
        opobj.update_tensor_counts([self], [outT])

        self.device.add_op(opobj)  # type: ignore[union-attr]

        return outT

    def get_layout(self):
        return self.layout

    def padded_shape(self):
        return self._padded_shape

    def _calculate_padded_shape(self, logical_shape, layout):
        """Calculate padded shape based on logical shape and layout."""
        if len(logical_shape) == 0:
            return Shape([])

        padded = logical_shape.as_list()
        if layout == Layout.TILE_LAYOUT and len(padded) >= 2:
            # Pad last two dimensions to tile boundaries
            padded[-2] = ((padded[-2] + TILE_HEIGHT - 1) // TILE_HEIGHT) * TILE_HEIGHT
            padded[-1] = ((padded[-1] + TILE_WIDTH - 1) // TILE_WIDTH) * TILE_WIDTH
        return Shape(padded)

    def view(self, *args):
        npdata = np.array(args, dtype=np.int64)
        opname = self.name + ".view_op"
        shapeT = Tensor(
            name=opname + ".shapeT",
            device=self.device,
            data=npdata,
            shape=list(npdata.shape),
            dtype=DataType.INT64,
            op_in=[opname],
        )
        optype = "Reshape"
        opinfo = {"name": opname, "optype": optype, "inList": [self.name, shapeT.name]}
        outT = Tensor(name=opname + ".out", op_out=[opname], device=self.device)
        opinfo["outList"] = [outT.name]

        opobj = SimOp(opinfo)
        pstats = opobj.get_perf_counts([self, shapeT], [outT])
        opobj.update_tensor_counts([self, shapeT], [outT])
        self.device.add_op(opobj)  # type: ignore[union-attr]

        return outT

    def unsqueeze(self, dim: int):
        """Unsqueeze the tensor at the specified dimension."""
        assert self.shape is not None
        if dim < 0:
            dim += len(self.shape) + 1
        new_shape = self.shape[:dim] + [1] + self.shape[dim:]
        return Tensor(shape=new_shape, dtype=DataType.from_numpy(self.dtype.name), device=self.device)

    def squeeze(self, dim: int):
        """Squeeze the tensor at the specified dimension."""
        assert self.shape is not None
        if dim < 0:
            dim += len(self.shape)
        if dim >= len(self.shape) or self.shape[dim] != 1:
            logger.warning(
                "Cannot squeeze dimension {} of shape {} (dim out of range or size != 1); "
                "returning tensor unchanged",
                dim,
                self.shape,
            )
            return self
        new_shape = self.shape[:dim] + self.shape[dim + 1 :]
        return Tensor(
            shape=new_shape,
            dtype=DataType.from_numpy(self.dtype.name),
            device=self.device,
        )

    def memory_config(self):
        return self._memory_config

    def element_size(self):
        """Return element size in bytes, preferring _ttnn_dtype when available."""
        if self._ttnn_dtype is not None:
            # Use _ttnn_dtype for accurate compact dtype sizes
            return self._ttnn_dtype.itemsize
        # Fall back to np.dtype itemsize
        return self.dtype.itemsize

    def storage_type(self):
        return self._storage_type

    def logical_shape(self):
        return self.shape # self._logical_shape

    def physical_volume(self):
        """Element count in padded storage shape; used by shim execute paths (e.g. untilize_with_unpadding)."""
        return self.padded_shape().volume()

    def has_data(self):
        return self.data is not None

    def get_data(self):
        return self.data

    def to(self, *args, **kwargs):
        if len(args) == 1 and not kwargs and isinstance(args[0], DataType):
            self._ttnn_dtype = args[0]
            self.dtype = args[0].to_numpy
            return self

        raise TypeError(
            "Tensor.to() only supports the signature Tensor.to(DataType); "
            f"got args={args}, kwargs={kwargs}"
        )
    def set_shape(self, newshape):
        super().set_shape(newshape) 
        if newshape is not None:
            self._padded_shape = self._calculate_padded_shape(Shape(newshape), self.layout)
        logger.debug("set_shape: {} layout {} newshape {} and padded_shape {}", self.name, self.layout, newshape, self._padded_shape)


    def item(self):
        """returns the Python scalar value of the tensor if the tensor has exactly one element
        (i.e., it is a 0-dimensional tensor or a scalar tensor). If the tensor has more than one
        element, calling item() will raise an error. If the tensor is empty/None item fails again!!
        """
        assert self.shape == [
            1
        ], f"Tensor item() is valid only for tensor with exactly one element: {self.shape}"
        assert (
            self.data is not None
        ), f"Tensor item() called for missing data: {self.data}"
        return self.data[0]

    def float(self):
        return Tensor(shape=self.shape, dtype=DataType.FLOAT32, device=self.device)

    def size(self, dim: Optional[int] = None) -> Union[Tuple[int, ...], int]:
        assert self.shape is not None
        if dim is None:
            # require_shape_list, not Shape.as_list: shape is a plain list
            # whenever it was assigned by a shape-inference descriptor rather
            # than through set_shape, and as_list raised AttributeError there.
            return tuple(require_shape_list(self.shape))
        return self.shape[dim]

    def gather(self, dim, index):
        import ttsim.front.ttnn.op as ttnn_op

        return ttnn_op.gather(self, dim, index)

    def expand(self, *sizes):
        """Expand tensor to specified size. Only singleton dimensions (size 1) can be expanded."""
        # Handle sizes input - can be passed as separate args or as a single tuple/list
        assert self.shape is not None
        if len(sizes) == 1 and isinstance(sizes[0], (list, tuple)):
            target_shape = list(sizes[0])
        else:
            target_shape = list(sizes)

        original_shape = self.shape
        # Same reason as size(): shape may be a plain list here.
        orig_dims = require_shape_list(original_shape)
        # Pad original shape with 1s if target has more dimensions
        if len(target_shape) > len(original_shape):
            padded_original = [1] * (
                len(target_shape) - len(original_shape)
            ) + orig_dims
        else:
            padded_original = orig_dims

        # Check that target shape has at least as many dimensions as original
        if len(target_shape) < len(original_shape):
            raise ValueError(
                f"Cannot expand tensor of shape {original_shape} to shape {target_shape}. "
                f"Target shape must have at least {len(original_shape)} dimensions."
            )

        # Validate expansion: can only expand singleton dimensions
        for i, (orig_dim, target_dim) in enumerate(zip(padded_original, target_shape)):
            if target_dim == -1:
                # -1 means keep original dimension
                target_shape[i] = orig_dim
            elif orig_dim != 1 and orig_dim != target_dim:
                raise ValueError(
                    f"Cannot expand dimension {i} from size {orig_dim} to {target_dim}. "
                    f"Only singleton dimensions (size 1) can be expanded."
                )
            elif target_dim < 0:
                raise ValueError(
                    f"Invalid target size {target_dim} for dimension {i}. "
                    f"Target sizes must be positive or -1."
                )

        # Create expanded tensor (this is a view operation in real PyTorch)
        return Tensor(
            shape=target_shape,
            dtype=DataType.from_numpy(self.dtype.name),
            device=self.device,
        )

    def flatten(self, start_dim=0, end_dim=-1):
        """Flatten tensor dimensions from start_dim to end_dim into a single dimension."""
        # Handle negative dimensions
        assert self.shape is not None
        ndim = len(self.shape)
        if start_dim < 0:
            start_dim += ndim
        if end_dim < 0:
            end_dim += ndim

        # Validate dimensions
        if start_dim < 0 or start_dim >= ndim:
            raise ValueError(
                f"start_dim {start_dim} is out of range for tensor with {ndim} dimensions"
            )
        if end_dim < 0 or end_dim >= ndim:
            raise ValueError(
                f"end_dim {end_dim} is out of range for tensor with {ndim} dimensions"
            )
        if start_dim > end_dim:
            raise ValueError(f"start_dim {start_dim} must be <= end_dim {end_dim}")

        # Calculate new shape
        new_shape: list[Any] = []

        # Add dimensions before start_dim
        new_shape.extend(self.shape[:start_dim])

        # Calculate flattened dimension size
        flattened_size = 1
        for i in range(start_dim, end_dim + 1):
            flattened_size *= self.shape[i]
        new_shape.append(flattened_size)

        # Add dimensions after end_dim
        new_shape.extend(self.shape[end_dim + 1 :])

        # Create flattened tensor (this is a view operation in real PyTorch)
        return Tensor(
            shape=new_shape,
            dtype=DataType.from_numpy(self.dtype.name),
            device=self.device,
        )

    def repeat(self, *repeats):
        """Repeat the tensor along specified dimensions."""
        assert self.shape is not None
        new_shape = [dim * repeat for dim, repeat in zip(self.shape, repeats)]
        return Tensor(
            shape=new_shape,
            dtype=DataType.from_numpy(self.dtype.name),
            device=self.device,
        )

    def clone(self, clone_num=0):
        """Faithful metadata clone: same shape, dtype, layout, memory_config, and device.
        Metadata only — SimTensors carry no data, so this is not a torch-style data copy.

        Real ttnn clone preserves the tensor's layout and memory_config — so the previous
        copy (shape+dtype+device only) was an unfaithful mimic that silently dropped
        ``layout`` and ``_memory_config``. Both are LUT-key fields, so a lossy clone would
        make any op built from the copy miss the LUT. Preserve the true ttnn dtype via
        ``_ttnn_dtype`` (bfloat8_b/bfloat4_b are indistinguishable once mapped to numpy
        float32). No active caller depended on the lossy behavior."""
        dtype = self._ttnn_dtype if self._ttnn_dtype is not None else DataType.from_numpy(self.dtype.name)
        return Tensor(
            shape=self.shape,
            dtype=dtype,
            layout=self.layout,
            memory_config=self._memory_config,
            device=self.device,
        )

    def rename(self, new_name):
        """Rename this tensor, keeping the front-end device registry consistent.

        ``Tensor.__init__`` registered this tensor in ``device.tensors`` under its original
        (often auto-generated) name; a bare ``self.name = x`` would leave a stale
        ``old_name -> tensor`` entry and never register the new name. Used by backend graph
        surgery (e.g. the lm_head column-split) that clones tensors and then renames them."""
        if self.device is not None:
            self.device.tensors.pop(self.name, None)
        self.name = new_name
        if self.device is not None:
            self.device.add_tensor(self)

    def new_zeros(self, shape):
        return Tensor(
            shape=shape,
            dtype=DataType.from_numpy(self.dtype.name),
            device=self.device,
            fill_value=0,
        )

    def new_tensor(self, data):
        if isinstance(data, list) and all(
            isinstance(item, np.ndarray) for item in data
        ):
            # Convert list of arrays to single concatenated array
            data = np.concatenate(data).reshape(1, -1)
        return Tensor(
            shape=data.shape,
            dtype=DataType.from_numpy(self.dtype.name),
            device=self.device,
            data=data,
        )


class ShardStrategy(Enum):
    HEIGHT = auto()
    WIDTH = auto()
    BLOCK = auto()

    @classmethod
    def enumvalue(cls, s: str):
        return ShardStrategy[s.upper()]

    @property
    def cname(self) -> str:
        return self.name.lower()


def require_ttnn_tensor(value, arg_name: str = "tensor") -> "Tensor":
    """Ensure a value is a TTNN front-end :class:`Tensor` (not a bare :class:`SimTensor`).

    TTNN ops require ``ttsim.front.ttnn.tensor.Tensor`` for device, layout, and graph wiring.
    """
    if isinstance(value, Tensor):
        return value
    if isinstance(value, SimTensor):
        raise TypeError(
            f"{arg_name} must be ttsim.front.ttnn.tensor.Tensor, not ttsim.ops.tensor.SimTensor "
            "(construct Tensor(...) or use as_tensor / numpy-backed Tensor APIs)."
        )
    raise TypeError(
        f"{arg_name} must be ttsim.front.ttnn.tensor.Tensor, got {type(value).__name__}"
    )


def as_tensor(
    tensor_like,
    dtype=None,
    layout=None,
    device=None,
    fill_value=None,
    mesh_mapper=None,
    memory_config=None,
    cache_file_name=None,
):
    if isinstance(tensor_like, Tensor):
        if device is None:
            return tensor_like
        return to_device(tensor_like, resolve_device(device))

    if isinstance(tensor_like, np.ndarray):
        shape = tensor_like.shape
        if dtype is None:
            dtype = DataType.from_numpy(tensor_like.dtype.name)
        return Tensor(
            shape=shape,
            dtype=dtype,
            layout=layout,
            device=device,
            fill_value=fill_value,
            data=tensor_like,
        )

    raise TypeError(f"Unsupported type for as_tensor: {type(tensor_like)}")


def _rand(shape, dtype, device=USE_DEFAULT_DEVICE):
    return Tensor(shape=shape, dtype=dtype, device=device)

def zeros(shape, dtype, layout=Layout.DEFAULT, device=USE_DEFAULT_DEVICE):
    return Tensor(shape=shape, dtype=dtype, layout=layout, device=device, fill_value=0)

def _normalize_shape_list(shape, arg_name="shape"):
     actual_shape = list(shape)
     if not all(isinstance(dim, Integral) and not isinstance(dim, bool) for dim in actual_shape):
         raise TypeError(f"{arg_name} must contain only integer dimensions")
     return actual_shape


def ones(*args, shape=None, dtype=None, layout=Layout.DEFAULT, device=USE_DEFAULT_DEVICE):
    if shape is not None:
        actual_shape = _normalize_shape_list(shape, arg_name="shape")
    elif len(args) == 1 and isinstance(args[0], (list, tuple)):
        actual_shape = _normalize_shape_list(args[0], arg_name="shape")
    else:
        actual_shape = _normalize_shape_list(args, arg_name="args")
    return Tensor(shape=actual_shape, dtype=dtype, layout=layout, device=device, fill_value=1)


def full(shape, fill_value, dtype, layout, device):
    return Tensor(
        shape=shape, dtype=dtype, layout=layout, device=device, fill_value=fill_value
    )


def arange(*args, **kwargs):
    # Support arange(length) or arange(start, end)
    if len(args) == 1:
        length = args[0]
        return Tensor(
            shape=[length],
            dtype=DataType.INT64,
            data=np.arange(length),
            device=kwargs.get("device", USE_DEFAULT_DEVICE),
        )
    elif len(args) == 2:
        start, end = args
        length = end - start
        return Tensor(
            shape=[length],
            dtype=DataType.INT64,
            data=np.arange(start, end),
            device=kwargs.get("device", USE_DEFAULT_DEVICE),
        )
    else:
        raise ValueError(
            "arange expects either a single argument (length) or two arguments (start, end)"
        )


def pad(input_tensor, pad, mode="constant", value=0):
    if mode != "constant":
        raise NotImplementedError("Only 'constant' padding mode is implemented")

    if len(pad) % 2 != 0:
        raise ValueError(
            "Padding length must be even, representing (before, after) pairs for each dimension"
        )

    num_dims = len(input_tensor.shape)
    if len(pad) // 2 > num_dims:
        raise ValueError(
            f"Padding length {len(pad)} is too large for tensor with {num_dims} dimensions"
        )

    # Create new shape after padding
    new_shape = list(input_tensor.shape)
    for i in range(len(pad) // 2):
        before = pad[2 * i]
        after = pad[2 * i + 1]
        dim_index = num_dims - 1 - i
        new_shape[dim_index] += before + after

    return Tensor(
        shape=new_shape,
        dtype=DataType.from_numpy(input_tensor.dtype.name),
        device=input_tensor.device,
    )



def ttnn_random(shape, low, high, dtype):
    if dtype in [DataType.INT64, DataType.INT32, DataType.UINT16, DataType.UINT8]:
        return _rand(shape, dtype=dtype)
    return _rand(shape, dtype=dtype)


def stack(tensors, dim=0):
    first_tensor = tensors[0]
    for i, tensor in enumerate(tensors[1:], 1):
        if tensor.shape != first_tensor.shape:
            raise ValueError(
                f"All tensors must have the same shape to be stacked. "
                f"Tensor 0 has shape {first_tensor.shape}, but tensor {i} has shape {tensor.shape}"
            )
        if tensor.dtype != first_tensor.dtype:
            raise ValueError(
                f"All tensors must have the same dtype to be stacked. "
                f"Tensor 0 has dtype {first_tensor.dtype}, but tensor {i} has dtype {tensor.dtype}"
            )

    # Handle negative dimension
    original_rank = len(first_tensor.shape)
    if dim < 0:
        dim += original_rank + 1  # +1 because we're adding a new dimension

    # Validate dimension
    if dim < 0 or dim > original_rank:
        raise ValueError(
            f"Dimension {dim} is out of range for tensors of rank {original_rank}. "
            f"Valid range is [-{original_rank + 1}, {original_rank}]"
        )

    # Create new shape for the stacked tensor
    new_shape = list(first_tensor.shape)
    new_shape.insert(dim, len(tensors))

    # Create the stacked tensor
    return Tensor(
        shape=new_shape,
        dtype=DataType.from_numpy(first_tensor.dtype.name),
        device=first_tensor.device,
    )


def unsqueeze_to_4D(input_tensor):
    """Unsqueeze a tensor to 4D shape by adding dimensions of size 1."""
    if len(input_tensor.shape) == 4:
        return input_tensor

    new_shape = [1] * (4 - len(input_tensor.shape)) + list(input_tensor.shape)
    return Tensor(
        shape=new_shape,
        dtype=DataType.from_numpy(input_tensor.dtype.name),
        device=input_tensor.device,
        layout=input_tensor.get_layout(),
    )


def ShardTensor2dMesh(device, dims=None, mesh_shape=None):
    # dummy implementation for ShardTensor2dMesh
    pass


def ReplicateTensorToMesh(device):
    # dummy implementation for ReplicateTensorToMesh
    pass


def typecast(input_tensor, dtype):
    """Typecast the input tensor to the specified dtype."""
    if not isinstance(input_tensor, Tensor):
        raise TypeError(
            f"Expected input_tensor to be a Tensor, got {type(input_tensor)}"
        )

    # Compare the true ttnn dtype, not input_tensor.dtype (a numpy dtype) — the latter never
    # equals a DataType enum, so the same-dtype elision was dead. Prefer _ttnn_dtype (bfloat8_b/
    # bfloat4_b are indistinguishable once mapped to numpy float32); fall back like clone().
    in_ttnn_dtype = (input_tensor._ttnn_dtype if input_tensor._ttnn_dtype is not None
                     else DataType.from_numpy(input_tensor.dtype.name))
    if in_ttnn_dtype == dtype:
        return input_tensor  # No typecasting needed (HW emits no op for a same-dtype cast)

    # Create the re-dtyped output. Preserve the input's layout and memory_config — HW typecast is
    # elementwise and does not change tiling/placement (e.g. the decode sampling typecast keeps the
    # lm_head logits TILE/L1, so the downstream TopK sees a TILE/L1 input). Dropping them defaulted
    # to ROW_MAJOR/DRAM and broke LUT-key matching.
    # Unique per-call op name via generate_new_op_name() (the shim-wide naming convention),
    # keeping a dtype tag for readability. Op names must be unique (add_op asserts), so a name
    # derived only from the input + dtype would still collide if the same tensor were typecast
    # to the same dtype twice.
    dtype_tag = getattr(dtype, 'name', str(dtype))
    opname = f"{generate_new_op_name()}.typecast_{dtype_tag}"
    outT = Tensor(shape=input_tensor.shape, device=input_tensor.device, dtype=dtype,
                  layout=input_tensor.get_layout(),
                  memory_config=getattr(input_tensor, '_memory_config', None),
                  op_out=[opname])
    # Emit a Typecast SimOp so the op appears in the graph (HW TypecastDeviceOperation), matching the
    # silicon capture. Previously typecast was metadata-only (no graph node), so the decode
    # sampling-tail typecasts were profiler-only. Elementwise: 1 input -> 1 output (see typecast_sinf).
    if input_tensor.device is not None:
        input_tensor.op_in.append(opname)
        opinfo = {
            "name": opname,
            "optype": "Typecast",
            "inList": [input_tensor.name],
            "outList": [outT.name],
            "attrs": {"dtype": str(dtype)},
        }
        opobj = SimOp(opinfo)
        opobj.get_perf_counts([input_tensor], [outT])
        opobj.update_tensor_counts([input_tensor], [outT])
        input_tensor.device.add_op(opobj)  # type: ignore[union-attr]
    return outT


def from_torch(torch_tensor_like, **kwargs):
    for k, v in kwargs.items():
        if hasattr(torch_tensor_like, k):
            if k == "dtype" and isinstance(v, DataType):
                torch_tensor_like._ttnn_dtype = v
                v = v.to_numpy
            setattr(torch_tensor_like, k, v)

    if "device" in kwargs:
        torch_tensor_like = to_device(torch_tensor_like, kwargs["device"])

    return torch_tensor_like


def to_torch(tt_tensor_like):
    return tt_tensor_like

def to_device(tt_tensor_like, device, memory_config=None):
    device = resolve_device(device)
    assert device is not None, "device=None passed to to_device"

    if tt_tensor_like.device:
        old_device = tt_tensor_like.device
        if tt_tensor_like.name in old_device.tensors:
            del old_device.tensors[tt_tensor_like.name]

    tt_tensor_like.device = device
    device.add_tensor(tt_tensor_like)

    # Update storage attributes to mark tensor as device-resident
    # (consistent with Tensor.__init__ behavior for device tensors)
    tt_tensor_like._storage_type = "DEVICE"
    tt_tensor_like._buffer = "buffer_placeholder"

    if memory_config is not None:
        tt_tensor_like._memory_config = memory_config

    return tt_tensor_like
