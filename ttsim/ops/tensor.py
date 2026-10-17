#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import functools, operator
import warnings
from typing import Any, Optional, Union

import numpy as np
from ttsim.utils.types import get_bpe, get_sim_dtype

class Shape:
    """
    Shape class
    """

    def __init__(self, shape):
        if isinstance(shape, (list, tuple)):
            self._shape = list(shape)
        elif isinstance(shape, Shape):
            self._shape = list(shape._shape)
        else:
            raise TypeError(f"Invalid shape type: {type(shape)}")

    def __eq__(self, other):
        if other is None:
            return False
        elif isinstance(other, (list, tuple)):
            return list(other) == list(self._shape)
        elif not isinstance(other, Shape):
            return False
        return self._shape == other._shape

    def __getitem__(self, index):
        return self._shape[index]

    def __setitem__(self, index, value):
        self._shape[index] = value

    def __len__(self):
        return len(self._shape)

    def __iter__(self):
        return iter(self._shape)

    def __repr__(self):
        return f"Shape({self._shape})"

    def rank(self):
        return len(self._shape)

    def copy(self):
        return Shape(self._shape)

    def volume(self):
        result = 1
        for dim in self._shape:
            result *= dim
        return result

    def to_rank(self, rank):
        """Convert shape to specified rank by padding with 1s or truncating."""
        current_rank = len(self._shape)
        if current_rank == rank:
            return Shape(self._shape)
        elif current_rank < rank:
            # Pad with 1s at the beginning
            new_shape = [1] * (rank - current_rank) + self._shape
            return Shape(new_shape)
        else:
            # Truncate from the beginning
            new_shape = self._shape[-rank:]
            return Shape(new_shape)

    def as_list(self) -> list[Any]:
        """Copy of dimension sizes as a plain ``list`` for APIs that need list/tuple.

        Element types are not narrowed to ``int`` (e.g. ``numpy.integer`` may appear);
        callers that require strict ``int`` should normalize explicitly.
        """
        return list(self._shape)

    def view(self) -> list[Any]:
        warnings.warn(
            "Shape.view() is deprecated; use Shape.as_list() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.as_list()


def _coerce_shape_to_list(shape: Union[Shape, list[Any], tuple[Any, ...]]) -> list[Any]:
    """Normalize ``Shape`` or a plain sequence (as stored on some graph tensors) to ``list``."""
    if isinstance(shape, Shape):
        return shape.as_list()
    if isinstance(shape, (list, tuple)):
        return list(shape)
    raise TypeError(f"shape must be Shape, list, or tuple, got {type(shape)}")


def shape_as_optional_list(shape: Optional[Any]) -> Optional[list[Any]]:
    """Copy dimensions to a plain ``list``, or ``None`` if shape is unknown.

    Accepts :class:`Shape`, a ``list``/``tuple`` (some workload-graph tensors use
    raw sequences), or ``None``.
    """
    if shape is None:
        return None
    return _coerce_shape_to_list(shape)


def require_shape_list(shape: Optional[Any], msg: str = "shape must be set") -> list[Any]:
    """Return a dimension ``list`` after asserting shape is not ``None``.

    Accepts :class:`Shape` or a plain ``list``/``tuple`` like :func:`shape_as_optional_list`.
    """
    assert shape is not None, msg
    return _coerce_shape_to_list(shape)


def nchw_to_nhwc_flat(shape: list[int]) -> list[int]:
    """Convert NCHW [N, C, H, W] to hardware NHWC-flattened [1, 1, N*H*W, C].

    Tenstorrent hardware stores activations as [1, 1, N*H*W, C] (spatial dims
    folded into rows, channels last).  This hw_shape drives LUT key construction
    and profiler-shape comparison; the logical NCHW shape is kept in tensor.shape.

    Only valid for rank-4 NCHW inputs with 2 spatial dims.

    TODO: extend for rank-5 (3-D conv) if needed.
    """
    assert len(shape) == 4, f'nchw_to_nhwc_flat: expected rank-4 NCHW, got {shape}'
    N, C, H, W = shape
    return [1, 1, N * H * W, C]


class SimTensor:
    def __init__(self, cfg):
        self.name        = cfg['name']                # String
        self.dtype       = cfg.get('dtype')           # Numpy datatype
        # Convert list/tuple data to numpy array for compatibility
        data = cfg.get('data', None)
        if data is not None and isinstance(data, (list, tuple)):
            if self.dtype is not None:
                if hasattr(self.dtype, 'to_numpy'):
                    dtype = self.dtype.to_numpy
                else:
                    dtype = self.dtype
            else:
                dtype = np.float32
            data = np.array(data, dtype=dtype)
        self.data        = data                       # Actual data (numpy array)
        self.resolve     = cfg.get('resolve','_')     # Has the tensor shape been resolved (intermediate tensor shapes) (Boolean)
        self.op_in       = cfg.get('op_in', [])       # Which operators is this "input" for (consumer list)
        self.op_out      = cfg.get('op_out', [])      # Which operators is this "output" of (producer list)
        self.is_param    = cfg.get('is_param', False) # Is it parameter? Boolean
        self.is_const    = cfg.get('is_const', False) # Is it constant? Boolean
        self.has_grad    = cfg.get('has_grad', True)  # Has a gradient during bwd pass? Boolean
        self.link_module = None                       # Associated Module
        # hw_shape: NHWC-flattened [1, 1, N*H*W, C] used for LUT key matching and profiler comparison.
        # Set by conv/pool sinfs; propagated by Move/ITS/STI/Reshard/Concat passthrough sinfs.
        # None for ops that produce non-NCHW outputs (attention projections, embeddings, etc.).
        # TODO: set hw_shape for MaxPool/AvgPool descriptors if LUT coverage is extended.
        self.hw_shape: list[int] | None = None
        # x_pad_logical: padded channel count for LUT key construction when this tensor feeds
        # a BLOCK_SHARDED conv2d/conv_transpose2d.  Set by the Device.execute_graph annotation
        # pass (post-shim, post-arch-load) via determine_block_sharded_channel_padding.
        # See doc/TTNN_SHIM_ARCHITECTURE.md §17. None when the tensor doesn't participate in
        # a BLOCK_SHARDED conv input chain; LUT key builders then fall back to shape's X dim.
        self.x_pad_logical: int | None = None
        # y_pad_logical: halo-extended Y stick count override for LUT key construction.
        # Set by the Device.execute_graph annotation pass when an arch-specific halo
        # extension differs from the base _HALO_EXT_Y table (e.g. BH p100a's d2.conv1
        # picks num_cores_nhw=10 → y=5620 vs WH's y=5280).  See doc/TTNN_SHIM_ARCHITECTURE.md §17.
        self.y_pad_logical: int | None = None
        SimTensor.set_shape(self, cfg.get('shape'))   # Other classes that subclass might override set_shape

    def set_module(self, m): self.link_module = m

    def memory_config(self):
        return getattr(self, '_memory_config', None)

    def __str__(self):
        s  = f"SimTensor({self.name}) shape={self.shape}, dtype={self.dtype}, "
        s += f"is_param={self.is_param}, "
        s += f"is_const={self.is_const}, "
        s += f"has_grad={self.has_grad}, "
        s += f"op_in={self.op_in}, "
        s += f"op_out={self.op_out}, "
        if self.data is None:
            s += f"data={self.data}"
        elif self.rank() > 0 and self.nelems() > 5:
            s += "data=(...)"
        else:
            s += f"data={self.data.tolist()}"
        if self.link_module is not None:
            s += f", link_module={self.link_module.name}"
        return s

    def rank(self): return len(self.shape) if self.shape is not None else 0

    # Note: data count may not be a simple product of shape dims - may need to provide a custom func
    def nelems(self):
        if self.shape is None:
            return 0
        trank = self.rank()
        if trank > 0:
            res = functools.reduce(operator.mul, (k for k in self.shape), 1)
        elif trank == 0:
            res = 1
        else:
            assert False, f"What kinda tensor {self.name} is this? {self.shape}"
        if self.data is not None:
            assert isinstance(self.data, tuple([np.ndarray, np.float32, np.bool_])), f'data should be ndarray, is {type(self.data)}'
            if isinstance(self.data, np.ndarray):
                res1 = self.data.size
                assert res1 == res, f"Mismatch SimTensor({self.name}).nelems = {res} and np.size={res1}"
        return res

    def numel(self):
        return self.nelems()

    def set_shape(self, newshape):
        if newshape is None:
            self.shape = None
        else:
            self.shape = Shape(newshape)

    # Note:
    #   data size may not be just data-count * precision, because you may have compression/sparsity
    #   how is the tensor stored in memory? channel first, batch later or something else. may have
    #   to represent tiling formats here.
    # Note: Caching nbytes for instance methods can cause memory leaks due to references held by lru_cache.
    # If caching is needed, consider using a static cache or external memoization.
    def nbytes(self, itemprec=None):
        def typesize(dtype):
            if isinstance(dtype, np.dtype):
                return dtype.itemsize
            elif isinstance(dtype, str):
                return get_bpe(get_sim_dtype(dtype))
            else:
                try:
                    return np.dtype(dtype).itemsize
                except Exception:
                    raise TypeError(f"Unsupported dtype type: {type(dtype)}")
        if itemprec is None:
            assert self.dtype is not None, f"SimTensor({self.name}) has no dtype to calculate nbytes"
            itemsize = typesize(self.dtype)
        else:
            itemsize = typesize(itemprec)
        return self.nelems() * itemsize #assumes np.dtype

    def check_shape(self):
        if self.shape is None:
            return False
        elif all([ isinstance(d, int) or isinstance(d, np.int64) for d in self.shape]):
            return True
        else:
            return False

    def clone(self, clone_num:int):
        cloned_tensor = make_tensor(self.name + '.clone_{clone_num}')
        cloned_tensor.shape       = self.shape
        cloned_tensor.dtype       = self.dtype
        cloned_tensor.data        = self.data
        cloned_tensor.resolve     = self.resolve
        cloned_tensor.op_in       = self.op_in
        cloned_tensor.op_out      = self.op_out
        cloned_tensor.is_param    = self.is_param
        cloned_tensor.is_const    = self.is_const
        cloned_tensor.has_grad    = self.has_grad
        cloned_tensor.link_module = self.link_module
        return cloned_tensor

    def clone_by_shape(self, /, data_maybe_missing = True):
        assert self.shape is not None, f"Illegal Data in Tensor {self}"  # For mypy type checking
        assert self.check_shape(), f"Illegal Shape in Tensor {self}"
        if data_maybe_missing:
            if self.data is None:
                if self.rank() == 0:
                    if self.dtype == np.float32:
                        clone_data = np.float32(1.0)
                    else:
                        assert False, "Only np.float32 rank-0 tensor clones supported right now!!!"
                else:
                    cloned_data = np.random.randn(*(self.shape)).astype(self.dtype)
                clone = SimTensor({
                    'name'   : self.name + '.clone_by_shape',
                    'shape'  : self.shape,
                    'dtype'  : self.dtype,
                    'data'   : cloned_data,
                    'resolve': self.resolve,
                    'op_in'  : self.op_in,
                    'op_out' : self.op_out
                    })
            else:
                clone = self
        else:
            assert self.data is not None, f"Illegal Data in Tensor {self}"
            clone = self
        return clone

def make_tensor(name: str) -> SimTensor:
    return SimTensor({'name': name, 'shape': [], 'dtype': None})
