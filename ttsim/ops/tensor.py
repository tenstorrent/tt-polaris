#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import functools, operator
import warnings
from collections.abc import Iterable, Iterator, Sequence
from typing import Any, Optional, Union, overload

import numpy as np
from ttsim.utils.types import get_bpe, get_sim_dtype

# One dimension of a shape, as stored by Shape.
#
# What this guarantees, precisely: a Shape built through Shape() or set_shape()
# normalises np.int64 inputs, so its dims are int and everything derived from
# them -- indexing, iteration, as_list, volume -- is int too.
#
# What it does NOT guarantee: that every SimTensor.shape is a Shape. Much of the
# shape-inference layer assigns a plain list straight to SimTensor.shape,
# bypassing set_shape entirely, and those dims are whatever the inference
# produced. See the declaration of SimTensor.shape, which states that union
# rather than hiding it. Routing those writes through set_shape is the fix, and
# is separate work.
#
# int is also the safer dimension type, independent of the annotation. Python
# ints are arbitrary precision; np.int64 is fixed width and wraps silently on
# overflow, reporting only a RuntimeWarning. volume() and nelems() multiply
# dims, so an element count above 2**63 wraps -- possibly to a negative number
# -- if the dims are numpy integers, and is exact if they are ints. Converting
# np.int64 to int is lossless across the whole int64 range, so the normalisation
# removes that hazard rather than trading one for another.
#
# Declaring the numpy type instead is not an option: int and np.int64 are not
# subtypes of each other, so `Dim = np.int64` rejects every plain-int dim, and a
# union pushes narrowing into every consumer of a dimension.
Dim = int

# What a Shape accepts on construction, before normalisation. Shape stores Dim;
# this is deliberately looser, because normalising the inputs is its job.
#
# np.int64 and not np.integer: check_shape accepted `int` and `np.int64` only,
# so normalising every np.integer would make a np.int32 dim valid where it was
# rejected before. Widening what counts as a valid shape is not this change's
# business.
DimLike = Union[int, np.int64]


def _normalise_dim(d: Any) -> Any:
    """Convert an accepted numpy dim to int, and leave anything else untouched.

    Returning a non-Dim unchanged is deliberate: a float or None dim stays as it
    is so that :meth:`SimTensor.check_shape` still rejects it, rather than being
    coerced into a plausible-looking int.
    """
    return int(d) if isinstance(d, np.int64) else d

class Shape:
    """
    Shape class
    """

    def __init__(self, shape: Union['Shape', Sequence[DimLike]]) -> None:
        self._shape: list[Dim]
        if isinstance(shape, Shape):
            self._shape = list(shape._shape)
        elif isinstance(shape, Sequence) and not isinstance(shape, (str, bytes)):
            # Any Sequence, not just list and tuple, because that is what the
            # signature promises: `Shape(range(3))` type-checked and then raised
            # before. str and bytes are excluded -- they are Sequences whose
            # elements are never dims.
            self._shape = [_normalise_dim(d) for d in shape]
        else:
            raise TypeError(f"Invalid shape type: {type(shape)}")

    def __eq__(self, other: object) -> bool:
        if other is None:
            return False
        elif isinstance(other, (list, tuple)):
            return list(other) == list(self._shape)
        elif not isinstance(other, Shape):
            return False
        return self._shape == other._shape

    @overload
    def __getitem__(self, index: int) -> Dim: ...
    @overload
    def __getitem__(self, index: slice) -> list[Dim]: ...

    def __getitem__(self, index: Union[int, slice]) -> Union[Dim, list[Dim]]:
        return self._shape[index]

    # DimLike, not Dim: assignment accepts what construction accepts and
    # normalises it. Declaring Dim here rejected `shape[0] = np.int64(2)`, which
    # is supported and tested.
    @overload
    def __setitem__(self, index: int, value: DimLike) -> None: ...
    @overload
    def __setitem__(self, index: slice, value: Iterable[DimLike]) -> None: ...

    def __setitem__(self, index: Union[int, slice], value: Union[DimLike, Iterable[DimLike]]) -> None:
        # Construction normalises, so mutation has to as well; otherwise
        # `shape[0] = np.int64(2)` puts a numpy value back into _shape and the
        # int contracts on indexing, iteration, as_list and volume break again.
        if isinstance(index, slice):
            self._shape[index] = [_normalise_dim(d) for d in value]  # type: ignore[union-attr]
        else:
            self._shape[index] = _normalise_dim(value)

    def __len__(self) -> int:
        return len(self._shape)

    def __iter__(self) -> Iterator[Dim]:
        return iter(self._shape)

    def __repr__(self) -> str:
        return f"Shape({self._shape})"

    def rank(self) -> int:
        return len(self._shape)

    def copy(self) -> 'Shape':
        return Shape(self._shape)

    def volume(self) -> int:
        result = 1
        for dim in self._shape:
            result *= dim
        return result

    def to_rank(self, rank: int) -> 'Shape':
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

    def as_list(self) -> list[Dim]:
        """Copy of dimension sizes as a plain ``list`` for APIs that need list/tuple."""
        return list(self._shape)

    def view(self) -> list[Dim]:
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
        # Declared as the union that is actually stored, not the Shape that
        # set_shape produces. Roughly 120 sites under ttsim/ops/desc assign a
        # plain list here directly, so declaring Optional[Shape] would let the
        # checker approve `shape.volume()` and `shape.as_list()` on a value that
        # is a list at run time. Narrowing this to Shape is the payoff for
        # routing those writes through set_shape, which is separate work.
        self.shape: Optional[Union[Shape, list[Any]]]
        SimTensor.set_shape(self, cfg.get('shape'))   # Other classes that subclass might override set_shape

    def set_module(self, m): self.link_module = m

    def memory_config(self):
        return getattr(self, '_memory_config', None)

    def __str__(self) -> str:
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

    def rank(self) -> int: return len(self.shape) if self.shape is not None else 0

    # Note: data count may not be a simple product of shape dims - may need to provide a custom func
    def nelems(self) -> int:
        if self.shape is None:
            return 0
        trank = self.rank()
        if trank > 0:
            # self.shape can be assigned directly, bypassing Shape's
            # normalisation, so each dim is normalised here before multiplying.
            # Coercing the result instead would be too late: numpy dims overflow
            # during the reduction, before any cast could run.
            #
            # Every numpy integer width is converted here, not just the np.int64
            # that _normalise_dim accepts. That helper is deliberately narrow
            # because widening it would change which shapes check_shape calls
            # valid; nelems decides no such thing, and a narrow conversion here
            # leaves the declared `-> int` false. A np.int32 shape of
            # [2**20, 2**12] otherwise reduces to np.int32(0) with an overflow
            # warning, and feeds a zero-element, zero-byte op to the cost model.
            res = functools.reduce(
                operator.mul,
                (int(k) if isinstance(k, np.integer) else k for k in self.shape),
                1,
            )
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

    def numel(self) -> int:
        return self.nelems()

    def set_shape(self, newshape: Optional[Union['Shape', Sequence[DimLike]]]) -> None:
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
    def nbytes(self, itemprec: Optional[Union[np.dtype, str]] = None) -> int:
        def typesize(dtype):
            if isinstance(dtype, np.dtype):
                return dtype.itemsize
            elif isinstance(dtype, str):
                return get_bpe(get_sim_dtype(dtype))
            else:
                raise TypeError(f"Unsupported dtype type: {type(dtype)}")
        if itemprec is None:
            assert self.dtype is not None, f"SimTensor({self.name}) has no dtype to calculate nbytes"
            itemsize = typesize(self.dtype)
        else:
            itemsize = typesize(itemprec)
        return self.nelems() * itemsize #assumes np.dtype

    def check_shape(self) -> bool:
        if self.shape is None:
            return False
        elif all(isinstance(d, (int, np.int64)) for d in self.shape):
            return True
        else:
            return False

    def clone(self, clone_num: int) -> 'SimTensor':
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

    def clone_by_shape(self, /, data_maybe_missing: bool = True) -> 'SimTensor':
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
