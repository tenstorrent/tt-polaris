#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import functools, operator
import math
import warnings
from collections.abc import Iterable, Iterator, Sequence
from typing import Any, Optional, Union, overload

import numpy as np
from loguru import logger
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


def _as_int_dim(d: Any) -> Any:
    """Convert any numpy integer dim to int, for exact arithmetic over dims.

    Wider than :func:`_normalise_dim` on purpose. That helper decides which
    shapes count as valid, so it accepts np.int64 only. Products such as
    :meth:`Shape.volume` and :meth:`SimTensor.nelems` decide no such thing, and
    a narrow conversion leaves their declared ``-> int`` false: a np.int32 shape
    of [2**20, 2**12] otherwise reduces to np.int32(0) with an overflow warning.
    """
    return int(d) if isinstance(d, np.integer) else d


# Bytes per 32x32 tile for TT block-float formats: mantissas + one shared exponent byte per 16 values.
# tt-metal tt_backend_api_types.hpp: Bfp8_b = 256*4 + 16*4, Bfp4_b = 128*4 + 16*4.
_BLOCK_FLOAT_TILE_BYTES = {'BFLOAT8_B': 1088, 'BFLOAT4_B': 576}
_TILE = 32


def _tile_exact_nbytes(shape: list, tile_bytes: int) -> int:
    """Bytes of a tiled tensor: last two dims rounded up to whole 32x32 tiles."""
    dims = [int(d) for d in shape] or [1]
    if len(dims) == 1:
        dims = [1] + dims
    lead = functools.reduce(operator.mul, dims[:-2], 1)
    return lead * math.ceil(dims[-2] / _TILE) * math.ceil(dims[-1] / _TILE) * tile_bytes


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
            # _shape can hold numpy widths other than np.int64, which
            # _normalise_dim leaves as they are; see _as_int_dim.
            result *= _as_int_dim(dim)
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
    folded into rows, channels last).  This hw_shape drives LUT key construction,
    profiler-shape comparison and byte accounting (SimTensor._storage_shape_list);
    the logical NCHW shape is kept in tensor.shape.

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
        # It also drives byte accounting: it is the stored shape nbytes() sizes the tensor on (see
        # _storage_shape_list), and is ignored, with a warning, if it holds fewer elements than shape --
        # except on halo outputs (_hw_shape_halo), where a strided halo is legitimately smaller.
        # Set by conv/pool sinfs; propagated by Move/ITS/STI/Reshard/Concat passthrough sinfs.
        # None for ops that produce non-NCHW outputs (attention projections, embeddings, etc.).
        # TODO: set hw_shape for MaxPool/AvgPool descriptors if LUT coverage is extended.
        self.hw_shape: list[int] | None = None
        # True when hw_shape is a halo output (set by halo_sinf, carried by _propagate_hw_shape). Its
        # hw_shape is the halo buffer: each core's shard plus the boundary sticks and padding the conv
        # window needs, so it can be larger than the logical tensor (zero-inserted transposed-conv input)
        # or smaller (a strided conv skips sticks), and byte accounting uses it at any size.
        # This is intended: an op that reads a halo output (Conv, ConvTranspose, MaxPool) reads that buffer
        # on hardware, and the capture records it as the op's input. Example, VGG-UNet WH's last
        # ConvTranspose: logical [1, 128, 128, 128] = 16384 sticks, halo buffer and captured input 82240
        # sticks. These buffers are L1-resident; the analytical model still charges them at the DRAM
        # rate, which is a separate L1-modelling gap.
        self._hw_shape_halo: bool = False
        # x_pad_logical: padded channel count for LUT key construction when this tensor feeds
        # a BLOCK_SHARDED conv2d/conv_transpose2d.  Set by the Device.execute_graph annotation
        # pass (post-shim, post-arch-load) via determine_block_sharded_channel_padding.
        # Written by ttsim/back/device.py's annotation pass; None when the tensor
        # doesn't participate in a BLOCK_SHARDED conv input chain; LUT key builders
        # then fall back to shape's X dim.
        self.x_pad_logical: int | None = None
        # y_pad_logical: halo-extended Y stick count override for LUT key construction.
        # Set by the Device.execute_graph annotation pass when an arch-specific halo
        # extension differs from the base _HALO_EXT_Y table (e.g. BH p100a's d2.conv1
        # picks num_cores_nhw=10 → y=5620 vs WH's y=5280).  Written by
        # ttsim/back/device.py::Device._annotate_halo_y_pad_logical.
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
            # Every numpy integer width is converted, not just np.int64 (see
            # _as_int_dim); otherwise a np.int32 shape wraps and feeds a
            # zero-element, zero-byte op to the cost model.
            res = functools.reduce(operator.mul, (_as_int_dim(k) for k in self.shape), 1)
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

    # Note: bytes are sized from the TT dtype and the stored shape: TILE layout pads to whole 32x32 tiles
    # and block-float formats use their tile sizes (see _storage_shape_list / _tt_dtype_nbytes).
    # Compression and sparsity are not modelled.
    def _storage_shape_list(self, storage_shape: Sequence[int] | None = None) -> list[int]:
        """Physical (stored) shape as a list. Precedence: explicit ``storage_shape`` > ``hw_shape``
        (conv-path NHWC-flattened / weight layout) > ``padded_shape()`` (ttnn tensors; carries explicit
        padding beyond tile rounding) > logical shape.

        ``hw_shape`` and ``padded_shape()`` are checked before use, because both now drive byte counts:
          * ``hw_shape`` must hold at least as many elements as the logical shape (it is a re-layout of the
            same data), unless it is a halo output (``_hw_shape_halo``): a strided halo keeps only the
            sticks the conv reads, so it can be smaller. Any other smaller one is stale or mis-propagated:
            it is logged once per tensor and the logical shape is used instead.
          * ``padded_shape()`` is used only while the tensor's shape still equals the shape it was computed
            for (``_padded_basis``, recorded wherever ``_padded_shape`` is assigned). A shape assigned
            directly afterwards makes it stale, and it is ignored.
        """
        if storage_shape is not None:
            return [int(d) for d in storage_shape]
        logical = [int(d) for d in require_shape_list(self.shape, 'storage shape: tensor shape must be set')]
        hw = getattr(self, 'hw_shape', None)
        if hw is not None:
            hw_list = [int(d) for d in hw]
            if (getattr(self, '_hw_shape_halo', False)
                    or functools.reduce(operator.mul, hw_list, 1) >= functools.reduce(operator.mul, logical, 1)):
                return hw_list
            if not getattr(self, '_hw_shape_mismatch_logged', False):
                self._hw_shape_mismatch_logged = True
                logger.warning(
                    "SimTensor({}): hw_shape {} holds fewer elements than shape {}; "
                    "sizing bytes on the logical shape", self.name, hw_list, logical)
        padded = getattr(self, 'padded_shape', None)
        if callable(padded):
            try:
                ps = padded()
            except (AttributeError, TypeError) as e:     # no padded shape on this tensor
                logger.debug("SimTensor({}): padded_shape() unavailable ({}); using logical shape", self.name, e)
                ps = None
            if ps is not None:
                ps_list = [int(d) for d in ps]
                basis = getattr(self, '_padded_basis', None)
                fresh = logical == basis if basis is not None else True
                if fresh and len(ps_list) == len(logical) and all(p >= l for p, l in zip(ps_list, logical)):
                    return ps_list
        return logical

    def _tt_dtype_nbytes(self, storage_shape: Sequence[int] | None = None) -> int | None:
        """Bytes from the tensor's own TT dtype (always on); None when the tensor has no TT dtype.

        dtype/layout precedence: ``_hw_dtype``/``_hw_layout`` (what the tensor is on HW, e.g. conv
        weights preprocessed to BFLOAT8_B TILE) over ``_ttnn_dtype``/``layout``. Tiled tensors of ANY
        dtype are padded to whole 32x32 tiles of the stored shape (see ``_storage_shape_list``);
        block-float formats use tt-metal tile sizes, other dtypes ``1024 * itemsize`` per tile.
        """
        if self.shape is None:
            return None
        hw_dtype = getattr(self, '_hw_dtype', None)
        tt_dtype = hw_dtype if hw_dtype is not None else getattr(self, '_ttnn_dtype', None)
        if tt_dtype is None:
            return None
        sshape = self._storage_shape_list(storage_shape)
        selems = functools.reduce(operator.mul, sshape, 1)
        layout = getattr(self, '_hw_layout', None) if hw_dtype is not None else None
        if layout is None:
            layout = getattr(self, 'layout', None)
        tiled = getattr(layout, 'name', '') == 'TILE_LAYOUT'
        if tt_dtype.name in _BLOCK_FLOAT_TILE_BYTES:
            tile_bytes = _BLOCK_FLOAT_TILE_BYTES[tt_dtype.name]
            # Tile padding applies only to tensors stored tiled. Row-major block-float tensors (e.g. conv
            # weights in OIHW) are re-laid-out before tiling on HW, so padding their logical last two dims
            # would overstate them; count exponent overhead only. int() floors the fractional exponent
            # bytes (under 1 B per tensor).
            if not tiled:
                return int(selems * tile_bytes / (_TILE * _TILE))
            return _tile_exact_nbytes(sshape, tile_bytes)
        if not tiled:
            return int(selems * tt_dtype.itemsize)
        return int(_tile_exact_nbytes(sshape, _TILE * _TILE * tt_dtype.itemsize))

    def storage_nbytes(self, storage_shape: Sequence[int] | None = None) -> int | None:
        """Dtype- and tile-aware bytes of this tensor stored as ``storage_shape`` (default: its own
        stored shape). ``None`` when the tensor has no TT dtype; callers
        then fall back to ``elements * element_size``."""
        return self._tt_dtype_nbytes(storage_shape)

    # Note: Caching nbytes for instance methods can cause memory leaks due to references held by lru_cache.
    # If caching is needed, consider using a static cache or external memoization.
    def nbytes(self, itemprec: Optional[Union[np.dtype, str]] = None) -> int:
        """Size of this tensor in bytes.

        Precedence:
          1. The tensor's own TT dtype (``_hw_dtype`` if set, else ``_ttnn_dtype``): bytes of its stored
             shape (see ``_storage_shape_list``). TILE tensors are padded to whole 32x32 tiles; block-float
             tiles are 1088 B (bfp8) / 576 B (bfp4); row-major block-float counts the exponent overhead.
          2. ``itemprec`` (descriptors pass ``op.precision``, i.e. the WL2Arch ``data_type_spec``):
             ``nelems() * itemsize``. It applies only to tensors with no TT dtype, so a ``data_type_spec``
             "what-if" precision does not change ttnn tensors that carry a TT dtype.
          3. The tensor's numpy ``dtype``: ``nelems() * itemsize``.
        """
        def typesize(dtype):
            if isinstance(dtype, np.dtype):
                return dtype.itemsize
            elif isinstance(dtype, str):
                return get_bpe(get_sim_dtype(dtype))
            else:
                raise TypeError(f"Unsupported dtype type: {type(dtype)}")
        # The tensor's own TT dtype wins over the caller's itemprec. Descriptors
        # pass op.precision, which the backend always sets (WorkloadGraph.set_precision), so gating on
        # itemprec is None would skip this for nearly every op.
        tt_nbytes = self._tt_dtype_nbytes()
        if tt_nbytes is not None:
            return tt_nbytes
        if itemprec is None:
            assert self.dtype is not None, f"SimTensor({self.name}) has no dtype to calculate nbytes"
            itemsize = typesize(self.dtype)
        else:
            itemsize = typesize(itemprec)
        return self.nelems() * itemsize

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
