#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import pytest

import numpy as np
from ttsim.ops.tensor import Shape, make_tensor
import ttsim.front.functional.op as F

@pytest.mark.unit
@pytest.mark.opunit
def test_check_shape_valid():
    """Test check_shape() with valid shapes"""
    test_cases = [
        ("1D tensor", [10]),
        ("2D tensor", [3, 4]),
        ("3D tensor", [2, 3, 4]),
        ("4D tensor", [1, 2, 3, 4]),
        ("5D tensor", [1, 1, 2, 3, 4]),
        ("Single element", [1]),
        ("Large dimensions", [1000, 2000]),
    ]

    for tmsg, shape in test_cases:
        tensor = F._from_shape('test_tensor', shape, np_dtype=np.float32)
        assert tensor.check_shape(), f"TEST {tmsg}: Valid shape {shape} should pass check_shape()"
        assert tensor.shape == shape, f"TEST {tmsg}: Shape mismatch"

@pytest.mark.unit
@pytest.mark.opunit
def test_check_shape_invalid():
    """Test check_shape() with invalid shapes"""
    test_cases = [
        ("None shape", None),
        ("Shape with None element", [1, None, 3]),
        ("Shape with string", [1, "invalid", 3]),
        ("Shape with float", [1.5, 2, 3]),
        ("Shape with list", [[1, 2], 3]),
    ]

    for tmsg, shape in test_cases:
        tensor = make_tensor('test_tensor')
        tensor.shape = shape
        assert not tensor.check_shape(), f"TEST {tmsg}: Invalid shape {shape} should fail check_shape()"

@pytest.mark.unit
@pytest.mark.opunit
def test_check_shape_edge_cases():
    """Test check_shape() with edge cases"""
    # Empty shape (scalar tensor)
    tensor = make_tensor('scalar')
    tensor.shape = []
    assert tensor.check_shape(), "Empty shape [] should be valid (scalar tensor)"

    # Zero-sized dimensions
    tensor = F._from_shape('zero_dim', [0], np_dtype=np.float32)
    assert tensor.check_shape(), "Shape with zero dimension [0] should be valid"

    tensor = F._from_shape('zero_dims', [1, 0, 3], np_dtype=np.float32)
    assert tensor.check_shape(), "Shape with zero dimensions [1, 0, 3] should be valid"

    # np.int64 type (should be valid)
    tensor = make_tensor('int64_tensor')
    tensor.shape = [np.int64(1), np.int64(2), np.int64(3)]
    assert tensor.check_shape(), "Shape with np.int64 elements should be valid"

    # Mixed int and np.int64
    tensor = make_tensor('mixed_tensor')
    tensor.shape = [1, np.int64(2), 3]
    assert tensor.check_shape(), "Shape with mixed int and np.int64 should be valid"

@pytest.mark.unit
@pytest.mark.opunit
def test_shape_indexing():
    """Test shape indexing operations"""
    shape = [2, 3, 4, 5]
    tensor = F._from_shape('test', shape, np_dtype=np.float32)

    # Test individual element access
    assert tensor.shape[0] == 2, "First dimension should be 2"
    assert tensor.shape[1] == 3, "Second dimension should be 3"
    assert tensor.shape[-1] == 5, "Last dimension should be 5"
    assert tensor.shape[-2] == 4, "Second to last dimension should be 4"

    # Test rank
    assert tensor.rank() == len(shape), "Rank should match shape length"
    assert len(tensor.shape) == 4, "Shape length should be 4"

@pytest.mark.unit
@pytest.mark.opunit
def test_shape_slicing():
    """Test shape slicing operations"""
    shape = [2, 3, 4, 5]
    tensor = F._from_shape('test', shape, np_dtype=np.float32)

    # Test full slice
    assert tensor.shape[:] == shape, "Full slice should return entire shape"

    # Test prefix slice
    assert tensor.shape[:2] == [2, 3], "Prefix slice [:2] should return [2, 3]"
    assert tensor.shape[:3] == [2, 3, 4], "Prefix slice [:3] should return [2, 3, 4]"

    # Test suffix slice
    assert tensor.shape[1:] == [3, 4, 5], "Suffix slice [1:] should return [3, 4, 5]"
    assert tensor.shape[2:] == [4, 5], "Suffix slice [2:] should return [4, 5]"

    # Test middle slice
    assert tensor.shape[1:3] == [3, 4], "Middle slice [1:3] should return [3, 4]"

    # Test negative indices
    assert tensor.shape[:-1] == [2, 3, 4], "Slice [:-1] should return [2, 3, 4]"
    assert tensor.shape[-2:] == [4, 5], "Slice [-2:] should return [4, 5]"
    assert tensor.shape[-3:-1] == [3, 4], "Slice [-3:-1] should return [3, 4]"

@pytest.mark.unit
@pytest.mark.opunit
def test_shape_comparison():
    """Test shape comparison operations"""
    shape1 = [2, 3, 4]
    shape2 = [2, 3, 4]
    shape3 = [2, 3, 5]

    tensor1 = F._from_shape('t1', shape1, np_dtype=np.float32)
    tensor2 = F._from_shape('t2', shape2, np_dtype=np.float32)
    tensor3 = F._from_shape('t3', shape3, np_dtype=np.float32)

    # Test equality
    assert tensor1.shape == tensor2.shape, "Identical shapes should be equal"
    assert tensor1.shape != tensor3.shape, "Different shapes should not be equal"

    # Test element-wise comparison
    assert tensor1.shape[0] == tensor2.shape[0], "First dimensions should be equal"
    assert tensor1.shape[0] == tensor3.shape[0], "First dimensions should be equal"
    assert tensor1.shape[2] != tensor3.shape[2], "Last dimensions should differ"

    # Test with different ranks
    tensor4 = F._from_shape('t4', [2, 3], np_dtype=np.float32)
    assert tensor1.shape != tensor4.shape, "Shapes with different ranks should not be equal"

@pytest.mark.unit
@pytest.mark.opunit
def test_shape_concatenation():
    """Test shape concatenation operations"""
    # Test simple concatenation
    shape1 = [2, 3]
    shape2 = [4, 5]
    result = shape1 + shape2
    assert result == [2, 3, 4, 5], "Shape concatenation should work"

    # Test with unpacking
    shape1 = [2, 3]
    shape2 = [4]
    result = [*shape1, *shape2]
    assert result == [2, 3, 4], "Shape concatenation with unpacking should work"

    # Test inserting shape in middle (as in Gather)
    data_shape = [2, 3, 4]
    indices_shape = [5]
    axis = 1
    result = data_shape[:axis] + indices_shape + data_shape[axis + 1:]
    assert result == [2, 5, 4], "Shape insertion should work: [2,3,4] with [5] at axis 1 = [2,5,4]"

    # Test prefix/suffix extraction (as in Flatten)
    shape = [2, 3, 4, 5]
    axis = 2
    prefix = shape[:axis]
    suffix = shape[axis:]
    assert prefix == [2, 3], "Prefix extraction should work"
    assert suffix == [4, 5], "Suffix extraction should work"

@pytest.mark.unit
@pytest.mark.opunit
def test_shape_filtering():
    """Test shape filtering operations"""
    # Test filtering by index (as in Squeeze)
    shape = [1, 3, 1, 4]
    indices_to_remove = [0, 2]
    result = [dim for i, dim in enumerate(shape) if i not in indices_to_remove]
    assert result == [3, 4], "Shape filtering by index should work"

    # Test filtering by condition (as in reduction ops)
    shape = [2, 3, 4]
    dim_to_remove = 1
    result = [s for i, s in enumerate(shape) if i != dim_to_remove]
    assert result == [2, 4], "Shape filtering by condition should work"

    # Test filtering by value
    shape = [1, 3, 1, 4, 1]
    result = [d for d in shape if d > 1]
    assert result == [3, 4], "Shape filtering by value should work"

@pytest.mark.unit
@pytest.mark.opunit
def test_shape_padding():
    """Test shape padding operations"""
    # Test prepending dimensions (as in Expand)
    shape = [3]
    n = 2
    result = [1] * n + shape
    assert result == [1, 1, 3], "Shape prepending should work"

    # Test appending dimensions
    shape = [3]
    n = 2
    result = shape + [1] * n
    assert result == [3, 1, 1], "Shape appending should work"

    # Test padding for broadcasting (reverse and pad)
    shape = [3, 4]
    max_len = 4
    reversed_shape = list(shape[::-1])
    padded = reversed_shape + [1] * (max_len - len(shape))
    assert padded == [4, 3, 1, 1], "Shape padding for broadcasting should work"

@pytest.mark.unit
@pytest.mark.opunit
def test_shape_reversal():
    """Test shape reversal operations"""
    shape = [2, 3, 4, 5]
    reversed_shape = shape[::-1]
    assert reversed_shape == [5, 4, 3, 2], "Shape reversal should work"

    # Test partial reversal
    shape = [2, 3, 4]
    reversed_suffix = shape[-1::-1]
    assert reversed_suffix == [4, 3, 2], "Partial shape reversal should work"

@pytest.mark.unit
@pytest.mark.opunit
def test_shape_iteration():
    """Test iterating over shape dimensions"""
    shape = [2, 3, 4]
    tensor = F._from_shape('test', shape, np_dtype=np.float32)

    # Test iteration
    dims = [d for d in tensor.shape]
    assert dims == shape, "Shape iteration should work"

    # Test enumerate
    indexed_dims = [(i, d) for i, d in enumerate(tensor.shape)]
    assert indexed_dims == [(0, 2), (1, 3), (2, 4)], "Shape enumeration should work"

    # Test with conditions
    large_dims = [d for d in tensor.shape if d > 2]
    assert large_dims == [3, 4], "Conditional shape iteration should work"

@pytest.mark.unit
@pytest.mark.opunit
def test_rank_and_nelems():
    """Test rank() and nelems() methods"""
    # Test rank
    tensor = F._from_shape('test', [2, 3, 4], np_dtype=np.float32)
    assert tensor.rank() == 3, "Rank should be 3 for [2, 3, 4]"
    assert len(tensor.shape) == tensor.rank(), "len(shape) should equal rank()"

    # Test scalar tensor (rank 0)
    tensor = make_tensor('scalar')
    tensor.shape = []
    assert tensor.rank() == 0, "Scalar tensor should have rank 0"

    # Test nelems
    tensor = F._from_shape('test', [2, 3, 4], np_dtype=np.float32)
    expected_nelems = 2 * 3 * 4
    assert tensor.nelems() == expected_nelems, f"nelems() should be {expected_nelems} for [2, 3, 4]"

    # Test scalar nelems
    tensor = make_tensor('scalar')
    tensor.shape = []
    assert tensor.nelems() == 1, "Scalar tensor should have nelems() == 1"

    # Test zero-sized dimension
    tensor = F._from_shape('zero', [2, 0, 4], np_dtype=np.float32)
    assert tensor.nelems() == 0, "Tensor with zero dimension should have nelems() == 0"


@pytest.mark.unit
@pytest.mark.opunit
def test_shape_normalises_accepted_numpy_dims():
    """Shape stores plain ints, so every dimension query returns int."""
    shape = Shape([np.int64(2), 3])
    assert all(type(d) is int for d in shape), f"dims not normalised: {shape}"
    assert type(shape[0]) is int, "int index must yield int"
    assert all(type(d) is int for d in shape[1:]), "slice must yield ints"
    assert all(type(d) is int for d in shape.as_list()), "as_list must yield ints"
    assert type(shape.volume()) is int, "volume must be int"


@pytest.mark.unit
@pytest.mark.opunit
def test_shape_normalises_on_mutation():
    """Assignment normalises too, or the int contracts break after a write."""
    shape = Shape([1, 2, 3])
    shape[0] = np.int64(9)
    assert type(shape[0]) is int, "int assignment must normalise"
    shape[1:3] = [np.int64(7), np.int64(8)]
    assert all(type(d) is int for d in shape), f"slice assignment must normalise: {shape}"
    assert type(shape.volume()) is int, "volume must stay int after mutation"


@pytest.mark.unit
@pytest.mark.opunit
def test_check_shape_still_rejects_unaccepted_dim_types():
    """Normalising must not widen which shapes count as valid.

    check_shape accepted `int` and `np.int64` only. A np.int32 or float dim was
    rejected before dimension normalisation was introduced and must stay
    rejected: normalising only the accepted numpy type keeps that boundary.
    """
    accepted = make_tensor('accepted')
    accepted.set_shape([np.int64(1), 2])
    assert accepted.check_shape(), "int and np.int64 dims must remain valid"

    for bad in ([np.int32(1), 2], [2.9, 3]):
        tensor = make_tensor('rejected')
        # Passing a rejected dim type is the point of this test, so the
        # argument deliberately does not satisfy set_shape's signature.
        tensor.set_shape(bad)  # type: ignore[arg-type]
        assert not tensor.check_shape(), f"{[type(d).__name__ for d in bad]} must stay invalid"


@pytest.mark.unit
@pytest.mark.opunit
def test_shape_accepts_any_sequence():
    """The constructor accepts what its signature promises, not just list/tuple."""
    assert Shape(range(3)) == [0, 1, 2], "a range is a Sequence of dims"
    assert Shape((1, 2)) == [1, 2], "a tuple is a Sequence of dims"
    assert Shape(Shape([4, 5])) == [4, 5], "a Shape copies"
    for bad in ('abc', b'abc', 3):
        with pytest.raises(TypeError):
            Shape(bad)  # type: ignore[arg-type]


@pytest.mark.unit
@pytest.mark.opunit
def test_nelems_is_int_when_shape_bypasses_normalisation():
    """`SimTensor.shape` can be assigned directly, skipping Shape's normalisation.

    That state is valid -- check_shape accepts np.int64 dims -- so nelems() has
    to normalise during its reduction. Coercing the result would be too late:
    np.int64 dims overflow while multiplying, before any cast could run.
    """
    tensor = make_tensor('direct')
    tensor.shape = [np.int64(2), 3]          # type: ignore[assignment]
    assert tensor.check_shape(), "np.int64 dims are a valid shape"
    assert type(tensor.nelems()) is int, "nelems must be int even via direct assignment"
    assert tensor.nelems() == 6

    big = make_tensor('big')
    big.shape = [np.int64(2 ** 62), 4]       # type: ignore[assignment]
    assert big.nelems() == 2 ** 64, "the product must be exact, not wrapped"


@pytest.mark.unit
@pytest.mark.opunit
def test_nelems_normalises_every_numpy_integer_width():
    """nelems() must return an exact int for any numpy integer dtype.

    Shape's own coercion is restricted to np.int64, because widening it would
    change which shapes check_shape calls valid. nelems decides no such thing,
    so it normalises every width: a np.int32 shape otherwise reduces in
    fixed-width numpy arithmetic and wraps, feeding a zero-element, zero-byte
    op to the cost model.
    """
    tensor = make_tensor('int32_dims')
    tensor.shape = [np.int32(2 ** 20), np.int32(2 ** 12)]   # type: ignore[assignment]
    assert type(tensor.nelems()) is int, "nelems must be int for any numpy width"
    assert tensor.nelems() == 2 ** 32, "the product must be exact, not wrapped"

    tensor.dtype = np.dtype(np.float32)
    assert tensor.nbytes() == 2 ** 32 * 4, "nbytes follows nelems"


@pytest.mark.unit
@pytest.mark.opunit
def test_shim_tensor_queries_accept_a_raw_list_shape():
    """Shape-inference descriptors assign a plain list to SimTensor.shape.

    Tensor.size() and Tensor.expand() read that attribute, so they must not
    assume a Shape. Both raised AttributeError on a list before.
    """
    import ttsim.front.ttnn as ttnn
    from ttsim.front.ttnn.tensor import Tensor

    device = ttnn.open_device(device_id=0)
    tensor = Tensor(shape=[2, 3], dtype=ttnn.float32, device=device)
    tensor.shape = [2, 3]                                    # type: ignore[assignment]
    assert type(tensor.shape) is list, "precondition: a raw list shape"

    assert tensor.size() == (2, 3)
    assert tensor.expand(2, 3) is not None
