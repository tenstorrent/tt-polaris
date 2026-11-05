#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TTNN Shim Functions for Simulation

This module provides drop-in replacements for ttnn operations that track
operations without actually executing them. These shims are designed for
simulation and performance modeling purposes.

All functions match the exact Python API of their ttnn counterparts and
track compute and memory operations that would be required.
"""

# Constants matching TTNN

from loguru import logger
import numpy as np

from ttsim.ops.op import SimOp

from .tensor import DataType, Layout, Shape
from .types import TILE_HEIGHT, TILE_WIDTH, TILE_HW
from .tensor import Tensor
from .op import _propagate_ttnn_dtype, _propagate_memory_config, generate_new_op_name, reshape as ttnn_reshape_simop

from .memory import MemoryConfig
from .buffer import BufferType, TensorMemoryLayout

DRAM_MEMORY_CONFIG = MemoryConfig.DRAM
L1_MEMORY_CONFIG = MemoryConfig.L1

# Execution mode control
class ExecutionMode:
    """Execution mode for shim functions."""
    TRACK_ONLY = "track_only"      # Only track operations, don't execute
    EXECUTE = "execute"             # Actually perform operations
    EXECUTE_AND_TRACK = "execute_and_track"  # Both execute and track

# Global execution mode (default: track only for backward compatibility)
_execution_mode = ExecutionMode.TRACK_ONLY

def set_execution_mode(mode):
    """Set global execution mode.
    Args:
        mode: ExecutionMode.TRACK_ONLY, ExecutionMode.EXECUTE, or ExecutionMode.EXECUTE_AND_TRACK
    """
    global _execution_mode
    if mode not in [ExecutionMode.TRACK_ONLY, ExecutionMode.EXECUTE, ExecutionMode.EXECUTE_AND_TRACK]:
        raise ValueError(f"Invalid execution mode: {mode}")
    _execution_mode = mode

def get_execution_mode():
    """Get current global execution mode."""
    return _execution_mode

# Operation tracking
class OperationTracker:
    def __init__(self):
        self.reset()
    def reset(self):
        self.tilize_count = 0
        self.untilize_count = 0
        self.tilize_with_val_padding_count = 0
        self.untilize_with_unpadding_count = 0
        self.to_layout_count = 0
        self.reshape_count = 0
        self.pad_count = 0
        self.memory_operations = []
        # Detailed internal operation tracking
        self.compute_operations = []  # Individual compute operations
        self.memory_reads = []  # Memory read operations
        self.memory_writes = []  # Memory write operations
        self.circular_buffer_ops = []  # Circular buffer operations
        self.kernel_invocations = []  # Kernel execution tracking
        self.data_movements = []  # Data movement between buffers

    def track_tilize(self, tensor_shape, num_tiles, num_tiles_per_row=None, num_tiles_per_col=None,
                     use_multicore=True, element_size=2):
        """Track tilize operation with internal operation details."""
        self.tilize_count += 1

        # Calculate internal operations
        if num_tiles_per_row is None or num_tiles_per_col is None:
            # Estimate from shape
            if len(tensor_shape) >= 2:
                width = tensor_shape[-1]
                height = tensor_shape[-2] if len(tensor_shape) >= 2 else 1
                num_tiles_per_row = (width + TILE_WIDTH - 1) // TILE_WIDTH
                num_tiles_per_col = (height + TILE_HEIGHT - 1) // TILE_HEIGHT
            else:
                num_tiles_per_row = 1
                num_tiles_per_col = 1
        # Calculate memory operations
        tile_size_bytes = TILE_HW * element_size
        total_input_bytes = num_tiles * tile_size_bytes  # Input in row-major
        total_output_bytes = num_tiles * tile_size_bytes  # Output in tile layout
        # Track memory reads (reading row-major input)
        self.memory_reads.append({
            'op': 'tilize_read',
            'bytes': total_input_bytes,
            'num_tiles': num_tiles,
            'tiles_per_row': num_tiles_per_row,
            'tiles_per_col': num_tiles_per_col
        })
        # Track memory writes (writing tile layout output)
        self.memory_writes.append({
            'op': 'tilize_write',
            'bytes': total_output_bytes,
            'num_tiles': num_tiles,
            'tiles_per_row': num_tiles_per_row,
            'tiles_per_col': num_tiles_per_col
        })
        # Track compute operations (one per tile)
        compute_ops = num_tiles
        self.compute_operations.append({
            'op': 'tilize_compute',
            'count': compute_ops,
            'type': 'datacopy_tilize',
            'num_tiles': num_tiles,
            'use_multicore': use_multicore
        })
        # Track circular buffer operations
        # Tilize typically uses: input CB -> unpack -> compute -> pack -> output CB
        self.circular_buffer_ops.append({
            'op': 'tilize_cb',
            'input_cb_ops': num_tiles,  # Reads from input CB
            'output_cb_ops': num_tiles,  # Writes to output CB
            'num_tiles': num_tiles
        })
        # Track kernel invocations
        if use_multicore:
            # Multicore: one kernel per core, estimate cores from tile distribution
            estimated_cores = min(num_tiles, 128)  # Reasonable upper bound
        else:
            estimated_cores = 1
        self.kernel_invocations.append({
            'op': 'tilize_kernel',
            'count': estimated_cores,
            'num_tiles_per_core': num_tiles // max(estimated_cores, 1),
            'use_multicore': use_multicore
        })
        # Track data movement
        self.data_movements.append({
            'op': 'tilize_movement',
            'from': 'row_major_buffer',
            'to': 'tile_layout_buffer',
            'bytes': total_input_bytes,
            'num_tiles': num_tiles
        })
        # High-level operation tracking
        self.memory_operations.append({
            'op': 'tilize',
            'shape': tensor_shape,
            'num_tiles': num_tiles,
            'num_tiles_per_row': num_tiles_per_row,
            'num_tiles_per_col': num_tiles_per_col,
            'total_bytes_read': total_input_bytes,
            'total_bytes_written': total_output_bytes,
            'compute_operations': compute_ops,
            'kernel_invocations': estimated_cores
        })
    def track_untilize(self, tensor_shape, num_tiles, num_tiles_per_row=None, num_tiles_per_col=None,
                       use_multicore=True, use_pack_untilize=True, element_size=2):
        """Track untilize operation with internal operation details."""
        self.untilize_count += 1
        # Calculate internal operations
        if num_tiles_per_row is None or num_tiles_per_col is None:
            # Estimate from shape
            if len(tensor_shape) >= 2:
                width = tensor_shape[-1]
                height = tensor_shape[-2] if len(tensor_shape) >= 2 else 1
                num_tiles_per_row = (width + TILE_WIDTH - 1) // TILE_WIDTH
                num_tiles_per_col = (height + TILE_HEIGHT - 1) // TILE_HEIGHT
            else:
                num_tiles_per_row = 1
                num_tiles_per_col = 1
        # Calculate memory operations
        tile_size_bytes = TILE_HW * element_size
        total_input_bytes = num_tiles * tile_size_bytes  # Input in tile layout
        total_output_bytes = num_tiles * tile_size_bytes  # Output in row-major
        # Track memory reads (reading tile layout input)
        self.memory_reads.append({
            'op': 'untilize_read',
            'bytes': total_input_bytes,
            'num_tiles': num_tiles,
            'tiles_per_row': num_tiles_per_row,
            'tiles_per_col': num_tiles_per_col
        })
        # Track memory writes (writing row-major output)
        self.memory_writes.append({
            'op': 'untilize_write',
            'bytes': total_output_bytes,
            'num_tiles': num_tiles,
            'tiles_per_row': num_tiles_per_row,
            'tiles_per_col': num_tiles_per_col
        })
        # Track compute operations
        # Untilize: unpack -> compute (datacopy) -> pack
        compute_ops = num_tiles
        self.compute_operations.append({
            'op': 'untilize_compute',
            'count': compute_ops,
            'type': 'datacopy_untilize',
            'num_tiles': num_tiles,
            'use_pack_untilize': use_pack_untilize,
            'use_multicore': use_multicore
        })
        # Track circular buffer operations
        self.circular_buffer_ops.append({
            'op': 'untilize_cb',
            'input_cb_ops': num_tiles,  # Reads from input CB (tile layout)
            'output_cb_ops': num_tiles,  # Writes to output CB (row-major)
            'num_tiles': num_tiles,
            'use_pack_untilize': use_pack_untilize
        })
        # Track kernel invocations
        if use_multicore:
            estimated_cores = min(num_tiles, 128)
        else:
            estimated_cores = 1
        self.kernel_invocations.append({
            'op': 'untilize_kernel',
            'count': estimated_cores,
            'num_tiles_per_core': num_tiles // max(estimated_cores, 1),
            'use_multicore': use_multicore,
            'use_pack_untilize': use_pack_untilize
        })
        # Track data movement
        self.data_movements.append({
            'op': 'untilize_movement',
            'from': 'tile_layout_buffer',
            'to': 'row_major_buffer',
            'bytes': total_input_bytes,
            'num_tiles': num_tiles
        })
        # High-level operation tracking
        self.memory_operations.append({
            'op': 'untilize',
            'shape': tensor_shape,
            'num_tiles': num_tiles,
            'num_tiles_per_row': num_tiles_per_row,
            'num_tiles_per_col': num_tiles_per_col,
            'total_bytes_read': total_input_bytes,
            'total_bytes_written': total_output_bytes,
            'compute_operations': compute_ops,
            'kernel_invocations': estimated_cores,
            'use_pack_untilize': use_pack_untilize
        })
    def track_tilize_with_val_padding(self, input_shape, output_shape, num_tiles):
        self.tilize_with_val_padding_count += 1
        self.memory_operations.append({
            'op': 'tilize_with_val_padding',
            'input_shape': input_shape,
            'output_shape': output_shape,
            'num_tiles': num_tiles
        })
    def track_untilize_with_unpadding(self, input_shape, output_shape, num_tiles):
        self.untilize_with_unpadding_count += 1
        self.memory_operations.append({
            'op': 'untilize_with_unpadding',
            'input_shape': input_shape,
            'output_shape': output_shape,
            'num_tiles': num_tiles
        })
    def track_to_layout(self, from_layout, to_layout):
        self.to_layout_count += 1
        self.memory_operations.append({
            'op': 'to_layout',
            'from_layout': from_layout,
            'to_layout': to_layout
        })
    def track_reshape(self, from_shape, to_shape):
        self.reshape_count += 1
        self.memory_operations.append({
            'op': 'reshape',
            'from_shape': from_shape,
            'to_shape': to_shape
        })
    def track_pad(self, from_shape, to_shape):
        self.pad_count += 1
        self.memory_operations.append({
            'op': 'pad',
            'from_shape': from_shape,
            'to_shape': to_shape
        })

    def track_interleaved_to_sharded(self, input_shape, element_size=2):
        total_bytes = 1
        for d in input_shape:
            total_bytes *= d
        total_bytes *= element_size
        self.memory_reads.append({'op': 'interleaved_to_sharded_read', 'bytes': total_bytes})
        self.memory_writes.append({'op': 'interleaved_to_sharded_write', 'bytes': total_bytes})
        self.data_movements.append({
            'op': 'interleaved_to_sharded_movement',
            'from': 'interleaved', 'to': 'sharded', 'bytes': total_bytes,
        })
        self.memory_operations.append({
            'op': 'interleaved_to_sharded',
            'shape': list(input_shape),
            'total_bytes_read': total_bytes,
            'total_bytes_written': total_bytes,
        })

    def track_sharded_to_interleaved(self, input_shape, element_size=2):
        total_bytes = 1
        for d in input_shape:
            total_bytes *= d
        total_bytes *= element_size
        self.memory_reads.append({'op': 'sharded_to_interleaved_read', 'bytes': total_bytes})
        self.memory_writes.append({'op': 'sharded_to_interleaved_write', 'bytes': total_bytes})
        self.data_movements.append({
            'op': 'sharded_to_interleaved_movement',
            'from': 'sharded', 'to': 'interleaved', 'bytes': total_bytes,
        })
        self.memory_operations.append({
            'op': 'sharded_to_interleaved',
            'shape': list(input_shape),
            'total_bytes_read': total_bytes,
            'total_bytes_written': total_bytes,
        })

    def track_reshard(self, input_shape, element_size=2):
        total_bytes = 1
        for d in input_shape:
            total_bytes *= d
        total_bytes *= element_size
        self.memory_reads.append({'op': 'reshard_read', 'bytes': total_bytes})
        self.memory_writes.append({'op': 'reshard_write', 'bytes': total_bytes})
        self.data_movements.append({
            'op': 'reshard_movement',
            'from': 'sharded', 'to': 'sharded', 'bytes': total_bytes,
        })
        self.memory_operations.append({
            'op': 'reshard',
            'shape': list(input_shape),
            'total_bytes_read': total_bytes,
            'total_bytes_written': total_bytes,
        })

    def track_nlp_create_qkv_heads(self, input_shape, q_shape, k_shape, v_shape, element_size=2):
        in_bytes = 1
        for d in input_shape:
            in_bytes *= d
        in_bytes *= element_size
        out_bytes = 0
        for s in (q_shape, k_shape, v_shape):
            b = element_size
            for d in s:
                b *= d
            out_bytes += b
        self.memory_reads.append({'op': 'nlp_create_qkv_heads_read', 'bytes': in_bytes})
        self.memory_writes.append({'op': 'nlp_create_qkv_heads_write', 'bytes': out_bytes})
        self.data_movements.append({
            'op': 'nlp_create_qkv_heads_movement',
            'from': 'fused_qkv', 'to': 'split_qkv', 'bytes': out_bytes,
        })
        self.memory_operations.append({
            'op': 'nlp_create_qkv_heads',
            'input_shape': list(input_shape),
            'q_shape': list(q_shape),
            'k_shape': list(k_shape),
            'v_shape': list(v_shape),
            'total_bytes_read': in_bytes,
            'total_bytes_written': out_bytes,
        })

    def track_nlp_concat_heads(self, input_shape, output_shape, element_size=2):
        total_bytes = 1
        for d in input_shape:
            total_bytes *= d
        total_bytes *= element_size
        self.memory_reads.append({'op': 'nlp_concat_heads_read', 'bytes': total_bytes})
        self.memory_writes.append({'op': 'nlp_concat_heads_write', 'bytes': total_bytes})
        self.data_movements.append({
            'op': 'nlp_concat_heads_movement',
            'from': 'split_heads', 'to': 'concat_heads', 'bytes': total_bytes,
        })
        self.memory_operations.append({
            'op': 'nlp_concat_heads',
            'input_shape': list(input_shape),
            'output_shape': list(output_shape),
            'total_bytes_read': total_bytes,
            'total_bytes_written': total_bytes,
        })

    def get_summary(self):
        """Get summary of all tracked operations."""
        total_memory_read_bytes = sum(op.get('bytes', 0) for op in self.memory_reads)
        total_memory_write_bytes = sum(op.get('bytes', 0) for op in self.memory_writes)
        total_compute_ops = sum(op.get('count', 0) for op in self.compute_operations)
        total_kernel_invocations = sum(op.get('count', 0) for op in self.kernel_invocations)
        total_cb_ops = sum(op.get('input_cb_ops', 0) + op.get('output_cb_ops', 0) for op in self.circular_buffer_ops)
        return {
            'tilize_count': self.tilize_count,
            'untilize_count': self.untilize_count,
            'tilize_with_val_padding_count': self.tilize_with_val_padding_count,
            'untilize_with_unpadding_count': self.untilize_with_unpadding_count,
            'to_layout_count': self.to_layout_count,
            'reshape_count': self.reshape_count,
            'pad_count': self.pad_count,
            'total_operations': len(self.memory_operations),
            # Internal operation tracking
            'total_memory_read_bytes': total_memory_read_bytes,
            'total_memory_write_bytes': total_memory_write_bytes,
            'total_compute_operations': total_compute_ops,
            'total_kernel_invocations': total_kernel_invocations,
            'total_circular_buffer_ops': total_cb_ops,
            'memory_read_count': len(self.memory_reads),
            'memory_write_count': len(self.memory_writes),
            'compute_operation_count': len(self.compute_operations),
            'kernel_invocation_count': len(self.kernel_invocations),
            'circular_buffer_op_count': len(self.circular_buffer_ops),
            'data_movement_count': len(self.data_movements)
        }
    def get_detailed_summary(self):
        """Get detailed summary with all internal operations."""
        return {
            'high_level_summary': self.get_summary(),
            'memory_reads': self.memory_reads,
            'memory_writes': self.memory_writes,
            'compute_operations': self.compute_operations,
            'kernel_invocations': self.kernel_invocations,
            'circular_buffer_ops': self.circular_buffer_ops,
            'data_movements': self.data_movements,
            'all_operations': self.memory_operations
        }

# Global operation tracker
_tracker = OperationTracker()

def get_tracker():
    """Get the global operation tracker."""
    return _tracker

def reset_tracker():
    """Reset the global operation tracker."""
    _tracker.reset()

TensorProxy = Tensor

def round_up(value, multiple):
    """Round up value to nearest multiple."""
    return ((value + multiple - 1) // multiple) * multiple


def pad_to_tile_shape(shape):
    """Pad shape to tile boundaries."""
    if len(shape) < 2:
        return Shape(shape)
    padded = list(shape)
    padded[-2] = round_up(padded[-2], TILE_HEIGHT)
    padded[-1] = round_up(padded[-1], TILE_WIDTH)
    return Shape(padded)


# ============================================================================
# Pure Python Layout Conversion Algorithms (for execute mode)
# ============================================================================

def _get_dtype_python_type(dtype):
    """Convert DataType to Python type."""
    dtype_map = {
        DataType.BFLOAT16: float,
        DataType.FLOAT32: float,
        DataType.UINT32: int,
        DataType.INT32: int,
        DataType.UINT16: int,
    }
    return dtype_map.get(dtype, float)


def _create_empty_data(shape, dtype, fill_value=0.0):
    """Create empty data array with given shape and dtype."""
    python_type = _get_dtype_python_type(dtype)
    total_elements = 1
    for dim in shape:
        total_elements *= dim
    return [python_type(fill_value) for _ in range(total_elements)]


def _flatten_shape(shape):
    """Flatten shape to 2D (H, W) for layout conversion."""
    if len(shape) == 0:
        return (1, 1)
    elif len(shape) == 1:
        return (1, shape[0])
    elif len(shape) == 2:
        return tuple(shape)
    else:
        # Flatten all leading dimensions
        H = 1
        for i in range(len(shape) - 2):
            H *= shape[i]
        H *= shape[-2]
        W = shape[-1]
        return (H, W)


def _unflatten_data(data, flat_shape, original_shape):
    """Reshape flattened data back to original shape."""
    if len(original_shape) <= 2:
        return data
    # Calculate total elements
    total_elements = 1
    for dim in original_shape:
        total_elements *= dim

    # If shapes match, return as-is
    if len(data) == total_elements:
        return data

    # Otherwise, we need to reshape
    # For simplicity, just return the data (actual reshaping would be more complex)
    return data[:total_elements]


def _row_major_to_tile_layout_swizzled(data, H, W):
    """
    Convert row-major data to tile layout (swizzled format).
    This implements a simplified version of the tilize operation:
    - Divides data into 32x32 tiles
    - Each tile is stored in swizzled format (4 faces of 16x16)
    - Face order: face0, face1, face2, face3 (row-major within tile)
    Args:
        data: List of values in row-major order
        H: Height dimension
        W: Width dimension
    Returns:
        List of values in tile layout
    """
    if len(data) != H * W:
        raise ValueError(f"Data size {len(data)} doesn't match shape {H}x{W}")

    num_tiles_h = (H + TILE_HEIGHT - 1) // TILE_HEIGHT
    num_tiles_w = (W + TILE_WIDTH - 1) // TILE_WIDTH
    output = []

    for tile_h in range(num_tiles_h):
        for tile_w in range(num_tiles_w):
            # Process one 32x32 tile
            for face_row in range(2):  # 2 rows of faces
                for face_col in range(2):  # 2 cols of faces
                    # Process one 16x16 face
                    for row_in_face in range(16):
                        for col_in_face in range(16):
                            # Calculate position in original row-major data
                            global_row = tile_h * TILE_HEIGHT + face_row * 16 + row_in_face
                            global_col = tile_w * TILE_WIDTH + face_col * 16 + col_in_face
                            if global_row < H and global_col < W:
                                idx = global_row * W + global_col
                                output.append(data[idx])
                            else:
                                # Padding - use 0
                                output.append(0.0)
    return output


def _tile_layout_swizzled_to_row_major(data, H, W):
    """
    Convert tile layout (swizzled format) back to row-major.

    This is the inverse of _row_major_to_tile_layout_swizzled.

    Args:
        data: List of values in tile layout
        H: Height dimension (logical)
        W: Width dimension (logical)
    Returns:
        List of values in row-major order
    """
    num_tiles_h = (H + TILE_HEIGHT - 1) // TILE_HEIGHT
    num_tiles_w = (W + TILE_WIDTH - 1) // TILE_WIDTH
    output = [0.0] * (H * W)
    data_idx = 0
    for tile_h in range(num_tiles_h):
        for tile_w in range(num_tiles_w):
            # Process one 32x32 tile
            for face_row in range(2):  # 2 rows of faces
                for face_col in range(2):  # 2 cols of faces
                    # Process one 16x16 face
                    for row_in_face in range(16):
                        for col_in_face in range(16):
                            # Calculate position in output row-major data
                            global_row = tile_h * TILE_HEIGHT + face_row * 16 + row_in_face
                            global_col = tile_w * TILE_WIDTH + face_col * 16 + col_in_face
                            if global_row < H and global_col < W and data_idx < len(data):
                                idx = global_row * W + global_col
                                output[idx] = data[data_idx]
                            data_idx += 1
    return output


def _perform_tilize_operation(input_data, input_shape, output_padded_shape, pad_value=0.0):
    """
    Perform actual tilize operation on data.
    Args:
        input_data: Input data in row-major format
        input_shape: Logical input shape
        output_padded_shape: Padded output shape
        pad_value: Value to use for padding
    Returns:
        Output data in tile layout format
    """
    # Flatten to 2D for processing
    H_in, W_in = _flatten_shape(input_shape)
    H_out, W_out = _flatten_shape(output_padded_shape)
    # Pad input data to output shape if needed
    if H_in < H_out or W_in < W_out:
        padded_data = _create_empty_data([H_out, W_out], DataType.FLOAT32, pad_value)
        # Copy input data
        for h in range(H_in):
            for w in range(W_in):
                idx_in = h * W_in + w
                idx_out = h * W_out + w
                if idx_in < len(input_data):
                    padded_data[idx_out] = input_data[idx_in]
        input_data = padded_data
        H_in, W_in = H_out, W_out

    # Convert to tile layout
    tiled_data = _row_major_to_tile_layout_swizzled(input_data, H_in, W_in)

    return tiled_data


def _perform_untilize_operation(input_data, input_padded_shape, output_shape):
    """
    Perform actual untilize operation on data.
    Args:
        input_data: Input data in tile layout format
        input_padded_shape: Padded input shape
        output_shape: Logical output shape (unpadded)
    Returns:
        Output data in row-major format
    """
    # Flatten to 2D for processing
    H_in, W_in = _flatten_shape(input_padded_shape)
    H_out, W_out = _flatten_shape(output_shape)

    # Convert from tile layout to row-major
    row_major_data = _tile_layout_swizzled_to_row_major(input_data, H_in, W_in)

    # Extract only the logical shape (unpad)
    if H_out < H_in or W_out < W_in:
        output_data = []
        for h in range(H_out):
            for w in range(W_out):
                idx = h * W_in + w
                if idx < len(row_major_data):
                    output_data.append(row_major_data[idx])
        return output_data
    return row_major_data


def requires_padding_change(tensor, layout):
    """Check if padding change is required for layout conversion."""
    if layout == Layout.ROW_MAJOR_LAYOUT:
        # For row major, there shouldn't be extra padding
        logger.debug(
            "requires_padding_change: row major, logical_shape {} and padded_shape {}, padding change is {}",
            tensor.logical_shape()._shape,
            tensor.padded_shape()._shape,
            tensor.logical_shape()._shape != tensor.padded_shape()._shape,
        )
        return tensor.logical_shape()._shape != tensor.padded_shape()._shape
    else:
        # For tile layout, check if current padding matches tile requirements
        logger.debug(" type of tensor logical shape is {}", type(tensor.logical_shape()))
        tile_spec_padded = pad_to_tile_shape(tensor.logical_shape()._shape)
        return tensor.padded_shape()._shape != tile_spec_padded._shape


def squeeze_from_ND_to_4D(tensor, sub_core_grids=None):
    """Squeeze tensor from N dimensions to 4D."""
    shape = tensor.logical_shape()
    rank = shape.rank()

    if rank < 4:
        raise RuntimeError(f"Tensor has to be of rank >= 4! Instead is {rank}")

    if rank == 4:
        return tensor

    # Handle leading 1s
    if shape[0] == 1:
        squeezed = tensor
        i = 0
        while rank > 4 and shape[i] == 1:
            # Simulate squeeze by removing dimension
            new_shape = list(shape._shape)
            new_shape.pop(0)
            # There are few different Tensor types, and this tensor is created
            # using the same type as the input tensor.
            squeezed = type(tensor)(
                shape=Shape(new_shape),
                dtype=tensor.dtype,
                layout=tensor.get_layout(),
                memory_config=tensor.memory_config(),
                padded_shape=tensor.padded_shape()
            )
            rank = squeezed.logical_shape().rank()
            i += 1
            if rank <= 4:
                return squeezed
    # Reshape to 4D
    squeezed_shape = squeeze_shape_to_4D(shape)
    return reshape(squeezed, squeezed_shape, None, None, None, sub_core_grids)


def squeeze_shape_to_4D(shape):
    """Convert shape to 4D by collapsing leading dimensions."""
    if shape.rank() <= 4:
        return Shape(shape)
    shape_4d = [1, 1, 1, 1]
    # Collapse all leading dimensions into first dimension
    shape_4d[0] = 1
    for i in range(shape.rank() - 4 + 1):
        shape_4d[0] *= shape[i]
    shape_4d[0] -= 1 if shape_4d[0] > 0 else 0
    # Copy last 3 dimensions
    extra_rank = shape.rank() - 4
    shape_4d[1] = shape[1 + extra_rank] if 1 + extra_rank < shape.rank() else 1
    shape_4d[2] = shape[2 + extra_rank] if 2 + extra_rank < shape.rank() else 1
    shape_4d[3] = shape[3 + extra_rank] if 3 + extra_rank < shape.rank() else 1

    return Shape(shape_4d)


def reshape(tensor, shape, padded_shape=None, memory_config=None, pad_value=None,
            reshape_map_mode=None, sub_core_grids=None):
    """Reshape tensor to new shape."""
    if isinstance(shape, (list, tuple)):
        shape = Shape(shape)

    if padded_shape is None:
        padded_shape = pad_to_tile_shape(shape._shape) if tensor.get_layout() == Layout.TILE_LAYOUT else shape

    if isinstance(padded_shape, (list, tuple)):
        padded_shape = Shape(padded_shape)

    # Check execution mode for tracking
    mode = get_execution_mode()
    should_track = (mode == ExecutionMode.TRACK_ONLY or mode == ExecutionMode.EXECUTE_AND_TRACK)

    if should_track:
        _tracker.track_reshape(tensor.logical_shape()._shape, shape._shape)

    # Preserve data if available (reshape doesn't change data, just view)
    output_data = None
    if hasattr(tensor, 'has_data') and tensor.has_data():
        output_data = tensor.get_data()
    elif hasattr(tensor, 'get_data'):
        try:
            output_data = tensor.get_data()
        except Exception:
            output_data = None
    # Use the same tensor type among the different tensor types as the input tensor
    return type(tensor)(
        shape=shape,
        dtype=tensor.dtype,
        layout=tensor.get_layout(),
        memory_config=memory_config or tensor.memory_config(),
        padded_shape=padded_shape,
        device=tensor.device,
        data=output_data
    )


def squeeze(tensor, dim):
    """Squeeze tensor at specified dimension."""
    shape = tensor.logical_shape()
    if dim >= shape.rank():
        raise RuntimeError(f"Dimension {dim} out of range for tensor of rank {shape.rank()}")

    if shape[dim] != 1:
        raise RuntimeError(f"Cannot squeeze dimension {dim} with size {shape[dim]}")

    new_shape = list(shape._shape)
    new_shape.pop(dim)

    new_padded_shape = list(tensor.padded_shape()._shape)
    if dim < len(new_padded_shape):
        new_padded_shape.pop(dim)

    # Preserve data if available (squeeze doesn't change data, just removes dimension)
    output_data = tensor.get_data() if tensor.has_data() else None

    # Use the same tensor type among the different tensor types as the input tensor
    return type(tensor)(
        shape=Shape(new_shape),
        dtype=tensor.dtype,
        layout=tensor.get_layout(),
        memory_config=tensor.memory_config(),
        padded_shape=Shape(new_padded_shape),
        device=tensor.device,
        data=output_data
    )


def pad(tensor, padding, pad_value, output_memory_config=None):
    """Pad tensor with specified padding."""
    logical_shape = tensor.logical_shape()
    padded_shape = tensor.padded_shape()
    # Calculate new padded shape
    new_padded = list(padded_shape._shape)
    if len(padding) >= 2:
        # Last two dimensions
        if len(new_padded) >= 2:
            new_padded[-2] += padding[-2][0] + padding[-2][1]
            new_padded[-1] += padding[-1][0] + padding[-1][1]
    # Check execution mode for tracking
    mode = get_execution_mode()
    should_track = (mode == ExecutionMode.TRACK_ONLY or mode == ExecutionMode.EXECUTE_AND_TRACK)
    should_execute = (mode == ExecutionMode.EXECUTE or mode == ExecutionMode.EXECUTE_AND_TRACK)

    if should_track:
        _tracker.track_pad(tensor.padded_shape()._shape, new_padded)

    # Perform actual padding if in execute mode
    output_data = None
    if should_execute and tensor.has_data():
        # Simple padding: create new data array and copy existing data
        input_data = tensor.get_data()
        output_shape = new_padded
        # Calculate total elements
        total_output_elements = 1
        for dim in output_shape:
            total_output_elements *= dim

        python_type = _get_dtype_python_type(tensor.dtype)
        output_data = [python_type(pad_value) for _ in range(total_output_elements)]

        # Copy input data to output (simplified - assumes padding at end)
        # This is a simplified implementation
        for i in range(min(len(input_data), len(output_data))):
            output_data[i] = input_data[i]

    # Use the same tensor type among the different tensor types as the input tensor
    return type(tensor)(
        shape=logical_shape,
        dtype=tensor.dtype,
        layout=tensor.get_layout(),
        memory_config=output_memory_config or tensor.memory_config(),
        padded_shape=Shape(new_padded),
        device=tensor.device,
        data=output_data
    )


def tilize(input_tensor, memory_config=None, output_dtype=None, use_multicore=True,
           use_low_perf=False, sub_core_grids=None):
    """
    Tilize operation - converts ROW_MAJOR_LAYOUT to TILE_LAYOUT.

    Args:
        input_tensor: Input tensor in ROW_MAJOR_LAYOUT
        memory_config: Optional output memory configuration
        output_dtype: Optional output data type
        use_multicore: Whether to use multicore (default: True)
        use_low_perf: Whether to use low performance mode (default: False)
        sub_core_grids: Optional sub-core grid specification
    Returns:
        TensorProxy with TILE_LAYOUT
    """
    # Validation
    if input_tensor.storage_type() != "DEVICE":
        raise RuntimeError("Operands to tilize need to be on device!")

    if input_tensor.buffer() is None:
        raise RuntimeError("Operands to tilize need to be allocated in buffers on device!")

    if input_tensor.get_layout() != Layout.ROW_MAJOR_LAYOUT:
        raise RuntimeError("Can only tilize row major data")

    # Check physical volume
    physical_vol = input_tensor.physical_volume()
    if physical_vol % TILE_HW != 0:
        raise RuntimeError(
            f"Input tensor physical volume ({physical_vol}) must be divisible by TILE_HW ({TILE_HW})"
        )

    # Check data type - handle both DataType enums and numpy dtypes
    # (Tensor class converts DataType to numpy dtype in __init__)
    tensor_dtype = input_tensor.dtype
    if isinstance(tensor_dtype, np.dtype):
        # Convert numpy dtype back to DataType for validation
        tensor_dtype = DataType.from_numpy(tensor_dtype)

    valid_dtypes = [DataType.BFLOAT16, DataType.FLOAT32, DataType.UINT32,
                    DataType.INT32, DataType.UINT16]
    if tensor_dtype not in valid_dtypes:
        raise RuntimeError(
            "Can only tilize bfloat16/float32 or int32/uint32/uint16 tensors"
        )

    # Handle empty tensors
    if physical_vol == 0:
        output_shape = input_tensor.logical_shape()
        output_padded = pad_to_tile_shape(output_shape._shape)
        # Preserve data if available (empty tensor case)
        output_data = input_tensor.get_data() if input_tensor.has_data() else None

        # Use the same tensor type among the different tensor types as the input tensor
        return type(input_tensor)(
            shape=output_shape,
            dtype=output_dtype or input_tensor.dtype,
            layout=Layout.TILE_LAYOUT,
            memory_config=memory_config or input_tensor.memory_config(),
            padded_shape=output_padded,
            device=input_tensor.device,
            data=output_data
        )

    # Handle rank > 4
    if input_tensor.logical_shape().rank() > 4:
        squeezed = squeeze_from_ND_to_4D(input_tensor, sub_core_grids)
        result = tilize(squeezed, memory_config, output_dtype, use_multicore,
                       use_low_perf, sub_core_grids)
        # Unsqueeze back
        return reshape(result, input_tensor.logical_shape(),
                     input_tensor.padded_shape(), memory_config, None, None, sub_core_grids)

    # Calculate number of tiles
    padded_shape = input_tensor.padded_shape()
    num_tiles_per_row = padded_shape[-1] // TILE_WIDTH
    num_tiles_per_col = padded_shape[-2] // TILE_HEIGHT if len(padded_shape) >= 2 else 1
    num_tiles = num_tiles_per_row * num_tiles_per_col

    # Calculate output shape (padded to tile boundaries)
    output_shape = input_tensor.logical_shape()
    output_padded = pad_to_tile_shape(output_shape._shape)

    # Check execution mode
    mode = get_execution_mode()
    should_execute = (mode == ExecutionMode.EXECUTE or mode == ExecutionMode.EXECUTE_AND_TRACK)
    should_track = (mode == ExecutionMode.TRACK_ONLY or mode == ExecutionMode.EXECUTE_AND_TRACK)
    # Perform actual tilize operation if in execute mode
    output_data = None
    if should_execute:
        # Get input data
        if input_tensor.has_data():
            input_data = input_tensor.get_data()
        else:
            # Create dummy data for testing (zeros)
            input_data = _create_empty_data(
                input_tensor.padded_shape()._shape,
                input_tensor.dtype,
                0.0
            )
        # Perform tilize
        output_data = _perform_tilize_operation(
            input_data,
            input_tensor.logical_shape()._shape,
            output_padded._shape,
            pad_value=0.0
        )
    # Track operation if needed
    if should_track:
        element_size = input_tensor.element_size()
        _tracker.track_tilize(
            input_tensor.logical_shape()._shape,
            num_tiles,
            num_tiles_per_row,
            num_tiles_per_col,
            use_multicore,
            element_size
        )

    # Use the same tensor type among the different tensor types as the input tensor
    return type(input_tensor)(
        shape=output_shape,
        dtype=output_dtype or input_tensor.dtype,
        layout=Layout.TILE_LAYOUT,
        memory_config=memory_config or input_tensor.memory_config(),
        padded_shape=output_padded,
        device=input_tensor.device,
        data=output_data
    )


def untilize(input_tensor, memory_config=None, use_multicore=True,
             use_pack_untilize=True, sub_core_grids=None):
    """
    Untilize operation - converts TILE_LAYOUT to ROW_MAJOR_LAYOUT.

    Args:
        input_tensor: Input tensor in TILE_LAYOUT
        memory_config: Optional output memory configuration
        use_multicore: Whether to use multicore (default: True)
        use_pack_untilize: Whether to use pack untilize (default: True)
        sub_core_grids: Optional sub-core grid specification
    Returns:
        TensorProxy with ROW_MAJOR_LAYOUT
    """
    # Validation
    if input_tensor.storage_type() != "DEVICE":
        raise RuntimeError("Operands to untilize need to be on device!")

    if input_tensor.buffer() is None:
        raise RuntimeError("Operands to untilize need to be allocated in buffers on device!")

    if input_tensor.get_layout() != Layout.TILE_LAYOUT:
        raise RuntimeError("Can only untilize tile major data")

    padded_shape = input_tensor.padded_shape()
    tensor_width = padded_shape[-1] if len(padded_shape) >= 1 else 1
    tensor_height = input_tensor.physical_volume() // tensor_width if tensor_width > 0 else 1

    # Validate tile alignment
    if tensor_width % TILE_WIDTH != 0:
        raise RuntimeError("Width must be evenly divisible into tiles")
    if tensor_height % TILE_HEIGHT != 0:
        raise RuntimeError("Height must be evenly divisible into tiles")

    # Check data type for pack untilize
    # Handle both DataType enums and numpy dtypes
    tensor_dtype = input_tensor.dtype
    if isinstance(tensor_dtype, np.dtype):
        tensor_dtype = DataType.from_numpy(tensor_dtype)

    if not use_pack_untilize:
        if tensor_dtype in [DataType.UINT32, DataType.INT32]:
            raise RuntimeError("Pack untilize must be enabled to support uint32/int32 data types")

    # Handle empty tensors
    if input_tensor.physical_volume() == 0:
        # Preserve data if available (empty tensor case)
        output_data = input_tensor.get_data() if input_tensor.has_data() else None

        # Use the same tensor type among the different tensor types as the input tensor
        return type(input_tensor)(
            shape=input_tensor.logical_shape(),
            dtype=input_tensor.dtype,
            layout=Layout.ROW_MAJOR_LAYOUT,
            memory_config=memory_config or input_tensor.memory_config(),
            padded_shape=input_tensor.logical_shape(),
            device=input_tensor.device,
            data=output_data
        )
    # Handle rank > 4
    if input_tensor.logical_shape().rank() > 4:
        original_logical = input_tensor.logical_shape()
        original_padded = input_tensor.padded_shape()
        squeezed = squeeze_from_ND_to_4D(input_tensor)
        result = untilize(squeezed, memory_config, use_multicore,
                         use_pack_untilize, sub_core_grids)
        # Unsqueeze back
        return reshape(result, original_logical, original_padded,
                     memory_config, None, None, sub_core_grids)

    # Calculate number of tiles
    num_tiles_per_row = tensor_width // TILE_WIDTH
    num_tiles_per_col = tensor_height // TILE_HEIGHT
    num_tiles = num_tiles_per_row * num_tiles_per_col

    # Output is unpadded (logical shape)
    output_shape = input_tensor.logical_shape()

    # Check execution mode
    mode = get_execution_mode()
    should_execute = (mode == ExecutionMode.EXECUTE or mode == ExecutionMode.EXECUTE_AND_TRACK)
    should_track = (mode == ExecutionMode.TRACK_ONLY or mode == ExecutionMode.EXECUTE_AND_TRACK)
    # Perform actual untilize operation if in execute mode
    output_data = None
    if should_execute:
        # Get input data
        if input_tensor.has_data():
            input_data = input_tensor.get_data()
        else:
            # Create dummy data for testing (zeros)
            input_data = _create_empty_data(
                input_tensor.padded_shape()._shape,
                input_tensor.dtype,
                0.0
            )
        # Perform untilize
        output_data = _perform_untilize_operation(
            input_data,
            input_tensor.padded_shape()._shape,
            output_shape._shape
        )
    # Track operation if needed
    if should_track:
        element_size = input_tensor.element_size()
        _tracker.track_untilize(
            input_tensor.logical_shape()._shape,
            num_tiles,
            num_tiles_per_row,
            num_tiles_per_col,
            use_multicore,
            use_pack_untilize,
            element_size
        )
    # Use the same tensor type among the different tensor types as the input tensor
    out = type(input_tensor)(
        shape=output_shape,
        dtype=input_tensor.dtype,
        layout=Layout.ROW_MAJOR_LAYOUT,
        memory_config=memory_config or input_tensor.memory_config(),
        padded_shape=output_shape,  # No padding in row major
        device=input_tensor.device,
        data=output_data
    )
    # (Un)tilize is elementwise on data; preserve the ttnn dtype (e.g. UINT32, which is stored as
    # INT32 numpy) so downstream LUT keys read the true dtype — else a uint32 index tensor is
    # mis-keyed INT32 (cf. the typecast layout/dtype preservation fix).
    src_dt = getattr(input_tensor, "_ttnn_dtype", None)
    if src_dt is not None:
        out._ttnn_dtype = src_dt
    # Emit an Untilize SimOp so the op appears in the graph (HW UntilizeDeviceOperation), matching
    # the silicon capture. Previously untilize was metadata-only (only the perf _tracker ran, no
    # graph node), so e.g. the decode sampling-tail untilize was profiler-only.
    if should_track and input_tensor.device is not None:
        op_name = generate_new_op_name()
        input_tensor.op_in.append(op_name)
        out.op_out = [op_name]
        opinfo = {
            'name': op_name,
            'optype': 'Untilize',
            'inList': [input_tensor.name],
            'outList': [out.name],
            'attrs': {'element_size': input_tensor.element_size()},
        }
        opobj = SimOp(opinfo)
        opobj.get_perf_counts([input_tensor], [out])
        opobj.update_tensor_counts([input_tensor], [out])
        input_tensor.device.add_op(opobj)
    return out


def tilize_with_val_padding(input_tensor, output_padded_shape, pad_value,
                            memory_config=None, output_dtype=None, dtype=None,
                            use_multicore=True, sub_core_grids=None):
    """
    Tilize with value padding - converts ROW_MAJOR_LAYOUT to TILE_LAYOUT with padding.

    Args:
        input_tensor: Input tensor in ROW_MAJOR_LAYOUT
        output_padded_shape: Target padded shape for output
        pad_value: Value to use for padding (float or int)
        memory_config: Optional output memory configuration
        output_dtype: Optional output data type
        dtype: Alias for ``output_dtype`` (tt-metal API compatibility)
        use_multicore: Whether to use multicore (default: True)
        sub_core_grids: Optional sub-core grid specification
    Returns:
        TensorProxy with TILE_LAYOUT
    """
    # Validation
    if input_tensor.storage_type() != "DEVICE":
        raise RuntimeError("Operands need to be on device!")

    if input_tensor.buffer() is None:
        raise RuntimeError("Operands need to be allocated in buffers on device!")

    if input_tensor.get_layout() != Layout.ROW_MAJOR_LAYOUT:
        raise RuntimeError("Can only tilize row major data")

    if output_dtype is None:
        output_dtype = dtype

    # Check data type - handle both DataType enums and numpy dtypes
    tensor_dtype = input_tensor.dtype
    if isinstance(tensor_dtype, np.dtype):
        tensor_dtype = DataType.from_numpy(tensor_dtype)

    valid_dtypes = [DataType.BFLOAT16, DataType.FLOAT32, DataType.UINT32,
                    DataType.INT32, DataType.UINT16]
    if tensor_dtype not in valid_dtypes:
        raise RuntimeError(
            "Can only tilize bfloat16/float32 or int32/uint32/uint16 tensors"
        )

    # Handle shape input
    if isinstance(output_padded_shape, (list, tuple)):
        output_padded_shape = Shape(output_padded_shape)
    elif not isinstance(output_padded_shape, Shape):
        raise TypeError(f"Invalid output_padded_shape type: {type(output_padded_shape)}")
    # Validate rank
    if output_padded_shape.rank() < 1:
        raise RuntimeError(
            f"Input tensor must be of rank >= 1, but its shape is {output_padded_shape}"
        )
    # Handle empty tensors
    if input_tensor.physical_volume() == 0:
        # Preserve data if available (empty tensor case)
        output_data = input_tensor.get_data() if input_tensor.has_data() else None

        # Use the same tensor type among the different tensor types as the input tensor
        return type(input_tensor)(
            shape=input_tensor.logical_shape(),
            dtype=output_dtype or input_tensor.dtype,
            layout=Layout.TILE_LAYOUT,
            memory_config=memory_config or input_tensor.memory_config(),
            padded_shape=output_padded_shape,
            device=input_tensor.device,
            data=output_data
        )
    # Handle rank > 4
    if input_tensor.logical_shape().rank() > 4:
        squeezed = squeeze_from_ND_to_4D(input_tensor, sub_core_grids)
        squeezed_output_shape = squeeze_shape_to_4D(output_padded_shape)
        result = tilize_with_val_padding(squeezed, squeezed_output_shape, pad_value,
                                        memory_config, output_dtype, use_multicore, sub_core_grids)
        # Unsqueeze back
        return reshape(result, input_tensor.logical_shape(), output_padded_shape,
                     memory_config, None, None, sub_core_grids)
    # Calculate number of tiles
    num_tiles_per_row = output_padded_shape[-1] // TILE_WIDTH
    num_tiles_per_col = output_padded_shape[-2] // TILE_HEIGHT if len(output_padded_shape) >= 2 else 1
    num_tiles = num_tiles_per_row * num_tiles_per_col

    # Output logical shape is same as input
    output_logical = input_tensor.logical_shape()

    # Check execution mode
    mode = get_execution_mode()
    should_execute = (mode == ExecutionMode.EXECUTE or mode == ExecutionMode.EXECUTE_AND_TRACK)
    should_track = (mode == ExecutionMode.TRACK_ONLY or mode == ExecutionMode.EXECUTE_AND_TRACK)
    # Perform actual tilize operation if in execute mode
    output_data = None
    if should_execute:
        # Get input data
        if input_tensor.has_data():
            input_data = input_tensor.get_data()
        else:
            # Create dummy data for testing (zeros)
            input_data = _create_empty_data(
                input_tensor.padded_shape()._shape,
                input_tensor.dtype,
                0.0
            )
        # Perform tilize with padding
        output_data = _perform_tilize_operation(
            input_data,
            input_tensor.logical_shape()._shape,
            output_padded_shape._shape,
            pad_value=pad_value
        )
    # Track operation if needed
    if should_track:
        _tracker.track_tilize_with_val_padding(
            input_tensor.logical_shape()._shape,
            output_padded_shape._shape,
            num_tiles
        )

    # Use the same tensor type among the different tensor types as the input tensor
    return type(input_tensor)(
        shape=output_logical,
        dtype=output_dtype or input_tensor.dtype,
        layout=Layout.TILE_LAYOUT,
        memory_config=memory_config or input_tensor.memory_config(),
        padded_shape=output_padded_shape,
        device=input_tensor.device,
        data=output_data
    )


def untilize_with_unpadding(input_tensor, output_tensor_end, memory_config=None,
                            use_multicore=True, use_pack_untilize=True,
                            sub_core_grids=None):
    """
    Untilize with unpadding - converts TILE_LAYOUT to ROW_MAJOR_LAYOUT with unpadding.

    Args:
        input_tensor: Input tensor in TILE_LAYOUT
        output_tensor_end: Shape specifying the end indices for output (unpadded)
        memory_config: Optional output memory configuration
        use_multicore: Whether to use multicore (default: True)
        use_pack_untilize: Whether to use pack untilize (default: True)
        sub_core_grids: Optional sub-core grid specification
    Returns:
        TensorProxy with ROW_MAJOR_LAYOUT
    """
    # Validation
    if input_tensor.storage_type() != "DEVICE":
        raise RuntimeError("Operands need to be on device!")

    if input_tensor.buffer() is None:
        raise RuntimeError("Operands need to be allocated in buffers on device!")

    if input_tensor.get_layout() != Layout.TILE_LAYOUT:
        raise RuntimeError("Can only untilize tile major data")

    # Handle shape input
    if isinstance(output_tensor_end, (list, tuple)):
        output_tensor_end = Shape(output_tensor_end)
    elif not isinstance(output_tensor_end, Shape):
        raise TypeError(f"Invalid output_tensor_end type: {type(output_tensor_end)}")
    # Handle empty tensors
    if input_tensor.physical_volume() == 0:
        output_shape = Shape([end + 1 for end in output_tensor_end._shape])
        # Preserve data if available (empty tensor case)
        output_data = input_tensor.get_data() if input_tensor.has_data() else None
        # Use the same tensor type among the different tensor types as the input tensor
        return type(input_tensor)(
            shape=output_shape,
            dtype=input_tensor.dtype,
            layout=Layout.ROW_MAJOR_LAYOUT,
            memory_config=memory_config or input_tensor.memory_config(),
            padded_shape=output_shape,
            device=input_tensor.device,
            data=output_data
        )
    # Handle rank > 4
    if input_tensor.logical_shape().rank() > 4:
        original_logical = input_tensor.logical_shape()
        squeezed = squeeze_from_ND_to_4D(input_tensor, sub_core_grids)
        squeezed_output_end = squeeze_shape_to_4D(output_tensor_end)
        result = untilize_with_unpadding(squeezed, squeezed_output_end, memory_config,
                                        use_multicore, use_pack_untilize, sub_core_grids)
        # Unsqueeze back
        return reshape(result, original_logical, None, memory_config, None, None, sub_core_grids)

    # Calculate output shape from end indices
    output_shape = Shape([end + 1 for end in output_tensor_end._shape])

    # Calculate number of tiles
    padded_shape = input_tensor.padded_shape()
    num_tiles_per_row = padded_shape[-1] // TILE_WIDTH
    num_tiles_per_col = padded_shape[-2] // TILE_HEIGHT if len(padded_shape) >= 2 else 1
    num_tiles = num_tiles_per_row * num_tiles_per_col
    # Check execution mode
    mode = get_execution_mode()
    should_execute = (mode == ExecutionMode.EXECUTE or mode == ExecutionMode.EXECUTE_AND_TRACK)
    should_track = (mode == ExecutionMode.TRACK_ONLY or mode == ExecutionMode.EXECUTE_AND_TRACK)
    # Perform actual untilize operation if in execute mode
    output_data = None
    if should_execute:
        # Get input data
        if input_tensor.has_data():
            input_data = input_tensor.get_data()
        else:
            # Create dummy data for testing (zeros)
            input_data = _create_empty_data(
                input_tensor.padded_shape()._shape,
                input_tensor.dtype,
                0.0
            )
        # Perform untilize with unpadding
        output_data = _perform_untilize_operation(
            input_data,
            padded_shape._shape,
            output_shape._shape
        )
    # Track operation if needed
    if should_track:
        _tracker.track_untilize_with_unpadding(
            input_tensor.logical_shape()._shape,
            output_shape._shape,
            num_tiles
        )

    # Use the same tensor type among the different tensor types as the input tensor
    out = type(input_tensor)(
        shape=output_shape,
        dtype=input_tensor.dtype,
        layout=Layout.ROW_MAJOR_LAYOUT,
        memory_config=memory_config or input_tensor.memory_config(),
        padded_shape=output_shape,  # No padding in row major
        device=input_tensor.device,
        data=output_data
    )
    # (Un)tilize is elementwise on data; preserve the ttnn dtype (e.g. UINT32, which is stored as
    # INT32 numpy) so downstream LUT keys read the true dtype — else a uint32 index tensor is
    # mis-keyed INT32 (cf. the typecast layout/dtype preservation fix).
    src_dt = getattr(input_tensor, "_ttnn_dtype", None)
    if src_dt is not None:
        out._ttnn_dtype = src_dt
    return out


# =============================================================================
# Layout operator APIs (SimOp-only; no execution, no _tracker)
# =============================================================================

def tilize_op(input_tensor, use_multicore=True, element_size=2, memory_config=None):
    """
    Create a Tilize SimOp and output tensor (tracking-only; no execution).
    Output logical shape = input shape; padded = last two dims rounded up to tile.
    """
    assert input_tensor.device is not None, "tilize_op requires input_tensor on device"
    op_name = generate_new_op_name()
    out_logical = input_tensor.logical_shape()
    out_padded = pad_to_tile_shape(out_logical._shape)
    out_tensor = Tensor(
        name=op_name + '.out',
        shape=out_logical._shape,
        dtype=input_tensor.dtype,
        layout=Layout.TILE_LAYOUT,
        padded_shape=out_padded._shape,
        op_out=[op_name],
        device=input_tensor.device,
    )
    input_tensor.op_in.append(op_name)
    opinfo = {
        'name': op_name,
        'optype': 'Tilize',
        'inList': [input_tensor.name],
        'outList': [out_tensor.name],
        'attrs': {'use_multicore': use_multicore, 'element_size': element_size},
    }
    opobj = SimOp(opinfo)
    opobj.get_perf_counts([input_tensor], [out_tensor])
    opobj.update_tensor_counts([input_tensor], [out_tensor])
    _propagate_ttnn_dtype([input_tensor], [out_tensor])
    if memory_config is not None:
        out_tensor._memory_config = memory_config
    input_tensor.device.add_op(opobj)
    return out_tensor


def untilize_op(input_tensor, use_multicore=True, use_pack_untilize=True, element_size=2, memory_config=None):
    """
    Create an Untilize SimOp and output tensor (tracking-only; no execution).
    Output ROW_MAJOR, same logical shape as input.
    """
    assert input_tensor.device is not None, "untilize_op requires input_tensor on device"
    op_name = generate_new_op_name()
    out_shape = input_tensor.logical_shape()
    out_tensor = Tensor(
        name=op_name + '.out',
        shape=out_shape._shape,
        dtype=input_tensor.dtype,
        layout=Layout.ROW_MAJOR_LAYOUT,
        padded_shape=out_shape._shape,
        op_out=[op_name],
        device=input_tensor.device,
    )
    input_tensor.op_in.append(op_name)
    opinfo = {
        'name': op_name,
        'optype': 'Untilize',
        'inList': [input_tensor.name],
        'outList': [out_tensor.name],
        'attrs': {
            'use_multicore': use_multicore,
            'use_pack_untilize': use_pack_untilize,
            'element_size': element_size,
        },
    }
    opobj = SimOp(opinfo)
    opobj.get_perf_counts([input_tensor], [out_tensor])
    opobj.update_tensor_counts([input_tensor], [out_tensor])
    _propagate_ttnn_dtype([input_tensor], [out_tensor])
    if memory_config is not None:
        out_tensor._memory_config = memory_config
    input_tensor.device.add_op(opobj)
    return out_tensor


def tilize_with_val_padding_op(input_tensor, output_padded_shape, pad_value,
                                use_multicore=True, element_size=2, memory_config=None,
                                output_logical_shape=None):
    """
    Create a TilizeWithValPadding SimOp and output tensor (tracking-only; no execution).
    output_padded_shape and pad_value are stored in op attrs.
    If output_logical_shape is set, the output tensor's logical shape (and attrs) use it;
    use after an Untilize when the row-major view was reshaped before tilize (hardware tile reshape).
    """
    assert input_tensor.device is not None, "tilize_with_val_padding_op requires input_tensor on device"
    if isinstance(output_padded_shape, Shape):
        output_padded_shape = output_padded_shape._shape
    output_padded_shape = list(output_padded_shape)
    op_name = generate_new_op_name()
    if output_logical_shape is not None:
        out_logical_shape = list(output_logical_shape)
    else:
        out_logical_shape = input_tensor.logical_shape()._shape
    out_tensor = Tensor(
        name=op_name + '.out',
        shape=out_logical_shape,
        dtype=input_tensor.dtype,
        layout=Layout.TILE_LAYOUT,
        padded_shape=output_padded_shape,
        op_out=[op_name],
        device=input_tensor.device,
    )
    input_tensor.op_in.append(op_name)
    attrs = {
        'output_padded_shape': output_padded_shape,
        'pad_value': pad_value,
        'use_multicore': use_multicore,
        'element_size': element_size,
    }
    if output_logical_shape is not None:
        attrs['output_logical_shape'] = out_logical_shape
    opinfo = {
        'name': op_name,
        'optype': 'TilizeWithValPadding',
        'inList': [input_tensor.name],
        'outList': [out_tensor.name],
        'attrs': attrs,
    }
    opobj = SimOp(opinfo)
    opobj.get_perf_counts([input_tensor], [out_tensor])
    opobj.update_tensor_counts([input_tensor], [out_tensor])
    _propagate_ttnn_dtype([input_tensor], [out_tensor])
    if memory_config is not None:
        out_tensor._memory_config = memory_config
    input_tensor.device.add_op(opobj)
    return out_tensor


def _pad_value_for_tilize_dtype(tensor_dtype):
    if isinstance(tensor_dtype, np.dtype):
        tensor_dtype = DataType.from_numpy(tensor_dtype)
    return 0.0 if tensor_dtype in [DataType.BFLOAT16, DataType.FLOAT32] else 0


def _device_tensor_has_buffer(tensor):
    if not isinstance(tensor, Tensor):
        return False
    if tensor.storage_type() != "DEVICE":
        return False
    try:
        return tensor.buffer() is not None
    except Exception:
        return False


def _resolve_reshape_out_dims(out_list: list[int], num_elements: int) -> list[int]:
    """Replace at most one ``-1`` with the inferred size so the shape has ``num_elements`` elements."""
    neg_one_idx = [i for i, d in enumerate(out_list) if d == -1]
    if not neg_one_idx:
        return out_list
    if len(neg_one_idx) > 1:
        raise RuntimeError(
            f"ttnn.reshape: at most one inferred dimension (-1) allowed, got shape {out_list}"
        )
    prod_known = 1
    for d in out_list:
        if d == -1:
            continue
        if d < 0:
            raise RuntimeError(
                f"ttnn.reshape: invalid dimension {d} (only -1 is allowed for size inference)"
            )
        prod_known *= d
    if prod_known == 0:
        raise RuntimeError(
            f"ttnn.reshape: cannot infer -1 when product of explicit dimensions is 0 (shape {out_list})"
        )
    if num_elements % prod_known != 0:
        raise RuntimeError(
            f"ttnn.reshape: element count mismatch: input has {num_elements} elements, "
            f"shape {out_list} implies inferred dim would not be integral (product of "
            f"specified dims = {prod_known})"
        )
    inferred = num_elements // prod_known
    resolved = list(out_list)
    resolved[neg_one_idx[0]] = inferred
    return resolved


def _is_reshape_dim_sequence(x) -> bool:
    """True if ``x`` is a Shape or a list/tuple of integer-like dimension sizes."""
    if isinstance(x, Shape):
        return True
    if not isinstance(x, (list, tuple)):
        return False
    return all(isinstance(v, (int, np.integer)) for v in x)


def _reshape_tile_device_execute(tensor, out_list, memory_config, sub_core_grids=None):
    """Execute path: TILE → row-major (untilize*) → logical reshape → tilize with padding."""
    ends = Shape([d - 1 for d in tensor.logical_shape()._shape])
    mc = memory_config or tensor.memory_config() or DRAM_MEMORY_CONFIG
    if tensor.logical_shape()._shape != tensor.padded_shape()._shape:
        r = untilize_with_unpadding(tensor, ends, mc, True, True, sub_core_grids)
    else:
        r = untilize(tensor, mc, True, True, sub_core_grids)
    r2 = reshape(r, Shape(out_list), None, mc, None, None, sub_core_grids)
    p_out = pad_to_tile_shape(out_list)
    pv = _pad_value_for_tilize_dtype(tensor.dtype)
    return tilize_with_val_padding(r2, p_out, pv, mc, None, True, sub_core_grids)


def _reshape_simop_tracked_output(
    tensor: Tensor,
    logical_list: list[int],
    phys_list: list[int],
    padded_list: list[int] | None,
) -> Tensor:
    """Run :class:`Reshape` shape inference on ``phys_list`` (matches storage element count), then
    set the front-end tensor's logical shape and optional explicit padded shape (TTNN dual-shape)."""
    from ttsim.ops.tensor import SimTensor

    out = ttnn_reshape_simop(tensor, phys_list)
    if padded_list is not None:
        SimTensor.set_shape(out, logical_list)
        out._padded_shape = Shape(padded_list)
    return out


def ttnn_reshape(tensor, shape, arg3=None, memory_config=None, sub_core_grids=None):
    """
    Graph-aware reshape for ``ttsim.front.ttnn``.

    * **TILE** tensors on device (with buffer): records **Untilize** / **UntilizeWithUnpadding** then
      **TilizeWithValPadding** (tile-native reshape on hardware), not a standalone **Reshape** op.
    * **ROW_MAJOR** on device + track: uses **Reshape** SimOp (``op.reshape``).
    * Otherwise: same as :func:`reshape` (metadata / host / execute-only).

    TTNN-style optional third argument: **padded_shape** (list, tuple, or :class:`Shape`) when the
    logical shape differs from the physical/padded layout (same element count as the input; logical
    metadata may report a different volume). If ``arg3`` is a :class:`MemoryConfig`, it is treated as
    ``memory_config`` (two-argument ``reshape`` with memory config only).
    """
    padded_src = None
    if _is_reshape_dim_sequence(arg3):
        padded_src = arg3
    elif arg3 is not None:
        if isinstance(arg3, MemoryConfig):
            memory_config = arg3
        else:
            raise TypeError(
                f"ttnn.reshape: third argument must be padded shape (list/tuple/Shape) or "
                f"MemoryConfig, got {type(arg3)}"
            )

    if isinstance(shape, Shape):
        logical_list = list(shape._shape)
    elif isinstance(shape, (list, tuple)):
        logical_list = [int(x) for x in shape]
    else:
        raise TypeError(f"ttnn_reshape: shape must be list, tuple, or Shape, got {type(shape)}")

    logical_vol = tensor.logical_shape().volume()
    storage_vol = (
        tensor.physical_volume()
        if hasattr(tensor, "physical_volume")
        else tensor.padded_shape().volume()
    )

    padded_list: list[int] | None = None
    if padded_src is not None:
        if isinstance(padded_src, Shape):
            padded_list = list(padded_src._shape)
        else:
            padded_list = [int(x) for x in padded_src]
        padded_list = _resolve_reshape_out_dims(padded_list, storage_vol)
        # Output padded layout may differ from input padded_shape (e.g. decode paths set a
        # hardware shard width); do not require volume equality with input storage.
        neg_logical = [i for i, d in enumerate(logical_list) if d == -1]
        if len(neg_logical) > 1:
            raise RuntimeError(
                f"ttnn.reshape: at most one -1 in logical shape when padded_shape is given, "
                f"got {logical_list}"
            )
        if len(neg_logical) == 1:
            logical_list = _resolve_reshape_out_dims(logical_list, logical_vol)
    else:
        logical_list = _resolve_reshape_out_dims(logical_list, logical_vol)
        if Shape(logical_list).volume() != logical_vol:
            raise RuntimeError(
                f"ttnn.reshape: element count mismatch input {tensor.logical_shape()._shape} "
                f"vs output {logical_list}"
            )

    out_shape = Shape(logical_list)
    out_padded_shape = Shape(padded_list) if padded_list is not None else None
    phys_list = padded_list if padded_list is not None else logical_list

    mode = get_execution_mode()
    should_execute = mode in (ExecutionMode.EXECUTE, ExecutionMode.EXECUTE_AND_TRACK)
    should_track = mode in (ExecutionMode.TRACK_ONLY, ExecutionMode.EXECUTE_AND_TRACK)

    dev_graph = _device_tensor_has_buffer(tensor)

    if dev_graph and tensor.get_layout() == Layout.TILE_LAYOUT:
        if should_track:
            _tracker.track_reshape(tensor.logical_shape()._shape, logical_list)
        out_exec = None
        if should_execute:
            out_exec = _reshape_tile_device_execute(
                tensor, phys_list, memory_config, sub_core_grids
            )
        if should_track:
            mc = memory_config or tensor.memory_config() or DRAM_MEMORY_CONFIG
            elem_sz = tensor.element_size() if hasattr(tensor, 'element_size') else 2
            if tensor.logical_shape()._shape != tensor.padded_shape()._shape:
                u = untilize_with_unpadding_op(
                    tensor,
                    tensor.logical_shape()._shape,
                    use_multicore=True,
                    use_pack_untilize=True,
                    element_size=elem_sz,
                    memory_config=mc,
                )
            else:
                u = untilize_op(
                    tensor,
                    use_multicore=True,
                    use_pack_untilize=True,
                    element_size=elem_sz,
                    memory_config=mc,
                )
            pv = _pad_value_for_tilize_dtype(tensor.dtype)
            elem_sz = u.element_size() if hasattr(u, 'element_size') else 2
            out_t = tilize_with_val_padding_op(
                u,
                pad_to_tile_shape(phys_list)._shape,
                pv,
                use_multicore=True,
                element_size=elem_sz,
                memory_config=mc,
                output_logical_shape=logical_list,
            )
            if should_execute:
                return out_exec
            return out_t
        return out_exec

    padded_for_reshape = out_padded_shape if out_padded_shape is not None else out_shape

    if dev_graph and tensor.get_layout() == Layout.ROW_MAJOR_LAYOUT:
        # Dual-shape reshape can target a padded layout with more elements than .nelems()
        # (logical view); reshape_sinf requires equal counts, so use metadata-only reshape.
        phys_vol = Shape(phys_list).volume()
        soft_dual = padded_list is not None and phys_vol != tensor.nelems()

        if soft_dual:
            out_rm = reshape(
                tensor,
                out_shape,
                padded_for_reshape,
                memory_config,
                None,
                None,
                sub_core_grids,
            )
            if should_track:
                _tracker.track_reshape(tensor.logical_shape()._shape, logical_list)
            return out_rm

        if should_execute:
            out_e = reshape(
                tensor,
                out_shape,
                padded_for_reshape,
                memory_config,
                None,
                None,
                sub_core_grids,
            )
            if should_track:
                ttnn_reshape_simop(tensor, phys_list)
            return out_e
        if should_track:
            _tracker.track_reshape(tensor.logical_shape()._shape, logical_list)
            return _reshape_simop_tracked_output(
                tensor, logical_list, phys_list, padded_list
            )
        return reshape(
            tensor,
            out_shape,
            padded_for_reshape,
            memory_config,
            None,
            None,
            sub_core_grids,
        )

    return reshape(
        tensor,
        out_shape,
        out_padded_shape,
        memory_config,
        None,
        None,
        sub_core_grids,
    )


def untilize_with_unpadding_op(input_tensor, output_shape,
                                use_multicore=True, use_pack_untilize=True, element_size=2, memory_config=None):
    """
    Create an UntilizeWithUnpadding SimOp and output tensor (tracking-only; no execution).
    (Python name ``untilize_with_unpadding_op`` is historical; optype matches hardware naming.)
    output_shape is the logical output shape (list or tuple); stored in op attrs.
    """
    assert input_tensor.device is not None, "untilize_with_unpadding_op requires input_tensor on device"
    if isinstance(output_shape, Shape):
        output_shape = output_shape._shape
    output_shape = list(output_shape)
    op_name = generate_new_op_name()
    out_tensor = Tensor(
        name=op_name + '.out',
        shape=output_shape,
        dtype=input_tensor.dtype,
        layout=Layout.ROW_MAJOR_LAYOUT,
        padded_shape=output_shape,
        op_out=[op_name],
        device=input_tensor.device,
    )
    # Propagate or apply memory_config to preserve sharding/placement metadata
    out_tensor._memory_config = memory_config if memory_config is not None else getattr(input_tensor, '_memory_config', None)
    input_tensor.op_in.append(op_name)
    attrs_dict = {
        'output_shape': output_shape,
        'use_multicore': use_multicore,
        'use_pack_untilize': use_pack_untilize,
        'element_size': element_size,
    }
    if memory_config is not None:
        attrs_dict['memory_config'] = memory_config
    opinfo = {
        'name': op_name,
        'optype': 'UntilizeWithUnpadding',
        'inList': [input_tensor.name],
        'outList': [out_tensor.name],
        'attrs': attrs_dict,
    }
    opobj = SimOp(opinfo)
    opobj.get_perf_counts([input_tensor], [out_tensor])
    opobj.update_tensor_counts([input_tensor], [out_tensor])
    _propagate_ttnn_dtype([input_tensor], [out_tensor])
    # Memory config propagation: explicit kwarg wins; otherwise inherit the input's
    # memory_config so direct callers (e.g. BH ViT patch embeddings) don't silently
    # land in DRAM_INTERLEAVED when the HW path is L1_INTERLEAVED.  Mirrors
    # untilize_op / tilize_op / tilize_with_val_padding_op behavior.
    if memory_config is not None:
        out_tensor._memory_config = memory_config
    elif getattr(input_tensor, "_memory_config", None) is not None:
        out_tensor._memory_config = input_tensor._memory_config
    input_tensor.device.add_op(opobj)
    return out_tensor


def permute_op(input_tensor, dims, memory_config=None):
    """
    Create a Permute SimOp and output tensor (tracking-only; no execution).

    Used by :func:`permute` (re-exported as ``ttnn.permute`` from the TT-Polaris package).
    ``from ttsim.front.ttnn.op import permute`` still refers to **Transpose** in ``op.py``.

    Requires ``input_tensor`` on a device (same contract as ``tilize_op``).
    ``dims`` may be a list or tuple of dimension indices (full permutation).
    """
    assert input_tensor.device is not None, "permute_op requires input_tensor on device"
    perm = list(dims)
    in_shape = list(input_tensor.logical_shape()._shape)
    if len(perm) != len(in_shape):
        raise ValueError(
            f"permute_op: len(dims)={len(perm)} must match input rank {len(in_shape)}"
        )
    out_shape = [in_shape[i] for i in perm]
    op_name = generate_new_op_name()
    out_tensor = Tensor(
        name=op_name + '.out',
        shape=out_shape,
        dtype=input_tensor.dtype,
        layout=input_tensor.layout,
        op_out=[op_name],
        device=input_tensor.device,
    )
    # Propagate or apply memory_config to preserve sharding/placement metadata
    out_tensor._memory_config = memory_config if memory_config is not None else getattr(input_tensor, '_memory_config', None)
    input_tensor.op_in.append(op_name)
    attrs: dict = {'perm': perm}
    if memory_config is not None:
        attrs['memory_config'] = memory_config
    opinfo = {
        'name': op_name,
        'optype': 'Permute',
        'inList': [input_tensor.name],
        'outList': [out_tensor.name],
        'attrs': attrs,
    }
    opobj = SimOp(opinfo)
    opobj.get_perf_counts([input_tensor], [out_tensor])
    opobj.update_tensor_counts([input_tensor], [out_tensor])
    _propagate_ttnn_dtype([input_tensor], [out_tensor])
    input_tensor.device.add_op(opobj)
    return out_tensor


def permute(input_tensor, dims, memory_config=None):
    """
    TT-Polaris ``ttnn.permute`` implementation: records optype **Permute** on the device graph.

    Matches ``ttnn.permute(tensor, dims)`` with ``dims`` a list or tuple of axis indices.
    For ``memory_config``, see :func:`permute_op`.

    Note: ``from ttsim.front.ttnn.op import permute`` is still the **Transpose** immediate op.
    """
    if isinstance(dims, (list, tuple)):
        dims_list = [int(x) for x in dims]
    else:
        raise TypeError(f"permute: dims must be a list or tuple of ints, got {type(dims)}")
    return permute_op(input_tensor, dims_list, memory_config=memory_config)


def to_layout(tensor, layout, dtype=None, memory_config=None, sub_core_grids=None):
    """
    Convert tensor to specified layout.
    Args:
        tensor: Input tensor
        layout: Target layout (ROW_MAJOR_LAYOUT or TILE_LAYOUT)
        dtype: Optional output data type (only for TILE_LAYOUT conversion)
        memory_config: Optional output memory configuration
        sub_core_grids: Optional sub-core grid specification
    Returns:
        TensorProxy with requested layout
    """
    # Check execution mode
    mode = get_execution_mode()
    should_execute = (mode == ExecutionMode.EXECUTE or mode == ExecutionMode.EXECUTE_AND_TRACK)
    should_track = (mode == ExecutionMode.TRACK_ONLY or mode == ExecutionMode.EXECUTE_AND_TRACK)

    # When not executing, we either use operator-style SimOps (_op) or _tracker; tracking is done per-branch below.

    # If already in requested layout
    if tensor.get_layout() == layout:
        # Check if dtype is specified when converting to ROW_MAJOR_LAYOUT (error case)
        if layout == Layout.ROW_MAJOR_LAYOUT and dtype is not None:
            raise RuntimeError("dtype cannot be specified when converting to ROW_MAJOR_LAYOUT!")

        if dtype is not None and dtype != tensor.dtype:
            import warnings
            warnings.warn(
                "ttnn::to_layout: dtype is specified but the tensor is already in the requested layout! "
                "So, the dtype won't be changed!"
            )
        if memory_config is not None:
            current_mem_config = tensor.memory_config()
            if memory_config != current_mem_config:
                import warnings
                warnings.warn(
                    "ttnn::to_layout: memory_config is specified but the tensor is already in the requested layout! "
                    "So, the memory_config won't be changed!"
                )
        return tensor
    # Validate supported layouts
    supported_layouts = [Layout.ROW_MAJOR_LAYOUT, Layout.TILE_LAYOUT]
    if layout not in supported_layouts:
        raise RuntimeError(
            f"ttnn::to_layout: Unsupported layout conversion from {tensor.get_layout()} to {layout}!"
        )

    # Get output memory config
    output_memory_config = memory_config or tensor.memory_config() or DRAM_MEMORY_CONFIG

    # Handle device tensors
    # Check if tensor has a buffer to distinguish real device tensors from host tensors (SHIM) with device reference.
    # ttsim Tensor sets _buffer to a placeholder when on device and implements buffer(); other types may have a real buffer.
    has_buffer = False
    if hasattr(tensor, 'buffer'):
        try:
            has_buffer = tensor.buffer() is not None
        except Exception:
            has_buffer = False

    is_device_tensor = tensor.storage_type() == "DEVICE" and has_buffer

    if is_device_tensor:
        needs_padding_change = requires_padding_change(tensor, layout)

        logger.debug(
            "to_layout: needs_padding_change {} for tensor {} to layout {}",
            needs_padding_change,
            tensor.name,
            layout,
        )
        if not needs_padding_change:
            # Simple conversion without padding change
            if layout == Layout.ROW_MAJOR_LAYOUT:
                if dtype is not None:
                    raise RuntimeError("dtype cannot be specified when converting to ROW_MAJOR_LAYOUT!")
                if should_execute:
                    return untilize(tensor, output_memory_config, True, True, sub_core_grids)
                elif should_track and tensor.device is not None and isinstance(tensor, Tensor):
                    logger.debug(
                        "to_layout: choosing untilize_op (device tensor, no padding change, TILE->ROW_MAJOR; logical_shape {} == padded_shape {})",
                        tensor.logical_shape()._shape,
                        tensor.padded_shape()._shape,
                    )
                    elem_sz = tensor.element_size() if hasattr(tensor, 'element_size') else 2
                    return untilize_op(tensor, use_multicore=True, use_pack_untilize=True, element_size=elem_sz, memory_config=output_memory_config)
                else:
                    # Not using untilize_op: input is not a ttsim Tensor (e.g. TensorProxy) or has no device; preserve API with tracker + manual tensor.
                    if should_track:
                        _tracker.track_to_layout(tensor.get_layout(), layout)
                    return type(tensor)(
                        shape=tensor.logical_shape(),
                        dtype=tensor.dtype,
                        layout=layout,
                        memory_config=output_memory_config,
                        padded_shape=tensor.logical_shape(),
                        device=tensor.device
                    )
            elif layout == Layout.TILE_LAYOUT:
                if should_execute:
                    return tilize(tensor, output_memory_config, dtype, True, False, sub_core_grids)
                elif should_track and tensor.device is not None and isinstance(tensor, Tensor):
                    logger.debug(
                        "to_layout: choosing tilize_op (device tensor, no padding change, ROW_MAJOR->TILE; logical_shape already tile-aligned)"
                    )
                    element_size = dtype.itemsize if dtype is not None and isinstance(dtype, DataType) else tensor.element_size()
                    out = tilize_op(
                        tensor,
                        use_multicore=True,
                        element_size=element_size,
                        memory_config=output_memory_config,
                    )
                    if dtype is not None and isinstance(dtype, DataType):
                        out._ttnn_dtype = dtype
                        out.dtype = dtype.to_numpy
                    return out
                else:
                    # Not using tilize_op: input is not a ttsim Tensor (e.g. TensorProxy) or has no device; preserve API with tracker + manual tensor.
                    if should_track:
                        _tracker.track_to_layout(tensor.get_layout(), layout)
                    return type(tensor)(
                        shape=tensor.logical_shape(),
                        dtype=dtype or tensor.dtype,
                        layout=layout,
                        memory_config=output_memory_config,
                        padded_shape=pad_to_tile_shape(tensor.logical_shape()._shape),
                        device=tensor.device
                    )
        else:
            # Conversion with padding change
            if layout == Layout.ROW_MAJOR_LAYOUT:
                if dtype is not None and dtype != tensor.dtype:
                    raise RuntimeError(
                        "dtype cannot be different from tensor dtype when converting to ROW_MAJOR_LAYOUT on device!"
                    )

                # Calculate output shape (unpadded)
                output_shape = tensor.logical_shape()
                output_tensor_end = Shape([dim - 1 for dim in output_shape._shape])

                if should_execute:
                    result = untilize_with_unpadding(
                        tensor, output_tensor_end, output_memory_config, True, True, sub_core_grids
                    )
                    out = reshape(result, output_shape, None, None, None, sub_core_grids)
                    if should_track and tensor.device is not None and isinstance(tensor, Tensor):
                        elem_sz = tensor.element_size() if hasattr(tensor, 'element_size') else 2
                        untilize_with_unpadding_op(
                            tensor,
                            output_shape._shape,
                            use_multicore=True,
                            use_pack_untilize=True,
                            element_size=elem_sz,
                            memory_config=output_memory_config,
                        )
                    return out
                elif should_track and tensor.device is not None and isinstance(tensor, Tensor):
                    logger.debug(
                        "to_layout: choosing untilize_with_unpadding_op (device tensor, padding change, TILE->ROW_MAJOR; padded_shape!=logical_shape, padded_shape={}, logical_shape={}, output_shape={})",
                        tensor.padded_shape()._shape,
                        tensor.logical_shape()._shape,
                        output_shape._shape,
                    )
                    # Single UntilizeWithUnpadding SimOp; logical shape is final ROW_MAJOR (no Reshape op).
                    elem_sz = tensor.element_size() if hasattr(tensor, 'element_size') else 2
                    return untilize_with_unpadding_op(
                        tensor,
                        output_shape._shape,
                        use_multicore=True,
                        use_pack_untilize=True,
                        element_size=elem_sz,
                        memory_config=output_memory_config,
                    )
                else:
                    # Not using untilize_with_unpadding_op: input is not a ttsim Tensor or has no device; preserve API with tracker + manual tensor.
                    if should_track:
                        _tracker.track_to_layout(tensor.get_layout(), layout)
                    return type(tensor)(
                        shape=output_shape,
                        dtype=tensor.dtype,
                        layout=layout,
                        memory_config=output_memory_config,
                        padded_shape=output_shape,
                        device=tensor.device
                    )
            elif layout == Layout.TILE_LAYOUT:
                # Calculate padded output shape
                logical_shape = tensor.logical_shape()
                padded_output_shape = pad_to_tile_shape(logical_shape._shape)
                # Check for height sharded tensors (if memory_layout is set)
                mem_config = tensor.memory_config()
                if mem_config is not None and hasattr(mem_config, 'memory_layout') and mem_config.memory_layout == TensorMemoryLayout.HEIGHT_SHARDED:
                    padding = [[0, 0], [0, 0]]
                    if len(logical_shape) >= 2:
                        padding.append([0, padded_output_shape[-2] - logical_shape[-2]])
                        padding.append([0, padded_output_shape[-1] - logical_shape[-1]])
                    else:
                        padding = [[0, 0], [0, 0]]
                    pad_value = _pad_value_for_tilize_dtype(tensor.dtype)
                    if should_execute:
                        padded_tensor = pad(tensor, padding, pad_value, output_memory_config)
                        return tilize(padded_tensor, output_memory_config, dtype, True, False, sub_core_grids)
                    elif should_track and tensor.device is not None and isinstance(tensor, Tensor):
                        logger.debug(
                            "to_layout: choosing tilize_with_val_padding_op (height-sharded device tensor, padding change, ROW_MAJOR->TILE; padded_output_shape={})",
                            padded_output_shape._shape,
                        )
                        elem_sz = tensor.element_size() if hasattr(tensor, 'element_size') else 2
                        out = tilize_with_val_padding_op(
                            tensor, padded_output_shape._shape, pad_value,
                            use_multicore=True, element_size=elem_sz, memory_config=output_memory_config,
                        )
                        # Symmetry with the no-padding-change tilize_op branch above:
                        # honor the `dtype` kwarg in track-only mode by tagging
                        # ``_ttnn_dtype`` on the SimOp output.
                        if dtype is not None and isinstance(dtype, DataType):
                            out._ttnn_dtype = dtype
                        return out
                    else:
                        if should_track:
                            _tracker.track_to_layout(tensor.get_layout(), layout)
                        return type(tensor)(
                            shape=logical_shape,
                            dtype=dtype or tensor.dtype,
                            layout=layout,
                            memory_config=output_memory_config,
                            padded_shape=padded_output_shape,
                            device=tensor.device
                        )
                else:
                    # Use tilize_with_val_padding
                    pad_value = _pad_value_for_tilize_dtype(tensor.dtype)
                    if should_execute:
                        return tilize_with_val_padding(
                            tensor, padded_output_shape, pad_value, output_memory_config, dtype, True, sub_core_grids
                        )
                    elif should_track and tensor.device is not None and isinstance(tensor, Tensor):
                        logger.debug(
                            "to_layout: choosing tilize_with_val_padding_op (device tensor, padding change, ROW_MAJOR->TILE; logical not tile-aligned, padded_output_shape={}, pad_value={})",
                            padded_output_shape._shape,
                            pad_value,
                        )
                        elem_sz = tensor.element_size() if hasattr(tensor, 'element_size') else 2
                        out = tilize_with_val_padding_op(
                            tensor, padded_output_shape._shape, pad_value, use_multicore=True, element_size=elem_sz, memory_config=output_memory_config
                        )
                        # Symmetry with the no-padding-change tilize_op branch above:
                        # honor the `dtype` kwarg in track-only mode by tagging
                        # ``_ttnn_dtype`` on the SimOp output.
                        if dtype is not None and isinstance(dtype, DataType):
                            out._ttnn_dtype = dtype
                        return out
                    else:
                        # Not using tilize_with_val_padding_op: input is not a ttsim Tensor or has no device; preserve API with tracker + manual tensor.
                        if should_track:
                            _tracker.track_to_layout(tensor.get_layout(), layout)
                        return type(tensor)(
                            shape=logical_shape,
                            dtype=dtype or tensor.dtype,
                            layout=layout,
                            memory_config=output_memory_config,
                            padded_shape=padded_output_shape,
                            device=tensor.device
                        )
    else:
        # Host tensor conversion: not using tiling _op (tilize_op, untilize_op, etc.); layout SimOps are for device op graph only; host path uses in-process conversion and tracker.
        if dtype is not None:
            raise RuntimeError("dtype cannot be specified when converting layout on host!")

        # Perform actual data conversion if in execute mode
        output_data = None
        if should_execute and tensor.has_data():
            input_data = tensor.get_data()
            input_layout = tensor.get_layout()
            logical_shape = tensor.logical_shape()

            if layout == Layout.ROW_MAJOR_LAYOUT:
                # Convert from tile layout to row major
                if input_layout == Layout.TILE_LAYOUT:
                    # Get padded shape for untilize
                    input_padded_shape = tensor.padded_shape()
                    output_data = _perform_untilize_operation(
                        input_data,
                        input_padded_shape._shape,
                        logical_shape._shape
                    )
                else:
                    # Already in row major, just use the data
                    output_data = input_data
            elif layout == Layout.TILE_LAYOUT:
                # Convert from row major to tile layout
                if input_layout == Layout.ROW_MAJOR_LAYOUT:
                    # Calculate padded output shape
                    padded_output_shape = pad_to_tile_shape(logical_shape._shape)
                    pad_value = _pad_value_for_tilize_dtype(tensor.dtype)
                    output_data = _perform_tilize_operation(
                        input_data,
                        logical_shape._shape,
                        padded_output_shape._shape,
                        pad_value=pad_value
                    )
                else:
                    # Already in tile layout, just use the data
                    output_data = input_data

        if not requires_padding_change(tensor, layout):
            # Simple conversion - just change layout attribute (host path; tiling _op not used—see host comment above).
            if should_track:
                _tracker.track_to_layout(tensor.get_layout(), layout)
            return type(tensor)(
                shape=tensor.logical_shape(),
                dtype=tensor.dtype,
                layout=layout,
                memory_config=output_memory_config,
                padded_shape=tensor.padded_shape(),
                device=tensor.device,
                data=output_data
            )
        else:
            if layout == Layout.ROW_MAJOR_LAYOUT:
                # Host path with padding change (tiling _op not used—see host comment above).
                if should_track:
                    _tracker.track_to_layout(tensor.get_layout(), layout)
                # Convert to row major and unpad
                result = type(tensor)(
                    shape=tensor.logical_shape(),
                    dtype=tensor.dtype,
                    layout=layout,
                    memory_config=output_memory_config,
                    padded_shape=tensor.logical_shape(),  # Unpadded
                    device=tensor.device,
                    data=output_data
                )
                return reshape(result, tensor.logical_shape(), None, None, None, sub_core_grids)
            elif layout == Layout.TILE_LAYOUT:
                # Host path with padding change (tiling _op not used—see host comment above).
                if should_track:
                    _tracker.track_to_layout(tensor.get_layout(), layout)
                # Pad and convert to tile
                logical_shape = tensor.logical_shape()
                padded_output_shape = pad_to_tile_shape(logical_shape._shape)

                # Convert to tile layout
                return type(tensor)(
                    shape=logical_shape,
                    dtype=tensor.dtype,
                    layout=layout,
                    memory_config=output_memory_config,
                    padded_shape=padded_output_shape,
                    device=tensor.device,
                    data=output_data
                )

    raise RuntimeError(f"ttnn::to_layout: Unsupported output layout: {layout}!")


# Convenience function for tilize_with_zero_padding
def tilize_with_zero_padding(input_tensor, memory_config=None, output_dtype=None,
                            use_multicore=True, sub_core_grids=None):
    """
    Tilize with zero padding - convenience function that auto-calculates padded shape.
    Args:
        input_tensor: Input tensor in ROW_MAJOR_LAYOUT
        memory_config: Optional output memory configuration
        output_dtype: Optional output data type
        use_multicore: Whether to use multicore (default: True)
        sub_core_grids: Optional sub-core grid specification
    Returns:
        TensorProxy with TILE_LAYOUT
    """
    # Calculate padded shape
    padded_shape = input_tensor.padded_shape()
    padded_shape = pad_to_tile_shape(padded_shape._shape)

    # Determine pad value based on dtype - handle both DataType enums and numpy dtypes
    tensor_dtype = input_tensor.dtype
    if isinstance(tensor_dtype, np.dtype):
        tensor_dtype = DataType.from_numpy(tensor_dtype)

    if tensor_dtype in [DataType.BFLOAT16, DataType.FLOAT32]:
        pad_value = 0.0
    else:
        pad_value = 0

    return tilize_with_val_padding(
        input_tensor, padded_shape, pad_value, memory_config, output_dtype,
        use_multicore, sub_core_grids
    )


# =============================================================================
# Shard layout operator shims + SimOp graph builders
# =============================================================================

def interleaved_to_sharded_op(input_tensor, memory_config=None, element_size=2):
    """Create an InterleavedToSharded SimOp (tracking-only; no execution)."""
    assert input_tensor.device is not None, "interleaved_to_sharded_op requires input_tensor on device"
    op_name = generate_new_op_name()
    out_shape = input_tensor.logical_shape()
    out_tensor = Tensor(
        name=op_name + '.out',
        shape=out_shape._shape,
        dtype=input_tensor.dtype,
        layout=input_tensor.get_layout(),
        padded_shape=input_tensor.padded_shape()._shape,
        op_out=[op_name],
        device=input_tensor.device,
    )
    input_tensor.op_in.append(op_name)
    memory_config_attr = None
    if memory_config is not None:
        memory_config_attr = {
            'memory_layout': memory_config.memory_layout,
            'buffer_type': memory_config.buffer_type,
            'shard_spec': memory_config.shard_spec,
            'nd_shard_spec': getattr(memory_config, 'nd_shard_spec', None),
        }
    opinfo = {
        'name': op_name,
        'optype': 'InterleavedToSharded',
        'inList': [input_tensor.name],
        'outList': [out_tensor.name],
        'attrs': {
            'element_size': element_size,
            'memory_config': memory_config_attr,
        },
    }
    opobj = SimOp(opinfo)
    opobj.get_perf_counts([input_tensor], [out_tensor])
    opobj.update_tensor_counts([input_tensor], [out_tensor])
    _propagate_ttnn_dtype([input_tensor], [out_tensor])
    if memory_config is not None:
        out_tensor._memory_config = memory_config
    input_tensor.device.add_op(opobj)
    return out_tensor


def interleaved_to_sharded(input_tensor, memory_config=None, output_dtype=None):
    """Convert tensor from interleaved to sharded memory layout.

    Args:
        input_tensor: Input tensor to convert
        memory_config: Target memory configuration (should specify sharding)
        output_dtype: Optional output data type

    Returns:
        Tensor with sharded memory layout
    """
    mode = get_execution_mode()
    should_track = (mode == ExecutionMode.TRACK_ONLY or mode == ExecutionMode.EXECUTE_AND_TRACK)
    should_execute = (mode == ExecutionMode.EXECUTE or mode == ExecutionMode.EXECUTE_AND_TRACK)

    if should_track and input_tensor.device is not None and isinstance(input_tensor, Tensor):
        elem_sz = input_tensor.element_size() if hasattr(input_tensor, 'element_size') else 2
        out_tensor = interleaved_to_sharded_op(input_tensor, memory_config=memory_config, element_size=elem_sz)
        if output_dtype is not None:
            if isinstance(output_dtype, DataType):
                out_tensor._ttnn_dtype = output_dtype
                out_tensor.dtype = output_dtype.to_numpy
            else:
                out_tensor.dtype = output_dtype
        return out_tensor

    if should_track:
        elem_sz = input_tensor.element_size() if hasattr(input_tensor, 'element_size') else 2
        _tracker.track_interleaved_to_sharded(
            input_tensor.logical_shape()._shape,
            element_size=elem_sz,
        )

    output_data = None
    if should_execute and hasattr(input_tensor, 'has_data') and input_tensor.has_data():
        output_data = input_tensor.get_data()

    # Preserve compact TTNN dtype when output_dtype not specified
    if output_dtype is None:
        ttnn_dtype = getattr(input_tensor, "_ttnn_dtype", None)
        dtype_arg = ttnn_dtype if ttnn_dtype is not None else input_tensor.dtype
    else:
        dtype_arg = output_dtype

    return type(input_tensor)(
        shape=input_tensor.logical_shape(),
        dtype=dtype_arg,
        layout=input_tensor.get_layout(),
        memory_config=memory_config or input_tensor.memory_config(),
        padded_shape=input_tensor.padded_shape(),
        device=input_tensor.device,
        data=output_data,
    )


def sharded_to_interleaved_op(input_tensor, memory_config=None, element_size=2):
    """Create a ShardedToInterleaved SimOp (tracking-only; no execution)."""
    assert input_tensor.device is not None, "sharded_to_interleaved_op requires input_tensor on device"
    op_name = generate_new_op_name()
    out_shape = input_tensor.logical_shape()
    out_tensor = Tensor(
        name=op_name + '.out',
        shape=out_shape._shape,
        dtype=input_tensor.dtype,
        layout=input_tensor.get_layout(),
        padded_shape=input_tensor.padded_shape()._shape,
        op_out=[op_name],
        device=input_tensor.device,
    )
    input_tensor.op_in.append(op_name)
    memory_config_attr = None
    if memory_config is not None:
        memory_config_attr = {
            'memory_layout': memory_config.memory_layout,
            'buffer_type': memory_config.buffer_type,
            'shard_spec': memory_config.shard_spec,
            'nd_shard_spec': getattr(memory_config, 'nd_shard_spec', None),
        }
    opinfo = {
        'name': op_name,
        'optype': 'ShardedToInterleaved',
        'inList': [input_tensor.name],
        'outList': [out_tensor.name],
        'attrs': {
            'element_size': element_size,
            'memory_config': memory_config_attr,
        },
    }
    opobj = SimOp(opinfo)
    opobj.get_perf_counts([input_tensor], [out_tensor])
    opobj.update_tensor_counts([input_tensor], [out_tensor])
    _propagate_ttnn_dtype([input_tensor], [out_tensor])
    if memory_config is not None:
        out_tensor._memory_config = memory_config
    else:
        input_memory_config = input_tensor.memory_config()
        if input_memory_config is not None:
            out_tensor._memory_config = MemoryConfig(
                TensorMemoryLayout.INTERLEAVED,
                input_memory_config.buffer_type,
            )
        else:
            out_tensor._memory_config = MemoryConfig(
                TensorMemoryLayout.INTERLEAVED,
                BufferType.DRAM,
            )
    input_tensor.device.add_op(opobj)
    return out_tensor


def sharded_to_interleaved(input_tensor, memory_config=None, output_dtype=None):
    """Convert tensor from sharded to interleaved memory layout."""
    mode = get_execution_mode()
    should_track = (mode == ExecutionMode.TRACK_ONLY or mode == ExecutionMode.EXECUTE_AND_TRACK)
    should_execute = (mode == ExecutionMode.EXECUTE or mode == ExecutionMode.EXECUTE_AND_TRACK)

    if should_track and input_tensor.device is not None and isinstance(input_tensor, Tensor):
        elem_sz = input_tensor.element_size() if hasattr(input_tensor, 'element_size') else 2
        out_tensor = sharded_to_interleaved_op(input_tensor, memory_config=memory_config, element_size=elem_sz)
        if output_dtype is not None:
            if isinstance(output_dtype, DataType):
                out_tensor._ttnn_dtype = output_dtype
                out_tensor.dtype = output_dtype.to_numpy
            else:
                out_tensor.dtype = output_dtype
        return out_tensor

    if should_track:
        elem_sz = input_tensor.element_size() if hasattr(input_tensor, 'element_size') else 2
        _tracker.track_sharded_to_interleaved(
            input_tensor.logical_shape()._shape,
            element_size=elem_sz,
        )

    output_data = None
    if should_execute and hasattr(input_tensor, 'has_data') and input_tensor.has_data():
        output_data = input_tensor.get_data()

    # Default to interleaved memory config if not provided
    resolved_memory_config = memory_config
    if resolved_memory_config is None:
        input_memory_config = input_tensor.memory_config()
        if input_memory_config is not None:
            resolved_memory_config = MemoryConfig(
                TensorMemoryLayout.INTERLEAVED,
                input_memory_config.buffer_type,
            )

    # Preserve compact TTNN dtype when output_dtype not specified
    if output_dtype is None:
        ttnn_dtype = getattr(input_tensor, "_ttnn_dtype", None)
        dtype_arg = ttnn_dtype if ttnn_dtype is not None else input_tensor.dtype
    else:
        dtype_arg = output_dtype

    return type(input_tensor)(
        shape=input_tensor.logical_shape(),
        dtype=dtype_arg,
        layout=input_tensor.get_layout(),
        memory_config=resolved_memory_config,
        padded_shape=input_tensor.padded_shape(),
        device=input_tensor.device,
        data=output_data,
    )


def reshard_op(input_tensor, memory_config=None, element_size=2):
    """Create a Reshard SimOp (tracking-only; no execution)."""
    assert input_tensor.device is not None, "reshard_op requires input_tensor on device"
    op_name = generate_new_op_name()
    out_shape = input_tensor.logical_shape()
    out_tensor = Tensor(
        name=op_name + '.out',
        shape=out_shape._shape,
        dtype=input_tensor.dtype,
        layout=input_tensor.get_layout(),
        padded_shape=input_tensor.padded_shape()._shape,
        op_out=[op_name],
        device=input_tensor.device,
    )
    input_tensor.op_in.append(op_name)
    memory_config_attr = None
    if memory_config is not None:
        memory_config_attr = {
            'memory_layout': memory_config.memory_layout,
            'buffer_type': memory_config.buffer_type,
            'shard_spec': memory_config.shard_spec,
            'nd_shard_spec': getattr(memory_config, 'nd_shard_spec', None),
        }
    opinfo = {
        'name': op_name,
        'optype': 'Reshard',
        'inList': [input_tensor.name],
        'outList': [out_tensor.name],
        'attrs': {
            'element_size': element_size,
            'memory_config': memory_config_attr,
        },
    }
    opobj = SimOp(opinfo)
    opobj.get_perf_counts([input_tensor], [out_tensor])
    opobj.update_tensor_counts([input_tensor], [out_tensor])
    _propagate_ttnn_dtype([input_tensor], [out_tensor])
    if memory_config is not None:
        out_tensor._memory_config = memory_config
    input_tensor.device.add_op(opobj)
    return out_tensor


def to_memory_config(input_tensor, memory_config=None):
    """Emit STI+ITS SimOps when resharding from one sharded config to another.

    On hardware, transitioning between two sharded layouts goes through an
    interleaved staging step: ShardedToInterleaved → InterleavedToSharded.
    This is emitted when both source and target are sharded but differ.
    """
    if memory_config is None:
        return input_tensor
    mode = get_execution_mode()
    should_track = (mode == ExecutionMode.TRACK_ONLY or mode == ExecutionMode.EXECUTE_AND_TRACK)
    if (should_track and memory_config.is_sharded()
            and isinstance(input_tensor, Tensor) and input_tensor.device is not None):
        input_mc = getattr(input_tensor, '_memory_config', None)
        if input_mc is not None and input_mc.is_sharded() and input_mc != memory_config:
            elem_sz = input_tensor.element_size() if hasattr(input_tensor, 'element_size') else 2
            input_tensor = sharded_to_interleaved_op(input_tensor, element_size=elem_sz)
            input_tensor = interleaved_to_sharded_op(input_tensor, memory_config=memory_config, element_size=elem_sz)
            return input_tensor
    if hasattr(input_tensor, '_memory_config'):
        input_tensor._memory_config = memory_config
    return input_tensor


def reshard(input_tensor, memory_config, output_tensor=None):
    """Change shard layout of an already-sharded tensor."""
    mode = get_execution_mode()
    should_track = (mode == ExecutionMode.TRACK_ONLY or mode == ExecutionMode.EXECUTE_AND_TRACK)
    should_execute = (mode == ExecutionMode.EXECUTE or mode == ExecutionMode.EXECUTE_AND_TRACK)

    if should_track and input_tensor.device is not None and isinstance(input_tensor, Tensor):
        elem_sz = input_tensor.element_size() if hasattr(input_tensor, 'element_size') else 2
        return reshard_op(input_tensor, memory_config=memory_config, element_size=elem_sz)

    if should_track:
        elem_sz = input_tensor.element_size() if hasattr(input_tensor, 'element_size') else 2
        _tracker.track_reshard(
            input_tensor.logical_shape()._shape,
            element_size=elem_sz,
        )

    output_data = None
    if should_execute and hasattr(input_tensor, 'has_data') and input_tensor.has_data():
        output_data = input_tensor.get_data()

    if output_tensor is not None:
        return output_tensor

    # Preserve compact TTNN dtype
    ttnn_dtype = getattr(input_tensor, "_ttnn_dtype", None)
    dtype_arg = ttnn_dtype if ttnn_dtype is not None else input_tensor.dtype

    return type(input_tensor)(
        shape=input_tensor.logical_shape(),
        dtype=dtype_arg,
        layout=input_tensor.get_layout(),
        memory_config=memory_config,
        padded_shape=input_tensor.padded_shape(),
        device=input_tensor.device,
        data=output_data,
    )


# =============================================================================
# Transformer head operator shims + SimOp graph builders
# =============================================================================

def nlp_concat_heads_op(input_tensor, memory_config=None, element_size=2):
    """Create a ConcatHeads SimOp and execute the operation if in EXECUTE mode.

    Input: [B, num_heads, S, head_dim] -> Output: [B, S, num_heads*head_dim]

    NOTE: HW emits a 4D output [B, 1, S, num_heads*head_dim] with
    seq_groups=1 in the Z position, then implicitly reinterprets it as
    [1, B, S, H] (W=1, Z=batch) for downstream ops. TT-Polaris currently
    emits 3D [B, S, H]. The comparison tool (compare_layers.py
    --strip-singleton-dims) normalizes this difference. Future work:
    emit 4D shapes and model the implicit view change.
    """
    assert input_tensor.device is not None, "nlp_concat_heads_op requires input_tensor on device"
    in_shape = input_tensor.logical_shape()._shape
    assert len(in_shape) == 4, f"ConcatHeads expects rank-4 input, got {len(in_shape)}"
    B, num_heads, S, head_dim = in_shape
    out_shape_list = [B, S, num_heads * head_dim]

    op_name = generate_new_op_name()
    out_tensor = Tensor(
        name=op_name + '.out',
        shape=out_shape_list,
        dtype=input_tensor.dtype,
        layout=input_tensor.get_layout(),
        op_out=[op_name],
        device=input_tensor.device,
    )
    input_tensor.op_in.append(op_name)
    opinfo = {
        'name': op_name,
        'optype': 'ConcatHeads',
        'inList': [input_tensor.name],
        'outList': [out_tensor.name],
        'attrs': {'element_size': element_size},
    }
    opobj = SimOp(opinfo)
    opobj.get_perf_counts([input_tensor], [out_tensor])
    opobj.update_tensor_counts([input_tensor], [out_tensor])
    _propagate_ttnn_dtype([input_tensor], [out_tensor])

    # nlp_concat_heads always produces L1_BLOCK_SHARDED output on hardware regardless of input.
    out_tensor._memory_config = memory_config if memory_config is not None else MemoryConfig(
        TensorMemoryLayout.BLOCK_SHARDED, BufferType.L1,
    )

    input_tensor.device.add_op(opobj)

    # Execute the computation if in EXECUTE mode and input has data
    mode = get_execution_mode()
    should_execute = (mode == ExecutionMode.EXECUTE or mode == ExecutionMode.EXECUTE_AND_TRACK)
    if should_execute and hasattr(input_tensor, 'has_data') and input_tensor.has_data():
        import numpy as _np
        data = input_tensor.get_data()
        arr = _np.array(data).reshape(in_shape)
        arr = arr.transpose(0, 2, 1, 3).reshape(out_shape_list)
        out_tensor.data = arr

    return out_tensor


def nlp_concat_heads(input_tensor, memory_config=None):
    """Concatenate attention heads: [B, num_heads, S, head_dim] -> [B, S, num_heads*head_dim]."""
    # When a device is present, delegate to the SimOp-based _op variant so that
    # the operation appears in the device graph (matching HW profiler traces).
    # The tracker-only path below is used for lightweight shape-tracking without
    # a device graph.
    if input_tensor.device is not None:
        elem_sz = input_tensor.element_size() if hasattr(input_tensor, 'element_size') else 2
        return nlp_concat_heads_op(input_tensor, memory_config=memory_config, element_size=elem_sz)

    in_shape = input_tensor.logical_shape()._shape
    assert len(in_shape) == 4, f"ConcatHeads expects rank-4 input, got {len(in_shape)}"
    B, num_heads, S, head_dim = in_shape
    out_shape_list = [B, S, num_heads * head_dim]

    mode = get_execution_mode()
    should_track = (mode == ExecutionMode.TRACK_ONLY or mode == ExecutionMode.EXECUTE_AND_TRACK)
    should_execute = (mode == ExecutionMode.EXECUTE or mode == ExecutionMode.EXECUTE_AND_TRACK)

    output_data = None
    if should_execute and hasattr(input_tensor, 'has_data') and input_tensor.has_data():
        import numpy as _np
        data = input_tensor.get_data()
        arr = _np.array(data).reshape(in_shape)
        arr = arr.transpose(0, 2, 1, 3).reshape(out_shape_list)
        output_data = arr.flatten().tolist()

    if should_track:
        elem_sz = input_tensor.element_size() if hasattr(input_tensor, 'element_size') else 2
        _tracker.track_nlp_concat_heads(
            in_shape, out_shape_list,
            element_size=elem_sz,
        )

    # Preserve compact TTNN dtype
    ttnn_dtype = getattr(input_tensor, "_ttnn_dtype", None)
    dtype_arg = ttnn_dtype if ttnn_dtype is not None else input_tensor.dtype

    return type(input_tensor)(
        shape=Shape(out_shape_list),
        dtype=dtype_arg,
        layout=input_tensor.get_layout(),
        memory_config=memory_config or input_tensor.memory_config(),
        device=input_tensor.device,
        data=output_data,
    )


def nlp_create_qkv_heads_op(input_tensor, kv_input_tensor=None, *,
                              num_heads, num_kv_heads=None,
                              transpose_k_heads=False, memory_config=None,
                              element_size=2):
    """Create a CreateQKVHeads SimOp with 3 outputs and execute the operation if in EXECUTE mode.

    Input: [B, S, (num_heads + 2*num_kv_heads) * head_dim]
    Outputs: Q=[B, num_heads, S, head_dim],
             K=[B, num_kv_heads, head_dim, S] if transpose_k_heads else [B, num_kv_heads, S, head_dim],
             V=[B, num_kv_heads, S, head_dim]

    NOTE: HW uses 4D WZYX shapes with a seq_groups=1 singleton dim:
    input [B, 1, S, hidden], output [B, heads, S, head_dim]. TT-Polaris currently
    emits 3D shapes [B, S, hidden] for the input. The comparison tool
    (compare_layers.py --strip-singleton-dims) normalizes this difference.
    Future work: emit 4D shapes here and propagate the implicit view change
    [B, 1, S, H] -> [1, B, S, H] that HW performs between ConcatHeads
    and downstream ops (Reshard, MatMul).
    """
    assert input_tensor.device is not None, "nlp_create_qkv_heads_op requires input_tensor on device"
    if num_kv_heads is None:
        num_kv_heads = num_heads

    in_shape = input_tensor.logical_shape()._shape
    hidden_size = in_shape[-1]
    if kv_input_tensor is not None:
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"nlp_create_qkv_heads_op requires input hidden size {hidden_size} "
                f"to be divisible by num_heads ({num_heads}) when kv_input_tensor is provided"
            )
        head_dim = hidden_size // num_heads
    else:
        total_qkv_heads = num_heads + 2 * num_kv_heads
        if hidden_size % total_qkv_heads != 0:
            raise ValueError(
                f"nlp_create_qkv_heads_op requires input hidden size {hidden_size} "
                f"to be divisible by num_heads + 2 * num_kv_heads ({total_qkv_heads})"
            )
        head_dim = hidden_size // total_qkv_heads

    # Handle both 3D [B, S, D] and 4D [B, seq_groups, seq_len, D] inputs
    if len(in_shape) == 4:
        B, seq_groups, seq_len, _ = in_shape
        S = seq_groups * seq_len
    elif len(in_shape) >= 3:
        B = in_shape[0]
        S = in_shape[-2]
    else:
        B = 1
        S = in_shape[0]
    q_shape = [B, num_heads, S, head_dim]
    # HW returns K pre-transposed ([B, heads, head_dim, S]) when
    # transpose_k_heads=True so Q @ K needs no extra Transpose op.
    k_shape = [B, num_kv_heads, head_dim, S] if transpose_k_heads else [B, num_kv_heads, S, head_dim]
    v_shape = [B, num_kv_heads, S, head_dim]

    op_name = generate_new_op_name()
    q_tensor = Tensor(
        name=f"{op_name}.out.0", shape=q_shape, dtype=input_tensor.dtype,
        layout=input_tensor.get_layout(), op_out=[op_name], device=input_tensor.device,
    )
    k_tensor = Tensor(
        name=f"{op_name}.out.1", shape=k_shape, dtype=input_tensor.dtype,
        layout=input_tensor.get_layout(), op_out=[op_name], device=input_tensor.device,
    )
    v_tensor = Tensor(
        name=f"{op_name}.out.2", shape=v_shape, dtype=input_tensor.dtype,
        layout=input_tensor.get_layout(), op_out=[op_name], device=input_tensor.device,
    )

    input_tensor.op_in.append(op_name)
    in_list = [input_tensor.name]
    in_tensors = [input_tensor]
    if kv_input_tensor is not None:
        kv_input_tensor.op_in.append(op_name)
        in_list.append(kv_input_tensor.name)
        in_tensors.append(kv_input_tensor)

    opinfo = {
        'name': op_name,
        'optype': 'CreateQKVHeads',
        'inList': in_list,
        'outList': [q_tensor.name, k_tensor.name, v_tensor.name],
        'attrs': {
            'num_heads': num_heads,
            'num_kv_heads': num_kv_heads,
            'head_dim': head_dim,
            'transpose_k_heads': transpose_k_heads,
            'element_size': element_size,
        },
    }
    opobj = SimOp(opinfo)
    out_tensors = [q_tensor, k_tensor, v_tensor]
    opobj.get_perf_counts(in_tensors, out_tensors)
    opobj.update_tensor_counts(in_tensors, out_tensors)
    _propagate_ttnn_dtype(in_tensors, out_tensors)

    out_mc = memory_config if memory_config is not None else MemoryConfig(
        TensorMemoryLayout.HEIGHT_SHARDED, BufferType.L1,
    )
    for t in out_tensors:
        t._memory_config = out_mc

    input_tensor.device.add_op(opobj)

    # Execute the computation if in EXECUTE mode and input has data
    mode = get_execution_mode()
    should_execute = (mode == ExecutionMode.EXECUTE or mode == ExecutionMode.EXECUTE_AND_TRACK)
    if should_execute and hasattr(input_tensor, 'has_data') and input_tensor.has_data():
        import numpy as _np
        data = input_tensor.get_data()
        if kv_input_tensor is not None:
            q_arr_3d = _np.array(data).reshape(B, S, num_heads * head_dim)
            q_arr = q_arr_3d.reshape(B, S, num_heads, head_dim).transpose(0, 2, 1, 3)
            q_tensor.data = q_arr
            if hasattr(kv_input_tensor, 'has_data') and kv_input_tensor.has_data():
                kv_data = _np.array(kv_input_tensor.get_data()).reshape(B, S, 2 * num_kv_heads * head_dim)
                k_arr = kv_data[:, :, :num_kv_heads * head_dim].reshape(B, S, num_kv_heads, head_dim).transpose(0, 2, 1, 3)
                v_arr = kv_data[:, :, num_kv_heads * head_dim:].reshape(B, S, num_kv_heads, head_dim).transpose(0, 2, 1, 3)
                # Physically transpose K data to match shape [B, heads, head_dim, S]
                if transpose_k_heads:
                    k_arr = k_arr.transpose(0, 1, 3, 2)
                k_tensor.data = k_arr
                v_tensor.data = v_arr
        else:
            fused_arr = _np.array(data).reshape(B, S, -1)
            q_end = num_heads * head_dim
            k_end = q_end + num_kv_heads * head_dim
            q_arr = fused_arr[:, :, :q_end].reshape(B, S, num_heads, head_dim).transpose(0, 2, 1, 3)
            k_arr = fused_arr[:, :, q_end:k_end].reshape(B, S, num_kv_heads, head_dim).transpose(0, 2, 1, 3)
            v_arr = fused_arr[:, :, k_end:].reshape(B, S, num_kv_heads, head_dim).transpose(0, 2, 1, 3)
            # Physically transpose K data to match shape [B, heads, head_dim, S]
            if transpose_k_heads:
                k_arr = k_arr.transpose(0, 1, 3, 2)
            q_tensor.data = q_arr
            k_tensor.data = k_arr
            v_tensor.data = v_arr

    return q_tensor, k_tensor, v_tensor


def nlp_create_qkv_heads(input_tensor, kv_input_tensor=None, *,
                           num_heads, num_kv_heads=None,
                           transpose_k_heads=False, memory_config=None):
    """Split fused QKV tensor into separate Q, K, V head tensors.

    Input: [B, S, (num_heads + 2*num_kv_heads) * head_dim]
    Returns: (Q, K, V) where Q=[B, num_heads, S, head_dim],
             K=[B, num_kv_heads, head_dim, S] if transpose_k_heads else [B, num_kv_heads, S, head_dim],
             V=[B, num_kv_heads, S, head_dim]
    """
    # When a device is present, delegate to the SimOp-based _op variant so that
    # the operation appears in the device graph (matching HW profiler traces).
    if input_tensor.device is not None:
        elem_sz = input_tensor.element_size() if hasattr(input_tensor, 'element_size') else 2
        return nlp_create_qkv_heads_op(
            input_tensor, kv_input_tensor,
            num_heads=num_heads, num_kv_heads=num_kv_heads,
            transpose_k_heads=transpose_k_heads, memory_config=memory_config,
            element_size=elem_sz,
        )

    if num_kv_heads is None:
        num_kv_heads = num_heads

    in_shape = input_tensor.logical_shape()._shape
    hidden_size = in_shape[-1]
    if kv_input_tensor is not None:
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"nlp_create_qkv_heads requires input hidden size {hidden_size} "
                f"to be divisible by num_heads ({num_heads}) when kv_input_tensor is provided"
            )
        head_dim = hidden_size // num_heads
    else:
        total_qkv_heads = num_heads + 2 * num_kv_heads
        if hidden_size % total_qkv_heads != 0:
            raise ValueError(
                f"nlp_create_qkv_heads requires input hidden size {hidden_size} "
                f"to be divisible by num_heads + 2 * num_kv_heads ({total_qkv_heads})"
            )
        head_dim = hidden_size // total_qkv_heads

    # Handle both 3D [B, S, D] and 4D [B, seq_groups, seq_len, D] inputs
    if len(in_shape) == 4:
        B, seq_groups, seq_len, _ = in_shape
        S = seq_groups * seq_len
    elif len(in_shape) >= 3:
        B = in_shape[0]
        S = in_shape[-2]
    else:
        B = 1
        S = in_shape[0]
    q_shape = [B, num_heads, S, head_dim]
    k_shape = [B, num_kv_heads, head_dim, S] if transpose_k_heads else [B, num_kv_heads, S, head_dim]
    v_shape = [B, num_kv_heads, S, head_dim]

    mode = get_execution_mode()
    should_track = (mode == ExecutionMode.TRACK_ONLY or mode == ExecutionMode.EXECUTE_AND_TRACK)
    should_execute = (mode == ExecutionMode.EXECUTE or mode == ExecutionMode.EXECUTE_AND_TRACK)

    if should_track:
        elem_sz = input_tensor.element_size() if hasattr(input_tensor, 'element_size') else 2
        _tracker.track_nlp_create_qkv_heads(
            in_shape, q_shape, k_shape, v_shape,
            element_size=elem_sz,
        )

    # Execute the computation if in EXECUTE mode and input has data
    q_data = k_data = v_data = None
    if should_execute and hasattr(input_tensor, 'has_data') and input_tensor.has_data():
        import numpy as _np
        data = input_tensor.get_data()
        if kv_input_tensor is not None:
            q_arr_3d = _np.array(data).reshape(B, S, num_heads * head_dim)
            q_arr = q_arr_3d.reshape(B, S, num_heads, head_dim).transpose(0, 2, 1, 3)
            q_data = q_arr.flatten().tolist()
            if hasattr(kv_input_tensor, 'has_data') and kv_input_tensor.has_data():
                kv_data = _np.array(kv_input_tensor.get_data()).reshape(B, S, 2 * num_kv_heads * head_dim)
                k_arr = kv_data[:, :, :num_kv_heads * head_dim].reshape(B, S, num_kv_heads, head_dim).transpose(0, 2, 1, 3)
                v_arr = kv_data[:, :, num_kv_heads * head_dim:].reshape(B, S, num_kv_heads, head_dim).transpose(0, 2, 1, 3)
                # Physically transpose K data to match shape [B, heads, head_dim, S]
                if transpose_k_heads:
                    k_arr = k_arr.transpose(0, 1, 3, 2)
                k_data = k_arr.flatten().tolist()
                v_data = v_arr.flatten().tolist()
        else:
            fused_arr = _np.array(data).reshape(B, S, -1)
            q_end = num_heads * head_dim
            k_end = q_end + num_kv_heads * head_dim
            q_arr = fused_arr[:, :, :q_end].reshape(B, S, num_heads, head_dim).transpose(0, 2, 1, 3)
            k_arr = fused_arr[:, :, q_end:k_end].reshape(B, S, num_kv_heads, head_dim).transpose(0, 2, 1, 3)
            v_arr = fused_arr[:, :, k_end:].reshape(B, S, num_kv_heads, head_dim).transpose(0, 2, 1, 3)
            # Physically transpose K data to match shape [B, heads, head_dim, S]
            if transpose_k_heads:
                k_arr = k_arr.transpose(0, 1, 3, 2)
            q_data = q_arr.flatten().tolist()
            k_data = k_arr.flatten().tolist()
            v_data = v_arr.flatten().tolist()

    # Preserve compact TTNN dtype
    ttnn_dtype = getattr(input_tensor, "_ttnn_dtype", None)
    dtype_arg = ttnn_dtype if ttnn_dtype is not None else input_tensor.dtype

    TensorType = type(input_tensor)
    q = TensorType(
        shape=Shape(q_shape), dtype=dtype_arg,
        layout=input_tensor.get_layout(),
        memory_config=memory_config or input_tensor.memory_config(),
        device=input_tensor.device, data=q_data,
    )
    k = TensorType(
        shape=Shape(k_shape), dtype=dtype_arg,
        layout=input_tensor.get_layout(),
        memory_config=memory_config or input_tensor.memory_config(),
        device=input_tensor.device, data=k_data,
    )
    v = TensorType(
        shape=Shape(v_shape), dtype=dtype_arg,
        layout=input_tensor.get_layout(),
        memory_config=memory_config or input_tensor.memory_config(),
        device=input_tensor.device, data=v_data,
    )
    return q, k, v


# ---------------------------------------------------------------------------
# Llama3 transformer device ops (rotary embedding, KV cache, head split/concat,
# SDPA). Tracking-only: each emits a single SimOp matching the HW profiler trace
# and returns synthetic output Tensor(s) whose shapes mirror tt-metal semantics.
# ---------------------------------------------------------------------------

def rotary_embedding_llama_op(x, cos, sin, trans_mat, is_decode_mode=False,
                               memory_config=None, element_size=2):
    """RotaryEmbeddingLlama (prefill): apply rotary embedding to x; output shape = x.

    Called separately for q and k in tt-metal. cos/sin/trans_mat are auxiliary
    inputs recorded on the op but do not change the output shape.
    """
    assert x.device is not None, "rotary_embedding_llama_op requires x on device"
    op_name = generate_new_op_name()
    in_tensors = [t for t in (x, cos, sin, trans_mat) if t is not None]
    out_tensor = Tensor(
        name=op_name + '.out',
        shape=x.logical_shape()._shape,
        dtype=x.dtype,
        layout=x.get_layout(),
        op_out=[op_name],
        device=x.device,
    )
    for t in in_tensors:
        t.op_in.append(op_name)
    opinfo = {
        'name': op_name,
        'optype': 'RotaryEmbeddingLlama',
        'inList': [t.name for t in in_tensors],
        'outList': [out_tensor.name],
        'attrs': {'is_decode_mode': bool(is_decode_mode), 'element_size': element_size},
    }
    opobj = SimOp(opinfo)
    opobj.get_perf_counts(in_tensors, [out_tensor])
    opobj.update_tensor_counts(in_tensors, [out_tensor])
    _propagate_ttnn_dtype([x], [out_tensor])
    out_tensor._memory_config = memory_config if memory_config is not None else getattr(x, '_memory_config', None)
    x.device.add_op(opobj)
    return out_tensor


def rotary_embedding_llama_fused_qk_op(q, k, cos, sin, trans_mat,
                                        memory_config=None, element_size=2):
    """RotaryEmbeddingLlamaFusedQK (decode): rotary on q and k in one op.

    Returns (q_out, k_out), each with the shape of its corresponding input.
    """
    assert q.device is not None, "rotary_embedding_llama_fused_qk_op requires q on device"
    op_name = generate_new_op_name()
    in_tensors = [t for t in (q, k, cos, sin, trans_mat) if t is not None]
    q_out = Tensor(
        name=op_name + '.out.0', shape=q.logical_shape()._shape, dtype=q.dtype,
        layout=q.get_layout(), op_out=[op_name], device=q.device,
    )
    k_out = Tensor(
        name=op_name + '.out.1', shape=k.logical_shape()._shape, dtype=k.dtype,
        layout=k.get_layout(), op_out=[op_name], device=q.device,
    )
    for t in in_tensors:
        t.op_in.append(op_name)
    opinfo = {
        'name': op_name,
        'optype': 'RotaryEmbeddingLlamaFusedQK',
        'inList': [t.name for t in in_tensors],
        'outList': [q_out.name, k_out.name],
        'attrs': {'element_size': element_size},
    }
    opobj = SimOp(opinfo)
    out_tensors = [q_out, k_out]
    opobj.get_perf_counts(in_tensors, out_tensors)
    opobj.update_tensor_counts(in_tensors, out_tensors)
    _propagate_ttnn_dtype([q], [q_out])
    _propagate_ttnn_dtype([k], [k_out])
    if memory_config is not None:
        q_out._memory_config = memory_config
        k_out._memory_config = memory_config
    else:
        q_out._memory_config = getattr(q, '_memory_config', None)
        k_out._memory_config = getattr(k, '_memory_config', None)
    q.device.add_op(opobj)
    return q_out, k_out


def paged_fill_cache_op(cache, input_tensor, page_table=None, batch_idx=0,
                         memory_config=None, element_size=2):
    """PagedFillCache (prefill): fill the KV cache in place; output shape = cache.

    The cache tensor is updated in place on hardware; we emit a passthrough
    SimOp and return a tensor with the cache's shape so the graph edge exists.
    """
    assert cache.device is not None, "paged_fill_cache_op requires cache on device"
    op_name = generate_new_op_name()
    in_tensors = [t for t in (cache, input_tensor, page_table) if t is not None]
    out_tensor = Tensor(
        name=op_name + '.out',
        shape=cache.logical_shape()._shape,
        dtype=cache.dtype,
        layout=cache.get_layout(),
        op_out=[op_name],
        device=cache.device,
    )
    for t in in_tensors:
        t.op_in.append(op_name)
    opinfo = {
        'name': op_name,
        'optype': 'PagedFillCache',
        'inList': [t.name for t in in_tensors],
        'outList': [out_tensor.name],
        'attrs': {'batch_idx': batch_idx, 'element_size': element_size},
    }
    opobj = SimOp(opinfo)
    opobj.get_perf_counts(in_tensors, [out_tensor])
    opobj.update_tensor_counts(in_tensors, [out_tensor])
    _propagate_ttnn_dtype([cache], [out_tensor])
    out_tensor._memory_config = memory_config if memory_config is not None else getattr(cache, '_memory_config', None)
    cache.device.add_op(opobj)
    return out_tensor


def manual_seed_op(seeds, user_ids=None, sub_core_grids=None, element_size=2):
    """ManualSeed: seed the on-device RNG for sampling (decode). Passthrough SimOp;
    output shape = seeds so the graph edge exists."""
    assert seeds.device is not None, "manual_seed_op requires seeds on device"
    op_name = generate_new_op_name()
    in_tensors = [t for t in (seeds, user_ids) if t is not None]
    out_tensor = Tensor(
        name=op_name + '.out',
        shape=seeds.logical_shape()._shape,
        dtype=seeds.dtype,
        layout=seeds.get_layout(),
        op_out=[op_name],
        device=seeds.device,
    )
    for t in in_tensors:
        t.op_in.append(op_name)
    opinfo = {
        'name': op_name,
        'optype': 'ManualSeed',
        'inList': [t.name for t in in_tensors],
        'outList': [out_tensor.name],
        'attrs': {'element_size': element_size},
    }
    opobj = SimOp(opinfo)
    opobj.get_perf_counts(in_tensors, [out_tensor])
    opobj.update_tensor_counts(in_tensors, [out_tensor])
    _propagate_ttnn_dtype([seeds], [out_tensor])
    seeds.device.add_op(opobj)
    return out_tensor


def sampling_op(topk_values, topk_indices, k=None, p=None, temp=None,
                output_tensor=None, sub_core_grids=None, memory_config=None, element_size=2):
    """Sampling (decode): sample one token per user from gathered top-k values/indices.

    Inputs: topk_values, topk_indices (+ optional k/p/temp param tensors). Output =
    output_tensor's shape when preallocated, else topk_values with last dim -> 1."""
    assert topk_values.device is not None, "sampling_op requires tensors on device"
    op_name = generate_new_op_name()
    # LUT-miss fix (arity): the HW SamplingDeviceOperation is arity-6 — the preallocated
    # output_tensor is logged by the profiler as input_5 (tt-metal tt_sampling.py passes
    # output_tensor=tt_out_tok). Record it as the 6th operand so the sim key matches the capture.
    in_tensors = [t for t in (topk_values, topk_indices, k, p, temp, output_tensor) if t is not None]
    if output_tensor is not None:
        out_shape = list(output_tensor.logical_shape()._shape)
        out_dtype = output_tensor.dtype
    else:
        out_shape = list(topk_values.logical_shape()._shape[:-1]) + [1]
        out_dtype = topk_indices.dtype
    out_tensor = Tensor(
        name=op_name + '.out',
        shape=out_shape,
        dtype=out_dtype,
        layout=topk_indices.get_layout(),
        op_out=[op_name],
        device=topk_values.device,
    )
    for t in in_tensors:
        t.op_in.append(op_name)
    opinfo = {
        'name': op_name,
        'optype': 'Sampling',
        'inList': [t.name for t in in_tensors],
        'outList': [out_tensor.name],
        'attrs': {'element_size': element_size},
    }
    opobj = SimOp(opinfo)
    opobj.get_perf_counts(in_tensors, [out_tensor])
    opobj.update_tensor_counts(in_tensors, [out_tensor])
    _propagate_ttnn_dtype([topk_indices], [out_tensor])
    topk_values.device.add_op(opobj)
    return out_tensor


def paged_fused_update_cache_op(k_cache, k, v_cache, v, update_idxs_tensor=None,
                                 page_table=None, memory_config=None, element_size=2):
    """PagedFusedUpdateCache (decode): update K and V caches in place.

    Returns (k_cache_out, v_cache_out), each with its cache input's shape.
    """
    assert k_cache.device is not None, "paged_fused_update_cache_op requires k_cache on device"
    op_name = generate_new_op_name()
    in_tensors = [t for t in (k_cache, k, v_cache, v, update_idxs_tensor, page_table) if t is not None]
    k_out = Tensor(
        name=op_name + '.out.0', shape=k_cache.logical_shape()._shape, dtype=k_cache.dtype,
        layout=k_cache.get_layout(), op_out=[op_name], device=k_cache.device,
    )
    v_out = Tensor(
        name=op_name + '.out.1', shape=v_cache.logical_shape()._shape, dtype=v_cache.dtype,
        layout=v_cache.get_layout(), op_out=[op_name], device=k_cache.device,
    )
    for t in in_tensors:
        t.op_in.append(op_name)
    opinfo = {
        'name': op_name,
        'optype': 'PagedFusedUpdateCache',
        'inList': [t.name for t in in_tensors],
        'outList': [k_out.name, v_out.name],
        'attrs': {'element_size': element_size},
    }
    opobj = SimOp(opinfo)
    out_tensors = [k_out, v_out]
    opobj.get_perf_counts(in_tensors, out_tensors)
    opobj.update_tensor_counts(in_tensors, out_tensors)
    _propagate_ttnn_dtype([k_cache], [k_out])
    _propagate_ttnn_dtype([v_cache], [v_out])
    if memory_config is not None:
        k_out._memory_config = memory_config
        v_out._memory_config = memory_config
    else:
        k_out._memory_config = getattr(k_cache, '_memory_config', None)
        v_out._memory_config = getattr(v_cache, '_memory_config', None)
    k_cache.device.add_op(opobj)
    return k_out, v_out


def nlp_create_qkv_heads_decode_op(input_tensor, *, num_heads, num_kv_heads=None,
                                    head_dim=None, memory_config=None, element_size=2):
    """NLPCreateQKVHeadsDecode (decode): split fused QKV into Q, K, V.

    Decode WZYX convention: input [..., 1, hidden] -> heads on the Y axis, head_dim on X:
      Q = [..lead.., num_heads, head_dim]
      K = V = [..lead.., num_kv_heads, head_dim]
    where lead is the leading dims (typically [1, 1]).
    """
    assert input_tensor.device is not None, "nlp_create_qkv_heads_decode_op requires input on device"
    if num_kv_heads is None:
        num_kv_heads = num_heads
    in_shape = input_tensor.logical_shape()._shape
    hidden = in_shape[-1]
    if head_dim is None:
        total_heads = num_heads + 2 * num_kv_heads
        if hidden % total_heads != 0:
            raise ValueError(
                f"nlp_create_qkv_heads_decode_op: hidden {hidden} not divisible by "
                f"num_heads + 2*num_kv_heads ({total_heads})"
            )
        head_dim = hidden // total_heads

    lead = list(in_shape[:-2]) if len(in_shape) >= 2 else [1, 1]
    while len(lead) < 2:
        lead = [1] + lead
    q_shape = lead + [num_heads, head_dim]
    k_shape = lead + [num_kv_heads, head_dim]
    v_shape = lead + [num_kv_heads, head_dim]

    op_name = generate_new_op_name()
    q_tensor = Tensor(name=f"{op_name}.out.0", shape=q_shape, dtype=input_tensor.dtype,
                      layout=input_tensor.get_layout(), op_out=[op_name], device=input_tensor.device)
    k_tensor = Tensor(name=f"{op_name}.out.1", shape=k_shape, dtype=input_tensor.dtype,
                      layout=input_tensor.get_layout(), op_out=[op_name], device=input_tensor.device)
    v_tensor = Tensor(name=f"{op_name}.out.2", shape=v_shape, dtype=input_tensor.dtype,
                      layout=input_tensor.get_layout(), op_out=[op_name], device=input_tensor.device)
    input_tensor.op_in.append(op_name)
    opinfo = {
        'name': op_name,
        'optype': 'NLPCreateQKVHeadsDecode',
        'inList': [input_tensor.name],
        'outList': [q_tensor.name, k_tensor.name, v_tensor.name],
        'attrs': {
            'num_heads': num_heads,
            'num_kv_heads': num_kv_heads,
            'head_dim': head_dim,
            'element_size': element_size,
        },
    }
    opobj = SimOp(opinfo)
    out_tensors = [q_tensor, k_tensor, v_tensor]
    opobj.get_perf_counts([input_tensor], out_tensors)
    opobj.update_tensor_counts([input_tensor], out_tensors)
    _propagate_ttnn_dtype([input_tensor], out_tensors)
    out_mc = memory_config if memory_config is not None else MemoryConfig(
        TensorMemoryLayout.HEIGHT_SHARDED, BufferType.L1,
    )
    for t in out_tensors:
        t._memory_config = out_mc
    input_tensor.device.add_op(opobj)
    return q_tensor, k_tensor, v_tensor


def nlp_concat_heads_decode_op(input_tensor, num_heads=None, memory_config=None, element_size=2):
    """NLPConcatHeadsDecode (decode): concatenate heads on the X axis.

    Decode WZYX convention: input [..lead.., num_heads, head_dim] (heads on Y) ->
    output [..lead.., num_heads, num_heads*head_dim] (head_dim folded into X).
    """
    assert input_tensor.device is not None, "nlp_concat_heads_decode_op requires input on device"
    in_shape = input_tensor.logical_shape()._shape
    assert len(in_shape) >= 2, f"NLPConcatHeadsDecode expects rank>=2 input, got {len(in_shape)}"
    if num_heads is None:
        num_heads = in_shape[-2]
    head_dim = in_shape[-1]
    out_shape = list(in_shape[:-1]) + [num_heads * head_dim]

    op_name = generate_new_op_name()
    out_tensor = Tensor(
        name=op_name + '.out', shape=out_shape, dtype=input_tensor.dtype,
        layout=input_tensor.get_layout(), op_out=[op_name], device=input_tensor.device,
    )
    input_tensor.op_in.append(op_name)
    opinfo = {
        'name': op_name,
        'optype': 'NLPConcatHeadsDecode',
        'inList': [input_tensor.name],
        'outList': [out_tensor.name],
        'attrs': {'num_heads': num_heads, 'element_size': element_size},
    }
    opobj = SimOp(opinfo)
    opobj.get_perf_counts([input_tensor], [out_tensor])
    opobj.update_tensor_counts([input_tensor], [out_tensor])
    _propagate_ttnn_dtype([input_tensor], [out_tensor])
    out_tensor._memory_config = memory_config if memory_config is not None else MemoryConfig(
        TensorMemoryLayout.WIDTH_SHARDED, BufferType.L1,
    )
    input_tensor.device.add_op(opobj)
    return out_tensor


def scaled_dot_product_attention_op(q, k, v, *extra_inputs, memory_config=None,
                                    element_size=None, **attrs):
    """ScaledDotProductAttention: prefill (q,k,v) or decode/paged (q, k_cache, v_cache, ...).

    Output shape always equals q. Extra positional tensor inputs (cur_pos,
    page_table) are recorded on the op. Non-shape kwargs (scale, is_causal, ...)
    are stored in attrs.
    """
    assert q.device is not None, "scaled_dot_product_attention_op requires q on device"
    op_name = generate_new_op_name()
    in_tensors = [t for t in (q, k, v, *extra_inputs)
                  if t is not None and isinstance(t, Tensor)]
    out_tensor = Tensor(
        name=op_name + '.out',
        shape=q.logical_shape()._shape,
        dtype=q.dtype,
        layout=q.get_layout(),
        op_out=[op_name],
        device=q.device,
    )
    for t in in_tensors:
        t.op_in.append(op_name)
    op_attrs = {k_: v_ for k_, v_ in attrs.items() if isinstance(v_, (int, float, bool, str))}
    # Derive from q so a bfp8/bfp4 op is not silently costed as bfloat16 downstream.
    op_attrs['element_size'] = element_size if element_size is not None else q.element_size()
    opinfo = {
        'name': op_name,
        'optype': 'ScaledDotProductAttention',
        'inList': [t.name for t in in_tensors],
        'outList': [out_tensor.name],
        'attrs': op_attrs,
    }
    opobj = SimOp(opinfo)
    opobj.get_perf_counts(in_tensors, [out_tensor])
    opobj.update_tensor_counts(in_tensors, [out_tensor])
    _propagate_ttnn_dtype([q], [out_tensor])
    # SDPA materializes a fresh output; on HW it is DRAM-interleaved (it does NOT preserve the
    # sharded q layout — cf. tt_transformers attention.py decode: sdpa-decode output is DRAM-
    # interleaved, then interleaved_to_sharded for concat-heads). Inheriting q's (height-sharded)
    # memory mis-tagged the output, so the downstream InterleavedToSharded got a sharded input and
    # missed its LUT key. Default to DRAM-interleaved unless the caller overrides.
    out_tensor._memory_config = memory_config if memory_config is not None else DRAM_MEMORY_CONFIG
    q.device.add_op(opobj)
    return out_tensor
