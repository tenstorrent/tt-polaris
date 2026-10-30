# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""TT-Polaris-side mimic of tt-metal's conv2d parallel-config picker.

Used by the conv-input channel-padding annotation pass (see
``ttsim/back/device.py::Device._annotate_conv_x_pad_logical``). Given a conv's input shape and the
device's logical compute grid, returns the same ``(num_cores_nhw, num_cores_c,
padded_channels)`` that ``determine_parallel_config`` in
``tt-metal/ttnn/cpp/ttnn/operations/conv/conv2d/conv2d_utils.cpp`` would pick.

Only the BLOCK_SHARDED path is implemented for now — that's the layout where
HW pads channels in a way TT-Polaris currently doesn't model.  HEIGHT_SHARDED and
WIDTH_SHARDED are no-ops here (channel padding doesn't apply); callers should
fall through to existing TT-Polaris behavior for those layouts.
"""

from __future__ import annotations

from typing import Tuple

# Mirror tt-metal constants. TILE_WIDTH from tt::constants; input_channels_alignment
# defaults to TILE_WIDTH=32 for the BLOCK_SHARDED case TT-Polaris cares about (see
# tt-metal get_input_channels_alignment fallback for non-WIDTH_SHARDED inputs).
TILE_WIDTH = 32
TILE_HEIGHT = 32
DEFAULT_INPUT_CHANNELS_ALIGNMENT = TILE_WIDTH


def _div_up(a: int, b: int) -> int:
    return (a + b - 1) // b


def _round_up(a: int, b: int) -> int:
    return _div_up(a, b) * b


def find_closest_largest_divisor_with_num_padding(num: int, start_divisor: int) -> int:
    """Port of ``find_closest_largest_divisor_with_num_padding``
    (``tt-metal/ttnn/cpp/ttnn/operations/conv/conv2d/conv2d_utils.cpp:55``).

    Walks divisors downward from ``start_divisor`` until the padding overhead
    becomes less than one shard's worth. Returns the picked divisor (which
    becomes ``num_cores_c`` for the channel axis or ``num_cores_nhw`` for the
    spatial axis under various paths).
    """
    if start_divisor <= 0:
        raise ValueError(f"start_divisor must be positive, got {start_divisor}")
    divisor = start_divisor
    padded_num = _round_up(num, divisor)
    while divisor > 1 and (padded_num - num) >= (padded_num // divisor):
        divisor -= 1
        padded_num = _round_up(num, divisor)
    return divisor


def find_closest_largest_divisor_with_num_padding_and_mult(
    num: int, start_divisor: int, mult: int
) -> int:
    """Port of ``find_closest_largest_divisor_with_num_padding_and_mult``
    (``tt-metal/ttnn/cpp/ttnn/operations/conv/conv2d/conv2d_utils.cpp:65``).
    """
    if start_divisor <= 0:
        raise ValueError(f"start_divisor must be positive, got {start_divisor}")
    if mult <= 0:
        raise ValueError(f"mult must be positive, got {mult}")
    divisor = start_divisor
    big_divisor = divisor * mult
    padded_num = _round_up(num, big_divisor)
    while divisor > 1 and (padded_num - num) >= (padded_num // divisor):
        divisor -= 1
        big_divisor = divisor * mult
        padded_num = _round_up(num, big_divisor)
    return divisor


def determine_block_sharded_channel_padding(
    input_channels: int,
    output_nhw: int,
    compute_grid_size: Tuple[int, int],
    *,
    input_channels_alignment: int = DEFAULT_INPUT_CHANNELS_ALIGNMENT,
    act_block_h_override: int = 0,
    shard_orientation_row_major: bool = True,
) -> Tuple[int, int, int]:
    """Return ``(num_cores_nhw, num_cores_c, padded_input_channels)`` for BLOCK_SHARDED conv2d.

    Mirrors the BLOCK_SHARDED branch of ``determine_parallel_config`` in
    ``tt-metal/ttnn/cpp/ttnn/operations/conv/conv2d/conv2d_utils.cpp:188-204``. Only the *padding* values are returned
    (callers
    don't need the full ``ParallelConfig`` struct yet).

    Args:
        input_channels: Logical channel count of the conv's input tensor (pre-padding).
        output_nhw: ``batch * output_height * output_width`` — the conv's output spatial volume.
        compute_grid_size: ``(x, y)`` logical tensix grid for this SKU (see arch YAML).
        input_channels_alignment: Per-stick channel-byte alignment.  32 (= ``TILE_WIDTH``)
            for the BLOCK_SHARDED path tt-metal takes here.
        act_block_h_override: ``act_block_h_override / TILE_HEIGHT``.  Defaults to 0 → mult=1.
        shard_orientation_row_major: When True (the tt-metal default for our workloads),
            ``start_divisor_c = compute_grid_size.x`` and ``start_divisor_nhw =
            compute_grid_size.y``.

    Returns:
        ``(num_cores_nhw, num_cores_c, padded_input_channels)``.
    """
    if input_channels_alignment <= 0:
        raise ValueError(
            f"input_channels_alignment must be positive, got {input_channels_alignment}"
        )
    grid_x, grid_y = compute_grid_size
    if grid_x <= 0 or grid_y <= 0:
        raise ValueError(f"compute_grid_size must be positive, got {compute_grid_size}")

    out_nhw_ntiles = _div_up(output_nhw, TILE_HEIGHT)
    act_block_h_override_ntiles = (
        1 if act_block_h_override == 0 else act_block_h_override // TILE_HEIGHT
    )
    input_channels_blocks = _div_up(input_channels, input_channels_alignment)

    start_divisor_nhw = grid_y if shard_orientation_row_major else grid_x
    start_divisor_c = grid_x if shard_orientation_row_major else grid_y

    num_cores_nhw = find_closest_largest_divisor_with_num_padding_and_mult(
        out_nhw_ntiles, start_divisor_nhw, act_block_h_override_ntiles
    )
    num_cores_c = find_closest_largest_divisor_with_num_padding(
        input_channels_blocks, start_divisor_c
    )

    padded_blocks = _round_up(input_channels_blocks, num_cores_c)
    padded_input_channels = padded_blocks * input_channels_alignment
    return num_cores_nhw, num_cores_c, padded_input_channels
