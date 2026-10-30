#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The eight shim gates this workload opts into, in one place.

Both arch infras — ``bh/resnet50_test_infra_polaris_bh.py`` and
``wh/resnet50_test_infra_polaris_wh.py`` — enabled these with an identical
50-line block.  Identical literally: the only line that differed between the two
copies was a tt-metal path citation reading ``blackhole/`` in one and
``wormhole/`` in the other.  Adding a ninth gate meant remembering two files,
and the two could drift without anything failing — a gate enabled on one arch
and not the other produces a plausible projection on both.

What stays per-arch, deliberately: the arch name, the capture SKU and the
compute grid.  Those are the values that genuinely differ, and they belong in
the file for the arch they describe.  This module holds only what is the same.

Every gate is OFF shim-wide by default and each is enabled against a specific
Device, so nothing here leaks into another workload built in the same process.
The comments record what each gate buys measured against the capture, because
a gate flipped off produces a working run with quietly wrong numbers rather
than a failure.

See ``ttsim/utils/shim_gates.py`` for the gate definitions and their
``TTSIM_*`` environment variables.
"""

from typing import Any

from ttsim.utils.shim_gates import (
    set_bcast_hw_shape_propagation,
    set_conv_input_reshard,
    set_conv_math_fidelity_in_key,
    set_halo_output_dtype_promotion,
    set_memory_config_propagation,
    set_move_on_reallocate_halo_output,
    set_tilize_before_matmul,
    set_to_memory_config_reshard,
)


def enable_resnet50_shim_gates(device: Any) -> None:
    """Enable the eight gates ResNet-50 needs, on ``device``.

    Measured on BH p100a; the WH n150 figures differ in magnitude but not in
    direction.  Each count below is ops/hits against the arch's own capture.
    """
    # Opt in to conv-output memory-config propagation (default OFF shim-wide).
    # Canonical's bottleneck reshard is
    #   if ds_out.memory_config() != out.memory_config(): to_memory_config(...)
    # (tt-metal/models/demos/vision/classification/resnet50/ttnn_resnet/tt/
    # ttnn_functional_resnet50.py:325-326), and the refrun shows a Reshard at
    # that position in every downsampling bottleneck.  Without propagation both
    # sides read None, the condition is always False, and polaris emits none.
    #
    # LOAD-BEARING: removing this opt-in takes p100a from 115 ops / 115 hits to
    # 118 ops / 114 hits.  The conv passes `device` inside `conv_kwargs`
    # (ttnn_functional_resnet50.py:586), so the gate resolves this Device and
    # its override, not the process-wide default.
    set_memory_config_propagation(True, device)

    # Opt in to the ROW_MAJOR -> TILE conversion before a lowered 1x1 matmul
    # (default OFF shim-wide).  The p100a capture records it at rows 12 and 19,
    # tilizing the maxpool's ROW_MAJOR output ahead of the first two matmuls;
    # without this polaris emits 0 Tilize against hardware's 2.
    set_tilize_before_matmul(True, device)

    # Opt in to hw_shape propagation across the residual Adds (default OFF).
    # Without it every bottleneck downstream of an Add keys on logical NCHW
    # instead of the (1, 1, N*H*W, C) the profiler records.
    set_bcast_hw_shape_propagation(True, device)

    # Opt in to carrying MathFidelity.LoFi into the 1x1-matmul keys (default
    # OFF).  The p100a capture keys every matmul under 'LoFi'; without this all
    # 33 lowered matmuls key 'N/A' and miss.
    set_conv_math_fidelity_in_key(True, device)

    # HaloDeviceOperation promotes its output to BFLOAT16 (tt-metal/ttnn/cpp/
    # ttnn/operations/sliding_window/halo/device/halo_device_operation.cpp:66-70);
    # without it every downstream conv2d keys on BFLOAT8_B while the capture
    # records BFLOAT16.
    set_halo_output_dtype_promotion(True, device)

    # reshard_if_not_optimal forces an input Reshard (+Move) in tt-metal/ttnn/
    # cpp/ttnn/operations/conv/conv2d/conv2d_utils.cpp:883-887; the capture has
    # them at rows 11, 17/18 and 59/60.
    set_conv_input_reshard(True, device)

    # Opt in to tt-metal's PRIMARY sharded->sharded path in to_memory_config
    # (default OFF: it needs both shard specs to decide, and emits a Reshard
    # where a workload without them round-trips via DRAM).  The p100a/n150
    # captures record Reshard rows at the bottleneck transitions, and this
    # workload propagates the configs those rows depend on.
    set_to_memory_config_reshard(True, device)

    # Gate the post-Halo Move on reallocate_halo_output rather than
    # deallocate_activation (default OFF, since the two flags default opposite
    # ways).  tt-metal/ttnn/cpp/ttnn/operations/conv/conv2d/conv2d.cpp:297-299
    # is the flag hardware actually uses; keying off deallocate_activation gave
    # 17 Moves against the refrun's 2.
    set_move_on_reallocate_halo_output(True, device)
