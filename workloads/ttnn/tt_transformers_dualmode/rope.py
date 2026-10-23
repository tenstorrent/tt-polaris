#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Dual-mode RoPE for the llama3 tt_transformers port.

Basis: shim-only workloads/ttnn/tt_transformers/rope.py + tt-metal
models/tt_transformers/tt/{rope.py,common.py}.

Audit fixes vs the shim-only base:
  - The shim-only RotarySetup.__init__ computed cos/sin via ttnn.arange/outer/cos/sin and
    then DISCARDED the values (overwrote them with ttnn._rand). Those compute calls emit
    SimOps that do NOT exist on HW: tt-metal precomputes the matrices on host with torch and
    only transfers them via from_torch (no profiled device ops at init). Dual-mode fabricates
    the cos/sin/trans matrices as dummy weights DIRECTLY (no init-time device ops), matching
    HW and the dummy-weights perf-capture contract (design doc §8b).
  - The rope APPLICATION is a single ttnn.rotary_embedding_llama op emitted in attention.py
    (matches the capture's RotaryEmbeddingLlamaDeviceOperation, ops 27-28), NOT the shim-only
    utils.rotary_embedding_llama matmul-decomposition. rope.py only provides the matrices.
  - Matrix shapes mirror tt-metal common.py: prefill rot_mats = [cos, sin] each
    (1,1,seq_len,head_dim) (common.py:445-451); trans_mat uses a single tile (1,1,32,32)
    (common.py:473-477, dhead forced to 32).
  - rope values are dummy: irrelevant to perf (rope is one fixed-shape op) and accuracy is
    not validated here (perf-capture only). Real rope math is intentionally not replicated.
"""
import math
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

IS_POLARIS = os.getenv('IRD_ARCH_NAME', '') == ''
if IS_POLARIS:
    import ttsim.front.ttnn as ttnn
else:
    import ttnn  # type: ignore[import-not-found, no-redef]

TILE = 32


def nearest_32(x):
    return math.ceil(x / 32) * 32


def _dummy(shape, device, dtype):
    """Dummy matrix of the given shape (config-only, no HF checkpoint), fabricated directly
    in both modes. Values are irrelevant to perf and accuracy is not validated, so real rope
    math is not computed. ttnn.zeros also tolerates device=None (host tensor) — callers such
    as get_rot_transformation_mat() may omit a device, which the old from_torch path crashed on.
    """
    return ttnn.zeros(list(shape), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)


def _shape_list(t):
    """Mode-aware shape-as-list (ttsim require_shape_list is shim-only)."""
    if IS_POLARIS:
        from ttsim.ops.tensor import require_shape_list
        return require_shape_list(t.shape, 'shape must be known')
    return list(t.shape)


def get_rot_transformation_mat(dhead=32, device=None, datatype=None):
    # ROPE op uses a single tile (tt-metal common.py:473-477, dhead forced to 32).
    datatype = datatype if datatype is not None else ttnn.bfloat16
    return _dummy((1, 1, TILE, TILE), device, datatype)


def prefill_rope_table_len(max_seq_len, trace_prefill_seq_lens=None):
    """Rows of the traced prefill cos/sin tables (tt-metal f8c634a model._get_trace_rope_table_len).

    From f8c634a tt-metal pads the prefill RoPE tables so every traced prefill length can slice a
    fixed-width window: round_up(max_seq_len + max(lens), lcm(lens)). Llama-3.1-8B traces
    [128, 1024] on N150 and P100 (model_config.get_trace_prefill_supported_seq_lens), so max_seq_len
    1024 -> 2048 (BH p100a 261018 and WH n150 261015 captures: RotaryEmbeddingLlama cos/sin
    [1,1,2048,128]). With no lens (a model/device that traces no prefill lengths) the table is max_seq_len.
    """
    if not trace_prefill_seq_lens:
        return max_seq_len
    align = math.lcm(*trace_prefill_seq_lens)
    min_len = max_seq_len + max(trace_prefill_seq_lens)
    return ((min_len + align - 1) // align) * align


def get_prefill_rot_mat(head_dim, mesh_device, seq_len, theta, scale_factor, orig_context_len, start_pos=0):
    """Prefill rot_mats = [cos, sin], each (1,1,seq_len,head_dim) (tt-metal common.py:445-451)."""
    cos_gathereds = _dummy((1, 1, seq_len, head_dim), mesh_device, ttnn.bfloat16)
    sin_gathereds = _dummy((1, 1, seq_len, head_dim), mesh_device, ttnn.bfloat16)
    return [cos_gathereds, sin_gathereds]


class RotarySetup:
    def __init__(
        self,
        device,
        batch_size: int,
        head_dim: int,
        max_seq_len: int,
        rope_theta: float,
        scale_factor,            # None to disable rope scaling
        orig_context_len=None,   # only used if scaling enabled
        datatype=None,
    ):
        datatype = datatype if datatype is not None else ttnn.bfloat16
        self.batch_size = batch_size
        self.head_dim = head_dim
        self.device = device
        self.is_mesh_device = False
        self.num_devices = 1
        self.batch_size_per_device_group = batch_size
        self.core_grid = device.compute_with_storage_grid_size()

        # cos/sin caches: (1,1,max_seq_len,head_dim). Dummy (see module audit note) — the
        # shim-only init-time arange/outer/cos/sin compute is dropped (it over-emitted ops).
        # cos/sin caches are 2D (max_seq_len, head_dim) ROW_MAJOR — the ttnn.embedding weight form
        # used directly in get_rot_mats (tt-metal embeds self.cos_matrix with no squeeze; decode
        # builds it rot_mats_layout=ROW_MAJOR). 2D avoids the shim/torch-only `.squeeze()` method
        # (absent on real ttnn Tensors); ROW_MAJOR matches the capture embedding weight (input_1).
        self.cos_matrix = ttnn.zeros([max_seq_len, head_dim], dtype=datatype,
                                     layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        self.sin_matrix = ttnn.zeros([max_seq_len, head_dim], dtype=datatype,
                                     layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

        num_cores_x, num_cores_y = 8, 4
        self.batch_grid = (
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores_x - 1, num_cores_y - 1))})
            if ttnn.get_arch_name() == 'blackhole'
            else ttnn.num_cores_to_corerangeset(batch_size, self.core_grid, row_wise=True)
        )

        # Fused-QK decode doubles the batch (rope applied to q AND k). tt-metal RotarySetup
        # (rope.py: doubled_batch_size = batch*2; trans_mat = get_rot_transformation_mat(TILE)
        # .repeat(1,1,batch_size_per_device_group,1), HEIGHT-sharded). Batch-1 -> (1,1,64,32)
        # L1_HEIGHT_SHARDED, matching the capture's RotaryEmbeddingLlamaFusedQK input_4.
        self.doubled_batch = 2 * batch_size
        self.transformation_mat = _dummy((1, 1, TILE * self.doubled_batch, TILE), device, datatype)
        # trans_mat is (1,1,TILE*doubled_batch,TILE) HEIGHT-sharded into doubled_batch shards. The
        # grid must have >= doubled_batch cores; tt-metal sizes it to batch_size_per_device_group
        # (get_batch_grid), NOT the fixed 8x4 self.batch_grid — a 32-core grid trips the HW
        # "shards(64) must not exceed cores(32)" TT_FATAL. Size a dedicated grid to doubled_batch.
        _trans_grid = ttnn.num_cores_to_corerangeset(self.doubled_batch, self.core_grid, row_wise=True)
        _trans_mem = ttnn.create_sharded_memory_config(
            shape=(TILE, TILE), core_grid=_trans_grid,
            strategy=ttnn.ShardStrategy.HEIGHT, orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        # Dual-mode: the TT-Polaris shim carries memory as the `_memory_config` attr; real ttnn tensors
        # have no such attr — apply via to_memory_config (cf. lm_head._dummy_weight).
        if IS_POLARIS:
            self.transformation_mat._memory_config = _trans_mem
        else:
            self.transformation_mat = ttnn.to_memory_config(self.transformation_mat, _trans_mem)
        # Prefill trans_mat: single tile, DRAM (tt-metal keeps prefill dhead=head_dim, DRAM).
        self.transformation_mat_prefill = _dummy((1, 1, TILE, TILE), device, datatype)

    def get_both_trans_mats(self):
        assert self.transformation_mat is not None, 'Transformation matrix not initialized'
        assert self.transformation_mat_prefill is not None, 'Prefill transformation matrix not initialized'
        return {'decode': self.transformation_mat, 'prefill': self.transformation_mat_prefill}

    def get_rot_idxs(self, position_idxs, on_host=False):
        assert isinstance(position_idxs, ttnn.Tensor), 'Position ids must be a tensor'
        dims0 = _shape_list(position_idxs)
        assert len(dims0) == 1, 'position idxs must be a [batch] tensor'
        batch = dims0[0]
        position_idxs = ttnn.reshape(position_idxs, [1, batch])
        assert _shape_list(position_idxs) == [1, batch], 'position idxs must be a [1, batch] tensor'

        pad_size = nearest_32(batch) - batch
        position_idxs = ttnn.pad(position_idxs, (0, pad_size), 'constant', 0)
        rot_idxs = ttnn.as_tensor(
            position_idxs,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=None if on_host else self.device,
            memory_config=None if on_host else ttnn.DRAM_MEMORY_CONFIG,
        )
        return rot_idxs, position_idxs

    def get_rot_mats(self, position_idxs, return_rot_idxs=False):
        """Decode rot_mats via embedding + transpose + interleaved_to_sharded (real device ops)."""
        device = self.device
        rot_idxs = position_idxs
        assert len(rot_idxs.shape) == 2 and rot_idxs.shape[0] == 1, 'rot_idxs must be a [1, batch] tensor'
        if rot_idxs.device != device:
            rot_idxs = ttnn.to_device(rot_idxs, device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        embedding_layout = ttnn.TILE_LAYOUT
        cos = ttnn.embedding(rot_idxs, self.cos_matrix, layout=embedding_layout)
        sin = ttnn.embedding(rot_idxs, self.sin_matrix, layout=embedding_layout)

        cos = ttnn.unsqueeze_to_4D(cos)
        sin = ttnn.unsqueeze_to_4D(sin)
        cos = ttnn.transpose(cos, 1, 2)
        sin = ttnn.transpose(sin, 1, 2)

        # Fused-QK: slice the batch dim (dim 1) to doubled_batch. tt-metal RotarySetup.get_rot_mats:
        # `cos = cos[:, :batch_size_per_device_group, :, :]` (rope.py:1003) — batch-1 fused -> z 32->2,
        # emitting the 2 rope SliceDeviceOperation ops the capture has, and giving the downstream
        # interleaved_to_sharded / RotaryEmbeddingLlamaFusedQK the z=2 cos/sin the capture records.
        db = getattr(self, 'doubled_batch', 2 * self.batch_size)
        # Dual-mode: real ttnn slices via Python indexing (tt-metal rope.py:1003
        # `cos[:, :batch_size_per_device_group, :, :]`); the TT-Polaris shim Tensor has no __getitem__,
        # so use ttnn.slice (emits the Slice op that matches the capture).
        if IS_POLARIS:
            _sl = (slice(None), slice(0, db), slice(None), slice(None))
            cos = ttnn.slice(cos, slice=_sl)
            sin = ttnn.slice(sin, slice=_sl)
        else:
            cos = cos[:, :db, :, :]
            sin = sin[:, :db, :, :]

        mem_config = ttnn.create_sharded_memory_config(
            shape=(TILE, self.head_dim),
            core_grid=self.batch_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        cos = ttnn.interleaved_to_sharded(cos, mem_config)
        sin = ttnn.interleaved_to_sharded(sin, mem_config)

        if return_rot_idxs:
            return [cos, sin], rot_idxs
        return [cos, sin]
