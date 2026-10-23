#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Dual-mode Attention for the llama3 tt_transformers port (single-chip, paged).

Basis: shim-only workloads/ttnn/tt_transformers/attention.py, audited against tt-metal
models/tt_transformers/tt/attention.py and the BH prefill capture
(project_llama3_prefill_op_sequence). Calls the PR1 (#468) single device ops directly.

Audit fixes vs the shim-only base:
  - Single-chip (num_devices=1): all TG / mesh-shard (ShardTensor2dMesh) / CCL machinery in
    the shim-only base is dead and is dropped. The fused QKV weight is one dummy
    (1,1,dim,qkv_size) and wo is (1,1,dim,dim) — no per-device cat loop. tt_all_reduce is the
    identity from ccl.py (design doc §8c).
  - Rope is a SINGLE ttnn.experimental.rotary_embedding_llama op per Q/K (capture ops 27-28),
    replacing the shim-only utils.rotary_embedding_llama matmul-decomposition (which did NOT
    match HW).
  - PAGED prefill: fills the paged KV cache via ttnn.experimental.paged_fill_cache for K and V
    (capture ops 33-34) before the prefill SDPA, matching the capture and the paged_attention=1
    pin. The shim-only forward_prefill was non-paged (it ignored page_table / kv_cache).
  - Weights use args.WEIGHTS_DTYPE (bfloat8_b) per ModelArgs / tt-metal llama3; the shim-only
    base used bfloat16. (dtype parity with the capture is re-verified during LUT correlation.)
"""
import os
import sys

from loguru import logger

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

IS_POLARIS = os.getenv('IRD_ARCH_NAME', '') == ''
if IS_POLARIS:
    import ttsim.front.ttnn as ttnn
else:
    import ttnn  # type: ignore[import-not-found, no-redef]

from workloads.ttnn.tt_transformers_dualmode.ccl import tt_all_reduce
from workloads.ttnn.tt_transformers_dualmode.model_config import dram_sharded_weight_memcfg

# DRAM-sharded decode matmuls REQUIRE a sharded output on HW ("Output memory config must be
# sharded for DRAM sharded program config", matmul_device_operation.cpp) — tt-metal sets these
# to L1_WIDTH_SHARDED (get_attn_qkv_mm_mem_config / get_attn_wo_output_mem_config, no-prefetcher).
# TT-Polaris keeps None: the shim's L1_WIDTH_SHARDED_MEMORY_CONFIG is a 0-stub (no sharded modeling)
# and the None path already emits the capture-matching output key (verified 713/713).
_DECODE_MM_MEMCFG = None if IS_POLARIS else ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG


def _dummy_weight(shape, device, dtype):
    """Config-only dummy weight (no HF checkpoint), fabricated directly on-device in both
    modes. Values are irrelevant to perf and accuracy is not validated, so we avoid
    materializing a large host torch tensor on the HW path.

    Weights are DRAM width-sharded (see dram_sharded_weight_memcfg) so the matmul's
    input_1 memory tag matches the silicon capture (DRAM_WIDTH_SHARDED, not INTERLEAVED)."""
    w = ttnn.zeros(list(shape), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    cfg = dram_sharded_weight_memcfg(device, int(shape[-2]), int(shape[-1]))
    if IS_POLARIS:
        w._memory_config = cfg
    else:
        w = ttnn.to_memory_config(w, cfg)
    return w


def _is_bf16(tensor):
    """True if tensor is bf16. tensor.dtype is a DataType on the shim and a ttnn dtype on
    HW; ttnn.bfloat16 resolves correctly in both, so a direct compare works for both modes
    (the previous DataType.from_numpy() path mis-classified shim DataTypes as FLOAT32)."""
    return tensor.dtype == ttnn.bfloat16


class Attention:
    def __init__(
        self,
        mesh_device,
        state_dict,
        weight_cache_path,
        layer_num,
        dtype,
        transformation_mats,
        configuration,
        paged_attention_config=None,
        use_paged_kv_cache=False,
    ):
        self.state_dict = state_dict
        self.mesh_device = mesh_device
        self.args = configuration  # for decode matmul program configs (dram_matmul_config)
        self.hidden_size = configuration.dim
        self.n_heads = configuration.n_heads
        self.head_dim = configuration.head_dim
        self.n_kv_heads = configuration.n_kv_heads
        self.max_seq_len = configuration.max_seq_len
        self.max_batch_size = configuration.max_batch_size
        self.paged_attention_config = paged_attention_config
        self.transformation_mats = transformation_mats
        self.dtype = dtype
        self.MAX_QKV_MM_SEQ_LEN = configuration.MAX_QKV_MM_SEQ_LEN
        # single-chip: no head/device splitting
        self.n_local_heads = self.n_heads
        self.n_local_kv_heads = self.n_kv_heads
        # KV cache dtype from ModelArgs (BFP8 under the decode perf preset — tt-metal
        # DecodersPrecision.performance KV_CACHE=BFP8; capture confirms BFLOAT8_B). Was hardcoded bf16.
        self.kv_cache_dtype = configuration.kv_cache_dtype
        self.activation_dtype = ttnn.bfloat16
        # Prefill follows tt-metal's `activation_dtype or <default>` against the ModelArgs override
        # (None for llama3); self.activation_dtype above is the decode setting.
        self.prefill_activation_dtype = configuration.activation_dtype
        # qkv / o / sdpa run at HiFi2 in decode under the performance preset (tt-metal
        # DecodersPrecision default: LI_QKV_DECODE / LI_O_DECODE / SDPA_DECODE = HIFI2).
        self.compute_kernel_config_hifi2 = configuration.compute_kernel_config_hifi2
        wdtype = configuration.WEIGHTS_DTYPE

        # Fused QKV weight (1,1,dim,qkv_size) and output proj (1,1,dim,dim) — dummy.
        self.wqkv = _dummy_weight((1, 1, self.hidden_size, configuration.qkv_size), mesh_device, wdtype)
        self.wo = _dummy_weight((1, 1, self.hidden_size, self.hidden_size), mesh_device, wdtype)

        # q/k norm are identity in llama3 (no QK-norm)
        self.q_norm = lambda x, mode: x
        self.k_norm = lambda x, mode: x

        if not use_paged_kv_cache:
            self.init_kv_cache(configuration, weight_cache_path, device=mesh_device)

        if configuration.query_pre_attn_scalar is not None:
            self.scale = configuration.query_pre_attn_scalar ** -0.5
        else:
            self.scale = self.head_dim ** -0.5

    def init_kv_cache(self, configuration, weight_cache_path, device=None):
        """Empty (dummy) KV cache on device. Paged shape per paged_attention_config."""
        if self.paged_attention_config:
            shape = [
                self.paged_attention_config.max_num_blocks,
                self.n_local_kv_heads,
                self.paged_attention_config.block_size,
                self.head_dim,
            ]
        else:
            shape = [self.max_batch_size, self.n_local_kv_heads, self.max_seq_len, self.head_dim]
        self.layer_past = [
            ttnn.zeros(shape, dtype=self.kv_cache_dtype, device=device, layout=ttnn.TILE_LAYOUT)
            for _ in range(2)
        ]

    def forward_prefill(
        self,
        x_11SH,
        rot_mats,
        user_id: int = 0,
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
        kv_cache=None,
    ):
        seq_len = x_11SH.shape[-2]
        assert seq_len % 128 == 0 and seq_len > 0, 'Seqlen must be divisible by 128'

        # reshape long sequences so the QKV matmul fits on device
        if seq_len > self.MAX_QKV_MM_SEQ_LEN:
            if seq_len % self.MAX_QKV_MM_SEQ_LEN != 0:
                raise ValueError(f'seq_len {seq_len} must be divisible by {self.MAX_QKV_MM_SEQ_LEN}')
            x_11SH = ttnn.reshape(x_11SH, [1, seq_len // self.MAX_QKV_MM_SEQ_LEN, self.MAX_QKV_MM_SEQ_LEN, -1])

        # Fused QKV projection (capture op 25). tt-metal: li_qkv_prefill_compute_kernel_cfg (HiFi2) and
        # get_attn_qkv_mm_mem_config(PREFILL) = DRAM; the shim's None default would shard the output.
        xqkv_fused = ttnn.linear(
            x_11SH, self.wqkv, memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config_hifi2,
            program_config=self.args.attn_qkv_prefill_program_config(seq_len),
        )
        xqkv_fused = tt_all_reduce(xqkv_fused)  # identity (single-chip)
        if seq_len > self.MAX_QKV_MM_SEQ_LEN:
            xqkv_fused = ttnn.reshape(xqkv_fused, [1, 1, seq_len, -1])
        ttnn.deallocate(x_11SH)

        # Split into heads (capture op 26)
        q_heads_pre, k_heads_pre, v_heads = ttnn.experimental.nlp_create_qkv_heads(
            xqkv_fused, num_heads=self.n_local_heads, num_kv_heads=self.n_local_kv_heads,
            transpose_k_heads=False, memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        q_heads_pre = self.q_norm(q_heads_pre, mode='prefill')
        k_heads_pre = self.k_norm(k_heads_pre, mode='prefill')
        ttnn.deallocate(xqkv_fused)

        # Rotary embeddings: ONE op each for Q, K (capture ops 27-28)
        if not _is_bf16(q_heads_pre):
            q_heads_pre = ttnn.typecast(q_heads_pre, dtype=ttnn.bfloat16)
        q_heads = ttnn.experimental.rotary_embedding_llama(
            q_heads_pre, rot_mats[0], rot_mats[1], self.transformation_mats['prefill'], is_decode_mode=False,
        )
        ttnn.deallocate(q_heads_pre)
        if not _is_bf16(k_heads_pre):
            k_heads_pre = ttnn.typecast(k_heads_pre, dtype=ttnn.bfloat16)
        k_heads = ttnn.experimental.rotary_embedding_llama(
            k_heads_pre, rot_mats[0], rot_mats[1], self.transformation_mats['prefill'], is_decode_mode=False,
        )
        ttnn.deallocate(k_heads_pre)

        # Typecast K/V for cache + SDPA (capture ops 29-30)
        k_heads_8b = ttnn.typecast(k_heads, dtype=ttnn.bfloat8_b)
        v_heads_8b = ttnn.typecast(v_heads, dtype=ttnn.bfloat8_b)
        ttnn.deallocate(k_heads)
        ttnn.deallocate(v_heads)

        # PAGED KV-cache fill for K and V (capture ops 33-34)
        kc = kv_cache if kv_cache is not None else self.layer_past
        ttnn.experimental.paged_fill_cache(kc[0], k_heads_8b, page_table=page_table, batch_idx=user_id)
        ttnn.experimental.paged_fill_cache(kc[1], v_heads_8b, page_table=page_table, batch_idx=user_id)

        # Prefill SDPA on the in-flight heads (capture op 36)
        # tt-metal attention.py forward_prefill: typecast(q, self.activation_dtype or bfloat8_b); the
        # llama3 perf preset leaves activation_dtype None -> bf8 (BH p100a prefill capture: typecast
        # [1,32,128,128] BF16 -> BF8 and SDPA input_0 BF8). self.activation_dtype (bf16) is the
        # decode setting and made this a bf16->bf16 no-op, dropping the op.
        q_heads_8b = ttnn.typecast(q_heads, dtype=self.prefill_activation_dtype or ttnn.bfloat8_b)
        ttnn.deallocate(q_heads)
        # sdpa_prefill_compute_kernel_cfg = compute_kernel_config_hifi4 (DecodersPrecision SDPA_PREFILL =
        # HIFI4) and get_attn_sdpa_program_config(PREFILL), tt-metal f8c634a. The captures key SDPA
        # fidelity as N/A, so the shim keeps None and its LUT key is unchanged.
        attn_output = ttnn.transformer.scaled_dot_product_attention(
            q_heads_8b, k_heads_8b, v_heads_8b, is_causal=True, scale=self.scale,
            compute_kernel_config=None if IS_POLARIS else self.args.compute_kernel_config_hifi4,
            program_config=self.args.attn_sdpa_prefill_program_config(seq_len),
        )
        ttnn.deallocate(q_heads_8b)
        ttnn.deallocate(k_heads_8b)
        ttnn.deallocate(v_heads_8b)

        # batch-1 SDPA output is already [1, n_heads, S, head_dim]; on HW that reshape is a free view
        # (no profiler row), while the shim lowers a TILE reshape to Untilize+TilizeWithValPadding.
        # Skip it when the shape already matches so both paths emit nothing here.
        target = [1, self.n_local_heads, attn_output.shape[-2], self.head_dim]
        if list(attn_output.shape) != target:
            attn_output = ttnn.reshape(attn_output, target)

        # Concat heads (capture op 37)
        attn_output = ttnn.experimental.nlp_concat_heads(attn_output, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        # Output projection (capture op 38): li_o_prefill_compute_kernel_cfg (HiFi2), DRAM output.
        output = ttnn.linear(
            attn_output, self.wo, compute_kernel_config=self.compute_kernel_config_hifi2,
            dtype=self.prefill_activation_dtype or ttnn.bfloat8_b,
            program_config=self.args.attn_wo_prefill_program_config(seq_len), memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(attn_output)
        return tt_all_reduce(output)  # identity (single-chip)

    def forward_decode(self, x, current_pos, rot_mats=None, page_table=None, kv_cache=None):
        """Paged decode: create-heads-decode -> fused rope -> fused cache update -> paged SDPA -> concat."""
        # DRAM-sharded weight -> needs the DRAM-sharded matmul program config on HW (see
        # model_config.dram_matmul_config); None on TT-Polaris (shim ignores program_config).
        xqkv_fused = ttnn.linear(x, self.wqkv, memory_config=_DECODE_MM_MEMCFG,
                                 compute_kernel_config=self.compute_kernel_config_hifi2,
                                 program_config=self.args.attn_qkv_decode_program_config())
        xqkv_fused = tt_all_reduce(xqkv_fused)
        # createqkv-decode requires interleaved bf16 input (capture STS, row 33)
        xqkv_fused = ttnn.sharded_to_interleaved(xqkv_fused)
        # Track the TRUE (unpadded) decode batch in the logical shape: reshape to logical
        # batch=max_batch_size, padded to 32 (tt-metal attention.py:642-645, commit d9d52dfe7b6,
        # "Reshape such that true unpadded batch is tracked in shape"). Without it, real ttnn's
        # nlp_create_qkv_heads_decode reads logical batch = act_users (32) and emits a 32-user
        # q_heads (32 height-shards), overrunning the fused-QK rope's per-batch grid.
        # HW-only guard: on the shim this reshape is NOT a free logical/padded re-label — it mutates
        # the tracked graph (op count 713->777, +96 LUT misses). The shim's create_qkv already derives
        # a batch-agnostic q=[1,1,32,128] straight from hidden+num_heads, matching the batch-1 capture,
        # so TT-Polaris needs no reshape. Real ttnn does (its create_qkv honors the logical batch).
        if not IS_POLARIS:
            _fq = xqkv_fused.shape[3]
            xqkv_fused = ttnn.reshape(xqkv_fused, (1, 1, self.max_batch_size, _fq), (1, 1, 32, _fq))
        # No head_dim kwarg: real ttnn.nlp_create_qkv_heads_decode has no such parameter (it infers
        # head_dim = hidden // (num_heads + 2*num_kv_heads)); the shim infers it identically when
        # head_dim is omitted. Dropping it keeps this call byte-identical across both modes.
        q_heads, k_heads, v_heads = ttnn.experimental.nlp_create_qkv_heads_decode(
            xqkv_fused, num_heads=self.n_local_heads, num_kv_heads=self.n_local_kv_heads,
            memory_config=None,
        )
        ttnn.deallocate(xqkv_fused)
        # rope-decode input must match the transformation_mat shard spec (capture Reshard row 35).
        # NOTE: q/k for the fused-QK rope are HEIGHT-sharded onto a BATCH-sized grid, NOT the full
        # device grid. tt-metal (commit d9d52dfe7b6) never uses the full ~120-core BH grid here:
        # to_qk_fused_memory_config (models/tt_transformers/tt/attention.py:440-497) shards q on
        # q_batch cores and k on k_batch disjoint cores, and RotaryEmbeddingLlamaFusedQK asserts
        # q_num_cores + k_num_cores <= 64 (rotary_embedding_llama_fused_qk_device_operation.cpp:92).
        # The batch-1 p100a capture confirms this: the q Reshard runs on 1 core and the fused-QK
        # rope on 2 cores. core_grid_from (the full 120-core grid) tripped the <=64 assert on HW.
        # Sizing to max_batch_size carves batch-many cores from the device pool (1 for batch-1).
        # (Core-grid count is not part of the LUT key, so TT-Polaris stays 713/713.)
        q_heads = ttnn.reshard(q_heads, ttnn.create_sharded_memory_config(
            shape=(32, self.head_dim),
            core_grid=ttnn.num_cores_to_corerangeset(
                self.max_batch_size, self.mesh_device.compute_with_storage_grid_size(), row_wise=True),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        ))
        if not IS_POLARIS:
            # HW-only: the fused-QK rope requires q and k on DISJOINT cores ("Q and K must not
            # overlap", rotary_embedding_llama_fused_qk_device_operation.cpp:95). q is resharded to
            # the origin above; k comes off create_qkv also at the origin -> they overlap. tt-metal's
            # to_qk_fused_memory_config (attention.py:479-497, commit d9d52dfe7b6) offsets k by
            # q_batch cores (k_start_core = CoreCoord(q_batch % 8, q_batch // 8)) and gives it a
            # k_batch-wide range via num_to_corerange(k_batch, start_core). q_batch == k_batch ==
            # max_batch_size here; the fused-QK op asserts q_num_cores + k_num_cores <= 64, so
            # batch <= 32. TT-Polaris keys on shape, not core placement, so this HW-only op keeps
            # TT-Polaris at 749/749.
            # tt-metal helper — only importable on HW (the models pkg isn't in the TT-Polaris env).
            # Imported rather than vendored on purpose: a local copy would drift from
            # model_config.py. Guarded so a misconfigured HW run fails with a usable message
            # instead of a bare ModuleNotFoundError from inside attention forward.
            try:
                from models.tt_transformers.tt.model_config import num_to_corerange  # type: ignore[import-not-found]
            except ImportError as exc:
                raise ImportError(
                    "llama3 decode on hardware needs tt-metal's models.tt_transformers package "
                    "(num_to_corerange, used for the fused-QK k-shard core range). Run from the "
                    "tt-metal checkout root so `models/` is importable, or unset IRD_ARCH_NAME to "
                    "run the TT-Polaris shim path instead."
                ) from exc
            if self.max_batch_size > 1:
                logger.warning(
                    f"llama3 decode HW path exercised with max_batch_size={self.max_batch_size}: the "
                    "fused-QK k-shard core range (k_batch cores offset by q_batch, batch<=32 per the "
                    "64-core fused-QK limit) is implemented but UNVALIDATED on silicon — there is no "
                    "batch>1 decode capture (p100a batch-32 OOMs). Verify against HW before relying on it."
                )
            _kstart = ttnn.CoreCoord(self.max_batch_size % 8, self.max_batch_size // 8)
            k_heads = ttnn.reshard(k_heads, ttnn.create_sharded_memory_config(
                shape=(32, self.head_dim),
                core_grid=ttnn.CoreRangeSet({num_to_corerange(self.max_batch_size, start_core=_kstart)}),
                strategy=ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            ))
        q_heads, k_heads = ttnn.experimental.rotary_embedding_llama_fused_qk(
            q_heads, k_heads, rot_mats[0], rot_mats[1], self.transformation_mats['decode'],
        )
        kc = kv_cache if kv_cache is not None else self.layer_past
        ttnn.experimental.paged_fused_update_cache(
            kc[0], k_heads, kc[1], v_heads, update_idxs_tensor=current_pos, page_table=page_table,
        )
        attn_output = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q_heads, kc[0], kc[1], page_table_tensor=page_table, cur_pos_tensor=current_pos, scale=self.scale,
        )
        # sdpa-decode outputs DRAM-interleaved; concat-heads needs L1 height-sharded (capture ITS, row 39).
        # nlp_concat_heads_decode asserts num_cores == num_users (input_shape[1])
        # (nlp_concat_heads_decode_device_operation.cpp:68), so the shard grid must be BATCH-sized, not
        # the full device grid. tt-metal get_attn_sdpa_output_mem_config (model_config.py:1765, no
        # prefetcher, commit d9d52dfe7b6) uses num_to_corerange(batch_size_per_device_group). Match it
        # (batch-1 -> 1 core). Core-grid count is not in the LUT key, so TT-Polaris stays 749/749.
        attn_output = ttnn.interleaved_to_sharded(attn_output, ttnn.create_sharded_memory_config(
            shape=(32, self.head_dim),
            core_grid=ttnn.num_cores_to_corerangeset(
                self.max_batch_size, self.mesh_device.compute_with_storage_grid_size(), row_wise=True),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        ))
        attn_output = ttnn.experimental.nlp_concat_heads_decode(attn_output, num_heads=self.n_local_heads)
        # wo weight is DRAM-sharded -> DRAM-sharded matmul program config on HW (None on TT-Polaris).
        output = ttnn.linear(attn_output, self.wo, compute_kernel_config=self.compute_kernel_config_hifi2,
                             dtype=self.activation_dtype,
                             program_config=self.args.attn_wo_decode_program_config(),
                             memory_config=_DECODE_MM_MEMCFG)
        return tt_all_reduce(output)

    def forward(self, x, current_pos, rot_mats=None, user_id=0, mode='decode', page_table=None,
                chunk_page_table=None, chunk_start_idx=None, kv_cache=None):
        if mode == 'prefill':
            return self.forward_prefill(x, rot_mats, user_id, page_table=page_table,
                                        chunk_page_table=chunk_page_table, chunk_start_idx=chunk_start_idx,
                                        kv_cache=kv_cache)
        return self.forward_decode(x, current_pos, rot_mats, page_table=page_table, kv_cache=kv_cache)

    def __call__(self, x, current_pos=None, rot_mats=None, user_id=0, mode='prefill', page_table=None,
                 chunk_page_table=None, chunk_start_idx=None, kv_cache=None):
        return self.forward(x, current_pos, rot_mats=rot_mats, user_id=user_id, mode=mode,
                            page_table=page_table, chunk_page_table=chunk_page_table,
                            chunk_start_idx=chunk_start_idx, kv_cache=kv_cache)