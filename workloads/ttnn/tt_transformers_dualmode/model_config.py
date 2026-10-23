#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Dual-mode ModelArgs for the llama3 tt_transformers port.

Based on the shim-only workloads/ttnn/tt_transformers/model_config.py, made
dual-mode (IS_POLARIS gate) and audited against tt-metal
models/tt_transformers/tt/model_config.py. Config-only + dummy weights — the
TT-Polaris path never reads an HF checkpoint (see design doc §5).

Audit fixes vs the shim-only base:
  - qkv_size now COMPUTED as head_dim*(2*n_kv_heads + n_heads) per tt-metal
    model_config.py:660 -> 6144 for llama3-8B (shim-only hardcoded a stale 5120).
  - hidden_dim (MLP intermediate, tt-metal `calculate_hidden_dim`) added: 14336
    for llama3-8B (absent in the shim-only base).
Scoped to the llama3 family (the workload); non-llama3 variants live in the
untouched shim-only model_config for mixtral et al.
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


def core_grid_from(device):
    """A create_sharded_memory_config-valid core_grid from device.compute_with_storage_grid_size().

    Real ttnn returns a CoreCoord there, which create_sharded_memory_config REJECTS ("Invalid
    core_grid type" — it accepts CoreGrid/tuple/list/CoreRangeSet); the TT-Polaris shim returns a
    (x, y) tuple. Normalize both to ttnn.CoreGrid(x, y). (The shim's create_sharded_memory_config
    ignores core_grid, so this only matters on hardware.)
    """
    g = device.compute_with_storage_grid_size()
    if IS_POLARIS:
        # The shim's create_sharded_memory_config ignores core_grid, and the shim grid is an
        # uninitialized (0,0) tuple — wrapping it in CoreGrid would build an illegal CoreCoord(-1,-1).
        # Pass it through unchanged (matches the pre-existing shim behavior).
        return g
    return ttnn.CoreGrid(x=g.x, y=g.y)


def _find_grid(n_tiles):
    """rows, cols for a grid that evenly divides n_tiles, closest to 32 cores (tt-metal
    ModelArgs.find_grid). Bounds: WH 8x8, BH 10x12."""
    wh = ttnn.get_arch_name() == 'wormhole_b0'
    max_rows, max_cols = (8, 8) if wh else (10, 12)
    max_cores = max_rows * max_cols
    possible = sorted((k for k in range(1, max_cores + 1) if n_tiles % k == 0), key=lambda x: abs(x - 32))
    for cores in possible:
        for rows in range(1, max_rows + 1):
            if cores % rows == 0 and cores // rows <= max_cols:
                return rows, cores // rows
    raise AssertionError(f"no grid for {n_tiles} tiles within {max_rows}x{max_cols}")


def dram_shard_core_grid_for_k(k):
    """Grid that width-shards a K-dim tensor evenly (tt-metal ModelArgs.dram_shard_core_grid_for_k):
    find_grid(k // TILE). Used for the decode WIDTH-sharded activation/residual configs — the full
    compute grid does not divide k evenly and trips 'Invalid sharding core_grid' on HW."""
    rows, cols = _find_grid(k // 32)
    return ttnn.CoreGrid(x=cols, y=rows)


def _num_cores_for_k(k):
    """num_cores of the K-only DRAM-shard grid (tt-metal dram_shard_core_grid_for_k(k).num_cores)."""
    rows, cols = _find_grid(k // 32)
    return rows * cols


def _find_grid_k_n(k_tiles, n_tiles):
    """rows, cols for a grid dividing BOTH k_tiles and n_tiles, largest such grid (tt-metal
    ModelArgs.find_grid_k_n). Note: fixed 8x8 bound on BOTH arches (unlike find_grid, which is
    arch-dependent) — this mirrors tt-metal find_grid_k_n verbatim."""
    max_rows = max_cols = 8
    max_cores = max_rows * max_cols
    possible = sorted((c for c in range(1, max_cores + 1) if k_tiles % c == 0 and n_tiles % c == 0),
                      reverse=True)
    for cores in possible:
        for rows in range(1, max_rows + 1):
            if cores % rows == 0 and cores // rows <= max_cols:
                return rows, cores // rows
    raise AssertionError(f"no grid dividing both {k_tiles} and {n_tiles} tiles within 8x8")


def _num_cores_for_k_and_n(k, n):
    """num_cores of the (K,N) DRAM-shard grid (tt-metal dram_shard_core_grid_for_k_and_n(k,n).num_cores)."""
    rows, cols = _find_grid_k_n(k // 32, n // 32)
    return rows * cols


def dram_shard_core_grid_for_k_and_n(k, n):
    """(K,N) DRAM-shard grid as a CoreGrid (tt-metal ModelArgs.dram_shard_core_grid_for_k_and_n)."""
    rows, cols = _find_grid_k_n(k // 32, n // 32)
    return ttnn.CoreGrid(x=cols, y=rows)


def _find_largest_divisor(n, max_divisor=8):
    """Largest divisor of n up to max_divisor (tt-metal ModelArgs.find_largest_divisor)."""
    for i in range(max_divisor, 0, -1):
        if n % i == 0:
            return i
    return 1


def _find_prefill_grid(row_tiles, col_tiles):
    """(rows, cols) with each dividing its tile count, capped at 8x8 (tt-metal ModelArgs.find_prefill_grid)."""
    cols = next(i for i in range(8, 0, -1) if col_tiles % i == 0)
    rows = next(i for i in range(8, 0, -1) if row_tiles % i == 0)
    return rows, cols


def _get_out_subblock_w(per_core_N, out_subblock_h):
    """Largest out_subblock_w <= 4 dividing per_core_N with h*w <= 4 (tt-metal common.get_out_subblock_w)."""
    out_subblock_w = 4
    while out_subblock_w > 1:
        if out_subblock_w * out_subblock_h <= 4 and per_core_N % out_subblock_w == 0:
            break
        out_subblock_w -= 1
    return out_subblock_w


def dram_sharded_weight_memcfg(mesh_device, k, n):
    """DRAM width-sharded memory config for a (k, n) weight tensor.

    Mirrors tt-metal ModelArgs.create_dram_sharded_mem_config (models/tt_transformers/
    tt/model_config.py): matmul weights are DRAM width-sharded on silicon, so the
    matmul's input_1 memory tag in the capture is ``DEV_*_DRAM_WIDTH_SHARDED``. Without
    this the shim emits ``DRAM_INTERLEAVED`` and every matmul misses the LUT on the
    weight-memory field.

    TT-Polaris only needs the canonical (WIDTH_SHARDED, DRAM) tag for LUT-key parity — the
    ShardSpec details are not part of the key. The HW branch builds the full ShardSpec
    across the DRAM grid to match tt-metal. NOTE: the HW branch is not exercised in the
    TT-Polaris CI here (no silicon at authoring time); it reproduces tt-metal verbatim.
    """
    if IS_POLARIS:
        return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM)
    dram_cores = mesh_device.dram_grid_size().x
    padded_size = math.ceil(n / (ttnn.TILE_SIZE * dram_cores)) * (ttnn.TILE_SIZE * dram_cores)
    dram_grid = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_cores - 1, 0))}
    )
    shard_spec = ttnn.ShardSpec(dram_grid, (k, padded_size // dram_cores), ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, shard_spec)


# llama3 family config: (dim, n_heads, n_kv_heads, head_dim, hidden_dim/MLP-intermediate)
_LLAMA3_CONFIGS = {
    'llama3-8b': {'dim': 4096, 'n_heads': 32, 'n_kv_heads': 8, 'head_dim': 128, 'hidden_dim': 14336},
    'llama3-3b': {'dim': 3072, 'n_heads': 24, 'n_kv_heads': 8, 'head_dim': 128, 'hidden_dim': 8192},
    'llama3-1b': {'dim': 2048, 'n_heads': 32, 'n_kv_heads': 8, 'head_dim': 64, 'hidden_dim': 8192},
    'llama3-70b': {'dim': 8192, 'n_heads': 64, 'n_kv_heads': 8, 'head_dim': 128, 'hidden_dim': 28672},
}


class ModelArgs:
    def __init__(self, mesh_device, model_name='llama3-8B', max_batch_size=1, max_seq_len=256,
                 instruct=False, dummy_weights=True):
        self.mesh_device = mesh_device

        name_lower = model_name.lower()
        cfg = _LLAMA3_CONFIGS.get(name_lower, _LLAMA3_CONFIGS['llama3-8b'])
        self.model_name = 'llama3-8B' if name_lower not in _LLAMA3_CONFIGS else model_name

        # 1. Architecture (audited vs tt-metal llama3 config)
        self.dim = cfg['dim']
        self.n_heads = cfg['n_heads']
        self.n_kv_heads = cfg['n_kv_heads']
        self.head_dim = cfg['head_dim']
        self.hidden_dim = cfg['hidden_dim']        # MLP intermediate (tt-metal calculate_hidden_dim)
        self.hidden_size = self.dim                # alias used by some callers
        self.norm_eps = 1e-5
        self.vocab_size = 128256
        self.n_layers = 1                          # capture/runner overrides (1-layer device-perf)
        self.num_devices = 1
        self.num_experts = 1
        self.moe = False

        # qkv_size = head_dim * (2*n_kv_heads + n_heads)  (tt-metal model_config.py:660)
        # llama3-8B -> 128 * (16 + 32) = 6144  (matches the HW fused-QKV proj 4096->6144)
        self.qkv_size = self.head_dim * (2 * self.n_kv_heads + self.n_heads)

        # 2. Simulator/general settings
        self.rms_norm_add_unit_offset = False
        self.max_batch_size = max_batch_size
        # Decode activation rows are tile-padded to a multiple of 32 (tt-metal
        # ModelArgs.tile_padded_batch_rows) — the M dim of every decode DRAM-sharded matmul.
        self.tile_padded_batch_rows = 32 * math.ceil(max_batch_size / 32)
        self.num_reduce_scatter_links = 1
        self.arch_name = ttnn.get_arch_name()
        # compute_kernel_config: on TT-Polaris a bare MathFidelity (the shim ops read it for the LUT
        # math_fidelity key); on real ttnn a WormholeComputeKernelConfig (ttnn.linear/rms_norm reject
        # a bare MathFidelity). Mirrors tt-metal ModelArgs (WormholeComputeKernelConfig, hifi2
        # math_approx_mode=True / hifi4 False, fp32_dest_acc_en + packer_l1_acc True). "Wormhole" is
        # the legacy type name used on BH too. LoFi (FF1/FF3 gate/up) per DecodersPrecision.performance.
        if IS_POLARIS:
            self.compute_kernel_config_hifi2 = ttnn.MathFidelity.HiFi2
            self.compute_kernel_config_hifi2_fp16 = ttnn.MathFidelity.HiFi2
            self.compute_kernel_config_hifi4 = ttnn.MathFidelity.HiFi4
            self.compute_kernel_config_lofi = ttnn.MathFidelity.LoFi
            self.compute_kernel_config_lofi_prefill = ttnn.MathFidelity.LoFi
        else:
            def _ckc(mf, approx):
                return ttnn.WormholeComputeKernelConfig(
                    math_fidelity=mf, math_approx_mode=approx,
                    fp32_dest_acc_en=True, packer_l1_acc=True,
                )
            self.compute_kernel_config_hifi2 = _ckc(ttnn.MathFidelity.HiFi2, True)
            self.compute_kernel_config_hifi4 = _ckc(ttnn.MathFidelity.HiFi4, False)
            self.compute_kernel_config_lofi = _ckc(ttnn.MathFidelity.LoFi, True)
            # tt-metal ModelArgs (f8c634a) builds lofi and hifi2_fp16 WITHOUT fp32 dest
            # accumulation or approx math ("FP32 acc does not appear to be needed"): BH p100a and WH
            # n150 prefill captures both run FF1/FF3 as LoFi/approx 0/fp32 0/packer 1 and FF2 + the
            # lm_head chunks as HiFi2/approx 0/fp32 0/packer 1. Used by the prefill paths only; decode
            # keeps compute_kernel_config_lofi as pinned to its own capture.
            def _ckc_fp16(mf):
                return ttnn.WormholeComputeKernelConfig(
                    math_fidelity=mf, math_approx_mode=False,
                    fp32_dest_acc_en=False, packer_l1_acc=True,
                )
            self.compute_kernel_config_hifi2_fp16 = _ckc_fp16(ttnn.MathFidelity.HiFi2)
            self.compute_kernel_config_lofi_prefill = _ckc_fp16(ttnn.MathFidelity.LoFi)
        # Real ttnn CoreGrid takes (x, y) — use that form so this runs on HW too (the TT-Polaris shim's
        # CoreGrid now accepts x=/y= as well). 8x8 grid == CoreCoord(0,0)..(7,7).
        self.max_grid_size = ttnn.CoreGrid(x=8, y=8)
        self.MAX_QKV_MM_SEQ_LEN = 2048
        self.num_all_gather_links = 1
        self.instruct = instruct
        self.ccl_dtype = ttnn.bfloat8_b
        self.tile_size = 32
        # min_kv_prefill_shard_seqlen = (TILE_SIZE*8*8) / n_kv_heads  (tt-metal model_config.py:661)
        self.min_kv_prefill_shard_seqlen = (32 * 8 * 8) / self.n_kv_heads
        self.max_seq_len = max_seq_len
        self.rope_theta = 500000.0
        # Llama-3.1 llama3-type rope scaling (HF meta-llama/Llama-3.1-8B-Instruct
        # config.json: rope_scaling = {rope_type: llama3, factor: 8, low_freq_factor: 1,
        # high_freq_factor: 4, original_max_position_embeddings: 8192}). RotarySetup
        # (rope.py) consumes rope_scaling_factor + orig_context_len; both None disables
        # scaling. Wiring the real values makes the cos/sin tables match the reference
        # model — shape/LUT-key-neutral (values only), but corrects rope fidelity.
        self.rope_scaling_factor = 8.0
        self.orig_context_len = 8192
        self.rope_scaling = {
            'rope_type': 'llama3', 'factor': 8.0,
            'low_freq_factor': 1.0, 'high_freq_factor': 4.0,
            'original_max_position_embeddings': 8192,
        }
        self.model_config = None
        self.is_multichip = False
        self.dummy_weights = dummy_weights
        self.cluster_shape = [1, 1]
        self.use_qk_fused = False
        self.use_fused_qkv_op = True
        self.query_pre_attn_scalar = None
        self.is_galaxy = False
        self.is_distributed_norm = False
        self.padded_vocab_size = None
        self.checkpoint_type = 'simulation'
        self.WEIGHTS_DTYPE = ttnn.bfloat8_b
        # MLP gate/up projections (w1/w3, FF1/FF3) are BFP4 on silicon; down_proj (w2)
        # and attention stay BFP8. Mirrors tt-metal decoders_optimizations ff1_3_dtype
        # ("All models use bfp4 in FF1 and FF3 MLPs", model_config.py) and the capture
        # (the 14336-wide w1/w3 matmuls have input_1 dtype BFLOAT4_B).
        self.FF1_FF3_WEIGHTS_DTYPE = ttnn.bfloat4_b
        # lm_head matmul output is downcast to bfp8 (tt-metal ModelArgs.lm_head_dtype).
        self.lm_head_dtype = ttnn.bfloat8_b
        # KV cache is BFP8 under the decode performance preset (tt-metal DecodersPrecision.performance:
        # TensorGroup.KV_CACHE = BFP8; bf16 only in the accuracy preset). The capture confirms the
        # paged-cache tensors are BFLOAT8_B. Attention reads this instead of hardcoding bf16.
        self.kv_cache_dtype = ttnn.bfloat8_b
        # Activation dtype override (tt-metal decoders_optimizations.get_tensor_dtype(TensorGroup.ACTIVATION)).
        # None under the llama3 performance preset, so prefill ops fall back to tt-metal's per-op default
        # (`activation_dtype or bfloat8_b` / `or bfloat16`). A model whose preset sets it overrides here.
        self.activation_dtype = None

    def dram_matmul_config(self, m, k, n, num_cores):
        """DRAM-sharded 1D matmul program config for a decode GEMM whose weight is DRAM
        width-sharded (tt-metal ModelArgs.dram_matmul_config -> MatmulMultiCoreReuseMultiCast
        DRAMShardedProgramConfig).

        WHY this is required on HW: real ttnn.linear's default (auto) program config asserts the
        weight (input_tensor_b) is INTERLEAVED and TT_FATALs on a WIDTH_SHARDED one. Our decode
        weights are DRAM width-sharded (dram_sharded_weight_memcfg) to match the silicon capture's
        input_1 memory tag, so each such matmul MUST carry this dedicated program config. The
        TT-Polaris shim ignores program_config, so return None there (keeps the shim graph identical).
        """
        if IS_POLARIS:
            return None
        return ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(  # type: ignore[attr-defined]
            in0_block_w=_find_largest_divisor(k // (ttnn.TILE_SIZE * num_cores)),
            per_core_M=math.ceil(m / ttnn.TILE_SIZE),
            per_core_N=math.ceil(n / (ttnn.TILE_SIZE * num_cores)),
            fused_activation=None,
        )

    # Per-matmul decode program configs (single-device, no prefetcher/galaxy — the only path this
    # port runs). Each mirrors the corresponding tt-metal ModelArgs getter's DECODE dram_matmul_config
    # branch (get_attn_qkv_program_config / get_attn_output_program_config / get_mlp_ff1_3_prg_config /
    # get_mlp_ff2_prg_config / get_lm_head_program_config). num_devices==1 so cluster_shape[1]==1.
    def attn_qkv_decode_program_config(self):
        return self.dram_matmul_config(
            self.tile_padded_batch_rows, self.dim, self.qkv_size // self.num_devices,
            _num_cores_for_k(self.dim),  # attn_input_grid = dram_shard_core_grid_for_k(dim)
        )

    def attn_wo_decode_program_config(self):
        return self.dram_matmul_config(
            self.tile_padded_batch_rows, (self.n_heads * self.head_dim) // self.num_devices, self.dim,
            self.n_heads // self.num_devices,  # get_attn_output_program_config num_cores
        )

    def mlp_ff13_decode_program_config(self):
        n = self.hidden_dim // self.num_devices
        return self.dram_matmul_config(
            self.tile_padded_batch_rows, self.dim, n,
            _num_cores_for_k_and_n(self.dim, n),  # mlp_core_grid
        )

    def mlp_ff2_decode_program_config(self):
        k = self.hidden_dim // self.num_devices
        return self.dram_matmul_config(
            self.tile_padded_batch_rows, k, self.dim,
            _num_cores_for_k_and_n(k, self.dim),  # mlp2_core_grid
        )

    # Per-op PREFILL program configs (single-device, non-galaxy), mirroring the PREFILL branches of
    # tt-metal get_attn_qkv_program_config / get_attn_wo_program_config / get_mlp_ff1_3_prg_config /
    # get_mlp_ff2_prg_config / get_attn_sdpa_prefill_program_config at tt-metal f8c634a, the version the
    # BH p100a (261018) and WH n150 (261015) prefill captures ran on; every config these return matches the
    # program_config both captures record. Required on HW for the same reason as the decode ones: the
    # weights are DRAM width-sharded and ttnn.linear's auto config (MatmulMultiCoreProgramConfig) TT_FATALs on
    # a sharded B. None on Polaris (the shim ignores program_config), so the simulated graph is unchanged.
    # For llama3-8B, seq_len=128 these give 28-core QKV/WO/FF1/FF3/FF2 grids on P100 and 32-core on N150.
    def _prefill_hw_params(self):
        """(is_blackhole, is_p100, dram_shard_grid_width, prefill_len_cutoff) from the live device."""
        is_bh = ttnn.device.is_blackhole(self.mesh_device)  # type: ignore[attr-defined]
        dram_x = self.mesh_device.dram_grid_size().x
        is_p100 = is_bh and self.num_devices == 1 and dram_x == 7  # tt-metal determine_device_name
        dram_shard_grid_width = dram_x if is_bh else 8
        # tt-metal: 512 on BH; 1024 on WH, reduced to 512 for the 4096-dim models it lists on N150
        # (Llama-3.1-8B, Mistral-7B, ...). Only matters for seq_len > 512.
        prefill_len_cutoff = 512 if is_bh or (self.num_devices == 1 and self.dim == 4096) else 1024
        return is_bh, is_p100, dram_shard_grid_width, prefill_len_cutoff

    def _prefill_matmul_config(self, m, k, n, grid_size, in0_block_w=None, fuse_batch=False, per_core_N=None):
        """tt-metal ModelArgs.matmul_config -> MatmulMultiCoreReuseMultiCastProgramConfig."""
        per_core_M = math.ceil(m / (ttnn.TILE_SIZE * grid_size[1]))
        if per_core_N is None:
            per_core_N = math.ceil(n / (ttnn.TILE_SIZE * grid_size[0]))
        if in0_block_w is None:
            assert k % (ttnn.TILE_SIZE * grid_size[1]) == 0, 'Input width must be divisible by tile size times grid size'
            in0_block_w = _find_largest_divisor(k // (ttnn.TILE_SIZE * grid_size[1]))
        return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(  # type: ignore[attr-defined]
            compute_with_storage_grid_size=grid_size,
            in0_block_w=in0_block_w,
            out_subblock_h=1,
            out_subblock_w=_get_out_subblock_w(per_core_N, 1),
            per_core_M=per_core_M,
            per_core_N=per_core_N,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=fuse_batch,
        )

    @staticmethod
    def _require_short_prefill(seq_len, what):
        # tt-metal switches QKV / FF2 to ttnn.experimental.minimal_matmul (a different device op) above
        # 128 tokens; that path is not ported, so fail loudly instead of emitting a mismatched config.
        if seq_len > 128:
            raise NotImplementedError(f'{what} prefill for seq_len > 128 uses minimal_matmul in tt-metal; not ported')

    def attn_qkv_prefill_program_config(self, seq_len):
        if IS_POLARIS:
            return None
        self._require_short_prefill(seq_len, 'QKV')
        is_bh, is_p100, grid_w, _ = self._prefill_hw_params()
        return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(  # type: ignore[attr-defined]
            compute_with_storage_grid_size=(8, 10) if is_bh else (8, 8),
            in0_block_w=1,
            out_subblock_h=1,
            out_subblock_w=1,
            # tt-metal f8c634a: 1 on P100 ("P100 runs OOM in L1 with a larger per_core_M", issue
            # #50656), else max(1, ceil(seq_len / 32 / 8)); this branch only runs for seq_len <= 128, so
            # both are 1. BH p100a capture 261018: grid 8x10, per_core_M=1, per_core_N=28, 28 cores.
            per_core_M=1 if is_p100 else max(1, math.ceil(seq_len / ttnn.TILE_SIZE / 8)),
            per_core_N=math.ceil(self.qkv_size / self.num_devices / 32 / grid_w),
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=seq_len <= self.MAX_QKV_MM_SEQ_LEN,
        )

    def attn_sdpa_prefill_program_config(self, seq_len):
        # No chunked prefill here (chunk_start_idx is None), so q/k chunks are 256 from 2048 tokens, else 64.
        # Capture: SDPAProgramConfig(8x8, q_chunk 64, k_chunk 64, exp_approx_mode 0) on 64 cores.
        if IS_POLARIS:
            return None
        chunk = 256 if seq_len >= 2048 else 64
        return ttnn.SDPAProgramConfig(  # type: ignore[attr-defined]
            compute_with_storage_grid_size=(8, 8),
            exp_approx_mode=False,
            q_chunk_size=chunk,
            k_chunk_size=chunk,
        )

    def attn_wo_prefill_program_config(self, seq_len):
        if IS_POLARIS:
            return None
        _, _, grid_w, _ = self._prefill_hw_params()
        k = (self.n_heads * self.head_dim) // self.num_devices
        return self._prefill_matmul_config(
            m=min(seq_len, 1024), k=k, n=self.dim,
            grid_size=_find_prefill_grid(8, k // ttnn.TILE_SIZE),  # prefill_rows = 8
            fuse_batch=seq_len <= 1024,
            per_core_N=math.ceil(self.dim / (ttnn.TILE_SIZE * grid_w)),  # dram-sharded wo
        )

    def mlp_ff13_prefill_program_config(self, seq_len):
        if IS_POLARIS:
            return None
        _, _, grid_w, cutoff = self._prefill_hw_params()
        n = self.hidden_dim // self.num_devices
        return self._prefill_matmul_config(
            m=min(seq_len, cutoff), k=self.dim, n=n,
            grid_size=_find_prefill_grid(8, self.dim // ttnn.TILE_SIZE),  # mlp1_3_grid
            per_core_N=math.ceil(n / (ttnn.TILE_SIZE * grid_w)),
        )

    def mlp_ff2_prefill_program_config(self, seq_len):
        if IS_POLARIS:
            return None
        self._require_short_prefill(seq_len, 'FF2')
        _, _, grid_w, cutoff = self._prefill_hw_params()
        return self._prefill_matmul_config(
            m=min(seq_len, cutoff), k=self.hidden_dim, n=self.dim,
            grid_size=_find_prefill_grid(8, self.hidden_dim // ttnn.TILE_SIZE),  # mlp2_grid
            per_core_N=math.ceil(self.dim / (ttnn.TILE_SIZE * grid_w)),
        )

    def mlp_binary_mult_mem_config(self):
        """FF2-input (w2_in) width-sharded memory config on the mlp2 grid (tt-metal
        get_mlp_binary_mult_mem_config, DECODE). The SiLU-mul output is resharded onto this
        grid before the down-proj; matches the capture's 32x14336 L1_WIDTH_SHARDED reshard."""
        k = self.hidden_dim // self.num_devices
        num_cores = _num_cores_for_k_and_n(k, self.dim)  # mlp2_core_grid.num_cores
        return ttnn.create_sharded_memory_config(
            (self.tile_padded_batch_rows, k // num_cores),
            dram_shard_core_grid_for_k_and_n(k, self.dim),
            ttnn.ShardStrategy.WIDTH,
            ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

    def _lm_head_num_cores(self):
        # tt-metal lm_head_core_grid: num_rows starts 8 (single-device), shrink until
        # dim % (TILE*num_rows*cores_per_row)==0 with cores_per_row=8. dim=4096 -> 8x8=64.
        rows, cols = 8, 8
        while self.dim % (32 * rows * cols) != 0:
            rows -= 1
            assert rows > 0, f"no lm_head grid for dim={self.dim}"
        return rows * cols

    def lm_head_decode_program_config(self, split_size):
        return self.dram_matmul_config(
            self.tile_padded_batch_rows, self.dim, split_size, self._lm_head_num_cores(),
        )

    def lm_head_split_sizes(self):
        """Column-split sizes for the lm_head vocab (tt-metal get_lm_head_max_columns_per_device +
        LMHead.split_sizes_dram_sharded, commit d9d52dfe7b6). The vocab is column-tiled so each chunk's
        matmul fits L1. Single-chip BH: LLAMA_VOCAB_SIZE // NUM_LM_HEAD_COLUMNS = vocab//8 = 16032 ->
        8 chunks (matches the p100a capture's 8x16032); WH: 668 * lm_head_cores. A too-wide chunk (e.g.
        the old hardcoded 3-way 42752 split) overflows BH L1 in the lm_head matmul's circular buffers."""
        vocab = self.vocab_size
        if ttnn.get_arch_name() == 'blackhole':
            max_cols = vocab // 8   # single-chip BH (num_devices not 4/8)
        else:
            max_cols = 668 * self._lm_head_num_cores()
        n = math.ceil(vocab / max_cols)
        sizes = [min(vocab, max_cols)] * (n - 1)
        sizes.append(vocab - sum(sizes))   # remaining columns (last chunk)
        return sizes

    def weight_cache_path(self, dtype):
        return None

    def ccl_topology(self):
        return None

    def prepare_residual_tensor_decode(self, x, input_mem_cfg, args=None, force_replicated=False, on_host=False):
        batch = x.shape[0]
        seq_len = x.shape[1]
        assert x.shape[2] == self.dim
        x = ttnn.transpose(x, 0, 1).unsqueeze(0)
        return x

    def load_state_dict(self):
        return None

    def is_vision(self):
        return False

    def is_simulation(self):
        return True
