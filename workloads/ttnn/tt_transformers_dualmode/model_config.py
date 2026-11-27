#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Dual-mode ModelArgs for the llama3 tt_transformers port.

Based on the shim-only workloads/ttnn/tt_transformers/model_config.py, made
dual-mode (IS_POLARIS gate) and audited against tt-metal
models/tt_transformers/tt/model_config.py. Config-only + dummy weights — the
TT-Polaris path never reads an HF checkpoint.

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
        else:
            def _ckc(mf, approx):
                return ttnn.WormholeComputeKernelConfig(
                    math_fidelity=mf, math_approx_mode=approx,
                    fp32_dest_acc_en=True, packer_l1_acc=True,
                )
            self.compute_kernel_config_hifi2 = _ckc(ttnn.MathFidelity.HiFi2, True)
            self.compute_kernel_config_hifi2_fp16 = _ckc(ttnn.MathFidelity.HiFi2, True)
            self.compute_kernel_config_hifi4 = _ckc(ttnn.MathFidelity.HiFi4, False)
            self.compute_kernel_config_lofi = _ckc(ttnn.MathFidelity.LoFi, True)
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
