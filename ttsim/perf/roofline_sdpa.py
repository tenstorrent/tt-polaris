# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Single-chip SDPA roofline for Blackhole (TEN-4716).

A shared engine-cycle core (matmul + softmax) feeds per-variant front-ends (prefill/MLA/decode/
chunked/window/mask/cross/sparse). Compute-bound regimes export a compute cost; decode/paged export
DRAM KV-stream bytes; device.execute_op takes max(compute, memory).
"""
from __future__ import annotations
import json
import math
from dataclasses import dataclass, field, fields as dc_fields, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

TILE_HW = 32
BYTES_PER_TILE = {"bfp4_b": 576, "bfp8_b": 1088, "bfloat16": 2048, "float32": 4096}


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _wall_core_chunks(total_q_chunks, q_num_chunks, num_cores, causal):
    """Q chunks on the most loaded core and the cores that get work, following the program factory: causal
    with an even chunk count per head hands out light/heavy pairs, else a flat split plus the remainder."""
    if causal and q_num_chunks % 2 == 0:
        pairs = total_q_chunks // 2
        heavy = (pairs // num_cores) * 2 + (2 if pairs % num_cores else 0)
        return heavy, min(num_cores, pairs)
    return _ceil_div(total_q_chunks, num_cores), min(num_cores, total_q_chunks)


def _tail_cores(total_q_chunks, q_num_chunks, num_cores, causal):
    """Cores still reading once the light cores are done, i.e. the ones that took the remainder. The tail
    phase of the split runs with only these on the DRAM, so their step is cheaper than a contended one."""
    unit = 2 if (causal and q_num_chunks % 2 == 0) else 1
    units = total_q_chunks // unit
    if units <= num_cores:
        return min(num_cores, units)
    return (units % num_cores) or num_cores


# Constants are calibrated for Blackhole p100a only; must match the ttsim device package name so
# Device.execute_op can refuse the cost on a non-BH device. ttsim archs do not model SKUs, so the
# gate is package-level today; the calibrated SKU rides along in perf_stats for a SKU-aware backend.
POLARIS_CALIBRATED_DEVNAME = "Blackhole"
POLARIS_CALIBRATED_SKU = "p100a"

# FPU/SFPU overlap on the streaming kernel, fit on the R1f MATH counters of the 14 grid 1 configs: the exp
# issued from PACK hides a share r of the matmul time that grows with the q chunk height, zero at qct 2.
OVERLAP_R_CAUSAL = 0.310
OVERLAP_R_NONCAUSAL = 0.347
# MLA (qct 1), the fp32 DEST kernel and the joint and sparse kernels issue the exp on MATH: MATH = FPU + SFPU.
OVERLAP_FRAC_MLA = 0.0


def _overlap_hidden(qct, fpu, sfpu, causal_like, is_mla):
    """SFPU cycles hidden under the FPU on the streaming kernel: r (1 - 4 / qct^2) FPU, capped at SFPU."""
    if is_mla or qct < 2:
        return 0.0
    r = OVERLAP_R_CAUSAL if causal_like else OVERLAP_R_NONCAUSAL
    return min(float(sfpu), max(0.0, r * (1.0 - 4.0 / (qct * qct))) * fpu)


# Default floor for the tail rate. The tail runs cores that were contending a moment earlier, so DRAM does
# not recover to the rate an isolated small grid sustains; charging the measured 16 to 48 core rates for
# every regime (block GRIDR) took causal S1024 from -2.4 to -9.3 percent. Per-regime override below.
TAIL_RATE_MIN_CORES = 64


@dataclass(frozen=True)
class WallTerms:
    """Per k-chunk step wall law of one regime on the wall-setting core (bh/zone_decomposition.md).
    The step is the longest of three lanes: the DRAM K/V stream, the PACK issue thread and the
    compute floor plus its serial front end. A zero term is either measured zero or not fit yet."""
    pack_per_qtile: float = 0.0            # PACK issue lane per step: pack_per_step + qct * (pack_per_qtile + (pack_per_qktile + pack_per_qk_dtile * dct_sum) * kct)
    pack_per_qktile: float = 0.0           # the score-block part per q tile per k tile (independent of the head dim)
    pack_per_qk_dtile: float = 0.0         # the part per q tile per k tile per K or V tile of the head (the matmul views)
    pack_per_step: float = 0.0
    fe_flat_per_kchunk: float = 0.0        # serial front end per step above the lanes (joint kernel)
    fe_per_tile_mac: float = 0.0           # serial front end per tile MAC of the step (joint kernel)
    dram_law: str = "none"                 # "all_cores" (every core streams K/V), "injector" (chain head reads), "none"
    stream_scale: float = 1.0              # fitted factor on the stream lane (R1a regimes), 1 = the causal lane
    tail_phase: bool = True                # price the wall core's extra steps with only the remainder cores reading
    # How far DRAM recovers during that tail, as the fewest readers the rate table may be read at. The
    # causal family keeps the 64-core floor it was fitted with; MLA streams about eight times the bytes
    # per step (d_qk 576 plus d_v 512 against 128 and 128), and its tail reaches the uncontended rate
    # (GRIDR: the per-core rate is flat from 16 to 48 cores). Fitted on the wall table: dropping the
    # floor for MLA takes nh16 S1024 from +6.9 to +2.5 percent and the whole table from 1.77 to 1.71
    # mean absolute, while any shared floor that helps MLA pushes causal S1024 past -5.
    tail_rate_floor_cores: int = TAIL_RATE_MIN_CORES
    mask_branch_per_kchunk: float = 0.0    # causal-like only: lightweight mask bracket cost on the wall per step
    mask_per_tile: float = 0.0             # provided dense mask: serial cost per mask tile streamed and applied
    control_per_kchunk: float = 0.0        # un-zoned control flow per step: control_per_kchunk + control_per_qtile * qct
    control_per_qtile: float = 0.0
    fe_per_token: float = 0.0              # sparse kernel: serial cost per query token (Q read, index row, output)
    gather_rate_bpc: float = 0.0           # sparse kernel: selected key rows gathered per core, bytes per cycle
    low_confidence: str = ""               # reason raised whenever this row is used (wall-only fits)


# S 4096 nh32 nkv8 bfp8 HiFi2 zone runs (T2.1 / T2.2 / T2.3) on the p100a, firmware 19.9.0.
MASK_BRACKET_PER_STEP = 57.0               # A2 ablation: -9398 cycles over 165 steps at the anchor
# Per-q-k-tile PACK cost: 348 causal / 319.5 non-causal at head_dim 128 (T2.1 and A4 PACK bound walls); the
# R1c non-causal head_dim 64 wall splits it into a score part plus 26.45 per K or V tile of the head.
_PACK_PER_QK_DTILE = 26.45
_PACK_CAUSAL: Dict[str, Any] = dict(pack_per_qtile=750.0, pack_per_qktile=348.0 - 8 * _PACK_PER_QK_DTILE, pack_per_qk_dtile=_PACK_PER_QK_DTILE)
_PACK_NONCAUSAL: Dict[str, Any] = dict(pack_per_step=1136.0, pack_per_qtile=663.0, pack_per_qktile=319.5 - 8 * _PACK_PER_QK_DTILE,
                       pack_per_qk_dtile=_PACK_PER_QK_DTILE)
_CONTROL_CAUSAL: Dict[str, Any] = dict(control_per_kchunk=247.0, control_per_qtile=60.0)      # TRISC1 un-zoned time, 7 T2.1 causal runs
_CONTROL_NONCAUSAL: Dict[str, Any] = dict(control_per_kchunk=117.0, control_per_qtile=24.0)   # same, 7 non-causal runs
_CAUSAL_LIKE: Dict[str, Any] = dict(mask_branch_per_kchunk=MASK_BRACKET_PER_STEP, **_CONTROL_CAUSAL)

# Regimes whose terms are fit on device walls alone (masked, sparse, joint: R1a) or that carry a fitted
# factor on the K/V stream lane (windowed, chunked, MLA: R1a and T2.3 walls) are flagged.
WALL_ONLY_FIT = "wall_only_fit"
STREAM_RESIDUAL_FIT = "stream_residual_fit"

WALL_TERMS_BH: Dict[str, WallTerms] = {
    "prefill_causal": WallTerms(**_PACK_CAUSAL, dram_law="all_cores", **_CAUSAL_LIKE),
    "prefill_noncausal": WallTerms(**_PACK_NONCAUSAL, dram_law="injector", **_CONTROL_NONCAUSAL),
    # Cross attention runs the non-causal kernel with one q head per KV head, so its chains leave few
    # DRAM readers and the R1a walls (1024/8192, 4096/16384) sit on the PACK lane within 5 percent.
    "cross": WallTerms(**_PACK_NONCAUSAL, **_CONTROL_NONCAUSAL),
    # Windowed, chunked and MLA ride the causal K/V stream at their own bytes per step (T2.3 zones); the
    # factor is the wall core's stream against the 110-core law, refit on the R1a and T2.3 walls with the
    # tail phase in place. Chunked keeps a single rate: its four walls are held better by one factor that
    # absorbs the tail (9,244 to 9,764 cycles on the chunked tails) than by the tail law plus a factor.
    "windowed": WallTerms(dram_law="all_cores", stream_scale=0.9753, **_CAUSAL_LIKE,
                          low_confidence=STREAM_RESIDUAL_FIT),
    "chunked": WallTerms(dram_law="all_cores", stream_scale=0.8028, tail_phase=False, **_CAUSAL_LIKE,
                         low_confidence=STREAM_RESIDUAL_FIT),
    "mla": WallTerms(dram_law="all_cores", stream_scale=1.0400, tail_rate_floor_cores=0, **_CAUSAL_LIKE,
                     low_confidence=STREAM_RESIDUAL_FIT),
    # Provided dense mask (R1a S4096 d0.25, S8192 d0.5): the non-causal kernel visits every k chunk and
    # streams the mask tiles on top of its PACK lane; the cost per tile is density independent.
    "masked": WallTerms(**_PACK_NONCAUSAL, **_CONTROL_NONCAUSAL, mask_per_tile=200.2, low_confidence=WALL_ONLY_FIT),
    # Sparse (R1a H32 S2048 TOPK 1024 / 2048 at T 8192 / 16384): per token a serial part plus the gather
    # of the selected key rows; T does not enter (within 0.6 percent from 8192 to 32768).
    "sparse": WallTerms(fe_per_token=56689.0, gather_rate_bpc=2.979, low_confidence=WALL_ONLY_FIT),
    # Joint (R1a N4096 L333 nh24 d128 / d64, q128 k512): a serial cost per tile MAC of the step above the
    # floor; the two walls leave no flat per-step part.
    "joint": WallTerms(fe_per_tile_mac=161.5, **_CONTROL_NONCAUSAL,
                       low_confidence=WALL_ONLY_FIT),
}

# Envelope the standard-kernel wall laws were measured in; anything outside is flagged.
CALIBRATED_NUM_CORES = 110
CALIBRATED_GRIDS = (110, 64)              # K/V stream rate measured at both (T2.3)
CALIBRATED_Q_CHUNKS = (64, 128, 256, 512)
CALIBRATED_K_CHUNKS = (128, 256, 512)
CALIBRATED_FIDELITY = "HiFi2"
CALIBRATED_INJECTORS = 32                 # non-causal chain heads behind the injector-lane fits (nh32, one chain per head)
# Decode: the paged KV stream rate was measured on the 8x8 and 11x10 grids at 32-row pages (T2.8), the
# non-paged rate on the 11x10 grid (R1b).
DECODE_CALIBRATED_GRIDS = (64, 110)
DECODE_NONPAGED_CALIBRATED_GRID = 110
DECODE_CALIBRATED_PAGE_BLOCK = 32
# Sparse kernel fit family (R1a): 32 heads, bf16 latent K of 576, k chunk 128.
SPARSE_CALIBRATED = dict(num_heads=32, kv_dtype="bfloat16", k_chunk=128)


@dataclass
class ArchConfig:
    """Per-Tensix-core hardware geometry. Defaults are BH P100."""
    name: str = "BH"
    fpu_m_per_cycle: int = 8       # llk_math_matmul.h
    fpu_k_per_cycle: int = 16
    fpu_n_per_cycle: int = 16
    # Compute floor on the R1f perf counters (unmodified kernel): 16 cycles per tile MAC per fidelity level, the
    # anchor's FPU stepping by 339431 cycles per level from LoFi to HiFi3 with a fidelity independent part.
    cycles_per_tile_mac: Dict[str, float] = field(
        default_factory=lambda: {"LoFi": 16.0, "HiFi2": 32.0, "HiFi3": 48.0, "HiFi4": 64.0}
    )
    # FPU = cpt x (tile MACs x (1 + frac) + 0.957 tile MACs per q tile per step) + 74.3 cycles per q tile per
    # step, fit on the 14 R1f grid 1 configs and the LoFi / HiFi2 / HiFi3 anchors, all within 0.25 percent.
    fpu_overhead_frac: float = 0.0489
    fpu_overhead_tile_macs_per_qtile_step: float = 0.957
    fpu_overhead_cycles_per_qtile_step: float = 74.3
    sfpu_lanes: int = 32
    exp_tile_cycles: float = 70.0             # approx exp per tile: A6 exp-stub ablation, 172032 SFPU cycles over 2457.6 tiles
    # Accurate exp per tile: the streaming kernel from the R1f exp-accurate anchor capture (SFPU 271839 against
    # 304463 approx over 2457.6 tiles); the fp32 DEST kernel issues it from MATH at 75.3 (T2.4 g64, R1f g110).
    exp_tile_cycles_accurate: float = 56.7
    exp_tile_cycles_accurate_legacy: float = 75.3
    recip_cycles: float = 17.0                # per q tile per q chunk, below the counter resolution
    # SFPU per k-chunk step beyond the exp: 47.8 per step plus 204.2 per q tile per step (T2.1 SFPU counters, 15 points)
    sfpu_overhead_per_step: float = 47.8
    sfpu_overhead_per_qtile_step: float = 204.2
    unpacker_bw_bytes_per_cycle: float = 80.0
    packer_bw_bytes_per_cycle: float = 64.0
    idle_per_inner_iter: float = 0.0
    # Wall fixed cost per launch on the device-wall basis: kernel prologue 171 + tail 1706 + between
    # q chunks 379 on the wall core plus the 630 to 720 cycle start skew (T2.1, zone_decomposition.md).
    wall_fixed_cycles: float = 2900.0
    # Prefill K/V stream per core, cycles per k tile = fixed + bytes / rate: two-point bfp8 / bf16 fit at 110
    # cores (T2.3), 64-grid rate measured, others interpolate; the R1c causal head_dim 64 wall splits fixed.
    kv_stream_fixed_per_ktile: float = 405.8
    kv_stream_fixed_per_kv_tile: float = 35.65
    # 110-core rate refit with the tail phase, pinned on the anchor (the old 2.730 was a single rate over
    # both phases of that same wall, so it read faster than the contended phase really is).
    # T2.3 fitted 110 and 64; block GRIDR added 48, 32 and 16 (same anchor config, smaller grids), where
    # the per-core rate saturates because DRAM stops being contended. Standalone grids only: the tail phase
    # keeps the old floor, see TAIL_RATE_MIN_CORES.
    kv_stream_rate_bpc: Dict[int, float] = field(default_factory=lambda: {
        16: 5.7446, 32: 5.7391, 48: 5.7170, 64: 4.121, 110: 2.6767})
    # Non-causal chains: only the chain heads read DRAM; their rate falls with the chunk size
    # (5.39 / 5.25 / 4.80 B per cycle at k128 / k256 / k512 over the 691 fixed part, T2.1 q64 k128, q128 k256 / k512).
    kv_injector_rate_bpc: float = 5.615
    kv_injector_rate_bpc_per_ktile: float = -0.05
    exp_issue_cycles_per_tile: float = 79.9   # PACK thread SFPU exp issue per tile (A6 zone EXP)
    # Legacy (fp32 DEST) path: DEST round trips and CB waits above the floor, a per-step part plus a part
    # per DEST window of 4 tiles, fit on the T2.4 S1024 q64 and S4096 q256 walls (window scaling inferred).
    legacy_step_cycles: float = 2406.3
    dest_roundtrip_cycles: float = 835.6
    dest_window_tiles: int = 4
    # Wall front-end terms per regime; None selects the BH table (WALL_TERMS_BH).
    wall_terms: Optional[Dict[str, WallTerms]] = None
    # Dedicated SFPU issue thread (off on BH): the kernel is priced as a graph of DEST sections (KernelDescriptor,
    # evaluate_step). Every constant below is None on purpose and comes from the caller (from_overrides).
    thread_split: bool = False
    fpu_cycles_per_pass: Optional[Dict[str, float]] = None    # FPU cycles per matmul tile pass, per fidelity
    t_unpack_pass: Optional[float] = None        # operand unpack per tile pass; the matmul lane is max(FPU, unpack)
    t_pack_pass: Optional[float] = None          # packer cycles per tile
    sfpu_slots_per_tile: Optional[Dict[str, float]] = None    # SFPU op name (exp paths included) -> slots per tile
    f_riscv: Optional[float] = None              # issue floor over engine time for RISC-V issued SFPU bodies, >= 1
    t_handoff: Optional[float] = None            # ring sync cost per DEST section
    t_l1_turn: Optional[float] = None            # L1 turnaround per dependent boundary (pack to first op)
    c_arb: Optional[float] = None                # DEST port arbitration share of min(SFPU, PACK) lanes, within [0, 1]
    startup_a: Optional[float] = None            # startup transient per pass: a + b * (qct - g0)
    startup_b: Optional[float] = None
    startup_g0: Optional[float] = None
    compute_units_per_core: Optional[int] = None  # compute units sharing a core's q chunks (1 when not thread_split)
    dest_tiles_half: Optional[int] = None        # 16-bit tiles one DEST half section holds
    # Decode (memory-bound, T2.8 paged sweep): a fixed launch cost, a cost per KV head group the wall
    # core runs in sequence (the batch axis) and the KV stream at the launched grid's rate.
    decode_fixed_overhead_cycles: float = 13000.0          # 9.6 us
    decode_fixed_per_head_group_cycles: float = 930.0      # 0.7 us per sequential (user, KV head) group
    decode_kv_stream_gbps_paged: Dict[int, float] = field(default_factory=lambda: {64: 298.9, 110: 330.5})
    # Non-paged decode on the 11x10 grid (R1b: b32 bfp8 and b8 bf16 K/V at cache 1024 / 4096, k_chunk 128).
    decode_kv_stream_gbps_nonpaged: float = 342.3
    # MLA decode (R1b, R1h): the latent KV stream at one rate for the non-paged bf16 (V read) and the paged
    # bfp8 (K reused as V, Q sharded) forms, plus a fixed cost that belongs to the path, not to the query
    # slicing. The R1h sweep varies the slice count 1 / 2 / 4 at batch 4 and 8 (query heads 32 / 64 / 128 on
    # a 64-core shard) and the fixed cost comes out flat per paged wall while the KV bytes scale with the
    # slice count, so the slice dependence sits in the bytes alone (the reader re-reads the latent cache
    # once per slice; sdpa_decode_program_factory.cpp:135-153). Each fixed cost is the median of measured
    # minus stream over its path's walls, rounded to the nearest hundred cycles.
    # The paged fixed cost also moves with the cores the factory leaves on one head group (R1i holds the
    # bytes and the slices and sweeps max_cores_per_head_batch): the wall reads 112.1 / 79.6 / 71.2 /
    # 80.4 / 91.8 us at 1 / 2 / 3 / 4 / 6 cores per group on the same bytes, so more cores per group is not
    # better and the factory's own cap lands on the wrong side of the minimum. The table is the
    # median of measured minus stream at each measured cores-per-group value, interpolated between them and
    # clamped outside.
    decode_kv_stream_gbps_mla: float = 342.1
    decode_fixed_overhead_cycles_mla: float = 25800.0         # 19.1 us, non-paged form (R1b cache sweep)
    decode_fixed_cycles_mla_paged_by_cores_per_group: Dict[int, float] = field(default_factory=lambda: {
        1: 63600.0, 2: 62900.0, 3: 55700.0, 4: 60500.0, 6: 79700.0, 8: 72400.0, 13: 78700.0, 16: 69200.0})
    clock_ghz: float = 1.35

    def cpt(self, fidelity: str) -> float:
        try:
            return self.cycles_per_tile_mac[fidelity]
        except KeyError:
            raise ValueError(
                f"unsupported fidelity {fidelity!r}; supported: {sorted(self.cycles_per_tile_mac)}"
            ) from None

    def wall_terms_for(self, regime: str) -> WallTerms:
        table = self.wall_terms if self.wall_terms is not None else WALL_TERMS_BH
        try:
            return table[regime]
        except KeyError:
            raise ValueError(f"no wall terms for regime {regime!r} on arch {self.name!r}") from None

    @classmethod
    def from_overrides(cls, path, base: Optional["ArchConfig"] = None) -> "ArchConfig":
        """Arch constants from a yaml or json mapping of ArchConfig field names to values that lives outside the
        repo; unknown keys are refused. Starts from base (the BH defaults when None), so a file may set a subset."""
        path = Path(path)
        text = path.read_text()
        if path.suffix.lower() in (".yaml", ".yml"):
            import yaml
            data = yaml.safe_load(text)
        else:
            data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError(f"{path}: expected a mapping of ArchConfig field names to values")
        names = {f.name for f in dc_fields(cls)}
        unknown = sorted(k for k in data if k not in names)
        if unknown:
            raise ValueError(f"{path}: unknown ArchConfig fields {unknown}")
        values = dict(data)
        # json keys are strings; the per-grid tables are keyed by core count.
        for key in ("kv_stream_rate_bpc", "decode_kv_stream_gbps_paged"):
            if isinstance(values.get(key), dict):
                values[key] = {int(k): float(v) for k, v in values[key].items()}
        if isinstance(values.get("wall_terms"), dict):
            values["wall_terms"] = {k: WallTerms(**v) for k, v in values["wall_terms"].items()}
        return replace(base if base is not None else cls(), **values)


# ---- dedicated SFPU issue thread: the kernel as a graph of DEST sections -------------------------
THREAD_SPLIT_UNCALIBRATED = "thread_split_uncalibrated_in_repo"
THREAD_SPLIT_FIELDS = ("fpu_cycles_per_pass", "t_unpack_pass", "t_pack_pass", "sfpu_slots_per_tile", "f_riscv",
                       "t_handoff", "t_l1_turn", "c_arb", "startup_a", "startup_b", "startup_g0",
                       "compute_units_per_core", "dest_tiles_half")
THREAD_SPLIT_TERMS = ("fpu_stream", "sfpu_stream", "pack_stream", "issue_share", "dep_serial", "dest_arb",
                      "sync_pingpong", "startup_transient")
DST_SYNCS = ("half", "full")
STATS_PLACEMENTS = ("fpu_pool", "sfpu")
HANDOFF_CHAINS = ("dest", "l1")


@dataclass(frozen=True)
class Section:
    """One DEST section of a k chunk step: what each thread issues for it. Every cost comes from ArchConfig."""
    name: str
    tile_passes: float = 0.0        # matmul tile passes on the FPU, one operand unpack pass each
    fpu_tiles: float = 0.0          # eltwise, reduce and copy tiles on the FPU pool, one unpack pass each
    sfpu: Dict[str, float] = field(default_factory=dict)   # SFPU op name -> tiles (keys of sfpu_slots_per_tile)
    pack_tiles: float = 0.0         # tiles the packer drains, L1 accumulate passes included


@dataclass(frozen=True)
class StepGraph:
    """Ordered sections of one k chunk step. dependent[i] marks the boundary from section i to i + 1 as an L1
    data edge (the next unpacker waits on tiles this section packs); the last boundary wraps to the next step."""
    sections: Tuple[Section, ...]
    dependent: Tuple[bool, ...]

    def __post_init__(self):
        if not self.sections or len(self.dependent) != len(self.sections):
            raise ValueError("StepGraph needs at least one section and one boundary flag per section")

    @property
    def n_dependent(self) -> int:
        return sum(1 for d in self.dependent if d)


@dataclass(frozen=True)
class KernelDescriptor:
    """How the split-thread kernel is written. With the graphs None the standard SDPA step is built for the
    config geometry (default_sdpa_step) from the choices below; a caller may supply its own graphs instead."""
    dst_sync: str = "half"             # half: two DEST banks ping-pong; full: one bank, every boundary serialises
    exp_path: str = "exp"              # sfpu_slots_per_tile key the exp sections use
    stats_placement: str = "fpu_pool"  # row max, sub and row sum on the FPU pool and packer, or on the SFPU
    handoff_chain: str = "dest"        # dest: the SFPU is a DEST client between FPU and PACK; l1: it reads packed tiles
    step: Optional[StepGraph] = None        # a visited k chunk with the running max, sum and output rescaled
    first_step: Optional[StepGraph] = None  # the first k chunk of a q chunk (nothing to rescale); None = step
    last_step: Optional[StepGraph] = None   # the last k chunk (rescale plus normalise); None = step

    def __post_init__(self):
        for value, allowed, what in ((self.dst_sync, DST_SYNCS, "dst_sync"),
                                     (self.stats_placement, STATS_PLACEMENTS, "stats_placement"),
                                     (self.handoff_chain, HANDOFF_CHAINS, "handoff_chain")):
            if value not in allowed:
                raise ValueError(f"{what} must be one of {allowed}, got {value!r}")


def default_sdpa_step(qct, kct, dct_qk, dct_v, tiles_per_section, *, rescale, normalize, mask=False,
                      exp_path="exp", stats_placement="fpu_pool", handoff_chain="dest") -> StepGraph:
    """Standard flash SDPA k chunk step as DEST sections, per q tile: QK^T in sub passes of tiles_per_section
    tiles (independent boundaries), the mask add when present, the row max and running max update, the max
    subtraction with the exp and row sum in the same sub passes, P.V, then the rescale, normalise or parking
    copies. Other boundaries are L1 data edges; each FPU-only dest section carries one SFPU passthrough."""
    if min(qct, kct, dct_qk, dct_v, tiles_per_section) < 1:
        raise ValueError("qct, kct, dct_qk, dct_v and tiles_per_section must be positive")
    on_fpu = stats_placement == "fpu_pool"
    ring = handoff_chain == "dest"
    pt = {"passthrough": 1.0} if ring else {}
    secs: List[Section] = []
    dep: List[bool] = []

    def add(name, dependent, **kw):
        secs.append(Section(name, **kw))
        dep.append(dependent)

    def sub_passes(tiles):
        full, rest = divmod(tiles, tiles_per_section)
        return [tiles_per_section] * full + ([rest] if rest else [])

    for _ in range(qct):
        qk = sub_passes(kct)
        for c, n in enumerate(qk):
            add(f"qk{c}", c == len(qk) - 1, tile_passes=n * dct_qk,
                sfpu=(pt if on_fpu else {"row_max": float(n)}), pack_tiles=n)
        if mask:
            add("mask", True, fpu_tiles=kct, sfpu=pt, pack_tiles=kct)
        if on_fpu:
            add("row_max", True, fpu_tiles=kct, sfpu=pt, pack_tiles=1)
        # the running max: the chunk max alone on the first visit, combined with the parked max afterwards
        if rescale:
            add("max_update", True, fpu_tiles=2, sfpu={"max_combine": 1.0}, pack_tiles=1)
        else:
            add("max_update", True, fpu_tiles=1, sfpu=pt, pack_tiles=1)
        ex = sub_passes(kct)
        for c, n in enumerate(ex):
            if on_fpu:
                add(f"exp{c}", c == len(ex) - 1, fpu_tiles=n, sfpu={exp_path: float(n)}, pack_tiles=2 * n)
            else:
                add(f"exp{c}", c == len(ex) - 1, fpu_tiles=n,
                    sfpu={"sub": float(n), exp_path: float(n), "row_sum": float(n)}, pack_tiles=n)
        pv = sub_passes(dct_v)
        for c, n in enumerate(pv):
            add(f"pv{c}", True, tile_passes=n * kct, sfpu=pt, pack_tiles=n)
        if rescale:
            add("alpha", True, fpu_tiles=1, sfpu={exp_path: 1.0}, pack_tiles=1)
            add("sum_rescale", True, fpu_tiles=1, sfpu=pt, pack_tiles=1)
        if normalize:
            add("row_sum_final", True, fpu_tiles=(1 if on_fpu else 0), sfpu={"recip": 1.0}, pack_tiles=1)
        if rescale:
            add("out_rescale", True, fpu_tiles=dct_v, sfpu=pt, pack_tiles=dct_v)
        if normalize:
            add("normalize", False, fpu_tiles=dct_v, sfpu=pt, pack_tiles=dct_v)
        else:
            add("park_max", True, fpu_tiles=1, sfpu=pt, pack_tiles=1)
            add("park_sum", True, fpu_tiles=1, sfpu=pt, pack_tiles=1)
            add("park_out", False, fpu_tiles=dct_v, sfpu=pt, pack_tiles=dct_v)
    return StepGraph(tuple(secs), tuple(dep))


@dataclass
class StepCost:
    """One k chunk step on a split-thread compute unit: the FPU thread's steady-state period split into the named
    terms (terms() sums to period) plus the three lane totals for reference (not summands)."""
    period: float = 0.0
    fpu_stream: float = 0.0        # FPU lane, always fully on the period
    sfpu_stream: float = 0.0       # SFPU engine time exposed at L1 data edges
    pack_stream: float = 0.0       # packer time exposed at L1 data edges
    issue_share: float = 0.0       # exposed excess of the SFPU issue floor over its engine time
    dep_serial: float = 0.0        # n_sections * t_handoff + n_dependent * t_l1_turn
    dest_arb: float = 0.0          # c_arb * min(SFPU lane, PACK lane) on the dest chain
    sync_pingpong: float = 0.0     # FPU stalls on DEST bank recycling at independent boundaries
    fpu_lane: float = 0.0
    sfpu_lane: float = 0.0
    pack_lane: float = 0.0
    n_sections: int = 0
    n_dependent: int = 0

    def terms(self) -> Dict[str, float]:
        return {"fpu_stream": self.fpu_stream, "sfpu_stream": self.sfpu_stream, "pack_stream": self.pack_stream,
                "issue_share": self.issue_share, "dep_serial": self.dep_serial, "dest_arb": self.dest_arb,
                "sync_pingpong": self.sync_pingpong}


@dataclass(frozen=True)
class _SplitConstants:
    """The thread_split fields of an ArchConfig once every one of them is supplied and in range."""
    fpu_cycles_per_pass: Dict[str, float]
    t_unpack_pass: float
    t_pack_pass: float
    sfpu_slots_per_tile: Dict[str, float]
    f_riscv: float
    t_handoff: float
    t_l1_turn: float
    c_arb: float
    startup_a: float
    startup_b: float
    startup_g0: float
    compute_units_per_core: int
    dest_tiles_half: int


def _split_constants(a: ArchConfig, fidelity: str) -> _SplitConstants:
    """Refuse a split-thread prediction until every constant is supplied and in range."""
    missing = [n for n in THREAD_SPLIT_FIELDS if getattr(a, n) is None]
    if missing:
        raise ValueError(f"thread_split needs these ArchConfig fields, None here: {', '.join(missing)}")
    k = _SplitConstants(**{n: getattr(a, n) for n in THREAD_SPLIT_FIELDS})
    if fidelity not in k.fpu_cycles_per_pass:
        raise ValueError(f"fpu_cycles_per_pass has no entry for fidelity {fidelity!r}")
    if min(k.t_unpack_pass, k.t_pack_pass, k.t_handoff, k.t_l1_turn) < 0:
        raise ValueError("t_unpack_pass, t_pack_pass, t_handoff and t_l1_turn must be >= 0")
    if k.f_riscv < 1.0:
        raise ValueError("f_riscv is the issue floor over the engine time and cannot be below 1")
    if not (0.0 <= k.c_arb <= 1.0):
        raise ValueError("c_arb must be within [0, 1]")
    if k.compute_units_per_core < 1 or k.dest_tiles_half < 1:
        raise ValueError("compute_units_per_core and dest_tiles_half must be >= 1")
    return k


def _section_lanes(graph: StepGraph, k: _SplitConstants, fidelity: str):
    """Per section lane times: FPU (matmul passes unpack gated, eltwise tiles one pass each), SFPU engine slots and
    PACK passes. Names the SFPU ops the arch table lacks."""
    fpu_pass = max(k.fpu_cycles_per_pass[fidelity], k.t_unpack_pass)
    missing = sorted({op for s in graph.sections for op in s.sfpu} - set(k.sfpu_slots_per_tile))
    if missing:
        raise ValueError(f"sfpu_slots_per_tile lacks the SFPU ops {missing} the kernel descriptor uses")
    f = [s.tile_passes * fpu_pass + s.fpu_tiles * k.t_unpack_pass for s in graph.sections]
    e = [sum(k.sfpu_slots_per_tile[op] * n for op, n in s.sfpu.items()) for s in graph.sections]
    p = [s.pack_tiles * k.t_pack_pass for s in graph.sections]
    return f, e, p


def _steady_period(f, s, p, dependent, *, n_banks, sfpu_in_ring, max_reps=64):
    """Steady-state FPU period of one step: max-plus recurrence over the sections, repeated until two periods
    agree (last two averaged). Returns fpu busy, sfpu exposed, pack exposed and bank stalls, each by cause."""
    N = len(f)
    fpu_end = s_end = p_end = 0.0
    prev_first = prev_out = 0.0           # the previous section: end of its first stage after the FPU, output ready
    pack_ends: List[float] = []
    acc: List[List[float]] = []
    for rep in range(max_reps):
        acc.append([0.0, 0.0, 0.0, 0.0])
        for i in range(N):
            k = rep * N + i
            bank = pack_ends[k - n_banks] if k >= n_banks else 0.0
            by_data = k > 0 and dependent[i - 1 if i else N - 1] and prev_out >= bank
            release = prev_out if by_data else bank
            stall = max(0.0, release - fpu_end)
            if stall > 0.0:
                bucket = acc[(k - 1) // N]     # the stall belongs to the boundary that caused it
                if by_data:
                    first = min(max(prev_first - fpu_end, 0.0), stall)
                    bucket[1 if sfpu_in_ring else 2] += first
                    bucket[2 if sfpu_in_ring else 1] += stall - first
                else:
                    bucket[3] += stall
            fpu_end = fpu_end + stall + f[i]
            acc[rep][0] += f[i]
            if sfpu_in_ring:
                s_end = max(s_end, fpu_end) + s[i]
                p_end = max(p_end, s_end) + p[i]
                prev_first, prev_out = s_end, p_end
            else:
                p_end = max(p_end, fpu_end) + p[i]
                s_end = max(s_end, p_end) + s[i]
                prev_first, prev_out = p_end, s_end
            pack_ends.append(p_end)
        # a repetition is complete once the stall before the next one's first section is booked
        done = [sum(x) for x in acc[:rep]]
        if len(done) >= 4 and abs(done[-1] - done[-3]) < 1e-9 and abs(done[-2] - done[-4]) < 1e-9:
            break
    last = acc[-3:-1] if len(acc) >= 3 else acc[:-1] or acc
    return [sum(x[j] for x in last) / len(last) for j in range(4)]


def evaluate_step(graph: StepGraph, kernel: KernelDescriptor, a: ArchConfig, fidelity: str) -> StepCost:
    """Cost of one k chunk step on one compute unit under the kernel's sync and chain choices."""
    k = _split_constants(a, fidelity)
    f, e, p = _section_lanes(graph, k, fidelity)
    s = [x * k.f_riscv for x in e]
    ring = kernel.handoff_chain == "dest"
    fpu, sfpu_x, pack_x, pingpong = _steady_period(f, s, p, graph.dependent,
                                                   n_banks=(2 if kernel.dst_sync == "half" else 1), sfpu_in_ring=ring)
    c = StepCost(fpu_lane=sum(f), sfpu_lane=sum(s), pack_lane=sum(p), n_sections=len(f),
                 n_dependent=graph.n_dependent)
    c.fpu_stream, c.pack_stream, c.sync_pingpong = fpu, pack_x, pingpong
    c.sfpu_stream = sfpu_x / k.f_riscv
    c.issue_share = sfpu_x - c.sfpu_stream
    c.dep_serial = c.n_sections * k.t_handoff + c.n_dependent * k.t_l1_turn
    c.dest_arb = k.c_arb * min(c.sfpu_lane, c.pack_lane) if ring else 0.0
    c.period = sum(c.terms().values())
    return c


def _thread_split_wall(r: "RooflineResult", cfg: "SdpaConfig", kernel: Optional[KernelDescriptor], *,
                       Q, K_eff, qct, kct, dct_qk, dct_v):
    """Wall of the most loaded compute unit: its q chunks, each visiting the first, middle and last step graphs
    K_eff times in all, priced by evaluate_step, plus the startup transient per pass. Fills r's components."""
    a = cfg.arch
    k = _split_constants(a, cfg.fidelity)
    kernel = kernel or KernelDescriptor()
    tiles = k.dest_tiles_half * (2 if kernel.dst_sync == "full" else 1) // (2 if cfg.accum_dtype == "float32" else 1)
    tiles = max(1, tiles)

    def build(rescale, normalize):
        return default_sdpa_step(qct, kct, dct_qk, dct_v, tiles, rescale=rescale, normalize=normalize,
                                 mask=cfg.has_attn_mask, exp_path=kernel.exp_path,
                                 stats_placement=kernel.stats_placement, handoff_chain=kernel.handoff_chain)

    step = kernel.step or build(True, False)
    first = kernel.first_step or (kernel.step if kernel.step else build(False, K_eff <= 1.0))
    last = kernel.last_step or (kernel.step if kernel.step else build(True, True))
    graphs = {"first": first, "step": step, "last": last}
    costs = {kind: evaluate_step(g, kernel, a, cfg.fidelity) for kind, g in graphs.items()}
    # visits per q chunk: the first chunk, the last one (normalise) and K_eff - 2 in between
    mix = {"first": min(1.0, K_eff), "last": min(1.0, max(0.0, K_eff - 1.0)), "step": max(0.0, K_eff - 2.0)}

    units = k.compute_units_per_core
    chunks_unit = _ceil_div(r.q_chunks_wall_core, units)
    steps_unit = r.steps_wall_core * chunks_unit / r.q_chunks_wall_core
    comps = {name: 0.0 for name in THREAD_SPLIT_TERMS}
    lanes = [0.0, 0.0, 0.0]
    for kind, n in mix.items():
        steps = steps_unit * n / K_eff
        for name, v in costs[kind].terms().items():
            comps[name] += steps * v
        for j, lane in enumerate((costs[kind].fpu_lane, costs[kind].sfpu_lane, costs[kind].pack_lane)):
            lanes[j] += steps * lane
    comps["startup_transient"] = chunks_unit * max(0.0, k.startup_a + k.startup_b * (qct - k.startup_g0))
    r.components = comps
    r.wall_clock_cycles = round(sum(comps.values()))
    r.kernel_path = "thread_split"
    r.compute_units, r.q_chunks_wall_unit = units, chunks_unit
    r.fpu_cycles, r.sfpu_cycles, r.pack_cycles = round(lanes[0]), round(lanes[1]), round(lanes[2])
    r.overlap_frac = 0.0
    r.dep_serial_cycles, r.dest_arb_cycles = round(comps["dep_serial"]), round(comps["dest_arb"])
    r.math_active_cycles = round(sum(comps.values()) - comps["startup_transient"])
    r.serial_sum_cycles = round(sum(lanes) + comps["dep_serial"] + comps["dest_arb"])
    r.init_overhead_cycles = 0
    r.compute_latency_cycles = r.wall_clock_cycles
    avg_steps_unit = Q * K_eff / units
    r.straggler_cycles = round(r.math_active_cycles * (steps_unit - avg_steps_unit) / steps_unit) if steps_unit else 0
    r.config_echo.update(compute_units=units, q_chunks_wall_unit=chunks_unit, dst_sync=kernel.dst_sync,
                         exp_path=kernel.exp_path, stats_placement=kernel.stats_placement,
                         handoff_chain=kernel.handoff_chain,
                         sections={kind: len(g.sections) for kind, g in graphs.items()})
    r.flag(THREAD_SPLIT_UNCALIBRATED)


@dataclass
class SdpaConfig:
    S: int                    # QUERY sequence length
    kv_seq: int = 0           # KV sequence length; 0 = same as S. Cross-attention sets kv_seq != S.
    batch: int = 1            # prefill work + DRAM scale linearly with batch
    head_dim: int = 128       # QK head dim (Q*K^T contraction)
    v_head_dim: int = 0       # P*V output dim; 0 means same as head_dim (symmetric SDPA)
    q_chunk: int = 128
    k_chunk: int = 128
    num_heads: int = 32       # query heads
    num_kv_heads: int = 0     # KV heads; 0 means same as num_heads (MHA). GQA/MQA read K/V fewer times.
    num_cores: int = 110
    fidelity: str = "HiFi2"
    input_dtype: str = "bfp8_b"
    accum_dtype: str = "bfloat16"
    is_causal: bool = True
    has_attn_mask: bool = False    # dense additive mask (masked prefill path)
    sliding_window: int = 0        # 0 = full attention; >0 = local window width (tokens)
    attention_sink: bool = False   # extra softmax-normalization pass over learned sink logits
    chunk_start_idx: int = 0       # chunked/paged prefill: absolute start of this Q chunk (prefix len)
    is_chunked: bool = False       # chunked prefill entry point; its first chunk has chunk_start_idx 0
    dram_scatter_derate: float = 1.0  # paged KV gather BW penalty (>1 inflates effective bytes)
    is_sparse: bool = False        # sparse-MLA: each query attends TOPK selected latent KV (kv_seq=TOPK)
    is_joint: bool = False         # SD3/Flux joint attention (own kernel; heavier per-iter dispatch)
    joint_seq: int = 0             # joint attention: the joint stream's length inside S (padded separately)
    exp_approx_mode: bool = True
    fp32_dest_acc: bool = False    # fp32 DEST accumulation selects the non-streaming compute kernel
    kv_input_dtype: str = ""       # K/V dtype when it differs from q (bf8 cache under a bf16 query); "" = input_dtype
    paged: bool = False            # K/V read through a page table (chunked prefill)
    page_block_size: int = 0       # paged block size in tokens, 0 = unknown
    is_windowed: bool = False      # cu_window_seqlens windowed mode (recorded, not priced)
    defaulted: str = ""            # shim echo: config fields that were absent at the call
    fallback_reasons: Tuple[str, ...] = ()   # model-side fallbacks taken while building the config
    arch: ArchConfig = field(default_factory=ArchConfig)
    kernel_rev: str = ""      # "" or "72620d5": the calibration tree; "main"/"2dbd14bf632": main tip


@dataclass
class RooflineResult:
    label: str = ""
    arch_name: str = ""
    regime: str = "prefill"            # prefill / decode / chunked / windowed / masked / sparse
    wall_regime: str = ""              # row of the wall-terms table used (prefill_causal, cross, ...)
    kernel_path: str = "streaming"     # "streaming" (bf16 DEST, exp on PACK), "legacy" (fp32 DEST), "thread_split"
    q_chunks_per_core: float = 0.0
    q_chunks_wall_core: int = 0        # chunks on the wall-setting (most loaded) core per the factory split
    steps_wall_core: float = 0.0       # k-chunk steps that core runs (chunks x visits; the max core under a window)
    active_cores: int = 0              # cores that receive at least one chunk (or pair)
    k_chunks_per_q: float = 0.0
    k_eff: float = 0.0
    inner_iters: float = 0.0
    fpu_matmul_cycles: int = 0
    fpu_overhead_cycles: int = 0
    fpu_cycles: int = 0
    sfpu_exp_cycles: int = 0
    sfpu_reduce_cycles: int = 0
    sfpu_recip_cycles: int = 0
    sfpu_overhead_cycles: int = 0
    sfpu_extra_cycles: int = 0         # mask-add + attention-sink + flash-decode combine passes
    sfpu_cycles: int = 0
    pack_cycles: int = 0               # thread_split: PACK lane on the wall unit
    overlap_frac: float = 0.0          # FPU/SFPU overlap applied (0 under thread_split)
    dep_serial_cycles: int = 0         # thread_split: ring hand-offs plus L1 turnarounds at dependent boundaries
    dest_arb_cycles: int = 0           # thread_split: DEST port arbitration share
    compute_units: int = 1             # thread_split: compute units sharing the core's q chunks
    q_chunks_wall_unit: int = 0        # thread_split: q chunks on the most loaded unit
    math_active_cycles: int = 0        # FPU/SFPU union (calibrated overlap) = the compute estimate
    serial_sum_cycles: int = 0         # fpu+sfpu, zero-overlap upper bracket (not the prediction)
    math_idle_cycles: int = 0
    init_overhead_cycles: int = 0
    compute_latency_cycles: int = 0    # compute FLOOR (FPU/SFPU busy + L1 floor + init)
    wall_clock_cycles: int = 0         # device wall (slowest core) = round(sum(components)) (decode: memory latency)
    components: Dict[str, float] = field(default_factory=dict)   # named wall terms; sum is the wall
    straggler_cycles: int = 0          # quantization loss: wall-core chunks over the average core
    unpack_bytes_total: int = 0        # per-core L1 unpacker bytes (drives the on-chip BW floor only)
    pack_bytes_total: int = 0
    unpack_min_cycles: int = 0
    pack_min_cycles: int = 0
    dram_in_bytes: int = 0             # whole-op DRAM read traffic (Q + K/V per KV head)
    dram_out_bytes: int = 0            # whole-op DRAM write traffic (output)
    is_mla: bool = False               # asymmetric v_head_dim != head_dim (FlashMLA)
    is_memory_bound: bool = False      # decode/paged: KV-stream DRAM bytes are the real bound
    low_confidence: bool = False       # any reason below applies
    low_confidence_reasons: List[str] = field(default_factory=list)
    defaulted: str = ""                # shim echo of config fields absent at the call
    config_echo: Dict = field(default_factory=dict)   # resolved config the prediction used
    clock_ghz: float = 0.0             # clock the DRAM terms were priced at (the device's, once re-priced)

    def flag(self, reason: str):
        if reason not in self.low_confidence_reasons:
            self.low_confidence_reasons.append(reason)
        self.low_confidence = True

    def to_polaris_op_perf_stats(self) -> Dict:
        """perf_stats for a fused SDPA op consumed by ttsim Device.execute_op.

        fused_compute_cycles is the full device wall-clock estimate (slowest-core wall; decode
        carries its memory latency). instrs is empty (ttsim reads it as instruction counts); the
        compute floor stays in the breakdown for reference.
        """
        return {
            "inBytes": self.dram_in_bytes,
            "outBytes": self.dram_out_bytes,
            "inElems": 0,
            "outElems": 0,
            "inParamCount": 0,
            "inActCount": 0,
            "outActCount": 0,
            "instrs": {},
            "fused_compute_cycles": self.wall_clock_cycles,
            # Device.execute_op rejects the cost off-BH and flags per-regime-calibrated variants.
            "sdpa_calibrated_arch": POLARIS_CALIBRATED_DEVNAME,
            "sdpa_calibrated_sku": POLARIS_CALIBRATED_SKU,
            "sdpa_compute_is_floor": False,
            "sdpa_mla_low_confidence": (self.is_mla or self.low_confidence),
            "sdpa_low_confidence_reasons": ",".join(self.low_confidence_reasons),
            # Device-dependent inputs of the estimate. Shape inference has no device, so it prices with
            # the calibration's own grid and clock and Device.execute_op re-prices with the device's.
            "sdpa_clock_ghz": self.clock_ghz,
            "sdpa_num_cores_defaulted": ("num_cores" in self.defaulted.split(",")
                                         or "grid_absent" in self.low_confidence_reasons),
            "sdpa_defaulted_attrs": self.defaulted,
            "sdpa_config": dict(self.config_echo),
            "sdpa_regime": self.regime,
            "sdpa_wall_regime": self.wall_regime,
            "sdpa_wall_clock_cycles": self.wall_clock_cycles,
            # Named wall terms on the wall-setting core; they sum to fused_compute_cycles.
            "sdpa_wall_components": dict(self.components),
            "sdpa_cycle_breakdown": {
                "compute_floor": self.compute_latency_cycles,
                "fpu_matmul": self.fpu_matmul_cycles,
                "fpu_overhead": self.fpu_overhead_cycles,
                "sfpu_exp": self.sfpu_exp_cycles,
                "sfpu_reduce": self.sfpu_reduce_cycles,
                "sfpu_recip": self.sfpu_recip_cycles,
                "sfpu_overhead": self.sfpu_overhead_cycles,
                "sfpu_extra": self.sfpu_extra_cycles,
                "math_active": self.math_active_cycles,
                "overlap_frac": self.overlap_frac,
                "dep_serial": self.dep_serial_cycles,
                "dest_arb": self.dest_arb_cycles,
                "straggler": self.straggler_cycles,
                "dram_in_bytes": self.dram_in_bytes,
                "is_memory_bound": self.is_memory_bound,
            },
        }


def _reprice_blob(regime: str, args: Dict) -> Dict:
    """The device-independent inputs of one op, JSON-safe, so the backend can re-price it (`reprice`)."""
    return {"regime": regime,
            "args": {k: (list(v) if isinstance(v, tuple) else v) for k, v in args.items() if k != "arch"}}


def reprice(blob: Dict, *, num_cores: Optional[int] = None, clock_ghz: Optional[float] = None,
            arch_name: Optional[str] = None) -> Dict:
    """Re-run the model for a recorded op with the device's own worker grid and clock. Shape inference
    is device independent by design, so it prices with the calibration's grid and clock and
    Device.execute_op calls this once it knows the device. The fitted constants do not move: they are
    calibration, and the SKU gate keeps them on the device they were measured on."""
    args = dict(blob["args"])
    arch = ArchConfig(name=arch_name or ArchConfig().name,
                      clock_ghz=float(clock_ghz) if clock_ghz else ArchConfig().clock_ghz)
    args["fallback_reasons"] = tuple(args.get("fallback_reasons") or ())
    if num_cores:
        args["num_cores"] = int(num_cores)
    if blob["regime"] == "decode":
        ps = predict_decode(arch=arch, **args).to_polaris_op_perf_stats()
    else:
        ps = predict(SdpaConfig(arch=arch, **args)).to_polaris_op_perf_stats()
    ps["sdpa_reprice"] = blob
    return ps


def sdpa_perf_stats(cfg: "SdpaConfig") -> Dict:
    ps = predict(cfg).to_polaris_op_perf_stats()
    ps["sdpa_reprice"] = _reprice_blob("prefill", {f.name: getattr(cfg, f.name) for f in dc_fields(cfg)})
    return ps


_ELEM_TO_DTYPE = {0.5: "bfp4_b", 1.0: "bfp8_b", 2.0: "bfloat16", 4.0: "float32"}


def _resolve_compute_attrs(attrs, reasons, *, fidelity_default="HiFi2"):
    """Compute kernel config fields with the kernel's own defaults when the call carried none."""
    fidelity = attrs.get("fidelity")
    if not fidelity:
        fidelity = fidelity_default
        reasons.append("fidelity_absent")
    fp32 = attrs.get("fp32_dest_acc_en")
    if fp32 is None:
        fp32 = False
        reasons.append("fp32_acc_absent")
    return str(fidelity), bool(fp32)


def _kv_dtype(attrs, input_dtype, reasons):
    kv_es = attrs.get("kv_element_size")
    if kv_es is None:
        reasons.append("kv_dtype_absent")
        return input_dtype
    return _ELEM_TO_DTYPE.get(float(kv_es), input_dtype)


def sdpa_config_from_shapes(q_shape, k_shape, v_shape, attrs=None, num_cores=110, arch=None,
                            page_table_shape=None):
    """Build an SdpaConfig from a prefill op's q/k/v shapes + attrs. BH-calibrated (arch=ARCH_BH).
    Absent program/compute config fields fall to what the kernel runs without them (32x32 chunks, the
    whole worker grid, approx exp, HiFi2, bf16 DEST) and are listed in fallback_reasons. Chunked
    prefill reads K/V through a page table, so its KV length is derived, not read from the tensor."""
    attrs = attrs or {}
    reasons: List[str] = []
    q, k, v = [list(map(int, s)) for s in (q_shape, k_shape, v_shape)]
    nh, S, head_dim = q[-3], q[-2], q[-1]
    batch = q[-4] if len(q) >= 4 else 1
    nkv, kv_seq, v_head_dim = k[-3], k[-2], v[-1]
    # MLA carries V in the latent form (V = K[..., :kv_lora]); its v_head_dim comes from the attr,
    # not the (larger) K tensor dim. Sparse/joint override kv_seq (TOPK / concat) via the attr too.
    v_head_dim = int(attrs.get("head_dim_v") or v_head_dim)
    kv_seq = int(attrs.get("kv_seq") or kv_seq)
    chunked = (attrs.get("sdpa_variant") == "chunked" or bool(attrs.get("paged"))
               or attrs.get("chunk_start_idx") is not None or bool(attrs.get("chunk_start_unknown")))
    chunk_start = int(attrs.get("chunk_start_idx") or 0)
    if chunked:
        # A paged K/V tensor is [blocks, nkv, block_size, D], so k[-2] is the block size: the attended
        # length is prefix + S. A tensor-form start is bounded by the page table capacity minus S.
        if not chunk_start and attrs.get("chunk_start_unknown") and page_table_shape:
            block = int(attrs.get("page_block_size") or k[-2])
            chunk_start = max(0, int(page_table_shape[-1]) * block - S)
        kv_seq = chunk_start + S
    dtype = str(attrs.get("input_dtype") or _ELEM_TO_DTYPE.get(float(attrs.get("element_size", 2)), "bfloat16"))
    kv_dtype = _kv_dtype(attrs, dtype, reasons)
    qc = int(attrs.get("q_chunk_size") or 0)
    kc = int(attrs.get("k_chunk_size") or 0)
    if not qc:
        qc, kc = 32, (kc or 32)
        reasons.append("program_config_absent")
    kc = kc or qc
    cores = int(attrs.get("num_cores") or 0)
    if not cores:
        cores = num_cores
        reasons.append("grid_absent")
    if "exp_approx_mode" in attrs:
        exp_approx = bool(attrs["exp_approx_mode"])
    else:
        exp_approx = True
        reasons.append("exp_mode_absent")
    fidelity, fp32 = _resolve_compute_attrs(attrs, reasons)
    if attrs.get("chunk_start_unknown"):
        reasons.append("chunk_start_unknown")
    if attrs.get("is_windowed"):
        reasons.append("windowed_mode_unmodelled")
    return SdpaConfig(
        S=S, kv_seq=(0 if kv_seq == S else kv_seq), batch=batch,
        head_dim=head_dim, v_head_dim=(0 if v_head_dim == head_dim else v_head_dim),
        q_chunk=qc, k_chunk=kc, num_heads=nh, num_kv_heads=(0 if nkv == nh else nkv),
        num_cores=cores, is_causal=bool(attrs.get("is_causal", True)),
        is_sparse=bool(attrs.get("is_sparse", False)),
        is_joint=bool(attrs.get("is_joint", False)),
        joint_seq=int(attrs.get("joint_seq") or 0),
        fidelity=fidelity,
        input_dtype=dtype, accum_dtype=("float32" if fp32 else "bfloat16"),
        kv_input_dtype=("" if kv_dtype == dtype else kv_dtype),
        has_attn_mask=bool(attrs.get("has_attn_mask", False)),
        sliding_window=int(attrs.get("sliding_window") or attrs.get("sliding_window_size") or 0),
        attention_sink=bool(attrs.get("attention_sink", False)),
        chunk_start_idx=chunk_start, is_chunked=chunked,
        dram_scatter_derate=float(attrs.get("dram_scatter_derate") or (1.15 if chunked else 1.0)),
        exp_approx_mode=exp_approx,
        fp32_dest_acc=fp32,
        paged=bool(attrs.get("paged", False) or chunked),
        page_block_size=int(attrs.get("page_block_size") or 0),
        is_windowed=bool(attrs.get("is_windowed", False)),
        defaulted=str(attrs.get("sdpa_defaulted") or ""),
        fallback_reasons=tuple(reasons),
        # Per-call arch instance: the module-level ARCH_BH singleton must never end up shared
        # (and mutable) across configs. Only honor an arch the caller actually supplied.
        arch=arch if arch is not None else ArchConfig(),
    )


def _causal_k_eff(S, q_chunk, k_chunk):
    """Average visited k chunks per q chunk on the causal path: q chunk i runs ceil((i + 1) q_chunk / k_chunk)
    chunks, capped at the chunk count; (K + 1) / 2 when k_chunk >= q_chunk (matches the T2.1 STEP_N)."""
    nq = max(1, _ceil_div(S, q_chunk))
    K = _ceil_div(S, k_chunk)
    return sum(min(K, _ceil_div((i + 1) * q_chunk, k_chunk)) for i in range(nq)) / nq


def _causal_visits(S, qct, kct, K):
    """Per q chunk: visited k chunks and k tiles processed. The streaming kernel narrows the diagonal chunk to
    the tiles at or below it (compute_streaming.hpp target_active_Sk): q chunk i does (i + 1) qct k tiles."""
    nq = max(1, _ceil_div(S, qct * TILE_HW))
    visits = [min(K, _ceil_div((i + 1) * qct, kct)) for i in range(nq)]
    tiles = [min(K * kct, (i + 1) * qct) for i in range(nq)]
    return visits, tiles


def _windowed_ranges(S, q_chunk, k_chunk, kv_seq, window, causal):
    """(k_lo, k_hi) chunk range of every q chunk under a sliding/local window. Saturates at window/k_chunk
    (unlike (K+1)/2 which grows with S)."""
    Sqt = q_chunk // TILE_HW
    Skt = k_chunk // TILE_HW
    nq = max(1, _ceil_div(S, q_chunk))
    win_tiles = _ceil_div(window, TILE_HW)
    kv_tiles = _ceil_div(kv_seq, TILE_HW)
    kv_chunks = _ceil_div(kv_tiles, Skt)
    ranges = []
    for g in range(nq):
        q_lo_t, q_hi_t = g * Sqt, g * Sqt + Sqt
        if causal:
            k_hi = min(kv_chunks, _ceil_div(q_hi_t, Skt))   # cap at kv length (windowed cross-attn)
            k_lo = max(0, q_lo_t - win_tiles) // Skt
        else:
            k_hi = min(_ceil_div(kv_tiles, Skt), _ceil_div(q_hi_t + win_tiles // 2, Skt))
            k_lo = max(0, q_lo_t - win_tiles // 2) // Skt
        ranges.append((k_lo, max(k_lo, k_hi)))
    return ranges


def _windowed_visits(S, q_chunk, k_chunk, kv_seq, window, causal):
    """Visited K-chunks of every q-chunk under a sliding/local window."""
    return [float(hi - lo) for lo, hi in _windowed_ranges(S, q_chunk, k_chunk, kv_seq, window, causal)]


def _windowed_tiles(S, q_chunk, k_chunk, kv_seq, window, causal):
    """K tiles processed per q chunk under a window: the band, narrowed at the causal diagonal."""
    qct, kct = q_chunk // TILE_HW, k_chunk // TILE_HW
    out = []
    for g, (lo, hi) in enumerate(_windowed_ranges(S, q_chunk, k_chunk, kv_seq, window, causal)):
        hi_tiles = min((g + 1) * qct, hi * kct) if causal else hi * kct
        out.append(float(max(0, hi_tiles - lo * kct)))
    return out


def _windowed_k_eff(S, q_chunk, k_chunk, kv_seq, window, causal):
    """Average visited K-chunks per q-chunk for a sliding/local window."""
    v = _windowed_visits(S, q_chunk, k_chunk, kv_seq, window, causal)
    return sum(v) / len(v)


def _wall_core_steps(visits, total_chunks, num_cores, causal):
    """Steps on the most loaded core when chunks visit different k chunk counts (windowed band): core i takes
    a contiguous flat range (sdpa_program_factory.cpp, zigzag when causal), the wall core the most visits."""
    nq = len(visits)
    if causal and nq % 2 == 0:
        pairs = total_chunks // 2
        base, extra_cores, extra = (pairs // num_cores) * 2, pairs % num_cores, 2
    else:
        base, extra_cores, extra = total_chunks // num_cores, total_chunks % num_cores, 1
    best = 0.0
    for i in range(num_cores):
        start = i * base + min(i, extra_cores) * extra
        if start >= total_chunks:
            break
        count = min(base + (extra if i < extra_cores else 0), total_chunks - start)
        steps = 0.0
        for idx in range(start, start + count):
            pos = idx % nq
            g = (nq - 1 - pos // 2) if (causal and pos % 2 == 1) else pos // 2 if causal else pos
            steps += visits[g]
        best = max(best, steps)
    return best


def _joint_split(num_cores, batch, num_heads, q_num_chunks):
    """Joint SDPA factory (joint_sdpa_program_factory.cpp): batch x heads groups of q_parallel cores, each
    with ceil(q chunks / q_parallel) chunks; (wall chunks, mean chunks, active cores) or None if too few."""
    groups = batch * num_heads
    if groups > num_cores:
        return None
    q_parallel = max(1, min(num_cores // groups, q_num_chunks))
    return _ceil_div(q_num_chunks, q_parallel), q_num_chunks / q_parallel, groups * q_parallel


def _wall_regime(cfg, r):
    """Row of the wall-terms table for this op: routed regime first, else derived from the config."""
    if cfg.is_sparse:
        return "sparse"
    if r.is_mla:
        return "mla"
    if r.regime in ("windowed", "masked", "chunked", "joint"):
        return r.regime
    if cfg.kv_seq and cfg.kv_seq != cfg.S:
        return "cross"
    return "prefill_causal" if cfg.is_causal else "prefill_noncausal"


def _loglog_by_cores(table, cores):
    """Log-log interpolation of a per-grid table between its measured grids, clamped outside them."""
    grids = sorted(table)
    if cores <= grids[0]:
        return table[grids[0]]
    if cores >= grids[-1]:
        return table[grids[-1]]
    lo = max(g for g in grids if g <= cores)
    hi = min(g for g in grids if g >= cores)
    if lo == hi:
        return table[lo]
    f = math.log(cores / lo) / math.log(hi / lo)
    return math.exp(math.log(table[lo]) * (1 - f) + math.log(table[hi]) * f)


def _kv_stream_rate(a, cores, *, tail=False):
    """Prefill K/V stream bytes per cycle per core at this many reading cores (T2.3 and GRIDR grids).
    tail=True applies the default tail floor; a regime with its own floor passes the clamped count."""
    return _loglog_by_cores(a.kv_stream_rate_bpc, max(cores, TAIL_RATE_MIN_CORES) if tail else cores)


def _decode_kv_stream_gbps(a, grid, paged):
    """Decode KV stream rate of the launched grid: the paged T2.8 rates interpolated between the measured
    grids; the non-paged R1b rate at the 110 grid, scaled by the paged grid ratio elsewhere."""
    rate = _loglog_by_cores(a.decode_kv_stream_gbps_paged, grid)
    if paged:
        return rate
    return a.decode_kv_stream_gbps_nonpaged * rate / _loglog_by_cores(a.decode_kv_stream_gbps_paged,
                                                                        DECODE_NONPAGED_CALIBRATED_GRID)


def _mla_paged_fixed_cycles(a, cores_per_group):
    """Paged MLA decode fixed cost at this cores-per-group split (R1i): the measured value where one was
    measured, a linear interpolation between the two neighbours otherwise, clamped at the ends."""
    t = a.decode_fixed_cycles_mla_paged_by_cores_per_group
    keys = sorted(t)
    c = max(keys[0], min(keys[-1], int(cores_per_group)))
    if c in t:
        return t[c], c in t and int(cores_per_group) == c
    lo = max(k for k in keys if k < c)
    hi = min(k for k in keys if k > c)
    return t[lo] + (t[hi] - t[lo]) * (c - lo) / (hi - lo), False


def _decode_core_split(num_cores, batch, num_kv_heads, max_cores_per_head_batch):
    """Factory split (sdpa_decode_program_factory.cpp:196-209): cores per KV head, KV head groups one core
    runs in sequence and the active core count (64 for batch 32 x 8 heads on both measured grids)."""
    groups = batch * num_kv_heads
    max_cph = max_cores_per_head_batch or num_cores
    cores_per_batch = max(1, min(num_cores, max_cph * groups) // batch)
    cores_per_head = max(1, cores_per_batch // num_kv_heads)
    heads_per_core = max(1, _ceil_div(num_kv_heads, cores_per_batch))
    while num_kv_heads % heads_per_core:
        heads_per_core += 1
    return cores_per_head, heads_per_core, cores_per_head * groups // heads_per_core


def _mla_q_head_slices(batch, num_q_heads, q_shard_cores):
    """q_heads_parallel_factor of the MLA decode factory (sdpa_decode_program_factory.cpp:134-148): ceil(heads
    / round32(batch x heads / shard cores)) for a non-replicated query; each head slice streams the cache."""
    if not q_shard_cores:
        return 1
    shard_h = max(TILE_HW, _ceil_div(batch * num_q_heads, q_shard_cores * TILE_HW) * TILE_HW)
    return max(1, _ceil_div(num_q_heads, shard_h))


def decode_dynamic_chunk_tiles(cur_pos, dst_tiles):
    """Kernel rule for k_chunk 0 on the paged entry (rt_args_common.hpp get_dynamic_Sk_chunk_t): the tile
    count of cur_pos + 1 rounded up to a power of two, capped at the DEST size in tiles."""
    seq_tiles = cur_pos // TILE_HW + 1
    tiles = 1
    while tiles < seq_tiles:
        tiles *= 2
    return min(tiles, dst_tiles)


def _noncausal_chains(q_num_chunks, total_q_chunks, num_cores):
    """Heads whose q chunks span more than one core under the flat non-causal split: the factory builds
    one K/V forwarding chain per such head and none for a head that sits on a single core."""
    if total_q_chunks <= 0 or q_num_chunks <= 0:
        return 0
    base, extra = divmod(total_q_chunks, num_cores)
    heavy = extra * (base + 1)               # the first `extra` cores take one chunk more

    def core_of(chunk):
        return chunk // (base + 1) if chunk < heavy else extra + (chunk - heavy) // base

    heads = total_q_chunks // q_num_chunks
    return sum(1 for h in range(heads) if core_of(h * q_num_chunks) != core_of((h + 1) * q_num_chunks - 1))


def _kv_stream_fixed(a, kct, dct_sum):
    """Fixed part of the stream lane per step: per k tile plus per K or V tile of the head (dct_sum tiles per k tile)."""
    return kct * (a.kv_stream_fixed_per_ktile + a.kv_stream_fixed_per_kv_tile * dct_sum)


def _pack_lane(t, qct, kct, dct_sum):
    """PACK issue lane per step: a per-step part, a per-q-tile part and a per-q-k-tile part with its per-head-tile share."""
    return t.pack_per_step + qct * (t.pack_per_qtile + (t.pack_per_qktile + t.pack_per_qk_dtile * dct_sum) * kct)


def _kv_stream_step(a, t, *, kct, dct_sum, kv_bytes_per_step, reading_cores, tail=False):
    """DRAM K/V stream lane per k-chunk step for one core, or 0 when the regime has no measured law."""
    if t.dram_law == "none":
        return 0.0
    if t.dram_law == "injector":
        rate = max(1.0, a.kv_injector_rate_bpc + a.kv_injector_rate_bpc_per_ktile * kct)
    else:
        rate = _kv_stream_rate(a, max(reading_cores, t.tail_rate_floor_cores) if tail else reading_cores)
    return max(0.0, (_kv_stream_fixed(a, kct, dct_sum) + kv_bytes_per_step / rate) * t.stream_scale)


def _wall_components(r, *, a, t, floor_per_core, Q, K_eff, qct, kct, dct_sum, causal_like, kv_bytes_per_step,
                     legacy=False, tail_cores=0):
    """Named wall terms: per step the longer of the compute lane (floor plus serial front end) and the K/V
    stream, reader_wait its exposed part; booked for the average core, straggler is the wall core's extra."""
    steps = Q * K_eff
    floor_step = floor_per_core / steps
    fpu_step = r.fpu_cycles / steps
    control = t.control_per_kchunk + t.control_per_qtile * qct
    bracket = t.mask_branch_per_kchunk if causal_like else 0.0
    mask_stream = t.mask_per_tile * qct * kct          # provided dense mask tiles, serial with the lanes
    dram = _kv_stream_step(a, t, kct=kct, dct_sum=dct_sum, kv_bytes_per_step=kv_bytes_per_step, reading_cores=r.active_cores)
    if legacy:
        # Exp runs on MATH: no PACK issue lane, but every DEST window pays a round trip.
        pack, sfpu_issue = 0.0, 0.0
        dest_rt = a.legacy_step_cycles + a.dest_roundtrip_cycles * qct * kct / a.dest_window_tiles
    else:
        pack = _pack_lane(t, qct, kct, dct_sum)
        # The exp issue runs on the PACK thread inside the fitted PACK lane (A6); it is exposed only beyond
        # every lane it can hide under, so it cannot double count that lane. Zero on every calibrated row.
        sfpu_issue = max(0.0, qct * kct * a.exp_issue_cycles_per_tile - max(pack, fpu_step, dram))
        dest_rt = 0.0
    fe = (max(0.0, pack - floor_step - bracket - control) + t.fe_flat_per_kchunk
          + t.fe_per_tile_mac * qct * kct * dct_sum)
    compute = floor_step + fe + bracket + mask_stream + control + sfpu_issue + dest_rt
    wait = max(0.0, dram - compute)
    c = {
        "init": float(a.wall_fixed_cycles),
        "compute_floor": float(floor_per_core),
        "fe_issue": steps * fe,
        "reader_wait": steps * wait,
        "mask_bracket": steps * (bracket + mask_stream),
        "sfpu_issue": steps * sfpu_issue,
        "dest_roundtrip": steps * dest_rt,
        "control": steps * control,
    }
    per_core = sum(c.values()) - c["init"]
    # The wall core's steps beyond the light cores' count run in the tail phase, with only the cores that
    # took the remainder still on the DRAM. Same lane max, fewer readers, so the tail step is the compute
    # lane wherever the stream stops binding.
    tail_dram = dram
    if tail_cores and tail_cores < r.active_cores and t.dram_law == "all_cores" and t.tail_phase:
        tail_dram = _kv_stream_step(a, t, kct=kct, dct_sum=dct_sum, kv_bytes_per_step=kv_bytes_per_step,
                                    reading_cores=tail_cores, tail=True)
    c["straggler"] = max(0.0, r.steps_wall_core - steps) * max(compute, tail_dram)
    return c


def _engine_cycles(r, *, a, steps, qct, exp_tiles, tile_macs, qtiles, cpt, exp_approx, causal_like, is_mla,
                   overlap, sink_passes_per_qtile=0.0, combine_passes=0.0, legacy=False):
    """Shared core on resolved work quantities (R1f and T2.1 counter laws): FPU tile MACs plus LLK overhead,
    SFPU exp per tile plus per-step overheads and the recip, union = SFPU minus the hidden share. Fills r."""
    qtile_steps = steps * qct
    r.fpu_matmul_cycles = round(tile_macs * cpt)
    r.fpu_overhead_cycles = round(tile_macs * cpt * a.fpu_overhead_frac
                                  + (a.fpu_overhead_tile_macs_per_qtile_step * cpt + a.fpu_overhead_cycles_per_qtile_step) * qtile_steps)
    r.fpu_cycles = r.fpu_matmul_cycles + r.fpu_overhead_cycles
    if exp_approx:
        exp_cyc = a.exp_tile_cycles
    else:
        exp_cyc = a.exp_tile_cycles_accurate_legacy if legacy else a.exp_tile_cycles_accurate
    r.sfpu_exp_cycles = round(exp_tiles * exp_cyc)
    r.sfpu_reduce_cycles = 0
    r.sfpu_recip_cycles = round(qtiles * a.recip_cycles)
    r.sfpu_overhead_cycles = round(a.sfpu_overhead_per_qtile_step * qtile_steps + a.sfpu_overhead_per_step * steps)
    # Attention sink and flash-decode combine passes: one exp and recip per q tile per pass (no capture).
    r.sfpu_extra_cycles = round((sink_passes_per_qtile * qtiles + combine_passes * qct) * (exp_cyc + a.recip_cycles))
    r.sfpu_cycles = (r.sfpu_exp_cycles + r.sfpu_recip_cycles + r.sfpu_overhead_cycles + r.sfpu_extra_cycles)
    hidden = _overlap_hidden(qct, r.fpu_cycles, r.sfpu_cycles, causal_like, is_mla) if overlap else 0.0
    low = min(r.fpu_cycles, r.sfpu_cycles)
    r.overlap_frac = hidden / low if low > 0 else 0.0
    r.math_active_cycles = round(r.fpu_cycles + r.sfpu_cycles - hidden)
    r.serial_sum_cycles = r.fpu_cycles + r.sfpu_cycles


def _apply_overrides(cfg, num_cores, fidelity, exp_approx_mode, fp32_dest_acc):
    """Per-config program/compute overrides. A value the model cannot use falls back to cfg and
    marks the prediction low confidence."""
    changes, fallback = {}, False
    if num_cores is not None:
        if isinstance(num_cores, int) and not isinstance(num_cores, bool) and num_cores > 0:
            changes["num_cores"] = num_cores
        else:
            fallback = True
    if fidelity is not None:
        if fidelity in cfg.arch.cycles_per_tile_mac:
            changes["fidelity"] = fidelity
        else:
            fallback = True
    if exp_approx_mode is not None:
        changes["exp_approx_mode"] = bool(exp_approx_mode)
    if fp32_dest_acc is not None:
        changes["fp32_dest_acc"] = bool(fp32_dest_acc)
    return (replace(cfg, **changes) if changes else cfg), fallback


def _sparse_wall(r, cfg, a, terms, *, qct, kct, dct_qk, dct_v, cpt, kvbpt, ibpt):
    """Sparse SDPA (sparse_sdpa_program_factory.cpp): query tokens split over the cores, each gathering its
    TOPK latent key rows for every head; per token a serial front end plus the longer of floor and gather."""
    topk = cfg.kv_seq
    tokens_total = cfg.S * cfg.batch
    tokens_mean = tokens_total / cfg.num_cores
    tokens_wall = _ceil_div(tokens_total, cfg.num_cores)
    chunks = max(1, _ceil_div(topk, cfg.k_chunk))
    steps = tokens_mean * chunks
    qct = _ceil_div(cfg.num_heads, TILE_HW)           # every head of the token is one q tile row
    exp_tiles = tokens_mean * qct * _ceil_div(topk, TILE_HW)
    _engine_cycles(r, a=a, steps=steps, qct=qct, exp_tiles=exp_tiles, tile_macs=exp_tiles * (dct_qk + dct_v),
                   qtiles=tokens_mean * qct, cpt=cpt, exp_approx=cfg.exp_approx_mode, causal_like=False,
                   is_mla=True, overlap=False)
    r.q_chunks_per_core, r.k_chunks_per_q, r.k_eff, r.inner_iters = tokens_mean, float(chunks), float(chunks), steps
    r.q_chunks_wall_core, r.active_cores = tokens_wall, min(cfg.num_cores, tokens_total)
    r.steps_wall_core = tokens_wall * chunks
    key_row_bytes = dct_qk * kvbpt / TILE_HW          # one selected key row of the latent K, tile-padded
    gather = topk * key_row_bytes / terms.gather_rate_bpc if terms.gather_rate_bpc > 0 else 0.0
    floor_per_core = max(r.math_active_cycles, max(r.unpack_min_cycles, r.pack_min_cycles))
    floor_token = floor_per_core / tokens_mean
    c = {
        "init": float(a.wall_fixed_cycles),
        "compute_floor": float(floor_per_core),
        "fe_issue": tokens_mean * terms.fe_per_token,
        "reader_wait": tokens_mean * max(0.0, gather - floor_token),
        "mask_bracket": 0.0, "sfpu_issue": 0.0, "dest_roundtrip": 0.0, "control": 0.0,
    }
    per_core = sum(c.values()) - c["init"]
    c["straggler"] = per_core * (tokens_wall - tokens_mean) / tokens_mean
    r.components = c
    r.init_overhead_cycles = round(a.wall_fixed_cycles)
    r.compute_latency_cycles = round(floor_per_core + a.wall_fixed_cycles)
    # DRAM: the query and output once, the gathered rows and the index row per token.
    sq_tiles = _ceil_div(cfg.S, TILE_HW)
    r.dram_in_bytes = round(cfg.num_heads * sq_tiles * dct_qk * ibpt * cfg.batch
                            + tokens_total * topk * (key_row_bytes + 4))
    r.dram_out_bytes = round(cfg.num_heads * sq_tiles * dct_v * ibpt * cfg.batch)   # written in q's dtype
    fam = SPARSE_CALIBRATED
    if (cfg.num_heads != fam["num_heads"] or (cfg.kv_input_dtype or cfg.input_dtype) != fam["kv_dtype"]
            or cfg.k_chunk != fam["k_chunk"]):
        r.flag("sparse_fit_family")


# The fp32_dest_acc_en (non-streaming) prefill kernel is the one path that moved between the calibration
# tree and main: since PR 49948 the QK intermediate and the row sums are fp32, so the mask and reduce passes
# over cb_qk_im carry twice the bytes. Everything else reproduces within 0.25 percent, so only these two
# constants are keyed by revision. Fit on four main-tip walls (S 1024 q64, 2048, 4096, 8192 q128, nh 32,
# nkv 8, HiFi4, packer L1 acc; block FP32M), worst 2.4 percent against 14.6 for the calibration-tree pair.
KERNEL_REV_MAIN_ALIASES = ("main", "main-tip", "2dbd14bf632", "2dbd14b")
LEGACY_STEP_CYCLES_MAIN = 4640.0
DEST_ROUNDTRIP_CYCLES_MAIN = 560.0


def _apply_kernel_rev(cfg: SdpaConfig) -> SdpaConfig:
    """Swap the legacy (fp32 DEST) constants for the main-tip set when the caller names that revision.
    No other constant differs between the two trees, so nothing else is keyed."""
    if not cfg.kernel_rev or cfg.kernel_rev.lower() not in KERNEL_REV_MAIN_ALIASES:
        return cfg
    return replace(cfg, arch=replace(cfg.arch, legacy_step_cycles=LEGACY_STEP_CYCLES_MAIN,
                                   dest_roundtrip_cycles=DEST_ROUNDTRIP_CYCLES_MAIN))


def predict(cfg: SdpaConfig, *, num_cores=None, fidelity=None, exp_approx_mode=None,
            fp32_dest_acc=None, kernel: Optional[KernelDescriptor] = None) -> RooflineResult:
    """Prefill / MLA / cross / windowed / masked / sparse SDPA (compute-bound). cfg.S is the query
    length; decode/paged have their own front-end (predict_decode). Keyword overrides carry a
    production program config (grid, fidelity, exp mode, DEST accumulation) over the cfg fields.
    kernel describes the section graph on a thread_split arch (the standard SDPA step when None)."""
    cfg, override_fallback = _apply_overrides(cfg, num_cores, fidelity, exp_approx_mode, fp32_dest_acc)
    cfg = _apply_kernel_rev(cfg)
    a = cfg.arch
    split = a.thread_split
    r = RooflineResult(label=f"S={cfg.S}", arch_name=a.name, regime="prefill", clock_ghz=a.clock_ghz)

    # head_dim need not be a multiple of 32 (vision uses 72/96/256); it is rounded up to whole tiles.
    v_head_dim = cfg.v_head_dim or cfg.head_dim
    r.is_mla = v_head_dim != cfg.head_dim
    chunked = cfg.is_chunked or cfg.chunk_start_idx > 0
    Skv = cfg.kv_seq or (cfg.S + cfg.chunk_start_idx)   # chunked: the prefix is attended too
    if not (cfg.q_chunk % TILE_HW == 0 and cfg.k_chunk % TILE_HW == 0):
        raise ValueError(f"q_chunk/k_chunk must be multiples of {TILE_HW}")
    if (cfg.input_dtype not in BYTES_PER_TILE or cfg.accum_dtype not in BYTES_PER_TILE
            or (cfg.kv_input_dtype or cfg.input_dtype) not in BYTES_PER_TILE):
        raise ValueError(f"unsupported dtype; supported: {sorted(BYTES_PER_TILE)}")
    # Negative dims pass the modulo checks (-4096 % 128 == 0) and hide inside a plausible total.
    if min(cfg.S, cfg.head_dim, cfg.num_heads, cfg.num_cores, cfg.batch, cfg.q_chunk, cfg.k_chunk) <= 0:
        raise ValueError("S/head_dim/num_heads/num_cores/batch/q_chunk/k_chunk must be positive")
    if min(cfg.kv_seq, cfg.v_head_dim, cfg.num_kv_heads, cfg.chunk_start_idx, cfg.sliding_window, cfg.joint_seq) < 0:
        raise ValueError("kv_seq/v_head_dim/num_kv_heads/chunk_start_idx/sliding_window/joint_seq must be non-negative")

    qct = cfg.q_chunk // TILE_HW
    kct = cfg.k_chunk // TILE_HW
    dct_qk = _ceil_div(cfg.head_dim, TILE_HW)   # Q*K^T contraction (tile-padded)
    dct_v = _ceil_div(v_head_dim, TILE_HW)      # softmax*V output (MLA: != head_dim)
    cpt = a.cpt(cfg.fidelity)
    ibpt = BYTES_PER_TILE[cfg.input_dtype]
    kvbpt = BYTES_PER_TILE[cfg.kv_input_dtype or cfg.input_dtype]
    abpt = BYTES_PER_TILE[cfg.accum_dtype]
    nkv = cfg.num_kv_heads or cfg.num_heads
    kv_frac = nkv / cfg.num_heads

    # The kernel pads the sequence up to a whole chunk, so round chunk counts up (exact-divisible
    # S is bit-identical to ceil); avoids falling to the passthrough mov-cost on non-128 shapes.
    nq = _ceil_div(cfg.S, cfg.q_chunk)
    K = _ceil_div(Skv, cfg.k_chunk)
    if cfg.is_joint and cfg.joint_seq:
        # The joint kernel pads the main and joint streams to whole chunks separately.
        nq = _ceil_div(cfg.S - cfg.joint_seq, cfg.q_chunk) + _ceil_div(cfg.joint_seq, cfg.q_chunk)
        K = _ceil_div(cfg.S - cfg.joint_seq, cfg.k_chunk) + _ceil_div(cfg.joint_seq, cfg.k_chunk)
    q_chunks_total = nq * cfg.num_heads * cfg.batch
    Q = q_chunks_total / cfg.num_cores
    # Visited k chunks and processed k tiles per q chunk (the diagonal chunk is narrowed on every
    # causal-like path; a provided mask visits every chunk when the op is not causal).
    if chunked:
        prefix = cfg.chunk_start_idx / cfg.k_chunk
        visits, tiles = _causal_visits(cfg.S, qct, kct, _ceil_div(cfg.S, cfg.k_chunk))
        visits = [v + prefix for v in visits]
        tiles = [t + prefix * kct for t in tiles]
    elif cfg.sliding_window > 0:
        visits = _windowed_visits(cfg.S, cfg.q_chunk, cfg.k_chunk, Skv, cfg.sliding_window, cfg.is_causal)
        tiles = _windowed_tiles(cfg.S, cfg.q_chunk, cfg.k_chunk, Skv, cfg.sliding_window, cfg.is_causal)
    elif cfg.is_causal and Skv == cfg.S:
        visits, tiles = _causal_visits(cfg.S, qct, kct, K)
    else:
        visits, tiles = [float(K)] * nq, [float(K * kct)] * nq
    K_eff = sum(visits) / len(visits)
    kt_per_chunk = sum(tiles) / len(tiles)
    r.q_chunks_per_core, r.k_chunks_per_q, r.k_eff = Q, float(K), K_eff
    r.inner_iters = Q * K_eff

    causal_like = cfg.is_causal or cfg.has_attn_mask
    # fp32 DEST selects the non-streaming kernel, where the exp is issued from MATH and serialises with
    # the matmuls (T2.4: MATH_COUNTER = FPU + SFPU), so the overlap law applies to the streaming kernel only.
    legacy = cfg.fp32_dest_acc and not r.is_mla
    r.kernel_path = "legacy" if legacy else "streaming"
    # The joint and sparse kernels also issue the exp on MATH (R1a counters: MATH = FPU + SFPU).
    overlap = not (legacy or cfg.is_joint or cfg.is_sparse)

    # Per-core L1 unpacker bytes (on-chip BW floor only). GQA/MQA share KV across a query group.
    unpack_q = Q * qct * dct_qk * ibpt
    unpack_k = Q * K_eff * kct * dct_qk * kvbpt * kv_frac
    unpack_v = Q * K_eff * kct * dct_v * kvbpt * kv_frac
    mask_chunks = (K_eff if (cfg.has_attn_mask and not cfg.is_causal) else (1.0 if cfg.is_causal else 0.0))
    unpack_mask = Q * mask_chunks * qct * kct * abpt if cfg.has_attn_mask or cfg.is_causal else 0.0
    r.unpack_bytes_total = round(unpack_q + unpack_k + unpack_v + unpack_mask)
    r.pack_bytes_total = round(Q * qct * dct_v * abpt)
    r.unpack_min_cycles = round(r.unpack_bytes_total / a.unpacker_bw_bytes_per_cycle)
    r.pack_min_cycles = round(r.pack_bytes_total / a.packer_bw_bytes_per_cycle)

    r.defaulted = cfg.defaulted
    for reason in cfg.fallback_reasons:
        r.flag(reason)
    if cfg.is_joint:
        r.regime = "joint"
        r.flag("regime_joint")
    elif cfg.is_sparse:
        r.regime = "sparse"
        r.flag("regime_sparse")
    elif chunked:
        r.regime = "chunked"
        r.flag("regime_chunked")
    elif cfg.sliding_window > 0 or cfg.has_attn_mask or cfg.attention_sink or cfg.kv_seq:
        r.regime = ("windowed" if cfg.sliding_window > 0 else
                    "masked" if cfg.has_attn_mask else "prefill")
        r.flag("regime_" + ("windowed" if cfg.sliding_window > 0 else "masked" if cfg.has_attn_mask
                            else "sink" if cfg.attention_sink else "cross"))
    # The wall laws were measured on the S 4096 chunk grid (q 64 to 512, k 128 to 512); other chunkings are
    # flagged. The BH envelope flags below do not apply to a split-thread arch, whose path carries its own flag.
    off_chunk = cfg.k_chunk not in CALIBRATED_K_CHUNKS or cfg.q_chunk not in CALIBRATED_Q_CHUNKS
    if off_chunk and not r.is_mla and not split and not cfg.is_sparse:
        r.flag("off_calibration_chunk")
    r.wall_regime = _wall_regime(cfg, r)
    terms = a.wall_terms_for(r.wall_regime)
    r.config_echo = dict(
        S=cfg.S, kv_seq=Skv, batch=cfg.batch, num_heads=cfg.num_heads, num_kv_heads=nkv,
        head_dim=cfg.head_dim, v_head_dim=v_head_dim, q_chunk=cfg.q_chunk, k_chunk=cfg.k_chunk,
        num_cores=cfg.num_cores, fidelity=cfg.fidelity, exp_approx_mode=cfg.exp_approx_mode,
        fp32_dest_acc=cfg.fp32_dest_acc, input_dtype=cfg.input_dtype,
        kv_input_dtype=(cfg.kv_input_dtype or cfg.input_dtype), is_causal=cfg.is_causal,
        has_attn_mask=cfg.has_attn_mask, sliding_window=cfg.sliding_window,
        attention_sink=cfg.attention_sink, chunk_start_idx=cfg.chunk_start_idx, is_chunked=chunked,
        paged=cfg.paged,
        page_block_size=cfg.page_block_size, is_sparse=cfg.is_sparse, is_joint=cfg.is_joint)
    if cfg.is_sparse and not split:
        if terms.low_confidence:
            r.flag(terms.low_confidence)
        if cfg.fidelity != "HiFi4" or (cfg.kv_input_dtype or cfg.input_dtype) != "bfloat16":
            r.flag("sparse_fit_family")
        _sparse_wall(r, cfg, a, terms, qct=qct, kct=kct, dct_qk=dct_qk, dct_v=dct_v, cpt=cpt, kvbpt=kvbpt,
                     ibpt=ibpt)
        r.wall_clock_cycles = round(sum(r.components.values()))
        r.straggler_cycles = round(r.components["straggler"])
        return r

    sink_passes = 1.0 if cfg.attention_sink else 0.0
    if not split:
        exp_tiles = Q * qct * kt_per_chunk
        _engine_cycles(r, a=a, steps=r.inner_iters, qct=qct, exp_tiles=exp_tiles,
                       tile_macs=exp_tiles * (dct_qk + dct_v), qtiles=Q * qct, cpt=cpt,
                       exp_approx=cfg.exp_approx_mode, causal_like=causal_like, is_mla=r.is_mla,
                       overlap=overlap, sink_passes_per_qtile=sink_passes, legacy=legacy)
    chains = None
    if r.wall_regime in ("prefill_noncausal", "cross", "masked"):
        # The factory forwards K/V along one chain per head that spans several cores; with no chain every
        # core streams its own K/V from DRAM, so the row moves to the all-cores lane.
        chains = _noncausal_chains(nq, q_chunks_total, cfg.num_cores)
        if chains == 0 and r.wall_regime != "masked":
            terms = replace(terms, dram_law="all_cores")
            if not split:
                r.flag("chain_geometry_off_calibration")
        elif terms.dram_law == "injector" and chains != CALIBRATED_INJECTORS and not split:
            r.flag("chain_geometry_off_calibration")
    if not r.is_mla and not split:
        grids = CALIBRATED_GRIDS if terms.dram_law == "all_cores" else (CALIBRATED_NUM_CORES,)
        if cfg.num_cores not in grids:
            r.flag("off_calibration_cores")
        # bf16 K/V validates the DRAM stream law only; the PACK lane per tile was measured on bfp8.
        if not cfg.is_joint and (cfg.input_dtype != "bfp8_b" or (cfg.kv_input_dtype or cfg.input_dtype) != "bfp8_b"):
            r.flag("off_calibration_dtype")
        if cfg.fidelity != CALIBRATED_FIDELITY:
            r.flag("off_calibration_fidelity")
        if not cfg.exp_approx_mode:
            r.flag("off_calibration_exp_mode")
    if legacy and not split:
        # Legacy-path terms come from one family (Llama 8B production: 64 cores, HiFi4, accurate exp,
        # bfp8, q=k 64 and 256); everything else on that kernel is extrapolation.
        r.flag("legacy_path_production_family")
    elif cfg.fp32_dest_acc and not split:
        r.flag("fp32_path_uncalibrated")
    if override_fallback:
        r.flag("override_unusable")

    # Whole-op DRAM traffic: Q/output sized by S, K/V by Skv and per KV head (GQA/MQA/MLA correct).
    sq_tiles = _ceil_div(cfg.S, TILE_HW)
    skv_tiles = _ceil_div(Skv, TILE_HW)
    derate = cfg.dram_scatter_derate or 1.0   # paged gather BW penalty on K/V reads
    r.dram_in_bytes = round((cfg.num_heads * sq_tiles * dct_qk * ibpt                   # Q
                             + (nkv * skv_tiles * dct_qk * kvbpt) * derate             # K
                             + (nkv * skv_tiles * dct_v * kvbpt) * derate) * cfg.batch)  # V
    # The output is written in q's dtype whatever the DEST accumulator (sdpa_device_operation.cpp
    # compute_output_specs), so fp32 DEST does not double the write traffic.
    r.dram_out_bytes = round(cfg.num_heads * sq_tiles * dct_v * ibpt * cfg.batch)    # output
    if chains is not None:
        r.config_echo["kv_chains"] = chains

    # Device wall = the most loaded core: its whole q-chunks (factory split, pairs when causal),
    # each carrying its share of the per-core compute floor plus the regime's front-end terms.
    r.q_chunks_wall_core, r.active_cores = _wall_core_chunks(q_chunks_total, nq, cfg.num_cores, cfg.is_causal)
    if cfg.is_joint:
        joint = _joint_split(cfg.num_cores, cfg.batch, cfg.num_heads, nq)
        if joint is not None:
            # The joint factory parallelises each head's q chunks over its own group of cores.
            r.q_chunks_wall_core, Q, r.active_cores = joint
            r.q_chunks_per_core = Q
            r.inner_iters = Q * K_eff
            if not split:
                exp_tiles = Q * qct * kt_per_chunk
                _engine_cycles(r, a=a, steps=r.inner_iters, qct=qct, exp_tiles=exp_tiles,
                               tile_macs=exp_tiles * (dct_qk + dct_v), qtiles=Q * qct, cpt=cpt,
                               exp_approx=cfg.exp_approx_mode, causal_like=False, is_mla=False, overlap=False)
        else:
            r.flag("joint_split_off_factory")
    r.steps_wall_core = r.q_chunks_wall_core * K_eff
    if cfg.sliding_window > 0 and not chunked:
        # Band pairs differ in length, so the wall core is the range with the most visits (W1024 at
        # S 8192: 10 chunks x 9 = 90 steps on the T2.3 wall core, the mean would say 84).
        r.steps_wall_core = max(r.steps_wall_core, _wall_core_steps(visits, q_chunks_total, cfg.num_cores, cfg.is_causal))
    if split:
        # Split-thread arch: the wall is the section graph makespan on the most loaded compute unit; the
        # BH lanes, L1 floor and front-end fits do not apply.
        _thread_split_wall(r, cfg, kernel, Q=Q, K_eff=K_eff, qct=qct, kct=kct, dct_qk=dct_qk, dct_v=dct_v)
        return r
    # Compute FLOOR = math union bounded below by the L1 BW floor, plus the launch cost (the wall's fixed part).
    l1_floor = max(r.unpack_min_cycles, r.pack_min_cycles)
    floor_per_core = max(r.math_active_cycles, l1_floor)
    r.init_overhead_cycles = round(a.wall_fixed_cycles)
    r.compute_latency_cycles = round(floor_per_core + a.wall_fixed_cycles)
    if terms.low_confidence:
        r.flag(terms.low_confidence)
    r.components = _wall_components(r, a=a, t=terms, floor_per_core=floor_per_core, Q=Q, K_eff=K_eff,
                                    qct=qct, kct=kct, dct_sum=dct_qk + dct_v, causal_like=causal_like,
                                    kv_bytes_per_step=kct * (dct_qk + dct_v) * kvbpt, legacy=legacy,
                                    tail_cores=_tail_cores(q_chunks_total, nq, cfg.num_cores, cfg.is_causal))
    if r.components["sfpu_issue"] > 0.0:
        r.flag("sfpu_issue_exposed")
    r.wall_clock_cycles = round(sum(r.components.values()))
    r.straggler_cycles = round(r.components["straggler"])
    r.math_idle_cycles = round(r.inner_iters * a.idle_per_inner_iter)
    return r


def decode_chunk_size(attended: int) -> int:
    """Kernel rule for an unset decode k_chunk: the largest power of two dividing the KV length,
    capped at 512."""
    p = 1
    while attended % (p * 2) == 0 and p < 512:
        p *= 2
    return max(32, p)


def predict_decode(cache_len, num_q_heads, num_kv_heads, head_dim, v_head_dim=0, k_chunk=128,
                   batch=1, cur_pos=None, sliding_window=0, fidelity="HiFi4", input_dtype="bfloat16",
                   accum_dtype="bfloat16", num_cores=110, num_cores_per_head=0, arch=None,
                   kv_input_dtype=None, cur_pos_unknown=False, paged=False, page_block_size=0,
                   is_causal=True, has_attn_mask=False, max_cores_per_head_batch=0, defaulted="",
                   fallback_reasons=(), q_in_dram=True, q_shard_cores=0, mla_v_read=False):
    """Single-token flash/paged decode, memory bound: one query token streams the KV cache up to cur_pos in
    whole k chunks, so the wall is a fixed cost plus the KV bytes at the grid's stream rate. cache_len (or
    cur_pos+1) is the attended length; k_chunk 0 lets the kernel pick it. MLA: q_shard_cores is the sharded
    query's core count (each head slice streams the cache as a user), mla_v_read whether V is passed."""
    a = arch if arch is not None else ArchConfig()
    if a.thread_split:
        raise ValueError("thread_split decode is not modelled: no section graph exists for the decode kernel")
    v_head_dim = v_head_dim or head_dim
    is_mla = v_head_dim != head_dim
    kv_input_dtype = kv_input_dtype or input_dtype
    if (input_dtype not in BYTES_PER_TILE or accum_dtype not in BYTES_PER_TILE
            or kv_input_dtype not in BYTES_PER_TILE):
        raise ValueError(f"unsupported dtype; supported: {sorted(BYTES_PER_TILE)}")
    num_kv_heads = num_kv_heads or num_q_heads   # MHA shorthand, before the core split divides by it
    if min(int(cache_len), num_q_heads, num_kv_heads, head_dim, num_cores, batch) <= 0 or k_chunk < 0:
        raise ValueError("cache_len/heads/head_dim/num_cores/batch must be positive, k_chunk >= 0")
    attended = int(cur_pos + 1) if cur_pos is not None else int(cache_len)
    if attended <= 0:
        raise ValueError("attended KV length must be positive (check cur_pos)")
    # Windowed decode (SWA) attends only the last `window` keys, so the KV stream caps at the window.
    if sliding_window and sliding_window > 0:
        attended = min(attended, int(sliding_window))
    r = RooflineResult(label=f"decode L={attended}", arch_name=a.name, regime="decode",
                       wall_regime=("mla_decode" if is_mla else "decode"),
                       is_memory_bound=True, is_mla=is_mla, defaulted=defaulted, clock_ghz=a.clock_ghz)
    r.flag("decode_family")
    for reason in fallback_reasons:
        r.flag(reason)
    if k_chunk == 0:
        # Paged entry: the kernel sizes the chunk at run time from the position, capped at the DEST size
        # (4 tiles with fp32 DEST); the non-paged op picks it from the cache length.
        if paged:
            k_chunk = decode_dynamic_chunk_tiles(attended - 1, 4 if accum_dtype == "float32" else 8) * TILE_HW
        else:
            k_chunk = decode_chunk_size(int(cache_len))
        r.flag("k_chunk_auto")
    if cur_pos_unknown:
        r.flag("cur_pos_unknown")
    if has_attn_mask or not is_causal:
        r.flag("decode_mask_unmodelled")

    ibpt = BYTES_PER_TILE[input_dtype]
    kvbpt = BYTES_PER_TILE[kv_input_dtype]
    dct_qk = _ceil_div(head_dim, TILE_HW)
    dct_v = _ceil_div(v_head_dim, TILE_HW)
    kc_tiles = max(1, _ceil_div(k_chunk, TILE_HW))
    n_chunks = _ceil_div(attended, k_chunk)
    st_tiles = n_chunks * kc_tiles     # whole chunks, as the reader streams them

    # MLA with a height sharded query: the factory splits the heads into slices of the shard height and
    # streams the cache once per (user, slice); the heads per slice are the q tiles of one step.
    q_slices = _mla_q_head_slices(batch, num_q_heads, q_shard_cores) if is_mla else 1
    users = batch * q_slices
    pnht = max(1, _ceil_div(_ceil_div(num_q_heads, TILE_HW) * TILE_HW // q_slices, TILE_HW))
    groups = max(1, users * num_kv_heads)
    cores_per_head, heads_per_core, active_cores = _decode_core_split(num_cores, users, num_kv_heads,
                                                                       max_cores_per_head_batch)
    if num_cores_per_head:
        cores_per_head, heads_per_core = num_cores_per_head, 1
        active_cores = max(1, min(num_cores, groups * cores_per_head))
    r.active_cores = active_cores

    # KV cache streamed once per user (or head slice), MLA reads V only when a V tensor is passed; Q is one
    # padded head-row tile block per core (reader_decode_all.cpp read_q), DRAM traffic only when not sharded.
    kv_v_tiles = dct_v if (not is_mla or mla_v_read) else 0
    kv_bytes = users * num_kv_heads * st_tiles * (dct_qk + kv_v_tiles) * kvbpt
    q_bytes = active_cores * pnht * dct_qk * ibpt if q_in_dram else 0
    r.dram_in_bytes = kv_bytes + q_bytes
    r.dram_out_bytes = batch * _ceil_div(num_q_heads, TILE_HW) * dct_v * ibpt    # the output core, per user

    # Fixed launch cost plus a cost per KV head group the wall core runs in sequence (one fixed cost per
    # path for MLA), then the KV stream at the calibrated rate.
    if is_mla:
        bw = a.decode_kv_stream_gbps_mla
        if paged:
            fixed, measured_split = _mla_paged_fixed_cycles(a, cores_per_head)
            if not measured_split:
                r.flag("mla_decode_cores_per_group_interpolated")
        else:
            fixed = a.decode_fixed_overhead_cycles_mla
        # Fit forms (R1b): non-paged bf16 with V read on the 110 grid; paged bfp8 with K reused, Q sharded.
        if num_cores != DECODE_NONPAGED_CALIBRATED_GRID or not ((not paged and mla_v_read and kv_input_dtype == "bfloat16")
                                                                or (paged and not mla_v_read and kv_input_dtype == "bfp8_b")):
            r.flag("mla_decode_fit_forms")
    else:
        bw = _decode_kv_stream_gbps(a, num_cores, paged)
        fixed = a.decode_fixed_overhead_cycles + a.decode_fixed_per_head_group_cycles * heads_per_core
        if num_cores not in DECODE_CALIBRATED_GRIDS:
            r.flag("decode_grid_uncalibrated")
        if not paged and num_cores != DECODE_NONPAGED_CALIBRATED_GRID:
            r.flag("decode_nonpaged_grid_inferred")
        elif paged and page_block_size and page_block_size != DECODE_CALIBRATED_PAGE_BLOCK:
            r.flag("decode_page_block_uncalibrated")
    r.config_echo = dict(
        cache_len=int(cache_len), attended=attended, cur_pos=cur_pos, batch=batch,
        num_heads=num_q_heads, num_kv_heads=num_kv_heads, head_dim=head_dim, v_head_dim=v_head_dim,
        k_chunk=k_chunk, num_cores=num_cores, cores_per_head=cores_per_head, heads_per_core=heads_per_core,
        active_cores=active_cores, fidelity=fidelity, input_dtype=input_dtype, kv_input_dtype=kv_input_dtype,
        is_causal=is_causal, has_attn_mask=has_attn_mask, sliding_window=sliding_window, paged=paged,
        page_block_size=page_block_size, q_in_dram=q_in_dram, kv_bytes=kv_bytes, q_bytes=q_bytes,
        kv_stream_gbps=round(bw, 1), q_head_slices=q_slices, mla_v_read=bool(is_mla and mla_v_read))

    # Compute (not binding on the measured points): the padded head-row block of every head group against
    # the core's share of the KV chunks; the flash-decode tree combine adds one pass per round.
    r.q_chunks_per_core = float(heads_per_core)
    chunks_per_core = n_chunks / cores_per_head
    k_tiles_per_core = st_tiles / cores_per_head
    r.k_eff = k_tiles_per_core
    r.inner_iters = heads_per_core * chunks_per_core
    combine_passes = float(math.ceil(math.log2(cores_per_head))) if cores_per_head > 1 else 0.0
    exp_tiles = heads_per_core * pnht * k_tiles_per_core
    _engine_cycles(r, a=a, steps=r.inner_iters, qct=pnht, exp_tiles=exp_tiles,
                   tile_macs=exp_tiles * (dct_qk + dct_v), qtiles=heads_per_core * pnht, cpt=a.cpt(fidelity),
                   exp_approx=False, causal_like=True, is_mla=True, overlap=False, combine_passes=combine_passes)

    mem_cycles = r.dram_in_bytes * a.clock_ghz / bw
    r.init_overhead_cycles = round(fixed)
    r.compute_latency_cycles = round(max(r.math_active_cycles, mem_cycles) + fixed)
    r.wall_clock_cycles = r.compute_latency_cycles
    # kv_stream_wait is the DRAM time beyond the math; the three terms sum to the wall.
    r.components = {"init": float(fixed), "compute_floor": float(r.math_active_cycles),
                    "kv_stream_wait": max(0.0, mem_cycles - r.math_active_cycles)}
    return r


def decode_config_from_shapes(q_shape, k_shape, v_shape=None, attrs=None, num_cores=110, arch=None,
                              page_table_shape=None):
    """Build predict_decode args from a decode op's q + k_cache (+ v_cache) shapes + attrs. q is the ttnn
    decode layout [1, batch, num_q_heads, head_dim] (or [batch, heads, 1, head_dim]); k_cache is [batch,
    num_kv_heads, cache_len, head_dim] or paged [num_blocks, num_kv_heads, block_size, head_dim] plus a page
    table. Absent config fields fall to the kernel defaults (auto k_chunk, whole grid, HiFi2, bf16 DEST)."""
    attrs = attrs or {}
    reasons: List[str] = []
    q = list(map(int, q_shape))
    k = list(map(int, k_shape))
    paged = bool(attrs.get("paged", False))
    # The batch the cache (or the page table) carries tells [1, batch, heads, d] from [batch, heads, 1, d]
    # when one head count is 1: an MQA query [1, 32, 1, 128] over a batch 32 cache is 32 users of one head.
    if page_table_shape is not None and len(page_table_shape) >= 2:
        cache_batch = int(page_table_shape[-2])
    elif not paged and len(k) >= 4:
        cache_batch = k[-4]
    else:
        cache_batch = 0
    if len(q) >= 4 and q[-2] == 1 and not (q[-4] == 1 and q[-3] == cache_batch != 1):
        batch, num_q_heads = q[-4], q[-3]                             # [batch, heads, 1, head_dim]
    else:
        batch, num_q_heads = (q[-3] if len(q) >= 3 else 1), q[-2]    # ttnn decode [1, batch, heads, head_dim]
    head_dim = q[-1]
    num_kv_heads = k[-3]
    page_block_size = int(attrs.get("page_block_size") or 0)
    if paged:
        # Paged cache: rows are blocks. The page table is sized for the longest sequence, so a user
        # can hold at most its width in blocks and never more than the cache has per user.
        page_block_size = page_block_size or k[-2]
        physical = max(1, (k[-4] if len(k) >= 4 else 1) // max(1, batch))
        blocks_per_user = min(int(page_table_shape[-1]), physical) if page_table_shape else physical
        cache_len = blocks_per_user * page_block_size
    else:
        cache_len = k[-2]
    dtype = _ELEM_TO_DTYPE.get(float(attrs.get("element_size", 2)), "bfloat16")
    kv_dtype = _kv_dtype(attrs, dtype, reasons)
    # V head dim: the MLA attr first (the latent V is a slice of K), else the v_cache tensor.
    v_head_dim = int(attrs.get("head_dim_v") or (int(v_shape[-1]) if v_shape else 0))
    cores = int(attrs.get("num_cores") or 0)
    if not cores:
        cores = num_cores
        reasons.append("grid_absent")
    fidelity, fp32 = _resolve_compute_attrs(attrs, reasons)
    cur_pos = attrs.get("cur_pos")
    q_in_l1 = bool(attrs.get("q_in_l1", False))
    # A sharded query's core count drives the MLA head slices; unknown, the launched grid stands in.
    q_shard_cores = int(attrs.get("q_shard_cores") or (cores if q_in_l1 else 0))
    return dict(cache_len=cache_len, num_q_heads=num_q_heads, num_kv_heads=num_kv_heads,
                head_dim=head_dim, v_head_dim=v_head_dim, k_chunk=int(attrs.get("k_chunk_size") or 0),
                batch=batch, cur_pos=(int(cur_pos) if cur_pos is not None else None),
                sliding_window=int(attrs.get("sliding_window") or attrs.get("sliding_window_size") or 0),
                fidelity=fidelity, input_dtype=dtype, accum_dtype=("float32" if fp32 else "bfloat16"),
                num_cores=cores, kv_input_dtype=kv_dtype,
                cur_pos_unknown=bool(cur_pos is None),
                paged=paged, page_block_size=page_block_size,
                is_causal=bool(attrs.get("is_causal", True)),
                has_attn_mask=bool(attrs.get("has_attn_mask", False)),
                max_cores_per_head_batch=int(attrs.get("max_cores_per_head_batch") or 0),
                defaulted=str(attrs.get("sdpa_defaulted") or ""), fallback_reasons=tuple(reasons),
                q_in_dram=not q_in_l1, q_shard_cores=q_shard_cores,
                mla_v_read=bool(attrs.get("mla_v_read", False)),
                arch=arch if arch is not None else ArchConfig())


def decode_perf_stats(q_shape, k_shape, v_shape=None, attrs=None, num_cores=110, arch=None,
                      page_table_shape=None) -> Dict:
    kw = decode_config_from_shapes(q_shape, k_shape, v_shape, attrs, num_cores, arch, page_table_shape)
    ps = predict_decode(**kw).to_polaris_op_perf_stats()
    ps["sdpa_reprice"] = _reprice_blob("decode", kw)
    return ps


ARCH_BH = ArchConfig(name="BH")
