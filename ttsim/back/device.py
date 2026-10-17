#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from loguru import logger

from ttsim.utils.types import get_bpe, get_sim_dtype
from ttsim.utils.lfc import resolve_lfc_path
from ttsim.config.simconfig import MemoryReadLatencyModel
from tools.perf_lookup.lookup_operator_perf import resolve_operator_lookup_core_count

LOG     = logger
INFO    = LOG.info
DEBUG   = LOG.debug
ERROR   = LOG.error
WARNING = LOG.warning

#: Elements in one 32x32 tile. One tile is the default DRAM read "page" for the
#: read-latency model. Distinct from ttsim.front.ttnn.types.TILE_HW, which is this
#: same 1024 but named for height*width.
TILE_ELEMS = 32 * 32

class Component:
    def __init__(self, name: str, atype: str, **kwargs):
        self.name = name
        self.type = atype
        return

    def __str__(self):
        return f"{self.name} type={self.type}"

class MEM(Component):
    def __init__(self, name, **kwargs):
        super().__init__(name, 'MEM', **kwargs)
        self.size_nbytes      = kwargs.get('size')
        self.bw_bytes_per_clk = kwargs.get('bpc')
        return

    def __str__(self):
        return "MEM " + super().__str__() + \
               f" size={self.size_nbytes:,d}B" + \
               f" bw={self.bw_bytes_per_clk:,d}B/Clk"

class NOC(Component):
    def __init__(self, name, **kwargs):
        super().__init__(name, 'NOC', **kwargs)
        self.nrows, self.ncols = kwargs.get('grid') #type: ignore
        return

    def __str__(self):
        return "NOC " + super().__str__() + \
               f" nrows={self.nrows}, ncols={self.ncols}"

class PE(Component):
    def __init__(self, name, **kwargs):
        super().__init__(name, 'COMPUTE', **kwargs)
        return

    def __str__(self):
        return "PE " + super().__str__()

class TTDevice:
    def __init__(self, name, **kwargs):
        self.name     = name
        self.mem_size = kwargs.get('mem_size')
        self.l1_size  = kwargs.get('l1_size')
        self.reg_size = kwargs.get('reg_size')
        self.mem_bw   = kwargs.get('mem_bw')
        self.l1_bw    = kwargs.get('l1_bw')
        self.reg_bw   = kwargs.get('reg_bw')
        self.noc_grid = kwargs.get('noc_grid')

        #components
        self.mem    = MEM(name + '.mem', size=self.mem_size, bpc=self.mem_bw)
        self.l1     = MEM(name + '.l1',  size=self.l1_size,  bpc=self.l1_bw)
        self.reg    = MEM(name + '.reg', size=self.reg_size, bpc=self.reg_bw)
        self.noc    = NOC(name + '.noc', grid=self.noc_grid)
        self.pe     = PE (name + '.pe')
        self.arch   = [self.mem, self.noc, self.l1, self.reg, self.pe]
        self.levels = len(self.arch)
        return

    def __str__(self):
        return "\n".join(f"{a}" for a in self.arch)

    def __getitem__(self, i):
        return self.arch[i]

class Device:
    """
        Hard coded for now

        For resource utilizations: decent estimates can follow a Tiler heuristic (compute)
        and profiling mem-stream benchmark measurements (memory)

        For fusion overlap, and SW/Host overhead guardband, we should get a better
        estimate from LLK team and UMD/Runtime

    """
    DG_COMPUTE_UTIL_CONSTANT        = 0.6
    DG_MEMORY_UTIL_CONSTANT         = 0.8
    G_FUSE_OP_OVERLAP_COST_CONSTANT = 0.10
    G_GUARDBAND                     = 0.25

    def __init__(
        self,
        simcfg_obj,
        *,
        operator_lookup_hybrid_curve: Optional[bool] = None,
        enable_dm_latency: bool = False,
    ):
        compute_ips = [ipg for ipg in simcfg_obj.ipgroups if ipg.iptype == 'compute']
        memory_ips  = [ipg for ipg in simcfg_obj.ipgroups if ipg.iptype == 'memory']
        assert len(compute_ips) == 1, "ERR-1"
        assert len(memory_ips)  == 1, "ERR-2"

        self.simconfig_obj  = simcfg_obj
        # Note: For backward compatibility, architecture package name is stored in 'devname'
        # and device instance name is stored in 'name'. In output, these are mapped to
        # 'archname' and 'devname' respectively.
        self.devname        = simcfg_obj.devname  # Architecture package name (e.g., "Grendel", "Wormhole")
        self.name           = simcfg_obj.name     # Device instance name (e.g., "Q1_A1", "n150")
        self.freq_MHz        = simcfg_obj.frequency('matrix', units='MHz')
        self.memfreq_MHz    = simcfg_obj.mem_frequency(units='MHz')
        self.compute_ip     = compute_ips[0]
        self.memory_ip      = memory_ips[0]
        self.peak_bw_bytes_per_cycle  = simcfg_obj.peak_bandwidth_per_cycle()
        self.eff_bw_bytes_per_cycle   = self.peak_bw_bytes_per_cycle * self.DG_MEMORY_UTIL_CONSTANT

        # DRAM read-latency model (off by default; --enable_dm_latency). When enabled,
        # execute_op takes the max of the bandwidth-limited read cost and the predicted
        # read latency, so small transactions become latency-bound instead of
        # bandwidth-bound. HW constants are sourced exclusively from the arch YAML memory
        # block's `read_latency` section (config/tt_bh.yaml -> MemoryReadLatencyModel);
        # the block is required when the feature is enabled (no Python-side fallback).
        self.enable_dm_latency = bool(enable_dm_latency)
        self.dm_read_cfg: Optional[MemoryReadLatencyModel] = None
        #: DRAM channels the reads interleave across; not part of the read_latency
        #: block, it comes from the memory IP's num_units.
        self.dm_read_channels: int = 1
        #: fclk (NoC) clock used to convert predicted fclk-cycle latency to devclk.
        self.dm_fclk_MHz: float = self.freq_MHz
        if self.enable_dm_latency:
            mem_block = getattr(self.memory_ip, 'ipobj', None)
            rl_params = getattr(mem_block, 'read_latency', None)
            if rl_params is None:
                raise ValueError(
                    "--enable_dm_latency requires a 'read_latency' calibration block in the "
                    f"memory IP of the arch spec, but device '{self.name}' (memory "
                    f"'{getattr(mem_block, 'name', '?')}') defines none. Add a read_latency "
                    "section (tdram_cyc, tdetect_cyc, noc_inbound_bpc, ...) to the memory block."
                )
            # fclk defaults to the matrix compute clock (BH: NoC tied to AI clock).
            self.dm_fclk_MHz = (
                self.freq_MHz if rl_params.fclk_mhz is None else float(rl_params.fclk_mhz)
            )
            self.dm_read_cfg = rl_params
            self.dm_read_channels = max(1, int(getattr(self.memory_ip, 'num_units', 1) or 1))

        # Load tt-perf master operator lookup if specified (see doc/tools/perf_lookup/LOOKUP_TABLE_MASTER.md)
        self.operator_perf_map: Optional[Any] = None
        self._operator_lookup_core_count = resolve_operator_lookup_core_count(
            simcfg_obj, simcfg_obj
        )
        if operator_lookup_hybrid_curve is None:
            _hybrid_curve = bool(getattr(simcfg_obj, "operator_lookup_hybrid_curve", False))
        else:
            _hybrid_curve = bool(operator_lookup_hybrid_curve)
        if hasattr(simcfg_obj, 'operator_lookup_file') and simcfg_obj.operator_lookup_file:
            lookup_file = simcfg_obj.operator_lookup_file
            if lookup_file.startswith('lfc://'):
                try:
                    lookup_file = resolve_lfc_path(lookup_file)
                except (RuntimeError, ValueError) as e:
                    logger.warning(
                        "Failed to resolve LFC path {}: {}. Continuing without operator performance lookup.",
                        simcfg_obj.operator_lookup_file,
                        e,
                    )
                    lookup_file = None

            if lookup_file:
                lookup_file_path = Path(os.getcwd()) / lookup_file
                if lookup_file_path.exists():
                    try:
                        from tools.perf_lookup.lookup_operator_perf import OperatorPerfMap

                        self.operator_perf_map = OperatorPerfMap(
                            lookup_file_path,
                            use_hybrid_curve=_hybrid_curve,
                        )
                        logger.info(
                            "Loaded operator performance master lookup from {} (core_count={}, hybrid_curve={})",
                            lookup_file_path,
                            self._operator_lookup_core_count,
                            _hybrid_curve,
                        )
                    except Exception as e:
                        logger.warning(
                            "Failed to load operator performance lookup from {}: {}",
                            lookup_file_path,
                            e,
                        )
                        self.operator_perf_map = None
                else:
                    logger.warning(f"Operator lookup file specified but not found: {lookup_file_path}")

        return

    def execute_graph(self, wlgraph, wlmapspec, *, disable_fusion: bool = False):
        # 0) ARCH-AWARE COLUMN SPLIT: expand a matched matmul (e.g. lm_head) into arch-
        # appropriate column tiles + concat.  Runs BEFORE get_ordered_nodes so precision /
        # resources / execute_op (LUT lookup) all see the expanded graph.  Keeps the front-end
        # graph device-independent (the arch is only known here at the backend).  See #477.
        self._split_column_ops(wlgraph, getattr(wlmapspec, 'split_spec', None))

        graph_ordered_nodes = wlgraph.get_ordered_nodes()

        # 1) SET PRECISION FOR ALL OPS
        wlgraph.set_precision(wlmapspec.data_type_spec)

        # 2) SET RESOURCES FOR ALL OPS
        wlgraph.set_resources(wlmapspec.rsrc_spec)

        # 2.5) ARCH-AWARE LUT-KEY ANNOTATIONS (must run before execute_op → LUT lookup)
        self._annotate_conv_x_pad_logical(wlgraph)
        self._annotate_halo_y_pad_logical(wlgraph)

        # 3) EXECUTE OPS RESOURCES FOR ALL OPS : FIND COMPUTE/MEM CYCLES
        for opname in graph_ordered_nodes:
            op = wlgraph.get_op(opname)
            self.execute_op(op, wlgraph)

        # 4) GRAPH OPTIMIZATION: REMOVE NODES IF POSSIBLE
        wlgraph.remove_nodes(wlmapspec.removal_spec)

        # 5) GRAPH OPTIMIZATION: FUSE NODES IF POSSIBLE
        # Note: even when a LUT is loaded, fusion still runs here.  The
        # per-op correction happens later in get_exec_stats: any fused op
        # that receives a LUT hit has its fused_in_optimization flag cleared
        # so it keeps its real hardware timing.  This is preferable to a
        # global --disable-fusion because it only reverts fusion for ops
        # that have actual measured data, while ops without LUT entries
        # still benefit from analytical fusion.
        if disable_fusion:
            DEBUG('Skipping graph op fusion (--disable-fusion)')
        else:
            fusion_candidates = wlgraph.fuse_nodes(wlmapspec.fusion_spec)

            #Now all out fusion candidates have been found, and we can apply the
            # op-fusion on the graph
            for fusion_nodes in fusion_candidates:
                """create a new fused node with combined operations"""
                pattern_len   = len(fusion_nodes)
                first_op_name = fusion_nodes[0]
                last_op_name  = fusion_nodes[-1]
                first_op      = wlgraph.get_op(first_op_name)
                last_op       = wlgraph.get_op(last_op_name)

                """
                #update fusion op cycles
                # TODO: add some checks to make sure that intermediate fused ops have
                #   only one input - one output

                #compute cycles = sum of all fused op compute cycles + overhead per operator overlap
                # TODO: should we add overlap cost only if COMPUTE PIPES CHANGE?
                # intermediate mem rd/wr are suppressed by fusion
                # mem rd cycles = first op mem rd cycles
                # mem wr cycles = last op mem rd cycles
                """
                fused_matrix_cycles  = first_op.compute_cycles if first_op.uses_compute_pipe == 'matrix' else 0
                fused_vector_cycles  = first_op.compute_cycles if first_op.uses_compute_pipe == 'vector' else 0
                fused_compute_cycles = first_op.compute_cycles
                fused_mem_rd_cycles  = first_op.mem_rd_cycles
                fused_mem_wr_cycles  = last_op.mem_wr_cycles
                for i in range(1, pattern_len):
                    matched_op_name  = fusion_nodes[i]
                    matched_op       = wlgraph.get_op(matched_op_name)

                    matrix_cycles = matched_op.compute_cycles if matched_op.uses_compute_pipe == 'matrix' else 0
                    vector_cycles = matched_op.compute_cycles if matched_op.uses_compute_pipe == 'vector' else 0

                    fused_matrix_cycles += math.ceil(matrix_cycles * (1.0 + self.G_FUSE_OP_OVERLAP_COST_CONSTANT))
                    fused_vector_cycles += math.ceil(vector_cycles * (1.0 + self.G_FUSE_OP_OVERLAP_COST_CONSTANT))
                    fused_compute_cycles += math.ceil(matched_op.compute_cycles * (1.0 + self.G_FUSE_OP_OVERLAP_COST_CONSTANT))
                    matched_op.fuse_op(first_op_name)

                first_op.fused_op_cycles = {
                        'compute_cycles': fused_compute_cycles,
                        'matrix_cycles' : fused_matrix_cycles,
                        'vector_cycles' : fused_vector_cycles,
                        'mem_rd_cycles' : fused_mem_rd_cycles,
                        'mem_wr_cycles' : fused_mem_wr_cycles,
                        }

        return

    def execute_op(self, op, wlgraph=None):
        if TYPE_CHECKING:
            assert op.perf_stats is not None, f"SimOp {op.name} has no perf_stats set, cannot execute"

        #find compute cycles
        op.compute_cycles = 0
        for instr,instr_count in op.perf_stats['instrs'].items():
            # Enhanced error handling to provide context when instruction lookup fails
            # (e.g., when an operation needs an instruction not in its primary pipe)
            try:
                peak_ipc = self.simconfig_obj.peak_ipc(op.uses_compute_pipe, instr, op.precision)
            except AssertionError as e:
                raise AssertionError(
                    f"Failed to get peak IPC for operation '{op.name}' (optype={op.optype}): "
                    f"instruction='{instr}', pipe='{op.uses_compute_pipe}', precision='{op.precision}'. "
                    f"Original error: {e}"
                ) from e
            real_ipc = peak_ipc * self.DG_COMPUTE_UTIL_CONSTANT
            op.compute_cycles += math.ceil(instr_count / real_ipc)

        # Find memory cycles.
        # NOTE: This calculation is done at the unit of bytes to avoid potential ambiguity of GB (1024 or 1000)
        devfreq_MHz      = self.simconfig_obj.frequency(op.uses_compute_pipe, units='MHz')
        mem_to_dev_ratio = devfreq_MHz / self.memfreq_MHz

        # find memory cycles
        mem_rd_bytes = op.perf_stats['inBytes']
        mem_wr_bytes = op.perf_stats['outBytes']

        # Convert memory bytes to memory cycles in memory clock domain
        mem_rd_cycles_memclk = mem_rd_bytes / self.eff_bw_bytes_per_cycle
        mem_wr_cycles_memclk = mem_wr_bytes / self.eff_bw_bytes_per_cycle

        # Convert memory cycles to device clock domain
        # Store both fractional and ceiled values to avoid accumulated rounding errors
        mem_rd_cycles_devclk_fractional = mem_rd_cycles_memclk * mem_to_dev_ratio
        mem_wr_cycles_devclk_fractional = mem_wr_cycles_memclk * mem_to_dev_ratio

        # Reads cost the worse of the bandwidth-limited time and the fixed read latency:
        # small transactions are latency-bound, large ones stay bandwidth-bound and are
        # unaffected.
        if self.enable_dm_latency and self._dm_read_model_applies(op, wlgraph):
            mem_rd_cycles_devclk_fractional = max(
                mem_rd_cycles_devclk_fractional, self._dm_read_latency_devclk(op, devfreq_MHz)
            )

        # Store fractional values for accurate aggregation
        op.mem_rd_cycles_fractional = mem_rd_cycles_devclk_fractional
        op.mem_wr_cycles_fractional = mem_wr_cycles_devclk_fractional

        # Store ceiled values for per-op scheduling (backward compatibility)
        op.mem_rd_cycles = math.ceil(mem_rd_cycles_devclk_fractional)
        op.mem_wr_cycles = math.ceil(mem_wr_cycles_devclk_fractional)

        return

    #: The one operand placement the read-latency model is calibrated for, as a
    #: MemoryConfig.to_canonical_memory_tag() string.
    DM_READ_SUPPORTED_MEMORY = 'DRAM_INTERLEAVED'

    @classmethod
    def _dm_read_model_applies(cls, op, wlgraph) -> bool:
        """Whether the read-latency model may price ``op``'s reads.

        The calibration constants (tdram, hop count, per-channel bandwidth, n*) describe
        one transaction: a NoC walk to an interleaved DRAM bank. An L1-resident or
        sharded operand is a different read -- shorter path, no bank round-trip, a
        fan-out set by the shard grid rather than by the channel count -- so pricing it
        with these constants over-charges it. Those ops keep the plain bandwidth estimate
        until the model covers them, which is why the gate is on every input: one operand
        placed elsewhere is enough to make the transaction something the model has not
        been calibrated against.

        Operands carrying no memory config are eligible. Absence is not evidence of
        sharding -- the ONNX front end never attaches one, and Polaris's flat memory model
        already assumes that traffic is DRAM -- so refusing them would switch the model
        off for every non-TTNN workload rather than just for the reads it mis-prices.
        """
        tensors = getattr(wlgraph, '_tensors', None)
        if not tensors:
            return True
        for tensor_name in getattr(op, 'inList', None) or []:
            tensor = tensors.get(tensor_name)
            mem_cfg = getattr(tensor, '_memory_config', None) if tensor is not None else None
            if mem_cfg is None:
                continue
            mem_tag = str(mem_cfg)
            if mem_tag != cls.DM_READ_SUPPORTED_MEMORY:
                DEBUG(
                    "DM-READ-LAT skip op={} input={} is {}; model covers {} only",
                    op.name, tensor_name, mem_tag, cls.DM_READ_SUPPORTED_MEMORY,
                )
                return False
        return True

    @staticmethod
    def _dm_read_bytes_per_elem(op, in_bytes: float) -> int:
        """Storage width, in bytes, of the elements ``op`` reads.

        Sizing the read page needs the width the data occupies in DRAM, which is the
        input tensors' dtype -- not ``op.precision``, which comes from the arch mapping
        keyed by optype (WorkloadGraph.set_precision) and describes the precision the
        math runs at. The two disagree on TTNN workloads: llama3_decode reads bfloat16
        tensors under an INT8 compute precision, so pricing the page at 1 byte/element
        both halves N and doubles the page count. ``inBytes`` already reflects storage
        width, so deriving the width from it keeps the page size and the page count
        consistent by construction. ``hlmstats.check_precision`` warns about the same
        divergence.

        Falls back to ``op.precision`` when the op reports no element count.
        """
        in_elems = op.perf_stats.get('inElems', 0) if op.perf_stats else 0
        if in_elems and in_elems > 0:
            # Element widths are whole bytes; the ratio is fractional only when an op
            # mixes dtypes or pads. Rounding then lands on the width carrying most of
            # the traffic, which is the page size that matters -- llama3_decode's
            # embedding MatMul reads 64 KB of int64 indices beside a 256 B fp32 table
            # and resolves to 8, not to the 4 of its smaller input.
            return max(1, round(in_bytes / in_elems))
        return get_bpe(get_sim_dtype(op.precision)) if op.precision else 1

    def _dm_read_latency_devclk(self, op, devfreq_MHz: float) -> float:
        """Predicted DRAM read latency for ``op``, in this op's device-clock domain.

        Maps arch details (DRAM channels, core count) and workload details (read bytes,
        element width) onto the model's transaction shape: reads are tile-paged and fanned
        across the compute grid, so each core issues Q = pages / cores transactions of
        one tile each. Both divisions round up: a partial tile still costs a whole
        transaction, and a partial round across the grid still leaves some core issuing
        one more read than its peers, and the barrier waits for that core.

        Also sets ``op.dm_read_latency_cycles`` (fclk), which is logged at DEBUG.
        """
        assert self.dm_read_cfg is not None, "read-latency model enabled without a config"
        in_bytes = op.perf_stats['inBytes'] if op.perf_stats else 0
        if not in_bytes or in_bytes <= 0:
            op.dm_read_latency_cycles = 0.0
            return 0.0

        bpe = self._dm_read_bytes_per_elem(op, in_bytes)
        page_bytes = TILE_ELEMS * bpe  # one 32x32 tile per DRAM read
        # get_bpe returns -1 for an unmapped dtype; fail loudly rather than let the
        # model's N <= 0 guard silently turn itself off for this op.
        assert page_bytes > 0, (
            f"op {op.name}: precision {op.precision!r} has no known bytes-per-element "
            f"(bpe={bpe}), cannot size a DRAM read page"
        )
        num_cores = max(1, int(getattr(self.compute_ip, 'num_units', 1) or 1))
        total_pages = max(1, math.ceil(in_bytes / page_bytes))
        # Queue depth = reads issued per core before the barrier.
        queue_depth = max(1, math.ceil(total_pages / num_cores))

        tlat_fclk = self.dm_read_cfg.predict_read_latency(
            page_bytes, Q=queue_depth, num_channels=self.dm_read_channels,
        )
        op.dm_read_latency_cycles = tlat_fclk
        DEBUG(
            "DM-READ-LAT op={} N={}B Q={} channels={} inBytes={} -> Tlat={:.0f} fclk cyc",
            op.name, page_bytes, queue_depth, self.dm_read_channels,
            in_bytes, tlat_fclk,
        )
        return tlat_fclk * devfreq_MHz / self.dm_fclk_MHz

    @staticmethod
    def _profiler_pct_to_exec_fraction(v: float) -> float:
        """Convert validated LUT utilization from percentage (0–100) to exec_stats fraction (0–1)."""
        return float(v) / 100.0

    def _is_blackhole(self) -> bool:
        """True when this device is a Blackhole package (arch package name 'Blackhole').

        Mirrors tt-metal is_blackhole(); used to pick arch-specific realizations at the
        backend (the front-end graph is device-independent)."""
        return str(getattr(self, 'devname', '') or '').lower() == 'blackhole'

    def _is_wormhole(self) -> bool:
        """True when this device is a Wormhole package. Mirrors _is_blackhole() (devname is
        the arch package name, e.g. 'Wormhole'/'Blackhole'/'Grendel')."""
        return str(getattr(self, 'devname', '') or '').lower() == 'wormhole'

    @staticmethod
    def _lm_head_core_grid_num_cores(dim: int) -> int:
        """Number of cores in tt-metal's lm_head core grid for a given model dim.

        Replicates tt-metal models/tt_transformers/tt/model_config.py (~lines 699-715):
        start lm_head_num_rows=8, cores_per_row=8; shrink until dim % (TILE*rows*cols)==0.
        For dim=4096 -> 8x8 = 64 cores (=> WH max_columns 668*64 = 42752)."""
        tile, rows, cols = 32, 8, 8
        while dim % (tile * rows * cols) != 0:
            rows -= 1
            if rows == 0:
                cols -= 1
                if cols == 0:
                    raise ValueError(f"lm_head core grid: no rows/cols with dim({dim}) % (32*rows*cols)==0")
                rows = 8
        return rows * cols

    def _lm_head_chunk_widths(self, vocab: int, dim: int) -> list:
        """Arch-specific lm_head vocab column-tile widths (single-chip, prefetcher off).

        Mirrors tt-metal ModelArgs.get_lm_head_max_columns_per_device + LMHead dram_sharded
        split (models/tt_transformers/tt/{model_config,lm_head}.py):
          - BH single-chip: max_columns = size_per_device // 8 -> 8 x 16032 for vocab=128256
          - WH:             max_columns = 668 * num_cores       -> 3 x 42752 for dim=4096
        size_per_device = padded_vocab (num_devices=1). BH max_columns is derived from the
        PADDED size (not the raw vocab) and rounded up to a tile, so a non-tile-aligned vocab
        still yields <=8 tile-aligned chunks; the 128256 llama3 vocab is already tile-aligned,
        so this is 8 x 16032 either way. The vocab column-split is a WH/BH L1-sizing
        optimization with no analogue on other packages, so a non-WH/BH device returns a single
        full-width chunk (no split) rather than borrowing the Wormhole formula."""
        tile = 32
        size_per_device = math.ceil(vocab / tile) * tile
        if self._is_blackhole():
            # tt-metal NUM_LM_HEAD_COLUMNS=8 (num_devices not 4/8 branch); split the PADDED
            # size into 8 tile-aligned columns.
            max_cols = math.ceil(size_per_device / 8 / tile) * tile
        elif self._is_wormhole():
            max_cols = 668 * self._lm_head_core_grid_num_cores(dim)
        else:
            # Non-WH/BH package (e.g. Grendel): no arch-specific column-tiling — skip the split
            # (single full-width chunk) instead of applying the Wormhole formula to an arch it
            # wasn't derived for. _split_one_matmul_columns treats len==1 as "leave the op as-is".
            return [size_per_device]
        n = math.ceil(size_per_device / max_cols)
        widths = [min(size_per_device, max_cols)] * (n - 1)
        widths.append(size_per_device - sum(widths))
        return widths

    def _split_column_ops(self, wlgraph: Any, split_spec: Any) -> None:
        """Expand matched matmuls into arch-appropriate column tiles + Concat (config-driven,
        wl2archmap op_split_spec). The arch is this device's. For each rule, finds MatMuls
        whose output-0 last-dim == match_output_x (optionally input-0 last-dim ==
        match_input_x), computes per-arch tile widths, and rewrites the op into N chunk
        matmuls (each cloning the weight/output tensor at the tile width) whose outputs
        Concat back into the original output tensor. Idempotent: after the split no MatMul
        still matches the rule — the inserted Concat re-emits the full vocab-width output,
        but it is not a MatMul, so a re-run finds no targets. See #477."""
        if split_spec is None or getattr(split_spec, 'is_empty', lambda: True)():
            return
        total_split = 0
        for rule in split_spec.rules:
            targets = []
            for opname in list(wlgraph._ops.keys()):  # snapshot; we mutate _ops in the loop
                op = wlgraph._ops[opname]
                if getattr(op, 'removed_in_optimization', False):
                    continue
                # Exactly 2 inputs (x, w) and 1 output (y): _split_one_matmul_columns rewrites
                # inList[0]/inList[1] -> outList[0], so a matmul/linear with a bias (3rd input)
                # or extra outputs would have those silently dropped — skip it rather than mangle.
                if str(op.optype).upper() != str(rule.op_type).upper() \
                        or len(op.inList) != 2 or len(op.outList) != 1:
                    continue
                out_t = wlgraph._tensors.get(op.outList[0])
                in0_t = wlgraph._tensors.get(op.inList[0])
                if out_t is None or in0_t is None or out_t.shape is None:
                    continue
                if int(out_t.shape[-1]) != int(rule.match_output_x):
                    continue
                if rule.match_input_x is not None and (
                    in0_t.shape is None or int(in0_t.shape[-1]) != int(rule.match_input_x)):
                    continue
                targets.append(opname)
            for opname in targets:
                if self._split_one_matmul_columns(wlgraph, opname, rule):
                    total_split += 1
        if total_split:
            wlgraph.rebuild_graph()
            logger.debug("Column-split {} matmul(s) into arch tiles ({})", total_split,
                         'blackhole' if self._is_blackhole() else 'wormhole')

    def _split_one_matmul_columns(self, wlgraph: Any, opname: str, rule: Any) -> bool:
        from ttsim.front.ttnn.buffer import BufferType, TensorMemoryLayout
        from ttsim.front.ttnn.memory import MemoryConfig
        from ttsim.ops import SimOp
        op = wlgraph._ops[opname]
        x_name, w_name = op.inList[0], op.inList[1]
        y_name = op.outList[0]
        x = wlgraph._tensors[x_name]
        w = wlgraph._tensors[w_name]
        y = wlgraph._tensors[y_name]
        if w.shape is None or len(w.shape) < 2:
            return False
        dim, vocab = int(w.shape[0]), int(w.shape[-1])
        if rule.kind == 'lm_head_vocab':
            widths = self._lm_head_chunk_widths(vocab, dim)
        else:
            raise ValueError(f"unknown column-split kind: {rule.kind}")
        if len(widths) <= 1:
            return False  # single tile — leave the op as-is
        if opname in x.op_in:
            x.op_in.remove(opname)
        # Scrub the split op from the weight's consumer list too — otherwise w.op_in keeps
        # the (soon-deleted) opname, leaving a dangling reference and defeating the
        # `not w.op_in` cleanup below (the original weight tensor would never be freed).
        if opname in w.op_in:
            w.op_in.remove(opname)
        chunk_out_names = []
        for i, wi in enumerate(widths):
            m_i_name = f"{opname}.split{i}"
            w_i = w.clone()
            w_i.rename(f"{w_name}.split{i}")
            w_i.set_shape([dim, int(wi)])
            w_i.op_in = [m_i_name]
            w_i.op_out = []
            wlgraph._tensors[w_i.name] = w_i
            y_i = y.clone()
            y_i.rename(f"{y_name}.split{i}")
            y_i.set_shape(list(y.shape)[:-1] + [int(wi)])
            y_i.op_in = []
            y_i.op_out = [m_i_name]
            wlgraph._tensors[y_i.name] = y_i
            m_i = SimOp({'name': m_i_name, 'optype': op.optype,
                         'inList': [x_name, w_i.name], 'outList': [y_i.name],
                         'attrs': dict(op.attrs)})
            x.op_in.append(m_i_name)
            m_i.get_perf_counts([x, w_i], [y_i])
            m_i.update_tensor_counts([x, w_i], [y_i])
            wlgraph.add_op(m_i)
            # HW: each chunk matmul output is L1 width-sharded and a ShardedToInterleaved
            # stages it to L1 interleaved before the Concat (capture STS: in0
            # TILE/BFLOAT8_B/L1_WIDTH_SHARDED -> out L1_INTERLEAVED). Restore the lm_head
            # output dtype (matmul shape-inf may reset it) so the STS key carries bf8.
            y_i._ttnn_dtype = y._ttnn_dtype
            y_i.dtype = y.dtype
            y_i._memory_config = MemoryConfig(TensorMemoryLayout.WIDTH_SHARDED, BufferType.L1)
            sts_name = f"{m_i_name}.sti"
            y_i.op_in = [sts_name]
            y_sti = y_i.clone()
            y_sti.rename(f"{y_name}.split{i}.sti")
            y_sti._memory_config = MemoryConfig(TensorMemoryLayout.INTERLEAVED, BufferType.L1)
            y_sti.op_in = []
            y_sti.op_out = [sts_name]
            wlgraph._tensors[y_sti.name] = y_sti
            sts = SimOp({'name': sts_name, 'optype': 'ShardedToInterleaved',
                         'inList': [y_i.name], 'outList': [y_sti.name], 'attrs': {}})
            sts.get_perf_counts([y_i], [y_sti])
            sts.update_tensor_counts([y_i], [y_sti])
            wlgraph.add_op(sts)
            chunk_out_names.append(y_sti.name)
        concat_name = f"{opname}.concat"
        concat_inputs = []
        for cn in chunk_out_names:
            ct = wlgraph._tensors[cn]
            ct.op_in.append(concat_name)
            concat_inputs.append(ct)
        y.op_out = [concat_name]
        concat = SimOp({'name': concat_name, 'optype': 'Concat',
                        'inList': list(chunk_out_names), 'outList': [y_name],
                        'attrs': {'axis': -1}})
        concat.get_perf_counts(concat_inputs, [y])
        concat.update_tensor_counts(concat_inputs, [y])
        # tt-metal lm_head concats the vocab chunks to L1 interleaved (ttnn.concat(...,
        # memory_config=L1_MEMORY_CONFIG)). Set the output memory so the downstream logits path
        # (typecast -> TopK) sees L1_INTERLEAVED, matching the capture.
        y._memory_config = MemoryConfig(TensorMemoryLayout.INTERLEAVED, BufferType.L1)
        wlgraph.add_op(concat)
        del wlgraph._ops[opname]
        if w_name in wlgraph._tensors and not w.op_in:
            del wlgraph._tensors[w_name]
        return True

    def _annotate_conv_x_pad_logical(self, wlgraph: Any) -> None:
        """Tag conv2d/conv_transpose2d input tensors (+ upstream passthrough chain) with HW-padded channels.

        For BLOCK_SHARDED convs on devices that declare ``compute_grid_size``, computes
        the channel padding tt-metal's ``determine_parallel_config`` would apply and
        writes it to the conv input tensor's ``x_pad_logical`` attr. The tag is then
        propagated BACKWARD through Move/ITS/STS/Reshard/Halo so each upstream tensor
        carries the same padded x — without this, the upstream Move's LUT key uses
        unpadded channels and analytical-fallbacks (over-estimating duration).

        See doc/TTNN_SHIM_ARCHITECTURE.md §17.
        """
        grid = getattr(self.simconfig_obj, "compute_grid_size", None)
        if grid is None or len(grid) != 2:
            return  # arch doesn't declare a grid — leave tensors alone (no-op fallback)

        from tools.perf_lookup.conv_parallel_config import (
            determine_block_sharded_channel_padding,
        )
        from ttsim.front.ttnn.buffer import TensorMemoryLayout

        # Backward-propagation traverses Move-class passthrough ops only.  Halo is excluded
        # intentionally: Halo's LUT key uses its PRE-halo (input) shape; the channel
        # padding applies to the POST-halo path that feeds the conv.  Walking through Halo
        # would tag the pre-halo tensor with the post-halo padded x and miss the Halo LUT.
        passthrough_ops_backward = frozenset(
            {"Move", "InterleavedToSharded", "ShardedToInterleaved", "Reshard"}
        )

        def _propagate_backward(start_tensor_name: str, padded: int) -> int:
            """Walk backward through single-producer Move-class passthroughs; tag each upstream tensor.

            Stops at Halo (or any non-passthrough op) so the Halo's LUT key — which uses the
            pre-halo input shape — is not perturbed.
            """
            count = 0
            cur_name = start_tensor_name
            while True:
                cur_t = wlgraph._tensors.get(cur_name)
                if cur_t is None:
                    break
                producers = getattr(cur_t, "op_out", None) or []
                if len(producers) != 1:
                    break
                upstream_op = wlgraph.get_op(producers[0])
                if upstream_op is None or getattr(upstream_op, "optype", "") not in passthrough_ops_backward:
                    break
                u_inList = getattr(upstream_op, "inList", None) or []
                if not u_inList:
                    break
                u_in_t = wlgraph._tensors.get(u_inList[0])
                if u_in_t is None:
                    break
                u_in_t.x_pad_logical = padded
                count += 1
                cur_name = u_inList[0]
            return count

        grid_xy = (int(grid[0]), int(grid[1]))
        tagged_convs = 0
        tagged_propagations = 0
        for opname in wlgraph.get_ordered_nodes():
            op = wlgraph.get_op(opname)
            optype = getattr(op, "optype", "")
            if optype not in ("Conv", "ConvTranspose"):
                continue
            inList = getattr(op, "inList", None) or []
            outList = getattr(op, "outList", None) or []
            if not inList or not outList:
                continue
            in_t = wlgraph._tensors.get(inList[0])
            out_t = wlgraph._tensors.get(outList[0])
            if in_t is None or out_t is None:
                continue
            mc = getattr(in_t, "_memory_config", None)
            if mc is None or getattr(mc, "memory_layout", None) != TensorMemoryLayout.BLOCK_SHARDED:
                continue
            # Channel count: prefer hw_shape (NHWC-flat [1, 1, N*H*W, C]); fall back to logical last dim.
            in_hw = getattr(in_t, "hw_shape", None)
            in_channels = int(in_hw[3]) if in_hw is not None and len(in_hw) >= 4 else None
            if in_channels is None:
                in_shape = getattr(in_t, "shape", None)
                if in_shape is None or len(in_shape) == 0:
                    continue
                in_channels = int(in_shape[1])  # NCHW: index 1 is channels
            out_hw = getattr(out_t, "hw_shape", None)
            if out_hw is None or len(out_hw) < 3:
                continue  # need output N*H*W; skip rather than guess
            out_nhw = int(out_hw[2])
            try:
                _, _, padded_channels = determine_block_sharded_channel_padding(
                    input_channels=in_channels,
                    output_nhw=out_nhw,
                    compute_grid_size=grid_xy,
                )
            except ValueError:
                continue
            if padded_channels != in_channels:
                in_t.x_pad_logical = padded_channels
                tagged_convs += 1
                tagged_propagations += _propagate_backward(inList[0], padded_channels)
        if tagged_convs:
            logger.debug(
                "Annotated x_pad_logical on {} conv input tensor(s) (+{} upstream passthrough) "
                "(BLOCK_SHARDED on compute_grid_size={})",
                tagged_convs,
                tagged_propagations,
                grid_xy,
            )

    def _annotate_halo_y_pad_logical(self, wlgraph: Any) -> None:
        """Tag halo output tensors (and their passthrough downstream chain) with arch-specific extended-Y.

        The base ``_HALO_EXT_Y`` table in ttsim/ops/desc/ttsim_layout.py was calibrated against
        WH n150's profiler trace; some halo positions emit a different extended-y on other
        arches because tt-metal's ``determine_parallel_config`` picks a different
        ``num_cores_nhw`` for the downstream conv.  ``_HALO_EXT_Y_OVERRIDES_BY_DEVICE`` declares
        the arch-specific values; when an entry exists for the current device, this pass
        writes ``y_pad_logical`` onto the halo output tensor AND propagates it forward through
        passthrough ops (Move, ITS, STI, Reshard) to the downstream conv input.  See
        doc/TTNN_SHIM_ARCHITECTURE.md §17 (D.2b).
        """
        device_name = getattr(self, "name", None)
        if not device_name:
            return
        from ttsim.ops.desc.ttsim_layout import _HALO_EXT_Y_OVERRIDES_BY_DEVICE

        overrides = _HALO_EXT_Y_OVERRIDES_BY_DEVICE.get(device_name)
        if not overrides:
            return

        passthrough_ops = frozenset({"Move", "InterleavedToSharded", "ShardedToInterleaved", "Reshard"})
        tagged_halos = 0
        tagged_propagations = 0

        def _propagate_forward(start_tensor_name: str, ext_y: int) -> int:
            """Walk forward through passthrough ops; tag every output tensor on the chain."""
            count = 0
            cur_name = start_tensor_name
            while True:
                cur_t = wlgraph._tensors.get(cur_name)
                if cur_t is None:
                    break
                consumers = getattr(cur_t, "op_in", None) or []
                # Stop when there are zero or multiple consumers (branching) — only single-consumer
                # passthrough chains are safe to propagate through.
                if len(consumers) != 1:
                    break
                downstream_op = wlgraph.get_op(consumers[0])
                if downstream_op is None or getattr(downstream_op, "optype", "") not in passthrough_ops:
                    break
                d_outList = getattr(downstream_op, "outList", None) or []
                if len(d_outList) != 1:
                    break
                d_out_t = wlgraph._tensors.get(d_outList[0])
                if d_out_t is None:
                    break
                d_out_t.y_pad_logical = ext_y
                count += 1
                cur_name = d_outList[0]
            return count

        for opname in wlgraph.get_ordered_nodes():
            op = wlgraph.get_op(opname)
            if getattr(op, "optype", "") != "Halo":
                continue
            inList = getattr(op, "inList", None) or []
            outList = getattr(op, "outList", None) or []
            if not inList or not outList:
                continue
            in_t = wlgraph._tensors.get(inList[0])
            out_t = wlgraph._tensors.get(outList[0])
            if in_t is None or out_t is None:
                continue
            in_shape = getattr(in_t, "shape", None)
            if in_shape is None or len(in_shape) < 4:
                continue
            attrs = getattr(op, "attrs", None) or {}
            ks = attrs.get("kernel_size")
            pd = attrs.get("padding")
            if ks is None or pd is None:
                continue
            try:
                kH = int(ks[0]) if hasattr(ks, "__getitem__") else int(ks)
                kW = int(ks[1]) if hasattr(ks, "__getitem__") else int(ks)
                pH = int(pd[0]) if hasattr(pd, "__getitem__") else int(pd)
                pW = int(pd[1]) if hasattr(pd, "__getitem__") else int(pd)
            except (TypeError, ValueError):
                continue
            N, C, H, W = int(in_shape[0]), int(in_shape[1]), int(in_shape[2]), int(in_shape[3])
            nhw = N * H * W
            is_tp = bool(attrs.get("is_transpose", False))
            key = (nhw, C, kH, kW, pH, pW, is_tp)
            override_y = overrides.get(key)
            if override_y is None:
                continue
            out_t.y_pad_logical = int(override_y)
            tagged_halos += 1
            tagged_propagations += _propagate_forward(outList[0], int(override_y))

        if tagged_halos:
            logger.debug(
                "Annotated y_pad_logical on {} halo output tensor(s) (+{} downstream passthrough) for device {!r}",
                tagged_halos,
                tagged_propagations,
                device_name,
            )

    def _try_operator_perf_lookup(
        self,
        op: Any,
        opname: str,
        wlgraph: Any,
        msecs: float
    ) -> tuple[float, bool, Optional[Any]]:
        """
        Resolve timing and optional profiler stats from tt-perf master lookup.

        Raises:
            OperatorPerfLUTValidationError: Invalid or incomplete LUT row (re-raised after log);
                see doc/tools/perf_lookup/LOOKUP_TABLE_MASTER.md.

        Returns:
            (msecs, uses_perf_lookup, master_stats_or_none)
        """
        uses_perf_lookup = False
        master_stats: Optional[Any] = None

        if self.operator_perf_map is not None:
            try:
                master_stats = self.operator_perf_map.lookup(
                    op, wlgraph, self._operator_lookup_core_count
                )
                if master_stats is not None:
                    msecs = master_stats.msecs
                    uses_perf_lookup = True
            except Exception as e:
                from tools.perf_lookup.lookup_operator_perf import OperatorPerfLUTValidationError

                if isinstance(e, OperatorPerfLUTValidationError):
                    logger.error(
                        "Operator perf LUT validation failed for op {!r}; terminating run.\n{}",
                        opname,
                        e,
                    )
                    raise
                logger.warning(f"Error during operator perf lookup for {opname}: {e}", once=True)

        return (msecs, uses_perf_lookup, master_stats)

    def _compute_lut_key_str(self, op: Any, wlgraph: Any, master_stats: Optional[Any]) -> Optional[str]:
        """Stringified literal LUT key for ``op``, emitted regardless of LUT hit/miss.

        On hit, ``master_stats.key_literal`` already carries the tuple — use it.
        On miss, ``master_stats`` is None, but the literal key can still be built
        from the op + tensor state via ``OperatorPerfMap.build_literal_key``.
        Returns None only when no LUT is configured or key construction fails
        (unsupported arity, missing tensor, missing shape, etc.).
        """
        if master_stats is not None and master_stats.key_literal is not None:
            return str(master_stats.key_literal)
        if self.operator_perf_map is None:
            return None
        key_t = self.operator_perf_map.build_literal_key(op, wlgraph)
        return str(key_t) if key_t is not None else None

    def get_exec_stats(self, wlgraph, bs):
        graph_ordered_nodes = wlgraph.get_ordered_nodes()

        for opnum,opname in enumerate(graph_ordered_nodes):
            op           = wlgraph.get_op(opname)

            # Logical completeness: covers both possible fusion states for every op.
            # Use fused_op_cycles as the analytical base whenever fusion ran.
            # If the LUT hits, msecs is overridden below regardless.
            # If the LUT misses, fused_op_cycles is the correct fallback:
            # inner ops are zeroed out (fused_in_optimization=True), so
            # using only standalone cycles for the head would undercount.
            if op.fused_op_cycles is None:
                compute_cycles = op.compute_cycles
                mem_rd_cycles  = op.mem_rd_cycles
                mem_wr_cycles  = op.mem_wr_cycles
                matrix_cycles  = op.compute_cycles if op.uses_compute_pipe == 'matrix' else 0
                vector_cycles  = op.compute_cycles if op.uses_compute_pipe == 'vector' else 0
            else:
                compute_cycles = op.fused_op_cycles['compute_cycles']
                mem_rd_cycles  = op.fused_op_cycles['mem_rd_cycles']
                mem_wr_cycles  = op.fused_op_cycles['mem_wr_cycles']
                matrix_cycles  = op.fused_op_cycles['matrix_cycles']
                vector_cycles  = op.fused_op_cycles['vector_cycles']

            mem_cycles       = mem_rd_cycles + mem_wr_cycles
            ramp_penalty     = self.simconfig_obj.ramp_penalty()
            dev_freq_MHz     = self.simconfig_obj.frequency(op.uses_compute_pipe, units='MHz')
            ideal_cycles     = int(math.ceil(max(compute_cycles, mem_cycles) + ramp_penalty))
            ideal_msecs      = ideal_cycles / dev_freq_MHz / 1e3
            cycles           = int(math.ceil((1 + self.G_GUARDBAND) * ideal_cycles))
            msecs            = cycles / dev_freq_MHz / 1e3

            # Try to use operator performance lookup if available
            msecs, uses_perf_lookup, master_stats = self._try_operator_perf_lookup(
                op, opname, wlgraph, msecs
            )

            # If msecs came from lookup, adjust cycles, ideal_cycles, and ideal_msecs accordingly.
            # LUT timing is NOT derated: the raw hardware-measured msecs is used as-is.
            # ideal_cycles is back-calculated by dividing out G_GUARDBAND so that when the
            # graph aggregate applies tot_msecs = (1 + G_GUARDBAND) * tot_ideal_msecs, the
            # guardband cancels out and the original LUT value is reproduced exactly.
            # Analytical ops (LUT miss) do get the full guardband deration as normal.
            if uses_perf_lookup:
                # Calculate cycles from the lookup msecs value
                cycles = int(math.ceil(msecs * dev_freq_MHz * 1e3))
                # Divide out guardband so the aggregate re-multiplication restores the LUT value.
                ideal_cycles = int(math.ceil(cycles / (1 + self.G_GUARDBAND)))
                # Calculate ideal_msecs from ideal_cycles
                ideal_msecs = ideal_cycles / dev_freq_MHz / 1e3

                # The LUT provides measured per-op hardware timing where the
                # device executes each op individually (no fusion).  When a
                # fused op gets a LUT hit, revert its fused_in_optimization
                # flag so it keeps its real duration instead of being zeroed
                # out below.  The parent op likewise uses standalone cycles
                # (see use_fused above) so there is no double-counting.
                # This is preferred over the global --disable-fusion knob
                # because non-LUT ops still benefit from analytical fusion.
                if op.fused_in_optimization:
                    op.fused_in_optimization = False

            assert ideal_cycles > 0, f"Error: ideal_cycles = {ideal_cycles}!!"
            assert ideal_msecs > 0, f"Error: ideal_msecs = {ideal_msecs}!!"

            memory_traffic = 0.0
            mem_util = 0.0
            if uses_perf_lookup and master_stats is not None:
                matrix_pipe_util = self._profiler_pct_to_exec_fraction(
                    master_stats.matrix_pipe_util
                )
                vector_pipe_util = self._profiler_pct_to_exec_fraction(
                    master_stats.vector_pipe_util
                )
                matrix_cycles = int(math.ceil(matrix_pipe_util * ideal_cycles / self.DG_COMPUTE_UTIL_CONSTANT))
                vector_cycles = int(math.ceil(vector_pipe_util * ideal_cycles / self.DG_COMPUTE_UTIL_CONSTANT))
                mem_rd_cycles = 0
                mem_wr_cycles = 0
                op.mem_rd_cycles_fractional = 0.0
                op.mem_wr_cycles_fractional = 0.0
                mem_rd_util = 0.0
                mem_wr_util = 0.0
                if master_stats.memory_traffic is not None:
                    memory_traffic = float(master_stats.memory_traffic)
                if master_stats.mem_util is not None:
                    mem_util = self._profiler_pct_to_exec_fraction(master_stats.mem_util)
            else:
                matrix_pipe_util = matrix_cycles / ideal_cycles * self.DG_COMPUTE_UTIL_CONSTANT
                vector_pipe_util = vector_cycles / ideal_cycles * self.DG_COMPUTE_UTIL_CONSTANT
                mem_rd_util      = mem_rd_cycles / ideal_cycles * self.DG_MEMORY_UTIL_CONSTANT
                mem_wr_util      = mem_wr_cycles / ideal_cycles * self.DG_MEMORY_UTIL_CONSTANT

            # Flag errors and raise exceptions if utilization > 1.0 (skip checks when using LUT timing:
            # mem_rd_util/mem_wr_util are zeroed above; matrix/vector may come from master without a >1 guard.)
            if not uses_perf_lookup:
                if matrix_pipe_util > 1.0:
                    raise ValueError(
                        f"Matrix pipe utilization exceeds 1.0 for op {opname}: {matrix_pipe_util}"
                    )
                if vector_pipe_util > 1.0:
                    raise ValueError(
                        f"Vector pipe utilization exceeds 1.0 for op {opname}: {vector_pipe_util}"
                    )
                if mem_rd_util > 1.0:
                    raise ValueError(f"Memory read utilization exceeds 1.0 for op {opname}: {mem_rd_util}")
                if mem_wr_util > 1.0:
                    raise ValueError(f"Memory write utilization exceeds 1.0 for op {opname}: {mem_wr_util}")

            if op.removed_in_optimization or op.fused_in_optimization:
                rsrc_bnck        = 'NA'
                ideal_cycles     = 0
                cycles           = 0
                ideal_msecs      = 0.0
                msecs            = 0.0
                matrix_cycles    = 0
                vector_cycles    = 0
                matrix_pipe_util = 0.0
                vector_pipe_util = 0.0
                mem_rd_util      = 0.0
                mem_wr_util      = 0.0
                memory_traffic   = 0.0
                mem_util         = 0.0
            elif compute_cycles >= mem_cycles:
                rsrc_bnck = 'COMP'
            else:
                rsrc_bnck = 'MEM'

            op.exec_stats = {
                    'ramp_penalty'     : ramp_penalty,
                    'rsrc_bnck'        : rsrc_bnck,
                    'ideal_cycles'     : float(ideal_cycles),
                    'ideal_msecs'      : ideal_msecs,
                    'cycles'           : float(cycles),
                    'matrix_cycles'    : matrix_cycles,
                    'vector_cycles'    : vector_cycles,
                    'msecs'            : msecs,
                    'matrix_pipe_util' : matrix_pipe_util,
                    'vector_pipe_util' : vector_pipe_util,
                    'mem_rd_util'      : mem_rd_util,
                    'mem_wr_util'      : mem_wr_util,
                    'memory_traffic'   : memory_traffic,
                    'mem_util'         : mem_util,
                    'uses_perf_lookup' : uses_perf_lookup,
                    # LUT-key trail: ``lut_key`` is the literal key built from the
                    # op + tensor state; ``lut_key_resolved`` is the entry the
                    # lookup chain actually matched after any fallback
                    # substitution (HEIGHT→BLOCK, L1→DRAM, ROW_MAJOR→TILE, arity
                    # dup, …).  Downstream compare_layers ``--by-lut-key`` groups
                    # by the resolved key so TT-Polaris + profiler rows that
                    # semantically share a LUT entry align in the rollup, even
                    # when their literal shapes differ in fallback-tolerable ways.
                    # ``lut_key`` is emitted unconditionally (even on miss) so
                    # the rollup can diagnose mismatches on workloads with low
                    # hit rates; ``lut_key_resolved`` is only meaningful on hit.
                    'lut_key'          : self._compute_lut_key_str(op, wlgraph, master_stats),
                    'lut_key_resolved' : (
                        str(master_stats.key_resolved)
                        if (uses_perf_lookup and master_stats is not None
                            and master_stats.key_resolved is not None) else None
                    ),
                    # Diagnostic: which lookup path produced the hit, "analytical"
                    # when a LUT was loaded but the lookup returned no entry, or None
                    # when no LUT is loaded at all for this device. See
                    # MasterPerfStats.hit_source for the enum of hit-path values.
                    'lut_hit_source'   : (
                        master_stats.hit_source
                        if (uses_perf_lookup and master_stats is not None and master_stats.hit_source)
                        else ('analytical' if self.operator_perf_map is not None else None)
                    ),
                    }

        #compute aggregate stats
        def op_stat_iter(statname, /, repeat=False, use_precision=False, skip_removed=True, skip_fused=True):
            """ utility: iterates over wlgraph for a specific op.stat """
            for opname in graph_ordered_nodes:
                op = wlgraph.get_op(opname)
                if skip_removed and op.removed_in_optimization:
                    continue
                if skip_fused and op.fused_in_optimization:
                    continue
                if hasattr(op, statname):
                    val = getattr(op, statname)
                elif statname in op.perf_stats:
                    val = op.perf_stats.get(statname)
                elif statname in op.exec_stats:
                    val = op.exec_stats.get(statname)
                else:
                    val = None
                    assert False, f"unable to find {statname} in {op}"

                if repeat:
                    val = val * op.repeat_count

                if use_precision:
                    val = val * get_bpe(get_sim_dtype(op.precision))

                yield val

        tot_ideal_cycles = sum(op_stat_iter('ideal_cycles', repeat=True))
        tot_ideal_msecs  = sum(op_stat_iter('ideal_msecs',  repeat=True))
        tot_inParamCount = sum(op_stat_iter('inParamCount', repeat=True))
        tot_inParamBytes = sum(op_stat_iter('inParamCount', repeat=True, use_precision=True))
        tot_inActCount   = sum(op_stat_iter('inActCount'))
        tot_inActBytes   = sum(op_stat_iter('inActCount',  repeat=False, use_precision=True))
        tot_inBytes      = sum(op_stat_iter('inBytes', repeat=True))
        tot_outActCount  = sum(op_stat_iter('outActCount'))
        tot_outActBytes  = sum(op_stat_iter('outActCount', repeat=False, use_precision=True))
        tot_outBytes     = sum(op_stat_iter('outBytes', repeat=True))
        tot_maxActCount  = max(x+y for x,y in zip(op_stat_iter('inActCount'), op_stat_iter('outActCount')))
        tot_maxActBytes  = max(x+y for x,y in zip(op_stat_iter('inActCount', repeat=False, use_precision=True),
                                                  op_stat_iter('outActCount', repeat=False, use_precision=True)))

        tot_comp_bound_cycles = sum(y for x, y in zip(op_stat_iter('rsrc_bnck'),
                                                      op_stat_iter('ideal_cycles', repeat=True)) if x == 'COMP')
        tot_mem_bound_cycles  = sum(y for x, y in zip(op_stat_iter('rsrc_bnck'),
                                                      op_stat_iter('ideal_cycles', repeat=True)) if x == 'MEM')


        assert tot_ideal_cycles > 0, f"Error: tot_ideal_cycles = {tot_ideal_cycles}!!"
        assert tot_ideal_msecs > 0, f"Error: tot_ideal_msecs = {tot_ideal_msecs}!!"

        tot_matrix_cycles     = sum(op_stat_iter('matrix_cycles', repeat=True))
        tot_vector_cycles     = sum(op_stat_iter('vector_cycles', repeat=True))

        # Use fractional cycles for accurate aggregation, then ceil at the aggregate level
        tot_mem_rd_cycles_fractional = sum(op_stat_iter('mem_rd_cycles_fractional', repeat=True))
        tot_mem_wr_cycles_fractional = sum(op_stat_iter('mem_wr_cycles_fractional', repeat=True))
        tot_mem_rd_cycles = math.ceil(tot_mem_rd_cycles_fractional)
        tot_mem_wr_cycles = math.ceil(tot_mem_wr_cycles_fractional)

        tot_matrix_pipe_util  = tot_matrix_cycles / tot_ideal_cycles * self.DG_COMPUTE_UTIL_CONSTANT
        tot_vector_pipe_util  = tot_vector_cycles / tot_ideal_cycles * self.DG_COMPUTE_UTIL_CONSTANT
        tot_mem_rd_util       = tot_mem_rd_cycles / tot_ideal_cycles * self.DG_MEMORY_UTIL_CONSTANT
        tot_mem_wr_util       = tot_mem_wr_cycles / tot_ideal_cycles * self.DG_MEMORY_UTIL_CONSTANT

        # Flag errors and raise exceptions if utilization > 1.0
        if tot_matrix_pipe_util > 1.0:
            raise ValueError(f"Matrix pipe utilization exceeds 1.0: {tot_matrix_pipe_util}")
        if tot_vector_pipe_util > 1.0:
            raise ValueError(f"Vector pipe utilization exceeds 1.0: {tot_vector_pipe_util}")
        if tot_mem_rd_util > 1.0:
            raise ValueError(f"Memory read utilization exceeds 1.0: {tot_mem_rd_util}")
        if tot_mem_wr_util > 1.0:
            raise ValueError(f"Memory write utilization exceeds 1.0: {tot_mem_wr_util}")

        #check if fits device memory...
        tot_mem_size_GB  = (tot_inParamBytes + tot_maxActBytes)/1024/1024/1024
        device_mem_GB    = self.simconfig_obj.mem_size(units='GB')
        fits_device      = tot_mem_size_GB <= device_mem_GB

        #total perf metrics
        # G_GUARDBAND derates analytical ops only.  LUT-hit ops store
        # ideal_msecs = LUT_msecs / (1 + G_GUARDBAND), so the multiplication
        # below restores their original hardware-measured value exactly.
        tot_ideal_throughput = bs * 1000 / tot_ideal_msecs
        tot_msecs            = (1 + self.G_GUARDBAND) * tot_ideal_msecs
        tot_throughput       = bs * 1000 / tot_msecs
        tot_cycles           = math.ceil((1 + self.G_GUARDBAND) * tot_ideal_cycles)

        summary_stats = {
                'inParams'              : tot_inParamCount,
                'inActs'                : tot_inActCount,
                'outActs'               : tot_outActCount,
                'maxActs'               : tot_maxActCount,
                'inParamBytes'          : tot_inParamBytes,
                'inActBytes'            : tot_inActBytes,
                'outActBytes'           : tot_outActBytes,
                'maxActBytes'           : tot_maxActBytes,
                'inBytes'               : tot_inBytes,
                'outBytes'              : tot_outBytes,
                'tot_ideal_cycles'      : tot_ideal_cycles,
                'tot_ideal_msecs'       : tot_ideal_msecs,
                'tot_cycles'            : tot_cycles,
                'tot_msecs'             : tot_msecs,
                'ideal_throughput'      : tot_ideal_throughput,
                'perf_projection'       : tot_throughput,

                'tot_matrix_cycles'     : tot_matrix_cycles,
                'tot_vector_cycles'     : tot_vector_cycles,
                'tot_mem_rd_cycles'     : tot_mem_rd_cycles,
                'tot_mem_wr_cycles'     : tot_mem_wr_cycles,

                'tot_matrix_pipe_util'  : tot_matrix_pipe_util,
                'tot_vector_pipe_util'  : tot_vector_pipe_util,
                'tot_mem_rd_util'       : tot_mem_rd_util,
                'tot_mem_wr_util'       : tot_mem_wr_util,

                'rsrc_comp'             : tot_comp_bound_cycles/tot_ideal_cycles,
                'rsrc_mem'              : tot_mem_bound_cycles/tot_ideal_cycles,

                'mem_size_GB'           : tot_mem_size_GB,
                'device_memsize_GB'     : device_mem_GB,
                'fits_device'           : fits_device,
                'device_peak_bw_GBps'   : self.simconfig_obj.peak_bandwidth(freq_units="GHz"),
                'device_peak_fp8_tflops': self.simconfig_obj.peak_flops('matrix', 'mac', 'fp8', mul_factor=2),
                }
        any_lut = any(
            wlgraph.get_op(n).exec_stats.get('uses_perf_lookup', False)
            for n in graph_ordered_nodes
            if not wlgraph.get_op(n).removed_in_optimization
        )
        if (tot_mem_rd_cycles > 0 or tot_mem_wr_cycles > 0) and not any_lut:
            # Validate that the memory bandwidth accounting is correct.
            # Skipped when any op uses the LUT, since LUT ops zero out
            # analytical mem cycles, making the aggregate inconsistent.
            mem_to_dev_ratio = self.freq_MHz / self.memfreq_MHz
            expected_bytes_per_device_clock = self.eff_bw_bytes_per_cycle / mem_to_dev_ratio

            if tot_mem_rd_cycles > 0:
                actual_bytes_per_device_clock = tot_inBytes / tot_mem_rd_cycles
                # Allow for a single cycle of rounding error from the final ceil operation
                expected_cycles = (tot_inBytes / self.eff_bw_bytes_per_cycle) * mem_to_dev_ratio
                # Check both directions: cycles should be close to expected (within +1 for
                # ceiling). With the read-latency model on, latency-bound ops legitimately
                # cost more than bytes/BW, so only the under-count direction is an error.
                too_low = tot_mem_rd_cycles < expected_cycles - 1
                too_high = tot_mem_rd_cycles > expected_cycles + 1 and not self.enable_dm_latency
                if too_low or too_high:
                    raise ValueError(
                        f"Memory bandwidth validation failed (read):\n"
                        f"  Calculated bytes_per_device_clock: {actual_bytes_per_device_clock:.2f}\n"
                        f"  Expected bytes_per_device_clock:   {expected_bytes_per_device_clock:.2f}\n"
                        f"  Ratio (actual/expected):           {actual_bytes_per_device_clock / expected_bytes_per_device_clock:.2f}\n"
                        f"  Actual cycles: {tot_mem_rd_cycles}, Expected: {expected_cycles:.2f}\n"
                        f"This indicates an inconsistency in memory traffic accounting."
                    )
            if tot_mem_wr_cycles > 0:
                actual_bytes_per_device_clock = tot_outBytes / tot_mem_wr_cycles
                expected_cycles = (tot_outBytes / self.eff_bw_bytes_per_cycle) * mem_to_dev_ratio
                # Check both directions: cycles should be close to expected (within +1 for ceiling)
                if tot_mem_wr_cycles > expected_cycles + 1 or tot_mem_wr_cycles < expected_cycles - 1:
                    raise ValueError(
                        f"Memory bandwidth validation failed (write):\n"
                        f"  Calculated bytes_per_device_clock: {actual_bytes_per_device_clock:.2f}\n"
                        f"  Expected bytes_per_device_clock:   {expected_bytes_per_device_clock:.2f}\n"
                        f"  Ratio (actual/expected):           {actual_bytes_per_device_clock / expected_bytes_per_device_clock:.2f}\n"
                        f"  Actual cycles: {tot_mem_wr_cycles}, Expected: {expected_cycles:.2f}\n"
                        f"This indicates an inconsistency in memory traffic accounting."
                    )

        return summary_stats

    def __str__(self):
        prefix = " "*4

        xstr  = "Device:\n"
        xstr += f"{prefix}devname: {self.devname}\n"
        xstr += f"{prefix}name   : {self.name}\n"

        xstr += f"{prefix}Compute:\n"
        xstr += f"{prefix*2}ipname      : {self.compute_ip.ipname}\n"
        xstr += f"{prefix*2}num_units   : {self.compute_ip.num_units}\n"
        xstr += f"{prefix*2}freq_MHz    : {self.compute_ip.freq_MHz}\n"
        xstr += f"{prefix*2}ramp_penalty: {self.compute_ip.ramp_penalty}\n"
        if self.compute_ip.ipobj.l2_cache:
            xstr += f"{prefix*2}L2:\n"
            xstr += f"{prefix*3}num_banks              : {self.compute_ip.ipobj.l2_cache.num_banks}\n"
            xstr += f"{prefix*3}bytes_per_clk_per_bank : {self.compute_ip.ipobj.l2_cache.bytes_per_clk_per_bank}\n"
        xstr += f"{prefix*2}Pipes:\n"
        for pipe in self.compute_ip.ipobj.pipes:
            xstr += f"{prefix*2}-   name     : {pipe.name}\n"
            xstr += f"{prefix*3}num_units: {pipe.num_units}\n"
            xstr += f"{prefix*3}freq_MHz : {pipe.freq_MHz}\n"
            xstr += f"{prefix*3}instructions:\n"
            for ins in pipe.instructions:
                xstr += f"{prefix*3}-   {{name: {ins.name}, tpt: {ins.tpt} }}\n"

        xstr += f"{prefix}Memory:\n"
        xstr += f"{prefix*2}ipname     : {self.memory_ip.ipname}\n"
        xstr += f"{prefix*2}num_units  : {self.memory_ip.num_units}\n"
        xstr += f"{prefix*2}freq_MHz   : {self.memory_ip.freq_MHz}\n"
        xstr += f"{prefix*2}technology : {self.memory_ip.ipobj.technology}\n"
        xstr += f"{prefix*2}data_bits  : {self.memory_ip.ipobj.data_bits}\n"
        xstr += f"{prefix*2}freq_MHz   : {self.memory_ip.ipobj.freq_MHz}\n"
        xstr += f"{prefix*2}size_GB    : {self.memory_ip.ipobj.size_GB}\n"
        xstr += f"{prefix*2}stacks     : {self.memory_ip.ipobj.stacks}\n"
        xstr += f"{prefix*2}data_rate  : {self.memory_ip.ipobj.data_rate}\n"
        return xstr

if __name__ == '__main__':
    dev_cfg = {
            'mem_size': 2**30, 'mem_bw': 64,
            'l1_size' : 2**20, 'l1_bw' : 256,
            'reg_size': 2**16, 'reg_bw': 2048,
            'noc_grid': (8,8),
            }
    wh = TTDevice('wh', **dev_cfg)
    print(wh)
