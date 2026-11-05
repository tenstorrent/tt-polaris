#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import os
from typing import Annotated, List, Literal, Optional

from pydantic import BaseModel, Field

type TypeFrequency = float
type TypeTimeMS    = float
type TypeMemsizeGB = float

class PYDWlMapDataSpecValidator(BaseModel, extra='forbid'):
    global_type: Optional[str] = None
    override: Optional[dict[str, str]] = None


class PYDWlMapResourceSpecValidator(BaseModel, extra='forbid'):
    compute: dict[str, list[str]]


class PYDWlMapSpecValidator(BaseModel, extra='forbid'):
    op_data_type_spec: PYDWlMapDataSpecValidator
    op_removal_spec: list[str]
    op_fusion_spec: list[list[str]]
    op_rsrc_spec: PYDWlMapResourceSpecValidator


class PYDPkgMemoryValidator(BaseModel, extra='forbid'):
    ip: str
    num_units: int
    freq_MHz: Optional[float] = None
    ip_overrides: Optional[dict[str, int | float]] = {}


class PYDPkgComputeValidator(BaseModel, extra='forbid'):
    ip: str
    num_units: int
    ramp_penalty: Optional[float] = 0.0
    freq_MHz: Optional[float] = None
    systolic_depth: Optional[int] = None
    ip_overrides: Optional[dict[str, int | float]] = {}


class PYDComputePipeValidator(BaseModel, extra='forbid'):
    num_units: int
    freq_MHz: float
    systolic_depth: Optional[int] = 1
    instructions: dict[str, dict[str, float]]


class PYDL2CacheValidator(BaseModel, extra='forbid'):
    num_banks: int
    bytes_per_clk_per_bank: int

class PYDMemoryBlockValidator(BaseModel, extra='forbid'):
    technology: str
    data_bits: int
    freq_MHz: float
    size_GB: float
    data_rate: Optional[float] = 1
    stacks: Optional[int] = 1


class PYDComputeBlockValidator(BaseModel, extra='forbid'):
    pipes: dict[str, PYDComputePipeValidator]
    l2_cache: Optional[PYDL2CacheValidator] = None

class PYDWorkloadBaseModel(BaseModel):
    name: str


class PYDWorkloadTTSIMModelValidator(PYDWorkloadBaseModel):
    api: Literal['TTSIM']
    name: str
    module: str
    basedir: str
    instances: dict
    params: Optional[dict] = {}



class PYDWorkloadONNXModelValidator(PYDWorkloadBaseModel):
    api: Literal['ONNX']
    name: str
    basedir: str
    instances: dict
    params: Optional[dict] = {}

    def get_instances(self):
        result = {}
        for iname, icfg in self.instances.items():
            xcfg = {}
            if self.params:
                xcfg.update(self.params)
            for xx, xv in icfg.items():
                xcfg[xx] = xv
            result[iname] = {'group': self.name, 'cfg': xcfg}
            result[iname]['path'] = os.path.join(self.basedir, xcfg['path'])
        return result

class PYDWorkloadTTNNModelValidator(PYDWorkloadBaseModel):
    api: Literal['TTNN']
    name: str
    module: str
    basedir: str
    instances: dict
    params: Optional[dict] = {}


AnyWorkload = Annotated[PYDWorkloadTTSIMModelValidator | PYDWorkloadONNXModelValidator | PYDWorkloadTTNNModelValidator  , Field(discriminator='api')]


class PYDWorkloadListValidator(BaseModel):
    workloads: List[AnyWorkload]

type TypeDeviceName = str
type TypePipeName = str
type TypePrecision = str
type TypeOpType = str
type TypeOpClass = str
type TypeResourceName = str
type TypeDomain = str
type TypeInstrName = str

# Option 1 : Direct representation of output CSV
class TTSimHLWlDevRunOpCSVPerfStats(BaseModel, extra='forbid'):
    archname: str = Field(
        description = 'Architecture Package Name (e.g., Grendel, Wormhole)'
    )
    devname: TypeDeviceName = Field(
        description = 'Device Name'
    )
    freq_MHz: float = Field(
        description = 'Frequency in MHz'
    )
    pipe: TypePipeName = Field(
        description = 'Pipe name'
    )
    precision: TypePrecision = Field(
        description = 'Precision'
    )
    wlgroup: str = Field(
        description = 'Workload group (API - TTSIM, ONNX etc)'
    )
    wlname: str = Field(
        description = 'Workload name'
    )
    wlinstance: str = Field(
        description = 'Workload instance (specific workload configuration)'
    )
    batch: int = Field(
        description = 'Batch Size'
    )
    opnum: int = Field(
        description = 'Operator sequence number'
    )
    opname: str = Field(
        description = 'Operator name'
    )
    is_input_node: bool = Field(
        description = 'Is this a network input node? (boolean)' # TODO: P1 Is this right?
    )
    is_output_node: bool = Field(
        description = 'Is this a network output node? (boolean)' # TODO: P1 Is this right?
    )
    optype: TypeOpType = Field(
        description = 'Operator type'
    )
    op_rpt_count: int = Field(
        description = 'Repeat count' # TODO: P1 What is this?
    )
    attrs: dict = Field(
        description = 'Operator attributes'
    )
    inList: list = Field(
        description = 'List of tensors input to this operator'
    )
    outList: list = Field(
        description = 'List of tensors output by this operator'
    )
    input_tensors: str = Field(
        description = 'String representation of input tensors: name[dim1xdim2]:precision;name2[dim1xdim2]:precision'
    )
    output_tensors: str = Field(
        description = 'String representation of output tensors: name[dim1xdim2]:precision;name2[dim1xdim2]:precision'
    )
    weight_tensors: str = Field(
        description = 'String representation of weight tensors: name[dim1xdim2]:precision;name2[dim1xdim2]:precision'
    )
    tensor_attributes: str = Field(
        default = '{}',
        description = 'JSON-serialized dict with per-tensor layout, memory, and dtype attributes'
    )
    domain: TypeDomain = Field(
        description = '???' # TODO: P1 What is this?
    )
    opclass: TypeOpClass = Field(
        description = 'Operator class' # TODO: P1 Distinguish between operator type and class
    )
    removed: bool = Field(
        description = 'Is this operator removed'
    )
    fused: bool = Field(
        description = 'Is this operator fused with another'
    )
    fused_with_op: str = Field(
        description = 'The operator (name) with which this operator is fused'
    )
    inElems: int = Field(
        description = 'Count of input elements'
    )
    outElems: int = Field(
        description = 'Count of output elements'
    )
    inBytes: int = Field(
        description = 'Size in bytes of input elements'
    )
    outBytes: int = Field(
        description = 'Size in bytes of output elements'
    )
    instrs: dict[TypeInstrName, int] = Field(
        description = 'Map of instruction name to its count, within this operator'
    )
    inParamCount: int = Field(
        description = 'Count of (input) parameters'
    )
    inActCount: int = Field(
        description = 'Count if input activation elements'
    )
    outActCount:int = Field(
        description = 'Count if output activation elements'
    )
    instr_count: int = Field(
        description = 'Count of instructions' # TODO: elaborate
    )
    compute_cycles: float = Field(
        description = 'Number of cycles required for compute during the execution this operator'
    )
    mem_rd_cycles: float = Field(
        description = 'Number of cycles required for reading memory during the execution this operator'
    )
    mem_wr_cycles: float = Field(
        description = 'Number of cycles required for writing memory during the execution this operator'
    )
    ramp_penalty: float = Field(
        description = 'Ramp Penalty cycles (roughly speaking, the time required for ramping the operator execution till the entire system is loaded)'
    )
    rsrc_bnck: TypeResourceName = Field(
        description = 'Bottleneck resource for this operator; among all resources, the operator required most cycles in this resource'
    )
    ideal_cycles: float = Field(
        description = 'Number of cycles required to execute the operator without SW/Host overhead guardband'
    )
    ideal_msecs: float = Field(
        description = 'Time (msec) required to execute the operator without SW/Host overhead guardband'
    )
    cycles: float = Field(
        description = 'Number of cycles required to execute the operator'
    )
    matrix_cycles: float = Field(
        description = 'Number of cycles required to execute the operator on MATRIX pipe'
    )
    vector_cycles: float = Field(
        description = 'Number of cycles required to execute the operator on VECTOR pipe'
    )
    msecs: float = Field(
        description = 'Time (msec) required to execute the operator'
    )
    matrix_pipe_util: float = Field(
        description = 'Matrix pipe utilization as fraction 0–1; from tt-perf master (percent/100) when uses_perf_lookup, else analytical'
    )
    vector_pipe_util: float = Field(
        description = 'Vector pipe utilization as fraction 0–1; from tt-perf master (percent/100) when uses_perf_lookup, else analytical'
    )
    mem_rd_util: float = Field(
        description = 'Memory read utilization fraction; 0 when uses_perf_lookup (not derived with LUT timing); else analytical'
    )
    mem_wr_util: float = Field(
        description = 'Memory write utilization fraction; 0 when uses_perf_lookup (not derived with LUT timing); else analytical'
    )
    memory_traffic: float = Field(
        description = 'Bytes of memory traffic from tt-perf master lookup when uses_perf_lookup; else 0'
    )
    mem_util: float = Field(
        description = 'DRAM utilization as fraction 0–1 from tt-perf master (LUT percent/100) when present and uses_perf_lookup; else 0'
    )
    uses_perf_lookup: bool = Field(
        description = 'True when timing and pipe util were obtained from operator performance master lookup (matrix/vector util required in LUT row)'
    )
    lut_key: Optional[str] = Field(
        default = None,
        description = 'Literal LUT key (str(tuple)) built from this operator + its tensor state, before any fallback substitution; emitted on both hit AND miss (diagnostic) whenever a LUT is configured and the literal key can be built. None only when no LUT is configured or key construction fails (unsupported arity, missing tensor/shape, etc.).'
    )
    lut_key_resolved: Optional[str] = Field(
        default = None,
        description = 'Resolved LUT key (str(tuple)) the lookup actually matched after any HEIGHT→BLOCK / L1→DRAM / ROW_MAJOR→TILE / arity-1→arity-2 fallback substitution; equals lut_key when no fallback was needed; None on LUT miss or when no LUT is configured for this device.'
    )
    lut_hit_source: Optional[str] = Field(
        default = None,
        description = 'Which lookup path produced the hit: "direct", one of the fallback names (e.g. "halo_height_to_block", "its_l1_to_dram", "move_arity_dup"), or "analytical" when a LUT is configured but no entry matched (LUT miss). None only when no LUT is configured for this device. See MasterPerfStats.hit_source for full enum.'
    )
    compute_is_lower_bound: bool = Field(
        default = False,
        description = 'True when compute_cycles is an SDPA roofline compute floor (lower bound), not a LUT-measured wall-clock; False for other ops and cleared on a LUT hit.'
    )

# Option 2 - Structured Stats

class TTSimHLWlDevRunOperatorPerfStats(BaseModel, extra='forbid'):
    pipe: TypePipeName
    precision: TypePrecision
    opnum: int
    opname: str
    is_input_node: bool
    is_output_node: bool
    optype: TypeOpType
    op_rpt_count: int
    attrs: dict
    inList: list
    outList: list
    input_tensors: str
    output_tensors: str
    weight_tensors: str
    tensor_attributes: str = '{}'
    domain: TypeDomain
    opclass: TypeOpClass
    removed: bool
    fused: bool
    fused_with_op: str
    inElems: int
    outElems: int
    inBytes: int
    outBytes: int
    instrs: dict[TypeInstrName, int]
    inParamCount: int
    inActCount: int
    outActCount:int
    instr_count: int
    compute_cycles: float
    mem_rd_cycles: float
    mem_wr_cycles: float
    ramp_penalty: float
    rsrc_bnck: TypeResourceName
    ideal_cycles: float
    ideal_msecs: float
    cycles: float
    matrix_cycles: float
    vector_cycles: float
    msecs: float
    matrix_pipe_util: float
    vector_pipe_util: float
    mem_rd_util: float
    mem_wr_util: float
    memory_traffic: float
    mem_util: float
    uses_perf_lookup: bool
    # Literal / resolved LUT key.  See TTSimHLWlDevRunOpCSVPerfStats for full semantics.
    # ``lut_key`` is the tuple built from this op + tensor state (pre-fallback) —
    # emitted on both hit AND miss whenever a LUT is configured (diagnostic).
    # ``lut_key_resolved`` is the tuple the lookup chain actually matched
    # (post-fallback substitution) — emitted only on hit.
    # ``lut_hit_source`` records which path produced the hit, or 'analytical' on
    # miss with a configured LUT. All three are None only when no LUT is configured.
    lut_key: Optional[str] = None
    lut_key_resolved: Optional[str] = None
    lut_hit_source: Optional[str] = None
    # True when compute_cycles is an SDPA roofline compute floor (lower bound), not a measured
    # wall-clock; False for other ops and cleared on a LUT hit.
    compute_is_lower_bound: bool = False

class TTSimHLWlDevRunPerfStats(BaseModel, extra='forbid'):
    """
        This model represents a "run" of the high level simulator on:
           - one specific workload instance,
           - one specific hardware

        The performance metrics of individual operators comprising the workload appear as a list in this model
    """
    archname: str
    devname: TypeDeviceName
    freq_MHz: float
    wlgroup: str
    wlname: str
    wlinstance: str
    batch: int    # TODO: Run or Operator?
    operatorstats: list[TTSimHLWlDevRunOperatorPerfStats]

class TTSimHLRunSummaryRow(BaseModel, extra='forbid'):
    archname              : str
    devname               : str
    freq_Mhz              : TypeFrequency
    wlgroup               : str
    wlname                : str
    wlinstance            : str
    bs                    : int
    inParams              : int
    inActs                : int
    outActs               : int
    maxActs               : int
    inParamBytes          : int
    inActBytes            : int
    outActBytes           : int
    maxActBytes           : int
    inBytes               : int
    outBytes              : int
    tot_cycles            : int
    tot_ideal_cycles      : int
    tot_msecs             : TypeTimeMS
    tot_ideal_msecs       : TypeTimeMS
    ideal_throughput      : float
    perf_projection       : float
    mem_size_GB           : TypeMemsizeGB
    device_memsize_GB     : int
    device_peak_bw_GBps   : float
    device_peak_fp8_tflops: float
    fits_device           : bool
    rsrc_mem              : float
    rsrc_comp             : float
    stat_filename         : str
    tot_matrix_cycles     : int
    tot_vector_cycles     : int
    tot_mem_rd_cycles     : int
    tot_mem_wr_cycles     : int
    tot_matrix_pipe_util  : float
    tot_vector_pipe_util  : float
    tot_mem_rd_util       : float
    tot_mem_wr_util       : float

class TTSimHLRunSummary(BaseModel, extra='forbid'):
    summary: list[TTSimHLRunSummaryRow]
