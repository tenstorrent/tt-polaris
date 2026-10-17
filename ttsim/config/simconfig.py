#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import os
import re
from collections import Counter
from math import isclose
from typing import TYPE_CHECKING, Any, List, Optional, Type, Union

from loguru import logger
from pydantic import BaseModel, model_validator

from ttsim.utils.common import convert_units

LOG   = logger
INFO  = LOG.info
DEBUG = LOG.debug


class SimConfig:
    def __init__(self, cfg):
        #currently only supports dicts and scalars
        for k, v in cfg.items():
            setattr(self, k, SimConfig(v) if isinstance(v, dict) else v)

    def __iter__(self):
        for i in vars(self).keys():
            yield i


class XlsxConfig:
    def __init__(self, cfgname):
        self.values         = dict()
        self.cfgname        = cfgname
        self.defaulted_keys = set()

    @staticmethod
    def normalize_param(param):
        return re.sub('[ ()/\\[\\].-]', '', param.strip()).lower()

    def set_value(self, name, value):
        k = XlsxConfig.normalize_param(name)
        if k in self.values and self.values[k] != value:
            raise RuntimeError(
                f'name "{name}"="{value}", already defined as {self.values[k]}; name normalized as "{k}"')
        self.values[k] = value

    def value(self, name, defvalue=None):
        try:
            return self.values[XlsxConfig.normalize_param(name)]
        except KeyError:
            if defvalue is None:
                raise AttributeError(f'undefined value for "{name}" in config "{self.cfgname}"')
            self.defaulted_keys.add(name)
            return defvalue


class SimCfgBlk: #Generic Sim Configuration Block
    def __init__(self, blkname, **kwargs):
        self.name = blkname

        required_fields = getattr(self, 'required_fields', [])
        optional_fields = getattr(self, 'optional_fields', {})
        missing_fields  = [f for f in required_fields if f not in kwargs]
        if len(missing_fields) > 0:
            raise ValueError(f"Missing required_fields: {missing_fields}")

        #update required_fields
        for f in required_fields:
            setattr(self, f, kwargs[f])

        #update optional fields, may need default values
        for f, dv in optional_fields.items():
            setattr(self, f, kwargs.get(f, dv))

    def kind(self):
        return type(self).__name__

    def set_param(self, pname, pvalue):
        if len(pname) == 0:
            INFO("??? CHECK THIS in {}.{}.set_param()", self.kind(), self.name)
            return False

        parts = pname.split('.')
        if len(parts) == 1:
            attr = parts[0]
            if hasattr(self, attr):
                setattr(self, attr, pvalue)
                return True
            else:
                return False

        cur_attr, next_pname = parts[0], ".".join(parts[1:])

        #search the obj hierachy for name match. child objects may be SimCfgBlk or dict
        cur_child_obj = getattr(self, cur_attr)
        cur_child_pos = 1
        while not isinstance(cur_child_obj, SimCfgBlk) and cur_child_pos < len(pname):
            if isinstance(cur_child_obj, dict):
                if '.' in next_pname:
                    cur_attr       = parts[cur_child_pos]
                    cur_child_obj  = cur_child_obj[cur_attr]
                    cur_child_pos += 1
                    next_pname     = ".".join(parts[cur_child_pos:])
                elif next_pname in cur_child_obj:
                    old_val      = cur_child_obj[next_pname]
                    old_val_type = type(old_val)
                    new_val_type = type(pvalue)
                    if old_val_type == new_val_type:
                        cur_child_obj[next_pname] = pvalue
                        return True
                    else:
                        INFO("WARNING: Type Mismatch: old({}) != new({}), IGNORED!!", old_val, pvalue)
                        return False
                else:
                    return False
            else:
                assert False, "WAAAAAH"

        return cur_child_obj.set_param(next_pname, pvalue)

    def __str__(self):

        def str_helper(obj, indent=0):
            lines = []
            prefix = ' ' * indent
            if isinstance(obj, SimCfgBlk):
                lines.append(f"{prefix}{obj.kind()}({getattr(obj, 'name', 'UNK')}):")
                for key, val in obj.__dict__.items():
                    assert not isinstance(val, list), "List[SimCfgBlk] not supported yet!!"
                    if isinstance(val, dict):
                        lines.append(f"{prefix}  {key}:")
                        for item_name, item_val in val.items():
                            lines.append(f"{prefix}    {item_name}:")
                            lines.append(str_helper(item_val, indent + 6))
                    elif isinstance(val, SimCfgBlk):
                        lines.append(f"{prefix}  {key}:")
                        lines.append(str_helper(val, indent + 4))
                    else:
                        lines.append(f"{prefix}  {key}: {val}")
            elif isinstance(obj, dict):
                for ikey, ival in obj.items():
                    if isinstance(ival, SimCfgBlk) or isinstance(ival, list) or isinstance(ival, dict):
                        assert False, f"DISASTER2 {ikey} {ival}"
                    else:
                        lines.append(f"{prefix} {ikey}: {ival}")
            else:
                DEBUG('... {}', obj)


            return '\n'.join(lines)
        return str_helper(self)


#####################
# Workloads
#####################


class WorkloadCfgBlk(SimCfgBlk):

    def __init__(self, wlname, **kwargs):
        super().__init__(wlname, **kwargs)

    def get_instances(self):
        pass


class WorkloadTTSIM(WorkloadCfgBlk):
    # Type declarations for INSTANCE attributes
    instances: dict
    params: dict
    module: str
    basedir: str
    # Class attributes
    required_fields = ['basedir', 'instances', 'module']
    optional_fields: dict[str, Any] = {'params': {}}

    def __init__(self, wlname, **kwargs):
        super().__init__(wlname, **kwargs)
        self.api = 'TTSIM'

    def get_instances(self):
        result = {}
        for iname, icfg in self.instances.items():
            xcfg = {}
            if self.params:
                xcfg.update(self.params)
            for xx,xv in icfg.items():
                xcfg[xx] = xv
            result[iname] = {'group': self.name, 'module': self.module, 'cfg': xcfg}
            result[iname]['path'] = os.path.join(self.basedir, self.module)
        return result


class WorkloadTTNN(WorkloadCfgBlk):
    # Type declarations for INSTANCE attributes
    instances: dict
    params: dict
    module: str
    basedir: str
    # Class attributes
    required_fields = ['basedir', 'instances', 'module']
    optional_fields: dict[str, Any] = {'params': {}}

    def __init__(self, wlname, **kwargs):
        super().__init__(wlname, **kwargs)
        self.api = 'TTNN'

    def get_instances(self):
        result = {}
        for iname, icfg in self.instances.items():
            xcfg = {}
            if self.params:
                xcfg.update(self.params)
            for xx,xv in icfg.items():
                xcfg[xx] = xv
            result[iname] = {'group': self.name, 'module': self.module, 'cfg': xcfg}
            result[iname]['path'] = os.path.join(self.basedir, self.module)
        return result


class WorkloadONNX(WorkloadCfgBlk):
    # Type declarations for INSTANCE attributes
    instances: dict
    params: dict
    basedir: str
    # Class attributes
    required_fields = ['basedir', 'instances']
    optional_fields: dict[str, Any] = {'params': {}}

    def __init__(self, wlname, **kwargs):
        super().__init__(wlname, **kwargs)
        self.api = 'ONNX'

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


class WorkloadMLIR(WorkloadCfgBlk):
    instances: dict
    params: dict
    basedir: str
    required_fields = ['basedir', 'instances']
    optional_fields: dict[str, Any] = {'params': {}}

    def __init__(self, wlname, **kwargs):
        super().__init__(wlname, **kwargs)
        self.api = 'MLIR'

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


TypeWorkloadClass = Type[WorkloadTTSIM] | Type[WorkloadONNX] | Type[WorkloadTTNN] | Type[WorkloadMLIR]
TypeWorkload = WorkloadTTSIM | WorkloadONNX | WorkloadTTNN | WorkloadMLIR


class AWorkload:
    WLCLS_TBL: dict[str, TypeWorkloadClass] = {'TTSIM': WorkloadTTSIM, 'ONNX': WorkloadONNX, 'TTNN': WorkloadTTNN, 'MLIR': WorkloadMLIR}

    @staticmethod
    def create_workload(apiname: str, **kwargs) -> TypeWorkload:
        wlclass = AWorkload.WLCLS_TBL[apiname.upper()]
        return wlclass(kwargs['name'], **kwargs)


class WorkloadGroup(WorkloadCfgBlk):
    # Type hints for instance attributes
    workloads: dict[str, WorkloadCfgBlk]
    # Class attributes
    WLCLS_TBL = {'TTSIM': WorkloadTTSIM, 'ONNX': WorkloadONNX, 'TTNN': WorkloadTTNN, 'MLIR': WorkloadMLIR}

    def __init__(self, apiname, **kwargs):
        WLCLS = self.WLCLS_TBL[apiname]
        self.workloads = {}
        for wl in kwargs:
            self.workloads[wl] = WLCLS(wl, **kwargs)
        super().__init__(apiname)

    def get_workload_names(self):
        return self.workloads.keys()

    def get_workload_instances(self, wl_name):
        wl_obj = self.workloads[wl_name]
        return wl_obj.get_instances()


#####################
# IPBlocks
#####################


class ComputeInsnModel(BaseModel, extra='forbid'):
    name: str
    tpt: dict[str, float]

class ComputePipeModel(BaseModel, extra='forbid'):
    name: str
    num_units: int
    freq_MHz: float
    systolic_depth: Optional[int] = 1
    instructions: List[ComputeInsnModel]

    def get_insn(self, instr: str) -> ComputeInsnModel:
        matches = [insn for insn in self.instructions if insn.name == instr]
        if len(matches) != 1:
            raise AssertionError(f'non-unique matches for instruction {instr} in {self.name} pipe: {matches=}')
        return matches[0]

    def frequency(self, units="MHz"):
        return convert_units(self.freq_MHz, 'MHz', units)

    def set_frequency(self, newfreq, units="MHZ"):
        self.freq_MHz = convert_units(newfreq, units, 'MHz')
        return


    def handle_missing_precision(self, insn: ComputeInsnModel, prec: str) -> Union[str, None]:
        tblx = {
            'int4' : ['int8', 'int16', 'int32'],
            'int8' : ['int16', 'int32'],
            'int16': ['int32'],
            'int32': [],
            'fp8'  : ['fp16', 'fp32', 'fp64'],
            'bf16' : ['fp32', 'fp64'],
            'fp16' : ['fp32', 'fp64'],
            'fp32' : ['fp64'],
            'tf32' : ['fp32', 'fp64'],
            'fp64' : [],
        }
        if prec not in tblx:
            raise AssertionError(f'unable to handle missing precision: {prec} for instruction= {insn.name} in ComputePipe.{self.name}!!')
        common_precisions = [new_prec for new_prec in tblx[prec] if new_prec in insn.tpt]
        if not common_precisions:
            raise AssertionError(f'no substitute precisions for {prec} for instruction {insn.name} under pipe {self.name}')
        return common_precisions[0]


    def peak_ipc(self, instr: str, prec: str) -> float:
        insn = self.get_insn(instr)
        if insn is None:
            raise AssertionError(f"Throughput for instruction= {instr} for ComputePipe.{self.name} not specified!!")
        try:
            ipc = insn.tpt[prec]
        except KeyError:
            logger.warning(f"WARNING: Missing Support for Precision={prec} with Instruction={instr}@ComputePipe={self.name}", once=True)
            upgraded_prec = self.handle_missing_precision(insn, prec)
            logger.warning(f">>>> upgraded_prec=   {upgraded_prec}", once=True)
            if upgraded_prec is None:
                raise AssertionError(f"Missing Support for Precision={prec} with Instruction={instr}@ComputePipe={self.name}")
            else:
                logger.warning(f"WARNING: Using Precision={upgraded_prec} with Instruction={instr}@ComputePipe={self.name} instead!!", once=True)
                ipc = insn.tpt[upgraded_prec]
        if TYPE_CHECKING:
            assert self.systolic_depth is not None
        return ipc * self.systolic_depth * self.num_units

    def peak_flops(self, instr: str, prec: str, /, units:str='TFLOPS', mul_factor:int=1) -> float:
        mflops = self.peak_ipc(instr, prec) * mul_factor * self.frequency(units="MHZ")
        tflops = convert_units(mflops, 'MFLOPS', 'TFLOPS')
        return tflops


class L2CacheModel(BaseModel, extra='forbid'):
    num_banks: int
    bytes_per_clk_per_bank: int


class ComputeBlockModel(BaseModel, extra='forbid'):
    name: str
    iptype: str
    l2_cache: Optional[L2CacheModel] = None
    pipes: List[ComputePipeModel]

    @model_validator(mode='after')
    def warn_if_pipes_have_different_frequencies(self):
        if len(self.pipes) > 1:
            frequencies = {pipe.name: pipe.freq_MHz for pipe in self.pipes}
            unique_freqs = set(frequencies.values())
            if len(unique_freqs) > 1:
                logger.warning(
                    f"ComputeBlockModel '{self.name}' has pipes with different frequencies: {frequencies}. "
                    f"This may cause issues in determining device frequency."
                )
        return self

    def get_pipe(self, pipename: str) -> ComputePipeModel:
        matches = [pipe for pipe in self.pipes if pipe.name == pipename]
        if len(matches) != 1:
            raise AssertionError(f'non-unique matches for pipe {pipename} in {self.name}')
        return matches[0]

    def set_frequency(self, newfreq, units="MHZ")->None:
        for pipe in self.pipes:
            pipe.set_frequency(newfreq, units)


class MemoryBlockModel(BaseModel, extra='forbid'):
    name: str
    iptype: str
    technology: str
    data_bits: int
    freq_MHz: float
    size_GB: int
    stacks: Optional[int] = 1
    data_rate: Optional[int] = 1
    transfer_rate_GTs: float  # Transfer rate in giga transfers per second (GT/s).
                              # This is a redundant, explicitly specified value used to cross-check
                              # the bandwidth derived from freq_MHz, stacks, and data_rate.
                              # We require the user to specify this field so that any mismatch between
                              # the computed and expected bandwidth is detected early.
                              # Expected value (for DDR): 2 * freq_GHz * stacks * data_rate

    @model_validator(mode='after')
    def validate_transfer_rate(self):
        """Validate that transfer_rate_GTs matches the calculated transfers per second.

        This ensures that the two alternative forms of specifying bandwidth are consistent:
        1. Explicit transfer_rate_GTs (giga transfers per second)
        2. Derived from freq_MHz, stacks, and data_rate

        The formula assumes DDR (Double Data Rate) operation: Tps = 2 * freq_GHz * stacks * data_rate
        where Tps is transfers per second in GT/s.
        """
        freq_ghz = convert_units(self.freq_MHz, 'MHz', 'GHz')
        stacks = self.get_stacks()
        data_rate = self.get_data_rate()
        expected_tps = 2 * freq_ghz * stacks * data_rate
        if not isclose(self.transfer_rate_GTs, expected_tps, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError(
                f"[simconfig.py:validate_transfer_rate] transfer_rate_GTs ({self.transfer_rate_GTs}) "
                f"does not match calculated transfers per second ({expected_tps:.6f} GT/s) "
                f"for memory block '{self.name}'. "
                f"Expected: 2 * freq_GHz ({freq_ghz}) * stacks ({stacks}) * data_rate ({data_rate})"
            )
        return self

    def size(self, /, units="GB"):
        return convert_units(self.size_GB, 'GB', units)

    def frequency(self, units="MHz"):
        return convert_units(self.freq_MHz, 'MHZ', units)

    def get_stacks(self):
        return self.stacks if self.stacks is not None else 1

    def get_data_rate(self):
        return self.data_rate if self.data_rate is not None else 1

    def peak_bandwidth_per_cycle(self):
        # Calculate transfers per memory clock cycle:
        # - 2: TODO: document interpretation of 2
        # - stacks: number of memory stacks
        transfers_per_cycle = 2 * self.get_stacks() * self.get_data_rate()
        bw   = transfers_per_cycle * self.data_bits / 8
        return bw

    def peak_bandwidth(self, freq_units="GHz"):
        freq = self.frequency(freq_units)
        Tps  = 2 * freq * self.get_stacks() * self.get_data_rate() #transfers-per-sec
        bw   = Tps * self.data_bits / 8
        return bw


type BlockModelType = Union[ComputeBlockModel, MemoryBlockModel]  # type: ignore[no-redef]

class IPGroupComponentModel(BaseModel, extra='forbid'):
    ipname: str
    iptype: str
    num_units: int
    freq_MHz: Optional[float] = None
    ramp_penalty: Optional[float] = 0.0
    ipobj: BlockModelType | None = None

class IPGroupComputeModel(IPGroupComponentModel):
    systolic_depth: Optional[int] = None

class IPGroupMemoryModel(IPGroupComponentModel):
    size_GB: Optional[float] = None

# IPBlocksModel = NewType('IPBlocksModel', BaseModel)
type IPGroupModel = Union[IPGroupComputeModel, IPGroupMemoryModel]


# #####################
# # Packages
# #####################


class PackageInstanceModel(BaseModel, extra='forbid'):
    devname: str
    name: str
    ipgroups: List[IPGroupModel]
    operator_lookup_file: Optional[str] = None
    #: Core count for tt-perf master ``single`` / ``curve`` evaluation; if unset, uses compute ``num_units``.
    operator_lookup_core_count: Optional[int] = None
    #: When True, ``entry_type: hybrid`` LUT rows use embedded ``curve`` stats at runtime core count.
    operator_lookup_hybrid_curve: bool = False
    #: Logical compute grid [x, y] = (num_cols, num_rows) — mirrors tt-metal's compute_with_storage_grid_range.
    #: Used by the conv2d parallel-config helper to compute BLOCK_SHARDED channel padding (see
    #: doc/TTNN_SHIM_ARCHITECTURE.md §17). Optional — only required for arches that participate in
    #: the LUT-key channel-padding pass; otherwise TT-Polaris falls back to unpadded channels.
    compute_grid_size: Optional[List[int]] = None

    def get_ipgroup(self, iptype: str) -> IPGroupModel:
        matching = [ipgroup for ipgroup in self.ipgroups if ipgroup.iptype == iptype]
        if len(matching) != 1:
            raise AssertionError(f'non-unique matches for {iptype} in {self.name}')
        return matching[0]

    # Compute group interfaces
    def set_frequency(self, newfreq, units="MHZ"):
        compute_group = self.get_ipgroup(iptype='compute')
        if TYPE_CHECKING:
            assert compute_group.ipobj is not None
            assert isinstance(compute_group.ipobj, ComputeBlockModel)
        compute_group.ipobj.set_frequency(newfreq, units)


    def peak_ipc(self, pipe: str, instr: str, precision: str):
        compute_group = self.get_ipgroup(iptype='compute')
        if TYPE_CHECKING:
            assert compute_group.ipobj is not None
            assert isinstance(compute_group.ipobj, ComputeBlockModel)
        N        = compute_group.num_units
        
        # Try primary pipe first
        pipe_obj = compute_group.ipobj.get_pipe(pipe)
        try:
            ipc = pipe_obj.peak_ipc(instr, precision)
            return N * ipc
        except AssertionError:
            # Instruction not found in primary pipe, search other pipes
            # This enables mixed-pipe operations (e.g., MatMul with fused activation:
            # matmul uses matrix pipe, activation uses vector pipe)
            pass
        
        # Search all other pipes for this instruction
        for other_pipe in compute_group.ipobj.pipes:
            if other_pipe.name == pipe:
                continue  # Already tried primary pipe
            try:
                ipc = other_pipe.peak_ipc(instr, precision)
                logger.debug(
                    f"Instruction '{instr}' not found in primary pipe '{pipe}', "
                    f"using pipe '{other_pipe.name}' instead"
                )
                return N * ipc
            except AssertionError:
                continue  # Try next pipe
        
        # Instruction not found in any pipe
        raise AssertionError(
            f"Instruction '{instr}' not found in any compute pipe "
            f"(searched: {[p.name for p in compute_group.ipobj.pipes]})"
        )

    def peak_flops(self, pipe, instr, precision, mul_factor=1, units='TFLOPS') -> float:
        compute_group = self.get_ipgroup(iptype='compute')
        if TYPE_CHECKING:
            assert compute_group.ipobj is not None
            assert isinstance(compute_group.ipobj, ComputeBlockModel)
        pipe_obj = compute_group.ipobj.get_pipe(pipe)
        flops    = pipe_obj.peak_flops(instr, precision, mul_factor=mul_factor, units=units)
        N        = compute_group.num_units
        return N * flops

    def frequency(self, pipe, units="MHz"):
        compute_group = self.get_ipgroup(iptype='compute')
        if TYPE_CHECKING:
            assert compute_group.ipobj is not None
            assert isinstance(compute_group.ipobj, ComputeBlockModel)
        pipe_obj = compute_group.ipobj.get_pipe(pipe)
        freq     = pipe_obj.frequency(units)
        return freq

    def ramp_penalty(self):
        compute_group = self.get_ipgroup(iptype='compute')
        if TYPE_CHECKING:
            assert compute_group.ipobj is not None
            assert isinstance(compute_group.ipobj, ComputeBlockModel)
        return compute_group.ramp_penalty

    # Memory group interfaces
    def mem_size(self, units='GB'):
        memory_group =  self.get_ipgroup('memory')
        if TYPE_CHECKING:
            assert memory_group.ipobj is not None
            assert isinstance(memory_group.ipobj, MemoryBlockModel)
        N = memory_group.num_units
        S = memory_group.ipobj.size(units)
        return N * S

    def peak_bandwidth_per_cycle(self) -> float:
        memory_group =  self.get_ipgroup('memory')
        if TYPE_CHECKING:
            assert memory_group.ipobj is not None
            assert isinstance(memory_group.ipobj, MemoryBlockModel)
        N = memory_group.num_units
        B = memory_group.ipobj.peak_bandwidth_per_cycle()
        return N * B

    def peak_bandwidth(self, freq_units='GHz') -> float:
        memory_group =  self.get_ipgroup('memory')
        if TYPE_CHECKING:
            assert memory_group.ipobj is not None
            assert isinstance(memory_group.ipobj, MemoryBlockModel)
        N = memory_group.num_units
        B = memory_group.ipobj.peak_bandwidth(freq_units=freq_units)
        return N * B

    def mem_frequency(self, units="MHz"):
        memory_group =  self.get_ipgroup('memory')
        if TYPE_CHECKING:
            assert memory_group.ipobj is not None
            assert isinstance(memory_group.ipobj, MemoryBlockModel)
        pipe_obj = memory_group.ipobj
        freq     = pipe_obj.frequency(units)
        return freq


def find_duplicates(counters: Counter):
    dups = {_key: _value for _key, _value in counters.items() if _value > 1}
    return dups


class IPBlocksModel(BaseModel, extra='forbid'): # type: ignore[no-redef]
    ipblocks: List[BlockModelType]

    @model_validator(mode='after')
    def validate_ipblocks(self):
        ipblocks_frequencies: Counter = Counter()
        for _tmp in self.ipblocks:
            ipblocks_frequencies[_tmp.name] += 1
        duplicates = find_duplicates(ipblocks_frequencies)
        if duplicates:
            raise AssertionError(f'ipblocks have multiple definitions: {duplicates}')
        return self

    def get_ipblock(self, blockname: str) -> BlockModelType:
        for _tmp in self.ipblocks:
            if _tmp.name == blockname:
                return _tmp
        raise AssertionError(f'IPBlock {blockname} not defined')
