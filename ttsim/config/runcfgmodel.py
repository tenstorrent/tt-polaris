#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
from typing import Literal, Optional, Tuple, Union

from pydantic import BaseModel, Field, ValidationInfo, field_validator

from ttsim.utils.types import validate_datatype

type TYPE_SWEEP = Literal['all', 'first-parent']
type TYPE_RUN = Literal['inference', 'training']
type TYPE_LOGLEVEL = Literal['critical', 'error', 'warning', 'info', 'debug']
type TYPE_OUTPUTFORMAT = Literal['json', 'yaml', 'pickle', 'none']
type TYPE_frequency = Tuple[int, int, int]
type TYPE_batchsize = Tuple[int, int, int]

class PolarisRunConfig(BaseModel, extra='forbid'):
    # Title is used only in the run config, it is not a command line attribute of TT-Polaris
    title: str = Field(description='Human readable title for the run')

    # Attributes corresponding to TT-Polaris command line options
    odir: str = Field(
        description='This value is considered to be a Jinja2 template, and expanded as one, with the variables that '
        'polproj automatically defines prior to each run. Hence, the value can use the complete expressive power of '
        'Jinja2 templates to construct the output directory names. '
        '(TBD: provide pointer to the list of variables defined by polproj that can be used in the template)'
    )
    study: str = Field(description='Study name')
    wlspec: str = Field(description='Workload Specification yaml file')
    archspec: str = Field(description='Architecture Specification yaml file')
    wlmapspec: str = Field(description='Workload To Architecture Mapping Specification yaml file')

    datatype: Optional[str] = Field(
        default=None,
        description='Activation Data Type to use for the projection (e.g., bf16, fp32, etc.)'
    )

    @field_validator('datatype')
    @classmethod
    def validate_datatype_field(cls, v: Optional[str], info: ValidationInfo) -> Optional[str]:
        if v is not None:
            validate_datatype(v)
        return v

    filterwlg: Optional[str] = Field(default=None, description='APIs to be considered for the run')

    filterwl: Optional[str] = Field(
        default=None, description='use only workloads specified in filterwl (comma sep list)'
    )

    filterwli: Optional[str] = Field(
        default=None, description='use only workload instances specified in filterwli (comma sep list)'
    )

    filterarch: Optional[str] = Field(
        default=None, description='use only architectures specified in filterarch (comma sep list)'
    )

    filterrun: Optional[TYPE_RUN] = Field(
        default='inference',
        description='Filter either inference or training runs',
    )

    frequency: Optional[TYPE_frequency] = Field(
        default=None, description='frequency (in MHz) range specification (arith-seq)'
    )

    @field_validator('frequency')
    @classmethod
    def validate_frequency(cls, v: TYPE_frequency, info: ValidationInfo) -> TYPE_frequency:
        assert v is not None, 'frequency must be specified'
        # if v is None:
        #     return v
        if any([v[0] <= 0, v[1] <= 0, v[2] <= 0]):
            raise AssertionError(f'frequency values should be positive in {v}')
        if v[0] >= v[1]:
            raise AssertionError(f'frequency first value {v[0]} should be <= last value {v[1]} in {v}')
        return v

    @field_validator('batchsize')
    @classmethod
    def validate_batchsize(cls, v: TYPE_batchsize, info: ValidationInfo) -> TYPE_batchsize:
        assert v is not None, 'batchsize must be specified'
        # if v is None:
        #     return v
        if any([v[0] <= 0, v[1] <= 0, v[2] <= 0]):
            raise AssertionError(f'batchsize values should be positive in {v}')
        if v[2] == 1:
            raise AssertionError(f'batchsize step {v[2]} can not be 1')
        if v[0] >= v[1]:
            raise AssertionError(f'batchsize first value {v[0]} should be < last value {v[1]} in {v}')
        return v

    batchsize: Optional[TYPE_batchsize] = Field(default=None, description='batchsize range specification (geom-seq)')

    knobs: Optional[str] = Field(default='', description='Additional knobs to be used the runs')

    instr_profile: Optional[bool] = Field(default=False, description='Collect instruction profile for workloads')

    dump_ttsim_onnx: Optional[bool] = Field(default=False, description='Dump ONNX graph for TTSIM Workload')

    ignorefails: Optional[bool] = Field(
        default=False, description='If true, run will continue even if some command(s) fails'
    )

    log_level: TYPE_LOGLEVEL = Field(
        default='warning', description='Logging level (error,warning,info,debug) for the run'
    )

    # githash attribute does not correspond to a TT-Polaris command line option
    githash: str | None = Field(default=None, description='Particular git commit')

    # Attributes corresponding to TT-Polaris command line options
    enable_memalloc: Optional[bool] = Field(default=False, description='Enable memory allocation stats')

    enable_cprofile: Optional[bool] = Field(default=False, description='Enable cProfiler stats')

    outputformat: str = Field(default='json', description='Output format (json, yaml, pickle, none) for the run')

    dump_stats_csv: Optional[bool] = Field(default=False, description='Dump stats in csv format')

    operator_lookup_hybrid_curve: Optional[bool] = Field(
        default=False,
        description='Use curve branch for hybrid operator LUT rows (polaris --operator-lookup-hybrid-curve)',
    )

    disable_fusion: Optional[bool] = Field(
        default=False,
        description='Skip op fusion in Device.execute_graph (polaris --disable-fusion)',
    )

    enable_dm_latency: Optional[bool] = Field(
        default=False,
        description=(
            'Enable the DRAM read-latency model, so small transactions are latency-bound '
            'rather than bandwidth-bound (polaris --enable_dm_latency).'
        ),
    )

    dryrun: Optional[bool] = Field(
        default=False, description='If true, run will not execute any commands, but will only print them'
    )

    # Attributes for use by TT-Polaris system only

    saved_copy: Optional[bool] = Field(
        default=False,
        description='Only for internal use. This attribute should be true for yaml copies saved by TT-Polaris system only',
    )




type TYPE_GITHASH = Union[str, None]
