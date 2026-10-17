#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""MLIR -> Polaris WorkloadGraph frontend (v3: real Forge TTIR + toy syntax)."""
import re
from dataclasses import dataclass, field
import numpy as np

from ttsim.graph import WorkloadGraph
from ttsim.ops import SimOp, SimTensor

OP_MAP = {
    "matmul": "MatMul", "dot_general": "MatMul", "dot": "MatMul",
    "add": "Add", "mul": "Mul", "multiply": "Mul", "subtract": "Sub", "sub": "Sub",
    "maximum": "Relu", "relu": "Relu",
    "transpose": "Transpose", "reshape": "Reshape",
}
TRANSPARENT = {"broadcast", "broadcast_in_dim", "convert", "typecast"}
CONST_OPS = {"constant", "empty", "iota"}
DT_MAP = {
    "f64": np.float64, "f32": np.float32, "f16": np.float16, "bf16": np.float16,
    "i64": np.int64, "i32": np.int32, "i8": np.int8, "i1": np.bool_,
}
_TRANSPARENT_ALL = TRANSPARENT | {"reshape"}


@dataclass
class FTensor:
    name: str; shape: list; dtype: str
    data: object = None; is_param: bool = False; is_const: bool = False
    op_in: list = field(default_factory=list)
    op_out: list = field(default_factory=list)


@dataclass
class FOp:
    name: str; optype: str; inList: list; outList: list
    attrs: dict = field(default_factory=dict); domain: str = "ai.onnx"


@dataclass
class FGraph:
    name: str
    ops: list = field(default_factory=list)
    tensors: dict = field(default_factory=dict)
    hdr: dict = field(default_factory=dict)


_TENSOR_RE = re.compile(r"tensor<([^>]*)>")


def _parse_tensor_type(t):
    m = _TENSOR_RE.search(t)
    if not m:
        return None, None
    body = m.group(1).split(",")[0]
    parts = body.split("x")
    dims = parts[:-1]
    if not all(d.strip().isdigit() for d in dims):
        return None, parts[-1]
    return [int(d) for d in dims], parts[-1]


def parse_mlir(path, wlname=None):
    src = open(path).read()
    fg = FGraph(name=wlname or "mlir_wl", hdr={"name": wlname or "mlir_wl", "framework_type": "MLIR"})

    sig = re.search(r"func\.func\s+(?:public\s+|private\s+)?@(\w+)\s*\((.*?)\)\s*->", src, re.S)
    assert sig, "no func.func signature found"
    fg.name = sig.group(1); fg.hdr["name"] = sig.group(1)
    for arg in [a.strip() for a in sig.group(2).split(",") if a.strip()]:
        am = re.match(r"%([\w.]+)\s*:\s*(tensor<[^>]*>)", arg)
        if not am:
            continue
        shp, dt = _parse_tensor_type(am.group(2))
        fg.tensors[am.group(1)] = FTensor(name=am.group(1), shape=shp, dtype=dt)

    op_re = re.compile(
        r'%([\w.]+)\s*=\s*"?(?:(\w+)\.)?(\w+)"?\s*\(([^)]*)\)\s*'
        r'(<\{.*?\}>)?\s*(\{.*?\})?\s*:\s*\(([^)]*)\)\s*->\s*(tensor<[^>]*>)')

    alias = {}
    const_results = set()
    raw = []
    for m in op_re.finditer(src):
        res, _dialect, opname, operands, aattr, battr, intypes, restype = m.groups()
        inames = [o.strip().lstrip("%") for o in operands.split(",") if o.strip()]
        if opname in CONST_OPS:
            const_results.add(res)
            continue
        if opname in _TRANSPARENT_ALL:
            alias[res] = inames[0] if inames else None
            continue
        raw.append((res, opname, inames, (aattr or "") + (battr or ""), restype))

    def resolve(t):
        seen = set()
        while t in alias and t not in seen:
            seen.add(t)
            t = alias[t]
        return t

    for i, (res, opname, inames, attrblk, restype) in enumerate(raw):
        assert opname in OP_MAP, f"unsupported MLIR op '{opname}' (supported: {sorted(set(OP_MAP))})"
        resolved = []
        for nm in inames:
            r = resolve(nm)
            if r in const_results:
                continue
            resolved.append(r)
        optype = OP_MAP[opname]
        if opname == "maximum" and len(resolved) == 1:
            optype = "Relu"
        attrs = {}
        pm = re.search(r"perm\s*=\s*\[([\d,\s]*)\]", attrblk)
        if pm:
            attrs["perm"] = [int(x) for x in pm.group(1).split(",") if x.strip()]
        rshp, rdt = _parse_tensor_type(restype)
        fg.tensors[res] = FTensor(name=res, shape=rshp, dtype=rdt)
        fg.ops.append(FOp(name=f"{opname}_{i}", optype=optype, inList=resolved, outList=[res], attrs=attrs))

    for op in fg.ops:
        for t in op.inList:
            if t in fg.tensors:
                fg.tensors[t].op_in.append(op.name)
        for t in op.outList:
            fg.tensors[t].op_out.append(op.name)
    return fg


def fgraph_to_polaris(fg):
    gg = WorkloadGraph(fg.name)
    gg.add_hdr_info(fg.hdr)
    for t in fg.tensors.values():
        gg.add_tensor(SimTensor({
            "name": t.name, "shape": [int(d) for d in t.shape], "dtype": np.dtype(DT_MAP[t.dtype]),
            "data": t.data, "resolve": "M", "op_in": t.op_in, "op_out": t.op_out,
            "is_param": t.is_param, "is_const": t.is_const,
        }))
    for op in fg.ops:
        gg.add_op(SimOp({
            "name": op.name, "optype": op.optype, "domain": op.domain,
            "inList": op.inList, "outList": op.outList, "attrs": op.attrs,
        }))
    gg.construct_graph()
    return gg


def mlir2graph(wlname, wlpath, **kwargs):
    return fgraph_to_polaris(parse_mlir(wlpath, wlname))
