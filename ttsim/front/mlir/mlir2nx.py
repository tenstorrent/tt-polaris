#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""MLIR -> Polaris WorkloadGraph frontend (v6).

Supports two compiler-generated MLIR dialects:
  * TTIR       (tt-mlir, destination-passing style)  -> parse_ttir
  * StableHLO  (framework-level, from tt-xla / XLA)   -> parse_stablehlo
Dispatch is automatic based on file contents.  Both dialects lower into the
same in-memory FGraph, then into a Polaris WorkloadGraph (ONNX-shaped ops),
so the existing Polaris shape-inference + device simulation is reused as-is.
"""
import re
from math import prod
from dataclasses import dataclass, field
import numpy as np

from ttsim.graph import WorkloadGraph
from ttsim.ops import SimOp, SimTensor

DT_MAP = {
    "f64": np.float64, "f32": np.float32, "f16": np.float16, "bf16": np.float16,
    "i64": np.int64, "i32": np.int32, "i16": np.int16, "i8": np.int8, "i1": np.bool_,
    "ui32": np.uint32, "ui8": np.uint8,
}
_TT_RE = re.compile(r"tensor<([^>]*)>")


def _parse_ttype(t):
    m = _TT_RE.search(t)
    if not m:
        return None, None
    body = m.group(1).split(",")[0].strip()
    parts = body.split("x")
    dims = parts[:-1]
    if not all(d.strip().lstrip("-").isdigit() for d in dims):
        if len(parts) == 1:              # scalar: tensor<f32>
            return [], parts[0]
        return None, parts[-1]
    return [int(d) for d in dims], parts[-1]


def _result_type(rhs):
    if "->" in rhs:
        m = _TT_RE.search(rhs.split("->")[-1])
        if m:
            return _parse_ttype("tensor<" + m.group(1) + ">")
    ms = _TT_RE.findall(rhs)
    if ms:
        return _parse_ttype("tensor<" + ms[-1] + ">")
    return None, None


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

    def T(self, name, shape, dtype, data=None, is_param=False, is_const=False):
        self.tensors[name] = FTensor(name=name, shape=shape, dtype=dtype,
                                     data=data, is_param=is_param, is_const=is_const)

    def add_shape_const(self, name, dims):
        self.T(name, [len(dims)], "i64", data=np.array(dims, dtype=np.int64), is_const=True)


# ============================================================ shared helpers
def _resolve_fn(alias):
    def resolve(t):
        seen = set()
        while t in alias and t not in seen:
            seen.add(t); t = alias[t]
        return t
    return resolve


def _emit_reshape(fg, i, src, tag, out_shape, dt):
    sh = tag + ".s"
    fg.add_shape_const(sh, out_shape)
    fg.T(tag, list(out_shape), dt)
    fg.ops.append(FOp(name=f"rs_{i}_{tag[-5:]}", optype="Reshape", inList=[src, sh], outList=[tag]))
    return tag


def _emit_dot_general(fg, i, res, a, b, cl, cr, bl, br, rshp, rdt):
    """Normalize an arbitrary dot_general into (optional Transpose)+Reshape+batched MatMul+Reshape."""
    L, R = fg.tensors[a].shape, fg.tensors[b].shape
    if (not bl and not br and cl == [len(L) - 1] and cr == [0] and len(R) == 2):
        fg.T(res, rshp, rdt)
        fg.ops.append(FOp(name=f"matmul_{i}", optype="MatMul", inList=[a, b], outList=[res]))
        return
    lf = [d for d in range(len(L)) if d not in bl and d not in cl]
    rf = [d for d in range(len(R)) if d not in br and d not in cr]
    permL, permR = bl + lf + cl, br + cr + rf
    B = prod(L[x] for x in bl) if bl else 1
    M = prod(L[x] for x in lf) if lf else 1
    Kd = prod(L[x] for x in cl) if cl else 1
    N = prod(R[x] for x in rf) if rf else 1
    if permL != list(range(len(L))):
        lt = res + ".lt"; fg.T(lt, [L[x] for x in permL], rdt)
        fg.ops.append(FOp(name=f"tL_{i}", optype="Transpose", inList=[a], outList=[lt], attrs={"perm": permL})); a = lt
    a = _emit_reshape(fg, i, a, res + ".lr", [B, M, Kd], rdt)
    if permR != list(range(len(R))):
        rt = res + ".rt"; fg.T(rt, [R[x] for x in permR], rdt)
        fg.ops.append(FOp(name=f"tR_{i}", optype="Transpose", inList=[b], outList=[rt], attrs={"perm": permR})); b = rt
    b = _emit_reshape(fg, i, b, res + ".rr", [B, Kd, N], rdt)
    mm = res + ".mm"; fg.T(mm, [B, M, N], rdt)
    fg.ops.append(FOp(name=f"mm_{i}", optype="MatMul", inList=[a, b], outList=[mm]))
    sh = res + ".s"; fg.add_shape_const(sh, rshp); fg.T(res, rshp, rdt)
    fg.ops.append(FOp(name=f"dgout_{i}", optype="Reshape", inList=[mm, sh], outList=[res]))


# ============================================================ StableHLO
_SH_OPMAP = {"add": "Add", "multiply": "Mul", "subtract": "Sub", "divide": "Div",
             "maximum": "Max", "minimum": "Min", "sqrt": "Sqrt", "rsqrt": "Sqrt",
             "exponential": "Exp", "tanh": "Tanh", "negate": "Neg", "logistic": "Sigmoid"}


def parse_stablehlo(text, wlname):
    fg = FGraph(name=wlname, hdr={"name": wlname, "framework_type": "MLIR"})
    # isolate the @main function body (ignore helper subfunctions like @_take)
    chunks = text.split("func.func")
    main = next((c for c in chunks if "@main" in c), None)
    if main is not None:
        text = "func.func" + main
    sig = re.search(r"@(\w+)\s*\((.*?)\)\s*->", text, re.S)
    assert sig, "no func.func signature"
    fg.name = "main"; fg.hdr["name"] = "main"
    args = re.findall(r"%(arg\d+)\s*:\s*(tensor<[^>]*>)", sig.group(2))
    _FLOATS = {"f64", "f32", "f16", "bf16"}
    for idx, (nm, ty) in enumerate(args):
        shp, dt = _parse_ttype(ty)
        # weights = float-typed args (embeddings, projections, layernorm scale/bias);
        # runtime inputs (input_ids / mask / position ids) are integer-typed -> not params.
        is_weight = (dt in _FLOATS) and (shp is not None) and (len(shp) >= 1)
        fg.T(nm, shp, dt, is_param=is_weight)
    alias: dict = {}
    resolve = _resolve_fn(alias)
    line_re = re.compile(r"^\s*%([\w.]+)\s*=\s*(.*)$")
    i = 0
    for raw in text.splitlines():
        m = line_re.match(raw)
        if not m:
            continue
        res, rhs = m.group(1), m.group(2).strip()
        i += 1
        if rhs.startswith("stablehlo.constant") or rhs.startswith('"stablehlo.constant'):
            shp, dt = _result_type(rhs)
            fg.T(res, shp if shp else [1], dt, is_const=True)
            continue
        if rhs.startswith("stablehlo.iota"):
            shp, dt = _result_type(rhs); fg.T(res, shp, dt, is_const=True); continue
        mo = re.match(r'"?(?:(stablehlo|chlo|mhlo)\.)?([a-z_]+)"?', rhs)
        if not mo:
            continue
        op = mo.group(2)
        rshp, rdt = _result_type(rhs)
        operands = re.findall(r"%([\w.]+)", rhs)
        if op == "reduce":
            operand = resolve(operands[0])
            mi = re.search(r"applies\s+\w+\.(\w+)\s+across", rhs)
            inner = mi.group(1) if mi else "add"
            dims = [int(x) for x in re.findall(r"dimensions\s*=\s*\[([0-9,\s]*)\]", rhs)[0].split(",") if x.strip()]
            optype = {"add": "ReduceSum", "maximum": "ReduceMax", "minimum": "ReduceMin"}.get(inner, "ReduceSum")
            fg.T(res, rshp, rdt)
            fg.ops.append(FOp(name=f"{inner}_{i}", optype=optype, inList=[operand], outList=[res],
                              attrs={"axes": dims, "keepdims": 0}))
            continue
        if op == "convert":
            alias[res] = resolve(operands[0]); continue
        if op == "broadcast_in_dim":
            src = resolve(operands[0]); sin = fg.tensors[src].shape
            nin = prod(sin) if sin else 1
            nout = prod(rshp) if rshp else 1
            if not sin:                       # scalar -> transparent
                alias[res] = src
            elif nin == nout:                 # pure reshape (unit-dim insert)
                _emit_reshape(fg, i, src, res, rshp, rdt)
            else:                             # true expand -> rely on Polaris broadcast
                alias[res] = src
            continue
        if op == "transpose":
            perm = [int(x) for x in re.findall(r"dims\s*=\s*\[([0-9,\s]*)\]", rhs)[0].split(",") if x.strip()]
            src = resolve(operands[0]); fg.T(res, rshp, rdt)
            fg.ops.append(FOp(name=f"transpose_{i}", optype="Transpose", inList=[src], outList=[res], attrs={"perm": perm}))
            continue
        if op == "reshape":
            src = resolve(operands[0]); sh = res + ".s"; fg.add_shape_const(sh, rshp); fg.T(res, rshp, rdt)
            fg.ops.append(FOp(name=f"reshape_{i}", optype="Reshape", inList=[src, sh], outList=[res]))
            continue
        if op == "dot_general":
            a, b = resolve(operands[0]), resolve(operands[1])
            cd = re.search(r"contracting_dims\s*=\s*\[([0-9,\s]*)\]\s*x\s*\[([0-9,\s]*)\]", rhs)
            bd = re.search(r"batching_dims\s*=\s*\[([0-9,\s]*)\]\s*x\s*\[([0-9,\s]*)\]", rhs)
            g = lambda mm, k: [int(x) for x in mm.group(k).split(",") if x.strip()] if mm else []
            _emit_dot_general(fg, i, res, a, b, g(cd, 1), g(cd, 2), g(bd, 1), g(bd, 2), rshp, rdt)
            continue
        if op in ("call", "gather", "compare", "select", "and", "or", "not", "slice",
                  "dynamic_slice", "dynamic_update_slice", "concatenate", "clamp", "reverse",
                  "is_finite", "sign", "floor"):
            # non-compute bookkeeping (embedding lookup / attention-mask machinery):
            # native BERT models do not model these either; keep result as a leaf so the
            # heavy transformer compute (the dot_generals/layernorm/softmax/ffn) is exact.
            fg.T(res, rshp if rshp else [1], rdt, is_const=True); continue
        if op == "gelu":
            src = resolve(operands[0]); fg.T(res, rshp, rdt)
            fg.ops.append(FOp(name=f"gelu_{i}", optype="Gelu", inList=[src], outList=[res])); continue
        if op in ("erf", "erfc"):
            src = resolve(operands[0]); fg.T(res, rshp, rdt)
            fg.ops.append(FOp(name=f"erf_{i}", optype="Erf", inList=[src], outList=[res])); continue
        if op in _SH_OPMAP:
            ins = [resolve(x) for x in operands]
            fg.T(res, rshp, rdt)
            fg.ops.append(FOp(name=f"{op}_{i}", optype=_SH_OPMAP[op], inList=ins, outList=[res]))
            continue
        raise AssertionError(f"unsupported StableHLO op '{op}' :: {rhs[:80]}")
    _cleanup(fg)
    return fg


# ============================================================ TTIR
_TTIR_OPMAP = {
    "matmul": "MatMul", "add": "Add", "mul": "Mul", "multiply": "Mul",
    "subtract": "Sub", "sub": "Sub", "div": "Div", "divide": "Div",
    "maximum": "Max", "minimum": "Min", "relu": "Relu", "gelu": "Gelu",
    "sigmoid": "Sigmoid", "tanh": "Tanh", "exp": "Exp", "sqrt": "Sqrt",
    "rsqrt": "Sqrt", "transpose": "Transpose", "permute": "Transpose",
    "sum": "ReduceSum", "max": "ReduceMax", "mean": "ReduceMean", "concat": "Concat",
}
_TTIR_TRANSPARENT = {"broadcast", "broadcast_in_dim", "convert", "typecast"}
_TTIR_DROP = {"empty", "iota"}


def _ttir_reduce_attrs(attrblk):
    attrs: dict = {}
    dm = re.search(r"dim_arg\s*=\s*\[([^\]]*)\]", attrblk)
    if dm:
        attrs["axes"] = [int(x) for x in re.findall(r"-?\d+", dm.group(1).split(":")[0])]
    kd = re.search(r"keep_dim\s*=\s*(true|false)", attrblk)
    if kd:
        attrs["keepdims"] = 1 if kd.group(1) == "true" else 0
    pm = re.search(r"permutation\s*=\s*array<i64:\s*([\d,\s]*)>", attrblk)
    if pm:
        attrs["perm"] = [int(x) for x in pm.group(1).split(",") if x.strip()]
    return attrs


def _ttir_dim_list(attrblk, key):
    m = re.search(key + r"\s*=\s*array<i64:?\s*([^>]*)>", attrblk)
    return [int(x) for x in re.findall(r"-?\d+", m.group(1))] if m else []


def parse_ttir(text, wlname):
    fg = FGraph(name=wlname, hdr={"name": wlname, "framework_type": "MLIR"})
    sig = re.search(r"func\.func\s+(?:public\s+|private\s+)?@(\w+)\s*\((.*?)\)\s*->", text, re.S)
    assert sig, "no func.func signature"
    fg.name = sig.group(1); fg.hdr["name"] = sig.group(1)
    for arg in [a.strip() for a in sig.group(2).split(",") if a.strip()]:
        am = re.match(r"%([\w.]+)\s*:\s*(tensor<[^>]*>)", arg)
        if not am:
            continue
        shp, dt = _parse_ttype(am.group(2)); fg.T(am.group(1), shp, dt)
    op_re = re.compile(
        r'%([\w.]+)\s*=\s*"?(?:(\w+)\.)?(\w+)"?\s*\(([^)]*)\)\s*'
        r'(<\{.*?\}>)?\s*(\{.*?\})?\s*:\s*\(([^)]*)\)\s*->\s*(tensor<[^>]*>)')
    alias = {}; raw = []
    for m in op_re.finditer(text):
        res, _d, opname, operands, aattr, battr, _it, restype = m.groups()
        inames = [o.strip().lstrip("%") for o in operands.split(",") if o.strip()]
        if opname == "constant":
            shp, dt = _parse_ttype(restype)
            fg.T(res, shp if shp is not None else [], dt, is_const=True); continue
        if opname in _TTIR_DROP:
            continue
        if opname in _TTIR_TRANSPARENT:
            alias[res] = inames[0] if inames else None; continue
        raw.append((res, opname, inames, (aattr or "") + (battr or ""), restype))
    resolve = _resolve_fn(alias)
    for i, (res, opname, inames, attrblk, restype) in enumerate(raw):
        resolved = [x for x in (resolve(y) for y in inames) if x is not None]
        rshp, rdt = _parse_ttype(restype)
        if opname == "dot_general":
            a, b = resolved[0], resolved[1]
            _emit_dot_general(fg, i, res, a, b,
                              _ttir_dim_list(attrblk, "contract_dims_lhs"),
                              _ttir_dim_list(attrblk, "contract_dims_rhs"),
                              _ttir_dim_list(attrblk, "batch_dims_lhs"),
                              _ttir_dim_list(attrblk, "batch_dims_rhs"), rshp, rdt)
            continue
        if opname == "reshape":
            sh = res + ".s"; fg.add_shape_const(sh, rshp); fg.T(res, rshp, rdt)
            fg.ops.append(FOp(name=f"reshape_{i}", optype="Reshape",
                              inList=[resolved[0] if resolved else None, sh], outList=[res]))
            continue
        assert opname in _TTIR_OPMAP, f"unsupported TTIR op '{opname}'"
        fg.T(res, rshp, rdt)
        fg.ops.append(FOp(name=f"{opname}_{i}", optype=_TTIR_OPMAP[opname],
                          inList=resolved, outList=[res], attrs=_ttir_reduce_attrs(attrblk)))
    _cleanup(fg)
    return fg


# ============================================================ finalize
def _cleanup(fg):
    """Graph optimization: drop no-op Reshape (in-shape == out-shape) and identity
    Transpose (perm == 0,1,2,...), re-pointing consumers. Removes redundant data-movement
    ops introduced by StableHLO decomposition / dot_general normalization, so vector and
    memory cycle counts better reflect the real (fused) hardware graph."""
    alias = {}
    kept = []
    for op in fg.ops:
        drop = False
        if op.optype == "Reshape" and op.inList and op.outList:
            src, out = op.inList[0], op.outList[0]
            if src in fg.tensors and out in fg.tensors and \
               list(fg.tensors[src].shape or []) == list(fg.tensors[out].shape or []):
                drop = True
        elif op.optype == "Transpose" and op.outList:
            perm = op.attrs.get("perm")
            if perm is not None and list(perm) == list(range(len(perm))):
                drop = True
        if drop:
            alias[op.outList[0]] = op.inList[0]
        else:
            kept.append(op)

    def res(t):
        seen = set()
        while t in alias and t not in seen:
            seen.add(t); t = alias[t]
        return t

    for op in kept:
        op.inList = [res(t) for t in op.inList]
    fg.ops = kept
    for t in list(alias):
        fg.tensors.pop(t, None)
    for t in fg.tensors.values():
        t.op_in = []; t.op_out = []
    _wire(fg)


def _wire(fg):
    for op in fg.ops:
        for t in op.inList:
            if t in fg.tensors:
                fg.tensors[t].op_in.append(op.name)
        for t in op.outList:
            if t in fg.tensors:
                fg.tensors[t].op_out.append(op.name)


def parse_mlir(path, wlname=None):
    text = open(path).read()
    wlname = wlname or "mlir_wl"
    # detect dialect by the actual ops. Note: real TTIR files can still carry leftover
    # `mhlo.*` module attributes, so check for ttir ops FIRST.
    if "ttir." in text:
        return parse_ttir(text, wlname)
    if "stablehlo." in text or "mhlo." in text:
        return parse_stablehlo(text, wlname)
    return parse_ttir(text, wlname)


def fgraph_to_polaris(fg, float_dtype=np.float16):
    # float_dtype = np.float16 acts as the bf16 (2-byte) proxy so tensor byte-counting
    # matches Polaris' bf16 compute precision (removes the 'bpe 4 != 2' warning).
    # Integer tensors (input_ids, shape/index consts) are kept as their integer dtype.
    gg = WorkloadGraph(fg.name)
    gg.add_hdr_info(fg.hdr)
    for t in fg.tensors.values():
        shp = [int(d) for d in t.shape] if t.shape else []
        npdt = np.dtype(DT_MAP[t.dtype])
        if npdt.kind == "f":                 # floating-point -> bf16 proxy
            npdt = np.dtype(float_dtype)
        gg.add_tensor(SimTensor({
            "name": t.name, "shape": shp, "dtype": npdt,
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
