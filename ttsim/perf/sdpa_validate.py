# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Validation harness: tt-metal tracy ops_perf_results CSV to SDPA/TopK roofline signed errors. Reads the ops
CSV from tools/tracy/process_ops_logs.py, keeps the SDPA and TopK device ops, rebuilds each config from the
ATTRIBUTES and INPUT columns through the shim readers (sdpa_config_from_shapes / decode_config_from_shapes),
runs the roofline and reports per-row signed errors plus the gap_analysis.md section 4.3 criterion."""
from __future__ import annotations

import argparse
import ast
import csv
import importlib
import json
import math
import re
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ttsim.perf.roofline_sdpa import (ArchConfig, decode_config_from_shapes, predict,
                                       predict_decode, sdpa_config_from_shapes)

PALETTE = ("#2a78d6", "#eb6834", "#1baf7a", "#e34948")
INK, BG = "#0b0b0b", "#fcfcfb"

TOLERANCE = 0.10            # +-10 percent per op
BIAS_LIMIT = 0.03           # |time-weighted mean signed error| ceiling
TIME_WITHIN_MIN = 0.90      # share of op time that must sit inside the tolerance
BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_SEED = 4716

MEAS_COL = "DEVICE KERNEL DURATION [ns]"

# tracy DATATYPE strings to the roofline dtype names and bytes per element.
_DTYPE_NAMES = {"BFLOAT4_B": "bfp4_b", "BFLOAT8_B": "bfp8_b", "BFLOAT16": "bfloat16", "FLOAT32": "float32",
                "INT32": "int32", "UINT32": "uint32", "UINT16": "uint16", "UINT8": "uint8", "INT8": "int8"}
_DTYPE_BYTES = {"bfp4_b": 0.5, "bfp8_b": 1.0, "bfloat16": 2.0, "float32": 4.0, "int32": 4.0, "uint32": 4.0,
                "uint16": 2.0, "uint8": 1.0, "int8": 1.0}
_INT_DTYPES = {"int32", "uint32", "uint16", "uint8", "int8"}


# ----------------------------------------------------------------------------------------------
# CSV reading and cell parsing
# ----------------------------------------------------------------------------------------------

def read_ops_csv(path) -> Tuple[List[str], List[Dict[str, str]]]:
    """DictReader over an ops CSV. Leading '#' lines (PROVENANCE) are returned separately."""
    comments: List[str] = []
    lines: List[str] = []
    with open(path, newline="") as f:
        for line in f:
            if not lines and line.startswith("#"):
                comments.append(line.rstrip("\n"))
            else:
                lines.append(line)
    return comments, list(csv.DictReader(lines))


def classify_op_code(code: str) -> Optional[str]:
    """Map an OP CODE to the harness op kind, or None for ops the harness ignores."""
    c = (code or "").lower()
    if "topk" in c or "top_k" in c:
        if "route" in c:
            return "topk_route"
        return "topk_large_indices" if "large" in c else "topk"
    if "scaleddotproductattention" in c or "sdpa" in c:
        return "sdpa_decode" if "decode" in c else "sdpa_prefill"
    return None


def _split_top_level(s: str) -> List[str]:
    parts: List[str] = []
    cur: List[str] = []
    depth = 0
    for ch in s:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    if cur:
        parts.append("".join(cur))
    return [p.strip() for p in parts if p.strip()]


_STRUCT_RE = re.compile(r"^([A-Za-z_][\w:]*)?\((.*)\)$", re.S)


def parse_attr_value(v):
    """One attribute string as tt-metal renders it (fmt on the value, reflection for structs)."""
    if not isinstance(v, str):
        return v
    s = v.strip()
    if s in ("std::nullopt", "nullopt", "None", ""):
        return None
    if s.lower() in ("true", "false"):
        return s.lower() == "true"
    m = _STRUCT_RE.match(s)
    if m and "=" in m.group(2):
        fields = {}
        for part in _split_top_level(m.group(2)):
            if "=" in part:
                k, val = part.split("=", 1)
                fields[k.strip()] = parse_attr_value(val)
        if set(fields) == {"x", "y"}:
            return (fields["x"], fields["y"])
        return fields
    if s.startswith("{") and s.endswith("}"):
        inner = s[1:-1].strip()
        return [parse_attr_value(p) for p in _split_top_level(inner)] if inner else []
    m = re.fullmatch(r"(\d+)-(\d+)", s)
    if m:   # CoreCoord in the T2.5 ops CSVs: compute_with_storage_grid_size=8-8
        return (int(m.group(1)), int(m.group(2)))
    if "::" in s and re.fullmatch(r"[\w:]+", s):
        return s.rsplit("::", 1)[-1]
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        return s


def parse_attributes(cell: str) -> Dict[str, object]:
    """ATTRIBUTES cell to {name: value}. The CSV writer turned every ',' into ';' so the cell is
    the Python repr of a dict of strings with ';' separators; undo that, then parse each value."""
    text = (cell or "").strip()
    if not text:
        return {}
    text = text.replace(";", ",")
    raw = None
    try:
        raw = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        try:
            raw = json.loads(text)
        except (ValueError, TypeError):
            raw = dict(re.findall(r"'([^']+)':\s*'([^']*)'", text))
    if not isinstance(raw, dict):
        return {}
    return {str(k): parse_attr_value(v) for k, v in raw.items()}


def parse_dim(cell) -> Tuple[int, int]:
    """Shape cell 'padded[logical]' (or a bare number) to (padded, logical)."""
    s = str(cell).strip()
    m = re.fullmatch(r"(\d+)\s*\[(\d+)\]", s)
    if m:
        return int(m.group(1)), int(m.group(2))
    n = int(float(s))
    return n, n


def normalize_dtype(cell: str) -> str:
    s = str(cell or "").strip().rsplit("::", 1)[-1].rsplit(".", 1)[-1].upper()
    return _DTYPE_NAMES.get(s, s.lower())


def normalize_fidelity(cell) -> str:
    return str(cell or "").strip().rsplit("::", 1)[-1].rsplit(".", 1)[-1]


@dataclass
class TensorInfo:
    shape: List[int]            # logical W, Z, Y, X
    padded: List[int]
    dtype: str                  # roofline dtype name
    layout: str = ""
    memory: str = ""

    @property
    def element_size(self) -> float:
        return _DTYPE_BYTES.get(self.dtype, 2.0)

    @property
    def is_int(self) -> bool:
        return self.dtype in _INT_DTYPES


def _tensor_columns(row: Dict[str, str], prefix: str) -> List[TensorInfo]:
    out, i = [], 0
    while f"{prefix}_{i}_X_PAD[LOGICAL]" in row:
        cells = [row.get(f"{prefix}_{i}_{d}_PAD[LOGICAL]", "") for d in ("W", "Z", "Y", "X")]
        if not all(str(c).strip() for c in cells):
            break
        dims = [parse_dim(c) for c in cells]
        out.append(TensorInfo(shape=[d[1] for d in dims], padded=[d[0] for d in dims],
                              dtype=normalize_dtype(row.get(f"{prefix}_{i}_DATATYPE", "")),
                              layout=str(row.get(f"{prefix}_{i}_LAYOUT", "") or ""),
                              memory=str(row.get(f"{prefix}_{i}_MEMORY", "") or "")))
        i += 1
    return out


def _num(cell, default=None):
    try:
        return float(cell)
    except (TypeError, ValueError):
        return default


@dataclass
class OpRow:
    index: int
    op_code: str
    kind: str
    global_call_count: str
    device_id: str
    trace_id: str
    attrs: Dict[str, object]
    inputs: List[TensorInfo]
    outputs: List[TensorInfo]
    meas_ns: float
    core_count: int
    fidelity_col: str
    per_core_max_ns: Optional[float] = None
    per_core_avg_ns: Optional[float] = None
    fw_ns: Optional[float] = None
    pm_ideal_ns: Optional[float] = None
    arch: str = ""
    replays: int = 1


def select_op_rows(rows: Sequence[Dict[str, str]]) -> List[OpRow]:
    """Keep the SDPA and TopK device op rows (OP TYPE tt_dnn_device, signposts dropped)."""
    out = []
    for i, r in enumerate(rows):
        op_type = str(r.get("OP TYPE", "")).strip()
        if op_type and op_type != "tt_dnn_device":      # signposts and host side rows
            continue
        kind = classify_op_code(r.get("OP CODE", ""))
        if kind is None:
            continue
        meas = _num(r.get(MEAS_COL))
        if meas is None or meas <= 0:                   # predict_row divides by the duration
            continue
        out.append(OpRow(
            index=i, op_code=str(r.get("OP CODE", "")).strip(), kind=kind,
            global_call_count=str(r.get("GLOBAL CALL COUNT", "")).strip(),
            device_id=str(r.get("DEVICE ID", "")).strip(), trace_id=str(r.get("METAL TRACE ID", "")).strip(),
            attrs=parse_attributes(r.get("ATTRIBUTES", "")),
            inputs=_tensor_columns(r, "INPUT"), outputs=_tensor_columns(r, "OUTPUT"),
            meas_ns=meas, core_count=int(_num(r.get("CORE COUNT"), 0) or 0),
            fidelity_col=normalize_fidelity(r.get("MATH FIDELITY", "")),
            per_core_max_ns=_num(r.get("DEVICE KERNEL DURATION PER CORE MAX [ns]")),
            per_core_avg_ns=_num(r.get("DEVICE KERNEL DURATION PER CORE AVG [ns]")),
            fw_ns=_num(r.get("DEVICE FW DURATION [ns]")), pm_ideal_ns=_num(r.get("PM IDEAL [ns]")),
            arch=str(r.get("DEVICE ARCH", "")).strip()))
    return out


def aggregate_replays(rows: List[OpRow]) -> List[OpRow]:
    """Traced programs appear once per replay session; collapse them to the median duration."""
    groups: Dict[tuple, List[OpRow]] = {}
    order = []
    for r in rows:
        key = (r.op_code, r.global_call_count, r.device_id, r.trace_id) if r.trace_id else ("row", r.index)
        if key not in groups:
            order.append(key)
        groups.setdefault(key, []).append(r)
    out = []
    for key in order:
        grp = groups[key]
        head = grp[0]
        head.meas_ns = float(statistics.median(g.meas_ns for g in grp))
        head.replays = len(grp)
        out.append(head)
    return out


def read_positions_sidecar(path) -> Dict[str, Dict[str, int]]:
    """positions.csv keyed by GLOBAL CALL COUNT: cur_pos (int or space separated list, the max is
    used) and an optional page_block_size column."""
    if not path:
        return {}
    _, rows = read_ops_csv(path)
    out = {}
    for r in rows:
        key = str(r.get("GLOBAL CALL COUNT", r.get("global_call_count", ""))).strip()
        if not key:
            continue
        entry: Dict[str, int] = {}
        pos = str(r.get("cur_pos", "")).replace("|", " ").replace(";", " ").split()
        if pos:
            entry["cur_pos"] = max(int(float(p)) for p in pos)
        bs = _num(r.get("page_block_size"))
        if bs:
            entry["page_block_size"] = int(bs)
        out[key] = entry
    return out


# ----------------------------------------------------------------------------------------------
# Row to config
# ----------------------------------------------------------------------------------------------

class RowError(ValueError):
    pass


def _struct(attrs, name) -> Dict[str, object]:
    v = attrs.get(name)
    return v if isinstance(v, dict) else {}


def _as_int(v, default: int = 0) -> int:
    """Attr values come off the CSV as object; anything not numeric or numeric-looking is the default."""
    if isinstance(v, bool) or v is None or v == "":
        return default
    try:
        return int(v)  # type: ignore[call-overload]
    except (TypeError, ValueError):
        return default


def _core_range_set_size(v) -> int:
    """Core count of a CoreRangeSet as tt-metal renders it, {[(x=0,y=0) - (x=7,y=3)], ...}, whether the
    attribute parser left it as text or turned the corners into (x, y) tuples. 0 when nothing parses."""
    corners: list = []

    def walk(o):
        if isinstance(o, tuple) and len(o) == 2 and all(isinstance(c, (int, float)) for c in o):
            corners.append((int(o[0]), int(o[1])))
        elif isinstance(o, (list, tuple)):
            for c in o:
                walk(c)
        elif isinstance(o, str):
            corners.extend((int(x), int(y)) for x, y in re.findall(r"x=(\d+)[;,]\s*y=(\d+)", o))

    walk(v)
    if not corners or len(corners) % 2:
        return 0
    return sum((abs(x1 - x0) + 1) * (abs(y1 - y0) + 1)
               for (x0, y0), (x1, y1) in zip(corners[0::2], corners[1::2]))


def _common_attrs(r: OpRow) -> Dict[str, object]:
    """Program and compute config attrs under the names the ttnn shim records."""
    a: Dict[str, object] = {}
    pc = _struct(r.attrs, "program_config")
    if pc:
        a["q_chunk_size"] = _as_int(pc.get("q_chunk_size"))
        a["k_chunk_size"] = _as_int(pc.get("k_chunk_size"))
        grid = pc.get("compute_with_storage_grid_size")
        sub = pc.get("sub_core_grids")
        if sub is not None:
            # sub_core_grids overrides the grid, counted the way the shim counts it; a set that does not
            # parse leaves the core count to CORE COUNT and says so
            cores = _core_range_set_size(sub)
            if cores:
                a["grid_cores"] = cores
            else:
                a["sub_core_grids_unparsed"] = True
        elif isinstance(grid, tuple) and len(grid) == 2:
            a["grid_cores"] = int(grid[0]) * int(grid[1])
        if pc.get("exp_approx_mode") is not None:
            a["exp_approx_mode"] = bool(pc["exp_approx_mode"])
        a["max_cores_per_head_batch"] = _as_int(pc.get("max_cores_per_head_batch"), 16)
    ck = _struct(r.attrs, "compute_kernel_config")
    if ck:
        if ck.get("math_fidelity") is not None:
            a["fidelity"] = normalize_fidelity(ck["math_fidelity"])
        for f in ("fp32_dest_acc_en", "packer_l1_acc", "math_approx_mode"):
            if ck.get(f) is not None:
                a[f] = bool(ck[f])
    if r.fidelity_col and a.get("fidelity") and r.fidelity_col != a["fidelity"]:
        raise RowError(f"fidelity_column_mismatch:{r.fidelity_col}!={a['fidelity']}")
    if r.fidelity_col and not a.get("fidelity"):
        a["fidelity"] = r.fidelity_col
    # CORE COUNT is the launched grid, measured; the program-config grid is only a cross-check.
    if r.core_count:
        a["num_cores"] = r.core_count
    elif a.get("grid_cores"):
        a["num_cores"] = a["grid_cores"]
    sw = r.attrs.get("sliding_window_size")
    if isinstance(sw, (int, float)) and sw:
        a["sliding_window_size"] = int(sw)
    hv = r.attrs.get("head_dim_v")
    if r.attrs.get("use_mla") and isinstance(hv, (int, float)) and hv:
        a["head_dim_v"] = int(hv)
    geo = _struct(r.attrs, "paged_cache_geometry")
    if geo.get("block_size"):
        a["page_block_size"] = _as_int(geo["block_size"])
    return a


def _classify_optional_inputs(r: OpRow, fixed: int, kv_len: int, decode: bool) -> Dict[str, TensorInfo]:
    """Optional tensor inputs are appended only when present, so they are told apart by dtype,
    shape and the ATTRIBUTES presence flags together with the declared order, not by position."""
    found: Dict[str, TensorInfo] = {}
    ints = [t for t in r.inputs[fixed:] if t.is_int]
    floats = [t for t in r.inputs[fixed:] if not t.is_int]
    paged = bool(r.attrs.get("paged_attention")) or r.attrs.get("chunk_start_idx") is not None \
        or r.attrs.get("chunk_start_idx_tensor") is not None
    if decode:
        # declared order cur_pos_tensor, page_table_tensor; a lone int tensor is the table when paged
        if len(ints) >= 2:
            found["cur_pos_tensor"], found["page_table"] = ints[0], ints[1]
        elif len(ints) == 1:
            found["page_table" if paged else "cur_pos_tensor"] = ints[0]
    else:
        # declared order page_table, chunk_start_idx_tensor, cu_window_seqlens
        for t in ints:
            if int(np.prod(t.shape)) == 1:
                found.setdefault("chunk_start_idx_tensor", t)
            elif paged and "page_table" not in found:
                found["page_table"] = t
            else:
                found.setdefault("cu_window_seqlens", t)
    for t in floats:
        if t.shape[-1] == kv_len and kv_len > 1 and "attn_mask" not in found:
            found["attn_mask"] = t
        else:
            found.setdefault("attention_sink", t)
    return found


def prefill_config(r: OpRow, sidecar: Optional[Dict[str, int]] = None):
    """SdpaConfig for a prefill / chunked / MLA / sparse row."""
    if len(r.inputs) < 2:
        raise RowError("parse_error:prefill row needs q and k inputs")
    a = _common_attrs(r)
    use_mla = bool(r.attrs.get("use_mla"))
    q, k = r.inputs[0], r.inputs[1]
    fixed = 2 if use_mla else 3
    v = k if use_mla or len(r.inputs) < 3 else r.inputs[2]
    a["input_dtype"] = q.dtype
    a["element_size"] = q.element_size
    a["kv_element_size"] = k.element_size
    a["is_causal"] = bool(r.attrs.get("is_causal", True))
    csi = r.attrs.get("chunk_start_idx")
    if isinstance(csi, (int, float)):
        a["chunk_start_idx"] = int(csi)
    elif r.attrs.get("chunk_start_idx_tensor") is not None:
        a["chunk_start_unknown"] = True
    if "joint" in r.op_code.lower():
        # JointSDPADeviceOperation: q, k, v, joint_q, joint_k, joint_v. Both streams attend their
        # concatenation, priced as one non causal prefill over S + joint_seq (ttsim_layout._sdpa_joint_perf).
        if len(r.inputs) < 6:
            raise RowError("parse_error:joint row needs the six q/k/v inputs")
        joint_seq = int(r.inputs[3].shape[-2])
        a["is_joint"], a["joint_seq"], a["is_causal"] = True, joint_seq, False
        seff = int(q.shape[-2]) + joint_seq
        qs, ks, vs = (list(t.shape[:-2]) + [seff, t.shape[-1]] for t in (q, k, v))
        return sdpa_config_from_shapes(qs, ks, vs, a, num_cores=a.get("num_cores", 110))
    opt = _classify_optional_inputs(r, fixed, k.shape[-2], decode=False)
    if "attn_mask" in opt:
        a["has_attn_mask"] = True
    if "attention_sink" in opt:
        a["attention_sink"] = True
    if "cu_window_seqlens" in opt or r.attrs.get("is_windowed"):
        a["is_windowed"] = True
    page_table_shape = None
    if "page_table" in opt:
        a["paged"] = True
        page_table_shape = opt["page_table"].shape
    if "chunk_start_idx_tensor" in opt and "chunk_start_idx" not in a:
        a["chunk_start_unknown"] = True
    if sidecar and sidecar.get("page_block_size") and not a.get("page_block_size"):
        a["page_block_size"] = sidecar["page_block_size"]
    if "sparse" in r.op_code.lower() and len(r.inputs) >= 3 and r.inputs[2].is_int:
        a["is_sparse"] = True
        a["kv_seq"] = r.inputs[2].shape[-1]
        v = k
    return sdpa_config_from_shapes(q.shape, k.shape, v.shape, a, num_cores=a.get("num_cores", 110),
                                   page_table_shape=page_table_shape)


def decode_config(r: OpRow, sidecar: Optional[Dict[str, int]] = None) -> Dict[str, object]:
    """predict_decode keyword arguments for a decode row (paged or flat cache)."""
    if len(r.inputs) < 2:
        raise RowError("parse_error:decode row needs q and k inputs")
    a = _common_attrs(r)
    use_mla = bool(r.attrs.get("use_mla"))
    q, k = r.inputs[0], r.inputs[1]
    fixed = 2 if use_mla else 3
    v = None if use_mla or len(r.inputs) < 3 else r.inputs[2]
    a["element_size"] = q.element_size
    a["kv_element_size"] = k.element_size
    if "L1" in q.memory.upper():
        a["q_in_l1"] = True          # height sharded query of the model-level rows: no DRAM read for Q
    a["is_causal"] = bool(r.attrs.get("is_causal", True))
    kc = r.attrs.get("k_chunk_size")
    if isinstance(kc, (int, float)) and kc:
        a["k_chunk_size"] = int(kc)
    opt = _classify_optional_inputs(r, fixed, k.shape[-2], decode=True)
    if "attn_mask" in opt:
        a["has_attn_mask"] = True
    page_table_shape = None
    if "page_table" in opt or r.attrs.get("paged_attention"):
        a["paged"] = True
        if "page_table" in opt:
            page_table_shape = opt["page_table"].shape
    cur = r.attrs.get("cur_pos")
    if isinstance(cur, list) and cur:
        a["cur_pos"] = int(max(cur))
    elif isinstance(cur, (int, float)) and not isinstance(cur, bool):
        a["cur_pos"] = int(cur)          # the shim records the slowest user's position as one number
    elif sidecar and "cur_pos" in sidecar:
        a["cur_pos"] = int(sidecar["cur_pos"])
    if sidecar and sidecar.get("page_block_size") and not a.get("page_block_size"):
        a["page_block_size"] = sidecar["page_block_size"]
    return decode_config_from_shapes(q.shape, k.shape, v.shape if v else None, a,
                                     num_cores=a.get("num_cores", 110), page_table_shape=page_table_shape)


def topk_config(r: OpRow) -> Dict[str, int]:
    """TopK row: N from the last logical dim (or valid_length), K from the attributes, rows from
    the leading dims."""
    if not r.inputs:
        raise RowError("parse_error:topk row has no input")
    t = r.inputs[0]
    k = r.attrs.get("k")
    if not isinstance(k, (int, float)) or k <= 0:
        raise RowError("parse_error:topk row has no k attribute")
    vl = r.attrs.get("valid_length")
    n = int(vl) if isinstance(vl, (int, float)) and vl else int(t.shape[-1])
    rows = int(np.prod(t.shape[:-1])) if len(t.shape) > 1 else 1
    return dict(N=n, K=int(k), rows=rows, num_cores=int(r.core_count or 110))


# ----------------------------------------------------------------------------------------------
# Prediction
# ----------------------------------------------------------------------------------------------

def make_device(archspec=None, device_name="p100a"):
    """Real ttsim Device for one package instance of config/tt_bh.yaml. The operator lookup file
    is dropped so the harness never resolves an lfc:// path; the SDPA path does not use it."""
    from ttsim.back.device import Device
    from ttsim.config import get_arspec_from_yaml
    if archspec is None:
        archspec = Path(__file__).resolve().parents[2] / "config" / "tt_bh.yaml"
    _, pkgs = get_arspec_from_yaml(str(archspec))
    if device_name not in pkgs:
        raise KeyError(f"device {device_name!r} not in {archspec}: {sorted(pkgs)}")
    pkg = pkgs[device_name]
    pkg.operator_lookup_file = None
    return Device(pkg)


def projected_op(device, perf_stats: Dict, fallback_elems: int = 0):
    """The op after Device.execute_op. fallback_elems seeds the generic per element instruction count
    that a device outside the calibration gate is priced with, as sdpa_sinf does, so the projection
    never books zero compute there. bf16 is the precision the device tables can price a generic
    instruction at (block float inputs are bytes to the roofline, not a pipe precision)."""
    op = SimpleNamespace(name="sdpa_validate", optype="ScaledDotProductAttention", uses_compute_pipe="matrix",
                         precision="bf16", repeat_count=1, removed_in_optimization=False,
                         fused_in_optimization=False, fused_with_op=None, fused_op_cycles=None, exec_stats={},
                         compute_cycles=0, mem_rd_cycles=0, mem_wr_cycles=0, mem_rd_cycles_fractional=0.0,
                         mem_wr_cycles_fractional=0.0, perf_stats=dict(perf_stats))
    if not op.perf_stats.get("instrs") and fallback_elems > 0:
        op.perf_stats["instrs"] = {"mov": int(fallback_elems)}
    op.perf_stats.setdefault("instrs", {})
    op.perf_stats.setdefault("inBytes", 0)
    op.perf_stats.setdefault("outBytes", 0)
    device.execute_op(op)
    return op


def device_projection_ns(device, perf_stats: Dict, fallback_elems: int = 0) -> float:
    """Op latency through Device.execute_op, the ideal a workload report would carry:
    max(compute, memory) plus the ramp penalty, in device clocks."""
    op = projected_op(device, perf_stats, fallback_elems)
    ramp = getattr(device.simconfig_obj, "ramp_penalty", lambda: 0.0)() or 0.0
    ideal = math.ceil(max(op.compute_cycles, op.mem_rd_cycles + op.mem_wr_cycles) + ramp)
    return ideal / device.freq_MHz * 1e3


@dataclass
class RowResult:
    row: OpRow
    regime: str = ""
    model: str = ""
    layer: int = 0
    pred_kernel_ns: float = float("nan")
    pred_device_ns: float = float("nan")
    err: float = float("nan")               # signed, fractional
    config: Dict[str, object] = field(default_factory=dict)
    reasons: List[str] = field(default_factory=list)
    defaulted: str = ""
    config_mismatch: str = ""
    exclude_reason: str = ""

    @property
    def excluded(self) -> bool:
        return bool(self.exclude_reason)


_ECHO_KEYS = {"q_chunk_size": "q_chunk", "k_chunk_size": "k_chunk", "num_cores": "num_cores",
              "fidelity": "fidelity", "exp_approx_mode": "exp_approx_mode", "fp32_dest_acc_en": "fp32_dest_acc",
              "is_causal": "is_causal"}


def _config_mismatch(parsed: Dict[str, object], echo: Dict[str, object]) -> str:
    bad = []
    for k, ek in _ECHO_KEYS.items():
        if k in parsed and ek in echo and parsed[k] not in (None, 0) and echo[ek] != parsed[k]:
            bad.append(f"{ek}:{echo[ek]}!={parsed[k]}")
    return ",".join(bad)


def _topk_predict(cfg: Dict[str, int]):
    try:
        rt = importlib.import_module("ttsim.perf.roofline_topk")   # the TopK roofline, where present
    except ImportError:
        return None, "topk_model_unavailable"
    res = rt.predict_topk(rt.TopkConfig(**cfg))
    return res, ""


def predict_row(r: OpRow, *, device=None, sidecar: Optional[Dict[str, int]] = None,
                clock_ghz: Optional[float] = None) -> RowResult:
    """Build the config for one row, run the roofline and fill the signed error. Rows the model
    cannot price honestly are kept but carry exclude_reason."""
    out = RowResult(row=r)
    clock = clock_ghz or ArchConfig().clock_ghz
    # The roofline constants are Blackhole p100a fits; rows from another arch are listed, not scored.
    if r.arch and r.arch.lower() != "blackhole":
        out.regime = r.kind
        out.exclude_reason = f"arch_not_blackhole:{r.arch}"
        return out
    try:
        if r.kind == "sdpa_prefill":
            cfg = prefill_config(r, sidecar)
            res = predict(cfg)
            out.regime = res.wall_regime or res.regime
            out.config = dict(res.config_echo)
            out.reasons = list(res.low_confidence_reasons)
            out.defaulted = res.defaulted
            out.config_mismatch = _config_mismatch(_common_attrs(r), res.config_echo)
            ps = res.to_polaris_op_perf_stats()
        elif r.kind == "sdpa_decode":
            kw = decode_config(r, sidecar)
            res = predict_decode(**kw)
            out.regime = res.wall_regime or "decode"
            out.config = dict(res.config_echo)
            out.reasons = list(res.low_confidence_reasons)
            out.defaulted = res.defaulted
            parsed = _common_attrs(r)
            parsed.pop("q_chunk_size", None)
            parsed.pop("k_chunk_size", None)
            out.config_mismatch = _config_mismatch(parsed, res.config_echo)
            if kw.get("cur_pos_unknown"):
                out.exclude_reason = "cur_pos_unknown"
            ps = res.to_polaris_op_perf_stats()
        elif r.kind in ("topk", "topk_large_indices"):
            cfg = topk_config(r)
            out.regime = r.kind
            out.config = dict(cfg)
            tk_res, why = _topk_predict(cfg)
            if tk_res is None:
                out.exclude_reason = why
                return out
            ps = tk_res.to_polaris_op_perf_stats()
        else:
            out.regime = r.kind
            out.exclude_reason = "no_model"
            return out
    except RowError as e:
        out.regime = r.kind
        out.exclude_reason = str(e)
        return out
    except (ValueError, KeyError, IndexError, TypeError) as e:
        out.regime = r.kind
        out.exclude_reason = f"parse_error:{type(e).__name__}:{e}"
        return out
    if out.config_mismatch and not out.exclude_reason:
        out.exclude_reason = "config_mismatch"
    out.pred_kernel_ns = ps["fused_compute_cycles"] / clock
    if device is not None:
        elems = int(np.prod(r.inputs[0].shape)) if r.inputs else 0
        out.pred_device_ns = device_projection_ns(device, ps, fallback_elems=elems)
    out.err = (out.pred_kernel_ns - r.meas_ns) / r.meas_ns
    return out


# ----------------------------------------------------------------------------------------------
# Statistics and criterion
# ----------------------------------------------------------------------------------------------

@dataclass
class RegimeSummary:
    regime: str
    n: int
    n_excluded: int
    mean_err: float
    tw_mean_err: float
    std_err: float
    frac_ops_within: float
    frac_time_within: float
    mean_ci: Tuple[float, float]
    tw_mean_ci: Tuple[float, float]
    passed: bool
    n_low_confidence: int = 0


def bootstrap_mean_ci(errs, weights=None, *, resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED,
                      level=0.95) -> Tuple[float, float]:
    """Percentile bootstrap interval on the (weighted) mean signed error, fixed seed."""
    e = np.asarray(errs, dtype=float)
    n = len(e)
    if n == 0:
        return (float("nan"), float("nan"))
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    if n == 1:
        return (float(e[0]), float(e[0]))
    rng = np.random.default_rng(seed)
    means = np.empty(resamples)
    block = 2000
    for start in range(0, resamples, block):
        m = min(block, resamples - start)
        idx = rng.integers(0, n, size=(m, n))
        ws = w[idx]
        means[start:start + m] = (ws * e[idx]).sum(axis=1) / ws.sum(axis=1)
    alpha = (1.0 - level) / 2.0
    return (float(np.quantile(means, alpha)), float(np.quantile(means, 1.0 - alpha)))


def passes_criterion(tw_mean_err: float, frac_time_within: float, *, bias_limit=BIAS_LIMIT,
                     time_within_min=TIME_WITHIN_MIN) -> bool:
    if math.isnan(tw_mean_err) or math.isnan(frac_time_within):
        return False
    return abs(tw_mean_err) <= bias_limit and frac_time_within >= time_within_min


def summarize_regime(regime: str, errs: Sequence[float], meas_ns: Sequence[float], *, n_excluded=0,
                     n_low_confidence=0, tolerance=TOLERANCE, resamples=BOOTSTRAP_RESAMPLES,
                     seed=BOOTSTRAP_SEED) -> RegimeSummary:
    e = np.asarray(errs, dtype=float)
    w = np.asarray(meas_ns, dtype=float)
    n = len(e)
    if n == 0:
        nan = float("nan")
        return RegimeSummary(regime, 0, n_excluded, nan, nan, nan, nan, nan, (nan, nan), (nan, nan), False,
                             n_low_confidence)
    mean = float(e.mean())
    tw = float((e * w).sum() / w.sum())
    std = float(e.std(ddof=1)) if n > 1 else 0.0
    within = np.abs(e) <= tolerance
    frac_ops = float(within.mean())
    frac_time = float(w[within].sum() / w.sum())
    return RegimeSummary(regime=regime, n=n, n_excluded=n_excluded, mean_err=mean, tw_mean_err=tw, std_err=std,
                         frac_ops_within=frac_ops, frac_time_within=frac_time,
                         mean_ci=bootstrap_mean_ci(e, None, resamples=resamples, seed=seed),
                         tw_mean_ci=bootstrap_mean_ci(e, w, resamples=resamples, seed=seed),
                         passed=passes_criterion(tw, frac_time), n_low_confidence=n_low_confidence)


def summarize(results: Sequence[RowResult], **kw) -> List[RegimeSummary]:
    """One summary per regime in order of first appearance; excluded rows are counted, not scored."""
    regimes: List[str] = []
    for res in results:
        if res.regime not in regimes:
            regimes.append(res.regime)
    out = []
    for reg in regimes:
        rows = [x for x in results if x.regime == reg]
        scored = [x for x in rows if not x.excluded]
        out.append(summarize_regime(reg, [x.err for x in scored], [x.row.meas_ns for x in scored],
                                    n_excluded=len(rows) - len(scored),
                                    n_low_confidence=sum(1 for x in scored if x.reasons), **kw))
    return out


# ----------------------------------------------------------------------------------------------
# Outputs
# ----------------------------------------------------------------------------------------------

ROW_COLUMNS = ["model", "regime", "layer", "GLOBAL CALL COUNT", "op_code", "S", "kv_len", "cur_pos", "q_chunk",
               "k_chunk", "num_cores", "fidelity", "exp_approx", "fp32_acc", "kernel_path", "input_dtype",
               "kv_dtype", "meas_ns", "pred_kernel_ns", "pred_device_ns", "err_pct", "per_core_max_over_avg",
               "pm_ideal_ns", "replays", "low_confidence_reasons", "defaulted_attrs", "config_mismatch",
               "excluded", "exclude_reason"]


def _fmt(x, nd=1):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def row_record(res: RowResult) -> Dict[str, object]:
    r, c = res.row, res.config
    fp32 = c.get("fp32_dest_acc")
    if fp32 is None and c.get("accum_dtype"):
        fp32 = c.get("accum_dtype") == "float32"
    ratio = (r.per_core_max_ns / r.per_core_avg_ns) if r.per_core_max_ns and r.per_core_avg_ns else None
    return {
        "model": res.model, "regime": res.regime, "layer": res.layer, "GLOBAL CALL COUNT": r.global_call_count,
        "op_code": r.op_code, "S": c.get("S", c.get("N", "")), "kv_len": c.get("kv_seq", c.get("attended", "")),
        "cur_pos": c.get("cur_pos", ""), "q_chunk": c.get("q_chunk", ""), "k_chunk": c.get("k_chunk", c.get("K", "")),
        "num_cores": c.get("num_cores", ""), "fidelity": c.get("fidelity", ""),
        "exp_approx": c.get("exp_approx_mode", ""), "fp32_acc": "" if fp32 is None else fp32,
        "kernel_path": ("" if fp32 is None else ("fp32_legacy" if fp32 else "streaming")),
        "input_dtype": c.get("input_dtype", ""), "kv_dtype": c.get("kv_input_dtype", ""),
        "meas_ns": _fmt(r.meas_ns, 0), "pred_kernel_ns": _fmt(res.pred_kernel_ns, 0),
        "pred_device_ns": _fmt(res.pred_device_ns, 0), "err_pct": _fmt(100.0 * res.err, 2),
        "per_core_max_over_avg": _fmt(ratio, 3), "pm_ideal_ns": _fmt(r.pm_ideal_ns, 0), "replays": r.replays,
        "low_confidence_reasons": ",".join(res.reasons), "defaulted_attrs": res.defaulted,
        "config_mismatch": res.config_mismatch, "excluded": int(res.excluded), "exclude_reason": res.exclude_reason,
    }


SUMMARY_COLUMNS = ["model", "regime", "n", "n_excluded", "n_low_confidence", "mean_err_pct", "tw_mean_err_pct",
                   "std_err_pct", "frac_ops_within_10", "frac_time_within_10", "mean_ci95_low_pct",
                   "mean_ci95_high_pct", "tw_mean_ci95_low_pct", "tw_mean_ci95_high_pct", "pass"]


def summary_record(s: RegimeSummary, model: str = "") -> Dict[str, object]:
    return {"model": model, "regime": s.regime, "n": s.n, "n_excluded": s.n_excluded,
            "n_low_confidence": s.n_low_confidence, "mean_err_pct": _fmt(100 * s.mean_err, 2),
            "tw_mean_err_pct": _fmt(100 * s.tw_mean_err, 2), "std_err_pct": _fmt(100 * s.std_err, 2),
            "frac_ops_within_10": _fmt(s.frac_ops_within, 3), "frac_time_within_10": _fmt(s.frac_time_within, 3),
            "mean_ci95_low_pct": _fmt(100 * s.mean_ci[0], 2), "mean_ci95_high_pct": _fmt(100 * s.mean_ci[1], 2),
            "tw_mean_ci95_low_pct": _fmt(100 * s.tw_mean_ci[0], 2),
            "tw_mean_ci95_high_pct": _fmt(100 * s.tw_mean_ci[1], 2), "pass": _verdict(s)}


def _verdict(s: RegimeSummary) -> str:
    if s.n == 0:
        return "NO DATA"
    return "PASS" if s.passed else "FAIL"


def write_csv(path, columns: Sequence[str], records: Iterable[Dict[str, object]], comments: Sequence[str] = ()):
    with open(path, "w", newline="") as f:
        for c in comments:
            f.write(c if c.startswith("#") else f"# {c}")
            f.write("\n")
        w = csv.DictWriter(f, fieldnames=list(columns))
        w.writeheader()
        for rec in records:
            w.writerow(rec)


def plot_signed_errors(results: Sequence[RowResult], path, *, title: str = "", tolerance=TOLERANCE):
    """Signed error per op in order of appearance, one colour per regime (README palette).
    Excluded rows are drawn hollow so they stay visible without counting."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    regimes: List[str] = []
    for res in results:
        if res.regime not in regimes:
            regimes.append(res.regime)
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=150, facecolor=BG)
    ax.set_facecolor(BG)
    ax.axhspan(-100 * tolerance, 100 * tolerance, color=INK, alpha=0.06, lw=0)
    ax.axhline(0.0, color=INK, lw=0.8)
    for i, reg in enumerate(regimes):
        color = PALETTE[i % len(PALETTE)]
        xs = [k for k, res in enumerate(results) if res.regime == reg and not res.excluded]
        ys = [100 * res.err for res in results if res.regime == reg and not res.excluded]
        ax.scatter(xs, ys, s=22, color=color, label=f"{reg} (n={len(xs)})", zorder=3)
        xe = [k for k, res in enumerate(results) if res.regime == reg and res.excluded and not math.isnan(res.err)]
        ye = [100 * res.err for res in results if res.regime == reg and res.excluded and not math.isnan(res.err)]
        if xe:
            ax.scatter(xe, ye, s=22, facecolors="none", edgecolors=color, lw=1.0, zorder=3)
    ax.set_xlabel("op (order of appearance)", color=INK)
    ax.set_ylabel("signed error, (pred - meas) / meas [%]", color=INK)
    if title:
        ax.set_title(title, color=INK, fontsize=10)
    for sp in ax.spines.values():
        sp.set_color(INK)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors=INK)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=BG)
    plt.close(fig)
    return path


# Wall term order and colours for the stacked component chart (README palette plus ink shades).
WALL_TERM_STYLE = [
    ("compute_floor", PALETTE[0], 1.0), ("reader_wait", PALETTE[1], 1.0), ("fe_issue", PALETTE[2], 1.0),
    ("dest_roundtrip", PALETTE[2], 0.55), ("control", PALETTE[3], 1.0), ("mask_bracket", PALETTE[3], 0.5),
    ("sfpu_issue", PALETTE[1], 0.5), ("kv_stream_wait", PALETTE[1], 0.8), ("straggler", INK, 0.3), ("init", INK, 0.7),
]


def plot_wall_components(entries: Sequence[Tuple[str, Dict[str, float], Optional[Dict[str, float]]]], path, *,
                         title: str = "", unit: float = 1e6, unit_label: str = "Mcycles"):
    """Stacked bar of the named wall terms per config: the model bar, and next to it the measured
    zone parts under the same names when given (None draws the model bar alone). Terms missing from
    a dict are zero; every dict's sum is its wall."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    n = len(entries)
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(7.0, 0.85 * n + 2.0), 4.8), dpi=150, facecolor=BG)
    ax.set_facecolor(BG)
    used = []
    for i, (label, model, measured) in enumerate(entries):
        bars = [(i - width / 2, model, False)] if measured is not None else [(i, model, False)]
        if measured is not None:
            bars.append((i + width / 2, measured, True))
        for x, comp, hatched in bars:
            bottom = 0.0
            for name, color, alpha in WALL_TERM_STYLE:
                v = float(comp.get(name, 0.0)) / unit
                if v <= 0.0:
                    continue
                ax.bar(x, v, width, bottom=bottom, color=color, alpha=alpha, edgecolor=BG, lw=0.4,
                       hatch=("//" if hatched else None))
                bottom += v
                if name not in used:
                    used.append(name)
            ax.text(x, bottom, f"{bottom:.2f}", ha="center", va="bottom", fontsize=6, color=INK)
    ax.set_xticks(range(n))
    ax.set_xticklabels([e[0] for e in entries], rotation=35, ha="right", fontsize=7, color=INK)
    ax.set_ylabel(f"wall [{unit_label}]", color=INK)
    if title:
        ax.set_title(title, color=INK, fontsize=10)
    handles = [Patch(facecolor=c, alpha=a, label=nm) for nm, c, a in WALL_TERM_STYLE if nm in used]
    if any(e[2] is not None for e in entries):
        handles.append(Patch(facecolor=BG, edgecolor=INK, hatch="//", label="measured zone parts (hatched)"))
    ax.legend(handles=handles, frameon=False, fontsize=7, labelcolor=INK, ncol=2)
    for sp in ax.spines.values():
        sp.set_color(INK)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=BG)
    plt.close(fig)
    return path


# ----------------------------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------------------------

@dataclass
class RunResult:
    results: List[RowResult]
    summaries: List[RegimeSummary]
    comments: List[str]
    paths: Dict[str, Path] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        scored = [s for s in self.summaries if s.n > 0]
        return bool(scored) and all(s.passed for s in scored)


def validate(csv_path, *, model: str = "", positions_csv=None, device=None, resamples=BOOTSTRAP_RESAMPLES,
             seed=BOOTSTRAP_SEED) -> RunResult:
    """Read the ops CSV, price every SDPA / TopK row and summarise per regime."""
    comments, raw = read_ops_csv(csv_path)
    rows = aggregate_replays(select_op_rows(raw))
    sidecar = read_positions_sidecar(positions_csv)
    model = model or Path(csv_path).stem
    results: List[RowResult] = []
    layer_ctr: Dict[str, int] = {}
    for r in rows:
        res = predict_row(r, device=device, sidecar=sidecar.get(r.global_call_count))
        res.model = model
        layer_ctr[res.regime] = layer_ctr.get(res.regime, 0) + 1
        res.layer = layer_ctr[res.regime]
        results.append(res)
    return RunResult(results=results, summaries=summarize(results, resamples=resamples, seed=seed),
                     comments=comments)


def write_outputs(run: RunResult, out_dir, *, model: str, provenance: Sequence[str] = (), figure=True) -> Dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    comments = list(run.comments) + [f"# PROVENANCE: {p}" for p in provenance]
    paths = {"rows": out_dir / f"{model}_signed_errors.csv", "summary": out_dir / f"{model}_summary.csv"}
    write_csv(paths["rows"], ROW_COLUMNS, (row_record(x) for x in run.results), comments)
    write_csv(paths["summary"], SUMMARY_COLUMNS, (summary_record(s, model) for s in run.summaries), comments)
    if figure:
        paths["figure"] = plot_signed_errors(run.results, out_dir / f"{model}_signed_errors.png",
                                             title=f"{model}: roofline signed error per op")
    run.paths = paths
    return paths


def format_summary(run: RunResult) -> str:
    lines = [f"{'regime':20} {'n':>4} {'excl':>4} {'mean%':>7} {'tw%':>7} {'std%':>6} {'ops<10':>7} "
             f"{'time<10':>8} {'tw CI95%':>18} result"]
    for s in run.summaries:
        ci = f"[{100 * s.tw_mean_ci[0]:+.1f}, {100 * s.tw_mean_ci[1]:+.1f}]" if s.n else ""
        lines.append(f"{s.regime:20} {s.n:>4} {s.n_excluded:>4} {_fmt(100 * s.mean_err):>7} "
                     f"{_fmt(100 * s.tw_mean_err):>7} {_fmt(100 * s.std_err):>6} {_fmt(s.frac_ops_within, 3):>7} "
                     f"{_fmt(s.frac_time_within, 3):>8} {ci:>18} {_verdict(s)}")
    lines.append(f"overall: {'PASS' if run.passed else 'FAIL'}")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="SDPA/TopK roofline validation against a tracy ops_perf_results CSV")
    p.add_argument("csv", help="ops_perf_results_*.csv from tools/tracy/process_ops_logs.py")
    p.add_argument("-o", "--out-dir", default=None, help="output directory (default: next to the CSV)")
    p.add_argument("--model", default=None, help="model label and output file stem (default: CSV stem)")
    p.add_argument("--positions", default=None, help="sidecar CSV with GLOBAL CALL COUNT,cur_pos[,page_block_size]")
    p.add_argument("--archspec", default=None, help="architecture yaml (default config/tt_bh.yaml)")
    p.add_argument("--device", default="p100a", help="package instance name in the archspec")
    p.add_argument("--no-device", action="store_true", help="skip the Device.execute_op projection column")
    p.add_argument("--no-figure", action="store_true")
    p.add_argument("--provenance", action="append", default=[], help="PROVENANCE line(s) for the outputs")
    p.add_argument("--resamples", type=int, default=BOOTSTRAP_RESAMPLES)
    p.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    args = p.parse_args(argv)

    device = None if args.no_device else make_device(args.archspec, args.device)
    model = args.model or Path(args.csv).stem
    run = validate(args.csv, model=model, positions_csv=args.positions, device=device,
                   resamples=args.resamples, seed=args.seed)
    paths = write_outputs(run, args.out_dir or Path(args.csv).parent, model=model, provenance=args.provenance,
                          figure=not args.no_figure)
    print(format_summary(run))
    for k, v in paths.items():
        print(f"{k}: {v}")
    return 0 if run.passed else 1


if __name__ == "__main__":
    sys.exit(main())
