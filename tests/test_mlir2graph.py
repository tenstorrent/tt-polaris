#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the MLIR -> Polaris frontend (ttsim/front/mlir/mlir2nx.py)."""
import pytest
from ttsim.front.mlir.mlir2nx import parse_mlir, mlir2graph

EXAMPLES = "workloads/mlir/examples"


@pytest.mark.unit
def test_parse_toy_matmul_add():
    """Toy MLIR (MatMul + Add) parses to the expected ops and shapes."""
    fg = parse_mlir(f"{EXAMPLES}/toy_mm_add.mlir")
    assert [op.optype for op in fg.ops] == ["MatMul", "Add"]
    shapes = {t.name: t.shape for t in fg.tensors.values()}
    assert shapes["X"] == [4, 8]
    assert shapes["W"] == [8, 16]
    assert shapes["Y"] == [4, 16]


@pytest.mark.unit
def test_mlir2graph_builds_workloadgraph():
    """mlir2graph produces a WorkloadGraph with the right node count."""
    gg = mlir2graph("toy", f"{EXAMPLES}/toy_mm_add.mlir")
    assert gg is not None
    assert gg.get_node_count() == 2   # MatMul + Add


@pytest.mark.unit
def test_real_forge_ttir_folds_to_clean_mlp():
    """Real Forge TTIR (dot_general/maximum + broadcast/reshape/constant plumbing)
    folds into a clean MatMul -> Add -> Relu -> MatMul -> Add graph."""
    fg = parse_mlir(f"{EXAMPLES}/mlp_forge_real.mlir")
    assert [op.optype for op in fg.ops] == ["MatMul", "Add", "Relu", "MatMul", "Add"]


@pytest.mark.unit
def test_shape_inference_runs_on_real_ttir():
    """Shape inference runs on the real-TTIR graph and resolves output shapes."""
    gg = mlir2graph("mlp", f"{EXAMPLES}/mlp_forge_real.mlir")
    for _, op in gg._ops.items():
        it = [gg._tensors[x] for x in op.inList]
        ot = [gg._tensors[x] for x in op.outList]
        op.get_perf_counts(it, ot)
        for t in ot:
            assert t.shape is not None
