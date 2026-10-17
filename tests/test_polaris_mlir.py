#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""End-to-end simulation test: real Forge TTIR runs through the full Polaris flow."""
import pytest
from tests.common import reset_typespec
import polaris


@pytest.mark.unit
def test_polaris_mlir_forge_ttir(reset_typespec):
    """Real Forge TTIR (mlp_forge_real.mlir) simulates end-to-end via polaris.main."""
    rc = polaris.main([
        '--odir', '__dummy', '--study', 'mlir_test',
        '--wlspec', 'config/mlir_workloads.yaml',
        '--archspec', 'config/all_archs.yaml',
        '--wlmapspec', 'config/wl2archmapping.yaml',
        '--filterwli', 'mlp_forge_real',
        '--datatype', 'fp32',
    ])
    assert rc == 0, "Polaris should simulate the MLIR (Forge TTIR) workload successfully"
