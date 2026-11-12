# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Analytical compute throughput follows ``compute_grid_size`` (x * y): WH n150 8x8 = 64, BH p100a 11x10 = 110."""

import pytest

from ttsim.config import get_arspec_from_yaml

# matrix-pipe fp16 MAC/cycle per core = 512 (peak_ipc(...) / num_units for any arch below)
PER_CORE = 512


def _pkg(yaml_path, name):
    _, packages = get_arspec_from_yaml(yaml_path)
    return packages[name]


@pytest.mark.unit
@pytest.mark.parametrize("yaml_path, name, cores", [("config/tt_wh.yaml", "n150", 64), ("config/tt_bh.yaml", "p100a", 110)])
def test_peak_ipc_uses_compute_grid_size(yaml_path, name, cores):
    pkg = _pkg(yaml_path, name)
    assert pkg.compute_units() == cores
    assert pkg.peak_ipc('matrix', 'mac', 'fp16') == cores * PER_CORE


@pytest.mark.unit
def test_instance_without_grid_falls_back_to_num_units():
    pkg = _pkg("config/tt_wh.yaml", "n300")           # declares no compute_grid_size
    assert pkg.compute_grid_size is None
    assert pkg.compute_units() == pkg.get_ipgroup('compute').num_units
