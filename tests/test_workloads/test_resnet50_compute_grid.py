# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""The resnet50 workload compute grids equal the arch YAML's ``compute_grid_size``.

The Polaris-mode resnet50 infras set the shim device's grid from a module constant
(``BH_P100A_COMPUTE_GRID`` / ``WH_N150_COMPUTE_GRID``) instead of reading the loaded arch spec. The
value feeds the fold shard shape and, on Blackhole, the fold core-count search, while the device cost
model reads ``compute_grid_size`` from the YAML. The two drifted before (12x10 / 8x9 against 11x10 /
8x8); this test fails as soon as either side changes alone.
"""

import pytest

from ttsim.config import get_arspec_from_yaml


def _grid(yaml_path, name):
    _, packages = get_arspec_from_yaml(yaml_path)
    return tuple(packages[name].compute_grid_size)


@pytest.mark.unit
def test_bh_p100a_workload_grid_matches_arch_yaml():
    from workloads.ttnn.resnet50.bh.resnet50_test_infra_polaris_bh import BH_P100A_COMPUTE_GRID

    assert tuple(BH_P100A_COMPUTE_GRID) == _grid("config/tt_bh.yaml", "p100a") == (11, 10)


@pytest.mark.unit
def test_wh_n150_workload_grid_matches_arch_yaml():
    from workloads.ttnn.resnet50.wh.resnet50_test_infra_polaris_wh import WH_N150_COMPUTE_GRID

    assert tuple(WH_N150_COMPUTE_GRID) == _grid("config/tt_wh.yaml", "n150") == (8, 8)
