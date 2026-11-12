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


def _with_grid(grid):
    """p100a's instance with only compute_grid_size replaced, validated as it would be on load."""
    pkg = _pkg("config/tt_bh.yaml", "p100a")
    data = pkg.model_dump()
    data["compute_grid_size"] = grid
    return type(pkg).model_validate(data)


@pytest.mark.unit
@pytest.mark.parametrize("grid", [
    [11],            # one value: used to fall back silently to num_units (120)
    [11, 10, 1],     # three values: same silent fallback
    [0, 10],         # zero: used to give 0 cores, then ZeroDivisionError in Device.execute_op
    [-11, 10],       # negative: used to give -110 cores
    [11, 0],
])
def test_malformed_compute_grid_size_is_rejected(grid):
    with pytest.raises(ValueError, match="compute_grid_size must be two positive integers"):
        _with_grid(grid)


@pytest.mark.unit
def test_valid_compute_grid_size_is_accepted():
    assert _with_grid([11, 10]).compute_units() == 110
    assert _with_grid(None).compute_units() == _with_grid(None).get_ipgroup('compute').num_units


# ---- one core count per instance: the LUT core count follows compute_units() ----
@pytest.mark.unit
@pytest.mark.parametrize("yaml_path, name, cores, num_units", [
    ("config/tt_wh.yaml", "n150", 64, 72),
    ("config/tt_bh.yaml", "p100a", 110, 120),
])
def test_lut_core_count_follows_compute_grid(yaml_path, name, cores, num_units):
    from tools.perf_lookup.lookup_operator_perf import resolve_operator_lookup_core_count

    pkg = _pkg(yaml_path, name)
    assert pkg.get_ipgroup('compute').num_units == num_units      # still declared, no longer used
    assert resolve_operator_lookup_core_count(pkg, pkg) == cores == pkg.compute_units()


@pytest.mark.unit
def test_lut_core_count_without_grid_is_num_units():
    from tools.perf_lookup.lookup_operator_perf import resolve_operator_lookup_core_count

    pkg = _pkg("config/tt_wh.yaml", "n300")
    assert resolve_operator_lookup_core_count(pkg, pkg) == pkg.get_ipgroup('compute').num_units


@pytest.mark.unit
def test_explicit_lut_core_count_still_wins():
    from tools.perf_lookup.lookup_operator_perf import resolve_operator_lookup_core_count

    pkg = _pkg("config/tt_bh.yaml", "p100a")
    data = pkg.model_dump()
    data["operator_lookup_core_count"] = 96
    assert resolve_operator_lookup_core_count(None, type(pkg).model_validate(data)) == 96


@pytest.mark.unit
def test_device_dump_shows_effective_core_count():
    from ttsim.back.device import Device

    # LUT off: the dump does not need it, and loading it would make this test depend on the
    # LFC cache / network (and emit the LUT schema-version warning).
    pkg = _pkg("config/tt_bh.yaml", "p100a")
    data = pkg.model_dump()
    data["ignore_lut"] = True
    text = str(Device(type(pkg).model_validate(data)))
    assert "num_units   : 120" in text
    assert "compute_units: 110" in text
    assert "lut_core_count: 110" in text
