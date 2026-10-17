#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for read-latency tests.

The read-latency HW constants (``tdram_cyc``, ``tdetect_cyc``, ``noc_inbound_bpc``)
live solely in the arch YAML (``config/tt_bh.yaml`` memory block's ``read_latency``
section). This fixture returns the parsed block itself so the tests validate the
*shipped* calibration instead of duplicating the numbers.
"""

from pathlib import Path

import pytest

from ttsim.config import get_arspec_from_yaml
from ttsim.config.simconfig import MemoryReadLatencyModel

POLARIS_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def bh_read_latency_cfg() -> MemoryReadLatencyModel:
    """Read-latency calibration for Blackhole p100a, sourced from config/tt_bh.yaml."""
    _, packages = get_arspec_from_yaml(POLARIS_ROOT / "config" / "tt_bh.yaml")
    mem = packages["p100a"].get_ipgroup("memory")
    rl = mem.ipobj.read_latency
    assert rl is not None, "config/tt_bh.yaml gddr6_bh is missing a read_latency block"
    # num_units is the DRAM channel count; it is not part of the read_latency block, so
    # tests pass it per call the same way Device does (see NUM_CHANNELS in the tests).
    assert mem.num_units == 7, "p100a DRAM channel count changed"
    return rl
