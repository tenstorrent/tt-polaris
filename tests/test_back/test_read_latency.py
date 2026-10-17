#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Validate the O2O read-latency model against the Blackhole DRAM-interleaved
read microbenchmark.

Ground-truth rows are taken verbatim from tt-metal's
``tests/.../data_movement/data/blackhole/DRAM Interleaved Page Read Numbers.csv``
(columns: transaction size N (bytes), number of transactions Q, measured
latency in cycles). Embedded here so the test is hermetic.
"""

import math

import pytest

from ttsim.back.device import TILE_ELEMS

# (N bytes, Q transactions, measured latency cycles) - Blackhole, riscv_1.
CSV_ROWS = [
    (64, 1, 472.0), (128, 1, 482.0), (256, 1, 553.0), (512, 1, 498.0),
    (1024, 1, 529.0), (2048, 1, 603.0), (4096, 1, 653.0), (8192, 1, 825.0),
    (16384, 1, 1169.0),
    (64, 4, 559.0), (128, 4, 567.0), (256, 4, 577.0), (512, 4, 584.0),
    (1024, 4, 616.0), (2048, 4, 681.0), (4096, 4, 806.0), (8192, 4, 1074.0),
    (16384, 4, 1601.0),
    (64, 16, 1055.0), (128, 16, 1052.0), (256, 16, 1165.0), (512, 16, 1074.0),
    (1024, 16, 1108.0), (2048, 16, 1190.0), (4096, 16, 1639.0),
    (8192, 16, 2676.0), (16384, 16, 4795.0),
    (64, 64, 2871.0), (128, 64, 2880.0), (256, 64, 2897.0), (512, 64, 2893.0),
    (1024, 64, 2930.0), (2048, 64, 3035.0), (4096, 64, 4802.0),
    (8192, 64, 9029.0), (16384, 64, 17436.0),
    (64, 256, 10209.0), (128, 256, 10160.0), (256, 256, 10180.0),
    (512, 256, 10186.0), (1024, 256, 10216.0), (2048, 256, 10527.0),
    (4096, 256, 17892.0), (8192, 256, 34917.0), (16384, 256, 67609.0),
]

# Interleaved microbenchmark uses all 7 p100a DRAM channels.
NUM_CHANNELS = 7
REL_TOL = 0.15


@pytest.mark.parametrize("N,Q,measured", CSV_ROWS)
def test_predicted_latency_matches_microbenchmark(N, Q, measured, bh_read_latency_cfg):
    predicted = bh_read_latency_cfg.predict_read_latency(N, Q, num_channels=NUM_CHANNELS)
    rel_err = abs(predicted - measured) / measured
    assert rel_err <= REL_TOL, (
        f"N={N} Q={Q}: predicted={predicted:.1f} measured={measured} "
        f"rel_err={rel_err:.1%} > {REL_TOL:.0%}"
    )


def test_mean_error_is_small(bh_read_latency_cfg):
    errs = [
        abs(bh_read_latency_cfg.predict_read_latency(N, Q, num_channels=NUM_CHANNELS) - m) / m
        for N, Q, m in CSV_ROWS
    ]
    mean_err = sum(errs) / len(errs)
    assert mean_err <= 0.05, f"mean rel err {mean_err:.1%} too high"


def _base(cfg, hops=None):
    hops = cfg.default_hops if hops is None else hops
    return cfg.tdram_cyc + 2.0 * hops * cfg.chop_cyc_per_hop


def test_issue_arm_closed_form(bh_read_latency_cfg):
    # N=64, Q=16 is deep in the issue regime (transport arm is ~570 vs ~1060), so the
    # prediction must equal the issue arm exactly. Pins the Tdetect barrier tail into
    # this arm and the delta_issue * Q slope: the 45-row gates above tolerate dropping
    # Tdetect entirely, so assert the structure here rather than only the fit.
    cfg = bh_read_latency_cfg
    N, Q = 64, 16
    expected = cfg.delta_issue_cyc * Q + _base(cfg) + cfg.tdetect_cyc + N / cfg.b_channel_bpc
    got = cfg.predict_read_latency(N, Q, num_channels=NUM_CHANNELS)
    assert math.isclose(got, expected, rel_tol=1e-12)


def test_transport_arm_closed_form(bh_read_latency_cfg):
    # N=16 KB, Q=3 is streaming-bound (transport ~1346 vs issue ~1246). Q=3 is chosen
    # to sit exactly at n* = ceil(62/24) = 3, so this also pins ceil (floor would give
    # n*=2 and drop one delta_issue) and the absence of Tdetect from this arm.
    cfg = bh_read_latency_cfg
    N, Q = 16384, 3
    n_star = math.ceil(cfg.noc_inbound_bpc / cfg.b_channel_bpc)
    assert n_star == Q, "test point no longer sits at n*; pick a new Q"
    b_eff = min(min(Q, NUM_CHANNELS) * cfg.b_channel_bpc, cfg.noc_inbound_bpc)
    expected = Q * cfg.delta_issue_cyc + _base(cfg) + (Q * N) / b_eff
    got = cfg.predict_read_latency(N, Q, num_channels=NUM_CHANNELS)
    assert math.isclose(got, expected, rel_tol=1e-12)
    # Tdetect is intentionally omitted here; if it leaked in, `got` would be larger.
    assert got < expected + cfg.tdetect_cyc


def test_channel_count_only_matters_below_n_star(bh_read_latency_cfg):
    # b_eff is capped at noc_inbound_bpc, so any channel count >= n* = 3 saturates the
    # inbound link and yields identical predictions. Documents why p100a (7 channels)
    # and p150a (8) score the same, and pins that num_channels is nonetheless wired
    # through: a 1-channel part is strictly slower once the queue is deep.
    cfg = bh_read_latency_cfg
    n_star = math.ceil(cfg.noc_inbound_bpc / cfg.b_channel_bpc)
    for N in (64, 2048, 16384):
        for Q in (1, 4, 64, 256):
            at_n_star = cfg.predict_read_latency(N, Q, num_channels=n_star)
            for channels in (n_star + 1, 7, 8, 64):
                assert math.isclose(
                    cfg.predict_read_latency(N, Q, num_channels=channels),
                    at_n_star, rel_tol=1e-12,
                ), f"N={N} Q={Q}: {channels} channels differs from {n_star}"
    single = cfg.predict_read_latency(16384, 64, num_channels=1)
    many = cfg.predict_read_latency(16384, 64, num_channels=n_star)
    assert single > many


def test_latency_monotonic_in_size_and_queue(bh_read_latency_cfg):
    # Larger transactions and deeper queues never reduce latency.
    for Q in (1, 4, 16, 64, 256):
        prev = 0.0
        for N in (64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384):
            cur = bh_read_latency_cfg.predict_read_latency(N, Q, num_channels=NUM_CHANNELS)
            assert cur >= prev - 1e-9
            prev = cur
    for N in (64, 4096, 16384):
        prev = 0.0
        for Q in (1, 4, 16, 64, 256):
            cur = bh_read_latency_cfg.predict_read_latency(N, Q, num_channels=NUM_CHANNELS)
            assert cur >= prev - 1e-9
            prev = cur


def test_effective_bandwidth_saturates_near_noc_ceiling(bh_read_latency_cfg):
    cfg = bh_read_latency_cfg
    # Deep queue + large pages should approach the NoC inbound ceiling.
    tlat = cfg.predict_read_latency(16384, 256, num_channels=NUM_CHANNELS)
    bw = (256 * 16384) / tlat
    assert 0.85 * cfg.noc_inbound_bpc <= bw <= cfg.noc_inbound_bpc + 1.0


def test_small_transactions_are_latency_bound(bh_read_latency_cfg):
    # The premise of the model: for small N the latency is dominated by the fixed
    # head and per-read issue cost, not by the bytes moved. Latency per byte is
    # therefore orders of magnitude worse than the streaming rate.
    cfg = bh_read_latency_cfg
    small_bw = (64 * 4) / cfg.predict_read_latency(64, 4, num_channels=NUM_CHANNELS)
    large_bw = (16384 * 256) / cfg.predict_read_latency(16384, 256, num_channels=NUM_CHANNELS)
    assert small_bw < 0.05 * large_bw


def test_tile_page_size_default():
    assert TILE_ELEMS == 1024


def test_zero_bytes_is_zero_latency(bh_read_latency_cfg):
    assert bh_read_latency_cfg.predict_read_latency(0, 4, num_channels=NUM_CHANNELS) == 0.0


@pytest.mark.parametrize("field,bad", [
    ('tdram_cyc', 0.0), ('tdram_cyc', -1.0),
    ('tdetect_cyc', -1.0),
    # Divisors: a mistyped 0 would otherwise surface as a bare ZeroDivisionError,
    # and a negative rate as a silently negative latency.
    ('noc_inbound_bpc', 0.0), ('b_channel_bpc', 0.0), ('b_channel_bpc', -24.0),
    ('delta_issue_cyc', 0.0), ('chop_cyc_per_hop', 0.0),
    ('default_hops', -1), ('fclk_mhz', 0.0),
])
def test_calibration_rejects_nonpositive_constants(field, bad):
    from pydantic import ValidationError

    from ttsim.config.simconfig import MemoryReadLatencyModel

    good = dict(
        tdram_cyc=375.0, tdetect_cyc=10.0, noc_inbound_bpc=62.0, delta_issue_cyc=38.0,
        chop_cyc_per_hop=8.0, b_channel_bpc=24.0, default_hops=4,
    )
    assert MemoryReadLatencyModel(**good) is not None
    with pytest.raises(ValidationError, match=field):
        MemoryReadLatencyModel(**{**good, field: bad})
