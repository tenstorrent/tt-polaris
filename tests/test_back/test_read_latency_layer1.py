#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Validate the unary (Layer 1) closed form against the O2O conf-page table.

Source: "O2O: Read Latency Modeling" (Confluence 2433646619), "Model Validation"
section - single core, single channel, queue depth Q=1, swept 64 B - 512 KB via
``test_bw_and_latency -m 1 -l -p <N> -i 1000``.

The unary closed form is a *different* formula from the production model in
``MemoryReadLatencyModel.predict_read_latency``: it models the >32 KB pipelined
("dual-rate") flit delivery regime, which the production two-arm model deliberately
does not. Polaris never calls it, so it lives here as a calibration reference — it
exists to confirm that the conf page's fixed-head number (438 cyc =
Tissue+Tnoc+Tdram+Tdetect) and the flit delivery rates are consistent with the
constants shipped in the arch YAML.

Rows are (N bytes, conf "Predicted" cyc, conf "Measured" cyc).
"""

import math

import pytest

# Layer 1 closed-form constants, from the conf page's "Model Validation" section.
TFIXED_UNARY_CYC = 438.0     # fixed latency to first data (Tissue+Tnoc+Tdram+Tdetect)
FLIT_BYTES = 64              # NoC flit width
RECV_CYC_PER_FLIT_LO = 2.8   # single-request receive rate, N <= 32 KB
RECV_CYC_PER_FLIT_HI = 1.83  # pipelined receive rate, N > 32 KB
RECV_KNEE_FLITS = 512        # delivery knee (32 KB / 64 B)


def predict_read_latency_unary(N: float) -> float:
    """Conf-page unary closed form::

        Tlat = Tfixed + Trecv
        Nflits = ceil(N / Wflit)
        N <= 32 KB:  Trecv = (Nflits - 1) * 2.8
        N >  32 KB:  Trecv = (512 - 1) * 2.8 + (Nflits - 512) * 1.83
    """
    if N <= 0:
        return 0.0
    nflits = math.ceil(N / FLIT_BYTES)
    if nflits <= RECV_KNEE_FLITS:
        trecv = (nflits - 1) * RECV_CYC_PER_FLIT_LO
    else:
        trecv = (
            (RECV_KNEE_FLITS - 1) * RECV_CYC_PER_FLIT_LO
            + (nflits - RECV_KNEE_FLITS) * RECV_CYC_PER_FLIT_HI
        )
    return TFIXED_UNARY_CYC + trecv

# (N, conf_predicted, conf_measured)
CONF_LAYER1 = [
    (64,        438.0,   443.0),
    (1024,      480.0,   500.0),
    (2048,      525.0,   541.0),
    (4096,      614.0,   625.0),
    (8192,      794.0,   833.0),
    (16384,    1152.0,  1153.0),
    (32768,    1869.0,  1836.0),
    (65536,    2805.0,  2771.0),
    # 128 KB: the conf "Predicted" cell reads 4714, but 438 + 511*2.8 + 1536*1.83
    # = 4679.68, which is what the formula (and the measured 4679) give. Treated
    # as a typo on the page; validated against the formula/measured value instead.
    (131072,   4680.0,  4679.0),
    (262144,   8428.0,  8347.0),
    (524288,  15923.0, 15866.0),
]

PRED_TOL_CYC = 1.0   # formula must reproduce the page's predicted column exactly
MEAS_REL_TOL = 0.06  # model vs hardware (page's own error is <= ~5%)


@pytest.mark.parametrize("N,conf_pred,conf_meas", CONF_LAYER1)
def test_unary_reproduces_conf_predicted(N, conf_pred, conf_meas):
    pred = predict_read_latency_unary(N)
    assert abs(pred - conf_pred) <= PRED_TOL_CYC, (
        f"N={N}: formula={pred:.2f} != conf predicted {conf_pred}"
    )


@pytest.mark.parametrize("N,conf_pred,conf_meas", CONF_LAYER1)
def test_unary_matches_measured(N, conf_pred, conf_meas):
    pred = predict_read_latency_unary(N)
    rel_err = abs(pred - conf_meas) / conf_meas
    assert rel_err <= MEAS_REL_TOL, (
        f"N={N}: pred={pred:.1f} measured={conf_meas} rel_err={rel_err:.1%}"
    )


def test_dual_rate_knee_at_32kb():
    # The pipelined (1.83 cyc/flit) regime only kicks in above 32 KB.
    assert predict_read_latency_unary(32768) == pytest.approx(438 + 511 * 2.8)
    # First flit past the knee uses the high rate.
    step = predict_read_latency_unary(32768 + 64) - predict_read_latency_unary(32768)
    assert step == pytest.approx(1.83, abs=1e-6)
