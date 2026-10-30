# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The resnet50 device-perf runners must declare the canonical operating point.

Both runners project and measure at act/weight ``bfloat8_b`` with
``MathFidelity.LoFi`` — the operating point tt-metal's own device-perf test
publishes its fps against
(``tt-metal/models/demos/vision/classification/resnet50/{wormhole,blackhole}/
tests/test_perf_device_resnet50.py:16``).  Anything else makes our number and
the reference number incomparable.

A superseded fork defaulted to ``bfloat16`` with a bare ``math_fidelity=1``.
That doubles modelled activation and weight bytes, which for a bandwidth-bound
CNN is the dominant error term — not a rounding difference, a wrong answer that
still looks like a plausible fps.  The constants are three lines in two files
and nothing else in the repository reads them back, so a silent edit would carry
all the way to a published comparison.

Text matching rather than import, because importing either runner pulls in
``ttnn`` and the tt-metal ``models.*`` packages, which are not installed here.
"""

from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_RUNNERS = {
    'wh': _REPO / 'workloads/ttnn/resnet50/wh/test_resnet50_device_perf_wh.py',
    'bh': _REPO / 'workloads/ttnn/resnet50/bh/test_resnet50_device_perf_bh.py',
}

#: The canonical device-perf operating point, as module-level constants in each
#: runner.  Matched verbatim including the ``ttnn.`` prefix so that, say,
#: ``MATH_FIDELITY = 1`` fails rather than matching loosely.
_EXPECTED = (
    ('ACT_DTYPE', 'ttnn.bfloat8_b'),
    ('WEIGHT_DTYPE', 'ttnn.bfloat8_b'),
    ('MATH_FIDELITY', 'ttnn.MathFidelity.LoFi'),
)


@pytest.mark.unit
@pytest.mark.parametrize('arch', sorted(_RUNNERS))
def test_runner_exists(arch) -> None:
    """Guards the guard: a moved or renamed runner must fail, not vacuously pass.

    ``test_dtypes_are_the_canonical_operating_point`` reads these files by path.
    Without this, a rename would turn it into a FileNotFoundError at collection
    on a good day, or — if the read were ever made tolerant — a test that
    asserts nothing.
    """
    assert _RUNNERS[arch].is_file(), f'{arch} runner not found at {_RUNNERS[arch]}'


@pytest.mark.unit
@pytest.mark.parametrize('arch', sorted(_RUNNERS))
def test_dtypes_are_the_canonical_operating_point(arch) -> None:
    """bfloat8_b / bfloat8_b / LoFi — not bfloat16, and not a bare int fidelity."""
    src = _RUNNERS[arch].read_text()
    for name, expected in _EXPECTED:
        assert f'{name} = {expected}' in src, (
            f'{arch}: expected `{name} = {expected}` — the canonical device-perf '
            'operating point from tt-metal/models/demos/vision/classification/'
            'resnet50/{wormhole,blackhole}/tests/test_perf_device_resnet50.py:16'
        )
