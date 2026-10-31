# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""resnet50 mode selection follows ttnn AVAILABILITY, not a box label.

The old rule was ``IS_POLARIS = os.getenv('IRD_ARCH_NAME', '') == ''``. That asks
whether the box is *labelled* hardware, which is not the same as whether hardware
is *available*, and the gap is a real failure: on an IRD box with the variable
unset it picked the Polaris branch and died on ``import ttsim`` — a package that
does not exist in a tt-metal checkout. The row marked ``<-- the bug`` below is
that case.
"""

import pytest

from workloads.ttnn.resnet50.common import ttnn_functional_resnet50 as runtime_mode


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv('IRD_ARCH_NAME', raising=False)
    monkeypatch.delenv('TTSIM_FORCE_POLARIS', raising=False)


def _fake_env(monkeypatch, *, ttnn: bool, ttsim: bool):
    monkeypatch.setattr(runtime_mode, 'real_ttnn_available', lambda: ttnn)
    monkeypatch.setattr(runtime_mode, 'polaris_checkout', lambda: ttsim)


# (ird_arch_name, force, ttnn present, ttsim present, expected is_polaris)
CASES = [
    # polaris checkout: ttsim present. Must ALWAYS pick the shim, with or
    # without a stray ttnn in the environment -- a conda env carrying a ttnn
    # package but no torch is what broke this the first time round.
    (None,        None,  False, True,  True),
    (None,        None,  True,  True,  True),   # <-- the Mac regression
    # IRD: no ttsim in a tt-metal checkout.
    (None,        None,  True,  False, False),  # variable unset -> still HW
    (None,        None,  False, False, True),   # neither -> shim, guard explains
    # explicit markers beat autodetect
    ('blackhole', None,  True,  True,  False),  # label wins over ttsim
    (None,        '1',   True,  False, True),   # force wins over ttnn
    ('blackhole', '1',   True,  False, True),   # force beats the label
    (None,        '0',   False, True,  False),  # force-off beats ttsim
    (None,        '',    False, True,  False),  # empty string is falsey, not "unset"
]


@pytest.mark.unit
@pytest.mark.parametrize('ird,force,have_ttnn,have_ttsim,expected', CASES)
def test_mode_resolution(monkeypatch, ird, force, have_ttnn, have_ttsim, expected) -> None:
    if ird is not None:
        monkeypatch.setenv('IRD_ARCH_NAME', ird)
    if force is not None:
        monkeypatch.setenv('TTSIM_FORCE_POLARIS', force)
    _fake_env(monkeypatch, ttnn=have_ttnn, ttsim=have_ttsim)
    assert runtime_mode.is_polaris() is expected


@pytest.mark.unit
def test_stray_ttnn_in_a_polaris_env_does_not_select_hardware(monkeypatch) -> None:
    """The regression that broke `python polaris.py ... resnet50_bh_b16_device_perf`.

    A dev environment had an importable `ttnn` with no `torch` behind it. Keying
    the decision on `ttnn` sent a polaris run down the hardware branch, straight
    into `ModuleNotFoundError: No module named 'torch'`. `ttsim` decides now.
    """
    _fake_env(monkeypatch, ttnn=True, ttsim=True)
    assert runtime_mode.is_polaris() is True


@pytest.mark.unit
def test_ird_without_env_var_selects_hardware(monkeypatch) -> None:
    """The regression this module exists for, stated on its own."""
    _fake_env(monkeypatch, ttnn=True, ttsim=False)   # an IRD box: ttnn yes, ttsim no
    assert 'IRD_ARCH_NAME' not in __import__('os').environ
    assert runtime_mode.is_polaris() is False, (
        'An IRD box with IRD_ARCH_NAME unset must run on silicon, not fall into '
        'the Polaris branch and fail on `import ttsim`.'
    )


@pytest.mark.unit
def test_polaris_box_is_unaffected(monkeypatch) -> None:
    """No top-level ttnn -> shim, exactly as before. The common case must not move."""
    _fake_env(monkeypatch, ttnn=False, ttsim=True)
    assert runtime_mode.is_polaris() is True


@pytest.mark.unit
@pytest.mark.parametrize('have_ttnn', [True, False])
def test_description_states_the_reason(monkeypatch, have_ttnn) -> None:
    """Every past confusion here was someone unable to tell which branch they got."""
    _fake_env(monkeypatch, ttnn=have_ttnn, ttsim=not have_ttnn)
    desc = runtime_mode.mode_description()
    assert ('HARDWARE' if have_ttnn else 'POLARIS') in desc
    assert '—' in desc, 'description must give the reason, not just the mode'


@pytest.mark.unit
def test_real_ttnn_available_is_honest_here() -> None:
    """Unmocked: a polaris checkout is detectable by ttsim being importable.

    The vendored tt-metal/ttnn is not on sys.path, and the shim is
    ttsim.front.ttnn, so this is the signal the resolver relies on.
    """
    assert runtime_mode.polaris_checkout() is True
