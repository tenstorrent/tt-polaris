#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the --no-op-info-cache option of tools/si_profiling_helpers/run-ttnn-profiler.py.

Without the option the tracy command line must stay exactly as before, so captures of every other
workload are unchanged. With it, every pass (raw, perf, trace) must hand --no-op-info-cache to tracy:
tracy deletes TT_METAL_PROFILER_NO_CACHE_OP_INFO from the environment unless it gets that flag.
"""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / 'tools' / 'si_profiling_helpers' / 'run-ttnn-profiler.py'
_spec = importlib.util.spec_from_file_location('run_ttnn_profiler', _SCRIPT)
assert _spec is not None and _spec.loader is not None
rtp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rtp)

FLAG = '--no-op-info-cache'
# (collect_noc_traces, collect_perf_counters) for the raw, perf and trace passes.
PASSES = {'raw': (False, False), 'perf': (False, True), 'trace': (True, False)}


class _QuietLogger:
    """Stands in for loguru's logger so main() cannot remove the test session's handlers."""

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


@pytest.fixture
def captured_argv(monkeypatch):
    """Run run_profiler without launching anything; return the tracy argv of each call."""
    calls = []

    def fake_run_and_capture(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(os, 'environ', dict(os.environ))  # run_profiler sets TT_METAL_* variables
    monkeypatch.setattr(rtp, 'setup_environment', lambda *args, **kwargs: None)
    monkeypatch.setattr(rtp, 'run_and_capture', fake_run_and_capture)
    return calls


def _run(pass_name, **kwargs):
    noc, perf = PASSES[pass_name]
    rtp.run_profiler('a_test.py -k perf', '/out', True, 'report',
                     collect_noc_traces=noc, collect_perf_counters=perf, **kwargs)


@pytest.mark.unit
@pytest.mark.parametrize('pass_name', PASSES)
def test_flag_absent_by_default(captured_argv, pass_name):
    _run(pass_name)
    assert FLAG not in captured_argv[0]


@pytest.mark.unit
@pytest.mark.parametrize('pass_name', PASSES)
def test_flag_passed_to_tracy_when_requested(captured_argv, pass_name):
    _run(pass_name, no_op_info_cache=True)
    argv = captured_argv[0]
    assert argv.count(FLAG) == 1
    # It must come before the wrapped command, or pytest rather than tracy would receive it.
    assert argv.index(FLAG) < argv.index('pytest')


@pytest.mark.unit
def test_default_raw_argv_unchanged(captured_argv):
    _run('raw')
    assert captured_argv[0] == [sys.executable, '-m', 'tracy', '-p', '-r', '-v', '--check-exit-code',
                                '--op-support-count=100000', '-o', '/out',
                                '-m', 'pytest', 'a_test.py', '-k', 'perf']


@pytest.mark.unit
@pytest.mark.parametrize('cli_flag, expected', [([], False), ([FLAG], True)])
def test_cli_option_reaches_every_pass(monkeypatch, tmp_path, cli_flag, expected):
    seen = []

    def fake_run_profiler(command, output_dir, pytest_mode, report_name, **kwargs):
        seen.append((os.path.basename(output_dir), kwargs['no_op_info_cache']))
        return None

    monkeypatch.setattr(rtp, 'logger', _QuietLogger())
    monkeypatch.setattr(rtp, 'run_profiler', fake_run_profiler)
    monkeypatch.setattr(sys, 'argv', ['run-ttnn-profiler.py', '--dryrun', '--command', 'a_test.py',
                                      '--pytest', '--output-dir', str(tmp_path / 'out'),
                                      '--report-name', 'r', *cli_flag])
    assert rtp.main() == 0
    assert seen == [('raw', expected), ('perf', expected), ('trace', expected)]
