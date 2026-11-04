#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import subprocess
import sys
from pathlib import Path

import pytest
from tests.common import reset_typespec
import polaris

@pytest.mark.unit
def test_polaris(reset_typespec):
    assert polaris.main(['--odir', '__dummy', '--study', 'dummy', '--wlspec', 'config/mlperf_inference.yaml',
                         '--archspec', 'config/all_archs.yaml', '--wlmapspec',  'config/wl2archmapping.yaml',
                         '--dryrun']) == 0, "TT-Polaris main function should return 0 on success"


@pytest.mark.unit
def test_polaris_disable_fusion_dryrun(reset_typespec):
    assert polaris.main(
        [
            '--odir',
            '__dummy',
            '--study',
            'dummy',
            '--wlspec',
            'config/mlperf_inference.yaml',
            '--archspec',
            'config/all_archs.yaml',
            '--wlmapspec',
            'config/wl2archmapping.yaml',
            '--dryrun',
            '--disable-fusion',
        ]
    ) == 0, "TT-Polaris accepts --disable-fusion with --dryrun"


BASE_ARGS = [
    '--odir', '__dummy', '--study', 'dummy',
    '--wlspec', 'config/mlperf_inference.yaml',
    '--archspec', 'config/all_archs.yaml',
    '--wlmapspec', 'config/wl2archmapping.yaml',
    '--dryrun',
]


@pytest.mark.unit
def test_apply_filter_rejects_unmatched_entry():
    """A filter entry naming nothing raises rather than silently selecting nothing."""
    devices = [('Q1_A1', None), ('Q2_A2', None)]
    with pytest.raises(polaris.SpecSelectionError) as excinfo:
        polaris.apply_filter(devices, 'Q1_A1,NoSuchDevice', lambda x: x[0], 'filterarch')
    msg = str(excinfo.value)
    assert 'NoSuchDevice' in msg, f"error should name the offending entry, got: {msg}"
    assert '--filterarch' in msg, f"error should name the flag, got: {msg}"


@pytest.mark.unit
def test_apply_filter_accepts_valid_entries():
    """Matching stays case-insensitive and tolerates whitespace around entries."""
    devices = [('Q1_A1', None), ('Q2_A2', None)]
    assert polaris.apply_filter(devices, 'q1_a1', lambda x: x[0], 'filterarch') == [('Q1_A1', None)]
    assert polaris.apply_filter(devices, 'Q1_A1, Q2_A2', lambda x: x[0], 'filterarch') == devices
    assert polaris.apply_filter(devices, None, lambda x: x[0], 'filterarch') == devices


@pytest.mark.unit
def test_apply_filter_validates_against_full_domain():
    """An entry valid in the spec but absent from an already-narrowed list is not an error."""
    narrowed = [('Q1_A1', None)]
    domain = {'Q1_A1', 'Q2_A2'}
    assert polaris.apply_filter(narrowed, 'Q2_A2', lambda x: x[0], 'filterarch', domain) == []


@pytest.mark.unit
def test_polaris_rejects_unknown_device(reset_typespec):
    """An arch name absent from the archspec fails the run instead of running nothing."""
    assert polaris.main(BASE_ARGS + ['--filterarch', 'NoSuchDevice']) != 0, \
        "TT-Polaris should fail when --filterarch names a device the archspec does not define"


@pytest.mark.unit
def test_polaris_rejects_unknown_workload_instance(reset_typespec):
    """Same for a workload-instance name absent from the workload spec."""
    assert polaris.main(BASE_ARGS + ['--filterwli', 'no_such_workload_instance']) != 0, \
        "TT-Polaris should fail when --filterwli names an instance the wlspec does not define"


@pytest.mark.unit
def test_polaris_rejects_valid_but_disjoint_filters(reset_typespec):
    """Individually valid filters that share no workload still select nothing -- also a failure."""
    assert polaris.main(BASE_ARGS + ['--filterwl', 'BERT_SQUAD_v1p1', '--filterwli', 'rn50_b1']) != 0, \
        "TT-Polaris should fail when valid filters jointly select no workload"


@pytest.mark.unit
def test_polaris_accepts_valid_filters(reset_typespec):
    """The rejections above must not cost us the ordinary filtered run."""
    assert polaris.main(BASE_ARGS + ['--filterarch', 'Q1_A1', '--filterwli', 'rn50_b1']) == 0, \
        "TT-Polaris should run normally when every filter entry names something real"


@pytest.mark.unit
def test_polaris_reports_empty_spec_rather_than_blaming_filters(reset_typespec, tmp_path):
    """An empty spec and a disjoint filter pair are different mistakes in different files.

    With no filter given there is nothing to blame the filters for, so the message has to
    point at the spec instead — otherwise it sends the reader to their command line when the
    defect is in the YAML.
    """
    empty_spec = tmp_path / 'empty_workloads.yaml'
    empty_spec.write_text('workloads: []\n')

    args = [
        '--odir', str(tmp_path), '--study', 'dummy',
        '--wlspec', str(empty_spec),
        '--archspec', 'config/all_archs.yaml',
        '--wlmapspec', 'config/wl2archmapping.yaml',
        '--dryrun',
    ]
    with pytest.raises(polaris.SpecSelectionError) as excinfo:
        polaris.get_workloads(str(empty_spec), polaris.RangeArgument('batchsize', None), None, None, None)
    msg = str(excinfo.value)
    assert 'defines no workload instances' in msg, f"should point at the spec, got: {msg}"
    assert '--filter' not in msg, f"should not blame filters that were never given, got: {msg}"

    assert polaris.main(args) != 0, "an empty spec should still fail the run"


# --- --ignore-lut propagation: polaris() must set ignore_lut on every device in devspec ---

def _polaris_args(source, tmp_path, ignore_lut, hybrid_curve=False):
    """Build the argument object polaris() receives: argparse.Namespace (CLI) or PolarisRunConfig (polproj)."""
    if source == 'namespace':
        argv = [
            '--odir', str(tmp_path), '--study', 'dummy',
            '--wlspec', 'config/mlperf_inference.yaml',
            '--archspec', 'config/all_archs.yaml',
            '--wlmapspec', 'config/wl2archmapping.yaml',
            '--filterarch', 'Q1_A1', '--filterwli', 'rn50_b1',
        ]
        if ignore_lut:
            argv.append('--ignore-lut')
        if hybrid_curve:
            argv.append('--operator-lookup-hybrid-curve')
        return polaris.setup_cmdline_args(argv)
    return polaris.runcfgmodel.PolarisRunConfig(
        title='ignore_lut test', odir=str(tmp_path), study='dummy',
        wlspec='config/mlperf_inference.yaml', archspec='config/all_archs.yaml',
        wlmapspec='config/wl2archmapping.yaml', filterarch='Q1_A1', filterwli='rn50_b1',
        ignore_lut=ignore_lut, operator_lookup_hybrid_curve=hybrid_curve,
    )


def _capture_devspec(monkeypatch):
    """Stub the execution stage and return a dict that receives the devspec polaris() passes to it."""
    seen = {}

    def fake_execute_wl_on_dev(workload_list, device_list, wlspec, devspec, *a, **kw):
        seen['devspec'] = devspec
        return 0, []

    monkeypatch.setattr(polaris, 'create_uniq_workloads_tbl', lambda wl_list: {})
    monkeypatch.setattr(polaris, 'execute_wl_on_dev', fake_execute_wl_on_dev)
    return seen


@pytest.mark.unit
@pytest.mark.parametrize('source', ['namespace', 'runconfig'])
def test_polaris_ignore_lut_sets_every_device(reset_typespec, monkeypatch, tmp_path, source):
    seen = _capture_devspec(monkeypatch)
    assert polaris.polaris(_polaris_args(source, tmp_path, ignore_lut=True)) == 0
    devspec = seen['devspec']
    assert len(devspec) > 1, "test needs several devices to show every entry is set, not just one"
    assert all(d.ignore_lut is True for d in devspec.values()), \
        {name: d.ignore_lut for name, d in devspec.items()}


@pytest.mark.unit
@pytest.mark.parametrize('source', ['namespace', 'runconfig'])
def test_polaris_without_ignore_lut_leaves_devices_unchanged(reset_typespec, monkeypatch, tmp_path, source):
    seen = _capture_devspec(monkeypatch)
    assert polaris.polaris(_polaris_args(source, tmp_path, ignore_lut=False)) == 0
    assert all(d.ignore_lut is False for d in seen['devspec'].values())


@pytest.mark.unit
def test_polaris_main_ignore_lut_flag_reaches_devices(reset_typespec, monkeypatch, tmp_path):
    """Same check through main(), i.e. the real command-line entry point."""
    seen = _capture_devspec(monkeypatch)
    rc = polaris.main([
        '--odir', str(tmp_path), '--study', 'dummy',
        '--wlspec', 'config/mlperf_inference.yaml',
        '--archspec', 'config/all_archs.yaml',
        '--wlmapspec', 'config/wl2archmapping.yaml',
        '--filterarch', 'Q1_A1', '--filterwli', 'rn50_b1',
        '--ignore-lut',
    ])
    assert rc == 0
    assert all(d.ignore_lut is True for d in seen['devspec'].values())


@pytest.mark.unit
def test_polaris_warns_when_hybrid_curve_given_with_ignore_lut(reset_typespec, monkeypatch, tmp_path):
    _capture_devspec(monkeypatch)
    # polaris() calls setup_logger(), which removes every loguru sink, so patch the WARNING hook instead.
    messages = []
    monkeypatch.setattr(polaris, 'WARNING', lambda msg, *a, **kw: messages.append(str(msg)))
    polaris.polaris(_polaris_args('namespace', tmp_path, ignore_lut=True, hybrid_curve=True))
    assert any('--operator-lookup-hybrid-curve has no effect' in m for m in messages), messages


@pytest.mark.unit
def test_polaris_no_hybrid_curve_warning_without_ignore_lut(reset_typespec, monkeypatch, tmp_path):
    # Separate test: polaris() sets the WL2ArchTypeSpec singleton, so it can run once per test.
    _capture_devspec(monkeypatch)
    messages = []
    monkeypatch.setattr(polaris, 'WARNING', lambda msg, *a, **kw: messages.append(str(msg)))
    polaris.polaris(_polaris_args('namespace', tmp_path, ignore_lut=False, hybrid_curve=True))
    assert not any('has no effect' in m for m in messages), messages


def _run_polaris_cli(extra_args, tmp_path):
    """Invoke polaris.py as a subprocess and return the CompletedProcess."""
    repo_root = Path(polaris.__file__).parent
    argv = [
        sys.executable, str(repo_root / 'polaris.py'),
        '--odir', str(tmp_path), '--study', 'dummy',
        '--wlspec', 'config/mlperf_inference.yaml',
        '--archspec', 'config/all_archs.yaml',
        '--wlmapspec', 'config/wl2archmapping.yaml',
        '--dryrun',
    ] + extra_args
    return subprocess.run(argv, cwd=repo_root, capture_output=True, text=True)


@pytest.mark.unit
def test_polaris_cli_exits_nonzero_on_unknown_filter(tmp_path):
    """The process exit code must carry the failure, not just main()'s return value.

    The in-process tests above cannot catch this: the original defect was that __main__
    computed the status and then discarded it, so main() returned non-zero while the
    process still exited 0. Only a subprocess observes that.
    """
    res = _run_polaris_cli(['--filterarch', 'NoSuchDevice'], tmp_path)
    assert res.returncode != 0, \
        f"polaris.py should exit non-zero on an unknown filter; got {res.returncode}\n{res.stdout}\n{res.stderr}"
    assert 'NoSuchDevice' in (res.stdout + res.stderr), \
        "the error should name the offending filter entry"


@pytest.mark.unit
def test_polaris_cli_exits_zero_on_valid_run(tmp_path):
    """Counterpart to the above, so the check cannot pass by always failing."""
    res = _run_polaris_cli(['--filterarch', 'Q1_A1', '--filterwli', 'rn50_b1'], tmp_path)
    assert res.returncode == 0, \
        f"polaris.py should exit 0 on a valid run; got {res.returncode}\n{res.stdout}\n{res.stderr}"
