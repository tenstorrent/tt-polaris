#!/usr/bin/env python3
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
TTNN Profiler Runner

This script runs the TTNN profiler with multiple profiling modes using tracy.
It supports basic profiling, NOC trace collection, and performance counter capture.

Usage:
    python3 run-ttnn-profiler.py --command "<test_command>" \
        --report-name <name> --output-dir <dir> [options]

Examples:
    # Basic profiling only
    python3 run-ttnn-profiler.py \
        --command "test_ttnn_functional_vit.py" \
        --report-name vit_test \
        --output-dir ./profiler_output \
        --basic-only

    # Full profiling (raw + NOC traces + perf counters)
    python3 run-ttnn-profiler.py \
        --command "test_ttnn_functional_vit.py" \
        --report-name vit_full \
        --output-dir ./profiler_output

    # Pytest mode with cleanup
    python3 run-ttnn-profiler.py \
        --command "tests/ttnn/unit_tests/test_model.py::test_forward" \
        --pytest \
        --report-name model_test \
        --output-dir ./profiler_output \
        --cleanup

    # Dry run to see commands without executing
    python3 run-ttnn-profiler.py \
        --command "test.py" \
        --report-name test \
        --output-dir ./out \
        --dryrun

Required Arguments:
    --command       Test script or command to run under the profiler
    --report-name   Name for the profiling report
    --output-dir    Directory to save profiler outputs (must not exist)

Optional Arguments:
    --pytest            Run command with pytest (-m pytest prefix)
    --basic-only        Skip NOC traces and performance counter collection
    --op-support-count  Maximum number of ops tracy will profile (default: 100000)
    --disable-logging   Disable TTNN logging (enable_logging=False in config overrides)
                        Some workloads like VGG have device synchronize calls that
                        fail when enable_logging is true. Use this flag for such cases.
    --show-output       Show stdout/stderr in real-time during command execution.
                        Useful for monitoring progress and debugging hanging commands.
                        By default, output is only captured and saved to results file.
    --cleanup           Remove npe_viz and .logs directories after profiling.
                        These generated directories can be very large; use this flag
                        if their contents are not needed for analysis.
    --dryrun, -n        Show commands without executing them
    --fail-check / --no-fail-check
                        Run an unprofiled (no-tracy) pre-flight before the Tracy passes.
                        DEFAULT: off (--no-fail-check). The pre-flight runs the command once
                        WITHOUT tracy and aborts before the expensive Tracy passes if it fails.
                        It is a COMPLETE workload run, not a quick check -- only trivially
                        broken commands fail quickly; a working workload costs a full run
                        (~278s for ResNet50 on BH p100a). Off by default
                        because tracy now runs with --check-exit-code, so a failed/timed-out
                        capture pass already aborts the run (see run_profiler); pass --fail-check
                        to also gate on a plain non-profiled run first.
    --disable-dram-drop-guard  Disable the DRAM-drop guard (on by default). Normally, if the
                        profiler DRAM-buffer-overflow marker appears too frequently (capture likely
                        incomplete), the command is given a grace period and then force-stopped.

Output Structure:
    <output-dir>/
        ├── <pass>_results_typescript.txt  # Per-pass output (failcheck_/raw_/perf_/trace_/merge_),
        │                                  # streamed live + flushed per line so a Ctrl-C on a hang
        │                                  # keeps it; stderr lines tagged "[stderr]"
        ├── run_status.json        # Live run-status sidecar: overall + per-step state (machine-readable)
        ├── STATUS.txt             # Same, rendered as a human table (watch/cat it during a run)
        ├── merged_ops_<RUNID>.csv  # 3-CSV merge (full mode only; RUNID = output-dir name)
        ├── failcheck/              # No-tracy pre-flight artifacts (only with --fail-check)
        ├── raw/                    # Tracy raw profiling output (.logs/, reports/, generated/)
        ├── perf/                   # Tracy performance counter output (full mode only)
        └── trace/                  # Tracy NOC trace output (full mode only)

    Each pass dir is self-contained: TT_METAL_LOGS_PATH and TT_METAL_PROFILER_DIR
    are pinned to it (see run_profiler), so the bulk of tt-metal's 'generated/' output
    lands under the pass dir rather than the cwd. A few paths don't honor those env
    vars (e.g. generated/fabric, generated/test_reports) and may still appear in the
    cwd — small, idempotent, low collision risk (see presets/README.md). This lets two
    runs execute concurrently from the same (e.g. shared/NFS) tt-metal checkout.

Prerequisites:
    - NPE tools must be on PATH (source tt-npe/ENV_SETUP)
    - Tracy profiler module must be available
    - loguru package must be installed

Notes:
    - By default the Tracy passes run directly; tracy is invoked with --check-exit-code so a
      failed/timed-out capture pass aborts the run at the point of failure. Pass --fail-check to
      additionally run an unprofiled (no-tracy) pre-flight first, so a trivially-broken command
      (import error, bad -k selector, arg error) fails before any capture starts. Note it is a
      complete workload run, so only broken commands fail quickly -- a working one pays for a
      full extra run.
    - The script runs up to 3 profiling passes:
      1. Raw profiling with TTNN config overrides
      2. Performance counter collection (unless --basic-only)
      3. NOC trace collection (unless --basic-only)
    - Execution stops if the pre-flight (when enabled with --fail-check) or any pass fails
    - DRAM-drop guard (on by default; --disable-dram-drop-guard turns it off): if the profiler
      DRAM-buffer-overflow marker ("Profiler DRAM buffers were full, markers were dropped!") is
      seen too frequently, the command is given a grace period to finish and then force-stopped,
      since its capture is almost certainly incomplete. With the guard off, the marker is still
      counted and the run-status step flagged dram_drop_tripped (a rc-0 pass is not silently trusted).
    - A live status sidecar (run_status.json + STATUS.txt) is written to the output dir, updated on
      every step transition and on a ~2s heartbeat, so `watch cat <dir>/STATUS.txt` shows overall +
      per-step progress (including stalls via last-output age, interrupts, and force-stops).
    - All tt-metal artifacts land directly under each pass dir (no cwd/generated);
      concurrent runs from one directory therefore do not collide
    - After a successful full run (raw+perf+trace), the three CSVs are merged into
      merged_ops_<RUNID>.csv on this node, so only it (+ hw_id.json, run_status.json, STATUS.txt)
      need be copied off the board. The per-board DRAM peak BW is resolved from an internal interim table
      keyed on tt-smi board_type (interim/placeholder values pending the DRAM-BW
      review). Skipped for --basic-only or when the board can't be resolved.
"""

import os
import sys
import argparse
import datetime
import importlib.util
import json
import re
import shlex
import shutil
import subprocess
import threading
import time
from collections import deque

from loguru import logger

def is_npe_on_path() -> bool:
    """Check if NPE tools are available on the Python path."""
    return importlib.util.find_spec('npe_analyze_noc_trace_dir') is not None


# DRAM-drop guard: tt-metal emits this exact line when the on-device profiler's DRAM marker buffer
# overflows and drops markers. Sustained bursts mean the capture is losing data and is likely
# incomplete. When the guard is enabled (default), if the marker is seen >= _DRAM_DROP_RATE_COUNT
# times within any _DRAM_DROP_RATE_WINDOW_S-second rolling window, the command is given
# _DRAM_DROP_GRACE_S to finish on its own; if it is still running after that, it is force-stopped.
_DRAM_DROP_MARKER = 'Profiler DRAM buffers were full, markers were dropped!'
_DRAM_DROP_RATE_COUNT = 25
_DRAM_DROP_RATE_WINDOW_S = 10.0
_DRAM_DROP_GRACE_S = 30.0
_DRAM_DROP_POLL_S = 0.5

# run_status sidecar: how often to rewrite run_status.json / STATUS.txt while a step is running.
_STATUS_HEARTBEAT_S = 2.0


def run_and_capture(argv: list[str], show_output: bool = False,
                    log_path: str | None = None,
                    dram_drop_guard: bool = True,
                    status: "RunStatus | None" = None,
                    step_name: str | None = None) -> subprocess.CompletedProcess[str]:
    """Execute a command, capturing its output and (if log_path is given) streaming it to disk live.

    When ``log_path`` is set, each output line is written to the file and flushed AS IT ARRIVES
    (independent of ``show_output``), and the file is finalized in a ``finally`` block. So if the
    command hangs and is Ctrl-C'd, the output produced up to that point is already on disk rather
    than lost with the un-returned in-memory buffer. stdout is written verbatim; stderr lines are
    tagged with a "[stderr] " prefix; the two streams are interleaved in arrival order (a true
    typescript). On SIGINT the child is terminated and the run reported as returncode 130.

    When ``dram_drop_guard`` is True (default), the output is watched for the profiler
    DRAM-buffer-overflow marker; if it appears too frequently (see the _DRAM_DROP_* constants) the
    command is given a grace period to finish and then force-stopped (reported as returncode 124),
    because such a run's capture is almost certainly incomplete.

    Args:
        argv (list[str]): Command and arguments as a list (shell=False).
        show_output (bool): If True, also echo stdout/stderr to this process's terminal in real time.
        log_path (str | None): If set, stream output there live (<mode>_results_typescript.txt).
        dram_drop_guard (bool): If True, force-stop the command on sustained DRAM-drop markers.
        status (RunStatus | None): If set, the run-status sidecar updated with this step's
            last-output heartbeat and DRAM-drop annotation.
        step_name (str | None): Name of this step in the run-status sidecar.

    Returns:
        subprocess.CompletedProcess: Result object with returncode, stdout, stderr.
    """
    logf = open(log_path, 'w', buffering=1) if log_path is not None else None  # line-buffered
    write_lock = threading.Lock()
    if logf is not None:
        logf.write(f'Command: {" ".join(argv)}\n')
        logf.write(f'Started (UTC): {datetime.datetime.now(datetime.timezone.utc).isoformat()}\n')
        logf.write('---- live output (stdout verbatim, stderr tagged "[stderr]"), flushed per line ----\n')
        logf.flush()

    try:
        process = subprocess.Popen(
            argv,
            shell=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
    except FileNotFoundError as e:
        # e.g. tt-smi not installed; surface as a failed run (returncode 127) rather
        # than crashing, so callers can detect it via anyfails() and exit non-zero.
        logger.error(f'command not found: {argv[0]!r} ({e})')
        if logf is not None:
            logf.write(f'{e}\n---- end (return code: 127; command not found) ----\n')
            logf.close()
        return subprocess.CompletedProcess(argv, returncode=127, stdout='', stderr=str(e))
    proc_stdout = process.stdout
    proc_stderr = process.stderr
    assert proc_stdout is not None
    assert proc_stderr is not None

    stdout_lines: list[str] = []
    stderr_lines: list[str] = []

    # DRAM-drop guard state: rolling window of marker-sighting timestamps; drop_trigger fires once
    # the rate threshold is crossed; force_stopped records that the watchdog killed the command.
    drop_times: deque[float] = deque()
    drop_lock = threading.Lock()
    drop_trigger = threading.Event()
    force_stopped = threading.Event()

    def note_drop_marker() -> None:
        if drop_trigger.is_set():
            return
        now = time.monotonic()
        with drop_lock:
            drop_times.append(now)
            while drop_times and now - drop_times[0] > _DRAM_DROP_RATE_WINDOW_S:
                drop_times.popleft()
            tripped = len(drop_times) >= _DRAM_DROP_RATE_COUNT
        if tripped:
            drop_trigger.set()
            if dram_drop_guard:
                msg = (f'DRAM-drop guard: profiler DRAM-buffer-overflow marker seen '
                       f'>={_DRAM_DROP_RATE_COUNT}x within {_DRAM_DROP_RATE_WINDOW_S:.0f}s -- capture is '
                       f'dropping data and is likely incomplete. Giving the command {_DRAM_DROP_GRACE_S:.0f}s '
                       f'to finish, then force-stopping.')
            else:
                msg = (f'DRAM-drop marker seen >={_DRAM_DROP_RATE_COUNT}x within '
                       f'{_DRAM_DROP_RATE_WINDOW_S:.0f}s -- capture is dropping data and is likely INCOMPLETE '
                       f'(guard disabled: NOT force-stopping; flagged in run_status).')
            logger.warning(msg)
            if logf is not None:
                with write_lock:
                    logf.write(f'---- {msg} ----\n')
                    logf.flush()
            if status is not None and step_name is not None:
                status.mark_dram_tripped(step_name)

    def emit(line: str, is_stderr: bool) -> None:
        # Buffer (for the returned CompletedProcess), stream to the log file live, and optionally
        # echo to the terminal. The file write is gated on log_path only -- NOT on show_output --
        # so a silent (default) run still lands its output on disk as it is produced.
        (stderr_lines if is_stderr else stdout_lines).append(line)
        if logf is not None:
            with write_lock:
                logf.write(f'[stderr] {line}' if is_stderr else line)
                logf.flush()
        if show_output:
            stream = sys.stderr if is_stderr else sys.stdout
            print(line, end='', file=stream)
            stream.flush()
        if status is not None and step_name is not None:
            status.note_output(step_name)
        # Count the DRAM-drop marker regardless of the guard: with the guard on it drives the
        # force-stop; with it off it still flags the step (dram_drop_tripped) as likely incomplete.
        if _DRAM_DROP_MARKER in line:
            note_drop_marker()

    def read_stdout() -> None:
        for line in iter(proc_stdout.readline, ''):
            if line:
                emit(line, False)
        proc_stdout.close()

    def read_stderr() -> None:
        for line in iter(proc_stderr.readline, ''):
            if line:
                emit(line, True)
        proc_stderr.close()

    def dram_watchdog() -> None:
        # Wait until the guard trips or the process ends (whichever first). poll() is used (never
        # wait()) so we never contend with the main thread's process.wait() for reaping.
        while process.poll() is None and not drop_trigger.is_set():
            time.sleep(_DRAM_DROP_POLL_S)
        if not drop_trigger.is_set():
            return  # process finished before the threshold was crossed
        # Threshold crossed while still running: grace period, then force-stop if it hasn't ended.
        deadline = time.monotonic() + _DRAM_DROP_GRACE_S
        while process.poll() is None and time.monotonic() < deadline:
            time.sleep(_DRAM_DROP_POLL_S)
        if process.poll() is None:
            force_stopped.set()
            logger.error(f'DRAM-drop guard: command still running {_DRAM_DROP_GRACE_S:.0f}s after the '
                         'threshold -- force-stopping.')
            process.terminate()
            kill_deadline = time.monotonic() + 10
            while process.poll() is None and time.monotonic() < kill_deadline:
                time.sleep(_DRAM_DROP_POLL_S)
            if process.poll() is None:
                process.kill()

    hb_stop = threading.Event()

    def status_heartbeat() -> None:
        # Refresh the run_status sidecar every ~2s while this step runs, so a reader sees liveness
        # (elapsed + last-output age). Stopped promptly when the process ends (hb_stop).
        while not hb_stop.wait(_STATUS_HEARTBEAT_S):
            if status is not None and step_name is not None:
                status.heartbeat(step_name)

    stdout_thread = threading.Thread(target=read_stdout)
    stderr_thread = threading.Thread(target=read_stderr)
    watchdog_thread = threading.Thread(target=dram_watchdog) if dram_drop_guard else None
    heartbeat_thread = (threading.Thread(target=status_heartbeat)
                        if (status is not None and step_name is not None) else None)
    stdout_thread.start()
    stderr_thread.start()
    if watchdog_thread is not None:
        watchdog_thread.start()
    if heartbeat_thread is not None:
        heartbeat_thread.start()

    interrupted = False
    try:
        process.wait()
    except KeyboardInterrupt:
        # Ctrl-C on a hung command: stop waiting, reap the child, and let the finally block finalize
        # the (already-flushed) log. Report as 130 so anyfails() trips and the run aborts cleanly.
        interrupted = True
        logger.error('Interrupted (SIGINT) -- terminating command; output captured so far is saved.')
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    finally:
        hb_stop.set()
        stdout_thread.join()
        stderr_thread.join()
        if watchdog_thread is not None:
            watchdog_thread.join()
        if heartbeat_thread is not None:
            heartbeat_thread.join()
        if logf is not None:
            with write_lock:
                if interrupted:
                    logf.write(f'\n---- INTERRUPTED (SIGINT); child return code: {process.returncode} '
                               '-- partial output above ----\n')
                elif force_stopped.is_set():
                    logf.write(f'\n---- FORCE-STOPPED by DRAM-drop guard (marker >={_DRAM_DROP_RATE_COUNT}x/'
                               f'{_DRAM_DROP_RATE_WINDOW_S:.0f}s, not finished within {_DRAM_DROP_GRACE_S:.0f}s); '
                               f'child return code {process.returncode} -- capture incomplete ----\n')
                elif drop_trigger.is_set():
                    # Reached only when the command finished on its own after tripping the threshold.
                    # With the guard on that means it beat the grace period; with the guard off there
                    # was no grace period at all -- word it accordingly so the typescript isn't misleading.
                    detail = 'finished within grace' if dram_drop_guard else 'ran to completion (guard disabled)'
                    logf.write(f'---- end (return code: {process.returncode}); WARNING: DRAM-drop threshold '
                               f'was tripped but the command {detail} -- capture may be incomplete ----\n')
                else:
                    logf.write(f'---- end (return code: {process.returncode}) at '
                               f'{datetime.datetime.now(datetime.timezone.utc).isoformat()} ----\n')
                logf.flush()
            logf.close()

    if interrupted:
        returncode = 130
    elif force_stopped.is_set():
        returncode = 124
    else:
        returncode = process.returncode
    result = subprocess.CompletedProcess(
        args=argv,
        returncode=returncode,
        stdout=''.join(stdout_lines),
        stderr=''.join(stderr_lines),
    )

    if interrupted:
        logger.error('Command interrupted.')
    elif force_stopped.is_set():
        logger.error('Command force-stopped by DRAM-drop guard (capture incomplete).')
    elif result.returncode == 0:
        if drop_trigger.is_set():
            logger.warning('Command finished, but the DRAM-drop threshold was tripped -- capture may be incomplete.')
        else:
            logger.info('Command ran successfully.')
    else:
        logger.error('Command failed to run.')

    return result


def setup_environment(report_name: str, enable_logging: bool = True) -> None:
    """Configure TTNN environment variables for profiling.

    Sets up TTNN_CONFIG_OVERRIDES with profiling-friendly settings and
    enables device profiler environment variables. Also sets TT_METAL_HOME
    and PYTHONPATH based on the current working directory.

    Note: This script is expected to be run from the tt-metal home directory.

    Args:
        report_name (str): Name for the profiling report
        enable_logging (bool): Whether to enable TTNN logging. Defaults to True.
            Set to False for workloads like VGG that have device synchronize
            calls which fail when logging is enabled.
    """
    metal_home = os.getcwd()
    os.environ['TT_METAL_HOME'] = metal_home

    current_pythonpath = os.environ.get('PYTHONPATH', '')
    if current_pythonpath:
        os.environ['PYTHONPATH'] = f'{metal_home}:{current_pythonpath}'
    else:
        os.environ['PYTHONPATH'] = metal_home

    environ = {
        'enable_fast_runtime_mode': False,
        'enable_logging': enable_logging,
        'report_name': report_name,
        'enable_graph_report': False,
        'enable_detailed_buffer_report': enable_logging,
        'enable_detailed_tensor_report': False,
        'enable_comparison_mode': False,
    }
    os.environ['TTNN_CONFIG_OVERRIDES'] = json.dumps(environ)

    os.environ['TT_METAL_DEVICE_PROFILER'] = '1'
    os.environ['TT_METAL_PROFILER_SYNC'] = '1'
    # Do NOT enable TT_METAL_PROFILER_CPP_POST_PROCESS here. The C++ post-process path emits the
    # leaner cpp_device_perf_report.csv and explicitly drops device_analysis_types (see tt-metal
    # process_ops_logs.py: "device_analysis_types is not supported when using
    # cpp_device_perf_report.csv; ignoring option"). That analysis is what produces the per-op
    # FPU/SFPU Util Median columns. ops_perf_three_csv_merge.py classifies the perf pass as the
    # "fpu" CSV via those columns, so enabling cpp post-process makes the merge fail with
    # "Expected exactly one fpu CSV, found 0". The legacy device-log parser (import_log_run_stats)
    # is the required path here; the "cpp_device_perf_report.csv not found" warning is expected.
    # NOTE: capture the workload's real command, trace included — do NOT add --disable_trace.
    # With trace on, an op is logged across capture + replay passes (capture = timed; replay =
    # same GLOBAL CALL COUNT, and for some demos empty-duration "shadow" rows). The 3-CSV merge
    # deduplicates these via its op-code-period iteration detection, keeping the refrun faithful
    # to how the model runs on hardware. Disabling trace would change the measured execution
    # path, so it is not used here or in the presets.


def run_failcheck(command: str, pytest_mode: bool, failcheck_dir: str,
                  show_output: bool = False, dryrun: bool = False,
                  log_path: str | None = None,
                  dram_drop_guard: bool = True,
                  status: "RunStatus | None" = None,
                  step_name: str | None = None) -> subprocess.CompletedProcess[str] | None:
    """Unprofiled pre-flight: run the workload command ONCE WITHOUT tracy, to fail fast on a
    trivially-broken command (import error, bad -k selector, collection/arg error) before the
    expensive 3-pass Tracy capture. This is a COMPLETE workload run -- only broken commands fail
    quickly. Returns CompletedProcess (None on dryrun). NOT a profiling pass -- no tracy, no
    perf/noc collection."""
    # The --command is encoded to survive tracy's report-mode round-trip: tracy joins its argv back
    # into one string and re-runs it under shell=True, so a spaced -k expression is wrapped in
    # nested quotes in the presets (e.g. -k '"performance and batch-32"'). This pre-flight runs the
    # command WITHOUT tracy, so mirror that same shlex.split -> join -> shell re-split to recover the
    # argv the workload actually runs with. A single shlex.split would leave literal quotes on the
    # -k value and pytest would reject it ("Wrong expression passed to '-k'").
    real_args = shlex.split(' '.join(shlex.split(command)))
    if pytest_mode:
        argv = [sys.executable, '-m', 'pytest'] + real_args
    else:
        argv = [sys.executable] + real_args
    logger.info(f'no-tracy fail-check: {" ".join(argv)}')
    if dryrun:
        return None
    # Pin tt-metal's artifact roots to a throwaway failcheck dir so this pre-run does not drop
    # 'generated/' into the cwd (mirrors the per-pass isolation in run_profiler; keeps concurrent
    # runs from a shared tt-metal checkout from colliding). No setup_environment/tracy overrides --
    # this is a plain run; TT_METAL_HOME/PYTHONPATH come from the shell env (setup-step-2).
    os.makedirs(failcheck_dir, exist_ok=True)
    os.environ['TT_METAL_LOGS_PATH'] = failcheck_dir
    os.environ['TT_METAL_PROFILER_DIR'] = failcheck_dir
    return run_and_capture(argv, show_output=show_output, log_path=log_path,
                           dram_drop_guard=dram_drop_guard, status=status, step_name=step_name)


def run_profiler(
    command: str,
    output_dir: str,
    pytest_mode: bool,
    report_name: str,
    collect_noc_traces: bool = False,
    collect_perf_counters: bool = False,
    enable_logging: bool = True,
    show_output: bool = False,
    dryrun: bool = False,
    op_support_count: int = 100000,
    no_op_info_cache: bool = False,
    log_path: str | None = None,
    dram_drop_guard: bool = True,
    status: "RunStatus | None" = None,
    step_name: str | None = None,
) -> subprocess.CompletedProcess[str] | None:
    """Run the TTNN profiler with specified options.

    Args:
        command (str): Test script or command to run under the profiler
        output_dir (str): Directory to save profiler output for this pass
        pytest_mode (bool): If True, prefix command with '-m pytest'
        report_name (str): Name for the profiling report
        collect_noc_traces (bool): If True, collect NOC traces
        collect_perf_counters (bool): If True, collect performance counters
        enable_logging (bool): Whether to enable TTNN logging in config overrides
        show_output (bool): If True, display command stdout/stderr in real-time
        dryrun (bool): If True, show commands without executing
        op_support_count (int): Maximum number of ops tracy will profile
        no_op_info_cache (bool): If True, pass --no-op-info-cache to tracy so every op's full info
            (input/output tensor shapes) is recorded, not reused from the first op with the same
            program hash

    Returns:
        subprocess.CompletedProcess or None: Result object or None if dryrun
    """
    # Use the current interpreter (sys.executable) so tracy runs under the same
    # venv/python as this wrapper, rather than relying on whatever 'python3' is on PATH.
    # --check-exit-code makes tracy propagate the wrapped command's non-zero exit
    # (tt-metal tools/tracy/__main__.py); without it tracy still generates its report and
    # exits 0 even when the pytest failed/timed out, so anyfails() below would see a false
    # success and run the remaining passes + merge on a truncated capture (the failure only
    # surfacing later as a confusing merge row-count mismatch).
    argv = [sys.executable, '-m', 'tracy', '-p', '-r', '-v', '--check-exit-code',
            f'--op-support-count={op_support_count}', '-o', output_dir]
    if no_op_info_cache:
        # tracy deletes TT_METAL_PROFILER_NO_CACHE_OP_INFO from the environment unless it gets this
        # flag (tt-metal tools/tracy/__main__.py), so exporting the variable has no effect.
        argv.append('--no-op-info-cache')
    if collect_noc_traces:
        argv.append('--collect-noc-traces')
    if collect_perf_counters:
        argv.append('--profiler-capture-perf-counters=fpu')
    if pytest_mode:
        argv.extend(['-m', 'pytest'] + shlex.split(command))
    else:
        argv.extend(shlex.split(command))

    logger.info(f'{argv=}')
    if collect_noc_traces:
        mode = 'trace'
    elif collect_perf_counters:
        mode = 'perf-counter'
    else:
        mode = 'raw'

    logger.info(f'Running command in {mode} mode: {" ".join(argv)}')

    if mode == 'raw':
        setup_environment(report_name, enable_logging)
    else:
        if 'TTNN_CONFIG_OVERRIDES' in os.environ:
            del os.environ['TTNN_CONFIG_OVERRIDES']

    # Pin every tt-metal artifact root to THIS pass's output dir so that nothing
    # is written into the current working directory.
    #
    # Motivation: tt-metal anchors its artifact trees to the cwd by default --
    #   * TT_METAL_LOGS_PATH defaults to the current working directory, and the
    #     'generated/{reports,inspector}/' tree hangs off it (rtoptions);
    #   * TT_METAL_PROFILER_DIR defaults to TT_METAL_HOME/generated/profiler, and
    #     TT_METAL_HOME is itself forced to cwd in setup_environment().
    # So two profiling runs launched from the SAME directory (a shared/NFS
    # tt-metal checkout, or two boards driven from one host) both create
    # cwd/generated and clobber each other's logs -- and the old guard in main()
    # even hard-aborted the second run. Pointing these env vars at the unique
    # per-pass output dir makes each run fully self-contained, so concurrent runs
    # from one directory coexist. This supersedes the previous "let tt-metal
    # create generated/ in cwd, then os.rename it into the output dir afterwards"
    # approach (which only worked for one run at a time).
    os.environ['TT_METAL_LOGS_PATH'] = output_dir
    os.environ['TT_METAL_PROFILER_DIR'] = output_dir

    if dryrun:
        return None

    result = run_and_capture(argv, show_output=show_output, log_path=log_path,
                             dram_drop_guard=dram_drop_guard, status=status, step_name=step_name)
    if result.returncode == 0:
        logger.info(f'Profiler {mode} mode output saved to: {output_dir} with report name: {report_name}')

    return result


def anyfails(results: list[subprocess.CompletedProcess[str] | None]) -> bool:
    """Check if any profiler run failed."""
    for result in results:
        if result is None:
            continue
        if result.returncode != 0:
            return True
    return False


def cleanup_directories(output_dir: str) -> None:
    """Remove npe_viz and .logs directories and profile_log_device.csv files recursively from output directory."""
    logger.info('Performing cleanup...')
    dirs_to_remove = ['npe_viz', '.logs']
    files_to_remove = ['profile_log_device.csv']
    for root, dirs, files in os.walk(output_dir, topdown=False):
        for dir_name in dirs:
            if dir_name in dirs_to_remove:
                dir_path = os.path.join(root, dir_name)
                logger.info(f'Removing directory: {dir_path}')
                shutil.rmtree(dir_path)
        for file_name in files:
            if file_name in files_to_remove:
                file_path = os.path.join(root, file_name)
                logger.info(f'Removing file: {file_path}')
                os.remove(file_path)
    logger.info('Cleanup completed.')


# Per-pass output goes to <output_dir>/<mode>_results_typescript.txt, written LIVE by
# run_and_capture (see its log_path arg) so it survives a Ctrl-C on a hung command.


# Interim board_type -> DRAM peak bandwidth (GB/s), used to merge a capture on the
# hardware node without a CLI value. These are PLACEHOLDER numbers pending the
# DRAM-BW review resolution: they feed only the util columns of merged_ops.csv (the
# merge structure is unaffected). Keyed by tt-smi board_type PREFIX so revision
# suffixes (e.g. "n150 L") still match. When that review lands, this table should
# move into ops_perf_three_csv_merge.py as a proper per-chip default (answering the
# "set defaults per chip" review comment) and be replaced with validated values.
_INTERIM_BOARD_DRAM_PEAK_BW_GBPS: dict[str, float] = {
    'n150': 288.0,   # Wormhole  (PLACEHOLDER)
    'n300': 288.0,   # Wormhole  (PLACEHOLDER)
    'p100': 448.0,   # Blackhole (PLACEHOLDER)
    'p150': 448.0,   # Blackhole (PLACEHOLDER)
}


def detect_board_type() -> str | None:
    """Best-effort tt-smi board_type (e.g. 'n150', 'p100a'); None if unavailable."""
    try:
        res = subprocess.run(['tt-smi', '-s'], capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return None
    m = re.search(r'"board_type"\s*:\s*"([^"]+)"', res.stdout)
    return m.group(1).strip() if m else None


def resolve_interim_dram_peak_bw(board_type: str) -> float | None:
    """Map a tt-smi board_type to the interim DRAM peak BW (GB/s) via prefix match."""
    key = board_type.strip().lower()
    for prefix, bw in _INTERIM_BOARD_DRAM_PEAK_BW_GBPS.items():
        if key.startswith(prefix):
            return bw
    return None


_MERGE_TOOL_BY_VARIANT = {
    # multi-iteration workloads (VGG UNet): the iterative reducer detects iteration
    # boundaries and medians across them.
    'iterative': 'ops_perf_three_csv_merge.py',
    # non-iterative trace+replay workloads (ViT, llama3): a single compile pass +
    # replay session(s), no iteration loop — drops the compile pass, medians replays.
    'trace_replay': 'ops_perf_trace_replay_merge.py',
}


def run_merge(output_dir: str, dram_peak_bw_gbps: float, show_output: bool = False,
              merge_variant: str = 'iterative',
              log_path: str | None = None,
              dram_drop_guard: bool = True,
              status: "RunStatus | None" = None,
              step_name: str | None = None) -> subprocess.CompletedProcess[str] | None:
    """Consolidate the three per-pass CSVs (raw/perf/trace) under output_dir into
    output_dir/merged_ops_<RUNID>.csv (RUNID = output_dir basename), ON THE HARDWARE
    NODE, right after a successful capture.

    ``merge_variant`` selects the reducer: 'iterative' (ops_perf_three_csv_merge, for
    multi-iteration workloads like VGG) or 'trace_replay' (ops_perf_trace_replay_merge,
    for non-iterative trace+replay workloads like ViT/llama3). Both share the same CLI
    (--input-dir / --dram-peak-bw-gbps / --output).

    The merge tool ships alongside this wrapper (same dir), so it is part of the
    rsynced si_profiling_helpers bundle and its only third-party dep (loguru) is
    already in the tt-metal run env. Merging here means only merged_ops_<RUNID>.csv
    (plus hw_id.json) need be copied off the board, not three raw CSVs. The follow-on
    compare-vs-refrun step needs tt-polaris/ttsim and stays off-device.
    """
    merge_tool = os.path.join(os.path.dirname(os.path.realpath(__file__)), _MERGE_TOOL_BY_VARIANT[merge_variant])
    if not os.path.exists(merge_tool):
        logger.error(f'merge tool not found at {merge_tool}; skipping merge')
        return None
    # Name the merged CSV after the run dir (RUNID = output_dir basename) so per-run
    # merges stay distinct when copied side by side. The merge tool stays generic
    # (its own default is merged_ops.csv); the RUNID-specific name is chosen here,
    # where the run is known and the merge is invoked.
    merged_csv = os.path.join(output_dir, f'merged_ops_{os.path.basename(output_dir)}.csv')
    argv = [sys.executable, merge_tool, '--input-dir', output_dir,
            '--dram-peak-bw-gbps', str(dram_peak_bw_gbps), '--output', merged_csv]
    logger.info(f'merging three CSVs in {output_dir} (--dram-peak-bw-gbps {dram_peak_bw_gbps})')
    result = run_and_capture(argv, show_output=show_output, log_path=log_path,
                             dram_drop_guard=dram_drop_guard, status=status, step_name=step_name)
    if result.returncode == 0:
        logger.info(f'{os.path.basename(merged_csv)} written under {output_dir}')
    else:
        logger.error('merge step failed (see output above)')
    return result


def _utcnow() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _step_status_from_rc(rc: "int | None") -> str:
    """Map a run_and_capture returncode to a per-step status."""
    if rc is None:
        return 'skipped'
    if rc == 0:
        return 'succeeded'
    if rc == 130:
        return 'interrupted'   # SIGINT
    if rc == 124:
        return 'force_stopped'  # DRAM-drop guard
    return 'failed'


class RunStatus:
    """Live run-status sidecar for a profiler run.

    Writes two files into the run's output dir, rewritten on every step transition and on a ~2s
    heartbeat while a step runs:
      * run_status.json -- canonical, machine-readable (overall + per-step state, timings,
        last-output timestamps, DRAM-drop flags).
      * STATUS.txt      -- the same rendered as a human table.
    Lets an operator cat/watch overall + per-step progress, including stalls (last-output age),
    SIGINT interrupts, and DRAM-drop force-stops. Copied off-board alongside hw_id.json.

    main() drives the step transitions (start/finish/skip/finalize); run_and_capture feeds each
    step's last-output heartbeat and the DRAM-drop annotation. The pipeline is sequential, so
    transitions are single-threaded; a lock guards writes against the heartbeat thread.
    """

    _BAD = ('failed', 'interrupted', 'force_stopped')

    def __init__(self, output_dir: str, command: str, pytest_mode: bool, step_names: list[str]) -> None:
        self._json_path = os.path.join(output_dir, 'run_status.json')
        self._txt_path = os.path.join(output_dir, 'STATUS.txt')
        self._lock = threading.Lock()
        self._start_mono: dict[str, float] = {}
        self._last_out_mono: dict[str, float] = {}
        now = _utcnow()
        self._d: dict = {
            'runid': os.path.basename(output_dir),
            'output_dir': output_dir,
            'command': command,
            'pytest': pytest_mode,
            'overall_status': 'pending',
            'started_utc': now,
            'updated_utc': now,
            'ended_utc': None,
            'failed_steps': [],
            'steps': [
                {
                    'name': n, 'status': 'pending', 'returncode': None,
                    'started_utc': None, 'ended_utc': None, 'duration_s': None,
                    'last_output_utc': None, 'dram_drop_tripped': False,
                    'log': (None if n == 'board_reset' else f'{n}_results_typescript.txt'),
                }
                for n in step_names
            ],
        }
        with self._lock:
            self._write_locked()

    def _find(self, name: str) -> "dict | None":
        for s in self._d['steps']:
            if s['name'] == name:
                return s
        return None

    def start(self, name: str) -> None:
        with self._lock:
            s = self._find(name)
            if s is not None:
                s['status'] = 'running'
                s['started_utc'] = _utcnow()
                self._start_mono[name] = time.monotonic()
                self._last_out_mono[name] = time.monotonic()
                self._recompute_locked()
                self._write_locked()
        logger.info(f'step {name} -> running')

    def note_output(self, name: str) -> None:
        # Per-output-line, deliberately lock-free and cheap: record last-activity monotonic time
        # (the key already exists from start()); the heartbeat/transition writes fold it into UTC.
        self._last_out_mono[name] = time.monotonic()

    def mark_dram_tripped(self, name: str) -> None:
        with self._lock:
            s = self._find(name)
            if s is not None and not s['dram_drop_tripped']:
                s['dram_drop_tripped'] = True
                self._write_locked()

    def heartbeat(self, name: str) -> None:
        with self._lock:
            self._write_locked()

    def finish(self, name: str, res: "subprocess.CompletedProcess[str] | None") -> None:
        rc = None if res is None else res.returncode
        new_status = _step_status_from_rc(rc)
        dur = None
        with self._lock:
            s = self._find(name)
            if s is not None:
                s['returncode'] = rc
                s['status'] = new_status
                s['ended_utc'] = _utcnow()
                if name in self._start_mono:
                    s['duration_s'] = round(time.monotonic() - self._start_mono[name], 1)
                dur = s['duration_s']
                self._recompute_locked()
                self._write_locked()
        logger.info(f'step {name} -> {new_status}' + (f' ({dur}s)' if dur is not None else ''))

    def skip(self, name: str) -> None:
        with self._lock:
            s = self._find(name)
            if s is not None and s['status'] == 'pending':
                s['status'] = 'skipped'
                self._recompute_locked()
                self._write_locked()

    def finalize(self) -> None:
        with self._lock:
            for s in self._d['steps']:
                if s['status'] in ('pending', 'running'):
                    s['status'] = 'skipped'
            self._recompute_locked()
            self._d['ended_utc'] = _utcnow()
            self._write_locked()

    def _recompute_locked(self) -> None:
        statuses = [s['status'] for s in self._d['steps']]
        self._d['failed_steps'] = [s['name'] for s in self._d['steps'] if s['status'] in self._BAD]
        if 'interrupted' in statuses:
            overall = 'interrupted'
        elif 'force_stopped' in statuses:
            overall = 'force_stopped'
        elif 'failed' in statuses:
            overall = 'failed'
        elif 'running' in statuses:
            overall = 'running'
        elif 'pending' in statuses:
            # some steps not yet started: 'pending' only at the very start; else in-between -> running
            overall = 'pending' if all(st == 'pending' for st in statuses) else 'running'
        else:
            overall = 'succeeded'  # every step succeeded or skipped
        self._d['overall_status'] = overall

    def _write_locked(self) -> None:
        now_mono = time.monotonic()
        now_dt = datetime.datetime.now(datetime.timezone.utc)
        self._d['updated_utc'] = now_dt.isoformat()
        for s in self._d['steps']:
            lo = self._last_out_mono.get(s['name'])
            if lo is not None:
                s['last_output_utc'] = (now_dt - datetime.timedelta(seconds=now_mono - lo)).isoformat()
        tmp = self._json_path + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(self._d, f, indent=2)
        os.replace(tmp, self._json_path)  # atomic swap: a reader never sees a half-written JSON
        with open(self._txt_path, 'w') as f:
            f.write(self._render_txt(now_mono))

    def _render_txt(self, now_mono: float) -> str:
        d = self._d
        out = [
            f"RUN  {d['runid']}      [{d['overall_status'].upper()}]",
            f"cmd  {d['command']}",
            (f"started {d['started_utc']}   updated {d['updated_utc']}"
             + (f"   ended {d['ended_utc']}" if d['ended_utc'] else "")),
        ]
        if d['failed_steps']:
            out.append(f"failed_steps: {', '.join(d['failed_steps'])}")
        out.append("")
        out.append(f"{'STEP':<12} {'STATUS':<13} {'DUR/ELAPSED':>12}  {'LAST-OUT':>9}  LOG")
        for s in d['steps']:
            if s['status'] == 'running' and s['name'] in self._start_mono:
                dur = f"({round(now_mono - self._start_mono[s['name']], 1)}s…)"
            elif s['duration_s'] is not None:
                dur = f"{s['duration_s']}s"
            else:
                dur = "-"
            lo = self._last_out_mono.get(s['name'])
            lastout = f"{int(now_mono - lo)}s ago" if (s['status'] == 'running' and lo is not None) else "-"
            st = s['status'] + ('*' if s['dram_drop_tripped'] else '')
            out.append(f"{s['name']:<12} {st:<13} {dur:>12}  {lastout:>9}  {s['log'] or '-'}")
        if any(s['dram_drop_tripped'] for s in d['steps']):
            out.append("")
            out.append("* DRAM-drop marker threshold tripped -- capture may be incomplete.")
        return "\n".join(out) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description='Run the TTNN profiler')
    parser.add_argument('--command', type=str, required=True, help='Test script or command to run under the profiler')
    parser.add_argument('--pytest', action='store_true', help='Run the profiler in pytest mode')
    parser.add_argument('--report-name', type=str, required=True, help='Name of the profiler report')
    parser.add_argument('--output-dir', type=str, required=True, help='Directory to save the profiler output')
    parser.add_argument(
        '--merge-variant', choices=['iterative', 'trace_replay'], default='iterative',
        help='On-device 3-CSV merge reducer: "iterative" (ops_perf_three_csv_merge, multi-iteration '
             'workloads like VGG UNet — the default) or "trace_replay" (ops_perf_trace_replay_merge, '
             'non-iterative trace+replay workloads like ViT / llama3).',
    )
    parser.add_argument(
        '--basic-only', action='store_true', help='Only run basic profiling (skip NOC traces and performance counters)'
    )
    parser.add_argument(
        '--disable-logging',
        action='store_true',
        help='Disable TTNN logging (enable_logging=False). Use for workloads like VGG with device synchronize calls that fail when logging is enabled. Defaults to logging enabled.',
    )
    parser.add_argument(
        '--show-output',
        action='store_true',
        help='Show stdout/stderr in real-time during command execution. Useful for monitoring progress and debugging hanging commands.',
    )
    parser.add_argument(
        '--cleanup',
        action='store_true',
        help='Remove npe_viz and .logs directories from output directory after run. These directories can be very large; use this flag if their contents are not needed.',
    )
    parser.add_argument('--dryrun', '-n', action='store_true', help='show but do not execute commands')
    parser.add_argument('--fail-check', action=argparse.BooleanOptionalAction, default=False,
                        help='Run an unprofiled (no-tracy) pre-flight before the Tracy passes '
                             '(default: off, --no-fail-check). It is a complete workload run, not a '
                             'quick check -- only trivially broken commands fail fast. '
                             'Off by default since tracy runs with --check-exit-code, '
                             'which already aborts the run on a failed/timed-out capture pass; pass '
                             '--fail-check to also gate on a plain non-profiled run first.')
    parser.add_argument('--disable-dram-drop-guard', action='store_true',
                        help='Disable the DRAM-drop guard (on by default). Normally, if the profiler '
                             'DRAM-buffer-overflow marker appears too frequently (capture likely incomplete), '
                             'the command is given a grace period and then force-stopped.')
    parser.add_argument(
        '--op-support-count',
        type=int,
        default=100000,
        help='Maximum number of ops tracy will profile (default: 100000)',
    )
    parser.add_argument(
        '--no-op-info-cache',
        action='store_true',
        help='Pass --no-op-info-cache to tracy in every pass: record each op\'s full info instead of '
             'reusing it from the first op with the same program hash. Without it, ops that share a '
             'program hash but differ in tensor shape are reported with the first op\'s shapes '
             '(seen on WH n150 llama3 prefill: the second residual add recorded as [1,1,32,64]).',
    )
    args = parser.parse_args()

    command = args.command
    output_dir = os.path.realpath(args.output_dir)
    pytest_mode = args.pytest
    report_name = args.report_name
    enable_logging = not args.disable_logging
    show_output = args.show_output
    dram_guard = not args.disable_dram_drop_guard
    logger.remove()
    logger.add(sys.stdout, level='INFO')

    if not args.dryrun:
        if os.path.exists(output_dir):
            logger.error(f'Output directory {output_dir} already exists. Please specify a new directory.')
            return 1
        # NOTE: no longer guard against a pre-existing cwd/'generated' dir -- each
        # pass now pins TT_METAL_LOGS_PATH / TT_METAL_PROFILER_DIR to its own
        # output dir (see run_profiler), so tt-metal writes nothing into cwd and
        # concurrent runs from the same directory no longer collide.
        if not args.basic_only and not is_npe_on_path():
            logger.error('NPE not on path; source tt-npe/ENV_SETUP')
            return 1
        os.makedirs(output_dir)

    if any([word in args.command for word in ['tests/', 'pytest']]) and not args.pytest:
        logger.error('possibly using pytest without providing --pytest knob to this script')

    if 'vgg' in args.command.lower() and enable_logging:
        logger.error('possibly running VGG without disabling logging')
        os.system('sleep 5')

    raw_dir = os.path.join(output_dir, 'raw')
    perf_dir = os.path.join(output_dir, 'perf')
    trace_dir = os.path.join(output_dir, 'trace')
    failcheck_dir = os.path.join(output_dir, 'failcheck')

    # Plan the pipeline steps for the live run-status sidecar (no sidecar in dryrun). 'merge' is
    # planned only for a full capture (not --basic-only). Steps that never start (short-circuited
    # after an earlier failure) are marked 'skipped' at finalize().
    step_names = ['board_reset']
    if args.fail_check:
        step_names.append('failcheck')
    step_names.append('raw')
    if not args.basic_only:
        step_names += ['perf', 'trace', 'merge']
    status = RunStatus(output_dir, command, pytest_mode, step_names) if not args.dryrun else None

    def _start(step: str) -> None:
        if status is not None:
            status.start(step)

    def _finish(step: str, res: "subprocess.CompletedProcess[str] | None") -> None:
        if status is not None:
            status.finish(step, res)

    def _skip(step: str) -> None:
        if status is not None:
            status.skip(step)

    def _finalize() -> None:
        if status is not None:
            status.finalize()

    results: list[subprocess.CompletedProcess[str] | None] = []

    if not args.dryrun:
        _start('board_reset')
        reset_res = run_and_capture(['tt-smi', '-r'], show_output=args.show_output,
                                    dram_drop_guard=dram_guard, status=status, step_name='board_reset')
        results.append(reset_res)
        _finish('board_reset', reset_res)

    # Pre-flight (opt-in via --fail-check): run the command once WITHOUT tracy so a trivially-broken
    # command aborts before the expensive 3-pass Tracy capture. Off by default — tracy now runs with
    # --check-exit-code, so a failed/timed-out capture pass already trips anyfails() and aborts.
    # Appending to results makes the pass guards below and the merge short-circuit, and main()
    # already exits non-zero when anyfails.
    if not anyfails(results) and args.fail_check:
        _start('failcheck')
        fc = run_failcheck(command, pytest_mode, failcheck_dir, show_output=show_output, dryrun=args.dryrun,
                           log_path=os.path.join(output_dir, 'failcheck_results_typescript.txt'),
                           dram_drop_guard=dram_guard, status=status, step_name='failcheck')
        results.append(fc)
        _finish('failcheck', fc)
        if anyfails(results):
            logger.error('no-tracy fail-check failed -- aborting before the Tracy capture passes. '
                         'Fix the command; see failcheck_results_typescript.txt.')

    if not anyfails(results):
        logger.info('board reset successful, starting profiling runs')
        _start('raw')
        result1 = run_profiler(
            command, raw_dir, pytest_mode, report_name,
            collect_noc_traces=False, collect_perf_counters=False,
            enable_logging=enable_logging, show_output=show_output, dryrun=args.dryrun,
            op_support_count=args.op_support_count,
            no_op_info_cache=args.no_op_info_cache,
            log_path=os.path.join(output_dir, 'raw_results_typescript.txt'),
            dram_drop_guard=dram_guard, status=status, step_name='raw',
        )
        results.append(result1)
        _finish('raw', result1)

    if not args.basic_only:
        if not anyfails(results):
            _start('perf')
            result2 = run_profiler(
                command, perf_dir, pytest_mode, report_name,
                collect_noc_traces=False, collect_perf_counters=True,
                enable_logging=enable_logging, show_output=show_output, dryrun=args.dryrun,
                op_support_count=args.op_support_count,
                no_op_info_cache=args.no_op_info_cache,
                log_path=os.path.join(output_dir, 'perf_results_typescript.txt'),
                dram_drop_guard=dram_guard, status=status, step_name='perf',
            )
            results.append(result2)
            _finish('perf', result2)
        if not anyfails(results):
            _start('trace')
            result3 = run_profiler(
                command, trace_dir, pytest_mode, report_name,
                collect_noc_traces=True, collect_perf_counters=False,
                enable_logging=enable_logging, show_output=show_output, dryrun=args.dryrun,
                op_support_count=args.op_support_count,
                no_op_info_cache=args.no_op_info_cache,
                log_path=os.path.join(output_dir, 'trace_results_typescript.txt'),
                dram_drop_guard=dram_guard, status=status, step_name='trace',
            )
            results.append(result3)
            _finish('trace', result3)

    if args.dryrun:
        return 0

    # On-device merge: produce merged_ops_<RUNID>.csv as the final artifact of a successful
    # full capture, so only it (+ hw_id.json, run_status.json, STATUS.txt) need be copied off the board. The
    # per-board DRAM peak BW is resolved from an INTERNAL interim table keyed on the
    # tt-smi board_type -- no CLI value needed. Needs raw+perf+trace, so skipped for
    # --basic-only / any failed pass; skipped (non-fatal) when the board is unknown
    # or tt-smi is unavailable (e.g. off-device) -- merge manually in that case.
    # Runs BEFORE cleanup so the source CSVs still exist.
    if args.basic_only:
        logger.info('--basic-only: skipping merge (needs raw+perf+trace).')
    elif anyfails(results):
        logger.warning('a profiling pass failed; skipping merge.')
        _skip('merge')
    else:
        board = detect_board_type()
        bw = resolve_interim_dram_peak_bw(board) if board else None
        if bw is None:
            logger.warning(
                f'skipping on-device merge: could not resolve DRAM peak BW for board_type={board!r} '
                f'(known prefixes: {sorted(_INTERIM_BOARD_DRAM_PEAK_BW_GBPS)}). Merge manually: '
                f'ops_perf_three_csv_merge.py --input-dir {output_dir} --dram-peak-bw-gbps <gbps>.'
            )
            _skip('merge')
        else:
            logger.info(f'on-device merge: board_type={board} -> interim DRAM peak BW {bw} GB/s (PLACEHOLDER)')
            _start('merge')
            merge_res = run_merge(output_dir, bw, show_output=show_output, merge_variant=args.merge_variant,
                                  log_path=os.path.join(output_dir, 'merge_results_typescript.txt'),
                                  dram_drop_guard=dram_guard, status=status, step_name='merge')
            results.append(merge_res)
            _finish('merge', merge_res)

    if args.cleanup:
        cleanup_directories(output_dir)

    # Propagate failures (board reset, any profiling pass, or the merge) as a non-zero
    # exit so automation/presets can detect them. Per-pass result files are already
    # written above, so a failed run still leaves its captured output on disk.
    _finalize()
    if anyfails(results):
        logger.error('one or more steps failed; exiting non-zero (see per-pass results files).')
        return 1
    return 0


if __name__ == '__main__':
    exit(main())
