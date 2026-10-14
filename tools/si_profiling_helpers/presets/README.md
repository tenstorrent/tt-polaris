# SI-profiling preset run-scripts

Thin wrappers around [`run-ttnn-profiler.py`](../run-ttnn-profiler.py) that capture per-op
device profiles for specific workloads on real silicon. Each script pins the workload command,
a dated `RUNID`, and the profiler flags so a capture is a single `bash` invocation.

## Workspace layout

These scripts are **run from the tt-metal checkout root**, with `si_profiling_helpers/`
(this directory's parent) and `tt-npe/` as siblings of `tt-metal/`:

```
workspace/
├── si_profiling_helpers/        # rsynced from polaris tools/si_profiling_helpers/
│   ├── run-ttnn-profiler.py
│   └── presets/                 # these scripts
├── tt-metal/                    # runs are launched from here (CWD)
│   └── workloads/               # polaris workloads/ rsynced in (NOT committed to tt-metal)
└── tt-npe/
```

The relative paths inside each script (`../si_profiling_helpers/...`, `workloads/ttnn/...`,
`models/demos/...`) all resolve from the **tt-metal root** as CWD.

## Prerequisites (per shell session)

1. **rsync the inputs** onto the machine: `si_profiling_helpers/` (sibling of `tt-metal/`)
   and the relevant polaris `workloads/` content into `tt-metal/workloads/`.
2. **Environment:** from `tt-metal/`, `source ../si_profiling_helpers/setup-step-2-new-login.sh`
   (activates the venv, sets `TT_METAL_HOME`/`PYTHONPATH`, sources tt-npe — required for NoC traces).
3. **`IRD_ARCH_NAME`** is set automatically by the IRD environment to the board's arch
   (`wormhole_b0` on WH, `blackhole` on BH). You don't set it manually, but it is **essential**:
   it must be present and non-empty, since an empty value would select the Polaris-shim path
   instead of hardware. Verify with `echo $IRD_ARCH_NAME` if a run behaves unexpectedly.
4. **Board reset is automatic:** `run-ttnn-profiler.py` runs `tt-smi -r` at the start of each
   capture and aborts if it fails — no manual reset needed. Reset by hand (`tt-smi -r`) only to
   recover a wedged board outside a capture.

## Running

```bash
cd ~/work/tt-metal            # or wherever the tt-metal checkout lives
bash ../si_profiling_helpers/presets/vit/run-vitdual-wh.sh
```

The profiler runs the command three times (basic / perf-counters / noc-traces) under Tracy,
producing `raw/`, `perf/`, and `trace/` CSVs under the `RUNID` output directory.

> **Merging:** consolidating the three CSVs into a single `merged_ops_<RUNID>.csv` (RUNID =
> the output-dir name) uses one of two reducers, selected by the profiler wrapper's
> `--merge-variant`: **`iterative`** (`ops_perf_three_csv_merge.py`, the default — for
> multi-iteration workloads like **VGG UNet**) or **`trace_replay`**
> (`ops_perf_trace_replay_merge.py` — for non-iterative trace+replay workloads like **ViT**
> and **llama3**, which have a prologue followed by replayed steps, no iteration loop; it keeps
> one step, including any eager ops that run after each replay).
> The `vit/` and `llama3/` presets pass `--merge-variant trace_replay`; `vggunet/` uses the
> default. Both tools are
> co-located with these presets (one directory up), so they ship with the rsynced bundle and
> the merge runs **on the hardware node** right after capture (the profiler wrapper passes an
> explicit `--output`; the tool's own default is plain `merged_ops.csv`). Only
> `merged_ops_<RUNID>.csv` (plus `hw_id.json`, and the run-status sidecar `run_status.json` / `STATUS.txt`) needs to be copied back. Its only third-party dependency is `loguru`, which is already in
> the tt-metal run environment. (The follow-on compare-vs-refrun step pulls in polaris/ttsim
> and stays on a polaris checkout.)

## Output isolation & known cwd residue

`run-ttnn-profiler.py` pins `TT_METAL_LOGS_PATH` and `TT_METAL_PROFILER_DIR` to each pass's
output dir, so the profiler artifacts, `.logs/`, `inspector/`, and `watcher/` land under the
`RUNID` directory rather than the current working directory. This lets concurrent captures
share one `tt-metal/` checkout without clobbering each other's logs.

**Known residual (expected — do not re-investigate):** a small `generated/` may still appear
in the CWD containing `generated/fabric/` and `generated/test_reports/`. tt-metal anchors
`fabric` to its *runtime root* (which resolves to the CWD here) and writes `test_reports` at a
CWD-relative path; neither honors `TT_METAL_LOGS_PATH` and they cannot be redirected via env
(repointing the runtime root breaks tt-metal's file lookup, and the CWD must stay the tt-metal
root for the relative command paths). These are a few small, idempotent/cosmetic files — safe
to ignore or delete between runs. For strict isolation of *concurrent* captures, give each run
its own tt-metal checkout.

> **Capture with the model's real command — do not add `--disable_trace`.** A refrun must run
> the workload exactly as it does on hardware, including trace capture + replay. tt-metal
> `device-perf` cases that default tracing **on** (e.g. `tt_transformers` `simple_text_demo`)
> repeat ops across capture/replay passes (duplicate `GLOBAL CALL COUNT`, plus empty-duration
> "shadow" rows for some demos); the 3-CSV merge **deduplicates** these via its op-code-period
> iteration detection. Disabling trace would change the measured execution path, so it is not
> used — the dedup happens in tooling, not by altering the command.

## `dual` vs `ref`

| Suffix | Command target | Purpose |
|--------|----------------|---------|
| `dual` | polaris dual-mode test under `workloads/ttnn/<model>/<arch>/` | runs the **single-pass** model entry (the dual-mode files run their real-`ttnn` branch when `IRD_ARCH_NAME` is set) — the per-op capture used for LUT building |
| `ref`  | upstream tt-metal test under `models/demos/...` (run with `--pytest`) | the canonical tt-metal reference run |

The `dual` scripts deliberately use the model's single-pass entry (not its averaged
`device-perf` perf-gate path): Tracy needs one clean execution per profiling mode to collect
per-op timings.

## Arch matching

`*-wh.sh` scripts run **only** on Wormhole nodes; `*-bh.sh` scripts **only** on Blackhole.
Running a script on the wrong board mismatches the captured arch.

Each script enforces this by sourcing `check_arch.sh` and calling `require_arch <arch>` before
doing anything: it aborts immediately (non-zero) if `IRD_ARCH_NAME` is unset (not on a hardware
node) or does not match the script's target arch. Matching is on arch family, so `wormhole_b0`
and minor spelling variants still pass.

## Scripts

| Script | Board | Target |
|--------|-------|--------|
| `vggunet/run-vggunetdual-wh.sh` | WH | polaris dual-mode VGG-UNet |
| `vggunet/run-vggunetdual-bh.sh` | BH | polaris dual-mode VGG-UNet |
| `vggunet/run-vggunetref-wh.sh`  | WH | upstream tt-metal VGG-UNet e2e |
| `vggunet/run-vggunetref-bh.sh`  | BH | upstream tt-metal VGG-UNet e2e |
| `vit/run-vitdual-wh.sh`         | WH | polaris dual-mode ViT |
| `vit/run-vitdual-bh.sh`         | BH | polaris dual-mode ViT |
| `vit/run-vitref-wh.sh`          | WH | upstream tt-metal ViT device-ops |
| `vit/run-vitref-bh.sh`          | BH | upstream tt-metal ViT device-ops |
| `llama3/run-llama3ref-decode-b32-wh.sh`  | WH | upstream tt-metal llama3, batch-32 (decode phase) |
| `llama3/run-llama3ref-decode-b32-bh.sh`  | BH | upstream tt-metal llama3, batch-32 (decode phase) |
| `llama3/run-llama3ref-prefill-b32-wh.sh` | WH | upstream tt-metal llama3, batch-32 (prefill phase) — ⛔ blocked, see below |
| `llama3/run-llama3ref-prefill-b32-bh.sh` | BH | upstream tt-metal llama3, batch-32 (prefill phase) |
| `llama3/run-llama3ref-decode-b1-wh.sh`   | WH | upstream tt-metal llama3, batch-1 (decode phase) — batch-matched to the b1 dual |
| `llama3/run-llama3ref-decode-b1-bh.sh`   | BH | upstream tt-metal llama3, batch-1 (decode phase) — batch-matched to the b1 dual |
| `llama3/run-llama3ref-prefill-b1-wh.sh`  | WH | upstream tt-metal llama3, batch-1 (prefill phase) — ⛔ blocked, see below |
| `llama3/run-llama3ref-prefill-b1-bh.sh`  | BH | upstream tt-metal llama3, batch-1 (prefill phase) |
| `llama3/run-llama3dual-decode-wh.sh`  | WH | polaris dual-mode llama3, batch-1 (decode phase) |
| `llama3/run-llama3dual-decode-bh.sh`  | BH | polaris dual-mode llama3, batch-1 (decode phase) |
| `llama3/run-llama3dual-prefill-wh.sh` | WH | polaris dual-mode llama3, batch-1 (prefill phase) — ⛔ blocked, see below |
| `llama3/run-llama3dual-prefill-bh.sh` | BH | polaris dual-mode llama3, batch-1 (prefill phase) |

`check_arch.sh` is a sourced helper, not a run-script (see "Arch matching").

## llama3 reference command & capture matrix

The llama3 `ref` presets run the upstream tt-metal demo with the **finalized reference command**:

```
models/tt_transformers/demo/simple_text_demo.py -k "performance and batch-32" --mode {decode,prefill}
```

`-k "performance and batch-32"` selects the perf case at batch-32 (the primary batch for **both**
p100a and n150); `--mode` splits the run into its decode and prefill phases for per-phase LUT prep.
**Every other knob is left at the marked case's default** — no `--num_layers`,
`--max_generated_tokens`, `--batch_size`, `--paged_attention`, `--use_prefetcher`, or seq_len. The
reference must run the *complete* command; earlier `--num_layers 1 --max_generated_tokens 1` trial
limits have been dropped and must not be re-added.

> **Why the scripts wrap the `-k` value in nested quotes** (`-k '"performance and batch-32"'`): tracy's
> report mode re-joins its argv with spaces and re-runs the command under `shell=True`
> (tt-metal `tools/tracy/__main__.py`), which would split the bare words of a spaced `-k` expression
> (`ERROR: file or directory not found: and`). The nested quotes keep the expression as one literal
> token through both `shlex.split` (in `run-ttnn-profiler.py`) and tracy's shell re-parse. Don't
> "simplify" them away.

The `dual` presets run the migrated Polaris llama3 workload
(`workloads/ttnn/llama3/test_llama3_{decode,prefill}.py`) through its real-`ttnn` branch on hardware,
to validate that dual-mode perf tracks the HW reference. The dual workload runs at **batch-1**
(decode default `bs=1`; prefill only supports `bs=1`), so the `…-b1-…` ref presets
(`-k "performance and batch-1 and not log-probs"`) exist to give a **batch-matched** reference for
that dual comparison; the `…-b32-…` ref presets are the primary LUT reference. The output-dir RUNID
carries the batch tag (`b32` / `b1`) so captures at different batches don't collide.

| Arch | Phase | Full-command capture available? |
|------|-------|---------------------------------|
| BH p100a | decode  | ✅ unblocked |
| BH p100a | prefill | ✅ unblocked |
| WH n150  | decode  | ✅ unblocked |
| WH n150  | prefill | ⛔ **BLOCKED** — the WH prefill phase currently fails NoC-trace capture. This is a NoC-trace tooling issue (raised and owned by the user), **not** worked around by limiting the command. The WH prefill presets (`run-llama3ref-prefill-b32-wh.sh`, `run-llama3ref-prefill-b1-wh.sh`, `run-llama3dual-prefill-wh.sh`) carry a matching block/note; run them only after the issue is resolved, then remove the flag. |
