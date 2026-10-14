# SPEC: `ops_perf_trace_replay_merge.py`

Merge three TT-Metal-style ops perf CSVs from a **non-iterative trace+replay** capture (ViT, llama3 decode/prefill) into one output CSV.

This is the trace+replay sibling of [`ops_perf_three_csv_merge.py`](SPEC_ops_perf_three_csv_merge.md) (the iterative, VGG-style merge). It **reuses that tool's classification, header-validation, join, MemTraffic, duration-check, and output machinery verbatim** — the only difference is the **row-reduction** step. Everything in the iterative spec's *File discovery and classification*, *Join keys and sorting*, *Duration check*, *Aggregate kernel duration summary*, *MemTraffic*, and *Output columns* sections applies unchanged; only *Iteration filtering* is replaced by *Step filtering* below.

- **Implementation:** stdlib **`csv`** only; **`loguru`** for warnings/info. Imports the shared helpers from `ops_perf_three_csv_merge.py` (dual import: package path under pytest/`mypy`, bare name when run as an on-device flat-bundle script).
- **Exit code:** `0` success, `1` on any validation error (message on stderr).

## When to use which tool

| Capture shape | Tool | Reduction |
|---|---|---|
| **Iterative** — `run_device_perf(num_iterations=N)` concatenates equal-size iterations (VGG-style) | `ops_perf_three_csv_merge.py` | select one **iteration** |
| **Trace+replay** — a prologue + N replays of a captured trace (ViT, llama3) | `ops_perf_trace_replay_merge.py` | select one **step** |

The trace+replay tool errors (directing the user to the iterative tool) if the input has no populated `METAL TRACE REPLAY SESSION ID` — i.e. it is not a trace+replay capture. Until the unified pass-classifier is built, tool choice is explicit.

## Definitions

| Term | Meaning |
|---|---|
| replay session | Rows with a non-blank `METAL TRACE REPLAY SESSION ID`. `METAL TRACE ID` identifies the trace. |
| capture row | A row with a blank session id and a blank `DEVICE KERNEL DURATION [ns]` cell. It is recorded while a trace is captured; nothing ran on device. An eager row with a malformed duration is not a capture row; the duration check reports it. |
| eager block | Contiguous rows with a blank session id and a numeric kernel duration. |
| first replayed trace | The `METAL TRACE ID` of the first row that has a session id. |
| prologue | All rows before the first replay session of the first replayed trace. |
| step | One replay session of the first replayed trace, plus every following row up to the start of that trace's next replay session. Capture rows are excluded. |
| step signature | The ordered `OP CODE` sequence of a step. |

## Capture layout, and why the iterative reducer fails

A trace+replay capture is **not** equal-size iterations. It has:

1. **A prologue:** one-time setup ops, an eager compile run of the traced region, and the capture rows of the trace.
2. **Steps:** each replay session of the traced region, followed by whatever runs until the next replay.

Replay sessions of one trace reuse identical `(GLOBAL CALL COUNT, OP CODE, OP TYPE)` join keys; eager ops always receive fresh `GLOBAL CALL COUNT`s. The compile run shares its op sequence with the replays.

Not every op of a model step is traced:

- **llama3 prefill:** only the embeddings and decoder layers are traced. The final norm, LM head and sampling run eagerly after each replay, as an eager block with a blank session id. That block is most of the step's device time.
- **llama3 decode:** sampling is a second trace, replayed right after the first. No eager rows occur between replay sessions.

So a replay session alone is not a complete step, and a blank session id does not mean compile/warmup.

The iterative tool splits rows into equal-size iterations from row 0. The prologue means no iteration size aligns with step boundaries, and the downstream join aborts on a **duplicate join key** (the same op present in several replay sessions).

## Step filtering (replaces *Iteration filtering*)

1. **Split into steps.** The first replayed trace is identified; if no row has a session id, error (not a trace+replay capture). A new step starts at each new replay session of that trace.
2. **Drop the prologue and capture rows.** Rows before the first step, and capture rows inside any step, never reach the output.
3. **Keep the most frequent step signature.** Only steps whose step signature is the most frequent one are candidates. This excludes warmup variants (for example, first steps whose sampling runs extra penalty ops) and steps that contain another trace's compile run. If two or more signatures tie for most frequent, error.
4. **Select one step** on the **vanilla** CSV, then take the same step index in all three variants. The three variants must have the same number of steps, else error.
   - `first` (default) — the first candidate step. In llama3 decode NOC traces, the last replay session under-reports `DRAM BW UTIL (%)` while its kernel time is unchanged; the first candidate carries plausible DRAM values.
   - `last` — the last step. In the llama3 demos the last call is the measured one; warmup and compile replays come before it. If the last step's signature is not the most frequent one, error.
   - `median` — lower-middle candidate by total `DEVICE KERNEL DURATION [ns]`.
   - `min` / `max` — least / greatest candidate total kernel ns.
5. **Result:** each variant is reduced to the selected step's rows, whose join keys are unique — handed unchanged to the shared join.

For a capture with no eager rows between replay sessions (llama3 decode), a step is exactly the rows of one session id across all traces, so step filtering keeps the same rows as selecting that session id.

### Composition with iteration reduction (trace+replay+iterative)

If a future capture is trace+replay **and** iterative (a step that itself contains several concatenated iterations), the selected step's join keys will **not** be unique. In that case the iterative reducer (`reduce_iterative`) is applied *within* the selected step, composing the two reductions. This is on by default; `--no-compose-iteration` turns it into a hard error instead. For the common one-inference-per-step case the keys are already unique, so no iteration reduction runs — and the iterative marker heuristic (and its warning) is never triggered.

## CLI

| Flag | Required | Default | Meaning |
|------|----------|---------|---------|
| `--input-dir` | yes | — | Root directory; recursively discovers exactly three `ops_perf_results_*.csv` files (falls back to `*.csv`). |
| `--dram-peak-bw-gbps` | yes | — | Peak DRAM bandwidth in **GB/s** (MemTraffic). |
| `--output` | no | `<input-dir>/merged_ops.csv` | Output CSV path. |
| `--duration-rel-tol` | no | `0.05` | Relative tolerance for fpu vs vanilla `DEVICE KERNEL DURATION [ns]`. |
| `--encoding` | no | `utf-8` | Input/output text encoding. |
| `--select-session` | no | `first` | Which step to keep: `first`, `last`, `median`, `min`, `max` (see *Step filtering*). The prologue and capture rows are always dropped. |
| `--no-compose-iteration` | no | off | Disable the within-step iteration reducer; a step with duplicate keys becomes a hard error instead. |
| `--ops-per-iteration` | no | — | Forwarded to the composed iteration reducer (trace+replay+iterative only). |
| `--measured-iteration-indices` | no | — | Forwarded to the composed iteration reducer (trace+replay+iterative only). |
| `--select-iteration` | no | `median` | Forwarded to the composed iteration reducer (trace+replay+iterative only). |

## Shared behavior (unchanged from the iterative tool)

Classification (noctrace / fpu / vanilla), header subsequence/omission rules, join keys and sorting, per-row and aggregate duration checks, MemTraffic derivation, and the full output column order are exactly as specified in [`SPEC_ops_perf_three_csv_merge.md`](SPEC_ops_perf_three_csv_merge.md). The reduction runs **after** classification/header validation and **before** the join, so the join always receives one unique-keyed row set per variant.

**Signpost-row filtering** (see the iterative spec's *Input row filtering* section) is likewise inherited: `signpost` marker rows are dropped in the shared `run_merge` immediately after each CSV is read — **before** classification and the step reduction — so a trace+replay capture whose passes are wrapped in Tracy `start`/`stop` signposts is handled identically to an iterative one.
