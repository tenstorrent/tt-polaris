# SPEC: `ops_perf_three_csv_merge.py`

Merge **exactly three** TT-Metal-style ops perf CSV files (or two, without the noctrace CSV, under `--allow-missing-noctrace`) found under a root directory, optionally filter multi-iteration runs to a single representative iteration, validate, merge row-by-row on shared keys, and write **one** output CSV.

- **Implementation:** stdlib **`csv`** only (no pandas; openpyxl not used). **`loguru`**: non-fatal **warnings** (utilization out-of-range, fpuutil vs vanilla per-row duration tolerance, iteration-arg mismatch) and **info** aggregate summary after a successful merge.
- **Exit code:** `0` success, `1` on any validation error (message on stderr).

## CLI

| Flag | Required | Default | Meaning |
|------|----------|---------|---------|
| `--input-dir` | yes | — | Root directory; recursively discovers exactly three `ops_perf_results_*.csv` files, or two with `--allow-missing-noctrace` (falls back to `*.csv` when no name-matched files found). |
| `--dram-peak-bw-gbps` | yes | — | Peak DRAM bandwidth in **GB/s** (MemTraffic). |
| `--output` | no | `<input-dir>/merged_ops.csv` | Output CSV path. The filename does not encode iteration metadata (kept runtime-info-invariant for consistency across workloads). |
| `--duration-rel-tol` | no | `0.05` | Relative tolerance for fpu vs vanilla `DEVICE KERNEL DURATION [ns]`. |
| `--encoding` | no | `utf-8` | Input/output text encoding. |
| `--num-iterations` | no | — | Expected count of complete iterations (e.g. `run_device_perf`'s `num_iterations`). Mismatch with auto-detect emits a warning. |
| `--ops-per-iteration` | no | — | Per-iteration op count (model size). When provided, bypasses marker-based auto-detection and uses fixed-size chunking. Errors if the total row count is not evenly divisible by this value. |
| `--measured-iteration-indices` | no | — | Comma-separated 0-based iteration indices to consider as candidates (e.g. `3,4,5` to skip warmup/sync). When omitted, all complete iterations are candidates. |
| `--select-iteration` | no | `median` | How to choose among candidate iterations: `median` (lower-middle by total kernel ns), `min`, `max`, `first`, `last`. |
| `--allow-missing-noctrace` | no | off | Accept a vanilla CSV and an fpu CSV without the noctrace CSV. See **Missing noctrace CSV**. |

## Utilization and cell parsing

- Utilization columns: `DRAM BW UTIL (%)`, `FPU Util Median (%)`, `SFPU Util Median (%)`, `NOC UTIL (%)`, `MULTICAST NOC UTIL (%)`, `ETH BW UTIL (%)`, `NPE CONG IMPACT (%)`.
- Treat as **percent**; if a cell parses as a **finite float** and is outside `[0, 100]`, a **`loguru` `warning`** is emitted but the merge continues. TT-Metal sometimes reports values above 100 for NoC/DRAM util due to multicast overcount; these are valid hardware measurements and should not block the pipeline.
- Empty, `-`, or non-numeric: **skip** the range check for that cell.
- Strip whitespace; whitespace-only is blank. **Zero** means parses to **0.0** after strip.

## Input row filtering (signpost markers)

Immediately after reading each CSV (before validation, classification, iteration detection, and join-key parsing), rows whose `OP TYPE` is **`signpost`** (case-insensitive) are dropped. These are Tracy signpost markers (e.g. the `start`/`stop` pair a perf test wraps around its profiled region), emitted by TT-Metal's ops-log post-processor with an empty `GLOBAL CALL COUNT`; `signpost` is the only non-op `OP TYPE` it writes. They carry no device-op data, so removing them keeps every downstream step operating on real ops only (and prevents the empty `GLOBAL CALL COUNT` from failing join-key parsing). A **`loguru` `info`** line records the count dropped per file. The same filter rule is applied to all three inputs; agreement of the three files after filtering is still enforced separately by the existing row-count-equality and identical-join-key-set checks (see *Join keys and sorting*), not by this step.

## Iteration filtering

TT-Metal device-perf harnesses (e.g. `run_device_perf(num_iterations=N)`) may concatenate multiple iterations of the same model in one CSV. This script reduces the input to a single iteration so downstream tools (e.g. `tt_perf_mapper`) consume one row per op.

- **Boundary detection:** the first row's `OP CODE` is the model entry marker; iteration starts are subsequent rows with the same `OP CODE`. All iterations must be the same size. If sizes differ, the merge stops with an error and the user should split inputs manually or specify `--ops-per-iteration`.
- **Completeness:** an iteration is *complete* if ≥95% of its rows have a `DEVICE KERNEL DURATION [ns]` that parses as a finite float (i.e. non-empty, non-`-`, numeric). Sparse sync/teardown iterations are excluded from the default candidate pool.
- **Selection:** by default, the median (lower-middle) of complete iterations' total kernel ns. `--select-iteration` and `--measured-iteration-indices` override.
- **Synchronization across variants:** all three input CSVs must have the same total row count (same iteration layout). The script detects iterations on the **vanilla** CSV and applies the chosen `[start, end)` row range to all three.
- **CLI validation:** `--ops-per-iteration` is authoritative — it bypasses marker-based auto-detection entirely and its value prevails. For `--num-iterations` and `--measured-iteration-indices`, auto-detect still runs; any mismatch emits a **`loguru` `warning`** but does not halt the merge, and the auto-detected values prevail when they conflict.
- **Single-iteration inputs:** when only one iteration is detected, the filter is a no-op (the entire CSV is passed through unchanged).

## File discovery and classification (disjoint, first match)

Process each CSV independently, in order **(1) noctrace → (2) fpu → (3) vanilla**:

1. **noctrace:** At least one row where `DRAM BW UTIL (%)` for classification is **non-zero** (missing/empty treated as **0**).
2. **fpu:** Else, at least one row with **`FPU Util Median (%)`** or **`SFPU Util Median (%)`** parseable as a finite float; and **every** row has DRAM blank/whitespace or parses to **0.0**. (`PM FPU UTIL (%)` is **not** used as a classification signal — it is present with non-zero values in all three CSV types and would cause false positives.)
3. **vanilla:** Else, every row DRAM blank/whitespace or **0.0**, and FPU/SFPU blank or **0.0**.

Require **exactly one** file in each class. The noctrace header must be an **ordered subsequence** of the **fpu** header — every noctrace column appears in the fpu header in the same relative order, with the fpu-only columns (the perf-counter analysis block) spliced in **anywhere**, not necessarily at the end. (tt-metal interleaves the analysis columns *before* trailing op-type columns such as `TT_DNN_DEVICE_OP_TT_HOST_FUNC [ns]` that all passes share, so a strict prefix relationship does not hold.) **Vanilla** must use the **same column order** as noctrace but may **omit** `FPU Util Median (%)` and/or `SFPU Util Median (%)` only (no other omissions or extras). Required input columns for every file: join keys, `DEVICE KERNEL DURATION [ns]`, `DRAM BW UTIL (%)`, the four NOC-related utilization columns. The **fpu** CSV must include **`FPU Util Median (%)`** or **`SFPU Util Median (%)`** (the same columns that drove its fpu classification). `PM FPU UTIL (%)` is used only as a per-row fallback when writing the output `FPU Util Median (%)_fpuutil` cell, not for classification or header validation. **`DEVICE KERNEL DURATION [ms]`** may be absent in inputs (derived from ns for outputs).

**Ambiguous triple “vanilla”:** If all three files would classify as vanilla (no row has non-zero `DRAM BW UTIL (%)` for step (1), and no file has parseable `FPU Util Median (%)` / `SFPU Util Median (%)` for step (2)), roles are assigned by **sorted full file path**: first → noctrace, second → fpu, third → vanilla. A **`loguru` `warning`** records the mapping.

## Missing noctrace CSV (`--allow-missing-noctrace`)

Long captures can lose the NOC-trace pass while the vanilla and fpu passes succeed. With `--allow-missing-noctrace`:

- Discovery accepts **two** CSV files. Classification must yield exactly one **fpu** and one **vanilla** file. Three files merge exactly as without the flag (byte-identical output).
- Without the flag, two files are a hard error. The message names the flag.
- The vanilla header stands in for the noctrace header: the fpu header must contain it as an ordered subsequence, and fpu-only columns are those not in the vanilla header. In tt-metal captures the noctrace and vanilla headers are identical, so the base and fpu-only output columns match a three-file merge.
- The output **omits** every column that only the noctrace CSV supplies: the whole `_noctrace` block (output step 4) and `MemUtil/Raw`. These columns are omitted, never written blank, because consumers read a blank cell as `0`. All other columns, rows and cells equal those of the three-file merge of the same vanilla and fpu CSVs.
- One **`loguru` `warning`** lists the omitted columns. The output cannot build a LUT: `tt_perf_mapper.py` requires the DRAM, NOC and NPE columns.

## Join keys and sorting

- Keys: `GLOBAL CALL COUNT` (integer), `OP CODE`, `OP TYPE`.
- Sort: ascending `GLOBAL CALL COUNT`, then case-sensitive `OP CODE`, `OP TYPE`.
- No duplicate keys within a file; key sets must be **identical** across all three files.
- After sort, row `i` keys must match across files; row counts must match.

## Duration check (fpuutil vs vanilla)

For each row, parse `DEVICE KERNEL DURATION [ns]` from **fpu** and **vanilla** as finite floats. If either is missing or non-finite, that remains a **hard error**. If both are `0`, pass. Otherwise compute  
`rel_diff = abs(fpu_ns - vanilla_ns) / max(|fpu_ns|, |vanilla_ns|, 1e-12)`.

- If `rel_diff <= duration-rel-tol`: pass.
- If `rel_diff > duration-rel-tol`: emit a **`loguru` `warning`** (row index, keys, ns values, `rel_diff`, tolerance) and **continue** the merge — **do not** exit with failure. Output still uses **vanilla** kernel ns/ms as the canonical unsuffixed duration columns; the **fpu** block still carries the fpu file’s ns/ms and util columns.

## Aggregate kernel duration summary (after successful write)

After the output CSV is written, **`loguru` `info`** logs:

- **fpuutil total:** sum of `DEVICE KERNEL DURATION [ns]` over all merged rows from the **fpu** file (same row order as output).
- **vanilla total:** sum of the same column from the **vanilla** file.
- **Relative difference:** if both totals are `0`, report `0`; else  
  `abs(fpu_total - vanilla_total) / max(abs(fpu_total), abs(vanilla_total), 1e-12)`  
  (same scale as the per-row check).

## MemTraffic (noctrace row)

`MemTraffic_bytes = round((U/100) * (dram_peak_bw_gbps * 1e9) * (ns/1e9))` where `U` is `DRAM BW UTIL (%)` when that cell parses as a finite float; if it is missing, blank, or non-finite, use **`U = 0`** for that row (one **`loguru` `warning`** per merge if any row needed this). `ns` is noctrace `DEVICE KERNEL DURATION [ns]` (finite float; hard error if not).

## Output columns (order)

1. Join keys (once).
2. **Vanilla base:** all vanilla columns except join keys and **overlay source** names (union of noctrace/fpu overlay source names from the spec, **excluding** `MemTraffic`, which is not an input column).
3. Unsuffixed `DEVICE KERNEL DURATION [ns]` and `[ms]` from **vanilla** (`[ms]` derived as `ns/1e6`).
4. **Noctrace** block: same logical columns with suffix `_noctrace`, including `MemTraffic_noctrace` and NOC/DRAM util copies.
5. **Fpu** block: kernel ns/ms and FPU/SFPU medians with suffix `_fpuutil`. The `FPU Util Median (%)_fpuutil` cell uses the fpu file’s `FPU Util Median (%)` when present, else **`PM FPU UTIL (%)`** from the fpu row.
6. **Fpu-only columns:** any columns present on the **fpu** CSV that are **not** in the noctrace header (identified by name, not position — the analysis block may be spliced before shared trailing columns) are copied unchanged from the fpu row, in fpu header order (e.g. extended NOC analyzer metrics). Values are **not** subject to the fixed-column `[0,100]` utilization rule (those metrics may exceed 100%).
7. **Derived overhead ratios** (appended last): `FPUUtil/Raw` = `DEVICE KERNEL DURATION [ms]_fpuutil / DEVICE KERNEL DURATION [ms]` and `MemUtil/Raw` = `DEVICE KERNEL DURATION [ms]_noctrace / DEVICE KERNEL DURATION [ms]`. Each cell is **blank** when the vanilla `[ms]` is `0` (no division). These quantify per-op data-collection overhead.

## Edge cases

- Use Python `csv` module; `newline=""` for I/O.
- Missing required columns: hard error with path and name.
- Non-numeric where required (e.g. kernel ns for checks): error with row index.
