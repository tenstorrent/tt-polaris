# Polaris Documentation

This folder holds the documentation for **Polaris** (`ttsim`), Tenstorrent's
high-level roofline performance simulator for AI workloads.

## Start here

- **[Introduction to Polaris](INTRODUCTION.md)** — the guided tour: what Polaris
  is, why it exists, how its pieces fit together, and where to go next. **Read
  this first** if you're new.
- **[Top-level README](../README.md)** — installation, environment setup, and
  quick-start commands.

## Using Polaris

- **[User Guide](user_guide.md)** — full CLI usage, configuration files, output
  formats, and debugging.

## Architecture & internals

- **[Overview](overview.md)** — the internals reference: `WorkloadGraph`, `SimOp`,
  `SimTensor`, `Device`, the frontends, config, and stats.
- **[Shape Inference](shape_inference.md)** — per-operator shape-inference rules
  and broadcasting.

## Frontends & authoring workloads

- **[Functional API](functional.md)** — the TTSim functional frontend for
  authoring models natively.
- **[PyTorch → TTSim](torch2ttsim.md)** — porting a PyTorch model to the
  functional API.
- **[TTNN Shim](ttnn_shims_README.md)** — the drop-in `ttnn` replacement:
  layout, sharding, and transformer ops the shim provides.
- **[TTNN Workload Flow](TTNN_WORKLOAD_FLOW.md)** — quick reference for the
  parameter flow and code locations of a TTNN workload.

## Performance lookup tables (LUTs)

- **[YAML Master Format](YAML_MASTER_FORMAT.md)** — the LUT wire format.
- **[Lookup Table Master](tools/perf_lookup/LOOKUP_TABLE_MASTER.md)** — loading
  and validating operator perf LUTs.
- **[Operator LUT via LFC](tools/perf_lookup/OPERATOR_LUT_LFC.md)** — how LUT
  files are stored and fetched through the large-file cache.
- **[Three-CSV Merge Spec](SPEC_ops_perf_three_csv_merge.md)** — merging the
  profiler's triple-CSV output into a single per-op perf CSV.
- **[Trace+Replay Merge Spec](SPEC_ops_perf_trace_replay_merge.md)** — the same
  merge for trace+replay captures, reduced to one step.

## Hardware correlation

- **[Correlation](tools/ttsi_corr/README_correlation.md)** — comparing Polaris
  projections against hardware performance.
- **[Reference Data Validation](tools/ttsi_corr/README_reference_data_validation.md)**
  — validating reference TT-Metal perf data.

## CI & infrastructure

- **[GitHub Actions Architecture](README_github_actions_architecture.md)** — CI
  workflow design.
- **[Dynamic Badges](README_dynamic_badges.md)** — status-badge setup.
- **[Large File Cache — usage](tools/ci/large_file_cache_usage.md)** /
  **[downloader guide](tools/ci/lfc_downloader_user_guide.md)** — LFC setup and
  use.
- Additional CI utilities under [`tools/ci/`](tools/ci/).
