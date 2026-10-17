# Configuration file examples

A Polaris run is wired together from four YAML files. This page shows an annotated,
minimal excerpt of each. (These are *excerpts* — see the files under `config/` for the
full, current content.)

## Workload registry — `config/all_workloads.yaml`

Declares each workload, its frontend (`api`), the Python module implementing it, shared
`params`, and named size `instances`.

```yaml
workloads:
  - api: TTSIM                 # frontend: TTSIM | TTNN | ONNX
    name: basic_llm
    basedir: workloads
    module: BasicLLM.py        # implements the workload
    params:
      vocab_sz: 50257
      norm_type: layer
      bs: 1
    instances:
      gpt_nano  : { nL:  3, nH:  3, dE:   48, nW:   32}   # tiny — good for smoke runs
      gpt_micro : { nL:  4, nH:  4, dE:  128, nW:   32}
      gpt1      : { nL: 12, nH: 12, dE:  768, nW:  512}
```

Each `instances` entry is a preset (`nL` layers, `nH` heads, `dE` embedding dim, `nW`
sequence/window). `--filterwli gpt_nano` selects one.

## Architecture spec — `config/tt_bh.yaml` / `config/tt_wh.yaml`

Two top-level keys: `ipblocks` (reusable compute/memory block definitions) and `packages`
(chip products and their SKU `instances`).

```yaml
packages:
  - name: Blackhole
    instances:
      - name: p100a
        operator_lookup_file: lfc://hlm-lut/bh_p100a_lut_v5.yaml   # per-SKU perf LUT
        compute_grid_size: [12, 10]
        ipgroups:
          - ipname: tensix                 # compute block from `ipblocks`
            iptype: compute
            num_units: 120                 # 120 Tensix cores
            ramp_penalty: 100
            ip_overrides:
              pipes.matrix.freq_MHz: 1350
              pipes.vector.freq_MHz: 1350
          - ipname: gddr6_bh               # memory block
            iptype: memory
            num_units: 7
```

The `operator_lookup_file` is how a per-SKU performance LUT is attached (see
[YAML_MASTER_FORMAT](../YAML_MASTER_FORMAT.md)). `ipblocks` (not shown) define the
per-instruction throughput tables the perf model reads.

## Graph-mapping spec — `config/wl2archmapping.yaml`

Post-processing rules applied to the workload graph after it is built, before execution.

```yaml
op_data_type_spec:                 # dtype assignment
  global_type: int8
  override:
    dropout: int32

op_removal_spec:                   # ops dropped from the graph
  - Dropout
  - Identity
  - Constant

op_fusion_spec:                    # op-sequence patterns fused into one node
  - [Add, LayerNormalization, Matmul, Add, Matmul, Add, Gelu]
  - [Conv, BatchNormalization, Relu]
  - [Matmul, Mul, Softmax]

op_rsrc_spec:                      # compute-pipe assignment (matrix vs vector) — not shown
op_split_spec:                     # arch-aware op splitting (e.g. lm_head) — not shown
```

## Run config — `config/runcfg_*.yaml`

Bundles the three files above plus study parameters for a run. Used with `polproj.py`.

```yaml
title: Run all workloads
study: runall
odir: __runall
wlspec: config/all_workloads.yaml       # → workload registry
archspec: config/all_archs.yaml         # → architecture spec
wlmapspec: config/wl2archmapping.yaml   # → graph-mapping spec
filterwlg: TTSIM                        # workload-group filter
dump_stats_csv: true                    # emit the per-op stats CSV
```
