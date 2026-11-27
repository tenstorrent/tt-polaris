# tt_transformers_dualmode — provenance & port decisions

TT-Polaris dual-mode port of the generic `tt_transformers` model, used by the
`llama3_dualmode` workload (PR2 of GH #466). Runs both on real silicon (HW path,
real `ttnn`) and analytically through the TT-Polaris shim (`ttsim.front.ttnn`).

## Why a NEW directory — copy-and-modify, NOT modify-in-place, NOT shared

These components are **copied** (from the shim-only `workloads/ttnn/tt_transformers/`,
or ported fresh from tt-metal — see below) and **modified here**. The existing
`workloads/ttnn/tt_transformers/` is left **untouched**.

Reason: `workloads/ttnn/tt_transformers/` is a **live dependency** of multiple
registered workloads — `llama3_prefill` / `llama3_decode` (via `workloads/ttnn/llama3/`)
**and** the mixtral workloads (`workloads/ttnn/mixtral/`) import its `model_config`,
`model`, `rope`, `decoder`, etc. So:

- **Modifying it in place** would break llama3 *and* mixtral. ✗
- **Sharing** (importing those components unchanged) is unsafe: dual-mode-ification
  touches nearly every component (add the IS_POLARIS / HW branch + the current op
  sequence), and the moment a shared component changes, those consumers break. ✗
- **Copy-and-modify into this dir** → existing consumers untouched, zero breakage. ✓

Cost is duplication, but it is not throwaway: `tt_transformers/` must stay anyway
(mixtral depends on it). After the dual-mode llama3 is verified, the shim-only
`llama3_*` workloads *may* be retired, but `tt_transformers/` remains for mixtral.

## Hybrid sourcing (per component stability)

- **Stable components** — `model_config` (config-only + dummy-weights), `embedding`,
  `rmsnorm`, `mlp`: **based on** the shim-only copies, then **audited against current
  tt-metal** (upstream `models/tt_transformers/tt/`) and made dual-mode. The
  shim-only base can be stale — e.g. `model_config` had a wrong hardcoded
  `qkv_size=5120` (should be `head_dim*(2*n_kv_heads+n_heads)=6144`) and was missing
  the MLP `hidden_dim=14336`; both fixed here per tt-metal `model_config.py:660`.
- **Changed components** — `rope`, `attention` (keystone), `decoder`: also based on the
  shim-only copies, but **reworked against current tt-metal** rather than merely audited,
  because they carry the new op sequence (rotary_embedding_llama (_fused_qk),
  paged_fill_cache, paged_fused_update_cache, nlp_create/concat_heads_decode,
  prefill/decode SDPA) that must match the HW capture. The difference from the stable
  components is how far each was reworked, not where it came from; each file's docstring
  records its own basis. These use the new shim ops added in #468.

## Dual-mode contract

```python
IS_POLARIS = os.getenv('IRD_ARCH_NAME', '') == ''
if IS_POLARIS:
    import ttsim.front.ttnn as ttnn          # analytical shim
else:
    import ttnn                              # real HW
# torch is HW-only: gate inside `if not IS_POLARIS:`, tag `# type: ignore[import-not-found]`
```

## Model / weight consumption — config-only, no HF checkpoint

The TT-Polaris path never reads an HF checkpoint. Config comes from a TT-Polaris-side
`ModelArgs` (llama3-8B params, audited vs tt-metal); weights are fabricated as
shape-correct dummies (`dummy_weights=True`, `ttnn._rand`/`ttnn.zeros`). No
`safetensors`/`transformers`/`huggingface_hub` on the TT-Polaris path.

## Config pins (so only the captured path runs)

`rope_type=llama3` (→ RotarySetup), `paged_attention=1`, `num_devices=1` (single-chip,
no CCL), `use_prefetcher=True`, no chunked prefill.

## HW capture

"The capture" in this directory's docstrings and comments is the tt-metal device-profiler
capture of upstream llama3 at batch 1. The presets in
[`tools/si_profiling_helpers/presets/llama3/`](../../../tools/si_profiling_helpers/presets/llama3/)
produce it: `run-llama3ref-prefill-b1-{bh,wh}.sh` and `run-llama3ref-decode-b1-{bh,wh}.sh`
(see that directory's README). The "BH prefill capture" is the output of
`run-llama3ref-prefill-b1-bh.sh`. An op or row number in a comment (for example
`capture ops 27-28`) is the op's position in that capture's per-op CSV.

## Per-file convention

Every ported file's module docstring records: its basis — either the tt-metal source path it
mirrors, or the shim-only file under `workloads/ttnn/tt_transformers/` it was made dual-mode
from — and any audit fixes vs that base. `ccl.py` is the exception: nothing is ported there,
and its docstring records what is deliberately left out and why.

## References

- Each ported file's module docstring names what it is based on: the tt-metal source path it
  mirrors, or — for the files made dual-mode from the shim-only port — that file under
  `workloads/ttnn/tt_transformers/`. See "Per-file convention" above, including the one
  file that ports nothing.
- Upstream reference implementation: `models/tt_transformers` in
  [tenstorrent/tt-metal](https://github.com/tenstorrent/tt-metal).
