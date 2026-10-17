# Qwen3.5-35B-A3B — Polaris TTSIM Workload

Perf simulation workload for **Qwen/Qwen3.5-35B-A3B** (35B total / ~3B active,
hybrid linear + full attention, sparse MoE), built on the TTSIM frontend.

## Architecture (pinned from HF `config.json`, `model_type: qwen3_5_moe_text`)

| Component | Value |
|---|---|
| Layers | 40 = 10 x (3 linear-attention + 1 full-attention), `full_attention_interval: 4` |
| Hidden size | 2048 |
| Vocab (padded) | 248,320 (untied embedding / LM head) |
| Full attention | GQA 16 Q-heads / 2 KV-heads, head_dim 256, gated output (`attn_output_gate: true`), partial RoPE 64/256, mrope interleaved [11,11,10], theta 1e7 |
| Linear attention | Gated DeltaNet: 16 key-heads x 128, 32 value-heads x 128, depthwise causal conv (kernel 4), fp32 SSM state |
| MoE (every layer) | 256 experts, 8 routed + 1 shared, expert intermediate 512, softmax router + top-k with normalized probs |
| Norm / act | RMSNorm (eps 1e-6), SiLU |

## Files

- `qwen35_model.py` — the TTSIM model (`Qwen35Moe` entry class + `RMSNorm`, `GatedDeltaNet`,
  `GatedGQAAttention`, `SparseMoE`, `MoEExpert`, `Qwen35HybridLayer`)
- `test_model.py` — standalone smoke test (`python workloads/qwen3_5/test_model.py`)
- `tests/test_workloads/test_qwen35.py` (repo root) — pytest unit tests

## Usage

```bash
# full pipeline (workload already registered in config/all_workloads.yaml)
python polaris.py \
  --archspec config/all_archs.yaml \
  --wlspec  config/all_workloads.yaml \
  --wlmapspec config/wl2archmapping.yaml \
  --filterwlg TTSIM --filterwl qwen3_5 \
  --filterarch Q1_A1 \
  --study qwen35 --odir __qwen35
```

Registered instances:

| Instance | Config |
|---|---|
| `qwen35_35b_a3b_decode` | bs=1, seq_len=1 (single decode step) |
| `qwen35_35b_a3b_prefill` | bs=1, seq_len=128 (prefill chunk) |
| `qwen35_tiny` | reduced config for CI/debug |

## Op decomposition

- **GDN layer** (per timestep, per v-head): sigmoid gates `a`/`beta`, two matvecs
  (`q·S`, `q·k`), eltwise delta-rule state update `S = a*S + k⊗(beta*(v - q·S))`,
  closed-form output `y = a*(q·S) + beta*(q·k)*(v - q·S)`; depthwise conv via
  `F.conv1d`; output RMSNormGated + `out_proj`. Follows the
  `workloads/mamba2` scan-loop pattern.
- **Full attention**: standard GQA decomposition (WQ/WK/WV, RoPE mix, scores matmul,
  causal mask, softmax, context matmul), per-head sigmoid output gate, `o_proj`.
- **MoE**: router linear + softmax + `F.topk(k=8)` + normalization, 8 active
  expert FFNs (SwiGLU gate/up/down) weighted-combined, plus 1 shared expert.
- **Layer multiplicity**: only one linear block and one full block are executed;
  op `repeat_count` scales them x30 / x10 (same pattern as `workloads/BasicLLM.py`).

## Modeling abstractions (perf-sim, not functional)

1. **Active-expert MoE**: the 8 selected experts are applied to the routed
   activations; expert *dispatch/combine* data movement is not modeled. For bs=1
   decode this is exact for compute; prefill aggregates top-8 slots across all
   tokens (slight overcount when routing is sparse per token).
2. **RoPE** is approximated by elementwise cos/sin mixing over the full head_dim
   (the real model rotates 64/256 dims). Same op types, negligible cost either way.
3. **KV cache** is not modeled: full attention runs over the current sequence
   window (decode = 1 token step).
4. **MTP / PLE** (multi-token-prediction head, n-gram path) are not part of the
   main inference loop and are not simulated.

## Validation

- Analytical parameter count: **34,724,128,768** (~0.07% off the published 34.7B).
- Active-parameter reads per decode step (Q1_A1 projection): **~3.37B**, tracking
  the advertised ~3B activated parameters (includes LM head + embedding).
- Graph builds for decode (seq_len=1), prefill (128), and a tiny CI config;
  `polaris.py` end-to-end projection runs clean on the Q1_A1 (Quasar-class) device.