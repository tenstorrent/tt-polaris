# TTNN shim correlation gates

Reference for the eight opt-in gates in `ttsim/utils/shim_gates.py`.

Each gate turns on one piece of tt-metal behaviour that the TTNN shim does not model
by default. They exist because each one changes the **op sequence or the LUT key** of
every workload that reaches the affected code path, and polaris builds all workload
graphs in one process — so a change that fixes one workload's correlation can silently
move another's.

**All eight default to OFF.** That is the conservative default: an un-opted-in workload
keeps exactly the behaviour it had before the gate existed.

This page is the index. The *semantics* — the two-layer resolution order, why
`Device.__init__` deliberately does not materialise these, and why you must never copy
a `_GATE` value into another module — live in the `ttsim/utils/shim_gates.py` module
docstring, which is the normative source.

---

## The eight gates

| env var | setter / accessor | `Device` attribute |
|---|---|---|
| `TTSIM_PROPAGATE_MEMORY_CONFIG` | `set_memory_config_propagation` / `memory_config_propagation_enabled` | `propagate_memory_config` |
| `TTSIM_TILIZE_BEFORE_MATMUL` | `set_tilize_before_matmul` / `tilize_before_matmul_enabled` | `tilize_before_matmul` |
| `TTSIM_CONV_MATH_FIDELITY_IN_KEY` | `set_conv_math_fidelity_in_key` / `conv_math_fidelity_in_key_enabled` | `conv_math_fidelity_in_key` |
| `TTSIM_PROPAGATE_BCAST_HW_SHAPE` | `set_bcast_hw_shape_propagation` / `bcast_hw_shape_propagation_enabled` | `propagate_bcast_hw_shape` |
| `TTSIM_HALO_OUTPUT_DTYPE_PROMOTION` | `set_halo_output_dtype_promotion` / `halo_output_dtype_promotion_enabled` | `halo_output_dtype_promotion` |
| `TTSIM_CONV_INPUT_RESHARD` | `set_conv_input_reshard` / `conv_input_reshard_enabled` | `conv_input_reshard` |
| `TTSIM_TO_MEMORY_CONFIG_RESHARD` | `set_to_memory_config_reshard` / `to_memory_config_reshard_enabled` | `to_memory_config_reshard` |
| `TTSIM_MOVE_ON_REALLOCATE_HALO_OUTPUT` | `set_move_on_reallocate_halo_output` / `move_on_reallocate_halo_output_enabled` | `move_on_reallocate_halo_output` |

Each env var is read once at module import and accepts `'1'` to enable; anything else
(including unset) is OFF.

### What each one does

- **`memory_config_propagation`** — give conv/matmul outputs the memory config implied
  by their `Conv2dConfig.shard_layout`, so model code can meaningfully ask
  `out.memory_config()`. Without it that call returns `None` and comparisons such as
  ResNet-50's bottleneck `if ds_out.memory_config() != out.memory_config()` cannot be
  evaluated.
- **`tilize_before_matmul`** — emit the ROW_MAJOR → TILE conversion ahead of a conv
  that lowers to a 1×1 matmul.
- **`conv_math_fidelity_in_key`** — carry `compute_config`'s math fidelity into the
  conv / 1×1-matmul op attrs, so it reaches the LUT key.
- **`bcast_hw_shape_propagation`** — propagate NHWC-flat `hw_shape` across elementwise
  binaries. Reached by all three front ends (`bidir_bcast` is the registered sinf for
  19 `ai.onnx` optypes), which is why it is gated rather than unconditional.
- **`halo_output_dtype_promotion`** — model `HaloDeviceOperation`'s output-dtype
  promotion.
- **`conv_input_reshard`** — emit the conv **input** Reshard (+Move) that
  `reshard_if_not_optimal` forces on hardware.
- **`to_memory_config_reshard`** — let `to_memory_config` take tt-metal's *primary*
  sharded→sharded path (`ttnn::reshard`) instead of the `sharded_to_interleaved →
  interleaved_to_sharded` pair that `tt-metal/ttnn/cpp/ttnn/operations/core/to_memory_config/to_memory_config_op.cpp:279-320` labels a
  *"Workaround"*.
- **`move_on_reallocate_halo_output`** — gate the post-Halo `Move` on
  `reallocate_halo_output` rather than on `deallocate_activation`.

---

## Resolution order

Every `<gate>_enabled(device)` resolves in two layers:

```
attribute absent on the Device  ->  the process-wide default in ttsim/utils/shim_gates.py,
                                    seeded from TTSIM_* at import
attribute present               ->  that device's explicit override, set only by
                                    set_<gate>(x, device)
```

So `set_<gate>(True, device)` scopes the change to one `Device` and cannot leak into
the next workload built in the same process. `set_<gate>(True)` with no device moves
the process-wide default and affects every workload that has no explicit override.

Import setters and accessors **from `ttsim.utils.shim_gates`**, not through the front
end — that keeps a matched pair arriving from one module, and `ttsim/front/ttnn/op.py`
imports only the accessors it calls itself.

---

## Who opts in today

Only the two ResNet-50 infras, and both turn on all eight, device-scoped.  The opt-in block is
shared rather than duplicated, because it was byte-identical across the two arches and a ninth
gate added to only one would still produce a plausible projection on the other:

- `workloads/ttnn/resnet50/common/shim_gate_opt_ins.py:43-102` — `enable_resnet50_shim_gates()`,
  the eight calls and the measurement notes for each
- `workloads/ttnn/resnet50/bh/resnet50_test_infra_polaris_bh.py:80` — call site (Blackhole p100a)
- `workloads/ttnn/resnet50/wh/resnet50_test_infra_polaris_wh.py:76` — call site (Wormhole n150)

No other in-tree workload sets any gate, so vgg_unet / vit / unet / tt_transformers
all run with all eight OFF.

---

## Intended end-state per gate

Default-off is right for the PR that introduced these: it makes the change reviewable
without re-validating every workload at once. It is not the intended resting state for
most of them.

The gates fall into two kinds, and the distinction decides the end-state. **Six model
tt-metal behaviour that hardware performs unconditionally** — leaving them off makes
polaris diverge from silicon for any workload reaching the path, so the end-state is
default-on. **Two change how polaris builds a LUT key** and correspond to no tt-metal
behaviour at all; those stay opt-in.

Blast radius below is measured from the captures in `__refrun_cache`: vgg_unet p100a is
28 conv2d / 32 halo / 13 move / 1 matmul and **no elementwise binaries**; vit p100a is
74 matmul / 37 binary and **no conv2d, no halo**. So the conv-and-halo gates can only
reach vgg_unet, and the binary gates can only reach vit and the LLM workloads.

| gate | reaches, today | end-state | blocked on |
|---|---|---|---|
| `conv_input_reshard` | nothing | **default-on; gate removable** | nothing |
| `tilize_before_matmul` | vgg_unet ×1 | default-on | one vgg_unet run |
| `halo_output_dtype_promotion` | vgg_unet ×32 | default-on | one vgg_unet run |
| `memory_config_propagation` | vgg_unet ×28 | default-on | vgg_unet run, jointly with the next row |
| `to_memory_config_reshard` | vgg_unet | default-on | same run; see coupling below |
| `move_on_reallocate_halo_output` | vgg_unet ×4 | default-on | a vgg_unet workload fix, not a sweep |
| `bcast_hw_shape_propagation` | vit ×37, all LLMs | **permanent opt-in** | — |
| `conv_math_fidelity_in_key` | conv workloads | **permanent opt-in** | — |

**`conv_input_reshard` is already inert and the gate can go.** tt-metal itself gates
this on `conv_config.reshard_if_not_optimal`
(`tt-metal/ttnn/cpp/ttnn/operations/conv/conv2d/conv2d_utils.cpp:883-887`), so the shim
gate is a second gate on top of a flag. In `_maybe_emit_conv_input_reshard`
(`ttsim/front/ttnn/op.py:2040`) the flag is tested at the first guard and the gate at
the third; every vgg_unet conv sets `reshard_if_not_optimal=False` explicitly, so vgg
returns at guard one and never reads the gate. The only `=True` in the tree is
`workloads/ttnn/vadv2/tt/common.py:45`, and vadv2 is registered in no config, so it is
not on the projection path. Removing this gate changes no registered workload. A future
workload that sets the flag would inherit the reshard, which is the correct behaviour.

**`memory_config_propagation` and `to_memory_config_reshard` are coupled and must be
re-validated together.** Taking the primary path needs *both* shard specs to evaluate
`use_reshard_workaround` (see `_needs_reshard_workaround`), and a workload that does not
propagate memory configs does not have them. With `to_memory_config_reshard` alone on
for `vgg_unet`, the HEIGHT→BLOCK transition at its s2 boundary emitted one `Reshard`
where the p100a capture (rows 20-21) round-trips through DRAM as STI+ITS — **two hits
became one miss**. Flipping either default on its own repeats that; flipping both is the
only meaningful experiment.

**`move_on_reallocate_halo_output` needs a vgg_unet fix, not a sweep.** Reading
`deallocate_activation` is the wrong flag —
`tt-metal/ttnn/cpp/ttnn/operations/conv/conv2d/conv2d.cpp:297-299` uses
`reallocate_halo_output` — so default-on is correct in principle. It cost
`vgg_bh_p100a` (bs=1) **102/102 → 98/102** before it was gated, and the cause is that
the two flags default opposite: reallocate `True`, deallocate `False`. vgg sets
`deallocate_activation` at every conv and never sets `reallocate_halo_output`, so at the
positions where vgg sets `deallocate_activation=False` the off-path suppresses the Move
while the on-path falls back to reallocate's `True` default and emits one. Four
positions, four ops.

Expressing those through `_HALO_MOVE_SUPPRESSED_BY_SKU`
(`ttsim/ops/desc/ttsim_layout.py:219`) would work mechanically but is the wrong vehicle:
the table is keyed `(nhw, channels)` per SKU with no workload axis, so it would encode a
vgg-specific flag choice as a property of the card, and any second workload sharing that
geometry would inherit the suppression. The fix is for vgg to set
`reallocate_halo_output` explicitly where it means it; both flags are then truthful and
the gate can be removed.

**The two key-construction gates stay opt-in.** `bcast_hw_shape_propagation`
corresponds to no tt-metal behaviour — it changes which view an elementwise binary keys
on. `hw_shape` is set by the matmul sinf (`ttsim/ops/desc/math.py:601`) as well as by
conv and pool, so turning it on process-wide would re-key every LLM `Add` sitting
downstream of a matmul onto an NHWC-flat view. It also carries the
`FUTURE-RISK(onnx-device)` hazard recorded at `ttsim/ops/desc/helpers.py:510`: the ONNX
exclusion is `dev is None`, which goes inert the moment `SimTensor` gains a `device`,
and fails silently. The durable fix named there — a ttnn-specific binary optype under
`com.tenstorrent.ttnn` — removes the need for a gate rather than changing its default.
`conv_math_fidelity_in_key` is the same kind: it decides what goes in a key, not what
hardware does.

---

## Running a one-off experiment

The env vars are the cheapest way to test a gate across every workload without editing
a workload file:

```bash
TTSIM_TO_MEMORY_CONFIG_RESHARD=1 python polaris.py ... -s myrun
```

Because the value is read at import, this moves the *process-wide default* — i.e. it
applies to every workload in the run except those with an explicit device override
(today, the two ResNet-50 infras, which set all eight explicitly and are therefore
unaffected by the env vars).

To verify nothing regressed, compare LUT hit counts per workload against the previous
run's `STATS/*-opstats.csv`; `lut_key` is emitted even on a miss, so a diff of the
`lut_key` / `lut_hit_source` columns localises any change to specific ops.

---

## Adding a gate

1. Add the `_GATE = os.getenv('TTSIM_...', '0') == '1'` line plus the `set_` / `_enabled`
   pair to `ttsim/utils/shim_gates.py`, keeping the file a leaf (it imports only `os`,
   which is what broke the `ttsim.ops → ttsim.front` cycle).
2. Add the row to the table above.
3. `tests/test_ttnn/test_shim_gate_defaults.py` parametrises over `GATES` and will
   cover the new gate automatically — default-off, device scoping, setter-returns-previous,
   and the process-wide/override precedence.
4. Run the six-workload sweep before considering any change to a default.
