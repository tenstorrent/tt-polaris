# TTNN shim → SimOp mapping

The [TTNN shim](../ttnn_shims_README.md) presents the `ttnn` API but records each call as
a `SimOp` instead of executing it. The table below lists the **public factory-bound** shims —
extracted statically from `ttsim/front/ttnn/op.py` — and the SimOp op-type each emits.
Several shim names deliberately map onto one op-type (e.g. `outer` → `MatMul`).

Beyond these, the shim has hand-written wrapper/composite functions whose emitted SimOp(s)
depend on their arguments. The note under the table names a **representative, non-exhaustive**
few and points to `op.py` as the authoritative set — it is not a complete listing.

```{ttnn-simop-map}
```
