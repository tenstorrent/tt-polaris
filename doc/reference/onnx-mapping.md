# ONNX → SimOp mapping

The [ONNX frontend](../overview.md) parses an `.onnx` model and turns each node into a
`SimOp`. The mapping is **identity**: an ONNX node's `op_type` becomes the SimOp op-type
directly, and shape inference is shared with the SimOp registry. The supported set is the
`ai.onnx`-domain entries that have an **implemented (callable) shape function**, listed
below (generated at build time); entries registered with an unimplemented string stub are
called out separately beneath the table.

```{onnx-simop-map}
```
