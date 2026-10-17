# SimOps reference

Every operation in a Polaris workload graph is a `SimOp` of some **op-type**. The table
below is generated at build time from the ops descriptor registry
(`ttsim/ops/desc/`), so it always matches the code.

Columns: the op-type name, its group (descriptor module), ONNX/Tenstorrent domain, input
and output arity bounds, the ONNX opset version, the shape-inference function, and whether
the op takes attributes. Shape functions marked *declared, unimplemented* are registered
as string stubs and are not yet callable.

```{simops-table}
```
