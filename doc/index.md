# Polaris Documentation

**Polaris** (`ttsim`) is a high-level, roofline performance simulator for AI workloads
on Tenstorrent hardware. Start with the [Introduction](INTRODUCTION.md) for the guided
tour, then dive into the guides and reference below.

```{toctree}
:maxdepth: 2
:caption: Guides

INTRODUCTION
user_guide
overview
torch2ttsim
```

```{toctree}
:maxdepth: 2
:caption: Frontends & authoring

functional
ttnn_shims_README
TTNN_WORKLOAD_FLOW
shape_inference
```

```{toctree}
:maxdepth: 2
:caption: Reference (generated)

reference/simops
reference/ttnn-mapping
reference/onnx-mapping
reference/config-examples
YAML_MASTER_FORMAT
```
