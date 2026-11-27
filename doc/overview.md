# TTSim Overview

This document provides an overview of all classes in the `ttsim` module and explains how they fit together to form a high-level simulator for Tenstorrent AI hardware.

## Version History

| SHA | Description |
|-----------|-----------|
| f8b22f711b812cef0679229798f3b9ece5bf8147 | Initial Dec 9, 2025 version reference SHA |

## Table of Contents

1. [Core Graph and Operations](#core-graph-and-operations)
2. [Device and Backend](#device-and-backend)
3. [Frontend Interfaces](#frontend-interfaces)
4. [Configuration Models](#configuration-models)
5. [Statistics and Performance](#statistics-and-performance)
6. [Utilities](#utilities)
7. [System Architecture](#system-architecture)

---

## Core Graph and Operations

### `WorkloadGraph` (`graph/wl_graph.py`)

**Purpose**: Represents a computational graph of operations (workload) as a directed acyclic graph (DAG). It manages the structure of neural network operations and tensors.

**Key Responsibilities**:

- Maintains a NetworkX MultiDiGraph for graph structure
- Tracks operations (`SimOp`) and tensors (`SimTensor`)
- Identifies input/output nodes and tensors
- Supports graph optimizations (node removal, fusion)
- Can export to ONNX format
- Sets precision and resource assignments for operations

**Relationships**:

- Contains multiple `SimOp` objects (operations)
- Contains multiple `SimTensor` objects (data tensors)
- Used by `Device` to execute workloads
- Used by `HLMStats` to collect performance statistics

### `SimTensor` (`ops/tensor.py`)

**Purpose**: Represents a tensor (multi-dimensional array) in the computation graph. Stores shape, data type, and metadata about data flow.

**Key Responsibilities**:

- Stores tensor shape, dtype, and optional data
- Tracks which operations produce (`op_out`) and consume (`op_in`) the tensor
- Distinguishes between parameters, constants, and activations
- Calculates element count and byte size
- Supports cloning operations

**Key Attributes**:

- `name`: Unique identifier
- `shape`: Tensor dimensions
- `dtype`: NumPy data type
- `is_param`: Whether tensor is a model parameter
- `is_const`: Whether tensor is a constant
- `op_in`: List of operations that consume this tensor
- `op_out`: List of operations that produce this tensor

**Relationships**:

- Referenced by `SimOp` objects in their `inList` and `outList`
- Stored in `WorkloadGraph._tensors`
- Extended by `Tensor` in `front/ttnn/tensor.py` for TTNN interface

### `SimOp` (`ops/op.py`)

**Purpose**: Represents a single operation (operator) in the computation graph. Encapsulates operation type, attributes, and performance characteristics.

**Key Responsibilities**:

- Stores operation metadata (name, type, attributes, domain)
- Manages input/output tensor lists
- Performs shape inference via registered shape inference functions
- Tracks performance statistics (instruction counts, element counts, cycles)
- Supports optimization flags (removed, fused)

**Key Attributes**:

- `optype`: Operation type (e.g., 'MatMul', 'Conv', 'Add')
- `precision`: Data precision for execution
- `uses_compute_pipe`: Which compute pipeline to use ('matrix' or 'vector')
- `compute_cycles`, `mem_rd_cycles`, `mem_wr_cycles`: Performance metrics
- `perf_stats`: Dictionary of performance statistics
- `exec_stats`: Execution statistics from device simulation

**Relationships**:

- Contains references to `SimTensor` objects via `inList` and `outList`
- Stored in `WorkloadGraph._ops`
- Executed by `Device.execute_op()`
- Uses `SimOpDescRegistry` for shape inference

### `SimOpDescRegistry` (`ops/desc/registry.py`)

**Purpose**: Global registry for operation descriptions, including shape inference functions and operation metadata.

**Key Responsibilities**:

- Registers operation descriptions with shape inference functions
- Provides lookup for operation descriptions
- Maintains operation metadata (min/max inputs/outputs, domain, etc.)

**Relationships**:

- Used by `SimOp.get_perf_counts()` to get shape inference functions
- Populated by operation descriptor modules in `ops/desc/`

---

## Device and Backend

### `Device` (`back/device.py`)

**Purpose**: Represents a Tenstorrent hardware device and simulates execution of workloads on it. This is the core execution engine.

**Key Responsibilities**:

- Executes workloads on simulated hardware
- Calculates compute and memory cycles for operations
- Applies graph optimizations (removal, fusion)
- Aggregates performance statistics across the entire workload
- Determines resource bottlenecks (compute vs memory bound)

**Key Methods**:

- `execute_graph()`: Main entry point to execute a workload graph
- `execute_op()`: Executes a single operation and calculates cycles
- `get_exec_stats()`: Aggregates statistics for the entire workload

**Key Attributes**:

- `simconfig_obj`: Device configuration (from `PackageInstanceModel`)
- `compute_ip`: Compute IP group configuration
- `memory_ip`: Memory IP group configuration
- `devname`: Architecture package name (e.g., "Grendel", "Wormhole")
- `name`: Device instance name (e.g., "Q1_A1", "n150")

**Relationships**:

- Uses `WorkloadGraph` to execute workloads
- Uses `WL2ArchMap` for workload-to-architecture mapping
- Uses `PackageInstanceModel` (via `simconfig_obj`) for device capabilities
- Produces statistics consumed by `HLMStats`

### `Component` (`back/device.py`)

**Purpose**: Base class for hardware components (memory, NOC, processing elements).

**Subclasses**:

- `MEM`: Represents memory components with size and bandwidth
- `NOC`: Represents network-on-chip with grid dimensions
- `PE`: Represents processing elements

### `TTDevice` (`back/device.py`)

**Purpose**: Legacy device representation (hardcoded configuration). Less commonly used than `Device`.

---

## Frontend Interfaces

### Functional Interface (`front/functional/`)

#### `SimOpHandle` (`front/functional/op.py`)

**Purpose**: Wrapper around `SimOp` that provides a PyTorch-like functional interface for building computation graphs.

**Key Responsibilities**:

- Manages parameter tensors and input positions
- Creates `SimOp` and `SimTensor` objects when called
- Links operations to modules
- Performs shape inference automatically

**Relationships**:

- Creates `SimOp` objects internally
- Used by `Module` to build computation graphs
- Provides functional operators (Add, MatMul, Conv, etc.)

#### `SplitOpHandle` (`front/functional/op.py`)

**Purpose**: Specialized handle for Split operations that produce multiple outputs.

#### `VariadicInputOpHandle` (`front/functional/op.py`)

**Purpose**: Handle for operations with variable number of inputs (e.g., Concat).

#### `SimOpHandleList` (`front/functional/op.py`)

**Purpose**: Immutable list of `SimOpHandle` objects that can be called sequentially (like a sequential layer).

#### `Module` (`front/functional/sim_nn.py`)

**Purpose**: PyTorch-like module base class for building neural network models. Manages submodules, operations, and tensors.

**Key Responsibilities**:

- Tracks tensors, operations, and submodules
- Builds computation graphs from module structure
- Provides `_get_forward_graph()` to construct `WorkloadGraph` from module

**Relationships**:

- Contains `SimOpHandle` objects
- Contains `SimTensor` objects
- Can contain other `Module` objects (submodules)
- Produces `WorkloadGraph` objects

#### `ModuleList` (`front/functional/sim_nn.py`)

**Purpose**: Immutable list of `Module` objects.

#### `Linear` (`front/functional/sim_nn.py`)

**Purpose**: Example module implementation for a linear (fully connected) layer.

### TTNN Interface (`front/ttnn/`)

#### `Tensor` (`front/ttnn/tensor.py`)

**Purpose**: Extends `SimTensor` with TTNN-specific functionality and PyTorch-like tensor operations.

**Key Responsibilities**:

- Provides tensor operations (view, transpose, unsqueeze, etc.)
- Manages device and layout information
- Supports tensor operations via operator overloading

**Relationships**:

- Extends `SimTensor`
- Associated with `Device` (from `front/ttnn/device.py`)
- Used by TTNN-style workloads

#### `Device` (`front/ttnn/device.py`)

**Purpose**: TTNN-style device interface. Simplified device representation for TTNN workloads.

**Key Responsibilities**:

- Manages tensors and operations
- Provides `get_graph()` to convert to `WorkloadGraph`
- Represents device configuration

**Relationships**:

- Contains `Tensor` objects
- Contains `SimOp` objects
- Can produce `WorkloadGraph` objects

#### `CoreCoord`, `CoreRange`, `CoreRangeSet` (`front/ttnn/core.py`)

**Purpose**: Represents spatial coordinates and ranges for core placement on hardware.

**Key Classes**:

- `CoreCoord`: (x, y) coordinate pair
- `CoreRange`: Rectangular range of cores
- `CoreRangeSet`: Collection of non-overlapping core ranges

#### `DataType`, `Layout` (`front/ttnn/tensor.py`)

**Purpose**: Enumerations for data types and memory layouts.

### Additional TTNN Classes

#### `Shape` (`front/ttnn/tensor.py`)

**Purpose**: Extends Python list to represent tensor shapes with utility methods.

#### `ShardStrategy` (`front/ttnn/tensor.py`)

**Purpose**: Enumeration for tensor sharding strategies (HEIGHT, WIDTH, BLOCK).

#### `MathFidelity` (`front/ttnn/op.py`)

**Purpose**: Enumeration for math fidelity levels (LoFi, HiFi2, HiFi3, HiFi4).

#### `transformer`, `experimental` (`front/ttnn/op.py`)

**Purpose**: Placeholder classes for transformer and experimental operations (not yet implemented).

#### `MemoryConfig` (`front/ttnn/memory.py`)

**Purpose**: Enumeration for memory configuration (DRAM, L1).

#### `TensorMemoryLayout` (`front/ttnn/buffer.py`)

**Purpose**: Enumeration for tensor memory layouts.

---

## Configuration Models

### `SimConfig` (`config/simconfig.py`)

**Purpose**: Generic configuration object that recursively converts dictionaries to attribute-based access.

### `XlsxConfig` (`config/simconfig.py`)

**Purpose**: Configuration loader from Excel files with parameter normalization.

### `SimCfgBlk` (`config/simconfig.py`)

**Purpose**: Base class for configuration blocks with required/optional field validation.

**Subclasses**:

- `WorkloadCfgBlk`: Base for workload configurations
- `WorkloadTTSIM`: Configuration for TTSIM API workloads
- `WorkloadONNX`: Configuration for ONNX API workloads
- `WorkloadGroup`: Group of workloads

### `PackageInstanceModel` (`config/simconfig.py`)

**Purpose**: Pydantic model representing a device package instance with IP groups (compute and memory).

**Key Responsibilities**:

- Models device architecture (compute pipes, memory blocks)
- Provides methods to query peak IPC, FLOPS, bandwidth
- Manages frequency settings

**Key Attributes**:

- `devname`: Architecture package name
- `name`: Device instance name
- `ipgroups`: List of IP group configurations

**Relationships**:

- Contains `IPGroupComputeModel` and `IPGroupMemoryModel`
- Used by `Device` via `simconfig_obj`
- Referenced by `HLMStats` for device information

### `ComputeBlockModel`, `MemoryBlockModel` (`config/simconfig.py`)

**Purpose**: Pydantic models for compute and memory IP blocks.

**Key Classes**:

- `ComputeBlockModel`: Models compute IP with pipes and L2 cache
- `MemoryBlockModel`: Models memory IP with technology, size, bandwidth
- `ComputePipeModel`: Models individual compute pipes with instructions
- `ComputeInsnModel`: Models instruction throughputs for different precisions
- `L2CacheModel`: Models L2 cache configuration

## Workload-to-Architecture Mapping

### `WL2ArchMap` (`config/wl2archmap.py`)

**Purpose**: Maps workload operations to architecture-specific configurations (data types, compute pipes, optimizations).

**Key Components**:

- `WL2ArchDatatypes`: Maps operations to data types
- `WL2ArchRemovalLayers`: Specifies operations to remove during optimization
- `WL2ArchFusedLayers`: Specifies operation sequences to fuse
- `WL2ArchLayer2ComputePipe`: Maps operations to compute pipes

**Relationships**:

- Used by `Device.execute_graph()` to configure workload execution
- Used by `WorkloadGraph.set_precision()` and `set_resources()`
- Singleton pattern via `WL2ArchTypeSpec` for global access

### `PolarisRunConfig` (`config/runcfgmodel.py`)

**Purpose**: Pydantic model for run configuration parameters (workloads, architectures, frequencies, batch sizes, etc.).

### Configuration Validators (`config/validators.py`)

**Purpose**: Pydantic validators for validating configuration files.

**Key Validator Classes**:

- `PYDWlMapDataSpecValidator`: Validates workload mapping data type specifications
- `PYDWlMapResourceSpecValidator`: Validates workload mapping resource specifications
- `PYDWlMapSpecValidator`: Validates complete workload mapping specifications
- `PYDPkgMemoryValidator`: Validates package memory configurations
- `PYDPkgComputeValidator`: Validates package compute configurations
- `PYDComputePipeValidator`: Validates compute pipe configurations
- `PYDL2CacheValidator`: Validates L2 cache configurations
- `PYDMemoryBlockValidator`: Validates memory block configurations
- `PYDComputeBlockValidator`: Validates compute block configurations
- `PYDWorkloadTTSIMModelValidator`: Validates TTSIM workload models
- `PYDWorkloadONNXModelValidator`: Validates ONNX workload models
- `PYDWorkloadListValidator`: Validates workload lists

---

## Statistics and Performance

### `HLMStats` (`stats/hlmstats.py`)

**Purpose**: High-level statistics collector that aggregates performance data from device execution and outputs results.

**Key Responsibilities**:

- Collects execution statistics from `Device`
- Formats and outputs statistics in multiple formats (CSV, YAML, JSON, Pickle)
- Validates precision consistency
- Produces summary statistics

**Key Methods**:

- `dump_stats()`: Main method to collect and output statistics
- `check_precision()`: Validates precision settings

**Relationships**:

- Uses `Device` to get execution statistics
- Uses `WorkloadGraph` to iterate over operations
- Outputs `TTSimHLWlDevRunPerfStats` models

### `OutputFormat` (`stats/hlmstats.py`)

**Purpose**: Enumeration for output formats (YAML, JSON, Pickle, None).

### Performance Statistics Models (`config/validators.py`)

**Purpose**: Pydantic models for structured performance statistics.

**Key Models**:

- `TTSimHLWlDevRunPerfStats`: Complete run statistics with operator details
- `TTSimHLWlDevRunOperatorPerfStats`: Per-operator statistics
- `TTSimHLRunSummaryRow`: Summary row for aggregated statistics
- `TTSimHLRunSummary`: Collection of summary rows

---

## Utilities

### `SimOpDescRegistry` (`ops/desc/registry.py`)

**Purpose**: Global registry for operation descriptions (already covered above).

### Common Utilities (`utils/common.py`)

**Purpose**: Common utility functions (CSV printing, YAML parsing, unit conversion, etc.).

**Key Classes**:

- `dict2obj`: Converts dictionaries to objects with attribute access
- `CustomLogger`: Custom logging utility

### Type Utilities (`utils/types.py`)

**Purpose**: 
- Utilities for data type conversion and byte-per-element calculations.
- Type definitions and enumerations for framework types, data types, tensor dimensions, etc.

**Key Enumerations**:

- `FrameworkType`: Framework identifiers
- `SimDataType`: Simulation data types
- `DataFormat`: Data format specifications
- `MathFidelity`: Math fidelity levels

### Cache Manager (`utils/cache.py`)

**Purpose**: `CacheManager` class for managing cached data.

### Prime Factorization (`utils/prime_factorization.py`)

**Purpose**: `PrimeFactorization` class for prime factorization utilities.

### File Locator (`utils/readfromurl.py`)

**Purpose**: `FileLocator` class for locating files from URLs or local paths.

---

## System Architecture

### How Classes Fit Together

1. **Workload Construction**:
   - Users build models using `Module` (functional interface) or `Tensor` (TTNN interface)
   - Operations are created as `SimOpHandle` objects that produce `SimTensor` objects
   - Modules collect operations and tensors into a `WorkloadGraph`

2. **Configuration**:
   - Device architecture is specified via `PackageInstanceModel` (from YAML)
   - Workload-to-architecture mapping is specified via `WL2ArchMap` (from YAML)
   - Run configuration is specified via `PolarisRunConfig`

3. **Execution**:
   - `Device.execute_graph()` takes a `WorkloadGraph` and `WL2ArchMap`
   - Device sets precision and resources for each operation
   - Device executes each operation, calculating compute and memory cycles
   - Device applies optimizations (removal, fusion)
   - Device aggregates statistics

4. **Statistics Collection**:
   - `HLMStats` collects statistics from `Device` and `WorkloadGraph`
   - Statistics are formatted and output in various formats
   - Performance models (`TTSimHLWlDevRunPerfStats`) are created

### Data Flow

```
User Code (Module/Tensor)
    ↓
WorkloadGraph (SimOp + SimTensor)
    ↓
Device.execute_graph() + WL2ArchMap
    ↓
Device.execute_op() (calculates cycles)
    ↓
Device.get_exec_stats() (aggregates)
    ↓
HLMStats.dump_stats() (formats and outputs)
```

### System Architecture Diagram

```mermaid
flowchart TD
    subgraph cli["polaris.py (CLI entry point)"]
        CLI["polaris.py\n--wlspec/-w\n--archspec/-a\n--wlmapspec/-m\n--filterarch / --filterwli"]
    end

    subgraph cfg["Configuration (YAML)"]
        WL_YAML["config/all_workloads.yaml\n(workload registry)"]
        ARCH_YAML["config/tt_wh.yaml / tt_bh.yaml\n(arch specs + LUT path)"]
        MAP_YAML["config/wl2archmapping.yaml\n(fusion / dtype / removal)"]
        LUT_YAML["*_lut_vN.yaml\n(profiled timing data)"]
    end

    subgraph front["Frontends (ttsim/front/)"]
        TTNN["TTNN Shim\nttsim/front/ttnn/\n(op.py, tensor.py, device.py)\nBuilds SimOp graph\nfrom ttnn.* API calls"]
        ONNX["ONNX Parser\nttsim/front/onnx/"]
        FUNC["Functional API\nttsim/front/functional/"]
    end

    subgraph graph["Graph Layer (ttsim/graph/)"]
        WLG["WorkloadGraph\nNetworkX DAG of\nSimOp nodes + SimTensor edges\n(topological order by seqno)"]
    end

    subgraph back["Backend (ttsim/back/)"]
        DEV["Device.execute_graph()\nApplies WL2ArchMap transforms:\n• dtype assignment\n• op fusion / removal\n• compute pipe assignment"]
        LUT_LOOKUP["_try_operator_perf_lookup()\nkey = op_code + shapes +\nlayout + dtype + memory\n(+shard_layout for halo/ITS v4)"]
        ANALYTICAL["Analytical models\n(fallback on LUT miss)"]
        STATS["HLMStats.dump_stats()\nCSV / JSON / reports"]
    end

    subgraph tools["Profiling Tools (tools/profiling/ + tools/perf_lookup/)"]
        CMP["compare_layers.py\nPolaris CSV vs HW profiler CSV\n--perf --by-lut-key --xlsx"]
        MAPPER["tt_perf_mapper.py\nHW profiler CSV → LUT YAML"]
        LFC["lfc_downloader.sh\nLFC ↔ local cache"]
    end

    CLI -->|"loads"| WL_YAML
    CLI -->|"loads"| ARCH_YAML
    CLI -->|"loads"| MAP_YAML
    ARCH_YAML -->|"operator_lookup_file"| LUT_YAML

    WL_YAML -->|"dispatches workload fn"| TTNN
    WL_YAML -->|"dispatches workload fn"| ONNX
    WL_YAML -->|"dispatches workload fn"| FUNC

    TTNN -->|"get_graph()"| WLG
    ONNX -->|"get_graph()"| WLG
    FUNC -->|"get_graph()"| WLG

    WLG -->|"passed to backend"| DEV
    MAP_YAML -->|"WL2ArchMap"| DEV
    LUT_YAML -->|"OperatorPerfMap"| LUT_LOOKUP

    DEV --> LUT_LOOKUP
    LUT_LOOKUP -->|"hit"| STATS
    LUT_LOOKUP -->|"miss → fallback"| ANALYTICAL
    ANALYTICAL --> STATS

    STATS -->|"opstats CSV"| CMP
    LFC -->|"merged_ops CSV"| CMP
    LFC -->|"merged_ops CSV"| MAPPER
    MAPPER -->|"new LUT YAML"| LUT_YAML
```

<details><summary>Text fallback (if mermaid does not render)</summary>

```
  polaris.py CLI
       │ loads
  ┌────┴──────────────────────────┐
  │  config/all_workloads.yaml    │
  │  config/tt_{wh,bh}.yaml  ────────► *_lut_vN.yaml (OperatorPerfMap)
  │  config/wl2archmapping.yaml   │
  └────┬──────────────────────────┘
       │ dispatches workload fn
  ┌────▼──────────────────────────────────────────────┐
  │  FRONTENDS  (ttsim/front/)                        │
  │  TTNN Shim  │  ONNX Parser  │  Functional API     │
  │  (op.py builds SimOp graph from ttnn.* API calls) │
  └────────────────────┬──────────────────────────────┘
                       │ get_graph()
              ┌────────▼──────────┐
              │  WorkloadGraph    │
              │  (NetworkX DAG:   │
              │  SimOp + SimTensor│
              │  topo by seqno)   │
              └────────┬──────────┘
                       │ execute_graph()
  ┌────────────────────▼──────────────────────────────┐
  │  BACKEND  (ttsim/back/device.py)                  │
  │  • Apply WL2ArchMap (dtype/fusion/removal)        │
  │  • _try_operator_perf_lookup()                    │
  │      hit → use LUT msecs + util                   │
  │      miss → analytical fallback                   │
  │  • HLMStats.dump_stats() → CSV / JSON             │
  └───────────────────────────────────────────────────┘
                       │ opstats CSV
  ┌────────────────────▼──────────────────────────────┐
  │  PROFILING TOOLS  (tools/)                        │
  │  compare_layers.py   — Polaris vs HW profiler CSV │
  │  tt_perf_mapper.py   — HW profiler CSV → LUT YAML │
  │  lfc_downloader.sh   — LFC ↔ local cache          │
  └───────────────────────────────────────────────────┘
```

</details>

### Key Design Patterns

1. **Registry Pattern**: `SimOpDescRegistry` for operation descriptions
2. **Singleton Pattern**: `WL2ArchTypeSpec` for global data type access
3. **Builder Pattern**: `Module` and `SimOpHandle` for constructing graphs
4. **Strategy Pattern**: Different frontends (functional, TTNN) for different use cases
5. **Model Pattern**: Pydantic models for configuration and statistics validation

### Module Organization

- **`graph/`**: Core graph data structures
- **`ops/`**: Operation and tensor definitions
- **`back/`**: Backend device simulation
  - **`back/tensix_neo/`**: Lower-level Tensix core simulation (not covered in detail - used for detailed cycle-accurate simulation)
- **`front/`**: Frontend interfaces (functional, TTNN, LLK, ONNX)
  - **`front/functional/`**: PyTorch-like functional interface
  - **`front/ttnn/`**: TTNN tensor interface
  - **`front/llk/`**: Low-level kernel interface (instruction decoding, etc.)
  - **`front/onnx/`**: ONNX import interface
- **`config/`**: Configuration models and validators
- **`stats/`**: Statistics collection and output
- **`utils/`**: Utility functions

### Lower-Level Simulation Components

The `back/tensix_neo/` directory contains lower-level simulation components for cycle-accurate simulation of Tensix cores. These classes are used for detailed hardware simulation but are not part of the high-level simulation flow:

- `tensixCore`: Detailed Tensix core simulation
- `neoCore`: Neo core simulation
- `tensixFunc`: Tensix function simulation
- `triscFunc`: TRISC function simulation
- `scratchpadRam`: Scratchpad memory simulation
- Various instruction and register models

These are typically used internally by the high-level `Device` class when detailed simulation is needed.

---

## Summary

The TTSim system provides a high-level simulator for Tenstorrent AI hardware that:

1. **Models workloads** as computation graphs (`WorkloadGraph` with `SimOp` and `SimTensor`)
2. **Simulates execution** on hardware devices (`Device` with architecture models)
3. **Maps workloads to architectures** via configuration (`WL2ArchMap`)
4. **Collects performance statistics** (`HLMStats`) and outputs results
5. **Supports multiple frontends** (functional PyTorch-like, TTNN) for different use cases

The system is designed to be modular, with clear separation between graph representation, device simulation, configuration, and statistics collection.
