# TT-Polaris User Guide

## Overview
TT-Polaris is a high-level simulator for performance analysis of AI architectures. It enables users to analyze and evaluate the performance characteristics of AI workloads on different hardware architectures through simulation.

## Key Features
- **Workload Analysis**: Input AI workloads are converted into DAG (Directed Acyclic Graph) representations
- **Architecture Simulation**: Simulate workloads on different hardware configurations
- **Performance Analysis**: Get detailed performance metrics and resource utilization data
- **Flexible Output Formats**: Support for YAML, JSON, and Pickle output formats

## Installation

### Prerequisites
- Python 3.13 or higher
- Miniforge package manager

### Download TT-Polaris
Execute ```git clone https://github.com/tenstorrent/tt-polaris.git```

### Setup Steps
1. Install Miniforge:
   ```bash
   curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
   bash Miniforge3-$(uname)-$(uname -m).sh
   conda update -n base -c conda-forge conda
   source <your-conda-install-path>/etc/profile.d/conda.sh
   ```

2. Create and activate the TT-Polaris environment:
   ```bash
   conda env create --file environment.yaml
   conda activate polaris
   ```

## Usage

### Basic Command Structure
```bash
python polaris.py [options] --archspec <arch_config> --wlspec <workload_spec> --wlmapspec <mapping_spec>
```

### Key Command Line Options
- `--archspec,    -a`: Path to architecture specification YAML file
- `--wlspec,      -w`: Path to workload specification YAML file
- `--wlmapspec,   -m`: Path to workload mapping specification YAML file
- `--study,       -s`: Name for the simulation study (default: "study")
- `--odir,        -o`: Output directory for results (default: ".")
- `--outputformat`: Output format for results (none/yaml/json/pickle)
- `--dump_stats_csv`: Enable CSV stats output
- `--dryrun,      -n`: Perform a dry run without actual simulation

#### Data and Output Options
- `--datatype,    -d`: Activation data type (fp64/fp32/tf32/fp16/bf16/fp8/int32/int8)
- `--dump_ttsim_onnx`: Dump ONNX graph for TTSIM workloads

#### Sweep Specifications
- `--frequency`: Frequency (MHz) range specification (start end step)
- `--batchsize`: Batch size range specification (start end step)

#### Profiling and Analysis Options
- `--enable_memalloc`: Enable memory allocation simulation
- `--instr_profile`: Enable instruction profiling
- `--enable_cprofile`: Enable Python cProfile for performance analysis
- `--disable-fusion`: Skip graph op fusion (`op_fusion_spec` in the workload mapping is ignored).
  Note: when a LUT file is loaded, disabling fusion is **not** needed to preserve LUT accuracy.
  TT-Polaris automatically clears the `fused_in_optimization` flag on any fused op that receives a
  LUT hit, so it keeps its real hardware timing. Ops without LUT entries continue to benefit from
  analytical fusion. Use `--disable-fusion` only to debug fusion behaviour itself.

**LUT timing and the analytical guardband:** LUT-sourced timing is **not** subject to the 25%
analytical guardband (`G_GUARDBAND`) that TT-Polaris applies to estimated op durations. When a LUT
hit occurs, the hardware-measured `msecs` value is used directly; the guardband is applied only
to ops that miss the LUT. In a mixed run both effects coexist with no double-counting. See
[LOOKUP_TABLE_MASTER.md](../doc/tools/perf_lookup/LOOKUP_TABLE_MASTER.md) for details.

#### Logging Options
- `--log_level,    -l`: Set logging level (debug/info/warning/error/critical, default: info)

### Filtering Options
- `--filterarch`: Filter architecture configurations
- `--filterwlg`: Filter workload groups
- `--filterwl`: Filter specific workloads
- `--filterwli`: Filter workload instances

## Configuration Files

### Architecture Specification
The architecture specification file (`archspec`) defines the hardware configuration including:
- Device specifications
- Memory hierarchy
- Compute resources
- Clock frequencies

### Workload Specification
The workload specification file (`wlspec`) defines:
- AI model configurations
- Batch sizes
- Input/output specifications
- Operator configurations

**Supported Workload APIs:**
- `TTSIM`: PyTorch-like functional API for model simulation
- `TTNN`: TT-NN (Tenstorrent Neural Network) API for hardware-native models
- `ONNX`: ONNX model format support

### Workload Mapping Specification
The workload mapping specification file (`wlmapspec`) defines:
- Operator to datatype mappings
- Resource requirements
- Operator fusion rules
- Null operations

## Performance Correlation Workflow

### Overview
TT-Polaris includes tools to correlate simulation results with actual hardware measurements from TT-Metal, enabling validation and calibration of simulation accuracy.

### Understanding Reference Data Tags

The correlation workflow uses **tags** to version and manage TT-Metal reference performance data:

**What are Tags?**
- Tags are version identifiers for snapshots of TT-Metal hardware measurements
- Example: `03nov25` represents measurements taken on November 3, 2025
- Each tag corresponds to a directory: `data/metal/inf/<TAG>/` containing metric YAML files


**Available Tags**
To see all valid tags, check `tools/ttsi_corr/ttsi_corr_utils.py`:
```python
TTSI_REF_VALID_TAGS = ['03nov25', '15oct25']  # Current valid tags
TTSI_REF_DEFAULT_TAG = TTSI_REF_VALID_TAGS[0]  # Default: '03nov25'
```

**Using Tags**
- **Default behavior**: Omitting `--tag` uses `TTSI_REF_DEFAULT_TAG` (currently `03nov25`)
- **Explicit tag**: Use `--tag <TAG>` to specify a particular version
- **Custom tags**: Create your own by parsing metrics with a new tag name

**Important Notes**
- Both parsing and correlation must use the **same tag** for consistency
- Tag directories must exist in `data/metal/inf/<TAG>/` before running correlation
- Tags are case-sensitive

### Workflow Steps

#### 1. Extract Hardware Metrics
Extract performance metrics from TT-Metal documentation or measurement data:

```bash
# Parse TT-Metal README to extract metrics
python tools/parse_ttsi_perf_results.py \
    --input https://raw.githubusercontent.com/tenstorrent/tt-metal/main/models/README.md \
    --output-dir __tmp/data/metal/inf
```

This will create categorized YAML files in `__tmp/data/metal/inf/`:
- `tensix_md_perf_metrics_llm.yaml` - LLM models
- `tensix_md_perf_metrics_vision.yaml` - Vision models
- `tensix_md_perf_metrics_detection.yaml` - Object detection models
- `tensix_md_perf_metrics_nlp.yaml` - NLP models
- `tensix_md_perf_metrics_diffusion.yaml` - Diffusion models

#### 2. Configure Workloads for Correlation
Create a workload configuration file (e.g., `config/ttsi_correlation_workloads.yaml`) defining the models to correlate:

```yaml
workloads:
  - api: TTSIM
    name: resnet50
    basedir: workloads
    module: BasicResNet@basicresnet.py
    instances:
      corr:  # Special instance for correlation
        bs: 32
        layers: 50
        # Additional parameters...
```

#### 3. Run Correlation Analysis
Execute the correlation tool to compare TT-Polaris simulation with hardware measurements:

```bash
python tools/run_ttsi_corr.py \
    --tag <TAG> \
    --workloads-config config/ttsi_correlation_workloads.yaml \
    --arch-config config/tt_wh.yaml \
    --output-dir __CORRELATION_OUTPUT
```

**Key Options:**
- `--tag`: Tag identifying the parsed metrics dataset (e.g., `03nov25`, `15oct25`). See `TTSI_REF_VALID_TAGS` in `tools/ttsi_corr/ttsi_corr_utils.py`
- `--input-dir`: Base directory containing tagged metrics (default: `data/metal/inf`)
- `--workloads-config`: Workload configuration for TT-Polaris simulation
- `--arch-config`: Architecture specification (e.g., Wormhole)
- `--workload-filter`: Filter specific workloads by name
- `--precision`: Override precision for all workloads (e.g., bf8, bf16, fp32)
- `--output-dir`: Directory for correlation results
- `--dry-run`: Preview actions without executing

**Note:** The data source format (html/md) is automatically detected from metadata created by `parse_ttsi_perf_results.py`.

#### 4. Analyze Results
The correlation tool generates:

**CSV Output** (`correlation_result.csv`):
- Side-by-side comparison of hardware vs. simulation
- Calculated ratios and differences
- Easy to import into spreadsheets

**XLSX Output** (`correlation_result.xlsx`):
- Formatted tables with color coding
- Excel formulas for ratios
- Conditional formatting for quick visual analysis
- Frozen panes for easier navigation

**JSON Output** (`correlation_geomean.json`):
- Geometric mean of correlation ratios
- Aggregate accuracy metrics
- Machine-readable format for automation

**Example Results Structure:**
```
__CORRELATION_OUTPUT/
├── correlation_result.csv         # CSV format
├── correlation_result.xlsx        # Excel format with formatting
├── correlation_geomean.json       # Geometric mean metrics
├── inputs/                        # Saved configuration
│   ├── runinfo.json
│   ├── tensix_workloads.yaml
│   └── tensix_runcfg.yaml
├── SIMPLE/                        # TT-Polaris simulation results
│   ├── CONFIG/
│   ├── STATS/
│   └── SUMMARY/
```

### Understanding Correlation Metrics

**Key Metrics in Results:**
- **Reference Performance**: Measured hardware performance (from TT-Metal)
- **Actual Performance**: TT-Polaris simulation performance
- **Ratio**: Simulation / Hardware (ideally close to 1.0)
- **Score-to-Ref**: Ratio expressed as percentage
- **Geometric Mean**: Aggregate accuracy across all workloads

**Interpretation:**
- Ratio ≈ 1.0: Excellent correlation
- Ratio > 1.0: Simulation overestimates performance
- Ratio < 1.0: Simulation underestimates performance
- Geometric Mean: Overall simulation accuracy

### Advanced Usage

#### Custom Metric Parsing
For custom hardware data formats:

```python
from tools.parsers.md_parser import extract_table_from_md_link, TensixMdPerfMetricModel

# Extract from custom source
metrics = extract_table_from_md_link(custom_url)

# Filter or transform as needed
filtered_metrics = [m for m in metrics if m.hardware == 'n150']

# Save for correlation
from tools.parsers.md_parser import save_md_metrics
from pathlib import Path
save_md_metrics(filtered_metrics, Path('data/custom'))
```

#### Selective Correlation
Run correlation for specific models:

```bash
# Only correlate specific workloads using workload filter
python tools/run_ttsi_corr.py \
    --tag <TAG> \
    --workloads-config config/ttsi_correlation_workloads.yaml \
    --arch-config config/tt_wh.yaml \
    --workload-filter bert,llama,resnet50 \
    --output-dir __CORRELATION_OUTPUT
```

#### Programmatic Access
Use correlation tools in Python scripts:

```python
from tools.run_ttsi_corr import main
from tools.ttsi_corr.ttsi_corr_utils import TTSI_REF_DEFAULT_TAG

# Run correlation programmatically using default tag
result = main([
    'run_ttsi_corr',
    '--tag', TTSI_REF_DEFAULT_TAG,
    '--workloads-config', 'config/ttsi_correlation_workloads.yaml',
    '--arch-config', 'config/tt_wh.yaml',
    '--output-dir', 'output'
])
```

### Troubleshooting Correlation

**Issue: No metrics extracted**
- Verify URL accessibility and format
- Check that tables have required columns (model, batch, hardware)
- Enable debug logging: Add at start of script:
  ```python
  from loguru import logger
  import sys
  logger.remove()
  logger.add(sys.stdout, level='DEBUG')
  ```

**Issue: Workload not found in configuration**
- Ensure workload name matches between hardware data and config file
- Check for `corr` instance in workload configuration
- Verify workload module exists and is importable

**Issue: Large correlation discrepancies**
- Review workload parameters (batch size, model size, etc.)
- Verify architecture configuration matches hardware
- Check data types and precision settings
- Consider hardware-specific optimizations not modeled

### ViT Device-Perf Correlation (TT-Polaris vs HW Profiler)

For the optimized sharded ViT workload, `tools/profiling/correlate_perf.sh` compares a
TT-Polaris opstats CSV against a Tracy HW profiler CSV (`merged_ops.csv`) and produces a
3-sheet XLSX workbook (Summary / By Layer Type / By Layer Signature).

Reference profiler CSVs are stored on LFC under `hlm-refrun/` and are fetched automatically.

**Quick start (from repo root):**

```bash
# Wormhole n150 — uses lfc://hlm-refrun/n150_vit_opt_sharded_wh_refrun_260415.csv
tools/profiling/correlate_perf_wh.sh

# Blackhole p100a — uses lfc://hlm-refrun/p100a_vit_opt_sharded_bh_refrun_260426.csv
tools/profiling/correlate_perf_bh.sh

# Re-run polproj first, then compare
tools/profiling/correlate_perf_wh.sh --regen
```

**Generic form** (custom arch or profiler CSV):

```bash
tools/profiling/correlate_perf.sh <arch> <hw_csv> [--regen] [--polaris-csv <path>]
#   arch      wh or bh
#   hw_csv    local path or lfc:// URL to merged_ops.csv
```

**Outputs** (gitignored, written to workspace root):

| File | Content |
|------|---------|
| `correlate_{arch}.xlsx` | 3-sheet comparison workbook |
| `correlate_{arch}.out` | Full text report |
| `correlate_{arch}.err` | Stderr / warnings |

## Output and Analysis

### Output Formats
TT-Polaris supports multiple output formats:
- YAML: Human-readable structured format
- JSON: Web-friendly format
- Pickle: Binary format for Python objects
- CSV: Tabular format for statistics

### Key Metrics
The simulation provides various performance metrics including:
- Architecture package name (`archname`) - Identifies the architecture family (e.g., Grendel, Wormhole)
- Device instance name (`devname`) - Specific device instance (e.g., Q1_A1, n150)
- Execution cycles and time
- Memory usage and requirements
- Resource bottlenecks
- Input/output parameter counts
- Resource utilization
- Tensor shapes for inputs, outputs, and weights (per operator)

### Output Directory Structure
```
output_dir/
├── study_name/
│   ├── SUMMARY/
│   │   └── study-summary.csv       # Includes archname and devname columns
│   └── STATS/
│       ├── device-workload-stats.csv
│       └── device-workload-opstats.[yaml|json|pickle|csv]
```

**Note**: The `archname` field distinguishes architecture packages (e.g., Wormhole, Grendel) from device instances (`devname` field, e.g., n150, n300). This enables easier grouping and filtering of results by architecture family.

**Operator Statistics Fields:**
Each operator in the statistics output includes:
- Performance metrics (cycles, time, utilization)
- Resource usage (memory, compute)
- Tensor information:
  - `input_tensors`: Input tensor names, shapes, and precisions (format: `name[dim1xdim2xdim3]:precision;name2[...]:precision`)
  - `output_tensors`: Output tensor names, shapes, and precisions
  - `weight_tensors`: Weight/parameter tensor names, shapes, and precisions

## Best Practices
1. Start with a dry run using `--dryrun` to validate configurations
2. Use filtering options to focus on specific architectures or workloads
3. Enable CSV output for easy data analysis
4. Use appropriate output formats based on your needs:
   - YAML for human readability
   - JSON for web integration
   - Pickle for Python processing
5. Monitor memory requirements using `--enable_memalloc`

## TTNN Workload Development Guide

### Overview
TTNN (TT-NN - Tenstorrent Neural Network) workloads represent models implemented using TT-NN operations that execute directly on Tenstorrent hardware. TT-Polaris supports TTNN workloads for performance simulation and analysis.

### TTNN Workload Architecture

#### Workload Dispatch Flow
```
YAML Config → TT-Polaris → TTNN Workload Function → Device Execution → Performance Graph
```

1. **Configuration** (`config/all_workloads.yaml`):
   - Defines workload API type, name, and instances
   - Specifies module path and hyperparameters

2. **Dispatch** (`polaris.py:165-180`):
   - Loads TTNN function from module
   - Opens device for execution
   - Calls workload with standard signature
   - Captures operation graph for analysis

3. **Execution** (e.g., `workloads/ttnn/resnet50/bh/test_resnet50_device_perf_bh.py`):
   - Implements model using TTNN operations
   - Returns output tensor
   - Device tracks all operations for performance analysis

   The entry point is thin; the ResNet-50 model itself is shared across both arches in
   `workloads/ttnn/resnet50/common/ttnn_functional_resnet50.py`, and the per-arch
   `resnet50_test_infra_polaris_{bh,wh}.py` supplies the arch, capture SKU and compute grid.

### Creating TTNN Workloads

#### Standard Function Signature
All TTNN workload functions must follow this signature:

```python
def run_<model_name>(wlname: str, device: TTNNDevice, cfg: dict):
    """
    Workload entry point for TT-Polaris TTNN framework.
    
    Args:
        wlname: Workload identifier from TT-Polaris (e.g., "Resnet50")
                This corresponds to the 'name' field in YAML config.
                
        device: TTNN device instance (lifecycle managed by TT-Polaris)
        
        cfg: Configuration dict with model hyperparameters
    
    Returns:
        Output tensor from model forward pass
    """
    # Implementation
    pass
```

#### Key Conventions

1. **Function Naming**: `run_<model_name>` where model_name matches the module context
   - Example: `run_vit`, `run_llama3`, `run_resnet50_e2e_entry`
   - A workload with several YAML entries names one function per entry rather than branching
     inside one: ResNet-50 exposes `run_resnet50_e2e_entry` and `run_resnet50_perf_device`

2. **Parameter Contract**:
   - `wlname` (str): Workload identifier from YAML config (used for logging/identification)
   - `device` (TTNNDevice): Device instance provided by TT-Polaris
   - `cfg` (dict): Configuration merged from YAML + runtime overrides

3. **Configuration Extraction**:
   ```python
   batch_size = cfg.get('batch_size', 8)  # With defaults
   img_height = cfg.get('img_height', 224)
   # ... extract other hyperparameters
   ```

4. **Device Lifecycle**: DO NOT open/close device - TT-Polaris manages it
   - Device opened by TT-Polaris before function call
   - Device closed by TT-Polaris after function returns
   - Operations automatically tracked on provided device

#### Example TTNN Workload

```python
# workloads/ttnn/my_model/ttnn_functional_my_model.py
from ttsim.front.ttnn.device import Device as TTNNDevice
import ttsim.front.ttnn as ttnn

def run_my_model(wlname: str, device: TTNNDevice, cfg: dict):
    """
    My Model workload entry point for TT-Polaris TTNN framework.
    
    Args:
        wlname: Workload identifier from TT-Polaris
        device: TTNN device instance (lifecycle managed by TT-Polaris)
        cfg: Configuration dict with model hyperparameters
    
    Returns:
        Output tensor from model forward pass
        
    Note:
        Invoked by polaris.py:120 as: ttnn_func(wln, ttnn_device, gcfg)
    """
    # Extract configuration
    batch_size = cfg.get('batch_size', 1)
    seq_length = cfg.get('seq_length', 512)
    hidden_dim = cfg.get('hidden_dim', 768)
    
    # Create input tensors
    input_tensor = ttnn.Tensor(
        shape=[batch_size, seq_length, hidden_dim],
        dtype=ttnn.bfloat16,
        device=device
    )
    
    # Implement model operations
    # (operations automatically tracked by device)
    x = ttnn.linear(input_tensor, weight, bias)
    x = ttnn.gelu(x)
    output = ttnn.linear(x, output_weight, output_bias)
    
    return output
```

### YAML Configuration

#### Workload Configuration Format

```yaml
workloads:
  - api: TTNN                              # API type
    name: MyModel                          # Workload name (passed as wlname)
    basedir: workloads/ttnn/my_model/      # Base directory
    module: run_my_model@ttnn_functional_my_model.py  # Note: filename is basename, no "/"
    params: 
      wl_param_1: 20                       # Parameter common across all instances of this workload
    instances:
      default:                             # Instance name
          batch_size: 8                    # Instance-specific config
          seq_length: 512
          hidden_dim: 768
      large:                               # Alternative instance
          batch_size: 16
          seq_length: 1024
          hidden_dim: 1024
```

#### Configuration Merging

The `cfg` parameter passed to your function is merged from multiple sources:

1. **Instance Config**: From YAML `instances.<instance_name>`
2. **Runtime Overrides**: From command-line options
   - `--batchsize`: Overrides `batch_size` in cfg
   - Other sweep parameters

Example:
```bash
# Original YAML has batch_size: 8
python polaris.py --wlspec config/my_workloads.yaml --batchsize 1 32 1

# Your function receives cfg with batch_size: 32 (overridden)
```

### Parameter Flow Reference

```
┌─────────────────────────────────────────────────────────────────┐
│ Configuration YAML (config/all_workloads.yaml)                  │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ workloads:                                                   │ │
│ │   - api: TTNN                                                │ │
│ │     name: "Resnet50"  ────► wln (workload name)             │ │
│ │     instances:                                               │ │
│ │       default:                                               │ │
│ │           batch_size: 8 ──► gcfg (config dict)               │ │
│ └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ TT-Polaris Dispatcher (polaris.py:111-122)                      │
│                                                                  │
│ ttnn_func = get_ttnn_functional_instance(wpath, wln, gcfg)      │
│ ttnn_device = open_device(device_id=0)                          │
│                                                                  │
│ ttnn_res = ttnn_func(wln, ttnn_device, gcfg)                    │
│                       │         │         │                      │
│                       │         │         └─► cfg dict           │
│                       │         └──────────► device instance     │
│                       └────────────────────► wlname string       │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ TTNN Workload Function (workloads/ttnn/.../run_*.py)            │
│                                                                  │
│ def run_resnet50_e2e_entry(wlname, device, cfg: dict):          │
│     # wlname = "resnet50_bh_device_perf" (logging/identity)     │
│     # device = TTNN device instance (managed by TT-Polaris)     │
│     # cfg = {bs: 16, ...} (merged configuration)                │
│                                                                  │
│     batch_size = cfg.get('bs', DEFAULT_BATCH_SIZE)              │
│     # ... implement model ...                                   │
│     return output_tensor                                        │
└─────────────────────────────────────────────────────────────────┘
```

### Best Practices

1. **Always Use `.get()` with Defaults**: Provides fallback values
   ```python
   batch_size = cfg.get('batch_size', 8)  # ✓ Good
   batch_size = cfg['batch_size']          # ✗ Bad - may fail
   ```

2. **Don't Manage Device Lifecycle**: TT-Polaris handles it
   ```python
   # ✗ Bad - Don't do this
   device = ttnn.open_device(device_id=0)
   ...
   ttnn.close_device(device)
   
   # ✓ Good - Use provided device
   def run_my_model(wlname: str, device: TTNNDevice, cfg: dict):
       # Use device parameter
       tensor = ttnn.Tensor(..., device=device)
   ```

3. **Include Comprehensive Docstrings**: Document the signature convention
   - Reference polaris.py invocation
   - List expected cfg parameters
   - Specify return type

4. **Use Meaningful Variable Names**: Match YAML config keys
   ```python
   # ✓ Good - matches YAML
   batch_size = cfg.get('batch_size', 8)
   
   # ✗ Less clear
   bs = cfg.get('bs', 8)
   ```

5. **Return TTNN Tensors**: TT-Polaris expects tensor outputs
   ```python
   return output_tensor  # ✓ Good
   return None          # ✗ Bad
   ```

### Debugging TTNN Workloads

#### Enable Debug Logging
```bash
python polaris.py --log_level debug --wlspec config/my_workloads.yaml ...
```

#### Check Operation Graph
After execution, TT-Polaris captures the operation graph:
- All TTNN operations are recorded
- Performance statistics computed per operation
- Graph available for analysis

#### Common Issues

1. **"ModuleNotFoundError"**: Check module path in YAML
2. **"KeyError in cfg"**: Use `.get()` with defaults
3. **"Device already opened"**: Don't open device in function
4. **Shape mismatch"**: Verify tensor shapes match TTNN requirements

### See Also
- `workloads/ttnn/resnet50/` - Reference implementation (entry points in `bh/` and `wh/`,
  shared model in `workloads/ttnn/resnet50/common/ttnn_functional_resnet50.py`)
- `polaris.py:165-180` - TTNN dispatch mechanism
- `config/all_workloads.yaml` - Configuration examples
- `ttsim/front/ttnn/` - TTNN API implementation
- `doc/shim_gates.md` - Opt-in shim behaviours and their `TTSIM_*` environment variables

## Troubleshooting

### Common Issues
1. Memory Constraints
   - Use `--enable_memalloc` to check memory requirements
   - Verify device memory specifications
   
2. Performance Issues
   - Enable profiling with `--enable_cprofile`
   - Check resource bottlenecks in output statistics

3. Configuration Errors
   - Validate YAML syntax in specification files
   - Use `--dryrun` to check configurations
   - Verify file paths and permissions

## Support
For issues and questions:
- Check the project repository: https://github.com/tenstorrent/tt-polaris
- Review existing issues or create new ones
- Consult the development team for advanced support
