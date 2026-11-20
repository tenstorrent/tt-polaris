[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/release/python-3132/)
&nbsp;&nbsp;&nbsp;![Checkin Tests](https://github.com/tenstorrent/tt-polaris/actions/workflows/checkin_tests.yml/badge.svg?branch=main)

<!-- RTL/NeoSim badges -->
<!--
![NeoSim Checkin Tests](https://github.com/tenstorrent/tt-polaris/actions/workflows/neosim_checkin_tests.yml/badge.svg?branch=main)
&nbsp;&nbsp;&nbsp;![RTL Status](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/ssapreTT/c230835a5065366c13d3bbfb79f23bf6/raw/polaris_rtl_scurve_status.json)
&nbsp;&nbsp;&nbsp;![RTL Correlation](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/ssapreTT/c230835a5065366c13d3bbfb79f23bf6/raw/polaris_rtl_scurve_ratio_geomean.json)
-->

<img alt="TT-Polaris logo" src="doc/polaris_logo.png" height="200">
</picture>

<p>
Yet Another High Level AI Simulator
</p>

</div>

----

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Download TT-Polaris](#download-tt-polaris)
  - [Environment Setup](#environment-setup)
    - [User Environment](#user-environment)
    - [Developer Environment](#developer-environment)
- [Usage](#usage)
  - [Basic Command Structure](#basic-command-structure)
  - [Configuration Files](#configuration-files)
  - [Output and Analysis](#output-and-analysis)
  - [Best Practices](#best-practices)
  - [Troubleshooting](#troubleshooting)
- [Performance Correlation Tools](#performance-correlation-tools)
  - [Tensix-Metal Correlation](#tensix-metal-correlation)
  - [Markdown Metrics Parser](#markdown-metrics-parser)
- [Support](#support)

## Introduction
*TT-Polaris* is a high level simulator for performance analysis of AI architectures. It takes as input an *AI Workload* and an *Architecture Configuration*. It represents the input workload into an in-memory directed acyclic graph (DAG) data structure, where each node represents a *computation* or a *communication* operator, and each edge represents a *dataflow*. The graph data structure represents an *intermediate representation* (IR) for the Simulator virtual machine (VM). We can execute various graph transformations, and eventually schedule the DAG on a *backend* for performance analysis.

## Installation

### Prerequisites
- Python 3.13 or higher
- Miniforge package manager

### Download TT-Polaris
Execute ```git clone https://github.com/tenstorrent/tt-polaris.git```

### Environment Setup
The recommended setup uses Python with the Miniforge installation manager. It is expected that the reader is familiar with conda environments, creating and switching between environments. One can familiarize oneself with these concepts at [Conda Getting Started](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html).

#### Installing Miniforge
1. Download and install Miniforge as described in the [Miniforge Installation Instructions](https://github.com/conda-forge/miniforge):
   ```bash
   curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
   bash Miniforge3-$(uname)-$(uname -m).sh
   ```
2. Update conda:
   ```bash
   conda update -n base -c conda-forge conda
   ```
3. Conda environment initialization: The install script prompts the user to initialize conda with your shell. After this step, the user needs to log out and login again  (or,
   exit the terminal and start a new terminal session). 

4. To enable conda command on the path, run 
    ```
    source <your-conda-install-path>/etc/profile.d/conda.sh
    ```

#### User Environment Setup
1. Create and activate the TT-Polaris environment. The conda environment will
   be created with the name 'polaris'. If you wish to provide a different name to the environment, run the command
   `conda env create --file environment.yaml --name <name-of-your-choice>` instead.
   ```bash
   conda env create --file environment.yaml
   conda activate polaris
   ```

#### Developer Environment Setup
1. Create and activate the development environment. The conda environment will
   be created with the name 'polarisdev'. If you wish to provide a different name to the environment, run the command
   `conda env create --file envdev.yaml --name <name-of-your-choice>` instead.
   ```bash
   conda env create --file envdev.yaml
   conda activate polarisdev
   ```
2. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

#### LFC Server Configuration

Workloads and tests that pull artifacts from the Large File Cache (LFC) — both
the Python helper `ttsim/utils/lfc.py` and the shell tool
`tools/ci/lfc_downloader.sh` — require the `LFC_SERVER_URLS` environment
variable. It must be set to a comma-separated list of LFC base URLs to try in
order; there are no built-in defaults, and any LFC download will fail fast
with a clear error if the variable is unset.

```bash
# In your shell rc (~/.zshrc, ~/.bashrc) or a local .env file:
export LFC_SERVER_URLS="<primary-url>,<fallback-url>"
```

**Where to get the URL values:** Tenstorrent developers should consult the
internal team documentation (Slack pinned message / internal wiki) for the
current LFC server URLs. These values are intentionally not shipped in the
public repository.

**Tailscale:** Dev access to external LFCache typically requires a Tailscale
connection. CI runs use a repository-secret-provided URL and do not need
Tailscale.

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
- `--outputformat`: Output format for results (none/yaml/json/pickle, default: json)
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

#### Logging Options
- `--log_level,    -l`: Set logging level (debug/info/warning/error/critical, default: info)

### Filtering Options
- `--filterarch`: Filter architecture configurations
- `--filterwlg`: Filter workload groups
- `--filterwl`: Filter specific workloads
- `--filterwli`: Filter workload instances

### Configuration Files

#### Architecture Specification
The architecture specification file (`archspec`) defines the hardware configuration including:
- Device specifications
- Memory hierarchy
- Compute resources
- Clock frequencies

#### Workload Specification
The workload specification file (`wlspec`) defines:
- AI model configurations
- Batch sizes
- Input/output specifications
- Operator configurations

#### Workload Mapping Specification
The workload mapping specification file (`wlmapspec`) defines:
- Operator to datatype mappings
- Resource requirements
- Operator fusion rules
- Null operations

### Output and Analysis

#### Output Formats
TT-Polaris supports multiple output formats:
- YAML: Human-readable structured format
- JSON: Web-friendly format
- Pickle: Binary format for Python objects
- CSV: Tabular format for statistics

#### Key Metrics
The simulation provides various performance metrics including:
- Execution cycles and time
- Memory usage and requirements
- Resource bottlenecks
- Input/output parameter counts
- Resource utilization
- Tensor shapes for inputs, outputs, and weights (per operator)

#### Output Directory Structure
```
output_dir/
├── study_name/
│   ├── SUMMARY/
│   │   └── study-summary.csv
│   └── STATS/
│       ├── device-workload-stats.csv
│       └── device-workload-opstats.[yaml|json|pickle|csv]
```

**Operator Statistics Fields:**
Each operator in the statistics output includes:
- Performance metrics (cycles, time, utilization)
- Resource usage (memory, compute)
- Tensor information:
  - `input_tensors`: Input tensor names, shapes, and precisions (format: `name[dim1xdim2xdim3]:precision;name2[...]:precision`)
  - `output_tensors`: Output tensor names, shapes, and precisions
  - `weight_tensors`: Weight/parameter tensor names, shapes, and precisions

### Best Practices
1. Start with a dry run using `--dryrun` to validate configurations
2. Use filtering options to focus on specific architectures or workloads
3. Enable CSV output for easy data analysis
4. Use appropriate output formats based on your needs:
   - YAML for human readability
   - JSON for web integration
   - Pickle for Python processing
5. Monitor memory requirements using `--enable_memalloc`

### Troubleshooting

#### Common Issues
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

## Performance Correlation Tools

TT-Polaris includes tools for correlating simulated performance with measured hardware performance.

### Tensix-Metal Correlation

The correlation tool compares TT-Polaris simulation results with actual hardware measurements from TT-Metal.

**Quick Start:**
```bash
# Step 1: Parse hardware metrics (creates metadata)
python tools/parse_ttsi_perf_results.py --tag <TAG>

# Step 2: Run correlation (reads data source from metadata)
python tools/run_ttsi_corr.py \
    --tag <TAG> \
    --workloads-config config/ttsi_correlation_workloads.yaml \
    --arch-config config/tt_wh.yaml \
    --output-dir __CORRELATION_OUTPUT
```

**Note:** `<TAG>` identifies the reference data version (e.g., `03nov25`, `15oct25`). See [Reference Data Tags](#reference-data-tags) below for details.

#### Reference Data Tags

TT-Polaris maintains versioned snapshots of TT-Metal reference performance data using tags:

- **What are tags?** Tags identify specific versions of TT-Metal hardware measurements (e.g., `03nov25` = November 3, 2025 measurements)
- **Current default:** The default tag is automatically updated to use the latest reference data
- **Available tags:** See `TTSI_REF_VALID_TAGS` in `tools/ttsi_corr/ttsi_corr_utils.py` for all valid tags
- **Using older data:** You can explicitly specify any valid tag with `--tag <TAG>` to reproduce results with older reference data
- **Creating custom tags:** To create a new tag:
  1. Parse metrics with a new tag name: `python tools/parse_ttsi_perf_results.py --tag <YOUR_TAG>`
  2. This creates `data/metal/inf/<YOUR_TAG>/` with metric files
  3. Run correlation with the same tag: `python tools/run_ttsi_corr.py --tag <YOUR_TAG>`

**Features:**
- Automated correlation analysis between TT-Polaris and TT-Metal
- XLSX and CSV export with formulas and formatting
- Geometric mean calculation for aggregate metrics
- Support for multiple input formats (markdown, directories, URLs)

**Documentation:** See [tools/parsers/](tools/parsers/) for detailed documentation.

### Markdown Metrics Parser

Extract performance metrics from TT-Metal markdown documentation:

```bash
python tools/parse_ttsi_perf_results.py \
    --input https://raw.githubusercontent.com/tenstorrent/tt-metal/main/models/README.md \
    --output-dir data/metal/inf
```

**Features:**
- Standards-compliant markdown parsing (CommonMark via markdown-it-py)
- Automatic categorization by model type (LLM, vision, detection, NLP, diffusion)
- YAML output format compatible with correlation tools
- Robust error handling and validation

**Documentation:** 
- [Parser README](tools/parsers/README.rst) - Architecture and API reference
- [Usage Examples](tools/parsers/USAGE_EXAMPLE.rst) - Examples and troubleshooting

## Support
For issues and questions:
- Check the project repository: https://github.com/tenstorrent/tt-polaris
- Review existing issues or create new ones
- Consult the development team for advanced support
