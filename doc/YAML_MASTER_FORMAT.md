# Master YAML file format (correqn tt-perf master)

Repository path: **`doc/YAML_MASTER_FORMAT.md`**. Excel/key-tuple and CLI pipeline: **`tools/perf_lookup/tt_perf_mapper.py`**.

Normative description of the YAML written by the **`tools/perf_lookup/tt_perf_mapper.py`** CLI when **`--update`** is set (inputs: repeatable **`--model-run`** / **`--sweep-run`**, optional existing **`--output`** file merged first). Files produced by the current tool carry **`schema_version: 6`** (`MASTER_YAML_SCHEMA_VERSION`); the loader also accepts **`schema_version`** **1–5** (legacy, emits `DeprecationWarning`). **Constants and parsing** live in **`tools/perf_lookup/tt_perf_master_schema.py`**; **reading** a file is **`tools.perf_lookup.tt_perf_master_loader.load_existing_yaml`**. Use this document to implement loaders, JSON Schema, Pydantic models, or contract tests in **other repositories**.

**Producer:** `tools/perf_lookup/tt_perf_mapper.py` — **`serialize_master_for_yaml`** builds a mapping; **`yaml.dump`** (`default_flow_style=False`, `sort_keys=False`, `allow_unicode=True`). **Loader:** `tools/perf_lookup/tt_perf_master_loader.py` — **`load_existing_yaml`**.

**Related:** Excel column names, grouping, and normalization are implemented in **`tools/perf_lookup/tt_perf_mapper.py`** (`KEY_TUPLE_COLUMN_NAMES`, `apply_polaris_layer_type_column`, etc.).

---

## Schema identity

Stable names and keys (defined in `tt_perf_master_schema.py`):

| Constant | Value | Role |
|----------|-------|------|
| **`MASTER_YAML_SCHEMA_NAME`** | **`correqn.tt-perf-master`** | **Unique format name** (reverse-DNS style). Use in loaders, Pydantic, or **`$id`**-style docs to detect the correct file type before deep parsing. **Do not reuse** this string for unrelated YAML. |
| **`MASTER_YAML_SCHEMA_NAME_KEY`** | **`schema_name`** | Top-level YAML key for the string above. |
| **`MASTER_YAML_SCHEMA_VERSION`** | **`6`** (integer, current) | Changelog in `tt_perf_master_schema.py`. **v2** added **`math_fidelity`** to the key tuple (9th field, between `input_0_memory` and `input_1_*`); **v6** added time-only entries (see **Stats coverage**). Loaders must reject versions outside the range **1–6**; older versions load with a **`DeprecationWarning`** (v1 default fill: `math_fidelity: N/A`). |
| **`MASTER_YAML_SCHEMA_VERSION_KEY`** | **`schema_version`** | Top-level YAML key for the integer above. |
| **`MASTER_YAML_ENTRIES_KEY`** | **`entries`** | Top-level YAML key holding the list of records. |
| **`MASTER_YAML_COVERAGE_KEY`** | **`coverage`** | Top-level YAML key: entry count per stats coverage (see **Stats coverage**). |
| **`MASTER_STATS_COVERAGE_KEY`** | **`stats_coverage`** | On a flat single or curve payload: **`full`** (default when absent) or **`time_only`** (see **Stats coverage**). |
| **`MASTER_YAML_RECORD_KEY_FIELD`** | **`key`** | On each **`entries[i]`**: nested mapping for the labeled **logical record key** (see **Record key**). |
| **`MASTER_YAML_ENTRY_VALUE_FIELD`** | **`value`** | On each **`entries[i]`**: mapping for the **entry payload** (**`entry_type`**, stats or curve fits). |
| **`MASTER_ENTRY_TYPE_KEY`** | **`entry_type`** | Discriminator on each entry **value**: **`single`**, **`curve`**, or **`hybrid`** (matmul: nested **`single`** + **`curve`**). |
| **`MASTER_SINGLE_NUM_CORES_KEY`** | **`num_cores`** | **`single`** and **`hybrid.single`**: core count (**`int`**; whole-number **float** normalized on load). |
| **`MASTER_HYBRID_SINGLE_KEY`** | **`single`** | **`hybrid`** only: flat **`num_cores`** + same stat keys as **`entry_type: single`** (not a core-count-keyed map). |
| **`MASTER_HYBRID_CURVE_KEY`** | **`curve`** | **`hybrid`** only: mapping **`curve_family`** + per-stat fits (same shape as top-level **`curve`** minus **`entry_type`**). |
| **`MASTER_CURVE_FAMILY_KEY`** | **`curve_family`** | **`curve`** and **`hybrid.curve`**: **`linear`** or **`power`** (same family for every stat; chosen from duration vs core count). |
| **`MASTER_DURATION_MS_KEY`** | **`msecs`** | Device kernel duration in YAML (**milliseconds**); Excel column is nanoseconds. |

**Consumers:** require **`schema_name`** **`correqn.tt-perf-master`**, **`schema_version`** in range **1–6**, and **`entries`** as defined below. Wrong **`schema_name`**, out-of-range **`schema_version`**, or missing fields → **`ValueError`**; **`schema_version`** below 6 → **`DeprecationWarning`** (see **`tt_perf_master_loader.load_existing_yaml`**).

**Record key wire shape:** **`entries[i]['key']`** is a **labeled mapping** (see **Record key**). **`yaml_labeled_key_to_tuple`** / **`labeled_key_map_to_tuple`** in **`tt_perf_master_schema`** parse it to an internal 9-, 16-, or 23-tuple.

**Excel vs YAML:** The Excel sheet still uses full column titles (e.g. **`DEVICE KERNEL DURATION [ns]`**); the serialized master uses short stat keys (**`msecs`**, **`mem_util`**, …). That mapping is applied when **building** the master from Excel, not when loading YAML.

---

## Top-level structure (current)

The file is a **YAML mapping** with these keys (current write path):

| Key | Type | Required | Meaning |
|-----|------|----------|---------|
| **`schema_name`** | string | yes | Must be **`correqn.tt-perf-master`**. |
| **`schema_version`** | integer | yes | Must match **`MASTER_YAML_SCHEMA_VERSION`** for files produced by the current tool. |
| **`coverage`** | mapping | no | **`full`** and **`time_only`** entry counts. Written by the producer (v6) from the entries. When present, the loader rejects a block that disagrees with the entries. |
| **`entries`** | sequence | yes | List of **records** (see next table). |

- **Empty `entries`:** valid: zero key tuples.
- **Empty file** / **`null`:** treat as **no entries** (producer maps to `{}`).
- **`entries` present but `schema_version` absent:** **invalid** (producer raises **`ValueError`** on load).

Each element of **`entries`** MUST be a **mapping** with **exactly** these two keys (no other keys):

| Key | Role | Type (conceptual) |
|-----|------|-------------------|
| **`key`** | **Record key** | **Mapping:** labeled fields **`op_code`**, **`input_0_*`**, optional **`input_1_*`** (see **Record key** section). |
| **`value`** | **Entry** | Mapping: **single**, **curve**, or **hybrid** payload (see below). |

Example shape (matmul after merge; **`schema_version`** = **`MASTER_YAML_SCHEMA_VERSION`**):

```yaml
schema_name: correqn.tt-perf-master
schema_version: 6
coverage:
  full: 1
  time_only: 0
entries:
  - key:
      op_code: matmul
      input_0_w_pad_logical: 1
      input_0_z_pad_logical: 8
      input_0_y_pad_logical: 224
      input_0_x_pad_logical: 768
      input_0_layout: TILE
      input_0_datatype: BFLOAT16
      input_0_memory: DEV_1_DRAM_INTERLEAVED
      math_fidelity: N/A
      input_1_w_pad_logical: 1
      input_1_z_pad_logical: 1
      input_1_y_pad_logical: 768
      input_1_x_pad_logical: 768
      input_1_layout: TILE
      input_1_datatype: BFLOAT16
      input_1_memory: DEV_1_DRAM_INTERLEAVED
    value:
      entry_type: hybrid
      single:
        num_cores: 64
        msecs: 0.05
        memory_traffic: 0.0
        mem_util: 0.0
        noc_util: 0.0
        noc_multicast_util: 0.0
        npe_cong_impact_pct: 0.0
        vector_pipe_util: 0.0
        matrix_pipe_util: 0.5
      curve:
        curve_family: linear
        msecs:
          a: 0.0012
          b: 0.015
          r2: 0.99
          equation: 'msecs = 0.0012 * Core_Count + 0.015'
        memory_traffic:
          a: 0.0
          b: 0.0
          r2: 1.0
          equation: memory_traffic = 0
        mem_util:
          a: 0.01
          b: 0.02
          r2: 0.95
          equation: 'mem_util = 0.01 * Core_Count + 0.02'
```

**Duplicate logical keys:** If a hand-edited file lists the same key twice, a naive loader should define behavior (e.g. last wins). The producer builds a unique key set before serializing.

---

## Unsupported top-level shapes

Only the **versioned** mapping (**`schema_name`**, **`schema_version`**, **`entries`**) is accepted by **`tt_perf_master_loader.load_existing_yaml`** (used by the **`tt_perf_mapper`** CLI when merging). Bare top-level lists, plain dicts without **`entries`**, **`entries`** items that are not **`{key, value}`** mappings, wrong **`schema_name`**, or **`schema_version`** outside the accepted range **1–6** are rejected on load. Older versions (legacy) load with a **`DeprecationWarning`**; re-export with the current tool to upgrade to v6.

---

## Record key (`entries[i]['key']`)

The logical key is a **full-arity tuple** of length **9 + 7·(n_inputs−1)** — i.e. **9, 16, 23, 30, 37, 44, 51, or 58** for 1–8 inputs (built from Excel columns in **`tt_perf_mapper.build_key_tuple`**). Standard ops key on **all** populated input slots (input_0..input_7), so >3-input ops (e.g. `pagedfusedupdatecache`, `scaleddotproductattention`) key on every operand; arity ≤3 keys are byte-identical to the historical 9/16/23-tuples (schema **v5**). Input slots must be **contiguous** (no gaps). Under **`key`**, the wire form is a **YAML mapping** whose field names are **`KEY_TUPLE_YAML_KEYS`** in `tt_perf_master_schema.py` (fixed order; values are the same scalars as the Excel-driven tuple).

- **`op_code`:** Polaris **layer type** (e.g. `matmul`, `eltwise`, `tilize`, `tilizewithvalpadding`, `untilize`, `untilizewithunpadding`).
- **`input_0_w_pad_logical`**, **`input_0_z_pad_logical`**, **`input_0_y_pad_logical`**, **`input_0_x_pad_logical`:** logical pad integers (aligned with **`INPUT_0_*_PAD[LOGICAL]`**).
- **`input_0_layout`**, **`input_0_datatype`**, **`input_0_memory`**
- **`math_fidelity`:** math fidelity string (e.g. `HiFi4`, `LoFi`); **`N/A`** for ops where it does not apply. Added in **v2**; v1 files missing this field are back-filled with **`N/A`** on load.
- If the tuple has a second input (**16** or **23** fields), all seven **`input_1_*`** keys with the same naming pattern; if **9** fields, **omit** every **`input_1_*`** and **`input_2_*`** key (the producer omits them rather than emitting nulls).
- If the tuple has a third input (**23** fields), all seven **`input_2_*`** keys are present; **`input_2_*` must not appear** unless every **`input_1_*`** field is also present.

Unknown fields in a labeled mapping: **`labeled_key_map_to_tuple`** raises **`ValueError`**. If any **`input_1_*`** field is present, **all** seven **`input_1_*`** fields are **required** (16- or 23-tuple). If any **`input_2_*`** field is present, **all** seven **`input_2_*`** and **all** **`input_1_*`** fields are **required** (23-tuple).

---

## Entry discriminator: `entry_type`

Each **`value`** mapping **must** include **`entry_type`**: **`single`**, **`curve`**, or **`hybrid`**. **`tt_perf_master_loader.load_existing_yaml`** rejects missing or other values (no inference from payload shape). **Curve** and **hybrid.curve** include **`curve_family`** (see below).

| Key | YAML value | Meaning |
|-----|------------|---------|
| **`entry_type`** | **`single`** | Flat payload: **`num_cores`** plus stat keys (**`msecs`**, …). |
| **`entry_type`** | **`curve`** | Per-stat regression parameters vs core count; must include **`curve_family`**. |
| **`entry_type`** | **`hybrid`** | Matmul: **`single`** (optional) + **`curve`** (optional) sub-mappings; at least one branch required. |
| **`curve_family`** | **`linear`** \| **`power`** | **Curve** or **hybrid.curve:** regression family shared by all stats (from **`msecs`** vs core count). |

Constants in code: **`MASTER_ENTRY_TYPE_KEY`**, **`MASTER_SINGLE_NUM_CORES_KEY`**, **`MASTER_SINGLE_STAT_KEYS`**, **`MASTER_CURVE_FAMILY_KEY`**, **`MASTER_ENTRY_TYPE_SINGLE`**, **`MASTER_ENTRY_TYPE_CURVE`**, **`MASTER_ENTRY_TYPE_HYBRID`**, **`MASTER_HYBRID_SINGLE_KEY`**, **`MASTER_HYBRID_CURVE_KEY`**, **`MASTER_CURVE_FAMILY_LINEAR`**, **`MASTER_CURVE_FAMILY_POWER`**.

**Mixed files:** After merges, a single YAML file **MAY** contain **`single`**, **`curve`**, and **`hybrid`** entries (different key tuples). **`op_code: matmul`** must use **`entry_type: hybrid`** on disk.

---

## Single entry (`value` with `entry_type: single`)

- **Required:** `entry_type: single`, **`num_cores`** (**`MASTER_SINGLE_NUM_CORES_KEY`**), and every key in **`MASTER_SINGLE_STAT_KEYS`** (same set as the table below). A **`stats_coverage: time_only`** payload instead requires **`MASTER_TIME_ONLY_STAT_KEYS`** and must not carry the NOC-trace stats (see **Stats coverage**). **Extra keys** are accepted but **`load_existing_yaml`** emits a **`warnings.warn`** (consumers should not rely on them).
- **Producer behavior:** Non-whole **`CORE COUNT`** in Excel is coerced with **`int(round(...))`** for **`num_cores`** and a log **warning** (see **`tt_perf_mapper`**).
- **PyYAML quirk:** **`num_cores`** may load as a whole-number **float** (e.g. `64.0`). **`load_existing_yaml`** normalizes **`64.0` → `64`**. External loaders **SHOULD** do the same.

### Stat keys (flat on the same mapping as `entry_type` and `num_cores`)

Keys are **exact strings** (always present in producer output):

| Key | Semantics |
|-----|-----------|
| `msecs` | `float` (device kernel duration in **milliseconds**; Excel input remains **`DEVICE KERNEL DURATION [ns]`**) |
| `memory_traffic` | `float` (bytes; derived from duration, DRAM util %, `--dram-bw-gbps`) |
| `mem_util` | `float` (from Excel **`DRAM BW UTIL (%)`**) |
| `noc_util` | `float` (from Excel **`NOC UTIL (%)`**) |
| `noc_multicast_util` | `float` (from Excel **`MULTICAST NOC UTIL (%)`**; 0 if column absent) |
| `npe_cong_impact_pct` | `float` (from Excel **`NPE CONG IMPACT (%)`**) |
| `vector_pipe_util` | `float` (from Excel **`SFPU Util Median (%)`**) |
| `matrix_pipe_util` | `float` (from Excel **`FPU Util Median (%)`**) |

Values must be **finite** real scalars: **`load_existing_yaml`** rejects **`inf`**, **`nan`**, and non-numeric types for stat fields (see **`is_real_stat_scalar`** in **`tt_perf_master_schema.py`**). PyYAML may still parse **`inf`** / **`nan`** from source text; such files fail strict load until edited.

---

## Hybrid entry (`value` with `entry_type: hybrid`)

- **Required:** `entry_type: hybrid`.
- **At least one** of **`single`** (flat **`num_cores`** + same stat keys as **Single entry**) and **`curve`** (mapping: **`curve_family`** + per-stat **`a`** / **`b`** / **`r2`** / **`equation`**).
- **`single`** branch: same flat shape as **Single entry** (without repeating **`entry_type`** on the inner mapping).
- **`curve`** branch: same rules as **Curve entry** except the payload lives under **`curve`** and there is no inner **`entry_type`**.
- **Producer:** **`tt_perf_mapper`** combines model-run Excels (**`--model-run`**) and sweep Excels (**`--sweep-run`**) in one CLI invocation; matmul keys are written as **`hybrid`** when applicable.

---

## Curve entry (`value` with `entry_type: curve`)

- **Required:** `entry_type: curve` and **`curve_family`**: **`linear`** or **`power`**. This is the family used for **every** stat’s fit for that key tuple (selected from duration **`msecs`** vs core count in **`tt_perf_mapper.choose_best_family_with_r2`**). The loader rejects curve entries with a missing or invalid **`curve_family`**.
- **Note:** The current **`tt_perf_mapper`** pipeline canonicalizes **matmul** to **`hybrid`** before write, so production files use **`hybrid.curve`** for matmul sweeps. A top-level **`curve`** entry remains valid for the format and for non-matmul keys if a future producer emits them.
- **Additional keys:** One per **stat** (string keys matching the same stat names as in the single-mode stats dict, including **`msecs`**, `memory_traffic`, etc.).
- **Each stat value** is a mapping with:

| Key | Type | Notes |
|-----|------|--------|
| `a` | number | Fit coefficient. |
| `b` | number | Fit coefficient (intercept or exponent context per family). |
| `r2` | number | Coefficient of determination; **may be negative**. |
| `equation` | string | Human-readable equation for that stat. |

Special cases: all-NaN/zero stats use **`_curve_zero_law_entry`**-style payloads in **`tt_perf_mapper`** (`a`/`b`/`r2`/`equation` as emitted there). The entry’s **`curve_family`** still reflects the duration-based choice; zero-law stats do not change **`curve_family`**.

---

## Stats coverage (time-only entries, v6)

Some captures have no NOC-trace pass (merged with **`--allow-missing-noctrace`**): they measure kernel time and pipe utilization, but not memory or NOC behaviour. Their entries are **time-only**.

- **Marker:** a flat single payload (**`single`**, **`hybrid.single`**) or a curve payload (**`curve`**, **`hybrid.curve`**) MAY carry **`stats_coverage`**: **`full`** or **`time_only`**. A payload without the key is **`full`**, so pre-v6 files stay valid. The producer writes the key only for **`time_only`**.
- **Time-only stats:** a **`time_only`** payload carries **`MASTER_TIME_ONLY_STAT_KEYS`** (**`msecs`**, **`vector_pipe_util`**, **`matrix_pipe_util`**) and MUST NOT carry **`MASTER_NOCTRACE_STAT_KEYS`** (**`memory_traffic`**, **`mem_util`**, **`noc_util`**, **`noc_multicast_util`**, **`npe_cong_impact_pct`**). An absent NOC-trace stat means **not measured**. It is never written as **0**.
- **Entry coverage:** an entry is **`time_only`** if any of its payloads is **`time_only`**; otherwise **`full`**.
- **Mixed files:** a file MAY mix **`full`** and **`time_only`** entries. The root **`coverage`** block records the count of each.
- **Merge rule (producer):** for the same key, a **`time_only`** payload does not replace a **`full`** one (the producer keeps the full payload and logs a warning). A **`full`** payload replaces a **`time_only`** one. **`hybrid.single`** and **`hybrid.curve`** are each replaced whole, never merged field by field.
- **Simulator:** a hit on a time-only payload reports **`memory_traffic`** and **`mem_util`** as empty (None) in the opstats, and **`lut_stats_coverage: time_only`**.

---

## What is not in the YAML

- In-process **`curve_meta`** extras (duration **R²** linear vs power, core count list for dry-run reporting) — **not serialized**. **`curve_family`** on the entry replaces the need to infer family from **`equation`** alone.
- Raw Excel rows, CV diagnostics, or profiler class names before Polaris **`OP CODE`** normalization.

---

## Suggested consumer algorithm

1. `raw = yaml.safe_load(stream)`; handle **`None`** → no entries.
2. Require `raw` is a **dict** with **`schema_name`**, **`schema_version`**, and **`entries`**. Reject if **`schema_name`** ≠ **`correqn.tt-perf-master`** or **`schema_version`** is outside **1–6** (see **`tt_perf_master_schema.py`**, **`tt_perf_master_loader.py`**). Emit **`DeprecationWarning`** for **`schema_version`** below 6. If **`coverage`** is present, reject it unless it equals the entry counts per coverage. Back-fill **`math_fidelity: N/A`** on any key entry that omits it.
3. Let `records = raw["entries"]` (a list).
4. For each `item` in `records`: require `item` is a **dict** with exactly **`key`** and **`value`** (and no other keys). Let `key_wire = item["key"]`, `entry = item["value"]`. Require **`entry`** is a **mapping**. Require **`key_wire`** is a **dict**; map fields in **`KEY_TUPLE_YAML_KEYS`** order (8, 15, or 22 keys per rules above) to `key_tuple` (same rules as **`labeled_key_map_to_tuple`**).
5. Require `entry["entry_type"]` is **`single`**, **`curve`**, or **`hybrid`** (do not guess from keys).
6. If **`key_tuple[0]`** (Polaris **`op_code`**) is **`matmul`**, require **`entry_type: hybrid`** (same rule as **`tt_perf_master_loader`**).
7. If **`single`**: read **`num_cores`**, **`stats_coverage`** (default **`full`**) and the stat keys (see **`MASTER_SINGLE_STAT_KEYS`**); require all of them, or for **`time_only`** exactly **`MASTER_TIME_ONLY_STAT_KEYS`** with no NOC-trace stat; unknown keys → warn (see **`load_existing_yaml`** / **`_validate_flat_single_payload`**). Stat values must be finite real scalars (including NumPy scalar types if your loader passes them through).
8. If **`curve`**: read **`curve_family`** (`linear` \| `power`). For each `k, v` in `entry.items()` with `k` not in `entry_type` / `curve_family` / `stats_coverage` and `v` a dict, require **`MASTER_CURVE_STAT_ENTRY_KEYS`**: `a`, `b`, `r2`, `equation` (extras → warn).
9. If **`hybrid`**: parse optional **`single`** (flat **`num_cores`** + stats) and optional **`curve`** (`curve_family` + stat fits); require at least one branch.
10. Optionally normalize **`num_cores`** whole-number floats to `int` (including under **`hybrid.single`**).

**Canonical stat keys** in YAML: **`msecs`**, **`memory_traffic`**, **`mem_util`**, **`noc_util`**, **`noc_multicast_util`**, **`npe_cong_impact_pct`**, **`vector_pipe_util`**, **`matrix_pipe_util`**. See **`MASTER_DURATION_MS_KEY`** in `tools/perf_lookup/tt_perf_master_schema.py` and **`OUTPUT_KEY_*`** in `tools/perf_lookup/tt_perf_mapper.py`.

For **strict** validation (e.g. Pydantic), pin to a **`correqn`** git tag or commit, import constants from **`tt_perf_master_schema`**, and add golden-file tests.
