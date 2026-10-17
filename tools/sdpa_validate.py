#!/usr/bin/env python3
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Validate the SDPA/TopK roofline against a tt-metal tracy ops_perf_results CSV.

Usage: python tools/sdpa_validate.py ops_perf_results_llama8b.csv -o handoff/out --model llama8b_prefill \
           [--positions positions.csv] [--provenance "tt-metal <sha>, p100a, <date>, <tracy cmd>"]
Writes <model>_signed_errors.csv, <model>_summary.csv and <model>_signed_errors.png; exit code 0 when
every regime meets the gap_analysis 4.3 criterion, 1 otherwise.
"""
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from ttsim.perf.sdpa_validate import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
