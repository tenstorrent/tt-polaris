#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Minimal CCL helpers for the single-chip dual-mode tt_transformers port.

tt-metal's full ccl.py (multi-chip all-gather / reduce-scatter) is intentionally NOT
ported — TT-Polaris is single-chip (num_devices=1). At num_devices=1 these
collectives are identity passthroughs, matching the shim-only tt_transformers, which
defined them as `return tensor`. Mixtral / multi-chip would need the real CCL; that is
out of scope for this port.
"""


def tt_all_reduce(tensor, *args, **kwargs):
    return tensor


def tt_all_gather(input_tensor, *args, **kwargs):
    return input_tensor
