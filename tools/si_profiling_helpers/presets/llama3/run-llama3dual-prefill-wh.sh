#!/usr/bin/env bash
# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
# Run from the tt-metal checkout root. Prereqs: see ../si_profiling_helpers/presets/README.md
# Dual-mode-on-HW validation: runs the migrated TT-Polaris llama3 prefill workload through its
# real-ttnn branch (selected because IRD_ARCH_NAME is set). Compare against the matching
# run-llama3ref-prefill preset to confirm dual-mode perf tracks the HW reference.
#
# ############################################################################################
# ⛔ NOTE — WH PREFILL NoC-TRACE CAPTURE IS CURRENTLY BLOCKED (same issue as the ref preset).
# The WH prefill phase fails NoC-trace capture (see run-llama3ref-prefill-wh.sh and the
# capture matrix in presets/README.md). This dual capture may not complete on Wormhole until
# that NoC-trace tooling issue — raised and owned by the user — is resolved. Decode is fine on
# WH; both phases are fine on BH. Remove this note once the issue is fixed.
# ############################################################################################
# Enforce the block above until the NoC-trace issue is fixed: refuse to run by default so a
# known-failing long capture isn't started accidentally. Set ALLOW_WH_PREFILL=1 to deliberately
# override; remove this guard together with the block once the issue is resolved.
[ -n "${ALLOW_WH_PREFILL:-}" ] || { echo "⛔ WH prefill capture blocked (NoC-trace issue); set ALLOW_WH_PREFILL=1 to override." >&2; exit 2; }
source "$(dirname "$0")/../check_arch.sh"; require_arch wormhole_b0
source "$(dirname "$0")/../hw_id.sh"
BOARD=$(hw_id_board)
HEAD=$(git rev-parse --short=7 HEAD)
RUNID=llama3dual-prefill-b1-${HEAD}-${IRD_ARCH_NAME}${BOARD:+-$BOARD}-$(date +%y%m%d)
CMD=workloads/ttnn/llama3_dualmode/test_llama3_prefill.py
require_dual_cmd "$CMD"
# Default (three-CSV) merge, not trace_replay: the dual-mode prefill test runs one eager forward
# with no trace capture or replay (the ref presets do trace).
python ../si_profiling_helpers/run-ttnn-profiler.py --command "$CMD" --output-dir $RUNID --report-name $RUNID --disable-logging "$@"
rc=$?
hw_id_write "$RUNID"
exit "$rc"
