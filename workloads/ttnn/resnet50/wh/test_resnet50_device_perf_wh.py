#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""ResNet-50 device-perf entry points — Wormhole (n150).

Mirrors the structure of ``workloads/ttnn/vgg_unet/wh/test_vgg_unet_device_perf_wh.py``.

Two entry points, and they are not two modes of one thing:

``e2e`` is dual-mode and is the port's validation. On HW it runs the PORTED model
through canonical's pipeline, which PCC-checks it against the torch golden; on
Polaris it builds the op graph the back-end turns into the opstats CSV.

``device-perf`` is Polaris-only — the projected-fps JSON via
``workloads/common/polaris_device_perf.py``, registered as
``resnet50_wh_perf_report``. Note it rebuilds the graph a second time (once on
polproj's device, again inside ``run_device_perf_polaris``) to produce that
report. Its former HW branch wrapped canonical's own pytest under Tracy, which
measured CANONICAL rather than this port, and was removed; see the comment above
``run_resnet50_device_perf_report``.

For the reference silicon number, run canonical's test directly.  The
command below is run FROM THE tt-metal CHECKOUT ROOT, so its path is
relative to that root, not to this repository::

    pytest 'models/demos/vision/classification/resnet50/wormhole/tests/\
test_resnet50_performant.py::test_run_resnet50_inference[True-16-DataType.BFLOAT8_B-DataType.BFLOAT8_B-MathFidelity.LoFi-device_params0]'

That is also the command to wrap under the profiler when a fresh capture is
needed for LUT regeneration; the capture is then converted with
``tools/profiling/profiler_to_polaris_converter.py`` and folded into the table
by ``tools/perf_lookup/update_operator_lookup_table.py``.

Canonical reference this port is written against:
  entry point   tt-metal/models/demos/vision/classification/resnet50/
                ttnn_resnet/tt/ttnn_functional_resnet50.py
  callable      run_resnet50_inference, in tt-metal/models/demos/vision/
                classification/resnet50/ttnn_resnet/tests/common/resnet50_performant.py
  batch         16 on both arches
  operating pt  act/weight bfloat8_b, MathFidelity.LoFi
  expected      6335.0 device-perf fps, from tt-metal/models/demos/
                vision/classification/resnet50/wormhole/tests/
                test_perf_device_resnet50.py:16
"""

import os
import sys
import argparse
from typing import Callable

sys.path.append('.')

# ---------------------------------------------------------------------------
# Layout-agnostic import of the rest of this workload.
#
# This tree lives in two places:
#   polaris : <polaris>/workloads/ttnn/resnet50/   -> package workloads.ttnn.resnet50
#   IRD     : <tt-metal>/resnet50/                 -> package resnet50   (copied to the
#                                                     tt-metal top level for HW runs)
# Putting this tree's PARENT on sys.path makes the short `resnet50.…` form resolve in
# both, so the fallback below works on IRD without disturbing the polaris import.
# ---------------------------------------------------------------------------
_RESNET50_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.dirname(_RESNET50_DIR) not in sys.path:
    sys.path.insert(0, os.path.dirname(_RESNET50_DIR))

# Runtime mode is resolved once, in the model module, on ttnn AVAILABILITY rather
# than on a box label.  Imported from there rather than recomputed here: three
# copies of the rule is what let it drift before, and the model module must own
# it anyway because the SystemExit guard has to run ahead of its own
# `import torch` / `import ttnn`.
try:  # polaris layout
    from workloads.ttnn.resnet50.common.ttnn_functional_resnet50 import (
        IS_POLARIS,
        mode_description,
    )
except ModuleNotFoundError as _exc:  # IRD layout: tree sits at <tt-metal>/resnet50/
    if not (_exc.name or '').startswith('workloads'):
        raise
    # import-not-found: this spelling resolves only when the module is loaded from
    # inside tt-metal's models tree, where the package root is `resnet50.` rather
    # than `workloads.ttnn.resnet50.`.  Unresolvable from polaris's root by design.
    from resnet50.common.ttnn_functional_resnet50 import (  # type: ignore[no-redef,import-not-found]
        IS_POLARIS, mode_description,
    )

from loguru import logger  # noqa: E402

# The mismatch case (hardware selected, ttnn absent) already raised SystemExit
# inside the model module, above its own imports.  Nothing left to check here.
logger.info(f'resnet50 WH runtime mode: {mode_description()}')

if not IS_POLARIS:
    import time  # noqa: E402

    import ttnn  # type: ignore[import-not-found]  # noqa: E402

    # Canonical harness (real torchvision weights, torch golden, PCC) driving the
    # PORTED model — see workloads/ttnn/resnet50/common/resnet50_test_infra_hw.py.  Importing canonical's
    # create_test_infra directly here would run tt-metal's own model, so the PCC
    # check would compare canonical against canonical and could not fail.
    try:
        from workloads.ttnn.resnet50.common.resnet50_test_infra_hw import (  # noqa: E402
            run_ported_resnet50_inference,
        )
    except ModuleNotFoundError as _exc:
        # Fall back ONLY when it is this package that is absent (the IRD layout,
        # where the tree sits at <tt-metal>/resnet50/).  A ModuleNotFoundError
        # raised by a dependency deep inside the module — tt-metal's `import
        # torch`, say — must propagate: retrying here just hits the same failure
        # and reports it twice, hiding the real cause.
        if not (_exc.name or '').startswith('workloads'):
            raise
        from resnet50.common.resnet50_test_infra_hw import (  # type: ignore[import-not-found, no-redef]  # noqa: E402
            run_ported_resnet50_inference,
        )
else:
    import ttsim.front.ttnn as ttnn
    from workloads.ttnn.resnet50.wh.resnet50_test_infra_polaris_wh import (
        create_test_infra,
    )

# Canonical device-perf batch for this workload x arch (models/README.md row 65).
DEFAULT_BATCH_SIZE = 16

# Canonical device-perf operating point, from
# tt-metal/models/demos/vision/classification/resnet50/wormhole/tests/test_perf_device_resnet50.py:16.  Passed
# EXPLICITLY at the
# create_test_infra call below: canonical's signature (resnet50_test_infra.py:342-352)
# takes act_dtype / weight_dtype / math_fidelity as REQUIRED positionals, and both the
# real-ttnn and shim namespaces expose these same three names.
ACT_DTYPE = ttnn.bfloat8_b
WEIGHT_DTYPE = ttnn.bfloat8_b
MATH_FIDELITY = ttnn.MathFidelity.LoFi


# ---------------------------------------------------------------------------
# run_resnet50_e2e — mirrors run_resnet50_inference from resnet50_performant.py
# ---------------------------------------------------------------------------

def run_resnet50_e2e(device, batch_size: int = DEFAULT_BATCH_SIZE):
    """Single iteration of the model.

    MUST be named ``batch_size`` and MUST return the device:
    ``run_device_perf_polaris`` calls ``test_fn(fe_device, batch_size=batch_size)``
    and then uses the RETURN VALUE as the device it reads the graph from
    (workloads/common/polaris_device_perf.py:57-59).  Returning the output
    tensor instead yields an AttributeError on ``.get_graph()``.
    """
    if not IS_POLARIS:
        # Hand off to canonical's own run_resnet50_inference with the ported model
        # swapped in.  It owns the input path (models.tt_cnn.tt.pipeline moves the
        # host tensor to device) and calls test_infra.validate() on the output, which
        # IS the PCC check against the torch golden.  Reimplementing that path here
        # is what produced "TT_FATAL: Device Operations expect device tensors as
        # inputs".
        t0 = time.time()
        run_ported_resnet50_inference(
            device,
            batch_size,
            ACT_DTYPE,
            WEIGHT_DTYPE,
            MATH_FIDELITY,
            model_location_generator=None,
            # Matches the canonical device-perf parametrization: the leading `True`
            # in that test id IS skip_compile_run
            # (tt-metal/models/demos/vision/classification/resnet50/wormhole/tests/test_resnet50_performant.py,
            # @parametrize("skip_compile_run")).
            # Left False, run_resnet50_pipeline does pipeline.compile() AND
            # pipeline.enqueue() — two executions — so a Tracy capture contains every
            # op TWICE and an ops-CSV comparison against the canonical capture shows a
            # uniform 2x on every op type.
            skip_compile_run=True,
        )
        t1 = time.time()
        logger.info(
            f'ttnn_resnet50_224x224_batch_size_{batch_size}: '
            f'ported model ran and PASSED canonical validate() (PCC vs torch golden). '
            f'wall time {round(t1 - t0, 3)}s'
        )
    else:
        test_infra = create_test_infra(
            device,
            batch_size,
            ACT_DTYPE,
            WEIGHT_DTYPE,
            MATH_FIDELITY,
            model_location_generator=None,
        )
        tt_inputs_host, _sharded_mem_config_DRAM, input_mem_config = (
            test_infra.setup_dram_sharded_input(device)
        )
        test_infra.input_tensor = ttnn.to_memory_config(tt_inputs_host, input_mem_config)
        output = test_infra.run()
        expected_shape = [batch_size, 1, 1, 1000]
        assert list(output.shape) == expected_shape, (
            f'Expected output shape {expected_shape}, but got {list(output.shape)}'
        )
        logger.info(f'run_resnet50_e2e: obtained expected output shape {expected_shape}')

    return device


def run_resnet50_e2e_entry(wlname: str, device, cfg: dict):
    """all_workloads.yaml entry point for resnet50_wh_device_perf."""
    batch_size = cfg.get('bs', DEFAULT_BATCH_SIZE)
    return run_resnet50_e2e(device, batch_size=batch_size)


def run_resnet50_perf_device(wlname: str, device, cfg: dict):
    """all_workloads.yaml entry point for resnet50_wh_perf_report.

    Builds the graph on the polproj-provided *device* (so polproj's back-end
    pipeline can run its own stats), then produces a standalone JSON perf
    report via the Polaris analytical path.

    NOTE this builds the graph TWICE: once here on polproj's device, and again
    inside ``run_device_perf_polaris``, which opens its own.  The second build is
    what emits the projected-fps JSON.
    """
    batch_size = cfg.get('bs', DEFAULT_BATCH_SIZE)
    run_resnet50_e2e(device, batch_size=batch_size)
    run_resnet50_device_perf_report(batch_size=batch_size)


# ---------------------------------------------------------------------------
# run_resnet50_device_perf_report — Polaris analytical projection
#
# Polaris-only by design.  There used to be a HW branch here that shelled out to
# canonical's own pytest and asserted 6335.0 AVG DEVICE KERNEL SAMPLES/S.  It was
# removed because it did not exercise this port at all: the ported model is
# injected by the `_ported_model_class()` monkeypatch inside
# `run_ported_resnet50_inference`, and a bare pytest invocation instantiates
# canonical's model class instead — so that branch measured canonical against
# canonical and would have passed or failed identically with this port deleted.
# `run_resnet50_e2e` is the HW-side check, and it is a real one: canonical's
# pipeline calls `test_infra.validate()` on the ported model's output, which is
# the PCC comparison against the torch golden.
#
# To get the reference silicon number, run canonical's test directly — the
# command is in this module's docstring.  That is also what to wrap under the
# profiler when a fresh capture is needed for LUT regeneration; see
# tools/profiling/profiler_to_polaris_converter.py for the conversion step.
# ---------------------------------------------------------------------------

def run_resnet50_device_perf_report(batch_size: int = DEFAULT_BATCH_SIZE):
    if not IS_POLARIS:
        raise SystemExit(
            'device-perf is Polaris-only: the HW branch measured canonical rather '
            'than the ported model and was removed.  Use `e2e` to validate the port '
            "on silicon (it runs the PCC check), or run canonical's "
            'test_run_resnet50_inference directly for the reference fps — see the '
            'comment above this function.'
        )

    from workloads.common.polaris_device_perf import (
        prep_device_perf_report_polaris,
        run_device_perf_polaris,
    )

    cols = ['DEVICE FW', 'DEVICE KERNEL', 'DEVICE BRISC KERNEL']
    post_processed_results = run_device_perf_polaris(
        test_fn=run_resnet50_e2e,
        batch_size=batch_size,
        cols=cols,
    )
    prep_device_perf_report_polaris(
        model_name=f'resnet50-wh-{batch_size}',
        batch_size=batch_size,
        post_processed_results=post_processed_results,
    )


# ---------------------------------------------------------------------------
# Registry and CLI
# ---------------------------------------------------------------------------

_STANDALONE_RUN_SPECS: list[tuple[str, Callable, str]] = [
    ('e2e', run_resnet50_e2e_entry, 'resnet50-wh-e2e'),
]

_STANDALONE_VALID_SHORT_NAMES = frozenset(s[0] for s in _STANDALONE_RUN_SPECS)


def run_one(callback, wlname: str, cfg: dict):
    if IS_POLARIS:
        from ttsim.front.ttnn.device import close_device, open_device
        device = open_device()
    else:
        from ttnn import close_device, open_device  # type: ignore[no-redef]
        device = open_device(device_id=0, l1_small_size=24576)
    callback(wlname, device, cfg)
    close_device(device)


def standalone(test_name: str) -> None:
    """Run one standalone WH ResNet-50 test by short name.

    ``test_name`` is required.  It used to be ``str | None`` with a None branch
    that ran every spec in ``_STANDALONE_RUN_SPECS``; that branch was dead --
    the only caller is ``standalone(_args.test)`` below and the argparse option
    carries ``default='e2e'``, so None never arrived.
    """
    all_names = _STANDALONE_VALID_SHORT_NAMES | {'device-perf'}

    if test_name == 'device-perf':
        run_resnet50_device_perf_report()
        return

    if test_name not in all_names:
        valid = ', '.join(sorted(all_names))
        logger.error(f'Unknown test {test_name}. Valid names: {valid}')
        sys.exit(1)

    for short, fn, wlname in _STANDALONE_RUN_SPECS:
        if short == test_name:
            run_one(fn, wlname, {})
            return


if __name__ == '__main__':
    logger.remove()
    logger.add(sys.stdout, level='INFO')
    parser = argparse.ArgumentParser(
        description='Run ResNet-50 (Wormhole) e2e standalone tests.'
    )
    parser.add_argument(
        'test',
        nargs='?',
        metavar='TEST',
        default='e2e',
        help='Run only this test by short name, e.g. e2e, device-perf. If omitted, runs e2e.',
    )
    _args = parser.parse_args()
    standalone(_args.test)
