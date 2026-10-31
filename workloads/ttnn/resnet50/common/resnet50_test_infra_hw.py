#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""HARDWARE-ONLY test infra: canonical harness driving the PORTED model.

Imports real ``ttnn`` and the tt-metal ``models.*`` packages, so it must only be
reached from an ``IS_POLARIS == False`` branch.

Why this exists
---------------
The numerical check on the HW path compares the ported workload against the
canonical one.  That only means something if the HW path actually runs the
ported model.  The two earlier migrations do not do this — ``vgg_unet`` and
``vit`` have Polaris-only model modules (no ``IS_POLARIS`` in them at all) and
their HW branches drive tt-metal's own runner, so their check compared canonical
against canonical and could not fail.  This module closes that gap for
ResNet-50.

How
---
Rather than duplicating canonical's ~60 lines of weight preprocessing — which
would drift from upstream and undermine the comparison — this swaps the
``resnet50`` symbol inside the canonical test-infra module for the ported class,
for the duration of construction.  Canonical then does everything it normally
does (torchvision weights through ``custom_preprocessing``, the torch golden
tensor, input setup, ``validate()``'s PCC check) but builds the PORTED model.

So the comparison is genuinely like-for-like: identical weights, identical
input, identical harness — only the model implementation differs.  And
``infra.validate()`` gives the PCC check for free, against canonical's own torch
reference.
"""

import os
import sys
import contextlib

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

# E402 is intentional for the rest of this file: the sys.path.insert above must
# run first, or the `resnet50.…` fallback below cannot resolve in the IRD layout.
from models.demos.vision.classification.resnet50.ttnn_resnet.tests.common import (  # type: ignore[import-not-found]  # noqa: E402
    resnet50_performant as _canonical_performant,
)
from models.demos.vision.classification.resnet50.ttnn_resnet.tests.common import resnet50_test_infra as _canonical_infra  # noqa: E402

try:
    from workloads.ttnn.resnet50.common.ttnn_functional_resnet50 import resnet50 as _ported_resnet50
except ModuleNotFoundError as _exc:
    # IRD layout only — see the note in the runners: a ModuleNotFoundError from a
    # nested dependency must propagate rather than be retried.
    if not (_exc.name or '').startswith('workloads'):
        raise
    # import-not-found: this spelling resolves only when the module is loaded from
    # inside tt-metal's models tree, where the package root is `resnet50.` rather
    # than `workloads.ttnn.resnet50.`.  Unresolvable from polaris's root by design.
    from resnet50.common.ttnn_functional_resnet50 import resnet50 as _ported_resnet50  # type: ignore[no-redef,import-not-found]


@contextlib.contextmanager
def _ported_model_class():
    """Point the canonical infra at the ported ``resnet50`` for this block."""
    original = _canonical_infra.resnet50
    _canonical_infra.resnet50 = _ported_resnet50
    try:
        yield
    finally:
        _canonical_infra.resnet50 = original


def create_test_infra(*args, **kwargs):
    """Canonical ``create_test_infra`` signature, ported model inside.

    Drop-in for the canonical import in the runners' HW branch:

        create_test_infra(device, batch_size, act_dtype, weight_dtype,
                          math_fidelity, use_pretrained_weight=True,
                          dealloc_input=True, final_output_mem_config=...,
                          model_location_generator=...)
    """
    with _ported_model_class():
        return _canonical_infra.create_test_infra(*args, **kwargs)


@contextlib.contextmanager
def _asserting_create_test_infra():
    """Check every infra built while this is active is running the ported model.

    ``run_resnet50_pipeline`` builds the infra internally, so the caller never
    sees it. Wrapping the factory is how the guard still fires.
    """
    original = _canonical_performant.create_test_infra

    def _checked(*args, **kwargs):
        infra = original(*args, **kwargs)
        assert_using_ported_model(infra)
        return infra

    _canonical_performant.create_test_infra = _checked
    try:
        yield
    finally:
        _canonical_performant.create_test_infra = original


def run_ported_resnet50_inference(
    device,
    batch_size,
    act_dtype,
    weight_dtype,
    math_fidelity,
    model_location_generator=None,
    skip_compile_run=False,
):
    """Canonical ``run_resnet50_inference``, running the PORTED model.

    Delegates the whole run to ``resnet50_performant.run_resnet50_inference``
    — the entry point tt-metal's own device-perf test uses — so the host/device
    plumbing is canonical's, not reimplemented here. That matters: the input
    path goes through ``models.tt_cnn.tt.pipeline``, which owns the host→device
    transfer.  Hand-rolling it (e.g. calling ``ttnn.to_memory_config`` on the
    host tensor returned by ``setup_dram_sharded_input``) fails with
    ``TT_FATAL: Device Operations expect device tensors as inputs``.

    Canonical's pipeline also calls ``test_infra.validate(output)`` on each
    output, so the PCC check against the torch golden happens inside this call.
    """
    with _ported_model_class(), _asserting_create_test_infra():
        _canonical_performant.run_resnet50_inference(
            device,
            batch_size,
            act_dtype,
            weight_dtype,
            math_fidelity,
            model_location_generator,
            skip_compile_run=skip_compile_run,
        )


def assert_using_ported_model(infra) -> None:
    """Fail loudly if the infra is not actually running the ported model.

    The class swap in ``_ported_model_class`` is the only thing making the HW
    path exercise this port.  If it silently fails — a renamed symbol upstream,
    an import ordering change — the run still succeeds and the PCC check still
    passes, because canonical would be validating canonical.  A green result
    that proves nothing is worse than a red one, so this turns that failure
    mode into an immediate, named error.
    """
    actual = type(infra.ttnn_resnet50_model)
    assert actual is _ported_resnet50, (
        f'HW infra is running {actual.__module__}.{actual.__name__}, not the ported '
        f'{_ported_resnet50.__module__}.{_ported_resnet50.__name__} — the model class '
        'swap did not take effect, so this run would compare canonical against '
        'canonical and could not fail.'
    )
