#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Dual-mode ResNet-50 — a faithful mirror of the tt-metal canonical model.

Canonical source (ground truth for every op, kwarg, layout and dtype here):
``tt-metal/models/demos/vision/classification/resnet50/ttnn_resnet/tt/ttnn_functional_resnet50.py``
(939 lines).  Canonical line numbers are cited throughout as ``C:<line>``.

Line numbers in THIS file are cited as ``P:<line>`` -- here and from the
tests that exercise it.  The two files differ in length, so a number is only
meaningful with its prefix.

Both the Wormhole and Blackhole tt-metal test trees import this same
implementation from ``ttnn_resnet/``, so there is no arch-specific model
variant — hence this module lives under ``common/`` with only the device-perf
runners split per arch.

Dual mode: ``IS_POLARIS`` selects the real ``ttnn`` (hardware path) or
``ttsim.front.ttnn`` (analytical shim path).  The op sequence is identical on
both paths.  This module was re-derived from canonical HEAD; it replaces an
earlier fork that predated tt-metal ``1881ff1542d`` and never executed.

Deliberate divergences on the Polaris path — all shim-forced, none discretionary
-------------------------------------------------------------------------------
1. ``Conv2dConfig.activation`` is typed ``str`` in the shim
   (``ttsim/front/ttnn/config.py:16``), so canonical's
   ``ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)`` (C:188, C:230, C:442) becomes
   the string ``'relu'``.  Same fused-ReLU semantics, different spelling.

2. ``Conv2dConfig`` has no ``enable_activation_reuse`` field, so canonical's
   C:131, C:241 and C:449 settings are dropped on the Polaris path.  They are
   still passed on the HW path.

3. ``ttnn.conv2d`` in the shim returns a bare ``Tensor``; it does not honour
   ``return_output_dim`` / ``return_weights_and_bias``.  ``_conv2d()`` below
   normalises the two return contracts so every call site keeps canonical's
   shape.  Output dims are computed with the standard convolution formula —
   the same arithmetic canonical uses at C:468-473.

4. ``ttnn.add_`` (in-place, C:329) does not exist in the shim.  The shim's
   fused equivalent is ``ttnn.add(a, b, input_tensor_a_activations=[...])``,
   which emits a SINGLE Add SimOp carrying the activation
   (``ttsim/front/ttnn/op.py:992-1010``) — matching HW's BinaryNg, which is the
   point of canonical's fused ReLU.

5. ``ttnn.fold`` — see the block comment at the call site in ``resnet50.run``.
   Canonical passes ``use_transpose_as_fold=True`` with a 6-element two-sided
   ``padding`` list (C:614).  That is a composite on hardware: the p100a
   capture records it as seven profiler rows.  The shim's
   ``use_transpose_as_fold`` path raises on its own 6-D reshape, and its
   first-class Fold SimOp would emit a single row against those seven, so the
   Polaris path calls ``_shim_fold_transpose_sharded_nchw`` instead — a port of
   tt-metal's ``fold_with_transpose_sharded_`` that emits the same seven.  The
   pad arguments differ in kind between the two: that helper takes ONE-sided
   pads and doubles them itself, where ``ttnn.fold`` needed them pre-doubled.
   Either way the geometry matches canonical exactly (115x115 into conv1).

6. The shim has no ``Shape.to_rank``.  Canonical reshapes the fc weight/bias to
   rank 4 at C:40-41; on the Polaris path ``model_preprocessing`` supplies them
   at rank 4 already, so the reshape is skipped.
"""

import os
import importlib.util
import math
from typing import Any, List, Sequence

from loguru import logger

# ---------------------------------------------------------------------------
# Runtime mode: real silicon, or the analytical shim?
#
# This used to be `IS_POLARIS = os.getenv('IRD_ARCH_NAME', '') == ''`, copied
# identically here and into both per-arch runners.  That test asks whether the
# box is *labelled* hardware, which is not the same question as whether hardware
# is *available* — and the gap is a real failure: on an IRD box with the variable
# unset it selected the Polaris branch, which then tried to `import ttsim`, a
# package that does not exist in a tt-metal checkout, and died with a bare
# ModuleNotFoundError that named nothing useful.  Worse, running from a polaris
# checkout *on* an IRD box, both branches import cleanly and you get an
# analytical projection while sitting on silicon: no error, plausible numbers,
# the device never touched.
#
# So resolve on the capability, with explicit overrides in front of it.  The
# resolver lives here, above the heavy imports below, because those imports are
# what it selects between — the guard has to run before them or it cannot help.
# ---------------------------------------------------------------------------

_FALSEY = frozenset({'0', 'false', 'no', 'off', ''})


def _importable(name: str) -> bool:
    """``find_spec`` raises for a partially-initialised module or a broken
    parent package; both mean "not usable here"."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def real_ttnn_available() -> bool:
    """True when a top-level ``ttnn`` can be imported."""
    return _importable('ttnn')


def polaris_checkout() -> bool:
    """True when ``ttsim`` is importable, i.e. we are inside a polaris tree.

    This, not ``ttnn``, is the load-bearing signal.  ``ttnn`` importable looked
    like a clean proxy for "real hardware", but it is not: a dev environment can
    have a ``ttnn`` package installed with no ``torch`` behind it, and that
    combination selected the hardware path on a Mac and died on ``import torch``.
    ``ttsim`` is unambiguous in the direction that matters — it exists in every
    polaris checkout and in no tt-metal checkout — and it fails safe, because a
    wrong answer here sends a polaris run to the shim rather than to silicon.
    """
    return _importable('ttsim')


def is_polaris() -> bool:
    """True -> analytical shim (``ttsim.front.ttnn``).  False -> real ``ttnn``.

    Precedence, most explicit first:

    1. ``TTSIM_FORCE_POLARIS`` — wins outright, both directions (``0``/``false``
       forces hardware).  The escape hatch for anything the rules below get
       wrong.
    2. ``IRD_ARCH_NAME`` — the historical marker, still honoured so existing
       invocations and the other workloads' convention keep working.
    3. ``ttsim`` importable -> POLARIS.  The common case, and the one that must
       never mis-fire.
    4. ``ttnn`` importable -> HARDWARE.  Reached only outside a polaris tree,
       i.e. the IRD layout, which is the case this ordering exists to serve: an
       IRD box with ``IRD_ARCH_NAME`` unset still runs on silicon instead of
       falling into the Polaris branch and dying on ``import ttsim``.
    5. Neither -> POLARIS, and the guard below turns it into a clear message.
    """
    forced = os.getenv('TTSIM_FORCE_POLARIS')
    if forced is not None:
        return forced.strip().lower() not in _FALSEY
    if os.getenv('IRD_ARCH_NAME', '').strip():
        return False
    if polaris_checkout():
        return True
    return not real_ttnn_available()


def mode_description() -> str:
    """One line naming the mode AND why, for logs and error messages.

    The reason is the point: every past confusion here has been someone unable
    to tell which branch they were on, or why.
    """
    forced = os.getenv('TTSIM_FORCE_POLARIS')
    if forced is not None:
        which = 'POLARIS (analytical)' if is_polaris() else 'HARDWARE (real ttnn)'
        return f'{which} — forced by TTSIM_FORCE_POLARIS={forced!r}'
    ird = os.getenv('IRD_ARCH_NAME', '').strip()
    if ird:
        return f'HARDWARE (real ttnn) — IRD_ARCH_NAME={ird!r} is set'
    if polaris_checkout():
        return 'POLARIS (analytical) — `ttsim` is importable, so this is a polaris tree'
    if real_ttnn_available():
        return 'HARDWARE (real ttnn) — no `ttsim`, but `ttnn` is importable (IRD layout)'
    return 'POLARIS (analytical) — neither `ttsim` nor `ttnn` is importable'


IS_POLARIS = is_polaris()

if not IS_POLARIS and not real_ttnn_available():
    # Guarded here rather than in the runners: the `import torch` / `import ttnn`
    # immediately below would otherwise raise first, and a runner-side check
    # could never fire.  That ordering is exactly how a stale IRD_ARCH_NAME once
    # surfaced as an unexplained "No module named 'torch'".
    raise SystemExit(
        f'ResNet-50: the HARDWARE path was selected ({mode_description()}), but '
        'real ttnn is not importable here.\n'
        '  - For an analytical polaris run:  TTSIM_FORCE_POLARIS=1 python polaris.py ...\n'
        '  - For a hardware run, use an IRD box with tt-metal built.\n'
        'Note: unsetting IRD_ARCH_NAME no longer forces the analytical path — the '
        'mode follows whether a top-level `ttnn` is importable, so that an IRD box '
        'without the variable set still runs on silicon instead of dying on '
        '`import ttsim`.'
    )

if not IS_POLARIS:
    import torch  # type: ignore[import-not-found]
    import ttnn  # type: ignore[import-not-found]
    from models.common.utility_functions import (  # type: ignore[import-not-found]
        _nearest_y,
        is_blackhole,
        is_wormhole_b0,
        nearest_32,
    )
    from models.demos.vision.classification.resnet50.ttnn_resnet.tt.ttnn_functional_resnet50_model_utils import (  # type: ignore[import-not-found]
        is_blackhole_p100,
    )
else:
    import ttsim.front.ttnn as ttnn
    from ttsim.front.ttnn.op import fold_transpose_sharded_nchw as _shim_fold_transpose_sharded_nchw

    # Mirrors tt-metal/models/common/utility_functions.py.  Defined locally
    # rather than imported from a shared helper module: the superseded fork kept
    # these in a workload-level utils module that this change removed, so there
    # is no in-repo path to import them from.
    def _nearest_y(x, y):
        return math.ceil(x / y) * y

    def nearest_32(x):
        return math.ceil(x / 32) * 32

    def is_wormhole_b0():
        return ttnn.get_arch_name() == 'wormhole_b0'

    def is_blackhole():
        return ttnn.get_arch_name() == 'blackhole'

    def is_blackhole_p100(device: Any) -> bool:
        # Mirrors tt-metal/models/demos/vision/classification/resnet50/
        # ttnn_resnet/tt/ttnn_functional_resnet50_model_utils.py:8-12.
        gx, gy = _grid_xy(device)
        return is_blackhole() and gx * gy != 130


def _grid_xy(device: Any) -> tuple[int, int]:
    """``device.compute_with_storage_grid_size()`` as an ``(x, y)`` pair.

    Divergence note 8: real ttnn returns a CoreCoord-like object exposing
    ``.x``/``.y`` (canonical reads it that way at C:525, C:532, C:539), while
    the shim returns a plain tuple (``ttsim/front/ttnn/device.py:91-96``).
    """
    grid = device.compute_with_storage_grid_size()
    if isinstance(grid, (tuple, list)):
        return grid[0], grid[1]
    return grid.x, grid.y


def _emit_conv2d(**kwargs):
    """``ttnn.conv2d``.

    This used to route 1x1 stride-2 convs — ResNet-50's layer2/3/4 downsamples
    — to a workload-local Conv-only dispatch, alongside a workload-local copy
    of tt-metal's ``use_matmul_for_1x1_conv`` predicate
    (``tt-metal/ttnn/cpp/ttnn/operations/conv/conv2d/conv2d_utils.cpp:519-531``).  Both existed because the shim's
    ``_conv2d_dispatch`` tested the kernel size alone: it lowered every 1x1 to
    MatMul, while the ``_with_halo`` wrapper around it had already decided to
    emit a Halo, producing a Halo + MatMul pair that cannot occur on silicon.

    The shim now shares one predicate (``_lowers_to_matmul``, the same port of
    ``use_matmul_for_1x1_conv``) between ``_with_halo`` and
    ``_conv2d_dispatch``, and that dispatch applies both post-conv passes.  So
    plain ``ttnn.conv2d`` reaches the same decision on both the Polaris and HW
    paths, and the whole workaround is gone.  Verified after removal: the three
    downsamples still emit Halo + Conv2d, and the op census is unchanged.
    """
    return ttnn.conv2d(**kwargs)


# ---------------------------------------------------------------------------
# conv2d return-contract shim
# ---------------------------------------------------------------------------

def _conv2d(
    *,
    return_output_dim: bool = False,
    return_weights_and_bias: bool = False,
    **kwargs,
):
    """``ttnn.conv2d`` with canonical's return contract on both paths.

    Real ttnn returns a tuple whose shape depends on the two ``return_*`` flags.
    The Polaris shim returns a bare ``Tensor`` and ignores both flags, so the
    output dims are computed here with the same convolution arithmetic canonical
    applies at C:468-473, and the weight/bias tensors are passed straight back
    (the shim's ``prepare_conv_weights``/``prepare_conv_bias`` are identity —
    ``ttsim/front/ttnn/__init__.py:88-93`` — so there is no device-cached
    variant to return).

    Returns, matching canonical:
      - ``out``                                    both flags False
      - ``out, [h, w]``                            output_dim only
      - ``out, [weight, bias]``                    weights_and_bias only
      - ``out, [h, w], [weight, bias]``            both
    """
    if not IS_POLARIS:
        return ttnn.conv2d(
            **kwargs,
            return_output_dim=return_output_dim,
            return_weights_and_bias=return_weights_and_bias,
        )

    out = _emit_conv2d(**kwargs)

    if not (return_output_dim or return_weights_and_bias):
        return out

    result = [out]

    if return_output_dim:
        pad_h, pad_w = kwargs['padding']
        k_h, k_w = kwargs['kernel_size']
        stride_h, stride_w = kwargs['stride']
        dil_h, dil_w = kwargs.get('dilation', (1, 1))
        h = (kwargs['input_height'] + 2 * pad_h - dil_h * (k_h - 1) - 1) // stride_h + 1
        w = (kwargs['input_width'] + 2 * pad_w - dil_w * (k_w - 1) - 1) // stride_w + 1
        result.append([h, w])

    if return_weights_and_bias:
        result.append([kwargs['weight_tensor'], kwargs['bias_tensor']])

    return tuple(result)


if IS_POLARIS:
    # -----------------------------------------------------------------------
    # conv1 fold, emitted the way silicon does it.
    #
    # `ttnn.fold(use_transpose_as_fold=True)` is a COMPOSITE on hardware: the
    # p100a capture shows it as seven profiled ops, not one.
    #
    # The code below does NOT use the shim's first-class Fold SimOp.  An earlier
    # revision did, and that is what this block exists to replace: that SimOp
    # emits a single row, so the graph carried Fold + 2 Reshape where hardware
    # ran Pad, Transpose, Pad, Transpose, Transpose, Transpose, Slice — a 9-op
    # gap, and the entire op-sequence divergence from the capture at this
    # position.  The Polaris path now calls ``fold_transpose_sharded_nchw``
    # (``ttsim/front/ttnn/op.py:1381``), which emits those seven.  The shim's
    # ``fold`` is deliberately left alone so ViT's ``Fold`` SimOp is unaffected
    # (see that helper's docstring at ``ttsim/front/ttnn/op.py:1399-1400``).
    #
    # Every shape below is READ OFF the capture
    # (__refrun_cache/resnet50/bh/p100a/merged_ops_…-260816.csv rows 0-6), not
    # derived or guessed:
    #
    #   #  op         input  (N,C,H,W)        output (N,C,H,W)
    #   0  Pad        16,  3, 224, 224   ->   16,  4, 256, 224
    #   1  Transpose  16,  4, 256, 224   ->   16,  4, 224, 256
    #   2  Pad        16,  4, 224, 256   ->   16,  4, 256, 256
    #   3  Transpose  16,  4, 256, 256   ->   16,256,   4, 256
    #   4  Transpose  16,128,   8, 256   ->   16,128, 256,   8
    #   5  Transpose  16,128, 128,  16   ->   16,128, 128,  16
    #   6  Slice      16,128, 128,  16   ->   16,115, 115,  16
    #
    # Note the gaps between #3's output and #4's input, and #4's output and
    # #5's input: hardware reshapes there, and the profiler records no row for
    # it because it is metadata only (256*4 == 128*8, 256*8 == 128*16).  Those
    # are reproduced below as in-place shape rewrites, NOT as ttnn.reshape —
    # emitting a Reshape SimOp would put back two of the rows this change
    # exists to remove.
    #
    # #5 transposes dims 1 and 2, which are both 128, so its shape is
    # unchanged.  That is not a no-op on hardware (the data moves) and it has
    # its own LUT entry, so it is emitted.
    #
    # The final (16, 115, 115, 16) is NHWC and feeds conv1 directly: 115 is
    # conv1_input_height (C:466) and 16 is conv1's in-channels, matching the
    # [64, 16, 4, 4] folded weight.
    # -----------------------------------------------------------------------


# ---------------------------------------------------------------------------
# SHIM WORKAROUNDS ON THE POLARIS PATH
# ---------------------------------------------------------------------------
# Four places below sit inside ``if IS_POLARIS:`` and do something the HW
# branch does not.  Dual-mode cannot catch drift at these four points, because
# the two halves are deliberately different there -- so each is listed with the
# shim gap it stands in for, and what closing that gap would look like.
#
# NARROW -- fixable in the shim without touching other workloads:
#
#   1. fold.  P:1233 calls ``fold_transpose_sharded_nchw`` directly instead of
#      ``ttnn.fold(..., use_transpose_as_fold=True)``.
#      Gap: that flag's path issues a 6-D reshape
#      (ttsim/front/ttnn/op.py:2685) which the shim's own ``reshape_pp``
#      rejects -- ``assert len(args_list) <= 3`` at
#      ttsim/front/ttnn/op.py:309 -- so the call raises.
#      Close it by having ``ttnn.fold`` dispatch to
#      ``fold_transpose_sharded_nchw`` when the flag is set.  ViT is the only
#      other ``ttnn.fold`` caller and does not pass it, so nothing else moves.
#
#   2. untilize_with_unpadding.  P:1635 calls the internal
#      ``untilize_with_unpadding_op`` instead of the public
#      ``ttnn.untilize_with_unpadding``.
#      The public entry point IS exported (ttsim/front/ttnn/__init__.py:33) and
#      DOES take ``output_tensor_end`` and ``memory_config``
#      (ttsim/front/ttnn/ttnn_shim.py:1271), so the signature is not the gap.
#      The recorded reason -- that it expands into an untilize/retilize
#      round-trip -- traces to a reshape path reached only when the input rank
#      exceeds 4 (ttsim/front/ttnn/ttnn_shim.py:1318).  This call site passes a
#      rank-4 ``desired_shape``, so that branch should not be taken and the
#      bypass may be stale.  Re-test against the public entry point before
#      keeping it; if it emits the single UntilizeWithUnpadding the capture's
#      row 114 shows, this branch can go away entirely.
#
# FOLLOW-UP -- these change shim behaviour many workloads share:
#
#   3. ``_reinterpret_shape`` in place of ``ttnn.reshape`` (P:409, P:1660).
#      Gap: ``ttnn.reshape`` always emits a Reshape SimOp, and hardware does
#      these regroupings for free -- the capture has no row for any of them.
#      Closing it needs a shim-level "relabel without emitting", which every
#      workload's op count would feel.
#
#   4. Hand-set ``x.layout`` / ``x.hw_shape`` after the pool (P:1560, P:1583).
#      Gap: the shim's ``avg_pool2d`` (ttsim/front/ttnn/op.py:2465) does not
#      honour ``output_layout``, which canonical sets at C:903, and the
#      downstream untilize rejects non-tile input
#      (ttsim/front/ttnn/ttnn_shim.py:1295).  Honouring the kwarg is the real
#      fix and affects every pooling workload.
# ---------------------------------------------------------------------------



    def _reinterpret_shape(tensor: Any, new_shape: Sequence[int]) -> Any:
        """Relabel a tensor's logical shape WITHOUT emitting an op.

        Hardware performs these regroupings for free — the p100a capture has no
        profiler row for any of them — so emitting a ``Reshape`` SimOp would
        create a polaris-only op — one with no counterpart in the capture — at a
        position where nothing actually runs.

        The honest caveat: at the post-fold site this relabels NHWC as NCHW,
        which for real data is a permute, not a reshape.  Polaris models shapes
        and costs, not values, and the reorder is absorbed into the conv
        reader's addressing on hardware, so no op is the correct *cost* model.
        It would be wrong in a simulator that computed values.

        Safe only because each of these tensors is consumed exactly once; a
        shared tensor would need a real op or an alias node.
        """
        assert math.prod(int(d) for d in tensor.shape) == math.prod(int(d) for d in new_shape), (
            f'_reinterpret_shape is element-count preserving; '
            f'{list(tensor.shape)} -> {list(new_shape)} is not'
        )
        tensor.shape = list(new_shape)
        return tensor



def _relu_activation():
    """Canonical's fused conv ReLU, spelled for whichever stack is active.

    See divergence note 1.  C:188, C:230, C:442.
    """
    return 'relu' if IS_POLARIS else ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)


def _sharded_mem_config_(shape: Sequence[int], grid: Any, mem_layout: Any,
                         orientation: Any, **kwargs: Any) -> Any:
    """``ttnn.create_sharded_memory_config_`` across both stacks.

    Divergence note 9: the shim's signature is
    ``(shape, grid, mem_layout, orientation, tile_layout)``
    (``ttsim/front/ttnn/memory.py:144``) and has no
    ``use_height_and_width_as_shard_shape`` parameter.  The flag is forwarded on
    the HW path, where it is load-bearing, and dropped on the Polaris path.

    ``shape`` is no longer ignored on the Polaris path: a 2-element sequence is
    stored as the per-core shard extent, which is what the three block/width
    config sites below already pass.  Anything else — including the full
    ``x.shape`` passed at P:1343 — still falls back to the ``(1, 1)``
    placeholder, so that call site is unchanged.  Pinned by
    tests/test_ttnn/test_sharded_memory_config_builders.py.
    """
    if IS_POLARIS:
        return ttnn.create_sharded_memory_config_(
            shape, grid, mem_layout, orientation, kwargs.get('tile_layout', False),
        )
    return ttnn.create_sharded_memory_config_(
        shape, grid, mem_layout, orientation, **kwargs,
    )


def _shard_dims(tensor: Any) -> tuple[int, int]:
    """The (height, width) a shard-shape expression should be built from.

    Canonical computes shard extents straight off ``x.shape[2]`` / ``x.shape[3]``
    because on hardware ``x`` is flattened NHWC ``(1, 1, N*H*W, C)`` at these
    points.  The Polaris path carries the logical shape as NCHW (conv_sinf reads
    C_in from dim 1 — see the fold note), so the same expression reads H and W:
    the pre-layer4 block config came out ``[nearest_32(14 // 10), 14 // 8]`` =
    ``[32, 1]`` instead of ``[nearest_32(3136 // 10), 1024 // 8]`` =
    ``[320, 128]``.

    ``hw_shape`` is the NHWC-flat view the shim already tracks for LUT keys, so
    read the dims from there when it is present and fall back to the logical
    shape otherwise.  HW path is untouched — real ttnn tensors have no
    ``hw_shape`` and are already flat.
    """
    hw = getattr(tensor, 'hw_shape', None)
    if hw is not None and len(hw) >= 4:
        return int(hw[2]), int(hw[3])
    return int(tensor.shape[2]), int(tensor.shape[3])


def _conv2d_config(**kwargs):
    """``ttnn.Conv2dConfig``, dropping fields the shim does not model.

    Only ``enable_activation_reuse`` is affected (divergence note 2); it is
    forwarded verbatim on the HW path.
    """
    if IS_POLARIS:
        kwargs.pop('enable_activation_reuse', None)
    return ttnn.Conv2dConfig(**kwargs)


# ---------------------------------------------------------------------------
# ResnetLinear — mirrors C:18-55
# ---------------------------------------------------------------------------

def ResnetLinear(
    weight,
    bias,
    output_mem_config,
    model_config,
    compute_kernel_config,
):
    """Returns a function for the linear operation in resnet, with bias.

    Mirrors C:18-55.  Canonical uses ONE hardcoded matmul program config for
    every batch size — grid (8, 4), ``in0_block_w=2`` (C:29-39).  The superseded
    fork carried a batch-keyed dict of configs from a stale upstream; that is
    deliberately not reproduced.
    """
    matmul_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(8, 4),
        in0_block_w=2,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=1,
        per_core_N=1,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )

    if not IS_POLARIS:
        # C:40-41.  The shim has no Shape.to_rank; model_preprocessing supplies
        # rank-4 fc parameters directly (divergence note 7).
        weight = weight.reshape(weight.shape.to_rank(4))
        bias = bias.reshape(bias.shape.to_rank(4))
    else:
        # model_preprocessing builds every parameter at the model's WEIGHTS_DTYPE
        # (bfloat8_b) in the default layout, but the classifier is not requantised
        # on hardware: the capture's fc matmul (row 113) records input_1 and
        # input_2 as TILE / BFLOAT16 / DRAM_INTERLEAVED.  Tag the HW view the same
        # way _matmul_1x1_with_hw_fields does for conv weights — these are
        # key-only overlays and do not touch activation dtype propagation.
        for _p in (weight, bias):
            if _p is not None:
                _p._hw_layout = ttnn.TILE_LAYOUT
                _p._hw_dtype = ttnn.bfloat16

    def linear_(act):
        # C:44-52
        output = ttnn.linear(
            act,
            weight,
            bias=bias,
            program_config=matmul_config,
            memory_config=output_mem_config,
            dtype=model_config['ACTIVATIONS_DTYPE'],
            compute_kernel_config=compute_kernel_config,
        )
        return output

    return linear_


# ---------------------------------------------------------------------------
# resnet50Bottleneck — mirrors C:58-335
# ---------------------------------------------------------------------------

class resnet50Bottleneck:
    expansion: int = 4

    def __init__(self, parameters, downsample, stride, model_config) -> None:
        # C:61-91 — init only pre-processes weight and bias tensors.
        self.conv1_weight_tensor = parameters.conv1.weight
        self.conv1_bias_tensor = parameters.conv1.bias
        self.conv1_input_channels = self.conv1_weight_tensor.shape[1]
        self.conv1_output_channels = self.conv1_weight_tensor.shape[0]
        assert self.conv1_weight_tensor.shape[2] == 1

        self.conv2_weight_tensor = parameters.conv2.weight
        self.conv2_bias_tensor = parameters.conv2.bias
        self.conv2_input_channels = self.conv2_weight_tensor.shape[1]
        self.conv2_output_channels = self.conv2_weight_tensor.shape[0]
        # NOTE: there is deliberately no ``self.conv2_stride`` here.  An earlier
        # version computed ``2 if downsample else 1`` and never read it, which
        # was as well: layer1's first block has downsample=True with stride=1
        # (P:1147 ``_make_layer`` — it downsamples on the channel change,
          # 64 -> 256, not on
        # space), so that expression disagrees with the stride conv2 must use.
        # Both conv2 call sites read ``self.stride``, the constructor argument.
        assert self.conv2_weight_tensor.shape[2] == 3

        self.conv3_weight_tensor = parameters.conv3.weight
        self.conv3_bias_tensor = parameters.conv3.bias
        self.conv3_input_channels = self.conv3_weight_tensor.shape[1]
        self.conv3_output_channels = self.conv3_weight_tensor.shape[0]
        assert self.conv3_weight_tensor.shape[2] == 1

        self.downsample = downsample
        self.stride = stride
        if downsample:
            self.ds_conv_weight_tensor = parameters.downsample.weight
            self.ds_conv_bias_tensor = parameters.downsample.bias
            self.ds_conv_input_channels = self.ds_conv_weight_tensor.shape[1]
            self.ds_conv_output_channels = self.ds_conv_weight_tensor.shape[0]
            assert self.ds_conv_weight_tensor.shape[2] == 1
        self.model_config = model_config
        return

    def run_downsample_if_req(
        self,
        x,
        device,
        batch_size,
        input_height,
        input_width,
        reshard_if_not_optimal=False,
        height_sharding=None,
        packer_l1_accum_enabled=True,
    ):
        # C:93-151
        if self.downsample:
            logger.debug('Running downsample')
            conv_kwargs = {
                'in_channels': self.ds_conv_input_channels,
                'out_channels': self.ds_conv_output_channels,
                'batch_size': batch_size,
                'input_height': input_height,
                'input_width': input_width,
                'kernel_size': (1, 1),
                'stride': (self.stride, self.stride),
                'padding': (0, 0),
                'dilation': (1, 1),
                'groups': 1,
                'device': device,
                'conv_config': _conv2d_config(
                    weights_dtype=self.model_config['WEIGHTS_DTYPE'],
                    # C:120-124 — the `input_height != 28` guard is load-bearing;
                    # it forces BLOCK sharding on the 28x28 downsample even when
                    # the caller asked for height sharding.
                    shard_layout=(
                        ttnn.TensorMemoryLayout.HEIGHT_SHARDED
                        if height_sharding and input_height != 28
                        else ttnn.TensorMemoryLayout.BLOCK_SHARDED
                    ),
                    deallocate_activation=True,
                    reallocate_halo_output=False,
                    reshard_if_not_optimal=reshard_if_not_optimal,
                    enable_act_double_buffer=(
                        True if not (is_blackhole_p100(device) and batch_size > 16) else False
                    ),
                    enable_weights_double_buffer=True if input_width < 56 else False,
                    full_inner_dim=True,
                    enable_activation_reuse=True if height_sharding and self.stride == 1 else False,
                ),
            }

            ds_out, [self.ds_conv_weight_tensor, self.ds_conv_bias_tensor] = _conv2d(
                input_tensor=x,
                weight_tensor=self.ds_conv_weight_tensor,
                bias_tensor=self.ds_conv_bias_tensor,
                **conv_kwargs,
                compute_config=ttnn.init_device_compute_kernel_config(
                    device.arch(),
                    math_fidelity=self.model_config['MATH_FIDELITY'],
                    packer_l1_acc=packer_l1_accum_enabled,
                ),
                return_output_dim=False,
                return_weights_and_bias=True,
                dtype=self.model_config['ACTIVATIONS_DTYPE'],
            )
        else:
            ds_out = x
        return ds_out

    def __call__(
        self,
        x,
        device,
        batch_size,
        input_height,
        input_width,
        reshard_if_not_optimal=False,
        height_sharding=None,
        packer_l1_acc=True,
        layer_module=None,
    ):
        # C:153-335
        logger.debug(
            f'==== Running {batch_size}, {input_height}, {input_width}, '
            f'{self.conv1_input_channels}, {self.conv1_output_channels}'
        )

        ds_input_height = input_height
        ds_input_width = input_width

        # conv1 is 1x1 conv — C:172-209
        logger.debug('Running conv1')
        conv_kwargs_1 = {
            'in_channels': self.conv1_input_channels,
            'out_channels': self.conv1_output_channels,
            'batch_size': batch_size,
            'input_height': input_height,
            'input_width': input_width,
            'kernel_size': (1, 1),
            'stride': (1, 1),
            'padding': (0, 0),
            'dilation': (1, 1),
            'groups': 1,
            'device': device,
            'conv_config': _conv2d_config(
                weights_dtype=self.model_config['WEIGHTS_DTYPE'],
                activation=_relu_activation(),
                shard_layout=(
                    ttnn.TensorMemoryLayout.HEIGHT_SHARDED
                    if height_sharding
                    else ttnn.TensorMemoryLayout.BLOCK_SHARDED
                ),
                reshard_if_not_optimal=reshard_if_not_optimal,
            ),
        }

        out, [input_height, input_width], [self.conv1_weight_tensor, self.conv1_bias_tensor] = _conv2d(
            input_tensor=x,
            weight_tensor=self.conv1_weight_tensor,
            bias_tensor=self.conv1_bias_tensor,
            **conv_kwargs_1,
            compute_config=ttnn.init_device_compute_kernel_config(
                device.arch(),
                math_fidelity=self.model_config['MATH_FIDELITY'],
                packer_l1_acc=packer_l1_acc,
            ),
            return_output_dim=True,
            return_weights_and_bias=True,
            dtype=self.model_config['ACTIVATIONS_DTYPE'],
        )

        act_block_h_override = 0
        ds_out = None

        logger.debug('Running conv2')

        # C:216-243
        conv_kwargs_2 = {
            'in_channels': self.conv2_input_channels,
            'out_channels': self.conv2_output_channels,
            'batch_size': batch_size,
            'input_height': input_height,
            'input_width': input_width,
            'kernel_size': (3, 3),
            'stride': (self.stride, self.stride),
            'padding': (1, 1),
            'dilation': (1, 1),
            'groups': 1,
            'device': device,
            'conv_config': _conv2d_config(
                weights_dtype=self.model_config['WEIGHTS_DTYPE'],
                activation=_relu_activation(),
                deallocate_activation=True,
                reallocate_halo_output=False,
                act_block_h_override=act_block_h_override,
                shard_layout=(
                    ttnn.TensorMemoryLayout.HEIGHT_SHARDED
                    if height_sharding
                    else ttnn.TensorMemoryLayout.BLOCK_SHARDED
                ),
                reshard_if_not_optimal=reshard_if_not_optimal,
                enable_act_double_buffer=True,
                enable_weights_double_buffer=True,
                full_inner_dim=True,
                enable_activation_reuse=True if height_sharding and self.stride == 1 else False,
            ),
        }

        # Per-module act_block_h overrides — C:245-258.  These are keyed on
        # layer_module, so every call site must pass it (canonical passes it for
        # all 16 blocks).
        if is_blackhole():
            if layer_module == 'layer1_module3':
                conv_kwargs_2['conv_config'].act_block_h_override = 16 * 32
            if batch_size == 32 and is_blackhole_p100(device):
                if layer_module in ('layer1_module2', 'layer1_module3', 'layer2_module1'):
                    conv_kwargs_2['conv_config'].act_block_h_override = 32

        if is_wormhole_b0():
            if layer_module in ('layer1_module2', 'layer1_module3'):
                conv_kwargs_2['conv_config'].act_block_h_override = 14 * 32

        out, [input_height, input_width], [self.conv2_weight_tensor, self.conv2_bias_tensor] = _conv2d(
            input_tensor=out,
            weight_tensor=self.conv2_weight_tensor,
            bias_tensor=self.conv2_bias_tensor,
            **conv_kwargs_2,
            compute_config=ttnn.init_device_compute_kernel_config(
                device.arch(),
                math_fidelity=self.model_config['MATH_FIDELITY'],
                packer_l1_acc=packer_l1_acc,
            ),
            return_output_dim=True,
            return_weights_and_bias=True,
            dtype=self.model_config['ACTIVATIONS_DTYPE'],
        )

        # conv3 is 1x1 conv — C:275-312
        logger.debug('Running conv3')
        conv_kwargs_3 = {
            'in_channels': self.conv3_input_channels,
            'out_channels': self.conv3_output_channels,
            'batch_size': batch_size,
            'input_height': input_height,
            'input_width': input_width,
            'kernel_size': (1, 1),
            'stride': (1, 1),
            'padding': (0, 0),
            'dilation': (1, 1),
            'groups': 1,
            'device': device,
            'conv_config': _conv2d_config(
                weights_dtype=self.model_config['WEIGHTS_DTYPE'],
                shard_layout=(
                    ttnn.TensorMemoryLayout.HEIGHT_SHARDED
                    if height_sharding
                    else ttnn.TensorMemoryLayout.BLOCK_SHARDED
                ),
                reshard_if_not_optimal=reshard_if_not_optimal,
                deallocate_activation=True,
            ),
        }

        out, [self.conv3_weight_tensor, self.conv3_bias_tensor] = _conv2d(
            input_tensor=out,
            weight_tensor=self.conv3_weight_tensor,
            bias_tensor=self.conv3_bias_tensor,
            **conv_kwargs_3,
            compute_config=ttnn.init_device_compute_kernel_config(
                device.arch(),
                math_fidelity=self.model_config['MATH_FIDELITY'],
                packer_l1_acc=packer_l1_acc,
            ),
            return_output_dim=False,
            return_weights_and_bias=True,
            dtype=self.model_config['ACTIVATIONS_DTYPE'],
        )

        # C:314-323
        ds_out = self.run_downsample_if_req(
            x,
            device,
            batch_size,
            ds_input_height,
            ds_input_width,
            reshard_if_not_optimal,
            height_sharding,
            packer_l1_accum_enabled=packer_l1_acc,
        )

        # C:325-326 — a real reshard when the configs differ, NOT an assert.
        # Exactly canonical (C:325-326).  This previously used
        # ttnn.get_memory_config(), which is NOT equivalent in the shim:
        # Tensor.memory_config() returns _memory_config and may be None, while
        # get_memory_config() substitutes MemoryConfig.L1 for None
        # (ttsim/front/ttnn/memory.py:134-138).  That substitution made the
        # comparison read L1 != L1 and skipped the reshard where canonical's
        # None != <cfg> fires — the refrun shows a Reshard at this position in
        # every downsampling bottleneck (rows 18, 39, ... of the p100a capture).
        if ds_out.memory_config() != out.memory_config():
            ds_out = ttnn.to_memory_config(ds_out, out.memory_config())

        # C:328-333 — residual add with fused ReLU.  Canonical uses the in-place
        # ttnn.add_; the shim's fused equivalent emits a single Add SimOp
        # carrying the activation (divergence note 4).  Dropping the ReLU here
        # would silently change the network's arithmetic on all 16 blocks.
        if IS_POLARIS:
            out = ttnn.add(
                out,
                ds_out,
                input_tensor_a_activations=[ttnn.UnaryOpType.RELU],
            )
        else:
            out = ttnn.add_(
                out,
                ds_out,
                activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)],
            )
        ttnn.deallocate(ds_out)
        return out, input_height, input_width


# ---------------------------------------------------------------------------
# resnet50 — mirrors C:338-939
# ---------------------------------------------------------------------------

class resnet50:
    def __init__(
        self,
        device,
        parameters,
        batch_size,
        model_config,
        input_shape,
        kernel_size,
        stride,
        dealloc_input=True,
        final_output_mem_config=None,
    ) -> None:
        # C:339-563.
        #
        # NOTE on `kernel_size` / `stride`: these are the FOLD padding and
        # stride (C:476-485), not the conv1 kernel.  The test infra passes 3 and
        # 2 respectively (resnet50_test_infra.py:194-195, 241-242).  The
        # superseded fork defaulted them to 4 and 1, which is what produced its
        # wrong first-conv geometry.
        super().__init__()
        if final_output_mem_config is None:
            final_output_mem_config = ttnn.L1_MEMORY_CONFIG
        layers = [3, 4, 6, 3]
        self.device = device
        self.batch_size = batch_size
        self.model_config = model_config
        self.inplanes = 64
        self.final_output_mem_config = final_output_mem_config
        compute_kernel_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=model_config['MATH_FIDELITY'],
            math_approx_mode=True,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        self.conv1_weight_tensor = parameters.conv1.weight
        self.conv1_bias_tensor = parameters.conv1.bias
        self.conv1_input_channels = self.conv1_weight_tensor.shape[1]
        self.conv1_output_channels = self.conv1_weight_tensor.shape[0]
        # C:371.  Canonical folds the torchvision 7x7/stride-2 conv1 into a 4x4
        # via pad_and_fold_conv_filters_for_unity_stride(w, 2, 2), which also
        # takes in-channels to _nearest_y(3, 4) * 2 * 2 = 16.  So the expected
        # weight is [64, 16, 4, 4] — this assert only checks the spatial half of
        # that, and passes on a wrong channel count, which is how the superseded
        # fork's [64, 3, 4, 4] survived.  model_preprocessing asserts the rest.
        assert self.conv1_weight_tensor.shape[2] == 4

        self.layer1 = self._make_layer(
            parameters=parameters.layer1, planes=64, blocks=layers[0], stride=1, model_config=model_config,
        )
        self.layer2 = self._make_layer(
            parameters=parameters.layer2, planes=128, blocks=layers[1], stride=2, model_config=model_config,
        )
        self.layer3 = self._make_layer(
            parameters=parameters.layer3, planes=256, blocks=layers[2], stride=2, model_config=model_config,
        )
        self.layer4 = self._make_layer(
            parameters=parameters.layer4, planes=512, blocks=layers[3], stride=2, model_config=model_config,
        )

        # C:402-422 — all modules unrolled; layers MUST equal [3, 4, 6, 3].
        assert layers == [3, 4, 6, 3]
        self.layer1_module1 = self.layer1[0]
        self.layer1_module2 = self.layer1[1]
        self.layer1_module3 = self.layer1[2]

        self.layer2_module1 = self.layer2[0]
        self.layer2_module2 = self.layer2[1]
        self.layer2_module3 = self.layer2[2]
        self.layer2_module4 = self.layer2[3]

        self.layer3_module1 = self.layer3[0]
        self.layer3_module2 = self.layer3[1]
        self.layer3_module3 = self.layer3[2]
        self.layer3_module4 = self.layer3[3]
        self.layer3_module5 = self.layer3[4]
        self.layer3_module6 = self.layer3[5]

        self.layer4_module1 = self.layer4[0]
        self.layer4_module2 = self.layer4[1]
        self.layer4_module3 = self.layer4[2]

        # C:424-430 — num_classes = 1000
        self.fc = ResnetLinear(
            weight=ttnn.to_device(parameters.fc.weight, device),
            bias=ttnn.to_device(parameters.fc.bias, device),
            output_mem_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            model_config=model_config,
            compute_kernel_config=compute_kernel_config,
        )

        # C:432-461
        act_block_h_override = 0

        if is_wormhole_b0():
            act_block_h_override = 1568

        if is_blackhole() and self.batch_size == 32:
            act_block_h_override = 32 * 32 if is_blackhole_p100(device) else 49 * 32

        self.conv1_config = _conv2d_config(
            weights_dtype=self.model_config['WEIGHTS_DTYPE'],
            activation=_relu_activation(),
            deallocate_activation=dealloc_input,
            act_block_h_override=act_block_h_override,
            enable_act_double_buffer=True,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            reshard_if_not_optimal=False,
            # otherwise act block h is not big enough for the reuse
            enable_activation_reuse=(not is_wormhole_b0() or device.get_num_devices() <= 8),
        )
        self.conv1_compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=self.model_config['MATH_FIDELITY'],
            packer_l1_acc=True,
        )
        if is_wormhole_b0():
            # Issue #13145: Temp workaround for Galaxy to avoid hangs
            if device.get_num_devices() > 8:
                self.conv1_config.act_block_h_override = 64
            else:
                self.conv1_config.act_block_h_override = 49 * 32

        # C:463-473
        self.conv1_kernel_size = (4, 4)
        self.conv1_stride = (1, 1)
        self.conv1_padding = (0, 0)
        self.conv1_input_height = 115
        self.conv1_input_width = 115
        self.conv1_output_height = (
            (self.conv1_input_height - self.conv1_kernel_size[0] + 2 * self.conv1_padding[0])
            // self.conv1_stride[0]
        ) + 1
        self.conv1_output_width = (
            (self.conv1_input_width - self.conv1_kernel_size[1] + 2 * self.conv1_padding[1])
            // self.conv1_stride[1]
        ) + 1

        # fold params — C:475-491
        self.fold_stride_h = stride
        self.fold_stride_w = stride
        _, c, h, w = input_shape
        n = batch_size
        h += kernel_size * 2
        w += kernel_size * 2
        C = _nearest_y(c, 4)
        self.fold_pad_c = C - c
        self.fold_pad_h = kernel_size
        self.fold_pad_w = kernel_size
        self.fold_output_shape = (
            n,
            h // self.fold_stride_h,
            w // self.fold_stride_w,
            C * (self.fold_stride_h * self.fold_stride_w),
        )

        # C:492-519.  Canonical has no `else` arm here: batch sizes other than
        # 16/20/32 simply leave fold_compute_grid_size unset.  The superseded
        # fork added an `exit(0)`, which silently killed the process at the
        # default batch size.  Upstream skips bs<=2, bs==8, and WH bs==20
        # outright (resnet50_test_infra.py:196-201).
        num_cores_x = 8
        num_cores_y = 8
        self.fold_compute_grid_size = None
        if self.batch_size == 16:
            num_cores_x = 8
            num_cores_y = 8
            self.fold_compute_grid_size = ttnn.CoreRangeSet(
                {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores_x - 1, num_cores_y - 1))}
            )
        elif self.batch_size == 20:
            if is_wormhole_b0():
                num_cores_x = 8
                num_cores_y = 5
            elif is_blackhole():
                num_cores_x = 10
                num_cores_y = 8
            self.fold_compute_grid_size = ttnn.CoreRangeSet(
                {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores_x - 1, num_cores_y - 1))}
            )
        elif self.batch_size == 32:
            core_grid = ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(12, 8)),
                    ttnn.CoreRange(ttnn.CoreCoord(0, 9), ttnn.CoreCoord(10, 9)),
                }
            )
            if is_blackhole_p100(device):
                core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))})
            self.fold_compute_grid_size = core_grid

        if not IS_POLARIS:
            # C:521-522.  Polaris builds no host tensors.
            conv_dummy_tensor = torch.rand((self.fold_output_shape), dtype=torch.bfloat16)
            conv_dummy_tensor = ttnn.from_torch(conv_dummy_tensor, layout=ttnn.ROW_MAJOR_LAYOUT)

        # Sharded memory config for the fold — C:524-563
        compute_grid = device.compute_with_storage_grid_size()
        compute_grid_x, compute_grid_y = _grid_xy(device)
        # The shim's Device defaults its compute grid to 0x0 (a placeholder —
        # ttsim/front/ttnn/device.py:70-72), so the caller must configure it
        # from the arch YAML's `compute_grid_size` before building the model.
        # Failing loudly beats silently defaulting to some plausible grid: the
        # value feeds the fold shard shape and, on BH, the core-count search.
        assert compute_grid_x > 0 and compute_grid_y > 0, (
            'device compute grid is unset '
            f'({compute_grid_x}x{compute_grid_y}); call '
            'device.compute_with_storage_grid_size(x, y) with this SKU\'s '
            "`compute_grid_size` from config/tt_{wh,bh}.yaml before "
            'constructing resnet50'
        )

        if is_blackhole():
            # Override num cores to avoid padding issues
            nhw_ntiles = math.ceil(
                self.batch_size * self.conv1_output_height * self.conv1_output_width / 32
            )
            num_cores_target = compute_grid_x * compute_grid_y
            while nhw_ntiles % num_cores_target != 0:
                num_cores_target -= 1
            core_grid = ttnn.num_cores_to_corerangeset(num_cores_target, compute_grid, row_wise=True)
        else:
            # Use full grid
            core_grid = ttnn.CoreRangeSet(
                {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid_x - 1, compute_grid_y - 1))}
            )

        input_channels_padded = (
            nearest_32(self.conv1_input_channels)
            if self.conv1_input_channels % 8 != 0
            else self.conv1_input_channels
        )
        if input_channels_padded % 8 != 0:
            input_channels_padded = ((input_channels_padded + 7) // 8) * 8

        tensor_height = self.conv1_input_width * self.conv1_input_height * self.batch_size
        tensor_width = input_channels_padded

        num_cores = core_grid.num_cores()
        shard_height = math.ceil(tensor_height / num_cores)
        shard_width = tensor_width

        self.override_fold_mem_config = ttnn.create_sharded_memory_config(
            shape=(1, 1, shard_height, shard_width),
            core_grid=core_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

    def __del__(self):
        # C:565-567 — nothing to do
        pass

    def _make_layer(
        self,
        parameters,
        planes: int,
        blocks: int,
        stride: int,
        model_config=None,
    ) -> List[resnet50Bottleneck]:
        # C:569-596
        layers = []
        layers.append(
            resnet50Bottleneck(
                parameters=parameters[0],
                downsample=stride != 1 or self.inplanes != planes * resnet50Bottleneck.expansion,
                stride=stride,
                model_config=model_config,
            )
        )
        self.inplanes = planes * resnet50Bottleneck.expansion
        for block_num in range(1, blocks):
            layers.append(
                resnet50Bottleneck(
                    parameters=parameters[block_num],
                    downsample=False,
                    stride=1,
                    model_config=model_config,
                )
            )
        return layers

    def __call__(self, input_tensor, device, ops_parallel_config=None):
        # C:598-602.  ops_parallel_config is accepted for call-compatibility and
        # ignored, exactly as canonical does — canonical's run() dropped the
        # parameter entirely.  The superseded fork subscripted it and crashed
        # when callers passed None.
        return self.run(input_tensor, device)

    ## merged runs (first and optimized)
    def run(self, input_tensor, device):
        # C:605-939
        logger.debug('==== fold on device')

        # ------------------------------------------------------------------
        # fold — C:609-617.  Canonical calls ttnn.fold with
        # use_transpose_as_fold=True and the 6-element two-sided list
        # `padding=[pad_h, pad_h, pad_w, pad_w, 0, pad_c]` (C:614), i.e.
        # H += 2*pad_h, W += 2*pad_w, C += pad_c.  The target geometry is
        # 224 + 2*3 = 230, 230 // 2 = 115 = self.conv1_input_height (C:466),
        # and channels (3 + 1) * 2 * 2 = 16 = conv1 in-channels.
        #
        # The Polaris path cannot reach that through the shim's ttnn.fold:
        #
        # (a) use_transpose_as_fold is unsupported.  The shim's transpose path
        #     issues a 6-D `reshape` that its own `reshape_pp` rejects
        #     (ttsim/front/ttnn/op.py:2685-2687 -> ttsim/front/ttnn/op.py:309,
        #     "ttnn.reshape has 3 inputs"), so it
        #     raises outright.
        # (b) Falling back to the shim's first-class Fold SimOp would emit ONE
        #     row.  The transpose form is a composite on hardware — the p100a
        #     capture records seven — so that row count is wrong on its own.
        #
        # The branch below therefore calls fold_transpose_sharded_nchw, which
        # emits the same seven ops.  Mind the pad convention, which differs
        # between the two entry points: that helper takes ONE-sided pads and
        # doubles them itself (ttsim/front/ttnn/op.py:1419-1420), whereas the
        # shim's ttnn.fold
        # applies its scalars ONCE (ttsim/front/ttnn/op.py:2673-2678) and would
        # need them
        # pre-doubled.  Handing fold_pad_h to a function expecting the other
        # convention gives (224+3)//2 = 113 and silently shrinks conv1's input
        # by two rows and columns.
        # ------------------------------------------------------------------
        if IS_POLARIS:
            # Canonical's use_transpose_as_fold=True is a composite; the p100a
            # capture records it as seven ops.  Emit those, not the shim's
            # single Fold SimOp.  Same fold parameters canonical passes (C:614)
            # — note pad_h/pad_w are the ONE-sided values here, because
            # fold_transpose_sharded_nchw doubles them itself, unlike the
            # shim's ttnn.fold which needed them pre-doubled.
            # Port of tt-metal's `fold_with_transpose_sharded_`
            # (ttnn/cpp/ttnn/operations/data_movement/fold/fold.cpp:149), which
            # dispatches Pad -> Transpose -> Pad -> Transpose -> view ->
            # Transpose -> view -> Transpose -> Slice; the two
            # ttnn::experimental::view calls are free, so silicon records SEVEN
            # rows.  It returns a logical-NCHW tensor already carrying the
            # NHWC-flat hw_shape, so no post-fold relabel is needed here.
            fold_output_tensor = _shim_fold_transpose_sharded_nchw(
                input_tensor,
                self.fold_stride_h,
                self.fold_stride_w,
                pad_h=self.fold_pad_h,
                pad_w=self.fold_pad_w,
                pad_c=self.fold_pad_c,
            )
        else:
            fold_output_tensor = ttnn.fold(
                input_tensor,
                self.fold_stride_h,
                self.fold_stride_w,
                use_transpose_as_fold=True,
                padding=[
                    self.fold_pad_h, self.fold_pad_h,
                    self.fold_pad_w, self.fold_pad_w,
                    0, self.fold_pad_c,
                ],
                grid_size=self.fold_compute_grid_size,
                override_memory_config=self.override_fold_mem_config,
            )

        # C:618-619.  Canonical names these n, c, h, w but the fold output is
        # NHWC, so the last dim is the folded channel count and the flatten
        # produces (1, 1, N*H*W, C) — a flattened-NHWC activation, which is what
        # real ttnn's conv2d consumes.
        #
        # Polaris divergence (same convention vgg_unet documents at
        # workloads/ttnn/vgg_unet/common/ttnn_vgg_unet.py:9-13): the shim's
        # `conv_sinf` reads C_in from dim 1, i.e. it expects NCHW, and does not
        # model memory reformatting.  Handing it the flattened-NHWC form makes
        # it read C_in = 1 and reject the conv.  So the Polaris path targets
        # NCHW at this same position — ONE Reshape op either way, matching
        # canonical's op count; only the target shape differs.
        # ``Tensor.shape`` is ``Shape | None`` in the ttnn API, and both branches
        # below index it.  Bind it once behind a guard rather than indexing the
        # Optional three times.
        fold_shape = fold_output_tensor.shape
        assert fold_shape is not None, 'fold output tensor has no shape'

        if IS_POLARIS:
            # Nothing to do: fold_transpose_sharded_nchw already hands back a
            # logical-NCHW tensor (conv_sinf reads C_in from dim 1) carrying the
            # NHWC-flat hw_shape the downstream Halo keys on.  The relabel that
            # used to live here is now the Slice at the end of that port, which
            # is where tt-metal puts the convention handoff too.
            assert list(fold_shape) == [
                self.batch_size, self.conv1_input_channels,
                self.conv1_input_height, self.conv1_input_width,
            ], (
                f'fold output {list(fold_shape)} != canonical '
                f'conv1 input geometry (C:466-467 fix it at 115x115)'
            )
        else:
            n, c, h, w = fold_shape
            fold_output_tensor = ttnn.reshape(fold_output_tensor, (1, 1, n * c * h, w))

        ttnn.deallocate(input_tensor)

        logger.debug('==== first conv')

        # first conv — C:626-650
        conv_kwargs = {
            'in_channels': self.conv1_input_channels,
            'out_channels': self.conv1_output_channels,
            'batch_size': self.batch_size,
            'input_height': self.conv1_input_height,
            'input_width': self.conv1_input_width,
            'kernel_size': self.conv1_kernel_size,
            'stride': self.conv1_stride,
            'padding': self.conv1_padding,
            'dilation': (1, 1),
            'groups': 1,
            'device': device,
            'conv_config': self.conv1_config,
        }

        x, [x_height, x_width], [self.conv1_weight_tensor, self.conv1_bias_tensor] = _conv2d(
            input_tensor=fold_output_tensor,
            weight_tensor=self.conv1_weight_tensor,
            bias_tensor=self.conv1_bias_tensor,
            **conv_kwargs,
            compute_config=self.conv1_compute_config,
            return_output_dim=True,
            return_weights_and_bias=True,
            dtype=self.model_config['ACTIVATIONS_DTYPE'],
        )

        # C:652-662
        x = ttnn.max_pool2d(
            input_tensor=x,
            batch_size=self.batch_size,
            input_h=x_height,
            input_w=x_width,
            channels=self.conv1_output_channels,
            kernel_size=[3, 3],
            stride=[2, 2],
            padding=[1, 1],
            dilation=[1, 1],
        )

        x_height = 56
        x_width = 56

        # C:667-677.  CoreGrid(x=8, y=7) is 56 cores; the superseded fork used a
        # 9x8=72-core CoreRangeSet here, with the correct line commented out.
        if is_wormhole_b0():
            core_range_set = ttnn.CoreGrid(x=8, y=7)
            mem_config = _sharded_mem_config_(
                x.shape,
                core_range_set,
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.ShardOrientation.ROW_MAJOR,
                tile_layout=True,
            )
            x = ttnn.to_memory_config(x, mem_config)
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT, dtype=self.model_config['ACTIVATIONS_DTYPE'])

        # ---------------- layer 1 — C:679-713 ----------------
        logger.debug('==== Running layer 1 module 1')

        reshard = is_blackhole()
        height_shard = True

        x, x_height, x_width = self.layer1_module1(
            x, device, self.batch_size, x_height, x_width,
            reshard_if_not_optimal=reshard,
            height_sharding=height_shard,
            layer_module='layer1_module1',
        )

        logger.debug('==== Running layer 1 module 2')
        x, x_height, x_width = self.layer1_module2(
            x, device, self.batch_size, x_height, x_width, layer_module='layer1_module2',
        )

        logger.debug('==== Running layer 1 module 3')
        x, x_height, x_width = self.layer1_module3(
            x, device, self.batch_size, x_height, x_width, layer_module='layer1_module3',
        )

        # ---------------- layer 2 — C:715-758 ----------------
        reshard = False
        height_shard = True

        logger.debug('==== Running layer 2 module 1')
        x, x_height, x_width = self.layer2_module1(
            x, device, self.batch_size, x_height, x_width,
            reshard_if_not_optimal=reshard,
            height_sharding=height_shard,
            layer_module='layer2_module1',
        )

        logger.debug('==== Running layer 2 module 2')
        x, x_height, x_width = self.layer2_module2(
            x, device, self.batch_size, x_height, x_width, layer_module='layer2_module2',
        )

        logger.debug('==== Running layer 2 module 3')
        x, x_height, x_width = self.layer2_module3(
            x, device, self.batch_size, x_height, x_width, layer_module='layer2_module3',
        )

        logger.debug('==== Running layer 2 module 4')
        x, x_height, x_width = self.layer2_module4(
            x, device, self.batch_size, x_height, x_width, layer_module='layer2_module4',
        )

        # ---------------- layer 3 — C:760-827 ----------------
        # Canonical: reshard is False on WH (is_blackhole() is False), and the
        # WH path instead does an explicit HEIGHT->BLOCK transition here.  The
        # superseded fork inverted both: reshard_if_not_optimal=True and no
        # transition at all.
        reshard = is_blackhole()
        height_shard = is_blackhole()
        if is_wormhole_b0():
            # NHWC-flat, not x.shape: canonical passes the shape hardware has at
            # this point, which is (1, 1, N*H*W, C).  The Polaris path carries
            # NCHW, and create_sharded_memory_config divides height by grid.y and
            # width by grid.x — fed (1, 1024, 14, 14) that yields a (1792, 2)
            # shard instead of (392, 128).  Same correction as _shard_dims makes
            # for the BLACKHOLE branch below.
            _h, _w = _shard_dims(x)
            x = ttnn.to_memory_config(
                x,
                ttnn.create_sharded_memory_config(
                    (1, 1, _h, _w), ttnn.CoreGrid(x=8, y=8), ttnn.ShardStrategy.BLOCK
                ),
            )

        logger.debug('==== Running layer 3 module 1')
        x, x_height, x_width = self.layer3_module1(
            x, device, self.batch_size, x_height, x_width,
            reshard_if_not_optimal=reshard,
            height_sharding=height_shard,
            layer_module='layer3_module1',
        )

        logger.debug('==== Running layer 3 module 2')
        x, x_height, x_width = self.layer3_module2(
            x, device, self.batch_size, x_height, x_width, layer_module='layer3_module2',
        )

        logger.debug('==== Running layer 3 module 3')
        x, x_height, x_width = self.layer3_module3(
            x, device, self.batch_size, x_height, x_width, layer_module='layer3_module3',
        )

        logger.debug('==== Running layer 3 module 4')
        x, x_height, x_width = self.layer3_module4(
            x, device, self.batch_size, x_height, x_width, layer_module='layer3_module4',
        )

        logger.debug('==== Running layer 3 module 5')
        x, x_height, x_width = self.layer3_module5(
            x, device, self.batch_size, x_height, x_width, layer_module='layer3_module5',
        )

        logger.debug('==== Running layer 3 module 6')
        x, x_height, x_width = self.layer3_module6(
            x, device, self.batch_size, x_height, x_width, layer_module='layer3_module6',
        )

        # ---------------- layer 4 — C:829-881 ----------------
        reshard = False
        height_shard = False

        if is_wormhole_b0():
            # NHWC-flat; see the pre-layer3 note.
            _h, _w = _shard_dims(x)
            block_mem_config = ttnn.create_sharded_memory_config(
                (1, 1, _h, _w),
                ttnn.CoreGrid(x=8, y=7),
                ttnn.ShardStrategy.BLOCK,
            )
            x = ttnn.to_memory_config(x, block_mem_config)
        if is_blackhole():
            grid_size = (8, 10)
            block_mem_config = _sharded_mem_config_(
                [nearest_32(_shard_dims(x)[0] // grid_size[1]),
                 _shard_dims(x)[1] // grid_size[0]],
                ttnn.CoreGrid(x=grid_size[0], y=grid_size[1]),
                ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                ttnn.ShardOrientation.ROW_MAJOR,
                tile_layout=True,
                use_height_and_width_as_shard_shape=True,
            )
            x = ttnn.to_memory_config(x, block_mem_config)

        logger.debug('==== Running layer 4 module 1')
        x, x_height, x_width = self.layer4_module1(
            x, device, self.batch_size, x_height, x_width,
            reshard_if_not_optimal=reshard,
            height_sharding=height_shard,
            layer_module='layer4_module1',
        )

        logger.debug('==== Running layer 4 module 2')
        x, x_height, x_width = self.layer4_module2(
            x, device, self.batch_size, x_height, x_width, layer_module='layer4_module2',
        )

        logger.debug('==== Running layer 4 module 3')
        x, x_height, x_width = self.layer4_module3(
            x, device, self.batch_size, x_height, x_width, layer_module='layer4_module3',
        )

        # ---------------- classifier tail — C:883-937 ----------------
        # Reproduced exactly.  The superseded fork rewrote this block: it added
        # six untilize/reshape/tilize ops, dropped both width-shard transitions,
        # and swapped avg_pool2d for global_avg_pool2d — a different shim op
        # descriptor with a different cost formula
        # (ttsim/front/ttnn/op.py:2465 vs :2454).
        grid_size = (8, 8)
        width_mem_config = _sharded_mem_config_(
            [nearest_32(_shard_dims(x)[0]),
             _shard_dims(x)[1] // (grid_size[0] * grid_size[1])],
            ttnn.CoreGrid(x=grid_size[0], y=grid_size[1]),
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.ShardOrientation.ROW_MAJOR,
            tile_layout=True,
            use_height_and_width_as_shard_shape=True,
        )
        x = ttnn.to_memory_config(x, width_mem_config)

        x = ttnn.avg_pool2d(
            input_tensor=x,
            batch_size=self.batch_size,
            input_h=x_height,
            input_w=x_width,
            channels=x.shape[3],
            kernel_size=[x_height, x_width],
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            output_layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat8_b,
            compute_kernel_config=ttnn.init_device_compute_kernel_config(
                self.device.arch(), math_fidelity=ttnn.MathFidelity.LoFi
            ),
        )

        grid_size = (8, 4)
        # Dims for the shard spec, read BEFORE the Polaris relabel below so both
        # paths compute the same config (on the Polaris path _sharded_mem_config_
        # ignores `shape` entirely — divergence note 9 — but reading x after the
        # relabel would still change what is passed).
        _pool_out_h, _pool_out_w = x.shape[2], x.shape[3]

        if IS_POLARIS:
            # These two relabels have to happen BEFORE the to_memory_config
            # below, because that call is the capture's row-112 Reshard and its
            # key is built from this tensor.  Silicon records that input as
            # (1, 1, 32, 2048) TILE — a flattened-NHWC, tile-major activation —
            # where the Polaris NCHW path has (16, 2048, 1, 1) ROW_MAJOR.
            # Neither relabel emits an op; both describe what avg_pool2d already
            # produced on hardware.
            #
            # Layout: canonical declares it via `output_layout=ttnn.TILE_LAYOUT`
            # (C:903), which the shim's avg_pool2d does not honour.  Setting the
            # metadata records what HW produces and is also what the downstream
            # untilize_with_unpadding requires (ttsim/front/ttnn/ttnn_shim.py:1286-1287).
            #
            # hw_shape: the flattened-NHWC view, same channel the fold head uses
            # (see fold_transpose_sharded_nchw).  The logical NCHW shape is left alone here and
            # relabelled separately after the reshard, because conv_sinf and the
            # fc matmul want different things from it.
            x.layout = ttnn.TILE_LAYOUT
            x.hw_shape = [1, 1, self.batch_size, int(x.shape[1])]

        width_mem_config = _sharded_mem_config_(
            [nearest_32(_pool_out_h), _pool_out_w // (grid_size[0] * grid_size[1])],
            ttnn.CoreGrid(x=grid_size[0], y=grid_size[1]),
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.ShardOrientation.ROW_MAJOR,
            tile_layout=True,
            use_height_and_width_as_shard_shape=True,
        )
        x = ttnn.to_memory_config(x, width_mem_config)

        if IS_POLARIS:
            # Canonical declares the pooled output tile-major via
            # `output_layout=ttnn.TILE_LAYOUT` (C:903).  The shim's avg_pool2d
            # does not honour that kwarg, and the downstream
            # untilize_with_unpadding rejects non-tile input
            # (ttsim/front/ttnn/ttnn_shim.py:1286-1287).  Setting the layout
            # metadata records what HW actually produces; it emits no op, in the
            # same spirit as the shim's own _apply_conv_output_layout.
            # Re-applied here because to_memory_config produces a fresh tensor
            # that inherits the shim's default layout, not the one set above.
            x.layout = ttnn.TILE_LAYOUT
            x.hw_shape = [1, 1, self.batch_size, int(x.shape[1])]

        if IS_POLARIS:
            # POLARIS-ONLY OP — a known, accepted exception with no counterpart
            # in the capture, and a consequence of the NCHW
            # convention forced by conv_sinf (see the fold/reshape note above).
            # On HW the tensor reaching fc is flattened NHWC (1, 1, N, 2048) and
            # canonical matmuls it directly (C:921).  On the NCHW path avg_pool
            # yields [N, 2048, 1, 1], whose last dim is 1, so the contraction
            # dim has to be restored before the matmul.
            #
            # FREE, like the post-fold site.  An earlier revision kept this as a
            # real Reshape on the grounds that relabelling it also deleted a
            # Tilize/Untilize pair "which hardware runs".  That was wrong: the
            # capture's two Tilize rows are at 12 and 19, converting the
            # maxpool's ROW_MAJOR output to TILE ahead of the first two matmuls,
            # and its only UntilizeWithUnpadding is the final row 114.  There is
            # nothing at this position.
            #
            # The pair was an artefact of the Reshape itself: it produces a fresh
            # tensor whose layout sends the shim into an untilize -> retilize
            # round-trip (ttsim/front/ttnn/ttnn_shim.py:1688-1710), emitting `Untilize` +
            # `TilizeWithValPadding` — two ops hardware never runs, mis-typed
            # against the four distinct tt-metal ops besides.  Relabelling
            # removes all three.
            x = _reinterpret_shape(x, (1, 1, self.batch_size, x.shape[1]))

        x = self.fc(x)
        desired_shape = list(x.shape)
        desired_shape[-1] = 1000
        if IS_POLARIS:
            # Emit ONE UntilizeWithUnpadding, matching the capture's last row
            # (114): (1, 1, 32, 1024) -> (1, 1, 16, 1000), TILE/BFLOAT8_B,
            # L1_WIDTH_SHARDED.
            #
            # The generic shim entry point does not produce that.  It routes
            # into a layout-normalisation helper (ttsim/front/ttnn/ttnn_shim.py:1688-1710) that
            # expands the call into an untilize -> retilize round-trip, emitting
            # `Untilize` + `TilizeWithValPadding` — two ops, neither of which
            # hardware runs here, and both mis-typed against the four distinct
            # tt-metal ops (Tilize / TilizeWithValPadding / Untilize /
            # UntilizeWithUnpadding).  Silicon's two real `Tilize` rows are
            # elsewhere entirely: rows 12 and 19, converting the maxpool's
            # ROW_MAJOR output to TILE ahead of the first two matmuls.
            #
            # `untilize_with_unpadding_op` is the shim's own primitive for this
            # and already emits the right optype; it is simply not what the
            # generic path reaches.  Calling it directly keeps the fix
            # workload-local rather than changing a helper other workloads use
            # for layout coercion.
            from ttsim.front.ttnn.ttnn_shim import untilize_with_unpadding_op as _shim_untilize_with_unpadding_op
            x = _shim_untilize_with_unpadding_op(
                x,
                output_shape=list(desired_shape),
                element_size=x.element_size(),
                memory_config=self.final_output_mem_config,
            )
        else:
            x = ttnn.untilize_with_unpadding(
                x,
                output_tensor_end=(
                    desired_shape[0] - 1, desired_shape[1] - 1,
                    desired_shape[2] - 1, desired_shape[3] - 1,
                ),
                memory_config=self.final_output_mem_config,
            )
        # Final un-flatten. The capture's last row is the UntilizeWithUnpadding
        # above (114); nothing follows it, so on the Polaris path this is a
        # relabel. Real ttnn keeps the call and decides for itself whether a
        # view suffices.
        _final_shape = (
            self.batch_size,
            x.shape[1],
            x.shape[2] // self.batch_size,
            x.shape[3],
        )
        x = _reinterpret_shape(x, _final_shape) if IS_POLARIS else ttnn.reshape(x, _final_shape)

        return x
