# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Opt-in shim gates: process-wide defaults plus their per-device overrides.

These gate ttnn-shim behaviours, so their natural home is
``ttsim/front/ttnn/op.py`` — and that is where they lived.  They are here
instead for one reason: ``ttsim.ops`` has to read one of them.

``_propagate_bcast_hw_shape`` is consumed by ``bidir_bcast`` in
``ttsim/ops/desc/helpers.py``, on the shared ONNX path.  With the state in the
front end, that read was a back-edge ``ttsim.ops -> ttsim.front.ttnn``, and
deferring the import into the function body moved it rather than removing it: a
single ``bidir_bcast`` on plain ``SimTensor`` operands pulled the whole ttnn
front end into a pure-ONNX process, and closed a real cycle —

    ttsim.ops.desc.helpers -> ttsim.front.ttnn.op -> ttsim.front.ttnn.tensor
      -> ttsim.front.ttnn.device -> ttsim.graph -> ttsim.graph.wl_graph
      -> ttsim.ops -> ttsim.ops.desc -> ttsim.ops.desc.tensor
      -> ttsim.ops.desc.helpers

which survived only because the deferred import ran after
``ttsim.ops.desc.helpers`` was already initialised.  ``ttsim.utils`` is below
both packages (``ttsim.ops.op`` already imports ``ttsim.utils.common``), so
putting the state here lets each side read it without depending on the other.

Import from HERE, not through the front end.  ``ttsim/front/ttnn/op.py`` pulls
in only the ``<gate>_enabled`` accessors it calls itself; re-exporting the
setters too would leave its import block describing its callers' needs rather
than its own.  Workload infras and gate tests do
``from ttsim.utils.shim_gates import set_<gate>, <gate>_enabled``, which also
keeps a matched pair arriving from one module.

Two layers, resolved in this order by every ``<gate>_enabled``:

    attribute absent on the Device -> the process-wide default here, which is
                                      seeded from ``TTSIM_*`` at import
    attribute present              -> that device's explicit override, set only
                                      by ``set_<gate>(x, device)``

Device.__init__ deliberately does not materialise these onto the instance; see
the note there.  Import this module for the state, never copy a ``_GATE`` value
into another module — ``from ... import _PROPAGATE_MEMORY_CONFIG`` would snapshot
the bool and stop tracking ``set_memory_config_propagation``.
"""

import os
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Conv/MatMul output memory-config propagation  (OPT-IN, default OFF)
# ---------------------------------------------------------------------------
#
# On hardware a conv output carries the memory config implied by its
# ``Conv2dConfig.shard_layout``, so model code can meaningfully ask
# ``out.memory_config()``.  The shim leaves conv/matmul outputs with
# ``_memory_config = None``, which makes that question unanswerable: ResNet-50's
# bottleneck reshard is
#
#     if ds_out.memory_config() != out.memory_config():
#         ds_out = ttnn.to_memory_config(ds_out, out.memory_config())
#
# (tt-metal/models/demos/vision/classification/resnet50/ttnn_resnet/tt/ttnn_functional_resnet50.py:325-326)
#
# and with both sides None the condition is always False, so the Reshard the
# refrun shows at every downsampling bottleneck is never emitted.
#
# This is OFF by default and changes nothing for existing workloads.  Turn it on
# per workload — ``set_memory_config_propagation(True)`` from the test infra, or
# ``TTSIM_PROPAGATE_MEMORY_CONFIG=1`` in the environment — after checking against
# that workload's own refrun, since enabling it can add data-movement ops.

_PROPAGATE_MEMORY_CONFIG = os.getenv('TTSIM_PROPAGATE_MEMORY_CONFIG', '0') == '1'


def set_memory_config_propagation(enabled: bool, device: Any = None) -> bool:
    """Enable/disable conv-output memory-config propagation.

    Prefer passing the device: the flag then lives on that Device and cannot leak
    into the next workload built in the same process.  Omitting it sets the
    process-wide fallback, which is what the env var controls.
    """
    global _PROPAGATE_MEMORY_CONFIG
    if device is not None:
        previous = getattr(device, 'propagate_memory_config', _PROPAGATE_MEMORY_CONFIG)
        device.propagate_memory_config = bool(enabled)
        return previous
    previous = _PROPAGATE_MEMORY_CONFIG
    _PROPAGATE_MEMORY_CONFIG = bool(enabled)
    return previous


def memory_config_propagation_enabled(device: Optional[object] = None) -> bool:
    if device is not None:
        return bool(getattr(device, 'propagate_memory_config', _PROPAGATE_MEMORY_CONFIG))
    return _PROPAGATE_MEMORY_CONFIG


_TILIZE_BEFORE_MATMUL = os.getenv('TTSIM_TILIZE_BEFORE_MATMUL', '0') == '1'


def set_tilize_before_matmul(enabled: bool, device: Any = None) -> bool:
    """Enable/disable the ROW_MAJOR -> TILE conversion before a lowered 1x1 matmul.

    Hardware matmuls consume TILE, so when a 1x1 conv is lowered to a matmul and
    its activation is still ROW_MAJOR, silicon dispatches a Tilize first.  The
    ResNet-50 p100a capture shows exactly that at rows 12 and 19, converting the
    maxpool's ROW_MAJOR output ahead of the first two matmuls; polaris emitted
    neither, which is the whole of its `tilize 0 vs 2` gap.
    ``_matmul_1x1_with_hw_fields`` already documented the behaviour ("the
    activation is tilized before the matmul") without modelling it.

    OFF by default and device-scoped, for the same reason as
    ``set_memory_config_propagation``: it adds ops, and polaris builds every
    workload graph in one process, so a process-wide flag would silently change
    whichever workload was built next.
    """
    global _TILIZE_BEFORE_MATMUL
    if device is not None:
        previous = getattr(device, 'tilize_before_matmul', _TILIZE_BEFORE_MATMUL)
        device.tilize_before_matmul = bool(enabled)
        return previous
    previous = _TILIZE_BEFORE_MATMUL
    _TILIZE_BEFORE_MATMUL = bool(enabled)
    return previous


def tilize_before_matmul_enabled(device: Optional[object] = None) -> bool:
    if device is not None:
        return bool(getattr(device, 'tilize_before_matmul', _TILIZE_BEFORE_MATMUL))
    return _TILIZE_BEFORE_MATMUL


_CONV_MATH_FIDELITY_IN_KEY = os.getenv('TTSIM_CONV_MATH_FIDELITY_IN_KEY', '0') == '1'


def set_conv_math_fidelity_in_key(enabled: bool, device: Any = None) -> bool:
    """Carry `compute_config`'s math fidelity into conv/1x1-matmul op attrs.

    `conv2d_pp` rebuilds its kwargs dict and drops `compute_config`, so a 1x1
    conv lowered to a MatMul keys math_fidelity='N/A' where the capture recorded
    the real value.

    OFF by default and device-scoped because the LUTs disagree: bh_p100a_lut_v6
    keys matmul under BOTH 'LoFi' (x14, resnet50) and 'N/A' (x9, ViT/VGG), and
    whb0_n150_lut_v5 is entirely 'N/A'.  There is no lookup-side bridge — a
    `math_fidelity -> N/A` fallback existed and was removed as dead code, since
    only workloads that opt in here emit a concrete fidelity and their LUT
    already carries matching entries.  So enabling this against a LUT recorded
    entirely under 'N/A' will make every matmul MISS; regenerate that LUT with
    real fidelities rather than reinstating the fallback.
    """
    global _CONV_MATH_FIDELITY_IN_KEY
    if device is not None:
        previous = getattr(device, 'conv_math_fidelity_in_key', _CONV_MATH_FIDELITY_IN_KEY)
        device.conv_math_fidelity_in_key = bool(enabled)
        return previous
    previous = _CONV_MATH_FIDELITY_IN_KEY
    _CONV_MATH_FIDELITY_IN_KEY = bool(enabled)
    return previous


def conv_math_fidelity_in_key_enabled(device: Optional[object] = None) -> bool:
    if device is not None:
        return bool(getattr(device, 'conv_math_fidelity_in_key', _CONV_MATH_FIDELITY_IN_KEY))
    return _CONV_MATH_FIDELITY_IN_KEY


_PROPAGATE_BCAST_HW_SHAPE = os.getenv('TTSIM_PROPAGATE_BCAST_HW_SHAPE', '0') == '1'


def set_bcast_hw_shape_propagation(enabled: bool, device: Any = None) -> bool:
    """Enable/disable NHWC-flat ``hw_shape`` propagation across elementwise binaries.

    ``bidir_bcast`` (``ttsim/ops/desc/helpers.py``) is the one data-movement sinf
    that does not carry ``hw_shape`` through; Move, Reshard, ITS and STI all do.
    The consequence is that a residual ``Add`` drops the flattened form and every
    downstream op keys on logical NCHW instead of the ``(1, 1, N*H*W, C)`` the
    profiler records.

    OFF by default and device-scoped: it changes the LUT key of every elementwise
    binary in every workload, and polaris builds all workload graphs in one
    process.
    """
    global _PROPAGATE_BCAST_HW_SHAPE
    if device is not None:
        previous = getattr(device, 'propagate_bcast_hw_shape', _PROPAGATE_BCAST_HW_SHAPE)
        device.propagate_bcast_hw_shape = bool(enabled)
        return previous
    previous = _PROPAGATE_BCAST_HW_SHAPE
    _PROPAGATE_BCAST_HW_SHAPE = bool(enabled)
    return previous


def bcast_hw_shape_propagation_enabled(device: Optional[object] = None) -> bool:
    if device is not None:
        return bool(getattr(device, 'propagate_bcast_hw_shape', _PROPAGATE_BCAST_HW_SHAPE))
    return _PROPAGATE_BCAST_HW_SHAPE


_HALO_OUTPUT_DTYPE_PROMOTION = os.getenv('TTSIM_HALO_OUTPUT_DTYPE_PROMOTION', '0') == '1'


def set_halo_output_dtype_promotion(enabled: bool, device: Any = None) -> bool:
    """Enable/disable HaloDeviceOperation's output-dtype promotion.

    ``tt-metal/ttnn/cpp/ttnn/operations/sliding_window/halo/device/halo_device_operation.cpp:66-70`` picks the halo
    output dtype from a
    three-way switch on the input dtype::

        switch (input_tensor.dtype()) {
            case FLOAT32: output_dtype = FLOAT32; break;
            case UINT16:  output_dtype = UINT16;  break;
            default:      output_dtype = BFLOAT16; break;
        }

    So a BFLOAT8_B activation entering the halo leaves it as BFLOAT16 — the
    p100a ResNet-50 capture shows exactly that on 21 of 22 halos, and every one
    of its 20 conv2d rows reports ``INPUT_0_DATATYPE=BFLOAT16`` as a result.
    The shim instead inherits the input dtype (``_propagate_ttnn_dtype`` keeps
    the more compact of the inputs), so conv keys carry BFLOAT8_B and miss.

    OFF by default and device-scoped: it changes the dtype field of the LUT key
    for every halo and every op downstream of one, in every workload, and
    polaris builds all workload graphs in one process.
    """
    global _HALO_OUTPUT_DTYPE_PROMOTION
    if device is not None:
        previous = getattr(device, 'halo_output_dtype_promotion', _HALO_OUTPUT_DTYPE_PROMOTION)
        device.halo_output_dtype_promotion = bool(enabled)
        return previous
    previous = _HALO_OUTPUT_DTYPE_PROMOTION
    _HALO_OUTPUT_DTYPE_PROMOTION = bool(enabled)
    return previous


def halo_output_dtype_promotion_enabled(device: Optional[object] = None) -> bool:
    if device is not None:
        return bool(getattr(device, 'halo_output_dtype_promotion', _HALO_OUTPUT_DTYPE_PROMOTION))
    return _HALO_OUTPUT_DTYPE_PROMOTION


_CONV_INPUT_RESHARD = os.getenv('TTSIM_CONV_INPUT_RESHARD', '0') == '1'


def set_conv_input_reshard(enabled: bool, device: Any = None) -> bool:
    """Enable/disable the conv INPUT Reshard (+Move) forced by ``reshard_if_not_optimal``.

    ``tt-metal/ttnn/cpp/ttnn/operations/conv/conv2d/conv2d_utils.cpp:750-751`` recomputes the parallel config when the
    flag is
    set, and ``shard_or_reshard_tensor_if_required`` then does::

        ttnn::to_memory_config(input_tensor, cfg, std::nullopt, resharded_input_tensor);   // :883
        if (conv_config.deallocate_activation && !input_tensor.memory_config().is_dram()) {
            input_tensor.deallocate(/*force*/ true);
            resharded_input_tensor = ttnn::move(resharded_input_tensor);                    // :886
        }

    which is why the capture shows them as a Reshard/Move **pair**.  The shim
    previously read ``reshard_if_not_optimal`` only in
    ``_apply_conv_output_memory_config``, to pick the conv OUTPUT config, and
    never emitted the input-side ops: rows 11, 17(+18) and 59(+60) of the p100a
    capture were all absent, leaving reshard 5-vs-9 and move 1-vs-2.

    Like tt-metal, the Reshard is emitted only when the target config actually
    DIFFERS from the input's — ``to_memory_config`` is a no-op otherwise.  That
    is what keeps it to one per flagged module rather than one per conv: in
    ResNet-50 only conv1 (fed by the previous stage) and the downsample (fed by
    the block input ``x``, since ``run_downsample_if_req`` runs after conv3) are
    out of position; conv2 and conv3 already sit in their optimal config.

    OFF by default and device-scoped: it ADDS ops to every conv whose config
    sets the flag, in every workload, and polaris builds all workload graphs in
    one process.
    """
    global _CONV_INPUT_RESHARD
    if device is not None:
        previous = getattr(device, 'conv_input_reshard', _CONV_INPUT_RESHARD)
        device.conv_input_reshard = bool(enabled)
        return previous
    previous = _CONV_INPUT_RESHARD
    _CONV_INPUT_RESHARD = bool(enabled)
    return previous


def conv_input_reshard_enabled(device: Optional[object] = None) -> bool:
    if device is not None:
        return bool(getattr(device, 'conv_input_reshard', _CONV_INPUT_RESHARD))
    return _CONV_INPUT_RESHARD


_TO_MEMORY_CONFIG_RESHARD = os.getenv('TTSIM_TO_MEMORY_CONFIG_RESHARD', '0') == '1'


def set_to_memory_config_reshard(enabled: bool, device: Any = None) -> bool:
    """Let ``to_memory_config`` take tt-metal's PRIMARY sharded->sharded path.

    ``tt-metal/ttnn/cpp/ttnn/operations/core/to_memory_config/to_memory_config_op.cpp:279-320`` calls
    ``ttnn::reshard`` first and reaches
    the ``sharded_to_interleaved -> interleaved_to_sharded`` pair only as a
    labelled *"Workaround"*.  The shim historically emitted the pair
    unconditionally and never a Reshard, so a capture with N
    ReshardDeviceOperation rows compared against a polaris graph with zero of
    them plus a surplus of STI/ITS pairs.

    OFF by default, because taking the primary path needs BOTH shard specs to
    evaluate ``use_reshard_workaround`` (see ``_needs_reshard_workaround``), and
    a workload that does not propagate memory configs does not have them.  When
    the predicate cannot be evaluated the shim cannot tell which path hardware
    took, and guessing wrong costs a real hit: with this on for vgg_unet, the
    HEIGHT->BLOCK transition at its s2 boundary emitted one Reshard where the
    p100a capture (rows 20-21) round-trips through DRAM as STI+ITS, turning two
    hits into one miss.

    Device-scoped: it changes the op sequence of every sharded->sharded
    transition in every workload, and polaris builds all workload graphs in one
    process.
    """
    global _TO_MEMORY_CONFIG_RESHARD
    if device is not None:
        previous = getattr(device, 'to_memory_config_reshard', _TO_MEMORY_CONFIG_RESHARD)
        device.to_memory_config_reshard = bool(enabled)
        return previous
    previous = _TO_MEMORY_CONFIG_RESHARD
    _TO_MEMORY_CONFIG_RESHARD = bool(enabled)
    return previous


def to_memory_config_reshard_enabled(device: Optional[object] = None) -> bool:
    if device is not None:
        return bool(getattr(device, 'to_memory_config_reshard', _TO_MEMORY_CONFIG_RESHARD))
    return _TO_MEMORY_CONFIG_RESHARD


_MOVE_ON_REALLOCATE_HALO_OUTPUT = os.getenv('TTSIM_MOVE_ON_REALLOCATE_HALO_OUTPUT', '0') == '1'


def set_move_on_reallocate_halo_output(enabled: bool, device: Any = None) -> bool:
    """Gate the post-Halo Move on ``reallocate_halo_output`` instead of ``deallocate_activation``.

    tt-metal gates that Move on ``reallocate_halo_output``
    (``tt-metal/ttnn/cpp/ttnn/operations/conv/conv2d/conv2d.cpp:297-299``)::

        input_tensor_post_tm = std::move(halo_output);
        if (conv_config.reallocate_halo_output) {
            input_tensor_post_tm = ttnn::move(input_tensor_post_tm);
        }

    ``deallocate_activation`` drives the deallocate two lines above
    (``:291-293``), which is a different thing and dispatches no op.  Keying off
    it made polaris emit a Move at every conv with ``deallocate_activation=True``
    -- 17 of them for ResNet-50 where the refrun shows 2.

    OFF by default because the two flags have OPPOSITE defaults:
    ``reallocate_halo_output`` defaults True
    (``tt-metal/ttnn/cpp/ttnn/operations/conv/conv2d/conv2d_nanobind.cpp:267``) and
    ``deallocate_activation`` defaults False.  So flipping which one is read also
    flips what an unset config means, and a workload that sets neither gains
    Moves it never had.  That is the vgg_unet d4 ConvTranspose at y=1320, where
    the p100a capture goes Halo(256->1320) straight into Conv2d with no Move.

    Device-scoped: it ADDS ops to conv positions in every workload, and polaris
    builds all workload graphs in one process.
    """
    global _MOVE_ON_REALLOCATE_HALO_OUTPUT
    if device is not None:
        previous = getattr(
            device, 'move_on_reallocate_halo_output', _MOVE_ON_REALLOCATE_HALO_OUTPUT)
        device.move_on_reallocate_halo_output = bool(enabled)
        return previous
    previous = _MOVE_ON_REALLOCATE_HALO_OUTPUT
    _MOVE_ON_REALLOCATE_HALO_OUTPUT = bool(enabled)
    return previous


def move_on_reallocate_halo_output_enabled(device: Optional[object] = None) -> bool:
    if device is not None:
        return bool(getattr(
            device, 'move_on_reallocate_halo_output', _MOVE_ON_REALLOCATE_HALO_OUTPUT))
    return _MOVE_ON_REALLOCATE_HALO_OUTPUT
