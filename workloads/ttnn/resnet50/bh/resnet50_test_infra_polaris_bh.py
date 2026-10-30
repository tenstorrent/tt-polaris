#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Polaris-side ResNet50TestInfra — Blackhole (p100a).

Mirrors the interface of the canonical
``tt-metal/models/demos/vision/classification/resnet50/ttnn_resnet/tests/common/resnet50_test_infra.py`` (cited as
``TI:<line>``)
so ``test_resnet50_device_perf_bh.py`` runs dual-mode against the same calls.

WH and BH share one model module; only this infra differs, and only in the
compute grid it configures.  See the WH variant for the n150 values.
"""

import ttsim.front.ttnn as ttnn
from ttsim.front.ttnn.buffer import BufferType, TensorMemoryLayout
from ttsim.front.ttnn.device import set_default_device
from ttsim.front.ttnn.memory import MemoryConfig
from ttsim.front.ttnn.tensor import ttnn_random
from workloads.ttnn.resnet50.common.model_preprocessing import create_resnet50_model_parameters
from workloads.ttnn.resnet50.common.shim_gate_opt_ins import enable_resnet50_shim_gates
from workloads.ttnn.resnet50.common.ttnn_functional_resnet50 import resnet50

# Logical tensix grid for this SKU, from config/tt_bh.yaml's `compute_grid_size`
# for p100a (12 x 10 = 120 cores, one column reserved).  The shim's Device
# defaults its grid to 0x0 (ttsim/front/ttnn/device.py:70-72) and the model
# asserts rather than guessing, because the value feeds the fold shard shape
# and, on Blackhole, the core-count search at tt-metal
# tt-metal/models/demos/vision/classification/resnet50/ttnn_resnet/tt/ttnn_functional_resnet50.py:528-535.
BH_P100A_COMPUTE_GRID = (12, 10)
# The arch-spec device instance every capture-derived table in this port was
# measured against (config/tt_bh.yaml:63).  Declared on the Device so SKU-keyed
# shim tables resolve to p100a's numbers and NOT to p150a/p150b, which sit under
# the same 'blackhole' arch (tt_bh.yaml:107,119) and have never been captured.
BH_CAPTURE_SKU = 'p100a'

# Fold padding/stride — NOT the conv1 kernel.  TI:194-195 passes these as
# `kernel_size` / `stride` into resnet50(...) (TI:241-242); they drive
# pad_and_fold_conv_filters_for_unity_stride, which is why conv1's weight is
# [64, 16, 4, 4].  Misreading them as the conv kernel is what produced the
# previous port's wrong first-conv geometry.
RESNET50_FIRST_CONV_KERNEL_SIZE = 3
RESNET50_FIRST_CONV_STRIDE = 2


class ResNet50TestInfra:
    """Polaris mirror of the upstream ResNet50TestInfra.

    Builds synthetic parameters plus an NHWC input tensor and runs one forward
    pass.  No golden/torch reference and no PCC validation: polaris models
    shapes and costs, not values.
    """

    def __init__(
        self,
        device,
        batch_size,
        act_dtype=None,
        weight_dtype=None,
        math_fidelity=None,
        use_pretrained_weight=False,
        dealloc_input=True,
        final_output_mem_config=None,
        model_location_generator=None,
    ):
        # Declare the target arch BEFORE anything reads it: ttnn.get_arch_name()
        # reports the default device's architecture, and the model's
        # is_wormhole_b0() / is_blackhole() branches key off it.  Without this
        # the shim reported wormhole_b0 for every device and the whole Blackhole
        # half of this model was unreachable.
        device.set_arch('blackhole')
        device.set_capture_sku(BH_CAPTURE_SKU)
        set_default_device(device)
        device.compute_with_storage_grid_size(*BH_P100A_COMPUTE_GRID)

        # The eight gates this workload opts into live in common/, because the
        # block was byte-identical across both arches and a ninth added to only
        # one would still produce a plausible projection on the other.
        enable_resnet50_shim_gates(device)

        # Canonical device-perf operating point
        # (tt-metal/models/demos/vision/classification/resnet50/blackhole/tests/test_perf_device_resnet50.py:16-17).
        if act_dtype is None:
            act_dtype = ttnn.bfloat8_b
        if weight_dtype is None:
            weight_dtype = ttnn.bfloat8_b
        if math_fidelity is None:
            math_fidelity = ttnn.MathFidelity.LoFi
        if final_output_mem_config is None:
            final_output_mem_config = ttnn.L1_MEMORY_CONFIG

        self.device = device
        self.batch_size = batch_size

        model_config = {
            'MATH_FIDELITY': math_fidelity,
            'WEIGHTS_DTYPE': weight_dtype,
            'ACTIVATIONS_DTYPE': act_dtype,
        }

        # TI:200-201 — upstream skips these outright, so modelling them would
        # project a configuration hardware never runs.
        assert batch_size > 2, 'batch sizes 1 and 2 are unsupported with sharded data'
        assert batch_size != 8, 'batch size 8 is skipped upstream (memory config issue)'
        # No bs==20 assert here: that skip (TI:202-203) is Wormhole-specific.

        input_shape = [batch_size, 3, 224, 224]
        parameters = create_resnet50_model_parameters(device, weights_dtype=weight_dtype)

        self.ttnn_resnet50_model = resnet50(
            device=device,
            parameters=parameters,
            batch_size=batch_size,
            model_config=model_config,
            input_shape=input_shape,
            kernel_size=RESNET50_FIRST_CONV_KERNEL_SIZE,
            stride=RESNET50_FIRST_CONV_STRIDE,
            dealloc_input=dealloc_input,
            final_output_mem_config=final_output_mem_config,
        )
        self.ops_parallel_config = {}

        # ttnn.fold consumes NHWC (ttsim/front/ttnn/op.py asserts rank-4
        # [N, H, W, C]); the model converts to NCHW after the fold.
        # ttnn_random is the in-repo idiom for synthetic activations
        # (minitorch_shim exposes only manual_seed, no randn).
        # NCHW, because that is what silicon feeds the fold: the p100a capture's
        # first op is Pad with input (16, 3, 224, 224).  This used to be NHWC
        # (batch, 224, 224, 3) to satisfy the shim's ttnn.fold, which asserts
        # rank-4 [N, H, W, C] — but the model no longer calls that fold, it
        # emits the seven ops hardware runs (see _emit_hw_fold), and those key
        # off the NCHW shape.
        self.torch_input = ttnn_random(
            (batch_size, 3, 224, 224), -1, 1, dtype=ttnn.bfloat16,
        )
        self.input_tensor = None

    def setup_l1_sharded_input(self, device, torch_input=None, mesh_mapper=None, mesh_composer=None):
        if torch_input is None:
            torch_input = self.torch_input
        tt_inputs_host = ttnn.from_torch(
            torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        input_mem_config = MemoryConfig(TensorMemoryLayout.HEIGHT_SHARDED, BufferType.L1)
        return tt_inputs_host, input_mem_config

    def setup_dram_sharded_input(self, device, torch_input=None, mesh_mapper=None, mesh_composer=None):
        tt_inputs_host, input_mem_config = self.setup_l1_sharded_input(
            device, torch_input, mesh_mapper=mesh_mapper, mesh_composer=mesh_composer,
        )
        sharded_mem_config_DRAM = MemoryConfig.DRAM
        return tt_inputs_host, sharded_mem_config_DRAM, input_mem_config

    def run(self, tt_input_tensor=None):
        # TI:316-322
        input_tensor = tt_input_tensor if tt_input_tensor is not None else self.input_tensor
        self.output_tensor = self.ttnn_resnet50_model(
            input_tensor,
            self.device,
            self.ops_parallel_config,
        )
        return self.output_tensor


def create_test_infra(
    device,
    batch_size,
    act_dtype=None,
    weight_dtype=None,
    math_fidelity=None,
    use_pretrained_weight=False,
    dealloc_input=True,
    final_output_mem_config=None,
    model_location_generator=None,
):
    """Signature mirrors TI:342-363 so the runner's call site is identical."""
    return ResNet50TestInfra(
        device,
        batch_size,
        act_dtype,
        weight_dtype,
        math_fidelity,
        use_pretrained_weight,
        dealloc_input,
        final_output_mem_config,
        model_location_generator,
    )
