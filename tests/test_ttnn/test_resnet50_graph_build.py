# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Build the ResNet-50 graph at each published operating point and assert its shape.

``P:<line>`` cites the ported model in this repository,
``workloads/ttnn/resnet50/common/ttnn_functional_resnet50.py``.  Its own
header cites the tt-metal canonical source as ``C:<line>``; the two files
differ in length, so the prefixes are not interchangeable.  ``P:`` is used
here because that file is in the repo, while the canonical checkout is not.

Everything below ``resnet50.__init__`` — ``_make_layer``, all sixteen
``resnet50Bottleneck`` instances, ``run_downsample_if_req``, the hardware fold,
the arch-specific transitions before layer3 and layer4, the classifier — runs
only when a graph is actually built.  No other test in the suite builds one: the
device-perf tests do, but they live under ``workloads/`` and are not collected by
the checkin run.  So this file is the difference between the model being
exercised and merely being imported.

It goes through ``create_test_infra`` rather than constructing ``resnet50``
directly, so the six device gates the infra opts into are the ones under test —
those gates are the port, and a model built without them is a different model.

Three configurations, because the divergences between them are not cosmetic.
``is_wormhole_b0`` / ``is_blackhole`` / ``is_blackhole_p100`` select different
shard schemes at the layer3 and layer4 transitions (P:1413-1428, P:1464-1487)
— the Blackhole half of that code was unreachable until the infra started
declaring the arch before anything read it — and batch 32 opens a further set of
Blackhole-only branches that batch 16 never reaches.  See BUILD_MATRIX.
"""

import pytest

import ttsim.front.ttnn as ttnn
from ttsim.front.ttnn.device import close_device, open_device, set_default_device
from workloads.ttnn.resnet50.common.ttnn_functional_resnet50 import IS_POLARIS, resnet50Bottleneck

#: (arch, batch) the build is exercised at.
#:
#: Blackhole publishes both 16 and 32 as device-perf operating points
#: (``tt-metal/models/demos/vision/classification/resnet50/blackhole/tests/test_perf_device_resnet50.py:16-17``);
#: Wormhole publishes
#: only 16.  Batch is not free to choose — the infra asserts ``> 2`` and ``!= 8``,
#: WH additionally rejects 20, and the shard configs are derived from it.
#:
#: bs32 is not redundant with bs16.  It is the only way into the Blackhole
#: batch-32 branches: the ``act_block_h_override`` split at P:997 (32*32 on
#: p100a vs 49*32 elsewhere), the layer1/layer2 override at P:786-788, the
#: ``enable_act_double_buffer`` inversion at P:654, and the whole fold core-grid
#: arm at P:1079-1088.  None of those execute at 16.
BUILD_MATRIX = [('blackhole', 16), ('blackhole', 32), ('wormhole_b0', 16)]

#: torchvision ResNet-50: 3 + 4 + 6 + 3 bottlenecks, four of them downsampling.
EXPECTED_BLOCKS_PER_STAGE = [3, 4, 6, 3]

#: conv1's ``act_block_h_override`` per (arch, batch), from C:432-438:
#:   0 by default; 1568 on Wormhole; 32*32 on a batch-32 Blackhole p100a
#:   (49*32 on a non-p100 Blackhole, which this matrix does not cover).
#: Three distinct values, so this is what proves each row of BUILD_MATRIX
#: actually took its own branch rather than quietly building the same model.
EXPECTED_CONV1_ACT_BLOCK_H = {
    ('blackhole', 16): 0,
    ('blackhole', 32): 32 * 32,
    ('wormhole_b0', 16): 1568,
}

pytestmark = pytest.mark.skipif(
    not IS_POLARIS, reason='builds a Polaris graph; the HW path needs silicon')


def _infra(arch):
    if arch == 'blackhole':
        from workloads.ttnn.resnet50.bh.resnet50_test_infra_polaris_bh import create_test_infra
    else:
        from workloads.ttnn.resnet50.wh.resnet50_test_infra_polaris_wh import create_test_infra
    return create_test_infra


@pytest.fixture(
    scope='module', params=BUILD_MATRIX, ids=[f'{a}-b{b}' for a, b in BUILD_MATRIX])
def built(request):
    """A device carrying one fully built and executed ResNet-50 graph.

    Module-scoped on purpose: building the graph is the expensive part of this
    file, and every test below reads the same one.  Three builds total, not
    eighteen.

    ``close_device`` also clears the default-device pointer (``device.py:194-201``);
    without that the next parametrization would inherit the previous
    ``get_arch_name()`` and silently build the wrong half of the model.
    """
    arch, batch = request.param
    device = open_device()
    try:
        infra = _infra(arch)(device, batch)
        # The input path run_resnet50_e2e uses on both arches.
        tt_inputs_host, _dram_cfg, input_mem_config = infra.setup_dram_sharded_input(device)
        infra.input_tensor = ttnn.to_memory_config(tt_inputs_host, input_mem_config)
        infra.run()
        yield arch, batch, device, infra
    finally:
        close_device(device)
        set_default_device(None)


@pytest.mark.unit
def test_the_infra_declares_the_arch_it_was_asked_for(built) -> None:
    """The precondition everything else here depends on.  The model's arch
    branches read ``ttnn.get_arch_name()``, which reads the default device — so
    if this is wrong, the build below silently exercises the other arch and
    still passes every shape assertion."""
    arch, _batch, device, _ = built
    assert device.get_arch_name() == arch


@pytest.mark.unit
def test_each_matrix_row_builds_a_genuinely_different_model(built) -> None:
    """THE CONTROL FOR THE MATRIX.  Without it, adding a second batch would just
    re-run the same six assertions against a model that happened to be identical,
    and the batch-32 branches could be dead while every test still passed.

    conv1's ``act_block_h_override`` is the cheapest observable that differs
    three ways (C:432-438), so it distinguishes all three rows at once.
    """
    arch, batch, _device, infra = built
    got = infra.ttnn_resnet50_model.conv1_config.act_block_h_override
    assert got == EXPECTED_CONV1_ACT_BLOCK_H[(arch, batch)], (arch, batch, got)


@pytest.mark.unit
def test_the_model_has_the_torchvision_block_structure(built) -> None:
    """``_make_layer`` called four times, with the first block of each stage
    downsampling.  Layer 1 is the interesting row: it downsamples on the channel
    change (64 -> 256) at stride 1, which is why a stride derived from
    ``downsample`` rather than from the argument would be wrong."""
    _arch, _batch, _device, infra = built
    model = infra.ttnn_resnet50_model
    stages = [model.layer1, model.layer2, model.layer3, model.layer4]
    assert [len(s) for s in stages] == EXPECTED_BLOCKS_PER_STAGE
    assert all(isinstance(b, resnet50Bottleneck) for s in stages for b in s)
    assert [s[0].stride for s in stages] == [1, 2, 2, 2]
    assert [b.stride for b in stages[0][1:]] == [1, 1]


@pytest.mark.unit
def test_building_the_graph_produces_the_classifier_output_shape(built) -> None:
    """The end-to-end assertion: run the model and check the logits shape.

    ``[batch, 1, 1, 1000]`` and not ``[batch, 1000]`` — the classifier keeps the
    rank-4 form the fc matmul produces, which is what
    ``run_resnet50_e2e`` asserts on both arches.
    """
    _arch, batch, _device, infra = built
    assert list(infra.output_tensor.shape) == [batch, 1, 1, 1000]


@pytest.mark.unit
def test_every_bottleneck_emits_exactly_one_residual_add(built) -> None:
    """3 + 4 + 6 + 3 = 16 bottlenecks, one residual ``Add`` each.

    Exact rather than a floor, and arch-independent, because it is topology: it
    does not depend on which convs lower to matmul or how many Halos the shard
    scheme needs.  A bottleneck that skipped its residual — the failure a shape
    assertion cannot see, since the output shape is unchanged — moves this number.
    """
    _arch, _batch, device, _ = built
    optypes = [o.optype for o in device.ops.values()]
    assert optypes.count('Add') == sum(EXPECTED_BLOCKS_PER_STAGE) == 16, optypes.count('Add')


@pytest.mark.unit
def test_the_convolution_count_is_the_torchvision_topology(built) -> None:
    """53 convolutions (49 in the bottlenecks + 4 downsample projections) plus the
    fc classifier = 54 convolution-like ops.

    The Conv/MatMul *split* is arch-dependent — a 1x1 conv lowers to a matmul
    under conditions that differ between Wormhole and Blackhole, and a real
    bs=16 p100a run records 20 Conv against 34 MatMul — but the total is
    topology and should not move.  If this ever fails on one arch and not the
    other, the split has changed, which is a finding rather than a bad number:
    record it per arch and say why.
    """
    _arch, _batch, device, _ = built
    optypes = [o.optype for o in device.ops.values()]
    conv_like = optypes.count('Conv') + optypes.count('MatMul')
    assert conv_like == 54, f'{conv_like}: Conv={optypes.count("Conv")} MatMul={optypes.count("MatMul")}'


@pytest.mark.unit
def test_the_graph_is_not_a_degenerate_stub(built) -> None:
    """A floor on the whole graph, to catch a build that returns a
    correctly-shaped tensor having emitted almost nothing.

    Deliberately loose: the exact multiset belongs to the op-by-op comparison
    against the profiler capture, which owns that claim and runs against real
    hardware data.  A real run is 115 ops on BH p100a and 108 on WH n150.
    """
    _arch, _batch, device, _ = built
    optypes = [o.optype for o in device.ops.values()]
    assert len(optypes) > 100, f'only {len(optypes)} ops — did the build degrade?'
    for expected in ('Conv', 'MatMul', 'Add', 'Halo'):
        assert expected in optypes, f'no {expected} in {sorted(set(optypes))}'
