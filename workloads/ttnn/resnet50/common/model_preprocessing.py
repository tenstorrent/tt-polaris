#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Polaris-side ResNet-50 parameter tree (synthetic, device-free weights).

Mirrors the canonical
``tt-metal/models/demos/vision/classification/resnet50/ttnn_resnet/tt/custom_preprocessing.py``
but builds shape stubs via ``ttsim.front.ttnn`` instead of loading torchvision
weights.  Only shape, dtype and layout matter for perf modelling — values are
never read.

Two canonical files are cited below.  Both live in
``tt-metal/models/demos/vision/classification/resnet50/ttnn_resnet/tt/``:

    ``CP:<line>``  -> that directory's ``custom_preprocessing.py``     (68 lines)
    ``C:<line>``   -> that directory's ``ttnn_functional_resnet50.py`` (939 lines)

They are different files, so the prefixes are not interchangeable.

Consumed by ``resnet50_test_infra_polaris_{wh,bh}.py``.

Two shapes here are easy to get wrong, and the superseded fork got both wrong.
They are asserted at the bottom of this file rather than left to trust.

conv1 — the folded first convolution
------------------------------------
torchvision's conv1 is ``[64, 3, 7, 7]`` stride 2.  tt-metal folds it to unity
stride (CP:43-44)::

    conv1_weight, conv1_bias = fold_batch_norm2d_into_conv2d(model.conv1, model.bn1)
    conv1_weight = pad_and_fold_conv_filters_for_unity_stride(conv1_weight, 2, 2)

``pad_and_fold_conv_filters_for_unity_stride`` (models/common/utility_functions.py:1133-1176)
pads in-channels to ``C = _nearest_y(3, 4) = 4``, pads H/W to a multiple of the
stride (7 -> 8), then folds::

    out = [64, C * stride_h * stride_w, 8 // 2, 8 // 2]
        = [64, 4 * 2 * 2,               4,      4    ]
        = [64, 16, 4, 4]

So conv1 in-channels is **16**, not 3.  Canonical's own guard
(``assert conv1_weight_tensor.shape[2] == 4``) only checks the spatial half and
passes on a wrong channel count, which is how ``[64, 3, 4, 4]`` survived
unnoticed in the previous port.

fc — transposed by the linear preprocessor
------------------------------------------
torchvision's ``fc`` is ``Linear(2048, 1000)``, i.e. weight ``[1000, 2048]``.
``preprocess_linear_weight`` (ttnn/ttnn/model_preprocessing.py:25-28) applies
``weight.T.contiguous()``, so the tensor that reaches the model is
``[2048, 1000]``; canonical then raises it to rank 4 (C:40).  The shim has no
``Shape.to_rank``, so it is supplied at rank 4 here: ``[1, 1, 2048, 1000]``.
Supplying ``[1000, 2048]`` makes the classifier matmul contract on the wrong
dimension.

Bias rank — a shim-forced divergence
------------------------------------
Canonical reshapes every conv/downsample bias to ``(1, 1, 1, -1)``
(CP:32-34, CP:39-41, CP:47).  The shim's ``conv_sinf`` rejects that::

    ValueError: Bias shape Shape([1, 1, 1, 64]) must be (C_out,) matching weight C_out 64

(``ttsim/ops/desc/nn.py:511`` — the ONNX Conv descriptor takes a 1-D bias.)
Conv biases are therefore rank 1 ``[C_out]`` here.  The fc bias keeps rank 4,
since it goes through ``ttnn.linear`` rather than ``conv_sinf``.  Bias rank
carries no cost signal, so this affects nothing but the declared shape.
"""

import ttsim.front.ttnn as ttnn
from ttsim.utils.common import dict2obj

# torchvision ResNet-50: block counts and per-stage planes (C:352, C:373-400).
_LAYERS = [3, 4, 6, 3]
_PLANES = [64, 128, 256, 512]
_EXPANSION = 4

# Fold-derived first-conv geometry — see the module docstring.
_CONV1_OUT_CHANNELS = 64
_CONV1_IN_CHANNELS_FOLDED = 16
_CONV1_KERNEL = 4

_NUM_CLASSES = 1000
_FC_IN_FEATURES = 2048


def _weight(name, shape, device, dtype, layout):
    return ttnn.Tensor(
        name=name, shape=ttnn.Shape(shape), dtype=dtype, layout=layout, device=device,
    )


def _conv(name, out_ch, in_ch, k, device, dtype, layout):
    """One conv's weight/bias pair.

    Bias is rank 1 ``[C_out]``, not canonical's ``(1, 1, 1, C_out)`` — see the
    bias-rank note in the module docstring.
    """
    return {
        'weight': _weight(f'{name}.weight', [out_ch, in_ch, k, k], device, dtype, layout),
        'bias': _weight(f'{name}.bias', [out_ch], device, dtype, layout),
    }


def create_resnet50_model_parameters(
    device,
    *,
    weights_dtype=None,
    layout=None,
    prefix='resnet50',
):
    """Build the synthetic ResNet-50 parameter tree.

    ``weights_dtype`` defaults to ``bfloat8_b`` — the canonical device-perf
    weight dtype on both Wormhole and Blackhole
    (``tt-metal/models/demos/vision/classification/resnet50/
    {wormhole,blackhole}/tests/test_perf_device_resnet50.py:16``), not an
    arbitrary choice.
    """
    if weights_dtype is None:
        weights_dtype = ttnn.bfloat8_b
    if layout is None:
        layout = ttnn.Layout.ROW_MAJOR

    params = {
        'conv1': _conv(
            f'{prefix}.conv1',
            _CONV1_OUT_CHANNELS,
            _CONV1_IN_CHANNELS_FOLDED,
            _CONV1_KERNEL,
            device, weights_dtype, layout,
        ),
    }

    # Bottleneck stages.  Mirrors _make_layer (C:569-596): the first block of
    # each stage downsamples (stride != 1 or a channel change), later blocks
    # consume the stage's expanded width.
    in_channels = _CONV1_OUT_CHANNELS
    for stage_idx, (blocks, planes) in enumerate(zip(_LAYERS, _PLANES), start=1):
        out_channels = planes * _EXPANSION
        stage_name = f'layer{stage_idx}'
        stage = []

        for block_idx in range(blocks):
            block_in = in_channels if block_idx == 0 else out_channels
            bname = f'{prefix}.{stage_name}.{block_idx}'

            block = {
                'conv1': _conv(f'{bname}.conv1', planes, block_in, 1, device, weights_dtype, layout),
                'conv2': _conv(f'{bname}.conv2', planes, planes, 3, device, weights_dtype, layout),
                'conv3': _conv(f'{bname}.conv3', out_channels, planes, 1, device, weights_dtype, layout),
            }
            if block_idx == 0:
                # Only the stage's first block carries a downsample projection.
                block['downsample'] = _conv(
                    f'{bname}.downsample', out_channels, block_in, 1, device, weights_dtype, layout,
                )
            stage.append(block)

        params[stage_name] = stage
        in_channels = out_channels

    # Classifier — already transposed and at rank 4; see the module docstring.
    params['fc'] = {
        'weight': _weight(
            f'{prefix}.fc.weight', [1, 1, _FC_IN_FEATURES, _NUM_CLASSES], device, weights_dtype, layout,
        ),
        'bias': _weight(
            f'{prefix}.fc.bias', [1, 1, 1, _NUM_CLASSES], device, weights_dtype, layout,
        ),
    }

    parameters = dict2obj(params)
    _assert_tree_matches_canonical(parameters)
    return parameters


def _assert_tree_matches_canonical(p):
    """Guard the two shapes the previous port got wrong, plus the stage widths.

    Cheap, and it fails at construction with a clear message rather than as an
    opaque shape error deep inside conv or matmul shape inference.
    """
    conv1_w = list(p.conv1.weight.shape)
    assert conv1_w == [64, 16, 4, 4], (
        f'conv1 weight must be the FOLDED [64, 16, 4, 4], got {conv1_w}. '
        'In-channels is _nearest_y(3, 4) * stride_h * stride_w = 16, not 3 — '
        'see pad_and_fold_conv_filters_for_unity_stride.'
    )

    fc_w = list(p.fc.weight.shape)
    assert fc_w == [1, 1, 2048, 1000], (
        f'fc weight must be transposed and rank 4, [1, 1, 2048, 1000], got {fc_w}. '
        'preprocess_linear_weight applies weight.T.contiguous().'
    )

    in_channels = 64
    for stage_idx, (blocks, planes) in enumerate(zip(_LAYERS, _PLANES), start=1):
        stage = getattr(p, f'layer{stage_idx}')
        out_channels = planes * _EXPANSION
        assert len(stage) == blocks, f'layer{stage_idx}: expected {blocks} blocks, got {len(stage)}'

        for block_idx, block in enumerate(stage):
            block_in = in_channels if block_idx == 0 else out_channels
            assert list(block.conv1.weight.shape) == [planes, block_in, 1, 1]
            assert list(block.conv2.weight.shape) == [planes, planes, 3, 3]
            assert list(block.conv3.weight.shape) == [out_channels, planes, 1, 1]

            has_ds = hasattr(block, 'downsample')
            assert has_ds == (block_idx == 0), (
                f'layer{stage_idx}.{block_idx}: downsample presence should follow '
                'the stage-first-block rule in _make_layer'
            )
            if has_ds:
                assert list(block.downsample.weight.shape) == [out_channels, block_in, 1, 1]

        in_channels = out_channels
