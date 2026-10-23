# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Decode sampling head — minimal port of the tt_transformers on-device sampler.

Ports the decode happy-path of ``models/common/sampling/tt_sampling.py``
(``TTSampling.sample``) so the TT-Polaris graph spans the full HW capture
(logits -> sampled token), instead of stopping at lm_head.

Uses the multi_step_reduction gather: split the vocab into ``num_splits`` chunks,
top-k each chunk, then concat — matching the capture's ``TopK x2 / Concat x2``
(chunk width = vocab/num_splits = 64128, matching the capture's TopK input).

The vocab chunking is the one place the two modes call different ttnn functions: real
ttnn uses ``ttnn.split`` (which HW lowers to ``SliceDeviceOperation`` kernels), while on
TT-Polaris the chunks are emitted as ``ttnn.slice`` ops directly, because the shim's
``ttnn.split`` collapses to a single ``Split`` op instead of one kernel per chunk. Both
sides therefore agree op-for-op with the capture. See the inline note at the gather; the
fork disappears once the shim emits split's sub-ops itself.

Host/multichip steps (deallocate / from_torch / all_gather / to_memory_config)
are omitted — they do not belong in the single-chip sim graph. Param tensors
(seeds / user_ids / k / p / temp / device offsets) are config-only dummies:
their values are irrelevant to op shapes and device time (accuracy not validated).
"""
import os

IS_POLARIS = os.getenv('IRD_ARCH_NAME', '') == ''
if IS_POLARIS:
    import ttsim.front.ttnn as ttnn
else:
    import ttnn  # type: ignore[import-not-found, no-redef]


def _dummy(shape, device, dtype, layout=None):
    """Config-only dummy tensor (fabricated on-device; values irrelevant).

    layout defaults to TILE; the sampling index/param tensors (seeds, user_ids, ...) are
    ROW_MAJOR on hardware (capture: ManualSeed / Sampling operands are ROW_MAJOR)."""
    return ttnn.zeros(list(shape), dtype=dtype,
                      layout=layout if layout is not None else ttnn.TILE_LAYOUT, device=device)


def sample_decode(logits, device, max_top_k=32, num_splits=2):
    """logits ``[1, 1, batch, vocab]`` -> sampled token id ``[1, 1, batch, 1]``.

    Sequence: typecast -> topk -> typecast -> add(device offsets) -> untilize
    -> manual_seed -> sampling.
    """
    batch = list(logits.shape)[-2]
    vocab = list(logits.shape)[-1]
    assert vocab % (2 * num_splits) == 0, (
        f"vocab ({vocab}) must be divisible by 2*num_splits ({2 * num_splits}): the values "
        f"chunk splits vocab into num_splits and the index chunk splits vocab//2, so vocab must "
        f"be even and both splits exact"
    )

    x_bf16 = ttnn.typecast(logits, dtype=ttnn.bfloat16)
    # multi_step_reduction: split the vocab across sub-core grids, top-k each chunk, then gather
    # (matches the capture's TopK x2 + Concat x2; values chunk width = vocab/2). Each topk takes a
    # pre-allocated uint16 index operand (tt-metal tt_sampling.py: indices_tensor per chunk) as its
    # 2nd input — the HW TopKDeviceOperation is arity-2 (values + indices). The index tensor is
    # vocab/2 wide (capture in1 = 32064/chunk = (vocab/2)/num_splits), TILE.
    tt_indices = _dummy([1, 1, batch, vocab // 2], device, ttnn.uint16, layout=ttnn.TILE_LAYOUT)
    # Dual-mode multi_step_reduction gather. Real ttnn uses ttnn.split (tt-metal tt_sampling.py:503-
    # 504), which HW lowers to SliceDeviceOperation. The TT-Polaris shim's ttnn.split emits ONE Split op,
    # so on TT-Polaris we emit the Slice chunks directly to match the capture (SliceDeviceOperation x4).
    if IS_POLARIS:
        def _slice_chunks(t, dim_size, n):
            step = dim_size // n
            return [
                ttnn.slice(t, slice=(slice(None), slice(None), slice(None), slice(i * step, (i + 1) * step)))
                for i in range(n)
            ]
        x_chunks = _slice_chunks(x_bf16, vocab, num_splits)
        idx_chunks = _slice_chunks(tt_indices, vocab // 2, num_splits)
    else:
        x_chunks = ttnn.split(x_bf16, vocab // num_splits, dim=3)
        idx_chunks = ttnn.split(tt_indices, (vocab // 2) // num_splits, dim=3)
    tv_list, ti_list = [], []
    for chunk, idx in zip(x_chunks, idx_chunks):
        tv, ti = ttnn.topk(chunk, k=max_top_k, dim=-1, indices_tensor=idx)
        tv_list.append(tv)
        ti_list.append(ti)
    topk_values = ttnn.concat(tv_list, dim=3)
    topk_indices = ttnn.concat(ti_list, dim=3)
    # LUT-miss fix (memory): HW sampling-tail tensors are DRAM-interleaved — the capture's
    # Sampling input_0/input_1 and the global-index Add operands are all DRAM. The sim otherwise
    # inherits the lm_head L1 logits memory down the typecast->slice->topk->concat chain (concat/add
    # propagate input memory), so the sampling/add keys came out L1_INTERLEAVED and missed. Retag the
    # concat outputs to DRAM (topk/concat write DRAM on HW); the typecast + add below then inherit it,
    # so Add (both operands DRAM) and Sampling (in0/in1 DRAM) match the capture. TT-Polaris-only: real
    # ttnn manages memory (topk/concat already produce DRAM there), and Tensor has no _memory_config.
    if IS_POLARIS:
        topk_values._memory_config = ttnn.DRAM_MEMORY_CONFIG
        topk_indices._memory_config = ttnn.DRAM_MEMORY_CONFIG
        # HW TopK indices are UINT16 (capture: the following Typecast reads a UINT16 32x64 input),
        # but the shim propagates the bfloat16 *values* dtype to both topk outputs. Retag the indices
        # to uint16 so the downstream Typecast keys on UINT16 (matches the capture). Contained to the
        # llama3 sampler — the generic shim topk keeps its ONNX-style dtype for other callers.
        topk_indices._ttnn_dtype = ttnn.uint16

    topk_indices_i32 = ttnn.typecast(topk_indices, dtype=ttnn.int32)
    offsets = _dummy(list(topk_indices_i32.shape), device, ttnn.int32)
    global_indices = ttnn.add(offsets, topk_indices_i32, dtype=ttnn.uint32)
    # LUT-miss fix (layout + op-count): the capture has an Untilize here
    # (UntilizeDeviceOperation: TILE UINT32 -> ROW_MAJOR) so the sampler reads ROW_MAJOR global
    # indices. global_indices is TILE from the add, so untilize is valid — emit it to match the
    # capture op AND give Sampling input_1 the ROW_MAJOR layout it records.
    global_indices = ttnn.untilize(global_indices)

    # Sampling param tensors are 1-D (size batch) on HW: tt-metal tt_sampling.py builds
    # k/p/temp = torch.ones/zeros(total_param_size) and seeds/user_ids = torch.arange(...), all rank-1.
    # ttnn.manual_seed asserts seeds/user_ids rank == 1 (manual_seed_operation.cpp:82); a [1,batch]
    # (rank-2) tensor fails. Rank-1 [batch] and rank-2 [1,batch] pad to the same 4-D LUT key (leading
    # 1s), so TT-Polaris is unchanged.
    seeds = _dummy([batch], device, ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
    user_ids = _dummy([batch], device, ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
    ttnn.manual_seed(seeds=seeds, user_ids=user_ids)

    # k/p/temp are ROW_MAJOR on HW (capture Sampling input_2/3/4 = ROW_MAJOR), p/temp bfloat16
    # (input_3/4 BFLOAT16), not float32/TILE; and 1-D [batch] (tt-metal builds them rank-1).
    k_t = _dummy([batch], device, ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
    p_t = _dummy([batch], device, ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    temp_t = _dummy([batch], device, ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    # LUT-miss fix (arity): HW ttnn.sampling takes a preallocated output_tensor (tt-metal
    # tt_sampling.py: output_tensor=tt_out_tok), which the profiler logs as Sampling input_5
    # (uint32, DRAM). Pass it so the sim emits the arity-6 op the capture records.
    out_tok = _dummy([1, 1, 1, batch], device, ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
    return ttnn.sampling(topk_values, global_indices, k=k_t, p=p_t, temp=temp_t, output_tensor=out_tok)


def sample_argmax(logits):
    """logits ``[1, 1, batch, vocab]`` -> greedy token ids, via tt-metal's force-argmax path.

    tt-metal f8c634a tt_sampling.py forward(): when every user is greedy (k=1, p in {0,1}, temp=1)
    and the model allows it, sampling skips top-k / top-p / RNG and runs _untilize_for_argmax then
    ttnn.argmax(dim=-1, keepdim=False). For llama3 the vocab (128256) needs no trim slice and fits
    one untilize program (<= 2 * TOPK_MAX_WIDTH), so the device ops are Untilize -> ArgMax (BH p100a
    261018 and WH n150 261015 captures: Untilize [1,1,32,128256] BFLOAT8_B -> BFLOAT16, ArgMax ->
    UINT32). No preallocated output: the captures' ArgMax has a single input.
    """
    x = ttnn.untilize(logits, use_multicore=True)
    return ttnn.argmax(x, dim=-1, keepdim=False)
