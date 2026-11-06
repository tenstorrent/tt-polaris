# #!/usr/bin/env python
# # SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# # SPDX-License-Identifier: Apache-2.0

# import os
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# import torch
# import onnx
# import onnxruntime as ort
# from transformers import WhisperForConditionalGeneration

# model_name = "openai/whisper-base"
# raw_onnx_path = "whisper-base-fixed-raw.onnx"
# final_onnx_path = "workloads/onnx/whisper-base-fixed.onnx"

# # Load model
# model = WhisperForConditionalGeneration.from_pretrained(
#     model_name,
#     attn_implementation="eager"
# )
# model.config.use_cache = False
# model.eval()

# BATCH_SIZE = 1
# FEATURE_DIM = 80
# AUDIO_SEQ_LEN = 3000
# DECODER_SEQ_LEN = 16

# input_features = torch.randn(BATCH_SIZE, FEATURE_DIM, AUDIO_SEQ_LEN, dtype=torch.float32)
# decoder_input_ids = torch.zeros((BATCH_SIZE, DECODER_SEQ_LEN), dtype=torch.long)
# decoder_attention_mask = torch.ones((BATCH_SIZE, DECODER_SEQ_LEN), dtype=torch.long)


# class WhisperWrapper(torch.nn.Module):
#     def __init__(self, model):
#         super().__init__()
#         self.model = model

#     def forward(self, input_features, decoder_input_ids, decoder_attention_mask):
#         return self.model(
#             input_features=input_features, 
#             decoder_input_ids=decoder_input_ids,
#             decoder_attention_mask=decoder_attention_mask,
#             return_dict=False
#         )[0]


# wrapped_model = WhisperWrapper(model)

# print("Exporting initial ONNX graph...")
# torch.onnx.export(
#     wrapped_model,
#     (input_features, decoder_input_ids, decoder_attention_mask),
#     raw_onnx_path,
#     input_names=["input_features", "decoder_input_ids", "decoder_attention_mask"],
#     output_names=["logits"],
#     opset_version=17,
#     dynamic_axes=None,
#     do_constant_folding=True,
#     export_params=True,
#     dynamo=False
# )

# print("Folding shape constants via ONNX Runtime graph optimization...")
# os.makedirs(os.path.dirname(final_onnx_path), exist_ok=True)

# opt_options = ort.SessionOptions()
# opt_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
# opt_options.optimized_model_filepath = final_onnx_path
# _ = ort.InferenceSession(raw_onnx_path, opt_options)

# if os.path.exists(raw_onnx_path):
#     os.remove(raw_onnx_path)

# print(f"Saved optimized, fully static ONNX: {final_onnx_path}")