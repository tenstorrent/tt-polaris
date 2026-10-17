# #!/usr/bin/env python
# # SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# # SPDX-License-Identifier: Apache-2.0

# import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# import torch
# from transformers import AutoModelForCausalLM

# class DeepSeekExportWrapper(torch.nn.Module):
#     """
#     Wraps the causal language model to cleanly extract the logits tensor,
#     avoiding Hugging Face dataclass tracing errors during ONNX export.
#     """
#     def __init__(self, model):
#         super().__init__()
#         self.model = model

#     def forward(self, input_ids, attention_mask):
#         return self.model(
#             input_ids=input_ids, 
#             attention_mask=attention_mask
#         ).logits

# model_name = "deepseek-ai/deepseek-coder-1.3b-base"

# # Load model
# print(f"Loading {model_name}...")
# model = AutoModelForCausalLM.from_pretrained(model_name)
# model.config.use_cache = False
# model.eval()

# wrapped_model = DeepSeekExportWrapper(model)

# BATCH_SIZE = 1
# SEQ_LEN = 128

# input_ids = torch.randint(0, model.config.vocab_size, (BATCH_SIZE, SEQ_LEN), dtype=torch.long)
# attention_mask = torch.ones((BATCH_SIZE, SEQ_LEN), dtype=torch.long)

# script_dir = os.path.dirname(os.path.abspath(__file__))
# onnx_dir = os.path.abspath(os.path.join(script_dir, ".."))
# output_path = os.path.join(onnx_dir, "deepseek-coder-1.3b-fixed.onnx")

# print("Exporting to ONNX...")
# torch.onnx.export(
#     wrapped_model,
#     (input_ids, attention_mask),
#     output_path,
#     input_names=["input_ids", "attention_mask"],
#     output_names=["logits"],
#     opset_version=17,
#     dynamic_axes=None
# )

# print(f"Saved fixed-shape ONNX to: {output_path}")