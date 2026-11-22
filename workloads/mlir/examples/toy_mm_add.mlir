// SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// Y = (X @ W) + B  — MatMul + Add
func.func @toy_mm_add(%X: tensor<4x8xf32>, %W: tensor<8x16xf32>, %B: tensor<16xf32>) -> tensor<4x16xf32> {
  %H = matmul(%X, %W) : (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<4x16xf32>
  %Y = add(%H, %B) : (tensor<4x16xf32>, tensor<16xf32>) -> tensor<4x16xf32>
  return %Y : tensor<4x16xf32>
}
