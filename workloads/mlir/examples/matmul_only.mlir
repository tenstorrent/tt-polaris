// SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
func.func @matmul_only(%A: tensor<32x64xf32>, %B: tensor<64x128xf32>) -> tensor<32x128xf32> {
  %C = matmul(%A, %B) : (tensor<32x64xf32>, tensor<64x128xf32>) -> tensor<32x128xf32>
  return %C : tensor<32x128xf32>
}
