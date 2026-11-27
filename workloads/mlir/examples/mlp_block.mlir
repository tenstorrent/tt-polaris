// SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// matmul -> add -> transpose -> matmul -> add
func.func @mlp_block(%X: tensor<4x8xf32>, %W1: tensor<8x16xf32>, %B1: tensor<16xf32>, %W2: tensor<4x16xf32>, %B2: tensor<16x16xf32>) -> tensor<16x16xf32> {
  %H1 = matmul(%X, %W1) : (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<4x16xf32>
  %H2 = add(%H1, %B1) : (tensor<4x16xf32>, tensor<16xf32>) -> tensor<4x16xf32>
  %H3 = transpose(%H2) {perm = [1, 0]} : (tensor<4x16xf32>) -> tensor<16x4xf32>
  %H4 = matmul(%H3, %W2) : (tensor<16x4xf32>, tensor<4x16xf32>) -> tensor<16x16xf32>
  %Y = add(%H4, %B2) : (tensor<16x16xf32>, tensor<16x16xf32>) -> tensor<16x16xf32>
  return %Y : tensor<16x16xf32>
}
