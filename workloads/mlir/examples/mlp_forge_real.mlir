// SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
module @jit_mlp attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  ttcore.device_module {
    builtin.module @jit_mlp attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
      func.func public @main(%arg0: tensor<4x8xf32>, %arg1: tensor<8x16xf32>, %arg2: tensor<16xf32>, %arg3: tensor<16x16xf32>, %arg4: tensor<16xf32>) -> (tensor<4x16xf32> {jax.result_info = "result"}) {
        %0 = "ttir.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
        %1 = "ttir.dot_general"(%arg0, %arg1) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<4x16xf32>
        %2 = "ttir.reshape"(%arg2) <{shape = [1 : i32, 16 : i32]}> : (tensor<16xf32>) -> tensor<1x16xf32>
        %3 = "ttir.broadcast"(%2) <{broadcast_dimensions = array<i64: 1, 1>}> : (tensor<1x16xf32>) -> tensor<1x16xf32>
        %4 = "ttir.broadcast"(%3) <{broadcast_dimensions = array<i64: 4, 1>}> : (tensor<1x16xf32>) -> tensor<4x16xf32>
        %5 = "ttir.add"(%1, %4) : (tensor<4x16xf32>, tensor<4x16xf32>) -> tensor<4x16xf32>
        %6 = "ttir.reshape"(%0) <{shape = [1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1xf32>
        %7 = "ttir.broadcast"(%6) <{broadcast_dimensions = array<i64: 4, 16>}> : (tensor<1x1xf32>) -> tensor<4x16xf32>
        %8 = "ttir.maximum"(%5, %7) : (tensor<4x16xf32>, tensor<4x16xf32>) -> tensor<4x16xf32>
        %9 = "ttir.dot_general"(%8, %arg3) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<4x16xf32>, tensor<16x16xf32>) -> tensor<4x16xf32>
        %10 = "ttir.reshape"(%arg4) <{shape = [1 : i32, 16 : i32]}> : (tensor<16xf32>) -> tensor<1x16xf32>
        %11 = "ttir.broadcast"(%10) <{broadcast_dimensions = array<i64: 1, 1>}> : (tensor<1x16xf32>) -> tensor<1x16xf32>
        %12 = "ttir.broadcast"(%11) <{broadcast_dimensions = array<i64: 4, 1>}> : (tensor<1x16xf32>) -> tensor<4x16xf32>
        %13 = "ttir.add"(%9, %12) : (tensor<4x16xf32>, tensor<4x16xf32>) -> tensor<4x16xf32>
        return %13 : tensor<4x16xf32>
      }
    }
  }
}
