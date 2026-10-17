// SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
module @jit_enc attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  ttcore.device_module {
    builtin.module @jit_enc attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
      func.func public @main(%arg0: tensor<1x512x768xf32>, %arg1: tensor<768x768xf32>, %arg2: tensor<768x768xf32>, %arg3: tensor<768x768xf32>, %arg4: tensor<768x768xf32>, %arg5: tensor<768xf32>, %arg6: tensor<768xf32>, %arg7: tensor<768x3072xf32>, %arg8: tensor<3072x768xf32>, %arg9: tensor<768xf32>, %arg10: tensor<768xf32>, %arg11: tensor<768x768xf32>, %arg12: tensor<768x768xf32>, %arg13: tensor<768x768xf32>, %arg14: tensor<768x768xf32>, %arg15: tensor<768xf32>, %arg16: tensor<768xf32>, %arg17: tensor<768x3072xf32>, %arg18: tensor<3072x768xf32>, %arg19: tensor<768xf32>, %arg20: tensor<768xf32>, %arg21: tensor<768x768xf32>, %arg22: tensor<768x768xf32>, %arg23: tensor<768x768xf32>, %arg24: tensor<768x768xf32>, %arg25: tensor<768xf32>, %arg26: tensor<768xf32>, %arg27: tensor<768x3072xf32>, %arg28: tensor<3072x768xf32>, %arg29: tensor<768xf32>, %arg30: tensor<768xf32>, %arg31: tensor<768x768xf32>, %arg32: tensor<768x768xf32>, %arg33: tensor<768x768xf32>, %arg34: tensor<768x768xf32>, %arg35: tensor<768xf32>, %arg36: tensor<768xf32>, %arg37: tensor<768x3072xf32>, %arg38: tensor<3072x768xf32>, %arg39: tensor<768xf32>, %arg40: tensor<768xf32>, %arg41: tensor<768x768xf32>, %arg42: tensor<768x768xf32>, %arg43: tensor<768x768xf32>, %arg44: tensor<768x768xf32>, %arg45: tensor<768xf32>, %arg46: tensor<768xf32>, %arg47: tensor<768x3072xf32>, %arg48: tensor<3072x768xf32>, %arg49: tensor<768xf32>, %arg50: tensor<768xf32>, %arg51: tensor<768x768xf32>, %arg52: tensor<768x768xf32>, %arg53: tensor<768x768xf32>, %arg54: tensor<768x768xf32>, %arg55: tensor<768xf32>, %arg56: tensor<768xf32>, %arg57: tensor<768x3072xf32>, %arg58: tensor<3072x768xf32>, %arg59: tensor<768xf32>, %arg60: tensor<768xf32>, %arg61: tensor<768x768xf32>, %arg62: tensor<768x768xf32>, %arg63: tensor<768x768xf32>, %arg64: tensor<768x768xf32>, %arg65: tensor<768xf32>, %arg66: tensor<768xf32>, %arg67: tensor<768x3072xf32>, %arg68: tensor<3072x768xf32>, %arg69: tensor<768xf32>, %arg70: tensor<768xf32>, %arg71: tensor<768x768xf32>, %arg72: tensor<768x768xf32>, %arg73: tensor<768x768xf32>, %arg74: tensor<768x768xf32>, %arg75: tensor<768xf32>, %arg76: tensor<768xf32>, %arg77: tensor<768x3072xf32>, %arg78: tensor<3072x768xf32>, %arg79: tensor<768xf32>, %arg80: tensor<768xf32>, %arg81: tensor<768x768xf32>, %arg82: tensor<768x768xf32>, %arg83: tensor<768x768xf32>, %arg84: tensor<768x768xf32>, %arg85: tensor<768xf32>, %arg86: tensor<768xf32>, %arg87: tensor<768x3072xf32>, %arg88: tensor<3072x768xf32>, %arg89: tensor<768xf32>, %arg90: tensor<768xf32>, %arg91: tensor<768x768xf32>, %arg92: tensor<768x768xf32>, %arg93: tensor<768x768xf32>, %arg94: tensor<768x768xf32>, %arg95: tensor<768xf32>, %arg96: tensor<768xf32>, %arg97: tensor<768x3072xf32>, %arg98: tensor<3072x768xf32>, %arg99: tensor<768xf32>, %arg100: tensor<768xf32>, %arg101: tensor<768x768xf32>, %arg102: tensor<768x768xf32>, %arg103: tensor<768x768xf32>, %arg104: tensor<768x768xf32>, %arg105: tensor<768xf32>, %arg106: tensor<768xf32>, %arg107: tensor<768x3072xf32>, %arg108: tensor<3072x768xf32>, %arg109: tensor<768xf32>, %arg110: tensor<768xf32>, %arg111: tensor<768x768xf32>, %arg112: tensor<768x768xf32>, %arg113: tensor<768x768xf32>, %arg114: tensor<768x768xf32>, %arg115: tensor<768xf32>, %arg116: tensor<768xf32>, %arg117: tensor<768x3072xf32>, %arg118: tensor<3072x768xf32>, %arg119: tensor<768xf32>, %arg120: tensor<768xf32>) -> (tensor<1x512x768xf32> {jax.result_info = "result"}) {
        %0 = "ttir.constant"() <{value = dense<5.000000e-01> : tensor<f32>}> : () -> tensor<f32>
        %1 = "ttir.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
        %2 = "ttir.constant"() <{value = dense<0.797884583> : tensor<f32>}> : () -> tensor<f32>
        %3 = "ttir.constant"() <{value = dense<4.471500e-02> : tensor<f32>}> : () -> tensor<f32>
        %4 = "ttir.constant"() <{value = dense<0xFF800000> : tensor<f32>}> : () -> tensor<f32>
        %5 = "ttir.constant"() <{value = dense<6.400000e+01> : tensor<f32>}> : () -> tensor<f32>
        %6 = "ttir.constant"() <{value = dense<9.99999974E-6> : tensor<f32>}> : () -> tensor<f32>
        %7 = "ttir.constant"() <{value = dense<7.680000e+02> : tensor<f32>}> : () -> tensor<f32>
        %8 = "ttir.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
        %9 = "ttir.sum"(%arg0) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x512x768xf32>) -> tensor<1x512xf32>
        %10 = "ttir.reshape"(%9) <{shape = [1 : i32, 512 : i32, 1 : i32]}> : (tensor<1x512xf32>) -> tensor<1x512x1xf32>
        %11 = "ttir.broadcast"(%10) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %12 = "ttir.reshape"(%7) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %13 = "ttir.broadcast"(%12) <{broadcast_dimensions = array<i64: 1, 512, 1>}> : (tensor<1x1x1xf32>) -> tensor<1x512x1xf32>
        %14 = "ttir.div"(%11, %13) : (tensor<1x512x1xf32>, tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %15 = "ttir.broadcast"(%14) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<1x512x1xf32>) -> tensor<1x512x768xf32>
        %16 = "ttir.subtract"(%arg0, %15) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %17 = "ttir.multiply"(%16, %16) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %18 = "ttir.sum"(%17) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x512x768xf32>) -> tensor<1x512xf32>
        %19 = "ttir.reshape"(%18) <{shape = [1 : i32, 512 : i32, 1 : i32]}> : (tensor<1x512xf32>) -> tensor<1x512x1xf32>
        %20 = "ttir.broadcast"(%19) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %21 = "ttir.reshape"(%7) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %22 = "ttir.broadcast"(%21) <{broadcast_dimensions = array<i64: 1, 512, 1>}> : (tensor<1x1x1xf32>) -> tensor<1x512x1xf32>
        %23 = "ttir.div"(%20, %22) : (tensor<1x512x1xf32>, tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %24 = "ttir.broadcast"(%14) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<1x512x1xf32>) -> tensor<1x512x768xf32>
        %25 = "ttir.subtract"(%arg0, %24) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %26 = "ttir.reshape"(%6) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %27 = "ttir.broadcast"(%26) <{broadcast_dimensions = array<i64: 1, 512, 1>}> : (tensor<1x1x1xf32>) -> tensor<1x512x1xf32>
        %28 = "ttir.add"(%23, %27) : (tensor<1x512x1xf32>, tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %29 = "ttir.sqrt"(%28) : (tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %30 = "ttir.broadcast"(%29) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<1x512x1xf32>) -> tensor<1x512x768xf32>
        %31 = "ttir.div"(%25, %30) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %32 = "ttir.reshape"(%arg5) <{shape = [1 : i32, 1 : i32, 768 : i32]}> : (tensor<768xf32>) -> tensor<1x1x768xf32>
        %33 = "ttir.broadcast"(%32) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x1x768xf32>
        %34 = "ttir.broadcast"(%33) <{broadcast_dimensions = array<i64: 1, 512, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x512x768xf32>
        %35 = "ttir.multiply"(%31, %34) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %36 = "ttir.reshape"(%arg6) <{shape = [1 : i32, 1 : i32, 768 : i32]}> : (tensor<768xf32>) -> tensor<1x1x768xf32>
        %37 = "ttir.broadcast"(%36) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x1x768xf32>
        %38 = "ttir.broadcast"(%37) <{broadcast_dimensions = array<i64: 1, 512, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x512x768xf32>
        %39 = "ttir.add"(%35, %38) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %40 = "ttir.dot_general"(%39, %arg1) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<1x512x768xf32>, tensor<768x768xf32>) -> tensor<1x512x768xf32>
        %41 = "ttir.dot_general"(%39, %arg2) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<1x512x768xf32>, tensor<768x768xf32>) -> tensor<1x512x768xf32>
        %42 = "ttir.dot_general"(%39, %arg3) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<1x512x768xf32>, tensor<768x768xf32>) -> tensor<1x512x768xf32>
        %43 = "ttir.reshape"(%40) <{shape = [1 : i32, 512 : i32, 12 : i32, 64 : i32]}> : (tensor<1x512x768xf32>) -> tensor<1x512x12x64xf32>
        %44 = "ttir.permute"(%43) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x512x12x64xf32>) -> tensor<1x12x512x64xf32>
        %45 = "ttir.reshape"(%41) <{shape = [1 : i32, 512 : i32, 12 : i32, 64 : i32]}> : (tensor<1x512x768xf32>) -> tensor<1x512x12x64xf32>
        %46 = "ttir.permute"(%45) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x512x12x64xf32>) -> tensor<1x12x512x64xf32>
        %47 = "ttir.reshape"(%42) <{shape = [1 : i32, 512 : i32, 12 : i32, 64 : i32]}> : (tensor<1x512x768xf32>) -> tensor<1x512x12x64xf32>
        %48 = "ttir.permute"(%47) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x512x12x64xf32>) -> tensor<1x12x512x64xf32>
        %49 = "ttir.permute"(%46) <{permutation = array<i64: 0, 1, 3, 2>}> : (tensor<1x12x512x64xf32>) -> tensor<1x12x64x512xf32>
        %50 = "ttir.reshape"(%44) <{shape = [12 : i32, 512 : i32, 64 : i32]}> : (tensor<1x12x512x64xf32>) -> tensor<12x512x64xf32>
        %51 = "ttir.dot_general"(%50, %49) <{batch_dims_lhs = array<i64: 0>, batch_dims_rhs = array<i64: 1>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 2>}> : (tensor<12x512x64xf32>, tensor<1x12x64x512xf32>) -> tensor<12x512x1x512xf32>
        %52 = "ttir.permute"(%51) <{permutation = array<i64: 2, 0, 1, 3>}> : (tensor<12x512x1x512xf32>) -> tensor<1x12x512x512xf32>
        %53 = "ttir.sqrt"(%5) : (tensor<f32>) -> tensor<f32>
        %54 = "ttir.typecast"(%53) <{conservative_folding = false}> : (tensor<f32>) -> tensor<f32>
        %55 = "ttir.reshape"(%54) <{shape = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1x1xf32>
        %56 = "ttir.broadcast"(%55) <{broadcast_dimensions = array<i64: 1, 12, 512, 512>}> : (tensor<1x1x1x1xf32>) -> tensor<1x12x512x512xf32>
        %57 = "ttir.div"(%52, %56) : (tensor<1x12x512x512xf32>, tensor<1x12x512x512xf32>) -> tensor<1x12x512x512xf32>
        %58 = "ttir.max"(%57) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<1x12x512x512xf32>) -> tensor<1x12x512xf32>
        %59 = "ttir.reshape"(%4) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %60 = "ttir.broadcast"(%59) <{broadcast_dimensions = array<i64: 1, 12, 512>}> : (tensor<1x1x1xf32>) -> tensor<1x12x512xf32>
        %61 = "ttir.maximum"(%60, %58) : (tensor<1x12x512xf32>, tensor<1x12x512xf32>) -> tensor<1x12x512xf32>
        %62 = "ttir.reshape"(%61) <{shape = [1 : i32, 12 : i32, 512 : i32, 1 : i32]}> : (tensor<1x12x512xf32>) -> tensor<1x12x512x1xf32>
        %63 = "ttir.broadcast"(%62) <{broadcast_dimensions = array<i64: 1, 1, 1, 1>}> : (tensor<1x12x512x1xf32>) -> tensor<1x12x512x1xf32>
        %64 = "ttir.broadcast"(%63) <{broadcast_dimensions = array<i64: 1, 1, 1, 512>}> : (tensor<1x12x512x1xf32>) -> tensor<1x12x512x512xf32>
        %65 = "ttir.subtract"(%57, %64) : (tensor<1x12x512x512xf32>, tensor<1x12x512x512xf32>) -> tensor<1x12x512x512xf32>
        %66 = "ttir.exp"(%65) : (tensor<1x12x512x512xf32>) -> tensor<1x12x512x512xf32>
        %67 = "ttir.sum"(%66) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<1x12x512x512xf32>) -> tensor<1x12x512xf32>
        %68 = "ttir.reshape"(%67) <{shape = [1 : i32, 12 : i32, 512 : i32, 1 : i32]}> : (tensor<1x12x512xf32>) -> tensor<1x12x512x1xf32>
        %69 = "ttir.broadcast"(%68) <{broadcast_dimensions = array<i64: 1, 1, 1, 1>}> : (tensor<1x12x512x1xf32>) -> tensor<1x12x512x1xf32>
        %70 = "ttir.broadcast"(%69) <{broadcast_dimensions = array<i64: 1, 1, 1, 512>}> : (tensor<1x12x512x1xf32>) -> tensor<1x12x512x512xf32>
        %71 = "ttir.div"(%66, %70) : (tensor<1x12x512x512xf32>, tensor<1x12x512x512xf32>) -> tensor<1x12x512x512xf32>
        %72 = "ttir.reshape"(%71) <{shape = [12 : i32, 512 : i32, 512 : i32]}> : (tensor<1x12x512x512xf32>) -> tensor<12x512x512xf32>
        %73 = "ttir.dot_general"(%72, %48) <{batch_dims_lhs = array<i64: 0>, batch_dims_rhs = array<i64: 1>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 2>}> : (tensor<12x512x512xf32>, tensor<1x12x512x64xf32>) -> tensor<12x512x1x64xf32>
        %74 = "ttir.permute"(%73) <{permutation = array<i64: 2, 0, 1, 3>}> : (tensor<12x512x1x64xf32>) -> tensor<1x12x512x64xf32>
        %75 = "ttir.permute"(%74) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x12x512x64xf32>) -> tensor<1x512x12x64xf32>
        %76 = "ttir.reshape"(%75) <{shape = [1 : i32, 512 : i32, 768 : i32]}> : (tensor<1x512x12x64xf32>) -> tensor<1x512x768xf32>
        %77 = "ttir.dot_general"(%76, %arg4) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<1x512x768xf32>, tensor<768x768xf32>) -> tensor<1x512x768xf32>
        %78 = "ttir.add"(%arg0, %77) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %79 = "ttir.sum"(%78) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x512x768xf32>) -> tensor<1x512xf32>
        %80 = "ttir.reshape"(%79) <{shape = [1 : i32, 512 : i32, 1 : i32]}> : (tensor<1x512xf32>) -> tensor<1x512x1xf32>
        %81 = "ttir.broadcast"(%80) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %82 = "ttir.reshape"(%7) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %83 = "ttir.broadcast"(%82) <{broadcast_dimensions = array<i64: 1, 512, 1>}> : (tensor<1x1x1xf32>) -> tensor<1x512x1xf32>
        %84 = "ttir.div"(%81, %83) : (tensor<1x512x1xf32>, tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %85 = "ttir.broadcast"(%84) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<1x512x1xf32>) -> tensor<1x512x768xf32>
        %86 = "ttir.subtract"(%78, %85) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %87 = "ttir.multiply"(%86, %86) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %88 = "ttir.sum"(%87) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x512x768xf32>) -> tensor<1x512xf32>
        %89 = "ttir.reshape"(%88) <{shape = [1 : i32, 512 : i32, 1 : i32]}> : (tensor<1x512xf32>) -> tensor<1x512x1xf32>
        %90 = "ttir.broadcast"(%89) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %91 = "ttir.reshape"(%7) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %92 = "ttir.broadcast"(%91) <{broadcast_dimensions = array<i64: 1, 512, 1>}> : (tensor<1x1x1xf32>) -> tensor<1x512x1xf32>
        %93 = "ttir.div"(%90, %92) : (tensor<1x512x1xf32>, tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %94 = "ttir.broadcast"(%84) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<1x512x1xf32>) -> tensor<1x512x768xf32>
        %95 = "ttir.subtract"(%78, %94) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %96 = "ttir.reshape"(%6) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %97 = "ttir.broadcast"(%96) <{broadcast_dimensions = array<i64: 1, 512, 1>}> : (tensor<1x1x1xf32>) -> tensor<1x512x1xf32>
        %98 = "ttir.add"(%93, %97) : (tensor<1x512x1xf32>, tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %99 = "ttir.sqrt"(%98) : (tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %100 = "ttir.broadcast"(%99) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<1x512x1xf32>) -> tensor<1x512x768xf32>
        %101 = "ttir.div"(%95, %100) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %102 = "ttir.reshape"(%arg9) <{shape = [1 : i32, 1 : i32, 768 : i32]}> : (tensor<768xf32>) -> tensor<1x1x768xf32>
        %103 = "ttir.broadcast"(%102) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x1x768xf32>
        %104 = "ttir.broadcast"(%103) <{broadcast_dimensions = array<i64: 1, 512, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x512x768xf32>
        %105 = "ttir.multiply"(%101, %104) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %106 = "ttir.reshape"(%arg10) <{shape = [1 : i32, 1 : i32, 768 : i32]}> : (tensor<768xf32>) -> tensor<1x1x768xf32>
        %107 = "ttir.broadcast"(%106) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x1x768xf32>
        %108 = "ttir.broadcast"(%107) <{broadcast_dimensions = array<i64: 1, 512, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x512x768xf32>
        %109 = "ttir.add"(%105, %108) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %110 = "ttir.dot_general"(%109, %arg7) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<1x512x768xf32>, tensor<768x3072xf32>) -> tensor<1x512x3072xf32>
        %111 = "ttir.multiply"(%110, %110) : (tensor<1x512x3072xf32>, tensor<1x512x3072xf32>) -> tensor<1x512x3072xf32>
        %112 = "ttir.multiply"(%111, %110) : (tensor<1x512x3072xf32>, tensor<1x512x3072xf32>) -> tensor<1x512x3072xf32>
        %113 = "ttir.reshape"(%3) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %114 = "ttir.broadcast"(%113) <{broadcast_dimensions = array<i64: 1, 512, 3072>}> : (tensor<1x1x1xf32>) -> tensor<1x512x3072xf32>
        %115 = "ttir.multiply"(%114, %112) : (tensor<1x512x3072xf32>, tensor<1x512x3072xf32>) -> tensor<1x512x3072xf32>
        %116 = "ttir.add"(%110, %115) : (tensor<1x512x3072xf32>, tensor<1x512x3072xf32>) -> tensor<1x512x3072xf32>
        %117 = "ttir.reshape"(%2) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %118 = "ttir.broadcast"(%117) <{broadcast_dimensions = array<i64: 1, 512, 3072>}> : (tensor<1x1x1xf32>) -> tensor<1x512x3072xf32>
        %119 = "ttir.multiply"(%118, %116) : (tensor<1x512x3072xf32>, tensor<1x512x3072xf32>) -> tensor<1x512x3072xf32>
        %120 = "ttir.tanh"(%119) : (tensor<1x512x3072xf32>) -> tensor<1x512x3072xf32>
        %121 = "ttir.reshape"(%1) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %122 = "ttir.broadcast"(%121) <{broadcast_dimensions = array<i64: 1, 512, 3072>}> : (tensor<1x1x1xf32>) -> tensor<1x512x3072xf32>
        %123 = "ttir.add"(%122, %120) : (tensor<1x512x3072xf32>, tensor<1x512x3072xf32>) -> tensor<1x512x3072xf32>
        %124 = "ttir.reshape"(%0) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %125 = "ttir.broadcast"(%124) <{broadcast_dimensions = array<i64: 1, 512, 3072>}> : (tensor<1x1x1xf32>) -> tensor<1x512x3072xf32>
        %126 = "ttir.multiply"(%125, %123) : (tensor<1x512x3072xf32>, tensor<1x512x3072xf32>) -> tensor<1x512x3072xf32>
        %127 = "ttir.multiply"(%110, %126) : (tensor<1x512x3072xf32>, tensor<1x512x3072xf32>) -> tensor<1x512x3072xf32>
        %128 = "ttir.dot_general"(%127, %arg8) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<1x512x3072xf32>, tensor<3072x768xf32>) -> tensor<1x512x768xf32>
        %129 = "ttir.add"(%78, %128) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %130 = "ttir.sum"(%129) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x512x768xf32>) -> tensor<1x512xf32>
        %131 = "ttir.reshape"(%130) <{shape = [1 : i32, 512 : i32, 1 : i32]}> : (tensor<1x512xf32>) -> tensor<1x512x1xf32>
        %132 = "ttir.broadcast"(%131) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %133 = "ttir.reshape"(%7) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %134 = "ttir.broadcast"(%133) <{broadcast_dimensions = array<i64: 1, 512, 1>}> : (tensor<1x1x1xf32>) -> tensor<1x512x1xf32>
        %135 = "ttir.div"(%132, %134) : (tensor<1x512x1xf32>, tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %136 = "ttir.broadcast"(%135) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<1x512x1xf32>) -> tensor<1x512x768xf32>
        %137 = "ttir.subtract"(%129, %136) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %138 = "ttir.multiply"(%137, %137) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %139 = "ttir.sum"(%138) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x512x768xf32>) -> tensor<1x512xf32>
        %140 = "ttir.reshape"(%139) <{shape = [1 : i32, 512 : i32, 1 : i32]}> : (tensor<1x512xf32>) -> tensor<1x512x1xf32>
        %141 = "ttir.broadcast"(%140) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %142 = "ttir.reshape"(%7) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %143 = "ttir.broadcast"(%142) <{broadcast_dimensions = array<i64: 1, 512, 1>}> : (tensor<1x1x1xf32>) -> tensor<1x512x1xf32>
        %144 = "ttir.div"(%141, %143) : (tensor<1x512x1xf32>, tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %145 = "ttir.broadcast"(%135) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<1x512x1xf32>) -> tensor<1x512x768xf32>
        %146 = "ttir.subtract"(%129, %145) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %147 = "ttir.reshape"(%6) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %148 = "ttir.broadcast"(%147) <{broadcast_dimensions = array<i64: 1, 512, 1>}> : (tensor<1x1x1xf32>) -> tensor<1x512x1xf32>
        %149 = "ttir.add"(%144, %148) : (tensor<1x512x1xf32>, tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %150 = "ttir.sqrt"(%149) : (tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %151 = "ttir.broadcast"(%150) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<1x512x1xf32>) -> tensor<1x512x768xf32>
        %152 = "ttir.div"(%146, %151) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %153 = "ttir.reshape"(%arg15) <{shape = [1 : i32, 1 : i32, 768 : i32]}> : (tensor<768xf32>) -> tensor<1x1x768xf32>
        %154 = "ttir.broadcast"(%153) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x1x768xf32>
        %155 = "ttir.broadcast"(%154) <{broadcast_dimensions = array<i64: 1, 512, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x512x768xf32>
        %156 = "ttir.multiply"(%152, %155) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %157 = "ttir.reshape"(%arg16) <{shape = [1 : i32, 1 : i32, 768 : i32]}> : (tensor<768xf32>) -> tensor<1x1x768xf32>
        %158 = "ttir.broadcast"(%157) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x1x768xf32>
        %159 = "ttir.broadcast"(%158) <{broadcast_dimensions = array<i64: 1, 512, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x512x768xf32>
        %160 = "ttir.add"(%156, %159) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %161 = "ttir.dot_general"(%160, %arg11) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<1x512x768xf32>, tensor<768x768xf32>) -> tensor<1x512x768xf32>
        %162 = "ttir.dot_general"(%160, %arg12) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<1x512x768xf32>, tensor<768x768xf32>) -> tensor<1x512x768xf32>
        %163 = "ttir.dot_general"(%160, %arg13) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<1x512x768xf32>, tensor<768x768xf32>) -> tensor<1x512x768xf32>
        %164 = "ttir.reshape"(%161) <{shape = [1 : i32, 512 : i32, 12 : i32, 64 : i32]}> : (tensor<1x512x768xf32>) -> tensor<1x512x12x64xf32>
        %165 = "ttir.permute"(%164) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x512x12x64xf32>) -> tensor<1x12x512x64xf32>
        %166 = "ttir.reshape"(%162) <{shape = [1 : i32, 512 : i32, 12 : i32, 64 : i32]}> : (tensor<1x512x768xf32>) -> tensor<1x512x12x64xf32>
        %167 = "ttir.permute"(%166) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x512x12x64xf32>) -> tensor<1x12x512x64xf32>
        %168 = "ttir.reshape"(%163) <{shape = [1 : i32, 512 : i32, 12 : i32, 64 : i32]}> : (tensor<1x512x768xf32>) -> tensor<1x512x12x64xf32>
        %169 = "ttir.permute"(%168) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x512x12x64xf32>) -> tensor<1x12x512x64xf32>
        %170 = "ttir.permute"(%167) <{permutation = array<i64: 0, 1, 3, 2>}> : (tensor<1x12x512x64xf32>) -> tensor<1x12x64x512xf32>
        %171 = "ttir.reshape"(%165) <{shape = [12 : i32, 512 : i32, 64 : i32]}> : (tensor<1x12x512x64xf32>) -> tensor<12x512x64xf32>
        %172 = "ttir.dot_general"(%171, %170) <{batch_dims_lhs = array<i64: 0>, batch_dims_rhs = array<i64: 1>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 2>}> : (tensor<12x512x64xf32>, tensor<1x12x64x512xf32>) -> tensor<12x512x1x512xf32>
        %173 = "ttir.permute"(%172) <{permutation = array<i64: 2, 0, 1, 3>}> : (tensor<12x512x1x512xf32>) -> tensor<1x12x512x512xf32>
        %174 = "ttir.sqrt"(%5) : (tensor<f32>) -> tensor<f32>
        %175 = "ttir.typecast"(%174) <{conservative_folding = false}> : (tensor<f32>) -> tensor<f32>
        %176 = "ttir.reshape"(%175) <{shape = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1x1xf32>
        %177 = "ttir.broadcast"(%176) <{broadcast_dimensions = array<i64: 1, 12, 512, 512>}> : (tensor<1x1x1x1xf32>) -> tensor<1x12x512x512xf32>
        %178 = "ttir.div"(%173, %177) : (tensor<1x12x512x512xf32>, tensor<1x12x512x512xf32>) -> tensor<1x12x512x512xf32>
        %179 = "ttir.max"(%178) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<1x12x512x512xf32>) -> tensor<1x12x512xf32>
        %180 = "ttir.reshape"(%4) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %181 = "ttir.broadcast"(%180) <{broadcast_dimensions = array<i64: 1, 12, 512>}> : (tensor<1x1x1xf32>) -> tensor<1x12x512xf32>
        %182 = "ttir.maximum"(%181, %179) : (tensor<1x12x512xf32>, tensor<1x12x512xf32>) -> tensor<1x12x512xf32>
        %183 = "ttir.reshape"(%182) <{shape = [1 : i32, 12 : i32, 512 : i32, 1 : i32]}> : (tensor<1x12x512xf32>) -> tensor<1x12x512x1xf32>
        %184 = "ttir.broadcast"(%183) <{broadcast_dimensions = array<i64: 1, 1, 1, 1>}> : (tensor<1x12x512x1xf32>) -> tensor<1x12x512x1xf32>
        %185 = "ttir.broadcast"(%184) <{broadcast_dimensions = array<i64: 1, 1, 1, 512>}> : (tensor<1x12x512x1xf32>) -> tensor<1x12x512x512xf32>
        %186 = "ttir.subtract"(%178, %185) : (tensor<1x12x512x512xf32>, tensor<1x12x512x512xf32>) -> tensor<1x12x512x512xf32>
        %187 = "ttir.exp"(%186) : (tensor<1x12x512x512xf32>) -> tensor<1x12x512x512xf32>
        %188 = "ttir.sum"(%187) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<1x12x512x512xf32>) -> tensor<1x12x512xf32>
        %189 = "ttir.reshape"(%188) <{shape = [1 : i32, 12 : i32, 512 : i32, 1 : i32]}> : (tensor<1x12x512xf32>) -> tensor<1x12x512x1xf32>
        %190 = "ttir.broadcast"(%189) <{broadcast_dimensions = array<i64: 1, 1, 1, 1>}> : (tensor<1x12x512x1xf32>) -> tensor<1x12x512x1xf32>
        %191 = "ttir.broadcast"(%190) <{broadcast_dimensions = array<i64: 1, 1, 1, 512>}> : (tensor<1x12x512x1xf32>) -> tensor<1x12x512x512xf32>
        %192 = "ttir.div"(%187, %191) : (tensor<1x12x512x512xf32>, tensor<1x12x512x512xf32>) -> tensor<1x12x512x512xf32>
        %193 = "ttir.reshape"(%192) <{shape = [12 : i32, 512 : i32, 512 : i32]}> : (tensor<1x12x512x512xf32>) -> tensor<12x512x512xf32>
        %194 = "ttir.dot_general"(%193, %169) <{batch_dims_lhs = array<i64: 0>, batch_dims_rhs = array<i64: 1>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 2>}> : (tensor<12x512x512xf32>, tensor<1x12x512x64xf32>) -> tensor<12x512x1x64xf32>
        %195 = "ttir.permute"(%194) <{permutation = array<i64: 2, 0, 1, 3>}> : (tensor<12x512x1x64xf32>) -> tensor<1x12x512x64xf32>
        %196 = "ttir.permute"(%195) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x12x512x64xf32>) -> tensor<1x512x12x64xf32>
        %197 = "ttir.reshape"(%196) <{shape = [1 : i32, 512 : i32, 768 : i32]}> : (tensor<1x512x12x64xf32>) -> tensor<1x512x768xf32>
        %198 = "ttir.dot_general"(%197, %arg14) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<1x512x768xf32>, tensor<768x768xf32>) -> tensor<1x512x768xf32>
        %199 = "ttir.add"(%129, %198) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %200 = "ttir.sum"(%199) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x512x768xf32>) -> tensor<1x512xf32>
        %201 = "ttir.reshape"(%200) <{shape = [1 : i32, 512 : i32, 1 : i32]}> : (tensor<1x512xf32>) -> tensor<1x512x1xf32>
        %202 = "ttir.broadcast"(%201) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %203 = "ttir.reshape"(%7) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %204 = "ttir.broadcast"(%203) <{broadcast_dimensions = array<i64: 1, 512, 1>}> : (tensor<1x1x1xf32>) -> tensor<1x512x1xf32>
        %205 = "ttir.div"(%202, %204) : (tensor<1x512x1xf32>, tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %206 = "ttir.broadcast"(%205) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<1x512x1xf32>) -> tensor<1x512x768xf32>
        %207 = "ttir.subtract"(%199, %206) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %208 = "ttir.multiply"(%207, %207) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %209 = "ttir.sum"(%208) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x512x768xf32>) -> tensor<1x512xf32>
        %210 = "ttir.reshape"(%209) <{shape = [1 : i32, 512 : i32, 1 : i32]}> : (tensor<1x512xf32>) -> tensor<1x512x1xf32>
        %211 = "ttir.broadcast"(%210) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %212 = "ttir.reshape"(%7) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %213 = "ttir.broadcast"(%212) <{broadcast_dimensions = array<i64: 1, 512, 1>}> : (tensor<1x1x1xf32>) -> tensor<1x512x1xf32>
        %214 = "ttir.div"(%211, %213) : (tensor<1x512x1xf32>, tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %215 = "ttir.broadcast"(%205) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<1x512x1xf32>) -> tensor<1x512x768xf32>
        %216 = "ttir.subtract"(%199, %215) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %217 = "ttir.reshape"(%6) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %218 = "ttir.broadcast"(%217) <{broadcast_dimensions = array<i64: 1, 512, 1>}> : (tensor<1x1x1xf32>) -> tensor<1x512x1xf32>
        %219 = "ttir.add"(%214, %218) : (tensor<1x512x1xf32>, tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %220 = "ttir.sqrt"(%219) : (tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %221 = "ttir.broadcast"(%220) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<1x512x1xf32>) -> tensor<1x512x768xf32>
        %222 = "ttir.div"(%216, %221) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %223 = "ttir.reshape"(%arg19) <{shape = [1 : i32, 1 : i32, 768 : i32]}> : (tensor<768xf32>) -> tensor<1x1x768xf32>
        %224 = "ttir.broadcast"(%223) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x1x768xf32>
        %225 = "ttir.broadcast"(%224) <{broadcast_dimensions = array<i64: 1, 512, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x512x768xf32>
        %226 = "ttir.multiply"(%222, %225) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %227 = "ttir.reshape"(%arg20) <{shape = [1 : i32, 1 : i32, 768 : i32]}> : (tensor<768xf32>) -> tensor<1x1x768xf32>
        %228 = "ttir.broadcast"(%227) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x1x768xf32>
        %229 = "ttir.broadcast"(%228) <{broadcast_dimensions = array<i64: 1, 512, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x512x768xf32>
        %230 = "ttir.add"(%226, %229) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %231 = "ttir.dot_general"(%230, %arg17) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<1x512x768xf32>, tensor<768x3072xf32>) -> tensor<1x512x3072xf32>
        %232 = "ttir.multiply"(%231, %231) : (tensor<1x512x3072xf32>, tensor<1x512x3072xf32>) -> tensor<1x512x3072xf32>
        %233 = "ttir.multiply"(%232, %231) : (tensor<1x512x3072xf32>, tensor<1x512x3072xf32>) -> tensor<1x512x3072xf32>
        %234 = "ttir.reshape"(%3) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %235 = "ttir.broadcast"(%234) <{broadcast_dimensions = array<i64: 1, 512, 3072>}> : (tensor<1x1x1xf32>) -> tensor<1x512x3072xf32>
        %236 = "ttir.multiply"(%235, %233) : (tensor<1x512x3072xf32>, tensor<1x512x3072xf32>) -> tensor<1x512x3072xf32>
        %237 = "ttir.add"(%231, %236) : (tensor<1x512x3072xf32>, tensor<1x512x3072xf32>) -> tensor<1x512x3072xf32>
        %238 = "ttir.reshape"(%2) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %239 = "ttir.broadcast"(%238) <{broadcast_dimensions = array<i64: 1, 512, 3072>}> : (tensor<1x1x1xf32>) -> tensor<1x512x3072xf32>
        %240 = "ttir.multiply"(%239, %237) : (tensor<1x512x3072xf32>, tensor<1x512x3072xf32>) -> tensor<1x512x3072xf32>
        %241 = "ttir.tanh"(%240) : (tensor<1x512x3072xf32>) -> tensor<1x512x3072xf32>
        %242 = "ttir.reshape"(%1) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %243 = "ttir.broadcast"(%242) <{broadcast_dimensions = array<i64: 1, 512, 3072>}> : (tensor<1x1x1xf32>) -> tensor<1x512x3072xf32>
        %244 = "ttir.add"(%243, %241) : (tensor<1x512x3072xf32>, tensor<1x512x3072xf32>) -> tensor<1x512x3072xf32>
        %245 = "ttir.reshape"(%0) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %246 = "ttir.broadcast"(%245) <{broadcast_dimensions = array<i64: 1, 512, 3072>}> : (tensor<1x1x1xf32>) -> tensor<1x512x3072xf32>
        %247 = "ttir.multiply"(%246, %244) : (tensor<1x512x3072xf32>, tensor<1x512x3072xf32>) -> tensor<1x512x3072xf32>
        %248 = "ttir.multiply"(%231, %247) : (tensor<1x512x3072xf32>, tensor<1x512x3072xf32>) -> tensor<1x512x3072xf32>
        %249 = "ttir.dot_general"(%248, %arg18) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<1x512x3072xf32>, tensor<3072x768xf32>) -> tensor<1x512x768xf32>
        %250 = "ttir.add"(%199, %249) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %251 = "ttir.sum"(%250) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x512x768xf32>) -> tensor<1x512xf32>
        %252 = "ttir.reshape"(%251) <{shape = [1 : i32, 512 : i32, 1 : i32]}> : (tensor<1x512xf32>) -> tensor<1x512x1xf32>
        %253 = "ttir.broadcast"(%252) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %254 = "ttir.reshape"(%7) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %255 = "ttir.broadcast"(%254) <{broadcast_dimensions = array<i64: 1, 512, 1>}> : (tensor<1x1x1xf32>) -> tensor<1x512x1xf32>
        %256 = "ttir.div"(%253, %255) : (tensor<1x512x1xf32>, tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %257 = "ttir.broadcast"(%256) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<1x512x1xf32>) -> tensor<1x512x768xf32>
        %258 = "ttir.subtract"(%250, %257) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %259 = "ttir.multiply"(%258, %258) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %260 = "ttir.sum"(%259) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x512x768xf32>) -> tensor<1x512xf32>
        %261 = "ttir.reshape"(%260) <{shape = [1 : i32, 512 : i32, 1 : i32]}> : (tensor<1x512xf32>) -> tensor<1x512x1xf32>
        %262 = "ttir.broadcast"(%261) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %263 = "ttir.reshape"(%7) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %264 = "ttir.broadcast"(%263) <{broadcast_dimensions = array<i64: 1, 512, 1>}> : (tensor<1x1x1xf32>) -> tensor<1x512x1xf32>
        %265 = "ttir.div"(%262, %264) : (tensor<1x512x1xf32>, tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %266 = "ttir.broadcast"(%256) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<1x512x1xf32>) -> tensor<1x512x768xf32>
        %267 = "ttir.subtract"(%250, %266) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %268 = "ttir.reshape"(%6) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %269 = "ttir.broadcast"(%268) <{broadcast_dimensions = array<i64: 1, 512, 1>}> : (tensor<1x1x1xf32>) -> tensor<1x512x1xf32>
        %270 = "ttir.add"(%265, %269) : (tensor<1x512x1xf32>, tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %271 = "ttir.sqrt"(%270) : (tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %272 = "ttir.broadcast"(%271) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<1x512x1xf32>) -> tensor<1x512x768xf32>
        %273 = "ttir.div"(%267, %272) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %274 = "ttir.reshape"(%arg25) <{shape = [1 : i32, 1 : i32, 768 : i32]}> : (tensor<768xf32>) -> tensor<1x1x768xf32>
        %275 = "ttir.broadcast"(%274) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x1x768xf32>
        %276 = "ttir.broadcast"(%275) <{broadcast_dimensions = array<i64: 1, 512, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x512x768xf32>
        %277 = "ttir.multiply"(%273, %276) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %278 = "ttir.reshape"(%arg26) <{shape = [1 : i32, 1 : i32, 768 : i32]}> : (tensor<768xf32>) -> tensor<1x1x768xf32>
        %279 = "ttir.broadcast"(%278) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x1x768xf32>
        %280 = "ttir.broadcast"(%279) <{broadcast_dimensions = array<i64: 1, 512, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x512x768xf32>
        %281 = "ttir.add"(%277, %280) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %282 = "ttir.dot_general"(%281, %arg21) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<1x512x768xf32>, tensor<768x768xf32>) -> tensor<1x512x768xf32>
        %283 = "ttir.dot_general"(%281, %arg22) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<1x512x768xf32>, tensor<768x768xf32>) -> tensor<1x512x768xf32>
        %284 = "ttir.dot_general"(%281, %arg23) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<1x512x768xf32>, tensor<768x768xf32>) -> tensor<1x512x768xf32>
        %285 = "ttir.reshape"(%282) <{shape = [1 : i32, 512 : i32, 12 : i32, 64 : i32]}> : (tensor<1x512x768xf32>) -> tensor<1x512x12x64xf32>
        %286 = "ttir.permute"(%285) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x512x12x64xf32>) -> tensor<1x12x512x64xf32>
        %287 = "ttir.reshape"(%283) <{shape = [1 : i32, 512 : i32, 12 : i32, 64 : i32]}> : (tensor<1x512x768xf32>) -> tensor<1x512x12x64xf32>
        %288 = "ttir.permute"(%287) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x512x12x64xf32>) -> tensor<1x12x512x64xf32>
        %289 = "ttir.reshape"(%284) <{shape = [1 : i32, 512 : i32, 12 : i32, 64 : i32]}> : (tensor<1x512x768xf32>) -> tensor<1x512x12x64xf32>
        %290 = "ttir.permute"(%289) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x512x12x64xf32>) -> tensor<1x12x512x64xf32>
        %291 = "ttir.permute"(%288) <{permutation = array<i64: 0, 1, 3, 2>}> : (tensor<1x12x512x64xf32>) -> tensor<1x12x64x512xf32>
        %292 = "ttir.reshape"(%286) <{shape = [12 : i32, 512 : i32, 64 : i32]}> : (tensor<1x12x512x64xf32>) -> tensor<12x512x64xf32>
        %293 = "ttir.dot_general"(%292, %291) <{batch_dims_lhs = array<i64: 0>, batch_dims_rhs = array<i64: 1>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 2>}> : (tensor<12x512x64xf32>, tensor<1x12x64x512xf32>) -> tensor<12x512x1x512xf32>
        %294 = "ttir.permute"(%293) <{permutation = array<i64: 2, 0, 1, 3>}> : (tensor<12x512x1x512xf32>) -> tensor<1x12x512x512xf32>
        %295 = "ttir.sqrt"(%5) : (tensor<f32>) -> tensor<f32>
        %296 = "ttir.typecast"(%295) <{conservative_folding = false}> : (tensor<f32>) -> tensor<f32>
        %297 = "ttir.reshape"(%296) <{shape = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1x1xf32>
        %298 = "ttir.broadcast"(%297) <{broadcast_dimensions = array<i64: 1, 12, 512, 512>}> : (tensor<1x1x1x1xf32>) -> tensor<1x12x512x512xf32>
        %299 = "ttir.div"(%294, %298) : (tensor<1x12x512x512xf32>, tensor<1x12x512x512xf32>) -> tensor<1x12x512x512xf32>
        %300 = "ttir.max"(%299) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<1x12x512x512xf32>) -> tensor<1x12x512xf32>
        %301 = "ttir.reshape"(%4) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %302 = "ttir.broadcast"(%301) <{broadcast_dimensions = array<i64: 1, 12, 512>}> : (tensor<1x1x1xf32>) -> tensor<1x12x512xf32>
        %303 = "ttir.maximum"(%302, %300) : (tensor<1x12x512xf32>, tensor<1x12x512xf32>) -> tensor<1x12x512xf32>
        %304 = "ttir.reshape"(%303) <{shape = [1 : i32, 12 : i32, 512 : i32, 1 : i32]}> : (tensor<1x12x512xf32>) -> tensor<1x12x512x1xf32>
        %305 = "ttir.broadcast"(%304) <{broadcast_dimensions = array<i64: 1, 1, 1, 1>}> : (tensor<1x12x512x1xf32>) -> tensor<1x12x512x1xf32>
        %306 = "ttir.broadcast"(%305) <{broadcast_dimensions = array<i64: 1, 1, 1, 512>}> : (tensor<1x12x512x1xf32>) -> tensor<1x12x512x512xf32>
        %307 = "ttir.subtract"(%299, %306) : (tensor<1x12x512x512xf32>, tensor<1x12x512x512xf32>) -> tensor<1x12x512x512xf32>
        %308 = "ttir.exp"(%307) : (tensor<1x12x512x512xf32>) -> tensor<1x12x512x512xf32>
        %309 = "ttir.sum"(%308) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<1x12x512x512xf32>) -> tensor<1x12x512xf32>
        %310 = "ttir.reshape"(%309) <{shape = [1 : i32, 12 : i32, 512 : i32, 1 : i32]}> : (tensor<1x12x512xf32>) -> tensor<1x12x512x1xf32>
        %311 = "ttir.broadcast"(%310) <{broadcast_dimensions = array<i64: 1, 1, 1, 1>}> : (tensor<1x12x512x1xf32>) -> tensor<1x12x512x1xf32>
        %312 = "ttir.broadcast"(%311) <{broadcast_dimensions = array<i64: 1, 1, 1, 512>}> : (tensor<1x12x512x1xf32>) -> tensor<1x12x512x512xf32>
        %313 = "ttir.div"(%308, %312) : (tensor<1x12x512x512xf32>, tensor<1x12x512x512xf32>) -> tensor<1x12x512x512xf32>
        %314 = "ttir.reshape"(%313) <{shape = [12 : i32, 512 : i32, 512 : i32]}> : (tensor<1x12x512x512xf32>) -> tensor<12x512x512xf32>
        %315 = "ttir.dot_general"(%314, %290) <{batch_dims_lhs = array<i64: 0>, batch_dims_rhs = array<i64: 1>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 2>}> : (tensor<12x512x512xf32>, tensor<1x12x512x64xf32>) -> tensor<12x512x1x64xf32>
        %316 = "ttir.permute"(%315) <{permutation = array<i64: 2, 0, 1, 3>}> : (tensor<12x512x1x64xf32>) -> tensor<1x12x512x64xf32>
        %317 = "ttir.permute"(%316) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x12x512x64xf32>) -> tensor<1x512x12x64xf32>
        %318 = "ttir.reshape"(%317) <{shape = [1 : i32, 512 : i32, 768 : i32]}> : (tensor<1x512x12x64xf32>) -> tensor<1x512x768xf32>
        %319 = "ttir.dot_general"(%318, %arg24) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<1x512x768xf32>, tensor<768x768xf32>) -> tensor<1x512x768xf32>
        %320 = "ttir.add"(%250, %319) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %321 = "ttir.sum"(%320) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x512x768xf32>) -> tensor<1x512xf32>
        %322 = "ttir.reshape"(%321) <{shape = [1 : i32, 512 : i32, 1 : i32]}> : (tensor<1x512xf32>) -> tensor<1x512x1xf32>
        %323 = "ttir.broadcast"(%322) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %324 = "ttir.reshape"(%7) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %325 = "ttir.broadcast"(%324) <{broadcast_dimensions = array<i64: 1, 512, 1>}> : (tensor<1x1x1xf32>) -> tensor<1x512x1xf32>
        %326 = "ttir.div"(%323, %325) : (tensor<1x512x1xf32>, tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %327 = "ttir.broadcast"(%326) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<1x512x1xf32>) -> tensor<1x512x768xf32>
        %328 = "ttir.subtract"(%320, %327) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %329 = "ttir.multiply"(%328, %328) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %330 = "ttir.sum"(%329) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x512x768xf32>) -> tensor<1x512xf32>
        %331 = "ttir.reshape"(%330) <{shape = [1 : i32, 512 : i32, 1 : i32]}> : (tensor<1x512xf32>) -> tensor<1x512x1xf32>
        %332 = "ttir.broadcast"(%331) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %333 = "ttir.reshape"(%7) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %334 = "ttir.broadcast"(%333) <{broadcast_dimensions = array<i64: 1, 512, 1>}> : (tensor<1x1x1xf32>) -> tensor<1x512x1xf32>
        %335 = "ttir.div"(%332, %334) : (tensor<1x512x1xf32>, tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %336 = "ttir.broadcast"(%326) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<1x512x1xf32>) -> tensor<1x512x768xf32>
        %337 = "ttir.subtract"(%320, %336) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %338 = "ttir.reshape"(%6) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %339 = "ttir.broadcast"(%338) <{broadcast_dimensions = array<i64: 1, 512, 1>}> : (tensor<1x1x1xf32>) -> tensor<1x512x1xf32>
        %340 = "ttir.add"(%335, %339) : (tensor<1x512x1xf32>, tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %341 = "ttir.sqrt"(%340) : (tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %342 = "ttir.broadcast"(%341) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<1x512x1xf32>) -> tensor<1x512x768xf32>
        %343 = "ttir.div"(%337, %342) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %344 = "ttir.reshape"(%arg29) <{shape = [1 : i32, 1 : i32, 768 : i32]}> : (tensor<768xf32>) -> tensor<1x1x768xf32>
        %345 = "ttir.broadcast"(%344) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x1x768xf32>
        %346 = "ttir.broadcast"(%345) <{broadcast_dimensions = array<i64: 1, 512, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x512x768xf32>
        %347 = "ttir.multiply"(%343, %346) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %348 = "ttir.reshape"(%arg30) <{shape = [1 : i32, 1 : i32, 768 : i32]}> : (tensor<768xf32>) -> tensor<1x1x768xf32>
        %349 = "ttir.broadcast"(%348) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x1x768xf32>
        %350 = "ttir.broadcast"(%349) <{broadcast_dimensions = array<i64: 1, 512, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x512x768xf32>
        %351 = "ttir.add"(%347, %350) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %352 = "ttir.dot_general"(%351, %arg27) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<1x512x768xf32>, tensor<768x3072xf32>) -> tensor<1x512x3072xf32>
        %353 = "ttir.multiply"(%352, %352) : (tensor<1x512x3072xf32>, tensor<1x512x3072xf32>) -> tensor<1x512x3072xf32>
        %354 = "ttir.multiply"(%353, %352) : (tensor<1x512x3072xf32>, tensor<1x512x3072xf32>) -> tensor<1x512x3072xf32>
        %355 = "ttir.reshape"(%3) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %356 = "ttir.broadcast"(%355) <{broadcast_dimensions = array<i64: 1, 512, 3072>}> : (tensor<1x1x1xf32>) -> tensor<1x512x3072xf32>
        %357 = "ttir.multiply"(%356, %354) : (tensor<1x512x3072xf32>, tensor<1x512x3072xf32>) -> tensor<1x512x3072xf32>
        %358 = "ttir.add"(%352, %357) : (tensor<1x512x3072xf32>, tensor<1x512x3072xf32>) -> tensor<1x512x3072xf32>
        %359 = "ttir.reshape"(%2) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %360 = "ttir.broadcast"(%359) <{broadcast_dimensions = array<i64: 1, 512, 3072>}> : (tensor<1x1x1xf32>) -> tensor<1x512x3072xf32>
        %361 = "ttir.multiply"(%360, %358) : (tensor<1x512x3072xf32>, tensor<1x512x3072xf32>) -> tensor<1x512x3072xf32>
        %362 = "ttir.tanh"(%361) : (tensor<1x512x3072xf32>) -> tensor<1x512x3072xf32>
        %363 = "ttir.reshape"(%1) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %364 = "ttir.broadcast"(%363) <{broadcast_dimensions = array<i64: 1, 512, 3072>}> : (tensor<1x1x1xf32>) -> tensor<1x512x3072xf32>
        %365 = "ttir.add"(%364, %362) : (tensor<1x512x3072xf32>, tensor<1x512x3072xf32>) -> tensor<1x512x3072xf32>
        %366 = "ttir.reshape"(%0) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %367 = "ttir.broadcast"(%366) <{broadcast_dimensions = array<i64: 1, 512, 3072>}> : (tensor<1x1x1xf32>) -> tensor<1x512x3072xf32>
        %368 = "ttir.multiply"(%367, %365) : (tensor<1x512x3072xf32>, tensor<1x512x3072xf32>) -> tensor<1x512x3072xf32>
        %369 = "ttir.multiply"(%352, %368) : (tensor<1x512x3072xf32>, tensor<1x512x3072xf32>) -> tensor<1x512x3072xf32>
        %370 = "ttir.dot_general"(%369, %arg28) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<1x512x3072xf32>, tensor<3072x768xf32>) -> tensor<1x512x768xf32>
        %371 = "ttir.add"(%320, %370) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %372 = "ttir.sum"(%371) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x512x768xf32>) -> tensor<1x512xf32>
        %373 = "ttir.reshape"(%372) <{shape = [1 : i32, 512 : i32, 1 : i32]}> : (tensor<1x512xf32>) -> tensor<1x512x1xf32>
        %374 = "ttir.broadcast"(%373) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %375 = "ttir.reshape"(%7) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %376 = "ttir.broadcast"(%375) <{broadcast_dimensions = array<i64: 1, 512, 1>}> : (tensor<1x1x1xf32>) -> tensor<1x512x1xf32>
        %377 = "ttir.div"(%374, %376) : (tensor<1x512x1xf32>, tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %378 = "ttir.broadcast"(%377) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<1x512x1xf32>) -> tensor<1x512x768xf32>
        %379 = "ttir.subtract"(%371, %378) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %380 = "ttir.multiply"(%379, %379) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %381 = "ttir.sum"(%380) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x512x768xf32>) -> tensor<1x512xf32>
        %382 = "ttir.reshape"(%381) <{shape = [1 : i32, 512 : i32, 1 : i32]}> : (tensor<1x512xf32>) -> tensor<1x512x1xf32>
        %383 = "ttir.broadcast"(%382) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %384 = "ttir.reshape"(%7) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %385 = "ttir.broadcast"(%384) <{broadcast_dimensions = array<i64: 1, 512, 1>}> : (tensor<1x1x1xf32>) -> tensor<1x512x1xf32>
        %386 = "ttir.div"(%383, %385) : (tensor<1x512x1xf32>, tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %387 = "ttir.broadcast"(%377) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<1x512x1xf32>) -> tensor<1x512x768xf32>
        %388 = "ttir.subtract"(%371, %387) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %389 = "ttir.reshape"(%6) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %390 = "ttir.broadcast"(%389) <{broadcast_dimensions = array<i64: 1, 512, 1>}> : (tensor<1x1x1xf32>) -> tensor<1x512x1xf32>
        %391 = "ttir.add"(%386, %390) : (tensor<1x512x1xf32>, tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %392 = "ttir.sqrt"(%391) : (tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %393 = "ttir.broadcast"(%392) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<1x512x1xf32>) -> tensor<1x512x768xf32>
        %394 = "ttir.div"(%388, %393) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %395 = "ttir.reshape"(%arg35) <{shape = [1 : i32, 1 : i32, 768 : i32]}> : (tensor<768xf32>) -> tensor<1x1x768xf32>
        %396 = "ttir.broadcast"(%395) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x1x768xf32>
        %397 = "ttir.broadcast"(%396) <{broadcast_dimensions = array<i64: 1, 512, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x512x768xf32>
        %398 = "ttir.multiply"(%394, %397) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %399 = "ttir.reshape"(%arg36) <{shape = [1 : i32, 1 : i32, 768 : i32]}> : (tensor<768xf32>) -> tensor<1x1x768xf32>
        %400 = "ttir.broadcast"(%399) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x1x768xf32>
        %401 = "ttir.broadcast"(%400) <{broadcast_dimensions = array<i64: 1, 512, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x512x768xf32>
        %402 = "ttir.add"(%398, %401) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %403 = "ttir.dot_general"(%402, %arg31) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<1x512x768xf32>, tensor<768x768xf32>) -> tensor<1x512x768xf32>
        %404 = "ttir.dot_general"(%402, %arg32) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<1x512x768xf32>, tensor<768x768xf32>) -> tensor<1x512x768xf32>
        %405 = "ttir.dot_general"(%402, %arg33) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<1x512x768xf32>, tensor<768x768xf32>) -> tensor<1x512x768xf32>
        %406 = "ttir.reshape"(%403) <{shape = [1 : i32, 512 : i32, 12 : i32, 64 : i32]}> : (tensor<1x512x768xf32>) -> tensor<1x512x12x64xf32>
        %407 = "ttir.permute"(%406) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x512x12x64xf32>) -> tensor<1x12x512x64xf32>
        %408 = "ttir.reshape"(%404) <{shape = [1 : i32, 512 : i32, 12 : i32, 64 : i32]}> : (tensor<1x512x768xf32>) -> tensor<1x512x12x64xf32>
        %409 = "ttir.permute"(%408) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x512x12x64xf32>) -> tensor<1x12x512x64xf32>
        %410 = "ttir.reshape"(%405) <{shape = [1 : i32, 512 : i32, 12 : i32, 64 : i32]}> : (tensor<1x512x768xf32>) -> tensor<1x512x12x64xf32>
        %411 = "ttir.permute"(%410) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x512x12x64xf32>) -> tensor<1x12x512x64xf32>
        %412 = "ttir.permute"(%409) <{permutation = array<i64: 0, 1, 3, 2>}> : (tensor<1x12x512x64xf32>) -> tensor<1x12x64x512xf32>
        %413 = "ttir.reshape"(%407) <{shape = [12 : i32, 512 : i32, 64 : i32]}> : (tensor<1x12x512x64xf32>) -> tensor<12x512x64xf32>
        %414 = "ttir.dot_general"(%413, %412) <{batch_dims_lhs = array<i64: 0>, batch_dims_rhs = array<i64: 1>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 2>}> : (tensor<12x512x64xf32>, tensor<1x12x64x512xf32>) -> tensor<12x512x1x512xf32>
        %415 = "ttir.permute"(%414) <{permutation = array<i64: 2, 0, 1, 3>}> : (tensor<12x512x1x512xf32>) -> tensor<1x12x512x512xf32>
        %416 = "ttir.sqrt"(%5) : (tensor<f32>) -> tensor<f32>
        %417 = "ttir.typecast"(%416) <{conservative_folding = false}> : (tensor<f32>) -> tensor<f32>
        %418 = "ttir.reshape"(%417) <{shape = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1x1xf32>
        %419 = "ttir.broadcast"(%418) <{broadcast_dimensions = array<i64: 1, 12, 512, 512>}> : (tensor<1x1x1x1xf32>) -> tensor<1x12x512x512xf32>
        %420 = "ttir.div"(%415, %419) : (tensor<1x12x512x512xf32>, tensor<1x12x512x512xf32>) -> tensor<1x12x512x512xf32>
        %421 = "ttir.max"(%420) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<1x12x512x512xf32>) -> tensor<1x12x512xf32>
        %422 = "ttir.reshape"(%4) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %423 = "ttir.broadcast"(%422) <{broadcast_dimensions = array<i64: 1, 12, 512>}> : (tensor<1x1x1xf32>) -> tensor<1x12x512xf32>
        %424 = "ttir.maximum"(%423, %421) : (tensor<1x12x512xf32>, tensor<1x12x512xf32>) -> tensor<1x12x512xf32>
        %425 = "ttir.reshape"(%424) <{shape = [1 : i32, 12 : i32, 512 : i32, 1 : i32]}> : (tensor<1x12x512xf32>) -> tensor<1x12x512x1xf32>
        %426 = "ttir.broadcast"(%425) <{broadcast_dimensions = array<i64: 1, 1, 1, 1>}> : (tensor<1x12x512x1xf32>) -> tensor<1x12x512x1xf32>
        %427 = "ttir.broadcast"(%426) <{broadcast_dimensions = array<i64: 1, 1, 1, 512>}> : (tensor<1x12x512x1xf32>) -> tensor<1x12x512x512xf32>
        %428 = "ttir.subtract"(%420, %427) : (tensor<1x12x512x512xf32>, tensor<1x12x512x512xf32>) -> tensor<1x12x512x512xf32>
        %429 = "ttir.exp"(%428) : (tensor<1x12x512x512xf32>) -> tensor<1x12x512x512xf32>
        %430 = "ttir.sum"(%429) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<1x12x512x512xf32>) -> tensor<1x12x512xf32>
        %431 = "ttir.reshape"(%430) <{shape = [1 : i32, 12 : i32, 512 : i32, 1 : i32]}> : (tensor<1x12x512xf32>) -> tensor<1x12x512x1xf32>
        %432 = "ttir.broadcast"(%431) <{broadcast_dimensions = array<i64: 1, 1, 1, 1>}> : (tensor<1x12x512x1xf32>) -> tensor<1x12x512x1xf32>
        %433 = "ttir.broadcast"(%432) <{broadcast_dimensions = array<i64: 1, 1, 1, 512>}> : (tensor<1x12x512x1xf32>) -> tensor<1x12x512x512xf32>
        %434 = "ttir.div"(%429, %433) : (tensor<1x12x512x512xf32>, tensor<1x12x512x512xf32>) -> tensor<1x12x512x512xf32>
        %435 = "ttir.reshape"(%434) <{shape = [12 : i32, 512 : i32, 512 : i32]}> : (tensor<1x12x512x512xf32>) -> tensor<12x512x512xf32>
        %436 = "ttir.dot_general"(%435, %411) <{batch_dims_lhs = array<i64: 0>, batch_dims_rhs = array<i64: 1>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 2>}> : (tensor<12x512x512xf32>, tensor<1x12x512x64xf32>) -> tensor<12x512x1x64xf32>
        %437 = "ttir.permute"(%436) <{permutation = array<i64: 2, 0, 1, 3>}> : (tensor<12x512x1x64xf32>) -> tensor<1x12x512x64xf32>
        %438 = "ttir.permute"(%437) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x12x512x64xf32>) -> tensor<1x512x12x64xf32>
        %439 = "ttir.reshape"(%438) <{shape = [1 : i32, 512 : i32, 768 : i32]}> : (tensor<1x512x12x64xf32>) -> tensor<1x512x768xf32>
        %440 = "ttir.dot_general"(%439, %arg34) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<1x512x768xf32>, tensor<768x768xf32>) -> tensor<1x512x768xf32>
        %441 = "ttir.add"(%371, %440) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %442 = "ttir.sum"(%441) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x512x768xf32>) -> tensor<1x512xf32>
        %443 = "ttir.reshape"(%442) <{shape = [1 : i32, 512 : i32, 1 : i32]}> : (tensor<1x512xf32>) -> tensor<1x512x1xf32>
        %444 = "ttir.broadcast"(%443) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %445 = "ttir.reshape"(%7) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %446 = "ttir.broadcast"(%445) <{broadcast_dimensions = array<i64: 1, 512, 1>}> : (tensor<1x1x1xf32>) -> tensor<1x512x1xf32>
        %447 = "ttir.div"(%444, %446) : (tensor<1x512x1xf32>, tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %448 = "ttir.broadcast"(%447) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<1x512x1xf32>) -> tensor<1x512x768xf32>
        %449 = "ttir.subtract"(%441, %448) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %450 = "ttir.multiply"(%449, %449) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %451 = "ttir.sum"(%450) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x512x768xf32>) -> tensor<1x512xf32>
        %452 = "ttir.reshape"(%451) <{shape = [1 : i32, 512 : i32, 1 : i32]}> : (tensor<1x512xf32>) -> tensor<1x512x1xf32>
        %453 = "ttir.broadcast"(%452) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %454 = "ttir.reshape"(%7) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %455 = "ttir.broadcast"(%454) <{broadcast_dimensions = array<i64: 1, 512, 1>}> : (tensor<1x1x1xf32>) -> tensor<1x512x1xf32>
        %456 = "ttir.div"(%453, %455) : (tensor<1x512x1xf32>, tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %457 = "ttir.broadcast"(%447) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<1x512x1xf32>) -> tensor<1x512x768xf32>
        %458 = "ttir.subtract"(%441, %457) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %459 = "ttir.reshape"(%6) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %460 = "ttir.broadcast"(%459) <{broadcast_dimensions = array<i64: 1, 512, 1>}> : (tensor<1x1x1xf32>) -> tensor<1x512x1xf32>
        %461 = "ttir.add"(%456, %460) : (tensor<1x512x1xf32>, tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %462 = "ttir.sqrt"(%461) : (tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %463 = "ttir.broadcast"(%462) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<1x512x1xf32>) -> tensor<1x512x768xf32>
        %464 = "ttir.div"(%458, %463) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %465 = "ttir.reshape"(%arg39) <{shape = [1 : i32, 1 : i32, 768 : i32]}> : (tensor<768xf32>) -> tensor<1x1x768xf32>
        %466 = "ttir.broadcast"(%465) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x1x768xf32>
        %467 = "ttir.broadcast"(%466) <{broadcast_dimensions = array<i64: 1, 512, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x512x768xf32>
        %468 = "ttir.multiply"(%464, %467) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %469 = "ttir.reshape"(%arg40) <{shape = [1 : i32, 1 : i32, 768 : i32]}> : (tensor<768xf32>) -> tensor<1x1x768xf32>
        %470 = "ttir.broadcast"(%469) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x1x768xf32>
        %471 = "ttir.broadcast"(%470) <{broadcast_dimensions = array<i64: 1, 512, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x512x768xf32>
        %472 = "ttir.add"(%468, %471) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %473 = "ttir.dot_general"(%472, %arg37) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<1x512x768xf32>, tensor<768x3072xf32>) -> tensor<1x512x3072xf32>
        %474 = "ttir.multiply"(%473, %473) : (tensor<1x512x3072xf32>, tensor<1x512x3072xf32>) -> tensor<1x512x3072xf32>
        %475 = "ttir.multiply"(%474, %473) : (tensor<1x512x3072xf32>, tensor<1x512x3072xf32>) -> tensor<1x512x3072xf32>
        %476 = "ttir.reshape"(%3) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %477 = "ttir.broadcast"(%476) <{broadcast_dimensions = array<i64: 1, 512, 3072>}> : (tensor<1x1x1xf32>) -> tensor<1x512x3072xf32>
        %478 = "ttir.multiply"(%477, %475) : (tensor<1x512x3072xf32>, tensor<1x512x3072xf32>) -> tensor<1x512x3072xf32>
        %479 = "ttir.add"(%473, %478) : (tensor<1x512x3072xf32>, tensor<1x512x3072xf32>) -> tensor<1x512x3072xf32>
        %480 = "ttir.reshape"(%2) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %481 = "ttir.broadcast"(%480) <{broadcast_dimensions = array<i64: 1, 512, 3072>}> : (tensor<1x1x1xf32>) -> tensor<1x512x3072xf32>
        %482 = "ttir.multiply"(%481, %479) : (tensor<1x512x3072xf32>, tensor<1x512x3072xf32>) -> tensor<1x512x3072xf32>
        %483 = "ttir.tanh"(%482) : (tensor<1x512x3072xf32>) -> tensor<1x512x3072xf32>
        %484 = "ttir.reshape"(%1) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %485 = "ttir.broadcast"(%484) <{broadcast_dimensions = array<i64: 1, 512, 3072>}> : (tensor<1x1x1xf32>) -> tensor<1x512x3072xf32>
        %486 = "ttir.add"(%485, %483) : (tensor<1x512x3072xf32>, tensor<1x512x3072xf32>) -> tensor<1x512x3072xf32>
        %487 = "ttir.reshape"(%0) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %488 = "ttir.broadcast"(%487) <{broadcast_dimensions = array<i64: 1, 512, 3072>}> : (tensor<1x1x1xf32>) -> tensor<1x512x3072xf32>
        %489 = "ttir.multiply"(%488, %486) : (tensor<1x512x3072xf32>, tensor<1x512x3072xf32>) -> tensor<1x512x3072xf32>
        %490 = "ttir.multiply"(%473, %489) : (tensor<1x512x3072xf32>, tensor<1x512x3072xf32>) -> tensor<1x512x3072xf32>
        %491 = "ttir.dot_general"(%490, %arg38) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<1x512x3072xf32>, tensor<3072x768xf32>) -> tensor<1x512x768xf32>
        %492 = "ttir.add"(%441, %491) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %493 = "ttir.sum"(%492) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x512x768xf32>) -> tensor<1x512xf32>
        %494 = "ttir.reshape"(%493) <{shape = [1 : i32, 512 : i32, 1 : i32]}> : (tensor<1x512xf32>) -> tensor<1x512x1xf32>
        %495 = "ttir.broadcast"(%494) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %496 = "ttir.reshape"(%7) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %497 = "ttir.broadcast"(%496) <{broadcast_dimensions = array<i64: 1, 512, 1>}> : (tensor<1x1x1xf32>) -> tensor<1x512x1xf32>
        %498 = "ttir.div"(%495, %497) : (tensor<1x512x1xf32>, tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %499 = "ttir.broadcast"(%498) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<1x512x1xf32>) -> tensor<1x512x768xf32>
        %500 = "ttir.subtract"(%492, %499) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %501 = "ttir.multiply"(%500, %500) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %502 = "ttir.sum"(%501) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x512x768xf32>) -> tensor<1x512xf32>
        %503 = "ttir.reshape"(%502) <{shape = [1 : i32, 512 : i32, 1 : i32]}> : (tensor<1x512xf32>) -> tensor<1x512x1xf32>
        %504 = "ttir.broadcast"(%503) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %505 = "ttir.reshape"(%7) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %506 = "ttir.broadcast"(%505) <{broadcast_dimensions = array<i64: 1, 512, 1>}> : (tensor<1x1x1xf32>) -> tensor<1x512x1xf32>
        %507 = "ttir.div"(%504, %506) : (tensor<1x512x1xf32>, tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %508 = "ttir.broadcast"(%498) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<1x512x1xf32>) -> tensor<1x512x768xf32>
        %509 = "ttir.subtract"(%492, %508) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %510 = "ttir.reshape"(%6) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %511 = "ttir.broadcast"(%510) <{broadcast_dimensions = array<i64: 1, 512, 1>}> : (tensor<1x1x1xf32>) -> tensor<1x512x1xf32>
        %512 = "ttir.add"(%507, %511) : (tensor<1x512x1xf32>, tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %513 = "ttir.sqrt"(%512) : (tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %514 = "ttir.broadcast"(%513) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<1x512x1xf32>) -> tensor<1x512x768xf32>
        %515 = "ttir.div"(%509, %514) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %516 = "ttir.reshape"(%arg45) <{shape = [1 : i32, 1 : i32, 768 : i32]}> : (tensor<768xf32>) -> tensor<1x1x768xf32>
        %517 = "ttir.broadcast"(%516) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x1x768xf32>
        %518 = "ttir.broadcast"(%517) <{broadcast_dimensions = array<i64: 1, 512, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x512x768xf32>
        %519 = "ttir.multiply"(%515, %518) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %520 = "ttir.reshape"(%arg46) <{shape = [1 : i32, 1 : i32, 768 : i32]}> : (tensor<768xf32>) -> tensor<1x1x768xf32>
        %521 = "ttir.broadcast"(%520) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x1x768xf32>
        %522 = "ttir.broadcast"(%521) <{broadcast_dimensions = array<i64: 1, 512, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x512x768xf32>
        %523 = "ttir.add"(%519, %522) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %524 = "ttir.dot_general"(%523, %arg41) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<1x512x768xf32>, tensor<768x768xf32>) -> tensor<1x512x768xf32>
        %525 = "ttir.dot_general"(%523, %arg42) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<1x512x768xf32>, tensor<768x768xf32>) -> tensor<1x512x768xf32>
        %526 = "ttir.dot_general"(%523, %arg43) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<1x512x768xf32>, tensor<768x768xf32>) -> tensor<1x512x768xf32>
        %527 = "ttir.reshape"(%524) <{shape = [1 : i32, 512 : i32, 12 : i32, 64 : i32]}> : (tensor<1x512x768xf32>) -> tensor<1x512x12x64xf32>
        %528 = "ttir.permute"(%527) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x512x12x64xf32>) -> tensor<1x12x512x64xf32>
        %529 = "ttir.reshape"(%525) <{shape = [1 : i32, 512 : i32, 12 : i32, 64 : i32]}> : (tensor<1x512x768xf32>) -> tensor<1x512x12x64xf32>
        %530 = "ttir.permute"(%529) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x512x12x64xf32>) -> tensor<1x12x512x64xf32>
        %531 = "ttir.reshape"(%526) <{shape = [1 : i32, 512 : i32, 12 : i32, 64 : i32]}> : (tensor<1x512x768xf32>) -> tensor<1x512x12x64xf32>
        %532 = "ttir.permute"(%531) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x512x12x64xf32>) -> tensor<1x12x512x64xf32>
        %533 = "ttir.permute"(%530) <{permutation = array<i64: 0, 1, 3, 2>}> : (tensor<1x12x512x64xf32>) -> tensor<1x12x64x512xf32>
        %534 = "ttir.reshape"(%528) <{shape = [12 : i32, 512 : i32, 64 : i32]}> : (tensor<1x12x512x64xf32>) -> tensor<12x512x64xf32>
        %535 = "ttir.dot_general"(%534, %533) <{batch_dims_lhs = array<i64: 0>, batch_dims_rhs = array<i64: 1>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 2>}> : (tensor<12x512x64xf32>, tensor<1x12x64x512xf32>) -> tensor<12x512x1x512xf32>
        %536 = "ttir.permute"(%535) <{permutation = array<i64: 2, 0, 1, 3>}> : (tensor<12x512x1x512xf32>) -> tensor<1x12x512x512xf32>
        %537 = "ttir.sqrt"(%5) : (tensor<f32>) -> tensor<f32>
        %538 = "ttir.typecast"(%537) <{conservative_folding = false}> : (tensor<f32>) -> tensor<f32>
        %539 = "ttir.reshape"(%538) <{shape = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1x1xf32>
        %540 = "ttir.broadcast"(%539) <{broadcast_dimensions = array<i64: 1, 12, 512, 512>}> : (tensor<1x1x1x1xf32>) -> tensor<1x12x512x512xf32>
        %541 = "ttir.div"(%536, %540) : (tensor<1x12x512x512xf32>, tensor<1x12x512x512xf32>) -> tensor<1x12x512x512xf32>
        %542 = "ttir.max"(%541) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<1x12x512x512xf32>) -> tensor<1x12x512xf32>
        %543 = "ttir.reshape"(%4) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %544 = "ttir.broadcast"(%543) <{broadcast_dimensions = array<i64: 1, 12, 512>}> : (tensor<1x1x1xf32>) -> tensor<1x12x512xf32>
        %545 = "ttir.maximum"(%544, %542) : (tensor<1x12x512xf32>, tensor<1x12x512xf32>) -> tensor<1x12x512xf32>
        %546 = "ttir.reshape"(%545) <{shape = [1 : i32, 12 : i32, 512 : i32, 1 : i32]}> : (tensor<1x12x512xf32>) -> tensor<1x12x512x1xf32>
        %547 = "ttir.broadcast"(%546) <{broadcast_dimensions = array<i64: 1, 1, 1, 1>}> : (tensor<1x12x512x1xf32>) -> tensor<1x12x512x1xf32>
        %548 = "ttir.broadcast"(%547) <{broadcast_dimensions = array<i64: 1, 1, 1, 512>}> : (tensor<1x12x512x1xf32>) -> tensor<1x12x512x512xf32>
        %549 = "ttir.subtract"(%541, %548) : (tensor<1x12x512x512xf32>, tensor<1x12x512x512xf32>) -> tensor<1x12x512x512xf32>
        %550 = "ttir.exp"(%549) : (tensor<1x12x512x512xf32>) -> tensor<1x12x512x512xf32>
        %551 = "ttir.sum"(%550) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<1x12x512x512xf32>) -> tensor<1x12x512xf32>
        %552 = "ttir.reshape"(%551) <{shape = [1 : i32, 12 : i32, 512 : i32, 1 : i32]}> : (tensor<1x12x512xf32>) -> tensor<1x12x512x1xf32>
        %553 = "ttir.broadcast"(%552) <{broadcast_dimensions = array<i64: 1, 1, 1, 1>}> : (tensor<1x12x512x1xf32>) -> tensor<1x12x512x1xf32>
        %554 = "ttir.broadcast"(%553) <{broadcast_dimensions = array<i64: 1, 1, 1, 512>}> : (tensor<1x12x512x1xf32>) -> tensor<1x12x512x512xf32>
        %555 = "ttir.div"(%550, %554) : (tensor<1x12x512x512xf32>, tensor<1x12x512x512xf32>) -> tensor<1x12x512x512xf32>
        %556 = "ttir.reshape"(%555) <{shape = [12 : i32, 512 : i32, 512 : i32]}> : (tensor<1x12x512x512xf32>) -> tensor<12x512x512xf32>
        %557 = "ttir.dot_general"(%556, %532) <{batch_dims_lhs = array<i64: 0>, batch_dims_rhs = array<i64: 1>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 2>}> : (tensor<12x512x512xf32>, tensor<1x12x512x64xf32>) -> tensor<12x512x1x64xf32>
        %558 = "ttir.permute"(%557) <{permutation = array<i64: 2, 0, 1, 3>}> : (tensor<12x512x1x64xf32>) -> tensor<1x12x512x64xf32>
        %559 = "ttir.permute"(%558) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x12x512x64xf32>) -> tensor<1x512x12x64xf32>
        %560 = "ttir.reshape"(%559) <{shape = [1 : i32, 512 : i32, 768 : i32]}> : (tensor<1x512x12x64xf32>) -> tensor<1x512x768xf32>
        %561 = "ttir.dot_general"(%560, %arg44) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<1x512x768xf32>, tensor<768x768xf32>) -> tensor<1x512x768xf32>
        %562 = "ttir.add"(%492, %561) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %563 = "ttir.sum"(%562) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x512x768xf32>) -> tensor<1x512xf32>
        %564 = "ttir.reshape"(%563) <{shape = [1 : i32, 512 : i32, 1 : i32]}> : (tensor<1x512xf32>) -> tensor<1x512x1xf32>
        %565 = "ttir.broadcast"(%564) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %566 = "ttir.reshape"(%7) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %567 = "ttir.broadcast"(%566) <{broadcast_dimensions = array<i64: 1, 512, 1>}> : (tensor<1x1x1xf32>) -> tensor<1x512x1xf32>
        %568 = "ttir.div"(%565, %567) : (tensor<1x512x1xf32>, tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %569 = "ttir.broadcast"(%568) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<1x512x1xf32>) -> tensor<1x512x768xf32>
        %570 = "ttir.subtract"(%562, %569) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %571 = "ttir.multiply"(%570, %570) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %572 = "ttir.sum"(%571) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x512x768xf32>) -> tensor<1x512xf32>
        %573 = "ttir.reshape"(%572) <{shape = [1 : i32, 512 : i32, 1 : i32]}> : (tensor<1x512xf32>) -> tensor<1x512x1xf32>
        %574 = "ttir.broadcast"(%573) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %575 = "ttir.reshape"(%7) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %576 = "ttir.broadcast"(%575) <{broadcast_dimensions = array<i64: 1, 512, 1>}> : (tensor<1x1x1xf32>) -> tensor<1x512x1xf32>
        %577 = "ttir.div"(%574, %576) : (tensor<1x512x1xf32>, tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %578 = "ttir.broadcast"(%568) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<1x512x1xf32>) -> tensor<1x512x768xf32>
        %579 = "ttir.subtract"(%562, %578) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %580 = "ttir.reshape"(%6) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %581 = "ttir.broadcast"(%580) <{broadcast_dimensions = array<i64: 1, 512, 1>}> : (tensor<1x1x1xf32>) -> tensor<1x512x1xf32>
        %582 = "ttir.add"(%577, %581) : (tensor<1x512x1xf32>, tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %583 = "ttir.sqrt"(%582) : (tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %584 = "ttir.broadcast"(%583) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<1x512x1xf32>) -> tensor<1x512x768xf32>
        %585 = "ttir.div"(%579, %584) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %586 = "ttir.reshape"(%arg49) <{shape = [1 : i32, 1 : i32, 768 : i32]}> : (tensor<768xf32>) -> tensor<1x1x768xf32>
        %587 = "ttir.broadcast"(%586) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x1x768xf32>
        %588 = "ttir.broadcast"(%587) <{broadcast_dimensions = array<i64: 1, 512, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x512x768xf32>
        %589 = "ttir.multiply"(%585, %588) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %590 = "ttir.reshape"(%arg50) <{shape = [1 : i32, 1 : i32, 768 : i32]}> : (tensor<768xf32>) -> tensor<1x1x768xf32>
        %591 = "ttir.broadcast"(%590) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x1x768xf32>
        %592 = "ttir.broadcast"(%591) <{broadcast_dimensions = array<i64: 1, 512, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x512x768xf32>
        %593 = "ttir.add"(%589, %592) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %594 = "ttir.dot_general"(%593, %arg47) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<1x512x768xf32>, tensor<768x3072xf32>) -> tensor<1x512x3072xf32>
        %595 = "ttir.multiply"(%594, %594) : (tensor<1x512x3072xf32>, tensor<1x512x3072xf32>) -> tensor<1x512x3072xf32>
        %596 = "ttir.multiply"(%595, %594) : (tensor<1x512x3072xf32>, tensor<1x512x3072xf32>) -> tensor<1x512x3072xf32>
        %597 = "ttir.reshape"(%3) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %598 = "ttir.broadcast"(%597) <{broadcast_dimensions = array<i64: 1, 512, 3072>}> : (tensor<1x1x1xf32>) -> tensor<1x512x3072xf32>
        %599 = "ttir.multiply"(%598, %596) : (tensor<1x512x3072xf32>, tensor<1x512x3072xf32>) -> tensor<1x512x3072xf32>
        %600 = "ttir.add"(%594, %599) : (tensor<1x512x3072xf32>, tensor<1x512x3072xf32>) -> tensor<1x512x3072xf32>
        %601 = "ttir.reshape"(%2) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %602 = "ttir.broadcast"(%601) <{broadcast_dimensions = array<i64: 1, 512, 3072>}> : (tensor<1x1x1xf32>) -> tensor<1x512x3072xf32>
        %603 = "ttir.multiply"(%602, %600) : (tensor<1x512x3072xf32>, tensor<1x512x3072xf32>) -> tensor<1x512x3072xf32>
        %604 = "ttir.tanh"(%603) : (tensor<1x512x3072xf32>) -> tensor<1x512x3072xf32>
        %605 = "ttir.reshape"(%1) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %606 = "ttir.broadcast"(%605) <{broadcast_dimensions = array<i64: 1, 512, 3072>}> : (tensor<1x1x1xf32>) -> tensor<1x512x3072xf32>
        %607 = "ttir.add"(%606, %604) : (tensor<1x512x3072xf32>, tensor<1x512x3072xf32>) -> tensor<1x512x3072xf32>
        %608 = "ttir.reshape"(%0) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %609 = "ttir.broadcast"(%608) <{broadcast_dimensions = array<i64: 1, 512, 3072>}> : (tensor<1x1x1xf32>) -> tensor<1x512x3072xf32>
        %610 = "ttir.multiply"(%609, %607) : (tensor<1x512x3072xf32>, tensor<1x512x3072xf32>) -> tensor<1x512x3072xf32>
        %611 = "ttir.multiply"(%594, %610) : (tensor<1x512x3072xf32>, tensor<1x512x3072xf32>) -> tensor<1x512x3072xf32>
        %612 = "ttir.dot_general"(%611, %arg48) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<1x512x3072xf32>, tensor<3072x768xf32>) -> tensor<1x512x768xf32>
        %613 = "ttir.add"(%562, %612) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %614 = "ttir.sum"(%613) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x512x768xf32>) -> tensor<1x512xf32>
        %615 = "ttir.reshape"(%614) <{shape = [1 : i32, 512 : i32, 1 : i32]}> : (tensor<1x512xf32>) -> tensor<1x512x1xf32>
        %616 = "ttir.broadcast"(%615) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %617 = "ttir.reshape"(%7) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %618 = "ttir.broadcast"(%617) <{broadcast_dimensions = array<i64: 1, 512, 1>}> : (tensor<1x1x1xf32>) -> tensor<1x512x1xf32>
        %619 = "ttir.div"(%616, %618) : (tensor<1x512x1xf32>, tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %620 = "ttir.broadcast"(%619) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<1x512x1xf32>) -> tensor<1x512x768xf32>
        %621 = "ttir.subtract"(%613, %620) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %622 = "ttir.multiply"(%621, %621) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %623 = "ttir.sum"(%622) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x512x768xf32>) -> tensor<1x512xf32>
        %624 = "ttir.reshape"(%623) <{shape = [1 : i32, 512 : i32, 1 : i32]}> : (tensor<1x512xf32>) -> tensor<1x512x1xf32>
        %625 = "ttir.broadcast"(%624) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %626 = "ttir.reshape"(%7) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %627 = "ttir.broadcast"(%626) <{broadcast_dimensions = array<i64: 1, 512, 1>}> : (tensor<1x1x1xf32>) -> tensor<1x512x1xf32>
        %628 = "ttir.div"(%625, %627) : (tensor<1x512x1xf32>, tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %629 = "ttir.broadcast"(%619) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<1x512x1xf32>) -> tensor<1x512x768xf32>
        %630 = "ttir.subtract"(%613, %629) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %631 = "ttir.reshape"(%6) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %632 = "ttir.broadcast"(%631) <{broadcast_dimensions = array<i64: 1, 512, 1>}> : (tensor<1x1x1xf32>) -> tensor<1x512x1xf32>
        %633 = "ttir.add"(%628, %632) : (tensor<1x512x1xf32>, tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %634 = "ttir.sqrt"(%633) : (tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %635 = "ttir.broadcast"(%634) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<1x512x1xf32>) -> tensor<1x512x768xf32>
        %636 = "ttir.div"(%630, %635) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %637 = "ttir.reshape"(%arg55) <{shape = [1 : i32, 1 : i32, 768 : i32]}> : (tensor<768xf32>) -> tensor<1x1x768xf32>
        %638 = "ttir.broadcast"(%637) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x1x768xf32>
        %639 = "ttir.broadcast"(%638) <{broadcast_dimensions = array<i64: 1, 512, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x512x768xf32>
        %640 = "ttir.multiply"(%636, %639) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %641 = "ttir.reshape"(%arg56) <{shape = [1 : i32, 1 : i32, 768 : i32]}> : (tensor<768xf32>) -> tensor<1x1x768xf32>
        %642 = "ttir.broadcast"(%641) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x1x768xf32>
        %643 = "ttir.broadcast"(%642) <{broadcast_dimensions = array<i64: 1, 512, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x512x768xf32>
        %644 = "ttir.add"(%640, %643) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %645 = "ttir.dot_general"(%644, %arg51) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<1x512x768xf32>, tensor<768x768xf32>) -> tensor<1x512x768xf32>
        %646 = "ttir.dot_general"(%644, %arg52) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<1x512x768xf32>, tensor<768x768xf32>) -> tensor<1x512x768xf32>
        %647 = "ttir.dot_general"(%644, %arg53) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<1x512x768xf32>, tensor<768x768xf32>) -> tensor<1x512x768xf32>
        %648 = "ttir.reshape"(%645) <{shape = [1 : i32, 512 : i32, 12 : i32, 64 : i32]}> : (tensor<1x512x768xf32>) -> tensor<1x512x12x64xf32>
        %649 = "ttir.permute"(%648) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x512x12x64xf32>) -> tensor<1x12x512x64xf32>
        %650 = "ttir.reshape"(%646) <{shape = [1 : i32, 512 : i32, 12 : i32, 64 : i32]}> : (tensor<1x512x768xf32>) -> tensor<1x512x12x64xf32>
        %651 = "ttir.permute"(%650) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x512x12x64xf32>) -> tensor<1x12x512x64xf32>
        %652 = "ttir.reshape"(%647) <{shape = [1 : i32, 512 : i32, 12 : i32, 64 : i32]}> : (tensor<1x512x768xf32>) -> tensor<1x512x12x64xf32>
        %653 = "ttir.permute"(%652) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x512x12x64xf32>) -> tensor<1x12x512x64xf32>
        %654 = "ttir.permute"(%651) <{permutation = array<i64: 0, 1, 3, 2>}> : (tensor<1x12x512x64xf32>) -> tensor<1x12x64x512xf32>
        %655 = "ttir.reshape"(%649) <{shape = [12 : i32, 512 : i32, 64 : i32]}> : (tensor<1x12x512x64xf32>) -> tensor<12x512x64xf32>
        %656 = "ttir.dot_general"(%655, %654) <{batch_dims_lhs = array<i64: 0>, batch_dims_rhs = array<i64: 1>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 2>}> : (tensor<12x512x64xf32>, tensor<1x12x64x512xf32>) -> tensor<12x512x1x512xf32>
        %657 = "ttir.permute"(%656) <{permutation = array<i64: 2, 0, 1, 3>}> : (tensor<12x512x1x512xf32>) -> tensor<1x12x512x512xf32>
        %658 = "ttir.sqrt"(%5) : (tensor<f32>) -> tensor<f32>
        %659 = "ttir.typecast"(%658) <{conservative_folding = false}> : (tensor<f32>) -> tensor<f32>
        %660 = "ttir.reshape"(%659) <{shape = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1x1xf32>
        %661 = "ttir.broadcast"(%660) <{broadcast_dimensions = array<i64: 1, 12, 512, 512>}> : (tensor<1x1x1x1xf32>) -> tensor<1x12x512x512xf32>
        %662 = "ttir.div"(%657, %661) : (tensor<1x12x512x512xf32>, tensor<1x12x512x512xf32>) -> tensor<1x12x512x512xf32>
        %663 = "ttir.max"(%662) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<1x12x512x512xf32>) -> tensor<1x12x512xf32>
        %664 = "ttir.reshape"(%4) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %665 = "ttir.broadcast"(%664) <{broadcast_dimensions = array<i64: 1, 12, 512>}> : (tensor<1x1x1xf32>) -> tensor<1x12x512xf32>
        %666 = "ttir.maximum"(%665, %663) : (tensor<1x12x512xf32>, tensor<1x12x512xf32>) -> tensor<1x12x512xf32>
        %667 = "ttir.reshape"(%666) <{shape = [1 : i32, 12 : i32, 512 : i32, 1 : i32]}> : (tensor<1x12x512xf32>) -> tensor<1x12x512x1xf32>
        %668 = "ttir.broadcast"(%667) <{broadcast_dimensions = array<i64: 1, 1, 1, 1>}> : (tensor<1x12x512x1xf32>) -> tensor<1x12x512x1xf32>
        %669 = "ttir.broadcast"(%668) <{broadcast_dimensions = array<i64: 1, 1, 1, 512>}> : (tensor<1x12x512x1xf32>) -> tensor<1x12x512x512xf32>
        %670 = "ttir.subtract"(%662, %669) : (tensor<1x12x512x512xf32>, tensor<1x12x512x512xf32>) -> tensor<1x12x512x512xf32>
        %671 = "ttir.exp"(%670) : (tensor<1x12x512x512xf32>) -> tensor<1x12x512x512xf32>
        %672 = "ttir.sum"(%671) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<1x12x512x512xf32>) -> tensor<1x12x512xf32>
        %673 = "ttir.reshape"(%672) <{shape = [1 : i32, 12 : i32, 512 : i32, 1 : i32]}> : (tensor<1x12x512xf32>) -> tensor<1x12x512x1xf32>
        %674 = "ttir.broadcast"(%673) <{broadcast_dimensions = array<i64: 1, 1, 1, 1>}> : (tensor<1x12x512x1xf32>) -> tensor<1x12x512x1xf32>
        %675 = "ttir.broadcast"(%674) <{broadcast_dimensions = array<i64: 1, 1, 1, 512>}> : (tensor<1x12x512x1xf32>) -> tensor<1x12x512x512xf32>
        %676 = "ttir.div"(%671, %675) : (tensor<1x12x512x512xf32>, tensor<1x12x512x512xf32>) -> tensor<1x12x512x512xf32>
        %677 = "ttir.reshape"(%676) <{shape = [12 : i32, 512 : i32, 512 : i32]}> : (tensor<1x12x512x512xf32>) -> tensor<12x512x512xf32>
        %678 = "ttir.dot_general"(%677, %653) <{batch_dims_lhs = array<i64: 0>, batch_dims_rhs = array<i64: 1>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 2>}> : (tensor<12x512x512xf32>, tensor<1x12x512x64xf32>) -> tensor<12x512x1x64xf32>
        %679 = "ttir.permute"(%678) <{permutation = array<i64: 2, 0, 1, 3>}> : (tensor<12x512x1x64xf32>) -> tensor<1x12x512x64xf32>
        %680 = "ttir.permute"(%679) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x12x512x64xf32>) -> tensor<1x512x12x64xf32>
        %681 = "ttir.reshape"(%680) <{shape = [1 : i32, 512 : i32, 768 : i32]}> : (tensor<1x512x12x64xf32>) -> tensor<1x512x768xf32>
        %682 = "ttir.dot_general"(%681, %arg54) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<1x512x768xf32>, tensor<768x768xf32>) -> tensor<1x512x768xf32>
        %683 = "ttir.add"(%613, %682) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %684 = "ttir.sum"(%683) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x512x768xf32>) -> tensor<1x512xf32>
        %685 = "ttir.reshape"(%684) <{shape = [1 : i32, 512 : i32, 1 : i32]}> : (tensor<1x512xf32>) -> tensor<1x512x1xf32>
        %686 = "ttir.broadcast"(%685) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %687 = "ttir.reshape"(%7) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %688 = "ttir.broadcast"(%687) <{broadcast_dimensions = array<i64: 1, 512, 1>}> : (tensor<1x1x1xf32>) -> tensor<1x512x1xf32>
        %689 = "ttir.div"(%686, %688) : (tensor<1x512x1xf32>, tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %690 = "ttir.broadcast"(%689) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<1x512x1xf32>) -> tensor<1x512x768xf32>
        %691 = "ttir.subtract"(%683, %690) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %692 = "ttir.multiply"(%691, %691) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %693 = "ttir.sum"(%692) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x512x768xf32>) -> tensor<1x512xf32>
        %694 = "ttir.reshape"(%693) <{shape = [1 : i32, 512 : i32, 1 : i32]}> : (tensor<1x512xf32>) -> tensor<1x512x1xf32>
        %695 = "ttir.broadcast"(%694) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %696 = "ttir.reshape"(%7) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %697 = "ttir.broadcast"(%696) <{broadcast_dimensions = array<i64: 1, 512, 1>}> : (tensor<1x1x1xf32>) -> tensor<1x512x1xf32>
        %698 = "ttir.div"(%695, %697) : (tensor<1x512x1xf32>, tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %699 = "ttir.broadcast"(%689) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<1x512x1xf32>) -> tensor<1x512x768xf32>
        %700 = "ttir.subtract"(%683, %699) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %701 = "ttir.reshape"(%6) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %702 = "ttir.broadcast"(%701) <{broadcast_dimensions = array<i64: 1, 512, 1>}> : (tensor<1x1x1xf32>) -> tensor<1x512x1xf32>
        %703 = "ttir.add"(%698, %702) : (tensor<1x512x1xf32>, tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %704 = "ttir.sqrt"(%703) : (tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %705 = "ttir.broadcast"(%704) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<1x512x1xf32>) -> tensor<1x512x768xf32>
        %706 = "ttir.div"(%700, %705) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %707 = "ttir.reshape"(%arg59) <{shape = [1 : i32, 1 : i32, 768 : i32]}> : (tensor<768xf32>) -> tensor<1x1x768xf32>
        %708 = "ttir.broadcast"(%707) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x1x768xf32>
        %709 = "ttir.broadcast"(%708) <{broadcast_dimensions = array<i64: 1, 512, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x512x768xf32>
        %710 = "ttir.multiply"(%706, %709) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %711 = "ttir.reshape"(%arg60) <{shape = [1 : i32, 1 : i32, 768 : i32]}> : (tensor<768xf32>) -> tensor<1x1x768xf32>
        %712 = "ttir.broadcast"(%711) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x1x768xf32>
        %713 = "ttir.broadcast"(%712) <{broadcast_dimensions = array<i64: 1, 512, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x512x768xf32>
        %714 = "ttir.add"(%710, %713) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %715 = "ttir.dot_general"(%714, %arg57) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<1x512x768xf32>, tensor<768x3072xf32>) -> tensor<1x512x3072xf32>
        %716 = "ttir.multiply"(%715, %715) : (tensor<1x512x3072xf32>, tensor<1x512x3072xf32>) -> tensor<1x512x3072xf32>
        %717 = "ttir.multiply"(%716, %715) : (tensor<1x512x3072xf32>, tensor<1x512x3072xf32>) -> tensor<1x512x3072xf32>
        %718 = "ttir.reshape"(%3) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %719 = "ttir.broadcast"(%718) <{broadcast_dimensions = array<i64: 1, 512, 3072>}> : (tensor<1x1x1xf32>) -> tensor<1x512x3072xf32>
        %720 = "ttir.multiply"(%719, %717) : (tensor<1x512x3072xf32>, tensor<1x512x3072xf32>) -> tensor<1x512x3072xf32>
        %721 = "ttir.add"(%715, %720) : (tensor<1x512x3072xf32>, tensor<1x512x3072xf32>) -> tensor<1x512x3072xf32>
        %722 = "ttir.reshape"(%2) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %723 = "ttir.broadcast"(%722) <{broadcast_dimensions = array<i64: 1, 512, 3072>}> : (tensor<1x1x1xf32>) -> tensor<1x512x3072xf32>
        %724 = "ttir.multiply"(%723, %721) : (tensor<1x512x3072xf32>, tensor<1x512x3072xf32>) -> tensor<1x512x3072xf32>
        %725 = "ttir.tanh"(%724) : (tensor<1x512x3072xf32>) -> tensor<1x512x3072xf32>
        %726 = "ttir.reshape"(%1) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %727 = "ttir.broadcast"(%726) <{broadcast_dimensions = array<i64: 1, 512, 3072>}> : (tensor<1x1x1xf32>) -> tensor<1x512x3072xf32>
        %728 = "ttir.add"(%727, %725) : (tensor<1x512x3072xf32>, tensor<1x512x3072xf32>) -> tensor<1x512x3072xf32>
        %729 = "ttir.reshape"(%0) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %730 = "ttir.broadcast"(%729) <{broadcast_dimensions = array<i64: 1, 512, 3072>}> : (tensor<1x1x1xf32>) -> tensor<1x512x3072xf32>
        %731 = "ttir.multiply"(%730, %728) : (tensor<1x512x3072xf32>, tensor<1x512x3072xf32>) -> tensor<1x512x3072xf32>
        %732 = "ttir.multiply"(%715, %731) : (tensor<1x512x3072xf32>, tensor<1x512x3072xf32>) -> tensor<1x512x3072xf32>
        %733 = "ttir.dot_general"(%732, %arg58) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<1x512x3072xf32>, tensor<3072x768xf32>) -> tensor<1x512x768xf32>
        %734 = "ttir.add"(%683, %733) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %735 = "ttir.sum"(%734) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x512x768xf32>) -> tensor<1x512xf32>
        %736 = "ttir.reshape"(%735) <{shape = [1 : i32, 512 : i32, 1 : i32]}> : (tensor<1x512xf32>) -> tensor<1x512x1xf32>
        %737 = "ttir.broadcast"(%736) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %738 = "ttir.reshape"(%7) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %739 = "ttir.broadcast"(%738) <{broadcast_dimensions = array<i64: 1, 512, 1>}> : (tensor<1x1x1xf32>) -> tensor<1x512x1xf32>
        %740 = "ttir.div"(%737, %739) : (tensor<1x512x1xf32>, tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %741 = "ttir.broadcast"(%740) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<1x512x1xf32>) -> tensor<1x512x768xf32>
        %742 = "ttir.subtract"(%734, %741) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %743 = "ttir.multiply"(%742, %742) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %744 = "ttir.sum"(%743) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x512x768xf32>) -> tensor<1x512xf32>
        %745 = "ttir.reshape"(%744) <{shape = [1 : i32, 512 : i32, 1 : i32]}> : (tensor<1x512xf32>) -> tensor<1x512x1xf32>
        %746 = "ttir.broadcast"(%745) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %747 = "ttir.reshape"(%7) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %748 = "ttir.broadcast"(%747) <{broadcast_dimensions = array<i64: 1, 512, 1>}> : (tensor<1x1x1xf32>) -> tensor<1x512x1xf32>
        %749 = "ttir.div"(%746, %748) : (tensor<1x512x1xf32>, tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %750 = "ttir.broadcast"(%740) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<1x512x1xf32>) -> tensor<1x512x768xf32>
        %751 = "ttir.subtract"(%734, %750) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %752 = "ttir.reshape"(%6) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %753 = "ttir.broadcast"(%752) <{broadcast_dimensions = array<i64: 1, 512, 1>}> : (tensor<1x1x1xf32>) -> tensor<1x512x1xf32>
        %754 = "ttir.add"(%749, %753) : (tensor<1x512x1xf32>, tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %755 = "ttir.sqrt"(%754) : (tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %756 = "ttir.broadcast"(%755) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<1x512x1xf32>) -> tensor<1x512x768xf32>
        %757 = "ttir.div"(%751, %756) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %758 = "ttir.reshape"(%arg65) <{shape = [1 : i32, 1 : i32, 768 : i32]}> : (tensor<768xf32>) -> tensor<1x1x768xf32>
        %759 = "ttir.broadcast"(%758) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x1x768xf32>
        %760 = "ttir.broadcast"(%759) <{broadcast_dimensions = array<i64: 1, 512, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x512x768xf32>
        %761 = "ttir.multiply"(%757, %760) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %762 = "ttir.reshape"(%arg66) <{shape = [1 : i32, 1 : i32, 768 : i32]}> : (tensor<768xf32>) -> tensor<1x1x768xf32>
        %763 = "ttir.broadcast"(%762) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x1x768xf32>
        %764 = "ttir.broadcast"(%763) <{broadcast_dimensions = array<i64: 1, 512, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x512x768xf32>
        %765 = "ttir.add"(%761, %764) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %766 = "ttir.dot_general"(%765, %arg61) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<1x512x768xf32>, tensor<768x768xf32>) -> tensor<1x512x768xf32>
        %767 = "ttir.dot_general"(%765, %arg62) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<1x512x768xf32>, tensor<768x768xf32>) -> tensor<1x512x768xf32>
        %768 = "ttir.dot_general"(%765, %arg63) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<1x512x768xf32>, tensor<768x768xf32>) -> tensor<1x512x768xf32>
        %769 = "ttir.reshape"(%766) <{shape = [1 : i32, 512 : i32, 12 : i32, 64 : i32]}> : (tensor<1x512x768xf32>) -> tensor<1x512x12x64xf32>
        %770 = "ttir.permute"(%769) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x512x12x64xf32>) -> tensor<1x12x512x64xf32>
        %771 = "ttir.reshape"(%767) <{shape = [1 : i32, 512 : i32, 12 : i32, 64 : i32]}> : (tensor<1x512x768xf32>) -> tensor<1x512x12x64xf32>
        %772 = "ttir.permute"(%771) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x512x12x64xf32>) -> tensor<1x12x512x64xf32>
        %773 = "ttir.reshape"(%768) <{shape = [1 : i32, 512 : i32, 12 : i32, 64 : i32]}> : (tensor<1x512x768xf32>) -> tensor<1x512x12x64xf32>
        %774 = "ttir.permute"(%773) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x512x12x64xf32>) -> tensor<1x12x512x64xf32>
        %775 = "ttir.permute"(%772) <{permutation = array<i64: 0, 1, 3, 2>}> : (tensor<1x12x512x64xf32>) -> tensor<1x12x64x512xf32>
        %776 = "ttir.reshape"(%770) <{shape = [12 : i32, 512 : i32, 64 : i32]}> : (tensor<1x12x512x64xf32>) -> tensor<12x512x64xf32>
        %777 = "ttir.dot_general"(%776, %775) <{batch_dims_lhs = array<i64: 0>, batch_dims_rhs = array<i64: 1>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 2>}> : (tensor<12x512x64xf32>, tensor<1x12x64x512xf32>) -> tensor<12x512x1x512xf32>
        %778 = "ttir.permute"(%777) <{permutation = array<i64: 2, 0, 1, 3>}> : (tensor<12x512x1x512xf32>) -> tensor<1x12x512x512xf32>
        %779 = "ttir.sqrt"(%5) : (tensor<f32>) -> tensor<f32>
        %780 = "ttir.typecast"(%779) <{conservative_folding = false}> : (tensor<f32>) -> tensor<f32>
        %781 = "ttir.reshape"(%780) <{shape = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1x1xf32>
        %782 = "ttir.broadcast"(%781) <{broadcast_dimensions = array<i64: 1, 12, 512, 512>}> : (tensor<1x1x1x1xf32>) -> tensor<1x12x512x512xf32>
        %783 = "ttir.div"(%778, %782) : (tensor<1x12x512x512xf32>, tensor<1x12x512x512xf32>) -> tensor<1x12x512x512xf32>
        %784 = "ttir.max"(%783) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<1x12x512x512xf32>) -> tensor<1x12x512xf32>
        %785 = "ttir.reshape"(%4) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %786 = "ttir.broadcast"(%785) <{broadcast_dimensions = array<i64: 1, 12, 512>}> : (tensor<1x1x1xf32>) -> tensor<1x12x512xf32>
        %787 = "ttir.maximum"(%786, %784) : (tensor<1x12x512xf32>, tensor<1x12x512xf32>) -> tensor<1x12x512xf32>
        %788 = "ttir.reshape"(%787) <{shape = [1 : i32, 12 : i32, 512 : i32, 1 : i32]}> : (tensor<1x12x512xf32>) -> tensor<1x12x512x1xf32>
        %789 = "ttir.broadcast"(%788) <{broadcast_dimensions = array<i64: 1, 1, 1, 1>}> : (tensor<1x12x512x1xf32>) -> tensor<1x12x512x1xf32>
        %790 = "ttir.broadcast"(%789) <{broadcast_dimensions = array<i64: 1, 1, 1, 512>}> : (tensor<1x12x512x1xf32>) -> tensor<1x12x512x512xf32>
        %791 = "ttir.subtract"(%783, %790) : (tensor<1x12x512x512xf32>, tensor<1x12x512x512xf32>) -> tensor<1x12x512x512xf32>
        %792 = "ttir.exp"(%791) : (tensor<1x12x512x512xf32>) -> tensor<1x12x512x512xf32>
        %793 = "ttir.sum"(%792) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<1x12x512x512xf32>) -> tensor<1x12x512xf32>
        %794 = "ttir.reshape"(%793) <{shape = [1 : i32, 12 : i32, 512 : i32, 1 : i32]}> : (tensor<1x12x512xf32>) -> tensor<1x12x512x1xf32>
        %795 = "ttir.broadcast"(%794) <{broadcast_dimensions = array<i64: 1, 1, 1, 1>}> : (tensor<1x12x512x1xf32>) -> tensor<1x12x512x1xf32>
        %796 = "ttir.broadcast"(%795) <{broadcast_dimensions = array<i64: 1, 1, 1, 512>}> : (tensor<1x12x512x1xf32>) -> tensor<1x12x512x512xf32>
        %797 = "ttir.div"(%792, %796) : (tensor<1x12x512x512xf32>, tensor<1x12x512x512xf32>) -> tensor<1x12x512x512xf32>
        %798 = "ttir.reshape"(%797) <{shape = [12 : i32, 512 : i32, 512 : i32]}> : (tensor<1x12x512x512xf32>) -> tensor<12x512x512xf32>
        %799 = "ttir.dot_general"(%798, %774) <{batch_dims_lhs = array<i64: 0>, batch_dims_rhs = array<i64: 1>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 2>}> : (tensor<12x512x512xf32>, tensor<1x12x512x64xf32>) -> tensor<12x512x1x64xf32>
        %800 = "ttir.permute"(%799) <{permutation = array<i64: 2, 0, 1, 3>}> : (tensor<12x512x1x64xf32>) -> tensor<1x12x512x64xf32>
        %801 = "ttir.permute"(%800) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x12x512x64xf32>) -> tensor<1x512x12x64xf32>
        %802 = "ttir.reshape"(%801) <{shape = [1 : i32, 512 : i32, 768 : i32]}> : (tensor<1x512x12x64xf32>) -> tensor<1x512x768xf32>
        %803 = "ttir.dot_general"(%802, %arg64) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<1x512x768xf32>, tensor<768x768xf32>) -> tensor<1x512x768xf32>
        %804 = "ttir.add"(%734, %803) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %805 = "ttir.sum"(%804) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x512x768xf32>) -> tensor<1x512xf32>
        %806 = "ttir.reshape"(%805) <{shape = [1 : i32, 512 : i32, 1 : i32]}> : (tensor<1x512xf32>) -> tensor<1x512x1xf32>
        %807 = "ttir.broadcast"(%806) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %808 = "ttir.reshape"(%7) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %809 = "ttir.broadcast"(%808) <{broadcast_dimensions = array<i64: 1, 512, 1>}> : (tensor<1x1x1xf32>) -> tensor<1x512x1xf32>
        %810 = "ttir.div"(%807, %809) : (tensor<1x512x1xf32>, tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %811 = "ttir.broadcast"(%810) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<1x512x1xf32>) -> tensor<1x512x768xf32>
        %812 = "ttir.subtract"(%804, %811) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %813 = "ttir.multiply"(%812, %812) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %814 = "ttir.sum"(%813) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x512x768xf32>) -> tensor<1x512xf32>
        %815 = "ttir.reshape"(%814) <{shape = [1 : i32, 512 : i32, 1 : i32]}> : (tensor<1x512xf32>) -> tensor<1x512x1xf32>
        %816 = "ttir.broadcast"(%815) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %817 = "ttir.reshape"(%7) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %818 = "ttir.broadcast"(%817) <{broadcast_dimensions = array<i64: 1, 512, 1>}> : (tensor<1x1x1xf32>) -> tensor<1x512x1xf32>
        %819 = "ttir.div"(%816, %818) : (tensor<1x512x1xf32>, tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %820 = "ttir.broadcast"(%810) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<1x512x1xf32>) -> tensor<1x512x768xf32>
        %821 = "ttir.subtract"(%804, %820) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %822 = "ttir.reshape"(%6) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %823 = "ttir.broadcast"(%822) <{broadcast_dimensions = array<i64: 1, 512, 1>}> : (tensor<1x1x1xf32>) -> tensor<1x512x1xf32>
        %824 = "ttir.add"(%819, %823) : (tensor<1x512x1xf32>, tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %825 = "ttir.sqrt"(%824) : (tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %826 = "ttir.broadcast"(%825) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<1x512x1xf32>) -> tensor<1x512x768xf32>
        %827 = "ttir.div"(%821, %826) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %828 = "ttir.reshape"(%arg69) <{shape = [1 : i32, 1 : i32, 768 : i32]}> : (tensor<768xf32>) -> tensor<1x1x768xf32>
        %829 = "ttir.broadcast"(%828) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x1x768xf32>
        %830 = "ttir.broadcast"(%829) <{broadcast_dimensions = array<i64: 1, 512, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x512x768xf32>
        %831 = "ttir.multiply"(%827, %830) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %832 = "ttir.reshape"(%arg70) <{shape = [1 : i32, 1 : i32, 768 : i32]}> : (tensor<768xf32>) -> tensor<1x1x768xf32>
        %833 = "ttir.broadcast"(%832) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x1x768xf32>
        %834 = "ttir.broadcast"(%833) <{broadcast_dimensions = array<i64: 1, 512, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x512x768xf32>
        %835 = "ttir.add"(%831, %834) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %836 = "ttir.dot_general"(%835, %arg67) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<1x512x768xf32>, tensor<768x3072xf32>) -> tensor<1x512x3072xf32>
        %837 = "ttir.multiply"(%836, %836) : (tensor<1x512x3072xf32>, tensor<1x512x3072xf32>) -> tensor<1x512x3072xf32>
        %838 = "ttir.multiply"(%837, %836) : (tensor<1x512x3072xf32>, tensor<1x512x3072xf32>) -> tensor<1x512x3072xf32>
        %839 = "ttir.reshape"(%3) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %840 = "ttir.broadcast"(%839) <{broadcast_dimensions = array<i64: 1, 512, 3072>}> : (tensor<1x1x1xf32>) -> tensor<1x512x3072xf32>
        %841 = "ttir.multiply"(%840, %838) : (tensor<1x512x3072xf32>, tensor<1x512x3072xf32>) -> tensor<1x512x3072xf32>
        %842 = "ttir.add"(%836, %841) : (tensor<1x512x3072xf32>, tensor<1x512x3072xf32>) -> tensor<1x512x3072xf32>
        %843 = "ttir.reshape"(%2) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %844 = "ttir.broadcast"(%843) <{broadcast_dimensions = array<i64: 1, 512, 3072>}> : (tensor<1x1x1xf32>) -> tensor<1x512x3072xf32>
        %845 = "ttir.multiply"(%844, %842) : (tensor<1x512x3072xf32>, tensor<1x512x3072xf32>) -> tensor<1x512x3072xf32>
        %846 = "ttir.tanh"(%845) : (tensor<1x512x3072xf32>) -> tensor<1x512x3072xf32>
        %847 = "ttir.reshape"(%1) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %848 = "ttir.broadcast"(%847) <{broadcast_dimensions = array<i64: 1, 512, 3072>}> : (tensor<1x1x1xf32>) -> tensor<1x512x3072xf32>
        %849 = "ttir.add"(%848, %846) : (tensor<1x512x3072xf32>, tensor<1x512x3072xf32>) -> tensor<1x512x3072xf32>
        %850 = "ttir.reshape"(%0) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %851 = "ttir.broadcast"(%850) <{broadcast_dimensions = array<i64: 1, 512, 3072>}> : (tensor<1x1x1xf32>) -> tensor<1x512x3072xf32>
        %852 = "ttir.multiply"(%851, %849) : (tensor<1x512x3072xf32>, tensor<1x512x3072xf32>) -> tensor<1x512x3072xf32>
        %853 = "ttir.multiply"(%836, %852) : (tensor<1x512x3072xf32>, tensor<1x512x3072xf32>) -> tensor<1x512x3072xf32>
        %854 = "ttir.dot_general"(%853, %arg68) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<1x512x3072xf32>, tensor<3072x768xf32>) -> tensor<1x512x768xf32>
        %855 = "ttir.add"(%804, %854) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %856 = "ttir.sum"(%855) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x512x768xf32>) -> tensor<1x512xf32>
        %857 = "ttir.reshape"(%856) <{shape = [1 : i32, 512 : i32, 1 : i32]}> : (tensor<1x512xf32>) -> tensor<1x512x1xf32>
        %858 = "ttir.broadcast"(%857) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %859 = "ttir.reshape"(%7) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %860 = "ttir.broadcast"(%859) <{broadcast_dimensions = array<i64: 1, 512, 1>}> : (tensor<1x1x1xf32>) -> tensor<1x512x1xf32>
        %861 = "ttir.div"(%858, %860) : (tensor<1x512x1xf32>, tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %862 = "ttir.broadcast"(%861) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<1x512x1xf32>) -> tensor<1x512x768xf32>
        %863 = "ttir.subtract"(%855, %862) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %864 = "ttir.multiply"(%863, %863) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %865 = "ttir.sum"(%864) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x512x768xf32>) -> tensor<1x512xf32>
        %866 = "ttir.reshape"(%865) <{shape = [1 : i32, 512 : i32, 1 : i32]}> : (tensor<1x512xf32>) -> tensor<1x512x1xf32>
        %867 = "ttir.broadcast"(%866) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %868 = "ttir.reshape"(%7) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %869 = "ttir.broadcast"(%868) <{broadcast_dimensions = array<i64: 1, 512, 1>}> : (tensor<1x1x1xf32>) -> tensor<1x512x1xf32>
        %870 = "ttir.div"(%867, %869) : (tensor<1x512x1xf32>, tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %871 = "ttir.broadcast"(%861) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<1x512x1xf32>) -> tensor<1x512x768xf32>
        %872 = "ttir.subtract"(%855, %871) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %873 = "ttir.reshape"(%6) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %874 = "ttir.broadcast"(%873) <{broadcast_dimensions = array<i64: 1, 512, 1>}> : (tensor<1x1x1xf32>) -> tensor<1x512x1xf32>
        %875 = "ttir.add"(%870, %874) : (tensor<1x512x1xf32>, tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %876 = "ttir.sqrt"(%875) : (tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %877 = "ttir.broadcast"(%876) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<1x512x1xf32>) -> tensor<1x512x768xf32>
        %878 = "ttir.div"(%872, %877) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %879 = "ttir.reshape"(%arg75) <{shape = [1 : i32, 1 : i32, 768 : i32]}> : (tensor<768xf32>) -> tensor<1x1x768xf32>
        %880 = "ttir.broadcast"(%879) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x1x768xf32>
        %881 = "ttir.broadcast"(%880) <{broadcast_dimensions = array<i64: 1, 512, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x512x768xf32>
        %882 = "ttir.multiply"(%878, %881) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %883 = "ttir.reshape"(%arg76) <{shape = [1 : i32, 1 : i32, 768 : i32]}> : (tensor<768xf32>) -> tensor<1x1x768xf32>
        %884 = "ttir.broadcast"(%883) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x1x768xf32>
        %885 = "ttir.broadcast"(%884) <{broadcast_dimensions = array<i64: 1, 512, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x512x768xf32>
        %886 = "ttir.add"(%882, %885) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %887 = "ttir.dot_general"(%886, %arg71) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<1x512x768xf32>, tensor<768x768xf32>) -> tensor<1x512x768xf32>
        %888 = "ttir.dot_general"(%886, %arg72) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<1x512x768xf32>, tensor<768x768xf32>) -> tensor<1x512x768xf32>
        %889 = "ttir.dot_general"(%886, %arg73) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<1x512x768xf32>, tensor<768x768xf32>) -> tensor<1x512x768xf32>
        %890 = "ttir.reshape"(%887) <{shape = [1 : i32, 512 : i32, 12 : i32, 64 : i32]}> : (tensor<1x512x768xf32>) -> tensor<1x512x12x64xf32>
        %891 = "ttir.permute"(%890) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x512x12x64xf32>) -> tensor<1x12x512x64xf32>
        %892 = "ttir.reshape"(%888) <{shape = [1 : i32, 512 : i32, 12 : i32, 64 : i32]}> : (tensor<1x512x768xf32>) -> tensor<1x512x12x64xf32>
        %893 = "ttir.permute"(%892) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x512x12x64xf32>) -> tensor<1x12x512x64xf32>
        %894 = "ttir.reshape"(%889) <{shape = [1 : i32, 512 : i32, 12 : i32, 64 : i32]}> : (tensor<1x512x768xf32>) -> tensor<1x512x12x64xf32>
        %895 = "ttir.permute"(%894) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x512x12x64xf32>) -> tensor<1x12x512x64xf32>
        %896 = "ttir.permute"(%893) <{permutation = array<i64: 0, 1, 3, 2>}> : (tensor<1x12x512x64xf32>) -> tensor<1x12x64x512xf32>
        %897 = "ttir.reshape"(%891) <{shape = [12 : i32, 512 : i32, 64 : i32]}> : (tensor<1x12x512x64xf32>) -> tensor<12x512x64xf32>
        %898 = "ttir.dot_general"(%897, %896) <{batch_dims_lhs = array<i64: 0>, batch_dims_rhs = array<i64: 1>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 2>}> : (tensor<12x512x64xf32>, tensor<1x12x64x512xf32>) -> tensor<12x512x1x512xf32>
        %899 = "ttir.permute"(%898) <{permutation = array<i64: 2, 0, 1, 3>}> : (tensor<12x512x1x512xf32>) -> tensor<1x12x512x512xf32>
        %900 = "ttir.sqrt"(%5) : (tensor<f32>) -> tensor<f32>
        %901 = "ttir.typecast"(%900) <{conservative_folding = false}> : (tensor<f32>) -> tensor<f32>
        %902 = "ttir.reshape"(%901) <{shape = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1x1xf32>
        %903 = "ttir.broadcast"(%902) <{broadcast_dimensions = array<i64: 1, 12, 512, 512>}> : (tensor<1x1x1x1xf32>) -> tensor<1x12x512x512xf32>
        %904 = "ttir.div"(%899, %903) : (tensor<1x12x512x512xf32>, tensor<1x12x512x512xf32>) -> tensor<1x12x512x512xf32>
        %905 = "ttir.max"(%904) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<1x12x512x512xf32>) -> tensor<1x12x512xf32>
        %906 = "ttir.reshape"(%4) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %907 = "ttir.broadcast"(%906) <{broadcast_dimensions = array<i64: 1, 12, 512>}> : (tensor<1x1x1xf32>) -> tensor<1x12x512xf32>
        %908 = "ttir.maximum"(%907, %905) : (tensor<1x12x512xf32>, tensor<1x12x512xf32>) -> tensor<1x12x512xf32>
        %909 = "ttir.reshape"(%908) <{shape = [1 : i32, 12 : i32, 512 : i32, 1 : i32]}> : (tensor<1x12x512xf32>) -> tensor<1x12x512x1xf32>
        %910 = "ttir.broadcast"(%909) <{broadcast_dimensions = array<i64: 1, 1, 1, 1>}> : (tensor<1x12x512x1xf32>) -> tensor<1x12x512x1xf32>
        %911 = "ttir.broadcast"(%910) <{broadcast_dimensions = array<i64: 1, 1, 1, 512>}> : (tensor<1x12x512x1xf32>) -> tensor<1x12x512x512xf32>
        %912 = "ttir.subtract"(%904, %911) : (tensor<1x12x512x512xf32>, tensor<1x12x512x512xf32>) -> tensor<1x12x512x512xf32>
        %913 = "ttir.exp"(%912) : (tensor<1x12x512x512xf32>) -> tensor<1x12x512x512xf32>
        %914 = "ttir.sum"(%913) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<1x12x512x512xf32>) -> tensor<1x12x512xf32>
        %915 = "ttir.reshape"(%914) <{shape = [1 : i32, 12 : i32, 512 : i32, 1 : i32]}> : (tensor<1x12x512xf32>) -> tensor<1x12x512x1xf32>
        %916 = "ttir.broadcast"(%915) <{broadcast_dimensions = array<i64: 1, 1, 1, 1>}> : (tensor<1x12x512x1xf32>) -> tensor<1x12x512x1xf32>
        %917 = "ttir.broadcast"(%916) <{broadcast_dimensions = array<i64: 1, 1, 1, 512>}> : (tensor<1x12x512x1xf32>) -> tensor<1x12x512x512xf32>
        %918 = "ttir.div"(%913, %917) : (tensor<1x12x512x512xf32>, tensor<1x12x512x512xf32>) -> tensor<1x12x512x512xf32>
        %919 = "ttir.reshape"(%918) <{shape = [12 : i32, 512 : i32, 512 : i32]}> : (tensor<1x12x512x512xf32>) -> tensor<12x512x512xf32>
        %920 = "ttir.dot_general"(%919, %895) <{batch_dims_lhs = array<i64: 0>, batch_dims_rhs = array<i64: 1>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 2>}> : (tensor<12x512x512xf32>, tensor<1x12x512x64xf32>) -> tensor<12x512x1x64xf32>
        %921 = "ttir.permute"(%920) <{permutation = array<i64: 2, 0, 1, 3>}> : (tensor<12x512x1x64xf32>) -> tensor<1x12x512x64xf32>
        %922 = "ttir.permute"(%921) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x12x512x64xf32>) -> tensor<1x512x12x64xf32>
        %923 = "ttir.reshape"(%922) <{shape = [1 : i32, 512 : i32, 768 : i32]}> : (tensor<1x512x12x64xf32>) -> tensor<1x512x768xf32>
        %924 = "ttir.dot_general"(%923, %arg74) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<1x512x768xf32>, tensor<768x768xf32>) -> tensor<1x512x768xf32>
        %925 = "ttir.add"(%855, %924) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %926 = "ttir.sum"(%925) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x512x768xf32>) -> tensor<1x512xf32>
        %927 = "ttir.reshape"(%926) <{shape = [1 : i32, 512 : i32, 1 : i32]}> : (tensor<1x512xf32>) -> tensor<1x512x1xf32>
        %928 = "ttir.broadcast"(%927) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %929 = "ttir.reshape"(%7) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %930 = "ttir.broadcast"(%929) <{broadcast_dimensions = array<i64: 1, 512, 1>}> : (tensor<1x1x1xf32>) -> tensor<1x512x1xf32>
        %931 = "ttir.div"(%928, %930) : (tensor<1x512x1xf32>, tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %932 = "ttir.broadcast"(%931) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<1x512x1xf32>) -> tensor<1x512x768xf32>
        %933 = "ttir.subtract"(%925, %932) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %934 = "ttir.multiply"(%933, %933) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %935 = "ttir.sum"(%934) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x512x768xf32>) -> tensor<1x512xf32>
        %936 = "ttir.reshape"(%935) <{shape = [1 : i32, 512 : i32, 1 : i32]}> : (tensor<1x512xf32>) -> tensor<1x512x1xf32>
        %937 = "ttir.broadcast"(%936) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %938 = "ttir.reshape"(%7) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %939 = "ttir.broadcast"(%938) <{broadcast_dimensions = array<i64: 1, 512, 1>}> : (tensor<1x1x1xf32>) -> tensor<1x512x1xf32>
        %940 = "ttir.div"(%937, %939) : (tensor<1x512x1xf32>, tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %941 = "ttir.broadcast"(%931) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<1x512x1xf32>) -> tensor<1x512x768xf32>
        %942 = "ttir.subtract"(%925, %941) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %943 = "ttir.reshape"(%6) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %944 = "ttir.broadcast"(%943) <{broadcast_dimensions = array<i64: 1, 512, 1>}> : (tensor<1x1x1xf32>) -> tensor<1x512x1xf32>
        %945 = "ttir.add"(%940, %944) : (tensor<1x512x1xf32>, tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %946 = "ttir.sqrt"(%945) : (tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %947 = "ttir.broadcast"(%946) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<1x512x1xf32>) -> tensor<1x512x768xf32>
        %948 = "ttir.div"(%942, %947) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %949 = "ttir.reshape"(%arg79) <{shape = [1 : i32, 1 : i32, 768 : i32]}> : (tensor<768xf32>) -> tensor<1x1x768xf32>
        %950 = "ttir.broadcast"(%949) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x1x768xf32>
        %951 = "ttir.broadcast"(%950) <{broadcast_dimensions = array<i64: 1, 512, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x512x768xf32>
        %952 = "ttir.multiply"(%948, %951) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %953 = "ttir.reshape"(%arg80) <{shape = [1 : i32, 1 : i32, 768 : i32]}> : (tensor<768xf32>) -> tensor<1x1x768xf32>
        %954 = "ttir.broadcast"(%953) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x1x768xf32>
        %955 = "ttir.broadcast"(%954) <{broadcast_dimensions = array<i64: 1, 512, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x512x768xf32>
        %956 = "ttir.add"(%952, %955) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %957 = "ttir.dot_general"(%956, %arg77) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<1x512x768xf32>, tensor<768x3072xf32>) -> tensor<1x512x3072xf32>
        %958 = "ttir.multiply"(%957, %957) : (tensor<1x512x3072xf32>, tensor<1x512x3072xf32>) -> tensor<1x512x3072xf32>
        %959 = "ttir.multiply"(%958, %957) : (tensor<1x512x3072xf32>, tensor<1x512x3072xf32>) -> tensor<1x512x3072xf32>
        %960 = "ttir.reshape"(%3) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %961 = "ttir.broadcast"(%960) <{broadcast_dimensions = array<i64: 1, 512, 3072>}> : (tensor<1x1x1xf32>) -> tensor<1x512x3072xf32>
        %962 = "ttir.multiply"(%961, %959) : (tensor<1x512x3072xf32>, tensor<1x512x3072xf32>) -> tensor<1x512x3072xf32>
        %963 = "ttir.add"(%957, %962) : (tensor<1x512x3072xf32>, tensor<1x512x3072xf32>) -> tensor<1x512x3072xf32>
        %964 = "ttir.reshape"(%2) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %965 = "ttir.broadcast"(%964) <{broadcast_dimensions = array<i64: 1, 512, 3072>}> : (tensor<1x1x1xf32>) -> tensor<1x512x3072xf32>
        %966 = "ttir.multiply"(%965, %963) : (tensor<1x512x3072xf32>, tensor<1x512x3072xf32>) -> tensor<1x512x3072xf32>
        %967 = "ttir.tanh"(%966) : (tensor<1x512x3072xf32>) -> tensor<1x512x3072xf32>
        %968 = "ttir.reshape"(%1) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %969 = "ttir.broadcast"(%968) <{broadcast_dimensions = array<i64: 1, 512, 3072>}> : (tensor<1x1x1xf32>) -> tensor<1x512x3072xf32>
        %970 = "ttir.add"(%969, %967) : (tensor<1x512x3072xf32>, tensor<1x512x3072xf32>) -> tensor<1x512x3072xf32>
        %971 = "ttir.reshape"(%0) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %972 = "ttir.broadcast"(%971) <{broadcast_dimensions = array<i64: 1, 512, 3072>}> : (tensor<1x1x1xf32>) -> tensor<1x512x3072xf32>
        %973 = "ttir.multiply"(%972, %970) : (tensor<1x512x3072xf32>, tensor<1x512x3072xf32>) -> tensor<1x512x3072xf32>
        %974 = "ttir.multiply"(%957, %973) : (tensor<1x512x3072xf32>, tensor<1x512x3072xf32>) -> tensor<1x512x3072xf32>
        %975 = "ttir.dot_general"(%974, %arg78) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<1x512x3072xf32>, tensor<3072x768xf32>) -> tensor<1x512x768xf32>
        %976 = "ttir.add"(%925, %975) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %977 = "ttir.sum"(%976) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x512x768xf32>) -> tensor<1x512xf32>
        %978 = "ttir.reshape"(%977) <{shape = [1 : i32, 512 : i32, 1 : i32]}> : (tensor<1x512xf32>) -> tensor<1x512x1xf32>
        %979 = "ttir.broadcast"(%978) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %980 = "ttir.reshape"(%7) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %981 = "ttir.broadcast"(%980) <{broadcast_dimensions = array<i64: 1, 512, 1>}> : (tensor<1x1x1xf32>) -> tensor<1x512x1xf32>
        %982 = "ttir.div"(%979, %981) : (tensor<1x512x1xf32>, tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %983 = "ttir.broadcast"(%982) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<1x512x1xf32>) -> tensor<1x512x768xf32>
        %984 = "ttir.subtract"(%976, %983) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %985 = "ttir.multiply"(%984, %984) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %986 = "ttir.sum"(%985) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x512x768xf32>) -> tensor<1x512xf32>
        %987 = "ttir.reshape"(%986) <{shape = [1 : i32, 512 : i32, 1 : i32]}> : (tensor<1x512xf32>) -> tensor<1x512x1xf32>
        %988 = "ttir.broadcast"(%987) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %989 = "ttir.reshape"(%7) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %990 = "ttir.broadcast"(%989) <{broadcast_dimensions = array<i64: 1, 512, 1>}> : (tensor<1x1x1xf32>) -> tensor<1x512x1xf32>
        %991 = "ttir.div"(%988, %990) : (tensor<1x512x1xf32>, tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %992 = "ttir.broadcast"(%982) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<1x512x1xf32>) -> tensor<1x512x768xf32>
        %993 = "ttir.subtract"(%976, %992) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %994 = "ttir.reshape"(%6) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %995 = "ttir.broadcast"(%994) <{broadcast_dimensions = array<i64: 1, 512, 1>}> : (tensor<1x1x1xf32>) -> tensor<1x512x1xf32>
        %996 = "ttir.add"(%991, %995) : (tensor<1x512x1xf32>, tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %997 = "ttir.sqrt"(%996) : (tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %998 = "ttir.broadcast"(%997) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<1x512x1xf32>) -> tensor<1x512x768xf32>
        %999 = "ttir.div"(%993, %998) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %1000 = "ttir.reshape"(%arg85) <{shape = [1 : i32, 1 : i32, 768 : i32]}> : (tensor<768xf32>) -> tensor<1x1x768xf32>
        %1001 = "ttir.broadcast"(%1000) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x1x768xf32>
        %1002 = "ttir.broadcast"(%1001) <{broadcast_dimensions = array<i64: 1, 512, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x512x768xf32>
        %1003 = "ttir.multiply"(%999, %1002) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %1004 = "ttir.reshape"(%arg86) <{shape = [1 : i32, 1 : i32, 768 : i32]}> : (tensor<768xf32>) -> tensor<1x1x768xf32>
        %1005 = "ttir.broadcast"(%1004) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x1x768xf32>
        %1006 = "ttir.broadcast"(%1005) <{broadcast_dimensions = array<i64: 1, 512, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x512x768xf32>
        %1007 = "ttir.add"(%1003, %1006) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %1008 = "ttir.dot_general"(%1007, %arg81) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<1x512x768xf32>, tensor<768x768xf32>) -> tensor<1x512x768xf32>
        %1009 = "ttir.dot_general"(%1007, %arg82) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<1x512x768xf32>, tensor<768x768xf32>) -> tensor<1x512x768xf32>
        %1010 = "ttir.dot_general"(%1007, %arg83) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<1x512x768xf32>, tensor<768x768xf32>) -> tensor<1x512x768xf32>
        %1011 = "ttir.reshape"(%1008) <{shape = [1 : i32, 512 : i32, 12 : i32, 64 : i32]}> : (tensor<1x512x768xf32>) -> tensor<1x512x12x64xf32>
        %1012 = "ttir.permute"(%1011) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x512x12x64xf32>) -> tensor<1x12x512x64xf32>
        %1013 = "ttir.reshape"(%1009) <{shape = [1 : i32, 512 : i32, 12 : i32, 64 : i32]}> : (tensor<1x512x768xf32>) -> tensor<1x512x12x64xf32>
        %1014 = "ttir.permute"(%1013) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x512x12x64xf32>) -> tensor<1x12x512x64xf32>
        %1015 = "ttir.reshape"(%1010) <{shape = [1 : i32, 512 : i32, 12 : i32, 64 : i32]}> : (tensor<1x512x768xf32>) -> tensor<1x512x12x64xf32>
        %1016 = "ttir.permute"(%1015) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x512x12x64xf32>) -> tensor<1x12x512x64xf32>
        %1017 = "ttir.permute"(%1014) <{permutation = array<i64: 0, 1, 3, 2>}> : (tensor<1x12x512x64xf32>) -> tensor<1x12x64x512xf32>
        %1018 = "ttir.reshape"(%1012) <{shape = [12 : i32, 512 : i32, 64 : i32]}> : (tensor<1x12x512x64xf32>) -> tensor<12x512x64xf32>
        %1019 = "ttir.dot_general"(%1018, %1017) <{batch_dims_lhs = array<i64: 0>, batch_dims_rhs = array<i64: 1>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 2>}> : (tensor<12x512x64xf32>, tensor<1x12x64x512xf32>) -> tensor<12x512x1x512xf32>
        %1020 = "ttir.permute"(%1019) <{permutation = array<i64: 2, 0, 1, 3>}> : (tensor<12x512x1x512xf32>) -> tensor<1x12x512x512xf32>
        %1021 = "ttir.sqrt"(%5) : (tensor<f32>) -> tensor<f32>
        %1022 = "ttir.typecast"(%1021) <{conservative_folding = false}> : (tensor<f32>) -> tensor<f32>
        %1023 = "ttir.reshape"(%1022) <{shape = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1x1xf32>
        %1024 = "ttir.broadcast"(%1023) <{broadcast_dimensions = array<i64: 1, 12, 512, 512>}> : (tensor<1x1x1x1xf32>) -> tensor<1x12x512x512xf32>
        %1025 = "ttir.div"(%1020, %1024) : (tensor<1x12x512x512xf32>, tensor<1x12x512x512xf32>) -> tensor<1x12x512x512xf32>
        %1026 = "ttir.max"(%1025) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<1x12x512x512xf32>) -> tensor<1x12x512xf32>
        %1027 = "ttir.reshape"(%4) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %1028 = "ttir.broadcast"(%1027) <{broadcast_dimensions = array<i64: 1, 12, 512>}> : (tensor<1x1x1xf32>) -> tensor<1x12x512xf32>
        %1029 = "ttir.maximum"(%1028, %1026) : (tensor<1x12x512xf32>, tensor<1x12x512xf32>) -> tensor<1x12x512xf32>
        %1030 = "ttir.reshape"(%1029) <{shape = [1 : i32, 12 : i32, 512 : i32, 1 : i32]}> : (tensor<1x12x512xf32>) -> tensor<1x12x512x1xf32>
        %1031 = "ttir.broadcast"(%1030) <{broadcast_dimensions = array<i64: 1, 1, 1, 1>}> : (tensor<1x12x512x1xf32>) -> tensor<1x12x512x1xf32>
        %1032 = "ttir.broadcast"(%1031) <{broadcast_dimensions = array<i64: 1, 1, 1, 512>}> : (tensor<1x12x512x1xf32>) -> tensor<1x12x512x512xf32>
        %1033 = "ttir.subtract"(%1025, %1032) : (tensor<1x12x512x512xf32>, tensor<1x12x512x512xf32>) -> tensor<1x12x512x512xf32>
        %1034 = "ttir.exp"(%1033) : (tensor<1x12x512x512xf32>) -> tensor<1x12x512x512xf32>
        %1035 = "ttir.sum"(%1034) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<1x12x512x512xf32>) -> tensor<1x12x512xf32>
        %1036 = "ttir.reshape"(%1035) <{shape = [1 : i32, 12 : i32, 512 : i32, 1 : i32]}> : (tensor<1x12x512xf32>) -> tensor<1x12x512x1xf32>
        %1037 = "ttir.broadcast"(%1036) <{broadcast_dimensions = array<i64: 1, 1, 1, 1>}> : (tensor<1x12x512x1xf32>) -> tensor<1x12x512x1xf32>
        %1038 = "ttir.broadcast"(%1037) <{broadcast_dimensions = array<i64: 1, 1, 1, 512>}> : (tensor<1x12x512x1xf32>) -> tensor<1x12x512x512xf32>
        %1039 = "ttir.div"(%1034, %1038) : (tensor<1x12x512x512xf32>, tensor<1x12x512x512xf32>) -> tensor<1x12x512x512xf32>
        %1040 = "ttir.reshape"(%1039) <{shape = [12 : i32, 512 : i32, 512 : i32]}> : (tensor<1x12x512x512xf32>) -> tensor<12x512x512xf32>
        %1041 = "ttir.dot_general"(%1040, %1016) <{batch_dims_lhs = array<i64: 0>, batch_dims_rhs = array<i64: 1>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 2>}> : (tensor<12x512x512xf32>, tensor<1x12x512x64xf32>) -> tensor<12x512x1x64xf32>
        %1042 = "ttir.permute"(%1041) <{permutation = array<i64: 2, 0, 1, 3>}> : (tensor<12x512x1x64xf32>) -> tensor<1x12x512x64xf32>
        %1043 = "ttir.permute"(%1042) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x12x512x64xf32>) -> tensor<1x512x12x64xf32>
        %1044 = "ttir.reshape"(%1043) <{shape = [1 : i32, 512 : i32, 768 : i32]}> : (tensor<1x512x12x64xf32>) -> tensor<1x512x768xf32>
        %1045 = "ttir.dot_general"(%1044, %arg84) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<1x512x768xf32>, tensor<768x768xf32>) -> tensor<1x512x768xf32>
        %1046 = "ttir.add"(%976, %1045) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %1047 = "ttir.sum"(%1046) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x512x768xf32>) -> tensor<1x512xf32>
        %1048 = "ttir.reshape"(%1047) <{shape = [1 : i32, 512 : i32, 1 : i32]}> : (tensor<1x512xf32>) -> tensor<1x512x1xf32>
        %1049 = "ttir.broadcast"(%1048) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %1050 = "ttir.reshape"(%7) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %1051 = "ttir.broadcast"(%1050) <{broadcast_dimensions = array<i64: 1, 512, 1>}> : (tensor<1x1x1xf32>) -> tensor<1x512x1xf32>
        %1052 = "ttir.div"(%1049, %1051) : (tensor<1x512x1xf32>, tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %1053 = "ttir.broadcast"(%1052) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<1x512x1xf32>) -> tensor<1x512x768xf32>
        %1054 = "ttir.subtract"(%1046, %1053) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %1055 = "ttir.multiply"(%1054, %1054) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %1056 = "ttir.sum"(%1055) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x512x768xf32>) -> tensor<1x512xf32>
        %1057 = "ttir.reshape"(%1056) <{shape = [1 : i32, 512 : i32, 1 : i32]}> : (tensor<1x512xf32>) -> tensor<1x512x1xf32>
        %1058 = "ttir.broadcast"(%1057) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %1059 = "ttir.reshape"(%7) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %1060 = "ttir.broadcast"(%1059) <{broadcast_dimensions = array<i64: 1, 512, 1>}> : (tensor<1x1x1xf32>) -> tensor<1x512x1xf32>
        %1061 = "ttir.div"(%1058, %1060) : (tensor<1x512x1xf32>, tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %1062 = "ttir.broadcast"(%1052) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<1x512x1xf32>) -> tensor<1x512x768xf32>
        %1063 = "ttir.subtract"(%1046, %1062) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %1064 = "ttir.reshape"(%6) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %1065 = "ttir.broadcast"(%1064) <{broadcast_dimensions = array<i64: 1, 512, 1>}> : (tensor<1x1x1xf32>) -> tensor<1x512x1xf32>
        %1066 = "ttir.add"(%1061, %1065) : (tensor<1x512x1xf32>, tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %1067 = "ttir.sqrt"(%1066) : (tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %1068 = "ttir.broadcast"(%1067) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<1x512x1xf32>) -> tensor<1x512x768xf32>
        %1069 = "ttir.div"(%1063, %1068) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %1070 = "ttir.reshape"(%arg89) <{shape = [1 : i32, 1 : i32, 768 : i32]}> : (tensor<768xf32>) -> tensor<1x1x768xf32>
        %1071 = "ttir.broadcast"(%1070) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x1x768xf32>
        %1072 = "ttir.broadcast"(%1071) <{broadcast_dimensions = array<i64: 1, 512, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x512x768xf32>
        %1073 = "ttir.multiply"(%1069, %1072) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %1074 = "ttir.reshape"(%arg90) <{shape = [1 : i32, 1 : i32, 768 : i32]}> : (tensor<768xf32>) -> tensor<1x1x768xf32>
        %1075 = "ttir.broadcast"(%1074) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x1x768xf32>
        %1076 = "ttir.broadcast"(%1075) <{broadcast_dimensions = array<i64: 1, 512, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x512x768xf32>
        %1077 = "ttir.add"(%1073, %1076) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %1078 = "ttir.dot_general"(%1077, %arg87) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<1x512x768xf32>, tensor<768x3072xf32>) -> tensor<1x512x3072xf32>
        %1079 = "ttir.multiply"(%1078, %1078) : (tensor<1x512x3072xf32>, tensor<1x512x3072xf32>) -> tensor<1x512x3072xf32>
        %1080 = "ttir.multiply"(%1079, %1078) : (tensor<1x512x3072xf32>, tensor<1x512x3072xf32>) -> tensor<1x512x3072xf32>
        %1081 = "ttir.reshape"(%3) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %1082 = "ttir.broadcast"(%1081) <{broadcast_dimensions = array<i64: 1, 512, 3072>}> : (tensor<1x1x1xf32>) -> tensor<1x512x3072xf32>
        %1083 = "ttir.multiply"(%1082, %1080) : (tensor<1x512x3072xf32>, tensor<1x512x3072xf32>) -> tensor<1x512x3072xf32>
        %1084 = "ttir.add"(%1078, %1083) : (tensor<1x512x3072xf32>, tensor<1x512x3072xf32>) -> tensor<1x512x3072xf32>
        %1085 = "ttir.reshape"(%2) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %1086 = "ttir.broadcast"(%1085) <{broadcast_dimensions = array<i64: 1, 512, 3072>}> : (tensor<1x1x1xf32>) -> tensor<1x512x3072xf32>
        %1087 = "ttir.multiply"(%1086, %1084) : (tensor<1x512x3072xf32>, tensor<1x512x3072xf32>) -> tensor<1x512x3072xf32>
        %1088 = "ttir.tanh"(%1087) : (tensor<1x512x3072xf32>) -> tensor<1x512x3072xf32>
        %1089 = "ttir.reshape"(%1) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %1090 = "ttir.broadcast"(%1089) <{broadcast_dimensions = array<i64: 1, 512, 3072>}> : (tensor<1x1x1xf32>) -> tensor<1x512x3072xf32>
        %1091 = "ttir.add"(%1090, %1088) : (tensor<1x512x3072xf32>, tensor<1x512x3072xf32>) -> tensor<1x512x3072xf32>
        %1092 = "ttir.reshape"(%0) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %1093 = "ttir.broadcast"(%1092) <{broadcast_dimensions = array<i64: 1, 512, 3072>}> : (tensor<1x1x1xf32>) -> tensor<1x512x3072xf32>
        %1094 = "ttir.multiply"(%1093, %1091) : (tensor<1x512x3072xf32>, tensor<1x512x3072xf32>) -> tensor<1x512x3072xf32>
        %1095 = "ttir.multiply"(%1078, %1094) : (tensor<1x512x3072xf32>, tensor<1x512x3072xf32>) -> tensor<1x512x3072xf32>
        %1096 = "ttir.dot_general"(%1095, %arg88) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<1x512x3072xf32>, tensor<3072x768xf32>) -> tensor<1x512x768xf32>
        %1097 = "ttir.add"(%1046, %1096) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %1098 = "ttir.sum"(%1097) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x512x768xf32>) -> tensor<1x512xf32>
        %1099 = "ttir.reshape"(%1098) <{shape = [1 : i32, 512 : i32, 1 : i32]}> : (tensor<1x512xf32>) -> tensor<1x512x1xf32>
        %1100 = "ttir.broadcast"(%1099) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %1101 = "ttir.reshape"(%7) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %1102 = "ttir.broadcast"(%1101) <{broadcast_dimensions = array<i64: 1, 512, 1>}> : (tensor<1x1x1xf32>) -> tensor<1x512x1xf32>
        %1103 = "ttir.div"(%1100, %1102) : (tensor<1x512x1xf32>, tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %1104 = "ttir.broadcast"(%1103) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<1x512x1xf32>) -> tensor<1x512x768xf32>
        %1105 = "ttir.subtract"(%1097, %1104) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %1106 = "ttir.multiply"(%1105, %1105) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %1107 = "ttir.sum"(%1106) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x512x768xf32>) -> tensor<1x512xf32>
        %1108 = "ttir.reshape"(%1107) <{shape = [1 : i32, 512 : i32, 1 : i32]}> : (tensor<1x512xf32>) -> tensor<1x512x1xf32>
        %1109 = "ttir.broadcast"(%1108) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %1110 = "ttir.reshape"(%7) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %1111 = "ttir.broadcast"(%1110) <{broadcast_dimensions = array<i64: 1, 512, 1>}> : (tensor<1x1x1xf32>) -> tensor<1x512x1xf32>
        %1112 = "ttir.div"(%1109, %1111) : (tensor<1x512x1xf32>, tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %1113 = "ttir.broadcast"(%1103) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<1x512x1xf32>) -> tensor<1x512x768xf32>
        %1114 = "ttir.subtract"(%1097, %1113) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %1115 = "ttir.reshape"(%6) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %1116 = "ttir.broadcast"(%1115) <{broadcast_dimensions = array<i64: 1, 512, 1>}> : (tensor<1x1x1xf32>) -> tensor<1x512x1xf32>
        %1117 = "ttir.add"(%1112, %1116) : (tensor<1x512x1xf32>, tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %1118 = "ttir.sqrt"(%1117) : (tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %1119 = "ttir.broadcast"(%1118) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<1x512x1xf32>) -> tensor<1x512x768xf32>
        %1120 = "ttir.div"(%1114, %1119) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %1121 = "ttir.reshape"(%arg95) <{shape = [1 : i32, 1 : i32, 768 : i32]}> : (tensor<768xf32>) -> tensor<1x1x768xf32>
        %1122 = "ttir.broadcast"(%1121) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x1x768xf32>
        %1123 = "ttir.broadcast"(%1122) <{broadcast_dimensions = array<i64: 1, 512, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x512x768xf32>
        %1124 = "ttir.multiply"(%1120, %1123) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %1125 = "ttir.reshape"(%arg96) <{shape = [1 : i32, 1 : i32, 768 : i32]}> : (tensor<768xf32>) -> tensor<1x1x768xf32>
        %1126 = "ttir.broadcast"(%1125) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x1x768xf32>
        %1127 = "ttir.broadcast"(%1126) <{broadcast_dimensions = array<i64: 1, 512, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x512x768xf32>
        %1128 = "ttir.add"(%1124, %1127) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %1129 = "ttir.dot_general"(%1128, %arg91) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<1x512x768xf32>, tensor<768x768xf32>) -> tensor<1x512x768xf32>
        %1130 = "ttir.dot_general"(%1128, %arg92) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<1x512x768xf32>, tensor<768x768xf32>) -> tensor<1x512x768xf32>
        %1131 = "ttir.dot_general"(%1128, %arg93) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<1x512x768xf32>, tensor<768x768xf32>) -> tensor<1x512x768xf32>
        %1132 = "ttir.reshape"(%1129) <{shape = [1 : i32, 512 : i32, 12 : i32, 64 : i32]}> : (tensor<1x512x768xf32>) -> tensor<1x512x12x64xf32>
        %1133 = "ttir.permute"(%1132) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x512x12x64xf32>) -> tensor<1x12x512x64xf32>
        %1134 = "ttir.reshape"(%1130) <{shape = [1 : i32, 512 : i32, 12 : i32, 64 : i32]}> : (tensor<1x512x768xf32>) -> tensor<1x512x12x64xf32>
        %1135 = "ttir.permute"(%1134) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x512x12x64xf32>) -> tensor<1x12x512x64xf32>
        %1136 = "ttir.reshape"(%1131) <{shape = [1 : i32, 512 : i32, 12 : i32, 64 : i32]}> : (tensor<1x512x768xf32>) -> tensor<1x512x12x64xf32>
        %1137 = "ttir.permute"(%1136) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x512x12x64xf32>) -> tensor<1x12x512x64xf32>
        %1138 = "ttir.permute"(%1135) <{permutation = array<i64: 0, 1, 3, 2>}> : (tensor<1x12x512x64xf32>) -> tensor<1x12x64x512xf32>
        %1139 = "ttir.reshape"(%1133) <{shape = [12 : i32, 512 : i32, 64 : i32]}> : (tensor<1x12x512x64xf32>) -> tensor<12x512x64xf32>
        %1140 = "ttir.dot_general"(%1139, %1138) <{batch_dims_lhs = array<i64: 0>, batch_dims_rhs = array<i64: 1>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 2>}> : (tensor<12x512x64xf32>, tensor<1x12x64x512xf32>) -> tensor<12x512x1x512xf32>
        %1141 = "ttir.permute"(%1140) <{permutation = array<i64: 2, 0, 1, 3>}> : (tensor<12x512x1x512xf32>) -> tensor<1x12x512x512xf32>
        %1142 = "ttir.sqrt"(%5) : (tensor<f32>) -> tensor<f32>
        %1143 = "ttir.typecast"(%1142) <{conservative_folding = false}> : (tensor<f32>) -> tensor<f32>
        %1144 = "ttir.reshape"(%1143) <{shape = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1x1xf32>
        %1145 = "ttir.broadcast"(%1144) <{broadcast_dimensions = array<i64: 1, 12, 512, 512>}> : (tensor<1x1x1x1xf32>) -> tensor<1x12x512x512xf32>
        %1146 = "ttir.div"(%1141, %1145) : (tensor<1x12x512x512xf32>, tensor<1x12x512x512xf32>) -> tensor<1x12x512x512xf32>
        %1147 = "ttir.max"(%1146) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<1x12x512x512xf32>) -> tensor<1x12x512xf32>
        %1148 = "ttir.reshape"(%4) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %1149 = "ttir.broadcast"(%1148) <{broadcast_dimensions = array<i64: 1, 12, 512>}> : (tensor<1x1x1xf32>) -> tensor<1x12x512xf32>
        %1150 = "ttir.maximum"(%1149, %1147) : (tensor<1x12x512xf32>, tensor<1x12x512xf32>) -> tensor<1x12x512xf32>
        %1151 = "ttir.reshape"(%1150) <{shape = [1 : i32, 12 : i32, 512 : i32, 1 : i32]}> : (tensor<1x12x512xf32>) -> tensor<1x12x512x1xf32>
        %1152 = "ttir.broadcast"(%1151) <{broadcast_dimensions = array<i64: 1, 1, 1, 1>}> : (tensor<1x12x512x1xf32>) -> tensor<1x12x512x1xf32>
        %1153 = "ttir.broadcast"(%1152) <{broadcast_dimensions = array<i64: 1, 1, 1, 512>}> : (tensor<1x12x512x1xf32>) -> tensor<1x12x512x512xf32>
        %1154 = "ttir.subtract"(%1146, %1153) : (tensor<1x12x512x512xf32>, tensor<1x12x512x512xf32>) -> tensor<1x12x512x512xf32>
        %1155 = "ttir.exp"(%1154) : (tensor<1x12x512x512xf32>) -> tensor<1x12x512x512xf32>
        %1156 = "ttir.sum"(%1155) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<1x12x512x512xf32>) -> tensor<1x12x512xf32>
        %1157 = "ttir.reshape"(%1156) <{shape = [1 : i32, 12 : i32, 512 : i32, 1 : i32]}> : (tensor<1x12x512xf32>) -> tensor<1x12x512x1xf32>
        %1158 = "ttir.broadcast"(%1157) <{broadcast_dimensions = array<i64: 1, 1, 1, 1>}> : (tensor<1x12x512x1xf32>) -> tensor<1x12x512x1xf32>
        %1159 = "ttir.broadcast"(%1158) <{broadcast_dimensions = array<i64: 1, 1, 1, 512>}> : (tensor<1x12x512x1xf32>) -> tensor<1x12x512x512xf32>
        %1160 = "ttir.div"(%1155, %1159) : (tensor<1x12x512x512xf32>, tensor<1x12x512x512xf32>) -> tensor<1x12x512x512xf32>
        %1161 = "ttir.reshape"(%1160) <{shape = [12 : i32, 512 : i32, 512 : i32]}> : (tensor<1x12x512x512xf32>) -> tensor<12x512x512xf32>
        %1162 = "ttir.dot_general"(%1161, %1137) <{batch_dims_lhs = array<i64: 0>, batch_dims_rhs = array<i64: 1>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 2>}> : (tensor<12x512x512xf32>, tensor<1x12x512x64xf32>) -> tensor<12x512x1x64xf32>
        %1163 = "ttir.permute"(%1162) <{permutation = array<i64: 2, 0, 1, 3>}> : (tensor<12x512x1x64xf32>) -> tensor<1x12x512x64xf32>
        %1164 = "ttir.permute"(%1163) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x12x512x64xf32>) -> tensor<1x512x12x64xf32>
        %1165 = "ttir.reshape"(%1164) <{shape = [1 : i32, 512 : i32, 768 : i32]}> : (tensor<1x512x12x64xf32>) -> tensor<1x512x768xf32>
        %1166 = "ttir.dot_general"(%1165, %arg94) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<1x512x768xf32>, tensor<768x768xf32>) -> tensor<1x512x768xf32>
        %1167 = "ttir.add"(%1097, %1166) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %1168 = "ttir.sum"(%1167) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x512x768xf32>) -> tensor<1x512xf32>
        %1169 = "ttir.reshape"(%1168) <{shape = [1 : i32, 512 : i32, 1 : i32]}> : (tensor<1x512xf32>) -> tensor<1x512x1xf32>
        %1170 = "ttir.broadcast"(%1169) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %1171 = "ttir.reshape"(%7) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %1172 = "ttir.broadcast"(%1171) <{broadcast_dimensions = array<i64: 1, 512, 1>}> : (tensor<1x1x1xf32>) -> tensor<1x512x1xf32>
        %1173 = "ttir.div"(%1170, %1172) : (tensor<1x512x1xf32>, tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %1174 = "ttir.broadcast"(%1173) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<1x512x1xf32>) -> tensor<1x512x768xf32>
        %1175 = "ttir.subtract"(%1167, %1174) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %1176 = "ttir.multiply"(%1175, %1175) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %1177 = "ttir.sum"(%1176) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x512x768xf32>) -> tensor<1x512xf32>
        %1178 = "ttir.reshape"(%1177) <{shape = [1 : i32, 512 : i32, 1 : i32]}> : (tensor<1x512xf32>) -> tensor<1x512x1xf32>
        %1179 = "ttir.broadcast"(%1178) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %1180 = "ttir.reshape"(%7) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %1181 = "ttir.broadcast"(%1180) <{broadcast_dimensions = array<i64: 1, 512, 1>}> : (tensor<1x1x1xf32>) -> tensor<1x512x1xf32>
        %1182 = "ttir.div"(%1179, %1181) : (tensor<1x512x1xf32>, tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %1183 = "ttir.broadcast"(%1173) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<1x512x1xf32>) -> tensor<1x512x768xf32>
        %1184 = "ttir.subtract"(%1167, %1183) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %1185 = "ttir.reshape"(%6) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %1186 = "ttir.broadcast"(%1185) <{broadcast_dimensions = array<i64: 1, 512, 1>}> : (tensor<1x1x1xf32>) -> tensor<1x512x1xf32>
        %1187 = "ttir.add"(%1182, %1186) : (tensor<1x512x1xf32>, tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %1188 = "ttir.sqrt"(%1187) : (tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %1189 = "ttir.broadcast"(%1188) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<1x512x1xf32>) -> tensor<1x512x768xf32>
        %1190 = "ttir.div"(%1184, %1189) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %1191 = "ttir.reshape"(%arg99) <{shape = [1 : i32, 1 : i32, 768 : i32]}> : (tensor<768xf32>) -> tensor<1x1x768xf32>
        %1192 = "ttir.broadcast"(%1191) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x1x768xf32>
        %1193 = "ttir.broadcast"(%1192) <{broadcast_dimensions = array<i64: 1, 512, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x512x768xf32>
        %1194 = "ttir.multiply"(%1190, %1193) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %1195 = "ttir.reshape"(%arg100) <{shape = [1 : i32, 1 : i32, 768 : i32]}> : (tensor<768xf32>) -> tensor<1x1x768xf32>
        %1196 = "ttir.broadcast"(%1195) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x1x768xf32>
        %1197 = "ttir.broadcast"(%1196) <{broadcast_dimensions = array<i64: 1, 512, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x512x768xf32>
        %1198 = "ttir.add"(%1194, %1197) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %1199 = "ttir.dot_general"(%1198, %arg97) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<1x512x768xf32>, tensor<768x3072xf32>) -> tensor<1x512x3072xf32>
        %1200 = "ttir.multiply"(%1199, %1199) : (tensor<1x512x3072xf32>, tensor<1x512x3072xf32>) -> tensor<1x512x3072xf32>
        %1201 = "ttir.multiply"(%1200, %1199) : (tensor<1x512x3072xf32>, tensor<1x512x3072xf32>) -> tensor<1x512x3072xf32>
        %1202 = "ttir.reshape"(%3) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %1203 = "ttir.broadcast"(%1202) <{broadcast_dimensions = array<i64: 1, 512, 3072>}> : (tensor<1x1x1xf32>) -> tensor<1x512x3072xf32>
        %1204 = "ttir.multiply"(%1203, %1201) : (tensor<1x512x3072xf32>, tensor<1x512x3072xf32>) -> tensor<1x512x3072xf32>
        %1205 = "ttir.add"(%1199, %1204) : (tensor<1x512x3072xf32>, tensor<1x512x3072xf32>) -> tensor<1x512x3072xf32>
        %1206 = "ttir.reshape"(%2) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %1207 = "ttir.broadcast"(%1206) <{broadcast_dimensions = array<i64: 1, 512, 3072>}> : (tensor<1x1x1xf32>) -> tensor<1x512x3072xf32>
        %1208 = "ttir.multiply"(%1207, %1205) : (tensor<1x512x3072xf32>, tensor<1x512x3072xf32>) -> tensor<1x512x3072xf32>
        %1209 = "ttir.tanh"(%1208) : (tensor<1x512x3072xf32>) -> tensor<1x512x3072xf32>
        %1210 = "ttir.reshape"(%1) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %1211 = "ttir.broadcast"(%1210) <{broadcast_dimensions = array<i64: 1, 512, 3072>}> : (tensor<1x1x1xf32>) -> tensor<1x512x3072xf32>
        %1212 = "ttir.add"(%1211, %1209) : (tensor<1x512x3072xf32>, tensor<1x512x3072xf32>) -> tensor<1x512x3072xf32>
        %1213 = "ttir.reshape"(%0) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %1214 = "ttir.broadcast"(%1213) <{broadcast_dimensions = array<i64: 1, 512, 3072>}> : (tensor<1x1x1xf32>) -> tensor<1x512x3072xf32>
        %1215 = "ttir.multiply"(%1214, %1212) : (tensor<1x512x3072xf32>, tensor<1x512x3072xf32>) -> tensor<1x512x3072xf32>
        %1216 = "ttir.multiply"(%1199, %1215) : (tensor<1x512x3072xf32>, tensor<1x512x3072xf32>) -> tensor<1x512x3072xf32>
        %1217 = "ttir.dot_general"(%1216, %arg98) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<1x512x3072xf32>, tensor<3072x768xf32>) -> tensor<1x512x768xf32>
        %1218 = "ttir.add"(%1167, %1217) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %1219 = "ttir.sum"(%1218) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x512x768xf32>) -> tensor<1x512xf32>
        %1220 = "ttir.reshape"(%1219) <{shape = [1 : i32, 512 : i32, 1 : i32]}> : (tensor<1x512xf32>) -> tensor<1x512x1xf32>
        %1221 = "ttir.broadcast"(%1220) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %1222 = "ttir.reshape"(%7) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %1223 = "ttir.broadcast"(%1222) <{broadcast_dimensions = array<i64: 1, 512, 1>}> : (tensor<1x1x1xf32>) -> tensor<1x512x1xf32>
        %1224 = "ttir.div"(%1221, %1223) : (tensor<1x512x1xf32>, tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %1225 = "ttir.broadcast"(%1224) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<1x512x1xf32>) -> tensor<1x512x768xf32>
        %1226 = "ttir.subtract"(%1218, %1225) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %1227 = "ttir.multiply"(%1226, %1226) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %1228 = "ttir.sum"(%1227) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x512x768xf32>) -> tensor<1x512xf32>
        %1229 = "ttir.reshape"(%1228) <{shape = [1 : i32, 512 : i32, 1 : i32]}> : (tensor<1x512xf32>) -> tensor<1x512x1xf32>
        %1230 = "ttir.broadcast"(%1229) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %1231 = "ttir.reshape"(%7) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %1232 = "ttir.broadcast"(%1231) <{broadcast_dimensions = array<i64: 1, 512, 1>}> : (tensor<1x1x1xf32>) -> tensor<1x512x1xf32>
        %1233 = "ttir.div"(%1230, %1232) : (tensor<1x512x1xf32>, tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %1234 = "ttir.broadcast"(%1224) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<1x512x1xf32>) -> tensor<1x512x768xf32>
        %1235 = "ttir.subtract"(%1218, %1234) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %1236 = "ttir.reshape"(%6) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %1237 = "ttir.broadcast"(%1236) <{broadcast_dimensions = array<i64: 1, 512, 1>}> : (tensor<1x1x1xf32>) -> tensor<1x512x1xf32>
        %1238 = "ttir.add"(%1233, %1237) : (tensor<1x512x1xf32>, tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %1239 = "ttir.sqrt"(%1238) : (tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %1240 = "ttir.broadcast"(%1239) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<1x512x1xf32>) -> tensor<1x512x768xf32>
        %1241 = "ttir.div"(%1235, %1240) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %1242 = "ttir.reshape"(%arg105) <{shape = [1 : i32, 1 : i32, 768 : i32]}> : (tensor<768xf32>) -> tensor<1x1x768xf32>
        %1243 = "ttir.broadcast"(%1242) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x1x768xf32>
        %1244 = "ttir.broadcast"(%1243) <{broadcast_dimensions = array<i64: 1, 512, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x512x768xf32>
        %1245 = "ttir.multiply"(%1241, %1244) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %1246 = "ttir.reshape"(%arg106) <{shape = [1 : i32, 1 : i32, 768 : i32]}> : (tensor<768xf32>) -> tensor<1x1x768xf32>
        %1247 = "ttir.broadcast"(%1246) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x1x768xf32>
        %1248 = "ttir.broadcast"(%1247) <{broadcast_dimensions = array<i64: 1, 512, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x512x768xf32>
        %1249 = "ttir.add"(%1245, %1248) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %1250 = "ttir.dot_general"(%1249, %arg101) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<1x512x768xf32>, tensor<768x768xf32>) -> tensor<1x512x768xf32>
        %1251 = "ttir.dot_general"(%1249, %arg102) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<1x512x768xf32>, tensor<768x768xf32>) -> tensor<1x512x768xf32>
        %1252 = "ttir.dot_general"(%1249, %arg103) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<1x512x768xf32>, tensor<768x768xf32>) -> tensor<1x512x768xf32>
        %1253 = "ttir.reshape"(%1250) <{shape = [1 : i32, 512 : i32, 12 : i32, 64 : i32]}> : (tensor<1x512x768xf32>) -> tensor<1x512x12x64xf32>
        %1254 = "ttir.permute"(%1253) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x512x12x64xf32>) -> tensor<1x12x512x64xf32>
        %1255 = "ttir.reshape"(%1251) <{shape = [1 : i32, 512 : i32, 12 : i32, 64 : i32]}> : (tensor<1x512x768xf32>) -> tensor<1x512x12x64xf32>
        %1256 = "ttir.permute"(%1255) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x512x12x64xf32>) -> tensor<1x12x512x64xf32>
        %1257 = "ttir.reshape"(%1252) <{shape = [1 : i32, 512 : i32, 12 : i32, 64 : i32]}> : (tensor<1x512x768xf32>) -> tensor<1x512x12x64xf32>
        %1258 = "ttir.permute"(%1257) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x512x12x64xf32>) -> tensor<1x12x512x64xf32>
        %1259 = "ttir.permute"(%1256) <{permutation = array<i64: 0, 1, 3, 2>}> : (tensor<1x12x512x64xf32>) -> tensor<1x12x64x512xf32>
        %1260 = "ttir.reshape"(%1254) <{shape = [12 : i32, 512 : i32, 64 : i32]}> : (tensor<1x12x512x64xf32>) -> tensor<12x512x64xf32>
        %1261 = "ttir.dot_general"(%1260, %1259) <{batch_dims_lhs = array<i64: 0>, batch_dims_rhs = array<i64: 1>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 2>}> : (tensor<12x512x64xf32>, tensor<1x12x64x512xf32>) -> tensor<12x512x1x512xf32>
        %1262 = "ttir.permute"(%1261) <{permutation = array<i64: 2, 0, 1, 3>}> : (tensor<12x512x1x512xf32>) -> tensor<1x12x512x512xf32>
        %1263 = "ttir.sqrt"(%5) : (tensor<f32>) -> tensor<f32>
        %1264 = "ttir.typecast"(%1263) <{conservative_folding = false}> : (tensor<f32>) -> tensor<f32>
        %1265 = "ttir.reshape"(%1264) <{shape = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1x1xf32>
        %1266 = "ttir.broadcast"(%1265) <{broadcast_dimensions = array<i64: 1, 12, 512, 512>}> : (tensor<1x1x1x1xf32>) -> tensor<1x12x512x512xf32>
        %1267 = "ttir.div"(%1262, %1266) : (tensor<1x12x512x512xf32>, tensor<1x12x512x512xf32>) -> tensor<1x12x512x512xf32>
        %1268 = "ttir.max"(%1267) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<1x12x512x512xf32>) -> tensor<1x12x512xf32>
        %1269 = "ttir.reshape"(%4) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %1270 = "ttir.broadcast"(%1269) <{broadcast_dimensions = array<i64: 1, 12, 512>}> : (tensor<1x1x1xf32>) -> tensor<1x12x512xf32>
        %1271 = "ttir.maximum"(%1270, %1268) : (tensor<1x12x512xf32>, tensor<1x12x512xf32>) -> tensor<1x12x512xf32>
        %1272 = "ttir.reshape"(%1271) <{shape = [1 : i32, 12 : i32, 512 : i32, 1 : i32]}> : (tensor<1x12x512xf32>) -> tensor<1x12x512x1xf32>
        %1273 = "ttir.broadcast"(%1272) <{broadcast_dimensions = array<i64: 1, 1, 1, 1>}> : (tensor<1x12x512x1xf32>) -> tensor<1x12x512x1xf32>
        %1274 = "ttir.broadcast"(%1273) <{broadcast_dimensions = array<i64: 1, 1, 1, 512>}> : (tensor<1x12x512x1xf32>) -> tensor<1x12x512x512xf32>
        %1275 = "ttir.subtract"(%1267, %1274) : (tensor<1x12x512x512xf32>, tensor<1x12x512x512xf32>) -> tensor<1x12x512x512xf32>
        %1276 = "ttir.exp"(%1275) : (tensor<1x12x512x512xf32>) -> tensor<1x12x512x512xf32>
        %1277 = "ttir.sum"(%1276) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<1x12x512x512xf32>) -> tensor<1x12x512xf32>
        %1278 = "ttir.reshape"(%1277) <{shape = [1 : i32, 12 : i32, 512 : i32, 1 : i32]}> : (tensor<1x12x512xf32>) -> tensor<1x12x512x1xf32>
        %1279 = "ttir.broadcast"(%1278) <{broadcast_dimensions = array<i64: 1, 1, 1, 1>}> : (tensor<1x12x512x1xf32>) -> tensor<1x12x512x1xf32>
        %1280 = "ttir.broadcast"(%1279) <{broadcast_dimensions = array<i64: 1, 1, 1, 512>}> : (tensor<1x12x512x1xf32>) -> tensor<1x12x512x512xf32>
        %1281 = "ttir.div"(%1276, %1280) : (tensor<1x12x512x512xf32>, tensor<1x12x512x512xf32>) -> tensor<1x12x512x512xf32>
        %1282 = "ttir.reshape"(%1281) <{shape = [12 : i32, 512 : i32, 512 : i32]}> : (tensor<1x12x512x512xf32>) -> tensor<12x512x512xf32>
        %1283 = "ttir.dot_general"(%1282, %1258) <{batch_dims_lhs = array<i64: 0>, batch_dims_rhs = array<i64: 1>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 2>}> : (tensor<12x512x512xf32>, tensor<1x12x512x64xf32>) -> tensor<12x512x1x64xf32>
        %1284 = "ttir.permute"(%1283) <{permutation = array<i64: 2, 0, 1, 3>}> : (tensor<12x512x1x64xf32>) -> tensor<1x12x512x64xf32>
        %1285 = "ttir.permute"(%1284) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x12x512x64xf32>) -> tensor<1x512x12x64xf32>
        %1286 = "ttir.reshape"(%1285) <{shape = [1 : i32, 512 : i32, 768 : i32]}> : (tensor<1x512x12x64xf32>) -> tensor<1x512x768xf32>
        %1287 = "ttir.dot_general"(%1286, %arg104) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<1x512x768xf32>, tensor<768x768xf32>) -> tensor<1x512x768xf32>
        %1288 = "ttir.add"(%1218, %1287) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %1289 = "ttir.sum"(%1288) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x512x768xf32>) -> tensor<1x512xf32>
        %1290 = "ttir.reshape"(%1289) <{shape = [1 : i32, 512 : i32, 1 : i32]}> : (tensor<1x512xf32>) -> tensor<1x512x1xf32>
        %1291 = "ttir.broadcast"(%1290) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %1292 = "ttir.reshape"(%7) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %1293 = "ttir.broadcast"(%1292) <{broadcast_dimensions = array<i64: 1, 512, 1>}> : (tensor<1x1x1xf32>) -> tensor<1x512x1xf32>
        %1294 = "ttir.div"(%1291, %1293) : (tensor<1x512x1xf32>, tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %1295 = "ttir.broadcast"(%1294) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<1x512x1xf32>) -> tensor<1x512x768xf32>
        %1296 = "ttir.subtract"(%1288, %1295) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %1297 = "ttir.multiply"(%1296, %1296) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %1298 = "ttir.sum"(%1297) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x512x768xf32>) -> tensor<1x512xf32>
        %1299 = "ttir.reshape"(%1298) <{shape = [1 : i32, 512 : i32, 1 : i32]}> : (tensor<1x512xf32>) -> tensor<1x512x1xf32>
        %1300 = "ttir.broadcast"(%1299) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %1301 = "ttir.reshape"(%7) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %1302 = "ttir.broadcast"(%1301) <{broadcast_dimensions = array<i64: 1, 512, 1>}> : (tensor<1x1x1xf32>) -> tensor<1x512x1xf32>
        %1303 = "ttir.div"(%1300, %1302) : (tensor<1x512x1xf32>, tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %1304 = "ttir.broadcast"(%1294) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<1x512x1xf32>) -> tensor<1x512x768xf32>
        %1305 = "ttir.subtract"(%1288, %1304) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %1306 = "ttir.reshape"(%6) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %1307 = "ttir.broadcast"(%1306) <{broadcast_dimensions = array<i64: 1, 512, 1>}> : (tensor<1x1x1xf32>) -> tensor<1x512x1xf32>
        %1308 = "ttir.add"(%1303, %1307) : (tensor<1x512x1xf32>, tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %1309 = "ttir.sqrt"(%1308) : (tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %1310 = "ttir.broadcast"(%1309) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<1x512x1xf32>) -> tensor<1x512x768xf32>
        %1311 = "ttir.div"(%1305, %1310) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %1312 = "ttir.reshape"(%arg109) <{shape = [1 : i32, 1 : i32, 768 : i32]}> : (tensor<768xf32>) -> tensor<1x1x768xf32>
        %1313 = "ttir.broadcast"(%1312) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x1x768xf32>
        %1314 = "ttir.broadcast"(%1313) <{broadcast_dimensions = array<i64: 1, 512, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x512x768xf32>
        %1315 = "ttir.multiply"(%1311, %1314) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %1316 = "ttir.reshape"(%arg110) <{shape = [1 : i32, 1 : i32, 768 : i32]}> : (tensor<768xf32>) -> tensor<1x1x768xf32>
        %1317 = "ttir.broadcast"(%1316) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x1x768xf32>
        %1318 = "ttir.broadcast"(%1317) <{broadcast_dimensions = array<i64: 1, 512, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x512x768xf32>
        %1319 = "ttir.add"(%1315, %1318) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %1320 = "ttir.dot_general"(%1319, %arg107) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<1x512x768xf32>, tensor<768x3072xf32>) -> tensor<1x512x3072xf32>
        %1321 = "ttir.multiply"(%1320, %1320) : (tensor<1x512x3072xf32>, tensor<1x512x3072xf32>) -> tensor<1x512x3072xf32>
        %1322 = "ttir.multiply"(%1321, %1320) : (tensor<1x512x3072xf32>, tensor<1x512x3072xf32>) -> tensor<1x512x3072xf32>
        %1323 = "ttir.reshape"(%3) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %1324 = "ttir.broadcast"(%1323) <{broadcast_dimensions = array<i64: 1, 512, 3072>}> : (tensor<1x1x1xf32>) -> tensor<1x512x3072xf32>
        %1325 = "ttir.multiply"(%1324, %1322) : (tensor<1x512x3072xf32>, tensor<1x512x3072xf32>) -> tensor<1x512x3072xf32>
        %1326 = "ttir.add"(%1320, %1325) : (tensor<1x512x3072xf32>, tensor<1x512x3072xf32>) -> tensor<1x512x3072xf32>
        %1327 = "ttir.reshape"(%2) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %1328 = "ttir.broadcast"(%1327) <{broadcast_dimensions = array<i64: 1, 512, 3072>}> : (tensor<1x1x1xf32>) -> tensor<1x512x3072xf32>
        %1329 = "ttir.multiply"(%1328, %1326) : (tensor<1x512x3072xf32>, tensor<1x512x3072xf32>) -> tensor<1x512x3072xf32>
        %1330 = "ttir.tanh"(%1329) : (tensor<1x512x3072xf32>) -> tensor<1x512x3072xf32>
        %1331 = "ttir.reshape"(%1) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %1332 = "ttir.broadcast"(%1331) <{broadcast_dimensions = array<i64: 1, 512, 3072>}> : (tensor<1x1x1xf32>) -> tensor<1x512x3072xf32>
        %1333 = "ttir.add"(%1332, %1330) : (tensor<1x512x3072xf32>, tensor<1x512x3072xf32>) -> tensor<1x512x3072xf32>
        %1334 = "ttir.reshape"(%0) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %1335 = "ttir.broadcast"(%1334) <{broadcast_dimensions = array<i64: 1, 512, 3072>}> : (tensor<1x1x1xf32>) -> tensor<1x512x3072xf32>
        %1336 = "ttir.multiply"(%1335, %1333) : (tensor<1x512x3072xf32>, tensor<1x512x3072xf32>) -> tensor<1x512x3072xf32>
        %1337 = "ttir.multiply"(%1320, %1336) : (tensor<1x512x3072xf32>, tensor<1x512x3072xf32>) -> tensor<1x512x3072xf32>
        %1338 = "ttir.dot_general"(%1337, %arg108) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<1x512x3072xf32>, tensor<3072x768xf32>) -> tensor<1x512x768xf32>
        %1339 = "ttir.add"(%1288, %1338) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %1340 = "ttir.sum"(%1339) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x512x768xf32>) -> tensor<1x512xf32>
        %1341 = "ttir.reshape"(%1340) <{shape = [1 : i32, 512 : i32, 1 : i32]}> : (tensor<1x512xf32>) -> tensor<1x512x1xf32>
        %1342 = "ttir.broadcast"(%1341) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %1343 = "ttir.reshape"(%7) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %1344 = "ttir.broadcast"(%1343) <{broadcast_dimensions = array<i64: 1, 512, 1>}> : (tensor<1x1x1xf32>) -> tensor<1x512x1xf32>
        %1345 = "ttir.div"(%1342, %1344) : (tensor<1x512x1xf32>, tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %1346 = "ttir.broadcast"(%1345) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<1x512x1xf32>) -> tensor<1x512x768xf32>
        %1347 = "ttir.subtract"(%1339, %1346) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %1348 = "ttir.multiply"(%1347, %1347) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %1349 = "ttir.sum"(%1348) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x512x768xf32>) -> tensor<1x512xf32>
        %1350 = "ttir.reshape"(%1349) <{shape = [1 : i32, 512 : i32, 1 : i32]}> : (tensor<1x512xf32>) -> tensor<1x512x1xf32>
        %1351 = "ttir.broadcast"(%1350) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %1352 = "ttir.reshape"(%7) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %1353 = "ttir.broadcast"(%1352) <{broadcast_dimensions = array<i64: 1, 512, 1>}> : (tensor<1x1x1xf32>) -> tensor<1x512x1xf32>
        %1354 = "ttir.div"(%1351, %1353) : (tensor<1x512x1xf32>, tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %1355 = "ttir.broadcast"(%1345) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<1x512x1xf32>) -> tensor<1x512x768xf32>
        %1356 = "ttir.subtract"(%1339, %1355) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %1357 = "ttir.reshape"(%6) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %1358 = "ttir.broadcast"(%1357) <{broadcast_dimensions = array<i64: 1, 512, 1>}> : (tensor<1x1x1xf32>) -> tensor<1x512x1xf32>
        %1359 = "ttir.add"(%1354, %1358) : (tensor<1x512x1xf32>, tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %1360 = "ttir.sqrt"(%1359) : (tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %1361 = "ttir.broadcast"(%1360) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<1x512x1xf32>) -> tensor<1x512x768xf32>
        %1362 = "ttir.div"(%1356, %1361) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %1363 = "ttir.reshape"(%arg115) <{shape = [1 : i32, 1 : i32, 768 : i32]}> : (tensor<768xf32>) -> tensor<1x1x768xf32>
        %1364 = "ttir.broadcast"(%1363) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x1x768xf32>
        %1365 = "ttir.broadcast"(%1364) <{broadcast_dimensions = array<i64: 1, 512, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x512x768xf32>
        %1366 = "ttir.multiply"(%1362, %1365) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %1367 = "ttir.reshape"(%arg116) <{shape = [1 : i32, 1 : i32, 768 : i32]}> : (tensor<768xf32>) -> tensor<1x1x768xf32>
        %1368 = "ttir.broadcast"(%1367) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x1x768xf32>
        %1369 = "ttir.broadcast"(%1368) <{broadcast_dimensions = array<i64: 1, 512, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x512x768xf32>
        %1370 = "ttir.add"(%1366, %1369) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %1371 = "ttir.dot_general"(%1370, %arg111) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<1x512x768xf32>, tensor<768x768xf32>) -> tensor<1x512x768xf32>
        %1372 = "ttir.dot_general"(%1370, %arg112) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<1x512x768xf32>, tensor<768x768xf32>) -> tensor<1x512x768xf32>
        %1373 = "ttir.dot_general"(%1370, %arg113) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<1x512x768xf32>, tensor<768x768xf32>) -> tensor<1x512x768xf32>
        %1374 = "ttir.reshape"(%1371) <{shape = [1 : i32, 512 : i32, 12 : i32, 64 : i32]}> : (tensor<1x512x768xf32>) -> tensor<1x512x12x64xf32>
        %1375 = "ttir.permute"(%1374) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x512x12x64xf32>) -> tensor<1x12x512x64xf32>
        %1376 = "ttir.reshape"(%1372) <{shape = [1 : i32, 512 : i32, 12 : i32, 64 : i32]}> : (tensor<1x512x768xf32>) -> tensor<1x512x12x64xf32>
        %1377 = "ttir.permute"(%1376) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x512x12x64xf32>) -> tensor<1x12x512x64xf32>
        %1378 = "ttir.reshape"(%1373) <{shape = [1 : i32, 512 : i32, 12 : i32, 64 : i32]}> : (tensor<1x512x768xf32>) -> tensor<1x512x12x64xf32>
        %1379 = "ttir.permute"(%1378) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x512x12x64xf32>) -> tensor<1x12x512x64xf32>
        %1380 = "ttir.permute"(%1377) <{permutation = array<i64: 0, 1, 3, 2>}> : (tensor<1x12x512x64xf32>) -> tensor<1x12x64x512xf32>
        %1381 = "ttir.reshape"(%1375) <{shape = [12 : i32, 512 : i32, 64 : i32]}> : (tensor<1x12x512x64xf32>) -> tensor<12x512x64xf32>
        %1382 = "ttir.dot_general"(%1381, %1380) <{batch_dims_lhs = array<i64: 0>, batch_dims_rhs = array<i64: 1>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 2>}> : (tensor<12x512x64xf32>, tensor<1x12x64x512xf32>) -> tensor<12x512x1x512xf32>
        %1383 = "ttir.permute"(%1382) <{permutation = array<i64: 2, 0, 1, 3>}> : (tensor<12x512x1x512xf32>) -> tensor<1x12x512x512xf32>
        %1384 = "ttir.sqrt"(%5) : (tensor<f32>) -> tensor<f32>
        %1385 = "ttir.typecast"(%1384) <{conservative_folding = false}> : (tensor<f32>) -> tensor<f32>
        %1386 = "ttir.reshape"(%1385) <{shape = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1x1xf32>
        %1387 = "ttir.broadcast"(%1386) <{broadcast_dimensions = array<i64: 1, 12, 512, 512>}> : (tensor<1x1x1x1xf32>) -> tensor<1x12x512x512xf32>
        %1388 = "ttir.div"(%1383, %1387) : (tensor<1x12x512x512xf32>, tensor<1x12x512x512xf32>) -> tensor<1x12x512x512xf32>
        %1389 = "ttir.max"(%1388) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<1x12x512x512xf32>) -> tensor<1x12x512xf32>
        %1390 = "ttir.reshape"(%4) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %1391 = "ttir.broadcast"(%1390) <{broadcast_dimensions = array<i64: 1, 12, 512>}> : (tensor<1x1x1xf32>) -> tensor<1x12x512xf32>
        %1392 = "ttir.maximum"(%1391, %1389) : (tensor<1x12x512xf32>, tensor<1x12x512xf32>) -> tensor<1x12x512xf32>
        %1393 = "ttir.reshape"(%1392) <{shape = [1 : i32, 12 : i32, 512 : i32, 1 : i32]}> : (tensor<1x12x512xf32>) -> tensor<1x12x512x1xf32>
        %1394 = "ttir.broadcast"(%1393) <{broadcast_dimensions = array<i64: 1, 1, 1, 1>}> : (tensor<1x12x512x1xf32>) -> tensor<1x12x512x1xf32>
        %1395 = "ttir.broadcast"(%1394) <{broadcast_dimensions = array<i64: 1, 1, 1, 512>}> : (tensor<1x12x512x1xf32>) -> tensor<1x12x512x512xf32>
        %1396 = "ttir.subtract"(%1388, %1395) : (tensor<1x12x512x512xf32>, tensor<1x12x512x512xf32>) -> tensor<1x12x512x512xf32>
        %1397 = "ttir.exp"(%1396) : (tensor<1x12x512x512xf32>) -> tensor<1x12x512x512xf32>
        %1398 = "ttir.sum"(%1397) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<1x12x512x512xf32>) -> tensor<1x12x512xf32>
        %1399 = "ttir.reshape"(%1398) <{shape = [1 : i32, 12 : i32, 512 : i32, 1 : i32]}> : (tensor<1x12x512xf32>) -> tensor<1x12x512x1xf32>
        %1400 = "ttir.broadcast"(%1399) <{broadcast_dimensions = array<i64: 1, 1, 1, 1>}> : (tensor<1x12x512x1xf32>) -> tensor<1x12x512x1xf32>
        %1401 = "ttir.broadcast"(%1400) <{broadcast_dimensions = array<i64: 1, 1, 1, 512>}> : (tensor<1x12x512x1xf32>) -> tensor<1x12x512x512xf32>
        %1402 = "ttir.div"(%1397, %1401) : (tensor<1x12x512x512xf32>, tensor<1x12x512x512xf32>) -> tensor<1x12x512x512xf32>
        %1403 = "ttir.reshape"(%1402) <{shape = [12 : i32, 512 : i32, 512 : i32]}> : (tensor<1x12x512x512xf32>) -> tensor<12x512x512xf32>
        %1404 = "ttir.dot_general"(%1403, %1379) <{batch_dims_lhs = array<i64: 0>, batch_dims_rhs = array<i64: 1>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 2>}> : (tensor<12x512x512xf32>, tensor<1x12x512x64xf32>) -> tensor<12x512x1x64xf32>
        %1405 = "ttir.permute"(%1404) <{permutation = array<i64: 2, 0, 1, 3>}> : (tensor<12x512x1x64xf32>) -> tensor<1x12x512x64xf32>
        %1406 = "ttir.permute"(%1405) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x12x512x64xf32>) -> tensor<1x512x12x64xf32>
        %1407 = "ttir.reshape"(%1406) <{shape = [1 : i32, 512 : i32, 768 : i32]}> : (tensor<1x512x12x64xf32>) -> tensor<1x512x768xf32>
        %1408 = "ttir.dot_general"(%1407, %arg114) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<1x512x768xf32>, tensor<768x768xf32>) -> tensor<1x512x768xf32>
        %1409 = "ttir.add"(%1339, %1408) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %1410 = "ttir.sum"(%1409) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x512x768xf32>) -> tensor<1x512xf32>
        %1411 = "ttir.reshape"(%1410) <{shape = [1 : i32, 512 : i32, 1 : i32]}> : (tensor<1x512xf32>) -> tensor<1x512x1xf32>
        %1412 = "ttir.broadcast"(%1411) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %1413 = "ttir.reshape"(%7) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %1414 = "ttir.broadcast"(%1413) <{broadcast_dimensions = array<i64: 1, 512, 1>}> : (tensor<1x1x1xf32>) -> tensor<1x512x1xf32>
        %1415 = "ttir.div"(%1412, %1414) : (tensor<1x512x1xf32>, tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %1416 = "ttir.broadcast"(%1415) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<1x512x1xf32>) -> tensor<1x512x768xf32>
        %1417 = "ttir.subtract"(%1409, %1416) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %1418 = "ttir.multiply"(%1417, %1417) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %1419 = "ttir.sum"(%1418) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x512x768xf32>) -> tensor<1x512xf32>
        %1420 = "ttir.reshape"(%1419) <{shape = [1 : i32, 512 : i32, 1 : i32]}> : (tensor<1x512xf32>) -> tensor<1x512x1xf32>
        %1421 = "ttir.broadcast"(%1420) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %1422 = "ttir.reshape"(%7) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %1423 = "ttir.broadcast"(%1422) <{broadcast_dimensions = array<i64: 1, 512, 1>}> : (tensor<1x1x1xf32>) -> tensor<1x512x1xf32>
        %1424 = "ttir.div"(%1421, %1423) : (tensor<1x512x1xf32>, tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %1425 = "ttir.broadcast"(%1415) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<1x512x1xf32>) -> tensor<1x512x768xf32>
        %1426 = "ttir.subtract"(%1409, %1425) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %1427 = "ttir.reshape"(%6) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %1428 = "ttir.broadcast"(%1427) <{broadcast_dimensions = array<i64: 1, 512, 1>}> : (tensor<1x1x1xf32>) -> tensor<1x512x1xf32>
        %1429 = "ttir.add"(%1424, %1428) : (tensor<1x512x1xf32>, tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %1430 = "ttir.sqrt"(%1429) : (tensor<1x512x1xf32>) -> tensor<1x512x1xf32>
        %1431 = "ttir.broadcast"(%1430) <{broadcast_dimensions = array<i64: 1, 1, 768>}> : (tensor<1x512x1xf32>) -> tensor<1x512x768xf32>
        %1432 = "ttir.div"(%1426, %1431) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %1433 = "ttir.reshape"(%arg119) <{shape = [1 : i32, 1 : i32, 768 : i32]}> : (tensor<768xf32>) -> tensor<1x1x768xf32>
        %1434 = "ttir.broadcast"(%1433) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x1x768xf32>
        %1435 = "ttir.broadcast"(%1434) <{broadcast_dimensions = array<i64: 1, 512, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x512x768xf32>
        %1436 = "ttir.multiply"(%1432, %1435) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %1437 = "ttir.reshape"(%arg120) <{shape = [1 : i32, 1 : i32, 768 : i32]}> : (tensor<768xf32>) -> tensor<1x1x768xf32>
        %1438 = "ttir.broadcast"(%1437) <{broadcast_dimensions = array<i64: 1, 1, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x1x768xf32>
        %1439 = "ttir.broadcast"(%1438) <{broadcast_dimensions = array<i64: 1, 512, 1>}> : (tensor<1x1x768xf32>) -> tensor<1x512x768xf32>
        %1440 = "ttir.add"(%1436, %1439) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        %1441 = "ttir.dot_general"(%1440, %arg117) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<1x512x768xf32>, tensor<768x3072xf32>) -> tensor<1x512x3072xf32>
        %1442 = "ttir.multiply"(%1441, %1441) : (tensor<1x512x3072xf32>, tensor<1x512x3072xf32>) -> tensor<1x512x3072xf32>
        %1443 = "ttir.multiply"(%1442, %1441) : (tensor<1x512x3072xf32>, tensor<1x512x3072xf32>) -> tensor<1x512x3072xf32>
        %1444 = "ttir.reshape"(%3) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %1445 = "ttir.broadcast"(%1444) <{broadcast_dimensions = array<i64: 1, 512, 3072>}> : (tensor<1x1x1xf32>) -> tensor<1x512x3072xf32>
        %1446 = "ttir.multiply"(%1445, %1443) : (tensor<1x512x3072xf32>, tensor<1x512x3072xf32>) -> tensor<1x512x3072xf32>
        %1447 = "ttir.add"(%1441, %1446) : (tensor<1x512x3072xf32>, tensor<1x512x3072xf32>) -> tensor<1x512x3072xf32>
        %1448 = "ttir.reshape"(%2) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %1449 = "ttir.broadcast"(%1448) <{broadcast_dimensions = array<i64: 1, 512, 3072>}> : (tensor<1x1x1xf32>) -> tensor<1x512x3072xf32>
        %1450 = "ttir.multiply"(%1449, %1447) : (tensor<1x512x3072xf32>, tensor<1x512x3072xf32>) -> tensor<1x512x3072xf32>
        %1451 = "ttir.tanh"(%1450) : (tensor<1x512x3072xf32>) -> tensor<1x512x3072xf32>
        %1452 = "ttir.reshape"(%1) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %1453 = "ttir.broadcast"(%1452) <{broadcast_dimensions = array<i64: 1, 512, 3072>}> : (tensor<1x1x1xf32>) -> tensor<1x512x3072xf32>
        %1454 = "ttir.add"(%1453, %1451) : (tensor<1x512x3072xf32>, tensor<1x512x3072xf32>) -> tensor<1x512x3072xf32>
        %1455 = "ttir.reshape"(%0) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %1456 = "ttir.broadcast"(%1455) <{broadcast_dimensions = array<i64: 1, 512, 3072>}> : (tensor<1x1x1xf32>) -> tensor<1x512x3072xf32>
        %1457 = "ttir.multiply"(%1456, %1454) : (tensor<1x512x3072xf32>, tensor<1x512x3072xf32>) -> tensor<1x512x3072xf32>
        %1458 = "ttir.multiply"(%1441, %1457) : (tensor<1x512x3072xf32>, tensor<1x512x3072xf32>) -> tensor<1x512x3072xf32>
        %1459 = "ttir.dot_general"(%1458, %arg118) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 0>}> : (tensor<1x512x3072xf32>, tensor<3072x768xf32>) -> tensor<1x512x768xf32>
        %1460 = "ttir.add"(%1409, %1459) : (tensor<1x512x768xf32>, tensor<1x512x768xf32>) -> tensor<1x512x768xf32>
        return %1460 : tensor<1x512x768xf32>
      }
    }
  }
}

