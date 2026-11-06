#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Qwen3.5-35B-A3B -- TTSIM workload (hybrid GatedDeltaNet + gated-GQA, sparse MoE).

Architecture pinned from HF ``Qwen/Qwen3.5-35B-A3B`` ``config.json``
(``model_type: qwen3_5_moe_text``):

  - vocab_size               : 248320 (padded)
  - hidden_size              : 2048
  - num_hidden_layers        : 40, arranged 10 x (3 x linear-attention + 1 x full-attention)
                               (``full_attention_interval: 4`` -> 30 GatedDeltaNet + 10 full attn)
  - full attention           : 16 Q-heads / 2 KV-heads (GQA), head_dim 256,
                               gated attention output (``attn_output_gate: true``),
                               partial RoPE (rotary dim 64/256), mrope interleaved [11,11,10],
                               rope_theta 1e7
  - linear attention (GDN)   : 16 key-heads x 128, 32 value-heads x 128,
                               depthwise causal conv, kernel 4, fp32 SSM state
  - MoE (every layer)        : 256 experts, 8 routed + 1 shared, expert intermediate 512,
                               router softmax + top-k with normalized top-k probs
  - RMSNorm (eps 1e-6), SiLU activations, untied LM head, MTP trained (not modeled here)

Performance-simulation abstractions (documented; see README.md in this directory):

  1. Layer multiplicity: only ONE linear-attention block and ONE full-attention block are
     instantiated and executed; op ``repeat_count`` scales them x30 / x10 respectively
     (same pattern as ``workloads/BasicLLM.py``).
  2. MoE routing models ACTIVE compute only: the top-8 experts (per token) are represented
     by 8 expert-FFN instances applied to the routed activations, plus the shared expert.
     Expert *dispatch/combine* data movement is not explicitly modeled.
  3. RoPE is approximated by elementwise cos/sin mixing over the full head_dim (true model
     rotates only 64/256 dims); op count/type is representative, cost is negligible.
  4. Full-attention is simulated over the current sequence window (prefill: L tokens; decode:
     a single step). A paged KV cache is not explicitly modeled.
  5. MTP (multi-token prediction) layers and the PLE/n-gram path of the Qwen3.5 family are
     not part of the main inference loop and are not simulated.
"""

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import math
import numpy as np
import ttsim.front.functional.op as F
import ttsim.front.functional.tensor_op as T
import ttsim.front.functional.sim_nn as SimNN
from ttsim.ops import SimTensor


# ----------------------------------------------------------------------------
class RMSNorm(SimNN.Module):
    """RMSNorm decomposition: x * rsqrt(mean(x^2) + eps) * weight"""

    def __init__(self, name, dim, eps=1e-6):
        super().__init__()
        self.name = name
        self.dim = dim
        self.eps = eps

        self.weight = F._from_shape(f'{name}.weight', shape=[dim], is_param=True)
        self.eps_t = F._from_data(f'{name}.eps', data=np.float32(eps), is_const=True)

        self.sq_op = F.Mul(f'{name}.sq')
        self.mean_op = F.Mean(f'{name}.mean', dim=-1)
        self.add_eps_op = F.Add(f'{name}.add_eps')
        self.sqrt_op = F.Sqrt(f'{name}.sqrt')
        self.recip_op = F.Reciprocal(f'{name}.recip')
        self.rstd_op = F.Mul(f'{name}.rstd_mul')
        self.scale_op = F.Mul(f'{name}.scale')

        super().link_op2module()

    def __call__(self, x):
        mu = self.mean_op(self.sq_op(x, x)).unsqueeze(-1)  # (B, L, 1)
        rstd = self.recip_op(self.sqrt_op(self.add_eps_op(mu, self.eps_t)))
        return self.scale_op(self.rstd_op(x, rstd), self.weight)  # (B, L, dim)


# ----------------------------------------------------------------------------
class MoEExpert(SimNN.Module):
    """SwiGLU expert FFN: down( silu(gate(x)) * up(x) )"""

    def __init__(self, name, dim, inter_dim):
        super().__init__()
        self.name = name
        self.gate_proj = SimNN.Linear(f'{name}.gate_proj', dim, inter_dim, bias=False)
        self.up_proj = SimNN.Linear(f'{name}.up_proj', dim, inter_dim, bias=False)
        self.down_proj = SimNN.Linear(f'{name}.down_proj', inter_dim, dim, bias=False)
        self.silu_sig_op = F.Sigmoid(f'{name}.silu_sig')
        self.silu_mul_op = F.Mul(f'{name}.silu_mul')
        self.h_mul_op = F.Mul(f'{name}.h_mul')
        super().link_op2module()

    def __call__(self, x):
        g = self.gate_proj(x)
        u = self.up_proj(x)
        h = self.h_mul_op(self.silu_mul_op(g, self.silu_sig_op(g)), u)
        return self.down_proj(h)

    def analytical_param_count(self, lvl=0):
        return 3 * self.gate_proj.in_features * self.gate_proj.out_features


# ----------------------------------------------------------------------------
class SparseMoE(SimNN.Module):
    """Qwen3.5 MoE: softmax router + top-k (normalized) + 1 shared expert.

    Perf abstraction: the k routed experts selected per token are represented by
    k expert-FFN instances (active-compute modeling, see module docstring).
    """

    def __init__(self, name, cfg):
        super().__init__()
        self.name = name
        self.hidden = cfg.get('hidden_size', 2048)
        self.num_experts = cfg.get('num_experts', 256)
        self.top_k = cfg.get('num_experts_per_tok', 8)
        self.moe_inter = cfg.get('moe_intermediate_size', 512)
        self.shared_inter = cfg.get('shared_expert_intermediate_size', 512)

        self.router = SimNN.Linear(f'{name}.router', self.hidden, self.num_experts, bias=False)
        self.softmax_op = F.Softmax(f'{name}.router_softmax')
        self.topk_op = F.topk(f'{name}.topk', k=self.top_k)
        self.wsum_op = F.Sum(f'{name}.topk_sum', dim=-1)
        self.wdiv_op = F.Div(f'{name}.topk_norm')

        # top_k active expert slots + 1 shared expert
        self.experts = SimNN.ModuleList(
            [MoEExpert(f'{name}.expert{i}', self.hidden, self.moe_inter) for i in range(self.top_k)]
        )
        self.shared_expert = MoEExpert(f'{name}.shared_expert', self.hidden, self.shared_inter)
        super().link_op2module()

    def __call__(self, x):
        logits = self.router(x)  # (B, L, E)
        probs = self.softmax_op(logits)
        top_vals, _top_idx = self.topk_op(probs)  # (B, L, k) x 2
        wsum = self.wsum_op(top_vals).unsqueeze(-1)  # (B, L, 1)
        wnorm = self.wdiv_op(top_vals, wsum)  # normalized top-k probs (norm_topk_prob)

        out = self.shared_expert(x)  # (B, L, H)
        for i in range(self.top_k):
            wi = wnorm[:, :, i:i + 1]  # (B, L, 1)
            ei = self.experts[i](x)  # (B, L, H)
            mulop = F.Mul(f'{self.name}.routeweight_mul{i}')
            addop = F.Add(f'{self.name}.routecombine_add{i}')
            mulop.set_module(self)
            addop.set_module(self)
            self._op_hndls[mulop.name] = mulop
            self._op_hndls[addop.name] = addop
            out = addop(out, mulop(wi, ei))
        return out

    def analytical_param_count(self, lvl=0):
        routed = self.num_experts * (3 * self.hidden * self.moe_inter)
        shared = 3 * self.hidden * self.shared_inter
        return self.hidden * self.num_experts + routed + shared


# ----------------------------------------------------------------------------
class GatedDeltaNet(SimNN.Module):
    """Linear-attention layer: Gated DeltaNet (Qwen3-Next lineage).

    Per timestep t (per v-head h):
        a_t     = sigmoid(in_proj_a(x)_t)          # decay gate  (B, Hv)
        beta_t  = sigmoid(in_proj_b(x)_t)          # beta gate   (B, Hv)
        S       = a_t * S + k_t x beta_t*(v_t - q_t.S)      # delta-rule state update
        y_t     = a_t * (q_t.S) + beta_t * (q_t.k_t) * (v_t - q_t.S)

    q: (B, Hv, dk) with dk = (Hk*dk_head)/Hv; k regrouped to (B, Hv, dk);
    v: (B, Hv, dv_head); S: (B, Hv, dk, dv_head).
    """

    def __init__(self, name, cfg):
        super().__init__()
        self.name = name
        self.hidden = cfg.get('hidden_size', 2048)
        self.n_v = cfg.get('linear_num_value_heads', 32)
        self.n_k = cfg.get('linear_num_key_heads', 16)
        self.dk_head = cfg.get('linear_key_head_dim', 128)
        self.dv_head = cfg.get('linear_value_head_dim', 128)
        self.conv_k = cfg.get('linear_conv_kernel_dim', 4)
        self.eps = cfg.get('rms_norm_eps', 1e-6)

        assert (self.n_k * self.dk_head) % self.n_v == 0, \
            f'GDN: (n_k*dk_head)={self.n_k * self.dk_head} must be divisible by n_v={self.n_v}'
        self.dk = (self.n_k * self.dk_head) // self.n_v  # per-v-head key dim (64)
        self.dv = self.n_v * self.dv_head  # value dim (4096)
        self.q_dim = self.n_v * self.dk  # regrouped q dim (2048)
        self.k_dim = self.n_k * self.dk_head  # (2048)
        self.qkv_dim = self.q_dim + self.k_dim + self.dv  # in_proj_qkv out (8192)

        self.in_proj_qkv = SimNN.Linear(f'{name}.in_proj_qkv', self.hidden, self.qkv_dim, bias=False)
        self.in_proj_z = SimNN.Linear(f'{name}.in_proj_z', self.hidden, self.dv, bias=False)
        self.in_proj_a = SimNN.Linear(f'{name}.in_proj_a', self.hidden, self.n_v, bias=False)
        self.in_proj_b = SimNN.Linear(f'{name}.in_proj_b', self.hidden, self.n_v, bias=False)

        # depthwise causal conv over the fused qkv projection
        self.conv1d_op = F.conv1d(f'{name}.conv1d', pads=[self.conv_k - 1, self.conv_k - 1],
                                  group=self.qkv_dim)
        self.conv1d_weight = F._from_shape(f'{name}.conv1d_weight',
                                          shape=[self.qkv_dim, 1, self.conv_k], is_param=True)
        self.conv1d_bias = F._from_shape(f'{name}.conv1d_bias', shape=[self.qkv_dim], is_param=True)

        self.qkv_split_op = F.Split(f'{name}.qkv_split', axis=2, count=3)
        self.split_sizes = F._from_data(f'{name}.qkv_split_sizes',
                                        data=np.array([self.q_dim, self.k_dim, self.dv],
                                                      dtype=np.int64), is_const=True)

        self.out_norm = RMSNorm(f'{name}.norm', self.dv, eps=self.eps)
        self.out_gate_sig_op = F.Sigmoid(f'{name}.out_gate_sig')
        self.out_gate_mul_op = F.Mul(f'{name}.out_gate_mul')
        self.out_proj = SimNN.Linear(f'{name}.out_proj', self.dv, self.hidden, bias=False)
        super().link_op2module()

    def __call__(self, x):
        B, L, _ = x.shape
        assert _ == self.hidden, f'{self.name}: input hidden dim mismatch'

        qkv = self.in_proj_qkv(x)  # (B, L, 8192)
        z = self.in_proj_z(x)  # (B, L, 4096)
        a_raw = self.in_proj_a(x)  # (B, L, Hv)
        b_raw = self.in_proj_b(x)  # (B, L, Hv)

        # causal depthwise conv on fused qkv (mamba2 pattern)
        qkv_conv = self.conv1d_op(
            qkv.transpose(1, 2), self.conv1d_weight, self.conv1d_bias
        ).transpose(1, 2)[:, :L, :]  # (B, L, 8192)

        q, k, v = self.qkv_split_op(qkv_conv, self.split_sizes)

        # recurrent state S: (B, Hv, dk, dv_head)
        S = F._from_shape(f'{self.name}.gdn_state', shape=[B, self.n_v, self.dk, self.dv_head])
        S.set_module(self)
        self._tensors[S.name] = S

        ys = []
        for t in range(L):
            q_t = q[:, t, :].reshape(B, self.n_v, self.dk)  # (B, Hv, dk)
            k_t = k[:, t, :].reshape(B, self.n_v, self.dk)  # k regrouped per v-head
            v_t = v[:, t, :].reshape(B, self.n_v, self.dv_head)  # (B, Hv, dv_head)

            # one-shot op handles -> fresh per timestep (mamba2 scan pattern)
            sig_a = F.Sigmoid(f'{self.name}.sig_a.{t}')
            sig_b = F.Sigmoid(f'{self.name}.sig_b.{t}')
            for op in (sig_a, sig_b):
                op.set_module(self)
                self._op_hndls[op.name] = op

            a_t = sig_a(a_raw[:, t, :]).unsqueeze(-1).unsqueeze(-1)  # (B, Hv, 1, 1)
            beta_t = sig_b(b_raw[:, t, :]).unsqueeze(-1).unsqueeze(-1)  # (B, Hv, 1, 1)

            kv_t = T.matmul(q_t.unsqueeze(2), S)  # (B, Hv, 1, dv_head)  <- q_t . S
            kk_t = T.matmul(q_t.unsqueeze(2), k_t.unsqueeze(3))  # (B, Hv, 1, 1)  <- q_t . k_t

            delta = v_t.unsqueeze(2) - kv_t  # (B, Hv, 1, dv_head)
            y_t = a_t * kv_t + beta_t * kk_t * delta  # (B, Hv, 1, dv_head)
            S = a_t * S + T.matmul(k_t.unsqueeze(3), beta_t * delta)  # (B, Hv, dk, dv_head)

            ys.append(y_t.reshape(B, 1, self.dv))

        y = ys[0] if L == 1 else T.cat(ys, dim=1)  # (B, L, dv)

        y = self.out_norm(y)
        y = self.out_gate_mul_op(y, self.out_gate_sig_op(z))  # norm(y) * silu(z)
        return self.out_proj(y)  # (B, L, hidden)

    def analytical_param_count(self, lvl=0):
        p = self.hidden * self.qkv_dim  # in_proj_qkv
        p += self.hidden * self.dv  # in_proj_z
        p += 2 * self.hidden * self.n_v  # in_proj_a + in_proj_b
        p += self.qkv_dim * self.conv_k + self.qkv_dim  # conv weight + bias
        p += self.dv * self.hidden  # out_proj
        p += self.dv  # out RMSNorm weight
        return p


# ----------------------------------------------------------------------------
class GatedGQAAttention(SimNN.Module):
    """Full-attention layer: GQA (16Q/2KV, head_dim 256) + per-head output gate.

    ``attn_output_gate: true`` -> out = ctx * sigmoid(wg(x)) with one scalar gate per
    Q-head. RoPE (partial 64/256, mrope interleaved) is approximated by elementwise
    cos/sin mixing (see module docstring, abstraction #3).
    """

    def __init__(self, name, cfg):
        super().__init__()
        self.name = name
        self.hidden = cfg.get('hidden_size', 2048)
        self.n_q = cfg.get('num_attention_heads', 16)
        self.n_kv = cfg.get('num_key_value_heads', 2)
        self.hd = cfg.get('head_dim', 256)
        self.seq_len = cfg.get('seq_len', 1)
        assert self.n_q % self.n_kv == 0, 'GQA: n_q must be divisible by n_kv'
        self.gqa_rep = self.n_q // self.n_kv

        self.wq = SimNN.Linear(f'{name}.q_proj', self.hidden, self.n_q * self.hd, bias=False)
        self.wk = SimNN.Linear(f'{name}.k_proj', self.hidden, self.n_kv * self.hd, bias=False)
        self.wv = SimNN.Linear(f'{name}.v_proj', self.hidden, self.n_kv * self.hd, bias=False)
        self.wg = SimNN.Linear(f'{name}.attn_output_gate', self.hidden, self.n_q, bias=False)
        self.wo = SimNN.Linear(f'{name}.o_proj', self.n_q * self.hd, self.hidden, bias=False)

        self.softmax_op = F.Softmax(f'{name}.softmax')
        self.gate_sig_op = F.Sigmoid(f'{name}.gate_sig')
        self.gate_mul_op = F.Mul(f'{name}.gate_mul')
        self.inv_sqrt_hd = F._from_data(f'{name}.inv_sqrt_hd',
                                        data=np.float32(1.0 / math.sqrt(self.hd)), is_const=True)
        # RoPE approximation constants (elementwise cos/sin mix, abstraction #3)
        self.rope_cos = F._from_shape(f'{name}.rope_cos', shape=[1, 1, 1, self.hd], is_const=True)
        self.rope_sin = F._from_shape(f'{name}.rope_sin', shape=[1, 1, 1, self.hd], is_const=True)
        if self.seq_len > 1:
            self.causal_mask = F._from_shape(f'{name}.causal_mask',
                                             shape=[1, 1, self.seq_len, self.seq_len], is_const=True)
        super().link_op2module()

    def __call__(self, x):
        B, L, _ = x.shape
        assert _ == self.hidden, f'{self.name}: input hidden dim mismatch'

        q = self.wq(x).reshape(B, L, self.n_q, self.hd).permute([0, 2, 1, 3])  # (B, nq, L, hd)
        k = self.wk(x).reshape(B, L, self.n_kv, self.hd).permute([0, 2, 1, 3])  # (B, nkv, L, hd)
        v = self.wv(x).reshape(B, L, self.n_kv, self.hd).permute([0, 2, 1, 3])

        # RoPE (approximated, full head_dim elementwise cos/sin mix)
        q = q * self.rope_cos + q * self.rope_sin
        k = k * self.rope_cos + k * self.rope_sin

        # GQA: expand KV heads
        k = k.repeat(1, self.gqa_rep, 1, 1)  # (B, nq, L, hd)
        v = v.repeat(1, self.gqa_rep, 1, 1)

        scores = T.matmul(q, k.transpose(2, 3)) * self.inv_sqrt_hd  # (B, nq, L, L)
        if L > 1:
            scores = scores + self.causal_mask
        probs = self.softmax_op(scores)  # (B, nq, L, L)

        ctx = T.matmul(probs, v)  # (B, nq, L, hd)
        ctx = ctx.permute([0, 2, 1, 3]).reshape(B, L, self.n_q, self.hd)  # (B, L, nq, hd)

        g = self.gate_sig_op(self.wg(x)).reshape(B, L, self.n_q, 1)  # (B, L, nq, 1)
        ctx = self.gate_mul_op(ctx, g)  # per-head output gate
        return self.wo(ctx.reshape(B, L, self.n_q * self.hd))  # (B, L, hidden)

    def analytical_param_count(self, lvl=0):
        p = 3 * self.hidden * (self.n_q * self.hd)  # wq + wk + wv (n_kv*hd == n_q*hd for this model)
        p += self.hidden * self.n_q  # output gate
        p += (self.n_q * self.hd) * self.hidden  # o_proj
        return p


# ----------------------------------------------------------------------------
class Qwen35HybridLayer(SimNN.Module):
    """One decoder layer: norm -> attn(GDN|GQA) -> residual -> norm -> MoE -> residual."""

    def __init__(self, name, cfg, layer_kind):
        super().__init__()
        assert layer_kind in ('linear_attention', 'full_attention'), \
            f'unknown layer kind: {layer_kind}'
        self.name = name
        self.layer_kind = layer_kind
        self.input_norm = RMSNorm(f'{name}.input_layernorm', cfg.get('hidden_size', 2048),
                                  eps=cfg.get('rms_norm_eps', 1e-6))
        self.attn: 'GatedDeltaNet | GatedGQAAttention'
        if layer_kind == 'linear_attention':
            self.attn = GatedDeltaNet(f'{name}.linear_attn', cfg)
        else:
            self.attn = GatedGQAAttention(f'{name}.attn', cfg)
        self.post_norm = RMSNorm(f'{name}.post_attention_layernorm', cfg.get('hidden_size', 2048),
                                 eps=cfg.get('rms_norm_eps', 1e-6))
        self.moe = SparseMoE(f'{name}.mlp', cfg)
        self.res1_op = F.Add(f'{name}.residual_add1')
        self.res2_op = F.Add(f'{name}.residual_add2')
        super().link_op2module()

    def __call__(self, x):
        h = self.res1_op(x, self.attn(self.input_norm(x)))
        return self.res2_op(h, self.moe(self.post_norm(h)))

    def analytical_param_count(self, lvl=0):
        return (self.attn.analytical_param_count(lvl + 1)
                + self.moe.analytical_param_count(lvl + 1)
                + 2 * self.input_norm.dim)


# ----------------------------------------------------------------------------
class Qwen35Moe(SimNN.Module):
    """Qwen3.5-35B-A3B top-level model (TTSIM entry class).

    Instantiated by polaris as ``Qwen35Moe(instance_name, merged_cfg)`` where
    ``merged_cfg`` carries the YAML ``params`` overridden by the instance entries.
    All Qwen3.5-35B-A3B defaults are built in; instances typically only override
    ``bs`` and ``seq_len``.
    """

    def __init__(self, name, cfg):
        super().__init__()
        self.name = name
        self.cfg = cfg
        self.vocab_size = cfg.get('vocab_size', 248320)
        self.hidden = cfg.get('hidden_size', 2048)
        self.num_layers = cfg.get('num_hidden_layers', 40)
        self.interval = cfg.get('full_attention_interval', 4)
        self.bs = cfg.get('bs', 1)
        self.seq_len = cfg.get('seq_len', 1)
        self.eps = cfg.get('rms_norm_eps', 1e-6)

        assert self.num_layers % self.interval == 0, \
            'num_hidden_layers must be divisible by full_attention_interval'
        self.n_full = self.num_layers // self.interval  # 10
        self.n_linear = self.num_layers - self.n_full  # 30

        # ops / modules
        self.wte = F.Embedding(f'{name}.wte', self.vocab_size, self.hidden)

        # proxy blocks: 1 linear-attention + 1 full-attention (repeat_count scales below)
        self.linear_block = Qwen35HybridLayer(f'{name}.linear_block', cfg, 'linear_attention')
        self.full_block = Qwen35HybridLayer(f'{name}.full_block', cfg, 'full_attention')

        self.final_norm = RMSNorm(f'{name}.final_norm', self.hidden, eps=self.eps)
        self.lm_head = SimNN.Linear(f'{name}.lm_head', self.hidden, self.vocab_size, bias=False)
        self.input_tensors = {}
        super().link_op2module()

    def create_input_tensors(self):
        self.input_tensors = {
            'tokens': F._from_shape('tokens', [self.bs, self.seq_len],
                                    is_param=False, np_dtype=np.int64),
        }
        return

    def get_forward_graph(self):
        GG = super()._get_forward_graph(self.input_tensors)
        return GG

    def __call__(self):
        assert len(self.input_tensors) == 1, \
            f'{self.name}: create_input_tensors() must be called before __call__()'

        Y = self.wte(self.input_tensors['tokens'])  # (B, L, hidden)
        Y = self.linear_block(Y)
        Y = self.full_block(Y)

        # scale the proxy blocks by layer multiplicity (BasicLLM repeat_count pattern)
        for blk, rpt in ((self.linear_block, self.n_linear), (self.full_block, self.n_full)):
            repeated_ops: dict = {}
            blk.get_ops(repeated_ops)
            for op_name, op_obj in repeated_ops.items():
                op_obj.repeat_count = rpt

        Y = self.final_norm(Y)
        Y = self.lm_head(Y)  # (B, L, vocab)
        return Y

    def analytical_param_count(self, lvl=0):
        lin = self.linear_block.analytical_param_count(lvl + 1)
        ful = self.full_block.analytical_param_count(lvl + 1)
        total = self.n_linear * lin + self.n_full * ful
        total += self.hidden  # final norm
        total += self.vocab_size * self.hidden  # embedding
        total += self.hidden * self.vocab_size  # untied lm_head
        return total


# ----------------------------------------------------------------------------
if __name__ == '__main__':
    # tiny smoke config
    tiny_cfg = {
        'vocab_size': 512, 'hidden_size': 64, 'num_hidden_layers': 8,
        'full_attention_interval': 4,
        'num_attention_heads': 4, 'num_key_value_heads': 2, 'head_dim': 32,
        'linear_num_value_heads': 4, 'linear_num_key_heads': 2,
        'linear_key_head_dim': 16, 'linear_value_head_dim': 16,
        'linear_conv_kernel_dim': 4,
        'num_experts': 8, 'num_experts_per_tok': 2,
        'moe_intermediate_size': 32, 'shared_expert_intermediate_size': 32,
        'bs': 1, 'seq_len': 4, 'rms_norm_eps': 1e-6,
    }
    model = Qwen35Moe('qwen35_tiny', tiny_cfg)
    model.create_input_tensors()
    out = model()
    print(f'tiny forward OK, output shape: {out.shape}')
    print(f'tiny param count: {model.analytical_param_count():,}')

    # full 35B-A3B decode step
    full_cfg = {'bs': 1, 'seq_len': 1}
    m35 = Qwen35Moe('qwen35_35b_a3b', full_cfg)
    m35.create_input_tensors()
    out = m35()
    print(f'35B-A3B decode forward OK, output shape: {out.shape}')
    print(f'35B-A3B total params: {m35.analytical_param_count():,}')