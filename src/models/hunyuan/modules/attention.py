"""Gated MLA attention (Hy4 backbone, one instance per layer).

Workbook Hy4-10T sheet, module row D9 and compute rows D63-D77.  Per layer:

  q_a [H, rq] -> q_norm [rq] -> q_b [rq, N*(kn+ro)]
  kv_a [H, rk+ro] -> kv_norm [rk] -> kv_b [rk, N*(kn+vd)]   (single KV head path)
  attention sink [N] — one learnable logit per head, appended to the QK row
  DSA: sparse attention with Ksel = min(index_topk, T); FLOPs are counted
       on the eager QK/PV path with T = S (full-sequence forward)
  gate_proj [H, N*vd] -> sigmoid -> elementwise gate on the attention output
  o_proj [N*vd, H]

GEMM FLOPs convention: 2*M*N*Q per multiplication ([M,Q] x [Q,N]).
"""
from __future__ import annotations

from typing import Any

from ....core.base_module import BaseModule, ModuleParamBreakdown
from ....core.dtype import DType
from .accounting import GroupedParamBreakdown


class GatedMLA(BaseModule):
    module_type = "MLA"

    def __init__(self, name: str, config: dict[str, Any]):
        super().__init__(name, config)
        self.hidden_size: int = config["hidden_size"]              # H
        self.num_heads: int = config["num_heads"]                  # N
        self.qk_nope_head_dim: int = config["qk_nope_head_dim"]    # kn
        self.qk_rope_head_dim: int = config["qk_rope_head_dim"]    # ro
        self.v_head_dim: int = config["v_head_dim"]                # vd
        self.q_lora_rank: int = config["q_lora_rank"]              # rq
        self.kv_lora_rank: int = config["kv_lora_rank"]            # rk
        self.dtype: DType = DType(config.get("dtype", "bf16"))
        self.bytes: dict[str, float] = config["bytes"]

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    def compute_params(self) -> list[ModuleParamBreakdown]:
        H, N = self.hidden_size, self.num_heads
        rq, rk = self.q_lora_rank, self.kv_lora_rank
        kn, ro, vd = self.qk_nope_head_dim, self.qk_rope_head_dim, self.v_head_dim
        return [
            GroupedParamBreakdown("q_a_proj", (H, rq), H * rq, self.dtype, group="dense"),
            GroupedParamBreakdown("q_norm", (rq,), rq, self.dtype, group="dense"),
            GroupedParamBreakdown("q_b_proj", (rq, N * (kn + ro)), rq * N * (kn + ro),
                                  self.dtype, group="dense"),
            GroupedParamBreakdown("kv_a_proj_with_mqa", (H, rk + ro), H * (rk + ro),
                                  self.dtype, group="dense"),
            GroupedParamBreakdown("kv_norm", (rk,), rk, self.dtype, group="dense"),
            GroupedParamBreakdown("kv_b_proj", (rk, N * (kn + vd)), rk * N * (kn + vd),
                                  self.dtype, group="dense"),
            GroupedParamBreakdown("attention_sink", (N,), N, self.dtype, group="dense"),
            GroupedParamBreakdown("gate_proj", (H, N * vd), H * N * vd, self.dtype, group="dense"),
            GroupedParamBreakdown("o_proj", (N * vd, H), N * vd * H, self.dtype, group="dense"),
        ]

    def compute_flops(self, seq_len: int, batch_size: int = 1) -> float:
        B, S, T = batch_size, seq_len, seq_len          # T = S, full-sequence forward
        H, N = self.hidden_size, self.num_heads
        rq, rk = self.q_lora_rank, self.kv_lora_rank
        kn, ro, vd = self.qk_nope_head_dim, self.qk_rope_head_dim, self.v_head_dim
        # Low-rank Q path
        q_a = 2 * B * S * rq * H
        q_b = 2 * B * S * (N * (kn + ro)) * rq
        # Low-rank KV path (single KV head; kv_b runs over T = S)
        kv_a = 2 * B * S * (rk + ro) * H
        kv_b = 2 * B * T * (N * (kn + vd)) * rk
        # DSA eager QK/PV: batch = N heads, [B,S,256] x [B,256,T] and back.
        qk = 2 * N * B * S * T * (kn + ro)
        pv = 2 * N * B * S * vd * T
        # Gated output path
        gate = 2 * B * S * (N * vd) * H
        out = 2 * B * S * H * (N * vd)
        return q_a + q_b + kv_a + kv_b + qk + pv + gate + out

    def compute_activations(self, seq_len: int, batch_size: int = 1) -> float:
        """Named tensors mirroring workbook rows D64-D76 (N77 = 1080.5 MiB)."""
        B, S = batch_size, seq_len
        H, N = self.hidden_size, self.num_heads
        rq, rk = self.q_lora_rank, self.kv_lora_rank
        kn, ro, vd = self.qk_nope_head_dim, self.qk_rope_head_dim, self.v_head_dim
        b = self.bytes
        a = b["activation"]
        q_resid = B * S * rq * a                      # q_a out / q_norm out, counted twice
        q_b_out = B * S * (N * (kn + ro)) * a
        kv_a_out = B * S * (rk + ro) * a
        kv_norm = B * S * rk * a                      # latent + rope cache view
        kv_b_out = B * S * (N * (kn + vd)) * a
        k_asm = B * S * (N * (kn + ro)) * a           # assembled K (V is a kv_b slice)
        attn_out = B * S * (N * vd) * a
        gate_out = B * S * (N * vd) * a               # gate_proj output
        gate_sig = B * S * (N * vd) * a               # sigmoid(gate)
        gated = B * S * (N * vd) * a                  # attn x gate
        o_out = B * S * H * a
        return (q_resid + q_resid + q_b_out + kv_a_out + kv_norm + kv_b_out
                + k_asm + attn_out + gate_out + gate_sig + gated + o_out)
