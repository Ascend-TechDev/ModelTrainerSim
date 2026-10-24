"""MLA (multi-head latent attention) layer for Kimi 10T.

Workbook rows I6-I13 (+I71): q_a [h,rq], q_norm [rq], kv_a [h,rk+ro],
kv_norm [rk], q_b [rq, n*(kn+ro)], kv_b [rk, n*(kn+vd)],
o_proj [n*vd, h], and the v2 补档 gated linear g_proj [h, n*vd].
Per-layer params = 232,196,096 (deep/wide share attention dims).

FLOPs follow the R-column GEMMs (per mbs token): q_a, q_b, kv_a, kv_b,
QK and PV on s^2, plus g_proj and o_proj.
Activation tensors follow the N-column counts; the per-layer aggregate
uses row M32 = SUM(M6:M14)-M13+SUM(M71:M74) (QK score matrix excluded).
"""
from __future__ import annotations

from typing import Any

from ....core.base_module import BaseModule
from ....core.dtype import DType
from .accounting import GroupedParamBreakdown


class MLA(BaseModule):
    module_type = "MLA"

    def __init__(self, name: str, config: dict[str, Any]):
        super().__init__(name, config)
        self.hidden_size: int = config["hidden_size"]              # h
        self.num_heads: int = config["num_heads"]                  # n
        self.qk_nope_head_dim: int = config["qk_nope_head_dim"]    # kn
        self.qk_rope_head_dim: int = config["qk_rope_head_dim"]    # ro
        self.v_head_dim: int = config["v_head_dim"]                # vd
        self.q_lora_rank: int = config["q_lora_rank"]              # rq
        self.kv_lora_rank: int = config["kv_lora_rank"]            # rk
        self.dtype: DType = DType(config.get("dtype", "bf16"))
        self.bytes: dict[str, float] = config["bytes"]

    def compute_params(self):
        H, N = self.hidden_size, self.num_heads
        rq, rk = self.q_lora_rank, self.kv_lora_rank
        kn, ro, vd = self.qk_nope_head_dim, self.qk_rope_head_dim, self.v_head_dim
        bf = self.dtype
        return [
            GroupedParamBreakdown("q_a_proj", (H, rq), H * rq, bf, group="muon", shard="tp"),        # I6
            GroupedParamBreakdown("q_norm", (rq,), rq, bf, group="muon", shard="tp"),                # I7
            GroupedParamBreakdown("kv_a_proj", (H, rk + ro), H * (rk + ro), bf,
                                  group="muon", shard="tp"),                                          # I8
            GroupedParamBreakdown("kv_norm", (rk,), rk, bf, group="muon", shard="tp"),               # I9
            GroupedParamBreakdown("q_b_proj", (rq, N * (kn + ro)), rq * N * (kn + ro), bf,
                                  group="muon", shard="tp"),                                         # I10
            GroupedParamBreakdown("kv_b_proj", (rk, N * (kn + vd)), rk * N * (kn + vd), bf,
                                  group="muon", shard="tp"),                                         # I11
            GroupedParamBreakdown("o_proj", (N * vd, H), N * vd * H, bf,
                                  group="muon", shard="tp"),                                         # I12
            GroupedParamBreakdown("g_proj", (H, N * vd), H * N * vd, bf,
                                  group="muon", shard="tp"),                                         # I71 补档
        ]

    def compute_flops(self, seq_len: int, batch_size: int = 1) -> float:
        B, S = batch_size, seq_len
        H, N = self.hidden_size, self.num_heads
        rq, rk = self.q_lora_rank, self.kv_lora_rank
        kn, ro, vd = self.qk_nope_head_dim, self.qk_rope_head_dim, self.v_head_dim
        q_a = 2 * B * S * rq * H                                  # R6
        q_b = 2 * B * S * N * (kn + ro) * rq                      # R10
        kv_a = 2 * B * S * (rk + ro) * H                          # R8
        kv_b = 2 * B * S * N * (kn + vd) * rk                     # R11
        qk = 2 * B * N * S * S * (kn + ro)                        # R13
        pv = 2 * B * N * S * S * (vd + ro)                        # R14
        gate = 2 * B * S * H * N * vd                             # R71
        out = 2 * B * S * N * vd * H                              # R12
        return q_a + q_b + kv_a + kv_b + qk + pv + gate + out

    def compute_activations(self, seq_len: int, batch_size: int = 1) -> float:
        """M32 = SUM(M6:M14)-M13+SUM(M71:M74)."""
        B, S = batch_size, seq_len
        H, N = self.hidden_size, self.num_heads
        rq, rk = self.q_lora_rank, self.kv_lora_rank
        kn, ro, vd = self.qk_nope_head_dim, self.qk_rope_head_dim, self.v_head_dim
        a = self.bytes["activation"]
        q_a_out = B * S * rq * a                                  # N6
        q_norm_out = B * S * rq * a                               # N7
        kv_a_out = B * S * (rk + ro) * a                          # N8
        kv_norm_out = B * S * rk * a                              # N9
        q_b_out = B * S * N * (kn + ro) * a                       # N10
        kv_b_out = B * S * N * (kn + vd) * a                      # N11
        residual = B * S * H * a                                  # N12
        pv_input = B * S * N * (vd + ro) * a                      # N14 (QK matrix N13 excluded)
        g_out = B * S * N * vd * a                                # N71
        gate = B * S * N * vd * a                                 # N72
        cats = B * S * N * (kn + ro) * 3 * a                      # N73
        o_out = B * S * H * a                                     # N74
        return (q_a_out + q_norm_out + kv_a_out + kv_norm_out + q_b_out + kv_b_out
                + residual + pv_input + g_out + gate + cats + o_out)
