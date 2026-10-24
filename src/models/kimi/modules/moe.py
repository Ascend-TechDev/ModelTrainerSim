"""Latent-MoE layer for Kimi 10T (workbook rows I20-I32).

  gate [h, E]                       I27 = D7*D9     (router on the hidden input)
  latent_moe_down [h, latent]       I30 = D9*D21
  latent_moe_up [latent, h]         I31 = D21*D9
  routed experts: E x (gate [latent, moe_h] + up [latent, moe_h]
                       + down [moe_h, latent])      I23 = SUM(I20:I22)
  shared experts: 2 x (gate [h, moe_h] + up [h, moe_h] + down [moe_h, h])
                                        I25 = D9*D11*2*3
Per-layer params I32 = I23*D7+I25+I27+I30+I31
  deep-v2  = 67,843,915,776;  wide-v2 = 129,113,260,032.

FLOPs follow R20/R22/R24/R26/R28 (router down, routed fc1 x2 branches,
fc2, router up, shared gate+up and down; per mbs token, sum over
top_k selected experts).  Activation aggregate uses row
M30 = SUM(M20:M29)-M26 (the router_expert_up output N26 excluded).
"""
from __future__ import annotations

from typing import Any

from ....core.base_module import BaseModule
from ....core.dtype import DType
from .accounting import GroupedParamBreakdown


class LatentMoE(BaseModule):
    module_type = "MoE"

    def __init__(self, name: str, config: dict[str, Any]):
        super().__init__(name, config)
        self.hidden_size: int = config["hidden_size"]                    # h
        self.n_routed_experts: int = config["n_routed_experts"]          # E
        self.n_shared_experts: int = config["n_shared_experts"]
        self.top_k: int = config["top_k"]
        self.latent_moe_size: int = config["latent_moe_size"]            # latent
        self.moe_intermediate_size: int = config["moe_intermediate_size"]  # moe_h
        self.dtype: DType = DType(config.get("dtype", "bf16"))
        self.bytes: dict[str, float] = config["bytes"]

    def compute_params(self):
        H, E = self.hidden_size, self.n_routed_experts
        latent, moe_h = self.latent_moe_size, self.moe_intermediate_size
        bf = self.dtype
        return [
            GroupedParamBreakdown("gate", (H, E), H * E, bf, group="muon", shard="tp"),       # I27
            GroupedParamBreakdown("latent_moe_down", (H, latent), H * latent, bf,
                                  group="muon", shard="tp"),                                   # I30
            GroupedParamBreakdown("latent_moe_up", (latent, H), latent * H, bf,
                                  group="muon", shard="tp"),                                   # I31
            GroupedParamBreakdown("routed_gate_proj", (E, latent, moe_h), E * latent * moe_h,
                                  bf, group="muon", shard="ep"),                               # I20
            GroupedParamBreakdown("routed_up_proj", (E, latent, moe_h), E * latent * moe_h,
                                  bf, group="muon", shard="ep"),                               # I21
            GroupedParamBreakdown("routed_down_proj", (E, moe_h, latent), E * moe_h * latent,
                                  bf, group="muon", shard="ep"),                               # I22
            GroupedParamBreakdown("shared_gate_up_proj", (H, self.n_shared_experts * 2 * moe_h),
                                  H * self.n_shared_experts * 2 * moe_h, bf,
                                  group="muon", shard="tp"),
            GroupedParamBreakdown("shared_down_proj",
                                  (self.n_shared_experts * moe_h, H),
                                  self.n_shared_experts * moe_h * H, bf,
                                  group="muon", shard="tp"),
        ]

    def compute_flops(self, seq_len: int, batch_size: int = 1) -> float:
        B, S = batch_size, seq_len
        H = self.hidden_size
        latent, moe_h = self.latent_moe_size, self.moe_intermediate_size
        K = self.top_k
        router_down = 2 * B * S * H * latent                      # R20
        routed_fc1 = 2 * B * S * K * latent * moe_h * 2           # R22 (gate+up branches)
        routed_fc2 = 2 * B * S * K * moe_h * latent               # R24
        router_up = 2 * B * S * latent * H                        # R26
        shared_up = 2 * B * S * H * moe_h * 2 * self.n_shared_experts  # R28 (gate+up)
        shared_down = 2 * B * S * self.n_shared_experts * moe_h * H
        return router_down + routed_fc1 + routed_fc2 + router_up + shared_up + shared_down

    def compute_activations(self, seq_len: int, batch_size: int = 1) -> float:
        """M30 = SUM(M20:M29)-M26."""
        B, S = batch_size, seq_len
        H = self.hidden_size
        latent, moe_h = self.latent_moe_size, self.moe_intermediate_size
        K = self.top_k
        a = self.bytes["activation"]
        router_down_out = B * S * latent * a                      # N20
        gathered = B * S * K * latent * a                         # N21
        fc1_out = B * S * K * moe_h * 2 * a                       # N22
        act_out = B * S * K * moe_h * a                           # N23
        fc2_out = B * S * K * latent * a                          # N24
        unpermute = B * S * latent * a                            # N25
        shared_up_out = B * S * moe_h * 2 * self.n_shared_experts * a  # N28
        shared_down_out = B * S * moe_h * 2 * a                   # N29
        return (router_down_out + gathered + fc1_out + act_out + fc2_out
                + unpermute + shared_up_out + shared_down_out)
