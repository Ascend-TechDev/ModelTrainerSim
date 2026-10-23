"""MoE layer (77 layers in the 10T topology).

Workbook Hy4-10T sheet, module row D13 and compute rows D101-D116:
  Router F.linear [H, E] -> sigmoid scores; e_score_correction_bias [E]
  (checkpoint buffer, counted in weights but not gradients/optimizer);
  Top-K selection (K_act = num_experts_per_tok); gather -> normalize by
  the per-token sum x routed_scaling_factor.
  Routed experts: gate_up [E, H, 2I] + down [E, I, H] with clamp (gate
  upper bound 10; up clamped to [-10, 10]); routing weight applied AFTER
  down_proj; index_add back to [B,S,H].
  Shared expert: gate_up [H, 2I] + down [I, H], no routing weight.

Routed expert FLOPs use the true per-expert token counts: sum over
selected experts of 2*N_e*... with sum(N_e) = B*S*K_act — no uniform
load assumption, parameters still counted for all E experts.
"""
from __future__ import annotations

from typing import Any

from ....core.base_module import BaseModule, ModuleParamBreakdown
from ....core.dtype import DType
from .accounting import GroupedParamBreakdown


class MoE(BaseModule):
    module_type = "MoE"

    def __init__(self, name: str, config: dict[str, Any]):
        super().__init__(name, config)
        self.hidden_size: int = config["hidden_size"]                # H
        self.n_routed_experts: int = config["n_routed_experts"]      # E
        self.num_experts_per_tok: int = config["num_experts_per_tok"]  # K_act
        self.n_shared_experts: int = config["n_shared_experts"]      # 1
        self.moe_intermediate_size: int = config["moe_intermediate_size"]  # I
        self.dtype: DType = DType(config.get("dtype", "bf16"))
        self.bytes: dict[str, float] = config["bytes"]

    def compute_params(self) -> list[ModuleParamBreakdown]:
        H, E, I = self.hidden_size, self.n_routed_experts, self.moe_intermediate_size
        return [
            GroupedParamBreakdown("router_weight", (H, E), H * E, self.dtype, group="dense"),
            GroupedParamBreakdown("e_score_correction_bias", (E,), E, self.dtype, group="buffer"),
            GroupedParamBreakdown("routed_gate_up_proj", (E, H, 2 * I), E * H * 2 * I,
                                  self.dtype, group="expert"),
            GroupedParamBreakdown("routed_down_proj", (E, I, H), E * I * H,
                                  self.dtype, group="expert"),
            GroupedParamBreakdown("shared_gate_up_proj", (H, 2 * I), H * 2 * I,
                                  self.dtype, group="dense"),
            GroupedParamBreakdown("shared_down_proj", (I, H), I * H, self.dtype, group="dense"),
        ]

    def compute_flops(self, seq_len: int, batch_size: int = 1) -> float:
        B, S = batch_size, seq_len
        H, E, I = self.hidden_size, self.n_routed_experts, self.moe_intermediate_size
        K = self.num_experts_per_tok
        router = 2 * B * S * E * H
        n_e = B * S * K                                   # sum of selected-expert tokens
        routed_up = 2 * n_e * (2 * I) * H
        routed_down = 2 * n_e * H * I
        shared = 3 * 2 * B * S * I * H                    # gate + up + down on all tokens
        return router + routed_up + routed_down + shared

    def compute_activations(self, seq_len: int, batch_size: int = 1) -> float:
        B, S = batch_size, seq_len
        H, E, I = self.hidden_size, self.n_routed_experts, self.moe_intermediate_size
        K = self.num_experts_per_tok
        b = self.bytes
        a, f = b["activation"], b["fp32"]
        n_e = B * S * K
        logits = B * S * E * b["logits"]                  # router logits FP32
        scores = B * S * E * b["logits"]                  # sigmoid scores FP32
        topk_ids = B * S * K * 8                          # int64 MoE IDs
        routing_w = B * S * K * f                         # normalized routing weights FP32
        routed_up = n_e * (2 * I) * a                     # [N_e, 2I]
        routed_act = n_e * I * a                          # clamp + SwiGLU
        routed_down = n_e * H * a                         # [N_e, H]
        weighted = n_e * H * a                            # x routing weights
        final = B * S * H * a                             # index_add result
        shared_up = B * S * (2 * I) * a
        shared_act = B * S * I * a
        shared_down = B * S * H * a
        summed = B * S * H * a                            # routed + shared
        return (logits + scores + topk_ids + routing_w + routed_up + routed_act
                + routed_down + weighted + final + shared_up + shared_act
                + shared_down + summed)

    def _extra_info(self) -> dict[str, Any]:
        """Per-token activated components, for the model-level active-params walk."""
        H, E, I = self.hidden_size, self.n_routed_experts, self.moe_intermediate_size
        return {
            "router_params": H * E,
            "bias_params": E,
            "per_routed_expert_params": H * 2 * I + I * H,
            "shared_expert_params": H * 2 * I + I * H,
        }
