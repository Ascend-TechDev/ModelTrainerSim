"""MoE — Mixture of Experts FFN module (DSV4Pro style).

DSV4Pro uses a fine-grained MoE with:
  - Shared experts (always-activated, dense)
  - Routed experts (top-k selection via gate)
  - rmsnorm-based input normalization
  - Grouped-Limited Expert Selection (GLSE) routing

Key parameters
--------------
hidden_size : int
intermediate_size : int   – per-expert intermediate dim
num_experts : int          – total routed experts
num_activated_experts : int – top-k experts activated per token
num_shared_experts : int   – always-on shared experts
moe_intermediate_size : int – MoE expert intermediate dim (if different)
expert_group_size : int    – GLSE group size for routing
"""

from __future__ import annotations

from typing import Any

from ....core.base_module import BaseModule, ModuleParamBreakdown
from ....core.dtype import DType


class MoE(BaseModule):
    module_type = "MoE"

    def __init__(self, name: str, config: dict[str, Any]):
        super().__init__(name, config)
        self.hidden_size: int = config["hidden_size"]
        self.intermediate_size: int = config.get("intermediate_size", 0)
        self.moe_intermediate_size: int = config.get(
            "moe_intermediate_size", self.intermediate_size)
        self.num_experts: int = config["num_experts"]
        self.num_activated_experts: int = config["num_activated_experts"]
        self.num_shared_experts: int = config.get("num_shared_experts", 0)
        self.expert_group_size: int = config.get("expert_group_size", 1)
        self.gate_dtype: DType = DType(config.get("gate_dtype", "fp32"))
        self.dtype: DType = DType(config.get("dtype", "bf16"))

    # -----------------------------------------------------------------
    # Parameters
    # -----------------------------------------------------------------

    def compute_params(self) -> list[ModuleParamBreakdown]:
        params: list[ModuleParamBreakdown] = []
        H = self.hidden_size
        I = self.moe_intermediate_size
        E = self.num_experts
        E_shared = self.num_shared_experts

        # --- Routed experts: each has (gate_proj, down_proj, up_proj) ---
        # gate_proj: H -> I  (SiLU)
        params.append(ModuleParamBreakdown(
            "routed_experts_gate_proj",
            (E, H, I), E * H * I, self.dtype))
        # down_proj: I -> H
        params.append(ModuleParamBreakdown(
            "routed_experts_down_proj",
            (E, I, H), E * I * H, self.dtype))
        # up_proj: H -> I  (SiLU)
        params.append(ModuleParamBreakdown(
            "routed_experts_up_proj",
            (E, H, I), E * H * I, self.dtype))

        # --- Shared experts (always activated) ---
        if E_shared > 0:
            params.append(ModuleParamBreakdown(
                "shared_experts_gate_proj",
                (E_shared, H, I), E_shared * H * I, self.dtype))
            params.append(ModuleParamBreakdown(
                "shared_experts_down_proj",
                (E_shared, I, H), E_shared * I * H, self.dtype))
            params.append(ModuleParamBreakdown(
                "shared_experts_up_proj",
                (E_shared, H, I), E_shared * H * I, self.dtype))

        # --- Router / gate ---
        params.append(ModuleParamBreakdown(
            "gate", (H, E), H * E, self.gate_dtype))

        # --- Norms ---
        params.append(ModuleParamBreakdown(
            "post_attention_layernorm", (H,), H, self.dtype))
        params.append(ModuleParamBreakdown(
            "input_layernorm", (H,), H, self.dtype))

        return params

    # -----------------------------------------------------------------
    # FLOPs
    # -----------------------------------------------------------------

    def compute_flops(self, seq_len: int, batch_size: int = 1) -> float:
        H = self.hidden_size
        I = self.moe_intermediate_size
        E_act = self.num_activated_experts
        E_shared = self.num_shared_experts
        B, S = batch_size, seq_len

        flops = 0.0

        # Routed experts (only activated experts process each token)
        # Each expert: gate(H->I) + up(H->I) + down(I->H)
        # 3 linear layers, each 2*H*I FLOPs per token
        expert_flops_per_token = 3 * 2 * H * I
        flops += B * S * E_act * expert_flops_per_token

        # Shared experts (always on)
        if E_shared > 0:
            flops += B * S * E_shared * expert_flops_per_token

        # Router gate
        flops += 2 * B * S * H * self.num_experts
        # Softmax + top-k selection
        flops += 3 * B * S * self.num_experts

        return flops

    # -----------------------------------------------------------------
    # Activations
    # -----------------------------------------------------------------

    def compute_activations(self, seq_len: int, batch_size: int = 1) -> float:
        from ....core.dtype import get_dtype_size
        B, S = batch_size, seq_len
        H = self.hidden_size
        I = self.moe_intermediate_size
        E_act = self.num_activated_experts
        E_shared = self.num_shared_experts
        elem = get_dtype_size(self.dtype)
        gate_elem = get_dtype_size(self.gate_dtype)

        act = 0.0
        # Activated expert intermediate activations
        act += B * S * E_act * I * elem * 2   # gate_proj + up_proj outputs
        act += B * S * E_act * H * elem        # down_proj output
        # Shared experts
        if E_shared > 0:
            act += B * S * E_shared * I * elem * 2
            act += B * S * E_shared * H * elem
        # Gate logits
        act += B * S * self.num_experts * gate_elem
        # Residual
        act += B * S * H * elem
        return act

    # -----------------------------------------------------------------
    # Extra info
    # -----------------------------------------------------------------

    def _extra_info(self) -> dict[str, Any]:
        # Per-expert param count
        H = self.hidden_size
        I = self.moe_intermediate_size
        expert_params = 3 * H * I  # gate + down + up
        return {
            "total_expert_params": self.num_experts * expert_params,
            "per_expert_params": expert_params,
            "activated_expert_params": self.num_activated_experts * expert_params,
            "shared_expert_params": self.num_shared_experts * expert_params
                if self.num_shared_experts > 0 else 0,
            "activation_ratio": (self.num_activated_experts + self.num_shared_experts)
                / (self.num_experts + self.num_shared_experts),
        }