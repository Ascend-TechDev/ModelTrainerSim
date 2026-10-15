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
        # Per Excel rows 87-89: the MoE gate also has a bias (B24) and a
        # `tid2eid` mapping table of size vocab × num_activated_experts.
        self.vocab_size: int = config.get("vocab_size", 0)
        # DSV4Pro quantizes expert weights to FP4 while keeping gate/norms in bf16.
        # When expert_dtype is set, expert weight bytes use that dtype; if absent
        # (e.g. for unquantized training analysis), experts use self.dtype.
        self.expert_dtype: DType = DType(config.get("expert_dtype", self.dtype.value))

    # -----------------------------------------------------------------
    # Parameters
    # -----------------------------------------------------------------

    def compute_params(self) -> list[ModuleParamBreakdown]:
        params: list[ModuleParamBreakdown] = []
        H = self.hidden_size
        I = self.moe_intermediate_size
        E = self.num_experts
        E_shared = self.num_shared_experts
        # Expert weights may be quantized (e.g. FP4 in DSV4Pro inference)
        e_dtype = self.expert_dtype

        # --- Routed experts: each has (gate_proj, down_proj, up_proj) ---
        # gate_proj: H -> I  (SiLU)
        params.append(ModuleParamBreakdown(
            "routed_experts_gate_proj",
            (E, H, I), E * H * I, e_dtype))
        # down_proj: I -> H
        params.append(ModuleParamBreakdown(
            "routed_experts_down_proj",
            (E, I, H), E * I * H, e_dtype))
        # up_proj: H -> I  (SiLU)
        params.append(ModuleParamBreakdown(
            "routed_experts_up_proj",
            (E, H, I), E * H * I, e_dtype))

        # --- Shared experts (always activated) ---
        if E_shared > 0:
            params.append(ModuleParamBreakdown(
                "shared_experts_gate_proj",
                (E_shared, H, I), E_shared * H * I, e_dtype))
            params.append(ModuleParamBreakdown(
                "shared_experts_down_proj",
                (E_shared, I, H), E_shared * I * H, e_dtype))
            params.append(ModuleParamBreakdown(
                "shared_experts_up_proj",
                (E_shared, H, I), E_shared * H * I, e_dtype))

        # --- Router / gate (per Excel rows 87-89) ---
        #   gate weight: [H, E]              -> H * E   (per layer)
        #   gate bias:   [E]                 -> E       (per layer)
        #   tid2eid:     [vocab, n_activated] -> vocab * n_activated
        params.append(ModuleParamBreakdown(
            "gate", (H, E), H * E, self.gate_dtype))
        params.append(ModuleParamBreakdown(
            "gate_bias", (E,), E, self.gate_dtype))
        if self.vocab_size > 0:
            params.append(ModuleParamBreakdown(
                "tid2eid", (self.vocab_size, self.num_activated_experts),
                self.vocab_size * self.num_activated_experts, self.gate_dtype))

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

        # --- Excel reference (U17 theoretical-peak formula) ---
        # Excel U10/U13/U17 cells apply an additional (1 + n_act) factor on
        # top of O83, treating MoE as 1 shared + n_act routed = (1+n_act)
        # experts computed in parallel (theoretical peak FLOPs).
        #
        # U17 = U10*31 + U13*30 + O18 + U8*(1+B27)
        #     = 31*CSA + 30*HCA + 1*SWA + 7*61*MoE   (B27=n_act=6)
        #
        # O83 itself already contains the n_act scaling for routed experts
        # (O79/O80/O81 = 2·B·S·H·I·n_act). Multiplying by (1+n_act) again
        # yields the U-column "theoretical peak" estimate — DIFFERENT from
        # the R-column per-module summary (R95) which uses O83 alone.
        #
        # We adopt the U17 (theoretical-peak) convention so that
        # total_flops_per_sample matches Excel cell U17.
        expert_flops = (3 * (2 * B * S * H * I * E_act)        # routed (O79+O80+O81)
                        + (2 + 1 + 1) * B * S * I * E_shared)    # shared (P79+P80+P81)
        flops += expert_flops * (1 + E_act)                       # U-column (1+n_act) factor

        # Gate (R90 / Q90): = 2·B·S·H·n_act  per layer
        # (NOT ×num_experts — Excel uses n_act, matching activated-token routing)
        flops += 2 * B * S * H * E_act

        return flops

    # -----------------------------------------------------------------
    # Activations
    # -----------------------------------------------------------------

    def compute_activations(self, seq_len: int, batch_size: int = 1) -> float:
        """Activation memory matching Excel ``N82 = (L79+L80+L81)*B41`` per
        layer, scaled by ``(1 + n_activated)`` (shared + routed).

        Excel counts three expert-buffers per layer (moe_gate, moe_up,
        moe_down) of sizes ``B*S*I`` (×2) and ``B*S*H``, using **fp4 bytes**
        (``B41=1``) — i.e. activations are quantized to FP4 like the expert
        weights.  The ``(1 + n_activated)`` factor scales one shared expert
        + ``n_activated`` routed experts.  The per-layer value is then
        summed across all 61 layers by the model's excel_summary.
        """
        from ....core.dtype import get_dtype_size
        B, S = batch_size, seq_len
        H = self.hidden_size
        I = self.moe_intermediate_size
        E_act = self.num_activated_experts
        # FP4 byte width — matches Excel B41=1 (activation bytes)
        act_elem = get_dtype_size(self.expert_dtype)

        # L79 = B*S*I (moe_gate), L80 = B*S*I (moe_up), L81 = B*S*H (moe_down)
        per_layer = (B * S * I + B * S * I + B * S * H) * act_elem
        # (1 + n_activated) shared+routed scaling per layer
        return per_layer * (1 + E_act)

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