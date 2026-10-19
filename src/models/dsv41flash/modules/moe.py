"""V4.1-Flash SwiGLU routed experts, shared expert and routing gate."""

from __future__ import annotations

from typing import Any

from ....core.base_module import ModuleParamBreakdown
from .common import FlashModule, _dimensions, _positive, _term


class MoE(FlashModule):
    module_type = "MoE"
    block_accounting = True

    def __init__(self, name: str, config: dict[str, Any], *, layer_id: int, dspark: bool = False):
        super().__init__(name, config, layer_id=layer_id, dspark=dspark)
        arch = self.arch
        self.hidden_size = _positive(arch["dim"], "dim")
        self.intermediate_size = _positive(arch["moe_inter_dim"], "moe_inter_dim")
        self.num_experts = _positive(
            (arch.get("dspark_n_routed_experts", 0) or arch["n_routed_experts"])
            if dspark else arch["n_routed_experts"], "n_routed_experts",
        )
        self.num_activated_experts = _positive(
            (arch.get("dspark_n_activated_experts", 0) or arch["n_activated_experts"])
            if dspark else arch["n_activated_experts"], "n_activated_experts",
        )
        self.num_shared_experts = _positive(arch.get("n_shared_experts", 1), "n_shared_experts")
        if self.num_shared_experts != 1:
            raise ValueError("Reference V4.1-Flash MoE requires exactly one shared expert")
        if self.num_activated_experts > self.num_experts:
            raise ValueError("n_activated_experts cannot exceed n_routed_experts")
        self.vision_enabled = bool(arch.get("vision_n_layers", 0) > 0)
        parameter_layer_id = layer_id - arch["n_layers"] if dspark else layer_id
        self.parameter_prefix = f"{'mtp' if dspark else 'layers'}.{parameter_layer_id}.ffn"

    # -----------------------------------------------------------------
    # Parameters
    # -----------------------------------------------------------------

    def compute_params(self) -> list[ModuleParamBreakdown]:
        H, I, E = self.hidden_size, self.intermediate_size, self.num_experts
        prefix = self.parameter_prefix
        parameters = [
            self.parameter(f"{prefix}.gate.weight", (E, H)),
            self.parameter(f"{prefix}.gate.bias", (E,)),
        ]
        if self.vision_enabled:
            parameters.append(self.parameter(f"{prefix}.gate.bias_vl", (E,)))
        # Leading E describes the inventory of E independently owned experts.
        for weight, shape in (("w1", (I, H)), ("w2", (H, I)), ("w3", (I, H))):
            parameters.append(self.parameter(f"{prefix}.experts.*.{weight}.weight", (E, *shape), "expert"))
            parameters.append(self.parameter(f"{prefix}.shared_experts.{weight}.weight", shape))
        return parameters

    # -----------------------------------------------------------------
    # FLOPs
    # -----------------------------------------------------------------

    def compute_flops(self, seq_len: int, batch_size: int = 1) -> float:
        H, S, B = _dimensions(self.arch, seq_len, batch_size)
        I, E = self.intermediate_size, self.num_experts
        K, shared = self.num_activated_experts, self.num_shared_experts
        tokens = B * S
        # Router scores all E experts; three expert GEMMs execute for K + shared.
        return float(2 * tokens * H * E + 6 * tokens * H * I * (K + shared))

    # -----------------------------------------------------------------
    # Activations
    # -----------------------------------------------------------------

    def compute_activations(self, seq_len: int, batch_size: int = 1) -> float:
        return float(sum(term["bytes"] for term in self.compute_activation_terms(seq_len, batch_size)))

    def compute_activation_terms(self, seq_len: int, batch_size: int = 1) -> list[dict[str, Any]]:
        H, S, B = _dimensions(self.arch, seq_len, batch_size)
        I, E = self.intermediate_size, self.num_experts
        K, shared = self.num_activated_experts, self.num_shared_experts
        tokens = B * S
        element_bytes = self.activation_element_bytes
        return [
            _term("moe_input", (B, S, H), element_bytes),
            _term("router_scores", (B, S, E), 4),
            _term("selected_weights", (B, S, K), 4),
            _term("selected_indices", (B, S, K), 8),
            _term("routed_expert_inputs", (tokens * K, H), element_bytes),
            _term("gate_linear_fp32", (tokens * (K + shared), I), 4),
            _term("up_linear_fp32", (tokens * (K + shared), I), 4),
            _term("down_linear_inputs", (tokens * (K + shared), I), element_bytes),
            _term("moe_output", (B, S, H), element_bytes),
        ]

    # -----------------------------------------------------------------
    # Extra info
    # -----------------------------------------------------------------

    def compute_details(self, seq_len: int, batch_size: int = 1) -> dict[str, Any]:
        H, I, E = self.hidden_size, self.intermediate_size, self.num_experts
        K, shared = self.num_activated_experts, self.num_shared_experts
        return {
            "source": "inference/model.py:Gate, Expert, MoE, ModelArgs.get_moe_config",
            "layer_id": self.layer_id,
            "routed_experts": E,
            "active_experts": K,
            "shared_experts": shared,
            "expert_intermediate_size": I,
            "active_parameter_elements": 3 * H * I * (K + shared) + E * H + E * (1 + self.vision_enabled),
            "vision_router_bias_included": self.vision_enabled,
            "note": ("bias_vl is present in the parameter inventory but inactive for text tokens. "
                     "Correction biases select experts and are not routing weights. No TID2EID table. "
                     "FP4/FP8 checkpoint scales are excluded from logical trainable parameters. "
                     "Routing imbalance, capacity padding, and dispatch communication buffers are excluded."),
        }
