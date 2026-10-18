"""RoutedExperts — Sparse routed experts FFN (GLM-5.2 style).

GLM-5.2 MoE layers (75 layers) include ``n_routed_experts`` total
experts, of which ``num_experts_per_tok`` are activated per token via
top-k routing.

Per the Excel reference (sheet 'GLM-5.2-743B', rows 42-46):

  L42 = D13 * D15 * 2                 (= H * moe_intermediate * 2, gate_up fused)
  L43 = D15 * D13                      (= moe_intermediate * H, down_proj)
  L44 = SUM(L42:L43) * H43             (total params, all routed experts × moe_layers)
  H43 = D4 * D19                       (= moe_layer_count * n_routed_experts)
  L45 = SUM(L42:L43) * H44             (activated params)
  H44 = D4 * D21                       (= moe_layer_count * num_experts_per_tok)

Activation rows (column U), per moe-layer:
  U42 = B*S*topk*H          (permute: [B,S,H] -> [B,S*topk,H])
  U43 = B*S*topk*moe_intermediate*2   (gate_up output)
  U44 = B*S*topk*moe_intermediate     (SwiGLU)
  U45 = B*S*topk*moe_intermediate     (probs scaling)
  U46 = B*S*topk*H           (down_proj output)

Key parameters (from GLM ``config.json``)
--------------------------------------------
hidden_size : int
moe_intermediate_size : int       – 2048
n_routed_experts : int            – 256
num_experts_per_tok : int         – 8
num_moe_layers : int              – 75
"""

from __future__ import annotations

from typing import Any

from ....core.base_module import BaseModule, ModuleParamBreakdown
from ....core.dtype import DType, get_dtype_size


class RoutedExperts(BaseModule):
    """Sparse routed experts FFN (GLM-5.2)."""

    module_type = "RoutedExperts"

    def __init__(self, name: str, config: dict[str, Any]):
        super().__init__(name, config)
        self.hidden_size: int = config["hidden_size"]
        self.moe_intermediate_size: int = config["moe_intermediate_size"]
        self.n_routed_experts: int = config.get("n_routed_experts", 1)
        self.num_experts_per_tok: int = config.get("num_experts_per_tok", 1)
        self.dtype: DType = DType(config.get("dtype", "bf16"))

    # -----------------------------------------------------------------
    # Parameters — Excel rows 42-43
    #   Returns params for ONE routed expert.  The model top-level
    #   multiplies by n_routed_experts * num_moe_layers (Excel H43).
    # -----------------------------------------------------------------

    def compute_params(self) -> list[ModuleParamBreakdown]:
        H = self.hidden_size
        I = self.moe_intermediate_size
        return [
            ModuleParamBreakdown("gate_up_proj", (H, I * 2), H * I * 2, self.dtype),
            ModuleParamBreakdown("down_proj", (I, H), I * H, self.dtype),
        ]

    # -----------------------------------------------------------------
    # FLOPs — derived from projection shapes (Excel does not provide cells).
    #   Per moe-layer, per token the activated top-k experts compute:
    #     gate_up: 2 * B * S * topk * H * (moe_intermediate*2)
    #     swiglu: 3 * B * S * topk * moe_intermediate
    #     down:   2 * B * S * topk * moe_intermediate * H
    #   Then scaled by num_moe_layers at the model level.
    # -----------------------------------------------------------------

    def compute_flops(self, seq_len: int, batch_size: int = 1) -> float:
        B, S = batch_size, seq_len
        H, I = self.hidden_size, self.moe_intermediate_size
        tk = self.num_experts_per_tok
        flops = 0.0
        # gate_up fused projection
        flops += 2 * B * S * tk * H * (I * 2)
        # SwiGLU activation
        flops += 3 * B * S * tk * I
        # down_proj
        flops += 2 * B * S * tk * I * H
        return flops

    # -----------------------------------------------------------------
    # Activations — Excel rows 42-46 (column U), per moe-layer.
    #   U42 = B*S*topk*H           (permute output)
    #   U43 = B*S*topk*moe_intermediate*2  (gate_up output)
    #   U44 = B*S*topk*moe_intermediate     (SwiGLU)
    #   U45 = B*S*topk*moe_intermediate     (probs scaling)
    #   U46 = B*S*topk*H           (down_proj output)
    # -----------------------------------------------------------------

    def compute_activations(self, seq_len: int, batch_size: int = 1) -> float:
        B, S = batch_size, seq_len
        H, I = self.hidden_size, self.moe_intermediate_size
        tk = self.num_experts_per_tok
        elem = get_dtype_size(self.dtype)

        act = 0.0
        act += B * S * tk * H * elem           # U42
        act += B * S * tk * I * 2 * elem       # U43
        act += B * S * tk * I * elem           # U44
        act += B * S * tk * I * elem           # U45
        act += B * S * tk * H * elem           # U46
        return act