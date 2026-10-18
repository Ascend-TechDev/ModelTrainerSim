"""SharedExperts — Always-on shared experts FFN (GLM-5.2 style).

GLM-5.2 MoE layers (75 layers) include ``n_shared_experts`` always-on
shared experts.  Each shared expert has the same SwiGLU structure as a
routed expert:

  gate_proj : [hidden_size, moe_intermediate_size]
  up_proj   : [hidden_size, moe_intermediate_size]
  down_proj : [moe_intermediate_size, hidden_size]

Per the Excel reference (sheet 'GLM-5.2-743B', rows 36-39):

  L36 = D13 * D15         (= H * moe_intermediate_size)
  L37 = D13 * D15         (= H * moe_intermediate_size)
  L38 = D15 * D13          (= moe_intermediate_size * H)
  L39 = SUM(L36:L38) * H37
  H37 = D4 * D20           (= moe_layer_count * n_shared_experts)

So the params are counted for ALL moe_layers × n_shared_experts instances.

Key parameters (from GLM ``config.json``)
--------------------------------------------
hidden_size : int
moe_intermediate_size : int       – 2048
num_shared_experts : int          – 1
num_moe_layers : int              – 75
"""

from __future__ import annotations

from typing import Any

from ....core.base_module import BaseModule, ModuleParamBreakdown
from ....core.dtype import DType, get_dtype_size


class SharedExperts(BaseModule):
    """Always-on shared experts FFN (GLM-5.2)."""

    module_type = "SharedExperts"

    def __init__(self, name: str, config: dict[str, Any]):
        super().__init__(name, config)
        self.hidden_size: int = config["hidden_size"]
        self.moe_intermediate_size: int = config["moe_intermediate_size"]
        self.num_shared_experts: int = config.get("num_shared_experts", 1)
        self.dtype: DType = DType(config.get("dtype", "bf16"))

    # -----------------------------------------------------------------
    # Parameters — Excel rows 36-38
    #   Returns params for ONE shared expert instance.  The model
    #   top-level multiplies by num_shared_experts * num_moe_layers
    #   (Excel H37 = D4 * D20).
    # -----------------------------------------------------------------

    def compute_params(self) -> list[ModuleParamBreakdown]:
        H = self.hidden_size
        I = self.moe_intermediate_size
        return [
            ModuleParamBreakdown("gate_proj", (H, I), H * I, self.dtype),
            ModuleParamBreakdown("up_proj", (H, I), H * I, self.dtype),
            ModuleParamBreakdown("down_proj", (I, H), I * H, self.dtype),
        ]

    # -----------------------------------------------------------------
    # FLOPs — derived from projection shapes (Excel does not provide cells)
    #   gate_proj: 2 * B * S * H * I
    #   up_proj  : 2 * B * S * H * I
    #   down_proj: 2 * B * S * I * H
    #   + SwiGLU (~3 * B * S * I)
    #   Scaled by num_shared_experts (always-on).
    # -----------------------------------------------------------------

    def compute_flops(self, seq_len: int, batch_size: int = 1) -> float:
        B, S = batch_size, seq_len
        H, I = self.hidden_size, self.moe_intermediate_size
        per_expert = (2 * B * S * H * I
                       + 2 * B * S * H * I
                       + 2 * B * S * I * H
                       + 3 * B * S * I)
        return per_expert * self.num_shared_experts

    # -----------------------------------------------------------------
    # Activations — Excel rows 36-37 (column U)
    #   U36 = B*S*moe_intermediate_size*2  * 2 bytes  (gate+up fused)
    #   U37 = B*S*hidden_size              * 2 bytes  (after down_proj)
    #   38 has no activation cell (down_proj weight only).
    #   Per-layer, then scaled by num_moe_layers (T col = D4).
    # -----------------------------------------------------------------

    def compute_activations(self, seq_len: int, batch_size: int = 1) -> float:
        B, S = batch_size, seq_len
        H, I = self.hidden_size, self.moe_intermediate_size
        elem = get_dtype_size(self.dtype)

        act = 0.0
        # S36 = B*S*moe_intermediate_size*2 (gate_up fused output)
        act += B * S * I * 2 * elem
        # S37 = B*S*hidden_size (down_proj output)
        act += B * S * H * elem
        return act