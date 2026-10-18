"""MoERouter — Top-k gating router (GLM-5.2 style).

GLM-5.2 uses a sigmoid-scoring top-k router (``noaux_tc``) to select
which routed experts each token is dispatched to.

Per the Excel reference (sheet 'GLM-5.2-743B', rows 47-48):

  L47 = D13 * D19         (= hidden_size * n_routed_experts)
  L48 = SUM(L47) * H48     (total params, all moe_layers)
  H48 = D4                 (= moe_layer_count)
  + gate bias of size n_routed_experts (small, Excel omits)

Key parameters (from GLM ``config.json``)
--------------------------------------------
hidden_size : int
n_routed_experts : int             – 256
gate_dtype : str                   – "float32"
num_moe_layers : int               – 75
"""

from __future__ import annotations

from typing import Any

from ....core.base_module import BaseModule, ModuleParamBreakdown
from ....core.dtype import DType


class MoERouter(BaseModule):
    """Top-k gating router for MoE layers (GLM-5.2)."""

    module_type = "MoERouter"

    def __init__(self, name: str, config: dict[str, Any]):
        super().__init__(name, config)
        self.hidden_size: int = config["hidden_size"]
        self.n_routed_experts: int = config.get("n_routed_experts", 1)
        self.num_experts_per_tok: int = config.get("num_experts_per_tok", 1)
        self.gate_dtype: DType = DType(config.get("gate_dtype", "fp32"))
        self.dtype: DType = DType(config.get("dtype", "bf16"))

    # -----------------------------------------------------------------
    # Parameters — Excel row 47
    # -----------------------------------------------------------------

    def compute_params(self) -> list[ModuleParamBreakdown]:
        H = self.hidden_size
        E = self.n_routed_experts
        # Excel L47 = H * E (gate_weight only; gate_bias deliberately
        # excluded to match the Excel sheet's parameter total).
        return [
            ModuleParamBreakdown("gate_weight", (H, E), H * E, self.gate_dtype),
        ]

    # -----------------------------------------------------------------
    # FLOPs — derived from shape:
    #   gate matmul: 2 * B * S * H * E
    #   sigmoid + top-k selection: 3 * B * S * E
    #   (per moe-layer; model scales by num_moe_layers)
    # -----------------------------------------------------------------

    def compute_flops(self, seq_len: int, batch_size: int = 1) -> float:
        H = self.hidden_size
        E = self.n_routed_experts
        B, S = batch_size, seq_len
        flops = 0.0
        flops += 2 * B * S * H * E       # gate matmul
        flops += 3 * B * S * E             # sigmoid + top-k
        return flops

    # -----------------------------------------------------------------
    # Activations — Excel rows 47-48 do not list a separate activation
    #   cell for the router.  We return a small bf16 gate-logits buffer.
    # -----------------------------------------------------------------

    def compute_activations(self, seq_len: int, batch_size: int = 1) -> float:
        from ....core.dtype import get_dtype_size
        B, S = batch_size, seq_len
        E = self.n_routed_experts
        elem = get_dtype_size(self.gate_dtype)
        return B * S * E * elem