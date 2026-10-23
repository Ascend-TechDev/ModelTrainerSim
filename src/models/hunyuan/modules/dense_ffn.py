"""Dense FFN (layer 0 only; replaces MoE on that layer).

Workbook Hy4-10T sheet, module row D12 and compute rows D94-D98:
  gate_proj + up_proj [H, 2*D] (two Linears), SwiGLU, down_proj [D, H].
"""
from __future__ import annotations

from typing import Any

from ....core.base_module import BaseModule, ModuleParamBreakdown
from ....core.dtype import DType
from .accounting import GroupedParamBreakdown


class DenseFFN(BaseModule):
    module_type = "DenseFFN"

    def __init__(self, name: str, config: dict[str, Any]):
        super().__init__(name, config)
        self.hidden_size: int = config["hidden_size"]
        self.intermediate_size: int = config["intermediate_size"]
        self.dtype: DType = DType(config.get("dtype", "bf16"))
        self.bytes: dict[str, float] = config["bytes"]

    def compute_params(self) -> list[ModuleParamBreakdown]:
        H, D = self.hidden_size, self.intermediate_size
        return [
            GroupedParamBreakdown("gate_up_proj", (H, 2 * D), H * 2 * D, self.dtype, group="dense"),
            GroupedParamBreakdown("down_proj", (D, H), D * H, self.dtype, group="dense"),
        ]

    def compute_flops(self, seq_len: int, batch_size: int = 1) -> float:
        H, D = self.hidden_size, self.intermediate_size
        return 3 * 2 * batch_size * seq_len * D * H   # gate + up + down

    def compute_activations(self, seq_len: int, batch_size: int = 1) -> float:
        B, S = batch_size, seq_len
        D = self.intermediate_size
        a = self.bytes["activation"]
        gate_up = B * S * (2 * D) * a                  # [B,S,2D]
        swiglu = B * S * D * a                         # SiLU(gate) x up
        down = B * S * self.hidden_size * a            # [B,S,H]
        return gate_up + swiglu + down
