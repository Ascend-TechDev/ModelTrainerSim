"""Dense FFN (the single Dense layer, no routing).

Workbook rows I15-I18: gate [h, D] + up [h, D] + down [D, h] with
D = intermediate_size = 33792; per-layer params = 726,663,168.
Activation M33 = SUM(M15:M18) (gate out, up out, residual M17, down out).
"""
from __future__ import annotations

from typing import Any

from ....core.base_module import BaseModule
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

    def compute_params(self):
        H, D = self.hidden_size, self.intermediate_size
        bf = self.dtype
        return [
            GroupedParamBreakdown("gate_proj", (H, D), H * D, bf, group="muon", shard="tp"),  # I15
            GroupedParamBreakdown("up_proj", (H, D), H * D, bf, group="muon", shard="tp"),    # I16
            GroupedParamBreakdown("down_proj", (D, H), D * H, bf, group="muon", shard="tp"),  # I17
        ]

    def compute_flops(self, seq_len: int, batch_size: int = 1) -> float:
        return 3 * 2 * batch_size * seq_len * self.hidden_size * self.intermediate_size

    def compute_activations(self, seq_len: int, batch_size: int = 1) -> float:
        B, S = batch_size, seq_len
        H, D = self.hidden_size, self.intermediate_size
        a = self.bytes["activation"]
        gate_out = B * S * D * a                                  # N15
        up_out = B * S * D * a                                    # N16
        residual = B * S * H * a                                  # N17 (M17)
        down_out = B * S * D * a                                  # N18
        return gate_out + up_out + residual + down_out
