"""iHC (identity Hyper-Connections) sub-layer and final head modules.

Workbook Hy4-10T sheet, module rows D7/D8/D14/D15 and compute rows
D51-D60 / D119-D128:

  * ``iHCSublayer`` — one per sub-layer (Attention and FFN side each, 156
    total): hc_fn [2R, R*H] + base/scale [2R+2], with the weighted
    pre-collapse and the post residual keeping each of the R streams an
    identity residual.
  * ``SublayerNorm`` — RMSNorm before every sub-layer input (156 total).
  * ``IHCHead`` — the final learned merge of the R streams after the last
    backbone layer: hc_fn [R, R*H] + base/scale [R+1].
  * ``FinalNorm`` — RMSNorm on the backbone output.
"""
from __future__ import annotations

from typing import Any

from ....core.base_module import BaseModule, ModuleParamBreakdown
from .accounting import GroupedParamBreakdown


class iHCSublayer(BaseModule):
    """iHC fn + gate coefficients for one sub-layer (fn [2R, R*H] + 10)."""

    module_type = "iHC"

    def __init__(self, name: str, config: dict[str, Any]):
        super().__init__(name, config)
        self.hidden_size: int = config["hidden_size"]
        self.hc_mult: int = config["hc_mult"]
        self.bytes: dict[str, float] = config["bytes"]

    def compute_params(self) -> list[ModuleParamBreakdown]:
        R, H = self.hc_mult, self.hidden_size
        return [
            GroupedParamBreakdown("hc_fn", (2 * R, R * H), 2 * R * R * H, group="dense"),
            GroupedParamBreakdown("hc_base_scale", (2 * R + 2,), 2 * R + 2, group="dense"),
        ]

    def compute_flops(self, seq_len: int, batch_size: int = 1) -> float:
        # [B,S,R*H] x fn^T[R*H, 2R] -> [B,S,2R]; M=B*S, N=2R, Q=R*H.
        R, H = self.hc_mult, self.hidden_size
        return 2 * batch_size * seq_len * (2 * R) * (R * H)

    def compute_activations(self, seq_len: int, batch_size: int = 1) -> float:
        B, S = batch_size, seq_len
        R, H = self.hc_mult, self.hidden_size
        b = self.bytes
        fn_out = B * S * (2 * R) * b["fp32"]            # [B,S,2R] checkpoint FP32
        gate_out = B * S * (2 * R) * b["fp32"]          # pre/post gates [B,S,2R]
        collapse = B * S * H * b["activation"]          # pre-weighted sum [B,S,H]
        residual = B * S * R * H * b["activation"]      # post residual [B,S,R,H]
        return fn_out + gate_out + collapse + residual


class SublayerNorm(BaseModule):
    """Per-sub-layer input RMSNorm (weight [H])."""

    module_type = "SublayerNorm"

    def __init__(self, name: str, config: dict[str, Any]):
        super().__init__(name, config)
        self.hidden_size: int = config["hidden_size"]
        self.bytes: dict[str, float] = config["bytes"]

    def compute_params(self) -> list[ModuleParamBreakdown]:
        return [GroupedParamBreakdown("weight", (self.hidden_size,),
                                      self.hidden_size, group="dense")]

    def compute_flops(self, seq_len: int, batch_size: int = 1) -> float:
        return 0.0  # reduction over H, no GEMM

    def compute_activations(self, seq_len: int, batch_size: int = 1) -> float:
        return batch_size * seq_len * self.hidden_size * self.bytes["activation"]


class IHCHead(BaseModule):
    """Final learned 4-way merge after the last backbone layer (fn [R, R*H] + 5)."""

    module_type = "IHCHead"

    def __init__(self, name: str, config: dict[str, Any]):
        super().__init__(name, config)
        self.hidden_size: int = config["hidden_size"]
        self.hc_mult: int = config["hc_mult"]
        self.bytes: dict[str, float] = config["bytes"]

    def compute_params(self) -> list[ModuleParamBreakdown]:
        R, H = self.hc_mult, self.hidden_size
        return [
            GroupedParamBreakdown("hc_fn", (R, R * H), R * R * H, group="dense"),
            GroupedParamBreakdown("hc_base_scale", (R + 1,), R + 1, group="dense"),
        ]

    def compute_flops(self, seq_len: int, batch_size: int = 1) -> float:
        R, H = self.hc_mult, self.hidden_size
        return 2 * batch_size * seq_len * R * (R * H)

    def compute_activations(self, seq_len: int, batch_size: int = 1) -> float:
        B, S = batch_size, seq_len
        R, H = self.hc_mult, self.hidden_size
        b = self.bytes
        mix = B * S * R * b["fp32"]                      # [B,S,R] mix logits
        pre = B * S * R * b["fp32"]                      # sigmoid gates [B,S,R]
        collapse = B * S * H * b["activation"]           # merged [B,S,H]
        return mix + pre + collapse


class FinalNorm(BaseModule):
    """Final RMSNorm on the backbone output (weight [H])."""

    module_type = "FinalNorm"

    def __init__(self, name: str, config: dict[str, Any]):
        super().__init__(name, config)
        self.hidden_size: int = config["hidden_size"]
        self.bytes: dict[str, float] = config["bytes"]

    def compute_params(self) -> list[ModuleParamBreakdown]:
        return [GroupedParamBreakdown("weight", (self.hidden_size,),
                                      self.hidden_size, group="dense")]

    def compute_flops(self, seq_len: int, batch_size: int = 1) -> float:
        return 0.0

    def compute_activations(self, seq_len: int, batch_size: int = 1) -> float:
        return batch_size * seq_len * self.hidden_size * self.bytes["activation"]
