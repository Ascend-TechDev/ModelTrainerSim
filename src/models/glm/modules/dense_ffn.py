"""DenseFFN — Dense Feed-Forward Network (GLM-5.2 style).

GLM-5.2 uses a standard SwiGLU dense FFN in the first
``first_k_dense_replace`` layers (3 layers in GLM-5.2-743B).

Per the Excel reference (sheet 'GLM-5.2-743B', rows 30-33), one Dense
block contains:

  gate_proj : [hidden_size, intermediate_size]
  up_proj   : [hidden_size, intermediate_size]
  down_proj : [intermediate_size, hidden_size]

Excel L33 (params for all dense layers) = SUM(L30:L32) * dense_layer_count
where dense_layer_count = D3 (3 in GLM-5.2-743B).

Key parameters (from GLM ``config.json``)
--------------------------------------------
hidden_size : int
intermediate_size : int       – 12288
"""

from __future__ import annotations

from typing import Any

from ....core.base_module import BaseModule, ModuleParamBreakdown
from ....core.dtype import DType, get_dtype_size


class DenseFFN(BaseModule):
    """Standard SwiGLU dense FFN (GLM-5.2)."""

    module_type = "DenseFFN"

    def __init__(self, name: str, config: dict[str, Any]):
        super().__init__(name, config)
        self.hidden_size: int = config["hidden_size"]
        self.intermediate_size: int = config["intermediate_size"]
        self.dtype: DType = DType(config.get("dtype", "bf16"))

    # -----------------------------------------------------------------
    # Parameters — Excel rows 30-32
    # -----------------------------------------------------------------

    def compute_params(self) -> list[ModuleParamBreakdown]:
        H = self.hidden_size
        I = self.intermediate_size
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
    #   + SwiGLU activation (~3 * B * S * I)
    # -----------------------------------------------------------------

    def compute_flops(self, seq_len: int, batch_size: int = 1) -> float:
        B, S = batch_size, seq_len
        H, I = self.hidden_size, self.intermediate_size
        flops = 0.0
        flops += 2 * B * S * H * I     # gate_proj
        flops += 2 * B * S * H * I     # up_proj
        flops += 2 * B * S * I * H     # down_proj
        flops += 3 * B * S * I         # SwiGLU
        return flops

    # -----------------------------------------------------------------
    # Activations — Excel rows 30-32 (column U)
    #   U30 = B*S*intermediate_size * 2 bytes  (gate_proj output)
    #   U31 = B*S*intermediate_size * 2 bytes  (up_proj output)
    #   U32 = B*S*hidden_size      * 2 bytes  (down_proj output)
    # -----------------------------------------------------------------

    def compute_activations(self, seq_len: int, batch_size: int = 1) -> float:
        B, S = batch_size, seq_len
        H, I = self.hidden_size, self.intermediate_size
        elem = get_dtype_size(self.dtype)

        act = 0.0
        act += B * S * I * elem     # U30
        act += B * S * I * elem     # U31
        act += B * S * H * elem     # U32
        return act