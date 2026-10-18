"""DSA — Output projection (GLM-5.2 style).

The DSA output block in GLM-5.2 is the attention output projection
(``o_proj``) that maps the concatenated multi-head attention output back
to the hidden dimension.

Per the Excel reference (sheet 'GLM-5.2-743B', rows 26-27), one DSA
block contains:

  o_proj : [num_kv_heads*qk_head_dim, hidden_size]

Excel L27 (params for all layers) = SUM(L26) * num_layers
where num_layers = D2 (78 in GLM-5.2-743B).

Key parameters (from GLM ``config.json``)
--------------------------------------------
hidden_size : int
num_key_value_heads : int       – 64
qk_head_dim : int               – 256
"""

from __future__ import annotations

from typing import Any

from ....core.base_module import BaseModule, ModuleParamBreakdown
from ....core.dtype import DType, get_dtype_size


class DSA(BaseModule):
    """DSA attention output projection (GLM-5.2)."""

    module_type = "DSA"

    def __init__(self, name: str, config: dict[str, Any]):
        super().__init__(name, config)
        self.hidden_size: int = config["hidden_size"]
        self.num_kv_heads: int = config.get("num_key_value_heads",
                                            config.get("num_heads", 1))
        self.qk_head_dim: int = config.get("qk_head_dim", 256)
        self.v_head_dim: int = config.get("v_head_dim", self.qk_head_dim)
        self.dtype: DType = DType(config.get("dtype", "bf16"))

    # -----------------------------------------------------------------
    # Parameters — Excel row 26
    #   L26 = D6 * D9 * D13 = kv_heads * qk_head_dim * hidden_size
    # -----------------------------------------------------------------

    def compute_params(self) -> list[ModuleParamBreakdown]:
        H = self.hidden_size
        n_kv = self.num_kv_heads
        d_qk = self.qk_head_dim
        params: list[ModuleParamBreakdown] = []
        params.append(ModuleParamBreakdown(
            "o_proj", (n_kv * d_qk, H), n_kv * d_qk * H, self.dtype))
        return params

    # -----------------------------------------------------------------
    # FLOPs — Excel does not provide explicit cells; derive from shape.
    #   o_proj: [B,S, n_kv*d_qk] * [n_kv*d_qk, H]
    #         = 2 * B * S * n_kv * d_qk * H
    # -----------------------------------------------------------------

    def compute_flops(self, seq_len: int, batch_size: int = 1) -> float:
        H = self.hidden_size
        n_kv = self.num_kv_heads
        d_qk = self.qk_head_dim
        B, S = batch_size, seq_len
        return 2 * B * S * n_kv * d_qk * H

    # -----------------------------------------------------------------
    # Activations — Excel rows 26-27 (column U)
    #   U26 = B*S*heads_num*head_dim  * 2 bytes  (attention output)
    #   U27 = B*S*hidden_size          * 2 bytes  (after o_proj)
    # -----------------------------------------------------------------

    def compute_activations(self, seq_len: int, batch_size: int = 1) -> float:
        B, S = batch_size, seq_len
        H = self.hidden_size
        n_kv = self.num_kv_heads
        d_qk = self.qk_head_dim
        elem = get_dtype_size(self.dtype)

        act = 0.0
        act += B * S * n_kv * d_qk * elem    # U26
        act += B * S * H * elem              # U27
        return act