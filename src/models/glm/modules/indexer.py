"""Indexer — DSA sparse-routing indexer (GLM-5.2 style).

GLM-5.2's DSA (Dynamic Sparse Attention) uses an Indexer module built on
top of the low-rank Q projection (``q_a_proj`` output, the ``q_lora_rank``
vector).  The Indexer produces per-token, per-head routing scores that
select the top-k KV tokens to attend to.

Per the Excel reference (sheet 'GLM-5.2-743B', rows 19-23), one Indexer
block contains:

  wq_b       : [q_lora_rank, index_n_heads*index_head_dim]
  wk         : [hidden_size, index_head_dim]
  k_norm     : [index_head_dim]  (L21 = D16*2  — two norms, q_norm + k_norm)
  weights_proj: [hidden_size, index_n_heads]

Excel L23 (params for all indexer layers) = SUM(L19:L22) * indexer_nums
where indexer_nums = D23 (21 in GLM-5.2-743B).

Key parameters (from GLM ``config.json``)
--------------------------------------------
hidden_size : int
q_lora_rank : int
index_head_dim : int            – 128
index_n_heads : int             – 32
index_topk : int                – 2048
num_indexer_layers : int        – 21 (Excel D23)
"""

from __future__ import annotations

from typing import Any

from ....core.base_module import BaseModule, ModuleParamBreakdown
from ....core.dtype import DType, get_dtype_size


class Indexer(BaseModule):
    """DSA sparse-routing Indexer block (GLM-5.2)."""

    module_type = "Indexer"

    def __init__(self, name: str, config: dict[str, Any]):
        super().__init__(name, config)
        self.hidden_size: int = config["hidden_size"]
        self.q_lora_rank: int = config.get("q_lora_rank", 0)
        self.index_head_dim: int = config.get("index_head_dim", 128)
        self.index_n_heads: int = config.get("index_n_heads", 32)
        self.index_topk: int = config.get("index_topk", 2048)
        self.dtype: DType = DType(config.get("dtype", "bf16"))

    # -----------------------------------------------------------------
    # Parameters — Excel rows 19-22
    # -----------------------------------------------------------------

    def compute_params(self) -> list[ModuleParamBreakdown]:
        params: list[ModuleParamBreakdown] = []
        H = self.hidden_size
        q_lora = self.q_lora_rank
        idx_d = self.index_head_dim
        idx_h = self.index_n_heads

        # L19: wq_b = q_lora * idx_h * idx_d
        params.append(ModuleParamBreakdown(
            "wq_b", (q_lora, idx_h * idx_d),
            q_lora * idx_h * idx_d, self.dtype))
        # L20: wk = H * idx_d
        params.append(ModuleParamBreakdown(
            "wk", (H, idx_d), H * idx_d, self.dtype))
        # L21: k_norm = idx_d * 2  (q_norm + k_norm)
        params.append(ModuleParamBreakdown(
            "k_norm", (idx_d,), idx_d * 2, self.dtype))
        # L22: weights_proj = H * idx_h
        params.append(ModuleParamBreakdown(
            "weights_proj", (H, idx_h), H * idx_h, self.dtype))

        return params

    # -----------------------------------------------------------------
    # FLOPs
    #   The Excel sheet does not provide explicit FLOPs cells for the
    #   Indexer.  We derive them from the projection shapes:
    #     wq_b       : 2 * B * S * q_lora * idx_h * idx_d
    #     wk         : 2 * B * S * H * idx_d
    #     k_norm     : 2 * B * S * idx_d * 2
    #     weights_proj: 2 * B * S * H * idx_h
    #   + top-k selection over idx_topk indices per token (int32 compare)
    # -----------------------------------------------------------------

    def compute_flops(self, seq_len: int, batch_size: int = 1) -> float:
        H = self.hidden_size
        q_lora = self.q_lora_rank
        idx_d = self.index_head_dim
        idx_h = self.index_n_heads
        B, S = batch_size, seq_len

        flops = 0.0
        flops += 2 * B * S * q_lora * idx_h * idx_d   # wq_b
        flops += 2 * B * S * H * idx_d                # wk
        flops += 2 * B * S * idx_d * 2                # k_norm (q_norm + k_norm)
        flops += 2 * B * S * H * idx_h                # weights_proj
        # top-k selection: B * S * topk comparisons (cheap)
        flops += B * S * self.index_topk
        return flops

    # -----------------------------------------------------------------
    # Activations — Excel rows 19-23 (column U)
    #   U19 = B*S*idx_h*idx_d  * 2 bytes
    #   U20 = B*S*idx_d         * 2 bytes
    #   U21 = B*S*idx_d         * 2 bytes
    #   U22 = B*S*idx_h         * 2 bytes
    #   U23 = B*S*topk           * 4 bytes   (int32 index, NOT bf16!)
    # -----------------------------------------------------------------

    def compute_activations(self, seq_len: int, batch_size: int = 1) -> float:
        B, S = batch_size, seq_len
        H = self.hidden_size
        q_lora = self.q_lora_rank
        idx_d = self.index_head_dim
        idx_h = self.index_n_heads
        elem = get_dtype_size(self.dtype)

        act = 0.0
        act += B * S * idx_h * idx_d * elem      # U19
        act += B * S * idx_d * elem               # U20
        act += B * S * idx_d * elem               # U21
        act += B * S * idx_h * elem               # U22
        # U23: int32 top-k index buffer (4 bytes per element)
        act += B * S * self.index_topk * 4
        return act