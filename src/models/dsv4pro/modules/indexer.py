"""Indexer — Sparse Attention Block Indexer.

In DSV4Pro, the Indexer is a lightweight routing module that determines
*which* KV blocks each query should attend to.  It produces a sparse index
that downstream attention modules (CSA/HCA) consume.

The indexer itself has a small two-layer MLP that maps each query token to
a block-selection logit, followed by a top-k selection.

Key parameters
--------------
hidden_size : int
num_blocks : int          – total number of KV blocks to index
indexer_inter_dim : int   – intermediate dimension of the indexer MLP
indexer_topk : int        – number of blocks selected per query
"""

from __future__ import annotations

from typing import Any

from ....core.base_module import BaseModule, ModuleParamBreakdown
from ....core.dtype import DType


class Indexer(BaseModule):
    module_type = "Indexer"

    def __init__(self, name: str, config: dict[str, Any]):
        super().__init__(name, config)
        self.hidden_size: int = config["hidden_size"]
        self.num_blocks: int = config.get("num_blocks", 0)  # 0 => derived at runtime
        self.indexer_inter_dim: int = config.get("indexer_inter_dim", 512)
        self.indexer_topk: int = config.get("indexer_topk", 16)
        self.dtype: DType = DType(config.get("dtype", "bf16"))

    # -----------------------------------------------------------------

    def _get_num_blocks(self, seq_len: int) -> int:
        if self.num_blocks > 0:
            return self.num_blocks
        # default: one block per 128 tokens
        return max(seq_len // 128, 1)

    def compute_params(self) -> list[ModuleParamBreakdown]:
        params: list[ModuleParamBreakdown] = []
        H = self.hidden_size
        I = self.indexer_inter_dim
        # For num_blocks we use a placeholder; will be set at runtime for
        # FLOPs. For params we need a fixed second-layer output dim.
        # Use a reasonable default if num_blocks is 0.
        NB = max(self.num_blocks, 256)

        # Layer 1: H -> I
        params.append(ModuleParamBreakdown(
            "indexer_fc1", (H, I), H * I, self.dtype))
        params.append(ModuleParamBreakdown(
            "indexer_fc1_bias", (I,), I, self.dtype))

        # Layer 2: I -> NB
        params.append(ModuleParamBreakdown(
            "indexer_fc2", (I, NB), I * NB, self.dtype))
        params.append(ModuleParamBreakdown(
            "indexer_fc2_bias", (NB,), NB, self.dtype))

        # RMSNorm
        params.append(ModuleParamBreakdown(
            "indexer_norm", (H,), H, self.dtype))

        return params

    # -----------------------------------------------------------------

    def compute_flops(self, seq_len: int, batch_size: int = 1) -> float:
        H = self.hidden_size
        I = self.indexer_inter_dim
        NB = self._get_num_blocks(seq_len)
        B, S = batch_size, seq_len

        flops = 0.0
        # FC1: H -> I
        flops += 2 * B * S * H * I
        # FC2: I -> NB
        flops += 2 * B * S * I * NB
        # Softmax over NB logits
        flops += 3 * B * S * NB
        # Top-k selection (approximate comparison cost)
        flops += B * S * NB  # ~N comparisons per token
        return flops

    # -----------------------------------------------------------------

    def compute_activations(self, seq_len: int, batch_size: int = 1) -> float:
        from ....core.dtype import get_dtype_size
        NB = self._get_num_blocks(seq_len)
        B, S = batch_size, seq_len
        elem = get_dtype_size(self.dtype)

        act = 0.0
        act += B * S * self.indexer_inter_dim * elem    # FC1 output
        act += B * S * NB * elem                          # FC2 output (logits)
        act += B * S * self.indexer_topk * elem           # index output
        return act