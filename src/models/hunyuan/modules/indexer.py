"""Full Indexer (sparse block indexer, 21 Full layers in the 10T topology).

Workbook Hy4-10T sheet, module rows D10/D11 and compute rows D80-D91.

Projection (per Full layer):
  wq_b [rq, ih*id] — reuses the MLA q_resid
  wk [H, id]       — projected directly from the sub-layer input X
  k_norm [2*id]    — LayerNorm weight+bias on the indexer key
  weights_proj [H, ih] — per-head score weight (FP32 output)

Selection (per Full layer, no parameters):
  per-query GEMMs Q[K]^T -> scores[B,S,ih,T]; ReLU; head-weighted sum
  -> [B,S,T]; causal mask -> TopK(min(index_topk, T)) int32 indices.
  Shared layers skip all of this and reuse the previous Full layer's
  Top-K indices (workbook 逐层拓扑 / 缓存与口径 A30).
"""
from __future__ import annotations

from typing import Any

from ....core.base_module import BaseModule, ModuleParamBreakdown
from ....core.dtype import DType
from .accounting import GroupedParamBreakdown


class Indexer(BaseModule):
    module_type = "Indexer"

    def __init__(self, name: str, config: dict[str, Any]):
        super().__init__(name, config)
        self.hidden_size: int = config["hidden_size"]          # H
        self.q_lora_rank: int = config["q_lora_rank"]          # rq
        self.index_n_heads: int = config["index_n_heads"]      # ih
        self.index_head_dim: int = config["index_head_dim"]    # id
        self.index_topk: int = config["index_topk"]
        self.dtype: DType = DType(config.get("dtype", "bf16"))
        self.bytes: dict[str, float] = config["bytes"]

    def compute_params(self) -> list[ModuleParamBreakdown]:
        H = self.hidden_size
        rq, ih, id_ = self.q_lora_rank, self.index_n_heads, self.index_head_dim
        return [
            GroupedParamBreakdown("wq_b", (rq, ih * id_), rq * ih * id_, self.dtype, group="dense"),
            GroupedParamBreakdown("wk", (H, id_), H * id_, self.dtype, group="dense"),
            GroupedParamBreakdown("k_norm", (2 * id_,), 2 * id_, self.dtype, group="dense"),
            GroupedParamBreakdown("weights_proj", (H, ih), H * ih, self.dtype, group="dense"),
        ]

    def projection_flops(self, seq_len: int, batch_size: int = 1) -> float:
        B, S = batch_size, seq_len
        H = self.hidden_size
        rq, ih, id_ = self.q_lora_rank, self.index_n_heads, self.index_head_dim
        wq = 2 * B * S * (ih * id_) * rq
        wk = 2 * B * S * id_ * H
        weights = 2 * B * S * ih * H
        return wq + wk + weights

    def selection_flops(self, seq_len: int, batch_size: int = 1) -> float:
        B, S, T = batch_size, seq_len, seq_len             # T = S
        ih, id_ = self.index_n_heads, self.index_head_dim
        # Per-query GEMMs: batch = B*S; scores M=ih,N=T,Q=id and
        # head-weighted sum M=1,N=T,Q=ih.
        scores = 2 * (B * S) * ih * T * id_
        weighted = 2 * (B * S) * 1 * T * ih
        return scores + weighted

    def compute_flops(self, seq_len: int, batch_size: int = 1) -> float:
        return self.projection_flops(seq_len, batch_size) + self.selection_flops(seq_len, batch_size)

    def projection_activations(self, seq_len: int, batch_size: int = 1) -> float:
        B, S = batch_size, seq_len
        H = self.hidden_size
        ih, id_ = self.index_n_heads, self.index_head_dim
        b = self.bytes
        wq_out = B * S * (ih * id_) * b["activation"]      # [B,S,ih,id]
        wk_out = B * S * id_ * b["activation"]             # [B,S,id]
        k_norm = B * S * id_ * b["activation"]             # normalized key
        weights = B * S * ih * b["fp32"]                   # scaled weights FP32
        return wq_out + wk_out + k_norm + weights

    def selection_activations(self, seq_len: int, batch_size: int = 1) -> float:
        B, S, T = batch_size, seq_len, seq_len
        ksel = min(self.index_topk, T)
        b = self.bytes
        # Workbook counts the head-weighted summary [B,S,T] (FP32); the
        # per-head [B,S,ih,T] score workspace is listed separately and
        # not added to the dynamic inventory (Hy4-10T sheet R89).
        scores = B * S * T * b["fp32"]
        topk = B * S * ksel * b["topk_index"]              # int32 indices
        return scores + topk

    def compute_activations(self, seq_len: int, batch_size: int = 1) -> float:
        return (self.projection_activations(seq_len, batch_size)
                + self.selection_activations(seq_len, batch_size))
