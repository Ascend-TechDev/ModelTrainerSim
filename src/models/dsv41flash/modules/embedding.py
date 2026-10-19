"""V4.1-Flash main embedding, shared by reference with DSpark."""

from __future__ import annotations

from typing import Any

from ....core.base_module import ModuleParamBreakdown
from .common import FlashModule, _dimensions, _positive, _term


class Embedding(FlashModule):
    module_type = "WordEmbedding"
    block_accounting = True

    # -----------------------------------------------------------------
    # Parameters
    # -----------------------------------------------------------------

    def compute_params(self) -> list[ModuleParamBreakdown]:
        H = _positive(self.arch["dim"], "dim")
        V = _positive(self.arch["vocab_size"], "vocab_size")
        return [self.parameter("embed.weight", (V, H))]

    # -----------------------------------------------------------------
    # FLOPs
    # -----------------------------------------------------------------

    def compute_flops(self, seq_len: int, batch_size: int = 1) -> float:
        _dimensions(self.arch, seq_len, batch_size)
        # Embedding is a table lookup; no GEMM FLOPs are charged.
        return 0.0

    # -----------------------------------------------------------------
    # Activations
    # -----------------------------------------------------------------

    def compute_activations(self, seq_len: int, batch_size: int = 1) -> float:
        return float(sum(term["bytes"] for term in self.compute_activation_terms(seq_len, batch_size)))

    def compute_activation_terms(self, seq_len: int, batch_size: int = 1) -> list[dict[str, Any]]:
        H, S, B = _dimensions(self.arch, seq_len, batch_size)
        return [_term("embedding_output", (B, S, H), self.activation_element_bytes)]

    # -----------------------------------------------------------------
    # Extra info
    # -----------------------------------------------------------------

    def compute_details(self, seq_len: int, batch_size: int = 1) -> dict[str, Any]:
        return {
            "source": "inference/model.py:ParallelEmbedding, Transformer.__init__",
            "tied_output": False,
            "note": "Main LM head is independent; expansion into mHC streams is budgeted in mHC.",
        }
