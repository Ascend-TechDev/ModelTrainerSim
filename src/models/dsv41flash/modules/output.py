"""V4.1-Flash final RMSNorm and full-sequence training LM head."""

from __future__ import annotations

from typing import Any

from ....core.base_module import ModuleParamBreakdown
from .common import FlashModule, _dimensions, _positive, _term


class OutputLayer(FlashModule):
    module_type = "OutputLayer"
    block_accounting = True

    # -----------------------------------------------------------------
    # Parameters
    # -----------------------------------------------------------------

    def compute_params(self) -> list[ModuleParamBreakdown]:
        H = _positive(self.arch["dim"], "dim")
        V = _positive(self.arch["vocab_size"], "vocab_size")
        return [self.parameter("norm.weight", (H,)), self.parameter("head.weight", (V, H))]

    # -----------------------------------------------------------------
    # FLOPs
    # -----------------------------------------------------------------

    def compute_flops(self, seq_len: int, batch_size: int = 1) -> float:
        H, S, B = _dimensions(self.arch, seq_len, batch_size)
        V = _positive(self.arch["vocab_size"], "vocab_size")
        return float(2 * B * S * H * V)

    # -----------------------------------------------------------------
    # Activations
    # -----------------------------------------------------------------

    def compute_activations(self, seq_len: int, batch_size: int = 1) -> float:
        return float(sum(term["bytes"] for term in self.compute_activation_terms(seq_len, batch_size)))

    def compute_activation_terms(self, seq_len: int, batch_size: int = 1) -> list[dict[str, Any]]:
        H, S, B = _dimensions(self.arch, seq_len, batch_size)
        V = _positive(self.arch["vocab_size"], "vocab_size")
        element_bytes = self.activation_element_bytes
        return [
            _term("final_norm_input", (B, S, H), element_bytes),
            _term("final_norm_output", (B, S, H), element_bytes),
            _term("final_norm_inverse_rms", (B, S, 1), 4),
            _term("full_logits", (B, S, V), self.logits_element_bytes),
        ]

    # -----------------------------------------------------------------
    # Extra info
    # -----------------------------------------------------------------

    def compute_details(self, seq_len: int, batch_size: int = 1) -> dict[str, Any]:
        return {
            "source": "inference/model.py:ParallelHead, Transformer.norm/head",
            "note": ("Full-sequence logits budget for training. Reference generation selects "
                     "only the last position; fused/chunked cross-entropy can avoid full logits. "
                     "Loss and softmax workspaces are excluded."),
        }
