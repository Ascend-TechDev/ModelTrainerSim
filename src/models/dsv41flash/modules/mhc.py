"""V4.1-Flash mHC transforms and sublayer RMSNorm accounting."""

from __future__ import annotations

from typing import Any

from ....core.base_module import ModuleParamBreakdown
from .common import FlashModule, _dimensions, _positive, _term


class mHC(FlashModule):
    module_type = "mHC_RMSNorm"
    block_accounting = True

    def __init__(self, name: str, config: dict[str, Any], *, layer_id: int, dspark: bool = False):
        super().__init__(name, config, layer_id=layer_id, dspark=dspark)
        self.hidden_size = _positive(self.arch["dim"], "dim")
        self.hc_mult = _positive(self.arch.get("hc_mult", 4), "hc_mult")
        self.mix_dim = self.hc_mult * (self.hc_mult + 2)
        is_dspark = layer_id >= self.arch["n_layers"]
        parameter_layer_id = layer_id - self.arch["n_layers"] if is_dspark else layer_id
        self.parameter_prefix = f"{'mtp' if is_dspark else 'layers'}.{parameter_layer_id}"

    # -----------------------------------------------------------------
    # Parameters
    # -----------------------------------------------------------------

    def compute_params(self) -> list[ModuleParamBreakdown]:
        H, R, mix = self.hidden_size, self.hc_mult, self.mix_dim
        prefix = self.parameter_prefix
        parameters = []
        for sublayer in ("attn", "ffn"):
            parameters.extend([
                self.parameter(f"{prefix}.hc_{sublayer}_fn", (mix, R * H)),
                self.parameter(f"{prefix}.hc_{sublayer}_base", (mix,)),
                self.parameter(f"{prefix}.hc_{sublayer}_scale", (3,)),
                self.parameter(f"{prefix}.{sublayer}_norm.weight", (H,)),
            ])
        return parameters

    # -----------------------------------------------------------------
    # FLOPs
    # -----------------------------------------------------------------

    def compute_flops(self, seq_len: int, batch_size: int = 1) -> float:
        H, S, B = _dimensions(self.arch, seq_len, batch_size)
        # Two coefficient projections, each with one multiply-add counted as 2.
        return float(4 * B * S * self.hc_mult * H * self.mix_dim)

    # -----------------------------------------------------------------
    # Activations
    # -----------------------------------------------------------------

    def compute_activations(self, seq_len: int, batch_size: int = 1) -> float:
        return float(sum(term["bytes"] for term in self.compute_activation_terms(seq_len, batch_size)))

    def compute_activation_terms(self, seq_len: int, batch_size: int = 1) -> list[dict[str, Any]]:
        H, S, B = _dimensions(self.arch, seq_len, batch_size)
        R, mix = self.hc_mult, self.mix_dim
        element_bytes = self.activation_element_bytes
        return [
            _term("two_residual_stream_inputs", (2, B, S, R, H), element_bytes),
            _term("two_collapsed_norm_inputs", (2, B, S, H), element_bytes),
            _term("two_hc_coefficient_sets", (2, B, S, mix), 4),
            _term("two_hc_inverse_rms", (2, B, S, 1), 4),
            _term("two_sublayer_norm_inverse_rms", (2, B, S, 1), 4),
        ]

    # -----------------------------------------------------------------
    # Extra info
    # -----------------------------------------------------------------

    def compute_details(self, seq_len: int, batch_size: int = 1) -> dict[str, Any]:
        return {
            "source": "inference/model.py:Block.__init__, hc_mixes, hc_pre, hc_post",
            "layer_id": self.layer_id,
            "hc_mult": self.hc_mult,
            "mix_dim": self.mix_dim,
            "note": ("Two mHC projections including base/scale and two RMSNorm weights. "
                     "Final model norm belongs to Output Layer. Sinkhorn, residual mixing and "
                     "collapse arithmetic are excluded from the GEMM-only FLOPs. Sinkhorn "
                     "iteration intermediates and temporary FP32 casts are excluded from retention."),
        }
