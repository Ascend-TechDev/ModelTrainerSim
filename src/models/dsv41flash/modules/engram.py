"""V4.1-Flash Engram lookup, WKV projection and context-aware gates."""

from __future__ import annotations

from typing import Any

from ....core.base_module import ModuleParamBreakdown
from .common import FlashModule, _dimensions, _positive, _term


class Engram(FlashModule):
    module_type = "Engram"
    block_accounting = True

    def __init__(self, name: str, config: dict[str, Any], *, layer_id: int, dspark: bool = False):
        super().__init__(name, config, layer_id=layer_id, dspark=dspark)
        arch = self.arch
        self.hidden_size = _positive(arch["dim"], "dim")
        self.hc_mult = _positive(arch.get("hc_mult", 4), "hc_mult")
        layer_ids = list(arch.get("engram_layer_ids", []))
        if layer_id not in layer_ids:
            raise ValueError(f"Layer {layer_id} has no Engram module")
        self.table_rows = _positive(arch["engram_num_embeddings"][layer_ids.index(layer_id)], "engram table rows")
        self.head_dim = _positive(arch["engram_head_dim"], "engram_head_dim")
        gram = _positive(arch["engram_max_ngram_size"], "engram_max_ngram_size")
        if gram < 2:
            raise ValueError("engram_max_ngram_size must be at least 2")
        self.hashes_per_token = (gram - 1) * _positive(arch["engram_n_heads"], "engram_n_heads")
        self.lookup_dim = self.hashes_per_token * self.head_dim
        self.output_dim = (self.hc_mult + 1) * self.hidden_size
        self.parameter_prefix = f"layers.{layer_id}.engram"

    # -----------------------------------------------------------------
    # Parameters
    # -----------------------------------------------------------------

    def compute_params(self) -> list[ModuleParamBreakdown]:
        H, R, D = self.hidden_size, self.hc_mult, self.head_dim
        prefix = self.parameter_prefix
        return [
            self.parameter(f"{prefix}.embed.weight", (self.table_rows, D), "engram"),
            self.parameter(f"{prefix}.wkv.weight", (self.output_dim, self.lookup_dim)),
            self.parameter(f"{prefix}.q_weight", (R, H)),
            self.parameter(f"{prefix}.k_weight", (R, H)),
        ]

    # -----------------------------------------------------------------
    # FLOPs
    # -----------------------------------------------------------------

    def compute_flops(self, seq_len: int, batch_size: int = 1) -> float:
        _, S, B = _dimensions(self.arch, seq_len, batch_size)
        # Hash lookup and gate scalar arithmetic are outside the GEMM scope.
        return float(2 * B * S * self.lookup_dim * self.output_dim)

    # -----------------------------------------------------------------
    # Activations
    # -----------------------------------------------------------------

    def compute_activations(self, seq_len: int, batch_size: int = 1) -> float:
        return float(sum(term["bytes"] for term in self.compute_activation_terms(seq_len, batch_size)))

    def compute_activation_terms(self, seq_len: int, batch_size: int = 1) -> list[dict[str, Any]]:
        H, S, B = _dimensions(self.arch, seq_len, batch_size)
        R, D = self.hc_mult, self.head_dim
        hashes, output_dim = self.hashes_per_token, self.output_dim
        element_bytes = self.activation_element_bytes
        return [
            _term("hash_row_ids", (B, S, hashes), 8),
            _term("lookup_vectors", (B, S, hashes, D), element_bytes),
            _term("projected_key_value", (B, S, output_dim), element_bytes),
            _term("key_fp32", (B, S, R, H), 4),
            _term("residual_for_gate_fp32", (B, S, R, H), 4),
            _term("gate_inverse_rms", (B, S, R), 4),
            _term("gate_scores", (B, S, R), 4),
            _term("gates", (B, S, R), 4),
        ]

    # -----------------------------------------------------------------
    # Extra info
    # -----------------------------------------------------------------

    def compute_details(self, seq_len: int, batch_size: int = 1) -> dict[str, Any]:
        H, R, D = self.hidden_size, self.hc_mult, self.head_dim
        hashes = self.hashes_per_token
        return {
            "source": "inference/model.py:ParallelEngramEmbedding, Engram; inference/engram.py:EngramLayout",
            "layer_id": self.layer_id,
            "table_rows": self.table_rows,
            "table_head_dim": D,
            "hashes_per_token": hashes,
            "lookup_dim": self.lookup_dim,
            "projection_output_dim": self.output_dim,
            "active_parameter_elements": hashes * D + self.output_dim * self.lookup_dim + 2 * R * H,
            "note": ("Only embed.weight belongs to the engram placement/sharding group. Projection "
                     "and gate weights remain dense. Hash buffers and checkpoint quantization scale "
                     "metadata are not trainable parameters. Gate arithmetic is excluded from FLOPs; "
                     "CPU table placement does not remove lookup/projected activations from accelerator memory. "
                     "Distinct rows touched across a batch cannot be inferred from sequence length alone."),
        }
