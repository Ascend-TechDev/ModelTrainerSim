"""Separate parameter, FLOPs and activation accounting for DSpark extras.

Sequence length means draft positions for this optional auxiliary estimate.
The main input projection consumes only the latest committed token, while the
Markov, confidence and shared LM heads consume all draft positions.
"""

from ....core.base_module import ModuleParamBreakdown

from .common import FlashModule, _dimensions, _positive, _term


def _settings(arch):
    hidden = _positive(arch["dim"], "dim")
    vocab = _positive(arch["vocab_size"], "vocab_size")
    rank = _positive(arch.get("dspark_markov_rank", 256), "dspark_markov_rank")
    stages = _positive(arch["n_mtp_layers"], "n_mtp_layers")
    targets = len(arch.get("dspark_target_layer_ids", []))
    if not targets:
        raise ValueError("DSpark requires dspark_target_layer_ids")
    return hidden, vocab, rank, stages, targets


def _details():
    return {
        "source": "inference/model.py:DSparkBlock, DSparkMarkovHead, DSparkConfidenceHead",
        "note": ("Optional auxiliary estimate over B*S draft positions, not the default "
                 "inference prefill schedule or a specified training objective. Main embedding "
                 "and LM-head parameters are shared and counted only once."),
    }


class DSparkInputProjection(FlashModule):
    """Project the selected main-model hidden states for one committed token."""

    module_type = "DSparkExtras"
    block_accounting = True

    def __init__(self, name, config, *, layer_id=None, dspark=False):
        super().__init__(name, config, layer_id=layer_id, dspark=dspark)
        self.hidden, self.vocab, self.rank, self.stages, self.targets = _settings(self.arch)

    def compute_params(self) -> list[ModuleParamBreakdown]:
        H = self.hidden
        return [
            self.parameter("mtp.0.main_proj.weight", (H, self.targets * H)),
            self.parameter("mtp.0.main_norm.weight", (H,)),
        ]

    def compute_flops(self, seq_len: int, batch_size: int = 1) -> float:
        H, _, B = _dimensions(self.arch, seq_len, batch_size)
        return float(2 * B * self.targets * H * H)

    def compute_activations(self, seq_len: int, batch_size: int = 1) -> float:
        return float(sum(term["bytes"] for term in self.compute_activation_terms(seq_len, batch_size)))

    def compute_activation_terms(self, seq_len: int, batch_size: int = 1) -> list[dict]:
        H, _, B = _dimensions(self.arch, seq_len, batch_size)
        element_bytes = self.activation_element_bytes
        return [
            _term("target_hidden_concat", (B, 1, self.targets * H), element_bytes),
            _term("projected_main_hidden", (B, 1, H), element_bytes),
            _term("normalized_main_hidden", (B, 1, H), element_bytes),
        ]

    def compute_details(self, seq_len: int, batch_size: int = 1) -> dict:
        return _details()


class DSparkHeads(FlashModule):
    """Final draft-stage normalization, Markov head and confidence head."""

    module_type = "DSparkExtras"
    block_accounting = True

    def __init__(self, name, config, *, layer_id=None, dspark=False):
        super().__init__(name, config, layer_id=layer_id, dspark=dspark)
        self.hidden, self.vocab, self.rank, self.stages, self.targets = _settings(self.arch)

    def compute_params(self) -> list[ModuleParamBreakdown]:
        H, V, R = self.hidden, self.vocab, self.rank
        prefix = f"mtp.{self.stages - 1}"
        return [
            self.parameter(f"{prefix}.norm.weight", (H,)),
            self.parameter(f"{prefix}.markov_head.embed.weight", (V, R)),
            self.parameter(f"{prefix}.markov_head.head.weight", (V, R)),
            self.parameter(f"{prefix}.confidence_head.proj.weight", (1, H + R)),
        ]

    def compute_flops(self, seq_len: int, batch_size: int = 1) -> float:
        H, S, B = _dimensions(self.arch, seq_len, batch_size)
        R = self.rank
        return float(2 * B * S * (R * self.vocab + H + R))

    def compute_activations(self, seq_len: int, batch_size: int = 1) -> float:
        return float(sum(term["bytes"] for term in self.compute_activation_terms(seq_len, batch_size)))

    def compute_activation_terms(self, seq_len: int, batch_size: int = 1) -> list[dict]:
        H, S, B = _dimensions(self.arch, seq_len, batch_size)
        V, R = self.vocab, self.rank
        element_bytes = self.activation_element_bytes
        return [
            _term("draft_head_norm_output", (B, S, H), element_bytes),
            _term("markov_embeddings", (B, S, R), element_bytes),
            _term("markov_logits", (B, S, V), self.logits_element_bytes),
            _term("confidence_input", (B, S, H + R), 4),
            _term("confidence", (B, S), 4),
        ]

    def compute_details(self, seq_len: int, batch_size: int = 1) -> dict:
        return _details()


class DSparkLMHead(FlashModule):
    """Execute the shared LM head without counting its parameters a second time."""

    module_type = "DSparkExtras"
    block_accounting = True

    def __init__(self, name, config, *, layer_id=None, dspark=False):
        super().__init__(name, config, layer_id=layer_id, dspark=dspark)
        self.hidden, self.vocab, self.rank, self.stages, self.targets = _settings(self.arch)

    def compute_params(self) -> list[ModuleParamBreakdown]:
        return []

    def compute_flops(self, seq_len: int, batch_size: int = 1) -> float:
        H, S, B = _dimensions(self.arch, seq_len, batch_size)
        return float(2 * B * S * H * self.vocab)

    def compute_activations(self, seq_len: int, batch_size: int = 1) -> float:
        return float(sum(term["bytes"] for term in self.compute_activation_terms(seq_len, batch_size)))

    def compute_activation_terms(self, seq_len: int, batch_size: int = 1) -> list[dict]:
        _, S, B = _dimensions(self.arch, seq_len, batch_size)
        return [_term("draft_logits", (B, S, self.vocab), self.logits_element_bytes)]

    def compute_details(self, seq_len: int, batch_size: int = 1) -> dict:
        return _details()
