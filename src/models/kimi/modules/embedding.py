"""Word embedding, output LM head, and the entry/exit norms.

Workbook rows (10T-深度-v2): I2 = D19*D9 (embedding, not tied),
I35 = D9*D19 (LM head), I3 = I4 = D9 (pre/post norm, counted once),
I34 = D9 (final norm).  Activation tensors N2/N3/N4 (residuals) and
N35 = logits in FP32 (4 bytes, row M35).
"""
from __future__ import annotations

from typing import Any

from ....core.base_module import BaseModule
from ....core.dtype import DType
from .accounting import GroupedParamBreakdown


class WordEmbedding(BaseModule):
    module_type = "WordEmbedding"

    def __init__(self, name: str, config: dict[str, Any]):
        super().__init__(name, config)
        self.vocab_size: int = config["vocab_size"]
        self.hidden_size: int = config["hidden_size"]
        self.dtype: DType = DType(config.get("dtype", "bf16"))
        self.bytes: dict[str, float] = config["bytes"]

    def compute_params(self):
        return [GroupedParamBreakdown("weight", (self.vocab_size, self.hidden_size),
                                      self.vocab_size * self.hidden_size,
                                      self.dtype, group="adamw", shard="tp")]

    def compute_flops(self, seq_len: int, batch_size: int = 1) -> float:
        return 0.0  # lookup, not a GEMM

    def compute_activations(self, seq_len: int, batch_size: int = 1) -> float:
        return batch_size * seq_len * self.hidden_size * self.bytes["activation"]  # N2


class OutputLayer(BaseModule):
    """LM head [h, v]; logits FP32 (row M35)."""

    module_type = "OutputLayer"

    def __init__(self, name: str, config: dict[str, Any]):
        super().__init__(name, config)
        self.vocab_size: int = config["vocab_size"]
        self.hidden_size: int = config["hidden_size"]
        self.output_all_positions: bool = config.get("output_all_positions", True)
        self.dtype: DType = DType(config.get("dtype", "bf16"))
        self.bytes: dict[str, float] = config["bytes"]

    def compute_params(self):
        return [GroupedParamBreakdown("weight", (self.hidden_size, self.vocab_size),
                                      self.hidden_size * self.vocab_size,
                                      self.dtype, group="adamw", shard="tp")]

    def compute_flops(self, seq_len: int, batch_size: int = 1) -> float:
        positions = seq_len if self.output_all_positions else 1
        return 2 * batch_size * positions * self.vocab_size * self.hidden_size  # R35

    def compute_activations(self, seq_len: int, batch_size: int = 1) -> float:
        positions = seq_len if self.output_all_positions else 1
        return batch_size * positions * self.vocab_size * self.bytes["logits"]  # M35


class _Norm(BaseModule):
    """Shared norm implementation (I3/I4/I34: one instance each)."""

    def __init__(self, name: str, config: dict[str, Any]):
        super().__init__(name, config)
        self.hidden_size: int = config["hidden_size"]
        self.dtype: DType = DType(config.get("dtype", "bf16"))
        self.bytes: dict[str, float] = config["bytes"]

    def compute_params(self):
        return [GroupedParamBreakdown("weight", (self.hidden_size,), self.hidden_size,
                                      self.dtype, group="muon", shard="tp")]

    def compute_flops(self, seq_len: int, batch_size: int = 1) -> float:
        return 0.0

    def compute_activations(self, seq_len: int, batch_size: int = 1) -> float:
        return batch_size * seq_len * self.hidden_size * self.bytes["activation"]  # N3/N4


class PreNorm(_Norm):
    module_type = "PreNorm"


class PostNorm(_Norm):
    module_type = "PostNorm"


class FinalNorm(_Norm):
    module_type = "FinalNorm"
