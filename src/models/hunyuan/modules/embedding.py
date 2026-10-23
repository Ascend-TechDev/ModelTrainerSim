"""Word embedding and output LM head (not weight-tied in Hy4).

Workbook Hy4-10T sheet, module rows D5/D16 and compute rows D40-D42 /
D131-D133.  Embedding lookup is a gather (no GEMM); the LM head runs on
all positions when ``output_all_positions`` is set (workbook B48).
"""
from __future__ import annotations

from typing import Any

from ....core.base_module import BaseModule, ModuleParamBreakdown
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

    def compute_params(self) -> list[ModuleParamBreakdown]:
        return [GroupedParamBreakdown("weight", (self.vocab_size, self.hidden_size),
                                      self.vocab_size * self.hidden_size,
                                      self.dtype, group="dense")]

    def compute_flops(self, seq_len: int, batch_size: int = 1) -> float:
        return 0.0  # lookup, not a GEMM

    def compute_activations(self, seq_len: int, batch_size: int = 1) -> float:
        return batch_size * seq_len * self.hidden_size * self.bytes["activation"]


class OutputLayer(BaseModule):
    """LM head [H, V]; logits default to FP32 (workbook A45 / D16)."""

    module_type = "OutputLayer"

    def __init__(self, name: str, config: dict[str, Any]):
        super().__init__(name, config)
        self.vocab_size: int = config["vocab_size"]
        self.hidden_size: int = config["hidden_size"]
        self.output_all_positions: bool = config.get("output_all_positions", True)
        self.dtype: DType = DType(config.get("dtype", "bf16"))
        self.bytes: dict[str, float] = config["bytes"]

    def compute_params(self) -> list[ModuleParamBreakdown]:
        return [GroupedParamBreakdown("weight", (self.hidden_size, self.vocab_size),
                                      self.hidden_size * self.vocab_size,
                                      self.dtype, group="dense")]

    def compute_flops(self, seq_len: int, batch_size: int = 1) -> float:
        positions = seq_len if self.output_all_positions else 1
        return 2 * batch_size * positions * self.vocab_size * self.hidden_size

    def compute_activations(self, seq_len: int, batch_size: int = 1) -> float:
        positions = seq_len if self.output_all_positions else 1
        return batch_size * positions * self.vocab_size * self.bytes["logits"]
