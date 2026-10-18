"""Embedding & LM Head modules (GLM-5.2 style).

Per the Excel reference (sheet 'GLM-5.2-743B'):

  Word Embedding  (rows 2-3):   [vocab_size, hidden_size]
  output_layernorm (rows 51-52): RMSNorm over [hidden_size]
  LM Head         (rows 55-56):  [hidden_size, vocab_size]

GLM-5.2 uses ``tie_word_embeddings=false``, so the LM head is a
separate parameter matrix.
"""

from __future__ import annotations

from typing import Any

from ....core.base_module import BaseModule, ModuleParamBreakdown
from ....core.dtype import DType, get_dtype_size


class Embedding(BaseModule):
    module_type = "Embedding"

    def __init__(self, name: str, config: dict[str, Any]):
        super().__init__(name, config)
        self.vocab_size: int = config["vocab_size"]
        self.hidden_size: int = config["hidden_size"]
        self.dtype: DType = DType(config.get("dtype", "bf16"))
        self.tie_word_embeddings: bool = config.get("tie_word_embeddings", False)

    def compute_params(self) -> list[ModuleParamBreakdown]:
        V, H = self.vocab_size, self.hidden_size
        return [ModuleParamBreakdown(
            "token_embedding", (V, H), V * H, self.dtype)]

    def compute_flops(self, seq_len: int, batch_size: int = 1) -> float:
        return 0.0   # lookup

    def compute_activations(self, seq_len: int, batch_size: int = 1) -> float:
        B, S = batch_size, seq_len
        elem = get_dtype_size(self.dtype)
        return B * S * self.hidden_size * elem


class InputLayerNorm(BaseModule):
    """RMSNorm applied to the embedding output (before first layer).

    Excel rows 6-7:   M6 = 1.14e-5 GB  (negligible);  L6 = D13 = 6144.
    Activations column U6 = B*S*H * num_layers.
    """

    module_type = "InputLayerNorm"

    def __init__(self, name: str, config: dict[str, Any]):
        super().__init__(name, config)
        self.hidden_size: int = config["hidden_size"]
        self.dtype: DType = DType(config.get("dtype", "bf16"))

    def compute_params(self) -> list[ModuleParamBreakdown]:
        H = self.hidden_size
        return [ModuleParamBreakdown("input_layernorm", (H,), H, self.dtype)]

    def compute_flops(self, seq_len: int, batch_size: int = 1) -> float:
        return 2 * batch_size * seq_len * self.hidden_size

    def compute_activations(self, seq_len: int, batch_size: int = 1) -> float:
        B, S = batch_size, seq_len
        return B * S * self.hidden_size * get_dtype_size(self.dtype)


class OutputLayerNorm(BaseModule):
    """Final RMSNorm before LM Head.

    Excel rows 51-52:   M51 = 1.14e-5 GB;  L51 = D13 = 6144.
    """

    module_type = "OutputLayerNorm"

    def __init__(self, name: str, config: dict[str, Any]):
        super().__init__(name, config)
        self.hidden_size: int = config["hidden_size"]
        self.dtype: DType = DType(config.get("dtype", "bf16"))

    def compute_params(self) -> list[ModuleParamBreakdown]:
        H = self.hidden_size
        return [ModuleParamBreakdown("output_layernorm", (H,), H, self.dtype)]

    def compute_flops(self, seq_len: int, batch_size: int = 1) -> float:
        return 2 * batch_size * seq_len * self.hidden_size

    def compute_activations(self, seq_len: int, batch_size: int = 1) -> float:
        B, S = batch_size, seq_len
        return B * S * self.hidden_size * get_dtype_size(self.dtype)


class LMHead(BaseModule):
    """Output projection to vocabulary (GLM-5.2).

    Excel rows 55-56:   L55 = D13 * D22 = hidden_size * vocab_size.
    Activations U55 = B*S*vocab_size * 2 bytes (only ONE instance).
    """

    module_type = "LMHead"

    def __init__(self, name: str, config: dict[str, Any]):
        super().__init__(name, config)
        self.hidden_size: int = config["hidden_size"]
        self.vocab_size: int = config["vocab_size"]
        self.dtype: DType = DType(config.get("dtype", "bf16"))

    def compute_params(self) -> list[ModuleParamBreakdown]:
        H, V = self.hidden_size, self.vocab_size
        return [ModuleParamBreakdown("lm_head", (H, V), H * V, self.dtype)]

    def compute_flops(self, seq_len: int, batch_size: int = 1) -> float:
        B, S = batch_size, seq_len
        return 2 * B * S * self.hidden_size * self.vocab_size

    def compute_activations(self, seq_len: int, batch_size: int = 1) -> float:
        B, S = batch_size, seq_len
        return B * S * self.vocab_size * get_dtype_size(self.dtype)