"""Configuration parser.

Reads a ``config.json`` file and validates its structure, then returns
a normalized config dict that the model builder can consume.

Supported config schema (top-level):

.. code-block:: json

    {
      "model": "DSV4Pro",
      "version": "v1",
      "hidden_size": 7168,
      "num_layers": 61,
      "vocab_size": 129280,
      "max_seq_len": 131072,
      "dtype": "bf16",
      "tie_word_embeddings": false,
      "layers": {
        "first_k_dense_replace": 3,
        "csa_layers": 3,
        "hca_layers": 16,
        "mtp_layers": 1,
        "attention": { ... },
        "moe": { ... },
        "dense_ffn": { ... },
        "indexer": { ... }
      }
    }
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


class ConfigError(Exception):
    """Raised when the configuration is invalid."""


class ConfigParser:
    """Load and validate model configuration files."""

    # Required top-level keys
    REQUIRED_KEYS = {"model", "hidden_size", "num_layers", "vocab_size"}

    # Required keys per layer type
    LAYER_REQUIRED = {
        "attention": {"kv_lora_rank", "num_heads"},
        "moe": {"num_experts", "num_activated_experts"},
        "dense_ffn": {"intermediate_size"},
    }

    def __init__(self, config_path: str | Path):
        self.config_path = Path(config_path)
        self._raw: dict[str, Any] | None = None

    # -----------------------------------------------------------------

    def load(self) -> dict[str, Any]:
        """Load and validate the JSON config file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, "r", encoding="utf-8") as f:
            self._raw = json.load(f)

        self._validate(self._raw)
        return copy.deepcopy(self._raw)

    # -----------------------------------------------------------------

    def _validate(self, cfg: dict[str, Any]) -> None:
        if not isinstance(cfg, dict):
            raise ConfigError("Top-level config must be a JSON object")

        missing = self.REQUIRED_KEYS - cfg.keys()
        if missing:
            raise ConfigError(f"Missing required keys: {missing}")

        # Type checks
        for key in ("hidden_size", "num_layers", "vocab_size"):
            if not isinstance(cfg[key], int) or cfg[key] <= 0:
                raise ConfigError(f"'{key}' must be a positive integer, got: {cfg[key]}")

        model = cfg["model"]
        if not isinstance(model, str) or not model:
            raise ConfigError(f"'model' must be a non-empty string, got: {model!r}")

        # Optional defaults
        cfg.setdefault("max_seq_len", 4096)
        cfg.setdefault("dtype", "bf16")
        cfg.setdefault("tie_word_embeddings", False)
        cfg.setdefault("layers", {})

        layers = cfg["layers"]
        if not isinstance(layers, dict):
            raise ConfigError("'layers' must be an object")

        layers.setdefault("first_k_dense_replace", 0)
        layers.setdefault("csa_layers", layers["first_k_dense_replace"])
        layers.setdefault("hca_layers", 0)
        layers.setdefault("mtp_layers", 0)
        layers.setdefault("attention", {})
        layers.setdefault("moe", {})
        layers.setdefault("dense_ffn", {})
        layers.setdefault("indexer", {})

        # Validate attention config if CSA/HCA/mHC layers exist
        if (layers["csa_layers"] + layers["hca_layers"]) > 0 or True:
            attn = layers["attention"]
            for k in ("kv_lora_rank", "num_heads"):
                if k in attn and (not isinstance(attn[k], int) or attn[k] <= 0):
                    raise ConfigError(f"attention.{k} must be a positive integer")

        # Validate MoE config
        moe = layers["moe"]
        for k in ("num_experts", "num_activated_experts"):
            if k in moe and (not isinstance(moe[k], int) or moe[k] <= 0):
                raise ConfigError(f"moe.{k} must be a positive integer")

    # -----------------------------------------------------------------

    @staticmethod
    def load_from_dict(cfg: dict[str, Any]) -> dict[str, Any]:
        """Validate a config dict without loading from disk."""
        parser = ConfigParser.__new__(ConfigParser)
        parser.config_path = Path("<dict>")
        parser._validate(cfg)
        return copy.deepcopy(cfg)