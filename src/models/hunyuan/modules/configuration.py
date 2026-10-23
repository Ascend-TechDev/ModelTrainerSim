"""Load the pinned official Hy4 structure plus a variant analysis scenario.

Follows dsv41flash's ``normalize_config`` pattern: the pinned file
(``config.json`` next to this package) holds the official Hy4-preview
HF config — the structure shared by all size variants (770B / 5T / 10T /
20T).  A scenario file (``configs/hy4_10t_config.json``) supplies an
``architecture`` block of variant overrides (e.g. E=2048, Top-64,
I=3456 for 10T) plus analysis settings (seq len, dtype, training
groups).  Official field names are aliased and the layer topology is
derived from the official ``mlp_layer_types`` / ``indexer_types`` lists.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from .accounting import OPTIMIZER_STATES

_TOP_LEVEL = (("hidden_size", "hidden_size"), ("num_layers", "num_hidden_layers"),
              ("vocab_size", "vocab_size"))

# Official HF config field name -> internal architecture key.
_ALIASES = {"num_heads": "num_attention_heads"}


def _positive(value, name, allow_zero=False):
    if isinstance(value, bool) or not isinstance(value, int) or value < (0 if allow_zero else 1):
        raise ValueError(f"{name} must be a {'nonnegative' if allow_zero else 'positive'} integer")


def _load_architecture_file(cfg: dict, base_dir: Path | None) -> tuple[dict, Path, str]:
    """Load the pinned base architecture (official HF config, verbatim)."""
    source = cfg.get("architecture_file")
    path = (base_dir or Path.cwd()) / source if source else Path(__file__).resolve().parent.parent / "config.json"
    data = path.read_bytes()
    loaded = json.loads(data)
    arch = loaded.get("architecture", loaded)   # wrapped form tolerated for back-compat
    return arch, path, hashlib.sha256(data).hexdigest()


def normalize_config(config: dict, base_dir: Path | None = None) -> dict:
    cfg = copy.deepcopy(config)
    if "architecture_file" in cfg or "architecture" not in cfg:
        # Base = official pinned structure; scenario "architecture" = overrides.
        arch, path, digest = _load_architecture_file(cfg, base_dir)
        overrides = cfg.pop("architecture", {})
        if not isinstance(overrides, dict):
            raise ValueError("architecture overrides must be an object")
        arch = {**arch, **overrides}
        # Store the resolved absolute path so a second normalize call
        # (e.g. inside Hy4Model.__init__) re-resolves identically.
        cfg["architecture_file"] = str(path.resolve())
        cfg["source"] = {**cfg.get("source", {}),
                         "architecture_file": str(path.resolve()),
                         "sha256": digest}
    else:
        arch = cfg["architecture"]
    for target, source in _ALIASES.items():
        if target not in arch and source in arch:
            arch[target] = arch[source]
    # Topology derived from the official per-layer type lists (逐层拓扑).
    if "dense_layers" not in arch and isinstance(arch.get("mlp_layer_types"), list):
        arch["dense_layers"] = [i for i, t in enumerate(arch["mlp_layer_types"]) if t == "dense"]
    if "full_indexer_layers" not in arch and isinstance(arch.get("indexer_types"), list):
        arch["full_indexer_layers"] = [i for i, t in enumerate(arch["indexer_types"]) if t == "full"]
    if "mtp_layers" not in arch and "num_nextn_predict_layers" in arch:
        arch["mtp_layers"] = arch["num_nextn_predict_layers"]
    cfg["architecture"] = arch
    if not isinstance(arch, dict):
        raise ValueError("architecture must be an object")
    required = ("hidden_size", "hc_mult", "num_heads", "qk_nope_head_dim", "qk_rope_head_dim",
                "v_head_dim", "q_lora_rank", "kv_lora_rank", "index_n_heads", "index_head_dim",
                "index_topk", "n_routed_experts", "num_experts_per_tok", "n_shared_experts",
                "moe_intermediate_size", "intermediate_size", "num_hidden_layers", "vocab_size")
    for key in required:
        if key not in arch:
            raise ValueError(f"architecture.{key} is required")
        _positive(arch[key], f"architecture.{key}")
    if arch["n_shared_experts"] != 1:
        raise ValueError("The workbook structure assumes exactly one shared expert")
    if arch["num_experts_per_tok"] > arch["n_routed_experts"]:
        raise ValueError("num_experts_per_tok cannot exceed n_routed_experts")
    n = arch["num_hidden_layers"]
    mlp_types = arch.get("mlp_layer_types", [])
    if mlp_types and (len(mlp_types) != n or any(t not in ("dense", "sparse") for t in mlp_types)):
        raise ValueError("mlp_layer_types must cover every layer with dense/sparse")
    idx_types = arch.get("indexer_types", [])
    if idx_types and (len(idx_types) != n or any(t not in ("full", "shared") for t in idx_types)):
        raise ValueError("indexer_types must cover every layer with full/shared")
    for key in ("dense_layers", "full_indexer_layers"):
        values = arch.get(key, [])
        if not isinstance(values, list) or any(isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < n for i in values):
            raise ValueError(f"{key} must contain valid zero-based backbone layer IDs")
        if values != sorted(set(values)):
            raise ValueError(f"{key} must be sorted and unique")
    dense = set(arch.get("dense_layers", [0]))
    full = set(arch.get("full_indexer_layers", []))
    if len(dense) != 1:
        raise ValueError("dense_layers must contain exactly one layer (the workbook uses layer 0)")
    # Topology parity with the workbook's 逐层拓扑 sheet: shared indexer
    # layers must be able to reuse a strictly earlier Full layer's Top-K.
    prev_full = None
    for i in range(n):
        if i in full:
            prev_full = i
        elif prev_full is None:
            raise ValueError(f"Layer {i} is Shared but no earlier Full Indexer layer exists")
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in (arch.get("mtp_layers", 0),)):
        raise ValueError("architecture.mtp_layers must be a nonnegative integer (excluded from analysis)")
    if not isinstance(arch.get("output_all_positions", True), bool):
        raise ValueError("architecture.output_all_positions must be a boolean")

    # Top-level keys consumed by ConfigParser / the standard report.
    for target, source in _TOP_LEVEL:
        if target in cfg and cfg[target] != arch[source]:
            raise ValueError(f"{target} disagrees with architecture.{source}")
        cfg[target] = arch[source]
    cfg.setdefault("max_seq_len", 4096)
    _positive(cfg["max_seq_len"], "max_seq_len")
    cfg.setdefault("dtype", "bf16")
    if cfg["dtype"] not in ("bf16", "fp16", "fp32"):
        raise ValueError("dtype must be bf16, fp16, or fp32")

    # Workbook byte assumptions (Hy4-10T sheet A38-A45): defaults match.
    byte_defaults = {"param": 2, "activation": 2, "fp32": 4, "grad": 2,
                     "optimizer": 8, "master": 0, "topk_index": 4, "logits": 4}
    by = cfg.setdefault("bytes", {})
    if not isinstance(by, dict):
        raise ValueError("bytes must be an object")
    for key, default in byte_defaults.items():
        by.setdefault(key, default)
        if isinstance(by[key], bool) or not isinstance(by[key], (int, float)) or by[key] < 0:
            raise ValueError(f"bytes.{key} must be a nonnegative number")

    training = cfg.setdefault("training", {})
    if not isinstance(training, dict):
        raise ValueError("training must be an object")
    devices = training.setdefault("num_devices", 1)
    _positive(devices, "training.num_devices")
    groups = training.setdefault("groups", {})
    if not isinstance(groups, dict) or set(groups) - {"dense", "expert", "buffer"}:
        raise ValueError("training.groups must be an object with dense/expert/buffer keys")
    for name in ("dense", "expert", "buffer"):
        group = groups.setdefault(name, {})
        if not isinstance(group, dict):
            raise ValueError(f"training.groups.{name} must be an object")
        defaults = {"weight_dtype": cfg["dtype"], "gradient_dtype": "bf16", "master_dtype": "none",
                    "optimizer_dtype": "fp32", "optimizer": "adamw",
                    "shards": 1, "device": "gpu", "trainable": name != "buffer"}
        for key, value in defaults.items():
            group.setdefault(key, value)
        if group["device"] not in ("gpu", "cpu") or not isinstance(group["trainable"], bool):
            raise ValueError(f"Invalid device/trainable in group {name}")
        if group["optimizer"] not in OPTIMIZER_STATES:
            raise ValueError(f"Unsupported optimizer in group {name}")
        for key in ("weight_dtype", "gradient_dtype", "master_dtype", "optimizer_dtype"):
            allowed = ("bf16", "fp16", "fp32", "none") if key == "master_dtype" else ("bf16", "fp16", "fp32")
            if group[key] not in allowed:
                raise ValueError(f"Unsupported {key} in group {name}; use {allowed}")
        _positive(group["shards"], f"{name}.shards")
        if group["shards"] > devices or devices % group["shards"]:
            raise ValueError(f"{name}.shards must divide num_devices")
    return cfg
