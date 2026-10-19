"""Load a pinned inference configuration plus an independent analysis scenario."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

from .accounting import OPTIMIZER_STATES


def _positive(value, name, allow_zero=False):
    if isinstance(value, bool) or not isinstance(value, int) or value < (0 if allow_zero else 1):
        raise ValueError(f"{name} must be a {'nonnegative' if allow_zero else 'positive'} integer")


def normalize_config(config: dict, base_dir: Path | None = None) -> dict:
    cfg = copy.deepcopy(config)
    if "architecture" in cfg:
        arch = cfg["architecture"]
    else:
        source = cfg.get("architecture_file")
        path = (base_dir or Path.cwd()) / source if source else Path(__file__).resolve().parent.parent / "config.json"
        data = path.read_bytes()
        arch = json.loads(data)
        cfg["architecture"] = arch
        cfg["source"] = {"architecture_file": str(path.resolve()),
                         "sha256": hashlib.sha256(data).hexdigest(),
                         "implementation": "DeepSeek-V4.1-Flash/inference/model.py",
                         "scope": "Text path; excludes vision tower/aligner and quantization metadata."}
    if not isinstance(arch, dict):
        raise ValueError("architecture must be an object")
    required = ("dim", "vocab_size", "n_layers", "moe_inter_dim", "n_heads", "head_dim",
                "q_lora_rank", "o_lora_rank", "o_groups", "window_size", "n_routed_experts",
                "n_activated_experts", "n_shared_experts", "index_n_heads", "index_head_dim",
                "index_topk", "hc_mult", "hc_sinkhorn_iters", "candidate_block_size", "candidate_topk_blocks")
    for key in required:
        if key not in arch:
            raise ValueError(f"architecture.{key} is required")
        _positive(arch[key], f"architecture.{key}", allow_zero=key == "n_shared_experts")
    if arch["n_heads"] % arch["o_groups"]:
        raise ValueError("n_heads must be divisible by o_groups")
    if arch["n_activated_experts"] > arch["n_routed_experts"]:
        raise ValueError("n_activated_experts cannot exceed n_routed_experts")
    if arch["n_shared_experts"] != 1:
        raise ValueError("The source implementation constructs exactly one shared expert")
    if cfg.get("include_vision", False):
        raise ValueError("Vision-tower analysis is not implemented; this version covers the text path")
    n = arch["n_layers"]
    ratios = arch.get("compress_ratios", [])
    if len(ratios) < n or any(isinstance(r, bool) or r not in (0, 1, 2) for r in ratios):
        raise ValueError("Flash compress_ratios must cover the backbone and contain only 0, 1, 2")
    for key in ("kv_source_layers", "index_source_layers"):
        values = arch.get(key, [])
        if not isinstance(values, list) or any(isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < n for i in values):
            raise ValueError(f"{key} must contain valid zero-based backbone layer IDs")
        if values != sorted(set(values)):
            raise ValueError(f"{key} must be sorted and unique")
    kv_sources = set(arch["kv_source_layers"])
    index_sources = set(arch["index_source_layers"])
    if not kv_sources <= index_sources:
        raise ValueError("Every KV source must also produce index keys and indices")
    owner = index_owner = None
    for i in range(n):
        if i in kv_sources:
            owner = i
        if i in index_sources:
            index_owner = i
        if ratios[i] == 0:
            if i in kv_sources or i in index_sources:
                raise ValueError("Pure SWA cannot own a global KV or Indexer")
        elif owner is None or index_owner is None or ratios[i] != ratios[owner]:
            raise ValueError(f"Layer {i} has no compatible earlier KV/Indexer source")
    candidate_source = arch.get("candidate_source_layer", -1)
    if candidate_source != -1 and candidate_source not in kv_sources:
        raise ValueError("candidate_source_layer must be a KV source, or -1")
    if candidate_source != -1 and any(ratios[i] != ratios[candidate_source]
                                     for i in index_sources if i > candidate_source):
        raise ValueError("Downstream indexers must share the candidate source's compressed-position length")
    for key in ("include_engram", "include_dspark"):
        cfg.setdefault(key, key == "include_engram")
        if not isinstance(cfg[key], bool):
            raise ValueError(f"{key} must be a boolean")
    if cfg["include_engram"]:
        ids, rows = arch.get("engram_layer_ids", []), arch.get("engram_num_embeddings", [])
        if len(ids) != len(rows) or ids != sorted(set(ids)):
            raise ValueError("Engram layer IDs and table sizes must match and IDs must be unique/sorted")
        if any(isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < n for i in ids):
            raise ValueError("Engram layer ID outside backbone")
        for row in rows:
            _positive(row, "engram_num_embeddings")
        for key in ("engram_n_heads", "engram_head_dim", "engram_max_ngram_size"):
            _positive(arch.get(key), key)
        if arch["engram_max_ngram_size"] < 2:
            raise ValueError("engram_max_ngram_size must be at least 2")
    if cfg["include_dspark"]:
        _positive(arch.get("n_mtp_layers"), "n_mtp_layers")
        for key in ("dspark_block_size", "dspark_markov_rank", "dspark_n_routed_experts", "dspark_n_activated_experts"):
            _positive(arch.get(key), key)
        if arch["dspark_n_activated_experts"] > arch["dspark_n_routed_experts"]:
            raise ValueError("DSpark activated experts cannot exceed routed experts")
        if len(ratios) < n + arch["n_mtp_layers"] or any(ratios[n:n + arch["n_mtp_layers"]]):
            raise ValueError("DSpark stages require explicit ratio-0 entries")
        targets = arch.get("dspark_target_layer_ids", [])
        if not targets or any(isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < n for i in targets):
            raise ValueError("DSpark target layers must be nonempty valid backbone IDs")
        if len(targets) != len(set(targets)):
            raise ValueError("DSpark target layers must be unique")
    for target, source in (("hidden_size", "dim"), ("num_layers", "n_layers"), ("vocab_size", "vocab_size")):
        if target in cfg and cfg[target] != arch[source]:
            raise ValueError(f"{target} disagrees with architecture.{source}")
        cfg[target] = arch[source]
    cfg.setdefault("max_seq_len", 4096)
    _positive(cfg["max_seq_len"], "max_seq_len")
    for key, default in (("dtype", "bf16"), ("activation_dtype", "bf16"),
                         ("logits_dtype", "fp32"), ("cache_dtype", "bf16")):
        cfg.setdefault(key, default)
        if cfg[key] not in ("bf16", "fp16", "fp32"):
            raise ValueError(f"{key} must be bf16, fp16, or fp32 in the first training-analysis version")
    training = cfg.setdefault("training", {})
    if not isinstance(training, dict):
        raise ValueError("training must be an object")
    devices = training.setdefault("num_devices", 1)
    _positive(devices, "training.num_devices")
    groups = training.setdefault("groups", {})
    if not isinstance(groups, dict) or set(groups) - {"dense", "expert", "engram"}:
        raise ValueError("training.groups must be an object with dense/expert/engram keys")
    for name in ("dense", "expert", "engram"):
        group = groups.setdefault(name, {})
        if not isinstance(group, dict):
            raise ValueError(f"training.groups.{name} must be an object")
        defaults = {"weight_dtype": cfg["dtype"], "gradient_dtype": "bf16", "master_dtype": "fp32",
                    "optimizer_dtype": "fp32", "optimizer": "momentum" if name == "engram" else "adamw",
                    "shards": 1, "device": "gpu", "trainable": True}
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
    buffers = training.setdefault("per_device_buffers_gb", {})
    if not isinstance(buffers, dict) or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in buffers.values()):
        raise ValueError("per_device_buffers_gb must contain finite nonnegative GB values")
    return cfg
