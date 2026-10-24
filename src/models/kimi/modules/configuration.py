"""Load the pinned common Kimi 10T structure plus a variant analysis scenario.

Follows the Hy4/DSV41Flash ``normalize_config`` pattern: the pinned file
(``config.json`` next to this package) holds the structure shared by all
size variants; a scenario file (``configs/kimi_10t_{deep,wide}_v2_config.json``)
supplies the variant-specific ``architecture`` fields (layer counts /
expert dims) plus analysis settings (seq len, dtype, training groups,
parallelism).  The pinned file alone is not a complete model.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

_TOP_LEVEL = (("hidden_size", "hidden_size"), ("num_layers", "num_hidden_layers"),
              ("vocab_size", "vocab_size"))


def _positive(value, name, allow_zero=False):
    if isinstance(value, bool) or not isinstance(value, int) or value < (0 if allow_zero else 1):
        raise ValueError(f"{name} must be a {'nonnegative' if allow_zero else 'positive'} integer")


def _load_architecture_file(cfg: dict, base_dir: Path | None) -> tuple[dict, Path, str]:
    source = cfg.get("architecture_file")
    path = (base_dir or Path.cwd()) / source if source else Path(__file__).resolve().parent.parent / "config.json"
    data = path.read_bytes()
    loaded = json.loads(data)
    return loaded, path, hashlib.sha256(data).hexdigest()


def normalize_config(config: dict, base_dir: Path | None = None) -> dict:
    cfg = copy.deepcopy(config)
    if "architecture_file" in cfg or "architecture" not in cfg:
        loaded, path, digest = _load_architecture_file(cfg, base_dir)
        arch = {**loaded.get("architecture", loaded), **cfg.pop("architecture", {})}
        cfg["architecture"] = arch
        cfg["architecture_file"] = str(path.resolve())
        for key in ("variant", "description"):
            if key not in cfg and key in loaded:
                cfg[key] = loaded[key]
        # Parallelism merges key-wise over the pinned base block.
        cfg["parallelism"] = {**loaded.get("parallelism", {}), **cfg.get("parallelism", {})}
        cfg.setdefault("bytes", loaded.get("bytes", {}))
        cfg["source"] = {**loaded.get("source", {}), **cfg.get("source", {}),
                         "architecture_file": str(path.resolve()),
                         "sha256": digest}
    elif "parallelism" not in cfg:
        cfg["parallelism"] = {}
    if not isinstance(cfg.get("architecture"), dict):
        raise ValueError("architecture must be an object")
    arch = cfg["architecture"]
    required = ("hidden_size", "num_heads", "qk_nope_head_dim", "qk_rope_head_dim",
                "v_head_dim", "head_dim", "q_lora_rank", "kv_lora_rank", "vocab_size",
                "intermediate_size", "moe_intermediate_size", "latent_moe_size", "conv_size",
                "num_hidden_layers", "moe_layers", "mla_layers", "kda_layers",
                "n_routed_experts", "n_shared_experts", "top_k")
    for key in required:
        if key not in arch:
            raise ValueError(f"architecture.{key} is required")
        _positive(arch[key], f"architecture.{key}", allow_zero=key == "kda_layers")
    arch.setdefault("dense_layers", 1)
    _positive(arch["dense_layers"], "architecture.dense_layers")
    if arch["num_hidden_layers"] != arch["dense_layers"] + arch["moe_layers"]:
        raise ValueError("num_hidden_layers must equal dense_layers + moe_layers")
    attention = arch["mla_layers"] + arch["kda_layers"]
    if attention == arch["moe_layers"]:
        # 深度-v2 style: the Dense layer carries no attention module.
        arch["dense_attention"] = "none"
    elif attention == arch["num_hidden_layers"]:
        # 宽度-v2 style: the Dense layer carries an MLA attention
        # (the workbook only multiplies counts — placement is our choice).
        arch["dense_attention"] = "mla"
    else:
        raise ValueError("mla_layers + kda_layers must equal moe_layers or "
                         "num_hidden_layers (attention coverage)")
    if arch["top_k"] > arch["n_routed_experts"]:
        raise ValueError("top_k cannot exceed n_routed_experts")
    if not isinstance(arch.get("output_all_positions", True), bool):
        raise ValueError("architecture.output_all_positions must be a boolean")

    for target, source in _TOP_LEVEL:
        if target in cfg and cfg[target] != arch[source]:
            raise ValueError(f"{target} disagrees with architecture.{source}")
        cfg[target] = arch[source]
    cfg.setdefault("max_seq_len", 4096)
    _positive(cfg["max_seq_len"], "max_seq_len")
    cfg.setdefault("dtype", "bf16")
    if cfg["dtype"] not in ("bf16", "fp16", "fp32"):
        raise ValueError("dtype must be bf16, fp16, or fp32")

    byte_defaults = {"param": 2, "activation": 2, "fp32": 4, "grad": 2,
                     "optimizer_adamw": 12, "optimizer_muon": 8, "master": 0, "logits": 4}
    by = cfg.setdefault("bytes", {})
    for key, default in byte_defaults.items():
        by.setdefault(key, default)
        if isinstance(by[key], bool) or not isinstance(by[key], (int, float)) or by[key] < 0:
            raise ValueError(f"bytes.{key} must be a nonnegative number")

    par = cfg["parallelism"]
    for key in ("world_size", "tp", "ep", "pp", "cp", "dp", "etp"):
        par.setdefault(key, 1)
        _positive(par[key], f"parallelism.{key}")
    for key, default in (("mbs", 1), ("gbs", 1)):
        par.setdefault(key, default)
        _positive(par[key], f"parallelism.{key}")
    if par["world_size"] % (par["tp"] * par["ep"] * par["pp"] * par["cp"]):
        raise ValueError("world_size must be divisible by tp*ep*pp*cp")
    par["edp"] = par["world_size"] // (par["tp"] * par["ep"] * par["pp"] * par["cp"])

    training = cfg.setdefault("training", {})
    if not isinstance(training, dict):
        raise ValueError("training must be an object")
    devices = training.setdefault("num_devices", par["world_size"])
    _positive(devices, "training.num_devices")
    groups = training.setdefault("groups", {})
    if not isinstance(groups, dict) or set(groups) - {"adamw", "muon"}:
        raise ValueError("training.groups must be an object with adamw/muon keys")
    for name in ("adamw", "muon"):
        group = groups.setdefault(name, {})
        if not isinstance(group, dict):
            raise ValueError(f"training.groups.{name} must be an object")
        defaults = {"weight_dtype": cfg["dtype"], "gradient_dtype": "bf16", "master_dtype": "none",
                    "optimizer": "adamw" if name == "adamw" else "muon",
                    "optimizer_bytes_per_param": by["optimizer_adamw"] if name == "adamw"
                    else by["optimizer_muon"],
                    "shards": 1, "device": "gpu", "trainable": True}
        for key, value in defaults.items():
            group.setdefault(key, value)
        if group["device"] not in ("gpu", "cpu") or not isinstance(group["trainable"], bool):
            raise ValueError(f"Invalid device/trainable in group {name}")
        if group["optimizer"] not in ("adamw", "muon"):
            raise ValueError(f"Unsupported optimizer in group {name}")
        for key in ("weight_dtype", "gradient_dtype", "master_dtype"):
            allowed = ("bf16", "fp16", "fp32", "none") if key == "master_dtype" else ("bf16", "fp16", "fp32")
            if group[key] not in allowed:
                raise ValueError(f"Unsupported {key} in group {name}; use {allowed}")
        if (isinstance(group["optimizer_bytes_per_param"], bool)
                or not isinstance(group["optimizer_bytes_per_param"], (int, float))
                or group["optimizer_bytes_per_param"] < 0):
            raise ValueError(f"optimizer_bytes_per_param in group {name} must be nonnegative")
    return cfg
