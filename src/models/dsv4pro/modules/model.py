"""DSV4Pro top-level model.

Assembles a stack of transformer layers, each containing:
  - An attention module (CSA / HCA / mHC depending on layer config)
  - Optionally an Indexer for sparse routing
  - An MoE FFN (or dense FFN for the first N layers)
  - Layer norms / residuals

The model also handles:
  - Token embedding + output head
  - KV-cache memory estimation
  - Training memory estimation (gradients + optimizer states)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ....core.base_module import BaseModule, ModuleStats, ModuleParamBreakdown
from ....core.dtype import DType, get_dtype_size
from .csa import CSA
from .hca import HCA
from .mhc import mHC
from .indexer import Indexer
from .moe import MoE


# ---------------------------------------------------------------------------
# Layer specification
# ---------------------------------------------------------------------------

@dataclass
class LayerSpec:
    """Description of a single transformer layer."""
    layer_id: int
    attention_type: str       # "CSA", "HCA", "mHC"
    attention_config: dict[str, Any]
    use_indexer: bool = False
    indexer_config: dict[str, Any] | None = None
    ffn_type: str = "MoE"     # "MoE" or "Dense"
    ffn_config: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Dense FFN (for first N layers in DSV4Pro)
# ---------------------------------------------------------------------------

class DenseFFN(BaseModule):
    module_type = "DenseFFN"

    def __init__(self, name: str, config: dict[str, Any]):
        super().__init__(name, config)
        self.hidden_size: int = config["hidden_size"]
        self.intermediate_size: int = config["intermediate_size"]
        self.dtype: DType = DType(config.get("dtype", "bf16"))

    def compute_params(self) -> list[ModuleParamBreakdown]:
        H, I = self.hidden_size, self.intermediate_size
        return [
            ModuleParamBreakdown("gate_proj", (H, I), H * I, self.dtype),
            ModuleParamBreakdown("down_proj", (I, H), I * H, self.dtype),
            ModuleParamBreakdown("up_proj", (H, I), H * I, self.dtype),
            ModuleParamBreakdown("input_layernorm", (H,), H, self.dtype),
            ModuleParamBreakdown("post_attention_layernorm", (H,), H, self.dtype),
        ]

    def compute_flops(self, seq_len: int, batch_size: int = 1) -> float:
        B, S = batch_size, seq_len
        H, I = self.hidden_size, self.intermediate_size
        return 3 * 2 * B * S * H * I

    def compute_activations(self, seq_len: int, batch_size: int = 1) -> float:
        B, S = batch_size, seq_len
        H, I = self.hidden_size, self.intermediate_size
        elem = get_dtype_size(self.dtype)
        return B * S * I * elem * 2 + B * S * H * elem


# ---------------------------------------------------------------------------
# Attention module factory
# ---------------------------------------------------------------------------

_ATTENTION_REGISTRY: dict[str, type[BaseModule]] = {
    "CSA": CSA,
    "HCA": HCA,
    "mHC": mHC,
}


def _build_attention(layer_spec: LayerSpec) -> BaseModule:
    cls = _ATTENTION_REGISTRY.get(layer_spec.attention_type)
    if cls is None:
        raise ValueError(f"Unknown attention type: {layer_spec.attention_type}")
    return cls(f"layer{layer_spec.layer_id}_{layer_spec.attention_type}",
               layer_spec.attention_config)


def _build_indexer(layer_spec: LayerSpec) -> Indexer | None:
    if not layer_spec.use_indexer:
        return None
    return Indexer(f"layer{layer_spec.layer_id}_Indexer",
                   layer_spec.indexer_config or {})


def _build_ffn(layer_spec: LayerSpec) -> BaseModule:
    if layer_spec.ffn_type == "MoE":
        return MoE(f"layer{layer_spec.layer_id}_MoE", layer_spec.ffn_config)
    elif layer_spec.ffn_type == "Dense":
        return DenseFFN(f"layer{layer_spec.layer_id}_DenseFFN", layer_spec.ffn_config)
    else:
        raise ValueError(f"Unknown FFN type: {layer_spec.ffn_type}")


# ---------------------------------------------------------------------------
# Embedding & output head
# ---------------------------------------------------------------------------

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
        params = [ModuleParamBreakdown(
            "token_embedding", (V, H), V * H, self.dtype)]
        if not self.tie_word_embeddings:
            params.append(ModuleParamBreakdown(
                "lm_head", (H, V), V * H, self.dtype))
        return params

    def compute_flops(self, seq_len: int, batch_size: int = 1) -> float:
        # Embedding lookup is negligible (gather)
        return 0.0

    def compute_activations(self, seq_len: int, batch_size: int = 1) -> float:
        B, S = batch_size, seq_len
        elem = get_dtype_size(self.dtype)
        return B * S * self.hidden_size * elem


# ---------------------------------------------------------------------------
# DSV4Pro Model
# ---------------------------------------------------------------------------

class DSV4ProModel:
    """Top-level DSV4Pro model builder and analyzer.

    Not a ``BaseModule`` itself — it is a *container* that builds and
    analyzes a stack of layers, embedding, and output head.
    """

    model_family = "DSV4Pro"

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.hidden_size: int = config["hidden_size"]
        self.num_layers: int = config["num_layers"]
        self.vocab_size: int = config["vocab_size"]
        self.seq_len: int = config.get("max_seq_len", 4096)
        self.tie_word_embeddings: bool = config.get("tie_word_embeddings", False)
        self.dtype: DType = DType(config.get("dtype", "bf16"))

        # Embedding
        self.embedding = Embedding("embedding", {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "dtype": self.dtype.value,
            "tie_word_embeddings": self.tie_word_embeddings,
        })

        # Build layer specs
        self.layer_specs: list[LayerSpec] = self._build_layer_specs()
        self.layers: list[list[BaseModule]] = self._build_layers()

    # -----------------------------------------------------------------
    # Layer construction
    # -----------------------------------------------------------------

    def _build_layer_specs(self) -> list[LayerSpec]:
        specs: list[LayerSpec] = []
        layer_config = self.config.get("layers", {})

        # First N layers are dense FFN, rest are MoE
        first_k_dense = layer_config.get("first_k_dense_replace", 0)
        mtp_layers = layer_config.get("mtp_layers", 0)
        effective_layers = self.num_layers - mtp_layers

        # Attention type per layer
        # CSA: first few layers, HCA: middle, mHC: later layers
        csa_layers = layer_config.get("csa_layers", first_k_dense)
        hca_layers = layer_config.get("hca_layers", 0)
        # mHC for the rest

        # Shared attention config
        attn_base = layer_config.get("attention", {})

        # MoE config
        moe_config = layer_config.get("moe", {})
        # Dense FFN config
        dense_ffn_config = layer_config.get("dense_ffn", {})

        # Indexer config (applied to HCA layers)
        indexer_config = layer_config.get("indexer", {})

        for i in range(effective_layers):
            # Determine attention type
            if i < csa_layers:
                attn_type = "CSA"
            elif i < csa_layers + hca_layers:
                attn_type = "HCA"
            else:
                attn_type = "mHC"

            attn_cfg = dict(attn_base)
            attn_cfg.setdefault("hidden_size", self.hidden_size)
            attn_cfg.setdefault("dtype", self.dtype.value)

            # mHC uses GQA — map num_heads to num_query_heads / num_kv_heads
            if attn_type == "mHC":
                n_h = attn_cfg.get("num_heads", 128)
                attn_cfg.setdefault("num_query_heads", n_h)
                attn_cfg.setdefault("num_kv_heads", max(n_h // 4, 8))

            # Determine FFN type
            if i < first_k_dense:
                ffn_type = "Dense"
                ffn_cfg = dict(dense_ffn_config)
                ffn_cfg.setdefault("hidden_size", self.hidden_size)
                ffn_cfg.setdefault("dtype", self.dtype.value)
            else:
                ffn_type = "MoE"
                ffn_cfg = dict(moe_config)
                ffn_cfg.setdefault("hidden_size", self.hidden_size)
                ffn_cfg.setdefault("dtype", self.dtype.value)

            # Indexer only for HCA layers (if enabled)
            use_indexer = (attn_type == "HCA" and indexer_config.get("enabled", True))

            idx_cfg = None
            if use_indexer:
                idx_cfg = dict(indexer_config)
                idx_cfg.pop("enabled", None)
                idx_cfg.setdefault("hidden_size", self.hidden_size)
                idx_cfg.setdefault("dtype", self.dtype.value)

            specs.append(LayerSpec(
                layer_id=i,
                attention_type=attn_type,
                attention_config=attn_cfg,
                use_indexer=use_indexer,
                indexer_config=idx_cfg,
                ffn_type=ffn_type,
                ffn_config=ffn_cfg,
            ))

        return specs

    def _build_layers(self) -> list[list[BaseModule]]:
        layers: list[list[BaseModule]] = []
        for spec in self.layer_specs:
            modules: list[BaseModule] = []
            modules.append(_build_attention(spec))
            idx = _build_indexer(spec)
            if idx is not None:
                modules.append(idx)
            modules.append(_build_ffn(spec))
            layers.append(modules)
        return layers

    # -----------------------------------------------------------------
    # Analysis
    # -----------------------------------------------------------------

    def analyze(self, seq_len: int | None = None,
                batch_size: int = 1) -> dict[str, Any]:
        """Run full analysis and return a structured report."""
        S = seq_len or self.seq_len
        B = batch_size

        total_params = 0
        total_bytes = 0.0
        total_flops = 0.0
        total_act = 0.0

        # Embedding
        emb_stats = self.embedding.analyze(S, B)
        total_params += emb_stats.num_params
        total_bytes += emb_stats.num_bytes
        total_flops += emb_stats.num_flops
        total_act += emb_stats.num_activations_bytes

        # Per-layer analysis
        layer_reports: list[dict[str, Any]] = []
        per_module_summary: dict[str, dict[str, float]] = {}

        for spec, modules in zip(self.layer_specs, self.layers):
            layer_params = 0
            layer_bytes = 0.0
            layer_flops = 0.0
            layer_act = 0.0
            module_details: list[dict[str, Any]] = []

            for mod in modules:
                stats = mod.analyze(S, B)
                layer_params += stats.num_params
                layer_bytes += stats.num_bytes
                layer_flops += stats.num_flops
                layer_act += stats.num_activations_bytes

                module_details.append({
                    "type": stats.module_type,
                    "name": mod.name,
                    "params": stats.num_params,
                    "bytes": stats.num_bytes,
                    "flops": stats.num_flops,
                    "act_bytes": stats.num_activations_bytes,
                    "extra": stats.extra,
                })

                summary = per_module_summary.setdefault(stats.module_type, {
                    "params": 0, "bytes": 0.0, "flops": 0.0, "act_bytes": 0.0, "count": 0
                })
                summary["params"] += stats.num_params
                summary["bytes"] += stats.num_bytes
                summary["flops"] += stats.num_flops
                summary["act_bytes"] += stats.num_activations_bytes
                summary["count"] += 1

            total_params += layer_params
            total_bytes += layer_bytes
            total_flops += layer_flops
            total_act += layer_act

            layer_reports.append({
                "layer_id": spec.layer_id,
                "attention_type": spec.attention_type,
                "use_indexer": spec.use_indexer,
                "ffn_type": spec.ffn_type,
                "params": layer_params,
                "bytes": layer_bytes,
                "flops": layer_flops,
                "act_bytes": layer_act,
                "modules": module_details,
            })

        # KV cache estimate
        kv_cache = self.estimate_kv_cache(S, B)

        # Training memory estimate
        training_mem = self.estimate_training_memory(total_bytes, S, B)

        return {
            "model_family": self.model_family,
            "config": {
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "vocab_size": self.vocab_size,
                "max_seq_len": self.seq_len,
                "dtype": self.dtype.value,
            },
            "model_params": {
                "total_params": total_params,
                "embedding_params": emb_stats.num_params,
                "total_weight_bytes": total_bytes,
                "active_params_per_token": self._active_params(total_params),
            },
            "compute": {
                "total_flops_per_sample": total_flops,
                "flops_per_token": total_flops / S if S > 0 else 0.0,
                "batch_size": B,
                "seq_len": S,
            },
            "memory": {
                "weight_memory_bytes": total_bytes,
                "activation_memory_bytes": total_act,
                "kv_cache_bytes": kv_cache,
                "training_total_bytes": training_mem["total"],
                "training_breakdown": training_mem,
            },
            "per_module_summary": per_module_summary,
            "layers": layer_reports,
        }

    def _active_params(self, total_params: int) -> int:
        """Estimate active parameters per token (MoE sparse activation)."""
        # Embedding params are always active
        emb_params = self.vocab_size * self.hidden_size * (1 if not self.tie_word_embeddings else 1)
        # Each MoE layer activates only num_activated_experts + shared_experts
        active = emb_params
        for spec, modules in zip(self.layer_specs, self.layers):
            for mod in modules:
                if isinstance(mod, MoE):
                    info = mod._extra_info()
                    active += info["activated_expert_params"] + info["shared_expert_params"]
                    # gate is always active
                    active += self.hidden_size * mod.num_experts
                else:
                    active += mod.params_count()
        return active

    # -----------------------------------------------------------------
    # KV cache estimate
    # -----------------------------------------------------------------

    def estimate_kv_cache(self, seq_len: int, batch_size: int = 1) -> float:
        """Estimate KV cache memory for the given sequence length.

        DSV4Pro uses MLA-style compressed KV cache: only the compressed
        KV representation (kv_lora_rank + rope_dim) is stored per token,
        not the full per-head K/V.
        """
        B, S = batch_size, seq_len
        elem = get_dtype_size(self.dtype)
        total = 0.0

        for spec, modules in zip(self.layer_specs, self.layers):
            for mod in modules:
                kv_lora = mod.config.get("kv_lora_rank", 0)
                if kv_lora > 0:
                    # Compressed KV: (kv_lora_rank + rope_dim) per token
                    rope_dim = mod.config.get("qk_rope_head_dim", 64)
                    kv_per_token = kv_lora + rope_dim
                    total += B * S * kv_per_token * elem
                elif mod.module_type in ("CSA", "HCA", "mHC"):
                    # Dense KV cache (less common in DSV4Pro)
                    n_kv = mod.config.get("num_kv_heads", mod.config.get("num_heads", 1))
                    head_dim = mod.config.get("head_dim", 128)
                    total += B * S * 2 * n_kv * head_dim * elem
        return total

    # -----------------------------------------------------------------
    # Training memory estimate
    # -----------------------------------------------------------------

    @staticmethod
    def estimate_training_memory(weight_bytes: float, seq_len: int,
                                  batch_size: int = 1,
                                  optimizer: str = "adamw") -> dict[str, float]:
        """Estimate total GPU memory for training.

        Memory components:
          - Weights (bf16)
          - Gradients (bf16, same as weights)
          - Optimizer states (adamw: m + v in fp32 = 8 bytes/param;
                              adamw_8bit: 2 bytes/param)
          - Master weights (fp32, = 2x weight bytes for bf16)
          - Activations
          - KV cache (for inference / generation)
        """
        # weight_bytes is bf16 (2 bytes/param)
        bytes_per_param = 2.0
        param_element_bytes = weight_bytes / bytes_per_param  # not used, but keep for clarity

        grad_bytes = weight_bytes          # gradients in bf16
        master_weight_bytes = weight_bytes * 2  # fp32 master copy

        if optimizer == "adamw":
            # m + v in fp32: 4 + 4 = 8 bytes per param
            opt_bytes = param_element_bytes * 8
        elif optimizer == "adamw_8bit":
            opt_bytes = param_element_bytes * 2
        elif optimizer == "sgd":
            opt_bytes = param_element_bytes * 4   # momentum in fp32
        else:
            opt_bytes = param_element_bytes * 8

        total = (weight_bytes + grad_bytes + master_weight_bytes
                 + opt_bytes)

        return {
            "weights": weight_bytes,
            "gradients": grad_bytes,
            "master_weights": master_weight_bytes,
            "optimizer_states": opt_bytes,
            "total": total,
            "optimizer": optimizer,
            "note": (
                "Excludes activation and KV cache.  Add "
                "memory['activation_memory_bytes'] and memory['kv_cache_bytes'] "
                "for full estimate."
            ),
        }