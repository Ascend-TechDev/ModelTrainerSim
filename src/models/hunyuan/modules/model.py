"""Hy4-preview 10T top-level model.

Assembles the 78-layer backbone described by the workbook sheets
Hy4-10T / 逐层拓扑 / 缓存与口径:

  * Layer 0: Dense FFN (no routing); layers 1-77: MoE (2048 experts, Top-64).
  * Full Indexer on layers {0, 1, 5, 9, ..., 77} (21 layers); the other 57
    Shared layers reuse the previous Full layer's Top-K indices.
  * Every layer: iHC sub-layer (fn + base/scale + pre/post) -> SublayerNorm
    -> Gated MLA (+ Indexer on Full layers) -> iHC sub-layer -> SublayerNorm
    -> FFN.  iHC expands the input into R=4 identity residual streams once
    at the backbone entry; the final iHC Head merges them after the last
    layer.
  * WordEmbedding and LM Head are NOT weight-tied.
  * The extra MTP layer (10.054B params) is excluded from the analysis,
    matching the workbook scope (Hy4-10T sheet D31).
  * Unsharded model; the parallel-config columns (TP1/PP4/VPP2/EP64/CP1)
    are informational and never divide anything.

``analyze()`` returns the standard report plus an Excel-aligned
``per_module_summary_excel`` whose rows mirror the workbook module table
(rows D4-D17: module / count / per-instance params / weights GiB /
outputs GiB).  Training states use the workbook byte assumptions
(param 2 / grad 2 / optimizer 8 / master 0; router bias excluded from
gradients and optimizer states).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ....core.base_module import BaseModule, ModuleStats
from ....core.dtype import DType
from .accounting import GroupedParamBreakdown, training_memory
from .configuration import normalize_config
from .ihc import iHCSublayer, SublayerNorm, IHCHead, FinalNorm
from .attention import GatedMLA
from .indexer import Indexer
from .dense_ffn import DenseFFN
from .moe import MoE
from .embedding import WordEmbedding, OutputLayer

# Excel-aligned module rows (Hy4-10T sheet D4-D17).
EXCEL_KEYS = ["wordembedding", "ihcinit", "ihc", "sublayernorm", "mla",
              "indexerproj", "indexersel", "denseffn", "moe", "ihchead",
              "finalnorm", "outputlayer"]

_MODULE_TO_EXCEL = {
    "WordEmbedding": "wordembedding",
    "iHC": "ihc",
    "SublayerNorm": "sublayernorm",
    "MLA": "mla",
    "DenseFFN": "denseffn",
    "MoE": "moe",
    "IHCHead": "ihchead",
    "FinalNorm": "finalnorm",
    "OutputLayer": "outputlayer",
}


@dataclass
class LayerSpec:
    """Description of a single Hy4 backbone layer."""
    layer_id: int
    ffn_type: str                                   # "Dense" | "MoE"
    use_indexer: bool                               # True on Full Indexer layers
    ihc_config: dict[str, Any] = field(default_factory=dict)
    norm_config: dict[str, Any] = field(default_factory=dict)
    mla_config: dict[str, Any] = field(default_factory=dict)
    indexer_config: dict[str, Any] = field(default_factory=dict)
    dense_ffn_config: dict[str, Any] = field(default_factory=dict)
    moe_config: dict[str, Any] = field(default_factory=dict)


class Hy4Model:
    """Top-level Hy4 model builder and analyzer (container, not a BaseModule)."""

    model_family = "Hy4"

    def __init__(self, config: dict[str, Any]):
        self.config = normalize_config(config)
        self.arch = self.config["architecture"]
        self.hidden_size: int = self.config["hidden_size"]
        self.num_layers: int = self.config["num_layers"]
        self.vocab_size: int = self.config["vocab_size"]
        self.seq_len: int = self.config.get("max_seq_len", 4096)
        self.dtype: DType = DType(self.config.get("dtype", "bf16"))
        self.bytes: dict[str, float] = self.config["bytes"]

        common = {"hidden_size": self.hidden_size, "dtype": self.dtype.value,
                  "bytes": self.bytes}
        self.embedding = WordEmbedding("embedding", dict(common, vocab_size=self.vocab_size))
        self.output = OutputLayer("output", dict(common, vocab_size=self.vocab_size,
                                                 output_all_positions=self.arch.get(
                                                     "output_all_positions", True)))
        self.ihc_head = IHCHead("final_ihc_head", dict(common, hc_mult=self.arch["hc_mult"]))
        self.final_norm = FinalNorm("final_rmsnorm", common)
        self.top_modules: list[BaseModule] = [self.embedding, self.ihc_head,
                                              self.final_norm, self.output]

        self.layer_specs: list[LayerSpec] = self._build_layer_specs()
        self.layers: list[list[BaseModule]] = self._build_layers()

    # -----------------------------------------------------------------
    # Layer construction
    # -----------------------------------------------------------------

    def _build_layer_specs(self) -> list[LayerSpec]:
        a = self.arch
        n = self.num_layers
        dense = set(a.get("dense_layers", [0]))
        full = set(a.get("full_indexer_layers", []))
        common = {"hidden_size": self.hidden_size, "dtype": self.dtype.value,
                  "bytes": self.bytes}
        ihc_cfg = dict(common, hc_mult=a["hc_mult"])
        mla_cfg = dict(common, num_heads=a["num_heads"],
                       qk_nope_head_dim=a["qk_nope_head_dim"],
                       qk_rope_head_dim=a["qk_rope_head_dim"],
                       v_head_dim=a["v_head_dim"], q_lora_rank=a["q_lora_rank"],
                       kv_lora_rank=a["kv_lora_rank"])
        idx_cfg = dict(common, q_lora_rank=a["q_lora_rank"],
                       index_n_heads=a["index_n_heads"],
                       index_head_dim=a["index_head_dim"], index_topk=a["index_topk"])
        dense_cfg = dict(common, intermediate_size=a["intermediate_size"])
        moe_cfg = dict(common, n_routed_experts=a["n_routed_experts"],
                       num_experts_per_tok=a["num_experts_per_tok"],
                       n_shared_experts=a["n_shared_experts"],
                       moe_intermediate_size=a["moe_intermediate_size"])
        specs = []
        for i in range(n):
            specs.append(LayerSpec(
                layer_id=i,
                ffn_type="Dense" if i in dense else "MoE",
                use_indexer=i in full,
                ihc_config=ihc_cfg, norm_config=common, mla_config=mla_cfg,
                indexer_config=idx_cfg, dense_ffn_config=dense_cfg, moe_config=moe_cfg,
            ))
        return specs

    def _build_layers(self) -> list[list[BaseModule]]:
        layers: list[list[BaseModule]] = []
        for spec in self.layer_specs:
            lid = spec.layer_id
            modules: list[BaseModule] = [
                iHCSublayer(f"layer{lid}_iHC_attn", spec.ihc_config),
                SublayerNorm(f"layer{lid}_norm_attn", spec.norm_config),
                GatedMLA(f"layer{lid}_MLA", spec.mla_config),
            ]
            if spec.use_indexer:
                modules.append(Indexer(f"layer{lid}_Indexer", spec.indexer_config))
            modules.append(iHCSublayer(f"layer{lid}_iHC_ffn", spec.ihc_config))
            modules.append(SublayerNorm(f"layer{lid}_norm_ffn", spec.norm_config))
            if spec.ffn_type == "Dense":
                modules.append(DenseFFN(f"layer{lid}_DenseFFN", spec.dense_ffn_config))
            else:
                modules.append(MoE(f"layer{lid}_MoE", spec.moe_config))
            layers.append(modules)
        return layers

    # -----------------------------------------------------------------
    # Analysis
    # -----------------------------------------------------------------

    def analyze(self, seq_len: int | None = None, batch_size: int = 1) -> dict[str, Any]:
        S = seq_len or self.seq_len
        B = batch_size

        # One walk over every module instance; all summaries derive from it.
        all_results: list[tuple[BaseModule, ModuleStats]] = (
            [(m, m.analyze(S, B)) for m in self.top_modules]
            + [(m, m.analyze(S, B)) for layer in self.layers for m in layer])

        # Instance summary (every BaseModule instance, module_type keys).
        per_module_summary: dict[str, dict[str, float]] = {}
        for mod, stats in all_results:
            summary = per_module_summary.setdefault(stats.module_type, {
                "params": 0, "bytes": 0.0, "flops": 0.0, "act_bytes": 0.0, "count": 0})
            summary["params"] += stats.num_params
            summary["bytes"] += stats.num_bytes
            summary["flops"] += stats.num_flops
            summary["act_bytes"] += stats.num_activations_bytes
            summary["count"] += 1

        # Excel-aligned module rows (workbook D4-D17; counts from column E).
        excel_summary: dict[str, dict[str, float]] = {
            key: {"params": 0, "bytes": 0.0, "flops": 0.0, "act_bytes": 0.0, "count": 0}
            for key in EXCEL_KEYS}
        for mod, stats in all_results:
            if isinstance(mod, Indexer):
                # Split into 投影 (params + projection GEMMs) and 选择 rows.
                proj_flops = mod.projection_flops(S, B)
                sel_flops = mod.selection_flops(S, B)
                proj_act = mod.projection_activations(S, B)
                sel_act = mod.selection_activations(S, B)
                s = excel_summary["indexerproj"]
                s["params"] += stats.num_params
                s["bytes"] += stats.num_bytes
                s["flops"] += proj_flops
                s["act_bytes"] += proj_act
                s["count"] += 1
                s = excel_summary["indexersel"]
                s["flops"] += sel_flops
                s["act_bytes"] += sel_act
                s["count"] += 1
                continue
            s = excel_summary[_MODULE_TO_EXCEL[stats.module_type]]
            s["params"] += stats.num_params
            s["bytes"] += stats.num_bytes
            s["flops"] += stats.num_flops
            s["act_bytes"] += stats.num_activations_bytes
            s["count"] += 1
        # Backbone-entry iHC expansion: params 0, one-time expand into R streams.
        ihc_init_act = B * S * self.arch["hc_mult"] * self.hidden_size * self.bytes["activation"]
        excel_summary["ihcinit"]["act_bytes"] += ihc_init_act
        excel_summary["ihcinit"]["count"] += 1

        total_params = int(sum(v["params"] for v in excel_summary.values()))
        total_bytes = float(sum(v["bytes"] for v in excel_summary.values()))
        total_flops = float(sum(v["flops"] for v in excel_summary.values()))
        total_act = float(sum(v["act_bytes"] for v in excel_summary.values()))

        # Training states: all parameter groups, workbook byte assumptions.
        parameters: list[GroupedParamBreakdown] = [
            p for _, stats in all_results for p in stats.param_breakdown]
        memory = training_memory(parameters, self.config["training"])
        groups = self.config["training"]["groups"]

        # Layer rows for the standard report (reuse the single walk).
        layer_reports = []
        top_count = len(self.top_modules)
        idx = top_count
        for spec, modules in zip(self.layer_specs, self.layers):
            layer_stats = all_results[idx: idx + len(modules)]
            idx += len(modules)
            layer_reports.append({
                "layer_id": spec.layer_id,
                "attention_type": "MLA",
                "ffn_type": spec.ffn_type,
                "use_indexer": spec.use_indexer,
                "params": sum(s.num_params for _, s in layer_stats),
                "bytes": sum(s.num_bytes for _, s in layer_stats),
                "flops": sum(s.num_flops for _, s in layer_stats),
                "act_bytes": sum(s.num_activations_bytes for _, s in layer_stats),
                "modules": [{"type": s.module_type, "name": m.name,
                             "params": s.num_params, "bytes": s.num_bytes,
                             "flops": s.num_flops, "act_bytes": s.num_activations_bytes}
                            for m, s in layer_stats],
            })

        return {
            "model_family": self.model_family,
            "config": {
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "vocab_size": self.vocab_size,
                "max_seq_len": self.seq_len,
                "dtype": self.dtype.value,
                "variant": self.config.get("variant", "10T"),
                "mtp_layers": self.arch.get("mtp_layers", 1),
            },
            "source": self.config.get("source", {}),
            "model_params": {
                "total_params": total_params,
                "trainable_params": sum(p.num_elements for p in parameters
                                        if groups[p.group]["trainable"]),
                "embedding_params": self.embedding.params_count(),
                "total_weight_bytes": total_bytes,
                "active_params_per_token": self._active_params(),
            },
            "compute": {
                "total_flops_per_sample": total_flops,
                "flops_per_token": total_flops / (B * S) if B * S > 0 else 0.0,
                "batch_size": B,
                "seq_len": S,
                "scope": "Backbone teacher-forced forward, T=S; GEMM main terms only "
                         "(2*M*N*Q). DSA counted on the eager QK/PV path. No backward, "
                         "kernel timing, or small elementwise operators.",
            },
            "memory": {
                "weight_memory_bytes": memory["weights"],
                "activation_memory_bytes": total_act,
                "activation_estimate_kind": "Sum of named forward tensors per the workbook "
                                            "dynamic inventory; not measured peak.",
                "kv_cache_bytes": self.estimate_kv_cache(S),
                "cache_scope": "Logical populated inference caches: MLA latent KV per layer, "
                               "Indexer K on Full layers, one in-flight Top-K index.",
                "training_total_bytes": memory["total"],
                "training_breakdown": memory,
            },
            "per_module_summary": per_module_summary,
            "per_module_summary_excel": excel_summary,
            "layers": layer_reports,
            "notes": [
                "Hy4-preview 10T is a planning extension, not an official release; "
                "dimensions follow the workbook planning assumptions (E=2048, Top-64).",
                "MTP layer excluded from the backbone analysis (workbook D31).",
                "Unsharded model; TP1/PP4/VPP2/EP64/CP1 are informational (workbook B7).",
                "Weights count in BF16-equivalent bytes; logits default to FP32.",
                "Router correction bias counts toward weights but not gradients/optimizer states.",
                "All displayed GiB use 1e9 bytes. JSON carries raw bytes.",
            ],
        }

    def _active_params(self) -> int:
        """Logical params touched per token (workbook A57: 342.569B).

        Full embedding table and full LM head; per layer the MLA, both iHC
        sub-layers and their norms, the router, the shared expert and
        K_act routed experts (Dense FFN on layer 0); the Indexer on Full
        layers; final iHC head and final norm.
        """
        first_layer = self.layers[0]
        ihc_one = sum(p.num_elements for p in first_layer[0].compute_params())
        norm_one = self.hidden_size
        mla_one = sum(p.num_elements for p in first_layer[2].compute_params())
        idx_modules = [m for m in first_layer if isinstance(m, Indexer)]
        idx_one = sum(p.num_elements for p in idx_modules[0].compute_params()) if idx_modules else 0

        active = (self.embedding.params_count() + self.output.params_count()
                  + self.ihc_head.params_count() + self.final_norm.params_count())
        for spec, modules in zip(self.layer_specs, self.layers):
            active += mla_one + 2 * ihc_one + 2 * norm_one
            if spec.use_indexer:
                active += idx_one
            for mod in modules:
                if isinstance(mod, MoE):
                    info = mod._extra_info()
                    active += info["router_params"] + info["shared_expert_params"] \
                        + self.arch["num_experts_per_tok"] * info["per_routed_expert_params"]
                elif isinstance(mod, DenseFFN):
                    active += mod.params_count()
        return active

    # -----------------------------------------------------------------
    # KV cache estimate
    # -----------------------------------------------------------------

    def estimate_kv_cache(self, seq_len: int) -> float:
        """Workbook 缓存与口径 sheet: MLA latent KV per layer (512+64 BF16),
        Indexer K on Full layers (128 BF16), and one in-flight Top-K index
        (int32, index length min(index_topk, S)).  Per request; no batch
        multiplier, no per-head expansion for the latent cache."""
        a = self.arch
        S = seq_len
        b = self.bytes
        ml = self.num_layers * S * (a["kv_lora_rank"] + a["qk_rope_head_dim"]) * b["activation"]
        idx_k = len(a.get("full_indexer_layers", [])) * S * a["index_head_dim"] * b["activation"]
        topk = S * min(a["index_topk"], S) * b["topk_index"]
        return ml + idx_k + topk
