"""Kimi 10T top-level model.

Assembles the workbook topology (sheet 10T-深度-v2 / 10T-宽度-v2):
  * Layer 0: Dense FFN only (rows I15-I18 — attention params are counted
    only on the MoE layers, I13*D25 + I51*D24).
  * MoE layers: MLA on the first ``mla_layers``, KDA on the remaining
    ``kda_layers``, each followed by a Latent-MoE FFN.
  * WordEmbedding and LM Head are not tied (I2 and I35 both = D9*D19);
    pre_norm/post_norm (I3/I4) and final norm (I34) are counted once.

The param totals reproduce the workbook formula chain exactly:
  I55 = I2+I3+I4+I13*D25+I18*D5+I32*D6+I51*D24+I34+I35
  deep-v2 = 9,283,004,226,880 / wide-v2 = 11,917,686,210,656.
FLOPs follow the R-column GEMMs (per mbs token; the S column scales by
gbs/mbs).  Activation tensors follow the N-column counts with the
M30/M32/M33 aggregation conventions.  Training states use AdamW 12 /
Muon 8 bytes per param and the workbook sharding 口径 (G63/G68/K66).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ....core.base_module import BaseModule, ModuleStats
from ....core.dtype import DType
from .accounting import GroupedParamBreakdown, training_memory
from .configuration import normalize_config
from .embedding import WordEmbedding, OutputLayer, PreNorm, PostNorm, FinalNorm
from .attention import MLA
from .kda import KDA
from .dense_ffn import DenseFFN
from .moe import LatentMoE

EXCEL_KEYS = ["wordembedding", "prenorm", "postnorm", "finalnorm", "mla",
              "kda", "denseffn", "moe", "outputlayer"]

_MODULE_TO_EXCEL = {
    "WordEmbedding": "wordembedding",
    "PreNorm": "prenorm",
    "PostNorm": "postnorm",
    "FinalNorm": "finalnorm",
    "MLA": "mla",
    "KDA": "kda",
    "DenseFFN": "denseffn",
    "MoE": "moe",
    "OutputLayer": "outputlayer",
}


@dataclass
class LayerSpec:
    layer_id: int
    attn_type: str                                   # "MLA" | "KDA" | "None"
    ffn_type: str                                    # "Dense" | "MoE"
    mla_config: dict[str, Any] = field(default_factory=dict)
    kda_config: dict[str, Any] = field(default_factory=dict)
    dense_ffn_config: dict[str, Any] = field(default_factory=dict)
    moe_config: dict[str, Any] = field(default_factory=dict)


class KimiModel:
    """Top-level Kimi 10T model builder and analyzer."""

    model_family = "Kimi-10T"

    def __init__(self, config: dict[str, Any]):
        self.config = normalize_config(config)
        self.arch = self.config["architecture"]
        self.hidden_size: int = self.config["hidden_size"]
        self.num_layers: int = self.config["num_layers"]
        self.vocab_size: int = self.config["vocab_size"]
        self.seq_len: int = self.config.get("max_seq_len", 4096)
        self.dtype: DType = DType(self.config.get("dtype", "bf16"))
        self.bytes: dict[str, float] = self.config["bytes"]
        self.parallelism: dict[str, Any] = self.config["parallelism"]

        common = {"hidden_size": self.hidden_size, "dtype": self.dtype.value,
                  "bytes": self.bytes}
        self.embedding = WordEmbedding("embedding", dict(common, vocab_size=self.vocab_size))
        self.pre_norm = PreNorm("pre_norm", common)
        self.post_norm = PostNorm("post_norm", common)
        self.final_norm = FinalNorm("final_norm", common)
        self.output = OutputLayer("output", dict(common, vocab_size=self.vocab_size,
                                                 output_all_positions=self.arch.get(
                                                     "output_all_positions", True)))
        self.top_modules: list[BaseModule] = [self.embedding, self.pre_norm,
                                              self.post_norm, self.final_norm, self.output]

        self.layer_specs: list[LayerSpec] = self._build_layer_specs()
        self.layers: list[list[BaseModule]] = self._build_layers()

    # -----------------------------------------------------------------

    def _build_layer_specs(self) -> list[LayerSpec]:
        a = self.arch
        common = {"hidden_size": self.hidden_size, "dtype": self.dtype.value,
                  "bytes": self.bytes}
        mla_cfg = dict(common, num_heads=a["num_heads"],
                       qk_nope_head_dim=a["qk_nope_head_dim"],
                       qk_rope_head_dim=a["qk_rope_head_dim"],
                       v_head_dim=a["v_head_dim"], q_lora_rank=a["q_lora_rank"],
                       kv_lora_rank=a["kv_lora_rank"])
        kda_cfg = dict(common, num_heads=a["num_heads"], head_dim=a["head_dim"],
                       conv_size=a["conv_size"])
        dense_cfg = dict(common, intermediate_size=a["intermediate_size"])
        moe_cfg = dict(common, n_routed_experts=a["n_routed_experts"],
                       n_shared_experts=a["n_shared_experts"], top_k=a["top_k"],
                       latent_moe_size=a["latent_moe_size"],
                       moe_intermediate_size=a["moe_intermediate_size"])
        specs = []
        # Layer 0: Dense FFN.  深度-v2: no attention on the Dense layer
        # (mla+kda == moe_layers).  宽度-v2: the Dense layer carries an MLA
        # (mla+kda == num_hidden_layers).  The workbook only multiplies the
        # attention counts, so the placement is ours.
        if a["dense_attention"] == "mla":
            specs.append(LayerSpec(0, "MLA", "Dense", mla_config=mla_cfg,
                                   dense_ffn_config=dense_cfg))
        else:
            specs.append(LayerSpec(0, "None", "Dense", dense_ffn_config=dense_cfg))
        mla_remaining = a["mla_layers"] - (1 if a["dense_attention"] == "mla" else 0)
        for i in range(1, self.num_layers):
            attn = "MLA" if i - 1 < mla_remaining else "KDA"
            specs.append(LayerSpec(i, attn, "MoE",
                                   mla_config=mla_cfg, kda_config=kda_cfg, moe_config=moe_cfg))
        return specs

    def _build_layers(self) -> list[list[BaseModule]]:
        layers: list[list[BaseModule]] = []
        for spec in self.layer_specs:
            modules: list[BaseModule] = []
            if spec.attn_type == "MLA":
                modules.append(MLA(f"layer{spec.layer_id}_MLA", spec.mla_config))
            elif spec.attn_type == "KDA":
                modules.append(KDA(f"layer{spec.layer_id}_KDA", spec.kda_config))
            if spec.ffn_type == "Dense":
                modules.append(DenseFFN(f"layer{spec.layer_id}_DenseFFN",
                                        spec.dense_ffn_config))
            else:
                modules.append(LatentMoE(f"layer{spec.layer_id}_MoE", spec.moe_config))
            layers.append(modules)
        return layers

    # -----------------------------------------------------------------

    def analyze(self, seq_len: int | None = None, batch_size: int = 1) -> dict[str, Any]:
        S = seq_len or self.seq_len
        B = batch_size

        all_results: list[tuple[BaseModule, ModuleStats]] = (
            [(m, m.analyze(S, B)) for m in self.top_modules]
            + [(m, m.analyze(S, B)) for layer in self.layers for m in layer])

        per_module_summary: dict[str, dict[str, float]] = {}
        for mod, stats in all_results:
            summary = per_module_summary.setdefault(stats.module_type, {
                "params": 0, "bytes": 0.0, "flops": 0.0, "act_bytes": 0.0, "count": 0})
            summary["params"] += stats.num_params
            summary["bytes"] += stats.num_bytes
            summary["flops"] += stats.num_flops
            summary["act_bytes"] += stats.num_activations_bytes
            summary["count"] += 1

        # Workbook module rows (I-column terms; counts D5/D6/D24/D25).
        excel_summary: dict[str, dict[str, float]] = {
            key: {"params": 0, "bytes": 0.0, "flops": 0.0, "act_bytes": 0.0, "count": 0}
            for key in EXCEL_KEYS}
        for mod, stats in all_results:
            s = excel_summary[_MODULE_TO_EXCEL[stats.module_type]]
            s["params"] += stats.num_params
            s["bytes"] += stats.num_bytes
            s["flops"] += stats.num_flops
            s["act_bytes"] += stats.num_activations_bytes
            s["count"] += 1

        total_params = int(sum(v["params"] for v in excel_summary.values()))
        total_bytes = float(sum(v["bytes"] for v in excel_summary.values()))
        total_flops = float(sum(v["flops"] for v in excel_summary.values()))
        total_act = float(sum(v["act_bytes"] for v in excel_summary.values()))

        parameters: list[GroupedParamBreakdown] = [
            p for _, stats in all_results for p in stats.param_breakdown]
        memory = training_memory(parameters, self.config["training"],
                                 self.parallelism, self.bytes)

        layer_reports = []
        top_count = len(self.top_modules)
        idx = top_count
        for spec, modules in zip(self.layer_specs, self.layers):
            layer_stats = all_results[idx: idx + len(modules)]
            idx += len(modules)
            layer_reports.append({
                "layer_id": spec.layer_id,
                "attention_type": spec.attn_type,
                "ffn_type": spec.ffn_type,
                "use_indexer": False,
                "params": sum(s.num_params for _, s in layer_stats),
                "bytes": sum(s.num_bytes for _, s in layer_stats),
                "flops": sum(s.num_flops for _, s in layer_stats),
                "act_bytes": sum(s.num_activations_bytes for _, s in layer_stats),
                "modules": [{"type": s.module_type, "name": m.name,
                             "params": s.num_params, "bytes": s.num_bytes,
                             "flops": s.num_flops, "act_bytes": s.num_activations_bytes}
                            for m, s in layer_stats],
            })

        gbs = self.parallelism["gbs"]
        mbs = self.parallelism["mbs"]
        return {
            "model_family": self.model_family,
            "config": {
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "vocab_size": self.vocab_size,
                "max_seq_len": self.seq_len,
                "dtype": self.dtype.value,
                "variant": self.config.get("variant", "deep-v2"),
            },
            "source": self.config.get("source", {}),
            "model_params": {
                "total_params": total_params,
                "embedding_params": self.embedding.params_count(),
                "total_weight_bytes": total_bytes,
                "active_params_per_token": self._active_params(),
            },
            "compute": {
                "total_flops_per_sample": total_flops,
                "total_flops_at_gbs": total_flops * (gbs / mbs) if mbs else 0.0,
                "flops_per_token": total_flops / (B * S) if B * S > 0 else 0.0,
                "batch_size": B,
                "seq_len": S,
                "mbs": mbs,
                "gbs": gbs,
                "scope": "GEMM main terms per the workbook R-column (2*M*N*Q); "
                         "attention QK/PV on s^2; no softmax/conv-state/elementwise counts.",
            },
            "memory": {
                "weight_memory_bytes": memory["weights"],
                "activation_memory_bytes": total_act,
                "activation_estimate_kind": "Sum of named forward tensors per the workbook "
                                            "N-column (x2B; logits x4B); not measured peak.",
                "kv_cache_bytes": self.estimate_kv_cache(S),
                "cache_scope": "MLA latent KV per MLA layer (kv_lora + rope dims, bf16); "
                               "KDA conv states are not modeled in the workbook.",
                "training_total_bytes": memory["total"],
                "training_breakdown": memory,
                "per_rank_static_bytes": memory["per_rank"]["total"],
                "per_rank": memory["per_rank"],
            },
            "per_module_summary": per_module_summary,
            "per_module_summary_excel": excel_summary,
            "layers": layer_reports,
            "notes": [
                "10T Kimi is a planning extension derived from the workbook "
                "'10T Kimi 显存建模.xlsx' (sheets 10T-深度-v2 / 10T-宽度-v2); "
                "no official HF config is pinned.",
                "Param totals reproduce the workbook I-column formula chain; the "
                "cached J55 on 深度-v2 predates the v2 MLA g_proj 补档 and is stale.",
                "Optimizer states: AdamW 12 B/param (embedding + LM head), "
                "Muon 8 B/param (everything else); gradients 2 B; master weights 0.",
                "Per-rank static memory: dense modules /TP, routed experts /EP, "
                "optimizer states /total cards (rows G63/G68/K66).",
                "Dense layer (layer 0) carries no attention in 深度-v2 "
                "(mla+kda == moe_layers) and an MLA in 宽度-v2 "
                "(mla+kda == num_hidden_layers), mirroring the workbook counts.",
                "All displayed GiB use 2^30 bytes. JSON carries raw bytes.",
            ],
        }

    def _active_params(self) -> int:
        """Logical params touched per token (MoE sparse activation).

        Not defined in the workbook; follows the Hy4 convention: full
        embedding and LM head, all dense modules per layer, and top_k
        routed experts per MoE layer.
        """
        a = self.arch
        per_expert = (a["latent_moe_size"] * a["moe_intermediate_size"] * 2
                      + a["moe_intermediate_size"] * a["latent_moe_size"])
        active = self.embedding.params_count() + self.output.params_count() \
            + self.pre_norm.params_count() + self.post_norm.params_count() \
            + self.final_norm.params_count()
        for spec, modules in zip(self.layer_specs, self.layers):
            for mod in modules:
                if isinstance(mod, LatentMoE):
                    active += a["top_k"] * per_expert
                    for p in mod.compute_params():
                        if p.shard == "tp":
                            active += p.num_elements
                else:
                    active += mod.params_count()
        return active

    # -----------------------------------------------------------------

    def estimate_kv_cache(self, seq_len: int) -> float:
        """MLA latent KV per MLA layer: kv_lora_rank + qk_rope_head_dim
        elements per position, bf16 (workbook N9/N8-style latent cache).
        KDA SSM conv states are not counted (not in the workbook)."""
        a = self.arch
        per_token = (a["kv_lora_rank"] + a["qk_rope_head_dim"]) * self.bytes["activation"]
        return a["mla_layers"] * seq_len * per_token
