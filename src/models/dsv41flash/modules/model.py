"""DSV4.1-Flash model assembly using the shared BaseModule/ModuleStats API."""
from __future__ import annotations

from collections import Counter

from ....core.base_module import BaseModule, ModuleStats
from .accounting import training_memory
from .attention import Attention
from .embedding import Embedding
from .output import OutputLayer
from .moe import MoE
from .mhc import mHC
from .engram import Engram
from .dspark import DSparkInputProjection, DSparkHeads, DSparkLMHead
from .configuration import normalize_config


class DSV41FlashModel:
    """Container that assembles modules and aggregates their independent analyses."""

    model_family = "DSV41Flash"

    def __init__(self, config: dict):
        self.config = normalize_config(config)
        self.arch = self.config["architecture"]
        self.embedding = Embedding("WordEmbedding", self.config)
        self.output = OutputLayer("Output Layer", self.config)
        self.layers = self._build_layers()
        self.draft_layers = self._build_draft_layers() if self.config["include_dspark"] else []
        self.draft_extras = [
            DSparkInputProjection("DSpark input projection", self.config),
            DSparkHeads("DSpark Markov and confidence heads", self.config),
            DSparkLMHead("DSpark shared LM head execution", self.config),
        ] if self.config["include_dspark"] else []

    def _build_layers(self) -> list[list[BaseModule]]:
        layers = []
        for layer_id in range(self.arch["n_layers"]):
            modules = [
                Attention(f"layers.{layer_id}.attention", self.config, layer_id=layer_id),
                MoE(f"MoE layer {layer_id}", self.config, layer_id=layer_id),
                mHC(f"mHC + RMSNorm layer {layer_id}", self.config, layer_id=layer_id),
            ]
            if self.config["include_engram"] and layer_id in self.arch.get("engram_layer_ids", []):
                modules.append(Engram(f"Engram layer {layer_id}", self.config, layer_id=layer_id))
            layers.append(modules)
        return layers

    def _build_draft_layers(self) -> list[list[BaseModule]]:
        layers = []
        for stage in range(self.arch["n_mtp_layers"]):
            layer_id = self.arch["n_layers"] + stage
            layers.append([
                Attention(f"mtp.{stage}.attention", self.config, layer_id=layer_id, dspark=True),
                MoE(f"DSpark MoE layer {layer_id}", self.config, layer_id=layer_id, dspark=True),
                mHC(f"mHC + RMSNorm layer {layer_id}", self.config, layer_id=layer_id, dspark=True),
            ])
        return layers

    @staticmethod
    def _module_report(module: BaseModule, stats: ModuleStats) -> dict:
        return {
            "name": module.name, "type": stats.module_type,
            "params": stats.num_params, "bytes": stats.num_bytes,
            "flops": stats.num_flops, "act_bytes": stats.num_activations_bytes,
            "cache_bytes": stats.extra["cache_bytes"], "details": stats.extra["details"],
            "parameter_breakdown": [
                {"name": p.name, "shape": list(p.shape), "params": p.num_elements,
                 "group": p.group, "bytes": p.num_bytes}
                for p in stats.param_breakdown],
        }

    def analyze(self, seq_len: int | None = None, batch_size: int = 1) -> dict:
        S = self.config["max_seq_len"] if seq_len is None else seq_len
        B = batch_size
        for key, value in (("seq_len", S), ("batch_size", B)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{key} must be a positive integer")
        a, cfg = self.arch, self.config
        results = [(self.embedding, self.embedding.analyze(S, B))]
        layer_reports = []
        for layer_id, modules in enumerate(self.layers):
            analyzed = [(module, module.analyze(S, B)) for module in modules]
            results.extend(analyzed)
            rows = [self._module_report(module, stats) for module, stats in analyzed]
            layer_reports.append({
                "layer_id": layer_id, "attention_type": modules[0].module_type,
                "ffn_type": "MoE", "use_indexer": layer_id in a["index_source_layers"],
                **{key: sum(row[key] for row in rows) for key in ("params", "bytes", "flops", "act_bytes")},
                "modules": rows,
            })
        results.append((self.output, self.output.analyze(S, B)))
        backbone_results = list(results)
        auxiliary = {"enabled": cfg["include_dspark"], "params": 0, "modules": [],
                     "draft_forward_flops": 0.0, "draft_activation_bytes": 0.0}
        if cfg["include_dspark"]:
            draft = []
            T = a["dspark_block_size"]
            for modules in self.draft_layers:
                # Attention consumes a committed prefix; other blocks consume draft positions.
                draft.extend((module, module.analyze(S if index == 0 else T, B))
                             for index, module in enumerate(modules))
            draft.extend((module, module.analyze(T, B)) for module in self.draft_extras)
            auxiliary.update({
                "modules": [self._module_report(module, stats) for module, stats in draft],
                "params": sum(stats.num_params for _, stats in draft),
                "draft_forward_flops": sum(stats.num_flops for _, stats in draft),
                "draft_activation_bytes": sum(stats.num_activations_bytes for _, stats in draft),
                "prefix_tokens": S, "draft_tokens": T,
                "note": "One draft pass at an already-filled prefix, separate from backbone teacher-forced forward. "
                        "Main embedding/head weights are shared; auxiliary Markov weights are distinct.",
            })
            # Resident auxiliary tensors count once; their draft execution stays separate.
            for module, stats in draft:
                resident = ModuleStats(
                    module_type="DSpark", num_params=stats.num_params, num_bytes=stats.num_bytes,
                    param_breakdown=stats.param_breakdown,
                    extra={"cache_bytes": stats.extra["cache_bytes"],
                           "details": {"compute_scope": "auxiliary", "original_type": stats.module_type}},
                )
                results.append((module, resident))
        parameters = [p for _, stats in results for p in stats.param_breakdown]
        names = [p.name for p in parameters]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate parameter identity in Flash inventory")
        memory = training_memory(parameters, cfg["training"])
        module_reports = [self._module_report(module, stats) for module, stats in results]
        summary = {}
        for row in module_reports:
            item = summary.setdefault(row["type"], {"count": 0, "params": 0, "bytes": 0.0,
                                                     "flops": 0.0, "act_bytes": 0.0, "cache_bytes": 0.0})
            item["count"] += 1
            for key in ("params", "bytes", "flops", "act_bytes", "cache_bytes"):
                item[key] += row[key]
        active = 0
        for module, stats in backbone_results:
            for p in stats.param_breakdown:
                if module is self.embedding:
                    active += a["dim"]
                elif p.group == "engram":
                    active += (a["engram_max_ngram_size"] - 1) * a["engram_n_heads"] * a["engram_head_dim"]
                elif p.group == "expert":
                    active += p.num_elements // a["n_routed_experts"] * a["n_activated_experts"]
                else:
                    active += p.num_elements
        total_flops = sum(stats.num_flops for _, stats in backbone_results)
        return {
            "model_family": self.model_family,
            "config": {"hidden_size": a["dim"], "num_layers": a["n_layers"],
                       "vocab_size": a["vocab_size"], "max_seq_len": cfg["max_seq_len"],
                       "dtype": cfg["dtype"], "include_engram": cfg["include_engram"],
                       "include_dspark": cfg["include_dspark"], "training": cfg["training"]},
            "source": cfg.get("source", {"architecture": "inline"}),
            "scope": {"path": "text", "vision_tower": False, "quantization_metadata": False,
                      "layer_counts": dict(Counter(r["attention_type"] for r in layer_reports)),
                      "auxiliary_compute_separate": True},
            "model_params": {
                "total_params": sum(p.num_elements for p in parameters),
                "backbone_params": sum(stats.num_params for _, stats in backbone_results),
                "embedding_params": results[0][1].num_params,
                "total_weight_bytes": memory["weights"], "active_params_per_token": active,
                "active_params_note": "Logical touched elements in backbone: one input embedding row, "
                                      "Top-K routed experts, 24 Engram rows/module, full LM head. Excludes DSpark.",
            },
            "compute": {"seq_len": S, "batch_size": B, "total_flops_per_sample": total_flops,
                        "total_forward_flops": total_flops, "flops_per_token": total_flops / (B * S),
                        "scope": "Backbone teacher-forced forward for B*S tokens; major GEMMs and attention. "
                                 "No backward, kernel timing, or small elementwise/quantization operators."},
            "memory": {"weight_memory_bytes": memory["weights"],
                       "activation_memory_bytes": sum(stats.num_activations_bytes for _, stats in backbone_results),
                       "activation_estimate_kind": "Unsharded sum of named forward tensors; NOT measured peak or exact backward-saved tensors.",
                       "kv_cache_bytes": sum(stats.extra["cache_bytes"] for _, stats in results),
                       "cache_scope": "Logical populated inference caches, not training activations; owner storage counted once.",
                       "training_total_bytes": memory["total"], "training_breakdown": memory},
            "per_module_summary": summary, "modules": module_reports,
            "layers": layer_reports, "auxiliary": auxiliary,
            "notes": [
                "All displayed GB use 1e9 bytes. JSON carries raw bytes.",
                "Model dimensions come from the pinned source config; precision, placement and sharding are analysis assumptions.",
                "Training states use independent dtypes per parameter group. Engram momentum counts one state, not Adam's two.",
                "Source inference code does not establish an official training optimizer/placement recipe; no runtime training is simulated.",
                "Engram CPU placement moves the table and its states only; projection/gate and lookup tensors remain in GPU computations.",
                "Per-device static assumes balanced shards; activation volume and inference caches are separate, not added into a claimed training peak.",
                "Vision tower/aligner and quantization scales are excluded; source-created router bias_vl is retained.",
            ],
        }
