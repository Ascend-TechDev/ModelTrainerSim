"""GLM-5.2 top-level model.

Assembles a stack of 78 transformer layers and surfaces the per-module
parameter/memory/FLOPs breakdown defined in the Excel reference sheet
``GLM-5.2-743B`` (file: GLM分析.xlsx).

Layer composition (per config ``mlp_layer_types`` and ``indexer_types``):

  * First 3 layers are **dense** FFN layers (no MoE, no Indexer).
  * Remaining 75 layers are **sparse** MoE layers.
  * Among the 75 MoE layers, ``num_indexer_layers`` (default 21) have a
    shared Indexer module (``indexer_types == "shared"``).
  * Every layer has: input_layernorm + MLA + (Indexer if shared) + DSA
    output + (DenseFFN | SharedExperts + RoutedExperts + MoERouter)
    + post-attention-layernorm.

The model's ``analyze()`` returns a report that mirrors the Excel sheet:

  WordEmbedding (Excel rows  2-3)
  InputLayerNorm (Excel rows  6-7)
  MLA            (Excel rows 10-16)
  Indexer        (Excel rows 19-23)
  DSA            (Excel rows 26-27)
  DenseFFN        (Excel rows 30-33)
  SharedExperts   (Excel rows 36-39)
  RoutedExperts   (Excel rows 42-46)
  MoERouter       (Excel rows 47-48)
  OutputLayerNorm  (Excel rows 51-52)
  LMHead          (Excel rows 55-56)
  Total params (Excel L58/L59)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ....core.base_module import BaseModule, ModuleParamBreakdown
from ....core.dtype import DType, get_dtype_size
from .mla import MLA
from .indexer import Indexer
from .dsa import DSA
from .dense_ffn import DenseFFN
from .shared_experts import SharedExperts
from .routed_experts import RoutedExperts
from .moe_router import MoERouter
from .embedding import Embedding, InputLayerNorm, OutputLayerNorm, LMHead


# ---------------------------------------------------------------------------
# Layer specification
# ---------------------------------------------------------------------------

@dataclass
class LayerSpec:
    """Description of a single GLM transformer layer."""
    layer_id: int
    is_dense: bool                          # True for first 3 layers
    use_indexer: bool = False                 # True for shared-indexer MoE layers
    attention_config: dict[str, Any] = field(default_factory=dict)
    indexer_config: dict[str, Any] = field(default_factory=dict)
    dense_ffn_config: dict[str, Any] = field(default_factory=dict)
    shared_experts_config: dict[str, Any] = field(default_factory=dict)
    routed_experts_config: dict[str, Any] = field(default_factory=dict)
    moe_router_config: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# GLM-5.2 Model
# ---------------------------------------------------------------------------

class GLMModel:
    """Top-level GLM-5.2 model builder and analyzer."""

    model_family = "GLM-5.2"

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.hidden_size: int = config["hidden_size"]
        self.num_layers: int = config["num_layers"]
        self.vocab_size: int = config["vocab_size"]
        self.seq_len: int = config.get("max_seq_len", 4096)
        self.tie_word_embeddings: bool = config.get("tie_word_embeddings", False)
        self.dtype: DType = DType(config.get("dtype", "bf16"))

        # MoE / dense split
        self.first_k_dense: int = config.get("first_k_dense_replace", 3)
        self.num_moe_layers: int = self.num_layers - self.first_k_dense

        # Indexer scheduling
        indexer_types = config.get("indexer_types", [])
        # Count shared-indexer layers (each shared Indexer covers a small
        # cluster of layers; GLM-5.2 uses 21 shared indexers across 75 MoE
        # layers per Excel D23).
        self.num_indexer_layers: int = sum(
            1 for t in indexer_types if t == "shared"
        ) if indexer_types else config.get("num_indexer_layers", 21)

        # Top-level modules
        self.embedding = Embedding("embedding", {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "dtype": self.dtype.value,
            "tie_word_embeddings": self.tie_word_embeddings,
        })
        self.input_layernorm = InputLayerNorm("input_layernorm", {
            "hidden_size": self.hidden_size,
            "dtype": self.dtype.value,
        })
        self.output_layernorm = OutputLayerNorm("output_layernorm", {
            "hidden_size": self.hidden_size,
            "dtype": self.dtype.value,
        })
        self.lm_head = LMHead("lm_head", {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "dtype": self.dtype.value,
        })

        # Build layer specs and module instances
        self.layer_specs: list[LayerSpec] = self._build_layer_specs()
        self.layers: list[list[BaseModule]] = self._build_layers()

    # -----------------------------------------------------------------
    # Layer construction
    # -----------------------------------------------------------------

    def _build_layer_specs(self) -> list[LayerSpec]:
        specs: list[LayerSpec] = []
        indexer_types = self.config.get("indexer_types", [])

        # Attention config (MLA plumbing)
        attn_base = {
            "hidden_size": self.hidden_size,
            "dtype": self.dtype.value,
            "q_lora_rank": self.config.get("q_lora_rank", 0),
            "kv_lora_rank": self.config.get("kv_lora_rank", 0),
            "num_heads": self.config.get("num_attention_heads", 1),
            "num_key_value_heads": self.config.get("num_key_value_heads", 1),
            "qk_head_dim": self.config.get("qk_head_dim", 0),
            "qk_nope_head_dim": self.config.get("qk_nope_head_dim", 0),
            "qk_rope_head_dim": self.config.get("qk_rope_head_dim", 0),
            "v_head_dim": self.config.get("v_head_dim", 0),
        }

        # Indexer config
        idx_base = {
            "hidden_size": self.hidden_size,
            "dtype": self.dtype.value,
            "q_lora_rank": self.config.get("q_lora_rank", 0),
            "index_head_dim": self.config.get("index_head_dim", 128),
            "index_n_heads": self.config.get("index_n_heads", 32),
            "index_topk": self.config.get("index_topk", 2048),
        }

        # Dense FFN config
        dense_base = {
            "hidden_size": self.hidden_size,
            "dtype": self.dtype.value,
            "intermediate_size": self.config.get("intermediate_size", 0),
        }

        # Shared experts config
        shared_base = {
            "hidden_size": self.hidden_size,
            "dtype": self.dtype.value,
            "moe_intermediate_size": self.config.get("moe_intermediate_size", 0),
            "num_shared_experts": self.config.get("n_shared_experts", 1),
        }

        # Routed experts config
        routed_base = {
            "hidden_size": self.hidden_size,
            "dtype": self.dtype.value,
            "moe_intermediate_size": self.config.get("moe_intermediate_size", 0),
            "n_routed_experts": self.config.get("n_routed_experts", 1),
            "num_experts_per_tok": self.config.get("num_experts_per_tok", 1),
        }

        # MoE router config
        router_base = {
            "hidden_size": self.hidden_size,
            "dtype": self.dtype.value,
            "n_routed_experts": self.config.get("n_routed_experts", 1),
            "num_experts_per_tok": self.config.get("num_experts_per_tok", 1),
            "gate_dtype": self.config.get("moe_router_dtype", "float32"),
        }

        for i in range(self.num_layers):
            is_dense = (i < self.first_k_dense)
            itype = indexer_types[i] if i < len(indexer_types) else "full"
            use_indexer = (not is_dense) and (itype == "shared")

            specs.append(LayerSpec(
                layer_id=i,
                is_dense=is_dense,
                use_indexer=use_indexer,
                attention_config=attn_base,
                indexer_config=idx_base,
                dense_ffn_config=dense_base,
                shared_experts_config=shared_base,
                routed_experts_config=routed_base,
                moe_router_config=router_base,
            ))
        return specs

    def _build_layers(self) -> list[list[BaseModule]]:
        layers: list[list[BaseModule]] = []
        for spec in self.layer_specs:
            modules: list[BaseModule] = []
            modules.append(MLA(f"layer{spec.layer_id}_MLA",
                                spec.attention_config))
            if spec.use_indexer:
                modules.append(Indexer(
                    f"layer{spec.layer_id}_Indexer",
                    spec.indexer_config))
            modules.append(DSA(f"layer{spec.layer_id}_DSA",
                                spec.attention_config))
            if spec.is_dense:
                modules.append(DenseFFN(
                    f"layer{spec.layer_id}_DenseFFN",
                    spec.dense_ffn_config))
            else:
                modules.append(SharedExperts(
                    f"layer{spec.layer_id}_SharedExperts",
                    spec.shared_experts_config))
                modules.append(RoutedExperts(
                    f"layer{spec.layer_id}_RoutedExperts",
                    spec.routed_experts_config))
                modules.append(MoERouter(
                    f"layer{spec.layer_id}_MoERouter",
                    spec.moe_router_config))
            layers.append(modules)
        return layers

    # -----------------------------------------------------------------
    # Analysis
    # -----------------------------------------------------------------

    def analyze(self, seq_len: int | None = None,
                batch_size: int = 1) -> dict[str, Any]:
        """Run full analysis and return a structured report.

        Aligns with Excel sheet 'GLM-5.2-743B' rows 2-59.
        """
        S = seq_len or self.seq_len
        B = batch_size

        from ....core.base_module import ModuleStats

        total_params = 0
        total_bytes = 0.0
        total_flops = 0.0
        total_act = 0.0

        # Top-level modules: Embedding, input_layernorm, output_layernorm, lm_head
        top_modules = [self.embedding, self.input_layernorm,
                       self.output_layernorm, self.lm_head]
        top_stats: list[tuple[str, ModuleStats]] = []
        for mod in top_modules:
            stats = mod.analyze(S, B)
            top_stats.append((mod.module_type, stats))
            total_params += stats.num_params
            total_bytes += stats.num_bytes
            total_flops += stats.num_flops
            # Excel U2 (embedding) is counted separately at the top level.
            if mod.module_type != "Embedding":
                total_act += stats.num_activations_bytes

        # Per-layer analysis
        layer_reports: list[dict[str, Any]] = []
        # Module-instance summary (every BaseModule instance, no scaling).
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
                "is_dense": spec.is_dense,
                "use_indexer": spec.use_indexer,
                "params": layer_params,
                "bytes": layer_bytes,
                "flops": layer_flops,
                "act_bytes": layer_act,
                "modules": module_details,
            })

        # ------------------------------------------------------------------
        # Excel-aligned per-module summary
        #
        # Mirrors the GLM-5.2-743B Excel sheet's grouped parameter/
        # memory rows (columns L-N), where each row aggregates ALL
        # instances of that module type with the appropriate count
        # multiplier (H column):
        #
        #   H2  = 1            (word embedding)
        #   H7  = num_layers    (input_layernorm, 1 per layer)
        #   H11 = num_layers    (MLA)
        #   H20 = 21            (Indexer — shared indexers)
        #   H27 = num_layers    (DSA output proj)
        #   H31 = 3             (Dense FFN — first 3 layers)
        #   H37 = num_moe_layers * n_shared   (Shared experts)
        #   H43 = num_moe_layers * n_routed    (Routed experts params)
        #   H44 = num_moe_layers * topk        (Routed experts activated)
        #   H48 = num_moe_layers  (MoE Router)
        #   H52 = 1             (output_layernorm)
        #   H56 = 1             (LM Head)
        #
        # Notes:
        #   * RoutedExperts module returns ONE expert's params; the
        #     summary scales by n_routed_experts * num_moe_layers (Excel H43).
        #   * Activated routed-expert params use H44 (topk * num_moe_layers).
        # ------------------------------------------------------------------
        excel_summary: dict[str, dict[str, float]] = {
            "wordembedding": {"params": 0, "bytes": 0.0, "flops": 0.0,
                              "act_bytes": 0.0, "count": 0},
            "inputlayernorm": {"params": 0, "bytes": 0.0, "flops": 0.0,
                                "act_bytes": 0.0, "count": 0},
            "mla": {"params": 0, "bytes": 0.0, "flops": 0.0,
                     "act_bytes": 0.0, "count": 0},
            "indexer": {"params": 0, "bytes": 0.0, "flops": 0.0,
                             "act_bytes": 0.0, "count": 0},
            "dsa": {"params": 0, "bytes": 0.0, "flops": 0.0,
                     "act_bytes": 0.0, "count": 0},
            "denseffn": {"params": 0, "bytes": 0.0, "flops": 0.0,
                              "act_bytes": 0.0, "count": 0},
            "sharedexperts": {"params": 0, "bytes": 0.0, "flops": 0.0,
                                   "act_bytes": 0.0, "count": 0},
            "routedexperts": {"params": 0, "bytes": 0.0, "flops": 0.0,
                                    "act_bytes": 0.0, "count": 0},
            "moerouter": {"params": 0, "bytes": 0.0, "flops": 0.0,
                               "act_bytes": 0.0, "count": 0},
            "outputlayernorm": {"params": 0, "bytes": 0.0, "flops": 0.0,
                                  "act_bytes": 0.0, "count": 0},
            "lmhead": {"params": 0, "bytes": 0.0, "flops": 0.0,
                            "act_bytes": 0.0, "count": 0},
        }

        # WordEmbedding (1 instance)
        _emb_stats = self.embedding.analyze(S, B)
        excel_summary["wordembedding"]["params"] += _emb_stats.num_params
        excel_summary["wordembedding"]["bytes"] += _emb_stats.num_bytes
        excel_summary["wordembedding"]["flops"] += _emb_stats.num_flops
        excel_summary["wordembedding"]["act_bytes"] += _emb_stats.num_activations_bytes
        excel_summary["wordembedding"]["count"] += 1

        # input_layernorm (num_layers instances, but Excel layers input_layernorm
        # into MLA's per-layer cost — H7 = D2 = 78).
        _iln_stats = self.input_layernorm.analyze(S, B)
        iln_count = self.num_layers
        excel_summary["inputlayernorm"]["params"] += _iln_stats.num_params * iln_count
        excel_summary["inputlayernorm"]["bytes"] += _iln_stats.num_bytes * iln_count
        excel_summary["inputlayernorm"]["flops"] += _iln_stats.num_flops * iln_count
        excel_summary["inputlayernorm"]["act_bytes"] += _iln_stats.num_activations_bytes * iln_count
        excel_summary["inputlayernorm"]["count"] += iln_count

        # output_layernorm (1 instance)
        _oln_stats = self.output_layernorm.analyze(S, B)
        excel_summary["outputlayernorm"]["params"] += _oln_stats.num_params
        excel_summary["outputlayernorm"]["bytes"] += _oln_stats.num_bytes
        excel_summary["outputlayernorm"]["flops"] += _oln_stats.num_flops
        excel_summary["outputlayernorm"]["act_bytes"] += _oln_stats.num_activations_bytes
        excel_summary["outputlayernorm"]["count"] += 1

        # LMHead (1 instance)
        _lm_stats = self.lm_head.analyze(S, B)
        excel_summary["lmhead"]["params"] += _lm_stats.num_params
        excel_summary["lmhead"]["bytes"] += _lm_stats.num_bytes
        excel_summary["lmhead"]["flops"] += _lm_stats.num_flops
        excel_summary["lmhead"]["act_bytes"] += _lm_stats.num_activations_bytes
        excel_summary["lmhead"]["count"] += 1

        # Per-layer modules — scaled by counts per Excel H column.
        # MLA: 1 per layer (H11 = D2 = 78)
        # DSA: 1 per layer (H27 = D2 = 78)
        # Indexer: 1 per shared-indexer layer (H20 = D23 = 21)
        # DenseFFN: 1 per dense layer (H31 = D3 = 3)
        # SharedExperts: 1 per moe-layer × n_shared (H37 = D4 * D20)
        #                 params scaled by n_shared; per-layer FFN one SharedExperts module
        #                 returns ALL n_shared experts' params in compute_flops.
        # RoutedExperts: 1 per moe-layer; params scaled by n_routed per moe-layer.
        #                  H43 = D4 * D19 = moe_layers * n_routed
        # MoERouter: 1 per moe-layer (H48 = D4 = 75)
        _mla_stats = MLA("tmpl", self.layer_specs[0].attention_config).analyze(S, B)
        _dsa_stats = DSA("tmpd", self.layer_specs[0].attention_config).analyze(S, B)
        mla_count = self.num_layers
        dsa_count = self.num_layers
        excel_summary["mla"]["params"] += _mla_stats.num_params * mla_count
        excel_summary["mla"]["bytes"] += _mla_stats.num_bytes * mla_count
        excel_summary["mla"]["flops"] += _mla_stats.num_flops * mla_count
        excel_summary["mla"]["act_bytes"] += _mla_stats.num_activations_bytes * mla_count
        excel_summary["mla"]["count"] += mla_count
        excel_summary["dsa"]["params"] += _dsa_stats.num_params * dsa_count
        excel_summary["dsa"]["bytes"] += _dsa_stats.num_bytes * dsa_count
        excel_summary["dsa"]["flops"] += _dsa_stats.num_flops * dsa_count
        excel_summary["dsa"]["act_bytes"] += _dsa_stats.num_activations_bytes * dsa_count
        excel_summary["dsa"]["count"] += dsa_count

        # Indexer (21 shared instances)
        _idx_stats = Indexer("tmpi", self.layer_specs[0].indexer_config).analyze(S, B)
        idx_count = self.num_indexer_layers
        excel_summary["indexer"]["params"] += _idx_stats.num_params * idx_count
        excel_summary["indexer"]["bytes"] += _idx_stats.num_bytes * idx_count
        excel_summary["indexer"]["flops"] += _idx_stats.num_flops * idx_count
        excel_summary["indexer"]["act_bytes"] += _idx_stats.num_activations_bytes * idx_count
        excel_summary["indexer"]["count"] += idx_count

        # DenseFFN (3 instances)
        _dffn_stats = DenseFFN("tmpf", self.layer_specs[0].dense_ffn_config).analyze(S, B)
        dffn_count = self.first_k_dense
        excel_summary["denseffn"]["params"] += _dffn_stats.num_params * dffn_count
        excel_summary["denseffn"]["bytes"] += _dffn_stats.num_bytes * dffn_count
        excel_summary["denseffn"]["flops"] += _dffn_stats.num_flops * dffn_count
        excel_summary["denseffn"]["act_bytes"] += _dffn_stats.num_activations_bytes * dffn_count
        excel_summary["denseffn"]["count"] += dffn_count

        # SharedExperts (per moe-layer × n_shared; one module returns one expert)
        _se_stats = SharedExperts("tmpse", self.layer_specs[self.first_k_dense].shared_experts_config).analyze(S, B)
        shared_count = self.num_moe_layers * self.config.get("n_shared_experts", 1)
        excel_summary["sharedexperts"]["params"] += _se_stats.num_params * shared_count
        excel_summary["sharedexperts"]["bytes"] += _se_stats.num_bytes * shared_count
        excel_summary["sharedexperts"]["flops"] += _se_stats.num_flops * self.num_moe_layers
        excel_summary["sharedexperts"]["act_bytes"] += _se_stats.num_activations_bytes * self.num_moe_layers
        excel_summary["sharedexperts"]["count"] += shared_count

        # RoutedExperts (per moe-layer × n_routed; one module returns one expert)
        _re_stats = RoutedExperts("tmpre", self.layer_specs[self.first_k_dense].routed_experts_config).analyze(S, B)
        routed_count = self.num_moe_layers * self.config.get("n_routed_experts", 1)
        excel_summary["routedexperts"]["params"] += _re_stats.num_params * routed_count
        excel_summary["routedexperts"]["bytes"] += _re_stats.num_bytes * routed_count
        excel_summary["routedexperts"]["flops"] += _re_stats.num_flops * self.num_moe_layers
        excel_summary["routedexperts"]["act_bytes"] += _re_stats.num_activations_bytes * self.num_moe_layers
        excel_summary["routedexperts"]["count"] += routed_count

        # MoERouter (per moe-layer)
        _mr_stats = MoERouter("tmpmr", self.layer_specs[self.first_k_dense].moe_router_config).analyze(S, B)
        mr_count = self.num_moe_layers
        excel_summary["moerouter"]["params"] += _mr_stats.num_params * mr_count
        excel_summary["moerouter"]["bytes"] += _mr_stats.num_bytes * mr_count
        excel_summary["moerouter"]["flops"] += _mr_stats.num_flops * mr_count
        excel_summary["moerouter"]["act_bytes"] += _mr_stats.num_activations_bytes * mr_count
        excel_summary["moerouter"]["count"] += mr_count

        # KV cache estimate
        kv_cache = self.estimate_kv_cache(S, B)

        # Excel-aligned totals: sum the per-module excel summary, which
        # applies the correct instance-count multipliers (H column).
        # The per-layer walk above counts one expert per RoutedExperts /
        # SharedExperts instance, so its total understates the real model;
        # the per-module summary is the source of truth.
        total_params = int(sum(v["params"] for v in excel_summary.values()))
        total_bytes = float(sum(v["bytes"] for v in excel_summary.values()))
        total_flops = float(sum(v["flops"] for v in excel_summary.values()))
        total_act = float(sum(v["act_bytes"] for v in excel_summary.values()))

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
                "embedding_params": self.embedding.params_count(),
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
            "per_module_summary_excel": excel_summary,
            "layers": layer_reports,
        }

    def _active_params(self, total_params: int) -> int:
        """Estimate activated params per token.

        Excel L59 (41.25B) intermittent per-token activation includes:
          - WordEmbedding        (×1)
          - input_layernorm      (×1, NOT all 78 — only the active layer)
          - MLA                  (×1 per token passing through 78 layers)
          - DSA                  (×1 per layer × 78)
          - Indexer              (×1 per active shared-indexer layer)
          - DenseFFN              (×3 dense layers)
          - SharedExperts         (×1 per MoE-layer, all n_shared always on)
          - RoutedExperts         (×num_experts_per_tok per MoE-layer)
          - MoERouter             (×1 per MoE-layer)
          - LMHead                (×1)
          - output_layernorm      (×1)
        """
        spec_dense = self.layer_specs[0]
        spec_moe = self.layer_specs[self.first_k_dense]
        mla_one = sum(p.num_elements for p in MLA("a", spec_dense.attention_config).compute_params())
        dsa_one = sum(p.num_elements for p in DSA("d", spec_dense.attention_config).compute_params())
        idx_one = sum(p.num_elements for p in Indexer("i", spec_moe.indexer_config).compute_params())
        dffn_one = sum(p.num_elements for p in DenseFFN("f", spec_dense.dense_ffn_config).compute_params())
        se_one = sum(p.num_elements for p in SharedExperts("s", spec_moe.shared_experts_config).compute_params())
        re_one = sum(p.num_elements for p in RoutedExperts("r", spec_moe.routed_experts_config).compute_params())
        mr_one = sum(p.num_elements for p in MoERouter("m", spec_moe.moe_router_config).compute_params())

        active = 0
        active += self.embedding.params_count()                          # word embedding
        active += self.input_layernorm.params_count()                     # input_layernorm
        active += int(mla_one * self.num_layers)                          # MLA ×78
        active += int(dsa_one * self.num_layers)                          # DSA ×78
        active += int(idx_one * self.num_indexer_layers)                  # Indexer ×21
        active += int(dffn_one * self.first_k_dense)                      # DenseFFN ×3
        active += int(se_one * self.num_moe_layers * self.config.get("n_shared_experts", 1))  # Shared ×75
        active += int(re_one * self.num_moe_layers * self.config.get("num_experts_per_tok", 1))  # Routed topk ×75
        active += int(mr_one * self.num_moe_layers)                       # MoERouter ×75
        active += self.output_layernorm.params_count()                    # output_layernorm
        active += self.lm_head.params_count()                             # LMHead
        return active

    # -----------------------------------------------------------------
    # KV cache estimate
    # -----------------------------------------------------------------

    def estimate_kv_cache(self, seq_len: int, batch_size: int = 1) -> float:
        """Estimate KV cache memory for the given sequence length.

        GLM-5.2 uses MLA-style compressed KV cache: only the low-rank
        KV (kv_lora_rank + qk_rope_head_dim) per token is stored per
        layer (NOT the full per-head K/V).
        """
        B, S = batch_size, seq_len
        elem = get_dtype_size(self.dtype)
        kv_lora = self.config.get("kv_lora_rank", 0)
        d_rope = self.config.get("qk_rope_head_dim", 0)
        per_token = (kv_lora + d_rope) * elem
        return B * S * per_token * self.num_layers

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
          - Master weights (fp32, = 2× weight bytes for bf16)
          - Optimizer states (adamw: m + v in fp32 = 8 bytes/param)
        """
        bytes_per_param = 2.0
        param_element_bytes = weight_bytes / bytes_per_param

        grad_bytes = weight_bytes
        master_weight_bytes = weight_bytes * 2

        if optimizer == "adamw":
            opt_bytes = param_element_bytes * 8
        elif optimizer == "adamw_8bit":
            opt_bytes = param_element_bytes * 2
        else:
            opt_bytes = param_element_bytes * 8

        total = (weight_bytes + grad_bytes + master_weight_bytes + opt_bytes)

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