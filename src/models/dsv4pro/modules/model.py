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
from .swa import SWA
from .indexer import Indexer
from .moe import MoE


# ---------------------------------------------------------------------------
# Layer specification
# ---------------------------------------------------------------------------

@dataclass
class LayerSpec:
    """Description of a single transformer layer.

    A DSV4Pro transformer block (excluding the special MTP layer) has the
    following structure::

        mHC_pre -> [CSA | HCA] (+ Indexer if HCA) -> MoE -> mHC_post

    The two mHC sub-modules form the residual connection of the block; they
    are always present.  The attention type (CSA when ``compress_ratio == 4``,
    HCA when ``compress_ratio == 128``) is driven by the per-layer
    ``compress_ratios`` schedule.  The MTP layer (``compress_ratio == 0``)
    is special and contains only mHC_pre + an output head.
    """
    layer_id: int
    attention_type: str       # "CSA", "HCA", or "MTP"
    attention_config: dict[str, Any]
    use_indexer: bool = False
    indexer_config: dict[str, Any] | None = None
    ffn_type: str = "MoE"     # "MoE" or "Dense"
    ffn_config: dict[str, Any] = field(default_factory=dict)
    # mHC residual sub-blocks (one pre, one post) — present in every layer.
    mhc_pre_config: dict[str, Any] = field(default_factory=dict)
    mhc_post_config: dict[str, Any] = field(default_factory=dict)
    is_mtp: bool = False


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
    "SWA": SWA,
}


def _build_attention(layer_spec: LayerSpec) -> BaseModule | None:
    if layer_spec.attention_type == "MTP":
        return None
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


def _build_mhc(layer_id: int, position: str, cfg: dict[str, Any]) -> mHC:
    """Build a mHC residual sub-module (position = 'pre' or 'post')."""
    return mHC(f"layer{layer_id}_mHC_{position}", cfg)


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

        # ------------------------------------------------------------------
        # DSV4Pro layer scheduling (per user clarification):
        #   - Every block:  mHC_pre -> [CSA|HCA] (+Indexer) -> MoE -> mHC_post
        #   - CSA  : layers with compress_ratio == 4
        #   - HCA  : layers with compress_ratio == 128   (HCA also has an Indexer)
        #   - MTP  : the special last layer with compress_ratio == 0
        #   - mHC  : a residual compression sub-block, present (twice) in every layer.
        # ------------------------------------------------------------------
        compress_ratios = layer_config.get("compress_ratios", [])
        mtp_layers = layer_config.get("mtp_layers", 0)
        first_k_dense = layer_config.get("first_k_dense_replace", 0)

        # Shared sub-module configs (base — copied and augmented per-layer).
        attn_base = layer_config.get("attention", {})
        moe_config = layer_config.get("moe", {})
        dense_ffn_config = layer_config.get("dense_ffn", {})
        indexer_config = layer_config.get("indexer", {})
        mhc_cfg = layer_config.get("mhc", {})   # optional override for mHC inter dim

        for i in range(self.num_layers):
            ratio = (compress_ratios[i] if i < len(compress_ratios) else 128)

            # --- Determine attention type ---
            # The last layer (index = num_layers-1) is SWA (sliding-window
            # attention). It has NO Compressor and NO Indexer — just standard
            # MLA Q/KV/O projections.  For all other layers, the compress_ratio
            # decides the attention kind:
            #   ratio == 4   -> CSA   (uses Compressor-4 + Indexer)
            #   ratio == 128 -> HCA  (uses Compressor-128, no Indexer)
            #   ratio == 0   -> MTP  (special last MTP layer, separately handled)
            is_last_layer = (i == self.num_layers - 1)

            if is_last_layer:
                # The final transformer layer is SWA (sliding-window attention).
                # Its ratio entry may be 4 (per compress_ratios) but it does NOT
                # use a Compressor or Indexer.
                attn_type = "SWA"
                is_mtp = False
            elif ratio == 4:
                attn_type = "CSA"
                is_mtp = False
            elif ratio == 128:
                attn_type = "HCA"
                is_mtp = False
            elif ratio == 0:
                attn_type = "MTP"
                is_mtp = True
            else:
                # Any other ratio: treat as HCA with that block size.
                attn_type = "HCA"
                is_mtp = False

            # --- Attention config (only meaningful for non-MTP layers) ---
            attn_cfg = dict(attn_base)
            attn_cfg.setdefault("hidden_size", self.hidden_size)
            attn_cfg.setdefault("dtype", self.dtype.value)
            attn_cfg.setdefault("compress_ratio", ratio)

            # --- FFN config (MTP layer has no FFN; first_k_dense layers use Dense FFN) ---
            if is_mtp:
                ffn_type = "None"
                ffn_cfg: dict[str, Any] = {}
            elif i < first_k_dense:
                ffn_type = "Dense"
                ffn_cfg = dict(dense_ffn_config)
                ffn_cfg.setdefault("hidden_size", self.hidden_size)
                ffn_cfg.setdefault("dtype", self.dtype.value)
            else:
                ffn_type = "MoE"
                ffn_cfg = dict(moe_config)
                ffn_cfg.setdefault("hidden_size", self.hidden_size)
                ffn_cfg.setdefault("dtype", self.dtype.value)
                # Pass vocab_size so MoE's tid2eid gate mapping table can be
                # included (per Excel row 89: G89 = vocab * num_activated).
                ffn_cfg.setdefault("vocab_size", self.vocab_size)

            # --- Indexer: only on CSA layers (Compressor-4 layers, per Excel reference) ---
            use_indexer = (attn_type == "CSA" and indexer_config.get("enabled", True))
            idx_cfg = None
            if use_indexer:
                idx_cfg = dict(indexer_config)
                idx_cfg.pop("enabled", None)
                idx_cfg.setdefault("hidden_size", self.hidden_size)
                idx_cfg.setdefault("dtype", self.dtype.value)

            # --- mHC residual sub-blocks (one pre, one post per layer) ---
            mhc_pre_cfg = dict(mhc_cfg)
            mhc_pre_cfg.setdefault("hidden_size", self.hidden_size)
            mhc_pre_cfg.setdefault("dtype", self.dtype.value)
            mhc_post_cfg = dict(mhc_cfg)
            mhc_post_cfg.setdefault("hidden_size", self.hidden_size)
            mhc_post_cfg.setdefault("dtype", self.dtype.value)

            specs.append(LayerSpec(
                layer_id=i,
                attention_type=attn_type,
                attention_config=attn_cfg,
                use_indexer=use_indexer,
                indexer_config=idx_cfg,
                ffn_type=ffn_type,
                ffn_config=ffn_cfg,
                mhc_pre_config=mhc_pre_cfg,
                mhc_post_config=mhc_post_cfg,
                is_mtp=is_mtp,
            ))

        return specs

    def _build_layers(self) -> list[list[BaseModule]]:
        layers: list[list[BaseModule]] = []
        for spec in self.layer_specs:
            modules: list[BaseModule] = []

            # 1) mHC_pre  (residual compression at block start — every layer)
            modules.append(_build_mhc(spec.layer_id, "pre", spec.mhc_pre_config))

            # 2) Attention (CSA or HCA) — skipped for the MTP layer
            attn = _build_attention(spec)
            if attn is not None:
                modules.append(attn)

            # 3) Indexer — CSA layers only (per Excel reference)
            idx = _build_indexer(spec)
            if idx is not None:
                modules.append(idx)

            # 4) FFN (MoE or Dense) — skipped for the MTP layer
            if spec.ffn_type != "None":
                modules.append(_build_ffn(spec))

            # 5) mHC_post (residual compression at block end — every layer)
            modules.append(_build_mhc(spec.layer_id, "post", spec.mhc_post_config))

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
        # Excel does not count WordEmbedding activations separately.
        total_act += 0

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

        # ------------------------------------------------------------------
        # Excel-aligned per-module summary
        #
        # Mirrors the DSV4Pro Excel sheet "Per-Module Summary" rows 89-95
        # (columns O-R), which lists FLOPs contributions per module type:
        #   r89  mHC          count=61   Q89 = P7  (mHC sub-block per-layer)
        #   r90  Gate         count=61   Q90 = 2·B·S·H·n_act
        #   r91  SWA          count=1    Q91 = O18 (single SWA layer attn)
        #   r92  CSA          count=31   Q92 = O31 (per-layer CSA attn+Indexer)
        #   r93  HCA          count=29   Q93 = O45 (per-layer HCA attn)
        #   r94  MoE          count=61   Q94 = O83 (per-layer routed+shared)
        #   r95  TOTAL                  R95 = SUM(R89:R94)
        #
        # Notes:
        #   * Attention modules (CSA / HCA / SWA) already include their
        #     Compressor and Indexer sub-modules.
        #   * MoE FLOPs are split: routed+shared experts → "routedexpert"
        #     (matches Excel R94 = O83), gate routing → "gate" (matches R90).
        #     Excel does not have a "sharedexpert" FLOPs row — its FLOPs are
        #     inside O83 (P79/P80/P81) and thus reported under "routedexpert"
        #     to mirror Excel R94.
        #   * mHC residual blocks are included as a separate row (Excel R89).
        # ------------------------------------------------------------------
        excel_summary: dict[str, dict[str, float]] = {
            "wordembedding": {"params": 0, "bytes": 0.0, "flops": 0.0,
                              "act_bytes": 0.0, "count": 0},
            "gate": {"params": 0, "bytes": 0.0, "flops": 0.0,
                     "act_bytes": 0.0, "count": 0},
            "routedexpert": {"params": 0, "bytes": 0.0, "flops": 0.0,
                             "act_bytes": 0.0, "count": 0},
            "sharedexpert": {"params": 0, "bytes": 0.0, "flops": 0.0,
                             "act_bytes": 0.0, "count": 0},
            "swa": {"params": 0, "bytes": 0.0, "flops": 0.0,
                    "act_bytes": 0.0, "count": 0},
            "csa": {"params": 0, "bytes": 0.0, "flops": 0.0,
                    "act_bytes": 0.0, "count": 0},
            "hca": {"params": 0, "bytes": 0.0, "flops": 0.0,
                    "act_bytes": 0.0, "count": 0},
            "mhc": {"params": 0, "bytes": 0.0, "flops": 0.0,
                    "act_bytes": 0.0, "count": 0},
        }

        # WordEmbedding — 1 entry
        # Excel F108 = H95 = token_embedding only (not the output lm_head;
        # Excel assumes tied weights even when tie_word_embeddings=False).
        emb_breakdown = emb_stats.param_breakdown
        emb_token_stats = [pb for pb in emb_breakdown
                           if pb.name == "token_embedding"]
        emb_params = sum(pb.num_elements for pb in emb_token_stats)
        emb_bytes = sum(pb.num_bytes for pb in emb_token_stats)
        excel_summary["wordembedding"]["params"] = emb_params
        excel_summary["wordembedding"]["bytes"] = emb_bytes
        excel_summary["wordembedding"]["flops"] += emb_stats.num_flops
        # Excel does not count WordEmbedding activations separately.
        excel_summary["wordembedding"]["act_bytes"] += 0
        excel_summary["wordembedding"]["count"] += 1

        # Per-layer modules
        for spec, modules in zip(self.layer_specs, self.layers):
            for mod in modules:
                stats = mod.analyze(S, B)
                mt = stats.module_type
                if mt in ("CSA", "HCA", "SWA"):
                    key = mt.lower()
                    s = excel_summary[key]
                    s["params"] += stats.num_params
                    s["bytes"] += stats.num_bytes
                    s["flops"] += stats.num_flops
                    s["act_bytes"] += stats.num_activations_bytes
                    s["count"] += 1
                elif mt == "Indexer":
                    # Indexer FLOPs are ALREADY included inside the CSA
                    # module's compute_flops (Excel O27/O28/P28/P29 are
                    # sub-rows of O31 = SUM(O21:P30)). We skip the standalone
                    # Indexer module's FLOPs here to avoid double-counting,
                    # but still accumulate its params/activations into the
                    # CSA row for completeness.
                    s = excel_summary["csa"]
                    s["params"] += stats.num_params
                    s["bytes"] += stats.num_bytes
                    # NOTE: stats.num_flops intentionally NOT added — already
                    # counted by CSA.compute_flops.
                    s["act_bytes"] += stats.num_activations_bytes
                elif mt == "MoE":
                    # Split MoE into gate / routed_experts / shared_experts by
                    # param_breakdown name prefix — matches Excel rows 87-89.
                    for pb in stats.param_breakdown:
                        n = pb.name
                        if n.startswith("routed_experts"):
                            key = "routedexpert"
                        elif n.startswith("shared_experts"):
                            key = "sharedexpert"
                        elif n in ("gate", "gate_bias", "tid2eid"):
                            key = "gate"
                        else:
                            # Norms and other small params: attribute to gate
                            # (matches Excel gate row G90 = G87+G88+G89).
                            key = "gate"
                        s = excel_summary[key]
                        s["params"] += pb.num_elements
                        s["bytes"] += pb.num_bytes
                    # Per Excel R89-R94, the per-layer MoE FLOPs are split as:
                    #   * Gate routing FLOPs (= 2·B·S·H·n_act) → "gate" row (R90)
                    #   * Routed + shared experts FLOPs (O83)   → "routedexpert"
                    #     row (R94). Excel does not list "sharedexpert" FLOPs
                    #     separately — its compute (P79:P81) is part of O83.
                    # We compute these components inline instead of re-using
                    # MoE.compute_flops as a black box.
                    H = mod.hidden_size
                    I = mod.moe_intermediate_size
                    E_act = mod.num_activated_experts
                    E_shared = mod.num_shared_experts
                    # Excel Q90: 2·B·S·H·n_act  (gate routing)
                    gate_flops = 2 * B * S * H * E_act
                    # Excel U17 formula uses O83 × (1 + n_act) — the
                    # "theoretical peak" convention (1 shared + n_act routed
                    # experts computed in parallel).  We adopt this U-column
                    # convention so per_module_summary_excel matches U17.
                    expert_flops = (3 * (2 * B * S * H * I * E_act)
                                    + (2 + 1 + 1) * B * S * I * E_shared)
                    expert_flops *= (1 + E_act)            # U-column (1+n_act) factor
                    excel_summary["gate"]["flops"] += gate_flops
                    # Excel J87 = B*S*B40*B27 — gate routing activations
                    # (tiny buffer of bf16 logits × n_activated, per layer).
                    _bf16 = get_dtype_size(mod.dtype)
                    excel_summary["gate"]["act_bytes"] += B * S * _bf16 * E_act
                    excel_summary["gate"]["count"] += 1
                    excel_summary["routedexpert"]["flops"] += expert_flops
                    excel_summary["routedexpert"]["act_bytes"] += stats.num_activations_bytes
                    excel_summary["routedexpert"]["count"] += 1
                    # sharedexpert also gets a count (per Excel H90 sub-row),
                    # but has no separate FLOPs (counted in MoE/O83).
                    excel_summary["sharedexpert"]["count"] += 1
                elif mt == "mHC":
                    # Excel R89: mHC is listed as a separate row, count=61
                    # (one per layer, treating pre+post as a single block).
                    # Each mHC instance now computes the full P7 per-layer
                    # FLOPs, so we only count FLOPs/params from ONE instance
                    # per layer (skip the "post" duplicate) and add 0.5 to
                    # count per instance (final count = 61).
                    s = excel_summary["mhc"]
                    is_post = "_mHC_post" in mod.name
                    if not is_post:
                        s["params"] += stats.num_params
                        s["bytes"] += stats.num_bytes
                        s["flops"] += stats.num_flops
                    s["act_bytes"] += stats.num_activations_bytes
                    s["count"] += 0.5

        # KV cache estimate
        kv_cache = self.estimate_kv_cache(S, B)

        # Training memory estimate
        training_mem = self.estimate_training_memory(total_bytes, S, B)

        # ------------------------------------------------------------------
        # Re-compute total_flops_per_sample using the Excel U17
        # "theoretical-peak" formula, so the report's top-level compute
        # field matches Excel cell U17 exactly:
        #
        #   U17 = U10*31 + U13*30 + O18 + U8*(1+B27)
        #       = (O31 + 7*O83)*31 + (O45 + 7*O83)*30 + O18 + 7*O83
        #       = 31*CSA单层 + 30*HCA单层 + 1*SWA单层 + 7*62*MoE单层(O83)
        #
        # where (1+n_act)=7 treats MoE as 1 shared + n_act routed experts
        # computed in parallel.  Note U17 uses HCA=30 layers and MoE=62
        # "slots" (31 CSA + 30 HCA + 1 extra from the trailing U8*7 term),
        # which differs slightly from the model's actual 29 HCA / 61 MoE
        # layers — this is the Excel sheet's own convention.
        #
        # routedexpert in excel_summary already carries the (1+E_act) factor
        # (see MoE accumulation above), so excel_summary["routedexpert"]["flops"]
        # = 7*O83*61.  We recover O83 (single-layer, no factor) from it.
        # ------------------------------------------------------------------
        csa_cnt = excel_summary["csa"]["count"] or 1
        hca_cnt = excel_summary["hca"]["count"] or 1
        swa_cnt = excel_summary["swa"]["count"] or 1
        moe_cnt = excel_summary["routedexpert"]["count"] or 1
        csa_single = excel_summary["csa"]["flops"] / csa_cnt
        hca_single = excel_summary["hca"]["flops"] / hca_cnt
        swa_single = excel_summary["swa"]["flops"] / swa_cnt
        # routedexpert flops already = O83 * (1+n_act) * moe_cnt, so:
        n_act = self.config["layers"]["moe"].get("num_activated_experts", 6)
        O83_total = excel_summary["routedexpert"]["flops"]
        O83_single = O83_total / ((1 + n_act) * moe_cnt)
        total_flops = (31 * csa_single
                       + 30 * hca_single
                       + 1 * swa_single
                       + (1 + n_act) * 62 * O83_single)

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
            "per_module_summary_excel": excel_summary,
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
        KV representation (single KV head's head_dim + rope_dim) is stored per
        token, not the full per-head K/V.
        """
        B, S = batch_size, seq_len
        elem = get_dtype_size(self.dtype)
        total = 0.0

        for spec, modules in zip(self.layer_specs, self.layers):
            for mod in modules:
                if mod.module_type not in ("CSA", "HCA"):
                    continue
                # Single KV head (num_key_value_heads=1): KV cache stores
                # K (head_dim) + V (v_head_dim) per token, for n_kv heads.
                n_kv = mod.config.get("num_key_value_heads",
                                       mod.config.get("num_kv_heads", 1))
                head_dim = mod.config.get("head_dim", 128)
                v_head_dim = mod.config.get("v_head_dim", head_dim)
                total += B * S * n_kv * (head_dim + v_head_dim) * elem
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