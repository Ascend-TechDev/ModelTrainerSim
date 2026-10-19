"""V4.1-Flash attention parameters, forward FLOPs and activation accounting.

SWA, Full, Reindex and Reuse share the same attention projections. Their
compression and index ownership determine the additional work and cache.
DSpark uses the same projections for one draft invocation, not a complete
teacher-forced forward pass. FLOPs count major matrix products; activation
bytes sum named forward tensors and are not a measured memory peak.
"""

from __future__ import annotations

from typing import Any

from ....core.base_module import ModuleParamBreakdown
from .common import FlashModule


def _causal_window_pairs(length: int, window: int) -> int:
    """sum(min(t, window), t=1..length), including the current position."""
    short = min(length, window)
    return short * (short + 1) // 2 + (length - short) * window


def _floor_sum(length: int, ratio: int) -> int:
    """sum(floor(t/ratio), t=1..length), without an O(sequence) loop."""
    groups, remainder = divmod(length, ratio)
    return ratio * groups * (groups - 1) // 2 + groups * (remainder + 1)


def _compressed_pairs(length: int, ratio: int, topk: int) -> int:
    cutoff = min(length, topk * ratio - 1)
    return _floor_sum(cutoff, ratio) + (length - cutoff) * topk


def _previous_source(sources: list[int], layer_id: int, label: str) -> int:
    previous = [i for i in sources if i <= layer_id]
    if not previous:
        raise ValueError(f"layer {layer_id}: compressed attention has no {label} source")
    return max(previous)


class Attention(FlashModule):
    """Attention block with independently callable accounting methods."""

    module_type = "Attention"
    block_accounting = False

    def __init__(
        self,
        name: str,
        config: dict[str, Any],
        *,
        layer_id: int,
        dspark: bool = False,
    ):
        super().__init__(name, config, layer_id=layer_id, dspark=dspark)
        arch = self.arch
        if self.activation_element_bytes <= 0 or self.cache_element_bytes <= 0:
            raise ValueError("element byte sizes must be positive")
        if layer_id < 0 or layer_id >= len(arch["compress_ratios"]):
            raise ValueError(f"layer_id {layer_id} is outside compress_ratios")

        self.hidden_size = int(arch["dim"])
        self.num_heads = int(arch["n_heads"])
        self.head_dim = int(arch["head_dim"])
        self.q_lora_rank = int(arch["q_lora_rank"])
        self.o_lora_rank = int(arch["o_lora_rank"])
        self.o_groups = int(arch["o_groups"])
        self.window_size = int(arch["window_size"])
        self.compress_ratio = int(arch["compress_ratios"][layer_id])
        if min(self.hidden_size, self.num_heads, self.head_dim, self.q_lora_rank,
               self.o_lora_rank, self.o_groups, self.window_size) <= 0:
            raise ValueError("attention dimensions and window_size must be positive")
        if self.num_heads % self.o_groups:
            raise ValueError("n_heads must be divisible by o_groups")
        if self.compress_ratio < 0:
            raise ValueError("compress_ratio cannot be negative")
        if dspark and (layer_id < arch["n_layers"] or self.compress_ratio != 0):
            raise ValueError("DSpark attention requires an auxiliary layer with compress_ratio=0")
        if not dspark and layer_id >= arch["n_layers"]:
            raise ValueError("auxiliary layer requires dspark=True")

        kv_sources = list(arch["kv_source_layers"])
        index_sources = list(arch["index_source_layers"])
        self.owns_kv = not dspark and layer_id in kv_sources
        self.owns_index = not dspark and layer_id in index_sources
        if (self.owns_kv or self.owns_index) and not self.compress_ratio:
            raise ValueError("a compression/index source cannot have compress_ratio=0")
        if self.owns_kv and not self.owns_index:
            raise ValueError("KV sources must also be index sources in this implementation")
        self.kv_source = _previous_source(kv_sources, layer_id, "KV") if self.compress_ratio else None
        self.index_source = _previous_source(index_sources, layer_id, "index") if self.compress_ratio else None
        if self.compress_ratio and arch["compress_ratios"][self.kv_source] != self.compress_ratio:
            raise ValueError("reuse layer and its KV owner have different compression ratios")
        if self.compress_ratio and self.index_source < self.kv_source:
            raise ValueError("index source precedes the current KV source")

        self.module_type = "DSpark-SWA" if dspark else (
            "SWA" if not self.compress_ratio else "Full" if self.owns_kv
            else "Reindex" if self.owns_index else "Reuse"
        )
        self.index_topk = int(arch["index_topk"])
        if self.index_topk <= 0:
            raise ValueError("index_topk must be positive")
        if self.owns_index:
            self.index_n_heads = int(arch["index_n_heads"])
            self.index_head_dim = int(arch["index_head_dim"])
            self.candidate_source = int(arch.get("candidate_source_layer", -1))
            self.candidate_positions = int(arch["candidate_topk_blocks"]) * int(arch["candidate_block_size"])

    def _dimensions(self, seq_len: int, batch_size: int) -> tuple[int, int, int]:
        """Return query length, query tokens and completed compressed length."""
        if seq_len < 1 or batch_size < 1:
            raise ValueError("seq_len and batch_size must be positive")
        queries = int(self.arch["dspark_block_size"]) if self.dspark else seq_len
        if queries <= 0:
            raise ValueError("dspark_block_size must be positive")
        compressed_length = seq_len // self.compress_ratio if self.compress_ratio else 0
        return queries, batch_size * queries, compressed_length

    def _attention_pairs(self, seq_len: int, queries: int) -> tuple[int, int, int]:
        """Window, selected compressed and draft pairs, for one sample."""
        if self.dspark:
            return queries * min(seq_len, self.window_size), 0, queries * queries
        return (
            _causal_window_pairs(seq_len, self.window_size),
            _compressed_pairs(seq_len, self.compress_ratio, self.index_topk)
            if self.compress_ratio else 0,
            0,
        )

    # -----------------------------------------------------------------
    # Parameters
    # -----------------------------------------------------------------

    def compute_params(self) -> list[ModuleParamBreakdown]:
        def parameter(name: str, shape: tuple[int, ...]) -> ModuleParamBreakdown:
            return self.parameter(f"{self.name}.{name}", shape)

        H, h, d = self.hidden_size, self.num_heads, self.head_dim
        rq, ro, G = self.q_lora_rank, self.o_lora_rank, self.o_groups
        # wo_a is stored [G*ro, h*d/G]; grouping does not add another factor G.
        params = [
            parameter("attn_sink", (h,)),
            parameter("wq_a.weight", (rq, H)),
            parameter("q_norm.weight", (rq,)),
            parameter("wq_b.weight", (h * d, rq)),
            parameter("wkv.weight", (d, H)),
            parameter("kv_norm.weight", (d,)),
            parameter("wo_a.weight", (G * ro, h * d // G)),
            parameter("wo_b.weight", (H, G * ro)),
        ]
        if self.owns_kv:
            params.extend([
                parameter("compressor.wkv.weight", (d, H)),
                parameter("compressor.norm.weight", (d,)),
            ])
            if self.compress_ratio > 1:
                params.append(parameter("compressor.wgate.weight", (d, H)))
        if self.owns_index:
            ih, idim = self.index_n_heads, self.index_head_dim
            params.extend([
                parameter("indexer.wq_b.weight", (ih * idim, rq)),
                parameter("indexer.weights_proj.weight", (ih, H)),
            ])
            if self.owns_kv:
                params.extend([
                    parameter("indexer.wk.weight", (idim, d)),
                    parameter("indexer.k_norm.weight", (idim,)),
                ])
        return params

    # -----------------------------------------------------------------
    # FLOPs
    # -----------------------------------------------------------------

    def compute_flops(self, seq_len: int, batch_size: int = 1) -> float:
        """Major forward GEMMs; one multiply-add counts as two operations."""
        return float(sum(self.compute_flops_breakdown(seq_len, batch_size).values()))

    def compute_flops_breakdown(self, seq_len: int, batch_size: int = 1) -> dict[str, int]:
        queries, tokens, compressed_length = self._dimensions(seq_len, batch_size)
        H, h, d = self.hidden_size, self.num_heads, self.head_dim
        rq, ro, G = self.q_lora_rank, self.o_lora_rank, self.o_groups
        projection_elements = H * rq + rq * h * d + H * d + h * d * ro + G * ro * H
        flops = {"attention_projections": 2 * tokens * projection_elements}
        if self.dspark:
            # The draft pass also projects the single new main-model token.
            flops["dspark_main_kv_projection"] = 2 * batch_size * H * d
        pairs = self._attention_pairs(seq_len, queries)
        # Shared K/V latent: QK and AV each cost 2*d per head per valid pair.
        flops["sparse_attention_qk_av"] = 4 * batch_size * h * d * sum(pairs)
        if self.owns_kv:
            flops["compressor_projection"] = 2 * tokens * H * d
            if self.compress_ratio > 1:
                flops["compressor_gate_projection"] = 2 * tokens * H * d
        # _compress_topk_idxs skips the Indexer when no group is complete.
        if self.owns_index and compressed_length:
            ih, idim = self.index_n_heads, self.index_head_dim
            flops["indexer_query_and_weights_projection"] = 2 * tokens * (rq * ih * idim + H * ih)
            if self.owns_kv:
                flops["indexer_key_projection"] = 2 * batch_size * compressed_length * d * idim
            # The reference einsum precedes causal and candidate masking.
            flops["indexer_dense_reference_qk"] = 2 * tokens * compressed_length * ih * idim
        return flops

    # -----------------------------------------------------------------
    # Activations
    # -----------------------------------------------------------------

    def compute_activations(self, seq_len: int, batch_size: int = 1) -> float:
        """Sum named forward tensors, excluding temporary integer buffers."""
        return float(sum(self.compute_activation_tensor_bytes(seq_len, batch_size).values()))

    def compute_activation_tensor_bytes(self, seq_len: int, batch_size: int = 1) -> dict[str, float]:
        _, tokens, compressed_length = self._dimensions(seq_len, batch_size)
        H, h, d = self.hidden_size, self.num_heads, self.head_dim
        elem = self.activation_element_bytes
        act = {
            "input": tokens * H * elem,
            "q_low_rank": tokens * self.q_lora_rank * elem,
            "query": tokens * h * d * elem,
            "window_kv_projection": tokens * d * elem,
            "attention_output": tokens * h * d * elem,
            "grouped_output": tokens * self.o_groups * self.o_lora_rank * elem,
            "output": tokens * H * elem,
        }
        if self.dspark:
            act["main_kv_projection"] = batch_size * d * elem
        if self.owns_kv:
            act["compressor_projection"] = tokens * d * (4 if self.compress_ratio > 1 else elem)
            act["compressed_latent"] = batch_size * compressed_length * d * elem
            if self.compress_ratio > 1:
                act["compressor_gate_projection"] = tokens * d * 4
        if self.owns_index and compressed_length:
            ih, idim = self.index_n_heads, self.index_head_dim
            if self.owns_kv:
                act["index_key_projection"] = batch_size * compressed_length * idim * elem
            act["index_query"] = tokens * ih * idim * elem
            act["index_head_weights"] = tokens * ih * elem
            act["index_per_head_scores_reference"] = tokens * compressed_length * ih * elem
            act["index_reduced_scores_reference"] = tokens * compressed_length * elem
        if self.compress_ratio:
            # Concatenated raw/compressed KV is a forward tensor, not an owner cache.
            act["concatenated_kv"] = batch_size * (seq_len + compressed_length) * d * elem
        return act

    # -----------------------------------------------------------------
    # Cache and temporary index buffers (separate from activation totals)
    # -----------------------------------------------------------------

    def compute_cache_bytes(self, seq_len: int, batch_size: int = 1) -> float:
        return float(sum(self.compute_cache_breakdown(seq_len, batch_size).values()))

    def compute_cache_breakdown(self, seq_len: int, batch_size: int = 1) -> dict[str, float]:
        _, _, compressed_length = self._dimensions(seq_len, batch_size)
        d, elem = self.head_dim, self.cache_element_bytes
        cache = {"window_kv": batch_size * min(seq_len, self.window_size) * d * elem}
        if self.owns_kv:
            if self.compress_ratio > 1:
                # Pooling states remain allocated even when seq_len divides ratio.
                cache["compressor_kv_state_fp32"] = batch_size * self.compress_ratio * d * 4
                cache["compressor_score_state_fp32"] = batch_size * self.compress_ratio * d * 4
            cache["compressed_kv"] = batch_size * compressed_length * d * elem
        if self.owns_index and self.owns_kv:
            cache["index_k"] = batch_size * compressed_length * self.index_head_dim * elem
        return cache

    def compute_temporary_index_buffers(self, seq_len: int, batch_size: int = 1) -> dict[str, int]:
        queries, tokens, compressed_length = self._dimensions(seq_len, batch_size)
        window = min(seq_len, self.window_size)
        selected = min(self.index_topk, compressed_length)
        if self.dspark:
            temporary = {"window_and_draft_indices_int32": batch_size * queries * (window + queries) * 4}
        else:
            temporary = {"window_indices_int32": tokens * window * 4}
        if self.owns_index:
            if self.layer_id == self.candidate_source and compressed_length:
                temporary["shared_candidate_mask_bool"] = tokens * compressed_length
            temporary["published_compressed_indices_int32"] = tokens * selected * 4
        elif self.compress_ratio:
            temporary["reused_compressed_indices_int32_reference_only"] = tokens * selected * 4
        if self.compress_ratio:
            temporary["concatenated_attention_indices_int32"] = tokens * (window + selected) * 4
        return temporary

    # -----------------------------------------------------------------
    # Report metadata
    # -----------------------------------------------------------------

    def compute_details(self, seq_len: int, batch_size: int = 1) -> dict[str, Any]:
        queries, _, compressed_length = self._dimensions(seq_len, batch_size)
        window_pairs, compressed_pairs, draft_pairs = self._attention_pairs(seq_len, queries)
        index_details = {}
        if self.owns_index:
            uses_candidates = 0 <= self.candidate_source < self.layer_id
            index_details = {
                "reference_score_scope": "all compressed positions; masking follows dense einsum",
                "causal_useful_score_flops": 2 * batch_size * self.index_n_heads * self.index_head_dim * _floor_sum(seq_len, self.compress_ratio),
                "candidate_source": self.candidate_source if self.candidate_source >= 0 else None,
                "produces_candidates": self.layer_id == self.candidate_source,
                "uses_candidates": uses_candidates,
                "candidate_positions_per_query_upper_bound": min(compressed_length, self.candidate_positions) if uses_candidates else None,
                "candidate_restricted_causal_pairs_upper_bound": _compressed_pairs(seq_len, self.compress_ratio, self.candidate_positions) if uses_candidates else None,
                "candidate_pruning_reduces_reference_gemm": False,
                "topk_sort_and_candidate_selection_flops_included": False,
            }
        return {
            "source": "inference/model.py: Compressor, Indexer, Attention, DSparkAttention",
            "layer_id": self.layer_id,
            "attention_kind": self.module_type,
            "compress_ratio": self.compress_ratio,
            "kv_source": self.kv_source,
            "index_source": self.index_source,
            "owns_compressed_kv": self.owns_kv,
            "owns_indexer": self.owns_index,
            "owns_index_k": self.owns_kv,
            "query_length": queries,
            "prefix_length": seq_len if self.dspark else None,
            "completed_compressed_positions": compressed_length,
            "valid_window_pairs_per_sample": window_pairs,
            "selected_compressed_pairs_per_sample": compressed_pairs,
            "draft_pairs_per_sample": draft_pairs,
            "flops_breakdown": self.compute_flops_breakdown(seq_len, batch_size),
            "activation_tensor_bytes": self.compute_activation_tensor_bytes(seq_len, batch_size),
            "cache_breakdown_bytes": self.compute_cache_breakdown(seq_len, batch_size),
            "temporary_index_buffer_bytes": self.compute_temporary_index_buffers(seq_len, batch_size),
            "indexer": index_details,
            "activation_accounting": "sum of named forward tensor sizes, including source-materialized index scores; not peak or autograd saved tensors",
            "cache_accounting": "logical filled caches plus allocated fp32 pooling state; owners only; no max-context preallocation, RoPE tables or quantization metadata",
            "temporary_index_buffers_included_in_activation_total": False,
            "flops_accounting": "major forward GEMMs plus valid sparse QK/AV; excludes elementwise, norm, softmax, topk, quantization and communication",
            "dspark_accounting": "one drafting invocation; seq_len is committed prefix length, query_length is draft block size; not backbone teacher-forced FLOPs" if self.dspark else None,
        }
