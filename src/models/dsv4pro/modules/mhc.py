"""mHC — mini Hierarchical Compression attention.

A lightweight variant of HCA used in DSV4Pro's later layers for efficiency.
mHC uses a single-level compression (no block hierarchy) with a smaller
compression rank and grouped-query attention (GQA) to further reduce KV cache
and parameter count.

Key parameters
--------------
hidden_size : int
kv_lora_rank : int       – small compression rank
num_query_heads : int    – number of query heads (Q)
num_kv_heads : int       – number of KV heads (GQA groups)
head_dim : int
qk_rope_head_dim : int
"""

from __future__ import annotations

from typing import Any

from ....core.base_module import BaseModule, ModuleParamBreakdown
from ....core.dtype import DType


class mHC(BaseModule):
    module_type = "mHC"

    def __init__(self, name: str, config: dict[str, Any]):
        super().__init__(name, config)
        self.hidden_size: int = config["hidden_size"]
        self.kv_lora_rank: int = config["kv_lora_rank"]
        self.num_query_heads: int = config["num_query_heads"]
        self.num_kv_heads: int = config.get("num_kv_heads", self.num_query_heads)
        self.head_dim: int = config.get("head_dim", 128)
        self.qk_rope_head_dim: int = config.get("qk_rope_head_dim", 64)
        self.v_head_dim: int = config.get("v_head_dim", self.head_dim)
        self.dtype: DType = DType(config.get("dtype", "bf16"))

    # -----------------------------------------------------------------

    def compute_params(self) -> list[ModuleParamBreakdown]:
        params: list[ModuleParamBreakdown] = []
        H = self.hidden_size
        kv_lora = self.kv_lora_rank
        n_q = self.num_query_heads
        n_kv = self.num_kv_heads
        d_h = self.head_dim
        d_rope = self.qk_rope_head_dim
        v_hd = self.v_head_dim

        # Q projection
        params.append(ModuleParamBreakdown(
            "q_proj", (H, n_q * (d_h + d_rope)),
            H * n_q * (d_h + d_rope), self.dtype))

        # KV compression (shared by GQA groups)
        kv_compress_dim = kv_lora + d_rope
        params.append(ModuleParamBreakdown(
            "kv_down_proj", (H, kv_compress_dim),
            H * kv_compress_dim, self.dtype))

        # K decompression (only n_kv groups, not n_q)
        params.append(ModuleParamBreakdown(
            "k_up_proj", (kv_lora, n_kv * d_h),
            kv_lora * n_kv * d_h, self.dtype))
        # V decompression
        params.append(ModuleParamBreakdown(
            "v_up_proj", (kv_lora, n_kv * v_hd),
            kv_lora * n_kv * v_hd, self.dtype))

        # Output projection
        params.append(ModuleParamBreakdown(
            "o_proj", (n_q * v_hd, H),
            n_q * v_hd * H, self.dtype))

        # Norms
        params.append(ModuleParamBreakdown(
            "input_layernorm", (H,), H, self.dtype))
        params.append(ModuleParamBreakdown(
            "post_attention_layernorm", (H,), H, self.dtype))

        return params

    # -----------------------------------------------------------------

    def compute_flops(self, seq_len: int, batch_size: int = 1) -> float:
        H = self.hidden_size
        kv_lora = self.kv_lora_rank
        n_q = self.num_query_heads
        n_kv = self.num_kv_heads
        d_h = self.head_dim
        d_rope = self.qk_rope_head_dim
        d_full = d_h + d_rope
        v_hd = self.v_head_dim
        B, S = batch_size, seq_len

        flops = 0.0

        # Q projection
        flops += 2 * B * S * H * n_q * d_full
        # KV compression
        kv_compress_dim = kv_lora + d_rope
        flops += 2 * B * S * H * kv_compress_dim
        # K/V decompression (only n_kv heads)
        flops += 2 * B * S * kv_lora * n_kv * d_h
        flops += 2 * B * S * kv_lora * n_kv * v_hd

        # Attention (standard dense, GQA broadcasting KV to Q heads)
        flops += 2 * n_q * B * S * S * d_h       # QK^T
        flops += 3 * n_q * B * S * S               # softmax
        flops += 2 * n_q * B * S * S * v_hd         # attn × V

        # Output projection
        flops += 2 * B * S * n_q * v_hd * H

        return flops

    # -----------------------------------------------------------------

    def compute_activations(self, seq_len: int, batch_size: int = 1) -> float:
        from ....core.dtype import get_dtype_size
        B, S = batch_size, seq_len
        n_q = self.num_query_heads
        d_full = self.head_dim + self.qk_rope_head_dim
        v_hd = self.v_head_dim
        elem = get_dtype_size(self.dtype)

        act = 0.0
        act += B * S * n_q * d_full * elem
        act += B * S * self.num_kv_heads * self.head_dim * elem * 0.5
        act += B * S * self.num_kv_heads * v_hd * elem * 0.5
        act += B * n_q * S * S * elem              # attn weights (dense)
        act += B * S * n_q * v_hd * elem
        act += B * S * self.hidden_size * elem
        return act