"""HCA — Hierarchical Compressed Attention.

Used in DSV4Pro for long-context layers.  HCA builds a two-level hierarchy:

1. **Anchor / block level** – tokens are grouped into blocks of ``block_size``;
   a single representative query per block attends to all block-level summaries
   (coarse attention).
2. **Token level** – within selected top blocks, individual tokens attend
   to all tokens in those blocks (fine attention).

This reduces the effective sequence length from ``S`` to ``S / block_size``
for the coarse pass, and then to ``topk_blocks * block_size`` for the fine pass.

Key parameters
--------------
hidden_size : int
kv_lora_rank : int       – compression rank for K/V
num_heads : int
head_dim : int
block_size : int          – tokens per block (anchor granularity)
topk_blocks : int         – number of blocks selected per query
qk_rope_head_dim : int   – RoPE head dim
"""

from __future__ import annotations

from typing import Any

from ....core.base_module import BaseModule, ModuleParamBreakdown
from ....core.dtype import DType


class HCA(BaseModule):
    module_type = "HCA"

    def __init__(self, name: str, config: dict[str, Any]):
        super().__init__(name, config)
        self.hidden_size: int = config["hidden_size"]
        self.kv_lora_rank: int = config["kv_lora_rank"]
        self.num_heads: int = config["num_heads"]
        self.head_dim: int = config.get("head_dim", 128)
        self.qk_rope_head_dim: int = config.get("qk_rope_head_dim", 64)
        self.v_head_dim: int = config.get("v_head_dim", self.head_dim)
        self.block_size: int = config.get("block_size", 128)
        self.topk_blocks: int = config.get("topk_blocks", 16)
        self.dtype: DType = DType(config.get("dtype", "bf16"))

    # -----------------------------------------------------------------

    def compute_params(self) -> list[ModuleParamBreakdown]:
        params: list[ModuleParamBreakdown] = []
        H = self.hidden_size
        kv_lora = self.kv_lora_rank
        n_h = self.num_heads
        d_h = self.head_dim
        d_rope = self.qk_rope_head_dim
        v_hd = self.v_head_dim

        # Q projection (direct)
        params.append(ModuleParamBreakdown(
            "q_proj", (H, n_h * (d_h + d_rope)),
            H * n_h * (d_h + d_rope), self.dtype))

        # KV compression
        kv_compress_dim = kv_lora + d_rope
        params.append(ModuleParamBreakdown(
            "kv_down_proj", (H, kv_compress_dim),
            H * kv_compress_dim, self.dtype))

        # KV decompression
        params.append(ModuleParamBreakdown(
            "k_up_proj", (kv_lora, n_h * d_h),
            kv_lora * n_h * d_h, self.dtype))
        params.append(ModuleParamBreakdown(
            "v_up_proj", (kv_lora, n_h * v_hd),
            kv_lora * n_h * v_hd, self.dtype))

        # Block summary projection (anchor aggregation)
        params.append(ModuleParamBreakdown(
            "block_summary_proj", (H, H),
            H * H, self.dtype))

        # Output projection
        params.append(ModuleParamBreakdown(
            "o_proj", (n_h * v_hd, H),
            n_h * v_hd * H, self.dtype))

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
        n_h = self.num_heads
        d_h = self.head_dim
        d_rope = self.qk_rope_head_dim
        d_full = d_h + d_rope
        v_hd = self.v_head_dim
        B, S = batch_size, seq_len
        bs = self.block_size
        num_blocks = max(S // bs, 1)
        topk_blk = min(self.topk_blocks, num_blocks)
        fine_len = topk_blk * bs

        flops = 0.0

        # Q projection
        flops += 2 * B * S * H * n_h * d_full

        # KV compression
        kv_compress_dim = kv_lora + d_rope
        flops += 2 * B * S * H * kv_compress_dim
        # K/V decompression
        flops += 2 * B * S * kv_lora * n_h * d_h
        flops += 2 * B * S * kv_lora * n_h * v_hd

        # Block summary projection
        flops += 2 * B * S * H * H

        # Coarse attention (S queries × num_blocks blocks)
        flops += 2 * n_h * B * S * num_blocks * d_h    # QK^T
        flops += 3 * n_h * B * S * num_blocks            # softmax
        flops += 2 * n_h * B * S * num_blocks * v_hd     # attn × V (coarse)

        # Fine attention (S queries × fine_len tokens)
        flops += 2 * n_h * B * S * fine_len * d_h       # QK^T
        flops += 3 * n_h * B * S * fine_len              # softmax
        flops += 2 * n_h * B * S * fine_len * v_hd       # attn × V (fine)

        # Output projection
        flops += 2 * B * S * n_h * v_hd * H

        return flops

    # -----------------------------------------------------------------

    def compute_activations(self, seq_len: int, batch_size: int = 1) -> float:
        from ....core.dtype import get_dtype_size
        B, S = batch_size, seq_len
        n_h = self.num_heads
        d_full = self.head_dim + self.qk_rope_head_dim
        v_hd = self.v_head_dim
        bs = self.block_size
        num_blocks = max(S // bs, 1)
        topk_blk = min(self.topk_blocks, num_blocks)
        fine_len = topk_blk * bs
        elem = get_dtype_size(self.dtype)

        act = 0.0
        act += B * S * n_h * d_full * elem
        act += B * S * n_h * self.head_dim * elem * 0.5
        act += B * S * n_h * v_hd * elem * 0.5
        act += B * n_h * S * num_blocks * elem      # coarse attn weights
        act += B * n_h * S * fine_len * elem         # fine attn weights
        act += B * S * n_h * v_hd * elem             # output
        act += B * S * self.hidden_size * elem       # residual
        return act