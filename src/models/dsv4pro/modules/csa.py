"""CSA — Compressed Sparse Attention.

Based on DSV4Pro's sparse-attention variant used in early layers.
CSA compresses the K/V dimensionally (low-rank projection) then applies
a top-k sparse selection over attention scores, reducing the quadratic
cost of standard MHA while retaining representational quality.

Key parameters
--------------
hidden_size : int      – model hidden dimension ``d``
q_lora_rank : int       – LoRA rank for Q projection (0 = no LoRA)
kv_lora_rank : int      – LoRA / compression rank for K/V (MLA-style)
num_heads : int         – number of attention heads
head_dim : int          – dimension per head (decompressed)
sparse_topk : int       – number of top tokens each query attends to
qk_nope_head_dim : int  – "nope" (non-positional) part of head dim
qk_rope_head_dim : int  – RoPE part of head dim
v_head_dim : int        – output head dimension
"""

from __future__ import annotations

from typing import Any

from ....core.base_module import BaseModule, ModuleParamBreakdown
from ....core.dtype import DType


class CSA(BaseModule):
    module_type = "CSA"

    def __init__(self, name: str, config: dict[str, Any]):
        super().__init__(name, config)
        self.hidden_size: int = config["hidden_size"]
        self.q_lora_rank: int = config.get("q_lora_rank", 0)
        self.kv_lora_rank: int = config["kv_lora_rank"]
        self.num_heads: int = config["num_heads"]
        self.qk_rope_head_dim: int = config.get("qk_rope_head_dim", 0)
        self.qk_nope_head_dim: int = config.get("qk_nope_head_dim", config.get("head_dim", 128))
        self.v_head_dim: int = config.get("v_head_dim", self.qk_nope_head_dim)
        self.sparse_topk: int = config.get("sparse_topk", 256)
        self.rms_norm_eps: float = config.get("rms_norm_eps", 1e-6)
        self.dtype: DType = DType(config.get("dtype", "bf16"))
        self._calculate()

    def _calculate(self) -> None:
        self.query_load = (self.num_heads * (self.qk_nope_head_dim + self.qk_rope_head_dim)
                           * self.hidden_size) if self.q_lora_rank == 0 else 0

    # -----------------------------------------------------------------
    # Parameter breakdown
    # -----------------------------------------------------------------

    def compute_params(self) -> list[ModuleParamBreakdown]:
        params: list[ModuleParamBreakdown] = []
        H = self.hidden_size
        kv_lora = self.kv_lora_rank
        n_h = self.num_heads
        d_nope = self.qk_nope_head_dim
        d_rope = self.qk_rope_head_dim
        v_hd = self.v_head_dim

        # --- Q projection ---
        if self.q_lora_rank > 0:
            # Down-proj + Up-proj (MLA-style)
            params.append(ModuleParamBreakdown(
                "q_down_proj", (H, self.q_lora_rank),
                H * self.q_lora_rank, self.dtype))
            params.append(ModuleParamBreakdown(
                "q_up_proj",
                (self.q_lora_rank, n_h * (d_nope + d_rope)),
                self.q_lora_rank * n_h * (d_nope + d_rope), self.dtype))
        else:
            # Direct Q projection
            params.append(ModuleParamBreakdown(
                "q_proj",
                (H, n_h * (d_nope + d_rope)),
                H * n_h * (d_nope + d_rope), self.dtype))

        # --- KV compression (down-proj) ---
        kv_compress_dim = kv_lora + d_rope   # compressed K, V share
        params.append(ModuleParamBreakdown(
            "kv_down_proj", (H, kv_compress_dim),
            H * kv_compress_dim, self.dtype))

        # --- KV decompression (up-proj) ---
        # K_nope up-proj
        params.append(ModuleParamBreakdown(
            "k_nope_up_proj", (kv_lora, n_h * d_nope),
            kv_lora * n_h * d_nope, self.dtype))
        # V up-proj
        params.append(ModuleParamBreakdown(
            "v_up_proj", (kv_lora, n_h * v_hd),
            kv_lora * n_h * v_hd, self.dtype))

        # --- RoPE ---
        # rotary embedding has negligible parameters (sin/cos tables)

        # --- Output projection ---
        params.append(ModuleParamBreakdown(
            "o_proj", (n_h * v_hd, H),
            n_h * v_hd * H, self.dtype))

        # --- RMSNorm (Q/KV input norms) ---
        params.append(ModuleParamBreakdown(
            "input_layernorm", (H,), H, self.dtype))
        params.append(ModuleParamBreakdown(
            "post_attention_layernorm", (H,), H, self.dtype))

        return params

    # -----------------------------------------------------------------
    # FLOPs
    # -----------------------------------------------------------------

    def compute_flops(self, seq_len: int, batch_size: int = 1) -> float:
        H = self.hidden_size
        kv_lora = self.kv_lora_rank
        n_h = self.num_heads
        d_nope = self.qk_nope_head_dim
        d_rope = self.qk_rope_head_dim
        d_h = d_nope + d_rope
        v_hd = self.v_head_dim
        B, S = batch_size, seq_len
        topk = min(self.sparse_topk, S)

        flops = 0.0

        # Q projection
        if self.q_lora_rank > 0:
            flops += 2 * B * S * H * self.q_lora_rank     # down
            flops += 2 * B * S * self.q_lora_rank * n_h * d_h  # up
        else:
            flops += 2 * B * S * H * n_h * d_h

        # KV compression
        kv_compress_dim = kv_lora + d_rope
        flops += 2 * B * S * H * kv_compress_dim          # down
        # K_nope / V decompression
        flops += 2 * B * S * kv_lora * n_h * d_nope        # K up
        flops += 2 * B * S * kv_lora * n_h * v_hd          # V up

        # Attention QK^T (sparse: each query only attends to topk keys)
        # Score computation: n_h * B * S * topk * d_h  (multiply-add)
        flops += 2 * n_h * B * S * topk * d_h
        # Softmax + scaling: ~3 * n_h * B * S * topk
        flops += 3 * n_h * B * S * topk
        # Attention * V: n_h * B * S * topk * v_hd
        flops += 2 * n_h * B * S * topk * v_hd

        # Output projection
        flops += 2 * B * S * n_h * v_hd * H

        return flops

    # -----------------------------------------------------------------
    # Activations
    # -----------------------------------------------------------------

    def compute_activations(self, seq_len: int, batch_size: int = 1) -> float:
        from ....core.dtype import get_dtype_size
        B, S = batch_size, seq_len
        n_h = self.num_heads
        d_h = self.qk_nope_head_dim + self.qk_rope_head_dim
        v_hd = self.v_head_dim
        topk = min(self.sparse_topk, S)
        elem = get_dtype_size(self.dtype)

        # Q, K, V tensors + attention weights
        act = 0.0
        act += B * S * n_h * d_h * elem          # Q
        act += B * S * n_h * d_h * elem * 0.5    # K (compressed, approx half)
        act += B * S * n_h * v_hd * elem * 0.5   # V (compressed, approx half)
        act += B * n_h * S * topk * elem         # attention weights (sparse)
        act += B * S * n_h * v_hd * elem          # attn output
        act += B * S * self.hidden_size * elem    # residual
        return act