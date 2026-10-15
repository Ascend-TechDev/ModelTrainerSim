"""SWA — Sliding Window Attention.

In DSV4Pro, the *last* transformer layer (index = num_layers - 1) uses plain
sliding-window attention.  Unlike CSA/HCA, the SWA layer has **no Compressor
and no Indexer** — it is a standard MLA attention block (Q LoRA, single KV
head, output LoRA) with a sparse top-k sliding window over the sequence.

Key parameters (from DSV4Pro ``config.json``)
---------------------------------------------
hidden_size : int                – model hidden dimension
q_lora_rank : int                – LoRA rank for Q projection (down then up)
o_lora_rank : int                – LoRA rank for output projection
o_groups : int                   – number of output projection groups
num_heads : int                  – number of query heads
num_key_value_heads : int        – number of KV heads (1 = single shared KV)
head_dim : int                   – dimension per head (nope + rope split)
qk_rope_head_dim : int           – RoPE sub-dim within head_dim
sliding_window : int             – sparse top-k window per query
"""

from __future__ import annotations

from typing import Any

from ....core.base_module import BaseModule, ModuleParamBreakdown
from ....core.dtype import DType


class SWA(BaseModule):
    module_type = "SWA"

    def __init__(self, name: str, config: dict[str, Any]):
        super().__init__(name, config)
        self.hidden_size: int = config["hidden_size"]
        self.q_lora_rank: int = config.get("q_lora_rank", 0)
        self.o_lora_rank: int = config.get("o_lora_rank", 0)
        self.o_groups: int = config.get("o_groups", 1)
        self.num_heads: int = config["num_heads"]
        self.num_kv_heads: int = config.get("num_key_value_heads", 1)
        self.head_dim: int = config.get("head_dim", 128)
        self.qk_rope_head_dim: int = config.get("qk_rope_head_dim", 64)
        self.qk_nope_head_dim: int = self.head_dim - self.qk_rope_head_dim
        self.v_head_dim: int = config.get("v_head_dim", self.head_dim)
        self.sliding_window: int = config.get("sliding_window", 128)
        # Compressed head dim for QK^T (per Excel: index_head_dim=128)
        self.index_head_dim: int = config.get("index_head_dim", 128)
        self.dtype: DType = DType(config.get("dtype", "bf16"))

    # -----------------------------------------------------------------
    # Parameter breakdown  (MLA plumbing: same as CSA but no Compressor)
    # -----------------------------------------------------------------

    def compute_params(self) -> list[ModuleParamBreakdown]:
        params: list[ModuleParamBreakdown] = []
        H = self.hidden_size
        n_h = self.num_heads
        n_kv = self.num_kv_heads
        d_h = self.head_dim
        v_hd = self.v_head_dim
        q_lora = self.q_lora_rank
        o_lora = self.o_lora_rank

        # --- Q projection (LoRA: down H->q_lora, up q_lora->n_h*d_h) ---
        if q_lora > 0:
            params.append(ModuleParamBreakdown(
                "q_down_proj", (H, q_lora),
                H * q_lora, self.dtype))
            params.append(ModuleParamBreakdown(
                "q_up_proj", (q_lora, n_h * d_h),
                q_lora * n_h * d_h, self.dtype))
        else:
            params.append(ModuleParamBreakdown(
                "q_proj", (H, n_h * d_h),
                H * n_h * d_h, self.dtype))

        # --- KV projection (single shared KV head) ---
        params.append(ModuleParamBreakdown(
            "k_proj", (H, n_kv * d_h),
            H * n_kv * d_h, self.dtype))
        params.append(ModuleParamBreakdown(
            "v_proj", (H, n_kv * v_hd),
            H * n_kv * v_hd, self.dtype))

        # --- Output projection (LoRA: down n_h*v_hd->o_lora, up o_lora->H) ---
        # wo_a: [o_groups, head_dim*num_heads/o_groups, o_lora_rank]
        #        = [16, 512*128/16, 1024] = [16, 4096, 1024]
        # wo_b: [o_groups*o_lora_rank, hidden_size] = [16384, 7168]
        params.append(ModuleParamBreakdown(
            "wo_a", (self.o_groups, (d_h * n_h) // self.o_groups, o_lora),
            self.o_groups * ((d_h * n_h) // self.o_groups) * o_lora,
            self.dtype))
        params.append(ModuleParamBreakdown(
            "wo_b", (self.o_groups * o_lora, H),
            self.o_groups * o_lora * H, self.dtype))

        # --- RMSNorm (input + post-attention) ---
        params.append(ModuleParamBreakdown(
            "input_layernorm", (H,), H, self.dtype))
        params.append(ModuleParamBreakdown(
            "post_attention_layernorm", (H,), H, self.dtype))

        return params

    # -----------------------------------------------------------------
    # FLOPs
    # -----------------------------------------------------------------

    def compute_flops(self, seq_len: int, batch_size: int = 1) -> float:
        """FLOPs matching Excel ``O18 = SUM(O10:O17) + P15 + P16``.

        SWA is a plain sliding-window attention layer with **no Compressor
        and no Indexer**.  Its attention uses the full sequence length S
        (per Excel P15 = P16 = 2·B·S·S·B26·B19 = 2·B·S·S·n_h·index_head_dim).
        """
        H = self.hidden_size
        n_h = self.num_heads
        v_hd = self.v_head_dim
        q_lora = self.q_lora_rank
        o_lora = self.o_lora_rank
        o_groups = self.o_groups
        idx_h_d = self.index_head_dim  # Excel B19 = 128
        B, S = batch_size, seq_len

        flops = 0.0
        # O10 = B*S*B33*B18*2          (W_q_a down: H -> q_lora)
        flops += 2 * B * S * q_lora * H
        # O11 = B*S*B33*2              (RMSNorm over q_lora)
        flops += 2 * B * S * q_lora
        # O12 = B*S*B26*B17*B33*2      (W_q_b up: q_lora -> n_h*v_hd)
        flops += 2 * B * S * n_h * v_hd * q_lora
        # O13 = B*S*B26*B17*2          (RMSNorm over n_h*v_hd)
        flops += 2 * B * S * n_h * v_hd
        # O14 = B*S*B17*B18*2          (W_kv: H -> v_hd, single shared KV)
        flops += 2 * B * S * v_hd * H
        # O15 = B*S*B17*2              (RMSNorm over v_hd)
        flops += 2 * B * S * v_hd
        # P15 = 2*B*S*S*B26*B19        (attention QK^T, full S x S)
        flops += 2 * B * S * S * n_h * idx_h_d
        # O16 = B*S*B32*B17*2*B26      (wo_a: n_h*v_hd -> o_lora*o_groups... )
        flops += 2 * B * S * o_lora * v_hd * n_h
        # P16 = 2*B*S*S*B26*B19        (attention × V)
        flops += 2 * B * S * S * n_h * idx_h_d
        # O17 = 2*B*S*B18*B32*B31      (wo_b: o_groups*o_lora -> H)
        flops += 2 * B * S * H * o_lora * o_groups
        return flops

    # -----------------------------------------------------------------
    # Activations
    # -----------------------------------------------------------------

    def compute_activations(self, seq_len: int, batch_size: int = 1) -> float:
        """Activation memory matching Excel ``L18 = SUM(L10:L17)``.

        Excel counts the *intermediate vectors* stored for backward (per
        projection input + shared-norm output), in bf16 bytes.  It does
        **not** include the Q/K/V buffers or attention-weight matrices.
        """
        from ....core.dtype import get_dtype_size
        B, S = batch_size, seq_len
        H = self.hidden_size
        n_h = self.num_heads
        v_hd = self.v_head_dim
        q_lora = self.q_lora_rank
        o_lora = self.o_lora_rank
        o_groups = self.o_groups
        elem = get_dtype_size(self.dtype)

        act = 0.0
        act += B * S * q_lora * elem            # L10  W_q_a input
        act += B * S * q_lora * elem            # L11  RMSNorm out
        act += B * S * n_h * v_hd * elem        # L12  W_q_b input
        act += B * S * n_h * v_hd * elem        # L13  RMSNorm out
        act += B * S * v_hd * elem              # L14  W_kv
        act += B * S * v_hd * elem              # L15  RMSNorm out
        act += B * S * o_lora * o_groups * elem # L16  wo_a
        act += B * S * H * elem                 # L17  wo_b
        return act