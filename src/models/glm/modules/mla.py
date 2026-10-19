"""MLA — Multi-head Latent Attention (GLM-5.2 style).

GLM-5.2 uses DeepSeek-style MLA with low-rank Q/KV projections and
decoupled RoPE.  Per the Excel reference (sheet 'GLM-5.2-743B', rows
10-16), one MLA block per layer contains:

  q_a_proj          : [hidden_size, q_lora_rank]
  q_a_layernorm     : [q_lora_rank]                       (RMSNorm)
  q_b_proj          : [q_lora_rank, num_heads*qk_head_dim]
  kv_a_proj_with_mqa: [hidden_size, kv_lora_rank+qk_rope_head_dim]
  kv_a_layernorm    : [kv_lora_rank]                      (RMSNorm)
  kv_b_proj         : [kv_lora_rank, kv_heads*(qk_nope_head_dim+v_head_dim)]

Excel L16 (params for one layer) = SUM(L10:L15) * num_layers.
Excel M16 (weight GB)            = L16 * 2 / 1024^3.
Excel N16 (static mem GB)        = M16 * D27 (static multiplier = 8).

Key parameters (from GLM ``config.json``)
--------------------------------------------
hidden_size : int                 – 6144
q_lora_rank : int                 – 2048
kv_lora_rank: int                 – 512
num_heads : int                   – 64
num_key_value_heads : int         – 64
qk_head_dim : int                 – 256 (= nope 192 + rope 64)
qk_nope_head_dim : int            – 192
qk_rope_head_dim : int            – 64
v_head_dim : int                  – 256
"""

from __future__ import annotations

from typing import Any

from ....core.base_module import BaseModule, ModuleParamBreakdown
from ....core.dtype import DType, get_dtype_size


class MLA(BaseModule):
    """Multi-head Latent Attention block (GLM-5.2)."""

    module_type = "MLA"

    def __init__(self, name: str, config: dict[str, Any]):
        super().__init__(name, config)
        self.hidden_size: int = config["hidden_size"]
        self.q_lora_rank: int = config.get("q_lora_rank", 0)
        self.kv_lora_rank: int = config.get("kv_lora_rank", 0)
        self.num_heads: int = config.get("num_heads", 1)
        self.num_kv_heads: int = config.get("num_key_value_heads", 1)
        self.qk_head_dim: int = config.get("qk_head_dim", 256)
        self.qk_nope_head_dim: int = config.get("qk_nope_head_dim", 192)
        self.qk_rope_head_dim: int = config.get("qk_rope_head_dim", 64)
        self.v_head_dim: int = config.get("v_head_dim", 256)
        self.dtype: DType = DType(config.get("dtype", "bf16"))

    # -----------------------------------------------------------------
    # Parameters — Excel rows 10-15
    # -----------------------------------------------------------------

    def compute_params(self) -> list[ModuleParamBreakdown]:
        params: list[ModuleParamBreakdown] = []
        H = self.hidden_size
        q_lora = self.q_lora_rank
        kv_lora = self.kv_lora_rank
        n_h = self.num_heads
        n_kv = self.num_kv_heads
        d_qk = self.qk_head_dim
        d_nope = self.qk_nope_head_dim
        d_rope = self.qk_rope_head_dim
        v_hd = self.v_head_dim

        # L10: q_a_proj = H * q_lora
        params.append(ModuleParamBreakdown(
            "q_a_proj", (H, q_lora), H * q_lora, self.dtype))
        # L11: q_a_layernorm = q_lora
        params.append(ModuleParamBreakdown(
            "q_a_layernorm", (q_lora,), q_lora, self.dtype))
        # L12: q_b_proj = q_lora * num_heads * qk_head_dim
        params.append(ModuleParamBreakdown(
            "q_b_proj", (q_lora, n_h * d_qk),
            q_lora * n_h * d_qk, self.dtype))
        # L13: kv_a_proj_with_mqa = H * (kv_lora + qk_rope)
        params.append(ModuleParamBreakdown(
            "kv_a_proj_with_mqa", (H, kv_lora + d_rope),
            H * (kv_lora + d_rope), self.dtype))
        # L14: kv_a_layernorm = kv_lora
        params.append(ModuleParamBreakdown(
            "kv_a_layernorm", (kv_lora,), kv_lora, self.dtype))
        # L15: kv_b_proj = kv_lora * num_kv_heads * (qk_nope + v_head_dim)
        params.append(ModuleParamBreakdown(
            "kv_b_proj", (kv_lora, n_kv * (d_nope + v_hd)),
            kv_lora * n_kv * (d_nope + v_hd), self.dtype))

        return params

    # -----------------------------------------------------------------
    # FLOPs — Excel aligns with the DSA-style attention compute
    #         The Excel sheet itself does not provide explicit FLOPs
    #         cells for MLA, so we derive them from the projection
    #         matrices: each matmul (M, N) * (N, K) = 2·M·N·K FLOPs.
    # -----------------------------------------------------------------

    def compute_flops(self, seq_len: int, batch_size: int = 1) -> float:
        H = self.hidden_size
        q_lora = self.q_lora_rank
        kv_lora = self.kv_lora_rank
        n_h = self.num_heads
        n_kv = self.num_kv_heads
        d_qk = self.qk_head_dim
        d_nope = self.qk_nope_head_dim
        d_rope = self.qk_rope_head_dim
        v_hd = self.v_head_dim
        B, S = batch_size, seq_len

        flops = 0.0
        # q_a_proj: [B,S,H] * [H, q_lora]
        flops += 2 * B * S * H * q_lora
        # q_a_layernorm: RMSNorm over q_lora  (~2 * B*S*q_lora)
        flops += 2 * B * S * q_lora
        # q_b_proj: [B,S,q_lora] * [q_lora, n_h*d_qk]
        flops += 2 * B * S * q_lora * n_h * d_qk
        # kv_a_proj_with_mqa: [B,S,H] * [H, kv_lora+d_rope]
        flops += 2 * B * S * H * (kv_lora + d_rope)
        # kv_a_layernorm: RMSNorm over kv_lora
        flops += 2 * B * S * kv_lora
        # kv_b_proj: [B,S,kv_lora] * [kv_lora, n_kv*(d_nope+v_hd)]
        flops += 2 * B * S * kv_lora * n_kv * (d_nope + v_hd)
        # attention QK^T + softmax + attn·V  (full S x S per head)
        # QK^T: 2 * B * n_h * S * S * d_qk
        flops += 2 * B * n_h * S * S * d_qk
        # attn · V: 2 * B * n_h * S * S * v_hd
        flops += 2 * B * n_h * S * S * v_hd
        return flops

    # -----------------------------------------------------------------
    # Activations — Excel rows 10-15 (column U = S * T * count)
    #   Each projection's activation = B*S*<proj_output_dim> * 2 bytes
    # -----------------------------------------------------------------

    def compute_activations(self, seq_len: int, batch_size: int = 1) -> float:
        B, S = batch_size, seq_len
        H = self.hidden_size
        q_lora = self.q_lora_rank
        kv_lora = self.kv_lora_rank
        d_rope = self.qk_rope_head_dim
        n_h = self.num_heads
        d_qk = self.qk_head_dim
        n_kv = self.num_kv_heads
        d_nope = self.qk_nope_head_dim
        v_hd = self.v_head_dim
        elem = get_dtype_size(self.dtype)

        act = 0.0
        act += B * S * q_lora * elem                  # U10
        act += B * S * q_lora * elem                  # U11
        act += B * S * n_h * d_qk * elem              # U12
        act += B * S * (kv_lora + d_rope) * elem      # U13
        act += B * S * kv_lora * elem                 # U14
        act += B * S * n_kv * (d_nope + v_hd) * elem  # U15
        return act