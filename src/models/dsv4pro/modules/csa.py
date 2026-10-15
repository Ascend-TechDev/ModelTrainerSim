"""CSA — Compressed Sparse Attention.

Used in DSV4Pro's first ``num_hash_layers`` layers (default 3).  CSA is the
"hash-based sparse attention" variant combining:

- MLA-style low-rank Q projection (``q_lora_rank``) and low-rank output
  projection (``o_lora_rank`` with ``o_groups``).
- Single KV head (``num_key_value_heads=1``) with ``head_dim`` per head,
  plus a RoPE sub-dimension (``qk_rope_head_dim``).
- Hash-based sparse top-k selection over tokens (``sliding_window``).

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
from .compressor import Compressor


class CSA(BaseModule):
    module_type = "CSA"

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
        # nope part of head_dim = head_dim - rope (RoPE applied to a sub-slice)
        self.qk_nope_head_dim: int = self.head_dim - self.qk_rope_head_dim
        self.v_head_dim: int = config.get("v_head_dim", self.head_dim)
        self.sliding_window: int = config.get("sliding_window", 128)
        # Compressed head dim used for attention QK^T (per Excel: index_head_dim=128)
        self.index_head_dim: int = config.get("index_head_dim", 128)
        self.dtype: DType = DType(config.get("dtype", "bf16"))

        # --- Compressor-4 (per Excel: every CSA layer has a Compressor) ---
        comp_cfg = dict(config.get("compressor", {}))
        comp_cfg.setdefault("hidden_size", self.hidden_size)
        comp_cfg.setdefault("compressor_dim", 1024)
        comp_cfg.setdefault("compress_ratio", 4)
        comp_cfg.setdefault("dtype", self.dtype.value)
        self.compressor: Compressor = Compressor(f"{name}_Compressor", comp_cfg)

    # -----------------------------------------------------------------
    # Parameter breakdown
    # -----------------------------------------------------------------

    def compute_params(self) -> list[ModuleParamBreakdown]:
        params: list[ModuleParamBreakdown] = []
        H = self.hidden_size
        n_h = self.num_heads
        n_kv = self.num_kv_heads
        d_h = self.head_dim            # full head dim (= nope + rope)
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

        # --- KV projection (single shared W_kv, per Excel G25 = H*v_hd) ---
        # Excel uses ONE fused W_kv of shape [H, head_dim] (no separate K/V,
        # no num_kv_heads multiplier).
        params.append(ModuleParamBreakdown(
            "w_kv_proj", (H, v_hd),
            H * v_hd, self.dtype))

        # --- Output projection (LoRA: down n_h*v_hd->o_lora, up o_lora->H) ---
        # Per Excel (rows 29-30):
        #   wo_a (o_down_proj): [o_groups, head_dim*n_heads/o_groups, o_lora]
        #       shape => params = n_h * d_h * o_lora   (o_groups factor cancels)
        #   wo_b (o_up_proj):   [o_groups * o_lora, hidden_size]
        #       params = o_groups * o_lora * H
        if o_lora > 0:
            params.append(ModuleParamBreakdown(
                "o_down_proj", (self.o_groups,
                                (n_h * v_hd) // self.o_groups, o_lora),
                self.o_groups * ((n_h * v_hd) // self.o_groups) * o_lora,
                self.dtype))
            params.append(ModuleParamBreakdown(
                "o_up_proj", (self.o_groups * o_lora, H),
                self.o_groups * o_lora * H, self.dtype))
        else:
            params.append(ModuleParamBreakdown(
                "o_proj", (n_h * v_hd, H),
                n_h * v_hd * H, self.dtype))

        # --- RMSNorm (input + post-attention) ---
        params.append(ModuleParamBreakdown(
            "input_layernorm", (H,), H, self.dtype))
        params.append(ModuleParamBreakdown(
            "post_attention_layernorm", (H,), H, self.dtype))

        # --- Compressor-4 (appended so it shows up in the layer breakdown) ---
        params.extend(self.compressor.compute_params())

        return params

    # -----------------------------------------------------------------
    # FLOPs
    # -----------------------------------------------------------------

    def compute_flops(self, seq_len: int, batch_size: int = 1) -> float:
        """Compute CSA per-layer FLOPs, matching Excel O31 = SUM(O21:P30).

        Excel reference (sheet 'DSV4Pro 1.6T', B=1, S=8192):
          O21 = 2*B*S*B33*B18         (W_q_a down: H→q_lora)
          O22 = 2*B*S*B33             (RMSNorm over q_lora)
          O23 = 2*B*S*B26*B17*B33     (W_q_b up: q_lora→n_h*d_h)
          O24 = 2*B*S*B26*B17         (RMSNorm over n_h*d_h)
          O25 = 2*B*S*B17*B18         (shared W_kv: H→d_h)
          O26 = 2*B*S*B17             (RMSNorm over d_h)
          O27 = O59 = 12*B*S*B18*B17  (indexer Q/K/V projections: 3× 4·B·S·H·d_h)
          O28 = O74 = O71+O72+O73+P73
              O71 = 2*B*S*B33*B20*B19 (indexer Q proj via q_lora·B20·n_h)
              O72 = 2*B*S*B18*B26     (indexer K proj H→n_h)
              O73 = O65 = 12*B*S*B18*B17  (indexer V proj)
              P73 = 2*S*B*S*B26*B19   (full-S dense attn QK^T)
          P28 = 2*B*(S/4+4)*S*B26*B19 (sparse QK^T with eff_seq=S/4+4)
          O29 = 2*B*S*B32*B17*B26     (W_o_a down)
          P29 = 2*B*(S/4+4)*S*B26*B19 (sparse attn×V with eff_seq=S/4+4)
          O30 = 2*B*S*B18*B32*B31     (W_o_b up)
        """
        H = self.hidden_size               # B18
        n_h = self.num_heads               # B19 = 128
        d_h = self.head_dim                # B17 = 512
        q_lora = self.q_lora_rank          # B33 = 1536
        o_lora = self.o_lora_rank          # B32 = 1024
        o_groups = self.o_groups           # B31 = 16
        # Excel B26 = qk dims = n_h = 128; B20 = indexer head dim = 64
        B26 = n_h
        B20 = 64
        B, S = batch_size, seq_len
        # CSA effective KV length is S/4 + 4 (compressed blocks of size 4 + 4 sink tokens)
        eff_seq = (S / 4) + 4

        flops = 0.0
        # O21 + O22: W_q_a down + RMSNorm over q_lora
        flops += 2 * B * S * q_lora * H           # O21
        flops += 2 * B * S * q_lora               # O22
        # O23 + O24: W_q_b up + RMSNorm over n_h*d_h
        flops += 2 * B * S * B26 * d_h * q_lora   # O23
        flops += 2 * B * S * B26 * d_h            # O24
        # O25 + O26: shared W_kv + RMSNorm over d_h
        flops += 2 * B * S * d_h * H              # O25
        flops += 2 * B * S * d_h                  # O26
        # O27 = O59 = 12*B*S*H*d_h (indexer Q/K/V projections, 3× 4·B·S·H·d_h)
        flops += 12 * B * S * H * d_h             # O27
        # O28 = O74 = O71 + O72 + O73 + P73
        flops += 2 * B * S * q_lora * B20 * n_h   # O71 (indexer Q)
        flops += 2 * B * S * H * B26              # O72 (indexer K)
        flops += 12 * B * S * H * d_h             # O73 = O65 (indexer V)
        flops += 2 * S * B * S * B26 * n_h        # P73 (full-S dense QK^T)
        # P28: sparse QK^T with eff_seq=S/4+4
        flops += 2 * B * eff_seq * S * B26 * n_h  # P28
        # O29: W_o_a down
        flops += 2 * B * S * o_lora * d_h * B26   # O29
        # P29: sparse attn×V with eff_seq=S/4+4
        flops += 2 * B * eff_seq * S * B26 * n_h  # P29
        # O30: W_o_b up
        flops += 2 * B * S * H * o_lora * o_groups  # O30

        return flops

    # -----------------------------------------------------------------
    # Activations
    # -----------------------------------------------------------------

    def compute_activations(self, seq_len: int, batch_size: int = 1) -> float:
        """Activation memory matching Excel ``L31 = SUM(L21:L30)``.

        Excel counts the per-projection intermediate vectors (input + shared
        RMSNorm output) in bf16 bytes — **not** the Q/K/V buffers or the
        sparse attention weights.  For CSA this is the standard 8-line pattern
        (L21-L26, L29-L30) plus a *Compressor-4* block (L27=L59=SUM(L56:L58))
        and an *Indexer* block (L28=L65=SUM(L62:L64)).
        """
        from ....core.dtype import get_dtype_size
        B, S = batch_size, seq_len
        H = self.hidden_size
        n_h = self.num_heads
        v_hd = self.v_head_dim
        q_lora = self.q_lora_rank
        o_lora = self.o_lora_rank
        o_groups = self.o_groups
        r = self.compressor.compress_ratio  # 4 for CSA
        elem = get_dtype_size(self.dtype)

        act = 0.0
        act += B * S * q_lora * elem            # L21  W_q_a input
        act += B * S * q_lora * elem            # L22  RMSNorm out
        act += B * S * n_h * v_hd * elem        # L23  W_q_b input
        act += B * S * n_h * v_hd * elem        # L24  RMSNorm out
        act += B * S * v_hd * elem              # L25  W_kv
        act += B * S * v_hd * elem              # L26  RMSNorm out
        # L27 = L59 = L56 + L57 + L58  (Compressor-4, D_c = head_dim)
        act += B * S * v_hd * elem              # L56  W_kv
        act += B * S * v_hd * elem              # L57  W_gate
        act += r * v_hd * elem                  # L58  ape
        # L28 = L65 = L62 + L63 + L64  (Indexer, same shape ×2)
        act += B * S * v_hd * 2 * elem          # L62  W_kv
        act += B * S * v_hd * 2 * elem          # L63  W_gate
        act += r * v_hd * 2 * elem             # L64  ape
        act += B * S * o_lora * o_groups * elem # L29  wo_a
        act += B * S * H * elem                 # L30  wo_b
        return act