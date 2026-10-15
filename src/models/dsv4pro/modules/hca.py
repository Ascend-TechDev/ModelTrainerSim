"""HCA — Hierarchical Compressed Attention.

Used in DSV4Pro for layers with ``compress_ratio = 128``.  HCA builds a
two-level hierarchy:

1. **Anchor / block level** – tokens are grouped into blocks of ``block_size``
   (= ``compress_ratio``); a single representative query per block attends to
   all block-level summaries (coarse attention).
2. **Token level** – within selected top blocks, individual tokens attend to
   all tokens in those blocks (fine attention).

Architecturally, the per-head MLA plumbing (Q LoRA, single KV head, output
LoRA) is the same as CSA; only the attention pattern differs.

Key parameters (from DSV4Pro ``config.json``)
---------------------------------------------
hidden_size : int
q_lora_rank : int                – Q projection LoRA rank
o_lora_rank : int                – output projection LoRA rank
o_groups : int
num_heads : int
num_key_value_heads : int        – 1 (single shared KV head)
head_dim : int                   – 512
qk_rope_head_dim : int           – 64
compress_ratio : int             – block size for hierarchy (=128)
index_topk : int                 – number of blocks selected per query
"""

from __future__ import annotations

from typing import Any

from ....core.base_module import BaseModule, ModuleParamBreakdown
from ....core.dtype import DType
from .compressor import Compressor


class HCA(BaseModule):
    module_type = "HCA"

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
        # HCA-specific: block size = compress_ratio, top-k blocks = index_topk
        self.block_size: int = config.get("compress_ratio",
                                          config.get("block_size", 128))
        self.topk_blocks: int = config.get("index_topk",
                                           config.get("topk_blocks", 16))
        # Compressed head dim for QK^T (per Excel: index_head_dim=128)
        self.index_head_dim: int = config.get("index_head_dim", 128)
        self.dtype: DType = DType(config.get("dtype", "bf16"))

        # --- Compressor-128 (per Excel: every HCA layer has a Compressor) ---
        comp_cfg = dict(config.get("compressor", {}))
        comp_cfg.setdefault("hidden_size", self.hidden_size)
        comp_cfg.setdefault("compressor_dim", 1024)
        comp_cfg.setdefault("compress_ratio", self.block_size)
        comp_cfg.setdefault("dtype", self.dtype.value)
        self.compressor: Compressor = Compressor(f"{name}_Compressor", comp_cfg)

    # -----------------------------------------------------------------
    # Parameters  (identical MLA plumbing to CSA)
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

        # Q projection (LoRA)
        if q_lora > 0:
            params.append(ModuleParamBreakdown(
                "q_down_proj", (H, q_lora), H * q_lora, self.dtype))
            params.append(ModuleParamBreakdown(
                "q_up_proj", (q_lora, n_h * d_h),
                q_lora * n_h * d_h, self.dtype))
        else:
            params.append(ModuleParamBreakdown(
                "q_proj", (H, n_h * d_h), H * n_h * d_h, self.dtype))

        # KV projection (single shared KV head)
        params.append(ModuleParamBreakdown(
            "k_proj", (H, n_kv * d_h), H * n_kv * d_h, self.dtype))
        params.append(ModuleParamBreakdown(
            "v_proj", (H, n_kv * v_hd), H * n_kv * v_hd, self.dtype))

        # Output projection (LoRA)
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
                "o_proj", (n_h * v_hd, H), n_h * v_hd * H, self.dtype))

        # Norms
        params.append(ModuleParamBreakdown(
            "input_layernorm", (H,), H, self.dtype))
        params.append(ModuleParamBreakdown(
            "post_attention_layernorm", (H,), H, self.dtype))

        # Compressor-128
        params.extend(self.compressor.compute_params())

        return params

    # -----------------------------------------------------------------
    # FLOPs
    # -----------------------------------------------------------------

    def compute_flops(self, seq_len: int, batch_size: int = 1) -> float:
        """FLOPs matching Excel ``O45 = SUM(O36:O44) + P43 + P44``.

        HCA attention uses an effective KV length of ``(S/128 + 128)``:
        ``S/128`` compressed blocks + 128 sink tokens.
        """
        H = self.hidden_size
        n_h = self.num_heads
        v_hd = self.v_head_dim
        q_lora = self.q_lora_rank
        o_lora = self.o_lora_rank
        o_groups = self.o_groups
        idx_h_d = self.index_head_dim  # Excel B19 = 128
        B, S = batch_size, seq_len
        eff_seq = (S / 128) + 128

        flops = 0.0
        # O36 = B*S*B33*B18*2          (W_q_a down: H -> q_lora)
        flops += 2 * B * S * q_lora * H
        # O37 = B*S*B33*2              (RMSNorm over q_lora)
        flops += 2 * B * S * q_lora
        # O38 = B*S*B26*B17*B33*2      (W_q_b up: q_lora -> n_h*v_hd)
        flops += 2 * B * S * n_h * v_hd * q_lora
        # O39 = B*S*B26*B17*2          (RMSNorm over n_h*v_hd)
        flops += 2 * B * S * n_h * v_hd
        # O40 = B*S*B17*B18*2          (W_kv: H -> v_hd, single shared KV)
        flops += 2 * B * S * v_hd * H
        # O41 = B*S*B17*2              (RMSNorm over v_hd)
        flops += 2 * B * S * v_hd
        # O42 = O52 = O49 + O50 + P51  (Compressor-128 scoring)
        flops += 4 * B * S * H * v_hd        # O49
        flops += 4 * B * S * H * v_hd        # O50
        flops += 4 * S * B * v_hd            # P51
        # O43 = B*S*B32*B17*2*B26      (wo_a: n_h*v_hd -> o_lora*o_groups...)
        flops += 2 * B * S * o_lora * v_hd * n_h
        # P43 = 2*B*(S/128+128)*S*B26*B19   (attention QK^T)
        flops += 2 * B * eff_seq * S * n_h * idx_h_d
        # O44 = 2*B*S*B18*B32*B31      (wo_b: o_groups*o_lora -> H)
        flops += 2 * B * S * H * o_lora * o_groups
        # P44 = 2*B*(S/128+128)*S*B26*B19   (attention × V)
        flops += 2 * B * eff_seq * S * n_h * idx_h_d
        return flops

    # -----------------------------------------------------------------
    # Activations
    # -----------------------------------------------------------------

    def compute_activations(self, seq_len: int, batch_size: int = 1) -> float:
        """Activation memory matching Excel ``L45 = SUM(L36:L44)``.

        Excel counts the per-projection intermediate vectors (input + shared
        RMSNorm output) in bf16 bytes — **not** the Q/K/V buffers nor the
        coarse/fine attention-weight matrices.  For HCA this is the same
        8-line attention pattern (L36-L41, L43-L44) plus a *Compressor-128*
        block (L42=L52=SUM(L49:L51)) whose APE has 128 buckets.
        """
        from ....core.dtype import get_dtype_size
        B, S = batch_size, seq_len
        H = self.hidden_size
        n_h = self.num_heads
        v_hd = self.v_head_dim
        q_lora = self.q_lora_rank
        o_lora = self.o_lora_rank
        o_groups = self.o_groups
        r = self.block_size  # 128 for HCA
        elem = get_dtype_size(self.dtype)

        act = 0.0
        act += B * S * q_lora * elem            # L36  W_q_a input
        act += B * S * q_lora * elem            # L37  RMSNorm out
        act += B * S * n_h * v_hd * elem        # L38  W_q_b input
        act += B * S * n_h * v_hd * elem        # L39  RMSNorm out
        act += B * S * v_hd * elem              # L40  W_kv
        act += B * S * v_hd * elem              # L41  RMSNorm out
        # L42 = L52 = L49 + L50 + L51  (Compressor-128, D_c = head_dim*2)
        act += B * S * v_hd * 2 * elem          # L49  W_kv
        act += B * S * v_hd * 2 * elem          # L50  W_gate
        act += r * v_hd * 2 * elem              # L51  ape
        act += B * S * o_lora * o_groups * elem # L43  wo_a
        act += B * S * H * elem                 # L44  wo_b
        return act