"""Compressor — KV compression block used inside CSA/HCA attention layers.

In DSV4Pro, attention layers further compress the KV stream with a small
"Compressor" MLP that operates *per-block* (block size = the layer's
``compress_ratio``).  Per the Excel reference table, the Compressor has three
weight matrices of shape :math:`[H, D_c]` where:

- :math:`H`  = ``hidden_size``        (7168 in DSV4Pro 1.6T)
- :math:`D_c` = ``compressor_dim``    (1024 in DSV4Pro 1.6T)

The three matrices are:

- ``W_kv``   – KV compression projection
- ``W_gate`` – gating projection (sigmoid gate)
- ``ape``    – absolute positional encoding bias table

Additional small weights:

- ``W_norm`` – RMSNorm weight of size :math:`D_c` (one per Compressor)
- ``bias``   – optional bias vector of size :math:`D_c`

This module is *passive* in the FLOPs sense — its job is to score and rank KV
blocks so the parent attention (CSA/HCA) can do top-k block selection.  We
therefore assign it only the small **scoring** FLOPs cost:

.. math::
   \\text{FLOPs}_\\text{Compressor} = 2 \\cdot B \\cdot N_b \\cdot (3 H D_c + 2 D_c)

where :math:`N_b = \\lceil S / r \\rceil` is the number of blocks.
"""

from __future__ import annotations

from typing import Any

from ....core.base_module import BaseModule, ModuleParamBreakdown
from ....core.dtype import DType


class Compressor(BaseModule):
    """KV-block Compressor sub-block (used inside CSA / HCA)."""

    module_type = "Compressor"

    def __init__(self, name: str, config: dict[str, Any]):
        super().__init__(name, config)
        self.hidden_size: int = config["hidden_size"]
        # Compressor inner dim — 1024 in DSV4Pro (per Excel)
        self.compressor_dim: int = config.get("compressor_dim", 1024)
        # Block size this Compressor operates on (4 for CSA, 128 for HCA)
        self.compress_ratio: int = config.get("compress_ratio", 4)
        self.dtype: DType = DType(config.get("dtype", "bf16"))
        # Optional APE table size (number of positional buckets); 0 disables
        self.ape_buckets: int = config.get("ape_buckets", 0)

    # -----------------------------------------------------------------
    # Parameters
    # -----------------------------------------------------------------

    def compute_params(self) -> list[ModuleParamBreakdown]:
        H = self.hidden_size
        D_c = self.compressor_dim
        r = self.compress_ratio
        # Per Excel (rows 27/28 CSA Compressor-4, row 42 HCA Compressor-128):
        #   W_kv   : [H, head_dim*2]            -> H * D_c
        #   W_gate : [H, head_dim*2]            -> H * D_c
        #   ape    : [compress_ratio, head_dim*2] -> r * D_c
        params: list[ModuleParamBreakdown] = [
            ModuleParamBreakdown("W_kv", (H, D_c), H * D_c, self.dtype),
            ModuleParamBreakdown("W_gate", (H, D_c), H * D_c, self.dtype),
            ModuleParamBreakdown("ape", (r, D_c), r * D_c, self.dtype),
        ]
        if self.ape_buckets > 0:
            params.append(ModuleParamBreakdown(
                "ape_table", (self.ape_buckets, D_c),
                self.ape_buckets * D_c, self.dtype))
        return params

    # -----------------------------------------------------------------
    # FLOPs  (block-level scoring, very small)
    # -----------------------------------------------------------------

    def compute_flops(self, seq_len: int, batch_size: int = 1) -> float:
        H = self.hidden_size
        D_c = self.compressor_dim
        r = max(self.compress_ratio, 1)
        B = batch_size
        n_blocks = max(1, (seq_len + r - 1) // r)
        # 3 projections (W_kv, W_gate, ape) + gate multiply + softmax over blocks
        proj_flops = 2 * B * n_blocks * (3 * H * D_c)
        gate_flops = 2 * B * n_blocks * D_c
        softmax_flops = 3 * B * n_blocks
        return proj_flops + gate_flops + softmax_flops

    # -----------------------------------------------------------------
    # Activations
    # -----------------------------------------------------------------

    def compute_activations(self, seq_len: int, batch_size: int = 1) -> float:
        from ....core.dtype import get_dtype_size
        H = self.hidden_size
        D_c = self.compressor_dim
        r = max(self.compress_ratio, 1)
        B = batch_size
        n_blocks = max(1, (seq_len + r - 1) // r)
        elem = get_dtype_size(self.dtype)
        # Compressed KV representation: B × n_blocks × D_c
        return B * n_blocks * D_c * elem