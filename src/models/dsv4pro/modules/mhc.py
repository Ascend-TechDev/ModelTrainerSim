"""mHC — mini Hierarchical Compression residual block.

In DSV4Pro, **mHC is a tiny mini-attention block** present in *every*
transformer block (61 layers).  It is *not* a full attention module — it
operates on a very small inner dimension ``mhc_dim`` (= 4 in DSV4Pro 1.6T,
Excel cell B15) and uses a Sinkhorn-normalization matrix of size
``mhc_dim * (mhc_dim + 2)`` (= 24, Excel cell C44 = B15*(B15+2)).

Excel reference (sheet 'DSV4Pro 1.6T', P7 = SUM(P3:P5,R3:R5)):
  P3 = 2 * B * S * B15 * H            (Q projection H -> mhc_dim)
  P4 = 2 * B15 * H * C44              (Sinkhorn weight matrix)
  P5 = 960                            (constant bias / small weights)
  R3 = B * S * B15 * H                (sinkhorn iteration · Q)
  R4 = 2 * B * S * B15^2 * H          (sinkhorn matmul)
  R5 = B * S * B15 * H                (final apply)

The total per-layer FLOPs are tiny (~0.021 GB/tok at S=8192).

Key parameters
--------------
hidden_size : int                 – model hidden dimension (B18)
mhc_dim : int                     – mHC inner dim (B15 = 4 in DSV4Pro)
dtype : DType
"""

from __future__ import annotations

from typing import Any

from ....core.base_module import BaseModule, ModuleParamBreakdown
from ....core.dtype import DType


class mHC(BaseModule):
    """Tiny mini-attention compression-residual block (mHC_pre / mHC_post)."""

    module_type = "mHC"

    def __init__(self, name: str, config: dict[str, Any]):
        super().__init__(name, config)
        self.hidden_size: int = config["hidden_size"]
        # mHC inner dim — Excel B15 = 4 in DSV4Pro 1.6T
        self.mhc_dim: int = int(config.get("mhc_dim", 4))
        # Sinkhorn matrix dim — Excel C44 = B15*(B15+2) = 24
        self.sinkhorn_dim: int = int(config.get(
            "sinkhorn_dim", self.mhc_dim * (self.mhc_dim + 2)))
        # Excel P5 = 960 (constant small weights / bias budget)
        self.const_flops: int = int(config.get("mhc_const_flops", 960))
        self.dtype: DType = DType(config.get("dtype", "bf16"))

    # -----------------------------------------------------------------
    # Parameters
    #   mHC weights (per Excel P3-P5 / R3-R5):
    #     W_q       : H -> mhc_dim           (H * mhc_dim)
    #     W_sinkhorn: mhc_dim*H -> sinkhorn_dim  (mhc_dim * H * sinkhorn_dim)
    #     small bias/const (~ 960 / 2 = 480 params)
    # -----------------------------------------------------------------

    def compute_params(self) -> list[ModuleParamBreakdown]:
        params: list[ModuleParamBreakdown] = []
        H = self.hidden_size
        d = self.mhc_dim
        c = self.sinkhorn_dim
        params.append(ModuleParamBreakdown(
            "mhc_q_proj", (H, d), H * d, self.dtype))
        params.append(ModuleParamBreakdown(
            "mhc_sinkhorn", (d * H, c), d * H * c, self.dtype))
        # constant bias budget (Excel P5 = 960 FLOPs ≈ 480 params × 2)
        params.append(ModuleParamBreakdown(
            "mhc_const", (self.const_flops // 2,), self.const_flops // 2,
            self.dtype))
        return params

    # -----------------------------------------------------------------
    # FLOPs (per Excel P7 = P3 + P4 + P5 + R3 + R4 + R5)
    # -----------------------------------------------------------------

    def compute_flops(self, seq_len: int, batch_size: int = 1) -> float:
        H = self.hidden_size
        d = self.mhc_dim
        c = self.sinkhorn_dim
        B, S = batch_size, seq_len
        flops = 0.0
        flops += 2 * B * S * d * H                  # P3 (Q projection)
        flops += 2 * d * H * c                      # P4 (Sinkhorn weight)
        flops += self.const_flops                   # P5 (constant)
        flops += B * S * d * H                      # R3 (sinkhorn iter · Q)
        flops += 2 * B * S * d * d * H              # R4 (sinkhorn matmul)
        flops += B * S * d * H                      # R5 (final apply)
        return flops

    # -----------------------------------------------------------------
    # Activations
    # -----------------------------------------------------------------

    def compute_activations(self, seq_len: int, batch_size: int = 1) -> float:
        """Excel ``per_module_summary`` does not count mHC activations —
        the mHC is a tiny intra-layer mixer whose intermediates are not
        surfaced separately in the activation summary.  Return 0.
        """
        return 0.0