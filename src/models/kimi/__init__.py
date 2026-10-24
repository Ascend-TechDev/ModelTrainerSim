"""Kimi 10T model family package.

``config.json`` pins the structure shared by the workbook's 10T variants
(hidden/attention/MLA dims, vocab, dense intermediate, E=2048, 2 shared
experts, byte assumptions) derived from ``~/kimi/10T Kimi 显存建模.xlsx``.
Variant scenarios supply the size-specific fields:
``configs/kimi_10t_deep_v2_config.json`` (137 layers: 1 Dense FFN + 34 MLA
+ 102 KDA + 136 Latent-MoE, Top-16) and ``configs/kimi_10t_wide_v2_config.json``
(93 layers, Top-32, wider experts).  There is no official HF baseline to
pin, so the common subset is our construction and the pinned file alone is
not a complete model.

Workbook alignment (sheet 10T-深度-v2 / 10T-宽度-v2):
  * Param totals follow the I-column formula chain
    I55 = I2+I3+I4+I13*D25+I18*D5+I32*D6+I51*D24+I34+I35 with
    I13 = SUM(I6:I12)+I71 (v2 adds the MLA gated-linear g_proj).
    Deep-v2 = 9,283,004,226,880; Wide-v2 = 11,917,686,210,656.
    (The workbook's cached J55 on 深度-v2 predates the g_proj 补档 and
    is stale — the formula is authoritative.)
  * FLOPs follow the R-column GEMM formulas (per mbs token).
  * Activation tensors follow the N-column element counts (x 2B; logits 4B)
    with the M30/M32/M33 aggregation conventions.
  * Training states: AdamW 12 bytes/param and Muon 8 bytes/param
    (rows C27/C28); gradients 2 bytes.  Sharding per rows G63/G68/K66:
    dense modules by TP, routed experts by EP, optimizer states
    effectively divided by the total card count.
"""
from ...core.model_registry import ModelRegistry
from .modules.model import KimiModel

ModelRegistry.register("Kimi-10T")(KimiModel)

__all__ = ["KimiModel"]
