"""Hunyuan Hy4 model family package.

``config.json`` pins the official Hy4-preview HF config — the structure
shared by all size variants (770B baseline: E=256, Top-8).  Variant
scenarios (e.g. ``configs/hy4_10t_config.json``: E=2048, Top-64, I=3456,
10.074T structural params / 342.6B activated per token) override it,
per the reference workbook ``~/hunyuan/Hy4-preview显存占用分析.xlsx``
(sheets: Hy4-10T / 逐层拓扑 / 缓存与口径).  Analysis scope: backbone
only — the extra MTP layer and sharding are excluded, mirroring the
workbook's stated scope.

Sources: official config at
https://huggingface.co/tencent/Hy4-preview/blob/705d81ee51566a186d645b74c974d642ef2828fe/config.json
and reference implementation at
https://github.com/huggingface/transformers/blob/7cd73d9df0c14b151c684b708a9f27d8d0349dfe/src/transformers/models/hy_v4/modeling_hy_v4.py
"""

from ...core.model_registry import ModelRegistry
from .modules.model import Hy4Model

ModelRegistry.register("Hy4")(Hy4Model)

__all__ = ["Hy4Model"]
