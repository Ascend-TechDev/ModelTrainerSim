"""Training-state and sharding accounting for Kimi 10T (workbook 口径).

Optimizer bytes per parameter follow workbook rows C27/C28:
  * AdamW group (embedding + LM head)     12 bytes/param
  * Muon group  (everything else)          8 bytes/param
Gradients 2 bytes (bf16); master weights 0.

Sharding follows workbook rows G63 / G68 / K66:
  * Dense modules shard by TP (row G68: "计算单个 module 时候，静态显存
    做了 TP 处理").
  * Routed experts shard by EP (row G68: "计算单层 MoE 时候，路由专家
    静态显存做了 EP 处理").  Shared experts / latent mapping / gate are
    placed with the dense modules (÷TP) — workbook leaves this implicit.
  * Optimizer states divide by the total card count (row K66: "用已经被
    TP、EP、PP 划分的静态参数去除以 DP，本质上就是直接除以总卡数").
  * The mesh identity tp*cp*dp*pp = etp*ep*edp*pp (row G63) leaves a
    residual expert-data-parallel dim edp = world/(tp*ep*pp*cp).
"""
from __future__ import annotations

from dataclasses import dataclass

from ....core.base_module import ModuleParamBreakdown

DTYPE_BYTES = {"fp32": 4.0, "bf16": 2.0, "fp16": 2.0, "fp8": 1.0, "none": 0.0}


@dataclass
class GroupedParamBreakdown(ModuleParamBreakdown):
    """Parameter record extended with optimizer group and shard dimension.

    ``group``: optimizer group ("adamw" | "muon").
    ``shard``: which parallel dimension shards this tensor for the
    per-rank static-memory count ("tp" | "ep").
    """

    group: str = "muon"
    shard: str = "tp"


def training_memory(parameters: list[GroupedParamBreakdown], training: dict,
                    parallelism: dict, bytes_cfg: dict) -> dict:
    """Whole-model training states plus per-rank sharded static memory."""
    totals = {"weights": 0.0, "gradients": 0.0, "master_weights": 0.0,
              "optimizer_states": 0.0}
    by_group = {}
    shard_counts = {"tp": 0, "ep": 0}
    shard_bytes = {"tp": 0.0, "ep": 0.0}
    for name, group in training["groups"].items():
        params = [p for p in parameters if p.group == name]
        count = sum(p.num_elements for p in params)
        opt_bpp = group["optimizer_bytes_per_param"]
        amounts = {
            "weights": count * DTYPE_BYTES[group["weight_dtype"]],
            "gradients": count * DTYPE_BYTES[group["gradient_dtype"]],
            "master_weights": count * DTYPE_BYTES[group["master_dtype"]],
            "optimizer_states": count * opt_bpp,
        }
        by_group[name] = dict(amounts, params=count, optimizer=group["optimizer"],
                              optimizer_bytes_per_param=opt_bpp, total=sum(amounts.values()))
        for key, amount in amounts.items():
            totals[key] += amount
        for p in params:
            shard_counts[p.shard] += p.num_elements
            shard_bytes[p.shard] += p.num_elements * DTYPE_BYTES[group["weight_dtype"]]
    tp, ep, world = parallelism["tp"], parallelism["ep"], parallelism["world_size"]
    # Per-rank weights/gradients: dense modules ÷TP, routed experts ÷EP.
    weights_rank = shard_bytes["tp"] / tp + shard_bytes["ep"] / ep
    gradients_rank = (shard_counts["tp"] / tp + shard_counts["ep"] / ep) * bytes_cfg["grad"]
    optimizer_rank = totals["optimizer_states"] / world          # row K66
    per_rank = {
        "weights": weights_rank,
        "gradients": gradients_rank,
        "master_weights": totals["master_weights"] / world,
        "optimizer_states": optimizer_rank,
        "total": weights_rank + gradients_rank + totals["master_weights"] / world + optimizer_rank,
        "note": "Weights/gradients: dense modules ÷TP, routed experts ÷EP (row G68). "
                "Optimizer states ÷ total cards (row K66). Shared experts, latent mapping "
                "and gate are placed with the dense modules (÷TP).",
    }
    return {
        **totals, "total": sum(totals.values()), "optimizer": "per_group",
        "groups": by_group, "per_rank": per_rank,
        "note": "Whole-model training states; per_rank carries the sharded static "
                "memory for one card under the workbook mesh (rows G63/G68/K66). "
                "Activation memory and all-gather traffic are reported separately.",
    }
