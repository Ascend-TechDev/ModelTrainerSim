"""Explicit training-state accounting for Hy4 (workbook byte assumptions).

The 10T workbook assumes, per parameter element:
  - weights        2 bytes  (BF16-equivalent, ``param_bytes``)
  - gradients      2 bytes  (``grad_bytes``)
  - optimizer      8 bytes  (adamw m+v in FP32, ``optimizer_bytes``)
  - master weights 0 bytes  (``master_weight_bytes`` = 0, to avoid double
                             counting against the 8-byte optimizer state)
The router correction bias (2048 elements per MoE layer, 77 x 2048 = 157,696
elements total) is a checkpoint buffer: it counts toward weights but is
excluded from gradients and optimizer states (workbook D19-D21 / A26).

Follows dsv41flash's ``training_memory`` design: states are sized from
element counts per parameter group, never from compressed weight bytes.
"""
from __future__ import annotations

from dataclasses import dataclass

from ....core.base_module import ModuleParamBreakdown

DTYPE_BYTES = {"fp32": 4.0, "bf16": 2.0, "fp16": 2.0,
               "fp8": 1.0, "fp4": 0.5, "none": 0.0}
OPTIMIZER_STATES = {"adamw": 2, "momentum": 1, "sgd": 0, "none": 0}


def element_bytes(dtype: str) -> float:
    if dtype not in DTYPE_BYTES:
        raise ValueError(f"Unsupported dtype {dtype!r}; choose {list(DTYPE_BYTES)}")
    return DTYPE_BYTES[dtype]


@dataclass
class GroupedParamBreakdown(ModuleParamBreakdown):
    """Core parameter record extended with its training-state group.

    Hy4 groups: ``dense`` (all backbone weights except routed experts),
    ``expert`` (routed experts), ``buffer`` (router correction bias —
    trainable=False by scenario config).
    """

    group: str = "dense"


def training_memory(parameters: list[GroupedParamBreakdown], training: dict) -> dict:
    """Aggregate training states per group; totals mirror workbook H21-H24.

    Returns ``weights`` / ``gradients`` / ``master_weights`` /
    ``optimizer_states`` / ``total`` plus per-group detail.  A group with
    ``trainable=False`` contributes weights only.
    """
    keys = ("weights", "gradients", "master_weights", "optimizer_states")
    by_group = {}
    totals = {k: 0.0 for k in keys}
    per_device = {device: {k: 0.0 for k in keys} for device in ("gpu", "cpu")}
    for name, group in training["groups"].items():
        count = sum(p.num_elements for p in parameters if p.group == name)
        trainable = group["trainable"]
        amounts = {
            "weights": count * element_bytes(group["weight_dtype"]),
            "gradients": count * element_bytes(group["gradient_dtype"]) if trainable else 0.0,
            "master_weights": count * element_bytes(group["master_dtype"]) if trainable else 0.0,
            "optimizer_states": count * element_bytes(group["optimizer_dtype"])
                * OPTIMIZER_STATES[group["optimizer"]] if trainable else 0.0,
        }
        by_group[name] = dict(amounts, params=count, device=group["device"],
                              shards=group["shards"], trainable=trainable,
                              optimizer=group["optimizer"], total=sum(amounts.values()))
        for key, amount in amounts.items():
            totals[key] += amount
            per_device[group["device"]][key] += amount / group["shards"]
    for values in per_device.values():
        values["total"] = sum(values.values())
    return {
        **totals, "total": sum(totals.values()), "optimizer": "per_group",
        "groups": by_group, "per_device": per_device,
        "note": "One logical model copy; per-device numbers assume balanced sharding. "
                "Master weights default to 0 (workbook master_weight_bytes assumption) so the "
                "8 bytes/param optimizer state is not double counted. Excludes activation peak, "
                "recompute, allocator/quantization metadata and communication overlap.",
    }
