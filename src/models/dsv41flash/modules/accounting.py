"""Logical tensor inventory and explicit training-state accounting.

This module deliberately does not inherit Pro's Excel-specific FP4 byte rule.
Quantization scales, allocation padding, and distributed replicas are not logical
model parameters. Training states are sized from element counts, never from
compressed weight bytes.
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
    """Core parameter record extended with its training-state group."""

    group: str = "dense"


def training_memory(parameters: list[GroupedParamBreakdown], training: dict) -> dict:
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
    buffer_items = {k: v * 1e9 for k, v in training["per_device_buffers_gb"].items()}
    return {
        **totals, "total": sum(totals.values()), "optimizer": "per_group",
        "groups": by_group, "per_device": per_device,
        "per_device_gpu_buffers_bytes": buffer_items,
        "per_device_gpu_static_with_buffers_bytes": per_device["gpu"]["total"] + sum(buffer_items.values()),
        "note": "One logical model copy; per-device numbers assume balanced sharding. "
                "Excludes activation peak, recompute, allocator/quantization metadata and communication overlap. "
                "CPU placement applies only to the configured parameter group, not its GPU computations.",
    }
