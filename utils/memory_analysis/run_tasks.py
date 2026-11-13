#!/usr/bin/env python3
"""EP/FSDP Memory Analysis: DDR & D2H requirements for DSV4Pro.

Follows the methodology in EP_FSDP_MEM_HBM_D2H.md exactly.
"""
import json
import math
from pathlib import Path

# ============================================================================
# Input Data
# ============================================================================
JSON_PATH = "test_output.json"
MODEL_DATA = json.loads(Path(JSON_PATH).read_text(encoding="utf-8"))

# Extract from model data
layers = MODEL_DATA["layers"]
N = len(layers)
total_weight_bytes = MODEL_DATA["memory"]["weight_memory_bytes"]  # BF16
total_act_bytes = sum(l["act_bytes"] for l in layers)
max_layer_act_bytes = max(l["act_bytes"] for l in layers)

# MoE vs Backbone split (from per_module_summary_excel)
excel = MODEL_DATA["per_module_summary_excel"]
moe_bf16 = excel["routedexpert"]["bytes"] + excel["sharedexpert"]["bytes"] + excel["gate"]["bytes"]
backbone_bf16 = total_weight_bytes - moe_bf16

# FP32 versions
moe_fp32 = moe_bf16 * 2
backbone_fp32 = backbone_bf16 * 2

# MoE activation total (per-layer MoE module act_bytes)
moe_act_total = sum(
    sum(m["act_bytes"] for m in l["modules"] if m["type"] == "MoE")
    for l in layers
)

# Per-layer data
per_layer_act = [l["act_bytes"] for l in layers]
per_layer_flops = [l["flops"] for l in layers]
per_layer_ffn = [l["ffn_type"] for l in layers]
per_layer_moe_act = [
    sum(m["act_bytes"] for m in l["modules"] if m["type"] == "MoE")
    for l in layers
]

# ============================================================================
# Hardware Constants
# ============================================================================
BF16_TFLOPS = 1769e12       # BF16 1769 TFLOPS
MFU = 0.95                  # A7 compute share
HBM_CAPACITY_GB = 288       # HBM per GPU
HBM_SAFETY = 0.95
BUFFER_GB = 3.0
GB = 1e9

D2H_BANDWIDTHS_GBS = [200, 250, 300, 350, 400, 450, 500]
D2H_UTILIZATION = 0.8
DDR_CAPACITIES_GB = [256, 384, 512, 540, 768]

# ============================================================================
# Parallel Strategies
# ============================================================================
STRATEGIES = [
    {"name": "策略一", "EP": 128, "EDP": 64, "FSDP": 8192},
    {"name": "策略二", "EP": 256, "EDP": 32, "FSDP": 8192},
    {"name": "策略三", "EP": 512, "EDP": 16, "FSDP": 8192},
]

# ============================================================================
# Per-Layer Time Estimation
# ============================================================================
def compute_layer_times():
    """Time per layer: layer_flops / (hardware_flops * MFU)"""
    times_ms = []
    for flops in per_layer_flops:
        time_s = flops / (BF16_TFLOPS * MFU)
        times_ms.append(time_s * 1000)
    return times_ms

# ============================================================================
# Core Calculations per Strategy
# ============================================================================
def analyze_strategy(ep, fsdp):
    """Run all calculations for one (EP, FSDP) combination."""

    # ---- Static HBM ----
    moe_per_card = moe_bf16 / ep * 2          # params + gradients
    backbone_per_card = backbone_bf16 / fsdp * 2
    static_hbm = max_layer_act_bytes + moe_per_card + backbone_per_card + BUFFER_GB * GB

    # ---- HBM Remaining ----
    hbm_budget = HBM_CAPACITY_GB * GB * HBM_SAFETY
    hbm_remaining = hbm_budget - static_hbm

    # ---- Optimizer + Master Weights per Card ----
    opt_master_per_card = (moe_fp32 / ep + backbone_fp32 / fsdp) * 2

    # ---- DDR Requirement ----
    ddr_requirement = opt_master_per_card + total_act_bytes - hbm_remaining

    # ---- Offload Ratios (clamped) ----
    moe_offload_ratio = min(1.0, max(0.0, (moe_act_total - hbm_remaining) / moe_act_total)) if moe_act_total > 0 else 0.0
    full_offload_ratio = min(1.0, max(0.0, (total_act_bytes - hbm_remaining) / total_act_bytes)) if total_act_bytes > 0 else 0.0

    # ---- Per-layer Times ----
    layer_times_ms = compute_layer_times()

    # ---- D2H Bandwidth Requirements ----
    max_d2h_moe_gbps = 0.0
    max_d2h_full_gbps = 0.0

    for i in range(N - 1):
        next_time_s = layer_times_ms[i + 1] / 1000.0
        if next_time_s <= 0:
            continue

        is_moe = per_layer_ffn[i] == "MoE"
        act_i = per_layer_act[i]
        moe_act_i = per_layer_moe_act[i]

        if is_moe and moe_act_i > 0:
            d2h_moe = (moe_act_i * moe_offload_ratio / next_time_s) / GB
            max_d2h_moe_gbps = max(max_d2h_moe_gbps, d2h_moe)

        d2h_full = (act_i * full_offload_ratio / next_time_s) / GB
        max_d2h_full_gbps = max(max_d2h_full_gbps, d2h_full)

    # ---- Task 3: D2H Bandwidth Sweep (ignore DDR constraint) ----
    bandwidth_benefits = {}
    for bw_gbs in D2H_BANDWIDTHS_GBS:
        effective_bw = bw_gbs * D2H_UTILIZATION * GB  # bytes/s

        min_cover = 1.0
        for i in range(N - 1):
            next_time_s = layer_times_ms[i + 1] / 1000.0
            if next_time_s <= 0:
                continue
            transfer_time = per_layer_act[i] * full_offload_ratio / effective_bw if effective_bw > 0 else float("inf")
            cover = min(1.0, next_time_s / transfer_time) if transfer_time > 0 else 1.0
            min_cover = min(min_cover, cover)

        benefit = min_cover * 25.0  # 25% = recompute share of total compute
        bandwidth_benefits[bw_gbs] = {
            "effective_gbps": round(bw_gbs * D2H_UTILIZATION, 1),
            "cover_ratio": round(min_cover, 4),
            "benefit_pct": round(benefit, 2),
        }

    # ---- Task 4: DDR Capacity Sweep (ignore D2H constraint) ----
    activation_needed = max(0, total_act_bytes - hbm_remaining)

    ddr_benefits = {}
    for ddr_gb in DDR_CAPACITIES_GB:
        available_ddr = max(0, ddr_gb * GB - opt_master_per_card)

        if activation_needed <= 0:
            ddr_offload = 1.0
        elif available_ddr >= activation_needed:
            ddr_offload = 1.0
        else:
            # offload_ratio = 1 - (needed - available) / needed
            shortage = activation_needed - available_ddr
            ddr_offload = max(0, 1.0 - shortage / activation_needed)

        benefit = ddr_offload * 25.0  # 25% = recompute share of total compute
        ddr_benefits[ddr_gb] = {
            "offload_ratio": round(ddr_offload, 4),
            "benefit_pct": round(benefit, 2),
        }

    return {
        "ep": ep,
        "fsdp": fsdp,
        "moe_bf16_gb": round(moe_bf16 / GB, 2),
        "backbone_bf16_gb": round(backbone_bf16 / GB, 2),
        "total_weight_bf16_gb": round(total_weight_bytes / GB, 2),
        "moe_per_card_gb": round(moe_per_card / GB, 2),
        "backbone_per_card_gb": round(backbone_per_card / GB, 2),
        "max_layer_act_gb": round(max_layer_act_bytes / GB, 4),
        "total_act_gb": round(total_act_bytes / GB, 2),
        "moe_act_total_gb": round(moe_act_total / GB, 2),
        "static_hbm_gb": round(static_hbm / GB, 2),
        "hbm_budget_gb": round(hbm_budget / GB, 2),
        "hbm_remaining_gb": round(hbm_remaining / GB, 2),
        "opt_master_per_card_gb": round(opt_master_per_card / GB, 2),
        "ddr_requirement_gb": round(ddr_requirement / GB, 2),
        "moe_offload_ratio": round(moe_offload_ratio, 4),
        "full_offload_ratio": round(full_offload_ratio, 4),
        "d2h_moe_gbps": round(max_d2h_moe_gbps, 2),
        "d2h_full_gbps": round(max_d2h_full_gbps, 2),
        "bandwidth_benefits": bandwidth_benefits,
        "ddr_benefits": ddr_benefits,
    }


# ============================================================================
# Main
# ============================================================================
results = [analyze_strategy(s["EP"], s["FSDP"]) for s in STRATEGIES]

# ---- Metadata ----
metadata = {
    "model": MODEL_DATA["model_family"],
    "config": "dsv4pro_config.json",
    "hardware": "A7 (BF16 1769 TFLOPS)",
    "seq_len": MODEL_DATA["compute"]["seq_len"],
    "batch_size": MODEL_DATA["compute"]["batch_size"],
    "hbm_capacity_gb": HBM_CAPACITY_GB,
    "mfU": MFU,
    "N_layers": N,
    "total_params": MODEL_DATA["model_params"]["total_params"],
}

final = {"metadata": metadata, "strategies": {}}
for s, r in zip(STRATEGIES, results):
    final["strategies"][s["name"]] = r

print(json.dumps(final, indent=2, ensure_ascii=False, default=str))