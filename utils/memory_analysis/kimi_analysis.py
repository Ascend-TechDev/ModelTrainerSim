#!/usr/bin/env python3
"""Kimi family EP/EDP/EFSDP/FSDP/HBM/DDR/D2H analysis for 970DT.

Key changes from prior version:
- MFU = 0.56 (was 0.95)
- MoE sharding uses EP * EFSDP (MoE/EP/EFSDP)
- EDP and EFSDP computed per strategy definition
"""
import json, math, sys
from pathlib import Path

# ============================================================================
# Config
# ============================================================================
MODEL_TO_K = {
    "kimi_10t_deep":    40120,
    "kimi_10t_wide":    64374,
    "kimi_50t":        299158,
    "kimi_100t":       638001,
}

EP_OPTIONS = [128, 256, 512, 1024]
HBM_OPTIONS = [192, 288]
HBM_SAFETY = 0.95
BUFFER_GB = 3.0
GB = 1e9

BF16_TFLOPS = 1769e12
MFU = 0.56

D2H_BANDWIDTHS_GBS = [200, 250, 300, 350, 400, 450, 500]
D2H_UTIL = 0.8
DDR_CAPS_GB = [256, 384, 512, 540, 768]

# ============================================================================
# Helpers
# ============================================================================
def compute_strategy(ep, K):
    """Returns (EDP, EFSDP, FSDP) per the document's strategy definitions."""
    fsdp = K
    edp = int(math.floor(K / ep / 2))
    efsdp = int(math.floor(K / 128 / 2))
    return edp, efsdp, fsdp

def load_model_data(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    layers = data["layers"]
    total_wt = data["memory"]["weight_memory_bytes"]
    excel = data["per_module_summary_excel"]

    if "routedexpert" in excel:
        moe_bf16 = excel["routedexpert"]["bytes"] + excel["sharedexpert"]["bytes"] + excel["gate"]["bytes"]
    else:
        moe_bf16 = excel["moe"]["bytes"]

    backbone_bf16 = total_wt - moe_bf16
    moe_fp32 = moe_bf16 * 2
    backbone_fp32 = backbone_bf16 * 2

    total_act = sum(l["act_bytes"] for l in layers)
    max_act = max(l["act_bytes"] for l in layers)
    moe_act = sum(
        sum(m["act_bytes"] for m in l["modules"] if m["type"] == "MoE")
        for l in layers
    )
    per_layer_act = [l["act_bytes"] for l in layers]
    per_layer_flops = [l["flops"] for l in layers]
    per_layer_ffn = [l["ffn_type"] for l in layers]
    per_layer_moe_act = [
        sum(m["act_bytes"] for m in l["modules"] if m["type"] == "MoE")
        for l in layers
    ]

    times_ms = [flops / (BF16_TFLOPS * MFU) * 1000 for flops in per_layer_flops]

    return {
        "total_params": data["model_params"]["total_params"],
        "total_weight_bytes": total_wt,
        "moe_bf16": moe_bf16,
        "backbone_bf16": backbone_bf16,
        "moe_fp32": moe_fp32,
        "backbone_fp32": backbone_fp32,
        "total_act_bytes": total_act,
        "max_layer_act_bytes": max_act,
        "moe_act_total": moe_act,
        "per_layer_act": per_layer_act,
        "per_layer_flops": per_layer_flops,
        "per_layer_ffn": per_layer_ffn,
        "per_layer_moe_act": per_layer_moe_act,
        "layer_times_ms": times_ms,
        "N": len(layers),
    }

def analyze(m, ep, edp, efsdp, fsdp, hbm_gb):
    """Run all calculations for one (EP, EDP, EFSDP, FSDP, HBM) combination."""
    N = m["N"]

    # Static HBM — MoE uses EP * EFSDP sharding
    moe_per_card = m["moe_bf16"] / ep / efsdp * 2
    backbone_per_card = m["backbone_bf16"] / fsdp * 2
    static_hbm = m["max_layer_act_bytes"] + moe_per_card + backbone_per_card + BUFFER_GB * GB
    hbm_budget = hbm_gb * GB * HBM_SAFETY
    hbm_remaining = hbm_budget - static_hbm

    # Optimizer + Master per card — MoE uses EP * EFSDP
    opt_master_per_card = (m["moe_fp32"] / ep / efsdp + m["backbone_fp32"] / fsdp) * 2

    # DDR requirement
    ddr_req = opt_master_per_card + m["total_act_bytes"] - hbm_remaining

    # Offload ratios
    moe_offload = min(1.0, max(0.0, (m["moe_act_total"] - hbm_remaining) / m["moe_act_total"])) if m["moe_act_total"] > 0 else 0.0
    full_offload = min(1.0, max(0.0, (m["total_act_bytes"] - hbm_remaining) / m["total_act_bytes"])) if m["total_act_bytes"] > 0 else 0.0

    # D2H bandwidth requirements
    max_d2h_moe = 0.0
    max_d2h_full = 0.0
    for i in range(N - 1):
        next_t = m["layer_times_ms"][i + 1] / 1000.0
        if next_t <= 0:
            continue
        if m["per_layer_ffn"][i] == "MoE" and m["per_layer_moe_act"][i] > 0:
            d2h_m = (m["per_layer_moe_act"][i] * moe_offload / next_t) / GB
            max_d2h_moe = max(max_d2h_moe, d2h_m)
        d2h_f = (m["per_layer_act"][i] * full_offload / next_t) / GB
        max_d2h_full = max(max_d2h_full, d2h_f)

    # Task 3: Bandwidth sweep
    bw_results = {}
    for bw in D2H_BANDWIDTHS_GBS:
        eff_bw = bw * D2H_UTIL * GB
        min_cover = 1.0
        for i in range(N - 1):
            next_t = m["layer_times_ms"][i + 1] / 1000.0
            if next_t <= 0:
                continue
            tx = m["per_layer_act"][i] * full_offload / eff_bw if eff_bw > 0 else float("inf")
            c = min(1.0, next_t / tx) if tx > 0 else 1.0
            min_cover = min(min_cover, c)
        bw_results[bw] = {"eff_gbps": round(bw * D2H_UTIL, 1),
                          "cover": round(min_cover, 4),
                          "benefit": round(min_cover * 25, 2)}

    # Task 4: DDR capacity sweep
    act_needed = max(0, m["total_act_bytes"] - hbm_remaining)
    ddr_results = {}
    for ddr_gb in DDR_CAPS_GB:
        avail = max(0, ddr_gb * GB - opt_master_per_card)
        if act_needed <= 0 or avail >= act_needed:
            offl = 1.0
        else:
            offl = max(0, 1.0 - (act_needed - avail) / act_needed)
        ddr_results[ddr_gb] = {"offload": round(offl, 4),
                               "benefit": round(offl * 25, 2)}

    return {
        "ep": ep, "edp": edp, "efsdp": efsdp, "fsdp": fsdp,
        "moe_per_card_gb": round(moe_per_card / GB, 2),
        "backbone_per_card_gb": round(backbone_per_card / GB, 6),
        "static_hbm_gb": round(static_hbm / GB, 2),
        "hbm_budget_gb": round(hbm_budget / GB, 1),
        "hbm_remaining_gb": round(hbm_remaining / GB, 2),
        "opt_master_per_card_gb": round(opt_master_per_card / GB, 2),
        "ddr_req_gb": round(ddr_req / GB, 2),
        "moe_offload": round(moe_offload, 4),
        "full_offload": round(full_offload, 4),
        "d2h_moe_gbps": round(max_d2h_moe, 2),
        "d2h_full_gbps": round(max_d2h_full, 2),
        "bandwidth_sweep": bw_results,
        "ddr_sweep": ddr_results,
    }

# ============================================================================
# Main
# ============================================================================
def main():
    results = {}
    for fname, K in MODEL_TO_K.items():
        path = f"/tmp/{fname}.json"
        if not Path(path).exists():
            print(f"SKIP {path} — not found", file=sys.stderr)
            continue
        m = load_model_data(path)
        label = f"{fname} ({m['total_params']/1e12:.3f}T params)"
        results[label] = {}
        for hbm_gb in HBM_OPTIONS:
            results[label][f"HBM={hbm_gb}GB"] = {}
            for ep in EP_OPTIONS:
                edp, efsdp, fsdp = compute_strategy(ep, K)
                r = analyze(m, ep, edp, efsdp, fsdp, hbm_gb)
                results[label][f"HBM={hbm_gb}GB"][f"EP={ep}"] = r

    print(json.dumps(results, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()