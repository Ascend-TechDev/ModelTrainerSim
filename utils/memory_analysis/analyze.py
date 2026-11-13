#!/usr/bin/env python3
"""EP/FSDP/HBM/DDR/D2H bandwidth analysis for all model configs.

Reference: utils/memory_analysis/EP_FSDP_MEM_HBM_D2H

Usage:
    cd /home/shawn/Repos/ModelTrainerSim
    python3 utils/memory_analysis/analyze.py [--config NAME] [--seq-len N] [--hbm GB]
"""
from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Ensure the project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.core.config_parser import ConfigParser
from src.core.model_registry import ModelRegistry

# ============================================================================
# Hardware Constants
# ============================================================================

# BF16 compute power (TFLOPS)
A5_BF16_TF = 432.0
A6_BF16_TF = 884.0   # ~2x A5
A7_BF16_TF = 1769.0   # 4.09x A5
A8_BF16_TF = 3538.0   # ~8x A5

# Vector FP32 compute power (TFLOPS)
A5_VEC_FP32_TF = 55.0
A7_VEC_FP32_TF = 27.0   # Yes, lower than A5

# BF16 compute ratio vs A5
ARCH_BF16_RATIO = {"A5": 1.0, "A6": 2.05, "A7": 4.09, "A8": 8.19}
# Vector FP32 ratio vs A5
ARCH_VEC_RATIO = {"A5": 1.0, "A6": 1.5, "A7": 2.04, "A8": 3.0}

# Communication overhead ratio (计算占比)
COMPUTE_SHARE = {"A5": 0.878, "A6": 0.92, "A7": 0.95, "A8": 0.95}

# Reference: Kimi3-2.8T on A5 @ 16K
REF_MLA_MATMUL_MS = 82.3    # ms
REF_MLA_FA_MS = 25.9        # ms (attention kernel)
REF_MLA_OTHER_MS = 19.5     # ms
REF_MLA_TOTAL_MS = 145.4    # ms
REF_MLA_TF = 33.21           # TFLOPS

REF_KDA_MATMUL_MS = 98.3    # ms
REF_KDA_KERNEL_MS = 27.4    # ms (KDA 6 operators)
REF_KDA_OTHER_MS = 20.3     # ms
REF_KDA_TOTAL_MS = 173.1    # ms
REF_KDA_TF = 38.24           # TFLOPS

REF_SEQ_LEN = 16384

# D2H Bandwidth (GB/s)
D2H_BANDWIDTHS_GBS = [200, 250, 300, 350, 400, 450, 500]
D2H_UTILIZATION = 0.8

# Memory constants
DDR_CAPACITY_GB = 384.0    # Per-card DDR capacity
HBM_SAFETY_FACTOR = 0.95
BUFFER_GB = 3.0            # Fixed per-GPU buffer
GB = 1e9                   # Bytes per GB

# GPU tier map (param range -> total GPUs)
TIER_CONFIG = [
    (0,        1e12,   10240,  [1024, 4096, 8192],       [128, 256]),
    (1e12,     10e12,  10240,  [1024, 4096, 8192],       [128, 256]),
    (10e12,    40e12,  204800, [10240, 51200, 102400],    [128, 1024]),
    (40e12,    80e12,  500000, [],                         []),
    (80e12,    1e15,   1000000,[],                         []),
]

WORLD_SIZE_MULTIPLIERS = [1.0, 1.5, 2.0, 4.0, 8.0, 16.0]
ARCH_NAMES = ["A5", "A6", "A7", "A8"]


def get_tier_info(total_params: int) -> dict:
    for lo, hi, gpu, fsdp, ep in TIER_CONFIG:
        if lo <= total_params < hi:
            return {"total_gpu": gpu, "fsdp_options": fsdp, "ep_options": ep}
    return {"total_gpu": 1000000, "fsdp_options": [], "ep_options": []}


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class MemoryData:
    """Extracted memory data from a model analysis report."""
    model_family: str
    config_name: str
    total_params: int
    total_weight_bytes: float
    training_breakdown: dict
    per_layer_act: list[float]      # per-layer activation bytes
    per_layer_flops: list[float]    # per-layer FLOPs
    per_layer_type: list[str]       # per-layer attention type (MLA/KDA/...)
    per_layer_ffn: list[str]        # per-layer FFN type (MoE/Dense/...)
    max_layer_act_bytes: float
    total_act_bytes: float          # sum of per-layer activations
    seq_len: int
    batch_size: int
    total_flops: float

    # Derived
    moe_weight_bytes: float = 0.0
    backbone_weight_bytes: float = 0.0
    optimizer_states_bytes: float = 0.0
    master_weight_bytes: float = 0.0


@dataclass
class WorldSizeResult:
    world_size: int
    multiplier: float
    static_hbm_gb: float
    dynamic_hbm_gb: float
    hbm_remaining_gb: float
    ddr_requirement_gb: float
    d2h_moe_gbps: dict     # bandwidth -> required Gbps
    d2h_full_gbps: dict    # bandwidth -> required Gbps
    moe_offload_ratio: float
    full_offload_ratio: float
    max_batch_size: int
    perf_benefits: dict    # arch_name -> benefit_percent
    per_layer_times_ms: list[float] = field(default_factory=list)


# ============================================================================
# Model Loading
# ============================================================================

def load_model(config_path: str, seq_len: int = 4096) -> dict:
    """Load a model config and run analysis. Returns the report dict."""
    cfg = ConfigParser(config_path).load()
    model = ModelRegistry.build(cfg["model"], cfg)
    return model.analyze(seq_len=seq_len, batch_size=1)


def extract_memory_data(report: dict, config_name: str) -> MemoryData:
    """Extract structured memory data from a model analysis report."""
    m = report["memory"]
    p = report["model_params"]
    c = report["compute"]
    tb = m["training_breakdown"]
    layers = report["layers"]

    per_layer_act = [lyr["act_bytes"] for lyr in layers]
    per_layer_flops = [lyr["flops"] for lyr in layers]
    per_layer_type = [lyr.get("attention_type", lyr.get("is_dense", "None")) for lyr in layers]
    per_layer_ffn = [lyr.get("ffn_type", "Dense" if lyr.get("is_dense") else "MoE") for lyr in layers]

    max_act = max(per_layer_act) if per_layer_act else 0.0
    total_act = sum(per_layer_act)

    # Extract group-level data from training_breakdown
    groups = tb.get("groups", {})
    moe_wt = sum(g["weights"] for name, g in groups.items()
                 if name in ("expert", "muon") and not name == "dense")
    backbone_wt = tb["weights"] - moe_wt

    opt_states = tb.get("optimizer_states", 0.0)
    master_wt = tb.get("master_weights", 0.0)

    return MemoryData(
        model_family=report["model_family"],
        config_name=config_name,
        total_params=p["total_params"],
        total_weight_bytes=m["weight_memory_bytes"],
        training_breakdown=tb,
        per_layer_act=per_layer_act,
        per_layer_flops=per_layer_flops,
        per_layer_type=per_layer_type,
        per_layer_ffn=per_layer_ffn,
        max_layer_act_bytes=max_act,
        total_act_bytes=total_act,
        seq_len=c.get("seq_len", 4096),
        batch_size=c.get("batch_size", 1),
        total_flops=c.get("total_flops_per_sample", 0.0),
        moe_weight_bytes=moe_wt,
        backbone_weight_bytes=backbone_wt,
        optimizer_states_bytes=opt_states,
        master_weight_bytes=master_wt,
    )


# ============================================================================
# Memory Calculations
# ============================================================================

def compute_static_hbm(mem: MemoryData, world_size: int) -> float:
    """Per-GPU static HBM: max_layer_act + (weights+grads)/world_size + buffer.

    (MoE+Backbone)*2 = weights + gradients (both bf16, so = total_weight_bytes * 2).
    """
    weights_grads = mem.total_weight_bytes * 2  # weights + gradients in bf16
    return mem.max_layer_act_bytes + weights_grads / world_size + BUFFER_GB * GB


def compute_dynamic_hbm(mem: MemoryData) -> float:
    """Per-GPU dynamic HBM: sum of per-layer input activations."""
    return mem.total_act_bytes


def compute_hbm_remaining(mem: MemoryData, world_size: int, hbm_capacity_gb: float) -> float:
    """HBM remaining per GPU after static + dynamic allocation."""
    static = compute_static_hbm(mem, world_size)
    dynamic = compute_dynamic_hbm(mem)
    return hbm_capacity_gb * GB * HBM_SAFETY_FACTOR - static - dynamic


def compute_ddr_requirement(
    mem: MemoryData, world_size: int, hbm_capacity_gb: float, batch_size: int = 1
) -> float:
    """Per-GPU DDR capacity needed for offload.

    DDR = (optimizer_states + master_weights) / world_size
          + max(0, static_hbm + dynamic_hbm - HBM * 0.95)
    The overflow term is what doesn't fit in HBM (must go to DDR).
    """
    opt_master = mem.optimizer_states_bytes + mem.master_weight_bytes
    static = compute_static_hbm(mem, world_size)
    dynamic = compute_dynamic_hbm(mem) * batch_size
    hbm_budget = hbm_capacity_gb * GB * HBM_SAFETY_FACTOR
    overflow = max(0, static + dynamic - hbm_budget)
    return (opt_master / world_size) + overflow


# ============================================================================
# Minimum World Size Search
# ============================================================================

def find_min_world_size(
    mem: MemoryData, hbm_capacity_gb: float, tier_gpus: int
) -> tuple[int, str]:
    """Find minimum world size where static HBM fits in 95% HBM.

    Returns (world_size, reason).

    Primary constraint: static_hbm < HBM * 0.95
    DDR constraint is checked as a secondary soft constraint.

    static_hbm = max_layer_act + weights_grads/ws + buffer
    => ws > weights_grads / (HBM * 0.95 - max_layer_act - buffer)
    """
    weights_grads = mem.total_weight_bytes * 2
    denom = hbm_capacity_gb * GB * HBM_SAFETY_FACTOR - mem.max_layer_act_bytes - BUFFER_GB * GB
    if denom <= 0:
        # Even infinite cards can't fit (max_layer_act alone exceeds HBM budget)
        return tier_gpus, "max_layer_act exceeds HBM budget; using tier max"

    # Minimum world_size from HBM static constraint
    ws_min_hbm = max(1, int(math.ceil(weights_grads / denom)))

    # Check DDR at ws_min_hbm; try doubling until feasible
    lo, hi = ws_min_hbm, tier_gpus
    best = hi

    # First find an upper bound where DDR < 384GB
    # Always check the tier max as well
    ddr_at_max = compute_ddr_requirement(mem, tier_gpus, hbm_capacity_gb)
    if ddr_at_max < DDR_CAPACITY_GB * GB * 0.98:
        hi = tier_gpus
    else:
        ws = ws_min_hbm
        while ws <= tier_gpus:
            ddr = compute_ddr_requirement(mem, ws, hbm_capacity_gb)
            if ddr < DDR_CAPACITY_GB * GB * 0.98:
                hi = ws
                break
            ws = min(ws * 2, tier_gpus + 1)
            # Ensure we check the exact tier max
            if ws > tier_gpus:
                ws = tier_gpus + 1  # will exit the while loop
        else:
            # Couldn't satisfy DDR even at tier max — return HBM minimum
            return ws_min_hbm, (
                f"HBM constraint (DDR={compute_ddr_requirement(mem, ws_min_hbm, hbm_capacity_gb) / GB:.0f}GB"
                f" > {DDR_CAPACITY_GB}GB; activation checkpointing needed)"
            )

    # Binary search between lo and hi for the minimum
    best = hi
    for _ in range(30):
        mid = (lo + hi) // 2 + 1
        if mid > hi:
            break
        ddr = compute_ddr_requirement(mem, mid, hbm_capacity_gb)
        if ddr < DDR_CAPACITY_GB * GB * 0.98 and compute_static_hbm(mem, mid) < hbm_capacity_gb * GB * HBM_SAFETY_FACTOR:
            best = mid
            hi = mid - 1
        else:
            lo = mid

    reason = "HBM+DDR balanced" if best > ws_min_hbm else "HBM constraint"
    return best, reason


# ============================================================================
# Per-Layer Time Estimation
# ============================================================================

def estimate_per_layer_times(
    mem: MemoryData, arch_name: str = "A7"
) -> list[float]:
    """Estimate per-layer forward time (ms) using A5 anchor + hardware scaling.

    Methodology from EP_FSDP_MEM_HBM_D2H:
      matmul_time(A7) = ref_matmul_ms * (layer_flops / ref_flops) / BF16_ratio
      attention_time(A7) = ref_attn_ms * (layer_flops_attn / ref_flops_attn) / BF16_ratio
      other_time(A7) = ref_other_ms * (seq_len / ref_seq_len) / VEC_ratio
      total_compute = sum of above
      total_time = total_compute / compute_share
    """
    bf16_ratio = ARCH_BF16_RATIO.get(arch_name, 4.09)
    vec_ratio = ARCH_VEC_RATIO.get(arch_name, 2.04)
    compute_share = COMPUTE_SHARE.get(arch_name, 0.95)
    seq_ratio = mem.seq_len / REF_SEQ_LEN

    times_ms = []
    for i, (flops, attn_type, ffn_type) in enumerate(
        zip(mem.per_layer_flops, mem.per_layer_type, mem.per_layer_ffn)
    ):
        is_kda = "KDA" in str(attn_type).upper()

        if is_kda:
            ref_total_ms = REF_KDA_TOTAL_MS
            ref_tf = REF_KDA_TF
        else:
            ref_total_ms = REF_MLA_TOTAL_MS
            ref_tf = REF_MLA_TF

        layer_tf = flops / 1e12
        ratio = layer_tf / ref_tf if ref_tf > 0 else 1.0

        # Split: 85% matmul, 10% attention kernel, 5% other (approximate)
        matmul_share = 0.85
        attn_share = 0.10
        other_share = 0.05

        matmul_ms = ref_total_ms * ratio * matmul_share / bf16_ratio
        attn_ms = ref_total_ms * ratio * attn_share / bf16_ratio
        other_ms = ref_total_ms * ratio * other_share * seq_ratio / vec_ratio

        compute_ms = matmul_ms + attn_ms + other_ms
        total_ms = compute_ms / compute_share
        times_ms.append(total_ms)

    return times_ms


# ============================================================================
# D2H Bandwidth Requirements
# ============================================================================

def compute_d2h_requirements(
    mem: MemoryData,
    hbm_remaining: float,
    layer_times_ms: list[float],
) -> dict:
    """Compute D2H bandwidth requirements for MoE-only and full activation offload.

    D2H bandwidth = max(layer_i_activation * offload_ratio / layer_i+1_forward_time)
    """
    n = len(mem.per_layer_act)
    if n < 2:
        return {
            "d2h_moe_gbps": 0.0,
            "d2h_full_gbps": 0.0,
            "moe_offload_ratio": 0.0,
            "full_offload_ratio": 0.0,
        }

    # Calculate MoE activation total
    moe_act_total = sum(
        act for act, ffn in zip(mem.per_layer_act, mem.per_layer_ffn)
        if ffn == "MoE"
    )
    if moe_act_total == 0:
        moe_act_total = mem.total_act_bytes

    # Offload ratios
    hbm_shortfall = max(0, -hbm_remaining)
    moe_offload_ratio = min(1.0, hbm_shortfall / moe_act_total) if moe_act_total > 0 else 0.0
    full_offload_ratio = min(1.0, hbm_shortfall / mem.total_act_bytes) if mem.total_act_bytes > 0 else 0.0

    # D2H bandwidth: max over layers of (act_i * ratio / time_i+1)
    max_d2h_moe = 0.0
    max_d2h_full = 0.0
    for i in range(n - 1):
        next_time_s = layer_times_ms[i + 1] / 1000.0 if i + 1 < len(layer_times_ms) else 0.001
        if next_time_s <= 0:
            continue

        is_moe = mem.per_layer_ffn[i] == "MoE"
        act_bytes = mem.per_layer_act[i]

        if is_moe:
            d2h_moe = (act_bytes * moe_offload_ratio / next_time_s) / 1e9  # GB/s
            max_d2h_moe = max(max_d2h_moe, d2h_moe)

        d2h_full = (act_bytes * full_offload_ratio / next_time_s) / 1e9  # GB/s
        max_d2h_full = max(max_d2h_full, d2h_full)

    return {
        "d2h_moe_gbps": max_d2h_moe,
        "d2h_full_gbps": max_d2h_full,
        "moe_offload_ratio": moe_offload_ratio,
        "full_offload_ratio": full_offload_ratio,
    }


# ============================================================================
# Performance Benefit Calculation
# ============================================================================

def compute_performance_benefit(
    mem: MemoryData,
    world_size: int,
    hbm_capacity_gb: float,
) -> dict[str, float]:
    """Compute end-to-end recompute elimination benefit per architecture.

    For each D2H bandwidth option:
      cover_ratio = min(1, compute_time / transfer_time)
      offload_ratio from DDR constraint = max(0, DDR_available / activation_needed)
      combined = min(cover_ratio, offload_ratio)
      end_to_end_benefit = combined * 25%
    """
    n = len(mem.per_layer_act)
    if n < 2:
        return {arch: 0.0 for arch in ARCH_NAMES}

    ddr = compute_ddr_requirement(mem, world_size, hbm_capacity_gb, batch_size=1)
    ddr_available = max(0, DDR_CAPACITY_GB * GB - ddr)
    ddr_offload_ratio = (min(1.0, ddr_available / mem.total_act_bytes)
                         if mem.total_act_bytes > 0 else 0.0)

    benefits = {}
    for arch_name in ARCH_NAMES:
        layer_times = estimate_per_layer_times(mem, arch_name)

        best_benefit = 0.0
        for bw_gbs in D2H_BANDWIDTHS_GBS:
            effective_bw = bw_gbs * D2H_UTILIZATION * 1e9  # bytes/s

            # D2H constraint: find min cover_ratio across layers
            min_cover = 1.0
            for i in range(n - 1):
                next_time_s = layer_times[i + 1] / 1000.0 if i + 1 < len(layer_times) else 0.001
                if next_time_s <= 0:
                    continue
                transfer_time = mem.per_layer_act[i] / effective_bw if effective_bw > 0 else float("inf")
                cover = min(1.0, next_time_s / transfer_time) if transfer_time > 0 else 1.0
                min_cover = min(min_cover, cover)

            combined = min(min_cover, ddr_offload_ratio)
            benefit = combined * 25.0  # max 25% end-to-end benefit
            best_benefit = max(best_benefit, benefit)

        benefits[arch_name] = best_benefit

    return benefits


# ============================================================================
# Max Batch Size
# ============================================================================

def compute_max_batch_size(
    mem: MemoryData, world_size: int, hbm_capacity_gb: float
) -> int:
    """Find maximum supported batch size (power of 2) without exceeding DDR capacity."""
    max_bs = 1
    for bs_pow in range(0, 10):  # 1, 2, 4, 8, ..., 512
        bs = 1 << bs_pow
        ddr = compute_ddr_requirement(mem, world_size, hbm_capacity_gb, batch_size=bs)
        if ddr < DDR_CAPACITY_GB * GB * 0.9:  # 90% safety
            max_bs = bs
        else:
            break
    return max_bs


# ============================================================================
# Single Config Analysis
# ============================================================================

def analyze_single(
    config_path: Path, seq_len: int, hbm_gb: float
) -> dict | None:
    """Run full analysis for one config + seq_len + HBM combination."""
    config_name = config_path.stem

    try:
        report = load_model(str(config_path), seq_len)
    except Exception as e:
        print(f"  ERROR loading {config_name}: {e}", file=sys.stderr)
        return None

    try:
        mem = extract_memory_data(report, config_name)
    except Exception as e:
        print(f"  ERROR extracting data from {config_name}: {e}", file=sys.stderr)
        return None

    tier = get_tier_info(mem.total_params)
    nmin, reason = find_min_world_size(mem, hbm_gb, tier["total_gpu"])

    results = {
        "model": mem.model_family,
        "config_file": config_name,
        "seq_len": mem.seq_len,
        "hbm_capacity_gb": hbm_gb,
        "total_params": mem.total_params,
        "total_weight_bytes": mem.total_weight_bytes,
        "max_layer_act_gb": mem.max_layer_act_bytes / GB,
        "total_act_gb": mem.total_act_bytes / GB,
        "nmin": nmin,
        "nmin_reason": reason,
        "tier_total_gpu": tier["total_gpu"],
        "weights_grads_gb": mem.total_weight_bytes * 2 / GB,
        "optimizer_states_gb": mem.optimizer_states_bytes / GB,
        "master_weights_gb": mem.master_weight_bytes / GB,
        "world_sizes": {},
    }

    for mult in WORLD_SIZE_MULTIPLIERS:
        ws = max(1, int(nmin * mult))
        static_gb = compute_static_hbm(mem, ws) / GB
        dynamic_gb = compute_dynamic_hbm(mem) / GB
        hbm_rem_gb = compute_hbm_remaining(mem, ws, hbm_gb) / GB
        ddr_gb = compute_ddr_requirement(mem, ws, hbm_gb) / GB

        layer_times = estimate_per_layer_times(mem, "A7")
        d2h = compute_d2h_requirements(mem, hbm_rem_gb * GB, layer_times)

        perf = compute_performance_benefit(mem, ws, hbm_gb)
        max_bs = compute_max_batch_size(mem, ws, hbm_gb)

        results["world_sizes"][f"{mult}x"] = {
            "world_size": ws,
            "multiplier": mult,
            "static_hbm_gb": round(static_gb, 3),
            "dynamic_hbm_gb": round(dynamic_gb, 3),
            "hbm_remaining_gb": round(hbm_rem_gb, 3),
            "ddr_requirement_gb": round(ddr_gb, 3),
            "d2h_moe_gbps": round(d2h["d2h_moe_gbps"], 2),
            "d2h_full_gbps": round(d2h["d2h_full_gbps"], 2),
            "moe_offload_ratio": round(d2h["moe_offload_ratio"], 4),
            "full_offload_ratio": round(d2h["full_offload_ratio"], 4),
            "max_batch_size": max_bs,
            "perf_benefit_pct": {k: round(v, 2) for k, v in perf.items()},
        }

    return results


# ============================================================================
# Summary Table
# ============================================================================

def generate_summary_table(all_results: list[dict], output_dir: Path) -> str:
    """Generate a markdown summary table matching the D2H document format."""
    lines = [
        "# EP/FSDP Memory Analysis Summary",
        "",
        f"Generated from {len(all_results)} config × HBM × seq_len combinations.",
        "",
        "## All Models at seq_len=4096, HBM=288GB",
        "",
    ]

    # Header
    header = (
        "| 参数范围 | 模型 | 总参数 | 激活/token | 训练token | "
        "Nmin | 1×Nmin static(GB) | 1×Nmin DDR(GB) | "
        "D2H MoE(Gbps) | D2H Full(Gbps) | "
        "Max BS | A5收益 | A7收益 |"
    )
    sep = "|---|---|---|---|---|---|---|---|---|---|---|---|---|"
    lines.extend([header, sep])

    # Sort by total params
    sorted_results = sorted(all_results, key=lambda r: r.get("total_params", 0))

    for r in sorted_results:
        total_params = r["total_params"]
        total_t = total_params / 1e12
        if total_t < 1:
            param_range = "0~1T"
        elif total_t < 10:
            param_range = "1T~10T"
        elif total_t < 20:
            param_range = "10T~20T"
        elif total_t < 40:
            param_range = "20T~40T"
        elif total_t < 80:
            param_range = "40T~80T"
        else:
            param_range = "80T+"

        ws1 = r["world_sizes"].get("1.0x", {})

        params_str = f"{total_t:.3f} T"
        # Activation per token: total activation / seq_len (activation per position)
        act_per_tok_bytes = r.get("total_act_gb", 0) * GB / r.get("seq_len", 4096) if r.get("seq_len") else 0
        act_str = f"{act_per_tok_bytes / GB:.2f} B" if act_per_tok_bytes > 0 else "-"

        model_name = r["model"]
        # Truncate very long model names for display
        if len(model_name) > 20:
            model_name = r["config_file"].replace("_config", "").replace(".json", "")[:20]

        lines.append(
            f"| {param_range} | {model_name} | {params_str} | {act_str} | - | "
            f"{r['nmin']:,} | {ws1.get('static_hbm_gb', 0):.1f} | {ws1.get('ddr_requirement_gb', 0):.1f} | "
            f"{ws1.get('d2h_moe_gbps', 0):.1f} | {ws1.get('d2h_full_gbps', 0):.1f} | "
            f"{ws1.get('max_batch_size', 1)} | "
            f"{ws1.get('perf_benefit_pct', {}).get('A5', 0):.1f}% | "
            f"{ws1.get('perf_benefit_pct', {}).get('A7', 0):.1f}% |"
        )

    lines.extend(["", "---", "", "## Detailed Reports", ""])
    for r in sorted_results:
        name = r["config_file"]
        lines.append(f"- [{name}]({name}_288GB_4K.json)")

    return "\n".join(lines)


# ============================================================================
# Main
# ============================================================================

def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="EP/FSDP/HBM/DDR/D2H bandwidth analysis"
    )
    parser.add_argument("--config", "-c", type=str, default=None,
                        help="Single config file name (e.g. dsv41flash_config.json)")
    parser.add_argument("--seq-len", "-s", type=int, default=None,
                        help="Sequence length (default: 4096)")
    parser.add_argument("--hbm", type=float, default=None,
                        help="HBM capacity in GB (default: 288)")
    parser.add_argument("--all", action="store_true", default=True,
                        help="Process all configs (default)")
    parser.add_argument("--output-dir", "-o", type=str,
                        default=str(_PROJECT_ROOT / "reports_analyse"),
                        help="Output directory")

    args = parser.parse_args()

    config_dir = _PROJECT_ROOT / "configs"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config_files = list(config_dir.glob("*.json"))
    if args.config:
        config_files = [config_dir / args.config]
        if not config_files[0].exists():
            print(f"Config not found: {args.config}", file=sys.stderr)
            return 1

    seq_lens = [args.seq_len] if args.seq_len else [4096]
    hbm_sizes = [args.hbm] if args.hbm else [288.0]

    print(f"Analyzing {len(config_files)} configs × {len(seq_lens)} seq_lens × {len(hbm_sizes)} HBM sizes")
    print(f"Output: {output_dir}")
    print()

    all_results = []
    for config_file in config_files:
        for seq_len in seq_lens:
            for hbm_gb in hbm_sizes:
                label = f"{config_file.stem} @ seq={seq_len} HBM={hbm_gb}GB"
                print(f"  {label} ...", end=" ", flush=True)
                result = analyze_single(config_file, seq_len, hbm_gb)
                if result:
                    all_results.append(result)
                    out_name = f"{config_file.stem}_{int(hbm_gb)}GB_{seq_len//1024}K.json"
                    out_path = output_dir / out_name
                    out_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
                    nmin = result["nmin"]
                    print(f"Nmin={nmin:,} ✓")
                else:
                    print("FAILED")

    # Generate summary
    if all_results:
        summary_path = output_dir / "summary.md"
        summary_path.write_text(generate_summary_table(all_results, output_dir), encoding="utf-8")
        print(f"\nSummary written to: {summary_path}")
        print(f"Total reports: {len(all_results)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())