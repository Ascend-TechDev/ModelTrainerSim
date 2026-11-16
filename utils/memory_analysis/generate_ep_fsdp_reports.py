#!/usr/bin/env python3
"""Generate per-model EP/FSDP HBM, DDR, D2H, and SSD reports."""
from __future__ import annotations

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config_parser import ConfigParser
from src.core.model_registry import ModelRegistry

GB = 1e9
MFU = 0.56
HBM_SAFETY = 0.95
BUFFER_GB = 3.0
HBM_CAPACITIES_GB = [192, 288]
BANDWIDTHS_GBS = [200, 250, 300, 350, 400, 450, 500]
DDR_CAPACITIES_GB = [256, 384, 512, 540, 768]
EP_VALUES = [128, 256, 512, 1024]
EFSDP_VALUES = [1, 2, 4, 8, 16, 32]

# Parameter counts and 970DT card counts come from the table in
# EP_FSDP_MEM_HBM_D2H.md. Configurations provide the layer and memory model.
MODELS = [
    ("Qwen3.5-397B", 0.397, (1541, 771), None),
    ("GLM-5.2", 0.743, (4859, 2430), "glm_5_2_config.json"),
    ("DSV41Flash", 0.748, (1900, 950), "dsv41flash_config.json"),
    ("DSV4Pro", 1.574, (5817, 2909), "dsv4pro_config.json"),
    ("Kimi-10T-D", 9.283, (40120, 20060), "kimi_10t_deep_v2_config.json"),
    ("Hy4-10T", 10.074, (88636, 44318), "hy4_10t_config.json"),
    ("GLM-10T", 11.295, (37595, 18798), "glm_10t_config.json"),
    ("Kimi-10T-W", 11.918, (64374, 32187), "kimi_10t_wide_v2_config.json"),
    ("DS-10T", 15.835, (36230, 18115), "dsv41next10t_config.json"),
    ("DS-20T", 31.656, (71689, 35845), "dsv41next20t_config.json"),
    ("Hy4-30T", 41.501, (150748, 75374), "hy4_30t_config.json"),
    ("Kimi-50T", 49.996, (299158, 149579), "kimi_50t_config.json"),
    ("DS-40T", 63.269, (154875, 77438), "dsv41next40t_config.json"),
    ("Kimi-100T", 105.003, (638001, 319001), "kimi_100t_config.json"),
    ("DS-80T", 126.484, (269718, 134859), "dsv41next80t_config.json"),
]


def load_report(config_name: str) -> dict:
    config_path = PROJECT_ROOT / "configs" / config_name
    config = ConfigParser(str(config_path)).load()
    model = ModelRegistry.build(config["model"], config)
    return model.analyze(seq_len=4096, batch_size=1)


def get_moe_weight_bytes(report: dict) -> float:
    module_summary = report.get("per_module_summary_excel", {})
    if module_summary:
        moe_keys = [
            key for key in module_summary
            if "expert" in key.lower()
            or key.lower() in {"moe", "gate", "moerouter"}
        ]
        if moe_keys:
            return sum(module_summary[key].get("bytes", 0.0) for key in moe_keys)

    groups = report["memory"]["training_breakdown"].get("groups", {})
    if "expert" in groups:
        return groups["expert"].get("weights", 0.0)
    if "moe" in groups:
        return groups["moe"].get("weights", 0.0)
    raise ValueError(f"Cannot determine MoE weight bytes for {report['model_family']}")


def extract_model_data(report: dict) -> dict:
    layers = report["layers"]
    total_weight_bytes = report["memory"]["weight_memory_bytes"]
    moe_weight_bytes = get_moe_weight_bytes(report)
    backbone_weight_bytes = total_weight_bytes - moe_weight_bytes
    if backbone_weight_bytes < 0:
        raise ValueError("MoE weight split exceeds total model weight bytes")

    layer_act_bytes = [layer["act_bytes"] for layer in layers]
    moe_layer_act_bytes = [
        sum(
            module.get("act_bytes", 0.0)
            for module in layer.get("modules", [])
            if "moe" in str(module.get("type", "")).lower()
            or "expert" in str(module.get("type", "")).lower()
        )
        for layer in layers
    ]
    layer_flops = [layer["flops"] for layer in layers]
    return {
        "config_params": report["model_params"]["total_params"],
        "layers": len(layers),
        "total_weight_bytes": total_weight_bytes,
        "moe_weight_bytes": moe_weight_bytes,
        "backbone_weight_bytes": backbone_weight_bytes,
        "master_weight_bytes": report["memory"]["training_breakdown"].get("master_weights", 0.0),
        "optimizer_state_bytes": report["memory"]["training_breakdown"].get("optimizer_states", 0.0),
        "training_groups": report["memory"]["training_breakdown"].get("groups", {}),
        "layer_act_bytes": layer_act_bytes,
        "moe_layer_act_bytes": moe_layer_act_bytes,
        "total_act_bytes": sum(layer_act_bytes),
        "moe_act_bytes": sum(moe_layer_act_bytes),
        "max_layer_act_bytes": max(layer_act_bytes, default=0.0),
        "layer_flops": layer_flops,
    }


def checkpoint_state_split(data: dict) -> tuple[float, float]:
    """Split master weights and optimizer state between MoE and backbone."""
    groups = data["training_groups"]
    moe_groups = [
        group for name, group in groups.items()
        if "expert" in name.lower() or "moe" in name.lower()
    ]
    if moe_groups:
        moe_state = sum(
            group.get("master_weights", 0.0) + group.get("optimizer_states", 0.0)
            for group in moe_groups
        )
        return moe_state, data["master_weight_bytes"] + data["optimizer_state_bytes"] - moe_state

    if groups:
        remaining_moe_weight = data["moe_weight_bytes"]
        moe_state = 0.0
        for name, group in sorted(
            groups.items(),
            key=lambda item: (
                "expert" not in item[0].lower() and "moe" not in item[0].lower(),
                -item[1].get("weights", 0.0),
            ),
        ):
            group_weight = group.get("weights", 0.0)
            group_state = group.get("master_weights", 0.0) + group.get("optimizer_states", 0.0)
            overlap = min(remaining_moe_weight, group_weight)
            if group_weight > 0:
                moe_state += group_state * overlap / group_weight
                remaining_moe_weight -= overlap
            if remaining_moe_weight <= 0:
                break
        total_state = data["master_weight_bytes"] + data["optimizer_state_bytes"]
        return moe_state, total_state - moe_state

    total_state = data["master_weight_bytes"] + data["optimizer_state_bytes"]
    moe_state = (
        total_state * data["moe_weight_bytes"] / data["total_weight_bytes"]
        if data["total_weight_bytes"] > 0 else 0.0
    )
    return moe_state, total_state - moe_state


def strategies_for(k: int) -> list[dict]:
    strategies = []
    for ep in EP_VALUES:
        for efsdp in EFSDP_VALUES:
            strategies.append({
                "ep": ep,
                "efsdp": efsdp,
                "edp": k // (ep * efsdp),
                "valid": ep < k and k // (ep * efsdp) > 0,
            })
    return strategies


def analyze_strategy(
    data: dict,
    k: int,
    strategy: dict,
    hbm_gb: int,
    bf16_tflops: float = 1769.0,
) -> dict:
    ep = strategy["ep"]
    efsdp = strategy["efsdp"]
    moe_per_card = data["moe_weight_bytes"] / ep / efsdp * 2
    backbone_per_card = data["backbone_weight_bytes"] / k * 2
    static_hbm = (
        data["max_layer_act_bytes"]
        + moe_per_card
        + backbone_per_card
        + BUFFER_GB * GB
    )
    hbm_remaining = hbm_gb * GB * HBM_SAFETY - static_hbm
    opt_master_per_card = (
        data["moe_weight_bytes"] * 2 / ep / efsdp
        + data["backbone_weight_bytes"] * 2 / k
    ) * 2
    moe_checkpoint_state, backbone_checkpoint_state = checkpoint_state_split(data)
    checkpoint_moe_per_card = (
        data["moe_weight_bytes"] + moe_checkpoint_state
    ) / ep / efsdp
    checkpoint_backbone_per_card = (
        data["backbone_weight_bytes"] + backbone_checkpoint_state
    ) / k
    ddr_requirement = (
        opt_master_per_card + data["total_act_bytes"] - hbm_remaining
    )

    moe_offload = (
        min(1.0, max(0.0, (data["moe_act_bytes"] - hbm_remaining) / data["moe_act_bytes"]))
        if data["moe_act_bytes"] > 0 else 0.0
    )
    full_offload = (
        min(1.0, max(0.0, (data["total_act_bytes"] - hbm_remaining) / data["total_act_bytes"]))
        if data["total_act_bytes"] > 0 else 0.0
    )

    layer_times_s = [
        flops / (bf16_tflops * 1e12 * MFU)
        for flops in data["layer_flops"]
    ]
    d2h_moe = 0.0
    d2h_full = 0.0
    for i in range(len(data["layer_act_bytes"]) - 1):
        next_time = layer_times_s[i + 1]
        if next_time <= 0:
            continue
        d2h_moe = max(
            d2h_moe,
            data["moe_layer_act_bytes"][i] * moe_offload / next_time / GB,
        )
        d2h_full = max(
            d2h_full,
            data["layer_act_bytes"][i] * full_offload / next_time / GB,
        )

    bandwidth_benefits = {}
    for bandwidth in BANDWIDTHS_GBS:
        effective_bytes_s = bandwidth * 0.8 * GB
        cover_ratio = 1.0
        for i in range(len(data["layer_act_bytes"]) - 1):
            transfer_bytes = data["layer_act_bytes"][i] * full_offload
            if transfer_bytes <= 0:
                continue
            next_time = layer_times_s[i + 1]
            if next_time <= 0:
                continue
            transfer_time = transfer_bytes / effective_bytes_s
            cover_ratio = min(cover_ratio, min(1.0, next_time / transfer_time))
        bandwidth_benefits[bandwidth] = cover_ratio * 25.0

    activation_needed = data["total_act_bytes"] - hbm_remaining
    ddr_benefits = {}
    for capacity in DDR_CAPACITIES_GB:
        available_ddr = max(0.0, capacity * GB - opt_master_per_card)
        if activation_needed <= 0 or available_ddr >= activation_needed:
            offload_ratio = 1.0
        else:
            offload_ratio = max(0.0, 1.0 - (activation_needed - available_ddr) / activation_needed)
        ddr_benefits[capacity] = offload_ratio * 25.0, offload_ratio

    return {
        "moe_per_card_gb": moe_per_card / GB,
        "backbone_per_card_gb": backbone_per_card / GB,
        "static_hbm_gb": static_hbm / GB,
        "hbm_remaining_gb": hbm_remaining / GB,
        "ddr_requirement_gb": ddr_requirement / GB,
        "moe_offload": moe_offload,
        "full_offload": full_offload,
        "opt_master_gb": opt_master_per_card / GB,
        "d2h_moe_gbps": d2h_moe,
        "d2h_full_gbps": d2h_full,
        "bandwidth_benefits": bandwidth_benefits,
        "ddr_benefits": ddr_benefits,
        "ssd_moe_checkpoint_gb": checkpoint_moe_per_card / GB,
        "ssd_backbone_checkpoint_gb": checkpoint_backbone_per_card / GB,
        "ssd_gb": (checkpoint_moe_per_card + checkpoint_backbone_per_card) / GB,
    }


def row_identity(strategy: dict, k: int) -> str:
    return (
        f"| EP={strategy['ep']} | {k:,} | {strategy['edp']:,} | "
        f"{strategy['efsdp']} |"
    )


def make_report(
    model_name: str,
    table_params_t: float,
    k: int,
    config_name: str | None,
    hardware: str = "970DT",
    bf16_tflops: float = 1769.0,
) -> str:
    if config_name is None:
        return (
            f"# EP/FSDP 内存分析 — {hardware}\n\n"
            f"## {model_name} ({table_params_t:g}T params) — K={k:,}\n\n"
            "**未生成数值结果：** 仓库中没有与 Qwen3.5-397B 对应的模型结构配置。"
            "现存 `qwen35_35b_config.json` 的 CLI 模型族为 DSV41Flash，且仅约 0.034T "
            "参数，不能作为 0.397T Qwen3.5 的结构代替。需要补充正确配置后才能按同一公式计算。\n"
        )

    report = load_report(config_name)
    data = extract_model_data(report)
    lines = [
        f"# EP/FSDP 内存分析 — {hardware}",
        "",
        f"## {model_name} ({table_params_t:g}T params) — K={k:,}",
        "",
        f"- 配置：[`{config_name}`](../../configs/{config_name})",
        f"- 配置实际参数量：{data['config_params'] / 1e12:.4f}T；层数：{data['layers']}",
        f"- 输入：seq_len=4096，batch_size=1；BF16={bf16_tflops:g} TFLOPS，MFU=0.56",
        "- 静态HBM按公式包含最大单层激活、权重与梯度及 3GB buffer；HBM预算为容量的 95%。",
    ]
    if abs(data["config_params"] / 1e12 - table_params_t) > 0.01:
        lines += [
            f"- **参数量不匹配：** 卡数表为 {table_params_t:g}T，但配置实际为 "
            f"{data['config_params'] / 1e12:.4f}T；以下结构内存和激活计算使用配置数据，K 仍使用卡数表。",
        ]
    lines.append("")

    strategies = strategies_for(k)
    analyzed_by_hbm = {}
    for hbm_gb in HBM_CAPACITIES_GB:
        rows = []
        for strategy in strategies:
            if strategy["valid"]:
                result = analyze_strategy(data, k, strategy, hbm_gb, bf16_tflops)
                analyzed_by_hbm[(hbm_gb, strategy["ep"], strategy["efsdp"])] = result
            else:
                result = None
            rows.append((strategy, result))

        lines.extend([
            f"### HBM={hbm_gb}GB",
            "",
            "#### 任务一: DDR容量诉求 / 任务二: D2H带宽诉求",
            "",
            "| EP | FSDP=K | EDP | EFSDP | 单卡MoE(GB) | 单卡BB(GB) | 静态HBM(GB) | HBM剩余(GB) | DDR诉求(GB) | MoE卸载% | 全卸载% | D2H-MoE(GB/s) | D2H-Full(GB/s) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for strategy, result in rows:
            identity = row_identity(strategy, k)
            if result is None:
                values = ["—"] * 8 + ["— ⚠️ 无效"]
                lines.append(identity + " " + " | ".join(values) + " |")
                continue
            hbm = result["hbm_remaining_gb"]
            ddr = result["ddr_requirement_gb"]
            hbm_marker = " ⚠️" if hbm < 0 else " ✓"
            ddr_marker = " ⚠️" if ddr > 0 else " ✓"
            values = [
                f"{result['moe_per_card_gb']:.2f}",
                f"{result['backbone_per_card_gb']:.3f}",
                f"{result['static_hbm_gb']:.2f}",
                f"{hbm:+.2f}{hbm_marker}",
                f"{ddr:+.2f}{ddr_marker}",
                f"{result['moe_offload'] * 100:.1f}%",
                f"{result['full_offload'] * 100:.1f}%",
                f"{result['d2h_moe_gbps']:.2f}",
                f"{result['d2h_full_gbps']:.2f}",
            ]
            lines.append(identity + " " + " | ".join(values) + " |")
        lines += [
            "",
            "注：HBM剩余为负标 ⚠️，DDR诉求为正标 ⚠️。无效策略违反 EP < K 或 EDP > 0 约束。",
            "",
            "#### 任务三: D2H带宽遍历 → 端到端收益 (不考虑DDR)",
            "",
            "| EP | FSDP=K | EDP | EFSDP | " + " | ".join(
                f"{bw}G→{bw * 0.8:.0f}G" for bw in BANDWIDTHS_GBS
            ) + " |",
            "|---|---:|---:|---:|" + "---:|" * len(BANDWIDTHS_GBS),
        ]
        for strategy in strategies:
            identity = row_identity(strategy, k)
            result = analyzed_by_hbm.get((hbm_gb, strategy["ep"], strategy["efsdp"]))
            if result is None:
                values = ["—"] * (len(BANDWIDTHS_GBS) - 1) + ["— ⚠️ 无效"]
                lines.append(identity + " " + " | ".join(values) + " |")
            else:
                values = [f"{result['bandwidth_benefits'][bw]:.2f}%" for bw in BANDWIDTHS_GBS]
                lines.append(identity + " " + " | ".join(values) + " |")
        lines += [
            "",
            "#### 任务四: DDR容量遍历 → 端到端收益 (不考虑D2H带宽)",
            "",
            "| EP | FSDP=K | EDP | EFSDP | 优化器+主权重(GB) | "
            + " | ".join(f"DDR={capacity}GB" for capacity in DDR_CAPACITIES_GB)
            + " |",
            "|---|---:|---:|---:|---:|" + "---:|" * len(DDR_CAPACITIES_GB),
        ]
        for strategy in strategies:
            identity = row_identity(strategy, k)
            result = analyzed_by_hbm.get((hbm_gb, strategy["ep"], strategy["efsdp"]))
            if result is None:
                values = ["—"] * (len(DDR_CAPACITIES_GB) - 1) + ["— ⚠️ 无效"]
                lines.append(identity + " — | " + " | ".join(values) + " |")
            else:
                values = [
                    f"{benefit:.2f}% ({offload * 100:.0f}%)"
                    for benefit, offload in (
                        result["ddr_benefits"][capacity]
                        for capacity in DDR_CAPACITIES_GB
                    )
                ]
                lines.append(
                    identity + f" {result['opt_master_gb']:.2f} | "
                    + " | ".join(values) + " |"
                )
        lines += [
            "",
            "注：括号内为 offload_ratio；端到端收益 = offload_ratio × 25%。",
            "",
        ]

    lines += [
        "### 任务五: 单卡Checkpoint分片 SSD 容量诉求",
        "",
        "单卡Checkpoint分片 = (MoE权重 + 对应master weight与optimizer state)/EP/EFSDP "
        "+ (Backbone权重 + 对应master weight与optimizer state)/FSDP；不计训练梯度。该值不依赖HBM容量。",
        "",
        "| EP | FSDP=K | EDP | EFSDP | 单卡MoE Checkpoint(GB) | 单卡BB Checkpoint(GB) | 单卡SSD分片(GB) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for strategy in strategies:
        identity = row_identity(strategy, k)
        result = analyzed_by_hbm.get((HBM_CAPACITIES_GB[0], strategy["ep"], strategy["efsdp"]))
        if result is None:
            lines.append(identity + " — | — | ⚠️ 无效 |")
            continue
        lines.append(
            identity
            + f" {result['ssd_moe_checkpoint_gb']:.2f} | "
            + f"{result['ssd_backbone_checkpoint_gb']:.3f} | {result['ssd_gb']:.2f} |"
        )
    lines += [
        "",
        "注：这是每GPU的checkpoint分片估算；训练状态按配置中的master weights和optimizer states分配，不含梯度。节点SSD需求需乘以每节点GPU数。",
        "",
    ]
    return "\n".join(lines)


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def main() -> None:
    out_dir = PROJECT_ROOT / "reports_analyse" / "ep_fsdp_mem_hbm_d2h"
    out_dir.mkdir(parents=True, exist_ok=True)
    hardware_specs = [("970DT", 1769.0, 0), ("980DT", 3538.0, 1)]
    for name, params_t, card_counts, config_name in MODELS:
        for hardware, bf16_tflops, card_count_index in hardware_specs:
            k = card_counts[card_count_index]
            text = make_report(
                name,
                params_t,
                k,
                config_name,
                hardware=hardware,
                bf16_tflops=bf16_tflops,
            )
            output = out_dir / f"{slugify(name)}_{hardware}.md"
            output.write_text(text, encoding="utf-8")
            print(f"Wrote {output.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
