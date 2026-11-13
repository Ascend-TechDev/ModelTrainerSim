#!/usr/bin/env python3
"""Generate readable markdown report for hy4 analysis results."""
import json, math, sys

data = json.loads(open("/tmp/hy4_results.json").read())

HBM_LABELS = ["HBM=192GB", "HBM=288GB"]
EP_LABELS = ["EP=128", "EP=256", "EP=512", "EP=1024"]
BW_KEYS = [200, 250, 300, 350, 400, 450, 500]
DDR_KEYS = [256, 384, 512, 540, 768]

def ep_val(label):
    return int(label.split("=")[1])

print("# Hy4 Family EP/FSDP Memory Analysis — 970DT")
print()
print("**Hardware**: A7 (BF16 1769 TFLOPS, MFU=0.56) | seq_len=4096 | batch_size=1")
print()

for model_name in data:
    print(f"## {model_name}")
    print()

    for hbm_label in HBM_LABELS:
        hbm_data = data[model_name][hbm_label]

        # ---- Task 1 & 2: Summary Table ----
        print(f"### {hbm_label}")
        print()
        print("#### 任务一: DDR容量诉求 / 任务二: D2H带宽诉求")
        print()
        print("| EP | FSDP=K | EDP | EFSDP | 单卡MoE(GB) | 单卡BB(GB) | 静态HBM(GB) | HBM剩余(GB) | DDR诉求(GB) | MoE卸载% | 全卸载% | D2H-MoE(GB/s) | D2H-Full(GB/s) |")
        print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for ep_label in EP_LABELS:
            r = hbm_data[ep_label]
            ep = ep_val(ep_label)
            K = r['fsdp']
            edp = r['edp']
            efsdp = r['efsdp']
            moe_off = f"{r['moe_offload']*100:.1f}%"
            full_off = f"{r['full_offload']*100:.1f}%"
            hbm_rem = r['hbm_remaining_gb']
            hbm_flag = " ⚠️" if hbm_rem < 0 else " ✓"
            ddr_flag = " ⚠️" if r['ddr_req_gb'] > 0 else " ✓"
            print(f"| EP={ep} | {K:,} | {edp:,} | {efsdp:,} | {r['moe_per_card_gb']:.1f} | {r['backbone_per_card_gb']:.3f} | {r['static_hbm_gb']:.1f} | {hbm_rem:+.1f}{hbm_flag} | {r['ddr_req_gb']:+.1f}{ddr_flag} | {moe_off} | {full_off} | {r['d2h_moe_gbps']:.1f} | {r['d2h_full_gbps']:.1f} |")
        print()

        # ---- Task 3: Bandwidth Sweep ----
        print("#### 任务三: D2H带宽遍历 → 端到端收益 (不考虑DDR)")
        print()
        header = "| EP | FSDP=K | EDP | EFSDP |" + "|".join(f" {bw}G→{bw*0.8:.0f}G " for bw in BW_KEYS) + "|"
        sep = "|---|---:|---:|---:|" + "|".join("---:" for _ in BW_KEYS) + "|"
        print(header)
        print(sep)
        for ep_label in EP_LABELS:
            r = hbm_data[ep_label]
            ep = ep_val(ep_label)
            K = r['fsdp']
            edp = r['edp']
            efsdp = r['efsdp']
            vals = "|".join(f" {r['bandwidth_sweep'][str(bw)]['benefit']:.2f}% " for bw in BW_KEYS)
            print(f"| EP={ep} | {K:,} | {edp:,} | {efsdp:,} |{vals}|")
        print()

        # ---- Task 4: DDR Capacity Sweep ----
        print("#### 任务四: DDR容量遍历 → 端到端收益 (不考虑D2H带宽)")
        print()
        print(f"| EP | FSDP=K | EDP | EFSDP | 优化器+主权重(GB) |" + "|".join(f" DDR={d}GB " for d in DDR_KEYS) + "|")
        print(f"|---|---:|---:|---:|:---:|" + "|".join("---:" for _ in DDR_KEYS) + "|")
        for ep_label in EP_LABELS:
            r = hbm_data[ep_label]
            ep = ep_val(ep_label)
            K = r['fsdp']
            edp = r['edp']
            efsdp = r['efsdp']
            opt_ = f"{r['opt_master_per_card_gb']:.1f}"
            vals = "|".join(f" {r['ddr_sweep'][str(d)]['benefit']:.2f}% ({r['ddr_sweep'][str(d)]['offload']*100:.0f}%)" for d in DDR_KEYS)
            print(f"| EP={ep} | {K:,} | {edp:,} | {efsdp:,} | {opt_} |{vals}|")
        print()

    print("---")
    print()

print("---")
print()
print("**说明**:")
print("- ⚠️ HBM剩余为负 = 需激活卸载; DDR诉求为正 = 需DDR支持")
print("- 任务三 & 四中, '无需卸载' 时 benefit=25% (重计算 100% 被消除), '完全无法卸载' 时 benefit=0%")
print("- 任务四括号内为 offload_ratio; 端到端收益 = offload_ratio x 25%")