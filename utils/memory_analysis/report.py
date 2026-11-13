#!/usr/bin/env python3
"""Generate readable report from task results."""
import json
from pathlib import Path

# Re-run analysis and format
exec(Path("utils/memory_analysis/run_tasks.py").read_text())

# ============================================================================
# Print Report
# ============================================================================

print("=" * 100)
print("  DSV4Pro EP/FSDP 内存分析报告")
print("  硬件: A7 (BF16 1769 TFLOPS)  |  HBM: 288GB/卡  |  总卡数: 8192")
print("  模型: DSV4Pro (1.57T params)  |  seq_len=4096  |  batch_size=1")
print("=" * 100)

for s in STRATEGIES:
    r = results[STRATEGIES.index(s)]
    print(f"\n{'─'*100}")
    print(f"  {s['name']}: EP={s['EP']}, EDP={s['EDP']}, FSDP={s['FSDP']}")
    print(f"{'─'*100}")

    # --- 模型分割 ---
    print(f"\n  ▸ 模型权重分割")
    print(f"    MoE(BF16) 总量:         {moe_bf16/GB:>10.2f} GB")
    print(f"    BackBone(BF16) 总量:     {backbone_bf16/GB:>10.2f} GB")
    print(f"    总权重(BF16):            {total_weight_bytes/GB:>10.2f} GB")

    # --- 任务一: DDR 容量诉求 ---
    print(f"\n  ▸ 任务一: DDR 容量诉求")
    print(f"    单卡MoE(BF16+梯度):      {r['moe_per_card_gb']:>10.2f} GB  (= MoE/EP*2)")
    print(f"    单卡BackBone(BF16+梯度):  {r['backbone_per_card_gb']:>10.2f} GB  (= Backbone/FSDP*2)")
    print(f"    最大层激活:               {r['max_layer_act_gb']:>10.4f} GB")
    print(f"    3GB buffer:               {'3.00':>10s} GB")
    print(f"    ─────────────────────────────────────")
    print(f"    静态HBM占用:              {r['static_hbm_gb']:>10.2f} GB")
    print(f"    HBM 95% 可用:             {r['hbm_budget_gb']:>10.2f} GB")
    print(f"    HBM剩余:                  {r['hbm_remaining_gb']:>10.2f} GB")
    print(f"    ─────────────────────────────────────")
    print(f"    全激活占用空间:           {total_act_bytes/GB:>10.2f} GB")
    print(f"    MoE激活占用空间:          {moe_act_total/GB:>10.2f} GB")
    print(f"    优化器+主权重(单卡):      {r['opt_master_per_card_gb']:>10.2f} GB")
    print(f"    ─────────────────────────────────────")
    print(f"    DDR容量诉求:              {r['ddr_requirement_gb']:>10.2f} GB  {'← 无需DDR!' if r['ddr_requirement_gb'] < 0 else '← 需要DDR'}")

    # --- 任务二: D2H 带宽诉求 ---
    print(f"\n  ▸ 任务二: D2H 带宽诉求")
    print(f"    MoE激活卸载比例:          {r['moe_offload_ratio']:>10.4f}  (0=无需卸载)")
    print(f"    全激活卸载比例:           {r['full_offload_ratio']:>10.4f}  (0=无需卸载)")
    print(f"    D2H诉求(MoE only):        {r['d2h_moe_gbps']:>10.2f} GB/s")
    print(f"    D2H诉求(Full offload):    {r['d2h_full_gbps']:>10.2f} GB/s")

    # --- 任务三: 带宽遍历 (不考虑DDR) ---
    print(f"\n  ▸ 任务三: D2H带宽遍历 (不考虑DDR容量)")
    print(f"    {'带宽(GB/s)':>10s}  {'实际带宽(GB/s)':>15s}  {'覆盖比':>8s}  {'端到端收益':>10s}")
    print(f"    {'─'*10}  {'─'*15}  {'─'*8}  {'─'*10}")
    for bw, v in r['bandwidth_benefits'].items():
        print(f"    {bw:>10}  {v['effective_gbps']:>15.1f}  {v['cover_ratio']:>8.4f}  {v['benefit_pct']:>9.2f}%")

    # --- 任务四: DDR遍历 (不考虑D2H) ---
    print(f"\n  ▸ 任务四: DDR容量遍历 (不考虑D2H带宽)")
    print(f"    优化器+主权重(单卡): {r['opt_master_per_card_gb']:.2f} GB (常驻DDR)")
    print(f"    激活需放入DDR:        {max(0, total_act_bytes/GB - r['hbm_remaining_gb']):.2f} GB")
    print(f"    {'DDR容量(GB)':>12s}  {'可用DDR(GB)':>12s}  {'卸载比例':>8s}  {'端到端收益':>10s}")
    print(f"    {'─'*12}  {'─'*12}  {'─'*8}  {'─'*10}")
    for ddrc, v in r['ddr_benefits'].items():
        avail = max(0, ddrc - r['opt_master_per_card_gb'])
        print(f"    {ddrc:>12}  {avail:>12.2f}  {v['offload_ratio']:>8.4f}  {v['benefit_pct']:>9.2f}%")

# Summary
print(f"\n{'='*100}")
print("  结论")
print(f"{'='*100}")
print(f"""
  在 8192 卡 × 288GB HBM 配置下，DSV4Pro (1.57T params) 的三个 EP/EDP/FSDP
  并行策略均无需激活卸载：

  • 策略一 (EP=128):  HBM剩余 244.59 GB, DDR诉求 -91.14 GB (无需DDR)
  • 策略二 (EP=256):  HBM剩余 256.72 GB, DDR诉求 -127.53 GB (无需DDR)
  • 策略三 (EP=512):  HBM剩余 262.78 GB, DDR诉求 -145.72 GB (无需DDR)

  原因: 8192 张卡分摊了大部分 MoE 参数内存，BackBone 仅 43.9 GB 除以 8192
  后几乎为零，导致单卡静态 HBM 仅 10.8~29.0 GB，远低于 288GB*0.95=273.6GB。

  激活总量 (104.9 GB) 完全可放入 HBM 剩余空间，无需任何 D2H 卸载或 DDR 存储。
  因此在所有 D2H 带宽和 DDR 容量遍历中，端到端收益均为 25%（即重计算被完全消除）。

  注: batch_size=1 且 seq_len=4096 时激活较小；若增大 batch_size 或使用长序列，
  激活显存将显著增长，届时 HBM 剩余可能不足以容纳全部激活，需要 D2H/DDR 卸载。
""")
print("=" * 100)