#!/usr/bin/env python3
"""Hy4-10T 三种分片策略（EP / FSDP / EFSDP）对比脚本。

一键运行三种策略场景，打印：
  * 每卡静态显存（权重 / 梯度 / 优化器 / 小计）
  * 每卡前向激活（B = mbs 全模型清单口径）与 KV cache 参考值
  * 与全模型总量（未分片）的对照

用法（在仓库根目录）:
    python examples/hy4_strategies.py [seq_len]

注意：激活/KV 为全模型口径（未按 rank 划分、未含 EP all-to-all /
FSDP all-gather 通信量），口径细节见打印区下方的说明。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.config_parser import ConfigParser
from src.core.formatter import format_bytes
from src.core.model_registry import ModelRegistry

GIB = 2 ** 30

STRATEGIES = [
    ("EP     (专家÷64, 稠密复制)", "configs/hy4_ep_config.json"),
    ("FSDP   (全参数÷1024)", "configs/hy4_fsdp_config.json"),
    ("EFSDP  (专家÷64, 稠密÷16)", "configs/hy4_efsdp_config.json"),
]


def main() -> None:
    seq_len = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
    rows = []
    whole = None
    for label, path in STRATEGIES:
        cfg = ConfigParser(path).load()
        report = ModelRegistry.build("Hy4", cfg).analyze(seq_len, 1)
        dev = report["memory"]["training_breakdown"]["per_device"]["gpu"]
        groups = cfg["training"]["groups"]
        rows.append((label, groups["dense"]["shards"], groups["expert"]["shards"],
                     dev, report["memory"]))
        if whole is None:
            whole = report["memory"]["training_breakdown"]

    print("=" * 100)
    print(f"  Hy4-10T 分片策略对比（每卡静态显存，seq_len={seq_len}, 1024 卡）")
    print("=" * 100)
    header = (f"  {'策略':<30}{'稠密shards':>10}{'专家shards':>10}"
              f"{'权重':>12}{'梯度':>12}{'优化器':>12}{'静态小计':>12}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for label, ds, es, dev, mem in rows:
        print(f"  {label:<30}{ds:>10}{es:>10}"
              f"{format_bytes(dev['weights']):>12}{format_bytes(dev['gradients']):>12}"
              f"{format_bytes(dev['optimizer_states']):>12}"
              f"{format_bytes(dev['total']):>12}")
    print()
    print("  全模型总量（未分片对照）:")
    print(f"    权重 {format_bytes(whole['weights'])}   "
          f"梯度 {format_bytes(whole['gradients'])}   "
          f"优化器 {format_bytes(whole['optimizer_states'])}   "
          f"合计 {format_bytes(whole['total'])}")
    print()
    print("  口径说明:")
    print("    - 权重/梯度: 稠密模块 ÷ shards_dense，路由专家 ÷ shards_expert；")
    print("      router bias 只计权重、不计梯度/优化器（workbook D19-D21 口径）。")
    print("    - 优化器: adamw fp32 m+v = 8B/参数，master 权重 0。")
    print("    - 前向激活与 KV cache 为全模型 B=mbs 口径，未按 rank 划分；")
    print("      EP 的 all-to-all token 分布与 FSDP all-gather 通信量未计入。")
    print("    - 每种策略每卡前向激活参考值（B=1 全模型口径）:")
    for label, _, _, dev, mem in rows:
        print(f"      {label:<30} 激活 {format_bytes(mem['activation_memory_bytes'])}"
              f"  KV {format_bytes(mem['kv_cache_bytes'])}")
    print("=" * 100)


if __name__ == "__main__":
    main()
