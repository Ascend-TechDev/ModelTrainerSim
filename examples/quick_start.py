#!/usr/bin/env python3
"""Quick-start example for ModelTrainerSimulator.

Demonstrates:
  1. Loading a config
  2. Building a model
  3. Running analysis
  4. Printing a formatted report
  5. Accessing raw data programmatically
"""

import sys
from pathlib import Path

# Ensure src is importable when run directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.config_parser import ConfigParser
from src.core.formatter import format_bytes, format_flops, format_number
from src.core.model_registry import ModelRegistry
from src.core.reporter import render_report


def main() -> None:
    # 1. Load config
    config_path = Path(__file__).resolve().parent.parent / "configs" / "dsv4pro_config.json"
    config = ConfigParser(config_path).load()

    print(f"Model: {config['model']}")
    print(f"Hidden size: {config['hidden_size']}")
    print(f"Layers: {config['num_layers']}")
    print()

    # 2. Build model
    model = ModelRegistry.build(config["model"], config)

    # 3. Analyze at different sequence lengths
    for seq_len in [4096, 32768, 131072]:
        report = model.analyze(seq_len=seq_len, batch_size=1)

        p = report["model_params"]
        m = report["memory"]
        c = report["compute"]

        print(f"── seq_len={format_number(seq_len)} ────────────────────────")
        print(f"  Total params       : {format_number(p['total_params'])}")
        print(f"  Active params/token : {format_number(p['active_params_per_token'])}")
        print(f"  Weight memory       : {format_bytes(m['weight_memory_bytes'])}")
        print(f"  KV cache            : {format_bytes(m['kv_cache_bytes'])}")
        print(f"  Activations (fwd)  : {format_bytes(m['activation_memory_bytes'])}")
        print(f"  Training (w+g+opt)  : {format_bytes(m['training_total_bytes'])}")
        print(f"  FLOPs/token         : {format_flops(c['flops_per_token'])}")
        print()

    # 4. Full text report
    print()
    full_report = model.analyze(seq_len=4096, batch_size=1)
    print(render_report(full_report))


if __name__ == "__main__":
    main()