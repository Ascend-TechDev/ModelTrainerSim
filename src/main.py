#!/usr/bin/env python3
"""ModelTrainerSimulator CLI entry point.

Usage:
    python -m src.main --config configs/dsv4pro_config.json [options]

Options:
    --config PATH          Path to model config.json
    --seq-len N            Override sequence length
    --batch-size N         Override batch size
    --format text|json     Output format (default: text)
    --output PATH          Write report to file
    --list-models          List registered model families
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .core.config_parser import ConfigParser
from .core.model_registry import ModelRegistry
from .core.reporter import render_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ModelTrainerSimulator",
        description="LLM architecture parameter & memory modeling toolkit",
    )
    parser.add_argument(
        "--config", "-c", type=str,
        help="Path to model config.json")
    parser.add_argument(
        "--seq-len", "-s", type=int, default=None,
        help="Override sequence length")
    parser.add_argument(
        "--batch-size", "-b", type=int, default=1,
        help="Batch size (default: 1)")
    parser.add_argument(
        "--format", "-f", choices=["text", "json"], default="text",
        help="Output format (default: text)")
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Write report to file")
    parser.add_argument(
        "--list-models", action="store_true",
        help="List registered model families and exit")

    args = parser.parse_args(argv)

    # List models
    if args.list_models:
        models = ModelRegistry.list_models()
        print("Registered model families:")
        for m in models:
            print(f"  - {m}")
        return 0

    if not args.config:
        parser.error("--config is required (unless --list-models)")

    # Load config
    try:
        cfg = ConfigParser(args.config).load()
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Config error: {e}", file=sys.stderr)
        return 1

    # Build model
    try:
        model = ModelRegistry.build(cfg["model"], cfg)
    except KeyError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # Analyze
    seq_len = args.seq_len or cfg.get("max_seq_len", 4096)
    report = model.analyze(seq_len=seq_len, batch_size=args.batch_size)

    # Output
    if args.format == "json":
        output = json.dumps(report, indent=2, default=str)
    else:
        output = render_report(report)

    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
        print(f"Report written to: {args.output}")
    else:
        print(output)

    return 0


if __name__ == "__main__":
    sys.exit(main())