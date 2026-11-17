#!/usr/bin/env python3
"""ModelTrainerSimulator CLI entry point.

Usage:
    python -m src.main --config configs/dsv4pro_config.json [options]

Options:
    --config PATH          Path to model config.json
    --seq-len N            Override sequence length
    --batch-size N         Override batch size
    --weight-dtype TYPE    Override model weight precision
    --activation-dtype TYPE Override activation precision
    --format text|json     Output format (default: text)
    --output PATH          Write report to file
    --list-models          List registered model families
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .core.base_module import BaseModule
from .core.config_parser import ConfigParser
from .core.dtype import DType, get_dtype_size
from .core.model_registry import ModelRegistry
from .core.reporter import render_report


def _apply_weight_dtype(config: dict, dtype: str) -> None:
    weight_keys = {"dtype", "weight_dtype", "expert_dtype", "gate_dtype",
                   "moe_router_dtype"}

    def override(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if key in weight_keys:
                    value[key] = dtype
                else:
                    override(child)
        elif isinstance(value, list):
            for child in value:
                override(child)

    override(config)
    config["dtype"] = dtype


def _preserve_activation_dtypes(config: dict) -> None:
    default_dtype = config.get("activation_dtype", config.get("dtype", "bf16"))

    def preserve(value):
        if isinstance(value, dict):
            if "activation_dtype" not in value:
                if "expert_dtype" in value:
                    value["activation_dtype"] = value["expert_dtype"]
                elif "dtype" in value:
                    value["activation_dtype"] = value["dtype"]
            for child in value.values():
                preserve(child)
        elif isinstance(value, list):
            for child in value:
                preserve(child)

    preserve(config)
    config.setdefault("activation_dtype", default_dtype)


def _apply_activation_dtype(config: dict, dtype: str) -> None:
    config["activation_dtype"] = dtype
    if "bytes" in config:
        config["bytes"]["activation"] = get_dtype_size(DType(dtype))

    def override(value):
        if isinstance(value, dict):
            if "dtype" in value or "expert_dtype" in value:
                value["activation_dtype"] = dtype
            for child in value.values():
                override(child)
        elif isinstance(value, list):
            for child in value:
                override(child)

    override(config)


def _set_model_activation_dtype(model, dtype: str, force: bool = False) -> None:
    activation_dtype = DType(dtype)
    visited = set()

    def visit(value):
        if id(value) in visited:
            return
        visited.add(id(value))
        if isinstance(value, BaseModule):
            value.activation_dtype = DType(
                dtype if force else value.config.get(
                    "activation_dtype", activation_dtype.value))
            return
        if isinstance(value, dict):
            if force and ("dtype" in value or "expert_dtype" in value):
                value["activation_dtype"] = activation_dtype.value
            elif "dtype" in value or "expert_dtype" in value:
                value.setdefault("activation_dtype", activation_dtype.value)
            for child in value.values():
                visit(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child)
        elif hasattr(value, "__dict__"):
            for child in vars(value).values():
                visit(child)

    model.activation_dtype = activation_dtype
    for child in vars(model).values():
        visit(child)


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
        "--weight-dtype", choices=["fp32", "fp16", "bf16", "fp8"],
        default=None, help="Override model weight precision")
    parser.add_argument(
        "--activation-dtype", choices=["fp32", "fp16", "bf16", "fp8"],
        default=None, help="Override activation precision")
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
    if args.seq_len is not None and args.seq_len <= 0:
        parser.error("--seq-len must be positive")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")

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
        activation_dtype = args.activation_dtype or cfg.get(
            "activation_dtype", cfg.get("dtype", "bf16"))
        if args.activation_dtype:
            _apply_activation_dtype(cfg, activation_dtype)
        elif args.weight_dtype:
            _preserve_activation_dtypes(cfg)
        if args.weight_dtype:
            _apply_weight_dtype(cfg, args.weight_dtype)
        model = ModelRegistry.build(cfg["model"], cfg)
        if args.activation_dtype or args.weight_dtype:
            _set_model_activation_dtype(
                model, activation_dtype, force=bool(args.activation_dtype))
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
