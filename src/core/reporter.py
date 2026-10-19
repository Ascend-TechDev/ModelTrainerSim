"""Shared report layout for model analysis; model adapters only prepare rows."""
from __future__ import annotations

from typing import Any, Callable

from .formatter import format_bytes, format_flops, format_number

REPORT_WIDTH = 91
MODULE_ORDER = ["wordembedding", "gate", "routedexpert", "sharedexpert", "swa", "csa", "hca", "mhc"]


def section(title: str) -> str:
    prefix = f"── {title} "
    return prefix + "─" * max(1, REPORT_WIDTH - len(prefix))


def render_report(report: dict[str, Any]) -> str:
    if report["model_family"] == "DSV41Flash":
        from ..models.dsv41flash.modules.report import render_flash_report
        return render_flash_report(report)
    return render_standard_report(report)


def render_standard_report(
    report: dict[str, Any],
    *,
    byte_formatter: Callable[[float], str] = format_bytes,
    module_order: list[str] | None = None,
    extra_configuration: list[tuple[str, str]] | None = None,
    appendix: list[str] | None = None,
) -> str:
    """Render the same six sections for every family without altering its data."""
    cfg, p, c, m = (report[k] for k in ("config", "model_params", "compute", "memory"))
    fmt = byte_formatter
    lines = ["=" * REPORT_WIDTH, f"  Model Analysis Report: {report['model_family']}", "=" * REPORT_WIDTH, ""]

    def field(label, value):
        lines.append(f"  {label:<20}: {value}")

    lines.append(section("Configuration"))
    for key, value in (("hidden_size", cfg["hidden_size"]), ("num_layers", cfg["num_layers"]),
                       ("vocab_size", format_number(cfg["vocab_size"])),
                       ("max_seq_len", format_number(cfg["max_seq_len"])), ("dtype", cfg["dtype"])):
        field(key, value)
    for label, value in extra_configuration or []:
        field(label, value)

    lines.extend(["", section("Parameters")])
    field("Total params", format_number(p["total_params"]))
    field("Embedding params", format_number(p["embedding_params"]))
    field("Active params/token", format_number(p["active_params_per_token"]))
    field("Weight memory", fmt(p["total_weight_bytes"]))

    lines.extend(["", section("Compute (per forward pass)")])
    field("Seq length", format_number(c["seq_len"]))
    field("Batch size", c["batch_size"])
    field("Total FLOPs", format_flops(c["total_flops_per_sample"]))
    field("FLOPs / token", format_flops(c["flops_per_token"]))

    lines.extend(["", section("Memory")])
    field("Weights", fmt(m["weight_memory_bytes"]))
    field("Activations (fwd)", fmt(m["activation_memory_bytes"]))
    field("KV cache", fmt(m["kv_cache_bytes"]))
    field("Training (weights+grad+opt)", fmt(m["training_total_bytes"]))
    tb = m["training_breakdown"]
    for label, value in (("weights", tb["weights"]), ("gradients", tb["gradients"]),
                         ("master weights", tb["master_weights"]),
                         (f"optimizer ({tb['optimizer']})", tb["optimizer_states"])):
        lines.append(f"    - {label:<24}: {fmt(value)}")

    pms = report.get("per_module_summary_excel") or report["per_module_summary"]
    order = module_order or MODULE_ORDER
    ordered = [(k, pms[k]) for k in order if k in pms]
    ordered.extend((k, pms[k]) for k in sorted(pms) if k not in order)
    name_width = max(14, *(len(k) for k, _ in ordered)) if ordered else 14
    header = (f"  {'Module':<{name_width}} {'Count':>6} {'Params':>16} {'Bytes':>16} "
              f"{'FLOPs/token':>16} {'Act Bytes':>16}")
    lines.extend(["", section("Per-Module Summary"), header, "  " + "-" * (len(header) - 2)])
    tokens = c["seq_len"] * c["batch_size"]
    for name, row in ordered:
        per_token = row["flops"] / tokens if tokens > 0 else 0
        count = f"{row['count']:g}"
        lines.append(f"  {name:<{name_width}} {count:>6} {format_number(row['params']):>16} "
                     f"{fmt(row['bytes']):>16} {format_flops(per_token):>16} {fmt(row['act_bytes']):>16}")

    layers = report["layers"]
    attn_width = max(7, *(len(row["attention_type"]) for row in layers)) if layers else 7
    header = (f"  {'Layer':>5} {'Attn':>{attn_width}} {'FFN':>5} {'Idx':>5} "
              f"{'Params':>16} {'Bytes':>16} {'FLOPs':>16}")
    lines.extend(["", section("Per-Layer Summary"), header, "  " + "-" * (len(header) - 2)])
    selected = set(range(min(5, len(layers)))) | set(range(max(0, len(layers) - 3), len(layers)))
    represented = {layers[idx]["attention_type"] for idx in selected}
    for idx, row in enumerate(layers):
        if row["attention_type"] not in represented:
            selected.add(idx)
            represented.add(row["attention_type"])
    show = sorted(selected)
    previous = -1
    for idx in show:
        if previous >= 0 and idx != previous + 1:
            lines.append(f"  {'...':>5}")
        row = layers[idx]
        lines.append(f"  {row['layer_id']:>5} {row['attention_type']:>{attn_width}} {row['ffn_type']:>5} "
                     f"{str(row['use_indexer']):>5} {format_number(row['params']):>16} "
                     f"{fmt(row['bytes']):>16} {format_flops(row['flops']):>16}")
        previous = idx
    if appendix:
        lines.extend(appendix)
    lines.extend(["", "=" * REPORT_WIDTH])
    return "\n".join(lines)
