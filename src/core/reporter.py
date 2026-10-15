"""Report rendering utilities — pretty-print analysis results."""

from __future__ import annotations

from typing import Any

from .formatter import format_bytes, format_flops, format_number


def render_report(report: dict[str, Any]) -> str:
    """Render a full analysis report as a formatted string."""
    lines: list[str] = []
    family = report["model_family"]
    cfg = report["config"]

    lines.append("=" * 78)
    lines.append(f"  Model Analysis Report: {family}")
    lines.append("=" * 78)
    lines.append("")
    lines.append("── Configuration ──────────────────────────────────────────────")
    lines.append(f"  hidden_size       : {cfg['hidden_size']}")
    lines.append(f"  num_layers         : {cfg['num_layers']}")
    lines.append(f"  vocab_size         : {format_number(cfg['vocab_size'])}")
    lines.append(f"  max_seq_len        : {format_number(cfg['max_seq_len'])}")
    lines.append(f"  dtype              : {cfg['dtype']}")

    # Parameters
    p = report["model_params"]
    lines.append("")
    lines.append("── Parameters ──────────────────────────────────────────────────")
    lines.append(f"  Total params       : {format_number(p['total_params'])}")
    lines.append(f"  Embedding params   : {format_number(p['embedding_params'])}")
    lines.append(f"  Active params/token: {format_number(p['active_params_per_token'])}")
    lines.append(f"  Weight memory      : {format_bytes(p['total_weight_bytes'])}")

    # Compute
    c = report["compute"]
    lines.append("")
    lines.append("── Compute (per forward pass) ──────────────────────────────────")
    lines.append(f"  Seq length         : {format_number(c['seq_len'])}")
    lines.append(f"  Batch size         : {c['batch_size']}")
    lines.append(f"  Total FLOPs       : {format_flops(c['total_flops_per_sample'])}")
    lines.append(f"  FLOPs / token      : {format_flops(c['flops_per_token'])}")

    # Memory
    m = report["memory"]
    lines.append("")
    lines.append("── Memory ──────────────────────────────────────────────────────")
    lines.append(f"  Weights            : {format_bytes(m['weight_memory_bytes'])}")
    lines.append(f"  Activations (fwd)  : {format_bytes(m['activation_memory_bytes'])}")
    lines.append(f"  KV cache           : {format_bytes(m['kv_cache_bytes'])}")
    lines.append(f"  Training (weights+grad+opt): {format_bytes(m['training_total_bytes'])}")
    tb = m["training_breakdown"]
    lines.append(f"    - weights        : {format_bytes(tb['weights'])}")
    lines.append(f"    - gradients      : {format_bytes(tb['gradients'])}")
    lines.append(f"    - master weights : {format_bytes(tb['master_weights'])}")
    lines.append(f"    - optimizer ({tb['optimizer']:>8}): {format_bytes(tb['optimizer_states'])}")

    # Excel Per-Module Summary order (matches sheet rows 89-95):
    #   mHC, Gate, SWA, CSA, HCA, MoE — but we report wordembedding first,
    #   then the Excel-ordered rows (gate/routed/shared are MoE components).
    pms = report.get("per_module_summary_excel") or report["per_module_summary"]
    excel_order = ["wordembedding", "gate", "routedexpert", "sharedexpert",
                    "swa", "csa", "hca", "mhc"]
    if pms and all(k in pms for k in excel_order):
        ordered = [(k, pms[k]) for k in excel_order if k in pms]
    else:
        ordered = sorted(pms.items())
    lines.append("")
    lines.append("── Per-Module Summary ──────────────────────────────────────────")
    lines.append(f"  {'Module':<14} {'Count':>6} {'Params':>16} {'Bytes':>14} "
                  f"{'FLOPs/token':>16} {'Act Bytes':>14}")
    lines.append("  " + "-" * 76)
    for mod_type, s in ordered:
        flops_per_tok = s["flops"] / c["seq_len"] if c["seq_len"] > 0 else 0
        lines.append(
            f"  {mod_type:<14} {s['count']:>6} {format_number(s['params']):>16} "
            f"{format_bytes(s['bytes']):>14} {format_flops(flops_per_tok):>16} "
            f"{format_bytes(s['act_bytes']):>14}"
        )

    # Per-layer breakdown (first 5 + last 3)
    layers = report["layers"]
    lines.append("")
    lines.append("── Per-Layer Summary ───────────────────────────────────────────")
    lines.append(f"  {'Layer':>5} {'Attn':>5} {'FFN':>5} {'Idx':>5} "
                  f"{'Params':>16} {'Bytes':>14} {'FLOPs':>16}")
    lines.append("  " + "-" * 72)

    show_indices = list(range(min(5, len(layers)))) + list(
        range(max(0, len(layers) - 3), len(layers)))
    show_indices = sorted(set(show_indices))
    prev = -1
    for idx in show_indices:
        if idx != prev + 1 and prev >= 0:
            lines.append(f"  {'...':>5}")
        lr = layers[idx]
        lines.append(
            f"  {lr['layer_id']:>5} {lr['attention_type']:>5} {lr['ffn_type']:>5} "
            f"{str(lr['use_indexer']):>5} {format_number(lr['params']):>16} "
            f"{format_bytes(lr['bytes']):>14} {format_flops(lr['flops']):>16}"
        )
        prev = idx

    lines.append("")
    lines.append("=" * 78)
    return "\n".join(lines)