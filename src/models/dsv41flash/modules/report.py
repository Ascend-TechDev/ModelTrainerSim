"""Flash display adapter for the shared Pro-style six-section report.

This adapter changes presentation only. Source-shaped modules and full notes
remain intact in JSON. MoE common input/output tensors are assigned to the gate
row once; routed/shared expert intermediates are split by active expert count.
"""
from copy import deepcopy

from ....core.formatter import format_flops, format_number


FLASH_MODULE_ORDER = ["wordembedding", "gate", "routedexpert", "sharedexpert",
                      "swa", "csa_full", "csa_reindex", "csa_reuse", "mhc",
                      "engram", "outputlayer", "dspark"]
TYPE_LABELS = {"WordEmbedding": "wordembedding", "SWA": "swa", "Full": "csa_full",
               "Reindex": "csa_reindex", "Reuse": "csa_reuse", "mHC_RMSNorm": "mhc",
               "Engram": "engram", "OutputLayer": "outputlayer", "DSpark": "dspark"}


def format_flash_bytes(value: float) -> str:
    return f"{value / 1e9:,.6f} GB"


def prepare_flash_report(report: dict) -> dict:
    """Return a display-only copy, keeping every MoE subtotal conserved."""
    view = deepcopy(report)
    summary = {TYPE_LABELS.get(k, k.lower()): deepcopy(v)
               for k, v in report["per_module_summary"].items() if k != "MoE"}
    for key in ("gate", "routedexpert", "sharedexpert"):
        summary[key] = {"count": 0, "params": 0, "bytes": 0.0,
                        "flops": 0.0, "act_bytes": 0.0, "cache_bytes": 0.0}
    B, S = report["compute"]["batch_size"], report["compute"]["seq_len"]
    H = report["config"]["hidden_size"]
    for module in report["modules"]:
        if module["type"] != "MoE":
            continue
        details = module["details"]
        K, shared = details["active_experts"], details["shared_experts"]
        I, E = details["expert_intermediate_size"], details["routed_experts"]
        for parameter in module["parameter_breakdown"]:
            name = parameter["name"]
            if ".shared_experts." in name:
                key = "sharedexpert"
            elif ".experts.*." in name:
                key = "routedexpert"
            elif ".gate." in name:
                key = "gate"
            else:
                raise ValueError(f"Unclassified MoE parameter in report: {name}")
            summary[key]["params"] += parameter["params"]
            summary[key]["bytes"] += parameter["bytes"]
        summary["gate"]["flops"] += 2 * B * S * H * E
        summary["routedexpert"]["flops"] += 6 * B * S * H * I * K
        summary["sharedexpert"]["flops"] += 6 * B * S * H * I * shared
        for term in details["activation_terms"]:
            name, size = term["name"], term["bytes"]
            if name in ("gate_linear_fp32", "up_linear_fp32", "down_linear_inputs"):
                summary["routedexpert"]["act_bytes"] += size * K / (K + shared)
                summary["sharedexpert"]["act_bytes"] += size * shared / (K + shared)
            elif name == "routed_expert_inputs":
                summary["routedexpert"]["act_bytes"] += size
            else:
                # Router and the common MoE input/output are assigned once.
                summary["gate"]["act_bytes"] += size
        for key in ("gate", "routedexpert", "sharedexpert"):
            summary[key]["count"] += 1
    view["per_module_summary"] = summary
    view.pop("per_module_summary_excel", None)
    view["memory"]["training_breakdown"]["optimizer"] = "mixed"
    return view


def render_flash_report(report: dict) -> str:
    from ....core.reporter import render_standard_report, section

    view = prepare_flash_report(report)
    cfg, tb = report["config"], report["memory"]["training_breakdown"]
    fmt = format_flash_bytes
    table_device = cfg["training"]["groups"]["engram"]["device"].upper()
    extra = [("engram", f"enabled (table: {table_device})" if cfg["include_engram"] else "disabled"),
             ("dspark", "enabled" if cfg["include_dspark"] else "disabled"),
             ("num_devices", str(cfg["training"]["num_devices"]))]
    appendix = []
    if (cfg["training"]["num_devices"] > 1 or tb["per_device"]["cpu"]["total"]
            or tb["per_device_gpu_buffers_bytes"]):
        appendix.extend(["", section("Per-Device Memory (static)")])
        for device in ("gpu", "cpu"):
            appendix.append(f"  {device.upper() + ' static':<24}: {fmt(tb['per_device'][device]['total'])}")
        for name, value in tb["per_device_gpu_buffers_bytes"].items():
            appendix.append(f"  {name:<24}: {fmt(value)}")
        appendix.append(f"  {'GPU static + buffers':<24}: {fmt(tb['per_device_gpu_static_with_buffers_bytes'])}")
    if report["auxiliary"]["enabled"]:
        aux = report["auxiliary"]
        appendix.extend(["", section("DSpark (separate draft pass)"),
                         f"  {'Resident params':<24}: {format_number(aux['params'])}",
                         f"  {'Prefix / draft tokens':<24}: {aux['prefix_tokens']} / {aux['draft_tokens']}",
                         f"  {'Draft FLOPs':<24}: {format_flops(aux['draft_forward_flops'])}",
                         f"  {'Draft tensor volume':<24}: {fmt(aux['draft_activation_bytes'])}"])
    return render_standard_report(view, byte_formatter=fmt, module_order=FLASH_MODULE_ORDER,
                                  extra_configuration=extra, appendix=appendix)
