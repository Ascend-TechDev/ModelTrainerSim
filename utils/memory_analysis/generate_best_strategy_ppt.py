#!/usr/bin/env python3
"""Select scenario-specific EP/FSDP strategies and create an editable PPTX."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.memory_analysis.generate_ep_fsdp_reports import (
    BANDWIDTHS_GBS,
    DDR_CAPACITIES_GB,
    MODELS,
    analyze_strategy,
    extract_model_data,
    load_report,
    strategies_for,
)

HARDWARES = [("970DT", 1769.0, 0), ("980DT", 3538.0, 1)]
SCENARIOS = [
    {"name": "HBM=288GB / DDR=768GB", "hbm": 288, "ddr": 768},
    {"name": "HBM=192GB / DDR=512GB", "hbm": 192, "ddr": 512},
]
HBM_DDR512_SCENARIOS = [
    {"name": f"HBM={hbm}GB / DDR=512GB", "hbm": hbm, "ddr": 512}
    for hbm in (96, 120, 144, 168, 192, 240, 288)
]
BASE_OUTPUT_DIR = PROJECT_ROOT / "reports_analyse" / "ep_fsdp_mem_hbm_d2h"

FONT = "Microsoft YaHei"
NAVY = RGBColor(31, 55, 85)
BLUE = RGBColor(56, 117, 173)
LIGHT_BLUE = RGBColor(225, 237, 247)
PALE = RGBColor(245, 248, 251)
WHITE = RGBColor(255, 255, 255)
DARK = RGBColor(43, 52, 61)


def collect_scenario_results(weight_dtype: str = "bf16",
                             activation_dtype: str = "bf16") -> dict:
    all_scenarios = SCENARIOS + HBM_DDR512_SCENARIOS
    results = {}
    for model_name, params_t, card_counts, config_name in MODELS:
        if config_name is None:
            for hardware, _, card_index in HARDWARES:
                for scenario in all_scenarios:
                    results[(hardware, scenario["name"], model_name)] = {
                        "model": model_name,
                        "params_t": params_t,
                        "k": card_counts[card_index],
                        "available": False,
                    }
            continue

        model_data = extract_model_data(
            load_report(config_name, weight_dtype, activation_dtype))
        for hardware, bf16_tflops, card_index in HARDWARES:
            compute_tflops = bf16_tflops * (2 if weight_dtype == "fp8" else 1)
            k = card_counts[card_index]
            candidates = []
            for strategy in strategies_for(k):
                if not strategy["valid"]:
                    continue
                for scenario in all_scenarios:
                    calc = analyze_strategy(
                        model_data,
                        k,
                        strategy,
                        scenario["hbm"],
                        bf16_tflops=compute_tflops,
                    )
                    benefit, offload = calc["ddr_benefits"][scenario["ddr"]]
                    candidates.append({
                        "model": model_name,
                        "params_t": params_t,
                        "k": k,
                        "available": True,
                        "strategy": strategy,
                        "calc": calc,
                        "ddr_benefit": benefit,
                        "ddr_offload": offload,
                        "scenario": scenario,
                        "hardware": hardware,
                        "model_data": model_data,
                    })

            for scenario in all_scenarios:
                scenario_candidates = [
                    candidate for candidate in candidates
                    if candidate["scenario"]["name"] == scenario["name"]
                ]
                best = min(
                    scenario_candidates,
                    key=lambda candidate: (
                        -candidate["ddr_benefit"],
                        candidate["calc"]["ddr_requirement_full_gb"],
                        max(candidate["calc"]["d2h_moe_gbps"], candidate["calc"]["d2h_full_gbps"]),
                        candidate["calc"]["ssd_gb"],
                        candidate["strategy"]["ep"],
                        candidate["strategy"]["efsdp"],
                    ),
                )
                results[(hardware, scenario["name"], model_name)] = best
    return results


def set_cell(cell, text: str, *, header: bool = False, shade: bool = False, font_size: float = 8.0) -> None:
    cell.text = str(text)
    cell.vertical_anchor = MSO_ANCHOR.MIDDLE
    cell.margin_left = Inches(0.04)
    cell.margin_right = Inches(0.04)
    cell.margin_top = Inches(0.015)
    cell.margin_bottom = Inches(0.015)
    cell.fill.solid()
    cell.fill.fore_color.rgb = NAVY if header else (PALE if shade else WHITE)
    for paragraph in cell.text_frame.paragraphs:
        paragraph.alignment = PP_ALIGN.CENTER
        paragraph.space_after = Pt(0)
        for run in paragraph.runs:
            run.font.name = FONT
            run.font.size = Pt(font_size)
            run.font.bold = header
            run.font.color.rgb = WHITE if header else DARK


def add_table(slide, headers: list[str], rows: list[list[str]], x, y, w, h, font_size=8.0, col_widths=None):
    table_shape = slide.shapes.add_table(
        len(rows) + 1,
        len(headers),
        x,
        y,
        w,
        h,
    )
    table = table_shape.table
    if col_widths:
        for column, width in zip(table.columns, col_widths):
            column.width = Inches(width)
    for col, header in enumerate(headers):
        set_cell(table.cell(0, col), header, header=True, font_size=font_size)
    for row_index, values in enumerate(rows, start=1):
        for col, value in enumerate(values):
            set_cell(
                table.cell(row_index, col),
                value,
                shade=row_index % 2 == 0,
                font_size=font_size,
            )
    return table_shape


def new_slide(prs: Presentation, title: str, subtitle: str = ""):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    title_box = slide.shapes.add_textbox(Inches(0.4), Inches(0.2), Inches(12.5), Inches(0.45))
    paragraph = title_box.text_frame.paragraphs[0]
    paragraph.text = title
    paragraph.font.name = FONT
    paragraph.font.size = Pt(22)
    paragraph.font.bold = True
    paragraph.font.color.rgb = NAVY
    if subtitle:
        subtitle_box = slide.shapes.add_textbox(Inches(0.42), Inches(0.72), Inches(12.4), Inches(0.35))
        paragraph = subtitle_box.text_frame.paragraphs[0]
        paragraph.text = subtitle
        paragraph.font.name = FONT
        paragraph.font.size = Pt(10)
        paragraph.font.color.rgb = DARK
    return slide


def add_line_chart(slide, title: str, labels: list[str], series: list[tuple[str, list[float]]], x, y, w, h, value_axis_title: str):
    chart_data = CategoryChartData()
    chart_data.categories = labels
    for name, values in series:
        chart_data.add_series(name, values)
    chart = slide.shapes.add_chart(
        XL_CHART_TYPE.LINE_MARKERS,
        x,
        y,
        w,
        h,
        chart_data,
    ).chart
    chart.has_title = True
    chart.chart_title.text_frame.text = title
    chart.chart_title.text_frame.paragraphs[0].font.name = FONT
    chart.chart_title.text_frame.paragraphs[0].font.size = Pt(12)
    chart.has_legend = len(series) > 1
    if chart.has_legend:
        chart.legend.position = XL_LEGEND_POSITION.BOTTOM
        chart.legend.include_in_layout = False
        chart.legend.font.name = FONT
        chart.legend.font.size = Pt(9)
    chart.value_axis.has_title = True
    chart.value_axis.axis_title.text_frame.text = value_axis_title
    chart.value_axis.tick_labels.font.name = FONT
    chart.value_axis.tick_labels.font.size = Pt(8)
    chart.category_axis.tick_labels.font.name = FONT
    chart.category_axis.tick_labels.font.size = Pt(7)
    chart.category_axis.tick_labels.rotation = -45
    chart.has_title = True
    plot = chart.plots[0]
    plot.has_data_labels = True
    plot.data_labels.show_value = True
    plot.data_labels.number_format = "0"
    plot.data_labels.font.name = FONT
    plot.data_labels.font.size = Pt(6)
    for plot_series in chart.series:
        plot_series.format.line.width = Pt(2)
    return chart


def result_for(results: dict, hardware: str, scenario: dict, model_name: str) -> dict:
    return results[(hardware, scenario["name"], model_name)]


def add_best_strategy_slide(prs: Presentation, results: dict, hardware_spec: tuple) -> None:
    hardware, _, _ = hardware_spec
    for scenario in SCENARIOS:
        slide = new_slide(
            prs,
            f"{hardware} 最佳并行策略 — {scenario['name']}",
            "选择规则：指定DDR容量下端到端收益最大；平局依次选择DDR诉求、D2H诉求、SSD诉求更低。Qwen3.5缺少匹配配置，数值未计算。",
        )
        headers = [
            "模型", "K", "EP", "EFSDP", "EDP", "HBM余量GB",
            "DDR全卸载GB", "DDR仅MoE卸载GB",
            "指定DDR收益", "单卡SSD分片GB", "D2H-MoE GB/s", "D2H-Full GB/s", "D2H=200收益",
        ]
        rows = []
        for model_name, _, _, _ in MODELS:
            item = result_for(results, hardware, scenario, model_name)
            if not item["available"]:
                rows.append([model_name, f"{item['k']:,}", "—", "—", "—", "—", "—", "—", "—", "—", "—", "—", "—"])
                continue
            strategy = item["strategy"]
            calc = item["calc"]
            rows.append([
                model_name,
                f"{item['k']:,}",
                str(strategy["ep"]),
                str(strategy["efsdp"]),
                f"{strategy['edp']:,}",
                f"{calc['hbm_remaining_gb']:.1f}",
                f"{calc['ddr_requirement_full_gb']:.1f}",
                f"{calc['ddr_requirement_moe_gb']:.1f}",
                f"{item['ddr_benefit']:.2f}%",
                f"{calc['ssd_gb']:.1f}",
                f"{calc['d2h_moe_gbps']:.1f}",
                f"{calc['d2h_full_gbps']:.1f}",
                f"{calc['bandwidth_benefits'][200]:.2f}%",
            ])
        add_table(
            slide,
            headers,
            rows,
            Inches(0.35),
            Inches(1.25),
            Inches(12.65),
            Inches(5.75),
            font_size=7.1,
            col_widths=[1.15, 0.55, 0.45, 0.5, 0.55, 0.75, 0.82, 0.9, 0.82, 0.72, 0.8, 0.88, 0.9],
        )
        slide.notes_slide.notes_text_frame.text = (
            f"Scenario: {scenario['name']}. DDR收益使用报告中的 DDR 扫描公式，收益=offload_ratio×25%。"
            "D2H=200GB/s使用80%带宽利用率（有效160GB/s）。"
        )


def add_capacity_charts(prs: Presentation, results: dict, hardware: str, scenario: dict) -> None:
    slide = new_slide(
        prs,
        f"{hardware} — DDR / SSD 容量诉求 — {scenario['name']}",
        "每个模型使用该场景选定的 EP/EFSDP 策略；缺少模型结构配置的条目不参与折线和统计。",
    )
    model_names = [model[0] for model in MODELS if model[3] is not None]
    labels = [name.replace("Kimi-", "K") for name in model_names]
    items = [result_for(results, hardware, scenario, name) for name in model_names]
    ddr_full_values = [item["calc"]["ddr_requirement_full_gb"] for item in items]
    ddr_moe_values = [item["calc"]["ddr_requirement_moe_gb"] for item in items]
    ssd_values = [item["calc"]["ssd_gb"] for item in items]
    add_line_chart(
        slide,
        "DDR容量诉求（单卡GB）",
        labels,
        [("全卸载DDR诉求", ddr_full_values), ("仅MoE卸载DDR诉求", ddr_moe_values)],
        Inches(0.2),
        Inches(1.25),
        Inches(6.45),
        Inches(5.6),
        "GB",
    )
    add_line_chart(
        slide,
        "单卡Checkpoint分片SSD（GB）",
        labels,
        [("SSD诉求", ssd_values)],
        Inches(6.75),
        Inches(1.25),
        Inches(6.4),
        Inches(5.6),
        "GB",
    )


def add_d2h_slide(prs: Presentation, results: dict, hardware: str, scenario: dict) -> None:
    slide = new_slide(
        prs,
        f"{hardware} — D2H带宽诉求与200GB/s收益 — {scenario['name']}",
        "带宽诉求分 MoE 激活和全激活；带宽遍历按有效带宽=标称带宽×80%计算，端到端收益上限为25%。",
    )
    model_names = [model[0] for model in MODELS if model[3] is not None]
    labels = [name.replace("Kimi-", "K") for name in model_names]
    items = [result_for(results, hardware, scenario, name) for name in model_names]
    add_line_chart(
        slide,
        "各模型D2H带宽诉求",
        labels,
        [
            ("MoE激活", [item["calc"]["d2h_moe_gbps"] for item in items]),
            ("全激活", [item["calc"]["d2h_full_gbps"] for item in items]),
        ],
        Inches(0.35),
        Inches(1.2),
        Inches(12.6),
        Inches(4.85),
        "GB/s",
    )
    note = slide.shapes.add_textbox(Inches(0.5), Inches(6.15), Inches(12.2), Inches(0.55))
    paragraph = note.text_frame.paragraphs[0]
    paragraph.text = "D2H=200GB/s时的端到端收益（逐模型详见上一页最佳策略表最后一列）："
    paragraph.font.name = FONT
    paragraph.font.size = Pt(9)
    paragraph.font.color.rgb = DARK
    benefit_row = "   |   ".join(
        f"{name}: {result_for(results, hardware, scenario, name)['calc']['bandwidth_benefits'][200]:.2f}%"
        for name in model_names
    )
    paragraph = note.text_frame.add_paragraph()
    paragraph.text = benefit_row
    paragraph.font.name = FONT
    paragraph.font.size = Pt(7.5)
    paragraph.font.color.rgb = DARK


def add_sweep_statistics_slide(prs: Presentation, results: dict, hardware: str, scenario: dict) -> None:
    slide = new_slide(
        prs,
        f"{hardware} — D2H / DDR扩容收益统计 — {scenario['name']}",
        "“有收益模型数”指端到端收益>0；“满额模型数”指收益达到25%；平均收益按14个有配置的模型计算（Qwen3.5不计入）。",
    )
    numeric_models = [model[0] for model in MODELS if model[3] is not None]
    items = [result_for(results, hardware, scenario, name) for name in numeric_models]
    rows = []
    for bandwidth in BANDWIDTHS_GBS:
        benefits = [
            item["calc"]["bandwidth_benefits"][bandwidth]
            for item in items
        ]
        rows.append([
            f"D2H {bandwidth}G→{bandwidth * 0.8:.0f}G",
            f"{sum(value > 0 for value in benefits)}/14",
            f"{sum(value >= 25 - 1e-8 for value in benefits)}/14",
            f"{sum(benefits) / len(benefits):.2f}%",
        ])
    for capacity in DDR_CAPACITIES_GB:
        benefits = [
            item["calc"]["ddr_benefits"][capacity][0]
            for item in items
        ]
        rows.append([
            f"DDR {capacity}GB",
            f"{sum(value > 0 for value in benefits)}/14",
            f"{sum(value >= 25 - 1e-8 for value in benefits)}/14",
            f"{sum(benefits) / len(benefits):.2f}%",
        ])
    add_table(
        slide,
        ["遍历条件", "有收益模型数", "满额模型数", "平均端到端收益"],
        rows,
        Inches(1.3),
        Inches(1.35),
        Inches(10.7),
        Inches(5.5),
        font_size=11,
        col_widths=[3.2, 2.2, 2.2, 2.9],
    )


def add_hbm_ddr512_screening_slide(prs: Presentation, results: dict, hardware: str) -> None:
    slide = new_slide(
        prs,
        f"{hardware} — DDR=512GB 对七档 HBM 的容量满足性",
        "每格显示该 HBM 下按DDR收益优先选出的策略及全卸载容量结论；“满足”表示该策略全激活DDR诉求≤512GB。",
    )
    headers = ["模型"] + [f"HBM={scenario['hbm']}GB" for scenario in HBM_DDR512_SCENARIOS]
    rows = []
    for model_name, _, _, _ in MODELS:
        cells = [model_name]
        for scenario in HBM_DDR512_SCENARIOS:
            item = result_for(results, hardware, scenario, model_name)
            if not item["available"]:
                cells.append("无匹配配置\n未计算")
                continue
            strategy = item["strategy"]
            calc = item["calc"]
            status = "满足" if item["ddr_benefit"] >= 25 - 1e-8 else "不足"
            cells.append(
                f"EP{strategy['ep']}/EF{strategy['efsdp']}\n"
                f"EDP={strategy['edp']:,}\n"
                f"DDR诉求 {calc['ddr_requirement_full_gb']:.1f}GB\n"
                f"{item['ddr_benefit']:.1f}% · {status}"
            )
        rows.append(cells)
    add_table(
        slide,
        headers,
        rows,
        Inches(0.25),
        Inches(1.2),
        Inches(12.83),
        Inches(5.95),
        font_size=6.2,
        col_widths=[1.05] + [1.68] * len(HBM_DDR512_SCENARIOS),
    )
    slide.notes_slide.notes_text_frame.text = (
        "容量满足性按DDR诉求（全卸载）判断：优化器与主权重 + 全激活 - HBM剩余 ≤ 512GB。"
        "该筛选在每个HBM容量下遍历全部有效EP/EFSDP策略，并按既定规则选择端到端DDR收益最高策略。"
        "“不足”表示该策略在DDR=512GB下只能部分卸载，表中收益为可实现的端到端收益。"
    )


def add_hbm_d2h_chart_slide(prs: Presentation, results: dict, hardware: str) -> None:
    slide = new_slide(
        prs,
        f"{hardware} — 七档 HBM 下的 D2H 带宽诉求",
        "每个 HBM 档位分别按 DDR=512GB 的策略选择规则确定 EP/EFSDP；曲线展示各模型对应策略的 D2H 需求。",
    )
    model_names = [model[0] for model in MODELS if model[3] is not None]
    labels = [name.replace("Kimi-", "K") for name in model_names]
    hbm_series = []
    for scenario in HBM_DDR512_SCENARIOS:
        items = [
            result_for(results, hardware, scenario, model_name)
            for model_name in model_names
        ]
        hbm_series.append((
            f"HBM={scenario['hbm']}GB",
            [item["calc"]["d2h_moe_gbps"] for item in items],
        ))
    add_line_chart(
        slide,
        "MoE 激活 D2H 带宽诉求",
        labels,
        hbm_series,
        Inches(0.2),
        Inches(1.25),
        Inches(6.45),
        Inches(5.65),
        "GB/s",
    )
    hbm_series = []
    for scenario in HBM_DDR512_SCENARIOS:
        items = [
            result_for(results, hardware, scenario, model_name)
            for model_name in model_names
        ]
        hbm_series.append((
            f"HBM={scenario['hbm']}GB",
            [item["calc"]["d2h_full_gbps"] for item in items],
        ))
    add_line_chart(
        slide,
        "全激活 D2H 带宽诉求",
        labels,
        hbm_series,
        Inches(6.75),
        Inches(1.25),
        Inches(6.4),
        Inches(5.65),
        "GB/s",
    )
    slide.notes_slide.notes_text_frame.text = (
        "每个HBM档位使用HBM_DDR512_SCENARIOS对应的最优策略（以DDR=512GB端到端收益优先选择），"
        "分别绘制MoE激活及全激活的D2H带宽诉求。折线图数据标签四舍五入为整数GB/s。"
    )


def add_hbm_strategy_details_slide(
    prs: Presentation,
    results: dict,
    hardware: str,
    scenario: dict,
) -> None:
    slide = new_slide(
        prs,
        f"{hardware} — HBM={scenario['hbm']}GB 最优并行策略与资源诉求",
        "策略按 DDR=512GB 场景的端到端收益优先选择；D2H=200GB/s按80%利用率，收益均为消除重计算的端到端比例。",
    )
    headers = [
        "模型",
        "K",
        "EP",
        "EFSDP",
        "EDP",
        "HBM余量GB",
        "DDR全卸载GB",
        "DDR仅MoE卸载GB",
        "D2H-MoE GB/s",
        "D2H-Full GB/s",
        "D2H=200收益",
        "DDR=512收益",
    ]
    rows = []
    for model_name, _, _, _ in MODELS:
        item = result_for(results, hardware, scenario, model_name)
        if not item["available"]:
            rows.append([
                model_name,
                f"{item['k']:,}",
                "—",
                "—",
                "—",
                "—",
                "—",
                "—",
                "—",
                "—",
                "—",
                "—",
            ])
            continue
        strategy = item["strategy"]
        calc = item["calc"]
        rows.append([
            model_name,
            f"{item['k']:,}",
            str(strategy["ep"]),
            str(strategy["efsdp"]),
            f"{strategy['edp']:,}",
            f"{calc['hbm_remaining_gb']:+.1f}",
            f"{calc['ddr_requirement_full_gb']:.1f}",
            f"{calc['ddr_requirement_moe_gb']:.1f}",
            f"{calc['d2h_moe_gbps']:.1f}",
            f"{calc['d2h_full_gbps']:.1f}",
            f"{calc['bandwidth_benefits'][200]:.2f}%",
            f"{item['ddr_benefit']:.2f}%",
        ])
    add_table(
        slide,
        headers,
        rows,
        Inches(0.35),
        Inches(1.25),
        Inches(12.65),
        Inches(5.75),
        font_size=6.8,
        col_widths=[1.35, 0.8, 0.55, 0.65, 0.7, 1.05, 1.1, 1.35, 1.25, 1.25, 1.3, 1.3],
    )
    slide.notes_slide.notes_text_frame.text = (
        f"HBM={scenario['hbm']}GB，策略按DDR=512GB场景的端到端收益优先选择；"
        "HBM余量=0.95×HBM容量−静态HBM占用；DDR全卸载诉求和仅MoE卸载诉求分别按报告定义计算；"
        "D2H-MoE及D2H-Full为对应卸载范围的带宽诉求；D2H=200GB/s收益使用有效带宽160GB/s，DDR收益基于512GB容量。"
    )


def add_hbm_resource_charts_slide(
    prs: Presentation,
    results: dict,
    hardware: str,
) -> None:
    slide = new_slide(
        prs,
        f"{hardware} — HBM 档位与 HBM余量 / DDR诉求",
        "每条曲线对应一种HBM档位；每个模型使用该档位下按DDR=512GB收益优先选出的策略。",
    )
    model_names = [model[0] for model in MODELS if model[3] is not None]
    labels = [name.replace("Kimi-", "K") for name in model_names]
    series = []
    for scenario in HBM_DDR512_SCENARIOS:
        items = [
            result_for(results, hardware, scenario, model_name)
            for model_name in model_names
        ]
        series.append((
            f"HBM={scenario['hbm']}GB",
            [item["calc"]["hbm_remaining_gb"] for item in items],
        ))
    add_line_chart(
        slide,
        "HBM余量（GB）",
        labels,
        series,
        Inches(0.2),
        Inches(1.25),
        Inches(6.45),
        Inches(5.65),
        "GB",
    )
    series = []
    for scenario in HBM_DDR512_SCENARIOS:
        items = [
            result_for(results, hardware, scenario, model_name)
            for model_name in model_names
        ]
        series.append((
            f"HBM={scenario['hbm']}GB",
            [item["calc"]["ddr_requirement_full_gb"] for item in items],
        ))
    add_line_chart(
        slide,
        "DDR全卸载容量诉求（GB）",
        labels,
        series,
        Inches(6.75),
        Inches(1.25),
        Inches(6.4),
        Inches(5.65),
        "GB",
    )


def add_hbm_benefit_charts_slide(
    prs: Presentation,
    results: dict,
    hardware: str,
) -> None:
    slide = new_slide(
        prs,
        f"{hardware} — HBM 档位与端到端收益",
        "收益表示可消除重计算比例；D2H=200GB/s按80%利用率（有效160GB/s），DDR收益按512GB容量计算。",
    )
    model_names = [model[0] for model in MODELS if model[3] is not None]
    labels = [name.replace("Kimi-", "K") for name in model_names]
    series = []
    for scenario in HBM_DDR512_SCENARIOS:
        items = [
            result_for(results, hardware, scenario, model_name)
            for model_name in model_names
        ]
        series.append((
            f"HBM={scenario['hbm']}GB",
            [item["calc"]["bandwidth_benefits"][200] for item in items],
        ))
    add_line_chart(
        slide,
        "D2H=200GB/s 时的端到端收益（%）",
        labels,
        series,
        Inches(0.2),
        Inches(1.25),
        Inches(6.45),
        Inches(5.65),
        "%",
    )
    series = []
    for scenario in HBM_DDR512_SCENARIOS:
        items = [
            result_for(results, hardware, scenario, model_name)
            for model_name in model_names
        ]
        series.append((
            f"HBM={scenario['hbm']}GB",
            [item["ddr_benefit"] for item in items],
        ))
    add_line_chart(
        slide,
        "DDR=512GB 时的端到端收益（%）",
        labels,
        series,
        Inches(6.75),
        Inches(1.25),
        Inches(6.4),
        Inches(5.65),
        "%",
    )


def make_presentation(results: dict, weight_dtype: str = "bf16",
                      activation_dtype: str = "bf16") -> Presentation:
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    slide = prs.slides.add_slide(prs.slide_layouts[6])
    title = slide.shapes.add_textbox(Inches(0.8), Inches(1.2), Inches(11.8), Inches(1.0))
    paragraph = title.text_frame.paragraphs[0]
    paragraph.text = "EP / FSDP 内存与卸载策略分析"
    paragraph.font.name = FONT
    paragraph.font.size = Pt(30)
    paragraph.font.bold = True
    paragraph.font.color.rgb = NAVY
    subtitle = slide.shapes.add_textbox(Inches(0.85), Inches(2.35), Inches(11.4), Inches(1.25))
    paragraph = subtitle.text_frame.paragraphs[0]
    paragraph.text = (
        f"weight_dtype={weight_dtype} | activation_dtype={activation_dtype}\n"
        "970DT 与 980DT | HBM=288GB / DDR=768GB、HBM=192GB / DDR=512GB\n"
        "按指定DDR下端到端收益优先选择策略；收益相同时依次比较DDR、D2H、SSD诉求。"
    )
    paragraph.font.name = FONT
    paragraph.font.size = Pt(18)
    paragraph.font.color.rgb = DARK
    note = slide.shapes.add_textbox(Inches(0.85), Inches(4.0), Inches(11.4), Inches(1.25))
    paragraph = note.text_frame.paragraphs[0]
    paragraph.text = (
        "范围：EP=128/256/512/1024/2048，EFSDP=1/2/4/8/16/32；seq_len=4096，batch_size=1，MFU=0.40。\n"
        "逐模型报告额外覆盖HBM=96/120/144/168/192/240/288GB；最佳策略表仍按任务指定的192GB与288GB场景。\n"
        "FP8权重场景使用FP8峰值算力；激活精度影响激活内存。\n"
        "参数结构按仓库配置计算；Qwen3.5-397B缺少匹配配置，GLM-10T配置参数量与卡数表不一致，已在报告中披露。"
    )
    paragraph.font.name = FONT
    paragraph.font.size = Pt(12)
    paragraph.font.color.rgb = DARK

    for hardware_spec in HARDWARES:
        add_best_strategy_slide(prs, results, hardware_spec)
    for hardware, _, _ in HARDWARES:
        add_hbm_ddr512_screening_slide(prs, results, hardware)
        add_hbm_d2h_chart_slide(prs, results, hardware)
        add_hbm_resource_charts_slide(prs, results, hardware)
        add_hbm_benefit_charts_slide(prs, results, hardware)
        for scenario in HBM_DDR512_SCENARIOS:
            add_hbm_strategy_details_slide(prs, results, hardware, scenario)
    for hardware, _, _ in HARDWARES:
        for scenario in SCENARIOS:
            add_capacity_charts(prs, results, hardware, scenario)
            add_d2h_slide(prs, results, hardware, scenario)
            add_sweep_statistics_slide(prs, results, hardware, scenario)
    return prs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weight-dtype", choices=["bf16", "fp8"], default="bf16")
    parser.add_argument("--activation-dtype", choices=["bf16", "fp8"], default="bf16")
    args = parser.parse_args()
    output_dir = (
        BASE_OUTPUT_DIR
        / f"weight_{args.weight_dtype}_activation_{args.activation_dtype}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    results = collect_scenario_results(args.weight_dtype, args.activation_dtype)
    prs = make_presentation(results, args.weight_dtype, args.activation_dtype)
    output_pptx = output_dir / "EP_FSDP_best_strategy_970_980DT.pptx"
    prs.save(output_pptx)
    print(f"Wrote {output_pptx.relative_to(PROJECT_ROOT)} ({len(prs.slides)} slides)")


if __name__ == "__main__":
    main()
