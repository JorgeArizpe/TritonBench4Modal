#!/usr/bin/env python
"""Create plots from nvidia iterative Modal JSON summaries.

Outputs:
  - speedup_boxplot_all.svg
  - speedup_boxplot_accepted.svg
  - accuracy_by_iteration.svg
  - iteration_metrics.csv

The script intentionally uses only the Python standard library so it can run in
the lightweight local environment for this repository.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Iterable


@dataclass
class IterationRecord:
    iteration: int
    path: Path
    call_passed: int
    call_rate: float
    exec_passed: int
    exec_rate: float
    attempted_results: int
    computed_results: int
    accepted_results: int
    json_mean_speedup: float | None
    speedups: list[float]
    accepted_speedups: list[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create SVG plots for the NVIDIA iterative Modal run JSONs."
    )
    parser.add_argument(
        "--json-dir",
        type=Path,
        default=Path("jsons"),
        help="Directory containing iter1.json ... Iter5.json and optionally best.json.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("plots"),
        help="Directory where SVG plots and CSV summary will be written.",
    )
    return parser.parse_args()


def iteration_sort_key(path: Path) -> tuple[int, str]:
    digits = "".join(ch for ch in path.stem if ch.isdigit())
    return (int(digits) if digits else 10_000, path.name.lower())


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def as_int(value: object, default: int = 0) -> int:
    return int(value) if isinstance(value, (int, float)) else default


def as_float(value: object) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def load_iterations(json_dir: Path) -> list[IterationRecord]:
    paths = sorted(
        [path for path in json_dir.glob("*.json") if path.name.lower().startswith("iter")],
        key=iteration_sort_key,
    )
    if not paths:
        raise FileNotFoundError(f"No iteration JSON files found under {json_dir}")

    records: list[IterationRecord] = []
    for path in paths:
        payload = load_json(path)
        phase1 = payload.get("phase1_call_acc", {})
        phase2 = payload.get("phase2_exec_acc", {})
        phase3 = payload.get("phase3_efficiency", {})
        per_file = phase3.get("per_file", {})

        speedups: list[float] = []
        accepted_speedups: list[float] = []
        for result in per_file.values():
            value = as_float(result.get("speedup_vs_pytorch"))
            if value is None or value <= 0:
                continue
            speedups.append(value)
            status = str(result.get("status", "")).lower()
            if "accepted" in status and "out_of_range" not in status:
                accepted_speedups.append(value)

        records.append(
            IterationRecord(
                iteration=as_int(payload.get("iteration"), iteration_sort_key(path)[0]),
                path=path,
                call_passed=as_int(phase1.get("passed")),
                call_rate=float(phase1.get("rate", 0.0)),
                exec_passed=as_int(phase2.get("passed")),
                exec_rate=float(phase2.get("rate", 0.0)),
                attempted_results=as_int(phase3.get("attempted_results")),
                computed_results=as_int(phase3.get("computed_results")),
                accepted_results=as_int(phase3.get("accepted_results")),
                json_mean_speedup=as_float(phase3.get("computed_mean_speedup_vs_pytorch")),
                speedups=speedups,
                accepted_speedups=accepted_speedups,
            )
        )
    return records


def median(sorted_values: list[float]) -> float:
    count = len(sorted_values)
    midpoint = count // 2
    if count % 2:
        return sorted_values[midpoint]
    return (sorted_values[midpoint - 1] + sorted_values[midpoint]) / 2.0


def five_number(values: Iterable[float]) -> dict[str, object]:
    sorted_values = sorted(values)
    if not sorted_values:
        raise ValueError("Cannot calculate boxplot statistics for an empty list")

    count = len(sorted_values)
    q2 = median(sorted_values)
    if count == 1:
        q1 = q3 = q2
    elif count % 2:
        q1 = median(sorted_values[: count // 2])
        q3 = median(sorted_values[count // 2 + 1 :])
    else:
        q1 = median(sorted_values[: count // 2])
        q3 = median(sorted_values[count // 2 :])

    iqr = q3 - q1
    if iqr == 0:
        lower_whisker = sorted_values[0]
        upper_whisker = sorted_values[-1]
    else:
        lower_fence = q1 - 1.5 * iqr
        upper_fence = q3 + 1.5 * iqr
        lower_whisker = min(value for value in sorted_values if value >= lower_fence)
        upper_whisker = max(value for value in sorted_values if value <= upper_fence)

    outliers = [
        value
        for value in sorted_values
        if value < lower_whisker or value > upper_whisker
    ]
    return {
        "min": sorted_values[0],
        "q1": q1,
        "median": q2,
        "q3": q3,
        "max": sorted_values[-1],
        "lower_whisker": lower_whisker,
        "upper_whisker": upper_whisker,
        "outliers": outliers,
    }


def fmt_speedup(value: float) -> str:
    if value >= 100:
        return f"{value:,.0f}x"
    if value >= 10:
        return f"{value:,.1f}x"
    if value >= 1:
        return f"{value:.2f}x"
    return f"{value:.3f}x"


def svg_text(
    x: float,
    y: float,
    text: str,
    *,
    size: int = 14,
    anchor: str = "middle",
    weight: str = "400",
    fill: str = "#1f2933",
    rotate: float | None = None,
) -> str:
    attrs = [
        f'x="{x:.2f}"',
        f'y="{y:.2f}"',
        f'font-size="{size}"',
        f'text-anchor="{anchor}"',
        f'font-weight="{weight}"',
        f'fill="{fill}"',
        'font-family="Arial, Helvetica, sans-serif"',
    ]
    if rotate is not None:
        attrs.append(f'transform="rotate({rotate:.2f} {x:.2f} {y:.2f})"')
    return f"<text {' '.join(attrs)}>{escape(text)}</text>"


def svg_line(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    stroke: str = "#5d6978",
    width: float = 1.0,
    dash: str | None = None,
) -> str:
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
        f'stroke="{stroke}" stroke-width="{width:.2f}"{dash_attr}/>'
    )


def svg_rect(
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    fill: str,
    stroke: str = "none",
    stroke_width: float = 1.0,
    opacity: float = 1.0,
) -> str:
    return (
        f'<rect x="{x:.2f}" y="{y:.2f}" width="{width:.2f}" height="{height:.2f}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width:.2f}" '
        f'opacity="{opacity:.3f}"/>'
    )


def create_speedup_boxplot(
    records: list[IterationRecord],
    out_path: Path,
    *,
    accepted_only: bool,
) -> None:
    series = [
        (
            record.iteration,
            record.accepted_speedups if accepted_only else record.speedups,
            record,
        )
        for record in records
    ]
    all_values = [value for _, values, _ in series for value in values]
    if not all_values:
        raise ValueError("No speedup values available for plotting")

    width = 1100
    height = 680
    left = 96
    right = 38
    top = 82
    bottom = 112
    plot_width = width - left - right
    plot_height = height - top - bottom

    log_min = math.floor(math.log10(min(all_values)))
    log_max = math.ceil(math.log10(max(all_values)))
    if log_min == log_max:
        log_min -= 1
        log_max += 1

    def y_pos(value: float) -> float:
        log_value = math.log10(value)
        return top + (log_max - log_value) / (log_max - log_min) * plot_height

    elements: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        svg_rect(0, 0, width, height, fill="#ffffff"),
        svg_text(
            width / 2,
            36,
            "Speedup vs PyTorch Baseline by Iteration",
            size=24,
            weight="700",
        ),
        svg_text(
            width / 2,
            62,
            "Accepted range only" if accepted_only else "All computed speedups (log scale)",
            size=14,
            fill="#52606d",
        ),
    ]

    for exponent in range(log_min, log_max + 1):
        tick_value = 10**exponent
        y = y_pos(tick_value)
        elements.append(svg_line(left, y, width - right, y, stroke="#d9e2ec", width=1.0))
        elements.append(svg_text(left - 12, y + 5, fmt_speedup(tick_value), size=12, anchor="end", fill="#52606d"))

    baseline_y = y_pos(1.0)
    if top <= baseline_y <= top + plot_height:
        elements.append(svg_line(left, baseline_y, width - right, baseline_y, stroke="#c2410c", width=1.5, dash="6 5"))
        elements.append(svg_text(width - right - 4, baseline_y - 8, "PyTorch baseline (1x)", size=12, anchor="end", fill="#9a3412"))

    elements.append(svg_line(left, top, left, top + plot_height, stroke="#243b53", width=1.2))
    elements.append(svg_line(left, top + plot_height, width - right, top + plot_height, stroke="#243b53", width=1.2))
    elements.append(svg_text(28, top + plot_height / 2, "Speedup", size=15, weight="700", rotate=-90))

    count = len(series)
    step = plot_width / count
    box_width = min(92, step * 0.46)
    fill = "#6aa5b8"
    stroke = "#1f5f6f"
    median_color = "#111827"
    outlier_color = "#c2410c"

    for index, (iteration, values, record) in enumerate(series):
        center = left + step * (index + 0.5)
        if not values:
            elements.append(svg_text(center, top + plot_height / 2, "No data", size=12, fill="#9aa5b1"))
            continue

        stats = five_number(values)
        q1_y = y_pos(float(stats["q1"]))
        q3_y = y_pos(float(stats["q3"]))
        median_y = y_pos(float(stats["median"]))
        lower_y = y_pos(float(stats["lower_whisker"]))
        upper_y = y_pos(float(stats["upper_whisker"]))
        box_top = min(q1_y, q3_y)
        box_height = max(abs(q3_y - q1_y), 2.0)

        elements.append(svg_line(center, upper_y, center, lower_y, stroke=stroke, width=1.7))
        elements.append(svg_line(center - box_width * 0.28, upper_y, center + box_width * 0.28, upper_y, stroke=stroke, width=1.7))
        elements.append(svg_line(center - box_width * 0.28, lower_y, center + box_width * 0.28, lower_y, stroke=stroke, width=1.7))
        elements.append(
            svg_rect(
                center - box_width / 2,
                box_top,
                box_width,
                box_height,
                fill=fill,
                stroke=stroke,
                stroke_width=1.5,
                opacity=0.72,
            )
        )
        elements.append(svg_line(center - box_width / 2, median_y, center + box_width / 2, median_y, stroke=median_color, width=2.2))

        outliers = list(stats["outliers"])
        for outlier_index, outlier in enumerate(outliers[:80]):
            jitter = ((outlier_index % 7) - 3) * 2.0
            elements.append(
                f'<circle cx="{center + jitter:.2f}" cy="{y_pos(outlier):.2f}" r="2.6" '
                f'fill="{outlier_color}" opacity="0.45"/>'
            )
        if len(outliers) > 80:
            elements.append(svg_text(center, y_pos(max(outliers)) - 8, f"+{len(outliers) - 80}", size=11, fill=outlier_color))

        elements.append(svg_text(center, top + plot_height + 30, f"Iter {iteration}", size=14, weight="700"))
        elements.append(svg_text(center, top + plot_height + 50, f"n={len(values)}", size=12, fill="#52606d"))
        elements.append(svg_text(center, top + plot_height + 68, f"med {fmt_speedup(float(stats['median']))}", size=12, fill="#52606d"))
        elements.append(svg_text(center, top + plot_height + 86, f"acc {record.exec_rate:.2f}%", size=12, fill="#52606d"))

    elements.append("</svg>")
    out_path.write_text("\n".join(elements), encoding="utf-8")


def create_accuracy_bar_chart(records: list[IterationRecord], out_path: Path) -> None:
    width = 980
    height = 560
    left = 86
    right = 36
    top = 74
    bottom = 98
    plot_width = width - left - right
    plot_height = height - top - bottom

    def y_pos(rate: float) -> float:
        return top + (100.0 - rate) / 100.0 * plot_height

    elements: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        svg_rect(0, 0, width, height, fill="#ffffff"),
        svg_text(width / 2, 36, "Execution Accuracy Across Iterations", size=24, weight="700"),
        svg_text(width / 2, 60, "Phase 2 execution accuracy from iteration JSON summaries", size=14, fill="#52606d"),
    ]

    for tick in range(0, 101, 20):
        y = y_pos(tick)
        elements.append(svg_line(left, y, width - right, y, stroke="#d9e2ec", width=1.0))
        elements.append(svg_text(left - 12, y + 5, f"{tick}%", size=12, anchor="end", fill="#52606d"))

    elements.append(svg_line(left, top, left, top + plot_height, stroke="#243b53", width=1.2))
    elements.append(svg_line(left, top + plot_height, width - right, top + plot_height, stroke="#243b53", width=1.2))
    elements.append(svg_text(28, top + plot_height / 2, "Accuracy", size=15, weight="700", rotate=-90))

    count = len(records)
    step = plot_width / count
    bar_width = min(94, step * 0.52)
    colors = ["#3f8f8c", "#4f9b72", "#9b8b3f", "#a35d4d", "#6d77a8"]

    for index, record in enumerate(records):
        center = left + step * (index + 0.5)
        bar_top = y_pos(record.exec_rate)
        bar_height = top + plot_height - bar_top
        color = colors[index % len(colors)]
        elements.append(
            svg_rect(
                center - bar_width / 2,
                bar_top,
                bar_width,
                bar_height,
                fill=color,
                stroke="#1f2933",
                stroke_width=1.0,
                opacity=0.88,
            )
        )
        elements.append(svg_text(center, bar_top - 12, f"{record.exec_rate:.2f}%", size=13, weight="700", fill="#1f2933"))
        total = infer_total(record)
        passed_label = f"{record.exec_passed}/{total}" if total else str(record.exec_passed)
        elements.append(svg_text(center, bar_top + 20, passed_label, size=12, fill="#ffffff", weight="700"))
        elements.append(svg_text(center, top + plot_height + 30, f"Iter {record.iteration}", size=14, weight="700"))
        elements.append(svg_text(center, top + plot_height + 51, f"computed {record.computed_results}", size=12, fill="#52606d"))
        elements.append(svg_text(center, top + plot_height + 70, f"accepted {record.accepted_results}", size=12, fill="#52606d"))

    elements.append("</svg>")
    out_path.write_text("\n".join(elements), encoding="utf-8")


def maybe_median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def maybe_quantile(values: list[float], key: str) -> float | None:
    if not values:
        return None
    return float(five_number(values)[key])


def infer_total(record: IterationRecord) -> int | None:
    if record.exec_rate <= 0:
        return None
    return round(record.exec_passed / (record.exec_rate / 100.0))


def write_summary_csv(records: list[IterationRecord], out_path: Path) -> None:
    fields = [
        "iteration",
        "call_passed",
        "call_rate",
        "exec_passed",
        "exec_rate",
        "attempted_results",
        "computed_results",
        "accepted_results",
        "json_mean_speedup",
        "speedup_count",
        "speedup_median",
        "speedup_q1",
        "speedup_q3",
        "accepted_speedup_count",
        "accepted_speedup_median",
        "accepted_speedup_q1",
        "accepted_speedup_q3",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "iteration": record.iteration,
                    "call_passed": record.call_passed,
                    "call_rate": f"{record.call_rate:.2f}",
                    "exec_passed": record.exec_passed,
                    "exec_rate": f"{record.exec_rate:.2f}",
                    "attempted_results": record.attempted_results,
                    "computed_results": record.computed_results,
                    "accepted_results": record.accepted_results,
                    "json_mean_speedup": "" if record.json_mean_speedup is None else f"{record.json_mean_speedup:.4f}",
                    "speedup_count": len(record.speedups),
                    "speedup_median": "" if maybe_median(record.speedups) is None else f"{maybe_median(record.speedups):.4f}",
                    "speedup_q1": "" if maybe_quantile(record.speedups, "q1") is None else f"{maybe_quantile(record.speedups, 'q1'):.4f}",
                    "speedup_q3": "" if maybe_quantile(record.speedups, "q3") is None else f"{maybe_quantile(record.speedups, 'q3'):.4f}",
                    "accepted_speedup_count": len(record.accepted_speedups),
                    "accepted_speedup_median": "" if maybe_median(record.accepted_speedups) is None else f"{maybe_median(record.accepted_speedups):.4f}",
                    "accepted_speedup_q1": "" if maybe_quantile(record.accepted_speedups, "q1") is None else f"{maybe_quantile(record.accepted_speedups, 'q1'):.4f}",
                    "accepted_speedup_q3": "" if maybe_quantile(record.accepted_speedups, "q3") is None else f"{maybe_quantile(record.accepted_speedups, 'q3'):.4f}",
                }
            )


def main() -> None:
    args = parse_args()
    records = load_iterations(args.json_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_boxplot = args.out_dir / "speedup_boxplot_all.svg"
    accepted_boxplot = args.out_dir / "speedup_boxplot_accepted.svg"
    accuracy_plot = args.out_dir / "accuracy_by_iteration.svg"
    csv_path = args.out_dir / "iteration_metrics.csv"

    create_speedup_boxplot(records, all_boxplot, accepted_only=False)
    create_speedup_boxplot(records, accepted_boxplot, accepted_only=True)
    create_accuracy_bar_chart(records, accuracy_plot)
    write_summary_csv(records, csv_path)

    for path in [all_boxplot, accepted_boxplot, accuracy_plot, csv_path]:
        print(path)


if __name__ == "__main__":
    main()
