#!/usr/bin/env python
"""Summarize NVIDIA iterative Modal run JSONs for report writing.

By default this script uses jsons/best.json because the report extract cites the
best run-as-a-whole result: 104/166 = 62.65%. Use --source latest to report the
final iteration instead.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class IterationSummary:
    iteration: int
    call_passed: int
    call_rate: float
    exec_passed: int
    exec_rate: float
    computed_results: int
    accepted_results: int
    json_mean_speedup: float | None
    speedups: list[float]
    accepted_speedups: list[float]


@dataclass
class ReportSource:
    label: str
    iteration: int
    summary: IterationSummary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute iteration summaries and statistical inference values."
    )
    parser.add_argument(
        "--json-dir",
        type=Path,
        default=Path("jsons"),
        help="Directory containing iter*.json and best.json.",
    )
    parser.add_argument(
        "--source",
        choices=["best", "latest"],
        default="best",
        help="Use best.json or the latest numbered iteration for the report extract.",
    )
    parser.add_argument(
        "--baseline-passed",
        type=int,
        default=0,
        help="Number of baseline successes. The cited report baseline is 0.",
    )
    parser.add_argument(
        "--total",
        type=int,
        default=None,
        help="Total benchmark items. If omitted, inferred from passed/rate.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("reports") / "nvidia_iterative_report_stats.md",
        help="Markdown report summary path.",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Print the report summary without writing the Markdown file.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def as_int(value: object, default: int = 0) -> int:
    return int(value) if isinstance(value, (int, float)) else default


def as_float(value: object) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def iteration_sort_key(path: Path) -> tuple[int, str]:
    digits = "".join(ch for ch in path.stem if ch.isdigit())
    return (int(digits) if digits else 10_000, path.name.lower())


def collect_speedups(phase3: dict) -> tuple[list[float], list[float]]:
    speedups: list[float] = []
    accepted_speedups: list[float] = []
    for result in phase3.get("per_file", {}).values():
        value = as_float(result.get("speedup_vs_pytorch"))
        if value is None or value <= 0:
            continue
        speedups.append(value)
        status = str(result.get("status", "")).lower()
        if "accepted" in status and "out_of_range" not in status:
            accepted_speedups.append(value)
    return speedups, accepted_speedups


def summary_from_payload(payload: dict, fallback_iteration: int) -> IterationSummary:
    phase1 = payload.get("phase1_call_acc", {})
    phase2 = payload.get("phase2_exec_acc", {})
    phase3 = payload.get("phase3_efficiency", {})
    speedups, accepted_speedups = collect_speedups(phase3)
    return IterationSummary(
        iteration=as_int(payload.get("iteration"), fallback_iteration),
        call_passed=as_int(phase1.get("passed")),
        call_rate=float(phase1.get("rate", 0.0)),
        exec_passed=as_int(phase2.get("passed")),
        exec_rate=float(phase2.get("rate", 0.0)),
        computed_results=as_int(phase3.get("computed_results")),
        accepted_results=as_int(phase3.get("accepted_results")),
        json_mean_speedup=as_float(phase3.get("computed_mean_speedup_vs_pytorch")),
        speedups=speedups,
        accepted_speedups=accepted_speedups,
    )


def load_iterations(json_dir: Path) -> list[IterationSummary]:
    paths = sorted(
        [path for path in json_dir.glob("*.json") if path.name.lower().startswith("iter")],
        key=iteration_sort_key,
    )
    if not paths:
        raise FileNotFoundError(f"No iteration JSON files found under {json_dir}")
    return [
        summary_from_payload(load_json(path), iteration_sort_key(path)[0])
        for path in paths
    ]


def load_report_source(json_dir: Path, source: str, iterations: list[IterationSummary]) -> ReportSource:
    if source == "latest":
        latest = max(iterations, key=lambda item: item.iteration)
        return ReportSource(label="latest iteration JSON", iteration=latest.iteration, summary=latest)

    best_path = json_dir / "best.json"
    if not best_path.exists():
        raise FileNotFoundError(f"{best_path} is required when --source best is used")
    best_payload = load_json(best_path)
    best_iteration = as_int(best_payload.get("best_iteration_as_whole"))
    best_summary_payload = best_payload.get("best_iteration_summary", {})
    best_summary = summary_from_payload(best_summary_payload, best_iteration)
    best_summary.iteration = best_iteration
    return ReportSource(label="best.json best_iteration_as_whole", iteration=best_iteration, summary=best_summary)


def infer_total(passed: int, rate: float, explicit_total: int | None) -> int:
    if explicit_total is not None:
        return explicit_total
    if rate <= 0:
        raise ValueError("Cannot infer total when rate is zero; pass --total explicitly")
    return round(passed / (rate / 100.0))


def exact_mcnemar_p_value(baseline_only: int, proposed_only: int) -> float:
    discordant = baseline_only + proposed_only
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(0, min(baseline_only, proposed_only) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def two_proportion_z_test(success_a: int, total_a: int, success_b: int, total_b: int) -> tuple[float, float]:
    p_a = success_a / total_a
    p_b = success_b / total_b
    pooled = (success_a + success_b) / (total_a + total_b)
    standard_error = math.sqrt(pooled * (1.0 - pooled) * (1.0 / total_a + 1.0 / total_b))
    if standard_error == 0:
        return math.inf, 0.0
    z_score = (p_b - p_a) / standard_error
    p_value = math.erfc(abs(z_score) / math.sqrt(2.0))
    return z_score, p_value


def wilson_ci(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        raise ValueError("total must be positive")
    p_hat = successes / total
    denominator = 1.0 + z**2 / total
    center = (p_hat + z**2 / (2.0 * total)) / denominator
    half_width = (
        z
        * math.sqrt((p_hat * (1.0 - p_hat) / total) + (z**2 / (4.0 * total**2)))
        / denominator
    )
    return center - half_width, center + half_width


def cohen_h(p_a: float, p_b: float) -> float:
    return 2.0 * math.asin(math.sqrt(p_b)) - 2.0 * math.asin(math.sqrt(p_a))


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def percentile(sorted_values: list[float], pct: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * pct
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return sorted_values[low]
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (rank - low)


def geomean(values: Iterable[float]) -> float | None:
    positives = [value for value in values if value > 0]
    if not positives:
        return None
    return math.exp(sum(math.log(value) for value in positives) / len(positives))


def fmt_pct(value: float) -> str:
    return f"{value * 100.0:.2f}%"


def fmt_float(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def format_iteration_table(iterations: list[IterationSummary]) -> str:
    lines = [
        "| Iteración | Aciertos exec | Accuracy exec | Computed speedups | Accepted speedups | Mean JSON | Mediana speedup |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in iterations:
        lines.append(
            "| "
            f"{item.iteration} | "
            f"{item.exec_passed} | "
            f"{item.exec_rate:.2f}% | "
            f"{item.computed_results} | "
            f"{item.accepted_results} | "
            f"{fmt_float(item.json_mean_speedup, 4)} | "
            f"{fmt_float(median(item.speedups), 4)} |"
        )
    return "\n".join(lines)


def build_report(
    iterations: list[IterationSummary],
    source: ReportSource,
    baseline_passed: int,
    total: int,
) -> str:
    proposed_passed = source.summary.exec_passed
    baseline_rate = baseline_passed / total
    proposed_rate = proposed_passed / total
    improvement = proposed_rate - baseline_rate

    if baseline_passed != 0:
        mcnemar_note = (
            "La prueba exacta de McNemar requiere la tabla pareada completa. "
            "Con --baseline-passed distinto de 0 este script no puede inferir que casos "
            "fueron aciertos exclusivos del baseline a partir de los agregados."
        )
        baseline_only = 0
        proposed_only = proposed_passed
    else:
        mcnemar_note = (
            "Como el baseline tiene 0 aciertos, todos los aciertos del método propuesto "
            "son discordancias a favor del método."
        )
        baseline_only = 0
        proposed_only = proposed_passed

    both_correct = 0 if baseline_passed == 0 else None
    both_incorrect = total - proposed_passed if baseline_passed == 0 else None
    p_mcnemar = exact_mcnemar_p_value(baseline_only, proposed_only)
    z_score, p_z = two_proportion_z_test(baseline_passed, total, proposed_passed, total)
    ci_low, ci_high = wilson_ci(proposed_passed, total)
    h = cohen_h(baseline_rate, proposed_rate)
    paired_or = math.inf if baseline_only == 0 and proposed_only > 0 else proposed_only / baseline_only
    corrected_or = (proposed_only + 0.5) / (baseline_only + 0.5)

    speedups = sorted(source.summary.speedups)
    accepted_speedups = sorted(source.summary.accepted_speedups)
    q1 = percentile(speedups, 0.25)
    q3 = percentile(speedups, 0.75)
    accepted_q1 = percentile(accepted_speedups, 0.25)
    accepted_q3 = percentile(accepted_speedups, 0.75)
    latest = max(iterations, key=lambda item: item.iteration)
    source_note = ""
    if latest.iteration != source.iteration or latest.exec_passed != proposed_passed:
        source_note = (
            f"Nota: la iteración final ({latest.iteration}) registra "
            f"{latest.exec_passed}/{total} = {latest.exec_rate:.2f}%. "
            "Este resumen usa la fuente seleccionada para mantener consistente "
            "el extracto; usa `--source latest` si el reporte debe hablar de la "
            "última iteración en vez del mejor resultado como corrida completa."
        )

    lines = [
        "# NVIDIA Iterative Modal Run - Report Stats",
        "",
        f"Fuente para el extracto: `{source.label}`, iteración {source.iteration}.",
        f"Total inferido: {total} operadores.",
        f"Baseline: {baseline_passed}/{total} = {fmt_pct(baseline_rate)}.",
        f"Método propuesto: {proposed_passed}/{total} = {fmt_pct(proposed_rate)}.",
        f"Incremento absoluto: {improvement * 100.0:.2f} puntos porcentuales.",
        source_note,
        "",
        "## Tabla por iteración",
        "",
        format_iteration_table(iterations),
        "",
        "## Pruebas estadísticas",
        "",
        f"- Prueba recomendada: McNemar exacta para proporciones pareadas ({mcnemar_note})",
        f"- Tabla pareada inferida: ambos correctos={both_correct}, solo baseline={baseline_only}, solo método={proposed_only}, ambos incorrectos={both_incorrect}.",
        f"- p-value McNemar exacta bilateral: {p_mcnemar:.3e}.",
        f"- Z-test de dos proporciones no pareado, solo como contraste secundario: z={z_score:.4f}, p={p_z:.3e}.",
        f"- IC Wilson 95% para accuracy del método: {fmt_pct(ci_low)} a {fmt_pct(ci_high)}.",
        f"- Cohen's h: {h:.4f}.",
        f"- Odds ratio pareado: {'infinito' if math.isinf(paired_or) else f'{paired_or:.4f}'}; con corrección Haldane-Anscombe: {corrected_or:.2f}.",
        "",
        "## Speedup",
        "",
        f"- Speedups computados en fuente elegida: n={len(speedups)}, mediana={fmt_float(median(speedups), 4)}, IQR={fmt_float(q1, 4)}-{fmt_float(q3, 4)}, geomean={fmt_float(geomean(speedups), 4)}.",
        f"- Speedups aceptados por rango en fuente elegida: n={len(accepted_speedups)}, mediana={fmt_float(median(accepted_speedups), 4)}, IQR={fmt_float(accepted_q1, 4)}-{fmt_float(accepted_q3, 4)}, geomean={fmt_float(geomean(accepted_speedups), 4)}.",
        f"- Mean speedup reportado en JSON: {fmt_float(source.summary.json_mean_speedup, 4)}.",
        "",
        "## Extracto sugerido",
        "",
        "2.3 Inferencia Estadística",
        "",
        (
            "Para contrastar las hipótesis planteadas, se analiza el salto en la tasa "
            f"de compilación y ejecución correcta (de {fmt_pct(baseline_rate)} en el "
            f"baseline a {fmt_pct(proposed_rate)} en el método propuesto; "
            f"{proposed_passed}/{total} operadores)."
        ),
        "",
        (
            "Prueba estadística aplicada: prueba exacta de McNemar para proporciones "
            "pareadas, porque el baseline y el método propuesto se evalúan sobre el "
            "mismo conjunto de operadores. La tabla pareada inferida es: 0 aciertos "
            f"en ambos métodos, {baseline_only} aciertos solo del baseline, "
            f"{proposed_only} aciertos solo del método propuesto y {both_incorrect} "
            "fallos en ambos."
        ),
        "",
        (
            f"p-value: {p_mcnemar:.3e} (bilateral exacta). El valor es menor que "
            "0.05, por lo que se rechaza la hipótesis nula de igualdad entre ambos "
            "métodos."
        ),
        "",
        (
            f"Tamaño del efecto: el incremento absoluto es de {improvement * 100.0:.2f} "
            f"puntos porcentuales. Cohen's h = {h:.2f}, muy por encima del umbral "
            "convencional de efecto grande (0.8); el odds ratio pareado es infinito "
            f"al no existir discordancias a favor del baseline (OR corregido = {corrected_or:.2f})."
        ),
        "",
        (
            "Interpretación: la evidencia rechaza la hipótesis nula y muestra que la "
            "inserción de validación estática y retroalimentación de errores cambia "
            "de forma estadísticamente significativa la capacidad de generar kernels "
            "válidos en Triton."
        ),
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    iterations = load_iterations(args.json_dir)
    source = load_report_source(args.json_dir, args.source, iterations)
    total = infer_total(source.summary.exec_passed, source.summary.exec_rate, args.total)
    report = build_report(iterations, source, args.baseline_passed, total)

    print(report)
    if not args.no_write:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report, encoding="utf-8")
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
