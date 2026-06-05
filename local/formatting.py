"""
Console formatting helpers for human-readable run summaries.
"""

from __future__ import annotations

from typing import Any


def _fmt_optional_speedup(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.4g}x"
    return "n/a"


def _speedup_note(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return ""
    value = float(value)
    if value > 1:
        return f"faster ({value:.2f}x)"
    if value > 0:
        return f"slower ({1 / value:.2f}x)"
    return "invalid"


def _compact_text(value: Any, limit: int = 180) -> str:
    if not value:
        return ""
    text = " ".join(str(value).split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _feedback_error_line(feedback: dict[str, Any] | None) -> str:
    if not feedback:
        return ""
    for phase, key in (
        ("call", "call_error_tail"),
        ("exec", "exec_error_tail"),
        ("perf", "perf_error_tail"),
    ):
        value = feedback.get(key)
        if value:
            return f"{phase}: {_compact_text(value)}"
    return ""


def _print_config(config: dict[str, Any]) -> None:
    print("\n=== Run configuration ===")
    for key in (
        "run_id",
        "dataset",
        "limit",
        "model",
        "iterations",
        "concurrency",
        "loop_mode",
        "target_speedup",
        "auto_optimize_min_exec_rate",
        "perf_batch_size",
        "skip_efficiency",
        "gpu",
    ):
        print(f"{key}: {config.get(key)}")


def _print_iteration_summary(summary: dict[str, Any]) -> None:
    total = summary.get("total_predictions", 0)
    call_acc = summary.get("phase1_call_acc", {})
    exec_acc = summary.get("phase2_exec_acc", {})
    phase3 = summary.get("phase3_efficiency", {})
    feedback = summary.get("feedback_by_file", {})

    print(f"\n=== Iteration {summary.get('iteration')} summary ===")
    print(f"predictions: {summary.get('predictions_path')}")
    print(
        "accuracy: "
        f"call {call_acc.get('passed', 0)}/{total} ({call_acc.get('rate', 0)}%, "
        f"{call_acc.get('cached', 0)} cached) | "
        f"exec {exec_acc.get('passed', 0)}/{total} ({exec_acc.get('rate', 0)}%, "
        f"{exec_acc.get('cached', 0)} cached)"
    )
    print(
        "efficiency: "
        f"aggregate {_fmt_optional_speedup(phase3.get('speedup_vs_pytorch'))} | "
        f"computed mean {_fmt_optional_speedup(phase3.get('computed_mean_speedup_vs_pytorch'))} | "
        f"accepted {phase3.get('accepted_results', 0)}/"
        f"{phase3.get('computed_results', 0)} computed, "
        f"{phase3.get('fresh_attempted_results', phase3.get('attempted_results', 0))} fresh tested, "
        f"{phase3.get('reused_results', 0)} cached"
    )

    per_file = phase3.get("per_file") or {}
    if per_file:
        print("performance:")
        for file_name, record in sorted(per_file.items()):
            speedup = record.get("speedup_vs_pytorch")
            note = _speedup_note(speedup)
            note = f" {note}" if note else ""
            status = str(record.get("status") or "unknown")
            status = status.replace("reused_", "cached ")
            print(
                f"  {file_name}: {status} "
                f"{_fmt_optional_speedup(speedup)}{note} "
                f"(gen {record.get('generated_ms_sum')} ms, "
                f"ref {record.get('reference_ms_sum')} ms, "
                f"cases {record.get('num_cases')})"
            )

    failed = []
    for file_name, item in sorted(feedback.items()):
        if item.get("phase2_exec_passed"):
            continue
        reason = _feedback_error_line(item)
        failed.append((file_name, reason))
    if failed:
        print("not execution-correct:")
        for file_name, reason in failed:
            suffix = f" - {reason}" if reason else ""
            print(f"  {file_name}{suffix}")

    print(f"artifacts: {summary.get('artifacts_subdir')}")


def _print_final_summary(final_summary: dict[str, Any]) -> None:
    best_iter = final_summary.get("best_iteration_as_whole")
    best_iter_summary = final_summary.get("best_iteration_summary", {})
    best_per_file = final_summary.get("best_per_file", {})
    selected = best_per_file.get("selected_versions", {})
    total = best_per_file.get("total", 0)

    print("\n=== Final best-version summary ===")
    print(f"best_predictions: {final_summary.get('best_predictions_path')}")
    print(f"best_generated_scripts: {final_summary.get('best_generated_scripts_dir')}")
    print(f"best whole iteration: {best_iter}")

    call_acc = best_iter_summary.get("phase1_call_acc", {})
    exec_acc = best_iter_summary.get("phase2_exec_acc", {})
    phase3 = best_iter_summary.get("phase3_efficiency", {})
    print(
        "best iteration metrics: "
        f"call {call_acc.get('passed', 0)}/{total} ({call_acc.get('rate', 0)}%) | "
        f"exec {exec_acc.get('passed', 0)}/{total} ({exec_acc.get('rate', 0)}%) | "
        f"speedup {_fmt_optional_speedup(phase3.get('speedup_vs_pytorch'))}"
    )
    print(
        "best per-file: "
        f"execution-correct {best_per_file.get('execution_accurate', 0)}/{total} | "
        f"mean speedup {_fmt_optional_speedup(best_per_file.get('mean_available_speedup_vs_pytorch'))}"
    )

    print("selected versions:")
    for file_name, item in sorted(selected.items()):
        result = item.get("result") or {}
        flags = []
        if result.get("phase2_exec_passed"):
            flags.append("exec-ok")
        else:
            flags.append("failed")
        if result.get("phase3_perf_passed"):
            flags.append("perf-ok")
        speedup = result.get("speedup_vs_pytorch")
        reason = _feedback_error_line(result)
        line = (
            f"  {file_name}: iter {item.get('iteration')} "
            f"{'/'.join(flags)} speedup {_fmt_optional_speedup(speedup)}"
        )
        if reason and not result.get("phase2_exec_passed"):
            line += f" - {reason}"
        print(line)

    print(f"artifacts: {final_summary.get('artifacts_subdir')}")
