"""
Modal function: select and materialize the best kernel version per file across all iterations.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from config import DATA_DIR, RUNS_DIR, VOLUME_NAME
from modal_app import app, cpu_image, data_volume
from code_utils import _extract_code
from data_utils import (
    _files_for_instructions,
    _load_predictions,
    _reload_volume,
    _safe_write_json,
)


def _score_feedback(
    feedback: dict[str, Any] | None,
    iteration: int,
) -> tuple[int, float, int, int, int]:
    # Lexicographic priority: execution correctness > accepted speedup > call accuracy
    # > static validity > later iteration (tie-break so the newest correct version wins).
    if not feedback:
        return (0, -1.0, 0, 0, iteration)
    phase2 = 1 if feedback.get("phase2_exec_passed") else 0
    phase1 = 1 if feedback.get("phase1_call_passed") else 0
    static_ok = 1 if feedback.get("static_validation_passed") else 0
    speedup = feedback.get("speedup_vs_pytorch")
    speed_value = (
        float(speedup)
        if feedback.get("phase3_perf_passed") and isinstance(speedup, (int, float))
        else -1.0
    )
    return (phase2, speed_value if phase2 else -1.0, phase1, static_ok, iteration)


@app.function(
    image=cpu_image,
    timeout=60 * 30,
    volumes={DATA_DIR: data_volume},
)
def materialize_best_versions(
    run_id: str,
    iteration_summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    _reload_volume()
    run_dir = Path(DATA_DIR) / RUNS_DIR / run_id
    best_dir = run_dir / "best"
    best_generated_dir = best_dir / "generated_scripts"
    best_generated_dir.mkdir(parents=True, exist_ok=True)

    by_file: dict[str, dict[str, Any]] = {}
    for summary in iteration_summaries:
        iteration = summary["iteration"]
        predictions = _load_predictions(Path(DATA_DIR) / summary["predictions_path"])
        files = _files_for_instructions([record["instruction"] for record in predictions])
        feedback_by_file = summary.get("feedback_by_file", {})

        for file_name, record in zip(files, predictions):
            feedback = feedback_by_file.get(file_name)
            candidate = {
                "file": file_name,
                "iteration": iteration,
                "record": record,
                "feedback": feedback,
                "score": _score_feedback(feedback, iteration),
            }
            current = by_file.get(file_name)
            if current is None or candidate["score"] > current["score"]:
                by_file[file_name] = candidate

    best_predictions_path = best_dir / "best_predictions.jsonl"
    with best_predictions_path.open("w", encoding="utf-8") as handle:
        for file_name in sorted(by_file):
            candidate = by_file[file_name]
            record = candidate["record"]
            code = _extract_code(record.get("predict", ""))
            best_script_path = best_generated_dir / file_name
            best_script_path.parent.mkdir(parents=True, exist_ok=True)
            best_script_path.write_text(code, encoding="utf-8")
            best_record = {
                "instruction": record["instruction"],
                "predict": code,
                "file": file_name,
                "best_iteration": candidate["iteration"],
                "benchmark_result": candidate["feedback"],
            }
            handle.write(json.dumps(best_record) + "\n")

    best_iteration = max(
        iteration_summaries,
        key=lambda item: (
            item["phase2_exec_acc"]["passed"],
            item["phase3_efficiency"].get("speedup_vs_pytorch") or -1,
            item["phase1_call_acc"]["passed"],
        ),
    )
    best_exec_count = sum(
        1
        for candidate in by_file.values()
        if candidate["feedback"] and candidate["feedback"].get("phase2_exec_passed")
    )
    best_speedups = [
        candidate["feedback"]["speedup_vs_pytorch"]
        for candidate in by_file.values()
        if candidate["feedback"]
        and candidate["feedback"].get("phase3_perf_passed")
        and isinstance(candidate["feedback"].get("speedup_vs_pytorch"), (int, float))
    ]

    final_summary = {
        "run_id": run_id,
        "best_predictions_path": str(best_predictions_path.relative_to(DATA_DIR)),
        "best_generated_scripts_dir": str(best_generated_dir.relative_to(DATA_DIR)),
        "best_iteration_as_whole": best_iteration["iteration"],
        "best_iteration_summary": {
            "phase1_call_acc": best_iteration["phase1_call_acc"],
            "phase2_exec_acc": best_iteration["phase2_exec_acc"],
            "phase3_efficiency": best_iteration["phase3_efficiency"],
        },
        "best_per_file": {
            "total": len(by_file),
            "execution_accurate": best_exec_count,
            "mean_available_speedup_vs_pytorch": (
                round(sum(best_speedups) / len(best_speedups), 4)
                if best_speedups
                else None
            ),
            "selected_versions": {
                file_name: {
                    "iteration": candidate["iteration"],
                    "result": candidate["feedback"],
                }
                for file_name, candidate in sorted(by_file.items())
            },
        },
        "artifacts_volume": VOLUME_NAME,
        "artifacts_subdir": str(run_dir.relative_to(DATA_DIR)),
    }
    _safe_write_json(best_dir / "final_summary.json", final_summary)
    data_volume.commit()
    return final_summary
