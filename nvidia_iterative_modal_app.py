"""
Iterative NVIDIA-powered TritonBench-T runner on Modal.

This script intentionally does not modify the baseline files. Run it with:

    py -m modal run nvidia_iterative_modal_app.py
    py -m modal run nvidia_iterative_modal_app.py -- --limit 5

It reads NVIDIA_KEY or NVIDIA_API_KEY from the local .env file when Modal
loads the app, generates Triton kernels with NVIDIA's chat-completions API,
benchmarks each generated version, feeds the benchmark result back to the
model, and repeats that loop three times by default. At the end it writes a
best_predictions.jsonl containing the best version found for each operator.

Module layout
-------------
config.py          — all constants and DEFAULT_* values
llm_secrets.py     — .env parsing and Modal Secret resolution
modal_app.py       — Modal App, images (cpu/gpu), and Volume
code_utils.py      — prompt building, NVIDIA API call, code extraction/validation
data_utils.py      — TritonBench dataset loading and file-mapping helpers
perf_utils.py      — perf JSON analysis, batch runner, subprocess helpers
generation.py      — generate_iteration Modal function
evaluation.py      — evaluate_iteration Modal function
best_versions.py   — materialize_best_versions Modal function
formatting.py      — console summary helpers
nvidia_iterative_modal_app.py  — local entrypoint (this file)
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

# Importing the function modules registers their @app.function decorators with
# the Modal app discovered via modal_app.app.
from modal_app import app  # noqa: F401
from generation import generate_iteration  # noqa: F401
from evaluation import evaluate_iteration  # noqa: F401
from best_versions import materialize_best_versions  # noqa: F401

from formatting import (
    _print_config,
    _print_final_summary,
    _print_iteration_summary,
)
from config import (
    DEFAULT_AUTO_OPTIMIZE_MIN_EXEC_RATE,
    DEFAULT_CHECKPOINT_EVERY,
    DEFAULT_CONCURRENCY,
    DEFAULT_FORCE_REGENERATE,
    DEFAULT_GPU,
    DEFAULT_INCLUDE_REFERENCE_SOURCE,
    DEFAULT_ITERATIONS,
    DEFAULT_LOOP_MODE,
    DEFAULT_MAX_FEEDBACK_HISTORY,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_PERF_BATCH_SIZE,
    DEFAULT_REFERENCE_SOURCE_CHAR_LIMIT,
    DEFAULT_REFINE_PASSING,
    DEFAULT_REPAIR_PERF_FAILURES,
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
    DEFAULT_RETRIES,
    DEFAULT_SKIP_EFFICIENCY,
    DEFAULT_TARGET_SPEEDUP,
    DEFAULT_TEMPERATURE,
    DEFAULT_USE_BEST_SO_FAR,
    RUNS_DIR,
    VOLUME_NAME,
)


@app.local_entrypoint()
def main(
    dataset: str = "simp",
    limit: int = 0,
    model: str = DEFAULT_MODEL,
    iterations: int = DEFAULT_ITERATIONS,
    concurrency: int = DEFAULT_CONCURRENCY,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    request_timeout_seconds: int = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    retries: int = DEFAULT_RETRIES,
    checkpoint_every: int = DEFAULT_CHECKPOINT_EVERY,
    include_reference_source: bool = DEFAULT_INCLUDE_REFERENCE_SOURCE,
    reference_source_char_limit: int = DEFAULT_REFERENCE_SOURCE_CHAR_LIMIT,
    use_best_so_far: bool = DEFAULT_USE_BEST_SO_FAR,
    refine_passing: bool = DEFAULT_REFINE_PASSING,
    repair_perf_failures: bool = DEFAULT_REPAIR_PERF_FAILURES,
    max_feedback_history: int = DEFAULT_MAX_FEEDBACK_HISTORY,
    force_regenerate: bool = DEFAULT_FORCE_REGENERATE,
    loop_mode: str = DEFAULT_LOOP_MODE,
    target_speedup: float = DEFAULT_TARGET_SPEEDUP,
    auto_optimize_min_exec_rate: float = DEFAULT_AUTO_OPTIMIZE_MIN_EXEC_RATE,
    perf_batch_size: int = DEFAULT_PERF_BATCH_SIZE,
    skip_efficiency: bool = DEFAULT_SKIP_EFFICIENCY,
    run_id: str = "",
) -> None:
    """Run NVIDIA generation -> benchmark -> feedback refinement loop."""
    if iterations < 1:
        raise ValueError("iterations must be at least 1")
    loop_mode = loop_mode.lower().strip()
    if loop_mode not in {"auto", "correctness", "optimize"}:
        raise ValueError("loop_mode must be one of: auto, correctness, optimize")
    if not run_id:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

    _print_config(
        {
            "run_id": run_id,
            "dataset": dataset,
            "limit": limit or None,
            "model": model,
            "iterations": iterations,
            "concurrency": concurrency,
            "max_tokens": max_tokens,
            "request_timeout_seconds": request_timeout_seconds,
            "retries": retries,
            "include_reference_source": include_reference_source,
            "use_best_so_far": use_best_so_far,
            "refine_passing": refine_passing,
            "repair_perf_failures": repair_perf_failures,
            "max_feedback_history": max_feedback_history,
            "force_regenerate": force_regenerate,
            "loop_mode": loop_mode,
            "target_speedup": target_speedup,
            "auto_optimize_min_exec_rate": auto_optimize_min_exec_rate,
            "perf_batch_size": perf_batch_size,
            "skip_efficiency": skip_efficiency,
            "gpu": DEFAULT_GPU,
        }
    )

    previous_predictions_path = ""
    previous_feedback_by_file: dict[str, dict[str, Any]] = {}
    previous_feedback_history_by_file: dict[str, list[dict[str, Any]]] = {}
    iteration_summaries: list[dict[str, Any]] = []

    for iteration in range(1, iterations + 1):
        generation = generate_iteration.remote(
            run_id=run_id,
            iteration=iteration,
            dataset=dataset,
            limit=limit or None,
            model=model,
            previous_predictions_path=previous_predictions_path,
            previous_feedback_by_file=previous_feedback_by_file,
            previous_feedback_history_by_file=previous_feedback_history_by_file,
            concurrency=concurrency,
            max_tokens=max_tokens,
            temperature=temperature,
            request_timeout_seconds=request_timeout_seconds,
            retries=retries,
            checkpoint_every=checkpoint_every,
            include_reference_source=include_reference_source,
            reference_source_char_limit=reference_source_char_limit,
            refine_passing=refine_passing,
            repair_perf_failures=repair_perf_failures,
            max_feedback_history=max_feedback_history,
            force_regenerate=force_regenerate,
            loop_mode=loop_mode,
            target_speedup=target_speedup,
            auto_optimize_min_exec_rate=auto_optimize_min_exec_rate,
        )
        previous_predictions_path = generation["predictions_path"]

        summary = evaluate_iteration.remote(
            run_id=run_id,
            iteration=iteration,
            predictions_path=previous_predictions_path,
            perf_batch_size=perf_batch_size,
            skip_efficiency=skip_efficiency,
        )
        previous_feedback_by_file = summary["feedback_by_file"]
        for file_name, feedback in previous_feedback_by_file.items():
            previous_feedback_history_by_file.setdefault(file_name, []).append(feedback)
        iteration_summaries.append(summary)

        _print_iteration_summary(summary)

        # Between iterations, replace the previous-iteration baseline with the
        # cross-iteration best so each new generation refines the best-known version
        # per file rather than only the most recent one.
        if use_best_so_far and iteration < iterations:
            interim_best = materialize_best_versions.remote(
                run_id=run_id,
                iteration_summaries=iteration_summaries,
            )
            previous_predictions_path = interim_best["best_predictions_path"]
            previous_feedback_by_file = {
                file_name: selected["result"]
                for file_name, selected in interim_best["best_per_file"][
                    "selected_versions"
                ].items()
            }

    final_summary = materialize_best_versions.remote(
        run_id=run_id,
        iteration_summaries=iteration_summaries,
    )

    _print_final_summary(final_summary)
    print(
        "\nDownload artifacts with:\n"
        f"  py -m modal volume get {VOLUME_NAME} "
        f"{RUNS_DIR}/{run_id} ./local-{run_id}"
    )
