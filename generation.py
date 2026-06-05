"""
Modal function: generate one full prediction set per iteration.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from config import (
    DATA_DIR,
    DEFAULT_AUTO_OPTIMIZE_MIN_EXEC_RATE,
    DEFAULT_CHECKPOINT_EVERY,
    DEFAULT_CONCURRENCY,
    DEFAULT_FORCE_REGENERATE,
    DEFAULT_INCLUDE_REFERENCE_SOURCE,
    DEFAULT_LOOP_MODE,
    DEFAULT_MAX_FEEDBACK_HISTORY,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_REFERENCE_SOURCE_CHAR_LIMIT,
    DEFAULT_REFINE_PASSING,
    DEFAULT_REPAIR_PERF_FAILURES,
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
    DEFAULT_RETRIES,
    DEFAULT_TARGET_SPEEDUP,
    DEFAULT_TEMPERATURE,
    RUNS_DIR,
)

DEFAULT_USE_GUIDED_JSON = False
from modal_app import app, cpu_image, data_volume
from llm_secrets import LLM_SECRET
from code_utils import (
    _build_messages,
    _extract_code,
    _gpu_issue_feedback,
    _llm_chat,
    _phase2_passed,
    _speedup_value,
    _static_validate_code,
)
from data_utils import (
    _files_for_instructions,
    _load_alpaca,
    _prediction_code_by_file,
    _reference_context_for_file,
    _reload_volume,
    _safe_write_json,
)


@app.function(
    image=cpu_image,
    timeout=60 * 60 * 4,
    cpu=4,
    volumes={DATA_DIR: data_volume},
    secrets=[LLM_SECRET],
)
def generate_iteration(
    run_id: str,
    iteration: int,
    dataset: str = "simp",
    limit: int | None = None,
    model: str = DEFAULT_MODEL,
    previous_predictions_path: str = "",
    previous_feedback_by_file: dict[str, dict[str, Any]] | None = None,
    previous_feedback_history_by_file: dict[str, list[dict[str, Any]]] | None = None,
    concurrency: int = DEFAULT_CONCURRENCY,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    request_timeout_seconds: int = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    retries: int = DEFAULT_RETRIES,
    checkpoint_every: int = DEFAULT_CHECKPOINT_EVERY,
    retry_failed_records: bool = True,
    include_reference_source: bool = DEFAULT_INCLUDE_REFERENCE_SOURCE,
    reference_source_char_limit: int = DEFAULT_REFERENCE_SOURCE_CHAR_LIMIT,
    refine_passing: bool = DEFAULT_REFINE_PASSING,
    repair_perf_failures: bool = DEFAULT_REPAIR_PERF_FAILURES,
    max_feedback_history: int = DEFAULT_MAX_FEEDBACK_HISTORY,
    force_regenerate: bool = DEFAULT_FORCE_REGENERATE,
    loop_mode: str = DEFAULT_LOOP_MODE,
    target_speedup: float = DEFAULT_TARGET_SPEEDUP,
    auto_optimize_min_exec_rate: float = DEFAULT_AUTO_OPTIMIZE_MIN_EXEC_RATE,
    use_guided_json: bool = DEFAULT_USE_GUIDED_JSON,
) -> dict[str, Any]:
    """Generate one complete prediction set for an iteration."""
    _reload_volume()
    loop_mode = loop_mode.lower().strip()
    if loop_mode not in {"auto", "correctness", "optimize"}:
        raise ValueError("loop_mode must be one of: auto, correctness, optimize")
    previous_feedback_by_file = previous_feedback_by_file or {}
    previous_feedback_history_by_file = previous_feedback_history_by_file or {}
    previous_exec_rate = (
        sum(1 for item in previous_feedback_by_file.values() if _phase2_passed(item))
        / len(previous_feedback_by_file)
        if previous_feedback_by_file
        else 0.0
    )

    items = _load_alpaca(dataset)
    if limit:
        items = items[:limit]
    files = _files_for_instructions([item["instruction"] for item in items])
    previous_code_by_file = (
        _prediction_code_by_file(previous_predictions_path)
        if previous_predictions_path
        else {}
    )

    iter_dir = Path(DATA_DIR) / RUNS_DIR / run_id / f"iter_{iteration:02d}"
    generated_dir = iter_dir / "generated_scripts"
    records_dir = iter_dir / "generation_records"
    prompts_dir = iter_dir / "prompts"
    generated_dir.mkdir(parents=True, exist_ok=True)
    records_dir.mkdir(parents=True, exist_ok=True)
    prompts_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = iter_dir / "predictions.jsonl"
    volume_predictions_path = str(predictions_path.relative_to(DATA_DIR))

    print(
        f"Generating iteration {iteration} with {model}; "
        f"items={len(items)} concurrency={concurrency} max_tokens={max_tokens} "
        f"timeout={request_timeout_seconds}s retries={retries} "
        f"mode={loop_mode} target_speedup={target_speedup} "
        f"prev_exec_rate={previous_exec_rate:.2f}",
        flush=True,
    )

    def _one(idx_item_file: tuple[int, dict[str, Any], str]) -> tuple[int, dict[str, Any]]:
        idx, item, file_name = idx_item_file
        feedback = previous_feedback_by_file.get(file_name)
        task_mode = (
            "optimize"
            if (
                loop_mode == "optimize"
                or (
                    loop_mode == "auto"
                    and previous_exec_rate >= auto_optimize_min_exec_rate
                )
            )
            and _phase2_passed(feedback)
            else "correctness"
        )
        feedback_history = previous_feedback_history_by_file.get(file_name, [])
        if feedback:
            selected_feedback_iteration = feedback.get("iteration")
            feedback_history = [
                item
                for item in feedback_history
                if item.get("iteration") != selected_feedback_iteration
            ]
        if max_feedback_history > 0:
            feedback_history = feedback_history[-max_feedback_history:]
        previous_code = previous_code_by_file.get(file_name)
        reference_context = (
            _reference_context_for_file(file_name, reference_source_char_limit)
            if include_reference_source
            else ""
        )
        messages = _build_messages(
            item=item,
            file_name=file_name,
            previous_code=previous_code,
            previous_feedback=feedback,
            previous_feedback_history=feedback_history,
            iteration=iteration,
            reference_context=reference_context,
            task_mode=task_mode,
            target_speedup=target_speedup,
            use_guided_json=use_guided_json,
        )
        (prompts_dir / f"{idx:04d}_{file_name}.json").write_text(
            json.dumps(messages, indent=2), encoding="utf-8"
        )
        try:
            raw = _llm_chat(
                messages,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                request_timeout_seconds=request_timeout_seconds,
                retries=retries,
                use_guided_json=use_guided_json,
            )
            code = _extract_code(raw)
            error = ""
        except Exception as exc:  # noqa: BLE001
            code = f"# generation failed: {exc}\n"
            error = str(exc)

        script_path = generated_dir / file_name
        script_path.parent.mkdir(parents=True, exist_ok=True)
        script_path.write_text(code, encoding="utf-8")

        # "instruction" must be first: TritonBench's 0_call_acc.py reads the first
        # key of each JSON object to map the row back to its reference file.
        record = {
            "instruction": item["instruction"],
            "predict": code,
            "file": file_name,
            "iteration": iteration,
            "task_mode": task_mode,
        }
        if error:
            record["generation_error"] = error
        return idx, record

    # First pass: reuse existing records or carry-forward passing kernels to avoid
    # burning API quota on operators that already work. Only truly new or broken
    # items end up in `pending` for the concurrent generation step.
    results: list[dict[str, Any] | None] = [None] * len(items)
    pending: list[tuple[int, dict[str, Any], str]] = []
    gpu_issue_regenerations = 0
    for idx, (item, file_name) in enumerate(zip(items, files)):
        record_path = records_dir / f"{idx:04d}.json"
        if record_path.exists() and not force_regenerate:
            record = json.loads(record_path.read_text(encoding="utf-8"))
            if retry_failed_records and record.get("generation_error"):
                pending.append((idx, item, file_name))
            else:
                results[idx] = record
                script_path = generated_dir / file_name
                script_path.parent.mkdir(parents=True, exist_ok=True)
                script_path.write_text(
                    _extract_code(record.get("predict", "")), encoding="utf-8"
                )
        else:
            feedback = previous_feedback_by_file.get(file_name) or {}
            previous_code = previous_code_by_file.get(file_name)
            speedup = _speedup_value(feedback)
            phase2_ok = _phase2_passed(feedback)
            gpu_issue = _gpu_issue_feedback(feedback)
            meets_speed_target = (
                bool(feedback.get("phase3_perf_passed"))
                and speedup is not None
                and speedup >= target_speedup
            )
            carry_forward = False
            carry_reason = ""
            if previous_code and phase2_ok and not refine_passing and not gpu_issue:
                if loop_mode == "correctness":
                    if (
                        not repair_perf_failures
                        or feedback.get("phase3_perf_passed")
                        or speedup is not None
                    ):
                        carry_forward = True
                        carry_reason = "previous version passed execution"
                elif loop_mode == "auto" and previous_exec_rate < auto_optimize_min_exec_rate:
                    carry_forward = True
                    carry_reason = (
                        "previous version passed execution; auto mode is still "
                        "prioritizing correctness"
                    )
                elif loop_mode == "auto" and not refine_passing:
                    carry_forward = True
                    carry_reason = (
                        "previous version passed execution; pass --refine-passing "
                        "to let auto rewrite correct kernels for speed"
                    )
                elif loop_mode in {"auto", "optimize"} and meets_speed_target:
                    carry_forward = True
                    carry_reason = (
                        f"previous version met target speedup {target_speedup:.3g}x"
                    )
            elif previous_code and phase2_ok and gpu_issue:
                gpu_issue_regenerations += 1
            if (
                previous_code
                and not refine_passing
                and carry_forward
            ):
                record = {
                    "instruction": item["instruction"],
                    "predict": previous_code,
                    "file": file_name,
                    "iteration": iteration,
                    "carried_forward_from_iteration": feedback.get("iteration"),
                    "carry_forward_reason": carry_reason,
                    "carried_forward_benchmark_result": feedback,
                }
                results[idx] = record
                record_path.write_text(json.dumps(record), encoding="utf-8")
                script_path = generated_dir / file_name
                script_path.parent.mkdir(parents=True, exist_ok=True)
                script_path.write_text(previous_code, encoding="utf-8")
            else:
                pending.append((idx, item, file_name))

    if len(pending) != len(items):
        print(
            f"  resumed {len(items) - len(pending)}/{len(items)} existing generations",
            flush=True,
        )
    if gpu_issue_regenerations:
        print(
            "  regenerating "
            f"{gpu_issue_regenerations} previously execution-correct kernel(s) "
            "because their prior perf run hit a GPU/container fault",
            flush=True,
        )

    done = 0
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as executor:
        futures = [executor.submit(_one, item_file) for item_file in pending]
        for future in as_completed(futures):
            idx, record = future.result()
            results[idx] = record
            (records_dir / f"{idx:04d}.json").write_text(
                json.dumps(record), encoding="utf-8"
            )
            done += 1
            completed = len(items) - len(pending) + done
            elapsed = time.monotonic() - started
            avg = elapsed / done if done else 0
            print(
                f"  generated {completed}/{len(items)} "
                f"(new {done}/{len(pending)}, avg {avg:.1f}s/new item)",
                flush=True,
            )
            if done % max(1, checkpoint_every) == 0:
                data_volume.commit()

    with predictions_path.open("w", encoding="utf-8") as handle:
        for record in results:
            if record is None:
                raise RuntimeError("internal generation error: missing record")
            handle.write(json.dumps(record) + "\n")

    manifest = {
        "run_id": run_id,
        "iteration": iteration,
        "dataset": dataset,
        "limit": limit,
        "model": model,
        "include_reference_source": include_reference_source,
        "refine_passing": refine_passing,
        "repair_perf_failures": repair_perf_failures,
        "force_regenerate": force_regenerate,
        "loop_mode": loop_mode,
        "target_speedup": target_speedup,
        "auto_optimize_min_exec_rate": auto_optimize_min_exec_rate,
        "use_guided_json": use_guided_json,
        "previous_exec_rate": previous_exec_rate,
        "predictions_path": volume_predictions_path,
        "generated_scripts_dir": str(generated_dir.relative_to(DATA_DIR)),
        "generation_records_dir": str(records_dir.relative_to(DATA_DIR)),
        "prompts_dir": str(prompts_dir.relative_to(DATA_DIR)),
        "total_predictions": len(items),
    }
    _safe_write_json(iter_dir / "generation_manifest.json", manifest)
    data_volume.commit()
    return manifest
