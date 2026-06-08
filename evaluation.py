"""
Modal function: run TritonBench-T Phase 1/2/3 and return per-file feedback.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from config import (
    DATA_DIR,
    DEFAULT_GPU,
    DEFAULT_PERF_BATCH_SIZE,
    DEFAULT_SKIP_EFFICIENCY,
    REPO_DIR,
    RUNS_DIR,
)
from modal_app import app, data_volume, image
from code_utils import _static_validate_code, _tail
from data_utils import (
    _files_for_instructions,
    _load_predictions,
    _reload_volume,
    _safe_write_json,
)
from perf_utils import (
    _analyze_perf_results,
    _chunks,
    _merge_perf_outputs,
    _perf_failure_details,
    _run_perf_batch,
    _run_python_file,
    _combined_output_tail,
)


def _ensure_eval_imports() -> None:
    # Dynamically inject the TritonBench EVAL directory into sys.path so the upstream modules can be imported.
    eval_dir = f"{REPO_DIR}/EVAL/eval_T"
    if eval_dir not in sys.path:
        sys.path.insert(0, eval_dir)
    os.environ["PYTHONPATH"] = eval_dir + os.pathsep + os.environ.get("PYTHONPATH", "")


@app.function(
    gpu=DEFAULT_GPU,
    memory=32768,
    timeout=60 * 60 * 6,
    volumes={DATA_DIR: data_volume},
)
def evaluate_iteration(
    run_id: str,
    iteration: int,
    predictions_path: str,
    per_script_timeout_seconds: int = 240,
    perf_batch_size: int = DEFAULT_PERF_BATCH_SIZE,
    skip_efficiency: bool = DEFAULT_SKIP_EFFICIENCY,
) -> dict[str, Any]:
    """Run TritonBench-T phases and return per-file feedback for refinement."""
    _reload_volume()
    _ensure_eval_imports()
    import call_acc  # noqa: E402

    pred_full = Path(DATA_DIR) / predictions_path
    if not pred_full.exists():
        raise FileNotFoundError(f"predictions file not found: {pred_full}")

    # Prepare the output directory for this iteration's evaluation results.
    iter_dir = Path(DATA_DIR) / RUNS_DIR / run_id / f"iter_{iteration:02d}"
    results_dir = iter_dir / "results"
    call_acc_dir = results_dir / "call_acc"
    perf_results_dir = results_dir / "perf_results"
    phase12_checkpoint_path = results_dir / "phase12_checkpoint.json"
    resume_phase12 = phase12_checkpoint_path.exists()
    if results_dir.exists() and not resume_phase12:
        shutil.rmtree(results_dir)
    call_acc_dir.mkdir(parents=True, exist_ok=True)
    perf_results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Evaluating iteration {iteration}: {predictions_path}", flush=True)

    # Read predictions and extract the TritonBench test scripts mapped to each file.
    predictions, tests, files = call_acc.get_codes_for_test(str(pred_full))
    prediction_records = _load_predictions(pred_full)
    instruction_by_file = {
        file_name: record.get("instruction", "")
        for file_name, record in zip(files, prediction_records)
    }
    prediction_record_by_file = dict(zip(files, prediction_records))
    carried_forward_feedback_by_file = {
        file_name: prior_feedback
        for file_name, record in prediction_record_by_file.items()
        if record.get("carried_forward_from_iteration") is not None
        and isinstance(
            prior_feedback := record.get("carried_forward_benchmark_result"),
            dict,
        )
    }
    # Identify files that we already know passed execution correctness from a prior iteration.
    cached_correctness_files = {
        file_name
        for file_name, prior_feedback in carried_forward_feedback_by_file.items()
        if prior_feedback.get("phase2_exec_passed")
    }
    total = len(files)
    # Initialize the default feedback structure for all files.
    feedback: dict[str, dict[str, Any]] = {
        file_name: {
            "file": file_name,
            "iteration": iteration,
            "phase1_call_passed": False,
            "phase2_exec_passed": False,
            "phase3_perf_passed": False,
            "speedup_vs_pytorch": None,
            "static_validation_passed": None,
            "static_validation_errors": [],
            "call_error_tail": "",
            "exec_error_tail": "",
            "perf_error_tail": "",
        }
        for file_name in files
    }

    if resume_phase12:
        # Load a prior Phase 1 & 2 checkpoint if the run was interrupted (e.g., by a timeout).
        phase12_checkpoint = json.loads(
            phase12_checkpoint_path.read_text(encoding="utf-8")
        )
        feedback = phase12_checkpoint["feedback"]
        call_survivors = phase12_checkpoint["call_survivors"]
        exec_survivors = phase12_checkpoint["exec_survivors"]
        cached_correctness_files = set(
            phase12_checkpoint.get("cached_correctness_files", [])
        )
        print(
            "\n=== Phase 1/2: resumed checkpoint ===\n"
            f"Phase 1 survivors: {len(call_survivors)} / {total}\n"
            f"Phase 2 survivors: {len(exec_survivors)} / {total}\n"
            f"Cached correctness results: {len(cached_correctness_files)}",
            flush=True,
        )
    else:
        if cached_correctness_files:
            print(
                "Phase 1/2 reusing cached correctness for "
                f"{len(cached_correctness_files)} carried-forward kernel(s)",
                flush=True,
            )

        # Phase 1: generated module plus the official generated test body must run.
        print("\n=== Phase 1: call accuracy ===", flush=True)
        # TritonBench uses exactly 146 `#` characters as the separator between the
        # generated module and the test body inside each eval file.
        delimiter = "#" * 146
        for idx, (script_content, test_content, file_name) in enumerate(
            zip(predictions, tests, files), start=1
        ):
            if file_name in cached_correctness_files:
                # Skip Phase 1 for carried-forward files and populate their feedback from cache.
                prior_feedback = carried_forward_feedback_by_file[file_name]
                feedback[file_name]["phase1_call_passed"] = True
                feedback[file_name]["phase2_exec_passed"] = True
                feedback[file_name]["static_validation_passed"] = bool(
                    prior_feedback.get("static_validation_passed", True)
                )
                feedback[file_name]["static_validation_errors"] = (
                    prior_feedback.get("static_validation_errors") or []
                )
                temp_path = call_acc_dir / file_name
                temp_path.parent.mkdir(parents=True, exist_ok=True)
                temp_path.write_text(
                    script_content + "\n" + delimiter + "\n" + test_content,
                    encoding="utf-8",
                )
                if idx % 10 == 0 or idx == total:
                    print(f"  phase 1 checked {idx}/{total}", flush=True)
                continue

            # Run static validation to catch syntax/compilation errors without wasting GPU time.
            static_errors = _static_validate_code(
                script_content,
                instruction_by_file.get(file_name, ""),
            )
            feedback[file_name]["static_validation_passed"] = not static_errors
            feedback[file_name]["static_validation_errors"] = static_errors
            if static_errors:
                feedback[file_name]["call_error_tail"] = (
                    "Static validation failed before execution:\n- "
                    + "\n- ".join(static_errors)
                )
                (call_acc_dir / file_name).unlink(missing_ok=True)
                if idx % 10 == 0 or idx == total:
                    print(f"  phase 1 checked {idx}/{total}", flush=True)
                continue

            temp_path = call_acc_dir / file_name
            temp_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path.write_text(
                script_content + "\n" + delimiter + "\n" + test_content,
                encoding="utf-8",
            )
            # Run the combined module + test script in a clean subprocess.
            run_result = _run_python_file(
                temp_path, gpu_id=0, timeout_seconds=per_script_timeout_seconds
            )
            if run_result["returncode"] == 0:
                feedback[file_name]["phase1_call_passed"] = True
            else:
                feedback[file_name]["call_error_tail"] = _combined_output_tail(run_result)
                temp_path.unlink(missing_ok=True)
            if idx % 10 == 0 or idx == total:
                print(f"  phase 1 checked {idx}/{total}", flush=True)

        call_survivors = sorted(path.name for path in call_acc_dir.glob("*.py"))
        print(f"Phase 1 survivors: {len(call_survivors)} / {total}", flush=True)

        # Phase 2: stdout must match the official reference script.
        print("\n=== Phase 2: execution accuracy ===", flush=True)
        gold_folder = Path(REPO_DIR) / "data/TritonBench_T_v1"
        for idx, file_name in enumerate(call_survivors, start=1):
            if file_name in cached_correctness_files:
                # Skip Phase 2 for carried-forward files.
                feedback[file_name]["phase2_exec_passed"] = True
                if idx % 10 == 0 or idx == len(call_survivors):
                    print(f"  phase 2 checked {idx}/{len(call_survivors)}", flush=True)
                continue

            generated_path = call_acc_dir / file_name
            gold_path = gold_folder / file_name
            # Run both the generated script and the golden reference, checking if stdout matches.
            generated_run = _run_python_file(
                generated_path, gpu_id=0, timeout_seconds=per_script_timeout_seconds
            )
            gold_run = _run_python_file(
                gold_path, gpu_id=0, timeout_seconds=per_script_timeout_seconds
            )
            generated_stdout = generated_run.get("stdout") or ""
            gold_stdout = gold_run.get("stdout") or ""
            if generated_run["returncode"] == 0 and generated_stdout == gold_stdout:
                feedback[file_name]["phase2_exec_passed"] = True
            else:
                feedback[file_name]["exec_error_tail"] = _tail(
                    "\n".join(
                        [
                            f"generated returncode: {generated_run['returncode']}",
                            f"gold returncode: {gold_run['returncode']}",
                            "[generated output]",
                            generated_stdout,
                            generated_run.get("stderr") or "",
                            "[gold output]",
                            gold_stdout,
                            gold_run.get("stderr") or "",
                        ]
                    ),
                    2200,
                )
                generated_path.unlink(missing_ok=True)
            if idx % 10 == 0 or idx == len(call_survivors):
                print(f"  phase 2 checked {idx}/{len(call_survivors)}", flush=True)

        exec_survivors = sorted(path.name for path in call_acc_dir.glob("*.py"))
        print(f"Phase 2 survivors: {len(exec_survivors)} / {total}", flush=True)
        # Checkpoint Phase 1 & 2 before starting the long and potentially unstable Phase 3.
        _safe_write_json(
            phase12_checkpoint_path,
            {
                "feedback": feedback,
                "call_survivors": call_survivors,
                "exec_survivors": exec_survivors,
                "cached_correctness_files": sorted(cached_correctness_files),
                "total": total,
            },
        )
        data_volume.commit()

    # Kernels carried forward unchanged already have a valid Phase 3 result from the
    # prior iteration. Reuse it so we don't waste GPU time re-benchmarking identical code.
    reused_perf_analysis: dict[str, dict[str, Any]] = {}
    reused_perf_files: set[str] = set()
    for file_name in exec_survivors:
        record = prediction_record_by_file.get(file_name, {})
        prior_feedback = record.get("carried_forward_benchmark_result")
        if not isinstance(prior_feedback, dict):
            continue
        prior_analysis = prior_feedback.get("phase3_perf_analysis")
        if isinstance(prior_analysis, dict):
            # If detailed prior analysis exists, port it over entirely.
            prior_status = str(prior_analysis.get("status", "unknown"))
            reused_status = (
                prior_status
                if prior_status.startswith("reused_")
                else f"reused_{prior_status}"
            )
            reused_perf_analysis[file_name] = dict(prior_analysis)
            reused_perf_analysis[file_name]["status"] = reused_status
            reused_perf_analysis[file_name]["message"] = (
                "Reused Phase 3 result from carried-forward kernel. "
                + str(prior_analysis.get("message", ""))
            )
        elif isinstance(prior_feedback.get("speedup_vs_pytorch"), (int, float)):
            # Reconstruct basic analysis properties if only a raw speedup was available.
            reused_perf_analysis[file_name] = {
                "file": file_name,
                "result_json": None,
                "golden_json": None,
                "status": (
                    "reused_accepted"
                    if prior_feedback.get("phase3_perf_passed")
                    else "reused_computed"
                ),
                "message": "Reused speedup from carried-forward kernel.",
                "speedup_vs_pytorch": prior_feedback.get("speedup_vs_pytorch"),
                "generated_ms_sum": None,
                "reference_ms_sum": None,
                "num_cases": 0,
            }
        else:
            continue
        reused_perf_files.add(file_name)
        feedback[file_name]["phase3_perf_passed"] = bool(
            prior_feedback.get("phase3_perf_passed")
        )
        feedback[file_name]["phase3_perf_status"] = reused_perf_analysis[file_name][
            "status"
        ]
        feedback[file_name]["phase3_perf_analysis"] = reused_perf_analysis[file_name]
        if isinstance(prior_feedback.get("speedup_vs_pytorch"), (int, float)):
            feedback[file_name]["speedup_vs_pytorch"] = prior_feedback[
                "speedup_vs_pytorch"
            ]
    if reused_perf_files:
        print(
            f"Phase 3 reusing cached perf for {len(reused_perf_files)} carried-forward kernel(s)",
            flush=True,
        )

    try:
        import gc
        import torch

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    except Exception:  # noqa: BLE001
        pass

    # Phase 3: benchmark phase-2 survivors.
    print("\n=== Phase 3: efficiency ===", flush=True)
    efficiency_output = "skipped (no execution-accurate operators)"
    speedup_summary = None
    perf_analysis: dict[str, dict[str, Any]] = {}
    skipped_perf_analysis: dict[str, dict[str, Any]] = {}
    speedups: dict[str, float] = {}
    computed_speedups: dict[str, float] = {}
    runner_details = ""
    perf_run_survivors = [
        file_name for file_name in exec_survivors if file_name not in reused_perf_files
    ]

    if skip_efficiency and perf_run_survivors:
        efficiency_output = "skipped by --skip-efficiency"
        for file_name in perf_run_survivors:
            feedback[file_name]["perf_error_tail"] = efficiency_output
    elif perf_run_survivors:
        if perf_batch_size <= 0:
            perf_batch_size = len(perf_run_survivors)
        perf_root = Path(REPO_DIR) / "performance_metrics/perf_T"
        batch_root = results_dir / "perf_batches"
        batch_root.mkdir(parents=True, exist_ok=True)
        batch_state_path = batch_root / "batch_state.json"
        batch_state = (
            json.loads(batch_state_path.read_text(encoding="utf-8"))
            if batch_state_path.exists()
            else {}
        )

        runner_detail_parts: list[str] = []
        # Group remaining files into batches for efficiency evaluation.
        batches = _chunks(perf_run_survivors, perf_batch_size)
        print(
            f"Phase 3 running {len(perf_run_survivors)} uncached survivor(s) in "
            f"{len(batches)} perf batch(es), batch_size={perf_batch_size}",
            flush=True,
        )
        for batch_index, file_names in enumerate(batches, start=1):
            batch_results_dir = batch_root / f"batch_{batch_index:04d}" / "results"
            batch_key = str(batch_index)
            print(
                f"  perf batch {batch_index}/{len(batches)}: "
                f"{', '.join(file_names)}",
                flush=True,
            )
            previous_state = batch_state.get(batch_key, {})
            if previous_state.get("status") == "done":
                # Skip batches that already completed successfully in a previous interrupted run.
                _merge_perf_outputs(batch_results_dir, perf_results_dir)
                runner_detail_parts.append(
                    f"[perf batch: {', '.join(file_names)}]\nresumed completed batch"
                )
                continue
            # If a batch was "started" but not "done", the container crashed during
            # the perf run (GPU fault or OOM). Mark it skipped so the next resume
            # doesn't retry the same kernel and crash again.
            if previous_state.get("status") == "started":
                message = (
                    "skipped because this batch killed a previous evaluator "
                    "container before it could complete"
                )
                batch_state[batch_key] = {
                    "status": "skipped_after_crash",
                    "files": file_names,
                    "message": message,
                }
                _safe_write_json(batch_state_path, batch_state)
                data_volume.commit()
                runner_detail_parts.append(
                    f"[perf batch: {', '.join(file_names)}]\n{message}"
                )
                for file_name in file_names:
                    skipped_perf_analysis[file_name] = {
                        "file": file_name,
                        "result_json": None,
                        "golden_json": None,
                        "status": "skipped_after_crash",
                        "message": message,
                        "speedup_vs_pytorch": None,
                        "generated_ms_sum": None,
                        "reference_ms_sum": None,
                        "num_cases": 0,
                    }
                continue

            # Mark the batch as started to detect crashes.
            batch_state[batch_key] = {
                "status": "started",
                "files": file_names,
            }
            _safe_write_json(batch_state_path, batch_state)
            data_volume.commit()
            # Execute the batch benchmark.
            runner_detail_parts.append(
                _run_perf_batch(
                    perf_root=perf_root,
                    source_call_acc_dir=call_acc_dir,
                    batch_results_dir=batch_results_dir,
                    file_names=file_names,
                )
            )
            _merge_perf_outputs(batch_results_dir, perf_results_dir)
            # Mark batch completed.
            batch_state[batch_key] = {
                "status": "done",
                "files": file_names,
            }
            _safe_write_json(batch_state_path, batch_state)
            data_volume.commit()

            try:
                import gc
                import torch

                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            except Exception:  # noqa: BLE001
                pass

        runner_details = "\n\n".join(runner_detail_parts)
        perf_analysis = _analyze_perf_results(perf_results_dir)
        perf_analysis.update(
            {
                file_name: record
                for file_name, record in skipped_perf_analysis.items()
                if file_name not in perf_analysis
            }
        )

    perf_analysis.update(
        {
            file_name: record
            for file_name, record in reused_perf_analysis.items()
            if file_name not in perf_analysis
        }
    )

    # Process parsed perf result jsons to evaluate valid speedups.
    accepted_statuses = {"accepted", "reused_accepted"}
    speedups = {
        file_name: record["speedup_vs_pytorch"]
        for file_name, record in perf_analysis.items()
        if record.get("status") in accepted_statuses
        and isinstance(record.get("speedup_vs_pytorch"), (int, float))
    }
    computed_speedups = {
        file_name: record["speedup_vs_pytorch"]
        for file_name, record in perf_analysis.items()
        if isinstance(record.get("speedup_vs_pytorch"), (int, float))
    }

    fresh_accepted_files = {
        file_name
        for file_name, record in perf_analysis.items()
        if record.get("status") == "accepted"
        and isinstance(record.get("speedup_vs_pytorch"), (int, float))
    }

    if speedups:
        speedup_summary = round(sum(speedups.values()) / len(speedups), 2)
        if fresh_accepted_files:
            # Run TritonBench's native efficiency aggregation script to get their canonical metric string.
            efficiency = subprocess.run(
                [
                    sys.executable,
                    "2_efficiency.py",
                    "--gen_folder",
                    str(perf_results_dir),
                ],
                cwd=Path(REPO_DIR) / "EVAL/eval_T",
                capture_output=True,
                text=True,
                check=False,
            )
            efficiency_output = efficiency.stdout
            if efficiency.stderr:
                efficiency_output += "\n[stderr]\n" + efficiency.stderr
            if reused_perf_files:
                efficiency_output = (
                    "Computed aggregate includes cached carried-forward Phase 3 "
                    "results. Upstream 2_efficiency.py output below only covers "
                    "fresh result JSON files.\n\n"
                    + efficiency_output
                )
        else:
            efficiency_output = (
                "All accepted Phase 3 speedups came from cached carried-forward "
                "kernels, so upstream 2_efficiency.py was skipped."
            )
    else:
        result_files = sorted(
            str(path.relative_to(perf_results_dir))
            for path in perf_results_dir.rglob("*.json")
        )
        efficiency_output = (
            "No accepted per-file speedups were produced, so upstream "
            "2_efficiency.py was skipped to avoid its empty-list "
            "ZeroDivisionError.\n\n"
            f"Perf JSON files: {result_files}\n"
            f"Per-file analysis: {json.dumps(perf_analysis, indent=2)}\n\n"
            + runner_details
        )

    # Populate the feedback dictionary with the parsed Phase 3 performance analysis.
    for file_name, analysis_record in perf_analysis.items():
        if file_name in feedback:
            feedback[file_name]["phase3_perf_passed"] = (
                analysis_record.get("status") in accepted_statuses
            )
            feedback[file_name]["phase3_perf_status"] = analysis_record.get("status")
            feedback[file_name]["phase3_perf_analysis"] = analysis_record
            speedup = analysis_record.get("speedup_vs_pytorch")
            if isinstance(speedup, (int, float)):
                feedback[file_name]["speedup_vs_pytorch"] = speedup
    for file_name in exec_survivors:
        if (
            not feedback[file_name]["phase3_perf_passed"]
            and not feedback[file_name].get("perf_error_tail")
        ):
            feedback[file_name]["perf_error_tail"] = _tail(
                _perf_failure_details(perf_results_dir, file_name, perf_analysis)
                + "\n\n"
                + runner_details,
                2400,
            )

    summary = {
        "run_id": run_id,
        "iteration": iteration,
        "predictions_path": predictions_path,
        "total_predictions": total,
        "phase1_call_acc": {
            "passed": len(call_survivors),
            "rate": round(100 * len(call_survivors) / total, 2) if total else 0,
            "cached": len(cached_correctness_files),
        },
        "phase2_exec_acc": {
            "passed": len(exec_survivors),
            "rate": round(100 * len(exec_survivors) / total, 2) if total else 0,
            "cached": len(cached_correctness_files),
        },
        "phase3_efficiency": {
            "speedup_vs_pytorch": speedup_summary,
            "computed_mean_speedup_vs_pytorch": (
                round(sum(computed_speedups.values()) / len(computed_speedups), 4)
                if computed_speedups
                else None
            ),
            "accepted_results": len(speedups),
            "computed_results": len(computed_speedups),
            "attempted_results": len(perf_run_survivors),
            "fresh_attempted_results": len(perf_run_survivors),
            "reused_results": len(reused_perf_files),
            "considered_results": len(exec_survivors),
            "perf_batch_size": perf_batch_size,
            "skip_efficiency": skip_efficiency,
            "per_file": perf_analysis,
            "raw_output_tail": _tail(efficiency_output, 3000),
        },
        "feedback_path": str((results_dir / "per_file_feedback.json").relative_to(DATA_DIR)),
        "artifacts_subdir": str(iter_dir.relative_to(DATA_DIR)),
    }

    _safe_write_json(results_dir / "per_file_feedback.json", feedback)
    _safe_write_json(results_dir / "summary.json", summary)
    data_volume.commit()

    summary["feedback_by_file"] = feedback
    return summary
