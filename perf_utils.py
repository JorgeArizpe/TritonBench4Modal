"""
Performance benchmark helpers: running scripts, parsing perf JSON, batch management.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from config import REPO_DIR
from code_utils import _tail


# --------------------------------------------------------------------------- #
# Subprocess execution
# --------------------------------------------------------------------------- #


def _run_python_file(path: Path, gpu_id: int, timeout_seconds: int) -> dict[str, Any]:
    def _text(value: str | bytes | None) -> str:
        if value is None:
            return ""
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return value

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    try:
        result = subprocess.run(
            [sys.executable, str(path)],
            capture_output=True,
            text=True,
            env=env,
            timeout=timeout_seconds,
        )
        return {
            "returncode": result.returncode,
            "stdout": _text(result.stdout),
            "stderr": _text(result.stderr),
            "timed_out": False,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "returncode": None,
            "stdout": _text(exc.stdout),
            "stderr": _text(exc.stderr),
            "timed_out": True,
        }


def _combined_output_tail(run_result: dict[str, Any]) -> str:
    parts = []
    if run_result.get("timed_out"):
        parts.append("Timed out.")
    stdout = run_result.get("stdout") or ""
    stderr = run_result.get("stderr") or ""
    if stdout:
        parts.append("[stdout]\n" + stdout)
    if stderr:
        parts.append("[stderr]\n" + stderr)
    return _tail("\n".join(parts), 2200)


def _subprocess_text(result: subprocess.CompletedProcess[str]) -> str:
    parts = [f"returncode: {result.returncode}"]
    if result.stdout:
        parts.append("[stdout]\n" + result.stdout)
    if result.stderr:
        parts.append("[stderr]\n" + result.stderr)
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Perf JSON helpers
# --------------------------------------------------------------------------- #


def _perf_json_candidates(perf_results_dir: Path, file_name: str) -> list[Path]:
    stem = Path(file_name).stem
    return [
        perf_results_dir / f"{stem}.json",
        perf_results_dir / f"{stem}_perf.json",
    ]


def _perf_file_name_from_json(gen_path: Path) -> str:
    stem = gen_path.stem
    if stem.endswith("_perf"):
        stem = stem[: -len("_perf")]
    return f"{stem}.py"


def _matching_golden_path(gen_path: Path) -> Path | None:
    ref_dir = Path(REPO_DIR) / "performance_metrics/perf_T/golden_results"
    candidates = [ref_dir / gen_path.name]
    if gen_path.stem.endswith("_perf"):
        candidates.append(ref_dir / f"{gen_path.stem[:-len('_perf')]}.json")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _extract_perf_rows(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        for key in ("results", "benchmarks", "data"):
            value = data.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    return []


def _perf_ms(row: dict[str, Any]) -> float | None:
    for key in ("ms", "median_ms", "mean_ms", "avg_ms", "latency_ms"):
        value = row.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _input_key(row: dict[str, Any]) -> str | None:
    for key in ("input_size", "size", "shape", "input_shape"):
        if key in row:
            return json.dumps(row[key], sort_keys=True, default=str)
    return None


def _align_perf_rows(
    gen_rows: list[dict[str, Any]], ref_rows: list[dict[str, Any]]
) -> tuple[list[tuple[dict[str, Any], dict[str, Any]]], str]:
    if len(gen_rows) == len(ref_rows):
        return list(zip(gen_rows, ref_rows)), "same_length"

    ref_by_key = {
        key: row
        for row in ref_rows
        if (key := _input_key(row)) is not None
    }
    pairs = []
    for gen_row in gen_rows:
        key = _input_key(gen_row)
        if key is not None and key in ref_by_key:
            pairs.append((gen_row, ref_by_key[key]))
    return pairs, "input_key"


# --------------------------------------------------------------------------- #
# Perf result analysis
# --------------------------------------------------------------------------- #


def _analyze_perf_results(perf_results_dir: Path) -> dict[str, dict[str, Any]]:
    analysis: dict[str, dict[str, Any]] = {}

    for gen_path in sorted(perf_results_dir.rglob("*.json")):
        # Intermediate per-batch result dirs are siblings of perf_results_dir, not children,
        # but skip any stray batch dir that lands inside to avoid double-counting.
        if "perf_batches" in gen_path.parts:
            continue
        file_name = _perf_file_name_from_json(gen_path)
        record: dict[str, Any] = {
            "file": file_name,
            "result_json": str(gen_path.relative_to(perf_results_dir)),
            "golden_json": None,
            "status": "unknown",
            "message": "",
            "speedup_vs_pytorch": None,
            "generated_ms_sum": None,
            "reference_ms_sum": None,
            "num_cases": 0,
        }
        analysis[file_name] = record

        ref_path = _matching_golden_path(gen_path)
        if ref_path is None:
            record["status"] = "missing_golden"
            record["message"] = "No matching golden result JSON was found."
            continue
        record["golden_json"] = ref_path.name

        try:
            gen_data = json.loads(gen_path.read_text(encoding="utf-8"))
            ref_data = json.loads(ref_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            record["status"] = "parse_error"
            record["message"] = f"Could not parse perf JSON: {exc}"
            continue

        gen_rows = _extract_perf_rows(gen_data)
        ref_rows = _extract_perf_rows(ref_data)
        if not gen_rows:
            record["status"] = "empty_result"
            record["message"] = (
                "Generated perf JSON is empty. The perf script completed but did "
                "not record timings."
            )
            continue
        if not ref_rows:
            record["status"] = "empty_golden"
            record["message"] = "Golden perf JSON is empty."
            continue

        pairs, alignment = _align_perf_rows(gen_rows, ref_rows)
        if not pairs:
            record["status"] = "shape_mismatch"
            record["message"] = (
                f"Could not align generated rows ({len(gen_rows)}) with golden "
                f"rows ({len(ref_rows)})."
            )
            continue

        gen_ms_values: list[float] = []
        ref_ms_values: list[float] = []
        case_details: list[dict[str, Any]] = []
        for case_idx, (gen_row, ref_row) in enumerate(pairs):
            gen_ms = _perf_ms(gen_row)
            ref_ms = _perf_ms(ref_row)
            if gen_ms is not None and ref_ms is not None:
                gen_ms_values.append(gen_ms)
                ref_ms_values.append(ref_ms)
                case_details.append(
                    {
                        "case_index": case_idx,
                        "input": _input_key(gen_row) or _input_key(ref_row),
                        "generated_ms": round(gen_ms, 6),
                        "reference_ms": round(ref_ms, 6),
                        "speedup_vs_pytorch": (
                            round(ref_ms / gen_ms, 4) if gen_ms > 0 else None
                        ),
                    }
                )

        if not gen_ms_values or len(gen_ms_values) != len(pairs):
            record["status"] = "missing_ms"
            record["message"] = "One or more aligned perf rows had no ms timing field."
            record["num_cases"] = len(pairs)
            continue

        gen_sum = sum(gen_ms_values)
        ref_sum = sum(ref_ms_values)
        record["generated_ms_sum"] = round(gen_sum, 6)
        record["reference_ms_sum"] = round(ref_sum, 6)
        record["num_cases"] = len(pairs)
        record["alignment"] = alignment
        record["worst_cases"] = sorted(
            case_details,
            key=lambda item: (
                float("inf")
                if item.get("speedup_vs_pytorch") is None
                else item["speedup_vs_pytorch"]
            ),
        )[:5]

        if gen_sum <= 0:
            record["status"] = "invalid_generated_time"
            record["message"] = "Generated total runtime was <= 0."
            continue

        speedup = round(ref_sum / gen_sum, 4)
        record["speedup_vs_pytorch"] = speedup
        # TritonBench's 2_efficiency.py only counts speedups in (0.1, 10); values outside
        # that range are treated as measurement artifacts and excluded from the aggregate.
        if 0.1 < speedup < 10:
            record["status"] = "accepted"
            record["message"] = "Speedup accepted by TritonBench range filter."
        else:
            record["status"] = "out_of_range"
            record["message"] = (
                "Speedup was computed, but TritonBench excludes values outside "
                "0.1 < speedup < 10 from its aggregate."
            )

    return analysis


def _perf_failure_details(
    perf_results_dir: Path,
    file_name: str,
    perf_analysis: dict[str, dict[str, Any]] | None = None,
) -> str:
    if perf_analysis and file_name in perf_analysis:
        record = perf_analysis[file_name]
        return (
            f"Phase 3 status for {file_name}: {record.get('status')}. "
            f"{record.get('message')} "
            f"speedup={record.get('speedup_vs_pytorch')}, "
            f"cases={record.get('num_cases')}, "
            f"result_json={record.get('result_json')}"
        )

    result_path = next(
        (path for path in _perf_json_candidates(perf_results_dir, file_name) if path.exists()),
        perf_results_dir / file_name.replace(".py", ".json"),
    )
    if not result_path.exists():
        return f"{result_path.name} was not produced by the performance runner."
    try:
        data = json.loads(result_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return f"{result_path.name} could not be parsed: {exc}"
    if not data:
        return (
            f"{result_path.name} is empty. The generated kernel passed the "
            "correctness harness but failed during the larger performance "
            "benchmark, often because a Triton kernel read or wrote out of bounds."
        )
    return (
        f"{result_path.name} exists but did not yield a valid speedup. "
        f"Raw result tail: {_tail(json.dumps(data), 1200)}"
    )


def _speedups_by_file(perf_results_dir: Path) -> dict[str, float]:
    return {
        file_name: record["speedup_vs_pytorch"]
        for file_name, record in _analyze_perf_results(perf_results_dir).items()
        if record.get("status") == "accepted"
        and isinstance(record.get("speedup_vs_pytorch"), (int, float))
    }


# --------------------------------------------------------------------------- #
# Batch runner
# --------------------------------------------------------------------------- #


def _chunks(values: list[str], size: int) -> list[list[str]]:
    if size <= 0 or size >= len(values):
        return [values]
    return [values[idx : idx + size] for idx in range(0, len(values), size)]


def _merge_perf_outputs(source_dir: Path, dest_dir: Path) -> None:
    for path in source_dir.rglob("*.json"):
        relative = path.relative_to(source_dir)
        target = dest_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def _run_perf_batch(
    perf_root: Path,
    source_call_acc_dir: Path,
    batch_results_dir: Path,
    file_names: list[str],
) -> str:
    batch_input_dir = batch_results_dir.parent / "input"
    if batch_input_dir.exists():
        shutil.rmtree(batch_input_dir)
    batch_input_dir.mkdir(parents=True, exist_ok=True)
    batch_results_dir.mkdir(parents=True, exist_ok=True)

    for file_name in file_names:
        src = source_call_acc_dir / file_name
        dst = batch_input_dir / file_name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    write_file_run = subprocess.run(
        [
            sys.executable,
            "run_bench/write_file.py",
            "--input_folder_path",
            str(batch_input_dir),
            "--results_path",
            str(batch_results_dir),
        ],
        cwd=perf_root,
        capture_output=True,
        text=True,
        check=False,
    )
    runner_run = subprocess.run(
        [sys.executable, "run_bench/multiprocess_gpu_run.py"],
        cwd=perf_root,
        capture_output=True,
        text=True,
        check=False,
    )

    return "\n\n".join(
        [
            f"[perf batch: {', '.join(file_names)}]",
            "[write_file.py]",
            _subprocess_text(write_file_run),
            "[multiprocess_gpu_run.py]",
            _subprocess_text(runner_run),
        ]
    )
