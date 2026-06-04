"""
TritonBench dataset loading, file mapping, and Modal Volume helpers.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from config import DATA_DIR, REPO_DIR
from code_utils import _extract_code


def _reload_volume() -> None:
    from modal_app import data_volume
    try:
        data_volume.reload()
    except Exception:  # noqa: BLE001
        pass


def _read_json_or_jsonl(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]


def _load_alpaca(dataset: str) -> list[dict[str, Any]]:
    if dataset not in {"simp", "comp"}:
        raise ValueError("dataset must be 'simp' or 'comp'")
    path = Path(REPO_DIR) / f"data/TritonBench_T_{dataset}_alpac_v1.json"
    data = _read_json_or_jsonl(path)
    if not isinstance(data, list):
        raise ValueError(f"unexpected dataset format in {path}")
    return data


@lru_cache(maxsize=1)
def _tritonbench_metadata() -> list[dict[str, Any]]:
    stats_path = Path(REPO_DIR) / "data/TritonBench_T_v1.jsonl"
    stats = _read_json_or_jsonl(stats_path)
    if not isinstance(stats, list):
        raise ValueError(f"unexpected stats format in {stats_path}")
    return stats


@lru_cache(maxsize=1)
def _metadata_by_file() -> dict[str, dict[str, Any]]:
    return {item["file"]: item for item in _tritonbench_metadata()}


def _reference_context_for_file(file_name: str, char_limit: int) -> str:
    # Each reference file in TritonBench_T_v1 embeds the reference PyTorch implementation
    # above the 146-`#` delimiter and the test harness below it; we only want the former.
    delimiter = "#" * 146
    module_path = Path(REPO_DIR) / "data/TritonBench_T_v1" / file_name
    parts: list[str] = []

    if module_path.exists():
        reference_source = module_path.read_text(encoding="utf-8").split(delimiter, 1)[0]
        if reference_source.strip():
            parts.append(reference_source.strip())

    metadata = _metadata_by_file().get(file_name, {})
    torch_code = metadata.get("torch_code")
    if torch_code:
        parts.append("# Reference torch operation sequence:\n" + str(torch_code).strip())
    other = metadata.get("other")
    if other:
        parts.append("# Additional semantic notes:\n" + str(other).strip())

    context = "\n\n".join(parts).strip()
    if char_limit > 0 and len(context) > char_limit:
        context = context[:char_limit] + "\n# ... reference context truncated ..."
    return context


def _description_from_instruction(instruction: str) -> str:
    if "Functional Description: " not in instruction:
        return ""
    return (
        instruction.split("Functional Description: ", 1)[1]
        .split("Wrapper Entry Information:", 1)[0]
        .replace("\n", "")
    )


def _files_for_instructions(instructions: list[str]) -> list[str]:
    stats = _tritonbench_metadata()

    files: list[str] = []
    for instruction in instructions:
        description = _description_from_instruction(instruction)
        matches = [
            item["file"]
            for item in stats
            if description and description in item["description"].replace("\n", "")
        ]
        if len(matches) != 1:
            raise ValueError(
                f"could not map instruction to exactly one TritonBench file; "
                f"matches={matches}"
            )
        files.append(matches[0])
    return files


def _safe_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _load_predictions(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _prediction_code_by_file(predictions_path: str) -> dict[str, str]:
    full_path = Path(DATA_DIR) / predictions_path
    if not full_path.exists():
        return {}
    records = _load_predictions(full_path)
    instructions = [record["instruction"] for record in records]
    files = _files_for_instructions(instructions)
    return {
        file_name: _extract_code(record.get("predict", ""))
        for file_name, record in zip(files, records)
    }
