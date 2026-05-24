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
"""

from __future__ import annotations

import ast
import json
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import modal

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

APP_NAME = "tritonbench-t-nvidia-iterative"
TRITONBENCH_REPO = "https://github.com/thunlp/TritonBench.git"

DEFAULT_GPU = os.environ.get("TRITONBENCH_GPU", "T4")
DEFAULT_MODEL = os.environ.get(
    # "NVIDIA_MODEL", "mistralai/mistral-large-3-675b-instruct-2512"
    "NVIDIA_MODEL", "qwen/qwen3-coder-480b-a35b-instruct"
)
DEFAULT_ITERATIONS = 3
DEFAULT_CONCURRENCY = 4
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TEMPERATURE = 0.15
DEFAULT_REQUEST_TIMEOUT_SECONDS = 600
DEFAULT_RETRIES = 3
DEFAULT_CHECKPOINT_EVERY = 1
DEFAULT_INCLUDE_REFERENCE_SOURCE = False
DEFAULT_REFERENCE_SOURCE_CHAR_LIMIT = 6000
DEFAULT_USE_BEST_SO_FAR = True
DEFAULT_REFINE_PASSING = False
DEFAULT_REPAIR_PERF_FAILURES = False
DEFAULT_MAX_FEEDBACK_HISTORY = 2
DEFAULT_FORCE_REGENERATE = False
DEFAULT_LOOP_MODE = "auto"
DEFAULT_TARGET_SPEEDUP = 1.0
DEFAULT_AUTO_OPTIMIZE_MIN_EXEC_RATE = 1.0
DEFAULT_PERF_BATCH_SIZE = 8
DEFAULT_SKIP_EFFICIENCY = False

VOLUME_NAME = "tritonbench-t-data"
DATA_DIR = "/data"
REPO_DIR = "/opt/TritonBench"
RUNS_DIR = "nvidia_iterative_runs"

NVIDIA_INVOKE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
LOCAL_DOTENV_PATH = Path(__file__).with_name(".env")
FALLBACK_SECRET_NAME = os.environ.get("TRITONBENCH_LLM_SECRET", "tritonbench-llm")

# --------------------------------------------------------------------------- #
# Modal image. Same TritonBench patches as the baseline, plus requests for the
# NVIDIA endpoint.
# --------------------------------------------------------------------------- #

PATCH_CALL_ACC = (
    f"""sed -i """
    f"""-e 's|^statis_path = .*|statis_path = "{REPO_DIR}/data/TritonBench_T_v1.jsonl"|' """
    f"""-e 's|^py_folder = .*|py_folder = "{REPO_DIR}/data/TritonBench_T_v1/"|' """
    f"""-e 's|^py_interpreter = .*|import sys; py_interpreter = sys.executable|' """
    f"""{REPO_DIR}/EVAL/eval_T/0_call_acc.py"""
)

PATCH_EXE_ACC = (
    f"""sed -i """
    f"""-e 's|^gold_folder = .*|gold_folder = "{REPO_DIR}/data/TritonBench_T_v1/"|' """
    f"""-e 's|^py_interpreter = .*|import sys; py_interpreter = sys.executable|' """
    f"""{REPO_DIR}/EVAL/eval_T/1_exe_acc.py"""
)

PATCH_PERF = (
    f"""sed -i 's|^gpu_count = .*|gpu_count = 1|' """
    f"""{REPO_DIR}/performance_metrics/perf_T/run_bench/multiprocess_gpu_run.py"""
)

cpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install("requests>=2.32")
    .run_commands(f"git clone --depth 1 {TRITONBENCH_REPO} {REPO_DIR}")
)

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.12"
    )
    .apt_install("git", "build-essential")
    .pip_install(
        "torch==2.5.1",
        "triton==3.1.0",
        "tqdm==4.66.5",
        "numpy<2",
        "requests>=2.32",
    )
    .run_commands(f"git clone --depth 1 {TRITONBENCH_REPO} {REPO_DIR}")
    .run_commands(PATCH_CALL_ACC, PATCH_EXE_ACC, PATCH_PERF)
    .run_commands(
        f"ln -s {REPO_DIR}/EVAL/eval_T/0_call_acc.py {REPO_DIR}/EVAL/eval_T/call_acc.py",
        f"ln -s {REPO_DIR}/EVAL/eval_T/1_exe_acc.py {REPO_DIR}/EVAL/eval_T/exe_acc.py",
    )
)

app = modal.App(APP_NAME, image=image)
data_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


# --------------------------------------------------------------------------- #
# Secret loading
# --------------------------------------------------------------------------- #


def _parse_dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if (
            len(value) >= 2
            and value[0] == value[-1]
            and value[0] in {"'", '"'}
        ):
            value = value[1:-1]
        values[key] = value
    return values


def _nvidia_secret() -> modal.Secret:
    env_values = _parse_dotenv(LOCAL_DOTENV_PATH)
    secret_values: dict[str, str] = {}

    for key in ("NVIDIA_KEY", "NVIDIA_API_KEY"):
        value = os.environ.get(key) or env_values.get(key)
        if value:
            secret_values[key] = value

    if secret_values:
        return modal.Secret.from_dict(secret_values)
    return modal.Secret.from_name(FALLBACK_SECRET_NAME)


NVIDIA_SECRET = _nvidia_secret()


# --------------------------------------------------------------------------- #
# Prompting and NVIDIA generation
# --------------------------------------------------------------------------- #

PROMPT_HEADER = (
    "You are an expert Triton programmer. Write fast and correct Triton kernels "
    "and Python wrapper functions from functional descriptions and wrapper "
    "signatures. The wrapper function must exactly match the requested "
    "signature and behavior.\n\n"
    "Return one self-contained Python module in a single ```python ... ``` code "
    "block. Include necessary imports for torch, triton, and "
    "triton.language as tl. Do not include test code, examples, prose outside "
    "the code block, fill-in-middle tokens, file I/O, network calls, or "
    "benchmark harness code. "
    "Correctness is mandatory; optimize only after preserving behavior. Every "
    "Triton load and store must be guarded with valid masks so larger benchmark "
    "inputs cannot read or write out of bounds."
)


def _extract_code(text: str) -> str:
    s = (text or "").strip()
    match = re.search(r"```(?:python|py)?\s*\n(.*?)\n```", s, re.DOTALL)
    if match:
        return _sanitize_generated_code(match.group(1))
    s = re.sub(r"^```(?:python|py)?\s*\n?", "", s)
    s = re.sub(r"\n?```\s*$", "", s)
    return _sanitize_generated_code(s)


def _sanitize_generated_code(code: str) -> str:
    """Remove common model artifacts without changing intended logic."""
    s = code or ""
    s = re.sub(r"<\|(?:fim_prefix|fim_middle|fim_suffix|fim_pad)\|>", "", s)
    s = re.sub(r"<\|(?:repo_name|file_sep|endoftext)\|>", "", s)
    s = re.sub(r"\btritorion\b", "triton", s)
    s = re.sub(r"\btritonion\b", "triton", s)
    s = re.sub(r"^python\s*\n", "", s.strip())
    return s.strip() + "\n"


def _tail(text: str | None, limit: int = 1800) -> str:
    if not text:
        return ""
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[-limit:]


def _feedback_one_line(feedback: dict[str, Any]) -> str:
    static_errors = feedback.get("static_validation_errors")
    if isinstance(static_errors, list) and static_errors:
        return _tail("; ".join(str(item) for item in static_errors), 500)
    for key in ("call_error_tail", "exec_error_tail", "perf_error_tail"):
        value = (feedback.get(key) or "").strip()
        if value:
            compact = " ".join(value.split())
            return _tail(compact, 500)
    return "No stderr/stdout detail was captured for this attempt."


def _speedup_value(feedback: dict[str, Any] | None) -> float | None:
    if not feedback:
        return None
    value = feedback.get("speedup_vs_pytorch")
    return float(value) if isinstance(value, (int, float)) else None


def _phase2_passed(feedback: dict[str, Any] | None) -> bool:
    return bool(feedback and feedback.get("phase2_exec_passed"))


def _gpu_issue_feedback(feedback: dict[str, Any] | None) -> bool:
    if not feedback:
        return False

    statuses = [str(feedback.get("phase3_perf_status") or "")]
    analysis = feedback.get("phase3_perf_analysis")
    if isinstance(analysis, dict):
        statuses.append(str(analysis.get("status") or ""))
    normalized_statuses = {
        status.removeprefix("reused_") for status in statuses if status
    }
    if "skipped_after_crash" in normalized_statuses:
        return True

    text = "\n".join(
        str(feedback.get(key) or "")
        for key in ("call_error_tail", "exec_error_tail", "perf_error_tail")
    )
    return any(
        marker in text
        for marker in (
            "Xid",
            "MMU Fault",
            "Runner killed",
            "SIGKILL",
            "killed a previous evaluator container",
            "out of bounds",
        )
    )


def _wrapper_name_from_instruction(instruction: str) -> str:
    match = re.search(
        r"Wrapper Entry Information:\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(",
        instruction or "",
    )
    return match.group(1) if match else ""


def _is_triton_jit_decorator(node: ast.expr) -> bool:
    if isinstance(node, ast.Attribute) and node.attr == "jit":
        return isinstance(node.value, ast.Name) and node.value.id == "triton"
    if isinstance(node, ast.Call):
        return _is_triton_jit_decorator(node.func)
    return False


def _triton_jit_boolop_errors(tree: ast.AST) -> list[str]:
    errors: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not any(_is_triton_jit_decorator(dec) for dec in node.decorator_list):
            continue
        for child in ast.walk(node):
            if isinstance(child, ast.BoolOp):
                errors.append(
                    f"Triton JIT function {node.name} contains Python boolean "
                    f"operator at line {getattr(child, 'lineno', '?')}; use "
                    "tensor mask expressions with &, |, and parentheses"
                )
                break
    return errors


def _static_validate_code(code: str, instruction: str = "") -> list[str]:
    errors: list[str] = []
    if not code.strip():
        return ["generated code is empty"]

    if re.search(r"<\|[^|]+?\|>", code):
        errors.append("model special tokens remain in the generated code")
    if "tritorion" in code or "tritonion" in code:
        errors.append("probable typo in triton module name")

    tree: ast.AST | None = None
    try:
        tree = ast.parse(code)
        compile(tree, "<generated_triton_module>", "exec")
    except SyntaxError as exc:
        errors.append(f"Python SyntaxError line {exc.lineno}: {exc.msg}")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"Python compile failed: {exc}")

    wrapper_name = _wrapper_name_from_instruction(instruction)
    if wrapper_name and not re.search(
        rf"^def\s+{re.escape(wrapper_name)}\s*\(", code, re.MULTILINE
    ):
        errors.append(f"missing required wrapper function def {wrapper_name}(...)")

    if tree is not None:
        errors.extend(_triton_jit_boolop_errors(tree))

    return errors


def _failure_kind(feedback: dict[str, Any] | None) -> str:
    if not feedback:
        return "initial"
    if _gpu_issue_feedback(feedback):
        return "gpu_fault"
    if feedback.get("static_validation_errors"):
        return "static"
    text = "\n".join(
        str(feedback.get(key) or "")
        for key in ("call_error_tail", "exec_error_tail", "perf_error_tail")
    )
    if "SyntaxError" in text or "invalid syntax" in text:
        return "syntax"
    if "UnsupportedLanguageConstruct" in text:
        return "triton_language"
    if "NameError" in text or "ModuleNotFoundError" in text or "ImportError" in text:
        return "name_import"
    if "missing required wrapper" in text:
        return "wrapper"
    if feedback.get("phase1_call_passed") and not feedback.get("phase2_exec_passed"):
        return "semantics"
    if feedback.get("phase2_exec_passed") and not feedback.get("phase3_perf_passed"):
        return "performance"
    return "runtime"


def _repair_focus_lines(feedback: dict[str, Any] | None, task_mode: str) -> list[str]:
    if task_mode == "optimize":
        return [
            "Repair focus: optimize an already execution-correct version.",
            "- Do not change the public wrapper signature or output semantics.",
            "- Avoid torch fallbacks or extra allocations in the hot path.",
            "- Prefer one efficient Triton launch for simple elementwise/fused work.",
        ]

    kind = _failure_kind(feedback)
    if kind in {"static", "syntax"}:
        return [
            "Repair focus: fix code validity first.",
            "- Return syntactically valid Python only.",
            "- Remove any fill-in-middle markers or prose.",
            "- Keep imports simple: torch, triton, and triton.language as tl.",
            "- Do not redesign the algorithm unless needed for syntax validity.",
        ]
    if kind == "triton_language":
        return [
            "Repair focus: use Triton-supported language constructs.",
            "- Avoid Python if/or/and chains inside @triton.jit kernels.",
            "- Build boolean masks with tensor expressions using &, |, and parentheses.",
            "- Keep constexpr values and runtime tensors clearly separated.",
        ]
    if kind == "wrapper":
        return [
            "Repair focus: implement the exact requested wrapper function.",
            "- The module must define the wrapper named in Wrapper Entry Information.",
            "- Preserve optional arguments such as out=, rounding_mode, dim, and keepdim.",
        ]
    if kind == "name_import":
        return [
            "Repair focus: fix names/imports.",
            "- Use triton, not misspelled module names.",
            "- Define every kernel/helper before the wrapper uses it.",
            "- Do not depend on unavailable packages.",
        ]
    if kind == "semantics":
        return [
            "Repair focus: match PyTorch semantics exactly.",
            "- The previous version ran but produced different outputs.",
            "- Re-check broadcasting, dtype promotion, shape handling, and optional arguments.",
            "- Prefer a simple correct implementation over an aggressive optimization.",
        ]
    if kind == "gpu_fault":
        return [
            "Repair focus: the previous version passed small correctness tests but failed the larger GPU performance run.",
            "- Assume an out-of-bounds load/store, unsafe pointer arithmetic, or shape case missed by the correctness harness.",
            "- Every tl.load and tl.store must use a valid mask before touching pointer offsets.",
            "- Use conservative grids and masks for all benchmark-sized tensors, including edge blocks.",
        ]
    return [
        "Repair focus: fix runtime behavior.",
        "- Re-check tensor shapes, strides, device placement, masks, and wrapper arguments.",
        "- Make the simplest execution-correct Triton implementation first.",
    ]


def _perf_feedback_lines(feedback: dict[str, Any]) -> list[str]:
    analysis = feedback.get("phase3_perf_analysis")
    if not isinstance(analysis, dict):
        return []

    lines = [
        "Performance analysis for the selected prior version:",
        f"- status: {analysis.get('status')}",
        f"- generated_ms_sum: {analysis.get('generated_ms_sum')}",
        f"- reference_ms_sum: {analysis.get('reference_ms_sum')}",
        f"- speedup_vs_pytorch: {analysis.get('speedup_vs_pytorch')}",
        f"- benchmark cases: {analysis.get('num_cases')}",
    ]
    worst_cases = analysis.get("worst_cases")
    if isinstance(worst_cases, list) and worst_cases:
        lines.append("- slowest/worst relative cases:")
        for case in worst_cases[:5]:
            lines.append(
                "  "
                + json.dumps(
                    {
                        "input": case.get("input"),
                        "generated_ms": case.get("generated_ms"),
                        "reference_ms": case.get("reference_ms"),
                        "speedup": case.get("speedup_vs_pytorch"),
                    },
                    sort_keys=True,
                )
            )
    return lines


def _nvidia_chat(
    messages: list[dict[str, str]],
    model: str,
    max_tokens: int,
    temperature: float,
    request_timeout_seconds: int,
    retries: int,
) -> str:
    import requests

    api_key = os.environ.get("NVIDIA_KEY") or os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise RuntimeError("NVIDIA_KEY or NVIDIA_API_KEY is not available in Modal")

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 1.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "stream": True,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "text/event-stream",
        "Content-Type": "application/json",
    }

    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = requests.post(
                NVIDIA_INVOKE_URL,
                headers=headers,
                json=payload,
                stream=True,
                timeout=(30, request_timeout_seconds),
            )
            if response.status_code in {408, 409, 429, 500, 502, 503, 504}:
                raise RuntimeError(
                    f"NVIDIA API retryable status {response.status_code}: "
                    f"{_tail(response.text, 800)}"
                )
            response.raise_for_status()

            chunks: list[str] = []
            for raw_line in response.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue
                line = raw_line.strip()
                if line.startswith("data:"):
                    line = line[len("data:") :].strip()
                if line == "[DONE]":
                    break
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue

                choice = (event.get("choices") or [{}])[0]
                delta = choice.get("delta") or {}
                message = choice.get("message") or {}
                content = delta.get("content") or message.get("content") or ""
                if content:
                    chunks.append(content)

            text = "".join(chunks).strip()
            if not text:
                raise RuntimeError("NVIDIA API returned an empty streamed response")
            return text
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt == retries - 1:
                break
            time.sleep(min(30, 2**attempt))

    raise RuntimeError(f"NVIDIA generation failed: {last_error}")


def _build_messages(
    item: dict[str, Any],
    file_name: str,
    previous_code: str | None,
    previous_feedback: dict[str, Any] | None,
    previous_feedback_history: list[dict[str, Any]] | None,
    iteration: int,
    reference_context: str = "",
    task_mode: str = "correctness",
    target_speedup: float = DEFAULT_TARGET_SPEEDUP,
) -> list[dict[str, str]]:
    user_parts = [item["instruction"]]
    item_input = item.get("input", "") or ""
    if item_input:
        user_parts.append(item_input)

    if reference_context:
        user_parts.append(
            "\n".join(
                [
                    "Reference PyTorch implementation/context for exact semantics.",
                    "Use this to match behavior, edge cases, argument handling, and out= semantics.",
                    "The replacement you return should still be a Triton-oriented implementation.",
                    "```python",
                    reference_context.strip(),
                    "```",
                ]
            )
        )

    if previous_feedback:
        speedup = previous_feedback.get("speedup_vs_pytorch")
        speedup_text = "not available" if speedup is None else f"{speedup:.4g}"
        feedback_lines = [
            "",
            "Benchmark feedback for the selected prior version:",
            f"- TritonBench file: {file_name}",
            f"- Prior version iteration: {previous_feedback.get('iteration', iteration - 1)}",
            f"- Phase 1 call accuracy passed: {previous_feedback.get('phase1_call_passed')}",
            f"- Phase 2 execution accuracy passed: {previous_feedback.get('phase2_exec_passed')}",
            f"- Phase 3 performance benchmark passed: {previous_feedback.get('phase3_perf_passed')}",
            f"- Speedup vs PyTorch, higher is better: {speedup_text}",
        ]
        call_error = previous_feedback.get("call_error_tail")
        exec_error = previous_feedback.get("exec_error_tail")
        perf_error = previous_feedback.get("perf_error_tail")
        static_errors = previous_feedback.get("static_validation_errors")
        if isinstance(static_errors, list) and static_errors:
            feedback_lines.extend(
                ["- Static validation errors:"]
                + [f"  - {error}" for error in static_errors[:5]]
            )
        if call_error:
            feedback_lines.extend(["- Phase 1 stderr/stdout tail:", call_error])
        if exec_error:
            feedback_lines.extend(["- Phase 2 mismatch/error tail:", exec_error])
        if perf_error:
            feedback_lines.extend(["- Phase 3 performance failure tail:", perf_error])
        perf_lines = _perf_feedback_lines(previous_feedback)
        if perf_lines:
            feedback_lines.extend([""] + perf_lines)
        feedback_lines.extend([""] + _repair_focus_lines(previous_feedback, task_mode))
        if previous_code:
            feedback_lines.extend(
                [
                    "",
                    "Previous generated module:",
                    "```python",
                    previous_code.strip(),
                    "```",
                ]
            )
        feedback_lines.extend(
            [
                "",
                (
                    "Produce an optimized replacement module. The prior version "
                    "passed execution correctness but is slower than the target "
                    f"speedup of {target_speedup:.3g}x. Preserve the wrapper "
                    "signature and exact output semantics. Focus on reducing "
                    "kernel launches, avoiding torch fallbacks in the hot path, "
                    "using vectorized/block Triton work, choosing reasonable "
                    "BLOCK_SIZE/num_warps, and guarding all memory operations "
                    "with masks. If an aggressive optimization risks "
                    "correctness, keep the correct behavior."
                    if task_mode == "optimize"
                    else "Produce an improved replacement module. Fix "
                    "correctness before performance. If the previous version "
                    "was correct, keep its public wrapper behavior and improve "
                    "efficiency conservatively."
                ),
            ]
        )
        user_parts.append("\n".join(feedback_lines))

    if previous_feedback_history:
        history_lines = [
            "Prior attempt history for this same TritonBench file. Avoid "
            "repeating these failure modes."
        ]
        for feedback in previous_feedback_history:
            speedup = feedback.get("speedup_vs_pytorch")
            speedup_text = "n/a" if speedup is None else f"{speedup:.4g}"
            history_lines.extend(
                [
                    (
                        f"- Iteration {feedback.get('iteration')}: "
                        f"phase1={feedback.get('phase1_call_passed')}, "
                        f"phase2={feedback.get('phase2_exec_passed')}, "
                        f"phase3={feedback.get('phase3_perf_passed')}, "
                        f"speedup={speedup_text}"
                    ),
                    f"  detail: {_feedback_one_line(feedback)}",
                ]
            )
        user_parts.append("\n".join(history_lines))

    return [
        {"role": "system", "content": PROMPT_HEADER},
        {"role": "user", "content": "\n\n".join(user_parts)},
    ]


# --------------------------------------------------------------------------- #
# TritonBench data helpers
# --------------------------------------------------------------------------- #


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


def _reload_volume() -> None:
    try:
        data_volume.reload()
    except Exception:  # noqa: BLE001
        pass


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


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #


@app.function(
    image=cpu_image,
    timeout=60 * 60 * 4,
    cpu=4,
    volumes={DATA_DIR: data_volume},
    secrets=[NVIDIA_SECRET],
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
        )
        (prompts_dir / f"{idx:04d}_{file_name}.json").write_text(
            json.dumps(messages, indent=2), encoding="utf-8"
        )
        try:
            raw = _nvidia_chat(
                messages,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                request_timeout_seconds=request_timeout_seconds,
                retries=retries,
            )
            code = _extract_code(raw)
            error = ""
        except Exception as exc:  # noqa: BLE001
            code = f"# generation failed: {exc}\n"
            error = str(exc)

        script_path = generated_dir / file_name
        script_path.parent.mkdir(parents=True, exist_ok=True)
        script_path.write_text(code, encoding="utf-8")

        # Keep "instruction" first; TritonBench's 0_call_acc.py uses the first
        # key in each JSON object to map a row back to its reference file.
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


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #


def _ensure_eval_imports() -> None:
    eval_dir = f"{REPO_DIR}/EVAL/eval_T"
    if eval_dir not in sys.path:
        sys.path.insert(0, eval_dir)
    os.environ["PYTHONPATH"] = eval_dir + os.pathsep + os.environ.get("PYTHONPATH", "")


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


def _analyze_perf_results(perf_results_dir: Path) -> dict[str, dict[str, Any]]:
    analysis: dict[str, dict[str, Any]] = {}

    for gen_path in sorted(perf_results_dir.rglob("*.json")):
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
    cached_correctness_files = {
        file_name
        for file_name, prior_feedback in carried_forward_feedback_by_file.items()
        if prior_feedback.get("phase2_exec_passed")
    }
    total = len(files)
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
        delimiter = "#" * 146
        for idx, (script_content, test_content, file_name) in enumerate(
            zip(predictions, tests, files), start=1
        ):
            if file_name in cached_correctness_files:
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
                feedback[file_name]["phase2_exec_passed"] = True
                if idx % 10 == 0 or idx == len(call_survivors):
                    print(f"  phase 2 checked {idx}/{len(call_survivors)}", flush=True)
                continue

            generated_path = call_acc_dir / file_name
            gold_path = gold_folder / file_name
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

    reused_perf_analysis: dict[str, dict[str, Any]] = {}
    reused_perf_files: set[str] = set()
    for file_name in exec_survivors:
        record = prediction_record_by_file.get(file_name, {})
        prior_feedback = record.get("carried_forward_benchmark_result")
        if not isinstance(prior_feedback, dict):
            continue
        prior_analysis = prior_feedback.get("phase3_perf_analysis")
        if isinstance(prior_analysis, dict):
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
                _merge_perf_outputs(batch_results_dir, perf_results_dir)
                runner_detail_parts.append(
                    f"[perf batch: {', '.join(file_names)}]\nresumed completed batch"
                )
                continue
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

            batch_state[batch_key] = {
                "status": "started",
                "files": file_names,
            }
            _safe_write_json(batch_state_path, batch_state)
            data_volume.commit()
            runner_detail_parts.append(
                _run_perf_batch(
                    perf_root=perf_root,
                    source_call_acc_dir=call_acc_dir,
                    batch_results_dir=batch_results_dir,
                    file_names=file_names,
                )
            )
            _merge_perf_outputs(batch_results_dir, perf_results_dir)
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


# --------------------------------------------------------------------------- #
# Best-version materialization
# --------------------------------------------------------------------------- #


def _score_feedback(
    feedback: dict[str, Any] | None,
    iteration: int,
) -> tuple[int, float, int, int, int]:
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


# --------------------------------------------------------------------------- #
# Local console formatting. Full JSON artifacts are still saved in the volume;
# these helpers keep the detached-run logs readable.
# --------------------------------------------------------------------------- #


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


# --------------------------------------------------------------------------- #
# Local entrypoint
# --------------------------------------------------------------------------- #


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
):
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
