"""
Prompt construction, LLM API calls, code extraction, and static validation.
"""

from __future__ import annotations

import ast
import json
import os
import re
import time
from typing import Any

from config import (
    DEFAULT_MAX_FEEDBACK_HISTORY,
    DEFAULT_MAX_TOKENS,
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
    DEFAULT_RETRIES,
    DEFAULT_TARGET_SPEEDUP,
    DEFAULT_TEMPERATURE,
    LLM_INVOKE_URL,
)

# --------------------------------------------------------------------------- #
# XGrammar schema
# --------------------------------------------------------------------------- #

try:
    import xgrammar as xgr
    _XGRAMMAR_AVAILABLE = True
except ImportError:
    xgr = None  # type: ignore[assignment]
    _XGRAMMAR_AVAILABLE = False

# Models that returned 400 for guided_json during this run; skip the attempt on
# subsequent calls so we don't burn a round-trip per item. Thread-safe under GIL
# for simple add/in operations.
_GUIDED_JSON_UNSUPPORTED: set[str] = set()

# JSON schema sent to vLLM-backed NIM endpoints via extra_body.guided_json.
# XGrammar on the server enforces this schema token-by-token during decoding,
# so the model cannot produce unparseable JSON or omit the code field.
_CODE_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "python_code": {
            "type": "string",
            "description": (
                "Complete self-contained Python module containing the Triton "
                "kernel(s) and the exact wrapper function. No prose, no markdown "
                "fences, no test code."
            ),
        }
    },
    "required": ["python_code"],
    "additionalProperties": False,
}

# Compile the schema at module load time so a malformed schema fails fast
# rather than during a live generation run.
if _XGRAMMAR_AVAILABLE:
    try:
        _xgr_compiled_schema = xgr.Grammar.from_json_schema(
            json.dumps(_CODE_OUTPUT_SCHEMA)
        )
    except Exception as _exc:
        raise RuntimeError(
            f"XGrammar could not compile the output schema: {_exc}"
        ) from _exc
else:
    _xgr_compiled_schema = None


def _xgrammar_validate(text: str) -> bool:
    """Client-side XGrammar check: returns True if text matches _CODE_OUTPUT_SCHEMA."""
    if not _XGRAMMAR_AVAILABLE or _xgr_compiled_schema is None:
        return True
    try:
        matcher = xgr.GrammarMatcher(_xgr_compiled_schema)
        return bool(matcher.accept_string(text)) and matcher.is_terminated()
    except Exception:
        return True  # if the matcher itself errors, don't block the response


# --------------------------------------------------------------------------- #
# System prompt
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

# Variant used when the API enforces _CODE_OUTPUT_SCHEMA via guided_json:
# asks the model to put raw code (no markdown fences) in the python_code field.
PROMPT_HEADER_GUIDED = (
    "You are an expert Triton programmer. Write fast and correct Triton kernels "
    "and Python wrapper functions from functional descriptions and wrapper "
    "signatures. The wrapper function must exactly match the requested "
    "signature and behavior.\n\n"
    "Respond with a JSON object whose single key is 'python_code' and whose "
    "value is a complete, self-contained Python module as a plain string "
    "(no markdown fences, no prose). Include imports for torch, triton, and "
    "triton.language as tl. Do not include test code, file I/O, network calls, "
    "or benchmark harness code. "
    "Correctness is mandatory; optimize only after preserving behavior. Every "
    "Triton load and store must be guarded with valid masks so larger benchmark "
    "inputs cannot read or write out of bounds."
)

# --------------------------------------------------------------------------- #
# Code extraction and sanitization
# --------------------------------------------------------------------------- #


def _extract_code(text: str) -> str:
    s = (text or "").strip()
    # Handle JSON-structured output produced when guided_json is active.
    # The model may still wrap the code in markdown fences inside the JSON value,
    # so we fall through to the fence-stripping logic after extracting the field.
    try:
        obj = json.loads(s)
        if isinstance(obj, dict) and "python_code" in obj:
            code = str(obj["python_code"])
            inner = re.search(r"```(?:python|py)?\s*\n(.*?)\n```", code, re.DOTALL)
            return _sanitize_generated_code(inner.group(1) if inner else code)
    except (json.JSONDecodeError, ValueError):
        pass
    # Fall back to markdown code-block extraction for non-JSON responses.
    match = re.search(r"```(?:python|py)?\s*\n(.*?)\n```", s, re.DOTALL)
    if match:
        return _sanitize_generated_code(match.group(1))
    s = re.sub(r"^```(?:python|py)?\s*\n?", "", s)
    s = re.sub(r"\n?```\s*$", "", s)
    return _sanitize_generated_code(s)


def _sanitize_generated_code(code: str) -> str:
    """Remove common model artifacts without changing intended logic."""
    s = code or ""
    # FIM (fill-in-middle) tokens appear when the model confuses completion with chat mode.
    s = re.sub(r"<\|(?:fim_prefix|fim_middle|fim_suffix|fim_pad)\|>", "", s)
    s = re.sub(r"<\|(?:repo_name|file_sep|endoftext)\|>", "", s)
    # Known model typos for "triton" observed in generated output.
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


# --------------------------------------------------------------------------- #
# Feedback inspection helpers
# --------------------------------------------------------------------------- #


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
    # Detects kernels that passed small correctness tests but caused a GPU/container
    # fault during the larger perf benchmark. These need to be regenerated with better
    # masking rather than carried forward, since the fault hides their true behaviour.
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


# --------------------------------------------------------------------------- #
# Static validation
# --------------------------------------------------------------------------- #


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
    # Triton JIT kernels do not support Python's `and`/`or` boolean operators; they raise
    # UnsupportedLanguageConstruct at compile time. Models frequently generate them anyway.
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
    # Pre-execution AST check so obviously broken code never reaches the GPU subprocess,
    # keeping error feedback cheaper and the call_acc_dir clean.
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


# --------------------------------------------------------------------------- #
# Failure classification and repair guidance
# --------------------------------------------------------------------------- #


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


# --------------------------------------------------------------------------- #
# LLM API call
# --------------------------------------------------------------------------- #


def _llm_chat(
    messages: list[dict[str, str]],
    model: str,
    max_tokens: int,
    temperature: float,
    request_timeout_seconds: int,
    retries: int,
    use_guided_json: bool = False,
) -> str:
    import requests

    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("DASHSCOPE_API_KEY is not set")

    use_gj = use_guided_json and model not in _GUIDED_JSON_UNSUPPORTED
    print(
        f"  [LLM] key prefix={api_key[:12]}... model={model} guided_json={use_gj}",
        flush=True,
    )

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if use_gj:
        payload["guided_json"] = _CODE_OUTPUT_SCHEMA

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = requests.post(
                LLM_INVOKE_URL,
                headers=headers,
                json=payload,
                timeout=(30, request_timeout_seconds),
            )

            # If the endpoint doesn't support guided_json, fall back and retry
            # the same request immediately (doesn't consume an attempt).
            if response.status_code == 400 and use_gj:
                _GUIDED_JSON_UNSUPPORTED.add(model)
                use_gj = False
                payload.pop("guided_json", None)
                print(
                    f"  [LLM] guided_json rejected (400) for {model}; retrying without",
                    flush=True,
                )
                response = requests.post(
                    LLM_INVOKE_URL,
                    headers=headers,
                    json=payload,
                    timeout=(30, request_timeout_seconds),
                )

            if response.status_code == 401:
                raise RuntimeError(
                    f"LLM API authentication failed (401): {_tail(response.text, 800)}\n"
                    "Check that DASHSCOPE_API_KEY in .env is correct."
                )
            if response.status_code in {408, 409, 429, 500, 502, 503, 504}:
                raise RuntimeError(
                    f"LLM API retryable status {response.status_code}: "
                    f"{_tail(response.text, 800)}"
                )
            response.raise_for_status()

            data = response.json()
            text = (data.get("choices") or [{}])[0].get("message", {}).get("content", "").strip()
            if not text:
                raise RuntimeError("LLM API returned an empty response")

            if use_gj and not _xgrammar_validate(text):
                print(
                    f"  [LLM] XGrammar validation failed; proceeding with best-effort extraction",
                    flush=True,
                )

            return text
        except RuntimeError as exc:
            last_error = exc
            if "401" in str(exc):
                break  # no point retrying auth failures
            if attempt < retries - 1:
                time.sleep(min(30, 2**attempt))
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < retries - 1:
                time.sleep(min(30, 2**attempt))

    raise RuntimeError(f"LLM generation failed: {last_error}")


# --------------------------------------------------------------------------- #
# Prompt assembly
# --------------------------------------------------------------------------- #


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
    use_guided_json: bool = True,
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

    system_prompt = PROMPT_HEADER_GUIDED if use_guided_json else PROMPT_HEADER
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "\n\n".join(user_parts)},
    ]
