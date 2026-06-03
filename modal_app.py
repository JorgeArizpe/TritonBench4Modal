"""
TritonBench-T on Modal — translate PyTorch ops to Triton kernels with an LLM,
then evaluate them on the cheapest available Modal GPU (NVIDIA T4).
 
Pipeline
--------
1. ``generate_predictions``  — calls a configured LLM provider on each Alpaca
   instruction in ``data/TritonBench_T_<simp|comp>_alpac_v1.json`` and writes a
   ``predictions.jsonl`` into a persistent Modal Volume.
 
   NEW: Each kernel goes through a Repair Loop (up to MAX_REPAIR_ATTEMPTS).
   After each failed attempt the error traceback is fed back to the LLM so it
   can correct its own code before the prediction is written to disk.
 
2. ``evaluate``              — runs the three TritonBench-T phases on a GPU:
       phase 1: call accuracy   (does the generated module run at all?)
       phase 2: execution acc.  (does it produce the same outputs as PyTorch?)
       phase 3: efficiency      (speedup vs. the golden PyTorch baseline)
 
A single ``main`` local entrypoint chains them end-to-end.
 
Quick start (see README.md for full instructions):
 
    pip install modal
    modal setup
    modal secret create tritonbench-llm ANTHROPIC_API_KEY=sk-ant-...
    modal run modal_app.py                        # generate + evaluate
    modal run modal_app.py -- --limit 5           # smoke test on 5 ops
    modal run modal_app.py -- --predictions ./preds.jsonl   # bring your own
"""
 
from __future__ import annotations
 
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
 
import modal
 
# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
 
APP_NAME = "tritonbench-t"
TRITONBENCH_REPO = "https://github.com/thunlp/TritonBench.git"
 
# Cheapest Modal GPU (compute capability 7.5 — Triton requires >= 7.0).
# Override at runtime via `--gpu A10` etc. on the local entrypoint.
DEFAULT_GPU = "T4"
 
VOLUME_NAME = "tritonbench-t-data"
DATA_DIR = "/data"           # mount point of the Modal Volume in the container
REPO_DIR = "/opt/TritonBench"
 
# Default model targets — students can override from the CLI.
DEFAULT_PROVIDER = "anthropic"
DEFAULT_MODEL = "claude-sonnet-4-6"
 
# Name of the Modal Secret that holds your LLM API key(s) (e.g. ANTHROPIC_API_KEY,
# OPENAI_API_KEY). Override with an env var if your existing secret is named
# differently — no code edit required:
#     export TRITONBENCH_LLM_SECRET=openai-secret
LLM_SECRET_NAME = os.environ.get("TRITONBENCH_LLM_SECRET", "tritonbench-llm")
 
# --------------------------------------------------------------------------- #
# Repair Loop configuration
# --------------------------------------------------------------------------- #
 
# Maximum number of LLM attempts per kernel before giving up.
# Attempt 1 = initial generation, attempts 2..N = repair iterations.
# Each failed attempt sends the error traceback back to the LLM.
MAX_REPAIR_ATTEMPTS = 3
 
# --------------------------------------------------------------------------- #
# Image — patches TritonBench's hardcoded paths so the eval scripts run inside
# a clean container without any local-machine assumptions.
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
        "anthropic>=0.40",
        "openai>=1.50",
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
# Generation — LLM-based PyTorch → Triton translation
# --------------------------------------------------------------------------- #
 
PROMPT_HEADER = (
    "You are an expert in Triton GPU programming using triton==3.1.0 and torch==2.5.1.\n\n"
 
    "STRICT RULES — violating any of these will cause a runtime crash:\n\n"
 
    # Prevents: AttributeError: module 'triton.language' has no attribute 'kernel'
    "1. Always decorate kernels with @triton.jit. "
    "NEVER use @tl.kernel or @tl.jit — those decorators do not exist.\n\n"
 
    # Prevents: AttributeError: module 'triton' has no attribute 'cudagraphs'
    "2. Only use APIs that exist in triton 3.1.0: triton.jit, triton.cdiv, "
    "tl.program_id, tl.load, tl.store, tl.arange, tl.zeros, tl.dot, tl.exp, "
    "tl.log, tl.sqrt, tl.maximum, tl.minimum, tl.where, tl.sum, tl.max. "
    "NEVER invent attributes like triton.cudagraphs, triton.Config, "
    "tl.kernel, or any other name you are not certain exists.\n\n"
 
    # Prevents: IndexError: tuple index out of range
    "3. Launch kernels as: kernel[grid](args...) where grid is a tuple of ints "
    "or a lambda, for example: kernel[(n_blocks,)](args...). "
    "NEVER write kernel[grid[0]](...) or kernel[grid[i]](...).\n\n"
 
    # Prevents: RuntimeError: Cannot call @triton.jit'd outside of the scope of a kernel
    "4. @triton.jit kernels are launched ONLY from Python host code using the "
    "bracket syntax above. NEVER call a @triton.jit function from inside "
    "another @triton.jit kernel.\n\n"
 
    # Prevents: AttributeError: 'str' object has no attribute 'dtype'
    "5. Kernel arguments must be tensors or plain Python ints/floats. "
    "NEVER pass strings, tuples, lists, or Python enums as kernel arguments. "
    "If the wrapper receives a string parameter (e.g. rounding_mode='floor'), "
    "convert it to an int flag before passing it to the kernel.\n\n"
 
    "OUTPUT FORMAT:\n"
    "Output a single, self-contained Python module containing: (a) the necessary "
    "imports (torch, triton, triton.language as tl), (b) the Triton kernel(s) "
    "decorated with @triton.jit, and (c) the wrapper function that fully matches "
    "the provided signature. Wrap the entire module in one ```python ... ``` "
    "fenced code block. Do NOT include any test code or example calls — "
    "tests will be appended separately."
)
 
 
def _load_alpaca(dataset: str) -> list[dict]:
    assert dataset in ("simp", "comp"), "dataset must be 'simp' or 'comp'"
    path = Path(REPO_DIR) / f"data/TritonBench_T_{dataset}_alpac_v1.json"
    return json.loads(path.read_text())
 
 
def _build_messages(item: dict) -> list[dict]:
    instr = item["instruction"]
    inp = item.get("input", "") or ""
    user = instr if not inp else f"{instr}\n\n{inp}"
    return [
        {"role": "system", "content": PROMPT_HEADER},
        {"role": "user", "content": user},
    ]
 
 
def _build_repair_messages(
    original_messages: list[dict],
    failed_code: str,
    error: str,
    attempt: int,
) -> list[dict]:
    """
    Extend the conversation history with the failed code and its error so the
    LLM can self-correct on the next attempt.
 
    Conversation shape after N failures:
        system  : PROMPT_HEADER
        user    : original instruction
        assistant: attempt 1 code
        user    : error feedback 1
        assistant: attempt 2 code        (if N >= 2)
        user    : error feedback 2       (if N >= 2)
        ...
    """
    return original_messages + [
        {
            "role": "assistant",
            "content": f"```python\n{failed_code}\n```",
        },
        {
            "role": "user",
            "content": (
                f"Your code (attempt {attempt}) failed with this error:\n\n"
                f"```\n{error}\n```\n\n"
                f"Fix the code and output the complete corrected module in a "
                f"single ```python ... ``` block. Re-check these rules before answering:\n"
                f"- @triton.jit only, never @tl.kernel\n"
                f"- No invented attributes (triton.cudagraphs, etc.)\n"
                f"- Kernel args: tensors or plain int/float scalars only\n"
                f"- Kernel launch: kernel[(n_blocks,)](args...) syntax\n"
                f"- Never call a @triton.jit kernel from inside another kernel"
            ),
        },
    ]
 
 
def _try_execution_check(code: str) -> tuple[bool, str]:
    """
    Run the generated module in an isolated subprocess to detect errors that
    occur before any kernel is called: syntax errors, bad decorators (@tl.kernel),
    and references to non-existent attributes (triton.cudagraphs, etc.).
 
    The GPU is available in the generate_predictions container so Triton can
    validate decorator-time JIT metadata when the module is loaded.
 
    Returns (passed, error_message).
    """
    import tempfile
 
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False
    ) as tmp:
        tmp.write(code)
        tmp_path = tmp.name
 
    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=20,
        )
        if result.returncode == 0:
            return True, ""
        error = (result.stderr or result.stdout).strip()
        # Trim to keep repair prompt tokens reasonable
        return False, error[:800]
    except subprocess.TimeoutExpired:
        return False, "TimeoutError: module took too long to load (> 20 s)"
    finally:
        try:
            Path(tmp_path).unlink()
        except OSError:
            pass
 
 
def _gen_anthropic(messages: list[dict], model: str) -> str:
    import anthropic
 
    client = anthropic.Anthropic()
    sys_prompt = next((m["content"] for m in messages if m["role"] == "system"), "")
    user_msgs = [m for m in messages if m["role"] != "system"]
    resp = client.messages.create(
        model=model,
        max_tokens=8192,
        system=sys_prompt,
        messages=user_msgs,
    )
    return resp.content[0].text
 
 
def _gen_openai(messages: list[dict], model: str) -> str:
    from openai import OpenAI
 
    client = OpenAI()
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        max_completion_tokens=8192,
    )
    return resp.choices[0].message.content
 
 
_GENERATORS = {"anthropic": _gen_anthropic, "openai": _gen_openai}
 
 
def _extract_code(text: str) -> str:
    """Strip Markdown code fences from an LLM reply; return raw Python source."""
    import re
 
    s = text.strip()
    m = re.search(r"```(?:python|py)?\s*\n(.*?)\n```", s, re.DOTALL)
    if m:
        return m.group(1).strip() + "\n"
    s = re.sub(r"^```(?:python|py)?\s*\n?", "", s)
    s = re.sub(r"\n?```\s*$", "", s)
    return s.strip() + "\n"
 
 
# --------------------------------------------------------------------------- #
# generate_predictions — now with GPU for the Repair Loop execution check
# --------------------------------------------------------------------------- #
 
@app.function(
    # GPU added so _try_execution_check can load Triton kernels and catch
    # decoration-time errors (wrong decorator, non-existent attributes, etc.)
    # in the same environment that the evaluator uses.
    gpu=DEFAULT_GPU,
    timeout=60 * 60 * 4,
    cpu=4,
    volumes={DATA_DIR: data_volume},
    secrets=[modal.Secret.from_name(LLM_SECRET_NAME)],
)
def generate_predictions(
    provider: str = DEFAULT_PROVIDER,
    model: str = DEFAULT_MODEL,
    dataset: str = "simp",
    output_path: str = "predictions.jsonl",
    limit: int | None = None,
    concurrency: int = 8,
) -> str:
    """Generate Triton translations with an iterative Repair Loop.
 
    Each kernel gets up to MAX_REPAIR_ATTEMPTS chances. On every failed attempt
    the error traceback is appended to the conversation and the LLM is asked to
    self-correct before the next generation.
 
    Returns the volume-relative path of the produced jsonl.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
 
    if provider not in _GENERATORS:
        raise ValueError(
            f"unknown provider {provider!r} — choose one of {list(_GENERATORS)}"
        )
 
    items = _load_alpaca(dataset)
    if limit:
        items = items[:limit]
 
    print(
        f"generating {len(items)} predictions with {provider}/{model} "
        f"(max {MAX_REPAIR_ATTEMPTS} attempts per kernel)",
        flush=True,
    )
 
    gen_fn = _GENERATORS[provider]
 
    def _do(idx_item: tuple[int, dict]) -> tuple[int, dict]:
        i, item = idx_item
        original_messages = _build_messages(item)
 
        code = "# generation failed\n"
        last_error = ""
        passed = False
 
        for attempt in range(1, MAX_REPAIR_ATTEMPTS + 1):
            try:
                # Build messages: plain instruction on attempt 1,
                # conversation history with error feedback on subsequent attempts.
                if attempt == 1:
                    messages = original_messages
                else:
                    messages = _build_repair_messages(
                        original_messages, code, last_error, attempt - 1
                    )
 
                raw = gen_fn(messages, model)
                code = _extract_code(raw)
 
                ok, error = _try_execution_check(code)
 
                if ok:
                    passed = True
                    print(
                        f"  [{i}] ✓ passed on attempt {attempt}/{MAX_REPAIR_ATTEMPTS}",
                        flush=True,
                    )
                    break
                else:
                    last_error = error
                    first_line = error.splitlines()[0][:80] if error else "unknown error"
                    print(
                        f"  [{i}] ✗ attempt {attempt}/{MAX_REPAIR_ATTEMPTS} "
                        f"— {first_line}",
                        flush=True,
                    )
 
            except Exception as exc:  # noqa: BLE001
                code = f"# generation error: {exc}\n"
                last_error = str(exc)
                print(
                    f"  [{i}] ✗ attempt {attempt}/{MAX_REPAIR_ATTEMPTS} "
                    f"— exception: {exc}",
                    flush=True,
                )
 
        if not passed:
            print(
                f"  [{i}] ✗ all {MAX_REPAIR_ATTEMPTS} attempts exhausted",
                flush=True,
            )
 
        return i, {"instruction": item["instruction"], "predict": code}
 
    # ---- Dispatch with a thread pool (same as before) ----------------------
    results: list[dict | None] = [None] * len(items)
    done = 0
 
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = [ex.submit(_do, (i, it)) for i, it in enumerate(items)]
        for fut in as_completed(futs):
            i, rec = fut.result()
            results[i] = rec
            done += 1
            if done % 5 == 0 or done == len(items):
                print(f"  {done}/{len(items)} kernels finished", flush=True)
 
    # ---- Persist to volume -------------------------------------------------
    out = Path(DATA_DIR) / output_path
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    data_volume.commit()
    print(f"wrote {out}", flush=True)
    return output_path
 
 
# --------------------------------------------------------------------------- #
# Evaluation — runs all three TritonBench-T phases on one GPU
# --------------------------------------------------------------------------- #
 
@app.function(
    gpu=DEFAULT_GPU,
    timeout=60 * 60 * 6,
    volumes={DATA_DIR: data_volume},
)
def evaluate(
    predictions_path: str = "predictions.jsonl",
    output_subdir: str = "results",
) -> dict:
    """Run TritonBench-T eval phases against an existing predictions.jsonl."""
    pred_full = Path(DATA_DIR) / predictions_path
    if not pred_full.exists():
        raise FileNotFoundError(f"predictions file not found in volume: {pred_full}")
 
    out_dir = Path(DATA_DIR) / output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    call_acc_dir = out_dir / "call_acc"
    perf_results_dir = out_dir / "perf_results"
 
    if call_acc_dir.exists():
        shutil.rmtree(call_acc_dir)
    if perf_results_dir.exists():
        shutil.rmtree(perf_results_dir)
 
    eval_dir = f"{REPO_DIR}/EVAL/eval_T"
    if eval_dir not in sys.path:
        sys.path.insert(0, eval_dir)
    os.environ["PYTHONPATH"] = eval_dir + os.pathsep + os.environ.get("PYTHONPATH", "")
 
    import call_acc  # noqa: E402
    import exe_acc   # noqa: E402
 
    total = sum(1 for _ in pred_full.open())
 
    # ---- Phase 1: call accuracy --------------------------------------------
    print("\n" + "=" * 70 + "\n=== Phase 1: call accuracy ===\n" + "=" * 70, flush=True)
    call_acc.call_4file(str(pred_full), str(call_acc_dir), gpus=[0])
    call_survivors = sorted(p.name for p in call_acc_dir.glob("*.py"))
    print(f"\ncall_acc survivors: {len(call_survivors)} / {total}", flush=True)
 
    # ---- Phase 2: execution accuracy ---------------------------------------
    print("\n" + "=" * 70 + "\n=== Phase 2: execution accuracy ===\n" + "=" * 70, flush=True)
    if call_survivors:
        exe_acc.execute_4folder(str(call_acc_dir), gpus=[0])
    exec_survivors = sorted(p.name for p in call_acc_dir.glob("*.py"))
    print(f"\nexe_acc survivors: {len(exec_survivors)} / {total}", flush=True)
 
    # ---- Phase 3: efficiency -----------------------------------------------
    print("\n" + "=" * 70 + "\n=== Phase 3: efficiency ===\n" + "=" * 70, flush=True)
    eff_summary = "skipped (no surviving operators)"
    speedup = None
    if exec_survivors:
        perf_root = f"{REPO_DIR}/performance_metrics/perf_T"
 
        subprocess.run(
            [
                sys.executable,
                "run_bench/write_file.py",
                "--input_folder_path", str(call_acc_dir),
                "--results_path", str(perf_results_dir),
            ],
            cwd=perf_root,
            check=True,
        )
 
        subprocess.run(
            [sys.executable, "run_bench/multiprocess_gpu_run.py"],
            cwd=perf_root,
            check=True,
        )
 
        eff = subprocess.run(
            [
                sys.executable,
                "2_efficiency.py",
                "--gen_folder", str(perf_results_dir),
            ],
            cwd=f"{REPO_DIR}/EVAL/eval_T",
            capture_output=True,
            text=True,
        )
        eff_summary = eff.stdout
        if eff.stderr:
            eff_summary += "\n[stderr]\n" + eff.stderr
        for line in eff.stdout.splitlines():
            if line.startswith("speed up:"):
                try:
                    speedup = float(line.split(":", 1)[1].strip())
                except ValueError:
                    pass
 
    data_volume.commit()
 
    summary = {
        "total_predictions": total,
        "phase1_call_acc": {
            "passed": len(call_survivors),
            "rate": round(100 * len(call_survivors) / total, 2) if total else 0,
        },
        "phase2_exec_acc": {
            "passed": len(exec_survivors),
            "rate": round(100 * len(exec_survivors) / total, 2) if total else 0,
        },
        "phase3_efficiency": {
            "speedup_vs_pytorch": speedup,
            "raw_output_tail": eff_summary[-2000:],
        },
        "artifacts_volume": VOLUME_NAME,
        "artifacts_subdir": output_subdir,
    }
    return summary
 
 
# --------------------------------------------------------------------------- #
# Volume helpers + local entrypoint
# --------------------------------------------------------------------------- #
 
def _upload_local_predictions(local_path: Path) -> str:
    if not local_path.exists():
        raise FileNotFoundError(local_path)
    remote = f"uploads/{local_path.name}"
    print(f"uploading {local_path} -> volume://{remote}", flush=True)
    with data_volume.batch_upload(force=True) as batch:
        batch.put_file(str(local_path), remote)
    return remote
 
 
@app.local_entrypoint()
def main(
    predictions: str = "",
    provider: str = DEFAULT_PROVIDER,
    model: str = DEFAULT_MODEL,
    dataset: str = "simp",
    limit: int = 0,
    output_subdir: str = "results",
    concurrency: int = 8,
):
    """End-to-end: (optionally) generate predictions, then evaluate."""
    if predictions:
        remote = _upload_local_predictions(Path(predictions))
    else:
        tag = f"{provider}_{model.replace('/', '_').replace(':', '_')}_{dataset}"
        remote = generate_predictions.remote(
            provider=provider,
            model=model,
            dataset=dataset,
            output_path=f"predictions/{tag}.jsonl",
            limit=limit or None,
            concurrency=concurrency,
        )
 
    print(f"\nevaluating: volume://{remote}\n", flush=True)
    summary = evaluate.remote(
        predictions_path=remote,
        output_subdir=output_subdir,
    )
    print("\n=== Final summary ===")
    print(json.dumps(summary, indent=2))
 
 
@app.local_entrypoint()
def evaluate_only(
    predictions: str,
    output_subdir: str = "results",
):
    """Evaluate an existing local predictions.jsonl without (re)generating."""
    remote = _upload_local_predictions(Path(predictions))
    summary = evaluate.remote(predictions_path=remote, output_subdir=output_subdir)
    print(json.dumps(summary, indent=2))
 
 
@app.local_entrypoint()
def generate_only(
    provider: str = DEFAULT_PROVIDER,
    model: str = DEFAULT_MODEL,
    dataset: str = "simp",
    limit: int = 0,
    output_path: str = "predictions/predictions.jsonl",
    concurrency: int = 8,
):
    """Generate predictions only; do not evaluate."""
    remote = generate_predictions.remote(
        provider=provider,
        model=model,
        dataset=dataset,
        output_path=output_path,
        limit=limit or None,
        concurrency=concurrency,
    )
    print(f"wrote volume://{remote}")