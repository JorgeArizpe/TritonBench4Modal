# TritonBench-T Iterative Runner

This repository runs TritonBench-T on Modal with an iterative DashScope (Qwen)
chat-completions generation, evaluation, and refinement loop. The app has been
split into focused Python modules; the active local entrypoint is `main.py`.

The pipeline generates Triton kernels from TritonBench-T instructions, evaluates
them, feeds per-file feedback back into the model, repeats for multiple
iterations, and writes a final best-version prediction set.

## Repository Layout

```text
.
|-- main.py                 # Modal local entrypoint
|-- modal_app.py            # Modal app, images, volume, local source packaging
|-- config.py               # Shared constants, defaults, image patch commands
|-- llm_secrets.py          # DashScope API key and Modal Secret resolution
|-- code_utils.py           # Prompting, LLM API calls, code extraction, validation
|-- data_utils.py           # TritonBench data loading and file mapping
|-- generation.py           # generate_iteration Modal function
|-- evaluation.py           # evaluate_iteration Modal function
|-- perf_utils.py           # Performance subprocess and JSON helpers
|-- best_versions.py        # Best-version selection and materialization
|-- formatting.py           # Console summary formatting
|-- requirements.txt        # Local Python dependencies
|-- requirements-local.txt  # Compatibility requirements file
`-- misc/
    |-- archive/            # Archived old scripts/examples
    `-- statistic_analysis/ # Saved JSONs, plots, reports, and analysis scripts
```

## Entry Point

Run the app through `main.py`:

```bash
py -m modal run main.py
```

Smoke test the first five simple-dataset items:

```bash
py -m modal run main.py -- --limit 5
```

Run the complex instruction set:

```bash
py -m modal run main.py -- --dataset comp
```

Run one iteration only:

```bash
py -m modal run main.py -- --iterations 1
```

Use a fixed run id for repeatable artifact paths and resume behavior:

```bash
py -m modal run main.py -- --run-id my-run-001
```

Skip performance benchmarking:

```bash
py -m modal run main.py -- --skip-efficiency
```

Download artifacts after a run:

```bash
py -m modal volume get tritonbench-t-data nvidia_iterative_runs/my-run-001 ./local-my-run-001
```

## Local Setup

Install local dependencies and authenticate Modal:

```bash
pip install -r requirements.txt
modal setup
```


## DashScope Credentials

`llm_secrets.py` resolves DashScope credentials in this order:

1. `DASHSCOPE_API_KEY` from the local environment.
2. `DASHSCOPE_API_KEY` from `.env` next to the Python modules.
3. A Modal secret named by `TRITONBENCH_LLM_SECRET`, defaulting to
   `tritonbench-llm`.

Example `.env`:

```text
DASHSCOPE_API_KEY=sk-...
```

Example fallback Modal secret:

```bash
modal secret create tritonbench-llm DASHSCOPE_API_KEY=sk-...
```

Use a different Modal secret name with:

```bash
export TRITONBENCH_LLM_SECRET=my-secret-name
```

## Defaults

Defaults live in `config.py` and `generation.py`.

| Setting | Default |
| --- | --- |
| Modal app name | `tritonbench-t-dashscope-iterative` |
| TritonBench repo | `https://github.com/thunlp/TritonBench.git` |
| Modal volume | `tritonbench-t-data` |
| Remote data dir | `/data` |
| Remote TritonBench repo dir | `/opt/TritonBench` |
| Run directory | `dashscope_iterative_runs` |
| GPU | `T4`, overridable with `TRITONBENCH_GPU` |
| Model | `qwen3-coder-flash`, overridable with `LLM_MODEL` |
| Iterations | `3` |
| Generation concurrency | `4` |
| Max tokens | `4096` |
| Temperature | `0.15` |
| Request timeout | `600` seconds |
| Retries | `3` |
| Checkpoint every | `1` generated item |
| Include reference source | `false` |
| Reference source character limit | `6000` |
| Use best so far | `true` |
| Refine passing kernels | `false` |
| Repair performance failures | `false` |
| Max feedback history | `2` |
| Force regenerate | `false` |
| Loop mode | `auto` |
| Target speedup | `1.0` |
| Auto optimize minimum execution rate | `1.0` |
| Performance batch size | `8` |
| Skip efficiency | `false` |
| Guided JSON output | `true` |

## CLI Options

`main.py` exposes these options:

| Option | Meaning |
| --- | --- |
| `--dataset` | TritonBench-T Alpaca split: `simp` or `comp`. |
| `--limit` | Number of items to run. `0` means all items. |
| `--model` | DashScope model id. |
| `--iterations` | Number of generate/evaluate/refine loops. |
| `--concurrency` | Parallel DashScope generation requests. |
| `--max-tokens` | Maximum response tokens per generation request. |
| `--temperature` | Sampling temperature for DashScope requests. |
| `--request-timeout-seconds` | Read timeout for each DashScope request. |
| `--retries` | Retry count for each DashScope request. |
| `--checkpoint-every` | Modal volume commit frequency during generation. |
| `--include-reference-source` | Include reference implementation/context in prompts. |
| `--reference-source-char-limit` | Character limit for reference context. |
| `--use-best-so-far` | Feed interim best predictions into later iterations. |
| `--refine-passing` | Regenerate execution-correct kernels instead of carrying them forward. |
| `--repair-perf-failures` | Regenerate execution-correct kernels with performance failures in correctness mode. |
| `--max-feedback-history` | Number of prior feedback entries included in prompts. |
| `--force-regenerate` | Ignore saved generation records and generate again. |
| `--loop-mode` | One of `auto`, `correctness`, or `optimize`. |
| `--target-speedup` | Speedup target used by optimize prompts and carry-forward decisions. |
| `--auto-optimize-min-exec-rate` | Exec-accuracy threshold before `auto` mode optimizes passing kernels. |
| `--perf-batch-size` | Number of execution-correct kernels per performance batch. |
| `--skip-efficiency` | Skip Phase 3 performance benchmarking. |
| `--use-guided-json` | Ask supported endpoints for JSON output with a `python_code` field. |
| `--run-id` | Artifact directory name. Defaults to a UTC timestamp. |

## Modal Images

`modal_app.py` defines two images.

CPU generation image:

```text
debian_slim(python_version="3.12")
apt: git
pip: requests>=2.32, xgrammar>=0.1.14
```

GPU evaluation image:

```text
nvidia/cuda:12.4.1-devel-ubuntu22.04 with Python 3.12
apt: git, build-essential
pip: torch==2.5.1, triton==3.1.0, tqdm==4.66.5, numpy<2, requests>=2.32
```

Both images clone TritonBench into `/opt/TritonBench`. `modal_app.py` also calls
`add_local_python_source` for the helper modules that Modal containers need at
import time.

## TritonBench Patches

`config.py` defines patch commands applied during image build:

- `EVAL/eval_T/0_call_acc.py`
  - Uses `/opt/TritonBench/data/TritonBench_T_v1.jsonl`.
  - Uses `/opt/TritonBench/data/TritonBench_T_v1/`.
  - Uses `sys.executable` as the Python interpreter.
- `EVAL/eval_T/1_exe_acc.py`
  - Uses `/opt/TritonBench/data/TritonBench_T_v1/`.
  - Uses `sys.executable` as the Python interpreter.
- `performance_metrics/perf_T/run_bench/multiprocess_gpu_run.py`
  - Sets `gpu_count = 1`.

The GPU image also symlinks the digit-prefixed eval scripts as importable
modules named `call_acc.py` and `exe_acc.py`.

## Generation

`generation.py` defines `generate_iteration`, which runs on the CPU image. It:

1. Loads the selected TritonBench-T Alpaca dataset.
2. Maps each instruction to a TritonBench file.
3. Builds prompts with `code_utils._build_messages`.
4. Optionally includes reference context from the matching TritonBench file.
5. Optionally includes previous code, prior feedback, and feedback history.
6. Calls DashScope's chat-completions endpoint through `code_utils._llm_chat`.
7. Extracts/sanitizes generated Python code.
8. Writes prompts, generated scripts, generation records, and `predictions.jsonl`.

Generation reuses existing records unless `--force-regenerate` is set. Passing
kernels can be carried forward to avoid spending API quota on already-correct
outputs.

## Guided JSON Output

The current modularized app includes a guided JSON mode, enabled by default with
`DEFAULT_USE_GUIDED_JSON = True` in `generation.py`.

When enabled:

- `code_utils.py` defines a JSON schema requiring a single `python_code` string.
- The request sends `extra_body: {"guided_json": schema}` to endpoints that
  support the vLLM/NIM extension.
- If the endpoint returns HTTP 400 for guided JSON, the model is cached as
  unsupported for the current run and requests continue without server-side
  guidance.
- If `xgrammar` is available locally in the generation container, the response
  can be checked client-side against the same JSON schema.
- `_extract_code` handles both JSON responses and markdown fenced code.

Disable guided JSON for comparison:

```bash
py -m modal run main.py -- --use-guided-json false
```

## Static Validation

`code_utils._static_validate_code` checks generated code before GPU execution:

- Empty generated code.
- Leftover model special tokens.
- Common misspellings of `triton`.
- Python AST parse and compile.
- Required wrapper function name from `Wrapper Entry Information`.
- Python boolean operators inside `@triton.jit` functions, where Triton mask
  expressions should use `&`, `|`, and parentheses.

Static validation failures are saved as per-file feedback and can be used in
later prompts.

## Evaluation

`evaluation.py` defines `evaluate_iteration`, which runs on the GPU image.

Phase 1: call accuracy

- Concatenates the generated module with the official TritonBench-T test body.
- Executes the combined script on GPU 0.
- Keeps scripts that run successfully.

Phase 2: execution accuracy

- Runs each Phase 1 survivor.
- Runs the corresponding golden reference file.
- Requires generated stdout to match golden stdout exactly.

Phase 3: efficiency

- Runs unless `--skip-efficiency` is set.
- Benchmarks execution-correct kernels in batches controlled by
  `--perf-batch-size`.
- Uses TritonBench performance scripts under `performance_metrics/perf_T`.
- Compares generated perf JSON files against golden perf JSON files.
- Computes per-file speedups and accepts values where `0.1 < speedup < 10`.
- Reuses prior performance feedback for carried-forward kernels when available.

## Iteration Strategy

The loop can carry forward already-correct kernels instead of regenerating them:

- Execution-correct kernels are carried forward by default.
- `--refine-passing` allows passing kernels to be regenerated.
- `--loop-mode correctness` prioritizes correctness repair.
- `--loop-mode optimize` optimizes execution-correct kernels.
- `--loop-mode auto` optimizes only after the previous execution rate reaches
  `--auto-optimize-min-exec-rate`.
- Kernels with GPU-fault-like feedback are regenerated, because the larger
  performance benchmark may have exposed unsafe memory access.
- `--use-best-so-far` materializes interim best predictions between iterations
  and uses them as the next iteration baseline.

## Artifacts

Artifacts are written to the Modal volume `tritonbench-t-data`.

```text
dashscope_iterative_runs/<run_id>/
|-- iter_01/
|   |-- predictions.jsonl
|   |-- generation_manifest.json
|   |-- prompts/
|   |-- generation_records/
|   |-- generated_scripts/
|   `-- results/
|       |-- call_acc/
|       |-- perf_results/
|       |-- perf_batches/
|       |-- phase12_checkpoint.json
|       |-- per_file_feedback.json
|       `-- summary.json
|-- iter_02/
|-- iter_03/
`-- best/
    |-- best_predictions.jsonl
    |-- generated_scripts/
    `-- final_summary.json
```

Important files:

- `prompts/*.json`: exact chat messages sent to DashScope.
- `generation_records/*.json`: per-file generation result and errors.
- `generated_scripts/*.py`: extracted generated modules.
- `predictions.jsonl`: TritonBench-compatible prediction rows.
- `results/per_file_feedback.json`: feedback used in later iterations.
- `results/summary.json`: iteration metrics and performance analysis.
- `best/best_predictions.jsonl`: final selected predictions.
- `best/final_summary.json`: final run summary.

## Resume Behavior

Reusing a `--run-id` lets the runner reuse saved work:

- Existing generation records are reused unless `--force-regenerate` is set.
- Generation errors can be retried.
- Volume commits happen during generation according to `--checkpoint-every`.
- Phase 1 and Phase 2 write `phase12_checkpoint.json`.
- Phase 3 writes `perf_batches/batch_state.json`.
- A perf batch previously marked `started` is treated as a likely
  evaluator-container crash and is marked `skipped_after_crash` on resume.

## Local Analysis Scripts

Archived analysis scripts live under `misc/statistic_analysis/scripts/`:

- `plot_iteration_metrics.py`
- `report_iteration_stats.py`

Both scripts use only the Python standard library.

## Current Notes

- `main.py` is the active entrypoint.
- The older monolithic scripts are archived under `misc/archive/`.

## References

- TritonBench repository: <https://github.com/thunlp/TritonBench>
- Modal documentation: <https://modal.com/docs/guide>
