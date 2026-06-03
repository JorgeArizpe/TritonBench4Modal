# TritonBench-T NVIDIA Iterative Runner

This repository runs TritonBench-T on Modal with an iterative NVIDIA
chat-completions generation loop. The main script is
`nvidia_iterative_modal_app.py`.

The pipeline generates Triton kernels, evaluates them against TritonBench-T,
feeds per-file feedback back into the model, repeats for multiple iterations,
and writes a final best-version prediction set.

## Project Files

```text
.
|-- nvidia_iterative_modal_app.py   # Main Modal app and local entrypoint
|-- old_modal_app.py                # Older single-pass baseline app kept locally
|-- nvidia_example.py               # Minimal NVIDIA API example
|-- requirements-local.txt          # Local Modal dependency
|-- jsons/                          # Saved JSON artifacts from prior runs
|-- plots/                          # Saved plots
|-- reports/                        # Saved report output
`-- scripts/                        # Local plotting/report scripts
```

`README.md` is the only README in this workspace.

## What The Pipeline Does

`nvidia_iterative_modal_app.py` defines one local entrypoint, `main`, and three
remote Modal functions:

- `generate_iteration`: calls NVIDIA's chat-completions endpoint and writes one
  prediction set for an iteration.
- `evaluate_iteration`: runs static validation, call accuracy, execution
  accuracy, and optional performance benchmarking.
- `materialize_best_versions`: selects the best generated version found for
  each TritonBench file and writes the final artifacts.

By default the loop runs for three iterations. Each new iteration can use the
best version found so far plus feedback from previous failures or benchmark
results.

## Local Setup

Install the local Modal dependency:

```bash
pip install -r requirements-local.txt
modal setup
```

`requirements-local.txt` currently contains:

```text
modal>=0.66
```

## NVIDIA Credentials

The app reads NVIDIA credentials in this order:

1. `NVIDIA_KEY` from the local environment.
2. `NVIDIA_API_KEY` from the local environment.
3. `NVIDIA_KEY` or `NVIDIA_API_KEY` from a local `.env` file next to
   `nvidia_iterative_modal_app.py`.
4. A Modal secret named by `TRITONBENCH_LLM_SECRET`, defaulting to
   `tritonbench-llm`.

Example local `.env`:

```text
NVIDIA_KEY=nvapi-...
```

Example fallback Modal secret:

```bash
modal secret create tritonbench-llm NVIDIA_KEY=nvapi-...
```

Use a different Modal secret name with:

```bash
export TRITONBENCH_LLM_SECRET=my-secret-name
```

## Run Commands

Smoke test the first five simple-dataset items:

```bash
py -m modal run nvidia_iterative_modal_app.py -- --limit 5
```

Run the default full job:

```bash
py -m modal run nvidia_iterative_modal_app.py
```

Run the complex instruction set:

```bash
py -m modal run nvidia_iterative_modal_app.py -- --dataset comp
```

Run one iteration:

```bash
py -m modal run nvidia_iterative_modal_app.py -- --iterations 1
```

Use a fixed run id for repeatable artifact paths and resume behavior:

```bash
py -m modal run nvidia_iterative_modal_app.py -- --run-id my-run-001
```

Skip performance benchmarking and only measure static, call, and execution
correctness:

```bash
py -m modal run nvidia_iterative_modal_app.py -- --skip-efficiency
```

Download run artifacts:

```bash
py -m modal volume get tritonbench-t-data nvidia_iterative_runs/my-run-001 ./local-my-run-001
```

## Defaults

The current defaults are defined near the top of
`nvidia_iterative_modal_app.py`.

| Setting | Default |
| --- | --- |
| Modal app name | `tritonbench-t-nvidia-iterative` |
| TritonBench repo | `https://github.com/thunlp/TritonBench.git` |
| Modal volume | `tritonbench-t-data` |
| Run directory | `nvidia_iterative_runs` |
| Dataset | `simp` |
| GPU | `T4` from `TRITONBENCH_GPU` |
| Model | `qwen/qwen3-coder-480b-a35b-instruct` from `NVIDIA_MODEL` |
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
| Repair perf failures | `false` |
| Max feedback history | `2` |
| Force regenerate | `false` |
| Loop mode | `auto` |
| Target speedup | `1.0` |
| Auto optimize minimum exec rate | `1.0` |
| Perf batch size | `8` |
| Skip efficiency | `false` |

## CLI Options

`main` accepts these options:

| Option | Meaning |
| --- | --- |
| `--dataset` | TritonBench-T Alpaca split: `simp` or `comp`. |
| `--limit` | Number of items to run. `0` means all items. |
| `--model` | NVIDIA model id. |
| `--iterations` | Number of generate/evaluate/refine loops. |
| `--concurrency` | Parallel NVIDIA generation requests. |
| `--max-tokens` | Maximum response tokens per generation request. |
| `--temperature` | Sampling temperature for NVIDIA requests. |
| `--request-timeout-seconds` | Read timeout for each NVIDIA request. |
| `--retries` | Retry count for each NVIDIA request. |
| `--checkpoint-every` | Volume commit frequency during generation. |
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
| `--run-id` | Artifact directory name. Defaults to a UTC timestamp. |

## Remote Images

The CPU generation image uses Debian slim with Python 3.12 and installs:

```text
git
requests>=2.32
```

The GPU evaluation image uses:

```text
nvidia/cuda:12.4.1-devel-ubuntu22.04
Python 3.12
git
build-essential
torch==2.5.1
triton==3.1.0
tqdm==4.66.5
numpy<2
requests>=2.32
```

Both images clone `https://github.com/thunlp/TritonBench.git` into
`/opt/TritonBench`.

## TritonBench Patches

During image build, the app patches TritonBench paths and runtime assumptions:

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

Generation runs in `generate_iteration` on the CPU image. For each selected
TritonBench item it:

1. Builds chat messages from the Alpaca instruction.
2. Optionally adds reference source context.
3. Optionally adds previous generated code and feedback.
4. Calls `https://integrate.api.nvidia.com/v1/chat/completions` with streaming
   enabled.
5. Extracts a Python module from a `python` Markdown code block when present.
6. Sanitizes common model artifacts and misspellings.
7. Writes generated scripts, prompts, generation records, and
   `predictions.jsonl`.

The prompt requires a single self-contained Python module with imports for
`torch`, `triton`, and `triton.language as tl`, Triton kernels, and the requested
wrapper function. It forbids tests, examples, prose outside the code block,
fill-in-middle tokens, file I/O, network calls, and benchmark harness code.

## Static Validation

Before generated code is executed, `evaluate_iteration` performs static checks:

- Empty generated code.
- Leftover model special tokens.
- Common misspellings of `triton`.
- Python `ast.parse` and compile.
- Required wrapper function name from `Wrapper Entry Information`.
- Python boolean operators inside `@triton.jit` functions, where tensor mask
  expressions should use `&`, `|`, and parentheses.

Static validation failures are stored in per-file feedback and can be included
in later prompts.

## Evaluation

Evaluation runs in `evaluate_iteration` on the GPU image.

Phase 1: call accuracy

- Concatenates the generated module with the official TritonBench-T test body.
- Runs the combined script with `CUDA_VISIBLE_DEVICES=0`.
- Keeps scripts that execute successfully.

Phase 2: execution accuracy

- Runs each Phase 1 survivor.
- Runs the corresponding golden reference file.
- Requires generated stdout to match golden stdout exactly.

Phase 3: efficiency

- Benchmarks execution-correct kernels unless `--skip-efficiency true` is set.
- Splits performance runs into batches controlled by `--perf-batch-size`.
- Runs TritonBench's performance script writer and GPU runner.
- Matches generated performance JSON files to golden performance JSON files.
- Computes per-file speedups and accepts values where `0.1 < speedup < 10`.
- Reuses prior performance results for carried-forward kernels when available.

## Iteration And Carry-Forward Behavior

The loop can carry forward already-correct kernels instead of regenerating them:

- Execution-correct kernels are carried forward by default.
- `--refine-passing true` allows passing kernels to be regenerated.
- `--loop-mode correctness` prioritizes correctness.
- `--loop-mode optimize` optimizes execution-correct kernels.
- `--loop-mode auto` optimizes only after the previous execution rate reaches
  `--auto-optimize-min-exec-rate`.
- Kernels with GPU-fault-like feedback are regenerated, because the larger
  performance benchmark may have exposed unsafe memory access.

`--use-best-so-far true` materializes interim best predictions between
iterations and uses them as the starting point for the next iteration.

## Artifacts

Artifacts are written to the Modal volume `tritonbench-t-data`.

Run-level layout:

```text
nvidia_iterative_runs/<run_id>/
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

- `prompts/*.json`: exact chat messages sent to NVIDIA.
- `generation_records/*.json`: per-file generation result and errors.
- `generated_scripts/*.py`: extracted generated modules.
- `predictions.jsonl`: TritonBench-compatible prediction rows.
- `results/per_file_feedback.json`: feedback used in later iterations.
- `results/summary.json`: iteration metrics and performance analysis.
- `best/best_predictions.jsonl`: final selected predictions.
- `best/final_summary.json`: final run summary.

## Resume Behavior

The runner can reuse saved work when a run id is reused:

- Existing generation records are reused unless `--force-regenerate true`.
- Generation errors can be retried.
- Volume commits happen during generation according to `--checkpoint-every`.
- Phase 1 and Phase 2 write `phase12_checkpoint.json`.
- Phase 3 writes `perf_batches/batch_state.json`.
- A batch previously marked `started` is treated as a likely evaluator-container
  crash and is marked `skipped_after_crash` on resume.

## Console Output

The local run prints:

- Run configuration.
- Per-iteration prediction path.
- Call and execution accuracy counts.
- Efficiency aggregate and computed mean speedups.
- Per-file performance status.
- Non-execution-correct files with compact failure reasons.
- Final best-version summary.
- A `modal volume get` command for downloading artifacts.

Full details are stored in JSON artifacts in the Modal volume.

## Notes

- There is no grammar-guided decoding or xGrammar integration in the current
  script.
- The only documented primary entrypoint is `nvidia_iterative_modal_app.py`.
- `old_modal_app.py` exists locally as an older baseline file, but this README
  documents the NVIDIA iterative runner.

## References

- TritonBench repository: <https://github.com/thunlp/TritonBench>
- Modal documentation: <https://modal.com/docs/guide>
