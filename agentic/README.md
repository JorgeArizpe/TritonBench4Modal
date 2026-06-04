# TritonBench-T Agentic Runner

This folder contains an experimental agentic variant of the main
TritonBench-T NVIDIA iterative runner. It keeps the same Modal workflow as the
root runner, but adds bounded tool-using repair turns around generated kernels.

The normal outer loop is still deterministic:

1. Generate one prediction set.
2. Evaluate static validity, call accuracy, execution accuracy, and optionally
   performance.
3. Feed per-file feedback into the next iteration.
4. Materialize the best version found for each operator.

The agentic changes happen inside individual kernel attempts. The harness runs
tools such as static validation and GPU call/execution checks, then gives the
model a compact tool observation so it can rewrite only the failing candidate.

## Entry Point

Run from the repository root:

```bash
py -m modal run agentic/main.py
```

Pass local-entrypoint options directly after `agentic/main.py`.

Small smoke test:

```bash
py -m modal run agentic/main.py --limit 3 --iterations 2 --skip-efficiency
```

Faster timeout behavior while testing slow API responses:

```bash
py -m modal run agentic/main.py --limit 3 --iterations 1 --skip-efficiency --request-timeout-seconds 120 --retries 1
```

Run a single item with performance benchmarking disabled:

```bash
py -m modal run agentic/main.py --limit 1 --iterations 1 --skip-efficiency
```

Use a fixed run id:

```bash
py -m modal run agentic/main.py --limit 3 --iterations 2 --skip-efficiency --run-id agentic-smoke-001
```

Download artifacts:

```bash
py -m modal volume get tritonbench-t-data nvidia_iterative_runs/agentic-smoke-001 ./local-agentic-smoke-001
```

## Agentic Controls

Defaults live in `agentic/config.py`.

| Setting | Default | Meaning |
| --- | ---: | --- |
| `DEFAULT_AGENTIC_GENERATION` | `True` | Enables generation-side static validation repair. |
| `DEFAULT_AGENTIC_STATIC_REPAIR_ATTEMPTS` | `1` | Maximum extra LLM repair calls after generated code fails static validation. |
| `DEFAULT_AGENTIC_EVAL_REPAIR_ATTEMPTS` | `1` | Maximum extra LLM repair calls after Phase 1 or Phase 2 evaluation failures. |
| `DEFAULT_USE_GUIDED_JSON` | `True` | Requests a JSON object with a `python_code` field when supported by the endpoint. |

Disable the extra repair calls for a quick A/B run while keeping the same
folder and entrypoint:

```bash
py -m modal run agentic/main.py --limit 3 --iterations 2 --skip-efficiency --agentic-static-repair-attempts 0 --agentic-eval-repair-attempts 0
```

## What The Tools Do

Generation-side tool loop:

1. `generation.py` calls the NVIDIA chat-completions endpoint for an initial
   candidate.
2. The harness runs `code_utils._static_validate_code`.
3. If static validation fails and the repair budget allows it, the model gets a
   compact repair prompt from `code_utils._build_agentic_repair_messages`.
4. The repaired code is validated again and written to `generated_scripts/`,
   `generation_records/`, and `predictions.jsonl`.

Evaluation-side tool loop:

1. `evaluation.py` runs Phase 1 call accuracy on the GPU image.
2. A failed static or call-accuracy result can trigger one bounded repair turn.
3. Phase 2 compares generated stdout against the golden TritonBench file.
4. A Phase 2 mismatch can trigger one bounded repair turn, then the repaired
   candidate must pass Phase 1 again before Phase 2 trusts it.
5. Repaired code is written back to `predictions.jsonl` and
   `generated_scripts/`, so `best_versions.py` selects from the actual evaluated
   code.

## Credentials

`agentic/llm_secrets.py` resolves NVIDIA credentials in this order:

1. `NVIDIA_KEY` from the local environment.
2. `NVIDIA_API_KEY` from the local environment.
3. `NVIDIA_KEY` or `NVIDIA_API_KEY` from `agentic/.env`.
4. `NVIDIA_KEY` or `NVIDIA_API_KEY` from the repository-root `.env`.
5. A Modal secret named by `TRITONBENCH_LLM_SECRET`, defaulting to
   `tritonbench-llm`.

Example `.env`:

```text
NVIDIA_KEY=nvapi-...
```

## Artifacts

Artifacts use the same Modal volume and run directory convention as the root
runner:

```text
nvidia_iterative_runs/<run_id>/
|-- iter_01/
|   |-- predictions.jsonl
|   |-- generation_manifest.json
|   |-- prompts/
|   |-- agent_traces/
|   |-- eval_agent_prompts/
|   |-- eval_agent_traces/
|   |-- generation_records/
|   |-- generated_scripts/
|   `-- results/
|       |-- call_acc/
|       |-- perf_results/
|       |-- phase12_checkpoint.json
|       |-- per_file_feedback.json
|       `-- summary.json
`-- best/
    |-- best_predictions.jsonl
    |-- generated_scripts/
    `-- final_summary.json
```

Important agentic files:

- `agent_traces/*.json`: generation-side static validation and repair events.
- `eval_agent_prompts/*.json`: repair prompts emitted during Phase 1/2
  evaluation.
- `eval_agent_traces/*.json`: evaluation-side tool observations, repair events,
  and prompt paths.
- `results/summary.json`: includes `agentic_eval.repaired_count` and repaired
  file names.

## Notes

- This is a bounded per-kernel repair system, not one long-running agent for the
  entire benchmark.
- The full 166-item benchmark can still run with coordinator-level concurrency.
  Extra repair attempts increase LLM calls and, for evaluation repairs, GPU time.
- `--skip-efficiency` is useful for timing/accuracy smoke tests because it avoids
  Phase 3 performance benchmarking.
- Generation prints a heartbeat every 30 seconds while waiting for NVIDIA
  requests. The default request timeout is 600 seconds per request, with 3
  retries.
