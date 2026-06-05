"""
Shared constants and defaults for the NVIDIA iterative TritonBench runner.
"""

from __future__ import annotations

import os
from pathlib import Path

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

# TritonBench eval scripts use hardcoded local paths and python interpreter lookups
# that break inside Modal containers; these patches redirect them to REPO_DIR and sys.executable.
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

# The original script auto-detects GPU count; force 1 so each Modal container only uses its assigned GPU.
PATCH_PERF = (
    f"""sed -i 's|^gpu_count = .*|gpu_count = 1|' """
    f"""{REPO_DIR}/performance_metrics/perf_T/run_bench/multiprocess_gpu_run.py"""
)
