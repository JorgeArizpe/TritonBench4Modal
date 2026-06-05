"""
Modal App, Volume, and container image definitions.
"""

from __future__ import annotations

import modal

from config import (
    APP_NAME,
    PATCH_CALL_ACC,
    PATCH_EXE_ACC,
    PATCH_PERF,
    REPO_DIR,
    TRITONBENCH_REPO,
    VOLUME_NAME,
    DATA_DIR,
)

# All local helper modules that containers need at import time. Modal only sends
# the decorated function's own file; add_local_python_source embeds these into
# the image so the transitive imports in generation/evaluation/best_versions work.
_LOCAL_MODULES = [
    "config",
    "llm_secrets",
    "modal_app",
    "code_utils",
    "data_utils",
    "perf_utils",
    "formatting",
]

# Generation only calls the LLM API (no GPU needed), so it runs on a cheap CPU image.
# Keeping it separate from the GPU image cuts cold-start time and avoids shipping
# torch/triton to containers that don't benchmark anything.
cpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install("requests>=2.32", "xgrammar>=0.1.14")
    .run_commands(f"git clone --depth 1 {TRITONBENCH_REPO} {REPO_DIR}")
    .add_local_python_source(*_LOCAL_MODULES)
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
    .add_local_python_source(*_LOCAL_MODULES)
)

app = modal.App(APP_NAME, image=image)
data_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
