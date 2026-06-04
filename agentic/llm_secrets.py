"""
NVIDIA API secret resolution: local .env file with fallback to a Modal Secret.
"""

from __future__ import annotations

import os
from pathlib import Path

import modal

from config import FALLBACK_SECRET_NAME, LOCAL_DOTENV_PATH


def _parse_dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()  # handle shell-style `export KEY=value` lines
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
    # The agentic runner lives in a subfolder; keep supporting the repo-root
    # .env used by the baseline runner.
    root_dotenv_path = LOCAL_DOTENV_PATH.parent.parent / ".env"
    if root_dotenv_path != LOCAL_DOTENV_PATH:
        env_values = {**_parse_dotenv(root_dotenv_path), **env_values}
    secret_values: dict[str, str] = {}

    for key in ("NVIDIA_KEY", "NVIDIA_API_KEY"):
        value = os.environ.get(key) or env_values.get(key)
        if value:
            secret_values[key] = value

    if secret_values:
        return modal.Secret.from_dict(secret_values)
    return modal.Secret.from_name(FALLBACK_SECRET_NAME)


NVIDIA_SECRET = _nvidia_secret()
