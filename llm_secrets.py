"""
LLM API secret resolution: local .env file with fallback to a Modal Secret.
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


def _llm_secret() -> modal.Secret:
    env_values = _parse_dotenv(LOCAL_DOTENV_PATH)
    value = os.environ.get("DASHSCOPE_API_KEY") or env_values.get("DASHSCOPE_API_KEY")
    if value:
        return modal.Secret.from_dict({"DASHSCOPE_API_KEY": value})
    return modal.Secret.from_name(FALLBACK_SECRET_NAME)


LLM_SECRET = _llm_secret()
