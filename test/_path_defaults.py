from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]


def _root(env_name: str, default_dir: str) -> Path:
    return Path(os.environ.get(env_name, REPO_ROOT / default_dir)).expanduser()


def env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def env_optional_int(name: str) -> Optional[int]:
    value = os.environ.get(name)
    return int(value) if value else None


def env_csv(name: str) -> Optional[List[str]]:
    values = [item.strip() for item in os.environ.get(name, "").split(",") if item.strip()]
    return values or None


def python_bin(env_name: str) -> str:
    return os.environ.get(env_name, sys.executable)


def model_path(env_name: str, relative_path: str, fallback: Optional[str] = None) -> str:
    value = os.environ.get(env_name)
    if value:
        return value
    path = _root("OPENWORLDLIB_MODEL_DIR", "models") / relative_path
    if fallback is not None and not path.exists():
        return fallback
    return str(path)


def optional_model_path(env_name: str, relative_path: str) -> Optional[str]:
    value = os.environ.get(env_name)
    if value:
        return value
    path = _root("OPENWORLDLIB_MODEL_DIR", "models") / relative_path
    return str(path) if path.exists() else None


def dataset_path(env_name: str, relative_path: str) -> str:
    value = os.environ.get(env_name)
    if value:
        return value
    return str(_root("OPENWORLDLIB_DATASET_DIR", "datasets") / relative_path)


def test_case_path(*parts: str) -> str:
    return str(REPO_ROOT / "data" / "test_case" / Path(*parts))
