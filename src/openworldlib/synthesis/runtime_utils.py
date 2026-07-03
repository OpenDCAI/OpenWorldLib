from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Union


class MethodRuntimeError(RuntimeError):
    pass


def ensure_path(path: Union[str, os.PathLike], name: str, expect_file: Optional[bool] = None) -> Path:
    resolved = Path(path).expanduser().resolve()
    if expect_file is True and not resolved.is_file():
        raise FileNotFoundError(f"{name} file not found: {resolved}")
    if expect_file is False and not resolved.is_dir():
        raise FileNotFoundError(f"{name} directory not found: {resolved}")
    if expect_file is None and not resolved.exists():
        raise FileNotFoundError(f"{name} path not found: {resolved}")
    return resolved


def package_dir(anchor_file: str, *parts: str) -> Path:
    return Path(anchor_file).resolve().parent.joinpath(*parts)


def run_method_command(
    command: Sequence[str],
    cwd: Union[str, os.PathLike],
    env: Optional[Dict[str, str]] = None,
    python_paths: Optional[Iterable[Union[str, os.PathLike]]] = None,
    timeout: Optional[int] = None,
) -> subprocess.CompletedProcess:
    cwd_path = Path(cwd).resolve()
    run_env = os.environ.copy()
    if env:
        run_env.update({k: str(v) for k, v in env.items()})

    path_entries = [str(cwd_path)]
    if python_paths:
        path_entries.extend(str(Path(path).resolve()) for path in python_paths)
    if run_env.get("PYTHONPATH"):
        path_entries.append(run_env["PYTHONPATH"])
    run_env["PYTHONPATH"] = os.pathsep.join(path_entries)

    completed = subprocess.run(
        list(command),
        cwd=str(cwd_path),
        env=run_env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )
    if completed.returncode != 0:
        raise MethodRuntimeError(
            "Method command failed with exit code "
            f"{completed.returncode}:\n{' '.join(command)}\n\n{completed.stdout}"
        )
    return completed


def require_outputs(paths: Iterable[Union[str, os.PathLike]]) -> List[str]:
    resolved = []
    missing = []
    for path in paths:
        candidate = Path(path).expanduser().resolve()
        if candidate.is_file() and candidate.stat().st_size > 0:
            resolved.append(str(candidate))
        else:
            missing.append(str(candidate))
    if missing:
        raise MethodRuntimeError(f"Expected output files were not created: {missing}")
    return resolved
