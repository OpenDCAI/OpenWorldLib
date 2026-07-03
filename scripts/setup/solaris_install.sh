#!/bin/bash
# scripts/setup/solaris_install.sh
# Usage: ENV_PATH=/path/to/env bash scripts/setup/solaris_install.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
METHOD_SRC="$PROJECT_ROOT/src/openworldlib/synthesis/visual_generation/solaris"
ENV_PATH=${ENV_PATH:-"$PROJECT_ROOT/.conda-envs/solaris"}
PKG_DIR=${CONDA_PKGS_DIRS:-"$PROJECT_ROOT/.conda-pkgs/solaris"}

if [ -n "${BASE_CONDA:-}" ]; then
  source "$BASE_CONDA"
fi
mkdir -p "$PKG_DIR"
export CONDA_PKGS_DIRS="$PKG_DIR"

if [ ! -x "$ENV_PATH/bin/python" ]; then
  conda create -p "$ENV_PATH" python=3.10 -y
fi

echo "=== [1/3] Installing Solaris upstream requirements ==="
conda run -p "$ENV_PATH" python -m pip install --upgrade pip
conda run -p "$ENV_PATH" python -m pip install -r "$METHOD_SRC/requirements_gpu.txt"

echo "=== [3/3] Installing OpenWorldLib ==="
conda run -p "$ENV_PATH" python -m pip install -e "$PROJECT_ROOT" --no-deps

echo "=== Setup completed! ==="
