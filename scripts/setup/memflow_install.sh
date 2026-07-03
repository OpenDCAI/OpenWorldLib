#!/bin/bash
# scripts/setup/memflow_install.sh
# Usage: ENV_PATH=/path/to/env bash scripts/setup/memflow_install.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
METHOD_SRC="$PROJECT_ROOT/src/openworldlib/synthesis/visual_generation/memflow"
ENV_PATH=${ENV_PATH:-"$PROJECT_ROOT/.conda-envs/memflow"}
PKG_DIR=${CONDA_PKGS_DIRS:-"$PROJECT_ROOT/.conda-pkgs/memflow"}

if [ -n "${BASE_CONDA:-}" ]; then
  source "$BASE_CONDA"
fi
mkdir -p "$PKG_DIR"
export CONDA_PKGS_DIRS="$PKG_DIR"

if [ ! -x "$ENV_PATH/bin/python" ]; then
  conda create -p "$ENV_PATH" python=3.10 -y
fi

echo "=== [1/3] Installing MemFlow upstream requirements ==="
conda run -p "$ENV_PATH" python -m pip install --upgrade pip
conda run -p "$ENV_PATH" python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
TMP_REQ=$(mktemp)
grep -Ev '^(nvidia-pyindex|nvidia-tensorrt|pycuda)([=<> ].*)?$' "$METHOD_SRC/requirements.txt" > "$TMP_REQ"
conda run -p "$ENV_PATH" python -m pip install -r "$TMP_REQ"
rm -f "$TMP_REQ"
conda run -p "$ENV_PATH" python -m pip install "pyarrow==24.0.0" "tqdm>=4.66.3"

echo "=== [2/3] Installing FlashAttention for MemFlow ==="
conda run -p "$ENV_PATH" env FLASH_ATTENTION_FORCE_BUILD=TRUE python -m pip install flash-attn --no-build-isolation --no-binary flash-attn --no-cache-dir --no-deps

echo "=== [3/3] Installing OpenWorldLib ==="
conda run -p "$ENV_PATH" python -m pip install -e "$PROJECT_ROOT" --no-deps

echo "=== Setup completed! ==="
