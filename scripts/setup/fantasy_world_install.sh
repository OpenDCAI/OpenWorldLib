#!/bin/bash
# scripts/setup/fantasy_world_install.sh
# Usage: ENV_PATH=/path/to/env bash scripts/setup/fantasy_world_install.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
METHOD_SRC="$PROJECT_ROOT/src/openworldlib/synthesis/visual_generation/fantasy_world"
ENV_PATH=${ENV_PATH:-"$PROJECT_ROOT/.conda-envs/fantasy-world"}
PKG_DIR=${CONDA_PKGS_DIRS:-"$PROJECT_ROOT/.conda-pkgs/fantasy-world"}

if [ -n "${BASE_CONDA:-}" ]; then
  source "$BASE_CONDA"
fi
mkdir -p "$PKG_DIR"
export CONDA_PKGS_DIRS="$PKG_DIR"

if [ ! -x "$ENV_PATH/bin/python" ]; then
  conda create -p "$ENV_PATH" python=3.10 -y
fi

echo "=== [1/3] Installing FantasyWorld upstream requirements ==="
conda run -p "$ENV_PATH" python -m pip install --upgrade pip
conda run -p "$ENV_PATH" python -m pip install -r "$METHOD_SRC/requirements.txt"
conda run -p "$ENV_PATH" python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

echo "=== [2/3] Installing FantasyWorld utility packages ==="
# MoGe is provided in src/openworldlib/base_models/three_dimensions/depth/moge.
conda run -p "$ENV_PATH" python -m pip install "numpy<2,>=1.23.5"

echo "=== [3/3] Installing OpenWorldLib ==="
conda run -p "$ENV_PATH" python -m pip install -e "$PROJECT_ROOT" --no-deps

echo "=== Setup completed! ==="
