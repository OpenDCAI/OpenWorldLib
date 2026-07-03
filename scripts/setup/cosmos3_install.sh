#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_PATH=${ENV_PATH:-"$PROJECT_ROOT/.conda-envs/cosmos3"}
PKG_DIR=${CONDA_PKGS_DIRS:-"$PROJECT_ROOT/.conda-pkgs/cosmos3"}

if [ -n "${BASE_CONDA:-}" ]; then
  source "$BASE_CONDA"
fi
mkdir -p "$PKG_DIR"
export CONDA_PKGS_DIRS="$PKG_DIR"

if [ ! -x "$ENV_PATH/bin/python" ]; then
  conda create -p "$ENV_PATH" python=3.11 -y
fi
conda run -p "$ENV_PATH" python -m pip install --upgrade pip
conda run -p "$ENV_PATH" python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
conda run -p "$ENV_PATH" python -m pip install \
  "diffusers @ git+https://github.com/huggingface/diffusers.git" \
  accelerate av cosmos_guardrail huggingface_hub imageio imageio-ffmpeg \
  transformers
conda run -p "$ENV_PATH" python -m pip install -e "$PROJECT_ROOT" --no-deps
