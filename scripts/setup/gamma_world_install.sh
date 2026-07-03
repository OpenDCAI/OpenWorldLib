#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
METHOD_SRC="$PROJECT_ROOT/src/openworldlib/synthesis/visual_generation/gamma_world"
ENV_PATH=${ENV_PATH:-"$PROJECT_ROOT/.conda-envs/gamma-world"}
PKG_DIR=${CONDA_PKGS_DIRS:-"$PROJECT_ROOT/.conda-pkgs/gamma-world"}

if [ -n "${BASE_CONDA:-}" ]; then
  source "$BASE_CONDA"
fi
mkdir -p "$PKG_DIR"
export CONDA_PKGS_DIRS="$PKG_DIR"

if [ ! -x "$ENV_PATH/bin/python" ]; then
  conda create -p "$ENV_PATH" python=3.10 -y
fi
conda run -p "$ENV_PATH" python -m pip install --upgrade pip
conda run -p "$ENV_PATH" python -m pip install -e "$METHOD_SRC/packages/cosmos-cuda"
conda run -p "$ENV_PATH" python -m pip install torch==2.7.0 torchvision --index-url https://download.pytorch.org/whl/cu128
conda run -p "$ENV_PATH" python -m pip install cmake ninja
conda run -p "$ENV_PATH" python -m pip install natten==0.21.0 --no-build-isolation
conda run -p "$ENV_PATH" python -m pip install transformer-engine==2.2.0 transformer_engine_cu12==2.2.0 --no-deps
conda run -p "$ENV_PATH" python -m pip install transformer-engine-torch==2.2.0 --no-build-isolation --no-deps
conda run -p "$ENV_PATH" python -m pip install xformers==0.0.30 --no-deps
conda run -p "$ENV_PATH" python -m pip install decord
conda run -p "$ENV_PATH" env FLASH_ATTENTION_FORCE_BUILD=TRUE python -m pip install flash-attn==2.7.3 --no-build-isolation --no-binary flash-attn --no-cache-dir --no-deps
conda run -p "$ENV_PATH" python -m pip install -e "$METHOD_SRC/packages/cosmos-oss"
conda run -p "$ENV_PATH" python -m pip install -e "$METHOD_SRC" --no-deps
conda run -p "$ENV_PATH" python -m pip install -e "$PROJECT_ROOT" --no-deps
