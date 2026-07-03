#!/bin/bash
# scripts/setup/ctrl_world_install.sh
# Usage: ENV_PATH=/path/to/env bash scripts/setup/ctrl_world_install.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
METHOD_SRC="$PROJECT_ROOT/src/openworldlib/synthesis/vla_generation/ctrl_world"
ENV_PATH=${ENV_PATH:-"$PROJECT_ROOT/.conda-envs/ctrl-world"}
PKG_DIR=${CONDA_PKGS_DIRS:-"$PROJECT_ROOT/.conda-pkgs/ctrl-world"}

if [ -n "${BASE_CONDA:-}" ]; then
  source "$BASE_CONDA"
fi
mkdir -p "$PKG_DIR"
export CONDA_PKGS_DIRS="$PKG_DIR"

if [ ! -x "$ENV_PATH/bin/python" ]; then
  conda create -p "$ENV_PATH" python=3.10 -y
fi

echo "=== [1/2] Installing Ctrl-World upstream requirements ==="
conda run -p "$ENV_PATH" python -m pip install --upgrade pip
conda run -p "$ENV_PATH" python -m pip install -r "$METHOD_SRC/requirements.txt"
conda run -p "$ENV_PATH" python -m pip install imageio-ffmpeg
FFMPEG_BIN=$(conda run -p "$ENV_PATH" python - <<'PY'
import imageio_ffmpeg
print(imageio_ffmpeg.get_ffmpeg_exe())
PY
)
ln -sf "$FFMPEG_BIN" "$ENV_PATH/bin/ffmpeg"

echo "=== [2/2] Installing OpenWorldLib ==="
conda run -p "$ENV_PATH" python -m pip install -e "$PROJECT_ROOT" --no-deps

echo "=== Setup completed! ==="
