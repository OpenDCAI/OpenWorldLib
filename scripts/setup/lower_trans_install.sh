#!/bin/bash
# scripts/setup/lower_trans_install.sh
# Description: Setup environment for lower transformers installation of SceneFlow
# Usage: bash scripts/setup/lower_trans_install.sh


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ENV_NAME="${1:-sceneflow}"

echo "=== [1/4] Creating conda environment: $ENV_NAME ==="
conda create -n "$ENV_NAME" python=3.9 -y
conda activate "$ENV_NAME"

echo "=== [2/4] Installing the pytorch ==="
pip install torch==2.5.1 torchvision torchaudio

echo "=== [3/4] Installing the requirements ==="
pip install -e ".[transformers_low]"

echo "=== [4/4] Installing the flash attention ==="
pip install "flash-attn==2.5.9.post1" --no-build-isolation

echo "=== Setup completed! ==="
