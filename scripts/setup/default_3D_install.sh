#!/bin/bash
# scripts/setup/default_3D_install.sh
# Description: Setup environment for default 3D installation of SceneFlow
# Usage: bash scripts/setup/default_3D_install.sh

echo "=== [1/3] Installing the base environment ==="
pip install torch==2.5.1 torchvision torchaudio
pip install git+https://github.com/openai/CLIP.git

echo "=== [2/3] Installing the requirements ==="
pip install -e ".[3d_ply_default]"

echo "=== [3/3] Installing the flash attention ==="
pip install "flash-attn==2.5.9.post1" --no-build-isolation

echo "=== Setup completed! ==="
