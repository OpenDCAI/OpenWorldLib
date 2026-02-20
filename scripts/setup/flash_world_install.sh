#!/bin/bash
# scripts/setup/flash_world_install.sh
# Description: Setup environment for flash world installation of SceneFlow
# Usage: bash scripts/setup/flash_world_install.sh

echo "=== [1/4] Installing the base environment ==="
pip install torch==2.5.1 torchvision torchaudio
pip install git+https://github.com/openai/CLIP.git

echo "=== [2/4] Installing the requirements ==="
pip install -e ".[3d_ply_default]"

echo "=== [3/4] Installing additional dependencies ==="
pip install git+https://github.com/nerfstudio-project/gsplat.git@32f2a54d21c7ecb135320bb02b136b7407ae5712 --no-build-isolation
pip install diffusers==0.36.0
pip install git+https://github.com/nianticlabs/spz.git@a4fc69e7948c7152e807e6501d73ddc9c149ce37

echo "=== [4/4] Installing the flash attention ==="
pip install "flash-attn==2.5.9.post1" --no-build-isolation

echo "=== Setup completed! ==="
