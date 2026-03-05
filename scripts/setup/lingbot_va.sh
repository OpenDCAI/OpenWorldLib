#!/bin/bash
# scripts/setup/default_lingbot_va.sh
# Description: Setup environment for LingBot-VA
# Usage: bash scripts/setup/default_lingbot_va.sh

echo "=== [1/3] Installing the base environment ==="
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu126

echo "=== [2/3] Installing the requirements ==="
pip install websockets einops diffusers==0.36.0 transformers==4.55.2 accelerate msgpack opencv-python matplotlib ftfy easydict

echo "=== [3/3] Installing the flash attention ==="
pip install flash-attn --no-build-isolation

echo "=== Setup completed! ==="