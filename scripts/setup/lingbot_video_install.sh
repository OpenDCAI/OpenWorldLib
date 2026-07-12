#!/usr/bin/env bash
set -euo pipefail

python -m pip install -U pip
python -m pip install "diffusers>=0.37.1" "transformers>=5.0,<6" "peft>=0.19" \
  "accelerate>=1.8.0" "json_repair>=0.30" "safetensors>=0.4.5" \
  "imageio>=2.35.0" "imageio-ffmpeg>=0.5.1" "decord>=0.6.0" \
  "numpy>=1.26" "pillow>=10.4.0" "requests>=2.31" "scipy>=1.11"

python -m pip install "git+https://github.com/robbyant/lingbot-video.git"

python -m pip install -e .
