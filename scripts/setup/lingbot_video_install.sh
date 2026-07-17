#!/usr/bin/env bash
set -euo pipefail

INSTALL_SGLANG=false
if [[ "${1:-}" == "--sglang" ]]; then
  INSTALL_SGLANG=true
elif [[ $# -gt 0 ]]; then
  echo "Usage: $0 [--sglang]" >&2
  exit 2
fi

python -m pip install -U pip
python -m pip install "diffusers>=0.37.1" "transformers>=5.0,<6" "peft>=0.19" \
  "accelerate>=1.8.0" "json_repair>=0.30" "safetensors>=0.4.5" \
  "imageio>=2.35.0" "imageio-ffmpeg>=0.5.1" "decord>=0.6.0" \
  "numpy>=1.26" "pillow>=10.4.0" "requests>=2.31" "scipy>=1.11"

python -m pip install "git+https://github.com/robbyant/lingbot-video.git"

if [[ "$INSTALL_SGLANG" == true ]]; then
  # Keep the LingBot-Video PyTorch and CUDA stack selected above.
  python -m pip install --no-deps \
    "sglang==0.5.13.post1" \
    "apache-tvm-ffi==0.1.9" \
    "tilelang==0.1.8" \
    "sglang-kernel==0.4.4" \
    "triton>=3.6.0"
fi

python -m pip install -e .
