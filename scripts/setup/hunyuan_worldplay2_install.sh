#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
METHOD_SRC="$PROJECT_ROOT/src/openworldlib/synthesis/visual_generation/hunyuan_world/hunyuan_worldplay2"
SUBMODULES_DIR="$PROJECT_ROOT/submodules"
GSPLAT_DIR="$SUBMODULES_DIR/hunyuan-worldplay2-gsplat-maskgaussian"
NAVMESH_DIR="$SUBMODULES_DIR/hunyuan-worldplay2-navmesh"
RECAST_DIR="${RECAST_DIR:-$SUBMODULES_DIR/hunyuan-worldplay2-recastnavigation}"
ENV_PATH=${ENV_PATH:-"$PROJECT_ROOT/.conda-envs/hunyuan-worldplay2"}
PKG_DIR=${CONDA_PKGS_DIRS:-"$PROJECT_ROOT/.conda-pkgs/hunyuan-worldplay2"}

if [ -n "${BASE_CONDA:-}" ]; then
  source "$BASE_CONDA"
fi
mkdir -p "$PKG_DIR"
export CONDA_PKGS_DIRS="$PKG_DIR"

if [ ! -x "$ENV_PATH/bin/python" ]; then
  conda create -p "$ENV_PATH" python=3.11.15 -y
fi
conda run -p "$ENV_PATH" python -m pip install --upgrade pip
REQ_FILE=$(mktemp)
grep -Ev '^(cupy)([<=>].*)?$' "$METHOD_SRC/requirements.txt" > "$REQ_FILE"
conda run -p "$ENV_PATH" python -m pip install cupy-cuda12x==13.6.0
conda run -p "$ENV_PATH" python -m pip install -r "$REQ_FILE"
rm -f "$REQ_FILE"
conda run -p "$ENV_PATH" python -m pip uninstall -y onnxruntime-gpu || true
conda run -p "$ENV_PATH" python -m pip install onnxruntime==1.23.2
ATTENTION_FILE="$METHOD_SRC/hyworld2/worldrecon/hyworldmirror/models/layers/attention.py"
python - "$ATTENTION_FILE" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text()
old = """try:
    from flash_attn_interface import flash_attn_func as flash_attn_func_v3
    _USE_FLASH_ATTN_V3 = True
except ImportError:
    from flash_attn.flash_attn_interface import flash_attn_func as flash_attn_func_v2
    _USE_FLASH_ATTN_V3 = False
"""
new = """try:
    from flash_attn_interface import flash_attn_func as flash_attn_func_v3
    _USE_FLASH_ATTN_V3 = True
except ImportError:
    try:
        from flash_attn.flash_attn_interface import flash_attn_func as flash_attn_func_v2
        _USE_FLASH_ATTN_V3 = False
    except ImportError:
        flash_attn_func_v2 = None
        _USE_FLASH_ATTN_V3 = None
"""
if old in text:
    text = text.replace(old, new)
text = text.replace(
    "if q.dtype==torch.bfloat16 or q.dtype==torch.float16:",
    "if (q.dtype==torch.bfloat16 or q.dtype==torch.float16) and _USE_FLASH_ATTN_V3 is not None:",
)
path.write_text(text)
PY
if [ -d "$GSPLAT_DIR" ]; then
  GLM_DIR="$GSPLAT_DIR/gsplat/cuda/csrc/third_party/glm"
  if [ ! -f "$GLM_DIR/glm/gtc/type_ptr.hpp" ]; then
    mkdir -p "$(dirname "$GLM_DIR")"
    rm -rf "$GLM_DIR"
    git clone --depth 1 https://github.com/g-truc/glm.git "$GLM_DIR"
  fi
  conda run -p "$ENV_PATH" env MAX_JOBS="${MAX_JOBS:-4}" TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}" python -m pip install "$GSPLAT_DIR" --no-build-isolation --no-deps
else
  conda run -p "$ENV_PATH" env MAX_JOBS="${MAX_JOBS:-4}" TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}" python -m pip install git+https://github.com/nerfstudio-project/gsplat.git --no-build-isolation --no-deps
fi
if [ -d "$NAVMESH_DIR" ]; then
  if [ ! -f "$RECAST_DIR/Recast/Include/Recast.h" ]; then
    rm -rf "$RECAST_DIR"
    git clone --depth 1 --recurse-submodules https://github.com/recastnavigation/recastnavigation.git "$RECAST_DIR"
  fi
  conda run -p "$ENV_PATH" env RECAST_PATH="$RECAST_DIR" python -m pip install "$NAVMESH_DIR" --no-build-isolation --no-deps
fi
conda run -p "$ENV_PATH" python -m pip install -e "$PROJECT_ROOT" --no-deps
