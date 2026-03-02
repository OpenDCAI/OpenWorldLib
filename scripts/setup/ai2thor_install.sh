#!/usr/bin/env bash
# scripts/setup/install_ai2thor.sh
# Description: Install AI2-THOR Unity build into submodules/thor
# Usage: bash scripts/setup/install_ai2thor.sh

set -euo pipefail

echo "=== [1/3] Locating project root and preparing directory ==="

# ===== download URL =====
URL="http://s3-us-west-2.amazonaws.com/ai2-thor-public/builds/thor-Linux64-f0825767cd50d69f666c7f282e54abfe58f1e917.zip"
ZIP_NAME="thor-Linux64.zip"

# ===== 定位项目根目录 =====
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# ===== 目标目录 =====
TARGET_DIR="$PROJECT_ROOT/submodules/thor"

echo "[INFO] Project root: $PROJECT_ROOT"
echo "[INFO] Target directory: $TARGET_DIR"

mkdir -p "$TARGET_DIR"
cd "$TARGET_DIR"

echo ""
echo "=== [2/3] Downloading and extracting AI2-THOR build ==="

if [[ -f "$ZIP_NAME" ]]; then
    echo "[INFO] $ZIP_NAME already exists, skipping download..."
else
    echo "[INFO] Downloading AI2-THOR Unity build..."
    wget -O "$ZIP_NAME" "$URL"
fi

echo "[INFO] Extracting..."
unzip -q -o "$ZIP_NAME"

echo ""
echo "=== [3/3] Finalizing installation ==="

echo "[INFO] Installed contents:"
ls -1 | sed 's/^/  - /'

echo ""
echo "[INFO] Executable candidates:"
ls -1 thor-Linux64-*/thor-Linux64-* 2>/dev/null || echo "  (no executable found yet)"

echo "[INFO] Cleaning up zip file..."
rm -f "$ZIP_NAME"

echo ""
echo "=== AI2-THOR installation completed ==="
echo ""
echo "[INFO] Use this path in your code:"
echo "  executable_path = '$TARGET_DIR/thor-Linux64-local/thor-Linux64-local'"