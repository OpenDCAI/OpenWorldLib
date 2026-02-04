#!/usr/bin/env bash
set -euo pipefail

# ===== download URL =====
URL="http://s3-us-west-2.amazonaws.com/ai2-thor-public/builds/thor-Linux64-f0825767cd50d69f666c7f282e54abfe58f1e917.zip"
ZIP_NAME="thor-Linux64.zip"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "[INFO] Working dir: $SCRIPT_DIR"

# ===== download (到脚本所在目录) =====
echo "[INFO] Downloading AI2-THOR Unity build..."
wget -O "$ZIP_NAME" "$URL"

# ===== unzip (直接解压到当前目录) =====
echo "[INFO] Extracting into: $SCRIPT_DIR"
unzip -q -o "$ZIP_NAME"

echo "[DONE] AI2-THOR Unity build extracted."
echo "[INFO] Top-level folders:"
ls -1 | sed 's/^/  - /'
echo "[INFO] Executable candidates:"
ls -1 thor-Linux64-*/thor-Linux64-* 2>/dev/null || true
