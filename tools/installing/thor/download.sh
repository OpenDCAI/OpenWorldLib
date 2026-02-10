#!/usr/bin/env bash
set -euo pipefail

# ===== download URL =====
URL="http://s3-us-west-2.amazonaws.com/ai2-thor-public/builds/thor-Linux64-f0825767cd50d69f666c7f282e54abfe58f1e917.zip"
ZIP_NAME="thor-Linux64.zip"

# ===== 定位项目根目录 =====
# 脚本在 tools/install/thor/ 目录，项目根目录是上三级
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")"

# ===== 目标目录 =====
TARGET_DIR="$PROJECT_ROOT/submodules/thor"

echo "[INFO] Project root: $PROJECT_ROOT"
echo "[INFO] Target directory: $TARGET_DIR"

# ===== 创建目标目录 =====
mkdir -p "$TARGET_DIR"
cd "$TARGET_DIR"

echo "[INFO] Working dir: $TARGET_DIR"

# ===== download =====
if [[ -f "$ZIP_NAME" ]]; then
    echo "[INFO] $ZIP_NAME already exists, skipping download..."
else
    echo "[INFO] Downloading AI2-THOR Unity build..."
    wget -O "$ZIP_NAME" "$URL"
fi

# ===== unzip =====
echo "[INFO] Extracting into: $TARGET_DIR"
unzip -q -o "$ZIP_NAME"

echo "[DONE] AI2-THOR Unity build extracted."
echo "[INFO] Contents of $TARGET_DIR:"
ls -1 | sed 's/^/  - /'

echo "[INFO] Executable candidates:"
ls -1 thor-Linux64-*/thor-Linux64-* 2>/dev/null || echo "  (no executable found yet)"

# ===== 可选：清理 zip 文件 =====
echo "[INFO] Cleaning up zip file..."
rm -f "$ZIP_NAME"

echo ""
echo "[SUCCESS] AI2-THOR installed at: $TARGET_DIR"
echo "[INFO] Use this path in your code:"
echo "  executable_path = '$TARGET_DIR/thor-Linux64-local/thor-Linux64-local'"
