#!/usr/bin/env bash
set -euo pipefail

OS_TYPE="${1:-linux}"   # linux | windows
PACKAGE_TYPE="${2:-base}"  # base | additional

if [[ "$OS_TYPE" != "linux" && "$OS_TYPE" != "windows" ]]; then
    echo "Usage: $0 [linux|windows] [base|additional]"
    exit 1
fi

if [[ "$PACKAGE_TYPE" != "base" && "$PACKAGE_TYPE" != "additional" ]]; then
    echo "Usage: $0 [linux|windows] [base|additional]"
    exit 1
fi

# Base UE server download links
BASE_LINUX_URL="https://huggingface.co/datasets/SimWorld-AI/SimWorld/resolve/main/Base20260201/Linux.zip"
BASE_WINDOWS_URL="https://huggingface.co/datasets/SimWorld-AI/SimWorld/resolve/main/Base20260201/Windows.zip"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

TARGET_DIR="$PROJECT_ROOT/submodules/simworld"

echo "[INFO] Project root: $PROJECT_ROOT"
echo "[INFO] Target dir: $TARGET_DIR"

mkdir -p "$TARGET_DIR"
cd "$TARGET_DIR"

if [[ "$PACKAGE_TYPE" == "base" ]]; then
    if [[ "$OS_TYPE" == "linux" ]]; then
        URL="$BASE_LINUX_URL"
        ZIP_NAME="SimWorld-Base-Linux.zip"
    elif [[ "$OS_TYPE" == "windows" ]]; then
        URL="$BASE_WINDOWS_URL"
        ZIP_NAME="SimWorld-Base-Windows.zip"
    else
        echo "[ERROR] Unsupported OS: $OS_TYPE"
        exit 1
    fi
else
    echo "[ERROR] Additional env auto-download not implemented yet."
    exit 1
fi

if [[ -f "$ZIP_NAME" ]]; then
    echo "[INFO] $ZIP_NAME exists, skipping download."
else
    echo "[INFO] Downloading SimWorld $PACKAGE_TYPE package..."
    wget -O "$ZIP_NAME" "$URL"
fi

echo "[INFO] Extracting..."
unzip -q -o "$ZIP_NAME"

echo "[INFO] Cleaning zip..."
rm -f "$ZIP_NAME"

echo ""
echo "[SUCCESS] SimWorld installed at:"
echo "  $TARGET_DIR"

echo ""
echo "[INFO] To start UE server (Linux):"
echo "  cd $TARGET_DIR"
echo "  ./SimWorld.sh /Game/Maps/demo_1"
