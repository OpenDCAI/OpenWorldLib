#!/bin/bash

# Function to display the help message and available methods
show_help() {
    echo "Usage: bash scripts/test_inference/test_3d_scene_gen.sh [method_name]"
    echo ""
    echo "Available methods:"
    echo "  - vggt                 : Run test_vggt.py"
    echo "  - infinite-vggt        : Run test_infinite_vggt.py"
    echo "  - hunyuan-worldplay2   : Run test_hunyuan_worldplay2.py"
    echo "  - fantasy-world        : Run test_fantasy_world.py"
    echo "  - flash-world          : Run test_flash_world.py"
    echo ""
}

PYTHON_BIN=${PYTHON_BIN:-python}
CUDA_VISIBLE_DEVICES_VALUE=${CUDA_VISIBLE_DEVICES:-0}
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    if command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN=python3
    else
        echo "Error: neither 'python' nor 'python3' is available in PATH."
        exit 1
    fi
fi

# Check if an argument is provided
if [ -z "$1" ]; then
    echo "Error: Please provide a method name to execute."
    show_help
    exit 1
fi

METHOD_NAME=$1

# Execute the corresponding command based on the input method name
case $METHOD_NAME in
    "vggt")
        echo "Executing: vggt..."
        CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" test/test_vggt.py
        ;;
    "infinite-vggt")
        echo "Executing: infinite_vggt..."
        CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" test/test_infinite_vggt.py
        ;;
    "flash-world")
        echo "Executing: flash_world..."
        CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" test/test_flash_world.py
        ;;
    "hunyuan-worldplay2"|"hy-world-2")
        echo "Executing: hunyuan_worldplay2..."
        CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_VALUE" "$PYTHON_BIN" test/test_hunyuan_worldplay2.py
        ;;
    "fantasy-world")
        echo "Executing: fantasy_world..."
        CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_VALUE" "$PYTHON_BIN" test/test_fantasy_world.py
        ;;
    *)
        # If the input does not match any method, show an error message
        echo "Error: Unknown method name '$METHOD_NAME'"
        show_help
        exit 1
        ;;
esac
