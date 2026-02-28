#!/bin/bash

# Function to display the help message and available methods
show_help() {
    echo "Usage: bash scripts/test_inference/test_3d_scene_gen.sh [method_name]"
    echo ""
    echo "Available methods:"
    echo "  - vggt                 : Run test_vggt.py"
    echo "  - infinite-vggt        : Run test_infinite_vggt.py"
    echo "  - flash-world          : Run test_flash_world.py"
    echo ""
}

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
        CUDA_VISIBLE_DEVICES=0 python test/test_vggt.py
        ;;
    "infinite-vggt")
        echo "Executing: infinite_vggt..."
        CUDA_VISIBLE_DEVICES=0 python test/test_infinite_vggt.py
        ;;
    "flash-world")
        echo "Executing: flash_world..."
        CUDA_VISIBLE_DEVICES=0 python test/test_flash_world.py
        ;;
    *)
        # If the input does not match any method, show an error message
        echo "Error: Unknown method name '$METHOD_NAME'"
        show_help
        exit 1
        ;;
esac
