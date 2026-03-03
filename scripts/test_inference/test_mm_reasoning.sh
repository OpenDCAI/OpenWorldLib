#!/bin/bash

# Function to display the help message and available methods
show_help() {
    echo "Usage: bash scripts/test_inference/test_mm_reasoning.sh [method_name]"
    echo ""
    echo "Available methods:"
    echo "  - qwen2.5-omni        : Run test_qwen2p5_omni.py"
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
    "qwen2.5-omni")
        echo "Executing: qwen2.5-omni..."
        CUDA_VISIBLE_DEVICES=0 python test/test_qwen2p5_omni.py
        ;;
    *)
        # If the input does not match any method, show an error message
        echo "Error: Unknown method name '$METHOD_NAME'"
        show_help
        exit 1
        ;;
esac
