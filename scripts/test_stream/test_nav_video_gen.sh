#!/bin/bash

# Function to display the help message and available methods
show_help() {
    echo "Usage: bash scripts/test_stream/test_nav_video_gen.sh [method_name]"
    echo ""
    echo "Available methods:"
    echo "  - matrix-game-2        : Run test_matrix_game_2.py"
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
    "matrix-game-2")
        echo "Executing: matrix_game_2..."
        CUDA_VISIBLE_DEVICES=0 python test_stream/test_matrix_game_2_stream.py
        ;;
    *)
        # If the input does not match any method, show an error message
        echo "Error: Unknown method name '$METHOD_NAME'"
        show_help
        exit 1
        ;;
esac
