#!/bin/bash

# Function to display the help message and available methods
show_help() {
    echo "Usage: bash scripts/test_inference/test_nav_video_gen.sh [method_name]"
    echo ""
    echo "Available methods:"
    echo "  - matrix-game-2        : Run test_matrix_game_2.py"
    echo "  - hunyuan-gamecraft    : Run test_hunyuan_gamecraft.py"
    echo "  - hunyuanworld-voyager : Run test_hunyuan_world_voyager.py"
    echo "  - astra                : Run test_astra.py"
    echo "  - yume                 : Run test_yume.py"
    echo "  - lingbot-world        : Run test_lingbot_world.py"
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
        CUDA_VISIBLE_DEVICES=0 python test/test_matrix_game_2.py
        ;;
    "hunyuan-gamecraft")
        echo "Executing: hunyuan_gamecraft..."
        torchrun --nproc_per_node=1 test/test_hunyuan_gamecraft.py
        ;;
    "hunyuanworld-voyager")
        echo "Executing: hunyuan_world_voyager..."
        CUDA_VISIBLE_DEVICES=0 python test/test_hunyuan_world_voyager.py
        ;;
    "astra")
        echo "Executing: astra..."
        CUDA_VISIBLE_DEVICES=0 python test/test_astra.py
        ;;
    "yume")
        echo "Executing: yume..."
        CUDA_VISIBLE_DEVICES=0 python test/test_yume.py
        ;;
    "lingbot-world")
        echo "Executing: lingbot_world..."
        torchrun --nproc_per_node=8 test/test_lingbot_world.py
        ;;
    *)
        # If the input does not match any method, show an error message
        echo "Error: Unknown method name '$METHOD_NAME'"
        show_help
        exit 1
        ;;
esac
