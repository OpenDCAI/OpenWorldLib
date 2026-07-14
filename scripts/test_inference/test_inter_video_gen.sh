#!/bin/bash

# Function to display the help message and available methods
show_help() {
    echo "Usage: bash scripts/test_inference/test_inter_video_gen.sh [method_name]"
    echo ""
    echo "Available methods:"
    echo "  - wan2.2                 : Run test_wan_2p2.py"
    echo "  - wow                    : Run test_wow.py"
    echo "  - cosmos-predict2.5      : Run test_cosmos_predict2p5.py"
    echo "  - recammaster            : Run test_recammaster.py"
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
    "wan2.2")
        echo "Executing: wan2.2..."
        CUDA_VISIBLE_DEVICES=0 python test/test_wan_2p2.py
        ;;
    "wow")
        echo "Executing: wow..."
        CUDA_VISIBLE_DEVICES=0 python test/test_wow.py
        ;;
    "cosmos-predict2.5")
        echo "Executing: cosmos-predict2.5..."
        CUDA_VISIBLE_DEVICES=0 python test/test_cosmos_predict2p5.py
        ;;
    "recammaster")
        echo "Executing: recammaster..."
        CUDA_VISIBLE_DEVICES=0 python test/test_recammaster.py
        ;;
    *)
        # If the input does not match any method, show an error message
        echo "Error: Unknown method name '$METHOD_NAME'"
        show_help
        exit 1
        ;;
esac
