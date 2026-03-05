#!/bin/bash

# ============================================================
# VLA (Vision-Language-Action) Inference Test Script
# ============================================================
#
# Usage:
#   bash scripts/test_inference/test_vla.sh [method_name] [options...]
#
# Examples:
#   bash scripts/test_inference/test_vla.sh pi0
#   bash scripts/test_inference/test_vla.sh pi0 --dataset aloha droid
#   bash scripts/test_inference/test_vla.sh pi05
#   bash scripts/test_inference/test_vla.sh pi05 --dataset libero
#   bash scripts/test_inference/test_vla.sh pi0-all
#   bash scripts/test_inference/test_vla.sh lingbot-va
#   bash scripts/test_inference/test_vla.sh giga-brain-0
#   bash scripts/test_inference/test_vla.sh all

# Function to display the help message and available methods
show_help() {
    echo "Usage: bash scripts/test_inference/test_vla.sh [method_name] [options...]"
    echo ""
    echo "Available methods:"
    echo "  pi0                 : Run PI0 tests on all datasets (aloha, libero, droid)"
    echo "  pi05                : Run PI0.5 tests on all datasets (libero, droid)"
    echo "  pi0-all             : Run both PI0 and PI0.5 tests on all datasets"
    echo "  lingbot-va          : Run LingBot-VA test"
    echo "  giga-brain-0        : Run GigaBrain-0 test"
    echo "  all                 : Run all VLA tests"
    echo ""
    echo "PI0/PI0.5 extra options (passed through to test_pi0.py):"
    echo "  --dataset <name...> : Specify dataset(s): aloha, libero, droid"
    echo ""
    echo "Examples:"
    echo "  bash scripts/test_inference/test_vla.sh pi0"
    echo "  bash scripts/test_inference/test_vla.sh pi0 --dataset aloha droid"
    echo "  bash scripts/test_inference/test_vla.sh pi05 --dataset libero"
    echo "  bash scripts/test_inference/test_vla.sh pi0-all --dataset droid"
    echo "  bash scripts/test_inference/test_vla.sh all"
    echo ""
}

# Check if an argument is provided
if [ -z "$1" ]; then
    echo "Error: Please provide a method name to execute."
    show_help
    exit 1
fi

METHOD_NAME=$1
shift  # Remove method name, remaining args are extra options

# Execute the corresponding command based on the input method name
case $METHOD_NAME in
    "pi0")
        echo "============================================================"
        echo "  Executing: PI0 tests"
        echo "============================================================"
        CUDA_VISIBLE_DEVICES=0 python test/test_pi0.py --model pi0 "$@"
        ;;
    "pi05")
        echo "============================================================"
        echo "  Executing: PI0.5 tests"
        echo "============================================================"
        CUDA_VISIBLE_DEVICES=0 python test/test_pi0.py --model pi05 "$@"
        ;;
    "pi0-all")
        echo "============================================================"
        echo "  Executing: PI0 + PI0.5 tests (all)"
        echo "============================================================"
        CUDA_VISIBLE_DEVICES=0 python test/test_pi0.py --model all "$@"
        ;;
    "lingbot-va")
        echo "============================================================"
        echo "  Executing: LingBot-VA test"
        echo "============================================================"
        CUDA_VISIBLE_DEVICES=0 python test/test_lingbot_va.py
        ;;
    "giga-brain-0")
        echo "============================================================"
        echo "  Executing: GigaBrain-0 test"
        echo "============================================================"
        CUDA_VISIBLE_DEVICES=0 python test/test_giga_brain_0.py
        ;;
    "all")
        echo "============================================================"
        echo "  Executing: ALL VLA tests"
        echo "============================================================"
        echo ""

        echo "[1/3] PI0 + PI0.5 tests..."
        CUDA_VISIBLE_DEVICES=0 python test/test_pi0.py --model all
        PI0_EXIT=$?

        echo ""
        echo "[2/3] LingBot-VA test..."
        CUDA_VISIBLE_DEVICES=0 python test/test_lingbot_va.py
        LINGBOT_EXIT=$?

        echo ""
        echo "[3/3] GigaBrain-0 test..."
        CUDA_VISIBLE_DEVICES=0 python test/test_giga_brain_0.py
        GIGA_EXIT=$?

        echo ""
        echo "============================================================"
        echo "  Summary"
        echo "============================================================"
        echo "  PI0/PI0.5:   $([ $PI0_EXIT -eq 0 ] && echo 'PASS' || echo 'FAIL')"
        echo "  LingBot-VA:  $([ $LINGBOT_EXIT -eq 0 ] && echo 'PASS' || echo 'FAIL')"
        echo "  GigaBrain-0: $([ $GIGA_EXIT -eq 0 ] && echo 'PASS' || echo 'FAIL')"
        echo "============================================================"

        # Exit with failure if any test failed
        if [ $PI0_EXIT -ne 0 ] || [ $LINGBOT_EXIT -ne 0 ] || [ $GIGA_EXIT -ne 0 ]; then
            exit 1
        fi
        ;;
    *)
        echo "Error: Unknown method name '$METHOD_NAME'"
        show_help
        exit 1
        ;;
esac
