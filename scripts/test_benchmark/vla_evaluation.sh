#!/bin/bash
# VLA Evaluation Script for LIBERO Benchmark
# This script runs the full pipeline: Action Generation + Environment Evaluation

# Full pipeline for VLA benchmark: Generation + Evaluation
CUDA_VISIBLE_DEVICES=0 \
python -m examples.run_benchmark \
    --task_type vla_evaluation \
    --benchmark_name vla_libero_test \
    --data_path ./data/benchmarks/vla/vla_libero_test \
    --model_type spirit-v1p5 \
    --model_path '{"pretrained_model_path": "Spirit-AI-robotics/Spirit-v1.5"}' \
    --norm_stats_path ./data/test_vla/norm_stats.json \
    --output_dir ./benchmark_results/vla_libero \
    --num_samples 50 \
    --run_eval

# # Use string as model_path (simplified version)
# CUDA_VISIBLE_DEVICES=0 \
# python -m examples.run_benchmark \
#     --task_type vla_eval \
#     --benchmark_name vla_libero_test \
#     --data_path ./data/benchmarks/vla/vla_libero_test \
#     --model_type spirit-v1p5 \
#     --model_path Spirit-AI-robotics/Spirit-v1.5 \
#     --norm_stats_path ./data/test_vla/norm_stats.json \
#     --output_dir ./benchmark_results/vla_libero \
#     --num_samples 5 \
#     --run_eval

# # Generate only (skip evaluation)
# # This will only generate action sequences without running in LIBERO environment
# python -m examples.run_benchmark \
#     --task_type vla_eval \
#     --benchmark_name vla_libero_test \
#     --data_path ./data/benchmarks/vla/vla_libero_test \
#     --model_type spirit-v1p5 \
#     --model_path Spirit-AI-robotics/Spirit-v1.5 \
#     --norm_stats_path ./data/test_vla/norm_stats.json \
#     --output_dir ./benchmark_results/vla_libero \
#     --num_samples 5

# # Evaluate only (skip generation)
# # This will load existing action sequences and evaluate them in LIBERO environment
# python -m examples.run_benchmark \
#     --task_type vla_eval \
#     --benchmark_name vla_libero_test \
#     --data_path ./data/benchmarks/vla/vla_libero_test \
#     --results_dir ./benchmark_results/vla_libero \
#     --run_eval

# # Advanced: Custom norm_stats_path for different robot configurations
# # CUDA_VISIBLE_DEVICES=0 \
# # python -m examples.run_benchmark \
# #     --task_type vla_eval \
# #     --benchmark_name vla_libero_test \
# #     --data_path ./data/benchmarks/vla/vla_libero_test \
# #     --model_type spirit-v1p5 \
# #     --model_path Spirit-AI-robotics/Spirit-v1.5 \
# #     --norm_stats_path /path/to/custom/norm_stats.json \
# #     --output_dir ./benchmark_results/vla_libero_custom \
# #     --num_samples 10 \
# #     --run_eval
