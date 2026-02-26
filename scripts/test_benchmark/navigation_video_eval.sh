# Full pipeline for benchmark: Generation + Evaluation
CUDA_VISIBLE_DEVICES=0 \
MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 \
python -m examples.run_benchmark \
    --task_type navigation_video_gen \
    --benchmark_name sf_nav_vidgen_test \
    --data_path ./data/benchmarks/generation/navigation_video_generation/sf_nav_vidgen_test \
    --model_type hunyuan-game-craft \
    --eval_model_type qwen2p5omni \
    --model_path tencent/Hunyuan-GameCraft-1.0 \
    --output_dir ./benchmark_results \
    --num_samples 2 \
    --run_eval \
    --eval_model_path Qwen/Qwen2.5-Omni-7B-Instruct

# # Generate only (skip evaluation)
# python -m examples.run_benchmark
#     --task_type navigation_video_gen
#     --benchmark_name sf_nav_vidgen_test
#     --data_path ./data/benchmarks/generation/navigation_video_generation/sf_nav_vidgen_test
#     --model_type matrix-game2
#     --eval_model_type qwen2p5omni
#     --model_path Skywork/Matrix-Game-2.0
#     --output_dir ./benchmark_results
#     --num_samples 2

# # Evaluate only (skip generation)
# python -m examples.run_benchmark
#     --task_type navigation_video_gen
#     --benchmark_name sf_nav_vidgen_test
#     --data_path ./data/benchmarks/generation/navigation_video_generation/sf_nav_vidgen_test
#     --eval_model_type omnivinci
#     --results_dir ./benchmark_results
#     --run_eval
#     --eval_model_path nvidia/omnivinci
