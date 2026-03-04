# Full pipeline for benchmark: Generation + Evaluation
CUDA_VISIBLE_DEVICES=0,1 \
MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 \
python -m examples.run_benchmark \
    --task_type imagetext2video_gen\
    --benchmark_name it2vgen_test \
    --data_path ./data/benchmarks/generation/imagetext2video_generation/it2vgen_test \
    --model_type wan2p2 \
    --model_path '{"pretrained_model_path":"Wan2.2/Wan2.2-TI2V-5B"}' \
    --eval_model_type qwen2p5-omni \
    --eval_model_path '{"pretrained_model_path": "Qwen/Qwen2.5-Omni-7B-Instruct"}' \
    --output_dir ./benchmark_results \
    --num_samples 1 \
    --run_eval

# # utilize the string serving as the model_path
# CUDA_VISIBLE_DEVICES=0, 1 \
# MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 \
# python -m examples.run_benchmark \
#     --task_type imagetext2video_gen \
#     --benchmark_name sf_nav_vidgen_test \
#     --data_path ./data/benchmarks/generation/imagetext2video_generation/it2vgen_test \
#     --model_type wan2p2 \
#     --model_path '{"pretrained_model_path":"Wan2.2/Wan2.2-TI2V-5B"}' \
#     --eval_model_type qwen2p5-omni \
#     --eval_model_path Qwen/Qwen2.5-Omni-7B-Instruct \
#     --output_dir ./benchmark_results \
#     --num_samples 1 \
#     --run_eval

# # Generate only (skip evaluation)
# python -m examples.run_benchmark \
#     --task_type imagetext2video_gen \
#     --benchmark_name it2vgen_test \
#     --data_path ./data/benchmarks/generation/imagetext2video_generation/it2vgen_test \
#     --model_type wan2p2 \
#     --model_path '{"pretrained_model_path":"Wan2.2/Wan2.2-TI2V-5B"}' \
#     --eval_model_type qwen2p5-omni \
#     --output_dir ./benchmark_results \
#     --num_samples 1

# # Evaluate only (skip generation)
# python -m examples.run_benchmark \
#     --task_type imagetext2video_gen \
#     --benchmark_name it2vgen_test \
#     --data_path ./data/benchmarks/generation/imagetext2video_generation/it2vgen_test \
#     --eval_model_type qwen2p5-omni \
#     --eval_model_path '{"pretrained_model_path": "Qwen/Qwen2.5-Omni-7B-Instruct"}' \
#     --results_dir ./benchmark_results \
#     --run_eval
