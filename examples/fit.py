"""
SceneFlow Benchmark Runner
Usage:
    # 完整流程：生成 + 评估
    python -m examples.run_benchmark
        --task_type navigation_video_gen
        --benchmark_name sf_nav_vidgen_test
        --data_path ./data/benchmarks/generation/navigation_video_generation/sf_nav_vidgen_test
        --model_type matrix-game2
        --eval_model_type qwen2p5omni
        --model_path Skywork/Matrix-Game-2.0
        --output_dir ./benchmark_results
        --num_samples 2
        --run_eval
        --eval_model_path Qwen/Qwen2.5-Omni-7B-Instruct
    
    # 仅评估已有结果（跳过生成）
    python -m examples.run_benchmark
        --task_type navigation_video_gen
        --benchmark_name sf_nav_vidgen_test
        --data_path ./data/benchmarks/generation/navigation_video_generation/sf_nav_vidgen_test
        --eval_model_type omnivinci
        --results_dir ./benchmark_results
        --run_eval
        --eval_model_path nvidia/omnivinci
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from data.benchmarks.tasks_map import tasks_map
from data.benchmarks.benchmark_loader import BenchmarkLoader
from examples.pipeline_mapping import video_gen_pipe, reasoning_pipe, vla_pipe
from examples.evaluation_tasks.eval_func_mapping import eval_func_mapping


# collect evaluation pipelines
# This loading way is used to verify whether the loaded pipe corresponds to the intended task.
ALL_PIPELINES = {**video_gen_pipe, **reasoning_pipe, **vla_pipe}

def parse_args():
    parser = argparse.ArgumentParser(description="SceneFlow Benchmark Runner")
    parser.add_argument("--task_type", type=str, required=True,
                        help="tasks_map contain various, like navigation_video_gen")
    parser.add_argument("--benchmark_name", type=str, required=True,
                        help="the name of benchmark , such as sf_nav_vidgen_test")
    parser.add_argument("--data_path", type=str, required=True,
                        help="local data file path HuggingFace repo id")
    parser.add_argument("--eval_model_path", type=str, default="Qwen/Qwen2.5-Omni-7B-Instruct",
                        help="evaluation MLLM model path or HuggingFace model id")
    parser.add_argument("--model_type", type=str,
                        help="pipeline_mapping matrix-game2")
    parser.add_argument("--eval_model_type", type=str, default="qwen2p5omni",
                        help="evaluation MLLM model type, like qwen2p5omni")
    parser.add_argument("--model_path", type=str,
                        help="model path or HuggingFace model id")
    parser.add_argument("--norm_stats_path", type=str, default=None,
                        help="normalization statistics path (for VLA models like spirit-v1p5)")
    parser.add_argument("--output_dir", type=str, default="./benchmark_results")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="test N samples, default ")
    parser.add_argument("--run_eval", action="store_true",
                        help="whether to carry out evaluation")
    parser.add_argument("--results_dir", type=str, default=None,
                        help="path to existing results directory (skip generation if provided)")
    return parser.parse_args()


# Pipeline loading here
def load_pipeline(model_type: str, model_path: str, device: str = "cuda", norm_stats_path: str = None):
    """load the pipeline according to the model_type."""
    if model_type not in ALL_PIPELINES:
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            f"Available: {list(ALL_PIPELINES.keys())}"
        )

    PipeClass = ALL_PIPELINES[model_type]
    
    # VLA pipelines (like spirit-v1p5) need norm_stats_path parameter
    if model_type in vla_pipe:
        return PipeClass(model_path, device, norm_stats_path)
    else:
        return PipeClass(model_path, device)


def load_existing_results(results_dir: Path) -> List[Dict]:
    """
    从已有结果目录加载生成结果。
    
    Args:
        results_dir: 结果目录路径
        
    Returns:
        结果列表，每个元素包含 sample_id 和 generated_video 路径（已转换为绝对路径）
    """
    results_file = results_dir / "results.json"
    if not results_file.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}")
    
    with open(results_file, "r", encoding="utf-8") as f:
        results = json.load(f)
    
    # 转换视频路径为绝对路径
    for result in results:
        if "generated_video" in result:
            video_path = result["generated_video"]
            video_path_obj = Path(video_path)
            
            if not video_path_obj.is_absolute():
                # 检查路径是否已包含 results_dir 名称（避免重复拼接）
                if video_path_obj.parts and video_path_obj.parts[0] == results_dir.name:
                    video_path = (results_dir.parent / video_path).resolve()
                else:
                    video_path = (results_dir / video_path).resolve()
            else:
                video_path = video_path_obj.resolve()
            
            result["generated_video"] = str(video_path)
    return results


## reference generation
def run_reference(pipeline, reference_func, samples, output_dir, output_key="generated_video"):
    """run reference_func, and collect the generated results"""
    
    # 根据 output_key 动态确定目录名和文件扩展名
    if output_key == "generated_video":
        output_subdir = "videos"
        file_extension = ".mp4"
    elif output_key == "generated_actions":
        output_subdir = "actions"
        file_extension = ".json"
    else:
        output_subdir = "outputs"
        file_extension = ""
    
    output_files_dir = Path(output_dir) / output_subdir
    output_files_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for idx, sample in enumerate(tqdm(samples, desc="Generating")):
        sample_id = sample.get("id", f"sample_{idx:04d}")
        sample["output_path"] = str(output_files_dir / f"{sample_id}{file_extension}")

        try:
            output = reference_func(pipeline, sample, output_key=output_key)
            results.append({"sample_id": sample_id, **output})
        except Exception as e:
            print(f"\n  ERROR [{sample_id}]: {e}")
            results.append({"sample_id": sample_id, "error": str(e)})

    return results


# Evaluation
def run_evaluation(eval_pipeline, eval_func, samples, reference_results, output_dir, data_info, output_key):
    print("Running evaluation ...")
    eval_dir = Path(output_dir) / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    # 创建 sample_id 到原始 sample 的映射
    sample_map = {s.get("id", f"sample_{i:04d}"): s for i, s in enumerate(samples)}
    
    eval_prompt_func = data_info.get("eval_prompt")
    
    eval_results = []
    for ref_result in tqdm(reference_results, desc="Evaluating"):
        sample_id = ref_result.get("sample_id")
        
        if "error" in ref_result:
            eval_results.append({
                "sample_id": sample_id,
                "error": f"Generation failed: {ref_result.get('error')}"
            })
            continue
        
        original_sample = sample_map.get(sample_id, {})
        
        input_data_info = original_sample.copy()
        
        # 动态构建生成结果的路径字段名
        # 例如：generated_video -> generated_video_path
        #      generated_actions -> generated_actions_path
        generated_output_path_key = f"{output_key}_path"
        input_data_info[generated_output_path_key] = ref_result.get(output_key)
        
        # 仅当 eval_prompt_func 存在时才生成提示词（用于 MLLM 评估）
        if eval_prompt_func:
            interaction_signal = original_sample.get("interaction_signal", [])
            scene_description = original_sample.get("scene_description", "")
            prompt_text = eval_prompt_func(interaction_signal, scene_description)
            input_data_info["eval_prompt"] = prompt_text
        
        try:
            eval_result = eval_func(
                input_data_info=input_data_info,
                eval_pipeline=eval_pipeline
            )
            eval_results.append(eval_result)
        except Exception as e:
            print(f"\n  ERROR evaluating [{sample_id}]: {e}")
            eval_results.append({
                "sample_id": sample_id,
                "error": str(e)
            })
    
    # 保存评估结果
    eval_results_file = eval_dir / "evaluation_results.json"
    with open(eval_results_file, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2, ensure_ascii=False, default=str)
    
    # 计算统计信息 - 支持不同类型的评估结果
    successful_evals = [r for r in eval_results if "error" not in r]
    
    if successful_evals:
        # 检查是分数评估（MLLM）还是成功率评估（VLA）
        if "scores" in successful_evals[0]:
            # 分数评估统计（导航视频等）
            avg_scores = {}
            score_keys = ['navigation_fidelity', 'visual_quality', 'temporal_consistency',
                         'scene_consistency', 'motion_smoothness', 'overall']
            
            for key in score_keys:
                values = [r["scores"].get(key) for r in successful_evals 
                         if r["scores"].get(key) is not None]
                if values:
                    avg_scores[key] = sum(values) / len(values)
            
            print(f"\nEvaluation Statistics:")
            print(f"  Successful evaluations: {len(successful_evals)}/{len(eval_results)}")
            if avg_scores:
                print(f"  Average Scores:")
                for key, value in avg_scores.items():
                    print(f"    {key}: {value:.2f}")
                    
        elif "success" in successful_evals[0]:
            # 成功率评估统计（VLA）
            total_success = sum(1 for r in successful_evals if r.get("success", False))
            success_rate = total_success / len(successful_evals) * 100
            
            print(f"\nEvaluation Statistics:")
            print(f"  Total samples: {len(successful_evals)}")
            print(f"  Successful: {total_success}")
            print(f"  Success rate: {success_rate:.2f}%")
            
            # 平均成功步数
            success_steps = [r['success_step'] for r in successful_evals if r.get('success') and r.get('success_step')]
            if success_steps:
                avg_steps = sum(success_steps) / len(success_steps)
                print(f"  Average success step: {avg_steps:.1f}")
    
    print(f"\nEvaluation results saved to {eval_results_file}")
    
    return eval_results


# Main
def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== SceneFlow Benchmark Runner ===")
    print(f"  task_type      : {args.task_type}")
    print(f"  benchmark_name : {args.benchmark_name}")
    print(f"  model_type     : {args.model_type}")
    print(f"  output_dir     : {output_dir}")
    print()

    # ── 1. get data_info from tasks_map ──
    if args.task_type not in tasks_map:
        raise ValueError(
            f"Unknown task_type '{args.task_type}'. "
            f"Available: {list(tasks_map.keys())}"
        )
    benchmarks = tasks_map[args.task_type]

    if args.benchmark_name not in benchmarks:
        raise ValueError(
            f"Unknown benchmark '{args.benchmark_name}'. "
            f"Available: {list(benchmarks.keys())}"
        )
    data_info = benchmarks[args.benchmark_name]

    # ── 2. utilize BenchmarkLoader to load the testing cases ──
    loader = BenchmarkLoader()
    samples = loader.load_benchmark(
        task_type=args.task_type,
        benchmark_name=args.benchmark_name,
        data_path=args.data_path,
        data_info=data_info,
    )
    if args.num_samples is not None:
        samples = samples[: args.num_samples]
    print(f"Loaded {len(samples)} samples\n")

    # ── 3. load the reference pipeline (skip if using existing results) ──
    if args.results_dir:
        pipeline = None
        print("Skipping pipeline loading (using existing results)\n")
    else:
        pipeline = load_pipeline(args.model_type, args.model_path, args.device, args.norm_stats_path)
        print("Pipeline loaded\n")

    # ── 4. obtain reference / eval function ──
    if args.task_type not in eval_func_mapping:
        raise ValueError(
            f"No functions registered for task_type '{args.task_type}'. "
            f"Available: {list(eval_func_mapping.keys())}"
        )
    funcs = eval_func_mapping[args.task_type]
    reference_func = funcs["reference_func"]
    output_key = data_info["output_keys"][0]

    # ── 5.  reference generation or load existing results ──
    if args.results_dir:
        # 跳过生成，加载已有结果
        results_dir = Path(args.results_dir).resolve()
        if not results_dir.exists():
            raise FileNotFoundError(f"Results directory not found: {results_dir}")
        print(f"Loading existing results from {results_dir} ...")
        results = load_existing_results(results_dir)
        print(f"Loaded {len(results)} results\n")
    else:
        # 正常生成
        print("Running reference generation ...")
        results = run_reference(pipeline, reference_func, samples, output_dir, output_key)
        results_file = output_dir / "results.json"

        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)

        successful = sum(1 for r in results if "error" not in r)
        failed = len(results) - successful
        print(f"\nDone — {successful}/{len(results)} successful, {failed} failed")
        print(f"Results saved to {results_file}")
    
    # ── 6. load the evaluation pipeline (if needed) ──
    if args.run_eval:
        eval_pipeline = None
    else:
        eval_pipeline = None

    # ── 7. Evaluation ──
    if args.run_eval:
        eval_func = funcs["eval_func"]
        run_evaluation(eval_pipeline, eval_func, samples, results, output_dir, data_info, output_key)



if __name__ == "__main__":
    main()
