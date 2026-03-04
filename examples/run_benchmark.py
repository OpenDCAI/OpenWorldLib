"""
SceneFlow Benchmark Runner
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import List, Dict, Union
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from data.benchmarks.tasks_map import tasks_map
from data.benchmarks.benchmark_loader import BenchmarkLoader
from examples.pipeline_load_mapping import video_gen_pipe, reasoning_pipe, three_dim_pipe, vla_pipe
from examples.pipeline_infer_mapping import video_gen_pipe_infer, reasoning_pipe_infer, three_dim_pipe_infer, vla_pipe_infer
from examples.evaluation_tasks.eval_func_mapping import eval_func_mapping


# collect evaluation pipelines
# This loading way is used to verify whether the loaded pipe corresponds to the intended task.
ALL_PIPELINES = {**video_gen_pipe, **reasoning_pipe, **three_dim_pipe, **vla_pipe}
ALL_PIPELINES_INFER = {**video_gen_pipe_infer, **reasoning_pipe_infer, **three_dim_pipe_infer, **vla_pipe_infer}


def parse_args():
    parser = argparse.ArgumentParser(description="SceneFlow Benchmark Runner")
    parser.add_argument("--task_type", type=str, required=True,
                        help="tasks_map contain various, like navigation_video_gen")
    parser.add_argument("--benchmark_name", type=str, required=True,
                        help="the name of benchmark , such as sf_nav_vidgen_test")
    parser.add_argument("--data_path", type=str, required=True,
                        help="local data file path HuggingFace repo id")
    parser.add_argument("--eval_model_path", type=str, default="Qwen/Qwen2.5-Omni-7B-Instruct",
                        help=(
                            "evaluation MLLM model path or HuggingFace model id. "
                            "Can be a plain string or a JSON dict string for multi-path models, "
                            "e.g. '{\"pretrained_model_path\": \"Qwen/Qwen2.5-Omni-7B-Instruct\"}'"
                        ))
    parser.add_argument("--model_type", type=str,
                        help="pipeline_mapping matrix-game2")
    parser.add_argument("--eval_model_type", type=str, default="qwen2p5omni",
                        help="evaluation MLLM model type, like qwen2p5omni")
    parser.add_argument("--model_path", type=str,
                        help=(
                            "model path or HuggingFace model id. "
                            "Can be a plain string or a JSON dict string for multi-path models, "
                            "e.g. '{\"synthesis_model_path\": \"tencent/Hunyuan-GameCraft-1.0\", "
                            "\"other_model_path\": \"some/other-model\"}'"
                        ))
    parser.add_argument("--output_dir", type=str, default="./benchmark_results")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="test N samples, default ")
    parser.add_argument("--run_eval", action="store_true",
                        help="whether to carry out evaluation")
    parser.add_argument("--results_dir", type=str, default=None,
                        help="path to existing results directory (skip generation if provided)")
    parser.add_argument("--norm_stats_path", type=str, default=None,
                        help="normalization stats path (required for VLA models)")
    return parser.parse_args()


def parse_model_path(model_path_str: str) -> Union[str, Dict[str, str], None]:
    """
    Parse --model_path / --eval_model_path CLI argument.

    - If the value is a valid JSON object string, parse and return as dict.
      Example: '{"synthesis_model_path": "tencent/Hunyuan-GameCraft-1.0"}'
    - Otherwise return the original string (single HuggingFace id / local path).
      Example: "tencent/Hunyuan-GameCraft-1.0"
    - Returns None if input is None.
    """
    if model_path_str is None:
        return None
    try:
        parsed = json.loads(model_path_str)
        if isinstance(parsed, dict):
            return parsed
    except (json.JSONDecodeError, ValueError):
        pass
    return model_path_str


# Pipeline loading here
def load_pipeline(
    model_type: str, 
    model_path: Union[str, Dict], 
    device: str = "cuda",
    norm_stats_path: str = None
):
    """Load the pipeline according to the model_type.

    Args:
        model_type: key registered in ALL_PIPELINES.
        model_path: either a plain string (single HuggingFace id / local path)
                    or a dict mapping path-keys to paths for multi-weight models.
        device: target device.
        norm_stats_path: normalization stats path (only for VLA models).
    """
    if model_type not in ALL_PIPELINES:
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            f"Available: {list(ALL_PIPELINES.keys())}"
        )

    PipeClass = ALL_PIPELINES[model_type]
    
    # Check if this is a VLA model that requires norm_stats_path
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
        结果列表，每个元素包含 sample_id 和输出路径（已转换为绝对路径）
        输出可能是 generated_video 或 generated_actions，取决于任务类型
    """
    results_file = results_dir / "results.json"
    if not results_file.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}")
    
    with open(results_file, "r", encoding="utf-8") as f:
        results = json.load(f)
    
    # 转换输出路径为绝对路径（video 或 action）
    for result in results:
        # 检查是 video 还是 action 输出
        output_key = None
        if "generated_video" in result:
            output_key = "generated_video"
        elif "generated_actions" in result:
            output_key = "generated_actions"
        
        if output_key:
            output_path = result[output_key]
            output_path_obj = Path(output_path)
            
            if not output_path_obj.is_absolute():
                # 检查路径是否已包含 results_dir 名称（避免重复拼接）
                if output_path_obj.parts and output_path_obj.parts[0] == results_dir.name:
                    output_path = (results_dir.parent / output_path).resolve()
                else:
                    output_path = (results_dir / output_path).resolve()
            else:
                output_path = output_path_obj.resolve()
            
            result[output_key] = str(output_path)
    return results


## reference generation
def run_reference(pipeline, pipeline_infer, reference_func, samples, output_dir, output_key="generated_video"):
    """run reference_func, and collect the generated results
    
    Args:
        pipeline: loaded pipeline instance
        pipeline_infer: inference function for the pipeline
        reference_func: reference function from eval_func_mapping
        samples: list of test samples
        output_dir: output directory
        output_key: key for the generated output ("generated_video" or "generated_actions")
    """
    # 根据 output_key 决定输出目录名称
    if output_key == "generated_video":
        output_subdir = "videos"
        file_extension = ".mp4"
    elif output_key == "generated_actions":
        output_subdir = "actions"
        file_extension = ".json"
    else:
        # 默认使用 output_key 作为子目录名
        output_subdir = output_key.replace("generated_", "")
        file_extension = ".mp4"  # 默认扩展名
    
    output_path_dir = Path(output_dir) / output_subdir
    output_path_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for idx, sample in enumerate(tqdm(samples, desc="Generating")):
        sample_id = sample.get("id", f"sample_{idx:04d}")
        sample["output_path"] = str(output_path_dir / f"{sample_id}{file_extension}")

        try:
            output = reference_func(pipeline, pipeline_infer, sample, output_key=output_key)
            results.append({"sample_id": sample_id, **output})
        except Exception as e:
            print(f"\n  ERROR [{sample_id}]: {e}")
            results.append({"sample_id": sample_id, "error": str(e)})

    return results


# Evaluation
def run_evaluation(eval_pipeline, eval_pipeline_infer, eval_func, samples, reference_results, output_dir, data_info):
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
        
        # 生成评估提示词文本（如果有 eval_prompt 函数）
        # eval_prompt 函数接收整个 sample 字典，内部自己提取需要的字段
        if eval_prompt_func:
            prompt_text = eval_prompt_func(original_sample)
        else:
            prompt_text = None
        
        input_data_info = original_sample.copy()
        
        # 根据输出类型设置相应的路径字段
        if "generated_video" in ref_result:
            input_data_info["generated_video_path"] = ref_result.get("generated_video")
        elif "generated_actions" in ref_result:
            input_data_info["generated_actions_path"] = ref_result.get("generated_actions")
        
        if prompt_text:
            input_data_info["eval_prompt"] = prompt_text
        
        try:
            eval_result = eval_func(
                input_data_info=input_data_info,
                eval_pipeline=eval_pipeline,
                eval_pipeline_infer=eval_pipeline_infer,
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
    
    # 计算统计信息
    successful_evals = [r for r in eval_results if "error" not in r]
    
    # 检查评估结果类型
    if successful_evals:
        # 检查是否为视频生成评估（带 scores）
        if successful_evals[0].get("scores") is not None:
            avg_scores = {}
            score_keys = ['navigation_fidelity', 'visual_quality', 'temporal_consistency',
                         'scene_consistency', 'motion_smoothness', 'overall',
                         'text_video_alignment', 'content_relevance', 'motion_naturalness']
            
            for key in score_keys:
                values = [r["scores"].get(key) for r in successful_evals 
                         if r.get("scores", {}).get(key) is not None]
                if values:
                    avg_scores[key] = sum(values) / len(values)
            
            print(f"\nEvaluation Statistics:")
            print(f"  Successful evaluations: {len(successful_evals)}/{len(eval_results)}")
            if avg_scores:
                print(f"  Average Scores:")
                for key, value in avg_scores.items():
                    print(f"    {key}: {value:.2f}")
        
        # 检查是否为 VLA 评估（带 success）
        elif "success" in successful_evals[0]:
            total_success = sum(1 for r in successful_evals if r.get("success", False))
            success_rate = total_success / len(successful_evals) * 100
            
            print(f"\nEvaluation Statistics:")
            print(f"  Successful evaluations: {len(successful_evals)}/{len(eval_results)}")
            print(f"  Task Success Rate: {success_rate:.2f}% ({total_success}/{len(successful_evals)})")
            
            # 计算平均成功步数
            success_steps = [r.get("success_step") for r in successful_evals 
                           if r.get("success", False) and r.get("success_step") is not None]
            if success_steps:
                avg_success_step = sum(success_steps) / len(success_steps)
                print(f"  Average Success Step: {avg_success_step:.2f}")
    
    
    print(f"\nEvaluation results saved to {eval_results_file}")
    
    return eval_results


# Main
def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Parse model_path arguments (str → str or dict) ──
    model_path = parse_model_path(args.model_path)
    eval_model_path = parse_model_path(args.eval_model_path)

    print("=== SceneFlow Benchmark Runner ===")
    print(f"  task_type      : {args.task_type}")
    print(f"  benchmark_name : {args.benchmark_name}")
    print(f"  model_type     : {args.model_type}")
    print(f"  model_path     : {model_path}")
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
        # 根据模型类型决定是否传入 norm_stats_path
        if args.model_type in vla_pipe:
            pipeline = load_pipeline(
                args.model_type, 
                model_path, 
                args.device, 
                norm_stats_path=args.norm_stats_path
            )
        else:
            pipeline = load_pipeline(args.model_type, model_path, args.device)
        print("Pipeline loaded\n")
    pipeline_infer = ALL_PIPELINES_INFER.get(args.model_type, None)

    # ── 4. obtain reference / eval function ──
    if args.task_type not in eval_func_mapping:
        raise ValueError(
            f"No functions registered for task_type '{args.task_type}'. "
            f"Available: {list(eval_func_mapping.keys())}"
        )
    funcs = eval_func_mapping[args.task_type]
    reference_func = funcs["reference_func"]
    output_key = data_info["output_keys"][0]

    # ── 5. reference generation or load existing results ──
    if args.results_dir:
        # skip the generation, directly load existing results
        results_dir = Path(args.results_dir).resolve()
        if not results_dir.exists():
            raise FileNotFoundError(f"Results directory not found: {results_dir}")
        print(f"Loading existing results from {results_dir} ...")
        results = load_existing_results(results_dir)
        print(f"Loaded {len(results)} results\n")
    else:
        print("Running reference generation ...")
        results = run_reference(pipeline, pipeline_infer, reference_func, samples, output_dir, output_key)
        results_file = output_dir / "results.json"

        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)

        successful = sum(1 for r in results if "error" not in r)
        failed = len(results) - successful
        print(f"\nDone — {successful}/{len(results)} successful, {failed} failed")
        print(f"Results saved to {results_file}")
    
    # ── 6. load the evaluation pipeline (if needed) ──
    if args.run_eval:
        eval_pipeline = load_pipeline(args.eval_model_type, eval_model_path, args.device)
        print("Evaluation pipeline loaded\n")
    else:
        eval_pipeline = None
    eval_pipeline_infer = ALL_PIPELINES_INFER.get(args.eval_model_type, None)

    # ── 7. Evaluation ──
    if args.run_eval:
        eval_func = funcs["eval_func"]
        run_evaluation(eval_pipeline, eval_pipeline_infer, eval_func, samples, results, output_dir, data_info)


if __name__ == "__main__":
    main()
