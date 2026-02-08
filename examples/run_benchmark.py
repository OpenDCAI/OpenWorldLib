"""
SceneFlow Benchmark Runner
Usage:
    python -m examples.run_benchmark
        --task_type navigation_video_generation
        --benchmark_name sf_nav_vidgen_test
        --data_path ./data/benchmarks/navigation_video_gen/sf_nav_vidgen_test
        --model_type matrix-game2
        --model_path Skywork/Matrix-Game-2.0
        --output_dir ./benchmark_results
        --num_samples 5
"""

import os
import sys
import argparse
import json
from pathlib import Path
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from data.benchmarks.tasks_map import tasks_map
from data.benchmarks.benchmark_loader import BenchmarkLoader
from examples.pipeline_mapping import video_gen_pipe, reasoning_pipe, three_dim_pipe
from examples.evaluation_tasks.eval_func_mapping import eval_func_mapping


# collect evaluation pipelines
ALL_PIPELINES = {**video_gen_pipe, **reasoning_pipe, **three_dim_pipe}

def parse_args():
    parser = argparse.ArgumentParser(description="SceneFlow Benchmark Runner")
    parser.add_argument("--task_type", type=str, required=True,
                        help="tasks_map contain various, like navigation_video_gen")
    parser.add_argument("--benchmark_name", type=str, required=True,
                        help="the name of benchmark , such as sf_nav_vidgen_test")
    parser.add_argument("--data_path", type=str, required=True,
                        help="local data file path HuggingFace repo id")
    parser.add_argument("--model_type", type=str, required=True,
                        help="pipeline_mapping matrix-game2")
    parser.add_argument("--model_path", type=str, required=True,
                        help="model path or HuggingFace model id")
    parser.add_argument("--output_dir", type=str, default="./benchmark_results")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="test N samples, default ")
    parser.add_argument("--run_eval", action="store_true",
                        help="whether to carry out evaluation")
    return parser.parse_args()


# Pipeline loading here
def load_pipeline(model_type: str, model_path: str, device: str = "cuda"):
    """load the pipeline according to the model_type."""
    if model_type not in ALL_PIPELINES:
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            f"Available: {list(ALL_PIPELINES.keys())}"
        )

    PipeClass = ALL_PIPELINES[model_type]

    if model_type == "matrix-game2":
        return PipeClass.from_pretrained(
            synthesis_model_path=model_path,
            mode="universal",
            device=device,
        )

    return PipeClass.from_pretrained(model_path, device=device)


## reference generation
def run_reference(pipeline, reference_func, samples, output_dir, output_key="generated_video"):
    """run reference_func, and collect the generated results"""
    videos_dir = Path(output_dir) / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for idx, sample in enumerate(tqdm(samples, desc="Generating")):
        sample_id = sample.get("id", f"sample_{idx:04d}")
        sample["output_path"] = str(videos_dir / f"{sample_id}.mp4")

        try:
            output = reference_func(pipeline, sample, output_key=output_key)
            results.append({"sample_id": sample_id, **output})
        except Exception as e:
            print(f"\n  ERROR [{sample_id}]: {e}")
            results.append({"sample_id": sample_id, "error": str(e)})

    return results


# Evaluation（占位，后续实现）
def run_evaluation(eval_func, samples, reference_results, output_dir):
    """todo: realise this later"""
    raise NotImplementedError("Evaluation will be implemented in a future update.")


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

    # ── 3. load the reference pipeline ──
    pipeline = load_pipeline(args.model_type, args.model_path, args.device)
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

    # ── 5.  reference generation ──
    print("Running reference generation ...")
    results = run_reference(pipeline, reference_func, samples, output_dir, output_key)

    # ──（暂未实现）──
    if args.run_eval:
        eval_func = funcs["eval_func"]
        run_evaluation(eval_func, samples, results, output_dir)

    results_file = output_dir / "results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    successful = sum(1 for r in results if "error" not in r)
    failed = len(results) - successful
    print(f"\nDone — {successful}/{len(results)} successful, {failed} failed")
    print(f"Results saved to {results_file}")


if __name__ == "__main__":
    main()
