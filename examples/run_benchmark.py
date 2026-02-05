## need load the dataset_info from the data/benchmarks/tasks_map.py
## then input the dataset information into the benchmarkloader
## the eval prompt also load from the dataset_info

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional
import importlib
from tqdm import tqdm

# Import benchmark loader
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from ..data.benchmarks.benchmark_loader import BenchmarkLoader
from .pipeline_mapping import video_gen_pipe, reasoning_pipe, three_dim_pipe


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run benchmarks for navigation video generation models'
    )
    
    # Task configuration
    parser.add_argument(
        '--task_type',
        type=str,
        required=True,
        choices=['navigation_video_generation', 'reasoning', '3d_generation'],
        help='Type of task to evaluate'
    )
    
    parser.add_argument(
        '--benchmark_name',
        type=str,
        required=True,
        help='Name of the benchmark to use (e.g., matrix_game_2)'
    )
    
    # Data configuration
    parser.add_argument(
        '--data_path',
        type=str,
        default=None,
        help='Local path to benchmark data (JSON file or directory)'
    )
    
    parser.add_argument(
        '--hf_repo_id',
        type=str,
        default=None,
        help='HuggingFace repository ID for benchmark data'
    )
    
    # Model configuration
    parser.add_argument(
        '--model_type',
        type=str,
        required=True,
        help='Type of model to test (e.g., matrix-game2)'
    )
    
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to pretrained model or HuggingFace model ID'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to run inference on'
    )
    
    # Output configuration
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./benchmark_results',
        help='Directory to save results'
    )
    
    parser.add_argument(
        '--save_videos',
        action='store_true',
        help='Save generated videos'
    )
    
    # Evaluation configuration
    parser.add_argument(
        '--run_eval',
        action='store_true',
        help='Run evaluation metrics (in addition to generation)'
    )
    
    parser.add_argument(
        '--num_samples',
        type=int,
        default=None,
        help='Number of samples to test (None for all samples)'
    )
    
    return parser.parse_args()


def load_pipeline(model_type: str, model_path: str, device: str = 'cuda'):
    """
    Load the inference pipeline based on model type.
    
    Args:
        model_type: Type of model (e.g., 'matrix-game2')
        model_path: Path to pretrained model
        device: Device to load model on
        
    Returns:
        Loaded pipeline instance
    """
    # Map task types to pipeline registries
    pipeline_registry = {
        'navigation_video_generation': video_gen_pipe,
        'reasoning': reasoning_pipe,
        '3d_generation': three_dim_pipe,
    }
    
    # Determine which registry to use (could be smarter)
    # For now, assume navigation_video_generation
    pipe_dict = video_gen_pipe
    
    if model_type not in pipe_dict:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Available types: {list(pipe_dict.keys())}"
        )
    
    # Get pipeline class
    PipelineClass = pipe_dict[model_type]
    
    # Load pipeline
    print(f"Loading {model_type} pipeline from {model_path}...")
    
    # Different pipelines might have different loading methods
    # For MatrixGame2Pipeline, it uses synthesis_model_path and mode
    if model_type == 'matrix-game2':
        pipeline = PipelineClass.from_pretrained(
            synthesis_model_path=model_path,
            mode="universal",
            device=device
        )
    else:
        # Generic loading
        pipeline = PipelineClass.from_pretrained(
            model_path,
            device=device
        )
    
    return pipeline


def load_evaluation_functions(task_type: str):
    """
    Load the evaluation functions for a specific task type.
    
    Args:
        task_type: Type of task
        
    Returns:
        Tuple of (reference_func, eval_func)
    """
    # Map task type to evaluation module
    eval_module_map = {
        'navigation_video_generation': 'examples.evaluation_tasks.navigation_video_generation',
    }
    
    if task_type not in eval_module_map:
        raise ValueError(f"No evaluation module found for task type: {task_type}")
    
    module_name = eval_module_map[task_type]
    
    try:
        eval_module = importlib.import_module(module_name)
        reference_func = getattr(eval_module, 'reference_func')
        eval_func = getattr(eval_module, 'eval_func')
        return reference_func, eval_func
    except ImportError as e:
        raise ImportError(f"Failed to import evaluation module {module_name}: {e}")


def run_benchmark(args):
    """
    Main function to run the benchmark.
    
    Args:
        args: Parsed command line arguments
    """
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"=== SceneFlow Benchmark Runner ===")
    print(f"Task: {args.task_type}")
    print(f"Benchmark: {args.benchmark_name}")
    print(f"Model: {args.model_type}")
    print(f"Output: {output_dir}")
    print()
    
    # Step 1: Load benchmark data
    print("Step 1: Loading benchmark data...")
    loader = BenchmarkLoader()
    
    benchmark_samples = loader.load_benchmark(
        task_type=args.task_type,
        benchmark_name=args.benchmark_name,
        data_path=args.data_path,
        hf_repo_id=args.hf_repo_id,
    )
    
    # Limit number of samples if specified
    if args.num_samples is not None:
        benchmark_samples = benchmark_samples[:args.num_samples]
    
    print(f"Loaded {len(benchmark_samples)} samples")
    print()
    
    # Step 2: Load inference pipeline
    print("Step 2: Loading inference pipeline...")
    pipeline = load_pipeline(args.model_type, args.model_path, args.device)
    print("Pipeline loaded successfully")
    print()
    
    # Step 3: Load evaluation functions
    print("Step 3: Loading evaluation functions...")
    reference_func, eval_func = load_evaluation_functions(args.task_type)
    print("Evaluation functions loaded")
    print()
    
    # Step 4: Run inference and evaluation
    print("Step 4: Running inference and evaluation...")
    print("-" * 60)
    
    results = []
    
    for idx, sample in enumerate(tqdm(benchmark_samples, desc="Processing samples")):
        print(f"\nSample {idx + 1}/{len(benchmark_samples)}: {sample.get('id', 'unknown')}")
        
        # Add output directory to sample info
        sample['output_dir'] = output_dir / 'videos'
        
        try:
            if args.run_eval:
                # Run full evaluation
                eval_result = eval_func(pipeline, sample)
                results.append(eval_result)
                print(f"  Generated: {eval_result.get('generated_video_path', 'N/A')}")
                print(f"  Metrics: {eval_result.get('metrics', {})}")
            else:
                # Just run reference generation
                gen_result = reference_func(pipeline, sample)
                results.append({
                    'sample_id': sample.get('id'),
                    'video_path': gen_result['video_path'],
                    'metadata': gen_result['metadata']
                })
                print(f"  Generated: {gen_result['video_path']}")
        
        except Exception as e:
            print(f"  ERROR: {str(e)}")
            results.append({
                'sample_id': sample.get('id'),
                'error': str(e)
            })
    
    print()
    print("-" * 60)
    
    # Step 5: Save results
    print("Step 5: Saving results...")
    
    results_file = output_dir / 'results.json'
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to: {results_file}")
    
    # Print summary
    print()
    print("=== Summary ===")
    successful = sum(1 for r in results if 'error' not in r)
    failed = len(results) - successful
    print(f"Total samples: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    if args.run_eval:
        # Compute average metrics if available
        all_metrics = [r.get('metrics', {}) for r in results if 'metrics' in r]
        if all_metrics:
            print("\n=== Average Metrics ===")
            # This is a simplified version - you'd want more sophisticated aggregation
            for key in all_metrics[0].keys():
                try:
                    values = [m[key] for m in all_metrics if key in m and isinstance(m[key], (int, float))]
                    if values:
                        avg = sum(values) / len(values)
                        print(f"{key}: {avg:.4f}")
                except:
                    pass
    
    print()
    print("Benchmark completed!")


def main():
    """Main entry point."""
    args = parse_args()
    run_benchmark(args)


if __name__ == '__main__':
    main()

