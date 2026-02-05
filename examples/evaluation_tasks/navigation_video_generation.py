from diffusers.utils import export_to_video
from PIL import Image
from pathlib import Path
from typing import Dict, Any, Optional
import torch


def reference_func(
    pipe,
    input_data_info: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generate reference video using the navigation video generation pipeline.
    
    This function takes the input data and uses the pipeline to generate a video
    based on the reference image and interaction signals.
    
    Args:
        pipe: The loaded pipeline (e.g., MatrixGame2Pipeline)
        input_data_info: Dictionary containing:
            - image_path: Path to the reference/input image
            - interaction_signals: List of navigation commands 
              (e.g., ["forward", "left", "right", "forward_left", "forward_right", "camera_l", "camera_r"])
            - num_frames: Number of output frames to generate (default: 150)
            - (optional) other model-specific parameters
    
    Returns:
        Dictionary containing:
            - output_video: Generated video frames (torch.Tensor or numpy array)
            - video_path: Path where the video was saved (if saved)
            - metadata: Additional generation metadata
    
    Example:
        >>> pipeline = MatrixGame2Pipeline.from_pretrained(...)
        >>> data = {
        ...     'image_path': './data/test_case1/ref_image.png',
        ...     'interaction_signals': ['forward', 'left', 'right'],
        ...     'num_frames': 150
        ... }
        >>> result = reference_func(pipeline, data)
    """
    
    # Extract input parameters
    image_path = input_data_info.get('image_path')
    interaction_signals = input_data_info.get('interaction_signals', 
                                               ["forward", "left", "right"])
    num_frames = input_data_info.get('num_frames', 150)
    
    # Validate inputs
    if image_path is None:
        raise ValueError("input_data_info must contain 'image_path'")
    
    # Load the input image
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    input_image = Image.open(image_path).convert('RGB')
    
    # Prepare pipeline kwargs
    pipe_kwargs = {
        'input_image': input_image,
        'num_output_frames': num_frames,
        'interaction_signal': interaction_signals,
    }
    
    # Add any additional parameters from input_data_info
    # that might be specific to the pipeline
    for key in ['guidance_scale', 'num_inference_steps', 'seed', 'height', 'width']:
        if key in input_data_info:
            pipe_kwargs[key] = input_data_info[key]
    
    # Generate video
    output_video = pipe(**pipe_kwargs)
    
    # Prepare output directory
    sample_id = input_data_info.get('id', 'unknown')
    output_dir = Path(input_data_info.get('output_dir', './outputs'))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the generated video
    video_filename = f"{sample_id}_generated.mp4"
    video_path = output_dir / video_filename
    
    # Export video (assuming output_video is a tensor or list of frames)
    fps = input_data_info.get('fps', 12)
    export_to_video(output_video, str(video_path), fps=fps)
    
    # Return results
    return {
        'output_video': output_video,
        'video_path': str(video_path),
        'metadata': {
            'sample_id': sample_id,
            'num_frames': num_frames,
            'interaction_signals': interaction_signals,
            'fps': fps,
        }
    }


def eval_func(
    pipe,
    input_data_info: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Evaluate the generated navigation video against ground truth or prompts.
    
    This function generates a video and optionally evaluates it against:
    1. Ground truth video (if provided)
    2. Evaluation prompts/criteria (if provided)
    
    Args:
        pipe: The loaded pipeline for generation
        input_data_info: Dictionary containing:
            - All fields required by reference_func
            - eval_prompt: (optional) Text prompt describing evaluation criteria
            - gt_video_path: (optional) Path to ground truth video
            - eval_metrics: (optional) List of metrics to compute
    
    Returns:
        Dictionary containing:
            - generated_video_path: Path to the generated video
            - eval_results: Dictionary of evaluation metrics and scores
            - comparison: (optional) Comparison with ground truth
    
    Example:
        >>> data = {
        ...     'image_path': './test.png',
        ...     'interaction_signals': ['forward', 'left'],
        ...     'eval_prompt': 'The camera should move forward then turn left',
        ...     'gt_video_path': './ground_truth.mp4'
        ... }
        >>> results = eval_func(pipeline, data)
        >>> print(results['eval_results'])
    """
    
    # First, generate the reference video
    gen_result = reference_func(pipe, input_data_info)
    
    # Extract evaluation parameters
    eval_prompt = input_data_info.get('eval_prompt', '')
    gt_video_path = input_data_info.get('gt_video_path', None)
    eval_metrics = input_data_info.get('eval_metrics', ['visual_quality', 'motion_consistency'])
    
    # Initialize evaluation results
    eval_results = {
        'sample_id': input_data_info.get('id', 'unknown'),
        'generated_video_path': gen_result['video_path'],
        'metrics': {}
    }
    
    # Evaluate based on prompt (if provided)
    if eval_prompt:
        eval_results['eval_prompt'] = eval_prompt
        # TODO: Implement prompt-based evaluation
        # This could use a vision-language model to assess if the video
        # matches the described navigation behavior
        eval_results['metrics']['prompt_alignment'] = _evaluate_prompt_alignment(
            gen_result['output_video'],
            eval_prompt
        )
    
    # Evaluate against ground truth (if provided)
    if gt_video_path and Path(gt_video_path).exists():
        eval_results['gt_video_path'] = gt_video_path
        # TODO: Implement ground truth comparison
        # This could compute metrics like PSNR, SSIM, FVD, etc.
        eval_results['metrics']['gt_comparison'] = _evaluate_against_gt(
            gen_result['video_path'],
            gt_video_path,
            eval_metrics
        )
    
    # Compute quality metrics
    eval_results['metrics']['quality'] = _evaluate_video_quality(
        gen_result['output_video']
    )
    
    # Evaluate motion consistency
    eval_results['metrics']['motion_consistency'] = _evaluate_motion_consistency(
        gen_result['output_video'],
        input_data_info.get('interaction_signals', [])
    )
    
    return eval_results


def _evaluate_prompt_alignment(video_frames, eval_prompt: str) -> Dict[str, float]:
    """
    Evaluate how well the generated video aligns with the evaluation prompt.
    
    Args:
        video_frames: Generated video frames
        eval_prompt: Text description of expected behavior
        
    Returns:
        Dictionary of alignment scores
    """
    # Placeholder implementation
    # In practice, this would use a vision-language model to assess alignment
    return {
        'score': 0.0,  # Placeholder
        'confidence': 0.0,
        'note': 'Prompt-based evaluation not yet implemented'
    }


def _evaluate_against_gt(gen_video_path: str, gt_video_path: str, 
                        metrics: list) -> Dict[str, float]:
    """
    Compare generated video against ground truth.
    
    Args:
        gen_video_path: Path to generated video
        gt_video_path: Path to ground truth video
        metrics: List of metrics to compute
        
    Returns:
        Dictionary of metric scores
    """
    # Placeholder implementation
    # In practice, this would load both videos and compute metrics like:
    # - PSNR (Peak Signal-to-Noise Ratio)
    # - SSIM (Structural Similarity Index)
    # - FVD (Fréchet Video Distance)
    # - LPIPS (Learned Perceptual Image Patch Similarity)
    
    results = {}
    for metric in metrics:
        results[metric] = 0.0  # Placeholder
    
    results['note'] = 'Ground truth comparison not yet implemented'
    return results


def _evaluate_video_quality(video_frames) -> Dict[str, float]:
    """
    Evaluate general video quality metrics.
    
    Args:
        video_frames: Video frames to evaluate
        
    Returns:
        Dictionary of quality scores
    """
    # Placeholder implementation
    # Could compute metrics like:
    # - Temporal consistency
    # - Sharpness
    # - Flickering detection
    
    return {
        'temporal_consistency': 0.0,
        'sharpness': 0.0,
        'overall_quality': 0.0,
        'note': 'Quality evaluation not yet implemented'
    }


def _evaluate_motion_consistency(video_frames, interaction_signals: list) -> Dict[str, Any]:
    """
    Evaluate if the video motion matches the interaction signals.
    
    Args:
        video_frames: Generated video frames
        interaction_signals: List of navigation commands
        
    Returns:
        Dictionary of consistency scores
    """
    # Placeholder implementation
    # Could analyze optical flow and verify it matches the commanded motions
    
    return {
        'signal_match_score': 0.0,
        'expected_signals': interaction_signals,
        'detected_motions': [],
        'note': 'Motion consistency evaluation not yet implemented'
    }

