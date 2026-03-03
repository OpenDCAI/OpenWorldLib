from diffusers.utils import export_to_video
from PIL import Image
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import json
import re

def reference_func(
    pipe,
    pipe_infer,
    input_data_info: Dict[str, Any],
    output_key: str = "generated_video"
) -> Dict[str, Any]:
    """
    根据 input_data_info（由 BenchmarkLoader 组装的单条测例），
    驱动 MatrixGame2Pipeline 生成导航视频并返回结果字典。

    Args:
        pipe:            已初始化的 MatrixGame2Pipeline 实例。
        input_data_info: 单条测例字典，至少包含
                         - ref_image:           参考图片的绝对路径（str）
                         - interaction_signal:   交互信号列表或 JSON 字符串
                         - scene_description:    场景描述（仅用于评估，不传入 pipeline）
                         可选：
                         - num_output_frames:    生成帧数，默认 150
                         - fps:                  保存视频帧率，默认 12
                         - output_path:          若提供，则将视频保存到该路径
        output_key:      输出字典中存放生成视频的键名。

    Returns:
        {output_key: 生成的视频帧列表} 或
        {output_key: 保存后的视频文件路径}（当 input_data_info 含 output_path 时）
    """
    ref_image_path = input_data_info["ref_image"]
    input_image = Image.open(ref_image_path).convert("RGB")

    interaction_signal = input_data_info["interaction_signal"]
    # 兼容 metadata 中将 list 存为 JSON 字符串的情况
    if isinstance(interaction_signal, str):
        try:
            interaction_signal = json.loads(interaction_signal)
        except json.JSONDecodeError:
            # 尝试逗号分隔的纯文本："forward,left,right"
            interaction_signal = [
                s.strip() for s in interaction_signal.split(",") if s.strip()
            ]

    output_path = input_data_info.get("output_path", None)
    fps = int(input_data_info.get("fps", 12))
    output_video = pipe_infer(pipe, input_image, interaction_signal, output_path, fps)
    if output_path is not None:
        return {output_key: str(output_path)}
    return {output_key: output_video}


# eval function
def eval_func(
    input_data_info: Dict[str, Any],
    eval_pipeline: None,
    eval_pipeline_infer: None,
) -> Dict[str, Any]:
    """
    使用多模态 LLM 评估生成的导航视频质量。
    
    Args:
        pipe: 生成视频的 pipeline（MatrixGame2Pipeline），此参数保留以兼容接口
        input_data_info: 单条测例字典，包含：
            - ref_image: 参考图片的绝对路径
            - interaction_signal: 交互信号列表
            - scene_description: 场景描述
            - generated_video_path: 生成的视频路径（从 reference_results 传入）
            - eval_prompt: 评估提示词函数（从 data_info 传入）
        eval_pipeline: 已初始化的评估用 MLLM pipeline（可选）
        eval_model_path: 评估模型路径（如果 eval_pipeline 为 None 则使用此路径加载）
        device: 设备类型
    
    Returns:
        包含评估结果的字典：
        {
            'sample_id': str,
            'generated_video_path': str,
            'scores': {
                'navigation_fidelity': float,
                'visual_quality': float,
                'temporal_consistency': float,
                'scene_consistency': float,
                'motion_smoothness': float,
                'overall': float
            },
            'comments': str,
            'raw_response': str  # LLM 的原始响应
        }
    """
    generated_video_path = input_data_info.get("generated_video_path")
    if not generated_video_path:
        raise ValueError("generated_video_path not found in input_data_info")
    
    ref_image_path = input_data_info["ref_image"]
    if not Path(ref_image_path).exists():
        raise FileNotFoundError(f"Reference image not found: {ref_image_path}")
    
    prompt_text = input_data_info.get("eval_prompt")
    if not prompt_text:
        raise ValueError("eval_prompt text not found in input_data_info")    
    if not isinstance(prompt_text, str):
        raise ValueError(f"eval_prompt should be a string, got {type(prompt_text)}")

    try:
        response_text = eval_pipeline_infer(eval_pipeline, prompt_text,
                                            ref_image_path, generated_video_path)
        
    except Exception as e:
        return {
            'sample_id': input_data_info.get('id', 'unknown'),
            'generated_video_path': generated_video_path,
            'error': f"Evaluation failed: {str(e)}"
        }
    
    # 6. 解析 LLM 输出，提取分数
    scores = _parse_evaluation_scores(response_text)
    
    # 7. 构建返回结果
    result = {
        'sample_id': input_data_info.get('id', 'unknown'),
        'generated_video_path': generated_video_path,
        'scores': scores,
        'raw_response': response_text
    }
    
    if 'comments' in scores:
        result['comments'] = scores['comments']
    
    return result


def _parse_evaluation_scores(response_text: str) -> Dict[str, Any]:
    """
    从 LLM 响应中解析评估分数。
    
    期望格式：
    [Navigation Fidelity Score]: <number>
    [Visual Quality Score]: <number>
    [Temporal Consistency Score]: <number>
    [Scene Consistency Score]: <number>
    [Motion Smoothness Score]: <number>
    [Overall Score]: <float>
    [Comments]: <string>
    """
    scores = {}
    
    # 定义正则表达式模式
    patterns = {
        'navigation_fidelity': r'\[Navigation Fidelity Score\]:\s*(\d+(?:\.\d+)?)',
        'visual_quality': r'\[Visual Quality Score\]:\s*(\d+(?:\.\d+)?)',
        'temporal_consistency': r'\[Temporal Consistency Score\]:\s*(\d+(?:\.\d+)?)',
        'scene_consistency': r'\[Scene Consistency Score\]:\s*(\d+(?:\.\d+)?)',
        'motion_smoothness': r'\[Motion Smoothness Score\]:\s*(\d+(?:\.\d+)?)',
        'overall': r'\[Overall Score\]:\s*(\d+\.?\d*)',
        'comments': r'\[Comments\]:\s*(.+?)(?=\n\n|\n\[|$)'
    }
    
    # 提取分数
    for key, pattern in patterns.items():
        match = re.search(pattern, response_text, re.IGNORECASE | re.DOTALL)
        if match:
            if key == 'comments':
                scores[key] = match.group(1).strip()
            else:
                try:
                    scores[key] = float(match.group(1))
                except ValueError:
                    scores[key] = None
        else:
            scores[key] = None
    
    # 验证分数范围
    for key in ['navigation_fidelity', 'visual_quality', 'temporal_consistency', 
                'scene_consistency', 'motion_smoothness']:
        if scores.get(key) is not None:
            scores[key] = max(1, min(10, scores[key]))
    
    if scores.get('overall') is not None:
        scores['overall'] = max(1.0, min(10.0, scores['overall']))
    
    return scores

