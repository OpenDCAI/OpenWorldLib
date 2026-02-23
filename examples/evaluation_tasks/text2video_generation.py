from diffusers.utils import export_to_video
from PIL import Image
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import json
import re

def reference_func(
    pipe,
    input_data_info: Dict[str, Any],
    output_key: str = "generated_video"
) -> Dict[str, Any]:
    """
    根据 input_data_info（由 BenchmarkLoader 组装的单条测例），
    驱动 pipeline（Wan2p2Pipeline 或其他）生成文本到视频并返回结果字典。

    Args:
        pipe:            已初始化的 pipeline 实例（Wan2p2Pipeline 或其他）。
        input_data_info: 单条测例字典，至少包含：
                         - generation_text: 文本提示词（必需）
                         可选：
                         - num_output_frames:    生成帧数，默认使用 pipeline 配置
                         - fps:                  保存视频帧率，默认 12
                         - output_path:          若提供，则将视频保存到该路径
        output_key:      输出字典中存放生成视频的键名。

    Returns:
        {output_key: 生成的视频张量或帧列表} 或
        {output_key: 保存后的视频文件路径}（当 input_data_info 含 output_path 时）
    """
    generation_text = input_data_info["generation_text"]
    
    pipeline_type = type(pipe).__name__
    if "Wan2p2" in pipeline_type:
        output_video = pipe(
            prompt=generation_text,
            size="1280*704",
        )

    # 这里保留作为其他pipeline的接口
    # elif "" in pipeline_type:
    #     pass
    
    output_path = input_data_info.get("output_path", None)
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fps = int(input_data_info.get("fps", 12))
        
        # 处理不同 pipeline 的输出格式,Wan2p2Pipeline 返回 torch.Tensor (C, T, H, W),需要转换为 PIL Image 列表供 export_to_video 使用
        if isinstance(output_video, torch.Tensor):
            from sceneflow.memories.visual_synthesis.wan.wan_2p2_memeory import tensor_frame_to_pil
            
            if output_video.ndim == 4:
                video_frames = []
                # (C, T, H, W) -> 遍历 T 维度
                for t in range(output_video.shape[1]):
                    frame = output_video[:, t, :, :]  # (C, H, W)
                    pil_img = tensor_frame_to_pil(frame)
                    video_frames.append(pil_img)
                export_to_video(video_frames, str(output_path), fps=fps)
            else:
                raise ValueError(f"Unexpected video tensor shape: {output_video.shape}")
        else:
            # 其他 pipeline 可能返回 PIL Image 列表
            export_to_video(output_video, str(output_path), fps=fps)
        
        return {output_key: str(output_path)}

    return {output_key: output_video}


# eval function need finish
def eval_func(
    input_data_info: Dict[str, Any],
    eval_pipeline: None,
) -> Dict[str, Any]:
    """
    使用多模态 LLM 评估生成的文本到视频质量。
    
    Args:
        input_data_info: 单条测例字典，包含：
            - generation_text: 文本提示词
            - generated_video_path: 生成的视频路径（从 reference_results 传入）
            - eval_prompt: 评估提示词文本（从 data_info 传入）
        eval_pipeline: 已初始化的评估用 MLLM pipeline（可选）
    
    Returns:
        包含评估结果的字典：
        {
            'sample_id': str,
            'generated_video_path': str,
            'scores': {
                'text_video_alignment': float,
                'visual_quality': float,
                'temporal_consistency': float,
                'content_relevance': float,
                'motion_naturalness': float,
                'overall': float
            },
            'comments': str,
            'raw_response': str  # LLM 的原始响应
        }
    """
    generated_video_path = input_data_info.get("generated_video_path")
    if not generated_video_path:
        raise ValueError("generated_video_path not found in input_data_info")
    
    prompt_text = input_data_info.get("eval_prompt")
    if not prompt_text:
        raise ValueError("eval_prompt text not found in input_data_info")    
    if not isinstance(prompt_text, str):
        raise ValueError(f"eval_prompt should be a string, got {type(prompt_text)}")

    
    try:
        response = eval_pipeline(
            text=prompt_text,
            videos=[generated_video_path],  # 生成的视频
            max_new_tokens=1024
        )
        
        # response 可能是字符串或列表，统一处理
        if isinstance(response, list):
            response_text = response[0] if response else ""
        else:
            response_text = str(response)
        
    except Exception as e:
        return {
            'sample_id': input_data_info.get('id', 'unknown'),
            'generated_video_path': generated_video_path,
            'error': f"Evaluation failed: {str(e)}"
        }
    
    scores = _parse_evaluation_scores(response_text)
    
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
    [Text-Video Alignment Score]: <number>
    [Visual Quality Score]: <number>
    [Temporal Consistency Score]: <number>
    [Content Relevance Score]: <number>
    [Motion Naturalness Score]: <number>
    [Overall Score]: <float>
    [Comments]: <string>
    """
    scores = {}
    
    # 定义正则表达式模式
    patterns = {
        'text_video_alignment': r'\[Text-Video Alignment Score\]:\s*(\d+(?:\.\d+)?)',
        'visual_quality': r'\[Visual Quality Score\]:\s*(\d+(?:\.\d+)?)',
        'temporal_consistency': r'\[Temporal Consistency Score\]:\s*(\d+(?:\.\d+)?)',
        'content_relevance': r'\[Content Relevance Score\]:\s*(\d+(?:\.\d+)?)',
        'motion_naturalness': r'\[Motion Naturalness Score\]:\s*(\d+(?:\.\d+)?)',
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
    for key in ['text_video_alignment', 'visual_quality', 'temporal_consistency', 
                'content_relevance', 'motion_naturalness']:
        if scores.get(key) is not None:
            scores[key] = max(1, min(10, scores[key]))
    
    if scores.get('overall') is not None:
        scores['overall'] = max(1.0, min(10.0, scores['overall']))
    
    return scores
