from diffusers.utils import export_to_video
from PIL import Image
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import json


def reference_func(
    pipe,
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

    num_output_frames = int(input_data_info.get("num_output_frames", 150))

    output_video = pipe(
        input_image=input_image,
        num_output_frames=num_output_frames,
        interaction_signal=interaction_signal,
    )

    output_path = input_data_info.get("output_path", None)
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fps = int(input_data_info.get("fps", 12))
        export_to_video(output_video, str(output_path), fps=fps)
        return {output_key: str(output_path)}

    return {output_key: output_video}


# eval function need finish
def eval_func(
    pipe,
    input_data_info: Dict[str, Any]
) -> Dict[str, Any]:
    """
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

