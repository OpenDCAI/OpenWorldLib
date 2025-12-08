import sys
sys.path.append("..") 
from src.sceneflow.pipelines.thinksound.pipeline_thinksound import ThinkSoundPipeline, ThinkSoundArgs
import torchaudio
import torch
from pathlib import Path
from loguru import logger


def save_audio_result(result, output_dir):
    """
    保存音频生成结果
    
    Args:
        result: pipeline 返回的结果字典
        output_dir: 输出目录
    
    Returns:
        保存的文件路径
    """
    audio = result["audio"]  
    sampling_rate = result["sampling_rate"]
    audio_id = result.get("id", "demo")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    waveform = audio[0]
    
    save_path = output_path / f"{audio_id}.wav"
    torchaudio.save(str(save_path), waveform, sampling_rate)
    logger.info(f"Audio saved to {save_path}")
    
    return str(save_path)

# thinksound不允许为none，duration-sec必须是匹配的
video_path = "/data0/hdl/sceneflow/SceneFlow/data/test_video_case1/talking_man.mp4"
title = "play guitar"
description = "A man is playing guitar gently"
output_dir = "./output/thinksound"
pretrained_model_path = "FunAudioLLM/ThinkSound"

args = ThinkSoundArgs(
    duration_sec=3.0,
    seed=42,
    compile=False,
    video_dir="videos",
    cot_dir="cot_coarse",
    results_dir="results",
    scripts_dir=".",
)


pipeline = ThinkSoundPipeline.from_pretrained(
    synthesis_model_path=pretrained_model_path,
    synthesis_args=args,
    device=None,  # 自动检测设备
)

result = pipeline(
    video_path=video_path,
    title=title,
    description=description,
    use_half=False,
    cfg_scale=5.0,
    num_steps=24,
)

save_path = save_audio_result(result, output_dir)

