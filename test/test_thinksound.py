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
    audio = result["audio"]  # [batch_size, channels, seq_len] int16，与原版 predict.py 保持一致
    sampling_rate = result["sampling_rate"]
    audio_id = result.get("id", "demo")
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 取第一个样本，并转换为 float32 + 归一化到 [-1, 1]
    # 原版 ThinkSound 约定 audio 维度为 [batch, channels, num_samples]
    waveform = audio[0].to(torch.float32) / 32767.0  # [channels, num_samples]

    # 保存音频（torchaudio.save 期望 [channels, num_frames]）
    save_path = output_path / f"{audio_id}.wav"
    torchaudio.save(str(save_path), waveform, sampling_rate)
    logger.info(f"Audio saved to {save_path}")
    
    return str(save_path)


video_path = "/data0/hdl/sceneflow/SceneFlow/data/test_case1/test_video.mp4"
title = "play guitar"
description = "A man is playing guitar gently"

model_config = "/data0/hdl/sceneflow/SceneFlow/src/sceneflow/synthesis/audio_generation/thinksound/ThinkSound/ThinkSound/configs/model_configs/thinksound.json"
ckpt_dir = "ckpts/thinksound_light.ckpt"
pretransform_ckpt_path = "ckpts/vae.ckpt"
output_dir = "./output/thinksound"

args = ThinkSoundArgs(
    model_config=model_config,
    ckpt_dir=ckpt_dir,
    pretransform_ckpt_path=pretransform_ckpt_path,
    duration_sec=8.0,
    seed=42,
    compile=False,
    video_dir="videos",
    cot_dir="cot_coarse",
    results_dir="results",
    scripts_dir=".",
)


pipeline = ThinkSoundPipeline.from_pretrained(
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

