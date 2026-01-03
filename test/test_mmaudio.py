import sys
from sceneflow.pipelines.mmaudio.pipeline_mmaudio import MMAudioPipeline, MMAudioArgs
from sceneflow.synthesis.audio_generation.mmaudio.mmaudio.eval_utils import make_video
from pathlib import Path
import torchaudio
from loguru import logger


def save_audio_result(result, skip_video_composite=False):
    audio = result["audio"]
    sampling_rate = result["sampling_rate"]
    video_info = result["video_info"]
    video_path_input = result["video_path_input"]
    
    audio_save_path = f"./mmaudio_testoutput.flac"
    torchaudio.save(str(audio_save_path), audio, sampling_rate)
    logger.info(f"Audio saved to {audio_save_path}")
    
    # 可选保存视频
    if video_info is not None and video_path_input is not None and not skip_video_composite:
        video_save_path = f"./mmaudio_testoutput.mp4"
        make_video(video_info, str(video_save_path), audio, sampling_rate=sampling_rate)
        logger.info(f"Video saved to {video_save_path}")

# 视频路径（可选，如果不提供则为 text-to-audio 模式）则设置为None
video_path = "./data/test_case1/test_video.mp4"  
test_prompt = "A man plays guitar."
pretrained_model_path = "hkchengrex/MMAudio" # 可以提供本地路径或者hugid路径

args = MMAudioArgs(
    variant='large_44k_v2', # 可选: 'small_16k', 'small_44k', 'medium_44k', 'large_44k', 'large_44k_v2'
    full_precision=False,
    num_steps=25,
    duration=8.0,
    cfg_strength=4.5,
    seed=42,
)

pipeline = MMAudioPipeline.from_pretrained(
    pretrained_model_path=pretrained_model_path,
    synthesis_args=args,
    device='cuda',  # 可以填写None则进入自动检测设备（cuda/mps/cpu），如果填写了则使用填写的设备
)

# mmaudio支持t2a和v2a下面分别是两种情况
if video_path and Path(video_path).exists():
    logger.info(f"Using video: {video_path}")
    result = pipeline(
        video=video_path,
        prompt=test_prompt,
        negative_prompt="",
        duration=8.0,
        cfg_strength=4.5,
        num_steps=25,
        seed=42,
        mask_away_clip=False,  # 是否屏蔽 CLIP 特征
    )
else:
    logger.info("Video not found or not provided, using text-to-audio mode")
    result = pipeline(
        video=None,
        prompt=test_prompt,
        negative_prompt="",
        duration=8.0,
        cfg_strength=4.5,
        num_steps=25,
        seed=42,
    )

save_audio_result(
    result=result,
    skip_video_composite=False  # 是否跳过视频合成
)
