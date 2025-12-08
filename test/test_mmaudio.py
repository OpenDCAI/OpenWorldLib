import sys
sys.path.append("..") 
from src.sceneflow.pipelines.mmaudio.pipeline_mmaudio import MMAudioPipeline, MMAudioArgs
from src.sceneflow.synthesis.audio_generation.mmaudio.mmaudio.eval_utils import make_video
from pathlib import Path
import torchaudio
from loguru import logger


def save_audio_result(result, output_dir, prompt, skip_video_composite=False):
    """
    保存音频生成结果，可选合成视频
    
    Args:
        result: pipeline 返回的结果字典
        output_dir: 输出目录
        prompt: 文本提示（用于生成文件名）
        skip_video_composite: 是否跳过视频合成
    
    Returns:
        保存的文件路径字典
    """
    audio = result["audio"]
    sampling_rate = result["sampling_rate"]
    video_info = result.get("video_info")
    video_path_input = result.get("video_path_input")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    
    # 确定音频文件名
    if video_path_input is not None:
        video_stem = Path(video_path_input).stem
        audio_save_path = output_path / f"{video_stem}.flac"
    else:
        # 使用 prompt 生成文件名（安全处理）
        safe_filename = prompt.replace(' ', '_').replace('/', '_').replace('.', '')[:50]
        audio_save_path = output_path / f"{safe_filename}.flac"
    
    # 保存音频
    torchaudio.save(str(audio_save_path), audio, sampling_rate)
    logger.info(f"Audio saved to {audio_save_path}")
    
    # 合成视频（如果有视频输入且未跳过）
    if video_info is not None and video_path_input is not None and not skip_video_composite:
        video_stem = Path(video_path_input).stem
        video_save_path = output_path / f"{video_stem}.mp4"
        make_video(video_info, str(video_save_path), audio, sampling_rate=sampling_rate)
        logger.info(f"Video with audio saved to {video_save_path}")
    



# 视频路径（可选，如果不提供则为 text-to-audio 模式）则设置为None
video_path = "./data/test_case1/test_video.mp4"  
test_prompt = "A man plays guitar"
output_dir = "./output/mmaudio"

args = MMAudioArgs(
    variant='large_44k_v2', # 可选: 'small_16k', 'small_44k', 'medium_44k', 'large_44k', 'large_44k_v2'
    full_precision=False,
    num_steps=25,
    duration=8.0,
    cfg_strength=4.5,
    seed=42,
)

pipeline = MMAudioPipeline.from_pretrained(
    synthesis_args=args,
    device=None,  # 自动检测设备 (cuda/mps/cpu)
)


# mmaudio支持t2a和v2a下面分别是两种情况
if video_path and Path(video_path).exists():
    logger.info(f"Using video: {video_path}")
    result = pipeline(
        video=video_path,
        prompt=test_prompt,
        negative_prompt="",
        duration=5.0,
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
        duration=5.0,
        cfg_strength=4.5,
        num_steps=25,
        seed=42,
    )

save_audio_result(
    result=result,
    output_dir=output_dir,
    prompt=test_prompt,
    skip_video_composite=False  # 是否跳过视频合成
)
