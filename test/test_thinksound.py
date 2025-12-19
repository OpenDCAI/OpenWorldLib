import sys
from sceneflow.pipelines.thinksound.pipeline_thinksound import ThinkSoundPipeline, ThinkSoundArgs
import torchaudio
from pathlib import Path
from loguru import logger


def save_audio_result(result):
    audio = result["audio"]  
    sampling_rate = result["sampling_rate"]
    waveform = audio[0]

    save_path = f"./thinksound_testoutput.wav"
    torchaudio.save(str(save_path), waveform, sampling_rate)



# thinksound不允许为none，duration-sec必须是匹配的
video_path = "./data/test_video_case1/talking_man.mp4"
title = "play guitar"
description = "A man is playing guitar gently"
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

save_audio_result(result)

