import json
import os
from pathlib import Path

import imageio
import numpy as np
import torch

from openworldlib.pipelines.longlive.pipeline_longlive import LongLivePipeline


DEFAULT_PROMPTS = [
    "A little boy in a blue T-shirt stands on a lush green lawn, arms relaxed at his sides, smiling at the camera, natural fresh realistic style.",
    "A little boy in a blue T-shirt starts to step forward, lightly running, with his arms swinging naturally. The grass and blue sky remain unchanged.",
    "A little boy in a blue T-shirt runs faster, leaning slightly forward, arms swinging naturally, feet lightly off the ground. The background remains the same grassy lawn.",
    "A little boy in a blue T-shirt jumps into the air, knees bent, arms raised, performing a light jump. The grass and sky stay consistent.",
    "A little boy in a blue T-shirt lands and continues running forward, arms swinging naturally, smiling as he enjoys the run. The background remains the same.",
    "A little boy in a blue T-shirt stops running, hands on his hips with a smile, body leaning slightly forward, sunlight falling on the grass, natural fresh realistic style.",
]


def load_longlive_example_prompts():
    example_path = Path("LongLive/example/interactive_example.jsonl")
    if not example_path.exists():
        return DEFAULT_PROMPTS

    with example_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            prompts = item.get("prompts")
            if isinstance(prompts, list) and len(prompts) > 0:
                return prompts

    return DEFAULT_PROMPTS


def save_uint8_video(video_frames, output_path, fps=16):
    if isinstance(video_frames, torch.Tensor):
        video_frames = video_frames.detach().cpu()
        if video_frames.ndim == 5:
            video_frames = video_frames[0]
        video_frames = video_frames.numpy()

    with imageio.get_writer(output_path, fps=fps, quality=8) as writer:
        for frame in video_frames:
            frame = np.asarray(frame)
            if frame.dtype != np.uint8:
                frame = np.clip(frame, 0, 255).astype(np.uint8)
            writer.append_data(frame)


pretrained_model_path = os.environ.get("LONGLIVE_MODEL_PATH", "checkpoints/LongLive")
wan_model_path = os.environ.get("LONGLIVE_WAN_MODEL_PATH", "checkpoints/Wan2.1-T2V-1.3B")
output_path = os.environ.get("LONGLIVE_OUTPUT_PATH", "longlive_demo.mp4")

prompts = load_longlive_example_prompts()
num_frames = int(os.environ.get("LONGLIVE_NUM_FRAMES", "120"))
fps = int(os.environ.get("LONGLIVE_FPS", "16"))

pipeline = LongLivePipeline.from_pretrained(
    model_path=pretrained_model_path,
    required_components={
        "wan_model_path": wan_model_path,
    },
    device="cuda",
)

output_video = pipeline(
    prompts=prompts,
    num_frames=num_frames,
    seed=1,
)

save_uint8_video(output_video, output_path, fps=fps)
print(f"Done! Video saved to {output_path}")
