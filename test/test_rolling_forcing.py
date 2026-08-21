import os

import imageio
import numpy as np
import torch

from openworldlib.pipelines.rolling_forcing.pipeline_rolling_forcing import RollingForcingPipeline


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


pretrained_model_path = os.environ.get("ROLLING_FORCING_MODEL_PATH", "checkpoints/RollingForcing")
wan_model_path = os.environ.get("ROLLING_FORCING_WAN_MODEL_PATH", "checkpoints/Wan2.1-T2V-1.3B")
output_path = os.environ.get("ROLLING_FORCING_OUTPUT_PATH", "rolling_forcing_demo.mp4")
num_frames = int(os.environ.get("ROLLING_FORCING_NUM_FRAMES", "126"))
fps = int(os.environ.get("ROLLING_FORCING_FPS", "16"))

pipeline = RollingForcingPipeline.from_pretrained(
    model_path=pretrained_model_path,
    required_components={
        "wan_model_path": wan_model_path,
    },
    device="cuda",
)

output_video = pipeline(
    prompt="A cinematic tracking shot of a quiet futuristic city street at sunrise.",
    num_frames=num_frames,
    seed=0,
)

save_uint8_video(output_video, output_path, fps=fps)
print(f"Done! Video saved to {output_path}")
