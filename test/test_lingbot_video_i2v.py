import gc
import json
import os

import numpy as np
from diffusers.utils import export_to_video
from PIL import Image

from openworldlib.pipelines.lingbot_video.pipeline_lingbot_video import LingBotVideoPipeline
from openworldlib.reasoning.visual_reasoning.lingbot_video import LingBotVideoReasoning


model_path = os.environ.get("LINGBOT_VIDEO_MODEL_PATH", "Robbyant/lingbot-video-dense-1.3b")
backend = os.environ.get("LINGBOT_VIDEO_BACKEND", "diffusers")
rewriter_base = os.environ.get(
    "LINGBOT_VIDEO_REWRITER_BASE_MODEL",
    "Qwen/Qwen3.6-27B",
)
rewriter_adapter = os.environ.get(
    "LINGBOT_VIDEO_REWRITER_ADAPTER",
    "Robbyant/lingbot-video-rewriter-lora",
)
image_path = "./data/test_case/test_vla_image_case1/init_frame.png"
output_dir = "./outputs/lingbot_video"
prompt = "A white robotic arm smoothly pours liquid from a light green pitcher into a glass on a white table."
duration = 5.0
fps = 24
height = 480
width = 832
steps = 40
seed = 42
num_frames = ((int(duration * fps) - 1) // 4 + 1) * 4 + 1

os.makedirs(output_dir, exist_ok=True)

with Image.open(image_path) as source_image:
    image = source_image.convert("RGB")

reasoning = LingBotVideoReasoning.from_pretrained(rewriter_base, rewriter_adapter)
reasoning_result = reasoning.inference(prompt, mode="ti2v", image=image, duration=duration)
rewritten_prompt = json.dumps(reasoning_result["json"], ensure_ascii=False, separators=(",", ":"))
reasoning = None
gc.collect()

pipe = LingBotVideoPipeline.from_pretrained(
    model_path,
    mode="i2v",
    backend=backend,
    strict_backend=backend == "sglang",
)
result = pipe(
    prompt=rewritten_prompt,
    images=image,
    height=height,
    width=width,
    num_frames=num_frames,
    steps=steps,
    seed=seed,
)

frames = result.frames[0] if hasattr(result, "frames") else result
if isinstance(frames, np.ndarray) and frames.ndim == 5:
    frames = frames[0]

output_path = os.path.join(output_dir, "lingbot_video_i2v.mp4")
export_to_video(frames, output_path, fps=fps)
print(output_path)
