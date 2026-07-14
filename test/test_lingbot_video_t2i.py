import gc
import json
import os

import numpy as np
from PIL import Image

from openworldlib.pipelines.lingbot_video.pipeline_lingbot_video import LingBotVideoPipeline
from openworldlib.reasoning.visual_reasoning.lingbot_video import LingBotVideoReasoning


model_path = os.environ.get("LINGBOT_VIDEO_MODEL_PATH", "Robbyant/lingbot-video-dense-1.3b")
rewriter_base = os.environ.get(
    "LINGBOT_VIDEO_REWRITER_BASE_MODEL",
    "Qwen/Qwen3.6-27B",
)
rewriter_adapter = os.environ.get(
    "LINGBOT_VIDEO_REWRITER_ADAPTER",
    "Robbyant/lingbot-video-rewriter-lora",
)
output_dir = "./outputs/lingbot_video"
prompt = "A robot arm carefully places a red cube onto a wooden table."
height = 480
width = 832
steps = 40
seed = 42

os.makedirs(output_dir, exist_ok=True)

reasoning = LingBotVideoReasoning.from_pretrained(rewriter_base, rewriter_adapter)
reasoning_result = reasoning.inference(prompt, mode="t2i")
rewritten_prompt = json.dumps(reasoning_result["structured_prompt"], ensure_ascii=False, separators=(",", ":"))
reasoning = None
gc.collect()

pipe = LingBotVideoPipeline.from_pretrained(model_path, mode="t2i")
result = pipe(prompt=rewritten_prompt, height=height, width=width, steps=steps, seed=seed)

frames = result.frames[0] if hasattr(result, "frames") else result
if isinstance(frames, np.ndarray) and frames.ndim == 5:
    frames = frames[0]
image = frames[0] if isinstance(frames, (list, tuple)) else frames
if isinstance(image, np.ndarray):
    while image.ndim > 3:
        image = image[0]
    if np.issubdtype(image.dtype, np.floating):
        image = (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)
    image = Image.fromarray(image)

output_path = os.path.join(output_dir, "lingbot_video_t2i.png")
image.save(output_path)
print(output_path)
