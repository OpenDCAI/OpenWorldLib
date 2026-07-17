import gc
import json
import os

import numpy as np
from diffusers.utils import export_to_video
from PIL import Image

from openworldlib.pipelines.lingbot_video.pipeline_lingbot_video import LingBotVideoPipeline
from openworldlib.reasoning.general_reasoning.lingbot_video import LingBotVideoReasoning


mode = os.environ.get("LINGBOT_VIDEO_TEST_MODE", "t2v").lower()
if mode not in {"t2i", "t2v", "i2v"}:
    raise ValueError("LINGBOT_VIDEO_TEST_MODE must be one of: t2i, t2v, i2v")

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
prompt = (
    "A white robotic arm smoothly pours liquid from a light green pitcher into a glass on a white table."
    if mode == "i2v"
    else "A robot arm carefully places a red cube onto a wooden table."
)
duration = 5.0
fps = 24
height = 480
width = 832
steps = 40
seed = 42
num_frames = ((int(duration * fps) - 1) // 4 + 1) * 4 + 1

os.makedirs(output_dir, exist_ok=True)

image = None
if mode == "i2v":
    with Image.open(image_path) as source_image:
        image = source_image.convert("RGB")

reasoning_mode = "ti2v" if mode == "i2v" else mode
reasoning = LingBotVideoReasoning.from_pretrained(rewriter_base, rewriter_adapter)
reasoning_result = reasoning.inference(
    prompt,
    mode=reasoning_mode,
    image=image,
    duration=duration,
)
rewritten_prompt = json.dumps(reasoning_result["json"], ensure_ascii=False, separators=(",", ":"))
reasoning = None
gc.collect()

pipe = LingBotVideoPipeline.from_pretrained(
    model_path,
    mode=mode,
    backend=backend,
    strict_backend=backend == "sglang",
)
inference_kwargs = {
    "prompt": rewritten_prompt,
    "height": height,
    "width": width,
    "steps": steps,
    "seed": seed,
}
if mode != "t2i":
    inference_kwargs["num_frames"] = num_frames
if mode == "i2v":
    inference_kwargs["images"] = image
result = pipe(**inference_kwargs)

frames = result.frames[0] if hasattr(result, "frames") else result
if isinstance(frames, np.ndarray) and frames.ndim == 5:
    frames = frames[0]

output_path = os.path.join(output_dir, f"lingbot_video_{mode}.{'png' if mode == 't2i' else 'mp4'}")
if mode == "t2i":
    output_image = frames[0] if isinstance(frames, (list, tuple)) else frames
    if isinstance(output_image, np.ndarray):
        while output_image.ndim > 3:
            output_image = output_image[0]
        if np.issubdtype(output_image.dtype, np.floating):
            output_image = (np.clip(output_image, 0.0, 1.0) * 255).astype(np.uint8)
        output_image = Image.fromarray(output_image)
    output_image.save(output_path)
else:
    export_to_video(frames, output_path, fps=fps)
print(output_path)
