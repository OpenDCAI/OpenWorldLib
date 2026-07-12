import os

import numpy as np
from diffusers.utils import export_to_video
from PIL import Image

from openworldlib.pipelines.lingbot_video.pipeline_lingbot_video import LingBotVideoPipeline


MODEL_PATH = os.environ.get("LINGBOT_VIDEO_MODEL_PATH", "")
IMAGE_PATH = os.environ.get("LINGBOT_VIDEO_IMAGE", "./data/test_case/test_image_case1/ref_image.png")
OUTPUT_DIR = os.environ.get("LINGBOT_VIDEO_OUTPUT_DIR", "./outputs/lingbot_video")
PROMPT = os.environ.get(
    "LINGBOT_VIDEO_PROMPT",
    "A robot arm carefully places a red cube onto a wooden table.",
)


def _require_model_path() -> str:
    if not MODEL_PATH:
        raise RuntimeError("Set LINGBOT_VIDEO_MODEL_PATH to run LingBot-Video integration tests.")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return MODEL_PATH


def _save_result(result, mode: str, filename: str, fps: int = 24) -> None:
    frames = result.frames[0] if hasattr(result, "frames") else result
    if isinstance(frames, np.ndarray) and frames.ndim == 5:
        frames = frames[0]
    if isinstance(frames, list) and len(frames) == 1 and isinstance(frames[0], list):
        frames = frames[0]
    output_path = os.path.join(OUTPUT_DIR, filename)
    if mode == "t2i":
        image = frames[0] if isinstance(frames, (list, tuple)) else frames
        if isinstance(image, np.ndarray):
            while image.ndim > 3:
                image = image[0]
            if np.issubdtype(image.dtype, np.floating):
                image = (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)
            image = Image.fromarray(image)
        image.save(output_path)
    else:
        export_to_video(frames, output_path, fps=fps)
    print(f"Saved {mode}: {output_path}")


def run_t2i() -> None:
    pipe = LingBotVideoPipeline.from_pretrained(_require_model_path(), mode="t2i")
    result = pipe(prompt=PROMPT, height=480, width=832, steps=40, seed=42)
    _save_result(result, "t2i", "lingbot_video_t2i.png")


def run_t2v() -> None:
    pipe = LingBotVideoPipeline.from_pretrained(_require_model_path(), mode="t2v")
    result = pipe(prompt=PROMPT, height=480, width=832, num_frames=81, steps=40, seed=42)
    _save_result(result, "t2v", "lingbot_video_t2v.mp4")


def run_i2v() -> None:
    with Image.open(IMAGE_PATH) as source_image:
        image = source_image.convert("RGB")
    pipe = LingBotVideoPipeline.from_pretrained(_require_model_path(), mode="i2v")
    result = pipe(prompt=PROMPT, images=image, height=480, width=832, num_frames=81, steps=40, seed=42)
    _save_result(result, "i2v", "lingbot_video_i2v.mp4")


def run_i2v_stream() -> None:
    with Image.open(IMAGE_PATH) as source_image:
        image = source_image.convert("RGB")
    pipe = LingBotVideoPipeline.from_pretrained(_require_model_path(), mode="i2v")
    first_result = pipe.stream(
        prompt=PROMPT,
        images=image,
        height=480,
        width=832,
        num_frames=81,
        steps=40,
        seed=42,
    )
    _save_result(first_result, "i2v_stream", "lingbot_video_i2v_stream_1.mp4")
    second_result = pipe.stream(
        prompt=os.environ.get("LINGBOT_VIDEO_STREAM_PROMPT", PROMPT),
        height=480,
        width=832,
        num_frames=81,
        steps=40,
        seed=43,
    )
    _save_result(second_result, "i2v_stream", "lingbot_video_i2v_stream_2.mp4")


if __name__ == "__main__":
    mode = os.environ.get("LINGBOT_VIDEO_TEST_MODE", "all").lower()
    valid_modes = {"all", "t2i", "t2v", "i2v", "ti2v", "i2v_stream", "ti2v_stream", "stream"}
    if mode not in valid_modes:
        raise ValueError(f"Unsupported LINGBOT_VIDEO_TEST_MODE: {mode}")
    if mode in {"all", "t2i"}:
        run_t2i()
    if mode in {"all", "t2v"}:
        run_t2v()
    if mode in {"all", "i2v", "ti2v"}:
        run_i2v()
    if mode in {"all", "i2v_stream", "ti2v_stream", "stream"}:
        run_i2v_stream()
