import argparse
import gc
import json
import os

import numpy as np
from diffusers.utils import export_to_video
from PIL import Image

from openworldlib.pipelines.lingbot_video.pipeline_lingbot_video import LingBotVideoPipeline
from openworldlib.reasoning.visual_reasoning.lingbot_video import LingBotVideoReasoning


DEFAULT_PROMPT = "A robot arm carefully places a red cube onto a wooden table."
DEFAULT_IMAGE = "./data/test_case/test_image_case1/ref_image.png"
DEFAULT_OUTPUT_DIR = "./outputs/lingbot_video"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LingBot-Video integration tests.")
    parser.add_argument("--mode", default=os.environ.get("LINGBOT_VIDEO_TEST_MODE", "all"))
    parser.add_argument("--model-path", default=os.environ.get("LINGBOT_VIDEO_MODEL_PATH", ""))
    parser.add_argument(
        "--rewriter-base",
        default=os.environ.get(
            "LINGBOT_VIDEO_REWRITER_BASE_MODEL",
            os.environ.get("REWRITER_BASE_MODEL", ""),
        ),
    )
    parser.add_argument(
        "--rewriter-adapter",
        default=os.environ.get(
            "LINGBOT_VIDEO_REWRITER_ADAPTER",
            os.environ.get("REWRITER_ADAPTER", ""),
        ),
    )
    parser.add_argument("--image", default=os.environ.get("LINGBOT_VIDEO_IMAGE", DEFAULT_IMAGE))
    parser.add_argument("--output-dir", default=os.environ.get("LINGBOT_VIDEO_OUTPUT_DIR", DEFAULT_OUTPUT_DIR))
    parser.add_argument("--prompt", default=os.environ.get("LINGBOT_VIDEO_PROMPT", DEFAULT_PROMPT))
    parser.add_argument("--duration", type=float, default=float(os.environ.get("LINGBOT_VIDEO_DURATION", "5")))
    parser.add_argument("--fps", type=int, default=int(os.environ.get("LINGBOT_VIDEO_FPS", "24")))
    parser.add_argument("--height", type=int, default=int(os.environ.get("LINGBOT_VIDEO_HEIGHT", "480")))
    parser.add_argument("--width", type=int, default=int(os.environ.get("LINGBOT_VIDEO_WIDTH", "832")))
    parser.add_argument("--steps", type=int, default=int(os.environ.get("LINGBOT_VIDEO_STEPS", "40")))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("LINGBOT_VIDEO_SEED", "42")))
    return parser.parse_args()


def _require_model_path(args: argparse.Namespace) -> str:
    if not args.model_path:
        raise RuntimeError("Set LINGBOT_VIDEO_MODEL_PATH to run LingBot-Video integration tests.")
    os.makedirs(args.output_dir, exist_ok=True)
    return args.model_path


def _require_rewriter(args: argparse.Namespace) -> LingBotVideoReasoning:
    if not args.rewriter_base or not args.rewriter_adapter:
        raise RuntimeError(
            "Set LINGBOT_VIDEO_REWRITER_BASE_MODEL/REWRITER_BASE_MODEL and "
            "LINGBOT_VIDEO_REWRITER_ADAPTER/REWRITER_ADAPTER to run LingBot-Video tests."
        )
    return LingBotVideoReasoning.from_pretrained(args.rewriter_base, args.rewriter_adapter)


def _clear_memory() -> None:
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _num_frames_from_duration(duration: float, fps: int) -> int:
    frame_count = int(float(duration) * int(fps))
    return ((frame_count - 1) // 4 + 1) * 4 + 1


def _caption_from_sample(sample: dict) -> str:
    if "caption" in sample:
        caption = sample["caption"]
    else:
        runtime_keys = {"duration", "fps", "height", "width", "num_frames", "resolution", "ratio"}
        caption = {key: value for key, value in sample.items() if key not in runtime_keys}
    if isinstance(caption, (dict, list)):
        return json.dumps(caption, ensure_ascii=False, separators=(",", ":"))
    return str(caption)


def _rewrite_prompt(
    rewriter: LingBotVideoReasoning,
    mode: str,
    prompt: str,
    fps: int,
    image=None,
    duration: float = 5.0,
) -> dict:
    reasoning = rewriter.inference(prompt, mode=mode, image=image, duration=duration)
    prompt_sample = {
        "caption": reasoning["structured_prompt"],
        "duration": None if mode == "t2i" else int(round(duration)),
    }
    return {
        "prompt_sample": prompt_sample,
        "prompt": _caption_from_sample(prompt_sample),
        "num_frames": _num_frames_from_duration(prompt_sample["duration"], fps)
        if prompt_sample["duration"] is not None
        else None,
    }


def _save_result(result, mode: str, output_dir: str, fps: int = 24) -> None:
    frames = result.frames[0] if hasattr(result, "frames") else result
    if isinstance(frames, np.ndarray) and frames.ndim == 5:
        frames = frames[0]
    if isinstance(frames, list) and len(frames) == 1 and isinstance(frames[0], list):
        frames = frames[0]
    filename = f"lingbot_video_{mode}.png" if mode == "t2i" else f"lingbot_video_{mode}.mp4"
    output_path = os.path.join(output_dir, filename)
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


def run_mode(mode: str, args: argparse.Namespace) -> None:
    normalized_mode = "ti2v" if mode in {"i2v", "ti2v"} else mode
    image = None
    if normalized_mode == "ti2v":
        with Image.open(args.image) as source_image:
            image = source_image.convert("RGB")

    rewriter = _require_rewriter(args)
    rewritten = _rewrite_prompt(
        rewriter,
        normalized_mode,
        prompt=args.prompt,
        fps=args.fps,
        image=image,
        duration=args.duration,
    )
    rewriter = None
    _clear_memory()

    pipe = LingBotVideoPipeline.from_pretrained(_require_model_path(args), mode=mode)
    call_kwargs = dict(
        prompt=rewritten["prompt"],
        height=args.height,
        width=args.width,
        steps=args.steps,
        seed=args.seed,
    )
    if mode != "t2i":
        call_kwargs["num_frames"] = rewritten["num_frames"]
    if normalized_mode == "ti2v":
        call_kwargs["images"] = image

    result = pipe(**call_kwargs)
    _save_result(result, mode, args.output_dir, args.fps)


if __name__ == "__main__":
    args = _parse_args()
    mode = args.mode.lower()
    valid_modes = {"all", "t2i", "t2v", "i2v", "ti2v"}
    if mode not in valid_modes:
        raise ValueError(f"Unsupported LingBot-Video test mode: {mode}")
    for current_mode in ("t2i", "t2v", "i2v") if mode == "all" else (mode,):
        run_mode("i2v" if current_mode == "ti2v" else current_mode, args)
