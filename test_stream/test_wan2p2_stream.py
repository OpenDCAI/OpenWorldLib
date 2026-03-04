from pathlib import Path
from typing import Optional

import imageio
from PIL import Image

from sceneflow.pipelines.wan.pipeline_wan_2p2 import Wan2p2Pipeline
from sceneflow.base_models.diffusion_model.video.wan_2p2.configs import WAN_CONFIGS


pretrained_model_path: str = "Wan-AI/Wan2.2-TI2V-5B"

pipeline = Wan2p2Pipeline.from_pretrained(
    synthesis_model_path=pretrained_model_path,
    task="ti2v-5B",
    device_id=0,
    rank=0,
)

save_file = "./wan2p2_interactive_output.mp4"

cfg = WAN_CONFIGS[pipeline.task]
pipeline.memory_module.manage(action="reset")

default_prompt = (
    "Summer beach vacation style, a white cat wearing sunglasses "
    "sits on a surfboard..."
)
user_prompt = input(
    f"Please input prompt (press Enter to use default)\n"
    f"Default: {default_prompt}\n> "
).strip()
if not user_prompt:
    user_prompt = default_prompt

turn_idx = 0
out_dir = Path("./wan2p2_interactive_frames")
out_dir.mkdir(parents=True, exist_ok=True)

print("\n--- Wan2p2 Interactive Generation Started ---")
print("Each round will generate a video, and the last frame of the video will be used as the starting image for the next round.")
print("Input 'q' / 'quit' / 'n' to end and export the final video.\n")

last_frame_img: Optional[Image.Image] = None

while True:
    print(f"\n[Turn {turn_idx}] Use prompt: {user_prompt}")

    if last_frame_img is None:
        image_path = ""  # Empty string for initial generation
        print("  This is the initial generation")
    else:
        image_path = None
        print("  This round continues from the last frame of the previous round (memory image)")

    video = pipeline.stream(
        prompt=user_prompt,
        image_path=image_path,
        image=last_frame_img,
    )

    # 从 memory 中取出当前“最后一帧”，保留在内存中供下一轮使用（如需可选落
    last_frame_img = pipeline.memory_module.select()

    next_prompt = input(
        "\nGeneration completed. Input new prompt to continue;"
        "Input 'q' / 'quit' / 'n' to end and export the final video.\n> "
    ).strip()
    if next_prompt.lower() in ("q", "quit", "n"):
        break
    if next_prompt:
        user_prompt = next_prompt
    # 如果用户只是回车，则沿用上一轮的 prompt

    turn_idx += 1

all_frames = getattr(pipeline.memory_module, "all_frames", [])
if not all_frames:
    print("\nNo video segments generated, exiting.")
    exit()

print("\nStarting to export the final video based on all frames in memory...")

save_path = save_file
imageio.mimsave(
    save_path,
    all_frames,
    fps=cfg.sample_fps,
)
print(f"Interactive generation ended, saved to: {save_path}")
