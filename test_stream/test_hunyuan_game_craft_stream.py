from PIL import Image
from sceneflow.pipelines.hunyuan_world.pipeline_hunyuan_game_craft import HunyuanGameCraftPipeline
import torch
import imageio

image_path = "./data/test_case1/ref_image.png"
input_image = Image.open(image_path).convert('RGB')

pretrained_model_path = "tencent/Hunyuan-GameCraft-1.0"
pipeline = HunyuanGameCraftPipeline.from_pretrained(
    synthesis_model_path=pretrained_model_path,
    device="cuda",
    cpu_offload=False,
    seed=250160
)

AVAILABLE_INTERACTIONS = ["forward", "left", "right", "backward", "camera_l", "camera_r", "camera_up", "camera_down"]

print("Available interactions:")
for i, interaction in enumerate(AVAILABLE_INTERACTIONS):
    print(f"  {i + 1}. {interaction}")
print("Tips:")
print("  - Input interactions separated by comma (e.g., 'forward,left')")
print("  - Input speeds separated by comma (e.g., '0.2,0.3')")
print("  - Input 'n' or 'q' to stop and export video")

interaction_text_prompt = "A charming medieval village with cobblestone streets, thatched-roof houses."
interaction_positive_prompt = "Realistic, High-quality."
interaction_negative_prompt = (
    "overexposed, low quality, deformation, a poor composition, bad hands, bad teeth, bad eyes, "
    "bad limbs, distortion, blurring, text, subtitles, static, picture, black border."
)

collected_interactions = []
collected_speeds = []
turn_idx = 0

print("--- Interaction Input Started ---")
while True:
    interaction_input = input(f"\n[Turn {turn_idx}] Enter interaction(s) (or 'n'/'q' to stop): ").strip().lower()

    if interaction_input in ['n', 'q']:
        print("Stopping interaction input...")
        break

    current_signal = [s.strip() for s in interaction_input.split(',') if s.strip()]

    invalid_signals = [s for s in current_signal if s not in AVAILABLE_INTERACTIONS]
    if invalid_signals:
        print(f"Invalid interaction(s): {invalid_signals}")
        print(f"Please choose from: {AVAILABLE_INTERACTIONS}")
        continue

    if not current_signal:
        print("No valid interaction provided. Please try again.")
        continue

    speeds_input = input(f"[Turn {turn_idx}] Enter speed(s) for each interaction: ").strip()
    try:
        speeds = [float(s.strip()) for s in speeds_input.split(',') if s.strip()]
    except ValueError:
        print("Invalid speed values. Please enter numeric values separated by comma.")
        continue

    if len(speeds) != len(current_signal):
        print("Number of speeds must match number of interactions.")
        continue

    if any(speed < 0 or speed > 3 for speed in speeds):
        print("Speeds must be in [0, 3].")
        continue

    collected_interactions.extend(current_signal)
    collected_speeds.extend(speeds)

    turn_idx += 1
    print(f"Collected interactions so far: {collected_interactions}")
    print(f"Collected speeds so far: {collected_speeds}")

if not collected_interactions:
    raise ValueError("No interactions collected. Please provide at least one interaction.")

output_video = pipeline(
    input_image=input_image,
    interaction_signal=collected_interactions,
    interaction_speed=collected_speeds,
    interaction_text_prompt=interaction_text_prompt,
    interaction_positive_prompt=interaction_positive_prompt,
    interaction_negative_prompt=interaction_negative_prompt,
    output_H=704,
    output_W=1216,
)

if torch.distributed.is_available() and torch.distributed.is_initialized():
    if torch.distributed.get_rank() == 0:
        imageio.mimsave("hunyuan_game_craft_demo.mp4", output_video, fps=24, quality=8)
else:
    imageio.mimsave("hunyuan_game_craft_demo.mp4", output_video, fps=24, quality=8)
