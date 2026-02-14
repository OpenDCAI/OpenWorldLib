import logging
import imageio
import warnings
from pathlib import Path

import torch

from sceneflow.pipelines.cosmos.pipeline_cosmos_predict2p5 import CosmosPredict2p5Pipeline

warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("transformers").setLevel(logging.ERROR)


"""
Huggingface token is required to download cosmos-series model
You can also skip download by specifying local ckpts path
"""
token = "hf_xxxxxxxxxxxxxx"
transformer_name_or_path = "nvidia/Cosmos-Predict2.5"
text_encoder_name_or_path = "nvidia/Cosmos-Reason1-7B"
vae_name_or_path = "Wan-AI/Wan2.1-T2V-1.3B"

pipeline = CosmosPredict2p5Pipeline.from_pretrained(
    transformer_name_or_path=transformer_name_or_path,
    text_encoder_name_or_path=text_encoder_name_or_path,
    vae_name_or_path=vae_name_or_path,
    token=token,
    task="img2world",
    device="cuda",
    dtype=torch.bfloat16
)

# Set default negative prompt
pipeline.set_negative_prompt()

pipeline.memory_module.manage(action="reset")

default_prompt = (
    "A robotic arm, primarily white with black joints and cables, is shown in a clean, modern indoor "
    "setting with a white tabletop. The arm, equipped with a gripper holding a small, light green pitcher, "
    "is positioned above a clear glass containing a reddish-brown liquid and a spoon. The robotic arm is "
    "in the process of pouring a transparent liquid into the glass. To the left of the pitcher, there is "
    "an opened jar with a similar reddish-brown substance visible through its transparent body. In the background, "
    "a vase with white flowers and a brown couch are partially visible, adding to the contemporary ambiance. The "
    "lighting is bright, casting soft shadows on the table. The robotic arm's movements are smooth and controlled, "
    "demonstrating precision in its task. As the video progresses, the robotic arm completes the pour, leaving the "
    "glass half-filled with the reddish-brown liquid. The jar remains untouched throughout the sequence, and the "
    "spoon inside the glass remains stationary. The other robotic arm on the right side also stays stationary "
    "throughout the video. The final frame captures the robotic arm with the pitcher finishing the pour, with the "
    "glass now filled to a higher level, while the pitcher is slightly tilted but still held securely by the gripper."
)
default_image_path = "data/test_case2/robot_arm.png"
user_prompt = input(
    f"Please input prompt (press Enter to use default)\n"
    f"Default: {default_prompt}\n> "
).strip()
if not user_prompt:
    user_prompt = default_prompt

turn_idx = 0

print("\n--- Cosmos-Predict2.5 Interactive Generation Started ---")
print("Each round will generate a video, and the last frame of the video will be used as the starting image for the next round.")
print("Input 'q' / 'quit' / 'n' to end and export the final video.\n")

last_frame_img = None

while True:
    print(f"\n[Turn {turn_idx}] Use prompt: {user_prompt}")

    if last_frame_img is None:
        image_path = default_image_path 
        print("  This is the initial generation")
    else:
        image_path = None
        print("  This round continues from the last frame of the previous round (memory image)")

    video = pipeline.stream(
        prompt=user_prompt,
        image=last_frame_img,
        image_path=image_path,
        output_type='pt',  # Optional[str] = 'pt', 'pil', 'np' ...
        num_inference_steps=1,
    )

    last_frame_img = pipeline.memory_module.select()

    next_prompt = input(
        "\nGeneration completed. Input new prompt to continue;"
        "Input 'q' / 'quit' / 'n' to end and export the final video.\n> "
    ).strip()
    if next_prompt.lower() in ("q", "quit", "n"):
        break
    if next_prompt:
        user_prompt = next_prompt

    turn_idx += 1

all_frames = getattr(pipeline.memory_module, "all_frames", [])
if not all_frames:
    print("\nNo video segments generated, exiting.")
    exit()

print("\nStarting to export the final video based on all frames in memory...")

save_path = "data/test_case2/cosmos_predict2p5_stream.mp4"
imageio.mimsave(
    save_path,
    all_frames,
    fps=28,
)
print(f"Interactive generation ended, saved to: {pipeline.save_file}")
