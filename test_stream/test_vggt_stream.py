import sys
import os
import numpy as np
from diffusers.utils import export_to_video

sys.path.append("..")

from sceneflow.pipelines.vggt.pipeline_vggt import VGGTPipeline


DATA_PATH = "../data/test_case1/ref_image.png"
MODEL_PATH = "facebook/VGGT-1B"
OUTPUT_DIR = "./vggt_stream_output"

POINT_CONF_THRESHOLD = 0.2
RESOLUTION = 518
PREPROCESS_MODE = "crop"
IMAGE_WIDTH = 704
IMAGE_HEIGHT = 480
FPS = 12


pipeline = VGGTPipeline.from_pretrained(
    representation_path=MODEL_PATH,
)

recon_info = pipeline.reconstruct_ply(
    input_=DATA_PATH,
    ply_path=OUTPUT_DIR,
    interaction="point_cloud_generation",
    point_conf_threshold=POINT_CONF_THRESHOLD,
    resolution=RESOLUTION,
    preprocess_mode=PREPROCESS_MODE,
)

ply_path = recon_info["ply_path"]
camera_range = recon_info["camera_range"]
camera_cfg = dict(recon_info["default_camera"])

AVAILABLE_INTERACTIONS = [
    "move_left",
    "move_right",
    "move_up",
    "move_down",
    "zoom_in",
    "zoom_out",
    "rotate_left",
    "rotate_right",
]

print("Stage-1 reconstruction done.")
print(f"PLY saved to: {ply_path}")
print("Camera range:", camera_range)
print("Default camera:", camera_cfg)

print("Available interactions:")
for i, interaction in enumerate(AVAILABLE_INTERACTIONS):
    print(f"  {i + 1}. {interaction}")
print("Tips:")
print("  - You can input multiple interactions separated by comma (e.g., 'move_left,zoom_in')")
print("  - Input 'n' or 'q' to stop and export video")

all_frames = []
first_frame = pipeline.render_with_3dgs(
    ply_path=ply_path,
    camera_config=camera_cfg,
    image_width=IMAGE_WIDTH,
    image_height=IMAGE_HEIGHT,
)
all_frames.append(np.array(first_frame))

print("--- Interactive Stream Started ---")
turn_idx = 0

while True:
    interaction_input = input(f"\n[Turn {turn_idx}] Enter interaction(s) (or 'n'/'q' to stop): ").strip().lower()

    if interaction_input in ['n', 'q']:
        print("Stopping interaction loop...")
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

    try:
        frames_input = input(f"[Turn {turn_idx}] Enter number of frame units (e.g., '1' or '2'): ").strip()
        frame_units = int(frames_input)
        if frame_units <= 0:
            print("Frame units must be a positive integer. Please try again.")
            continue
        num_frames = frame_units * len(current_signal) * 6
    except ValueError:
        print("Invalid input. Please enter a valid integer.")
        continue

    print(f"Processing turn {turn_idx} with signals: {current_signal}, frames: {num_frames}")

    turn_frames = []
    for sig in current_signal:
        for _ in range(frame_units * 6):
            camera_cfg = pipeline.apply_interaction_to_camera(
                camera_cfg=camera_cfg,
                interaction=sig,
                camera_range=camera_range,
                yaw_step=2.0,
                pitch_step=1.5,
                zoom_factor=0.98,
            )
            frame = pipeline.render_with_3dgs(
                ply_path=ply_path,
                camera_config=camera_cfg,
                image_width=IMAGE_WIDTH,
                image_height=IMAGE_HEIGHT,
            )
            turn_frames.append(np.array(frame))

    all_frames.extend(turn_frames)
    turn_idx += 1
    print(f"Frames generated in this turn: {len(turn_frames)}, Total frames: {len(all_frames)}")

print(f"Total frames generated: {len(all_frames)}")
os.makedirs(OUTPUT_DIR, exist_ok=True)
output_video_path = os.path.join(OUTPUT_DIR, "vggt_stream_demo.mp4")
export_to_video(all_frames, output_video_path, fps=FPS)
print(f"Rendered VGGT stream video saved to: {output_video_path}")
