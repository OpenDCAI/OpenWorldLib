from diffusers.utils import export_to_video
from PIL import Image
from sceneflow.pipelines.matrix_game.pipeline_matrix_game_2 import MatrixGame2Pipeline
import torch


image_path = "./data/test_case1/ref_image.png"
input_image = Image.open(image_path).convert('RGB')

pretrained_model_path = "Skywork/Matrix-Game-2.0"
pipeline = MatrixGame2Pipeline.from_pretrained(
    model_path=pretrained_model_path,
    mode="universal",
    device="cuda"
)

AVAILABLE_INTERACTIONS = ["forward", "left", "right", "forward_left", "forward_right", "camera_l", "camera_r"]

print("Available interactions:")
for i, interaction in enumerate(AVAILABLE_INTERACTIONS):
    print(f"  {i + 1}. {interaction}")
print("Tips:")
print("  - You can input multiple interactions separated by comma (e.g., 'forward,left')")
print("  - Input 'n' or 'q' to stop and export video")

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

    start_img = input_image if turn_idx == 0 else None

    video_output = pipeline.stream(
        images=start_img,  # only the first turn uses the input image, subsequent turns use the last generated frame
        interactions=current_signal,
        num_frames=num_frames,
        size = (352, 640),
        visualize_ops=False
    )

    turn_idx += 1
    print(f"Frames generated in this turn: {len(video_output)}, Total frames: {len(pipeline.memory_module.all_frames)}")

print(f"Total frames generated: {len(pipeline.memory_module.all_frames)}")

export_to_video(pipeline.memory_module.all_frames, "matrix_game_2_demo.mp4", fps=12)
