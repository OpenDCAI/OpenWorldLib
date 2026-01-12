from diffusers.utils import export_to_video
from PIL import Image
from sceneflow.pipelines.matrix_game.pipeline_matrix_game_2 import MatrixGame2Pipeline
import torch


image_path = "./data/test_case1/ref_image.png"
input_image = Image.open(image_path).convert('RGB')

pretrained_model_path = "Skywork/Matrix-Game-2.0"
pipeline = MatrixGame2Pipeline.from_pretrained(
    synthesis_model_path=pretrained_model_path,
    mode="universal",
    device="cuda"
)

interaction_signals_per_turn = [
    ["forward"],
    ["forward", "left"],
    ["forward"],
    ["forward", "right"],
    ["forward"],
    ["camera_l"],
    ["forward"],
    ["camera_r"],
    ["forward_left"],
    ["forward_right"],
]

num_frames_per_turns = [6, 12, 9, 12, 6, 6, 6, 6, 9, 12]

print("--- Interactive Stream Started ---")
all_frames = []

# 5. 循环执行生成 (不再使用 generator/send)
for turn_idx in range(len(interaction_signals_per_turn)):
    current_signal = interaction_signals_per_turn[turn_idx]
    print(f"Processing turn {turn_idx} with signals: {current_signal}")

    start_img = input_image if turn_idx == 0 else None

    video_output = pipeline.stream(
        interaction_signal=current_signal,
        initial_image=start_img,  # 仅第一轮非空
        num_output_frames=num_frames_per_turns[turn_idx],
        resize_H=352,
        resize_W=640,
        operation_visualization=False
    )

    all_frames.extend(video_output)

print(f"Total frames generated: {len(all_frames)}")

export_to_video(all_frames, "matrix_game_2_demo.mp4", fps=12)
