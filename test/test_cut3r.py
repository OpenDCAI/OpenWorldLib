import sys
from pathlib import Path

sys.path.append("..")

from src.sceneflow.pipelines.cut3r.pipeline_cut3r import CUT3RPipeline


DATA_PATH = "../data/test_case/test_image_seq_case1"
MODEL_NAME = "cut3r_224_linear_4"  # or "cut3r_512_dpt_4_64"

SIZE = 224
VIS_THRESHOLD = 1.5
OUTPUT_DIR = "./cut3r_output"

# Interaction sequence for camera control in the second stage.
# Keep None to use a default orbit; or set to a list like:
# ["move_left", "move_right", "zoom_in"].
INTERACTION = "move_left"

# Two-stage camera config for 3DGS rendering.
CAMERA_RADIUS = 4.0
CAMERA_YAW = 0.0
CAMERA_PITCH = 0.0

IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224


pipeline = CUT3RPipeline.from_pretrained(
    representation_path=MODEL_NAME,
    size=SIZE,
)

output_video_path = pipeline.run_two_stage_3dgs_video(
    data_path=DATA_PATH,
    interaction=INTERACTION,
    size=SIZE,
    vis_threshold=VIS_THRESHOLD,
    output_dir=OUTPUT_DIR,
    camera_radius=CAMERA_RADIUS,
    camera_yaw=CAMERA_YAW,
    camera_pitch=CAMERA_PITCH,
    image_width=IMAGE_WIDTH,
    image_height=IMAGE_HEIGHT,
)

print(f"Rendered CUT3R 3DGS video saved to: {output_video_path}")
