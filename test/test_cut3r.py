import sys
from pathlib import Path
sys.path.append("..")

from sceneflow.pipelines.cut3r.pipeline_cut3r import (
    CUT3RPipeline,
)

# Configure before running
DATA_PATH = "../data/test_case1/ref_image.png"
# Model options:
# - "cut3r_224_linear_4": Linear model with 224 input size (faster, lower quality)
# - "cut3r_512_dpt_4_64": DPT model with 512 input size (slower, higher quality)
# - Or use HuggingFace repo ID or local path
MODEL_NAME = "cut3r_224_linear_4"  # or "cut3r_512_dpt_4_64"

# Inference parameters
SIZE = 224  # Input image size (None = auto-detect from model, or specify 224/512)
VIS_THRESHOLD = 1.5  # Confidence threshold for filtering point clouds (1.0 to INF, higher = more filtering)
INTERACTION = None  # "move_left", "move_right", "move_up", "move_down","zoom_in", "zoom_out"
OUTPUT_DIR = None


pipeline = CUT3RPipeline.from_pretrained(
    representation_path=MODEL_NAME,
    size=SIZE,  
)

results = pipeline(
    DATA_PATH,
    interaction=INTERACTION,
    size=SIZE,  
    vis_threshold=VIS_THRESHOLD,  # Filter point clouds by confidence
)

results.save(OUTPUT_DIR)

