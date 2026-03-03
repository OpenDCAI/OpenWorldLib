import sys
from pathlib import Path
sys.path.append("..")

from src.sceneflow.pipelines.depth_anything.pipeline_depth_anything_v1 import (  
    DepthAnything1Pipeline,
)

# Configure before running
DATA_TYPE = "image"  # or "video"
DATA_PATH = "../data/test_case1"
MODEL_PATH = "LiheYoung/depth_anything_vitl14"

ENCODER = "vitl"  # 'vits', 'vitb', or 'vitl'
OUTPUT_DIR = None 
GRAYSCALE = False  # True outputs grayscale image, False outputs color heat map (Only used for image mode)


pipeline = DepthAnything1Pipeline.from_pretrained(
    pretrained_model_path=MODEL_PATH,
    encoder=ENCODER,
    data_type=DATA_TYPE,
)

results = pipeline(
    DATA_PATH,
    grayscale=GRAYSCALE,
)

results.save(OUTPUT_DIR)

