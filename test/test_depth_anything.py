import sys
from pathlib import Path
sys.path.append("..")

from src.sceneflow.pipelines.depth_anything.pipeline_depth_anything import (  
    DepthAnythingPipeline,
)

# Configure before running
DATA_TYPE = "image"  # or "video"
DATA_PATH = "/YOUR/IMAGE/OR/VIDEO/PATH" 
MODEL_PATH = "LiheYoung/depth_anything_vitl14"

ENCODER = "vitl"
OUTPUT_DIR = str(Path(__file__).parent / ("vis_depth" if DATA_TYPE == "image" else "vis_video_depth"))
GRAYSCALE = False  # # True outputs grayscale image, False outputs color heat map (Only used for image mode)

pipeline = DepthAnythingPipeline.from_pretrained(
    pretrained_model_path=None if MODEL_PATH == "/YOUR/MODEL/PATH" else MODEL_PATH,
    encoder=ENCODER,
    data_type=DATA_TYPE,
)

pipeline(
    DATA_PATH,
    outdir=OUTPUT_DIR,
    grayscale=GRAYSCALE,
)
