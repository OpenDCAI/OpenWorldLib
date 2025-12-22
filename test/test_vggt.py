import sys
sys.path.append("..")

from src.sceneflow.pipelines.vggt.pipeline_vggt import VGGTPipeline

# Configure before running
DATA_PATH = "/YOUR/IMAGE/OR/DIRECTORY/PATH"
MODEL_PATH = "facebook/VGGT-1B"
OUTPUT_DIR = None


pipeline = VGGTPipeline.from_pretrained(
    representation_path=MODEL_PATH,
)

results = pipeline(
    DATA_PATH,
    return_visualization=True,
)

results.save(OUTPUT_DIR)

