import sys
from pathlib import Path
sys.path.append("..")

from sceneflow.pipelines.infinite_vggt.pipeline_infinite_vggt import InfiniteVGGTPipeline

# Configure before running: HuggingFace repo or local dir with .pth
MODEL_PATH = "lch01/StreamVGGT"
DATA_PATH = "../data/test_case/test_image_seq_case1" #"/YOUR/IMAGE/DIR/OR/VIDEO/PATH"
OUTPUT_DIR = None

# Optional: set interaction (export_ply / export_glb / export_depth)
INTERACTION = None  # or e.g. "export_glb"

pipeline = InfiniteVGGTPipeline.from_pretrained(
    pretrained_model_path=MODEL_PATH,
)

if INTERACTION is not None:
    pipeline.operator.get_interaction(INTERACTION)

results = pipeline(DATA_PATH)
results.save(OUTPUT_DIR)
