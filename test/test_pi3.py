import sys
sys.path.append("..")

from sceneflow.pipelines.pi3.pipeline_pi3 import Pi3Pipeline

# Pi3X (multimodal, recommended)
DATA_PATH = "../data/test_case1/ref_image.png"
MODEL_PATH = "yyfz233/Pi3X"
OUTPUT_DIR = "output_pi3"
INTERACTION = "point_cloud_generation"

pipeline = Pi3Pipeline.from_pretrained(
    representation_path=MODEL_PATH,
    model_type="pi3x",
)

results = pipeline(
    DATA_PATH,
    interaction=INTERACTION,
)

results.save(OUTPUT_DIR)
