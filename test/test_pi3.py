import sys
sys.path.append("..")

from sceneflow.pipelines.pi3.pipeline_pi3 import Pi3Pipeline

# --- Test Pi3X (multimodal, recommended) ---
DATA_PATH = "../data/test_case1/ref_image.png"
MODEL_PATH = "yyfz233/Pi3X"
OUTPUT_DIR = "output_pi3x"

pipeline = Pi3Pipeline.from_pretrained(
    representation_path=MODEL_PATH,
    model_type="pi3x",
)

results = pipeline(
    DATA_PATH,
    interaction="point_cloud_generation",
)

results.save(OUTPUT_DIR)

# --- Test Pi3 ---
MODEL_PATH_PI3 = "yyfz233/Pi3"
OUTPUT_DIR_PI3 = "output_pi3"

pipeline_pi3 = Pi3Pipeline.from_pretrained(
    representation_path=MODEL_PATH_PI3,
    model_type="pi3",
)

results_pi3 = pipeline_pi3(
    DATA_PATH,
    interaction="point_cloud_generation",
)

results_pi3.save(OUTPUT_DIR_PI3)
