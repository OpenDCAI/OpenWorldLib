import sys
sys.path.append("..")

from sceneflow.pipelines.vggt.pipeline_vggt import VGGTPipeline

# Configure before running
DATA_PATH = "../data/test_case1/ref_image.png"
MODEL_PATH = "facebook/VGGT-1B"
OUTPUT_DIR = "output"
INTERACTION = "single_view_reconstruction"  # Options: "single_view_reconstruction", "multi_view_reconstruction", "camera_pose_estimation", "depth_estimation", "point_cloud_generation", "point_tracking"



pipeline = VGGTPipeline.from_pretrained(
    representation_path=MODEL_PATH,
)

results = pipeline(
    DATA_PATH,
    interaction=INTERACTION,
    return_visualization=True,
)

results.save(OUTPUT_DIR)

