from sceneflow.pipelines.vggt.pipeline_vggt import VGGTPipeline


# Configure before running
DATA_PATH = "./data/test_case1/ref_image.png"
MODEL_PATH = "facebook/VGGT-1B"
OUTPUT_DIR = "./output/vggt"
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

# Note: 生成的可以查看的图片也是depth结果，还是看看能否重建出colorful的结果
# Note: 流程上最好是先重建，然后支持移动或者camera-view的输入
