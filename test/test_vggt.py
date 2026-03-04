import sys
sys.path.append("..")

from sceneflow.pipelines.vggt.pipeline_vggt import VGGTPipeline


DATA_PATH = "../data/test_case1/ref_image.png"
MODEL_PATH = "facebook/VGGT-1B"
OUTPUT_DIR = "./vggt_output"

# None -> default orbit video
# or e.g. ["move_left", "move_left", "zoom_in", "rotate_right"]
INTERACTION = "move_left"

POINT_CONF_THRESHOLD = 0.2
RESOLUTION = 518
PREPROCESS_MODE = "crop"
IMAGE_WIDTH = 704
IMAGE_HEIGHT = 480
FPS = 12


pipeline = VGGTPipeline.from_pretrained(
    representation_path=MODEL_PATH,
)

recon_info = pipeline.reconstruct_ply(
    input_=DATA_PATH,
    ply_path=OUTPUT_DIR,
    interaction="point_cloud_generation",
    point_conf_threshold=POINT_CONF_THRESHOLD,
    resolution=RESOLUTION,
    preprocess_mode=PREPROCESS_MODE,
)

print(f"PLY saved to: {recon_info['ply_path']}")
print("Camera range:", recon_info["camera_range"])
print("Default camera:", recon_info["default_camera"])

output_video_path = pipeline.run_stage2_3dgs_video_from_reconstruction(
    recon_info=recon_info,
    interaction=INTERACTION,
    output_dir=OUTPUT_DIR,
    image_width=IMAGE_WIDTH,
    image_height=IMAGE_HEIGHT,
    fps=FPS,
    output_name="vggt_3dgs_demo.mp4",
)

print(f"Rendered VGGT 3DGS video saved to: {output_video_path}")

