import sys
sys.path.append("..")

from sceneflow.pipelines.flash_world.pipeline_flash_world import FlashWorldPipeline

MODEL_PATH = "imlixinyang/FlashWorld"
WAN_MODEL_PATH = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
OFFLOAD_T5 = True
OFFLOAD_VAE = False
OFFLOAD_TRANSFORMER_DURING_VAE = True

TEXT_PROMPT = "A cozy medieval-style village square on a winter evening, with timber-framed cottages"
IMAGE_PATH = "../data/test_case1/ref_image.png"
NUM_FRAMES = 16
IMAGE_HEIGHT = 480
IMAGE_WIDTH = 704
IMAGE_INDEX = 0
RETURN_VIDEO = True
VIDEO_FPS = 15
OUTPUT_DIR = "./output/flash_world"

# Custom output paths - directory or file path, optional
CUSTOM_PLY_DIR = "./output/flash_world/custom_ply"
CUSTOM_SPZ_DIR = "./output/flash_world/custom_spz"
CUSTOM_VIDEO_DIR = "./output/flash_world/custom_video"

INTERACTIONS = ["camera_l", "forward"]

pipeline = FlashWorldPipeline.from_pretrained(
    model_path=MODEL_PATH,
    required_components={"wan_model_path": WAN_MODEL_PATH},
    offload_t5=OFFLOAD_T5,
    offload_vae=OFFLOAD_VAE,
    offload_transformer_during_vae=OFFLOAD_TRANSFORMER_DURING_VAE,
)

results = pipeline(
    images=IMAGE_PATH,
    prompt=TEXT_PROMPT,
    interactions=INTERACTIONS,
    num_frames=NUM_FRAMES,
    fps=VIDEO_FPS,
    image_height=IMAGE_HEIGHT,
    image_width=IMAGE_WIDTH,
    image_index=IMAGE_INDEX,
    return_video=RETURN_VIDEO,
)

pipeline.save_results(
    results=results, 
    output_dir=OUTPUT_DIR,
    ply_path=CUSTOM_PLY_DIR,
    spz_path=CUSTOM_SPZ_DIR,
    video_path=CUSTOM_VIDEO_DIR,
)

