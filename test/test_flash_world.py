import sys
sys.path.append("..")

from sceneflow.pipelines.flash_world.pipeline_flash_world import FlashWorldPipeline

# Configure before running
MODEL_PATH = "imlixinyang/FlashWorld"

# Offload options (reduce GPU memory usage)
OFFLOAD_T5 = True  # Offload text encoder to CPU
OFFLOAD_VAE = False  # Offload VAE to CPU (greatly reduces GPU memory but increases generation time)
OFFLOAD_TRANSFORMER_DURING_VAE = True  # Offload transformer during VAE processing

# Default configuration 
TEXT_PROMPT = "A cozy medieval-style village square on a winter evening, with timber-framed cottages"
IMAGE_PATH = "../data/test_case1/ref_image.png"
NUM_FRAMES = 16
IMAGE_HEIGHT = 480
IMAGE_WIDTH = 704
IMAGE_INDEX = 0
RETURN_VIDEO = True
VIDEO_FPS = 15
OUTPUT_DIR = "./output/flash_world"

pipeline = FlashWorldPipeline.from_pretrained(
    representation_path=MODEL_PATH,
    offload_t5=OFFLOAD_T5,
    offload_vae=OFFLOAD_VAE,
    offload_transformer_during_vae=OFFLOAD_TRANSFORMER_DURING_VAE,
)


results = pipeline(
    input_=IMAGE_PATH,
    text_prompt=TEXT_PROMPT,
    num_frames=NUM_FRAMES,
    image_height=IMAGE_HEIGHT,
    image_width=IMAGE_WIDTH,
    image_index=IMAGE_INDEX,
    return_video=RETURN_VIDEO,
    video_fps=VIDEO_FPS,
)

pipeline.save_results(results=results, output_dir=OUTPUT_DIR)

