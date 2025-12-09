import sys
sys.path.append("..")

from src.sceneflow.pipelines.flash_world.pipeline_flash_world import FlashWorldPipeline

# Configure before running
MODEL_PATH = "imlixinyang/FlashWorld"
JSON_CONFIG_PATH = "/data/zhukaixin/FlashWorld/examples" #None  # Path to JSON config file, or None to use defaults

# Offload options (reduce GPU memory usage)
OFFLOAD_T5 = True  # Offload text encoder to CPU
OFFLOAD_VAE = True  # Offload VAE to CPU (greatly reduces GPU memory but increases generation time)
OFFLOAD_TRANSFORMER_DURING_VAE = True  # Offload transformer during VAE processing

# Default configuration (used when JSON_CONFIG_PATH is None)
TEXT_PROMPT = ""
IMAGE_PATH = None
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

if JSON_CONFIG_PATH is not None:
    config = FlashWorldPipeline.load_config_from_json(JSON_CONFIG_PATH)
    results = pipeline(**config)
else:
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

