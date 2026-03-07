from pathlib import Path
from PIL import Image


from sceneflow.pipelines.wan.pipeline_wan_2p2 import Wan2p2Pipeline
from sceneflow.base_models.diffusion_model.video.wan_2p2.utils.utils import save_video
from sceneflow.base_models.diffusion_model.video.wan_2p2.configs import WAN_CONFIGS


model_path: str = "Wan-AI/Wan2.2-TI2V-5B"

pipeline = Wan2p2Pipeline.from_pretrained(
    model_path=model_path,
    mode="ti2v-5B",
    device=0,
    rank=0,
)

image_path: str = "path to image"
images = Image.open(image_path).convert('RGB')

output_video = pipeline(
    prompt=(
        "Summer beach vacation style, a white cat wearing sunglasses "
        "sits on a surfboard..."
    ),
    images=images,
    size="1280*704",
)

save_file_path = "./wan_app_demo_output.mp4"
save_video(
    tensor=output_video[None],
    save_file=save_file_path,
    fps=WAN_CONFIGS[pipeline.mode].sample_fps,
    nrow=1,
    normalize=True,
    value_range=(-1, 1),
)