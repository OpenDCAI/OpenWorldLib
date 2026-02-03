from pathlib import Path


from sceneflow.pipelines.wan.pipeline_wan2p2 import Wan2p2Args, Wan2p2Pipeline
from sceneflow.base_models.diffusion_model.video.wan_2p2.utils.utils import save_video
from sceneflow.base_models.diffusion_model.video.wan_2p2.configs import WAN_CONFIGS


pretrained_model_path = "Wan2.2/Wan2.2-TI2V-5B"

args = Wan2p2Args(
    task="ti2v-5B",
    size="1280*704",
    # ckpt_dir=ckpt_dir,
    prompt=(
        "Summer beach vacation style, a white cat wearing sunglasses "
        "sits on a surfboard..."
    ),
    image="",
    save_file="./wan_app_demo_output.mp4",
    base_seed=42,
)

pipeline = Wan2p2Pipeline.from_pretrained(
    synthesis_model_path = pretrained_model_path,
    args=args,
    device_id=0,
    rank=0,
)


output_video = pipeline(
    prompt=pipeline.args.prompt,
    image_path=pipeline.args.image,
    save=True,
)

save_video(
    tensor=output_video[None],
    save_file=args.save_file,
    fps=WAN_CONFIGS[args.task].sample_fps,
    nrow=1,
    normalize=True,
    value_range=(-1, 1),
)


