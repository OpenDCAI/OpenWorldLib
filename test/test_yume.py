import torch
import torch.distributed as dist
from sceneflow.base_models.diffusion_model.video.wan_2p2.utils.utils import save_video
from sceneflow.pipelines.yume.pipeline_yume import YumePipeline


pretrained_model_path = "stdstu123/Yume-5B-720P"  # or "stdstu123/Yume-I2V-540P"
image_path = "./data/test_case1/ref_image.png"  # optional, set None for pure t2v
video_path = None  # optional, set a path for v2v
prompt = "A fire-breathing dragon appeared."
caption = (
    "First-person perspective. The camera pushes forward (W). "
    "The rotation direction of the camera remains stationary (.). "
    "Actual distance moved:4 at 100 meters per second. "
    "Angular change rate (turn speed):0. View rotation speed:0."
)
seed = 43
size = None  # use model defaults
sampling_method = "ode"  # "ode" or "sde"


pipeline = YumePipeline.from_pretrained(
    synthesis_model_path=pretrained_model_path,
)

output_video = pipeline(
    prompt=prompt,
    caption=caption,
    image_path=image_path,
    video_path=video_path,
    size=size,
    seed=seed,
    sampling_method=sampling_method,
)

is_main_process = (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0
if is_main_process and isinstance(output_video, torch.Tensor):
    save_video(
        tensor=output_video[None],
        save_file="./yume_demo.mp4",
        fps=16,
        nrow=1,
        normalize=True,
        value_range=(-1, 1),
    )

if dist.is_available() and dist.is_initialized():
    dist.barrier()
    dist.destroy_process_group()
