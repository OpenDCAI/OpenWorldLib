from sceneflow.base_models.diffusion_model.video.wan_2p2.utils.utils import save_video
from sceneflow.pipelines.yume.pipeline_yume_5b import Yume5bPipeline


pretrained_model_path = "stdstu123/Yume-5B-720P"

prompt = "A fire-breathing dragon appeared."
caption = "First-person perspective. The camera pushes forward (W). The rotation direction of the camera remains stationary (·). Actual distance moved:4 at 100 meters per second. Angular change rate (turn speed):0. View rotation speed:0."
# Inference mode is inferred from inputs:
# - t2v: image_path is None and video_path is None
# - i2v: provide image_path
# - v2v: provide video_path
image_path = "./data/test_case1/ref_image.png"
video_path = None

prompt_schedule = [f"{caption}{prompt}"]
pipeline = Yume5bPipeline.from_pretrained(
    synthesis_model_path=pretrained_model_path,
    size=(1280, 704),
    prompt=prompt,
    num_euler_timesteps=4,
    sigma_shift=7.0,
    base_seed=43,
    device_id=0,
)

output_video = pipeline(
    prompt=prompt,
    image_path=image_path,
    video_path=video_path,
    prompt_schedule=prompt_schedule,
    rollout_steps=len(prompt_schedule),
    num_euler_timesteps=4,
    sigma_shift=7.0,
)

save_video(
    tensor=output_video[None],
    save_file="./yume5b_i2v_demo.mp4",
    fps=16,
    nrow=1,
    normalize=True,
    value_range=(-1, 1),
)
