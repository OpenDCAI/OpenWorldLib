import os
from sceneflow.pipelines.hunyuan_world.pipeline_hunyuan_worldplay import HunyuanWorldPlayPipeline

image_path = "./data/test_case/test_image_seq_case1/image_0001.jpg"
prompt = "A paved pathway leads towards a stone arch bridge spanning a calm body of water."
interaction_signal = "w-10, right-10, d-11"
model_path = "tencent/HunyuanVideo-1.5"
action_ckpt = "tencent/HY-WorldPlay"

output_path = "./outputs"
os.makedirs(output_path, exist_ok=True)

pipeline = HunyuanWorldPlayPipeline.from_pretrained(
    synthesis_path=model_path,
    transformer_version="480p_i2v",
    action_ckpt=action_ckpt,
    enable_offloading=True,
    device="cuda"
)
output = pipeline(
    prompt=prompt,
    reference_image=image_path,
    interaction_signal=interaction_signal,
)

save_video_path = os.path.join(output_path, "hunyuan_worldplay_demo.mp4")
HunyuanWorldPlayPipeline.save_video(output.videos, save_video_path)
