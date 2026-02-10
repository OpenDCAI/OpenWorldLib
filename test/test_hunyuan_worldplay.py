import os
from sceneflow.pipelines.hunyuan_world.pipeline_hunyuan_worldplay import HunyuanWorldPlayPipeline
from sceneflow.synthesis.visual_generation.hunyuan_world.hunyuan_worldplay.generate import save_video
from sceneflow.synthesis.visual_generation.hunyuan_world.hunyuan_worldplay.commons.infer_state import initialize_infer_state

# Initialize infer state with default parameters
initialize_infer_state(
    sage_blocks_range="0-53",
    use_sageattn=False,
    enable_torch_compile=False,
    use_fp8_gemm=False,
    quant_type="fp8-per-block",
    include_patterns="double_blocks",
    use_vae_parallel=False
)

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
    interaction_signal=interaction_signal
)

save_video_path = os.path.join(output_path, "hunyuan_worldplay_demo.mp4")
save_video(output.videos, save_video_path)
