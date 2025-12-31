from pathlib import Path
from sceneflow.pipelines.wow.pipeline_wow import WoWPipeline, WoWArgs
from sceneflow.base_models.diffusion_model.diffsynth import save_video


def save_video_to_file(
    output_video, 
    output_path: str = './wow_output.mp4', 
    fps: int = 15, 
    quality: int = 5):

    save_video(output_video, output_path, fps=fps, quality=quality)
    print(f"Video saved to {output_path}")
    return output_path



input_path = "./data/test_case1/init_frame.png"  # 可以是图片或视频
text_prompt = "Put the screw driver into the drawer."
output_path = "./wow_output.mp4"

# 创建参数对象
args = WoWArgs(
    gpu=7,
    steps=50,
    seed=42,
    num_frames=81,
    no_tiled=False,
    enable_vram_management=True,
    no_vram_management=False,
)

# 模型路径（可以是本地路径或HuggingFace repo_id）
pretrained_model_path = "/data0/hdl/sceneflow/SceneFlow/ WoW-1-Wan-1.3B-2M"  # 注意：路径中有空格，如果路径不存在请去掉空格

# 创建pipeline
pipeline = WoWPipeline.from_pretrained(
    synthesis_model_path=pretrained_model_path,
    synthesis_args=args,
    device=f"cuda:{args.gpu}",
)

# 生成视频
output_video = pipeline(
    input_path=input_path,
    text_prompt=text_prompt,
    args=args
)

save_video_to_file(output_video, output_path=output_path)
