from PIL import Image
from sceneflow.pipelines.hunyuan_world.pipeline_hunyuan_game_craft import HunyuanGameCraftPipeline
import torch
import imageio

image_path = "./data/test_case1/ref_image.png"
input_image = Image.open(image_path).convert('RGB')

pretrained_model_path = "tencent/Hunyuan-GameCraft-1.0"
pipeline = HunyuanGameCraftPipeline.from_pretrained(
    synthesis_model_path=pretrained_model_path,
    device="cuda",
    cpu_offload=False,
    seed=250160
)
output_video = pipeline(
    input_image=input_image,
    interaction_signal=["backward", "camera_l"], # ["forward", "left", "right", "backward", "camera_l", "camera_r", "camera_up", "camera_down"]
    interaction_speed=[0.2, 0.3], # value in [0, 3]
    interaction_text_prompt="A charming medieval village with cobblestone streets, thatched-roof houses.",
    interaction_positive_prompt="Realistic, High-quality.",
    interaction_negative_prompt="overexposed, low quality, deformation, a poor composition, bad hands, bad teeth, bad eyes, bad limbs, distortion, blurring, text, subtitles, static, picture, black border.",
    output_H=704,
    output_W=1216,
)

if torch.distributed.get_rank() == 0:
    imageio.mimsave("hunyuan_game_craft_demo.mp4", output_video, fps=24, quality=8)
