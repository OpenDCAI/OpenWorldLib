from sceneflow.pipelines.wall_oss.pipeline_wall_oss import WallOssPipeline
from PIL import Image

model_path = "x-square-robot/wall-oss-flow"
train_config_path = ".data/test_vla/config_qact_from_vlm.yml"
image_path = "./data/test_case1/ref_image.png"

test_prompt = "To move the red block in the plate with same color, what should you do next? Think step by step."

pipeline = WallOssPipeline.from_pretrained(
    pretrained_model_path=model_path,
    train_config_path=train_config_path,
    device="cuda",
)

answer = pipeline(
    text=test_prompt,
    image=Image.open(image_path).convert("RGB"),
    max_new_tokens=1024
)

print(answer)
