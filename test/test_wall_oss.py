from sceneflow.pipelines.wall_oss.pipeline_wall_oss import WallOssPipeline
from PIL import Image

model_path = "x-square-robot/wall-oss-flow"
image_path = "./data/test_vla/main_view.png"

test_prompt = "To move the red cup in the table, what should you do next? Think step by step."

# No longer need to pass train_config_path, the pipeline will use default config
pipeline = WallOssPipeline.from_pretrained(
    pretrained_model_path=model_path,
    device="cuda",
)

answer = pipeline(
    text=test_prompt,
    image=Image.open(image_path).convert("RGB"),
    max_new_tokens=1024
)

print(answer)
