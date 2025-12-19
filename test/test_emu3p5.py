from sceneflow.pipelines.emu.pipeline_emu3p5 import Emu3p5Pipeline
from PIL import Image

image_path = "./data/test_case1/ref_image.png"
model_path = "BAAI/Emu3.5"
test_prompt = "Translate this house into a school."

pipeline = Emu3p5Pipeline.from_pretrained(
    pretrained_model_path=model_path,
    use_image=True
)

pipeline(prompt=test_prompt, reference_image=image_path,save_content=True)
