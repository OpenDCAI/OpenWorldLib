import sys 
sys.path.append("..") 
from src.sceneflow.pipelines.emu.pipeline_emu3p5 import Emu3p5Pipeline
from PIL import Image

image_path = "/YOUR/IMAGE/PATH"
model_path = "/YOUR/MODEL/PATH"
test_prompt = "Translate this house into a school."

pipeline = Emu3p5Pipeline.from_pretrained(
    pretrained_model_path=model_path,
    use_image=True,
)

pipeline(prompt=test_prompt, reference_image=image_path,save_content=True)
