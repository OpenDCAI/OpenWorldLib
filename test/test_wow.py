import sys 
sys.path.append("..") 
from src.sceneflow.pipelines.wow.pipeline_wow import WowPipeline
from PIL import Image

def save(image_path, generation_json_path, output_dir):
    pass

image_path = "./data/test_case1/ref_image.png"
generation_json_path = "./data/test_case1/generation.json"
output_dir = "./output/wow"
pretraind_model_path = "your/model/path"

pipeline = WowPipeline.from_pretrained(
    pretrained_model_path=pretraind_model_path,
    use_image=True
)

result = pipeline(
    generation_json_path=generation_json_path,
    image_path=image_path,
    output_dir=output_dir
) # 其实好像可以不用image_path

save(image_path, generation_json_path, output_dir)