from sceneflow.pipelines.hunyuan_world.pipeline_hunyuan_world_voyager import HunyuanWorldVoyagerPipeline
from PIL import Image


image_path = "./data/test_case1/ref_image.png"
moge_model_path = "/ytech_m2v5_hdd/CheckPoints/moge-vitl"
hunyuan_world_voyager_model_path = "/ytech_m2v5_hdd/CheckPoints/HunyuanWorld-Voyager"

input_image = Image.open(image_path).convert('RGB')
test_prompt = "An old-fashioned European village with thatched roofs on the houses."

pipeline = HunyuanWorldVoyagerPipeline.from_pretrained(
    represent_model_path=moge_model_path,
    rendering_model_path=hunyuan_world_voyager_model_path,
    save_representation_video=True
)

pipeline(input_image=input_image, interaction_text_prompt=test_prompt)
