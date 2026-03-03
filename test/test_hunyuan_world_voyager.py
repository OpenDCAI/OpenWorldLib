# from diffusers.utils import export_to_video
import imageio
from PIL import Image
from sceneflow.pipelines.hunyuan_world.pipeline_hunyuan_world_voyager import HunyuanWorldVoyagerPipeline


image_path = "./data/test_case1/ref_image.png"
moge_model_path = "Ruicheng/moge-vitl"
hunyuan_world_voyager_model_path = "tencent/HunyuanWorld-Voyager"

input_image = Image.open(image_path).convert('RGB')
test_prompt = "An old-fashioned European village with thatched roofs on the houses."

pipeline = HunyuanWorldVoyagerPipeline.from_pretrained(
    model_path=hunyuan_world_voyager_model_path,
    required_components = {"represent_model_path": moge_model_path},
    save_representation_video=True
)

output_video = pipeline(images=input_image, prompt=test_prompt)
imageio.mimsave("hunyuan_world_voyager.mp4", output_video, fps=12)
