from PIL import Image
from sceneflow.pipelines.wonder_journey.pipeline_wonder_world import WonderWorldPipeline


## Initialize the WonderWorld Pipeline
inpaint_model_id = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
segmentation_model_id = "shi-labs/oneformer_ade20k_swin_large"
mask_model_id = "zbhpku/repvit-sam-hf-mirror"
depth_predict_model_id = "prs-eth/marigold-depth-v1-0"
normal_predict_model_id = "prs-eth/marigold-normals-v1-1"

pipeline = WonderWorldPipeline.from_pretrained(
    inpaint_model_path=inpaint_model_id,
    segment_model_path=segmentation_model_id,
    mask_model_path=mask_model_id,
    depth_predict_model_path=depth_predict_model_id,
    normal_predict_model_path=normal_predict_model_id,
    device="cuda"
)

## Load and preprocess the input
image_path = "./data/test_case1/ref_image.png"
input_image = Image.open(image_path).convert('RGB').resize((512, 512))

print("Test Case 1: Forward and Turn Left")
interactions = ["forward", "right", "left"]

prompt_list = [
    "a beautiful mountain landscape",
    "a forest path with tall trees",
    "a lake with reflections"
]

## Run the pipeline
output_dict = pipeline(
    input_image=input_image,
    prompt_list=prompt_list,
    interactions=interactions
)
