import os

from openworldlib.pipelines.fantasy_world.pipeline_fantasy_world import FantasyWorldPipeline


model_path = os.environ.get("FANTASY_WORLD_CKPT", "./models/FantasyWorld-Wan2.1-I2V-14B-480P/model.pth")
required_components = {
    "wan_ckpt_path": os.environ.get("FANTASY_WORLD_WAN_CKPT", "./models/Wan2.1-I2V-14B-480P"),
}
image_path = os.environ.get("FANTASY_WORLD_IMAGE", "./data/test_case/test_image_seq_case1/image_0001.jpg")
camera_json_path = os.environ.get("FANTASY_WORLD_CAMERA", "./data/test_case/fantasy_world/camera_data.json")
prompt = os.environ.get(
    "FANTASY_WORLD_PROMPT",
    "In the Open Loft Living Room, sunlight streams through large windows, highlighting the sleek fireplace and elegant wooden stairs.",
)

pipeline = FantasyWorldPipeline.from_pretrained(
    model_path=model_path,
    required_components=required_components,
    python_bin=os.environ.get("FANTASY_WORLD_PYTHON", "python"),
)

result = pipeline(
    image_path=image_path,
    camera_json_path=camera_json_path,
    prompt=prompt,
    output_dir=os.environ.get("FANTASY_WORLD_OUTPUT", "./output/fantasy_world"),
    sample_steps=int(os.environ.get("FANTASY_WORLD_SAMPLE_STEPS", "50")),
    frames=int(os.environ.get("FANTASY_WORLD_FRAMES", "17")),
    cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
    timeout=None,
)

print(result["video_path"])
print(result["point_cloud_path"])
