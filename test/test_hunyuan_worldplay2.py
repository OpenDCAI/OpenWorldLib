import os

from openworldlib.pipelines.hunyuan_world.pipeline_hunyuan_worldplay2 import HunyuanWorldPlay2Pipeline


model_path = os.environ.get("HY_WORLD2_MODEL_PATH", "tencent/HY-World-2.0")
input_path = os.environ.get("HY_WORLD2_INPUT", "./data/test_case/test_image_seq_case1/image_0001.jpg")

pipeline = HunyuanWorldPlay2Pipeline.from_pretrained(
    model_path=model_path,
    subfolder=os.environ.get("HY_WORLD2_SUBFOLDER", "HY-WorldMirror-2.0"),
    python_bin=os.environ.get("HY_WORLD2_PYTHON", "python"),
)

result = pipeline(
    input_path=input_path,
    output_dir=os.environ.get("HY_WORLD2_OUTPUT", "./output/hunyuan_worldplay2"),
    target_size=int(os.environ.get("HY_WORLD2_TARGET_SIZE", "952")),
    video_max_frames=int(os.environ.get("HY_WORLD2_VIDEO_MAX_FRAMES", "32")),
    cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
    timeout=None,
)

print(result["gaussians_path"])
print(result["points_path"])
