import os

from openworldlib.pipelines.ctrl_world.pipeline_ctrl_world import CtrlWorldPipeline


model_path = os.environ.get("CTRL_WORLD_CKPT", "./models/Ctrl-World/checkpoint-10000.pt")
required_components = {
    "svd_model_path": os.environ.get("CTRL_WORLD_SVD_MODEL", "./models/stable-video-diffusion-img2vid"),
    "clip_model_path": os.environ.get("CTRL_WORLD_CLIP_MODEL", "./models/clip-vit-base-patch32"),
    "dataset_root_path": os.environ.get("CTRL_WORLD_DATASET_ROOT", "./datasets/ctrl_world/dataset_example"),
    "dataset_meta_info_path": os.environ.get("CTRL_WORLD_DATASET_META", "./datasets/ctrl_world/dataset_meta_info"),
}

pipeline = CtrlWorldPipeline.from_pretrained(
    model_path=model_path,
    required_components=required_components,
    python_bin=os.environ.get("CTRL_WORLD_PYTHON", "python"),
)

result = pipeline(
    interactions=os.environ.get("CTRL_WORLD_KEYBOARD", "ddcu"),
    output_dir=os.environ.get("CTRL_WORLD_OUTPUT", "./output/ctrl_world"),
    cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
    timeout=None,
)

print(result["video_path"])
