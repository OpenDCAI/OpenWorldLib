from _path_defaults import dataset_path, env_str, model_path, python_bin
from openworldlib.pipelines.ctrl_world.pipeline_ctrl_world import CtrlWorldPipeline


def main():
    pipe = CtrlWorldPipeline.from_pretrained(
        model_path=model_path("CTRL_WORLD_CKPT", "Ctrl-World/checkpoint-10000.pt"),
        svd_model_path=model_path("CTRL_WORLD_SVD_MODEL", "stable-video-diffusion-img2vid"),
        clip_model_path=model_path("CTRL_WORLD_CLIP_MODEL", "clip-vit-base-patch32"),
        dataset_root_path=dataset_path("CTRL_WORLD_DATASET_ROOT", "ctrl_world/dataset_example"),
        dataset_meta_info_path=dataset_path("CTRL_WORLD_DATASET_META", "ctrl_world/dataset_meta_info"),
        python_bin=python_bin("CTRL_WORLD_PYTHON"),
    )
    result = pipe(
        interactions=env_str("CTRL_WORLD_KEYBOARD", "ddcu"),
        output_dir=env_str("CTRL_WORLD_OUTPUT", "./output/ctrl_world"),
        cuda_visible_devices=env_str("CUDA_VISIBLE_DEVICES", "0"),
        timeout=None,
    )
    print(result["video_path"])


if __name__ == "__main__":
    main()
