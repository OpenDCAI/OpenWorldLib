from _path_defaults import env_int, env_str, model_path, python_bin, test_case_path
from openworldlib.pipelines.hunyuan_world.pipeline_hunyuan_worldplay2 import HunyuanWorldPlay2Pipeline


def main():
    pipe = HunyuanWorldPlay2Pipeline.from_pretrained(
        model_path=model_path("HY_WORLD2_MODEL_PATH", "HY-World-2.0", fallback="tencent/HY-World-2.0"),
        subfolder=env_str("HY_WORLD2_SUBFOLDER", "HY-WorldMirror-2.0"),
        python_bin=python_bin("HY_WORLD2_PYTHON"),
    )
    result = pipe(
        input_path=env_str("HY_WORLD2_INPUT", test_case_path("test_image_seq_case1", "image_0001.jpg")),
        output_dir=env_str("HY_WORLD2_OUTPUT", "./output/hunyuan_worldplay2"),
        target_size=env_int("HY_WORLD2_TARGET_SIZE", 952),
        video_max_frames=env_int("HY_WORLD2_VIDEO_MAX_FRAMES", 32),
        cuda_visible_devices=env_str("CUDA_VISIBLE_DEVICES", "0"),
        timeout=None,
    )
    print(result["gaussians_path"])
    print(result["points_path"])


if __name__ == "__main__":
    main()
