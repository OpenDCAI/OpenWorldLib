from _path_defaults import env_int, env_str, model_path, python_bin, test_case_path
from openworldlib.pipelines.fantasy_world.pipeline_fantasy_world import FantasyWorldPipeline


def main():
    pipe = FantasyWorldPipeline.from_pretrained(
        model_path=model_path("FANTASY_WORLD_CKPT", "FantasyWorld-Wan2.1-I2V-14B-480P/model.pth"),
        wan_ckpt_path=model_path("FANTASY_WORLD_WAN_CKPT", "Wan2.1-I2V-14B-480P"),
        python_bin=python_bin("FANTASY_WORLD_PYTHON"),
    )
    result = pipe(
        image_path=env_str("FANTASY_WORLD_IMAGE", test_case_path("test_image_seq_case1", "image_0001.jpg")),
        camera_json_path=env_str("FANTASY_WORLD_CAMERA", test_case_path("fantasy_world", "camera_data.json")),
        prompt=env_str(
            "FANTASY_WORLD_PROMPT",
            "In the Open Loft Living Room, sunlight streams through large windows, highlighting the sleek fireplace and elegant wooden stairs.",
        ),
        output_dir=env_str("FANTASY_WORLD_OUTPUT", "./output/fantasy_world"),
        sample_steps=env_int("FANTASY_WORLD_SAMPLE_STEPS", 50),
        frames=env_int("FANTASY_WORLD_FRAMES", 17),
        cuda_visible_devices=env_str("CUDA_VISIBLE_DEVICES", "0"),
        timeout=None,
    )
    print(result["video_path"])
    print(result["point_cloud_path"])


if __name__ == "__main__":
    main()
