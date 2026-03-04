import json
import torch
from PIL import Image
from pathlib import Path

from sceneflow.pipelines.spirit_ai.pipeline_spirit_v1p5 import SpiritV1p5Pipeline


def main():
    ckpt_path = "Spirit-AI-robotics/Spirit-v1.5"
    norm_stats_path = "./data/test_vla/norm_stats.json"
    meta_path = "./data/test_vla/meta.json"

    pipeline = SpiritV1p5Pipeline.from_pretrained(
        pretrained_model_path=ckpt_path,
        norm_stats_path=norm_stats_path,
        device="cuda",
        use_bf16=True,
    )

    with open(meta_path, "r") as f:
        obs_data = json.load(f)

    images = {
        "cam_high": Image.open(obs_data["paths"]["main_view"]).convert("RGB"),
        "cam_left_wrist": Image.open(obs_data["paths"]["wrist_view"]).convert("RGB"),
    }
    raw_state = obs_data["observation"]["state"]
    task = obs_data.get("task", "Pick up the object")
    robot_type = obs_data.get("robot_type", "Franka")

    actions = pipeline(
        images=images,
        raw_state=raw_state,
        task=task,
        robot_type=robot_type,
        return_all_steps=True,
    )

    print(f"Number of action steps: {len(actions)}")
    print(f"First action: {actions[0]}")

    # stream_gen = pipeline.stream(
    #     images=images,
    #     raw_state=raw_state,
    #     task=task,
    #     robot_type=robot_type,
    # )

    # for step_idx, action in enumerate(stream_gen):
    #     print(f"Step {step_idx}: {action}")
    #     if step_idx >= 2:
    #         break


if __name__ == "__main__":
    main()
