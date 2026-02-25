import os
from typing import Any, Dict

from .base_operator import BaseOperator
from ..synthesis.visual_generation.hunyuan_world.hunyuan_worldplay.commons.pose_utils import (
    parse_pose_string,
    pose_to_latent_num,
    pose_to_input,
)


class HunyuanWorldPlayOperator(BaseOperator):
    def __init__(self, operation_types=None, interaction_template=None):
        if operation_types is None:
            operation_types = ["action_instruction"]
        super().__init__(operation_types=operation_types)
        self.interaction_template = interaction_template or ["w", "s", "a", "d", "up", "down", "left", "right"]
        self.interaction_template_init()

    def process_perception(self, input_):
        if input_ is None:
            raise ValueError("reference_image must be provided")
        return input_

    def check_interaction(self, interaction):
        if isinstance(interaction, str):
            pose_str = interaction.strip()
            if pose_str == "":
                raise ValueError("interaction cannot be empty")
            if pose_str.endswith(".json"):
                if not os.path.exists(pose_str):
                    raise ValueError(f"Pose json not found: {pose_str}")
                return True
            parse_pose_string(pose_str)
            return True
        if isinstance(interaction, dict):
            if len(interaction) == 0:
                raise ValueError("interaction cannot be empty")
            sample = next(iter(interaction.values()))
            if not isinstance(sample, dict) or "extrinsic" not in sample or "K" not in sample:
                raise ValueError("pose dict must contain {'extrinsic','K'} per frame")
            return True
        raise TypeError(f"interaction must be str or dict, got {type(interaction)}")

    def get_interaction(self, interaction):
        if not isinstance(interaction, list):
            interaction = [interaction]
        if len(interaction) == 0:
            raise ValueError("interaction cannot be empty")
        for item in interaction:
            self.check_interaction(item)
        self.current_interaction.append(interaction)

    def process_interaction(self, latent_frames: int) -> Dict[str, Any]:
        if len(self.current_interaction) == 0:
            raise ValueError("No interaction to process")
        now_interaction = self.current_interaction[-1]
        self.interaction_history.append(now_interaction)
        pose_data = now_interaction[-1]
        viewmats, Ks, action = pose_to_input(pose_data, latent_frames)
        return {
            "viewmats": viewmats,
            "Ks": Ks,
            "action": action,
        }

    def infer_video_length(self, interaction) -> int:
        if isinstance(interaction, list):
            if len(interaction) == 0:
                raise ValueError("interaction cannot be empty")
            interaction = interaction[-1]
        latent_num = pose_to_latent_num(interaction)
        if latent_num <= 0:
            raise ValueError("pose must not be empty")
        return latent_num * 4 - 3
