from typing import Any, Dict

from .base_operator import BaseOperator
from ..synthesis.visual_generation.hunyuan_world.hunyuan_worldplay.generate import pose_to_input


class HunyuanWorldPlayOperator(BaseOperator):
    def __init__(self, operation_types=None, interaction_template=None):
        if operation_types is None:
            operation_types = ["action_instruction"]
        super().__init__(operation_types=operation_types)
        self.interaction_template = interaction_template or []
        self.interaction_template_init()
        self.current_interaction = []

    def check_interaction(self, interaction):
        if isinstance(interaction, str):
            if interaction.strip() == "":
                raise ValueError("interaction cannot be empty")
            return True
        if isinstance(interaction, dict):
            if len(interaction) == 0:
                raise ValueError("interaction cannot be empty")
            return True
        raise TypeError(f"interaction must be str or dict, got {type(interaction)}")

    def get_interaction(self, interaction):
        if isinstance(interaction, (list, tuple)):
            if len(interaction) == 0:
                raise ValueError("interaction cannot be empty")
            for item in interaction:
                self.check_interaction(item)
            self.current_interaction.append(list(interaction))
            return
        self.check_interaction(interaction)
        self.current_interaction.append(interaction)

    def process_interaction(self, latent_frames: int) -> Dict[str, Any]:
        if len(self.current_interaction) == 0:
            raise ValueError("No interaction to process")
        pose_data = self.current_interaction[-1]
        if isinstance(pose_data, (list, tuple)):
            if len(pose_data) == 0:
                raise ValueError("interaction cannot be empty")
            pose_data = pose_data[-1]
        viewmats, Ks, action = pose_to_input(pose_data, latent_frames)
        return {
            "viewmats": viewmats,
            "Ks": Ks,
            "action": action,
        }
