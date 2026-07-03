from typing import Dict

from .base_operator import BaseOperator


class HunyuanWorldPlay2Operator(BaseOperator):
    def __init__(self, operation_types=None, interaction_template=None) -> None:
        if operation_types is None:
            operation_types = ["visual_instruction"]
        super().__init__(operation_types=operation_types)
        self.interaction_template = interaction_template or ["target_size", "video_max_frames"]
        self.interaction_template_init()

    def check_interaction(self, interaction: Dict[str, int]) -> bool:
        if int(interaction["target_size"]) <= 0:
            raise ValueError("HY-World-2.0 target_size must be positive.")
        if int(interaction["video_max_frames"]) < 1:
            raise ValueError("HY-World-2.0 video_max_frames must be >= 1.")
        return True

    def get_interaction(self, interaction: Dict[str, int]) -> None:
        self.check_interaction(interaction)
        self.current_interaction.append(interaction)

    def process_interaction(self, target_size: int = 952, video_max_frames: int = 32, **kwargs) -> Dict[str, int]:
        interaction = {
            "target_size": int(target_size),
            "video_max_frames": int(video_max_frames),
        }
        self.get_interaction(interaction)
        now_interaction = self.current_interaction[-1]
        self.interaction_history.append(now_interaction)
        return now_interaction
