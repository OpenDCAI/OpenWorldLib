from typing import Dict

from .base_operator import BaseOperator


class SolarisOperator(BaseOperator):
    def __init__(self, operation_types=None, interaction_template=None) -> None:
        if operation_types is None:
            operation_types = ["action_instruction", "visual_instruction"]
        super().__init__(operation_types=operation_types)
        self.interaction_template = interaction_template or ["eval_num_samples"]
        self.interaction_template_init()

    def check_interaction(self, interaction: Dict[str, int]) -> bool:
        if int(interaction["eval_num_samples"]) < 1:
            raise ValueError("Solaris eval_num_samples must be >= 1.")
        return True

    def get_interaction(self, interaction: Dict[str, int]) -> None:
        self.check_interaction(interaction)
        self.current_interaction.append(interaction)

    def process_interaction(self, eval_num_samples: int = 1, **kwargs) -> Dict[str, int]:
        interaction = {"eval_num_samples": int(eval_num_samples)}
        self.get_interaction(interaction)
        now_interaction = self.current_interaction[-1]
        self.interaction_history.append(now_interaction)
        return now_interaction
