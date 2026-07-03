from typing import Dict, Optional

from .base_operator import BaseOperator


class Cosmos3Operator(BaseOperator):
    def __init__(self, operation_types=None, interaction_template=None) -> None:
        if operation_types is None:
            operation_types = ["textual_instruction", "visual_instruction"]
        super().__init__(operation_types=operation_types)
        self.interaction_template = interaction_template or ["prompt"]
        self.interaction_template_init()

    def check_interaction(self, interaction: str) -> bool:
        if not isinstance(interaction, str):
            raise TypeError(f"Cosmos3 prompt must be a string, got {type(interaction)}")
        if not interaction.strip():
            raise ValueError("Cosmos3 prompt must be a non-empty string.")
        return True

    def get_interaction(self, interaction: str) -> None:
        self.check_interaction(interaction)
        self.current_interaction.append(interaction.strip())

    def process_interaction(self, prompt: Optional[str] = None, **kwargs) -> Dict[str, str]:
        if prompt is not None:
            self.get_interaction(prompt)
        if len(self.current_interaction) == 0:
            raise ValueError("No interaction to process")
        prompt = self.current_interaction[-1]
        prompt = str(prompt).strip()
        self.interaction_history.append(prompt)
        return {"prompt": prompt}
