from typing import Dict, Iterable, Optional, Union

from .base_operator import BaseOperator


class CtrlWorldOperator(BaseOperator):
    def __init__(self, operation_types=None, interaction_template=None) -> None:
        if operation_types is None:
            operation_types = ["action_instruction"]
        super().__init__(operation_types=operation_types)
        self.interaction_template = interaction_template or ["l", "r", "f", "b", "u", "d", "o", "c"]
        self.interaction_template_init()

    def check_interaction(self, interaction: str) -> bool:
        if not isinstance(interaction, str):
            raise TypeError(f"Ctrl-World interaction must be a string, got {type(interaction)}")
        if not interaction:
            raise ValueError("Ctrl-World interaction cannot be empty.")
        for key in interaction:
            if key not in self.interaction_template:
                raise ValueError(f"{key} not in template {self.interaction_template}")
        return True

    def get_interaction(self, interaction: Union[Iterable[str], str]) -> None:
        keyboard = interaction if isinstance(interaction, str) else "".join(interaction)
        self.check_interaction(keyboard)
        self.current_interaction.append(keyboard)

    def process_interaction(
        self,
        interactions: Optional[Union[Iterable[str], str]] = None,
        **kwargs,
    ) -> Dict[str, str]:
        if interactions is not None:
            self.get_interaction(interactions)
        if len(self.current_interaction) == 0:
            raise ValueError("No interaction to process")
        keyboard = self.current_interaction[-1]
        self.check_interaction(keyboard)
        self.interaction_history.append(keyboard)
        return {"keyboard": keyboard}
