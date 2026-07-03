from typing import Dict, Optional

from .base_operator import BaseOperator


class GammaWorldOperator(BaseOperator):
    def __init__(self, operation_types=None, interaction_template=None) -> None:
        if operation_types is None:
            operation_types = ["action_instruction", "visual_instruction", "textual_instruction"]
        super().__init__(operation_types=operation_types)
        self.interaction_template = interaction_template or ["bidirectional", "causal", "causal_few_step"]
        self.interaction_template_init()

    def check_interaction(self, interaction: Dict[str, Optional[int]]) -> bool:
        mode = interaction.get("mode")
        n_players = interaction.get("n_players")
        max_eval_samples = interaction.get("max_eval_samples")
        if mode not in self.interaction_template:
            raise ValueError(f"Gamma-World mode must be one of {self.interaction_template}, got {mode!r}.")
        if int(n_players) < 1:
            raise ValueError("Gamma-World n_players must be >= 1.")
        if max_eval_samples is not None and int(max_eval_samples) < 1:
            raise ValueError("Gamma-World max_eval_samples must be >= 1 when provided.")
        return True

    def get_interaction(self, interaction: Dict[str, Optional[int]]) -> None:
        self.check_interaction(interaction)
        self.current_interaction.append(interaction)

    def process_interaction(
        self,
        mode: str = "causal_few_step",
        n_players: int = 2,
        max_eval_samples: Optional[int] = 1,
        **kwargs,
    ) -> Dict[str, Optional[int]]:
        interaction = {
            "mode": mode,
            "n_players": int(n_players),
            "max_eval_samples": None if max_eval_samples is None else int(max_eval_samples),
        }
        self.get_interaction(interaction)
        now_interaction = self.current_interaction[-1]
        self.interaction_history.append(now_interaction)
        return now_interaction
