from .base_operator import BaseOperator


class SpatialReasonerOperator(BaseOperator):
    """
    Lightweight operator placeholder for SpatialReasoner.
    Tracks interactions to align with BaseOperator interface; no preprocessing or reasoning inside.
    """

    def __init__(self, operation_types=None, interaction_template=None):
        super().__init__(operation_types=operation_types or ["reasoning"])
        self.interaction_template = interaction_template or []
        self.interaction_template_init()

    @classmethod
    def from_pretrained(cls, *args, **kwargs) -> "SpatialReasonerOperator":
        return cls()

    def check_interaction(self, interaction):
        # Accept any interaction; extend validation if needed.
        return True

    def get_interaction(self, interaction):
        if self.check_interaction(interaction):
            self.current_interaction.append(interaction)
            self.interaction_history.append(interaction)

    def process_interaction(self, *args, **kwargs):
        return self.current_interaction
