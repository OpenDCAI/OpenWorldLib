from .base_operator import BaseOperator


class SpatialLadderOperator(BaseOperator):
    """
    Lightweight operator placeholder for SpatialLadder.
    It tracks interactions to keep a consistent interface with BaseOperator,
    but does not alter inputs or perform reasoning.
    """

    def __init__(self, operation_types=None, interaction_template=None):
        super().__init__(operation_types=operation_types or ["reasoning"])
        self.interaction_template = interaction_template or []
        self.interaction_template_init()

    @classmethod
    def from_pretrained(cls, *args, **kwargs) -> "SpatialLadderOperator":
        # No weights to load; kept for API symmetry.
        return cls()

    def check_interaction(self, interaction):
        # Accept any interaction; extend if stricter validation is needed.
        return True

    def get_interaction(self, interaction):
        if self.check_interaction(interaction):
            self.current_interaction.append(interaction)
            self.interaction_history.append(interaction)

    def process_interaction(self, *args, **kwargs):
        # No processing needed; keep for interface compatibility.
        return self.current_interaction
