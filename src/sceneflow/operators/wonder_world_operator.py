from .base_operator import BaseOperator


class WonderWorldOperator(BaseOperator):
    def __init__(self,
                 operation_types=["action_instruction"],
                 interaction_template=["forward", "left", "right", "backward", "camera_l", "camera_r", "camera_up", "camera_down"]
        ):
        super().__init__(operation_types)
    
    def check_interaction(self, interaction):
        if interaction not in self.interaction_template:
            raise ValueError(f"{interaction} not in template")
        return True

    def get_interaction(self, interaction):
        if not isinstance(interaction, list):
            interaction = [interaction]
        for act in interaction:
            self.check_interaction(act)
        self.current_interaction.append(interaction)
    
    def process_interaction(self):
        pass

    def process_perception(self, image):
        pass
