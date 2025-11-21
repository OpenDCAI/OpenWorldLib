
class BaseOperator(object):
    def __init__(self,
                 operation_types=[]):
        """
        operation includes:
            - textual_instruction
            - visual_instruction
            - action_instruction (including mouse and keyboard to 
              control the trajectory and viewpoint.)
        """
        self.interaction_template = []
        self.current_interaction = []
        self.interaction_history = []

    def interaction_template_init(self):
        if type(self.interaction_template) is not list:
            raise ValueError("interaction_template should be a list")

    def check_interaction(self, interaction):
        pass

    def process_interaction(self):
        pass

    def get_interaction(self, interaction):
        """
        utilize this function to update the interaction list
        """
        pass

    def get_interaction_template(self):
        return self.interaction_template
    
    def get_interaction_history(self):
        return self.interaction_history
    
    def delete_last_interaction(self):
        self.current_interaction = self.current_interaction[:-1]
