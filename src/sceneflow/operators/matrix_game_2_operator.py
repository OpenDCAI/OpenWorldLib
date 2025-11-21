from .base_operator import BaseOperator


class MatrixGame2Operator(BaseOperator):
    def __init__(self,
                 operation_types=[],
                 interaction_template = []
        ):
        super(MatrixGame2Operator, self).__init__(operation_types=operation_types)
        self.interaction_template = interaction_template
        self.interaction_template_init()