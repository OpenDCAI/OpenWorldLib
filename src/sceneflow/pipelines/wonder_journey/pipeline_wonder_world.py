from ..pipeline_utils import PipelineABC
from ...operators.wonder_world_operator import WonderWorldOperator
from ...representations.point_clouds_generation.wonder_journey.wonder_world_representation import WonderWorldRepresentation


class WonderWorldPipeline(PipelineABC):
    def __init__(self):
        super().__init__()
    
    @classmethod
    def from_pretrained(cls,
                        segmentation_model_path,
                        inpaint_model_path,
                        depth_predict_model_path,
                        normal_predict_model_path,
                        device=None,
                        **kwargs) -> 'WonderWorldPipeline':
        pass

    def process(self, input_image, interaction_signal):
        pass

    def __call__(self,
                 input_image,
                 interaction_signal,
                 **kwargs):
        pass
