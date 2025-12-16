from ...base_representation import BaseRepresentation
from transformers import OneFormerForUniversalSegmentation, OneFormerProcessor
from .wonder_world.core_function.key_frame_gen import KeyframeGen
from .wonder_world.marigold_lcm.marigold_pipeline import MarigoldPipeline, MarigoldNormalsPipeline
from .wonder_world.utils.utils import prepare_scheduler, load_example_yaml, convert_pt3d_cam_to_3dgs_cam, soft_stitching
from .wonder_world.utils.segment_utils import create_mask_generator_repvit
from .wonder_world.arguments import GSParams
from ....base_models.three_dimensions.point_clouds.gaussian_splatting.scene import Scene, GaussianModel
from .wonder_world.utils.loss import l1_loss, ssim


class WonderWorldRepresentation(BaseRepresentation):
    def __init__(self):
        super().__init__()

    @classmethod
    def from_pretrained(cls,
                        segmentation_model_path,
                        inpaint_model_path,
                        depth_predict_model_path,
                        normal_predict_model_path,
                        device=None,
                        **kwargs):
        pass

    def get_representation(self, data):
        pass
