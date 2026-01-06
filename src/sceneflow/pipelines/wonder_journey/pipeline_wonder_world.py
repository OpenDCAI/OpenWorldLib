import torch
from PIL import Image
from typing import Optional
from ..pipeline_utils import PipelineABC
from ...operators.wonder_world_operator import WonderWorldOperator
from ...synthesis.visual_generation.wonder_journey.wonder_world_synthesis import WonderWorldSynthesis
from ...representations.point_clouds_generation.wonder_journey.wonder_world_representation import WonderWorldRepresentation


class WonderWorldPipeline(PipelineABC):
    def __init__(self,
                 operator: Optional[WonderWorldOperator] = None,
                 representation_model: Optional[WonderWorldRepresentation] = None,
                 synthesis_model: Optional[WonderWorldSynthesis] = None
                 ):
        super().__init__()
        self.operator = operator
        self.representation_model = representation_model
        self.synthesis_model = synthesis_model
    
    @classmethod
    def from_pretrained(cls,
                        inpaint_model_path="diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
                        segment_model_path="shi-labs/oneformer_ade20k_swin_large",
                        mask_model_path="zbhpku/repvit-sam-hf-mirror",
                        depth_predict_model_path="prs-eth/marigold-depth-v1-0",
                        normal_predict_model_path="prs-eth/marigold-normals-v1-1",
                        device="cuda",
                        **kwargs) -> 'WonderWorldPipeline':
        """
        load representation, synthesis for inpaint
        initial the operator
        """
        operator = WonderWorldOperator()
        synthesis_model = WonderWorldSynthesis.from_pretrained(
            pretrained_model_path=inpaint_model_path,
            device=device
        )
        representation_model = WonderWorldRepresentation.from_pretrained(
            segment_model_path=segment_model_path,
            mask_model_path=mask_model_path,
            depth_predict_model_path=depth_predict_model_path,
            normal_predict_model_path=normal_predict_model_path,
            device=device
        )
        return cls(operator, representation_model, synthesis_model)

    def process(self,
                input_image,
                prompt_list=[],
                interactions=[],
                is_gaussian_train=True,
                **kwargs):
        """
        在这里进行input_image和interaction_signal的处理
        在operator里面进行处理
        """
        #### 首先在representation里面执行init_keyframe_generator
        self.representation_model.init_keyframe_generator(self.synthesis_model)

        #### 检查input_image是不是PIL.Image
        if not isinstance(input_image, Image.Image) and is_gaussian_train:
            raise TypeError(f"input_image must be a PIL.Image object, current type: {type(input_image)}")
        #### 处理交互指令
        self.operator.clear_interaction()
        if len(interactions) > 0:
            self.operator.get_interaction(interactions)
        view_matrices = self.operator.process_interaction()
        return view_matrices

    def __call__(self,
                 input_image,
                 sky_prompt="blue sky",
                 prompt_list=[],
                 interactions=[],
                 is_gaussian_train=True,
                 **kwargs):
        """
        在这里调用process对输入进行处理（还需要处理下旋转角度）
        支持用户切换模式，先确认用户希望重建的范围以及对应的prompt，利用synthesis加入representation进行训练数据生成
        支持用户对优化好的3dgs进行交互，直接利用representation内的渲染函数进行渲染
        """
        #### kwargs 还应该带有别的参数
        if is_gaussian_train:
            #### 训练3DGS
            view_matrices = self.process(
                input_image=input_image,
                prompt_list=prompt_list,
                interactions=interactions
            )
            output_dict = self.representation_model.get_representation(
                input_image=input_image,
                sky_prompt=sky_prompt,
                prompt_list=prompt_list,
                rotation_list=view_matrices
            )
        else:
            #### 直接渲染交互序列
            view_matrices = self.process(
                input_image=None,
                prompt_list=prompt_list,
                interactions=interactions,
                is_gaussian_train=False
            )
            output_dict = self.representation_model.get_representation(
                input_image=None,
                rotation_list=view_matrices,
                is_gaussian_train=False
            )
        return output_dict
