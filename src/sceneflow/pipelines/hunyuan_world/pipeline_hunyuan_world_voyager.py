"""
input image and interaction signal output rendering video
load operators, representations, and rendering model
"""
import torch
import numpy as np
import os
from PIL import Image
from typing import Optional, Any
from ..pipeline_utils import PipelineABC
from ...operators.hunyuan_world_voager_operator import HunyuanWorldVoyagerOperator
from ...representations.point_clouds_generation.hunyuan_world.hunyuan_world_voyager_representation import HunyuanWorldVoyagerRepresentation
from ...synthesis.visual_generation.hunyuan_world.hunyuan_world_voyager_synthesis import HunyuanWorldVoyagerSynthesis
from ...synthesis.visual_generation.hunyuan_world.hunyuan_world_voyager.config import parse_args
from ...synthesis.visual_generation.hunyuan_world.hunyuan_world_voyager.utils.file_utils import video_output


class HunyuanWorldVoyagerPipeline(PipelineABC):
    def __init__(self,
                 operators: Optional[HunyuanWorldVoyagerOperator] = None,
                 represent_model: Optional[HunyuanWorldVoyagerRepresentation] = None,
                 rendering_model: Optional[HunyuanWorldVoyagerSynthesis] = None,
                 rendering_args = None,
                 save_representation_video = False,
                 device: str = 'cuda'):
        super(HunyuanWorldVoyagerPipeline, self).__init__()
        self.operators = operators
        self.represent_model = represent_model
        self.rendering_model = rendering_model
        self.rendering_args = rendering_args
        self.save_representation_video = save_representation_video
        self.device = device

        os.makedirs(self.rendering_args.input_path, exist_ok=True)
    
    @classmethod
    def from_pretrained(cls,
                        model_path: Optional[str] = None,
                        required_components = {"represent_model_path": "Ruicheng/moge-vitl"},
                        device: str = "cuda",
                        represent_render_dir: str = './output/hunyuan_world_voyager/represent_render',
                        save_representation_video: bool = False,
                        **kwargs) -> 'HunyuanWorldVoyagerPipeline':
        """
        Load the complete pipeline from a pretrained model

        Args:
            pretrained_model_name_or_path: Path or name of the main model
            represent_model_path: Path to the representation model; uses default path if None
            model_path: Path to the rendering model; uses default path if None
            represent_render_dir: Directory for rendering output
            device: Device (e.g., 'cuda', 'cpu')
            **kwargs: Additional parameters passed to sub-models

        Returns:
            HunyuanWorldVoyagerPipeline: Initialized pipeline instance
        """
        # 设置默认路径
        if model_path is None:
            model_path = "tencent/HunyuanWorld-Voyager"
        if isinstance(required_components, dict) and "represent_model_path" in required_components.keys():
            represent_model_path = required_components.get("represent_model_path", "Ruicheng/moge-vitl")
        else:
            represent_model_path = "Ruicheng/moge-vitl"

        print(f"Loading representation model from {represent_model_path}")
        represent_model = HunyuanWorldVoyagerRepresentation.from_pretrained(
            represent_model_path,
            device=device,
            depth_model_name='moge_v1', 
            **kwargs
        )

        print(f"Loading rendering model from {model_path}")
        rendering_args = parse_args()
        rendering_args.model_base = model_path
        rendering_args.input_path = represent_render_dir

        rendering_model = HunyuanWorldVoyagerSynthesis.from_pretrained(
            model_path, 
            rendering_args,
            # **{k: v for k, v in kwargs.items() if k in ['cache_dir', 'force_download', 'resume_download']}
        )

        operators = HunyuanWorldVoyagerOperator()

        pipeline = cls(
            operators=operators,
            represent_model=represent_model,
            rendering_model=rendering_model,
            rendering_args=rendering_args,
            save_representation_video=save_representation_video,
            device=device
        )
        
        return pipeline

    def process(self, input_image, interaction_signal="forward"):
        # transform input image and interaction signal into hunyuan video input
        input_image, image_tensor = self.operators.process_perception(input_image, self.device)
        Height, Width = input_image.shape[:2] if hasattr(input_image, 'shape') else (256, 256)
        
        # generate intrinsics and extrinsics for the first frame based on the interaction signal
        self.operators.get_interaction(interaction_signal)
        intrinsics, extrinsics = self.operators.process_interaction(
            num_frames=1, Width=Width, Height=Height, fx=256, fy=256
        )
        
        # generate representation (points, colors, depth) based on the input image and camera parameters
        input_data = {
            'image': input_image,
            'image_tensor': image_tensor,
            'intrinsics': intrinsics,
            'extrinsics': extrinsics
        }
        points, colors, depth = self.represent_model.get_representation(input_data)
        
        # generate intrinsics and extrinsics for the whole video based on the interaction signal
        intrinsics, extrinsics = self.operators.process_interaction(
            num_frames=49, Width=Width//2, Height=Height//2, fx=128, fy=128
        )
        self.operators.delete_last_interaction()
        
        # rendering the video
        render_list, mask_list, depth_list = self.represent_model.render_video(
            points, colors, extrinsics, intrinsics, height=Height//2, width=Width//2
        )
        hunyuan_video_input = self.rendering_model.create_hunyuan_video_input(render_list, mask_list, depth_list,
                                                                              Width=Width, Height=Height)

        if self.save_representation_video:
            self.represent_model.save_representation_video(
                render_list, mask_list, depth_list, self.rendering_args.input_path, separate=True, 
                ref_image=input_image, ref_depth=depth, Width=Width, Height=Height
            )
        
        return hunyuan_video_input

    def __call__(self,
                 images,
                 interactions="forward",
                 prompt = "",
                 output_save_path = "./output/hunyuan_world_voyager/final_render",
                 i2v_stability=True,
                 **kwargs):
        """调用接口，支持额外参数"""
        hunayuan_video_input = self.process(images, interactions, **kwargs)
        outputs = self.rendering_model.predict(
            prompt=prompt,
            height=self.rendering_args.video_size[0],
            width=self.rendering_args.video_size[1],
            video_length=self.rendering_args.video_length,
            seed=self.rendering_args.seed,
            negative_prompt=self.rendering_args.neg_prompt,
            infer_steps=self.rendering_args.infer_steps,
            guidance_scale=self.rendering_args.cfg_scale,
            num_videos_per_prompt=self.rendering_args.num_videos,
            flow_shift=self.rendering_args.flow_shift,
            batch_size=self.rendering_args.batch_size,
            embedded_guidance_scale=self.rendering_args.embedded_cfg_scale,
            i2v_mode=self.rendering_args.i2v_mode,
            i2v_resolution=self.rendering_args.i2v_resolution,
            i2v_image_path=self.rendering_args.i2v_image_path,
            i2v_condition_type=self.rendering_args.i2v_condition_type,
            i2v_stability=i2v_stability,
            ulysses_degree=self.rendering_args.ulysses_degree,
            ring_degree=self.rendering_args.ring_degree,
            ref_image=hunayuan_video_input['ref_image'],
            ref_depth=hunayuan_video_input['ref_depth'],
            render_list=hunayuan_video_input['render_list'],
            depth_list=hunayuan_video_input['depth_list'],
            mask_list=hunayuan_video_input['mask_list'],
        )
        samples = outputs['samples']

        # Save generated videos to disk
        # Only save on the main process in distributed settings
        if 'LOCAL_RANK' not in os.environ or int(os.environ['LOCAL_RANK']) == 0:
            sample = samples[0].unsqueeze(0)
            output_video = video_output(sample, fps=24)
        return output_video
