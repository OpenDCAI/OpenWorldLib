import torch
import numpy as np
import tqdm
from PIL import Image
from typing import List, Union, Dict, Any, Optional
from torchvision.transforms import ToTensor, ToPILImage

from operators import WonderJourneyOperator
from representations import WonderJourneyRepresentation
from synthesis import WonderJourneySynthesis

class WonderJourneyPipeline:
    def __init__(
        self, 
        operator: WonderJourneyOperator, 
        representation: WonderJourneyRepresentation, 
        synthesis: WonderJourneySynthesis,
    ):
        self.operator = operator
        self.representation = representation
        self.synthesis = synthesis
        
        self.device = self.representation.device

    @classmethod
    def from_pretrained(
        cls, 
        pretrained_model_path: str, 
        device: str = "cuda",
        **kwargs
    ):
        raise NotImplementedError("Please implement specific model loading logic or use the constructor directly.")

    @torch.no_grad()
    def process(self, image: Image.Image, prompt: str, **kwargs):
        """
        Template 要求接口：单步处理
        这里可以理解为生成下一帧的关键帧逻辑
        """
        pass

    @torch.no_grad()
    def __call__(
        self, 
        initial_image: Image.Image, 
        prompt: str, 
        num_frames: int = 60,
        interactions: List[Dict[str, Any]] = None,
        enable_finetune_depth: bool = False,
        enable_finetune_decoder: bool = False,
        enable_upsample: bool = False,
        enable_visibility_check: bool = False,
        **kwargs
    ) -> List[Image.Image]:
        """
        Pipeline 的调用入口。执行完整的 3D 漫游生成过程。
        
        Args:
            initial_image: 起始图片
            prompt: 提示词
            num_frames: 总帧数
            interactions: 交互指令列表
            enable_finetune_depth: 是否开启深度微调 (耗时，提质)
            enable_finetune_decoder: 是否开启解码器微调 (耗时，提质)
            enable_upsample: 是否开启上采样 (耗显存，提质)
            enable_visibility_check: 是否开启遮挡剔除 (耗时，去鬼影)
        """
        
        print(f"Start WonderJourney Pipeline: {num_frames} frames, Prompt: '{prompt}'")
        
        image_tensor = ToTensor()(initial_image).unsqueeze(0).to(self.device) # [1, 3, H, W]
        result_frames = [initial_image]
        depth, _ = self.representation.get_depth(image_tensor)
        
        self.operator.init_camera()
        
        self.representation.reset_cloud()

        if enable_upsample:
             full_mask = torch.ones_like(depth)
             image_up, depth_up, mask_up, grid_up = self.representation.upsample_data(image_tensor, depth, full_mask, coef=2)
             self.representation.update_cloud(depth_up, image_up, valid_mask=None, camera=self.operator.current_camera, points_2d=grid_up)
        else:
             self.representation.update_cloud(depth, image_tensor, valid_mask=None, camera=self.operator.current_camera)

        if interactions:
            for interaction in interactions:
                self.operator.get_interaction(interaction)
        else:
            self.operator.get_interaction({"type": "movement", "content": "straight", "frames": num_frames})

        current_image_tensor = image_tensor
        
        for i in tqdm.tqdm(range(num_frames)):
            next_camera = self.operator.process_interaction()
            points_3d = self.representation.points_3d
            colors = self.representation.colors
            
            rendered_image, rendered_depth, inpaint_mask = self.synthesis.render_scene(
                camera=next_camera,
                points_3d=points_3d,
                colors=colors
            )
            
            inpainted_image, inpainted_latent = self.synthesis.inpaint(
                rendered_image=rendered_image,
                inpaint_mask=inpaint_mask,
                prompt=prompt
            )

            if enable_finetune_decoder:
                self.synthesis.finetune_decoder(
                    inpainted_image=inpainted_image,
                    inpainted_image_latent=inpainted_latent,
                    rendered_image=rendered_image,
                    inpaint_mask=inpaint_mask,
                    steps=10
                )
                inpainted_image = self.synthesis.decode_latents(inpainted_latent)

            inpainted_pil = ToPILImage()(inpainted_image.squeeze())
            result_frames.append(inpainted_pil)
        
            new_depth, _ = self.representation.get_depth(inpainted_image)
            
            if enable_finetune_depth:
                self.representation.finetune_depth_model(
                    target_depth=rendered_depth,
                    inpainted_image=inpainted_image,
                    mask_align=(~inpaint_mask.bool()).float(), # 对齐已知区域
                    mask_cutoff=inpaint_mask.float(),          # 限制未知区域
                    cutoff_depth=self.synthesis.background_hard_depth,
                    steps=5
                )
                new_depth, _ = self.representation.get_depth(inpainted_image)

            # 上采样数据准备
            update_image = inpainted_image
            update_depth = new_depth
            update_mask = inpaint_mask
            update_grid = None
            
            if enable_upsample:
                update_image, update_depth, update_mask, update_grid = self.representation.upsample_data(
                    inpainted_image, new_depth, inpaint_mask, coef=2
                )
            
            # 更新点云
            self.representation.update_cloud(
                rendered_depth=update_depth,
                image=update_image,
                valid_mask=update_mask, 
                camera=next_camera,
                points_2d=update_grid
            )
            
            # 遮挡剔除 
            if enable_visibility_check:
                frag_idx = self.synthesis.get_fragment_indices(
                    camera=next_camera,
                    points_3d=self.representation.points_3d,
                    colors=self.representation.colors
                )
                
                # 这里需要再改一下
                # bad_indices = self.representation.calculate_inconsistent_points(frag_idx)
                # self.representation.remove_occluded_points(bad_indices)
                pass

            # 清理显存 
            if i % 5 == 0:
                torch.cuda.empty_cache()

        print("Generation finished.")
        return result_frames