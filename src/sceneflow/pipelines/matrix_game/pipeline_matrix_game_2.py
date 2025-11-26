import torch
import numpy as np
import os
from PIL import Image
from typing import Optional, Any
from torchvision.transforms import v2
from ...operators.matrix_game_2_operator import MatrixGame2Operator
from ...synthesis.visual_generation.matrix_game.matrix_game_2_synthesis import MatrixGame2Synthesis


class MatrixGame2Pipeline:
    def __init__(self,
                 operators: Optional[MatrixGame2Operator] = None,
                 synthesis_model: Optional[MatrixGame2Synthesis] = None,
                 device: str = "cuda",
                 weight_dtype = torch.bfloat16,
                 ):
        self.synthesis_model = synthesis_model 
        self.operators = operators
        self.device = device
        self.weight_dtype = weight_dtype

        self.frame_process = v2.Compose([
            v2.Resize(size=(352, 640), antialias=True),
            v2.ToTensor(),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    @classmethod
    def from_pretrained(cls,
                        synthesis_model_path: Optional[str] = None,
                        mode = "universal",
                        weight_dtype = torch.bfloat16,
                        device: str = "cuda",
                        **kwargs) -> "MatrixGame2Pipeline":
        if synthesis_model_path is None:
            synthesis_model_path = "Skywork/Matrix-Game-2.0"
        
        print(f"Loading MatrixGame2 synthesis model from {synthesis_model_path}...")
        synthesis_model = MatrixGame2Synthesis.from_pretrained(
            pretrained_model_path=synthesis_model_path,
            device=device,
            mode=mode,
            weight_dtype=weight_dtype,
            **kwargs
        )
        operators = MatrixGame2Operator(mode=mode)

        pipeline = cls(
            operators=operators,
            synthesis_model=synthesis_model,
            device=device,
            weight_dtype=weight_dtype
        )
        return pipeline

    def _resizecrop(self, image, th, tw):
        w, h = image.size
        if h / w > th / tw:
            new_w = int(w)
            new_h = int(new_w * th / tw)
        else:
            new_h = int(h)
            new_w = int(new_h * tw / th)
        left = (w - new_w) / 2
        top = (h - new_h) / 2
        right = (w + new_w) / 2
        bottom = (h + new_h) / 2
        image = image.crop((left, top, right, bottom))
        return image
    
    def process(self,
                input_image,
                num_output_frames,
                resize_H=352,
                resize_W=640,
                interaction_signal=["forward", "left", "right",
                                    "forward_left", "forward_right",
                                    "camera_l", "camera_r"]):
        """
        the input_image is PIL image
        """
        image = self._resizecrop(input_image, resize_H, resize_W)
        image = self.frame_process(image)[None, :, None, :, :].to(dtype=self.weight_dtype, device=self.device)

        padding_video = torch.zeros_like(image).repeat(1, 1, 4 * (num_output_frames - 1), 1, 1)
        img_cond = torch.concat([image, padding_video], dim=2)
        tiler_kwargs={"tiled": True, "tile_size": [resize_H//8, resize_W//8], "tile_stride": [resize_H//16+1, resize_W//16-2]}
        img_cond = self.synthesis_model.vae.encode(img_cond, device=self.device, **tiler_kwargs).to(self.device)
        mask_cond = torch.ones_like(img_cond)
        mask_cond[:, :, 1:] = 0
        cond_concat = torch.cat([mask_cond[:, :4], img_cond], dim=1) 
        visual_context = self.synthesis_model.vae.clip.encode_video(image)

        output_dict = {
            "cond_concat": cond_concat,
            "visual_context": visual_context
        }

        # define the interaction
        self.operators.get_interaction(interaction_signal)
        num_frames = (num_output_frames - 1) * 4 + 1
        operator_condition = self.operators.process_interaction(num_frames=num_frames)

        output_dict['operator_condition'] = operator_condition

        self.operators.delete_last_interaction()
        
        return output_dict

    def __call__(self,
                 input_image,
                 num_output_frames,
                 resize_H=352,
                 resize_W=640,
                 interaction_signal=["forward", "left", "right",
                                     "forward_left", "forward_right",
                                     "camera_l", "camera_r"],
                 operation_visualization=True,
                 **kwds):
        output_dict = self.process(
            input_image=input_image,
            num_output_frames=num_output_frames,
            resize_H=resize_H,
            resize_W=resize_W,
            interaction_signal=interaction_signal
        )
        output_video = self.synthesis_model.predict(
            cond_concat=output_dict['cond_concat'],
            visual_context=output_dict['visual_context'],
            operator_condition=output_dict['operator_condition'],
            num_output_frames=num_output_frames,
            operation_visualization=operation_visualization,
            **kwds
        )
        return output_video
