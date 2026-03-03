import torch
import numpy as np
import cv2
import os
from PIL import Image
from typing import Optional, Any, List, Union
from torchvision.transforms import v2
from ...operators.matrix_game_2_operator import MatrixGame2Operator
from ...synthesis.visual_generation.matrix_game.matrix_game_2_synthesis import MatrixGame2Synthesis
from ...memories.visual_synthesis.matrix_game.matrix_game_2_memory import MatrixGame2Memory


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    last_frame = (tensor * 255).astype(np.uint8)
    pil_image = Image.fromarray(last_frame)
    return pil_image


class MatrixGame2Pipeline:
    def __init__(self,
                 operators: Optional[MatrixGame2Operator] = None,
                 synthesis_model: Optional[MatrixGame2Synthesis] = None,
                 memory_module: Optional[Any] = None,
                 device: str = "cuda",
                 weight_dtype = torch.bfloat16,
                 ):
        self.synthesis_model = synthesis_model 
        self.operators = operators
        self.memory_module = memory_module
        self.device = device
        self.weight_dtype = weight_dtype
        self.current_image = None

    @classmethod
    def from_pretrained(cls,
                        model_path: Optional[str] = None,
                        required_components: Optional[dict] = None,
                        device: str = "cuda",
                        weight_dtype = torch.bfloat16,
                        mode = "universal",
                        **kwargs) -> "MatrixGame2Pipeline":
        if model_path is not None:
            synthesis_model_path = model_path
        else:
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
        memory_module = MatrixGame2Memory()

        pipeline = cls(
            operators=operators,
            synthesis_model=synthesis_model,
            memory_module=memory_module,
            device=device,
            weight_dtype=weight_dtype
        )
        return pipeline
    
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
        preception_dict = self.operators.process_perception(input_image, num_output_frames, resize_H, resize_W,
                                                            device=self.device, weight_dtype=self.weight_dtype)

        img_cond = self.synthesis_model.vae.encode(preception_dict["img_cond"], device=self.device,
                                                   **preception_dict["tiler_kwargs"]).to(self.device)
        mask_cond = torch.ones_like(img_cond)
        mask_cond[:, :, 1:] = 0
        cond_concat = torch.cat([mask_cond[:, :4], img_cond], dim=1) 
        visual_context = self.synthesis_model.vae.clip.encode_video(preception_dict["image"])

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
                 images,
                 interactions=["forward", "left", "right",
                               "forward_left", "forward_right",
                               "camera_l", "camera_r"],
                 num_frames=None,
                 size = (352, 640),
                 visualize_ops=True,
                 **kwds):
        if isinstance(images, Image.Image):
            input_image = images
        else:
            raise ValueError("Unsupported image type. Expected PIL.Image.")
        num_output_frames = len(interactions) * 12 if num_frames is None else num_frames
        resize_H, resize_W = size

        output_dict = self.process(
            input_image=input_image,
            num_output_frames=num_output_frames,
            resize_H=resize_H,
            resize_W=resize_W,
            interaction_signal=interactions
        )
        output_video = self.synthesis_model.predict(
            cond_concat=output_dict['cond_concat'],
            visual_context=output_dict['visual_context'],
            operator_condition=output_dict['operator_condition'],
            num_output_frames=num_output_frames,
            operation_visualization=visualize_ops,
            **kwds
        )
        return output_video
    
    def stream(self,
               images: Optional[Image.Image],
               interactions: List[str],
               num_frames: int = 15,
               size = (352, 640),
               visualize_ops: bool = False,
               **kwds) -> torch.Tensor:
        if images is not None:
            print("--- Stream Started ---")
            self.memory_module.record(images)
        
        current_image = self.memory_module.select()
        if current_image is None:
            raise ValueError("No image in storage. Provide 'images' first.")

        video_output = self.__call__(
            images=current_image,
            interactions=interactions,
            num_frames=num_frames,
            size=size,
            visualize_ops=visualize_ops,
            **kwds
        )

        self.memory_module.record(video_output)

        return video_output
