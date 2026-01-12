import torch
import numpy as np
import cv2
import os
from PIL import Image
from typing import Optional, Any, List, Union
from torchvision.transforms import v2
from ...operators.matrix_game_2_operator import MatrixGame2Operator
from ...synthesis.visual_generation.matrix_game.matrix_game_2_synthesis import MatrixGame2Synthesis


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    last_frame = (tensor * 255).astype(np.uint8)
    pil_image = Image.fromarray(last_frame)
    return pil_image


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
        self.current_image = None

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
    
    def stream(self,
               interaction_signal: List[str],
               initial_image: Optional[Image.Image] = None,
               num_output_frames: int = 15,
               resize_H: int = 352,
               resize_W: int = 640,
               operation_visualization: bool = False,
               **kwds) -> torch.Tensor:
        """
        执行单步交互生成，并自动更新内部状态 (self.current_image)。
        
        Args:
            interaction_signal (List[str]): 本次交互的控制信号。
            initial_image (Optional[Image.Image]): 
                - 如果提供，将重置当前状态，以此图片作为起点（通常用于第一轮）。
                - 如果为 None，则使用上一轮生成的最后一帧作为起点。
            num_output_frames (int): 本次生成的帧数 (可以每轮动态改变)。
            resize_H, resize_W: 本次生成的分辨率 (可以每轮动态改变)。
            
        Returns:
            torch.Tensor: 本次生成的视频片段。
        """
        
        # 1. 状态管理：如果有新图片传入，则重置起点；否则检查是否有历史状态
        if initial_image is not None:
            print("--- Stream Session Reset/Started with new image ---")
            self.current_image = initial_image
        
        if self.current_image is None:
            raise ValueError("Current image is None. Please provide 'initial_image' for the first step.")

        # 2. 执行生成 (直接复用 __call__ 的逻辑)
        # 这里的参数完全由你本次调用决定，不再受限于生成器的初始化
        video_output = self.__call__(
            input_image=self.current_image,
            num_output_frames=num_output_frames,
            interaction_signal=interaction_signal,
            resize_H=resize_H,
            resize_W=resize_W,
            operation_visualization=operation_visualization,
            **kwds
        )

        # 3. 更新状态：提取最后一帧，保存到 self.current_image 供下一轮使用
        # 假设 video_output 是 tensor，我们需要将其转回 PIL
        # 注意：这里直接使用你外部定义的 tensor_to_pil 函数
        last_frame_tensor = video_output[-1] 
        self.current_image = tensor_to_pil(last_frame_tensor)

        return video_output
