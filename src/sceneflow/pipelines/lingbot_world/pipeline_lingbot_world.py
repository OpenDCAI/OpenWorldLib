import torch
import numpy as np
from PIL import Image
from typing import Optional, Any, List, Union, Dict

from ...operators.lingbot_world_operator import LingBotOperator
from ...synthesis.visual_generation.lingbot.lingbot_world_synthesis import LingBotSynthesis
from ...memories.visual_synthesis.lingbot_world.lingbot_world_memory import LingBotMemory

class LingBotPipeline:
    def __init__(self,
                 operators: Optional[LingBotOperator] = None,
                 synthesis_model: Optional[LingBotSynthesis] = None,
                 memory_module: Optional[LingBotMemory] = None,
                 device: str = "cuda"
                 ):
        self.synthesis_model = synthesis_model
        self.operators = operators
        self.memory_module = memory_module
        self.device = device

    @classmethod
    def from_pretrained(cls,
                        model_path: str,
                        mode: str = "i2v-A14B",
                        device: str = "cuda",
                        **kwargs) -> "LingBotPipeline":
        
        print(f"Loading LingBot World Model from {model_path}...")
        
        synthesis_model = LingBotSynthesis.from_pretrained(
            pretrained_model_path=model_path,
            task=mode,
            device=device,
            **kwargs
        )
        
        operators = LingBotOperator()
        memory_module = LingBotMemory() # Now implements BaseMemory

        pipeline = cls(
            operators=operators,
            synthesis_model=synthesis_model,
            memory_module=memory_module,
            device=device
        )
        return pipeline
    
    def process(self,
                images: Any = None,
                prompt: Optional[str] = None,
                interactions: Optional[list[str]] = None,
                resize_H: int = 480,
                resize_W: int = 832,
                num_frames: Optional[int] = 81): 
        
        if isinstance(images, str):
            images = Image.open(images).convert("RGB")

        interaction_signal = {
            "prompt": prompt if prompt is not None else "",
            "action_list": interactions if interactions is not None else [],
        }
        
        perception_dict = self.operators.process_perception(
            images, 
            resize_H=resize_H, 
            resize_W=resize_W,
            device=self.device
        )
        
        self.operators.get_interaction(interaction_signal)
        interaction_dict = self.operators.process_interaction(
            resize_H=resize_H,
            resize_W=resize_W,
            num_frames=num_frames,
            device=self.device
        )
        
        output_dict = {
            "image_tensor": perception_dict["image_tensor"], 
            "prompt": interaction_dict["prompt"],
            "camera_data": interaction_dict.get("camera_data", None)
        }
        
        self.operators.current_interaction = [] 
        return output_dict

    def __call__(self,
                  images: Any = None,
                  num_frames: Optional[int] = 81,
                  prompt: Optional[str] = None,
                  interactions: Optional[list[str]] = None,
                  resize_H: int = 480,
                  resize_W: int = 832,
                  seed: int = 42, 
                  **kwds):
        
        processed_inputs = self.process(
            images=images,
            prompt=prompt,
            interactions=interactions,
            resize_H=resize_H,
            resize_W=resize_W,
            num_frames=num_frames 
        )
        
        output_video = self.synthesis_model.predict(
            image_tensor=processed_inputs["image_tensor"],
            prompt=processed_inputs["prompt"],
            camera_data=processed_inputs["camera_data"],
            num_output_frames=num_frames,
            height=resize_H,
            width=resize_W,
            seed=seed,
            **kwds
        )
        
        return output_video
    
    def stream(self,
                prompt: Optional[str] = None,
                interactions: Optional[list[str]] = None,
                images: Any = None,
                num_frames: Optional[int] = 81,
                resize_H: int = 480,
                resize_W: int = 832,
                seed: int = 42,
                **kwds) -> np.ndarray:
        
        # 1. Initialize Memory if images provided (First Turn)
        if images is not None:
            print("--- Stream Started ---")
            self.memory_module.manage(action="reset") # Clear old memory
            self.memory_module.record(images, type="image")
        
        # 2. Retrieve Context (Input for this turn)
        current_img = self.memory_module.select()
        if current_img is None:
            raise ValueError("No image in storage. Provide 'images' first.")

        # 3. Generate Video
        video_output = self.__call__(
            images=current_img,
            num_frames=num_frames,
            prompt=prompt,
            interactions=interactions,
            resize_H=resize_H,
            resize_W=resize_W,
            seed=seed,
            **kwds
        ) # Returns numpy array [T, H, W, C]

        # 4. Record Result (Updates context for next turn)
        if video_output is not None:
            self.memory_module.record(video_output, type="video_chunk")

        return video_output