import torch
import os
import numpy as np
import imageio
from PIL import Image
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

from ...operators.astra_operator import AstraOperator
from ...synthesis.visual_generation.kling.astra_synthesis import AstraSynthesis
from ...memories.visual_synthesis.kling.astra_memory import AstraMemory


@dataclass
class AstraConfig:
    # pretrained model paths
    astra_path: str = ""
    wan_model_path: str = ""
    
    # 生成参数 (默认值)
    start_frame: int = 0
    initial_condition_frames: int = 1
    frames_per_generation: int = 8
    total_frames_to_generate: int = 32  # 默认生成 32 帧
    max_history_frames: int = 49        # 滑动窗口大小
    
    # 采样与引导参数
    use_camera_cfg: bool = True         # 默认开启相机引导
    camera_guidance_scale: float = 2.0
    text_guidance_scale: float = 1.0
    
    # MoE 模型结构参数 (通常固定)
    moe_num_experts: int = 3
    moe_top_k: int = 1
    moe_hidden_dim: Optional[int] = None
    
    # 数据集/模态参数
    modality_type: str = "sekai"
    use_real_poses: bool = False
    scene_info_path: Optional[str] = None
    use_gt_prompt: bool = False
    
    # 运行时参数
    device: str = "cuda"
    add_icons: bool = True


class AstraPipeline(object):
    def __init__(self, operator, synthesis, memory, config):
        self.operator = operator
        self.synthesis = synthesis
        self.memory = memory
        self.config = config
        self.device = synthesis.device

    @classmethod
    def from_pretrained(cls, 
                        astra_path: str, 
                        wan_model_path: str, 
                        device: str = "cuda", 
                        **kwargs):
        """
        pretrained model loading
        """
        config = AstraConfig(
            astra_path=astra_path,
            wan_model_path=wan_model_path,
            device=device
        )
        for k, v in kwargs.items():
            if hasattr(config, k):
                setattr(config, k, v)
        
        print(f"Loading Astra components on {device}...")
        synthesis = AstraSynthesis.from_pretrained(config, device=device)
        operator = AstraOperator(device=device)
        memory = AstraMemory(capacity=config.max_history_frames)
        
        return cls(operator, synthesis, memory, config)
        
    def process(self, input_: str, interaction: Dict[str, Any]):
        args = self.config
        condition_image = input_
        prompt = interaction.get("prompt", "")
        direction = interaction.get("direction", "forward")

        # 1. 加载并编码条件图像 (Perception -> Representation)
        print(f"Processing image: {condition_image}")
        frames = self.operator.process_perception(condition_image=condition_image)
        latents = self.synthesis.encode_frames(frames)

        target_height, target_width = 60, 104
        C, T, H, W = latents.shape
        if H > target_height or W > target_width:
            h_start = (H - target_height) // 2
            w_start = (W - target_width) // 2
            latents = latents[:, :, h_start:h_start+target_height, w_start:w_start+target_width]
        
        model_dtype = next(self.synthesis.pipe.dit.parameters()).dtype
        history_latents = latents[:, :args.initial_condition_frames, :, :].to(self.device, dtype=model_dtype)
        initial_latents = history_latents # 备份用于最终拼接

        # 2. 编码提示词 (Interaction -> Representation)
        print(f"Encoding prompt: {prompt}")
        prompt_emb_pos = self.synthesis.pipe.encode_prompt(prompt)
        prompt_emb_neg = None
        if args.text_guidance_scale > 1.0:
            prompt_emb_neg = self.synthesis.pipe.encode_prompt("")
        
        # 3. 生成相机控制信号 (Interaction -> Control Signal)
        print(f"Generating camera embeddings for direction: {direction}...")
        self.operator.current_interaction = [] 
        self.operator.get_interaction(direction)
        
        camera_embedding_full = self.operator.process_interaction(
            modality_type=args.modality_type,
            start_frame=args.start_frame,
            initial_condition_frames=args.initial_condition_frames,
            total_frames_to_generate=args.total_frames_to_generate,
            use_real_poses=args.use_real_poses,
            scene_info_path=args.scene_info_path
        )

        camera_embedding_uncond = None
        if args.use_camera_cfg:
            camera_embedding_uncond = torch.zeros_like(camera_embedding_full)

        # 返回处理好的数据包
        return {
            "initial_latents": initial_latents,
            "prompt_emb_pos": prompt_emb_pos,
            "prompt_emb_neg": prompt_emb_neg,
            "camera_embedding_full": camera_embedding_full,
            "camera_embedding_uncond": camera_embedding_uncond
        }

    def __call__(self, 
                input_: str, 
                interaction: Dict[str, Any]) -> List[Image.Image]:
        """
        pipeline的调用入口
        
        Args:
            input_ (str): 图片路径
            interaction (dict): 交互参数 {'prompt': ..., 'direction': ...}
            
        Returns:
            frames (List[PIL.Image]): 生成的视频帧列表
        """
        args = self.config
        
        # 1. 预处理 (调用 process)
        processed_data = self.process(input_, interaction)
        
        initial_latents = processed_data["initial_latents"]
        camera_embedding_full = processed_data["camera_embedding_full"]
        prompt_emb_pos = processed_data["prompt_emb_pos"]
        prompt_emb_neg = processed_data["prompt_emb_neg"]
        camera_embedding_uncond = processed_data["camera_embedding_uncond"]

        # 2. 初始化记忆
        # 清空 Memory，并记录第一帧
        self.memory.storage = [] 
        self.memory.record(initial_latents)

        # 3. 循环生成
        total_generated = 0
        all_generated_frames = []

        while total_generated < args.total_frames_to_generate:
            current_generation = min(args.frames_per_generation, args.total_frames_to_generate - total_generated)
            print(f"Generation step: {total_generated}/{args.total_frames_to_generate}")
            
            framepack_data = self.memory.select(
                current_generation, 
                camera_embedding_full, 
                args.start_frame, 
                args.modality_type
            )
            
            new_latents = self.synthesis.predict(
                framepack_data, 
                current_generation, 
                prompt_emb_pos, 
                prompt_emb_neg, 
                args,
                camera_embedding_uncond
            )
            
            new_latents_squeezed = new_latents.squeeze(0)
            self.memory.record(new_latents_squeezed)
            
            all_generated_frames.append(new_latents_squeezed)
            total_generated += current_generation

        # 5. 解码
        all_generated = torch.cat(all_generated_frames, dim=1)
        final_video_latents = torch.cat([initial_latents.to(all_generated.device), all_generated], dim=1).unsqueeze(0)
        
        print("Decoding video...")
        # decode_video 返回的是 uint8 numpy array [T, H, W, C]
        video_np = self.synthesis.decode_video(final_video_latents)

        pil_frames = [Image.fromarray(frame) for frame in video_np]
        return pil_frames
