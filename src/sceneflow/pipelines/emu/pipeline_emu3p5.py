import torch
import numpy as np
import os
from PIL import Image
from typing import Optional, Any, Union, Dict, List
from pathlib import Path

from ...operators.emu3p5_operator import Emu3p5Operator
from ...synthesis.visual_generation.emu.emu3p5_synthesis import Emu3p5Synthesis, load_models


class Args:
    model_path = None # 如果为None，则使用 pretrained_model_path/Emu3.5
    tokenizer_path = None  # 如果为None，则使用 pretrained_model_path/tokenizer_emu3_ibq
    vq_path = None  # 如果为None，则使用 pretrained_model_path/Emu3.5-VisionTokenizer
    vq_type = "ibq"
    model_device = "auto"
    vq_device = "cuda:0"
    task_type = "t2i"  # 可选: "t2i", "x2i", "howto", "story", "explore", "vla"
    use_image = True  # 是否使用参考图像
    image_area = 1048576
    classifier_free_guidance = 5.0
    unconditional_type = "no_text"
    max_new_tokens = 5120
    text_top_k = 1024
    text_top_p = 0.9
    text_temperature = 1.0
    image_top_k = 5120
    image_top_p = 1.0
    image_temperature = 1.0
    use_differential_sampling = True
    streaming = False

args = Args()

class Emu3p5Pipeline:
    """
    
    将输入通过 operator 处理后再传给模型进行推理，
    实现数据预处理和模型推理的分离。
    """
    
    def __init__(
        self,
        operator: Optional[Emu3p5Operator] = None,
        synthesis_model: Optional[Emu3p5Synthesis] = None,
        synthesis_args=None,
        device: str = 'cuda'
    ):
        """
        初始化 EmuPipeline
        
        Args:
            operator: EMU operator 实例
            synthesis_model: EMU synthesis 模型实例
            synthesis_args: synthesis 模型参数
            device: 设备
        """
        self.operator = operator
        self.synthesis_model = synthesis_model
        self.synthesis_args = args
        self.device = device
        
    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: Union[str, Path],
        synthesis_args=args,
        task_type: str = "t2i",
        use_image: bool = True,
        device: str = "cuda",
        logger=None,
        **kwargs
    ) -> 'EmuPipeline':
        """
        从预训练模型加载完整的 pipeline
        
        Args:
            pretrained_model_path: 预训练模型路径
            synthesis_args: synthesis 模型参数
            operator_config: operator 配置参数
            device: 设备
            logger: 日志记录器
            **kwargs: 额外参数
            
        Returns:
            EmuPipeline: 初始化的 pipeline 实例
        """
        if logger:
            logger.info(f"Loading EMU pipeline from {pretrained_model_path}")
        
        # 加载 synthesis 模型
        if logger:
            logger.info("Loading EMU synthesis model...")
        
        synthesis_model = Emu3p5Synthesis.from_pretrained(
            pretrained_model_path=pretrained_model_path,
            args=synthesis_args,
            device=device,
            logger=logger,
            **kwargs
        )
        
        # 初始化 operator
        if logger:
            logger.info("Initializing EMU operator...")
        
        operator = Emu3p5Operator(
            tokenizer=synthesis_model.tokenizer,
            vq_model=synthesis_model.vq_model,
            task_type=task_type,
            use_image=use_image,
            image_area=args.image_area,
        )
        
        # 创建并返回 pipeline 实例
        pipeline = cls(
            operator=operator,
            synthesis_model=synthesis_model,
            synthesis_args=synthesis_args,
            device=device
        )
        
        if logger:
            logger.info("EMU pipeline loaded successfully")
        
        return pipeline
    
    def process(
        self,
        prompt: str,
        reference_image: Optional[Union[str, Image.Image]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        处理输入，通过 operator 预处理后传给 synthesis 模型
        
        Args:
            prompt: 文本提示
            reference_image: 参考图像（可选）
            **kwargs: 其他参数
            
        Returns:
            Dict 包含处理后的数据
        """
        if self.operator is None:
            raise ValueError("Operator is not initialized")
        
        # 通过 operator 处理输入
        processed_data = self.operator.process_interaction(
            prompt=prompt,
            reference_image=reference_image,
            **kwargs
        )
        
        return processed_data
    
    def __call__(
        self,
        prompt: str,
        reference_image: Optional[Union[str, Image.Image]] = None,
        seed: Optional[Union[int, List[int]]] = None,
        batch_size: int = 1,
        num_images_per_prompt: int = 1,
        max_new_tokens: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        use_operator: bool = True,
        save_content= True,
        **kwargs
    ) -> Dict:
        """
        生成预测结果
        
        Args:
            prompt: 文本提示
            reference_image: 参考图像（可选）
            seed: 随机种子
            batch_size: 批次大小
            num_images_per_prompt: 每个 prompt 生成的图像数量
            max_new_tokens: 最大生成 token 数
            guidance_scale: CFG 引导尺度
            use_operator: 是否使用 operator 进行预处理
            **kwargs: 其他参数
            
        Returns:
            Dict 包含生成的结果
        """
        if self.synthesis_model is None:
            raise ValueError("Synthesis model is not initialized")
        
        # 使用 operator 预处理输入
        processed_data = self.process(prompt, reference_image, **kwargs)
        result = self.synthesis_model.predict(
            processed_data=processed_data,
            seed=seed,
            batch_size=batch_size,
            num_images_per_prompt=num_images_per_prompt,
            max_new_tokens=max_new_tokens,
            guidance_scale=guidance_scale,
            **kwargs
        )
        if save_content:
            for idx, (item_type, content) in enumerate(result["samples"]):
                if item_type == "image":
                    content.save(f"output_image_{idx}.png")
                    print(f"Saved image: output_image_{idx}.png")
                elif item_type == "text":
                    print(f"Generated text: {content}")
        
        return result
    
    def save_pretrained(self, save_directory: str):
        """
        保存 pipeline 到指定目录
        
        Args:
            save_directory: 保存目录
        """
        os.makedirs(save_directory, exist_ok=True)
        
        # 保存 synthesis 模型（如果有的话）
        if self.synthesis_model:
            synthesis_dir = os.path.join(save_directory, "synthesis_model")
            os.makedirs(synthesis_dir, exist_ok=True)
        
        # 保存 operator 配置
        if self.operator:
            operator_config = {
                'task_type': self.operator.task_type,
                'use_image': self.operator.use_image,
                'image_area': self.operator.image_area,
                'operation_types': self.operator.opration_types if hasattr(self.operator, 'opration_types') else []
            }
            torch.save(operator_config, os.path.join(save_directory, "operator_config.pt"))
        
        # 保存 pipeline 配置
        pipeline_config = {
            'device': self.device,
            'synthesis_args': self.synthesis_args
        }
        torch.save(pipeline_config, os.path.join(save_directory, "pipeline_config.pt"))
        
        print(f"EMU Pipeline saved to {save_directory}")
    
    def update_operator_config(self, **kwargs):
        """
        更新 operator 配置
        
        Args:
            **kwargs: 配置参数
        """
        if self.operator:
            self.operator.update_config(**kwargs)
    
    def get_operator(self) -> Optional[Emu3p5Operator]:
        """获取 operator 实例"""
        return self.operator
    
    def get_synthesis_model(self) -> Optional[Emu3p5Synthesis]:
        """获取 synthesis 模型实例"""
        return self.synthesis_model