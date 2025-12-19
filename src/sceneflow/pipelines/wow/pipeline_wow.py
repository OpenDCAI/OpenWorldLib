import torch
import numpy as np
import os
from PIL import Image
from typing import Optional, Any, Union, Dict, List
from pathlib import Path

from ...operators.wow_operator import WowOperator
from ...synthesis.visual_generation.wow.wow_synthesis import WowSynthesis


class Args:
    pass

class WowPipeline:
    def __init__(self, 
    operator: Optional[WowOperator] = None, 
    synthesis_model: Optional[WowSynthesis] = None,
    synthesis_args=None,
    device: str = 'cuda'
    ):
        """
        初始化 WowPipeline
        Args:
            operator: WowOperator 实例
            synthesis_model: WowSynthesis 实例
            synthesis_args: WowSynthesis 参数
            device: 设备
        """
        if synthesis_args is None:
            synthesis_args = Args()
        
        self.operator = operator
        self.synthesis_model = synthesis_model
        self.synthesis_args = synthesis_args
        self.device = device


    @classmethod
    def from_pretrained(
        cls, 
        pretrained_model_path: str,
        synthesis_args=None,
        device: str = 'cuda',
        logger=None,
        **kwargs
    ) -> 'WowPipeline':
        """
        从预训练模型加载 WowPipeline
        Args:
            pretrained_model_path: 预训练模型路径
        Returns:
            WowPipeline: WowPipeline 实例
        """
        if logger:
            logger.info(f"Loading WowPipeline from {pretrained_model_path}")
        
        if synthesis_args is None:
            synthesis_args = Args()
        
        if logger:
            logger.info("Loading WowSynthesis model...")
        
        synthesis_model = WowSynthesis.from_pretrained(
            pretrained_model_path=pretrained_model_path, 
            synthesis_args=synthesis_args, 
            device=device, 
            logger=logger, 
            **kwargs)

        operator = WowOperator(

        )
        
        pipeline = cls(
            operator=operator, 
            synthesis_model=synthesis_model, 
            synthesis_args=synthesis_args, 
            device=device)

        return pipeline

    def process(
        self,
        prompt: str,
        image_path: str,
        **kwargs
    ) -> Dict[str, Any]:
        
        if self.operator is None:
            raise ValueError("Operator is not initialized")
        
        processed_data = self.operator.process_interaction(
            prompt=prompt,
            image_path=image_path,
            **kwargs
        )
        return processed_data

    def __call__(self, prompt: str, image_path: str):
        pass
        

    def save_pretrained(self, save_directory: str):
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
        
        print(f"WowPipeline saved to {save_directory}")

    def update_operator_config(self, **kwargs):
        if self.operator:
            self.operator.update_config(**kwargs)

    def get_operator(self) -> Optional[WowOperator]:
        return self.operator

    def get_synthesis_model(self) -> Optional[WowSynthesis]:
        return self.synthesis_model