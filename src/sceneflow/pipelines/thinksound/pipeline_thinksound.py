import os
import torch
from pathlib import Path
from typing import Optional, Union, Dict, Any
from dataclasses import dataclass
from loguru import logger

from ...operators.thinksound_operator import ThinkSoundOperator
from ...synthesis.audio_generation.thinksound.thinksound_synthesis import ThinkSoundSynthesis


@dataclass
class ThinkSoundArgs:
    """ThinkSound 配置参数"""
    model_config: str = "ThinkSound/configs/model_configs/thinksound.json"
    # 主模型权重（对应 infer.sh 里的 ckpts/thinksound.ckpt）
    ckpt_dir: str = "ckpts/thinksound.ckpt"
    # VAE 权重（对应 infer.sh / defaults.ini 里的 ckpts/vae.ckpt）
    pretransform_ckpt_path: str = "ckpts/vae.ckpt"
    duration_sec: float = 8.0  # 用于初始化模型配置，可按需调整
    seed: int = 42
    compile: bool = False
    video_dir: str = "videos"
    cot_dir: str = "cot_coarse"
    results_dir: str = "results"
    scripts_dir: str = "."


class ThinkSoundPipeline:
    """
    ThinkSound Pipeline
    
    对外暴露 “视频 + 文本 -> 音频” 的统一接口
    """
    
    def __init__(
        self, 
        operator: Optional[ThinkSoundOperator] = None,
        synthesis_model: Optional[ThinkSoundSynthesis] = None, 
        synthesis_args: Optional[ThinkSoundArgs] = None, 
        device: str = 'cuda'
    ):
        """
        初始化 ThinkSoundPipeline
        
        Args:
            operator: ThinkSoundOperator 实例
            synthesis_model: ThinkSoundSynthesis 实例
            synthesis_args: 配置参数
            device: 设备
        """
        self.operator = operator
        self.synthesis_model = synthesis_model
        self.synthesis_args = synthesis_args or ThinkSoundArgs()
        self.device = device
    
    @classmethod
    def from_pretrained(
        cls, 
        synthesis_args: Optional[ThinkSoundArgs] = None, 
        device: str = None, 
        logger_obj=None,
        **kwargs
    ) -> 'ThinkSoundPipeline':
        """
        从预训练模型加载完整的 pipeline
        
        Args:
            synthesis_args: synthesis 模型参数，包含所有必要的路径（model_config, ckpt_dir 等）
            device: 设备
            logger_obj: 日志记录器
            
        Returns:
            ThinkSoundPipeline 实例
        """
        # 使用默认 args 如果未提供
        if synthesis_args is None:
            synthesis_args = ThinkSoundArgs()
        
        if logger_obj:
            logger_obj.info("Loading ThinkSound pipeline...")
        
        # 加载 synthesis 模型
        if logger_obj:
            logger_obj.info("Loading ThinkSound synthesis model...")
        
        synthesis_model = ThinkSoundSynthesis.from_pretrained(
            args=synthesis_args,
            device=device,
            logger_obj=logger_obj,
            **kwargs
        )
        
        operator = ThinkSoundOperator(
            video_dir=synthesis_args.video_dir,
            cot_dir=synthesis_args.cot_dir,
            results_dir=synthesis_args.results_dir,
            scripts_dir=synthesis_args.scripts_dir,
        )
        
        pipeline = cls(
            operator=operator,
            synthesis_model=synthesis_model,
            synthesis_args=synthesis_args,
            device=synthesis_model.device
        )
        
        if logger_obj:
            logger_obj.info("ThinkSound pipeline loaded successfully")
        
        return pipeline
    
    def process(
        self,
        video_path: Union[str, Path],
        title: str,
        description: str,
        use_half: bool = False,
        **kwargs
    ) -> Dict[str, Any]:

        if self.operator is None:
            raise ValueError("Operator is not initialized")
        
        processed_data = self.operator.process_perception(
            video_path=video_path,
            title=title,
            description=description,
            use_half=use_half,
            device=self.device,
            **kwargs
        )
        
        return processed_data
    
    def __call__(
        self,
        video_path: Union[str, Path],
        title: str,
        description: str,
        use_half: bool = False,
        cfg_scale: float = 5.0,
        num_steps: int = 24,
        **kwargs
    ) -> Dict[str, Any]:
        if self.synthesis_model is None:
            raise ValueError("Synthesis model is not initialized")
        
        processed_data = self.process(
            video_path=video_path,
            title=title,
            description=description,
            use_half=use_half,
            **kwargs
        )
        
        result = self.synthesis_model.predict(
            processed_data=processed_data,
            cfg_scale=cfg_scale,
            num_steps=num_steps,
            **kwargs
        )
        
        result["video_path"] = str(video_path)
        
        return result
    
    def save_pretrained(self, save_directory: Union[str, Path]):
        """
        保存 pipeline 配置到指定目录
        
        Args:
            save_directory: 保存目录路径
        """
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        
        # 保存配置
        if self.synthesis_args:
            import json
            config = {
                'model_config': self.synthesis_args.model_config,
                'ckpt_dir': self.synthesis_args.ckpt_dir,
                'pretransform_ckpt_path': self.synthesis_args.pretransform_ckpt_path,
                'duration_sec': self.synthesis_args.duration_sec,
                'seed': self.synthesis_args.seed,
                'compile': self.synthesis_args.compile,
            }
            with open(save_directory / "thinksound_config.json", 'w') as f:
                json.dump(config, f, indent=2)
        
        logger.info(f"ThinkSound Pipeline saved to {save_directory}")
    
    def get_operator(self) -> Optional[ThinkSoundOperator]:
        """获取 operator 实例"""
        return self.operator
    
    def get_synthesis_model(self) -> Optional[ThinkSoundSynthesis]:
        """获取 synthesis 模型实例"""
        return self.synthesis_model
    
