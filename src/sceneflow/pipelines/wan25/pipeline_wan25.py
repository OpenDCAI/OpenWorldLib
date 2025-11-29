# 这里面包含了sora2， veo3 和wan2.5，主要包含api调用
from PIL import Image
from typing import Optional, Union, Dict, Any
from pathlib import Path

from ...operators.wan_2p5_operator import Wan25Operator
from ...synthesis.visual_generation.wan.wan_2p5.wan_2p5_synthesis import Wan25Synthesis


class Wan25Pipeline:
    """
    将输入通过 operator 处理后再传给模型进行推理，
    实现数据预处理和模型推理的分离。
    """
    
    def __init__(
        self,
        operator: Optional[Wan25Operator] = None,
        synthesis_model: Optional[Wan25Synthesis] = None,
        base_url: str = "https://dashscope.aliyuncs.com/api/v1",
        api_key: str = "your_api_key",
    ):
        """
        初始化 Wan25Pipeline
        
        Args:
            operator: Wan25 operator 实例（如果为None则自动创建）
            synthesis_model: Wan25 synthesis 模型实例（如果为None则自动创建）
            base_url: API基础URL
            api_key: API密钥
        """
        self.base_url = base_url
        self.api_key = api_key
        self.operator = operator
        self.synthesis_model = synthesis_model
    
    @classmethod
    def from_pretrained(
        cls,
        base_url: str = "https://dashscope.aliyuncs.com/api/v1",
        api_key: str = "your_api_key",
        logger=None,
        **kwargs
    ) -> 'Wan25Pipeline':
        """
        从配置加载完整的 pipeline
        
        Args:
            base_url: API基础URL
            api_key: API密钥
            logger: 日志记录器
            **kwargs: 额外参数
            
        Returns:
            Wan25Pipeline: 初始化的 pipeline 实例
        """
        if logger:
            logger.info(f"Loading Wan25 pipeline with base_url: {base_url}")
        
        # 加载 synthesis 模型
        if logger:
            logger.info("Loading Wan25 synthesis model...")
        
        synthesis_model = Wan25Synthesis.from_pretrained(
            base_url=base_url,
            api_key=api_key,
            logger=logger,
            **kwargs
        )
        
        if logger:
            logger.info("Initializing Wan25 operator...")
        
        operator = Wan25Operator()
        
        pipeline = cls(
            operator=operator,
            synthesis_model=synthesis_model,
            base_url=base_url,
            api_key=api_key
        )
        
        if logger:
            logger.info("Wan25 pipeline loaded successfully")
        
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
            prompt: 文本提示词
            reference_image: 参考图像（可选）
            **kwargs: 其他参数
            
        Returns:
            Dict 包含处理后的数据
        """
        if self.operator is None:
            raise ValueError("Operator is not initialized")
        
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
        task_type: str = "auto",  # "auto", "t2av", "i2av"
        size: str = '832*480',
        resolution: str = '480P',
        duration: int = 10,
        negative_prompt: str = "",
        audio: bool = True,
        prompt_extend: bool = True,
        watermark: bool = False,
        seed: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        统一的调用接口，自动判断任务类型
        
        Args:
            prompt: 文本提示词
            reference_image: 参考图像（可选），如果提供则使用 i2av，否则使用 t2av
            task_type: 任务类型，"auto" 自动判断，"t2av" 文本到视频，"i2av" 图像到视频
            size: t2av 任务的视频尺寸
            resolution: i2av 任务的分辨率
            duration: 视频时长（秒）
            negative_prompt: 负面提示词
            audio: 是否生成音频
            prompt_extend: 是否扩展提示词
            watermark: 是否添加水印
            seed: 随机种子
            **kwargs: 其他参数
            
        Returns:
            Dict 包含生成的结果：
                - response: API响应对象
                - task_type: 实际使用的任务类型
                - prompt: 使用的提示词
        """
        if self.synthesis_model is None:
            raise ValueError("Synthesis model is not initialized")
        
        if self.operator is None:
            raise ValueError("Operator is not initialized")
        
        # 使用 operator 预处理输入
        processed_data = self.process(
            prompt=prompt,
            reference_image=reference_image,
            **kwargs
        )
        
        # 使用 synthesis 模型的 predict 方法进行推理
        result = self.synthesis_model.predict(
            processed_data=processed_data,
            task_type=task_type,
            size=size,
            resolution=resolution,
            duration=duration,
            negative_prompt=negative_prompt,
            audio=audio,
            prompt_extend=prompt_extend,
            watermark=watermark,
            seed=seed,
            **kwargs
        )
        
        return result
    
    def get_operator(self) -> Optional[Wan25Operator]:
        """获取 operator 实例"""
        return self.operator
    
    def get_synthesis_model(self) -> Optional[Wan25Synthesis]:
        """获取 synthesis 模型实例"""
        return self.synthesis_model
