from openai import OpenAI
from PIL import Image
from typing import Optional, Union, Dict, Any

from ...operators.sora2_operator import Sora2Operator
from ...synthesis.visual_generation.sora2.sora2_synthesis import Sora2Synthesis



class Sora2Pipeline:
    def __init__(
        self, 
        operator: Optional[Sora2Operator] = None,
        synthesis_model: Optional[Sora2Synthesis] = None,
        base_url: str = "https://api.openai.com/v1", 
        api_key: str = "your_api_key"):
        """
        初始化 Sora2Pipeline
        Args:
            operator: Sora2 operator 实例（如果为None则自动创建）
            synthesis_model: Sora2 synthesis 模型实例（如果为None则自动创建）
            base_url: API基础URL
            api_key: API密钥
        """
        self.operator = operator
        self.synthesis_model = synthesis_model
        self.base_url = base_url
        self.api_key = api_key

    @classmethod
    def from_pretrained(
        cls,
        base_url: str = "https://api.openai.com/v1",
        api_key: str = "your_api_key",
        logger=None,
        **kwargs
    ) -> 'Sora2Pipeline':
        """
        从配置加载完整的 pipeline
        
        Args:
            base_url: API基础URL
            api_key: API密钥
            logger: 日志记录器
            **kwargs: 额外参数
        """
        if logger:
            logger.info(f"Loading Sora2 pipeline with base_url: {base_url}")
        
        # 加载 synthesis 模型
        if logger:
            logger.info("Loading Sora2 synthesis model...")
        synthesis_model = Sora2Synthesis.from_pretrained(
            base_url=base_url,
            api_key=api_key,
            logger=logger,
            **kwargs
        )
        
        if logger:
            logger.info("Initializing Sora2 operator...")
        operator = Sora2Operator()
        
        pipeline = cls(
            operator=operator,
            synthesis_model=synthesis_model,
            base_url=base_url,
            api_key=api_key
        )
        
        if logger:
            logger.info("Sora2 pipeline loaded successfully")
        
        return pipeline

    def process(
        self,
        prompt: str,
        reference_image: Optional[Union[str, Image.Image]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        处理输入，通过 operator 预处理后传给 synthesis 模型
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
        size: str = "1280x720",
        duration: int = 8,
        task_type: str = "auto",
        **kwargs
    ) -> Dict[str, Any]:
        """
        自动根据是否提供 reference_image 选择 T2V 或 I2V
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
        response = self.synthesis_model.predict(
            processed_data=processed_data,
            task_type=task_type,
            size=size,
            duration=duration,
            **kwargs
        )

        return response

    def get_operator(self) -> Optional[Sora2Operator]:
        """获取 operator 实例"""
        return self.operator
    
    def get_synthesis_model(self) -> Optional[Sora2Synthesis]:
        """获取 synthesis 模型实例"""
        return self.synthesis_model