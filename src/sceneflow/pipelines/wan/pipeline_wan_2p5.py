# 这里面包含了sora2， veo3 和wan2.5，主要包含api调用
from PIL import Image
from typing import Optional, Union, Dict, Any
from pathlib import Path
from http import HTTPStatus

from ...operators.wan_2p5_operator import Wan2p5Operator
from ...synthesis.visual_generation.wan.wan_2p5.wan_2p5_synthesis import Wan2p5Synthesis


class Wan2p5Pipeline:
    """
    将输入通过 operator 处理后再传给模型进行推理，
    实现数据预处理和模型推理的分离。
    """
    
    def __init__(
        self,
        operator: Optional[Wan2p5Operator] = None,
        synthesis_model: Optional[Wan2p5Synthesis] = None,
        endpoint: str = "https://dashscope.aliyuncs.com/api/v1",
        api_key: str = "your_api_key",
    ):
        """
        初始化 Wan2p5Pipeline
        
        Args:
            operator: Wan2p5 operator 实例（如果为None则自动创建）
            synthesis_model: Wan2p5 synthesis 模型实例（如果为None则自动创建）
            endpoint: API基础URL
            api_key: API密钥
        """
        self.endpoint = endpoint
        self.api_key = api_key
        self.operator = operator
        self.synthesis_model = synthesis_model
    
    @classmethod
    def api_init(
        cls,
        endpoint: str = "https://dashscope.aliyuncs.com/api/v1",
        api_key: str = "your_api_key",
        logger=None,
        **kwargs
    ) -> 'Wan2p5Pipeline':
        """
        从配置加载完整的 pipeline
        
        Args:
            endpoint: API基础URL
            api_key: API密钥
            logger: 日志记录器
            **kwargs: 额外参数
            
        Returns:
            Wan2p5Pipeline: 初始化的 pipeline 实例
        """
        if logger:
            logger.info(f"Loading Wan2p5 pipeline with endpoint: {endpoint}")
        
        # 加载 synthesis 模型
        if logger:
            logger.info("Loading Wan2p5 synthesis model...")
        
        synthesis_model = Wan2p5Synthesis.api_init(
            endpoint=endpoint,
            api_key=api_key,
            logger=logger,
            **kwargs
        )
        
        if logger:
            logger.info("Initializing Wan2p5 operator...")
        
        operator = Wan2p5Operator()
        
        pipeline = cls(
            operator=operator,
            synthesis_model=synthesis_model,
            endpoint=endpoint,
            api_key=api_key
        )
        
        if logger:
            logger.info("Wan2p5 pipeline loaded successfully")
        
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

        processed_data:Dict[str, Any] = {}

        self.operator.get_interaction(prompt)
        processed_interaction = self.operator.process_interaction()  
        processed_data['prompt'] = processed_interaction['processed_prompt']
        
        processed_perception = self.operator.process_perception(
            reference_image=reference_image,
            **kwargs
        )
        processed_data['encoded_image'] = processed_perception['encoded_image']
        processed_data['reference_image'] = processed_perception['reference_image']
        
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
        save_content: bool = True,
        output_dir: str = "./output/wan25",
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
                - video_url: 视频URL（如果API调用成功）
                - task_id: 任务ID（如果API调用成功）
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

        # 提取视频信息（video_url 和 task_id）
        response = result.get("response")
        task_type = result.get("task_type")
        
        # 检查响应状态
        if response and hasattr(response, 'status_code') and response.status_code == HTTPStatus.OK:
            print(f"API调用成功，状态码: {response.status_code}")
            
            video_url = None
            task_id = None
            if hasattr(response, 'output') and response.output:
                output_data = response.output
                video_url = output_data.get('video_url')
                task_id = output_data.get('task_id')
            
            result['video_url'] = video_url
            result['task_id'] = task_id
        else:
            if response and hasattr(response, 'status_code'):
                print(f"API调用失败，状态码: {response.status_code}")
                if hasattr(response, 'message'):
                    print(f"错误信息: {response.message}")
            result['video_url'] = None
            result['task_id'] = None
        
        return result
    
    def get_operator(self) -> Optional[Wan2p5Operator]:
        """获取 operator 实例"""
        return self.operator
    
    def get_synthesis_model(self) -> Optional[Wan2p5Synthesis]:
        """获取 synthesis 模型实例"""
        return self.synthesis_model
