from __future__ import annotations

from multiprocessing import process
from typing import Optional, Union, Dict, Any, List

from PIL import Image

from ...operators.veo3_operator import Veo3Operator
from ...synthesis.visual_generation.veo.veo3.veo3_synthesis import Veo3Synthesis


class Veo3Pipeline:
    """
    使用 Chat Completions 端点实现的 Veo3 管线。
    通过检测是否携带图像自动区分 T2V / I2V。
    """

    def __init__(
        self,
        operator: Optional[Veo3Operator] = None,
        synthesis_model: Optional[Veo3Synthesis] = None,
        endpoint: str = "",
        api_key: str = "your_api_key",
    ):
        """
        初始化 Veo3Pipeline
        
        Args:
            operator: Veo3 operator 实例（如果为None则自动创建）
            synthesis_model: Veo3 synthesis 模型实例（如果为None则自动创建）
            api_key: API密钥
            endpoint: API基础URL
        """
        self.api_key = api_key
        self.endpoint = endpoint
        self.operator = operator
        self.synthesis_model = synthesis_model

    @classmethod
    def api_init(
        cls,
        api_key: str,
        endpoint: str = "",
        logger=None,
        **kwargs
    ) -> 'Veo3Pipeline':
        """
        从配置加载完整的 pipeline
        
        Args:
            api_key: API密钥
            endpoint: API基础URL
            logger: 日志记录器
            **kwargs: 额外参数
            
        Returns:
            Veo3Pipeline: 初始化的 pipeline 实例
        """
        if logger:
            logger.info(f"Loading Veo3 pipeline with endpoint: {endpoint}")
        
        # 加载 synthesis 模型
        if logger:
            logger.info("Loading Veo3 synthesis model...")
        synthesis_model = Veo3Synthesis.api_init(
            endpoint=endpoint,
            api_key=api_key,
            logger=logger,
            **kwargs
        )
        
        if logger:
            logger.info("Initializing Veo3 operator...")
        operator = Veo3Operator()
        
        pipeline = cls(
            operator=operator,
            synthesis_model=synthesis_model,
            api_key=api_key,
            endpoint=endpoint
        )
        
        if logger:
            logger.info("Veo3 pipeline loaded successfully")
        
        return pipeline

    def process(
        self,
        prompt: str,
        image: Optional[Union[str, Image.Image]] = None,
        aspect_ratio: str = "16:9",
        resolution: str = "720p",
        duration_seconds: int = 8,
        negative_prompt: Optional[str] = None,
        seed: Optional[int] = None,
        last_frame: Optional[Union[str, Image.Image]] = None,
        reference_images: Optional[List[Union[str, Image.Image]]] = None,
        person_generation: Optional[str] = None,
        enhance_prompt: Optional[bool] = None,
        generate_audio: Optional[bool] = None,
        fps: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        处理输入，通过 operator 预处理后传给 synthesis 模型
        
        Args:
            prompt: 文本提示词
            image: 主图像（可选）
            aspect_ratio: 宽高比
            resolution: 分辨率
            duration_seconds: 视频时长（秒）
            negative_prompt: 负面提示词
            seed: 随机种子
            last_frame: 最后一帧图像（可选）
            reference_images: 参考图像列表（可选）
            person_generation: 人物生成设置（可选）
            **kwargs: 其他参数
            
        Returns:
            Dict 包含处理后的数据
        """
        if self.operator is None:
            raise ValueError("Operator is not initialized")

        processed_data: Dict[str, Any] = {}

        self.operator.get_interaction(prompt)
        processed_interaction = self.operator.process_interaction()
        processed_data['prompt'] = processed_interaction['processed_prompt']
        
        processed_perception = self.operator.process_perception(
            prompt=prompt,
            image=image,
            aspect_ratio=aspect_ratio,
            resolution=resolution,
            duration_seconds=duration_seconds,
            negative_prompt=negative_prompt,
            seed=seed,
            last_frame=last_frame,
            reference_images=reference_images,
            person_generation=person_generation,
            enhance_prompt=enhance_prompt,
            generate_audio=generate_audio,
            fps=fps,
            **kwargs
        )

        processed_data['user_content'] = processed_perception['user_content']
        processed_data['image'] = processed_perception['image']
        processed_data['reference_images'] = processed_perception['reference_images']

        return processed_data

    def __call__(
        self,
        prompt: str,
        image: Optional[Union[str, Image.Image]] = None,
        task_type: str = "auto",
        aspect_ratio: str = "16:9",
        resolution: str = "720p",
        duration_seconds: int = 8,
        negative_prompt: Optional[str] = None,
        seed: Optional[int] = None,
        last_frame: Optional[Union[str, Image.Image]] = None,
        reference_images: Optional[List[Union[str, Image.Image]]] = None,
        person_generation: Optional[str] = None,
        enhance_prompt: Optional[bool] = None,
        generate_audio: Optional[bool] = None,
        fps: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        自动根据是否提供 image 选择 T2V 或 I2V
        
        Args:
            prompt: 文本提示词
            image: 主图像（可选），如果提供则自动使用 i2av
            task_type: 任务类型，"auto" 自动判断，"t2av" 文本到视频，"i2av" 图像到视频
            aspect_ratio: 宽高比
            resolution: 分辨率
            duration_seconds: 视频时长（秒）
            negative_prompt: 负面提示词
            seed: 随机种子
            last_frame: 最后一帧图像（可选）
            reference_images: 参考图像列表（可选）
            person_generation: 人物生成设置（可选）
            **kwargs: 其他参数
            
        Returns:
            Dict 包含生成结果：
                - task_type: 任务类型
                - result: 生成结果
        """
        if self.synthesis_model is None:
            raise ValueError("Synthesis model is not initialized")
        
        if self.operator is None:
            raise ValueError("Operator is not initialized")
        
        # 使用 operator 预处理输入
        processed_data = self.process(
            prompt=prompt,
            image=image,
            aspect_ratio=aspect_ratio,
            resolution=resolution,
            duration_seconds=duration_seconds,
            negative_prompt=negative_prompt,
            seed=seed,
            last_frame=last_frame,
            reference_images=reference_images,
            person_generation=person_generation,
            enhance_prompt=enhance_prompt,
            generate_audio=generate_audio,
            fps=fps,
            **kwargs
        )
        
        # 使用 synthesis 模型的 predict 方法进行推理
        response = self.synthesis_model.predict(
            processed_data=processed_data,
            task_type=task_type,
            **kwargs
        )

        return response

    def get_operator(self) -> Optional[Veo3Operator]:
        """获取 operator 实例"""
        return self.operator
    
    def get_synthesis_model(self) -> Optional[Veo3Synthesis]:
        """获取 synthesis 模型实例"""
        return self.synthesis_model
