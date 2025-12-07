from openai import OpenAI
from PIL import Image
from typing import Optional, Union, Dict, Any, Tuple
import mimetypes
import io
import logging


class Sora2Synthesis(object):
    """
    Sora2 生成合成类，提供统一的接口用于音视频生成
    
    负责API调用和模型推理相关的工作
    """
    
    def __init__(
        self,
        endpoint: str = "https://api.openai.com/v1",
        api_key: str = "your_api_key",
        logger=None,
    ):
        """
        初始化 Sora2Synthesis
        
        Args:
            endpoint: API基础URL
            api_key: API密钥
            logger: 日志记录器
        """
        self.endpoint = endpoint
        self.api_key = api_key
        self.logger = logger
        
        # 设置API基础URL
        self.client = OpenAI(api_key=self.api_key, base_url=self.endpoint)
        
        # 设置日志记录器
        if logger is not None:
            self.logger = logger
        else:
            self.logger = logging.getLogger(__name__)
    
    @classmethod
    def api_init(
        cls,
        endpoint: str = "https://api.openai.com/v1",
        api_key: str = "your_api_key",
        logger=None,
        **kwargs
    ):
        """
        从配置加载完整的 Sora2Synthesis
        """
        return cls(
            endpoint=endpoint, \
            api_key=api_key, 
            logger=logger,
        )

    def generate_t2av(
        self,
        input_prompt: str,
        size: str = "1280x720",
        duration: int = 8,
    ):
        """文本到视频生成（T2V）"""
        size = size.replace('*', 'x')  # 兼容用户误用 *
        return self.client.videos.create(
            model="sora-2",
            prompt=input_prompt,
            size=size,
            seconds=str(duration)
        )

    def generate_i2av(
        self,
        encoded_image: Tuple[str, bytes, str],
        input_prompt: str,
        size: str = "1280x720",
        duration: int = 8,
    ):
        """
        图像到视频生成（I2V）
        
        Args:
            encoded_image: 图像数据元组 (filename, bytes, mime_type)
            input_prompt: 输入提示词
            size: 视频尺寸
            duration: 视频时长（秒）
        """
        size = size.replace('*', 'x')
        
        filename, image_bytes, mime_type = encoded_image
        input_reference = (filename, image_bytes, mime_type)
        
        return self.client.videos.create(
            model="sora-2",
            prompt=input_prompt,
            size=size,
            seconds=str(duration),
            input_reference=input_reference
        )

    
    def predict(
        self,
        processed_data: Dict[str, Any],
        task_type: str = "auto",
        size: str = "1280x720",
        duration: int = 8,
        **kwargs
    ) -> Dict[str, Any]:
        prompt = processed_data.get("prompt", "")
        encoded_image = processed_data.get("encoded_image", None)

        if task_type == "auto":
            if encoded_image is not None:
                task_type = "i2av"
            else:
                task_type = "t2av"
        
        if task_type == "i2av":
            if encoded_image is None:
                raise ValueError("i2av 任务需要提供 encoded_image 参数")
            response = self.generate_i2av(
                encoded_image=encoded_image,
                input_prompt=prompt,
                size=size,
                duration=duration,
                **kwargs
            )
        elif task_type == "t2av":
            response = self.generate_t2av(
                input_prompt=prompt,
                size=size,
                duration=duration,
                **kwargs
            )
        else:
            raise ValueError(f"不支持的任务类型: {task_type}")
        
        return {
            "task_type": task_type,
            "prompt": prompt,
            "response": response
        }