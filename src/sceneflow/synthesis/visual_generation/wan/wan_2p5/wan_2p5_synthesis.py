from dashscope import VideoSynthesis
from typing import Optional, Union, Dict, Any
from PIL import Image
import dashscope


class Wan2p5Synthesis(object):
    """
    Wan2.5 生成合成类，提供统一的接口用于音视频生成
    
    负责API调用和模型推理相关的工作
    """
    
    def __init__(
        self,
        endpoint: str = "https://dashscope.aliyuncs.com/api/v1",
        api_key: str = "your_api_key",
        logger=None,
    ):
        """
        初始化 Wan25Synthesis
        
        Args:
            endpoint: API基础URL
            api_key: API密钥
            logger: 日志记录器
        """
        self.endpoint = endpoint
        self.api_key = api_key
        self.logger = logger
        
        # 设置API基础URL
        dashscope.base_http_api_url = self.endpoint
    
    @classmethod
    def api_init(
        cls,
        endpoint: str = "https://dashscope.aliyuncs.com/api/v1",
        api_key: str = "your_api_key",
        logger=None,
        **kwargs
    ):
        """
        从配置创建 Wan25Synthesis 实例
        
        Args:
            endpoint: API基础URL
            api_key: API密钥
            logger: 日志记录器
            **kwargs: 其他参数
            
        Returns:
            Wan25Synthesis 实例
        """
        return cls(
            endpoint=endpoint,
            api_key=api_key,
            logger=logger,
        )
    
    def generate_t2av(
        self, 
        input_prompt: str,
        size: str = '832*480',
        duration: int = 10,
        negative_prompt: str = "",
        audio: bool = True,
        prompt_extend: bool = True,
        watermark: bool = False,
        seed: Optional[int] = None,
        **kwargs
    ):
        """
        文本到音视频生成
        
        Args:
            input_prompt: 输入提示词
            size: 视频尺寸，格式为 'width*height'
            duration: 视频时长（秒）
            negative_prompt: 负面提示词
            audio: 是否生成音频
            prompt_extend: 是否扩展提示词
            watermark: 是否添加水印
            seed: 随机种子
            **kwargs: 其他参数
            
        Returns:
            API响应结果
        """
        dashscope.base_http_api_url = self.endpoint
        rsp = VideoSynthesis.call(
            api_key=self.api_key,
            model='wan2.5-t2v-preview',
            prompt=input_prompt,
            size=size,
            duration=duration,
            negative_prompt=negative_prompt,
            audio=audio,
            prompt_extend=prompt_extend,
            watermark=watermark,
            seed=seed if seed is not None else 12345,
            **kwargs
        )
        return rsp

    def generate_i2av(
        self, 
        encoded_image: str,
        input_prompt: str,
        resolution: str = '480P',
        duration: int = 10,
        negative_prompt: str = "",
        audio: bool = True,
        prompt_extend: bool = True,
        watermark: bool = False,
        seed: Optional[int] = None,
        **kwargs
    ):
        """
        图像到音视频生成
        
        Args:
            encoded_image: 编码后的图像（base64格式）
            input_prompt: 输入提示词
            resolution: 视频分辨率
            duration: 视频时长（秒）
            negative_prompt: 负面提示词
            audio: 是否生成音频
            prompt_extend: 是否扩展提示词
            watermark: 是否添加水印
            seed: 随机种子
            **kwargs: 其他参数
            
        Returns:
            API响应结果
        """
        dashscope.base_http_api_url = self.endpoint
        rsp = VideoSynthesis.call(
            api_key=self.api_key,
            model='wan2.5-i2v-preview',
            prompt=input_prompt,
            img_url=encoded_image,
            resolution=resolution,
            duration=duration,
            negative_prompt=negative_prompt,
            audio=audio,
            prompt_extend=prompt_extend,
            watermark=watermark,
            seed=seed if seed is not None else 12345,
            **kwargs
        )
        return rsp
    
    
    def predict(
        self,
        processed_data: Dict[str, Any],
        task_type: str = "auto",
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
        生成预测结果
        
        Args:
            processed_data: 处理后的数据（来自operator）
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
        prompt = processed_data.get("prompt", "")
        encoded_image = processed_data.get("encoded_image", None)
        
        # 自动判断任务类型
        if task_type == "auto":
            if encoded_image is not None:
                task_type = "i2av"
            else:
                task_type = "t2av"
        
        # 根据任务类型调用相应的方法
        if task_type == "i2av":
            if encoded_image is None:
                raise ValueError("i2av 任务需要提供 encoded_image 参数")
            response = self.generate_i2av(
                encoded_image=encoded_image,
                input_prompt=prompt,
                resolution=resolution,
                duration=duration,
                negative_prompt=negative_prompt,
                audio=audio,
                prompt_extend=prompt_extend,
                watermark=watermark,
                seed=seed,
                **kwargs
            )
        elif task_type == "t2av":
            response = self.generate_t2av(
                input_prompt=prompt,
                size=size,
                duration=duration,
                negative_prompt=negative_prompt,
                audio=audio,
                prompt_extend=prompt_extend,
                watermark=watermark,
                seed=seed,
                **kwargs
            )
        else:
            raise ValueError(f"不支持的任务类型: {task_type}")
        
        return {
            "task_type": task_type,
            "prompt": prompt,
            "response": response
        }

