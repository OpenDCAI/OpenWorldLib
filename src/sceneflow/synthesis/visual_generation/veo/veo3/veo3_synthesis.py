from __future__ import annotations

from typing import Dict, Any, List
import logging

from openai import OpenAI


class Veo3Synthesis(object):
    """
    Veo3 生成合成类，提供统一的接口用于音视频生成
    
    负责API调用和模型推理相关的工作
    """
    
    def __init__(
        self,
        endpoint: str,
        api_key: str,
        logger=None,
    ):
        """
        初始化 Veo3Synthesis
        
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
        endpoint: str,
        api_key: str,
        logger=None,
        **kwargs
    ):
        """
        从配置加载完整的 Veo3Synthesis
        
        Args:
            endpoint: API基础URL
            api_key: API密钥
            logger: 日志记录器
            **kwargs: 其他参数
            
        Returns:
            Veo3Synthesis 实例
        """
        return cls(
            endpoint=endpoint,
            api_key=api_key,
            logger=logger,
        )
    
    def _invoke_chat_completion(
        self,
        *,
        messages: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        调用 Chat Completions API
        
        Args:
            messages: 消息列表
            
        Returns:
            Dict 包含响应信息：
                - response: API响应对象
                - assistant_message: 助手消息
                - parsed: 解析后的视频负载
        """
        response = self.client.chat.completions.create(
            model="veo3.1",
            messages=messages,
        )

        return response
    
    def generate_t2av(
        self,
        processed_data: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        文本到视频生成（T2V）
        
        Args:
            processed_data: 处理后的数据（来自operator），包含已构建好的 user_content
            **kwargs: 其他参数（保留以兼容接口）
            
        Returns:
            Dict 包含生成结果
        """
        user_content = processed_data.get("user_content", [])
        messages: List[Dict[str, Any]] = []
        messages.append({"role": "user", "content": user_content})
        return self._invoke_chat_completion(messages=messages)
    
    def generate_i2av(
        self,
        processed_data: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        图像到视频生成（I2V）
        
        Args:
            processed_data: 处理后的数据（来自operator），包含已构建好的 user_content
            **kwargs: 其他参数（保留以兼容接口）
            
        Returns:
            Dict 包含生成结果
        """
        user_content = processed_data.get("user_content", [])
        messages: List[Dict[str, Any]] = []
        messages.append({"role": "user", "content": user_content})
        return self._invoke_chat_completion(messages=messages)
    
    def predict(
        self,
        processed_data: Dict[str, Any],
        task_type: str = "auto",
        **kwargs
    ) -> Dict[str, Any]:
        """
        预测方法，统一接口
        
        Args:
            processed_data: 处理后的数据（来自operator）
            task_type: 任务类型，"auto" 自动判断，"t2av" 文本到视频，"i2av" 图像到视频
            **kwargs: 其他参数
            
        Returns:
            Dict 包含生成结果：
                - task_type: 任务类型
                - result: 生成结果
        """
        image = processed_data.get("image", None)
        
        if task_type == "auto":
            if image is not None:
                task_type = "i2av"
            else:
                task_type = "t2av"
        
        if task_type == "i2av":
            if image is None:
                raise ValueError("i2av 任务需要提供 image 参数")
            result = self.generate_i2av(
                processed_data=processed_data,
                **kwargs
            )
        elif task_type == "t2av":
            result = self.generate_t2av(
                processed_data=processed_data,
                **kwargs
            )
        else:
            raise ValueError(f"不支持的任务类型: {task_type}")
        
        return {
            "task_type": task_type,
            "result": result,
        }

