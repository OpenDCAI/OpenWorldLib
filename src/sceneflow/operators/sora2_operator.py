from PIL import Image
from typing import Union, Optional, Dict, Any, Tuple
import mimetypes
import io
import os

from .base_operator import BaseOperator


def image_to_bytes(image_input: Union[str, Image.Image]) -> Tuple[bytes, str, str]:
    """
    将图像转为字节并返回 mime 与文件名
    
    Args:
        image_input: 图像路径或 PIL Image 对象
        
    Returns:
        Tuple[bytes, str, str]: (图像字节, mime类型, 文件名)
    """
    if isinstance(image_input, Image.Image):
        # 处理 PIL Image
        if image_input.mode != 'RGB':
            image_input = image_input.convert('RGB')
        
        # 将 PIL Image 转换为字节
        buffer = io.BytesIO()
        image_input.save(buffer, format='PNG')
        image_bytes = buffer.getvalue()
        mime_type = 'image/png'
        filename = 'reference.png'
    elif isinstance(image_input, str):
        # 处理文件路径
        mime_type, _ = mimetypes.guess_type(image_input)
        if not mime_type or not mime_type.startswith('image/'):
            raise ValueError(f"无效图像路径或格式: {image_input}")
        with open(image_input, 'rb') as f:
            image_bytes = f.read()
        filename = os.path.basename(image_input)
    else:
        raise TypeError("image_input 必须是文件路径或 PIL Image")
    
    return image_bytes, mime_type, filename

class Sora2Operator(BaseOperator):
    """
    Sora2 数据处理 Operator
    
    负责图像编码、数据预处理等数据预处理工作
    不涉及模型推理和API调用
    """
    
    def __init__(
        self,
        operation_types: list = None
    ):
        """
        初始化 Sora2Operator
        
        Args:
            operation_types: 操作类型列表
        """
        if operation_types is None:
            operation_types = ["image_processing", "prompt_processing"]
        super(Sora2Operator, self).__init__(operation_types)
        
        # 初始化交互模板   
        self.interaction_template = ["text_prompt", "image_prompt", "multimodal_prompt"]
        self.interaction_template_init()
    
    def check_interaction(self, interaction):
        """检查交互类型是否有效"""
        if not isinstance(interaction, str):
            raise TypeError(f"Invalid interaction")
        return True
    
    def get_interaction(self, interaction):
        """获取交互类型"""
        if self.check_interaction(interaction):
            self.current_interaction = interaction

    def process_image(self, image_input: Union[str, Image.Image]) -> Tuple[str, bytes, str]:
        """
        处理图像，返回文件名、字节和mime类型（API所需格式）
        
        Args:
            image_input: 图像路径或 PIL Image 对象
            
        Returns:
            Tuple[str, bytes, str]: (文件名, 图像字节, mime类型)
        """
        image_bytes, mime_type, filename = image_to_bytes(image_input)
        return (filename, image_bytes, mime_type)

    def process_interaction(
        self,
        prompt: str,
        reference_image: Optional[Union[str, Image.Image]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        处理交互输入，生成模型所需的输入格式
        
        Args:
            prompt: 文本提示词
            reference_image: 参考图像（可选）
            **kwargs: 其他参数
            
        Returns:
            Dict 包含处理后的输入数据：
                - prompt: 文本提示词
                - encoded_image: 图像元组 (filename, bytes, mime_type)（如果有）
                - reference_image: 原始参考图像（如果有）
        """
        self.get_interaction(prompt)
        result: Dict[str, Any] = {
            "prompt": self.current_interaction,
            "encoded_image": None,
            "reference_image": None
        }
        
        # 处理图像（如果提供）
        if reference_image is not None:
            result["encoded_image"] = self.process_image(reference_image)
            result["reference_image"] = reference_image
        
        return result