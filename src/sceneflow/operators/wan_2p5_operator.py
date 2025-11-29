from PIL import Image
from typing import Union, Optional, Dict, Any
import mimetypes
import base64
import io

from .base_operator import BaseOperator


def encode_file(image_input: Union[str, Image.Image]) -> str:
    '''
    将图片编码为base64 格式
    
    Args:
        image_input: 图像路径或 PIL Image 对象
        
    Returns:
        base64编码的图像字符串
    '''
    if isinstance(image_input, Image.Image):
        if image_input.mode != 'RGB':
            image_input = image_input.convert('RGB')
        
        buffer = io.BytesIO()
        image_input.save(buffer, format='PNG')
        image_bytes = buffer.getvalue()
        mime_type = 'image/png'
    elif isinstance(image_input, str):
        mime_type, _ = mimetypes.guess_type(image_input)
        if not mime_type or not mime_type.startswith("image/"):
            raise ValueError("不支持或无法识别的图像格式")
        with open(image_input, "rb") as image_file:
            image_bytes = image_file.read()
    
    encoded_string = base64.b64encode(image_bytes).decode('utf-8')
    return f"data:{mime_type};base64,{encoded_string}"


class Wan25Operator(BaseOperator):
    """
    Wan2.5 数据处理 Operator
    
    负责图像编码、数据预处理等数据预处理工作
    不涉及模型推理和API调用
    """
    
    def __init__(
        self,
        operation_types: list = None
    ):
        """
        初始化 Wan25Operator
        
        Args:
            operation_types: 操作类型列表
        """
        if operation_types is None:
            operation_types = ["image_processing", "prompt_processing"]
        super(Wan25Operator, self).__init__(operation_types)
        
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
    
    def process_image(self, image_input: Union[str, Image.Image]) -> str:
        """
        编码图像为base64格式
        """
        return encode_file(image_input)
    
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
                - encoded_image: 编码后的图像（如果有）
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

