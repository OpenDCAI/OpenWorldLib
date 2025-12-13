from PIL import Image
from typing import Union, Optional, Dict, Any
from pathlib import Path
import mimetypes
import base64
import io

from .base_operator import BaseOperator


def encode_file(image_input: Union[str, Path, Image.Image]) -> str:
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
    elif isinstance(image_input, (str, Path)):
        img_path = Path(image_input)
        mime_type, _ = mimetypes.guess_type(img_path)
        if not mime_type or not mime_type.startswith("image/"):
            raise ValueError("不支持或无法识别的图像格式")
        if not img_path.exists():
            raise FileNotFoundError(f"Image file not found: {img_path}")
        with open(img_path, "rb") as image_file:
            image_bytes = image_file.read()
    else:
        raise TypeError("image_input 必须是文件路径(Path/str)或 PIL Image")
    
    encoded_string = base64.b64encode(image_bytes).decode('utf-8')
    return f"data:{mime_type};base64,{encoded_string}"


class Wan2p5Operator(BaseOperator):
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
        super(Wan2p5Operator, self).__init__(operation_types)
        
        # 初始化交互模板
        self.interaction_template = ["text_prompt", "image_prompt", "multimodal_prompt"]
        self.interaction_template_init()
    
    def process_image(self, image_input: Union[str, Path, Image.Image]) -> str:
        """
        编码图像为base64格式
        """
        return encode_file(image_input)
    
    def get_interaction(self, interaction):
        if self.check_interaction(interaction):
            self.current_interaction.append(interaction)

    def check_interaction(self, interaction):
        if not isinstance(interaction, str):
            raise TypeError(f"Interaction must be a string, got {type(interaction)}")
        return True

    def process_interaction(self,**kwargs) -> Dict[str, Any]:
        if len(self.current_interaction) == 0:
            raise ValueError("No interaction to process")
        now_interaction = self.current_interaction[-1]
        self.interaction_history.append(now_interaction)
        return {
            "processed_prompt": now_interaction
        }
    
    def process_perception(
        self,
        reference_image: Optional[Union[str, Image.Image, Path]] = None,
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
        result: Dict[str, Any] = {
            "encoded_image": None,
            "reference_image": None
        }
        
        # 简单判断路径是否存在
        if reference_image is not None:
            if isinstance(reference_image, (str, Path)):
                img_path = Path(reference_image)
                if not img_path.exists():
                    raise FileNotFoundError(f"Image file not found: {reference_image}")
            result["encoded_image"] = self.process_image(reference_image)
            result["reference_image"] = reference_image
        
        return result

