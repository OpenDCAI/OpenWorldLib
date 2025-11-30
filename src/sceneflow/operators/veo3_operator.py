from __future__ import annotations

import base64
import json
import mimetypes
from typing import Optional, Union, Dict, Any, List, Tuple

from PIL import Image
from .base_operator import BaseOperator


def _image_to_data_url(image_input: Union[str, Image.Image]) -> Tuple[str, str]:
    """
    将图像转为 data URL 与 MIME 类型，方便在 Chat Completion 消息中以 base64 形式携带。
    
    Args:
        image_input: 图像路径或 PIL Image 对象
        
    Returns:
        Tuple[str, str]: (data_url, mime_type)
    """
    mime_type = "image/png"
    if isinstance(image_input, Image.Image):
        if image_input.mode != "RGB":
            image_input = image_input.convert("RGB")
        from io import BytesIO

        buffer = BytesIO()
        image_input.save(buffer, format="PNG")
        content = buffer.getvalue()
    elif isinstance(image_input, str):
        mime_type_guess, _ = mimetypes.guess_type(image_input)
        if mime_type_guess:
            mime_type = mime_type_guess
        with open(image_input, "rb") as f:
            content = f.read()
    else:
        raise TypeError("image_input 必须是文件路径或 PIL.Image")

    data_url = f"data:{mime_type};base64,{base64.b64encode(content).decode('utf-8')}"
    return data_url, mime_type


class Veo3Operator(BaseOperator):
    """
    Veo3 数据处理 Operator
    
    负责图像编码、数据预处理等数据预处理工作
    不涉及模型推理和API调用
    """
    
    def __init__(
        self,
        operation_types: list = None
    ):
        """
        初始化 Veo3Operator
        
        Args:
            operation_types: 操作类型列表
        """
        if operation_types is None:
            operation_types = ["image_processing", "prompt_processing"]
        super(Veo3Operator, self).__init__(operation_types)
        
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
    
    def process_image(self, image_input: Union[str, Image.Image]) -> Tuple[str, str]:
        """
        处理图像，返回 data URL 和 mime 类型
        
        Args:
            image_input: 图像路径或 PIL Image 对象
            
        Returns:
            Tuple[str, str]: (data_url, mime_type)
        """
        return _image_to_data_url(image_input)

    def build_user_content(
        self,
        prompt: str,
        *,
        aspect_ratio: str,
        resolution: str,
        duration_seconds: int,
        negative_prompt: Optional[str] = None,
        seed: Optional[int] = None,
        person_generation: Optional[str] = None,
        reference_images: Optional[List[Union[str, Image.Image]]] = None,
        image: Optional[Union[str, Image.Image]] = None,   # 可为 None（T2V）
        last_frame: Optional[Union[str, Image.Image]] = None,
        enhance_prompt: Optional[bool] = None,
        generate_audio: Optional[bool] = None,
        fps: Optional[int] = None,
    ) -> List[Dict[str, Any]]:

        # 自动判断模式
        is_i2v = image is not None

        metadata = {
            "aspect_ratio": aspect_ratio,
            "resolution": resolution,
            "duration_seconds": duration_seconds,
            "negative_prompt": negative_prompt,
            "seed": seed,
            "person_generation": person_generation,
            "enhance_prompt": enhance_prompt,
            "generate_audio": generate_audio,
            "fps": fps,
        }

        # I2V 特有字段（仅在 I2V 时设置）
        if is_i2v:
            metadata["has_last_frame"] = last_frame is not None
            metadata["reference_image_count"] = len(reference_images) if reference_images else 0

        content: List[Dict[str, Any]] = [
            {
                "type": "text",
                "text": json.dumps({
                    "instruction": prompt,
                    "metadata": metadata
                }, ensure_ascii=False),
            }
        ]

        def to_data_url(img):
            url, _ = _image_to_data_url(img)
            return {"type": "image_url", "image_url": {"url": url}}

        if is_i2v:
            content.append(to_data_url(image))

            if reference_images:
                for img in reference_images:
                    content.append(to_data_url(img))

            if last_frame is not None:
                content.append(to_data_url(last_frame))

        # # 调试
        # with open('./test/content.json', 'w', encoding='utf-8') as f:
        #     json.dump(content, f, ensure_ascii=False, indent=4)
        # print('-------------content saved to content.json-------------')

        return content
        
    
    def process_interaction(
        self,
        prompt: str,
        *,
        image: Optional[Union[str, Image.Image]] = None,
        aspect_ratio: str = "16:9",
        resolution: str = "720p",
        duration_seconds: int = 8,
        negative_prompt: Optional[str] = None,
        seed: Optional[int] = None,
        last_frame: Optional[Union[str, Image.Image]] = None,
        reference_images: Optional[List[Union[str, Image.Image]]] = None,
        person_generation: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        处理交互输入，生成模型所需的输入格式
        
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
            Dict 包含处理后的输入数据：
                - prompt: 文本提示词
                - user_content: 构建好的用户内容列表
                - image: 主图像（如果有）
                - reference_images: 参考图像列表（如果有）
        """
        self.get_interaction(prompt)
        
        user_content = self.build_user_content(
            prompt=self.current_interaction,
            aspect_ratio=aspect_ratio,
            resolution=resolution,
            duration_seconds=duration_seconds,
            negative_prompt=negative_prompt,
            seed=seed,
            person_generation=person_generation,
            reference_images=reference_images,
            image=image,
            last_frame=last_frame,
        )
        
        result: Dict[str, Any] = {
            "prompt": self.current_interaction,
            "user_content": user_content,
            "image": image,
            "reference_images": reference_images,
        }
        
        return result

