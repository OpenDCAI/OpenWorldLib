import numpy as np
from PIL import Image
import torch
from typing import Union, Optional, Dict, Any
import torchvision.transforms as T

from .base_operator import BaseOperator

def smart_resize(image: Image.Image, area: int = 512 * 512, ds_factor: int = 16):
    width, height = image.size
    aspect_ratio = width / height
    new_height = int((area / aspect_ratio) ** 0.5)
    new_width = int(new_height * aspect_ratio)
    # Round to nearest multiple of divisible_by
    new_height = ((new_height + ds_factor//2) // ds_factor) * ds_factor
    new_width = ((new_width + ds_factor//2) // ds_factor) * ds_factor
    return image.resize((new_width, new_height), Image.BICUBIC)

def format_image_string(tokenizer, image_tokens):
    image_string = ""
    h, w = image_tokens.shape
    for _h in range(h):
        row_string = ""
        for _w in range(w):
            row_string += "<|visual token {token_id:0>6d}|>".format(token_id=image_tokens[_h, _w])

        if _h < h - 1:
            row_string += tokenizer.eol_token
        image_string += row_string

    return "{image_start}{token_height}*{token_width}{image_token}{token_str}{image_end}".format(
        image_start=tokenizer.boi_token,
        token_height=h,
        token_width=w,
        image_token=tokenizer.img_token,
        token_str=image_string,
        image_end=tokenizer.eoi_token,
    )

@torch.no_grad()
def build_image(image, cfg, tokenizer, vq_model):
    image = smart_resize(image, cfg.image_area)
    w, h = image.size
    device = next(vq_model.parameters()).device
    dtype = next(vq_model.parameters()).dtype
    image = torch.tensor((np.array(image) / 127.5 - 1.0)).to(device, dtype).permute(2, 0, 1)
    _, _, token = vq_model.encode(image[None])
    token = token[-1].view(h // 16, w // 16)
    return format_image_string(tokenizer, token)

class Emu3p5Operator(BaseOperator):
    
    def __init__(
        self,
        tokenizer=None,
        vq_model=None,
        task_type: str = "t2i",
        use_image: bool = True,
        image_area: int = 518400,
        operation_types=["prompt_processing", "image_processing"]
    ):
        """
        初始化 EmuOperator
        
        Args:
            tokenizer: 文本 tokenizer
            vq_model: Vision tokenizer (VQ model)
            task_type: 任务类型，默认为 "story"
            use_image: 是否使用图像，默认为 True
            image_area: 图像区域大小，默认为 518400
            operation_types: 操作类型列表
        """
        super(Emu3p5Operator, self).__init__(operation_types)
        
        self.tokenizer = tokenizer
        self.vq_model = vq_model
        self.task_type = task_type
        self.use_image = use_image
        self.image_area = image_area
        
        # 构建 prompt template
        self._build_prompt_template()
        
        # 初始化交互模板（如果需要）
        self.interaction_template = ["text_prompt", "image_prompt", "multimodal_prompt"]
        self.interaction_template_init()
    
    def _build_prompt_template(self):
        """构建 prompt template 和 unconditional prompt"""
        task_str = self.task_type.lower()
        if self.use_image:
            self.unconditional_prompt = "<|extra_203|>You are a helpful assistant. USER: <|IMAGE|> ASSISTANT: <|extra_100|>"
            self.template = f"<|extra_203|>You are a helpful assistant for {task_str} task. USER: {{question}}<|IMAGE|> ASSISTANT: <|extra_100|>"
        else:
            self.unconditional_prompt = "<|extra_203|>You are a helpful assistant. USER:  ASSISTANT: <|extra_100|>"
            self.template = f"<|extra_203|>You are a helpful assistant for {task_str} task. USER: {{question}} ASSISTANT: <|extra_100|>"
    
    def check_interaction(self, interaction):
        """检查交互类型是否有效"""
        if not isinstance(interaction, str):
            raise TypeError(f"Invalid interaction")
        return True
    

    def get_interaction(self, interaction):
        """获取交互类型"""
        if self.check_interaction(interaction):
            self.current_interaction= interaction
    
    def process_image(self, image_input: Union[str, Image.Image]) -> Image.Image:
        """
        处理图像，调整为合适的尺寸
        
        Args:
            image_input: 图像路径或 PIL Image
            
        Returns:
            处理后的 PIL Image
        """
        return self.load_image(image_input)
    
    def load_image(self, path, image_size=None):
        """
        加载图像并进行预处理
        
        Args:
            path: 图像路径或 PIL Image
            image_size: 目标尺寸 (width, height)，如果为 None 则使用智能调整
            
        Returns:
            处理后的 PIL Image
        """
        if isinstance(path, tuple):
            # 处理多图像输入（如果支持）
            return [self.load_image(p, image_size) for p in path]
        
        if isinstance(path, str):
            pil_img = Image.open(path)
        else:
            pil_img = path
        
        if pil_img.mode != 'RGB':
            pil_img = pil_img.convert('RGB')
        
        if image_size is not None:
            pil_img = pil_img.resize((image_size[1], image_size[0]), Image.BICUBIC)
        else:
            # 使用智能调整
            pil_img = smart_resize(pil_img, area=self.image_area, ds_factor=16)
        
        return pil_img
    
    def process_pil_image(self, pil_img):
        """
        处理 PIL 图像，转换为模型所需的格式
        
        Args:
            pil_img: PIL Image
            
        Returns:
            处理后的 torch.Tensor
        """
        if pil_img.mode == 'L':
            pil_img = pil_img.convert('RGB')
        image = np.asarray(pil_img, dtype=np.float32) / 255.
        image = image[:, :, :3]
        image = torch.from_numpy(image).permute(2, 0, 1).contiguous().float()
        image = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)(image)
        return image
    
    def build_input_ids(
        self,
        prompt: str,
        reference_image: Optional[Union[str, Image.Image]] = None,
    ) -> torch.LongTensor:
        """
        构建输入 token IDs
        
        Args:
            prompt: 文本提示
            reference_image: 参考图像（可选）
            
        Returns:
            input_ids: [1, seq_len]
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer is required for building input IDs")
        
        # 构建完整的 prompt
        if self.use_image and reference_image is not None:
            # 处理参考图像
            image = self.process_image(reference_image)
            # 创建简单的配置对象用于 build_image
            class ImageConfig:
                image_area = self.image_area
            img_cfg = ImageConfig()
            # 编码图像为字符串
            image_string = build_image(image, img_cfg, self.tokenizer, self.vq_model)
            # 替换模板中的 <|IMAGE|>
            full_prompt = self.template.format(question=prompt).replace("<|IMAGE|>", image_string)
        else:
            # 无图像情况
            full_prompt = self.template.format(question=prompt).replace("<|IMAGE|>", "")
        
        # Tokenize
        input_ids = self.tokenizer.encode(full_prompt, return_tensors="pt", add_special_tokens=False)
        
        return input_ids
    
    def build_unconditional_ids(self, reference_image=None) -> torch.LongTensor:
        """
        构建无条件输入的 token IDs
        
        Args:
            reference_image: 参考图像（可选）
            
        Returns:
            unconditional_ids: [1, seq_len]
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer is required for building unconditional IDs")
        
        if self.use_image and reference_image is not None:
            # 处理参考图像
            image = self.process_image(reference_image)
            # 创建简单的配置对象用于 build_image
            class ImageConfig:
                image_area = self.image_area
            img_cfg = ImageConfig()
            image_string = build_image(image, img_cfg, self.tokenizer, self.vq_model)
            full_prompt = self.unconditional_prompt.replace("<|IMAGE|>", image_string)
        else:
            full_prompt = self.unconditional_prompt.replace("<|IMAGE|>", "")
        
        unconditional_ids = self.tokenizer.encode(full_prompt, return_tensors="pt", add_special_tokens=False)
        
        return unconditional_ids
    
    def process_interaction(
        self,
        prompt: str,
        reference_image: Optional[Union[str, Image.Image]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        处理交互输入，生成模型所需的输入格式
        
        Args:
            prompt: 文本提示
            reference_image: 参考图像（可选）
            **kwargs: 其他参数
            
        Returns:
            Dict 包含处理后的输入数据：
                - input_ids: 输入 token IDs
                - unconditional_ids: 无条件 token IDs
                - processed_image: 处理后的图像（如果有）
                - prompt: 原始 prompt
        """
        self.get_interaction(prompt)
        result: dict[str, str | None] = {
            "prompt": self.current_interaction,
            "processed_image": None
        }
        
        # 处理图像（如果提供）
        if reference_image is not None:
            result["processed_image"] = self.process_image(reference_image)
        
        # 构建输入 IDs
        if self.tokenizer is not None:
            result["input_ids"] = self.build_input_ids(self.current_interaction, reference_image)
            result["unconditional_ids"] = self.build_unconditional_ids(reference_image)
        
        return result
