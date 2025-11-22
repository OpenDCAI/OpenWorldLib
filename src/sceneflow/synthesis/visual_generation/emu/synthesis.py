from typing import Union, Optional, List, Tuple, Dict
from pathlib import Path
from loguru import logger
import os
import time
import random

import torch
import torch.distributed as dist
from transformers import GenerationConfig
from transformers.generation import LogitsProcessorList
from PIL import Image
import numpy as np
import torchvision.transforms as T

from emu.utils.model_utils import build_emu3p5
from emu.utils.input_utils import build_image, smart_resize
from emu.utils.generation_utils import non_streaming_generate, build_logits_processor, multimodal_decode


def load_models(args, device, logger_obj, pretrained_model_path):
    """
    加载 Emu3.5 模型、tokenizer 和 vision tokenizer
    
    Args:
        args: 配置参数，包含模型路径等配置
        device: 设备
        logger_obj: 日志记录器
        pretrained_model_path: 预训练模型根路径
        
    Returns:
        model, tokenizer, vq_model
    """
    model_path = getattr(args, 'model_path', None) or f"{pretrained_model_path}/Emu3.5"
    tokenizer_path = getattr(args, 'tokenizer_path', None) or f"{pretrained_model_path}/tokenizer_emu3_ibq"
    vq_path = getattr(args, 'vq_path', None) or f"{pretrained_model_path}/Emu3.5-VisionTokenizer"
    vq_type = getattr(args, 'vq_type', 'ibq')
    model_device = getattr(args, 'model_device', 'auto') or device
    vq_device = getattr(args, 'vq_device', None) or device
    
    if logger_obj:
        logger_obj.info(f"Loading Emu3.5 model from {model_path}")
        logger_obj.info(f"Loading tokenizer from {tokenizer_path}")
        logger_obj.info(f"Loading vision tokenizer from {vq_path}")
    
    model, tokenizer, vq_model = build_emu3p5(
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        vq_path=vq_path,
        vq_type=vq_type,
        model_device=model_device,
        vq_device=vq_device,
    )
    
    # 初始化 vision tokenizer
    model.init_vision(tokenizer, vq_model)
    
    if logger_obj:
        logger_obj.info("Models loaded successfully")
    
    return model, tokenizer, vq_model


class Emu3Synthesis(object):
    """
    Emu3.5 生成合成类，提供统一的接口用于图像和文本生成
    
    参考 HunyuanVideoSynthesis 的结构，适配 Emu3.5 的特点
    """
    
    def __init__(
        self,
        args,
        model,
        tokenizer,
        vq_model,
        use_cpu_offload=False,
        device=None,
        logger=None,
        parallel_args=None,
    ):
        """
        初始化 Emu3Synthesis
        
        Args:
            args: 配置参数
            model: Emu3ForCausalLM 模型
            tokenizer: 文本 tokenizer
            vq_model: Vision tokenizer (VQ model)
            use_cpu_offload: 是否使用 CPU offload
            device: 设备
            logger: 日志记录器
            parallel_args: 并行参数（兼容接口）
        """
        self.model = model
        self.tokenizer = tokenizer
        self.vq_model = vq_model
        self.args = args
        self.use_cpu_offload = use_cpu_offload
        self.device = (
            device
            if device is not None
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
        self.logger = logger
        self.parallel_args = parallel_args or {}
        
        # 配置参数
        self.task_type = getattr(args, 'task_type', 'story')
        self.use_image = getattr(args, 'use_image', True)
        self.image_area = getattr(args, 'image_area', 518400)
        self.classifier_free_guidance = getattr(args, 'classifier_free_guidance', 3.0)
        self.unconditional_type = getattr(args, 'unconditional_type', 'no_text')
        
        # 构建 prompt template
        self._build_prompt_template()
        
        # 配置采样参数
        self._setup_sampling_params()
        
        # 配置特殊 token IDs
        self._setup_special_tokens()
        
        self.model.eval()
    
    def _build_prompt_template(self):
        """构建 prompt template 和 unconditional prompt"""
        task_str = self.task_type.lower()
        if self.use_image:
            self.unconditional_prompt = "<|extra_203|>You are a helpful assistant. USER: <|IMAGE|> ASSISTANT: <|extra_100|>"
            self.template = f"<|extra_203|>You are a helpful assistant for {task_str} task. USER: {{question}}<|IMAGE|> ASSISTANT: <|extra_100|>"
        else:
            self.unconditional_prompt = "<|extra_203|>You are a helpful assistant. USER:  ASSISTANT: <|extra_100|>"
            self.template = f"<|extra_203|>You are a helpful assistant for {task_str} task. USER: {{question}} ASSISTANT: <|extra_100|>"
    
    def _setup_sampling_params(self):
        """设置采样参数"""
        self.sampling_params = dict(
            use_cache=True,
            text_top_k=getattr(self.args, 'text_top_k', 1024),
            text_top_p=getattr(self.args, 'text_top_p', 0.9),
            text_temperature=getattr(self.args, 'text_temperature', 1.0),
            image_top_k=getattr(self.args, 'image_top_k', 10240),
            image_top_p=getattr(self.args, 'image_top_p', 1.0),
            image_temperature=getattr(self.args, 'image_temperature', 1.0),
            top_k=getattr(self.args, 'top_k', 131072),
            top_p=getattr(self.args, 'top_p', 1.0),
            temperature=getattr(self.args, 'temperature', 1.0),
            num_beams_per_group=getattr(self.args, 'num_beams_per_group', 1),
            num_beam_groups=getattr(self.args, 'num_beam_groups', 1),
            diversity_penalty=getattr(self.args, 'diversity_penalty', 0.0),
            max_new_tokens=getattr(self.args, 'max_new_tokens', 32768),
            guidance_scale=getattr(self.args, 'guidance_scale', 1.0),
            use_differential_sampling=getattr(self.args, 'use_differential_sampling', True),
        )
        self.sampling_params["do_sample"] = self.sampling_params["num_beam_groups"] <= 1
        self.sampling_params["num_beams"] = self.sampling_params["num_beams_per_group"] * self.sampling_params["num_beam_groups"]
    
    def _setup_special_tokens(self):
        """设置特殊 token IDs"""
        self.special_token_ids = {
            "BOS": self.tokenizer.convert_tokens_to_ids(self.tokenizer.bos_token) if hasattr(self.tokenizer, 'bos_token') else None,
            "EOS": self.tokenizer.convert_tokens_to_ids(self.tokenizer.eos_token) if hasattr(self.tokenizer, 'eos_token') else None,
            "PAD": self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token) if hasattr(self.tokenizer, 'pad_token') else None,
        }
    
    @classmethod
    def from_pretrained(cls, pretrained_model_path, args, device=None, logger=None, **kwargs):
        """
        从预训练模型路径加载 Emu3Synthesis
        
        Args:
            pretrained_model_path (str or pathlib.Path): 预训练模型根路径
            args: 配置参数
            device: 设备，默认为 None（自动检测）
            logger: 日志记录器，默认为 None
            
        Returns:
            Emu3Synthesis 实例
        """
        logger_inst = logger
        if logger_inst:
            logger_inst.info(f"Got text-to-image model root path: {pretrained_model_path}")
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        torch.set_grad_enabled(False)
        
        # 加载模型
        model, tokenizer, vq_model = load_models(args, device, logger_inst, pretrained_model_path)
        
        return cls(
            args=args,
            model=model,
            tokenizer=tokenizer,
            vq_model=vq_model,
            use_cpu_offload=getattr(args, 'use_cpu_offload', False),
            device=device,
            logger=logger_inst,
        )
    
    def process(self, pil_img):
        """
        处理 PIL 图像，转换为模型所需的格式
        
        参考 HunyuanVideoSynthesis.process 方法
        
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
    
    def load_image(self, path, image_size=None):
        """
        加载图像，参考 HunyuanVideoSynthesis.load_image 方法
        
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
    
    def process_image(self, image_path: Union[str, Image.Image]) -> Image.Image:
        """
        处理图像，调整为合适的尺寸
        
        Args:
            image_path: 图像路径或 PIL Image
            
        Returns:
            处理后的 PIL Image
        """
        return self.load_image(image_path)
    
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
        # 构建完整的 prompt
        if self.use_image and reference_image is not None:
            # 处理参考图像（已经在 process_image 中调整了尺寸）
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
        input_ids = self.tokenizer.encode(full_prompt, return_tensors="pt")
        
        return input_ids
    
    def build_unconditional_ids(self) -> torch.LongTensor:
        """
        构建无条件输入的 token IDs
        
        Returns:
            unconditional_ids: [1, seq_len]
        """
        if self.use_image:
            # 对于有图像的情况，需要构建一个空的图像占位符
            # 这里使用一个小的占位图像字符串
            image_string = self.tokenizer.boi_token + "1*1" + self.tokenizer.img_token + self.tokenizer.eoi_token
            full_prompt = self.unconditional_prompt.replace("<|IMAGE|>", image_string)
        else:
            full_prompt = self.unconditional_prompt
        
        unconditional_ids = self.tokenizer.encode(full_prompt, return_tensors="pt")
        
        return unconditional_ids
    
    @torch.no_grad()
    def predict(
        self,
        prompt: str,
        reference_image: Optional[Union[str, Image.Image]] = None,
        seed: Optional[Union[int, List[int]]] = None,
        batch_size: int = 1,
        num_images_per_prompt: int = 1,
        max_new_tokens: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        **kwargs,
    ) -> Dict:
        """
        生成预测结果
        
        参考 HunyuanVideoSynthesis.predict 方法的接口风格
        
        Args:
            prompt: 文本提示
            reference_image: 参考图像路径或 PIL Image（可选）
            seed: 随机种子，可以是 int、list[int] 或 None
            batch_size: 批次大小
            num_images_per_prompt: 每个 prompt 生成的图像数量
            max_new_tokens: 最大生成 token 数
            guidance_scale: CFG 引导尺度（等同于 classifier_free_guidance）
            **kwargs: 其他参数
            
        Returns:
            Dict 包含生成的结果：
                - samples: 生成的多模态输出列表
                - prompts: 原始 prompt 列表
                - seeds: 使用的种子列表
        """
        out_dict = dict()
        
        # 处理种子，参考 HunyuanVideoSynthesis 的逻辑
        if isinstance(seed, torch.Tensor):
            seed = seed.tolist()
        if seed is None:
            seeds = [
                random.randint(0, 1_000_000)
                for _ in range(batch_size * num_images_per_prompt)
            ]
        elif isinstance(seed, int):
            seeds = [
                seed + i
                for _ in range(batch_size)
                for i in range(num_images_per_prompt)
            ]
        elif isinstance(seed, (list, tuple)):
            if len(seed) == batch_size:
                seeds = [
                    int(seed[i]) + j
                    for i in range(batch_size)
                    for j in range(num_images_per_prompt)
                ]
            elif len(seed) == batch_size * num_images_per_prompt:
                seeds = [int(s) for s in seed]
            else:
                raise ValueError(
                    f"Length of seed must be equal to number of prompt(batch_size) or "
                    f"batch_size * num_images_per_prompt ({batch_size} * {num_images_per_prompt}), got {seed}."
                )
        else:
            raise ValueError(
                f"Seed must be an integer, a list of integers, or None, got {seed}."
            )
        
        out_dict["seeds"] = seeds
        out_dict["prompts"] = [prompt] * batch_size if isinstance(prompt, str) else prompt
        
        # 设置随机种子（使用第一个种子）
        if seeds:
            random.seed(seeds[0])
            torch.manual_seed(seeds[0])
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seeds[0])
        
        # 构建输入
        input_ids = self.build_input_ids(prompt, reference_image)
        unconditional_ids = self.build_unconditional_ids()
        
        # 移动到设备
        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)
        unconditional_ids = unconditional_ids.to(device)
        
        # 设置生成参数
        generation_config_dict = self.sampling_params.copy()
        if max_new_tokens is not None:
            generation_config_dict['max_new_tokens'] = max_new_tokens
        
        # 支持 guidance_scale 参数（等同于 classifier_free_guidance）
        cfg_scale = guidance_scale if guidance_scale is not None else self.classifier_free_guidance
        
        # 创建配置对象（用于 generation_utils）
        cfg = type('Config', (), {
            'sampling_params': generation_config_dict,
            'special_token_ids': self.special_token_ids,
            'classifier_free_guidance': cfg_scale,
            'unconditional_type': self.unconditional_type,
            'image_area': self.image_area,
            'streaming': getattr(self.args, 'streaming', False),
        })()
        
        # 生成
        start_time = time.time()
        
        # 使用 generation_utils 中的函数
        gen_token_ids = non_streaming_generate(
            cfg=cfg,
            model=self.model,
            tokenizer=self.tokenizer,
            input_ids=input_ids,
            unconditional_ids=unconditional_ids,
            force_same_image_size=True,
        )
        
        # 解码生成的 tokens
        gen_tokens_str = self.tokenizer.decode(gen_token_ids, skip_special_tokens=False)
        
        # 多模态解码
        multimodal_outputs = multimodal_decode(
            outputs=gen_tokens_str,
            tokenizer=self.tokenizer,
            vision_tokenizer=self.vq_model,
        )
        
        gen_time = time.time() - start_time
        
        if self.logger:
            self.logger.info(f"Success, time: {gen_time:.2f}s")
        
        out_dict["samples"] = multimodal_outputs
        
        return out_dict

