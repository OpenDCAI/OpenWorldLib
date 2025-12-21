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
from .utils.emu3p5.model_utils import build_emu3p5
from .utils.emu3p5.input_utils import build_image, smart_resize
from .utils.emu3p5.generation_utils import non_streaming_generate, build_logits_processor, multimodal_decode


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
    model_path = getattr(args, 'model_path', None) or pretrained_model_path
    tokenizer_path = getattr(args, 'tokenizer_path', None) or "sceneflow/synthesis/visual_generation/emu/emu3p5/tokenizer_emu3_ibq"
    vq_path = getattr(args, 'vq_path', None) or "BAAI/Emu3.5-VisionTokenizer"
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


class Emu3p5Synthesis(object):
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
        self.task_type = getattr(args, 'task_type', 't2i')
        self.use_image = getattr(args, 'use_image', True)
        self.image_area = getattr(args, 'image_area', 518400)
        self.classifier_free_guidance = getattr(args, 'classifier_free_guidance', 3.0)
        self.unconditional_type = getattr(args, 'unconditional_type', 'no_text')
        
        
        # 配置采样参数
        self._setup_sampling_params()
        
        # 配置特殊 token IDs
        self._setup_special_tokens()
        
        self.model.eval()
    

    def _setup_sampling_params(self):
        """设置采样参数"""
        self.sampling_params = dict(
            use_cache=True,
            text_top_k=getattr(self.args, 'text_top_k', 1024),
            text_top_p=getattr(self.args, 'text_top_p', 0.9),
            text_temperature=getattr(self.args, 'text_temperature', 1.0),
            image_top_k=getattr(self.args, 'image_top_k', 5120),
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
    
    # def _setup_special_tokens(self):
    #     """设置特殊 token IDs"""
    #     self.special_token_ids = {
    #         "BOS": self.tokenizer.convert_tokens_to_ids(self.tokenizer.bos_token) if hasattr(self.tokenizer, 'bos_token') else None,
    #         "EOS": self.tokenizer.convert_tokens_to_ids(self.tokenizer.eos_token) if hasattr(self.tokenizer, 'eos_token') else None,
    #         "PAD": self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token) if hasattr(self.tokenizer, 'pad_token') else None,
    #     }
    def _setup_special_tokens(self):
        """设置特殊 token IDs"""
        # 尝试从 tokenizer 属性获取，如果没有则使用默认值
        special_tokens_map = {
            # 标准 token
            "BOS": getattr(self.tokenizer, 'bos_token', "<|extra_203|>"),
            "EOS": getattr(self.tokenizer, 'eos_token', "<|extra_204|>"),
            "PAD": getattr(self.tokenizer, 'pad_token', "<|endoftext|>"),
            "EOL": getattr(self.tokenizer, 'eol_token', "<|extra_200|>"),
            "EOF": getattr(self.tokenizer, 'eof_token', "<|extra_201|>"),
            "TMS": getattr(self.tokenizer, 'tms_token', "<|extra_202|>"),
            "IMG": getattr(self.tokenizer, 'img_token', "<|image token|>"),
            "BOI": getattr(self.tokenizer, 'boi_token', "<|image start|>"),
            "EOI": getattr(self.tokenizer, 'eoi_token', "<|image end|>"),
            "BSS": getattr(self.tokenizer, 'bss_token', "<|extra_100|>"),
            "ESS": getattr(self.tokenizer, 'ess_token', "<|extra_101|>"),
            "BOG": getattr(self.tokenizer, 'bog_token', "<|extra_60|>"),
            "EOG": getattr(self.tokenizer, 'eog_token', "<|extra_61|>"),
            "BOC": getattr(self.tokenizer, 'boc_token', "<|extra_50|>"),
            "EOC": getattr(self.tokenizer, 'eoc_token', "<|extra_51|>"),
        }
        
        # 转换为 token IDs
        self.special_token_ids = {}
        for key, token in special_tokens_map.items():
            if token is not None:
                try:
                    token_id = self.tokenizer.convert_tokens_to_ids(token)
                    self.special_token_ids[key] = token_id
                except:
                    self.special_token_ids[key] = None
            else:
                self.special_token_ids[key] = None
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
    
    @torch.no_grad()
    def predict(
        self,
        processed_data: dict[str, str | None],
        seed: Optional[Union[int, List[int]]] = None,
        batch_size: int = 1,
        num_images_per_prompt: int = 1,
        max_new_tokens: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        **kwargs,
    ) -> Dict:
        """
        生成预测结果
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
        out_dict["prompts"] = [processed_data['prompt']] * batch_size if isinstance(processed_data['prompt'], str) else prompt
        
        # 设置随机种子（使用第一个种子）
        if seeds:
            random.seed(seeds[0])
            torch.manual_seed(seeds[0])
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seeds[0])
        # 构建输入
        input_ids = processed_data["input_ids"]
        unconditional_ids = processed_data["unconditional_ids"]
        
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
        start_time = time.time()
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

