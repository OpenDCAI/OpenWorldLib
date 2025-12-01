# python demo.py --duration=8 --video=<path to video> --prompt "your prompt" 
from typing import Union, Optional, List, Tuple, Dict
from pathlib import Path
import os
import time
import random
import logging

import torch
from loguru import logger

from .mmaudio.eval_utils import (ModelConfig, all_model_cfg, generate, make_video)
from .mmaudio.model.flow_matching import FlowMatching
from .mmaudio.model.networks import MMAudio, get_my_mmaudio
from .mmaudio.model.utils.features_utils import FeaturesUtils


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

log = logging.getLogger()

def load_models(args, device, logger_obj):
    """
    加载 MMAudio 模型
    
    Args:
        args: 配置参数，包含 variant, full_precision, num_steps 等
        device: 设备 (cuda/cpu/mps)
        logger_obj: 日志记录器
        
    Returns:
        net: MMAudio 主网络
        feature_utils: 特征工具
        fm: FlowMatching 实例
        seq_cfg: 序列配置
        model: ModelConfig 实例
    """
    if logger_obj:
        logger_obj.info(f"Loading MMAudio model variant: {args.variant}")
    
    model: ModelConfig = all_model_cfg[args.variant]
    model.download_if_needed()
    seq_cfg = model.seq_cfg

    dtype = torch.float32 if args.full_precision else torch.bfloat16
    
    if logger_obj:
        logger_obj.info(f"Loading network: {model.model_name} on {device} with dtype {dtype}")
    
    net: MMAudio = get_my_mmaudio(model.model_name).to(device, dtype).eval()
    net.load_weights(torch.load(model.model_path, map_location=device, weights_only=True))

    if logger_obj:
        logger_obj.info("Loading feature utils...")
    
    feature_utils = FeaturesUtils(
        tod_vae_ckpt=model.vae_path,
        synchformer_ckpt=model.synchformer_ckpt,
        enable_conditions=True,
        mode=model.mode,
        bigvgan_vocoder_ckpt=model.bigvgan_16k_path,
        need_vae_encoder=False
    ).to(device, dtype).eval()

    fm = FlowMatching(min_sigma=0, inference_mode='euler', num_steps=args.num_steps)

    if logger_obj:
        logger_obj.info(f"Successfully loaded MMAudio model from {model.model_path}")

    return net, feature_utils, fm, seq_cfg, model
    

class MMAudioSynthesis:
    """
    MMAudio 生成合成类，提供统一的接口用于音频生成
    """
    def __init__(self, args, net, feature_utils, fm, seq_cfg, model_config, device, logger_obj):
        """
        初始化 MMAudioSynthesis
        
        Args:
            args: 配置参数
            net: MMAudio 主网络
            feature_utils: 特征工具
            fm: FlowMatching 实例
            seq_cfg: 序列配置
            model_config: 模型配置
            device: 设备
            logger_obj: 日志记录器
        """
        self.args = args
        self.device = device
        self.logger = logger_obj
        self.net = net
        self.feature_utils = feature_utils
        self.fm = fm
        self.seq_cfg = seq_cfg
        self.model_config = model_config

        # 初始化随机数生成器
        self.rng = torch.Generator(device=device)
        
        if self.logger:
            self.logger.info("MMAudioSynthesis initialized successfully")
        
    @classmethod
    def from_pretrained(cls, args, device=None, logger_obj=None, **kwargs):
        """
        从预训练模型路径加载 MMAudioSynthesis
        
        Args:
            args: 配置参数，包含 variant 等
            device: 设备，默认为 None（自动检测）
            logger_obj: 日志记录器，默认为 None
            **kwargs: 额外参数
            
        Returns:
            MMAudioSynthesis 实例
        """
        logger_inst = logger_obj
        
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
                if logger_inst:
                    logger_inst.warning('CUDA/MPS are not available, running on CPU')
        
        torch.set_grad_enabled(False)
        
        # 加载模型组件
        net, feature_utils, fm, seq_cfg, model_config = load_models(args, device, logger_inst)
        
        return cls(
            args=args,
            net=net,
            feature_utils=feature_utils,
            fm=fm,
            seq_cfg=seq_cfg,
            model_config=model_config,
            device=device,
            logger_obj=logger_inst
        )

    @torch.no_grad()
    def predict(
        self, 
        processed_data: Dict[str, any], 
        seed: Optional[Union[int, List[int]]] = None, 
        cfg_strength: float = 4.5, 
        **kwargs
    ) -> Dict:
        """
        生成音频预测结果
        
        Args:
            processed_data: 从 operator 处理后的数据，包含：
                - clip_frames: CLIP 视频帧 [B, C, T, H, W] 或 None
                - sync_frames: Sync 视频帧 [B, C, T, H, W] 或 None
                - prompt: 文本提示列表
                - negative_prompt: 负面提示列表
                - duration: 音频时长（秒）
                - video_info: VideoInfo 对象（可选，用于后续视频合成）
            seed: 随机种子
            cfg_strength: Classifier-free guidance 强度
            num_steps: 生成步数，如果为 None 则使用 args 中的值
            **kwargs: 额外参数
            
        Returns:
            Dict 包含生成的结果：
                - audio: 生成的音频 tensor [seq_len]
                - sampling_rate: 采样率
                - duration: 实际音频时长
                - prompt: 使用的 prompt
                - video_info: VideoInfo 对象（如果有）
        """
        
        # 设置随机种子
        self.rng.manual_seed(seed)
        if self.logger:
            self.logger.info(f"Using seed: {seed}")
        
        duration = processed_data["duration"]
        self.seq_cfg.duration = duration
        self.net.update_seq_lengths(
            self.seq_cfg.latent_seq_len, 
            self.seq_cfg.clip_seq_len, 
            self.seq_cfg.sync_seq_len
        )
        
        if self.logger:
            self.logger.info(f"Generating audio for duration: {duration}s with cfg_strength: {cfg_strength}")
        
        clip_frames = processed_data["clip_frames"]
        sync_frames = processed_data["sync_frames"]
        prompt = processed_data["prompt"]
        negative_prompt = processed_data["negative_prompt"]
        
        audios = generate(
            clip_frames, 
            sync_frames, 
            prompt, 
            negative_text=negative_prompt, 
            feature_utils=self.feature_utils, 
            net=self.net, 
            fm=self.fm, 
            rng=self.rng, 
            cfg_strength=cfg_strength
        )
        
        audio = audios.float().cpu()[0]
        
        
        result = {
            "audio": audio,
            "sampling_rate": self.seq_cfg.sampling_rate,
            "duration": duration,
            "prompt": prompt[0] if isinstance(prompt, list) and len(prompt) > 0 else prompt,
        }
        
        if "video_info" in processed_data and processed_data["video_info"] is not None:
            result["video_info"] = processed_data["video_info"]
        
        return result
