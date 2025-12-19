
from typing import Union, Optional, List, Tuple, Dict
from pathlib import Path
import os
import json
import re
import time
import random
import logging
from datetime import datetime

import torch
import torchaudio
import numpy as np
from loguru import logger
from lightning.pytorch import seed_everything
from prefigure.prefigure import get_all_args, push_wandb_config
from huggingface_hub import snapshot_download, hf_hub_download

from .ThinkSound.ThinkSound.models import create_model_from_config
from .ThinkSound.ThinkSound.models.utils import load_ckpt_state_dict, remove_weight_norm_from_model
from .ThinkSound.ThinkSound.inference.sampling import sample, sample_discrete_euler
from ...base_synthesis import BaseSynthesis


def load_models(args, device, logger_obj):
    """
    加载 ThinkSound 模型
    
    Args:
        args: 配置参数，包含 model_config, ckpt_dir, pretransform_ckpt_path, duration_sec 等
        device: 设备 (cuda/cpu)
        logger_obj: 日志记录器
        
    Returns:
        model: ThinkSound 模型
        model_config: 模型配置字典
    """

    if logger_obj:
        logger_obj.info(f"Loading ThinkSound model from {args.model_config}")

    #Get JSON config from args.model_config
    with open(args.model_config) as f:
        model_config = json.load(f)

    duration=(float)(args.duration_sec)
    model_config["sample_size"] = duration * model_config["sample_rate"]
    model_config["model"]["diffusion"]["config"]["sync_seq_len"] = 24*int(duration)
    model_config["model"]["diffusion"]["config"]["clip_seq_len"] = 8*int(duration)
    model_config["model"]["diffusion"]["config"]["latent_seq_len"] = round(44100/64/32*duration)

    model = create_model_from_config(model_config)
    
    if args.compile:
        model = torch.compile(model)
    
    # 加载主模型权重
    model.load_state_dict(torch.load(args.ckpt_dir))

    # 加载vae权重
    load_vae_state = load_ckpt_state_dict(args.pretransform_ckpt_path, prefix='autoencoder.')
    model.pretransform.load_state_dict(load_vae_state)

    return model, model_config
   

class ThinkSoundSynthesis(BaseSynthesis):
    def __init__(self, args, model_config, model, device, logger_obj):
        """
        初始化 ThinkSoundSynthesis
        
        Args:
            args: 配置参数
            model: ThinkSound 模型
            model_config: 模型配置字典
            device: 设备
            logger_obj: 日志记录器
        """
        self.args = args
        self.model = model.to(device).eval()
        self.model_config = model_config
        self.device = device
        self.diffusion_objective = model_config["model"]["diffusion"].get(
            "diffusion_objective", "rectified_flow"
        )
        self.logger = logger_obj


    @classmethod
    def from_pretrained(
        cls, 
        synthesis_model_path: str,
        args, 
        device=None, 
        logger_obj=None, 
        **kwargs
    ):
        """
        从预训练模型路径加载 ThinkSoundSynthesis
        
        Args:
            synthesis_model_path: 模型路径，可以是本地目录路径或 HuggingFace repo_id
            args: 配置参数，包含 model_config, ckpt_dir, pretransform_ckpt_path 等
            device: 设备
            logger_obj: 日志记录器
            **kwargs: 额外参数
            
        Returns:
            ThinkSoundSynthesis 实例
        """
        if os.path.isdir(synthesis_model_path):
            model_root = synthesis_model_path
            if logger_obj:
                logger_obj.info(f"Using local model directory: {model_root}")
            
            # 假设模型文件在目录根目录或 ckpts 子目录中
            if os.path.exists(os.path.join(model_root, "thinksound.ckpt")):
                args.ckpt_dir = os.path.join(model_root, "thinksound.ckpt")
            elif os.path.exists(os.path.join(model_root, "thinksound_light.ckpt")):
                args.ckpt_dir = os.path.join(model_root, "thinksound_light.ckpt")
            elif os.path.exists(os.path.join(model_root, "ckpts", "thinksound.ckpt")):
                args.ckpt_dir = os.path.join(model_root, "ckpts", "thinksound.ckpt")
            elif os.path.exists(os.path.join(model_root, "ckpts", "thinksound_light.ckpt")):
                args.ckpt_dir = os.path.join(model_root, "ckpts", "thinksound_light.ckpt")
            
            # 更新 VAE 路径
            if os.path.exists(os.path.join(model_root, "vae.ckpt")):
                args.pretransform_ckpt_path = os.path.join(model_root, "vae.ckpt")
            elif os.path.exists(os.path.join(model_root, "ckpts", "vae.ckpt")):
                args.pretransform_ckpt_path = os.path.join(model_root, "ckpts", "vae.ckpt")
            
            # 更新 Synchformer 路径
            if hasattr(args, 'synchformer_ckpt_path'):
                if os.path.exists(os.path.join(model_root, "synchformer_state_dict.pth")):
                    args.synchformer_ckpt_path = os.path.join(model_root, "synchformer_state_dict.pth")
                elif os.path.exists(os.path.join(model_root, "ckpts", "synchformer_state_dict.pth")):
                    args.synchformer_ckpt_path = os.path.join(model_root, "ckpts", "synchformer_state_dict.pth")

        else:
            if logger_obj:
                logger_obj.info(f"Downloading weights from HuggingFace repo: {synthesis_model_path}")
            else:
                print(f"Downloading weights from HuggingFace repo: {synthesis_model_path}")
            
            download_dir = os.path.join(os.getcwd(), "hugid")
            os.makedirs(download_dir, exist_ok=True)
            
            model_root = snapshot_download(synthesis_model_path, local_dir=download_dir)
            
            if logger_obj:
                logger_obj.info(f"Model downloaded to: {model_root}")
            else:
                print(f"Model downloaded to: {model_root}")
            
            # 更新 args 中的路径
            # 尝试查找模型文件（根据 HuggingFace 仓库的实际结构）
            if os.path.exists(os.path.join(model_root, "thinksound.ckpt")):
                args.ckpt_dir = os.path.join(model_root, "thinksound.ckpt")
            elif os.path.exists(os.path.join(model_root, "thinksound_light.ckpt")):
                args.ckpt_dir = os.path.join(model_root, "thinksound_light.ckpt")
            elif os.path.exists(os.path.join(model_root, "ckpts", "thinksound.ckpt")):
                args.ckpt_dir = os.path.join(model_root, "ckpts", "thinksound.ckpt")
            elif os.path.exists(os.path.join(model_root, "ckpts", "thinksound_light.ckpt")):
                args.ckpt_dir = os.path.join(model_root, "ckpts", "thinksound_light.ckpt")

            
            # 更新 VAE 路径
            if os.path.exists(os.path.join(model_root, "vae.ckpt")):
                args.pretransform_ckpt_path = os.path.join(model_root, "vae.ckpt")
            elif os.path.exists(os.path.join(model_root, "ckpts", "vae.ckpt")):
                args.pretransform_ckpt_path = os.path.join(model_root, "ckpts", "vae.ckpt")

            
            # 更新 Synchformer 路径
            if hasattr(args, 'synchformer_ckpt_path'):
                if os.path.exists(os.path.join(model_root, "synchformer_state_dict.pth")):
                    args.synchformer_ckpt_path = os.path.join(model_root, "synchformer_state_dict.pth")
                elif os.path.exists(os.path.join(model_root, "ckpts", "synchformer_state_dict.pth")):
                    args.synchformer_ckpt_path = os.path.join(model_root, "ckpts", "synchformer_state_dict.pth")


        if logger_obj:
            logger_obj.info(f"Loading ThinkSound synthesis model from config: {args.model_config}")
            logger_obj.info(f"Using checkpoint: {args.ckpt_dir}")
            logger_obj.info(f"Using VAE checkpoint: {args.pretransform_ckpt_path}")
            if hasattr(args, 'synchformer_ckpt_path'):
                logger_obj.info(f"Using Synchformer checkpoint: {args.synchformer_ckpt_path}")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        seed = args.seed
        if os.environ.get("SLURM_PROCID") is not None:
            seed += int(os.environ.get("SLURM_PROCID"))

        seed_everything(seed, workers=True)

        torch.set_grad_enabled(False)
        
        model, model_config = load_models(args, device, logger_obj)
        return cls(
            args=args,
            model=model,
            model_config=model_config,
            logger_obj=logger_obj,
            device=device
        )


    @torch.no_grad()
    def predict(
        self, 
        processed_data: Dict[str, any], 
        cfg_scale: float = 5.0,
        num_steps: int = 24,
        **kwargs
    ) -> Dict:
        """
        生成音频预测结果
        
        Args:
            processed_data: 从 operator 处理后的数据
            seed: 随机种子
            cfg_scale: CFG 强度
            num_steps: 采样步数
            
        Returns:
            Dict 包含生成的结果
        """
        
        # 提取 batch 数据
        batch = processed_data["batch"]
        reals, metadata_tuple = batch
        metadata = metadata_tuple[0] if isinstance(metadata_tuple, tuple) else metadata_tuple
        
        batch_size = reals.shape[0]
        length = reals.shape[2]
        
        if self.logger:
            self.logger.info(f"Predicting {batch_size} samples with length {length} for id: {metadata.get('id')}")
        
        # 准备 conditioning
        with torch.amp.autocast('cuda'):
            conditioning = self.model.conditioner([metadata], self.device)
        
        video_exist = metadata.get('video_exist', torch.tensor(True, device=self.device))
        if video_exist.dim() == 0:
            video_exist = video_exist.unsqueeze(0)
        conditioning["metaclip_features"][~video_exist] = self.model.model.model.empty_clip_feat
        conditioning["sync_features"][~video_exist] = self.model.model.model.empty_sync_feat
        
        cond_inputs = self.model.get_conditioning_inputs(conditioning)
        
        if batch_size > 1:
            noise_list = []
            for _ in range(batch_size):
                noise_1 = torch.randn([1, self.model.io_channels, length]).to(self.device)
                noise_list.append(noise_1)
            noise = torch.cat(noise_list, dim=0)
        else:
            noise = torch.randn([batch_size, self.model.io_channels, length]).to(self.device)
        
        if self.logger:
            self.logger.info(f"Sampling with {self.diffusion_objective}, steps={num_steps}, cfg_scale={cfg_scale}")
        
        with torch.amp.autocast('cuda'):
            if self.diffusion_objective == "v":
                fakes = sample(
                    self.model.model,
                    noise,
                    num_steps,
                    0,
                    **cond_inputs,
                    cfg_scale=cfg_scale,
                    batch_cfg=True
                )
            elif self.diffusion_objective == "rectified_flow":
                fakes = sample_discrete_euler(
                    self.model.model,
                    noise,
                    num_steps,
                    **cond_inputs,
                    cfg_scale=cfg_scale,
                    batch_cfg=True
                )
            else:
                raise ValueError(f"Unknown diffusion_objective: {self.diffusion_objective}")
        
        # VAE 解码
        if self.model.pretransform is not None:
            fakes = self.model.pretransform.decode(fakes)
        
        # 后处理：归一化到 int16
        audios = (
            fakes.to(torch.float32)
            .div(torch.max(torch.abs(fakes)))
            .clamp(-1, 1)
            .mul(32767)
            .to(torch.int16)
            .cpu()
        )
    
        
        # 构建返回结果
        result = {
            "audio": audios,
            "sampling_rate": 44100,
            "duration": processed_data["duration"],
            "id": processed_data.get("id", "unknown"),
        }
        
        return result