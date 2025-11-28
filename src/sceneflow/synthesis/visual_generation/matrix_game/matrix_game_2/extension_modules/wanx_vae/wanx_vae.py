from .wrapper import VAEWrapper
import os
import torch
import torch.nn as nn
from pathlib import Path
from .wanx_vae_src import WanVAE, CLIPModel

class WanxVAEWrapper(VAEWrapper):
    def __init__(self, vae, clip):
        # super().__init__()
        self.vae = vae
        self.vae.requires_grad_(False)
        self.vae.eval()
        self.clip = clip
        if clip is not None:
            self.clip.requires_grad_(False)
            self.clip.eval()

    def encode(self, x, device, tiled=False, tile_size=(34, 34), tile_stride=(18, 16)):
        x = self.vae.encode(x, device=device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride) # already scaled
        return x # torch.stack(x, dim=0)

    def clip_img(self, x):
        x = self.clip(x)
        return x

    def decode(self, latents, device, tiled=False, tile_size=(34, 34), tile_stride=(18, 16)):
        videos = self.vae.decode(latents, device=device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        return videos # self.vae.decode(videos, dim=0) # already scaled
    
    def to(self, device, dtype):
        # 移动 vae 到指定设备
        self.vae = self.vae.to(device, dtype)
        
        # 如果 clip 存在，也移动到指定设备
        if self.clip is not None:
            self.clip = self.clip.to(device, dtype)
        
        return self

def get_wanx_vae_wrapper(model_path, weight_dtype):
    vae = WanVAE(pretrained_path = os.path.join(model_path, "Wan2.1_VAE.pth")).to(weight_dtype)
    clip = CLIPModel(checkpoint_path = os.path.join(model_path, "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"),
    tokenizer_path = os.path.join(model_path, 'xlm-roberta-large'))
    return WanxVAEWrapper(vae, clip)