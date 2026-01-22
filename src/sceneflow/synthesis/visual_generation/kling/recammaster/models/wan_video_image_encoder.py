import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from ......base_models.diffusion_model.video.wan_2p1.modules.clip import clip_xlm_roberta_vit_h_14


class WanImageEncoder(torch.nn.Module):

    def __init__(self):
        super().__init__()
        # init model
        self.model, self.transforms = clip_xlm_roberta_vit_h_14(
            pretrained=False,
            return_transforms=True,
            return_tokenizer=False,
            dtype=torch.float32,
            device="cpu")

    def encode_image(self, videos):
        # preprocess
        size = (self.model.image_size,) * 2
        videos = torch.cat([
            F.interpolate(
                u,
                size=size,
                mode='bicubic',
                align_corners=False) for u in videos
        ])
        videos = self.transforms.transforms[-1](videos.mul_(0.5).add_(0.5))

        # forward
        dtype = next(iter(self.model.visual.parameters())).dtype
        videos = videos.to(dtype)
        out = self.model.visual(videos, use_31_block=True)
        return out
        
    @staticmethod
    def state_dict_converter():
        return WanImageEncoderStateDictConverter()
    
    
class WanImageEncoderStateDictConverter:
    def __init__(self):
        pass

    def from_diffusers(self, state_dict):
        return state_dict
    
    def from_civitai(self, state_dict):
        state_dict_ = {}
        for name, param in state_dict.items():
            if name.startswith("textual."):
                continue
            name = "model." + name
            state_dict_[name] = param
        return state_dict_
