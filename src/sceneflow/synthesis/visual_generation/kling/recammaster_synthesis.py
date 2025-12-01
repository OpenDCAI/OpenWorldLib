import os
import torch
import torch.nn as nn
import numpy as np
from torchvision.transforms import v2
from huggingface_hub import snapshot_download
from ...base_synthesis import BaseSynthesis

from ...visual_generation.wan.wan_2p1.modules.model import WanModel
from .recammaster.models.wan_video_text_encoder import WanTextEncoder
from ...visual_generation.wan.wan_2p1.modules.t5 import T5RelativeEmbedding, T5LayerNorm
from ...visual_generation.wan.wan_2p1.modules.model import WanRMSNorm, sinusoidal_embedding_1d
from ...visual_generation.wan.wan_2p1.modules.vae import RMS_norm, CausalConv3d, Upsample
from .recammaster.model_manager import ModelManager
from .recammaster.pipelines.wan_video_recammaster import WanVideoReCamMasterPipeline


class RecamMasterSynthesis(BaseSynthesis):
    def __init__(self,
                 pipeline,
                 height=480,
                 width=832,
                 device="cuda"):
        super(RecamMasterSynthesis, self).__init__()
        self.pipeline = pipeline
        self.device = device

        self.frame_process = v2.Compose([
            v2.CenterCrop(size=(height, width)),
            v2.Resize(size=(height, width), antialias=True),
            v2.ToTensor(),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    @classmethod
    def from_pretrained(cls,
                        pretrained_model_path="Wan-AI/Wan2.1-T2V-1.3B",
                        recammaster_ckpt_path="KlingTeam/ReCamMaster-Wan2.1",
                        device="cuda",
                        **kwargs):
        if os.path.isdir(pretrained_model_path):
            model_root = pretrained_model_path
        else:
            # download from HuggingFace repo_id
            print(f"Downloading weights from HuggingFace repo: {pretrained_model_path}")
            model_root = snapshot_download(pretrained_model_path)
            print(f"Model downloaded to: {model_root}")
        
        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        model_manager.load_models([
            os.path.join(model_root, "diffusion_pytorch_model.safetensors"),
            os.path.join(model_root, "models_t5_umt5-xxl-enc-bf16.pth"),
            os.path.join(model_root, "Wan2.1_VAE.pth"),
        ])
        pipe = WanVideoReCamMasterPipeline.from_model_manager(model_manager, device="cuda")

        # Initialize additional modules introduced in ReCamMaster
        dim=pipe.dit.blocks[0].self_attn.q.weight.shape[0]
        for block in pipe.dit.blocks:
            block.cam_encoder = nn.Linear(12, dim)
            block.projector = nn.Linear(dim, dim)
            block.cam_encoder.weight.data.zero_()
            block.cam_encoder.bias.data.zero_()
            block.projector.weight = nn.Parameter(torch.eye(dim))
            block.projector.bias = nn.Parameter(torch.zeros(dim))

        # Load ReCamMaster checkpoint
        state_dict = torch.load(recammaster_ckpt_path, map_location="cpu")
        pipe.dit.load_state_dict(state_dict, strict=True)
        pipe.to(device)
        pipe.to(dtype=torch.bfloat16)
        return cls(pipeline=pipe, device=device)

    @torch.no_grad()
    def predict(self,
                input_text,
                input_video,
                camera_trajotry,
                cfg_scale=5.0):
        video = self.pipeline(
            prompt=input_text,
            negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
            source_video=input_video,
            target_camera=camera_trajotry,
            cfg_scale=cfg_scale,
            num_inference_steps=50,
            seed=0, tiled=True
        )
        return video
