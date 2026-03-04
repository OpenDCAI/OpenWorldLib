# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
from easydict import EasyDict

from .shared_config import wan_shared_cfg

#------------------------ Wan animate 14B ------------------------#
animate_14B = EasyDict(__name__='Config: Wan animate 14B')
animate_14B.update(wan_shared_cfg)

animate_14B.t5_checkpoint = 'models_t5_umt5-xxl-enc-bf16.pth'
animate_14B.t5_tokenizer = 'google/umt5-xxl'

animate_14B.clip_checkpoint = 'models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth'
animate_14B.clip_tokenizer = 'xlm-roberta-large'
animate_14B.lora_checkpoint = 'relighting_lora.ckpt'
# vae
animate_14B.vae_checkpoint = 'Wan2.1_VAE.pth'
animate_14B.vae_stride = (4, 8, 8)

# transformer
animate_14B.patch_size = (1, 2, 2)
animate_14B.dim = 5120
animate_14B.ffn_dim = 13824
animate_14B.freq_dim = 256
animate_14B.num_heads = 40
animate_14B.num_layers = 40
animate_14B.window_size = (-1, -1)
animate_14B.qk_norm = True
animate_14B.cross_attn_norm = True
animate_14B.eps = 1e-6
animate_14B.use_face_encoder = True
animate_14B.motion_encoder_dim = 512

# inference
animate_14B.sample_shift = 5.0
animate_14B.sample_steps = 20
animate_14B.sample_guide_scale = 1.0
animate_14B.frame_num = 77
animate_14B.sample_fps = 30
animate_14B.prompt = '视频中的人在做动作'
