# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
from easydict import EasyDict

from .shared_config import wan_shared_cfg

#------------------------ Wan S2V 14B ------------------------#

s2v_14B = EasyDict(__name__='Config: Wan S2V 14B')
s2v_14B.update(wan_shared_cfg)

# t5
s2v_14B.t5_checkpoint = 'models_t5_umt5-xxl-enc-bf16.pth'
s2v_14B.t5_tokenizer = 'google/umt5-xxl'

# vae
s2v_14B.vae_checkpoint = 'Wan2.1_VAE.pth'
s2v_14B.vae_stride = (4, 8, 8)

# wav2vec
s2v_14B.wav2vec = "wav2vec2-large-xlsr-53-english"

s2v_14B.num_heads = 40
# transformer
s2v_14B.transformer = EasyDict(
    __name__="Config: Transformer config for WanModel_S2V")
s2v_14B.transformer.patch_size = (1, 2, 2)
s2v_14B.transformer.dim = 5120
s2v_14B.transformer.ffn_dim = 13824
s2v_14B.transformer.freq_dim = 256
s2v_14B.transformer.num_heads = 40
s2v_14B.transformer.num_layers = 40
s2v_14B.transformer.window_size = (-1, -1)
s2v_14B.transformer.qk_norm = True
s2v_14B.transformer.cross_attn_norm = True
s2v_14B.transformer.eps = 1e-6
s2v_14B.transformer.enable_adain = True
s2v_14B.transformer.adain_mode = "attn_norm"
s2v_14B.transformer.audio_inject_layers = [
    0, 4, 8, 12, 16, 20, 24, 27, 30, 33, 36, 39
]
s2v_14B.transformer.zero_init = True
s2v_14B.transformer.zero_timestep = True
s2v_14B.transformer.enable_motioner = False
s2v_14B.transformer.add_last_motion = True
s2v_14B.transformer.trainable_token = False
s2v_14B.transformer.enable_tsm = False
s2v_14B.transformer.enable_framepack = True
s2v_14B.transformer.framepack_drop_mode = 'padd'
s2v_14B.transformer.audio_dim = 1024

s2v_14B.transformer.motion_frames = 73
s2v_14B.transformer.cond_dim = 16

# inference
s2v_14B.sample_neg_prompt = "画面模糊，最差质量，画面模糊，细节模糊不清，情绪激动剧烈，手快速抖动，字幕，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
s2v_14B.drop_first_motion = True
s2v_14B.sample_shift = 3
s2v_14B.sample_steps = 40
s2v_14B.sample_guide_scale = 4.5
