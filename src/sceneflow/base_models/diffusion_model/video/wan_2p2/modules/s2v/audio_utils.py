# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math
from typing import Tuple, Union

import torch
import torch.cuda.amp as amp
import torch.nn as nn
from diffusers.models.attention import AdaLayerNorm

from ..model import WanAttentionBlock, WanCrossAttention
from .auxi_blocks import MotionEncoder_tc


class CausalAudioEncoder(nn.Module):

    def __init__(self,
                 dim=5120,
                 num_layers=25,
                 out_dim=2048,
                 video_rate=8,
                 num_token=4,
                 need_global=False):
        super().__init__()
        self.encoder = MotionEncoder_tc(
            in_dim=dim,
            hidden_dim=out_dim,
            num_heads=num_token,
            need_global=need_global)
        weight = torch.ones((1, num_layers, 1, 1)) * 0.01

        self.weights = torch.nn.Parameter(weight)
        self.act = torch.nn.SiLU()

    def forward(self, features):
        with amp.autocast(dtype=torch.float32):
            # features B * num_layers * dim * video_length
            weights = self.act(self.weights)
            weights_sum = weights.sum(dim=1, keepdims=True)
            weighted_feat = ((features * weights) / weights_sum).sum(
                dim=1)  # b dim f
            weighted_feat = weighted_feat.permute(0, 2, 1)  # b f dim
            res = self.encoder(weighted_feat)  # b f n dim

        return res  # b f n dim


class AudioCrossAttention(WanCrossAttention):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class AudioInjector_WAN(nn.Module):

    def __init__(self,
                 all_modules,
                 all_modules_names,
                 dim=2048,
                 num_heads=32,
                 inject_layer=[0, 27],
                 root_net=None,
                 enable_adain=False,
                 adain_dim=2048,
                 need_adain_ont=False):
        super().__init__()
        num_injector_layers = len(inject_layer)
        self.injected_block_id = {}
        audio_injector_id = 0
        for mod_name, mod in zip(all_modules_names, all_modules):
            if isinstance(mod, WanAttentionBlock):
                for inject_id in inject_layer:
                    if f'transformer_blocks.{inject_id}' in mod_name:
                        self.injected_block_id[inject_id] = audio_injector_id
                        audio_injector_id += 1

        self.injector = nn.ModuleList([
            AudioCrossAttention(
                dim=dim,
                num_heads=num_heads,
                qk_norm=True,
            ) for _ in range(audio_injector_id)
        ])
        self.injector_pre_norm_feat = nn.ModuleList([
            nn.LayerNorm(
                dim,
                elementwise_affine=False,
                eps=1e-6,
            ) for _ in range(audio_injector_id)
        ])
        self.injector_pre_norm_vec = nn.ModuleList([
            nn.LayerNorm(
                dim,
                elementwise_affine=False,
                eps=1e-6,
            ) for _ in range(audio_injector_id)
        ])
        if enable_adain:
            self.injector_adain_layers = nn.ModuleList([
                AdaLayerNorm(
                    output_dim=dim * 2, embedding_dim=adain_dim, chunk_dim=1)
                for _ in range(audio_injector_id)
            ])
            if need_adain_ont:
                self.injector_adain_output_layers = nn.ModuleList(
                    [nn.Linear(dim, dim) for _ in range(audio_injector_id)])
