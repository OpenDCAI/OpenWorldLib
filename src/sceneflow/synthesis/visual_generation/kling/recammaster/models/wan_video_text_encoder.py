import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from ......base_models.diffusion_model.video.wan_2p1.modules.t5 import (
    T5RelativeEmbedding, T5LayerNorm, T5SelfAttention, init_weights)


class WanTextEncoder(torch.nn.Module):

    def __init__(self,
                 vocab=256384,
                 dim=4096,
                 dim_attn=4096,
                 dim_ffn=10240,
                 num_heads=64,
                 num_layers=24,
                 num_buckets=32,
                 shared_pos=False,
                 dropout=0.1):
        super(WanTextEncoder, self).__init__()
        self.dim = dim
        self.dim_attn = dim_attn
        self.dim_ffn = dim_ffn
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_buckets = num_buckets
        self.shared_pos = shared_pos

        # layers
        self.token_embedding = vocab if isinstance(vocab, nn.Embedding) \
            else nn.Embedding(vocab, dim)
        self.pos_embedding = T5RelativeEmbedding(
            num_buckets, num_heads, bidirectional=True) if shared_pos else None
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            T5SelfAttention(dim, dim_attn, dim_ffn, num_heads, num_buckets,
                            shared_pos, dropout) for _ in range(num_layers)
        ])
        self.norm = T5LayerNorm(dim)

        # initialize weights
        self.apply(init_weights)

    def forward(self, ids, mask=None):
        x = self.token_embedding(ids)
        x = self.dropout(x)
        e = self.pos_embedding(x.size(1),
                               x.size(1)) if self.shared_pos else None
        for block in self.blocks:
            x = block(x, mask, pos_bias=e)
        x = self.norm(x)
        x = self.dropout(x)
        return x
    
    @staticmethod
    def state_dict_converter():
        return WanTextEncoderStateDictConverter()
    
    
class WanTextEncoderStateDictConverter:
    def __init__(self):
        pass

    def from_diffusers(self, state_dict):
        return state_dict
    
    def from_civitai(self, state_dict):
        return state_dict