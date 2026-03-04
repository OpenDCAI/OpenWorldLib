# Copyright 2025 The NVIDIA Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin
from diffusers.utils import is_torchvision_available
from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import Timesteps, apply_rotary_emb
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import RMSNorm

from .attention import Attention as _AttentionOp 


if is_torchvision_available():
    from torchvision import transforms


class CosmosPatchEmbed(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, patch_size: Tuple[int, int, int], bias: bool = True
    ) -> None:
        super().__init__()
        self.patch_size = patch_size

        self.proj = nn.Linear(in_channels * patch_size[0] * patch_size[1] * patch_size[2], out_channels, bias=bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.patch_size
        hidden_states = hidden_states.reshape(
            batch_size, num_channels, num_frames // p_t, p_t, height // p_h, p_h, width // p_w, p_w
        )
        hidden_states = hidden_states.permute(0, 2, 4, 6, 1, 3, 5, 7).flatten(4, 7)
        hidden_states = self.proj(hidden_states)
        return hidden_states


class CosmosTimestepEmbedding(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(in_features, out_features, bias=False)
        self.activation = nn.SiLU()
        self.linear_2 = nn.Linear(out_features, 3 * out_features, bias=False)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        emb = self.linear_1(timesteps)
        emb = self.activation(emb)
        emb = self.linear_2(emb)
        return emb


class CosmosEmbedding(nn.Module):
    def __init__(self, embedding_dim: int, condition_dim: int) -> None:
        super().__init__()

        self.time_proj = Timesteps(embedding_dim, flip_sin_to_cos=True, downscale_freq_shift=0.0)
        self.t_embedder = CosmosTimestepEmbedding(embedding_dim, condition_dim)
        self.norm = RMSNorm(embedding_dim, eps=1e-6, elementwise_affine=True)

    def forward(self, hidden_states: torch.Tensor, timestep: torch.LongTensor) -> torch.Tensor:
        timesteps_proj = self.time_proj(timestep).type_as(hidden_states)
        temb = self.t_embedder(timesteps_proj)
        embedded_timestep = self.norm(timesteps_proj)
        return temb, embedded_timestep


class CosmosAdaLayerNorm(nn.Module):
    def __init__(self, in_features: int, hidden_features: int) -> None:
        super().__init__()
        self.embedding_dim = in_features

        self.activation = nn.SiLU()
        self.norm = nn.LayerNorm(in_features, elementwise_affine=False, eps=1e-6)
        self.linear_1 = nn.Linear(in_features, hidden_features, bias=False)
        self.linear_2 = nn.Linear(hidden_features, 2 * in_features, bias=False)

    def forward(
        self, hidden_states: torch.Tensor, embedded_timestep: torch.Tensor, temb: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        embedded_timestep = self.activation(embedded_timestep)
        embedded_timestep = self.linear_1(embedded_timestep)
        embedded_timestep = self.linear_2(embedded_timestep)

        if temb is not None:
            embedded_timestep = embedded_timestep + temb[..., : 2 * self.embedding_dim]

        shift, scale = embedded_timestep.chunk(2, dim=-1)
        hidden_states = self.norm(hidden_states)

        if embedded_timestep.ndim == 2:
            shift, scale = (x.unsqueeze(1) for x in (shift, scale))

        hidden_states = hidden_states * (1 + scale) + shift
        return hidden_states


class CosmosAdaLayerNormZero(nn.Module):
    def __init__(self, in_features: int, hidden_features: Optional[int] = None) -> None:
        super().__init__()

        self.norm = nn.LayerNorm(in_features, elementwise_affine=False, eps=1e-6)
        self.activation = nn.SiLU()

        if hidden_features is None:
            self.linear_1 = nn.Identity()
        else:
            self.linear_1 = nn.Linear(in_features, hidden_features, bias=False)

        self.linear_2 = nn.Linear(hidden_features, 3 * in_features, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        embedded_timestep: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        embedded_timestep = self.activation(embedded_timestep)
        embedded_timestep = self.linear_1(embedded_timestep)
        embedded_timestep = self.linear_2(embedded_timestep)

        if temb is not None:
            embedded_timestep = embedded_timestep + temb

        shift, scale, gate = embedded_timestep.chunk(3, dim=-1)
        hidden_states = self.norm(hidden_states)

        if embedded_timestep.ndim == 2:
            shift, scale, gate = (x.unsqueeze(1) for x in (shift, scale, gate))

        hidden_states = hidden_states * (1 + scale) + shift
        return hidden_states, gate


class CosmosAttnProcessor2_0:
    def __init__(self):
        self.attn_op = _AttentionOp(backend='sdpa', qkv_format='bhsd')

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # 1. QKV projections
        is_self_attn = encoder_hidden_states is None
        self.attn_op.is_selfattn = is_self_attn
        
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        # Transpose [B, H, S, D] to match bhsd format
        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        # 2. QK normalization
        query = attn.norm_q(query)
        key = attn.norm_k(key)

        # 3. Apply RoPE
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, use_real=True, use_real_unbind_dim=-2)
            key = apply_rotary_emb(key, image_rotary_emb, use_real=True, use_real_unbind_dim=-2)

        # 4. Prepare for GQA
        if torch.onnx.is_in_onnx_export():
            query_idx = torch.tensor(query.size(3), device=query.device)
            key_idx = torch.tensor(key.size(3), device=key.device)
            value_idx = torch.tensor(value.size(3), device=value.device)
        else:
            query_idx = query.size(3)
            key_idx = key.size(3)
            value_idx = value.size(3)
        
        # GQA copy attention heads
        key = key.repeat_interleave(query_idx // key_idx, dim=3)
        value = value.repeat_interleave(query_idx // value_idx, dim=3)

        hidden_states = self.attn_op(query, key, value, attn_mask=attention_mask)
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3).type_as(query)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


class CosmosAttnProcessor2_5:
    def __init__(self):
        self.attn_op = _AttentionOp(backend='sdpa', qkv_format='bhsd')

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]],
        attention_mask: Optional[Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]],
        image_rotary_emb=None,
    ) -> torch.Tensor:
        if not isinstance(encoder_hidden_states, tuple):
            raise ValueError("Expected encoder_hidden_states as (text_context, img_context) tuple.")

        text_context, img_context = encoder_hidden_states if encoder_hidden_states else (None, None)
        text_mask, img_mask = attention_mask if attention_mask else (None, None)

        is_text_self_attn = text_context is None or text_context is hidden_states
        if text_context is None:
            text_context = hidden_states

        self.attn_op.is_selfattn = is_text_self_attn

        query = attn.to_q(hidden_states)
        key = attn.to_k(text_context)
        value = attn.to_v(text_context)

        # Reshape [B, H, S, D]
        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, use_real=True, use_real_unbind_dim=-2)
            key = apply_rotary_emb(key, image_rotary_emb, use_real=True, use_real_unbind_dim=-2)

        # GQA repeat
        if torch.onnx.is_in_onnx_export():
            query_idx = torch.tensor(query.size(3), device=query.device)
            key_idx = torch.tensor(key.size(3), device=key.device)
            value_idx = torch.tensor(value.size(3), device=value.device)
        else:
            query_idx = query.size(3)
            key_idx = key.size(3)
            value_idx = value.size(3)
        key = key.repeat_interleave(query_idx // key_idx, dim=3)
        value = value.repeat_interleave(query_idx // value_idx, dim=3)

        # execute Attention the first time (Text/Self)
        attn_out = self.attn_op(query, key, value, attn_mask=text_mask)
        attn_out = attn_out.transpose(1, 2).flatten(2, 3).type_as(query)

        if img_context is not None:
            # Cross Attention
            self.attn_op.is_selfattn = False
            
            q_img = attn.q_img(hidden_states)
            k_img = attn.k_img(img_context)
            v_img = attn.v_img(img_context)

            batch_size = hidden_states.shape[0]
            dim_head = attn.out_dim // attn.heads

            # Reshape [B, H, S, D]
            q_img = q_img.view(batch_size, -1, attn.heads, dim_head).transpose(1, 2)
            k_img = k_img.view(batch_size, -1, attn.heads, dim_head).transpose(1, 2)
            v_img = v_img.view(batch_size, -1, attn.heads, dim_head).transpose(1, 2)

            q_img = attn.q_img_norm(q_img)
            k_img = attn.k_img_norm(k_img)

            # GQA repeat for image context
            q_img_idx = q_img.size(3)
            k_img_idx = k_img.size(3)
            v_img_idx = v_img.size(3)
            k_img = k_img.repeat_interleave(q_img_idx // k_img_idx, dim=3)
            v_img = v_img.repeat_interleave(q_img_idx // v_img_idx, dim=3)

            # execute Attention the second time (Image Cross)
            img_out = self.attn_op(q_img, k_img, v_img, attn_mask=img_mask)
            img_out = img_out.transpose(1, 2).flatten(2, 3).type_as(q_img)
            
            hidden_states = attn_out + img_out
        else:
            hidden_states = attn_out

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class CosmosAttention(Attention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # add parameters for image q/k/v
        inner_dim = self.heads * self.to_q.out_features // self.heads
        self.q_img = nn.Linear(self.query_dim, inner_dim, bias=False)
        self.k_img = nn.Linear(self.query_dim, inner_dim, bias=False)
        self.v_img = nn.Linear(self.query_dim, inner_dim, bias=False)
        self.q_img_norm = RMSNorm(self.to_q.out_features // self.heads, eps=1e-6, elementwise_affine=True)
        self.k_img_norm = RMSNorm(self.to_k.out_features // self.heads, eps=1e-6, elementwise_affine=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]],
        attention_mask: Optional[torch.Tensor] = None,
        **cross_attention_kwargs,
    ) -> torch.Tensor:
        return super().forward(
            hidden_states=hidden_states,
            # NOTE: type-hint in base class can be ignored
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )


class CosmosTransformerBlock(nn.Module):
    def __init__(
        self,
        num_attention_heads: int,
        attention_head_dim: int,
        cross_attention_dim: int,
        mlp_ratio: float = 4.0,
        adaln_lora_dim: int = 256,
        qk_norm: str = "rms_norm",
        out_bias: bool = False,
        img_context: bool = False,
        before_proj: bool = False,
        after_proj: bool = False,
    ) -> None:
        super().__init__()

        hidden_size = num_attention_heads * attention_head_dim

        self.norm1 = CosmosAdaLayerNormZero(in_features=hidden_size, hidden_features=adaln_lora_dim)
        self.img_context = img_context
        self.attn1 = Attention(
            query_dim=hidden_size,
            cross_attention_dim=None,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            qk_norm=qk_norm,
            elementwise_affine=True,
            out_bias=out_bias,
            processor=CosmosAttnProcessor2_0(),
        )

        self.norm2 = CosmosAdaLayerNormZero(in_features=hidden_size, hidden_features=adaln_lora_dim)
        if img_context:
            self.attn2 = CosmosAttention(
                query_dim=hidden_size,
                cross_attention_dim=cross_attention_dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                qk_norm=qk_norm,
                elementwise_affine=True,
                out_bias=out_bias,
                processor=CosmosAttnProcessor2_5(),
            )
        else:
            self.attn2 = Attention(
                query_dim=hidden_size,
                cross_attention_dim=cross_attention_dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                qk_norm=qk_norm,
                elementwise_affine=True,
                out_bias=out_bias,
                processor=CosmosAttnProcessor2_0(),
            )

        self.norm3 = CosmosAdaLayerNormZero(in_features=hidden_size, hidden_features=adaln_lora_dim)
        self.ff = FeedForward(hidden_size, mult=mlp_ratio, activation_fn="gelu", bias=out_bias)

        # NOTE: zero conv for CosmosControlNet
        self.before_proj = None
        self.after_proj = None
        if before_proj:
            self.before_proj = nn.Linear(hidden_size, hidden_size)
        if after_proj:
            self.after_proj = nn.Linear(hidden_size, hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Union[
            Optional[torch.Tensor], Optional[Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]]
        ],
        embedded_timestep: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        extra_pos_emb: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        controlnet_residual: Optional[torch.Tensor] = None,
        latents: Optional[torch.Tensor] = None,
        block_idx: Optional[int] = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if self.before_proj is not None:
            hidden_states = self.before_proj(hidden_states) + latents

        if extra_pos_emb is not None:
            hidden_states = hidden_states + extra_pos_emb

        # 1. Self Attention
        norm_hidden_states, gate = self.norm1(hidden_states, embedded_timestep, temb)
        attn_output = self.attn1(norm_hidden_states, image_rotary_emb=image_rotary_emb)
        hidden_states = hidden_states + gate * attn_output

        # 2. Cross Attention
        norm_hidden_states, gate = self.norm2(hidden_states, embedded_timestep, temb)
        attn_output = self.attn2(
            norm_hidden_states, encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask
        )
        hidden_states = hidden_states + gate * attn_output

        # 3. Feed Forward
        norm_hidden_states, gate = self.norm3(hidden_states, embedded_timestep, temb)
        ff_output = self.ff(norm_hidden_states)
        hidden_states = hidden_states + gate * ff_output

        if controlnet_residual is not None:
            assert self.after_proj is None
            # NOTE: this is assumed to be scaled by the controlnet
            hidden_states += controlnet_residual

        if self.after_proj is not None:
            assert controlnet_residual is None
            hs_proj = self.after_proj(hidden_states)
            return hidden_states, hs_proj

        return hidden_states


class CosmosRotaryPosEmbed(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        max_size: Tuple[int, int, int] = (128, 240, 240),
        patch_size: Tuple[int, int, int] = (1, 2, 2),
        base_fps: int = 24,
        rope_scale: Tuple[float, float, float] = (2.0, 1.0, 1.0),
    ) -> None:
        super().__init__()

        self.max_size = [size // patch for size, patch in zip(max_size, patch_size)]
        self.patch_size = patch_size
        self.base_fps = base_fps

        self.dim_h = hidden_size // 6 * 2
        self.dim_w = hidden_size // 6 * 2
        self.dim_t = hidden_size - self.dim_h - self.dim_w

        self.h_ntk_factor = rope_scale[1] ** (self.dim_h / (self.dim_h - 2))
        self.w_ntk_factor = rope_scale[2] ** (self.dim_w / (self.dim_w - 2))
        self.t_ntk_factor = rope_scale[0] ** (self.dim_t / (self.dim_t - 2))

    def forward(self, hidden_states: torch.Tensor, fps: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        pe_size = [num_frames // self.patch_size[0], height // self.patch_size[1], width // self.patch_size[2]]
        device = hidden_states.device

        h_theta = 10000.0 * self.h_ntk_factor
        w_theta = 10000.0 * self.w_ntk_factor
        t_theta = 10000.0 * self.t_ntk_factor

        seq = torch.arange(max(self.max_size), device=device, dtype=torch.float32)
        dim_h_range = (
            torch.arange(0, self.dim_h, 2, device=device, dtype=torch.float32)[: (self.dim_h // 2)] / self.dim_h
        )
        dim_w_range = (
            torch.arange(0, self.dim_w, 2, device=device, dtype=torch.float32)[: (self.dim_w // 2)] / self.dim_w
        )
        dim_t_range = (
            torch.arange(0, self.dim_t, 2, device=device, dtype=torch.float32)[: (self.dim_t // 2)] / self.dim_t
        )
        h_spatial_freqs = 1.0 / (h_theta**dim_h_range)
        w_spatial_freqs = 1.0 / (w_theta**dim_w_range)
        temporal_freqs = 1.0 / (t_theta**dim_t_range)

        emb_h = torch.outer(seq[: pe_size[1]], h_spatial_freqs)[None, :, None, :].repeat(pe_size[0], 1, pe_size[2], 1)
        emb_w = torch.outer(seq[: pe_size[2]], w_spatial_freqs)[None, None, :, :].repeat(pe_size[0], pe_size[1], 1, 1)

        # Apply sequence scaling in temporal dimension
        if fps is None:
            # Images
            emb_t = torch.outer(seq[: pe_size[0]], temporal_freqs)
        else:
            # Videos
            emb_t = torch.outer(seq[: pe_size[0]] / fps * self.base_fps, temporal_freqs)

        emb_t = emb_t[:, None, None, :].repeat(1, pe_size[1], pe_size[2], 1)
        freqs = torch.cat([emb_t, emb_h, emb_w] * 2, dim=-1).flatten(0, 2).float()
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)
        return cos, sin


class CosmosLearnablePositionalEmbed(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        max_size: Tuple[int, int, int],
        patch_size: Tuple[int, int, int],
        eps: float = 1e-6,
    ) -> None:
        super().__init__()

        self.max_size = [size // patch for size, patch in zip(max_size, patch_size)]
        self.patch_size = patch_size
        self.eps = eps

        self.pos_emb_t = nn.Parameter(torch.zeros(self.max_size[0], hidden_size))
        self.pos_emb_h = nn.Parameter(torch.zeros(self.max_size[1], hidden_size))
        self.pos_emb_w = nn.Parameter(torch.zeros(self.max_size[2], hidden_size))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        pe_size = [num_frames // self.patch_size[0], height // self.patch_size[1], width // self.patch_size[2]]

        emb_t = self.pos_emb_t[: pe_size[0]][None, :, None, None, :].repeat(batch_size, 1, pe_size[1], pe_size[2], 1)
        emb_h = self.pos_emb_h[: pe_size[1]][None, None, :, None, :].repeat(batch_size, pe_size[0], 1, pe_size[2], 1)
        emb_w = self.pos_emb_w[: pe_size[2]][None, None, None, :, :].repeat(batch_size, pe_size[0], pe_size[1], 1, 1)
        emb = emb_t + emb_h + emb_w
        emb = emb.flatten(1, 3)

        norm = torch.linalg.vector_norm(emb, dim=-1, keepdim=True, dtype=torch.float32)
        norm = torch.add(self.eps, norm, alpha=np.sqrt(norm.numel() / emb.numel()))
        return (emb / norm).type_as(hidden_states)
