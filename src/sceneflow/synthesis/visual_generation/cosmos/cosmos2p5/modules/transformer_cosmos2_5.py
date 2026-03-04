from typing import Dict, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import Timesteps, apply_rotary_emb
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import RMSNorm
from einops import rearrange, repeat
from torchvision import transforms

from ..acceleration import gather_forward_split_backward, get_sequence_parallel_group, split_forward_gather_backward
from .attention import Attention as _AttentionOp
from .transformer_cosmos import CosmosAdaLayerNorm, CosmosPatchEmbed, CosmosTimestepEmbedding, CosmosTransformerBlock


class Cosmos25AttnProcessor2_0:
    """An attention processor for the Cosmos2.5 model, optimized for
    performance.

    It uses a custom attention operation backend (e.g., 'sdpa', 'flash_attention_2').
    """

    def __init__(self, backend='sdpa'):
        self.attn_op = _AttentionOp(backend=backend, qkv_format='bhsd')

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Performs the attention operation."""
        # 1. Project to Q, K, V
        self.attn_op.is_selfattn = encoder_hidden_states is None
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        # Reshape for multi-head attention
        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        # 2. Normalize Q and K
        query = attn.norm_q(query)
        key = attn.norm_k(key)

        # 3. Apply Rotary Positional Embeddings (RoPE)
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, use_real=True, use_real_unbind_dim=-2)
            key = apply_rotary_emb(key, image_rotary_emb, use_real=True, use_real_unbind_dim=-2)

        # 4. Compute attention scores and values
        hidden_states = self.attn_op(query, key, value, attn_mask=attention_mask)
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3).type_as(query)

        # 5. Output projection
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


class Cosmos25RotaryPosEmbed(nn.Module):
    """Rotary Positional Embedding (RoPE) for 3D data (time, height, width).

    This module generates positional embeddings that are applied to the query and key tensors in the attention mechanism.
    """

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

        # Allocate dimensions for each axis
        self.dim_h = hidden_size // 6 * 2
        self.dim_w = hidden_size // 6 * 2
        self.dim_t = hidden_size - self.dim_h - self.dim_w

        # NTK-aware scaling factors for RoPE
        self.h_ntk_factor = rope_scale[1] ** (self.dim_h / (self.dim_h - 2))
        self.w_ntk_factor = rope_scale[2] ** (self.dim_w / (self.dim_w - 2))
        self.t_ntk_factor = rope_scale[0] ** (self.dim_t / (self.dim_t - 2))

    def forward(self, hidden_states: torch.Tensor, fps: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generates the cosine and sine components of the rotary
        embeddings."""
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        sp_group = get_sequence_parallel_group()
        if sp_group is not None:
            world_size = dist.get_world_size(sp_group)
            num_frames *= world_size

        pe_size = [num_frames // self.patch_size[0], height // self.patch_size[1], width // self.patch_size[2]]
        device = hidden_states.device

        # Calculate theta values with NTK scaling
        h_theta = 10000.0 * self.h_ntk_factor
        w_theta = 10000.0 * self.w_ntk_factor
        t_theta = 10000.0 * self.t_ntk_factor

        # Generate frequency ranges
        seq = torch.arange(max(self.max_size), device=device, dtype=torch.float32)
        dim_h_range = torch.arange(0, self.dim_h, 2, device=device, dtype=torch.float32)[: (self.dim_h // 2)] / self.dim_h
        dim_w_range = torch.arange(0, self.dim_w, 2, device=device, dtype=torch.float32)[: (self.dim_w // 2)] / self.dim_w
        dim_t_range = torch.arange(0, self.dim_t, 2, device=device, dtype=torch.float32)[: (self.dim_t // 2)] / self.dim_t
        h_spatial_freqs = 1.0 / (h_theta**dim_h_range)
        w_spatial_freqs = 1.0 / (w_theta**dim_w_range)
        temporal_freqs = 1.0 / (t_theta**dim_t_range)

        # Create embeddings for each dimension
        emb_h = torch.outer(seq[: pe_size[1]], h_spatial_freqs)[None, :, None, :].repeat(pe_size[0], 1, pe_size[2], 1)
        emb_w = torch.outer(seq[: pe_size[2]], w_spatial_freqs)[None, None, :, :].repeat(pe_size[0], pe_size[1], 1, 1)

        # Apply FPS-based scaling for the temporal dimension
        if fps is None:  # Image case
            emb_t = torch.outer(seq[: pe_size[0]], temporal_freqs)
        else:  # Video case
            emb_t = torch.outer(seq[: pe_size[0]] / fps * self.base_fps, temporal_freqs)

        emb_t = emb_t[:, None, None, :].repeat(1, pe_size[1], pe_size[2], 1)

        # Concatenate and flatten frequencies
        freqs = torch.cat([emb_t, emb_h, emb_w] * 2, dim=-1).flatten(0, 2).float()
        if sp_group is not None:
            freqs = split_forward_gather_backward(freqs, dim=0, group=sp_group)

        cos = torch.cos(freqs)
        sin = torch.sin(freqs)
        return cos, sin


class Cosmos25TimeEmbed(nn.Module):
    """A module for creating time embeddings from a timestep."""

    def __init__(self, embedding_dim: int, condition_dim: int) -> None:
        super().__init__()
        self.time_proj = Timesteps(embedding_dim, flip_sin_to_cos=True, downscale_freq_shift=0.0)
        self.t_embedder = CosmosTimestepEmbedding(embedding_dim, condition_dim)

    def forward(self, hidden_states: torch.Tensor, timestep: torch.LongTensor) -> Tuple[torch.Tensor, torch.Tensor]:
        timesteps_proj = self.time_proj(timestep).type_as(hidden_states)
        temb = self.t_embedder(timesteps_proj)
        return timesteps_proj, temb


class Cosmos25ActionEmbed(nn.Module):
    """A simple MLP for embedding action information."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.activation = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Cosmos25Transformer3DModel(ModelMixin, ConfigMixin):
    """The main 3D Transformer model for the Cosmos2.5 architecture.

    This model processes latent representations of video data, conditioned on text, time, and optional action and control signals, to perform the
    denoising step in the diffusion process.
    """

    _skip_layerwise_casting_patterns = ['patch_embed', 'final_layer', 'norm']
    _no_split_modules = ['CosmosTransformerBlock']

    @register_to_config
    def __init__(
        self,
        in_channels: int = 16,
        out_channels: int = 16,
        num_attention_heads: int = 16,
        attention_head_dim: int = 128,
        num_layers: int = 28,
        mlp_ratio: float = 4.0,
        text_in_channels: int = 100352,
        text_embed_dim: int = 1024,
        adaln_lora_dim: int = 256,
        max_size: Tuple[int, int, int] = (128, 240, 240),
        patch_size: Tuple[int, int, int] = (1, 2, 2),
        rope_scale: Tuple[float, float, float] = (1.0, 3.0, 3.0),
        concat_padding_mask: bool = True,
        action_dim: int = 0,
        num_action_per_latent_frame: int = 4,
    ) -> None:
        super().__init__()
        hidden_size = num_attention_heads * attention_head_dim

        # Input patch embedding
        patch_embed_in_channels = in_channels + 1 if concat_padding_mask else in_channels
        self.patch_embed = CosmosPatchEmbed(patch_embed_in_channels, hidden_size, patch_size, bias=False)

        # Positional Embedding
        self.rope = Cosmos25RotaryPosEmbed(hidden_size=attention_head_dim, max_size=max_size, patch_size=patch_size, rope_scale=rope_scale)

        # Text Embedding Projection
        self.text_embed = nn.Sequential(
            nn.Linear(text_in_channels, text_embed_dim, bias=True),
            nn.GELU(),
        )

        # Action Embedding
        if action_dim > 0:
            self.action_embed = Cosmos25ActionEmbed(
                in_features=action_dim * num_action_per_latent_frame,
                hidden_features=hidden_size * 4,
                out_features=hidden_size,
                act_layer=lambda: nn.GELU(approximate='tanh'),
                drop=0,
            )
            self.action_embed_3d = Cosmos25ActionEmbed(
                in_features=action_dim * num_action_per_latent_frame,
                hidden_features=hidden_size * 4,
                out_features=hidden_size * 3,
                act_layer=lambda: nn.GELU(approximate='tanh'),
                drop=0,
            )

        # Time Embedding
        self.time_embed = Cosmos25TimeEmbed(hidden_size, hidden_size)
        self.time_norm = RMSNorm(hidden_size, eps=1e-6, elementwise_affine=True)

        # Transformer Blocks
        self.transformer_blocks = nn.ModuleList(
            [
                CosmosTransformerBlock(
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    cross_attention_dim=text_embed_dim,
                    mlp_ratio=mlp_ratio,
                    adaln_lora_dim=adaln_lora_dim,
                    qk_norm='rms_norm',
                    out_bias=False,
                )
                for _ in range(num_layers)
            ]
        )
        self.set_processor(Cosmos25AttnProcessor2_0())

        # Output normalization and projection
        self.norm_out = CosmosAdaLayerNorm(hidden_size, adaln_lora_dim)
        self.proj_out = nn.Linear(hidden_size, patch_size[0] * patch_size[1] * patch_size[2] * out_channels, bias=False)

    def set_processor(self, processor):
        """Sets the attention processor for all attention layers."""
        for module in self.modules():
            if isinstance(module, Attention):
                module.set_processor(processor)

    def set_attn_backend(self, backend):
        """Sets the attention backend (e.g., 'xformers') for optimization."""
        self.set_processor(Cosmos25AttnProcessor2_0(backend))

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        fps: Optional[int] = None,
        condition_mask: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
        control_hidden_states: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """The forward pass of the transformer model."""
        sp_group = get_sequence_parallel_group()
        # Handle sequence parallelism for distributed training
        if sp_group is not None:
            if timestep.shape[1] == hidden_states.shape[2]:
                timestep = split_forward_gather_backward(timestep, dim=1, group=sp_group)
            hidden_states = split_forward_gather_backward(hidden_states, dim=2, group=sp_group)
            condition_mask = split_forward_gather_backward(condition_mask, dim=2, group=sp_group)
        batch_size, num_channels, num_frames, height, width = hidden_states.shape

        # Concatenate condition mask (for i2v) if provided
        if condition_mask is not None:
            hidden_states = torch.cat([hidden_states, condition_mask], dim=1)

        # Concatenate padding mask if configured
        if self.config.concat_padding_mask:
            padding_mask = transforms.functional.resize(
                padding_mask, list(hidden_states.shape[-2:]), interpolation=transforms.InterpolationMode.NEAREST
            )
            padding_mask = padding_mask.unsqueeze(2).repeat(1, 1, num_frames, 1, 1)
            hidden_states = torch.cat([hidden_states, padding_mask], dim=1)

        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, S]

        # Generate rotary positional embeddings
        image_rotary_emb = self.rope(hidden_states, fps=fps)
        extra_pos_emb = None

        # Patchify the input latents
        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w
        hidden_states = self.patch_embed(hidden_states)
        hidden_states = hidden_states.flatten(1, 3)  # [B, T, H, W, C] -> [B, THW, C]

        # Project text embeddings
        encoder_hidden_states = self.text_embed(encoder_hidden_states)

        # Prepare action embeddings if provided
        if action is not None:
            action = rearrange(action, 'B (T A) C -> B T (A C)', A=self.config.num_action_per_latent_frame)
            embedded_action = self.action_embed(action)
            temb_action = self.action_embed_3d(action)
            # Prepend a zero embedding for the unconditional part
            zero_embedded_action = torch.zeros_like(embedded_action[:, :1, :])
            zero_temb_action = torch.zeros_like(temb_action[:, :1, :])
            embedded_action = torch.cat([zero_embedded_action, embedded_action], dim=1)
            temb_action = torch.cat([zero_temb_action, temb_action], dim=1)
            if sp_group is not None:
                embedded_action = split_forward_gather_backward(embedded_action, dim=1, group=sp_group)
                temb_action = split_forward_gather_backward(temb_action, dim=1, group=sp_group)

        # Prepare time embeddings
        timestep = timestep.flatten()
        embedded_timestep, temb = self.time_embed(hidden_states, timestep)
        embedded_timestep = rearrange(embedded_timestep, '(B T) C -> B T C', B=batch_size)
        temb = rearrange(temb, '(B T) C -> B T C', B=batch_size)
        # Add action embeddings to time embeddings
        if action is not None:
            embedded_timestep = embedded_timestep + embedded_action
            temb = temb + temb_action
        embedded_timestep = self.time_norm(embedded_timestep)
        # Repeat time embeddings for each patch
        temb, embedded_timestep = (
            repeat(
                x,
                'B T C -> B (T T2 H W) C',
                T=x.shape[1],
                T2=post_patch_num_frames if x.shape[1] == 1 else 1,
                H=post_patch_height,
                W=post_patch_width,
            )
            for x in (temb, embedded_timestep)
        )

        # Main transformer blocks loop
        for block_id, block in enumerate(self.transformer_blocks):
            hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                embedded_timestep=embedded_timestep,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                extra_pos_emb=extra_pos_emb,
                attention_mask=attention_mask,
            )
            # Add control signal from ControlNet if available
            if control_hidden_states is not None and str(block_id) in control_hidden_states:
                hidden_states = hidden_states + control_hidden_states[str(block_id)]

        # Final normalization and output projection
        hidden_states = self.norm_out(hidden_states, embedded_timestep, temb)
        hidden_states = self.proj_out(hidden_states)

        # Unpatchify to get the final latent representation
        hidden_states = rearrange(
            hidden_states,
            'B (T H W) (p1 p2 t C) -> B C (T t) (H p1) (W p2)',
            T=post_patch_num_frames,
            H=post_patch_height,
            W=post_patch_width,
            p1=p_h,
            p2=p_w,
            t=p_t,
        )
        if sp_group is not None:
            hidden_states = gather_forward_split_backward(hidden_states, dim=2, group=sp_group)
        return hidden_states
