from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.attention_processor import Attention
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import RMSNorm
from diffusers.models.transformers.transformer_cosmos import CosmosPatchEmbed, CosmosTransformerBlock
from einops import rearrange, repeat
from torchvision import transforms

from ..acceleration import get_sequence_parallel_group, split_forward_gather_backward
from .transformer_cosmos2_5 import Cosmos25AttnProcessor2_0, Cosmos25RotaryPosEmbed, Cosmos25TimeEmbed


def zero_module(module):
    """Zero out the parameters of a module and return it."""
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


class Cosmos25ControlNet3DModel(ModelMixin, ConfigMixin):
    r"""A 3D ControlNet model for the Cosmos2.5 architecture.

    This model takes a conditioning signal (e.g., depth map, canny edges) and injects it into the
    intermediate layers of a base transformer model (like `Cosmos25Transformer3DModel`) to guide the
    video generation process.

    Args:
        in_channels (`int`, defaults to `16`):
            The number of channels in the input to the base model.
        control_in_channels (`int`, defaults to `130`):
            The number of channels in the control condition input.
        num_attention_heads (`int`, defaults to `16`):
            The number of heads for multi-head attention.
        attention_head_dim (`int`, defaults to `128`):
            The dimension of each attention head.
        block_ids (`list`, defaults to `[0, 7, 14, 21]`):
            A list of block indices where the control signal should be injected.
        mlp_ratio (`float`, defaults to `4.0`):
            The ratio for the MLP's hidden dimension.
        text_in_channels (`int`, defaults to `100352`):
            Input dimension of raw text embeddings.
        text_embed_dim (`int`, defaults to `1024`):
            The dimension of the projected text embeddings.
        adaln_lora_dim (`int`, defaults to `256`):
            The dimension for the AdaLN LoRA layers.
        max_size (`Tuple[int, int, int]`, defaults to `(128, 240, 240)`):
            Maximum (temporal, height, width) dimensions for positional embeddings.
        patch_size (`Tuple[int, int, int]`, defaults to `(1, 2, 2)`):
            The (temporal, height, width) patch size.
        rope_scale (`Tuple[float, float, float]`, defaults to `(1.0, 3.0, 3.0)`):
            Scaling factors for RoPE.
        concat_padding_mask (`bool`, defaults to `True`):
            Whether to concatenate a padding mask to the input.
    """

    _skip_layerwise_casting_patterns = ['patch_embed', 'norm']
    _no_split_modules = ['CosmosTransformerBlock']

    @register_to_config
    def __init__(
        self,
        in_channels: int = 16,
        control_in_channels: int = 130,
        num_attention_heads: int = 16,
        attention_head_dim: int = 128,
        block_ids: list = [0, 7, 14, 21],
        mlp_ratio: float = 4.0,
        text_in_channels: int = 100352,
        text_embed_dim: int = 1024,
        adaln_lora_dim: int = 256,
        max_size: Tuple[int, int, int] = (128, 240, 240),
        patch_size: Tuple[int, int, int] = (1, 2, 2),
        rope_scale: Tuple[float, float, float] = (1.0, 3.0, 3.0),
        concat_padding_mask: bool = True,
    ) -> None:
        super().__init__()
        hidden_size = num_attention_heads * attention_head_dim

        # Patch Embedding for both main input and control input
        patch_embed_in_channels = in_channels + 1 if concat_padding_mask else in_channels
        self.patch_embed = CosmosPatchEmbed(patch_embed_in_channels, hidden_size, patch_size, bias=False)
        self.control_patch_embed = CosmosPatchEmbed(control_in_channels, hidden_size, patch_size, bias=False)

        # Rotary Positional Embedding
        self.rope = Cosmos25RotaryPosEmbed(hidden_size=attention_head_dim, max_size=max_size, patch_size=patch_size, rope_scale=rope_scale)

        # Text Embedding Projection
        self.text_embed = nn.Sequential(
            nn.Linear(text_in_channels, text_embed_dim, bias=True),
            nn.GELU(),
        )

        # Time Embedding
        self.time_embed = Cosmos25TimeEmbed(hidden_size, hidden_size)
        self.time_norm = RMSNorm(hidden_size, eps=1e-6, elementwise_affine=True)

        # A selection of transformer blocks from the base model are duplicated here.
        transformer_blocks = dict()
        for block_id in block_ids:
            transformer_block = CosmosTransformerBlock(
                num_attention_heads=num_attention_heads,
                attention_head_dim=attention_head_dim,
                cross_attention_dim=text_embed_dim,
                mlp_ratio=mlp_ratio,
                adaln_lora_dim=adaln_lora_dim,
                qk_norm='rms_norm',
                out_bias=False,
            )
            transformer_blocks[str(block_id)] = transformer_block
        self.transformer_blocks = nn.ModuleDict(transformer_blocks)
        self.set_processor(Cosmos25AttnProcessor2_0())

        # Zero-initialized layers to connect control features to the base model.
        self.input_block = zero_module(nn.Linear(hidden_size, hidden_size))
        control_blocks = dict()
        for block_id in block_ids:
            control_block = zero_module(nn.Linear(hidden_size, hidden_size))
            control_blocks[str(block_id)] = control_block
        self.control_blocks = nn.ModuleDict(control_blocks)

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
        control_cond: torch.Tensor,
        control_scale: float = 1.0,
        attention_mask: Optional[torch.Tensor] = None,
        fps: Optional[int] = None,
        condition_mask: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """The forward pass for the ControlNet.

        Args:
            hidden_states (`torch.Tensor`): The input latents from the base model.
            timestep (`torch.Tensor`): The current timestep.
            encoder_hidden_states (`torch.Tensor`): The text embeddings.
            control_cond (`torch.Tensor`): The control condition (e.g., depth map).
            control_scale (`float`, defaults to `1.0`): The scale of the control signal.
            attention_mask (`torch.Tensor`, *optional*): An attention mask.
            fps (`int`, *optional*): Frames per second, used for positional embeddings.
            condition_mask (`torch.Tensor`, *optional*): Mask for conditional frames (image-to-video).
            padding_mask (`torch.Tensor`, *optional*): Mask for padded areas.

        Returns:
            `Dict[str, torch.Tensor]`: A dictionary of control signals for each specified block ID.
        """
        sp_group = get_sequence_parallel_group()
        # Sequence parallelism for distributed training
        if sp_group is not None:
            if timestep.shape[1] == hidden_states.shape[2]:
                timestep = split_forward_gather_backward(timestep, dim=1, group=sp_group)
            hidden_states = split_forward_gather_backward(hidden_states, dim=2, group=sp_group)
            control_cond = split_forward_gather_backward(control_cond, dim=2, group=sp_group)
            condition_mask = split_forward_gather_backward(condition_mask, dim=2, group=sp_group)
        batch_size, num_channels, num_frames, height, width = hidden_states.shape

        # Pad control condition if its channel dimension is smaller than expected
        if control_cond.shape[1] < self.config.control_in_channels - 2:
            padding_shape = list(control_cond.shape)
            padding_shape[1] = self.config.control_in_channels - 2 - control_cond.shape[1]
            padding_control_cond = torch.zeros(padding_shape, device=control_cond.device, dtype=control_cond.dtype)
            control_cond = torch.cat([control_cond, padding_control_cond], dim=1)

        # Concatenate condition mask (for i2v) if provided
        if condition_mask is not None:
            hidden_states = torch.cat([hidden_states, condition_mask], dim=1)
            control_cond = torch.cat([control_cond, condition_mask], dim=1)

        # Concatenate padding mask if configured
        if self.config.concat_padding_mask:
            padding_mask = transforms.functional.resize(
                padding_mask, list(hidden_states.shape[-2:]), interpolation=transforms.InterpolationMode.NEAREST
            )
            padding_mask = padding_mask.unsqueeze(2).repeat(1, 1, num_frames, 1, 1)
            hidden_states = torch.cat([hidden_states, padding_mask], dim=1)
            control_cond = torch.cat([control_cond, padding_mask], dim=1)

        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, S]

        # Generate rotary positional embeddings
        image_rotary_emb = self.rope(hidden_states, fps=fps)
        extra_pos_emb = None

        # Patchify input and control condition
        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w
        hidden_states = self.patch_embed(hidden_states)
        hidden_states = hidden_states.flatten(1, 3)  # [B, T, H, W, C] -> [B, THW, C]

        control_cond = self.control_patch_embed(control_cond)
        control_cond = control_cond.flatten(1, 3)  # [B, T, H, W, C] -> [B, THW, C]

        # Add control condition to the main hidden states
        control_cond = self.input_block(control_cond)
        hidden_states = hidden_states + control_cond

        # Project text embeddings
        encoder_hidden_states = self.text_embed(encoder_hidden_states)

        # Prepare time embeddings
        timestep = timestep.flatten()
        embedded_timestep, temb = self.time_embed(hidden_states, timestep)
        embedded_timestep = rearrange(embedded_timestep, '(B T) C -> B T C', B=batch_size)
        temb = rearrange(temb, '(B T) C -> B T C', B=batch_size)
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

        # Pass through the duplicated transformer blocks
        outputs = dict()
        for block_id, block in self.transformer_blocks.items():
            hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                embedded_timestep=embedded_timestep,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                extra_pos_emb=extra_pos_emb,
                attention_mask=attention_mask,
            )
            # Generate the control signal for this block and scale it
            control_hidden_states = self.control_blocks[str(block_id)](hidden_states) * control_scale
            outputs[str(block_id)] = control_hidden_states

        return outputs
