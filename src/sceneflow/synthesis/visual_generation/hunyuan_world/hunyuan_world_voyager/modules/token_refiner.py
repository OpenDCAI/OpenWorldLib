from typing import Optional

from einops import rearrange
import torch
import torch.nn as nn

from .activation_layers import get_activation_layer
from .attenion import attention
from .norm_layers import get_norm_layer
from .embed_layers import TimestepEmbedder, TextProjection
from .attenion import attention
from .mlp_layers import MLP
from .modulate_layers import modulate, apply_gate


class IndividualTokenRefinerBlock(nn.Module):
    """Individual token refinement block with self-attention and MLP
    
    This block refines individual tokens using self-attention and MLP layers
    with adaptive layer normalization (AdaLN) modulation. It's designed to
    process token-level representations with conditioning information.
    """

    def __init__(
        self,
        hidden_size,
        heads_num,
        mlp_width_ratio: str = 4.0,
        mlp_drop_rate: float = 0.0,
        act_type: str = "silu",
        qk_norm: bool = False,
        qk_norm_type: str = "layer",
        qkv_bias: bool = True,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        """Initialize the token refinement block
        
        Args:
            hidden_size (int): Size of hidden representations
            heads_num (int): Number of attention heads
            mlp_width_ratio (float): Ratio of MLP hidden dimension to input dimension
            mlp_drop_rate (float): Dropout rate for MLP layers
            act_type (str): Type of activation function
            qk_norm (bool): Whether to apply QK normalization
            qk_norm_type (str): Type of normalization for QK
            qkv_bias (bool): Whether to use bias in QKV projection
            dtype: Data type for parameters
            device: Device to place the module on
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.heads_num = heads_num
        head_dim = hidden_size // heads_num
        mlp_hidden_dim = int(hidden_size * mlp_width_ratio)

        # First normalization layer for self-attention
        self.norm1 = nn.LayerNorm(
            hidden_size, elementwise_affine=True, eps=1e-6, **factory_kwargs
        )
        
        # QKV projection for self-attention
        self.self_attn_qkv = nn.Linear(
            hidden_size, hidden_size * 3, bias=qkv_bias, **factory_kwargs
        )
        
        # QK normalization layers (optional)
        qk_norm_layer = get_norm_layer(qk_norm_type)
        self.self_attn_q_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True,
                          eps=1e-6, **factory_kwargs)
            if qk_norm
            else nn.Identity()
        )
        self.self_attn_k_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True,
                          eps=1e-6, **factory_kwargs)
            if qk_norm
            else nn.Identity()
        )
        
        # Output projection for self-attention
        self.self_attn_proj = nn.Linear(
            hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs
        )

        # Second normalization layer for MLP
        self.norm2 = nn.LayerNorm(
            hidden_size, elementwise_affine=True, eps=1e-6, **factory_kwargs
        )
        
        # MLP layer
        act_layer = get_activation_layer(act_type)
        self.mlp = MLP(
            in_channels=hidden_size,
            hidden_channels=mlp_hidden_dim,
            act_layer=act_layer,
            drop=mlp_drop_rate,
            **factory_kwargs,
        )

        # AdaLN modulation network for conditioning
        self.adaLN_modulation = nn.Sequential(
            act_layer(),
            nn.Linear(hidden_size, 2 * hidden_size,
                      bias=True, **factory_kwargs),
        )
        # Zero-initialize the modulation for stable training
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,  # timestep_aware_representations + context_aware_representations
        attn_mask: torch.Tensor = None,
    ):
        """Forward pass through the token refinement block
        
        Args:
            x (torch.Tensor): Input token representations
            c (torch.Tensor): Conditioning information (timestep + context)
            attn_mask (torch.Tensor, optional): Attention mask for padding
            
        Returns:
            torch.Tensor: Refined token representations
        """
        # Generate gating parameters from conditioning
        gate_msa, gate_mlp = self.adaLN_modulation(c).chunk(2, dim=1)

        # Self-attention branch
        norm_x = self.norm1(x)
        qkv = self.self_attn_qkv(norm_x)
        q, k, v = rearrange(qkv, "B L (K H D) -> K B L H D",
                            K=3, H=self.heads_num)
        
        # Apply QK normalization if enabled
        q = self.self_attn_q_norm(q).to(v)
        k = self.self_attn_k_norm(k).to(v)

        # Compute self-attention
        attn = attention(q, k, v, mode="torch", attn_mask=attn_mask)

        # Apply gated residual connection for self-attention
        x = x + apply_gate(self.self_attn_proj(attn), gate_msa)

        # MLP branch with gated residual connection
        x = x + apply_gate(self.mlp(self.norm2(x)), gate_mlp)

        return x


class IndividualTokenRefiner(nn.Module):
    """Stack of individual token refinement blocks
    
    This module stacks multiple IndividualTokenRefinerBlock layers to create
    a deep token refinement network with proper attention masking.
    """

    def __init__(
        self,
        hidden_size,
        heads_num,
        depth,
        mlp_width_ratio: float = 4.0,
        mlp_drop_rate: float = 0.0,
        act_type: str = "silu",
        qk_norm: bool = False,
        qk_norm_type: str = "layer",
        qkv_bias: bool = True,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        """Initialize the token refiner stack
        
        Args:
            hidden_size (int): Size of hidden representations
            heads_num (int): Number of attention heads
            depth (int): Number of refinement blocks
            mlp_width_ratio (float): MLP width ratio
            mlp_drop_rate (float): MLP dropout rate
            act_type (str): Activation function type
            qk_norm (bool): Whether to use QK normalization
            qk_norm_type (str): QK normalization type
            qkv_bias (bool): Whether to use bias in QKV
            dtype: Data type for parameters
            device: Device to place the module on
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        
        # Create stack of refinement blocks
        self.blocks = nn.ModuleList(
            [
                IndividualTokenRefinerBlock(
                    hidden_size=hidden_size,
                    heads_num=heads_num,
                    mlp_width_ratio=mlp_width_ratio,
                    mlp_drop_rate=mlp_drop_rate,
                    act_type=act_type,
                    qk_norm=qk_norm,
                    qk_norm_type=qk_norm_type,
                    qkv_bias=qkv_bias,
                    **factory_kwargs,
                )
                for _ in range(depth)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        c: torch.LongTensor,
        mask: Optional[torch.Tensor] = None,
    ):
        """Forward pass through the token refiner stack
        
        Args:
            x (torch.Tensor): Input token representations
            c (torch.LongTensor): Conditioning information
            mask (torch.Tensor, optional): Padding mask for attention
            
        Returns:
            torch.Tensor: Refined token representations
        """
        # Initialize attention mask
        self_attn_mask = None
        if mask is not None:
            batch_size = mask.shape[0]
            seq_len = mask.shape[1]
            mask = mask.to(x.device)
            
            # Create attention mask for padding tokens
            # batch_size x 1 x seq_len x seq_len
            self_attn_mask_1 = mask.view(batch_size, 1, 1, seq_len).repeat(
                1, 1, seq_len, 1
            )
            # batch_size x 1 x seq_len x seq_len
            self_attn_mask_2 = self_attn_mask_1.transpose(2, 3)
            # batch_size x 1 x seq_len x seq_len, 1 for broadcasting of heads_num
            self_attn_mask = (self_attn_mask_1 & self_attn_mask_2).bool()
            # avoids self-attention weight being NaN for padding tokens
            self_attn_mask[:, :, :, 0] = True

        # Pass through all refinement blocks
        for block in self.blocks:
            x = block(x, c, self_attn_mask)
        return x


class SingleTokenRefiner(nn.Module):
    """A single token refiner for LLM text embedding refinement
    
    This module refines LLM text embeddings by incorporating timestep and
    context information through a stack of transformer blocks.
    """

    def __init__(
        self,
        in_channels,
        hidden_size,
        heads_num,
        depth,
        mlp_width_ratio: float = 4.0,
        mlp_drop_rate: float = 0.0,
        act_type: str = "silu",
        qk_norm: bool = False,
        qk_norm_type: str = "layer",
        qkv_bias: bool = True,
        attn_mode: str = "torch",
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        """Initialize the single token refiner
        
        Args:
            in_channels (int): Input embedding dimension
            hidden_size (int): Hidden dimension for refinement
            heads_num (int): Number of attention heads
            depth (int): Number of refinement blocks
            mlp_width_ratio (float): MLP width ratio
            mlp_drop_rate (float): MLP dropout rate
            act_type (str): Activation function type
            qk_norm (bool): Whether to use QK normalization
            qk_norm_type (str): QK normalization type
            qkv_bias (bool): Whether to use bias in QKV
            attn_mode (str): Attention mode (only "torch" supported)
            dtype: Data type for parameters
            device: Device to place the module on
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.attn_mode = attn_mode
        assert self.attn_mode == "torch", "Only support 'torch' mode for token refiner."

        # Input projection layer
        self.input_embedder = nn.Linear(
            in_channels, hidden_size, bias=True, **factory_kwargs
        )

        act_layer = get_activation_layer(act_type)
        
        # Timestep embedding layer for diffusion conditioning
        self.t_embedder = TimestepEmbedder(
            hidden_size, act_layer, **factory_kwargs)
        
        # Context embedding layer for text conditioning
        self.c_embedder = TextProjection(
            in_channels, hidden_size, act_layer, **factory_kwargs
        )

        # Stack of token refinement blocks
        self.individual_token_refiner = IndividualTokenRefiner(
            hidden_size=hidden_size,
            heads_num=heads_num,
            depth=depth,
            mlp_width_ratio=mlp_width_ratio,
            mlp_drop_rate=mlp_drop_rate,
            act_type=act_type,
            qk_norm=qk_norm,
            qk_norm_type=qk_norm_type,
            qkv_bias=qkv_bias,
            **factory_kwargs,
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.LongTensor,
        mask: Optional[torch.LongTensor] = None,
    ):
        """Forward pass through the token refiner
        
        Args:
            x (torch.Tensor): Input LLM embeddings
            t (torch.LongTensor): Timestep information
            mask (torch.LongTensor, optional): Padding mask
            
        Returns:
            torch.Tensor: Refined embeddings
        """
        # Generate timestep-aware representations
        timestep_aware_representations = self.t_embedder(t)

        # Generate context-aware representations from input embeddings
        if mask is None:
            # Average pooling if no mask provided
            context_aware_representations = x.mean(dim=1)
        else:
            # Masked average pooling for padded sequences
            mask_float = mask.float().unsqueeze(-1)  # [b, s1, 1]
            context_aware_representations = (x * mask_float).sum(
                dim=1
            ) / mask_float.sum(dim=1)
        
        # Project context representations to hidden dimension
        context_aware_representations = self.c_embedder(
            context_aware_representations)
        
        # Combine timestep and context information
        c = timestep_aware_representations + context_aware_representations

        # Project input embeddings to hidden dimension
        x = self.input_embedder(x)

        # Apply token refinement
        x = self.individual_token_refiner(x, c, mask)

        return x
