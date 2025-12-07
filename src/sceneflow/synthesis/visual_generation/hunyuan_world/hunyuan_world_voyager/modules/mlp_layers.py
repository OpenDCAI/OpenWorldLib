# Modified from timm library:
# https://github.com/huggingface/pytorch-image-models/blob/648aaa41233ba83eb38faf5ba9d415d574823241/timm/layers/mlp.py#L13

from functools import partial

import torch
import torch.nn as nn

from .modulate_layers import modulate
from ..utils.helpers import to_2tuple


class MLP(nn.Module):
    """Multi-Layer Perceptron (MLP) as used in Vision Transformer, MLP-Mixer and related networks
    
    This is a flexible MLP implementation that can be used as a building block
    in various transformer-based architectures. It supports both linear and
    convolutional layers, with optional normalization and dropout.
    """

    def __init__(
        self,
        in_channels,
        hidden_channels=None,
        out_features=None,
        act_layer=nn.GELU,
        norm_layer=None,
        bias=True,
        drop=0.0,
        use_conv=False,
        device=None,
        dtype=None,
    ):
        """Initialize MLP layer
        
        Args:
            in_channels (int): Number of input channels/features
            hidden_channels (int, optional): Number of hidden channels. Defaults to in_channels
            out_features (int, optional): Number of output features. Defaults to in_channels
            act_layer: Activation function class (default: nn.GELU)
            norm_layer: Normalization layer class (default: None)
            bias (bool or tuple): Whether to use bias in linear layers
            drop (float or tuple): Dropout probability for each layer
            use_conv (bool): Whether to use Conv2d instead of Linear layers
            device: Device to place the module on
            dtype: Data type for the module parameters
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        
        # Set default values for output and hidden dimensions
        out_features = out_features or in_channels
        hidden_channels = hidden_channels or in_channels
        
        # Convert single values to tuples for consistency
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        
        # Choose between linear and convolutional layers
        linear_layer = partial(
            nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        # First fully connected layer
        self.fc1 = linear_layer(
            in_channels, hidden_channels, bias=bias[0], **factory_kwargs
        )
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        
        # Optional normalization layer
        self.norm = (
            norm_layer(hidden_channels, **factory_kwargs)
            if norm_layer is not None
            else nn.Identity()
        )
        
        # Second fully connected layer
        self.fc2 = linear_layer(
            hidden_channels, out_features, bias=bias[1], **factory_kwargs
        )
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        """Forward pass through the MLP
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor after MLP transformation
        """
        x = self.fc1(x)      # First linear transformation
        x = self.act(x)      # Apply activation function
        x = self.drop1(x)    # Apply dropout
        x = self.norm(x)     # Apply normalization (if any)
        x = self.fc2(x)      # Second linear transformation
        x = self.drop2(x)    # Apply dropout
        return x


class MLPEmbedder(nn.Module):
    """MLP-based embedding layer
    
    A simple MLP used for embedding transformations, copied from the Flux library.
    This is typically used for processing conditional information or embeddings.
    """

    def __init__(self, in_dim: int, hidden_dim: int, device=None, dtype=None):
        """Initialize MLP embedder
        
        Args:
            in_dim (int): Input dimension
            hidden_dim (int): Hidden dimension for the MLP
            device: Device to place the module on
            dtype: Data type for the module parameters
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        
        # Input projection layer
        self.in_layer = nn.Linear(
            in_dim, hidden_dim, bias=True, **factory_kwargs)
        self.silu = nn.SiLU()  # Swish/SiLU activation function
        
        # Output projection layer
        self.out_layer = nn.Linear(
            hidden_dim, hidden_dim, bias=True, **factory_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MLP embedder
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Embedded tensor with same dimension as hidden_dim
        """
        return self.out_layer(self.silu(self.in_layer(x)))


class FinalLayer(nn.Module):
    """The final layer of DiT (Diffusion Transformer)
    
    This layer is responsible for the final output projection in diffusion models.
    It includes adaptive layer normalization (AdaLN) modulation and projects
    the hidden representations to the output space (e.g., pixel values).
    """

    def __init__(
        self, hidden_size, patch_size, out_channels, act_layer, device=None, dtype=None
    ):
        """Initialize the final layer
        
        Args:
            hidden_size (int): Size of the hidden representations
            patch_size (int or tuple): Size of patches (H, W) or single value
            out_channels (int): Number of output channels
            act_layer: Activation function for the modulation network
            device: Device to place the module on
            dtype: Data type for the module parameters
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        # Layer normalization without learnable parameters
        self.norm_final = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs
        )
        
        # Output projection layer
        if isinstance(patch_size, int):
            # For 2D patches (H=W=patch_size)
            self.linear = nn.Linear(
                hidden_size,
                patch_size * patch_size * out_channels,
                bias=True,
                **factory_kwargs
            )
        else:
            # For 3D patches (H, W, T)
            self.linear = nn.Linear(
                hidden_size,
                patch_size[0] * patch_size[1] * patch_size[2] * out_channels,
                bias=True,
            )
        
        # Zero-initialize the output projection for stable training
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

        # Adaptive Layer Normalization (AdaLN) modulation network
        # This network generates scale and shift parameters based on conditioning
        self.adaLN_modulation = nn.Sequential(
            act_layer(),
            nn.Linear(hidden_size, 2 * hidden_size,
                      bias=True, **factory_kwargs),
        )
        
        # Zero-initialize the modulation network for stable training
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(self, x, c):
        """Forward pass through the final layer
        
        Args:
            x (torch.Tensor): Input hidden representations
            c (torch.Tensor): Conditioning information for AdaLN modulation
            
        Returns:
            torch.Tensor: Final output projections
        """
        # Generate scale and shift parameters from conditioning
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        
        # Apply adaptive layer normalization with modulation
        x = modulate(self.norm_final(x), shift=shift, scale=scale)
        
        # Project to output space
        x = self.linear(x)
        return x
