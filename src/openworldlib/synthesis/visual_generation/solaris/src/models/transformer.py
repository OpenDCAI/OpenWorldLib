import math

import jax
import jax.numpy as jnp
from einops import rearrange
from flax import nnx
from jax import lax

from .transformer_utils import rope_apply


class WanRMSNorm(nnx.Module):
    """RMS normalization layer."""

    def __init__(
        self,
        dim,
        eps=1e-5,
        dtype=jnp.bfloat16,  # unused for now
        param_dtype=jnp.float32,
    ):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nnx.Param(
            jnp.ones(dim, dtype=param_dtype), param_dtype=param_dtype
        )
        self.param_dtype = param_dtype

    def __call__(self, x):
        """
        Args:
            x: Input tensor of shape [batch, seq_len, dim]
        """
        input_dtype = x.dtype
        x = x.astype(jnp.float32)
        x = x * jax.lax.rsqrt(jnp.mean(x**2, axis=-1, keepdims=True) + self.eps)
        x = x.astype(input_dtype) * self.weight.value.astype(input_dtype)
        return x


class WanLayerNorm(nnx.LayerNorm):
    """
    A wrapper around flax.nnx.LayerNorm that mimics the `elementwise_affine`
    parameter from the PyTorch nn.LayerNorm API for compatibility.
    """

    def __init__(
        self,
        num_features,
        eps=1e-6,
        elementwise_affine=False,
        rngs=nnx.Rngs(0),
        param_dtype=jnp.float32,
        dtype=jnp.float32,
    ):
        """
        Initializes the LayerNorm wrapper.

        Args:
            num_features: The number of num_features in the input.
            eps: A small float added to variance to avoid dividing by zero.
            elementwise_affine: If True, this module has learnable affine
                parameters (scale and bias). Corresponds to `use_scale` and
                `use_bias` in the parent `nnx.LayerNorm`. Default is False
                according to the PyTorch model at
                https://github.com/SkyworkAI/SkyReels-V2/blob/main/skyreels_v2_infer/modules/transformer.py#L104.
            rngs: The random number generator to use for initialization.
        """
        # The `use_scale` and `use_bias` arguments in nnx.LayerNorm directly
        # correspond to the behavior of `elementwise_affine`.
        super().__init__(
            num_features=num_features,
            epsilon=eps,
            use_bias=elementwise_affine,
            use_scale=elementwise_affine,
            rngs=rngs,
            dtype=dtype,
            param_dtype=param_dtype,
        )


class Conv3d(nnx.Module):
    """3D Convolution implementation using JAX's lax.conv_general_dilated."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        rngs=nnx.Rngs(0),
        strides=(1, 1, 1),
        padding="VALID",
        param_dtype=jnp.float32,
        dtype=jnp.bfloat16,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding

        # Initialize 3D convolution kernel
        # Kernel shape: [kernel_d, kernel_h, kernel_w, in_channels, out_channels]
        key = jax.random.PRNGKey(42)
        kernel_shape = (*kernel_size, in_channels, out_channels)
        self.kernel = nnx.Param(
            jax.random.normal(key, kernel_shape, dtype=jnp.float32)
            * (
                2.0
                / math.sqrt(
                    in_channels * kernel_size[0] * kernel_size[1] * kernel_size[2]
                )
            )
        )
        # Initialize bias
        self.bias = nnx.Param(jnp.zeros(out_channels, dtype=param_dtype))
        self.dtype = dtype
        self.param_dtype = param_dtype

    def __call__(self, x):
        """
        Args:
            x: Input tensor of shape [batch, channels, depth, height, width]
        """
        # Define dimension numbers for 3D convolution
        # Input: [batch, channels, depth, height, width] -> [batch, depth, height, width, channels]
        # Kernel: [depth, height, width, in_channels, out_channels]
        # Output: [batch, depth_out, height_out, width_out, out_channels] -> [batch, out_channels, depth_out, height_out, width_out]

        # Transpose input to NHWDC format
        x = x.transpose(0, 2, 3, 4, 1)  # [batch, depth, height, width, channels]

        # Define dimension numbers for 3D convolution
        # N=batch, D=depth, H=height, W=width, C=channels, I=input_channels, O=output_channels
        # The input is transposed to [batch, depth, height, width, channels]
        # The kernel is [depth, height, width, in_channels, out_channels]
        dn = ("NDHWC", "DHWIO", "NDHWC")

        # Handle padding
        if self.padding == "SAME":
            padding = "SAME"
        elif self.padding == "VALID":
            padding = "VALID"
        else:
            padding = ((0, 0), (0, 0), (0, 0))

        kernel = self.kernel
        if x.dtype != kernel.dtype:
            kernel = kernel.astype(x.dtype)

        output = lax.conv_general_dilated(
            lhs=x,  # Input tensor
            rhs=kernel,  # Kernel tensor
            window_strides=self.strides,  # Strides
            padding=padding,  # Padding
            lhs_dilation=(1, 1, 1),  # Input dilation
            rhs_dilation=(1, 1, 1),  # Kernel dilation
            dimension_numbers=dn,  # Dimension numbers
        )
        bias = self.bias
        if output.dtype != bias.dtype:
            bias = bias.astype(output.dtype)
        output = output + bias

        output = output.transpose(0, 4, 1, 2, 3)
        return output


class WanSelfAttention(nnx.Module):
    """Self-attention layer with RoPE."""

    def __init__(
        self,
        dim,
        num_heads,
        window_size=(-1, -1),
        qk_norm=True,
        eps=1e-6,
        rngs=nnx.Rngs(0),
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # Linear projections
        self.q = nnx.Linear(
            dim, dim, rngs=rngs, param_dtype=jnp.float32, dtype=jnp.bfloat16
        )
        self.k = nnx.Linear(
            dim, dim, rngs=rngs, param_dtype=jnp.float32, dtype=jnp.bfloat16
        )
        self.v = nnx.Linear(
            dim, dim, rngs=rngs, param_dtype=jnp.float32, dtype=jnp.bfloat16
        )
        self.o = nnx.Linear(
            dim, dim, rngs=rngs, param_dtype=jnp.float32, dtype=jnp.bfloat16
        )

        # Normalization layers
        self.norm_q = (
            WanRMSNorm(dim, eps, dtype=jnp.bfloat16, param_dtype=jnp.float32)
            if qk_norm
            else lambda x: x
        )
        self.norm_k = (
            WanRMSNorm(dim, eps, dtype=jnp.bfloat16, param_dtype=jnp.float32)
            if qk_norm
            else lambda x: x
        )

    def __call__(
        self,
        x,
        grid_sizes,
        freqs,
        block_mask=None,
    ):
        """
        Args:
            x: Input tensor of shape [batch, seq_len, dim]
            grid_sizes: Grid sizes [F, H, W]
            freqs: RoPE frequency parameters
            block_mask: Optional attention mask
        """
        b, n, d = x.shape[0], self.num_heads, self.head_dim

        q = self.norm_q(self.q(x)).reshape(b, -1, n, d)
        k = self.norm_k(self.k(x)).reshape(b, -1, n, d)
        v = self.v(x).reshape(b, -1, n, d)

        q = rope_apply(q, grid_sizes, freqs)
        k = rope_apply(k, grid_sizes, freqs)

        x = jax.nn.dot_product_attention(q, k, v, is_causal=False)

        x = x.reshape(b, -1, self.dim)
        x = self.o(x)
        return x


class WanI2VCrossAttention(WanSelfAttention):
    def __call__(self, x_bsd, context_btd):
        "b: batch, s: seq_len, d: dim, t: number of embeddings in context"
        q = rearrange(
            self.norm_q(self.q(x_bsd)), "b s (nh hd) -> b s nh hd", nh=self.num_heads
        )
        k = rearrange(
            self.norm_k(self.k(context_btd)),
            "b t (nh hd) -> b t nh hd",
            nh=self.num_heads,
        )
        v = rearrange(
            self.v(context_btd), "b t (nh hd) -> b t nh hd", nh=self.num_heads
        )
        x = jax.nn.dot_product_attention(q, k, v)
        x = rearrange(x, "b s nh hd -> b s (nh hd)")
        x = self.o(x)
        return x


class MLPProj(nnx.Module):
    """
    MLP projection for image embeddings.
    CORRECTED: This class now uses a standard Python list to hold modules,
    which nnx correctly nests to match the PyTorch nn.Sequential structure.
    """

    def __init__(self, in_dim, out_dim):
        super().__init__()
        # This epsilon is the hardcoded default in torch.nn.LayerNorm.
        # The original MLPProj module does NOT receive the main model's
        # `eps` from the config, so we must match the PyTorch default here.
        TORCH_DEFAULT_EPS = 1e-5
        self.proj = [
            WanLayerNorm(
                in_dim, eps=TORCH_DEFAULT_EPS, elementwise_affine=True, rngs=nnx.Rngs(0)
            ),  # index 0
            nnx.Linear(
                in_dim,
                in_dim,
                rngs=nnx.Rngs(0),
                param_dtype=jnp.float32,
                dtype=jnp.bfloat16,
            ),  # index 1
            # nn.GELU is just a function, so it doesn't have parameters and
            # doesn't need a placeholder in the list.
            nnx.Linear(
                in_dim,
                out_dim,
                rngs=nnx.Rngs(0),
                param_dtype=jnp.float32,
                dtype=jnp.bfloat16,
            ),  # index 3 in PyTorch
            WanLayerNorm(
                out_dim,
                eps=TORCH_DEFAULT_EPS,
                elementwise_affine=True,
                rngs=nnx.Rngs(0),
            ),  # index 4 in PyTorch
        ]

    def __call__(self, image_embeds):
        # Apply the layers sequentially, just like nn.Sequential.
        x = self.proj[0](image_embeds.astype(jnp.float32)).astype(image_embeds.dtype)
        x = self.proj[1](x)
        x = jax.nn.gelu(x, approximate=True)  # Apply GELU activation.
        x = self.proj[2](x)  # Corresponds to index 3 in PyTorch
        x = self.proj[3](x.astype(jnp.float32)).astype(image_embeds.dtype)
        return x
