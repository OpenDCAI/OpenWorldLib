
import einops
import jax
import jax.numpy as jnp
from einops import rearrange


def sinusoidal_embedding_1d(dim, position):
    """
    Create sinusoidal positional embeddings for 1D sequences.
    Args:
        dim: Embedding dimension (must be even)
        position: Position indices of shape [seq_len]
    Returns:
        Sinusoidal embeddings of shape [seq_len, dim]
    """
    assert dim % 2 == 0, "Dimension must be even"
    half = dim // 2
    try:
        position = position.astype(jnp.float64)
        dtype = jnp.float64
    except Exception:
        position = position.astype(jnp.float32)
        dtype = jnp.float32
    freqs = jnp.outer(
        position, jnp.power(10000.0, -jnp.arange(half, dtype=dtype) / dtype(half))
    )
    x = jnp.concatenate([jnp.cos(freqs), jnp.sin(freqs)], axis=1)
    return x


def rope_params(max_seq_len, dim, theta=10000.0):
    """
    Generate RoPE (Rotary Position Embedding) parameters.
    Args:
        max_seq_len: Maximum sequence length
        dim: Embedding dimension (must be even)
        theta: Base frequency for RoPE
    Returns:
        RoPE frequency parameters of shape [max_seq_len, dim//2]
    """
    assert dim % 2 == 0, "Dimension must be even"
    freqs = jnp.outer(
        jnp.arange(max_seq_len), 1.0 / jnp.power(theta, jnp.arange(0, dim, 2) / dim)
    )
    freqs = jnp.exp(1j * freqs)
    return freqs


def rope_apply(
    x,
    grid_sizes,
    freqs,
    start_frame=0,
):
    _, c = x.shape[2], x.shape[3] // 2
    split_boundaries = [
        c - 2 * (c // 3),
        c - 2 * (c // 3) + c // 3,
    ]
    freqs = jnp.split(freqs, split_boundaries, axis=1)
    f, h, w = grid_sizes
    seq_len = f * h * w
    sliced_time_freq_FD = jax.lax.dynamic_slice_in_dim(freqs[0], start_frame, f, axis=0)

    freqs_3d = jnp.concatenate(
        [
            einops.repeat(sliced_time_freq_FD, "f d -> f h w d", w=w, h=h),
            einops.repeat(freqs[1][:h], "h d -> f h w d", f=f, w=w),
            einops.repeat(freqs[2][:w], "w d -> f h w d", f=f, h=h),
        ],
        axis=-1,
    ).reshape(1, seq_len, 1, -1)

    x_reshaped = rearrange(x.astype(jnp.float32), "b s n (c r) -> b s n c r", r=2)
    x_complex = (x_reshaped[:, :, :, :, 0] + x_reshaped[:, :, :, :, 1] * 1j).astype(
        jnp.complex64
    )
    x_roped = x_complex * freqs_3d
    x_i = rearrange(
        jnp.stack([x_roped.real, x_roped.imag], axis=-1), "b s n c r -> b s n (c r)"
    )
    return x_i.astype(x.dtype)


def apply_rope_mp(
    x,
    grid_sizes,
    freqs,
    f_arg,
    s_arg,
    current_start=0,
):
    """
    Apply RoPE (Rotary Position Embedding) to input tensor with multiplayer support.

    This function handles the reshape required for multiplayer scenarios where the input
    has shape [B, F*P*S, N, D] with P players interleaved per frame.

    Args:
        x: Input tensor of shape [B, F*P*S, N, D] where:
           - B = batch size
           - F = number of frames
           - P = number of players (inferred from tensor shape)
           - S = spatial tokens per frame (H * W)
           - N = number of heads
           - D = head dimension
        grid_sizes: Tuple of (F, H, W) for the video grid
        freqs: Precomputed RoPE frequency parameters
        f_arg: Number of frames (F)
        s_arg: Number of spatial tokens per frame (S = H * W)
        current_start: Starting frame index for RoPE positions (for KV cache inference)

    Returns:
        Tensor with RoPE applied, same shape as input [B, F*P*S, N, D]
    """
    b = x.shape[0]
    return rearrange(
        rope_apply(
            rearrange(x, "b (f p s) n d -> (b p) (f s) n d", f=f_arg, s=s_arg),
            grid_sizes,
            freqs,
            start_frame=current_start,
        ).astype(x.dtype),
        "(b p) (f s) n d -> b (f p s) n d",
        b=b,
        f=f_arg,
    )


def mul_add(x, y, z):
    orig_dtype = x.dtype
    result = x.astype(jnp.float32) + y.astype(jnp.float32) * z.astype(jnp.float32)
    return result.astype(orig_dtype)


def mul_add_add(x, y, z):
    orig_dtype = x.dtype
    result = x.astype(jnp.float32) * (1 + y) + z
    return result.astype(orig_dtype)
