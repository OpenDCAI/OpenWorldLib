
import jax
import jax.numpy as jnp


def _to_tuple(x, dim=2):
    if isinstance(x, int):
        return (x,) * dim
    elif len(x) == dim:
        return x
    else:
        raise ValueError(f"Expected length {dim} or int, but got {x}")


def get_meshgrid_nd(start, *args, dim=2):
    """
    Get n-D meshgrid with start, stop and num.

    Args:
        start (int or tuple): If len(args) == 0, start is num; If len(args) == 1, start is start, args[0] is stop,
            step is 1; If len(args) == 2, start is start, args[0] is stop, args[1] is num. For n-dim, start/stop/num
            should be int or n-tuple. If n-tuple is provided, the meshgrid will be stacked following the dim order in
            n-tuples.
        *args: See above.
        dim (int): Dimension of the meshgrid. Defaults to 2.

    Returns:
        grid (np.ndarray): [dim, ...]
    """

    if len(args) == 0:
        # start is grid_size
        num = _to_tuple(start, dim=dim)
        start = (0,) * dim
        stop = num

    elif len(args) == 1:
        # start is start, args[0] is stop, step is 1
        start = _to_tuple(start, dim=dim)
        stop = _to_tuple(args[0], dim=dim)
        num = [stop[i] - start[i] for i in range(dim)]

    elif len(args) == 2:
        # start is start, args[0] is stop, args[1] is num
        start = _to_tuple(start, dim=dim)  # Left-Top       eg: 12,0
        stop = _to_tuple(args[0], dim=dim)  # Right-Bottom   eg: 20,32
        num = _to_tuple(args[1], dim=dim)  # Target Size    eg: 32,124
    else:
        raise ValueError(f"len(args) should be 0, 1 or 2, but got {len(args)}")

    # PyTorch implement of np.linspace(start[i], stop[i], num[i], endpoint=False)
    axis_grid = []
    for i in range(dim):
        a, b, n = start[i], stop[i], num[i]
        g = jnp.linspace(a, b, n + 1, dtype=jnp.float32)[:n]
        axis_grid.append(g)
    grid = jnp.meshgrid(*axis_grid, indexing="ij")  # dim x [W, H, D]
    grid = jnp.stack(grid, axis=0)  # [dim, W, H, D]

    return grid


#################################################################################
#                   Rotary Positional Embedding Functions                       #
#################################################################################
# https://github.com/meta-llama/llama/blob/be327c427cc5e89cc1d3ab3d3fec4484df771245/llama/model.py#L80


def reshape_for_broadcast(
    freqs_cis,
    x,
    head_first=False,
):
    """
    Reshape frequency tensor for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Notes:
        When using FlashMHAModified, head_first should be False.
        When using Attention, head_first should be True.

    Args:
        freqs_cis (Union[torch.Tensor, Tuple[torch.Tensor]]): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.
        head_first (bool): head dimension first (except batch dim) or not.

    Returns:
        torch.Tensor: Reshaped frequency tensor.

    Raises:
        AssertionError: If the frequency tensor doesn't match the expected shape.
        AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim

    if isinstance(freqs_cis, tuple):
        # freqs_cis: (cos, sin) in real space
        if head_first:
            assert freqs_cis[0].shape == (
                x.shape[-2],
                x.shape[-1],
            ), f"freqs_cis shape {freqs_cis[0].shape} does not match x shape {x.shape}"
            shape = [
                d if i == ndim - 2 or i == ndim - 1 else 1
                for i, d in enumerate(x.shape)
            ]
        else:
            shape = [1, freqs_cis[0].shape[0], 1, freqs_cis[0].shape[1]]
        return freqs_cis[0].reshape(*shape), freqs_cis[1].reshape(*shape)
    else:
        # freqs_cis: values in complex space
        if head_first:
            assert freqs_cis.shape == (
                x.shape[-2],
                x.shape[-1],
            ), f"freqs_cis shape {freqs_cis.shape} does not match x shape {x.shape}"
            shape = [
                d if i == ndim - 2 or i == ndim - 1 else 1
                for i, d in enumerate(x.shape)
            ]
        else:
            assert freqs_cis.shape == (
                x.shape[1],
                x.shape[-1],
            ), f"freqs_cis shape {freqs_cis.shape} does not match x shape {x.shape}"
            shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return freqs_cis.reshape(*shape)


def rotate_half(x):
    x = x.astype(jnp.float32)
    x_reshaped = x.reshape(*x.shape[:-1], -1, 2)
    x_real = x_reshaped[..., 0]
    x_imag = x_reshaped[..., 1]
    return jnp.stack([-x_imag, x_real], axis=-1).reshape(x.shape)


def apply_rotary_emb(
    xq,
    xk,
    freqs_cis,
    head_first=False,
    start_offset=0,
):
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors using the provided
    frequency tensor 'freqs_cis'. The input tensors are reshaped as complex numbers, and the frequency tensor
    is reshaped for broadcasting compatibility. The resulting tensors contain rotary embeddings and are
    returned as real tensors.

    Args:
        xq (torch.Tensor): Query tensor to apply rotary embeddings. [B, S, H, D]
        xk (torch.Tensor): Key tensor to apply rotary embeddings.   [B, S, H, D]
        freqs_cis (torch.Tensor or tuple): Precomputed frequency tensor for complex exponential.
        head_first (bool): head dimension first (except batch dim) or not.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.

    """
    xk_out = None
    cos, sin = reshape_for_broadcast(freqs_cis, xq, head_first)  # [S, D]
    xq_float = xq.astype(jnp.float32)
    xk_float = xk.astype(jnp.float32)
    xq_out = (
        xq_float * jax.lax.dynamic_slice_in_dim(cos, start_offset, xq.shape[1], axis=1)
        + rotate_half(xq_float)
        * jax.lax.dynamic_slice_in_dim(sin, start_offset, xq.shape[1], axis=1)
    ).astype(xq.dtype)
    xk_out = (
        xk_float * jax.lax.dynamic_slice_in_dim(cos, start_offset, xk.shape[1], axis=1)
        + rotate_half(xk_float)
        * jax.lax.dynamic_slice_in_dim(sin, start_offset, xk.shape[1], axis=1)
    ).astype(xk.dtype)

    return xq_out, xk_out


def get_nd_rotary_pos_embed(
    rope_dim_list,
    start,
    *args,
    theta=10000.0,
    use_real=False,
    theta_rescale_factor=1.0,
    interpolation_factor=1.0,
):
    """
    This is a n-d version of precompute_freqs_cis, which is a RoPE for tokens with n-d structure.

    Args:
        rope_dim_list (list of int): Dimension of each rope. len(rope_dim_list) should equal to n.
            sum(rope_dim_list) should equal to head_dim of attention layer.
        start (int | tuple of int | list of int): If len(args) == 0, start is num; If len(args) == 1, start is start,
            args[0] is stop, step is 1; If len(args) == 2, start is start, args[0] is stop, args[1] is num.
        *args: See above.
        theta (float): Scaling factor for frequency computation. Defaults to 10000.0.
        use_real (bool): If True, return real part and imaginary part separately. Otherwise, return complex numbers.
            Some libraries such as TensorRT does not support complex64 data type. So it is useful to provide a real
            part and an imaginary part separately.
        theta_rescale_factor (float): Rescale factor for theta. Defaults to 1.0.

    Returns:
        pos_embed (torch.Tensor): [HW, D/2]
    """

    grid = get_meshgrid_nd(
        start, *args, dim=len(rope_dim_list)
    )  # [3, W, H, D] / [2, W, H]

    if isinstance(theta_rescale_factor, int) or isinstance(theta_rescale_factor, float):
        theta_rescale_factor = [theta_rescale_factor] * len(rope_dim_list)
    elif isinstance(theta_rescale_factor, list) and len(theta_rescale_factor) == 1:
        theta_rescale_factor = [theta_rescale_factor[0]] * len(rope_dim_list)
    assert len(theta_rescale_factor) == len(
        rope_dim_list
    ), "len(theta_rescale_factor) should equal to len(rope_dim_list)"

    if isinstance(interpolation_factor, int) or isinstance(interpolation_factor, float):
        interpolation_factor = [interpolation_factor] * len(rope_dim_list)
    elif isinstance(interpolation_factor, list) and len(interpolation_factor) == 1:
        interpolation_factor = [interpolation_factor[0]] * len(rope_dim_list)
    assert len(interpolation_factor) == len(
        rope_dim_list
    ), "len(interpolation_factor) should equal to len(rope_dim_list)"

    # use 1/ndim of dimensions to encode grid_axis
    embs = []
    for i in range(len(rope_dim_list)):
        emb = get_1d_rotary_pos_embed(
            rope_dim_list[i],
            grid[i].reshape(-1),
            theta,
            use_real=use_real,
            theta_rescale_factor=theta_rescale_factor[i],
            interpolation_factor=interpolation_factor[i],
        )  # 2 x [WHD, rope_dim_list[i]]
        embs.append(emb)

    if use_real:
        cos = jnp.concatenate([emb[0] for emb in embs], axis=1)  # (WHD, D/2)
        sin = jnp.concatenate([emb[1] for emb in embs], axis=1)  # (WHD, D/2)
        return cos, sin
    else:
        emb = jnp.concatenate(embs, axis=1)  # (WHD, D/2)
        return emb


def get_1d_rotary_pos_embed(
    dim,
    pos,
    theta=10000.0,
    use_real=False,
    theta_rescale_factor=1.0,
    interpolation_factor=1.0,
):
    """
    Precompute the frequency tensor for complex exponential (cis) with given dimensions.
    (Note: `cis` means `cos + i * sin`, where i is the imaginary unit.)

    This function calculates a frequency tensor with complex exponential using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency tensor.
        pos (int or torch.FloatTensor): Position indices for the frequency tensor. [S] or scalar
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.
        use_real (bool, optional): If True, return real part and imaginary part separately.
                                   Otherwise, return complex numbers.
        theta_rescale_factor (float, optional): Rescale factor for theta. Defaults to 1.0.

    Returns:
        freqs_cis: Precomputed frequency tensor with complex exponential. [S, D/2]
        freqs_cos, freqs_sin: Precomputed frequency tensor with real and imaginary parts separately. [S, D]
    """
    if isinstance(pos, int):
        pos = jnp.arange(pos).astype(jnp.float32)

    # proposed by reddit user bloc97, to rescale rotary embeddings to longer sequence length without fine-tuning
    # has some connection to NTK literature
    if theta_rescale_factor != 1.0:
        theta *= theta_rescale_factor ** (dim / (dim - 2))

    freqs = 1.0 / (
        theta ** (jnp.arange(0, dim, 2, dtype=jnp.float32)[: (dim // 2)] / dim)
    )  # [D/2]
    # assert interpolation_factor == 1.0, f"interpolation_factor: {interpolation_factor}"
    freqs = jnp.outer(pos * interpolation_factor, freqs)  # [S, D/2]
    if use_real:
        freqs_cos = jnp.repeat(jnp.cos(freqs), 2, axis=1)  # [S, D]
        freqs_sin = jnp.repeat(jnp.sin(freqs), 2, axis=1)  # [S, D]
        return freqs_cos, freqs_sin
    else:
        freqs_cis = jnp.exp(1j * freqs)
        return freqs_cis
