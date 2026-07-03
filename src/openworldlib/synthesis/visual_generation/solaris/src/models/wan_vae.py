import jax
import jax.numpy as jnp
from einops import rearrange
from flax import nnx

CACHE_T = 2
VAE_MEAN = jnp.array(
    [
        -0.7571,
        -0.7089,
        -0.9113,
        0.1075,
        -0.1745,
        0.9653,
        -0.1517,
        1.5508,
        0.4134,
        -0.0715,
        0.5517,
        -0.3632,
        -0.1922,
        -0.9497,
        0.2503,
        -0.2921,
    ]
)
VAE_STD = jnp.array(
    [
        2.8184,
        1.4541,
        2.3275,
        2.6558,
        1.2196,
        1.7708,
        2.6052,
        2.0743,
        3.2687,
        2.1526,
        2.8652,
        1.5579,
        1.6382,
        1.1253,
        2.8251,
        1.9160,
    ]
)


def get_cache(model):
    conv_num = count_conv3d(model)
    return [None] * conv_num


VAE_SCALE = [VAE_MEAN, 1.0 / VAE_STD]


class CausalConv3d(nnx.Conv):
    def __init__(self, *args, **kwargs):
        orig_padding = kwargs.get("padding", 0)
        super().__init__(*args, **kwargs)
        if isinstance(orig_padding, tuple):
            pad = orig_padding
        elif isinstance(orig_padding, int):
            pad = (orig_padding, orig_padding, orig_padding)
        else:
            pad = (0, 0, 0)  # Default to no padding

        self._padding = (
            (0, 0),
            (2 * pad[0], 0) if pad[0] > 0 else (0, 0),
            (pad[1], pad[1]),
            (pad[2], pad[2]),
            (0, 0),
        )
        self.padding = 0

    def __call__(self, x, cache_x=None, time_padding=None):
        padding = list(self._padding)

        if cache_x is not None and self._padding[1][0] > 0:
            x = jnp.concatenate([cache_x, x], axis=1)
            padding[1] = (padding[1][0] - cache_x.shape[1], padding[1][1])
        if time_padding is not None:
            padding[1] = time_padding

        x = jnp.pad(x, padding)
        x = super().__call__(x)
        return x


class RMS_norm(nnx.Module):
    def __init__(self, dim, images=True, bias=False):
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims)
        self.scale = dim**0.5
        self.images = images
        self.gamma = nnx.Param(
            jnp.ones(shape), dtype=jnp.bfloat16, param_dtype=jnp.bfloat16
        )
        self.bias = (
            nnx.Param(jnp.zeros(shape), dtype=jnp.bfloat16, param_dtype=jnp.bfloat16)
            if bias
            else 0.0
        )

    def __call__(self, x):
        original_dtype = x.dtype
        x_fp32 = x.astype(jnp.bfloat16)
        norm = jnp.linalg.norm(x_fp32, axis=-1, keepdims=True)
        normalized = x_fp32 / jnp.maximum(norm, 1e-12)
        normalized = normalized.astype(original_dtype)

        if x.ndim == 5:
            fixed_gamma = jnp.expand_dims(self.gamma.reshape(-1), axis=(0, 1, 2, 3))
        else:
            fixed_gamma = jnp.expand_dims(self.gamma.reshape(-1), axis=(0, 1, 2))

        if not isinstance(self.bias, float):
            fixed_bias = jnp.expand_dims(self.bias.reshape(-1), axis=(0, 1, 2, 3))
        else:
            fixed_bias = 0.0
        return normalized * self.scale * fixed_gamma + fixed_bias


class ResidualBlock(nnx.Module):
    def __init__(self, in_dim, out_dim, rngs, dropout=0.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.residual = [
            RMS_norm(in_dim, images=False),
            nnx.silu,
            CausalConv3d(
                in_dim,
                out_dim,
                (3, 3, 3),
                padding=(1, 1, 1),
                rngs=rngs,
                dtype=jnp.bfloat16,
                param_dtype=jnp.bfloat16,
            ),
            RMS_norm(out_dim, images=False),
            nnx.silu,
            nnx.Dropout(dropout, rngs=rngs),
            CausalConv3d(
                out_dim,
                out_dim,
                (3, 3, 3),
                padding=(1, 1, 1),
                rngs=rngs,
                dtype=jnp.bfloat16,
                param_dtype=jnp.bfloat16,
            ),
        ]

        self.shortcut = (
            CausalConv3d(
                in_dim,
                out_dim,
                (1, 1, 1),
                padding=(0, 0, 0),
                rngs=rngs,
                dtype=jnp.bfloat16,
                param_dtype=jnp.bfloat16,
            )
            if in_dim != out_dim
            else None
        )

    def __call__(
        self,
        x,
        feat_cache=None,
        new_cache=None,
        feat_idx=[0],
    ):
        h = self.shortcut(x) if self.shortcut is not None else x

        for ix, layer in enumerate(self.residual):
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, -CACHE_T:, :, :, :]
                if cache_x.shape[1] < 2 and feat_cache[idx] is not None:
                    # Cache last frame of last two chunks
                    last_frame = jnp.expand_dims(
                        feat_cache[idx][:, -1, :, :, :], axis=1
                    )
                    cache_x = jnp.concatenate([last_frame, cache_x], axis=1)

                x = layer(x, feat_cache[idx])
                new_cache[idx] = cache_x
                feat_idx[0] += 1
            elif isinstance(layer, nnx.Module):
                # For nnx Modules (RMS_norm, Dropout)
                x = layer(x)
            else:
                # For functions (nnx.silu)
                x = layer(x)

        return x + h


class Upsample(nnx.Module):
    def __init__(self, scale_factor=(2.0, 2.0)):
        super().__init__()
        self.scale_factor = scale_factor

    def __call__(self, x):
        # x is in format N, H, W, C for JAX
        b, h, w, c = x.shape
        new_h = int(h * self.scale_factor[0])
        new_w = int(w * self.scale_factor[1])
        return jax.image.resize(x, (b, new_h, new_w, c), method="nearest")


class Resample(nnx.Module):
    def __init__(self, dim, mode, rngs):
        assert mode in (
            "none",
            "upsample2d",
            "upsample3d",
            "downsample2d",
            "downsample3d",
        )
        super().__init__()
        self.dim = dim
        self.mode = mode

        # layers
        if mode == "upsample2d":
            self.upsample = Upsample(scale_factor=(2.0, 2.0))
            self.conv = nnx.Conv(
                dim,
                dim // 2,
                (3, 3),
                padding=1,
                rngs=rngs,
                dtype=jnp.bfloat16,
                param_dtype=jnp.bfloat16,
            )
        elif mode == "upsample3d":
            self.upsample = Upsample(scale_factor=(2.0, 2.0))
            self.conv = nnx.Conv(dim, dim // 2, (3, 3), padding=1, rngs=rngs)
            self.time_conv = CausalConv3d(
                dim,
                dim * 2,
                (3, 1, 1),
                padding=(1, 0, 0),
                rngs=rngs,
                dtype=jnp.bfloat16,
                param_dtype=jnp.bfloat16,
            )
        elif mode == "downsample2d":
            self.pad = lambda x: jnp.pad(x, ((0, 0), (0, 1), (0, 1), (0, 0)))
            self.conv = nnx.Conv(
                dim,
                dim,
                (3, 3),
                strides=(2, 2),
                padding=0,
                rngs=rngs,
                dtype=jnp.bfloat16,
                param_dtype=jnp.bfloat16,
            )
        elif mode == "downsample3d":
            self.pad = lambda x: jnp.pad(x, ((0, 0), (0, 1), (0, 1), (0, 0)))
            self.conv = nnx.Conv(
                dim,
                dim,
                (3, 3),
                strides=(2, 2),
                padding=0,
                rngs=rngs,
                dtype=jnp.bfloat16,
                param_dtype=jnp.bfloat16,
            )
            self.time_conv = CausalConv3d(
                dim,
                dim,
                (3, 1, 1),
                strides=(2, 1, 1),
                padding=(0, 0, 0),
                rngs=rngs,
                dtype=jnp.bfloat16,
                param_dtype=jnp.bfloat16,
            )
        else:
            self.resample = None

    def __call__(self, x, feat_cache=None, new_cache=None, feat_idx=[0]):
        b, f, h, w, c = x.shape
        if self.mode == "upsample3d":
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    b, f, h, w, c = x.shape
                    new_cache[idx] = jnp.zeros((b, CACHE_T, h, w, c))
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, -CACHE_T:, :, :, :]
                    cache_x = jnp.concatenate([feat_cache[idx], cache_x], axis=1)[
                        :, -2:, :, :, :
                    ]

                    if feat_cache[idx] == "Rep":
                        x = self.time_conv(x)
                    else:
                        x = self.time_conv(x, feat_cache[idx])

                    new_cache[idx] = cache_x
                    feat_idx[0] += 1
                    x = rearrange(x, "b t h w (r c) -> b t h w r c", r=2)
                    x = rearrange(x, "b t h w r c -> b (t r) h w c")

        t = x.shape[1]
        x = rearrange(x, "b t h w c -> (b t) h w c")

        if self.mode in ["upsample2d", "upsample3d"]:
            x = self.upsample(x)
            x = self.conv(x)
        elif self.mode in ["downsample2d", "downsample3d"]:
            x = self.pad(x)
            x = self.conv(x)

        x = rearrange(x, "(b t) h w c -> b t h w c", t=t)

        if self.mode == "downsample3d":
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    new_cache[idx] = x
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, -1:, :, :, :]
                    last_frame = feat_cache[idx][:, -1:, :, :, :]
                    x_concat = jnp.concatenate([last_frame, x], axis=1)
                    x = self.time_conv(x_concat)
                    new_cache[idx] = cache_x
                    feat_idx[0] += 1
            else:
                first_frame = x[:, :1, :, :, :]
                x = self.time_conv(
                    x, time_padding=(2, 0)
                )  # do the convolution but but let the first frame through.
                x = jnp.concatenate([first_frame, x[:, 1:, :, :, :]], axis=1)
        return x


class AttentionBlock(nnx.Module):
    """
    Causal self-attention with a single head.
    """

    def __init__(self, dim, rngs):
        super().__init__()
        self.dim = dim

        # layers
        self.norm = RMS_norm(dim)
        self.to_qkv = nnx.Conv(
            dim,
            dim * 3,
            (1, 1),
            rngs=rngs,
            dtype=jnp.bfloat16,
            param_dtype=jnp.bfloat16,
        )
        self.proj = nnx.Conv(
            dim, dim, (1, 1), rngs=rngs, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16
        )

        # zero out the last layer params
        self.proj.kernel.value = jnp.zeros_like(
            self.proj.kernel.value, dtype=jnp.bfloat16
        )

    def __call__(self, x):
        identity = x
        b, t, h, w, c = x.shape
        x = rearrange(x, "b t h w c -> (b t) h w c")
        x = self.norm(x)
        qkv = self.to_qkv(x)
        qkv = rearrange(qkv, "b h w (c nh)-> b (h w) nh c", nh=1, h=h)
        q, k, v = jnp.split(qkv, 3, axis=-1)

        x = jax.nn.dot_product_attention(
            q,
            k,
            v,
            scale=1 / jnp.sqrt(c),
        )

        x = rearrange(x, "b (h w) nh c -> b h w (c nh)", h=h, w=w)
        # b (h w) nh c we want to now convolve over it
        x = self.proj(x)
        x = rearrange(x, "(b t) h w c -> b t h w c", t=t)
        return x + identity


def count_conv3d(module):
    """Count the number of CausalConv3d layers in a module"""
    count = 0
    if isinstance(module, list):
        for layer in module:
            if isinstance(layer, CausalConv3d):
                count += 1
            elif isinstance(
                layer, (ResidualBlock, Resample, AttentionBlock, Encoder3d, Decoder3d)
            ):
                count += count_conv3d(layer)
            elif hasattr(layer, "__iter__"):
                count += count_conv3d(layer)
    elif hasattr(module, "__dict__"):
        for attr_name in dir(module):
            if not attr_name.startswith("_"):
                attr = getattr(module, attr_name)
                if isinstance(attr, CausalConv3d):
                    count += 1
                elif isinstance(
                    attr,
                    (ResidualBlock, Resample, AttentionBlock, Encoder3d, Decoder3d),
                ):
                    count += count_conv3d(attr)
                elif isinstance(attr, list):
                    count += count_conv3d(attr)
    return count


class WanVAE_(nnx.Module):
    def __init__(
        self,
        rngs,
        dim=128,
        z_dim=4,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[True, True, False],
        dropout=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample
        self.temperal_upsample = temperal_downsample[::-1]

        # modules
        self.encoder = Encoder3d(
            rngs,
            dim,
            z_dim * 2,
            dim_mult,
            num_res_blocks,
            attn_scales,
            self.temperal_downsample,
            dropout,
        )
        self.conv1 = CausalConv3d(
            z_dim * 2,
            z_dim * 2,
            (1, 1, 1),
            padding=(0, 0, 0),
            rngs=rngs,
            dtype=jnp.bfloat16,
            param_dtype=jnp.bfloat16,
        )
        self.conv2 = CausalConv3d(
            z_dim,
            z_dim,
            (1, 1, 1),
            padding=(0, 0, 0),
            rngs=rngs,
            dtype=jnp.bfloat16,
            param_dtype=jnp.bfloat16,
        )
        self.decoder = Decoder3d(
            rngs,
            dim,
            z_dim,
            dim_mult,
            num_res_blocks,
            attn_scales,
            self.temperal_upsample,
            dropout,
        )

        self._conv_num = count_conv3d(self.decoder)
        self._enc_conv_num = count_conv3d(self.encoder)

    def encode_frame(self, x, cache, scale=VAE_SCALE):
        out, cache = self.encoder(x, feat_cache=cache)
        conv1_out = self.conv1(out)
        mu, log_var = jnp.split(conv1_out, 2, axis=-1)
        if isinstance(scale[0], jnp.ndarray) and scale[0].size > 1:
            scale0 = jnp.expand_dims(scale[0], axis=(0, 1, 2, 3))
            scale1 = jnp.expand_dims(scale[1], axis=(0, 1, 2, 3))
            mu = (mu - scale0) * scale1
        else:
            mu = (mu - scale[0]) * scale[1]
        return mu, cache

    def decode_frame(self, z, cache, scale=VAE_SCALE):
        if isinstance(scale[0], jnp.ndarray) and scale[0].size > 1:
            scale0 = jnp.expand_dims(scale[0], axis=(0, 1, 2, 3))
            scale1 = jnp.expand_dims(scale[1], axis=(0, 1, 2, 3))
            z = z / scale1 + scale0
        else:
            z = z / scale[1] + scale[0]
        x = self.conv2(z)
        out, cache = self.decoder(
            x,
            feat_cache=cache,
        )
        return out, cache

    # x: B T H W C
    def cacheless_encode(self, x, scale):
        out, _ = self.encoder(x)
        conv1_out = self.conv1(out)
        mu, _ = jnp.split(conv1_out, 2, axis=-1)
        if isinstance(scale[0], jnp.ndarray) and scale[0].size > 1:
            scale0 = jnp.expand_dims(scale[0], axis=(0, 1, 2, 3))
            scale1 = jnp.expand_dims(scale[1], axis=(0, 1, 2, 3))
            mu = (mu - scale0) * scale1
        else:
            mu = (mu - scale[0]) * scale[1]
        return mu

    # x: B T H W C
    def encode(self, x, scale):
        t = x.shape[1]
        iter_ = 1 + (t - 1) // 4
        cache = get_cache(self.encoder)
        ## Split input x by time: 1, 4, 4, 4....

        out0, cache = self.encoder(
            x[:, :1, :, :, :],
            feat_cache=cache,
        )
        cache = _pad_cache_time_to(cache, T=CACHE_T)
        out = jnp.zeros((out0.shape[0], iter_) + out0.shape[2:], dtype=out0.dtype)
        out = out.at[:, :1, :, :, :].set(out0)

        # Change second dimension of out to be iter_ length
        # after generating first frame, pad to iter_ on time dimension if needed
        # Pad out along time dimension to length iter_, with zeros on the right
        def encode_chunk(i, res):
            out, cache = res
            chunk = jax.lax.dynamic_slice_in_dim(
                x, start_index=1 + 4 * (i - 1), slice_size=4, axis=1
            )
            out_, cache = self.encoder(
                chunk,
                feat_cache=cache,
            )
            cache = _pad_cache_time_to(cache, T=CACHE_T)
            out = jax.lax.dynamic_update_slice_in_dim(out, out_, i, axis=1)
            return out, cache

        out, _ = jax.lax.fori_loop(1, iter_, encode_chunk, (out, cache))
        time_major = rearrange(out, "b t h w c -> t b h w c")

        def _per_frame_conv1(frame_bhwc):
            return self.conv1(frame_bhwc[:, None, ...])[:, 0, ...]

        ys = jax.lax.map(_per_frame_conv1, time_major)
        conv1_out = rearrange(ys, "t b h w c -> b t h w c")
        mu, log_var = jnp.split(conv1_out, 2, axis=-1)

        if isinstance(scale[0], jnp.ndarray) and scale[0].size > 1:
            scale0 = jnp.expand_dims(scale[0], axis=(0, 1, 2, 3))
            scale1 = jnp.expand_dims(scale[1], axis=(0, 1, 2, 3))
            mu = (mu - scale0) * scale1
        else:
            mu = (mu - scale[0]) * scale[1]
        return mu

    def decode(self, z, scale):
        cache = get_cache(self.decoder)
        if isinstance(scale[0], jnp.ndarray) and scale[0].size > 1:
            scale0 = jnp.expand_dims(scale[0], axis=(0, 1, 2, 3))
            scale1 = jnp.expand_dims(scale[1], axis=(0, 1, 2, 3))
            z = z / scale1 + scale0
        else:
            z = z / scale[1] + scale[0]
        iter_ = z.shape[1]
        time_major = rearrange(z, "b t h w c -> t b h w c")

        def _per_frame_conv2(frame_bhwc):
            return self.conv2(frame_bhwc[:, None, ...])[:, 0, ...]

        xs = jax.lax.map(_per_frame_conv2, time_major)
        x = rearrange(xs, "t b h w c -> b t h w c")
        out0, cache = self.decoder(
            x[:, :1, :, :, :],
            feat_cache=cache,
        )
        cache = _pad_cache_time_to(cache, T=CACHE_T)
        out = jnp.zeros(
            (out0.shape[0], (iter_ - 1) * 4 + 1) + out0.shape[2:], dtype=out0.dtype
        )
        out = out.at[:, :1, :, :, :].set(out0)

        def decode_chunk(i, res):
            out, cache = res
            chunk = jax.lax.dynamic_slice_in_dim(x, start_index=i, slice_size=1, axis=1)
            out_, cache = self.decoder(
                chunk,
                feat_cache=cache,
            )
            cache = _pad_cache_time_to(cache, T=CACHE_T)
            out = jax.lax.dynamic_update_slice_in_dim(
                out, out_, 1 + 4 * (i - 1), axis=1
            )
            return out, cache

        out, _ = jax.lax.fori_loop(1, iter_, decode_chunk, (out, cache))
        return out

    def sample(self, imgs, deterministic=False):
        mu, log_var = self.encode(imgs, scale=[0, 1])
        if deterministic:
            return mu
        std = jnp.exp(0.5 * jnp.clip(log_var, -30.0, 20.0))
        key = jax.random.PRNGKey(0)
        eps = jax.random.normal(key, std.shape)
        return mu + std * eps

    def get_encoder_cache(self):
        _enc_conv_idx = [0]
        _enc_feat_map = [None] * self._enc_conv_num
        return _enc_feat_map, _enc_conv_idx

    def get_decoder_cache(self):
        _conv_idx = [0]
        _feat_map = [None] * self._conv_num
        return _feat_map, _conv_idx


def _pad_cache_time_to(cache_list, T=2):
    def pad_one(a):
        # a: (b, t, h, w, c)
        if a is None:
            return (
                None  # keep structure; this entry will be set by encoder on next call
            )
        t = a.shape[1]
        if t == T:
            return a
        # left-pad zeros to reach T
        pad = T - t
        zeros = jnp.zeros((a.shape[0], pad, *a.shape[2:]), dtype=a.dtype)
        return jnp.concatenate([zeros, a], axis=1)

    return [pad_one(a) for a in cache_list]


class Encoder3d(nnx.Module):
    def __init__(
        self,
        rngs,
        dim=128,
        z_dim=4,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[True, True, False],
        dropout=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample

        dims = [dim * u for u in [1] + dim_mult]

        scale = 1.0
        # init block
        self.conv1 = CausalConv3d(
            3,
            dims[0],
            (3, 3, 3),
            padding=(1, 1, 1),
            rngs=rngs,
            dtype=jnp.bfloat16,
            param_dtype=jnp.bfloat16,
        )

        # downsample blocks
        self.downsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # residual (+attention) blocks
            for _ in range(num_res_blocks):
                self.downsamples.append(ResidualBlock(in_dim, out_dim, rngs, dropout))
                if scale in attn_scales:
                    self.downsamples.append(AttentionBlock(out_dim, rngs))
                in_dim = out_dim

            # downsample block
            if i != len(dim_mult) - 1:
                #
                mode = "downsample3d" if temperal_downsample[i] else "downsample2d"
                self.downsamples.append(Resample(out_dim, mode=mode, rngs=rngs))
                scale /= 2.0

        # middle blocks
        self.middle = [
            ResidualBlock(out_dim, out_dim, rngs, dropout),
            AttentionBlock(out_dim, rngs),
            ResidualBlock(out_dim, out_dim, rngs, dropout),
        ]

        # output blocks
        self.head = [
            RMS_norm(out_dim, images=False),
            nnx.silu,
            CausalConv3d(out_dim, z_dim, (3, 3, 3), padding=(1, 1, 1), rngs=rngs),
        ]

    def __call__(self, x, feat_cache=None):
        new_cache = [None] * len(feat_cache) if feat_cache else None
        feat_idx = [0]

        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, -CACHE_T:, :, :, :]
            if cache_x.shape[1] < 2 and feat_cache[idx] is not None:
                # cache last frame of last two chunk
                last_frame = jnp.expand_dims(feat_cache[idx][:, -1, :, :, :], axis=1)
                cache_x = jnp.concatenate([last_frame, cache_x], axis=1)
            x = self.conv1(x, feat_cache[idx])

            new_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        #
        # phase 1: median abs difference: 0
        #

        ## downsamples
        for ix, layer in enumerate(self.downsamples):
            if feat_cache is not None:
                x = layer(x, feat_cache, new_cache, feat_idx)
            else:
                x = layer(x)

        ## middle
        for layer in self.middle:
            if isinstance(layer, ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, new_cache, feat_idx)
            else:
                x = layer(x)

        ## head
        for layer in self.head:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, -CACHE_T:, :, :, :]
                if cache_x.shape[1] < 2 and feat_cache[idx] is not None:
                    # cache last frame of last two chunk
                    last_frame = jnp.expand_dims(
                        feat_cache[idx][:, -1, :, :, :], axis=1
                    )
                    cache_x = jnp.concatenate([last_frame, cache_x], axis=1)
                x = layer(x, feat_cache[idx])
                new_cache[idx] = cache_x
                feat_idx[0] += 1
            elif callable(layer):
                x = layer(x)
            else:
                x = layer(x)
        return x, new_cache


class Decoder3d(nnx.Module):
    def __init__(
        self,
        rngs,
        dim=128,
        z_dim=4,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_upsample=[False, True, True],
        dropout=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_upsample = temperal_upsample

        # dimensions
        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        scale = 1.0 / 2 ** (len(dim_mult) - 2)

        # init block
        self.conv1 = CausalConv3d(
            z_dim, dims[0], (3, 3, 3), padding=(1, 1, 1), rngs=rngs
        )

        # middle blocks
        self.middle = [
            ResidualBlock(dims[0], dims[0], rngs, dropout),
            AttentionBlock(dims[0], rngs),
            ResidualBlock(dims[0], dims[0], rngs, dropout),
        ]

        # upsample blocks
        self.upsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # residual (+attention) blocks
            if i == 1 or i == 2 or i == 3:
                in_dim = in_dim // 2
            for _ in range(num_res_blocks + 1):
                self.upsamples.append(ResidualBlock(in_dim, out_dim, rngs, dropout))
                if scale in attn_scales:
                    self.upsamples.append(AttentionBlock(out_dim, rngs))
                in_dim = out_dim

            # upsample block
            if i != len(dim_mult) - 1:
                mode = "upsample3d" if temperal_upsample[i] else "upsample2d"
                self.upsamples.append(Resample(out_dim, mode=mode, rngs=rngs))
                scale *= 2.0

        # output blocks
        self.head = [
            RMS_norm(out_dim, images=False),
            nnx.silu,
            CausalConv3d(out_dim, 3, (3, 3, 3), padding=(1, 1, 1), rngs=rngs),
        ]

    def __call__(self, x, feat_cache=None):
        feat_idx = [0]
        new_cache = [None] * len(feat_cache)

        ## conv1
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, -CACHE_T:, :, :, :]
            if cache_x.shape[1] < 2 and feat_cache[idx] is not None:
                # cache last frame of last two chunk
                last_frame = jnp.expand_dims(feat_cache[idx][:, -1, :, :, :], axis=1)
                cache_x = jnp.concatenate([last_frame, cache_x], axis=1)
            x = self.conv1(x, feat_cache[idx])
            new_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        ## middle
        for layer in self.middle:
            if isinstance(layer, ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, new_cache, feat_idx)
            else:
                x = layer(x)

        ## upsamples
        for layer in self.upsamples:
            if feat_cache is not None:
                x = layer(x, feat_cache, new_cache, feat_idx)
            else:
                x = layer(x)

        ## head
        for i, layer in enumerate(self.head):
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, -CACHE_T:, :, :, :]
                if cache_x.shape[1] < 2 and feat_cache[idx] is not None:
                    # cache last frame of last two chunk
                    last_frame = jnp.expand_dims(
                        feat_cache[idx][:, -1, :, :, :], axis=1
                    )
                    cache_x = jnp.concatenate([last_frame, cache_x], axis=1)
                x = layer(x, feat_cache[idx])
                new_cache[idx] = cache_x
                feat_idx[0] += 1
            elif callable(layer):
                x = layer(x)
            else:
                x = layer(x)
        return x, new_cache
