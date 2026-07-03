"""JAX implementation of CLIP model."""

import math

import jax
import jax.numpy as jnp
from einops import rearrange
from flax import nnx


class QuickGELU(nnx.Module):
    """Quick GELU activation function."""

    def __call__(self, x):
        return x * jax.nn.sigmoid(1.702 * x)


class LayerNorm(nnx.Module):
    """Layer normalization with float32 computation."""

    def __init__(self, dim, eps=1e-5):
        self.dim = dim
        self.eps = eps
        self.scale = nnx.Param(jnp.ones(dim, dtype=jnp.float32))
        self.bias = nnx.Param(jnp.zeros(dim, dtype=jnp.float32))

    def __call__(self, x):
        # Compute in float32 for stability
        x_float = x.astype(jnp.float32)
        mean = jnp.mean(x_float, axis=-1, keepdims=True)
        var = jnp.var(x_float, axis=-1, keepdims=True)
        normalized = (x_float - mean) / jnp.sqrt(var + self.eps)
        output = normalized * self.scale.value + self.bias.value
        return output.astype(x.dtype)


class SelfAttention(nnx.Module):
    """Multi-head self-attention."""

    def __init__(
        self,
        dim,
        num_heads,
        causal=False,
        attn_dropout=0.0,
        proj_dropout=0.0,
    ):
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.causal = causal
        self.attn_dropout = attn_dropout
        self.proj_dropout = proj_dropout
        # Layers
        self.to_qkv = nnx.Linear(dim, dim * 3, use_bias=True, rngs=nnx.Rngs(0))
        self.proj = nnx.Linear(dim, dim, use_bias=True, rngs=nnx.Rngs(0))

    def __call__(self, x):
        b, s, c = x.shape
        n, d = self.num_heads, self.head_dim
        # Compute query, key, value
        qkv = self.to_qkv(x).reshape(b, s, 3, n, d)
        q, k, v = jnp.split(qkv, 3, axis=2)
        q = q.squeeze(2)
        k = k.squeeze(2)
        v = v.squeeze(2)
        x = jax.nn.dot_product_attention(
            q,
            k,
            v,
            scale=1 / jnp.sqrt(d),
            is_causal=self.causal,
            implementation="xla",
        )
        x = rearrange(x, "b s n d -> b s (n d)")
        x = self.proj(x)
        return x


class AttentionBlock(nnx.Module):
    """Transformer attention block."""

    def __init__(
        self,
        dim,
        mlp_ratio,
        num_heads,
        post_norm=False,
        causal=False,
        activation="quick_gelu",
        attn_dropout=0.0,
        proj_dropout=0.0,
        norm_eps=1e-5,
    ):
        assert activation in ["quick_gelu", "gelu"]
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.num_heads = num_heads
        self.post_norm = post_norm
        self.causal = causal
        self.norm_eps = norm_eps

        # Layers
        self.norm1 = LayerNorm(dim, eps=norm_eps)
        self.attn = SelfAttention(dim, num_heads, causal, attn_dropout, proj_dropout)
        self.norm2 = LayerNorm(dim, eps=norm_eps)

        # MLP
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nnx.Sequential(
            nnx.Linear(dim, mlp_dim, use_bias=True, rngs=nnx.Rngs(0)),
            QuickGELU() if activation == "quick_gelu" else nnx.gelu,
            nnx.Linear(mlp_dim, dim, use_bias=True, rngs=nnx.Rngs(0)),
        )

    def __call__(self, x):
        if self.post_norm:
            x = x + self.norm1(self.attn(x))
            x = x + self.norm2(self.mlp(x))
        else:
            x = x + self.attn(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
        return x


class AttentionPool(nnx.Module):
    """Attention pooling layer."""

    def __init__(
        self,
        dim,
        mlp_ratio,
        num_heads,
        activation="gelu",
        proj_dropout=0.0,
        norm_eps=1e-5,
    ):
        assert dim % num_heads == 0
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.proj_dropout = proj_dropout
        self.norm_eps = norm_eps

        # Layers
        gain = 1.0 / math.sqrt(dim)
        self.cls_embedding = nnx.Param(
            gain
            * jax.random.normal(jax.random.PRNGKey(0), (1, 1, dim), dtype=jnp.float32)
        )
        self.to_q = nnx.Linear(dim, dim, use_bias=True, rngs=nnx.Rngs(0))
        self.to_kv = nnx.Linear(dim, dim * 2, use_bias=True, rngs=nnx.Rngs(0))
        self.proj = nnx.Linear(dim, dim, use_bias=True, rngs=nnx.Rngs(0))
        self.norm = LayerNorm(dim, eps=norm_eps)

        # MLP
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nnx.Sequential(
            nnx.Linear(dim, mlp_dim, use_bias=True, rngs=nnx.Rngs(0)),
            QuickGELU() if activation == "quick_gelu" else nnx.gelu,
            nnx.Linear(mlp_dim, dim, use_bias=True, rngs=nnx.Rngs(0)),
        )

    def __call__(self, x):
        b, s, c = x.shape
        n, d = self.num_heads, self.head_dim

        # Compute query from cls_embedding
        q = self.to_q(self.cls_embedding.value).reshape(1, 1, n, d)
        q = jnp.broadcast_to(q, (b, 1, n, d))

        # Compute key and value from input
        kv = self.to_kv(x).reshape(b, s, 2, n, d)
        k, v = jnp.split(kv, 2, axis=2)
        k = k.squeeze(2)
        v = v.squeeze(2)

        # Compute attention scores
        scores = jnp.einsum("bqnd,bsnd->bnqs", q, k) / jnp.sqrt(d)
        attn = jax.nn.softmax(scores, axis=-1)

        # Apply attention to values
        x = jnp.einsum("bnqs,bsnd->bqnd", attn, v)
        x = x.reshape(b, 1, c)

        # Output projection
        x = self.proj(x)

        # MLP
        x = x + self.mlp(self.norm(x))
        return x[:, 0]  # Return only the pooled output


class VisionTransformer(nnx.Module):
    """Vision Transformer model."""

    def __init__(
        self,
        image_size=224,
        patch_size=16,
        dim=768,
        mlp_ratio=4,
        out_dim=512,
        num_heads=12,
        num_layers=12,
        pool_type="token",
        pre_norm=True,
        post_norm=False,
        activation="quick_gelu",
        attn_dropout=0.0,
        proj_dropout=0.0,
        embedding_dropout=0.0,
        norm_eps=1e-5,
    ):
        assert pool_type in ("token", "token_fc", "attn_pool")
        out_dim = out_dim or dim

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.pool_type = pool_type
        self.post_norm = post_norm
        self.norm_eps = norm_eps

        # Embeddings
        gain = 1.0 / math.sqrt(dim)
        self.patch_embedding = nnx.Conv(
            3,
            dim,
            kernel_size=(patch_size, patch_size),
            strides=(patch_size, patch_size),
            use_bias=not pre_norm,
            rngs=nnx.Rngs(0),
        )

        if pool_type in ("token", "token_fc"):
            self.cls_embedding = nnx.Param(
                gain
                * jax.random.normal(
                    jax.random.PRNGKey(1), (1, 1, dim), dtype=jnp.float32
                )
            )

        pos_embed_len = self.num_patches + (
            1 if pool_type in ("token", "token_fc") else 0
        )
        self.pos_embedding = nnx.Param(
            gain
            * jax.random.normal(
                jax.random.PRNGKey(2), (1, pos_embed_len, dim), dtype=jnp.float32
            )
        )

        # Note: Dropout is not implemented in this simplified version

        # Transformer
        self.pre_norm = LayerNorm(dim, eps=norm_eps) if pre_norm else None
        self.transformer = [
            AttentionBlock(
                dim,
                mlp_ratio,
                num_heads,
                post_norm,
                False,
                activation,
                attn_dropout,
                proj_dropout,
                norm_eps,
            )
            for _ in range(num_layers)
        ]
        self.post_norm = LayerNorm(dim, eps=norm_eps)

        # Head
        if pool_type == "token":
            self.head = nnx.Param(
                gain
                * jax.random.normal(
                    jax.random.PRNGKey(3), (dim, out_dim), dtype=jnp.float32
                )
            )
        elif pool_type == "token_fc":
            self.head = nnx.Linear(dim, out_dim, use_bias=True, rngs=nnx.Rngs(0))
        elif pool_type == "attn_pool":
            self.head = AttentionPool(
                dim, mlp_ratio, num_heads, activation, proj_dropout, norm_eps
            )

    def __call__(self, x, interpolation=False, use_31_block=False):
        b = x.shape[0]

        # Patch embedding
        x = rearrange(
            self.patch_embedding(rearrange(x, "b c h w -> b h w c")),
            "b h w c -> b c h w",
        )
        x = rearrange(x, "b c h w -> b (h w) c")

        # Add class token
        if self.pool_type in ("token", "token_fc"):
            cls_token = jnp.broadcast_to(self.cls_embedding.value, (b, 1, self.dim))
            x = jnp.concatenate([cls_token, x], axis=1)
        if interpolation:
            raise NotImplementedError(
                "Position interpolation not implemented in this simplified version"
            )
        else:
            pos_embed = self.pos_embedding.value

        x = x + pos_embed
        # Apply pre-norm if exists
        if self.pre_norm is not None:
            x = self.pre_norm(x)

        return self.transformer[0](x)

        # Apply transformer blocks
        if use_31_block:
            for block in self.transformer[:-1]:
                x = block(x)
        else:
            for block in self.transformer:
                x = block(x)
        return x


_MEAN = jnp.array([0.48145466, 0.4578275, 0.40821073], jnp.float32)
_STD = jnp.array([0.26862954, 0.26130258, 0.27577711], jnp.float32)


def _prep_frame(frame_bchw, image_size):
    x = frame_bchw
    x = x * 0.5 + 0.5
    x = jax.image.resize(
        x,
        (frame_bchw.shape[0], frame_bchw.shape[1], image_size, image_size),
        method="cubic",
    )
    x = (x - _MEAN.reshape(1, 3, 1, 1)) / _STD.reshape(1, 3, 1, 1)
    return x


def clip_preprocess_videos(videos_bcfhw, image_size=224):
    videos_bchw = rearrange(videos_bcfhw, "b c f h w -> (b f) c h w")
    videos_bchw = _prep_frame(videos_bchw, image_size)
    videos_bcfhw = rearrange(
        videos_bchw, "(b f) c h w -> b c f h w", f=videos_bcfhw.shape[2]
    )
    return videos_bchw


class CLIPModel(nnx.Module):
    """XLM-RoBERTa CLIP model (vision part only for now)."""

    def __init__(
        self,
        embed_dim=1024,
        image_size=224,
        patch_size=14,
        vision_dim=1280,
        vision_mlp_ratio=4,
        vision_heads=16,
        vision_layers=32,
        vision_pool="token",
        vision_pre_norm=True,
        vision_post_norm=False,
        activation="gelu",
        attn_dropout=0.0,
        proj_dropout=0.0,
        embedding_dropout=0.0,
        norm_eps=1e-5,
        **kwargs,  # Ignore text model parameters for now
    ):
        self.embed_dim = embed_dim
        self.image_size = image_size
        self.patch_size = patch_size
        self.vision_dim = vision_dim
        self.vision_mlp_ratio = vision_mlp_ratio
        self.vision_heads = vision_heads
        self.vision_layers = vision_layers
        self.vision_pre_norm = vision_pre_norm
        self.vision_post_norm = vision_post_norm
        self.activation = activation
        self.norm_eps = norm_eps

        # Vision model
        self.model = VisionTransformer(
            image_size=image_size,
            patch_size=patch_size,
            dim=vision_dim,
            mlp_ratio=vision_mlp_ratio,
            out_dim=embed_dim,
            num_heads=vision_heads,
            num_layers=vision_layers,
            pool_type=vision_pool,
            pre_norm=vision_pre_norm,
            post_norm=vision_post_norm,
            activation=activation,
            attn_dropout=attn_dropout,
            proj_dropout=proj_dropout,
            embedding_dropout=embedding_dropout,
            norm_eps=norm_eps,
        )
        # Log scale parameter

    def encode_video(
        self,
        videos_bcfhw,
    ):
        """
        Input:
        videos_bcfhw: (B, C, F, H, W)
        return (B, F, D)
        """
        videos = clip_preprocess_videos(videos_bcfhw)
        xi = self.model(videos, use_31_block=True)
        return xi
