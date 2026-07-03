import math
from functools import partial

import einops
import jax
import jax.numpy as jnp
from einops import rearrange
from flax import nnx
from jax.sharding import PartitionSpec

from src.models.kv_cache import KVCache, KVCacheDict
from src.models.transformer import Conv3d, MLPProj
from src.models.transformer import WanI2VCrossAttention as WanI2VCrossAttention
from src.models.transformer import WanLayerNorm, WanRMSNorm
from src.models.transformer_utils import (
    rope_apply,
    rope_params,
    sinusoidal_embedding_1d,
)

from ..action_module import ActionModule


class Identity(nnx.Module):
    """Simple identity module that passes input through unchanged."""

    def __call__(self, x, *args, **kwargs):
        return x


class SelfAttention(nnx.Module):
    def __init__(
        self,
        dim,
        num_heads,
        local_attn_size=-1,
        sink_size=0,
        qk_norm=True,
        eps=1e-6,
        rngs=nnx.Rngs(0),
        platform="tpu",
    ):
        super().__init__()
        assert dim % num_heads == 0

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.qk_norm = qk_norm
        self.eps = eps
        self.platform = platform
        self.max_attention_size = (
            15 * 1 * 880 if local_attn_size == -1 else local_attn_size * 880
        )

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

        self.norm_q = WanRMSNorm(dim, eps) if qk_norm else lambda x: x
        self.norm_k = WanRMSNorm(dim, eps) if qk_norm else lambda x: x

    def __call__(
        self,
        x_BTD,
        _seq_lens,
        grid_sizes,
        freqs,
        block_mask=None,
        kv_cache=None,
        current_start=0,
        mesh=None,
        bidirectional=False,
        teacher_forcing=False,
    ):
        """
        Forward pass with optional KV caching.
        Args:
            x: Input tensor of shape [B, L, C]
            seq_lens: Sequence lengths [B]
            grid_sizes: Grid sizes [B, 3] or [3]
            freqs: RoPE frequencies
            block_mask: Optional attention mask
            kv_cache: Optional KV cache dictionary
            current_start: Current position in sequence
            cache_start: Cache start position
        """
        q_BTHD = rearrange(
            self.norm_q(self.q(x_BTD)), "b s (n d) -> b s n d", n=self.num_heads
        )
        k_BTHD = rearrange(
            self.norm_k(self.k(x_BTD)), "b s (n d) -> b s n d", n=self.num_heads
        )
        v_BTHD = rearrange(self.v(x_BTD), "b s (n d) -> b s n d", n=self.num_heads)

        if bidirectional:
            if teacher_forcing or kv_cache is not None:
                raise ValueError(
                    "bidirectional=True is incompatible with teacher_forcing=True or kv_cache!=None"
                )

            roped_query_BTHD = rope_apply(q_BTHD, grid_sizes, freqs).astype(
                v_BTHD.dtype
            )
            roped_key_BTHD = rope_apply(k_BTHD, grid_sizes, freqs).astype(v_BTHD.dtype)
            if self.platform == "tpu":
                from src.utils.tpu.splash_attn import full_splash_attn

                sharded_attn = jax.shard_map(
                    partial(full_splash_attn),
                    in_specs=(
                        PartitionSpec(
                            "data",
                        ),
                        PartitionSpec(
                            "data",
                        ),
                        PartitionSpec(
                            "data",
                        ),
                    ),
                    out_specs=PartitionSpec(
                        "data",
                    ),
                    mesh=mesh,
                    check_vma=False,
                )
                x_BTHD = sharded_attn(roped_query_BTHD, roped_key_BTHD, v_BTHD)
            else:
                x_BTHD = jax.nn.dot_product_attention(
                    roped_query_BTHD, roped_key_BTHD, v_BTHD, mask=block_mask
                )
            new_kv_cache = None

        elif teacher_forcing:
            # we apply rope to the keys and queries in the same way.
            new_kv_cache = None
            grid_sizes = (grid_sizes[0] // 2, grid_sizes[1], grid_sizes[2])
            s_tf = (
                grid_sizes[1] * grid_sizes[2]
            )  # spatial size per frame (single player)
            roped_query_BTHD = rearrange(
                rope_apply(
                    rearrange(q_BTHD, "b (r s) n d -> (b r) s n d", r=2),
                    grid_sizes,
                    freqs,
                    start_frame=current_start,
                ).astype(v_BTHD.dtype),
                "(b r) s n d -> b (r s) n d",
                r=2,
            )
            roped_key_BTHD = rearrange(
                rope_apply(
                    rearrange(k_BTHD, "b (r s) n d -> (b r) s n d", r=2),
                    grid_sizes,
                    freqs,
                    start_frame=current_start,
                ).astype(v_BTHD.dtype),
                "(b r) s n d -> b (r s) n d",
                r=2,
            )
            if self.platform == "tpu":
                from src.utils.tpu.splash_attn import (
                    teacher_forcing_block_causal_splash_attn,
                )

                sharded_attn = jax.shard_map(
                    partial(teacher_forcing_block_causal_splash_attn, block_size=s_tf),
                    in_specs=(
                        PartitionSpec(
                            "data",
                        ),
                        PartitionSpec(
                            "data",
                        ),
                        PartitionSpec(
                            "data",
                        ),
                    ),
                    out_specs=PartitionSpec(
                        "data",
                    ),
                    mesh=mesh,
                    check_vma=False,
                )
                x_BTHD = sharded_attn(roped_query_BTHD, roped_key_BTHD, v_BTHD)
            else:
                x_BTHD = jax.nn.dot_product_attention(
                    roped_query_BTHD, roped_key_BTHD, v_BTHD, mask=block_mask
                )

        elif kv_cache is None:
            roped_query_BTHD = rope_apply(q_BTHD, grid_sizes, freqs).astype(
                v_BTHD.dtype
            )
            roped_key_BTHD = rope_apply(k_BTHD, grid_sizes, freqs).astype(v_BTHD.dtype)
            new_kv_cache = None
            s0 = grid_sizes[1] * grid_sizes[2]  # spatial size per frame (single player)
            if self.platform == "tpu":
                from src.utils.tpu.splash_attn import block_causal_splash_attn

                sharded_attn = jax.shard_map(
                    partial(block_causal_splash_attn, block_size=s0),
                    in_specs=(
                        PartitionSpec(
                            "data",
                        ),
                        PartitionSpec(
                            "data",
                        ),
                        PartitionSpec(
                            "data",
                        ),
                    ),
                    out_specs=PartitionSpec(
                        "data",
                    ),
                    mesh=mesh,
                    check_vma=False,
                )
                x_BTHD = sharded_attn(roped_query_BTHD, roped_key_BTHD, v_BTHD)
            else:
                x_BTHD = jax.nn.dot_product_attention(
                    roped_query_BTHD, roped_key_BTHD, v_BTHD, mask=block_mask
                )

        else:
            roped_query_BTHD = rope_apply(
                q_BTHD, grid_sizes, freqs, start_frame=current_start
            ).astype(v_BTHD.dtype)
            roped_key_BTHD = rope_apply(
                k_BTHD, grid_sizes, freqs, start_frame=current_start
            ).astype(v_BTHD.dtype)
            new_kv_cache = kv_cache.update(roped_key_BTHD, v_BTHD)
            kv_len = new_kv_cache.k.shape[1]

            mask = jnp.arange(kv_len) >= (kv_len - new_kv_cache.length)
            x_BTHD = jax.nn.dot_product_attention(
                roped_query_BTHD,
                new_kv_cache.k,
                new_kv_cache.v,
                mask=mask[None, None, :],
            )
        # Output projection
        x_BTD = rearrange(x_BTHD, "b s n d -> b s (n d)")
        x_BTD = self.o(x_BTD)
        return x_BTD, new_kv_cache


class SolarisSPBlock(nnx.Module):
    def __init__(
        self,
        cross_attn_type,
        dim,
        ffn_dim,
        num_heads,
        local_attn_size=-1,
        sink_size=0,
        qk_norm=True,
        cross_attn_norm=False,
        action_config={},
        action_module=False,
        eps=1e-6,
        rngs=nnx.Rngs(0),
        platform="tpu",
    ):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.platform = platform

        # Action module if configured
        if len(action_config) != 0 and action_module:
            self.action_model = ActionModule(
                **action_config, local_attn_size=self.local_attn_size, rngs=rngs
            )
        else:
            self.action_model = None

        self.norm1 = WanLayerNorm(dim, eps, rngs=rngs)
        self.self_attn = SelfAttention(
            dim,
            num_heads,
            local_attn_size,
            sink_size,
            qk_norm,
            eps,
            rngs=rngs,
            platform=platform,
        )
        self.norm3 = (
            WanLayerNorm(dim, eps, elementwise_affine=True, rngs=rngs)
            if cross_attn_norm
            else lambda x: x
        )
        self.cross_attn = WanI2VCrossAttention(
            dim, num_heads, (-1, -1), qk_norm, eps, rngs=rngs
        )
        self.norm2 = WanLayerNorm(dim, eps, rngs=rngs)
        self.ffn = nnx.Sequential(
            nnx.Linear(
                dim, ffn_dim, rngs=rngs, param_dtype=jnp.float32, dtype=jnp.bfloat16
            ),
            nnx.gelu,
            nnx.Linear(
                ffn_dim, dim, rngs=rngs, param_dtype=jnp.float32, dtype=jnp.bfloat16
            ),
        )
        self.modulation = nnx.Param(
            jax.random.normal(rngs.params(), (1, 6, dim)).astype(jnp.float32)
            / math.sqrt(dim)
        )

    def __call__(
        self,
        x,  # (B, F, L, dim)
        e,  # (B, F, 6, dim)
        seq_lens,  # (B,)
        grid_sizes,  # = (F, H, W)
        freqs,  # = (freq_dim,)
        context,  # = (B, F, dim)
        block_mask=None,
        block_mask_mouse=None,
        block_mask_keyboard=None,
        num_frame_per_block=1,
        mouse_cond=None,
        keyboard_cond=None,
        kv_cache=None,
        kv_cache_mouse=None,
        kv_cache_keyboard=None,
        crossattn_cache=None,
        current_start=0,
        use_action_module=jnp.array([True]),
        mesh=None,
        teacher_forcing=False,
        bidirectional=False,
    ):
        """Forward pass through the attention block."""
        assert e.ndim == 4

        num_frames = e.shape[1]
        pack = lambda x: rearrange(x, "b f p c -> b (f p) c", f=num_frames)
        unpack = lambda x: rearrange(x, "b (f p) c -> b f p c", f=num_frames)

        # for the duration of the function, keep x in unpacked format (i.e. separate the frames from the time)
        # only convert to packed format for self-attention.

        casted_modulation = self.modulation.value.astype(jnp.bfloat16)
        modulation = jnp.expand_dims(casted_modulation, axis=1)  # 1, 6, 1, dim
        e = jnp.split(modulation + e, 6, axis=2)
        x = unpack(x)

        y_packed, new_kv_cache = self.self_attn(
            pack(self.norm1(x.astype(jnp.float32)).astype(x.dtype) * (1 + e[1]) + e[0]),
            seq_lens,
            grid_sizes,
            freqs,
            block_mask,
            kv_cache,
            current_start,
            mesh=mesh,
            teacher_forcing=teacher_forcing,
            bidirectional=bidirectional,
        )
        y = unpack(y_packed)
        x = x + y * e[2]

        x = x + unpack(
            self.cross_attn(
                pack(self.norm3(x.astype(jnp.float32)).astype(x.dtype)), context
            )
        )

        if kv_cache_mouse is not None:

            def run_action_module(x):
                x_packed, new_kv_cache_mouse, new_kv_cache_keyboard = self.action_model(
                    pack(x.astype(context.dtype)),
                    grid_sizes[0],
                    grid_sizes[1],
                    grid_sizes[2],
                    mouse_cond,
                    keyboard_cond,
                    block_mask_mouse,
                    block_mask_keyboard,
                    kv_cache_mouse=kv_cache_mouse,
                    kv_cache_keyboard=kv_cache_keyboard,
                    start_frame=current_start,
                    num_frame_per_block=num_frame_per_block,
                )
                return unpack(x_packed), new_kv_cache_mouse, new_kv_cache_keyboard

            x, new_kv_cache_mouse, new_kv_cache_keyboard = jax.lax.cond(
                jnp.asarray(use_action_module).any(),
                run_action_module,
                lambda x: (x, kv_cache_mouse, kv_cache_keyboard),
                x,
            )
        else:

            def run_action_module_no_kv_cache(x):
                x_packed, _, _ = self.action_model(
                    pack(x.astype(context.dtype)),
                    grid_sizes[0],
                    grid_sizes[1],
                    grid_sizes[2],
                    mouse_cond,
                    keyboard_cond,
                    block_mask_mouse,
                    block_mask_keyboard,
                    kv_cache_mouse=None,
                    kv_cache_keyboard=None,
                    start_frame=current_start,
                    num_frame_per_block=num_frame_per_block,
                    teacher_forcing=teacher_forcing,
                )
                return unpack(x_packed)

            x = jax.lax.cond(
                jnp.asarray(use_action_module).any(),
                run_action_module_no_kv_cache,
                lambda x: x,
                x,
            )
            new_kv_cache_mouse = None
            new_kv_cache_keyboard = None

        y = (
            self.ffn(
                self.norm2(x.astype(jnp.float32)).astype(x.dtype) * (1 + e[4]) + e[3]
            )
            * e[5]
        )
        x = x + y

        return pack(x), new_kv_cache, new_kv_cache_mouse, new_kv_cache_keyboard


class OutputHead(nnx.Module):

    def __init__(
        self,
        dim,
        out_dim,
        patch_size,
        eps=1e-6,
    ):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps
        # Output projection
        out_dim_total = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nnx.Linear(
            dim,
            out_dim_total,
            rngs=nnx.Rngs(0),
            param_dtype=jnp.float32,
            dtype=jnp.bfloat16,
        )
        key = jax.random.PRNGKey(42)
        self.modulation = nnx.Param(
            jax.random.normal(key, (1, 2, dim)).astype(jnp.float32) / math.sqrt(dim)
        )

    def __call__(self, x, e):
        num_frames = e.shape[1]
        casted_modulation = self.modulation.value.astype(jnp.bfloat16)
        modulation = jnp.expand_dims(casted_modulation, axis=1)
        e = jnp.split(modulation + e, 2, axis=2)

        norm_x = self.norm(x.astype(jnp.float32)).astype(x.dtype)
        norm_x_frames = rearrange(norm_x, "b (f p) c -> b f p c", f=num_frames)
        modulated_x = norm_x_frames * (1 + e[1]) + e[0]
        x = self.head(modulated_x)
        return x


class SolarisSPModel(nnx.Module):
    def __init__(
        self,
        model_type="i2v",
        patch_size=(1, 2, 2),
        text_len=512,
        in_dim=36,
        dim=1536,
        ffn_dim=8960,
        freq_dim=256,
        text_dim=4096,
        out_dim=16,
        num_heads=12,
        num_layers=30,
        local_attn_size=-1,
        sink_size=0,
        qk_norm=True,
        cross_attn_norm=True,
        action_config={},
        inject_sample_info=False,
        eps=1e-6,
        rngs=nnx.Rngs(0),
        platform="tpu",
    ):
        super().__init__()

        assert model_type in ["i2v"], "Only 'i2v' model type is supported"

        self.model_type = model_type
        self.use_action_module = len(action_config) > 0
        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.platform = platform

        # Patch embedding /
        self.patch_embedding = Conv3d(
            in_dim, dim, kernel_size=patch_size, strides=patch_size, rngs=rngs
        )

        # Time embeddings /
        self.time_embedding = nnx.Sequential(
            nnx.Linear(
                freq_dim, dim, rngs=rngs, param_dtype=jnp.float32, dtype=jnp.bfloat16
            ),
            nnx.silu,
            nnx.Linear(
                dim, dim, rngs=rngs, param_dtype=jnp.float32, dtype=jnp.bfloat16
            ),
        )
        self.time_projection = nnx.Sequential(
            nnx.silu,
            nnx.Linear(
                dim, dim * 6, rngs=rngs, param_dtype=jnp.float32, dtype=jnp.bfloat16
            ),
        )

        # Transformer blocks

        @nnx.split_rngs(splits=num_layers)
        @nnx.vmap(in_axes=(0,), out_axes=0)
        def create_layers(r):
            return SolarisSPBlock(
                "i2v_cross_attn",
                dim,
                ffn_dim,
                num_heads,
                local_attn_size,
                sink_size,
                qk_norm,
                cross_attn_norm,
                action_config=action_config,
                action_module=True,
                eps=eps,
                rngs=r,
                platform=platform,
            )

        self.blocks = create_layers(rngs)
        # Output head
        self.head = OutputHead(dim, out_dim, patch_size, eps)
        # RoPE frequencies
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        self.d = dim // num_heads
        # Image embedding for i2v mode
        self.img_emb = MLPProj(1280, dim)
        self.num_frame_per_block = 1
        assert (
            self.num_frame_per_block == 1
        ), "num_frame_per_block other than 1 is not supported!"

    @staticmethod
    def get_causal_attn_mask(
        num_q_blocks,
        num_k_blocks,
        q_block_size,
        k_block_size,
        sliding_block_size=-1,
    ):
        """Prepare block-wise causal attention mask."""
        # currently we decode 1 frame at a time.
        i = jnp.arange(num_q_blocks)[:, None]  # (num_q_blocks, 1)
        j = jnp.arange(num_k_blocks)[None, :]  # (1, num_k_blocks)
        if sliding_block_size == -1:
            block_mask = j <= i
        else:
            block_mask = (j >= i - sliding_block_size + 1) & (j <= i)
        block_mask = einops.repeat(
            block_mask, "q k -> (q p1) (k p2)", p1=q_block_size, p2=k_block_size
        )
        return block_mask

    # get a causal attention mask that does not have diagonals
    @staticmethod
    def get_causal_attn_mask_no_diagonals(
        num_q_blocks,
        num_k_blocks,
        q_block_size,
        k_block_size,
        sliding_block_size=-1,
    ):
        """Prepare block-wise causal attention mask."""
        # currently we decode 1 frame at a time.
        i = jnp.arange(num_q_blocks)[:, None]  # (num_q_blocks, 1)
        j = jnp.arange(num_k_blocks)[None, :]  # (1, num_k_blocks)
        if sliding_block_size == -1:
            block_mask = j < i
        else:
            block_mask = (j >= i - sliding_block_size + 1) & (j < i)
        block_mask = einops.repeat(
            block_mask, "q k -> (q p1) (k p2)", p1=q_block_size, p2=k_block_size
        )
        return block_mask

    @staticmethod
    def get_block_mask_teacher_forcing(
        num_q_blocks,
        num_k_blocks,
        q_block_size,
        k_block_size,
        sliding_block_size=-1,
    ):
        """prepare teacher forcing mask"""
        # clean-clean block mask
        clean_clean_block_mask = SolarisSPModel.get_causal_attn_mask(
            num_q_blocks, num_k_blocks, q_block_size, k_block_size, sliding_block_size
        )
        # unclean-unclean block mask is just a diagonal matrix.
        unclean_unclean_block_mask = SolarisSPModel.get_causal_attn_mask(
            num_q_blocks, num_k_blocks, q_block_size, k_block_size, 1
        )
        # unclean-clean block mask attends to the past.
        unclean_clean_block_mask = SolarisSPModel.get_causal_attn_mask_no_diagonals(
            num_q_blocks, num_k_blocks, q_block_size, k_block_size, sliding_block_size
        )
        clean_unclean_block_mask = jnp.zeros_like(unclean_clean_block_mask)

        top_rows = jnp.concatenate(
            [clean_clean_block_mask, clean_unclean_block_mask], axis=1
        )
        bottom_rows = jnp.concatenate(
            [unclean_clean_block_mask, unclean_unclean_block_mask], axis=1
        )
        block_mask = jnp.concatenate([top_rows, bottom_rows], axis=0)
        return block_mask

    def __call__(
        self,
        x_BFHWC,
        t_BT,
        visual_context,
        cond_concat,
        mouse_cond=None,
        keyboard_cond=None,
        kv_cache=None,
        kv_cache_mouse=None,
        kv_cache_keyboard=None,
        current_start=0,
        matrix_game_forward=True,
        teacher_forcing=False,
        mesh=None,
        bidirectional=False,
    ):
        x_BFHWC = jnp.concatenate([x_BFHWC, cond_concat], axis=-1)
        x_BFHWC = rearrange(
            self.patch_embedding(rearrange(x_BFHWC, "b f h w c -> b c f h w")),
            "b c f h w -> b f h w c",
        )
        grid_sizes = x_BFHWC.shape[1:4]
        x_BFC = rearrange(x_BFHWC, "b f h w c -> b (f h w) c")
        seq_lens = jnp.array([x_BFC.shape[1]], dtype=jnp.int64)
        e_BD = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t_BT.flatten()).astype(x_BFC.dtype)
        )
        e0 = rearrange(
            self.time_projection(e_BD), "(b f) (r d) -> b f r d", b=t_BT.shape[0], r=6
        )
        context = self.img_emb(visual_context)

        n_frames = grid_sizes[0]
        num_patches = grid_sizes[1] * grid_sizes[2]

        if bidirectional:
            block_mask = None
            block_mask_mouse = None
            block_mask_keyboard = None
        elif kv_cache is None and not teacher_forcing:
            block_mask = self.get_causal_attn_mask(
                num_q_blocks=n_frames,
                num_k_blocks=n_frames,
                q_block_size=num_patches,
                k_block_size=num_patches,
                sliding_block_size=self.local_attn_size,
            )
            block_mask_mouse = self.get_causal_attn_mask(
                num_q_blocks=n_frames,
                num_k_blocks=n_frames,
                q_block_size=1,
                k_block_size=1,
                sliding_block_size=self.local_attn_size,
            )
            block_mask_keyboard = self.get_causal_attn_mask(
                num_q_blocks=n_frames,
                num_k_blocks=n_frames,
                q_block_size=1,
                k_block_size=1,
                sliding_block_size=self.local_attn_size,
            )
        elif kv_cache is None and teacher_forcing:
            block_mask = self.get_block_mask_teacher_forcing(
                num_q_blocks=n_frames // 2,
                num_k_blocks=n_frames // 2,
                q_block_size=num_patches,
                k_block_size=num_patches,
                sliding_block_size=self.local_attn_size,
            )
            block_mask_mouse = self.get_block_mask_teacher_forcing(
                num_q_blocks=n_frames // 2,
                num_k_blocks=n_frames // 2,
                q_block_size=1,
                k_block_size=1,
                sliding_block_size=self.local_attn_size,
            )
            block_mask_keyboard = self.get_block_mask_teacher_forcing(
                num_q_blocks=n_frames // 2,
                num_k_blocks=n_frames // 2,
                q_block_size=1,
                k_block_size=1,
                sliding_block_size=self.local_attn_size,
            )
        else:
            block_mask = None
            block_mask_mouse = None
            block_mask_keyboard = None

        d = self.d
        freqs = jnp.concatenate(
            [
                rope_params(1024, d - 4 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
            ],
            axis=1,
        )

        def apply_block_fn(
            block,
            x,
            e0,
            seq_lens,
            grid_sizes,
            freqs,
            context,
            mouse_cond,
            keyboard_cond,
            block_mask,
            block_mask_mouse,
            block_mask_keyboard,
            num_frame_per_block,
            current_start,
            use_action_module,
            kv_cache,
            kv_cache_mouse,
            kv_cache_keyboard,
        ):
            return block(
                x,
                e0,
                seq_lens,
                grid_sizes,
                freqs,
                context,
                block_mask=block_mask,
                mouse_cond=mouse_cond,
                keyboard_cond=keyboard_cond,
                block_mask_mouse=block_mask_mouse,
                block_mask_keyboard=block_mask_keyboard,
                num_frame_per_block=num_frame_per_block,
                current_start=current_start,
                use_action_module=use_action_module,
                kv_cache=kv_cache,
                kv_cache_mouse=kv_cache_mouse,
                kv_cache_keyboard=kv_cache_keyboard,
                mesh=mesh,
                teacher_forcing=teacher_forcing,
                bidirectional=bidirectional,
            )

        if matrix_game_forward:
            action_module_zipper = jnp.array([True] * 15 + [False] * 15)
        else:
            action_module_zipper = jnp.array([True] * 30)

        if kv_cache is not None:

            @nnx.scan(
                in_axes=(0, 0, 0, 0, 0, nnx.Carry),
                out_axes=(nnx.Carry, 0, 0, 0),
                unroll=True,
            )
            def layers(
                model,
                kv_cache,
                kv_cache_mouse,
                kv_cache_keyboard,
                action_module,
                x,
            ):
                x, kv_cache, kv_cache_mouse, kv_cache_keyboard = nnx.remat(
                    apply_block_fn, static_argnums=(4, 12)
                )(
                    model,  # 0
                    x,  # 1
                    e0,  # 2
                    seq_lens,  # 3
                    grid_sizes,  # 4 --
                    freqs,  # 5
                    context,  # 6
                    mouse_cond,  # 7
                    keyboard_cond,  # 8
                    block_mask=block_mask,  # 9
                    block_mask_mouse=block_mask_mouse,  # 10
                    block_mask_keyboard=block_mask_keyboard,  # 11
                    num_frame_per_block=self.num_frame_per_block,  # 12 -- static
                    current_start=current_start,  # 13
                    use_action_module=action_module,  # 14
                    kv_cache=kv_cache,  # 15
                    kv_cache_mouse=kv_cache_mouse,  # 16
                    kv_cache_keyboard=kv_cache_keyboard,  # 17
                )
                return x, kv_cache, kv_cache_mouse, kv_cache_keyboard

            x_BFC, new_kv_cache, new_kv_cache_mouse, new_kv_cache_keyboard = layers(
                self.blocks,
                kv_cache,
                kv_cache_mouse,
                kv_cache_keyboard,
                action_module_zipper,
                x_BFC,
            )
        else:

            @nnx.scan(in_axes=(0, 0, nnx.Carry), out_axes=(nnx.Carry), unroll=False)
            def layers(model, action_module, x):
                x, _, _, _ = nnx.remat(apply_block_fn, static_argnums=(4, 12))(
                    model,  # 0
                    x,  # 1
                    e0,  # 2
                    seq_lens,  # 3
                    grid_sizes,  # 4
                    freqs,  # 5
                    context,  # 6
                    mouse_cond,  # 7
                    keyboard_cond,  # 8
                    block_mask=block_mask,  # 9
                    block_mask_mouse=block_mask_mouse,  # 10
                    block_mask_keyboard=block_mask_keyboard,  # 11
                    num_frame_per_block=self.num_frame_per_block,  # 12 -- static
                    current_start=current_start,  # 13
                    use_action_module=action_module,  # 14
                    kv_cache=kv_cache,  # 15
                    kv_cache_mouse=kv_cache_mouse,  # 16
                    kv_cache_keyboard=kv_cache_keyboard,  # 17
                )
                return x

            x_BFC = layers(self.blocks, action_module_zipper, x_BFC)
            new_kv_cache, new_kv_cache_mouse, new_kv_cache_keyboard = None, None, None

        x_BFPC = self.head(
            x_BFC, rearrange(e_BD, "(b f) (r d) -> b f r d", b=t_BT.shape[0], r=1)
        )
        x = self.unpatchify(x_BFPC, grid_sizes)
        return x, new_kv_cache, new_kv_cache_mouse, new_kv_cache_keyboard

    def unpatchify(self, x, grid_sizes):
        f, h, w = grid_sizes
        p1, p2, p3 = self.patch_size
        x = rearrange(
            x,
            "b f (h w) (p1 p2 p3 c) -> b (f p1) (h p2) (w p3) c",
            f=f,
            h=h,
            w=w,
            p1=p1,
            p2=p2,
            p3=p3,
            c=self.out_dim,
        )
        return x

    def initialize_kv_cache(
        self,
        batch_size,
        latent_height,
        latent_width,
        dtype=jnp.bfloat16,
    ):
        assert self.local_attn_size != -1
        frames_block_size = self.local_attn_size * latent_height * latent_width
        head_dim = self.dim // self.num_heads
        actions_head_num = self.blocks.action_model.heads_num
        mouse_head_dim = self.blocks.action_model.mouse_head_dim
        keyboard_head_dim = self.blocks.action_model.keyboard_head_dim

        @nnx.vmap(in_axes=0, out_axes=0)
        def create_cache(_key):
            return KVCacheDict(
                kv_cache=KVCache(
                    jnp.zeros(
                        (batch_size, frames_block_size, self.num_heads, head_dim),
                        dtype=dtype,
                    ),
                    jnp.zeros(
                        (batch_size, frames_block_size, self.num_heads, head_dim),
                        dtype=dtype,
                    ),
                ),
                kv_cache_mouse=KVCache(
                    jnp.zeros(
                        (
                            batch_size * latent_height * latent_width,
                            self.local_attn_size,
                            actions_head_num,
                            mouse_head_dim,
                        ),
                        dtype=dtype,
                    ),
                    jnp.zeros(
                        (
                            batch_size * latent_height * latent_width,
                            self.local_attn_size,
                            actions_head_num,
                            mouse_head_dim,
                        ),
                        dtype=dtype,
                    ),
                ),
                kv_cache_keyboard=KVCache(
                    jnp.zeros(
                        (
                            batch_size,
                            self.local_attn_size,
                            actions_head_num,
                            keyboard_head_dim,
                        ),
                        dtype=dtype,
                    ),
                    jnp.zeros(
                        (
                            batch_size,
                            self.local_attn_size,
                            actions_head_num,
                            keyboard_head_dim,
                        ),
                        dtype=dtype,
                    ),
                ),
            )

        keys = jax.random.split(jax.random.key(0), self.num_layers)
        return create_cache(keys)
