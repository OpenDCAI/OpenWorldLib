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

from ..action_module import ActionModule
from ..transformer_utils import apply_rope_mp, rope_params, sinusoidal_embedding_1d


class Identity(nnx.Module):
    """Simple identity module that passes input through unchanged."""

    def __call__(self, x, *args, **kwargs):
        return x


class MPSelfAttention(nnx.Module):

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
            15 * 2 * 880 if local_attn_size == -1 else local_attn_size * 880 * 2
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
        eff_p,
        block_mask=None,
        kv_cache=None,
        current_start=0,
        mesh=None,
        bidirectional=False,
        teacher_forcing=False,
        no_kv_backprop_teacher_forcing=False,
    ):
        """
        Forward pass with optional KV caching.
        Args:
            x: Input tensor of shape [B, L, C]
            seq_lens: Sequence lengths [B]
            grid_sizes: Grid sizes [B, 3] or [3]
            freqs: RoPE frequencies
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
        f0 = grid_sizes[0]
        s0 = grid_sizes[1] * grid_sizes[2]
        if bidirectional:
            if teacher_forcing or kv_cache is not None:
                raise ValueError(
                    "bidirectional=True is incompatible with teacher_forcing=True or kv_cache!=None"
                )

            roped_query_BTHD = apply_rope_mp(q_BTHD, grid_sizes, freqs, f0, s0)
            roped_key_BTHD = apply_rope_mp(k_BTHD, grid_sizes, freqs, f0, s0)

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
            # After changing grid_sizes, recompute f/s so RoPE lengths match.
            f_tf = grid_sizes[0]
            s_tf = grid_sizes[1] * grid_sizes[2]
            # Split by role (clean/noisy) before applying RoPE, then recombine.
            q_BTHD = rearrange(q_BTHD, "b (r s) n d -> (b r) s n d", r=2)
            k_BTHD = rearrange(k_BTHD, "b (r s) n d -> (b r) s n d", r=2)
            roped_query_BTHD = apply_rope_mp(q_BTHD, grid_sizes, freqs, f_tf, s_tf)
            roped_key_BTHD = apply_rope_mp(k_BTHD, grid_sizes, freqs, f_tf, s_tf)
            roped_query_BTHD = rearrange(
                roped_query_BTHD, "(b r) s n d -> b (r s) n d", r=2
            )
            roped_key_BTHD = rearrange(
                roped_key_BTHD, "(b r) s n d -> b (r s) n d", r=2
            )

            if no_kv_backprop_teacher_forcing:
                Tq = roped_query_BTHD.shape[1]
                past_keys, past_values = (
                    roped_key_BTHD[:, : Tq // 2, :, :],
                    v_BTHD[:, : Tq // 2, :, :],
                )
                present_keys, present_values = (
                    roped_key_BTHD[:, Tq // 2 :, :, :],
                    v_BTHD[:, Tq // 2 :, :, :],
                )
                past_keys = jax.lax.stop_gradient(past_keys)
                past_values = jax.lax.stop_gradient(past_values)
                roped_key_BTHD = jnp.concatenate([past_keys, present_keys], axis=1)
                v_BTHD = jnp.concatenate([past_values, present_values], axis=1)
            if self.platform == "tpu":
                from src.utils.tpu.splash_attn import (
                    teacher_forcing_block_causal_splash_attn,
                )

                # block_size must account for all players (s_tf * 2 for 2 players)
                # to ensure tokens from all players within a frame form a single block
                sharded_attn = jax.shard_map(
                    partial(
                        teacher_forcing_block_causal_splash_attn,
                        block_size=s_tf * eff_p,
                    ),
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
            roped_query_BTHD = apply_rope_mp(q_BTHD, grid_sizes, freqs, f0, s0)
            roped_key_BTHD = apply_rope_mp(k_BTHD, grid_sizes, freqs, f0, s0)
            new_kv_cache = None
            if self.platform == "tpu":
                from src.utils.tpu.splash_attn import block_causal_splash_attn

                sharded_attn = jax.shard_map(
                    partial(block_causal_splash_attn, block_size=s0 * eff_p),
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
            roped_query_BTHD = apply_rope_mp(
                q_BTHD, grid_sizes, freqs, f0, s0, current_start=current_start
            )
            roped_key_BTHD = apply_rope_mp(
                k_BTHD, grid_sizes, freqs, f0, s0, current_start=current_start
            )
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


class SolarisMPBlock(nnx.Module):
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
        self.self_attn = MPSelfAttention(
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
        player_embed_PD,  # (P, dim) - shared player embeddings
        block_mask=None,
        block_mask_mouse=None,  # = (B, F, L, L)
        block_mask_keyboard=None,  # = (B, F, L, L)
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
        no_kv_backprop_teacher_forcing=False,
    ):
        """Forward pass through the attention block."""
        b, p = x.shape[0], x.shape[1]

        num_frames = e.shape[2]
        pack = lambda x: rearrange(x, "b p f s c -> b p (f s) c", f=num_frames)
        unpack = lambda x: rearrange(x, "b p (f s) c -> b p f s c", f=num_frames)

        casted_modulation = self.modulation.value.astype(jnp.bfloat16)
        modulation = casted_modulation[:, None, None, :, :]
        e = jnp.split(modulation + e, 6, axis=3)  # B P F R D

        x = unpack(x)

        # Use shared player embeddings passed from parent model
        player_embed_BPFSD = player_embed_PD[None, :, None, None, :].astype(x.dtype)
        x = x + player_embed_BPFSD

        y_packed, new_kv_cache = self.self_attn(
            rearrange(
                self.norm1(x.astype(jnp.float32)).astype(x.dtype) * (1 + e[1]) + e[0],
                "b p f s c -> b (f p s) c",
            ),
            seq_lens,
            grid_sizes,
            freqs,
            p,
            block_mask,
            kv_cache,
            current_start,
            mesh=mesh,
            teacher_forcing=teacher_forcing,
            no_kv_backprop_teacher_forcing=no_kv_backprop_teacher_forcing,
            bidirectional=bidirectional,
        )
        y = rearrange(y_packed, "b (f p s) c -> b p f s c", f=num_frames, p=p)

        x = x + y * e[2]

        x = x + rearrange(
            self.cross_attn(
                rearrange(
                    self.norm3(x.astype(jnp.float32)).astype(x.dtype),
                    "b p f s c -> (b p) (f s) c",
                ),
                rearrange(context, "b p f d -> (b p) f d"),
            ),
            "(b p) (f s) c -> b p f s c",
            b=b,
            f=num_frames,
        )

        if kv_cache_mouse is not None:

            def run_action_module(x):
                x_packed, new_kv_cache_mouse, new_kv_cache_keyboard = self.action_model(
                    rearrange(x.astype(context.dtype), "b p f s c -> (b p) (f s) c"),
                    grid_sizes[0],
                    grid_sizes[1],
                    grid_sizes[2],
                    rearrange(mouse_cond, "b p f d -> (b p) f d"),
                    rearrange(keyboard_cond, "b p f d -> (b p) f d"),
                    block_mask_mouse,
                    block_mask_keyboard,
                    kv_cache_mouse=kv_cache_mouse,
                    kv_cache_keyboard=kv_cache_keyboard,
                    start_frame=current_start,
                    num_frame_per_block=num_frame_per_block,
                    teacher_forcing=teacher_forcing,
                    no_kv_backprop_teacher_forcing=no_kv_backprop_teacher_forcing,
                )
                return (
                    rearrange(
                        x_packed, "(b p) (f s) c -> b p f s c", b=b, f=num_frames
                    ),
                    new_kv_cache_mouse,
                    new_kv_cache_keyboard,
                )

            x, new_kv_cache_mouse, new_kv_cache_keyboard = jax.lax.cond(
                jnp.asarray(use_action_module).any(),
                run_action_module,
                lambda x: (x, kv_cache_mouse, kv_cache_keyboard),
                x,
            )
        else:

            def run_action_module_no_kv_cache(x):
                x_packed, _, _ = self.action_model(
                    rearrange(x.astype(context.dtype), "b p f s c -> (b p) (f s) c"),
                    grid_sizes[0],
                    grid_sizes[1],
                    grid_sizes[2],
                    rearrange(mouse_cond, "b p f d -> (b p) f d"),
                    rearrange(keyboard_cond, "b p f d -> (b p) f d"),
                    block_mask_mouse,
                    block_mask_keyboard,
                    kv_cache_mouse=None,
                    kv_cache_keyboard=None,
                    start_frame=current_start,
                    num_frame_per_block=num_frame_per_block,
                    teacher_forcing=teacher_forcing,
                    no_kv_backprop_teacher_forcing=no_kv_backprop_teacher_forcing,
                )
                return rearrange(
                    x_packed, "(b p) (f s) c -> b p f s c", b=b, f=num_frames
                )

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
        f = e.shape[2]
        casted_modulation = self.modulation.value.astype(jnp.bfloat16)
        modulation = casted_modulation[:, None, None]
        e = jnp.split(modulation + e, 2, axis=3)
        norm_x = self.norm(x.astype(jnp.float32)).astype(x.dtype)
        norm_x_frames = rearrange(norm_x, "b p (f s) c -> b p f s c", f=f)
        modulated_x = norm_x_frames * (1 + e[1]) + e[0]
        x = self.head(modulated_x)
        return x


class SolarisMPModel(nnx.Module):

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
        multiplayer_method="multiplayer_attn",
        num_players=2,
        platform="tpu",
    ):
        super().__init__()

        assert model_type in ["i2v"], "Only 'i2v' model type is supported"

        assert multiplayer_method in [
            "multiplayer_attn",
            "concat_c",
        ], f"multiplayer_method must be 'multiplayer_attn', or 'concat_c', got {multiplayer_method}"

        self.model_type = model_type
        self.multiplayer_method = multiplayer_method
        self.num_players = num_players
        self.use_action_module = len(action_config) > 0
        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.platform = platform
        # In multiplayer_attn mode we keep an explicit player dimension P=2. Under
        # concat_* modes we expect video tensors to have P=1 and encode the
        # original player axis into spatial or channel dimensions instead.
        self.expected_p_size = 1 if multiplayer_method == "concat_c" else 2

        # For concat_c, input/output channels are multiplied by num_players since
        # players are stacked in channel dimension. For multiplayer_attn,
        # channels stay the same.
        c_factor = num_players if multiplayer_method == "concat_c" else 1
        in_dim_eff = in_dim * c_factor
        out_dim_eff = out_dim * c_factor

        # Store effective out_dim for unpatchify
        self.out_dim = out_dim_eff

        # Patch embedding /
        self.patch_embedding = Conv3d(
            in_dim_eff, dim, kernel_size=patch_size, strides=patch_size, rngs=rngs
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
        # Effective action-module input dims. In concat_* modes we have already
        # reshaped actions to (B, 1, F, num_players * D_*), so we scale the
        # configured per-player dims by num_players. In multiplayer_attn mode we keep
        # per-player dims as-is.
        if len(action_config) > 0:
            # Validate left_action_padding is divisible by vae_time_compression_ratio
            left_action_padding = action_config.left_action_padding
            if left_action_padding != 12 and left_action_padding != 11:
                raise ValueError(
                    "left_action_padding must be 12 or 11 (tight padding given first latent frame is not temporally compressed)"
                )

            p_factor = num_players if multiplayer_method == "concat_c" else 1
            action_config_eff = dict(action_config)
            if "mouse_dim_in" in action_config_eff:
                action_config_eff["mouse_dim_in"] = (
                    action_config_eff["mouse_dim_in"] * p_factor
                )
            if "keyboard_dim_in" in action_config_eff:
                action_config_eff["keyboard_dim_in"] = (
                    action_config_eff["keyboard_dim_in"] * p_factor
                )
        else:
            action_config_eff = action_config

        @nnx.split_rngs(splits=num_layers)
        @nnx.vmap(in_axes=(0,), out_axes=0)
        def create_layers(r):
            return SolarisMPBlock(
                "i2v_cross_attn",
                dim,
                ffn_dim,
                num_heads,
                local_attn_size,
                sink_size,
                qk_norm,
                cross_attn_norm,
                action_config=action_config_eff,
                action_module=True,
                eps=eps,
                rngs=r,
                platform=platform,
            )

        self.blocks = create_layers(rngs)
        # Output head (uses effective out_dim for concat_c)
        self.head = OutputHead(dim, out_dim_eff, patch_size, eps)
        # RoPE frequencies
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        self.d = dim // num_heads
        # Image embedding for i2v mode. Under concat_* we receive
        # visual_context_BPFD with D_v = 1280 * num_players, otherwise 1280.
        eff_clip_dim = 1280 * num_players if multiplayer_method == "concat_c" else 1280
        self.img_emb = MLPProj(eff_clip_dim, dim)
        self.num_frame_per_block = 1
        assert (
            self.num_frame_per_block == 1
        ), "num_frame_per_block other than 1 is not supported!"

        # Shared player embedding across all blocks
        self.player_embed = nnx.Embed(
            num_embeddings=num_players, features=dim, rngs=rngs
        )

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
        clean_clean_block_mask = SolarisMPModel.get_causal_attn_mask(
            num_q_blocks, num_k_blocks, q_block_size, k_block_size, sliding_block_size
        )
        # unclean-unclean block mask is just a diagonal matrix.
        unclean_unclean_block_mask = SolarisMPModel.get_causal_attn_mask(
            num_q_blocks, num_k_blocks, q_block_size, k_block_size, 1
        )
        # unclean-clean block mask attends to the past.
        unclean_clean_block_mask = SolarisMPModel.get_causal_attn_mask_no_diagonals(
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
        x_BPFHWC,
        t_BPT,
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
        no_kv_backprop_teacher_forcing=False,
    ):
        b = x_BPFHWC.shape[0]
        p = x_BPFHWC.shape[1]

        assert (
            p == self.expected_p_size
        ), f"Expected P={self.expected_p_size} for multiplayer_method={self.multiplayer_method}, got P={p}"

        x_BPFHWC = jnp.concatenate([x_BPFHWC, cond_concat], axis=-1)
        x_BPFHWC = rearrange(
            self.patch_embedding(rearrange(x_BPFHWC, "b p f h w c -> (b p) c f h w")),
            "(b p) c f h w -> b p f h w c",
            b=b,
        )
        grid_sizes = x_BPFHWC.shape[2:5]

        x_BPFC = rearrange(x_BPFHWC, "b p f h w c -> b p (f h w) c")
        seq_lens = jnp.array([x_BPFC.shape[2]], dtype=jnp.int64)
        e_BD = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t_BPT.flatten()).astype(x_BPFC.dtype)
        )
        e0 = rearrange(
            self.time_projection(e_BD),
            "(b p f) (r d) -> b p f r d",
            b=t_BPT.shape[0],
            p=p,
            r=6,
        )
        context = rearrange(
            self.img_emb(rearrange(visual_context, "b p f d -> (b p) f d")),
            "(b p) f d -> b p f d",
            b=b,
        )
        n_frames = grid_sizes[0]
        num_patches = grid_sizes[1] * grid_sizes[2]

        # Compute shared player embeddings once
        player_ids_P = jnp.arange(p, dtype=jnp.int32)
        player_embed_PD = self.player_embed(player_ids_P).astype(x_BPFC.dtype)

        if bidirectional:
            block_mask = None
            block_mask_mouse = None
            block_mask_keyboard = None
        elif kv_cache is None and not teacher_forcing:
            block_mask = self.get_causal_attn_mask(
                num_q_blocks=n_frames,
                num_k_blocks=n_frames,
                q_block_size=num_patches * p,
                k_block_size=num_patches * p,
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
                q_block_size=num_patches * p,
                k_block_size=num_patches * p,
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
            player_embed_PD,
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
                player_embed_PD=player_embed_PD,
                mouse_cond=mouse_cond,
                keyboard_cond=keyboard_cond,
                block_mask=block_mask,
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
                no_kv_backprop_teacher_forcing=no_kv_backprop_teacher_forcing,
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
                    apply_block_fn, static_argnums=(4, 13)
                )(
                    model,  # 0
                    x,  # 1
                    e0,  # 2
                    seq_lens,  # 3
                    grid_sizes,  # 4 -- static
                    freqs,  # 5
                    context,  # 6
                    player_embed_PD,  # 7
                    mouse_cond,  # 8
                    keyboard_cond,  # 9
                    block_mask=block_mask,  # 10
                    block_mask_mouse=block_mask_mouse,  # 11
                    block_mask_keyboard=block_mask_keyboard,  # 12
                    num_frame_per_block=self.num_frame_per_block,  # 13 -- static
                    current_start=current_start,  # 14
                    use_action_module=action_module,  # 15
                    kv_cache=kv_cache,  # 16
                    kv_cache_mouse=kv_cache_mouse,  # 17
                    kv_cache_keyboard=kv_cache_keyboard,  # 18
                )
                return x, kv_cache, kv_cache_mouse, kv_cache_keyboard

            x_BPFC, new_kv_cache, new_kv_cache_mouse, new_kv_cache_keyboard = layers(
                self.blocks,
                kv_cache,
                kv_cache_mouse,
                kv_cache_keyboard,
                action_module_zipper,
                x_BPFC,
            )
        else:

            @nnx.scan(in_axes=(0, 0, nnx.Carry), out_axes=(nnx.Carry), unroll=False)
            def layers(model, action_module, x):
                x, _, _, _ = nnx.remat(apply_block_fn, static_argnums=(4, 13))(
                    model,  # 0
                    x,  # 1
                    e0,  # 2
                    seq_lens,  # 3
                    grid_sizes,  # 4 -- static
                    freqs,  # 5
                    context,  # 6
                    player_embed_PD,  # 7
                    mouse_cond,  # 8
                    keyboard_cond,  # 9
                    block_mask=block_mask,  # 10
                    block_mask_mouse=block_mask_mouse,  # 11
                    block_mask_keyboard=block_mask_keyboard,  # 12
                    num_frame_per_block=self.num_frame_per_block,  # 13 -- static
                    current_start=current_start,  # 14
                    use_action_module=action_module,  # 15
                    kv_cache=kv_cache,  # 16
                    kv_cache_mouse=kv_cache_mouse,  # 17
                    kv_cache_keyboard=kv_cache_keyboard,  # 18
                )
                return x

            x_BPFC = layers(self.blocks, action_module_zipper, x_BPFC)
            new_kv_cache, new_kv_cache_mouse, new_kv_cache_keyboard = None, None, None

        x_BPFC = self.head(
            x_BPFC,
            rearrange(
                e_BD,
                "(b p f) (r d) -> b p f r d",
                b=t_BPT.shape[0],
                p=t_BPT.shape[1],
                r=1,
            ),
        )
        x = self.unpatchify(x_BPFC, grid_sizes)
        return x, new_kv_cache, new_kv_cache_mouse, new_kv_cache_keyboard

    def unpatchify(self, x, grid_sizes):
        f, h, w = grid_sizes
        p1, p2, p3 = self.patch_size
        x = rearrange(
            x,
            "b p f (h w) (p1 p2 p3 c) -> b p (f p1) (h p2) (w p3) c",
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
        num_players=2,
    ):
        assert self.local_attn_size != -1
        frames_block_size = (
            self.local_attn_size * latent_height * latent_width * num_players
        )
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
                            num_players * batch_size * latent_height * latent_width,
                            self.local_attn_size,
                            actions_head_num,
                            mouse_head_dim,
                        ),
                        dtype=dtype,
                    ),
                    jnp.zeros(
                        (
                            batch_size * num_players * latent_height * latent_width,
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
                            num_players * batch_size,
                            self.local_attn_size,
                            actions_head_num,
                            keyboard_head_dim,
                        ),
                        dtype=dtype,
                    ),
                    jnp.zeros(
                        (
                            batch_size * num_players,
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
