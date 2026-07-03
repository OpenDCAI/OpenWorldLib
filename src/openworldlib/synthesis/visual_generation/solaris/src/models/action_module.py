from functools import partial

import einops
import jax
import jax.numpy as jnp
from einops.einops import rearrange
from flax import nnx

from .rope import apply_rotary_emb, get_nd_rotary_pos_embed
from .transformer import WanRMSNorm


class ActionModule(nnx.Module):
    def __init__(
        self,
        mouse_dim_in=2,
        keyboard_dim_in=6,
        hidden_size=128,
        img_hidden_size=1536,
        keyboard_hidden_dim=1024,
        mouse_hidden_dim=1024,
        vae_time_compression_ratio=4,
        windows_size=3,
        heads_num=16,
        patch_size=[1, 2, 2],
        qk_norm=True,
        qkv_bias=False,
        rope_dim_list=[8, 28, 28],
        rope_theta=256,
        mouse_qk_dim_list=[8, 28, 28],
        enable_mouse=True,
        enable_keyboard=True,
        local_attn_size=6,
        blocks=[],
        rngs=nnx.Rngs(0),
        left_action_padding=11,
    ):

        super().__init__()
        self.local_attn_size = local_attn_size
        self.enable_mouse = enable_mouse
        self.enable_keyboard = enable_keyboard

        self.rope_dim_list = rope_dim_list
        self.rope_theta = rope_theta
        if self.enable_keyboard:
            self.keyboard_embed = nnx.Sequential(
                nnx.Linear(
                    keyboard_dim_in,
                    hidden_size,
                    use_bias=True,
                    rngs=rngs,
                    param_dtype=jnp.float32,
                    dtype=jnp.bfloat16,
                ),
                lambda x: nnx.silu(x),
                nnx.Linear(
                    hidden_size,
                    hidden_size,
                    use_bias=True,
                    rngs=rngs,
                    param_dtype=jnp.float32,
                    dtype=jnp.bfloat16,
                ),
            )

        self.mouse_qk_dim_list = mouse_qk_dim_list
        self.heads_num = heads_num
        self.mouse_head_dim = mouse_hidden_dim // heads_num
        self.keyboard_head_dim = keyboard_hidden_dim // heads_num

        self.keyboard_dims = keyboard_dim_in
        if self.enable_mouse:
            c = mouse_hidden_dim
            self.mouse_mlp = nnx.Sequential(
                nnx.Linear(
                    mouse_dim_in * vae_time_compression_ratio * windows_size
                    + img_hidden_size,
                    c,
                    use_bias=True,
                    rngs=rngs,
                    param_dtype=jnp.float32,
                    dtype=jnp.bfloat16,
                ),
                partial(nnx.gelu, approximate=True),
                nnx.Linear(
                    c, c, rngs=rngs, param_dtype=jnp.float32, dtype=jnp.bfloat16
                ),
                nnx.LayerNorm(
                    c,
                    rngs=rngs,
                    epsilon=1e-5,
                    use_bias=True,
                    use_scale=True,
                    param_dtype=jnp.float32,
                    dtype=jnp.float32,
                ),
            )

            head_dim = c // heads_num
            self.t_qkv = nnx.Linear(
                c,
                c * 3,
                use_bias=qkv_bias,
                rngs=rngs,
                param_dtype=jnp.float32,
                dtype=jnp.bfloat16,
            )
            self.img_attn_q_norm = (
                WanRMSNorm(
                    head_dim, eps=1e-6, param_dtype=jnp.float32, dtype=jnp.bfloat16
                )
                if qk_norm
                else lambda x: x
            )
            self.img_attn_k_norm = (
                WanRMSNorm(
                    head_dim, eps=1e-6, param_dtype=jnp.float32, dtype=jnp.bfloat16
                )
                if qk_norm
                else lambda x: x
            )
            self.proj_mouse = nnx.Linear(
                c,
                img_hidden_size,
                use_bias=qkv_bias,
                rngs=rngs,
                param_dtype=jnp.bfloat16,
            )

        if self.enable_keyboard:
            head_dim_key = keyboard_hidden_dim // heads_num
            self.key_attn_q_norm = (
                WanRMSNorm(
                    head_dim_key, eps=1e-6, param_dtype=jnp.float32, dtype=jnp.bfloat16
                )
                if qk_norm
                else lambda x: x
            )
            self.key_attn_k_norm = (
                WanRMSNorm(
                    head_dim_key, eps=1e-6, param_dtype=jnp.float32, dtype=jnp.bfloat16
                )
                if qk_norm
                else lambda x: x
            )

            self.mouse_attn_q = nnx.Linear(
                img_hidden_size,
                keyboard_hidden_dim,
                use_bias=qkv_bias,
                rngs=rngs,
                param_dtype=jnp.float32,
                dtype=jnp.bfloat16,
            )
            self.keyboard_attn_kv = nnx.Linear(
                hidden_size * windows_size * vae_time_compression_ratio,
                keyboard_hidden_dim * 2,
                use_bias=qkv_bias,
                rngs=rngs,
                param_dtype=jnp.float32,
                dtype=jnp.bfloat16,
            )
            self.proj_keyboard = nnx.Linear(
                keyboard_hidden_dim,
                img_hidden_size,
                use_bias=qkv_bias,
                rngs=rngs,
                param_dtype=jnp.float32,
                dtype=jnp.bfloat16,
            )

        self.vae_time_compression_ratio = vae_time_compression_ratio
        self.latents_window_size = windows_size
        self.patch_size = patch_size

    def get_rotary_pos_embed(
        self,
        video_length,
        height,
        width,
        head_dim,
        rope_dim_list=None,
        start_offset=0,
    ):
        target_ndim = 3
        ndim = 5 - 2

        latents_size = [video_length + start_offset, height, width]


        if isinstance(self.patch_size, int):
            assert all(s % self.patch_size == 0 for s in latents_size), (
                f"Latent size(last {ndim} dimensions) should be divisible by patch size({self.patch_size}), "
                f"but got {latents_size}."
            )
            rope_sizes = [s // self.patch_size for s in latents_size]

        else:
            assert all(
                s % self.patch_size[idx] == 0 for idx, s in enumerate(latents_size)
            ), (
                f"Latent size(last {ndim} dimensions) should be divisible by patch size({self.patch_size}), "
                f"but got {latents_size}."
            )
            rope_sizes = [
                s // self.patch_size[idx] for idx, s in enumerate(latents_size)
            ]


        if len(rope_sizes) != target_ndim:
            rope_sizes = [1] * (target_ndim - len(rope_sizes)) + rope_sizes

        if rope_dim_list is None:
            rope_dim_list = [head_dim // target_ndim for _ in range(target_ndim)]
        assert (
            sum(rope_dim_list) == head_dim
        ), "sum(rope_dim_list) should equal to head_dim of attention layer"

        freqs_cos, freqs_sin = get_nd_rotary_pos_embed(
            rope_dim_list,
            rope_sizes,
            theta=self.rope_theta,
            use_real=True,
            theta_rescale_factor=1,
        )
        return (
            freqs_cos[
                -video_length * rope_sizes[1] * rope_sizes[2] // self.patch_size[0] :
            ],
            freqs_sin[
                -video_length * rope_sizes[1] * rope_sizes[2] // self.patch_size[0] :
            ],
        )

    def __call__(
        self,
        x,
        tt,
        th,
        tw,
        mouse_condition=None,
        keyboard_condition=None,
        block_mask_mouse=None,
        block_mask_keyboard=None,
        kv_cache_mouse=None,
        kv_cache_keyboard=None,
        start_frame=0,
        num_frame_per_block=1,
        early_exit=None,
        teacher_forcing=False,
        no_kv_backprop_teacher_forcing=False,
    ):
        new_kv_cache_mouse, new_kv_cache_keyboard = None, None
        B, N_frames, C = keyboard_condition.shape
        _freqs_cos, _freqs_sin = self.get_rotary_pos_embed(
            7500,
            self.patch_size[1],
            self.patch_size[2],
            64,
            self.mouse_qk_dim_list,
            start_offset=0,
        )
        freqs_cis = (_freqs_cos, _freqs_sin)

        assert tt * th * tw == x.shape[1]

        if self.enable_mouse and mouse_condition is not None:
            hidden_states = (
                x.reshape(B, tt, th * tw, x.shape[-1])
                .transpose(0, 2, 1, 3)
                .reshape(B * th * tw, tt, x.shape[-1])
            )
        else:
            hidden_states = x

        frames_window_size = self.vae_time_compression_ratio * self.latents_window_size
        if self.enable_mouse and mouse_condition is not None:
            S = th * tw
            if kv_cache_mouse is not None:
                group_mouse = [
                    mouse_condition[
                        :,
                        i
                        * self.vae_time_compression_ratio : i
                        * self.vae_time_compression_ratio
                        + frames_window_size,
                        :,
                    ]
                    for i in range(num_frame_per_block)
                ]
            else:
                if teacher_forcing:
                    group_mouse = [
                        mouse_condition[
                            :,
                            i
                            * self.vae_time_compression_ratio : i
                            * self.vae_time_compression_ratio
                            + frames_window_size,
                            :,
                        ]
                        for i in range(tt // 2)
                    ]
                    group_mouse = group_mouse + group_mouse
                else:
                    group_mouse = [
                        mouse_condition[
                            :,
                            i
                            * self.vae_time_compression_ratio : i
                            * self.vae_time_compression_ratio
                            + frames_window_size,
                            :,
                        ]
                        for i in range(tt)
                    ]
            group_mouse_btwd = jnp.stack(group_mouse, axis=1)  # w = window_size
            group_mouse_btd = rearrange(
                group_mouse_btwd, "B T W D -> B T (W D)"
            )  # group windows together
            group_mouse_btd = einops.repeat(group_mouse_btd, "B T D -> (B S) T D", S=S)

            group_mouse_btd = jnp.concatenate([hidden_states, group_mouse_btd], axis=-1)
            group_mouse_btd = self.mouse_mlp(group_mouse_btd)
            mouse_qkv = self.t_qkv(group_mouse_btd)
            q_blhd, k_blhd, v_blhd = rearrange(
                mouse_qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num
            )  # BHW F H C

            q_blhd = self.img_attn_q_norm(q_blhd).astype(v_blhd.dtype)
            k_blhd = self.img_attn_k_norm(k_blhd).astype(v_blhd.dtype)

            if teacher_forcing:
                q_blhd = rearrange(q_blhd, "b (r s) n d -> (b r) s n d", r=2)
                k_blhd = rearrange(k_blhd, "b (r s) n d -> (b r) s n d", r=2)
                q_blhd, k_blhd = apply_rotary_emb(
                    q_blhd,
                    k_blhd,
                    freqs_cis,
                    start_offset=start_frame,
                    head_first=False,
                )
                q_blhd = rearrange(q_blhd, "(b r) s n d -> b (r s) n d", r=2)
                k_blhd = rearrange(k_blhd, "(b r) s n d -> b (r s) n d", r=2)
            else:
                q_blhd, k_blhd = apply_rotary_emb(
                    q_blhd,
                    k_blhd,
                    freqs_cis,
                    start_offset=start_frame,
                    head_first=False,
                )

            attn, new_kv_cache_mouse = self.compute_attention_causal(
                q_blhd,
                k_blhd,
                v_blhd,
                repeat_kv=False,
                kv_cache=kv_cache_mouse,
                block_mask=block_mask_mouse,
                teacher_forcing=teacher_forcing,
                no_kv_backprop_teacher_forcing=no_kv_backprop_teacher_forcing,
            )

            attn = einops.rearrange(attn, "(b S) T h d -> b (T S) (h d)", b=B)
            hidden_states = einops.rearrange(x, "(B S) T C -> B (T S) C", B=B)
            attn = self.proj_mouse(attn)
            hidden_states = hidden_states + attn

        if self.enable_keyboard and keyboard_condition is not None:
            keyboard_condition = self.keyboard_embed(keyboard_condition)
            if kv_cache_keyboard is not None:
                group_keyboard = [
                    keyboard_condition[
                        :,
                        self.vae_time_compression_ratio * (i - self.latents_window_size)
                        + frames_window_size : i * self.vae_time_compression_ratio
                        + frames_window_size,
                        :,
                    ]
                    for i in range(num_frame_per_block)
                ]
            else:
                if teacher_forcing:
                    group_keyboard = [
                        keyboard_condition[
                            :,
                            self.vae_time_compression_ratio
                            * (i - self.latents_window_size)
                            + frames_window_size : i * self.vae_time_compression_ratio
                            + frames_window_size,
                            :,
                        ]
                        for i in range(tt // 2)
                    ]
                    group_keyboard = group_keyboard + group_keyboard
                else:
                    group_keyboard = [
                        keyboard_condition[
                            :,
                            self.vae_time_compression_ratio
                            * (i - self.latents_window_size)
                            + frames_window_size : i * self.vae_time_compression_ratio
                            + frames_window_size,
                            :,
                        ]
                        for i in range(tt)
                    ]

            group_keyboard = jnp.stack(group_keyboard, axis=1)
            group_keyboard = group_keyboard.reshape(
                group_keyboard.shape[0], group_keyboard.shape[1], -1
            )

            mouse_q = self.mouse_attn_q(hidden_states)
            keyboard_kv = self.keyboard_attn_kv(group_keyboard)

            B, L, HD = mouse_q.shape
            D = HD // self.heads_num

            q_blhd = mouse_q.reshape(B, L, self.heads_num, D)

            B, L, KHD = keyboard_kv.shape
            k_blhd, v_blhd = jnp.split(
                keyboard_kv.reshape(B, L, 2, self.heads_num, D), 2, axis=2
            )
            k_blhd = k_blhd.squeeze(2)
            v_blhd = v_blhd.squeeze(2)
            q_blhd = self.key_attn_q_norm(q_blhd).astype(v_blhd.dtype)
            k_blhd = self.key_attn_k_norm(k_blhd).astype(v_blhd.dtype)

            S = th * tw
            B, TS, H, D = q_blhd.shape
            T_ = TS // S
            assert S == 880 or S == 1760  # if using mp method

            q_blhd = (
                q_blhd.reshape(B, T_, S, H, D)
                .transpose(0, 2, 1, 3, 4)
                .reshape(B * S, T_, H, D)
            )



            if teacher_forcing:
                q_blhd = rearrange(q_blhd, "b (r s) n d -> (b r) s n d", r=2)
                k_blhd = rearrange(k_blhd, "b (r s) n d -> (b r) s n d", r=2)
                q_blhd, k_blhd = apply_rotary_emb(
                    q_blhd,
                    k_blhd,
                    freqs_cis,
                    start_offset=start_frame,
                    head_first=False,
                )
                q_blhd = rearrange(q_blhd, "(b r) s n d -> b (r s) n d", r=2)
                k_blhd = rearrange(k_blhd, "(b r) s n d -> b (r s) n d", r=2)
            else:
                q_blhd, k_blhd = apply_rotary_emb(
                    q_blhd,
                    k_blhd,
                    freqs_cis,
                    start_offset=start_frame,
                    head_first=False,
                )

            attn, new_kv_cache_keyboard = self.compute_attention_causal(
                q_blhd,
                k_blhd,
                v_blhd,
                repeat_kv=True,
                kv_cache=kv_cache_keyboard,
                block_mask=block_mask_keyboard,
                teacher_forcing=teacher_forcing,
                no_kv_backprop_teacher_forcing=no_kv_backprop_teacher_forcing,
            )

            attn = einops.rearrange(attn, "(B S) T H D -> B (T S) (H D)", S=S)
            attn = self.proj_keyboard(attn)
            hidden_states = hidden_states + attn

        return hidden_states, new_kv_cache_mouse, new_kv_cache_keyboard

    def compute_attention_causal(
        self,
        q_blhd,
        k_blhd,
        v_blhd,
        repeat_kv=False,
        kv_cache=None,
        block_mask=None,
        teacher_forcing=False,
        no_kv_backprop_teacher_forcing=False,
    ):
        if kv_cache is None:
            if teacher_forcing and no_kv_backprop_teacher_forcing:
                Tq = q_blhd.shape[1]
                past_keys, past_values = (
                    k_blhd[:, : Tq // 2, :, :],
                    v_blhd[:, : Tq // 2, :, :],
                )
                present_keys, present_values = (
                    k_blhd[:, Tq // 2 :, :, :],
                    v_blhd[:, Tq // 2 :, :, :],
                )
                past_keys = jax.lax.stop_gradient(past_keys)
                past_values = jax.lax.stop_gradient(past_values)
                k_blhd = jnp.concatenate([past_keys, present_keys], axis=1)
                v_blhd = jnp.concatenate([past_values, present_values], axis=1)
            if repeat_kv:
                assert q_blhd.shape[0] % k_blhd.shape[0] == 0
                repeat_factor = q_blhd.shape[0] // k_blhd.shape[0]
                k_blhd = k_blhd.repeat(repeat_factor, axis=0)
                v_blhd = v_blhd.repeat(repeat_factor, axis=0)
            x = jax.nn.dot_product_attention(q_blhd, k_blhd, v_blhd, mask=block_mask)
            return x, None
        else:
            new_kv_cache = kv_cache.update(k_blhd, v_blhd)
            k = new_kv_cache.k
            v = new_kv_cache.v

            if repeat_kv:
                assert q_blhd.shape[0] % k_blhd.shape[0] == 0
                repeat_factor = q_blhd.shape[0] // k_blhd.shape[0]
                k = k.repeat(repeat_factor, axis=0)
                v = v.repeat(repeat_factor, axis=0)

            kv_len = k.shape[1]

            mask = jnp.arange(kv_len) >= (kv_len - new_kv_cache.length)
            x = jax.nn.dot_product_attention(
                q_blhd,
                k,
                v,
                mask=mask[None, None, :],
            )
            return x, new_kv_cache

    def compute_keyboard_attention(
        self,
        q_blhd,
        k_blhd,
        v_blhd,
    ):
        repeat_factor = q_blhd.shape[0] // k_blhd.shape[0]
        k_blhd = k_blhd.repeat(repeat_factor, axis=0)
        v_blhd = v_blhd.repeat(repeat_factor, axis=0)
        return jax.nn.dot_product_attention(q_blhd, k_blhd, v_blhd)
