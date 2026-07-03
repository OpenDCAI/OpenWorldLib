
import jax
import jax.numpy as jnp
from einops import rearrange, repeat
from flax import nnx

from src.models import wan_vae
from src.models.kv_cache import KVCacheDict
from src.utils.multiplayer import handle_multiplayer_input, handle_multiplayer_output


def flow_match_inference_timesteps(
    num_inference_steps,
    *,
    timestep_shift=5.0,
    sigma_min=0.0,
    sigma_max=1.0,
    num_train_timesteps=1000,
    denoising_strength=1.0,
    extra_one_step=True,
):
    """Create FlowMatch-style timesteps (JAX) with a trailing 0.0.

    Mirrors the relevant parts of:
    - FlowMatchScheduler.set_timesteps(...)

    Hardcoded-by-default to the user's requested settings:
    - timestep_shift=5.0
    - sigma_min=0.0
    - extra_one_step=True

    Returns:
        jnp.ndarray with shape (num_inference_steps + 1,)
    """
    sigma_start = sigma_min + (sigma_max - sigma_min) * denoising_strength

    # With extra_one_step=True, mimic:
    # sigmas = linspace(sigma_start, sigma_min, N+1)[:-1]  -> length N
    if extra_one_step:
        sigmas = jnp.linspace(
            sigma_start, sigma_min, num_inference_steps + 1, dtype=jnp.float32
        )[:-1]
    else:
        sigmas = jnp.linspace(
            sigma_start, sigma_min, num_inference_steps, dtype=jnp.float32
        )

    # shift: sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
    sigmas = timestep_shift * sigmas / (1.0 + (timestep_shift - 1.0) * sigmas)

    timesteps = sigmas * jnp.asarray(num_train_timesteps, dtype=jnp.float32)

    # Always append an explicit 0.0 so the length is N+1.
    return jnp.concatenate([timesteps, jnp.zeros((1,), dtype=timesteps.dtype)], axis=0)


def change_tensor_range(image, old_range, new_range, dtype=None):
    if dtype is None:
        dtype = image.dtype
    """
    old_range: [min, max]
    new_range: [min, max]
    """
    image = (image - old_range[0]) / (old_range[1] - old_range[0])
    image = (image * (new_range[1] - new_range[0])) + new_range[0]
    image = jnp.clip(image, new_range[0], new_range[1])
    return image.astype(jnp.uint8)


def left_repeat_padding(x, pad, axis):
    if axis == 1:
        return jnp.concatenate([x[:, 0:1].repeat(pad, axis=1), x], axis=1)
    elif axis == 2:
        return jnp.concatenate([x[:, :, 0:1].repeat(pad, axis=2), x], axis=2)
    else:
        raise ValueError(f"Invalid axis: {axis}")


def fully_denoise_frame_bidirectional_multiplayer(
    graph,
    state,
    visual_context_BPFD,
    cond_concat_BPFHWC,
    frame_noise_BPFHWC,
    mouse_actions_BPTD,
    keyboard_actions_BPTD,
    rngs,
    mesh,
    left_action_padding,
    num_denoising_steps=None,
):
    """
    Fully denoise a frame using bidirectional multiplayer model.


    Returns:
        Otherwise: (final_frame, intermediate_latents_dict, rngs)
    """
    model = nnx.merge(graph, state)
    if num_denoising_steps is None:
        num_denoising_steps = 10
        inference_timesteps = jnp.array(
            [1000.0, 937.0, 869.0, 795.0, 714.0, 625.0, 526.0, 416.0, 294.0, 156.0, 0.0]
        )
    else:
        inference_timesteps = flow_match_inference_timesteps(num_denoising_steps)
    sigma_t = inference_timesteps / 1000
    B, P, F, _, _, _ = frame_noise_BPFHWC.shape

    padded_mouse_actions_BPTD = left_repeat_padding(
        mouse_actions_BPTD, left_action_padding, axis=2
    )
    padded_keyboard_actions_BPTD = left_repeat_padding(
        keyboard_actions_BPTD, left_action_padding, axis=2
    )

    frame = frame_noise_BPFHWC

    for i in range(num_denoising_steps):

        v, _, _, _ = model(
            frame.astype(jnp.bfloat16),
            inference_timesteps[i] * jnp.ones((B, P, F), dtype=jnp.int32),
            visual_context_BPFD.astype(jnp.bfloat16),
            cond_concat_BPFHWC.astype(jnp.bfloat16),
            padded_mouse_actions_BPTD.astype(jnp.bfloat16),
            padded_keyboard_actions_BPTD.astype(jnp.bfloat16),
            mesh=mesh,
            bidirectional=True,
        )
        x_start_f32 = flow_prediction_to_x0(
            v.astype(jnp.float32), frame.astype(jnp.float32), sigma_t[i]
        )
        is_last_step = i == num_denoising_steps - 1
        rngs, step_rng = jax.random.split(rngs)
        x_pred = jax.lax.cond(
            is_last_step,
            lambda: x_start_f32,
            lambda: (1 - sigma_t[i + 1]) * x_start_f32
            + sigma_t[i + 1]
            * jax.random.normal(step_rng, x_start_f32.shape, dtype=jnp.float32),
        )
        frame = x_pred.astype(frame.dtype)

    return frame, rngs


def flow_prediction_to_x0(
    flow_pred,  # (B, F, H, W, C)
    x_t,  # (B, F, H, W, C)
    sigma,
):
    return x_t - sigma * flow_pred


def perform_multiplayer_rollout(
    model,
    cond_concat_BPFHWC,
    visual_context_BPFD,
    mouse_actions_BPFD,
    keyboard_actions_BPFD,
    rngs,
    mesh,
    left_action_padding,
    num_denoising_steps=None,
):
    mouse_actions_BPFD = left_repeat_padding(
        mouse_actions_BPFD, left_action_padding, axis=2
    )
    keyboard_actions_BPFD = left_repeat_padding(
        keyboard_actions_BPFD, left_action_padding, axis=2
    )
    P, F = cond_concat_BPFHWC.shape[1], cond_concat_BPFHWC.shape[2]

    mouse_actions_BPFWD = []
    keyboard_actions_BPFWD = []

    for i in range(F):
        mouse_actions_BPFWD.append(mouse_actions_BPFD[:, :, (i * 4) : (i * 4) + 12, :])
        keyboard_actions_BPFWD.append(
            keyboard_actions_BPFD[:, :, (i * 4) : (i * 4) + 12, :]
        )

    mouse_actions_BPFWD = jnp.stack(mouse_actions_BPFWD, axis=2)
    keyboard_actions_BPFWD = jnp.stack(keyboard_actions_BPFWD, axis=2)

    B, P, F, H, W, C = cond_concat_BPFHWC.shape

    # --- Diffusion schedule ---
    if num_denoising_steps is None:
        num_denoising_steps = 4
        denoising_timesteps = jnp.array(
            [1000.0, 750.0, 500.0, 250.0, 0.0], dtype=jnp.float32
        )
    else:
        denoising_timesteps = flow_match_inference_timesteps(num_denoising_steps)
    sigma_t = denoising_timesteps / 1000

    last_timestep = num_denoising_steps - 1

    # === Inner scan (denoising steps) ===
    @nnx.scan(in_axes=(2, 2, 2, None, nnx.Carry), out_axes=(2, nnx.Carry))
    def denoise_frame(
        cond_concat_BPFHWC, mouse_actions_BPFD, keyboard_actions_BPFD, _model, carry
    ):
        rngs, kv_cache, frame_index = carry
        rngs, step_rng = jax.random.split(rngs)
        B, P, F, H, W, _ = cond_concat_BPFHWC.shape
        # Use model's out_dim for correct latent channels (32 for concat_c, 16 otherwise)
        latent_channels = _model.out_dim
        noise_shape = (B, P, F, H, W, latent_channels)
        frame = jax.random.normal(step_rng, noise_shape, dtype=jnp.bfloat16)

        for i in range(num_denoising_steps):
            t = denoising_timesteps[i]

            def run_step(frame, t, exit, _model, rngs):
                v, _, _, _ = _model(
                    frame.astype(jnp.bfloat16),
                    t * jnp.ones((B, P, 1), dtype=jnp.int32),
                    visual_context_BPFD.astype(jnp.bfloat16),
                    cond_concat_BPFHWC.astype(jnp.bfloat16),
                    mouse_actions_BPFD.astype(jnp.bfloat16),
                    keyboard_actions_BPFD.astype(jnp.bfloat16),
                    kv_cache=kv_cache.kv_cache,
                    kv_cache_mouse=kv_cache.kv_cache_mouse,
                    kv_cache_keyboard=kv_cache.kv_cache_keyboard,
                    mesh=mesh,
                    bidirectional=False,
                    current_start=frame_index,
                )
                x_start_f32 = flow_prediction_to_x0(
                    v.astype(jnp.float32), frame.astype(jnp.float32), t / 1000.0
                ).astype(jnp.float32)
                rngs, step_rng = jax.random.split(rngs)
                frame = jax.lax.cond(
                    exit,
                    lambda: x_start_f32,
                    lambda: (1 - sigma_t[i + 1]) * x_start_f32
                    + sigma_t[i + 1]
                    * jax.random.normal(step_rng, x_start_f32.shape, dtype=jnp.float32),
                )
                return frame.astype(jnp.bfloat16), rngs

            exit = i == last_timestep

            # only update frame if it is within our bounds.
            frame, rngs = run_step(frame, t, exit, _model, rngs)

        _, new_kv_cache, new_kv_cache_mouse, new_kv_cache_keyboard = _model(
            frame,
            jnp.zeros((B, P, 1), dtype=jnp.int32),
            visual_context_BPFD.astype(jnp.bfloat16),
            cond_concat_BPFHWC.astype(jnp.bfloat16),
            mouse_actions_BPFD.astype(jnp.bfloat16),
            keyboard_actions_BPFD.astype(jnp.bfloat16),
            kv_cache=kv_cache.kv_cache,
            kv_cache_mouse=kv_cache.kv_cache_mouse,
            kv_cache_keyboard=kv_cache.kv_cache_keyboard,
            mesh=mesh,
            bidirectional=False,
            current_start=frame_index,
        )
        new_kv_cache = KVCacheDict(
            new_kv_cache, new_kv_cache_mouse, new_kv_cache_keyboard
        )
        return frame, (rngs, new_kv_cache, frame_index + 1)

    kv_cache = model.initialize_kv_cache(B, H // 2, W // 2, num_players=P)

    final_frame_BPFRHWC, (rngs, _, _) = denoise_frame(
        repeat(cond_concat_BPFHWC, "b p f h w c -> b p f r h w c", r=1),
        mouse_actions_BPFWD,
        keyboard_actions_BPFWD,
        model,
        (rngs, kv_cache, 0),
    )
    final_frame_BPFHWC = rearrange(
        final_frame_BPFRHWC, "b p f r h w c -> b p (f r) h w c"
    )
    last_timestep = jax.lax.dynamic_index_in_dim(
        denoising_timesteps, last_timestep, keepdims=False
    )
    return final_frame_BPFHWC, last_timestep, rngs


# wrapper around rollout state
def perform_bidirectional_multiplayer_rollout(
    model_graph,
    model_state,
    vae_graph,
    vae_state,
    clip_graph,
    clip_state,
    first_frames_BPHWC,
    mouse_actions_BPTD,
    keyboard_actions_BPTD,
    frames_to_decode,
    mesh,
    left_action_padding,
    multiplayer_method="multiplayer_attn",
    num_denoising_steps=None,
):
    """
    Perform bidirectional multiplayer video rollout with optional intermediate latent collection.


    Returns:
        Otherwise: (decoded_video, intermediate_latents_dict) where latents are in post-multiplayer-transform space
    """
    assert frames_to_decode <= mouse_actions_BPTD.shape[2]

    clip_model = nnx.merge(clip_graph, clip_state)
    vae_model = nnx.merge(vae_graph, vae_state)

    p_vae_encode = jax.jit(vae_model.encode)
    p_vae_decode = jax.jit(vae_model.decode)
    p_clip_encode = jax.jit(clip_model.encode_video)

    p_fully_denoise_frame_bidirectional_multiplayer = jax.jit(
        fully_denoise_frame_bidirectional_multiplayer,
        static_argnames=(
            "graph",
            "mesh",
            "left_action_padding",
            "num_denoising_steps",
        ),
        donate_argnames=("rngs",),
    )

    B, P, H, W, C = first_frames_BPHWC.shape
    num_players = P  # Store original player count for output transformation
    padded_image_BPFHWC = jnp.concat(
        [
            jnp.expand_dims(first_frames_BPHWC, 2),
            jnp.zeros((B, P, frames_to_decode - 1, H, W, C)),
        ],
        axis=2,
    )
    img_cond_BPFHWC = rearrange(
        p_vae_encode(
            rearrange(padded_image_BPFHWC, "b p f h w c -> (b p) f h w c"),
            scale=wan_vae.VAE_SCALE,
        ).astype(jnp.bfloat16),
        "(b p) f h w c -> b p f h w c",
        b=B,
    )
    mask_cond_BPFHWC = (
        jnp.ones(img_cond_BPFHWC.shape, dtype=jnp.bfloat16).at[:, :, 1:].set(0)
    )
    cond_concat_BPFHWC = jnp.concatenate(
        [mask_cond_BPFHWC[:, :, :, :, :, :4], img_cond_BPFHWC], axis=-1
    ).astype(jnp.bfloat16)
    visual_context_BPFD = rearrange(
        p_clip_encode(
            rearrange(
                first_frames_BPHWC[:, :, None], "b p f h w c -> (b p) c f h w"
            ).astype(jnp.bfloat16)
        ),
        "(b p) f d -> b p f d",
        b=B,
    )

    (
        cond_concat_BPFHWC,
        mouse_actions_BPTD,
        keyboard_actions_BPTD,
        visual_context_BPFD,
        _,
    ) = handle_multiplayer_input(
        cond_concat_BPFHWC,
        mouse_actions_BPTD,
        keyboard_actions_BPTD,
        multiplayer_method,
        visual_context_BPFD,
        video_BPFHWC=None,
    )

    rngs = jax.random.PRNGKey(0)

    # Get shape after multiplayer transformation (P may now be 1 for concat methods)
    _, P_eff, F, Hl, Wl, _ = cond_concat_BPFHWC.shape
    # For concat_c, latent channels are doubled (16 * num_players = 32)
    latent_channels = 16 * num_players if multiplayer_method == "concat_c" else 16
    rngs, frame_rng = jax.random.split(rngs)
    frame_noise_BPFHWC = jax.random.normal(
        frame_rng, (B, P_eff, F, Hl, Wl, latent_channels), dtype=jnp.bfloat16
    )

    # Call denoising with intermediate latent collection if requested

    frame, _ = p_fully_denoise_frame_bidirectional_multiplayer(
        model_graph,
        model_state,
        visual_context_BPFD,
        cond_concat_BPFHWC,
        frame_noise_BPFHWC,
        mouse_actions_BPTD,
        keyboard_actions_BPTD,
        rngs,
        mesh=mesh,
        left_action_padding=left_action_padding,
        num_denoising_steps=num_denoising_steps,
    )

    # Reverse multiplayer transformation before VAE decoding
    frame = handle_multiplayer_output(
        frame, multiplayer_method, num_players=num_players
    )

    decoded_frames_BPFHWC = rearrange(
        p_vae_decode(
            rearrange(frame, "b p f h w c -> (b p) f h w c"), scale=wan_vae.VAE_SCALE
        ),
        "(b p) f h w c -> b p f h w c",
        b=B,
    )
    output = change_tensor_range(
        decoded_frames_BPFHWC, [-1, 1], [0, 255], dtype=jnp.uint8
    )

    return output


def left_pad(x_BFHWC, pad_length):
    pad_t = repeat(x_BFHWC[:, 0, :], "b d -> b t d", t=pad_length)
    return jnp.concatenate([pad_t, x_BFHWC], axis=1)


def fully_denoise_frame_bidirectional(
    graph,
    state,
    visual_context_BFD,
    cond_concat_BFHWC,
    frame_noise_BFHWC,
    mouse_actions_BTD,
    keyboard_actions_BTD,
    rngs,
    mesh,
    left_action_padding,
    num_denoising_steps=None,
):
    model = nnx.merge(graph, state)
    if num_denoising_steps is None:
        num_denoising_steps = 10
        inference_timesteps = jnp.array(
            [1000.0, 937.0, 869.0, 795.0, 714.0, 625.0, 526.0, 416.0, 294.0, 156.0, 0.0]
        )
    else:
        inference_timesteps = flow_match_inference_timesteps(num_denoising_steps)
    sigma_t = inference_timesteps / 1000
    B, F, _, _, _ = frame_noise_BFHWC.shape

    padded_mouse_actions_BTD = left_pad(mouse_actions_BTD, left_action_padding)
    padded_keyboard_actions_BTD = left_pad(keyboard_actions_BTD, left_action_padding)

    frame = frame_noise_BFHWC
    for i in range(num_denoising_steps):
        v, _, _, _ = model(
            frame.astype(jnp.bfloat16),
            inference_timesteps[i] * jnp.ones((B, F), dtype=jnp.int32),
            visual_context_BFD.astype(jnp.bfloat16),
            cond_concat_BFHWC.astype(jnp.bfloat16),
            padded_mouse_actions_BTD.astype(jnp.bfloat16),
            padded_keyboard_actions_BTD.astype(jnp.bfloat16),
            mesh=mesh,
            bidirectional=True,
        )
        x_start_f32 = flow_prediction_to_x0(
            v.astype(jnp.float32), frame.astype(jnp.float32), sigma_t[i]
        )
        is_last_step = i == num_denoising_steps - 1
        rngs, step_rng = jax.random.split(rngs)
        x_pred = jax.lax.cond(
            is_last_step,
            lambda: x_start_f32,
            lambda: (1 - sigma_t[i + 1]) * x_start_f32
            + sigma_t[i + 1]
            * jax.random.normal(step_rng, x_start_f32.shape, dtype=jnp.float32),
        )
        frame = x_pred.astype(frame.dtype)
    return frame, rngs


def perform_bidirectional_rollout(
    model_graph,
    model_state,
    vae_graph,
    vae_state,
    clip_graph,
    clip_state,
    first_frames_BHWC,
    mouse_actions_BTD,
    keyboard_actions_BTD,
    frames_to_decode,
    mesh,
    left_action_padding,
    num_denoising_steps=None,
):
    assert frames_to_decode <= mouse_actions_BTD.shape[1]

    clip_model = nnx.merge(clip_graph, clip_state)
    vae_model = nnx.merge(vae_graph, vae_state)
    p_vae_encode = jax.jit(vae_model.encode)
    p_vae_decode = jax.jit(vae_model.decode)
    p_clip_encode = jax.jit(clip_model.encode_video)
    p_fully_denoise_frame_bidirectional = jax.jit(
        fully_denoise_frame_bidirectional,
        static_argnames=("graph", "mesh", "left_action_padding", "num_denoising_steps"),
        donate_argnames=("rngs"),
    )

    B, H, W, C = first_frames_BHWC.shape
    padded_image_BFHWC = jnp.concat(
        [
            jnp.expand_dims(first_frames_BHWC, 1),
            jnp.zeros((B, frames_to_decode - 1, H, W, C)),
        ],
        axis=1,
    )
    img_cond_BFHWC = p_vae_encode(padded_image_BFHWC, scale=wan_vae.VAE_SCALE).astype(
        jnp.bfloat16
    )
    mask_cond_BFHWC = (
        jnp.ones(img_cond_BFHWC.shape, dtype=jnp.bfloat16).at[:, 1:].set(0)
    )
    cond_concat_BFHWC = jnp.concatenate(
        [mask_cond_BFHWC[:, :, :, :, :4], img_cond_BFHWC], axis=-1
    ).astype(jnp.bfloat16)
    visual_context_BFD = p_clip_encode(
        rearrange(first_frames_BHWC[:, None], "b f h w c -> b c f h w").astype(
            jnp.bfloat16
        )
    )

    rngs = jax.random.PRNGKey(0)

    B, F, Hl, Wl, _ = img_cond_BFHWC.shape
    rngs, frame_rng = jax.random.split(rngs)
    frame_noise_BFHWC = jax.random.normal(
        frame_rng, (B, F, Hl, Wl, 16), dtype=jnp.bfloat16
    )
    frame, _ = p_fully_denoise_frame_bidirectional(
        model_graph,
        model_state,
        visual_context_BFD,
        cond_concat_BFHWC,
        frame_noise_BFHWC,
        mouse_actions_BTD,
        keyboard_actions_BTD,
        rngs,
        mesh=mesh,
        left_action_padding=left_action_padding,
        num_denoising_steps=num_denoising_steps,
    )

    decoded_frames_BFHWC = p_vae_decode(frame, scale=wan_vae.VAE_SCALE)
    output = change_tensor_range(
        decoded_frames_BFHWC, [-1, 1], [0, 255], dtype=jnp.uint8
    )
    return output
