import functools

import jax
import jax.experimental
import jax.experimental.multihost_utils
import jax.numpy as jnp
import orbax.checkpoint as ocp
from einops import rearrange, repeat
from flax import nnx

import src.utils.sharding as sharding_utils
from src.models.kv_cache import KVCacheDict
from src.runners.base_mp_runner import BaseMPRunner, get_model_inputs
from src.runners.base_trainer import BaseTrainer, tree_l2_norm
from src.utils.config import instantiate_from_config
from src.utils.multiplayer import handle_multiplayer_input
from src.utils.rollout import flow_prediction_to_x0, left_repeat_padding


class Trainer(BaseTrainer, BaseMPRunner):
    test_loss_enabled = False

    def __init__(
        self,
        use_grad_norm,
        generator_lr_scheduler_config,
        fake_lr_scheduler_config,
        generator_optimizer_config,
        fake_optimizer_config,
        causal_checkpoint_path,
        bidirectional_checkpoint_path,
        no_kv_backprop_teacher_forcing=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.use_grad_norm = use_grad_norm
        self.generator_lr_scheduler_config = generator_lr_scheduler_config
        self.fake_lr_scheduler_config = fake_lr_scheduler_config
        self.generator_optimizer_config = generator_optimizer_config
        self.fake_optimizer_config = fake_optimizer_config
        self.causal_checkpoint_path = causal_checkpoint_path
        self.bidirectional_checkpoint_path = bidirectional_checkpoint_path
        self.no_kv_backprop_teacher_forcing = no_kv_backprop_teacher_forcing

        self.generator_lr_scheduler = instantiate_from_config(
            self.generator_lr_scheduler_config
        )
        self.fake_lr_scheduler = instantiate_from_config(self.fake_lr_scheduler_config)

        (
            self.generator_optimizer_graph,
            self.generator_optimizer_state,
            self.optimizer_sharding,
        ) = self.build_optimizer_from_pretrained_model(
            pretrained_model_path=self.causal_checkpoint_path,
            network_config=self.network_config,
            optimizer_config=self.generator_optimizer_config,
            lr_scheduler=self.generator_lr_scheduler,
            rngs=self.rngs,
            mesh=self.mesh,
            repl_sharding=self.repl_sharding,
        )
        (
            self.fake_optimizer_graph,
            self.fake_optimizer_state,
            self.optimizer_sharding,
        ) = self.build_optimizer_from_pretrained_model(
            pretrained_model_path=self.bidirectional_checkpoint_path,
            network_config=self.network_config,
            optimizer_config=self.fake_optimizer_config,
            lr_scheduler=self.fake_lr_scheduler,
            rngs=self.rngs,
            mesh=self.mesh,
            repl_sharding=self.repl_sharding,
        )

        def get_model(rngs):
            nnx_rngs = nnx.Rngs(params=rngs)
            model = instantiate_from_config(self.network_config, rngs=nnx_rngs)
            return nnx.split(model)

        _, state_shape = jax.eval_shape(functools.partial(get_model, rngs=self.rngs))
        self.model_sharding = sharding_utils.apply_sharding(state_shape, self.mesh)
        self.real_model_graph, model_state = jax.jit(
            get_model, out_shardings=(self.repl_sharding, self.model_sharding)
        )(self.rngs)
        self.real_model_state = self.pretrained_checkpointer.restore(
            self.bidirectional_checkpoint_path, model_state
        )

    def _restore_step(self, restore_step):
        state_restored = self.ckpt_mngr.restore(
            restore_step,
            args=ocp.args.Composite(
                generator_optimizer_state=ocp.args.StandardRestore(
                    self.generator_optimizer_state
                ),
                fake_optimizer_state=ocp.args.StandardRestore(
                    self.fake_optimizer_state
                ),
            ),
        )
        self.generator_optimizer_state = state_restored.generator_optimizer_state
        self.fake_optimizer_state = state_restored.fake_optimizer_state

    def _compile_step_functions(self):
        self.p_generator_train_step = jax.jit(
            sf_generator_train_step,
            in_shardings=(
                self.optimizer_sharding,  # generator opt state
                self.model_sharding,  # fake model state
                self.model_sharding,  # real model state
                self.repl_sharding,  # vae_state
                self.repl_sharding,  # clip_state
                self.ddp_sharding,
                self.ddp_sharding,
                self.ddp_sharding,
                self.repl_sharding,
            ),
            out_shardings=(
                self.optimizer_sharding,
                self.repl_sharding,
                self.repl_sharding,
            ),
            static_argnames=(
                "vae_graph",
                "clip_graph",
                "generator_opt_graph",
                "fake_model_graph",
                "real_model_graph",
                "mesh",
                "no_kv_backprop_teacher_forcing",
                "multiplayer_method",
                "use_grad_norm",
                "left_action_padding",
            ),
            donate_argnums=(0,),
        )

        self.p_fake_train_step = jax.jit(
            sf_fake_train_step,
            in_shardings=(
                self.optimizer_sharding,  # fake opt state
                self.model_sharding,  # generator model state
                self.repl_sharding,  # vae_state
                self.repl_sharding,  # clip_state
                self.ddp_sharding,
                self.ddp_sharding,
                self.ddp_sharding,
                self.repl_sharding,
            ),
            out_shardings=(
                self.optimizer_sharding,
                self.repl_sharding,
                self.repl_sharding,
            ),
            static_argnames=(
                "vae_graph",
                "clip_graph",
                "fake_opt_graph",
                "generator_model_graph",
                "mesh",
                "multiplayer_method",
                "left_action_padding",
            ),
            donate_argnums=(0,),
        )

    def _train_step(
        self,
        i,
        vae_graph,
        vae_state,
        clip_graph,
        clip_state,
        video,
        actions_mouse,
        actions_keyboard,
        rngs,
    ):
        metric_dict = {}

        # train generator every 5 steps
        if i % 5 == 0:
            fake_opt = nnx.merge(self.fake_optimizer_graph, self.fake_optimizer_state)
            fake_model_graph, fake_model_state = nnx.split(fake_opt.model)

            self.generator_optimizer_state, generator_metric_dict, rngs = (
                self.p_generator_train_step(
                    self.generator_optimizer_state,
                    self.generator_optimizer_graph,
                    fake_model_state,
                    fake_model_graph,
                    self.real_model_state,
                    self.real_model_graph,
                    vae_state,
                    vae_graph,
                    clip_state,
                    clip_graph,
                    video,
                    actions_mouse,
                    actions_keyboard,
                    rngs,
                    self.mesh,
                    self.no_kv_backprop_teacher_forcing,
                    self.left_action_padding,
                    self.multiplayer_method,
                    self.use_grad_norm,
                )
            )

            metric_dict.update(generator_metric_dict)

        generator_opt = nnx.merge(
            self.generator_optimizer_graph, self.generator_optimizer_state
        )
        generator_model_graph, generator_model_state = nnx.split(generator_opt.model)
        self.fake_optimizer_state, fake_metric_dict, rngs = self.p_fake_train_step(
            self.fake_optimizer_state,
            self.fake_optimizer_graph,
            generator_model_state,
            generator_model_graph,
            vae_state,
            vae_graph,
            clip_state,
            clip_graph,
            video,
            actions_mouse,
            actions_keyboard,
            rngs,
            self.mesh,
            self.left_action_padding,
            self.multiplayer_method,
        )
        metric_dict.update(fake_metric_dict)

        return metric_dict, rngs

    def _test_step(
        self,
        vae_graph,
        vae_state,
        clip_graph,
        clip_state,
        video,
        actions_mouse,
        actions_keyboard,
        rngs,
    ):
        raise NotImplementedError("Test loss is not implemented for SF trainer")

    def _update_metric_summary(self, summary, step):
        summary["generator_learning_rate"] = self.generator_lr_scheduler(step)
        summary["fake_learning_rate"] = self.fake_lr_scheduler(step)

    def _save_checkpoint(self, step):
        self.ckpt_mngr.save(
            step,
            args=ocp.args.Composite(
                generator_optimizer_state=ocp.args.StandardSave(
                    self.generator_optimizer_state
                ),
                fake_optimizer_state=ocp.args.StandardSave(self.fake_optimizer_state),
            ),
        )

    def _get_model_optimizer(self):
        return nnx.merge(self.generator_optimizer_graph, self.generator_optimizer_state)

    def _evaluate(
        self,
        model_state,
        model_graph,
        vae_state,
        vae_graph,
        clip_state,
        clip_graph,
        video,
        mouse_actions,
        keyboard_actions,
        real_lengths,
        eval_dir,
        mesh,
        left_action_padding,
        num_denoising_steps=None,
    ):
        bidirectional = False
        return self.evaluate_mp(
            bidirectional,
            model_state,
            model_graph,
            vae_state,
            vae_graph,
            clip_state,
            clip_graph,
            video,
            mouse_actions,
            keyboard_actions,
            real_lengths,
            eval_dir,
            mesh,
            left_action_padding,
            num_denoising_steps,
        )


def self_forcing_rollout(
    model,
    cond_concat_BPFHWC,
    visual_context_BPFD,
    mouse_actions_BPFD,
    keyboard_actions_BPFD,
    rngs,
    mesh,
    left_action_padding,
):
    mouse_actions_BPFD = left_repeat_padding(
        mouse_actions_BPFD, left_action_padding, axis=2
    )
    keyboard_actions_BPFD = left_repeat_padding(
        keyboard_actions_BPFD, left_action_padding, axis=2
    )

    F = cond_concat_BPFHWC.shape[2]

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
    denoising_timesteps = jnp.array(
        [1000.0, 750.0, 500.0, 250.0, 0.0], dtype=jnp.float32
    )
    sigma_t = jnp.array([1.0, 0.75, 0.5, 0.25, 0.0], dtype=jnp.float32)
    num_denoising_steps = 4

    rngs, last_timestep_rng = jax.random.split(rngs)
    last_timestep = jax.random.randint(
        last_timestep_rng, shape=(), minval=0, maxval=num_denoising_steps
    )

    # === Inner scan (denoising steps) ===
    @nnx.scan(in_axes=(2, 2, 2, None, nnx.Carry), out_axes=(2, 2, nnx.Carry))
    def denoise_frame(
        cond_concat_BPFHWC, mouse_actions_BPFD, keyboard_actions_BPFD, _model, carry
    ):
        rngs, kv_cache, frame_index = carry
        rngs, step_rng = jax.random.split(rngs)
        B, P, F, H, W, _ = cond_concat_BPFHWC.shape
        latent_channels = _model.out_dim
        noise_shape = (B, P, F, H, W, latent_channels)
        frame = jax.random.normal(step_rng, noise_shape, dtype=jnp.bfloat16)
        previous_frame = frame

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
                rngs, rng_noise = jax.random.split(rngs)
                frame = jax.lax.cond(
                    exit,
                    lambda: x_start_f32,
                    lambda: (1 - sigma_t[i + 1]) * x_start_f32
                    + sigma_t[i + 1]
                    * jax.random.normal(
                        rng_noise, x_start_f32.shape, dtype=jnp.float32
                    ),
                )
                return frame.astype(jnp.bfloat16), rngs

            skip = i > last_timestep
            exit = i == last_timestep

            # only update frame if it is within our bounds.
            new_frame, rngs = nnx.cond(
                skip,
                lambda x, y, z, h, rngs: (x, rngs),
                run_step,
                frame,
                t,
                exit,
                _model,
                rngs,
            )
            previous_frame = nnx.cond(
                skip, lambda x, y: y, lambda x, y: x, frame, previous_frame
            )
            frame = new_frame

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
        return previous_frame, frame, (rngs, new_kv_cache, frame_index + 1)

    kv_cache = model.initialize_kv_cache(B, H // 2, W // 2, num_players=P)
    previous_frame_BPFRHWC, final_frame_BPFRHWC, (rngs, _, _) = denoise_frame(
        repeat(cond_concat_BPFHWC, "b p f h w c -> b p f r h w c", r=1),
        mouse_actions_BPFWD,
        keyboard_actions_BPFWD,
        model,
        (rngs, kv_cache, 0),
    )
    previous_frame_BPFHWC = rearrange(
        previous_frame_BPFRHWC, "b p f r h w c -> b p (f r) h w c"
    )
    final_frame_BPFHWC = rearrange(
        final_frame_BPFRHWC, "b p f r h w c -> b p (f r) h w c"
    )
    last_timestep = jax.lax.dynamic_index_in_dim(
        denoising_timesteps, last_timestep, keepdims=False
    )

    return final_frame_BPFHWC, previous_frame_BPFHWC, last_timestep, rngs


# train the generator
def sf_generator_train_step(
    generator_opt_state,
    generator_opt_graph,
    fake_model_state,
    fake_model_graph,
    real_model_state,
    real_model_graph,
    vae_state,
    vae_graph,
    clip_state,
    clip_graph,
    video_BPFHWC,
    mouse_actions_BPFD,
    keyboard_actions_BPFD,
    rngs,
    mesh,
    no_kv_backprop_teacher_forcing,
    left_action_padding,
    multiplayer_method="multiplayer_attn",
    use_grad_norm=False,
):
    generator_opt = nnx.merge(generator_opt_graph, generator_opt_state)
    real_model = nnx.merge(real_model_graph, real_model_state)
    fake_model = nnx.merge(fake_model_graph, fake_model_state)
    vae_model = nnx.merge(vae_graph, vae_state)
    clip_model = nnx.merge(clip_graph, clip_state)
    generator_model = generator_opt.model

    def loss_fn(m, rm, fm, rngs):
        _frames, cond_concat_BPFHWC, visual_context_BPFD = get_model_inputs(
            video_BPFHWC, clip_model, vae_model
        )

        # Apply multiplayer preprocessing
        (
            cond_concat_BPFHWC,
            mouse_actions_BPFD_processed,
            keyboard_actions_BPFD_processed,
            visual_context_BPFD,
            _,
        ) = handle_multiplayer_input(
            cond_concat_BPFHWC,
            mouse_actions_BPFD,
            keyboard_actions_BPFD,
            multiplayer_method,
            visual_context_BPFD,
            video_BPFHWC=None,
        )

        # 1. Self-forcing rollout
        final_frame_BPFHWC, previous_frame_BPFHWC, last_timestep, rngs = (
            self_forcing_rollout(
                m,
                cond_concat_BPFHWC,
                visual_context_BPFD,
                mouse_actions_BPFD_processed,
                keyboard_actions_BPFD_processed,
                rngs,
                mesh,
                left_action_padding,
            )
        )

        final_frame_BPFHWC = jax.lax.stop_gradient(final_frame_BPFHWC)
        previous_frame_BPFHWC = jax.lax.stop_gradient(previous_frame_BPFHWC)

        B, P, F, _, _, _ = final_frame_BPFHWC.shape
        t_final = jnp.zeros((B, P, F), dtype=jnp.int32)
        t_previous = last_timestep * jnp.ones((B, P, F), dtype=jnp.int32)
        sigma_previous_BPFHWC = t_previous[:, :, :, None, None, None] / 1000

        t = jnp.concatenate([t_final, t_previous], axis=2)
        generator_frames_BPFHWC = jnp.concatenate(
            [final_frame_BPFHWC, previous_frame_BPFHWC], axis=2
        )

        padded_mouse_actions_BPFD = left_repeat_padding(
            mouse_actions_BPFD_processed, left_action_padding, axis=2
        )
        padded_keyboard_actions_BPFD = left_repeat_padding(
            keyboard_actions_BPFD_processed, left_action_padding, axis=2
        )

        v, _, _, _ = m(
            generator_frames_BPFHWC.astype(jnp.bfloat16),
            t.astype(jnp.int32),
            visual_context=visual_context_BPFD.astype(jnp.bfloat16),
            cond_concat=jnp.concatenate(
                [cond_concat_BPFHWC, cond_concat_BPFHWC], axis=2
            ).astype(jnp.bfloat16),
            mouse_cond=padded_mouse_actions_BPFD.astype(jnp.bfloat16),
            keyboard_cond=padded_keyboard_actions_BPFD.astype(jnp.bfloat16),
            bidirectional=False,
            teacher_forcing=True,
            no_kv_backprop_teacher_forcing=no_kv_backprop_teacher_forcing,
            mesh=mesh,
        )
        v_previous = v[:, :, v.shape[2] // 2 :]
        generator_frames_previous = generator_frames_BPFHWC[
            :, :, generator_frames_BPFHWC.shape[2] // 2 :
        ]
        x0 = flow_prediction_to_x0(
            v_previous.astype(jnp.float32),
            generator_frames_previous.astype(jnp.float32),
            sigma_previous_BPFHWC,
        )

        # --- Add noise ---
        rngs, t_rng = jax.random.split(rngs)
        t = repeat(
            jax.random.randint(t_rng, shape=(B,), minval=0, maxval=1001),
            "b -> b p f",
            f=F,
            p=P,
        )
        t = t.astype(jnp.int32)
        sigma_BPFHWC = t[:, :, :, None, None, None] / 1000

        rngs, noise_rng = jax.random.split(rngs)
        noise = jax.random.normal(noise_rng, x0.shape)
        x0_noised = (1 - sigma_BPFHWC) * x0 + sigma_BPFHWC * noise

        # --- Real model ---
        real_flow, _, _, _ = rm(
            x0_noised.astype(jnp.bfloat16),
            t.astype(jnp.int32),
            visual_context=visual_context_BPFD.astype(jnp.bfloat16),
            cond_concat=cond_concat_BPFHWC.astype(jnp.bfloat16),
            mouse_cond=padded_mouse_actions_BPFD.astype(jnp.bfloat16),
            keyboard_cond=padded_keyboard_actions_BPFD.astype(jnp.bfloat16),
            bidirectional=True,
            mesh=mesh,
        )
        real_x0 = flow_prediction_to_x0(
            real_flow.astype(jnp.float32), x0_noised.astype(jnp.float32), sigma_BPFHWC
        )

        # --- Fake model ---
        fake_flow, _, _, _ = fm(
            x0_noised.astype(jnp.bfloat16),
            t.astype(jnp.int32),
            visual_context=visual_context_BPFD.astype(jnp.bfloat16),
            cond_concat=cond_concat_BPFHWC.astype(jnp.bfloat16),
            mouse_cond=padded_mouse_actions_BPFD.astype(jnp.bfloat16),
            keyboard_cond=padded_keyboard_actions_BPFD.astype(jnp.bfloat16),
            bidirectional=True,
            mesh=mesh,
        )
        fake_x0 = flow_prediction_to_x0(
            fake_flow.astype(jnp.float32), x0_noised.astype(jnp.float32), sigma_BPFHWC
        )

        grad = fake_x0 - real_x0

        if use_grad_norm:
            weighting_factor = x0 - real_x0
            normalizer = jnp.mean(
                jnp.abs(weighting_factor), axis=(1, 2, 3, 4, 5), keepdims=True
            )
            normalizer = jnp.maximum(normalizer, 1e-8)  # Avoid division by zero
            grad = grad / normalizer

        # Always sanitize gradients to prevent NaN propagation
        grad = jnp.nan_to_num(grad, nan=0.0, posinf=10.0, neginf=-10.0)
        grad = jnp.clip(grad, -10.0, 10.0)

        # --- DMD Loss ---
        dmd_loss = 0.5 * jnp.mean(
            jnp.square(
                x0.astype(jnp.float32)
                - jax.lax.stop_gradient(
                    x0.astype(jnp.float32) - grad.astype(jnp.float32)
                )
            )
        )
        return dmd_loss, rngs

    (loss, rngs), grads = nnx.value_and_grad(loss_fn, has_aux=True)(
        generator_model, real_model, fake_model, rngs
    )
    generator_opt.update(grads)
    _, new_generator_opt_state = nnx.split(generator_opt)

    return (
        new_generator_opt_state,
        {
            "generator_loss": loss,
            "generator_grad_norm": tree_l2_norm(grads),
            "generator_param_norm": tree_l2_norm(new_generator_opt_state["model"]),
        },
        rngs,
    )


# train the fake model
def sf_fake_train_step(
    fake_opt_state,
    fake_opt_graph,
    generator_model_state,
    generator_model_graph,
    vae_state,
    vae_graph,
    clip_state,
    clip_graph,
    video_BPFHWC,
    mouse_actions_BPFD,
    keyboard_actions_BPFD,
    rngs,
    mesh,
    left_action_padding,
    multiplayer_method="multiplayer_attn",
):
    fake_opt = nnx.merge(fake_opt_graph, fake_opt_state)
    generator_model = nnx.merge(generator_model_graph, generator_model_state)
    vae_model = nnx.merge(vae_graph, vae_state)
    clip_model = nnx.merge(clip_graph, clip_state)
    fake_model = fake_opt.model

    def loss_fn(fake_model, generator_model, rngs):
        _frames, cond_concat_BPFHWC, visual_context_BPFD = get_model_inputs(
            video_BPFHWC, clip_model, vae_model
        )

        # Apply multiplayer preprocessing
        (
            cond_concat_BPFHWC,
            mouse_actions_BPFD_processed,
            keyboard_actions_BPFD_processed,
            visual_context_BPFD,
            _,
        ) = handle_multiplayer_input(
            cond_concat_BPFHWC,
            mouse_actions_BPFD,
            keyboard_actions_BPFD,
            multiplayer_method,
            visual_context_BPFD,
            video_BPFHWC=None,
        )

        # make synthetic data
        generated_frame_BPFHWC, _, _, rngs = self_forcing_rollout(
            generator_model,
            cond_concat_BPFHWC,
            visual_context_BPFD,
            mouse_actions_BPFD_processed,
            keyboard_actions_BPFD_processed,
            rngs,
            mesh,
            left_action_padding,
        )

        F = generated_frame_BPFHWC.shape[2]

        generated_frame_BPFHWC = jax.lax.stop_gradient(generated_frame_BPFHWC)

        B, P, F, _, _, _ = generated_frame_BPFHWC.shape
        rngs, t_rng = jax.random.split(rngs)
        t_BPF = repeat(
            jax.random.randint(t_rng, shape=(B,), minval=0, maxval=1001).astype(
                jnp.int32
            ),
            "b -> b p f",
            f=F,
            p=P,
        )
        sigma_BPFHWC = t_BPF[:, :, :, None, None, None] / 1000

        rngs, noise_rng = jax.random.split(rngs)
        noise = jax.random.normal(noise_rng, generated_frame_BPFHWC.shape)
        final_frame_BPFHWC = (
            1 - sigma_BPFHWC
        ) * generated_frame_BPFHWC + sigma_BPFHWC * noise

        padded_mouse_actions_BPFD = left_repeat_padding(
            mouse_actions_BPFD_processed, left_action_padding, axis=2
        )
        padded_keyboard_actions_BPFD = left_repeat_padding(
            keyboard_actions_BPFD_processed, left_action_padding, axis=2
        )

        v, _, _, _ = fake_model(
            final_frame_BPFHWC.astype(jnp.bfloat16),
            t_BPF.astype(jnp.int32),
            visual_context=visual_context_BPFD.astype(jnp.bfloat16),
            cond_concat=cond_concat_BPFHWC.astype(jnp.bfloat16),
            mouse_cond=padded_mouse_actions_BPFD.astype(jnp.bfloat16),
            keyboard_cond=padded_keyboard_actions_BPFD.astype(jnp.bfloat16),
            bidirectional=True,
            mesh=mesh,
        )
        target_velocity = noise - generated_frame_BPFHWC
        flow_matching_loss = ((v - target_velocity.astype(jnp.float32)) ** 2).mean()

        flow_matching_loss = jnp.nan_to_num(
            flow_matching_loss, nan=0.0, posinf=1e6, neginf=0.0
        )

        return flow_matching_loss, rngs

    (loss, rngs), grads = nnx.value_and_grad(loss_fn, has_aux=True)(
        fake_model, generator_model, rngs
    )
    fake_opt.update(grads)
    _, new_fake_opt_state = nnx.split(fake_opt)

    return (
        new_fake_opt_state,
        {
            "fake_loss": loss,
            "fake_grad_norm": tree_l2_norm(grads),
            "fake_param_norm": tree_l2_norm(new_fake_opt_state["model"]),
        },
        rngs,
    )
