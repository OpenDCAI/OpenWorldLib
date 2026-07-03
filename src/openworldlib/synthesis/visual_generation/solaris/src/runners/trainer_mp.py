import jax
import jax.experimental
import jax.experimental.multihost_utils
import jax.numpy as jnp
from absl import logging
from einops import repeat

from src.runners.base_mp_runner import BaseMPRunner, get_model_inputs
from src.runners.base_ssl_trainer import BaseSSLTrainer
from src.utils.multiplayer import handle_multiplayer_input
from src.utils.rollout import left_repeat_padding


class Trainer(BaseSSLTrainer, BaseMPRunner):

    def _load_pretrained_model(self, pretrained_model_path, model_state):
        logging.info(
            "Restoring world model from checkpoint %s for multiplayer_method=%s",
            pretrained_model_path,
            self.multiplayer_method,
        )
        # mp causal model starts from mp bidirectional so it should be strict always.
        if self.multiplayer_method == "concat_c" and self.bidirectional:
            model_state = self.pretrained_checkpointer.restore(
                pretrained_model_path,
                model_state,
                strict=False,
            )
        else:
            model_state = self.pretrained_checkpointer.restore(
                pretrained_model_path,
                model_state,
            )
        return model_state

    def _diffusion_loss_core(
        self,
        model,
        vae_model,
        clip_model,
        video,
        mouse_actions,
        keyboard_actions,
        *,
        rngs,
        bidirectional,
        noise_type,  # standard | diffusion_forcing | teacher forcing
        mesh,
        left_action_padding,
    ):
        frames, cond_concat_BPFHWC, visual_context_BPFD = get_model_inputs(
            video, clip_model, vae_model
        )
        multiplayer_method = model.multiplayer_method

        # Apply multiplayer preprocessing in a single place. This reshapes video-like
        # tensors (frames, cond_concat_BPFHWC) for concat_* methods and, if provided,
        # aggregates visual context over players so that its player dimension matches
        # the concatenated representation expected by the causal multiplayer model.
        (
            cond_concat_BPFHWC,
            mouse_actions_BPFD,
            keyboard_actions_BPFD,
            visual_context_BPFD,
            frames,
        ) = handle_multiplayer_input(
            cond_concat_BPFHWC,
            mouse_actions,
            keyboard_actions,
            multiplayer_method,
            visual_context_BPFD,
            video_BPFHWC=frames,
        )
        actions_mouse_BPFD = mouse_actions_BPFD.astype(jnp.bfloat16)
        actions_keyboard_BPFD = keyboard_actions_BPFD.astype(jnp.bfloat16)

        actions_mouse_BPFD = left_repeat_padding(
            actions_mouse_BPFD, left_action_padding, axis=2
        )
        actions_keyboard_BPFD = left_repeat_padding(
            actions_keyboard_BPFD, left_action_padding, axis=2
        )

        rngs, rng_t = jax.random.split(rngs)

        b = frames.shape[0]
        p = frames.shape[1]
        timesteps = frames.shape[2]

        rngs, rng_noise = jax.random.split(rngs)

        if noise_type == "standard":
            t = repeat(
                jax.random.randint(rng_t, shape=(b,), minval=0, maxval=1001),
                "b -> b p t",
                p=p,
                t=timesteps,
            )
            t = t.astype(jnp.int32)
            noise = jax.random.normal(rng_noise, frames.shape)
            t_BPFHWC = t[:, :, :, None, None, None] / 1000
            noised_frame = (1 - t_BPFHWC) * frames + t_BPFHWC * noise
            x = noised_frame
        elif noise_type == "diffusion_forcing":
            t = jax.random.randint(
                rng_t, shape=(b, p, timesteps), minval=0, maxval=1001
            )
            t = t.astype(jnp.int32)

            t_BPFHWC = t[:, :, :, None, None, None] / 1000
            noise = jax.random.normal(rng_noise, frames.shape)
            noised_frame = (1 - t_BPFHWC) * frames + t_BPFHWC * noise
            x = noised_frame
        elif noise_type == "teacher_forcing":
            t = jax.random.randint(rng_t, shape=(b, timesteps), minval=0, maxval=1001)
            t = t.astype(jnp.int32)
            t_BPFHWC = t[:, :, None, None, None] / 1000
            noise = jax.random.normal(rng_noise, frames.shape)
            noised_frame = (1 - t_BPFHWC) * frames + t_BPFHWC * noise
            x = jnp.concatenate(
                [frames, noised_frame], axis=1
            )  # clean frames first, then noised frames.
            t = jnp.concatenate(
                [jnp.zeros((b, timesteps)).astype(jnp.float32), t], axis=1
            )  # clean frames first, then noised frames.
            cond_concat_BPFHWC = jnp.concatenate(
                [cond_concat_BPFHWC, cond_concat_BPFHWC], axis=1
            )
        else:
            raise ValueError(f"Invalid noise type: {noise_type}")

        is_teacher_forcing = noise_type == "teacher_forcing"

        res = model(
            x.astype(jnp.bfloat16),
            t,
            visual_context=visual_context_BPFD,
            cond_concat=cond_concat_BPFHWC,
            mouse_cond=actions_mouse_BPFD,
            keyboard_cond=actions_keyboard_BPFD,
            bidirectional=bidirectional,
            teacher_forcing=is_teacher_forcing,
            mesh=mesh,
        )

        flow_pred, _, _, _ = res

        if noise_type == "teacher_forcing":
            flow_pred = flow_pred[:, timesteps:, :, :, :]

        target_velocity = noise - frames
        mse_loss = (
            (flow_pred.astype(jnp.float32) - target_velocity.astype(jnp.float32)) ** 2
        ).mean()
        return mse_loss, rngs

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
        return self.evaluate_mp(
            bidirectional=self.bidirectional,
            model_state=model_state,
            model_graph=model_graph,
            vae_state=vae_state,
            vae_graph=vae_graph,
            clip_state=clip_state,
            clip_graph=clip_graph,
            video=video,
            mouse_actions=mouse_actions,
            keyboard_actions=keyboard_actions,
            real_lengths=real_lengths,
            eval_dir=eval_dir,
            mesh=mesh,
            left_action_padding=left_action_padding,
            num_denoising_steps=num_denoising_steps,
        )
