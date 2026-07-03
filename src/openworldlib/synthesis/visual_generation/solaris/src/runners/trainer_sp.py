import jax
import jax.experimental
import jax.experimental.multihost_utils
import jax.numpy as jnp
import numpy as np
import torch
from einops import rearrange, repeat
from flax import nnx
from torchvision.io import write_video

import src.data.utils as data_utils
import src.utils.sharding as sharding_utils
from src.metrics.metrics_sp import psnr, ssim
from src.models.utils import jax_to_torch
from src.models.wan_vae import VAE_SCALE
from src.runners.base_ssl_trainer import BaseSSLTrainer
from src.utils.preprocessing_sp import wan_image_condition_preprocess
from src.utils.rollout import change_tensor_range, perform_bidirectional_rollout


class Trainer(BaseSSLTrainer):

    def save_model_state(self, save_model_state_to):
        optimizer = nnx.merge(self.optimizer_graph, self.optimizer_state)
        model_graph, model_state = nnx.split(optimizer.model)
        # When exporting a single-player checkpoint to be later loaded by the
        # new multiplayer model, we need to ensure that the saved state
        # already contains a `player_embed` parameter. Older single-player
        # models did not have this parameter, which causes a tree-structure
        # mismatch at restore time in the multiplayer runner.
        #
        # To fix this, if `player_embed` is missing we synthesize a dummy
        # embedding with random values. The exact values do not matter,
        # because the multiplayer model will treat this as a normal,
        # trainable parameter; we only need the tree structure to match.
        if "player_embed" not in model_state:
            # Match multiplayer model's config
            model_dim = int(self.network_config.params.dim)
            num_players = 2  # or whatever you actually use

            # 1) Make a dummy Embed to get the *exact* nnx state structure
            dummy_embed = nnx.Embed(
                num_embeddings=num_players,
                features=model_dim,
                rngs=nnx.Rngs(params=jax.random.PRNGKey(12)),
            )
            _, embed_state = nnx.split(dummy_embed)
            # embed_state is a State, typically something like:
            #   State({'embedding': Param(value=<jax.Array>)})

            # 2) Make its underlying arrays multi-host friendly
            def to_global(x):
                if isinstance(x, jax.Array):
                    host = np.asarray(jax.device_get(x), dtype=x.dtype)
                    # replicate across your mesh so Orbax sees a global array
                    return jax.device_put(host, self.repl_sharding)
                return x

            embed_state = jax.tree_util.tree_map(to_global, embed_state)

            # 3) Plug that state under the "player_embed" key
            model_state["player_embed"] = embed_state

        self.pretrained_checkpointer.save(save_model_state_to, model_state)
        self.pretrained_checkpointer.wait_until_finished()

    def _get_curr_batch(self, train_loader_iter):
        batch = self.robust_batch_sample(train_loader_iter)
        batch = batch.to_dict()
        batch = data_utils.torch_pytree_to_numpy(batch)
        batch = self.globalize_batch(batch)
        real_lengths = batch["real_lengths"]

        video_BFHWC = batch["obs"].astype(jnp.bfloat16)
        action_BFD = batch["act"].astype(jnp.bfloat16)

        actions_mouse_BFD = action_BFD[:, :, -2:]
        actions_keyboard_BFD = action_BFD[:, :, :-2]
        video_BFHWC_unprocessed = video_BFHWC
        video_BFHWC = wan_image_condition_preprocess(video_BFHWC, 352, 640).astype(
            jnp.bfloat16
        )
        return (
            video_BFHWC,
            video_BFHWC_unprocessed,
            actions_mouse_BFD,
            actions_keyboard_BFD,
            real_lengths,
        )

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
        noise_type,
        mesh,
        left_action_padding,
    ):
        frames, cond_concat_BFHWC, visual_context_BFD = get_model_inputs(
            video, clip_model, vae_model
        )
        actions_mouse_BFD = mouse_actions.astype(jnp.bfloat16)
        actions_keyboard_BFD = keyboard_actions.astype(jnp.bfloat16)

        def left_repeat_padding(x, pad):
            return jnp.concatenate([x[:, 0:1, :].repeat(pad, axis=1), x], axis=1)

        actions_mouse_BFD = left_repeat_padding(actions_mouse_BFD, left_action_padding)
        actions_keyboard_BFD = left_repeat_padding(
            actions_keyboard_BFD, left_action_padding
        )

        rngs, rng_t = jax.random.split(rngs)

        b = frames.shape[0]
        timesteps = frames.shape[1]

        rngs, rng_noise = jax.random.split(rngs)
        assert (
            noise_type == "standard"
        ), "Only standard noise type is supported for single-player training"

        t = repeat(
            jax.random.randint(rng_t, shape=(b,), minval=0, maxval=1001),
            "b -> b t",
            t=timesteps,
        )
        t = t.astype(jnp.int32)
        noise = jax.random.normal(rng_noise, frames.shape)
        t_BFHWC = t[:, :, None, None, None] / 1000
        noised_frame = (1 - t_BFHWC) * frames + t_BFHWC * noise
        x = noised_frame

        is_teacher_forcing = noise_type == "teacher_forcing"
        res = model(
            x.astype(jnp.bfloat16),
            t,
            visual_context=visual_context_BFD,
            cond_concat=cond_concat_BFHWC,
            mouse_cond=actions_mouse_BFD,
            keyboard_cond=actions_keyboard_BFD,
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
        num_eval_frames = video.shape[1]

        processed_video_BFHWC = wan_image_condition_preprocess(video, 352, 640)
        video = change_tensor_range(
            processed_video_BFHWC, [-1, 1], [0, 255], dtype=jnp.uint8
        )  # we
        first_frame_BHWC = processed_video_BFHWC[:, 0, :, :, :].astype(jnp.bfloat16)

        from functools import partial

        rollout_func = partial(
            perform_bidirectional_rollout, left_action_padding=left_action_padding
        )

        D = len(jax.devices())
        first_frame_DBHWC = rearrange(first_frame_BHWC, "(d b) h w c -> d b h w c", d=D)
        mouse_actions_DBFD = rearrange(
            mouse_actions, "(device b) f d -> device b f d", device=D
        )
        keyboard_actions_DBFD = rearrange(
            keyboard_actions, "(device b) f d -> device b f d", device=D
        )
        local_batch_size = first_frame_DBHWC.shape[1]
        rollouts = []
        for i in range(local_batch_size):
            rollout = rollout_func(
                model_graph,
                model_state,
                vae_graph,
                vae_state,
                clip_graph,
                clip_state,
                first_frame_DBHWC[:, i, :, :, :],
                mouse_actions_DBFD[:, i, :, :],
                keyboard_actions_DBFD[:, i, :, :],
                num_eval_frames,
                mesh=mesh,
                num_denoising_steps=num_denoising_steps,
            )
            rollouts.append(rollout)
        res_DBFHWC = jnp.stack(rollouts, axis=1)
        rollout_frames_BFHWC = rearrange(
            res_DBFHWC, "d b f h w c -> (d b) f h w c", d=D
        )

        ssim_T = ssim(rollout_frames_BFHWC, video)
        psnr_T = psnr(rollout_frames_BFHWC, video)

        if eval_dir is not None:
            rollout_local_slice = sharding_utils.get_local_slice_from_fsarray(
                rollout_frames_BFHWC
            )
            gt_local_slice = sharding_utils.get_local_slice_from_fsarray(video)

            torch_rollout_local_slice = jax_to_torch(rollout_local_slice)
            torch_gt_local_slice = jax_to_torch(gt_local_slice)

            torch_side_by_side_video_BFHWC = torch.concatenate(
                [torch_gt_local_slice, torch_rollout_local_slice], axis=3
            )

            num_processes = jax.process_count()
            process_index = jax.process_index()
            global_i = process_index
            for i in range(torch_side_by_side_video_BFHWC.shape[0]):
                video_int8 = torch_side_by_side_video_BFHWC[i]
                write_video(
                    f"{eval_dir}/video_{global_i}_side_by_side.mp4", video_int8, fps=20
                )
                global_i += num_processes

        # logging metrics.
        gathered_psnr_T = jax.experimental.multihost_utils.process_allgather(psnr_T)
        gathered_ssim_T = jax.experimental.multihost_utils.process_allgather(ssim_T)
        gathered_psnr_T = np.array(gathered_psnr_T)
        gathered_ssim_T = np.array(gathered_ssim_T)

        # gather the metrics to process index 0
        return {"psnr": gathered_psnr_T, "ssim": gathered_ssim_T}


def get_model_inputs(
    video_BFHWC,
    clip_model,
    vae_model,
):
    B, F, H, W, C = video_BFHWC.shape
    first_frame_BFHWC = video_BFHWC[:, 0:1, :, :, :]
    visual_context_BFD = jax.lax.stop_gradient(
        clip_model.encode_video(rearrange(first_frame_BFHWC, "b f h w c -> b c f h w"))
    ).astype(jnp.bfloat16)
    encoded_outputs = jax.lax.stop_gradient(
        vae_model.cacheless_encode(video_BFHWC, VAE_SCALE)
    )
    img_cond_BFHWC = jax.lax.stop_gradient(
        vae_model.cacheless_encode(
            jnp.concatenate(
                [video_BFHWC[:, :1, :, :, :], jnp.zeros((B, F - 1, H, W, C))], axis=1
            ).astype(jnp.bfloat16),
            VAE_SCALE,
        )
    )
    mask_cond_BFHWC = (
        jnp.ones(img_cond_BFHWC.shape, dtype=jnp.bfloat16).at[:, 1:].set(0)
    )
    cond_concat_BFHWC = jnp.concatenate(
        [mask_cond_BFHWC[:, :, :, :, :4], img_cond_BFHWC], axis=-1
    ).astype(jnp.bfloat16)
    return encoded_outputs, cond_concat_BFHWC, visual_context_BFD
