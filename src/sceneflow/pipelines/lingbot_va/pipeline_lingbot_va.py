# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
# Adapted from lingbot-va/wan_va/wan_va_server.py for SceneFlow integration.
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Generator, List

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from tqdm import tqdm

from ...operators.lingbot_va_operator import LingBotVAOperator
from ...synthesis.vla_generation.lingbot_va.lingbot_va_synthesis import LingBotVASynthesis
from ...synthesis.vla_generation.lingbot_va.lingbot_va.data_utils_lingbot_va import data_seq_to_patch


@dataclass
class LingBotVAOutput:
    """Output container for LingBot-VA pipeline."""
    actions: np.ndarray                          # predicted actions [used_channels, total_frames * action_per_frame]
    latents: torch.Tensor | None = None          # video latents [1, 48, F, H, W]
    video: np.ndarray | None = None              # decoded video frames (optional)


class LingBotVAPipeline:
    """Pipeline wrapper for LingBot-VA autoregressive video-action generation."""

    def __init__(
        self,
        synthesis: LingBotVASynthesis,
        operator: LingBotVAOperator,
        config: Any,
        device: str | torch.device | None = None,
    ):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.synthesis = synthesis
        self.operator = operator.to(self.device)
        self.config = config
        self.cache_name = 'pos'

        # State
        self.frame_st_id = 0
        self.init_latent: torch.Tensor | None = None
        self.prompt_embeds: torch.Tensor | None = None
        self.negative_prompt_embeds: torch.Tensor | None = None
        self.use_cfg = False

        # Derived dimensions
        self._compute_latent_dims()

    def _compute_latent_dims(self):
        cfg = self.config
        if cfg.env_type == 'robotwin_tshape':
            self.latent_height = ((cfg.height // 16) * 3) // 2
            self.latent_width = cfg.width // 16
        else:
            self.latent_height = cfg.height // 16
            self.latent_width = cfg.width // 16 * len(cfg.obs_cam_keys)

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        config: Any = None,
        device: str | torch.device | None = None,
        **kwargs: Any,
    ) -> 'LingBotVAPipeline':
        if config is None:
            raise ValueError("config must be provided.")
        device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        synthesis = LingBotVASynthesis.from_pretrained(model_path, config=config, device=device, **kwargs)
        operator = LingBotVAOperator(config=config)
        return cls(synthesis=synthesis, operator=operator, config=config, device=device)

    def to(self, device: str | torch.device):
        self.device = device
        self.synthesis.to(device)
        self.operator.to(device)
        return self

    # ── PipelineABC interface (all methods implemented) ───────────────────────

    def process(
        self,
        images: dict[str, np.ndarray] | list[dict[str, np.ndarray]],
        prompt: str,
    ) -> dict[str, Any]:
        """Preprocess inputs using the operator."""
        videos = self.operator.process_perception(images)
        cleaned_prompt = self.operator.process_interaction(prompt)
        return {
            'videos': videos,
            'prompt': cleaned_prompt,
        }

    @torch.no_grad()
    def __call__(
        self,
        images: dict[str, np.ndarray] | list[dict[str, np.ndarray]],
        prompt: str,
        num_chunks: int = 10,
        decode_video: bool = False,
    ) -> LingBotVAOutput:
        """Main inference entry: i2va mode — generate multiple chunks from initial images."""
        cfg = self.config
        dtype = cfg.param_dtype
        frame_chunk_size = cfg.frame_chunk_size

        self.reset(prompt)
        assert self.prompt_embeds is not None, "prompt_embeds must be set. Call reset(prompt) first."

        # Encode initial observation
        processed = self.process(images, prompt)
        videos = processed['videos']
        init_latent = self.synthesis.encode_images(
            videos, env_type=cfg.env_type,
            height=cfg.height, width=cfg.width,
        )
        self.init_latent = init_latent

        pred_latent_lst = []
        pred_action_lst = []

        for chunk_id in range(num_chunks):
            frame_st_id = chunk_id * frame_chunk_size

            # ── Random noise ──────────────────────────────────────────────
            latents = torch.randn(1, 48, frame_chunk_size, self.latent_height, self.latent_width,
                                   device=self.device, dtype=dtype)
            actions = torch.randn(1, cfg.action_dim, frame_chunk_size, cfg.action_per_frame, 1,
                                   device=self.device, dtype=dtype)

            # ── Schedulers ────────────────────────────────────────────────
            self.synthesis.scheduler.set_timesteps(cfg.num_inference_steps)
            self.synthesis.action_scheduler.set_timesteps(cfg.action_num_inference_steps)
            timesteps = F.pad(self.synthesis.scheduler.timesteps, (0, 1), mode='constant', value=0)
            action_timesteps = F.pad(self.synthesis.action_scheduler.timesteps, (0, 1), mode='constant', value=0)

            video_step = cfg.video_exec_step
            if video_step != -1:
                timesteps = timesteps[:video_step]

            with torch.amp.autocast('cuda', dtype=dtype):
                # ── Stage 1: Video denoising ──────────────────────────────
                for i, t in enumerate(tqdm(timesteps, desc=f'Chunk {chunk_id} video', leave=False)):
                    last_step = (i == len(timesteps) - 1)

                    latent_cond: torch.Tensor | None = None
                    if frame_st_id == 0 and init_latent is not None:
                        latent_cond = init_latent[:, :, 0:1].to(dtype)

                    raw_input = self.operator.prepare_model_input(
                        latents, None, latent_t=t, action_t=t,
                        latent_cond=latent_cond, action_cond=None,
                        frame_st_id=frame_st_id, patch_size=cfg.patch_size, device=self.device,
                    )
                    video_input = self.operator.repeat_input_for_cfg(
                        raw_input['latent_res_lst'], self.prompt_embeds, self.negative_prompt_embeds,
                        use_cfg=self.use_cfg, dtype=dtype,
                    )

                    video_noise_pred = self.synthesis.predict(
                        video_input, action_mode=False,
                        update_cache=1 if last_step else 0, cache_name=self.cache_name,
                    )

                    if not last_step or video_step != -1:
                        video_noise_pred = data_seq_to_patch(
                            cfg.patch_size, video_noise_pred, frame_chunk_size,
                            self.latent_height, self.latent_width,
                            batch_size=2 if self.use_cfg else 1,
                        )
                        if cfg.guidance_scale > 1:
                            video_noise_pred = video_noise_pred[1:] + cfg.guidance_scale * (video_noise_pred[:1] - video_noise_pred[1:])
                        else:
                            video_noise_pred = video_noise_pred[:1]
                        latents = self.synthesis.scheduler.step(video_noise_pred, t, latents, return_dict=False)

                    if latent_cond is not None:
                        latents[:, :, 0:1] = latent_cond
                    else:
                        latents[:, :, 0:1] = latents[:, :, 0:1]

                # ── Stage 2: Action denoising ─────────────────────────────
                for i, t in enumerate(tqdm(action_timesteps, desc=f'Chunk {chunk_id} action', leave=False)):
                    last_step = (i == len(action_timesteps) - 1)

                    action_cond: torch.Tensor | None = None
                    if frame_st_id == 0:
                        action_cond = torch.zeros(
                            [1, cfg.action_dim, 1, cfg.action_per_frame, 1],
                            device=self.device, dtype=dtype,
                        )

                    raw_input = self.operator.prepare_model_input(
                        None, actions, latent_t=t, action_t=t,
                        latent_cond=None, action_cond=action_cond,
                        frame_st_id=frame_st_id, patch_size=cfg.patch_size, device=self.device,
                    )
                    action_input = self.operator.repeat_input_for_cfg(
                        raw_input['action_res_lst'], self.prompt_embeds, self.negative_prompt_embeds,
                        use_cfg=self.use_cfg, dtype=dtype,
                    )

                    action_noise_pred = self.synthesis.predict(
                        action_input, action_mode=True,
                        update_cache=1 if last_step else 0, cache_name=self.cache_name,
                    )

                    if not last_step:
                        action_noise_pred = rearrange(action_noise_pred, 'b (f n) c -> b c f n 1', f=frame_chunk_size)
                        if cfg.action_guidance_scale > 1:
                            action_noise_pred = action_noise_pred[1:] + cfg.action_guidance_scale * (action_noise_pred[:1] - action_noise_pred[1:])
                        else:
                            action_noise_pred = action_noise_pred[:1]
                        actions = self.synthesis.action_scheduler.step(action_noise_pred, t, actions, return_dict=False)

                    if action_cond is not None:
                        actions[:, :, 0:1] = action_cond
                    else:
                        actions[:, :, 0:1] = actions[:, :, 0:1]

            # ── Post-process this chunk ───────────────────────────────────
            action_mask = self.operator.action_mask.to(actions.device)
            actions[:, ~action_mask] *= 0
            actions_np = self.operator.postprocess_action(actions)

            pred_latent_lst.append(latents)
            pred_action_lst.append(torch.from_numpy(actions_np))

        pred_latent = torch.cat(pred_latent_lst, dim=2)
        pred_action = torch.cat(pred_action_lst, dim=1).flatten(1).numpy()

        # Cleanup
        self.synthesis.clear_cache(self.cache_name)
        self.synthesis.clear_vae_cache()

        video_np = None
        if decode_video:
            video_np = self.synthesis.decode_latents(pred_latent)

        torch.cuda.empty_cache()
        return LingBotVAOutput(actions=pred_action, latents=pred_latent, video=video_np)

    def stream(self, *args, **kwds):
        """Not applicable for LingBot-VA pipeline."""
        pass

    def save_pretrained(self, save_directory: str):
        """Placeholder — not yet implemented for LingBot-VA."""
        pass

    # ── LingBot-VA specific methods ───────────────────────────────────────────

    def reset(self, prompt: str | None = None):
        """Reset all internal state: caches, frame counter, prompt encoding."""
        cfg = self.config
        self.use_cfg = (cfg.guidance_scale > 1) or (cfg.action_guidance_scale > 1)
        self.frame_st_id = 0
        self.init_latent = None

        self.synthesis.clear_cache(self.cache_name)
        self.synthesis.clear_vae_cache()

        patch_size = cfg.patch_size
        latent_token_per_chunk = (cfg.frame_chunk_size * self.latent_height * self.latent_width) // (
            patch_size[0] * patch_size[1] * patch_size[2])
        action_token_per_chunk = cfg.frame_chunk_size * cfg.action_per_frame
        self.synthesis.create_cache(
            self.cache_name, cfg.attn_window, latent_token_per_chunk,
            action_token_per_chunk, batch_size=2 if self.use_cfg else 1,
        )

        if prompt is not None:
            self.prompt_embeds, self.negative_prompt_embeds = self.synthesis.encode_prompt(
                prompt, negative_prompt=None,
                do_classifier_free_guidance=(cfg.guidance_scale > 1),
            )
        else:
            self.prompt_embeds = self.negative_prompt_embeds = None

        torch.cuda.empty_cache()

    def compute_kv_cache(self, obs: dict):
        """Encode historical observations and write into KV cache (for server-mode compatibility)."""
        cfg = self.config
        dtype = cfg.param_dtype

        assert self.prompt_embeds is not None, "prompt_embeds must be set before compute_kv_cache. Call reset(prompt) first."

        self.synthesis.clear_pred_cache(self.cache_name)

        images_list = obs['obs']
        if not isinstance(images_list, list):
            images_list = [images_list]
        videos = self.operator.process_perception(images_list)
        latent_model_input: torch.Tensor = self.synthesis.encode_images(
            videos, env_type=cfg.env_type, height=cfg.height, width=cfg.width,
        )

        if self.frame_st_id == 0 and self.init_latent is not None:
            latent_model_input = torch.cat([self.init_latent, latent_model_input], dim=2)
        elif self.frame_st_id == 0 and self.init_latent is None:
            pass  # use latent_model_input as-is

        action_model_input = self.operator.preprocess_action(obs['state']).to(latent_model_input)

        raw_input = self.operator.prepare_model_input(
            latent_model_input, action_model_input,
            frame_st_id=self.frame_st_id, patch_size=cfg.patch_size, device=self.device,
        )

        with torch.amp.autocast('cuda', dtype=dtype), torch.no_grad():
            latent_input = self.operator.repeat_input_for_cfg(
                raw_input['latent_res_lst'], self.prompt_embeds, self.negative_prompt_embeds,
                use_cfg=self.use_cfg, dtype=dtype,
            )
            self.synthesis.predict(latent_input, action_mode=False, update_cache=2, cache_name=self.cache_name)

            action_input = self.operator.repeat_input_for_cfg(
                raw_input['action_res_lst'], self.prompt_embeds, self.negative_prompt_embeds,
                use_cfg=self.use_cfg, dtype=dtype,
            )
            self.synthesis.predict(action_input, action_mode=True, update_cache=2, cache_name=self.cache_name)

        torch.cuda.empty_cache()
        self.frame_st_id += latent_model_input.shape[2]
