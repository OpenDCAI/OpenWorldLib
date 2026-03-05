# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
# Adapted from lingbot-va/wan_va/wan_va_server.py for SceneFlow integration.
from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from diffusers.pipelines.wan.pipeline_wan import prompt_clean
from diffusers.video_processor import VideoProcessor
from tqdm import tqdm

from ...base_synthesis import BaseSynthesis
from .lingbot_va.modeling_lingbot_va_utils import (
    WanVAEStreamingWrapper,
    load_text_encoder,
    load_tokenizer,
    load_transformer,
    load_vae,
)
from .lingbot_va.scheduling_lingbot_va import FlowMatchScheduler

from einops import rearrange


class LingBotVASynthesis(BaseSynthesis):
    """Synthesis wrapper for LingBot-VA: loads all model components and provides inference primitives."""

    def __init__(
        self,
        transformer,
        vae,
        streaming_vae,
        streaming_vae_half,
        text_encoder,
        tokenizer,
        scheduler: FlowMatchScheduler,
        action_scheduler: FlowMatchScheduler,
        config,
        device: str | torch.device = 'cpu',
    ):
        super().__init__()
        self.transformer = transformer
        self.vae = vae
        self.streaming_vae = streaming_vae
        self.streaming_vae_half = streaming_vae_half
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.scheduler = scheduler
        self.action_scheduler = action_scheduler
        self.config = config
        self.device = device
        self.dtype = config.param_dtype
        self.video_processor = VideoProcessor(vae_scale_factor=1)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: str,
        config: Any = None,
        device: str | torch.device | None = None,
        **kwargs: Any,
    ) -> 'LingBotVASynthesis':
        if config is None:
            raise ValueError("config must be provided.")
        device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        config.wan22_pretrained_model_name_or_path = pretrained_model_path
        dtype = config.param_dtype

        vae = load_vae(os.path.join(pretrained_model_path, 'vae'), torch_dtype=dtype, torch_device=device)
        streaming_vae = WanVAEStreamingWrapper(vae)

        tokenizer = load_tokenizer(os.path.join(pretrained_model_path, 'tokenizer'))
        text_encoder = load_text_encoder(os.path.join(pretrained_model_path, 'text_encoder'), torch_dtype=dtype, torch_device=device)
        transformer = load_transformer(os.path.join(pretrained_model_path, 'transformer'), torch_dtype=dtype, torch_device=device)

        streaming_vae_half = None
        if config.env_type == 'robotwin_tshape':
            vae_half = load_vae(os.path.join(pretrained_model_path, 'vae'), torch_dtype=dtype, torch_device=device)
            streaming_vae_half = WanVAEStreamingWrapper(vae_half)

        scheduler = FlowMatchScheduler(shift=config.snr_shift, sigma_min=0.0, extra_one_step=True)
        action_scheduler = FlowMatchScheduler(shift=config.action_snr_shift, sigma_min=0.0, extra_one_step=True)

        return cls(
            transformer=transformer,
            vae=vae,
            streaming_vae=streaming_vae,
            streaming_vae_half=streaming_vae_half,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=scheduler,
            action_scheduler=action_scheduler,
            config=config,
            device=device,
        )

    def api_init(self, api_key: str, endpoint: str):
        """Not applicable for local LingBot-VA model."""
        pass

    # ── Main generation entry ─────────────────────────────────────────────────

    @torch.no_grad()
    def predict(
        self,
        operator,
        init_latent: torch.Tensor,
        prompt_embeds: torch.Tensor,
        negative_prompt_embeds: torch.Tensor | None,
        num_chunks: int = 10,
        decode_video: bool = False,
        cache_name: str = 'pos',
    ) -> dict:
        """Run the full autoregressive video-action denoising loop.

        Args:
            operator: LingBotVAOperator instance for input assembly & postprocessing.
            init_latent: Encoded initial observation latent.
            prompt_embeds: Text prompt embeddings.
            negative_prompt_embeds: Negative prompt embeddings (or None).
            num_chunks: Number of autoregressive chunks to generate.
            decode_video: Whether to decode latents into pixel-space video.
            cache_name: KV-cache identifier.

        Returns:
            dict with keys 'actions', 'latents', and optionally 'video'.
        """
        cfg = self.config
        dtype = cfg.param_dtype
        frame_chunk_size = cfg.frame_chunk_size
        device = self.device

        use_cfg = (cfg.guidance_scale > 1) or (cfg.action_guidance_scale > 1)

        latent_height, latent_width = self._get_latent_dims()

        pred_latent_lst = []
        pred_action_lst = []

        for chunk_id in range(num_chunks):
            frame_st_id = chunk_id * frame_chunk_size

            # ── Random noise ──────────────────────────────────────────────
            latents = torch.randn(
                1, 48, frame_chunk_size, latent_height, latent_width,
                device=device, dtype=dtype,
            )
            actions = torch.randn(
                1, cfg.action_dim, frame_chunk_size, cfg.action_per_frame, 1,
                device=device, dtype=dtype,
            )

            # ── Schedulers ────────────────────────────────────────────────
            self.scheduler.set_timesteps(cfg.num_inference_steps)
            self.action_scheduler.set_timesteps(cfg.action_num_inference_steps)
            timesteps = F.pad(self.scheduler.timesteps, (0, 1), mode='constant', value=0)
            action_timesteps = F.pad(self.action_scheduler.timesteps, (0, 1), mode='constant', value=0)

            video_step = cfg.video_exec_step
            if video_step != -1:
                timesteps = timesteps[:video_step]

            # ── Stage 1: Video denoising ──────────────────────────────
            latents = self._denoise_video(
                operator, latents, timesteps, video_step,
                init_latent, frame_st_id, frame_chunk_size,
                latent_height, latent_width,
                prompt_embeds, negative_prompt_embeds,
                use_cfg, cache_name, dtype, device,
            )

            # ── Stage 2: Action denoising ─────────────────────────────
            actions = self._denoise_action(
                operator, actions, action_timesteps,
                frame_st_id, frame_chunk_size,
                prompt_embeds, negative_prompt_embeds,
                use_cfg, cache_name, dtype, device,
            )

            # ── Post-process this chunk ───────────────────────────────
            action_mask = operator.action_mask.to(actions.device)
            actions[:, ~action_mask] *= 0
            actions_np = operator.postprocess_action(actions)

            pred_latent_lst.append(latents)
            pred_action_lst.append(torch.from_numpy(actions_np))

        pred_latent = torch.cat(pred_latent_lst, dim=2)
        pred_action = torch.cat(pred_action_lst, dim=1).flatten(1).numpy()

        video_np = None
        if decode_video:
            video_np = self.decode_latents(pred_latent)

        return {
            'actions': pred_action,
            'latents': pred_latent,
            'video': video_np,
        }

    # ── Internal: transformer forward helpers ─────────────────────────────────

    @staticmethod
    def _data_seq_to_patch(patch_size, data_seq, latent_num_frames, latent_height, latent_width, batch_size=1):
        """Reshape transformer sequence output back to spatial patch layout."""
        p_t, p_h, p_w = patch_size
        post_patch_num_frames = latent_num_frames // p_t
        post_patch_height = latent_height // p_h
        post_patch_width = latent_width // p_w

        data_patch = data_seq.reshape(batch_size, post_patch_num_frames,
                                      post_patch_height, post_patch_width, p_t,
                                      p_h, p_w, -1)
        data_patch = data_patch.permute(0, 7, 1, 4, 2, 5, 3, 6)
        data_patch = data_patch.flatten(6, 7).flatten(4, 5).flatten(2, 3)
        return data_patch

    @torch.no_grad()
    def _forward_video_noise(
        self,
        input_dict: dict,
        frame_chunk_size: int,
        latent_height: int,
        latent_width: int,
        update_cache: int = 0,
        cache_name: str = 'pos',
        batch_size: int = 1,
    ) -> torch.Tensor:
        """Transformer forward for video + reshape output back to spatial layout."""
        raw_output = self.transformer(input_dict, update_cache=update_cache, cache_name=cache_name, action_mode=False)
        return self._data_seq_to_patch(
            self.config.patch_size, raw_output, frame_chunk_size,
            latent_height, latent_width, batch_size=batch_size,
        )

    @torch.no_grad()
    def _forward_action_noise(
        self,
        input_dict: dict,
        frame_chunk_size: int,
        update_cache: int = 0,
        cache_name: str = 'pos',
    ) -> torch.Tensor:
        """Transformer forward for action + reshape output back to [B, C, F, N, 1]."""
        raw_output = self.transformer(input_dict, update_cache=update_cache, cache_name=cache_name, action_mode=True)
        return rearrange(raw_output, 'b (f n) c -> b c f n 1', f=frame_chunk_size)

    def to(self, device: str | torch.device):
        self.device = device
        self.transformer = self.transformer.to(device)
        self.vae = self.vae.to(device)
        self.text_encoder = self.text_encoder.to(device)
        return self

    # ── Text encoding ─────────────────────────────────────────────────────────

    def encode_prompt(
        self,
        prompt: str,
        negative_prompt: str | None = None,
        do_classifier_free_guidance: bool = True,
        max_sequence_length: int = 512,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt = [prompt_clean(u) for u in prompt]
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt, padding="max_length", max_length=max_sequence_length,
            truncation=True, add_special_tokens=True, return_attention_mask=True, return_tensors="pt",
        )
        text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()

        prompt_embeds = self.text_encoder(text_input_ids.to(self.device), mask.to(self.device)).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=self.dtype, device=self.device)
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
        prompt_embeds = torch.stack([
            torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in prompt_embeds
        ], dim=0)

        negative_prompt_embeds = None
        if do_classifier_free_guidance:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
            neg_inputs = self.tokenizer(
                [prompt_clean(u) for u in negative_prompt],
                padding="max_length", max_length=max_sequence_length,
                truncation=True, add_special_tokens=True, return_attention_mask=True, return_tensors="pt",
            )
            neg_ids, neg_mask = neg_inputs.input_ids, neg_inputs.attention_mask
            neg_lens = neg_mask.gt(0).sum(dim=1).long()
            negative_prompt_embeds = self.text_encoder(neg_ids.to(self.device), neg_mask.to(self.device)).last_hidden_state
            negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.dtype, device=self.device)
            negative_prompt_embeds = [u[:v] for u, v in zip(negative_prompt_embeds, neg_lens)]
            negative_prompt_embeds = torch.stack([
                torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in negative_prompt_embeds
            ], dim=0)

        return prompt_embeds, negative_prompt_embeds

    # ── VAE encode / decode ───────────────────────────────────────────────────

    def encode_images(
        self,
        videos: list[torch.Tensor],
        env_type: str = 'none',
    ) -> torch.Tensor:
        """Encode multi-view image tensors (already preprocessed by operator) into latents."""
        if env_type == 'robotwin_tshape':
            assert self.streaming_vae_half is not None
            videos_high = videos[0]
            videos_left_and_right = torch.cat(videos[1:], dim=0)
            enc_out_high = self.streaming_vae.encode_chunk(videos_high.to(self.device).to(self.dtype))
            enc_out_lr = self.streaming_vae_half.encode_chunk(videos_left_and_right.to(self.device).to(self.dtype))
            enc_out = torch.cat([
                torch.cat(enc_out_lr.split(1, dim=0), dim=-1),
                enc_out_high,
            ], dim=-2)
        else:
            videos_cat = torch.cat(videos, dim=0)
            enc_out = self.streaming_vae.encode_chunk(videos_cat.to(self.device).to(self.dtype))

        mu, logvar = torch.chunk(enc_out, 2, dim=1)
        latents_mean = torch.tensor(self.vae.config.latents_mean).to(mu.device)
        latents_std = torch.tensor(self.vae.config.latents_std).to(mu.device)
        mu_norm = self._normalize_latents(mu, latents_mean, 1.0 / latents_std)
        # Align with original: always concat multi-view latents along width
        video_latent = torch.cat(mu_norm.split(1, dim=0), dim=-1)
        return video_latent

    def _normalize_latents(self, latents: torch.Tensor, latents_mean: torch.Tensor, latents_std: torch.Tensor) -> torch.Tensor:
        latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(device=latents.device)
        latents_std = latents_std.view(1, -1, 1, 1, 1).to(device=latents.device)
        return ((latents.float() - latents_mean) * latents_std).to(latents)

    def decode_latents(self, latents: torch.Tensor) -> np.ndarray:
        latents = latents.to(self.vae.dtype)
        latents_mean = torch.tensor(self.vae.config.latents_mean).view(1, self.vae.config.z_dim, 1, 1, 1).to(latents.device, latents.dtype)
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(latents.device, latents.dtype)
        latents = latents / latents_std + latents_mean
        video = self.vae.decode(latents, return_dict=False)[0]
        video = self.video_processor.postprocess_video(video, output_type='np')
        return video[0]

    # ── KV cache management ───────────────────────────────────────────────────

    def create_cache(self, cache_name: str, attn_window: int, latent_token_per_chunk: int,
                     action_token_per_chunk: int, batch_size: int):
        self.transformer.create_empty_cache(
            cache_name, attn_window, latent_token_per_chunk, action_token_per_chunk,
            device=self.device, dtype=self.dtype, batch_size=batch_size,
        )

    def clear_cache(self, cache_name: str):
        self.transformer.clear_cache(cache_name)

    def clear_pred_cache(self, cache_name: str):
        self.transformer.clear_pred_cache(cache_name)

    def clear_vae_cache(self):
        self.streaming_vae.clear_cache()
        if self.streaming_vae_half is not None:
            self.streaming_vae_half.clear_cache()

    # ── Internal: latent dims & denoising loops ───────────────────────────────

    def _get_latent_dims(self) -> tuple[int, int]:
        """Compute latent spatial dimensions from config."""
        cfg = self.config
        if cfg.env_type == 'robotwin_tshape':
            latent_height = ((cfg.height // 16) * 3) // 2
            latent_width = cfg.width // 16
        else:
            latent_height = cfg.height // 16
            latent_width = cfg.width // 16 * len(cfg.obs_cam_keys)
        return latent_height, latent_width

    def _denoise_video(
        self,
        operator,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        video_step: int,
        init_latent: torch.Tensor | None,
        frame_st_id: int,
        frame_chunk_size: int,
        latent_height: int,
        latent_width: int,
        prompt_embeds: torch.Tensor,
        negative_prompt_embeds: torch.Tensor | None,
        use_cfg: bool,
        cache_name: str,
        dtype: torch.dtype,
        device: str | torch.device,
    ) -> torch.Tensor:
        """Run the video denoising loop for one chunk."""
        cfg = self.config

        for i, t in enumerate(tqdm(timesteps, desc='Video denoising', leave=False)):
            last_step = (i == len(timesteps) - 1)

            latent_cond: torch.Tensor | None = None
            if frame_st_id == 0 and init_latent is not None:
                latent_cond = init_latent[:, :, 0:1].to(dtype)

            raw_input = operator.prepare_model_input(
                latents, None, latent_t=t, action_t=t,
                latent_cond=latent_cond, action_cond=None,
                frame_st_id=frame_st_id, patch_size=cfg.patch_size, device=device,
            )
            video_input = operator.repeat_input_for_cfg(
                raw_input['latent_res_lst'], prompt_embeds, negative_prompt_embeds,
                use_cfg=use_cfg, dtype=dtype,
            )

            # Transformer forward
            video_noise_pred = self.transformer(
                video_input, update_cache=1 if last_step else 0,
                cache_name=cache_name, action_mode=False,
            )

            if not last_step or video_step != -1:
                video_noise_pred = self._data_seq_to_patch(
                    cfg.patch_size, video_noise_pred,
                    frame_chunk_size, latent_height, latent_width,
                    batch_size=2 if use_cfg else 1,
                )
                if cfg.guidance_scale > 1:
                    video_noise_pred = video_noise_pred[1:] + cfg.guidance_scale * (video_noise_pred[:1] - video_noise_pred[1:])
                else:
                    video_noise_pred = video_noise_pred[:1]
                latents = self.scheduler.step(video_noise_pred, t, latents, return_dict=False)

            latents[:, :, 0:1] = latent_cond if latent_cond is not None else latents[:, :, 0:1]

        return latents

    def _denoise_action(
        self,
        operator,
        actions: torch.Tensor,
        action_timesteps: torch.Tensor,
        frame_st_id: int,
        frame_chunk_size: int,
        prompt_embeds: torch.Tensor,
        negative_prompt_embeds: torch.Tensor | None,
        use_cfg: bool,
        cache_name: str,
        dtype: torch.dtype,
        device: str | torch.device,
    ) -> torch.Tensor:
        """Run the action denoising loop for one chunk."""
        cfg = self.config

        for i, t in enumerate(tqdm(action_timesteps, desc='Action denoising', leave=False)):
            last_step = (i == len(action_timesteps) - 1)

            action_cond: torch.Tensor | None = None
            if frame_st_id == 0:
                action_cond = torch.zeros(
                    [1, cfg.action_dim, 1, cfg.action_per_frame, 1],
                    device=device, dtype=dtype,
                )

            raw_input = operator.prepare_model_input(
                None, actions, latent_t=t, action_t=t,
                latent_cond=None, action_cond=action_cond,
                frame_st_id=frame_st_id, patch_size=cfg.patch_size, device=device,
            )
            action_input = operator.repeat_input_for_cfg(
                raw_input['action_res_lst'], prompt_embeds, negative_prompt_embeds,
                use_cfg=use_cfg, dtype=dtype,
            )

            # Transformer forward
            action_noise_pred = self.transformer(
                action_input, update_cache=1 if last_step else 0,
                cache_name=cache_name, action_mode=True,
            )

            if not last_step:
                action_noise_pred = rearrange(action_noise_pred, 'b (f n) c -> b c f n 1', f=frame_chunk_size)
                if cfg.action_guidance_scale > 1:
                    action_noise_pred = action_noise_pred[1:] + cfg.action_guidance_scale * (action_noise_pred[:1] - action_noise_pred[1:])
                else:
                    action_noise_pred = action_noise_pred[:1]
                actions = self.action_scheduler.step(action_noise_pred, t, actions, return_dict=False)

            actions[:, :, 0:1] = action_cond if action_cond is not None else actions[:, :, 0:1]

        return actions

    # ── Cache setup for generation ────────────────────────────────────────────

    def setup_cache(self, cache_name: str, use_cfg: bool):
        """Create KV cache for autoregressive generation."""
        cfg = self.config
        latent_height, latent_width = self._get_latent_dims()
        patch_size = cfg.patch_size
        latent_token_per_chunk = (cfg.frame_chunk_size * latent_height * latent_width) // (
            patch_size[0] * patch_size[1] * patch_size[2])
        action_token_per_chunk = cfg.frame_chunk_size * cfg.action_per_frame
        self.create_cache(
            cache_name, cfg.attn_window, latent_token_per_chunk,
            action_token_per_chunk, batch_size=2 if use_cfg else 1,
        )
