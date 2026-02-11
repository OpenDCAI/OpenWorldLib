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

from ...base_synthesis import BaseSynthesis
from .lingbot_va.modeling_lingbot_va_utils import (
    WanVAEStreamingWrapper,
    load_text_encoder,
    load_tokenizer,
    load_transformer,
    load_vae,
)
from .lingbot_va.scheduling_lingbot_va import FlowMatchScheduler
from .lingbot_va.data_utils_lingbot_va import data_seq_to_patch, get_mesh_id


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

    @torch.no_grad()
    def predict(
        self,
        input_dict: dict,
        action_mode: bool = False,
        update_cache: int = 0,
        cache_name: str = 'pos',
    ):
        """Single-step transformer forward pass."""
        return self.transformer(input_dict, update_cache=update_cache, cache_name=cache_name, action_mode=action_mode)

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
        height: int = 256,
        width: int = 320,
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

        if env_type == 'robotwin_tshape':
            video_latent = mu_norm
        else:
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
