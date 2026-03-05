# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
# Adapted from lingbot-va/wan_va/wan_va_server.py for SceneFlow integration.
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from ...operators.lingbot_va_operator import LingBotVAOperator
from ...synthesis.vla_generation.lingbot_va.lingbot_va_synthesis import LingBotVASynthesis


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

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        device: str | torch.device | None = None,
        param_dtype: torch.dtype = torch.bfloat16,
        patch_size: tuple[int, int, int] = (1, 2, 2),
        env_type: str = 'robotwin_tshape',
        height: int = 256,
        width: int = 320,
        action_dim: int = 30,
        action_per_frame: int = 16,
        frame_chunk_size: int = 2,
        attn_window: int = 72,
        obs_cam_keys: list[str] | None = None,
        used_action_channel_ids: list[int] | None = None,
        action_norm_method: str = 'quantiles',
        norm_stat: dict | None = None,
        snr_shift: float = 5.0,
        action_snr_shift: float = 1.0,
        **kwargs: Any,
    ) -> 'LingBotVAPipeline':
        if obs_cam_keys is None:
            obs_cam_keys = [
                'observation_images_cam_high',
                'observation_images_cam_left_wrist',
                'observation_images_cam_right_wrist',
            ]
        if used_action_channel_ids is None:
            used_action_channel_ids = list(range(0, 7)) + list(range(28, 29)) + list(range(7, 14)) + list(range(29, 30))
        
        inverse_used_action_channel_ids = [len(used_action_channel_ids)] * action_dim
        for _i, _j in enumerate(used_action_channel_ids):
            inverse_used_action_channel_ids[_j] = _i

        from types import SimpleNamespace
        config = SimpleNamespace(
            param_dtype=param_dtype,
            patch_size=patch_size,
            env_type=env_type,
            height=height,
            width=width,
            action_dim=action_dim,
            action_per_frame=action_per_frame,
            frame_chunk_size=frame_chunk_size,
            attn_window=attn_window,
            obs_cam_keys=obs_cam_keys,
            used_action_channel_ids=used_action_channel_ids,
            inverse_used_action_channel_ids=inverse_used_action_channel_ids,
            action_norm_method=action_norm_method,
            norm_stat=norm_stat,
            snr_shift=snr_shift,
            action_snr_shift=action_snr_shift,
        )

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
        guidance_scale: float = 5.0,
        action_guidance_scale: float = 1.0,
        num_inference_steps: int = 25,
        action_num_inference_steps: int = 50,
        video_exec_step: int = -1,
    ) -> LingBotVAOutput:
        cfg = self.config

        # Override config with call arguments
        cfg.guidance_scale = guidance_scale
        cfg.action_guidance_scale = action_guidance_scale
        cfg.num_inference_steps = num_inference_steps
        cfg.action_num_inference_steps = action_num_inference_steps
        cfg.video_exec_step = video_exec_step

        self.reset(prompt)
        assert self.prompt_embeds is not None, "prompt_embeds must be set. Call reset(prompt) first."

        # Encode initial observation
        processed = self.process(images, prompt)
        videos = processed['videos']
        init_latent = self.synthesis.encode_images(
            videos, env_type=cfg.env_type,
        )

        # Delegate full denoising to synthesis
        result = self.synthesis.predict(
            operator=self.operator,
            init_latent=init_latent,
            prompt_embeds=self.prompt_embeds,
            negative_prompt_embeds=self.negative_prompt_embeds,
            num_chunks=num_chunks,
            decode_video=decode_video,
            cache_name=self.cache_name,
        )

        # Cleanup
        self.synthesis.clear_cache(self.cache_name)
        self.synthesis.clear_vae_cache()
        torch.cuda.empty_cache()

        return LingBotVAOutput(
            actions=result['actions'],
            latents=result['latents'],
            video=result['video'],
        )

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
        self.synthesis.setup_cache(self.cache_name, self.use_cfg)

        if prompt is not None:
            self.prompt_embeds, self.negative_prompt_embeds = self.synthesis.encode_prompt(
                prompt, negative_prompt=None,
                do_classifier_free_guidance=(cfg.guidance_scale > 1),
            )
        else:
            self.prompt_embeds = self.negative_prompt_embeds = None

        torch.cuda.empty_cache()