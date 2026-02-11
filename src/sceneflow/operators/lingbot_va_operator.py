# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
# Adapted from lingbot-va/wan_va/wan_va_server.py for SceneFlow integration.
from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .base_operator import BaseOperator
from ..synthesis.vla_generation.lingbot_va.lingbot_va.data_utils_lingbot_va import get_mesh_id


class LingBotVAOperator(BaseOperator):
    """Operator for LingBot-VA: handles all data pre/post-processing without any model components."""

    def __init__(
        self,
        config: Any,
        operation_types: list[str] | None = None,
    ) -> None:
        if operation_types is None:
            operation_types = ["image_processing", "action_processing", "prompt_processing"]
        super().__init__(operation_types=operation_types)
        self.config = config
        self.device = 'cpu'

        self.interaction_template = []
        self.interaction_template_init()

        # Action normalization tensors
        self.action_mask = torch.zeros([config.action_dim]).bool()
        self.action_mask[config.used_action_channel_ids] = True

        self.actions_q01 = torch.tensor(config.norm_stat['q01'], dtype=torch.float32).reshape(-1, 1, 1)
        self.actions_q99 = torch.tensor(config.norm_stat['q99'], dtype=torch.float32).reshape(-1, 1, 1)

    def to(self, device: str | torch.device):
        self.device = device
        self.actions_q01 = self.actions_q01.to(device)
        self.actions_q99 = self.actions_q99.to(device)
        self.action_mask = self.action_mask.to(device)
        return self

    # ── BaseOperator interface ────────────────────────────────────────────────

    def interaction_template_init(self):
        if not isinstance(self.interaction_template, list):
            raise ValueError("interaction_template should be a list")

    def get_interaction(self, interaction: str):
        if self.check_interaction(interaction):
            self.current_interaction.append(interaction)

    def check_interaction(self, interaction: str) -> bool:
        if not isinstance(interaction, str):
            raise TypeError(f"Interaction must be a string, got {type(interaction)}")
        return True

    def process_interaction(self, prompt: str) -> str:
        """Record interaction and return the cleaned prompt string."""
        self.get_interaction(prompt)
        self.interaction_history.append(prompt)
        return self.current_interaction[-1]

    def process_perception(
        self,
        images: dict[str, np.ndarray] | list[dict[str, np.ndarray]],
    ) -> list[torch.Tensor]:
        """Preprocess multi-view observation images: resize + normalize to [-1, 1].

        Args:
            images: dict mapping camera key -> numpy HWC uint8 image,
                    or a list of such dicts (for multiple timesteps).

        Returns:
            List of [1, C, F, H, W] tensors, one per camera view.
        """
        config = self.config
        if not isinstance(images, list):
            images = [images]

        videos = []
        for k_i, k in enumerate(config.obs_cam_keys):
            if config.env_type == 'robotwin_tshape':
                if k_i == 0:
                    height_i, width_i = config.height, config.width
                else:
                    height_i, width_i = config.height // 2, config.width // 2
            else:
                height_i, width_i = config.height, config.width

            frames = torch.from_numpy(
                np.stack([each[k] for each in images])
            ).float().permute(3, 0, 1, 2)  # C, F, H, W
            frames = F.interpolate(frames, size=(height_i, width_i), mode='bilinear', align_corners=False)
            frames = frames.unsqueeze(0)  # 1, C, F, H, W
            frames = frames / 255.0 * 2.0 - 1.0
            videos.append(frames)

        return videos

    def get_interaction_template(self):
        return self.interaction_template

    def get_interaction_history(self):
        return self.interaction_history

    def delete_last_interaction(self):
        self.current_interaction = self.current_interaction[:-1]

    # ── Action pre/post-processing ────────────────────────────────────────────

    def preprocess_action(self, action: np.ndarray) -> torch.Tensor:
        """Normalize raw action: channel remap + quantile normalization.

        Args:
            action: numpy array of shape [C, F, H].

        Returns:
            Tensor of shape [1, action_dim, F, H, 1].
        """
        config = self.config
        action_t = torch.from_numpy(action)
        action_t = F.pad(action_t, [0, 0, 0, 0, 0, 1], mode='constant', value=0)
        action_t = action_t[config.inverse_used_action_channel_ids]

        q01 = self.actions_q01.to(action_t.device)
        q99 = self.actions_q99.to(action_t.device)
        action_t = (action_t - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
        return action_t.unsqueeze(0).unsqueeze(-1)  # B, C, F, H, W

    def postprocess_action(self, action: torch.Tensor) -> np.ndarray:
        """Denormalize predicted action back to original space.

        Args:
            action: Tensor of shape [1, action_dim, F, H, 1].

        Returns:
            numpy array of shape [used_channels, F, H].
        """
        config = self.config
        action_cpu = action.cpu()
        action_cpu = action_cpu[0, ..., 0]  # C, F, H

        q01 = self.actions_q01.cpu()
        q99 = self.actions_q99.cpu()
        action_cpu = (action_cpu + 1) / 2 * (q99 - q01 + 1e-6) + q01
        action_np = action_cpu.squeeze(0).detach().numpy()
        return action_np[config.used_action_channel_ids]

    # ── Model input assembly ──────────────────────────────────────────────────

    def prepare_model_input(
        self,
        latent_model_input: torch.Tensor | None,
        action_model_input: torch.Tensor | None,
        latent_t: Any = 0,
        action_t: Any = 0,
        latent_cond: torch.Tensor | None = None,
        action_cond: torch.Tensor | None = None,
        frame_st_id: int = 0,
        patch_size: tuple[int, int, int] = (1, 2, 2),
        device: torch.device | str = 'cpu',
    ) -> dict:
        """Assemble input dicts for video and/or action transformer forward."""
        input_dict = {}

        if latent_model_input is not None:
            input_dict['latent_res_lst'] = {
                'noisy_latents': latent_model_input,
                'timesteps': torch.ones([latent_model_input.shape[2]], dtype=torch.float32, device=device) * latent_t,
                'grid_id': get_mesh_id(
                    latent_model_input.shape[-3] // patch_size[0],
                    latent_model_input.shape[-2] // patch_size[1],
                    latent_model_input.shape[-1] // patch_size[2],
                    0, 1, frame_st_id,
                ).to(device),
            }
            if latent_cond is not None:
                input_dict['latent_res_lst']['noisy_latents'][:, :, 0:1] = latent_cond[:, :, 0:1]
                input_dict['latent_res_lst']['timesteps'][0:1] *= 0

        if action_model_input is not None:
            input_dict['action_res_lst'] = {
                'noisy_latents': action_model_input,
                'timesteps': torch.ones([action_model_input.shape[2]], dtype=torch.float32, device=device) * action_t,
                'grid_id': get_mesh_id(
                    action_model_input.shape[-3],
                    action_model_input.shape[-2],
                    action_model_input.shape[-1],
                    1, 1, frame_st_id, action=True,
                ).to(device),
            }
            if action_cond is not None:
                input_dict['action_res_lst']['noisy_latents'][:, :, 0:1] = action_cond[:, :, 0:1]
                input_dict['action_res_lst']['timesteps'][0:1] *= 0
            action_mask = self.action_mask.to(action_model_input.device)
            input_dict['action_res_lst']['noisy_latents'][:, ~action_mask] *= 0

        return input_dict

    def repeat_input_for_cfg(
        self,
        input_dict: dict,
        prompt_embeds: torch.Tensor,
        negative_prompt_embeds: torch.Tensor | None,
        use_cfg: bool,
        dtype: torch.dtype,
    ) -> dict:
        """Duplicate inputs for classifier-free guidance."""
        if use_cfg and negative_prompt_embeds is not None:
            input_dict['noisy_latents'] = input_dict['noisy_latents'].repeat(2, 1, 1, 1, 1)
            input_dict['text_emb'] = torch.cat([
                prompt_embeds.to(dtype).clone(),
                negative_prompt_embeds.to(dtype).clone(),
            ], dim=0)
            input_dict['grid_id'] = input_dict['grid_id'][None].repeat(2, 1, 1)
            input_dict['timesteps'] = input_dict['timesteps'][None].repeat(2, 1)
        else:
            input_dict['text_emb'] = prompt_embeds.to(dtype).clone()
            input_dict['grid_id'] = input_dict['grid_id'][None]
            input_dict['timesteps'] = input_dict['timesteps'][None]
        return input_dict
