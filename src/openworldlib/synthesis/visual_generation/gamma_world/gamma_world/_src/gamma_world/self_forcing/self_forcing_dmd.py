# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
from typing import Dict, List, Literal, Optional, Tuple

import attrs
import torch
import torch.distributed as dist
import torch.nn.functional as F
from einops import rearrange
from megatron.core import parallel_state
from torch.nn.attention.flex_attention import BlockMask

from gamma_world._src.imaginaire.utils import misc
from gamma_world._src.imaginaire.utils.context_parallel import (
    broadcast,
    broadcast_split_tensor,
    cat_outputs_cp_with_grad,
)
from gamma_world._src.imaginaire.utils.parallel_state_helper import is_tp_cp_pp_rank0
from gamma_world._src.predict2.conditioner import DataType
from gamma_world._src.gamma_world.self_forcing.dmd import ImaginaireDMDBaseModel, ImaginaireDMDBaseModelConfig
from gamma_world._src.gamma_world.third_party.self_forcing.loss import FlowPredLoss
from gamma_world._src.gamma_world.third_party.self_forcing.pipeline import SelfForcingTrainingPipeline
from gamma_world._src.gamma_world.third_party.self_forcing.scheduler import FlowMatchScheduler


def print_rank0(*msgs):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(*msgs)
    else:
        print(*msgs)


@attrs.define(slots=False)
class BaseModelConfig(ImaginaireDMDBaseModelConfig):
    denoising_step_list: List[int] = [1000, 750, 500, 250]
    warp_denoising_step: bool = True
    num_train_timestep: int = 1000


class BaseModel(ImaginaireDMDBaseModel):
    config: BaseModelConfig

    def __init__(self, config: BaseModelConfig):
        super().__init__(config)
        self.scheduler = FlowMatchScheduler(shift=5.0, sigma_min=0.0, extra_one_step=True)
        self.scheduler.set_timesteps(config.num_train_timestep, training=True)
        self.scheduler.timesteps = self.scheduler.timesteps.to(self.device)

        self.denoising_step_list = torch.LongTensor(config.denoising_step_list)

        if self.config.warp_denoising_step:


            timesteps = torch.cat(
                (
                    self.scheduler.timesteps.cpu(),
                    torch.tensor([0], dtype=torch.float32),
                )
            )
            self.denoising_step_list = timesteps[self.config.num_train_timestep - self.denoising_step_list]

    def _get_timestep(
        self,
        min_timestep: int,
        max_timestep: int,
        batch_size: int,
        num_frame: int,
        num_frame_per_block: int,
        uniform_timestep: bool = False,
    ) -> torch.Tensor:

        if uniform_timestep:
            timestep = torch.randint(
                min_timestep,
                max_timestep,
                [batch_size, 1],
                device=self.device,
                dtype=torch.long,
            ).repeat(1, num_frame)
            return timestep
        else:
            timestep = torch.randint(
                min_timestep,
                max_timestep,
                [batch_size, num_frame],
                device=self.device,
                dtype=torch.long,
            )

            timestep = timestep.reshape(timestep.shape[0], -1, num_frame_per_block)
            timestep[:, :, 1:] = timestep[:, :, 0:1]
            timestep = timestep.reshape(timestep.shape[0], -1)
            return timestep


@attrs.define(slots=False)
class SelfForcingModelConfig(BaseModelConfig):
    num_training_frames: int = 21
    num_gradient_enabled_frames: int = 21
    num_frame_per_block: int = 3

    same_step_across_blocks: bool = True
    last_step_only: bool = False

    model_type: str = "t2v"
    i2v_zero_latent_condition: bool = False


    context_noise: float = 0.0
    add_context_noise_in_training: bool = False

    independent_denoising_step_list: bool = False
    shrink_list: list[int] = [3]


    min_num_conditional_frames: int = 0
    max_num_conditional_frames: int = 1
    conditional_frame_timestep: float = (
        -1.0
    )
    conditioning_strategy: str = "frame_replace"
    denoise_replace_gt_frames: bool = False
    conditional_frames_probs: Optional[Dict[int, float]] = None


class SelfForcingModel(BaseModel):
    config: SelfForcingModelConfig

    def __init__(self, config: SelfForcingModelConfig):
        super().__init__(config)
        self.denoising_loss_func = FlowPredLoss()
        if hasattr(self.net, "num_frame_per_block"):
            self.net.num_frame_per_block = config.num_frame_per_block
        if hasattr(self.net_real_score, "num_frame_per_block"):
            self.net_real_score.num_frame_per_block = config.net_real_score.state_t
        if hasattr(self.net_fake_score, "num_frame_per_block"):
            self.net_fake_score.num_frame_per_block = config.net_fake_score.state_t

        if self.config.add_context_noise_in_training:
            self.denoising_step_list = torch.cat(
                [self.denoising_step_list, torch.tensor([self.config.context_noise]).to(self.denoising_step_list)]
            )

        if self.config.independent_denoising_step_list:
            self.denoising_step_list = [self.denoising_step_list] * (
                self.config.num_training_frames // config.num_frame_per_block
            )

            for shrink_list in config.shrink_list:
                for i in range(shrink_list, len(self.denoising_step_list)):
                    if self.denoising_step_list[i].shape[0] > 2:
                        self.denoising_step_list[i] = torch.cat(
                            [self.denoising_step_list[i][:-2], self.denoising_step_list[i][-1:]]
                        )
                    else:
                        self.denoising_step_list[i] = self.denoising_step_list[i][:-1]
            print_rank0("self.denoising_step_list: ", self.denoising_step_list)

    def _run_generator(
        self,
        image_or_video_shape,
        conditional_dict: dict,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[int], Optional[int]]:


        noise_shape = image_or_video_shape.copy()





        min_num_frames = noise_shape[1]
        max_num_frames = noise_shape[1]

        assert max_num_frames % self.config.num_frame_per_block == 0
        assert min_num_frames % self.config.num_frame_per_block == 0
        max_num_blocks = max_num_frames // self.config.num_frame_per_block
        min_num_blocks = min_num_frames // self.config.num_frame_per_block
        num_generated_blocks = torch.randint(min_num_blocks, max_num_blocks + 1, (1,), device=self.device)
        if dist.is_initialized():
            dist.broadcast(num_generated_blocks, src=0)
        num_generated_blocks = num_generated_blocks.item()
        num_generated_frames = num_generated_blocks * self.config.num_frame_per_block


        noise_shape[1] = num_generated_frames

        pred_image_or_video, denoised_timestep_from, denoised_timestep_to = self.generate_samples_from_batch(
            is_training=True,
            noise_B_T_C_H_W=torch.randn(
                noise_shape, device=self.tensor_kwargs["device"], dtype=self.tensor_kwargs["dtype"]
            ),
            image_or_video_shape=noise_shape,
            conditional_dict=conditional_dict,
        )





        pred_image_or_video_last_21 = pred_image_or_video

        if num_generated_frames != min_num_frames:

            gradient_mask = torch.ones_like(pred_image_or_video_last_21, dtype=torch.bool)
            gradient_mask[:, : self.config.num_frame_per_block] = False
        else:
            gradient_mask = None






        pred_image_or_video_last_21 = pred_image_or_video_last_21.to(self.dtype)
        return (
            pred_image_or_video_last_21,
            gradient_mask,
            denoised_timestep_from,
            denoised_timestep_to,
        )

    def _consistency_backward_simulation(self, noise: torch.Tensor, **conditional_dict: dict) -> torch.Tensor:

        if self.inference_pipeline is None:
            self._initialize_inference_pipeline()

        return self.inference_pipeline.inference_with_trajectory(noise=noise, **conditional_dict)

    def generator(self, *args, **kwargs):
        return self.denoise(net_choice="generator", scheduler=self.scheduler, *args, **kwargs)

    def real_score(self, *args, **kwargs):
        return self.denoise(net_choice="real_score", scheduler=self.scheduler, *args, **kwargs)

    def fake_score(self, *args, **kwargs):
        return self.denoise(net_choice="fake_score", scheduler=self.scheduler, *args, **kwargs)


@attrs.define(slots=False)
class DMDSelfForcingModelConfig(SelfForcingModelConfig):
    real_guidance_scale: float = 3.0
    fake_guidance_scale: float = 0.0
    timestep_shift: float = 5.0
    ts_schedule: bool = False
    ts_schedule_max: bool = False
    min_score_timestep: int = 0
    image_or_video_shape: list[int] = [16, 21, 60, 104]
    state_ch: int = 16
    state_t: int = 24



    noise_scheme: str = "diffusion_forcing"


    use_vidprom_dataset: bool = False


NUM_CONDITIONAL_FRAMES_KEY: str = "num_conditional_frames"


class DMDSelfForcingModel(SelfForcingModel):
    config: DMDSelfForcingModelConfig

    def __init__(self, config: DMDSelfForcingModelConfig):

        super().__init__(config)


        self.inference_pipeline: Optional[SelfForcingTrainingPipeline] = None


        self.min_step = int(0.02 * self.config.num_train_timestep)
        self.max_step = int(0.98 * self.config.num_train_timestep)

        self.frame_seq_length = None

    def _extract_extra_conditional_kwargs(self, data_batch: dict[str, torch.Tensor]) -> dict:

        return {}

    def denoise(
        self,
        scheduler,
        net_choice: Literal["generator", "real_score", "fake_score"],
        noisy_image_or_video: torch.Tensor,
        conditional_dict: dict,
        timestep: torch.Tensor,
        kv_cache: Optional[List[dict]] = None,
        crossattn_cache: Optional[List[dict]] = None,
        current_start: Optional[int] = None,
        current_end: Optional[int] = None,
        start_frame_for_rope: Optional[int] = None,
        block_mask: Optional[BlockMask] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if net_choice == "generator":
            model = self.net
            uniform_timestep = False
        elif net_choice == "real_score":
            model = self.net_real_score
            uniform_timestep = True
        elif net_choice == "fake_score":
            model = self.net_fake_score
            uniform_timestep = True
        else:
            raise ValueError(f"Invalid net choice: {net_choice}")

        n_views = noisy_image_or_video.shape[1] // self.config.state_t

        if uniform_timestep:
            if n_views == 1:


                input_timestep = timestep[:, :1]
            else:

                input_timestep = timestep
        else:
            input_timestep = timestep

        xt_B_C_T_H_W = noisy_image_or_video.permute(0, 2, 1, 3, 4)
        timesteps_B_T = input_timestep

        condition_video_mask = None

        if True:
            condition_state_in_B_C_T_H_W = conditional_dict["gt_frames"].type_as(xt_B_C_T_H_W)
            if not conditional_dict["use_video_condition"]:

                condition_state_in_B_C_T_H_W = condition_state_in_B_C_T_H_W * 0

            _, C, _, _, _ = xt_B_C_T_H_W.shape
            condition_video_mask = (
                conditional_dict["condition_video_input_mask_B_C_T_H_W"].repeat(1, C, 1, 1, 1).type_as(xt_B_C_T_H_W)
            )


            xt_B_C_T_H_W = condition_state_in_B_C_T_H_W * condition_video_mask + xt_B_C_T_H_W * (
                1 - condition_video_mask
            )

            if self.config.conditional_frame_timestep >= 0:
                condition_video_mask_B_1_T_1_1 = condition_video_mask.mean(dim=[1, 3, 4], keepdim=True)
                timestep_cond_B_1_T_1_1 = (
                    torch.ones_like(condition_video_mask_B_1_T_1_1) * self.config.conditional_frame_timestep
                )

                timesteps_B_1_T_1_1 = timestep_cond_B_1_T_1_1 * condition_video_mask_B_1_T_1_1 + timesteps_B_T * (
                    1 - condition_video_mask_B_1_T_1_1
                )

                timesteps_B_T = timesteps_B_1_T_1_1.squeeze()
                timesteps_B_T = (
                    timesteps_B_T.unsqueeze(0) if timesteps_B_T.ndim == 1 else timesteps_B_T
                )


        cp_group = self.get_context_parallel_group()
        cp_size = 1 if cp_group is None else cp_group.size()
        if cp_size > 1 and hasattr(model, "enable_context_parallel"):
            model.enable_context_parallel(cp_group)
        if cp_size == 1 and hasattr(model, "disable_context_parallel"):
            model.disable_context_parallel()

        if net_choice in ["real_score", "fake_score"]:



            if cp_size > 1 and n_views == 1:
                input_xt_B_C_T_H_W = broadcast_split_tensor(xt_B_C_T_H_W, seq_dim=2, process_group=cp_group)
                if timesteps_B_T.shape[1] > 1:
                    input_timesteps_B_T = broadcast_split_tensor(timesteps_B_T, seq_dim=1, process_group=cp_group)
                else:
                    input_timesteps_B_T = timesteps_B_T



                gt_frames = conditional_dict.get("gt_frames")
                cond_mask = conditional_dict.get("condition_video_input_mask_B_C_T_H_W")
                view_indices = conditional_dict.get("view_indices_B_T")
                control_input_hdmap_bbox = conditional_dict.get("control_input_hdmap_bbox")
                state_t = self.config.state_t


                input_conditional_dict = {}
                for k, v in conditional_dict.items():
                    if k in {"gt_frames", "condition_video_input_mask_B_C_T_H_W", "view_indices_B_T"}:
                        continue
                    if v is None:
                        input_conditional_dict[k] = None
                    elif not isinstance(v, torch.Tensor):
                        input_conditional_dict[k] = v
                    else:
                        input_conditional_dict[k] = broadcast(v, cp_group)


                if gt_frames is not None and cond_mask is not None and view_indices is not None:
                    _, _, T, _, _ = gt_frames.shape
                    assert T % state_t == 0, f"T must be a multiple of state_t. Got T={T} and state_t={state_t}."
                    if T > 1 and cp_group.size() > 1:
                        n_views = T // state_t
                        gt_frames = rearrange(gt_frames, "B C (V T) H W -> B C V T H W", V=n_views)
                        cond_mask = rearrange(cond_mask, "B C (V T) H W -> B C V T H W", V=n_views)
                        view_indices = rearrange(view_indices, "B (V T) -> B V T", V=n_views)

                        gt_frames = broadcast_split_tensor(gt_frames, seq_dim=3, process_group=cp_group)
                        cond_mask = broadcast_split_tensor(cond_mask, seq_dim=3, process_group=cp_group)
                        view_indices = broadcast_split_tensor(view_indices, seq_dim=2, process_group=cp_group)

                        gt_frames = rearrange(gt_frames, "B C V T H W -> B C (V T) H W", V=n_views)
                        cond_mask = rearrange(cond_mask, "B C V T H W -> B C (V T) H W", V=n_views)
                        view_indices = rearrange(view_indices, "B V T -> B (V T)", V=n_views)
                        if control_input_hdmap_bbox is not None:
                            control_input_hdmap_bbox_B_C_V_T_H_W = rearrange(
                                control_input_hdmap_bbox, "B C (V T) H W -> B C V T H W", V=n_views
                            )
                            control_input_hdmap_bbox_B_C_V_T_H_W = broadcast_split_tensor(
                                control_input_hdmap_bbox_B_C_V_T_H_W, seq_dim=3, process_group=cp_group
                            )
                            control_input_hdmap_bbox = rearrange(
                                control_input_hdmap_bbox_B_C_V_T_H_W, "B C V T H W -> B C (V T) H W", V=n_views
                            )
                    else:
                        gt_frames = broadcast(gt_frames, cp_group)
                        cond_mask = broadcast(cond_mask, cp_group)
                        view_indices = broadcast(view_indices, cp_group)
                        if control_input_hdmap_bbox is not None:
                            control_input_hdmap_bbox = broadcast(control_input_hdmap_bbox, cp_group)

                input_conditional_dict["gt_frames"] = gt_frames
                input_conditional_dict["condition_video_input_mask_B_C_T_H_W"] = cond_mask
                input_conditional_dict["view_indices_B_T"] = view_indices
                input_conditional_dict["control_input_hdmap_bbox"] = control_input_hdmap_bbox
            else:
                input_xt_B_C_T_H_W, input_conditional_dict, input_timesteps_B_T = (
                    xt_B_C_T_H_W,
                    conditional_dict,
                    timesteps_B_T,
                )

            flow_pred = model(
                input_xt_B_C_T_H_W.to(**self.tensor_kwargs),
                input_timesteps_B_T.to(**self.tensor_kwargs),
                block_mask=block_mask,
                **input_conditional_dict,
                **kwargs,
            ).permute(0, 2, 1, 3, 4)


            if cp_size > 1 and n_views == 1:
                flow_pred = cat_outputs_cp_with_grad(flow_pred.contiguous(), seq_dim=1, cp_group=cp_group)

        else:
            assert net_choice == "generator"
            assert kv_cache is not None
            flow_pred = model(
                xt_B_C_T_H_W.to(**self.tensor_kwargs),
                timesteps_B_T.to(**self.tensor_kwargs),
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                current_start=current_start,
                current_end=current_end,
                start_frame_for_rope=start_frame_for_rope,
                block_mask=block_mask,
                **conditional_dict,
                **kwargs,
            ).permute(0, 2, 1, 3, 4)

        pred_x0 = self._convert_flow_pred_to_x0(
            scheduler=scheduler,
            flow_pred=flow_pred.flatten(0, 1),
            xt=noisy_image_or_video.flatten(0, 1),
            timestep=timestep.flatten(0, 1),
        ).unflatten(0, flow_pred.shape[:2])


        if self.config.denoise_replace_gt_frames:
            gt_frames_x0 = conditional_dict["gt_frames"].type_as(pred_x0)
            pred_x0 = (
                gt_frames_x0 * condition_video_mask + pred_x0.permute(0, 2, 1, 3, 4) * (1 - condition_video_mask)
            ).permute(0, 2, 1, 3, 4)

        return flow_pred, pred_x0

    def _compute_kl_grad(
        self,
        noisy_image_or_video: torch.Tensor,
        estimated_clean_image_or_video: torch.Tensor,
        timestep: torch.Tensor,
        conditional_dict: dict,
        unconditional_dict: dict,
        normalization: bool = True,
    ) -> Tuple[torch.Tensor, dict]:


        n_views = noisy_image_or_video.shape[1] // self.config.state_t


        _, pred_fake_image_cond = self.fake_score(
            noisy_image_or_video=noisy_image_or_video,
            conditional_dict=conditional_dict,
            timestep=timestep,
            n_views=n_views,
        )

        if self.config.fake_guidance_scale != 0.0:
            _, pred_fake_image_uncond = self.fake_score(
                noisy_image_or_video=noisy_image_or_video,
                conditional_dict=unconditional_dict,
                timestep=timestep,
                n_views=n_views,
            )
            pred_fake_image = (
                pred_fake_image_cond + (pred_fake_image_cond - pred_fake_image_uncond) * self.config.fake_guidance_scale
            )
        else:
            pred_fake_image = pred_fake_image_cond




        _, pred_real_image_cond = self.real_score(
            noisy_image_or_video=noisy_image_or_video,
            conditional_dict=conditional_dict,
            timestep=timestep,
            n_views=n_views,
        )

        _, pred_real_image_uncond = self.real_score(
            noisy_image_or_video=noisy_image_or_video,
            conditional_dict=unconditional_dict,
            timestep=timestep,
            n_views=n_views,
        )

        pred_real_image = (
            pred_real_image_cond + (pred_real_image_cond - pred_real_image_uncond) * self.config.real_guidance_scale
        )


        if (not dist.is_initialized()) or dist.get_rank() == 0:
            _PRED_REAL_SAVE_MAX = 5
            _save_cnt = getattr(self, "_pred_real_save_cnt", 0)
            if _save_cnt < _PRED_REAL_SAVE_MAX:
                import os

                from gamma_world._src.imaginaire.visualize.video import save_img_or_video

                self._pred_real_save_cnt = _save_cnt + 1
                save_dir = "./tmp"
                os.makedirs(save_dir, exist_ok=True)

                _latent_BCTHW = pred_real_image.permute(0, 2, 1, 3, 4).contiguous()
                _video = self.decode(_latent_BCTHW)
                if n_views > 1:
                    _video = rearrange(_video, "B C (V T) H W -> B C T H (V W)", V=n_views)
                _video = ((_video.float() + 1.0) / 2.0).clamp(0, 1)
                save_img_or_video(
                    _video[0].cpu(),
                    os.path.join(save_dir, f"pred_real_image_iter{_save_cnt:04d}"),
                    fps=24,
                )


        grad = pred_fake_image - pred_real_image


        if normalization:

            p_real = estimated_clean_image_or_video - pred_real_image
            normalizer = torch.abs(p_real).mean(dim=[1, 2, 3, 4], keepdim=True)
            grad = grad / normalizer
        grad = torch.nan_to_num(grad)

        return grad, {
            "dmdtrain_gradient_norm": torch.mean(torch.abs(grad)).detach(),

        }

    def compute_distribution_matching_loss(
        self,
        image_or_video: torch.Tensor,
        conditional_dict: dict,
        unconditional_dict: dict,
        gradient_mask: Optional[torch.Tensor] = None,
        denoised_timestep_from: Optional[int] = None,
        denoised_timestep_to: Optional[int] = None,
    ) -> Tuple[torch.Tensor, dict]:

        original_latent = image_or_video

        batch_size, num_frame = image_or_video.shape[:2]

        with torch.no_grad():

            min_timestep = (
                denoised_timestep_to
                if self.config.ts_schedule and denoised_timestep_to is not None
                else self.config.min_score_timestep
            )
            max_timestep = (
                denoised_timestep_from
                if self.config.ts_schedule_max and denoised_timestep_from is not None
                else self.config.num_train_timestep
            )
            timestep = self._get_timestep(
                min_timestep,
                max_timestep,
                batch_size,
                num_frame,
                self.config.num_frame_per_block,
                uniform_timestep=True,
            )
            if self.config.timestep_shift > 1:

                timestep = (
                    self.config.timestep_shift
                    * (timestep / self.config.num_train_timestep)
                    / (1 + (self.config.timestep_shift - 1) * (timestep / self.config.num_train_timestep))
                    * self.config.num_train_timestep
                )
            timestep = timestep.clamp(self.min_step, self.max_step)

            noise = torch.randn_like(image_or_video)
            noisy_latent = (
                self.scheduler.add_noise(
                    image_or_video.flatten(0, 1),
                    noise.flatten(0, 1),
                    timestep.flatten(0, 1),
                )
                .detach()
                .unflatten(0, (batch_size, num_frame))
            )


            cp_group = self.get_context_parallel_group()
            cp_size = 1 if cp_group is None else cp_group.size()
            if cp_size > 1:
                noisy_latent = broadcast(noisy_latent.contiguous(), cp_group)
                timestep = broadcast(timestep, cp_group)


            grad, dmd_log_dict = self._compute_kl_grad(
                noisy_image_or_video=noisy_latent,
                estimated_clean_image_or_video=original_latent,
                timestep=timestep,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
            )

        if gradient_mask is not None:
            dmd_loss = 0.5 * F.mse_loss(
                original_latent.double()[gradient_mask],
                (original_latent.double() - grad.double()).detach()[gradient_mask],
                reduction="mean",
            )
        else:
            dmd_loss = 0.5 * F.mse_loss(
                original_latent.double(),
                (original_latent.double() - grad.double()).detach(),
                reduction="mean",
            )
        return dmd_loss, dmd_log_dict

    def generator_loss(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        conditional_dict_score_models: dict = None,
        unconditional_dict_score_models: dict = None,
    ) -> Tuple[torch.Tensor, dict]:


        pred_image, gradient_mask, denoised_timestep_from, denoised_timestep_to = self._run_generator(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
        )


        if (not dist.is_initialized()) or dist.get_rank() == 0:
            _PRED_IMAGE_SAVE_MAX = 5
            _save_cnt = getattr(self, "_pred_image_save_cnt", 0)
            if _save_cnt < _PRED_IMAGE_SAVE_MAX:
                import os

                from gamma_world._src.imaginaire.visualize.video import save_img_or_video

                self._pred_image_save_cnt = _save_cnt + 1
                save_dir = "./tmp"
                os.makedirs(save_dir, exist_ok=True)

                _n_views = image_or_video_shape[1] // self.config.state_t
                with torch.no_grad():
                    _latent_BCTHW = pred_image.detach().permute(0, 2, 1, 3, 4).contiguous()
                    _video = self.decode(_latent_BCTHW)
                    if _n_views > 1:
                        _video = rearrange(_video, "B C (V T) H W -> B C T H (V W)", V=_n_views)
                    _video = ((_video.float() + 1.0) / 2.0).clamp(0, 1)
                    save_img_or_video(
                        _video[0].cpu(),
                        os.path.join(save_dir, f"pred_image_iter{_save_cnt:04d}"),
                        fps=24,
                    )


        dmd_loss, dmd_log_dict = self.compute_distribution_matching_loss(
            image_or_video=pred_image,
            conditional_dict=conditional_dict
            if conditional_dict_score_models is None
            else conditional_dict_score_models,
            unconditional_dict=unconditional_dict
            if unconditional_dict_score_models is None
            else unconditional_dict_score_models,
            gradient_mask=gradient_mask,
            denoised_timestep_from=denoised_timestep_from,
            denoised_timestep_to=denoised_timestep_to,
        )

        return dmd_loss, dmd_log_dict

    def critic_loss(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        conditional_dict_score_models: dict = None,
    ) -> Tuple[torch.Tensor, dict]:

        n_views = image_or_video_shape[1] // self.config.state_t


        with torch.no_grad():
            generated_image, _, denoised_timestep_from, denoised_timestep_to = self._run_generator(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
            )


        assert (not self.config.ts_schedule) and (not self.config.ts_schedule_max), "TS schedule is not supported now!"
        min_timestep = (
            denoised_timestep_to
            if self.config.ts_schedule and denoised_timestep_to is not None
            else self.config.min_score_timestep
        )
        max_timestep = (
            denoised_timestep_from
            if self.config.ts_schedule_max and denoised_timestep_from is not None
            else self.config.num_train_timestep
        )
        critic_timestep = self._get_timestep(
            min_timestep,
            max_timestep,
            image_or_video_shape[0],
            image_or_video_shape[1],
            self.config.num_frame_per_block,
            uniform_timestep=True,
        )

        if self.config.timestep_shift > 1:
            critic_timestep = (
                self.config.timestep_shift
                * (critic_timestep / self.config.num_train_timestep)
                / (1 + (self.config.timestep_shift - 1) * (critic_timestep / self.config.num_train_timestep))
                * self.config.num_train_timestep
            )

        critic_timestep = critic_timestep.clamp(self.min_step, self.max_step)

        critic_noise = torch.randn_like(generated_image)
        noisy_generated_image = self.scheduler.add_noise(
            generated_image.flatten(0, 1),
            critic_noise.flatten(0, 1),
            critic_timestep.flatten(0, 1),
        ).unflatten(0, image_or_video_shape[:2])


        cp_group = self.get_context_parallel_group()
        cp_size = 1 if cp_group is None else cp_group.size()
        if cp_size > 1:
            critic_noise = broadcast(critic_noise, cp_group)
            noisy_generated_image = broadcast(noisy_generated_image, cp_group)
            critic_timestep = broadcast(critic_timestep, cp_group)

        _, pred_fake_image = self.fake_score(
            noisy_image_or_video=noisy_generated_image,
            conditional_dict=conditional_dict
            if conditional_dict_score_models is None
            else conditional_dict_score_models,
            timestep=critic_timestep,
            n_views=n_views,
        )


        flow_pred = self._convert_x0_to_flow_pred(
            scheduler=self.scheduler,
            x0_pred=pred_fake_image.flatten(0, 1),
            xt=noisy_generated_image.flatten(0, 1),
            timestep=critic_timestep.flatten(0, 1),
        )
        pred_fake_noise = None

        denoising_loss = self.denoising_loss_func(
            x=generated_image.flatten(0, 1),
            x_pred=pred_fake_image.flatten(0, 1),
            noise=critic_noise.flatten(0, 1),
            noise_pred=pred_fake_noise,
            alphas_cumprod=None,
            timestep=critic_timestep.flatten(0, 1),
            flow_pred=flow_pred,
        )



        critic_log_dict = {}

        return denoising_loss, critic_log_dict

    def get_data_batch_with_latent_view_indices(self, data_batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if "latent_view_indices_B_T" in data_batch:
            return data_batch
        num_video_frames_per_view = int(data_batch["num_video_frames_per_view"].cpu().item())
        n_views = data_batch["view_indices"].shape[1] // num_video_frames_per_view
        view_indices_B_V_T = rearrange(data_batch["view_indices"], "B (V T) -> B V T", V=n_views)
        latent_view_indices_B_V_T = view_indices_B_V_T[:, :, 0 : self.config.state_t]
        latent_view_indices_B_T = rearrange(latent_view_indices_B_V_T, "B V T -> B (V T)")
        data_batch["latent_view_indices_B_T"] = latent_view_indices_B_T
        return data_batch

    def get_data_and_condition(self, data_batch: dict[str, torch.Tensor], with_uncondition: bool = True):
        self._normalize_video_databatch_inplace(data_batch)
        self._augment_image_dim_inplace(data_batch)
        is_image_batch = self.is_image_batch(data_batch)


        self.inplace_compute_text_embeddings_online(
            data_batch,
            use_negative_prompt=with_uncondition,
        )

        data_batch_original = copy.deepcopy(data_batch)

        raw_state = data_batch[self.input_image_key if is_image_batch else self.input_data_key]
        latent_state = self.encode(raw_state).contiguous().float()


        condition, uncondition = self.conditioner.get_condition_with_negative_prompt(data_batch)
        condition = condition.edit_data_type(DataType.IMAGE if is_image_batch else DataType.VIDEO)
        uncondition = uncondition.edit_data_type(DataType.IMAGE if is_image_batch else DataType.VIDEO)

        condition_original, uncondition_original = self.conditioner.get_condition_with_negative_prompt(
            data_batch_original
        )
        condition_original = condition_original.edit_data_type(DataType.IMAGE if is_image_batch else DataType.VIDEO)
        uncondition_original = uncondition_original.edit_data_type(DataType.IMAGE if is_image_batch else DataType.VIDEO)

        state_t = int(
            (data_batch["num_video_frames_per_view"].cpu().item() - 1) // self.tokenizer.temporal_compression_factor + 1
        )
        if NUM_CONDITIONAL_FRAMES_KEY in data_batch:
            num_conditional_frames = data_batch[NUM_CONDITIONAL_FRAMES_KEY]
        else:
            num_conditional_frames = None

        condition = condition.set_video_condition(
            state_t=state_t,
            gt_frames=latent_state.to(**self.tensor_kwargs),
            condition_locations=["first_random_n"],
            random_min_num_conditional_frames_per_view=self.config.min_num_conditional_frames,
            random_max_num_conditional_frames_per_view=self.config.max_num_conditional_frames,
            num_conditional_frames_per_view=num_conditional_frames,
            view_condition_dropout_max=0,
            conditional_frames_probs=self.config.conditional_frames_probs,
        )
        condition_original = condition_original.set_video_condition(
            state_t=state_t,
            gt_frames=latent_state.to(**self.tensor_kwargs),
            condition_locations=["first_random_n"],
            random_min_num_conditional_frames_per_view=self.config.min_num_conditional_frames,
            random_max_num_conditional_frames_per_view=self.config.max_num_conditional_frames,
            num_conditional_frames_per_view=num_conditional_frames,
            view_condition_dropout_max=0,
            conditional_frames_probs=self.config.conditional_frames_probs,
        )

        uncondition = uncondition.set_video_condition(
            state_t=state_t,
            gt_frames=latent_state.to(**self.tensor_kwargs),
            condition_locations=["first_random_n"],
            random_min_num_conditional_frames_per_view=self.config.min_num_conditional_frames,
            random_max_num_conditional_frames_per_view=self.config.max_num_conditional_frames,
            num_conditional_frames_per_view=num_conditional_frames,
            view_condition_dropout_max=0,
            conditional_frames_probs=self.config.conditional_frames_probs,
        )

        uncondition_original = uncondition_original.set_video_condition(
            state_t=state_t,
            gt_frames=latent_state.to(**self.tensor_kwargs),
            condition_locations=["first_random_n"],
            random_min_num_conditional_frames_per_view=self.config.min_num_conditional_frames,
            random_max_num_conditional_frames_per_view=self.config.max_num_conditional_frames,
            num_conditional_frames_per_view=num_conditional_frames,
            view_condition_dropout_max=0,
            conditional_frames_probs=self.config.conditional_frames_probs,
        )

        if with_uncondition:
            return raw_state, latent_state, (condition, uncondition, condition_original, uncondition_original)
        else:
            return raw_state, latent_state, (condition, condition_original)

    def training_step(
        self, data_batch: dict[str, torch.Tensor], iteration: int
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        train_generator = self.is_student_phase(iteration)

        self.eval()
        if iteration % 20 == 0:
            torch.cuda.empty_cache()

        data_batch = self.get_data_batch_with_latent_view_indices(data_batch)

        _, x0_B_C_T_H_W, (condition, uncondition, condition_original, uncondition_original) = (
            self.get_data_and_condition(data_batch, with_uncondition=True)
        )


        cp_group = self.get_context_parallel_group()
        cp_size = 1 if cp_group is None else cp_group.size()
        if cp_size > 1:
            x0_B_C_T_H_W = broadcast(x0_B_C_T_H_W, cp_group)
            condition = condition.broadcast(cp_group, split=False)
            uncondition = uncondition.broadcast(cp_group, split=False)
            condition_original = condition_original.broadcast(cp_group, split=False)
            uncondition_original = uncondition_original.broadcast(cp_group, split=False)



        extra_cond = self._extract_extra_conditional_kwargs(data_batch)

        if train_generator:

            generator_loss, generator_log_dict = self.generator_loss(
                image_or_video_shape=list(x0_B_C_T_H_W.permute(0, 2, 1, 3, 4).shape),
                conditional_dict={**condition.to_dict(), **extra_cond},
                unconditional_dict={**uncondition.to_dict(), **extra_cond},
                conditional_dict_score_models={**condition_original.to_dict(), **extra_cond},
                unconditional_dict_score_models={**uncondition_original.to_dict(), **extra_cond},
            )
            generator_log_dict.update({"generator_loss": generator_loss.detach()})
            return generator_log_dict, generator_loss
        else:

            critic_loss, critic_log_dict = self.critic_loss(
                image_or_video_shape=list(x0_B_C_T_H_W.permute(0, 2, 1, 3, 4).shape),
                conditional_dict={**condition.to_dict(), **extra_cond},
                conditional_dict_score_models={**condition_original.to_dict(), **extra_cond},
            )
            critic_log_dict.update({"critic_loss": critic_loss.detach()})
            return critic_log_dict, critic_loss

    def get_x0_fn_from_batch(
        self,
        data_batch: dict[str, torch.Tensor] | None = None,
        guidance: float = 1.0,
        is_negative_prompt: bool = False,
        conditional_dict: dict = None,
    ):
        assert data_batch is not None or conditional_dict is not None, "data_batch or conditional_dict must be provided"

        if data_batch is not None:
            data_batch = self.get_data_batch_with_latent_view_indices(data_batch)
            if NUM_CONDITIONAL_FRAMES_KEY in data_batch:
                num_conditional_frames = data_batch[NUM_CONDITIONAL_FRAMES_KEY]
            else:
                num_conditional_frames = None

        if conditional_dict is None:
            _, latent_state, _ = self.get_data_and_condition(
                data_batch, with_uncondition=False
            )
            is_image_batch = self.is_image_batch(data_batch)

            condition, uncondition = self.conditioner.get_condition_with_negative_prompt(data_batch)


            condition = condition.edit_data_type(DataType.IMAGE if is_image_batch else DataType.VIDEO)
            _, condition, _, _ = self.broadcast_split_for_model_parallelsim(None, condition, None, None)

            state_t = int(
                (data_batch["num_video_frames_per_view"].cpu().item() - 1) // self.tokenizer.temporal_compression_factor
                + 1
            )

            condition = condition.set_video_condition(
                state_t=state_t,
                gt_frames=latent_state.to(**self.tensor_kwargs),
                condition_locations=["first_random_n"],
                random_min_num_conditional_frames_per_view=self.config.min_num_conditional_frames,
                random_max_num_conditional_frames_per_view=self.config.max_num_conditional_frames,
                num_conditional_frames_per_view=num_conditional_frames,
                view_condition_dropout_max=0,
                conditional_frames_probs=self.config.conditional_frames_probs,
            )

            extra_cond = self._extract_extra_conditional_kwargs(data_batch)
            conditional_dict = {**condition.to_dict(), **extra_cond}



        if parallel_state.is_initialized():
            pass
        else:
            assert not self.net.is_context_parallel_enabled, (
                "parallel_state is not initialized, context parallel should be turned off."
            )

        def x0_fn(
            noise_x: torch.Tensor,
            timestep: torch.Tensor,
            kv_cache: Optional[List[dict]] = None,
            i2v_force_add_into_cache: bool = False,
            **kwargs,
        ) -> torch.Tensor:
            assert self.config.model_type == "i2v"

            noise_x = noise_x.permute(0, 2, 1, 3, 4)
            new_condition_dict = copy.deepcopy(conditional_dict)

            if (
                new_condition_dict["gt_frames"] is not None
                and new_condition_dict["gt_frames"].shape[2] != noise_x.shape[2]
            ):
                assert kwargs.get("start_frame_for_rope", None) is not None, "start_frame_for_rope is not provided"
                start_frame = kwargs.get("start_frame_for_rope")
                end_frame = start_frame + noise_x.shape[2]


                new_condition_dict["gt_frames"] = new_condition_dict["gt_frames"][:, :, start_frame:end_frame, :, :]
                if new_condition_dict["condition_video_input_mask_B_C_T_H_W"] is not None:
                    new_condition_dict["condition_video_input_mask_B_C_T_H_W"] = new_condition_dict[
                        "condition_video_input_mask_B_C_T_H_W"
                    ][:, :, start_frame:end_frame, :, :]
                if new_condition_dict["view_indices_B_T"] is not None:
                    new_condition_dict["view_indices_B_T"] = new_condition_dict["view_indices_B_T"][
                        :, start_frame:end_frame
                    ]

            _, denoised_pred = self.generator(
                noisy_image_or_video=noise_x.permute(0, 2, 1, 3, 4),
                conditional_dict=new_condition_dict,
                timestep=timestep,
                kv_cache=kv_cache,
                **kwargs,
            )
            return denoised_pred

        return x0_fn

    def generate_samples_from_batch(
        self,
        data_batch: dict[str, torch.Tensor] | None = None,
        guidance: float = 1.0,
        seed: int = 1,
        state_shape: Tuple | None = None,
        n_sample: int | None = None,
        is_negative_prompt: bool = False,
        start_latents: Optional[torch.Tensor] = None,
        verbose: bool = False,
        conditional_dict: dict = None,
        image_or_video_shape: Tuple | None = None,
        noise_B_T_C_H_W: Optional[torch.Tensor] = None,
        is_training: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[int], Optional[int]]:




        if data_batch is not None:
            self._normalize_video_databatch_inplace(data_batch)
            self._augment_image_dim_inplace(data_batch)
            is_image_batch = self.is_image_batch(data_batch)
            input_key = self.input_image_key if is_image_batch else self.input_data_key

            if n_sample is None:
                n_sample = data_batch[input_key].shape[0]
            if state_shape is None:
                _T, _H, _W = data_batch[input_key].shape[-3:]
                state_shape = [
                    self.tokenizer.get_latent_num_frames(_T),
                    self.config.state_ch,
                    _H // self.tokenizer.spatial_compression_factor,
                    _W // self.tokenizer.spatial_compression_factor,
                ]
            else:
                state_shape = (state_shape[1], state_shape[0], *state_shape[2:])

        assert state_shape is not None or image_or_video_shape is not None, (
            "data_batch or image_or_video_shape must be provided"
        )

        if noise_B_T_C_H_W is None:
            noise_B_T_C_H_W = misc.arch_invariant_rand(
                (n_sample,) + tuple(state_shape) if image_or_video_shape is None else image_or_video_shape,
                torch.float32,
                self.tensor_kwargs["device"],
                seed,
            )
            misc.set_random_seed(seed=seed, by_rank=False)

        self.frame_seq_length = int(noise_B_T_C_H_W.shape[-1] * noise_B_T_C_H_W.shape[-2] / 4)


        cp_group = self.get_context_parallel_group()
        cp_size = 1 if cp_group is None else cp_group.size()
        if cp_size > 1:
            self.net.enable_context_parallel(cp_group)

            noise_B_T_C_H_W = broadcast(noise_B_T_C_H_W.contiguous(), cp_group)
            if start_latents is not None:
                start_latents = broadcast(start_latents.contiguous(), cp_group)
        else:

            assert not getattr(self.net, "is_context_parallel_enabled", False), (
                "context parallel should be disabled if parallel_state is not initialized"
            )

        if cp_group is not None and not is_tp_cp_pp_rank0():
            verbose = False

        flow_pred_fn = self.get_x0_fn_from_batch(
            data_batch=data_batch,
            guidance=guidance,
            is_negative_prompt=is_negative_prompt,
            conditional_dict=conditional_dict,
        )

        def x0_fn(
            noisy_image_or_video: torch.Tensor,
            timestep: torch.Tensor,
            kv_cache: Optional[List[dict]] = None,
            crossattn_cache: Optional[List[dict]] = None,
            current_start: Optional[int] = None,
            current_end: Optional[int] = None,
            start_frame_for_rope: Optional[int] = None,
        ):
            return flow_pred_fn(
                noise_x=noisy_image_or_video,
                timestep=timestep,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                current_start=current_start,
                current_end=current_end,
                start_frame_for_rope=start_frame_for_rope,
            )

        batch_size, num_frames, num_channels, height, width = noise_B_T_C_H_W.shape

        num_input_frames = 0
        num_output_frames = num_frames + num_input_frames

        output_B_T_C_H_W = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise_B_T_C_H_W.device,
            dtype=noise_B_T_C_H_W.dtype,
        )

        assert num_frames % self.config.num_frame_per_block == 0
        num_blocks = num_frames // self.config.num_frame_per_block


        self._initialize_kv_cache(
            batch_size=batch_size,
            dtype=self.tensor_kwargs["dtype"],
            device=self.tensor_kwargs["device"],
            num_training_frames=num_output_frames,
            is_training=is_training,
        )






        self.crossattn_cache = None


        current_start_frame = 0


        all_num_frames = [self.config.num_frame_per_block] * num_blocks

        exit_flags = []
        if self.config.independent_denoising_step_list:
            for block_index in range(len(all_num_frames)):
                if is_training:
                    exit_flag = self.generate_and_sync_list(
                        1, len(self.denoising_step_list[block_index]), device=noise_B_T_C_H_W.device
                    )
                else:
                    exit_flag = [len(self.denoising_step_list[block_index]) - 1]
                exit_flags.append(exit_flag[0])
        else:
            num_denoising_steps = len(self.denoising_step_list)
            if is_training:
                exit_flags = self.generate_and_sync_list(
                    len(all_num_frames), num_denoising_steps, device=noise_B_T_C_H_W.device
                )
            else:
                exit_flags = [num_denoising_steps - 1] * len(all_num_frames)



        start_gradient_frame_index = 0

        for block_index, current_num_frames in enumerate(all_num_frames):
            current_end_frame = current_start_frame + current_num_frames
            noisy_input = noise_B_T_C_H_W[
                :,
                current_start_frame - num_input_frames : current_end_frame - num_input_frames,
            ]


            denoising_step_list = (
                self.denoising_step_list[block_index]
                if self.config.independent_denoising_step_list
                else self.denoising_step_list
            )
            for index, current_timestep in enumerate(denoising_step_list):
                if self.config.same_step_across_blocks:
                    exit_flag = index == exit_flags[0]
                else:
                    exit_flag = (
                        index == exit_flags[block_index]
                    )
                timestep = (
                    torch.ones(
                        [batch_size, current_num_frames],
                        device=noise_B_T_C_H_W.device,
                        dtype=torch.int64,
                    )
                    * current_timestep
                )

                if not exit_flag:
                    with torch.no_grad():
                        denoised_pred = x0_fn(
                            noisy_image_or_video=noisy_input,
                            timestep=timestep,
                            kv_cache=self.kv_cache1,
                            crossattn_cache=self.crossattn_cache,
                            current_start=current_start_frame * self.frame_seq_length // cp_size,
                            current_end=current_end_frame * self.frame_seq_length // cp_size,
                            start_frame_for_rope=current_start_frame,
                        )
                        next_timestep = denoising_step_list[index + 1]
                        current_noise = torch.randn_like(denoised_pred.flatten(0, 1))
                        if cp_size > 1:
                            current_noise = broadcast(current_noise.contiguous(), cp_group)
                        noisy_input = self.scheduler.add_noise(
                            denoised_pred.flatten(0, 1),
                            current_noise,
                            next_timestep
                            * torch.ones(
                                [batch_size * current_num_frames],
                                device=noise_B_T_C_H_W.device,
                                dtype=torch.long,
                            ),
                        ).unflatten(0, denoised_pred.shape[:2])

                else:

                    if current_start_frame < start_gradient_frame_index or not is_training:
                        with torch.no_grad():
                            denoised_pred = x0_fn(
                                noisy_image_or_video=noisy_input,
                                timestep=timestep,
                                kv_cache=self.kv_cache1,
                                crossattn_cache=self.crossattn_cache,
                                current_start=current_start_frame * self.frame_seq_length // cp_size,
                                current_end=current_end_frame * self.frame_seq_length // cp_size,
                                start_frame_for_rope=current_start_frame,
                            )
                    else:
                        denoised_pred = x0_fn(
                            noisy_image_or_video=noisy_input,
                            timestep=timestep,
                            kv_cache=self.kv_cache1,
                            crossattn_cache=self.crossattn_cache,
                            current_start=current_start_frame * self.frame_seq_length // cp_size,
                            current_end=current_end_frame * self.frame_seq_length // cp_size,
                            start_frame_for_rope=current_start_frame,
                        )
                    break


            output_B_T_C_H_W[:, current_start_frame:current_end_frame] = denoised_pred


            context_timestep = torch.ones_like(timestep) * self.config.context_noise
            if self.config.context_noise > 0:

                current_noise = torch.randn_like(denoised_pred.flatten(0, 1))
                if cp_size > 1:
                    current_noise = broadcast(current_noise.contiguous(), cp_group)
                denoised_pred = self.scheduler.add_noise(
                    denoised_pred.flatten(0, 1),
                    current_noise,
                    context_timestep
                    * torch.ones(
                        [batch_size * current_num_frames],
                        device=noise_B_T_C_H_W.device,
                        dtype=torch.long,
                    ),
                ).unflatten(0, denoised_pred.shape[:2])

            with torch.no_grad():
                x0_fn(
                    noisy_image_or_video=denoised_pred,
                    timestep=context_timestep,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length // cp_size,
                    current_end=current_end_frame * self.frame_seq_length // cp_size,
                    start_frame_for_rope=current_start_frame,
                )


            current_start_frame = current_end_frame



        denoised_timestep_from, denoised_timestep_to = None, None

        if is_training:
            return output_B_T_C_H_W, denoised_timestep_from, denoised_timestep_to
        else:
            return output_B_T_C_H_W.permute(0, 2, 1, 3, 4)

    def _initialize_kv_cache(self, batch_size, dtype, device, num_training_frames=None, is_training=False):

        if num_training_frames is None:
            num_training_frames = self.config.num_training_frames

        local_attn_size = getattr(self.net, "local_attn_size", -1)
        if local_attn_size == -1 or is_training:

            kv_cache_size = self.frame_seq_length * num_training_frames
        else:
            if local_attn_size > num_training_frames:
                raise ValueError(
                    f"local_attn_size {local_attn_size} is larger than num_training_frames {num_training_frames}, "
                    f"which is not supported"
                )
            kv_cache_size = self.frame_seq_length * local_attn_size

        cp_group = self.get_context_parallel_group()
        cp_size = 1 if cp_group is None else cp_group.size()
        if cp_size > 1:
            assert kv_cache_size % cp_size == 0, "kv_cache_size must be divisible by cp_size"
            kv_cache_size = kv_cache_size // cp_size

        kv_cache1 = []
        for _ in range(self.net.num_layers):
            kv_cache1.append(
                {
                    "k": torch.zeros(
                        [batch_size, int(kv_cache_size), self.net.num_heads, 128],
                        dtype=dtype,
                        device=device,
                    ),
                    "v": torch.zeros(
                        [batch_size, int(kv_cache_size), self.net.num_heads, 128],
                        dtype=dtype,
                        device=device,
                    ),
                    "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                    "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
                }
            )

        self.kv_cache1 = kv_cache1

    def _initialize_crossattn_cache(self, batch_size, dtype, device):

        crossattn_cache = []

        for _ in range(self.net.num_layers):
            crossattn_cache.append(
                {
                    "k": torch.zeros([batch_size, 512, self.net.num_heads, 128], dtype=dtype, device=device),
                    "v": torch.zeros([batch_size, 512, self.net.num_heads, 128], dtype=dtype, device=device),
                    "is_init": False,
                }
            )
        self.crossattn_cache = crossattn_cache

    def generate_and_sync_list(self, num_blocks, num_denoising_steps, device) -> List[int]:
        rank = dist.get_rank() if dist.is_initialized() else 0

        if rank == 0:

            indices = torch.randint(low=0, high=num_denoising_steps, size=(num_blocks,), device=device)
            if self.config.last_step_only:
                indices = torch.ones_like(indices) * (num_denoising_steps - 1)
        else:
            indices = torch.empty(num_blocks, dtype=torch.long, device=device)

        if dist.is_initialized():
            dist.broadcast(indices, src=0)
        return indices.tolist()
