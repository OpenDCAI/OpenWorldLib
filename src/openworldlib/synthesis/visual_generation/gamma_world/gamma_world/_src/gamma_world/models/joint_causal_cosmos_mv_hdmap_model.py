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

from typing import Callable, Optional

import attrs
import torch
from einops import rearrange
from megatron.core import parallel_state

from gamma_world._src.imaginaire.utils import distributed, log
from gamma_world._src.predict2.conditioner import DataType
from gamma_world._src.predict2_multiview.configs.vid2vid.defaults.conditioner import (
    MultiViewCondition,
)
from gamma_world._src.gamma_world.models.joint_causal_cosmos_mv_model import (
    CausalJointCosmosMVModel,
    CausalJointCosmosMVModelConfig,
)

NUM_CONDITIONAL_FRAMES_KEY: str = "num_conditional_frames"
DEBUG = False


@attrs.define(slots=False)
class CausalJointCosmosMVModelHdmapConfig(CausalJointCosmosMVModelConfig):
    preset_hint_keys: list[str] | None = None
    hdmap_process_method: str = "vae_encoding"
    hdmap_selection_mode: str = "all"


class CausalJointCosmosMVModelHdmap(CausalJointCosmosMVModel):
    def __init__(self, config: CausalJointCosmosMVModelHdmapConfig):
        super().__init__(config)
        self.preset_hint_keys = config.preset_hint_keys
        self.hdmap_process_method = config.hdmap_process_method
        self.hdmap_selection_mode = config.hdmap_selection_mode
        assert self.preset_hint_keys, "The preset hint keys list is empty. Please ensure it contains valid keys."

    def process_hint_value(self, data_batch, hint_key):

        hint_value = data_batch[hint_key]
        if torch.is_floating_point(hint_value):
            assert torch.all((hint_value >= -1.0001) & (hint_value <= 1.0001)), (
                f"Video data is not in the range [-1, 1]. get data range [{hint_value.min()}, {hint_value.max()}]"
            )
        else:
            assert hint_value.dtype == torch.uint8, "Video data is not in uint8 format."
            hint_value = hint_value.to(**self.tensor_kwargs) / 127.5 - 1.0
        data_batch[hint_key] = hint_value
        return hint_value

    def get_data_and_condition(
        self, data_batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, MultiViewCondition]:
        for hint_key in self.preset_hint_keys:
            assert hint_key in data_batch, f"preset hint key {hint_key} not in data_batch"

            hint_value = data_batch[hint_key]
            if self.hdmap_process_method == "vae_encoding":

                if torch.is_floating_point(hint_value) and not (
                    torch.all((hint_value >= -1.0001) & (hint_value <= 1.0001))
                ):
                    warning_msg = f"Hint value {hint_key} is not in the range [-1, 1]. get data range [{hint_value.min()}, {hint_value.max()}]"
                    log.warning(warning_msg)
                else:
                    control_input = self.process_hint_value(data_batch, hint_key)

                    if DEBUG:
                        import torchvision

                        rank = distributed.get_rank()
                        b, c, t, h, w = control_input.shape

                        for i in range(b):
                            save_path = (
                                f"control_input_b{i}_rank{rank}.mp4" if b > 1 else f"control_input_rank{rank}.mp4"
                            )

                            video = control_input[i].detach().cpu().clamp(-1, 1)

                            video = (video + 1) / 2

                            video_thwc = video.permute(1, 2, 3, 0)
                            torchvision.io.write_video(
                                save_path,
                                (video_thwc * 255).to(torch.uint8),
                                fps=10,
                            )
                            log.info(f"Saved control_input to {save_path}")
                    data_batch[hint_key] = self.encode(control_input).contiguous().to(**self.tensor_kwargs)

            elif self.hdmap_process_method == "pixel_shuffle":
                if hint_value.shape[1] != 3:
                    warning_msg = f"Hint value {hint_key} is shape {hint_value.shape}, no need to pixel shuffle again"
                    log.warning(warning_msg)
                else:
                    control_input = self.process_hint_value(data_batch, hint_key)

                    num_video_frames_per_view = int(data_batch["num_video_frames_per_view"].cpu().item())
                    num_views = data_batch["view_indices"].shape[1] // num_video_frames_per_view


                    control_input_unfolded = rearrange(control_input, "B C (V T) H W -> B C V T H W", V=num_views)


                    T_per_view = control_input_unfolded.shape[3]
                    if self.hdmap_selection_mode == "first_frame":
                        indices = [0] + [i for i in range(1, T_per_view, 4)]
                    elif self.hdmap_selection_mode == "last_frame":
                        indices = [0] + [i for i in range(4, T_per_view, 4)]
                    elif self.hdmap_selection_mode == "all":
                        indices = [i for i in range(T_per_view)]
                    else:
                        raise ValueError(f"Invalid hdmap selection mode: {self.hdmap_selection_mode}")
                    indices = torch.tensor(indices, dtype=torch.long)


                    control_input_selected = control_input_unfolded[:, :, :, indices, :, :]


                    control_input_down = rearrange(
                        control_input_selected, "B C V T (H h8) (W w8) -> B (C h8 w8) V T H W", h8=8, w8=8
                    )


                    control_input_final = rearrange(control_input_down, "B C V T H W -> B C (V T) H W")

                    data_batch[hint_key] = control_input_final.contiguous().to(**self.tensor_kwargs)
            else:
                raise ValueError(f"Invalid hdmap process method: {self.hdmap_process_method}")

        return super().get_data_and_condition(data_batch)

    def get_velocity_fn_from_batch(
        self,
        data_batch: dict,
        n_views: int,
        guidance: float = 1.5,
        is_negative_prompt: bool = False,
    ) -> Callable:
        data_batch_with_latent_view_indices = self.get_data_batch_with_latent_view_indices(data_batch)

        if NUM_CONDITIONAL_FRAMES_KEY in data_batch_with_latent_view_indices:
            num_conditional_frames = data_batch_with_latent_view_indices[NUM_CONDITIONAL_FRAMES_KEY]
            log.debug(f"Using {num_conditional_frames=} from data batch")
        else:
            num_conditional_frames = 1

        if is_negative_prompt:
            condition, uncondition = self.conditioner.get_condition_with_negative_prompt(
                data_batch_with_latent_view_indices
            )
        else:
            condition, uncondition = self.conditioner.get_condition_uncondition(data_batch_with_latent_view_indices)

        is_image_batch = self.is_image_batch(data_batch_with_latent_view_indices)
        condition = condition.edit_data_type(DataType.IMAGE if is_image_batch else DataType.VIDEO)
        uncondition = uncondition.edit_data_type(DataType.IMAGE if is_image_batch else DataType.VIDEO)

        _, x0, _ = self.get_data_and_condition(data_batch_with_latent_view_indices)


        condition = condition.set_video_condition(
            state_t=self.state_t,
            gt_frames=x0,
            condition_locations=["first_random_n"],
            random_min_num_conditional_frames_per_view=self.config.min_num_conditional_frames,
            random_max_num_conditional_frames_per_view=self.config.max_num_conditional_frames,
            num_conditional_frames_per_view=num_conditional_frames,
            view_condition_dropout_max=0,
            conditional_frames_probs=self.config.conditional_frames_probs,
        )
        uncondition = uncondition.set_video_condition(
            state_t=self.state_t,
            gt_frames=x0,
            condition_locations=["first_random_n"],
            random_min_num_conditional_frames_per_view=self.config.min_num_conditional_frames,
            random_max_num_conditional_frames_per_view=self.config.max_num_conditional_frames,
            num_conditional_frames_per_view=num_conditional_frames,
            view_condition_dropout_max=0,
            conditional_frames_probs=self.config.conditional_frames_probs,
        )
        condition = condition.edit_for_inference(
            is_cfg_conditional=True,
            condition_locations=["first_random_n"],
            num_conditional_frames_per_view=num_conditional_frames,
        )
        uncondition = uncondition.edit_for_inference(
            is_cfg_conditional=False,
            condition_locations=["first_random_n"],
            num_conditional_frames_per_view=num_conditional_frames,
        )
        _, condition, _, _ = self.broadcast_split_for_model_parallelsim(x0, condition, None, None)
        if guidance != 1.0:
            _, uncondition, _, _ = self.broadcast_split_for_model_parallelsim(x0, uncondition, None, None)


        if parallel_state.is_initialized():
            pass
        else:
            assert not self.net.is_context_parallel_enabled, (
                "parallel_state is not initialized, context parallel should be turned off."
            )

        def velocity_fn(
            noise_x: torch.Tensor,
            timestep: torch.Tensor,
            skip_uncond: bool = False,
            kv_cache: Optional[list[dict]] = None,
            kv_cache_uncond: Optional[list[dict]] = None,
            use_uncond_kvcache: bool = False,
            noise: Optional[torch.Tensor] = None,
            **kwargs,
        ) -> torch.Tensor:
            if use_uncond_kvcache:
                assert kv_cache_uncond is not None
            else:
                kv_cache_uncond = kv_cache


            def _unfold(tensor_flat: torch.Tensor) -> torch.Tensor:
                return rearrange(tensor_flat, "B C (V T) H W -> B V C T H W", V=n_views)

            def _fold(tensor_unfold: torch.Tensor) -> torch.Tensor:
                return rearrange(tensor_unfold, "B V C T H W -> B C (V T) H W")

            noise_unfold = _unfold(noise_x)
            gt_frames_unfold: torch.Tensor | None = None
            if condition.gt_frames is not None:
                gt_frames_unfold = _unfold(condition.gt_frames)

            new_condition = condition
            new_uncondition = uncondition
            start_frame = kwargs.get("start_frame_for_rope", 0)
            end_frame = start_frame + noise_unfold.shape[3]
            new_condition_dict = condition.to_dict()
            new_uncondition_dict = uncondition.to_dict()

            if gt_frames_unfold is not None and gt_frames_unfold.shape[3] != noise_unfold.shape[3]:
                assert kwargs.get("start_frame_for_rope", None) is not None, "start_frame_for_rope is not provided"
                sliced_gt = gt_frames_unfold[:, :, :, start_frame:end_frame, :, :]
                new_condition_dict["gt_frames"] = _fold(sliced_gt)
                if condition.condition_video_input_mask_B_C_T_H_W is not None:
                    mask_unfold = _unfold(condition.condition_video_input_mask_B_C_T_H_W)
                    sliced_mask = mask_unfold[:, :, :, start_frame:end_frame, :, :]
                    new_condition_dict["condition_video_input_mask_B_C_T_H_W"] = _fold(sliced_mask)
                if guidance != 1.0:
                    if uncondition.gt_frames is not None:
                        un_gt_unfold = _unfold(uncondition.gt_frames)
                        sliced_un_gt = un_gt_unfold[:, :, :, start_frame:end_frame, :, :]
                        new_uncondition_dict["gt_frames"] = _fold(sliced_un_gt)
                    if uncondition.condition_video_input_mask_B_C_T_H_W is not None:
                        un_mask_unfold = _unfold(uncondition.condition_video_input_mask_B_C_T_H_W)
                        sliced_un_mask = un_mask_unfold[:, :, :, start_frame:end_frame, :, :]
                        new_uncondition_dict["condition_video_input_mask_B_C_T_H_W"] = _fold(sliced_un_mask)
            if hasattr(condition, "control_input_hdmap_bbox") and condition.control_input_hdmap_bbox is not None:
                control_input_hdmap_bbox_unfold = _unfold(condition.control_input_hdmap_bbox)
                if control_input_hdmap_bbox_unfold.shape[3] != noise_unfold.shape[3]:
                    new_condition_dict["control_input_hdmap_bbox"] = _fold(
                        control_input_hdmap_bbox_unfold[:, :, :, start_frame:end_frame, :, :]
                    )
                    if guidance != 1.0:
                        new_uncondition_dict["control_input_hdmap_bbox"] = _fold(
                            control_input_hdmap_bbox_unfold[:, :, :, start_frame:end_frame, :, :]
                        )

            if hasattr(condition, "view_indices_B_T") and condition.view_indices_B_T is not None:
                view_indices_unfold = rearrange(condition.view_indices_B_T, "B (V T) -> B V T", V=n_views)
                if view_indices_unfold.shape[2] != noise_unfold.shape[3]:
                    new_condition_dict["view_indices_B_T"] = rearrange(
                        view_indices_unfold[:, :, start_frame:end_frame], "B V T -> B (V T)"
                    )

                    if guidance != 1.0:
                        if hasattr(uncondition, "view_indices_B_T") and uncondition.view_indices_B_T is not None:
                            flat_view_indices_B_V_T = rearrange(
                                uncondition.view_indices_B_T, "B (V T) -> B V T", V=n_views
                            )
                            new_uncondition_dict["view_indices_B_T"] = rearrange(
                                flat_view_indices_B_V_T[:, :, start_frame:end_frame], "B V T -> B (V T)"
                            )

            new_condition = type(condition)(**new_condition_dict)
            if guidance != 1.0:
                new_uncondition = type(uncondition)(**new_uncondition_dict)

            noise_fold = _fold(noise_unfold)

            cond_v = self.denoise(
                xt_B_C_T_H_W=noise_fold,
                timesteps_B_T=timestep,
                condition=new_condition,
                kv_cache=kv_cache,
                noise=noise,
                n_views=n_views,
                **kwargs,
            )

            if guidance != 1.0 and not skip_uncond:
                uncond_v = self.denoise(
                    xt_B_C_T_H_W=noise_fold,
                    timesteps_B_T=timestep,
                    condition=new_uncondition,
                    kv_cache=kv_cache_uncond,
                    noise=noise,
                    n_views=n_views,
                    **kwargs,
                )
                velocity_pred = uncond_v + guidance * (cond_v - uncond_v)
            else:
                velocity_pred = cond_v

            return velocity_pred

        return velocity_fn
