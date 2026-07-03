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

from typing import Callable, Dict, List, Optional, Tuple

import attrs
import torch
from einops import rearrange
from megatron.core import parallel_state

from gamma_world._src.imaginaire.utils import log
from gamma_world._src.predict2.conditioner import DataType
from gamma_world._src.predict2.configs.video2world.defaults.conditioner import Video2WorldCondition
from gamma_world._src.gamma_world.models.joint_causal_cosmos_model import (
    CausalJointCosmosModel,
    CausalJointCosmosModelConfig,
)

NUM_CONDITIONAL_FRAMES_KEY: str = "num_conditional_frames"


@attrs.define(slots=False)
class CausalJointCosmosModelHdmapConfig(CausalJointCosmosModelConfig):
    preset_hint_keys: list[str] = None
    hdmap_process_method: str = "vae_encoding"
    hdmap_selection_mode: str = "all"


class CausalJointCosmosModelHdmap(CausalJointCosmosModel):
    def __init__(self, config: CausalJointCosmosModelHdmapConfig):
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
    ) -> Tuple[torch.Tensor, torch.Tensor, Video2WorldCondition]:
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
                    data_batch[hint_key] = self.encode(control_input).contiguous().to(**self.tensor_kwargs)

            elif self.hdmap_process_method == "pixel_shuffle":
                if hint_value.shape[1] != 3:
                    warning_msg = f"Hint value {hint_key} is shape {hint_value.shape}, no need to pixel shuffle again"
                    log.warning(warning_msg)
                else:
                    control_input = self.process_hint_value(data_batch, hint_key)
                    if self.hdmap_selection_mode == "first_frame":
                        indices = [0] + [i for i in range(1, hint_value.shape[2], 4)]
                    elif self.hdmap_selection_mode == "last_frame":
                        indices = [0] + [i for i in range(4, hint_value.shape[2], 4)]
                    elif self.hdmap_selection_mode == "all":
                        indices = [i for i in range(hint_value.shape[2])]
                    else:
                        raise ValueError(f"Invalid hdmap selection mode: {self.hdmap_selection_mode}")
                    indices = torch.tensor(indices, dtype=torch.long)
                    control_input = control_input[:, :, indices, :, :]
                    control_input_down = rearrange(
                        control_input, "b c t (h h8) (w w8) -> b (c h8 w8) t h w", h8=8, w8=8
                    )
                    data_batch[hint_key] = control_input_down.contiguous().to(**self.tensor_kwargs)
            else:
                raise ValueError(f"Invalid hdmap process method: {self.hdmap_process_method}")

        return super().get_data_and_condition(data_batch)

    def get_velocity_fn_from_batch(
        self,
        data_batch: Dict,
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


        state_t = int(
            (data_batch["num_video_frames_per_view"].cpu().item() - 1) // self.tokenizer.temporal_compression_factor + 1
        )


        condition = condition.set_video_condition(
            state_t=state_t,
            gt_frames=x0,
            condition_locations=["first_random_n"],
            random_min_num_conditional_frames_per_view=self.config.min_num_conditional_frames,
            random_max_num_conditional_frames_per_view=self.config.max_num_conditional_frames,
            num_conditional_frames_per_view=num_conditional_frames,
            view_condition_dropout_max=0,
            conditional_frames_probs=self.config.conditional_frames_probs,
        )
        uncondition = uncondition.set_video_condition(
            state_t=state_t,
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
            kv_cache: Optional[List[dict]] = None,
            kv_cache_uncond: Optional[List[dict]] = None,
            use_uncond_kvcache: bool = False,
            noise: Optional[torch.Tensor] = None,
            **kwargs,
        ) -> torch.Tensor:
            if use_uncond_kvcache:
                assert kv_cache_uncond is not None
            else:
                kv_cache_uncond = kv_cache



            new_condition = condition
            new_uncondition = uncondition
            if condition.gt_frames is not None and condition.gt_frames.shape[2] != noise_x.shape[2]:
                assert kwargs.get("start_frame_for_rope", None) is not None, "start_frame_for_rope is not provided"
                start_frame = kwargs.get("start_frame_for_rope")
                end_frame = start_frame + noise_x.shape[2]


                new_condition_dict = condition.to_dict()
                new_condition_dict["gt_frames"] = condition.gt_frames[:, :, start_frame:end_frame, :, :]
                if condition.condition_video_input_mask_B_C_T_H_W is not None:
                    new_condition_dict["condition_video_input_mask_B_C_T_H_W"] = (
                        condition.condition_video_input_mask_B_C_T_H_W[:, :, start_frame:end_frame, :, :]
                    )
                if hasattr(condition, "view_indices_B_T") and condition.view_indices_B_T is not None:
                    new_condition_dict["view_indices_B_T"] = condition.view_indices_B_T[:, start_frame:end_frame]
                if hasattr(condition, "control_input_hdmap_bbox") and condition.control_input_hdmap_bbox is not None:
                    new_condition_dict["control_input_hdmap_bbox"] = condition.control_input_hdmap_bbox[
                        :, :, start_frame:end_frame, :, :
                    ]
                new_condition = type(condition)(**new_condition_dict)


                if guidance != 1.0:
                    new_uncondition_dict = uncondition.to_dict()
                    new_uncondition_dict["gt_frames"] = uncondition.gt_frames[:, :, start_frame:end_frame, :, :]
                    if uncondition.condition_video_input_mask_B_C_T_H_W is not None:
                        new_uncondition_dict["condition_video_input_mask_B_C_T_H_W"] = (
                            uncondition.condition_video_input_mask_B_C_T_H_W[:, :, start_frame:end_frame, :, :]
                        )
                    if hasattr(uncondition, "view_indices_B_T") and uncondition.view_indices_B_T is not None:
                        new_uncondition_dict["view_indices_B_T"] = uncondition.view_indices_B_T[
                            :, start_frame:end_frame
                        ]
                    if (
                        hasattr(uncondition, "control_input_hdmap_bbox")
                        and uncondition.control_input_hdmap_bbox is not None
                    ):
                        new_uncondition_dict["control_input_hdmap_bbox"] = uncondition.control_input_hdmap_bbox[
                            :, :, start_frame:end_frame, :, :
                        ]
                    new_uncondition = type(uncondition)(**new_uncondition_dict)

            cond_v = self.denoise(
                xt_B_C_T_H_W=noise_x,
                timesteps_B_T=timestep,
                condition=new_condition,
                kv_cache=kv_cache,
                noise=noise,
                **kwargs,
            )

            if guidance != 1.0 and not skip_uncond:
                uncond_v = self.denoise(
                    xt_B_C_T_H_W=noise_x,
                    timesteps_B_T=timestep,
                    condition=new_uncondition,
                    kv_cache=kv_cache_uncond,
                    noise=noise,
                    **kwargs,
                )
                velocity_pred = uncond_v + guidance * (cond_v - uncond_v)
            else:
                velocity_pred = cond_v

            return velocity_pred

        return velocity_fn
