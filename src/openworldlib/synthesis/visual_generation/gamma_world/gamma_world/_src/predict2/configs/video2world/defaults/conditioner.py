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

import random
from dataclasses import dataclass
from typing import Dict, Optional

import torch
from einops import rearrange
from hydra.core.config_store import ConfigStore
from torch.distributed import get_process_group_ranks

from gamma_world._src.imaginaire.lazy_config import LazyCall as L
from gamma_world._src.imaginaire.lazy_config import LazyDict
from gamma_world._src.imaginaire.utils.context_parallel import broadcast_split_tensor, find_split
from gamma_world._src.predict2.conditioner import (
    BooleanFlag,
    GeneralConditioner,
    ReMapkey,
    Text2WorldCondition,
    TextAttr,
    TextAttrEmptyStringDrop,
)
from gamma_world._src.predict2.models.video2world_wan2pt1_model import WAN2PT1_I2V_COND_LATENT_KEY
from gamma_world._src.predict2.networks.clip import Wan2pt1CLIPEmb

@dataclass(frozen=True)
class Video2WorldCondition(Text2WorldCondition):
    use_video_condition: bool = False

    gt_frames: Optional[torch.Tensor] = None
    condition_video_input_mask_B_C_T_H_W: Optional[torch.Tensor] = None
    num_conditional_frames_B: Optional[torch.Tensor] = None

    def set_video_condition(
        self,
        gt_frames: torch.Tensor,
        random_min_num_conditional_frames: int,
        random_max_num_conditional_frames: int,
        num_conditional_frames: Optional[int] = None,
        conditional_frames_probs: Optional[Dict[int, float]] = None,
    ) -> "Video2WorldCondition":
        kwargs = self.to_dict(skip_underscore=False)
        kwargs["gt_frames"] = gt_frames

        B, _, T, H, W = gt_frames.shape
        condition_video_input_mask_B_C_T_H_W = torch.zeros(
            B, 1, T, H, W, dtype=gt_frames.dtype, device=gt_frames.device
        )
        if T == 1:
            num_conditional_frames_B = torch.zeros(B, dtype=torch.int32)
        else:
            if num_conditional_frames is not None:
                if isinstance(num_conditional_frames, torch.Tensor):
                    num_conditional_frames_B = torch.ones(B, dtype=torch.int32) * num_conditional_frames.cpu()
                else:
                    num_conditional_frames_B = torch.ones(B, dtype=torch.int32) * num_conditional_frames
            elif conditional_frames_probs is not None:

                frames_options = list(conditional_frames_probs.keys())
                weights = list(conditional_frames_probs.values())
                num_conditional_frames_B = torch.tensor(
                    random.choices(frames_options, weights=weights, k=B), dtype=torch.int32
                )
            else:
                num_conditional_frames_B = torch.randint(
                    random_min_num_conditional_frames, random_max_num_conditional_frames + 1, size=(B,)
                )
        for idx in range(B):
            condition_video_input_mask_B_C_T_H_W[idx, :, : num_conditional_frames_B[idx], :, :] += 1

        kwargs["condition_video_input_mask_B_C_T_H_W"] = condition_video_input_mask_B_C_T_H_W
        kwargs["num_conditional_frames_B"] = num_conditional_frames_B.to(device=gt_frames.device)
        return type(self)(**kwargs)

    def edit_for_inference(
        self, is_cfg_conditional: bool = True, num_conditional_frames: int = 1
    ) -> "Video2WorldCondition":
        _condition = self.set_video_condition(
            gt_frames=self.gt_frames,
            random_min_num_conditional_frames=0,
            random_max_num_conditional_frames=0,
            num_conditional_frames=num_conditional_frames,
        )
        if not is_cfg_conditional:

            _condition.use_video_condition.fill_(True)
        return _condition

    def broadcast(self, process_group: torch.distributed.ProcessGroup) -> "Video2WorldCondition":
        if self.is_broadcasted:
            return self

        gt_frames = self.gt_frames
        condition_video_input_mask_B_C_T_H_W = self.condition_video_input_mask_B_C_T_H_W
        kwargs = self.to_dict(skip_underscore=False)
        kwargs["gt_frames"] = None
        kwargs["condition_video_input_mask_B_C_T_H_W"] = None
        new_condition = Text2WorldCondition.broadcast(
            type(self)(**kwargs),
            process_group,
        )

        kwargs = new_condition.to_dict(skip_underscore=False)
        _, _, T, _, _ = gt_frames.shape
        if process_group is not None:
            cp_ranks = get_process_group_ranks(process_group)
            cp_size = len(cp_ranks)
            use_spatial_split = (
                cp_size > condition_video_input_mask_B_C_T_H_W.shape[2]
                or condition_video_input_mask_B_C_T_H_W.shape[2] % cp_size != 0
            )
            after_split_shape = (
                find_split(condition_video_input_mask_B_C_T_H_W.shape, cp_size) if use_spatial_split else None
            )

            if T > 1 and process_group.size() > 1:
                if use_spatial_split:
                    condition_video_input_mask_B_C_T_H_W = rearrange(
                        condition_video_input_mask_B_C_T_H_W, "b c t h w -> b c (t h w)"
                    )
                    gt_frames = rearrange(gt_frames, "b c t h w -> b c (t h w)")
                gt_frames = broadcast_split_tensor(gt_frames, seq_dim=2, process_group=process_group)
                condition_video_input_mask_B_C_T_H_W = broadcast_split_tensor(
                    condition_video_input_mask_B_C_T_H_W, seq_dim=2, process_group=process_group
                )
                if use_spatial_split:
                    condition_video_input_mask_B_C_T_H_W = rearrange(
                        condition_video_input_mask_B_C_T_H_W,
                        "b c (t h w) -> b c t h w",
                        t=after_split_shape[0],
                        h=after_split_shape[1],
                    )
                    gt_frames = rearrange(
                        gt_frames, "b c (t h w) -> b c t h w", t=after_split_shape[0], h=after_split_shape[1]
                    )
        kwargs["gt_frames"] = gt_frames
        kwargs["condition_video_input_mask_B_C_T_H_W"] = condition_video_input_mask_B_C_T_H_W
        return type(self)(**kwargs)

class Video2WorldConditionV2(Video2WorldCondition):

    def set_video_condition(
        self,
        gt_frames: torch.Tensor,
        random_min_num_conditional_frames: int,
        random_max_num_conditional_frames: int,
        num_conditional_frames: Optional[int] = None,
        conditional_frames_probs: Optional[Dict[int, float]] = None,
    ) -> "Video2WorldConditionV2":
        num_conditional_frames = 0 if not self.use_video_condition else num_conditional_frames
        return super().set_video_condition(
            gt_frames=gt_frames,
            random_min_num_conditional_frames=random_min_num_conditional_frames,
            random_max_num_conditional_frames=random_max_num_conditional_frames,
            num_conditional_frames=num_conditional_frames,
            conditional_frames_probs=conditional_frames_probs,
        )

    def edit_for_inference(
        self, is_cfg_conditional: bool = True, num_conditional_frames: int = 1
    ) -> "Video2WorldConditionV2":
        del is_cfg_conditional
        _condition = super().set_video_condition(
            gt_frames=self.gt_frames,
            random_min_num_conditional_frames=0,
            random_max_num_conditional_frames=0,
            num_conditional_frames=num_conditional_frames,
        )
        return _condition

class Video2WorldConditioner(GeneralConditioner):
    def forward(
        self,
        batch: Dict,
        override_dropout_rate: Optional[Dict[str, float]] = None,
    ) -> Video2WorldCondition:
        output = super()._forward(batch, override_dropout_rate)
        return Video2WorldCondition(**output)

class Video2WorldConditionerV2(GeneralConditioner):
    def forward(
        self,
        batch: Dict,
        override_dropout_rate: Optional[Dict[str, float]] = None,
    ) -> Video2WorldConditionV2:
        output = super()._forward(batch, override_dropout_rate)
        return Video2WorldConditionV2(**output)

_SHARED_CONFIG = dict(
    fps=L(ReMapkey)(
        input_key="fps",
        output_key="fps",
        dropout_rate=0.0,
        dtype=None,
    ),
    padding_mask=L(ReMapkey)(
        input_key="padding_mask",
        output_key="padding_mask",
        dropout_rate=0.0,
        dtype=None,
    ),
    text=L(TextAttr)(
        input_key=["t5_text_embeddings"],
        dropout_rate=0.2,
        use_empty_string=False,
    ),
    use_video_condition=L(BooleanFlag)(
        input_key="fps",
        output_key="use_video_condition",
        dropout_rate=0.2,
    ),
)

VideoPredictionConditioner: LazyDict = L(Video2WorldConditioner)(
    **_SHARED_CONFIG,
)

VideoPredictionConditionerV2: LazyDict = L(Video2WorldConditionerV2)(
    **_SHARED_CONFIG,
)

@dataclass(frozen=True)
class VideoPredictionWan2pt1Condition(Text2WorldCondition):
    frame_cond_crossattn_emb_B_L_D: Optional[torch.Tensor] = None
    y_B_C_T_H_W: Optional[torch.Tensor] = None

    def broadcast(self, process_group: torch.distributed.ProcessGroup) -> "Video2WorldCondition":
        if self.is_broadcasted:
            return self

        y_B_C_T_H_W = self.y_B_C_T_H_W
        kwargs = self.to_dict(skip_underscore=False)
        kwargs["y_B_C_T_H_W"] = None
        new_condition = Text2WorldCondition.broadcast(
            type(self)(**kwargs),
            process_group,
        )
        kwargs = new_condition.to_dict(skip_underscore=False)
        if process_group is not None:
            y_B_C_T_H_W = broadcast_split_tensor(y_B_C_T_H_W, seq_dim=2, process_group=process_group)
        kwargs["y_B_C_T_H_W"] = y_B_C_T_H_W
        return type(self)(**kwargs)

class VideoPredictionWan2pt1Conditioner(GeneralConditioner):
    def forward(
        self,
        batch: Dict,
        override_dropout_rate: Optional[Dict[str, float]] = None,
    ) -> VideoPredictionWan2pt1Condition:
        output = super()._forward(batch, override_dropout_rate)
        return VideoPredictionWan2pt1Condition(**output)

VideoConditionerFpsPaddingEmptyStringDrppConfig: LazyDict = L(VideoPredictionWan2pt1Conditioner)(
    text=L(TextAttrEmptyStringDrop)(
        input_key=["t5_text_embeddings"],
        dropout_rate=0.2,
    ),
    fps=L(ReMapkey)(
        input_key="fps",
        output_key="fps",
        dropout_rate=0.0,
        dtype=None,
    ),
    padding_mask=L(ReMapkey)(
        input_key="padding_mask",
        output_key="padding_mask",
        dropout_rate=0.0,
        dtype=None,
    ),
    wanclip=L(Wan2pt1CLIPEmb)(
        input_key=["images", "video", WAN2PT1_I2V_COND_LATENT_KEY],
        dropout_rate=0.0,
        dtype="bfloat16",
    ),
)

def register_conditioner():
    cs = ConfigStore.instance()
    cs.store(
        group="conditioner",
        package="model.config.conditioner",
        name="video_prediction_conditioner",
        node=VideoPredictionConditioner,
    )

    cs.store(
        group="conditioner",
        package="model.config.conditioner",
        name="video_prediction_conditioner_v2",
        node=VideoPredictionConditionerV2,
    )

    cs.store(
        group="conditioner",
        package="model.config.conditioner",
        name="wan2pt1_video_prediction_conditioner_empty_string_drop",
        node=VideoConditionerFpsPaddingEmptyStringDrppConfig,
    )
