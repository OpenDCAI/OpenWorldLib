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
from dataclasses import dataclass
from typing import Dict, Optional

import torch
from einops import rearrange
from hydra.core.config_store import ConfigStore

from gamma_world._src.imaginaire.lazy_config import LazyCall as L
from gamma_world._src.imaginaire.lazy_config import LazyDict
from gamma_world._src.imaginaire.utils import log
from gamma_world._src.imaginaire.utils.context_parallel import broadcast, broadcast_split_tensor
from gamma_world._src.predict2.conditioner import Text2WorldCondition
from gamma_world._src.predict2.configs.video2world.defaults.conditioner import (
    _SHARED_CONFIG,
    ReMapkey,
)
from gamma_world._src.predict2_multiview.conditioner import MVTextAttr
from gamma_world._src.predict2_multiview.configs.vid2vid.defaults.conditioner import (
    MultiViewCondition,
    MultiViewConditioner,
)

@dataclass(frozen=True)
class MultiViewConditionCausal(MultiViewCondition):
    state_t: Optional[int] = None
    view_indices_B_T: Optional[torch.Tensor] = None
    ref_cam_view_idx_sample_position: Optional[torch.Tensor] = None
    control_input_hdmap_bbox: Optional[torch.Tensor] = None

    def broadcast(
        self, process_group: torch.distributed.ProcessGroup, split: bool = False
    ) -> "MultiViewConditionCausal":
        if self.is_broadcasted:
            return self

        gt_frames_B_C_T_H_W = self.gt_frames
        view_indices_B_T = self.view_indices_B_T
        condition_video_input_mask_B_C_T_H_W = self.condition_video_input_mask_B_C_T_H_W
        control_input_hdmap_bbox = self.control_input_hdmap_bbox
        assert gt_frames_B_C_T_H_W is not None
        assert view_indices_B_T is not None
        assert condition_video_input_mask_B_C_T_H_W is not None
        assert self.state_t is not None

        kwargs = self.to_dict(skip_underscore=False)
        kwargs["gt_frames"] = None
        kwargs["condition_video_input_mask_B_C_T_H_W"] = None
        kwargs["view_indices_B_T"] = None
        new_condition = Text2WorldCondition.broadcast(
            type(self)(**kwargs),
            process_group,
        )

        kwargs = new_condition.to_dict(skip_underscore=False)

        _, _, T, _, _ = gt_frames_B_C_T_H_W.shape

        assert T % self.state_t == 0, f"T must be a multiple of state_t. Got T={T} and state_t={self.state_t}."

        if process_group is not None:
            if split:
                n_views = T // self.state_t
                if T > 1 and process_group.size() > 1:
                    log.debug(f"Broadcasting {gt_frames_B_C_T_H_W.shape=} to {n_views=} views")
                    gt_frames_B_C_V_T_H_W = rearrange(gt_frames_B_C_T_H_W, "B C (V T) H W -> B C V T H W", V=n_views)
                    condition_video_input_mask_B_C_V_T_H_W = rearrange(
                        condition_video_input_mask_B_C_T_H_W, "B C (V T) H W -> B C V T H W", V=n_views
                    )
                    view_indices_B_V_T = rearrange(view_indices_B_T, "B (V T) -> B V T", V=n_views)

                    gt_frames_B_C_V_T_H_W = broadcast_split_tensor(
                        gt_frames_B_C_V_T_H_W, seq_dim=3, process_group=process_group
                    )
                    condition_video_input_mask_B_C_V_T_H_W = broadcast_split_tensor(
                        condition_video_input_mask_B_C_V_T_H_W, seq_dim=3, process_group=process_group
                    )
                    view_indices_B_V_T = broadcast_split_tensor(
                        view_indices_B_V_T, seq_dim=2, process_group=process_group
                    )

                    gt_frames_B_C_T_H_W = rearrange(gt_frames_B_C_V_T_H_W, "B C V T H W -> B C (V T) H W", V=n_views)
                    condition_video_input_mask_B_C_T_H_W = rearrange(
                        condition_video_input_mask_B_C_V_T_H_W, "B C V T H W -> B C (V T) H W", V=n_views
                    )
                    view_indices_B_T = rearrange(view_indices_B_V_T, "B V T -> B (V T)", V=n_views)
                    if control_input_hdmap_bbox is not None:
                        control_input_hdmap_bbox_B_C_V_T_H_W = rearrange(
                            control_input_hdmap_bbox, "B C (V T) H W -> B C V T H W", V=n_views
                        )
                        control_input_hdmap_bbox_B_C_V_T_H_W = broadcast_split_tensor(
                            control_input_hdmap_bbox_B_C_V_T_H_W, seq_dim=3, process_group=process_group
                        )
                        control_input_hdmap_bbox = rearrange(
                            control_input_hdmap_bbox_B_C_V_T_H_W, "B C V T H W -> B C (V T) H W", V=n_views
                        )
            else:
                gt_frames_B_C_T_H_W = broadcast(gt_frames_B_C_T_H_W, process_group)
                condition_video_input_mask_B_C_T_H_W = broadcast(condition_video_input_mask_B_C_T_H_W, process_group)
                view_indices_B_T = broadcast(view_indices_B_T, process_group)
                if control_input_hdmap_bbox is not None:
                    control_input_hdmap_bbox = broadcast(control_input_hdmap_bbox, process_group)

        kwargs["gt_frames"] = gt_frames_B_C_T_H_W
        kwargs["condition_video_input_mask_B_C_T_H_W"] = condition_video_input_mask_B_C_T_H_W
        kwargs["view_indices_B_T"] = view_indices_B_T
        kwargs["control_input_hdmap_bbox"] = control_input_hdmap_bbox
        return type(self)(**kwargs)

class MultiViewCausalConditioner(MultiViewConditioner):
    def forward(
        self, batch: Dict, override_dropout_rate: Optional[Dict[str, float]] = None
    ) -> MultiViewConditionCausal:
        output = super()._forward(batch, override_dropout_rate)
        return MultiViewConditionCausal(**output)

_SHARED_CONFIG_PER_VIEW_DROPOUT = copy.deepcopy(_SHARED_CONFIG)
_SHARED_CONFIG_PER_VIEW_DROPOUT["text"] = L(MVTextAttr)(
    input_key=["t5_text_embeddings"],
    dropout_rate=0.2,
    use_empty_string=False,
)

MultiViewCausalConditionerPerViewDropoutConfig: LazyDict = L(MultiViewCausalConditioner)(
    **_SHARED_CONFIG_PER_VIEW_DROPOUT,
    view_indices_B_T=L(ReMapkey)(
        input_key="latent_view_indices_B_T",

        output_key="view_indices_B_T",
        dropout_rate=0.0,
        dtype=None,
    ),
    ref_cam_view_idx_sample_position=L(ReMapkey)(
        input_key="ref_cam_view_idx_sample_position",
        output_key="ref_cam_view_idx_sample_position",
        dropout_rate=0.0,
        dtype=None,
    ),
)

def register_conditioner():
    cs = ConfigStore.instance()
    cs.store(
        group="conditioner",
        package="model.config.conditioner",
        name="video_prediction_multiview_causal_conditioner_per_view_dropout",
        node=MultiViewCausalConditionerPerViewDropoutConfig,
    )
