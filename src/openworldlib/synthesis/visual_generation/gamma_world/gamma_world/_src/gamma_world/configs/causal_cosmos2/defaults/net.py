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
from hydra.core.config_store import ConfigStore

from gamma_world._src.imaginaire.lazy_config import LazyCall as L
from gamma_world._src.imaginaire.lazy_config import LazyDict
from gamma_world._src.predict2.networks.minimal_v4_dit import SACConfig
from gamma_world._src.gamma_world.networks.causal_cosmos import CosmosCausalDiT
from gamma_world._src.gamma_world.networks.minimal_v1_lvg_dit import MinimalV1LVGDiT

COSMOS_V2_2B_NET_MININET: LazyDict = L(CosmosCausalDiT)(
    max_img_h=240,
    max_img_w=240,
    max_frames=128,
    in_channels=16,
    out_channels=16,
    patch_spatial=2,
    patch_temporal=1,
    model_channels=2048,
    num_blocks=28,
    num_layers=28,
    num_heads=16,
    concat_padding_mask=True,
    pos_emb_cls="rope3d",
    pos_emb_learnable=True,
    pos_emb_interpolation="crop",
    use_adaln_lora=True,
    adaln_lora_dim=256,

    extra_per_block_abs_pos_emb=False,
    rope_h_extrapolation_ratio=1.0,
    rope_w_extrapolation_ratio=1.0,
    rope_t_extrapolation_ratio=1.0,

    sac_config=SACConfig(mode="block_wise"),
    use_crossattn_projection=True,
    crossattn_proj_in_channels=100352,
    crossattn_emb_channels=1024,
    use_wan_fp32_strategy=True,
)

COSMOS_V1_2B_NET_MININET: LazyDict = L(MinimalV1LVGDiT)(
    max_img_h=240,
    max_img_w=240,
    max_frames=128,
    in_channels=16,
    out_channels=16,
    patch_spatial=2,
    patch_temporal=1,
    model_channels=2048,
    num_blocks=28,
    num_layers=28,
    num_heads=16,
    concat_padding_mask=True,
    pos_emb_cls="rope3d",
    pos_emb_learnable=True,
    pos_emb_interpolation="crop",
    use_adaln_lora=True,
    adaln_lora_dim=256,

    extra_per_block_abs_pos_emb=False,
    rope_h_extrapolation_ratio=1.0,
    rope_w_extrapolation_ratio=1.0,
    rope_t_extrapolation_ratio=1.0,

    sac_config=SACConfig(mode="block_wise"),
    use_crossattn_projection=True,
    crossattn_proj_in_channels=100352,
    crossattn_emb_channels=1024,
    use_wan_fp32_strategy=True,
)

def register_net():
    cs = ConfigStore.instance()
    cs.store(group="net", package="model.config.net", name="cosmos_v2_2b_causal", node=COSMOS_V2_2B_NET_MININET)
    cs.store(group="net", package="model.config.net", name="cosmos_v1_2B", node=COSMOS_V1_2B_NET_MININET)
