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

from dataclasses import dataclass, field
from typing import Optional

CONFIG_CAUSAL = "gamma_world/_src/gamma_world/configs/causal_cosmos2/config.py"
CONFIG_SELF_FORCING = "gamma_world/_src/gamma_world/configs/self_forcing/config.py"

@dataclass(frozen=True)
class ModelSpec:
    experiment: str
    config_file: str
    sampler: str
    default_checkpoint: str
    default_num_steps: Optional[int] = None
    denoising_step_list: Optional[tuple] = None
    config_overrides: dict = field(default_factory=dict)

_SHARE_ACTION_ENCODER = "model.config.net.multi_agent_rope_share_action_encoder"
_DENOISING_STEP_LIST = "model.config.denoising_step_list"

MODEL_SPECS = {
    "bidirectional": ModelSpec(
        experiment="bidirectional",
        config_file=CONFIG_CAUSAL,
        sampler="bidirectional",
        default_checkpoint="hf://chijw/Gamma-World/bidirectional/model.safetensors",
        default_num_steps=35,
    ),
    "causal": ModelSpec(
        experiment="causal",
        config_file=CONFIG_CAUSAL,
        sampler="causal",
        default_checkpoint="hf://chijw/Gamma-World/causal/model.safetensors",
        default_num_steps=35,
    ),
    "causal_few_step": ModelSpec(
        experiment="causal_few_step",
        config_file=CONFIG_SELF_FORCING,
        sampler="causal_few_step",
        default_checkpoint="hf://chijw/Gamma-World/causal-few-step/model.safetensors",
        default_num_steps=4,
        denoising_step_list=(1000, 750, 500, 250),
        config_overrides={
            _SHARE_ACTION_ENCODER: True,
            _DENOISING_STEP_LIST: (1000, 750, 500, 250),
            "model.config.net_ckpt": "",
            "model.config.net_real_score": None,
            "model.config.net_fake_score": None,
        },
    ),
}
