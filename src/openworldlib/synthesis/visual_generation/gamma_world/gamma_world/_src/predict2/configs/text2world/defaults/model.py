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
from gamma_world._src.predict2.models.text2world_model import DiffusionModel as Text2WorldModel
from gamma_world._src.predict2.models.text2world_model import Text2WorldModelConfig
from gamma_world._src.predict2.models.text2world_wan2pt1_model import Text2WorldModelWan2pt1Config
from gamma_world._src.predict2.models.text2world_wan2pt1_model import WANDiffusionModel as Text2WorldWan2pt1Model

DDP_CONFIG = dict(
    trainer=dict(
        distributed_parallelism="ddp",
    ),
    model=L(Text2WorldModel)(
        config=Text2WorldModelConfig(),
        _recursive_=False,
    ),
)

FSDP_CONFIG = dict(
    trainer=dict(
        distributed_parallelism="fsdp",
    ),
    model=L(Text2WorldModel)(
        config=Text2WorldModelConfig(
            fsdp_shard_size=8,
        ),
        _recursive_=False,
    ),
)

FSDP_WAN2PT1_CONFIG = dict(
    trainer=dict(
        distributed_parallelism="fsdp",
    ),
    model=L(Text2WorldWan2pt1Model)(
        config=Text2WorldModelWan2pt1Config(
            fsdp_shard_size=8,
        ),
        _recursive_=False,
    ),
)

def register_model():
    cs = ConfigStore.instance()
    cs.store(group="model", package="_global_", name="ddp", node=DDP_CONFIG)
    cs.store(group="model", package="_global_", name="fsdp", node=FSDP_CONFIG)
    cs.store(group="model", package="_global_", name="fsdp_wan2pt1", node=FSDP_WAN2PT1_CONFIG)
