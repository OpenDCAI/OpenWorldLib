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
from gamma_world._src.gamma_world.datasets.solaris_data import (
    SolarisAugmentationConfig,
    get_solaris_video_loader,
)

def register_solaris_dataloader() -> None:
    cs = ConfigStore.instance()

    cs.store(
        group="data_train",
        package="dataloader_train",
        name="video_solaris_action",
        node=L(get_solaris_video_loader)(
            is_train=True,
            augmentation_config=L(SolarisAugmentationConfig)(
                resolution_hw=(480, 832),
                num_video_frames=93,
            ),
        ),
    )
    cs.store(
        group="data_val",
        package="dataloader_val",
        name="video_solaris_action",
        node=L(get_solaris_video_loader)(
            is_train=False,
            augmentation_config=L(SolarisAugmentationConfig)(
                resolution_hw=(480, 832),
                num_video_frames=93,
            ),
            batch_size=1,
            num_workers=2,
        ),
    )
