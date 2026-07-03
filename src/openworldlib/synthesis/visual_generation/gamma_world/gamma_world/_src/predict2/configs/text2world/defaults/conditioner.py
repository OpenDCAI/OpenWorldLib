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
from gamma_world._src.predict2.conditioner import ReMapkey, TextAttr, TextAttrEmptyStringDrop, VideoConditioner

VideoConditionerFpsPaddingConfig: LazyDict = L(VideoConditioner)(
    text=L(TextAttr)(
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
)

VideoConditionerFpsPaddingEmptyStringDrppConfig: LazyDict = L(VideoConditioner)(
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
)

def register_conditioner():
    cs = ConfigStore.instance()
    cs.store(
        group="conditioner",
        package="model.config.conditioner",
        name="add_fps_padding_mask",
        node=VideoConditionerFpsPaddingConfig,
    )
    cs.store(
        group="conditioner",
        package="model.config.conditioner",
        name="add_fps_padding_mask_empty_string_drop",
        node=VideoConditionerFpsPaddingEmptyStringDrppConfig,
    )
