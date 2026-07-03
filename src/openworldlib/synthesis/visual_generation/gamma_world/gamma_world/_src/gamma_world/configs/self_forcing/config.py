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

from typing import Any, List

import attrs

from gamma_world._src.imaginaire import config
from gamma_world._src.imaginaire.utils.config_helper import import_all_modules_from_package
from gamma_world._src.predict2.configs.common.defaults.checkpoint import register_checkpoint
from gamma_world._src.predict2.configs.common.defaults.ckpt_type import register_ckpt_type
from gamma_world._src.predict2.configs.common.defaults.optimizer import register_optimizer
from gamma_world._src.predict2.configs.common.defaults.scheduler import register_scheduler
from gamma_world._src.predict2.configs.common.defaults.dataloader import register_training_and_val_data
from gamma_world._src.predict2.configs.text2world.defaults.callbacks import register_callbacks
from gamma_world._src.gamma_world.configs.causal_cosmos2.defaults.conditioner import register_conditioner
from gamma_world._src.gamma_world.configs.causal_cosmos2.defaults.dataloader import register_solaris_dataloader
from gamma_world._src.gamma_world.configs.causal_cosmos2.defaults.tokenizer import (
    register_tokenizer as register_wan2pt1_tokenizer,
)
from gamma_world._src.gamma_world.configs.defaults.callbacks import (
    register_callbacks as register_callbacks_causal,
)
from gamma_world._src.gamma_world.configs.self_forcing.defaults.model import register_model
from gamma_world._src.gamma_world.configs.self_forcing.defaults.net import register_net
from gamma_world._src.gamma_world.trainer.trainer_distillation import (
    ImaginaireTrainer as DistillationTrainer,
)

@attrs.define(slots=False)
class Config(config.Config):

    defaults: List[Any] = attrs.field(
        factory=lambda: [
            "_self_",
            {"data_train": "video_solaris_action"},
            {"data_val": "mock"},
            {"optimizer": "adamw"},
            {"scheduler": "lambdalinear"},
            {"model": "fsdp_mv"},
            {"callbacks": "basic"},
            {"net": "causal_cosmosv2_2b"},
            {"net_real_score": "cosmos_v1_2B_mc"},
            {"net_fake_score": "cosmos_v1_2B_mc"},
            {"tokenizer": "wan2pt1_tokenizer"},
            {"conditioner": "video_prediction_multiview_causal_conditioner_per_view_dropout"},
            {"checkpoint": "s3"},
            {"ckpt_type": "dcp"},

            {"experiment": None},
        ]
    )

def make_config() -> Config:
    c = Config(
        model=None,
        optimizer=None,
        scheduler=None,
        dataloader_train=None,
        dataloader_val=None,
    )

    c.job.project = "cosmos_diffusion_v2"
    c.job.group = "debug"
    c.job.name = "delete_${now:%Y-%m-%d}_${now:%H-%M-%S}"

    c.trainer.type = DistillationTrainer
    c.trainer.straggler_detection.enabled = False
    c.trainer.max_iter = 1_000
    c.trainer.logging_iter = 10
    c.trainer.validation_iter = 100
    c.trainer.run_validation = False
    c.trainer.callbacks = None

    register_optimizer()
    register_scheduler()
    register_training_and_val_data()
    register_solaris_dataloader()
    register_conditioner()

    register_net()
    register_checkpoint()
    register_wan2pt1_tokenizer()
    register_callbacks_causal()
    register_callbacks()

    register_ckpt_type()
    register_model()

    import_all_modules_from_package("gamma_world._src.gamma_world.configs.self_forcing.experiment", reload=True)
    return c
